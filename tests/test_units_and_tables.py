"""Dimensional-consistency and published-figure regression tests.

WHY THIS FILE EXISTS
--------------------
Three separate errors in this project were the same error: a scale factor applied
in the wrong power.

1. The risk-free rate was never subtracted, so "Sharpe" was a return-to-volatility
   ratio inflated by ~0.35.
2. A per-period variance was compared against an annualised standard error - a
   factor of sqrt(12).
3. A STANDARD ERROR ratio (1.248) was applied where a VARIANCE ratio (1.558)
   belonged, understating required sample size by 25%.

None was a reasoning failure. Every one was catchable by recomputing a published
number from its own primitives, and none was caught by reading.

DESIGN RULES, learned from the failures of the previous version of this file:

* **No test may silently skip.** An earlier version guarded the paper tests with
  ``skipif(not PAPER.exists())``, which would have left them dormant across
  exactly the revisions that needed them. The published tables now live in
  ``tests/fixtures/published_tables.md``; a missing fixture, README or generated
  data file is a hard failure here.
* **No published constant is hardcoded twice.** The inflation factor is read from
  the generated CSV, never from a literal, so the paper and the tests cannot drift
  apart while both appearing self-consistent.
* **Prose is checked, not only tables.** Stale figures once survived in the
  manuscript's prose while every table was correct, so the headline figures,
  test count and file names quoted in README.md are checked too.
* **Guards reconstruct rather than blacklist.** Asserting that one obsolete number
  is absent catches the last error; recomputing the quantity catches the next one.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy import stats as sps

from macroregime.config import REPORT_DIR, ROOT
from macroregime.evaluation.power import (
    MONTHS,
    achieved_power,
    memmel_se_annual,
    memmel_variance_factor,
    minimum_detectable_effect,
    simulate_se,
    years_required,
)

BENCHMARK_SHARPE = 0.682   # Table 2 (Section 5.1); also asserted against the CSV below


# ---------------------------------------------------------------------------
# Locating inputs - deliberately intolerant
# ---------------------------------------------------------------------------

FIXTURE = Path(__file__).parent / "fixtures" / "published_tables.md"


def _published() -> Path:
    """The tables exactly as published in the working paper. Missing FAILS."""
    assert FIXTURE.exists(), f"{FIXTURE} is missing; the table tests cannot run"
    return FIXTURE


def _readme() -> Path:
    """README.md, whose quoted figures are checked like the tables. Missing FAILS."""
    path = ROOT / "README.md"
    assert path.exists(), "README.md is missing"
    return path


def _csv(name: str) -> pd.DataFrame:
    """Load generated data. Missing data FAILS rather than skips."""
    path = REPORT_DIR / f"{name}.csv"
    assert path.exists(), (
        f"{path} is missing. Run `python -m macroregime.run` and "
        f"`python -m macroregime.evaluation.power` before the test suite."
    )
    return pd.read_csv(path)


def _inflation() -> float:
    """THE single source of truth for the standard-error inflation factor.

    Derived from the generated data, never hardcoded. The paper quotes 1.25 in
    prose and computes its tables from this unrounded value; hardcoding either
    printed form here is what let the two drift apart previously.
    """
    return float(_csv("power_reconciliation")["ratio"].median())


def _markdown_tables(text: str) -> list[list[list[str]]]:
    tables, current = [], []
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("|") and stripped.endswith("|"):
            cells = [c.strip() for c in stripped.strip("|").split("|")]
            if not all(set(c) <= set("-: ") for c in cells):
                current.append(cells)
        elif current:
            tables.append(current)
            current = []
    if current:
        tables.append(current)
    return tables


def _table(text: str, number: int) -> list[list[str]]:
    """Return the first markdown table after the caption ``Table <number>.``.

    Tables are located by their caption rather than by section number. Section
    numbers changed when the manuscript was restructured (the years table moved
    from Section 7.3 to Section 3.5, the detectability table from 7.4 to 7.1, the
    primary results from 4.1 to 5.1), and section-keyed slicing would have pointed
    each test at the wrong table. A missing caption FAILS rather than skips.
    """
    m = re.search(rf"Table {number}\.", text)
    assert m, f"manuscript has no caption 'Table {number}.'"
    rest = text[m.end():]
    nxt = re.search(r"Table \d+\.", rest)
    tables = _markdown_tables(rest[: nxt.start()] if nxt else rest)
    assert tables, f"no markdown table follows the caption 'Table {number}.'"
    return tables[0]


def _num(cell: str) -> float | None:
    cleaned = cell.replace("−", "-").replace("**", "").replace("%", "").strip()
    try:
        return float(cleaned)
    except ValueError:
        return None


# ---------------------------------------------------------------------------
# Unit convention
# ---------------------------------------------------------------------------

def test_years_scales_with_variance_not_standard_error():
    """THE REGRESSION GUARD for error #3.

    Required sample scales with variance, so doubling the SE inflation must
    QUADRUPLE the requirement. Applying the SE ratio linearly is the bug.
    """
    base = years_required(0.10, BENCHMARK_SHARPE, 0.89, se_inflation=1.0)
    doubled = years_required(0.10, BENCHMARK_SHARPE, 0.89, se_inflation=2.0)
    assert doubled == pytest.approx(4 * base, rel=1e-9)


def test_measured_inflation_applied_as_variance():
    """At the measured ratio, T carries its square - value read from data."""
    r = _inflation()
    plain = years_required(0.10, BENCHMARK_SHARPE, 0.89, se_inflation=1.0)
    inflated = years_required(0.10, BENCHMARK_SHARPE, 0.89, se_inflation=r)
    assert inflated / plain == pytest.approx(r**2, rel=1e-9)


def test_years_scales_with_inverse_square_of_effect():
    big = years_required(0.20, BENCHMARK_SHARPE, 0.89)
    small = years_required(0.10, BENCHMARK_SHARPE, 0.89)
    assert small / big == pytest.approx(4.0, rel=0.02)


def test_annualisation_applied_exactly_once():
    """THE REGRESSION GUARD for error #2 (the sqrt(12) conflation)."""
    s1_ann, s2_ann, rho, T = 0.655, 0.682, 0.952, 535
    got = memmel_se_annual(s1_ann, s2_ann, rho, T)

    s1, s2 = s1_ann / np.sqrt(MONTHS), s2_ann / np.sqrt(MONTHS)
    per_period = np.sqrt(memmel_variance_factor(s1, s2, rho) / T)

    assert got == pytest.approx(per_period * np.sqrt(MONTHS), rel=1e-12)
    assert got / per_period == pytest.approx(np.sqrt(MONTHS), rel=1e-12)


def test_closed_form_matches_simulation():
    for s1, s2, rho, T in [(0.66, 0.68, 0.95, 535), (0.55, 0.68, 0.80, 535)]:
        assert simulate_se(s1, s2, rho, T, n_sims=3000, seed=11) == pytest.approx(
            memmel_se_annual(s1, s2, rho, T), rel=0.05
        )


def test_mde_thresholds():
    se = 0.0842
    assert minimum_detectable_effect(se, 0.50) == pytest.approx(1.96 * se, rel=1e-3)
    assert minimum_detectable_effect(se, 0.80) == pytest.approx(2.8016 * se, rel=1e-3)
    assert minimum_detectable_effect(se, 0.80) / minimum_detectable_effect(
        se, 0.50
    ) == pytest.approx(1.43, rel=0.01)


def test_power_at_null_effect_equals_alpha():
    """Power against a zero effect is the size of the test, not evidence."""
    assert achieved_power(0.0, 0.12, alpha=0.05) == pytest.approx(0.05, abs=1e-9)


def test_rho_enters_through_two_one_minus_rho():
    low = years_required(0.10, BENCHMARK_SHARPE, 0.85)
    high = years_required(0.10, BENCHMARK_SHARPE, 0.95)
    assert low / high == pytest.approx(3.0, rel=0.15)


# ---------------------------------------------------------------------------
# Generated data internal consistency
# ---------------------------------------------------------------------------

def test_reconciliation_csv_is_internally_consistent():
    rec = _csv("power_reconciliation")
    for _, r in rec.iterrows():
        assert memmel_se_annual(
            r["S1"], r["S2"], r["rho"], int(r["n_months"])
        ) == pytest.approx(r["memmel_se"], rel=1e-6)
        assert r["ratio"] == pytest.approx(r["boot_se"] / r["memmel_se"], rel=1e-9)
        assert r["mde_80"] == pytest.approx(
            minimum_detectable_effect(r["boot_se"], 0.80), rel=1e-9
        )
        assert r["mde_50"] == pytest.approx(
            minimum_detectable_effect(r["boot_se"], 0.50), rel=1e-9
        )
        assert r["achieved_power"] == pytest.approx(
            achieved_power(r["observed_effect"], r["boot_se"]), rel=1e-9
        )


def test_benchmark_sharpe_constant_matches_generated_data():
    """The constant used throughout the tests must be the one the pipeline produced."""
    rec = _csv("power_reconciliation")
    assert float(rec["S2"].iloc[0]) == pytest.approx(BENCHMARK_SHARPE, abs=5e-4)


# ---------------------------------------------------------------------------
# Published-table regression
# ---------------------------------------------------------------------------

def test_table_1_matches_closed_form():
    """Reconstruct all thirty cells of Table 1 (Section 3.5) from primitives.

    Column effect sizes are parsed from the header rather than assumed, so an
    added or reordered column cannot pass silently, and the cell count is asserted
    exactly rather than as a lower bound.
    """
    tbl = _table(_published().read_text(encoding="utf-8"), 1)

    header = tbl[0]
    deltas: list[float] = []
    for cell in header[1:]:
        m = re.search(r"([0-9]*\.[0-9]+)", cell.replace("**", ""))
        assert m, f"could not parse an effect size from header cell {cell!r}"
        deltas.append(float(m.group(1)))
    assert len(deltas) == 5, f"expected 5 effect-size columns, parsed {deltas}"

    infl = _inflation()
    checked = 0
    for row in tbl[1:]:
        rho = _num(row[0])
        assert rho is not None, f"unparseable rho in row {row}"
        assert len(row) == len(deltas) + 1, f"ragged row: {row}"
        for j, d in enumerate(deltas, start=1):
            printed = _num(row[j])
            assert printed is not None, f"unparseable cell {row[j]!r}"
            expected = years_required(d, BENCHMARK_SHARPE, rho, se_inflation=infl)
            assert printed == pytest.approx(expected, abs=1.0), (
                f"Table 1 (rho={rho}, delta={d}): paper {printed}, "
                f"closed form {expected:.1f} at inflation {infl:.6f}"
            )
            checked += 1

    assert checked == 30, f"Table 1 must publish exactly 30 cells, found {checked}"


DETECTABILITY_EFFECT = 0.10   # the pre-specified effect of Table 5's power column


def _nearest(frame: pd.DataFrame, rho: float, where: str) -> pd.Series:
    """The CSV row whose rho is nearest a published rho.

    Match on nearest rho rather than exact equality: a value such as 0.8885
    rounds to 0.888 under banker's rounding and 0.889 conventionally, and both
    are legitimate displays of the same number. The tolerance is one unit in the
    last published place, which is tight enough that two different comparisons
    can never collide.
    """
    gaps = (frame["rho"] - rho).abs()
    assert gaps.min() <= 1e-3, (
        f"{where} row rho={rho} has no match in the CSV "
        f"(nearest differs by {gaps.min():.5f})"
    )
    return frame.loc[gaps.idxmin()]


def test_detectability_tables_match_generated_data():
    """Reconstruct Tables 5 and 6 - the paper's headline artefacts - from the CSVs.

    Table 5 (Section 7.1): Comparison | rho | MDE (50%) | MDE (80%) | Observed Δ |
    Power at Δ = 0.10. Power is recomputed at the PRE-SPECIFIED effect from each
    comparison's bootstrap SE; power at the observed effect was dropped from the
    manuscript because it is a transformation of the p-value (Hoenig and Heisey).
    Rows are matched on rho rather than on label, so renaming a strategy in prose
    cannot silently detach a row from its data.

    Table 6 (Section 7.2): Cell | rho vs honest | MDE (80%) | Observed Δ, matched on
    the cell label against ``ablation_detectability.csv`` (fixed weights).
    """
    text = _published().read_text(encoding="utf-8")

    # ---- Table 5
    tbl = _table(text, 5)
    rec = _csv("power_reconciliation")
    checked = 0
    for row in tbl[1:]:
        rho = _num(row[1])
        if rho is None:
            continue
        r = _nearest(rec, rho, "Table 5")
        assert _num(row[2]) == pytest.approx(r["mde_50"], abs=6e-4), f"MDE50 rho={rho}"
        assert _num(row[3]) == pytest.approx(r["mde_80"], abs=6e-4), f"MDE80 rho={rho}"
        assert abs(_num(row[4])) == pytest.approx(
            r["observed_effect"], abs=6e-4
        ), f"observed effect rho={rho}"
        power = achieved_power(DETECTABILITY_EFFECT, r["boot_se"])
        assert _num(row[5]) / 100.0 == pytest.approx(power, abs=6e-3), (
            f"power at Δ = {DETECTABILITY_EFFECT} rho={rho}: paper {row[5]}, "
            f"recomputed {100 * power:.1f}%"
        )
        checked += 1
    assert checked == len(rec), (
        f"Table 5 must publish all {len(rec)} comparisons, found {checked}"
    )

    # ---- Table 6
    tbl = _table(text, 6)
    abl = _csv("ablation_detectability")
    abl = abl[abl["weights"] == "fixed"].set_index("cell")
    checked = 0
    for row in tbl[1:]:
        cell = row[0].replace("**", "").strip()
        if cell not in abl.index:
            continue
        a = abl.loc[cell]
        assert a["mde_80"] == pytest.approx(
            minimum_detectable_effect(a["boot_se"], 0.80), rel=1e-9
        ), f"CSV mde_80 inconsistent for cell {cell}"
        assert _num(row[1]) == pytest.approx(a["rho"], abs=6e-4), f"rho cell {cell}"
        assert _num(row[2]) == pytest.approx(a["mde_80"], abs=6e-4), f"MDE80 cell {cell}"
        assert _num(row[3]) == pytest.approx(a["delta"], abs=6e-4), f"Δ cell {cell}"
        checked += 1
    assert checked == len(abl), (
        f"Table 6 must publish all {len(abl)} fixed-weight cells, found {checked}"
    )


def test_paper_primary_table_reports_excess_return_sharpe():
    """THE GENERAL GUARD for error #1 - reconstruction, not a blacklist.

    Each Sharpe in Table 2 (Section 5.1) must equal the ``sharpe`` column of the generated
    summary, and must NOT equal that row's ``return_vol_ratio``. A pipeline run
    that dropped the risk-free rate would make those two columns coincide, and this
    fails on that condition rather than on the specific stale values it produced.
    """
    summary = _csv("summary_long")
    assert summary["rf_subtracted"].all(), "summary was generated without a risk-free rate"

    for _, r in summary.iterrows():
        assert r["sharpe"] != pytest.approx(r["return_vol_ratio"], abs=1e-3), (
            f"{r['strategy']}: Sharpe equals the return/volatility ratio, which means "
            "the risk-free rate was not subtracted"
        )

    primary = _table(_published().read_text(encoding="utf-8"), 2)

    published = {
        round(v, 3)
        for row in primary[1:]
        for c in row
        if (v := _num(c)) is not None
    }
    for _, r in summary.iterrows():
        assert round(r["sharpe"], 3) in published, (
            f"{r['strategy']}: Sharpe {r['sharpe']:.3f} absent from Table 2"
        )
        assert round(r["return_vol_ratio"], 3) not in published, (
            f"{r['strategy']}: the return/volatility ratio "
            f"{r['return_vol_ratio']:.3f} appears in the Sharpe table"
        )


# ---------------------------------------------------------------------------
# README registry - stale figures once survived in prose while tables were correct
# ---------------------------------------------------------------------------

def _published_rho_grid() -> list[float]:
    """The rho values Table 1 (Section 3.5) actually publishes.

    The registry must evaluate at the grid the manuscript prints, not at the raw
    maximum rho in the data. An earlier version used ``rec["rho"].max()`` = 0.9524
    and derived 122 years, while the manuscript published 128 computed at the grid
    value 0.95 - the Revision 3 error (two quantities evaluated at different rho)
    reappearing inside the test written to prevent it.
    """
    tbl = _table(_published().read_text(encoding="utf-8"), 1)
    grid = [_num(row[0]) for row in tbl[1:]]
    grid = [g for g in grid if g is not None]
    assert grid, "could not parse the rho grid from Table 1"
    return grid


def _appears_as_number(value: str, text: str) -> bool:
    """Whether ``value`` appears in ``text`` as a standalone number.

    Bare substring matching is not sufficient and previously produced a false
    pass: the registry asserted "122" and found it inside "0.1222" in an unrelated
    table. Digits may not abut the match on either side, nor may a decimal point
    precede it.
    """
    pattern = r"(?<![\d.])" + re.escape(value) + r"(?![\d])"
    return re.search(pattern, text) is not None


def _power_at_effect(boot_se: float) -> float:
    return achieved_power(DETECTABILITY_EFFECT, boot_se)


def _published_constants() -> dict[str, str]:
    """Figures that must appear in the prose, derived from generated data.

    Every entry is evaluated at the same parameters the manuscript publishes.
    """
    rec = _csv("power_reconciliation")
    infl = _inflation()
    s2 = float(rec["S2"].iloc[0])
    grid = _published_rho_grid()

    years_at_grid = [
        years_required(0.10, s2, rho, se_inflation=infl) for rho in grid
    ]

    # Power is registered at the pre-specified effect of Table 5, not at the observed
    # effect. The manuscript no longer reports observed-effect power (5%-34%), and
    # leaving those entries here would pass by accident: "5%" and "34%" occur in
    # unrelated sentences ("a 5% test", "15-34% larger").
    power_at_effect = [_power_at_effect(se) for se in rec["boot_se"]]

    return {
        "MDE range low": f"{rec['mde_80'].min():.2f}",
        "MDE range high": f"{rec['mde_80'].max():.2f}",
        "power at Δ=0.10, low": f"{round(min(power_at_effect) * 100)}%",
        "power at Δ=0.10, high": f"{round(max(power_at_effect) * 100)}%",
        "years at least favourable published rho": str(round(max(years_at_grid))),
        "years at most favourable published rho": str(round(min(years_at_grid))),
        "inflation (3dp)": f"{infl:.3f}",
        "variance inflation": f"{infl**2:.3f}",
    }


def test_published_constants_appear_in_readme():
    """Every headline figure the README quotes must be derived, not remembered.

    Matching is number-aware: an earlier substring test passed because a required
    figure happened to occur inside a longer decimal elsewhere in the text.
    """
    text = _readme().read_text(encoding="utf-8").replace("\u2212", "-")
    missing = {
        k: v for k, v in _published_constants().items()
        if not _appears_as_number(v, text)
    }
    assert not missing, f"headline figures absent from README.md: {missing}"


def test_registry_uses_the_published_rho_grid():
    """Guard the guard: the registry must not drift back to the raw maximum rho."""
    rec = _csv("power_reconciliation")
    grid = _published_rho_grid()
    raw_max = float(rec["rho"].max())
    s2, infl = float(rec["S2"].iloc[0]), _inflation()

    at_grid = years_required(0.10, s2, max(grid), se_inflation=infl)
    at_raw = years_required(0.10, s2, raw_max, se_inflation=infl)

    # the two genuinely differ, which is why the distinction matters
    assert abs(at_grid - at_raw) > 1.0, (
        "grid and raw rho now coincide; this test no longer discriminates"
    )
    published = _published_constants()["years at most favourable published rho"]
    assert published == str(round(at_grid)), (
        f"registry published {published} but the grid value is {at_grid:.1f}"
    )


def test_readme_test_count_is_current():
    """The README's stated test count must match the suite it describes.

    The count has been quoted as 29, 40, 44 and 50 tests, each correct when
    written and stale within a revision. A claim about the repository that the
    repository can check should be checked.

    The count is recorded by ``pytest_collection_finish`` in ``conftest.py`` rather
    than obtained by shelling out to a nested pytest run, which made the suite
    depend on how it was invoked.
    """
    text = _readme().read_text(encoding="utf-8")
    m = re.search(r"#\s*(\d+)\s+tests", text)
    assert m, "README.md must state how many tests the suite contains"
    claimed = int(m.group(1))

    count_file = Path(__file__).parent / ".collected_count"
    assert count_file.exists(), (
        "conftest.pytest_collection_finish did not record a count; run the whole "
        "suite rather than a single test"
    )
    actual = int(count_file.read_text(encoding="utf-8").strip())

    assert claimed == actual, (
        f"README.md claims {claimed} tests; the suite collects {actual}"
    )


def test_every_file_named_in_readme_exists():
    """Every file the README names must actually exist.

    The manuscript once listed power_gaussian.csv and power_empirical.csv as
    reproducible artefacts after the module that wrote them had been rewritten
    and no longer produced either. Documentation that cites files a reader cannot
    find is the failure it exists to rule out.
    """
    tick = chr(96)
    text = _readme().read_text(encoding="utf-8")
    pattern = tick + r"([A-Za-z0-9_./-]+\.(?:csv|json|html|py|pdf))" + tick
    cited = sorted(set(re.findall(pattern, text)))
    assert cited, "expected README.md to name at least one file"

    def exists(name: str) -> bool:
        candidates = [ROOT / name, REPORT_DIR / name, ROOT / "src" / "macroregime" / name]
        return any(c.exists() for c in candidates)

    missing = [c for c in cited if not exists(c)]
    assert not missing, f"README.md names files that do not exist: {missing}"

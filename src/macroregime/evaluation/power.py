"""Statistical power analysis for backtest comparisons.

    python -m macroregime.evaluation.power

THE QUESTION THIS ANSWERS
-------------------------
Given a sample of N months, how large must a true Sharpe improvement be before a
backtest can reliably detect it?

The answer reframes every null result in this project. A study with 20% power that
reports "no significant difference" has not shown the absence of an effect; it has
shown it could not have seen one.

PARAMETERISATION DISCIPLINE
---------------------------
An earlier version of this module carried a hardcoded ``correlation=0.85`` while
the closed form was evaluated at each comparison's measured rho and on a different
sample. The residual between the two was then reported as "tail inflation" - i.e. a
parameter mismatch was mislabelled as a distributional finding.

Nothing here defaults rho. It must be supplied, and every routine reports the
(rho, S1, S2, T) it used, so that two numbers can never be compared unless they
were computed under the same parameterisation. This is the module's own
Section 7.1 recommendation applied to itself.

TWO ROUTES, NOT THREE
---------------------
* **Analytic** - Jobson-Korkie (1981) with the Memmel (2003) correction.
* **Empirical** - stationary block bootstrap of realised strategy returns.

Monte Carlo simulation of Gaussian returns reproduces the analytic result to
within 1% and is therefore a check on the algebra, not independent evidence. It is
retained for that purpose and labelled as such.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from scipy import stats as sps

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# UNIT CONVENTION - read before touching anything below.
#
# Three errors in this project have been the same error: a scale factor applied
# in the wrong power. Omitting the risk-free rate; comparing a per-period
# variance to an annualised standard error (a factor of sqrt(12)); and applying a
# STANDARD ERROR ratio where a VARIANCE ratio belongs (a factor of the ratio
# itself). None was a reasoning failure; all three were unit failures.
#
# The convention, applied without exception:
#
#   1. All internal algebra is in PER-PERIOD VARIANCE units.
#   2. Annualisation happens ONCE, at the boundary, via sqrt(MONTHS).
#   3. Functions taking an inflation factor accept it as a STANDARD ERROR ratio
#      (the observable quantity: boot_se / analytic_se) and square it internally
#      wherever a variance is required.
#   4. Required-sample-size scales with VARIANCE, so it carries the square.
#
# tests/test_units_and_tables.py enforces (3) and (4) directly.
# ---------------------------------------------------------------------------

MONTHS = 12
DEFAULT_DELTAS = (0.05, 0.10, 0.20, 0.30, 0.50)
DEFAULT_RHOS = (0.76, 0.80, 0.85, 0.89, 0.92, 0.95)


# ---------------------------------------------------------------------------
# Closed form
# ---------------------------------------------------------------------------

def memmel_variance_factor(s1: float, s2: float, rho: float) -> float:
    """Variance factor (to be divided by T) for the difference of two Sharpe ratios.

    Inputs are PER-PERIOD Sharpe ratios. Jobson and Korkie (1981) derived the
    asymptotic distribution; Memmel (2003) corrected an error in their expression.
    """
    return 2 - 2 * rho + 0.5 * (s1**2 + s2**2 - 2 * s1 * s2 * rho**2)


def memmel_se_annual(s1_ann: float, s2_ann: float, rho: float, n_months: int) -> float:
    """ANNUALISED standard error of the Sharpe difference.

    The variance factor is in per-period units; the sqrt(MONTHS) at the end is what
    converts it. Omitting that factor understates the standard error by 3.46 and
    was a real error in an earlier draft of the paper.
    """
    s1, s2 = s1_ann / np.sqrt(MONTHS), s2_ann / np.sqrt(MONTHS)
    return float(np.sqrt(memmel_variance_factor(s1, s2, rho) / n_months) * np.sqrt(MONTHS))


def years_required(
    delta_ann: float,
    benchmark_sharpe_ann: float,
    rho: float,
    power: float = 0.80,
    alpha: float = 0.05,
    se_inflation: float = 1.0,
) -> float:
    """Years of monthly data needed to detect ``delta_ann`` at ``power``.

    ``se_inflation`` is a ratio of STANDARD ERRORS - the directly observable
    bootstrap-to-analytic ratio from :func:`reconcile`, around 1.25 here. It is
    **squared internally**, because required sample size scales with variance, not
    with standard error.

    Passing the standard-error ratio unsquared understates the requirement by the
    ratio itself (25% here), which is exactly the error this signature exists to
    prevent. Pass 1.0 for the pure Gaussian result.
    """
    z = sps.norm.ppf(1 - alpha / 2) + sps.norm.ppf(power)
    d = delta_ann / np.sqrt(MONTHS)
    s2 = benchmark_sharpe_ann / np.sqrt(MONTHS)
    s1 = s2 + d
    variance_inflation = se_inflation ** 2   # SE ratio -> variance ratio
    t_months = memmel_variance_factor(s1, s2, rho) * variance_inflation * (z / d) ** 2
    return float(t_months / MONTHS)


def minimum_detectable_effect(se: float, power: float = 0.80, alpha: float = 0.05) -> float:
    """Smallest detectable Sharpe difference given a standard error.

    At power = 0.50 this reduces to 1.96*SE, the half-width of a 95% interval - the
    effect that would be *just* significant. The conventional 80% threshold needs
    2.80*SE. Reporting a CI half-width as "the MDE" understates it by 1.43x.
    """
    return float((sps.norm.ppf(1 - alpha / 2) + sps.norm.ppf(power)) * se)


def achieved_power(effect: float, se: float, alpha: float = 0.05) -> float:
    """Probability of detecting ``effect`` given ``se``."""
    z = sps.norm.ppf(1 - alpha / 2)
    lam = abs(effect) / se
    return float(1 - sps.norm.cdf(z - lam) + sps.norm.cdf(-z - lam))


# ---------------------------------------------------------------------------
# Simulation - a check on the algebra, not an independent method
# ---------------------------------------------------------------------------

def simulate_se(
    s1_ann: float,
    s2_ann: float,
    rho: float,
    n_months: int,
    n_sims: int = 4000,
    vol_annual: float = 0.10,
    seed: int = 42,
) -> float:
    """Monte Carlo standard error under i.i.d. Gaussian returns.

    Reproduces :func:`memmel_se_annual` to within ~1%, as it must - both describe
    the same model. Use it to verify the algebra, never to corroborate an empirical
    finding.
    """
    rng = np.random.default_rng(seed)
    mv = vol_annual / np.sqrt(MONTHS)
    out = np.empty(n_sims)

    for i in range(n_sims):
        z1 = rng.normal(size=n_months)
        z2 = rng.normal(size=n_months)
        r2 = (s2_ann / np.sqrt(MONTHS)) * mv + mv * z1
        r1 = (s1_ann / np.sqrt(MONTHS)) * mv + mv * (rho * z1 + np.sqrt(1 - rho**2) * z2)
        f = lambda x: x.mean() * MONTHS / (x.std(ddof=1) * np.sqrt(MONTHS))  # noqa: E731
        out[i] = f(r1) - f(r2)

    return float(out.std(ddof=1))


# ---------------------------------------------------------------------------
# Tables
# ---------------------------------------------------------------------------

def years_table(
    benchmark_sharpe_ann: float,
    rhos: tuple[float, ...] = DEFAULT_RHOS,
    deltas: tuple[float, ...] = DEFAULT_DELTAS,
    se_inflation: float = 1.0,
    power: float = 0.80,
) -> pd.DataFrame:
    """Years required, by rho and effect size. Reproduces paper Section 7.3.

    ``se_inflation`` is a standard-error ratio; :func:`years_required` squares it.
    """
    rows = []
    for rho in rhos:
        row = {"rho": rho}
        for d in deltas:
            row[f"delta_{d:.2f}"] = round(
                years_required(d, benchmark_sharpe_ann, rho,
                               power=power, se_inflation=se_inflation)
            )
        rows.append(row)
    return pd.DataFrame(rows)


def reconcile(
    results: dict[str, pd.DataFrame],
    benchmark_key: str,
    rf: pd.Series | None,
    n_iterations: int = 8000,
    seed: int = 7,
) -> pd.DataFrame:
    """Pinned comparison of empirical and analytic standard errors.

    Every row is evaluated at the SAME T and benchmark Sharpe, and at that
    comparison's OWN measured rho.

    The ``ratio`` column is a ratio of STANDARD ERRORS (boot_se / memmel_se). It
    reflects non-normality AND serial dependence jointly, since the stationary
    bootstrap with a 12-month mean block preserves both. Square it before using it
    anywhere a variance is required.
    """
    from ..backtest.stats import bootstrap_sharpe_difference

    common = None
    for res in results.values():
        idx = res["net_return"].dropna().index
        common = idx if common is None else common.intersection(idx)

    bench = results[benchmark_key]["net_return"].loc[common]
    n = len(common)
    rows = []

    for name, res in results.items():
        if name == benchmark_key:
            continue
        b = bootstrap_sharpe_difference(
            res["net_return"].loc[common], bench, rf=rf,
            n_iterations=n_iterations, seed=seed,
        )
        boot_se = (b["ci_upper_95"] - b["ci_lower_95"]) / (2 * 1.96)
        ana_se = memmel_se_annual(
            b["sharpe_strategy"], b["sharpe_benchmark"], b["correlation"], n
        )
        rows.append(
            {
                "comparison": name,
                "rho": round(b["correlation"], 4),
                "S1": b["sharpe_strategy"],
                "S2": b["sharpe_benchmark"],
                "n_months": n,
                # full precision stored: the published tables round for display,
                # but tests/test_units_and_tables.py recomputes each derived column
                # from these, and rounding here would make that check vacuous
                "boot_se": boot_se,
                "memmel_se": ana_se,
                "ratio": boot_se / ana_se,
                "mde_80": minimum_detectable_effect(boot_se, 0.80),
                "mde_50": minimum_detectable_effect(boot_se, 0.50),
                "observed_effect": abs(b["sharpe_difference"]),
                "achieved_power": achieved_power(b["sharpe_difference"], boot_se),
            }
        )

    return pd.DataFrame(rows).sort_values("rho", ascending=False, ignore_index=True)


def main() -> None:  # pragma: no cover
    import warnings

    from ..backtest.engine import run_strategy_suite, states_to_probabilities
    from ..backtest.stats import risk_free_monthly
    from ..config import REPORT_DIR, model_config
    from ..data.assets import build_asset_returns
    from ..data.fred import fetch_all
    from ..data.registry import load_specs
    from ..features import build_features
    from ..models.hmm import expanding_window_states
    from ..models.rules import classify_cycle, classify_quadrant

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    warnings.filterwarnings("ignore")

    raw = fetch_all(load_specs())
    rf = risk_free_monthly(raw)
    F = build_features(raw, start="1962-01-31")
    comp = F["composites"]
    returns, _ = build_asset_returns(raw)
    R = returns[["equity", "bonds", "commodities"]].loc["1970-01-31":].dropna()

    cfg = model_config()["hmm"]
    X = comp[["growth_composite", "inflation_composite", "financial_composite"]].dropna()
    walk = expanding_window_states(
        X, n_states=int(cfg.get("n_states", 3)),
        initial_train_months=int(cfg["refit"]["initial_train_months"]),
        refit_every_months=int(cfg["refit"]["refit_every_months"]), n_init=6,
    )
    pcols = [c for c in walk.columns if c.startswith("p_state_")]
    hmm_probs = walk[pcols].dropna()
    hmm_probs.columns = ["hmm_" + c.split("_")[-1] for c in pcols]

    probs = {
        "rule_cycle": states_to_probabilities(classify_cycle(comp)["regime"]),
        "rule_quadrant": states_to_probabilities(classify_quadrant(comp)["regime"]),
        "hmm_walkforward": hmm_probs,
    }
    results = run_strategy_suite(R, probs)

    rec = reconcile(results, "sixty_forty", rf)
    rec.to_csv(REPORT_DIR / "power_reconciliation.csv", index=False)

    s2 = float(rec["S2"].iloc[0])
    infl = float(rec["ratio"].median())
    years = years_table(s2, se_inflation=infl)
    years.to_csv(REPORT_DIR / "power_years_by_rho.csv", index=False)

    print("\nPINNED RECONCILIATION (paper Section 7.2)")
    print(rec[["comparison", "rho", "S1", "S2", "n_months",
               "boot_se", "memmel_se", "ratio"]].round(4).to_string(index=False))
    print(f"\n  median variance inflation = {infl:.3f}"
          f"  (range {rec['ratio'].min():.3f}-{rec['ratio'].max():.3f})")

    print("\nALGEBRA CHECK - simulation vs closed form at matched rho")
    for _, r in rec.iterrows():
        mc = simulate_se(r["S1"], r["S2"], r["rho"], int(r["n_months"]), n_sims=3000)
        print(f"  {r['comparison']:>20}  rho={r['rho']:.3f}  "
              f"MC={mc:.4f}  Memmel={r['memmel_se']:.4f}  ratio={mc / r['memmel_se']:.3f}")

    print(f"\nYEARS REQUIRED (paper Section 7.3), S2={s2:.3f}, inflation x{infl:.3f}")
    print(years.to_string(index=False))

    print("\nWHAT THIS STUDY COULD DETECT (paper Section 7.4)")
    print(rec[["comparison", "rho", "mde_80", "mde_50",
               "observed_effect", "achieved_power"]].round(3).to_string(index=False))
    print(f"\nwritten to {REPORT_DIR}")


if __name__ == "__main__":  # pragma: no cover
    main()

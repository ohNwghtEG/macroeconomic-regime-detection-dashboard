"""The backbone test suite: prove the feature matrix contains no future information.

Every other result in this project is worthless if these fail. They are written to
be adversarial rather than confirmatory - in particular ``test_future_mutation_*``
actively tries to leak the future and asserts that it cannot.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from macroregime.data.fred import fetch_all
from macroregime.data.registry import (
    SeriesSpec,
    asof_feature_matrix,
    feature_specs,
    load_specs,
    period_end,
    release_dates,
    to_point_in_time,
)


@pytest.fixture(scope="module")
def raw():
    """Cached FRED data. Runs offline; no API key required."""
    return fetch_all(load_specs())


@pytest.fixture(scope="module")
def specs():
    return feature_specs()


# ---------------------------------------------------------------------------
# Structural guarantees
# ---------------------------------------------------------------------------

def test_release_date_strictly_after_period_end():
    """A value cannot be published before the period it measures has ended."""
    for spec in load_specs().values():
        idx = pd.DatetimeIndex(pd.date_range("2000-01-01", periods=24, freq="MS"))
        ends = period_end(idx, spec.frequency)
        rels = release_dates(idx, spec)
        assert (rels > ends).all(), f"{spec.id}: release_date must postdate period_end"


def test_monthly_period_end_is_month_end():
    """FRED stamps monthly data at month START. Mis-handling this shifts every
    release date by up to a month - a silent one-month look-ahead."""
    spec = SeriesSpec("X", "X", "monthly", 16, "level", "test")
    idx = pd.DatetimeIndex(["2008-03-01", "2008-04-01"])
    ends = period_end(idx, spec.frequency)
    assert list(ends) == [pd.Timestamp("2008-03-31"), pd.Timestamp("2008-04-30")]

    rels = release_dates(idx, spec)
    # March 2008 CPI was published mid-April 2008, not in March.
    assert rels[0] == pd.Timestamp("2008-04-16")
    assert rels[0] > pd.Timestamp("2008-03-31")


def test_validation_series_cannot_enter_features(specs):
    """NBER dates are announced 6-18 months late. Using USREC as a feature would be
    the most extreme look-ahead possible, so it is blocked structurally."""
    assert "USREC" not in specs
    assert "SAHMREALTIME" not in specs
    assert load_specs()["USREC"].is_feature is False


def test_every_feature_series_has_positive_lag(specs):
    for sid, spec in specs.items():
        assert spec.publication_lag_days >= 1, f"{sid} claims zero publication lag"


# ---------------------------------------------------------------------------
# THE CENTRAL PROPERTY
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "asof",
    [pd.Timestamp("1995-06-30"), pd.Timestamp("2008-09-30"), pd.Timestamp("2020-03-31")],
)
def test_row_uses_only_data_released_by_asof(raw, specs, asof):
    """Directly assert the defining property.

    For each series, the value in row T must equal the last observation whose
    release date is <= T - and must NOT equal a later one.
    """
    X = asof_feature_matrix(raw, specs, pd.DatetimeIndex([asof]))

    for sid, spec in specs.items():
        if sid not in raw:
            continue
        pit = to_point_in_time(raw[sid], spec)
        if pit.empty:
            continue

        available = pit[pit["release_date"] <= asof]
        got = X.loc[asof, sid]

        if available.empty:
            assert pd.isna(got), f"{sid}: value present before any release existed"
            continue

        expected = available.iloc[-1]["value"]
        assert got == pytest.approx(expected, rel=1e-12), f"{sid} mismatch at {asof}"

        future = pit[pit["release_date"] > asof]
        if not future.empty:
            nxt = future.iloc[0]["value"]
            if not np.isclose(nxt, expected, rtol=1e-9):
                assert not np.isclose(got, nxt, rtol=1e-12), (
                    f"{sid}: row {asof} carries a value not released until "
                    f"{future.iloc[0]['release_date'].date()}"
                )


def test_truncating_the_future_changes_nothing(raw, specs):
    """The strongest formulation: delete everything after T and rebuild.

    If the matrix is honest, rows at or before T must be bit-identical whether or
    not the future exists in the input.
    """
    cutoff = pd.Timestamp("2010-12-31")
    idx = pd.date_range("1995-01-31", cutoff, freq="ME")

    full = asof_feature_matrix(raw, specs, idx)

    truncated_raw = {
        sid: s[pd.DatetimeIndex(s.index) <= cutoff] for sid, s in raw.items()
    }
    truncated = asof_feature_matrix(truncated_raw, specs, idx)

    common = [c for c in full.columns if c in truncated.columns]
    pd.testing.assert_frame_equal(
        full[common], truncated[common], check_dtype=False, rtol=1e-12
    )


def test_future_mutation_does_not_change_the_past(raw, specs):
    """Mutation test: corrupt future observations and assert the past is untouched.

    This is the test that actually catches centred rolling windows, full-sample
    z-scores, and other subtle leaks - the ones that survive casual inspection.
    """
    cutoff = pd.Timestamp("2005-12-31")
    idx = pd.date_range("1990-01-31", cutoff, freq="ME")

    before = asof_feature_matrix(raw, specs, idx)

    corrupted = {}
    for sid, s in raw.items():
        s2 = s.copy().astype("float64")
        mask = pd.DatetimeIndex(s2.index) > cutoff
        if mask.any():
            s2[mask] = s2[mask] * 1000.0 + 12345.0   # wildly implausible values
        corrupted[sid] = s2

    after = asof_feature_matrix(corrupted, specs, idx)

    common = [c for c in before.columns if c in after.columns]
    pd.testing.assert_frame_equal(
        before[common], after[common], check_dtype=False, rtol=1e-12
    )


def test_transforms_are_backward_looking_only(raw, specs):
    """A transform must never reach forward.

    Applied per series so a failure names the culprit instead of failing the whole
    matrix anonymously.
    """
    cutoff = pd.Timestamp("2012-06-30")

    for sid, spec in specs.items():
        if sid not in raw or raw[sid].dropna().empty:
            continue

        s = raw[sid].astype("float64")
        pit_full = to_point_in_time(s, spec)
        pit_full = pit_full[pit_full["release_date"] <= cutoff]

        s_trunc = s[pd.DatetimeIndex(s.index) <= cutoff]
        pit_trunc = to_point_in_time(s_trunc, spec)
        pit_trunc = pit_trunc[pit_trunc["release_date"] <= cutoff]

        if pit_full.empty or pit_trunc.empty:
            continue

        n = min(len(pit_full), len(pit_trunc))
        np.testing.assert_allclose(
            pit_full["value"].to_numpy()[-n:],
            pit_trunc["value"].to_numpy()[-n:],
            rtol=1e-10,
            err_msg=f"{sid}: transform {spec.transform!r} appears to use future data",
        )


def test_asof_matrix_is_monotonic_and_unique(raw, specs):
    idx = pd.date_range("2000-01-31", "2020-12-31", freq="ME")
    X = asof_feature_matrix(raw, specs, idx)
    assert X.index.is_monotonic_increasing
    assert X.index.is_unique
    assert len(X) == len(idx)

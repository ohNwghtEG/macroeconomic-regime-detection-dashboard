"""Backtest and model integrity tests.

The centrepiece is ``test_breaking_the_lag_improves_performance``: deliberately
reintroduce look-ahead and require that results get BETTER. If they don't, the lag
was never doing anything and every other guarantee is suspect.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from macroregime.backtest.engine import (
    DEFAULT_REGIME_WEIGHTS,
    backtest,
    benchmark_weights,
    momentum_weights,
    probability_weighted_allocation,
    states_to_probabilities,
)
from macroregime.backtest.stats import (
    bootstrap_sharpe_difference,
    performance_summary,
)
from macroregime.models.hmm import (
    filtered_probabilities,
    fit_hmm,
    relabel_by_growth,
    smoothed_probabilities,
)
from macroregime.models.hysteresis import apply_confirmation, regime_episodes


@pytest.fixture(scope="module")
def synthetic():
    """Two-regime market: a calm bull state and a volatile bear state."""
    rng = np.random.default_rng(7)
    n = 480
    idx = pd.date_range("1985-01-31", periods=n, freq="ME")

    state = np.zeros(n, dtype=int)
    for t in range(1, n):
        state[t] = state[t - 1] if rng.random() < 0.94 else 1 - state[t - 1]

    equity = np.where(state == 0, rng.normal(0.010, 0.03, n), rng.normal(-0.008, 0.06, n))
    bonds = np.where(state == 0, rng.normal(0.003, 0.015, n), rng.normal(0.007, 0.02, n))
    commodities = rng.normal(0.002, 0.04, n)

    returns = pd.DataFrame(
        {"equity": equity, "bonds": bonds, "commodities": commodities}, index=idx
    )
    regimes = pd.Series(
        np.where(state == 0, "expansion", "recession"), index=idx, name="regime"
    )
    return returns, regimes


# ---------------------------------------------------------------------------
# THE INVERSE TEST
# ---------------------------------------------------------------------------

def test_breaking_the_lag_improves_performance():
    """Trading on the SAME bar as the signal must beat trading on the next one.

    This is the sanity check that the trade lag is load-bearing. A zero-lag
    backtest sees each month's regime before earning that month's return - exactly
    the look-ahead the pipeline exists to prevent - so it must look conspicuously
    better. If it doesn't, the lag is decorative.

    Dedicated data rather than the shared fixture: this test is about MECHANICS,
    so the regime is made to switch often and the return spread made large. With
    realistically persistent regimes the lagged signal is already right ~94% of the
    time and foresight's advantage disappears into the noise - which makes for a
    weak test, not a passing one.
    """
    n = 360
    idx = pd.date_range("1990-01-31", periods=n, freq="ME")
    state = np.array([(i // 3) % 2 for i in range(n)])   # flips every 3 months

    rng = np.random.default_rng(99)
    equity = np.where(state == 0, 0.020, -0.020) + rng.normal(0, 0.005, n)
    bonds = np.where(state == 0, 0.001, 0.008) + rng.normal(0, 0.003, n)
    commodities = rng.normal(0.001, 0.005, n)

    returns = pd.DataFrame(
        {"equity": equity, "bonds": bonds, "commodities": commodities}, index=idx
    )
    regimes = pd.Series(
        np.where(state == 0, "expansion", "recession"), index=idx, name="regime"
    )

    weights = probability_weighted_allocation(
        states_to_probabilities(regimes), DEFAULT_REGIME_WEIGHTS, list(returns.columns)
    )

    honest = performance_summary(backtest(weights, returns, 10, signal_lag_months=1))["sharpe"]
    cheating = performance_summary(backtest(weights, returns, 10, signal_lag_months=0))["sharpe"]

    assert cheating > honest, (
        f"Look-ahead ({cheating:.3f}) did not beat the honest backtest "
        f"({honest:.3f}) - the trade lag is not actually binding"
    )
    # and the gap should be large, not marginal
    assert cheating - honest > 0.5, (
        f"look-ahead advantage of only {cheating - honest:.3f} is suspiciously small"
    )


def test_signal_lag_actually_shifts_weights(synthetic):
    """The traded weight at T must equal the target weight from T-1."""
    returns, regimes = synthetic
    probs = states_to_probabilities(regimes)
    W = probability_weighted_allocation(probs, DEFAULT_REGIME_WEIGHTS, list(returns.columns))

    res = backtest(W, returns, cost_bps=0, signal_lag_months=1)
    implied = res["gross_return"] / (W.shift(1) * returns).sum(axis=1).reindex(res.index)
    assert np.allclose(implied.dropna(), 1.0, atol=1e-9)


# ---------------------------------------------------------------------------
# Engine mechanics
# ---------------------------------------------------------------------------

def test_weights_sum_to_one(synthetic):
    returns, regimes = synthetic
    W = probability_weighted_allocation(
        states_to_probabilities(regimes), DEFAULT_REGIME_WEIGHTS, list(returns.columns)
    )
    totals = W.sum(axis=1)
    assert np.allclose(totals[totals > 0], 1.0, atol=1e-9)


def test_no_negative_weights(synthetic):
    """Long-only: the engine must never imply a short."""
    returns, regimes = synthetic
    W = probability_weighted_allocation(
        states_to_probabilities(regimes), DEFAULT_REGIME_WEIGHTS, list(returns.columns)
    )
    assert (W >= -1e-12).all().all()


def test_costs_reduce_returns(synthetic):
    """Higher costs must lower net return - guards against a sign error."""
    returns, regimes = synthetic
    W = probability_weighted_allocation(
        states_to_probabilities(regimes), DEFAULT_REGIME_WEIGHTS, list(returns.columns)
    )
    cheap = backtest(W, returns, cost_bps=0)
    dear = backtest(W, returns, cost_bps=100)
    assert dear["net_return"].mean() < cheap["net_return"].mean()
    assert np.allclose(cheap["cost"], 0.0)


def test_static_benchmark_has_no_turnover(synthetic):
    """A constant-weight benchmark should incur no rebalancing turnover here."""
    returns, _ = synthetic
    W = benchmark_weights(returns.index, list(returns.columns), "sixty_forty")
    res = backtest(W, returns, cost_bps=10)
    assert res["turnover"].iloc[1:].max() < 1e-9


def test_momentum_uses_no_future_data(synthetic):
    """Momentum weights at T must not change when the future is corrupted."""
    returns, _ = synthetic
    cutoff = returns.index[300]

    before = momentum_weights(returns).loc[:cutoff]

    corrupted = returns.copy()
    corrupted.loc[corrupted.index > cutoff] *= 100.0
    after = momentum_weights(corrupted).loc[:cutoff]

    pd.testing.assert_frame_equal(before, after)


# ---------------------------------------------------------------------------
# HMM
# ---------------------------------------------------------------------------

def test_filtered_equals_smoothed_at_final_step():
    """A mathematical identity: at T both condition on all available data.

    If these differ, the forward filter is wrong. If they match EVERYWHERE, the
    "filter" is secretly returning smoothed values - the exact bug this guards.
    """
    rng = np.random.default_rng(0)
    X = np.vstack([rng.normal(-1, 0.5, (200, 2)), rng.normal(1, 0.5, (200, 2))])

    model, _ = fit_hmm(X, 2, n_init=4, seed=0)
    model, _ = relabel_by_growth(model)

    filt = filtered_probabilities(model, X)
    smooth = smoothed_probabilities(model, X)

    np.testing.assert_allclose(filt[-1], smooth[-1], atol=1e-9)
    assert np.abs(filt[:-1] - smooth[:-1]).mean() > 1e-6, "filter is not causal"


def test_filtered_probabilities_are_valid():
    rng = np.random.default_rng(1)
    X = rng.normal(0, 1, (300, 3))
    model, _ = fit_hmm(X, 3, n_init=3, seed=1)
    probs = filtered_probabilities(model, X)

    assert probs.shape == (300, 3)
    np.testing.assert_allclose(probs.sum(axis=1), 1.0, atol=1e-9)
    assert (probs >= 0).all() and (probs <= 1).all()


def test_filtered_ignores_the_future():
    """Filtered probability at t must not change when data after t changes."""
    rng = np.random.default_rng(2)
    X = rng.normal(0, 1, (300, 2))
    model, _ = fit_hmm(X, 2, n_init=3, seed=2)
    model, _ = relabel_by_growth(model)

    full = filtered_probabilities(model, X)
    truncated = filtered_probabilities(model, X[:150])

    np.testing.assert_allclose(full[:150], truncated, atol=1e-10)


def test_relabelling_is_deterministic():
    """State 0 must always be the weakest-growth state, whatever the seed."""
    rng = np.random.default_rng(3)
    X = np.vstack([rng.normal(-2, 0.5, (150, 2)), rng.normal(2, 0.5, (150, 2))])

    for seed in (0, 10, 20):
        model, _ = fit_hmm(X, 2, n_init=3, seed=seed)
        model, _ = relabel_by_growth(model, growth_col=0)
        assert model.means_[0, 0] < model.means_[1, 0]


# ---------------------------------------------------------------------------
# Hysteresis
# ---------------------------------------------------------------------------

def test_confirmation_suppresses_single_month_blips():
    states = pd.Series(
        ["a", "a", "a", "b", "a", "a", "a"],
        index=pd.date_range("2000-01-31", periods=7, freq="ME"),
    )
    out = apply_confirmation(states, confirm_periods=2)
    assert (out == "a").all(), "a one-month blip should not flip the regime"


def test_confirmation_adopts_persistent_change():
    states = pd.Series(
        ["a", "a", "b", "b", "b"],
        index=pd.date_range("2000-01-31", periods=5, freq="ME"),
    )
    out = apply_confirmation(states, confirm_periods=2)
    assert out.iloc[-1] == "b"
    assert out.iloc[2] == "a", "switch should date from the confirming month, not the first"


def test_confirmation_is_causal():
    """Truncating the future must not change earlier confirmed states."""
    rng = np.random.default_rng(5)
    raw = pd.Series(
        rng.choice(["a", "b"], 200),
        index=pd.date_range("2000-01-31", periods=200, freq="ME"),
    )
    full = apply_confirmation(raw, 3)
    trunc = apply_confirmation(raw.iloc[:100], 3)
    pd.testing.assert_series_equal(full.iloc[:100], trunc)


def test_episodes_count_contiguous_blocks():
    states = pd.Series(
        ["a", "a", "b", "b", "b", "a"],
        index=pd.date_range("2000-01-31", periods=6, freq="ME"),
    )
    ep = regime_episodes(states)
    assert len(ep) == 3
    assert list(ep["months"]) == [2, 3, 1]


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------

def test_bootstrap_detects_no_difference_between_identical_series():
    rng = np.random.default_rng(11)
    r = pd.Series(
        rng.normal(0.005, 0.03, 400),
        index=pd.date_range("1990-01-31", periods=400, freq="ME"),
    )
    res = bootstrap_sharpe_difference(r, r.copy(), n_iterations=400)
    assert abs(res["sharpe_difference"]) < 1e-9
    assert not res["significant_at_5pct"]


def test_bootstrap_detects_a_large_real_difference():
    rng = np.random.default_rng(12)
    idx = pd.date_range("1990-01-31", periods=600, freq="ME")
    good = pd.Series(rng.normal(0.012, 0.03, 600), index=idx)
    bad = pd.Series(rng.normal(0.001, 0.03, 600), index=idx)

    res = bootstrap_sharpe_difference(good, bad, n_iterations=1500)
    assert res["sharpe_difference"] > 0
    assert res["significant_at_5pct"], "a very large true difference should register"


def test_newey_west_se_exceeds_naive_for_overlapping_data():
    """On autocorrelated data the HAC error must exceed the naive one.

    That gap IS the significance that overlapping forward returns would otherwise
    manufacture.
    """
    from macroregime.evaluation.conditional import newey_west_se

    rng = np.random.default_rng(13)
    shocks = rng.normal(0, 1, 500)
    overlapping = pd.Series(np.convolve(shocks, np.ones(12), mode="valid"))

    naive = overlapping.std(ddof=1) / np.sqrt(len(overlapping))
    hac = newey_west_se(overlapping, lags=11)

    assert hac > naive * 1.5

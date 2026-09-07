"""Look-ahead bias ablation - the central experiment.

THE QUESTION
------------
Every published version of this pipeline reports a flattering Sharpe. This module
asks *which shortcut buys how much of it* by re-running the identical strategy with
each bias switched back on, alone and in combination.

THREE SWITCHES
--------------
**A - publication lag ignored.** Align each series to the end of its reference
period rather than its release date, i.e. pretend March CPI is available on 31
March instead of ~16 April. This is what ``df.resample("ME").last()`` on raw FRED
data silently does, and it is the single most common form of the error.

**B - smoothed decoding.** Use ``predict_proba()`` over the whole series
(``P(s_t | ALL data)``) instead of the forward filter (``P(s_t | data <= t)``).

**C - full-sample fitting.** Estimate HMM parameters once on all history instead of
refitting on an expanding window.

WHY ALL EIGHT COMBINATIONS
--------------------------
One-at-a-time ablation assumes the biases are additive. They need not be: a model
that already sees revised data may gain less from smoothing, or more. Running the
full 2x2x2 factorial exposes the interactions, and the interaction terms are a
result in their own right.

A NOTE ON WHAT IS AND IS NOT MEASURED HERE
------------------------------------------
Switch A captures publication *timing* only. It does NOT capture *revisions* - the
fact that FRED serves today's restated value for 1975. Measuring that requires
ALFRED vintages and is scoped separately. So the numbers below are a LOWER BOUND on
total look-ahead: the revision channel sits on top of everything measured here.
"""

from __future__ import annotations

import itertools
import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

from ..backtest.engine import (
    DEFAULT_REGIME_WEIGHTS,
    backtest,
    benchmark_weights,
    probability_weighted_allocation,
)
from ..backtest.stats import performance_summary
from ..config import model_config
from ..data.registry import SeriesSpec, asof_feature_matrix, feature_specs
from ..features.build import build_composites, build_raw_with_derived
from ..features.standardize import expanding_zscore_frame
from ..models.hmm import (
    filtered_probabilities,
    fit_hmm,
    relabel_by_growth,
    smoothed_probabilities,
)

log = logging.getLogger(__name__)

HMM_FEATURES = ["growth_composite", "inflation_composite", "financial_composite"]


@dataclass(frozen=True)
class AblationConfig:
    """One cell of the factorial design."""

    ignore_publication_lag: bool   # A
    smoothed_decoding: bool        # B
    full_sample_fit: bool          # C

    @property
    def label(self) -> str:
        flags = "".join(
            letter
            for letter, on in zip(
                "ABC",
                (self.ignore_publication_lag, self.smoothed_decoding, self.full_sample_fit),
            )
            if on
        )
        return flags or "none (honest)"

    @property
    def n_biases(self) -> int:
        return sum(
            (self.ignore_publication_lag, self.smoothed_decoding, self.full_sample_fit)
        )


# ---------------------------------------------------------------------------
# Switch A - features with and without release-date discipline
# ---------------------------------------------------------------------------

def build_features_variant(
    raw: dict[str, pd.Series],
    asof_index: pd.DatetimeIndex,
    ignore_publication_lag: bool,
) -> pd.DataFrame:
    """Build composites either honestly or with publication lag stripped out."""
    raw2, specs2 = build_raw_with_derived(raw, feature_specs())

    if ignore_publication_lag:
        # zero every lag: data becomes available the instant its period ends
        specs2 = {
            sid: SeriesSpec(
                id=s.id, name=s.name, frequency=s.frequency,
                publication_lag_days=0, transform=s.transform,
                dimension=s.dimension, note=s.note, usage=s.usage,
            )
            for sid, s in specs2.items()
        }

    cfg = model_config()["standardization"]
    levels = asof_feature_matrix(raw2, specs2, asof_index)
    z = expanding_zscore_frame(
        levels,
        min_periods=int(cfg.get("min_periods", 60)),
        clip=cfg.get("clip_sigma"),
    )
    return build_composites(z)


# ---------------------------------------------------------------------------
# Switches B and C - regime probabilities under each fitting/decoding scheme
# ---------------------------------------------------------------------------

def regime_probabilities_variant(
    X: pd.DataFrame,
    smoothed_decoding: bool,
    full_sample_fit: bool,
    n_states: int = 3,
    initial_train_months: int = 180,
    refit_every_months: int = 12,
    n_init: int = 6,
    seed: int = 42,
) -> pd.DataFrame:
    """Regime probabilities under one (B, C) combination.

    The four cells are genuinely distinct:

    * ``(filtered, expanding)``  - honest.
    * ``(filtered, full-sample)`` - causal decoding, but parameters that already saw
      the whole history.
    * ``(smoothed, expanding)``  - refits correctly, then decodes over the entire
      series. Incoherent, and exactly what happens when someone refits in a loop and
      then calls ``predict_proba(X_full)``.
    * ``(smoothed, full-sample)`` - the textbook naive implementation.
    """
    X_np = X.to_numpy(dtype="float64")
    cols = [f"hmm_{i}" for i in range(n_states)]

    if full_sample_fit:
        model, _ = fit_hmm(X_np, n_states, n_init=n_init, seed=seed)
        model, _ = relabel_by_growth(model, growth_col=0)
        probs = (
            smoothed_probabilities(model, X_np)
            if smoothed_decoding
            else filtered_probabilities(model, X_np)
        )
        return pd.DataFrame(probs, index=X.index, columns=cols)

    # expanding-window refits
    out = pd.DataFrame(np.nan, index=X.index, columns=cols)
    model = None
    last_refit = -np.inf

    for t in range(initial_train_months, len(X)):
        if (t - last_refit) >= refit_every_months or model is None:
            try:
                model, _ = fit_hmm(X_np[:t], n_states, n_init=n_init, seed=seed)
                model, _ = relabel_by_growth(model, growth_col=0)
                last_refit = t
            except Exception as exc:  # noqa: BLE001
                log.warning("refit failed at %s: %s", X.index[t], exc)
                if model is None:
                    continue

        if smoothed_decoding:
            # THE LEAK: decode over the entire series, future included
            out.iloc[t] = smoothed_probabilities(model, X_np)[t]
        else:
            out.iloc[t] = filtered_probabilities(model, X_np[: t + 1])[-1]

    return out.dropna()


# ---------------------------------------------------------------------------
# The experiment
# ---------------------------------------------------------------------------

def run_ablation(
    raw: dict[str, pd.Series],
    returns: pd.DataFrame,
    asof_index: pd.DatetimeIndex | None = None,
    n_states: int = 3,
    cost_bps: float = 10.0,
    signal_lag_months: int = 1,
    n_init: int = 6,
    seed: int = 42,
) -> pd.DataFrame:
    """Run all eight cells and report Sharpe for each.

    Every cell uses identical assets, weights, costs and trade lag, on a common
    sample. Only the three bias switches vary, so differences are attributable.
    """
    if asof_index is None:
        asof_index = pd.date_range("1962-01-31", "2026-08-31", freq="ME")

    assets = list(returns.columns)
    cfg = model_config()["hmm"]["refit"]

    # cache the two feature variants (switch A) - each is reused across B and C
    features: dict[bool, pd.DataFrame] = {}
    for ignore_lag in (False, True):
        comp = build_features_variant(raw, asof_index, ignore_lag)
        features[ignore_lag] = comp[HMM_FEATURES].dropna()
        log.info("features (ignore_lag=%s): %s", ignore_lag, features[ignore_lag].shape)

    # cache probabilities per (A, B, C) - the expanding runs are the slow part
    prob_cache: dict[tuple, pd.DataFrame] = {}
    for key in itertools.product((False, True), repeat=3):
        ignore_lag, smoothed, full_fit = key
        conf = AblationConfig(*key)
        log.info("running cell %s", conf.label)
        prob_cache[key] = regime_probabilities_variant(
            features[ignore_lag],
            smoothed_decoding=smoothed,
            full_sample_fit=full_fit,
            n_states=n_states,
            initial_train_months=int(cfg["initial_train_months"]),
            refit_every_months=int(cfg["refit_every_months"]),
            n_init=n_init,
            seed=seed,
        )

    # CRITICAL: every cell must be scored on the SAME months.
    # Full-sample fitting emits a signal from the first observation, while the
    # expanding-window cells only begin after their burn-in. Left unaligned, the
    # comparison confounds the bias under test with a different sample period -
    # and the burn-in gap here is over a decade, spanning the entire Volcker
    # disinflation. Intersecting the indices is what makes the cells comparable.
    common = None
    for probs in prob_cache.values():
        idx = probs.dropna().index
        common = idx if common is None else common.intersection(idx)
    common = common.intersection(returns.index)
    log.info("common ablation sample: %d months (%s to %s)",
             len(common), common.min().date(), common.max().date())

    rows = []
    for key in itertools.product((False, True), repeat=3):
        ignore_lag, smoothed, full_fit = key
        conf = AblationConfig(*key)

        probs = prob_cache[key].loc[common]
        W = probability_weighted_allocation(probs, DEFAULT_REGIME_WEIGHTS, assets)
        res = backtest(W, returns.loc[common], cost_bps, signal_lag_months,
                       name=conf.label)

        if res.empty:
            log.warning("cell %s produced no results", conf.label)
            continue

        summary = performance_summary(res)
        rows.append(
            {
                "cell": conf.label,
                "A_ignore_lag": ignore_lag,
                "B_smoothed": smoothed,
                "C_full_sample": full_fit,
                "n_biases": conf.n_biases,
                "sharpe": summary["sharpe"],
                "ann_return_%": summary["ann_return_%"],
                "ann_vol_%": summary["ann_vol_%"],
                "max_drawdown_%": summary["max_drawdown_%"],
                "months": summary["months"],
                "start": summary["start"],
            }
        )

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    honest = df.loc[df["n_biases"] == 0, "sharpe"]
    if not honest.empty:
        df["sharpe_inflation"] = (df["sharpe"] - float(honest.iloc[0])).round(3)

    return df.sort_values("n_biases", ignore_index=True)


def attribute_effects(ablation: pd.DataFrame) -> pd.DataFrame:
    """Decompose the total inflation into main effects and interactions.

    Standard factorial attribution: each main effect is the mean Sharpe difference
    from switching that bias on, averaged over the other two. Interaction terms
    capture whether the biases compound or partially substitute for one another -
    if they were additive, the interactions would be zero.
    """
    if ablation.empty:
        return pd.DataFrame()

    df = ablation.copy()
    A = df["A_ignore_lag"].astype(int).to_numpy()
    B = df["B_smoothed"].astype(int).to_numpy()
    C = df["C_full_sample"].astype(int).to_numpy()
    y = df["sharpe"].to_numpy(dtype="float64")

    # +/-1 coding makes the coefficients interpretable as main effects
    a, b, c = 2 * A - 1, 2 * B - 1, 2 * C - 1
    design = {
        "A: ignore publication lag": a,
        "B: smoothed decoding": b,
        "C: full-sample fit": c,
        "A x B": a * b,
        "A x C": a * c,
        "B x C": b * c,
        "A x B x C": a * b * c,
    }

    rows = []
    for name, col in design.items():
        # effect = mean(y | +1) - mean(y | -1)
        effect = float(y[col > 0].mean() - y[col < 0].mean())
        rows.append({"effect": name, "sharpe_impact": round(effect, 4)})

    out = pd.DataFrame(rows)
    out["abs_impact"] = out["sharpe_impact"].abs()
    return out.sort_values("abs_impact", ascending=False, ignore_index=True).drop(
        columns="abs_impact"
    )


# ---------------------------------------------------------------------------
# In-sample weight optimization - the free parameter
# ---------------------------------------------------------------------------

def max_sharpe_long_only(returns: pd.DataFrame) -> np.ndarray:
    """Long-only max-Sharpe weights, fitted on whatever sample it is given.

    Deliberately naive: no shrinkage, no regularisation, no out-of-sample split.
    That is the point - this stands in for the free parameter a practitioner adds
    when they "calibrate" allocations to each regime.
    """
    from scipy.optimize import minimize

    mu = returns.mean().to_numpy(dtype="float64")
    cov = returns.cov().to_numpy(dtype="float64")
    n = len(mu)

    def neg_sharpe(w: np.ndarray) -> float:
        vol = float(np.sqrt(w @ cov @ w))
        return 1e6 if vol <= 0 else -float(w @ mu) / vol

    result = minimize(
        neg_sharpe,
        x0=np.full(n, 1.0 / n),
        method="SLSQP",
        bounds=[(0.0, 1.0)] * n,
        constraints=[{"type": "eq", "fun": lambda w: w.sum() - 1.0}],
        options={"maxiter": 500, "ftol": 1e-10},
    )
    w = result.x if result.success else np.full(n, 1.0 / n)
    return np.clip(w, 0.0, 1.0) / max(np.clip(w, 0.0, 1.0).sum(), 1e-12)


def optimize_regime_weights(
    returns: pd.DataFrame,
    probabilities: pd.DataFrame,
    min_obs: int = 24,
) -> dict[str, dict[str, float]]:
    """Fit per-regime allocations IN SAMPLE.

    This is the treatment condition for the paper's central claim. Every cell gets
    the same optimiser, so any extra benefit accruing to the look-ahead cells is
    attributable to the interaction between the leak and the free parameter - not
    to optimisation itself, which is a constant across cells.
    """
    states = probabilities.idxmax(axis=1)
    idx = returns.index.intersection(states.index)
    states, R = states.loc[idx], returns.loc[idx]

    weights: dict[str, dict[str, float]] = {}
    for state in states.dropna().unique():
        block = R.loc[states == state].dropna()
        if len(block) < min_obs:
            continue
        w = max_sharpe_long_only(block)
        weights[str(state)] = dict(zip(R.columns, w))

    return weights


def run_ablation_weight_modes(
    raw: dict[str, pd.Series],
    returns: pd.DataFrame,
    asof_index: pd.DatetimeIndex | None = None,
    n_states: int = 3,
    cost_bps: float = 10.0,
    signal_lag_months: int = 1,
    n_init: int = 6,
    seed: int = 42,
) -> pd.DataFrame:
    """The decisive experiment: the 8-cell ablation under two weighting schemes.

    * ``fixed``     - hand-specified regime weights (no free parameters)
    * ``optimized`` - per-regime weights fitted in sample (one free parameter set)

    THE HYPOTHESIS
    Look-ahead bias is close to harmless under fixed weights, because better regime
    classification cannot express itself through a coarse pre-committed mapping.
    Add an in-sample optimiser and the leaked information acquires a channel to act
    through, so the same biases should inflate results markedly.

    If that holds, the danger is not the leak by itself - it is the leak in the
    presence of a fitted parameter.
    """
    if asof_index is None:
        asof_index = pd.date_range("1962-01-31", "2026-08-31", freq="ME")

    assets = list(returns.columns)
    cfg = model_config()["hmm"]["refit"]

    features: dict[bool, pd.DataFrame] = {}
    for ignore_lag in (False, True):
        comp = build_features_variant(raw, asof_index, ignore_lag)
        features[ignore_lag] = comp[HMM_FEATURES].dropna()

    prob_cache: dict[tuple, pd.DataFrame] = {}
    for key in itertools.product((False, True), repeat=3):
        ignore_lag, smoothed, full_fit = key
        log.info("fitting cell %s", AblationConfig(*key).label)
        prob_cache[key] = regime_probabilities_variant(
            features[ignore_lag],
            smoothed_decoding=smoothed,
            full_sample_fit=full_fit,
            n_states=n_states,
            initial_train_months=int(cfg["initial_train_months"]),
            refit_every_months=int(cfg["refit_every_months"]),
            n_init=n_init,
            seed=seed,
        )

    common = None
    for probs in prob_cache.values():
        i = probs.dropna().index
        common = i if common is None else common.intersection(i)
    common = common.intersection(returns.index)
    log.info("common sample: %d months", len(common))

    rows = []
    for key in itertools.product((False, True), repeat=3):
        conf = AblationConfig(*key)
        probs = prob_cache[key].loc[common]
        R = returns.loc[common]

        for mode in ("fixed", "optimized"):
            weights = (
                DEFAULT_REGIME_WEIGHTS
                if mode == "fixed"
                else optimize_regime_weights(R, probs)
            )
            W = probability_weighted_allocation(probs, weights, assets)
            res = backtest(W, R, cost_bps, signal_lag_months, name=conf.label)
            if res.empty:
                continue

            s = performance_summary(res)
            rows.append(
                {
                    "weight_mode": mode,
                    "cell": conf.label,
                    "n_biases": conf.n_biases,
                    "A_ignore_lag": key[0],
                    "B_smoothed": key[1],
                    "C_full_sample": key[2],
                    "sharpe": s["sharpe"],
                    "ann_return_%": s["ann_return_%"],
                    "max_drawdown_%": s["max_drawdown_%"],
                }
            )

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    # inflation is measured WITHIN each weighting scheme, so the constant benefit
    # of optimisation nets out and only the leak's contribution remains
    for mode in df["weight_mode"].unique():
        mask = df["weight_mode"] == mode
        base = df.loc[mask & (df["n_biases"] == 0), "sharpe"]
        if not base.empty:
            df.loc[mask, "sharpe_inflation"] = (
                df.loc[mask, "sharpe"] - float(base.iloc[0])
            ).round(3)

    return df.sort_values(["weight_mode", "n_biases"], ignore_index=True)

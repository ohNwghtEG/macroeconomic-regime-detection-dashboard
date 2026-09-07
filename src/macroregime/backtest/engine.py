"""Backtest engine.

MECHANICS THE ORIGINAL BRIEF LEFT UNSPECIFIED
---------------------------------------------
"Backtest a regime-rotation strategy vs static 60/40" hides four decisions, each
of which can manufacture performance on its own:

1. **Signal-to-trade lag.** The regime for month-end T is known at T; the trade
   happens at T+1. Trading at T on T's own signal earns a month of free foresight.
2. **Transaction costs.** Applied to turnover, not ignored. A strategy that flips
   allocation monthly can look excellent gross and be unimplementable net.
3. **The benchmark must be fair.** A REBALANCED 60/40 carrying the same cost model.
   Comparing an actively rebalanced strategy to a buy-and-hold benchmark credits
   rebalancing itself as alpha.
4. **Hard switching vs blending.** Probability-weighted blending
   (``w = sum_k p_k * w_k``) avoids a cliff-edge at 51% confidence and cuts
   turnover substantially.

The engine is deliberately simple and long-only. The interesting question is
whether the SIGNAL has value, and leverage or shorting would only obscure that.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from ..config import asset_config

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Regime -> weights
# ---------------------------------------------------------------------------

DEFAULT_REGIME_WEIGHTS: dict[str, dict[str, float]] = {
    # Business-cycle phases
    "expansion": {"equity": 0.70, "bonds": 0.20, "commodities": 0.10, "gold": 0.00},
    "slowdown":  {"equity": 0.40, "bonds": 0.40, "gold": 0.20, "commodities": 0.00},
    "recession": {"equity": 0.20, "bonds": 0.60, "gold": 0.20, "commodities": 0.00},
    "recovery":  {"equity": 0.60, "bonds": 0.25, "commodities": 0.15, "gold": 0.00},
    # Growth x inflation quadrants
    "goldilocks":  {"equity": 0.70, "bonds": 0.30, "gold": 0.00, "commodities": 0.00},
    "reflation":   {"equity": 0.45, "commodities": 0.30, "gold": 0.15, "bonds": 0.10},
    "stagflation": {"gold": 0.35, "commodities": 0.30, "equity": 0.20, "bonds": 0.15},
    "deflation":   {"bonds": 0.65, "equity": 0.25, "gold": 0.10, "commodities": 0.00},
    # HMM states. relabel_by_growth() sorts states ascending by mean growth, so
    # state 0 is ALWAYS the weakest-growth state and state K-1 the strongest, in
    # every refit. That lets weights be assigned by position without inspecting a
    # particular fit - which would be in-sample tuning.
    "hmm_0": {"bonds": 0.55, "gold": 0.20, "equity": 0.25, "commodities": 0.00},
    "hmm_1": {"equity": 0.45, "bonds": 0.35, "commodities": 0.10, "gold": 0.10},
    "hmm_2": {"equity": 0.65, "bonds": 0.20, "commodities": 0.15, "gold": 0.00},
    "hmm_3": {"equity": 0.70, "bonds": 0.15, "commodities": 0.15, "gold": 0.00},
}


def probability_weighted_allocation(
    probabilities: pd.DataFrame,
    regime_weights: dict[str, dict[str, float]],
    assets: list[str],
) -> pd.DataFrame:
    """Blend per-regime target weights by regime probability.

    ``w_t = sum_k p_k(t) * w_k``

    Preferred over hard switching on the argmax: a 51/49 probability split should
    not produce the same allocation as 99/1, and blending materially reduces
    turnover (and therefore cost) without discarding information.
    """
    W = pd.DataFrame(0.0, index=probabilities.index, columns=assets)

    for regime in probabilities.columns:
        target = regime_weights.get(str(regime))
        if target is None:
            continue
        p = probabilities[regime].fillna(0.0)
        for asset, weight in target.items():
            if asset in W.columns:
                W[asset] += p * weight

    total = W.sum(axis=1).replace(0.0, np.nan)
    return W.div(total, axis=0).fillna(0.0)


def states_to_probabilities(states: pd.Series) -> pd.DataFrame:
    """One-hot a hard state series so it can share the blending code path."""
    s = states.dropna()
    return pd.get_dummies(s).astype("float64").reindex(states.index).fillna(0.0)


def benchmark_weights(
    index: pd.DatetimeIndex,
    assets: list[str],
    kind: str = "sixty_forty",
) -> pd.DataFrame:
    """Static benchmark weights on the given index."""
    W = pd.DataFrame(0.0, index=index, columns=assets)

    if kind == "sixty_forty":
        if "equity" in W.columns:
            W["equity"] = 0.60
        if "bonds" in W.columns:
            W["bonds"] = 0.40
    elif kind == "equal_weight":
        W.loc[:, :] = 1.0 / len(assets)
    else:
        raise ValueError(f"Unknown benchmark {kind!r}")

    total = W.sum(axis=1).replace(0.0, np.nan)
    return W.div(total, axis=0).fillna(0.0)


def momentum_weights(
    returns: pd.DataFrame,
    lookback_months: int = 12,
    top_n: int = 2,
) -> pd.DataFrame:
    """THE CRITICAL BENCHMARK - allocation using prices only, no macro data.

    Markets price the regime before the data confirms it. So the question that
    actually matters is not "does this beat 60/40" but "does macro data add
    anything over a signal that ignores it?"

    Trailing ``lookback_months`` return, equal weight across the top ``top_n``.
    Strictly causal: the signal at T uses returns through T and is traded at T+1,
    exactly like the macro strategy.
    """
    momentum = (1 + returns).rolling(lookback_months).apply(np.prod, raw=True) - 1

    W = pd.DataFrame(0.0, index=returns.index, columns=returns.columns)
    for date in momentum.index:
        row = momentum.loc[date].dropna()
        if len(row) < top_n:
            continue
        winners = row.nlargest(top_n).index
        W.loc[date, winners] = 1.0 / top_n

    return W


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------

def backtest(
    weights: pd.DataFrame,
    returns: pd.DataFrame,
    cost_bps: float = 10.0,
    signal_lag_months: int = 1,
    name: str = "strategy",
) -> pd.DataFrame:
    """Run a long-only backtest.

    ``weights`` are the TARGET weights implied by information at each date; they
    are shifted forward by ``signal_lag_months`` before being applied to returns,
    so the signal at T is traded at T+1 and earns T+1's return.

    Costs are charged on turnover, defined as the one-way sum of absolute weight
    changes.
    """
    idx = weights.index.intersection(returns.index)
    W = weights.loc[idx].copy()
    R = returns.loc[idx, [c for c in W.columns if c in returns.columns]].copy()
    W = W[R.columns]

    # THE critical line: trade on lagged signal, never on the same bar
    W_traded = W.shift(signal_lag_months).fillna(0.0)

    valid = R.notna().all(axis=1) & (W_traded.sum(axis=1) > 0)
    W_traded, R = W_traded[valid], R[valid]
    if W_traded.empty:
        return pd.DataFrame()

    gross = (W_traded * R).sum(axis=1)

    turnover = W_traded.diff().abs().sum(axis=1).fillna(0.0)
    costs = turnover * (cost_bps / 10_000.0)
    net = gross - costs

    out = pd.DataFrame(
        {
            "gross_return": gross,
            "turnover": turnover,
            "cost": costs,
            "net_return": net,
            "equity_curve": (1 + net).cumprod(),
        }
    )
    out.attrs["name"] = name
    out.attrs["cost_bps"] = cost_bps
    out.attrs["signal_lag_months"] = signal_lag_months
    return out


def run_strategy_suite(
    returns: pd.DataFrame,
    regime_probabilities: dict[str, pd.DataFrame],
    regime_weights: dict[str, dict[str, float]] | None = None,
    cost_bps: float | None = None,
    signal_lag_months: int | None = None,
) -> dict[str, pd.DataFrame]:
    """Backtest every regime model plus all benchmarks under identical mechanics.

    Every strategy and benchmark receives the same cost model and the same trade
    lag, so differences reflect the signal rather than the accounting.
    """
    cfg = asset_config().get("backtest", {})
    cost_bps = cfg.get("transaction_cost_bps", 10.0) if cost_bps is None else cost_bps
    signal_lag_months = (
        cfg.get("signal_to_trade_lag_months", 1)
        if signal_lag_months is None else signal_lag_months
    )
    regime_weights = regime_weights or DEFAULT_REGIME_WEIGHTS

    assets = list(returns.columns)
    results: dict[str, pd.DataFrame] = {}

    for model_name, probs in regime_probabilities.items():
        unmapped = [str(c) for c in probs.columns if str(c) not in regime_weights]
        if unmapped:
            # fail loudly: an unmapped state silently zeroes the allocation and the
            # strategy disappears from the results table without any error
            log.error(
                "%s: no target weights for state(s) %s - strategy will be skipped",
                model_name, ", ".join(unmapped),
            )
        W = probability_weighted_allocation(probs, regime_weights, assets)
        res = backtest(W, returns, cost_bps, signal_lag_months, name=model_name)
        if res.empty:
            log.error("%s produced no backtest results", model_name)
            continue
        results[model_name] = res

    idx = returns.index
    for bench in ("sixty_forty", "equal_weight"):
        W = benchmark_weights(idx, assets, kind=bench)
        res = backtest(W, returns, cost_bps, signal_lag_months, name=bench)
        if not res.empty:
            results[bench] = res

    W_mom = momentum_weights(returns)
    res = backtest(W_mom, returns, cost_bps, signal_lag_months, name="momentum_no_macro")
    if not res.empty:
        results["momentum_no_macro"] = res

    return results

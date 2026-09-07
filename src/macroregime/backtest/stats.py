"""Performance statistics and significance testing.

A single backtest Sharpe is not evidence. Two strategies differing by 0.2 Sharpe
over 40 years of monthly data is well within what luck produces, and reporting the
difference without a test is the single most common overclaim in this genre.

Three tests are applied:

* **Stationary bootstrap** (Politis-Romano) of the Sharpe DIFFERENCE. Resamples in
  blocks, preserving the autocorrelation and volatility clustering that an i.i.d.
  bootstrap would destroy.
* **Deflated Sharpe Ratio** (Bailey & Lopez de Prado), which penalises for the
  number of configurations tried. Testing twenty variants and reporting the best
  guarantees an impressive number even from noise.
* **Newey-West t-statistic** on mean excess return.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats as sps

MONTHS = 12


def risk_free_monthly(raw: dict, series: str = "DTB3") -> pd.Series:
    """Monthly risk-free return from a FRED bill-rate series.

    NOT optional. Over 1981-2026 the 3-month bill averaged 3.65% annualised, and
    7.80% during the 1980s portion of the sample. Omitting it does not merely
    shift every strategy equally: the Sharpe ratio divides by each strategy's own
    volatility, so a common subtraction still changes the RANKING and the
    DIFFERENCES between strategies. A "Sharpe" computed on raw returns is a
    return-to-volatility ratio and must be labelled as such.
    """
    d = raw[series].dropna().resample("ME").last()
    return (d / 100.0 / MONTHS).rename("rf")


def to_excess(returns: pd.Series, rf: pd.Series | None) -> pd.Series:
    """Subtract the risk-free rate, aligned on the return index."""
    if rf is None:
        return returns
    return returns - rf.reindex(returns.index).ffill().fillna(0.0)


# ---------------------------------------------------------------------------
# Summary statistics
# ---------------------------------------------------------------------------

def performance_summary(result: pd.DataFrame, rf: pd.Series | None = None) -> dict:
    """Headline statistics for one backtest result."""
    r = result["net_return"].dropna()
    if len(r) < 12:
        return {}

    excess = to_excess(r, rf)

    ann_return = float((1 + r).prod() ** (MONTHS / len(r)) - 1)
    ann_vol = float(r.std(ddof=1) * np.sqrt(MONTHS))
    sharpe = (
        float(excess.mean() * MONTHS / (excess.std(ddof=1) * np.sqrt(MONTHS)))
        if excess.std(ddof=1) > 0 else np.nan
    )
    # reported alongside, and separately named, so the two are never conflated
    ret_vol_ratio = float(r.mean() * MONTHS / ann_vol) if ann_vol > 0 else np.nan

    downside = r[r < 0]
    sortino = (
        float(r.mean() * MONTHS / (downside.std(ddof=1) * np.sqrt(MONTHS)))
        if len(downside) > 1 and downside.std(ddof=1) > 0 else np.nan
    )

    curve = result["equity_curve"]
    drawdown = curve / curve.cummax() - 1

    return {
        "ann_return_%": round(100 * ann_return, 2),
        "ann_vol_%": round(100 * ann_vol, 2),
        "sharpe": round(sharpe, 3),
        "return_vol_ratio": round(ret_vol_ratio, 3),
        "rf_subtracted": rf is not None,
        "sortino": round(sortino, 3) if sortino == sortino else None,
        "max_drawdown_%": round(100 * float(drawdown.min()), 2),
        "calmar": round(ann_return / abs(float(drawdown.min())), 2)
        if float(drawdown.min()) < 0 else None,
        "hit_rate_%": round(100 * float((r > 0).mean()), 1),
        "ann_turnover": round(float(result["turnover"].mean() * MONTHS), 2),
        "ann_cost_%": round(100 * float(result["cost"].mean() * MONTHS), 3),
        "months": int(len(r)),
        "years": round(len(r) / MONTHS, 1),
        "start": r.index.min().date(),
        "end": r.index.max().date(),
    }


def summary_table(results: dict[str, pd.DataFrame], rf: pd.Series | None = None) -> pd.DataFrame:
    """Summary statistics for every strategy, aligned to a common sample.

    Alignment matters: comparing a strategy that starts in 1980 to a benchmark
    starting in 1993 compares different economic histories, not different signals.
    """
    common = None
    for res in results.values():
        idx = res["net_return"].dropna().index
        common = idx if common is None else common.intersection(idx)

    rows = []
    for name, res in results.items():
        aligned = res.loc[common].copy()
        aligned["equity_curve"] = (1 + aligned["net_return"]).cumprod()
        summary = performance_summary(aligned, rf=rf)
        if summary:
            rows.append({"strategy": name} | summary)

    df = pd.DataFrame(rows)
    return df.sort_values("sharpe", ascending=False, ignore_index=True) if not df.empty else df


# ---------------------------------------------------------------------------
# Significance
# ---------------------------------------------------------------------------

def _sharpe(x: np.ndarray) -> float:
    sd = x.std(ddof=1)
    return float(x.mean() * MONTHS / (sd * np.sqrt(MONTHS))) if sd > 0 else np.nan


def stationary_bootstrap_indices(
    n: int, block_length: float, rng: np.random.Generator
) -> np.ndarray:
    """Politis-Romano stationary bootstrap indices.

    Block lengths are geometric with mean ``block_length``, which keeps the
    resampled series stationary - unlike a fixed-block bootstrap, whose seams
    create artificial discontinuities.
    """
    p = 1.0 / max(block_length, 1.0)
    idx = np.empty(n, dtype=int)
    idx[0] = rng.integers(0, n)

    for i in range(1, n):
        if rng.random() < p:
            idx[i] = rng.integers(0, n)          # start a new block
        else:
            idx[i] = (idx[i - 1] + 1) % n        # continue the current one

    return idx


def bootstrap_sharpe_difference(
    strategy: pd.Series,
    benchmark: pd.Series,
    rf: pd.Series | None = None,
    n_iterations: int = 5000,
    block_length: float = 12.0,
    seed: int = 42,
) -> dict:
    """Test whether a Sharpe difference is distinguishable from luck.

    Resamples the PAIRED series (same indices for both) so the comparison holds the
    market path fixed and isolates the allocation decision.

    Returns the observed difference, a bootstrap confidence interval, and a
    two-sided p-value for H0: no difference.
    """
    idx = strategy.dropna().index.intersection(benchmark.dropna().index)
    s = to_excess(strategy.loc[idx], rf).to_numpy(dtype="float64")
    b = to_excess(benchmark.loc[idx], rf).to_numpy(dtype="float64")
    n = len(s)

    if n < 24:
        return {"error": "insufficient overlapping observations", "n": n}

    observed = _sharpe(s) - _sharpe(b)
    rng = np.random.default_rng(seed)

    diffs = np.empty(n_iterations)
    for i in range(n_iterations):
        take = stationary_bootstrap_indices(n, block_length, rng)
        diffs[i] = _sharpe(s[take]) - _sharpe(b[take])

    diffs = diffs[np.isfinite(diffs)]

    # The p-value is derived from the SAME percentile bootstrap distribution as the
    # confidence interval, so the two can never disagree. An earlier version
    # computed the interval by percentiles but the p-value from a mean-centred
    # distribution; bootstrap bias and skew then made cells appear to have an
    # interval excluding zero alongside p > 0.05, which is incoherent on its face.
    p_value = float(
        min(1.0, 2.0 * min((diffs <= 0).mean(), (diffs >= 0).mean()))
    )

    return {
        "sharpe_strategy": round(_sharpe(s), 3),
        "sharpe_benchmark": round(_sharpe(b), 3),
        "sharpe_difference": round(observed, 3),
        "ci_lower_95": round(float(np.percentile(diffs, 2.5)), 3),
        "ci_upper_95": round(float(np.percentile(diffs, 97.5)), 3),
        "p_value": round(p_value, 4),
        "significant_at_5pct": bool(p_value < 0.05),
        "n_months": int(n),
        "n_bootstrap": int(len(diffs)),
        "correlation": round(float(np.corrcoef(s, b)[0, 1]), 4),
        "rf_subtracted": rf is not None,
    }


def deflated_sharpe_ratio(
    returns: pd.Series,
    n_trials: int = 1,
    benchmark_sharpe: float = 0.0,
) -> dict:
    """Deflated Sharpe Ratio (Bailey & Lopez de Prado, 2014).

    Adjusts for (a) how many configurations were tried, and (b) skew and kurtosis,
    which make the usual Sharpe standard error optimistic for the fat-tailed,
    negatively-skewed returns typical of these strategies.

    ``n_trials`` should be honest: every state count, deadband and weighting scheme
    evaluated counts as a trial, whether or not it was reported.
    """
    r = returns.dropna().to_numpy(dtype="float64")
    n = len(r)
    if n < 24:
        return {"error": "insufficient observations", "n": n}

    sr = _sharpe(r) / np.sqrt(MONTHS)         # per-period Sharpe
    skew = float(sps.skew(r))
    kurt = float(sps.kurtosis(r, fisher=False))

    # expected maximum Sharpe from n_trials draws of pure noise
    if n_trials > 1:
        euler = 0.5772156649
        e_max = (1 - euler) * sps.norm.ppf(1 - 1.0 / n_trials) + \
            euler * sps.norm.ppf(1 - 1.0 / (n_trials * np.e))
        sr0 = benchmark_sharpe / np.sqrt(MONTHS) + e_max * _sr_std(sr, n, skew, kurt)
    else:
        sr0 = benchmark_sharpe / np.sqrt(MONTHS)

    denom = _sr_std(sr, n, skew, kurt)
    dsr = float(sps.norm.cdf((sr - sr0) * np.sqrt(n - 1) / denom)) if denom > 0 else np.nan

    return {
        "sharpe_annualized": round(sr * np.sqrt(MONTHS), 3),
        "skew": round(skew, 3),
        "excess_kurtosis": round(kurt - 3, 3),
        "n_trials_assumed": n_trials,
        "deflated_sharpe_prob": round(dsr, 4) if dsr == dsr else None,
        "interpretation": _dsr_verdict(dsr),
        "n_months": n,
    }


def _sr_std(sr: float, n: int, skew: float, kurt: float) -> float:
    """Standard error of the Sharpe ratio under non-normality."""
    var = 1 - skew * sr + ((kurt - 1) / 4.0) * sr**2
    return float(np.sqrt(max(var, 1e-12)))


def _dsr_verdict(dsr: float) -> str:
    if dsr != dsr:
        return "not computable"
    if dsr > 0.95:
        return "Sharpe survives deflation - unlikely to be selection luck"
    if dsr > 0.80:
        return "suggestive but not conclusive"
    return "NOT distinguishable from selection luck"


def newey_west_tstat(returns: pd.Series, lags: int = 6) -> float:
    """HAC t-statistic on the mean monthly return."""
    from ..evaluation.conditional import newey_west_se

    r = returns.dropna()
    se = newey_west_se(r, lags=lags)
    return float(r.mean() / se) if se and se > 0 else np.nan

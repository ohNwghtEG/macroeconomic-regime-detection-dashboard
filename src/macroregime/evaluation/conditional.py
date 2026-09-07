"""Regime-conditional asset performance, with statistically valid standard errors.

TWO THINGS THE ORIGINAL BRIEF WOULD HAVE GOT WRONG
--------------------------------------------------

**1. Overlapping forward returns inflate significance.**
Sampling 12-month forward returns monthly gives 11 months of overlap between
consecutive observations. They are heavily autocorrelated by construction, and a
naive t-statistic on them is inflated by roughly sqrt(12). "Gold returns 14% in
stagflation, p < 0.01" is an artefact of that overlap unless corrected. Newey-West
standard errors are used throughout, with the lag set from the horizon.

**2. Only returns were to be reported.**
The reason regimes matter for allocation is not mainly that mean returns differ -
it is that CORRELATIONS differ. Stock/bond correlation flipped sign in 2022 and
broke every risk model calibrated on the prior two decades. Conditional
correlation matrices are therefore a first-class output here, not an afterthought.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Newey-West
# ---------------------------------------------------------------------------

def newey_west_se(x: pd.Series | np.ndarray, lags: int | None = None) -> float:
    """Newey-West (HAC) standard error of the mean.

    Corrects for the autocorrelation induced by overlapping windows. With ``lags=0``
    this reduces to the usual ``s/sqrt(n)``.
    """
    v = np.asarray(pd.Series(x).dropna(), dtype="float64")
    n = len(v)
    if n < 3:
        return float("nan")

    if lags is None:
        lags = int(np.floor(4 * (n / 100.0) ** (2.0 / 9.0)))  # Newey-West rule of thumb
    lags = max(0, min(lags, n - 2))

    resid = v - v.mean()
    gamma0 = float(resid @ resid) / n
    variance = gamma0

    for lag in range(1, lags + 1):
        cov = float(resid[lag:] @ resid[:-lag]) / n
        weight = 1.0 - lag / (lags + 1.0)   # Bartlett kernel
        variance += 2.0 * weight * cov

    variance = max(variance, 1e-18)
    return float(np.sqrt(variance / n))


def mean_with_hac(x: pd.Series, horizon_months: int = 1) -> dict:
    """Mean, HAC standard error and t-statistic for one group of returns.

    The HAC lag is set to ``horizon - 1``, the exact overlap induced by sampling
    ``horizon``-month forward returns at monthly frequency.
    """
    v = pd.Series(x).dropna()
    if len(v) < 3:
        return {"mean": np.nan, "se": np.nan, "t_stat": np.nan, "n": len(v)}

    lags = max(0, horizon_months - 1)
    mean = float(v.mean())
    se = newey_west_se(v, lags=lags)
    naive_se = float(v.std(ddof=1) / np.sqrt(len(v)))

    return {
        "mean": mean,
        "se": se,
        "naive_se": naive_se,
        "t_stat": mean / se if se and se == se and se > 0 else np.nan,
        "naive_t_stat": mean / naive_se if naive_se > 0 else np.nan,
        "n": int(len(v)),
    }


# ---------------------------------------------------------------------------
# Conditional performance
# ---------------------------------------------------------------------------

def conditional_performance(
    returns: pd.DataFrame,
    regimes: pd.Series,
    horizon_months: int = 1,
    annualize: bool = True,
    min_obs: int = 12,
) -> pd.DataFrame:
    """Mean return, risk and HAC-corrected significance per asset x regime.

    ``horizon_months=1`` uses contemporaneous monthly returns (no overlap). Larger
    horizons expect ``returns`` to already hold forward cumulative returns, and set
    the HAC lag accordingly.
    """
    idx = returns.index.intersection(regimes.dropna().index)
    R = returns.loc[idx]
    S = regimes.loc[idx]

    rows = []
    for state in sorted(S.dropna().unique(), key=str):
        mask = S == state
        n_months = int(mask.sum())

        for asset in R.columns:
            block = R.loc[mask, asset].dropna()
            if len(block) < min_obs:
                continue

            stats = mean_with_hac(block, horizon_months=horizon_months)
            scale = 12 if (annualize and horizon_months == 1) else 1
            vol = float(block.std(ddof=1)) * (np.sqrt(12) if scale == 12 else 1)
            mean_ann = stats["mean"] * scale

            downside = block[block < 0]
            rows.append(
                {
                    "regime": state,
                    "asset": asset,
                    "mean_return_%": round(100 * mean_ann, 2),
                    "volatility_%": round(100 * vol, 2),
                    "sharpe": round(mean_ann / vol, 2) if vol > 0 else np.nan,
                    "hit_rate_%": round(100 * float((block > 0).mean()), 1),
                    "worst_month_%": round(100 * float(block.min()), 2),
                    "max_drawdown_%": round(100 * _max_drawdown(block), 2),
                    "t_stat_hac": round(stats["t_stat"], 2)
                    if stats["t_stat"] == stats["t_stat"] else None,
                    "t_stat_naive": round(stats["naive_t_stat"], 2)
                    if stats["naive_t_stat"] == stats["naive_t_stat"] else None,
                    "n_months": stats["n"],
                    "regime_months": n_months,
                    "downside_months": int(len(downside)),
                }
            )

    return pd.DataFrame(rows)


def _max_drawdown(returns: pd.Series) -> float:
    """Max drawdown of a return series (as a negative fraction).

    Computed on the returns as ordered, which for a regime subset means "drawdown
    across the months spent in this regime" rather than a contiguous market
    drawdown - a subtlety worth stating rather than glossing.
    """
    curve = (1 + returns.fillna(0)).cumprod()
    peak = curve.cummax()
    return float((curve / peak - 1).min())


def conditional_correlations(
    returns: pd.DataFrame,
    regimes: pd.Series,
    min_obs: int = 24,
) -> dict[str, pd.DataFrame]:
    """Correlation matrix per regime.

    THE MOST ALLOCATION-RELEVANT OUTPUT IN THE PROJECT. Regime models earn their
    keep less because expected returns shift than because the diversification you
    were relying on stops working in exactly the states where it is needed.
    """
    idx = returns.index.intersection(regimes.dropna().index)
    R, S = returns.loc[idx], regimes.loc[idx]

    out: dict[str, pd.DataFrame] = {"__unconditional__": R.corr().round(3)}
    for state in sorted(S.dropna().unique(), key=str):
        block = R.loc[S == state].dropna(how="all")
        if len(block) >= min_obs:
            out[str(state)] = block.corr().round(3)

    return out


def correlation_shift_table(
    returns: pd.DataFrame,
    regimes: pd.Series,
    pair: tuple[str, str] = ("equity", "bonds"),
    min_obs: int = 24,
) -> pd.DataFrame:
    """How one asset pair's correlation moves across regimes.

    Defaults to equity/bond - the correlation every 60/40 portfolio implicitly bets
    on, and the one that inverted in 2022.
    """
    a, b = pair
    if a not in returns.columns or b not in returns.columns:
        return pd.DataFrame()

    idx = returns.index.intersection(regimes.dropna().index)
    R, S = returns.loc[idx], regimes.loc[idx]

    rows = [
        {
            "regime": "ALL (unconditional)",
            "correlation": round(float(R[a].corr(R[b])), 3),
            "n_months": int(len(R.dropna(subset=[a, b]))),
        }
    ]
    for state in sorted(S.dropna().unique(), key=str):
        block = R.loc[S == state].dropna(subset=[a, b])
        if len(block) >= min_obs:
            rows.append(
                {
                    "regime": str(state),
                    "correlation": round(float(block[a].corr(block[b])), 3),
                    "n_months": int(len(block)),
                }
            )

    df = pd.DataFrame(rows)
    df.attrs["pair"] = f"{a} vs {b}"
    return df


def regime_return_matrix(
    conditional: pd.DataFrame,
    value_col: str = "mean_return_%",
) -> pd.DataFrame:
    """Pivot conditional performance into an asset x regime matrix for heatmaps."""
    if conditional.empty:
        return pd.DataFrame()
    return conditional.pivot(index="asset", columns="regime", values=value_col)

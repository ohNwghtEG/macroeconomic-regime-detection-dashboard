"""Expanding-window standardization.

WHY NOT A FULL-SAMPLE Z-SCORE
-----------------------------
``(x - x.mean()) / x.std()`` is the natural thing to write and it is a look-ahead
violation. The mean and standard deviation are computed over the *whole* sample,
so the z-score for 1985 embeds knowledge of the 2008 and 2020 distributions. It
leaks quietly - the series still looks perfectly plausible - and it inflates
backtests because the model implicitly knows how extreme a given reading will turn
out to be relative to history it has not lived through yet.

Every statistic here is computed from data up to and including ``t`` only.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def expanding_zscore(
    s: pd.Series,
    min_periods: int = 60,
    clip: float | None = 4.0,
) -> pd.Series:
    """Standardize using only history available at each point in time.

    Parameters
    ----------
    s
        Input series, indexed by as-of date.
    min_periods
        Observations required before emitting a value. Early z-scores computed off
        a handful of points are noise dressed as signal.
    clip
        Clip to +/- this many sigma. Guards against a single extreme (COVID) both
        dominating the scale and destabilising downstream covariance estimates.
        Clipping is applied to the OUTPUT, so it never feeds back into the mean or
        standard deviation.
    """
    x = s.astype("float64")
    mean = x.expanding(min_periods=min_periods).mean()
    std = x.expanding(min_periods=min_periods).std(ddof=1)

    z = (x - mean) / std.replace(0.0, np.nan)
    z = z.where(std.notna() & (std > 0))

    if clip is not None:
        z = z.clip(lower=-clip, upper=clip)

    return z.rename(s.name)


def expanding_zscore_frame(
    df: pd.DataFrame,
    min_periods: int = 60,
    clip: float | None = 4.0,
) -> pd.DataFrame:
    """Column-wise :func:`expanding_zscore`."""
    return pd.DataFrame(
        {c: expanding_zscore(df[c], min_periods=min_periods, clip=clip) for c in df.columns},
        index=df.index,
    )


def expanding_rank(s: pd.Series, min_periods: int = 60) -> pd.Series:
    """Expanding percentile rank in [0, 1].

    A distribution-free alternative to the z-score, useful in the sensitivity
    analysis: if conclusions hold under both, they are not an artefact of assuming
    normality in series that are visibly not normal.
    """
    x = s.astype("float64")
    out = np.full(len(x), np.nan)
    values = x.to_numpy()

    for i in range(len(values)):
        if i + 1 < min_periods or np.isnan(values[i]):
            continue
        hist = values[: i + 1]
        hist = hist[~np.isnan(hist)]
        if len(hist) < min_periods:
            continue
        out[i] = (hist <= values[i]).sum() / len(hist)

    return pd.Series(out, index=x.index, name=x.name)

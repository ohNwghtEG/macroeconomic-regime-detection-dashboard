"""Whipsaw suppression for rule-based classifiers.

A raw sign rule flips state every time a noisy series wobbles across zero. That is
not a regime change, it is measurement noise, and acting on it produces enormous
turnover for no economic reason.

Two devices are used together:

* **Deadband** - a zone around the threshold where the axis reads "unchanged", so a
  series has to move meaningfully before it counts as having flipped.
* **Confirmation** - a candidate new state must persist for N consecutive periods
  before it is adopted.

Both are strictly causal: the state at ``t`` depends only on observations up to
``t``. The obvious implementation of confirmation (checking whether the next N
months agree) reaches into the future and would silently reintroduce look-ahead.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def apply_deadband(s: pd.Series, threshold: float = 0.25) -> pd.Series:
    """Map a continuous axis to -1 / 0 / +1 with a neutral zone.

    Returns 0 inside +/- ``threshold``, which callers treat as "hold the previous
    reading" rather than as a state of its own.
    """
    out = pd.Series(0, index=s.index, dtype="int64")
    out[s > threshold] = 1
    out[s < -threshold] = -1
    return out.where(s.notna())


def hold_last_nonzero(s: pd.Series) -> pd.Series:
    """Carry the last decisive reading forward through neutral (0) stretches.

    Sitting in the deadband means "no new information", not "no regime", so the
    prior reading persists until the axis moves decisively the other way.
    """
    x = s.replace(0, np.nan)
    return x.ffill()


def apply_confirmation(states: pd.Series, confirm_periods: int = 2) -> pd.Series:
    """Adopt a new state only after it has persisted for ``confirm_periods``.

    Causal by construction: at each step only the current and previous observations
    are consulted, never the future.

    With ``confirm_periods=2`` a one-month blip is ignored; a genuine two-month
    move is adopted, with the switch dated at the *second* month - which is the
    honest date, since that is when a real-time observer would have been convinced.
    """
    if confirm_periods <= 1:
        return states.copy()

    values = states.to_numpy(dtype=object)
    out = np.empty(len(values), dtype=object)

    current = None
    candidate = None
    run = 0

    for i, v in enumerate(values):
        if v is None or (isinstance(v, float) and np.isnan(v)):
            out[i] = current
            continue

        if current is None:
            current = v          # bootstrap: adopt the first observed state
            candidate, run = v, 0
        elif v == current:
            candidate, run = v, 0
        elif v == candidate:
            run += 1
            if run >= confirm_periods - 1:
                current = v
                run = 0
        else:
            candidate, run = v, 0

        out[i] = current

    return pd.Series(out, index=states.index, name=states.name)


def regime_episodes(states: pd.Series) -> pd.DataFrame:
    """Collapse a state series into contiguous episodes.

    The honest unit of sample size. A monthly series from 1966 has ~700 rows but
    only a few dozen episodes, and it is the episode count that governs how much
    can actually be inferred. Reporting n=700 would badly overstate the evidence.
    """
    s = states.dropna()
    if s.empty:
        return pd.DataFrame(columns=["state", "start", "end", "months"])

    changed = s != s.shift(1)
    group = changed.cumsum()

    rows = []
    for _, block in s.groupby(group):
        rows.append(
            {
                "state": block.iloc[0],
                "start": block.index[0],
                "end": block.index[-1],
                "months": len(block),
            }
        )

    return pd.DataFrame(rows)


def episode_summary(states: pd.Series) -> pd.DataFrame:
    """Per-state episode counts and durations - the real sample size."""
    ep = regime_episodes(states)
    if ep.empty:
        return pd.DataFrame()

    return (
        ep.groupby("state")["months"]
        .agg(episodes="count", total_months="sum", mean_months="mean",
             median_months="median", max_months="max")
        .round(1)
        .sort_values("total_months", ascending=False)
    )

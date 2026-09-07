"""Validation against NBER recession dates.

The original brief never asked whether the regimes were CORRECT - it went straight
from classification to allocation. ``USREC`` provides an external binary benchmark,
so accuracy is measurable rather than assumed.

Three things are measured:

1. **Classification quality** - precision, recall, F1 on recession months.
2. **Lead/lag** - does the signal fire before or after the NBER peak? A model that
   only recognises a recession once it is over is useless regardless of its F1.
3. **False positives** - the growth scares that were not recessions (1998, 2011,
   2015-16, 2022). These are the interesting failures, because a tactical overlay
   that de-risks four times per cycle bleeds return in the seven years out of ten
   when nothing is wrong.

A note on the benchmark itself: NBER dates are announced 6-18 months after the
fact and are themselves revised. They are the right yardstick for *classification*
accuracy, and must never be used as a model input - ``registry.SeriesSpec`` blocks
that structurally.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def align_recession_flag(usrec: pd.Series, index: pd.DatetimeIndex) -> pd.Series:
    """Align the NBER monthly indicator to an as-of month-end index."""
    s = usrec.dropna().sort_index()
    s.index = pd.DatetimeIndex(s.index) + pd.offsets.MonthEnd(0)
    return s.reindex(index).ffill().fillna(0).astype(int).rename("nber_recession")


def recession_classification_report(
    predicted_recession: pd.Series,
    usrec: pd.Series,
) -> dict:
    """Precision / recall / F1 for a binary recession call.

    ``predicted_recession`` must be a boolean or 0/1 series on the as-of index.
    """
    truth = align_recession_flag(usrec, predicted_recession.index)
    pred = predicted_recession.reindex(truth.index).fillna(0).astype(int)

    mask = pred.notna() & truth.notna()
    pred, truth = pred[mask], truth[mask]

    tp = int(((pred == 1) & (truth == 1)).sum())
    fp = int(((pred == 1) & (truth == 0)).sum())
    fn = int(((pred == 0) & (truth == 1)).sum())
    tn = int(((pred == 0) & (truth == 0)).sum())

    precision = tp / (tp + fp) if (tp + fp) else float("nan")
    recall = tp / (tp + fn) if (tp + fn) else float("nan")
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision and recall and not np.isnan(precision) and not np.isnan(recall)
        else float("nan")
    )

    return {
        "true_positive": tp,
        "false_positive": fp,
        "false_negative": fn,
        "true_negative": tn,
        "precision": round(precision, 3) if precision == precision else None,
        "recall": round(recall, 3) if recall == recall else None,
        "f1": round(f1, 3) if f1 == f1 else None,
        "accuracy": round((tp + tn) / len(pred), 3) if len(pred) else None,
        "months_evaluated": int(len(pred)),
        "recession_months_in_sample": int(truth.sum()),
    }


def nber_episodes(usrec: pd.Series) -> pd.DataFrame:
    """Extract NBER recession episodes as (peak, trough) pairs."""
    s = usrec.dropna().sort_index()
    s.index = pd.DatetimeIndex(s.index) + pd.offsets.MonthEnd(0)
    s = s.astype(int)

    changes = s.diff().fillna(0)
    starts = s.index[changes == 1]
    ends = s.index[changes == -1]

    rows = []
    for start in starts:
        later = ends[ends > start]
        end = later[0] if len(later) else s.index[-1]
        rows.append({"peak": start, "trough": end, "months": len(s.loc[start:end]) - 1})

    return pd.DataFrame(rows)


def lead_lag_analysis(
    predicted_recession: pd.Series,
    usrec: pd.Series,
    search_window_months: int = 24,
) -> pd.DataFrame:
    """For each NBER recession, how many months early or late did the signal fire?

    Negative lead = fired BEFORE the NBER peak (leading, useful).
    Positive lead = fired AFTER the peak (lagging, less useful).
    NaN = never fired within the window (missed entirely).

    This matters more than F1. A classifier can score well by recognising
    recessions in their final month, which is worth nothing to an allocator.
    """
    pred = predicted_recession.astype("float64")
    episodes = nber_episodes(usrec)

    # Restrict to recessions that fall inside the model's own sample. USREC starts
    # in 1854; scoring a model that begins in 1966 against 34 recessions it could
    # never have seen would understate detection rates by a factor of four.
    if not episodes.empty and len(pred.index):
        lo, hi = pred.index.min(), pred.index.max()
        episodes = episodes[(episodes["peak"] >= lo) & (episodes["peak"] <= hi)]
        episodes = episodes.reset_index(drop=True)

    rows = []

    for _, ep in episodes.iterrows():
        peak, trough = ep["peak"], ep["trough"]
        window_start = peak - pd.DateOffset(months=search_window_months)

        window = pred.loc[
            (pred.index >= window_start) & (pred.index <= trough)
        ]
        fired = window[window > 0]

        if fired.empty:
            lead = np.nan
            first = pd.NaT
        else:
            first = fired.index[0]
            lead = (first.year - peak.year) * 12 + (first.month - peak.month)

        rows.append(
            {
                "nber_peak": peak.date(),
                "nber_trough": trough.date(),
                "recession_months": int(ep["months"]),
                "signal_first_fired": first.date() if pd.notna(first) else None,
                "lead_months": lead,
                "detected": bool(not fired.empty),
            }
        )

    return pd.DataFrame(rows)


def false_positive_episodes(
    predicted_recession: pd.Series,
    usrec: pd.Series,
    min_months: int = 2,
) -> pd.DataFrame:
    """Contiguous stretches where the model called recession and NBER did not.

    These are the growth scares - 1998, 2011, 2015-16, 2022. Reporting them
    explicitly is the honest counterweight to a good F1 score.
    """
    truth = align_recession_flag(usrec, predicted_recession.index)
    pred = predicted_recession.reindex(truth.index).fillna(0).astype(int)

    fp = ((pred == 1) & (truth == 0)).astype(int)
    if fp.sum() == 0:
        return pd.DataFrame(columns=["start", "end", "months"])

    group = (fp.diff() != 0).cumsum()
    rows = []
    for _, block in fp.groupby(group):
        if block.iloc[0] == 1 and len(block) >= min_months:
            rows.append(
                {
                    "start": block.index[0].date(),
                    "end": block.index[-1].date(),
                    "months": len(block),
                }
            )

    return pd.DataFrame(rows)


def compare_to_sahm(
    predicted_recession: pd.Series,
    sahm: pd.Series,
    usrec: pd.Series,
) -> pd.DataFrame:
    """Benchmark the model against the real-time Sahm rule.

    Sahm is a single published line of arithmetic. If an HMM over twenty-odd macro
    series cannot beat it, that is a genuine and reportable negative result - and a
    far more informative one than another flattering table.
    """
    idx = predicted_recession.index
    sahm_aligned = sahm.dropna().copy()
    sahm_aligned.index = pd.DatetimeIndex(sahm_aligned.index) + pd.offsets.MonthEnd(0)
    sahm_binary = (sahm_aligned.reindex(idx).ffill() >= 0.5).astype(int)

    rows = []
    for name, pred in [("model", predicted_recession), ("sahm_rule", sahm_binary)]:
        rep = recession_classification_report(pred, usrec)
        ll = lead_lag_analysis(pred, usrec)
        rep["mean_lead_months"] = (
            round(float(ll["lead_months"].mean(skipna=True)), 1)
            if not ll.empty else None
        )
        rep["recessions_detected"] = int(ll["detected"].sum()) if not ll.empty else 0
        rep["recessions_total"] = int(len(ll))
        rep["model"] = name
        rows.append(rep)

    cols = [
        "model", "precision", "recall", "f1", "mean_lead_months",
        "recessions_detected", "recessions_total", "false_positive",
    ]
    return pd.DataFrame(rows)[cols]

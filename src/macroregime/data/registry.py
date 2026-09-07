"""Series registry and point-in-time alignment.

THE CENTRAL IDEA
----------------
FRED indexes an observation by its *reference period*, and serves the latest
*revised* value. Neither is what a real-time decision maker actually had.

For every observation we therefore compute::

    release_date = period_end + publication_lag_days

and when building the feature row for as-of date ``T`` we take, for each series,
the most recent observation whose ``release_date <= T``.

This turns a naive (and look-ahead-contaminated) index join into a point-in-time
join. ``tests/test_no_lookahead.py`` asserts the property directly.

Note on transforms: every transform here is strictly backward-looking (YoY,
n-period difference, n-week percent change). That matters, because transforms are
applied in *period space* before the release-date join. A centred or
forward-looking transform would smuggle future data in ahead of the join and
silently break the guarantee.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import pandas as pd

from ..config import series_config

Frequency = Literal["daily", "weekly", "monthly", "quarterly"]


@dataclass(frozen=True)
class SeriesSpec:
    """Metadata for one economic series."""

    id: str
    name: str
    frequency: Frequency
    publication_lag_days: int
    transform: str = "level"
    dimension: str = "unknown"
    note: str = ""
    usage: str = "feature"

    @property
    def is_feature(self) -> bool:
        """Whether this series may enter the feature matrix.

        NBER recession dates (``USREC``) are announced 6-18 months after the fact.
        Using them as a feature would be the most extreme look-ahead possible, so
        validation-only series are hard-blocked here rather than by convention.
        """
        return self.usage == "feature"


def load_specs() -> dict[str, SeriesSpec]:
    """Load every registered series (features + validation) keyed by id."""
    cfg = series_config()
    specs: dict[str, SeriesSpec] = {}

    for entry in cfg.get("macro_series", []):
        specs[entry["id"]] = SeriesSpec(
            id=entry["id"],
            name=entry["name"],
            frequency=entry["frequency"],
            publication_lag_days=int(entry["publication_lag_days"]),
            transform=entry.get("transform", "level"),
            dimension=entry.get("dimension", "unknown"),
            note=entry.get("note", ""),
            usage="feature",
        )

    for entry in cfg.get("validation_series", []):
        specs[entry["id"]] = SeriesSpec(
            id=entry["id"],
            name=entry["name"],
            frequency=entry["frequency"],
            publication_lag_days=int(entry["publication_lag_days"]),
            transform=entry.get("transform", "level"),
            dimension="validation",
            note=entry.get("note", ""),
            usage=entry.get("usage", "validation_only"),
        )

    return specs


def feature_specs() -> dict[str, SeriesSpec]:
    """Only the series permitted to enter the feature matrix."""
    return {k: v for k, v in load_specs().items() if v.is_feature}


def all_series_ids() -> list[str]:
    return list(load_specs().keys())


# ---------------------------------------------------------------------------
# Period end / release date
# ---------------------------------------------------------------------------

def period_end(index: pd.DatetimeIndex, frequency: str) -> pd.DatetimeIndex:
    """Map a FRED observation index to the END of its reference period.

    FRED stamps monthly series with the FIRST day of the month: ``CPIAUCSL`` at
    2008-03-01 is March 2008 CPI, whose reference period ends 2008-03-31. Getting
    this wrong shifts every release date by up to a month.

    Weekly series (``ICSA``) are stamped at the week-ending date and daily series
    at the day itself, so both are already at period end.
    """
    idx = pd.DatetimeIndex(index)
    if frequency == "monthly":
        return idx + pd.offsets.MonthEnd(0)
    if frequency == "quarterly":
        return idx + pd.offsets.QuarterEnd(0)
    return idx


def release_dates(index: pd.DatetimeIndex, spec: SeriesSpec) -> pd.DatetimeIndex:
    """Date on which each observation first became publicly available."""
    ends = period_end(index, spec.frequency)
    return ends + pd.Timedelta(days=spec.publication_lag_days)


# ---------------------------------------------------------------------------
# Transforms (all strictly backward-looking)
# ---------------------------------------------------------------------------

_PERIODS_PER_YEAR = {"daily": 252, "weekly": 52, "monthly": 12, "quarterly": 4}


def apply_transform(s: pd.Series, spec: SeriesSpec) -> pd.Series:
    """Apply the configured stationarity transform, in period space.

    Raw levels (unemployment rate, credit spread) are non-stationary: a Gaussian
    HMM fed those will cheerfully learn "the 1980s" as a state - a time period
    rather than an economic condition.
    """
    t = spec.transform

    if t == "level":
        out = s
    elif t == "yoy":
        out = s.pct_change(_PERIODS_PER_YEAR[spec.frequency]) * 100.0
    elif t == "diff3":
        out = s.diff(3)
    elif t == "diff12":
        out = s.diff(12)
    elif t == "pct13w":
        out = s.pct_change(13) * 100.0
    else:
        raise ValueError(f"Unknown transform {t!r} for series {spec.id}")

    return out.rename(spec.id)


# ---------------------------------------------------------------------------
# Point-in-time join
# ---------------------------------------------------------------------------

def to_point_in_time(s: pd.Series, spec: SeriesSpec) -> pd.DataFrame:
    """Convert a raw series into ``(release_date, period_end, value)`` rows."""
    empty = pd.DataFrame(
        {
            "release_date": pd.Series(dtype="datetime64[ns]"),
            "period_end": pd.Series(dtype="datetime64[ns]"),
            "value": pd.Series(dtype="float64"),
        }
    )

    s = s.dropna().sort_index()
    if s.empty:
        return empty

    transformed = apply_transform(s, spec).dropna()
    if transformed.empty:
        return empty

    idx = pd.DatetimeIndex(transformed.index)
    return pd.DataFrame(
        {
            "release_date": release_dates(idx, spec),
            "period_end": period_end(idx, spec.frequency),
            "value": transformed.to_numpy(dtype="float64"),
        }
    ).sort_values("release_date", ignore_index=True)


def asof_feature_matrix(
    raw: dict[str, pd.Series],
    specs: dict[str, SeriesSpec],
    asof_index: pd.DatetimeIndex,
) -> pd.DataFrame:
    """Build a feature matrix that is correct as of each date in ``asof_index``.

    Row ``T`` contains, for each series, the latest value **released on or before
    T**. This is the only function that should ever be used to construct model
    inputs.
    """
    asof = pd.DatetimeIndex(asof_index).sort_values()
    base = pd.DataFrame({"asof": asof})
    out = base.copy()

    for sid, spec in specs.items():
        if sid not in raw:
            continue

        pit = to_point_in_time(raw[sid], spec)
        if pit.empty:
            out[sid] = float("nan")
            continue

        merged = pd.merge_asof(
            base,
            pit[["release_date", "value"]],
            left_on="asof",
            right_on="release_date",
            direction="backward",  # latest release at or before asof
        )
        out[sid] = merged["value"].to_numpy()

    return out.set_index("asof")


def staleness_report(
    raw: dict[str, pd.Series],
    specs: dict[str, SeriesSpec],
    asof: pd.Timestamp | None = None,
) -> pd.DataFrame:
    """Diagnostic: how stale is each series at a month-end decision point?

    Answers "at the moment I trade, how old is the newest CPI print I have?" -
    which is the practical meaning of the publication lag.
    """
    asof = pd.Timestamp(asof) if asof is not None else pd.Timestamp.today().normalize()
    rows = []

    for sid, spec in specs.items():
        if sid not in raw or raw[sid].dropna().empty:
            continue

        s = raw[sid].dropna()
        idx = pd.DatetimeIndex(s.index)
        pit = to_point_in_time(s, spec)
        available = pit[pit["release_date"] <= asof]
        newest_period = available["period_end"].max() if not available.empty else pd.NaT
        staleness = (
            (asof - newest_period).days if isinstance(newest_period, pd.Timestamp) else None
        )

        rows.append(
            {
                "series": sid,
                "name": spec.name,
                "dimension": spec.dimension,
                "frequency": spec.frequency,
                "transform": spec.transform,
                "lag_days": spec.publication_lag_days,
                "first_obs": idx.min().date(),
                "last_obs": idx.max().date(),
                "n_obs": len(s),
                "newest_available_period": (
                    newest_period.date() if isinstance(newest_period, pd.Timestamp) else None
                ),
                "staleness_days": staleness,
            }
        )

    return pd.DataFrame(rows).sort_values(["dimension", "series"], ignore_index=True)

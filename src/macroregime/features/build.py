"""Assemble the model-ready feature matrix.

Pipeline order matters and is deliberate::

    raw FRED series
      -> derived series      (computed in period space, lag = MAX of inputs)
      -> point-in-time join  (release_date <= asof)
      -> expanding z-score   (statistics from data <= t only)
      -> composites          (equal-weight mean of z-scores, signs aligned)

Standardizing *after* the point-in-time join is essential: the z-score must be
computed on the data as it was actually seen, not on the revised series.
"""

from __future__ import annotations

import logging

import pandas as pd

from ..config import model_config
from ..data.registry import SeriesSpec, asof_feature_matrix, feature_specs
from .standardize import expanding_zscore_frame

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Derived series
# ---------------------------------------------------------------------------

def build_raw_with_derived(
    raw: dict[str, pd.Series],
    specs: dict[str, SeriesSpec] | None = None,
) -> tuple[dict[str, pd.Series], dict[str, SeriesSpec]]:
    """Add configured derived series to the raw pool.

    A derived series is only as timely as its SLOWEST input, so its publication lag
    is the max over inputs. Taking the min (or the lag of the faster series) is a
    quiet way for look-ahead to creep back in after the join is otherwise correct.
    """
    specs = dict(specs if specs is not None else feature_specs())
    raw = dict(raw)
    cfg = model_config().get("derived_series", {})

    # --- Baa - Aaa default spread: monthly, back to 1919 ---
    if "BAA_AAA_SPREAD" in cfg and {"BAA", "AAA"} <= raw.keys():
        baa, aaa = raw["BAA"].dropna(), raw["AAA"].dropna()
        common = baa.index.intersection(aaa.index)
        if len(common):
            spread = (baa.loc[common] - aaa.loc[common]).sort_index()
            spread.name = "BAA_AAA_SPREAD"
            raw["BAA_AAA_SPREAD"] = spread
            lag = max(specs["BAA"].publication_lag_days, specs["AAA"].publication_lag_days)
            specs["BAA_AAA_SPREAD"] = SeriesSpec(
                id="BAA_AAA_SPREAD",
                name="Baa-Aaa Default Spread",
                frequency="monthly",
                publication_lag_days=lag,
                transform="level",
                dimension="financial",
                note="Derived: BAA - AAA. Longest-history credit stress measure (1919).",
            )
            log.info("derived BAA_AAA_SPREAD: %d obs from %s",
                     len(spread), spread.index.min().date())

    # --- Real fed funds rate: policy rate net of realised inflation ---
    if "REAL_FED_FUNDS" in cfg and {"FEDFUNDS", "CPIAUCSL"} <= raw.keys():
        ff = raw["FEDFUNDS"].dropna()
        cpi_yoy = (raw["CPIAUCSL"].dropna().pct_change(12) * 100.0).dropna()
        common = ff.index.intersection(cpi_yoy.index)
        if len(common):
            real = (ff.loc[common] - cpi_yoy.loc[common]).sort_index()
            real.name = "REAL_FED_FUNDS"
            raw["REAL_FED_FUNDS"] = real
            lag = max(
                specs["FEDFUNDS"].publication_lag_days,
                specs["CPIAUCSL"].publication_lag_days,
            )
            specs["REAL_FED_FUNDS"] = SeriesSpec(
                id="REAL_FED_FUNDS",
                name="Real Federal Funds Rate",
                frequency="monthly",
                publication_lag_days=lag,   # gated by CPI, the slower input
                transform="level",
                dimension="policy",
                note="Derived: FEDFUNDS - CPI YoY. Lag inherited from CPI.",
            )
            log.info("derived REAL_FED_FUNDS: %d obs", len(real))

    return raw, specs


# ---------------------------------------------------------------------------
# Composites
# ---------------------------------------------------------------------------

def build_composites(z: pd.DataFrame) -> pd.DataFrame:
    """Equal-weight composites of standardized features, with signs aligned.

    Sign alignment is the fiddly part. Every composite must read "higher = more of
    the thing it is named after":

    * growth    - rising unemployment and claims mean LESS growth, so both invert.
    * financial - NFCI, credit spreads and VIX rise when conditions TIGHTEN, so all
      invert; yield-curve slopes stay positive, since a steep curve is the easy
      state and inversion is the warning.

    Getting a sign backwards produces a model that is confidently wrong rather than
    obviously broken, which is much harder to notice later.
    """
    cfg = model_config()["composites"]
    out = pd.DataFrame(index=z.index)

    for name, spec in cfg.items():
        inputs = [c for c in spec["inputs"] if c in z.columns]
        missing = [c for c in spec["inputs"] if c not in z.columns]
        if missing:
            log.warning("composite %s missing: %s", name, ", ".join(missing))
        if not inputs:
            log.error("composite %s has no available inputs - skipping", name)
            continue

        invert = set(spec.get("invert") or [])
        block = z[inputs].copy()
        for col in inputs:
            if col in invert:
                block[col] = -block[col]

        # mean over available inputs so the composite survives ragged start dates
        out[f"{name}_composite"] = block.mean(axis=1, skipna=True)
        out[f"{name}_n_inputs"] = block.notna().sum(axis=1)

    # momentum of the growth composite drives the business-cycle phase model
    if "growth_composite" in out.columns:
        out["growth_composite_3m_change"] = out["growth_composite"].diff(3)
    if "inflation_composite" in out.columns:
        out["inflation_composite_3m_change"] = out["inflation_composite"].diff(3)

    return out


# ---------------------------------------------------------------------------
# Top level
# ---------------------------------------------------------------------------

def build_features(
    raw: dict[str, pd.Series],
    asof_index: pd.DatetimeIndex | None = None,
    start: str = "1962-01-31",
    end: str | None = None,
) -> dict[str, pd.DataFrame]:
    """Run the full feature pipeline.

    Returns a dict with ``levels`` (point-in-time raw features), ``z`` (expanding
    z-scores) and ``composites``.
    """
    cfg = model_config()["standardization"]

    if asof_index is None:
        end_ts = pd.Timestamp(end) if end else pd.Timestamp.today().normalize()
        asof_index = pd.date_range(start, end_ts, freq="ME")

    raw2, specs2 = build_raw_with_derived(raw)

    levels = asof_feature_matrix(raw2, specs2, asof_index)
    z = expanding_zscore_frame(
        levels,
        min_periods=int(cfg.get("min_periods", 60)),
        clip=cfg.get("clip_sigma"),
    )
    composites = build_composites(z)

    log.info("features built: levels=%s z=%s composites=%s",
             levels.shape, z.shape, composites.shape)

    return {"levels": levels, "z": z, "composites": composites}

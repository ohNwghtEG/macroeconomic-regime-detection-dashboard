"""FRED data acquisition.

Fetch policy: use the parquet cache when present, hit the API only when the cache
is missing or ``refresh=True``. Every successful fetch is cached immediately, so a
partial run never has to be repeated from scratch and the repo stays runnable
offline.
"""

from __future__ import annotations

import logging
import time

import pandas as pd

from ..config import fred_api_key
from . import cache
from .registry import SeriesSpec, load_specs

log = logging.getLogger(__name__)

_RETRIES = 3
_BACKOFF_SECONDS = 2.0


class FredUnavailableError(RuntimeError):
    """Raised when data is neither cached nor fetchable."""


def _client():
    """Construct a fredapi client, or None when no key is configured."""
    key = fred_api_key()
    if not key:
        return None
    try:
        from fredapi import Fred
    except ImportError as exc:  # pragma: no cover
        raise FredUnavailableError("fredapi is not installed") from exc
    return Fred(api_key=key)


def fetch_series(series_id: str, refresh: bool = False) -> pd.Series:
    """Fetch one FRED series, preferring the cache.

    Raises FredUnavailableError only when the series is both uncached and
    unfetchable - so an offline clone with a populated cache never fails.
    """
    if not refresh:
        cached = cache.load_series(series_id)
        if cached is not None:
            log.debug("cache hit: %s (%d obs)", series_id, len(cached))
            return cached

    client = _client()
    if client is None:
        cached = cache.load_series(series_id)
        if cached is not None:
            return cached
        raise FredUnavailableError(
            f"{series_id} is not cached and no FRED_API_KEY is set. "
            "Add a key to .env, or restore the committed data/cache."
        )

    last_error: Exception | None = None
    for attempt in range(1, _RETRIES + 1):
        try:
            s = client.get_series(series_id)
            s = pd.Series(s).dropna()
            s.index = pd.DatetimeIndex(s.index)
            s = s.sort_index()
            s.name = series_id
            cache.save_series(series_id, s)
            log.info("fetched %s: %d obs (%s to %s)",
                     series_id, len(s), s.index.min().date(), s.index.max().date())
            return s
        except Exception as exc:  # noqa: BLE001 - fredapi raises bare exceptions
            last_error = exc
            log.warning("fetch %s failed (attempt %d/%d): %s",
                        series_id, attempt, _RETRIES, exc)
            if attempt < _RETRIES:
                time.sleep(_BACKOFF_SECONDS * attempt)

    stale = cache.load_series(series_id)
    if stale is not None:
        log.warning("using stale cache for %s after fetch failure", series_id)
        return stale

    raise FredUnavailableError(f"Could not fetch {series_id}: {last_error}")


def fetch_all(
    specs: dict[str, SeriesSpec] | None = None,
    refresh: bool = False,
) -> dict[str, pd.Series]:
    """Fetch every registered series. Failures are logged, not fatal.

    A single unavailable series (FRED does discontinue them - see ISM and the LBMA
    gold fix) should degrade the feature set, not kill the pipeline.
    """
    specs = specs if specs is not None else load_specs()
    out: dict[str, pd.Series] = {}
    failures: list[str] = []

    for sid in specs:
        try:
            out[sid] = fetch_series(sid, refresh=refresh)
        except FredUnavailableError as exc:
            log.error("skipping %s: %s", sid, exc)
            failures.append(sid)

    if failures:
        log.warning("%d series unavailable: %s", len(failures), ", ".join(failures))

    return out


def availability_check(refresh: bool = True) -> pd.DataFrame:
    """Probe every registered series and report what FRED actually serves today.

    Run this before trusting the series registry: FRED removes series (ISM PMI in
    2016, the LBMA gold fix more recently) and a stale registry fails confusingly.
    """
    specs = load_specs()
    rows = []

    for sid, spec in specs.items():
        row = {
            "series": sid,
            "name": spec.name,
            "dimension": spec.dimension,
            "expected_freq": spec.frequency,
            "usage": spec.usage,
        }
        try:
            s = fetch_series(sid, refresh=refresh)
            row |= {
                "status": "OK",
                "n_obs": len(s),
                "first": s.index.min().date(),
                "last": s.index.max().date(),
            }
        except Exception as exc:  # noqa: BLE001
            row |= {
                "status": "FAILED",
                "n_obs": 0,
                "first": None,
                "last": None,
                "error": str(exc)[:120],
            }
        rows.append(row)

    return pd.DataFrame(rows)

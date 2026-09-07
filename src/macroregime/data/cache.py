"""Parquet cache.

Why this exists: a reviewer who clones the repo and cannot run it will close the
tab. Every fetched series is cached to parquet and the cache is committed, so
``python -m macroregime.run`` works offline with **no API key**. A key is needed
only to refresh.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from ..config import CACHE_DIR

MANIFEST = "manifest.json"


def _cache_path(name: str, subdir: str) -> Path:
    d = CACHE_DIR / subdir
    d.mkdir(parents=True, exist_ok=True)
    return d / f"{name}.parquet"


def save_series(name: str, s: pd.Series, subdir: str = "fred") -> Path:
    """Persist one series to parquet."""
    path = _cache_path(name, subdir)
    df = pd.DataFrame({"date": pd.DatetimeIndex(s.index), "value": s.to_numpy()})
    df.to_parquet(path, index=False)
    _touch_manifest(name, subdir, len(df))
    return path


def load_series(name: str, subdir: str = "fred") -> pd.Series | None:
    """Load one cached series, or None if not cached."""
    path = _cache_path(name, subdir)
    if not path.exists():
        return None
    df = pd.read_parquet(path)
    return pd.Series(
        df["value"].to_numpy(),
        index=pd.DatetimeIndex(df["date"]),
        name=name,
    ).sort_index()


def is_cached(name: str, subdir: str = "fred") -> bool:
    return _cache_path(name, subdir).exists()


def save_frame(name: str, df: pd.DataFrame, subdir: str = "derived") -> Path:
    path = _cache_path(name, subdir)
    df.to_parquet(path)
    _touch_manifest(name, subdir, len(df))
    return path


def load_frame(name: str, subdir: str = "derived") -> pd.DataFrame | None:
    path = _cache_path(name, subdir)
    return pd.read_parquet(path) if path.exists() else None


def _touch_manifest(name: str, subdir: str, n_rows: int) -> None:
    """Record what was cached and when, so data provenance is auditable."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    path = CACHE_DIR / MANIFEST

    manifest: dict = {}
    if path.exists():
        try:
            manifest = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            manifest = {}

    manifest.setdefault(subdir, {})[name] = {
        "rows": int(n_rows),
        "fetched_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")


def manifest() -> dict:
    path = CACHE_DIR / MANIFEST
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}

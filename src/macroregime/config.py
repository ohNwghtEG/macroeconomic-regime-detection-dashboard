"""Configuration loading.

All tunable parameters live in YAML under ``config/`` so that the sensitivity
analyses can sweep them without touching code.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv

# repo root = three levels up from this file (src/macroregime/config.py)
ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = ROOT / "config"
DATA_DIR = ROOT / "data"
CACHE_DIR = DATA_DIR / "cache"
REPORT_DIR = ROOT / "reports"


def _load_yaml(name: str) -> dict[str, Any]:
    path = CONFIG_DIR / name
    if not path.exists():
        raise FileNotFoundError(f"Missing config file: {path}")
    with path.open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


@lru_cache(maxsize=1)
def series_config() -> dict[str, Any]:
    return _load_yaml("series.yaml")


@lru_cache(maxsize=1)
def asset_config() -> dict[str, Any]:
    return _load_yaml("assets.yaml")


@lru_cache(maxsize=1)
def model_config() -> dict[str, Any]:
    return _load_yaml("model.yaml")


def fred_api_key() -> str | None:
    """Return the FRED API key, or None if unset.

    None is not an error: the project is designed to run fully offline from the
    committed parquet cache. A key is only needed to refresh data.
    """
    load_dotenv(ROOT / ".env")
    key = os.getenv("FRED_API_KEY")
    return key.strip() if key else None


def ensure_dirs() -> None:
    for d in (DATA_DIR, CACHE_DIR, REPORT_DIR):
        d.mkdir(parents=True, exist_ok=True)

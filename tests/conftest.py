"""Shared fixtures and collection hooks.

The collected test count is recorded here at collection time and asserted against
the count stated in README.md by ``test_units_and_tables.py``. An earlier version obtained the
count by shelling out to ``pytest --collect-only`` from inside a collected test,
which made the suite depend on its own invocation environment and was slow and
brittle under CI. A collection hook gives the same guarantee for free.
"""

from __future__ import annotations

from pathlib import Path

COUNT_FILE = Path(__file__).parent / ".collected_count"


def pytest_collection_finish(session):
    """Record how many tests were collected, for the README-count assertion."""
    try:
        COUNT_FILE.write_text(str(len(session.items)), encoding="utf-8")
    except OSError:  # pragma: no cover - read-only checkouts
        pass

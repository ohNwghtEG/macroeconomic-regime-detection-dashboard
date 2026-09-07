"""Rule-based regime classifiers.

Two DIFFERENT frameworks, which the original project brief ran together:

**A. Growth x Inflation quadrants** (Bridgewater "four quadrants" / All Weather).
A state space over the *direction* of growth and inflation surprises. Asset logic
falls straight out: equities like growth up, gold likes inflation up, nominal bonds
like both down.

**B. Business-cycle phases** (expansion -> slowdown -> recession -> recovery).
A sequence of phases defined by growth *level* combined with growth *momentum*,
tracing the familiar clockwise cycle.

These are related but genuinely distinct objects, and comparing them is one of the
more informative outputs of the project.

An honest caveat on (A): Bridgewater's real framing is growth and inflation
relative to what markets have *discounted*, which is not recoverable from FRED.
Realized momentum against trend is a proxy for that, and the difference is stated
plainly rather than glossed over.
"""

from __future__ import annotations

import pandas as pd

from ..config import model_config
from .hysteresis import apply_confirmation, apply_deadband, hold_last_nonzero

# ---------------------------------------------------------------------------
# Canonical state names
# ---------------------------------------------------------------------------

QUADRANT_STATES = ["goldilocks", "reflation", "stagflation", "deflation"]
CYCLE_STATES = ["expansion", "slowdown", "recession", "recovery"]

QUADRANT_DESCRIPTIONS = {
    "goldilocks": "Growth rising, inflation falling - the best case for equities",
    "reflation": "Growth rising, inflation rising - commodities and equities",
    "stagflation": "Growth falling, inflation rising - the hardest quadrant; gold",
    "deflation": "Growth falling, inflation falling - nominal bonds",
}

CYCLE_DESCRIPTIONS = {
    "expansion": "Growth above trend and still accelerating",
    "slowdown": "Growth above trend but decelerating - the classic warning phase",
    "recession": "Growth below trend and still falling",
    "recovery": "Growth below trend but re-accelerating - typically the best entry",
}


# ---------------------------------------------------------------------------
# Model A - growth x inflation quadrants
# ---------------------------------------------------------------------------

def classify_quadrant(
    composites: pd.DataFrame,
    deadband: float | None = None,
    confirm_months: int | None = None,
    axis_definition: str | None = None,
) -> pd.DataFrame:
    """Classify each month into a growth x inflation quadrant.

    ``axis_definition`` controls what "up" means, and the choice materially changes
    the timeline - so it is a parameter, and the sensitivity analysis sweeps it
    rather than pretending one reading is canonical:

    * ``momentum`` - sign of the 3-month change (direction of travel; default,
      closest to the quadrant framework's intent)
    * ``level``    - sign of the level versus its expanding mean
    """
    cfg = model_config()["quadrant_model"]
    deadband = cfg["deadband_sigma"] if deadband is None else deadband
    confirm_months = cfg["confirm_months"] if confirm_months is None else confirm_months
    axis_definition = cfg["axis_definition"] if axis_definition is None else axis_definition

    growth, inflation = _axes(composites, axis_definition, "growth", "inflation")

    g = hold_last_nonzero(apply_deadband(growth, deadband))
    i = hold_last_nonzero(apply_deadband(inflation, deadband))

    raw = pd.Series(index=composites.index, dtype=object)
    raw[(g > 0) & (i < 0)] = "goldilocks"
    raw[(g > 0) & (i > 0)] = "reflation"
    raw[(g < 0) & (i > 0)] = "stagflation"
    raw[(g < 0) & (i < 0)] = "deflation"

    confirmed = apply_confirmation(raw, confirm_months)

    return pd.DataFrame(
        {
            "growth_axis": growth,
            "inflation_axis": inflation,
            "growth_sign": g,
            "inflation_sign": i,
            "regime_raw": raw,
            "regime": confirmed,
        },
        index=composites.index,
    )


# ---------------------------------------------------------------------------
# Model B - business cycle phases
# ---------------------------------------------------------------------------

def classify_cycle(
    composites: pd.DataFrame,
    deadband: float | None = None,
    confirm_months: int | None = None,
) -> pd.DataFrame:
    """Classify each month into a business-cycle phase.

    Level and momentum of the growth composite give the clockwise cycle:
    expansion (high, rising) -> slowdown (high, falling) -> recession (low, falling)
    -> recovery (low, rising).
    """
    cfg = model_config()["cycle_model"]
    deadband = cfg["deadband_sigma"] if deadband is None else deadband
    confirm_months = cfg["confirm_months"] if confirm_months is None else confirm_months

    if "growth_composite" not in composites.columns:
        raise KeyError("composites must contain 'growth_composite'")

    level = composites["growth_composite"]
    momentum = composites.get(
        "growth_composite_3m_change", level.diff(3)
    )

    lv = hold_last_nonzero(apply_deadband(level, deadband))
    mm = hold_last_nonzero(apply_deadband(momentum, deadband))

    raw = pd.Series(index=composites.index, dtype=object)
    raw[(lv > 0) & (mm > 0)] = "expansion"
    raw[(lv > 0) & (mm < 0)] = "slowdown"
    raw[(lv < 0) & (mm < 0)] = "recession"
    raw[(lv < 0) & (mm > 0)] = "recovery"

    confirmed = apply_confirmation(raw, confirm_months)

    return pd.DataFrame(
        {
            "level_axis": level,
            "momentum_axis": momentum,
            "level_sign": lv,
            "momentum_sign": mm,
            "regime_raw": raw,
            "regime": confirmed,
        },
        index=composites.index,
    )


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _axes(
    composites: pd.DataFrame,
    axis_definition: str,
    *names: str,
) -> tuple[pd.Series, ...]:
    """Build the requested axis representation for each named composite."""
    out = []
    for name in names:
        col = f"{name}_composite"
        if col not in composites.columns:
            raise KeyError(f"composites must contain {col!r}")
        base = composites[col]

        if axis_definition == "momentum":
            change_col = f"{name}_composite_3m_change"
            axis = composites[change_col] if change_col in composites else base.diff(3)
        elif axis_definition == "level":
            # level relative to its own expanding mean - causal by construction
            axis = base - base.expanding(min_periods=36).mean()
        else:
            raise ValueError(f"Unknown axis_definition {axis_definition!r}")

        out.append(axis.rename(f"{name}_axis"))

    return tuple(out)


def transition_matrix(states: pd.Series, order: list[str] | None = None) -> pd.DataFrame:
    """Empirical month-to-month transition probabilities."""
    s = states.dropna()
    pairs = pd.DataFrame({"from": s.shift(1), "to": s}).dropna()
    counts = pd.crosstab(pairs["from"], pairs["to"])

    if order:
        present = [o for o in order if o in counts.index or o in counts.columns]
        counts = counts.reindex(index=present, columns=present, fill_value=0)

    totals = counts.sum(axis=1).replace(0, pd.NA)
    return (counts.div(totals, axis=0)).astype("float64")


def expected_duration(states: pd.Series, order: list[str] | None = None) -> pd.Series:
    """Expected regime duration in months, from the self-transition probability.

    For a Markov chain with self-transition p, expected dwell time is 1/(1-p).

    This feeds the persistence-versus-lag diagnostic: if a regime lasts 9 months on
    average and the total signal lag is 3, roughly two-thirds of it is capturable.
    That comparison is what determines whether the whole enterprise can work at all.
    """
    tm = transition_matrix(states, order)
    if tm.empty:
        return pd.Series(dtype="float64")

    self_p = pd.Series(
        {s: tm.loc[s, s] for s in tm.index if s in tm.columns}, dtype="float64"
    )
    return (1.0 / (1.0 - self_p.clip(upper=0.999))).round(1)

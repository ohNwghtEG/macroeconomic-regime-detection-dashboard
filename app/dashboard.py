"""Streamlit dashboard.

    streamlit run app/dashboard.py

Reads the artefacts written by ``python -m macroregime.run``, so the dashboard is
a viewer rather than a second implementation - there is exactly one code path that
produces numbers, which keeps the app and the report from ever disagreeing.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from macroregime.config import REPORT_DIR  # noqa: E402
from macroregime.data.assets import build_asset_returns  # noqa: E402
from macroregime.data.fred import fetch_all  # noqa: E402
from macroregime.data.registry import load_specs  # noqa: E402
from macroregime.evaluation.conditional import regime_return_matrix  # noqa: E402
from macroregime.features import build_features  # noqa: E402
from macroregime.models.rules import (  # noqa: E402
    CYCLE_DESCRIPTIONS,
    QUADRANT_DESCRIPTIONS,
    classify_cycle,
    classify_quadrant,
    transition_matrix,
)
from macroregime.viz.charts import (  # noqa: E402
    composite_timeline,
    conditional_heatmap,
    correlation_shift_chart,
    equity_curves,
    regime_shaded_price,
    transition_heatmap,
)

st.set_page_config(
    page_title="Macro Regime Dashboard",
    page_icon="📊",
    layout="wide",
)


@st.cache_data(show_spinner="Loading macro data...")
def load_everything():
    raw = fetch_all(load_specs())
    F = build_features(raw, start="1962-01-31")
    returns, provenance = build_asset_returns(raw)
    cycle = classify_cycle(F["composites"])
    quadrant = classify_quadrant(F["composites"])
    return raw, F, returns, provenance, cycle, quadrant


def read_csv(name: str) -> pd.DataFrame | None:
    path = Path(REPORT_DIR) / f"{name}.csv"
    if not path.exists():
        return None
    return pd.read_csv(path, index_col=0)


raw, F, returns, provenance, cycle, quadrant = load_everything()
composites = F["composites"]

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
st.sidebar.title("Macro Regime Model")
model_choice = st.sidebar.radio(
    "Regime framework",
    ["Business cycle", "Growth x Inflation"],
    help=(
        "Two genuinely different frameworks. The business cycle is a sequence of "
        "phases; the quadrants are a state space over the direction of growth and "
        "inflation. They are often conflated - here they are built separately."
    ),
)

regimes = cycle["regime"] if model_choice == "Business cycle" else quadrant["regime"]
descriptions = (
    CYCLE_DESCRIPTIONS if model_choice == "Business cycle" else QUADRANT_DESCRIPTIONS
)

start_year = st.sidebar.slider("Start year", 1966, 2020, 1970)
regimes_v = regimes[regimes.index.year >= start_year]

st.sidebar.markdown("---")
st.sidebar.caption(
    "Every figure uses point-in-time data: each month only sees series already "
    "published by that date. Regenerate with `python -m macroregime.run`."
)

# ---------------------------------------------------------------------------
# Nowcast
# ---------------------------------------------------------------------------
st.title("Macroeconomic Regime Detection")

current = regimes.dropna()
if not current.empty:
    state = current.iloc[-1]
    asof = current.index[-1]

    run = 1
    for i in range(len(current) - 2, -1, -1):
        if current.iloc[i] == state:
            run += 1
        else:
            break

    tm = transition_matrix(regimes)
    persistence = float(tm.loc[state, state]) if state in tm.index else float("nan")
    expected = 1 / (1 - persistence) if persistence == persistence and persistence < 1 else float("nan")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Current regime", str(state).title())
    c2.metric("As of", asof.strftime("%b %Y"))
    c3.metric("Months in regime", run)
    c4.metric(
        "Expected duration",
        f"{expected:.0f} mo" if expected == expected else "n/a",
        delta=f"{expected - run:+.0f} mo remaining" if expected == expected else None,
    )
    st.caption(descriptions.get(str(state), ""))

st.info(
    "**Headline finding.** Over the full 55-year sample the macro regime strategies "
    "do **not** beat a rebalanced 60/40 by a statistically significant margin - and a "
    "price-only momentum signal using no macro data beats both. See the Backtest tab.",
    icon="🔍",
)

tabs = st.tabs(
    ["Regime timeline", "Macro factors", "Conditional returns",
     "Backtest", "Validation", "Data provenance"]
)

# ---------------------------------------------------------------------------
with tabs[0]:
    st.subheader("Regimes over the equity market")
    if "equity" in returns.columns:
        eq = returns["equity"].loc[returns.index.year >= start_year]
        curve = (1 + eq.fillna(0)).cumprod()
        st.plotly_chart(
            regime_shaded_price(curve, regimes_v, title=f"US equities, shaded by {model_choice.lower()} regime"),
            use_container_width=True,
        )

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Transition probabilities**")
        st.plotly_chart(transition_heatmap(transition_matrix(regimes)), use_container_width=True)
    with c2:
        st.markdown("**Episodes per regime**")
        st.caption(
            "The effective sample size is EPISODES, not months - a state entered "
            "four times cannot support a conditional return estimate."
        )
        from macroregime.models.hysteresis import episode_summary
        st.dataframe(episode_summary(regimes), use_container_width=True)

# ---------------------------------------------------------------------------
with tabs[1]:
    st.subheader("Macro composites")
    st.caption(
        "Expanding-window z-scores: the mean and standard deviation at each date "
        "use only history up to that date. A full-sample z-score would leak the "
        "future into every historical observation."
    )
    st.plotly_chart(
        composite_timeline(
            composites[composites.index.year >= start_year],
            columns=("growth_composite", "inflation_composite", "financial_composite"),
        ),
        use_container_width=True,
    )

# ---------------------------------------------------------------------------
with tabs[2]:
    st.subheader("Performance conditional on regime")
    key = "conditional_cycle" if model_choice == "Business cycle" else "conditional_quadrant"
    cond = read_csv(key)

    if cond is not None and not cond.empty:
        st.plotly_chart(
            conditional_heatmap(regime_return_matrix(cond, "mean_return_%")),
            use_container_width=True,
        )
        st.markdown("**Full statistics**")
        st.caption(
            "`t_stat_hac` is Newey-West corrected; `t_stat_naive` is not. The gap "
            "between them is the significance that overlapping returns would have "
            "manufactured."
        )
        st.dataframe(cond, use_container_width=True)
    else:
        st.warning("Run `python -m macroregime.run` first.")

    shift = read_csv("correlation_shift")
    if shift is not None and not shift.empty:
        st.subheader("Equity/bond correlation by regime")
        st.caption(
            "The most allocation-relevant output here. Regime models matter less "
            "because expected returns shift than because the diversification a "
            "60/40 depends on stops working in particular states."
        )
        st.plotly_chart(correlation_shift_chart(shift), use_container_width=True)

# ---------------------------------------------------------------------------
with tabs[3]:
    st.subheader("Backtest vs benchmarks")
    summary = read_csv("summary_long")
    if summary is not None:
        st.dataframe(summary, use_container_width=True)

    sig = read_csv("significance_long")
    if sig is not None:
        st.markdown("**Is any difference distinguishable from luck?**")
        st.caption(
            "Stationary bootstrap of the Sharpe DIFFERENCE vs 60/40, 5000 iterations, "
            "12-month mean block length."
        )
        st.dataframe(sig, use_container_width=True)

        st.error(
            "No strategy beats 60/40 at the 5% level. `momentum_no_macro` - which "
            "uses only prices and no macro data at all - has the highest Sharpe of "
            "any strategy tested. The honest reading is that markets price the "
            "regime before the macro data confirms it.",
            icon="⚠️",
        )

# ---------------------------------------------------------------------------
with tabs[4]:
    st.subheader("Validation against NBER recession dates")

    vs_sahm = read_csv("vs_sahm")
    if vs_sahm is not None:
        st.markdown("**Model vs the real-time Sahm rule**")
        st.caption(
            "Sahm is a single published line of arithmetic. If a multi-series model "
            "cannot beat it, that is the finding."
        )
        st.dataframe(vs_sahm, use_container_width=True)

    ll = read_csv("nber_lead_lag")
    if ll is not None:
        st.markdown("**Lead/lag per recession** (negative = fired before the NBER peak)")
        st.dataframe(ll, use_container_width=True)

    fp = read_csv("nber_false_positives")
    if fp is not None and not fp.empty:
        st.markdown("**False positives** - called recession, NBER disagreed")
        st.caption(
            "The growth scares. A tactical overlay that de-risks this often bleeds "
            "return in the years when nothing is actually wrong."
        )
        st.dataframe(fp, use_container_width=True)

# ---------------------------------------------------------------------------
with tabs[5]:
    st.subheader("Where the data comes from")
    st.markdown("**Asset history and splice points**")
    st.dataframe(provenance, use_container_width=True)

    st.markdown("**Macro series staleness at a month-end decision point**")
    st.caption(
        "`staleness_days` is how old the newest available print is when a trade is "
        "made. This is the real signal lag the strategy has to overcome."
    )
    stale = read_csv("staleness")
    if stale is not None:
        st.dataframe(stale, use_container_width=True)

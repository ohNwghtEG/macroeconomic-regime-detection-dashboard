"""Plotly charts.

Palette note: regime colours are chosen to stay distinguishable in greyscale and
for common forms of colour vision deficiency, since red/green is the obvious
choice for "bad/good" and the worst one available.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

REGIME_COLORS = {
    # business cycle
    "expansion": "#2A9D8F",
    "slowdown": "#E9C46A",
    "recession": "#C1495A",
    "recovery": "#5B8FCC",
    # growth x inflation quadrants
    "goldilocks": "#2A9D8F",
    "reflation": "#E76F51",
    "stagflation": "#8B4A62",
    "deflation": "#4A6FA5",
    # HMM states
    "0": "#C1495A",
    "1": "#E9C46A",
    "2": "#2A9D8F",
    "3": "#5B8FCC",
}

LAYOUT = dict(
    template="plotly_white",
    font=dict(family="Inter, Segoe UI, system-ui, sans-serif", size=12),
    margin=dict(l=60, r=30, t=60, b=50),
    hovermode="x unified",
)


def _color(state) -> str:
    return REGIME_COLORS.get(str(state), "#999999")


def regime_shaded_price(
    price: pd.Series,
    regimes: pd.Series,
    title: str = "Equity index with regime shading",
    log_scale: bool = True,
) -> go.Figure:
    """Cumulative equity curve with the regime timeline as background bands.

    The signature chart of the project: it shows at a glance whether the regime
    calls line up with what markets actually did.
    """
    fig = go.Figure()
    s = regimes.dropna()

    if not s.empty:
        blocks = (s != s.shift(1)).cumsum()
        seen = set()
        for _, block in s.groupby(blocks):
            state = block.iloc[0]
            fig.add_vrect(
                x0=block.index[0],
                x1=block.index[-1],
                fillcolor=_color(state),
                opacity=0.18,
                layer="below",
                line_width=0,
                annotation_text=None,
            )
            if state not in seen:
                seen.add(state)
                # invisible trace purely to populate the legend
                fig.add_trace(
                    go.Scatter(
                        x=[block.index[0]], y=[np.nan], mode="markers",
                        marker=dict(size=10, color=_color(state), symbol="square"),
                        name=str(state), showlegend=True,
                    )
                )

    fig.add_trace(
        go.Scatter(
            x=price.index, y=price.to_numpy(),
            mode="lines", name="Cumulative return",
            line=dict(color="#1D3557", width=1.8),
        )
    )

    fig.update_layout(
        title=title,
        yaxis_title="Growth of $1 (log scale)" if log_scale else "Growth of $1",
        xaxis_title=None,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
        **LAYOUT,
    )
    if log_scale:
        fig.update_yaxes(type="log")
    return fig


def regime_probability_chart(
    probabilities: pd.DataFrame,
    title: str = "HMM filtered state probabilities",
) -> go.Figure:
    """Stacked area of filtered state probabilities.

    Filtered, not smoothed - this is what was knowable in real time.
    """
    fig = go.Figure()
    for i, col in enumerate(probabilities.columns):
        fig.add_trace(
            go.Scatter(
                x=probabilities.index,
                y=probabilities[col].to_numpy(),
                mode="lines", stackgroup="one", name=str(col),
                line=dict(width=0.5, color=_color(str(i))),
                hovertemplate="%{y:.1%}<extra>" + str(col) + "</extra>",
            )
        )
    fig.update_layout(
        title=title, yaxis_title="P(state | data through t)",
        yaxis=dict(range=[0, 1], tickformat=".0%"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
        **LAYOUT,
    )
    return fig


def conditional_heatmap(
    matrix: pd.DataFrame,
    title: str = "Annualized return by asset and regime (%)",
    colorbar_title: str = "% p.a.",
) -> go.Figure:
    """Asset x regime heatmap with a diverging scale centred on zero."""
    if matrix.empty:
        return go.Figure()

    vmax = float(np.nanmax(np.abs(matrix.to_numpy(dtype="float64"))))
    fig = go.Figure(
        go.Heatmap(
            z=matrix.to_numpy(dtype="float64"),
            x=[str(c) for c in matrix.columns],
            y=[str(i) for i in matrix.index],
            colorscale="RdBu",
            zmid=0, zmin=-vmax, zmax=vmax,
            text=np.round(matrix.to_numpy(dtype="float64"), 1),
            texttemplate="%{text}",
            textfont=dict(size=12),
            colorbar=dict(title=colorbar_title),
            hovertemplate="%{y} in %{x}: %{z:.2f}<extra></extra>",
        )
    )
    fig.update_layout(title=title, **LAYOUT)
    return fig


def equity_curves(
    results: dict[str, pd.DataFrame],
    title: str = "Strategy vs benchmarks (net of costs)",
    highlight: tuple[str, ...] = ("sixty_forty", "momentum_no_macro"),
) -> go.Figure:
    """Equity curves on a common sample, rebased to 1."""
    fig = go.Figure()

    common = None
    for res in results.values():
        idx = res["net_return"].dropna().index
        common = idx if common is None else common.intersection(idx)
    if common is None or len(common) == 0:
        return fig

    for name, res in results.items():
        curve = (1 + res["net_return"].loc[common]).cumprod()
        is_bench = name in highlight
        fig.add_trace(
            go.Scatter(
                x=curve.index, y=curve.to_numpy(), mode="lines", name=name,
                line=dict(
                    width=2.4 if is_bench else 1.6,
                    dash="dash" if name == "sixty_forty" else None,
                ),
            )
        )

    fig.update_layout(
        title=title, yaxis_title="Growth of $1 (log scale)",
        yaxis=dict(type="log"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
        **LAYOUT,
    )
    return fig


def transition_heatmap(
    tm: pd.DataFrame, title: str = "Regime transition probabilities"
) -> go.Figure:
    """Month-to-month transition matrix. The diagonal is regime persistence."""
    if tm.empty:
        return go.Figure()

    fig = go.Figure(
        go.Heatmap(
            z=tm.to_numpy(dtype="float64"),
            x=[str(c) for c in tm.columns],
            y=[str(i) for i in tm.index],
            colorscale="Blues", zmin=0, zmax=1,
            text=np.round(tm.to_numpy(dtype="float64"), 2),
            texttemplate="%{text}",
            colorbar=dict(title="P"),
            hovertemplate="from %{y} to %{x}: %{z:.1%}<extra></extra>",
        )
    )
    fig.update_layout(
        title=title, xaxis_title="to state", yaxis_title="from state", **LAYOUT
    )
    return fig


def correlation_shift_chart(
    shift: pd.DataFrame, title: str = "Equity/bond correlation by regime"
) -> go.Figure:
    """Bar chart of a pair's correlation across regimes.

    The chart that justifies the whole exercise for an allocator: if correlation is
    stable across regimes, conditional allocation has far less to offer.
    """
    if shift.empty:
        return go.Figure()

    colors = [
        "#1D3557" if r == "ALL (unconditional)" else _color(r)
        for r in shift["regime"]
    ]
    fig = go.Figure(
        go.Bar(
            x=shift["regime"].astype(str),
            y=shift["correlation"],
            marker_color=colors,
            text=[f"{v:+.2f}" for v in shift["correlation"]],
            textposition="outside",
            hovertemplate="%{x}: %{y:+.3f}<extra></extra>",
        )
    )
    fig.add_hline(y=0, line_width=1, line_color="#333")
    fig.update_layout(
        title=title, yaxis_title="correlation",
        yaxis=dict(range=[-1, 1]), **LAYOUT,
    )
    return fig


def composite_timeline(
    composites: pd.DataFrame,
    columns: tuple[str, ...] = ("growth_composite", "inflation_composite"),
    title: str = "Macro composites (expanding z-scores)",
) -> go.Figure:
    """The underlying macro factors driving every classification."""
    fig = make_subplots(
        rows=len(columns), cols=1, shared_xaxes=True,
        subplot_titles=[c.replace("_", " ").title() for c in columns],
        vertical_spacing=0.08,
    )
    palette = ["#2A9D8F", "#E76F51", "#4A6FA5", "#8B4A62"]

    for i, col in enumerate(columns, start=1):
        if col not in composites.columns:
            continue
        fig.add_trace(
            go.Scatter(
                x=composites.index, y=composites[col].to_numpy(),
                mode="lines", name=col,
                line=dict(color=palette[(i - 1) % len(palette)], width=1.6),
            ),
            row=i, col=1,
        )
        fig.add_hline(y=0, line_width=1, line_dash="dot", line_color="#888",
                      row=i, col=1)

    fig.update_layout(title=title, showlegend=False, height=180 * len(columns) + 120,
                      **LAYOUT)
    return fig

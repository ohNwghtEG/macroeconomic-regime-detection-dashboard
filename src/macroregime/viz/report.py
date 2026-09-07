"""Static HTML report generator.

    python -m macroregime.viz.report

Produces a single self-contained ``reports/report.html``. This exists because a
reviewer will not `pip install` anything: the repo needs to show its results to
someone who only opens a file.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd
import plotly.io as pio

from ..config import REPORT_DIR
from ..data.assets import build_asset_returns
from ..data.fred import fetch_all
from ..data.registry import load_specs
from ..evaluation.conditional import regime_return_matrix
from ..features import build_features
from ..models.rules import classify_cycle, classify_quadrant, transition_matrix
from .charts import (
    composite_timeline,
    conditional_heatmap,
    correlation_shift_chart,
    equity_curves,
    regime_shaded_price,
    transition_heatmap,
)

log = logging.getLogger(__name__)

CSS = """
:root { --ink:#1a1a1a; --muted:#5f6b7a; --rule:#e3e8ef; --bg:#ffffff;
        --accent:#1D3557; --warn:#8a5a00; --warnbg:#fff8e6; }
* { box-sizing:border-box; }
body { margin:0; padding:0; background:var(--bg); color:var(--ink);
       font:16px/1.65 -apple-system,BlinkMacSystemFont,"Segoe UI",Inter,sans-serif; }
.wrap { max-width:1080px; margin:0 auto; padding:48px 24px 96px; }
h1 { font-size:2.1rem; line-height:1.2; margin:0 0 8px; letter-spacing:-.02em; }
h2 { font-size:1.4rem; margin:56px 0 12px; padding-top:20px;
     border-top:1px solid var(--rule); letter-spacing:-.01em; }
h3 { font-size:1.05rem; margin:32px 0 8px; color:var(--accent); }
p { margin:0 0 14px; } .lede { font-size:1.1rem; color:var(--muted); margin-bottom:28px; }
.finding { background:var(--warnbg); border-left:3px solid var(--warn);
           padding:16px 20px; margin:24px 0; border-radius:0 6px 6px 0; }
.finding strong { color:var(--warn); }
table { border-collapse:collapse; width:100%; font-size:.87rem; margin:16px 0; }
th,td { text-align:right; padding:7px 10px; border-bottom:1px solid var(--rule); }
th:first-child,td:first-child { text-align:left; }
th { font-weight:600; color:var(--muted); font-size:.78rem;
     text-transform:uppercase; letter-spacing:.04em; }
tbody tr:hover { background:#f7f9fc; }
.chart { margin:24px 0 8px; }
.note { font-size:.86rem; color:var(--muted); margin:6px 0 28px; }
.scroll { overflow-x:auto; }
code { background:#f2f4f7; padding:1px 5px; border-radius:3px; font-size:.85em; }
footer { margin-top:64px; padding-top:20px; border-top:1px solid var(--rule);
         color:var(--muted); font-size:.85rem; }
@media (prefers-color-scheme: dark) {
  :root { --ink:#e8eaed; --muted:#9aa4b2; --rule:#2a3038; --bg:#14171c;
          --accent:#7aa2d6; --warnbg:#2a2413; --warn:#d4a94a; }
  tbody tr:hover { background:#1b1f26; } code { background:#22272e; }
}
"""


def _table(df: pd.DataFrame, index: bool = False) -> str:
    if df is None or (hasattr(df, "empty") and df.empty):
        return "<p class='note'>Not available - run <code>python -m macroregime.run</code>.</p>"
    return f"<div class='scroll'>{df.to_html(index=index, border=0, escape=False)}</div>"


def _chart(fig, height: int = 460) -> str:
    if fig is None:
        return ""
    return (
        "<div class='chart'>"
        + pio.to_html(fig, include_plotlyjs=False, full_html=False,
                      default_height=height, config={"displayModeBar": False})
        + "</div>"
    )


def _csv(name: str) -> pd.DataFrame | None:
    path = Path(REPORT_DIR) / f"{name}.csv"
    return pd.read_csv(path, index_col=0) if path.exists() else None


def build_report(output: Path | None = None) -> Path:
    """Render the full static report."""
    output = Path(output) if output else Path(REPORT_DIR) / "report.html"

    raw = fetch_all(load_specs())
    F = build_features(raw, start="1962-01-31")
    returns, provenance = build_asset_returns(raw)
    composites = F["composites"]
    cycle = classify_cycle(composites)
    quadrant = classify_quadrant(composites)

    summary_long = _csv("summary_long")
    summary_wf = _csv("summary_all_assets")
    sig = _csv("significance_long")
    cond = _csv("conditional_cycle")
    shift = _csv("correlation_shift")
    vs_sahm = _csv("vs_sahm")
    lead_lag = _csv("nber_lead_lag")
    hmm_states = _csv("hmm_states")
    stale = _csv("staleness")

    eq = returns["equity"].loc["1970-01-31":]
    curve = (1 + eq.fillna(0)).cumprod()

    parts: list[str] = []
    A = parts.append

    A("<h1>Macroeconomic Regime Detection</h1>")
    A("<p class='lede'>Does a systematic macro regime model beat a static 60/40 "
      "after realistic data lags and costs &mdash; and does it beat a signal that "
      "uses no macro data at all?</p>")

    A("<div class='finding'><strong>Finding.</strong> No. Across 55 years of "
      "point-in-time data, every macro regime strategy lands <em>below</em> a "
      "rebalanced 60/40 on Sharpe, and none of the differences are statistically "
      "significant. A price-only momentum signal ranks above every macro model "
      "tested. The likeliest explanation is that markets price the regime before "
      "the macro data confirms it.</div>")

    A("<h2>Regime timeline</h2>")
    A(_chart(regime_shaded_price(
        curve, cycle["regime"].loc["1970-01-31":],
        title="US equities, shaded by business-cycle regime"), 480))
    A("<p class='note'>Regimes are assigned from data available at each month-end, "
      "not from revised history.</p>")

    A("<h2>Macro factors</h2>")
    A(_chart(composite_timeline(
        composites.loc["1970-01-31":],
        columns=("growth_composite", "inflation_composite", "financial_composite")), 640))
    A("<p class='note'>Expanding-window z-scores: the mean and standard deviation "
      "at each date use only history up to that date. A full-sample z-score would "
      "leak the future into every historical observation.</p>")

    A("<h2>Backtest</h2>")
    A("<h3>Full sample, 1970&ndash;2026</h3>")
    A(_table(summary_long))
    if summary_wf is not None:
        A("<h3>Including the walk-forward HMM</h3>")
        A(_table(summary_wf))
    A("<h3>Is any difference vs 60/40 real?</h3>")
    A(_table(sig, index=True))
    A("<p class='note'>Stationary bootstrap of the Sharpe difference, 5000 "
      "iterations, 12-month mean block length. No strategy reaches the 5% level.</p>")

    A("<h2>Regime-conditional performance</h2>")
    if cond is not None and not cond.empty:
        A(_chart(conditional_heatmap(regime_return_matrix(cond, "mean_return_%")), 380))
        A(_table(cond))
        A("<p class='note'><code>t_stat_hac</code> is Newey-West corrected for the "
          "overlap in forward returns; <code>t_stat_naive</code> is not. The gap "
          "between them is significance that overlapping windows would have "
          "manufactured.</p>")

    if shift is not None and not shift.empty:
        A("<h3>Equity/bond correlation by regime</h3>")
        A(_chart(correlation_shift_chart(shift), 380))
        A("<p class='note'>Notably flat. The correlation shift across regimes is "
          "small in this sample, which is part of why regime-conditional "
          "allocation adds so little here.</p>")

    A("<h2>Validation against NBER</h2>")
    A("<h3>Model vs the real-time Sahm rule</h3>")
    A(_table(vs_sahm))
    A("<p class='note'>The Sahm rule is a single published line of arithmetic and "
      "beats the rule-based model on F1. A model that calls recession constantly "
      "achieves a good average lead time for uninteresting reasons.</p>")
    A("<h3>Lead/lag per recession</h3>")
    A(_table(lead_lag))
    A("<p class='note'>Negative = signal fired before the NBER peak.</p>")

    A("<h2>Model detail</h2>")
    A("<h3>Transition probabilities</h3>")
    A(_chart(transition_heatmap(transition_matrix(cycle["regime"])), 380))
    if hmm_states is not None:
        A("<h3>HMM state characteristics</h3>")
        A(_table(hmm_states))
        A("<p class='note'>Three states, selected on seed-to-seed stability and "
          "episode adequacy rather than likelihood &mdash; BIC falls monotonically "
          "across the whole grid tested and so selects nothing.</p>")

    A("<h2>Data provenance</h2>")
    A(_table(provenance))
    A("<p class='note'>Long-history proxies are spliced behind each ETF on "
      "<em>returns</em>, not levels.</p>")
    if stale is not None:
        A("<h3>Series staleness at a decision point</h3>")
        A(_table(stale))
        A("<p class='note'><code>staleness_days</code> is how old the newest "
          "available print is at the moment of trading &mdash; the real signal lag "
          "the strategy must overcome.</p>")

    A("<footer>Generated by <code>python -m macroregime.viz.report</code> on "
      f"{pd.Timestamp.now():%Y-%m-%d}. Sources: FRED, Yahoo Finance, "
      "Ken French data library. Not investment advice.</footer>")

    html = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Macroeconomic Regime Detection</title>
<script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
<style>{CSS}</style></head>
<body><div class="wrap">{''.join(parts)}</div></body></html>"""

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(html, encoding="utf-8")
    log.info("report written to %s (%.0f KB)", output, output.stat().st_size / 1024)
    return output


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    path = build_report()
    print(f"Report written to {path}")

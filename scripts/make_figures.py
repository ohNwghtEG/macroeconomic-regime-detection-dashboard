"""Paper Figures 1 and 2, generated from the committed report CSVs.

    python scripts/make_figures.py

Figure 1  reports/power_years_by_rho.csv
          Years of monthly data needed for 80% power against rho, log scale,
          one line per effect size, reference line at the sample length.
Figure 2  reports/significance_long.csv (+ mde_80 from power_reconciliation.csv)
          Observed Sharpe difference with its 95% bootstrap interval, beside the
          band of effects smaller than the 80% minimum detectable effect.

Every plotted value is read from the CSVs; nothing is typed in by hand except
the sample length, which is itself read from summary_long.csv.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

REPORTS = Path(__file__).resolve().parents[1] / "reports"

INK, INK2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#ffffff"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]   # fixed categorical order
BAND = "#d9d8d3"

LABELS = {
    "hmm_walkforward": "HMM (walk-forward)",
    "rule_cycle": "Rule: business cycle",
    "equal_weight": "Equal weight",
    "rule_quadrant": "Rule: growth × inflation",
    "momentum_no_macro": "Momentum (no macro)",
}

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 9, "axes.edgecolor": INK2,
    "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
    "axes.spines.top": False, "axes.spines.right": False,
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
})


def _3dp(x: float) -> str:
    """Round half up, as the manuscript tables do (0.8885 -> 0.889)."""
    from decimal import ROUND_HALF_UP, Decimal
    return str(Decimal(str(x)).quantize(Decimal("0.001"), rounding=ROUND_HALF_UP))


def sample_years() -> float:
    s = pd.read_csv(REPORTS / "summary_long.csv")
    return round(float(s["months"].iloc[0]) / 12, 1)


def figure_1(out: Path) -> None:
    df = pd.read_csv(REPORTS / "power_years_by_rho.csv")
    years = sample_years()
    fig, ax = plt.subplots(figsize=(6.2, 4.0))
    for colour, d in zip(SERIES, ("0.10", "0.20", "0.30", "0.50")):
        y = df[f"delta_{d}"]
        ax.plot(df["rho"], y, color=colour, lw=2, marker="o", ms=4,
                markeredgecolor=SURFACE, markeredgewidth=1, label=f"Δ = {d}")
        ax.annotate(f"Δ = {d}", (df["rho"].iloc[-1], y.iloc[-1]),
                    xytext=(6, 0), textcoords="offset points", va="center",
                    color=INK, fontsize=8.5)
    ax.axhline(years, color=INK, lw=1, ls=(0, (4, 3)))
    ax.text(df["rho"].iloc[0], years * 0.82,
            f"Sample available: {years} years", color=INK, fontsize=8.5, va="top")
    ax.set_yscale("log")
    ax.set_yticks([5, 10, 20, 50, 100, 200, 500, 1000])
    ax.set_yticklabels(["5", "10", "20", "50", "100", "200", "500", "1,000"])
    ax.set_xticks(df["rho"])
    ax.set_xlim(df["rho"].min() - 0.01, df["rho"].max() + 0.035)
    ax.set_xlabel("Correlation between strategy and benchmark (ρ)")
    ax.set_ylabel("Years of monthly data required (log scale)")
    ax.grid(axis="y", which="major", color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, loc="upper right", fontsize=8.5, ncol=2)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(out.with_suffix(f".{ext}"), dpi=200)
    plt.close(fig)


def figure_2(out: Path) -> None:
    sig = pd.read_csv(REPORTS / "significance_long.csv", index_col=0)
    rec = pd.read_csv(REPORTS / "power_reconciliation.csv").set_index("comparison")
    df = sig.join(rec[["mde_80"]]).sort_values("correlation", ascending=True)

    fig, ax = plt.subplots(figsize=(6.2, 3.4))
    for i, (name, r) in enumerate(df.iterrows()):
        m = r["mde_80"]
        ax.barh(i, 2 * m, left=-m, height=0.62, color=BAND, zorder=1,
                label="Not detectable at 80% power (|Δ| < 80% MDE)" if i == 0 else None)
        ax.plot([r["ci_lower_95"], r["ci_upper_95"]], [i, i], color=SERIES[0],
                lw=2, solid_capstyle="round", zorder=2,
                label="95% bootstrap CI" if i == 0 else None)
        ax.plot(r["sharpe_difference"], i, "o", color=SERIES[0], ms=6.5,
                markeredgecolor=SURFACE, markeredgewidth=1.5, zorder=3,
                label="Observed Δ" if i == 0 else None)
        ax.text(0.515, i, f"MDE ±{m:.3f}", va="center", ha="left",
                color=INK2, fontsize=8)
    ax.axvline(0, color=INK2, lw=0.8, zorder=0)
    ax.set_yticks(range(len(df)))
    ax.set_yticklabels([f"{LABELS.get(n, n)}  (ρ = {_3dp(c)})"
                        for n, c in zip(df.index, df["correlation"])])
    ax.set_xlim(-0.5, 0.5)
    ax.set_xlabel("Sharpe difference vs 60/40 (annualised)")
    ax.grid(axis="x", color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.42, -0.2),
              ncol=3, fontsize=7.5, handlelength=1.4)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(out.with_suffix(f".{ext}"), dpi=200, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    figure_1(REPORTS / "figure1_years_by_rho")
    figure_2(REPORTS / "figure2_detectability")
    print(f"figures written to {REPORTS}")


if __name__ == "__main__":
    main()

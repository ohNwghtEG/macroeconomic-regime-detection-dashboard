"""End-to-end pipeline.

    python -m macroregime.run              # from cache, no API key needed
    python -m macroregime.run --refresh    # re-fetch from FRED and yfinance
    python -m macroregime.run --quick      # skip the expanding-window HMM

Writes every artefact to ``reports/`` and caches intermediates to ``data/cache``.
"""

from __future__ import annotations

import argparse
import json
import logging
import warnings
from pathlib import Path

import pandas as pd

from .backtest.engine import run_strategy_suite, states_to_probabilities
from .backtest.stats import (
    bootstrap_sharpe_difference,
    deflated_sharpe_ratio,
    risk_free_monthly,
    summary_table,
    to_excess,
)
from .config import REPORT_DIR, ensure_dirs, model_config
from .data.assets import build_asset_returns
from .data.fred import fetch_all
from .data.registry import load_specs, staleness_report
from .evaluation.conditional import (
    conditional_correlations,
    conditional_performance,
    correlation_shift_table,
)
from .evaluation.nber import (
    compare_to_sahm,
    false_positive_episodes,
    lead_lag_analysis,
    recession_classification_report,
)
from .features import build_features
from .models.hmm import describe_states, expanding_window_states, full_sample_states
from .models.hysteresis import episode_summary
from .models.rules import classify_cycle, classify_quadrant, expected_duration

log = logging.getLogger(__name__)

HMM_FEATURES = ["growth_composite", "inflation_composite", "financial_composite"]


def run_pipeline(refresh: bool = False, quick: bool = False) -> dict:
    """Run every stage and persist results."""
    ensure_dirs()
    warnings.filterwarnings("ignore")
    out: dict[str, object] = {}

    # ---------------- data ----------------
    log.info("[1/7] loading macro data")
    specs = load_specs()
    raw = fetch_all(specs, refresh=refresh)
    out["staleness"] = staleness_report(raw, specs)

    # Risk-free rate. Sharpe ratios are computed on EXCESS returns throughout; the
    # 3-month bill averaged 3.65% over this sample and 7.80% in the 1980s portion,
    # so omitting it inflates every ratio and distorts the rankings.
    rf = risk_free_monthly(raw)
    out["risk_free"] = rf

    log.info("[2/7] building features")
    F = build_features(raw, start="1962-01-31")
    composites = F["composites"]
    out["features"] = F

    log.info("[3/7] loading asset returns")
    returns, provenance = build_asset_returns(raw, refresh=refresh)
    out["provenance"] = provenance

    # ---------------- regime models ----------------
    log.info("[4/7] fitting rule-based models")
    cycle = classify_cycle(composites)
    quadrant = classify_quadrant(composites)
    out["cycle"] = cycle
    out["quadrant"] = quadrant
    out["cycle_episodes"] = episode_summary(cycle["regime"])
    out["quadrant_episodes"] = episode_summary(quadrant["regime"])
    out["cycle_duration"] = expected_duration(cycle["regime"])
    out["quadrant_duration"] = expected_duration(quadrant["regime"])

    log.info("[5/7] fitting HMM")
    cfg = model_config()["hmm"]
    n_states = int(cfg.get("n_states", 3))
    X = composites[HMM_FEATURES].dropna()

    full = full_sample_states(X, n_states=n_states, n_init=int(cfg.get("n_init", 20)))
    out["hmm_full"] = full
    out["hmm_states"] = describe_states(full.attrs["model"], HMM_FEATURES, n_states)
    out["hmm_filtered_vs_smoothed_disagreement"] = float(
        (full["state_filtered"] != full["state_smoothed"]).mean()
    )

    if not quick:
        log.info("      expanding-window refits (this is the slow part)")
        walk = expanding_window_states(
            X,
            n_states=n_states,
            initial_train_months=int(cfg["refit"]["initial_train_months"]),
            refit_every_months=int(cfg["refit"]["refit_every_months"]),
            n_init=6,
        )
        out["hmm_walkforward"] = walk

    # ---------------- validation ----------------
    log.info("[6/7] validating against NBER")
    recession_pred = (cycle["regime"] == "recession").astype(int)
    out["nber_report"] = recession_classification_report(recession_pred, raw["USREC"])
    out["lead_lag"] = lead_lag_analysis(recession_pred, raw["USREC"])
    out["false_positives"] = false_positive_episodes(recession_pred, raw["USREC"], 3)
    out["vs_sahm"] = compare_to_sahm(
        recession_pred, raw["SAHMREALTIME"], raw["USREC"]
    )

    # ---------------- conditional analysis + backtest ----------------
    log.info("[7/7] conditional analysis and backtest")
    long_assets = ["equity", "bonds", "commodities"]
    all_assets = ["equity", "bonds", "gold", "commodities"]

    R_long = returns[long_assets].loc["1970-01-31":].dropna()
    R_all = returns[[c for c in all_assets if c in returns.columns]].loc["1970-01-31":]

    out["conditional_cycle"] = conditional_performance(R_long, cycle["regime"])
    out["conditional_quadrant"] = conditional_performance(R_long, quadrant["regime"])
    out["correlations_cycle"] = conditional_correlations(R_long, cycle["regime"])
    out["correlation_shift"] = correlation_shift_table(
        R_long, cycle["regime"], pair=("equity", "bonds")
    )

    probs = {
        "rule_cycle": states_to_probabilities(cycle["regime"]),
        "rule_quadrant": states_to_probabilities(quadrant["regime"]),
    }
    if "hmm_walkforward" in out:
        walk = out["hmm_walkforward"]
        pcols = [c for c in walk.columns if c.startswith("p_state_")]
        hmm_probs = walk[pcols].dropna()
        hmm_probs.columns = [f"hmm_{c.split('_')[-1]}" for c in pcols]
        probs["hmm_walkforward"] = hmm_probs

    for label, R in (("long", R_long), ("all_assets", R_all.dropna())):
        if R.empty:
            continue
        results = run_strategy_suite(R, probs)
        if not results:
            continue

        out[f"backtest_{label}"] = results
        out[f"summary_{label}"] = summary_table(results, rf=rf)

        common = None
        for res in results.values():
            idx = res["net_return"].dropna().index
            common = idx if common is None else common.intersection(idx)

        if "sixty_forty" in results and common is not None and len(common) > 24:
            bench = results["sixty_forty"]["net_return"].loc[common]
            sig = {}
            for name, res in results.items():
                if name == "sixty_forty":
                    continue
                sig[name] = bootstrap_sharpe_difference(
                    res["net_return"].loc[common], bench, rf=rf, n_iterations=5000
                )
            out[f"significance_{label}"] = pd.DataFrame(sig).T
            out[f"deflated_{label}"] = {
                name: deflated_sharpe_ratio(
                    to_excess(res["net_return"].loc[common], rf), n_trials=12
                )
                for name, res in results.items()
            }

    _persist(out)
    return out


def _persist(out: dict) -> None:
    """Write results to reports/ as CSV and JSON."""
    d = Path(REPORT_DIR)
    d.mkdir(parents=True, exist_ok=True)

    frames = {
        "staleness": out.get("staleness"),
        "asset_provenance": out.get("provenance"),
        "hmm_states": out.get("hmm_states"),
        "cycle_episodes": out.get("cycle_episodes"),
        "quadrant_episodes": out.get("quadrant_episodes"),
        "nber_lead_lag": out.get("lead_lag"),
        "nber_false_positives": out.get("false_positives"),
        "vs_sahm": out.get("vs_sahm"),
        "conditional_cycle": out.get("conditional_cycle"),
        "conditional_quadrant": out.get("conditional_quadrant"),
        "correlation_shift": out.get("correlation_shift"),
        "summary_long": out.get("summary_long"),
        "summary_all_assets": out.get("summary_all_assets"),
        "significance_long": out.get("significance_long"),
        "significance_all_assets": out.get("significance_all_assets"),
    }
    for name, df in frames.items():
        if isinstance(df, pd.DataFrame) and not df.empty:
            df.to_csv(d / f"{name}.csv", index=True)

    if "cycle" in out:
        regimes = pd.DataFrame(
            {
                "cycle": out["cycle"]["regime"],
                "quadrant": out["quadrant"]["regime"],
            }
        )
        if "hmm_walkforward" in out:
            regimes["hmm_state"] = out["hmm_walkforward"]["state"]
        regimes.to_csv(d / "regime_timeline.csv")

    meta = {
        "nber_report": out.get("nber_report"),
        "filtered_vs_smoothed_disagreement": out.get(
            "hmm_filtered_vs_smoothed_disagreement"
        ),
        "generated_utc": pd.Timestamp.utcnow().isoformat(),
    }
    (d / "summary.json").write_text(json.dumps(meta, indent=2, default=str), "utf-8")
    log.info("results written to %s", d)


def main() -> None:
    parser = argparse.ArgumentParser(description="Macro regime detection pipeline")
    parser.add_argument("--refresh", action="store_true",
                        help="re-fetch data from FRED/yfinance (needs FRED_API_KEY)")
    parser.add_argument("--quick", action="store_true",
                        help="skip the expanding-window HMM refits")
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )

    out = run_pipeline(refresh=args.refresh, quick=args.quick)

    print("\n" + "=" * 78)
    print("PIPELINE COMPLETE")
    print("=" * 78)

    if "summary_long" in out:
        print("\nBacktest, long sample (equity/bonds/commodities):")
        print(out["summary_long"][
            ["strategy", "ann_return_%", "ann_vol_%", "sharpe", "max_drawdown_%", "years"]
        ].to_string(index=False))

    if "significance_long" in out:
        print("\nIs any difference vs 60/40 real?")
        sig = out["significance_long"]
        cols = [c for c in ["sharpe_difference", "p_value", "significant_at_5pct"]
                if c in sig.columns]
        print(sig[cols].to_string())

    print(f"\nArtefacts written to {REPORT_DIR}")


if __name__ == "__main__":
    main()

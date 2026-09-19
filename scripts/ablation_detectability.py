"""Ablation detectability: rho and 80% MDE for each look-ahead cell (paper Table 6).

    python scripts/ablation_detectability.py

Rebuilds the fixed- and optimised-weight ablation cells with the functions in
``macroregime.evaluation.ablation`` (identical features, HMM variants, common
sample, weights, costs and trade lag), keeps each cell's monthly net-return
series, and compares every biased cell with its own reference cell using
``bootstrap_sharpe_difference`` - the same paired stationary bootstrap, risk-free
rate and settings (5,000 draws, seed 7) that reproduce ``ablation_bootstrap.csv``.

For each cell it reports:

* ``rho``     - correlation of the cell's excess returns with the reference cell's
* ``boot_se`` - bootstrap standard error of the Sharpe difference, recovered from
                the percentile interval exactly as ``power.reconcile`` does
* ``mde_80``  - ``minimum_detectable_effect(boot_se, 0.80)``

Nothing here changes a method. The script first checks that it reproduces the
published Sharpe ratios, differences, intervals and p-values in
``ablation_bootstrap.csv`` and refuses to write if it does not, so the new
columns are guaranteed to come from the same runs as Table 3.

Outputs:
    reports/ablation_detectability.csv   Table 6 (all 14 cells, both weight modes)
    reports/ablation_sample.json         the common sample actually used
"""

from __future__ import annotations

import itertools
import json
import logging
import warnings

import numpy as np
import pandas as pd

from macroregime.backtest.engine import (
    DEFAULT_REGIME_WEIGHTS,
    backtest,
    probability_weighted_allocation,
)
from macroregime.backtest.stats import bootstrap_sharpe_difference, risk_free_monthly
from macroregime.config import REPORT_DIR, model_config
from macroregime.data.assets import build_asset_returns
from macroregime.data.fred import fetch_all
from macroregime.data.registry import load_specs
from macroregime.evaluation.ablation import (
    HMM_FEATURES,
    AblationConfig,
    build_features_variant,
    optimize_regime_weights,
    regime_probabilities_variant,
)
from macroregime.evaluation.power import minimum_detectable_effect

log = logging.getLogger("ablation_detectability")

N_ITER, SEED = 5000, 7           # reproduces ablation_bootstrap.csv / Table 3 exactly
TOL = 1.5e-3                     # published values are rounded to 3 dp


def build_cells(raw, returns, asof_index):
    cfg = model_config()["hmm"]["refit"]
    feats = {}
    for ignore_lag in (False, True):
        comp = build_features_variant(raw, asof_index, ignore_lag)
        feats[ignore_lag] = comp[HMM_FEATURES].dropna()

    probs = {}
    for key in itertools.product((False, True), repeat=3):
        log.info("cell %s", AblationConfig(*key).label)
        probs[key] = regime_probabilities_variant(
            feats[key[0]], smoothed_decoding=key[1], full_sample_fit=key[2],
            n_states=3,
            initial_train_months=int(cfg["initial_train_months"]),
            refit_every_months=int(cfg["refit_every_months"]),
            n_init=6, seed=42,
        )

    common = None
    for p in probs.values():
        i = p.dropna().index
        common = i if common is None else common.intersection(i)
    common = common.intersection(returns.index)

    out = {}
    for key in itertools.product((False, True), repeat=3):
        conf = AblationConfig(*key)
        P, R = probs[key].loc[common], returns.loc[common]
        for mode in ("fixed", "optimized"):
            w = DEFAULT_REGIME_WEIGHTS if mode == "fixed" else optimize_regime_weights(R, P)
            W = probability_weighted_allocation(P, w, list(returns.columns))
            out[(mode, conf.label)] = backtest(W, R, 10.0, 1, name=conf.label)
    return out, common


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    warnings.filterwarnings("ignore")

    raw = fetch_all(load_specs())
    rf = risk_free_monthly(raw)
    returns, _ = build_asset_returns(raw)
    R = returns[["equity", "bonds", "commodities"]].loc["1970-01-31":].dropna()
    asof = pd.date_range("1962-01-31", "2026-08-31", freq="ME")

    cache = REPORT_DIR.parent / "data" / "cache" / "ablation_cells.pkl"
    if cache.exists():
        cells, common = pd.read_pickle(cache)
    else:
        cells, common = build_cells(raw, R, asof)
        pd.to_pickle((cells, common), cache)

    rows = []
    for mode in ("fixed", "optimized"):
        ref = cells[(mode, "none (honest)")]["net_return"]
        for (m, label), res in cells.items():
            if m != mode or label == "none (honest)":
                continue
            b = bootstrap_sharpe_difference(res["net_return"], ref, rf=rf,
                                            n_iterations=N_ITER, seed=SEED)
            se = (b["ci_upper_95"] - b["ci_lower_95"]) / (2 * 1.96)
            rows.append({
                "weights": mode, "cell": label,
                "sharpe": b["sharpe_strategy"], "sharpe_ref": b["sharpe_benchmark"],
                "delta": b["sharpe_difference"],
                "ci_low": b["ci_lower_95"], "ci_high": b["ci_upper_95"],
                "p": b["p_value"], "n_months": b["n_months"],
                "rho": b["correlation"], "boot_se": se,
                "mde_80": minimum_detectable_effect(se, 0.80),
                "mde_50": minimum_detectable_effect(se, 0.50),
            })
    df = pd.DataFrame(rows)

    # ---- guard: must reproduce the published ablation before writing anything
    pub = pd.read_csv(REPORT_DIR / "ablation_bootstrap.csv").dropna(subset=["p"])
    chk = df.merge(pub, on=["weights", "cell"], suffixes=("", "_pub"))
    assert len(chk) == 14, f"expected 14 published cells, matched {len(chk)}"
    bad = []
    for col in ("sharpe", "delta", "ci_low", "ci_high"):
        d = (chk[col] - chk[f"{col}_pub"]).abs()
        bad += [f"{r.weights}/{r.cell} {col}: {r[col]} vs {r[col + '_pub']}"
                for (_, r), x in zip(chk.iterrows(), d) if x > TOL]
    dp = (chk["p"] - chk["p_pub"]).abs()
    bad += [f"{r.weights}/{r.cell} p: {r.p} vs {r.p_pub}"
            for (_, r), x in zip(chk.iterrows(), dp) if x > 5e-4]
    if bad:
        raise SystemExit("does not reproduce ablation_bootstrap.csv:\n  " + "\n  ".join(bad))

    df.to_csv(REPORT_DIR / "ablation_detectability.csv", index=False)
    honest = cells[("fixed", "none (honest)")]["net_return"]
    (REPORT_DIR / "ablation_sample.json").write_text(json.dumps({
        "common_index_months": int(len(common)),
        "common_index_start": str(common.min().date()),
        "common_index_end": str(common.max().date()),
        "backtest_months": int(len(honest)),
        "backtest_start": str(honest.index.min().date()),
        "backtest_end": str(honest.index.max().date()),
    }, indent=2), "utf-8")

    print(df[df.weights == "fixed"][["cell", "rho", "boot_se", "mde_80", "delta"]]
          .round(4).to_string(index=False))


if __name__ == "__main__":
    main()

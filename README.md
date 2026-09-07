# Macroeconomic Regime Detection

**I built a systematic macro regime model, tested whether it beats a static 60/40, and found that a backtest of realistic length cannot answer the question.**

Over 44.6 years of point-in-time data, the walk-forward hidden Markov model returns an excess-return Sharpe of **0.655** against a rebalanced 60/40's **0.682** (p = 0.64). Nothing beats the benchmark. But the more useful result is *why* that null is uninformative: the study had between **5% and 34% power** against the effects it observed, and detecting a 0.1 Sharpe improvement would require between **128 and 610 years** of monthly data depending on the correlation between strategy and benchmark.

The full write-up is in **[PAPER.md](PAPER.md)**. This README is the orientation.

---

## The finding, in one table

December 1981 – August 2026, 535 months, equities / bonds / commodities, 10bps costs, one-month trade lag. **Sharpe ratios are on excess returns**, using the 3-month T-bill.

| Strategy | Ann. return | Ann. vol | Sharpe | Max DD | Δ vs 60/40 | p |
|---|---|---|---|---|---|---|
| **60/40 (rebalanced)** | 10.61% | 10.32% | **0.682** | −28.45% | — | — |
| Momentum (no macro data) | 10.07% | 9.52% | 0.679 | −27.99% | −0.003 | 0.999 |
| HMM (walk-forward, filtered) | 10.08% | 9.93% | 0.655 | **−23.30%** | −0.027 | 0.636 |
| Rule: business cycle | 10.16% | 10.12% | 0.652 | −32.46% | −0.031 | 0.707 |
| Rule: growth × inflation | 8.83% | 9.79% | 0.546 | −40.97% | −0.136 | 0.298 |

The HMM's one real advantage is a smaller drawdown (−23.3% vs −28.5%): it de-risks into contractions without improving risk-adjusted return.

## Why the null is the interesting part

Required sample size for a Sharpe difference follows in closed form from Jobson–Korkie with the Memmel correction:

```
T = [ 2 − 2ρ + ½(S₁² + S₂² − 2·S₁·S₂·ρ²) ] · [ (z₁₋α/₂ + z₁₋β) / δ ]²
```

Two things fall out. Required sample scales with **1/δ²**, so halving the effect quadruples the data needed. And the leading `2(1 − ρ)` means the answer depends heavily on the strategy–benchmark correlation — a parameter almost nobody reports.

**Years needed for 80% power at Δ = 0.10:**

| ρ | Years |
|---|---|
| 0.76 | 610 |
| 0.85 | 382 |
| 0.89 | 281 |
| 0.95 | 128 |

Our five comparisons span ρ = 0.76 to 0.95. Even the most favourable needs 128 years against the 44.6 available.

---

## Three shortcuts that turn out not to matter

A 2×2×2 factorial ablation reintroduces three forms of look-ahead bias — ignoring publication lag (**A**), smoothed instead of filtered decoding (**B**), and full-sample instead of walk-forward fitting (**C**):

| Cell | Sharpe | Δ vs honest | p |
|---|---|---|---|
| none (honest) | 0.655 | — | — |
| A | 0.553 | −0.102 | 0.013 |
| B | 0.685 | +0.030 | 0.036 |
| **ABC** | **0.654** | **−0.001** | 0.998 |

Across 14 tests, two are nominally significant (0.7 expected under the null) and **none survives** Holm–Bonferroni or Benjamini–Hochberg. The fully naive implementation is indistinguishable from the careful one.

This does **not** show look-ahead bias is harmless — power against effects that size was 5–7%. It shows the effort bought no measurable change in conclusions, which is a different and more uncomfortable claim.

---

## What's actually built

Everything is real-time-correct, and the correctness is proven rather than asserted:

- **Point-in-time data.** Every FRED series carries its true publication lag; row *T* contains only what had been released by *T*. An adversarial test corrupts all future observations and requires earlier rows to be bit-identical.
- **Filtered, not smoothed, HMM probabilities.** `predict()` and `predict_proba()` condition on the whole sample. The forward recursion is implemented directly and validated three ways (identity at the terminal step to 2.6e-14, independent linear-space implementation to 4.4e-16, and confirmed to differ elsewhere — 5.4% of months).
- **Expanding-window refitting** — 45 refits, live signal from Dec 1981, with deterministic state relabelling.
- **Honest benchmarks** — a *rebalanced* 60/40 under the same cost model, plus a price-only momentum signal that uses no macro data at all.

## Four things the data contradicted

| Assumption | What testing showed |
|---|---|
| Use `BAMLH0A0HYM2` credit spread | **Gone** — FRED serves only a rolling 3-year window now. Replaced with NFCI (1971) and Baa−Aaa (1919) |
| Four regimes | **Unsupported.** BIC falls monotonically to K=8; seed stability collapses 0.78→0.5. K=3 chosen on stability and episode adequacy |
| ISM PMI from FRED | Withdrawn 2016. CFNAI substituted |
| ETF returns suffice | 2 recessions. Ken French splice takes equities to 1926, ~15 recessions |

---

## Running it

```bash
python -m venv .venv && .venv/Scripts/activate    # Windows
pip install -e .
```

> **Python 3.12 required.** `hmmlearn` ships no wheel for 3.13+, and building from source needs the MSVC C++ toolchain.

Runs fully offline from the committed parquet cache — **no API key needed**:

```bash
python -m macroregime.run                # pipeline and primary results
python -m macroregime.evaluation.power   # power analysis (Paper §7)
python -m macroregime.viz.report         # static HTML report
streamlit run app/dashboard.py           # interactive dashboard
pytest                                   # 48 tests
```

To refresh from source, add a free [FRED key](https://fred.stlouisfed.org/docs/api/api_key.html) to `.env` and pass `--refresh`.

## Layout

```
config/            series registry with publication lags, assets, model params
src/macroregime/
  data/            FRED + yfinance loaders, point-in-time registry, caching
  features/        derived series, expanding standardization, composites
  models/          rule-based classifiers, hysteresis, Gaussian HMM
  evaluation/      NBER validation, conditional performance, ablation, power
  backtest/        walk-forward engine, bootstrap, deflated Sharpe
  viz/             Plotly charts and the static report
app/dashboard.py   Streamlit front end
tests/             look-ahead proofs and published-figure regression
reports/           generated artefacts, including report.html
PAPER.md           the full write-up
```

Worth reading, if you're reading rather than running:

| File | Why |
|---|---|
| `data/registry.py` | The point-in-time join everything rests on |
| `models/hmm.py` | Filtered probabilities, relabelling, expanding refits |
| `evaluation/power.py` | The closed form and its unit convention |
| `tests/test_no_lookahead.py` | The adversarial mutation test |
| `tests/test_units_and_tables.py` | Reconstructs every published figure from its primitives |

## A note on the test suite

Three separate errors in this project were the same error: a scale factor applied in the wrong power — an omitted risk-free rate, a √12 conflation, and a standard-error ratio used where a variance ratio belonged. None was a reasoning failure. All three were invisible to careful reading and trivially catchable by recomputation.

`tests/test_units_and_tables.py` does that recomputation mechanically: it parses the tables out of `PAPER.md` and rebuilds every cell from the underlying data, so the manuscript and the code cannot drift apart silently. It has caught four further errors since being written, including one while this README was being updated.

That is a small argument for the paper's own thesis: the remedy for this class of failure is not more diligence, it's mechanical verification.

---

*Not investment advice. Sources: FRED, Yahoo Finance, Ken French data library.*

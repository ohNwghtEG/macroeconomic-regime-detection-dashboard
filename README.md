# Macroeconomic Regime Detection

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.22737848.svg)](https://doi.org/10.5281/zenodo.22737848)

I built a systematic macro regime model to see whether it beats a static 60/40. It turns out a backtest of realistic length can't answer that question.

Over 44.6 years of point-in-time data, the walk-forward hidden Markov model returns an excess-return Sharpe of **0.655** against a rebalanced 60/40's **0.682** (p = 0.64), and nothing else beats the benchmark either. The more useful part is why that null tells you so little. The study had between **5% and 34% power** against the effects it observed, and detecting a 0.1 Sharpe improvement would take between **128 and 610 years** of monthly data, depending on the correlation between strategy and benchmark.

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

The HMM's one real advantage is a smaller drawdown (−23.3% vs −28.5%). It cuts risk heading into contractions, but its risk-adjusted return is no better.

## Why the null matters

The sample size needed to detect a Sharpe difference has a closed form, from Jobson–Korkie with the Memmel correction:

```
T = [ 2 − 2ρ + ½(S₁² + S₂² − 2·S₁·S₂·ρ²) ] · [ (z₁₋α/₂ + z₁₋β) / δ ]²
```

Required sample scales with **1/δ²**, so halving the effect quadruples the data needed. The leading `2(1 − ρ)` term also makes the answer depend heavily on the correlation between strategy and benchmark, which almost nobody reports.

**Years needed for 80% power at Δ = 0.10:**

| ρ | Years |
|---|---|
| 0.76 | 610 |
| 0.85 | 382 |
| 0.89 | 281 |
| 0.95 | 128 |

My five comparisons span ρ = 0.76 to 0.95. Even the most favourable needs 128 years against the 44.6 available.

---

## Look-ahead shortcuts made no measurable difference

A 2×2×2 factorial ablation puts back three kinds of look-ahead bias: ignoring publication lag (**A**), smoothed instead of filtered decoding (**B**), and full-sample instead of walk-forward fitting (**C**).

| Cell | Sharpe | Δ vs honest | p |
|---|---|---|---|
| none (honest) | 0.655 | — | — |
| A | 0.553 | −0.102 | 0.013 |
| B | 0.685 | +0.030 | 0.036 |
| **ABC** | **0.654** | **−0.001** | 0.998 |

Across 14 tests, two are nominally significant (0.7 would be expected under the null), and none survives Holm–Bonferroni or Benjamini–Hochberg. The fully naive implementation performs about the same as the careful one.

That doesn't show look-ahead bias is harmless, since power against effects of that size was only 5–7%. What it shows is that all the extra care didn't change any conclusion here.

---

## What's actually built

The whole pipeline only uses information that was available at the time, and there are tests that check this:

- **Point-in-time data.** Every FRED series carries its true publication lag, so row *T* contains only what had been released by *T*. An adversarial test corrupts all future observations and requires earlier rows to stay bit-identical.
- **Filtered HMM probabilities.** `predict()` and `predict_proba()` condition on the whole sample, so the forward recursion is implemented directly. It's validated against the terminal-step identity (2.6e-14), against an independent linear-space implementation (4.4e-16), and by checking that it differs from the smoothed output elsewhere (5.4% of months).
- **Expanding-window refitting**: 45 refits, live signal from Dec 1981, with deterministic state relabelling.
- **Fair benchmarks**: a *rebalanced* 60/40 under the same cost model, plus a price-only momentum signal that uses no macro data at all.

## Where the data didn't match my plan

| Assumption | What testing showed |
|---|---|
| Use `BAMLH0A0HYM2` credit spread | FRED now serves only a rolling 3-year window. Replaced with NFCI (1971) and Baa−Aaa (1919) |
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

Everything runs offline from the committed parquet cache, so **no API key is needed**:

```bash
python -m macroregime.run                # pipeline and primary results
python -m macroregime.evaluation.power   # power analysis (Paper §7)
python -m macroregime.viz.report         # static HTML report
streamlit run app/dashboard.py           # interactive dashboard
pytest                                   # 50 tests
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

Three errors in earlier drafts turned out to be the same mistake, a scale factor applied in the wrong power: an omitted risk-free rate, a √12 conflation, and a standard-error ratio used where a variance ratio belonged. Rereading didn't catch any of them, but recomputing the numbers caught all three.

`tests/test_units_and_tables.py` does that recomputation automatically. It parses the tables out of `PAPER.md` and rebuilds every cell from the underlying data, so the paper and the code can't quietly drift apart. It has caught four more errors since I wrote it, one of them while I was updating this README.

---

*Not investment advice. Sources: FRED, Yahoo Finance, Ken French data library.*

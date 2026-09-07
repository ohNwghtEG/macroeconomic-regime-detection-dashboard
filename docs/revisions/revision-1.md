# Underpowered by Construction: Why Backtests Cannot Adjudicate the Strategies They Claim to Test

**Evidence from a fully real-time macroeconomic regime-switching pipeline**

---

*Working paper. Draft of September 2026.*

---

## Abstract

We construct a macroeconomic regime-detection and asset-allocation pipeline to institutional standards — point-in-time data respecting every series' publication lag, forward-filtered rather than smoothed hidden-Markov state probabilities, and expanding-window parameter re-estimation — and use it to ask not whether the strategy works, but whether a backtest of realistic length is capable of answering that question.

It is not. Over 536 months (44.7 years) of monthly data, no regime-based allocation differs significantly from a rebalanced 60/40 benchmark (Sharpe 1.020 vs 1.033, p = 0.86). A full 2×2×2 factorial ablation, in which three distinct forms of look-ahead bias are deliberately reintroduced, produces no significant effect in any of fourteen comparisons; the fully naive implementation scores 1.041 against the honest 1.020, a difference of 0.021. Repeating the ablation with in-sample-optimised regime weights — adding a free parameter through which leaked information might act — does not change this.

The explanation is statistical power. The median 95% confidence interval width on a Sharpe difference in this sample is 0.225, implying a minimum detectable effect of approximately 0.11. Simulation shows that detecting a true Sharpe improvement of 0.10 at 80% power requires **231 years** of monthly data under Gaussian assumptions, and **213 years** when bootstrapped from actual return distributions with their observed fat tails and skew. Our own study attains 23% power against its largest measured effect. A conventional 25-year backtest attains 16%.

The implication is uncomfortable and general. Effect sizes that practitioners treat as decisive — 0.1 to 0.2 Sharpe — sit below the detection threshold of any sample that will exist in our lifetimes. When such studies are filtered for statistical significance before publication, the surviving literature is necessarily dominated by false positives. This applies with equal force to our own null results, which we accordingly decline to interpret as evidence that regime models do not work. The correct conclusion, for us and for the literature we are describing, is that the question has not been answered and cannot be answered by this method.

**Keywords:** regime switching, hidden Markov models, backtesting, statistical power, look-ahead bias, tactical asset allocation, replication

---

## 1. Introduction

Macroeconomic regime models occupy an unusual position in quantitative finance. They are intellectually compelling, institutionally widespread, and — measured by published evidence — remarkably difficult to evaluate.

The premise is sound. Portfolio construction rests on assumptions about expected returns, volatilities and correlations that are demonstrably conditional on the macroeconomic environment rather than fixed. Equity and bond returns were negatively correlated for roughly two decades and then were not, most visibly in 2022, when a portfolio calibrated to the earlier relationship experienced simultaneous losses in both legs. If the joint distribution of asset returns depends on an underlying macroeconomic state, then identifying that state and conditioning allocation upon it is the natural response. Bridgewater's All Weather framework is built explicitly on a growth-versus-inflation state space; multi-asset teams at large institutions run structurally similar overlays.

The empirical record, however, is thin in a specific way. Published regime-switching studies overwhelmingly report favourable results, and the methodological criticism directed at them has concentrated on a particular failure mode: look-ahead bias. The critique is well founded. Economic data is revised, released with a lag, and served by databases in its latest revised form; standard hidden-Markov implementations decode states using the entire sample; and models are routinely estimated once on all available history. Each practice allows information into a backtest that was unavailable at the time, and each is easy to commit without noticing.

This paper takes that critique seriously enough to test it. We construct the pipeline correctly — every series aligned to its actual release date, states inferred by forward filtering only, parameters re-estimated on an expanding window — and then, holding everything else fixed, we deliberately switch each bias back on and measure what it is worth.

The answer is: not much. And in establishing why, we arrive at a result more general than the one we set out to find.

Our contribution is threefold.

**First**, we provide a fully specified, reproducible, real-time regime pipeline as a reference implementation, including a test suite that proves the absence of look-ahead rather than asserting it.

**Second**, we report a factorial ablation of three distinct look-ahead channels, with bootstrap confidence intervals. We find no channel worth more than 0.05 Sharpe, and none statistically distinguishable from zero — a result that runs mildly against the methodological consensus.

**Third**, and most importantly, we explain that null result and show it generalises. We compute the sample length required to detect Sharpe improvements of various sizes and find that the effects the field routinely debates are undetectable in any realistic sample. This reframes the interpretation of the existing literature and, symmetrically, of our own findings.

We are explicit throughout that the third result undermines our own first two. A study with 23% power cannot claim to have shown that look-ahead bias is unimportant. It can only show that the question was not answerable — which is the point.

---

## 2. Related literature

The finding that predictive relationships in finance degrade or vanish out of sample is long established. Welch and Goyal (2008) evaluated a comprehensive set of equity-premium predictors and found that most failed to outperform a simple historical mean out of sample, despite strong in-sample performance. Our headline null result is consistent with that tradition and should be read as a replication within it rather than a novel empirical claim.

Regime-switching methodology descends from Hamilton (1989), whose Markov-switching model of the business cycle established the framework we use. Applications to asset allocation include Ang and Bekaert (2002), who documented regime-dependent international correlations, and Kritzman, Page and Turkington (2012), who reported that regime-aware dynamic strategies improved risk-adjusted performance. It is worth noting that the latter finding is precisely the sort we are unable to confirm or refute at conventional sample lengths.

On the statistics of evaluation, Lo (2002) derived the sampling distribution of the Sharpe ratio and showed that its standard errors are considerably larger than commonly assumed, particularly under non-normality. Ledoit and Wolf (2008) developed robust tests for differences between Sharpe ratios under autocorrelation and fat tails. Our bootstrap approach follows Politis and Romano (1994).

The multiple-testing problem in backtesting has been examined by Harvey, Liu and Zhu (2016) and Harvey and Liu (2015), who argued that conventional significance thresholds are far too permissive given the number of strategies searched. Bailey and López de Prado (2014) formalised this as the Deflated Sharpe Ratio, adjusting for both the number of trials and higher moments.

Our contribution is adjacent to, and complementary with, this last strand. Where the multiple-testing literature asks *"given that many strategies were tried, how should we discount the winner?"*, we ask the prior question: *"given the sample available, could any single comparison have been informative in the first place?"* The answer — that it could not, for effect sizes in the range under debate — implies that multiple-testing corrections are being applied to studies that were uninformative before the correction was considered.

The structural argument is the one Ioannidis (2005) made for the biomedical literature: when statistical power is low and results are filtered for significance before publication, the proportion of published findings that are false rises above one half. We show that backtesting at conventional sample lengths sits deep within that regime.

---

## 3. Pipeline construction

Our purpose in building the pipeline carefully is not to produce a better strategy but to eliminate the obvious explanations for a null result. A finding of "no effect" is uninformative if the implementation is simply weak, so the construction below is deliberately conservative at every choice point.

### 3.1 Data

Twenty-five series were drawn from the Federal Reserve Economic Database (FRED), spanning four dimensions: growth, inflation, financial conditions and monetary policy. Asset returns cover US equities, long Treasuries, commodities, gold and cash.

Two data decisions materially affected the study and are worth recording, because both contradict the specification we began with.

**The conventional credit-spread series is no longer usable.** The ICE BofA US High Yield Option-Adjusted Spread (`BAMLH0A0HYM2`), standard in this literature, is now served by FRED only for a rolling three-year window; as of 5 September 2026 its metadata reports an observation start of 2023-09-05, reflecting a licensing restriction. Three years spans no complete business cycle. We substitute the Chicago Fed National Financial Conditions Index (weekly, from 1971) and the Moody's Baa–Aaa default spread (monthly, from 1919). Similarly, ISM PMI series were withdrawn from FRED in 2016 over licensing; we substitute the Chicago Fed National Activity Index. Researchers replicating older studies should verify series availability rather than assuming it.

**ETF histories are too short for regime-conditional estimation.** SPY begins in 1993, TLT in 2002, GLD in 2004. The four-asset panel therefore spans roughly nineteen years containing two recessions. We splice long-history total-return proxies behind each ETF: the Ken French market factor for equities (from 1926), a duration-based reconstruction from ten-year Treasury yields for bonds, and PPI All Commodities for commodities (from 1913). This extends the equity sample from two recessions to approximately fifteen. Gold remains limited to 2000 onward, as no free bullion series with 1970s coverage survives; gold results are accordingly excluded from the primary analysis.

### 3.2 Point-in-time alignment

Each series is registered with its actual publication lag, and the feature vector for month-end *T* contains only observations whose release date falls on or before *T*:

```
release_date = period_end + publication_lag_days
```

Two details are easy to get wrong and both introduce roughly a month of look-ahead. First, FRED stamps monthly series with the *first* day of the reference month, so March CPI appears at `2008-03-01` though it was published around 16 April. Second, derived series inherit the lag of their *slowest* input: a real policy rate constructed from the fed funds rate (2-day lag) and CPI inflation (16-day lag) is available only on CPI's schedule.

We emphasise that this addresses publication *timing* only, not *revision*. FRED serves today's restated value for historical dates. Full real-time reconstruction requires ALFRED vintages and was not undertaken here; our look-ahead measurements are therefore lower bounds.

### 3.3 Feature construction

Raw levels of series such as the unemployment rate or credit spreads are non-stationary, and a Gaussian emission model supplied with them will learn time periods rather than economic conditions. All series are transformed to approximately stationary form and standardised using **expanding-window** z-scores, where the mean and standard deviation at each date derive only from history to that date. A full-sample z-score is a subtle but complete look-ahead violation: it embeds the eventual distribution into every historical observation.

Features are aggregated into four equal-weighted composites with signs aligned so that each reads "higher equals more of the named quantity."

### 3.4 Regime models

Three models are estimated. Two are rule-based, and we distinguish them deliberately because the literature frequently conflates them: a **growth × inflation quadrant** model (a state space over the direction of macroeconomic surprises, in the Bridgewater tradition) and a **business-cycle phase** model (a sequence: expansion, slowdown, recession, recovery). Both employ a ±0.25σ deadband and two-month confirmation to suppress whipsaw.

The third is a Gaussian hidden Markov model, subject to three requirements that constitute the paper's methodological core:

1. **Filtered, not smoothed, probabilities.** The standard `predict()` and `predict_proba()` interfaces return quantities conditioned on the entire sample. Real-time inference requires the forward filter, `P(s_t | data ≤ t)`. We implement the recursion directly in log space and validate it three ways: against the identity that filtered and smoothed probabilities must coincide at the terminal observation (agreement to 2.6 × 10⁻¹⁴), against an independent linear-space implementation (4.4 × 10⁻¹⁶), and by confirming they differ elsewhere. On our data they disagree on 5.4% of months.

2. **Expanding-window re-estimation.** Parameters are re-estimated every twelve months on data available to that point — 45 refits, with the live signal beginning December 1981.

3. **Deterministic state relabelling.** Hidden states are unidentified; the "state 0" of one refit bears no relation to another's. States are sorted by mean growth loading after every fit.

### 3.5 Selecting the number of states

We had intended four states, matching the economic narrative. The data does not support this, and how it fails is instructive:

- **BIC declines monotonically** across the entire grid tested, from 2811 at K = 2 to 1804 at K = 8. It never turns, and therefore selects nothing; it rewards parameters.
- **Held-out likelihood is non-monotonic and unstable**, because the held-out tail contains the COVID period. Whichever K best accommodates that single episode wins, for reasons unrelated to regime structure.
- **Seed-to-seed stability collapses.** Measured by adjusted Rand index across random initialisations, agreement falls from 0.78 at K = 3 to approximately 0.5 at K = 4–6. Different starting values produce different regime histories.
- **Episode adequacy fails.** At K ≥ 4, states are entered as few as four times. The effective sample for conditional estimation is the number of *episodes*, not months.

We therefore select K = 3 on stability and episode adequacy. The recovered states are economically legible, including one with growth −0.51σ and inflation +0.97σ that isolates the stagflationary 1970s.

### 3.6 Backtest protocol

Signals formed at month-end *T* are traded at *T+1*. Costs of 10 bps are charged on turnover. Allocation is probability-weighted (`w = Σ p_k w_k`) rather than switched on the argmax. The benchmark is a **rebalanced** 60/40 carrying the identical cost model; comparison against a buy-and-hold benchmark would credit rebalancing itself as alpha.

Regime weights are hand-specified and not optimised, so that the primary results contain no fitted allocation parameters.

---

## 4. Primary results

### 4.1 Strategy performance

Over 1981–2026 (536 months), on equities, bonds and commodities:

| Strategy | Ann. return | Ann. vol | Sharpe | Max DD | Δ vs 60/40 | p |
|---|---|---|---|---|---|---|
| Momentum (no macro data) | 10.07% | 9.52% | 1.060 | −27.99% | +0.027 | 0.845 |
| **60/40 (rebalanced)** | 10.61% | 10.32% | **1.033** | −28.45% | — | — |
| HMM (walk-forward, filtered) | 10.08% | 9.93% | 1.020 | **−23.30%** | −0.013 | 0.857 |
| Rule: business cycle | 10.16% | 10.12% | 1.010 | −32.46% | −0.023 | 0.795 |
| Rule: growth × inflation | 8.83% | 9.79% | 0.917 | −40.97% | −0.116 | 0.397 |

No strategy differs significantly from the benchmark. Every macro model ranks below it on Sharpe, though the HMM achieves a materially smaller maximum drawdown (−23.3% vs −28.5%), suggesting that its principal effect is de-risking into contractions rather than return enhancement.

A price-only momentum signal using no macroeconomic data ranks highest. It is not significantly superior to the benchmark, so this is a ranking rather than a demonstrated edge; but it is consistent with the view that markets incorporate regime information before macroeconomic data confirms it.

### 4.2 Three ancillary findings

**Sample length determines the sign.** On a 25-year sample the business-cycle model outperforms 60/40 by +0.158 Sharpe. Extended to 55 years the same model underperforms by −0.030. A shorter study would have reported the opposite conclusion with equal confidence.

**The model is a poor recession classifier.** Against NBER dates it achieves recall of 0.906 but precision of only 0.326, producing 159 false-positive months across 27 distinct episodes. The real-time Sahm rule — a single line of arithmetic — attains a higher F1 (0.525 vs 0.480).

**Conditional correlations are more stable than the framework assumes.** Equity–bond correlation across regimes ranges only from 0.02 to 0.17 against an unconditional 0.08. Since regime-conditional allocation is largely motivated by correlation instability, this weakens the mechanism in this sample.

---

## 5. Experiment 1 — Look-ahead ablation

We re-ran the pipeline with three biases reintroduced in a full 2×2×2 factorial design:

- **A** — publication lag ignored (series aligned to reference-period end)
- **B** — smoothed decoding over the entire series
- **C** — single full-sample parameter estimation

All eight cells are evaluated on an identical 536-month sample with identical assets, weights, costs and trade lag. Sample alignment is essential and easily overlooked: full-sample fitting emits a signal from the first observation while expanding-window cells begin only after burn-in, a gap spanning the entire Volcker disinflation. Unaligned, the design confounds the bias under test with the sample period.

| Cell | Biases | Sharpe | Δ vs honest | 95% CI | p |
|---|---|---|---|---|---|
| none (honest) | 0 | 1.020 | — | — | — |
| B | 1 | 1.052 | +0.032 | [+0.003, +0.069] | 0.061 |
| C | 1 | 0.990 | −0.031 | [−0.141, +0.083] | 0.594 |
| A | 1 | 0.916 | −0.105 | [−0.228, −0.015] | 0.052 |
| BC | 2 | 1.027 | +0.006 | [−0.104, +0.122] | 0.917 |
| AC | 2 | 1.001 | −0.020 | [−0.127, +0.091] | 0.734 |
| AB | 2 | 0.981 | −0.040 | [−0.101, +0.022] | 0.200 |
| **ABC** | 3 | **1.041** | **+0.020** | [−0.087, +0.135] | 0.721 |

**No cell is significant.** The fully naive implementation exceeds the honest one by 0.020 Sharpe — approximately one-tenth of the confidence interval width.

Two observations. Ignoring publication lag *degraded* performance (−0.105), plausibly because fresher data produces a noisier, more whipsaw-prone signal while regimes are persistent enough that arriving two weeks earlier confers little advantage. And a factorial decomposition yields no main effect exceeding 0.044 Sharpe.

---

## 6. Experiment 2 — The free-parameter hypothesis

We hypothesised that look-ahead bias is inert under fixed weights because improved classification cannot express itself through a coarse pre-committed mapping, but would become material once an optimiser could exploit the leaked information. We therefore repeated the ablation with per-regime weights fitted in-sample by long-only maximum-Sharpe optimisation.

| Weighting scheme | Honest | Fully naive (ABC) | Inflation |
|---|---|---|---|
| Fixed | 1.020 | 1.041 | +0.021 |
| Optimised in-sample | 1.117 | 1.114 | −0.003 |

**The hypothesis is not supported.** Adding the free parameter did not amplify the biases. The cross-cell spread widened (0.136 → 0.227) without systematic direction.

The likely explanation is that in every cell the optimiser already observes full-sample *returns*; leaking regime *identity* adds little beyond this, and a mapping of three regimes onto three assets offers limited scope for exploitation. We report the negative result rather than discarding it, and note that a richer asset universe might yield a different answer.

In-sample optimisation itself raised Sharpe from 1.020 to 1.117 — a gain of 0.096, larger than any look-ahead channel, though itself not significant (p = 0.408).

---

## 7. Experiment 3 — Statistical power

The preceding null results admit two interpretations: the effects are genuinely small, or the study cannot see them. We now show it is the latter, and that this holds far more generally.

The median 95% confidence interval width on a Sharpe difference in our sample is **0.225**, implying a minimum detectable effect of approximately 0.11.

We estimate power by simulation, generating paired strategy and benchmark return series with a known true Sharpe difference and measuring detection frequency. Two specifications are used: Gaussian returns, and a stationary block bootstrap of *actual* 60/40 monthly returns, which preserves the observed fat tails (excess kurtosis +1.59) and left skew (−0.28). Strategy and benchmark are correlated at 0.85, a generous assumption that *increases* power through pairing.

**Power to detect a true Sharpe improvement:**

| Sample | Δ = 0.05 | Δ = 0.10 | Δ = 0.20 | Δ = 0.30 | Δ = 0.50 |
|---|---|---|---|---|---|
| 25 years | 8% | 16% | 48% | 82% | 100% |
| 45 years | 9% | 25% | 73% | 97% | 100% |
| 60 years | 12% | 31% | 84% | 99% | 100% |
| 100 years | 16% | 50% | 97% | 100% | 100% |
| 250 years | 33% | 85% | 100% | 100% | 100% |

**Years required for 80% power:**

| True Δ Sharpe | Gaussian | Real-return bootstrap |
|---|---|---|
| 0.05 | 930 years | 858 years |
| 0.10 | 231 years | 213 years |
| 0.20 | 58 years | 55 years |
| 0.30 | 27 years | 25 years |
| 0.50 | 10 years | 10 years |

The two specifications agree closely, so the result does not rest on a normality assumption. Notably the empirical bootstrap requires slightly *fewer* years than the Gaussian case; this reflects the pairing construction rather than real returns being easier to work with, and the difference is within simulation error.

All figures are Monte Carlo estimates with simulation error of roughly one to two percentage points in the power tables, and a few percent in the years-required column. They are reproducible via `python -m macroregime.evaluation.power`.

Our study, at 44.7 years, has **23% power** against its largest measured effect (Δ = 0.096) and **9%** against the look-ahead effects (Δ ≈ 0.05). A conventional 25-year backtest has **16% power** to detect a 0.10 Sharpe improvement.

---

## 8. Implications

### 8.1 For interpreting the existing literature

A 0.1 to 0.2 Sharpe improvement is the range in which tactical allocation overlays are typically evaluated, funded, and marketed. Detecting the lower end at conventional confidence requires roughly two centuries of monthly data (213-231 years). No such sample exists, and none will.

The consequence follows from Ioannidis's argument. When power is low, the ratio of true to false positives among *statistically significant* results deteriorates sharply, because a low-powered study rarely detects real effects but retains its full 5% false-positive rate. Add publication filtering — favourable backtests are written up, unfavourable ones are not — and the surviving literature is dominated by false positives, irrespective of the integrity of any individual researcher.

This is a structural claim, not an accusation. It requires no misconduct, no data mining, and no methodological error. It follows from sample sizes and effect sizes alone.

### 8.2 For methodological priorities

Considerable effort in this literature — including a substantial portion of our own — is directed at eliminating look-ahead bias. Our ablation found that the three principal channels are jointly worth approximately 0.02 Sharpe, well below the detection threshold.

We are careful about what this licenses. It does **not** show that look-ahead bias is unimportant; our power against such effects was 9%. It shows that the effort spent eliminating it purchased no measurable change in conclusions here, and that a study confident it has removed look-ahead bias should not therefore be confident its results are informative. Methodological rigour of this kind is necessary but nowhere near sufficient.

The larger threats are those that operate through *selection*: which sample period, which assets, which specification, which of many attempted variants gets reported. Our own sample-length result — a swing of 0.19 Sharpe, sufficient to reverse the sign, from a defensible choice about start date — moved the answer more than every look-ahead channel combined.

### 8.3 For practice

Three recommendations follow.

**Report power, not merely significance.** A backtest should state the effect size it could have detected. A null result at 16% power carries almost no information, and reporting it as evidence of absence is an error.

**Treat sample-period choice as a first-order specification decision.** It should be pre-committed and its sensitivity reported, since it can exceed the effect under study.

**Prefer hypotheses with larger expected effects.** If 0.3 Sharpe is detectable in 25 years and 0.1 is not detectable in a century, then research should concentrate where effects are large enough to be seen. Strategies whose expected edge is 0.1 Sharpe may be real and may be worth deploying, but they cannot be *validated* by backtesting, and should be justified on other grounds.

---

## 9. Limitations

We state these plainly, as several bear directly on the paper's own claims.

**Our null results are underpowered, including the ones we like.** We cannot conclude that regime models fail to add value, nor that look-ahead bias is immaterial. The honest reading of Sections 4 through 6 is that the question was not answerable at this sample size. Interpreting our nulls as findings would repeat the error the paper describes.

**Revisions are not addressed.** Switch A captures publication timing only, not data revision. Full real-time reconstruction requires ALFRED vintages. Since revisions are largest at turning points, our look-ahead estimates are lower bounds.

**Regime weights are hand-specified in the primary analysis.** The null result is for these weights. Section 6 explores fitted weights, but the space of possible regime-conditional allocations is not exhausted.

**Proxy series are approximations.** Bond returns are duration-reconstructed, ignoring convexity and roll-down. The commodity proxy is a price index rather than an investable return. Gold lacks 1970s coverage entirely and is excluded from the primary analysis, which removes the asset most associated with the stagflation regime.

**One asset universe, one frequency, one economy.** Three to five assets, monthly, United States. A richer universe might permit regime information to express itself more effectively, and Section 6 is the natural place that would appear.

**Power analysis assumes a stable true effect.** If the genuine advantage of regime allocation is concentrated in rare episodes rather than distributed evenly, our simulation understates detectability for that pattern — though it correspondingly raises the sample requirement for a stable estimate.

**The power argument is not novel in kind.** Harvey and Liu (2015), Harvey, Liu and Zhu (2016), and Bailey and López de Prado (2014) advance closely related arguments about multiple testing and backtest overfitting. Our contribution is the explicit sample-length requirement, its application to a complete reproducible pipeline, and its reflexive application to our own results.

---

## 10. Conclusion

We built a macroeconomic regime model to a standard we believe exceeds normal practice: point-in-time data, forward-filtered state inference, walk-forward re-estimation, adversarial tests proving the absence of look-ahead, and stability-based rather than likelihood-based model selection. We then attempted to determine whether that care mattered.

It did not measurably matter, and neither did the strategy. The fully naive implementation performs indistinguishably from the rigorous one, and both perform indistinguishably from a rebalanced 60/40.

The explanation is not that regime models work, nor that they fail, but that a backtest of realistic length cannot tell us which. Detecting the effect sizes at issue would require centuries of data. Our study, at 44.7 years and by construction one of the more carefully assembled examples of its kind, had roughly one chance in four of detecting the largest effect it measured.

We think this is the more useful finding, and we hold it symmetrically. A field that debates 0.1 Sharpe differences using 25-year samples is not conducting weak empirical work; it is conducting empirical work whose central quantity is not estimable from the available evidence. The appropriate response is not better hygiene — we applied the best hygiene we know and it changed nothing — but a change in which questions are asked, and considerably more humility about which have been answered.

Including ours.

---

## References

*Citations require verification against original sources before submission; volume and page details in particular have not been independently checked.*

Ang, A., & Bekaert, G. (2002). International asset allocation with regime shifts. *Review of Financial Studies*, 15(4), 1137–1187.

Bailey, D. H., & López de Prado, M. (2014). The deflated Sharpe ratio: Correcting for selection bias, backtest overfitting, and non-normality. *Journal of Portfolio Management*, 40(5), 94–107.

Hamilton, J. D. (1989). A new approach to the economic analysis of nonstationary time series and the business cycle. *Econometrica*, 57(2), 357–384.

Harvey, C. R., & Liu, Y. (2015). Backtesting. *Journal of Portfolio Management*, 42(1), 13–28.

Harvey, C. R., Liu, Y., & Zhu, H. (2016). …and the cross-section of expected returns. *Review of Financial Studies*, 29(1), 5–68.

Ioannidis, J. P. A. (2005). Why most published research findings are false. *PLoS Medicine*, 2(8), e124.

Kritzman, M., Page, S., & Turkington, D. (2012). Regime shifts: Implications for dynamic strategies. *Financial Analysts Journal*, 68(3), 22–39.

Ledoit, O., & Wolf, M. (2008). Robust performance hypothesis testing with the Sharpe ratio. *Journal of Empirical Finance*, 15(5), 850–859.

Lo, A. W. (2002). The statistics of Sharpe ratios. *Financial Analysts Journal*, 58(4), 36–52.

Newey, W. K., & West, K. D. (1987). A simple, positive semi-definite, heteroskedasticity and autocorrelation consistent covariance matrix. *Econometrica*, 55(3), 703–708.

Politis, D. N., & Romano, J. P. (1994). The stationary bootstrap. *Journal of the American Statistical Association*, 89(428), 1303–1313.

Welch, I., & Goyal, A. (2008). A comprehensive look at the empirical performance of equity premium prediction. *Review of Financial Studies*, 21(4), 1455–1508.

---

## Appendix A — Reproducibility

All results are generated by a single command from a committed data cache and require no API credentials:

```bash
python -m macroregime.run          # pipeline and primary results
python -m macroregime.viz.report   # static HTML report
python -m macroregime.evaluation.power   # Section 7 power analysis
pytest                             # 29 tests, including look-ahead proofs
```

Environment: Python 3.12, `hmmlearn` 0.3.3, `statsmodels` 0.15.0, `numpy` 2.5.2, `pandas` 3.0.5. Random seeds are fixed throughout; HMM estimation uses 20 initialisations selected by log-likelihood.

### A.1 Look-ahead test suite

The absence of look-ahead is proven rather than asserted. The central test is adversarial: every observation after a cutoff is replaced with implausible values, the feature matrix is rebuilt, and all earlier rows are required to be bit-identical. A complementary inverse test deliberately removes the trade lag and requires performance to *improve*, confirming the lag is binding rather than decorative.

### A.2 Generated artefacts

| File | Contents |
|---|---|
| `ablation.csv` | Eight-cell factorial results |
| `ablation_bootstrap.csv` | Bootstrap confidence intervals per cell |
| `ablation_weight_modes.csv` | Fixed vs optimised weighting comparison |
| `summary_long.csv`, `significance_long.csv` | Primary backtest and significance tests |
| `vs_sahm.csv`, `nber_lead_lag.csv` | Recession-classification validation |
| `staleness.csv` | Per-series publication lag at decision points |
| `power_gaussian.csv`, `power_empirical.csv` | Power by sample length and effect size |
| `power_years_required.csv` | Sample length required for 80% power |

# Underpowered by Construction: Sharpe Differences of 0.1 Are Not Estimable from Realistic Samples

**Evidence from a fully real-time macroeconomic regime-switching pipeline**

**Ethan Gao**
*Independent researcher*

---

*Working paper. Revision 5, September 2026.*

Code, data and the test suite that reproduces every figure in this paper:
`https://github.com/ohNwghtEG/macroeconomic-regime-detection-dashboard`

Code is released under the MIT License; this manuscript under CC BY 4.0.

---

## Abstract

We construct a macroeconomic regime-detection and asset-allocation pipeline to institutional standards — point-in-time data respecting every series' publication lag, forward-filtered rather than smoothed hidden-Markov state probabilities, and expanding-window parameter re-estimation — and use it to ask not whether the strategy works, but whether a backtest of realistic length is capable of answering that question.

For effect sizes in the range the field actually debates, it is not. Over 535 months (44.6 years), no regime-based allocation differs significantly from a rebalanced 60/40 benchmark; the walk-forward hidden Markov model returns an excess-return Sharpe of 0.655 against the benchmark's 0.682 (Δ = −0.027, p = 0.64). A full 2×2×2 factorial ablation, in which three distinct forms of look-ahead bias are deliberately reintroduced, yields no effect surviving correction across fourteen tests; the fully naive implementation scores 0.654 against the honest 0.655, a difference of −0.001. Repeating the ablation with in-sample-optimised regime weights does not change this.

The explanation is statistical power, and we derive it in closed form. Using the Jobson–Korkie statistic with the Memmel correction, the sample length required to detect a true Sharpe difference Δ at 80% power is a simple function of Δ, the benchmark Sharpe, and the correlation ρ between the strategies. That last parameter dominates and is almost never reported: across the five comparisons in our own study ρ ranges from 0.76 to 0.95, and the sample required to detect Δ = 0.10 ranges correspondingly from **610 years to 128 years**. Even the most favourable case exceeds any available history by a factor of nearly three.

Two artifacts are offered for reuse. The first is a per-comparison detectability table (Section 7.4). Our study's minimum detectable effect at 80% power ranges from **0.18 to 0.36 Sharpe** across its five comparisons, against observed effects of 0.003 to 0.153, giving achieved power of **5% to 34%**. We suggest reporting that table — minimum detectable effect alongside observed effect, per comparison — as standard practice, since it is what makes a null result interpretable. The second is a test suite (Appendix A.2) that mechanically reconstructs every published figure from its primitives. Three separate errors in this paper's own drafts were a single error — a scale factor applied in the wrong power — and all three were invisible to careful reading and trivially catchable by recomputation. That is itself evidence for the paper's argument: the remedy for this class of failure is not more diligence but mechanical verification.

The implication is bounded but consequential. Sharpe differences of 0.3 and above are detectable within a career only at high ρ (15 years at ρ = 0.95, but 69 at ρ = 0.76). Differences of 0.1 to 0.2 — the range in which tactical allocation overlays are evaluated, funded and marketed — require between three decades and six centuries, and below ρ = 0.9 are effectively out of reach. When studies in that range are filtered for statistical significance before publication, the surviving literature is dominated by false positives. This applies with equal force to our own null results, which we accordingly decline to interpret as evidence that regime models do not work.

**Keywords:** regime switching, hidden Markov models, backtesting, statistical power, Sharpe ratio inference, look-ahead bias, tactical asset allocation

---

## 1. Introduction

Macroeconomic regime models occupy an unusual position in quantitative finance. They are intellectually compelling, institutionally widespread, and — measured by published evidence — remarkably difficult to evaluate.

The premise is sound. Portfolio construction rests on assumptions about expected returns, volatilities and correlations that are demonstrably conditional on the macroeconomic environment rather than fixed. Equity and bond returns were negatively correlated for roughly two decades and then were not, most visibly in 2022. If the joint distribution of asset returns depends on an underlying macroeconomic state, then identifying that state and conditioning allocation upon it is the natural response. Bridgewater's All Weather framework is built explicitly on a growth-versus-inflation state space; multi-asset teams at large institutions run structurally similar overlays.

Methodological criticism of this literature has concentrated on look-ahead bias. The critique is well founded: economic data is revised, released with a lag, and served by databases in its latest revised form; standard hidden-Markov implementations decode states using the entire sample; and models are routinely estimated once on all available history.

This paper takes that critique seriously enough to test it. We construct the pipeline correctly and then, holding everything else fixed, switch each bias back on and measure what it is worth. The measured effects are small and do not survive multiple-testing correction. But the more important finding is why: at the effect sizes involved, the experiment was never capable of resolving the question.

Our contribution is threefold.

**First**, a fully specified, reproducible real-time regime pipeline, including a test suite that proves the absence of look-ahead rather than asserting it.

**Second**, a factorial ablation of three look-ahead channels with bootstrap confidence intervals and multiple-testing correction, finding no channel that survives.

**Third**, and principally, a closed-form expression for the sample length required to detect a given Sharpe difference, validated against simulation, together with its application to our own results. This converts a case-specific null into a general and reusable statement about what backtesting can and cannot establish.

**The central result, stated up front.** For two strategies with per-period Sharpe ratios *S*₁, *S*₂ and correlation ρ, the sample size required to detect a true per-period difference δ at level α and power 1 − β is

> **T = [ 2 − 2ρ + ½(S₁² + S₂² − 2·S₁·S₂·ρ²) ] · [ (z₁₋α/₂ + z₁₋β) / δ ]²**

Required sample scales with 1/δ², and the leading term 2(1 − ρ) means the answer depends critically on a parameter the literature rarely reports. At our benchmark Sharpe of 0.68 and 80% power, detecting Δ = 0.10 requires 128 years at ρ = 0.95 and 610 years at ρ = 0.76. Sections 3 to 6 present the pipeline and ablation that motivated this; a reader interested only in the methodological result may go directly to Section 7.

We are explicit that this result constrains our own first two contributions. A study with 5–34% power cannot claim to have shown that look-ahead bias is unimportant.

### 1.1 Note on this revision

An earlier draft reported performance ratios computed on **raw** rather than excess returns. Over this sample the three-month Treasury bill averaged 3.65% annualised — 7.80% during the 1980s portion — so those figures overstated risk-adjusted performance by roughly 0.35 and, because the Sharpe ratio divides by each strategy's own volatility, distorted the differences between strategies as well. All results here are computed on excess returns; the earlier quantities are retained in the output files under the name `return_vol_ratio`, which is what they were.

Two conclusions changed materially. The price-momentum benchmark no longer outranks 60/40 (Section 4.3), which removes the support an earlier draft claimed for the "markets price regimes first" interpretation. And the specification-sensitivity result is smaller than first reported and is partly attributable to asset-set choice rather than sample window (Section 4.2).

Revision 3 corrects a further error of the same species as the risk-free omission: quantities computed under **different parameterisations were compared as though they were not**. The closed-form standard error, the Monte Carlo simulation and the empirical bootstrap were each evaluated at a different correlation ρ (0.90, 0.85 and 0.90 respectively) and on two different samples, and the residual between them was reported as "tail inflation." With ρ, *S*₁, *S*₂ and *T* pinned to identical values, two things follow. The Monte Carlo reproduces the closed form to within 1%, so it validates the algebra rather than providing independent corroboration — an earlier claim that "three independent methods agree" was wrong. And the genuine non-normality inflation, measured consistently, is **1.25×** (range 1.15–1.34), somewhat larger than the 1.17× previously reported. Section 7 is rewritten accordingly. That this paper had to make its own Section 7.1 recommendation — report ρ, do not assume it — before it could get its own arithmetic right is not lost on us.

Revision 4 corrects a third instance of the same failure mode. Section 7.3 applied the measured inflation factor of 1.248 linearly to required sample size, when required sample scales with **variance** and the factor is a ratio of **standard errors**; the correct multiplier is 1.248² = 1.558. All Section 7.3 figures rise by 25%. Three errors — an omitted risk-free rate, a √12 conflation, and a squared-versus-linear ratio — have now been the same error: a scale factor applied in the wrong power, in each case invisible to reading and trivially catchable by recomputation. We have accordingly added a unit convention (Section 7.1) and a test suite that reconstructs every published table cell from its primitives (`tests/test_units_and_tables.py`), rather than relying on further care.

---

## 2. Related literature

The finding that predictive relationships in finance degrade or vanish out of sample is long established. Welch and Goyal (2008) evaluated a comprehensive set of equity-premium predictors and found most failed to outperform a historical mean out of sample despite strong in-sample performance. Our headline null is a replication within that tradition rather than a novel empirical claim.

Regime-switching methodology descends from Hamilton (1989). Applications to allocation include Ang and Bekaert (2002) on regime-dependent international correlations and Kritzman, Page and Turkington (2012), who reported that regime-aware dynamic strategies improved risk-adjusted performance. The latter is precisely the class of finding we argue is not resolvable at conventional sample lengths — in either direction.

On inference, Jobson and Korkie (1981) derived the asymptotic distribution of the difference between two Sharpe ratios; Memmel (2003) corrected an error in their variance expression. That corrected expression is the analytic backbone of Section 7. Lo (2002) showed that Sharpe ratio standard errors are substantially larger than commonly assumed under non-normality, and Ledoit and Wolf (2008) developed robust tests under autocorrelation and fat tails. Our bootstrap follows Politis and Romano (1994).

The multiple-testing problem in backtesting has been examined by Harvey, Liu and Zhu (2016) and Harvey and Liu (2015); Bailey and López de Prado (2014) formalised it as the Deflated Sharpe Ratio.

Our contribution is adjacent and complementary. Where that literature asks *"given that many strategies were tried, how should we discount the winner?"*, we ask the prior question: *"given the sample available, could a single comparison have been informative at all?"* For Δ ≥ 0.3 the answer is yes, and multiple-testing correction is the binding constraint. For Δ ≈ 0.1 the answer is no, and corrections are being applied to comparisons that were uninformative before correction was considered.

The structural argument is Ioannidis's (2005): when power is low and results are filtered for significance, the proportion of published findings that are false rises. We locate backtesting at conventional sample lengths within that regime, for a specific and identifiable range of effect sizes.

---

## 3. Pipeline construction

Our purpose in building carefully is not to produce a better strategy but to eliminate alternative explanations for a null. Every choice below is deliberately conservative.

### 3.1 Data

Twenty-five FRED series span growth, inflation, financial conditions and monetary policy. Two data decisions contradict the specification we began with and are worth recording.

**The conventional credit-spread series is no longer usable.** The ICE BofA US High Yield Option-Adjusted Spread (`BAMLH0A0HYM2`) is now served by FRED only for a rolling three-year window; as of 5 September 2026 its metadata reports an observation start of 2023-09-05, reflecting a licensing restriction. We substitute the Chicago Fed National Financial Conditions Index (weekly, from 1971) and the Moody's Baa–Aaa default spread (monthly, from 1919). ISM PMI series were similarly withdrawn in 2016; we substitute the Chicago Fed National Activity Index. Replications of older studies should verify availability rather than assume it.

**ETF histories are too short.** SPY begins in 1993, TLT in 2002, GLD in 2004 — two recessions between them. We splice long-history total-return proxies behind each ETF: the Ken French market factor for equities (from 1926), a duration-based reconstruction from ten-year yields for bonds, and PPI All Commodities for commodities (from 1913). Gold remains limited to 2000; no free bullion series with 1970s coverage survives, so gold is excluded from the primary analysis.

### 3.2 Point-in-time alignment

Each series carries its actual publication lag, and the feature vector for month-end *T* contains only observations released on or before *T*:

```
release_date = period_end + publication_lag_days
```

Two details are easy to get wrong, each worth roughly a month of look-ahead. FRED stamps monthly series with the *first* day of the reference month, so March CPI appears at `2008-03-01` though published around 16 April. And derived series inherit the lag of their *slowest* input.

This addresses publication *timing* only, not *revision*. Full real-time reconstruction requires ALFRED vintages and was not undertaken; our look-ahead estimates are lower bounds.

### 3.3 Features

Series are transformed to approximately stationary form and standardised with **expanding-window** z-scores, whose mean and standard deviation at each date derive only from prior history. A full-sample z-score embeds the eventual distribution into every historical observation.

### 3.4 Regime models

Two rule-based models are estimated, deliberately distinguished because the literature conflates them: a **growth × inflation quadrant** model (a state space over the direction of macroeconomic surprises) and a **business-cycle phase** model (a sequence). Both use a ±0.25σ deadband and two-month confirmation.

The third is a Gaussian hidden Markov model subject to three requirements:

1. **Filtered, not smoothed, probabilities.** Standard `predict()` and `predict_proba()` interfaces condition on the entire sample. We implement the forward recursion directly in log space, validated three ways: against the identity that filtered and smoothed must coincide at the terminal observation (agreement to 2.6 × 10⁻¹⁴), against an independent linear-space implementation (4.4 × 10⁻¹⁶), and by confirming they differ elsewhere. They disagree on 5.4% of months.
2. **Expanding-window re-estimation** — 45 refits, live signal from December 1981.
3. **Deterministic relabelling** by mean growth loading after every fit, since hidden states are otherwise unidentified across refits.

### 3.5 Selecting the number of states

We intended four states. The data does not support this:

- **BIC declines monotonically** from 2811 at K = 2 to 1804 at K = 8, never turning.
- **Held-out likelihood is non-monotonic**, dominated by whether a given K accommodates COVID.
- **Seed-to-seed stability collapses**: adjusted Rand index falls from 0.78 at K = 3 to ≈0.5 at K = 4–6.
- **Episode adequacy fails**: at K ≥ 4 some states are entered only four times, and the effective sample for conditional estimation is episodes, not months.

We select K = 3 on stability and episode adequacy.

A caveat on the BIC behaviour, which we initially misdiagnosed as the penalty simply being too weak. Monotonically declining BIC across an entire grid is a recognised symptom of **emission misspecification**: when a Gaussian emission cannot represent the conditional distribution, additional states are recruited to approximate a non-Gaussian shape, and the likelihood gain outruns the penalty indefinitely. Given the fat tails documented in Section 7, this is the likely mechanism. It implies the number of states is not identified by information criteria under this emission family, and that a *t*-distributed or mixture emission is the natural next specification. We did not pursue it, and flag it as an open limitation.

### 3.6 Backtest protocol

Signals formed at month-end *T* are traded at *T+1*. Costs of 10 bps are charged on turnover. Allocation is probability-weighted rather than switched on the argmax. The benchmark is a **rebalanced** 60/40 carrying the identical cost model. Regime weights are hand-specified, so the primary results contain no fitted allocation parameters.

**All performance ratios are Sharpe ratios on excess returns**, using the three-month Treasury bill as the risk-free rate.

---

## 4. Primary results

### 4.1 Strategy performance

December 1981 – August 2026 (535 months, 44.6 years), on equities, bonds and commodities:

| Strategy | Ann. return | Ann. vol | **Sharpe** | Max DD | ρ vs 60/40 | Δ vs 60/40 | 95% CI | p |
|---|---|---|---|---|---|---|---|---|
| **60/40 (rebalanced)** | 10.61% | 10.32% | **0.682** | −28.45% | — | — | — | — |
| Momentum (no macro data) | 10.07% | 9.52% | 0.679 | −27.99% | 0.758 | −0.003 | [−0.256, +0.233] | 0.999 |
| HMM (walk-forward, filtered) | 10.08% | 9.93% | 0.655 | **−23.30%** | 0.952 | −0.027 | [−0.150, +0.097] | 0.636 |
| Rule: business cycle | 10.16% | 10.12% | 0.652 | −32.46% | 0.889 | −0.031 | [−0.197, +0.131] | 0.707 |
| Rule: growth × inflation | 8.83% | 9.79% | 0.546 | −40.97% | 0.804 | −0.136 | [−0.377, +0.110] | 0.298 |
| Equal weight | 7.78% | 7.94% | 0.529 | −32.62% | 0.866 | −0.153 | [−0.346, +0.040] | 0.121 |

*Note: confidence intervals and p-values derive from the same stationary-bootstrap distribution (5000 iterations, 12-month mean block length). The p-value is computed as 2·min(P(Δ* ≤ 0), P(Δ* ≥ 0)) over the bootstrap replicates, so an interval excluding zero and p < 0.05 are equivalent by construction. An earlier draft computed the interval by percentiles but the p-value from a separately mean-centred distribution, which produced cells whose interval excluded zero alongside p > 0.05 — an incoherence we have removed.*

No strategy differs significantly from the benchmark, and every macro model ranks below it. The HMM achieves a materially smaller maximum drawdown (−23.3% vs −28.5%), suggesting its principal effect is de-risking into contractions rather than return enhancement — the one dimension on which it plausibly earns its complexity.

### 4.2 Specification sensitivity

We promote this from a secondary observation because it is the most immediately persuasive demonstration of the paper's thesis. All rows use the **business-cycle rule model** against the rebalanced 60/40 benchmark; only the sample start and asset set vary.

| Sample start | Asset set | Years | Business-cycle rule | 60/40 | Δ | p |
|---|---|---|---|---|---|---|
| 2001 | 4-asset (incl. gold) | 24.8 | 0.729 | 0.596 | **+0.133** | 0.393 |
| 1996 | 3-asset | 30.5 | 0.580 | 0.629 | −0.050 | 0.666 |
| 1990 | 3-asset | 36.5 | 0.617 | 0.685 | −0.067 | 0.513 |
| 1981 | 3-asset | 45.5 | 0.613 | 0.638 | −0.025 | 0.757 |
| 1970 | 3-asset | 56.5 | 0.477 | 0.510 | −0.033 | 0.639 |

The extremes differ by 0.166 Sharpe and carry opposite signs. A researcher reporting only the first row would describe a successful strategy; one reporting only the last would describe a failure.

**This range measures instability, not an effect.** Every row carries p between 0.39 and 0.76, so no row is distinguishable from zero and the spread between them is not an estimate that asset-set choice is "worth" 0.166 Sharpe. What the table shows is that the point estimate is unstable under defensible specification choices by an amount larger than the estimate itself — a statement about the resolution of the method, not about a causal magnitude. Treating the spread as an effect size would be the same inferential move this paper elsewhere argues against.

**We are precise about what drives this.** The 2001 specification differs in two ways at once: a later start date *and* a fourth asset (gold, whose usable history begins in 2000 and therefore forces the short window). Comparing like-for-like three-asset samples, the pure start-date effect is small — −0.033 at 1970 versus −0.050 at 1996, a range of 0.017. The large swing is therefore attributable principally to the **asset universe**, with the sample window entangled as a consequence of that choice.

This is arguably a stronger illustration than a pure start-date effect would be. Including gold in a regime study is not a questionable decision — gold is the canonical inflation hedge and its *exclusion* is what requires justification. An entirely reasonable specification choice, made for substantive reasons, moves the estimate by more than every look-ahead channel in Section 5 combined.

### 4.3 The momentum benchmark

We included a price-only momentum signal to test whether macroeconomic data adds anything beyond what prices already reflect. On raw-return ratios this benchmark ranked highest, and an earlier draft interpreted that as evidence that markets price regimes before the data confirms them.

On excess returns that interpretation does not survive. Momentum returns a Sharpe of 0.679 against the benchmark's 0.682 — a difference of −0.003 with p = 0.999, about as close to exactly nothing as this data can produce. On the longer three-asset sample beginning in 1970 (667 months) it ranks slightly above 60/40, by +0.102, again insignificantly. That figure is distinct from the −0.102 of ablation cell A in Section 5; the numerical coincidence is unfortunate and we flag it rather than disguise it.

The defensible statement is narrower than we first made: a signal using no macroeconomic data performs indistinguishably from both the benchmark and the macro models. That is consistent with markets pre-empting macro data, but equally consistent with none of these signals containing information at a detectable magnitude. We report the ranking and decline the mechanism.

### 4.4 Recession classification and conditional correlations

Regime-conditional return statistics use Newey–West (1987) standard errors, since sampling twelve-month forward returns monthly induces eleven months of overlap and inflates naive t-statistics by roughly √12. Against NBER dates the business-cycle model achieves recall of 0.906 but precision of only 0.326, producing 159 false-positive months across 33 episodes, of which 27 lasted three months or longer. The real-time Sahm rule — a single line of arithmetic — attains a higher F1 (0.525 vs 0.480).

Equity–bond correlation across regimes ranges from −0.04 in recovery to 0.17 in slowdown, against an unconditional 0.08. The sign does change — recovery is the one regime in which the two assets are negatively correlated — but the spread is 0.21, which is narrow relative to the sampling error on a correlation estimated from 131 months. Since regime-conditional allocation is substantially motivated by correlation instability, a shift of this size weakens the mechanism in this sample.

---

## 5. Experiment 1 — Look-ahead ablation

We re-ran the pipeline with three biases reintroduced in a 2×2×2 factorial design:

- **A** — publication lag ignored (series aligned to reference-period end)
- **B** — smoothed decoding over the entire series
- **C** — single full-sample parameter estimation

All cells are evaluated on an identical 536-month sample with identical assets, weights, costs and trade lag. (The ablation sample is one month longer than the 535 months of Section 4.1 because the primary table includes the momentum benchmark, which requires a twelve-month formation window and therefore begins one month later. All comparisons are internally aligned; the two tables are never pooled.) Sample alignment is essential: full-sample fitting emits a signal from the first observation while expanding-window cells begin only after burn-in, a gap spanning the entire Volcker disinflation.

**Fixed weights:**

| Cell | Biases | Sharpe | Δ vs honest | 95% CI | p | Holm | BH |
|---|---|---|---|---|---|---|---|
| none (honest) | 0 | 0.655 | — | — | — | — | — |
| A | 1 | 0.553 | −0.102 | [−0.222, −0.014] | 0.013 | 0.179 | 0.179 |
| B | 1 | 0.685 | +0.030 | [+0.001, +0.069] | 0.036 | 0.468 | 0.252 |
| C | 1 | 0.604 | −0.051 | [−0.150, +0.050] | 0.313 | 1.000 | 0.877 |
| AB | 2 | 0.614 | −0.041 | [−0.100, +0.019] | 0.171 | 1.000 | 0.694 |
| AC | 2 | 0.615 | −0.040 | [−0.137, +0.058] | 0.419 | 1.000 | 0.934 |
| BC | 2 | 0.640 | −0.015 | [−0.117, +0.087] | 0.799 | 1.000 | 0.934 |
| **ABC** | 3 | **0.654** | **−0.001** | [−0.096, +0.098] | 0.998 | 1.000 | 0.998 |

**The fully naive implementation is indistinguishable from the honest one** (Δ = −0.001).

### 5.1 Multiple testing

Across both weighting schemes we conduct **fourteen** comparisons. Two return p < 0.05 uncorrected (cells A and B, at p = 0.013 and 0.036). Under the global null one would expect 0.7 such results, so two is unremarkable.

Applying Holm–Bonferroni, **no cell survives**; the smallest adjusted p-value is 0.179. Under Benjamini–Hochberg control of the false discovery rate, again **none survives**.

We state this explicitly rather than leaving it to the reader, and observe that it would have been straightforward to report cells A and B as significant findings by omitting the correction — in a paper citing Harvey, Liu and Zhu approvingly. The nominal significance of those two cells is what fourteen tests produce under the null.

### 5.2 Direction

Ignoring publication lag *degraded* performance (−0.102), plausibly because fresher data produces a noisier, more whipsaw-prone signal while regimes are persistent enough that arriving two weeks earlier confers little advantage. We do not press this interpretation, since the effect does not survive correction.

---

## 6. Experiment 2 — The free-parameter hypothesis

We hypothesised that look-ahead bias is inert under fixed weights because improved classification cannot express itself through a coarse pre-committed mapping, but would become material once an optimiser could exploit leaked information. We repeated the ablation with per-regime weights fitted in-sample by long-only maximum-Sharpe optimisation.

| Weighting | Honest | Fully naive (ABC) | Δ |
|---|---|---|---|
| Fixed | 0.655 | 0.654 | −0.001 |
| Optimised in-sample | 0.688 | 0.676 | −0.012 |

**The hypothesis is not supported.** No cell in the optimised design reaches even nominal significance (smallest p = 0.198).

The likely explanation is that in every cell the optimiser already observes full-sample *returns*; leaking regime *identity* adds little beyond this, and three regimes mapped onto three assets offers limited scope for exploitation. We report the negative result rather than discarding it, and note that a richer asset universe might yield a different answer.

In-sample optimisation itself raised Sharpe from 0.655 to 0.688 — larger than any look-ahead channel, though also not significant.

---

## 7. Statistical power

The preceding nulls admit two interpretations: the effects are genuinely small, or the study cannot see them. We show it is the latter, and derive the result in closed form so readers can apply it to their own parameters.

### 7.1 Closed form

Let two strategies have per-period Sharpe ratios *S*₁ and *S*₂, correlation ρ, and *T* observations. Jobson and Korkie (1981), with the correction of Memmel (2003), give the asymptotic variance of the difference:

> **Var(Ŝ₁ − Ŝ₂) = (1/T) · [ 2 − 2ρ + ½(S₁² + S₂² − 2·S₁·S₂·ρ²) ]**

For a two-sided test at level α and power 1 − β, the sample size required to detect a true per-period difference δ is:

> **T = [ 2 − 2ρ + ½(S₁² + S₂² − 2·S₁·S₂·ρ²) ] · [ (z₁₋α/₂ + z₁₋β) / δ ]²**

**Unit convention.** Three of our own errors have been a scale factor applied in the wrong power, so we state the convention once and enforce it in code:

1. All algebra is in **per-period variance** units.
2. Annualisation happens **once**, at the boundary: *S*ₐₙₙ = *S*·√12, δₐₙₙ = δ·√12. Comparing a per-period variance to an annualised standard error understates the latter by 3.46.
3. Inflation factors are supplied as ratios of **standard errors** (the observable quantity) and **squared** wherever a variance is required.
4. Required sample size scales with **variance**, and therefore carries the square. Applying a standard-error ratio linearly understates the requirement by the ratio itself.

Points 2 and 4 are each a correction to an earlier draft of this paper.

Two features drive everything that follows. Required sample scales with **1/δ²**, so halving the effect quadruples the data needed. And the leading term is **2(1 − ρ)**: high correlation between strategy and benchmark sharply reduces the variance of the difference. The difference between ρ = 0.85 and ρ = 0.95 is a factor of three in required sample. **ρ must therefore be reported, not assumed** — a recommendation this paper arrived at by violating it, as Section 1.1 records.

### 7.2 Pinned reconciliation

Comparing standard errors computed under different parameterisations is precisely the error that produced our earlier √12 mistake, so we state the protocol explicitly: every quantity below is evaluated at the **same** *T* = 535, the same benchmark Sharpe *S*₂ = 0.682, and each comparison's **own measured ρ**.

| Comparison | ρ | *S*₁ | Bootstrap SE | Memmel SE | Ratio |
|---|---|---|---|---|---|
| HMM (walk-forward) | 0.952 | 0.655 | 0.0630 | 0.0470 | 1.34 |
| Rule: business cycle | 0.889 | 0.652 | 0.0842 | 0.0720 | 1.17 |
| Equal weight | 0.866 | 0.529 | 0.0982 | 0.0787 | 1.25 |
| Rule: growth × inflation | 0.804 | 0.546 | 0.1276 | 0.0953 | 1.34 |
| Momentum (no macro) | 0.758 | 0.679 | 0.1222 | 0.1059 | 1.15 |

**Median ratio 1.248** (range 1.154–1.339), quoted as 1.25 in prose. Note carefully what this is: a ratio of **standard errors**, not of variances. The corresponding variance inflation is 1.248² = **1.558**, and it is the variance figure that enters any sample-size calculation.

It reflects non-normality **and serial dependence jointly**. The realised series carries excess kurtosis of +1.59 and skew of −0.28, which is the phenomenon Lo (2002) describes; but the stationary bootstrap with a twelve-month mean block also preserves dependence, which the i.i.d. closed form ignores. Attributing the whole ratio to fat tails would overclaim, and would sit awkwardly beside the limitation we record in Section 9.

The factor is also **heterogeneous**: 1.15 to 1.34 across comparisons, with no evident relation to ρ, which becomes a 1.32–1.80 range once squared. A single-factor table conceals a 36% spread in required sample. Section 7.3 uses the median and should be read as indicative rather than exact.

**Monte Carlo is not an independent third method.** Simulating Gaussian returns at each comparison's own ρ reproduces the closed-form standard error to within 0.2–1.0%, as it must: both describe the same i.i.d. normal model. The simulation validates the algebra; it does not corroborate the empirical result. An earlier draft described "three independent methods agreeing" when in fact two were the same method and the third differed from them by the tail inflation above, evaluated at a mismatched ρ.

The honest structure is therefore two routes, not three: an **analytic** result (closed form, confirmed by simulation) and an **empirical** one (stationary bootstrap), with the latter consistently 15–34% larger.

### 7.3 Years required

At *S*₂ = 0.682, 80% power, α = 0.05, applying the measured standard-error ratio as a **variance** inflation — that is, squared. The ratio is the median of the per-comparison column in Section 7.2, whose unrounded value is 1.24808; we quote it as 1.25 in prose but the table is computed from the unrounded figure, and `tests/test_units_and_tables.py` reads it from the generated data rather than from either printed form:

| ρ | Δ = 0.05 | **Δ = 0.10** | Δ = 0.20 | Δ = 0.30 | Δ = 0.50 |
|---|---|---|---|---|---|
| 0.76 | 2434 | **610** | 154 | 69 | 25 |
| 0.80 | 2030 | **509** | 128 | 58 | 21 |
| 0.85 | 1524 | **382** | 96 | 43 | 16 |
| 0.89 | 1119 | **281** | 71 | 32 | 12 |
| 0.92 | 814 | **204** | 52 | 23 | 9 |
| 0.95 | 509 | **128** | 33 | 15 | 6 |

The ρ range shown is not hypothetical: it spans the five comparisons actually made in Section 4.1. Detecting Δ = 0.10 requires between one and six centuries depending on which pair is compared — and the most favourable case, our own HMM against 60/40 at ρ = 0.952, still requires 128 years against the 44.6 available.

As an independent check on the table's scale: applying each comparison's **own** bootstrap standard error and solving directly for the sample at which the effect becomes detectable gives 139 years for the HMM pair, against the 128 the table returns at ρ = 0.95. The two routes agree to within the heterogeneity noted in Section 7.2.

### 7.4 What this study could detect

| Comparison | ρ | MDE at 80% | MDE at 50% | Observed effect | Power |
|---|---|---|---|---|---|
| HMM (walk-forward) | 0.952 | 0.177 | 0.123 | 0.027 | 7% |
| Rule: business cycle | 0.889 | 0.236 | 0.165 | 0.031 | 7% |
| Equal weight | 0.866 | 0.275 | 0.192 | 0.153 | 34% |
| Rule: growth × inflation | 0.804 | 0.357 | 0.250 | 0.136 | 19% |
| Momentum (no macro) | 0.758 | 0.342 | 0.239 | 0.003 | 5% |

Minimum detectable effects range from **0.18 to 0.36 Sharpe**; power against the effects actually observed ranges from **5% to 34%**.

The momentum row requires a caveat. Power of 5% against an observed effect of 0.003 is not power in any useful sense — it is the size of the test, since power against a null effect equals α by construction. It should be read as "this comparison observed nothing," not as "this comparison had low power to detect something." The informative rows are those with substantial observed effects and still-inadequate power: equal weight (0.153 observed, 34% power) and the growth × inflation rule (0.136 observed, 19% power).

We flag a distinction an earlier draft got wrong. The half-width of a 95% confidence interval is the effect that would be *just* significant, corresponding to **50%** power — an even chance of detection. The conventional 80% threshold requires (z₀.₉₇₅ + z₀.₈₀) = 2.80 standard errors, not 1.96. Reporting a CI half-width as "the minimum detectable effect" understates the requirement by a factor of 1.43, and both columns are given above so the distinction is visible.

## 8. Implications

### 8.1 A bounded claim

We are not arguing that backtests are uninformative in general. The evidence supports a specific and bounded claim:

- **Δ ≥ 0.3** — requires 15–69 years depending on ρ. This band is **not** uniformly testable: at ρ = 0.95 it needs 15 years, but at ρ = 0.76 it needs 69, which is outside a working lifetime and longer than most available histories. Whether a 0.3 Sharpe claim is testable at all depends on a parameter that is almost never reported — which sharpens rather than softens the recommendation in Section 8.4.
- **Δ ≈ 0.2** — requires 33–154 years. Marginal at the top of the ρ range and out of reach below it.
- **Δ ≈ 0.1 or below** — requires 128–610 years. Not estimable from any sample that exists, at any ρ we observed.

The difficulty is that the third band is where tactical allocation overlays are evaluated, funded and marketed. A manager claiming to add 0.1 Sharpe is claiming something that cannot be verified or refuted by backtest.

### 8.2 The false-positive consequence

When power is low, the ratio of true to false positives among *significant* results deteriorates, because a low-powered study rarely detects real effects while retaining its full 5% false-positive rate. Add publication filtering and the surviving literature in the Δ ≈ 0.1 band is dominated by false positives.

This requires no misconduct, no data mining, and no methodological error. It follows from sample sizes and effect sizes alone.

### 8.3 For methodological priorities

Considerable effort in this literature — including much of our own — targets look-ahead bias. Our ablation found the joint effect of the three principal channels **not distinguishable from zero** (point estimate −0.001, 95% CI [−0.096, +0.098], p = 0.998, nothing surviving correction). We quote the interval rather than the point estimate deliberately: −0.001 lands near zero partly by chance, given that cell A alone is −0.102, and presenting it as a magnitude would overstate what a single point estimate at this power can establish.

We are careful about what this licenses. It does **not** show look-ahead bias is unimportant; our power against effects of that size was 5–7% for the relevant comparisons. It shows the effort purchased no measurable change in conclusions here, and that a study confident it has eliminated look-ahead should not therefore be confident its results are informative.

The larger threats operate through **selection**. Our own specification-sensitivity result — a 0.166 swing that reverses sign, arising from the entirely defensible decision to include gold — spans a wider range than any bias we deliberately introduced, though as Section 4.2 notes that is a statement about the instability of the estimate rather than a measured effect.

### 8.4 Recommendations

**Report power alongside significance.** State the effect size the study could have detected, per comparison, in the form of Section 7.4. A null result at the power levels reported there — 5% to 34% — is nearly uninformative, and presenting one as evidence of absence is an error.

**Report ρ.** The required sample depends on it strongly and it is rarely reported. A comparison at ρ = 0.95 needs roughly a third the data of one at ρ = 0.85.

**Pre-commit the specification.** Asset universe and sample window should be fixed in advance and their sensitivity reported, since either can exceed the effect under study.

**Prefer hypotheses with larger expected effects.** Research should concentrate where effects are large enough to be seen. Strategies with an expected edge of 0.1 Sharpe may well be real and worth deploying, but they cannot be *validated* by backtest and should be justified on other grounds.

---

## 9. Limitations

**Our null results are underpowered, including the ones that favour our argument.** We cannot conclude that regime models fail to add value, nor that look-ahead bias is immaterial. Interpreting our nulls as findings would repeat the error the paper describes.

**Revisions are not addressed.** Switch A captures publication timing only. Since revisions are largest at turning points, our look-ahead estimates are lower bounds.

**Emission misspecification is unresolved.** The monotonic BIC decline (Section 3.5) suggests the Gaussian emission is inadequate. A *t*-distributed or mixture emission is the natural next specification and was not attempted.

**Regime weights are hand-specified in the primary analysis**, so the null is for these weights.

**Proxy series are approximations.** Bond returns are duration-reconstructed, ignoring convexity and roll-down. The commodity proxy is a price index, not an investable return. Gold lacks 1970s coverage and is excluded from the primary analysis, removing the asset most associated with the stagflation regime — while Section 4.2 shows that including it is what moves the estimate most.

**One asset universe, one frequency, one economy.** Three to five assets, monthly, United States.

**The power analysis assumes a stable true effect.** If the genuine advantage of regime allocation is concentrated in rare episodes rather than distributed evenly, our simulation understates detectability for that pattern — though it correspondingly raises the sample required for a stable estimate.

**The closed form assumes i.i.d. returns.** We correct empirically, using a standard-error ratio of 1.248 (variance 1.558) measured at pinned parameters in Section 7.2, but do not model the dependence structure analytically. That ratio conflates non-normality with serial dependence and is heterogeneous across comparisons (1.15–1.34), so Section 7.3 is indicative rather than exact. Ledoit and Wolf (2008) provide the more rigorous route.

**The power argument is not novel in kind.** Harvey and Liu (2015), Harvey, Liu and Zhu (2016), and Bailey and López de Prado (2014) advance closely related arguments. Our contribution is the explicit sample-length formula, its validation, its application to a complete reproducible pipeline, and its reflexive application to our own results.

---

## 10. Conclusion

We built a macroeconomic regime model to a standard exceeding normal practice: point-in-time data, forward-filtered inference, walk-forward re-estimation, adversarial tests proving the absence of look-ahead, and stability-based model selection. We then asked whether that care mattered.

It did not measurably matter, and neither did the strategy. The fully naive implementation performs indistinguishably from the rigorous one, and both perform indistinguishably from a rebalanced 60/40.

The explanation is not that regime models work, nor that they fail, but that a backtest of realistic length cannot resolve differences of the size at issue. Detecting a 0.1 Sharpe improvement requires between one and six centuries of monthly data, depending on a correlation parameter the literature rarely reports. Our study, at 44.6 years and by construction a careful example of its kind, had between a 5% and a 34% chance of detecting the effects it actually observed.

This is a bounded claim, and we prefer it bounded. Large effects remain testable, though not uniformly: a strategy promising 0.5 Sharpe over a benchmark can be evaluated in six years at ρ = 0.95 but needs twenty-five at ρ = 0.76. But a field that debates 0.1 Sharpe differences using 25-year samples is not doing weak empirical work so much as work whose central quantity is not estimable from the available evidence. The appropriate response is not better hygiene — we applied the best hygiene we know and it changed nothing — but a change in which questions are asked, and considerably more humility about which have been answered.

Including ours.

---

## Disclosure of methods and tooling

This paper and its accompanying pipeline were developed with the assistance of an
AI coding assistant (Anthropic Claude), used for implementation, drafting, and
iterative review. The author directed the research questions, adjudicated the
methodological choices, and is responsible for all claims.

The disclosure is made for two reasons beyond convention. First, several of the
substantive turns in this project - the discovery that the conventional
credit-spread series is no longer served in usable form, the finding that
information criteria do not identify a state count here, and each of the three
unit errors documented in Section 1.1 - emerged from that iterative process, and
the paper's argument about verification is not separable from how it was written.

Second, and more importantly, the argument of this paper is that trust in a
quantitative result should rest on mechanical reproducibility rather than on
confidence in the care of whoever produced it. That standard does not become less
applicable when the producer is partly a machine; if anything it becomes more so.
Every figure reported here is regenerated from the committed data by the commands
in Appendix A, and `tests/test_units_and_tables.py` reconstructs each published
table cell from its primitives and fails if the manuscript and the code disagree.
Readers are invited to verify the numbers rather than to take either author's word
for them.

---

## References

Ang, A., & Bekaert, G. (2002). International asset allocation with regime shifts. *Review of Financial Studies*, 15(4), 1137–1187.

Bailey, D. H., & López de Prado, M. (2014). The deflated Sharpe ratio: Correcting for selection bias, backtest overfitting, and non-normality. *Journal of Portfolio Management*, 40(5), 94–107.

Hamilton, J. D. (1989). A new approach to the economic analysis of nonstationary time series and the business cycle. *Econometrica*, 57(2), 357–384.

Harvey, C. R., & Liu, Y. (2015). Backtesting. *Journal of Portfolio Management*, 42(1), 13–28.

Harvey, C. R., Liu, Y., & Zhu, H. (2016). …and the cross-section of expected returns. *Review of Financial Studies*, 29(1), 5–68.

Ioannidis, J. P. A. (2005). Why most published research findings are false. *PLoS Medicine*, 2(8), e124.

Jobson, J. D., & Korkie, B. M. (1981). Performance hypothesis testing with the Sharpe and Treynor measures. *Journal of Finance*, 36(4), 889–908.

Kritzman, M., Page, S., & Turkington, D. (2012). Regime shifts: Implications for dynamic strategies. *Financial Analysts Journal*, 68(3), 22–39.

Ledoit, O., & Wolf, M. (2008). Robust performance hypothesis testing with the Sharpe ratio. *Journal of Empirical Finance*, 15(5), 850–859.

Lo, A. W. (2002). The statistics of Sharpe ratios. *Financial Analysts Journal*, 58(4), 36–52.

Memmel, C. (2003). Performance hypothesis testing with the Sharpe ratio. *Finance Letters*, 1(1), 21–23.

Newey, W. K., & West, K. D. (1987). A simple, positive semi-definite, heteroskedasticity and autocorrelation consistent covariance matrix. *Econometrica*, 55(3), 703–708.

Politis, D. N., & Romano, J. P. (1994). The stationary bootstrap. *Journal of the American Statistical Association*, 89(428), 1303–1313.

Welch, I., & Goyal, A. (2008). A comprehensive look at the empirical performance of equity premium prediction. *Review of Financial Studies*, 21(4), 1455–1508.

---

## Appendix A — Reproducibility

All results are generated from a committed data cache and require no API credentials:

```bash
python -m macroregime.run                # pipeline and primary results
python -m macroregime.evaluation.power   # Section 7
python -m macroregime.viz.report         # static HTML report
pytest                                   # 49 tests: look-ahead proofs, unit
                                         # conventions, and published-table regression
```

Environment: Python 3.12, `hmmlearn` 0.3.3, `statsmodels` 0.15.0, `numpy` 2.5.2, `pandas` 3.0.5. Seeds are fixed throughout; HMM estimation uses 20 initialisations selected by log-likelihood.

### A.1 Look-ahead test suite

Absence of look-ahead is proven rather than asserted. The central test is adversarial: every observation after a cutoff is replaced with implausible values, the feature matrix is rebuilt, and all earlier rows are required to be bit-identical. A complementary inverse test removes the trade lag and requires performance to *improve*, confirming the lag is binding rather than decorative.

### A.2 Unit and table-regression tests

Every error found in review was a scale factor applied in the wrong power, and every one was catchable by recomputing a published number from its own primitives. `tests/test_units_and_tables.py` does this mechanically: it asserts that required sample size scales with the *square* of a standard-error ratio and the inverse square of the effect, that annualisation is applied exactly once, that power against a null effect equals α — and it reconstructs all thirty cells of the Section 7.3 table directly from the closed form, failing if the paper and the code ever diverge again.

### A.3 Generated artefacts

| File | Contents |
|---|---|
| `summary_long.csv` | Primary backtest; Sharpe and `return_vol_ratio` reported separately |
| `significance_long.csv` | Bootstrap tests including ρ for each comparison |
| `ablation_bootstrap.csv` | Factorial cells with CIs, Holm and BH adjusted p-values |
| `ablation_weight_modes.csv` | Fixed vs optimised weighting |
| `power_reconciliation.csv` | Section 7.2 pinned per-comparison standard errors |
| `power_years_by_rho.csv` | Section 7.3 sample-size requirements by ρ |
| `power_gaussian.csv`, `power_empirical.csv` | Simulation-based power |
| `vs_sahm.csv`, `nber_lead_lag.csv` | Recession-classification validation |
| `staleness.csv` | Per-series publication lag at decision points |

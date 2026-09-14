# Underpowered by Construction: Sharpe Differences of 0.1 Are Not Estimable from Realistic Samples

**Evidence from a fully real-time macroeconomic regime-switching pipeline**

**Ethan Gao**
*Independent researcher*

---

*Working paper. September 2026.*

Code, data and the test suite that reproduces every figure in this paper:
`https://github.com/ohNwghtEG/macroeconomic-regime-detection-dashboard`
Archived release (v1.0-preprint): `https://doi.org/10.5281/zenodo.22737848`

Code is released under the MIT License; this manuscript under CC BY 4.0.

---

## Abstract

I build a macroeconomic regime-detection and asset-allocation pipeline using point-in-time data that respects each series' publication lag, forward-filtered hidden Markov state probabilities, and expanding-window parameter re-estimation. I then use it to ask whether a backtest of realistic length is able to tell if such a strategy works at all.

For effect sizes in the range the field actually argues about, it cannot. Over 535 months (44.6 years), no regime-based allocation differs significantly from a rebalanced 60/40 benchmark. The walk-forward hidden Markov model returns an excess-return Sharpe of 0.655 against the benchmark's 0.682 (Δ = −0.027, p = 0.64). I also ran a 2×2×2 factorial ablation that deliberately reintroduces three forms of look-ahead bias, and none of the fourteen resulting tests survives multiple-testing correction. The fully naive implementation scores 0.654 against the honest 0.655. Repeating the ablation with in-sample-optimised regime weights does not change this.

The explanation is statistical power, which can be computed in closed form. Using the Jobson–Korkie statistic with the Memmel correction, the sample length needed to detect a true Sharpe difference Δ at 80% power depends on Δ, the benchmark Sharpe, and the correlation ρ between the two strategies. The correlation matters most, and it is almost never reported. Across the five comparisons in my study ρ ranges from 0.76 to 0.95, and the sample required to detect Δ = 0.10 falls correspondingly from **610 years to 128 years**. Even the most favourable case needs nearly three times the history that exists.

Two pieces of this work are meant for reuse. The first is a per-comparison detectability table (Section 7.4). In this study the minimum detectable effect at 80% power ranges from **0.18 to 0.36 Sharpe** across five comparisons, while the observed effects range from 0.003 to 0.153, giving achieved power of **5% to 34%**. I suggest that studies routinely report the minimum detectable effect next to the observed effect, since a null result is hard to interpret without it. The second is a test suite (Appendix A.2) that rebuilds every published figure from its primitives and fails if the manuscript and the code disagree.

The implication has limits. Sharpe differences of 0.3 and above can be detected within a career only at high ρ (15 years at ρ = 0.95, 69 at ρ = 0.76). Differences of 0.1 to 0.2, where tactical allocation overlays are usually evaluated and sold, need anywhere from three decades to six centuries, and below ρ = 0.9 they are effectively out of reach. If studies in that range are filtered for significance before publication, most of what gets published will be false positives. The same reasoning applies to my own null results, so I do not read them as evidence that regime models fail.

**Keywords:** regime switching, hidden Markov models, backtesting, statistical power, Sharpe ratio inference, look-ahead bias, tactical asset allocation

---

## 1. Introduction

Macroeconomic regime models are intellectually appealing and widely used by institutions, yet the published evidence on whether they work is surprisingly hard to evaluate.

The premise is sound. Portfolio construction rests on assumptions about expected returns, volatilities and correlations that are demonstrably conditional on the macroeconomic environment rather than fixed. Equity and bond returns were negatively correlated for roughly two decades and then were not, most visibly in 2022. If the joint distribution of asset returns depends on an underlying macroeconomic state, then identifying that state and conditioning allocation upon it is the natural response. Bridgewater's All Weather framework is built explicitly on a growth-versus-inflation state space, and multi-asset teams at large institutions run structurally similar overlays.

Methodological criticism of this literature has concentrated on look-ahead bias. The critique is well founded. Economic data is revised, released with a lag, and served by databases in its latest revised form; standard hidden-Markov implementations decode states using the entire sample; and models are routinely estimated once on all available history.

This paper tests that critique directly. I build the pipeline correctly and then, holding everything else fixed, switch each bias back on to measure what it is worth. The measured effects are small and do not survive multiple-testing correction. The more important finding is the reason for that: at these effect sizes, the experiment could never have resolved the question.

The paper provides a fully specified, reproducible real-time regime pipeline whose test suite checks for look-ahead directly, and a factorial ablation of three look-ahead channels with bootstrap confidence intervals and multiple-testing correction, in which no channel survives. Its main contribution, though, is to show how the sample length needed to detect a given Sharpe difference depends on the correlation between the strategies being compared. The expression itself is standard, following from Jobson and Korkie (1981) and Memmel (2003), and closely related minimum-sample results appear in Opdyke (2007) and Bailey and López de Prado (2012). What has been missing is its application across the correlations that real comparisons produce, together with a per-comparison statement of what a given study could have detected. Applied to my own results, this turns a case-specific null into a general statement about what a backtest can establish.

The central result can be stated now. For two strategies with per-period Sharpe ratios *S*₁, *S*₂ and correlation ρ, the sample size required to detect a true per-period difference δ at level α and power 1 − β is

> **T = [ 2 − 2ρ + ½(S₁² + S₂² − 2·S₁·S₂·ρ²) ] · [ (z₁₋α/₂ + z₁₋β) / δ ]²**

Required sample scales with 1/δ², and because of the leading term 2(1 − ρ), the answer depends heavily on a parameter the literature rarely reports. At the benchmark Sharpe of 0.68 and 80% power, detecting Δ = 0.10 requires 128 years at ρ = 0.95 and 610 years at ρ = 0.76. Sections 3 to 6 present the pipeline and ablation that led to this result; readers interested only in the methodology can skip to Section 7.

This result also limits what the pipeline and the ablation can show. A study with 5–34% power has not established that look-ahead bias is unimportant.

---

## 2. Related literature

The finding that predictive relationships in finance degrade or vanish out of sample is long established. Welch and Goyal (2008) evaluated a comprehensive set of equity-premium predictors and found most failed to outperform a historical mean out of sample despite strong in-sample performance. My headline null is a replication within that tradition and makes no new empirical claim.

Regime-switching methodology descends from Hamilton (1989). Applications to allocation include Ang and Bekaert (2002) on regime-dependent international correlations and Kritzman, Page and Turkington (2012), who reported that regime-aware dynamic strategies improved risk-adjusted performance. The latter is exactly the kind of finding I argue cannot be resolved at conventional sample lengths, in either direction.

On inference, Jobson and Korkie (1981) derived the asymptotic distribution of the difference between two Sharpe ratios, and Memmel (2003) corrected an error in their variance expression. That corrected expression is the analytic basis of Section 7. Lo (2002) showed that Sharpe ratio standard errors are substantially larger than commonly assumed under non-normality, and Ledoit and Wolf (2008) developed robust tests under autocorrelation and fat tails. Opdyke (2007) extended the test to returns that are serially correlated and non-normal, and gave the minimum number of observations needed to show that one Sharpe ratio exceeds another. Bailey and López de Prado (2012) introduced the minimum track record length, the history required before a single Sharpe ratio can be declared above a chosen threshold. My bootstrap follows Politis and Romano (1994).

The multiple-testing problem in backtesting has been examined by Harvey, Liu and Zhu (2016) and Harvey and Liu (2015); Bailey and López de Prado (2014) formalised it as the Deflated Sharpe Ratio.

This paper sits alongside that work. The multiple-testing literature asks *"given that many strategies were tried, how should one discount the winner?"* I ask an earlier question: *"given the sample available, could a single comparison have been informative at all?"* For Δ ≥ 0.3 the answer is yes, and multiple-testing correction is the binding constraint. For Δ ≈ 0.1 it is no, and corrections end up being applied to comparisons that were uninformative to begin with.

The structural argument comes from Ioannidis (2005): when power is low and results are filtered for significance, the share of published findings that are false goes up. I argue that backtesting at conventional sample lengths falls into that situation for a specific, identifiable range of effect sizes.

---

## 3. Pipeline construction

The point of building carefully here is to rule out alternative explanations for a null result. Where a choice had to be made, I took the conservative option.

### 3.1 Data

Twenty-five FRED series span growth, inflation, financial conditions and monetary policy. Two data decisions departed from the specification I started with and are worth recording.

The conventional credit-spread series can no longer be used. FRED now serves the ICE BofA US High Yield Option-Adjusted Spread (`BAMLH0A0HYM2`) only for a rolling three-year window; as of 5 September 2026 its metadata reports an observation start of 2023-09-05, reflecting a licensing restriction. I substitute the Chicago Fed National Financial Conditions Index (weekly, from 1971) and the Moody's Baa–Aaa default spread (monthly, from 1919). ISM PMI series were similarly withdrawn in 2016, and I use the Chicago Fed National Activity Index in their place. Anyone replicating an older study should check that its series are still available.

ETF histories are also too short. SPY begins in 1993, TLT in 2002 and GLD in 2004, which together cover only two recessions. I splice long-history total-return proxies behind each ETF: the Ken French market factor for equities (from 1926), a duration-based reconstruction from ten-year yields for bonds, and PPI All Commodities for commodities (from 1913). Gold data only goes back to 2000, since no free bullion series with 1970s coverage survives, so gold is excluded from the primary analysis.

### 3.2 Point-in-time alignment

Each series carries its actual publication lag, and the feature vector for month-end *T* contains only observations released on or before *T*:

```
release_date = period_end + publication_lag_days
```

Two details are easy to get wrong, and each is worth roughly a month of look-ahead. FRED stamps monthly series with the *first* day of the reference month, so March CPI appears at `2008-03-01` even though it is published around 16 April. Derived series also inherit the lag of their *slowest* input.

This handles publication timing only. Accounting for revisions would require ALFRED vintages, which I did not use, so my look-ahead estimates are lower bounds.

### 3.3 Features

Series are transformed to approximately stationary form and standardised with **expanding-window** z-scores, whose mean and standard deviation at each date use only prior history. A full-sample z-score would leak the eventual distribution into every historical observation.

### 3.4 Regime models

I estimate two rule-based models and keep them separate, since the literature often treats them as one. The **growth × inflation quadrant** model describes the direction of macroeconomic surprises, while the **business-cycle phase** model describes a sequence of phases. Both use a ±0.25σ deadband and two-month confirmation.

The third model is a Gaussian hidden Markov model, built with the following requirements:

1. **Filtered, not smoothed, probabilities.** Standard `predict()` and `predict_proba()` interfaces condition on the entire sample. I implement the forward recursion directly in log space and validate it three ways: against the identity that filtered and smoothed probabilities must coincide at the terminal observation (agreement to 2.6 × 10⁻¹⁴), against an independent linear-space implementation (4.4 × 10⁻¹⁶), and by confirming that they differ elsewhere. They disagree on 5.4% of months.
2. **Expanding-window re-estimation**, with 45 refits and a live signal from December 1981.
3. **Deterministic relabelling** by mean growth loading after every fit, since hidden states are otherwise unidentified across refits.

### 3.5 Selecting the number of states

I had planned on four states, but the data does not support that choice:

- **BIC declines monotonically** from 2811 at K = 2 to 1804 at K = 8, never turning.
- **Held-out likelihood is non-monotonic**, dominated by whether a given K accommodates COVID.
- **Seed-to-seed stability collapses**: adjusted Rand index falls from 0.78 at K = 3 to ≈0.5 at K = 4–6.
- **Episode adequacy fails**: at K ≥ 4 some states are entered only four times, and the effective sample for conditional estimation is the number of episodes, not months.

I select K = 3 on stability and episode adequacy.

It is tempting to read this as a penalty that is simply too weak. But BIC that keeps falling across an entire grid is a recognised symptom of **emission misspecification**. When a Gaussian emission cannot represent the conditional distribution, extra states get recruited to approximate a non-Gaussian shape, and the likelihood gain keeps outrunning the penalty. Given the fat tails documented in Section 7, this is the likely mechanism. It means information criteria cannot identify the number of states under this emission family, and a *t*-distributed or mixture emission would be the natural next step. I did not pursue that here, and it is listed among the limitations in Section 9.

### 3.6 Backtest protocol

Signals formed at month-end *T* are traded at *T+1*. Costs of 10 bps are charged on turnover. Allocation is probability-weighted rather than switched on the argmax. The benchmark is a **rebalanced** 60/40 carrying the identical cost model. Regime weights are hand-specified, so the primary results contain no fitted allocation parameters.

All performance ratios are Sharpe ratios on excess returns, using the three-month Treasury bill as the risk-free rate.

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

*Note: confidence intervals and p-values come from the same stationary-bootstrap distribution (5000 iterations, 12-month mean block length). The p-value is computed as 2·min(P(Δ* ≤ 0), P(Δ* ≥ 0)) over the bootstrap replicates, so an interval excluding zero and p < 0.05 are equivalent by construction.*

No strategy differs significantly from the benchmark, and every macro model ranks below it. The HMM does have a materially smaller maximum drawdown (−23.3% vs −28.5%), which suggests it mainly works by cutting risk heading into contractions. Its risk-adjusted return is no better, so drawdown is the one measure on which it plausibly justifies its complexity.

### 4.2 Specification sensitivity

This result illustrates the paper's thesis more directly than anything else in the study. All rows use the **business-cycle rule model** against the rebalanced 60/40 benchmark; only the sample start and asset set vary.

| Sample start | Asset set | Years | Business-cycle rule | 60/40 | Δ | p |
|---|---|---|---|---|---|---|
| 2001 | 4-asset (incl. gold) | 24.8 | 0.729 | 0.596 | **+0.133** | 0.393 |
| 1996 | 3-asset | 30.5 | 0.580 | 0.629 | −0.050 | 0.666 |
| 1990 | 3-asset | 36.5 | 0.617 | 0.685 | −0.067 | 0.513 |
| 1981 | 3-asset | 45.5 | 0.613 | 0.638 | −0.025 | 0.757 |
| 1970 | 3-asset | 56.5 | 0.477 | 0.510 | −0.033 | 0.639 |

The extremes differ by 0.166 Sharpe and carry opposite signs. A researcher reporting only the first row would describe a successful strategy, and one reporting only the last would describe a failure.

The spread should be read as instability in the estimate. Every row has p between 0.39 and 0.76, so no row is distinguishable from zero, and the gap between rows cannot be taken to mean that asset-set choice is "worth" 0.166 Sharpe. What the table does show is that defensible specification choices move the point estimate by more than the size of the estimate itself. That says something about the resolution of the method and nothing about a causal magnitude. Treating the spread as an effect size would be the same inferential mistake this paper warns about elsewhere.

It is worth being specific about where the swing comes from. The 2001 specification differs in two ways at once: it has a later start date *and* a fourth asset (gold, whose usable history begins in 2000 and therefore forces the short window). Comparing three-asset samples with each other, the pure start-date effect is small, −0.033 at 1970 versus −0.050 at 1996, a range of 0.017. Most of the swing therefore comes from the **asset universe**, with the shorter window following from that choice.

If anything this makes the illustration stronger. Including gold in a regime study is a perfectly reasonable decision; gold is the standard inflation hedge, and it is leaving it out that needs justifying. A choice made for good substantive reasons moves the estimate by more than all of the look-ahead channels in Section 5 combined.

### 4.3 The momentum benchmark

I included a price-only momentum signal to test whether macroeconomic data adds anything beyond what prices already reflect. If markets price regimes before the data confirms them, a signal built only from prices should beat the macro models by a detectable margin.

Momentum returns a Sharpe of 0.679 against the benchmark's 0.682, a difference of −0.003 with p = 0.999. On the longer three-asset sample beginning in 1970 (667 months) it ranks slightly above 60/40, by +0.102, again insignificantly. That figure happens to have the same magnitude as ablation cell A in Section 5 (−0.102), but the two are unrelated.

What the data supports is a narrow statement. A signal that uses no macroeconomic data performs about the same as both the benchmark and the macro models. That fits the idea that markets move before macro data arrives, but it fits equally well with none of these signals carrying enough information to detect, and this data cannot tell the two apart.

### 4.4 Recession classification and conditional correlations

Regime-conditional return statistics use Newey–West (1987) standard errors, since sampling twelve-month forward returns monthly creates eleven months of overlap and inflates naive t-statistics by roughly √12. Against NBER dates the business-cycle model achieves recall of 0.906 but precision of only 0.326, producing 159 false-positive months across 33 episodes, of which 27 lasted three months or longer. The real-time Sahm rule, which is a single line of arithmetic, attains a higher F1 (0.525 vs 0.480).

Equity–bond correlation across regimes ranges from −0.04 in recovery to 0.17 in slowdown, against an unconditional 0.08. The sign does change, since recovery is the only regime in which the two assets are negatively correlated, but the total spread is 0.21. That is narrow relative to the sampling error of a correlation estimated from 131 months. Much of the case for regime-conditional allocation rests on correlations shifting between regimes, so a shift this small weakens that case in this sample.

---

## 5. Experiment 1: Look-ahead ablation

I re-ran the pipeline with three biases reintroduced in a 2×2×2 factorial design:

- **A**: publication lag ignored (series aligned to reference-period end)
- **B**: smoothed decoding over the entire series
- **C**: single full-sample parameter estimation

All cells are evaluated on an identical 536-month sample with identical assets, weights, costs and trade lag. (The ablation sample is one month longer than the 535 months of Section 4.1 because the primary table includes the momentum benchmark, which needs a twelve-month formation window and so begins one month later. All comparisons are internally aligned, and the two tables are never pooled.) Aligning the samples matters because full-sample fitting produces a signal from the first observation, whereas expanding-window cells only start after burn-in, and that gap covers the whole Volcker disinflation.

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

The fully naive implementation cannot be distinguished from the honest one (Δ = −0.001).

### 5.1 Multiple testing

Across both weighting schemes I conduct **fourteen** comparisons. Two return p < 0.05 uncorrected (cells A and B, at p = 0.013 and 0.036). Under the global null one would expect 0.7 such results, so two is unremarkable.

After Holm–Bonferroni adjustment no cell is significant, and the smallest adjusted p-value is 0.179. Controlling the false discovery rate with Benjamini–Hochberg gives the same outcome.

It would have been easy to report cells A and B as significant findings by leaving out the correction, even in a paper that cites Harvey, Liu and Zhu approvingly. Fourteen tests run under the null produce a couple of nominally significant results fairly often, and that is the most likely explanation for these two.

### 5.2 Direction

Ignoring publication lag *degraded* performance (−0.102). A plausible reason is that fresher data gives a noisier signal that whipsaws more, while regimes last long enough that getting the data two weeks earlier helps little. Since the effect does not survive correction, I would not put much weight on this explanation.

---

## 6. Experiment 2: The free-parameter hypothesis

I hypothesised that look-ahead bias has no effect under fixed weights because better classification cannot express itself through a coarse, pre-committed mapping, but that it would start to matter once an optimiser could exploit the leaked information. To test this I repeated the ablation with per-regime weights fitted in-sample by long-only maximum-Sharpe optimisation.

| Weighting | Honest | Fully naive (ABC) | Δ |
|---|---|---|---|
| Fixed | 0.655 | 0.654 | −0.001 |
| Optimised in-sample | 0.688 | 0.676 | −0.012 |

The results do not support the hypothesis. No cell in the optimised design reaches even nominal significance (smallest p = 0.198).

The most likely reason is that the optimiser already sees full-sample *returns* in every cell, so leaking regime *identity* as well adds little. Three regimes mapped onto three assets also leave limited room to exploit it. A richer asset universe might give a different answer.

In-sample optimisation on its own raised the Sharpe from 0.655 to 0.688. That is more than any look-ahead channel moved it, although it is not significant either.

---

## 7. Statistical power

There are two ways to read the nulls above. Either the effects really are small, or the study is unable to see them. This section shows that the second reading holds, and gives the result in closed form so that readers can apply it to their own parameters.

### 7.1 Closed form

Let two strategies have per-period Sharpe ratios *S*₁ and *S*₂, correlation ρ, and *T* observations. Jobson and Korkie (1981), with the correction of Memmel (2003), give the asymptotic variance of the difference:

> **Var(Ŝ₁ − Ŝ₂) = (1/T) · [ 2 − 2ρ + ½(S₁² + S₂² − 2·S₁·S₂·ρ²) ]**

For a two-sided test at level α and power 1 − β, the sample size required to detect a true per-period difference δ is:

> **T = [ 2 − 2ρ + ½(S₁² + S₂² − 2·S₁·S₂·ρ²) ] · [ (z₁₋α/₂ + z₁₋β) / δ ]²**

A scale factor applied in the wrong power is an easy error to make in this calculation and a hard one to spot on reading, so I state the unit convention once here and enforce it in code:

1. All algebra is in **per-period variance** units.
2. Annualisation happens **once**, at the boundary: *S*ₐₙₙ = *S*·√12, δₐₙₙ = δ·√12. Comparing a per-period variance to an annualised standard error understates the latter by 3.46.
3. Inflation factors are supplied as ratios of **standard errors** (the observable quantity) and **squared** wherever a variance is required.
4. Required sample size scales with **variance**, and therefore carries the square. Applying a standard-error ratio linearly understates the requirement by the ratio itself.

Two features of the formula matter for everything that follows. Required sample scales with **1/δ²**, so halving the effect quadruples the data needed. The leading term is **2(1 − ρ)**, so a high correlation between strategy and benchmark sharply reduces the variance of the difference; moving from ρ = 0.85 to ρ = 0.95 cuts the required sample by a factor of three. For that reason ρ should always be reported.

### 7.2 Pinned reconciliation

Standard errors computed under different parameterisations cannot be meaningfully compared, so every quantity below is evaluated at the **same** *T* = 535, the same benchmark Sharpe *S*₂ = 0.682, and each comparison's **own measured ρ**.

| Comparison | ρ | *S*₁ | Bootstrap SE | Memmel SE | Ratio |
|---|---|---|---|---|---|
| HMM (walk-forward) | 0.952 | 0.655 | 0.0630 | 0.0470 | 1.34 |
| Rule: business cycle | 0.889 | 0.652 | 0.0842 | 0.0720 | 1.17 |
| Equal weight | 0.866 | 0.529 | 0.0982 | 0.0787 | 1.25 |
| Rule: growth × inflation | 0.804 | 0.546 | 0.1276 | 0.0953 | 1.34 |
| Momentum (no macro) | 0.758 | 0.679 | 0.1222 | 0.1059 | 1.15 |

The median ratio is **1.248** (range 1.154–1.339), which I round to 1.25 in the text. This is a ratio of **standard errors**. The corresponding variance inflation is 1.248² = **1.558**, and the variance figure is the one that belongs in any sample-size calculation.

The ratio reflects both non-normality and serial dependence. The realised series has excess kurtosis of +1.59 and skew of −0.28, the pattern Lo (2002) describes, but the stationary bootstrap with a twelve-month mean block also preserves dependence, which the i.i.d. closed form ignores. The ratio should therefore not be attributed to fat tails alone (see also Section 9).

The factor also varies across comparisons, from 1.15 to 1.34 with no obvious relation to ρ; squared, that becomes 1.32 to 1.80. Using a single factor hides a 36% spread in required sample, so the Section 7.3 figures, which use the median, are approximate.

The Monte Carlo simulation does not count as a third, independent method. Simulating Gaussian returns at each comparison's own ρ reproduces the closed-form standard error to within 0.2–1.0%, as it has to, because both describe the same i.i.d. normal model. The simulation checks the algebra and tells us nothing further about the empirical result.

That leaves two routes: an **analytic** one (the closed form, confirmed by simulation) and an **empirical** one (the stationary bootstrap), with the empirical standard errors 15–34% larger in every comparison.

### 7.3 Years required

The table below uses *S*₂ = 0.682, 80% power and α = 0.05, with the measured standard-error ratio applied as a **variance** inflation, meaning it is squared. The ratio is the median of the per-comparison column in Section 7.2, whose unrounded value is 1.24808. The table is computed from that unrounded figure, and `tests/test_units_and_tables.py` reads it from the generated data rather than from either printed version:

| ρ | Δ = 0.05 | **Δ = 0.10** | Δ = 0.20 | Δ = 0.30 | Δ = 0.50 |
|---|---|---|---|---|---|
| 0.76 | 2434 | **610** | 154 | 69 | 25 |
| 0.80 | 2030 | **509** | 128 | 58 | 21 |
| 0.85 | 1524 | **382** | 96 | 43 | 16 |
| 0.89 | 1119 | **281** | 71 | 32 | 12 |
| 0.92 | 814 | **204** | 52 | 23 | 9 |
| 0.95 | 509 | **128** | 33 | 15 | 6 |

The ρ values in the table are the ones actually observed across the five comparisons in Section 4.1. Detecting Δ = 0.10 takes between one and six centuries depending on the pair. Even the most favourable case, my HMM against 60/40 at ρ = 0.952, needs 128 years, and only 44.6 are available.

As a rough check on scale, I also took each comparison's own bootstrap standard error and solved directly for the sample at which the effect becomes detectable. For the HMM pair this gives 139 years, compared with 128 from the table at ρ = 0.95, a gap that is within the variation described in Section 7.2.

### 7.4 What this study could detect

| Comparison | ρ | MDE at 80% | MDE at 50% | Observed effect | Power |
|---|---|---|---|---|---|
| HMM (walk-forward) | 0.952 | 0.177 | 0.123 | 0.027 | 7% |
| Rule: business cycle | 0.889 | 0.236 | 0.165 | 0.031 | 7% |
| Equal weight | 0.866 | 0.275 | 0.192 | 0.153 | 34% |
| Rule: growth × inflation | 0.804 | 0.357 | 0.250 | 0.136 | 19% |
| Momentum (no macro) | 0.758 | 0.342 | 0.239 | 0.003 | 5% |

Minimum detectable effects range from **0.18 to 0.36 Sharpe**, and power against the effects actually observed ranges from **5% to 34%**.

The momentum row needs a caveat. Power of 5% against an observed effect of 0.003 is really just the size of the test, because power against a null effect equals α by construction. The row means this comparison observed nothing; it does not mean the comparison had low power to find something. The informative rows are the ones with sizeable observed effects and still inadequate power: equal weight (0.153 observed, 34% power) and the growth × inflation rule (0.136 observed, 19% power).

Two thresholds are easy to confuse here. The half-width of a 95% confidence interval is the effect that would be *just* significant, which corresponds to **50%** power, an even chance of detection. The conventional 80% threshold requires (z₀.₉₇₅ + z₀.₈₀) = 2.80 standard errors instead of 1.96. Calling a CI half-width "the minimum detectable effect" understates the requirement by a factor of 1.43, which is why the table shows both columns.

## 8. Implications

### 8.1 A bounded claim

None of this means backtests are uninformative in general. What the evidence supports depends on the size of the effect:

- **Δ ≥ 0.3** requires 15–69 years depending on ρ. This band is **not** uniformly testable. At ρ = 0.95 it needs 15 years, but at ρ = 0.76 it needs 69, which is longer than a working life and longer than most available histories. Whether a 0.3 Sharpe claim can be tested at all therefore depends on a parameter that is almost never reported, which strengthens the recommendation in Section 8.4.
- **Δ ≈ 0.2** requires 33–154 years. This is marginal at the top of the ρ range and out of reach below it.
- **Δ ≈ 0.1 or below** requires 128–610 years, which no existing sample can provide at any ρ I observed.

The trouble is that this last band is where tactical allocation overlays are usually pitched and judged. A manager who claims to add 0.1 Sharpe is making a claim that no backtest can verify or refute.

### 8.2 The false-positive consequence

When power is low, the ratio of true to false positives among *significant* results deteriorates, because a low-powered study rarely detects real effects while keeping its full 5% false-positive rate. Once publication filters on significance, the published literature in the Δ ≈ 0.1 band will be dominated by false positives.

None of this requires misconduct or data mining. It follows from sample sizes and effect sizes alone.

### 8.3 For methodological priorities

A lot of effort in this literature, including much of mine in this project, goes into eliminating look-ahead bias. My ablation found the joint effect of the three main channels **not distinguishable from zero** (point estimate −0.001, 95% CI [−0.096, +0.098], p = 0.998, with nothing surviving correction). I quote the interval on purpose. The point estimate lands near zero partly by chance, given that cell A alone is −0.102, and a single point estimate at this power cannot support a claim about magnitude.

This result needs to be read with care. It does **not** show that look-ahead bias is unimportant, because my power against effects of that size was 5–7% for the relevant comparisons. What it shows is that the effort changed no conclusion in this study, and that removing look-ahead bias does not by itself make a study's results informative.

Selection looks like a larger threat. The specification-sensitivity result in Section 4.2, a 0.166 swing that reverses sign after the defensible decision to include gold, spans a wider range than any bias I introduced on purpose. As that section notes, the swing describes how unstable the estimate is and should not be read as a measured effect.

### 8.4 Recommendations

Studies should report power alongside significance, giving for each comparison the effect size they could have detected, in the form of Section 7.4. At the power levels found here (5% to 34%), a null result says very little, and presenting one as evidence that no effect exists is a mistake.

They should also report ρ. The required sample depends heavily on it, yet it rarely appears in published comparisons. A comparison at ρ = 0.95 needs roughly a third of the data of one at ρ = 0.85.

The asset universe and sample window should be fixed in advance, with the sensitivity to each reported, since either can move the estimate by more than the effect being studied.

Finally, research effort is better spent where the expected effects are large enough to see. A strategy with an expected edge of 0.1 Sharpe may be real and worth running, but a backtest cannot validate it, so the case for it has to be made on other grounds.

---

## 9. Limitations

My null results are underpowered, including those that happen to support my argument. I cannot conclude that regime models fail to add value, or that look-ahead bias is immaterial, and reading my nulls as findings would repeat the error the paper describes.

Data revisions are not addressed. Switch A captures publication timing only, and since revisions tend to be largest at turning points, my look-ahead estimates should be treated as lower bounds.

The emission model is probably misspecified. The steady decline in BIC (Section 3.5) suggests the Gaussian emission is inadequate, and I did not try a *t*-distributed or mixture emission.

Regime weights in the primary analysis are set by hand, so the null applies to those particular weights.

Several series are proxies. Bond returns are reconstructed from duration and ignore convexity and roll-down, and the commodity proxy is a price index rather than an investable return. Gold lacks 1970s coverage and is left out of the primary analysis. That removes the asset most associated with stagflation, and Section 4.2 shows that including it moves the estimate more than anything else.

The study covers a single economy (the United States) at monthly frequency, with three to five assets.

The power analysis assumes the true effect is stable over time. If the benefit of regime allocation is concentrated in a few rare episodes, my simulation understates how detectable that pattern is, although a stable estimate of it would need an even longer sample.

The closed form assumes i.i.d. returns. I correct for this empirically with a standard-error ratio of 1.248 (variance 1.558) measured at pinned parameters in Section 7.2, but I do not model the dependence structure analytically. Because that ratio mixes non-normality with serial dependence and varies across comparisons (1.15–1.34), the Section 7.3 figures are approximate. Ledoit and Wolf (2008) offer a more rigorous approach.

Neither the power argument nor the sample-length expression is new. Harvey and Liu (2015), Harvey, Liu and Zhu (2016), and Bailey and López de Prado (2014) make closely related points about backtest inference, and Opdyke (2007) and Bailey and López de Prado (2012) give closely related minimum-sample results. What this paper adds is the application of the expression across realistic values of ρ, the per-comparison detectability table, and a complete, reproducible pipeline to which both are applied, including my own results.

---

## 10. Conclusion

I built a macroeconomic regime model more carefully than is usual, with point-in-time data, forward-filtered inference, walk-forward re-estimation, adversarial tests for look-ahead, and model selection based on stability. I then checked whether that care made any difference.

It made no measurable difference, and neither did the strategy. The fully naive implementation performs about the same as the careful one, and both perform about the same as a rebalanced 60/40.

These results cannot say whether regime models work, because a backtest of realistic length cannot resolve differences of the size in question. Detecting a 0.1 Sharpe improvement needs between one and six centuries of monthly data, depending on a correlation that the literature rarely reports. My study covered 44.6 years and had between a 5% and a 34% chance of detecting the effects it actually observed.

Large effects can still be tested, though not in every case. A strategy promising 0.5 Sharpe over a benchmark can be evaluated in six years at ρ = 0.95, but needs twenty-five at ρ = 0.76. The real difficulty is with debates over 0.1 Sharpe differences based on 25-year samples, where the quantity in dispute cannot be estimated from the data available. Better data hygiene does not solve this; in this study it changed nothing. What would help is focusing on questions a backtest can actually answer, and being more cautious about which questions have already been settled.

---

## Declaration of generative AI and AI-assisted technologies

During this project I used Anthropic's Claude, through the Claude Code tool, to write and debug code, draft and edit text, and review calculations. I set the research questions, made the methodological decisions, and checked the results, and I take full responsibility for the content of the paper.

I did not rely on the tool's output being correct. Every figure in the paper is generated by code in the public repository and can be regenerated from the committed data with the commands in Appendix A. The test suite in `tests/test_units_and_tables.py` recomputes each published table value from its inputs and fails if the paper and the code disagree, so an error in the reported numbers shows up as a failing test whoever introduced it. Readers can run the suite and check the results for themselves.

---

## References

Ang, A., & Bekaert, G. (2002). International asset allocation with regime shifts. *Review of Financial Studies*, 15(4), 1137–1187.

Bailey, D. H., & López de Prado, M. (2012). The Sharpe ratio efficient frontier. *Journal of Risk*, 15(2), 3–44.

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

Opdyke, J. D. (2007). Comparing Sharpe ratios: So where are the p-values? *Journal of Asset Management*, 8(5), 308–336.

Politis, D. N., & Romano, J. P. (1994). The stationary bootstrap. *Journal of the American Statistical Association*, 89(428), 1303–1313.

Welch, I., & Goyal, A. (2008). A comprehensive look at the empirical performance of equity premium prediction. *Review of Financial Studies*, 21(4), 1455–1508.

---

## Appendix A: Reproducibility

All results are generated from a committed data cache and require no API credentials. The code, data and test suite are archived on Zenodo (DOI 10.5281/zenodo.22737848, release v1.0-preprint), so every figure can be reproduced from a fixed snapshot rather than a moving branch. That archive was made from an earlier version of this manuscript, which differs from this one in wording, in its coverage of the literature, and in a revision log that has since been removed. No figure, table or result has changed. The results are reproduced with:

```bash
python -m macroregime.run                # pipeline and primary results
python -m macroregime.evaluation.power   # Section 7
python -m macroregime.viz.report         # static HTML report
pytest                                   # 50 tests: look-ahead proofs, unit
                                         # conventions, and published-table regression
```

Environment: Python 3.12, `hmmlearn` 0.3.3, `statsmodels` 0.15.0, `numpy` 2.5.2, `pandas` 3.0.5. Seeds are fixed throughout; HMM estimation uses 20 initialisations selected by log-likelihood.

### A.1 Look-ahead test suite

The look-ahead tests check the property directly. The main test is adversarial: every observation after a cutoff is replaced with implausible values, the feature matrix is rebuilt, and all earlier rows must be bit-identical. A second test removes the trade lag and requires performance to *improve*, which confirms that the lag actually binds.

### A.2 Unit and table-regression tests

Errors of scale, such as a factor applied in the wrong power, are easy to miss on reading but show up immediately when a published number is recomputed from its primitives. `tests/test_units_and_tables.py` does that recomputation automatically. It checks that required sample size scales with the *square* of a standard-error ratio and the inverse square of the effect, that annualisation is applied exactly once, and that power against a null effect equals α. It also rebuilds all thirty cells of the Section 7.3 table from the closed form and fails if the paper and the code diverge.

### A.3 Generated artefacts

| File | Contents |
|---|---|
| `summary_long.csv` | Primary backtest; Sharpe and `return_vol_ratio` reported separately |
| `significance_long.csv` | Bootstrap tests including ρ for each comparison |
| `ablation_bootstrap.csv` | Factorial cells with CIs, Holm and BH adjusted p-values |
| `ablation_weight_modes.csv` | Fixed vs optimised weighting |
| `power_reconciliation.csv` | Section 7.2 pinned per-comparison standard errors |
| `power_years_by_rho.csv` | Section 7.3 sample-size requirements by ρ |
| `vs_sahm.csv`, `nber_lead_lag.csv` | Recession-classification validation |
| `staleness.csv` | Per-series publication lag at decision points |

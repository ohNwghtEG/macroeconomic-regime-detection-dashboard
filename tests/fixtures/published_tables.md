# Published tables

The tables below are reproduced exactly as published in the working paper
*Underpowered by Construction: Why Backtests Cannot Detect Sharpe Differences
of 0.1* (Ethan Gao, 2026). `tests/test_units_and_tables.py` rebuilds every
value from the pipeline's output and fails if any published number and the
code disagree.

**Table 1. Years of monthly data required to detect an annualised Sharpe difference Δ (80% power, α = 0.05, variance inflation 1.558)**

| ρ | Δ = 0.05 | Δ = 0.10 | Δ = 0.20 | Δ = 0.30 | Δ = 0.50 |
| --- | --- | --- | --- | --- | --- |
| 0.76 | 2434 | 610 | 154 | 69 | 25 |
| 0.80 | 2030 | 509 | 128 | 58 | 21 |
| 0.85 | 1524 | 382 | 96 | 43 | 16 |
| 0.89 | 1119 | 281 | 71 | 32 | 12 |
| 0.92 | 814 | 204 | 52 | 23 | 9 |
| 0.95 | 509 | 128 | 33 | 15 | 6 |

**Table 2. Strategy performance, December 1981 – August 2026 (535 months; equities, bonds, commodities)**

| Strategy | Ann. return | Ann. vol | Sharpe | Max DD | ρ vs 60/40 | Δ vs 60/40 | 95% CI | p |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 60/40 (rebalanced) | 10.61% | 10.32% | 0.682 | −28.45% | — | — | — | — |
| Momentum (no macro data) | 10.07% | 9.52% | 0.679 | −27.99% | 0.758 | −0.003 | [−0.256, +0.233] | 0.999 |
| HMM (walk-forward, filtered) | 10.08% | 9.93% | 0.655 | −23.30% | 0.952 | −0.027 | [−0.150, +0.097] | 0.636 |
| Rule: business cycle | 10.16% | 10.12% | 0.652 | −32.46% | 0.889 | −0.031 | [−0.197, +0.131] | 0.707 |
| Rule: growth × inflation | 8.83% | 9.79% | 0.546 | −40.97% | 0.804 | −0.136 | [−0.377, +0.110] | 0.298 |
| Equal weight | 7.78% | 7.94% | 0.529 | −32.62% | 0.866 | −0.153 | [−0.346, +0.040] | 0.121 |

**Table 5. Minimum detectable effects, strategy comparisons against 60/40 (annualised Sharpe)**

| Comparison | ρ | MDE (50%) | MDE (80%) | Observed Δ | Power at Δ = 0.10 |
| --- | --- | --- | --- | --- | --- |
| HMM (walk-forward) | 0.952 | 0.123 | 0.177 | −0.027 | 35% |
| Rule: business cycle | 0.889 | 0.165 | 0.236 | −0.031 | 22% |
| Equal weight | 0.866 | 0.192 | 0.275 | −0.153 | 17% |
| Rule: growth × inflation | 0.804 | 0.250 | 0.357 | −0.136 | 12% |
| Momentum (no macro data) | 0.758 | 0.239 | 0.342 | −0.003 | 13% |

**Table 6. Minimum detectable effects, ablation cells against the honest pipeline (fixed weights)**

| Cell | ρ vs honest | MDE (80%) | Observed Δ |
| --- | --- | --- | --- |
| A | 0.974 | 0.149 | −0.102 |
| B | 0.993 | 0.049 | +0.030 |
| C | 0.961 | 0.143 | −0.051 |
| AB | 0.982 | 0.085 | −0.041 |
| AC | 0.963 | 0.139 | −0.040 |
| BC | 0.960 | 0.146 | −0.015 |
| ABC | 0.962 | 0.139 | −0.001 |

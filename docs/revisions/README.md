# Manuscript revision history

Each file is the manuscript as it stood at that revision. They are archived here
rather than replayed as git commits, because the code history that accompanied
them was not preserved: replaying the earlier manuscripts against the current
pipeline would produce commits whose own test suite fails, since
`tests/test_units_and_tables.py` asserts that the manuscript's figures match the
generated data. Fabricated history would be worse than none in a paper whose
argument is about verifiability.

The substantive narrative is in the manuscript itself (Section 1.1), which
documents each correction and why it was made.

| File | What changed |
|---|---|
| `revision-1.md` | First complete draft. Reported **raw-return** ratios throughout: the risk-free rate was never subtracted, inflating every figure by ~0.35 and distorting the differences between strategies. Claimed "three independent methods agree" on the power result. |
| `revision-2.md` | Sharpe ratios corrected to excess returns; bootstrap CI and p-value derived from the same distribution; multiple-testing correction added; closed-form (Jobson–Korkie / Memmel) sample-size expression introduced. Two conclusions died: the momentum benchmark no longer outranked 60/40, and the specification-sensitivity result shrank. |
| `revision-3.md` | Corrected a parameterisation mismatch: the closed form, the Monte Carlo and the bootstrap had each been evaluated at a different ρ, and the residual between them had been reported as "tail inflation". Monte Carlo was reclassified as a check on the algebra rather than independent evidence. |
| `revision-4.md` | Corrected the inflation factor from a linear to a squared application — required sample scales with variance, and the measured factor is a ratio of standard errors. All Section 7.3 figures rose by 25%. Unit convention and the first table-regression tests added. |
| *(current)* `../../PAPER.md` | Registry evaluated at the published ρ grid rather than the raw maximum; number-aware matching; false-positive and conditional-correlation statistics corrected. |

Three of the four corrections were the same error — a scale factor applied in the
wrong power — which is why the current revision ships a test suite that
reconstructs every published figure from its primitives rather than relying on
careful reading.

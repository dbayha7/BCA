# Checks fixed before this stage's GPU execution

This plan was written after reviewing the separate development diagnostic. It is not a retrospective preregistration of that earlier result. No development return labels or confirmation data are read by this stage.

1. **Provenance and scope:** exact native/BCA source, checkpoint, recovery and fitting-bank identities; valid finite-ready frozen reference; no fitting/calibration group overlap; at most 2,304 candidate rows and exactly the first 32 fitting rows for gradients.
2. **Action contrasts:** total and state-centered RMS of Q1, U and simple control signals; stable feasible candidate deduplication; record unavailable zero or unresolved scales. No magnitude or performance threshold determines a winner.
3. **Constant shifts:** demonstrate unchanged mathematical ranking and centered contrasts for action-independent constants; detached offset leaves action derivatives live. Flag rounding/tie ambiguity. The separate relative usability rule is offset-sensitive and is not claimed invariant.
4. **Gradient routing:** actual Q/U action gradients; separate actor-parameter Q/U/BC gradients; actual candidate kappa-zero/native equivalence and kappa-one gradient identity; frozen reference and proposed scale-statistic gradients zero. Non-detached counterexamples must expose a changed derivative.
5. **Penalty grid:** report norm ratios, alignment and cancellation for kappa {0, .03, .1, .3, 1, 3, 10}. Action-bank contrast ratios, action-gradient ratios and actor-parameter gradient ratios remain distinct outputs. No coefficient is chosen.
6. **Coherent units:** positive multipliers {.1, 1, 10}; actual width uses a copied scaled residual unit, unchanged dimensionless threshold/parameters. Compare scores, gradients and usable centered factors. Synthetic reward/TD algebra is labeled separately.
7. **Native epsilon:** preserve 1e-7 and verify its predicted local scaling deviation; a co-scaled epsilon appears only as a labeled counterfactual. BC remains unchanged. Zero Q with a nonzero derivative must show finite native amplification.
8. **Numerical edge cases:** zero/constant/near-constant Q and U, nonfinite input, near-zero native Q, unusable gradient ratios, and float32-overflow/underflow of hypothetical factors. No silent scale fallback or production epsilon change.
9. **Immutability:** parameters/reference, source bytes and checkpoint remain unchanged; zero learner/auxiliary optimizer calls, refreshes and simulator steps. Any failure preserves the partial report.

The float32 comparison thresholds are rtol 2e-5, atol 2e-6, fixed before this GPU stage. Gradient identities use L2(error) <= atol + rtol * max(L2(left), L2(right)); scalar/array algebra uses elementwise allclose. Exact parameter/source/reference identities are checked without a numerical tolerance. Near ties and unavailable ratios are reported as such, not converted into positive evidence.

A passing result means these implementation and local scaling identities held. It does not establish a useful uncertainty signal, a good coefficient, cross-environment transfer, Q-error coverage or improved returns. Full training remains on hold.

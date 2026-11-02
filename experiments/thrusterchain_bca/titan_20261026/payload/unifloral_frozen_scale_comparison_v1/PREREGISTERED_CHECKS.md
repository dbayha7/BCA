# Protocol frozen before new development labels

The design follows earlier development and fitting audits. This is a new prospective development batch, not untouched confirmation of a selected method.

1. Verify exact sources, native baseline, core, checkpoint and passing gradient-audit receipts. Verify training/calibration/new development/old development/confirmation seed separation where applicable.
2. Enforce 64 independent starts, fixed times and shared feasible candidate banks. Preserve all state/time and continuation semantics of the validated deterministic evaluator. Stochastic policy or dynamics remain unsupported.
3. Freeze the four families, seven coefficients and three contrast factors exactly. Reject invalid calibration or unavailable factors; never silently count disabled BCA as successful.
4. Store hashes of prospective protocol, query banks, scores and selections before computing candidate returns. No result-dependent coefficient or action-bank changes.
5. Report paired gain over raw Q1 as the primary outcome, candidate-bank regret and paired control gains as secondary. Reuse 2,000 stratified episode-bootstrap draws across all methods and paired comparisons; preserve every action from a state within its episode group.
6. Check the added bootstrap implementation reproduces the existing raw-Q gain intervals before trusting new paired-control intervals. Report ties within the unchanged atol/rtol 1e-6 band as numerically inconclusive.
7. Preserve the frozen state, sources, scales and original outputs. Enforce zero optimizer and calibrator updates, no fitting/refresh, no confirmation generation, at most 576 continuations and 60,000 total simulator transitions.

There is no efficacy pass/fail threshold or automatic best-method selection in this development run. Pointwise intervals are uncorrected across the fixed grid. Exact deterministic continuation removes simulator Monte Carlo sampling error, but floating-point error is not rigorously certified. Regret refers to this finite candidate bank and frozen continuation policy only.

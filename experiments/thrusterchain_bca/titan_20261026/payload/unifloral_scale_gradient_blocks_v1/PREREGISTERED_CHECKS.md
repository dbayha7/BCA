# Checks fixed before this extension executes

This extension follows observed first-block results. Its entire 256-row fitting bank and order are fixed; it is not untouched confirmation.

1. Verify source, checkpoint, runtime and previous result hashes. Load only the authenticated fitting bank.
2. Cover rows 0 through 255 exactly once across eight equal consecutive blocks. Recompute the first block as an anchor. Generate no new candidate actions or data.
3. Preserve actual native loss reductions, coefficient, epsilon and frozen observation normalization. Verify kappa-zero decomposition and kappa-one gradient increment on every block using the actual candidate implementation.
4. Verify frozen-reference routing and report action derivatives, including any zero-gradient or undefined-ratio cases without treating them as positive evidence.
5. Keep the factor and grid fixed. Report raw and hypothetical scaled penalty gradients, BC gradients, cancellation and descriptive block variation. No desired ratio or angle determines pass status or selects kappa.
6. Aggregate raw component gradient trees by row count, then apply one full-batch coefficient. Distinguish this derived whole-batch gradient from descriptive averages of block ratios.
7. Preserve the checkpoint and sources, with zero optimizer/calibrator updates, refreshes, simulator steps, development labels or confirmation starts.

Gradient identities use L2(error) <= 2e-6 + 2e-5*max(L2(left),L2(right)). Scalar/array comparisons use the same atol/rtol elementwise. Exact hashes and frozen state use exact identity. These thresholds are unchanged from the previous stage. Nonzero width does not imply a useful penalty direction, and a passing audit does not justify normalization or full training by itself.

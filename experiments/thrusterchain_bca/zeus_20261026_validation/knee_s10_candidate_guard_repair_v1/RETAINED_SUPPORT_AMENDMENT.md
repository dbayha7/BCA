# Retained-state support repair, version 1

This is an isolated code repair, not an executed or approved G2 revision. The copied `G2_RULE_r3.md` remains historical and its all-fitting-row D3 definition must not be used to describe the repaired check.

D3's reference population is now **CUTOFF-retained fitting rows only**, including its normalization, reference groups and neighbor pool. Observations, episode IDs, times and deep-episode flags use the same exact boolean retention mask.

No numerical threshold changes: 64th-nearest-row standardized Euclidean distance; reference queries exclude their own episode; 0.99 same-time deep-reference quantile; minimum 64 reference rows at every queried time; at least 95% start coverage and 90% anchor coverage. These are rows, not 64 independent neighboring episodes. Distinct episode counts are reported in each same-time reference group.

Insufficient reference rows, malformed IDs/shapes, missing/empty/nonfinite inputs or a nonfinite threshold cannot pass. They produce INCONCLUSIVE, with missing shares null. Adequate data below the original share bars fails. No full-fitting fallback, top-up or automatic retry is allowed.

The coordinate-range B1 check remains a range diagnostic; it is not empirical joint state/action support. The deterministic reference-policy calculation is unchanged. Scientific `main()` is disabled until a separate reviewed contract is adopted and sealed. Historical outcomes, streams and reservations are untouched.

See `analysis/knee_guard_repair_v1/README.md` for verification, saved-evidence limits and the unexecuted support-only proposal.

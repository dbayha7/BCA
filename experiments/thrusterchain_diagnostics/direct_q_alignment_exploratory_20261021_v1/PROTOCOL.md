# Saved-record uncertainty alignment — exploratory protocol

2026-10-21. This protocol precedes this analysis, but the diagnostic bank and earlier results have already been inspected. The bank is exploratory, not held out, and this is not a retrospective preregistration.

Use only the completed `direct_q_diagnostic_20261021_v2` saved NPZ/CSV records, source text and reference/provenance metadata. Do not load or update a model, train, generate new oracle labels, roll out a policy, tune a penalty, or change historical files. Save reproducible analysis code, input hashes, complete numeric summaries and limitations in this new directory.

For each saved state/candidate action, distinguish signed error Q1-Qpi, absolute error, and positive-part overestimation. Audit why within-state rank correlations are defined or undefined: insufficient finite pairs, constant width, constant error, or both; separate all-zero overestimation from other constant values. Use exact ties for Spearman ranks, report full ranges and unique counts, and retain undefined correlations as null with a reason. Do not convert null correlations to zero or omit affected states.

Report pooled associations, each state separately, context/time strata where available, associations among state means, and state-centered/ranked-within-state summaries. These are descriptive diagnostics of confounding, not causal adjustment or population inference. Report denominators. Do not use candidate actions as independent trained-seed replicates or add significance claims.

Inspect within-state width variation, scale/eta floor evidence where saved data permits, and association with saved or reconstructible action distances to fixed reference actions. Distinguish observed constant widths/action-boundary saturation from unobserved derivatives or network activation saturation. Document any unavailable diagnostic instead of inventing it.

Compare Q1 with saved Q2/twin minimum against the same oracle continuation target. Document the 95k reference versus 100k checkpoint lag and the native smoothed-target versus original-actor continuation mismatch. Measure only associations and arithmetic head differences; no causal lag or alternative-calibration counterfactual is identifiable without new evidence.

Prespecify no outcome-based coefficient or method selection. The final recommendation will be one of: signal warrants a small fresh-bank confirmation, current evidence is inconclusive, or redesign the uncertainty signal. Propose the smallest independent-bank confirmation with frozen predictor/target definitions, action law, controls and complete error/coverage/refusal reporting; do not execute it here.

Before computation, parent-requested additions: examine all within-state unordered candidate pairs using the fixed 1e-6 near-tie convention, comparing width differences with advantage-error differences (signed error of action a minus signed error of action b); connect them to actual raw-Q versus Q1-U selection changes. Add the saved original-actor candidate as a fixed baseline. Interpret the result as value-error alignment, not identified epistemic uncertainty. Analyze target-noise variance or a matched target/current-network contrast only if the requisite repeated saved labels and readouts already exist; otherwise mark them unavailable and perform no additional model calls.

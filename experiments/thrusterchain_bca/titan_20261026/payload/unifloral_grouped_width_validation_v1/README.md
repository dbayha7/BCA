# Grouped incremental-width diagnostic

This is a fixed, exploratory predictive diagnostic on two previously examined
development batches: 128 starting episodes and nine candidate actions each.
It is not new confirmation. No confirmation files, policy training, simulator
calls, GPU inference, calibration updates, or new candidate actions are used.
All original sources, checkpoints and results remain unchanged.

The question is whether the existing frozen residual width adds information
about within-state signed `Q1 - V` discrepancy beyond Q margins and actor distance.
The saved V values are deterministic simulator continuations: candidate first,
then the same frozen actor. There is no conditional rollout sampling noise here,
but the critic's smoothed lagged-policy target differs from this evaluation
target. These are numerical values, not certified exact oracle values.

## Fixed design

Read `protocol.json` and its definition-freeze receipt for the exact design.
Eight predetermined folds hold out complete episodes, balanced across both
batches and four start times. All nine actions of an episode remain together.
Every feature scaling and width residualization uses training episodes only.
Centering an input across its own candidate bank uses no labels; held-out target
centering is evaluation-only.

The base linear ridge diagnostic uses centered Q1 and actor distance. The added
feature is width after a training-only linear projection removes those base
features. Standardizing that residual on training rows avoids duplicated-feature
ridge shrinkage masquerading as new information. Exactly redundant width must
collapse to the base result. The ridge coefficient is fixed at one, with no
hyperparameter selection. One fixed within-episode permutation of width provides
a negative-control realization; it is not a permutation significance test.

The primary endpoint is equal-episode held-out MSE(base) minus MSE(base+width).
Positive means better prediction, not a better policy. Pairwise ranking is a
secondary equal-episode metric, with fixed tie rules and undefined denominators.
The primary paired episode bootstrap is stratified by batch and time. Its
nominal interval conditions on the fixed predictions and models, and omits
shared-training/refitting uncertainty and prior analytical selection.

Cross-batch transfer in both directions checks stability with the same fixed
models. Eight CV splits and two transfer splits evaluate 30 predictor
specifications. There are at most 30 ridge solves and 20 width-projection solves;
inactive width skips its ridge solve and copies the base prediction exactly.
These small supervised diagnostic fits are explicitly distinct from actor,
critic, optimizer or calibrator updates, all of which remain zero.

## Execution and limits

The assistant can run the authorized Windows CPU saved-data calculation; no
manual WSL/GPU command is necessary. `prepare.py` performs synthetic tests and
seals the files. `run_checks.py --preflight` checks pinned sources and metadata.
`run_checks.py --execute` runs the audit in a CPU child with a 60-second wall
limit and one BLAS thread. Input hashing and supervisor postchecks are outside
that child limit. Every result goes into a new exclusive `results_...` folder.
`report.json` and its detailed diagnostic are authoritative only on successful
source/input postchecks. The runner reuses the exact pinned previous input
validator, not the previous outcome-analysis worker.

For local reproduction after preparation:

```powershell
& 'C:\Users\David Bayha\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe' -B 'C:\Users\David Bayha\Documents\Codex\2026-10-22\task\unifloral_grouped_width_validation_v1\run_checks.py' --execute
```

## Interpretation and repair choices

Improvement would justify investigating a better decision-use rule for the
existing width on development data, not immediately changing the production
penalty. Stability against the simple control and across batches matters.
Failure would show that this fixed linear use of width does not supply the
needed incremental signal; it does not prove that no nonlinear information
exists. Increasing the penalty cannot fix unfavorable action ordering.

If the uncertainty target needs redesign, preserve the original BCA as the
baseline and register the new candidate separately. A signed-error correction
is not calibrated uncertainty, and an earlier signed propagated-error candidate
already failed propagation/generalization checks. It is not an established fix.
Any new target must align its prediction, calibration and evaluation quantities,
and eventually demonstrate an advantage over simple controls on fresh data.
No new method, coefficient, practical effect threshold or full experiment is
selected or approved by this diagnostic. The completed confirmation remains
inconclusive and full policy training stays on hold.

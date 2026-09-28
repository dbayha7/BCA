# ReBRAC Hopper: first accepted standard host/BCA pair

Seed202609171, no importance weighting, native host losses and both Bayesian and
conformal components retained. This is a descriptive comparison of one training
seed; all five declared paired seeds remain required.

| Normalized score | Host | BCA | BCA minus host |
|---|---:|---:|---:|
| Final20-episode mean |102.023956 |102.112239 |+0.088284 |
| Mean of200 periodic banks |92.521521 |93.032778 |+0.511257 |

BCA is ahead in10/20 paired final episodes, with no ties. The final difference
is small. These results do not establish a reliable benefit or better handling
of OOD actions. Evaluation episodes share a single trained pair; they do not
provide five-seed uncertainty. No action- or episode-as-training-seed interval
is reported. The prospective v2 action-support experiment remains unstarted and
its one-stream amendment remains awaiting explicit direction.

The saved BCA run completed1M host/critic updates,500k actor updates,1M accepted
scale fits and198 posterior refreshes, with zero abstained fits. Worker actual
exit0 at2026-09-27T21:26:41.063448+00:00. All three checkpoints,108 frozen source
files, actual data/conversion/partitions/reference selection, all1M metric rows,
phase/update/refresh/evaluation order,2,020 recorded evaluation episodes and
checkpoint RNG/posterior/radius identities passed the saved CPU audit. The closed
host audit and exact exports were reused; the host was not rerun or repulled.

Final saved effective/Bayesian radius1.0380635261535645, conformal
0.9678961038589478. Bayesian exceeds conformal at all198 recorded refreshes.
The frozen-reference floor-dose diagnostic changes at0/198 refreshes; this does
not remove the conformal component or establish a causal explanation. Final
live residual scale0.8828000426292419 and frozen scale0.8921124339103699 retain
their distinct meanings. Scale/radius values alone are not OOD ranking evidence.

The independent audit and separate export arithmetic review both exited0.
Eight synthetic audit guard tests passed, with no skips. No model query,
simulator call, learner update or GPU request was added. This is acceptance of
recorded training evidence, not a training replay or OOD execution acceptance.

Evidence limits: checkpoint calibrator mean/std arrays exactly match preparation
hashes and the frozen training-only preparation source; GPU population-reduction
arithmetic was not replayed. Full posterior hashes were independently rebuilt
at the three saved checkpoints. ReBRAC does not archive per-scan posterior
hashes or every refresh's residual-unit contents/explicit row IDs; none were
invented. Every refresh key/radius/reference/holdout identity was checked. Native
target parameters are embedded and finite; there is no separate saved target
counter. Full journal state hashes are preserved without claiming an exact
reconstruction across different serialized field orders. Fresh ReBRAC targets
requiring unavailable recorded next actions remain unavailable.

Preserved preparation failures: initial metadata inspection exceeded its40MB
guard; observed preparation size138,550,556 bytes then allowed a separately
versioned150MB read-only schema inspection. Initial remote bootstrap exited1
on a string-quoting syntax error before executing any audit; a separate corrected
bootstrap compiled before dispatch. The real scientific-input audit ran once,
unchanged, and exited0. Its stderr retains optional CUDA-plugin discovery's
CUDA_ERROR_NO_DEVICE with CUDA_VISIBLE_DEVICES empty/JAX_PLATFORMS cpu; CPU key
checks and the audit completed successfully. No scientific retry is inferred.

The six exact JSON exports are under verified-v1. The evidence package retains
code, tests, actual exits, failed preparation receipts, independent review and
publication integrity prerequisites; no checkpoint weights, SQLite or NPZ.

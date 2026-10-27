# Zeus frozen operator-alignment result

The single authorized audit completed successfully. Its scientific result is unfavorable to a simple explanation that the radius already measures useful action uncertainty and only needs a different critic target: **within each state's nine candidates, the radius has weak linear association with its native absolute TD residual, and that absolute residual has almost no linear association with signed Q1 error.** Positive pooled association and high residual containment do not establish useful action selection. No correction, normalization, model or training configuration was changed.

This is exploratory mechanism evidence on the existing development banks. The prior locked confirmation remains inconclusive against actor-distance; this audit supplies no new confirmation or policy-benefit result. Full training remains paused.

**Subsequent authorized follow-up:** the one fixed cached target/action-scoring comparison below was completed once. Its privileged mean-target control passed the two-batch exploratory screen; the full results, harms, verification and one proposed deployable Q-penalty candidate are in [the companion report](cached_score_mechanism_v1/RESULTS.md). It used zero new transitions, model forwards, fitting or training. This does not change the original operator audit's correlations or validate a trained BCA policy.

**Current decision:** a separately authorized [lower-only feasibility check](cached_lower_only_v1/RESULTS.md) also passed both batches and is fully verified. The user's BCA-preservation clarification supersedes the generic correction proposal. The current proposal retains local uncertainty modeling and Bayesian/conformal calibration while investigating a Q1-centered one-sided residual bound. No fit or method change has been executed.

## Progress toward the main objective

| Question | Current evidence | What remains unproven |
|---|---|---|
| Are we running the intended Titan UniFloral model and operator correctly? | Source/checkpoint/normalization identity, restoration, forward replay, target arithmetic and unchanged state passed. | This does not validate every future training run or platform. |
| Does the radius measure its native residual? | Modest pooled fitting association; weak linear association among development candidates within states. | Fresh held-out native-target generalization and useful action discrimination. |
| Does this information improve decisions? | Earlier signed correction did not establish utility; the present audit is descriptive. | A reliable advantage over Q1 and actor-distance controls. |
| Does BCA improve trained-policy return or robustness under distribution shift? | No new policy training or OOD evaluation occurred here; the audited checkpoint has zero BCA penalty. | Matched policy experiments and OOD tests. No completion percentage is justified. |

The fitting-bank measurements are complete for the authorized 256 rows. The missing evidence is independent validation, not an unexecuted part of this audit: those rows contributed to fitting, and the four declared draws are not the original consumed refresh draws. We do not need every offline correlation to become positive before considering a small policy pilot. A pilot should be judged on its direct return/robustness endpoint and budget, with baseline controls; the current full-training hold still applies until such a pilot is separately specified and authorized.

## What was tested

The exact transferred standalone UniFloral checkpoint was restored, including current and target networks, reference state, normalization, optimizer state and RNG. The saved development bank contains 128 states, nine existing actions per state, across two batches and query times 0, 25, 50 and 75. The candidates are the frozen actor action, zero action and seven seeded uniform actions. Saved values `V` are deterministic current-actor continuations for the finite 100-step task, discount 0.98, with no terminal bootstrap.

For each existing candidate, one evaluator step supplied reward and next state. Four prespecified target-policy smoothing draws produced the native target:

```text
y_j = r + 0.98 * (1 - done) * min(Q1_target(s', a'_j), Q2_target(s', a'_j))
R_j = current min(Q1, Q2) - y_j
H   = current Q1 - current min(Q1, Q2)
G_j = y_j - V
E   = current Q1 - V = R_j + H + G_j
```

`U` is the frozen residual **radius**, despite the implementation's width name. Its target is `abs(R_j)`; containment is `abs(R_j) <= U`, with no factor of two. Harmful optimism is `positive E = max(E, 0)`. Target noise uses native standard deviation 0.2, clipping at 0.5 and action clipping at 1.0. The four new declared draws do not reproduce the noise consumed at the original refresh and do not estimate the exact expected target.

The other bank contains 256 already recorded fitting transitions from 221 episode groups. No simulator calls were used for it. These are training rows, not held-out calibration or test data.

## Core results

Development correlations below are **Pearson correlations within nine candidates**, averaged equally over states where defined, then descriptively averaged over the four draws. They are not rank correlations, independent action samples or confidence intervals.

| Development relationship | Four-draw mean | Draw minimum to maximum | Defined states |
|---|---:|---:|---:|
| U vs absolute native residual | 0.068565 | 0.038244 to 0.103938 | 128 |
| Absolute native residual vs signed E | 0.008895 | -0.028636 to 0.046989 | 128 |
| Absolute native residual vs positive E | 0.082144 | 0.059852 to 0.111114 | 99 |
| U vs signed E | -0.161308 | Unchanged across draws | 128 |
| U vs positive E | -0.201139 | Unchanged across draws | 99 |
| U vs absolute E | -0.159347 | Unchanged across draws | 128 |

The 29 undefined positive-error correlations have constant outcomes and are explicitly excluded, not replaced by zero. The negative U–E associations do not support treating a larger radius as a reliable signed optimism correction on this bank. They also do not contradict earlier incremental prediction diagnostics: those answer different questions and did not establish decision benefit.

On fitting rows, pooled `corr(U, abs(R))` is **0.219644**, range 0.185418–0.235534. This uses a different aggregation from the within-state development metric. An independent, **posthoc** saved-array check gives development pooled correlation **0.298293**, range 0.272952–0.329711. Therefore, 0.219644 versus 0.068565 is not evidence of a simple generalization decline. Pooling can include differences between states; the decision-relevant concern here is weak linear discrimination among actions at the same state. The fitting and candidate banks also differ in composition: five fitting rows are terminal, versus none of the development one-step rows. There is no fresh controlled distribution-shift validation in this audit.

Residual containment is **98.1554%** on development: `[1123, 1131, 1132, 1137] / 1152` by draw. Fitting containment is **94.9219%** row weighted, `[242, 244, 243, 243] / 256`, and **94.8152%** with equal episode-group weights. The fitting groups comprise 189 singletons, 30 pairs, one triple and one quadruple. Containment describes this finite bank and is neither a held-out coverage guarantee nor evidence that a penalty selects better actions.

Fitting U has mean 5.095184; its absolute residual has four-draw mean 2.160735. Supplementary posthoc development means are U 4.432635 and absolute residual 1.348324. These magnitude comparisons help explain why high containment can coexist with poor action discrimination; they do not identify a cause or justify changing the scale.

## Critic heads, residual and bootstrap gap

Every one of the 4,608 candidate/draw accounting identities passed. Maximum absolute error in `E - (R + H + G)` was **1.7763568394002505e-15**.

| Term | Mean state-centered action RMS | Mean within-state Pearson with E |
|---|---:|---:|
| E: current-Q1 value error | 1.771757 | 1 |
| R: signed native residual | 1.119713 | 0.318950 |
| H: current critic-head gap | 0.422146 | 0.200634 |
| G: native target minus saved continuation | 1.681974 | 0.703708 |

H correlations are defined for 101 states; 27 have constant H. H is smaller in centered action magnitude than R or G, so the current Q1-versus-twin-min difference alone is not the largest term by this descriptive measure. G varies across actions on a scale similar to E. But G and E share `-V`, the decomposition is algebraic, and terms can cancel. These numbers are not causal shares, a variance partition, or an independent prediction result.

G combines target-critic approximation, target-network lag, target-policy versus current-policy differences, smoothing, and one-step bootstrap versus full continuation. This audit does not isolate those mechanisms. The final reference/current critic snapshot gap is verified zero; that does **not** eliminate target-network lag. Supplementary finite-four-draw target variability has mean per-candidate population SD **0.625320** on development and **0.974316** on fitting rows. This measures smoothing sensitivity only. Saved V has no conditional Monte Carlo return noise; numerical error and variation across states or training seeds are separate issues. `mean(abs(R))` was retained, never replaced by `abs(mean(R))`.

## Every prespecified stratum

Values below are descriptive four-draw means. U–E is fixed across draws. The full [CSV](all_prespecified_strata.csv) contains all 840 metric records across the 15 strata and four draws, including every draw, RMS component, undefined count and reason. [summary.json](summary.json) preserves the structured summaries and draw ranges.

| Stratum | States | U vs abs(R) | U vs E | abs(R) vs E | G vs E | Containment |
|---|---:|---:|---:|---:|---:|---:|
| Overall | 128 | 0.068565 | -0.161308 | 0.008895 | 0.703708 | 98.1554% |
| Development 1 | 64 | 0.063358 | -0.191163 | 0.031937 | 0.690529 | 97.6562% |
| Development 2 | 64 | 0.073772 | -0.131453 | -0.014146 | 0.716887 | 98.6545% |
| Time 0 | 32 | 0.113410 | -0.025101 | -0.129069 | 0.642834 | 97.2222% |
| Time 25 | 32 | 0.108929 | -0.317881 | 0.026416 | 0.737702 | 97.5694% |
| Time 50 | 32 | -0.047024 | -0.384094 | 0.103563 | 0.754215 | 98.4375% |
| Time 75 | 32 | 0.098946 | 0.081844 | 0.034672 | 0.680081 | 99.3924% |
| Development 1 / time 0 | 16 | 0.134515 | -0.114345 | -0.155792 | 0.575527 | 95.6597% |
| Development 1 / time 25 | 16 | 0.108335 | -0.409567 | 0.052976 | 0.734882 | 96.5278% |
| Development 1 / time 50 | 16 | -0.024961 | -0.265742 | 0.096257 | 0.754367 | 99.1319% |
| Development 1 / time 75 | 16 | 0.035545 | 0.025002 | 0.134308 | 0.697339 | 99.3056% |
| Development 2 / time 0 | 16 | 0.092306 | 0.064143 | -0.102346 | 0.710142 | 98.7847% |
| Development 2 / time 25 | 16 | 0.109524 | -0.226195 | -0.000144 | 0.740521 | 98.6111% |
| Development 2 / time 50 | 16 | -0.069088 | -0.502445 | 0.110870 | 0.754062 | 97.7431% |
| Development 2 / time 75 | 16 | 0.162348 | 0.138686 | -0.064964 | 0.662823 | 99.4792% |

U–E is negative at times 25 and 50 in both batches, weakly positive at time 75, and mixed at time 0. At time 50, U–abs(R) is negative on average in both batches. These descriptive differences prevent treating the overall mean as uniform behavior. The 16-state cells are exploratory; no favorable cell or draw was selected as a result.

## Integrity, runtime and resource evidence

The report, worker, wrapper and execution ledger agree on completion. Receipt/hash links, raw-array hashes and before/after identities agree; full snapshots, canonical parameter leaves, reference state, normalization, RNG and counters are unchanged. Current-actor, Q and U replays passed the frozen tolerances; maximum differences were 1.19209e-7, 1.52588e-5 and 9.53674e-7, respectively. Native float32 target reconstruction had maximum difference zero. The independent saved-array review matched 38 array descriptors/hashes, 12,295 numeric comparisons, all 512 state/draw records and all 60 stratum/draw summaries.

The loaded source was the transferred `unifloral_td3bc_validation_v1/upstream/algorithms/td3_bc.py`, native revision `f2dc1278eae18ed3e22c92255119c54885414d17`. Source SHA-256: `6f5ab6f98d706d902a18f11e1c81e99e2691dc4c8a53843bf6521c9f95d00eee`. Full checkpoint SHA-256: `874d65b7f4dcb5ec285a0d4360439c73a787b5e8c567ff7c0138e0c1e22948e8`. This is the same Titan UniFloral model; no CORL code, weights or normalization were substituted. Historical checkpoint counters remain 20,000 critic, 10,000 actor and 20,000 auxiliary updates. The checkpoint's kappa penalty is zero, so this is not evidence from training a BCA-penalized actor.

Repository HEAD remains `448cce46c922862cd9177f28e982ac0898bb38ca` on `codex/thrusterchain-bca-handoff-20261026`. The transferred six-member runtime archive has 32,601,033 bytes and SHA-256 `5c2b44d3ea9b52b49bcd1251a3aa733e399fe91e9e2696f282f6a3f5ab3d3d78`; its restoration and source/normalization checks are recorded in the validation receipts. There are no tracked source edits.

The run used the isolated `/home/david/bca-operator-audit-20261026` environment in Zeus's Ubuntu WSL distro: Python 3.12.14, RTX 5070 Ti Laptop GPU, JAX/jaxlib/CUDA plugin/PJRT 0.6.2, NumPy 2.5.4, Flax 0.10.7, Optax 0.2.8, Distrax 0.1.5, Chex 0.1.90, TensorFlow Probability 0.25.0, SciPy 1.18.1 and Gymnasium 1.0.0. Highest matmul precision, JIT enabled, x64 disabled and the frozen XLA flag were preserved. This runtime is outside upstream JAX/NumPy constraints, as disclosed on Titan. The reviewed Zeus transitive package lock is not claimed identical to Titan. `corlverify` supplied only the Python executable used to create an isolated environment, with no inherited site packages.

| Resource | Actual | Frozen limit / interpretation |
|---|---:|---|
| Worker elapsed | 42.852852 seconds | 300 seconds |
| Supervisor elapsed | 46.505334 seconds | Includes orchestration; distinct from worker cap |
| One-step evaluator transitions | 1,152 attempted and completed | 1,152 |
| Current actor rows | 128 | Diagnostic counter |
| Current critic / radius / target actor rows | 1,408 each | 1,152 development + 256 fitting |
| Twin-target-critic input rows | 5,632 | 5,632 |
| Smoothing draws | 8 bank-level draws | Four per bank, not independent episodes |
| New full continuations / candidate actions | 0 / 0 | No new rollout or proposal generation |
| Fitting / optimizer / calibration updates | 0 / 0 / 0 | Model and reference unchanged |
| Confirmation files read | 0 | Confirmation remains untouched |

Model-row counters describe diagnostic work, not every restoration/template-validation forward. No new training or model inference was performed during the independent arithmetic review; that review used saved numeric arrays only.

Report SHA-256: `c11b32408b59815cb7a75585b9c3aeacb97655827e40ab1110250a981adcefd7`.

Checks SHA-256: `2f8afe4e1057f7a6bc84405039090fa6a9ff350d03dfdce3b2e696e0830ed529`.

Raw NPZ SHA-256: `cb87c6fe809e94d38715fc45df07605a386222fb9bc2c9a33059c708463d576b`.

Protocol SHA-256: `b30d1c5cb88004135070f12050bb97b3b1059f436017f25ca005bf52087f3aa2`.

The original [report](../../titan_20261026/payload/unifloral_operator_alignment_v1/results_20261026T221700_762125Z/report.json), [checks](../../titan_20261026/payload/unifloral_operator_alignment_v1/results_20261026T221700_762125Z/checks.json), [wrapper result](../OPERATOR_ATTEMPT_RESULT.json) and [identity record](../unifloral_identity.json) remain unchanged. To reproduce the independent saved-array arithmetic from the repository root, use the existing native NumPy environment:

```powershell
.\.venv\Scripts\python.exe -B experiments/thrusterchain_bca/zeus_20261026_validation/operator_analysis/verify_saved_arrays.py
```

This command verifies saved evidence only. It is not the audit launcher. The permanent one-attempt reservation remains in place; do not launch another audit or delete its reservation. See [independent_review.json](independent_review.json) for the separately labeled posthoc calculations.

## Prespecified follow-up — subsequently completed

Freeze one cached-bank decision comparison before calculating its outcomes: choose the action maximizing the **fixed four-draw mean native target**, with raw-Q1, current twin-min and the frozen actor action (minimum squared actor distance) as fixed controls. Preserve the same 128-by-nine bank, original candidate order and existing tie tolerance; use the lowest original candidate index for ties. Report equal-state paired saved-V gain versus Q1, candidate-bank regret, action-switch rate, harmful switches and all 15 strata. Do not select a smoothing draw, coefficient or favorable stratum. The actor control here has no fitted penalty coefficient; it is not a newly tuned version of the earlier actor-distance comparison.

This directly checks whether changing the score's operator helps decisions on the saved bank before designing a new correction target. The mean native target uses evaluator-derived reward and next-state information: it is a privileged mechanism control, not a deployable repair or policy-benefit claim. Its gap G still remains. The comparison required **zero new transitions, rollouts, fits, calibration changes or confirmation reads**, and left original BCA intact. Its observed gain remains development-only evidence requiring a separately reviewed validation plan. It was subsequently authorized and executed once; full training remains on hold.

**Falsifiable question and stopping rule:** does the fixed mean-target selector improve saved continuation value over Q1 in both existing development batches? Freeze that single primary comparison before reading its selected-action outcomes. For each batch, require equal-state mean paired gain to exceed the mean numerical tie allowance `1e-6 + 1e-6 * max(abs(V_target_selected), abs(V_Q1_selected))`. This is a numerical direction-consistency rule, not a confidence bound or efficacy threshold. Evaluate once and stop: if either batch fails it, do not use this audit to justify the operator-replacement repair, tune a coefficient, choose a favorable draw/control, or append another diagnostic sequence. If both pass, consider one concrete deployable repair/pilot proposal; the privileged mean-target selector itself cannot be deployed. Report all strata and harmful switches, but do not require every stratum or offline predictor to succeed. Either outcome ends this mechanism-test branch and allows an explicit decision about a bounded original-BCA policy pilot; it does not authorize or permanently veto one.

There is no remaining blocker to reading this completed result. Assistant WSL and broad process-inspection access remain denied and were not retried; the user manually executed the approved isolated setup, readiness probe and single reserved audit. No new results were committed or pushed.

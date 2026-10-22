# One choice improves; adaptive uncertainty benefit remains unproven

Completed 384/384 candidate continuations on 32 designed states: 21,696 main plant transitions and 608 validation transitions. New training updates: 0.

For Q1 minus width versus raw Q1, the mean selected-action true-Q difference is **0.1538**, with the predeclared paired state-bootstrap interval **[0.0000, 0.4615]**. There are 1 improved, 0 harmed and 31 unchanged state-level values.

These are one-action deviations followed by the original actor. They are not returns of a new policy that repeatedly chooses Q-minus-width actions. No actor or critic was updated.

The sole full-bank improvement is state 6 (stall/low load, time 0): width changes candidate 2 to candidate 6, the original actor action, gaining 4.922153. This restores the original actor choice rather than demonstrating a new learned-policy improvement. The distance control has a larger descriptive mean gain (0.405389), but no direct distance-versus-width superiority test was planned.

The common-grid restriction exposes a negative case: state 18 (monotone/low load, time 99) loses 0.3. Across its 13 eligible states, width changes no state for the better and one for the worse; the mean is −0.023077 and its interval is unavailable because some strata contain only one eligible state. On unsupported commands, one selection improves while mean pairwise concordance worsens from 0.65625 to 0.65.

## Placement: what was previously tested, and what changes here

Native BCA multiplies the behavior-cloning loss by a detached width-derived dose. That changes the strength of copying logged actions; it does not directly subtract width from the actor-action Q1 score. The prior mechanical stage passed 18/18 tests and showed the proposed live width has the intended action derivative. Its actual-checkpoint audit found a nonzero live-minus-host actor-gradient norm, while a detached additive width reproduced the host gradient. This proves a mechanical path, not that the width ranks harmful actions correctly. The present diagnostic checks that separate ranking question without gradients or learning.

## Paired candidate decisions

| Score | Mean true-Q change vs raw Q1 | Paired 95% interval | Improved / harmed / unchanged | Candidate choice changed | Mean candidate opportunity loss | Mean advantage vs original actor | Mean pairwise concordance |
|---|---|---|---|---|---|---|---|
| Raw Q1 | 0.0000 | [0.0000, 0.0000] | 0 / 0 / 32 | 0 | 3.1365 | 0.1641 | 0.6619 |
| Q1 minus width | 0.1538 | [0.0000, 0.4615] | 1 / 0 / 31 | 1 | 2.9827 | 0.3179 | 0.6619 |
| Matched constant-coefficient distance | 0.4054 | [0.0775, 0.8096] | 4 / 0 / 28 | 4 | 2.7311 | 0.5695 | 0.6771 |
| State-constant width offset | 0.0000 | [0.0000, 0.0000] | 0 / 0 / 32 | 0 | 3.1365 | 0.1641 | 0.6619 |
| Detached width (same forward score) | 0.1538 | [0.0000, 0.4615] | 1 / 0 / 31 | 1 | 2.9827 | 0.3179 | 0.6619 |

Opportunity loss is the best true Q among this state’s eligible fixed candidates minus the selected candidate’s true Q. It is not regret to a continuous-action optimum. Improvement/harm counts use the saved exact signed differences. Pairwise ranking ignores true near-ties within 1e−6 and gives predicted near-ties half credit. The original actor action is candidate 6 in every state.

The detached-width and live-width forward scores are identical. Their identical rankings are an algebraic control, not evidence that detaching leaves actor optimization unchanged. A state-constant subtraction preserves raw-Q ranking. The distance comparator uses one coefficient matched to within-state score dispersion on eight adaptation states before any oracle labels; it does not match gradient strength.

## Source → target law and score target

- Historical band source: 128 pooled rows, with one uniformly selected transition from each reserved behavior episode. Under full retention, time is uniform over 0..99 and physical states follow the behavior trajectories. Commands follow the 625-action product categorical behavior law: each coordinate assigns probability 0.84 to its mode and 0.04 to each other grid value. The calibration rows were adaptively reused, with a reference last refreshed at 95k. The calibrated quantity was a noisy one-step target residual for the current twin-min critic.
- New evaluator law: 32 independent fresh physical-state draws, balanced across four actuator-regime/load contexts; 8 states per context with times 0/50/99 represented 3/3/2. All ten position/velocity coordinates are uniform on [−0.5,0.5], current is 0.3 or 1.1, and wave amplitude is 0.2 with uniform phase.
- Each state has six distinct categorical grid actions, then the original continuous actor and five uniform continuous commands. These are designed candidates, not samples of policy occupancy or the historical calibration law.
- Scoring uses terminal100k Q1 and the historical frozen width. Truth takes the candidate once, then follows the unchanged unsmoothed actor through horizon 100 with discount 1. The target and distribution both differ from the band’s calibration target.
- No importance-ratio correction, Q-coverage certificate or support guarantee is inferred from a finite width. This is the simple Q-minus-width candidate, not Option C.

## Support, refusal and error alignment

Candidate counts: {"common_on_grid": 47, "rare_on_grid": 145, "unsupported": 192}. Native width refusals: 0/384. Failure events: 0/384. All continuations remain in the saved records.

Common means categorical mass at least 0.01; rare means positive mass below 0.01; unsupported means zero categorical mass. Classification uses actual float32 atom equality, not the candidate slot. Conditional action support does not establish state or calibration support.

| Candidate class | Defined width/positive-error correlations | Mean within-state Spearman | Positive / negative / zero |
|---|---|---|---|
| all | 10/32 (22 undefined) | -0.1549 | 2 / 8 / 0 |
| common_on_grid | 3/32 (29 undefined) | -0.3333 | 1 / 2 / 0 |
| rare_on_grid | 8/32 (24 undefined) | -0.1852 | 2 / 6 / 0 |
| unsupported | 8/32 (24 undefined) | -0.1221 | 5 / 3 / 0 |

These correlations compare width with max(Q1−true Q,0), within a state. Constant or insufficient ranks remain undefined. Only 10/32 full-bank correlations are defined ; 2 are positive and 8 negative (mean −0.154882). The other 22 are not discarded or assigned zero. A correlation does not establish calibration or policy benefit.

## Lower overestimation can conceal worse underestimation

| Score, all 384 candidates | Mean signed error | Mean positive overestimation | Mean underestimation magnitude | RMSE | Positive-error rows |
|---|---|---|---|---|---|
| Raw Q1 | -16.0066 | 1.6790 | 17.6856 | 25.6136 | 61/384 |
| Q1 minus width | -18.6278 | 1.2751 | 19.9029 | 27.3810 | 41/384 |

Subtracting a nonnegative width necessarily cannot increase positive overestimation. That arithmetic fact is not evidence of calibrated uncertainty. The raw critic mostly underestimates on this bank (only 61/384 positive errors); the penalized score can deepen those underestimates and worsen RMSE. These descriptive score errors use saved labels only and are not an interval-coverage test.

## Numerical deployment correction and failure accounting

V1 stopped before the main bank because the same observation produced batch-size-dependent native float32 actor outputs. A 2.38e−7 action difference amplified through feedback to a 0.01839038 return discrepancy. V1 used 304 fixture transitions; its preserved failure trace added 304. These failed checks remain evidence.
V2 calls the unchanged native actor on exactly one observation for candidate generation, every continuation and replay. No parameter, precision, scientific seed, budget, coefficient rule or tolerance was changed. This is a specified numerical deployment convention, not a claim of bitwise equivalence to V1’s batch-dependent policy. V2 records completed fixture phases immediately.

V2 numerical fixture results: `{"batch_scalar_max_error": 8.526512829121202e-14, "fixtures": 4, "plant_transitions": 608, "policy_action_q_equals_v_max_error": 5.684341886080802e-14, "replay_atol": 0.0001, "replay_rtol": 1e-06, "status": "passed", "terminal99_atol": 1e-10, "terminal99_max_error": 0.0}`.

## Uncertainty and negative findings that remain in scope

- The independent inference unit is the state, not each action. The fixed 10,000 bootstrap draws resample states within context×time strata and keep all actions at a sampled state together.
- Full-bank strata contain only 2–3 states. Support-class analyses rerank within the class and require at least two candidates per state. Intervals are refused if any nonempty eligible stratum contains fewer than two states; absent strata are not invented.
- Intervals are conditional on one trained checkpoint, the designed state law and the fixed candidate realization. They do not cover variation across training seeds. Deterministic numerical labels have no Monte Carlo rollout uncertainty; replay tolerance is still a numerical limitation.
- Finite native width usability is not Q coverage. The width was calibrated to a different one-step residual target, and the historical bank was adaptively reused.
- The matched-distance control matches score dispersion, not gradient strength. This comparison cannot attribute a later learning benefit to adaptive uncertainty.
- No actor update, critic-update isolation experiment, learned-policy benefit, full policy rollout under the new ranking rule, or Option C test occurred.

See `report_summary.json` for every pooled and subgroup result, undefined-correlation count, source hash and numerical receipt. All original candidate and decision rows remain in the untouched results folder.

Reproduce this report and SVG from saved records only: `python report_assets/build_report.py`. It requires NumPy to read the saved NPZ, validates output/freeze hashes, and performs no model inference or extra rollout.

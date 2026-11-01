# Signed BCA and local action compatibility: frozen-checkpoint diagnostic

**Outcome: completed; the prespecified action-selection screen was inconclusive in both contexts.** The new evaluation banks contained fewer than the required eight high-load raw-Q1 ranking mistakes in every checkpoint/context cell. The combined score improved high-band absolute Q error relative to raw Q1 in all 12 cells but was less accurate than recalibrated symmetric Q1 BCA in all 12. This diagnostic did not update or train a policy and cannot establish a policy-return effect.

## What was tested

Six saved 20,000-update hole-condition checkpoints (native and original-BCA branches, three training seeds) were frozen. A fresh signed auxiliary predicted the online Q1 minus the original smoothed one-step twin-min Bellman target. A one-sided Bayesian threshold was calibrated using 64 new behavior episodes, with another 64 independent new behavior episodes reserved for the TD-residual check. Fresh symmetric Q1 and twin-min auxiliaries from the prior diagnostic were recalibrated on the same 64 episodes. The episodes use the original recorded-start behavior law, and one randomly chosen non-hole transition is used per episode. The checks concern this behavior-row law and a one-step target, not true-Q coverage under shift.

The revised compatibility estimator used retained training data only, time-stratified 64-neighbor state matching, and a 4-D action kernel. A five-fold logged-action likelihood selected the final bandwidth multiplier; nested episode-fold exclusion selected bandwidth and distance threshold for each training matching fold. The final selected multiplier was **0.5**, giving action bandwidths approximately (0.307, 0.288, 0.288, 0.289). A 95th-percentile training-distance rule gates the added compatibility term; it is a heuristic, not proof of support or safety.

All candidate scores use the same fresh anchor families, candidate actions, saved critic, and each saved checkpoint's own deterministic-actor continuation. The six-action bank contains the actor action and five fixed complete action vectors with first coordinates -0.5, -0.25, 0, 0.25, 0.5 and all other coordinates zero. The reference is finite-horizon discounted Qπ, not Q-star. The two contexts have 32 fresh anchors each; 15 ordinary and 32 strong-current anchors were high-load. There are three independent training seeds and one retained dataset/removal mask. Shared anchors, calibration episodes, and actions are paired observations, not extra independent training seeds.

## Q accuracy and action choices

Each value below is an equal-seed mean. High-band MAE uses the three fixed in-band actions at high-load anchors; bank regret is against the best of the six candidates at all 32 anchors, in discounted-return units. Smaller is better. These are different subsets and metrics.

| Context and checkpoint group | Raw Q1 MAE | Symmetric Q1 BCA MAE | Combined MAE | Raw bank regret | Combined bank regret | Compatibility-only bank regret |
|---|---:|---:|---:|---:|---:|---:|
| Ordinary, native | 27.611 | 15.697 | 20.041 | 0.105 | 0.074 | 0.096 |
| Ordinary, BCA | 27.251 | 14.571 | 19.911 | 0.106 | 0.083 | 0.067 |
| Strong current, native | 36.070 | 25.873 | 31.100 | 0.129 | 0.119 | 0.106 |
| Strong current, BCA | 35.083 | 25.517 | 30.505 | 0.114 | 0.123 | 0.100 |

The symmetric control is `Q1 - 3*width`; the combined score is `Q1 - signed_bound - eta*compatibility`. Their magnitudes are not assumed equal. A separately reported symmetric-width control is matched to the combined score's action-gradient RMS on cross-fitted training probes.

For the BCA-trained checkpoints under strong current, the per-seed all-anchor bank regrets were:

| Score | Seed 0 | Seed 1 | Seed 2 |
|---|---:|---:|---:|
| Raw Q1 | 0.056 | 0.065 | 0.222 |
| Symmetric Q1 BCA | 0.056 | 0.114 | 0.197 |
| Signed + revised compatibility | 0.056 | 0.065 | 0.248 |
| Revised compatibility only | 0.056 | 0.024 | 0.221 |
| Strength-matched symmetric width | 0.056 | 0.114 | 0.174 |

The combined score changed 0, 0, and 1 choices on those 32-anchor banks; the one changed seed-2 choice was harmful by the ±0.1 decision tolerance. The descriptive mean can hide this seed-level variation. For ordinary current, combined bank regret was lower than raw in both equal-seed group means, but this neither satisfies the stronger prespecified gate nor proves a learned-policy gain.

The compatibility gate was active at **30/32 ordinary** and **11/32 strong-current** anchor states, identically across checkpoints because the state bank and training reference were shared. Thus the added compatibility cost was zero at 21/32 strong-current anchors. This describes the implemented heuristic; it is not an estimate of the number of truly unsupported states.

## Calibration check and decision

Across the six saved checkpoints, signed prediction reduced the mean absolute held-out one-step TD residual relative to the zero predictor by **0.028–0.174 units**. The nominal one-sided Bayesian TD check covered **61–64 of 64** new check rows. Both are descriptive on the same 64 fresh behavior episodes shared across models. They do not show that signed residuals recover full-return error, and they do not guarantee selected-action or shifted-distribution coverage.

The gate required at least eight high-load raw-Q1 bank mistakes larger than 0.1 in each eligible cell. Ordinary had 0–3 such mistakes across checkpoints; strong current had 2–7. Consequently no cell had sufficient opportunity, and both context screens are **inconclusive**. Conditional on these data, the combined score's high-band MAE was nevertheless descriptively worse than symmetric Q1 BCA by **0.30–9.55** units in each of the 12 comparisons. Do not use these fresh outcomes to tune or present the proposed model as confirmed.

## Accounting and reproducibility

The run completed exactly **197,704 simulator transitions**, including 12,800 fresh behavior transitions, 2,400 anchor-prefix steps, 104 parity steps, 144,000 forced-action continuations, and 38,400 free-policy steps. It performed **12,000 signed auxiliary updates**, **18 calibration refreshes**, and **zero actor, critic, or target optimizer updates**. The preflight separately used six discarded one-step auxiliary test updates and no simulator steps. The prototype [actor loss](actor_loss.py) is opt-in and is not wired into the training learner.

The [protocol](PROTOCOL.md), [source seal](SOURCE_SEAL.json), [fresh episode archive](diagnostic_run_01/fresh_behavior_episodes.npz), [terminal result](diagnostic_run_01/RESULT.json), per-checkpoint records, and [plot guide](plots_revision_02/FIGURE_GUIDE.md) retain the definitions, source hashes, seeds, diagnostics, and full per-probe data. The outcome is exploratory following the previous development screen. No full training study was started.

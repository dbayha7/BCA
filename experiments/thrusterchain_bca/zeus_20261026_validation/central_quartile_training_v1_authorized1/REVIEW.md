# Inherited original-package review

This review body describes central_quartile_training_v1 before its failed target validation. It is retained for provenance, not as amendment verification. Read AMENDMENT_REVIEW.md, AMENDMENT_RECONCILIATION.md and TEST_RESULTS.json for current status. Training remains held.

# Preparation verdict

**Ready for the separately bounded manual target-runtime validation; not ready for production training.** The source package is held. No production reservation, training update, stored-policy evaluation, installation, push or WSL command was executed by preparation.

The completed ordinary-task `matched_policy_pilot_v1` is preserved and never reused as a new reservation. Its positive continuation result does not validate this new conditional-hole study. The central-quartile gate is development design evidence, and `environment_audit_v1` supports the declared synthetic environment semantics only.

## Findings resolved in implementation

| Review concern | Implemented response / evidence |
|---|---|
| Removed labels leaking into preprocessing | Identical native/auxiliary initialization fixture from 60,924 fitting observations retained by both hole and random controls. Synthetic role/row tests and CPU reproduction of DATA_PLAN passed. Real source-to-model normalization equality is a pending target check. |
| A pretrained broad actor invalidating the hole study | Every seed/condition initializes fresh. Nine separate 10k warmups precede 27 branches. Pairing authenticates full model/optimizer hashes, shared statistics and source-row provenance; mismatched or resealed wrong-cell fixtures are rejected. |
| Inexact fork or changed native kernel | Full state includes online/target networks, optimizer state, PRNG, phase-held minibatch, auxiliary residual scale/posterior and actor reference. CPU toy tests cover byte-exact restoration, next-step equality, κ=0 native parity, batch reuse and refresh ordering. Actual pinned-model parity/restore remains pending target validation. |
| BC attraction mistaken for critic exploitation | Added a fixed retained-training gradient audit at 10k/15k/20k. It separates Q, BC and applied penalty parameter gradients, stopped λ, and action-gradient directions. Tests demonstrate inward BC pull with a flat critic and verify component sums against the pinned total actor gradient. These are local diagnostics, not causal training ablations. |
| An absolute value gap treated as proof that OOD actions are bad | Evaluator now reports signed center-minus-best-retained advantages, in-hole best actions, signed score margins, regret and harmful corrections. Synthetic cases include good, bad and tied hole actions. |
| General BCA gains interpreted as hole rescue | Added native Hole-minus-Random harm explicitly. Rescue flags require prespecified baseline harm, the relevant BCA gain screen, and a positive BCA-specific interaction in every seed. A fixture rejects positive interaction without native harm. |
| Different actions or continuation policies conflated | Common fixed probes use identical full four-action vectors. Own-actor slots remain separate. Own-policy Q-pi is the operational reference; a separate cached full/native/20k panel provides one fixed continuation policy per seed/context. Neither is Q-star or the smoothed lagged twin-min target. |
| Support, error and exposure inferred from separate aggregates | At four times on each free trajectory, cached remaining return, actual action, Q/adjusted-score error and fixed mask label are joined on the same visited state. This adds no simulator steps. Local empirical/joint support at those states is explicitly unknown. |
| Calibration replacement selecting forbidden rows | Exactly one eligible nonhole row per original calibration episode; 13 replacements, 128 rows shared across conditions. Empty eligible episodes stop; no outcome-dependent backfill. Adaptive reuse and lack of conditional hole guarantees are explicit. |
| Misleading strength match | Distance keeps the existing frozen-actor squared-distance definition. Its fixed coefficient matches only RMS action gradients on logged fitting probes. Zero gradient at the fork actor is tested and disclosed; no on-policy or Bayesian-specific claim. |
| Pseudoreplication and selective reporting | Three training-seed effects always shown; common starts are paired and checkpoint/candidate rows do not count as independent training seeds. All fixed checkpoints/strata/signs/harms retained. One dataset and one random-removal mask do not estimate collection/mask variability. |
| Lost interrupted work / duplicate attempts | One-shot reservation, immutable 1k checkpoints, exact exported identities, prospective counts, durable validation progress, owned-child cleanup and preserved failure receipts. No automatic retry/resume. Synthetic tests cover kill, interruption, spawn failure, invalid calibration, partial checkpoints and forged execution flags. |

## Tests actually run

The combined Windows CPU suite passed **96 tests in 36.571 seconds** using `C:\Users\David\dev\BCA\.venv\Scripts\python.exe`, followed by affected supervisor/worker/pairing checks after final accounting edits. It includes toy JAX models with the pinned actor-loss/microstep interfaces, the real reference adapter with synthetic physics, actual checkpoint byte/seal operations, and mocked startup/failure paths. These are not target CUDA tests or production update counts.

One evaluator startup-failure fixture encountered Windows `WinError 5` on an atomic rename in a temporary directory. That filesystem action was not retried through another route. The fixture now explicitly mocks atomic receipt writing to test failure control flow; actual target filesystem writes remain part of the pending validation. The bare shell `python` also lacked NumPy; testing used the already existing `.venv` rather than installing dependencies.

Data preparation decoded only the already-used 768 fitting and 128 calibration episodes (89,600 tuples); zero protected or development-probe transition records. It reproduced the passed gate's masks and calibration identities. No new policy/value outcomes were computed.

## Remaining verification and interpretation limits

1. The actual Ubuntu GPU runtime has not run this package. The manual 900-second validation must verify pinned packages/device, real-source initialization and normalization, synthetic real-model parity/restoration/gradient checks, the full Gymnasium checker and target receipt writes. Float32 finite differences can be below roundoff resolution; such a case is labeled inconclusive and must be inspected, not claimed as independent gradient confirmation.
2. The 12-hour global proposal needs explicit resource approval. Prior 70/150-minute budgets do not cover it. No production command is enabled by a successful synthetic check alone. The existing assistant-side WSL denial remains in force; only the user may launch the prepared terminal command through their authorized terminal.
3. Runtime and disk are estimates. Full-state integrity hashing, compilation and diagnostic gradients are real overhead. The 2-hour evaluator cap has not been benchmarked; any timeout is incomplete, with no automatic extension. About 405 full-state files at prior sizes use 4.19 GB before actors/logs/receipts; at least 6 GiB workspace headroom is required.
4. Random removal matches episode/time/load strata and row count, not every transition attribute. Retained terminal rows are full 768, hole 653, random 663. Reward/successor and finer state differences remain; this is a controlled restricted-data comparison, not isolated label removal at identical states.
5. Three seeds, one development-selected mask, reused calibration and prior κ choice are exploratory. The support gate is relative 13D state typicalness and marginal a0 occupancy with only 32 reference episodes per stratum; it is not dense joint support or proof of a physics-calibrated task. No Bayesian-specific efficacy claim is tested.

## Complete proposed resource envelope

| Resource | Maximum / plan |
|---|---:|
| Fresh cells / warmups | 9 |
| Branches / final trained policies | 27 |
| Critic / actor / auxiliary updates | 360,000 / 180,000 / 360,000 |
| Actor-target / critic-target updates | 180,000 each |
| Calibrations / posterior draws | 36 / 36,000 |
| Replay draws / critic-row uses / auxiliary-row uses | 46,080,000 / 92,160,000 / 92,160,000 |
| Matching rows | 6,912 |
| Gradient-audit probe rows / logical base row uses | 48,384 / 193,536 |
| Extra distance-anchor rows in gradient audit | 13,824 |
| Free episodes / forced Q-pi values | 4,608 / 24,192 |
| Simulator transitions, including prefix and parity | 1,975,304 |
| Evaluation actor / critic / width rows | 1,953,792 / 40,320 / 40,320 |
| Production global ceiling | 43,200 seconds (12 hours) |
| Training job / evaluation ceilings | 1,800 / 7,200 seconds, within global remainder |
| Separate manual validation | 900 seconds; no real-data fitting |

The fixed-reference and visited-state panels reuse existing evaluation values/trajectories and add no transitions. Query/gradient row uses are logical accounting, not FLOPs. Read-only target validation permits five fresh source initializations, capped at 448,000 repeated tuple decodes; synthetic dispatch caps remain distinct from production counters.

The minimum remaining decision is approval of this explicitly limited study and resource envelope **conditional on a passing, inspected target-runtime validation**. Do not add a density model, new method, head ablation, additional reference rollouts or a region search under that approval.


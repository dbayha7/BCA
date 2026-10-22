# Independent action-only evaluation preflight

**PREFLIGHT APPROVED** — for the reviewed source hashes and protocol below, after the driver creates its immutable freeze record. No registered replication or main result existed when this review finished. This is approval of the experiment's implementation and scope, not approval of an empirical result or a coverage theorem.

Reviewer: independent scientific/code review agent. Date: 2026-10-21 UTC. Code was read only; this report is the reviewer's only authored deliverable. The reviewer did not launch the main evaluation, retrain, alter checkpoints, or access Titan.

## Scope and decision

The proposed 200 independent calibration replications, 128 calibration draws each, 4,096 evaluation draws per mixture per replication, and mixtures 0, 0.5, 1 implement the requested controlled action shift. Each comparison conditions on the same authenticated full/replicate_01/BCA 100k checkpoint and the same finite state distribution. There are no remaining material preflight findings in the reviewed implementation.

The source checkout has changed since training. The engine instead authenticates and imports the preserved release source closure, verifies the restored checkpoint and normalization, and authenticates the original source arrays. It does not silently substitute the current checkout's implementation. The checkpoint SHA-256 is `4088278230fc413c509e27e189daa342468cfc4fb6d2479e3e8c29e149dac3d8`.

## Scientific checks

- **Common state law:** `engine.py:124` selects 64 distinct source episodes without using scores or rewards, 16 per context, with actual saved states at times 0/50/99. The time counts are 6/5/5 within each context. Fresh calibration and evaluation state IDs are uniform over those same 64 rows; these draws are not new source episodes or samples from policy occupancy.
- **Exact action measure:** `engine.py:257` constructs the 625-command grid and the original product categorical behavior probabilities, 0.84 at each coordinate's greedy atom and 0.04 elsewhere. Every grid command has positive source mass. The projected fixed actor uses nearest-grid coordinates with lower-atom ties. The target is exactly `(1-lambda)*beta + lambda*delta_projected_actor`; the driver uses its mass divided by beta without a denominator floor or weight clipping (`run_experiment.py:170`). Independent checks verified normalization, `E_beta[w]=1`, the lambda-zero identity, and `E_q[w|z]=1+lambda^2*(1/beta(h(z)|z)-1)` across all 64 states and all three mixtures. The latter identity is an audit check, not a replacement for the implemented per-query test mass.
- **Independent, paired sampling:** calibration state/action/noise streams 1–3 are separate from evaluation streams 5–8; the replicate ID namespaces every stream (`run_experiment.py:42`, `:155`). Each method receives the same calibration scores and evaluation rows. Mixture levels share state IDs, base actions, mixture uniforms, and label noise as declared. This coupling preserves the correct marginal target law at each level. Independent calibration replications, rather than the nested evaluation rows, are the inference unit.
- **One fixed label/score kernel:** `engine.py:306` and `:369` use the final 100k online twin-min critic, final target actor/critic, finite H=100/gamma=1 masks, and the same fresh native target-noise law for calibration and evaluation. Time-99 targets do not bootstrap. Frozen 95k eta and residual unit remain fixed; the old 95k scores and threshold are not reused to calibrate this comparison. Native float32 residual subtraction is followed by float64 score normalization. Clipped next-action noise belongs to the common conditional label kernel and receives no importance ratio.
- **Actual native threshold calculation:** `calibration_methods.py:132` calls the preserved native `calibration/wbcp.py` for every exact distinct positive query mass, sharing the posterior exponential stream and retaining all zero-weight calibration slots. Native common normalization, rational empirical boundary correction, the posterior crossing order statistic, and the maximum of empirical/posterior thresholds are retained. An all-zero weighted bank is an explicit no-positive-weight refusal; it is distinct from structural unsupported queries and posterior-infinite refusal. Lambda-zero equality is tested and asserted by the driver. The per-query test mass is an explicitly declared variant; the native module's scalar `E_target[w]` contract is not silently claimed to be identical.
- **Coverage and fallback semantics:** `calibration_methods.py:268` separates mathematical interval containment including infinite intervals, finite conditional coverage, and covered-and-finite fraction with all queries in the denominator. Refusal counts and finite denominators are retained. Infinite/refused bands give neutral multiplier 1. Widths and doses use matched float64 arithmetic; the protocol explicitly avoids claiming bitwise deployed FrozenReference float32 equivalence. The deployed off-grid actor has a separate exact-membership support audit and receives no fabricated finite ratios or coverage result.

## Findings resolved before approval

1. The first driver version reported importance-weight subgroup ESS and positive-weight counts under both method names. `run_experiment.py:197` now uses unit weights for the uniform method and importance weights for the weighted method.
2. The first resume check authenticated the raw file but did not bind a completed record to its frozen configuration. Records now carry the freeze SHA-256 (`run_experiment.py:212`); resume verifies that identity, replicate ID, expected raw basename/hash, all three mixture records, checkpoint, state bank, and sample sizes (`:232`).
3. Initial score normalization divided float32 arrays directly. `engine.py:386` now follows native refresh arithmetic: form the float32 residual, then convert residual and scale to float64 before division.

## Verification evidence

- Independently executed all 15 calibration adapter tests against the preserved WBCP source: **15 passed in 0.158 seconds**, CPU cores 4–7, IDLE priority, BLAS/OMP one thread. These include an independent tail-loss enumeration, explicit exponential-draw replay/chunking, exact query masses, scaling invariance, zero-weight preservation, all-one reduction, distinct refusal reasons, and coverage/fallback accounting.
- Read the six genuine-checkpoint engine fixtures and their passing transcript (`engine_test_run.txt`). They check native target/scale parity, terminal masking, saved transition and logger probability parity, outcome-blind selection, and unchanged model state. The parent additionally reported a final combined **21/21 passing** rerun; that combined rerun was not performed by this reviewer.
- Independently read the saved state bank and verified checkpoint identity, 64 unique episode IDs, 16 rows per context, 6/5/5 time counts per context, observation/state/regime/time consistency, grid projection, positive normalized beta, and the analytic ratio identities above. The preparatory throughput fixture reported finite scores; it did not evaluate registered coverage replications.
- Confirmed that neither `freeze.json` nor the main `results` directory existed at the review's identity check.

## Reviewed identities

SHA-256 values:

| File | SHA-256 |
| --- | --- |
| `calibration_methods.py` | `8596d2ec5eae975def315dbf77bdf7a792c7757d1b405111d56fd56b610236d6` |
| `engine.py` | `491bbbc8992b936e5b75fcf5991ac57faeba6166428236a793957e23d6b00d36` |
| `run_experiment.py` | `e96d8eabc042f62537d5bf04037ba53ee74f584d645d9ae6505f4cd4f62e9c26` |
| `protocol.json` | `4eb608f62a8e05b20589275e1648b5a0c80d482c16734c13178b72e8b5f004ea` |
| `engine_identity.json` | `199a9a595076e254a5b15bcf8e21d18488a0f8c84b31c62c52166048c5a2a792` |
| `state_bank.npz` | `2001f5444bb004063dce61e8f05d65defac70ecd2a05ac43385cb0cacc0bb63e` |
| `preflight.json` | `f21e3b1f917e4087218fe2157f7da9274c365126bb6d58c583165174fb5ba6fe` |

The final freeze must bind the reviewed implementation/protocol. A preparatory artifact regenerated without changing the protocol, selected arrays, or scorer may change timestamps/container bytes; retain its final identity and provenance in the freeze.

## Limits on later interpretation

This is conditional Monte Carlo evidence for one frozen trained model and one fixed finite state bank. It is neither across-training-seed inference nor a new-episode population experiment. The per-query Bayesian procedure has no frequentist guarantee imported by this review. High coverage caused by infinite intervals is vacuous and must be reported beside refusal and usable coverage. A zero-hit bank at lambda=1 is a finite-sample problem despite positive population support on the grid. No result establishes coverage for the off-grid actor, long-horizon policy Q, policy occupancy, an updated actor, or policy improvement. All 200 planned replications, failures, and undefined finite-only metrics must remain visible. Aggregate arithmetic, paired replicate-level uncertainty, completeness, and final prose require post-execution verification; no main outcome is signed off here.

# Proposed inference export contract

This is an input format for a future, separately reviewed trained UniFloral export. No producer or eligible trained checkpoint exists here. Passing the loader establishes internal consistency of the consumed bytes; it does not attest that training occurred or that provenance declarations are true. The training run, original counters, data split and export procedure still require independent review. Synthetic validation snapshots and CORL artifacts are ineligible.

`read_contract(path, protocol)` performs stdlib-only checks and returns `(manifest, components, receipt)`. Each component contains its resolved `path`, verified `bytes`, `sha256`, and, for JSON receipts, parsed `json`. `load(path, protocol)` consumes those exact bytes and returns `(policy, scale_bank, receipt)`. Each process checks its own input snapshot. Callers comparing runs must compare the returned manifest and component hashes; hashes identify bytes rather than their truth.

## Manifest

Required exact fields:

- `schema`: `unifloral.thrusterchain.inference_export.v1`
- `host`: `unifloral_standalone_td3_bc`
- `native_revision`: `f2dc1278eae18ed3e22c92255119c54885414d17`
- `native_source_sha256`: `6f5ab6f98d706d902a18f11e1c81e99e2691dc4c8a53843bf6521c9f95d00eee`
- `task`: `ThrusterChain`; `gamma`: `0.98`; `horizon`: `100`; `task_semantics`: `finite_horizon`; `observation_mode`: `time_aware_14d`
- `terminal_bootstrap`: `false`; `reward_scale`: `1.0`; `physics`: the exact protocol object, currently `{"stall": true, "stall_slope": 1.5}`
- `policy_kind`: `frozen_deterministic_current_actor`; `synthetic_fixture`: `false`; `matmul_precision`: `highest`
- `selection_rule`: `prespecified_terminal_checkpoint_no_diagnostic_return_selection`; `training_stage_reviewed`: `true`
- `optimizer_counts`: exactly `critic`, `actor`, `auxiliary`, each a positive integer; `critic == 2 * actor` denotes a complete native outer-step boundary. These are export-point counts, not aggregate validation work.
- `finite_ready_refreshes`: a positive integer.
- `train_group_ids`: a nonempty list of distinct nonnegative integers; `calibration_group_ids`: at least 128 distinct nonnegative integers, disjoint from training.
- `training_and_calibration_episode_seeds`: the complete distinct uint32-range episode-seed list, disjoint from both development and reserved confirmation episode starts. The producer must substantiate completeness from its original data records.
- `files`: five roles, `parameters`, `scale_bank`, `training_receipt`, `normalization_receipt`, `calibration_receipt`; each supplies a nonempty relative `path` inside the manifest directory and the lowercase `sha256` of its exact bytes.

Exact scalar types matter: Boolean fields cannot be numbers, integer counts cannot be Booleans or floats, and the stated floating fields use JSON floats. Duplicate JSON keys and nonfinite JSON constants are rejected. No ready-looking template with fabricated weights or receipts is provided.

## Three JSON receipts

Every receipt repeats these manifest fields exactly: `host`, `native_revision`, `native_source_sha256`, `task`, `gamma`, `horizon`, `task_semantics`, `observation_mode`, `terminal_bootstrap`, `reward_scale`, `physics`, `synthetic_fixture`, `matmul_precision`. Each also contains `parameters_sha256`, matching the declared parameter file. All receipts are themselves hash-bound by the manifest.

1. `training_receipt`: schema `unifloral.thrusterchain.training_receipt.v1`. Repeat `optimizer_counts`, `policy_kind`, `selection_rule`, `training_stage_reviewed`, `train_group_ids`, `calibration_group_ids`, `training_and_calibration_episode_seeds`, `finite_ready_refreshes`; add `scale_bank_sha256`, matching the exact NPZ. Underlying data, producer-source and training-run provenance may be recorded as additional fields and must be independently reviewed.
2. `normalization_receipt`: schema `unifloral.thrusterchain.normalization_receipt.v1`; `dtype`: `float32`; `observation_dimensions`: `14`; `rule`: `raw_native_mean_std_plus_0.001`; `training_group_ids`: exactly the manifest training groups; `obs_mean` and `obs_std`: 14 finite numeric values each. Encode the exact stored float32 values as full JSON numbers: the loader compares these values with the payload without rounding the receipt to float32 first.
3. `calibration_receipt`: schema `unifloral.thrusterchain.calibration_receipt.v1`; `training_group_ids`, `calibration_group_ids`, and `finite_ready_refreshes` must match the manifest; `reference_ready`: `true`; `reference_dtype`: `float32`. Supply exact stored float32 values for `residual_scale`, `threshold`, `lambda_hat`, `lambda_hpd`, `n_eff`. All must be finite and nonnegative; residual scale and effective sample size must be positive; threshold must equal `max(lambda_hat, lambda_hpd)`. These are the exported reference values, not higher-precision pre-storage calibration diagnostics.

## Numeric payload and scale bank

The Flax msgpack payload has exactly `actor_params`, `critic_params`, `obs_mean`, `obs_std`, `reference`. No pickle, executable checkpoint object, optimizer or initialization fallback is used.

Actor, critic and reference `cal_params` retain their complete native Flax variables dictionaries, with the outer `{"params": ...}` wrapper. The pinned native TrainState stores `network.init(...)` in `.params`; stripping that wrapper would break `Module.apply`. All parameter leaves must be finite float32. The native actor/critics have two 256-unit hidden layers; the actual state-dependent Calibrator has two 64-unit hidden layers. `obs_mean` and `obs_std` are float32 arrays of shape `(14,)`, with nonnegative standard deviations. Raw observations are normalized once using the pinned models' `std + 0.001` denominator.

`reference` contains exactly the actual `FrozenReference` fields: `cal_params`, `residual_scale`, `threshold`, `lambda_hat`, `lambda_hpd`, `n_eff`, `ready`. Scalar numeric fields are float32; `ready` is a scalar Boolean. The loader compares every stored scalar with its receipt and rejects unready, infinite, invalid or numerically unusable references. Inference uses current actor and twin critic parameters, and frozen calibration parameters/reference; no target networks or update call is needed.

The NPZ scale bank has exactly `observations` `(256,14)` and `actions` `(256,4)`, both finite float32, and `group_ids` `(256,)` with an integer dtype excluding Boolean. Actions must lie in `[-1,1]`. The unique group IDs must be a subset of declared training groups and disjoint from calibration groups. Multiple observed rows may come from one training episode, so repeated row group IDs are permitted. No returns or simulator labels belong in this bank.

The final two-row inference checks call only the frozen actor, critics and width function. They do not initialize weights, fit anything or invoke the simulator. This contract does not establish calibration coverage or usefulness of a policy learned with BCA.

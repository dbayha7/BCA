# Frozen-model action-only engine: source and validation map

The engine reads the completed `core_20261020T2250Z/jobs/replicate_01/full/bca` checkpoint at update 100000. It evaluates the final actor and critics while holding the posterior scale network and residual unit at their last successful refresh, update 95000. It performs no training, scale fitting, calibration, optimizer step, or checkpoint write. The earlier threshold is recorded for provenance and is not used to choose the new experiment's thresholds.

This is a conditional action-only experiment on 64 fixed, observed source states. Its score is a standardized one-step temporal-difference residual. It does not measure long-horizon Q error, establish policy improvement, or establish coverage under a changed state-occupancy distribution.

## Input identity and preservation

The trained source revision is `b007052a2494dc651ac00469fa9e922b287cd1e5`. The current checkout was separately observed at `30c5c68cbcfab7339ed4b69c109513e7acbb121b`; `protocol.json` records that observation. The prepared `engine_identity.json` has a null `current_checkout_head` because its loader call omitted the optional `repository_root`; the checkpoint and source hashes remain explicit and verified.

The current canonical suite's Python and JSON metadata were unavailable when preparing this experiment. The engine uses the preserved, authenticated release source and metadata in `../results_20261021/release/ThrusterChain_results_20261021`, together with surviving canonical checkpoint, normalization, and source-audit bytes. It never restores or overwrites missing canonical files. `MANIFEST.json` authenticates the preserved members, and the original job manifest authenticates the checkpoint. The original full source audit is approximately 41.4 MB; compact portability is a separate task, not a reason to alter the selected state bank or frozen engine.

| Object | Verified SHA-256 |
| --- | --- |
| `engine.py` | `491bbbc8992b936e5b75fcf5991ac57faeba6166428236a793957e23d6b00d36` |
| `tests/test_engine.py` | `25b868ef3eac2f738e8f53046e39cfeffb4162592fa54fc757ab87054067aee5` |
| `engine_test_run.txt` | `f9ee6774dc01b61d4853c3103f155f6504b883ba27af8af2140d62fd86254619` |
| `checkpoint_000100000.msgpack` | `4088278230fc413c509e27e189daa342468cfc4fb6d2479e3e8c29e149dac3d8` |
| `normalization.npz`, file bytes | `9733b58f790d03a5a0ac877cd8e7974ea44f8d8a1ca55d2f6b2656bbfde768e2` |
| Normalization arrays | `a754892f887dbc410be691b4efafaa45b090d9d005bb79ae83b501bdbcdf40f1` |
| `source_audit.npz`, file bytes | `73b306de531a4fd4a77bec88f2c41b087dbeae32f41fd842a3b168e92affec5f` |
| Source audit arrays | `d5c0b41cf9d7b623d019cc90d77c3d39a919bd73199e117860ebbfc46783891d` |
| Final actor parameters | `1352d9276bffa0d7d67371591e8e55f0f57a8640013c031845cf77e6bb648076` |
| Final target actor parameters | `05f890160bda600ab7e40a88cd86b68080c56bb1c87b847f6dd7f968bbd0c8cc` |
| Final current critic parameters | `379b1fabe4cca5fde39d4c71067129f07be0a4ad7ae741d3e2319921c0fd058a` |
| Final target critic parameters | `e9b754b2091e91ae4ff7494a1454f7642694a48968a49eb5b5d900afab1fb7bd` |
| Frozen posterior scale-network parameters | `8f90c41ae7d4e822dadcadf2be64232bb58db79c985f8bb1314c1058ce5f031d` |

The frozen residual unit is `1.4656671285629272`. All source files and model hashes are also enumerated in `engine_identity.json`; the following source hashes identify the principal code paths.

| Preserved source path, relative to release `source/` | SHA-256 |
| --- | --- |
| `algorithms/td3_bc_bca.py` | `3f8452df67b1ac51c78b67c36a6f6269c849aa9746f5e15df771d553fce0536e` |
| `algorithms/td3_bc.py` | `bf95a54dcb7402cca6533bcb0cb7c5ed5a383ef2e71a8c118e24b67310300bb8` |
| `calibration/reference.py` | `f26273784c458f058c9ebd9b4ebe1e3e7e20c1fc05abfc3297aead724e3b6b1b` |
| `experiments/thrusterchain_diagnostics/training.py` | `935db9f71326379f452b89c1f69de2cbd4920c005048c5b5d45cd3a3cbafa2b7` |
| `experiments/thrusterchain_diagnostics/training_data.py` | `bdcbdf6276745d8d0622741a4abd1ffa131191a96f881c8a4946cbded0ee6866` |
| `experiments/thrusterchain_diagnostics/continuous.py` | `bc20fb31f776cca523314b7a12f330065bc2c93f97ead7a3a4d9d3b1f9ab8315` |

## Stages and actual code-derived pseudocode

Line numbers below refer to the preserved source hashes above and the final engine hash, not to the current checkout. `suite/` abbreviates `experiments/thrusterchain_diagnostics/`.

| Stage | Purpose and pseudocode | Actual source | Invariant and evidence |
| --- | --- | --- | --- |
| E1. Authenticate and restore | Verify release member hashes; require completed full/BCA/seed2101/update100000 and source seed1101; verify source closure; initialize a serialization template; deserialize the complete authenticated carry. | `engine.py:149–181,203–254`; `suite/training.py:328–355,422–433` | No learner update occurs. Checkpoint hash, final counter, normalization arrays, actor hash, and source audit hash must agree. Identity test passes. |
| E2. Select actual source states | For each of four contexts, permute sorted episode IDs with a deterministic seed; choose 16 distinct episodes; assign times `0,50,99` cyclically. | `engine.py:124–146,256` | Seed `2026102101`; 64 unique episodes; each context has 6/5/5 rows at the three times. Selection reads episode/context/time, not rewards or outcomes. Outcome-blind fixture passes. |
| E3. Enumerate behavior support | Enumerate the lexicographic Cartesian product of five deployed action atoms; compute the original controller-centered categorical mass for each coordinate; multiply four masses. | `engine.py:103–110,257–264`; `suite/training_data.py:21,60–80` | Exactly 625 atoms. Coordinate masses are .84 at the nearest controller atom and .04 elsewhere, hence all joint probabilities are positive. Sum-to-one and logged-action probability parity fixtures pass. |
| E4. Define the target action law | Deploy the final actor with original observation normalization and float32 output; project to the nearest grid atom with ties to the lower coordinate; set `q_lambda(a|z)=(1-lambda)*beta(a|z)+lambda*1[a=h(z)]`. | `engine.py:113–121,265–266,323–329`; `suite/training.py:378–386` | Projection round-trips all 625 atoms; exact ties choose the first/lower atom. Lambda 0 equals beta and lambda 1 selects only the projected actor. The original off-grid actor and projected actor are distinct policies. |
| E5. Construct the transition | Apply each requested grid action to the stored physical source state and observed regime; obtain native next state/reward; append observed regime and remaining time; normalize using the original train-only normalizer. | `engine.py:340–354`; `suite/continuous.py:112–132,136–150` | Horizon 100, gamma 1, raw reward, terminal bootstrap masked at time99. Logged selected transitions match their original next observations and rewards exactly. |
| E6. Read fixed networks | Compute `q=min(Q1,Q2)` from the final current critic, `eta` from frozen95k posterior parameters, `sigma=max(eta,1e-6)*unit95`, and target-actor next action. | `engine.py:306–315`; `algorithms/td3_bc_bca.py:252–267`; `calibration/reference.py:148–149` | Scale is positive and in residual units; it is not raw eta. New labels change neither q nor scale. Genuine native scale/critic parity fixtures pass. |
| E7. Form the stochastic one-step label | Take caller-supplied standard-normal noise; multiply by .2 and clip to +/- .5; add to final target actor, clip action to [-1,1]; evaluate target twin minimum; set `y=r+(1-done)*Q_target_min`. | `engine.py:316–321,369–385`; `algorithms/td3_bc_bca.py:146–156` | Same native target semantics. Terminal labels equal immediate reward and ignore target noise. Native target parity is checked with the same JAX-generated noise. |
| E8. Standardize without fitting | Form `residual=y-q` in float32; convert residual and positive scale to float64; return `abs(residual)/sigma`. | `engine.py:383–389`; `algorithms/td3_bc_bca.py:263–267`; `calibration/reference.py:160–175` | Matches native subtraction/conversion/division order. Tests compare against native target, critic, and positive scale. The same path is used for calibration and evaluation labels. |
| E9. Preserve model state and audit | Return scores, signed residual, absolute residual, scale, target, q, and eta. Expose exact beta and target probabilities. The caller owns sampling, independent label-noise streams, posterior draws, and experiment records. | `engine.py:184–201,273–303,323–329,369–389` | Complete carry hash is unchanged after scoring. No actor, critic, calibrator, optimizer, PRNG counter, or posterior reference is modified. Invalid index/noise inputs are rejected. |

The optional `prepare_catalog` at `engine.py:356–367` caches deterministic model/transition reads for all 40000 state/action pairs. It does not sample labels or calibrate. Parent preflight separately exercises its use. `create_snapshot` at lines57–100 is an optional copier for an authenticated input subset; the main prepared run loads the preserved source plus canonical binary inputs directly. It does not provide a compact selected-state replay loader.

## Probability and support interpretation

For a requested target mixture, the exact action ratio is `w(z,a)=q_lambda(a|z)/beta(a|z)`. Because beta is strictly positive on all 625 atoms, every projected-actor mixture is structurally supported. Off the projected actor, the ratio is `1-lambda`; at its atom, it is `1-lambda+lambda/beta(z,h(z))`. A lambda1 calibration bank with no actor-atom hit has zero usable calibration weight despite structural support. That finite-sample refusal must not be reported as an off-support event. The unprojected continuous actor generally has zero probability under the discrete behavior law; it requires the separate off-grid support audit.

These formulas define a conditional action law at fixed observed states. They do not imply that the distribution of future states under the projected actor equals the original behavior occupancy. The engine exposes laws and scores; the runner is responsible for using independent calibration/query samples, recording seeds and row identities, and preserving paired draws across methods.

## Validation

Command, run in a fresh process before any NumPy/JAX import:

```powershell
& 'C:\Users\David\dev\BCA\.venv\Scripts\python.exe' -B 'C:\Users\David\Documents\Codex\2026-10-20\task\action_only_20261021\tests\test_engine.py'
```

The test file calls `configure_runtime` before numerical imports. On Windows this restricts only the test process to logical CPU cores8–11, IDLE priority, CPU JAX, and one numerical-library thread. It does not touch other processes or GPUs.

Result: **6 tests passed in 3.657 seconds**, recorded in `engine_test_run.txt`. The tests cover grid/projection/ties; outcome-blind episode/time selection; genuine checkpoint identity and finite support; logged transition and probability parity; native target/current critic/frozen scale/score parity including terminal masking; and complete carry immutability under changed label noise. The score parity assertions use explicit small float32/JIT tolerances, not a claim of bitwise equivalence for all accelerator/compiler versions.

This evidence verifies the engine's score and law construction. It does not itself establish that a newly calibrated threshold meets target coverage, that a larger penalty ranks harmful actions correctly, or that changing actor training would improve returns. Those are separate empirical contracts owned by the calibration kernel and frozen experiment runner.

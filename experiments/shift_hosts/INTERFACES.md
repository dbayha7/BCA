# experiments/shift_hosts: interfaces every module follows

Stage 0 of the approved plan `PLAN_UNIFLORAL.md` (15 October 2026): BCA attached to Unifloral's TD3+BC, CQL, IQL and
ReBRAC on the ThrusterChain weighting-test data, 12 setups (each learner alone, with standard BCA, with IW BCA).
"Plan 4.3" below means section 4.3 of that plan. This file is the contract the build agents write against. When a
module needs something this file does not give, write it in your report; do not edit another agent's file.

**What every module is.** An experiment-local attachment (plan 3.1). It changes none of the repository's declared
methods: the four host+BCA algorithms of `ALGORITHMS.md` stay the CORL-derived ports in `algorithms/`, and the vendored
Unifloral files in `baselines/unifloral/` (commit f2dc127, pinned in `configs/unifloral.json`) are imported unchanged
and never edited. Every module's docstring says so in one sentence.

**What is allowed now (stage 0).** Code; tests on made-up arrays (BCA's code included, approval item 8); copying the
weighting test's data with SHA-256 checks; data preparation with no learning; scoring fixed, non-learned controllers.
**Not allowed until the user's terminal approval exists** (`gate.require_run_approval()` passes): any learner step or
band fit on real ThrusterChain arrays, the learner alone included. No learned policy's return is ever produced or read
in stage 0.

Contents: 1 ownership, 2 environments and running, 3 conventions, 4 seeds and keys, 5 `unifloral_import`, 6 the
prepared-data file, 7 `handoff`, 8 `gate` and `guard`, 9 `bca_core`, 10 the restated steps, 11 `iw`, 12 scoring-side
files, 13 the run record, 14 `tests/madeup.py` and test conventions, 15 words, 16 run folders, lanes, the stage
driver's files and command lines, 17 `results.json` and where each of its fields comes from (the export/runner
mapping), 18 version 4's S7 supplement (`supplement.py`, its root, `state.json` `s7_results` and the keys it adds to
`results.json`).

This file was brought up to date with the code as built at the end of stage 0 (sections 3, 10, 13, 16 and 17); it is
frozen with the code (it is in the code manifest). Version 4 (post hoc, PRESPEC.md section 21) adds section 18 and
marks its changes in sections 16 and 17 "version 4".

---

## 1. Build-phase file ownership

All paths are under `experiments/shift_hosts/`. Edit only your own files, in the Windows staging folder; `sync.sh`
copies them into WSL.

| Agent | Files |
|---|---|
| scaffold | `__init__.py`, `INTERFACES.md`, `unifloral_import.py`, `gate.py`, `guard.py`, `handoff.py`, `bca_core.py`, `tests/__init__.py`, `tests/test_import.py`, `tests/test_guard_compare.py`, `tests/test_handoff.py`, `tests/test_bca_core.py`, `tests/madeup.py` (shared made-up-array builder, registered with the guard) |
| td3bc | `unifloral_td3_bc_bca.py`, `tests/test_td3_bc.py` |
| rebrac | `unifloral_rebrac_bca.py`, `tests/test_rebrac.py` |
| cql | `unifloral_cql_bca.py`, `tests/test_cql.py` |
| iql | `unifloral_iql_bca.py`, `tests/test_iql.py` |
| data | `prepare.py`, `data_copy.py`, `tests/test_prepare.py` |
| iw | `iw.py`, `tests/test_iw.py`, `tests/test_iw_compare.py` |
| scoring | `policies.py`, `actors_f32.py`, `evaluate.py`, `scoring_cache.py`, `tests/test_scoring.py` |
| runner (later) | `runner.py`, `band_readout.py`, `stages.py`, `budget.py`, `tests/test_runner.py`, `tests/test_stages.py` |
| claims (later) | `export.py`, `claims.py`, `tests/test_claims.py` |
| suite (later) | `tests/test_all_stage0.py` (card timing, the S0 summary and the code manifest draft) |

Not in this table: `PRESPEC.md`, `prespec.json` and `approve_hosts.py` (written after the build, frozen before any
run; `approve_hosts.py` writes the approval file of section 8, and only the user runs it), and `supplement.py`, which
version 4 added (section 18). `run_tests.sh` sits in the staging folder beside the package and is not part of it.

## 2. Environments and how to run

| | Training: `/opt/bca-venv` | Data and scoring: `/opt/bca-shift-venv` |
|---|---|---|
| Python, numpy | 3.10.12, numpy 1.26.4 | 3.10.12, numpy 2.2.6 |
| JAX | 0.6.2 with the CUDA 12 plugin (card) | 0.6.2, processor only |
| Imports | the Unifloral files (distrax, d4rl, gym, tyro, wandb), `calibration/`, `algorithms/`, this package | `experiments/shift`, `experiments/shift_iw`, `calibration/`, this package's numpy-only modules |
| Cannot import | `experiments/shift*` (they refuse numpy 1.26.4, `experiments/shift/__init__.py:101-103`) | the Unifloral files (no distrax, d4rl, gym, wandb) |

- No process holds both a Unifloral network and the simulator; whatever needs both goes by file (`handoff.py`).
- Modules that must import in both environments: `__init__.py`, `handoff.py`, `gate.py`, `guard.py`, `tests/__init__.py`,
  `tests/madeup.py`, `supplement.py` (version 4: the standard library, `gate` and `handoff` at module level; `claims`,
  `export` and `budget` inside the functions that use them), and (it needs JAX, flax, optax, scipy only) `bca_core.py`.
  Keep the package `__init__.py` empty of imports.
- Work from `/root/bca-hosts` (WSL, as root). `/root/bca-shift` is read-only. Never edit files inside
  `/root/bca-hosts`: edit the staging folder, then
  `MSYS_NO_PATHCONV=1 wsl -u root -- bash /mnt/c/Users/David/AppData/Local/Temp/claude/C--Users-David-dev-BCA/3bdab4f0-86ee-439c-a05a-34828d6b0a71/scratchpad/wf8/pkg/sync.sh`.
- Every Python call: `PYTHONDONTWRITEBYTECODE=1`, never `python -I`, `OMP_NUM_THREADS=1`, pinned with `taskset -c` to
  the cores your task names (at most 4). Processor tests: `JAX_PLATFORMS=cpu`. Card: `XLA_PYTHON_CLIENT_PREALLOCATE=false`
  (prepare() enforces it), short runs only; the card is shared.
- Training-environment processes need Unifloral's Dockerfile variables at start-up (`baselines/unifloral/Dockerfile:42-45,
  57-58`): `D4RL_SUPPRESS_IMPORT_ERROR=1 MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa DISPLAY= MUJOCO_PY_MUJOCO_PATH=/root/.mujoco/mujoco210
  LD_LIBRARY_PATH=/root/.mujoco/mujoco210/bin`. `unifloral_import.launch_env()` returns them for a subprocess.
- Example (training environment, processor):
  ```bash
  cd /root/bca-hosts && env PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 JAX_PLATFORMS=cpu SHIFT_HOSTS_DEVICE=cpu \
    D4RL_SUPPRESS_IMPORT_ERROR=1 MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa DISPLAY= \
    MUJOCO_PY_MUJOCO_PATH=/root/.mujoco/mujoco210 LD_LIBRARY_PATH=/root/.mujoco/mujoco210/bin \
    taskset -c 0-3 /opt/bca-venv/bin/python -m experiments.shift_hosts.tests.test_td3_bc
  ```
  For the card: `SHIFT_HOSTS_DEVICE=gpu XLA_PYTHON_CLIENT_PREALLOCATE=false` and no `JAX_PLATFORMS`.
- **Import order in every training-environment process:** first
  `from experiments.shift_hosts import unifloral_import as UI; UI.prepare()`, before anything can start JAX's backend.
  Creating any JAX array starts it, so **no package module may create a JAX array at import time** (no module-level
  `jax.random.PRNGKey(...)`, `jnp.zeros(...)`, etc.). prepare() refuses (BackendStartedError) if the backend already
  started. Modules that need Unifloral's pieces may call `UI.load(learner)` at import time, so import them after
  prepare().
- No pytest: tests are plain `unittest` modules run as `python -m experiments.shift_hosts.tests.<name>` (section 14).
- Outputs of tests: `runs/shift_hosts/stage0/<your area>/` (git ignores `runs/`). Nothing is written anywhere else in
  `/root/bca-hosts`.

## 3. Conventions

- Learner names: `"td3_bc"`, `"cql"`, `"iql"`, `"rebrac"` (`unifloral_import.LEARNERS`, `bca_core.LEARNERS`).
- Setup names: `"alone"` (imported Unifloral step), `"standard"` (restated step, one threshold), `"iw"` (restated
  step, a threshold per row), `"room"` (alone on the full log); stage-1 extras `"iw_ones"` (IW with every weight 1, S2),
  `"iw_repeat"` (S8), and `"sham_alone"` (choice 19: BCA present, never refreshed, so the dose stays exactly 1; seed 0
  of each quality, for TD3+BC and IQL, `stages.CHOICE19`). `runner.SETUPS` = alone, room, standard, iw, iw_ones,
  iw_repeat, sham_alone.
- Stage 3's setup names (plan 5.2; the claims area's names, `claims.STAGE3_SETUPS`): `"standard_flattened"`,
  `"iw_flattened"`, `"iw_shuffled"` and `"iw_estimated"` (the optional estimated-weight IW arm). Claim 6 reads them
  from `<root>/stage3/...` (section 16). runner.py does not build them: stage 3 runs only where Claim 1 or 2 passes,
  and its setups need their own code and approval then.
- Qualities: `"medium-expert"`, `"medium"`. Data seeds `rep` 0-4.
- Banks, in this index order (used by CQL's noise stream, section 4): 0 `"none"`, 1 `"mild"`, 2 `"strong"` (the
  weighting test's three start banks), 3 `"recorded"`, 4 `"strong_current"`, 5 `"big_waves"`, 6 `"big_kicks"` (the main
  study's stalling-world banks). `bca_core.BANKS`.
- **Bit for bit** means: for every leaf (in `jax.tree_util` order) the same dtype, the same shape and the same
  `np.asarray(x).tobytes()`. **To float32 rounding** means a relative difference of at most 1e-5 (plan 4.5).
- **Tree hash**, used everywhere a state, parameter set, key or data set is hashed: `guard.tree_sha256(tree)` (SHA-256
  over each leaf's dtype string, shape string and bytes, restated from `experiments/shift/learners.py:184-190`) and
  `guard.state_sha256(tree)` (the same after `jnp.asarray` on every leaf, `learners_iw.py:380-383`; use it for JAX
  states and keys).
- Keys are JAX's legacy `uint32[2]` keys (`jax.random.PRNGKey`), as Unifloral uses.
- JSON: `handoff.write_json(path, obj)` (sorted keys, a `.sha256` sidecar, `inf`/`-inf`/`nan` written as strings by
  `handoff.plain`); read back with `handoff.read_json`, which checks the sidecar.
- Code style: docstrings cite the lines they restate ("restated from `algorithms/td3_bc_bca.py:159-231`"). Inside a
  restated Unifloral step every line is Unifloral's, verbatim, except BCA's additions, each marked `# BCA:`. Plain
  language, American spelling, no unexplained acronyms.

## 4. Seeds and keys (plan 3.4)

| What | Rule | Where |
|---|---|---|
| Unifloral's `Args.seed` for data seed `rep` | `default_rng([20261016, rep, 31]).integers(2**31 - 1)` | `bca_core.seeds(rep).unifloral` |
| Unifloral's own stream | `PRNGKey(Args.seed)`, then the training block's network split (3 ways TD3+BC and ReBRAC, 4 ways CQL and IQL); one split per 2,500-step chunk for the (discarded) evaluation key | `unifloral_import.init_like_training_block`; the runner's chunk loop |
| BCA fit stream `s91` | `default_rng([20261016, rep, 91]).integers(2**31 - 1)` | `bca_core.seeds(rep).s91` |
| Fit key at outer step t (t = 1 ... 100,000) | `fold_in(PRNGKey(s91), t)` | `bca_core.fit_key(fit_key_base(s91), t)` |
| Scale network's start | `fold_in(fold_in(PRNGKey(s91), 0), C_init)` | `bca_core.scale_init_key(base, learner)` |
| Fit target noise / bootstrap masses | `fold_in(fit key, C_noise)` / `fold_in(fit key, C_boot)` (IQL folds `C_boot` inside its fitter) | `bca_core.fit_noise_key`; inside `fit_scale_core` / `iql_fit_core` |
| BCA refresh stream `s39` | `default_rng([20261016, rep, 39]).integers(2**31 - 1)` (`learners.TAG_REFRESH`) | `bca_core.seeds(rep).s39` |
| Refresh key after outer step t | `fold_in(PRNGKey(s39), t)`, t in 10,000, 15,000, ..., 95,000 | `bca_core.refresh_key(s39, t)`, `bca_core.refresh_steps()` |
| Refresh target noise / posterior | `fold_in(refresh key, C_refresh)` / `fold_in(refresh key, C_post)` | `bca_core.refresh_noise_key`, `bca_core.posterior_key` |
| Per-row target keys | `jax.random.split(noise key, n)`, through `jax.vmap` of the learner's own per-row target function, as Unifloral does (`td3_bc.py:210-212`) | `bca_core.per_row_keys` |
| CQL's scoring noise | `default_rng([20261016, 1004, bank]).standard_normal((1000, 100, 4)).astype(float32)`, bank = index of section 3 | scoring agent (`bca_core.CQL_NOISE_STREAM = 1004`) |

The ports' constants keep their roles (`bca_core.FOLDS`):

| Learner | C_init | C_noise (fit) | C_boot | C_refresh (refresh noise) | C_post | Source |
|---|---|---|---|---|---|---|
| TD3+BC | 1128352841 | 1179210836 | 1128352850 | 1213156420 | 1347375956 | `td3_bc_bca.py:135, 167, 176, 249, 268` |
| CQL | 1128352841 | 1179210836 | 1128352850 | 1213156420 | 1347375956 | `cql_bca.py:135, 182, 192, 244, 259` |
| ReBRAC | 1380141390 | 1380338004 | 1380077391 | 1380470604 | 1380994899 | `rebrac_bca.py:24-28` |
| IQL | 1111703881 | none | 1111704112 | none | 1347375956 | `iql_imports.py:8`, `iql_loss.py:14`, `iql_reference.py:14` |

BCA's randomness never touches Unifloral's key, so the three setups of a learner draw the same batches; the standard
and IW setups share s91 and s39 and differ only in their thresholds (and the refusal rule).

## 5. `unifloral_import.py` (training environment)

```python
UI.prepare(learners=UI.LEARNERS, device=None, *, start_backend=True, extra_xla_flags=()) -> dict   # once, first
UI.load(learner) -> module            # the imported, fingerprint-checked Unifloral module
UI.record() -> dict                   # JSON copy of the import record (for run records)
UI.args_for(learner, **overrides) -> Args            # dataclasses.replace(U.Args(), **overrides)
UI.make_transition(learner, rows, *, next_action="roll") -> U.Transition
UI.init_like_training_block(learner, args, rows, *, next_action="roll") -> Built
UI.make_unifloral_step(built, args) -> _train_step   # Unifloral's own (imported) make_train_step
UI.launch_env(base=None) -> dict      # the Dockerfile's variables for a subprocess
UI.check_pinned(learner) / UI.verify_file(path, sha) / UI.manifest_sha256()   # stop rule S1
```
- `prepare()` checks the Dockerfile's variables, checks each file's SHA-256 against `configs/unifloral.json` (S1), imports
  it by path as `shift_hosts_unifloral_<learner>` (never `__main__`, so the training block never runs), sets
  `XLA_FLAGS` to choice 8's flags after the imports (`--xla_gpu_deterministic_ops=true`, and on the processor also
  `--xla_cpu_multi_thread_eigen=false`) after checking JAX's backend has not started, sets
  `jax_default_matmul_precision="highest"`, and starts the backend on `device` (`"cpu"` or `"gpu"`, default from
  `SHIFT_HOSTS_DEVICE`, else `"cpu"`). Errors: `PinnedCodeError`, `BackendStartedError`, `LaunchEnvironmentError`.
- The record holds: revision, every file's SHA-256, module names, `XLA_FLAGS` before the imports / after each import /
  in effect, whether the backend had started before the repair (must be false), matmul precision, `jax_platforms`, the
  environment variables, Python, executable, package versions (jax, jaxlib, the CUDA plugin, numpy, scipy, flax, optax,
  distrax, chex, gym, d4rl, tyro, wandb, mujoco-py), the processor affinity, the backend, devices and, on the card,
  `card` = {name, driver_version, memory_total} from nvidia-smi.
- `rows` is a mapping with `obs, action, reward, next_obs, done` (numpy or JAX; float32). For ReBRAC, `next_action="roll"`
  gives Unifloral's `jnp.roll(actions, -1, axis=0)` (`rebrac.py:335`, training rows); pass an array for bank rows (the
  bank rule, plan 3.8).
- `Built` = `(learner, rng, agent_state, dataset, obs_mean, obs_std, nets, apply_fns, dummy_obs, dummy_action)`:
  `rng` is the key after the network split (where the chunk loop starts); `obs_mean = dataset.obs.mean(axis=0)` and
  `obs_std = jnp.nan_to_num(dataset.obs.std(axis=0), nan=1.0)` computed on the device as the training block does (for
  every learner; they also build the scale network, choice 3); `apply_fns` is in the order of the learner's
  `make_train_step` apply arguments: TD3+BC and ReBRAC `(actor, q)`, CQL `(actor, q, alpha)`, IQL `(actor, q, value)`;
  `nets` = `{"actor", "q", "alpha"|"value"}` flax modules.
- Settings for real runs (plan 3.2): `UI.args_for(learner, seed=bca_core.seeds(rep).unifloral, gamma=0.98,
  num_updates=100_000)`; `eval_interval` stays 2,500 (the chunk length); nothing else moves.

## 6. The prepared-data file (data agent writes it in `/opt/bca-shift-venv`; the runner reads it)

**Path:** `runs/shift_hosts/data/<quality>/seed<k>/prepared.npz`, written with `handoff.write`, so beside it:
`prepared.npz.sha256`, `prepared.json` (the manifest: SHA-256, dtype and shape of every array) and
`prepared.json.sha256`. Read with `handoff.read(path)`, which checks everything. No learning happens while it is made.

**Row sets.** Every set is in stored log order (episode, then t).

| Set | Rows | Source |
|---|---|---|
| `train` | the gapped log's kept training rows (episodes 0-499 minus the ball; 44,902-45,372 on medium-expert) | `train_law.hdf5`, removal "gap" |
| `full` | every training row (50,000), no removal: the room references only | `train_law.hdf5`, removal "none" |
| `bank` | the kept bank rows (5 per calibration episode 500-699 before the cut; 869-924 on medium-expert) | `data.calibration_bank` rows, cut by the ball |
| `test_strong`, `test_mild` | every step of the 300-episode strong and mild test logs (30,000 each), never cut | `test_strong.hdf5`, `test_mild.hdf5` |

**Arrays** (`<set>/<field>`; n = the set's rows):

| Field | dtype | shape | Sets | Meaning |
|---|---|---|---|---|
| `obs` | float32 | [n, 13] | all | the log's `observations` (raw; Unifloral's networks normalize inside where they do) |
| `action` | float32 | [n, 4] | all | recorded commands, in [-1, 1] |
| `reward` | float32 | [n] | all | the log's `rewards` |
| `next_obs` | float32 | [n, 13] | all | the log's `next_observations` (the recorded next state, also at t = 99) |
| `done` | float32 | [n] | all | 0.0 everywhere (choice 2: time-limit rows kept, never masked) |
| `next_action` | float32 | [n, 4] | all | ReBRAC's a': `train` and `full`: `np.roll(action, -1, axis=0)` over the set (equal to Unifloral's `jnp.roll`, test (e)); `bank`: the roll over the cut calibration episodes in stored order, then the bank rows; `test_*`: the roll over the test log |
| `next_action_true` | bool | [n] | all | a' is the same episode's command at t + 1 (False where substituted) |
| `episode`, `t` | int32 | [n] | all | `infos/episode`, `infos/t` |
| `row` | int64 | [n] | all | the row's index in its source log |
| `state`, `next_state` | float64 | [n, 13] | all | `infos/state`, `infos/next_state` (exact float64) |
| `reward64` | float64 | [n] | all | `infos/reward64` |
| `current` | float64 | [n] | all | c = `infos/state[:, 10]` |
| `weight_strong` | float64 | [n] | train, bank, test_strong, test_mild | exact w(c) = `shift_iw.world.ratio(c, "strong")` (equals `truth/weight_strong`) |
| `weight_strong_estimated` | float64 | [n] | train, bank, test_strong, test_mild | the logistic classifier's estimate (`shift_iw/classifier.py`, 300 episodes per law), reported only |
| `in_ball` | bool | [n] | all | `truth/in_ball`: the command lies in the seed's ball (False on every kept `train`/`bank` row) |

**Manifest meta** (`prepared.json` → `meta`): `quality`, `seed`, `rows` (per set), `recorded_rows` (from
`runs/shift_iw_v5/<quality>/host/gap/seed<k>/record.json`) and the S11 comparison, `source` (every copied file's path
and SHA-256 and the SHA-256 of its `.manifest.json`, `world.json`), the ball, the shares of substituted next commands
(train overall, train's highest-current tenth, bank), the bank's Kish effective size under `weight_strong`, the
classifier's record, the code SHA-256 of `prepare.py` and the weighting-test modules it used, and the environment
(Python, numpy). Normalization statistics are **not** stored: the training block computes them (section 5).

## 7. `handoff.py` (both environments; numpy only)

```python
H.write(path_npz, {name: array}, meta=None) -> manifest   # <stem>.npz, .npz.sha256, <stem>.json, .json.sha256
H.read(path_npz, names=None) -> ({name: array}, manifest) # verifies sidecars, manifest, members, dtypes, shapes, hashes
H.read_manifest(path_npz); H.write_json(path, obj); H.read_json(path); H.plain(obj)
H.write_sidecar(path); H.verify_sidecar(path); H.array_sha256(a); H.group(arrays, "train") -> {field: array}
H.flatten_dict(tree, prefix) / H.unflatten_dict(arrays, prefix)   # flax parameter trees <-> flat names
```
- One `.npz` is an uncompressed zip of `.npy` members (format 1.0, no pickles), sorted, dated 1980-01-01: the same
  arrays give the same bytes (numpy 1.26.4 and 2.2.6 wrote identical bytes in test (i)).
- Names: letters, digits, `_ . - /`; groups as `train/obs`. Dtypes: bool, integers, floats. Shapes kept exactly
  (0-d arrays stay 0-d).
- `HandoffError` on any failed check. Every file passed between the environments goes through this module (S7's
  states and commands, coverage states, actor parameters, the prepared data).

## 8. `gate.py` and `guard.py`: who may run what

**Real ThrusterChain data: `gate.require_run_approval(purpose) -> gate.Approval`.** Every run on real data calls it at
the start (the learner alone and the room references included) and every BCA path re-checks it through
`guard.require_bca`. It refuses (`gate.ApprovalError`) unless:
1. `experiments/shift_hosts/prespec.json` exists (today it does not, so everything is refused);
2. `experiments/shift_hosts/approval_<fp16>.json` exists for the current fingerprint, where **fingerprint** = SHA-256 of
   the canonical JSON `{"kind": "shift-hosts-fingerprint-v1", "prespec_sha256": ..., "code_manifest": {...}}`
   (`gate.fingerprint()`), and `fp16` is its first 16 hex characters. The **code manifest** (`gate.code_manifest()`)
   holds the SHA-256 of every regular file under `experiments/shift_hosts/` whatever its name or suffix
   (`gate.package_files()`), except `prespec.json`, the approval files at the package root (`approval_<16 hex>.json`
   and its `.sha256`, `gate.APPROVAL_FILE`; a nested or differently named `approval_*` file is hashed like any other)
   and bytecode, and of `gate.EXTERNAL_CODE`: Unifloral's
   four files, its Dockerfile and requirements, `configs/unifloral.json`, `calibration/*.py`, `algorithms/*.py`,
   `runtime/*.py`, `experiments/__init__.py`, `experiments/shift/*.py`, `experiments/shift_iw/*.py` and `launch.sh`,
   `experiments/truth/*.py`, `experiments/signal/*.py` (added by the stage-0 suite: `experiments/shift`'s simulator
   imports it), `outputs/shift/oracle/oracle.py`. Any changed, added or removed file changes the
   fingerprint (stop rule S1). If your module imports another repository file, say so in your report so it is added;
3. no compiled bytecode (`__pycache__` folders or `*.pyc` files) exists anywhere in the package or in the folders of
   EXTERNAL_CODE's files (`gate.bytecode_files()`): a timestamp-checked `.pyc` could run in place of the hashed source;
4. the approval file matches its sidecar `approval_<fp16>.json.sha256` (sha256sum format) and holds
   `{"kind": "shift-hosts-approval", "fingerprint": <64 hex>, "typed": <the 16 characters the user typed>,
   "prespec_sha256": ..., "code_manifest": {...same...}, ...}` (other keys, such as the time and the user's words, are
   free);
5. the approval file, its sidecar and `prespec.json` are committed in git HEAD byte for byte, and the latest commit that
   touched the approval file has the author email `35092758+dbayha7@users.noreply.github.com`.
Nothing in the package writes the approval file; `approve_hosts.py` (the user's, at the terminal) does. `gate.verify(a)`
re-checks an Approval from disk. Git is only read.

**Stop rule S0: `gate.s0_status()` / `gate.require_s0(purpose)`.** S0 passes only when
`runs/shift_hosts/stage0/SUMMARY.json` (the stage-0 suite's summary) verifies against its sidecar, says the suite
passed (`status` 0, `s0.passed` true, `made_up_arrays_only` true) and was made on exactly the code on disk
(`manifest.code_manifest_sha256` equals `gate.manifest_sha256(gate.code_manifest())`, runner.canonical_sha256's rule).
The stage driver requires it before a real start; `approve_hosts.py` shows it.

**Other entry points.** `runner.execute` checks the permission against the data again (made-up file: `guard.MADE_UP`
and the registered arrays; real data: a verified `gate.Approval`), so Unifloral's imported step (alone, room) cannot
train on real arrays through the library. `runner s7-commands` needs the approval unless its actor file and its states
file are both marked made-up. Every actor file a run writes says `trained: true` (`kind: "run-snapshot"`), and
`evaluate.require_permission` never rolls a trained actor out under `guard.MADE_UP` (only `open_loop_states` with no
bank, which steps no simulator, may read its commands).

**Made-up arrays: `guard`** (restated from `experiments/shift_iw/learners_iw.py:54-57, 273-322`; test
`test_guard_compare` shows the same decisions and hashes as the original):
```python
guard.MADE_UP                         # the token (the weighting test's SYNTHETIC)
guard.MADE_UP_WORLD                   # "synthetic arrays (not ThrusterChain data)", the same label as the original
guard.register_made_up(container)     # world == MADE_UP_WORLD, registers tree_sha256((container.train, container.bank))
guard.register_arrays(tree, world)    # the declared extension: registers a bare pytree (Unifloral's Transition)
guard.require_made_up(*items)         # refuses unless every item's content hash is registered (containers: label too)
guard.require_bca(permission, *data)  # THE check at every BCA entry point (outside jit):
                                      #   MADE_UP -> require_made_up(*data); gate.Approval -> gate.verify; else refuse
guard.tree_sha256(tree); guard.state_sha256(tree); guard.data_sha256(data)
```
- **Every BCA entry point** calls `guard.require_bca(permission, <the arrays it runs on>)` before running BCA's code:
  a restated step's factory with BCA present (on `dataset`), a refresh (on `bank`), a band readout (on its rows), the
  IW tables' caller. bca_core's and iw.py's arithmetic run inside jit and do not check.
- Only `tests/madeup.py` and test fixtures register arrays. Registration is per process. The guard hashes leaves only,
  so a Unifloral `Transition` built from registered made-up arrays (same order, dtypes, shapes, bytes) passes; a subset
  of rows, another dtype or any changed value does not.

## 9. `bca_core.py` (training environment; also imports in the scoring environment)

```python
# seeds and keys (section 4)
BC.seeds(rep) -> Seeds(rep, unifloral, s91, s39); BC.stream_seed(rep, tag)
BC.fit_key_base(s91); BC.fit_key(base, t); BC.scale_init_key(base, learner); BC.fit_noise_key(k, learner)
BC.refresh_key(s39, t); BC.refresh_noise_key(rk, learner); BC.posterior_key(rk, learner); BC.per_row_keys(key, n)
BC.refresh_steps(num_updates=100_000, warmup=10_000, every=5_000) -> [10000, ..., 95000]   # 18
BC.FOLDS[learner] -> Folds(scale_init, fit_noise, bootstrap, refresh_noise, posterior); BC.BANKS; BC.CQL_NOISE_STREAM
# settings
BC.BCASettings(blend=0.5, cal_lr=1e-3, cal_beta=20.0, width_penalty=0.005, scale_ema=0.99, miscoverage=0.1,
               credibility=0.95, draws=1000, scale_hidden=64, balance_weight=0.5, decision_gain=1.0,
               min_bank_rows=36); BC.SETTINGS; settings.wbcp() -> calibration.reference.WBCPConfig
BC.BCASetup(learner, fit_key_base, permission, settings=SETTINGS)       # static config a restated step closes over
BC.TestOptions(unifloral_keys=False, dose_override=None, width_override=None)
# scale networks (choice 3: given Built.obs_mean / obs_std)
BC.make_scale_network(learner, obs_mean, obs_std) -> flax module
BC.make_scale_state(learner, obs_mean, obs_std, key) -> TrainState    # TD3+BC/CQL/ReBRAC: Calibrator(state_dep=True),
                                                                       # optax.adam(1e-3) eps 1e-8; IQL:
                                                                       # StateDependentCalibrator(64), adam eps 1e-5
# BCA's carry
BC.BCAState(scale, unit, reference, thresholds, most_caution, step)
BC.initial_state(scale, n_train, *, most_caution=False) -> BCAState    # unit 1.0 (float32), reference not ready,
                                                                       # thresholds +inf (float32 [n_train]), step 0
# fit cores (inside the jitted step)
BC.fit_scale_core(scale, unit, obs, action, y, q, k, *, bootstrap_fold, settings) -> FitResult   # TD3+BC, CQL, ReBRAC
BC.iql_fit_core(scale, unit, obs, action, y, q_heads, k, *, settings) -> FitResult                # IQL
#   FitResult(scale, unit, loss, eta, masses, valid): scale and unit are already the accepted ones (unchanged when
#   not valid), as in the ports. Restated from td3_bc_bca.py:170-231 and iql_scale.py:36-116; test (c1) checks them
#   bit for bit against all four ports, op by op and compiled.
# width and dose (one threshold or one per row)
BC.width(R, unit, eta) -> R * (max(eta, 1e-6) * unit)                  # frozen_level_dose's operand order
BC.dose(width, unit, blend, inputs_valid=True) -> CQLLevelDose         # calibration.dose.level_critic_dose: 1 + blend U/(U+u)
BC.frozen_dose(reference, eta, blend)                                  # calibration.dose.frozen_level_dose (the ports')
# refresh (eager, between chunks)
BC.calibrate_bank(scale_params, unit, eta, residuals, posterior_key) -> Calibrated(reference, valid, diagnostics,
                                                                                    scores, stream)
BC.apply_refresh(bca_state, y, q, eta, posterior_key, *, inputs_valid=True, check_new_reference=False)
    -> Refreshed(state, record, scores, stream)       # check_new_reference=True: ReBRAC (its port's rebrac_bca.py:370)
BC.bank_scores(eta, unit, residuals); BC.posterior_stream(key)
```
- `calibrate_bank` = `calibration.reference.freeze_reference` (wbcp.calibrate, miscoverage 0.1, credibility 0.95,
  1,000 draws, uniform weights), refusing a bank below 36 rows (`reference.certifiable`). R = +inf means "cannot
  certify" (refused); it is counted, not a stop. NaN anywhere is a stop (S5).
- `scores` (float64) and `stream` (uint32 words) are exactly what freeze_reference used: an unweighted
  `wbcp.calibrate(scores, np.random.default_rng(stream))` reproduces the stored threshold, lambda_hat, lambda_hpd and
  n_eff bit for bit (checked in test (c1); stop rule S3 checks it at every IW refresh).
- `apply_refresh` freezes the **live** scale network and unit into the new reference and fills every row's threshold
  with the standard R (`jnp.full_like(thresholds, R)`); an invalid refresh keeps the previous state. With
  `check_new_reference=True` (ReBRAC's refresh, as its port does) the NEW frozen reference must also pass
  `reference_valid`; the TD3+BC, CQL and IQL ports check only the previous one, so their refreshes keep the default.
  Its `record`: `valid, stored_inputs_valid, new_reference_valid, certified, threshold, lambda_hat, lambda_hpd, n_eff,
  sigma_post, clamp_binds, unit, bank_rows, scores_sha256, stream` (JSON-ready).
- Imported unchanged and re-exported for the other modules: `postcap_level_actor_weights`, `LevelActorWeights`,
  `level_critic_dose`, `frozen_level_dose`, `CQLLevelDose`, `tree_finite`, `FrozenReference`, `initial_reference`,
  `reference_valid`, `positive_scale`, `certifiable`, `WBCPConfig`.

## 10. The restated steps: `unifloral_<learner>_bca.py` (td3bc, rebrac, cql, iql agents)

Same shape for all four learners.

```python
LEARNER = "<learner>"
def make_train_step(args, <Unifloral's apply-function arguments, unchanged>, dataset, *, bca=None, test=None)
#   td3_bc, rebrac: (args, actor_apply_fn, q_apply_fn, dataset, *, bca=None, test=None)
#   cql:            (args, actor_apply_fn, q_apply_fn, alpha_apply_fn, dataset, *, bca=None, test=None,
#                    dose_active=True)                                   # two programs per run, below
#   iql:            (args, actor_apply_fn, q_apply_fn, value_apply_fn, dataset, *, bca=None, test=None)
def band_parts(args, apply_fns, agent_state, rows, noise_key) -> (y, q)       # float32 [n] each (IQL: noise_key=None)
def refresh(args, apply_fns, agent_state, bca_state, bank, refresh_key, *, permission) -> bca_core.Refreshed
FLAGS                                                   # the step's per-step validity flags (td3_bc, rebrac)
require_valid(chunk outputs)                            # td3_bc, rebrac: the port's S5 check, FloatingPointError
recompute_bank_scores(args, apply_fns, agent_state, bca_state, bank, refresh_key, *, permission)   # td3_bc only
critic_value_parts(agent_state) -> (dual_q, dual_q_target, value)                               # iql only
batch_valid(batch); check_rows(rows, name)                                                      # rebrac only
```

**BCA absent** (`bca=None`, `test=None`): Unifloral's `make_train_step` copied line for line, nothing added. Carry
`(rng, agent_state)`, output Unifloral's loss dict. Test (a1): bit for bit against `UI.load(learner).make_train_step`
from `UI.init_like_training_block`'s state, 60 outer steps, processor and card.

**BCA present** (`bca=bca_core.BCASetup(...)`): the factory first checks `bca.learner == LEARNER` and calls
`guard.require_bca(bca.permission, dataset)`. The step is `_train_step(runner_state, _)` with carry
`(rng, agent_state, bca_state)` (`bca_core.BCAState`); the runner scans it exactly as Unifloral's step. Inside, in
order:
1. Unifloral's batch draw, verbatim (`rng, rng_batch = split(rng)`, `batch_indices = randint(...)`, `batch = ...`).
2. `t = bca_state.step + 1`; `k = BC.fit_key(bca.fit_key_base, t)`.
3. The band fit at the learner's place (plan 4.3; choices 13-15), y and q by the learner's own expressions:

   | Learner | Where | y | q | Core |
   |---|---|---|---|---|
   | TD3+BC | right after the batch, before the critic scan | the learner's own per-row `_compute_target` (`td3_bc.py:191-208`) with the step-start state, keys `per_row_keys(fit_noise_key(k), B)` | min of the 2 online critics at (s, a), step start | `fit_scale_core(..., bootstrap_fold=FOLDS["td3_bc"].bootstrap)` |
   | ReBRAC | as TD3+BC | own `_compute_target` (`rebrac.py:216-235`), a' from `batch.next_action` | as TD3+BC | `fit_scale_core(..., FOLDS["rebrac"].bootstrap)` |
   | CQL | just before `_q_loss_fn` (after temperature, actor and target updates and Unifloral's own target and action draws) | r + 0.98 (1 - d) [min of 10 target critics - alpha log pi(a'|s')], a' from the updated actor via own `_sample_next_v` with `per_row_keys(fit_noise_key(k), B)` | min of the 10 online critics at (s, a) (not yet updated) | `fit_scale_core(..., FOLDS["cql"].bootstrap)` |
   | IQL | after the target averaging and `q_targets` (`iql.py:206-221`), before the Q and V update | `q_targets` (`iql.py:221`) | heads = the 2-head target values at (s, a) (`iql.py:218` before `.min(-1)`) | `iql_fit_core(..., y, heads, k)` |

4. The dose (or IQL's weights) from the **frozen** band at the recorded (s, a), no gradient:
   `eta = bca_state.scale.apply_fn(bca_state.reference.cal_params, batch.obs, batch.action)`,
   `R_b = bca_state.thresholds[batch_indices]` (Unifloral's own `batch_indices`), then
   `iw.per_row_dose(bca_state.reference, eta, R_b, bca.settings.blend, bca_state.most_caution)` (TD3+BC, ReBRAC, CQL) or
   `iw.per_row_iql_weights(...)` (IQL), section 11. Before the first refresh (`reference.ready` False) every dose is
   exactly 1 and IQL's weights are exactly Unifloral's own `jnp.exp(adv * args.beta).clip(max=args.exp_adv_clip)`.
5. The dose `m = jax.lax.stop_gradient(dose)` enters exactly one term, keeping Unifloral's order of operations (plan 4.3):
   TD3+BC `actor_loss = -lambda_ * q.mean() + (m * bc_loss).mean()` (`td3_bc.py:245`); ReBRAC
   `actor_loss = -lambda_ * q.mean() + args.actor_bc_coef * (m * bc_loss).mean()` (`rebrac.py:274`); CQL
   `q_diff = (jnp.expand_dims(m, 1) * (jnp.expand_dims(q_ood, 1) - q_pred)).mean()` (`cql.py:304`), with the port's
   guard (a negative or non-finite dose makes the step invalid); IQL `exp_adv` (`iql.py:245`) replaced by BCA's
   weights, with the skip rule (no supported row: no actor update, `calibration/iql_state.py:45-79`). Rows BCA does not
   change keep Unifloral's own expression bit for bit.
6. Everything else is Unifloral's, verbatim.
7. Validity (the ports' per-step flag, plan 4.1): `valid = fit.valid & dose.inputs_valid & tree_finite((new agent state,
   Unifloral's loss dict)) & <learner checks>` (CQL's dose guard; ReBRAC's doses in [1, 2], `rebrac_bca.py:416`; IQL's
   finite weights). The new carry is `(rng, agent_state, bca_state._replace(scale=fit.scale, unit=fit.unit, step=t))`,
   and `jax.lax.cond(valid, new, old)` keeps the old carry whole, random key included, when not valid
   (`td3_bc_bca.py:340-342`), for TD3+BC, CQL and ReBRAC. **IQL's exception** (its Decision 1, below): the learner part
   `(rng, agent_state)` is kept unless `learner_valid` (`_keep_unless(learner_valid, ...)`), and the BCA part unless
   `valid` (`_keep_unless(valid, ...)`); `inputs_valid = fit.valid & learner_valid` is still the reported flag, so the
   runner stops the run on it. The runner stops on any false flag (S5).
   Every step also checks, at trace time, that `bca_state.thresholds` holds one float32 threshold per training row
   (TD3+BC, CQL and ReBRAC in the step, IQL in `_check_carry`): JAX clamps an out-of-range gather index silently.
8. Output: Unifloral's loss dict plus BCA's metrics, all scalars per step:
   `inputs_valid` (bool, the step's flag), `scale_inputs_valid` (`fit.valid`), `dose_inputs_valid`, `scale_loss`,
   `unit` (after the step), `posterior_ready`, `dose_mean`, `dose_min`, `dose_max` (IQL: `weight_mean`, `weight_min`,
   `weight_max`, `actor_updated`, `supported_fraction`), `refused_fraction` (rows with `ready` and R = +inf),
   `support_fraction`; CQL adds `dose_guard_ok`, ReBRAC adds `batch_valid` and `dose_in_range`.

**The runner's S5 flags** (`runner.FLAGS`; a false one anywhere in a chunk stops the run): TD3+BC and IQL
`inputs_valid, scale_inputs_valid, dose_inputs_valid`; CQL those and `dose_guard_ok`; ReBRAC those and `batch_valid,
dose_in_range` (`unifloral_rebrac_bca.FLAGS`; TD3+BC's own `FLAGS` are its three). The runner also stops on any
non-finite floating output, a dose outside [1, 1.5] (ReBRAC [1, 2]; `runner.DOSE_BOUNDS`), a negative IQL weight, or a
non-finite learner parameter or optimizer state after a chunk.

**As built, per learner** (each module's docstring has the details and the lines it restates):
- **TD3+BC** (`unifloral_td3_bc_bca.py`). Only two of Unifloral's lines change (`CHANGED_LINES`: the signature,
  td3_bc.py:174, and the dosed actor loss, td3_bc.py:245); the reported `bc_loss` stays Unifloral's undosed mean.
  `recompute_bank_scores(...)` -> (scores float64 [n], stream) restates the weighting test's `_scores_from_state`
  (learners_iw.py:540-552): the bank scores from the stored states with the refresh key's target noise and the frozen
  reference's scale parameters, through the very compiled program `refresh` runs (`_refresh_program`), so it gives
  that refresh's scores bit for bit (stop rule S3; a separately compiled recomputation agrees only to float32
  rounding). `require_valid(metrics)` restates the port's check (td3_bc_bca.py:348-351). Test mode adds `eta`, `R_rows`
  and `refused` to the keys above. On the card the logged `actor_loss` and `bc_loss` may differ from the alone setup by
  a rounding unit while the parameters, optimizer states and key stay bit for bit (test a2's record; choice 19 follows
  the plan's letter for TD3+BC).
- **CQL** (`unifloral_cql_bca.py`). `make_train_step(..., dose_active=False)` builds the step for the outer steps
  before the first refresh (the band is not ready and every dose is exactly 1): the conservative gap is Unifloral's own
  line cql.py:304, and a step that finds a ready band is not valid. On the card the multiply by an exact 1 changes how
  the critic's gradient is compiled (the single program drifts 1.55e-4 relative), so **CQL's BCA runs use two compiled
  programs**: `dose_active=False` for the chunks up to outer step 10,000 (`Schedule.switch_step`, the first refresh
  step), then the default `dose_active=True` from there on, shared by standard and IW (`runner.build_setup`'s
  `steps = {"pre", "post"}`; the run record's `settings.programs` and each chunk's `losses[*].program` say which ran).
  The sham alone switches at the same step without refreshing. band_core reads a copy of the state through
  `jax.lax.optimization_barrier`, so the card's compiler cannot merge BCA's reads into Unifloral's critic update. Test
  mode adds `eta`, `refused` and Unifloral's draws for the conservative term (`cql_random_actions`, `pi_actions`,
  `pi_next_actions`).
- **IQL** (`unifloral_iql_bca.py`). **Decision 1** (what a false flag keeps): BCA's part of the carry (scale network,
  unit, step count) is kept whenever `valid` is false; the learner's part (Unifloral's key and agent state) is kept
  whenever `learner_valid` (the weights' inputs valid, finite weights, every leaf of the new agent state and of
  Unifloral's loss dict finite) is false. So a step is discarded whole on every failure except one: when only the band
  fit fails while every learner value is finite, the fit is rejected and the flag is false, but the learner's own
  update of that step stands, the actor's included. The IQL port also never rolls back its Q and V update, but it
  skips that step's actor update (`algorithms/iql_bca.py:265-280`: its dependency_valid includes the fit's
  numerical_valid); this step keeps the actor update too, and its `actor_updated` output (`actor_ok & valid`) reads
  False there although the actor moved. The difference exists only on a step whose flag is false, where the runner
  stops the run (S5). Selecting the learner's
  state on `valid` instead changes how XLA compiles Unifloral's own arithmetic on the processor and breaks test (a2).
  Test options are refused unless the permission is `guard.MADE_UP`, as in the other three steps. The skip rule: no
  supported row means no actor update (the
  actor's parameters, optimizer state and step count are kept, so its cosine schedule does not advance).
  `critic_value_parts(agent_state)` = `(dual_q, dual_q_target, value)`: what the three IQL runs of a seed share
  (Unifloral's Q and V updates never read the actor): bit for bit on the processor; on the card bit for bit between
  standard and IW, while the alone run drifts at rounding level (test a2's card exception). The run record's
  `hashes.final.parts.{dual_q, dual_q_target, value}` hold their hashes. Test mode adds `bca_heads`, `adv`, `eta`,
  `width`, `weight`, `unifloral_weight`, `refused`, `support_mask`, `has_support`; `width_override` replaces the
  band's width (test a3); `dose_override` is refused.
- **ReBRAC** (`unifloral_rebrac_bca.py`). `batch_valid(batch)` restates the port's `transition_valid`
  (rebrac_bca.py:150-157): every field finite, every command and next command in [-1, 1], done 0 or 1; `check_rows`
  restates its shape check (123-147) once, outside jit. Both enter the step's flag (`batch_valid` is also an output),
  with the doses in [1, 2] (`dose_in_range`, rebrac_bca.py:415-417). `FLAGS` = the five above; `require_valid(metrics)`
  restates the port's check (rebrac_bca.py:456-461). Test mode adds `band_dose`, `width`, `threshold` (each batch row's
  R through batch_indices), `eta`, `refused`; `width_override` is refused. Unifloral's lines are verbatim in text and
  order; seven of them (`REINDENTED_LINES`: the critic scan, rebrac.py:255-260, and the actor loss, rebrac.py:274) sit
  4 spaces deeper inside BCA's `else:` branches, which test_rebrac's source test checks with indentation. Its refresh
  also checks the NEW frozen reference (`check_new_reference=True`, rebrac_bca.py:370).

**Per-row thresholds.** `bca_state.thresholds` is a float32 array over **all training rows**, read through
`batch_indices`. Standard BCA: every entry the standard threshold (`apply_refresh` fills it). IW: the runner overwrites
it after each refresh with `iw.row_thresholds(table, train_weights)`. `bca_state.most_caution` is False for standard and
True for IW: the two setups compile the same program and differ only in carried values.

**Test mode** (`test=bca_core.TestOptions(...)`, only with BCA present) adds to the output dict:
`batch_indices` (int32 [B]), `bca_y`, `bca_q` (the values fed to the fit), `unifloral_y`, `unifloral_q` (Unifloral's
own: TD3+BC and ReBRAC the first critic update's `targets` and the minimum over critics of its `q_pred`; CQL `target`
and the minimum over the ten of `_q_loss_fn`'s `q_pred`; IQL `q_targets` and `v_target` after `.min(-1)`), `dose`
(IQL `weight`) and `width` per batch row as used, `dosed_loss` (the loss holding the dose: TD3+BC/ReBRAC actor loss,
CQL critic loss, IQL actor loss) and `dosed_grad` (its gradient). Options: `unifloral_keys=True` makes the fit's
per-row target keys Unifloral's own at that point, so y equals `unifloral_y` bit for bit on the processor (test (a4));
`dose_override` / `width_override` (float32 over training rows, read through `batch_indices`) replace the band's dose /
width (test (a3)). Normal runs never pass `test`; tests (a1)-(a2) must run without it.

**`band_parts(args, apply_fns, agent_state, rows, noise_key)`**: the learner's own y and q at `rows` (a Unifloral
`Transition`) from a stored state, per-row keys `per_row_keys(noise_key, n)` (IQL ignores the key): TD3+BC/ReBRAC as in
the fit with the stored state; CQL with the stored actor, temperature and target critics, q = min of the 10 stored
critics; IQL y = r + 0.98 (1 - d) V(s'), q = min over the 2 heads of `Polyak(stored target, stored critic,
args.polyak_step_size)` (choice 15). Used by `refresh` and by `band_readout.py`.

**`refresh(...)`** (eager, between chunks): `guard.require_bca(permission, bank)`; `y, q = band_parts(..., bank,
BC.refresh_noise_key(refresh_key, LEARNER))` (IQL: no noise); `eta = bca_state.scale.apply_fn(bca_state.scale.params,
bank.obs, bank.action)` (the live scale network, frozen now); return `BC.apply_refresh(bca_state, y, q, eta,
BC.posterior_key(refresh_key, LEARNER), inputs_valid=tree_finite((agent_state, bank)))` (ReBRAC: also `batch_valid` of
the bank in `inputs_valid`, and `check_new_reference=True`). For ReBRAC, `bank.next_action` is the prepared bank rule.

**Setups from these pieces** (`runner.build_setup`): alone and room = `UI.make_unifloral_step` (carry
`(rng, agent_state)`); standard and IW = `make_train_step(..., bca=BCASetup(...))` with
`BC.initial_state(BC.make_scale_state(L, built.obs_mean, built.obs_std, BC.scale_init_key(base, L)), n_train,
most_caution=iw.most_caution(setup))` (False for standard, True for IW); CQL's BCA setups build both programs
(`dose_active=False` and the default) and switch at the first refresh step; sham alone = the BCA step never refreshed;
iw_ones = IW with every bank weight and test mass 1 (grid [1.0]); iw_repeat = IW again (S8).

## 11. `iw.py` (iw agent; training environment, its table part is numpy)

Restates `shift_iw/coverage.py:153-205` and `shift_iw/learners_iw.py:540-561, 589-597, 655-683` (test (c3) compares
with the originals in `/opt/bca-shift-venv`):
```python
IW.GRID_RATIO = 1.01
IW.mass_grid(masses, ratio=1.01) -> float64 grid t = 1.01**k covering every mass       # coverage.py:153-170
IW.threshold_table(scores, stream, weights, grid) -> Table   # one wbcp.calibrate per grid mass, same stream;
                                                             # running maximum if R ever falls   (coverage.py:173-186)
IW.lookup(table, masses) -> (R float64, index)               # smallest grid mass >= the row's own (coverage.py:196-205)
IW.row_thresholds(table, masses) -> float32 [n]              # lookup, then float32 (table_float32's refusal of a
                                                             # finite R that overflows, learners_iw.py:589-597)
IW.reproduces(reference, scores, stream) -> bool             # S3: unweighted calibrate == stored threshold, bit for bit
IW.per_row_dose(reference, eta, R_rows, blend, most_caution) -> (CQLLevelDose, refused)            # traceable
IW.per_row_iql_weights(reference, eta, R_rows, adv, beta, cap, most_caution) -> (LevelActorWeights, refused)
```
- `per_row_dose`: width_i = R_i x max(eta_i, 1e-6) x u with `frozen_level_dose`'s checks row by row, dose by
  `level_critic_dose`; not ready: width 0, dose 1; refused (ready and R_i = +inf): `most_caution` False keeps 1
  (standard BCA's rule), True gives 1 + blend = 1.5 (IW's rule; the weighting test's own code for TD3+BC, a new rule
  for CQL and ReBRAC). With every R_i equal to the reference's finite threshold it equals `frozen_level_dose` bit for
  bit. `most_caution` is a traced boolean (no recompilation between standard and IW).
- `per_row_iql_weights`: `postcap_level_actor_weights` with each row's own width on positive-advantage supported rows;
  Unifloral's own weight `jnp.exp(adv * beta).clip(max=cap)` on every other row and on every row before the first
  refresh; refused rows: `most_caution` False gives weight 0 (unsupported; `has_support` False when no row is
  supported, so the step skips the actor update), True gives weight 1 where A > 0 and Unifloral's weight elsewhere.
- The table's inputs: `scores, stream` from `refresh(...)`; bank weights = `bank/weight_strong`; masses =
  `train/weight_strong`; grid built on those masses. With every weight 1 the grid is [1.0] and IW equals standard bit
  for bit (S2, test (b)).

## 12. Scoring-side files (scoring agent; summarized for the others)

- Actor parameters go from the training environment to the scoring environment as hand-off files:
  `handoff.write(path, handoff.flatten_dict(agent_state.actor.params, "actor"), meta=...)`, read there with
  `handoff.unflatten_dict(arrays, "actor")`. Normalization constants (TD3+BC, IQL) travel with them as
  `norm/obs_mean`, `norm/obs_std` (the network's own fields).
- S7's open loop: states and commands by file (`states` float64 [n, 13] rounded to float32 for the networks), commands
  back as float32 [n, 4].
- CQL scoring noise: section 4; one float32 file per bank, shared by every CQL setup and seed.
- A scores file (`evaluate.score_actor(actor, permission, out=...)`, `evaluate.write_scores`; the driver writes it as
  `lanes/scores-step<T>.npz`, section 16): per-start arrays `<readout>/<bank>/<metric>` (J, Jd, J_capped, points,
  points_capped, regret, f1, f2, late_peak, stall_share, tracking, energy, start; the viewer's episodes under
  `<readout>/<bank>/episodes/<field>` with --keep-viewer) and the meta {schema "shift-hosts-scores-v1", learner,
  actor_npz_sha256, actor_meta (the actor file's meta), permission ({kind: "made-up"} or {kind: "approval", ...}),
  summaries.<readout>.<bank> (evaluate.score's summary), cache, anchors.<bank> {scale, bottom, top, ...}, code_sha256
  (evaluate.code_sha256), environment}. Readouts: "scored" for every learner, and CQL's "tanh_mean".
- The scoring cache is pinned in `evaluate.py`: `load_cache` reads only the bank files whose SHA-256 is
  `CACHE_NPZ_SHA256[bank]`, and `check_cache_files()` checks those and CQL's noise files (`CQL_NOISE_NPZ_SHA256`), so
  a cache rebuilt after the freeze changes the code (S1) before it can change a score.
- Permission (section 8): `guard.MADE_UP` rolls out only made-up actor files that no run trained; a trained actor
  (`meta.trained` true, `kind` "run-snapshot", as the runner writes every actor file) needs the verified approval for
  `score_actor`, `visited_transitions`, `actor_transitions`, `closed_loop_check` and `open_loop_states` with banks.

## 13. The run record (runner.py; one JSON per run, as `runner.execute` writes it)

Written with `handoff.write_json` to the run's folder (section 16) as `record.json` (with sidecar); `started.json`
beside it is written at the start ({identity, started_utc, pid, boot_id}; `--replace-incomplete` deletes nothing while
that process is alive on this boot). "Hash" means `guard.state_sha256` unless said otherwise. The plan's schedule
(`runner.make_schedule()`): 40 chunks of 2,500 outer steps, refreshes after 10,000, 15,000, ..., 95,000 (18), key
hashes at 2,500 / 50,000 / 100,000, S10 at 10,000, snapshots at 10,000, 50,000, 95,000 and 100,000; stage 1 truncates
it to 15,000 (one refresh); made-up runs may shrink it (the dry run: 40 steps in chunks of 10, refreshes after 20 and
30).
```text
schema            "shift-hosts-run-v1"
kind              "run" | "stage1-check" (stage 1) | "stage0-test" (made-up arrays)
identity          {schema, run_id, stage, quality, seed, learner, setup, made_up, device, schedule: Schedule._asdict()
                   {num_updates, eval_interval, steps, refresh_steps, switch_step, key_hash_steps, s10_step,
                   snapshot_steps, last_refresh}, data: {npz_sha256, manifest_sha256}, code: {code_manifest_sha256 (the
                   canonical SHA-256 of gate.code_manifest()), files, fingerprint, prespec_sha256 (both null for
                   made-up runs: no approval)}}; a finished run is skipped when a new start has the same identity
identity_sha256   runner.canonical_sha256(identity) (SHA-256 of the sorted, compact JSON of handoff.plain)
run_id            "<stage>/<quality>/seed<k>/<learner>-<setup>" (= identity.run_id = the folder under the root)
stage, quality,   as in identity; quality is the prepared file's label: the plan's quality for real runs,
seed, learner,    "made-up-<quality>" in the dry run
setup
outcome, stop     "complete" | "stopped"; stop: null | {rule: "S5" | "S3", step: int, reason: str} (S11 on the
                   prepared rows and S9 before the first chunk stop before a run starts, so no record exists: the
                   runner prints {"status": "stopped", "record": false, "stop": {rule, step, reason}} and exits 3)
data              {path, npz_sha256, manifest_sha256, made_up, rows: {train, full, bank, test_strong}, train_set: "train"
                   | "full" (room), train_sha256 (guard.tree_sha256 of the training Transition), bank_sha256 (BCA
                   setups), next_action_substituted (ReBRAC: the prepared file's shares {train: {share,
                   highest_current_tenth: {share}}, bank: {share}}; else null)}
seeds             {rep, unifloral, s91, s39}
settings          {args: asdict(Args), bca: asdict(BCASettings) | null, num_updates, eval_interval, steps,
                   refresh_steps ([] for the setups that never refresh), snapshots, key_hash_steps, s10_step, refusal:
                   "standard" | "most_caution" | null, refusal_record, weights_one, programs (CQL's BCA setups: {pre,
                   post}), iw_grid: {points, first, last, sha256} | null}
code              identity.code plus approval: Approval.describe() | null and unifloral_sha256 (each file's SHA-256)
environment       {import_record: unifloral_import.record() (Python, executable, packages, card {name, driver_version,
                   memory_total}, env, xla_flags {before, after each import, in_effect}, matmul_precision, jax_platforms,
                   backend, devices, device, unifloral_revision, manifest_sha256, ...), host, taskset_cores}
timings           {started_utc, finished_utc, wall_seconds, processor_seconds (user + system, self and children),
                   card_seconds (the wall seconds on the card; 0 on the processor), compile_seconds (JAX's trace,
                   lowering and compile time in the run, jax.monitoring), first_chunk_seconds,
                   steady_ms_per_outer_step (the median over chunks 2+ of chunk time minus the compile time inside the
                   chunk, per outer step; S6), outer_steps, chunk_seconds: [...], chunk_compile_seconds: [...],
                   refresh_seconds: [...], table_seconds: [...], readout_seconds: [...]}
hashes            initial: {agent_state: {<each AgentTrainState field>, all}, key, scale, bca_state (BCA setups)}
                     (taken before chunk 1; S4 compares agent_state.all across a learner's setups, room included)
                  key_carry: {"<t>": hash of Unifloral's key carried out of the chunk ending at t} for t in
                     schedule.key_hash_steps, before the evaluation split (S4)
                  params_10000: {step, agent_state, key, parts: {<field>, all}} at schedule.s10_step, just before the
                     first refresh (S10); null when the run stopped before it
                  final: {step, losses_sha256 (every chunk's outputs, chained), parts: {<field>, all}, key, agent_state,
                     bca_state and bca_state_without_refusal_rule (BCA setups)} at the run's last step (S2, S8; IQL's
                     parts dual_q, dual_q_target, value are its Q and V)
                  The AgentTrainState fields: TD3+BC and ReBRAC actor, actor_target, dual_q, dual_q_target; CQL actor,
                  vec_q, vec_q_target, alpha; IQL actor, dual_q, dual_q_target, value.
snapshots         {"<t>": {path (the whole carry, handoff file snapshots/step<t>.npz), sha256, actor: {path
                   (actors/step<t>.npz, actors_f32.write_actor_file), sha256 (its npz SHA-256)}}} for t in
                   schedule.snapshot_steps; a snapshot is the state the next chunk starts from (after the evaluation
                   split and the refresh). The actor file's meta names learner, setup, step, quality, seed, made_up,
                   run_id, identity_sha256, actor_tree_sha256, run_dir, trained (true) and kind ("run-snapshot")
                   (scores files carry it as actor_meta)
validity          {all_steps_valid: bool, first_invalid_step: int | null, flags: [runner.FLAGS[learner]] ([] alone),
                   losses_checked}
losses            per chunk: {step, program ("main", or CQL's "pre" / "post"), seconds, compile_seconds, sha256, mean:
                   {every output's mean over the chunk}, dose_min/dose_max or weight_min/weight_max}
refreshes         [{step, seconds, valid, stored_inputs_valid, new_reference_valid, certified, threshold, lambda_hat,
                    lambda_hpd, n_eff,
                    sigma_post, clamp_binds, unit, bank_rows, scores_sha256, stream (apply_refresh's record), refusal,
                    s3_reproduced (IW: true; else null), iw (IW: {table: Table summary, grid_points, monotone,
                    running_max_applied, max_mass_gap, max_threshold_gap, possibly_rounded_into_refusal, rows,
                    unweighted_threshold, kish, bank_rows, train_rows, *_sha256}; else null),
                    s3_recomputed_equals_refresh_scores, iw_refused_rows, table_seconds (IW), train_rows (the band at
                    every training row: mean_dose, min_dose, max_dose, refused_rows, supported_share, inputs_valid; IQL
                    mean_weight, mean_unifloral_weight, mean_weight_shrinkage, positive_advantage_share, refused_rows,
                    supported_share, inputs_valid), refused_rows, mean_dose (IQL: the mean weight)}]; a refresh that
                    fails stops the run (S5, or S3 when the IW recomputation does not reproduce it) and is the last
                    entry. R = +inf ("cannot certify") is counted in refused_rows, not a stop; a NaN threshold stops
readouts          IW setups at the last refresh, else null: {info, value: "dose" | "actor weight" (IQL), dose_change:
                   iw.dose_change(IW dose, the dose at the standard threshold under standard BCA's refusal rule, the
                   highest-current tenth) = {mean (of the IW dose), change_vs_unweighted (mean of IW - standard),
                   mean_abs_change, max_abs_change, high_current_rows, high_current_change}, exact: {test_steps},
                   estimated_weights and shuffled_weights: {table, kish, train_rows, test_steps, dose_change,
                   train_readout}}
```
The design effect of plan 5.4 (the correlation of the 5 bank scores within an episode) is not in the record (an open
item for the runner's owner); `refreshes[*].n_eff` is the calibration weights' Kish size, not the design effect.

## 14. `tests/madeup.py` and test conventions

```python
from experiments.shift_hosts.tests import madeup
m = madeup.build(seed=0)            # deterministic; registers with the guard (register=False to skip)
m.train, m.bank, m.strong           # Rows(obs f32 [n,13], action f32 [n,4] in [-1,1], reward f32 [n], next_obs, done)
m.n                                 # about 1,980 kept training rows; bank 300 rows; strong 1,000 steps
m.rows("train") -> dict             # for UI.make_transition / UI.init_like_training_block
m.leaves(which, learner)            # Unifloral's field order (ReBRAC adds next_action)
m.next_action[set], m.next_true[set]    # ReBRAC's a' by the set's rule, and whether it is the true next command
m.episode[set], m.t[set], m.current[set], m.weight[set] (exact strong-law w(c)), m.in_ball[set]
m.R_levels      # float32 [n]: thresholds at 4 levels [0.5, 1, 2, 4][row % 4] (test b2)
m.R_refused     # float32 [n]: +inf where row % 3 == 0, else 1.0 (test b3)
m.dose          # float32 [n] in [1, 1.5) (test a3); m.width: float32 [n] positive (test a3, IQL)
m.arrays()      # flat names for handoff.write
```
- Pass the made-up arrays whole and unchanged (float32) to anything the guard checks.
- A test module ends with
  ```python
  if __name__ == "__main__":
      UI.prepare()                                   # training environment only; first
      from experiments.shift_hosts import tests as T
      sys.exit(T.run(sys.modules[__name__], "<area>", extra=UI.record))
  ```
  `T.run` prints the tests and writes `runs/shift_hosts/stage0/<area>/<module>_<device>.json` (+ sidecar) with every
  case's outcome, the seconds and processor seconds; exit status 0 only if all passed. Device: `SHIFT_HOSTS_DEVICE`
  (`cpu`, `gpu`) or `device_name="none"` for modules without JAX. Call `UI.prepare()` again in `setUpClass` (it is
  idempotent) so the module also runs under another loader.
- Cross-environment tests start the other environment as a subprocess of
  `python -m experiments.shift_hosts.tests.<module> --role <name> --dir <folder>` and pass arrays by hand-off files
  (see `test_guard_compare.py`, `test_handoff.py`).

## 15. Words

- **Outer step**: one call of a Unifloral file's training step. **Chunk**: 2,500 outer steps in one `lax.scan`.
- **Band**: q +- U around the critic's value. **Width** U = R x u x max(eta, 1e-6). **Scale network** eta.
  **Unit** u (running average of the error size). **Threshold** R; R = +inf is **refused** ("cannot certify").
- **Dose**: 1 + 0.5 U / (U + u), between 1 and 1.5, multiplying one per-row term. **Refresh**: freezing eta and u and
  setting R on the bank. **Bank**: held-out rows used only to set R.
- **Standard BCA** weighs every bank row equally; **IW** (importance-weighted) BCA weighs bank rows by w(c) and gives
  each training row its own threshold R(w_i).
- **Made-up arrays**: random arrays shaped like ThrusterChain's, never its data. **Hand-off**: a file passed between
  the two Python environments. **Fingerprint**: SHA-256 of the frozen pre-specification and the code manifest.

## 16. Run folders, lanes, the stage driver's files and command lines (runner.py, stages.py, band_readout.py)

`<root>` is the stage driver's `--root`: for the real study `runs/shift_hosts` (`runner.RUNS`) up to version 3, and
version 4's `runs/shift_hosts/supplement_v4` (`supplement.ROOT`, the default and the only real root; section 18); the
dry run defaults to `runs/shift_hosts/stage0/runner/dryrun` and refuses both real roots and any folder inside the
supplement's. `<label>` is the prepared file's quality label: the quality itself for real runs, `made-up-<quality>` in
the dry run (`stages.Driver.card_job`).
```text
<root>/data/<quality>/seed<k>.npz                      the dry run's made-up prepared files (real runs read
                                                       runs/shift_hosts/data/<quality>/seed<k>/prepared.npz, section 6)
<root>/<stage>/<label>/seed<k>/<learner>-<setup>/      one run (runner.run_dir); stage = stage1 | stage2
    started.json, record.json (+ sidecars)             section 13
    snapshots/step<t>.npz, actors/step<t>.npz          the whole carry; the actor parameter file (actors_f32)
    lanes/scores-step<T>.npz (+ manifest, sidecars)    the scoring lane, evaluate.score_actor(snapshots[T].actor.path,
                                                       permission, out=...), at T = steps / 2 (50,000) and T = the final
                                                       step (100,000), for every complete stage-2 run (real runs only:
                                                       the dry run scores nothing)
    lanes/viewer-step<final>.npz                       seed 0's final standard and IW actors (band_readout viewer; real)
    lanes/coverage-step<R>/coverage.json, coverage.npz the coverage lane of standard and IW runs at the last refresh R
                                                       (95,000): band_readout coverage (real runs first write
                                                       actor_transitions.npz and visited_transitions.npz there). Note:
                                                       coverage.npz's hand-off manifest is also named coverage.json, and
                                                       band_readout writes coverage.json over it, so coverage.npz is
                                                       checked by its sidecar and by the SHA-256 coverage.json names
                                                       (arrays_npz_sha256), not by handoff.read
    lanes/s7-step<final>/states.npz, commands.npz, s7.json   S7's lane: seed 0's alone, standard and IW final actors
                                                       and every stage-1 actor ({passed, s7_passed, closed_loop, ...});
                                                       version 4 (s7_all_seeds): every seed's (lane <run>/s7)
    lanes/s7-step<half>/ (the same files)              version 4 (s7_half): S7 on the 50% actor (50,000 steps) of
                                                       the same runs (lane <run>/s7-<half>); stages._lane_step reads
                                                       the step from the folder name
<root>/driver/status.json (+ sidecar)                  {schema "shift-hosts-driver-status-v1", mode "real" | "dry-run",
                                                       state, phase, pid, started_utc, updated_utc, concurrency, card,
                                                       lanes, counts, budget, decisions, checks {name: passed},
                                                       claim0_failures, accepted_s5, s7_stopped, stop, notes,
                                                       last_events; version 4: s7_results (how many), supplement};
                                                       version 4 drops nothing: counts.cards_dropped and
                                                       counts.lanes_dropped stay 0
<root>/driver/state.json (+ sidecar, + .prev)          {schema "shift-hosts-driver-state-v1", mode, config (learners,
                                                       qualities, seeds, device, choice 19, schedule; version 4: plan,
                                                       s7_all_seeds, s7_half: a resume must match it), decisions {s6
                                                       (with stage1_records, decided_utc: a resume reuses it while
                                                       stage 1's records are unchanged), concurrency}, checks,
                                                       claim0_failures [{run, step, reason}], accepted_s5 [run ids],
                                                       card_walls, s9_reference {signature (runner.env_signature),
                                                       source}, lanes_done,
                                                       s7_stopped [{quality, learner, scope "learner"|"iw", setup,
                                                       run, stage, file, at_utc; version 4: step, version 4}]
                                                       (version 3: a failed S7 check stops its learner, or only its
                                                       IW setup, in that quality; stages.S7_SCOPE; version 4: it is
                                                       recorded once per run and scope and drops nothing; the key
                                                       keeps its name), s7_results and supplement (version 4,
                                                       section 18)}
    checks (each with passed)                          S2/<learner> (with uncertified_refreshes and, when a refresh
                                                       could not certify, its note), S8/<learner>, S4/<learner>,
                                                       S10/<learner>/<setup> (stage 1; readout true for choice 19's
                                                       learners), S7/<learner>/<setup> (every stage-1 actor;
                                                       scoped_stop "learner"|"iw" on a failed check, version 3), S6,
                                                       S6b/<quality>/<unit> (skipped true when every run of the unit
                                                       had a record), S4/<quality>/seed<k>/<learner>,
                                                       S4/<quality>/seed<k>/<learner>/room,
                                                       S7/<quality>/seed0/<learner>/<setup> (scoped_stop on a
                                                       failed check; version 3: skipped for a run of a scope S7
                                                       stopped before it trained or before its lane ran; version 4:
                                                       every planned seed-0 run, none skipped; seeds 1-4 and the 50%
                                                       actors write no check key, only s7_results), S9/<run id> (on
                                                       failure)
<root>/driver/driver.log                               "[<utc>] <message>" lines; "<run id>: S5 accepted by the operator
                                                       as the setup's outcome (--accept-s5)" (also state.json
                                                       accepted_s5); version 4: "version 4: stage-2 plan ...; S7 on
                                                       ...", "S7 (version 4): <lane>: the float32 check failed (see its
                                                       s7.json); recorded with its scope (<learner L | L's IW setup>
                                                       on <quality>); nothing is dropped; no judged result changes
                                                       (version 3's are carried), and the supplement's numbers are
                                                       shown, never judged"
<root>/driver/driver.lock                              flock held for the driver's life: one driver per root
<root>/driver/STOP.json, STOP.txt (+ sidecar)          a halt {schema "shift-hosts-driver-stop-v1", mode, rule, reason,
                                                       at_utc, phase, run, budget, claim0_failures, s7_stopped,
                                                       next_steps}; version 4's STOP.txt adds a line with the number
                                                       of S7 failures recorded so far and their runs;
                                                       --resume-after-stop moves them to STOP.json.<time>,
                                                       STOP.json.sha256.<time>, STOP.txt.<time>
<root>/driver/ledger.json (+ .prev)                    budget.Ledger: mode, stage0, card intervals and processor
                                                       amounts (open entries of running children, moved by the
                                                       heartbeat; recovered entries of runs that finished unwatched;
                                                       the driver's own time under kind "driver")
<root>/driver/logs/<job>.log, nohup.out, STOP_REQUEST
outputs/shift_hosts/results.json (+ sidecar)           export.py (section 17); version 4: supplement.py export,
                                                       version 3's file plus three keys (section 18)
```
Stage 3's twins (section 3), when they exist, go under `<root>/stage3/` in the same layout.

Command lines (from `/root/bca-hosts`, as root in WSL; the driver pins every child with taskset and gives it its
environment). Everything that touches real data refuses without the user's terminal approval:
```bash
# the stage driver (training environment)
python -m experiments.shift_hosts.stages start --stage0-card-hours <h> --stage0-processor-hours <h>  # detached
python -m experiments.shift_hosts.stages run [--dry-run] [--root R] [--learners ...] [--qualities ...] [--seeds ...] \
    [--card-device cpu|gpu] [--card-cores 0 1] [--lane-cores 2 ... 9] [--choice19 learner=yes|no ...] \
    [--accept-s5 RUN_ID ...] [--inject RUN_ID=unit0:<step>|nan:<step>|s7fail] [--replace-incomplete RUN_ID ...] \
    [--resume-after-stop] [--caps card,processor] [--supplement-plan PLAN.json] [--s7-all-seeds] [--s7-half]
# On real data the driver takes learners, qualities, seeds, choice 19, the S9 signature and (version 4) the supplement
# plan and S7 lanes from prespec.json, and refuses --dry-run roots, --root other than runs/shift_hosts/supplement_v4
# (version 4; version 3's was runs/shift_hosts), --learners/--qualities/--seeds/--choice19/--card-device that differ
# from prespec.json, --supplement-plan, --s7-all-seeds, --s7-half, --caps, --inject, more than 2 card cores or 8 lanes;
# before it writes anything it requires S0 (gate.require_s0), stage 0's compute entry (version 4: in the supplement's
# own ledger, so the first start names both hours) and, version 4, supplement.check_sources and an empty
# supplement.code_unchanged. A dry run never uses runs/shift_hosts, runs/shift_hosts/supplement_v4 or a folder inside
# it; without --supplement-plan (a {quality: {learner: [setups]}} file, supplement.check_plan) it trains the full grid.
python -m experiments.shift_hosts.stages status [--root R | --dry-run]          # real default: version 4's root
python -m experiments.shift_hosts.stages request-stop [--root R | --dry-run]    # real default: version 4's root
# the lanes' scoring-environment steps (the driver starts these)
python -m experiments.shift_hosts.stages lane score --actor A --out <run>/lanes/scores-step<T>.npz [--keep-viewer] [--made-up]
python -m experiments.shift_hosts.stages lane transitions --actor A --prepared P --out-actor F --out-visited F
python -m experiments.shift_hosts.stages lane s7-states --actor A --out states.npz [--made-up [--made-up-states F]]
python -m experiments.shift_hosts.stages lane s7-check --states S --commands C --actor A --out s7.json [--no-closed-loop] [--made-up [--inject-failure]]
# one run (training environment; the driver starts these)
python -m experiments.shift_hosts.runner run --learner L --setup S [--weights-one] --stage stage1|stage2 --out-root R \
    --device cpu|gpu --steps N (--quality Q --seed K | --made-up-file F [--num-updates N --eval-interval N \
    --refresh-warmup N --refresh-every N] [--inject-fault unit0:<step>|nan:<step>]) [--replace-incomplete]
python -m experiments.shift_hosts.runner s7-commands --states S --actor A --out commands.npz
python -m experiments.shift_hosts.runner made-up-file --seed K [--rep K] [--quality made-up-<q>] --out F
# band readouts (training environment, processor)
python -m experiments.shift_hosts.band_readout coverage --run-dir D [--step R] --out <run>/lanes/coverage-step<R> \
    [--actor-transitions F --visited-transitions F | --made-up-file F]
python -m experiments.shift_hosts.band_readout viewer --run-dir D [--step T] --scores <run>/lanes/scores-step<T>.npz \
    --out <run>/lanes/viewer-step<T>.npz [--made-up-file F]
# results.json (scoring environment)
python -m experiments.shift_hosts.export [--out outputs/shift_hosts/results.json]   # up to version 3; never under
#   version 4: it would read version 3's root, leave out every record there under S1 and write an all-excluded build to
#   outputs/shift_hosts/results.json (its code is version 3's, so it cannot refuse by itself; section 18)
python -m experiments.shift_hosts.supplement export [--out outputs/shift_hosts/results.json]   # version 4 (section 18)
```
Exit statuses: runner 0 complete (or skipped), 3 stopped by a stop rule (a stop before training prints the rule on
its status line), 2 refused, 1 any other error; the driver 0 finished, 3 halted or interrupted (STOP.json written), 2
refused, 1 error (STOP.json written too, after waiting up to an hour for the running children); a lane step 0 done,
2 refused, 3 S7 failed (s7-check), 4 a non-finite return in its scores (score; the driver decides: Claim 0 outcome or
S5), 1 error; band_readout 0, 2 refused, 1 error; supplement export 0 written, 2 refused (nothing written), 1 error.

## 17. `results.json` and where each of its fields comes from (export.py; the export/runner mapping)

`export.build(approval)` (real; every record kind "run", made_up false, the approval's fingerprint and
pre-specification, the frozen code manifest's SHA-256 and the plan's schedule; driver mode "real") or
`export.build(guard.MADE_UP, frozen=Frozen(None, None, code_manifest, made_up=True, code_manifest_sha256=...))`
(stage-0 tests: records kind "stage0-test" in a dry-run root). export.py reads the files of section 16 where and as
they are written, renames nothing, and restates the few rules it shares with runner.py and stages.py (canonical_sha256,
run_id, the plan's schedule and the 50% step, env_signature, choice 19, the schemas, the accepted-S5 log line);
`tests/test_claims.py` checks them against the originals and runs `export.build` over the driver's own dry run.

| What (results.json, claims input) | Read from |
|---|---|
| made-up mark | `record.data.made_up` and `identity.made_up`; scores `meta.permission.kind` and `meta.actor_meta.made_up`; S7 `states.npz` meta `made_up`; the driver's `status.json` `mode` |
| run identity | `record.run_id` = `identity.run_id` = its folder; `record.{stage, quality, learner, setup, seed}`; `identity_sha256` = canonical_sha256(identity) |
| code identity (S1) | `record.code` and `identity.code` {fingerprint, prespec_sha256, code_manifest_sha256}; scores `meta.code_sha256` against the frozen code manifest |
| schedule, final and 50% steps | `identity.schedule`: steps, and num_updates / 2 rounded up to a chunk |
| per-seed returns, points, F1/F2 counts, starts, finiteness | `lanes/scores-step<final>.npz` `meta.summaries.scored.<bank>` {J_mean, points_mean, f1_rate, f2_rate, n, finite} and every other summary field; each bank's J_mean, F1 and F2 rates recomputed from the per-start arrays `scored/<bank>/{J, f1, f2, start}` bit for bit |
| the actor scored | scores `meta.actor_npz_sha256` = `record.snapshots[T].actor.sha256`; `meta.actor_meta.{run_id, step, identity_sha256}` = the record's |
| 50% snapshot; CQL's tanh of the mean | `lanes/scores-step<half>.npz`; readout `tanh_mean` (shown, never judged) |
| coverage for Claims 4a, 4b | `lanes/coverage-step<R>/coverage.json` (`record_sha256`, `run_id`, `learner`, `setup`, `step`, `iw_table`, `arrays_npz_sha256` checked) `groups.recorded.<arm>.{outside, inside, all}.coverage`: own = arm `iw` for an IW setup, `standard` for a standard setup; standard = arm `standard` |
| coverage at the three kinds of pairs | `groups.{recorded, actor, visited}.{standard, iw}.{outside, inside, all}.{steps, coverage, interval, refused_share, certified_coverage}` (the dry run has the recorded pairs only) |
| Claim 2's pointer {change, extra} | IW records' `readouts.dose_change` at the last refresh: change = `change_vs_unweighted`; extra = (`mean` - `change_vs_unweighted`) - 1, the mean standard-threshold dose minus 1; IQL: minus `refreshes[-1].train_rows.mean_unifloral_weight` (the mean standard weight minus Unifloral's own) |
| a stop and its kind | `outcome` "stopped", `stop` {rule, step, reason}: an outcome (not finite) when S5 on alone or room, or an S5 on a BCA setup the operator accepted (driver.log's line; `state.json` `accepted_s5` if a later driver writes it); any other stop halts and its run counts as missing |
| S4 | `hashes.initial.agent_state.all` (room included), `data.train_sha256` (not room), `hashes.key_carry` before a stop (alone, standard, IW); the driver's S4 checks |
| S10 | `hashes.params_10000.{agent_state, key}` against the alone setup's; the driver's `S10/<learner>/<setup>` (a readout under choice 19) |
| S3; S9; S7 | `refreshes[*].s3_reproduced` (IW); `environment.import_record` through env_signature against `state.json` `s9_reference.signature` and the driver's `S9/<run>`; `lanes/s7-step<T>/s7.json` `s7_passed` (and `closed_loop.passed` for real runs), bound to the record's actor by `states.npz`'s meta |
| S2, S6, S6b, S8 | the driver's `state.json` checks |
| S5 | `state.json` `claim0_failures`, the accepted S5s, the records' stops |
| S7's scopes (version 3) | `state.json` `s7_stopped`: per quality, a learner stopped (alone, standard or weights-of-1 actor failed) or only its IW setup (IW actor or its repeat failed); the driver drops that scope's later runs and lanes and goes on. Version 4's supplement records the same entries (with the actor's step and version 4) and drops nothing; every lane's verdict is in `state.json` `s7_results` (section 18) |
| S11 | each scores file's weighting-bank anchors (bottom, top) against start_banks.json |
| S0 | not recorded here (stage 0's `runs/shift_hosts/stage0/SUMMARY.json`) |
| the driver's halts | `STOP.json`; `STOP.json.<time>` with `STOP.json.sha256.<time>` (moved by --resume-after-stop) |
| choice 19 | the driver's stage-1 S10 checks (`readout`), else `export.CHOICE19` (= `stages.CHOICE19`) |
| IQL's Q and V identical across its three runs | `hashes.final.parts.{dual_q, dual_q_target, value}` (three-way, and standard against IW) |
| ReBRAC's substituted next commands | `record.data.next_action_substituted`, else the prepared file's manifest |
| band readouts | `refreshes[*]` {certified, threshold, refused_rows, iw_refused_rows, mean_dose, train_rows}; `readouts.{estimated_weights, shuffled_weights}.dose_change`; the design effect is not recorded |
| compute | `timings` {wall_seconds, card_seconds, processor_seconds, compile_seconds, steady_ms_per_outer_step}; `environment.host`, `identity.device`, `environment.taskset_cores` |
| the reference row | data_copy's verified copies of the weighting test's evaluation records and RESULTS.json (only under a verified approval, or a made-up copy in tests) |

`results.json` (schema `shift-hosts-results-v1`): `label`, `exploratory`, `made_up`, `caveats`; `built` (time,
permission, frozen identity, export's code SHA-256, the driver's files read and its `s7_stopped`, the anchors, the
environment); `badges`, `reading_order_rule`; `integrity` (S0-S11: per rule the driver's status, export's own checks
and the halts, the worst of them, and why when nothing records it; S7 reads "outcome" when every failed check is
covered by a scope the driver recorded and its `s7.json` verifies; `unknown_stops`; `driver`, with `s7_stopped`);
`qualities.<quality>` {label, reading_order, claim0_bar, complete (with `s7_not_expected`: the setups a stopped scope
leaves out), families (each with `s7_stopped`: the learners entered with p = 1), claims.<learner>.<claim>, table (the
12 rows), expanded.<learner> (setups, pointers, readouts), writeups (an S7 write-up per stopped scope), stage3_order,
totals (card and processor seconds of every run read, scored or not), s7_stopped {scopes, excluded (the runs left out,
those that trained but were never scored included), rule}}; `study_rule`; `reference_row`; `runs` (every run read:
quality, learner, setup, seed, record and its SHA-256, run_id, kind, and each field "read" or "not recorded: <why>"; a
stopped scope's run that was never scored carries `s7_stopped`); `excluded` (folders left out, with the reason);
`unscored` (complete runs without final scores; a stopped scope's carry `s7_stopped`); `halted_runs`. Made-up results are
written only under `runs/shift_hosts/stage0/`; real ones to `outputs/shift_hosts/results.json` or under
`runs/shift_hosts/`. Version 4: version 3's file built this way is carried byte for byte, and
`outputs/shift_hosts/results.json` is written by `supplement export` with exactly three keys added (section 18).

## 18. Version 4's S7 supplement (`supplement.py`; PRESPEC.md section 21, post hoc)

Every stage-2 run of every scope version 3's S7 check stopped is trained and scored in full under the supplement's own
root and shown beside version 3's results, never judged. A failed S7 check there is recorded and drops nothing. Version
3's `results.json` is carried byte for byte, and every file of version 3's approved code manifest but
`supplement.CHANGEABLE` stays byte-identical (`claims.py`, `export.py`, `runner.py`, `evaluate.py`, `band_readout.py`,
`gate.py`, `budget.py`, `approve_hosts.py` and every training and scoring file included), so nothing judged can move.
`supplement.py` imports only the standard library, `gate` and `handoff` at module level, so the stage driver imports it
in the training environment; `claims`, `export` and `budget` are imported inside the functions that use them (the export
runs in the scoring environment).

**The frozen block**, `prespec.json` `"supplement"` (`supplement.validate` refuses anything but exactly these keys and
values that fit them; no agent and no code changes it):
```text
version, root   4 and "runs/shift_hosts/supplement_v4" (supplement.ROOT_REL; supplement.ROOT = gate.REPO / ROOT_REL)
source          {fingerprint, approval_commit}: version 3's (supplement.V3); {state_sha256, ledger_sha256,
                results_sha256}: version 3's runs/shift_hosts/driver/state.json, driver/ledger.json and
                outputs/shift_hosts_v3/results.json (supplement.SOURCE_FILES)
scopes          [{quality, learner, scope "learner" | "iw"}]: version 3's state.json "s7_stopped" as claims.s7_scopes
                reads it (a stopped learner outranks a stopped IW setup)
plan            {quality: {learner: [setups]}} = plan_of(scopes, choice 19): a learner scope sham_alone (choice-19
                learners only), alone, standard, iw, room; an IW scope alone, standard, iw
s7              {seeds [0] or every data seed, steps [100000] or [50000, 100000], setups ["alone", "standard", "iw"]}
                -> s7_all_seeds (seeds != [0]) and s7_half (50000 in steps)
```

**Functions** (`S` is `experiments.shift_hosts.supplement`, as `stages.py` imports it; every refusal is a
`gate.ApprovalError`):
```python
S.scopes_of(entries, qualities)           # {q: {L: "learner"|"iw"}} from a state.json's s7_stopped (claims.s7_scopes)
S.scope_map(items, learners, qualities)   # the block's scopes list -> {q: {L: scope}}, every quality present
S.plan_of(scopes, choice19, learners)     # the plan of the scopes; qualities without a scope left out
S.check_plan(plan, learners, qualities, choice19)  # a plan as the driver uses it (also a dry run's --supplement-plan):
                                          #   known names, each setup once, sham_alone only with choice 19, room only
                                          #   beside alone; returns a plain copy
S.run_ids(plan, seeds, learners, made_up) # {q: [stage-2 run ids]} in plan_units / unit_jobs order: sham-alone runs,
                                          #   each seed's runs (learner order, then alone, standard, iw), room runs
S.validate(block, learners, qualities, seeds, choice19)  # -> dict(plan, scopes, s7_all_seeds, s7_half)
S.prespec_block(package)                  # prespec.json's "supplement" (refused when missing)
S.provenance(block, plan)                 # state.json "supplement": {version, root, plan_sha256, v3, source}
S.code_unchanged(manifest_now, package)   # [{path, change "changed"|"added"|"removed"}] outside CHANGEABLE, against
                                          #   approval_fe9fb77b0851c75b.json's code_manifest (its sidecar checked);
                                          #   empty: every other file, external code included, is version 3's
S.check_sources(block, repo, package)     # stages.preflight, before a real start: version 3's state.json, ledger.json
                                          #   and committed results.json by SHA-256 (runs/shift_hosts/results_v3.json
                                          #   too when it exists); that results.json and version 3's approval file and
                                          #   sidecar committed in HEAD as on disk; the approval's fingerprint and the
                                          #   commit that last touched it (6f9ce2b); the scopes recomputed from
                                          #   state.json's s7_stopped equal to the block's; returns what it checked
S.read_carried(path, sha256)              # version 3's results.json: sidecar verified, SHA-256 equal to the block's
S.judged_scope_failures(s7_results, scopes)  # the failed S7 checks on actors of scopes version 3 judged: a learner it
                                          #   did not stop in that quality, or, where it stopped only the IW setup, an
                                          #   actor outside it (claims.S7_SCOPE "learner": an IW scope's comparators)
S.caveat_of(judged)                       # the one appended caveat: CAVEAT, plus a sentence naming those actors
S.assert_judged_unchanged(results, v3, caveat=CAVEAT)  # without s7_shown, s7_supplement and the last caveat (which
                                          #   must equal caveat, and caveat start with CAVEAT), results must equal v3
                                          #   in canonical JSON (S.canonical: handoff.plain, sorted keys)
S.reproduction(v4_root, v3_root, v3_fingerprint)  # R1, a readout that never raises (below)
S.shown(q, col, plan_q, scopes_q, v3_q, r1, state, caveat)  # qualities.<q>.s7_shown (below)
S.build(permission, *, frozen=None, root=S.ROOT, block=None, carried_path=None, v3_root=S.V3_ROOT, anchors=None)
                                          # the whole results.json: export.resolve, read_driver, collect,
                                          #   check_score_anchors and integrity over the supplement's root; version
                                          #   3's file deep-copied; the three additions; assert_judged_unchanged
S.main(argv)                              # python -m experiments.shift_hosts.supplement export [--out]:
                                          #   gate.require_run_approval first, then build, then export.write
```

**Constants**: `VERSION` 4; `ROOT_REL`, `ROOT`; `V3_ROOT` (`runs/shift_hosts`, read only); `LABEL` "S7 failed: trained
and scored in full, shown, not judged"; `CAVEAT` (the appended caveat, or its start, PRESPEC.md section 21.7); `V3`
{version, fingerprint, approval_file, freeze_commit, approval_commit}; `SOURCE_FILES`; `RESULTS_V3_PREVIEW`
(`runs/shift_hosts/results_v3.json`); `CHANGEABLE` (`stages.py`, `supplement.py`, `PRESPEC.md`, `INTERFACES.md`,
`tests/test_stages.py`, `tests/test_claims.py`) and `ADDED` (`supplement.py`); `FINAL_STEP` 100,000 and `HALF_STEP`
50,000 (`stages.frozen_config` checks them against `runner.NUM_UPDATES`); `SETUP_FIELDS` (what a shown setup summary
keeps of `claims.summary`: the per-seed values with their mean and sd) and `PAIRED_FIELDS` (those plus se, lower and
upper: a paired difference's one-sided 95% bounds, the owner's decision D5; never t or p_one_sided); `PAIRS` (the
shown paired differences per scope kind); `PAIRED_BANKS` (none, mild, strong);
`EARLIER_SPEND` (versions 1 and 2's spend, an "of which" already counted in version 3's ledger).

**The stage driver with the block** (`stages.py`, version 4):
- `Config` gains `plan` (None: the full grid, a dry run without `--supplement-plan`), `s7_all_seeds`, `s7_half` and
  `supplement` (the frozen block, real mode); `frozen_config` fills them from `supplement.validate` and sets the root to
  `supplement.ROOT`; `config_record` records the plan and both S7 settings, so a resume with others is refused.
- `Driver.in_plan(quality, learner, setup)`: `plan_units` and `unit_jobs` (which lost its `every` parameter) keep only
  planned runs, and a unit without one is skipped ("nothing to run: this unit has no supplement run (version 4)"). The
  projected S7 lanes per run are 1, or 2 with `s7_half`, at seed 0 or every seed with `s7_all_seeds`.
- `lanes_for`: an S7 lane on every stage-1 actor and, in stage 2, on the final alone, standard and IW actor at seed 0
  (every seed with `s7_all_seeds`); with `s7_half` also on that run's 50% actor (`lanes/s7-step<half>`, label
  `<run>/s7-<half>`). Every completed run gets its lanes.
- `_verdict`: every verified `s7.json` is upserted into `state.json` `s7_results` (`_record_s7`); `s7_passed` false goes
  to `s7_failed`, which records the entry once in `s7_stopped` (same five-field key as version 3: quality, learner,
  scope, setup, run; the entry adds stage, file, step, version 4 and the time), logs it and returns without dropping or
  halting. A setup with no scope still halts. An `s7.json` that does not verify or says neither halts.
- `_lane_step_done`: exit 3 at the check's last step reads the verdict and halts unless this lane's own `s7_results`
  entry says it failed; exit 3 at an earlier step halts. A lane that finishes with exit 0 has its `s7.json` read at
  once (`_verdict`): recorded in `s7_results`, a failure through `s7_failed`, a file that does not verify halts, and a
  file that says it failed halts too (the exit status and the file must agree either way). A resumed driver reads every
  finished lane's file again (recorded, never halted for the exit status, which it no longer has).
- `stage1_checks` and `s7_seed0_checks` (every planned seed-0 run): a failure is recorded with `scoped_stop`; a missing
  or unverified record or result halts.
- `seed_checks` keeps its fewer-than-two-runs guard; `room_checks` checks every room run (they come with their alone
  runs).
- `preflight` (real mode, before anything is written): the approval, `frozen_config`, `check_real_args`,
  `gate.require_s0`, `supplement.check_sources`, an empty `supplement.code_unchanged(gate.code_manifest())`, and stage
  0's entry in `<supplement root>/driver/ledger.json` (absent on the first start, so both stage-0 hours are named).
  `check_real_args` refuses `--root` other than `supplement.ROOT`, `--supplement-plan`, `--s7-all-seeds` and
  `--s7-half`.

**The supplement's root and the files version 4 reads**:
```text
runs/shift_hosts/supplement_v4/                     supplement.ROOT: the real driver's default and only root
    driver/state.json, status.json, driver.log, driver.lock, ledger.json (its own: decision D2), logs/, nohup.out,
    STOP.json, STOP.txt, STOP_REQUEST                section 16's driver files
    stage1/medium-expert/seed0/<learner>-<setup>/   stage 1, all 20 runs (section 16's run folders)
    stage2/<quality>/seed<k>/<learner>-<setup>/     the plan's runs only, with lanes/s7-step100000/ and
                                                    lanes/s7-step50000/ on every alone, standard and IW run
runs/shift_hosts/{stage1,stage2,driver}/            version 3's (immutable, chattr +i): read only, never written;
                                                    driver/state.json and ledger.json checked by SHA-256 before every
                                                    start (and by a stage-0 test, read only), the ledger read again
                                                    for the study-wide total; run records, actor manifests and S7
                                                    files read for R1 only, whose readout (reproduction, each shown
                                                    run's r1) feeds no shown score
runs/shift_hosts/results_v3.json (+ sidecar)        version 3's export as built (git-ignored; checked when present)
outputs/shift_hosts_v3/results.json (+ sidecar)     version 3's results.json, committed (a8f96da): the carried file
outputs/shift_hosts/results.json (+ sidecar)        supplement export: version 3's file plus three additions
```

**`state.json` additions** (section 16): `config` gains `plan`, `s7_all_seeds` and `s7_half`; `s7_stopped` entries gain
`step` (the actor's step, from the lane folder) and `version` 4; and two keys:
```text
s7_results  [{label (the lane: <run>/s7 or <run>/s7-<half>), run, quality, learner, setup, seed, step, s7_passed,
              open_loop_passed, open_loop_max_abs {scored, tanh_mean}, closed_loop {"<readout>/<bank>": {max_points,
              mean_points, max_return, passed}} (null without a closed loop: the dry run), file, sha256, at_utc}]
            one entry per lane, upserted by label: a resumed driver reading the same file again adds nothing
supplement  {version 4, root, plan_sha256 (supplement.sha256_of(plan)), v3 (supplement.V3), source (the block's)}
            real mode only; version 3's s7_stopped entries are never copied in
```
`status.json` gains `s7_results` (how many) and `supplement`; `counts.cards_dropped` and `counts.lanes_dropped` stay 0.

**What `results.json` gains** (`supplement.build`; every other key is version 3's, byte for byte):
```text
qualities.<q>.s7_shown      {label, caveat, post_hoc true, version 4, scopes {<learner>: {
                              scope, label, setups_trained,
                              setups {<setup>: {name, seeds, records [per seed: {seed, run_id, state, record_sha256,
                                scores_sha256, scores_half_sha256, trained_under "version 4", v3_record_sha256, r1,
                                r1_differs, v3_excluded}], final and half {<bank>: {J, points: {per_seed, n, nonfinite,
                                missing, complete, mean, sd, seeds}}}}}   (a learner scope: every trained setup; an IW
                                scope: iw only)
                              comparators {<setup>: records}            (an IW scope's alone and standard runs)
                              comparisons [{comparison, bank (none|mild|strong), field (J|points), per_seed, missing,
                                summary {per_seed, n, nonfinite, missing, complete, mean, sd}}]
                                (claims.paired; learner scope: standard - alone, iw - standard, iw - alone, room -
                                alone, sham_alone - alone; IW scope: iw - standard, iw - alone)
                              s7 [the scope's own s7_results entries + verified, bound_to_record: the planned stage-2
                                actors and, on medium-expert, the learner's stage-1 actors; an IW scope's IW and
                                stage 1's iw_repeat only]
                              comparator_s7 [an IW scope's other S7 results, alone, standard and stage 1's alone,
                                standard, iw_ones: version 3's judged runs; [] for a learner scope]
                              v3 (version 3's s7_stopped.scopes entry), v4_entries (version 4's s7_stopped entries)
                              writeup {quality, learner, claim "S7", status LABEL, title, idea, tested, happened, why,
                                cite, markdown}}}}
                            or {label, note} for a quality without a stopped scope
s7_supplement               {version 4, label, caveat, post_hoc true,
                              carried_from {version 3, path, sha256, fingerprint},
                              built {utc, permission, frozen {fingerprint, prespec_sha256, code_manifest_sha256},
                                code_sha256 (export.py's, claims.py's, handoff.py's and supplement.py's), root,
                                driver {path, read, mode, files, why, problems}},
                              block, plan, s7_settings {all_seeds, half_step}, planned_runs (run_ids),
                              plan_check {planned, recorded, not_recorded, outside_the_plan},
                              integrity (export.integrity over the supplement's root), s7_results, s7_stopped,
                              s7_failures_in_judged_scopes (judged_scope_failures: label, run, quality, learner,
                                setup, seed, step, file, sha256),
                              driver_supplement (state.json "supplement"), reproduction (R1), runs, excluded,
                              unscored, halted_runs, totals {card_seconds, processor_seconds},
                              budget {caps, booking, ledgers {"version 4", "version 3"}, study_wide {totals: version
                                3's ledger (which holds versions 1-2's sessions) plus version 4's, earlier: versions
                                1-2's share, an "of which" not added again}}}
caveats                     version 3's list, then caveat_of(s7_failures_in_judged_scopes) last: supplement.CAVEAT,
                            and a sentence naming those actors when there are any (s7_shown's and s7_supplement's
                            caveat field hold the same text)
```
No shown block carries a t, p-value, Holm entry, badge, verdict, mark, bar or window; only paired differences carry
a standard error and one-sided 95% bounds (with a `bounds` note: "shown, not judged"), D5.

**R1** (`supplement.reproduction`): for every `record.json` under the supplement's root, version 3's record of the same
run id under `runs/shift_hosts` (one that names another fingerprint than version 3's is not compared: "missing" with a
note), one entry per run: `run_id`, `result` ("equal", "differs", "missing" or "unreadable"), `differs` and
`not_compared` (lists of fields), `v4_record_sha256`, `v3_record_sha256` and `note`. Compared: `hashes.initial`,
`key_carry`, `params_10000` and `final`, the outcome, the refresh records without timing fields, every common snapshot
actor's `actor_tree_sha256` (an actor file readable on neither side is listed under `not_compared`), and the common S7
lanes' closed-loop largest and mean points. Identity, code and file SHA-256 values differ by design and are not
compared. It never stops anything.

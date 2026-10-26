# UniFloral standalone TD3+BC: first parity gate

Prepared for user execution. No GPU probe or optimizer test was executed during preparation. This is an isolated first integration gate, not a full BCA implementation, penalty selection, or efficacy experiment. The prior CORL suite and paused pilot are separate reference artifacts.

## Identity

- Upstream: https://github.com/EmptyJackson/Unifloral
- Revision: `f2dc1278eae18ed3e22c92255119c54885414d17`
- Standalone source: `upstream/algorithms/td3_bc.py`
- Source SHA256: `6f5ab6f98d706d902a18f11e1c81e99e2691dc4c8a53843bf6521c9f95d00eee`
- Manifest: `provenance/unifloral.json`, SHA256 `7d0c0eccda6730c0b40bf9085cdbc31f565acbd8010db337b01ef5e64cf26a58`
- All 40 vendored files are unchanged copies of the separately pinned local snapshot. Their hashes are checked before and after user execution. `prepared_files.json` pins the new harness separately.
- This uses standalone TD3+BC, not the differently initialized/reduced unified TD3 profile, and imports no CORL code.

## What is retained and what changes

The native networks, Flax initialization, TrainState creation with Adam epsilon `1e-5`, observation normalization `(obs - mean)/(std + 1e-3)`, head-mean critic loss, action-mean BC loss, first critic actor Q, and normalizer `c = stop_gradient(alpha/(mean(abs(Q1)) + 1e-7))` remain unchanged. Native training samples one batch, updates critics twice with renewed target noise, updates the actor once, then updates both targets once.

`native_loader.py` compiles the exact computational AST definitions from the hash-checked native file. It explicitly excludes upstream CLI, logging, D4RL/Gym data ingestion and evaluation. It does not stub or replace Distrax. Initializer setup mirrors the native main block with fixture shapes in place of Gym spaces.

The new candidate exposes one critic update per call while storing the pending batch, phase and RNG. Its second call completes the actor and target updates. Only the actor objective is extended:

`L_actor = -c mean(Q1) + mean(BC) + c * kappa_penalty * mean(U)`.

At `kappa_penalty=0` the uncertainty path is bypassed, preserving the native expression. The uncertainty reference is frozen; action derivatives remain live. In this first gate, U is an explicitly synthetic differentiable fixture. A learned BCA width network, residual units, calibration, refresh schedule, and scale-selection protocol are **not yet integrated or validated**.

The existing production runner also owns batch sampling. Satisfying its one-critic counter check alone does not preserve UniFloral. See `DESIGN.md`: a versioned integration must retain native sampling and pending state, or use a native-loop runner with explicit critic counts. This suite changes neither existing runner nor pilot.

## Fixed tiny protocol

- Native Args defaults, including gamma 0.99, alpha 2.5, 2x256 networks, and two critics per outer step. Batch size is explicitly reduced from 256 to 8 for the fixture. Resolved arguments and fixture fingerprints are saved.
- 32 deterministic synthetic rows, 14 observation coordinates and 4 action coordinates, including terminal rows. These are synthetic coordinates, not a ThrusterChain dataset.
- Six native outer steps: 12 critic + 6 actor updates.
- Six candidate kappa-zero outer steps: 12 critic + 6 actor updates.
- Two positive-penalty outer steps: 4 critic + 2 actor updates.
- Resume from a saved mid-outer-step state and replay three calls: 3 critic + 2 actor updates.
- Total maximum: **31 critic + 16 actor optimizer updates**, zero environment rollouts, zero real dataset generation. Additional loss/gradient checks have no optimizer updates.
- Floating parity tolerance is fixed prospectively at `rtol=2e-5, atol=2e-6`; RNG and integer counters must match exactly. This does not claim bitwise floating equality and does not alter an upstream test tolerance.
- No final evaluation labels, historical audit labels, or training checkpoints are used to choose the synthetic fixture or coefficient.
- Future ThrusterChain gamma 0.98, horizon 100, finite-horizon terminal masks, and discounted frozen evaluation are task adaptations that must be separately declared and validated. They are not activated here.

## Runtime compatibility and explicit limitations

Upstream requires JAX `<0.6` and NumPy `<2.0`; its Dockerfile uses Python 3.10. Titan previously reported Python 3.12.3, JAX 0.6.2 and NumPy 2.5.4. The explicit flag below permits **provisional computational checks outside those requirements**; it does not make the environment officially supported. All installed versions and constraint checks are recorded. No packages are installed.

The native script assigns `XLA_FLAGS=--xla_gpu_triton_gemm_any=True` after imports. The harness declares the same flag **before** JAX import in a fresh subprocess, avoiding ambiguous backend initialization. This timing change is an explicit harness adaptation. Conflicting existing XLA flags cause a stop. Unsupported native flags, missing/incompatible dependencies, non-GPU selection or failed precision checks also stop with no fallback. All native and candidate checks use `JAX_DEFAULT_MATMUL_PRECISION=highest` and no preallocation.

The supervisor first permits at most 180 seconds for a tiny dependency/device probe, then at most 600 seconds for the parity stage. These are failure bounds, not throughput estimates. Logs and subprocess exit statuses are saved even if the worker aborts. No stage starts after a failed stage.

## One user-run PowerShell command

```powershell
wsl -d Ubuntu -- /home/dbayha/bca-gpu/bin/python -B -u "/mnt/c/Users/David Bayha/Documents/Codex/2026-10-22/task/unifloral_td3bc_validation_v1/run_checks.py" --allow-outside-upstream-constraints --run-parity
```

The command creates a new `results_<UTC>/` inside this folder; it cannot overwrite an earlier results directory. Read `report.json`, `probe.json`, `parity.json`, and `probe.log`/`parity.log`. A failing import, unsupported XLA flag, timeout or failed parity is the precise next blocker; do not install or change settings without reviewing it. `--run-parity` can be omitted for a zero-optimizer probe only.

Even a complete pass leaves `scientific_validation_status=incomplete_design_review_required`. It establishes only the tested computational parity and actor hook behavior. Full BCA units/calibration, coherent scaling and normalization design, production runner/resume, task-specific data/evaluation, and multi-seed efficacy remain future work.

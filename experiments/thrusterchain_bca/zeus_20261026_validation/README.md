> Current handoff (2026-10-27): see [Zeus retry and preserved experiment record](HANDOFF_20261027.md). The interrupted support-shift attempt is archived by copy; the separate retry requires a process-absence check and has not started. The following notes are chronological records, not the current run status.

# Zeus validation receipt — 2026-10-26

File and metadata preflight passed. The scientific audit was **not started**: the accessible repository environment reports a CPU backend and fails several frozen package requirements. WSL discovery was denied, so availability of another compatible GPU environment remains unknown. Full training remains paused.

## Verified inputs

- Branch: `codex/thrusterchain-bca-handoff-20261026`.
- Checkout: `448cce46c922862cd9177f28e982ac0898bb38ca`; parent source commit: `b84f30c61ada17d20f2a274f3554673efe18bbc8`.
- Corrected `launch.py --preflight` exited 0, returned `preflight_pass` and `scientific_execution: false`, and verified all 360 declared files. Raw receipt: [preflight.stdout.jsonl](preflight.stdout.jsonl). Stderr was empty.
- Runtime ZIP: 32,601,033 bytes; all six members verified against manifest sizes and SHA-256 values, independently of the restored-file preflight. Receipt: [archive_integrity.json](archive_integrity.json).
- Handoff manifest SHA-256: `28df1b6bbffacda0a6659a203bf9aab2a2e82a259d3621642cdae4ad44c73adf`.
- Runtime ZIP SHA-256: `5c2b44d3ea9b52b49bcd1251a3aa733e399fe91e9e2696f282f6a3f5ab3d3d78`.
- Frozen audit protocol SHA-256: `b30d1c5cb88004135070f12050bb97b3b1059f436017f25ca005bf52087f3aa2`.
- Checkpoint state SHA-256: `874d65b7f4dcb5ec285a0d4360439c73a787b5e8c567ff7c0138e0c1e22948e8`.
- Checkpoint metadata SHA-256: `b2a4b47570974319efbc7548083bda1f3362ce0c325eb8c5443bc9a71c58329b`.
- Checkpoint commit receipt SHA-256: `686b034f2793c2a6e70be5dc72d1595f22b22977916d41bee634f18843114570`.
- Dataset SHA-256: `4c857aaa0373bee4c74db33036303b4cf75c703b289a459525f75090741f3226`.
- Original UniFloral revision: `f2dc1278eae18ed3e22c92255119c54885414d17`; 40 native snapshot files verified.
- Metadata retains 20,000 critic, 10,000 actor and 20,000 auxiliary updates, final reference refresh at critic 20,000, and a ready reference. This preflight verifies persisted identities and metadata; runtime state restoration and forward replay remain unexecuted.
- The two existing development banks retain 64 states each, nine candidates per state, and the same frozen state identity. No confirmation input was read.

## Runtime discovery

NVIDIA-SMI reported NVIDIA GeForce RTX 5070 Ti Laptop GPU, driver 610.62, 12,227 MiB. Its compute-process table reported no running processes, but this does not establish the absence of every audit process. CIM process inspection was denied. The operator suite contained no `results_*` attempt directory when inspected; its launcher has no global execution lock.

The default PATH Python is the Hermes Python 3.11.15 environment and has no JAX/JAXlib or numerical audit dependencies. The registered Store Python 3.12 could not launch. Neither environment was modified.

The repository `.venv` runs Python 3.10.11 and reports backend `cpu`, device `TFRT_CPU_0`, matmul precision `None`, x64 false, JIT enabled. No model or simulator was loaded by the JAX device probe. Full receipt: [native_environment.json](native_environment.json).

| Package | Repository `.venv` | Frozen requirement |
| --- | --- | --- |
| JAX / jaxlib | 0.6.2 / 0.6.2 | 0.6.2 / 0.6.2 |
| jax-cuda12-plugin / jax-cuda12-pjrt | absent / absent | 0.6.2 / 0.6.2 |
| NumPy | 2.2.6 | 2.5.4 |
| Flax | 0.10.7 | 0.10.7 |
| Optax | 0.2.8 | 0.2.8 |
| Distrax | absent | 0.1.5 |
| Chex | 0.1.90 | 0.1.90 |
| TensorFlow Probability | 0.25.0 | 0.25.0 |
| SciPy | 1.15.3 | 1.18.1 |
| Gymnasium | absent | 1.0.0 in tested evaluator dependencies; not equality-gated |

The original runtime gate also requires GPU backend, highest matmul precision, x64 disabled, JIT enabled and `XLA_FLAGS=--xla_gpu_triton_gemm_any=True`. The supervisor configures the declared values, but rejects conflicting XLA flags. Titan recorded Python 3.12.3; Python version is recorded rather than equality-gated.

This is the preserved provisional runtime outside upstream constraints. Upstream requirements specify JAX <0.6 and NumPy <2.0, so installing them wholesale would conflict with this frozen audit. No package installation, driver change or security change was attempted.

## Blockers and next prerequisite

1. **Denied:** `wsl --list --verbose` returned `Wsl/E_ACCESSDENIED` / `Access is denied`. No retry or alternate WSL route was attempted. The compatible GPU-environment search is incomplete.
2. **Denied:** `Get-CimInstance Win32_Process` returned `Access to a CIM resource was not available to the client`. No retry or alternate process-inspection route was attempted. Global no-overlap verification remains incomplete.
3. **Observed incompatible:** the accessible repository `.venv` is CPU-only and has the missing/different packages above. The audit worker was not launched to rediscover this failure.

The minimal prerequisite is access to an existing CUDA-capable environment matching the frozen gate, plus a reliable check for another active/completed audit. If none exists, prepare a separately authorized isolated environment with the exact packages above; first verify package availability and compatibility. Do not alter the frozen runtime requirements or modify the current environments implicitly. No usable WSL interpreter path has been established on Zeus.

## Reproduction

From `C:\Users\David\dev\BCA` the completed read-only preflight is reproducible with the current default Python:

```powershell
python -B -u experiments/thrusterchain_bca/titan_20261026/launch.py --preflight
```

After the GPU environment and no-overlap prerequisite are resolved, enter the handoff directory in that verified environment and invoke its Python once:

```text
python -B -u launch.py --execute --allow-outside-upstream-constraints
```

Do not substitute a Titan interpreter path, use the incompatible Windows `.venv`, repeat a completed audit, or treat the passing file preflight as a passing GPU runtime gate. Inspect any new attempt and its terminal receipt before a subsequent launch.

## Work accounting and interpretation

This validation used **zero** evaluator transitions, full rollouts, target-smoothing draws, model fits, policy/critic/uncertainty updates, calibration updates and confirmation reads. No operator-alignment metrics were produced. No GPU audit attempt or worker was created. Prior historical work is not included in these zero counts.

The pending authorized audit retains its 300-second worker cap, 1,152 one-step development transitions, 256 existing fitting transitions, 128 current-actor rows, 1,408 current-critic/radius/target-actor rows each, and 5,632 twin-target-critic input rows. Four prespecified smoothing draws apply to each bank: eight bank-level draws total. Supervisor verification is outside the worker cap.

The audit remains exploratory and descriptive, not efficacy or new confirmation. It separates native absolute twin-min smoothed-target TD residuals from current-Q1 error against deterministic current-policy continuation values. Operator differences are not an established cause of failed decision utility. The previous inconclusive confirmation is unchanged.

Only this validation-receipt directory was added. Frozen payload, checkpoints, normalization and method were preserved; no commit or push was made. Pre-existing untracked `.claude/worktrees/` and `bca/` were left untouched.

## Manual environment discovery helper

The user subsequently reported Ubuntu-22.04 running under WSL 2, with `/usr/bin/python3` version 3.10.12 and the RTX 5070 Ti Laptop GPU visible from their terminal. This does not resolve the assistant session's WSL access denial; no WSL command was retried here.

`discover_python_envs.py` is a read-only helper for the user to run manually. It checks the running/PATH Python, active venv/conda paths, explicitly named project venvs, `~/bca-gpu`, common conda roots, and one level beneath named home environment directories. It does not recursively scan home. Discovery is capped at 16 interpreters and 12 entries per named directory, with a 60-second probe budget. Missing/unreadable locations and timeouts do not establish global absence. Distinct virtualenv invocation paths are preserved even if they share a Python symlink target.

The helper reads package metadata first. Only an exact frozen package match with Gymnasium present permits a bounded JAX device probe; it loads no experiment model or evaluator. Probe configuration is confined to child processes, and conflicting inherited XLA flags are reported. It writes no files, installs nothing, and never starts WSL or the audit. Local validation covered Python syntax and the CLI; Linux discovery remains pending manual execution.

Run this exact command in PowerShell:

```powershell
wsl -d Ubuntu-22.04 -- python3 -B /mnt/c/Users/David/dev/BCA/experiments/thrusterchain_bca/zeus_20261026_validation/discover_python_envs.py
```

Output is concise text for review. A candidate passing package/device checks still needs a no-overlap check and the frozen audit's own integrity/replay gates. Add `--python /absolute/path/to/python` only for another known interpreter; the helper does not install or automatically select an environment.

Later manual discovery in the separate Ubuntu distro found corlverify's Python 3.12.14 and an incompatible older sdbca environment. The user's dry run then resolved 58 packages; [RESOLUTION_REVIEW.json](RESOLUTION_REVIEW.json) records the passing review. See [SETUP_PLAN.md](SETUP_PLAN.md) for official package/Python compatibility evidence and the prepared one-command installer targeting a new isolated venv. Environment creation and package installation remain pending explicit authorization; the installer has run only in its non-mutating plan mode.

The user explicitly reaffirmed that the model must remain UniFloral from Titan. [MODEL_IDENTITY.md](MODEL_IDENTITY.md) records the passing repeated file preflight, exact native/checkpoint/normalization hashes, and the required live module-origin verification after runtime setup. The corlverify environment name selects no CORL model or code.

The user's subsequent isolated installation and readiness probe succeeded. The saved readiness receipt confirms the GPU runtime and actual UniFloral definition origins with zero forwards/transitions. [RUN_ONCE.md](RUN_ONCE.md) provides the exclusive one-attempt wrapper command and preserved-evidence rules. The scientific audit has not been launched by this task.

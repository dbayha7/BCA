# Isolated runtime preparation — no installation performed

**Later manual status:** the user reported successful isolated installation at 2026-10-26 22:03 UTC with status `installed_pending_gpu_and_model_identity_checks` and `audit_started: false`. The assistant did not execute WSL or the installer. The Linux setup receipt has not been read from this Windows session; the readiness helper below reads and validates it when run manually in Ubuntu. The reported systemd user-session warning did not prevent the successful installer receipt, and no systemd/security change is proposed. This update supersedes the pending-installation wording in the preserved preparation plan below.

Next manual PowerShell command, **no evaluator transitions or model forwards**:

```powershell
wsl -d Ubuntu -- timeout 60s /home/david/bca-operator-audit-20261026/bin/python -B /mnt/c/Users/David/dev/BCA/experiments/thrusterchain_bca/zeus_20261026_validation/probe_runtime_identity.py
```

The probe reuses the exact frozen runtime gate and native loader, verifies source/checkpoint files, validates the isolated prefix and setup receipt, and checks native module/class/function origins. It loads definitions only. Existing audit result directories or matching audit processes block it. Its process snapshot is limited to this user/current Ubuntu namespace; the eventual one-time audit also needs an exclusive shared attempt reservation. Do not infer an audit completed from readiness success. Timeout or a failed probe requires inspection before further work.

Use the existing `/home/david/miniconda3/envs/corlverify/bin/python` in the **Ubuntu** distro. The user reported CPython **3.12.14** there. It supplies a suitable interpreter for resolution and a later new venv; its packages must not be changed. No new Python installation, conda environment, conda channel, repository or terms acceptance is needed for this route.

The RL implementation and model must be the exact pinned UniFloral source and Titan checkpoint; `corlverify` supplies only CPython. See [MODEL_IDENTITY.md](MODEL_IDENTITY.md) and [unifloral_identity.json](unifloral_identity.json) for the renewed source/checkpoint/normalization checks and the live module-origin check required before audit execution. No CORL code, checkpoint or configuration substitution is permitted.

The user also found `sdbca` Python 3.10.20 with much of the JAX stack, but its NumPy/SciPy are older and Gymnasium is absent. Preserve that environment. Ubuntu-22.04's system Python 3.10.12 is likewise unsuitable for the exact frozen package set.

## Official registry evidence, checked 2026-10-26

| Exact release | Requires-Python | Relevant wheel evidence |
| --- | --- | --- |
| [NumPy 2.5.4](https://pypi.org/project/numpy/2.5.4/) | `>=3.12` | `numpy-2.5.4-cp312-cp312-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl`, 16.7 MB |
| [SciPy 1.18.1](https://pypi.org/project/scipy/1.18.1/) | `>=3.12` | `scipy-1.18.1-cp312-cp312-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl`, 35.3 MB |
| [JAX 0.6.2](https://pypi.org/pypi/jax/0.6.2/json) | `>=3.10` | Universal Python wheel |
| [jaxlib 0.6.2](https://pypi.org/pypi/jaxlib/0.6.2/json) | `>=3.10` | CPython 3.12 manylinux2014 x86_64 wheel |
| [jax-cuda12-plugin 0.6.2](https://pypi.org/pypi/jax-cuda12-plugin/0.6.2/json) | `>=3.10` | CPython 3.12 manylinux2014 x86_64 wheel |
| [jax-cuda12-pjrt 0.6.2](https://pypi.org/pypi/jax-cuda12-pjrt/0.6.2/json) | Unspecified | `py3-none-manylinux2014_x86_64` wheel |
| [Flax 0.10.7](https://pypi.org/pypi/flax/0.10.7/json) | `>=3.10` | Universal wheel |
| [Optax 0.2.8](https://pypi.org/pypi/optax/0.2.8/json) | `>=3.10` | Universal wheel |
| [Distrax 0.1.5](https://pypi.org/pypi/distrax/0.1.5/json) | `>=3.9` | Universal wheel |
| [Chex 0.1.90](https://pypi.org/pypi/chex/0.1.90/json) | `>=3.9` | Universal wheel |
| [TensorFlow Probability 0.25.0](https://pypi.org/pypi/tensorflow-probability/0.25.0/json) | `>=3.9` | Universal wheel |
| [Gymnasium 1.0.0](https://pypi.org/pypi/gymnasium/1.0.0/json) | `>=3.8` | Universal wheel |

Python 3.10 is blocked by the NumPy and SciPy declared requirements, not merely by missing wheels. Their versions must not be reduced to make Python 3.10 work. Python 3.12.14 differs from Titan's recorded 3.12.3 patch version, but the frozen runtime gate records Python rather than requiring patch equality. Preserve and report the actual interpreter.

The manual resolution completed successfully at `resolution_20261026T215529_236287Z`. Review verified all 58 official hashed wheels, all 12 requested pins and 106 active dependency requirements. All entries accept Python 3.12.14, with no prerelease/nightly/yanked packages or source builds. The chosen Orbax 0.12.8 accepts JAX 0.6.2. Installation and runtime compatibility remain untested. See [RESOLUTION_REVIEW.json](RESOLUTION_REVIEW.json).

`runtime_requirements.txt` contains all eleven frozen gate versions plus Gymnasium 1.0.0. `jax[cuda12]==0.6.2` selects the official `jax-cuda12-plugin[with-cuda]==0.6.2` extra, which requests NVIDIA CUDA libraries as Python wheels. Merely installing the plugin without that extra would not request those libraries. No system CUDA toolkit or driver changes are proposed. The transitive wheel versions are not specified by the frozen gate, so the resolver records and hashes the entire selected set for review and later reproduction; this does not assert byte identity with Titan's transitive runtime.

## Completed command: resolution only

Run manually in PowerShell:

```powershell
wsl -d Ubuntu -- /home/david/miniconda3/envs/corlverify/bin/python -B /mnt/c/Users/David/dev/BCA/experiments/thrusterchain_bca/zeus_20261026_validation/resolve_runtime.py --dry-run
```

This creates a new `resolution_<UTC>/` evidence directory beside the helper. It does **not** create a venv, install packages, update pip, invoke conda, import JAX or launch the audit. It uses the existing interpreter's pip in documented `--dry-run --ignore-installed` mode, wheel-only, no persistent pip cache, with only official PyPI selected. Configuration is isolated to the subprocess. [Pip dry-run documentation](https://pip.pypa.io/en/stable/cli/pip_install/), [pip configuration documentation](https://pip.pypa.io/en/stable/topics/configuration/).

Resolution may download metadata or wheels. Missing pip/pip older than 22.2, missing versions/wheels, dependency conflicts or a 300-second resolver timeout cause a stop; the script does not install an alternate pip, substitute versions or retry automatically. Its resolver timeout is separate from the scientific worker's 300-second cap.

On success, `resolution.json` says `resolved_for_review`; `pip-report.json` records the full dependency selection and `resolved-requirements.txt` pins every wheel with SHA-256. All frozen versions, official wheel URLs, required CUDA dependencies and hashes are checked before producing the lock. On failure, retain `resolver.log`; no environment has been modified. A failed resolver does not establish that all possible isolated setups are impossible.

Local tests covered syntax, CLI, and synthetic lock validation. The user executed the dry run manually and its saved receipt was reviewed through the authorized Windows workspace. The assistant's WSL denial remains in force; it was not retried.

## Proposed installation, pending explicit authorization

The reviewed one-command installer is [install_reviewed_runtime.py](install_reviewed_runtime.py). It binds the exact resolution receipt/report/lock hashes, re-verifies all 360 frozen Titan files, atomically reserves the new target, installs the 58 hash-pinned wheels only into that venv, checks dependency consistency and exact package versions, then stops. It rejects an existing target, including failed attempts, and preserves failure receipts. Its default mode only prints the verified plan; that mode has passed locally. GPU and loaded-model-path checks are separate and remain required before an audit.

Exact PowerShell command **only after setup approval**:

```powershell
wsl -d Ubuntu -- /home/david/miniconda3/envs/corlverify/bin/python -B /mnt/c/Users/David/dev/BCA/experiments/thrusterchain_bca/zeus_20261026_validation/install_reviewed_runtime.py --install-reviewed
```

Reviewed lock SHA-256: `017b189f4786193db03c32fd7b3d691ae320c1274eff11e1e0c3766764655d14`.

Target a **new** directory: `/home/david/bca-operator-audit-20261026` in Ubuntu. Refuse if that path already exists, including a dangling symlink. Use corlverify's Python 3.12.14 to create a venv without system-site packages, then use only the new venv's Python/pip to install the reviewed hashed wheel lock. The standard venv operation bootstraps pip from the interpreter's bundled ensurepip; no pip upgrade or conda command is needed. If ensurepip/venv is unavailable, stop and report it instead of modifying corlverify. [Python venv documentation](https://docs.python.org/3.12/library/venv.html).

The reviewable command sequence, **not executed**, is:

```bash
task_env=/home/david/bca-operator-audit-20261026
# Set task_lock to the reviewed resolution directory's resolved-requirements.txt.
test -f "$task_lock" || exit 1
if test -e "$task_env" || test -L "$task_env"; then
  echo "Refusing an existing environment path" >&2
  exit 1
fi
/home/david/miniconda3/envs/corlverify/bin/python -B -m venv "$task_env" || exit 1
env -u PIP_EXTRA_INDEX_URL -u PIP_INDEX_URL PIP_CONFIG_FILE=/dev/null \
  "$task_env/bin/python" -B -m pip --isolated --disable-pip-version-check install \
  --require-hashes --only-binary=:all: --no-cache-dir --no-input \
  --index-url https://pypi.org/simple -r "$task_lock" || exit 1
"$task_env/bin/python" -B -m pip check
```

The command sequence above explains the operations; prefer the prepared one-command installer, which binds the reviewed lock and refuses an existing target. Preserve the underlying corlverify interpreter because the new venv depends on it. No source build, source-package fallback, driver change, shell-profile change or existing-environment upgrade is included. A later package/device probe must still pass before any audit launch, and a reliable process/attempt check remains required.

Network/disk: official registry and `files.pythonhosted.org` access is needed. CUDA wheels can require several GB of downloads and expanded storage. Exact byte totals remain unknown: the saved pip report contains hashes/URLs without sizes, and the resolver log records metadata downloads. Reserving 10 GB free is a conservative planning allowance, not a measured requirement. The installer records available disk space before/after installation; persistent pip cache reuse is disabled.

Suggested approval wording **after successful resolution review**: “Approve creating `/home/david/bca-operator-audit-20261026` using the existing Python 3.12.14 and installing the reviewed hashed PyPI/CUDA runtime there, allowing several GB of downloads? This setup leaves existing environments intact and starts no training or audit.”

Approval is needed because the current instruction explicitly allows preparation only and withholds environment creation/package installation. No automatic-approval rejection caused this pause.

For closer transitive matching to Titan, obtain a reviewed package inventory or wheel lock from its existing validated environment, then compare it to the resolution. No Titan access, export or transfer was performed or assumed. The direct frozen package versions, checkpoint and method remain unchanged in either approach.

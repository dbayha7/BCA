# Path A operator runbook

**Optional Colab route (prepared November 2, 2026):** use [Path_A_Colab.ipynb](../path_a_colab_v1/Path_A_Colab.ipynb) and its [validation report](../path_a_colab_v1/VALIDATION.md). It contains the complete code, requires the existing verified ZIP, and remains untested on an actual Colab GPU. Its manual migration declaration must establish that neither Titan validation nor recovery started. Unknown or consumed target work blocks it. Keep Zeus held; select only one target. All Colab setup/wait time consumes the existing allowance. The commands below remain the separate Titan route and must not be run concurrently with Colab.

Use the [current checklist](README.md) and the existing approved Titan profile. This document adds no checks or workload to the sealed launcher. If an attempt already exists, inspect its receipts; do not rerun or remove it. The operator must ensure the same recovery is not running in another clone. Keep Zeus's migration hold.

## 1. Transfer and verify

On Zeus, the files are in `C:\Users\David\AppData\Local\Temp\BCA_PathA_Titan_20261102`:

- `path_a_titan_recovery.zip` — 112,349,548 bytes; 671 payload files.
- `unpack_path_a.py` — 3,775 bytes.
- `HANDOFF_RECEIPT.json` and `RUN_ON_TITAN.md` contain the recorded package verification and instructions.

Manually copy the ZIP and unpacker to `C:\Users\David Bayha\Downloads\BCA_PathA_Titan_20261102` on Titan. No upload or transfer has been established by this task. The isolated destination is `C:\Users\David Bayha\BCA-PathA`, leaving the existing BCA checkout intact. No installation is required if the user-reported runtime remains present.

Verify the helper in Titan PowerShell:

```powershell
$handoffDir = 'C:\Users\David Bayha\Downloads\BCA_PathA_Titan_20261102'
if ((Get-FileHash -LiteralPath "$handoffDir\unpack_path_a.py" -Algorithm SHA256).Hash.ToLowerInvariant() -ne '651905b0c22bfa9361b5e87d05d47735d276478ecb86cea902175401eae21492') { throw 'Unpacker hash mismatch; stop.' }
```

Unpack; the existing helper verifies every member and refuses conflicting files:

```powershell
wsl -d Ubuntu -- /home/dbayha/bca-gpu/bin/python -B '/mnt/c/Users/David Bayha/Downloads/BCA_PathA_Titan_20261102/unpack_path_a.py' --archive '/mnt/c/Users/David Bayha/Downloads/BCA_PathA_Titan_20261102/path_a_titan_recovery.zip' --sha256 'c7f492c7e3d46a8508f8f7df30a975e216230e9e7e6289e2edb53f4793ef64fc' --destination '/mnt/c/Users/David Bayha/BCA-PathA'
```

Run the prepared source-only check:

```powershell
wsl -d Ubuntu -- /home/dbayha/bca-gpu/bin/python -B '/mnt/c/Users/David Bayha/BCA-PathA/experiments/thrusterchain_bca/two_scenario_path_a/titan_launch.py' --check
```

Expected source-check status: `source_checked_titan_not_executed`. This is not a GPU pass. Inspect any reported existing validation/production attempt before proceeding.

## 2. One bounded target compatibility check

```powershell
wsl -d Ubuntu -- bash -lc "cd '/mnt/c/Users/David Bayha/BCA-PathA' && timeout --signal=TERM --kill-after=5s 115s /home/dbayha/bca-gpu/bin/python -B -u experiments/thrusterchain_bca/two_scenario_path_a/titan_launch.py --validate"
```

It verifies exact source/data and runtime settings, initializes one model template, restores the original full warmup, checks eight-row actor/Q1/width inference and 102 controller calls / 294 exact action comparisons. It performs **zero optimizer updates, calibrations, resets or simulator steps**. The ceiling is 120 seconds including kill grace. The prior real-update smoke remains Zeus evidence; this target check does not claim bitwise equivalence of every future GPU update.

Required receipt: `titan_validation_1/RESULT.json` with `passed_titan_validation`, matching manifest/approval and unchanged identities. Failure is a technical blocker, not a BCA result. Preserve it and report it; do not adjust tolerances or retry.

## 3. Conditional recovery, once

Only after that authenticated pass:

```powershell
wsl -d Ubuntu -- bash -lc "cd '/mnt/c/Users/David Bayha/BCA-PathA' && timeout --signal=TERM --kill-after=5s 10475s /home/dbayha/bca-gpu/bin/python -B -u experiments/thrusterchain_bca/two_scenario_path_a/titan_launch.py --execute"
```

Keep the foreground terminal open. Existing exclusive output paths prevent the standard command from silently repeating an attempt. No cross-machine distributed lock is claimed. The 10,480-second recovery ceiling includes kill grace and the unchanged evaluation.

## Read-only progress and completion evidence

Under `C:\Users\David Bayha\BCA-PathA\experiments\thrusterchain_bca\two_scenario_path_a`:

- `titan_validation_1/RESULT.json`: target compatibility outcome.
- `run_titan_recovery_1/ATTEMPT.json`: selected host, source lineage, start, caps and validation proof.
- `run_titan_recovery_1/<job>/progress.jsonl`, `report.json`, `SUPERVISOR.json`: job work and acceptance.
- `run_titan_recovery_1/RESULT.json`: whole recovery terminal outcome and accounting.
- `run_titan_recovery_1/evaluation/report.json`: final scientific comparisons.

Do not infer whole-study completion from a single job exiting 0. Require `completed_titan_path_a_recovery`, accepted remaining jobs, authenticated inherited warmup, and complete evaluation denominators. Then perform the already planned saved-result analysis in the checklist. Do not tune or discard weak seeds.

Titan manifest SHA256: `32718856c6b4714ea1a4004554d61b80223d502e3517b5563c372a97047a15ed`.

Titan approval SHA256: `4fc24a5bb025d54372cdd14736cd24894b4053516e4393130bb4e8712dd3a156`.

Assistant WSL execution remains denied. These are manual operator commands, not actions executed during this documentation task.

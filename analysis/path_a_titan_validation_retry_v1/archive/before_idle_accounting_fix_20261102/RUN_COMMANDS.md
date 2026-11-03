# Manual Titan commands — conditional, not executed here

First follow README.md to resolve `path_a_a1.py`, verify the delivered ZIP hash, and install into the existing isolated Titan tree. Do not install/run if that helper was used or another Path A attempt started. Do not delete or move the first validation.

The following uses the already observed Titan interpreter. No installation, environment change, fresh warmup, simulation or training occurs in preparation or validation. Keep the same WSL instance alive between preparation, validation and any production launch; an unknown/rebooted timing lineage refuses execution.

```powershell
$runner = '/mnt/c/Users/David Bayha/BCA-PathA/analysis/path_a_titan_validation_retry_v1/titan_retry.py'
# Enter a conservative upper bound for additional setup/transfer time not recorded
# by the installer. It is a charged upper bound, not a claimed measurement.
# If the bound cannot be established, stop and reconcile the ledger.
$unrecordedUpper = Read-Host 'Upper bound in seconds for unrecorded new setup time'
wsl -d Ubuntu -- /home/dbayha/bca-gpu/bin/python -B -u $runner --prepare --unrecorded-setup-upper-seconds $unrecordedUpper --confirm-no-other-attempts --confirm-helper-absent-not-used
if ($LASTEXITCODE -ne 0) { throw 'Preparation failed; preserve evidence and do not retry.' }
wsl -d Ubuntu -- timeout --signal=TERM --kill-after=5s 115s /home/dbayha/bca-gpu/bin/python -B -u $runner --validate
```

Stop after validation and inspect the terminal receipt. A process exit alone is not a compatibility pass. A missing/failed receipt blocks production and must be diagnosed without rerunning the reservation. The runtime authenticates the original source/data/numerical proof, new approval, same target and complete attempt-2 receipt; neither the old incomplete attempt nor CPU tests suffice.

Only after the new authenticated pass, use this conditional production command. It computes the current remaining budget and immediately starts GNU timeout around the production supervisor; no stale manually copied duration is used. It reuses the completed warmup and runs only the unchanged remaining 17 jobs and evaluation.

```powershell
wsl -d Ubuntu -- /home/dbayha/bca-gpu/bin/python -B -u $runner --execute
```

The user has already approved conditional continuation; no new scientific approval is needed if the original checks pass and scope remains unchanged. The new output directory is `experiments/thrusterchain_bca/two_scenario_path_a/run_titan_recovery_validation2`. The validation directory is `titan_validation_2`, including separate `stdout.log`, `stderr.log`, `PROCESS.json`, `RESULT.json` and nested unchanged worker evidence. No old reservation is reset.

# Same Path A recovery on Titan

This is a transport/runtime amendment, not a new study. Preserve the completed Honeypot seed-0 warmup, original passed Zeus smoke, stopped production attempt, fixed data and seeds, native versus original actor `Q1-3U`, and unchanged BC. Execute only the 17 remaining training jobs and unchanged evaluation. No smoke or completed warmup is rerun.

## Destination and transport

Use the isolated directory `C:\Users\David Bayha\BCA-PathA`, corresponding to `/mnt/c/Users/David Bayha/BCA-PathA`. This avoids overwriting the existing Titan checkout or other experiments. The complete, hash-pinned repository-relative file closure is included; no Git checkout, installation or historical minimal-comparison launcher is needed.

The runtime reported by the user is `/home/dbayha/bca-gpu/bin/python`, Python 3.12.3. Package metadata matches the frozen JAX 0.6.2 stack, NumPy 2.5.4, SciPy 1.18.1 and Gymnasium 1.0.0. This is not proof that GPU restoration/inference passes. Use only the manual validation below; do not borrow the unrelated screen's `OPENBLAS_CORETYPE=Haswell` setting or relax numerical checks.

Manually copy `path_a_titan_recovery.zip` and `unpack_path_a.py` from the delivered transfer folder to `C:\Users\David Bayha\Downloads\BCA_PathA_Titan_20261102`. Choose your own USB or trusted transfer route. Nothing has been uploaded or published. Verify the two SHA256 values given in the external handoff receipt. The unpacker verifies the archive and every member before writing, rejects path traversal/symlinks and conflicting existing files, and copies only absent files. It never overwrites a file. An identical previous extraction is harmless and performs no scientific work.

## Preserved newline evidence

One original frozen approval was stored with CRLF while Git stores LF. Its 517-byte current checkout SHA is `87a9ab59e4872411ab483cb1d8dee0bdc7d6e600e7921c37fb73aa7e6e258f7b`. Reconstructing only its eight line endings gives the exact 525-byte frozen original SHA `35fac7db435090e49c7bdc2da2586274736d8191848d2d8ab641f4a23262814b`.

Only the isolated staging/archive contains those authenticated CRLF bytes. JSON content and approval semantics are identical. The live Zeus file is unchanged, no historical manifest is resealed, and all runtime pins remain strict. Binary extraction preserves the frozen bytes; do not run Git normalization over the extracted directory.

## Single selected machine and budget

Zeus's `run_recovery_1/HANDOFF_HOLD.json` is an explicit non-running hold. It occupies the original launcher's exclusive output path before any checkpoint copy or model initialization, blocking the standard Zeus recovery command. The archive includes its provenance as `TITAN_SOURCE_HOLD.json`, but does not copy the source hold directory. Keep the hold in place. Do not run another clone. This is an operational handoff, not a distributed lock.

The Titan profile refuses execution outside the exact destination/interpreter and records its actual host/device identity. It never interprets Zeus PIDs 435/1693 as Titan processes. Exclusive Titan validation and recovery directories block repeated attempts; failed validation cannot authorize training. Each job authenticates the same successful Titan validation and original smoke proof.

The original 10,660-second remaining allowance is split into a 60-second conservative preparation/failed-source-check-overhead charge, 120 seconds for one compatibility check, and **10,480 seconds (2h 54m 40s)** for scientific recovery. The failed Zeus source-check's elapsed time was not measured; the 60 seconds is a reserved charge, not an invented timing observation. No four-hour allowance is reset. Validation's external timeout is 115 seconds plus 5 seconds kill grace; scientific recovery uses 10,475 seconds plus 5 seconds grace.

No extra optimizer, calibration or simulator work is added by compatibility validation. Cumulative physical caps remain 190,014 critic / 95,008 actor / 190,014 auxiliary / 22 calibrations; the old interrupted tail stays unknown within its bounds. Maximum cumulative simulator work remains 430,575 of 498,400, including 237,600 reserved evaluation steps and 1,536 return episodes. Path B remains excluded.

## Zero-update target validation

The manual check authenticates source/data/approvals/evidence; verifies the frozen GPU/JIT/float32/matmul settings and package versions; checks both conditions' exact retained metadata; initializes one Honeypot seed-0 template; and restores the original 10k warmup including normalization, optimizers, targets, RNG, calibration and counters. It then checks a small finite actor/Q1/width inference batch and compares the controller's deployed actions with existing saved smoke states. It calls no optimizer step, calibration fit, dynamics, reset or rollout. Controller comparisons are bounded and never select replacement cases. A mismatch is recorded and blocks the recovery; no automatic retry or tolerance change is allowed.

This verifies the requested interfaces on Titan, not bitwise equality of every future GPU update or a BCA efficacy result. The original calibration and learning smoke remain Zeus evidence and are not mislabeled Titan measurements.

After verified unpacking, source-only validation is:

```powershell
wsl -d Ubuntu -- /home/dbayha/bca-gpu/bin/python -B '/mnt/c/Users/David Bayha/BCA-PathA/experiments/thrusterchain_bca/two_scenario_path_a/titan_launch.py' --check
```

Run the target compatibility check once:

```powershell
wsl -d Ubuntu -- bash -lc "cd '/mnt/c/Users/David Bayha/BCA-PathA' && timeout --signal=TERM --kill-after=5s 115s /home/dbayha/bca-gpu/bin/python -B -u experiments/thrusterchain_bca/two_scenario_path_a/titan_launch.py --validate"
```

Only after its authenticated terminal receipt reports `passed_titan_validation`, run:

```powershell
wsl -d Ubuntu -- bash -lc "cd '/mnt/c/Users/David Bayha/BCA-PathA' && timeout --signal=TERM --kill-after=5s 10475s /home/dbayha/bca-gpu/bin/python -B -u experiments/thrusterchain_bca/two_scenario_path_a/titan_launch.py --execute"
```

Keep the foreground terminal open. Receipts are `titan_validation_1/RESULT.json`, `run_titan_recovery_1/RESULT.json` and `run_titan_recovery_1/evaluation/report.json`. Assistant-side WSL execution remains denied; no GPU, learning, calibration or simulator operation was executed while preparing this handoff.

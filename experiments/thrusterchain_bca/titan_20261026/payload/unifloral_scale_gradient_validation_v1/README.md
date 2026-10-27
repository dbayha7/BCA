# Fitting-only scale and gradient validation

Separate bounded diagnostic for the verified native UniFloral TD3+BC checkpoint after 20,000 critic updates. The checkpoint, baseline, thresholds and previous results are preserved.

The previous development diagnostic found that total-RMS matching did not match action contrasts. This stage asks whether contrast scales and gradient scales tell a consistent story, and tests local unit-rescaling behavior. It cannot answer whether BCA improves returns.

**Budget:** the existing 256 fitting-bank rows, at most nine candidate actions per row (2,304 rows), and 32 fixed rows for action/actor-parameter gradients. Zero optimizer updates, calibrator updates, calibration refreshes and simulator steps. No development return labels or confirmation data. A 300-second child timeout bounds GPU execution; no throughput promise is made.

```powershell
wsl -d Ubuntu -- /home/dbayha/bca-gpu/bin/python -B -u "/mnt/c/Users/David Bayha/Documents/Codex/2026-10-22/task/unifloral_scale_gradient_validation_v1/run_checks.py" --execute --allow-outside-upstream-constraints
```

The launcher sets and checks the existing GPU/highest-precision runtime before scientific imports. No packages are installed. Without `--execute`, only file/provenance preflight runs. Results use a new exclusive timestamped directory; sources and old results are never overwritten.

Read `PSEUDOCODE.md`, `PREREGISTERED_CHECKS.md` and `protocol.json` for the fixed design. `contract.py` implements only standard-library diagnostic scale rules; `test_contract.py` covers its algebra and invalid-input behavior. `checks.py` imports the actual native candidate and BCA width implementation for GPU measurements; `run_checks.py` verifies provenance, runtime and zero-work receipts.

The optional factor C_Q/C_U is a diagnostic candidate. U already has reward/Q units, so this dimensionless factor matches contrasts rather than supplying a missing unit conversion. It is detached and would multiply only the penalty inside the existing native c-weighted objective. This does not approve production normalization, choose kappa or remove the native 1e-7 epsilon. Subtracting an action-independent constant alone cannot change rankings. Candidate action contrasts and actor optimization gradients are reported separately.

Local verification uses permitted Windows standard-library tests, AST parsing and hashing only. GPU numerical checks remain pending manual execution because the earlier WSL access denial remains binding.

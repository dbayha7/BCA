# Explicit runtime validation retry, attempt 2

This wrapper runs the unchanged, sealed `../validate_runtime.py` worker. It does not enable production training, refreeze the study, change tests or loosen tolerances. The first attempt is preserved in `../runtime_validation_v1_archived_interrupted_20261028T193817Z`.

The user interrupted the first validation. No terminal report exists. The last durable snapshot records zero optimizer dispatches and simulator steps, but final work counts and elapsed time are unknown. The user checked the recorded supervisor PID 390 and worker PID 549; neither existed at that observation. This is not a global process absence claim. All four original files were hash-verified before and after moving them into the archive. Its receipt records this provenance.

## Repair and accounting

The original supervisor restored normal signal handlers before its final hash check and caught only `Exception`. A second Ctrl-C could therefore escape before `RESULT.json` was written. This separate launcher keeps record-only handlers during child creation, owned-child cleanup and receipt finalization. It saves `CHILD_OUTCOME.json` and a provisional `RESULT.json` before the interruptible final identity check. A stopped or unverified run cannot pass.

An immutable retry `ATTEMPT.json` and exclusive original worker output directory prevent duplicate use. The original worker requires that exact output directory; its new reservation explicitly says attempt 2 and points to the archive. Old source, PREPARED.json, protocol and results remain unchanged. Nothing is automatically restarted. An interruption requires review, not another call of this command.

The worker and postcheck share a 900-second cap after reservation. Cleanup may require up to 10 additional seconds to reap the owned child, plus filesystem receipt latency. Read-only identity prechecks occur before reservation. The prior attempt remains separately recorded with unknown final counts; even a successful retry cannot make cumulative execution counts exact.

Unchanged checks use generated-data optimizer fixtures (caps 64 critic, 32 actor, 64 auxiliary updates and 4 calibrations), five read-only real-source initializations (at most 4,480 repeated episode reads / 448,000 tuples), and the full Gymnasium checker (at most 512 steps / 128 resets). No real-data optimizer updates or production evaluation is authorized. This is target-runtime validation, not the expanded training study.

## Manual command

From PowerShell, use the existing isolated Ubuntu interpreter:

```powershell
wsl -d Ubuntu -- /home/david/bca-operator-audit-20261026/bin/python -B /mnt/c/Users/David/dev/BCA/experiments/thrusterchain_bca/zeus_20261026_validation/central_quartile_training_v1/validation_retry_20261028_v1/retry.py --execute-validation
```

`--preflight` instead performs hash-only checks and creates no attempt. The supervisor checks the pinned runtime, exact known BCA/validation process paths and immutable archive before reserving. Permission-denied process inspection stops execution. Assistant-side WSL access remains denied, so the assistant must not run either command through that route.

CPU fixtures test the actual CLI import, exclusive reservation/duplicate rejection, failure before worker entry, timeout cleanup, repeated signals during spawn/cleanup/postcheck, terminal receipt preservation, deadline enforcement and cumulative unknown accounting. The first local test run hit WinError 5 at atomic file replacement in the managed Windows sandbox (4 passed, 8 blocked). Windows fixtures explicitly substitute ordinary writes to test the supervisor logic; they do not establish target atomic-write behavior. Linux tests retain the real writer. Target reservation writing precedes worker creation and fails closed if unavailable. CPU fixtures do not validate CUDA, the real model or Gymnasium. Those checks remain pending the manual run. Production training stays held.

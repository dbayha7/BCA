# Conditional OOD design gate: separately recorded retry

Prepared after the first attempt's startup failure on 28 October 2026. No retry has been launched or reserved by the assistant. The first attempt, its sources, frozen manifest, reservation and receipts remain in `../conditional_ood_gate_v1/` unchanged.

## Cause and repair

The core loader constructs the canonical module dynamically from authenticated source definitions. That module has no `__file__`. The original gate incorrectly read that attribute during source authentication. Its exception handler then used `type(exc).__name()` instead of `type(exc).__name__`, which masked the original failure.

The repaired authenticator locates the original canonical file relative to the already authenticated environment adapter, verifies the dynamics function's compilation filename and global namespace, and retains the unchanged source SHA-256 and function-binding checks. It does not change the simulator, actor, critic, normalization, dataset or any scientific threshold. The exception handler now saves the original type, message, traceback and phase.

The original failure was reproduced with the same AST-defined function and a dynamic module without `__file__`, without dispatching dynamics. All 38 CPU tests pass: the original 32 synthetic checks plus six startup/lineage regression tests. Those tests validate code paths, not a successful CUDA gate. The saved first-attempt runtime report confirms CUDA/package readiness and checkpoint/core file hashes; full canonical authentication, model inference and scientific measurements remain pending.

## Unchanged decision and accounting

`contract.json` and `DECISION_FROZEN.json` are byte-identical to attempt 1. Contract SHA-256:

`7b27bcd379f30b3b627f1fcdd91ec9b3537d06cd1a31bbd0855faa4eba94d877`

The original attempt ended at 15:00:34.954739 UTC with exit code 1 after 37.399107418 seconds, with zero decoded episode rows, model forwards, resets, simulator transitions, optimizer or calibration updates. `LINEAGE.json` pins those receipts and the old manifest. The original report SHA-256 is `2fc5f1a019cda0a62858a0dcb12ec0ac73c91fd7d316b40c55996ce366116f1a`.

This retry remains capped at 900 seconds and 28,104 simulator transitions, with zero training/calibration updates. Prior setup time remains separate, so the maximum combined elapsed allowance is 937.399107418 seconds. Prior scientific work was zero; the combined real-transition ceiling remains 28,104. The new `run/ATTEMPT.json` will identify ordinal 2. No old reservation is removed or reused.

## Manual command

Run once from PowerShell on Zeus, when ready to launch this separately identified retry:

```powershell
wsl -d Ubuntu -- /home/david/bca-operator-audit-20261026/bin/python -B /mnt/c/Users/David/dev/BCA/experiments/thrusterchain_bca/zeus_20261026_validation/conditional_ood_gate_v1_retry_20261028/launch.py --execute
```

Use `--preflight` instead for file/package metadata checks without reserving or executing. The assistant-side WSL denial remains in force and was not retried. No software was installed. Do not rerun the original launcher or delete either reservation to repeat an attempt. A scientific failure remains a recorded failed gate, without changing the band, radius or pass marks.

Progress, report and terminal accounting will be saved under this retry's own `run/` directory. The broader training hold remains unchanged.

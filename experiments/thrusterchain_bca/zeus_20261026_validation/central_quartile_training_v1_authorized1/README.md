# Approved central-quartile study: one production attempt

User approved the prepared three-seed, twelve-hour maximum study on 2026-10-28 at 21:49:53 UTC (Sentinel_2f0ccb63d1d88191ad138e242854c996). AUTHORIZATION.json records that approval and binds the exact passing runtime evidence. This release has not started training.

The validated amendment and its receipts are unchanged. Runtime attempt 3 passed on the RTX 5070 Ti Laptop GPU in 78.0217 seconds including postcheck. Its report SHA256 is 490cc4e6f63ded5e1e8d6e1448d4f06e93733d64d0f5d1d42c381b2e04fb3553. The prior incomplete attempt's total work remains unknown. No additional runtime validation is needed or enabled in this release.

## What changed

The validated launcher intentionally had no activation path. This separately sealed release adds authorization.py and AUTHORIZATION.json and replaces only the launch authorization gate. All scientific implementation, protocol.json, PROTOCOL.md and DATA_PLAN.json are byte-identical to the passing amendment. Historical false approval flags inside protocol.json and earlier preparation documents describe preparation time; the separate sealed authorization is authoritative for this single release. They have not been rewritten.

The release launcher also recognizes the known validation workers in its scoped overlap check and rechecks the original absolute deadline after file verification. The mathematical method, seeds, data, updates, calibration boundaries, evaluation and thresholds are unchanged. Test fixtures and this README reflect the release entry point. The inherited prepare.py remains a historical amendment helper; use no resealing command after PREPARED.json exists.

## Fixed approved budget

- Three independent initialization seeds, full / conditional hole / state-count-matched random-removal datasets.
- Nine 10,000-critic-update warmups; 27 native / original BCA kappa 3 / logged-action-gradient-matched distance branches continuing to 20,000 steps.
- 360,000 critic, 180,000 actor and 360,000 auxiliary updates; exactly 36 scheduled calibrations, at 10k and 15k, with 36,000 posterior draws. No 11k checkpoint or 20k recalibration has been added.
- 72 free-policy snapshots at the existing fixed checkpoints; 4,608 free episodes, 24,192 forced-action Q-pi values, and at most 1,975,304 simulator transitions.
- One 43,200-second scientific execution deadline measured from exclusive reservation. Each warmup/branch has a 1,800-second cap; evaluation has a 7,200-second cap, always bounded by the remaining global time. Owned children are stopped at the deadline. Cleanup may take up to ten additional seconds; final read-only hash verification and receipt writing may extend command wall time. There is no extension, automatic retry or resume.
- Require 6 GiB of free workspace capacity for approximately 4.2 GB of checkpoint exports plus other evidence. No installs or downloads. High GPU utilization is expected; no allocator or batch changes were made.

## Manual launch on Zeus

Assistant-side WSL access remains denied and has not been retried. Run once yourself:

```powershell
wsl -d Ubuntu -- /home/david/bca-operator-audit-20261026/bin/python -B -u /mnt/c/Users/David/dev/BCA/experiments/thrusterchain_bca/zeus_20261026_validation/central_quartile_training_v1_authorized1/launch.py --execute
```

Before reserving anything, this command rechecks the release/input/validation identities, explicit approval, isolated Python path, package-inheritance setting, free disk and scoped process overlap. Each worker verifies its owned token and parent. Atomic exclusive ATTEMPT.json and run/ creation prevent duplicate attempts. Do not run the old pilot, amendment validator, another release, or a second copy concurrently. A failed/interrupted attempt remains recorded and requires review; never delete its reservation to retry.

For a read-only status snapshot, replace --execute with --status. --preflight verifies files and authorization without reserving or importing models; its Windows result does not establish current Linux process/GPU availability. Those live checks remain at manual launch.

Progress is PROGRESS.json plus run/<cell>/<warmup-or-arm>/progress.json and worker.log. Terminal receipts are RUN_RESULT.json, per-job SUPERVISOR_RESULT.json, and run/evaluation/report.json. RUN_AUTHORIZATION.json binds the attempt to the approval. Full-state checkpoints remain immutable. Further studies, retries and expanded training remain held.

## Evidence and limits

Thirty-two affected CPU authorization/launcher/receipt tests passed. One new fixture initially hit a Windows sandbox permission error for atomic replacement in the OS-selected temporary directory; it was corrected to use the explicitly writable workspace, then all thirty-two passed with the production atomic writer unchanged. Prior 133 CPU checks and actual CUDA validation remain preserved in the amendment.

CUDA validation established exact full-state restoration, all three branches' next-step parity, native parity, BCA gradients, five read-only real-source initializations and the Gymnasium checker. It did not establish long-run stability or efficacy. Alternative rendering modes were not checked because the environment has no registered spec. The training control is matched on logged-action gradient RMS, not actor-policy gradient strength. Three seeds and one fixed dataset/mask remain exploratory; reused calibration does not supply a fresh conditional-hole coverage guarantee.

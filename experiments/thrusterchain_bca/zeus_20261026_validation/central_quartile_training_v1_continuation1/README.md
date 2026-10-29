# Approved continuation of the interrupted central-quartile study

This separately reserved continuation preserves the original authorized study, its 32 completed jobs, partial job, full-state checkpoints and terminal receipt. It does not restart the original reservation. RECOVERY.json pins all prior artifacts; AUTHORIZATION.json records the explicit user approval for this continuation. PREPARED.json seals this package once.

## Remaining work and accounting

1. Restore seed2_hole/bca at 16,000 critic steps and finish at 20,000: 4,000 critic, 2,000 actor and 4,000 auxiliary updates, with no new calibration. The completed 15k calibration and evaluation export are authenticated and retained.
2. Run the untouched seed2_random distance, native and BCA branches from their original 10k warmup: 30,000 critic, 15,000 actor and 30,000 auxiliary updates and three calibrations.
3. Run the unchanged evaluation: 72 snapshots, 4,608 episodes, 24,192 forced-action Q-pi values and at most 1,975,304 simulator transitions. The 32 original successful jobs are read from their original paths; only the four remaining jobs write here.

New work totals 34,000 critic, 17,000 actor and 34,000 auxiliary updates and three calibrations. Original recorded work is a lower bound of 326,600 critic, 163,300 actor and 326,600 auxiliary updates with 33 calibrations. Completing the recovery therefore costs at least 360,600 critic, 180,300 actor and 360,600 auxiliary updates overall. At least 600 critic/auxiliary and 300 actor updates repeat work lost after the durable checkpoint. The interruption tail remains unknown; it is not counted as zero. Logical checkpoint counters and physically consumed work are separate. The checkpoint does not preserve the Python unique-row bitmap: resumed unique-row counts describe only the new segment.

## Time limit

The scientific deadline is the earliest of four hours after the new reservation, the remaining original consumed-time allowance of 26,748.234014457 seconds, and October 29 at 5:31:39.070521 a.m. EST (fixed UTC-05:00). It never resets the original twelve-hour allowance. Each training worker remains capped at 1,800 seconds and evaluation at 7,200 seconds, also bounded by the global deadline. The supervisor stops only its owned child process group. Cleanup and final read-only verification may finish later; if the final postcheck exceeds the deadline, the run is explicitly classified as completed_work_postcheck_exceeded_deadline rather than successful. There is no automatic restart or additional continuation authorization.

## Restore and integrity safeguards

The unchanged runtime builds the original adapter from the authenticated 10k warmup once, then restores the 16k full state exactly. Parameters, target networks, optimizer states, both random streams, held minibatch, phase, refresh schedule, Bayesian calibration state and counters are checked before the first update. No reseeding, distance rematching or 15k recalibration is allowed. The original 15k gradient and evaluation artifacts are copied byte-for-byte with lineage; no diagnostics are recomputed for that prefix. Core runtime, model kernels, data plan, scientific protocol, normalization, seeds and evaluation math remain unchanged.

Real CUDA validation of the original runtime previously passed. The new continuation routing and accounting have CPU fixture coverage; its actual GPU entry has not been executed during preparation. The exact saved-state restore gate runs inside the first user-launched worker before any optimizer update. A failed gate stops and preserves the attempt. This is not a claim of long-run stability or BCA efficacy.

## Manual command on Zeus

```powershell
wsl -d Ubuntu -- /home/david/bca-operator-audit-20261026/bin/python -B -u /mnt/c/Users/David/dev/BCA/experiments/thrusterchain_bca/zeus_20261026_validation/central_quartile_training_v1_continuation1/launch.py --execute
```

The command verifies seals, approval, original terminal evidence, all pinned prior artifacts, paired initializations, existing isolated Python, free disk and scoped process overlap before exclusive reservation. It requires 6 GiB free workspace. No software installation, GPU allocator change or batch change is made. Assistant-side WSL access remains denied; no bypass was attempted. Do not run the old reservation or another copy.

Progress: PROGRESS.json and run/<cell>/<arm>/progress.json. Final receipts: RUN_RESULT.json and run/evaluation/report.json. Replace --execute with --status for a read-only snapshot. --preflight checks files and authorization without reserving or importing models; live process/GPU checks remain at manual launch.

## Evidence and historical files

CONTINUATION_CHECKS.json and CONTINUATION_REVIEW.md record current checks. Copied TEST_RESULTS.json, RELEASE_CHECKS.json, older reviews and release-specific fixtures describe earlier packages; their historical command paths and original 37-job launcher expectations do not apply to this continuation. The old prepare.py and seal_authorized_release.py are retained historical source, not a way to reset or reseal this attempt. Only seal_continuation.py creates this package seal, and refuses an existing seal or reservation.

All original scientific limitations remain: three exploratory seeds, one fixed dataset/mask, logged-action rather than on-policy gradient matching, reused calibration without a fresh conditional-hole guarantee, and no interpretation of benchmark validity as method efficacy. No further studies or tuning are authorized.

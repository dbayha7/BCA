# Zeus support-shift retry, attempt 2

This is a separately recorded retry of the original approved protocol, using the
same scientific code, data, seeds, batch size, allocator environment, and limits.
The original Zeus attempt was stopped by the user over 100% GPU utilization;
no OOM or scientific failure is inferred. Its copy-preserving archive and original
reservation remain intact. RETRY.json links that archive. Its last saved counters
are lower bounds, and its unknown final counts stay separate from this attempt.

The retry has not started. At 04:10 UTC the user reported that a manual `ps` query
for the recorded supervisor PID 372 and worker PID 486 returned only its header.
ORIGIN_STOPPED.json may record this trusted report as absence of those two PIDs
at that time; it explicitly does not claim a global/scoped process scan or exact
final work counts. Execution always performs a fresh scoped check of old/new/prior
BCA launcher paths before reserving. A permission error or matching process blocks
execution. CPU preparation cannot establish live process state.

This optional read-only command performs the full scoped check and records a fresh
clearance receipt. It does not reserve an attempt, initialize models, or touch the
original attempt files; the already supplied recorded-PID report does not need to
be repeated merely to prepare the retry.

```powershell
wsl -d Ubuntu -- /home/david/bca-operator-audit-20261026/bin/python -B -u /mnt/c/Users/David/dev/BCA/experiments/thrusterchain_bca/zeus_20261026_validation/support_shift_pilot_v1_retry_20261027/launch.py --check-origin-stopped
```

--preflight checks frozen files and inputs without loading a model. --status
reads saved retry status and lineage. Execution requires the successful clearance
receipt and repeats the live process check before its permanent reservation. No
execution starts in a preparation/check mode; no automatic retry or resume exists.
Do not run the original attempt again or remove its reservation.

The unchanged per-attempt caps are 64,000 critic, 32,000 actor and 64,000 auxiliary
updates, 16 calibrations, 512 evaluation episodes, 51,200 simulator transitions,
and 150 minutes including compilation. Original consumed work is additional and
reported separately. Actor/critic/BCA mathematics, data restriction, pairing,
calibration boundaries, matching rule, and final-only evaluation are unchanged.

PREPARED.json seals this retry's launcher and lineage changes. Every original
frozen scientific file retains its exact bytes. The copied launch.py, README.md,
and test_launch.py differ only for retry supervision, instructions, and fixtures.
Original CPU receipts describe original tests; test_retry.py adds synthetic
process-gate and retry-isolation coverage without CUDA or real scientific work.

With the recorded user PID-check receipt present on Zeus, the run-once command is:

```powershell
wsl -d Ubuntu -- /home/david/bca-operator-audit-20261026/bin/python -B -u /mnt/c/Users/David/dev/BCA/experiments/thrusterchain_bca/zeus_20261026_validation/support_shift_pilot_v1_retry_20261027/launch.py --execute
```

The live process check happens before reservation. A new checkout does not inherit
machine-local clearance; use the check mode there first. Read progress with the
same command replacing `--execute` with `--status`. No new head-count experiment or
broader training is enabled.

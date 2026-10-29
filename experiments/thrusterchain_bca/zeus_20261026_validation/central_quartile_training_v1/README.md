# Matched conditional-hole study: preparation and verification

The completed `matched_policy_pilot_v1` is a different, earlier ordinary-task continuation experiment. Never rerun its reservation. This new package starts from fresh initialization for every dataset condition. No production attempt has been reserved here, and `--execute` is disabled in code.

Read [PROTOCOL.md](PROTOCOL.md) and [REVIEW.md](REVIEW.md) before interpreting readiness. `protocol.json` and `DATA_PLAN.json` give exact seeds, masks, row identities, query/update caps and decision rules. `PREPARED.json`, when present, seals implementation, specification and dependencies for one target validation. It is not training authorization.

## Proposed production budget

- Three independent initialization seeds × full, hole and state/count-matched random-removal datasets.
- Nine 10k native warmups, each forked into native / original BCA κ=3 / logged-action-gradient-strength-matched distance branches ending at 20k: 27 final trained policies.
- 360,000 critic updates, 180,000 actor and each target update, 360,000 shadow/active auxiliary updates, 36 calibrations and 36,000 posterior draws.
- Fixed 0/10k/15k/20k checkpoints: 72 free-policy snapshots, 63 trained Q snapshots, 4,608 free episodes, 24,192 forced-action Q-pi values.
- 1,975,304 simulator transitions maximum; 1,953,792 actor rows; 40,320 critic rows and 40,320 uncertainty-width rows, including cached-trajectory joins.
- 63 fixed training-gradient diagnostics / 48,384 probe rows, plus 9 matching diagnostics / 6,912 logged-action rows; zero optimizer or simulator work in these diagnostics.
- 12-hour global ceiling, including compilation, diagnostics, I/O and evaluation. Each training job <= 30 minutes; evaluation <= 2 hours; no extension beyond remaining global time.
- Approximately 4.2 GB of full-state exports plus actors/logs/receipts; require 6 GiB free workspace capacity. No install or network download.

The previous Zeus receipts imply roughly 7–8 hours training plus evaluation. This is an extrapolation, not a benchmark of the new package. In particular, per-call full-state hash guards may make evaluation slower than a linear extrapolation of the design gate. A time limit yields an incomplete study, never an automatic retry.

## Target validation, separate from production training

Assistant-side WSL access remains denied. Do not route around that denial. The user may manually run the following after the package is sealed:

```powershell
wsl -d Ubuntu -- /home/david/bca-operator-audit-20261026/bin/python /mnt/c/Users/David/dev/BCA/experiments/thrusterchain_bca/zeus_20261026_validation/central_quartile_training_v1/launch.py --validate-runtime
```

This command has a 900 second supervisor cap. It checks the pinned runtime, five untrained real-source initialization cases with no fitting, bounded synthetic native parity/fork/restore/gradient checks, and the full Gymnasium checker. Maximum real-source decode is 4,480 repeated episode reads / 448,000 tuples from the same 768 training + 128 calibration roles; protected transition records remain unread. No optimizer or calibration update uses real source rows. Synthetic optimizer caps are 64 critic / 32 actor / 64 auxiliary updates; actual planned dispatch counts are reported. Gymnasium checker cap 512 steps / 128 resets. These validation steps are not production training, OOD efficacy evaluation or permission to launch.

Validation writes a distinct `runtime_validation_v1` reservation/report. A failure is preserved and must be inspected; the command will not silently reuse its reservation. The actual GPU models, Gymnasium checker, complete data-to-model integration and target filesystem writes have not been verified until this command returns a passing sealed receipt. Inspect all reported unresolved numerical checks, including any finite-difference resolution limit, rather than treating exit status alone as a scientific guarantee.

The production launch remains unavailable until its 12-hour/resource proposal is explicitly approved and all must-fix checks are resolved. A reviewed authorization change must bind the validated scientific specification and preserve immutable lineage; do not hand-edit flags, clear reservations or reuse an interrupted identity.

## Local checks

Use the existing Windows interpreter, which has the CPU test dependencies; the shell's bare `python` points elsewhere:

```powershell
C:\Users\David\dev\BCA\.venv\Scripts\python.exe -B -m unittest discover -s experiments/thrusterchain_bca/zeus_20261026_validation/central_quartile_training_v1 -p "test_*.py"
```

These fixtures use toy/generated data and pinned computational interfaces. They do not execute production updates, stored-policy simulator evaluations or WSL. Their scope and any mocks are recorded in the review. Protected data, old runs and source files are preserved.

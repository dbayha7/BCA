# Evaluation-only rerun of the central-quartile study

This package reruns the study's single evaluation once. It trains nothing.

## Why

- **Training is complete:** 32 jobs in `central_quartile_training_v1_authorized1` and 4 in `central_quartile_training_v1_continuation1`.
- **The first evaluation stopped early.** On 2026-10-29 at 02:16 EDT it had finished 143 of its 144 snapshot contexts when replacing its own `progress.json` raised `PermissionError: [Errno 13]`. The cause was a momentary file lock on the Windows-mounted drive, not a numerical problem.
- **Where that left the study:** the frozen stop rule then ended it with `incomplete_no_utility_conclusion`.
- **The approval:** you approved this rerun on 2026-10-29. Your exact words are in `AUTHORIZATION.json` and `RERUN_PROTOCOL.md`.

## What it does

1. **Checks everything first.** It verifies this package's seal, the approval, every pinned file of the original study (`RECOVERY.json`, 1,317 files) and of continuation1 (`RERUN_INPUTS.json`, 368 files), the paired initializations, the isolated Python and the absence of other BCA processes. Then it reserves the single attempt.
2. **Runs the unchanged evaluation.** It covers the 72 saved snapshots in both contexts, for 144 snapshot contexts, using the same seeds, banks, simulator, reference adapter and evaluation math.
   - **Inputs:** the 32 original jobs are read from `authorized1` and the 4 continuation jobs from `continuation1`, each file checked against its pin.
   - **Outputs:** everything is written only under this package's `run/`.
3. **Applies the reproduction gate as it goes** (kill rules K1 to K3 in `RERUN_PROTOCOL.md`):
   - Both rebuilt state banks must equal continuation1's exactly.
   - Each of the 143 recorded contexts must be byte-identical (Tier 1), or else match under the approved near-exact rule (Tier 2): same structure, strings, booleans, nulls and integers, with every float within |new−old| ≤ 1e-6 × (1 + |old|).
   - Only `seed2_random__distance__20000/strong_current` may be new.
   - Anything else stops the run as incomplete.
4. **Decides from the rerun only.** If all of that passes, the frozen decision rules in `PROTOCOL.md` are applied by the unchanged code to this rerun's own 144 contexts.

## Limits

- **Time:** 7,200 seconds (2 hours) from reservation, within the unspent 12-hour allowance of 24,295.97 seconds. The original calendar deadline is waived by your approval.
- **Work caps:** the evaluation caps are unchanged: 4,608 episodes, 1,975,304 simulator transitions and 1,953,792 actor rows. Training and calibration updates are zero.
- **Attempts:** there is no automatic retry and no third evaluation attempt.

## Command on Zeus (your terminal)

Run this read-only check first:

```powershell
wsl -d Ubuntu -- /home/david/bca-operator-audit-20261026/bin/python -B -u /mnt/c/Users/David/dev/BCA/experiments/thrusterchain_bca/zeus_20261026_validation/central_quartile_training_v1_evaluation_rerun1/launch.py --preflight
```

Then launch the run:

```powershell
wsl -d Ubuntu -- /home/david/bca-operator-audit-20261026/bin/python -B -u /mnt/c/Users/David/dev/BCA/experiments/thrusterchain_bca/zeus_20261026_validation/central_quartile_training_v1_evaluation_rerun1/launch.py --execute
```

The first evaluation took about 10 minutes, and the checks before and after add about a minute each.

- **When it's finished:** the run prints one long JSON line at the end; that line means `RUN_RESULT.json` is written.
- **After the evaluation:** a Ctrl+C during the final verification is recorded and deferred, not lost.
- **During the evaluation:** a Ctrl+C still stops it, and the attempt is then used up.
- **Status:** `--status` prints a read-only snapshot.

## Reading the result

- `RUN_RESULT.json` → `status`: success is `completed_evaluation_rerun`. Anything else is incomplete.
- `RUN_RESULT.json` → `reproduction`: how many contexts matched at Tier 1 and at Tier 2, the largest scaled Tier 2 deviation, and whether the bank archives were byte-identical.
- `RUN_RESULT.json` → `disclosure`: the facts any write-up must carry:
  - two continuations and two evaluation attempts;
  - the waived calendar deadline;
  - training totals that are lower bounds, at least 360,600 critic updates (`cumulative_execution_counts_known=false`);
  - both continuations approved before any evaluation result was seen.
- `run/evaluation/report.json` holds the frozen screens: learning adequacy, the policy screen, the accuracy screen, the interactions and all three seed effects.

## Package contents

- **Byte-identical to the sealed continuation1 package:** `common.py`, `recovery_inputs.py`, `evaluation.py`, `runtime.py`, `data.py`, `design.py`, `state_exact.py`, `gradient_audit.py`, `train_worker.py` (never dispatched; kept for test fixtures), `protocol.json`, `PROTOCOL.md`, `DATA_PLAN.json`, `RECOVERY.json` and their unchanged tests. `authorization.py` enforces this list.
- **Changed:**
  - `evaluate_worker.py`: routing, lock-tolerant writes, a progress throttle and the gate.
  - `launch.py`: a single job, the new time cap, both pin sets, interrupts deferred after the work, and disclosure fields.
  - `authorization.py`
  - `test_evaluate_worker.py`: fixture roots and the writer mock only.
  - `README.md`
- **New:**
  - `lock_tolerant.py`, `rerun_inputs.py`, `prepare_rerun.py`, `seal_rerun.py`
  - `RERUN_PROTOCOL.md`, `RERUN_INPUTS.json`, `AUTHORIZATION.json`, `RERUN_CHECKS.json`, `RERUN_REVIEW.md`
  - `test_rerun_*.py`

## What was and wasn't verified

**Run on Windows CPU:**
- tests of unchanged code;
- a real Windows lock reproduction showing the old writer fails and the new one recovers;
- an end-to-end fake-model flow through the gate;
- launcher control flow;
- read-only checks of the real pins and inventory.

**Not run during preparation:**
- the evaluation entry on the real GPU;
- any check of whether the GPU reproduces the earlier numbers bit for bit (Tier 2 exists for that).

Nothing here claims BCA efficacy. All limitations of the original protocol remain.

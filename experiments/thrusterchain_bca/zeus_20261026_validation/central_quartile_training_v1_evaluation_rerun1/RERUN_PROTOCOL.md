# Evaluation-only rerun: scope and kill rules

Written 2026-10-29, before any code for this rerun was written and before any test or run. Nothing here changes the scientific protocol in `PROTOCOL.md`, which stays byte-identical.

## Why this exists

The central-quartile study's training finished: 32 jobs in `central_quartile_training_v1_authorized1` plus 4 in `central_quartile_training_v1_continuation1`. Its single evaluation then stopped after 143 of 144 snapshot contexts. The error was `PermissionError` while replacing its own `run/evaluation/progress.json` on the Windows-mounted drive, which is a momentary file lock, not a numerical or scientific failure. Under the frozen stop rule, the study ended with the decision `incomplete_no_utility_conclusion`.

The 143 finished result files were checked by hash only. As of this writing, no person or agent has read their contents.

## Approval

The user approved this rerun in chat on 2026-10-29 (answers received by 17:00:19 UTC):

- "Rerun eval".
- Question: "Approve one evaluation-only rerun? No training: it re-evaluates the 72 existing snapshots with the frozen evaluation code, plus a fix so a briefly locked file can't crash it. One attempt, launched by you, capped at 2 hours. It runs past the original Oct 29, 5:31 a.m. EST deadline, using unspent time from the 12-hour budget, and is disclosed as a second continuation." Answer: "Yes, approve (Recommended)".
- Question: "The GPU may not reproduce the 143 earlier results down to the last bit. What should count as reproduced?" Answer: "Exact or near-exact (Recommended)". The option read: byte-identical; or, failing that, every number within |new-old| <= 1e-6*(1+|old|) and every count, label and string identical; anything else stops the run as incomplete; which one applied is disclosed.

Launching the command from the user's own terminal is the final approval step.

## What runs

There is one evaluation job and nothing else: no training, calibration, distance matching or checkpoint creation.

- **What it evaluates:** the same 72 snapshots (9 cells × untrained 0k, common 10k, and each of the three arms at 15k and 20k) in the same two contexts, "ordinary" and "strong-current".
- **What is unchanged:** the same frozen seeds, state banks, reference adapter, simulator and evaluation math. `evaluation.py`, `runtime.py`, `data.py`, `design.py`, `state_exact.py`, `gradient_audit.py`, `common.py`, `recovery_inputs.py`, `protocol.json`, `PROTOCOL.md`, `DATA_PLAN.json` and `RECOVERY.json` are byte-identical to the sealed continuation package.
- **Where the inputs come from:**
  - The 32 original jobs are read from `central_quartile_training_v1_authorized1`, authenticated against the existing `RECOVERY.json` pins.
  - The 4 continuation jobs are read from `central_quartile_training_v1_continuation1`, authenticated against new pins in `RERUN_INPUTS.json`, which records every file in that package and its `run/` folder.
- **Where outputs go:** only to this package's `run/evaluation/`. No earlier file is moved, edited or deleted.

The only intended code changes are four:

1. **Routing.** The four continuation jobs are read from `continuation1` instead of the running package's own folder.
2. **Lock-tolerant writes.**
   - Every file the evaluation writes by temporary file plus rename retries when Windows reports a lock (`PermissionError`).
   - For progress files, which are evidence only and never results, a write still failing after its retries is skipped and counted, not fatal.
   - For results, banks, reports and receipts, a write still failing after its retries stops the run.
   - The bytes written are identical to the original writer's.
   - Progress saves are also throttled to at most one per second; the first attempt saved about 90 per second. This cuts the rename churn behind the collision and changes no computed value, because the progress callback ignores its argument.
3. **Reproduction gate.** This is the check below.
4. **Launcher and authorization.** These cover the single job, the new time limit, the pins and this approval. In addition, once scientific work ends, interrupt signals are deferred until `RUN_RESULT.json` is written, so a late Ctrl+C cannot lose the terminal receipt.

## Time and work limits

- **Time cap:** 7,200 seconds (2 hours) from reservation, covering pre-dispatch verification and the evaluation worker. It is also limited by the unspent 12-hour allowance: 43,200 − 16,451.765985543 (original attempt) − 2,452.264564082 (continuation1) = 24,295.969450375 seconds.
- **Calendar deadline:** the original calendar deadline (2026-10-29 05:31:39 EST) is waived by the approval above.
- **Postcheck:** cleanup and read-only verification may finish after the cap. If they do, the run is labeled `completed_work_postcheck_exceeded_deadline`, not success.
- **Work caps:** the evaluation caps are unchanged and fail closed: 4,608 free episodes, 1,975,304 simulator transitions, 1,953,792 actor rows, 40,320 critic rows, 40,320 width rows and 129 resets. There are zero optimizer and calibration updates.
- **Cumulative work to disclose:**
  - The first evaluation attempt consumed at least 1,961,832 completed simulator transitions before stopping, so total evaluation work becomes at most 3,937,168 transitions, roughly twice the plan.
  - Training totals stay as continuation1 recorded them: lower bounds of 360,600 critic, 180,300 actor and 360,600 auxiliary updates with 36 calibrations, not exact.

## Kill rules

Any one of these stops the run as incomplete, with no conclusion about utility. There is no automatic retry, and no third evaluation attempt is authorized.

- **K1. Banks.** For each context, the rebuilt state bank must reproduce continuation1's bank exactly:
  - the same `bank_id`, a content hash of states, times, starts and seeds;
  - the same descriptive fields, apart from the `arrays` file descriptor;
  - the same array names, dtypes, shapes and raw bytes, compared against continuation1's `.npz`.

  The decision rests on array contents. The archive's own bytes are also expected to match, because `numpy.savez` writes each entry with zipfile's fixed 1980-01-01 date rather than the clock. Whether they matched is recorded but not required.
- **K2. Context results.** Each of the 143 contexts recorded in `continuation1/run/evaluation/report.json` is checked as soon as the rerun writes it:
  - **Tier 1 (exact):** identical byte length and SHA-256.
  - **Tier 2 (near-exact), used only if Tier 1 fails:**
    - both files parse to the same JSON structure, with the same keys and the same list lengths;
    - every string, boolean, null and integer is identical;
    - every float satisfies |new − old| ≤ 1e-6 × (1 + |old|);
    - an integer on one side and a float on the other counts as a failure.
  - **Neither tier passes:** stop.
- **K3. Coverage.** The only context allowed to have no recorded counterpart is `seed2_random__distance__20000/strong_current`. Any other unrecorded context stops the run, and so does finishing with a recorded context not reproduced.
- **K4. Original stop rules.** Every stop rule in `PROTOCOL.md` still applies: numerical invalidity, source mismatch, unsafe calibration, budget exhaustion, a missing job, or unknown counts. So do all of the evaluator's own checks, including exact final accounting and unchanged model state.
- **K5. Changed inputs.** Stop if any pinned input changes, whether a package file, an original-study artifact or a continuation1 artifact, or if the scientific protocol differs. This is checked before reservation and again after the run.
- **K6. Interruptions and limits.** Stop on a timeout, an interrupt, or reaching the cap. A retried result write that still fails also stops the run.

## Decision if the run completes

The frozen decision rules (`PROTOCOL.md`, "Decision rules and independent units") are applied by the unchanged `summarize` function to this rerun's own 144 contexts. The 143 recorded files serve only as the reproduction check, never as inputs to the decision. All three seed effects are reported, and a mixed result fails the screen; it does not show equivalence.

## Required disclosure

Any report of this study must state:

- **Two continuations.** The study needed two separately approved continuations: continuation1 resumed training after an interrupt, and this one reran the evaluation after a file-lock failure.
- **Evaluation attempts.** There were 2.
- **Gate outcome.** How many contexts matched at Tier 1 and how many at Tier 2.
- **The waived deadline.** The original calendar deadline was waived.
- **Training counts.** Training update totals are lower bounds (`cumulative_execution_counts_known=false`). At least 600 critic and auxiliary updates and 300 actor updates were repeated work.
- **Outcome-blindness.** Both continuations were approved before anyone had seen any evaluation result.

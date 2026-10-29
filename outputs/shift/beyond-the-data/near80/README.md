# The main setting on fresh seeds (version 9, near-80 main): read me first

**DRAFT, not frozen, awaiting the owner. Nothing has been trained.**

## What this is

Version 8 ran on 19 October 2026. Its claim 4 (step 2's result in the main setting on fresh seeds 10-14) failed by
the pre-set rule: on seeds 10, 12 and 13 BCA's held-out full-throttle group had 19, 22 and 18 rows, too few for
BCA's certificate (it needs about 29), the height was infinite, and the approved code stopped the whole actor phase
of those seeds. The switch's on/off call itself was right in 10 of 10 seed-worlds.

Version 9 is the same test with the cause fixed and nothing else moved:

- **One setting**, the main test's (the densest 15% of recorded commands removed). No band, no random rows.
- **Ten fresh seeds, 40-49**, from a block no study of this generator has used (50-59 are reserved for a later
  version).
- **BCA's two groups pooled** over every held-out recorded row outside the removal ball (the whole calibration split
  and the whole evaluation split), fixed in advance; every switch arm reads the same groups; the training rows, the
  test start sets and every rival's own rule are unchanged; no scored test reads the pooled rows. On the 20 known
  seed-worlds the group grows from 18-60 rows to 35-135 and BCA refuses on none of them (a preview, nothing trained).
- **An outcome-free count** of the full-throttle group per seed and world, recorded before any training (reported).
- **The refusal rule:** if BCA still refuses on a seed, its own arms count as the controller trained without caution
  in that seed-world for C1, and the seed is left out of C2 and C3; every other arm trains and is scored; nothing
  stops.
- **Four new trained arms** reading the same groups: the mean-difference switch with the plain 90% gap, with twice
  that gap, turning on the hand-set range limit (the three plain twins), and BCA's switched ramp trained alone.
- **Three claims:** C1 = version 8's claim 4 marks, unchanged (the hand-set range limit joins the rules that cannot
  switch themselves off, against which BCA's lead is measured); C2 = BCA's switched ramp beats each plain twin by 5.0
  points on the net across worlds (gain where full throttle stalls minus loss where it helps), paired by seed, in
  at least ceil(0.8 n) of the n certified seeds (8 of 10), all three twins; C3 = the mirror, a twin beats BCA by 5.0
  points with BCA ahead in at most 1 seed, one claim per twin. C2 and C3 need at least 8 certified seeds.
- **Two entries:** the university cluster hpc.charlotte.edu (`run_near80_cluster.sh` with `near80_slurm_job.sh`:
  one Slurm job per fresh seed and a group job for the lock, the evaluations and the look, the same frozen code and
  checks; its stub run finished all 19 phases there on 20 October, on a synthetic fixture; the label seed 40 was
  used once, before the seed-0 rule; nothing of that seed was generated) runs version 9 by your answer; the desktop
  (`run_near80.sh`, as version 8) is the fallback, and the place the results are copied to, checked and committed
  from (your second answer).

## Fixed by DECISION.md on 20 October 2026

Ten seeds 40-49; C2 5.0 points and ceil(0.8 n) seeds, all three twins; C3 -5.0 points and at most 1 seed; at least 8
certified seeds; the refusal read per arm (C1: no caution; C2 and C3: the seed left out); version 7's memory rule; 10
lanes; version 9 in place in `near80/`; the twins' factors 1 and 2; no second held-out log; no fresh start bank; no
retrain of seeds 0-4's second critics. `REVISION_LOG.md` beside `DECISION.md` records the three reviews that shaped
these values.

## Your answers (20 October 2026, in the chat with the cloud assistant)

At 18:29 UTC, "Accept the defaults and cluster": every default of section 16, with the university cluster as the run
machine (choice 4) and the desktop kept as the fallback. Asked whether "cluster" also meant where results are
committed from (choice 9), at about 19:20 UTC: "committed from desktop after copying over from cluster", that is the
default, the desktop: the results folder and the output root are copied from the cluster to the desktop over your
own SSH session, checked by SHA-256 there, and committed and pushed from the desktop's clone, as version 8's records
were. Both are recorded with their times and place in `OWNER_ANSWER` of `code/near80_constants.py`; `prereg.json`
will carry them.

## What is open (the freeze refuses while it remains)

1. The facts the frozen code computes before approval (section 4.2; about 10-15 CPU-minutes, nothing trained) and
   the number of redrafts after their kill rule; the freeze fills them.

`V9_BUILD_NOTES.md` lists every change of this build, every test, the time estimates and the items the freeze fills.

## How to approve (after the freeze)

As in version 8, on the Framework Desktop, inside Ubuntu as root:

```bash
cd /home/user/bca
git pull --ff-only
python3 -I outputs/shift/beyond-the-data/near80/approve_local_near80.py --check
python3 -I outputs/shift/beyond-the-data/near80/approve_local_near80.py
```

It writes `approval_near80_<first 16 characters>.json` (+ `.sha256`) and prints the commands to commit and push them
under your own name. The session imports your approval and pushes `import_near80_<first 16 characters>.json`. Nothing
trains before that record is on the branch. Version 8's own approval and import records stay in this folder as
history; the script allows exactly them.

## How to run on the cluster (after the import)

On a login node of hpc.charlotte.edu, never as root, from a git clone of the approved commit under `/scratch/dbayha`
with the environment of `V9_BUILD_NOTES.md` (the pinned Python environment at `/scratch/dbayha/bca-near80/env`, the
shift dependencies at `/scratch/dbayha/bca-near80/shiftdeps`, the five `runs/` folders copied read-only from the
desktop), `.../run_near80_cluster.sh` standing for `sh $NEAR80_ROOT/outputs/shift/beyond-the-data/near80/run_near80_cluster.sh`:

1. `export NEAR80_HOST=cluster NEAR80_ROOT=<the clone> NEAR80_PYTHON=/scratch/dbayha/bca-near80/env/bin/python BCA_SHIFT_DEPS=/scratch/dbayha/bca-near80/shiftdeps`,
   then `git -C $NEAR80_ROOT pull --ff-only`.
2. `.../run_near80_cluster.sh selftest` must end with `READY`.
3. `.../run_near80_cluster.sh plumbing` submits one job (4 processors, 16 GB, 2 hours); its log must end with
   `plumbing PASSED`. If its reproduction check differs, version 9 moves to the desktop (below).
4. `.../run_near80_cluster.sh run` submits one job per fresh seed (8 processors, 24 GB, 3 hours; Cascade Lake nodes)
   and the group job (16 processors) that waits for them: about 1.7 hours in all. `.../run_near80_cluster.sh status`
   shows what is happening; `scancel` of a job is a safe stop (the running phases finish), and the same `run` command
   continues afterwards.
5. When the group job's log prints `Finished`: `.../run_near80_cluster.sh results` gathers the records into
   `near80/results-<utc>/` with `SHA256SUMS` and writes `OUTPUT_ROOT_SHA256SUMS` (every file of
   `runs/shift_explore/near80_v9/`, model files included), and prints the steps that follow. Nothing is committed
   or pushed from the login node (your answer to choice 9). In your own SSH session, copy the results folder and the
   output root to the desktop clone (`scp -r` or `tar` over SSH: `near80/results-<utc>/` into
   `/home/user/bca/outputs/shift/beyond-the-data/near80/`, `near80_v9/` into `/home/user/bca/runs/shift_explore/`),
   check both lists there (`sha256sum -c`), then on the desktop commit and push the results folder with the commands
   it printed (`git add -f`) and run
   `sh /home/user/bca/outputs/shift/beyond-the-data/near80/push_runfiles_near80.sh` (it needs a terminal and your
   GitHub sign-in).

## How to run on the desktop (the fallback)

Inside Ubuntu as root, `.../run_near80.sh` standing for `sh /home/user/bca/outputs/shift/beyond-the-data/near80/run_near80.sh`:

1. `git -C /home/user/bca pull --ff-only`
2. `.../run_near80.sh selftest` must end with `READY`.
3. In tmux (`tmux new -s near80`): `.../run_near80.sh plumbing --lanes 10` (about an hour) must end with
   `plumbing PASSED`.
4. In tmux: `.../run_near80.sh run --lanes 10` (about 5-5.5 hours in one sitting). Ctrl-C once stops safely; the
   same command continues after any restart. `.../run_near80.sh status` shows what is happening.
5. When it prints `Finished`, commit and push the results folder with the commands it prints (`git add -f`: the
   repository's `.gitignore` hides `data/` folders), then
   `sh /home/user/bca/outputs/shift/beyond-the-data/near80/push_runfiles_near80.sh`.

The messages (`UNFINISHED`, `WAITING FOR RETRY`, `PAUSED`, `HALT`, `REFUSED`, `FAILED`) and what to do are as in
version 8's README; `retry <phase>`, `budget-continue fresh` and `report` are the answers.

## Files

- `PREREGISTRATION.md`: the full pre-registration.
- `prereg_draft.json`: the machine-readable draft; `code/freeze.py write` turns it into `prereg.json` once your
  answer is recorded.
- `approve_local_near80.py`, `run_near80_cluster.sh` with `near80_slurm_job.sh`, `run_near80.sh`,
  `push_runfiles_near80.sh`: the approval script, the cluster entry, the desktop entry and the run-files push.
- `code/`: the pooled groups and the count (`near80lib.py`, `facts_near80.py`), the approval chain
  (`approval_check.py`, `import_approval.py`, `near80_prereg.py`, `freeze.py`), the runner (`near80_runner.py`,
  `phases.py`, `phase_main.py`, `settingdata.py`, `learner_phases.py`, `policy_phase.py`, `evaluate_phase.py`,
  `ledger.py`, `stubs.py`), the look (`pooled.py`, `claims_text.py`), the constants (`near80_constants.py`) and five
  unit test files (`test_near80.py`, `test_approval.py`, `test_runner.py`, `test_pooled.py`, `test_v9.py`).
- `V9_BUILD_NOTES.md`: every change, test, estimate and open item of this build (not pinned).
- `decision/`: copies of the session's decision files (`DECISION.md`, whose verdict table is PREREGISTRATION.md
  section 14.6, `REVISION_LOG.md`, `credit.md`, the power calculations, `PREVIEW.md`), kept as history (not pinned).
- Version 8's history: its approval and import records, `amendment_1.json`, `pre_approval/`,
  `results-20261020T115246Z/`, and (after the freeze) `prereg_history/prereg_superseded_51e677ab76c8d2f7.json` and
  `code_manifest_history/code_manifest_v6.json`.

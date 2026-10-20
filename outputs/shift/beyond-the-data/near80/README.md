# The main setting on fresh seeds (version 9, near-80 main): read me first

**DRAFT, not frozen, awaiting the owner. Nothing has been trained.**

## What this is

Version 8 ran on 19 October 2026. Its claim 4 (step 2's result in the main setting on fresh seeds 10-14) failed by
the pre-set rule: on seeds 10, 12 and 13 BCA's held-out full-throttle group had 19, 22 and 18 rows, too few for
BCA's certificate (it needs about 29), the height was infinite, and the approved code stopped the whole actor phase
of those seeds. The switch's on/off call itself was right in 10 of 10 seed-worlds.

Version 9 is the same test with the cause fixed and nothing else moved:

- **One setting**, the main test's (the densest 15% of recorded commands removed). No band, no random rows.
- **Fresh seeds** from a block no study of this generator has used (40-59). The count is an open item.
- **BCA's two groups pooled** over every held-out recorded row outside the removal ball (the whole calibration split
  and the whole evaluation split), fixed in advance; every switch arm reads the same groups; the training rows, the
  test start sets and every rival's own rule are unchanged; no scored test reads the pooled rows.
- **An outcome-free count** of the full-throttle group per seed and world, recorded before any training (reported).
- **The refusal rule:** if BCA still refuses on a seed, that counts as BCA failing that seed for the arms that depend
  on BCA's height only; every other arm trains and is scored.
- **Two new trained arms** reading the same groups: the mean-difference switch with the ramp, and the mean-difference
  switch turning on the hand-set range limit.
- **Two claims:** C1 = version 8's claim 4 marks, unchanged; C2 = BCA's switched ramp beats each plain twin (the
  split-conformal switch and the two mean-difference switches), paired by seed, with margins and a multiplicity rule
  that `DECISION.md` fixes.
- **Two entries:** the desktop (as version 8) and the cluster at hpc.charlotte.edu (Slurm).

## What is open (the freeze refuses while any remains)

`code/near80_constants.py` lists them in `PLACEHOLDERS`, and the documents carry `TODO(DECISION.md: ...)` marks:

1. the number of fresh seeds (5, 10, 15 or 20; `FRESH` lists the first five of the block);
2. C2's margin on the paired net across worlds, in points;
3. C2's least number of seeds with a positive paired difference;
4. C2's multiplicity rule: `all_three` (one claim, every twin beaten) or `each_separately` (one claim per twin);
5. your acceptance of section 16's defaults (`OWNER_ANSWER`).

`V9_BUILD_NOTES.md` lists where each is read.

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

## How to run on the desktop (after the import)

Inside Ubuntu as root, `.../run_near80.sh` standing for `sh /home/user/bca/outputs/shift/beyond-the-data/near80/run_near80.sh`:

1. `git -C /home/user/bca pull --ff-only`
2. `.../run_near80.sh selftest` must end with `READY`.
3. In tmux (`tmux new -s near80`): `.../run_near80.sh plumbing --lanes 10` (about 2.5 hours) must end with
   `plumbing PASSED`.
4. In tmux: `.../run_near80.sh run --lanes 10`. Ctrl-C once stops safely; the same command continues after any
   restart. `.../run_near80.sh status` shows what is happening.
5. When it prints `Finished`, commit and push the results folder with the commands it prints, then
   `sh /home/user/bca/outputs/shift/beyond-the-data/near80/push_runfiles_near80.sh`.

The messages (`UNFINISHED`, `WAITING FOR RETRY`, `PAUSED`, `HALT`, `REFUSED`, `FAILED`) and what to do are as in
version 8's README; `retry <phase>`, `budget-continue fresh` and `report` are the answers.

## How to run on the cluster (hpc.charlotte.edu)

From a login node, with a clone under `/scratch/dbayha`, never as root, your own HOME, and a venv built from the
package's pins (or `module load python/3.12.3` and the pins on top):

```bash
export NEAR80_HOST=cluster NEAR80_ROOT=/scratch/dbayha/bca NEAR80_PYTHON=/scratch/dbayha/bcavenv/bin/python
sh $NEAR80_ROOT/outputs/shift/beyond-the-data/near80/run_near80_cluster.sh selftest
sh $NEAR80_ROOT/outputs/shift/beyond-the-data/near80/run_near80_cluster.sh plumbing      # one job
sh $NEAR80_ROOT/outputs/shift/beyond-the-data/near80/run_near80_cluster.sh run           # one job per seed, then the look
sh $NEAR80_ROOT/outputs/shift/beyond-the-data/near80/run_near80_cluster.sh status
```

`run` submits one job per fresh seed (its 17 phases) and a last job for the lock and the look, all under account
`staruser` on partition `Orion`, 10 processors and 24 GB each, 30-day limit. The jobs share one output root; each
runner starts only its own seed's phases. `results` and `push_runfiles_near80.sh` then run on the login node with
your GitHub sign-in. The header of `run_near80_cluster.sh` lists what the desktop entry assumed and the cluster
entry changes (root, the path, tmux, the memory rule's source, version 7's placement).

## Files

- `PREREGISTRATION.md`: the full pre-registration (DRAFT).
- `prereg_draft.json`: the machine-readable draft; `code/freeze.py write` turns it into `prereg.json` once DECISION.md
  has fixed the open items.
- `approve_local_near80.py`, `run_near80.sh`, `run_near80_cluster.sh`, `near80_slurm_job.sh`,
  `push_runfiles_near80.sh`: the approval script and the entries.
- `code/`: the pooled groups and the count (`near80lib.py`, `facts_near80.py`), the approval chain
  (`approval_check.py`, `import_approval.py`, `near80_prereg.py`, `freeze.py`), the runner (`near80_runner.py`,
  `phases.py`, `phase_main.py`, `settingdata.py`, `learner_phases.py`, `policy_phase.py`, `evaluate_phase.py`,
  `ledger.py`, `stubs.py`), the look (`pooled.py`, `claims_text.py`), the constants (`near80_constants.py`) and five
  unit test files (`test_near80.py`, `test_approval.py`, `test_runner.py`, `test_pooled.py`, `test_v9.py`).
- `V9_BUILD_NOTES.md`: every change, test, estimate and open item of this build.
- Version 8's history: its approval and import records, `amendment_1.json`, `pre_approval/`,
  `results-20261020T115246Z/`, and (after the freeze) `prereg_history/prereg_superseded_51e677ab76c8d2f7.json` and
  `code_manifest_history/code_manifest_v6.json`.

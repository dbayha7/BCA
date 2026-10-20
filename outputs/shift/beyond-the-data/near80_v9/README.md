# The deciding test (version 9, near-80): read me first

**DRAFT, not frozen, awaiting the owner. Nothing has been trained.**

## What this is

Version 8's claim 4 (step 2 confirmed in the main setting, fresh seeds 10-14) failed because BCA refused on three of
its five fresh stalling-world seeds: its held-out full-throttle group had 18-22 rows, below the about 29 it needs, and
the refusal stopped every arm of those seeds, the rivals included (`decide/refusal.md`). This folder is the draft of
the version that fixes that and asks the question once more, on seeds nobody has seen. It is a draft: every value
your decision fixes (`DECISION.md`) is an open marker, and the freeze refuses until each one is filled.

## The one question

In the main test's setting, with BCA's two groups built from every held-out recorded row outside the ball, does BCA's
switched ramp meet step 2's three marks on fresh seeds (C1), and does it beat the plain rules that read the same
groups (C2)?

## What it does

- **The main setting only.** The main test's removal of the densest 15% of commands; no band, no random rows.
- **Pooled groups.** BCA's full-throttle and rest groups take every held-out recorded row outside the ball (the whole
  calibration split and the whole evaluation split), fixed in advance. Every switch arm reads the same groups. No
  score reads them: scores come from the four fixed test start sets.
- **Counted first.** The pooled full-throttle group is counted from commands alone before any training, and every look
  writes the count first.
- **A refusal fails only BCA's arms.** Where BCA still refuses, its height arms and the rivals matched to its strength
  fail that seed; everything else trains and is scored.
- **Plain twins.** The mean-difference switch (0.10 margin) with the ramp, and the same switch turning on the hand-set
  range limit, are trained arms.
- **Claims.** C1: version 8's claim 4, marks unchanged. C2: BCA's switched ramp against the split-conformal switch
  and both plain twins, paired by seed; its margins, test and multiplicity rule are open until `DECISION.md`.
- **Seeds.** The first 5, 10 or 15 of 35-49 (open). 0-4 for the plumbing and the no-training checks only; 5-34,
  100-119 and 900-909 refused.
- **Cost.** About 1.6 CPU-hours per fresh seed plus about 2.5 for the plumbing: about 3, 5-5.5 or 7.5-8 hours of wall
  time for 5, 10 or 15 seeds, plus 1-2.5 hours of plumbing (`PREREGISTRATION.md` section 13).

## What is still open

Every open marker: the number of fresh seeds, C2's margins, statistic, alpha, seeds-ahead rule,
multiplicity rule and noise floor, the decision rule, and your answer (text, time and place). `code/freeze.py check`
lists each one.

## How it will be approved (after the freeze, not now)

As version 8. On the Framework Desktop, inside Ubuntu as root:

```bash
cd /home/user/bca
git pull --ff-only
python3 -I outputs/shift/beyond-the-data/near80_v9/approve_local_near80_v9.py --check
python3 -I outputs/shift/beyond-the-data/near80_v9/approve_local_near80_v9.py
```

The first run only checks and shows the summary. The second writes `approval_near80v9_<first 16>.json` under your
own git identity. Then the session imports it. Nothing trains before the import record is pushed.

## How it will run (after the import)

```bash
tmux new -s near80v9
sh /home/user/bca/outputs/shift/beyond-the-data/near80_v9/run_near80_v9.sh plumbing
sh /home/user/bca/outputs/shift/beyond-the-data/near80_v9/run_near80_v9.sh run
sh /home/user/bca/outputs/shift/beyond-the-data/near80_v9/run_near80_v9.sh results
```

## Files

- `PREREGISTRATION.md`: the plan, the claims and every sentence the looks will write.
- `prereg_draft.json`: the machine-readable draft.
- `code/`: version 9's code, copied from version 8's and changed as `near80/V9_BUILD_NOTES.md` lists.
- `approve_local_near80_v9.py`, `run_near80_v9.sh`, `push_runfiles_near80_v9.sh`.

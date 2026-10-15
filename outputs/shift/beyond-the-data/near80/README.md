# Hiding the commands near 80% (version 8, near-80): read me first

**DRAFT for the owner's approval. Nothing has been trained.**

## What this is

Your comment on the stall chart asked that the values close to 80% be "deliberately left out of the training data
and introduced during testing so we can really see OOD" (out of distribution). You then said yes to a new version of
the main test with a band around 80% removed. This folder is the draft pre-registration for that version: the plan,
its kill rules, its pass marks and the exact sentences it will write, all fixed before anything is trained.

## The one question

When the good commands just below the 80% stall are hidden from training, does BCA still let the controller use them
while keeping it from going past 80%? And does it do better than simply staying inside the recorded data?

## What the draft proposes

- **Hide the band.** Every recorded row in which any thruster's throttle is from 70% to 90% leaves the training rows
  and BCA's calibration bank. This comes on top of the main test's removal of the densest 15% of commands. Held-out
  rows and test episodes are never touched.
- **What that really hides.** After the main removal, 80%-90% is already empty. So the band hides 70%-80%: about
  1,300-1,500 rows of normal driving per seed (3.1-3.6% of the kept rows). The data then stop at 70%, while the best
  command in strong current is exactly 80%.
- **Why it is a real test.** In the main test, a limit to the recorded range scored 93.1 points in strong current,
  because unseen and harmful were nearly the same. Now a rule that stays inside the data loses 17.5-19.0 points
  (seeds 0-4, nothing trained).
- **What is tested.** BCA's switched ramp (restructured BCA, step 2) on TD3+BC, with step 2's rivals, controls and
  pass marks. Every controller also gets two readouts that answer your question: what the throttles it never saw
  earned it, and what going past 80% cost it. A control removes the same number of random rows, drawn from calm and
  from hard water in the same numbers as the band's (most band rows are hard-water rows), so the band's own effect can
  be read.
- **Kill rules come first.** Ten checks that train nothing, then stop rules on seeds 0-4 (enough harm; the band
  changes what a range limit can do). The claims are read only on fresh seeds 10-14.
- **Step 2's confirmation stays separate.** The main setting on seeds 10-14 is its own claim. The near-80 stops and
  checks never cancel or fail it.
- **Cost.** About 28-40 CPU-hours: about 8-11 hours of wall time on your Framework Desktop, in three sittings (version
  7's memory rule lets at most 2 policy phases run at once). Version 6 was estimated at 72-92 process-hours.

## What you need to decide

Section 16 of `PREREGISTRATION.md` lists 15 choices, each with a recommended default. The ones that matter most:
1. **Which thrusters:** any thruster (default) or thruster 1 only.
2. **The edges:** 70% to 90% (default), or narrower or wider.
3. **The hard-water floor:** 35% (default; set after seeing seeds 0-4), or the gate's 40%, which fails two seeds.
4. **Also confirm step 2 on the same fresh seeds 10-14:** yes (default, a separate claim), or keep those seeds for
   this test alone.
5. **After a poor BCA result on seeds 0-4:** still run seeds 10-14 (default), or stop.

You can simply say "accept the defaults", or name the choices you want changed.

## How to approve (not yet)

A draft cannot be approved. After your answers, the session writes the final version and its approval script. Then,
as in versions 6 and 7:
1. Pull the branch on your own machine.
2. At your own terminal, run `python3 outputs/shift/beyond-the-data/near80/approve_local_near80.py`.
3. Type the first 16 characters of the fingerprint it shows.
4. It writes `outputs/shift/beyond-the-data/near80/approval_near80_<first 16 characters>.json` (and a `.sha256`
   file). Commit and push them under your own GitHub identity.

No agent writes that file or makes that commit, and no message counts as approval. Nothing trains until the session
has imported your approval and pushed its import record.

## Files

- `PREREGISTRATION.md`: the full draft (the facts on seeds 0-4, arms, tests, marks, claims, compute, choices).
- `prereg_draft.json`: the same, machine-readable (status "draft").
- `code/near80lib.py`: the band, the removal, the random-rows control (drawn by kind of water), the leak check, the
  training-file check and the no-training checks.
- `code/facts_near80.py`: the no-training facts script (seeds 0-4 only).
- `code/test_near80.py`: 44 unit tests on synthetic logs (a few seconds).

# The deciding test (version 9, near-80): read me first

**DRAFT, not frozen, awaiting the owner. Nothing has been trained.**

Until the package is frozen its code and documents may still change. Its claims, marks, bar, multiplicity rule,
imputation rule and decision table may not: the decision note (`DECISION.md`, draft 2) fixed them, and
`PREREGISTRATION.md` writes them out word for word.

## What this is

Your request (20 October 2026): "WE ARE LONG OVERDUE SOLVING THE BIG MAIN PROBLEM OF BCA WORKING OR NOT. TOP PRIORITY
NOW". This folder is the pre-registration of the test that settles it on ThrusterChain's main setting. It holds the
plan, the stops, the pass marks, the decision table and the exact sentences the look will write, all fixed before any
fresh seed exists.

Why a new version: version 8's claim 4 (step 2's result on fresh seeds 10-14) failed by its own rule, and the cause
was BCA's own refusal, not a low score. On 3 of 5 seeds BCA's held-out full-throttle group had 19, 22 and 18 rows,
where about 29 are needed. BCA refused to set the ramp's height, and the code stopped every arm of those seeds, the
rivals included. Where BCA could act, it worked, but a plain rule with no BCA in it did as well or better. No test has
yet shown that BCA works.

## The one question

On fresh seeds, and with enough held-out rows that BCA can act, does BCA's switched ramp work? Does BCA beat the same
pipeline with BCA's part swapped for a plain switch turning on a hand-set limit? Does its height beat plain split
conformal's? (On this task, with equal weights, no test can credit BCA's Bayesian part itself.)

## What is new

- **Main setting only**, both worlds, on **15 fresh seeds (35-49)** that nobody has generated or seen. No band, no
  random rows, no exploratory training stage.
- **More held-out rows.** BCA's two groups read every held-out recorded row outside the ball: the calibration split
  and the evaluation split. No critic or actor trains on them, and no score reads them (scores come from the four
  fixed test start sets).
- **Counted first, topped up if needed.** Before any training, the full-throttle group is counted per seed-world from
  the stored commands alone (no reward, no outcome). If it has fewer than 36 rows, BCA's own "certifiable" size,
  blocks of 100 new held-out episodes are added until it has 36, at most 10 blocks. On the records this is rarely
  needed: 1 of 50 seed-worlds was below 36, at 35 rows.
- **A refusal fails only BCA.** If BCA still refuses on a seed, its arms that need the height get no number there and
  claim 1 fails. Every other arm still trains and is scored. It is never a stop, and the seed is never replaced.
- **Two new paired comparisons** beside the old marks (claims 2a and 2b). They are computed from arms the run trains
  anyway, plus two reported-only dose arms (the plain switch at fixed heights 2.5 and 5.0).
- **The split-conformal arm uses the textbook rank**, the fairest plain twin.
- **Missing seeds.** A number missing for a reason of BCA's own fails claim 1. A number missing for a reason that is
  not BCA's (a failed data check, a plumbing stop, a rival-side failure) is replaced by the least favourable value seen
  on the other seeds, on at most 2 of 15 seeds.
- **A validity condition** on the main setting itself: the harm it was built to test must be there. The no-caution
  actor must lose at least 30 points to the full-log actor (on record about 98-99), and must go past 80% on at least
  25% of steps (on record 73%).
- **Checks before you are asked.** A no-training preview in the cloud, pinned by fingerprint; the plumbing ties the
  frozen code to it; the run starts only after the plumbing passes; the results step refuses if `.gitignore` would drop
  a file it wrote.

What was seen before freezing (no actor trained) is in `PREREGISTRATION.md` section 4. In short: a first preview on
the seen seeds 0-4 and the spent seeds 10-14 found no refusal with the pooled rows (6 of 20 seed-worlds refused with
version 8's rows), the right on/off call by BCA in 20 of 20, and smaller BCA heights (mean 2.78 to 1.69 on seeds 0-4).

## The three claims, in plain words

1. **Claim 1: does BCA work?** Step 2's three marks, unchanged, on the fresh seeds: BCA's switched ramp must net at
   least 10 points more across the two worlds than the best rule that cannot switch itself off; where full throttle
   stalls, lose at most 3 points to no caution in the other three situations, and beat the matched even pull by at
   least 15 in strong current. All three must pass.
2. **Claim 2a: does BCA's height beat plain split conformal's?** The same ramp, the same rows and the same 0.10
   margin, with only BCA's certificate swapped for the textbook 90% point. "Shown" needs an average lead of at least 5
   points with the lower end of its 95% interval above 0; otherwise "not shown". A pass credits a larger height (a
   dose), never BCA's Bayesian part.
3. **Claim 2b: is BCA needed?** BCA against a plain mean-difference switch on the same rows and margin, turning on the
   hand-set range limit: the best rule without BCA on record. "Shown" as in claim 2a; "excluded" if the upper end of
   the interval is below 5, which rules a 5-point BCA lead out; "unresolved" otherwise. "Shown" is practically
   unreachable on this task (BCA would need about 98-100 of 100 points), and that is stated before the run.

## The decision rule

Exactly one row of the decision table applies, chosen in a fixed order. First come the rows where no conclusion is
drawn: the plumbing fails, the main setting does not produce its harm, or claim 1 cannot be read for reasons that are
not BCA's. Then claim 1: if it fails for a reason of BCA's own (a missed mark, a refusal, BCA's switch staying off
where full throttle stalls), the verdict is "BCA does not work on ThrusterChain's main setting" or the refusal's or the
switch's sentence. If claim 1 passes, claims 2a and 2b pick one of six rows, such as "BCA works on ThrusterChain and
beat the best plain rule" or "BCA works as a harm switch on ThrusterChain, but nothing specific to BCA earned credit".
The expected outcome, written before any run, is "BCA works as a harm switch, but is not needed". Every row's
sentence is fixed now: `PREREGISTRATION.md` section 14 (the decision note's section 3 adds what the project does next
in each row).

## What you decided

You said "build now" and accepted the defaults of the decision note's section 4.4: 15 fresh seeds (35-49); both
held-out splits pooled plus the top-up to 36; version 8's memory rule; the two dose arms; no exploratory retrain of
seeds 0-4; the branch `claude/near80-v9`. This is not the approval. The approval is the fingerprint you type at your
own terminal, after the freeze ("How to approve").

## Before you are asked to approve

The session does each of these in the cloud, with no actor trained and nothing for you to do:
1. The code takes every choice of the decision note: the top-up, the arm set, the refusal rule, the claims, the
   imputation rule and the stop table.
2. The tests, and stub runs of the whole plan on fixtures, including a forced refusal, a forced top-up, a forced
   rival-side failure and a forced plumbing stop.
3. The pinned preview on the seen and spent seeds, with this version's code. If BCA's switch makes a wrong call on any
   of the 20 seed-worlds, or a group stays below 36 rows after the top-up, you are not asked, and only the evidence
   rule may be redrafted.
4. The freeze, then an independent review in a separate session told to refute the package, re-freezing until it
   passes. Only then does the session say "ready for approval".

## Cost

About 26 processor-hours: **about 9 hours of wall time on your Framework Desktop, in one sitting** after the approval.
One command runs the plumbing (about 1 hour) and then the 15 fresh seeds (about 7.6-8 hours). Version 8's memory rule
lets at most 2 actor phases run at once.

The 10-seed option (seeds 35-44) takes about 2.5 hours less: about 5.1-5.4 hours plus the plumbing. Its price: claim
2a's chance of "shown" near the bar is a few points lower (0.84 against 0.89 at a true lead of 6), and 1 missing seed
is tolerated instead of 2. You chose 15. To change that, say so before the freeze: the seed count is part of the
fingerprint you approve.

## How to approve (after the session says "ready for approval")

Keep the `-I` on every `python3` line: it stops Python from loading anything from the script's own folder before the
checks run. Never set git's name or email with `git config`: the runner would then refuse to run.

1. In PowerShell: `wsl -d Ubuntu-24.04 -u root`. Then, inside Ubuntu, type:

   ```bash
   cd /home/user/bca && git fetch origin && git switch claude/near80-v9 && git pull --ff-only
   python3 -I outputs/shift/beyond-the-data/near80_v9/approve_local_near80_v9.py --check
   python3 -I outputs/shift/beyond-the-data/near80_v9/approve_local_near80_v9.py
   ```

   The first run only checks every file against its fingerprint and shows the summary; it writes nothing. The second
   shows the same summary and asks you to type the first 16 characters of the fingerprint.
2. It writes `outputs/shift/beyond-the-data/near80_v9/approval_near80v9_<first 16 characters>.json` and its
   `.sha256` file, and prints the exact commands to commit and push them under your own name. Type the lines it
   prints. They have this form:

   ```bash
   cd /home/user/bca
   git add outputs/shift/beyond-the-data/near80_v9/approval_near80v9_<first 16 characters>.json outputs/shift/beyond-the-data/near80_v9/approval_near80v9_<first 16 characters>.json.sha256
   git -c user.name="David Bayha" -c user.email=35092758+dbayha7@users.noreply.github.com commit -m "Near-80: approve version 9 <first 16 characters>"
   git push origin claude/near80-v9
   ```

   If the push is rejected ("fetch first" or "non-fast-forward"), type the `pull --rebase` line it printed, then push
   again.
3. Tell the session you pushed. It imports your approval (`code/import_approval.py import`) and pushes its import
   record, `import_near80v9_<first 16 characters>.json` (version 8: 2 minutes).

No agent writes your approval file or makes that commit, and no message counts as approval. Nothing trains until the
session has imported your approval and pushed its import record.

## How to run (in the same sitting, after the import)

Everything below is typed inside Ubuntu as root, on the Framework Desktop. `.../run_near80_v9.sh` below stands for
`sh /home/user/bca/outputs/shift/beyond-the-data/near80_v9/run_near80_v9.sh`.

1. Start tmux, bring the import record, and start everything, all **inside tmux**:

   ```bash
   tmux new -s near80v9
   git -C /home/user/bca pull --ff-only
   sh /home/user/bca/outputs/shift/beyond-the-data/near80_v9/run_near80_v9.sh selftest
   sh /home/user/bca/outputs/shift/beyond-the-data/near80_v9/run_near80_v9.sh plumbing-then-run --lanes 10
   ```

   The self-test must end with `READY`. The last command runs the plumbing (about 1 hour) and starts the run (about 8
   hours) only if the plumbing ends with `plumbing PASSED`. If it ends with `plumbing FAILED` and does not say the
   reproduction differed, paste the lines into the session; once the cause is fixed, type the same command again.
2. To leave tmux running: press Ctrl-b, then d. To come back: `tmux attach -t near80v9`. Keep the Ubuntu window open
   (minimized is fine).
   - **To stop safely**: press Ctrl-C once. No new phase starts and the running ones finish (this can take over half
     an hour). A second Ctrl-C stops them at once; they rerun from the start next time.
   - **To continue** after a stop, a closed window or a restart of Windows or WSL, once the plumbing has passed: open
     Ubuntu as root, `tmux attach -t near80v9` (or `tmux new -s near80v9` if tmux says there is none), then inside
     tmux `sh /home/user/bca/outputs/shift/beyond-the-data/near80_v9/run_near80_v9.sh run --lanes 10`. It continues
     where it stopped.
   - `--lanes` can be 1 to 16 on any restart; it changes no result.
3. **To see what is happening**, from a second Ubuntu window:
   `sh /home/user/bca/outputs/shift/beyond-the-data/near80_v9/run_near80_v9.sh status`
4. When a message asks for it (see the table):
   - after `UNFINISHED` or `WAITING FOR RETRY`: `.../run_near80_v9.sh retry <phase>`
   - after `HALT`: `.../run_near80_v9.sh report`
   - after `PAUSED`: `.../run_near80_v9.sh budget-continue <stage>`

## After the run

1. At `Finished`, the results are gathered into `outputs/shift/beyond-the-data/near80_v9/results-<utc>/`: every
   record (no model file), each with its fingerprint in `SHA256SUMS`. A finished run does this itself and prints the
   commands to commit them; otherwise type
   `sh /home/user/bca/outputs/shift/beyond-the-data/near80_v9/run_near80_v9.sh results`. It refuses if
   `git check-ignore` matches any file it wrote (version 8's results commit lost its 90 data records to
   `.gitignore`'s `data/`).
2. Commit and push them under your own name with the commands it prints. They have this form:

   ```bash
   cd /home/user/bca
   git add outputs/shift/beyond-the-data/near80_v9/results-<utc>
   git -c user.name="David Bayha" -c user.email=35092758+dbayha7@users.noreply.github.com commit -m "Near-80: results <utc>"
   git push origin claude/near80-v9
   ```

3. Push the run files (the model files):
   `sh /home/user/bca/outputs/shift/beyond-the-data/near80_v9/push_runfiles_near80_v9.sh`. It commits them, as you,
   to the branch `claude/near80-v9-run-files` (your checkout is not touched) and retries the push up to five times.
4. Tell the session. It writes the report, with every number re-derived by reviewers told to refute it.

## If something stops

Paste what the table says into the session. **Never delete or edit anything under `/home/user/bca/runs/`.** A BCA
refusal is not a stop: the run goes on, and the look reports it.

| Message | Meaning | What to do |
|---|---|---|
| `READY` | the self-test passed | go on |
| `NOT READY (...)` with `FAIL <check>: ...` above it | a check of the self-test failed | paste the lines; the `FAIL` line says what is missing |
| `refused: git's user.email here is ...` | a git identity is set in root | type `git config --global --unset user.email`, then the same command again |
| `plumbing PASSED: ...` | the plumbing passed; `plumbing-then-run` goes on to the run | nothing to do |
| `plumbing FAILED: the reproduction check differed ...` | the new code did not reproduce a saved record (section 8.2) | nothing official can start; paste the lines and type `.../run_near80_v9.sh report` |
| `plumbing FAILED: <cause>. Recorded in ...` | a test, the smoke or the reproduction check did not finish (not a difference) | paste the lines; once the cause is fixed, type the same command again inside tmux |
| `refused: the plumbing has not passed yet ...` | `run` was typed before the plumbing passed | use `plumbing-then-run` ("How to run", step 1) |
| `refused: another near-80 runner is running on this machine` | the run is still going, perhaps in a tmux window you left | `tmux attach -t near80v9`, or `.../run_near80_v9.sh status`; do not start a second one |
| `waiting: memory (...)` | the memory rule | nothing to do |
| `skip <phase>: ...` or `STOPPED <phase>: ...` | a phase was skipped by rule, or stopped by a built-in check (written once, never rerun) | nothing to do; the look reports it |
| `interrupted: <phase> (...)` | a restart, a closed terminal or the machine stopped a phase; it reruns | nothing to do |
| `UNFINISHED: <phase>` or `WAITING FOR RETRY: <phases>` | a phase was interrupted again when run alone | paste the `tail -n 30 ...` output the message names, then `.../run_near80_v9.sh retry <phase>` |
| `PAUSED (budget): ...` | a stage used more than twice its forecast, or the run reached the cap | paste it; if you agree to go on, `.../run_near80_v9.sh budget-continue <stage>`, then the `run` command inside tmux |
| `HALT: ...` | the runner halted (a phase can never fit in memory, unexplained files, a stuck plan); nothing is deleted | paste it and `.../run_near80_v9.sh report`; do not restart until the session has looked |
| `REFUSED: <phase>: ...` or `FAILED: <phase> exit N ...` | a safety check refused a phase, or a phase exited unexplained | paste it and the `tail -n 30 ...` output; do not restart until the session has looked |
| `Stopped before the end (nothing failed)` | you pressed Ctrl-C | the `run` command again inside tmux |
| `Finished: ...` | the end | "After the run" |
| `the push failed five times` | the network or the sign-in (run files) | check the token, then `push_runfiles_near80_v9.sh` again; it rebuilds the same commit |

## Files

- `PREREGISTRATION.md`: the full pre-registration: the evidence change, what was seen before freezing, the arms, the
  tests, the marks, the stops, every claim and verdict sentence, the compute and the choices.
- `prereg_draft.json`: its machine-readable twin; the freeze writes `prereg.json`, whose fingerprint you approve.
- `approve_local_near80_v9.py`: the approval script you run at your own terminal.
- `run_near80_v9.sh`: the desktop entry (selftest, plumbing-then-run, run, status, plan, retry, budget-continue,
  report, results, stub).
- `push_runfiles_near80_v9.sh`: pushes the run files to the branch `claude/near80-v9-run-files`.
- `pre_approval/`: the outcome-free counts, the pinned preview and the frozen no-training checks.
- `code/`: version 8's code, copied and changed for version 9, with its unit tests.

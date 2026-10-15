# Hiding the commands near 80% (version 8, near-80): read me first

**FINAL, awaiting the owner's approval. Nothing has been trained.**

## What this is

Your comment on the stall chart asked that the values close to 80% be "deliberately left out of the training data
and introduced during testing so we can really see OOD" (out of distribution). You then said yes to a new version of
the main test with a band around 80% removed. This folder is the final pre-registration for that version: the plan,
its kill rules, its pass marks and the exact sentences it will write, all fixed before anything is trained. You
accepted every default of its choices on 15 October 2026.

## The one question

When the good commands just below the 80% stall are hidden from training, does BCA still let the controller use them
while keeping it from going past 80%? And does it do better than simply staying inside the recorded data?

## What it does

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
- **Cost.** About 31-43 CPU-hours: about 10-13 hours of wall time on your Framework Desktop, in three sittings (version
  7's memory rule lets at most 2 policy phases run at once). Version 6 was estimated at 72-92 process-hours.

## What you decided

You accepted every default of section 16 of `PREREGISTRATION.md` on 15 October 2026 at 19:24 UTC, in a comment on the
ThrusterChain page: "accept the defaults".

## How to approve

As in versions 6 and 7. Do it on the Framework Desktop, inside Ubuntu as root: the script checks repository files
under `runs/` that only version 7's restore put there, so it refuses from a fresh clone anywhere else.

1. In PowerShell: `wsl -d Ubuntu-24.04 -u root`. Then, inside Ubuntu, type:

   ```bash
   cd /home/user/bca
   git pull --ff-only
   python3 -I outputs/shift/beyond-the-data/near80/approve_local_near80.py --check
   python3 -I outputs/shift/beyond-the-data/near80/approve_local_near80.py
   ```

   The first run only checks every file against its fingerprint and shows the summary; it writes nothing. The second
   shows the same summary and asks you to type the first 16 characters of the fingerprint. Keep the `-I`: it stops
   Python from loading anything from the script's own folder before the checks run.
2. It writes `outputs/shift/beyond-the-data/near80/approval_near80_<first 16 characters>.json` and its `.sha256` file,
   and prints the exact commands to commit and push them under your own name. They are:

   ```bash
   cd /home/user/bca
   git add outputs/shift/beyond-the-data/near80/approval_near80_<first 16 characters>.json outputs/shift/beyond-the-data/near80/approval_near80_<first 16 characters>.json.sha256
   git -c user.name="David Bayha" -c user.email=35092758+dbayha7@users.noreply.github.com commit -m "Near-80: approve version 8 <first 16 characters>"
   git push origin claude/main-bca-connection-x4o46s
   ```

   **If the push is rejected** ("fetch first" or "non-fast-forward": the session pushed something meanwhile), type:

   ```bash
   git -c user.name="David Bayha" -c user.email=35092758+dbayha7@users.noreply.github.com pull --rebase origin claude/main-bca-connection-x4o46s
   git push origin claude/main-bca-connection-x4o46s
   ```

   Never set the name or email with `git config`: the runner would then refuse to run.
3. Tell the session you pushed. It imports your approval (`code/import_approval.py import`) and pushes its import
   record, `import_near80_<first 16 characters>.json`.
4. When the session says it has pushed the import record: `git -C /home/user/bca pull --ff-only`.

No agent writes your approval file or makes that commit, and no message counts as approval. Nothing trains until the
session has imported your approval and pushed its import record.

## How to run (after the import)

Everything below is typed inside Ubuntu as root (`wsl -d Ubuntu-24.04 -u root` in PowerShell), on the Framework
Desktop. The commands are written out in full; `.../run_near80.sh` in the table below stands for
`sh /home/user/bca/outputs/shift/beyond-the-data/near80/run_near80.sh`.

1. **Start only after version 7 has finished**: its run printed `Finished`, and its steps 13 and 14 (the results and
   the run files) are done. The near-80 runner refuses while version 7's runner runs, and refuses until version 7's
   results are gathered. If version 7 stopped for good without finishing (a halt you and the session agreed to end, or
   a stop you chose), type once: `sh /home/user/bca/outputs/shift/beyond-the-data/near80/run_near80.sh version7-ended`
   (it is recorded, write-once).
2. Bring the import record: `git -C /home/user/bca pull --ff-only`.
3. The self-test: `sh /home/user/bca/outputs/shift/beyond-the-data/near80/run_near80.sh selftest`. It must end with
   `READY`.
4. The plumbing (about 2-2.5 hours). Start tmux, `tmux new -s near80`, then **inside tmux** type:

   ```bash
   sh /home/user/bca/outputs/shift/beyond-the-data/near80/run_near80.sh plumbing --lanes 10
   ```

   It must end with `plumbing PASSED`. If it ends with `plumbing FAILED` and does not say the reproduction differed,
   paste the lines into the session; once the cause is fixed, type the same command again (it starts afresh).
5. The run, **inside tmux** (the same tmux window is fine):

   ```bash
   sh /home/user/bca/outputs/shift/beyond-the-data/near80/run_near80.sh run --lanes 10
   ```

6. To leave tmux running: press Ctrl-b, then d. To come back: `tmux attach -t near80`. Keep the Ubuntu window open
   (minimized is fine).
   - **To stop safely**: press Ctrl-C once. No new phase starts and the running ones finish (this can take over half an
     hour). A second Ctrl-C stops them at once; they rerun from the start next time.
   - **After a closed window**: open Ubuntu as root, then `tmux attach -t near80`. If tmux says there is none (`no
     server running`, `no sessions` or `can't find session`), the run has stopped: type `tmux new -s near80`, then
     inside tmux the same `run` command. It continues where it stopped.
   - **After a restart of Windows or WSL**: open Ubuntu as root, `tmux new -s near80`, then inside tmux the same `run`
     command.
   - `--lanes` can be 1 to 16 on any restart; it changes no result.
7. **To see what is happening**, from a second Ubuntu window:
   `sh /home/user/bca/outputs/shift/beyond-the-data/near80/run_near80.sh status`
8. When a message asks for it (see the table):
   - after `UNFINISHED` or `WAITING FOR RETRY`:
     `sh /home/user/bca/outputs/shift/beyond-the-data/near80/run_near80.sh retry <phase>`
   - after `HALT`: `sh /home/user/bca/outputs/shift/beyond-the-data/near80/run_near80.sh report`
   - after `PAUSED`:
     `sh /home/user/bca/outputs/shift/beyond-the-data/near80/run_near80.sh budget-continue <stage>`

## After the run

When the run prints `Finished`, it has already gathered the results into
`outputs/shift/beyond-the-data/near80/results-<utc>/` (every record with its fingerprint in `SHA256SUMS`; no model
file) and printed the commands to commit them. They are:

```bash
cd /home/user/bca
git add outputs/shift/beyond-the-data/near80/results-<utc>
git -c user.name="David Bayha" -c user.email=35092758+dbayha7@users.noreply.github.com commit -m "Near-80: results <utc>"
git push origin claude/main-bca-connection-x4o46s
```

If the push is rejected, use the `pull --rebase` line of "How to approve", then push again. Then push the run files
(the model files; an upload of perhaps 0.3-0.6 GB):

```bash
sh /home/user/bca/outputs/shift/beyond-the-data/near80/push_runfiles_near80.sh
```

It commits them, as you, to the new branch `claude/near80-run-files` (your checkout is not touched) and retries the
push up to five times. Then tell the session. (Only when the session asks for results before the end:
`sh /home/user/bca/outputs/shift/beyond-the-data/near80/run_near80.sh results` gathers them as they are.)

## If something stops

Paste what the table says into the session. **Never delete or edit anything under `/home/user/bca/runs/`.**

| Message | Meaning | What to do |
|---|---|---|
| `READY` | the self-test passed | go on |
| `NOT READY (...)` with `FAIL <check>: ...` above it | a check of the self-test failed (the Python setup, the approval, the placement of version 6's code, git's identity, version 7, the saved inputs) | paste the lines; the `FAIL` line says what is missing |
| `refused: git's user.email here is ...` | a git identity is set in root | type `git config --global --unset user.email`, then the same command again |
| `refused: version 7's runner is running ...` | version 7 is still running | wait until it prints `Finished`; do its steps 13-14 |
| `refused: version 7's run has not finished ...` | version 7's results are not gathered | finish version 7 (its steps 13-14); only if it stopped for good, type `.../run_near80.sh version7-ended` |
| `waiting: version 7 is running ...` | version 7's runner or one of its phases started while near-80 runs | nothing starts until it ends; stop one of the two (Ctrl-C in its tmux window) and paste the message |
| `refused: the plumbing has not passed yet ...` | `run` was typed before the plumbing passed | do step 4 of "How to run" |
| `plumbing PASSED: ...` | the plumbing passed | step 5 |
| `plumbing FAILED: the reproduction check differed ...` | with the band empty, the new code did not reproduce a saved file | the end of this version (section 8.2); paste the lines and type `.../run_near80.sh report` |
| `plumbing FAILED: <cause>. Recorded in ...` | a test, the smoke or the reproduction check did not finish (not a difference; also a saved file or training file that fails its fingerprint check inside the reproduction check) | paste the lines; once the cause is fixed, type the same `plumbing` command again inside tmux |
| `refused: ... reproduction_differs.json exists ...` | the reproduction check differed earlier | nothing official can start in this version; paste it |
| `refused: another near-80 runner is running on this machine` | the run is still going (perhaps in a tmux window you left) | type `tmux attach -t near80`, or `.../run_near80.sh status`; do not start a second one |
| `refused: phases from an earlier run are still running` | the runner stopped, but some of its phases still run | wait; check with `.../run_near80.sh status`; then the same `run` command again |
| `waiting: memory (...)` | the memory rule | nothing to do |
| `skip <phase>: ...` or `STOPPED <phase>: ...` | a phase was skipped by rule, or stopped by a built-in check (written once, never rerun) | nothing to do; the looks report it |
| `the exploratory look stops the near-80 settings on seeds 10-14` | a stop rule of section 8.3 applied | nothing to do; the main setting still runs on seeds 10-14 |
| `interrupted: <phase> (...)` | a restart, a closed terminal or the machine stopped a phase; its files were moved aside and it reruns | nothing to do |
| `waiting to run alone: <phase>` | a third interruption of that phase; it runs alone once nothing else is runnable | nothing to do |
| `UNFINISHED: <phase>` | it was interrupted again when run alone; it waits; every other phase continues | paste the output of the `tail -n 30 ...` command the message gives, then type `.../run_near80.sh retry <phase>` (in a second window while the run continues, or inside tmux if it has stopped) |
| `WAITING FOR RETRY: <phases>` | nothing else can run until an unfinished phase is done | as for `UNFINISHED` |
| `PAUSED (budget): ...` or `paused: ...` | a stage, or the whole run, used more than the budget guard allows (section 13); the run stops starting phases (exit 5) | paste it; if you agree to go on, type `.../run_near80.sh budget-continue <stage>` (the stage the line names), then the same `run` command inside tmux. Each continuation is recorded and allows one more forecast of that stage |
| `HALT: <phase> reserves ... it can never start` | WSL has too little memory for this phase | paste it; give WSL more memory (Windows settings) and restart WSL, then the same `run` command |
| `HALT: a no-training check failed on seeds 0-4 ...` | a check of section 8.1 failed with the frozen code | type `.../run_near80.sh report` and paste its sentence |
| `HALT: <phase> has files that no attempt of this runner wrote ...` | files appeared under `runs/` that the runner did not write | paste it; do not restart until the session has looked; delete nothing |
| `REFUSED: <phase>: ...` or `REFUSED (exit 97): <phase>: ...` | a safety check refused a phase | the run halts; paste it and the `tail -n 30 ...` output |
| `FAILED: <phase> exit N ...` | an unexplained exit; no new phase starts | paste the output of `tail -n 30` of the log it names; do not restart until the session has looked |
| `Stopped before the end (nothing failed)` | you pressed Ctrl-C | type the same command again inside tmux to continue |
| `Finished: ...` | the end | "After the run" |
| `the push failed five times` | the network or the sign-in (run files) | check the token, then `push_runfiles_near80.sh` again; it rebuilds the same commit |

## Files

- `PREREGISTRATION.md`: the full pre-registration (the facts on seeds 0-4, arms, tests, marks, claims, compute,
  choices).
- `prereg.json` (+ `.sha256`): the same, machine-readable, status "final, awaiting approval"; its fingerprint is what
  you approve.
- `code_manifest.json` (+ `.sha256`): the fingerprints of the code, the documents and the repository files the code
  uses.
- `approve_local_near80.py`: the approval script you run at your own terminal.
- `run_near80.sh`: the desktop entry (run, plumbing, status, self-test, plan, retry, budget-continue,
  version7-ended, report, results, stub).
- `push_runfiles_near80.sh`: pushes the run files (model files included) to the branch `claude/near80-run-files`.
- `prereg_history/`: the draft is kept in `prereg_history/`, unchanged, with its fingerprint.
- `pre_approval/no_training_checks_seeds0_4.json`: the frozen code's no-training checks on seeds 0-4 (nothing
  trained).
- `code/`: the band and its checks (`near80lib.py`, `facts_near80.py`), the approval chain (`approval_check.py`,
  `import_approval.py`, `near80_prereg.py`, `freeze.py`), the runner (`near80_runner.py`, `phases.py`,
  `phase_main.py`, `settingdata.py`, `learner_phases.py`, `policy_phase.py`, `evaluate_phase.py`, `v6bca.py`,
  `ledger.py`, `stubs.py`), the looks (`pooled.py`, `claims_text.py`), the shared constants (`near80_constants.py`)
  and four unit test files (`test_near80.py`, `test_approval.py`, `test_runner.py`, `test_pooled.py`).

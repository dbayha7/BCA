# Stage 2, version 7: finishing the run on your laptop

**"The laptop" now means your Framework Desktop.** Version 7 was first written for your Lenovo laptop. The run has
moved to your Framework Desktop (`PREREGISTRATION_V7.md`, change C12). The names are kept: wherever this package says
"the laptop" (also in the folder name `laptop/` and in the code), it means the Framework Desktop.

**Steps 0-7 are done.** Claude (the Claude Code session on your desktop) did them on the Framework Desktop on 14 Oct
2026: WSL has 24 GB (step 0), `Ubuntu-24.04` is installed (step 1), and the restore (step 7) printed `PASS restore:
2370 run files of 901c87766ab0 (pre-handoff snapshot) match the pinned sums`. Step 8 comes next, once this amended
draft is committed and pushed (the self-test's records bind to the committed draft).

All seven questions of the plan are answered: the world's numbers from the cloud's file, Ubuntu 24.04, 12 lanes (16
at most), re-made reference copies when the laptop does not reproduce the saved ones, the cloud running until the
handoff, pushing the laptop's run files, and **Question 7: stop**. That is: if the self-test forecasts that check (b)
would stop seeds for a reason that lies in the laptop's arithmetic (words below), version 7 stops before anything
starts. You answered "stop" on 14 Oct at 23:27 UTC; the draft records your words and the time. Once the full
self-test of step 8 has written its first probe record, the answer can no longer change. Still wanted before step 8:
whether running cloud phases stop at once or finish at the handoff (default: stop at once).

## What this is

Stage 2 is the test of BCA pre-registered in `outputs/shift/beyond-the-data/stage2/PREREGISTRATION.md`, version 6,
which you approved (commit `7dab7d5`). Version 6 stays the governing document. Version 7 ("v7") changes only how and
where the remaining work runs: on your laptop instead of the cloud. Its declaration is
[`PREREGISTRATION_V7.md`](PREREGISTRATION_V7.md); read it before step 10.

Words used below:

- **The laptop**: your local machine, where version 7 runs: now your Framework Desktop (see the top of this page).
- **Phase**: one step of the pre-registered work, run as one Python process (for example "train the policy test on
  seed 6").
- **Lane**: a slot in which one phase runs, on its own processor. 12 lanes means 12 phases at once.
- **WSL** (Windows Subsystem for Linux): runs Ubuntu Linux inside Windows. Everything from step 2 on runs there.
- **tmux**: a program that keeps a command running inside Ubuntu after you close or leave its window.
- **The cloud**: the Claude session that has run Stage 2 so far.
- **Handoff**: the moment the cloud stops running Stage 2 and the laptop takes over (step 9).
- **Self-test**: a check of the laptop that runs no part of the study itself (step 8). Its **full** form runs once
  (step 8); its **short** form runs before every run.
- **Probes**: short trainings inside the full self-test that check whether the laptop reproduces the cloud's saved
  results bit for bit (they run no BCA). **P1** and **P2** compare the laptop's copies of three saved "reference"
  controllers with the cloud's; **P1b** runs the closest model of check (b) without BCA; **P3** runs the same short
  training six times, with the laptop idle and busy, on two processors, and needs identical results each time.
- **Check (b)**: the policy test's stop rule. BCA with all its settings at 0 must give the same controller as the
  saved "none" controller (no caution), to within 0.00001 in every number; otherwise that seed stops for good.
- **Check (b) forecast**: the self-test's measurement, before anything is approved, of how far apart the laptop's own
  programs come in check (b)'s comparison. Above 0.00001 it says `NOT READY` and version 7 ends (your answer to
  Question 7 was "stop").
- **Anchors**: 64 numbers recorded by the earlier ThrusterChain test that define 0 and 100 points on the score scale.
  Every phase recomputes them and requires them to match exactly.
- **Rival method** (also "stand-in"): the strongest of the other methods BCA is compared with in the policy test
  (`range_hinge`). A **stand-in error** is a failure of that rival recorded in a policy lock.
- **Policy lock**: the policy test's saved record of the trained controllers and settings (`lock.json`).
- **Deployment check**: the test of a rule that, while the controller runs, replaces its command by a limited one
  when BCA's width (its measure of how unfamiliar the situation is) is above a threshold.
- **Full host**: the second training test: BCA added to a complete TD3+BC training run (the controller and its critic
  both trained), compared with the plain run and with rival runs, in both worlds.
- **Pooled look**: the step that gathers all seeds' results into the final numbers: "exploratory" for seeds 0-4,
  "confirmatory" for seeds 5-9 (only these carry the claim).
- **Fingerprint**: a SHA-256 checksum; it changes if a single byte of a file changes.
- **Personal access token**: a password-like key that GitHub makes for you, used instead of your password (step 4).

Why v7 is needed: every phase starts with the earlier ThrusterChain test's own exact checks. On a processor without
AVX-512 (a set of wide arithmetic instructions; your Lenovo laptop's Core Ultra 9 275HX has none) three numbers of
the simulated world come out different in their last digits, and version 6 refuses to run. Version 7 supplies those
numbers from a file made in the cloud, so every check still runs exactly (`PREREGISTRATION_V7.md`, change C1).

Your Framework Desktop's processor (AMD Ryzen AI MAX+ 395) does have AVX-512, and version 6's own self-test already
passed this check on it (`PASS anchors`). There the file is expected to change nothing; it is kept so that everything
runs one way (change C12). Version 7 is still needed for the rest: up to 16 phases at once instead of 2, the rules for
interrupted phases, the handoff, and the checks for continuing seeds 0-4 on a second machine.

## The whole sequence

1. Done: steps 0-7 (Windows settings, Ubuntu, the tools, the repository, Python, the restore), on the Framework
   Desktop. The cloud has finished every phase of seeds 0-4 (14 Oct, 09:21 UTC).
2. Next: this amended draft is committed and pushed (and the cloud pulls it); then step 8 (the full self-test).
3. The handoff (step 9). From then on the cloud runs no Stage 2 phase, unless v7 is withdrawn.
4. Your approval and its import (steps 10-11).
5. The run (step 12).
6. The results, the run files and the cloud's check (steps 13-14).

Steps 0 and 1 are in **Windows PowerShell**. Everything else is typed **inside Ubuntu, as root** (the `#` prompt).
Type the commands **one line at a time**. Times are rough.

## Step 0. Windows settings (5 minutes)

(Done on 14 Oct: your `.wslconfig` holds `memory=24GB` and `swap=16GB`.) Your Framework Desktop has 64 GB, but 32 GB
of it is reserved for the built-in graphics, so Windows sees 31.8 GB and WSL gets 24 GB. If you later lower that
reservation, you may write `memory=48GB` instead; nothing else changes.

Open PowerShell (not as administrator is fine) and paste:

```powershell
@"
[wsl2]
memory=24GB
swap=16GB
"@ | Set-Content -Encoding ascii "$env:USERPROFILE\.wslconfig"
Get-Content "$env:USERPROFILE\.wslconfig"
```

Then, in Windows Settings: System, Power: never sleep when plugged in. (On a laptop, also set, in Control Panel, Power
Options, "Choose what closing the lid does", "When I close the lid" to **Do nothing** for "Plugged in"; the Framework
Desktop has no lid.) Windows Update: pause updates for a week. Keep the machine plugged in for the whole run.

## Step 1. Install Ubuntu 24.04 in WSL (15-30 minutes, with a restart)

(Done on 14 Oct: the new distribution `Ubuntu-24.04` is installed, Ubuntu 24.04.5 with glibc 2.39, on WSL 2.6.3.)

In PowerShell **as administrator**, type these one at a time. First:

```powershell
wsl --install -d Ubuntu-24.04
```

Restart Windows if asked. When Ubuntu opens the first time it asks for a Linux user name and password; choose any.
When it then shows its prompt (ending in `$`), type `exit` to return to PowerShell. Then:

```powershell
wsl --update
wsl -l -v
wsl --shutdown
```

`wsl -l -v` must show `Ubuntu-24.04` with VERSION `2`. An Ubuntu-22.04 you already have can stay; it is not used.
`wsl --shutdown` stops WSL so that the 24 GB memory setting of step 0 takes effect the next time Ubuntu opens.

## Step 2. Open Ubuntu as root (1 minute)

In PowerShell:

```powershell
wsl -d Ubuntu-24.04 -u root
```

Then, inside Ubuntu:

```bash
cd /root
free -g
```

In the `free -g` output, the `total` on the `Mem:` line must be about 23 (about 47 if you wrote `memory=48GB`). If it
is about 15, the setting of step 0 is not in effect: type `exit`, run `wsl --shutdown` in PowerShell, and open Ubuntu
again with the line above. Every later step runs in this window (or another one opened the same way).

## Step 3. Tools (5 minutes)

```bash
apt-get update && apt-get install -y git curl tmux ca-certificates
```

## Step 4. The repository (5 minutes, about 80 MB)

The repository is private, so git needs a personal access token. Make one on GitHub (Settings, Developer settings,
Personal access tokens). It must allow **pushing**, because steps 10, 13 and 14 push: a "classic" token with the
`repo` box ticked, or a "fine-grained" token for the `bca` repository with "Contents" set to "Read and write".
Then:

```bash
git config --global credential.helper 'cache --timeout=604800'
mkdir -p /home/user && git clone --single-branch --branch claude/main-bca-connection-x4o46s https://github.com/dbayha7/bca.git /home/user/bca
```

When git asks, give your GitHub user name and, as the password, the token. The first line makes git keep them in
memory only, for up to 7 days, so later steps do not ask again; nothing is written to disk. After a restart of
Windows or `wsl --shutdown`, git asks once more (the restore in step 7 also works when git asks).
Do not add `--depth`. Do not set a git name or email in this root account: the code refuses to run under the
identity of your approval commit (steps 10 and 13 show how to commit anyway).

## Step 5. Python and libraries (10-15 minutes)

Exactly as `stage2_gpu/README.md` step 6 (its lines 109-121). In the same window:

```bash
curl -LsSf https://astral.sh/uv/0.11.32/install.sh | sh
. /root/.local/bin/env
uv --version
uv python install 3.10.20
uv venv --python 3.10.20 /root/bcavenv
uv pip install --python /root/bcavenv/bin/python --no-deps --require-hashes --only-binary :all: -r /home/user/bca/outputs/shift/beyond-the-data/stage2_gpu/requirements-lock.txt
```

`uv --version` must print `uv 0.11.32`. (v7's copy of the lock file, `stage2_v7/laptop/requirements-lock.txt`, is the
same bytes.)

## Step 6. The environment's dependency folder (1-2 minutes)

In the **same window** as step 5 (so that uv is found):

```bash
BCA_SHIFT_PYTHON=/root/bcavenv/bin/python sh /home/user/bca/experiments/shift/install_deps.sh
```

It must end with `ok: gymnasium 1.0.0 ...`.

**Steps 0-6 are everything you can do before the cloud has pushed the v7 draft.**

## Step 7. Restore (after the draft is pushed; 10-25 minutes, about 0.8 GB download)

(Done on 14 Oct; it printed `PASS restore: 2370 run files of 901c87766ab0 (pre-handoff snapshot) match the pinned
sums`.)

```bash
git -C /home/user/bca pull --ff-only
sh /home/user/bca/outputs/shift/beyond-the-data/stage2_v7/laptop/restore_v7.sh
```

It downloads only the cloud's run files (the branch `claude/stage2-run-files`; git shows its progress and may ask for
your user name and token), checks that they are exactly the pinned snapshot, unpacks them under
`/home/user/bca/runs/` without overwriting anything (leaving out the cloud's logs and the files of phases the cloud
did not finish), checks every file's fingerprint, puts the version 6 code where it must be, and links
`/tmp/claude-0/bcavenv` to `/root/bcavenv`. It ends with `PASS restore: ...`.

## Step 8. The full self-test (25-40 minutes)

**First**: if you want the cloud's running phases to finish at the handoff rather than stop at once, say so in the
cloud session **now**. It cannot be changed after the self-test has written its first probe record; with no message,
"stop at once" holds. Your answer to Question 7 ("stop") is already in the draft, and it too cannot change after the
first probe record.

Then:

```bash
sh /home/user/bca/outputs/shift/beyond-the-data/stage2_v7/laptop/selftest_v7.sh
```

Expected: a `WAIT handoff: ...` line, `READY (waiting for handoff/approval)`, then the probe results and the forecast
gaps, then a block of lines between `BEGIN V7 SELF-TEST BLOCK` and `END V7 SELF-TEST BLOCK`. (Before any probe runs,
the self-test checks that the laptop's clock is right to within 2 minutes; if it is not, it says
`NOT READY (clock: ...)`: see the table at the end.) **The cloud needs the whole output, every line of the block
included.** Copying it from the window is error-prone; instead copy the saved copy to the Windows clipboard with:

```bash
clip.exe < "$(ls -t /home/user/bca/runs/shift_explore/stage2_v7/selftest/report_full_*.txt | head -1)"
```

and paste it (Ctrl-V) into the cloud session. The cloud reads it from its own copy of your message, so nothing is
retyped.

Before step 12, also test once that tmux keeps running when the window closes: `tmux new -s test`, then `sleep 600`,
then close the Ubuntu window. **Wait at least two minutes** (WSL stops Ubuntu only some seconds after its last window
closes, so opening it again at once proves nothing). Then open it again (step 2) and run `tmux ls`. If `test` is
listed, it works (type `tmux kill-session -t test` to end it). If `test` is not listed, never close the Ubuntu window
while the run is going, and say so in the cloud session.

## Step 9. The handoff (cloud: 10-20 minutes)

At the first convenient time, tell the cloud session: go ahead with the handoff. Until you approve (step 10), this can
still be undone (v7 would be withdrawn and the cloud would resume version 6). When the cloud says it is done:

```bash
git -C /home/user/bca pull --ff-only
sh /home/user/bca/outputs/shift/beyond-the-data/stage2_v7/laptop/restore_v7.sh
sh /home/user/bca/outputs/shift/beyond-the-data/stage2_v7/laptop/selftest_v7.sh --short
```

Expected: a `WAIT approval: ...` line and `READY (waiting for handoff/approval)` (it says plain `READY` only after
step 11).

## Step 10. Your approval (5 minutes)

```bash
python3 /home/user/bca/outputs/shift/beyond-the-data/stage2_v7/approve_local_v7.py
```

It shows the fingerprint of the final `prereg_v7.json`, version 6's fingerprint, what v7 changes, the failures v7 can
cause, the handoff, the full self-test's verdict and repeatability result, the answer to Question 7, the check (b)
forecast (with a plain warning if a gap above 0.00001 was accepted under Question 7), the probe results and the
references re-made on the laptop. Type the first 16 characters of the fingerprint when asked. Then commit and push the
file it wrote, under your own name:

```bash
cd /home/user/bca
git add outputs/shift/beyond-the-data/stage2_v7/approval_v7_*.json outputs/shift/beyond-the-data/stage2_v7/approval_v7_*.json.sha256
git -c user.name="David Bayha" -c user.email=35092758+dbayha7@users.noreply.github.com commit -m "Stage 2: approve v7"
git push origin claude/main-bca-connection-x4o46s
```

**If the push is rejected** (the message says "fetch first" or "non-fast-forward": the cloud pushed something in the
meantime), type:

```bash
git -c user.name="David Bayha" -c user.email=35092758+dbayha7@users.noreply.github.com pull --rebase origin claude/main-bca-connection-x4o46s
git push origin claude/main-bca-connection-x4o46s
```

This puts your one commit on top of the cloud's, still under your name. Never set the name or email with
`git config`: the code would then refuse to run.

## Step 11. The import (cloud: 10 minutes)

Tell the cloud session you pushed it. It imports your approval and pushes the import record. Then:

```bash
git -C /home/user/bca pull --ff-only
```

## Step 12. The run (at least about 4.5-5 hours at the cloud's speed; the real time on the laptop is unknown)

On the Framework Desktop it takes longer than that: with 24 GB for WSL, the memory rule lets only about 2 phases run
at once until their memory use has been measured, and more after. This changes no result; the time is only an
estimate and stops nothing.

First start tmux:

```bash
tmux new -s stage2
```

Then, **inside tmux**, type:

```bash
sh /home/user/bca/outputs/shift/beyond-the-data/stage2_v7/laptop/run_v7.sh --lanes 12
```

To leave tmux running: press Ctrl-b, then d. To come back: `tmux attach -t stage2`. Keep the Ubuntu window open
(minimised is fine) for the whole run. If the tmux test of step 8 failed, this is required: closing the last Ubuntu
window then stops every running phase, and each reruns from its start.

- **To stop safely**: press Ctrl-C once. No new phase starts and the running ones finish, which can take over an hour.
  A second Ctrl-C stops them at once; they rerun from the start next time.
- **After a closed window**: do step 2, then first try `tmux attach -t stage2`. If the run is still there, you are
  back. If tmux says there is none (`no server running`, `no sessions` or `can't find session`), the run has
  stopped: type `tmux new -s stage2`, then inside tmux the same `run_v7.sh` command. It continues where it stopped.
- **After a restart of Windows or WSL**: do step 2, then `tmux new -s stage2`, then inside tmux the same `run_v7.sh`
  command.
- **To see what is happening** from a second Ubuntu window:
  `sh /home/user/bca/outputs/shift/beyond-the-data/stage2_v7/laptop/run_v7.sh status`
- `--lanes` can be 1 to 16 on any restart; it changes no result.

Every start first puts the version 6 code back (WSL empties `/tmp`) and runs the short self-test; it refuses until
steps 9-11 are complete.

## Step 13. The results (10 minutes)

When the run prints a line starting `Finished`:

```bash
cd /home/user/bca
git add outputs/shift/beyond-the-data/stage2_v7/results
git -c user.name="David Bayha" -c user.email=35092758+dbayha7@users.noreply.github.com commit -m "Stage 2 v7: results"
git push origin claude/main-bca-connection-x4o46s
```

If the push is rejected, do as in step 10 (the `pull --rebase` line with the same `-c` name and email, then push
again). Then tell the cloud. No model file is in `results/`; they go to step 14.

(Only when the cloud asks for results before the end, for example while a phase waits for `--retry`:
`sh /home/user/bca/outputs/shift/beyond-the-data/stage2_v7/laptop/run_v7.sh results` gathers them as they are.)

## Step 14. The run files (upload of roughly 0.35-0.45 GB, an estimate)

```bash
sh /home/user/bca/outputs/shift/beyond-the-data/stage2_v7/laptop/push_runfiles_v7.sh
```

It commits the laptop's new run files (the ledger, the probe records, the reference copies in `refs/`, the logs, and
every new output of the laptop's phases, model files included) to a new branch `stage2-v7-run-files`, as you, with a
temporary git index (your checkout is not touched), and pushes it, retrying up to five times if the network fails.
Then tell the cloud; it runs the return check before reading any result.

## If something stops

Paste what the table says into the cloud session. **Never delete or edit anything under `/home/user/bca/runs/`.**
Below, `.../stage2_v7` stands for `/home/user/bca/outputs/shift/beyond-the-data/stage2_v7`; the commands to type are
written out in full.

| Message | Meaning | What to do |
|---|---|---|
| `NOT READY (versions ...)`, `(installed files)`, `(dependency folder)` | the Python setup differs | fixable: redo steps 5-6 in one window, then step 8 |
| `NOT READY (git ...)` | not a full clone, a git identity in root, or a changed tracked file (the message names it; changes under `stage2_v7/results/` are not counted) | fixable: as the message says (step 4); for a changed file, paste the message. Then type the same command again (step 8 only if it was the full self-test) |
| `NOT READY (clock: the laptop's clock is off)` | WSL's clock differs from the internet's time by more than 2 minutes (it can fall behind after Windows sleeps); no probe was started | in PowerShell type `wsl --shutdown`, open Ubuntu again (step 2), then type the same command again |
| `WARN ST9 clock ...` | the clock could not be checked, or (outside step 8) it is off | nothing to do in step 8 if it could not be checked; otherwise as for `NOT READY (clock ...)` |
| `NOT READY (restore: the run files could not be downloaded ...)`, or git says `could not read Username ... terminal prompts disabled` or `Authentication failed` | the network, or git could not sign in | check the token of step 4 (it must not have expired), then rerun step 7; if it persists, paste the output |
| `NOT READY (restore: ... run files differ from the pinned sums ...)` | a run file on the laptop differs from the pinned fingerprints | rerun step 7; if it persists, paste the output |
| `NOT READY (constants ...)`, `(gate ...)`, `(load_seed ...)` | the world's numbers are beyond tolerance, or the earlier test's exact checks refuse even with the cloud's numbers | ends v7; paste the output; you decide with the cloud |
| `NOT READY (P3: results depend on lane or load)` | the same training gave different results on different lanes or under load | ends v7; paste the output; any other route is a new version |
| `NOT READY (P3: a run did not finish)` | a repeatability run stopped (it shows nothing about repeatability) | run step 8 again; if it repeats, paste the output |
| `NOT READY (P1b: check (b) forecast ...)` | on this laptop, two programs that must agree within 0.00001 do not, so check (b) would likely stop seeds for a machine reason | ends v7 as specified (you answered Question 7 "stop"; the answer can no longer change now); the cloud keeps running version 6; paste the output; any other route is a new version, which you decide with the cloud |
| `NOT READY (P1/P2: records bind to another draft)` | the draft `prereg_v7.json` changed after your self-test wrote its probe records (which are never rerun) | paste the output; the cloud must put the earlier draft back, then step 8 again |
| `NOT READY (P1 p1_seed... did not finish)`, `(P2 p2_seed... did not finish)` or `(P1/P2: records missing)` | a probe did not finish (for example the window closed) | run step 8 again (a probe whose record is written is never rerun); if it repeats, paste the output |
| `NOT READY (one processor ...)` | WSL shows more than one processor to a lane | paste the output |
| `refused: another self-test is running` (also above `refused: the short self-test is not READY`) | a self-test, R1 or the results step is still running in another window | wait for it to end, then type the same command again |
| `re-made: seed s world w` in the probe results | the laptop did not reproduce that saved reference bit for bit | not a failure: that seed and world will use the laptop's own copy; nothing to do |
| `gate's file kept (P1b): seed s world w` | one program differed on the laptop, but the closer model of check (b) reproduced the saved controller | not a failure: that seed and world reads the saved file; nothing to do |
| `refused: the short self-test is not READY ...` with `WAIT handoff: ...` or `WAIT approval: ...` and `READY (waiting for handoff/approval)` above it | the WAIT line says what is missing | if you have done steps 9-11, type `git -C /home/user/bca pull --ff-only` (the cloud may have pushed the import record since) and the same command again; otherwise do steps 9-11. If the WAIT line repeats, paste it |
| `refused: the short self-test is not READY ...` with `NOT READY (...)` or `refused: ...` above it | a check failed, or something refused | see that row |
| `the short self-test ended without a verdict; its last lines:` | the self-test stopped early (for example a Python error) | paste the lines it shows |
| `refused: another runner is running on this machine` | the run is still going (perhaps in a tmux window you left) | type `tmux attach -t stage2`, or `sh .../stage2_v7/laptop/run_v7.sh status` to see it; do not start a second one |
| `refused: phases from an earlier run are still running` | the runner stopped, but some of its phases still run | wait; check with `sh .../stage2_v7/laptop/run_v7.sh status`; then type the same `run_v7.sh` command again |
| `interrupted: <phase> (machine-wide ...)` | a restart, WSL shutdown or closed terminal stopped a phase; its files were moved aside and it reruns | nothing to do |
| `interrupted: <phase> (phase-specific k of 3 ...)` | the phase was killed while the machine stayed up, or ran out of memory; it reruns | nothing to do |
| `waiting to run alone: <phase>` | a third phase-specific interruption; the phase runs alone once nothing else is runnable | nothing to do |
| `UNFINISHED: <phase>` | it was interrupted again when run alone; it waits; every other phase continues | paste the output of the `tail -n 30 ...` command the message gives. It must be retried until done: free memory or restart WSL if the cloud suggests it, change nothing else, then type `sh /home/user/bca/outputs/shift/beyond-the-data/stage2_v7/laptop/run_v7.sh --retry <phase>`. While the run continues, a second Ubuntu window is fine; if the run has stopped, type it **inside tmux** (it starts the run again) |
| `WAITING FOR RETRY: <phases>` | nothing else can run until an unfinished phase is done | as for `UNFINISHED`; the cloud may ask for the results so far (step 13, last paragraph) |
| `REVIEW: stand-in error in <lock>` | a policy lock records a failure of the rival method; no new policy training starts; every other phase continues | paste it; once the cloud has looked, type the command the message gives, `sh /home/user/bca/outputs/shift/beyond-the-data/stage2_v7/laptop/run_v7.sh --lanes 12 --reviewed <phase>`, in a second Ubuntu window while the run continues (the runner picks it up within a minute), or inside tmux if the run has stopped. Typing it twice is harmless |
| `WAITING FOR REVIEW: only policy training is left` | the run has stopped and waits for the review | as for `REVIEW`, inside tmux |
| `FAILED: <phase> exit N` | an unexplained exit; no new phase starts | paste the output of the `tail -n 30 ...` command the message gives; do not restart until the cloud has looked |
| `REFUSED (v7, exit 97)` | a v7 safety check stopped a phase | ends the run; paste the output and the `tail -n 30 ...` output |
| `HALT: <phase> has files ... that no attempt on this laptop wrote` or `HALT: <phase> is done on file, but ...` | files appeared under `runs/` that the laptop's own phases did not write | paste the output; do not restart until the cloud has looked; delete nothing |
| `HALT: reference ... differs from its pin` or `HALT: probe record ...` | a reference file or probe record no longer matches its fingerprint (the disk was damaged) | paste the output; the cloud will guide `sh /home/user/bca/outputs/shift/beyond-the-data/stage2_v7/laptop/selftest_v7.sh --remake <probe>`, which puts a damaged probe record back from the cloud's copy and remakes a damaged reference file (each only when the fingerprint matches); before the handoff a damaged probe record cannot be repaired |
| `HALT: reference: the lock of <phase> does not record the re-made actors` | a phase did not use the laptop's own copy as it must | paste the output; do not restart until the cloud has looked |
| `waiting: memory` or `waiting: slot` | the memory rule or the lane limit | nothing to do |
| `Stopped before the end (nothing failed)` | you pressed Ctrl-C | type the same `run_v7.sh` command again (inside tmux) to continue |
| `the push failed five times` (step 14) | the network or the sign-in | check the token (step 4) and run step 14 again; it rebuilds the same commits |
| `Finished.` | the end | steps 13 and 14 |

## A correction to the earlier package

`stage2_gpu/README.md` (its lines 275-281, "Notes on the numbers") says the anchors are the only check that compares
your machine's arithmetic with the cloud's. For seeds 0-4 run on another machine that is not so: check (b) and the
strength matching compare laptop-trained controllers with controllers the earlier test saved in the cloud. Version 7
handles this with the probes and, where needed, references re-made on the laptop (`PREREGISTRATION_V7.md`, change C7).

## What the package holds

| File | What it is |
|---|---|
| `README.md` | this page |
| `PREREGISTRATION_V7.md` | the declaration you approve (its machine-readable form is `prereg_v7.json`) |
| `prereg_v7.json` (+ `.sha256`) | version 7: draft first, final after the handoff; you approve its fingerprint |
| `code_manifest_v7.json` (+ `.sha256`) | the fingerprint of every v7 file |
| `world_constants.json` (+ `.sha256`) | the world's numbers A, B and K, made on the cloud's processor |
| `approve_local_v7.py` | your approval (step 10) |
| `code/` | the wrapper every phase runs through, its checks, the probes, the approval import, the writers, the tests |
| `laptop/restore_v7.sh`, `selftest_v7.sh`, `run_v7.sh`, `push_runfiles_v7.sh` | steps 7, 8, 12 and 14 |
| `laptop/lib/` | the runner (`runner_v7.py`), the self-test and restore (`selftest_v7.py`), the results (`make_results_v7.py`), shared shell settings (`common_v7.sh`); `envfacts.py` and `gpu_probe.py` are byte copies from `stage2_gpu/lib/` |
| `laptop/requirements-lock.txt` | byte copy of `stage2_gpu/requirements-lock.txt` |
| `cloud/` | what the cloud runs: the handoff (`make_handoff.py`) and the return check (`check_return_v7.py`) |
| `evidence/` | the cloud's measurements behind the world's numbers, and its information-only forecast on an emulated processor without AVX-512 (`evidence/xla/`) |
| `handoff/` | written by the cloud at the handoff: `HANDOFF.json`, the probe records you pasted |
| `results/` | written by the run (step 13) |

What runs on the laptop: every remaining phase. For seeds 0-4: the policy test, the deployment check and the full
host where the cloud did not finish them (the cloud finished all of them on 14 Oct, so none is expected to be left),
then the exploratory pooled look. For seeds 5-9: everything, from new data to the confirmatory pooled look. Nothing
runs in the cloud after the handoff.

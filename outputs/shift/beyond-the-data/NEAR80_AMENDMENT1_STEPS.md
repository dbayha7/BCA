# Near-80 amendment 1: your steps on the desktop, in order

Written 19 October 2026 (cloud build session); revised the same day after the first independent review (fix round
1), again after the second (fix round 2), and again after the third, independent review (fix round "indep",
`FIX_indep.md`). This is AMENDMENT_DESIGN.md section 8, made exact: the commands you type, what each should print,
and what to paste back to the session. Nothing here trains anything until the very last command (`run`), and no
phase runs under the amended code until your approval is imported (before that, only the desktop check reads and
tests it).

`<A16>` below means the first 16 characters of `amendment_1.json`'s SHA-256. The session tells you the value when it
says amendment 1 is final. At the time of writing it is **`52fe05b92900290b`**
(`52fe05b92900290b990e27d7f5ab6351e0c6c964300e8f8bd25d03bdfa981bbf`). If the session gives you another value (for
example because a later review changed a word), use the session's value everywhere below.

A draft of the amendment, with fingerprint `140e23e1e79bbaa8…`, was pushed earlier (commit `e105352`, marked "DRAFT
… not for approval yet"). It is superseded: do not run the desktop check on it or approve it.

---

## Now (before the session says amendment 1 is final)

1. **Do not type `run`.** Under the code on the desktop now, the same 10 phases would fail again (attempt 2) and the
   run would halt again.
2. **Answer one question in the chat, before the session says amendment 1 is final.** Section 19.6 asks you to
   confirm: *has anyone, you included, opened a record or log of the exploratory stage that holds a score or a
   learner number, or read such a number from one?* On your desktop, under `runs/shift_explore/near80/`, these are:
   - the look: `looks/exploratory.json` and `looks/exploratory.md` (`run_near80.sh report` also prints from them);
   - the records of seeds 0-4 under `learners/`, `policy/`, `eval/`, `readouts/` and `stops/`;
   - the logs of seeds 0-4's phases under `logs/`, whose training lines print learner numbers such as critic and
     cloning losses (for example, a `tail` of such a log while the exploratory stage ran).

   Seeing the runner's status output, or its line "the exploratory look allows (or stops) the near-80 settings on
   seeds 10-14", does not count: they show counts, CPU-hours and the look's decision, not a score.

   This question is wider than in the earlier versions of these steps (the first asked only about the look; the
   second did not name the logs). If you answered an earlier one already, answer this one too.

   Answer "nobody", or tell the session who opened which files: **name the files only; do not say what they
   showed.** If the answer is not "nobody", the session states it in section 19.6, changes the approval script's
   matching "Before you type:" paragraph, reruns its tests and writes `amendment_1.json` again, so `<A16>` changes.
   When the session gives you the new `<A16>`, and if you have already run the desktop check, first remove the
   earlier results:
   ```bash
   rm outputs/shift/beyond-the-data/near80_amendment1/desktop_check_*
   ```
   Then do Step 1 again from the start, including the desktop check: a result for an earlier fingerprint does not
   count.

Paste back: your answer to question 2.

---

## Step 1. Check, pull, and run the desktop check (after the session says amendment 1 is final and independently reviewed, and gives `<A16>`)

In PowerShell type `wsl -d Ubuntu-24.04 -u root`, then inside Ubuntu:

```bash
cd /home/user/bca
```
Expected: nothing printed.

```bash
sh outputs/shift/beyond-the-data/near80/run_near80.sh status
```
Expected:
- the first line starts with `runner running: no;`
- the last line starts with `halted: FAILED: ` and ends with `No new phase starts; nothing was deleted` (no full stop
  at the end);
- lines like `  halted critic.main.stall.s10: None` (the halted rows carry no note; that is a known cosmetic gap).

If it says `runner running: yes`, or the `halted:` line is not as above, **stop here**. Do not press Ctrl-C, and
approve nothing. Paste the first line, the `halted:` line if there is one, and, of the lines that start with
`  running`, only those whose phase name contains `.main.` or `critic_full`. Section 19.2 says nothing was retried;
a running runner means something was.

Never paste the `counts:` line, a `stopped` line, or a `running` line of any other phase: they can show the
exploratory look's decision, and the cloud session stays blind to it. If everything is as expected, paste back only
the first line and the `halted:` line.

If you are doing this step again for a new `<A16>` and ran the desktop check before, remove its earlier results first
(the second `git status` below shows them otherwise):
```bash
rm outputs/shift/beyond-the-data/near80_amendment1/desktop_check_*
```

```bash
git status --short --untracked-files=no
git status --short -- outputs/shift/beyond-the-data/near80 outputs/shift/beyond-the-data/near80_amendment1
find outputs/shift/beyond-the-data/near80 -name __pycache__ -o -name '*.pyc'
```
Expected: all three print nothing at all. If any prints anything, stop and paste it. (The second shows a stray
untracked file in the near-80 folders; the third a bytecode folder, which git does not show. The approval check
refuses both.)

```bash
git pull --ff-only
```
Expected: `Updating <old>..<new>` and `Fast-forward`, with a file list that includes
`outputs/shift/beyond-the-data/near80/amendment_1.json` and `amendment_1.json.sha256`. If you had not pulled the
draft before, the list also includes the other changed near-80 files (`PREREGISTRATION.md`,
`approve_local_near80.py`, `code/approval_check.py`, `code/import_approval.py`, `code/settingdata.py`,
`code/stubs.py`, `code/test_approval.py`, `code/test_runner.py`) and the new folder
`outputs/shift/beyond-the-data/near80_amendment1/` (3 files); if you had, it lists only the files that changed since
the draft. It may also list commits the desktop has not seen yet that touch other folders. If it says
`Already up to date.`, you pulled it earlier; that is fine. The next command (`sha256sum`) is the real check. If it
says `Not possible to fast-forward`, stop and paste it.

```bash
sha256sum outputs/shift/beyond-the-data/near80/amendment_1.json
```
Expected: one line that starts with `<A16>` (now `52fe05b92900290b990e27d7f5ab6351e0c6c964300e8f8bd25d03bdfa981bbf`).
If it starts with anything else (for example `140e23e1e79bbaa8`, the draft), stop and paste it.

```bash
sh outputs/shift/beyond-the-data/near80_amendment1/check_amendment1.sh
```
Type it exactly so, without `--no-tests`: approving needs a result in which the four unit test files ran and
passed. About 20-30 minutes; it trains nothing and writes only one result file (plus its `.sha256`). **It prints
nothing for the first several minutes** (it is rebuilding the 10 seed-worlds). Then it prints one line per unit test
file as each finishes: `PASS test_near80.py (…s)`, `PASS test_approval.py (…s)`, `PASS test_runner.py (…s)`,
`PASS test_pooled.py (…s)`. Then, expected exactly (seconds and the time stamp vary):

```
code is amendment_1.json's: True; the approval check's code check passes: True
failed phases: 10; exactly the cleared ones: True; they left no record or output: True; each ran once: True
model files of seeds 10-14: 0
seed-worlds rebuilt with every hash equal and the training file passing: 10; not rebuilt (failed the gate's checks): 0; record missing: 0; FAILED: 0 []
unit tests: test_near80.py PASS, test_approval.py PASS, test_runner.py PASS, test_pooled.py PASS
written: /home/user/bca/outputs/shift/beyond-the-data/near80_amendment1/desktop_check_<utc>.json
amendment 1 desktop check: PASSED (…s)
```

- The last line must be `amendment 1 desktop check: PASSED`. If it says `FAILED`, or a line starts with `refused:`,
  **stop here, approve nothing**, and paste every line it printed. (The approval script refuses anyway until a result
  file says the check passed, with the four unit test files and the check code `amendment_1.json` names.)
- `refused: this check (check_amendment1.py, check_amendment1.sh) is not the one amendment_1.json names` means the
  check's files on the desktop differ from the pushed ones: stop and paste it.
- If the fourth line shows a number other than 0 after `not rebuilt (failed the gate's checks):`, **stop and paste
  it**, even if the last line says `PASSED`: section 19.2 says all 10 records hold a training file, which is written
  only for a seed-world that passed the gate's checks, so the count must be 0.

Paste back: every line from `code is amendment_1.json's` to the last line. They are booleans and counts only.

```bash
python3 -I outputs/shift/beyond-the-data/near80/approve_local_near80.py --check
```
Expected (the script wraps its lines at 100 characters):
- it starts with `Near-80 (version 8): amendment 1`;
- under `What changes (SHA-256, first 16 characters, before -> after):` exactly 8 entries, each starting with
  `  - `, one per changed file. The first wraps onto a second line ending `the approved text)`:
  ```
    - PREREGISTRATION.md: 9869b18c02bfcfad -> 787e658f3d1cb7f0 (appended to; the first 90867 bytes are
      the approved text)
  ```
- under `The desktop check of this amendment (…/check_amendment1.sh):` two entries, each wrapping onto a second
  line:
  ```
    - its code, as amendment_1.json names it: check_amendment1.py 6843a7bfdd9f7b10,
      check_amendment1.sh 0a6b0e13fdf7c48b
    - passed: desktop_check_<utc>.json (official mode, this amendment_1.json, that code;
      the four unit test files ran and passed)
  ```
  (one `passed:` entry per passing result file of this `<A16>`);
- a paragraph `Before you type:` (the confirmation of section 19.6's item for you, the same question as "Now" item 2);
- `Fingerprint (SHA-256) of amendment_1.json:` followed by the full `<A16>…` fingerprint;
- the last line: `--check: every check passed. Nothing was asked and nothing was written.`

If it says `Refused.`, stop and paste it. If the desktop-check entry says `none has passed yet`, stop and paste it.
Paste back: nothing, if it passed.

---

## Step 2. Read, then decide

Read section 19 of `outputs/shift/beyond-the-data/near80/PREREGISTRATION.md` (everything after the line `---` that
follows section 18). To see the change itself, type:

```bash
git diff 287380b HEAD -- outputs/shift/beyond-the-data/near80
```
(`287380b` is the session's import of your approval of version 8; since then nothing in that folder changed except
this amendment, so the diff shows exactly the 8 files, `amendment_1.json` and its `.sha256`.)

Before approving, confirm the item of section 19.6 marked **For you to confirm before you approve**: to your
knowledge nobody, you included, opened a record or log of the exploratory stage that holds a score or a learner
number, or read such a number from one (the files listed in "Now" item 2). If someone did, do not approve this
version: tell the session who opened which files (name the files only; do not say what they showed). It states it
in section 19.6 and gives you a new `<A16>` to approve. When it does, first remove the earlier results with
`rm outputs/shift/beyond-the-data/near80_amendment1/desktop_check_*`, then do Step 1 again from the start, including
the desktop check: a result for an earlier fingerprint does not count.

---

## Step 3. Approve

```bash
python3 -I outputs/shift/beyond-the-data/near80/approve_local_near80.py
```
It prints the same summary as `--check`, then asks:
`To approve, type the first 16 characters of the fingerprint above (or press Enter to stop):`
Type `52fe05b92900290b` and press Enter. (If the session gave you another value, use that instead, here and in every
command below.)

Expected:
```
Approval written:
  outputs/shift/beyond-the-data/near80/approval_near80_amendment1_52fe05b92900290b.json
  outputs/shift/beyond-the-data/near80/approval_near80_amendment1_52fe05b92900290b.json.sha256

It covers amendment 1 with fingerprint 52fe05b92900290b… only, on top of the pre-registration 51e677ab76c8d2f7fa7779d21db7f53bd40d42ff5e6be278d4abfb34c3396d4a.
```
followed by the git commands. **Type the git commands exactly as the script printed them.** They are these (the
`git add` line also adds the desktop check's result file; keep it exactly so: the session's import refuses unless
your commit holds the result file your approval names):

```bash
git add outputs/shift/beyond-the-data/near80/approval_near80_amendment1_52fe05b92900290b.json outputs/shift/beyond-the-data/near80/approval_near80_amendment1_52fe05b92900290b.json.sha256 outputs/shift/beyond-the-data/near80_amendment1/desktop_check_*
git -c user.name="David Bayha" -c user.email=35092758+dbayha7@users.noreply.github.com commit -m "Near-80: approve amendment 1 52fe05b92900290b"
git push origin claude/main-bca-connection-x4o46s
```
Expected:
- `git add` prints nothing;
- `git commit` prints `[claude/main-bca-connection-x4o46s <commit>] Near-80: approve amendment 1 52fe05b92900290b`
  and `4 files changed` (the approval, its `.sha256`, the desktop check's `.json` and its `.sha256`). If you ran the
  check more than once for this `<A16>`, every result file is added; that is fine;
- `git push` ends with a line like `<old>..<new>  claude/main-bca-connection-x4o46s -> claude/main-bca-connection-x4o46s`.

If the push is rejected (`fetch first` or `non-fast-forward`), type:

```bash
git -c user.name="David Bayha" -c user.email=35092758+dbayha7@users.noreply.github.com pull --rebase origin claude/main-bca-connection-x4o46s
git push origin claude/main-bca-connection-x4o46s
```

Never set the name or email with `git config`. Once pushed, do not delete, edit or re-add these files.

After the push has succeeded, type:

```bash
git log -1 --format='%H %an <%ae> | %cn <%ce>' -- outputs/shift/beyond-the-data/near80/approval_near80_amendment1_52fe05b92900290b.json
```
Expected: one line, `<commit> David Bayha <35092758+dbayha7@users.noreply.github.com> | David Bayha <35092758+dbayha7@users.noreply.github.com>`.
(After a `pull --rebase` the commit's hash differs from the one `git commit` printed; this line shows the one that
was pushed.)

Paste back: that line, and "pushed".

---

## Step 4. The session imports (you wait)

The session runs `code/import_approval.py import-amendment` in the cloud and pushes
`outputs/shift/beyond-the-data/near80/import_near80_amendment1_52fe05b92900290b.json` (+ `.sha256`). It refuses if
the approval commit's author or committer is the session's identity or is not you, and unless your commit holds the
passing desktop check result file your approval names.

When it says it has pushed: open that import record on GitHub and check that the commit it names (`"commit": …`) is
the one from the `git log` line you pasted in Step 3 (same hash, your name).

Paste back: "the commit is mine" (or what differs).

---

## Step 5. Run

```bash
git -C /home/user/bca pull --ff-only
```
Expected: `Fast-forward` with `import_near80_amendment1_52fe05b92900290b.json` and its `.sha256`.

```bash
sh /home/user/bca/outputs/shift/beyond-the-data/near80/run_near80.sh selftest
```
Expected: the last line is `READY`. If it says `NOT READY (…)` or `refused:`, stop and paste every line.

```bash
tmux new -s near80
```
(or `tmux attach -t near80` if that session exists), then inside tmux:

```bash
sh /home/user/bca/outputs/shift/beyond-the-data/near80/run_near80.sh run --lanes 10
```
Expected: it repeats the self-test (`READY`), then lines like
`<utc> start  critic.main.stall.s10 on processor N (attempt 2; reserves … GiB)` for the failed phases (attempt 2),
then `start` lines of phases not yet started (attempt 1). No line `FAILED: critic…` or `refused:`.

If anything refuses, nothing starts: paste the `refused:` lines. Paste back otherwise: only the `start` lines whose
phase name contains `.main.` or `critic_full` (never a `skip` line, nor a `start` line of any other phase: they can
show the exploratory look's decision).

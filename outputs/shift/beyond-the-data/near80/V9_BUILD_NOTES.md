# Near-80 version 9: build notes (not pinned)

Written 20 October 2026 by the session, on branch `claude/near80-v9` of the line-ending-safe clone. Nothing here is
part of the code manifest (`approve_local_near80.py` and the freeze allow this file beside the pinned ones). Nothing
has been trained, frozen, approved or run officially; the DRAFT markers stay until the freeze step itself.

## 1. Where the build stands

- The pre-registration (`PREREGISTRATION.md`), its machine-readable twin (`prereg_draft.json`), the code and its
  five test files hold every value `DECISION.md` fixed on 20 October 2026 (the session's scratchpad: `decide/DECISION.md`,
  `BUILD_LIST.md`, `REVISION_LOG.md`): ten fresh seeds 40-49 (50-59 reserved), C2 at 5.0 points with ceil(0.8 n)
  positive seeds over all three twins, C3 at -5.0 and at most 1, `MIN_CERTIFIED` 8, the twins `meandiff_q90x1`,
  `meandiff_q90x2`, `meandiff_limit`, the refusal readings (C1: no caution; C2 and C3: the seed left out), the lead
  over `WORLD_BLIND + (range_hinge,)`.
- **The owner's answer** (20 October 2026, 14:29 Eastern = 18:29 UTC, in the chat with the cloud assistant):
  "Accept the defaults and cluster". Recorded verbatim in `OWNER_ANSWER` of `code/near80_constants.py` with the time,
  the place, a line per choice, and `taken` for the two choices that are not the defaults: choice 4 (the machine) is
  the university cluster hpc.charlotte.edu and choice 9 (where results come from) is the cluster's login node in the
  owner's own SSH session; the desktop is the fallback for both. `near80_prereg.choices_taken` writes those two from
  `OWNER_ANSWER["taken"]`; `placeholders_present()` is empty.
- **The cluster entry is part of version 9**: `run_near80_cluster.sh` and `near80_slurm_job.sh` are back in the
  folder and in `TOP_FILES` (the code manifest will pin them), with the cluster smoke's patch list applied
  (section 4). `PREREGISTRATION.md` 12.2 describes the cluster run, 12.1 the desktop as the fallback, 13 the measured
  cluster timing beside the desktop's 5-5.5 hours, 8.2 reads a reproduction difference per host.
- **Open until the freeze** (by rule): the two `TODO(FREEZE: ...)` marks (section 4.2's table of the pre-approval
  facts and section 0's redraft count: filled by hand from the record `freeze.py write` produces), the DRAFT status
  lines of `PREREGISTRATION.md` and `README.md`, and "DRAFT, not frozen" in the code files' first docstring lines
  (section 7).

## 2. The two rules of 20 October 2026, and how the build reads them

1. **Pinned files may be edited, moved or deleted with a complete change log.** Section 3 is that log. Short form:
   every file version 8's code manifest lists under `outputs/shift/beyond-the-data/near80/` was edited (and one,
   `code/v6bca.py`, deleted) because version 9 supersedes version 8 in place (choice 7); no pinned repository file
   and nothing of the gate's pre-registration was touched.
2. **Seeds: every stub, smoke or plumbing test uses seed 0; no test, fixture, default or example names a reserved
   fresh seed (40-59).** Applied as follows.
   - Every stub run in the tests runs on seed 0 (`test_runner.TestStubEndToEnd` already did; `test_v9.TestStubRefuse`
     now refuses seed 0 itself, `--seeds 0 --refuse 0`; the stub generator's refusal test uses seed 0). The cluster
     entry's `stub` command defaults to `--seeds 0` (`near80_slurm_job.sh stub` too). The runner's help texts name
     seed 0 only.
   - The tests that need a fresh GROUP of several seeds without running anything real (the look's arithmetic on
     synthetic records in `test_pooled` and `test_v9`; the plan, the fake phases, the per-seed runners and the
     fresh-seed record round trip in `test_runner`) use the synthetic labels `K.TEST_LABELS = 1000-1009`: no study
     uses them, no registry names them, nothing is ever generated or trained on them. `test_pooled` and `test_v9`
     patch `FRESH` to them inside the test process (their `Base.setUp`); `test_runner` sets
     `NEAR80_STUB_FRESH_LABELS=1` before the constants load, because its fixtures are read by phase processes that
     must see the same `FRESH` as the process that wrote them (`approval_check.check_constants` compares). The
     constants honour that variable (then `FRESH = TEST_LABELS`); the runner forwards it to phase processes in stub
     mode only; the entries never set it; an official run cannot be moved by it, since `prereg.json` pins `FRESH`
     and every phase's approval check refuses a constants module whose `FRESH` differs from the pinned one.
   - Symbolic references to the block stay where a test checks the rule itself (`K.FRESH[0]`, `K.RESERVED_SEEDS[0]`,
     `K.FRESH_SEED_BLOCK`): they name no literal and run nothing.
   - What this reading does NOT do: it does not use seeds 1-4 as test labels (they are seen seeds; `group_of`
     refuses them), and it does not put the fresh group's arithmetic tests on seed 0 alone (one seed cannot exercise
     the n = 8-10 thresholds). If the owner wants the labels elsewhere than 1000-1009, `TEST_LABELS` is the one
     place to change.
   - The cluster smoke of 20 October (section 4) ran the draft entry's stub on seed 40 before this rule; that record
     stands as it is, and every stub from now on uses seed 0.

## 3. Changes to pinned files

**What is pinned today.** Version 8's `prereg.json` (fingerprint `51e677ab76c8d2f7...`, at the top of the folder
until the freeze moves it to `prereg_history/`) pins version 8's `code_manifest.json`, which lists 28 near-80 files
(`PREREGISTRATION.md`, `README.md`, `approve_local_near80.py`, `run_near80.sh`, `push_runfiles_near80.sh` and 23
files under `code/`) and 133 repository files under the pinned folders (`experiments/shift`, `calibration`,
`algorithms`, `runtime`, `restructure/step1`, `restructure/step2`, `stage2/code`, `runs/shift_explore/room/code`,
`stage2_v7/code`, `stage2_v7/laptop/lib`, plus step 1's and step 2's results and version 6's `prereg.json`). The
gate's own pre-registration (`runs/shift/preregistration.json`) pins the gate's code and the shift dependencies.

**Repository files pinned by version 8's manifest: none changed** (`git diff 52914e4 --name-status` outside the
near-80 folder is empty). **The gate's pre-registration and the shift dependencies: nothing changed.** The near-80
code still only *reads* version 6's `stage2_common.py`, step 1's and step 2's libraries and the room harness from
their verified bytes.

**Near-80 files listed in version 8's code manifest: every one changed, by design.** Version 9 supersedes version 8
in place (DECISION.md choice 7, taken by default): the folder is rebuilt as version 9's, version 8's chain stays
checkable at commit `52914e4`, and version 8's approval, import, amendment, pre-approval and results files stay
untouched as history (`V8_HISTORY_FILES`, `V8_HISTORY_DIRS`, `V8_FROZEN_NAME`). Which manifest or pre-registration
this breaks, and what follows: **version 8's `code_manifest.json` no longer matches the folder**, so version 8's
`approve_local_near80.py --check` passes only at `52914e4` (its `prereg.json` is still the one at the top until the
freeze). **No amendment is needed**: version 9 is a new pre-registration with a new fingerprint; `freeze.py write`
moves version 8's `prereg.json` to `prereg_history/prereg_superseded_51e677ab76c8d2f7.json`, archives version 8's
manifest as `code_manifest_history/code_manifest_v6.json`... (the next free number) and writes a new
`code_manifest.json` that re-hashes every file below plus the three new ones (`code/test_v9.py`,
`run_near80_cluster.sh`, `near80_slurm_job.sh`; `prereg_draft.json` is moved into `prereg_history/`, not pinned).

| file (relative to `near80/`) | what changed | why |
|---|---|---|
| `PREREGISTRATION.md` | rewritten as version 9's pre-registration: the pooled groups, the refusal rule, the twins, C2 / C3, the seeds, section 14's sentences, the hosts (12), the compute (13), the owner's answer (0, 16) | version 9 |
| `README.md` | version 9's one-page summary; the owner's answer; the cluster run and the desktop fallback | version 9 |
| `approve_local_near80.py` | version 9's names (label, frozen facts file, the history files allowed beside the pinned ones, the build notes); the statement the owner types accepts the choices as recorded | version 9 |
| `run_near80.sh` | the version 7 checks gone; the fresh-seed path; the refusal text names the cluster entry | version 9 |
| `push_runfiles_near80.sh` | the cluster branch (`NEAR80_HOST=cluster`, never root), the run-files branch `claude/near80-v9-run-files`, the output root `near80_v9` | version 9 |
| `code/near80_constants.py` | every value of DECISION.md; `OWNER_ANSWER`; `TOP_FILES` with the cluster scripts; `CLUSTER` (the job sizes); `TEST_LABELS` and the `NEAR80_STUB_FRESH_LABELS` hook; the host profiles | version 9; the owner's answer; the seed rule |
| `code/near80lib.py` | `pooled_groups`, `pooled_count_record`, the certification chance, `check_spent_seeds` | the refusal fix |
| `code/facts_near80.py` | the pre-approval facts: the counts on seeds 0-4 and the spent-seed switch facts on 10-14 with the kill rule | DECISION.md 8.1 check 5 |
| `code/learner_phases.py` | `switch_scores_pooled`, the sorted tail scores in the switch record | the pooled groups, the height ranks |
| `code/policy_phase.py` | the per-arm refusal, the twins, the lone calls in order, the height-0 check, the stub's stand-in log line | the arms of section 5 |
| `code/evaluate_phase.py` | a refused arm's values null; `evaluate_step2_saved` gone | the refusal rule |
| `code/pooled.py` | the look: the certified seeds, C2 / C3 with the three readings, the lead over six rivals, the sentences, the report tables | sections 9 and 14 |
| `code/claims_text.py` | the anchors of section 14 (the new clauses and last sentences) | section 14 |
| `code/near80_prereg.py` | `MARKS_C3`, `MIN_CERTIFIED`, the choices taken from `OWNER_ANSWER`, the cluster entry in `runner` | version 9; the owner's answer |
| `code/approval_check.py` | the hosts line, the statement, `choices_taken`; version 6's `stage2_common` marked as loaded from verified bytes (`__step2_sha256__`) | the owner's answer; a bug (section 6) |
| `code/freeze.py` | the spent-switch arguments and the kill-rule check; the `TODO(...)` and DRAFT refusals | DECISION.md |
| `code/import_approval.py`, `code/ledger.py`, `code/phase_main.py`, `code/phases.py`, `code/settingdata.py` | version 9's names, the one setting, the fresh-seed path, the seed groups | version 9 |
| `code/near80_runner.py` | `--seeds` / `--group-only` / `--tag` for the cluster's jobs; `mine()` leaves the lock-dependent phases to the group job; the self-test checks version 6's run folder; `git add -f` in the results step; the refusal check of `--refuse`; the stub labels forwarded in stub mode | the cluster entry; DECISION 4.3; the seed rule |
| `code/stubs.py` | the synthetic fixtures carry the new arms and the refusing seed | the tests |
| `code/test_near80.py`, `code/test_approval.py`, `code/test_runner.py`, `code/test_pooled.py` | retargeted to version 9; the seed rule | version 9 |
| `code/v6bca.py` | **deleted** | version 6's BCA arm is not in version 9 (section 5.7) |

New, not in version 8's manifest (the freeze pins them): `code/test_v9.py`, `run_near80_cluster.sh`,
`near80_slurm_job.sh`; `prereg_draft.json` (moved to `prereg_history/` by the freeze); this file (never pinned).

## 4. The cluster entry and its test record

The record is the session's scratchpad `cluster/SMOKE.md` (with `SETUP.md`, `BASELINE.md`, `CLUSTER_READY.md`,
`review.md`, `review_log.md` and the job logs). In short, from 20 October 2026, 15:25-17:50 UTC, through the
owner's SSH control socket, 56.8 of the 90 pre-approved CPU-minutes (about 67 counting the desktop's unit tests):

- **Environment** (`/scratch/dbayha/bca-near80/env`): the desktop's Python 3.10.20 build, uv 0.11.32, the 67 wheels
  of the stage2_gpu lock file installed with `--require-hashes`; `shiftdeps/` file for file the desktop's
  `/root/.cache/bca-shift/py` (267 files). numpy's OpenBLAS picks the same `SkylakeX` kernel on the desktop and on
  the caslake, genoa and turin nodes; a numerics probe and the paired BCA-free driver (critic, cloned actor,
  no-caution actor, two evaluations) gave bit-identical results on all of them.
- **Unit tests** (version 8's four files, job 27394071, 4 processors, with the five pinned `runs/` folders shipped
  read-only): test_approval 105 OK (1 skipped), test_runner 85 OK (test_near80 46 OK and test_pooled 60 OK in job
  27393665).
- **The entry in stub mode** (nothing trained, a synthetic fixture): the draft entry's run (job 27394073) finished
  all 19 stub phases plainly; the gate's dependency-path check refuses any path other than the literal
  `/root/.cache/bca-shift/py` (job 27394257); with the namespace wrapper the patched entry's own `stub` command
  submitted job 27394258, which ran all 19 phases as uid 4560 through the job script. (Seed 40 was used there,
  before the seed rule; section 2.)
- **Timing** (jobs 27393747, 27393750, 27394072): caslake core 1.62 and turin core 0.85 times an idle desktop core;
  the package's own phases at the official sizes on caslake 0.64 (data), 1.25 (critic), 1.16 (clone), 1.41 (actor
  training) times version 8's desktop ledger; one seed's chain 86 min (version 8's phases) on caslake, about 96 with
  version 9's lone programs; 2.05 + about 0.2 CPU-hours per seed; ten seeds about 1.7 hours of wall.
- **Not covered by any cluster test:** BCA's certificate, the second critic, the matching, the lock, the look; the
  plumbing's reproduction check decides, and a difference moves the run to the desktop (PREREGISTRATION.md 8.2, 12).

**The patch list of SMOKE.md section 6, as applied in this build:**

| item | where |
|---|---|
| 1. the pinned dependency path inside an unprivileged user namespace | `near80_slurm_job.sh` (the wrapper at its top, every mode) |
| 2. a `stub LANES FIXTURE [SEEDS]` mode of the job script | `near80_slurm_job.sh` (SEEDS defaults to 0) |
| 3. a `stub --fixture DIR ...` command of the entry | `run_near80_cluster.sh` (default `--seeds 0`, `--time 01:00:00`) |
| 4. `--job-name` as the jobs' prefix | `run_near80_cluster.sh` (default `near80v9`) |
| 5. the run folders the self-test does not check | `near80_runner.default_checks`: "version 6's run folder" (`runs/shift_explore/stage2/prereg.json`); `stage2_v7` is not read by version 9 (its check went with version 7's) |
| 6. HOME and the git identity | `NEAR80_HOME` (default the folder `home` beside the clone, under `/scratch/dbayha`), exported to every command and job |
| 7. memory per job | the per-command defaults: plumbing 4 lanes 16 GB 2 h, seed 8 lanes 24 GB 3 h, group 16 lanes 24 GB 1 h (`K.CLUSTER`; `test_v9` checks the scripts carry them) |
| 8. processor ids | nothing to change (noted in the entry's header) |
| 9. the stub's stand-in log line | `policy_phase.standin_log`: "stand-in not run (stub mode, by design)" |
| 10. the group job owns every evaluation | `near80_runner.Cfg.mine` / `group_dependent`: a `--seeds` runner ends once only lock-dependent phases remain; `--group-only` takes the lock, the evaluations and the look (a seed job that owned an evaluation would wait for a lock no runner holds, and the group job waits for the seed jobs) |
| 11. the approval | unchanged: `selftest`, `plumbing`, `run` refuse until the frozen version 9 is approved and imported |
| 12. node type | `--constraint=caslake` by default (`--constraint ''` for none) |
| 13. the build's direction | reversed by the owner's answer |

How the results come back (PREREGISTRATION.md 12.2 step 4): `run_near80_cluster.sh results` copies the records into
`near80/results-<utc>/` in the cluster clone and prints the `git add -f` and push commands; the owner commits and
pushes them in their own SSH session (`push_runfiles_near80.sh` likewise: it needs a terminal and the owner's GitHub
sign-in); or the two folders come back to the desktop by `tar` and are pushed from there.

## 5. Every change of this build, by commit (on top of `52914e4`)

- `0b886d1` WIP: the pooled groups, the outcome-free count, the per-arm refusal, the first new arms, the host
  profiles, placeholders (cut off by a usage limit).
- `d2ae569` BUILD_LIST 1: DECISION's values into the constants, the draft and the documents.
- `438563b` BUILD_LIST 2: the twins `meandiff_q90x1` and `meandiff_q90x2`; the lone calls in order.
- `096aeda` BUILD_LIST 3: the refusal readings; C2 / C3 on the certified seeds; the lead over six rivals; the
  switch-call table, the ranks, the ladder; `certify_chance`.
- `795d450` BUILD_LIST 4-7, first pass: the pre-approval facts (`--spent-switch`), the freeze's kill rule, `git add
  -f`, the cluster scripts out (undone later), the tests retargeted, `test_v9.py`.
- `b750c35` BUILD_LIST 1 closed: the preamble names the open marks without spelling `TODO(...)`.
- `096c2a1` the owner's answer and the tested cluster entry (section 1).
- `cd6d8a8` the seed rule for the tests; the first test round's fixes (section 6).
- `1731ca2` the second round's fixes.
- `e9a7264` the synthetic fresh labels through the environment; the third round's fixes.
- (the commit after this file: the build notes; any fix of the fourth round)

## 6. Tests

**How they were run.** Inside WSL (Ubuntu-24.04, as root, through the bridge `wslrun.ps1`), in the pinned environment
`/tmp/claude-0/bcavenv/bin/python`, exactly as `freeze.run_unit_tests` does: each test file alone, from a work folder
under `/tmp`, with `env -i` and the phases' environment (`PYTHONPATH=/home/user/bca`, `PYTHONDONTWRITEBYTECODE=1`,
`PYTHONPYCACHEPREFIX`, `JAX_PLATFORMS=cpu`, `OMP_NUM_THREADS=1`, `HOME=/root`, `NEAR80_HOST` unset = the desktop).
The branch was shipped as a git bundle and cloned under `/tmp/bridge/work/v9b/repo`; each test ran in its own mount
namespace (`unshare -m --propagation private`) in which that clone is bound over `/home/user/bca` and the owner's
`runs/` is bound read-only at `/home/user/bca/runs`, so the owner's clone was never written (checked: its `git
status` unchanged, no file newer than the shipment under its near-80 folder or `runs/`; a write test on the bound
`runs/` returned "Read-only file system"). The scripts: the session's scratchpad `v9build/jobs/`.

**Round 1** (`096c2a1`): test_near80 27 OK; test_pooled 45 OK; test_approval 107 run, 7 failures; test_runner 84 run,
7 failures and 6 errors; test_v9 28 run, 2 failures and 1 error. What they found:

- `approval_check.load_v6_common` loaded version 6's `stage2_common.py` from its verified bytes without marking it
  (`__step2_sha256__`), so step 2's pinned import check (`machinery.check_imports`) refused every process that then
  loaded step 2's machinery: `policy_phase.load_step2`, and so every policy phase of a real run. Fixed by marking the
  module with the hash its bytes were verified against. (Version 8's chain did not trip on this; version 9's policy
  code loads step 2 after the approval check.)
- `Cfg(seeds=[a used seed])` raised `ValueError` where the runner's callers expect `Refused`.
- `--refuse` accepted a seed the stub run did not have (and wrote the fixture before refusing).
- `approval_check.STATEMENT` and `approve_local.STATEMENT` differed after the owner's answer.
- The stub look reads `invalid` (the stand-ins score 0, so the validity check fails by design) where the tests
  expected `not_read`; the tests now say so.
- The synthetic log of `test_near80` had no `world`; a refusal does not lower the hand-set range limit's net (the
  test expected it would); the placeholder tests still expected the owner's answer to be open.
- `test_values_equal_the_draft_and_the_gate` compared the draft's C1 rival set with `WORLD_BLIND` (it is
  `LEAD_RIVALS`); `DEFAULTS["seeds"]` read "ten fresh seeds" where the draft says "ten seeds".

**Round 2** (`cd6d8a8`): test_near80 OK; test_pooled OK; test_v9 1 failure; test_approval 3 failures; test_runner
could not start (`mock` not imported). **Round 3** (`1731ca2`): test_near80 OK; test_pooled OK; test_v9 28 OK;
test_approval 2 failures; test_runner 12 failures and 1 error, all from one cause: a `FRESH` patched inside the test
process is not what the phase processes see, so the fixtures the process wrote failed their constants check
(`near80_constants.py differs from prereg.json's constants: ['FRESH']`) and the fake phases refused seed 1000;
version 8's `test_changed_point` expected a stop version 9 no longer has (its `evaluate_step2_saved` phase is gone).

**Round 4** (`e9a7264`, the code as committed; the five files at once, each in its own namespace):

| file | tests | result | wall seconds |
|---|---|---|---|
| test_near80.py | 27 | OK | 4 |
| test_approval.py | 107 | OK (1 skipped) | 16 |
| test_runner.py | 83 | OK | 138 |
| test_pooled.py | 45 | OK | 4 |
| test_v9.py | 28 | OK | 78 |

290 tests, all pass (version 8's four files had 296: `test_changed_point` went with its phase, and the test files were
rewritten for version 9's claims). The logs: `/tmp/bridge/work/v9b/test_<name>.log` in the WSL distro.

## 7. `freeze.py check` in the scratch clone (writes nothing)

Run at `1731ca2` and again at `e9a7264` (the final code; the same output) in the same namespace (the real paths:
`/home/user/bca`, the pinned Python). It reports 0
placeholders and would: move version 8's `prereg.json` into `prereg_history/`, move `prereg_draft.json` to
`prereg_history/prereg_draft_<first 16>.json`, write the manifest (30 near-80 files, the repository files under 10
pinned folders), run the five test files, run `facts_near80.py --seeds 0 1 2 3 4 --spent-switch 10 11 12 13 14
--worlds stall no_stall`, write `prereg.json`. **It still refuses on:**

- `PREREGISTRATION.md carries 2 TODO(...) marks` (section 4.2's facts table and section 0's redraft count: filled at
  the freeze from the facts record);
- `PREREGISTRATION.md line 3` and `README.md line 3` are not the FINAL line; both `still say **DRAFT`;
- `code/evaluate_phase.py`, `near80_prereg.py`, `near80_runner.py`, `phase_main.py`, `pooled.py`, `test_v9.py` say
  DRAFT in their first docstring line (the other code files say it on a later line, which the freeze does not read;
  `run_near80.sh`, `run_near80_cluster.sh` and `near80_slurm_job.sh` say "DRAFT, not frozen" in their headers, which
  nothing checks: the freeze step should drop all of them in one pass).

Section 14 extracts cleanly (37 sentences, no mark). The code folder holds exactly the 23 planned files, no
`__pycache__`.

## 8. For the owner's attention

1. The freeze runs on the desktop (`freeze.py write` inside WSL): the facts step reads version 8's run files under
   `runs/shift_explore/near80` (the recorded second critics of seeds 10-14) and the gate's data, which only the
   desktop holds; the cluster clone then fetches the frozen commit.
2. PREREGISTRATION.md 8.2 now reads a reproduction difference per host: a difference on the cluster moves the run to
   the desktop with its own plumbing; a difference on the desktop too ends the version. This is the one place where
   the cluster choice changed a rule's wording; the review of the draft should look at it.
3. Section 11: the approval is typed on the desktop as before; the statement the owner types says "every choice of
   its section 16 as it records them (the defaults; the university cluster ...; the desktop as the fallback)".
4. The cluster smoke read five pinned `runs/` folders of the WSL clone (nothing written) and ran the draft entry's
   stub on seed 40 (`cluster/CLUSTER_READY.md` part A asks whether both were within the rules); from now on every
   stub uses seed 0.
5. The cluster clone for the real run must be a git clone of the approved commit under `/scratch/dbayha` with the
   `runs/` folders hard-linked from `runs_pinned/` (`SETUP.md`, `CLUSTER_READY.md` B8); `/scratch/dbayha/bca-near80/repo_v9`
   is the draft at `0b886d1` and must be replaced.
6. The `NEAR80_STUB_FRESH_LABELS` hook in the constants (section 2) exists for the tests only; if the owner would
   rather have the tests run the fresh group on real seed labels, the hook and `TEST_LABELS` are the two lines to
   remove, and the tests would again refer to `K.FRESH[0]` and so on.

## 9. Time estimates (unchanged from DECISION.md, with the cluster's numbers)

The facts before approval about 0.2 CPU-hours on the desktop; the plumbing about an hour on the desktop, about
1-1.5 hours as one cluster job; the ten seeds about 1.7 hours of wall on the cluster (about 24 CPU-hours) or about
5-5.5 hours on the desktop (about 18); the budget guard's forecast 2.2 per seed, the pause at 44, the cap at 55.

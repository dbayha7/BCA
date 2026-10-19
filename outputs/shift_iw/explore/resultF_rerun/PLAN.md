# Result F's actor for the live panel: plan of the seed-0 cloud rerun

**EXPLORATORY: outside the version-5 pre-registration.** Planning and dry checks only; nothing has been trained.
Written 2026-10-16; revised the same day after review (section 9). Inputs: `SCOUT.md` (same folder), the sandbox
`iw_rerun` (its `PLAN.md`, `CHECK.md`, `run_seed0.sh`, `driver/jobs.py`), and the repository at `/home/user/bca`
(read only).

This is the toy world: ThrusterChain, the project's own small simulated task.

## 0. Bottom line

- **What it trains.** For data seed 0, medium-expert and expert, in the cloud sandbox, from the sandbox's own seed-0
  gapped TD3+BC host (the live panel's TD3+BC learner: its cloned actor and its frozen critic), with result F's code
  unchanged and the desktop's training settings:
  1. **"Fixed fallback, exact weights"** (`A2_cmd_strong`, `command_le_train_f1.py train`). The script also trains B2
     and C2 in the same job and cannot train A2 alone.
  2. **"No extra caution"** (`le_none`, from `run.py room` then `run.py room-lower-edge`). The same job also trains
     the "Perfect lower edge" (`le_oracle`).
  3. **Optional, on by default (cheap):** "Estimated weights" (`A3_estimated` + `C3_untargeted`,
     `command_le_train_est.py`), and version 5's lower edge, unweighted (`explore_arms.py --lower-edge unweighted`).
     They are left out of the DONE condition.
- **One CPU and one thread for every job.** This keeps A2 and "No extra caution" on the same arithmetic in the
  sandbox. It differs from the desktop's setting for result F's arms, which ran unpinned with two threads, so the
  clone-hash comparison with the desktop is information only (sections 2 and 4).
- **What it scores.** Each trained actor is scored on both mixes' 1,000 fixed starts (and 'mild'), as result F scored
  them (`learners_iw.evaluate_actor`, by the scripts' own evaluate modes). `eval_resultF.py` then:
  - checks that each score equals the script's own record;
  - checks that A2 (and A3) started from the very cloned actor "No extra caution" started from;
  - compares each score with result F's desktop seed-0 numbers;
  - records the start-1 episodes of both mixes;
  - rolls every actor in the main test's four situations, including the start-11 episodes, in the shapes the page
    already reads.
- **Approvals: nothing stops the plan.** The committed `outputs/shift_iw/v5/approval.json` covers every approval gate
  these jobs check (`band` and `bca_lower_edge`, both qualities). The sandbox already links it read-only. No approval,
  pass-mark, spec or declined file is created, edited or deleted anywhere.
- **The band's report gate is already met.** `data_seeds1to4.sh` (started by the main session) built data seeds 1-4 of
  both qualities and ran `run.py report` per quality; it finished at 15:14 UTC. The run's done checks accept those
  records, so it skips them and never rebuilds them. Today's dry run showed that `require_design` accepts both real
  reports and that D1 of seeds 0-4 equals the committed report's bit for bit on both qualities (section 6).
- **Time:** on 3 CPUs (1, 2, 3), about **2 to 2½ hours** if other sessions load the machine as they did during the
  earlier rerun; about 1¼ hours if it stays quiet. Section 7 has the details.
- **Start** (after an optional repeat of the dry checks, `bash run_resultF.sh dry`, about 2½ min):

```
cd /tmp/claude-0/-home-user/9e341792-006d-5a11-a8e9-8fac16ac7f5b/scratchpad/resultF && setsid nohup bash run_resultF.sh > logs/nohup.out 2>&1 < /dev/null &
```

  - It refuses to start while any process runs `experiments.shift_iw.run` with its working directory in the sandbox,
    or while `data_seeds1to4.sh` runs.
  - Watch: `bash run_resultF.sh status` (read only).
  - Stop: `bash run_resultF.sh stop`.
  - Resume: the same start command; finished jobs are never redone.
  - Finished when `DONE` exists in this folder.
  - Leaner run: `EST=0 V5LE=0` before `setsid` drops the optional rows. `CPUS="0 1 2 3"` uses all four CPUs.

## 1. Files

| File | What it is |
|---|---|
| `run_resultF.sh` | The driver (bash, `set -euo pipefail`). Modes: `run` (default), `dry`, `status`, `stop`, `verify`. It schedules the jobs on the CPUs in `CPUS` (default `1 2 3`), one job per CPU, each pinned to it. |
| `data_seeds1to4.sh` | The main session's separate script that built data seeds 1-4 and the reports. Not run by `run_resultF.sh`. Logs: `logs/data_*.log`, `logs/report_*.log`. |
| `driver/gates.py` | Read-only gate check, run first in the jobs' own environment. Writes `state/gates.json`. |
| `driver/rf.py` | Standard-library helper: done checks, D1 comparison, baseline and untouched checks, provenance of `data_seeds1to4.sh`'s outputs, EXPLORATORY labels, quarantine of an interrupted arm. |
| `driver/report_dry.py` | Read-only dry check of the band's report gate (dry mode only). |
| `driver/smoke_eval.py` | Smoke test of `eval_resultF.py run --allow-missing`, with the host actor standing in for A2 and No extra caution (dry mode only). Writes `smoke/eval/` only, labelled SMOKE TEST. |
| `smoke/test_scheduler.sh` | Stub test of the scheduler (section 2). Writes `smoke/sched_test/` only. |
| `eval_resultF.py` | Scoring, comparison, episodes and four situations, after training. Modes: `check`, `run`, `run --allow-missing`. |
| `logs/<job>.log`, `logs/run.log` | One log per job, and the runner's log. |
| `state/` | Gates, baselines, D1 comparison, `data_seeds1to4.json` (that script's outputs by path), `job_times.tsv`, pid and failed markers, the lock, the quarantine. |
| `results/` | `scores_seed0.json`, `COMPARE.md`, `episodes_seed0.json`, `situations_seed0.json`, all labelled EXPLORATORY. |

## 2. The jobs (seed 0; Q = medium-expert and expert)

Every job runs in the same environment:

```
cd <iw_rerun>/repo; PYTHONPATH=<iw_rerun>/repo:<iw_rerun>/explore PYTHONDONTWRITEBYTECODE=1 JAX_PLATFORMS=cpu OMP_NUM_THREADS=1 OPENBLAS_CORETYPE=Haswell NPY_DISABLE_CPU_FEATURES=<AVX-512 list> XLA_FLAGS=--xla_cpu_max_isa=AVX2 nice -n 10 taskset -c <one CPU> python3 -B ...
```

- **One CPU and one thread for every job.** This is how version 5's stages ran on the desktop (one process per CPU,
  `OMP_NUM_THREADS=1`, `nice -n 10`).
- **It is not how the desktop ran result F's arms.** The desktop trained the channel arms (cmd, f1, est) unpinned with
  `OMP_NUM_THREADS=2` (`outputs/shift_iw/explore/curves/orchestrate.py`, lines 3-4).
- The plan keeps the pinning on purpose: A2 and "No extra caution" then share their arithmetic in this sandbox, and
  the paired difference between them is a like-for-like comparison. XLA's CPU arithmetic depends on how many CPUs and
  threads a process may use, so the comparison of the cloned actor's hash with the desktop's is information only.
- On top of this come the processor settings the sandbox's earlier rerun used (`ARITH=desktop`). `gates.py` checks
  them before anything starts.

| Job | Command | Needs | Writes (sandbox `runs/`) |
|---|---|---|---|
| `room_Q` | `-m experiments.shift_iw.run room --quality Q --seeds 0` | data 0, host gap 0 (present) | `shift_iw_v5/Q/room/seed0/` |
| `data_Q_sK` (K = 1-4) | `run data --quality Q --seeds K` | — | already built by `data_seeds1to4.sh`; skipped |
| `report_Q` | `run report --quality Q` | data 0-4 | already built by `data_seeds1to4.sh`; skipped |
| `d1_Q` | `rf.py d1 Q`: the new data's D1 against the committed report, bit for bit | data 0-4 | `resultF/state/d1_Q.json` only |
| `band_Q` | `run band --quality Q --seeds 0 --kinds gap` (approval `band`; report gate) | room, report | `shift_iw_v5/Q/band/gap/seed0/` |
| `roomle_Q` | `run room-lower-edge --quality Q --seeds 0` → **No extra caution**, Perfect lower edge | room | `shift_iw_v5/Q/room_lower_edge/seed0/` |
| `f1train_Q` | `<repo>/outputs/shift_iw/explore/channel/command_le_train_f1.py train Q 0` → **A2**, then B2, then C2 | band | `shift_iw_v5_explore/channel/train_f1/Q/<arm>/seed0/` |
| `f1eval_Q` | `command_le_train_f1.py evaluate Q 0` | f1train | `.../train_f1/Q/evaluate_seed0.json` |
| `roomleeval_Q` | `run evaluate --quality Q --seeds 0 --groups room_lower_edge` | roomle | `shift_iw_v5/Q/evaluate/seed0/room_lower_edge.json` |
| `f1checks_Q` | `command_le_train_f1.py checks Q 0` (the desktop ran it for seed 0, before training; it writes nothing that training reads) | band, roomle | `.../train_f1/Q/checks_seed0.json` |
| `esttrain_Q` (EST=1, optional) | `command_le_train_est.py train Q 0` → A3, then C3 | band | `.../train_est/Q/<arm>/seed0/` |
| `esteval_Q`, `estchecks_Q` (EST=1, optional) | `command_le_train_est.py evaluate / checks Q 0` | esttrain / band, roomle | `.../train_est/Q/` |
| `v5le_Q` (V5LE=1, optional) | `<iw_rerun>/explore/explore_arms.py --quality Q --seed 0 --lower-edge unweighted` (approval `bca_lower_edge` + the owner's exploratory go) | band, room | `shift_iw_v5_explore/Q/bca_lower_edge/unweighted/seed0/` |
| `v5leeval_Q` (V5LE=1, optional) | `explore_eval.py --quality Q --seed 0 --groups bca_lower_edge`. Only this group; the live panel's `bca.json` is never redone. | v5le | `shift_iw_v5_explore/Q/evaluate/seed0/bca_lower_edge.json` |
| `evalF` | `resultF/eval_resultF.py run`, or `run --allow-missing` (below); last, after every other job has settled | f1eval and roomleeval of each quality, or A2 alone | `resultF/results/` only |

**Data seeds 1-4 and the reports.** `data_seeds1to4.sh` built them with the same environment, except that its jobs
were not pinned with `taskset`. The run's done checks accept them: a data record counts when its record checksum
matches, it parses, it is finished under this specification, and it names its seed and quality; a report counts when
it is finished, covers seeds 0-4, was made by the same code as the data, and is newer than every data record. The dry
run confirmed that the run will skip all ten jobs. `rf.py d1` still checks their D1 bit for bit, and
`state/data_seeds1to4.json` and the EXPLORATORY label list their files by path.

**Scheduling.**
- A free CPU takes the first ready job in the order above, so the two room checks (the critical path) start first.
- A failed job blocks only the jobs that need it.
- `evalF` waits until every other job has settled. Then:
  - if `f1eval` and `roomleeval` finished on both qualities, it runs `eval_resultF.py run`, which requires every core
    learner (A2, B2, C2, No extra caution, Perfect lower edge);
  - otherwise, if A2's record and actor are good on both qualities, it runs `eval_resultF.py run --allow-missing`:
    A2 is scored, No extra caution is scored when it exists, every other learner that exists is scored, and the
    missing rows are listed in COMPARE.md. This covers B2 or C2 failing after A2, and room-lower-edge's check (e)
    stopping the seed;
  - otherwise it is blocked.
- **DONE** needs every core job finished, `evalF` complete (not `--allow-missing`) and the untouched check passed. The
  optional jobs (`est*`, `v5le*`) are left out of the DONE condition; if one fails, DONE still comes and names it.
- **Stub test** (`smoke/test_scheduler.sh`, run today): a copy of `run_resultF.sh` with the helper, the jobs and the
  gates replaced by stubs, on a fake sandbox. All 24 checks passed:
  - an optional failure still gives DONE, which names it; the ten data and report jobs that `data_seeds1to4.sh` left
    were skipped and never started; a stranger process outside the sandbox did not stop the run;
  - B2 failing after A2 gave `evalF --allow-missing` and no DONE; the resume ran only f1train, f1eval and a complete
    evalF, then DONE;
  - f1train failing before A2 blocked evalF;
  - a process running `experiments.shift_iw.run` in the sandbox, and one running `data_seeds1to4.sh`, made both `run`
    and `dry` refuse (exit 75).

## 3. Approvals, the owner's go, and every gate

- **Approval.** `<iw_rerun>/repo/runs/shift_iw_v5/approval.json` (and `.sha256`) are the sandbox's existing
  symbolic links to the committed `outputs/shift_iw/v5/approval.json` (SHA-256 c6b374d9…, matches its sidecar).
  - Scope: `band, coverage, bca, bca_lower_edge`, all five qualities.
  - `run.approval_for` verified it read only today, against the sandbox's code copy, for `band` and `bca_lower_edge`
    on both qualities. The owner's exploratory go was found: "Run the arms anyway, labelled exploratory
    (Recommended)".
  - Nothing writes to the link: no stage writes `approval.json`, and `rf.py untouched` checks the repository's
    `outputs/shift_iw/v5` folder at the end.
- **The owner's go for these rows.**
  - `PASSMARKS_fallback_rerun.md` L6-11: "Fix the fallback, then rerun (Recommended)".
  - `PASSMARKS_estimated_weights.md`: "Proceed with all three" (the optional row).
  - Today in chat: "result f is solid lets do that".
  - The owner's ruling on records: the package's own record checksums and the sandbox's own `report.json` are
    allowed (section 9, question 1).
- **Gates the code enforces, and how each is met:**
  - **Spec:** `iw/spec.json` equals its sidecar (a71af9bd…). Checked.
  - **Code:** the sandbox's code copy is byte-identical to the repository (45 of 45 files, no extra code file). Checked.
  - **Result F's scripts:** byte-identical to the code named in the desktop records (`this_code`). They are run in
    place, so no copy was needed:
    - `command_le_train_f1.py` f3f3b734…
    - `command_le_train.py` c91f4608…
    - `command_band_check.py` 3c4d4dfa…
    - `command_le_train_est.py` 2006a117…

    Their frozen pass marks match their sidecars: f1 8b8f7a27…, first test 204f53ff…, est 4560769a…. Their output
    folders resolve inside the sandbox's `runs/` (from the imported package, not from their own folder). All of this
    is checked by `gates.py`.
  - **Inputs:** the start banks, data 0 and host gap 0 are finished and sidecar-verified. The host's
    `snapshot_100.msgpack` is the very file the live panel's TD3+BC learner came from (medium-expert b2845d8d…,
    expert 7f3b7bac…). Checked.
  - **Band stage's report gate:** met by `data_seeds1to4.sh`'s reports (medium-expert 8caddd13…, expert b3739da8…).
    `require_design` accepted both in today's dry run.
  - **Room-lower-edge's check (e):** can stop a seed. On the desktop it did not stop seed 0. If it does here, "No
    extra caution" is missing; A2 is still scored by `evalF --allow-missing` (section 9, question 3).

## 4. Settings: the desktop's training settings, with one difference

- All rows use `learners_iw.settings()` and the frozen spec, through unchanged code:
  - master seed 20261016, alpha 2.5, batch 256, Adam 3e-4;
  - 20,000 actor-only steps in 5 chunks of 4,000 from the 20,000-step clone, refreshed at each chunk start;
  - k = 1, grid ratio 1.05 with floor 1e-4, actor spread 0.2 clipped at 0.5.
- No command-line option changes a setting. Every job is the desktop's own command with the seed set to 0.
- **The one difference: threads.** The desktop trained result F's arms unpinned with two threads; this run pins every
  job to one CPU with one thread (section 2). It changes no setting, only how XLA's arithmetic is carried out.
- **Not bit for bit.** The sandbox's host differs from the desktop's in the last bits (iw_rerun `checks/REPRO.md`), and
  the thread setting differs, so the critic, band and actors cannot match the desktop exactly.
- **Bit checks that do not depend on the host or the threads** (each depends only on the data and the settings):
  - the row stream's hash: medium-expert c837711b…, expert 062ebb7a…;
  - D1 of seeds 0-4 (already identical on both qualities in today's dry run).
- **Information only:** the clone's hash (medium-expert 3545e423…, expert e0096a04… on the desktop). The clone is
  trained by XLA in each job, so the different thread setting can change its last bits.
- **What does matter:** that A2 and "No extra caution" start from the same clone in this sandbox. Both clone it the
  same way (`learners.clone`, same data, seed and settings) on one pinned CPU with one thread. `eval_resultF.py`
  compares the clone SHA-256 in their records (section 5).

COMPARE.md reports each of these checks.

## 5. Scoring and outputs

**Scores.** The scripts' own evaluate modes call `learners_iw.evaluate_actor` on the shared start banks:
- the exact float64 100-step return from all 1,000 fixed starts of 'none' (the training mix), 'mild' and 'strong';
- the gapped normalization;
- points = 100 (J − J_random) / (J_stall_aware − J_random), with the banks' anchors.

`eval_resultF.py run` recomputes each score in the same way and records whether it equals the script's record.

**Same cloned actor.** For each quality, `eval_resultF.py` compares the clone SHA-256 in A2's record (and A3's, when
present) with the one in room-lower-edge's record for "No extra caution". The result is in `scores_seed0.json`
(`comparison.<Q>.same_clone`) and on its own line in COMPARE.md. If they differ, or one is missing, the overall
verdict cannot be "consistent with result F".

**`results/scores_seed0.json` and `results/COMPARE.md`.** For each learner and bank they hold:
- the sandbox's J and points beside the desktop's seed-0 values;
- the gap, with a tolerance of 3 × the SD of the desktop's own J over its learning-curve checkpoints at
  16,000-20,000 steps.

They also hold the paired differences against the desktop's seed 0 and seeds 0-4, the same-clone check, the bit
checks, the "no value credit" share per chunk, and the four-situation points. `complete` says whether every core
learner was scored. Under `--allow-missing`, COMPARE.md says so at the top and lists the missing rows and why.
COMPARE.md also notes that the optional v5 lower-edge record's `pre_registered_report` is the sandbox's partial report
(data seeds 0-4 only: D1 pass, every other mark pending or not run), not the committed version-5 report.

The meaningful check is the sandbox's own gap, A2 minus No extra caution on the strong starts. The desktop's values:

| Quality | Desktop seed-0 A2 strong J (points) | No extra caution strong J (points) | A2 − No extra caution, seed 0 | Seeds 0-4 |
|---|---|---|---|---|
| medium-expert | −152.44 (94.35) | −176.31 (92.47) | +23.87 | +23.4 to +29.6 |
| expert | −116.40 (97.18) | −197.35 (90.81) | +80.94 | +65.6 to +80.9 |

**`results/episodes_seed0.json`.** It has the same shape as `iw_rerun/results/episodes_seed0.json`: the same
top-level keys, plus `qualities`, `learners`, `learner_labels` and `panel_learner` = `f1_A2_cmd_strong`. It holds:
- the start-1 episodes of both mixes (the start rule still picks start 1; checked);
- states, commands, rewards, J, Jd, points and push;
- `q_actor`, the frozen gapped host critic's min(Q1, Q2) at the actor's command (the critic every one of these actors
  was trained against);
- `qpi_actor`, the exact Q^π (horizon 600, discount 0.98).

**`learner_facts`** (in both files) carries the keys the page builder reads: `failed` (false), `snapshot_100_file_sha256`
(these actor-only learners have no snapshot_100, so it is the SHA-256 of the file the actor was loaded from:
`actor.msgpack`, or room-lower-edge's `actors.msgpack`, named in `snapshot_100_file`), `actor_tree_sha256` and
`critic_tree_sha256`.

**`results/situations_seed0.json`.** It has the same shape as `sit4/situations_seed0.json`, including its top-level
`seconds`:
- the four situations (`thruster_chain.bank_starts`, 1,000 starts each, the stalling world);
- `avg` (J, SE, Jd, points, points SE) and the start-11 episodes, keyed `quality|learner|situation`.

Points use sit4's stored anchors, so every learner on the page shares one scale. The recomputed anchors agree with them
to 5.5e-16 relative; above 1e-9 the script refuses.

## 6. Dry checks done today (nothing trained; nothing written outside this folder)

The final `bash run_resultF.sh dry` passed every check (2026-10-16, 15:35:51-15:37:53 UTC, load 0.2). It ran after
`data_seeds1to4.sh` had finished (15:14 UTC) and no process was working in the sandbox. The files it checked, by
SHA-256 (first 12 digits): `run_resultF.sh` 4a79fcf74b28, `eval_resultF.py` 59c1bfd067ab, `driver/gates.py`
82c741f3e739, `driver/report_dry.py` 0dcd815beb99, `driver/rf.py` 4e4239ffbc89, `driver/smoke_eval.py` fe4f7669b847,
`data_seeds1to4.sh` 5a5be6ae1df1. The stub test (`smoke/test_scheduler.sh` 1ccfed03180f) passed at 15:34:48 UTC on
the same `run_resultF.sh`.

- **Script:** `run_resultF.sh` parses, and pinning to each of CPUs 1-3 works.
- **`gates.py --dry`:** all 23 checks passed (`state/gates_dry.json`):
  - code copy 45/45 identical;
  - the scripts' hashes equal the desktop records;
  - the three pass-mark files are unchanged;
  - the outputs resolve inside the sandbox, and every module is imported from the sandbox's copy;
  - spec, approval link, `approval_for` band and bca_lower_edge for both qualities, the exploratory go;
  - start banks, data 0 and host 0 = the live panel's snapshot;
  - the processor settings equal the earlier rerun's;
  - result F's `World` refuses only at the missing band.
- **Data seeds 1-4 and the reports** (`data_seeds1to4.sh`'s): the run's own done checks recognize all ten records on
  each quality, so the run will skip them. D1 of seeds 0-4 equals the committed report's bit for bit on both
  qualities (`rf.py d1 --dry`, `state/dry_d1_<Q>.json`).
- **The band's report gate** (`driver/report_dry.py`, `state/report_dry.json`): the package's own `build_report` on
  the real data records gives D1 "pass" and every other mark "pending" or "not run"; `require_design` accepts it, and
  it also accepts the real report file on both qualities.
- **Argument parsing:** `run.py data ... --help` and `report ... --help` parse. `run.py status` runs.
- **Each real command, run once while its input is missing, refused at the expected place and wrote nothing.** Each
  probe's output went to its own file (`logs/dry_probe_<name>.log`), and only that file was judged:
  - band and room-lower-edge: at the room check;
  - evaluate room_lower_edge: at room-lower-edge;
  - f1 checks, est checks, and `explore_arms --smoke --lower-edge unweighted`: at the band.
- **`eval_resultF.py check`** passed (`state/eval_check.json`): the start rule picks start 1, the anchors agree,
  and the desktop reference values load.
- **`eval_resultF.py` smoke test** (`driver/smoke_eval.py`), with the host actor standing in for A2 and No extra
  caution, writing `smoke/eval/` only, labelled SMOKE TEST:
  - `run` without `--allow-missing` refused at once, naming the missing B2, C2 and Perfect lower edge;
  - `run --allow-missing` ran end to end, marked the scores partial, listed the missing rows, printed the same-clone
    line, and wrote `failed`, `snapshot_100_file_sha256` and the situations file's `seconds`;
  - the stand-in's four-situation points equal sit4's for the same actor (medium-expert 98.3 / 92.0 / 96.2 / 91.3).
- **Nothing changed:** `rf.py untouched` found no change, removal or new file in the sandbox, no change in the
  repository folders read in place, and no path in the repository (outside `.git`) newer than the dry run's start.
- **Scheduler:** the stub test passed (section 2).
- **No dry or smoke mode exists for the training scripts.** `command_le_train*.py`, `run.py room`,
  `room-lower-edge` and `band` have none, so nothing beyond the refusals above was run.

## 7. Time on 3 CPUs

| Job (per quality) | Desktop | Cloud, quiet (×2) | Cloud, loaded (×4) |
|---|---|---|---|
| data seeds 1-4, report | — | already built by `data_seeds1to4.sh` (15:05-15:14 UTC) | — |
| d1 | < 1 min | seconds | seconds |
| room | 12-15 min (spec estimate) | 25-30 min | 50-60 min |
| band | ≈ 1 min (spec estimate) | 2-3 min | 4-6 min |
| room-lower-edge | 7-10 min (spec estimate) | 14-20 min | 28-40 min |
| f1 train (A2 139 s, B2 ≈ 135 s, C2 ≈ 105 s, setup ≈ 1 min; measured) | 7.3 min | 15 min | 30 min |
| est train (A3 151-172 s, C3 ≈ 114 s, setup; measured) | ≈ 6 min | 12 min | 24 min |
| v5 lower edge, unweighted | 1.5-2.7 min (spec estimate) | 3-5 min | 6-10 min |
| checks (f1, est) | 1-2 min each | 2-4 min | 4-8 min |
| evaluations | seconds each | < 1 min | 1-2 min |
| evalF (all 16 learners) | — | measured here: about 10 s per learner, about 3 min | 6 min |

- **CPUs.** Other sessions share this 4-CPU machine, so the default is `CPUS="1 2 3"`, leaving CPU 0 to them.
- **Slowdown factor.** The sandbox's earlier rerun ran TD3+BC 3.6-4.6× slower than the desktop with 4 jobs and other
  sessions on the CPUs, against 1.7× in its unloaded probe. The load average was 0.5 at 15:29 UTC.
- **Order on 3 CPUs.** The two room checks run first (the third CPU only does the two D1 checks). Then the two bands
  and the first room-lower-edge; then the two f1 trainings; the second room-lower-edge waits for a free CPU, and is
  the longest pole. The optional rows and the checks fill the gaps. `evalF` comes last.
- **Total:**
  - loaded: about **2 to 2½ h** (room ≈ 55 min, band ≈ 5, f1 train ≈ 30, the second room-lower-edge ≈ 35, then the
    evaluations, the checks and evalF);
  - quiet: about 1¼ h;
  - with `EST=0 V5LE=0`: only about 10 min less, because the second room-lower-edge sets the end;
  - with `CPUS="0 1 2 3"`: about 20-30 min less when loaded, at the other sessions' expense.
- **The largest uncertainty** is the room check and room-lower-edge. Their desktop times are the spec's estimates,
  not measurements.
- **Disk:** under 0.1 GB more (the data are already built); about 14 GB is free. **Memory:** 14 GB is available.

## 8. Resume, stop and safety

- **Refusal at start.** `run` and `dry` refuse (exit 75) while any process running `experiments.shift_iw.run` has its
  working directory in the sandbox, or while `data_seeds1to4.sh` runs. The check reads each process's arguments one
  by one, so a shell command that merely mentions the module name does not count.
- **Resume.**
  - Each job's done check parses its output: a record with a matching record checksum, finished under this
    specification and naming its seed, quality and the keys the check reads; or the script's own record with
    matching actor SHA-256, pass marks, label and clone. A record that does not parse or lacks a key counts as not
    finished, never as a crash.
  - A rerun of `run_resultF.sh` skips finished jobs and redoes interrupted ones. The package's stages redo any seed
    without a finished record.
  - Result F's scripts skip an arm whose `record.json` merely exists. So before `f1train` / `esttrain` restarts,
    `rf.py quarantine` moves an interrupted arm folder of this run into `state/quarantine/`, and its script then
    trains it again. An interrupted arm is one whose record does not parse or whose actor does not match. The move is
    refused for anything that existed before the run.
  - A partial `evalF` (`--allow-missing`) does not count as finished; a resume runs it again, complete once the
    missing jobs have finished.
  - The driver's own state files (`d1`, gates, report dry check, untouched, baseline, provenance, `eval_check`) and
    `eval_resultF.py`'s outputs are written whole or not at all (a temporary file, then `os.replace`).
- **Stop.** `run_resultF.sh stop` stops the runner and its jobs; a stopped job restarts from scratch on resume.
  `flock` on `state/run.lock` prevents two runners.
- **Untouched.** At the first start, `rf.py baseline` hashes every file of the sandbox and of the repository folders
  this run reads in place. At the end, `rf.py untouched` requires all of the following, and records this run's new
  files in `state/untouched.json` (`sandbox.new_under_runs_files`):
  - no baseline file changed or vanished;
  - no new file in the sandbox outside `repo/runs/`;
  - no change in the repository.

  Paths in the repository newer than the start (outside `.git`, which other sessions' git use touches) are listed as
  a warning. When the check fails, each path it names is listed with its modification and change times and the jobs
  of this run that were running then (from `state/job_times.tsv`).
- **Labels.**
  - `results/*` and result F's and `explore_arms`' records carry EXPLORATORY labels themselves.
  - The package's own stage records (data, report, room, room-lower-edge, band, evaluate) cannot carry one without
    code changes. So `rf.py label` writes `EXPLORATORY_resultF_rerun.txt` into both `runs/` roots at the start. At the
    end of every run it rewrites them with an explicit list of every file the run created (in the sandbox and in this
    folder) and of every file `data_seeds1to4.sh` built (with SHA-256 and time; also in `state/data_seeds1to4.json`).

## 9. Review findings (applied 2026-10-16)

| # | Finding | Outcome |
|---|---|---|
| 1 | Thread settings | Real. Pinning kept; sections 2 and 4 and the script's header now say it differs from the desktop's channel arms (unpinned, two threads) and that the clone-hash comparison is information only. |
| 2 | Same cloned actor | Real (not checked before). `eval_resultF.py` now records and prints it; the verdict falls short of "consistent" if they differ. |
| 3 | A2 scorable if B2 or C2 fail | Real (evalF was blocked, and a bad B2 record refused the whole scoring). `run --allow-missing` and the automatic switch added. |
| 4 | Done checks parse JSON; atomic writes | Real. Done. |
| 5 | Optional jobs out of DONE | Real (any failure blocked DONE). Done. |
| 6 | Dry probes judged on their own files | Real (a shared log tail was judged). Done. |
| 7 | Labels: explicit file list; v5 report note | Real. Done. |
| 8 | learner_facts and `seconds` | Real (`failed`, `snapshot_100_file_sha256`, top-level `seconds` were missing). Done. |
| 9 | Untouched: `.git` out, details on failure | Real. Done. |
| 10 | Default CPUs | Done: `CPUS="1 2 3"`; time quoted in section 7. |

## 10. Open questions

1. **Record checksums: resolved by the owner's ruling.** The package's own record checksums (the `.sha256` files the
   unchanged package writes beside its records in the sandbox's `runs/` tree, such as `*.json.sha256` and
   `actor.msgpack.sha256`) and the sandbox's own `report.json` are allowed: they are ordinary package records, as in
   the earlier rerun. Only approval, spec, pass-mark and declined files are off limits, and none is created, edited
   or deleted.
2. **Optional rows.** "Estimated weights" and version 5's unweighted lower edge are on by default and left out of the
   DONE condition. `EST=0 V5LE=0` drops them.
3. **If room-lower-edge's check (e) stops seed 0 here** (it did not on the desktop), "No extra caution" does not
   exist. `evalF` then scores A2 alone (`--allow-missing`) and the run is not DONE; the paired comparison would need
   a decision.
4. **Data seeds 1-4 were built unpinned** by `data_seeds1to4.sh`. Their D1 equals the committed report's bit for bit,
   and they feed only the band's report gate, so this should not matter; it is recorded in
   `state/data_seeds1to4.json`.
5. **The live panel itself.** Embedding the A2 actor's episodes and situations in the ThrusterChain page is the next
   step after this run. This plan only produces the data.

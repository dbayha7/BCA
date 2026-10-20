# Result F in the live panel: scout for a seed-0 cloud rerun

**EXPLORATORY: outside the version-5 pre-registration.** This scout only reads and checks; nothing was trained.
Written 2026-10-16.

**Is this the toy world?** Yes. ThrusterChain is the project's own small simulated task: a Gymnasium environment with
five floats, four thrusters and currents set by dials. The page calls it "a small simulated task, built so that every
true value in it can be computed" (`docs/thrusterchain/thrusterchain.html`). Result F's own limits say the same:
"One environment, with its own data" (`outputs/shift_iw/explore/channel/RESULTS_fallback_rerun.md` L86).

## 0. Bottom line

- **Result F's row is arm `A2_cmd_strong`.** The page calls it "Fixed fallback, exact weights". It is trained by
  `outputs/shift_iw/explore/channel/command_le_train_f1.py`, whose code would run unchanged.
- **Approvals do not block a seed-0 sandbox rerun.** The committed `outputs/shift_iw/v5/approval.json` covers every
  approval gate the code checks: scopes `band` and `bca_lower_edge` for both qualities. I verified this read-only
  against the sandbox's code copy (section 4).
- **What does block it today: two inputs are missing in the sandbox.** The sandbox has only the data and the gapped
  TD3+BC host for seed 0. Result F also needs:
  - **BCA's band on that host** (`run.py band`). This stage refuses until a report on all five data seeds, made by
    this code, says D1 passed (`run.require_design`, run.py L616-631). The sandbox has no report and no data for seeds
    1-4. This gate is not an approval.
  - **The room check and the lower edge's room check** (`run.py room`, `run.py room-lower-edge`). These produce
    "No extra caution" (`le_none`) and "Perfect lower edge" (`le_oracle`). They need no approval.
- **Recommended route (b), using only existing code:**
  1. Build data for seeds 1-4 of medium-expert and expert in the sandbox (8 jobs of about 2 min).
  2. Run `run.py report` per quality. D1 is the only mark the band gate reads.
  3. Run room, then room-lower-edge and band, then result F's script.

  Route (a) is the alternative: link the committed report read-only. It passes the gate, but it is risky (section 4).
- **Time:** about 1-1.5 h wall clock on 4 CPUs. Up to about 2.5 h if other sessions keep the CPUs as busy as now
  (load average 7.7 on 4 CPUs at 13:42 UTC). The details are in section 5.

## 1. What trained each row on the desktop, what it reads and writes, and its gates

The desktop ran everything from `/root/bca-shift` with:

```
PYTHONPATH=/root/bca-shift PYTHONDONTWRITEBYTECODE=1 JAX_PLATFORMS=cpu OMP_NUM_THREADS=1 nice -n 10 taskset -c <cpu>
```

There was one process per job, and outputs went under `runs/shift_iw_v5` (official stages) or
`runs/shift_iw_v5_explore` (exploratory). The channel scripts lived in `/mnt/c/Users/David/dev/bca-shift-work/channel`
(curves.py L44). The committed copies are byte-identical: the `this_code` hashes in the records equal today's files.
- command_le_train_f1.py: f3f3b734…
- command_le_train.py: c91f4608…
- command_band_check.py: 3c4d4dfa…
- command_le_train_est.py: 2006a117…

| Page row | Arm | Desktop command (per quality Q, seed k) | Scored by |
|---|---|---|---|
| No extra caution | `le_none` (k = 0 arm of the oracle program) | `run.py room --quality Q --seeds k`, then `run.py room-lower-edge --quality Q --seeds k` (launch.sh) | `run.py evaluate --groups room_lower_edge` → `runs/shift_iw_v5/Q/evaluate/seed k/room_lower_edge.json` |
| Perfect lower edge (scale) | `le_oracle` | the same room-lower-edge job | the same file |
| Version 5's lower edge, unweighted | `le_bca_unweighted` | `explore_arms.py --quality Q --seed k --lower-edge unweighted exact_strong exact_mild` (launch_explore.sh; each arm in its own program, so running `unweighted` alone gives the same actor) | `explore_eval.py --quality Q --seed k` → `runs/shift_iw_v5_explore/Q/evaluate/seed k/bca_lower_edge.json` |
| **Fixed fallback, exact weights** | **`A2_cmd_strong`** | `command_le_train_f1.py checks Q 0` (seed 0 only), then `command_le_train_f1.py train Q k`. The train mode trains A2, B2 and C2 in that order, skipping finished ones; there is no option to train one arm alone. | `command_le_train_f1.py evaluate Q k` |
| Fixed fallback, training mix of currents | `B2_cmd_train` (w(c) = 1) | the same train job | the same |
| Untargeted control of the fixed fallback | `C2_untargeted` (reads A2's per-chunk no-credit shares) | the same train job, after A2 | the same |
| Estimated weights | `A3_estimated` | `command_le_train_est.py checks Q k` (all seeds), then `command_le_train_est.py train Q k` | `command_le_train_est.py evaluate Q k` |
| Untargeted control of the estimated weights | `C3_untargeted` | the same train job, after A3 | the same |

- Qualities: medium-expert and expert. Seeds: 0-4.
- Desktop window: 30 runs on 2026-10-14, 03:43-03:50Z (`RESULTS_fallback_rerun.md` L8).
- The pass marks were scored by `score_fallback_rerun.py` (on top of `score_training_test.py`), which wrote
  `train_f1/SCORE.json`. Its paths are hard-coded to `/root/bca-shift/runs/...` (score_training_test.py L15-16), so it
  cannot run in the sandbox as is.

### Reads (result F's `T.World`, command_le_train.py L134-164)

All paths below are under `<root>` = `REPO/runs/shift_iw_v5`, and REPO comes from the imported package.
- **Data:** `<root>/Q/data/seed k/` (`run.load_data`: stage.json with sidecar, world.json).
- **Frozen gapped host:** `<root>/Q/host/gap/seed k/` (record.json with sidecar, snapshot_100.msgpack); the critic is
  used.
- **Band:** `<root>/Q/band/gap/seed k/` (record.json with sidecar, band.msgpack, `band_valid`, the threshold R_u).
- **Approval:** `<root>/approval.json` with sidecar (`approval_for(ctx, "bca_lower_edge")`).
- **Start banks:** `<root>/start_banks/` (evaluate only).
- **From the script's own folder:** `PASSMARKS_fallback_rerun.md` with its sidecar, `command_le_train.py` and
  `command_band_check.py`.
- **checks mode also reads:** the room-lower-edge record and actors (`le_none`).
- **command_le_train_est.py also reads:** `PASSMARKS_estimated_weights.md`, and fits version 5's classifiers
  (`run.classifiers_of`). Its checks mode also uses `CB.truth_error`.
- **room and room-lower-edge read:** data and the gapped host. Room-lower-edge also reads the room record and its
  `none` actor (for check (e) and strengths).
- **band reads:** data, host, the room record, approval `band`, and the report (`require_design`).
- **explore_arms --lower-edge reads:** data, host, band, room, and approval `bca_lower_edge` with its `exploratory_go`.

### Writes

The channel scripts write no sidecars. Their outputs are:
- `REPO/runs/shift_iw_v5_explore/channel/train_f1/Q/{A2_cmd_strong,B2_cmd_train,C2_untargeted}/seed k/{actor.msgpack, record.json}`
- `.../train_f1/Q/checks_seed0.json`
- `.../train_f1/Q/evaluate_seed k.json`
- the estimated-weights equivalents under `train_est/`

The `run.py` stages write records with sidecars under `<root>/Q/{room,room_lower_edge,band/gap,evaluate}/seed k/`.

### Gates

- **Spec:** iw/spec.json must equal its sidecar (a71af9bd…).
- **Records:** every input record must be finished, with a matching sidecar and the same spec.
- **Pass marks:** the frozen pass-mark file must equal its sidecar (`passmarks_ok`: f1 8b8f7a27…, first test
  204f53ff…, est 4560769a…).
- **Approval:** `verify_approval` checks the spec SHA, the 41 code hashes and the scope, and requires the scope to
  cover the quality.
- **Band loading:** `LI.load_band` with stage `bca_lower_edge` checks the band's data SHA through `require_gate`.
- **Grid size:** the table must have at most 4,096 grid points.
- **Band stage:** approval `band`, the room record, and the report gate (D1 pass on seeds 0-4, same code).
- **Room-lower-edge:** check (e) on the oracle penalty can stop a seed. On the desktop it did not stop any seed.
- **explore_arms:** skips the report's M1/L1/L2 gate under the exploratory go. The official `bca-lower-edge` stage
  would refuse, because L2 failed.

## 2. Settings, scoring and banks

**Settings** are `learners_iw.settings()`, printed by the dry check:
- master seed 20261016, alpha 2.5, batch 256, Adam 3e-4, discount 0.98, policy delay 2, target noise 0.2 clipped at
  0.5;
- host: 100,000 updates;
- band fit: 20,000 steps and one refresh on the kept calibration bank.

**Result F's training**, on top of `lower_edge._loop` and `actor_step` unchanged; only the penalty differs:
- Actor-only training against the frozen gapped host critic.
- It starts from `learners.clone`: 20,000 steps of cloning on the kept rows, deterministic from the data.
- 20,000 steps in 5 chunks of 4,000, on one shared stream of batch rows.
- Data seed k seeds the clone, the rows and the band.
- **Refresh at the start of every chunk (5 refreshes):**
  - the bank weights at the current actor's commands, using the exact ratio
    w(c) × π_h(a|s) × Z(s) / β(a|s): the strong law, an actor spread of 0.2 clipped at 0.5, and the recording
    controller's exact command density;
  - the table of thresholds R, on a grid of ratio 1.05 with floor 1e-4;
  - U(s, a_i) at the recorded commands.
- **Penalty, with k = 1:**
  - where the ratio is infinite (inside the removed hold, or a density of 0): W = [Q1(s,a) − Q1(s,a_i)]_+;
  - where the bank is too small: version 5's capped threshold;
  - elsewhere: [U(s,a) − U(s,a_i)]_+.

**Scoring** is `learners_iw.evaluate_actor`:
- the exact float64 100-step return (`truth.rollout`) from all 1,000 fixed starts of each start bank: none (the
  training mix), mild and strong;
- the mean J and its SE;
- points = 100 (J − J_random) / (J_stall_aware − J_random), using the banks' anchors;
- every actor uses the gapped normalization (`data.prepare(data, "gap")`).

**Pass marks** are paired by seed, using the mean and the one-sided 95% t bound with 4 degrees of freedom. The bar is
1.8 return units.

## 3. Result F's recorded seed-0 numbers (desktop)

J is the exact 100-step return; closer to 0 is better. Points are in brackets. Each cell cites its file and line.

| Row | Quality | Strong J (points) | No-shift J (points) | Mild J | Source (line of the strong J; none is L8 / mild L16 for A2) |
|---|---|---|---|---|---|
| No extra caution (`le_none`) | medium-expert | −176.3052 (92.47) | −65.6174 (96.02) | −111.7186 | `outputs/shift_iw/explore/RESULTS.json` L13348 (none L13272), `per_seed[0]` |
| Version 5's lower edge, unweighted | medium-expert | −176.4890 (92.46) | −65.5007 (96.04) | −111.6984 | RESULTS.json L12768 (none L12692) |
| Perfect lower edge (`le_oracle`) | medium-expert | −107.9638 (97.85) | −51.5200 (97.87) | −75.7694 | RESULTS.json L13464 |
| **A2, fixed fallback, exact** | medium-expert | **−152.4372 (94.35)** | **−61.0905 (96.62)** | −99.6825 | `channel/results/train_f1/medium-expert/evaluate_seed0.json` L24 |
| B2, training mix | medium-expert | −152.0057 (94.38) | −61.0367 (96.62) | −99.4472 | same file, L50 |
| C2, untargeted | medium-expert | −176.4892 (92.46) | −66.4802 (95.91) | −112.4117 | same file, L76 |
| A3, estimated | medium-expert | −177.2695 (92.39) | −66.6624 (95.89) | −112.8909 | `channel/results/train_est/medium-expert/evaluate_seed0.json` L24 |
| C3, untargeted (estimated) | medium-expert | −175.3967 (92.54) | −65.7154 (96.01) | −111.3893 | same file, L50 |
| No extra caution (`le_none`) | expert | −197.3478 (90.81) | −66.2427 (95.94) | −120.4864 | RESULTS.json L17408 (none L17332) |
| Version 5's lower edge, unweighted | expert | −196.7898 (90.86) | −66.0150 (95.97) | −120.2404 | RESULTS.json L16828 |
| Perfect lower edge (`le_oracle`) | expert | −87.7753 (99.44) | −42.1624 (99.09) | −61.9830 | RESULTS.json L17524 |
| **A2, fixed fallback, exact** | expert | **−116.4035 (97.18)** | **−49.4519 (98.14)** | −78.0021 | `channel/results/train_f1/expert/evaluate_seed0.json` L24 |
| B2, training mix | expert | −118.8165 (96.99) | −49.9335 (98.07) | −79.2414 | same file, L50 |
| C2, untargeted | expert | −197.2922 (90.82) | −66.6005 (95.89) | −120.6149 | same file, L76 |
| A3, estimated | expert | −165.3733 (93.33) | −59.3553 (96.84) | −102.5188 | `channel/results/train_est/expert/evaluate_seed0.json` L24 |
| C3, untargeted (estimated) | expert | −197.8844 (90.77) | −66.6274 (95.89) | −120.8813 | same file, L50 |

Notes on the table:
- `channel/` means `outputs/shift_iw/explore/channel/`.
- The repository has no per-seed v5 evaluation files (`runs/` is not committed). The reference rows come from
  `RESULTS.json`, whose `per_seed` lists are in seed order 0-4.
- They agree exactly with SCORE.json's paired differences: A2 − `le_none` reproduces SCORE's seed-0 value to 4
  decimals.
- The learning-curve reruns' final points (`curves/results/Q/seed0/{f1,oracle,band,est}.json`) equal these values,
  and all have `final_matches_saved: true`.

### Paired differences in SCORE.json (`channel/results/train_f1/SCORE.json`)

Each entry gives the seed-0 value (points in brackets), the 5-seed mean, the lower bound and the per-seed values.

| Difference | medium-expert | expert |
|---|---|---|
| A2 − v5 unweighted, strong (U1) | seed 0 **+24.05** (1.89 points); mean +24.97, lower bound +22.57; per seed 24.1, 23.0, 26.2, 22.8, 28.8 (L7) | seed 0 **+80.39** (6.33 points); mean +74.24, lower bound +68.29; per seed 80.4, 65.6, 80.0, 71.0, 74.3 (L557) |
| A2 − No extra caution, strong | seed 0 +23.87 (1.88 points); mean +25.28 (1.99 points); per seed 23.4 to 29.6 (L295) | seed 0 +80.94 (6.37 points); mean +74.10 (5.83 points); per seed 65.6 to 80.9 (L845) |
| A2 − v5 unweighted, no shift (U3) | seed 0 +4.41; mean +4.45 (L55) | seed 0 +16.56; mean +15.58 (L605) |
| A2 − C2, strong (U2) | seed 0 +24.05; mean +23.86 (L79) | seed 0 +80.89; mean +73.44 (L629) |

- The +2.0 / +5.8 points in the task are formally U1, which is measured against version 5's unweighted lower edge.
  Against "No extra caution" the gains are +1.99 and +5.83 points; they round the same.
- Per-seed U1 is also in `RESULTS_fallback_rerun.md` L35.

### Comparison aids for the sandbox rerun

- **The desktop's own spread.** It is the SD of J over the f1 curve checkpoints at 16,000-20,000 steps, from
  `curves/results/Q/seed0/*.json`:

  | Run | Quality | Strong SD | No-shift SD |
  |---|---|---|---|
  | A2 | medium-expert | 5.0 | 2.2 |
  | A2 | expert | 4.3 | 0.6 |
  | `le_none` | medium-expert | 3.5 | 1.6 |
  | `le_none` | expert | 3.2 | 0.6 |
- **Free bit checks.** The f1 records hold the clone's and the stream's hashes. Both depend only on data and settings,
  so the sandbox should reproduce them if its arithmetic matches.
  - medium-expert: clone `3545e4235f4820c9…`, rows `c837711bb99631f8…`
  - expert: clone `e0096a0416e96dfa…`, rows `062ebb7ab80d685a…` (`train_f1/expert/A2_cmd_strong/seed0/record.json` L123
    and L65)
- **The trained actors will not match bit for bit.** The sandbox's host already differs from the desktop's (REPRO.md:
  "within the run's own spread", not bit for bit), and the critic and band feed every arm.
- **The meaningful check** is therefore the sandbox's own paired gap, A2 − `le_none` on the strong starts. It should
  be positive and close to the desktop's 5-seed range: medium-expert 23.4 to 29.6, expert 65.6 to 80.9.

## 4. Approval and the owner's go: what covers these rows

**Committed approval:** `outputs/shift_iw/v5/approval.json`, SHA-256 c6b374d9…, which matches its sidecar.
- `scope`: `["band", "coverage", "bca", "bca_lower_edge"]`.
- `qualities`: `{"bca": [all five], "bca_lower_edge": [all five]}`.
- `user_approval_verbatim`: "Question: Approve version 5 (freeze the spec, then data, TD3+BC, room checks, BCA's band
  and coverage for all five qualities), with confirmations (a) and (b) as described? Answer: Yes, freeze and run".
- `exploratory_go.answer_verbatim`: "Run the arms anyway, labelled exploratory (Recommended)" (2026-10-13). The earlier
  go is recorded in `experiments/iw/owner_go_bca_arms.txt`: "Yes, both kinds, all qualities (Recommended)", covering
  the capped pull and the lower edge.

**Result F's own go:** `PASSMARKS_fallback_rerun.md` L6-11, frozen with its sidecar 8b8f7a27…:
- Question: "Expert data show the return gain; medium-expert loses … What next?"
- Answer: **"Fix the fallback, then rerun (Recommended)"**.

The tests before it:
- `PASSMARKS_training_test.md` L4-6: "Approve as drafted (Recommended)".
- `PASSMARKS_estimated_weights.md` L6-7: "Proceed with all three".

**Today's request:** the owner's request in chat, "result f is solid lets do that".

**Dry check, read only** (`resultF/dry_gates.py` → `resultF/dry_gates.out.json`). It imports the sandbox's copy of the
code and the channel scripts from the repository folder.
- **Verified for both qualities:** `approval_for` with `band`, `coverage`, `bca` and `bca_lower_edge`.
- **Refused by scope:** `bca_estimated` and `bca_lower_edge_estimated`. No result F script asks for them, though:
  `command_le_train_est.py` checks only `bca_lower_edge`.
- **The pass-mark checks of f1, the first test and est all pass.**
- **The scripts' output folders resolve inside the sandbox.**
- **`T.World("expert", 0)` refuses at the band:** "the gap band of seed 0 has not finished".
- **Nothing in `/home/user/bca` or `iw_rerun/` changed** (directory listing and `find -newer` checked before and
  after).
- **The code copy is still byte-identical** to the repository: 45 of 45, rechecked today.

**Is a seed-0 rerun in the sandbox, with the sandbox's own host, covered?**
- **Code gates: yes.** The approval binds the spec SHA, the 41 code hashes, the scopes and the qualities. It does not
  bind a machine or a host, and the sandbox's code is identical.
- **The owner's words: yes.** Result F was a go to train these arms (seeds 0-4, both qualities). Today's message asks
  for its actor in the live panel, which needs this rerun.
- **Estimated weights:** covered by "Proceed with all three" and by the code gate. Neither the approval file nor
  `owner_go_bca_arms.txt` names this arm, and `owner_go_bca_arms.txt` excludes version 5's own estimated arms. It is
  optional; I would include it only if the orchestrator wants it.
- **No new approval file is needed, and none may be written.**

**The one non-approval gate, the band stage's report (D1 pass on seeds 0-4).** There are three ways to meet it:
- **(b) Recommended: build it in the sandbox.**
  - `run.py data --quality Q --seeds 1 2 3 4` (8 jobs, about 98-125 s each as measured here), then
    `run.py report --quality Q` (no approval; marks without inputs are "pending").
  - Free bit check: the committed `outputs/shift_iw/v5/Q/report.json` holds each seed's D1 values at full precision,
    so seeds 1-4 can be compared with the desktop's.
  - Disk: about 67 MB per data seed, so 8 × 67 MB ≈ 540 MB. 15 GB is free.
- **(a) Faster but hazardous: link the committed report read-only**, as the approval was linked:
  `<root>/Q/report/report.json` → `outputs/shift_iw/v5/Q/report.json`.
  - It would pass: the sidecar matches, status is finished, the spec matches, the seeds are 0-4, D1 is pass, and its
    `code_sha256` equals the sandbox copy's.
  - **The hazard:** `data.write_json` uses `Path.write_text`, which follows symbolic links. Any later `run.py report`
    in that tree would overwrite the committed report in `/home/user/bca`.
- **(c) Not recommended: a new driver that calls `LI.fit_band` without the gate.** That would be new code in the chain.

## 5. How long each row takes

**Desktop, per quality and seed** (`record.json` `seconds`; training only; 10 processes in parallel):

| Arm | Seconds | Note |
|---|---|---|
| A2 | 128-140 | seed 0: medium-expert 139.4, expert 138.7 |
| B2 | 125-138 | |
| C2 | 103-110 | |
| A3 | 151-172 | |
| C3 | 113-115 | |

- Setup before A2 took about 1 min: loading, bank scores and the 20,000-step clone. A2 started at 03:44:01Z.
- One f1 job (A2 + B2 + C2) took about 7.3 min of wall clock on the desktop.

**Desktop estimates for the other stages** (spec.json `compute`, measured on synthetic arrays or estimated):

| Stage | Estimate |
|---|---|
| Room check | 12-15 process-min |
| Room-lower-edge | 7-10 process-min (397 s measured without matching, plus about 45 s per matching round) |
| Band and coverage | about 1 min each |
| Lower edge, three arms | about 2.7 min, so one arm ≈ 1.5 min |
| Evaluation | seconds per actor |

**Cloud speed against the desktop** (from `iw_rerun` logs and PLAN.md):
- TD3+BC: 6.5 ms/update in the probe, against the spec's 6.5 min for 100,000 updates on the desktop. Under the actual
  4-job load, 23-30 min.
- So the cloud runs about 1.5-2× slower unloaded, and about 3-4.5× slower when other sessions load the CPUs.

**Plan on 4 CPUs, per quality, cloud estimate** (one CPU per job, single-threaded, ARITH=desktop as in `run_seed0.sh`):

| Step | Cloud estimate | Needs |
|---|---|---|
| data seeds 1-4 (4 jobs) | about 8 CPU-min | — |
| report | under 1 min | data |
| room | 20-45 min | host |
| room-lower-edge (No extra caution, Perfect lower edge) | 12-30 min | room |
| band, `--kinds gap` | 2-6 min | room, report |
| f1 train (A2, B2, C2) | 11-22 min | band |
| f1 checks (optional, 300 steps) | 2-4 min | band, room-lower-edge |
| explore_arms `--lower-edge unweighted` (optional comparison row) | 3-5 min | band, room |
| est train (A3, C3; optional) | 8-17 min | band |
| evaluations (f1, est, room_lower_edge, explore_eval) | 1-3 min | — |

- **Critical path** per quality: room → band → f1 → evaluate, about 35-75 min.
- **Both qualities together:** about 120-260 CPU-min including the optional rows. The data and report jobs, and
  room-lower-edge, fit beside the critical path on the other two CPUs.
- **Expected wall clock:** about 1-1.5 h, or up to about 2.5 h at today's load.

## 6. Notes for the plan

- **Working tree.** Make a fresh `resultF/repo` holding:
  - a byte-identical copy of the 45 files, checked against `/home/user/bca`;
  - SHA-checked copies of `iw_rerun/repo/runs/shift_iw_v5/{start_banks, Q/data/seed0, Q/host/gap/seed0}`. Record links
    are relative to `<root>`, so they stay valid;
  - the approval symlinks, as in the earlier rerun.

  All outputs then land under `resultF/` and `iw_rerun` stays as it was.
- **Run result F's scripts in place, from the read-only repository folder.** Use `PYTHONPATH=resultF/repo`, cwd
  `resultF/repo`, and `-B`.
  - Their output folders come from the imported package, not from their own folder. The dry check confirmed this
    (`f1:OUT` = `.../repo/runs/shift_iw_v5_explore/channel/train_f1`).
  - This means no pass-mark file is copied or linked anywhere.
  - Check the repository folder for changes before and after each job.
- **Hygiene.** Some process created `__pycache__/*.pyc` in `iw_rerun/repo` after its DONE (23:55). The sources are
  unchanged and the stale bytecode is harmless, but every job here should run with `-B`.
- **Scoring.** `score_fallback_rerun.py` cannot run as is because of its hard-coded `/root/bca-shift` paths. A small
  read-only sandbox scorer would read the sandbox's evaluate files and the desktop values in section 3. It is new code,
  so it should be labelled EXPLORATORY.
- **No smoke mode.** `command_le_train*.py`, room, room-lower-edge and band have no documented smoke or dry mode, so
  nothing beyond the read-only dry check was run.

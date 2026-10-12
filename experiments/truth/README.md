# D4RL-RT: truth by simulator rollouts for real Hopper and Walker2d critics

This directory is the instrument for test world C, "D4RL with rollout truth". The world uses the two datasets the
repo trains TD3+BC on (hopper-medium-v2 and walker2d-medium-replay-v2) and their learned TD3+BC critics. The
critic's true error at a logged or visited (s, a) is measured by putting MuJoCo back into that state and rolling
the frozen policy out. Seven acceptance tests (R1-R7), each with a pass mark fixed in advance, decide whether this
world can answer the BCA question: does putting caution where the critic is actually wrong improve return,
compared with spreading the same caution evenly?

**Status.** The instrument is built and tested end to end on synthetic data only. The D4RL files cannot be
downloaded in the build container, and the trained hosts live on the user's machine. No real number exists yet.
No acceptance stage has run on real data. The world has not been judged. No BCA arm exists here. The one BCA-band
step (R5-signal) is implemented but stays off unless `--allow-bca-band` is given. It was never run.

## Files

| File | What it does |
|---|---|
| `state_restore.py` | Converted row to raw row to qpos/qvel (`infos/*`, or rebuilt from the observation). Exact env step on bare MjSims. The one-step reproduction gate. |
| `labeller.py` | Batched rollout truth Q^pi(s, a): many MjSims in lockstep, one fixed-shape jitted actor call per step. Deterministic, jitter and noisy-target modes. Resumable shards. |
| `synthetic.py` | D4RL-format HDF5 files made in the real simulator. Tests only. Never findings. |
| `pools.py` | Restores a frozen host from its train.py run directory. Builds pools L, Pi and T. kNN thinness. Critic queries. |
| `extraction.py` | The Level-1 screen (R4a) and actor extraction on the frozen critic (R3b, R4b, R7). |
| `acceptance.py` | Stage 0, the pre-registration and its SHA-256 gate, labels, R1-R7 scoring, the graft, the verdict table, the command line. |
| `gap.py` | Action gap, data only: the densest-cluster search, the frozen manifest `gap.json` and its sha256, the gapped HDF5 copy, the projected reservation, the frozen test pools, G0a and G0b. |
| `gap_train.py` | Trains a gapped TD3+BC host with the repo's own `train.py` (patched `resolve` and `_reserve` at run time; no file changes). No BCA path. |
| `gap_acceptance.py` | Action gap protocol: the gap pre-registration, Stage 0 and H0, the final freeze and its SHA-256 gate, pools and labels per host, G1-pre to G4b and R7, the stop rule, the verdict table, the command line. |
| `test_*.py` | CPU tests on synthetic data (see Tests). `test_review.py` pins the defects found in the adversarial review. `test_gap.py` covers the action gap. |

The module docstrings hold the exact definitions: every pool, slot, statistic and pass mark.

## What each stage needs

W is a world directory, one per task and plan, for example `runs/truth/hopper`. Every stage after `freeze`
takes `--prereg-sha256 HEX`. It refuses to run unless `W/prereg.json` hashes to HEX, the sidecar agrees and the
embedded body hash is right. It also refuses if any analysis source file changed since the freeze (override:
`--allow-code-change`, which marks every output). The stage functions repeat this check themselves
(`acceptance.verify_stage`), so a direct Python call with an edited body or another hash refuses too. Every stage
also checks that the host's discount and target-smoothing noise equal the pre-registered label settings, and the
host restore checks that the critic was trained on the raw logged rewards (the units of the rollout truth).

| Stage | Data file | Host checkpoint | Simulator | Trains | Pre-registration | Writes |
|---|---|---|---|---|---|---|
| `stage0` | yes | optional (for the split and the eps-noise share) | 1,000-row gate | no | not needed | `W/stage0/summary.json` |
| `freeze` | no | reads its sha256 | no | no | creates it | `W/prereg.json`, `.sha256` |
| `pools` | yes | yes | Pi episodes | no | yes | `W/critic_<seed>/pools/` |
| `labels` | yes | yes | yes (bulk) | no | yes | `W/critic_<seed>/labels/<job>/` |
| `score` | no | no | no | no | yes | `W/critic_<seed>/scores/R1, R2, R3a, R5_world, R6` |
| `screen` | yes | yes | no | no | yes | `W/critic_<seed>/scores/R4a.json` |
| `extract` | yes | yes | forbidden | **actors** | yes | `W/extraction/critic_<seed>/matched.json` + lock |
| `evaluate` | yes | yes | yes | no | yes | `W/extraction/critic_<seed>/evaluation.json` |
| `extraction-score` | no | no | no | no | yes | `W/extraction/R3b, R4b, R7.json` |
| `verdict` | no | no | no | no | yes | `W/verdict.json`, `W/verdict.md` |
| `r5-signal` | yes | yes | no | **BCA band fit: OFF** | yes + `--allow-bca-band` | `R5_signal.json` |

`run` executes every planned stage in order and skips finished ones. `combine` applies the design's cross-task
R4b rule (pass on one task, the other not negative) to two `verdict.json` files.

**Where assets are found.** The data file is `<data dir>/<filename>` from `configs/td3_bc.yaml`
(`hopper_medium-v2.hdf5`, `walker2d_medium_replay-v2.hdf5`), sha256-checked against the pin. The default data
dir is `~/.d4rl/datasets`. A host is a train.py output directory holding `resolved.json`, `preparation.json`,
`result.json` and `checkpoint_<step>.msgpack`. It is restored the way `experiments/wbcp/freeze_scores.py`
restores runs: the row is typed, the preparation is rebuilt and checked against `preparation.json`, and the
checkpoint is restored leaf for leaf. The two existing 1M hosts (seed 202609171) have accepted audits in
`outputs/standard_bca/td3_bc/{hopper,walker2d}/host/s202609171/verified-*/audit.json`. When the run's
`resolved.json` is the audited file, the checkpoint must be the audited one (Hopper 1M sha256 `615886be...`,
Walker2d `6bcf5c59...`). Those rows come from the archived study. They type on this branch after one adapter
(`pools.adapt_archived_row`), which drops `protocol.reference` (None) and sets `rows_per_episode` to None. A test
checks this against `experiments/standard_bca/manifest.json.gz`. A missing file stops the stage with its path and
pin in the message.

## Commands on the user's machine (Windows + WSL)

Run these in the WSL shell from the repo root, with the venv the repo's scripts use (`/opt/bca-venv`, as in
`experiments/wbcp/host_matrix.py`). Use WSL paths for Windows drives (`/mnt/c/...`).

```bash
export JAX_PLATFORMS=cpu OMP_NUM_THREADS=1 D4RL_SUPPRESS_IMPORT_ERROR=1
export LD_LIBRARY_PATH=$HOME/.mujoco/mujoco210/bin:$LD_LIBRARY_PATH
PY=/opt/bca-venv/bin/python
HOP=/path/to/runs/td3_bc-hopper-host-s202609171      # holds checkpoint_1000000.msgpack
```

### 1. The graft first (judge's external-validity check; no training, about 1.5 process-hours)

Pools L and Pi on the one existing Hopper checkpoint, scored for R1, R2, R5-world and R6.

```bash
$PY -m experiments.truth.acceptance stage0 --world runs/truth/hopper-graft --task hopper --run-dir $HOP
# read runs/truth/hopper-graft/stage0/summary.json: gate misses, negative control, infos keys, thinness
SHA=$($PY -m experiments.truth.acceptance freeze --world runs/truth/hopper-graft --task hopper \
      --host 202609171=$HOP --plan graft)      # add --tol-next-obs X etc. if the pilot asks for it
$PY -m experiments.truth.acceptance pools  --world runs/truth/hopper-graft --prereg-sha256 $SHA --critic 202609171
for i in 0 1 2 3 4 5 6 7; do
  $PY -m experiments.truth.acceptance labels --world runs/truth/hopper-graft --prereg-sha256 $SHA \
      --critic 202609171 --worker $i --workers 8 > runs/truth/hopper-graft/labels_$i.log 2>&1 &
done; wait
$PY -m experiments.truth.acceptance run --world runs/truth/hopper-graft --prereg-sha256 $SHA
cat runs/truth/hopper-graft/verdict.md
```

`freeze` refuses when the Stage 0 pilot gate does not pass under the chosen tolerances, and also when the
misaligned-by-one negative control passes under them (tolerances that loose cannot certify a restore). If the
file's `infos/*` states fail the gate but `gate_rebuilt` in the Stage 0 summary passes, freeze with
`--state-source rebuilt`: every pool state is then rebuilt from the observation (x = 0; rows at the velocity clip
dropped and counted), and the choice is hashed with the rest. Choose the tolerances from the
pilot's p99 misses before freezing. They are hashed with everything else.

### 2. The full gate (needs the user's approval: it trains actors)

R4 needs three critics per task. Two new host seeds per task must be trained first (about 49 min each on the RTX
5070 Ti, per the design):

```bash
$PY train.py --algorithm td3_bc --dataset hopper --method host --seed 202609172 --output runs/td3-hopper-host-s202609172
$PY train.py --algorithm td3_bc --dataset hopper --method host --seed 202609173 --output runs/td3-hopper-host-s202609173
```

Caveat: on this branch train.py withholds K = 6 rows per episode (Walker2d 23) for the bank. The archived
202609171 hosts withheld whole episodes (1,103 and 1,149 rows). Each critic is scored on its own split, so this
works, but the three critics then differ in their training pools. Retraining 202609171 on this branch avoids that.

```bash
W=runs/truth/hopper
$PY -m experiments.truth.acceptance stage0 --world $W --task hopper --run-dir $HOP
SHA=$($PY -m experiments.truth.acceptance freeze --world $W --task hopper --plan full \
      --host 202609171=$HOP --host 202609172=runs/td3-hopper-host-s202609172 \
      --host 202609173=runs/td3-hopper-host-s202609173)
for C in 202609171 202609172 202609173; do
  $PY -m experiments.truth.acceptance pools --world $W --prereg-sha256 $SHA --critic $C
  for i in 0 1 2 3 4 5 6 7; do
    $PY -m experiments.truth.acceptance labels --world $W --prereg-sha256 $SHA --critic $C --worker $i --workers 8 &
  done; wait
done
$PY -m experiments.truth.acceptance run --world $W --prereg-sha256 $SHA    # score, screen, extract, evaluate, verdict
# repeat with --task walker2d and W=runs/truth/walker2d, then:
$PY -m experiments.truth.acceptance combine runs/truth/hopper/verdict.json runs/truth/walker2d/verdict.json
```

`extract` can also run per critic in parallel (`extract --critic C`), then `evaluate --critic C`, then `run`.
Label jobs resume shard by shard after an interruption. The shard-to-worker map is rotated per job, so the
one-shard roughness and gap jobs spread over workers instead of all landing on worker 0. Scoring checks that each
label job's manifest inputs equal its pool's states and actions, so labels made for an older pool are refused. A
pools directory is written once: delete it (and its labels) to rebuild.

## Measured throughput here (engineering timings, not findings)

Build container: 4 CPUs shared with another agent, one process, synthetic data, 2026-10-11. All POST HOC.

| Quantity | Hopper | Walker2d |
|---|---:|---:|
| Labeller, 256 slots, upright PD policy (1,000-step rollouts) | 9,215 env steps/s | 5,898 env steps/s |
| Labeller, 256 slots, random-init host actor (short rollouts) | 8,343 env steps/s | 6,144 env steps/s |
| Physics only (no policy, no restores) | 13,414 env steps/s | 9,629 env steps/s |
| Actor-only extraction update, batch 256 (design assumed 3 ms) | 1.58 ms | 1.61 ms |
| Level-1 operator at 50,000 pool-T rows: setup / 20 arms | 2.4 s / 5.3 s | 2.5 s / 5.0 s |

The full synthetic pipeline (plan `full`, two random-init hosts, smoke sizes) runs in about 40 s for Hopper and
70 s for Walker2d here.

## Estimated cost on the user's machine (arithmetic from the rates above)

| Item | Hopper | Walker2d |
|---|---:|---:|
| Labels per critic (design: about 169k / 205k rollouts, 118M / 184M env steps) | 3.6 process-h | 8.7 process-h |
| Graft (one critic, pools L and Pi only: about 69k rollouts, about 48M env steps) | 1.5 process-h | not planned |
| Extraction per task (12 units x 55 actor runs x 20k updates at 1.6 ms) | 5.9 process-h | 5.9 process-h |
| Evaluation per task (12 units x 20 policies x 100 episodes, at most 1,000 steps) | at most 0.7 process-h | at most 1.1 process-h |
| R3b on-policy labels per task (12 units x 500 states x 2 rollouts) | at most 0.4 process-h | at most 0.6 process-h |

Three critics per task: labels 36.7 process-h in all, extraction 11.8, evaluation and R3b labels at most 2.8. That
is about 51 process-h, plus 4 new host runs (about 3.3 h on the GPU). On 8 worker processes that is about 7 h of
wall time. WSL can be 2-3x slower than this container, which would make it 15-20 h. Stage 0 takes minutes.

## What never runs here

- **R5-signal** (BCA's frozen q1 band against the truth). It fits a BCA scale network. It refuses without
  `--allow-bca-band`. It was never run, so beyond its refusal it is untested.
- **BCA arms.** None are implemented. Once the world passes, BCA's dose (`bc_readout`, blend 0.5) would go into the
  same `bc_multiplier` slot the arms use.
- **D-thin / D-mix hosts.** `pools.d_thin_mask` computes the D-thin(f) row mask, and Stage 0 reports its row counts
  for f = 0.1. Training hosts on it needs a data-mask flag in train.py. That is not built, because existing files
  are not modified in this work.

## Tests

```bash
cd /path/to/BCA
JAX_PLATFORMS=cpu OMP_NUM_THREADS=2 LD_LIBRARY_PATH=$HOME/.mujoco/mujoco210/bin D4RL_SUPPRESS_IMPORT_ERROR=1 \
  python -m unittest experiments.truth.test_state_restore experiments.truth.test_labeller \
  experiments.truth.test_pools experiments.truth.test_acceptance experiments.truth.test_review \
  experiments.truth.test_gap
```

85 tests, 187 s here (4 shared CPUs, 2026-10-12, after the action-gap review): 53 for option C (44 from the build, 9
in test_review.py) and 32 in test_gap.py (about 80 s; see the action-gap section below).
The end-to-end tests write synthetic files and random-init host runs, then run Stage 0, freeze, and every planned
stage. Their outputs say SYNTHETIC and are test fixtures, not findings.

## Action gap (hopper-medium-v2 first)

The user's requirement (2026-10-12): remove a common action from the training data and reintroduce it during
evaluation. The common action is the densest cluster of logged actions, about 15% of the training rows (the
smallest Euclidean ball in raw action space holding 15% of them). It is fixed and frozen before any training. Its
states still appear, through the other rows and as the next observation of the row before. Evaluation does two
things. (a) The held-out error test: the removed rows become a test pool, and the gapped critic's true error there
(rollout truth) is compared with kept rows, and causally with full-data critics of the same seed. (b) The policy
test: gapped hosts run in the simulator, where the action is allowed again. We measure whether they choose it, how
wrong their critic is there, and the return cost against full-data hosts of the same seed. Hopper-medium-v2 comes
first; walker2d-medium-replay-v2 then uses the same code.

**Status.** Built and pre-registrable; tested on synthetic data only. Nothing ran on real data: the D4RL files and
the hosts are not in this container. No host was trained here. No BCA arm exists and no BCA band is fit: `gap_train`
refuses `--method bca` with the `--allow-bca-band` wording and exposes no BCA flag.

### How it works (definitions are in the module docstrings)

- **The gap** (`gap.py define`). Computed on this branch's native training split (configs/td3_bc.yaml reservation:
  size 1024, seed 911, K = 6 for Hopper and 23 for Walker2d). c* minimises rho_q(c), the 15%-quantile distance. The
  search is seeded (20261012): 4,096 training actions, every box corner and the median as candidates, scored on
  200,000 rows (Walker2d: all rows), then the best 16 refined by compass search. The ball is inclusive, so a tie block
  is never split. Fallbacks: a tie block over 16% steps down to the next distinct distance (accepted at >= 14%); if
  that fails, the next refined centre is tried. G0 fails when no centre lands in [14%, 16%] or rho < 0.05 (the
  point-mass guard).
- **The gapped copy.** It is the original file with `timeouts[i] = True` at every gap row and the attribute
  `bca_action_gap`. `_convert` and `d4rl.qlearning_dataset` skip timeout rows, so the gapped conversion is exactly the
  native rows minus the gap rows. Row i-1 keeps its true next observation s_i. The original is never touched.
  The copy is written next to the original, as `hopper_medium-v2.gap15-<8 hex>.hdf5`.
- **The split** of a gapped host is the native split projected onto the kept rows by raw row: native training,
  withheld and bank, each minus the gap. `gap_train.py` and `pools.restore_host` patch `runtime.td3_bc._reserve` at
  run time, and only after verifying the manifest's sha256 and the pinned sha256 of `runtime/td3_bc.py`. No file
  outside experiments/truth changes. The gapped bank is about 870 of 1,024 rows; no BCA run may use it without new
  dependence evidence.
- **The frozen pools** (in `gap.json`, raw rows, the same for all six hosts of a task): G, up to 3,000 gap rows of
  native training; G_held, up to 1,000 gap rows of native withheld; K_match, each G row's nearest kept training
  state from another original episode within the caliper r_s, greedy without replacement; K_far, 3,000 kept rows
  whose 10 nearest training neighbours in (s, a) hold no gap row.
- **Pre-registration before training.** `gap_acceptance preregister` freezes the plan, sizes, seeds and every G mark
  (`W/gap_prereg.json` with sha256), embedding the gap sha256. train.py prints returns while it trains, so this
  comes first. `gap_train` enforces the order: it refuses without this file and records its sha256 in each gapped
  run before training, and H0 checks it. The marks and the protocol are compared with pinned digests of the
  approved design (`PINNED_SHA256`), so an edit made in code before preregistering shows as POST HOC in the
  verdict. After training, `stage0` (option C's restore gate on the gapped copy, plus H0 on the six hosts) and
  `freeze` write `W/prereg.json`. That file embeds the gap sha256, the gap pre-registration sha256 and the
  checkpoints. Every later stage refuses on a hash mismatch, on an edited gap.json or gap pre-registration, or on a
  changed analysis source.
- **Stages and stop rule.** G0a/G0b (data only, `gap check`); H0; G1-pre; G1; G2; G3; G4a; G4b with R7. Their pass
  marks are those of the approved design (`gap_acceptance.PASS_MARKS`, documented in its docstring). G4a runs only
  when three things pass: G1 (all of a-e), G2, and G3's mechanism part. G4b runs only when G4a passes for at least
  one placement (OR_abs or OR_GAP). `run` applies this rule. Stages it blocks show "not run (stop rule)" in the
  verdict.

### Commands on the user's machine (Windows + WSL), in order

Use the same shell set-up as option C above (`PY=/opt/bca-venv/bin/python`, `LD_LIBRARY_PATH` exported before any
command, so train.py does not need to re-execute). Run everything from the repo root on this branch.

```bash
export JAX_PLATFORMS=cpu OMP_NUM_THREADS=1 D4RL_SUPPRESS_IMPORT_ERROR=1
export LD_LIBRARY_PATH=$HOME/.mujoco/mujoco210/bin:$LD_LIBRARY_PATH
PY=/opt/bca-venv/bin/python
W=runs/truth/hopper-gap
SEEDS="202609171 202609172 202609173"
```

**1. G0: define, check and freeze the gap (data only).** Needs the pinned `~/.d4rl/datasets/hopper_medium-v2.hdf5`
(sha256 5bdf1bc4..., or pass `--data-dir`). No host, no simulator, no training. Under 15 CPU-minutes and 2 GB of RAM,
plus disk for one copy of the file.

```bash
$PY -m experiments.truth.gap define --world $W --task hopper              # prints gap_sha256 and the gapped copy
GAP=$(sha256sum $W/gap/gap.json | cut -d' ' -f1)
$PY -m experiments.truth.gap check --world $W --gap-sha256 $GAP --workers 4   # G0a + G0b -> $W/gap/g0.json
$PY -m experiments.truth.gap freeze --world $W --gap-sha256 $GAP          # refuses unless G0 passed
```

If G0 fails, stop: do not train. Report the numbers in `$W/gap/g0.json`. Any change to the gap (q = 15% is the
user's choice) needs the user's decision and a new G0 in a new world directory. Record `$GAP` before training,
for example by committing `$W/gap/gap.json` and `gap.json.sha256`, so it carries a timestamp.

**2. Pre-register the marks (before the first host run).**

```bash
GPRE=$($PY -m experiments.truth.gap_acceptance preregister --world $W --task hopper --gap-sha256 $GAP)
```

Record `$GPRE` with `$GAP`.

**3. Train 3 gapped and 3 full-data hosts** (approved to run on the user's machine). Each run uses the GPU lock,
the configs/td3_bc.yaml host block, 1M updates, checkpoints at 10k/50k/1M and the seed's evaluation bank. The
full-data 202609172/173 runs are the same runs option C's R4 needs; train each once (skip a run whose output
already exists). The archived 202609171 host is not a valid control: it withheld whole episodes and ran archived
code. Seed 202609171 is retrained on this branch.

```bash
for S in $SEEDS; do
  $PY -m experiments.truth.gap_train --task hopper --gap $W/gap/gap.json --gap-sha256 $GAP \
      --gap-prereg-sha256 $GPRE --seed $S --output runs/td3-hopper-gap15-s$S --device cuda
  [ -d runs/td3-hopper-host-s$S ] || \
  $PY train.py --algorithm td3_bc --dataset hopper --method host --seed $S --output runs/td3-hopper-host-s$S \
      --device cuda
done
```

`gap_train` refuses unless `$W/gap_prereg.json` exists and hashes to `$GPRE`. It writes that sha256 into the run's
`resolved.json` before training, and H0 checks it. It also refuses a seed that is not pre-registered. It picks the
JAX platform from `--device` before JAX loads, so the exported `JAX_PLATFORMS=cpu` does not turn a `--device cuda`
run into a CPU run. It checks the backend JAX actually uses before training and records it in `action_gap.json`.
An existing `runs/td3-hopper-host-s$S` is reused only if it is a valid control. H0 refuses any pair whose two
`resolved.json` files differ in anything but the data pin and the run id, or whose training code differs (the
`PAIR_CODE` files in `source.json`). In that case, retrain the control.

**4. Stage 0 + H0, then the final freeze** (CPU minutes; no outcome is read). `stage0` runs option C's one-step
gate on the gapped copy and H0 on the six hosts. H0 checks, by raw row, that gapped training, withheld and bank
equal native minus gap, and that the full hosts' split equals the native split. It also checks:
- checkpoint hashes, 1M updates, the checkpoint schedule and train.py's `verify_events`;
- the reservation record in each run's own `preparation.json`;
- that each gapped run was trained under `$GPRE`;
- that the two runs of each seed are identical apart from their data (the resolved row and the training code).

It reports normalization differences, a JAX platform difference within a pair, and a full control written before
the gap pre-registration. These are flagged, never a stop. If H0 fails, fix the cause and run `stage0` again: it
reuses the Stage 0 gate, keeps the failed result as `h0.failed-<n>.json` and checks again. A passed `h0.json` is
never replaced.
`freeze` refuses unless H0 passed, the gate passes and the misaligned negative control fails under the tolerances
(add `--tol-next-obs X` etc. from `$W/stage0/summary.json` if the pilot needs them; `--state-source rebuilt` as in
option C).

```bash
H=""; for S in $SEEDS; do H="$H --host gap-$S=runs/td3-hopper-gap15-s$S --host full-$S=runs/td3-hopper-host-s$S"; done
$PY -m experiments.truth.gap_acceptance stage0 --world $W --gap-sha256 $GAP --gap-prereg-sha256 $GPRE $H
SHA=$($PY -m experiments.truth.gap_acceptance freeze --world $W --gap-sha256 $GAP --gap-prereg-sha256 $GPRE $H)
```

**5. Pools and labels per host (8 workers), then score G1-pre to G3.**

```bash
for S in $SEEDS; do for K in gap-$S full-$S; do
  $PY -m experiments.truth.gap_acceptance pools --world $W --prereg-sha256 $SHA --host $K
  for i in 0 1 2 3 4 5 6 7; do
    $PY -m experiments.truth.gap_acceptance labels --world $W --prereg-sha256 $SHA --host $K --worker $i --workers 8 \
        > $W/labels_${K}_$i.log 2>&1 &
  done; wait
done; done
$PY -m experiments.truth.gap_acceptance score --world $W --prereg-sha256 $SHA      # G1-pre, G1, G2, G3
```

If `$W/scores/G1_pre.json` says `smoothed_truth_required`, `score` refuses G1 until the smoothed truth exists.
Rerun the `labels` loop (it adds job `G1_smooth` on every host by itself), then `score` again.

**6. G4 (only if the stop rule allows; trains actors: oracle and control arms only, approved for later).**

```bash
$PY -m experiments.truth.gap_acceptance run --world $W --prereg-sha256 $SHA       # screen, extract, evaluate, verdict
cat $W/verdict.md
```

`run` skips finished stages and applies the stop rule. To use more cores, run the G4 stages per gapped critic in
parallel: `screen --critic S`, then `extract --critic S`, then `evaluate --critic S`. Then run `run` once more
(it writes `g4-score` and the verdict).

**7. Walker2d** (after Hopper's verdict, same code; pause only if Hopper exposed a code or protocol fault): repeat
1-6 with `--task walker2d`, `W=runs/truth/walker2d-gap` and run names `td3-walker2d-...`. The cross-task G4b
claim: `$PY -m experiments.truth.gap_acceptance combine runs/truth/hopper-gap/scores/G4b.json
runs/truth/walker2d-gap/scores/G4b.json`. Optional extension (needs the user's OK): pairs 202609174/175 for G3a's
single pre-registered second look. G3 flags `second_look_eligible` when the 3-pair point estimate is <= -3 but the
interval holds 0.

### What each step needs and costs (arithmetic from the design and option C's measured rates; WSL may be 2-3x slower)

| Step | Needs | Hopper | Walker2d |
|---|---|---:|---:|
| G0 define/check/freeze | original HDF5 | < 15 CPU-min, < 2 GB, one file copy | same |
| Host training, 6 runs (3 gapped + 3 full; full 172/173 shared with option C) | GPU (RTX 5070 Ti, 49 min each) | 4.9 h GPU (3.3 h gap-specific) | 4.9 h GPU |
| Stage 0 + H0 + freeze | hosts, both files | CPU minutes | CPU minutes |
| G1 labels (10,000 rows x 2 x 6 hosts = 120k rollouts) | hosts, simulator | 2.6 process-h | 5.0 process-h |
| G1-pre (150 G rows x 2 x (8 + 16), 3 gapped hosts = 21.6k) | as above | 0.5 | 0.9 |
| G2 (100 episodes per host) | as above | minutes | minutes |
| G3 (29.7k rollouts) | as above | 0.65 | 1.2 |
| G4a: pool T (300k rollouts) and pool Pi central differences (81k) | gapped hosts | 6.5 + 1.8 | 12.5 + 5.6 |
| G4b extraction (12 units x 67 actor runs x 20k updates) + evaluation | gapped hosts | 7.2 + <= 1.0 | 7.2 + <= 1.5 |
| Only if G1-pre triggers it: G1_smooth (8 jitter rollouts on every G1 point, 6 hosts) | as above | about 21 | about 40 |

Total: about 3.8 process-h for G1-G3 and 16.5 for G4 on Hopper, and 7.1 and 26.8 on Walker2d. On 8 processes
that is about 7 h of wall time in all (14-20 h under WSL), plus 9.8 h of GPU. If the stop rule ends a task before
G4, the task costs G0, its training and G1-G3 (Hopper: about 4.9 h GPU + 3.8 process-h). The main power risk is
from the design: with 3 seed pairs, G3a's minimum detectable return cost is about 8 points.

### Tests and what ran here (synthetic only; not findings)

`test_gap.py` has 32 tests and takes about 80 s here. It plants an action cluster in a synthetic Hopper file made in
the real simulator (`synthetic.generate(planted=...)`). The planted centre sits near the random-init hosts' actions,
so the gapped hosts visit the gap and every in-gap path runs. The tests check:
- the gap holds 15% +- 1% and is defined identically twice;
- G0a's integrity checks all pass;
- no gap row reaches a gapped host's training rows, withheld rows or calibration bank, while a leaked row (one
  timeout flag cleared) is refused by the projected reservation and counted by the leak detector;
- the matched pool is matched;
- `gap_train --prepare-only` works through the real train.py, with a reservation override that refuses real data;
- `--method bca` refuses;
- a gapped host cannot be restored without its manifest;
- the scorers give known answers on fabricated labels;
- the whole protocol runs end to end on four random-init hosts, about 50 s, with every output marked SYNTHETIC;
- the hash gate refuses an edited pre-registration, an edited gap.json and a forged body;
- the smoothed-truth path works;
- the fixes from the adversarial review (`GapReviewTest`, below) hold.

On the test's synthetic file the planted centre is recovered within 0.008 (rho 0.095, training mass 0.150). G0a
passes. G0b fails (ii) and (iii) while (i), (iv) and (v) pass: 1,500 rows in 14-D are too few for the thinness marks.
So the tests exercise the later stages through a synthetic-only override
(`gap freeze` with `exercise_failed_g0`, recorded in `W/gap/freeze.json`) and the `synthetic_smoke` profile's
`stop_rule: exercise_all_stages`. Both refuse real data, and both are printed on every output.

### Deviations and interpretations (beyond the design's text)

- G0b's state-only counts and K_match use other ORIGINAL episodes; (iv)'s ratio is (c_s_kept + 1) / (c_s_full + 1),
  the same smoothing as (iii). G0b also reports K_match rows' d / rho.
- G1 uses only rows whose truth is finite under all six hosts; the count dropped is reported. The placebo magnitude
  is max(r, 1/r). The core/rim and termination-tercile ratios use the gapped host's pi(s) rollout lengths.
- G1-pre's smoothed truth is the job `G1_smooth`: 8 jitter rollouts on every G1 point, on all six hosts, so that
  every ratio uses one kind of truth. This is costly (table above).
- G2's pooled share sums visited steps over the gapped hosts; the bootstrap resamples hosts, then episodes;
  `host_count_min` is 2 (2 of 3). Cross-policy rates restore both hosts of a pair during `score`.
- G3(a): the t-interval is over pairs (between-pair SD, 2 df for 3 pairs) at 97.5%. Episodes are paired by reset
  seed within a pair; the per-pair episode bootstrap is reported, and it is the interval when only one pair exists.
  G3(b): o_rel is normalized by each host's mean |Q^pi| over its own labelled states; the bootstrap blocks are (host,
  episode). a_out for pi(s) = c* uses the first axis.
- G4: OR_GAP's ranking variable and CNG's d_i are ||pi_host(s_i) - c*|| at pool T rows (logged actions there are
  outside the gap by construction). CNG is averaged over the gapped critics. G4b's policies are scored on option
  C's extraction reset bank (802609171 + i). G4a/G4b and R7 use option C's code paths (`extraction.Level1Screen`,
  `Extractor`, blind matching under `forbid_simulator`, the lock) with OR_GAP and ANTI_GAP added.
- The Stage 0 gate samples converted rows of the gapped copy. Gap rows are not converted rows there, but their
  states are the same raw rows (`infos/*`) as in the original.
- Additions from the review (below): G0a reruns the whole search and requires settings equal to the pinned
  approved digest; G0b reports K_match's covariate balance; H0 gates each gapped run's gap pre-registration and
  each seed pair's identity; G2 and G3, like G1, wait for G1-pre (stop rule 3); every host's jitter scale is the
  data SD over the native rows.

### Adversarial review of the action-gap code (2026-10-12; synthetic only, not findings)

Fixed, with regression tests in `test_gap.py` (`GapReviewTest`):
- **Device (major).** `gap.projection` imported `runtime.td3_bc`, and with it JAX, before `train.main` set
  `JAX_PLATFORMS` from `--device`. With the `export JAX_PLATFORMS=cpu` above, every `--device cuda` gapped host
  would have trained on the CPU, slowly and while holding the GPU lock. Its `source.json` would still have said
  cuda, and its full-data control (plain train.py) would have trained on the GPU. `gap_train` now picks the platform
  before JAX loads, checks the backend before training and records it.
- **Pre-registration order (major).** Nothing stopped a gapped host from training, and printing returns, before
  the marks were frozen. `preregister` only looked for `W/host_*` analysis folders. `gap_train` now requires the
  frozen gap pre-registration and records its sha256 in `resolved.json`, and H0 checks it.
- **Seed-pair identity (major).** H0 never compared a gapped host with its full-data control beyond discount and
  noise. A control trained with another config or code would have passed, and the difference would have been
  credited to the gap; 172/173 are shared with option C and may be older. H0 now gates on equal resolved rows
  (except the data pin and run id) and equal training code (`PAIR_CODE` in `source.json`).
- **Minor fixes.**
  - `pass_marks_equal_design` compared against the live constants; it now uses pinned digests.
  - The verdict re-verifies `g0.json` and `h0.json`, and `verdict.md` prints a POST HOC line.
  - G2 and G3 ran without G1-pre.
  - G3 now checks that the two runs of a pair used the same reset bank.
  - Jitter scales differed by arm.
  - H0 read the restore's reservation record instead of the run's own.
  - `gap_train` did not re-check `g0.json`, and its output lacked the SYNTHETIC mark.
  - H0 could not be re-verified after a failure.

Checked and not defects (evidence in the review record):
- No gap row reaches training, the bank or the normalization fit: the conversion, the projected reservation,
  `validate_prepared`'s normalization identity and H0 checks.
- Every critic is queried in its own normalization (`CriticProbe` and the policy standardize with their host's
  statistics).
- e_A uses each host's own policy and labels.
- The in-gap rule at evaluation is the frozen manifest's c* and rho, the same ones that removed the training rows.
- Returns are paired by seed and reset state.
- The `$(...)` captures in the commands receive only the hash; imports print nothing to stdout.

## Deviations from the design (design_d4rl_rt.json)

- Evaluation runs through `pools.policy_episodes` on bare MjSims (the same `Physics.step` the labeller uses, equal
  to `env.step` bit for bit) instead of `runtime/common.eval_policy`'s vector environment. The dynamics, policy and
  time limit are the same, and every arm starts from the same reset bank.
- Default slots 256, not 512 (C1's measurement: the policy already costs about 1.5 us per env step against about
  100 us of physics).
- "2 rollouts each" for R3b's on-policy states is read as one deterministic label plus one jitter repeat (1e-4 of the
  data SD), both under the P0(0.25) arm's own policy.
- R3a's "-3 jitter SEs" uses sigma_J = the median single-rollout jitter SD over the roughness pairs. Each gain uses
  single rollouts.
- R3b's thin share of the summed overestimate uses the positive part, (Q1 - Q^pi)+.
- R4a: the raw first-order gain decides the pass. The Adam-preconditioned gain (the host checkpoint's Adam second
  moment) is reported beside it.
- R4b's 5-point beta ladder is centred on each arm's Level-1 matched beta from R4a (x 0.25, 0.5, 1, 2, 4). The fallback
  ladder is 0.1 to 10.
- R4b's "P0 at its best strength" includes m = 1 (NONE). Without it, a world where NONE beats every other P0 arm and
  every OR-L-abs arm could pass the best-strength margin (review fix; `test_review.py`).
- Pool-T rows whose rollout hit a simulator error (no truth) leave the oracle pool; R4a and the extraction use the
  same kept rows and record how many were dropped.
- A per-critic requirement passes only when it passes on every planned critic. The design does not say how critics
  combine.
- R5-world (a) divides the mean within-row variance of 32 target draws by the variance over rows of one draw's
  residual t_0 - Q1. (c) uses L's roughness pairs and Var(e) over L's pi and logged-action points.
- The smoothed-truth rule (jitter variance above 5% of Var(e_A)) is reported as a flag. Only the roughness pairs have
  jitter rollouts, so the single rollout stays the truth elsewhere.
- Pool Pi keeps every recorded state when 100 episodes give fewer than 3,000 states.

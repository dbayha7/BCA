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
| `test_*.py` | CPU tests on synthetic data (see Tests). `test_review.py` pins the defects found in the adversarial review. |

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
  experiments.truth.test_pools experiments.truth.test_acceptance experiments.truth.test_review
```

53 tests, 97 s here (4 shared CPUs; 44 from the build, 9 in test_review.py). The end-to-end tests write
synthetic files and random-init host runs, then run Stage 0,
freeze, and every planned stage. Their outputs say SYNTHETIC and are test fixtures, not findings.

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

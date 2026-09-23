# Bayesian Conformal Aggregation

Four JAX hosts. Four BCA extensions. Two methods: **`host`** and **`bca`**.

The BCA files import the corresponding host. They reuse its networks, initialization,
optimizers and update equations. Both methods use the same host hyperparameters.

Read the **[complete pseudocode for all four hosts + BCA](ALGORITHMS.md)**:
host losses, fitting weights, ESS abstention, Bayesian/conformal radius refresh,
frozen-width consumption, update order, and evaluation/checkpoint schedules.

| JAX host | Host + BCA | What BCA changes |
| --- | --- | --- |
| [IQL](algorithms/iql.py) | [IQL + BCA](algorithms/iql_bca.py) | Actor advantage weights; shared Q/V updates stay the same. |
| [CQL](algorithms/cql.py) | [CQL + BCA](algorithms/cql_bca.py) | Per-sample conservative critic gap; the dual update uses the original unweighted gap. |
| [TD3+BC](algorithms/td3_bc.py) | [TD3+BC + BCA](algorithms/td3_bc_bca.py) | Actor behavior-cloning multiplier. |
| [ReBRAC](algorithms/rebrac.py) | [ReBRAC + BCA](algorithms/rebrac_bca.py) | Actor behavior-cloning multiplier; critic BC stays unchanged. |

## Layout

```text
algorithms/       # Four plain hosts and four *_bca.py extensions
calibration/      # Shared scale fitting, importance weights and radius math
runtime/          # Data preparation, evaluation, checkpoints and validation
configs/          # experiment.yaml plus one YAML per algorithm
train.py          # Run one selected method
check.py          # Check all declared configurations
ALGORITHMS.md     # Complete, source-linked pseudocode for every host + BCA
requirements.txt
LICENSE
NOTICE
```

Start with a host file, then its `_bca.py` counterpart. `calibration/` and `runtime/`
contain supporting math and execution code.

## Configuration

Each algorithm YAML has one `host` block, one `bca` block and dataset overrides.
Both methods read **exactly the same host hyperparameters**, training reservation,
normalization and evaluation banks. A host baseline uses the training complement
of the reserved calibration data; it is not a full-data published baseline.
Dataset overrides merge into the corresponding block. `--print-config` shows all
resolved values, seeds and schedules before execution.

`experiment.yaml` declares five seeds and **1,000,000 host updates**. TD3+BC and
ReBRAC perform 500,000 delayed actor updates. BCA retains its 10k warmup and 198
posterior refreshes. Evaluation has 200 periodic banks and a separate 20-episode
final bank; periodic banks contain ten episodes, or two per actor for IQL.

IQL `bca` explicitly trains **two actors: `host` and `bca`**, with one shared Q/V
state and one AWR-weighted calibrator. Their beta is the native value and decision
gain is fixed at 1. The standalone `host` option trains only the host actor on the
same pool. There are no beta sweeps, floor-only actors, permutation arms or fixed-
strength control arms in the active interface.

This is a new minimal comparison design. The previous exact reproduction matrix,
including its IQL beta choices and eight-actor groups, remains available at
[`reproduction-matrix-v1`](https://github.com/dbayha7/BCA/tree/reproduction-matrix-v1).
Existing results cannot be relabeled as results of this new design.

## Run

Use Python 3.10 on Linux/WSL2, with MuJoCo 2.1 installed in
`~/.mujoco/mujoco210` and the original D4RL HDF5 files in `~/.d4rl/datasets`.
The pinned cache hashes are checked before training; missing data stops the run.

```bash
pip install -r requirements.txt
# NVIDIA GPU support for this pinned JAX version:
pip install 'jax[cuda12]==0.6.2'

python check.py
python check.py --runtime  # CPU protocol validation; no training or simulator steps
python train.py --algorithm td3_bc --list
python train.py --algorithm td3_bc --dataset hopper --method bca \
  --seed 202609171 --print-config

python train.py --algorithm td3_bc --dataset hopper --method host \
  --seed 202609171 --output runs/td3-hopper-host
python train.py --algorithm td3_bc --dataset hopper --method bca \
  --seed 202609171 --output runs/td3-hopper-bca
```

Use `--prepare-only` with a new output directory to verify a real cached split
without learner updates. `--data-dir`, `--device cpu` and `--lock` are optional.
Every output directory must be new. Each run records its resolved config, source
snapshot, data identity, metrics, evaluations, checkpoints and completion/failure.
There is no automatic retry or queue. Checkpoint counters and event banks are
validated before completion is recorded.

GPU runs hold an exclusive lock. On the existing research machine set
`BCA_GPU_LOCK=/home/dbayha/bca-work/resource-locks/local-rtx5070ti.lock` so these
runs share the existing GPU reservation. Do not edit a checkout while it trains.

## Calibration and provenance

BCA retains both Bayesian and conformal radii and consumes their maximum.
TD3+BC/ReBRAC use action-affinity fitting weights, CQL uses policy-density fitting
weights, and IQL uses capped pre-BCA AWR fitting weights. These weights affect
scale fitting; posterior residual-score calibration remains unweighted. A fitted
scale or conformal floor does not by itself guarantee behavioral-harm detection
or improved return under adaptive training.

`configs/sources.json` records source identity and verification limits. The code
is derived from the JAX CORL hosts used in this research, with attribution to
[CORL](https://github.com/corl-team/CORL) and
[Unifloral](https://github.com/EmptyJackson/Unifloral). These host versions must
remain distinct from other native candidates and published reference means.
See `LICENSE` and `NOTICE`. Historical outputs and checkpoint weights stay outside
this repository.

Validation of this layout covers all 280 resolved run declarations, all 28 cached
dataset preparations and paired training pools, and CPU numerical parity fixtures
for all eight host/BCA paths. IQL's paired execution/checkpoint boundary also passes
with generated data and mock evaluations. These checks do not constitute completed
1M runs, fresh-environment installation validation, or GPU reproducibility results.

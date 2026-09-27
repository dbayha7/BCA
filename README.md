# Bayesian Conformal Aggregation

Four JAX hosts. Four BCA extensions. Two methods: **`host`** and **`bca`**.

These are the CORL-derived hosts used by BCA. The **complete original Unifloral
repository** is included in [baselines/unifloral/](baselines/unifloral/), with its
algorithms, configs and environment files preserved exactly at a pinned revision.

The BCA files import the corresponding host. They reuse its networks, initialization,
optimizers and update equations. Both methods use the same host hyperparameters.

Read the **[complete pseudocode for all four hosts + BCA](ALGORITHMS.md)**:
host losses, fitting weights, ESS abstention, Bayesian/conformal radius refresh,
frozen-width consumption, update order, and evaluation/checkpoint schedules.

Read **[every BCA entry point, with exact code snippets](INTEGRATION.md)** to see
where BCA reads the host, changes its losses, adds state and gates updates.

Read the **[OOD experiment design](docs/superpowers/plans/2026-09-24-ood-experiment.md)**
for separate tests of support novelty, residual coverage and behavioral harm,
including controls, sample budgets and a checked implementation sequence.

| JAX host | Host + BCA | What BCA changes |
| --- | --- | --- |
| [IQL](algorithms/iql.py) | [IQL + BCA](algorithms/iql_bca.py) | Actor advantage weights; shared Q/V updates stay the same. |
| [CQL](algorithms/cql.py) | [CQL + BCA](algorithms/cql_bca.py) | Per-sample conservative critic gap; the dual update uses the original unweighted gap. |
| [TD3+BC](algorithms/td3_bc.py) | [TD3+BC + BCA](algorithms/td3_bc_bca.py) | Actor behavior-cloning multiplier. |
| [ReBRAC](algorithms/rebrac.py) | [ReBRAC + BCA](algorithms/rebrac_bca.py) | Actor behavior-cloning multiplier; critic BC stays unchanged. |

## What owns what, and why

| Component | Responsibility | Why it is separate |
| --- | --- | --- |
| `algorithms/<host>.py` | Networks, initialization, optimizers and the host's original update equations | You can read the learning rule without reading calibration. Plain hosts import no BCA code. |
| `algorithms/<host>_bca.py` | Connect that host's residuals to calibration and insert the resulting detached adjustment into one loss term | The exact intervention is visible; the host equations are reused. |
| `calibration/` | Fitting weights, positive residual scale, Bayesian/conformal radii and width-to-weight formulas | These are distinct mathematical operations, shared where their definitions agree. |
| `runtime/` | Resolve settings, prepare data, execute schedules, evaluate and save/verify state; shared initialization/data types live in `networks.py` | Execution bookkeeping should not obscure the host's update. |
| `configs/` | One experiment schedule; one host/BCA configuration per algorithm with dataset overrides | Each setting has one owner; the resolved config is saved with every run. |

Read **host → host + BCA → shared calibration**. Start with
[TD3+BC](algorithms/td3_bc.py) and [its BCA extension](algorithms/td3_bc_bca.py),
then use the [component-level pseudocode map](ALGORITHMS.md#the-calibration-pipeline).

The BCA pipeline has three separate operations:

1. **Fit a scale** to detached Bellman residuals, with host-specific importance
   weights and an ESS gate. Only the scale network is optimized here.
2. **Freeze a reference** using held-out residual scores. Retain both the Bayesian
   and conformal radii and take their maximum.
3. **Use the frozen width** in the host objective. It strengthens actor BC in
   TD3+BC/ReBRAC, weights CQL's conservative gap, or shrinks IQL's excess AWR
   weight above one. The host cannot backpropagate through calibration.

Fitting weights, bootstrap masses and the final host multiplier are different
quantities. The pseudocode names each separately. The current radius stage uses
**unweighted held-out scores**; fitting importance weights do not turn it into a
weighted-conformal covariate-shift theorem.

## Layout

```text
algorithms/       # Four plain hosts and four *_bca.py extensions
calibration/      # Shared scale fitting, importance weights and radius math
runtime/          # Data preparation, evaluation, checkpoints and validation
configs/          # experiment.yaml, four algorithm YAMLs, source/reference manifests
baselines/        # Complete pinned original Unifloral tree and reference notes
docs/             # Proposed OOD experiment and implementation plan
train.py          # Run one selected method
check.py          # Check all declared configurations
ALGORITHMS.md     # Complete, source-linked pseudocode for every host + BCA
INTEGRATION.md    # Every BCA entry point, with source-checked code excerpts
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

## Unifloral baseline references

[baselines/unifloral/](baselines/unifloral/) contains all **40 original files** at
Unifloral commit `f2dc1278eae18ed3e22c92255119c54885414d17`: standalone and unified
algorithms, both config directories, README, license, requirements, Dockerfile and
evaluation sources. The 256 kB snapshot is included directly in a normal clone.
There is no submodule, nested Git checkout or additional source-fetch step.

[One reference manifest](configs/unifloral.json) records the upstream tree and
every Git blob/SHA256 identity. It also records literal `Args` defaults for the
four standalone reference hosts, three locomotion datasets and seed 0. Other
upstream algorithms/configs are preserved, not added to the BCA experiment matrix.
The root four host/BCA pairs remain the only active learning interface.

From this BCA repository:

```bash
python check.py  # Includes all bundled Unifloral file checks
python check.py --reference-config cql hopper
```

The check reads hashes, Git blob identities and literal configuration values
without importing a training script. It works without an upstream Git checkout.
An optional `--unifloral /path/to/checkout` also verifies an external copy and its
Git revision when present. `--reference-config` prints the full original
configuration, data hash, evaluation mode, update counts and launch command.
Reference algorithm names are `iql`, `cql`, `td3_bc`, `rebrac`; dataset names are
`hopper`, `walker`, `halfcheetah`. The active paired config uses `walker2d`.

Use Unifloral's preserved requirements/Dockerfile in a **separate environment**.
After provisioning it, the upstream native-default command is, for example:

```bash
cd baselines/unifloral
python algorithms/cql.py --dataset hopper-medium-v2 --seed 0 --num-updates 1000000
```

This command starts training and follows the original upstream output behavior;
the BCA runner's output guards, GPU lock and checkpoints do not wrap it. Reference
inspection does not launch it. On a shared machine, reserve the GPU through the
same external resource lock before an authorized execution.

| Distinction | Original Unifloral default reference | Active BCA host/control design |
| --- | --- | --- |
| Source | Exact pinned standalone files | CORL-derived JAX ports with explicit BCA attachment |
| Data pool | Full converted native dataset | Same reserved training complement for both methods |
| TD3+BC/ReBRAC budget | 1M outer steps = **2M critic / 1M actor** updates | 1M critic / 500k actor updates |
| CQL | Ten critics, raw observations, sampled tanh-Gaussian evaluation | Twin critics, configured preprocessing, deterministic tanh-mean evaluation |
| Evaluations | 400 periodic banks of 8 episodes; 1,000 final episodes | 200 periodic banks of 10 episodes (IQL: 2); 20 final episodes |
| Evaluation randomness | Split from the evolving upstream training RNG | Separate declared evaluation streams |

The original YAMLs define **sweeps**, not uniquely identified table-winning
configurations. The reference manifest chooses the unchanged Python defaults;
it does not claim to reproduce the published Table 1 means. The closed native
acquisitions also used an observer, synchronous vector evaluation and a recorded
runtime correction. Their saved results retain that provenance; a direct
upstream invocation is not a reconstruction of their instrumentation.

## Reproduction status

The OOD implementation has one fully commented [configuration](configs/ood.yaml),
one [protocol module](experiments/ood/protocol.py), and tested read-only adapters
for the initial tranche. It declares
TD3+BC/ReBRAC × Hopper/Walker2d × the five existing seeds: **20 paired comparisons,
40 required 1M checkpoints**. It reuses the existing host/BCA settings without
changing training code or adding model variants.

```bash
python -m unittest experiments.ood.test_protocol
python -m experiments.ood.protocol --output runs/ood/declaration-v1.json
```

These commands need only Python and the existing PyYAML dependency. They execute
no training, model forward passes or simulator transitions. The declaration pins
source/config identities, all intended rows, distinct random-stream seeds, the
statistical contract and resource ceilings. An existing output file is never
overwritten. Regenerate to a new path after changing the repository revision;
validation rejects stale or edited declarations. The maximum future budget is
39,998,400 environment transitions including engineering checks, not a launch.

All checkpoint rows start **pending**, meaning no evidence has been bound; this
does not assert that compatible artifacts are absent from other locations.
`validate_evidence` checks explicitly supplied file hashes and reported metadata.
Its seven roles are `resolved`, `source`, `preparation`, `result`, `checkpoint`,
`events` and `actual_exit`, each with a path and SHA256. The separate supervisor
receipt uses schema `ood-process-exit-v1`, run ID, actual integer exit code, and
the result-file hash; the learner's own `exit.json` cannot substitute for it.
Even accepted metadata stays pending: the actual 1M checkpoints and journals,
paired data/normalization, and collection-specific engineering gates must still
be verified. Synthetic checkpoint tests now pass for TD3+BC/ReBRAC, with and
without BCA. Hopper/Walker adapters pass exact restored-state and repeated-step
checks, including independent reward reconstruction within absolute `1e-7`.
Every YAML setting has an explanation; the scientific values are unchanged.
ReBRAC's fresh live residual-coverage target also
remains unresolved because its training target reads a recorded next action.

See the [adapter test record](docs/OOD_ADAPTER_TESTS.md) for the fixed CPU smoke
budget, preserved failures, runtime requirements and exact reproduction command.
The tests use untrained synthetic checkpoints and a few real simulator steps;
they are engineering evidence, not OOD results or accepted 1M runs. No scientific
OOD collection has launched. Task C's calculation and static-figure tests now
pass: see the [calculation test record](docs/OOD_CALCULATION_TESTS.md),
[synthetic reader](docs/ood-demo/index.html), and
[all ten example figures](docs/ood-demo/all-figures.pdf). Browser/mobile visual
review remains unverified because the browser tool blocks local-file pages.
Next are the remaining reader review, actual checkpoint binding and Task D's
collection-specific engineering gates. All demonstration data are synthetic.

Validation of this layout covers all 280 resolved run declarations, all 28 cached
dataset preparations and paired training pools, and CPU numerical parity fixtures
for all eight host/BCA paths. IQL's paired execution/checkpoint boundary also passes
with generated data and mock evaluations. These checks do not constitute completed
1M runs, fresh-environment installation validation, or GPU reproducibility results.

The Unifloral check verifies all 40 upstream files and four default argument sets;
the eight algorithm/config files also match the archived acquisition bundle
byte for byte. Upstream dependency ranges and its Docker base tag are not a
complete binary environment lock. A fresh Unifloral build remains unverified.

For a new BCA run, `resolved.json` identifies the experiment, `source/` contains
the code/config/docs snapshot, and `source.json` records Python, installed package
versions and execution flags. Data preparation records the cache, split and
normalization identities; checkpoints retain optimizer/RNG state; result and
exit records verify the scheduled budget and evaluations. Preserve the actual
process exit as well. These records make mismatches inspectable; bitwise replay
across different hardware, drivers or compiler versions is not established.

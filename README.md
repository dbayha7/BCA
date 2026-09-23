# Bayesian Conformal Aggregation

The BCA implementations and explicit configurations for IQL, CQL, TD3+BC, and
ReBRAC. This repository contains code and reproduction settings. Experiment
outputs and historical research archives live separately.

## Structure

```text
algorithms/
  iql.py                 # Actors and scale variants sharing nuisance Q/V
  cql.py                 # BCA on the conservative critic gap
  td3_bc.py              # BCA on the actor behavior-cloning term
  rebrac.py              # BCA on actor BC; native critic BC retained
  _*.py                  # Native hosts, calibration, data and execution helpers
configs/
  experiment.yaml        # Seeds, update budget and evaluation/refresh schedules
  iql.yaml               # IQL defaults, controls and seven dataset overrides
  cql.yaml
  td3_bc.yaml
  rebrac.yaml
  sources.json           # Source provenance and configuration matrix identity
  corl_reference_scores.*
train.py                 # One explicit run
check.py                 # Configuration and protocol validation
requirements.txt
LICENSE
NOTICE
```

Start with the algorithm file and its matching YAML file. Files beginning with
`_native_` contain the underlying host equations. The `_bca_`, `_iql_` and other
private helpers implement shared math or host-specific calibration and execution.

## Configurations

There is **one configuration per algorithm**, containing all seven datasets.
The resolution order is:

1. Algorithm `defaults`.
2. The selected dataset's `defaults`.
3. The selected algorithm `arm`.
4. That dataset's override for the arm, then any explicit seed override.

An arm marked `bca: true` takes the algorithm's `bca_defaults` before its arm
settings, and the dataset's `bca_defaults` before that dataset's arm settings.
This keeps shared calibration parameters in one place without applying them
to native controls. There is no recursive chain of configuration files.

Only arms listed under a dataset are enabled there. An empty `{}` means the
dataset inherits the named arm unchanged. Lists replace lists; mappings merge
recursively. No additional inheritance framework is required.

`experiment.yaml` defines common values referenced as `{experiment.num_updates}`,
etc. Evaluation schedules use explicit arithmetic progressions instead of long
lists of episode seeds. `--print-config` expands every reference and schedule
so the complete effective configuration can be inspected before a run.
Distinct reset and refresh banks keep their explicit identities.

```bash
python train.py --config configs/td3_bc.yaml --list
python train.py --config configs/td3_bc.yaml --dataset hopper \
  --arm posterior_affinity --seed 202609171 --print-config
```

The recorded matrix contains four algorithms, seven datasets, five seeds,
810 physical runs/groups and 1,055 actor trajectories. The number of controls
differs by algorithm and dataset. IQL `shared` runs contain eight actors with
shared nuisance Q/V and five scale variants; `native` is a separate full-data
control. These declarations are **not a claim that the matrix has completed**.

Each run starts from zero and targets 1,000,000 host updates. TD3+BC/ReBRAC use
500,000 delayed actor updates; CQL uses 1,000,000 actor updates. IQL records
accepted actor and scale updates separately from attempted updates. BCA arms
retain Bayesian and conformal components and their declared warmup/refresh
rules. TD3+BC/ReBRAC/CQL have 200 ten-episode periodic banks and a separate
20-episode final bank. IQL has 200 two-episode periodic banks per actor and a
separate 20-episode final bank per actor.

The controls remain distinct: fitting-weight permutation versus consumed-dose
permutation; constant 1.5 from initialization versus a development-derived fixed
strength after warmup; original versus matched scale objectives. The YAML files
retain their actual settings. Reward transforms, reservation sizes, normalization,
ESS gates and evaluation semantics remain host/dataset-specific.

## Environment

Use **Python 3.10 on Linux or WSL2**. These versions preserve the existing D4RL /
Gym / MuJoCo setup; this is not a Gymnasium or new-MuJoCo port.

```bash
python3.10 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
# For NVIDIA training, install the matching CUDA-enabled JAX build:
pip install 'jax[cuda12]==0.6.2'
```

Install MuJoCo **2.1** in `~/.mujoco/mujoco210` (or set
`MUJOCO_PY_MUJOCO_PATH`) and the Linux build/OpenGL libraries required by
[mujoco-py](https://github.com/openai/mujoco-py#install-mujoco).
On Ubuntu these typically include `build-essential`, `patchelf`,
`libosmesa6-dev`, `libgl1-mesa-dev`, and `libglfw3`.
The runner adds the MuJoCo library directory before imports. When exactly one
compatible prebuilt CPU extension is installed, it loads that binary without
recompiling it. Otherwise mujoco-py performs its normal build.

Place the original D4RL HDF5 datasets in `~/.d4rl/datasets`, or pass
`--data-dir`. Every config pins the filename and SHA256. A missing or mismatching
file stops preparation; training does not silently download or substitute data.
Obtain datasets using the original [D4RL dataset instructions](https://github.com/Farama-Foundation/D4RL).

## Validate and run

```bash
# Syntax and all 810 resolved configurations; PyYAML is sufficient.
python check.py
# All typed host protocols, on CPU; no learner or simulator steps.
python check.py --runtime
# Inspect a real cached split before creating any learner.
python train.py --config configs/iql.yaml --dataset pen-human \
  --arm shared --seed 202609171 --prepare-only --output runs/iql-pen-preparation
# Run one explicitly selected experiment into a new directory.
python train.py --config configs/td3_bc.yaml --dataset hopper \
  --arm posterior_affinity --seed 202609171 --output runs/td3-hopper-affinity-s1
```

`check.py` also checks the resolved matrix against the recorded reproduction
identity. Deliberate experiment changes require a reviewed new identity; editing
a parameter is not automatically the same reproduction.

GPU runs use an exclusive file lock. Set `BCA_GPU_LOCK` or `--lock` to the same
path used by other workers on the machine. For the existing research machine,
that path is `/home/dbayha/bca-work/resource-locks/local-rtx5070ti.lock`.
Use `--device cpu` for CPU execution. There is no automatic queue or retry.

Each new output directory records the resolved configuration, code/config
snapshot, dependency versions, data preparation, journals, checkpoints and
completion or failure evidence. TD3+BC/ReBRAC/CQL save detached checkpoints at
10k, 50k and the final update; IQL retains its reference and training-end
checkpoints. Existing output directories are rejected. Do not edit a checkout
while one of its runs is active.

## Interpretation and provenance

The reorganization preserves the declared scientific settings; it does not
establish new calibration guarantees or reproduce completed 1M results by
itself. An accepted scale update is not evidence of improved return or reliable
harm ranking. Bayesian/conformal residual widths and behavioral risk remain
different quantities. Published reference means, local controls and native
candidates must be reported separately.

Exact numerical replay also depends on the data, software stack, hardware and
evaluation banks. `configs/sources.json` records the migration checks and their
limits. Historical results, failures and checkpoint weights are not imported
into this clean repository.

Derived from [CORL](https://github.com/corl-team/CORL) and
[Unifloral](https://github.com/EmptyJackson/Unifloral). See `LICENSE` and `NOTICE`
for licensing and attribution.

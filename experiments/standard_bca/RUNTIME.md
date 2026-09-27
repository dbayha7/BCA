# Reproduce the accepted cluster runtime

Use Linux x86-64 and Python 3.10.20. Create a clean virtual environment without
`--system-site-packages`; inherited packages caused the preserved earlier
cluster preflight failure. Install `cluster-requirements.lock` into that environment.
The lock includes JAX/CUDA packages and the exact MJRL commit required by Pen.
The standard root requirements remain the portable scientific package pins.

Install MuJoCo 2.1 at `~/.mujoco/mujoco210`. Install the separate graphics prefix
using `conda create --prefix "$PWD/graphics" --file graphics-explicit.lock`.
The graphics lock includes Mesa 24.1.0 and its exact dependency builds.

Set these paths for compilation and execution (replace the example prefixes):

```bash
export PATH="$PWD/venv/bin:$PWD/graphics/bin:$PATH"
export CPATH="$PWD/graphics/include"
export LIBRARY_PATH="$PWD/graphics/lib"
export LD_LIBRARY_PATH="$PWD/graphics/lib:$HOME/.mujoco/mujoco210/bin:${LD_LIBRARY_PATH:-}"
export PYTHONNOUSERSITE=1
unset PYTHONPATH
export MUJOCO_PY_FORCE_CPU=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export D4RL_SUPPRESS_IMPORT_ERROR=1 WANDB_MODE=disabled
python -m pip check
python check.py --runtime
```

Run GPU checks on an allocated compute node. Training additionally sets
`JAX_PLATFORMS=cuda`; CPU preparation sets `JAX_PLATFORMS=cpu`. Rendering is not
used for rewards. The accepted simulator smoke test took seven explicit test
steps, one for each dataset environment; these are engineering checks, not OOD
outcomes or training episodes. The run's `source.json` records actual installed
packages and execution variables. GPU/driver identity is recorded separately.

All Gym, D4RL and MJRL Python sources matched the existing local environment
byte for byte. Different GPU models still need their own recorded provenance;
CPU numerical equality is not a claim of bitwise equality across GPUs.

The first three cluster preflights remain failed attempts. The fourth passed
all phases. No old environment was upgraded, no training run was retried, and
no scientific source or tolerance was changed to obtain that pass.

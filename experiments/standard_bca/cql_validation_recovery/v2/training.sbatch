#!/bin/bash
#SBATCH --job-name=bca-cql-td3-recovery
#SBATCH --partition=GPU
#SBATCH --account=staruser
#SBATCH --gres=gpu:A100:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=64G
#SBATCH --time=14-00:00:00
#SBATCH --output=/users/dbayha/bca-standard-noiw-v1/cql-validation-recovery-v2/training-slurm-%j.log
set -u
export PATH=/users/dbayha/bca-standard-noiw-v1/venv-v2/bin:/users/dbayha/bca-standard-noiw-v1/graphics/bin:$PATH
export LD_LIBRARY_PATH=/users/dbayha/bca-standard-noiw-v1/graphics/lib:/users/dbayha/.mujoco/mujoco210/bin:${LD_LIBRARY_PATH:-}
export CPATH=/users/dbayha/bca-standard-noiw-v1/graphics/include
export LIBRARY_PATH=/users/dbayha/bca-standard-noiw-v1/graphics/lib
export PYTHONNOUSERSITE=1
unset PYTHONPATH
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export MUJOCO_PY_FORCE_CPU=1 D4RL_SUPPRESS_IMPORT_ERROR=1 WANDB_MODE=disabled
/users/dbayha/bca-standard-noiw-v1/venv-v2/bin/python /users/dbayha/bca-standard-noiw-v1/cql-validation-recovery-v2/supervise.py
code=$?
printf '{"actual_exit_code": %d, "slurm_job_id": "%s"}\n' "$code" "$SLURM_JOB_ID" > /users/dbayha/bca-standard-noiw-v1/cql-validation-recovery-v2/training_slurm_actual_exit.json
exit "$code"

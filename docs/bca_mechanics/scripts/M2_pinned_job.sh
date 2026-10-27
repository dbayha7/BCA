# The WSL bridge job that runs M2_pinned.py in the pinned environment (see the header of M2_pinned.py).
# From Windows:
#   pwsh -NoProfile -File <scratchpad>/wslrun.ps1 -ScriptFile <repo>/docs/bca_mechanics/scripts/M2_pinned_job.sh -TimeoutSec 1800 -Name m2pinned
# The job runs as root with the working directory /home/user/bca (the owner's clone, read-only); it writes only under
# /tmp/bridge/work/m2. The Python script is copied out of the Windows worktree first (M2_SCRIPT overrides) so that
# Python's script directory is a plain Linux folder.
set -e
unset JAX_ENABLE_X64 XLA_FLAGS JAX_DEFAULT_MATMUL_PRECISION
export JAX_PLATFORMS=cpu OMP_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
SCRIPT="${M2_SCRIPT:-/mnt/c/Users/DAVIDB~1/AppData/Local/Temp/claude/C--Users-David-Bayha-BCA/9ab1f036-9335-42af-93a3-a9c39c11b8cc/scratchpad/mech/docs/bca_mechanics/scripts/M2_pinned.py}"
OUT="${M2_OUT:-/tmp/bridge/work/m2}"
mkdir -p "$OUT"
cp "$SCRIPT" "$OUT/M2_pinned.py"
cd /home/user/bca
/tmp/claude-0/bcavenv/bin/python -c "import sys, numpy, jax; print(sys.version.split()[0], numpy.__version__, jax.__version__)"
git log --oneline -1
/tmp/claude-0/bcavenv/bin/python "$OUT/M2_pinned.py" "$OUT" ${M2_ARGS}

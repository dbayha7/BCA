# The WSL bridge job that runs M1_real.py in the pinned environment (see the header of M1_real.py).
# From Windows:
#   pwsh -NoProfile -File <scratchpad>/wslrun.ps1 -ScriptFile <repo>/docs/bca_mechanics/scripts/M1_real.sh -TimeoutSec 3000 -Name m1real
# The job runs as root with the working directory /home/user/bca (the owner's clone, read-only); it writes only under
# /tmp/bridge/work/m1. The Python script is read from the Windows worktree through /mnt/c (M1_SCRIPT overrides).
set -e
unset JAX_ENABLE_X64 XLA_FLAGS JAX_DEFAULT_MATMUL_PRECISION
export JAX_PLATFORMS=cpu OMP_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
SCRIPT="${M1_SCRIPT:-/mnt/c/Users/DAVIDB~1/AppData/Local/Temp/claude/C--Users-David-Bayha-BCA/9ab1f036-9335-42af-93a3-a9c39c11b8cc/scratchpad/mech/docs/bca_mechanics/scripts/M1_real.py}"
OUT="${M1_OUT:-/tmp/bridge/work/m1}"
mkdir -p "$OUT"
cp "$SCRIPT" "$OUT/M1_real.py"
cd /home/user/bca
/tmp/claude-0/bcavenv/bin/python "$OUT/M1_real.py" "$OUT" ${M1_ARGS}

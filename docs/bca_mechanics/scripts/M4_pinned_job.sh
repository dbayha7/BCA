# The bridge job that runs M4_pinned.py in the pinned environment (WSL Ubuntu-24.04, /home/user/bca, as root):
#   pwsh -NoProfile -File <scratchpad>\wslrun.ps1 -ScriptFile <this file> -TimeoutSec 1500 -Name m4pinned
# The gate's software check (experiments/shift/acceptance.check_software) accepts exactly these flags: JAX_PLATFORMS=cpu
# and OMP_NUM_THREADS=1 set, JAX_ENABLE_X64, XLA_FLAGS and JAX_DEFAULT_MATMUL_PRECISION unset (runs/shift/preregistration
# .json, software.flags). The script is copied out of the Windows worktree first so that Python's script directory is a
# plain Linux folder. Output: /tmp/bridge/work/m4/M4_real_data.json, then committed as docs/bca_mechanics/data/M4_real_data.json.
set -e
unset JAX_ENABLE_X64 XLA_FLAGS JAX_DEFAULT_MATMUL_PRECISION
export JAX_PLATFORMS=cpu OMP_NUM_THREADS=1 PYTHONPATH=/home/user/bca PYTHONDONTWRITEBYTECODE=1
SRC="${M4_SCRIPT:-/mnt/c/Users/David Bayha/AppData/Local/Temp/claude/C--Users-David-Bayha-BCA/9ab1f036-9335-42af-93a3-a9c39c11b8cc/scratchpad/mech/docs/bca_mechanics/scripts/M4_pinned.py}"
OUT=/tmp/bridge/work/m4
mkdir -p "$OUT"
cp "$SRC" "$OUT/M4_pinned.py"
cd /home/user/bca
/tmp/claude-0/bcavenv/bin/python -c "import sys, numpy, jax, scipy; print(sys.version.split()[0], numpy.__version__, jax.__version__, scipy.__version__)"
git log --oneline -1
/tmp/claude-0/bcavenv/bin/python "$OUT/M4_pinned.py" --out "$OUT/M4_real_data.json"

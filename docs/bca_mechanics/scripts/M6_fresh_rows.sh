# The WSL bridge job that runs M6_fresh_rows.py in the project's pinned environment (Python 3.10.20, numpy 2.2.6,
# jax 0.6.2 on CPU) inside the owner's clone /home/user/bca, read-only. From Windows:
#   pwsh -NoProfile -File <scratchpad>\wslrun.ps1 -ScriptFile M:\docs\bca_mechanics\scripts\M6_fresh_rows.sh -TimeoutSec 1200 -Name m6all
# The job ran on 20 October 2026 (bridge job "m6all", 16:12:48 to 16:15:55 UTC, exit 0; 6 to 16 s per seed-world) and
# wrote /tmp/bridge/work/m6/M6_fresh_rows_coverage.json, copied to docs/bca_mechanics/data/M6_fresh_rows_coverage.json.
# M6_fresh_rows.py is read from the Windows worktree through /mnt/c (M6_SCRIPT overrides); output stays under
# /tmp/bridge/work/m6. Nothing under runs/ is written.
set -e
unset JAX_ENABLE_X64 XLA_FLAGS JAX_DEFAULT_MATMUL_PRECISION
export PYTHONDONTWRITEBYTECODE=1 JAX_PLATFORMS=cpu OMP_NUM_THREADS=1 PYTHONPATH=/home/user/bca
SCRIPT="${M6_SCRIPT:-/mnt/c/Users/DAVIDB~1/AppData/Local/Temp/claude/C--Users-David-Bayha-BCA/9ab1f036-9335-42af-93a3-a9c39c11b8cc/scratchpad/mech/docs/bca_mechanics/scripts/M6_fresh_rows.py}"
OUT="${M6_OUT:-/tmp/bridge/work/m6}"
mkdir -p "$OUT"
cp "$SCRIPT" "$OUT/M6_fresh_rows.py"
cd /home/user/bca
date -u
/tmp/claude-0/bcavenv/bin/python -u "$OUT/M6_fresh_rows.py" --settings near80,random_rows --worlds stall,no_stall \
    --seeds 0,1,2,3,4 --out "$OUT/M6_fresh_rows_coverage.json"
echo "exit $?"
chmod -R a+rwX /tmp/bridge/work
date -u

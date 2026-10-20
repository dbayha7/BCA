# Module M3 companion job: the held-out rows' current and throttle per band (see M3_bank_rows_pinned.py).
#   pwsh -NoProfile -File wslrun.ps1 -ScriptFile M3_bank_rows_pinned.sh -TimeoutSec 1200 -Name m3rows
# Same environment rule as M3_recompute_pinned.sh (the gate's software check needs JAX_PLATFORMS=cpu, OMP_NUM_THREADS=1).
set -e
cd /home/user/bca
unset JAX_ENABLE_X64 XLA_FLAGS JAX_DEFAULT_MATMUL_PRECISION
export JAX_PLATFORMS=cpu OMP_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
echo "repository commit: $(git rev-parse HEAD)"
/tmp/claude-0/bcavenv/bin/python -u /tmp/bridge/work/m3/M3_bank_rows_pinned.py 2>&1

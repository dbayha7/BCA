# Module M3 real-data recompute in the pinned environment (see M3_recompute_pinned.py next to this file).
# Runs as a bridge job from Windows:
#   pwsh -NoProfile -File wslrun.ps1 -ScriptFile M3_recompute_pinned.sh -TimeoutSec 3000 -Name m3recompute2
# The job's working directory is /home/user/bca (the owner's clone, read-only); it writes under /tmp/bridge/work/m3 only.
# Before running, copy M3_recompute_pinned.py to /tmp/bridge/work/m3/ (the bridge folder is \wsl.localhost\Ubuntu-24.04\tmp\bridge).
# The gate's software check (experiments/shift/acceptance.py check_software) requires exactly the flags the
# pre-registration recorded (runs/shift/preregistration.json, software.flags): JAX_PLATFORMS=cpu, OMP_NUM_THREADS=1,
# and JAX_ENABLE_X64, XLA_FLAGS, JAX_DEFAULT_MATMUL_PRECISION unset.
set -e
cd /home/user/bca
unset JAX_ENABLE_X64 XLA_FLAGS JAX_DEFAULT_MATMUL_PRECISION
export JAX_PLATFORMS=cpu OMP_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
echo "repository commit: $(git rev-parse HEAD)"
/tmp/claude-0/bcavenv/bin/python -u /tmp/bridge/work/m3/M3_recompute_pinned.py 2>&1

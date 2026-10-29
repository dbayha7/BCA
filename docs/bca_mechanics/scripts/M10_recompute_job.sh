# M10 (harm switch certificate): the job file run through the WSL bridge to recompute the near-80 certificates from
# their recorded inputs in the PINNED environment (Python 3.10.20, numpy 2.2.6, jax 0.6.2, CPU).
#
# How it was run on 20 October 2026 (once per setting; each run took about 90 s):
#   1. copy M10_recompute_pinned.py to \wsl.localhost\Ubuntu-24.04\tmp\bridge\m10_recompute_m10.py
#   2. write this file with SETTING replaced by main, near80 or random_rows (the three copies were
#      scratchpad/mech_work/m10/job_{main,near80,random_rows}.sh)
#   3. pwsh -NoProfile -File <scratchpad>\wslrun.ps1 -ScriptFile <that copy> -TimeoutSec 1500 -Name m10rc_<setting>
#   4. read back /tmp/bridge/work/m10/recomp_<setting>.json and overestimates/*.npz over \wsl.localhost and merge
#      them into docs/bca_mechanics/data/M10/recomputed_certificates.json and overestimates.npz
# The job reads the owner's clone /home/user/bca read-only (runs/shift_explore/near80 and runs/shift) and writes only
# under /tmp/bridge/work/m10/. The environment flags are the ones the gate's software check requires
# (runs/shift/preregistration.json, software.flags). Nothing is trained.
SETTING=${1:-main}
cd /home/user/bca
mkdir -p /tmp/bridge/work/m10
cp /tmp/bridge/m10_recompute_m10.py /tmp/bridge/work/m10/recompute_m10.py
unset JAX_ENABLE_X64 XLA_FLAGS JAX_DEFAULT_MATMUL_PRECISION
export PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=/home/user/bca JAX_PLATFORMS=cpu OMP_NUM_THREADS=1
/tmp/claude-0/bcavenv/bin/python -I /tmp/bridge/work/m10/recompute_m10.py "$SETTING"

# Module M5 (BCA's threshold with refusal) in the project's PINNED environment: Python 3.10.20, NumPy 2.2.6, JAX 0.6.2
# on CPU, inside WSL Ubuntu-24.04. Run on 2026-10-20 through the owner's bridge (scratchpad/wslrun.ps1):
#     pwsh -NoProfile -File <scratchpad>\wslrun.ps1 -ScriptFile <this file> -TimeoutSec 900 -Name m5pinned
# The job runs as root with the working directory /home/user/bca (the owner's clone, read-only use; the run files
# under runs/shift_explore are read, never written). docs/bca_mechanics/scripts/M5_test.py was copied to
# /tmp/bridge/work/m5/M5_test.py beforehand (the clone may not carry this branch). The output is kept at
# docs/bca_mechanics/data/M5_pinned_output.txt. Nothing is trained; the script only recomputes recorded thresholds
# from recorded held-out scores and runs synthetic checks.
PY=/tmp/claude-0/bcavenv/bin/python
cd /home/user/bca || exit 97
echo "--- environment"
$PY -c "import sys, numpy, jax, scipy; print('pinned:', sys.version.split()[0], 'numpy', numpy.__version__, 'jax', jax.__version__, 'scipy', scipy.__version__)"
git log --oneline -1
echo "--- input checksums (recorded held-out scores)"
sha256sum runs/shift_explore/near80/learners/near80/stall/seed0/v6_bank_scores.npz \
          runs/shift_explore/near80/learners/near80/no_stall/seed0/v6_bank_scores.npz \
          runs/shift_explore/stage2/detect/uniform/stall/fixed/seed0/bank_scores.npz
echo "--- the module's own unit tests: experiments/wbcp/test_wbcp.py"
PYTHONPATH=/home/user/bca PYTHONDONTWRITEBYTECODE=1 $PY -m unittest -v experiments.wbcp.test_wbcp 2>&1 | tail -20
echo "--- M5_test.py, both parts, in the pinned environment"
mkdir -p /tmp/bridge/work/m5/cache
PYTHONPATH=/home/user/bca PYTHONDONTWRITEBYTECODE=1 $PY /tmp/bridge/work/m5/M5_test.py --root /home/user/bca \
    --part all --no-figure --cache /tmp/bridge/work/m5/cache --sets 400 --json /tmp/bridge/work/m5/M5_pinned_results.json
echo "--- exit $?"

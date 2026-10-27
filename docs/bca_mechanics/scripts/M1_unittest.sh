# The WSL bridge job that runs the repository's own M1 unit test in the pinned environment:
#   experiments/shift_iw/test_learners_iw.py::SyntheticBandTest::test_critic_parts_are_bcas_own_target_and_q
# (synthetic rows and a synthetic host; nothing from the study data). From Windows:
#   pwsh -NoProfile -File <scratchpad>/wslrun.ps1 -ScriptFile <repo>/docs/bca_mechanics/scripts/M1_unittest.sh -TimeoutSec 1800 -Name m1unittest
# Runs as root with the working directory /home/user/bca (the owner's clone, read-only); temporary files go under
# /tmp/bridge/work/m1/unittest. Afterwards it lists any file the run left inside the clone (must print nothing).
set -u
unset JAX_ENABLE_X64 XLA_FLAGS JAX_DEFAULT_MATMUL_PRECISION
export JAX_PLATFORMS=cpu OMP_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 PYTHONPYCACHEPREFIX=/nonexistent/no_pycache
WORK=/tmp/bridge/work/m1/unittest
mkdir -p "$WORK" && export TMPDIR="$WORK"
MARK="$WORK/marker"; : > "$MARK"
cd /home/user/bca || exit 97
echo "== start $(date -u +%Y-%m-%dT%H:%M:%SZ)"
/tmp/claude-0/bcavenv/bin/python -c "import sys, numpy, jax, flax; print('python', sys.version.split()[0], 'numpy', numpy.__version__, 'jax', jax.__version__, 'flax', flax.__version__)"
/tmp/claude-0/bcavenv/bin/python -m unittest experiments.shift_iw.test_learners_iw -k critic_parts_are_bcas_own_target -v 2>&1 | tail -15
echo "== exit ${?}"
echo "== files newer than the marker inside the clone's experiments tree (must be empty):"
find /home/user/bca/experiments -newer "$MARK" 2>/dev/null | head
echo "== end $(date -u +%Y-%m-%dT%H:%M:%SZ)"

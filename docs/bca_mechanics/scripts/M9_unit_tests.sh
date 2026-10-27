# M9 synthetic tests: the module's own unit tests of the lower edge, pinned environment (Python 3.10, numpy 2.2.6, jax 0.6.2).
cd /home/user/bca || exit 1
export JAX_PLATFORMS=cpu PYTHONDONTWRITEBYTECODE=1
echo "commit: $(git log --oneline -1)"
echo "python: $(/tmp/claude-0/bcavenv/bin/python -c 'import sys,numpy,jax; print(sys.version.split()[0], numpy.__version__, jax.__version__)')"
time /tmp/claude-0/bcavenv/bin/python -m unittest -v experiments.shift_iw.test_lower_edge \
  -k k_zero -k width_is_bcas_own -k gradient_reaches -k cannot_certify -k per_step_R -k weights_of_one \
  -k test_actor_step -k one_step_oracle -k the_step_is_exact -k per_step 2>&1
echo "unittest exit: $?"

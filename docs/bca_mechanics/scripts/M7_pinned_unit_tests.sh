#!/bin/sh
# Module M7 (the dose): the seven unit tests of the bit-for-bit identities, run in the project's PINNED environment
# (WSL Ubuntu-24.04, /home/user/bca, Python 3.10.20, NumPy 2.2.6, JAX 0.6.2, CPU) on synthetic data (the tests build
# their own rows; nothing on record is trained). They are experiments/shift_iw/test_learners_iw.py:
#   test_weights_of_one_equal_the_unweighted_arm (line 494): weights of 1 give the standard arm, state for state;
#   test_per_row_arm_with_weights_of_one_is_the_unweighted_arm (line 524): the per-row arm with weights of 1 is BCA itself;
#   test_per_step_equals_unweighted_under_weights_of_one, test_per_row_dose_uses_each_rows_own_threshold,
#   test_refusal_gives_the_largest_dose, test_refused_rows_get_the_largest_dose_and_are_counted,
#   test_uncertifiable_threshold_refuses_every_step.
# Run from Windows through the owner's bridge:
#   pwsh -NoProfile -File <scratchpad>\wslrun.ps1 -ScriptFile docs\bca_mechanics\scripts\M7_pinned_unit_tests.sh -TimeoutSec 1800 -Name m7tests
# or inside WSL: sh docs/bca_mechanics/scripts/M7_pinned_unit_tests.sh.  Output on record: data/M7_pinned_unit_tests.out
# (7 tests, OK, 46 seconds, clone at commit 67209d2).
cd /home/user/bca || exit 97
export PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=/home/user/bca JAX_PLATFORMS=cpu
PY=/tmp/claude-0/bcavenv/bin/python
$PY -c "import sys, numpy, jax; print(sys.version.split()[0], numpy.__version__, jax.__version__)"
git log --oneline -1
timeout 1700 $PY -m unittest -v experiments.shift_iw.test_learners_iw -k per_row_dose_uses_each_rows_own_threshold -k refusal_gives_the_largest_dose -k per_step_equals_unweighted_under_weights_of_one -k weights_of_one_equal_the_unweighted_arm -k per_row_arm_with_weights_of_one_is_the_unweighted_arm -k refused_rows_get_the_largest_dose_and_are_counted -k uncertifiable_threshold_refuses_every_step 2>&1 | tail -40
echo "unittest exit: $?"

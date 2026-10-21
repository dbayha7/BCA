#!/usr/bin/env bash
# Calibration of probe_s7.py's readings on made-up ReBRAC actors (v2notes/rr/calibrate.py), 4 cases at a time.
S=/tmp/claude-0/-home-user-BCA/8720079f-929e-55d9-81af-561745cb590b/scratchpad
one() { env PYTHONDONTWRITEBYTECODE=1 JAX_PLATFORMS=cpu OMP_NUM_THREADS=1 BCA_SHIFT_DEPS=$S/v2notes/smoke/deps \
  $S/smokevenv/bin/python -B $S/v2notes/rr/calibrate.py $S/v2wt $S/v2notes/rr/new $S/v2notes/rr/cal/root_$1_$2 \
  $S/v2notes/rr/cal/out_$1_$2.txt $1 $2 > $S/v2notes/rr/cal/run_$1_$2.log 2>&1; }
rm -f $S/v2notes/rr/cal/out_*.txt
one 1 none & one 1.25 none & one 1.5 none & one 2 none & wait
one 3 none & one 1 bias & one 2 bias & one 1 branch & wait
echo done

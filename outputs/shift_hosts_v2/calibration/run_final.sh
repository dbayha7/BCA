#!/usr/bin/env bash
# Final cloud runs of v2notes/probe_s7.py (made-up actors only): the smoke test (4 learners, steps 15000 and 10000,
# refusals), the negative controls, and the calibration cases of v2notes/rr/calibrate.py. At most 4 at once.
S=/tmp/claude-0/-home-user-BCA/8720079f-929e-55d9-81af-561745cb590b/scratchpad
PY="env PYTHONDONTWRITEBYTECODE=1 JAX_PLATFORMS=cpu OMP_NUM_THREADS=1 BCA_SHIFT_DEPS=$S/v2notes/smoke/deps $S/smokevenv/bin/python -B"
job() {
  case "$1" in
    smoke) $PY $S/v2notes/smoke/smoke_probe.py $S/v2wt $S/v2notes $S/v2notes/smoke/root $S/v2notes/rr/final/smoke_out_15000.txt $S/v2notes/rr/final/smoke_out_10000.txt > $S/v2notes/rr/final/smoke_run.log 2>&1 ;;
    *) set -- $1; $PY $S/v2notes/rr/calibrate.py $S/v2wt $S/v2notes $S/v2notes/rr/cal/root_$1_$2 $S/v2notes/rr/cal/out_$1_$2.txt $1 $2 > $S/v2notes/rr/cal/run_$1_$2.log 2>&1 ;;
  esac
}
export -f job; export S PY
date -u +%T
printf '%s\n' smoke "1 none" "1.25 none" "1.5 none" "2 none" "3 none" "1 bias" "2 bias" "3 bias" "1 branch" | xargs -P 4 -I{} bash -c 'job "{}"'
$PY $S/v2notes/smoke/smoke_negative.py $S/v2wt $S/v2notes $S/v2notes/smoke/root > $S/v2notes/rr/final/negative.log 2>&1
date -u +%T
echo done

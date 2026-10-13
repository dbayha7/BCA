#!/bin/bash
# EXPLORATORY: wait for process $1 (0: none) to end, then run jobs "seed world actor plan" one after another.
cd /home/user/bca
LOGDIR=/home/user/bca/runs/shift_explore/fallback/logs
mkdir -p $LOGDIR
waitpid=$1; shift
if [ "$waitpid" != "0" ]; then while kill -0 $waitpid 2>/dev/null; do sleep 5; done; echo "$(date +%H:%M:%S) process $waitpid ended"; fi
for job in "$@"; do
  set -- $job
  echo "$(date +%H:%M:%S) start $job"
  PYTHONPATH=/home/user/bca PYTHONDONTWRITEBYTECODE=1 JAX_PLATFORMS=cpu OMP_NUM_THREADS=1 nice -n 10 \
    /tmp/claude-0/bcavenv/bin/python /tmp/claude-0/-home-user/9e341792-006d-5a11-a8e9-8fac16ac7f5b/scratchpad/ood/fallback/fallback.py $1 $2 $3 --n 200 --plan $4 \
    > $LOGDIR/seed$1_$2_$3.log 2>&1
  rc=$?
  echo "$(date +%H:%M:%S) done $job exit $rc"
done
echo "$(date +%H:%M:%S) queue finished"

#!/bin/sh
# EXPLORATORY deploy component (plan 4.3; no training, no BCA). Runs the jobs of the given seeds one after another in
# one pinned low-priority process, then optionally the pooled summary. Each job refuses unless the deploy spec, the
# code manifest, the pinned room harness and fallback.py, and the gate files check out, and the room test's pooled
# look exists for the pinned room spec and passed. --after-room-failure (recorded in every output) lets the jobs run
# after a room failure instead: a pooled look that did not pass, or a stop at the room's futility look (no pooled look
# is written then). Whether to use it is the user's decision (spec deviation D7).
# Usage: sh launch.sh <seeds|summary> <cpu> [--after-room-failure]
#   seeds: comma-separated gate seeds, e.g. 0,1   (each seed runs its 5 jobs:
#          stall/gapped stall/td3bc_gapped no_stall/no_stall_gapped no_stall/td3bc_gapped no_stall/gapped)
#   e.g. sh launch.sh 0,1 0      sh launch.sh 2,3 1      sh launch.sh 4 2      sh launch.sh summary 0
# Logs: /home/user/bca/runs/shift_explore/deploy/logs/seed<k>.log (summary.log for the summary).
D=/tmp/claude-0/-home-user/9e341792-006d-5a11-a8e9-8fac16ac7f5b/scratchpad/ood/deploy
LOGDIR=/home/user/bca/runs/shift_explore/deploy/logs
SEEDS=$1; CPU=$2; EXTRA=$3
mkdir -p "$LOGDIR"
cd /home/user/bca || exit 1
export PYTHONPATH=/home/user/bca PYTHONDONTWRITEBYTECODE=1 JAX_PLATFORMS=cpu OMP_NUM_THREADS=1
PY=/tmp/claude-0/bcavenv/bin/python
if [ "$SEEDS" = summary ]; then
  LOG="$LOGDIR/summary.log"
  echo "$(date -u +%FT%TZ) summary start (cpu $CPU)" >> "$LOG"
  nice -n 15 taskset -c "$CPU" $PY "$D/summarise.py" >> "$LOG" 2>&1 \
    && echo "$(date -u +%FT%TZ) summary done" >> "$LOG" || { echo "$(date -u +%FT%TZ) summary FAILED" >> "$LOG"; exit 1; }
  exit 0
fi
FAILED=0
for SEED in $(echo "$SEEDS" | tr ',' ' '); do
  LOG="$LOGDIR/seed${SEED}.log"
  for JOB in stall:gapped stall:td3bc_gapped no_stall:no_stall_gapped no_stall:td3bc_gapped no_stall:gapped; do
    WORLD=${JOB%%:*}; ACTOR=${JOB#*:}
    echo "$(date -u +%FT%TZ) seed $SEED $WORLD $ACTOR start (cpu $CPU)" >> "$LOG"
    # shellcheck disable=SC2086
    if nice -n 15 taskset -c "$CPU" $PY "$D/run_job.py" --seed "$SEED" --world "$WORLD" --actor "$ACTOR" $EXTRA >> "$LOG" 2>&1; then
      echo "$(date -u +%FT%TZ) seed $SEED $WORLD $ACTOR done" >> "$LOG"
    else
      echo "$(date -u +%FT%TZ) seed $SEED $WORLD $ACTOR FAILED" >> "$LOG"; FAILED=1
    fi
  done
done
exit $FAILED

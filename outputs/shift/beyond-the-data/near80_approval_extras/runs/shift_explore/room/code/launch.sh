#!/bin/sh
# EXPLORATORY room test (plan 4.0/4.1/7): one seed (or the pooled looks), one pinned low-priority process, phases in
# order. Every phase refuses unless the spec, the gate files and the code manifest check out.
# Usage: sh launch.sh <seed|all> <cpu> <phase> [<phase> ...]
#   e.g. sh launch.sh 0 0 train eval      sh launch.sh 1 1 train eval
#        sh launch.sh 0 0 no_stall_train no_stall_eval replication_train replication_eval
#        sh launch.sh all 0 futility       sh launch.sh all 0 pooled
# Each phase runs only if the previous one succeeded (a failed check stops the seed with stop.json).
# Logs: /home/user/bca/runs/shift_explore/room/logs/seed<k>.log (looks.log for futility and pooled).
R=/tmp/claude-0/-home-user/9e341792-006d-5a11-a8e9-8fac16ac7f5b/scratchpad/ood/room
LOGDIR=/home/user/bca/runs/shift_explore/room/logs
SEED=$1; CPU=$2; shift 2
mkdir -p "$LOGDIR"
cd /home/user/bca || exit 1
export PYTHONPATH=/home/user/bca PYTHONDONTWRITEBYTECODE=1 JAX_PLATFORMS=cpu OMP_NUM_THREADS=1
if [ "$SEED" = all ]; then LOG="$LOGDIR/looks.log"; SARG=""; else LOG="$LOGDIR/seed${SEED}.log"; SARG="--seed $SEED"; fi
for PHASE in "$@"; do
  echo "$(date -u +%FT%TZ) seed $SEED phase $PHASE start (cpu $CPU)" >> "$LOG"
  # shellcheck disable=SC2086
  nice -n 5 taskset -c "$CPU" /tmp/claude-0/bcavenv/bin/python "$R/run_seed.py" $SARG --phase "$PHASE" \
    >> "$LOG" 2>&1 || { echo "$(date -u +%FT%TZ) seed $SEED phase $PHASE FAILED" >> "$LOG"; exit 1; }
  echo "$(date -u +%FT%TZ) seed $SEED phase $PHASE done" >> "$LOG"
done

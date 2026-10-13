#!/bin/sh
# EXPLORATORY data-only stand-ins (plan 4.2): one seed (or the marks look), one pinned low-priority process, phases in
# order. Every phase refuses unless the spec, the gate files, the room harness's pinned bytes and the code manifest
# check out; train also refuses until the room test's pooled look has passed its room, recovery and
# keeps-improvement marks.
# Usage: sh launch.sh <seed|all> <cpu> <phase> [<phase> ...]
#   e.g. sh launch.sh 0 0 train eval no_stall_train no_stall_eval
#        sh launch.sh all 0 marks
# Each phase runs only if the previous one succeeded (a failed check stops the seed with stop.json).
# Logs: /home/user/bca/runs/shift_explore/standins/logs/seed<k>.log (marks.log for the marks look).
R=/tmp/claude-0/-home-user/9e341792-006d-5a11-a8e9-8fac16ac7f5b/scratchpad/ood/standins
LOGDIR=/home/user/bca/runs/shift_explore/standins/logs
SEED=$1; CPU=$2; shift 2
mkdir -p "$LOGDIR"
cd /home/user/bca || exit 1
export PYTHONPATH=/home/user/bca PYTHONDONTWRITEBYTECODE=1 JAX_PLATFORMS=cpu OMP_NUM_THREADS=1
if [ "$SEED" = all ]; then LOG="$LOGDIR/marks.log"; SARG=""; else LOG="$LOGDIR/seed${SEED}.log"; SARG="--seed $SEED"; fi
for PHASE in "$@"; do
  echo "$(date -u +%FT%TZ) EXPLORATORY stand-ins (plan 4.2), not a gated output: seed $SEED phase $PHASE start (cpu $CPU)" >> "$LOG"
  # shellcheck disable=SC2086
  nice -n 5 taskset -c "$CPU" /tmp/claude-0/bcavenv/bin/python "$R/run_standins.py" $SARG --phase "$PHASE" \
    >> "$LOG" 2>&1 || { echo "$(date -u +%FT%TZ) seed $SEED phase $PHASE FAILED" >> "$LOG"; exit 1; }
  echo "$(date -u +%FT%TZ) seed $SEED phase $PHASE done" >> "$LOG"
done

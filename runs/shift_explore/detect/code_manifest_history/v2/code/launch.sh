#!/bin/sh
# EXPLORATORY detection rehearsal (plan 4.4), no BCA. Runs ONLY after the room test's pooled look passes.
# One seed (or the pooled phase), one low-priority process pinned to one CPU, phases in order, each phase its own
# process (the label script never shares a process with signal code). Every phase refuses unless the spec, the gate
# files, the room pins and the code manifest check out.
# Usage: sh launch.sh <seed|all> <cpu> <stall|no_stall|both> <phase> [<phase> ...]
#   e.g. sh launch.sh 0 3 both pairs signals labels measures
#        sh launch.sh all 3 both pooled
# Logs: /home/user/bca/runs/shift_explore/detect/logs/seed<k>.log (pooled.log for the pooled phase).
D=/tmp/claude-0/-home-user/9e341792-006d-5a11-a8e9-8fac16ac7f5b/scratchpad/ood/detect
LOGDIR=/home/user/bca/runs/shift_explore/detect/logs
SEED=$1; CPU=$2; WORLD=$3; shift 3
mkdir -p "$LOGDIR"
cd /home/user/bca || exit 1
export PYTHONPATH=/home/user/bca PYTHONDONTWRITEBYTECODE=1 JAX_PLATFORMS=cpu OMP_NUM_THREADS=1
if [ "$SEED" = all ]; then LOG="$LOGDIR/pooled.log"; SARG=""; else LOG="$LOGDIR/seed${SEED}.log"; SARG="--seed $SEED"; fi
for PHASE in "$@"; do
  echo "$(date -u +%FT%TZ) seed $SEED world $WORLD phase $PHASE start (cpu $CPU)" >> "$LOG"
  # shellcheck disable=SC2086
  nice -n 15 taskset -c "$CPU" /tmp/claude-0/bcavenv/bin/python "$D/run_detect.py" $SARG --world "$WORLD" \
    --phase "$PHASE" >> "$LOG" 2>&1 || { echo "$(date -u +%FT%TZ) seed $SEED phase $PHASE FAILED" >> "$LOG"; exit 1; }
  echo "$(date -u +%FT%TZ) seed $SEED world $WORLD phase $PHASE done" >> "$LOG"
done

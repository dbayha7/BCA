#!/bin/sh
# Stage 2 (the flags claim): one phase in one pinned low-priority process. Every BCA phase refuses unless
# runs/shift_explore/stage2/approval.json approves the CURRENT prereg.json and every pinned script is unchanged.
# Usage: sh launch.sh <cpu> --phase <phase> [--seed K] [--world stall|no_stall] [--critic fixed|host]
#                                           [--version uniform|weighted] [--group exploratory|confirmatory]
#   e.g. sh launch.sh 0 --phase select_standin
#        sh launch.sh 0 --phase detect_fit --seed 0 --world stall --critic fixed
#        sh launch.sh 1 --phase policy_train --seed 1
# Logs: /home/user/bca/runs/shift_explore/stage2/logs/<phase>_<args>.log
S=/tmp/claude-0/-home-user/9e341792-006d-5a11-a8e9-8fac16ac7f5b/scratchpad/ood/stage2
LOGDIR=/home/user/bca/runs/shift_explore/stage2/logs
CPU=$1; shift
mkdir -p "$LOGDIR"
cd /home/user/bca || exit 1
export PYTHONPATH=/home/user/bca PYTHONDONTWRITEBYTECODE=1 JAX_PLATFORMS=cpu OMP_NUM_THREADS=1
NAME=$(echo "$*" | sed 's/--//g; s/  */_/g')
LOG="$LOGDIR/$NAME.log"
echo "$(date -u +%FT%TZ) start: $* (cpu $CPU)" >> "$LOG"
nice -n 5 taskset -c "$CPU" /tmp/claude-0/bcavenv/bin/python "$S/run_stage2.py" "$@" >> "$LOG" 2>&1 \
  || { echo "$(date -u +%FT%TZ) FAILED: $*" >> "$LOG"; exit 1; }
echo "$(date -u +%FT%TZ) done: $*" >> "$LOG"

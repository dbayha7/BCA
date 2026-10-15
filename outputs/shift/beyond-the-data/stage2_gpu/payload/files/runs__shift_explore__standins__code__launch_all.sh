#!/bin/sh
# EXPLORATORY data-only stand-ins (plan 4.2): seeds 0-4 at most 2 stand-in processes at once (plan: "at most 2 run at
# once here"; spec run_order), on the CPUs given (spec change C7). First waits until every process ID given has exited
# (e.g. the launcher of whatever now runs on those CPUs), then until no room-test process
# (scratchpad/ood/room/run_seed.py) runs, so the stand-ins never share the CPUs with the room test. Then two lanes, each
# one pinned low-priority process at a time (launch.sh): lane A seeds 0, 2, 4 on <cpu_A>; lane B seeds 1, 3 on <cpu_B>.
# Training and evaluation do not need the user's choice of mark forms (spec change C6). After the lanes, the marks look
# runs on <cpu_A> only if approval.json exists; otherwise it is left for the user's choice (approve.py), then:
#   sh launch.sh all <cpu> marks
# Every phase still refuses unless the spec, the gate files, the room import and the code manifest check out; train
# also refuses until the room's pooled look passed.
# Usage (from anywhere): sh launch_all.sh <cpu_A> <cpu_B> [<pid> ...]
#   detach it, e.g. nohup sh launch_all.sh 2 3 8598 >/dev/null 2>&1 &
# Logs: /home/user/bca/runs/shift_explore/standins/logs/ (seed<k>.log, marks.log, launch_all.log).
R=/tmp/claude-0/-home-user/9e341792-006d-5a11-a8e9-8fac16ac7f5b/scratchpad/ood/standins
OUT=/home/user/bca/runs/shift_explore/standins
LOGDIR=$OUT/logs
if [ "$#" -lt 2 ]; then
  echo "usage: sh launch_all.sh <cpu_A> <cpu_B> [<pid> ...]   (both CPUs are required; spec change C7)" >&2
  exit 2
fi
CA=$1; CB=$2; shift 2
for C in "$CA" "$CB"; do
  case "$C" in
    ''|*[!0-9]*) echo "refused: each lane's cpu must be one CPU number, got '$C'" >&2; exit 2 ;;
  esac
  if ! taskset -c "$C" true 2>/dev/null; then
    echo "refused: cpu '$C' is not available on this machine ($(nproc) CPUs)" >&2; exit 2
  fi
done
for P in "$@"; do
  case "$P" in
    ''|*[!0-9]*) echo "refused: process IDs to wait for must be numbers, got '$P'" >&2; exit 2 ;;
  esac
done
mkdir -p "$LOGDIR"
LOG="$LOGDIR/launch_all.log"
PH="train eval no_stall_train no_stall_eval"
for P in "$@"; do
  echo "$(date -u +%FT%TZ) EXPLORATORY stand-ins: waiting until process $P has exited" >> "$LOG"
  while kill -0 "$P" 2>/dev/null; do sleep 60; done
done
echo "$(date -u +%FT%TZ) EXPLORATORY stand-ins: waiting until no room-test process runs" >> "$LOG"
while pgrep -f "scratchpad/ood/room/run_seed.py" >/dev/null 2>&1; do sleep 60; done
echo "$(date -u +%FT%TZ) starting lanes A (seeds 0 2 4, cpu $CA) and B (seeds 1 3, cpu $CB)" >> "$LOG"
# shellcheck disable=SC2086
( for K in 0 2 4; do sh "$R/launch.sh" "$K" "$CA" $PH; done ) &
A=$!
# shellcheck disable=SC2086
( for K in 1 3; do sh "$R/launch.sh" "$K" "$CB" $PH; done ) &
B=$!
wait $A
wait $B
if [ -f "$OUT/approval.json" ]; then
  echo "$(date -u +%FT%TZ) lanes done; approval.json exists; marks look on cpu $CA" >> "$LOG"
  sh "$R/launch.sh" all "$CA" marks
  echo "$(date -u +%FT%TZ) launch_all finished (marks exit $?)" >> "$LOG"
else
  echo "$(date -u +%FT%TZ) lanes done; no approval.json yet: the marks look waits for the user's choice (approve.py), then: sh $R/launch.sh all <cpu> marks" >> "$LOG"
fi

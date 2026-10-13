#!/bin/sh
# EXPLORATORY data-only stand-ins (plan 4.2): the whole run, at most 2 stand-in processes at once (plan: "at most 2
# run at once here"; spec run_order). Waits first until no room-test process (scratchpad/ood/room/run_seed.py) is
# running, so the stand-ins never share the CPUs with the room test. Then two lanes, each one pinned low-priority
# process at a time (launch.sh): lane A seeds 0, 2, 4 on CPU 0; lane B seeds 1, 3 on CPU 1. Then the marks look.
# Every phase still refuses unless the spec, the gate files, the room import, the code manifest and the user's
# approval (approval.json, written by approve.py) check out; train also refuses until the room's pooled look passed.
# Usage (from anywhere): sh launch_all.sh            (detach it, e.g. nohup sh launch_all.sh >/dev/null 2>&1 &)
# Logs: /home/user/bca/runs/shift_explore/standins/logs/ (seed<k>.log, marks.log, launch_all.log).
R=/tmp/claude-0/-home-user/9e341792-006d-5a11-a8e9-8fac16ac7f5b/scratchpad/ood/standins
LOGDIR=/home/user/bca/runs/shift_explore/standins/logs
mkdir -p "$LOGDIR"
LOG="$LOGDIR/launch_all.log"
PH="train eval no_stall_train no_stall_eval"
echo "$(date -u +%FT%TZ) EXPLORATORY stand-ins: waiting until no room-test process runs" >> "$LOG"
while pgrep -f "scratchpad/ood/room/run_seed.py" >/dev/null 2>&1; do sleep 60; done
echo "$(date -u +%FT%TZ) no room-test process; starting lanes A (seeds 0 2 4, cpu 0) and B (seeds 1 3, cpu 1)" >> "$LOG"
# shellcheck disable=SC2086
( for K in 0 2 4; do sh "$R/launch.sh" "$K" 0 $PH; done ) &
A=$!
# shellcheck disable=SC2086
( for K in 1 3; do sh "$R/launch.sh" "$K" 1 $PH; done ) &
B=$!
wait $A
wait $B
echo "$(date -u +%FT%TZ) lanes done; marks look" >> "$LOG"
sh "$R/launch.sh" all 0 marks
echo "$(date -u +%FT%TZ) launch_all finished (exit $?)" >> "$LOG"

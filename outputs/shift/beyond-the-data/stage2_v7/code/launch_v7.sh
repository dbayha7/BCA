#!/bin/sh
# Stage 2, version 7 (V7_SPEC.md C2-C5, C10): one phase in one pinned low-priority process, through the v7 wrapper.
# It is v6's launch.sh (outputs/shift/beyond-the-data/stage2/code/launch.sh) plus:
#   - the processor must be ONE integer (C3); the wrapper also refuses unless exactly that processor is visible;
#   - one of 16 slot locks (flock -n R7/slots/slot_<i>.lock), refusing when none is free, so the cap of 16 holds even
#     outside the runner (C2); and a per-phase lock (flock -n R7/locks/<phase id>.lock), because the v6 writers check
#     first and write second and cannot stop a double start (S2C/stage2_common.py:247-252);
#   - env -i with exactly eight variables (C4): /bin/sh is dash, which exports PWD and OLDPWD after a cd;
#   - the real exit status, kept in the log and the ledger (v6's launch.sh turns every failure into 1);
#   - log folder R7/logs/ (same log names as launch.sh); the ledger R7/ledger/<phase id>/<n>.writes and <n>.json (C10);
#   - the wrapper run_stage2_v7.py as entry; it hands over to the pinned run_stage2.py unchanged.
# Usage: sh launch_v7.sh <cpu> --phase <phase> [--seed K] [--world stall|no_stall] [--critic fixed|host]
#                                               [--version uniform|weighted] [--group exploratory|confirmatory]
#   e.g. sh launch_v7.sh 3 --phase policy_train --seed 6
# Exit status: the phase's own (0; 2 for a recorded stop; 128+N when killed by signal N), or 97 for a v7 refusal.
# '--v7-test <fixture>' as the first two arguments is for test_v7.py only (a synthetic fixture outside /home/user/bca).
S=/tmp/claude-0/-home-user/9e341792-006d-5a11-a8e9-8fac16ac7f5b/scratchpad/ood/stage2
CODE=/home/user/bca/outputs/shift/beyond-the-data/stage2_v7/code
R7=/home/user/bca/runs/shift_explore/stage2_v7
OUTDIR=/home/user/bca/runs/shift_explore/stage2
PY=/tmp/claude-0/bcavenv/bin/python
SLOTS=16
CLEAN="env -i PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin HOME=/root LANG=C.UTF-8 LC_ALL=C.UTF-8 PYTHONPATH=/home/user/bca PYTHONDONTWRITEBYTECODE=1 JAX_PLATFORMS=cpu OMP_NUM_THREADS=1"
TESTCFG=""
if [ "$1" = "--v7-test" ]; then
    case "$2" in
        /home/user/bca|/home/user/bca/*|"") echo "refused: a test fixture must lie outside /home/user/bca" >&2; exit 97;;
    esac
    R7=$2/r7; OUTDIR=$2/out; TESTCFG=$2/config.json
    shift 2
fi
CPU=$1
[ $# -gt 0 ] && shift
case "$CPU" in
    ''|*[!0-9]*) echo "refused: the processor must be one integer (got '$CPU'); every phase runs on exactly one" >&2
                 exit 97;;
esac
cd /home/user/bca || exit 97
[ -d "$S" ] || { echo "refused: $S (the v6 code folder) is missing; run restore_v7.sh" >&2; exit 97; }
mkdir -p "$R7/logs" "$R7/slots" "$R7/locks" "$R7/ledger" || exit 97
PHASE_ID=$($CLEAN "$PY" "$CODE/v7_ledger.py" phase-id -- "$@") || exit 97
exec 8>"$R7/locks/$PHASE_ID.lock"
if ! flock -n 8; then
    echo "refused: phase $PHASE_ID is already running (its lock $R7/locks/$PHASE_ID.lock is held)" >&2
    exit 97
fi
SLOT=""
i=1
while [ "$i" -le "$SLOTS" ]; do
    exec 9>"$R7/slots/slot_$i.lock"
    if flock -n 9; then
        SLOT=$i
        break
    fi
    exec 9>&-
    i=$((i + 1))
done
if [ -z "$SLOT" ]; then
    echo "refused: all $SLOTS slots are busy (at most $SLOTS Stage 2 phases run at once)" >&2
    exit 97
fi
TESTFLAG=""
[ -n "$TESTCFG" ] && TESTFLAG="--test"
LEDGER=$($CLEAN "$PY" "$CODE/v7_ledger.py" begin --r7 "$R7" --out "$OUTDIR" --cpu "$CPU" --slot "$SLOT" $TESTFLAG \
         -- "$@") || exit 97
NAME=$(echo "$*" | sed 's/--//g; s/  */_/g')
LOG="$R7/logs/$NAME.log"
echo "$(date -u +%FT%TZ) start: $* (cpu $CPU, slot $SLOT, ledger $LEDGER)" >> "$LOG"
ARGS="$*"
if [ -n "$TESTCFG" ]; then
    set -- --v7-ledger "$LEDGER" --v7-cpu "$CPU" --v7-test-config "$TESTCFG" -- "$@"
else
    set -- --v7-ledger "$LEDGER" --v7-cpu "$CPU" -- "$@"
fi
$CLEAN nice -n 5 taskset -c "$CPU" "$PY" "$CODE/run_stage2_v7.py" "$@" >> "$LOG" 2>&1
STATUS=$?
if [ "$STATUS" -eq 0 ]; then
    echo "$(date -u +%FT%TZ) done: $ARGS" >> "$LOG"
else
    echo "$(date -u +%FT%TZ) FAILED (exit status $STATUS): $ARGS" >> "$LOG"
fi
$CLEAN "$PY" "$CODE/v7_ledger.py" end --ledger "$LEDGER" --status "$STATUS" --log "$LOG" --out "$OUTDIR" \
    --r7 "$R7" >> "$LOG" 2>&1
exit "$STATUS"

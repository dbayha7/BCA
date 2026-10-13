#!/bin/sh
# EXPLORATORY trace test bed (plan 4.6), spec v3: one gate seed (or the pooled look) in one process pinned to one CPU,
# at low priority. No BCA. Every phase refuses unless the spec, the gate files and the code manifest check out, and
# every job phase refuses unless the process is pinned to one CPU (taskset below does this).
# Spec v3 is POST HOC: its shares 0.01 0.03 0.05 0.1 were chosen after seeing the 0.5% and 2% results. The spec-v2
# jobs (share 0, 0.005, 0.02) are complete and kept as they are: their references are skipped, and a spec-v2 share
# given explicitly is refused.
# Usage: sh launch.sh <seed> <cpu> [<world>|all] [<share>|all]
#        sh launch.sh pooled <cpu>
#        sh launch.sh all <cpu A> [<cpu B>]
#   Every CPU is given explicitly as one number, and only CPU 0 or CPU 1 is allowed: CPUs 2 and 3 run the stand-ins
#   and are refused before anything is pinned or run (code manifest version 4). The launch also refuses a CPU it
#   cannot pin to (taskset) and, for 'all', two equal CPUs.
#   e.g. sh launch.sh 0 1             seed 0, both worlds, the four post-hoc shares, on CPU 1
#        sh launch.sh 2 0 no_stall    seed 2, no-stall world only, on CPU 0
#        sh launch.sh 1 1 stall 0.1   seed 1, stalling world, the 10% log only, on CPU 1
#        sh launch.sh all 0 1         seeds 0 and 1 side by side on CPUs 0 and 1, then seed 2 split by world on the
#                                     same CPUs, then the pooled look (pooled_v3.json)
#        sh launch.sh all 1           everything on CPU 1 alone: seeds 0, 1, 2 in turn, then the pooled look
# This script runs in the foreground. 'all' does the waiting itself; start it with
#        nohup sh launch.sh all 0 1 >/dev/null 2>&1 &
# To run two seeds side by side by hand, start each in the background (two commands joined by ';' run in sequence).
# Per world: the share-0 reference is a spec-v2 job (skipped when its signals.json exists, else reported missing);
# then per post-hoc share: build, train (lock), eval, signals.
# A job with stop.json is skipped (logged), and a failed job does not stop the others: the launch moves on to the
# next job and exits 1 at the end if any job failed. Rerunning the same launch resumes: phases whose output exists are
# skipped (a locked job is never retrained; a saved clone or critic is reused by the train phase).
# Logs: /home/user/bca/runs/shift_explore/trace/logs/v3_seed<k>[_<world>][_<share>].log (v3_pooled.log, v3_all.log);
# the spec-v2 logs (seed<k>*.log, pooled.log, all.log) are kept as they are.
# NICE sets the priority (default 15). PY and OUTDIR exist only for this script's own unit test (test_trace.py, with a
# stub interpreter); leave them unset.
R=/tmp/claude-0/-home-user/9e341792-006d-5a11-a8e9-8fac16ac7f5b/scratchpad/ood/trace
OUTDIR=${OUTDIR:-/home/user/bca/runs/shift_explore/trace}
LOGDIR=$OUTDIR/logs
PY=${PY:-/tmp/claude-0/bcavenv/bin/python}
NICE=${NICE:-15}
POST_HOC_SHARES="0.01 0.03 0.05 0.1"
SEED=$1; CPU=$2; WORLDS=${3:-all}; SHARES=${4:-all}
[ -n "$SEED" ] && [ -n "$CPU" ] || { echo "usage: sh launch.sh <seed|pooled|all> <cpu> [world|all] [share|all]"; exit 2; }

check_cpu() {  # check_cpu <cpu>: CPU 0 or CPU 1 (CPUs 2 and 3 run the stand-ins), and this process can be pinned to it
  case "$1" in ''|*[!0-9]*) echo "cpu must be one CPU number (got '$1')"; exit 2 ;; esac
  case "$1" in
    0|1) ;;
    *) echo "cpu $1 is not allowed: only CPU 0 or CPU 1 may run the trace jobs (CPUs 2 and 3 run the stand-ins)"
       exit 2 ;;
  esac
  taskset -c "$1" true 2>/dev/null || { echo "cpu $1 cannot be used here (taskset refused it)"; exit 2; }
}
check_cpu "$CPU"
if [ "$SEED" = all ] && [ -n "$3" ]; then   # 'all' on two CPUs: the second one is checked before anything is written
  check_cpu "$3"
  [ "$CPU" != "$3" ] || { echo "the two CPUs must differ (got $CPU and $3)"; exit 2; }
fi
mkdir -p "$LOGDIR"
cd /home/user/bca || exit 1
export PYTHONPATH=/home/user/bca PYTHONDONTWRITEBYTECODE=1 JAX_PLATFORMS=cpu OMP_NUM_THREADS=1

if [ "$SEED" = all ]; then
  CPU_B=$3
  ALOG="$LOGDIR/v3_all.log"
  if [ -z "$CPU_B" ]; then
    echo "$(date -u +%FT%TZ) all: seeds 0, 1 and 2 in turn on CPU $CPU" >> "$ALOG"
    ST=""
    for K in 0 1 2; do
      sh "$R/launch.sh" "$K" "$CPU"; S=$?; ST="$ST$S"
      echo "$(date -u +%FT%TZ) all: seed $K exit $S" >> "$ALOG"
    done
    sh "$R/launch.sh" pooled "$CPU"; S4=$?
    echo "$(date -u +%FT%TZ) all: pooled exit $S4" >> "$ALOG"
    [ "$ST$S4" = 0000 ] || exit 1
    exit 0
  fi
  echo "$(date -u +%FT%TZ) all: seeds 0 and 1 on CPUs $CPU and $CPU_B" >> "$ALOG"
  sh "$R/launch.sh" 0 "$CPU" & P0=$!
  sh "$R/launch.sh" 1 "$CPU_B" & P1=$!
  wait $P0; S0=$?; wait $P1; S1=$?
  echo "$(date -u +%FT%TZ) all: seed 0 exit $S0, seed 1 exit $S1; seed 2 split by world" >> "$ALOG"
  sh "$R/launch.sh" 2 "$CPU" stall & P2=$!
  sh "$R/launch.sh" 2 "$CPU_B" no_stall & P3=$!
  wait $P2; S2=$?; wait $P3; S3=$?
  echo "$(date -u +%FT%TZ) all: seed 2 exits $S2 (stall), $S3 (no stall); pooled look" >> "$ALOG"
  sh "$R/launch.sh" pooled "$CPU"; S4=$?
  echo "$(date -u +%FT%TZ) all: pooled exit $S4" >> "$ALOG"
  [ "$S0$S1$S2$S3$S4" = 00000 ] || exit 1
  exit 0
fi

[ "$WORLDS" = all ] && WORLDS="stall no_stall"
for W in $WORLDS; do
  case "$W" in stall|no_stall) ;; *) echo "unknown world $W"; exit 2 ;; esac
done
[ "$SHARES" = all ] && SHARES="$POST_HOC_SHARES"
JOBS=""
for P in $SHARES; do   # every share is checked before any job starts
  case "$P" in
    0.01) SD=share_1pct ;; 0.03) SD=share_3pct ;; 0.05) SD=share_5pct ;; 0.1|0.10) P=0.1; SD=share_10pct ;;
    0|0.0|0.005|0.02)
      echo "share $P is a spec-v2 job, complete and kept as it is; spec v3 runs only the post-hoc shares $POST_HOC_SHARES"
      exit 2 ;;
    *) echo "unknown share $P"; exit 2 ;;
  esac
  JOBS="$JOBS $P:$SD"
done
FAILED=0

run() {  # run <log> <args...>: one job; a failure is logged and counted, and the launch goes on
  LG=$1; shift
  echo "$(date -u +%FT%TZ) start: $* (cpu $CPU, nice $NICE)" >> "$LG"
  if nice -n "$NICE" taskset -c "$CPU" "$PY" "$R/run_job.py" "$@" >> "$LG" 2>&1; then
    echo "$(date -u +%FT%TZ) done: $*" >> "$LG"
  else
    echo "$(date -u +%FT%TZ) FAILED: $* (the launch goes on with the next job)" >> "$LG"
    FAILED=1
  fi
}

if [ "$SEED" = pooled ]; then
  run "$LOGDIR/v3_pooled.log" --phase pooled
  exit $FAILED
fi
case "$SEED" in *[!0-9]*) echo "seed must be a number, 'pooled' or 'all' (got '$SEED')"; exit 2 ;; esac
SUFFIX=""
[ "${3:-all}" = all ] || SUFFIX="${SUFFIX}_$3"
[ "${4:-all}" = all ] || SUFFIX="${SUFFIX}_$4"
LOG="$LOGDIR/v3_seed${SEED}${SUFFIX}.log"
for W in $WORLDS; do
  D=$OUTDIR/$W/seed$SEED/share_0pct
  if [ -f "$D/stop.json" ] || [ -f "$D/signals.json" ]; then
    echo "$(date -u +%FT%TZ) skip: $W seed $SEED reference is a spec-v2 job, already written (kept as it is)" >> "$LOG"
  else
    echo "$(date -u +%FT%TZ) MISSING: $W seed $SEED reference (a spec-v2 job; not rerun under spec v3)" >> "$LOG"
    FAILED=1
  fi
  for J in $JOBS; do
    P=${J%%:*}; SD=${J#*:}
    D=$OUTDIR/$W/seed$SEED/$SD
    if [ -f "$D/stop.json" ]; then
      echo "$(date -u +%FT%TZ) skip: $W seed $SEED $SD has stopped (stop.json)" >> "$LOG"
      continue
    fi
    PH="build"                                   # build is idempotent (an identical record is kept)
    [ -f "$D/lock.json" ] || PH="$PH train"      # a locked job is never retrained
    [ -f "$D/values.json" ] || PH="$PH eval"
    [ -f "$D/signals.json" ] || PH="$PH signals"
    # shellcheck disable=SC2086
    run "$LOG" --world "$W" --seed "$SEED" --share "$P" --phase $PH
  done
done
exit $FAILED

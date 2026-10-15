#!/usr/bin/env bash
# EXPLORATORY: outside the pre-registered claims; the version-5 gates did not pass.
#
# Launch the exploratory BCA arms of spec version 5 (explore_arms.py), their evaluation (explore_eval.py) and the
# summary (explore_summary.py). One process per job, each pinned to its own CPU (taskset), at most one job per CPU at a
# time, nice 10, single-threaded, JAX on the CPU (the official launch.sh's way).
#
#   bash launch_explore.sh            (no arguments; settings through the environment below)
#
# Environment (defaults in brackets):
#   CPUS       the CPUs to use ["4 5 ... 23"]; CPUs 0-3 belong to another session
#   SEEDS      ["0 1 2 3 4"]
#   QUALITIES  ["random medium-replay medium medium-expert expert"]
#   EXPLORE_OUT  the output folder [/root/bca-shift/runs/shift_iw_v5_explore]
#   SMOKE      1 for a smoke test (a few updates; needs EXPLORE_OUT set to a scratch folder) [0]
#
# Phases (each waits for the one before):
#   1. training: the capped pull's arms one job per quality, seed and arm (the spec's estimate is about 13 minutes
#      each, but version 5's own stages ran 2.3-2.7 times their estimates with 20 processes at once, so expect about
#      30; started first), then the lower edge's arms one job per quality and seed (its three arms one after another
#      in one process; estimate about 3 minutes, expect about 7); 100 jobs on 20 CPUs, about 2 hours of wall clock
#      (about 55 minutes if the spec's estimates held)
#   2. evaluation: one job per quality and seed (both groups; seconds each)
#   3. summary: one job on the first CPU; prints the table and writes summary/summary.txt and summary.json
# Logs: $EXPLORE_OUT/logs/<job>_<date-time>.log and $EXPLORE_OUT/logs/launch_<date-time>.log (this script's own
# lines). Finished records are never redone, so running this again resumes after a crash. The exit status is 0 only
# if every job exited 0 (2 means a script refused: see its log).
set -u
REPO=/root/bca-shift
PY=/opt/bca-shift-venv/bin/python
HERE=/mnt/c/Users/David/dev/bca-shift-work/iw_explore
OUT=${EXPLORE_OUT:-$REPO/runs/shift_iw_v5_explore}
CPUS=${CPUS:-"4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23"}
SEEDS=${SEEDS:-"0 1 2 3 4"}
QUALITIES=${QUALITIES:-"random medium-replay medium medium-expert expert"}
SMOKE=${SMOKE:-0}
# refuse a protected output folder before anything is created (the Python scripts refuse it too, but only after this
# script has made the logs folder): runs/shift_iw_v5 (read only), the code folders, and, for a smoke test, the real
# output folder
OUT_REAL=$(realpath -m "$OUT")
DEFAULT_REAL=$(realpath -m "$REPO/runs/shift_iw_v5_explore")
for P in "$REPO/runs/shift_iw_v5" "$REPO/experiments" "$REPO/runs/shift" "/mnt/c/Users/David/dev/BCA" \
         "/mnt/c/Users/David/dev/bca-shift-work"; do
  P_REAL=$(realpath -m "$P")
  case "$OUT_REAL/" in
    "$P_REAL/"*) echo "refused: output folder $OUT_REAL is inside $P_REAL" >&2; exit 64 ;;
  esac
done
if [ "$SMOKE" = "1" ]; then
  case "$OUT_REAL/" in
    "$DEFAULT_REAL/"*) echo "refused: a smoke test needs EXPLORE_OUT set to a scratch folder, not $DEFAULT_REAL" >&2
                       exit 64 ;;
  esac
fi
LOGS=$OUT/logs
mkdir -p "$LOGS"
STAMP=$(date +%Y%m%d-%H%M%S)
MAIN_LOG=$LOGS/launch_$STAMP.log
FLAGS=(--out "$OUT")
if [ "$SMOKE" = "1" ]; then FLAGS+=(--smoke); fi
for c in $CPUS; do
  if [ "$c" -lt 4 ]; then echo "refused: CPU $c belongs to another session (use 4-23)" | tee -a "$MAIN_LOG"; exit 64; fi
done

say() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$MAIN_LOG"; }

run_one() {  # run_one CPU LOGNAME SCRIPT ARGS... (foreground)
  local cpu=$1 name=$2 script=$3
  shift 3
  echo "[$(date +%H:%M:%S)] cpu $cpu: $script $*" >> "$LOGS/${name}_$STAMP.log"
  (cd "$REPO" && PYTHONPATH=$REPO PYTHONDONTWRITEBYTECODE=1 JAX_PLATFORMS=cpu OMP_NUM_THREADS=1 \
    nice -n 10 taskset -c "$cpu" "$PY" "$HERE/$script" "$@" "${FLAGS[@]}") >> "$LOGS/${name}_$STAMP.log" 2>&1
}

read -r -a CPU_LIST <<< "$CPUS"
FIRST_CPU=${CPU_LIST[0]}
FAILED=0

run_jobs() {  # run_jobs "name|script args" ...: one job per free CPU, waiting for a CPU when none is free
  declare -A CPU_OF NAME_OF
  local FREE=("${CPU_LIST[@]}") job name args cpu pid rc
  reap() {
    wait -n -p pid
    rc=$?
    if [ "$rc" -ne 0 ]; then
      FAILED=1
      say "FAILED (exit $rc): ${NAME_OF[$pid]} (log $LOGS/${NAME_OF[$pid]}_$STAMP.log)"
    else
      say "done: ${NAME_OF[$pid]}"
    fi
    FREE+=("${CPU_OF[$pid]}")
    unset "CPU_OF[$pid]" "NAME_OF[$pid]"
  }
  for job in "$@"; do
    name=${job%%|*}
    args=${job#*|}
    while [ ${#FREE[@]} -eq 0 ]; do reap; done
    cpu=${FREE[0]}
    FREE=("${FREE[@]:1}")
    # shellcheck disable=SC2086
    run_one "$cpu" "$name" $args &
    CPU_OF[$!]=$cpu
    NAME_OF[$!]=$name
    say "started: $name on cpu $cpu (pid $!)"
  done
  while [ ${#CPU_OF[@]} -gt 0 ]; do reap; done
}

say "EXPLORATORY: outside the pre-registered claims; the version-5 gates did not pass"
say "out $OUT; qualities $QUALITIES; seeds $SEEDS; cpus $CPUS; smoke $SMOKE"

# phase 1: training (the long capped-pull jobs first, then the short lower-edge jobs fill the CPUs)
TRAIN=()
for arm in unweighted exact_strong exact_strong_single; do
  for q in $QUALITIES; do
    for s in $SEEDS; do TRAIN+=("capped_${q}_seed${s}_${arm}|explore_arms.py --quality $q --seed $s --capped $arm"); done
  done
done
for q in $QUALITIES; do
  for s in $SEEDS; do
    TRAIN+=("lower_edge_${q}_seed${s}|explore_arms.py --quality $q --seed $s --lower-edge unweighted exact_strong exact_mild")
  done
done
say "phase 1: ${#TRAIN[@]} training jobs"
run_jobs "${TRAIN[@]}"

# phase 2: evaluation (runs even if a training job failed: missing arms are listed in the records and the summary)
EVAL=()
for q in $QUALITIES; do
  for s in $SEEDS; do EVAL+=("evaluate_${q}_seed${s}|explore_eval.py --quality $q --seed $s"); done
done
say "phase 2: ${#EVAL[@]} evaluation jobs"
run_jobs "${EVAL[@]}"

# phase 3: summary
say "phase 3: summary"
# shellcheck disable=SC2086
run_one "$FIRST_CPU" "summary" explore_summary.py --qualities $QUALITIES || FAILED=1
say "summary: $OUT/summary/summary.txt (log $LOGS/summary_$STAMP.log)"
say "$([ $FAILED -eq 0 ] && echo 'all jobs exited 0' || echo 'some jobs FAILED: see the lines above')"
exit $FAILED

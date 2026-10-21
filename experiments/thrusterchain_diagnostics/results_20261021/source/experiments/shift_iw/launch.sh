#!/usr/bin/env bash
# Launch one stage of the importance-weighting experiment (experiments/shift_iw/run.py) as one process per job, each
# pinned to its own CPU (taskset), at most one job per CPU at a time, at nice 10, single-threaded, JAX on the CPU.
#
#   bash experiments/shift_iw/launch.sh STAGE "SEEDS" "CPUS" [extra arguments for run.py]
#
# Dataset qualities (spec version 5): QUALITIES="random medium-replay medium medium-expert expert" (the default: all
# five) multiplies the job list; every job runs for one quality (run.py --quality) and reads and writes only that
# quality's folder. The start banks and the approval are shared.
#
# Jobs per stage:
#   data            the shared start banks once first (on the first CPU), then one job per quality and seed
#   host band coverage   one job per quality, seed and host kind (gap, full); SPLIT="gap" (for example) narrows the
#                   kinds (band refuses until the quality's report on all five seeds, made by this code, says D1 passed;
#                   the gapped band also reads the seed's room check, for L2's proxy check)
#   bca             the capped pull's arms: one job per quality, seed and arm (unweighted, exact_strong per row,
#                   exact_strong_single for comparison); SPLIT="estimated_strong" for the optional arm (per row).
#                   Approval scope 'bca' naming the quality (the optional arm also 'bca_estimated', which only the
#                   user's request adds); set QUALITIES to the qualities the user's go names
#   room room-lower-edge evaluate   one job per quality and seed (room-lower-edge after room: it reads the room
#                   check's clone and 'none' actor; evaluate defaults to hosts, room and room_lower_edge)
#   bca-lower-edge  the lower edge's arms: one job per quality and seed, its arms one after another in one process
#                   (unweighted, exact_strong, exact_mild). Approval scope 'bca_lower_edge' naming the quality (its own
#                   go, separate from 'bca'); extra arguments "--arms estimated_strong" for the optional arm (scope
#                   'bca_lower_edge_estimated'); set QUALITIES to the qualities the user's go names
#   report status   one job per quality, all seeds
#   summary freeze-spec approval-template   one job on the first CPU (no quality)
# Logs: $ROOT/logs/<stage>_<quality>_seed<k>[_<kind>]_<date-time>.log (ROOT defaults to
# /root/bca-shift/runs/shift_iw_v5; set SHIFT_IW_ROOT to use another root, for example a smoke test). The exit status
# is 0 only if every job exited 0 (2 means run.py refused: see the log).
set -u
if [ $# -lt 3 ]; then
  echo "usage: launch.sh STAGE \"SEEDS\" \"CPUS\" [extra run.py arguments]" >&2
  exit 64
fi
STAGE=$1
SEEDS=$2
CPUS=$3
shift 3
EXTRA=("$@")
REPO=/root/bca-shift
PY=/opt/bca-shift-venv/bin/python
ROOT=${SHIFT_IW_ROOT:-$REPO/runs/shift_iw_v5}
LOGS=$ROOT/logs
mkdir -p "$LOGS"
STAMP=$(date +%Y%m%d-%H%M%S)
ALL_QUALITIES="random medium-replay medium medium-expert expert"
QUALITIES=${QUALITIES:-$ALL_QUALITIES}
for q in $QUALITIES; do
  case " $ALL_QUALITIES " in
    *" $q "*) ;;
    *) echo "unknown quality $q (one of: $ALL_QUALITIES)" >&2; exit 64 ;;
  esac
done

case $STAGE in
  host|band|coverage) DEFAULT_SPLIT="gap full"; FLAG=--kinds ;;
  bca) DEFAULT_SPLIT="unweighted exact_strong exact_strong_single"; FLAG=--arms ;;
  data|room|room-lower-edge|bca-lower-edge|evaluate) DEFAULT_SPLIT=""; FLAG="" ;;
  report|status|summary|freeze-spec|approval-template) DEFAULT_SPLIT=""; FLAG="" ;;
  *) echo "unknown stage $STAGE" >&2; exit 64 ;;
esac
SPLIT=${SPLIT:-$DEFAULT_SPLIT}

run_one() {  # run_one CPU LOGNAME run.py-arguments... (foreground)
  local cpu=$1 name=$2
  shift 2
  echo "[$(date +%H:%M:%S)] cpu $cpu: run.py $*" >> "$LOGS/${name}_$STAMP.log"
  (cd "$REPO" && PYTHONPATH=$REPO PYTHONDONTWRITEBYTECODE=1 JAX_PLATFORMS=cpu OMP_NUM_THREADS=1 \
    nice -n 10 taskset -c "$cpu" "$PY" -m experiments.shift_iw.run "$@" --root "$ROOT") \
    >> "$LOGS/${name}_$STAMP.log" 2>&1
}

read -r -a CPU_LIST <<< "$CPUS"
read -r -a SEED_LIST <<< "$SEEDS"
FIRST_CPU=${CPU_LIST[0]}

case $STAGE in
  summary|freeze-spec|approval-template)
    run_one "$FIRST_CPU" "$STAGE" "$STAGE" "${EXTRA[@]}"
    rc=$?
    echo "$STAGE: exit $rc (log $LOGS/${STAGE}_$STAMP.log)"
    exit $rc ;;
  data)
    run_one "$FIRST_CPU" "data_start_banks" data --start-banks "${EXTRA[@]}" || {
      echo "data: the start banks failed (log $LOGS/data_start_banks_$STAMP.log)"; exit 1; } ;;
esac

# the job list: "name|run.py arguments"
JOBS=()
for q in $QUALITIES; do
  case $STAGE in
    report|status)
      JOBS+=("${STAGE}_${q}|$STAGE --quality $q --seeds ${SEED_LIST[*]}") ;;
    *)
      for s in "${SEED_LIST[@]}"; do
        if [ -n "$SPLIT" ]; then
          for k in $SPLIT; do JOBS+=("${STAGE}_${q}_seed${s}_${k}|$STAGE --quality $q --seeds $s $FLAG $k"); done
        else
          JOBS+=("${STAGE}_${q}_seed${s}|$STAGE --quality $q --seeds $s")
        fi
      done ;;
  esac
done

declare -A CPU_OF NAME_OF
FREE=("${CPU_LIST[@]}")
FAILED=0
reap() {  # wait for any job, free its CPU
  local pid rc
  wait -n -p pid
  rc=$?
  if [ "$rc" -ne 0 ]; then
    FAILED=1
    echo "[$(date +%H:%M:%S)] FAILED (exit $rc): ${NAME_OF[$pid]} (log $LOGS/${NAME_OF[$pid]}_$STAMP.log)"
  else
    echo "[$(date +%H:%M:%S)] done: ${NAME_OF[$pid]}"
  fi
  FREE+=("${CPU_OF[$pid]}")
  unset "CPU_OF[$pid]" "NAME_OF[$pid]"
}
for job in "${JOBS[@]}"; do
  name=${job%%|*}
  args=${job#*|}
  while [ ${#FREE[@]} -eq 0 ]; do reap; done
  cpu=${FREE[0]}
  FREE=("${FREE[@]:1}")
  # shellcheck disable=SC2086
  run_one "$cpu" "$name" $args "${EXTRA[@]}" &
  CPU_OF[$!]=$cpu
  NAME_OF[$!]=$name
  echo "[$(date +%H:%M:%S)] started: $name on cpu $cpu (pid $!)"
done
while [ ${#CPU_OF[@]} -gt 0 ]; do reap; done
echo "$STAGE: $([ $FAILED -eq 0 ] && echo 'all jobs exited 0' || echo 'some jobs FAILED')"
exit $FAILED

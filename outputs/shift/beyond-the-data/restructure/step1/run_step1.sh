#!/bin/sh
# Step 1 of the restructured BCA (the decisive test). EXPLORATORY; see PRESPEC.md and prespec.json.
# 1. runs the unit tests; 2. trains the 10 step-1 critics (seeds 0-4, both worlds) over 4 CPUs, each training pinned
# to ONE CPU with taskset (3 + 3 + 2 + 2 jobs, about 7-9 minutes each); 3. if all 10 records exist, evaluates on CPU 0.
# Writes only under runs/shift_explore/restructure/step1/ (critics/, logs/, results.json, results.md, run.lock).
# Write-once: a critic with a record is skipped, never retrained; train_critics.py and evaluate.py refuse to overwrite.
# Usage: sh run_step1.sh            (unattended; about 25-30 minutes of training wall time plus the evaluation)
#   Launch it DETACHED so that closing the session or a tool timeout cannot kill it half-way:
#   setsid nohup sh /home/user/bca/outputs/shift/beyond-the-data/restructure/step1/run_step1.sh </dev/null >/dev/null 2>&1 &
#   Progress: runs/shift_explore/restructure/step1/logs/run.log (one line per start / done / FAILED).
# Smoke test only: STEP1_SCRATCH=<folder outside the repository> [STEP1_UPDATES=200] [STEP1_EVAL_EXTRA='--perms 20']
#   sh run_step1.sh  runs the same steps with tiny trainings into that folder (the Python scripts refuse a scratch
#   folder inside the repository); it never touches the official folder.
set -u
HERE=$(cd "$(dirname "$0")" && pwd)
PY=${PY:-/tmp/claude-0/bcavenv/bin/python}
if [ -n "${STEP1_SCRATCH:-}" ]; then
  case "$STEP1_SCRATCH" in
    /home/user/bca|/home/user/bca/*) echo "refused: STEP1_SCRATCH must lie outside the repository"; exit 1 ;;
    /*) ;;
    *) echo "refused: STEP1_SCRATCH must be an absolute path"; exit 1 ;;
  esac
  OUT=$STEP1_SCRATCH
  TRAIN_EXTRA="--out-root $OUT/critics --updates ${STEP1_UPDATES:-200}"
  EVAL_EXTRA="--critics-root $OUT/critics --out $OUT ${STEP1_EVAL_EXTRA:-}"
else
  OUT=/home/user/bca/runs/shift_explore/restructure/step1
  TRAIN_EXTRA=""
  EVAL_EXTRA=""
fi
LOGS=$OUT/logs
export PYTHONPATH=/home/user/bca PYTHONDONTWRITEBYTECODE=1 JAX_PLATFORMS=cpu OMP_NUM_THREADS=1
mkdir -p "$LOGS"
LOCK=$OUT/run.lock
if ! mkdir "$LOCK" 2>/dev/null; then
  echo "refused: $LOCK exists (another run is going, or a previous run was killed: check, then remove it)"
  exit 1
fi
stamp() { date -u +%Y-%m-%dT%H:%M:%SZ; }
# Normal end: release the lock. Killed by a signal: KEEP the lock, because the background trainings may still be
# running, and a second launch would start duplicate trainings into the same write-once folders (review fix).
trap 'rmdir "$LOCK" 2>/dev/null' EXIT
trap 'echo "$(stamp) interrupted by a signal: $LOCK is kept; trainings already started may still be running (check with ps); when none is, remove $LOCK and rerun this script (finished critics are skipped)" | tee -a "$LOGS/run.log"; trap - EXIT; exit 130' INT TERM HUP
echo "$(stamp) step 1 start" | tee -a "$LOGS/run.log"

if ! "$PY" "$HERE/test_step1.py" > "$LOGS/tests.log" 2>&1; then
  echo "$(stamp) unit tests failed: see $LOGS/tests.log" | tee -a "$LOGS/run.log"
  exit 1
fi

queue() {  # $1 = CPU, then world:seed jobs, run one after another on that CPU
  cpu=$1; shift
  for job in "$@"; do
    w=${job%:*}; s=${job#*:}
    if [ -f "$OUT/critics/$w/seed$s/record.json" ]; then
      echo "$(stamp) skip $w/seed$s (record exists)" >> "$LOGS/run.log"; continue
    fi
    echo "$(stamp) start $w/seed$s on CPU $cpu" >> "$LOGS/run.log"
    if taskset -c "$cpu" "$PY" "$HERE/train_critics.py" --seed "$s" --world "$w" $TRAIN_EXTRA >> "$LOGS/train_${w}_seed${s}.log" 2>&1
    then echo "$(stamp) done $w/seed$s" >> "$LOGS/run.log"
    else echo "$(stamp) FAILED $w/seed$s (see $LOGS/train_${w}_seed${s}.log)" >> "$LOGS/run.log"
    fi
  done
}
queue 0 stall:0 no_stall:2 stall:4 &
queue 1 no_stall:0 stall:3 no_stall:4 &
queue 2 stall:1 no_stall:3 &
queue 3 no_stall:1 stall:2 &
wait

missing=""
for w in stall no_stall; do for s in 0 1 2 3 4; do
  [ -f "$OUT/critics/$w/seed$s/record.json" ] || missing="$missing $w/seed$s"
done; done
if [ -n "$missing" ]; then
  echo "$(stamp) not evaluated: critic records missing for$missing (an interrupted run can be rerun with this script)" \
    | tee -a "$LOGS/run.log"
  exit 1
fi
echo "$(stamp) all 10 critics trained; evaluating on CPU 0" | tee -a "$LOGS/run.log"
if taskset -c 0 "$PY" "$HERE/evaluate.py" $EVAL_EXTRA > "$LOGS/evaluate.log" 2>&1; then
  echo "$(stamp) evaluation done: $OUT/results.md" | tee -a "$LOGS/run.log"
  tail -n 1 "$LOGS/evaluate.log" | tee -a "$LOGS/run.log"
else
  echo "$(stamp) evaluation FAILED: see $LOGS/evaluate.log" | tee -a "$LOGS/run.log"
  exit 1
fi

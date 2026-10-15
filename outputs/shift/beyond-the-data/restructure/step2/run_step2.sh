#!/bin/sh
# Step 2 of the restructured BCA (the policy pilot). EXPLORATORY; see PRESPEC.md and prespec.json.
# 1. runs the unit tests;
# 2. training: for each seed 0-4, the stalling world, then (after its lock) the no-stall world, over STEP2_LANES
#    parallel lanes (default 4), every process pinned to ONE CPU with taskset (one CPU per physical core:
#    STEP2_CPUS, default 0 2 4 ...); a lane starts a phase only when at least STEP2_MIN_FREE_MB of memory is
#    available (default 4000 MB), and the lanes start STEP2_STAGGER_S seconds apart (default 60);
# 3. once every seed-world is locked or stopped: the 10 evaluations over the same lanes;
# 4. the pooled look on one CPU: results.json and results.md.
# Writes only under runs/shift_explore/restructure/step2/ (policy/, logs/, results.json, results.md, run.lock).
# Write-once: a phase whose lock.json, stop.json or values.json exists is skipped, never redone; the Python scripts
# refuse to overwrite anything.
# Usage (unattended). Launch it DETACHED so that closing the session or a tool timeout cannot kill it half-way:
#   setsid nohup sh /home/user/bca/outputs/shift/beyond-the-data/restructure/step2/run_step2.sh </dev/null >/dev/null 2>&1 &
#   On the laptop (WSL), Ubuntu stops when nothing runs in it: keep one Ubuntu window open, or start it in tmux:
#   tmux new -d -s step2 'sh /home/user/bca/outputs/shift/beyond-the-data/restructure/step2/run_step2.sh'
#   Progress: runs/shift_explore/restructure/step2/logs/run.log (one line per start / done / STOPPED / FAILED);
#   the verdict is the last line of that file and the heading of results.md.
# Smoke test only: STEP2_SCRATCH=<folder outside the repository> [STEP2_STEPS=40] [STEP2_CHUNK=20] [STEP2_SEEDS="0"]
#   sh run_step2.sh  runs the same steps with tiny trainings into that folder (the Python scripts refuse a scratch
#   folder inside the repository); it never touches the official folder.
set -u
HERE=$(cd "$(dirname "$0")" && pwd)
PY=${PY:-/root/bcavenv/bin/python}
LANES=${STEP2_LANES:-4}
CPUS=${STEP2_CPUS:-"0 2 4 6 8 10 12 14 16 18 20 22 24 26 28 30"}
MIN_FREE_MB=${STEP2_MIN_FREE_MB:-4000}
STAGGER=${STEP2_STAGGER_S:-60}
if [ -n "${STEP2_SCRATCH:-}" ]; then
  case "$STEP2_SCRATCH" in
    /home/user/bca|/home/user/bca/*) echo "refused: STEP2_SCRATCH must lie outside the repository"; exit 1 ;;
    /*) ;;
    *) echo "refused: STEP2_SCRATCH must be an absolute path"; exit 1 ;;
  esac
  OUT=$STEP2_SCRATCH
  SEEDS=${STEP2_SEEDS:-"0 1 2 3 4"}
  TRAIN_EXTRA="--out-root $OUT --steps ${STEP2_STEPS:-40} --chunk ${STEP2_CHUNK:-20}"
  EVAL_EXTRA="--out-root $OUT"
  POOL_EXTRA="--out-root $OUT --seeds $SEEDS"
else
  if [ -n "${STEP2_SEEDS:-}${STEP2_STEPS:-}${STEP2_CHUNK:-}" ]; then
    echo "refused: STEP2_SEEDS, STEP2_STEPS and STEP2_CHUNK are for scratch runs only (STEP2_SCRATCH)"; exit 1
  fi
  OUT=/home/user/bca/runs/shift_explore/restructure/step2
  SEEDS="0 1 2 3 4"
  TRAIN_EXTRA=""
  EVAL_EXTRA=""
  POOL_EXTRA=""
fi
n_cpus=$(echo $CPUS | wc -w)
if [ "$LANES" -lt 1 ] || [ "$LANES" -gt "$n_cpus" ]; then
  echo "refused: STEP2_LANES must be between 1 and the number of CPUs in STEP2_CPUS ($n_cpus)"; exit 1
fi
LOGS=$OUT/logs
export PYTHONPATH=/home/user/bca PYTHONDONTWRITEBYTECODE=1 JAX_PLATFORMS=cpu OMP_NUM_THREADS=1
mkdir -p "$LOGS"
LOCK=$OUT/run.lock
if ! mkdir "$LOCK" 2>/dev/null; then
  echo "refused: $LOCK exists (another run is going, or a previous run was killed: check with ps, then remove it)"
  exit 1
fi
CLAIMS=$LOCK/claims
mkdir -p "$CLAIMS"
RUNLOG=$LOGS/run.log
stamp() { date -u +%Y-%m-%dT%H:%M:%SZ; }
say() { echo "$(stamp) $*" | tee -a "$RUNLOG"; }
# Normal end: release the lock. Killed by a signal: KEEP the lock, because phases already started may still be
# running, and a second launch could start duplicate trainings into the same write-once folders.
trap 'rm -rf "$LOCK" 2>/dev/null' EXIT
trap 'say "interrupted by a signal: $LOCK is kept; phases already started may still be running (check with ps); when none is, remove $LOCK and rerun this script (finished phases are skipped)"; trap - EXIT; exit 130' INT TERM HUP
say "step 2 start: seeds $SEEDS, $LANES lanes on CPUs $(echo $CPUS | cut -d' ' -f1-$LANES), at least $MIN_FREE_MB MB free per phase start; output $OUT"

if ! "$PY" "$HERE/test_step2.py" > "$LOGS/tests.log" 2>&1; then
  say "unit tests failed: see $LOGS/tests.log"
  exit 1
fi
say "unit tests passed ($(grep -c ' ok$' "$LOGS/tests.log") tests)"

cpu_of_lane() { echo $CPUS | awk -v i=$(($1 + 1)) '{print $i}'; }
avail_mb() { awk '/^MemAvailable:/ {print int($2 / 1024)}' /proc/meminfo; }
wait_memory() {  # $1 = what is waiting
  n=0
  while [ "$(avail_mb)" -lt "$MIN_FREE_MB" ]; do
    if [ $((n % 10)) -eq 0 ]; then
      echo "$(stamp) waiting for memory before $1: $(avail_mb) MB available, $MIN_FREE_MB MB wanted" >> "$RUNLOG"
    fi
    n=$((n + 1))
    sleep 30
  done
}
done_or_stopped() { [ -f "$1/lock.json" ] || [ -f "$1/stop.json" ]; }

train_seed() {  # $1 = CPU, $2 = seed: the stalling world, then the no-stall world
  cpu=$1; s=$2
  for w in stall no_stall; do
    d=$OUT/policy/seed$s/$w
    if done_or_stopped "$d"; then echo "$(stamp) skip train $w/seed$s (lock or stop exists)" >> "$RUNLOG"; continue; fi
    wait_memory "train $w/seed$s"
    echo "$(stamp) start train $w/seed$s on CPU $cpu ($(avail_mb) MB available)" >> "$RUNLOG"
    log=$LOGS/train_${w}_seed${s}.log
    if taskset -c "$cpu" "$PY" "$HERE/train_policy.py" --seed "$s" --world "$w" $TRAIN_EXTRA >> "$log" 2>&1; then
      echo "$(stamp) done train $w/seed$s: $(tail -n 1 "$log")" >> "$RUNLOG"
    elif [ -f "$d/stop.json" ]; then
      echo "$(stamp) STOPPED train $w/seed$s (see $d/stop.json and $log)" >> "$RUNLOG"
    else
      echo "$(stamp) FAILED train $w/seed$s (see $log)" >> "$RUNLOG"
      return 1
    fi
  done
}

lane_train() {  # $1 = lane index
  cpu=$(cpu_of_lane "$1")
  sleep $(($1 * STAGGER))
  for s in $SEEDS; do
    mkdir "$CLAIMS/train_seed$s" 2>/dev/null || continue
    train_seed "$cpu" "$s"
  done
}

lane_eval() {  # $1 = lane index
  cpu=$(cpu_of_lane "$1")
  sleep $(($1 * 5))
  for s in $SEEDS; do for w in stall no_stall; do
    mkdir "$CLAIMS/eval_${w}_seed$s" 2>/dev/null || continue
    d=$OUT/policy/seed$s/$w
    if [ -f "$d/values.json" ]; then echo "$(stamp) skip eval $w/seed$s (values exist)" >> "$RUNLOG"; continue; fi
    if [ -f "$d/stop.json" ]; then echo "$(stamp) skip eval $w/seed$s (stopped)" >> "$RUNLOG"; continue; fi
    wait_memory "eval $w/seed$s"
    echo "$(stamp) start eval $w/seed$s on CPU $cpu" >> "$RUNLOG"
    log=$LOGS/eval_${w}_seed${s}.log
    if taskset -c "$cpu" "$PY" "$HERE/evaluate_policy.py" --seed "$s" --world "$w" $EVAL_EXTRA >> "$log" 2>&1; then
      echo "$(stamp) done eval $w/seed$s: $(tail -n 1 "$log")" >> "$RUNLOG"
    else
      echo "$(stamp) FAILED eval $w/seed$s (see $log)" >> "$RUNLOG"
    fi
  done; done
}

i=0
while [ "$i" -lt "$LANES" ]; do lane_train "$i" & i=$((i + 1)); done
wait

missing=""
for s in $SEEDS; do for w in stall no_stall; do
  done_or_stopped "$OUT/policy/seed$s/$w" || missing="$missing $w/seed$s"
done; done
if [ -n "$missing" ]; then
  say "not evaluated: no lock or stop for$missing (an interrupted phase can be rerun with this script)"
  exit 1
fi
say "every seed-world is locked or stopped; evaluating"

i=0
while [ "$i" -lt "$LANES" ]; do lane_eval "$i" & i=$((i + 1)); done
wait

missing=""
for s in $SEEDS; do for w in stall no_stall; do
  d=$OUT/policy/seed$s/$w
  [ -f "$d/values.json" ] || [ -f "$d/stop.json" ] || missing="$missing $w/seed$s"
done; done
if [ -n "$missing" ]; then
  say "pooled look not run: no values for$missing (see the eval logs; rerun this script)"
  exit 1
fi
say "all evaluations done; pooled look on CPU $(cpu_of_lane 0)"
if taskset -c "$(cpu_of_lane 0)" "$PY" "$HERE/pooled.py" $POOL_EXTRA > "$LOGS/pooled.log" 2>&1; then
  say "pooled look done: $OUT/results.md"
  tail -n 1 "$LOGS/pooled.log" | tee -a "$RUNLOG"
else
  say "pooled look FAILED: see $LOGS/pooled.log"
  exit 1
fi

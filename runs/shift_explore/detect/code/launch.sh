#!/bin/sh
# EXPLORATORY detection rehearsal (plan 4.4), no BCA. Runs ONLY after the room test's pooled look passes.
# One seed (or the pooled phase), one low-priority process pinned to one CPU, phases in order, each phase its own
# process (the label script never shares a process with signal code). Every phase refuses unless the spec, the gate
# files, the room pins and the code manifest check out.
# It runs the PINNED COPIES in /home/user/bca/runs/shift_explore/detect/code (the code manifest's copies), never the
# scratchpad sources, so it works if the scratchpad is cleared.
# Usage: sh /home/user/bca/runs/shift_explore/detect/code/launch.sh <seed|all> <cpu> <stall|no_stall|both> <phase> [...]
#   e.g. sh /home/user/bca/runs/shift_explore/detect/code/launch.sh 0 3 both pairs signals labels measures
#        sh /home/user/bca/runs/shift_explore/detect/code/launch.sh all 3 both pooled
# Logs: /home/user/bca/runs/shift_explore/detect/logs/seed<k>.log (pooled.log for the pooled phase). Each log starts
# with the EXPLORATORY label; its .sha256 sidecar is rewritten when each phase ends (done or failed).
# Only the seed-0 check (checks_seed0.py) sets DETECT_CHECK_OUT (a folder under runs/shift_explore/detect, passed as
# --out), DETECT_LOGDIR (a folder under runs/shift_explore/detect) and DETECT_CHECK_SIGNALS (passed as --signals to
# the signals phase); the real run leaves all three unset (every Stage 1 signal is scored).
OUTROOT=/home/user/bca/runs/shift_explore/detect
D=$OUTROOT/code
LOGDIR=${DETECT_LOGDIR:-$OUTROOT/logs}
LABEL="EXPLORATORY: log of the Stage 1 detection rehearsal (plan section 4.4); outside the gate; not a gated output; no BCA band, calibrator, threshold, dose or BCA update"
case "$LOGDIR" in "$OUTROOT"/*) ;; *) echo "refused: DETECT_LOGDIR must lie under $OUTROOT" >&2; exit 1 ;; esac
[ -f "$D/run_detect.py" ] || { echo "refused: $D/run_detect.py is missing (write the code manifest first)" >&2; exit 1; }
[ $# -ge 4 ] || { echo "usage: sh $D/launch.sh <seed|all> <cpu> <stall|no_stall|both> <phase> [<phase> ...]" >&2; exit 1; }
SEED=$1; CPU=$2; WORLD=$3; shift 3
OUTARG=""
if [ -n "$DETECT_CHECK_OUT" ]; then OUTARG="--out $DETECT_CHECK_OUT"; fi
mkdir -p "$LOGDIR"
cd /home/user/bca || exit 1
export PYTHONPATH=/home/user/bca PYTHONDONTWRITEBYTECODE=1 JAX_PLATFORMS=cpu OMP_NUM_THREADS=1
if [ "$SEED" = all ]; then NAME="pooled.log"; SARG=""; else NAME="seed${SEED}.log"; SARG="--seed $SEED"; fi
LOG="$LOGDIR/$NAME"
[ -s "$LOG" ] || echo "$LABEL" > "$LOG"
sidecar() { (cd "$LOGDIR" && sha256sum "$NAME" > "$NAME.sha256"); }
for PHASE in "$@"; do
  SIGARG=""
  if [ "$PHASE" = signals ] && [ -n "$DETECT_CHECK_SIGNALS" ]; then SIGARG="--signals $DETECT_CHECK_SIGNALS"; fi
  echo "$(date -u +%FT%TZ) EXPLORATORY seed $SEED world $WORLD phase $PHASE start (cpu $CPU, code $D)" >> "$LOG"
  # shellcheck disable=SC2086
  nice -n 15 taskset -c "$CPU" /tmp/claude-0/bcavenv/bin/python "$D/run_detect.py" $SARG --world "$WORLD" \
    --phase "$PHASE" $OUTARG $SIGARG >> "$LOG" 2>&1 || { echo "$(date -u +%FT%TZ) seed $SEED phase $PHASE FAILED" >> "$LOG"; sidecar; exit 1; }
  echo "$(date -u +%FT%TZ) seed $SEED world $WORLD phase $PHASE done" >> "$LOG"
  sidecar
done

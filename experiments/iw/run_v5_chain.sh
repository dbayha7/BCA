#!/bin/bash
# Run the approved version-5 stages in order (no BCA arms). Stops at the first stage that exits non-zero.
set -u
cd /root/bca-shift || exit 1
LOG=/root/bca-shift/runs/shift_iw_v5/chain.log
mkdir -p "$(dirname "$LOG")"
CPUS="4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23"
SEEDS="0 1 2 3 4"
run() {
  echo "$(date -u +%FT%TZ) START $*" >> "$LOG"
  bash experiments/shift_iw/launch.sh "$@" >> "$LOG" 2>&1
  rc=$?
  echo "$(date -u +%FT%TZ) END rc=$rc $*" >> "$LOG"
  if [ $rc -ne 0 ]; then echo "$(date -u +%FT%TZ) CHAIN STOPPED at: $*" >> "$LOG"; exit $rc; fi
}
run data "$SEEDS" "$CPUS"
run host "$SEEDS" "$CPUS"
run room "$SEEDS" "$CPUS"
run room-lower-edge "$SEEDS" "$CPUS"
run evaluate "$SEEDS" "$CPUS"
run report "$SEEDS" "4 5 6 7 8"
run band "$SEEDS" "$CPUS"
run coverage "$SEEDS" "$CPUS"
run report "$SEEDS" "4 5 6 7 8"
run summary "0" "4"
echo "$(date -u +%FT%TZ) CHAIN FINISHED" >> "$LOG"

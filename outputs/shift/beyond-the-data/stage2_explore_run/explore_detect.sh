#!/bin/sh
# Stage 2, seeds 0-4, the first decisive look (pre-registration section 10, steps 1-2, uniform version):
# select_standin, then per seed the no-BCA verification of the rehearsal's pairs, labels and stand-in scores,
# the three band fits and the three width scores, then the exploratory detection marks.
# Every phase goes through the pinned launch.sh unchanged (one pinned, low-priority process each); at most 2 at once.
# A failed phase ends that seed's chain (the pinned code records or refuses; nothing is retried or deleted).
L=/tmp/claude-0/-home-user/9e341792-006d-5a11-a8e9-8fac16ac7f5b/scratchpad/ood/stage2/launch.sh
LOG=/home/user/bca/runs/shift_explore/stage2/logs/explore_detect.log
mkdir -p "$(dirname "$LOG")"
say() { echo "$(date -u +%FT%TZ) $*" >> "$LOG"; }
run() { cpu=$1; shift; say "start cpu $cpu: $*"; if sh "$L" "$cpu" "$@"; then say "done: $*"; return 0; else say "FAILED: $*"; return 1; fi; }
seed_chain() {
  cpu=$1; K=$2
  for W in stall no_stall; do
    run $cpu --phase detect_pairs --seed $K --world $W || return 1
    run $cpu --phase detect_labels --seed $K --world $W || return 1
    run $cpu --phase detect_standins --seed $K --world $W || return 1
  done
  run $cpu --phase detect_fit --seed $K --world stall --critic fixed || return 1
  run $cpu --phase detect_fit --seed $K --world stall --critic host || return 1
  run $cpu --phase detect_fit --seed $K --world no_stall --critic fixed || return 1
  run $cpu --phase detect_score --seed $K --world stall --critic fixed --version uniform || return 1
  run $cpu --phase detect_score --seed $K --world stall --critic host --version uniform || return 1
  run $cpu --phase detect_score --seed $K --world no_stall --critic fixed --version uniform || return 1
  say "seed $K chain finished"
}
lane() { cpu=$1; shift; for K in "$@"; do seed_chain $cpu $K || say "seed $K chain stopped (see its phase log)"; done; say "lane cpu $cpu finished"; }
say "explore_detect start"
run 2 --phase select_standin || { say "select_standin failed: stopping"; exit 1; }
lane 2 0 2 4 & A=$!
lane 3 1 3 & B=$!
wait $A; wait $B
run 2 --phase detect_marks --group exploratory --version uniform
say "explore_detect finished"

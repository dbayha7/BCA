#!/bin/sh
# Stage 2, seeds 0-4: the declared weighted version (pre-registration section 10 step 2: run only because the uniform
# version failed direction or size; the marks file says "weighted version declared"). Pinned launch.sh, 2 lanes.
L=/tmp/claude-0/-home-user/9e341792-006d-5a11-a8e9-8fac16ac7f5b/scratchpad/ood/stage2/launch.sh
LOG=/home/user/bca/runs/shift_explore/stage2/logs/explore_weighted.log
say() { echo "$(date -u +%FT%TZ) $*" >> "$LOG"; }
run() { cpu=$1; shift; say "start cpu $cpu: $*"; if sh "$L" "$cpu" "$@"; then say "done: $*"; return 0; else say "FAILED: $*"; return 1; fi; }
seed_chain() {
  cpu=$1; K=$2
  for W in stall no_stall; do run $cpu --phase detect_fit_weighted --seed $K --world $W --critic fixed || return 1; done
  for W in stall no_stall; do run $cpu --phase detect_score --seed $K --world $W --critic fixed --version weighted || return 1; done
  say "seed $K weighted chain finished"
}
lane() { cpu=$1; shift; for K in "$@"; do seed_chain $cpu $K || say "seed $K chain stopped"; done; say "lane cpu $cpu finished"; }
say "explore_weighted start"
lane 2 0 2 4 & A=$!
lane 3 1 3 & B=$!
wait $A; wait $B
run 2 --phase detect_marks --group exploratory --version weighted
say "explore_weighted finished"

#!/bin/sh
# Stage 2, seeds 0-4, after the weighted version passed detection direction and size (pre-registration section 10
# step 3). Resumable: a phase whose output file already exists is skipped (every phase writes once). Per seed: the
# policy test (stalling world, then the no-stall world unless the seed stopped), the deployment check (both worlds,
# after the policy lock, as the confirmatory runner does), the full host (both worlds); then the exploratory pooled
# look. Pinned launch.sh unchanged, 2 lanes. A lock file stops a second copy from starting.
L=/tmp/claude-0/-home-user/9e341792-006d-5a11-a8e9-8fac16ac7f5b/scratchpad/ood/stage2/launch.sh
OUT=/home/user/bca/runs/shift_explore/stage2
LOG=$OUT/logs/explore_policy.log
LOCK=$OUT/logs/explore_policy.lock
say() { echo "$(date -u +%FT%TZ) $*" >> "$LOG"; }
exec 9>"$LOCK"; flock -n 9 || { say "another explore_policy is running; exiting"; exit 0; }
run() { cpu=$1; shift; say "start cpu $cpu: $*"; if sh "$L" "$cpu" "$@"; then say "done: $*"; return 0; else say "FAILED: $*"; return 1; fi; }
have() { for f in "$@"; do [ -e "$f" ] && return 0; done; return 1; }
step() { # step <cpu> <done-file...> -- <phase args...>
  cpu=$1; shift; files=""
  while [ "$1" != "--" ]; do files="$files $1"; shift; done; shift
  if have $files; then say "skip (done): $*"; return 0; fi
  run $cpu "$@"
}
seed_chain() {
  cpu=$1; K=$2; P=$OUT/policy/seed$K
  step $cpu $P/lock.json $P/stop.json -- --phase policy_train --seed $K || return 1
  step $cpu $P/values.json $P/stop.json -- --phase policy_eval --seed $K || return 1
  if [ -e "$P/stop.json" ]; then
    say "seed $K: the stalling world's policy test stopped this seed (stop.json); no no-stall policy arm"
  else
    step $cpu $P/no_stall/lock.json $P/no_stall/stop.json -- --phase policy_no_stall_train --seed $K || return 1
    step $cpu $P/no_stall/values.json $P/no_stall/stop.json -- --phase policy_no_stall_eval --seed $K || return 1
  fi
  for W in stall no_stall; do step $cpu $OUT/deploy/$W/seed$K/values.json -- --phase deploy --seed $K --world $W || return 1; done
  for W in stall no_stall; do
    step $cpu $OUT/host/$W/seed$K/lock.json -- --phase host_train --seed $K --world $W || return 1
    step $cpu $OUT/host/$W/seed$K/values.json -- --phase host_eval --seed $K --world $W || return 1
  done
  say "seed $K chain finished"
}
lane() { cpu=$1; shift; for K in "$@"; do seed_chain $cpu $K || say "seed $K chain stopped (see its phase log)"; done; say "lane cpu $cpu finished"; }
say "explore_policy start (resumable)"
lane 2 0 2 4 & A=$!
lane 3 1 3 & B=$!
wait $A; wait $B
step 2 $OUT/pooled_exploratory.json -- --phase pooled --group exploratory
say "explore_policy finished"

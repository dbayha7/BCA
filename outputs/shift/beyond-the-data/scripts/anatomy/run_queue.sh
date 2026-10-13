#!/bin/sh
# EXPLORATORY anatomy runs. Usage: run_queue.sh <cpu> <script> <seeds...> [-- extra args]
D=/tmp/claude-0/-home-user/9e341792-006d-5a11-a8e9-8fac16ac7f5b/scratchpad/ood/anatomy
cpu=$1; shift
script=$1; shift
extra=""
for s in "$@"; do
  case "$s" in --*) extra="$extra $s";; esac
done
cd /home/user/bca || exit 1
for s in "$@"; do
  case "$s" in --*) continue;; esac
  base=$(basename "$script" .py)
  echo "start $base seed $s $(date +%T)"
  PYTHONPATH=/home/user/bca PYTHONDONTWRITEBYTECODE=1 JAX_PLATFORMS=cpu OMP_NUM_THREADS=1 \
    taskset -c "$cpu" nice -n 10 /tmp/claude-0/bcavenv/bin/python "$D/$script" "$s" $extra > "$D/log_${base}_$s.txt" 2>&1
  echo "end $base seed $s rc=$? $(date +%T)"
done

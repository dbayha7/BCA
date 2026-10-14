#!/bin/sh
# read-only forward evaluation; pinned to 2 CPUs at low priority
cd /tmp/claude-0/-home-user/9e341792-006d-5a11-a8e9-8fac16ac7f5b/scratchpad/restructure/consumer || exit 1
export PYTHONPATH=/home/user/bca PYTHONDONTWRITEBYTECODE=1 JAX_PLATFORMS=cpu OMP_NUM_THREADS=1
exec taskset -c 0,1 nice -n 10 /tmp/claude-0/bcavenv/bin/python "$@"

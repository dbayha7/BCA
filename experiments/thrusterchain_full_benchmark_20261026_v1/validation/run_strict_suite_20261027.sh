#!/usr/bin/env bash
set -u -o pipefail
cd /mnt/c/Users/David/dev/BCA/experiments/thrusterchain_full_benchmark_20261026_v1
export PYTHONPATH=.:../thrusterchain_unifloral_gpu_compat_v1:../..
export JAX_PLATFORMS=cuda JAX_DEFAULT_MATMUL_PRECISION=highest XLA_PYTHON_CLIENT_PREALLOCATE=false PYTHONDONTWRITEBYTECODE=1
log=validation/strict_suite_20261027.log
statusfile=validation/strict_suite_20261027.status
test ! -e "$log" && test ! -e "$statusfile" || exit 3
status=0
/home/david/.cache/thrusterchain-unifloral-gpu-v1/bin/python -B -m unittest compiled.test_engine -v > "$log" 2>&1 || status=$?
printf '%s\n' "$status" > "$statusfile"
cat "$log"
exit "$status"

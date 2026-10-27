#!/usr/bin/env bash
set -euo pipefail
cd /mnt/c/Users/David/dev/BCA/experiments/thrusterchain_full_benchmark_20261026_v1
export JAX_PLATFORMS=cuda JAX_DEFAULT_MATMUL_PRECISION=highest XLA_PYTHON_CLIENT_PREALLOCATE=false PYTHONDONTWRITEBYTECODE=1
python=/home/david/.cache/thrusterchain-unifloral-gpu-v1/bin/python
study="$PWD/validation/smoke_20261026_v2"
test ! -e "$study"
for stage in prepare collect train evaluate report status; do
  test ! -e "validation/smoke_v2_${stage}.log"
  if [[ "$stage" == prepare ]]; then
    "$python" -B study.py "$stage" --spec validation/smoke_spec.json --study "$study" 2>&1 | tee "validation/smoke_v2_${stage}.log"
  else
    "$python" -B study.py "$stage" --study "$study" 2>&1 | tee "validation/smoke_v2_${stage}.log"
  fi
done

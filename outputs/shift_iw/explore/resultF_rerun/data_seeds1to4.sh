#!/bin/bash
# Result F rerun, step 1 (owner's go in chat, 16 Oct: "yes for result f"): build data seeds 1-4 of medium-expert and
# expert in the iw_rerun sandbox, then the report per quality, so the band stage's report gate (D1 on seeds 0-4) can be
# met with the sandbox's own records. Data only: no training, no band fit. EXPLORATORY.
set -euo pipefail
IW=/tmp/claude-0/-home-user/9e341792-006d-5a11-a8e9-8fac16ac7f5b/scratchpad/iw_rerun; REPO=$IW/repo; PY=/tmp/claude-0/bcavenv/bin/python3
NPY_OFF="AVX512F AVX512CD AVX512_KNL AVX512_KNM AVX512_SKX AVX512_CLX AVX512_CNL AVX512_ICL AVX512_SPR"
run() { (cd "$REPO" && env OPENBLAS_CORETYPE=Haswell NPY_DISABLE_CPU_FEATURES="$NPY_OFF" XLA_FLAGS=--xla_cpu_max_isa=AVX2   PYTHONPATH="$REPO:$IW/explore" PYTHONDONTWRITEBYTECODE=1 JAX_PLATFORMS=cpu OMP_NUM_THREADS=1 nice -n 10 "$PY" -B "$@"); }
for Q in medium-expert expert; do
  ( for k in 1 2 3 4; do run -m experiments.shift_iw.run data --quality $Q --seeds $k; done ) > /tmp/claude-0/-home-user/9e341792-006d-5a11-a8e9-8fac16ac7f5b/scratchpad/resultF/logs/data_$Q.log 2>&1 &
done
wait
for Q in medium-expert expert; do run -m experiments.shift_iw.run report --quality $Q > /tmp/claude-0/-home-user/9e341792-006d-5a11-a8e9-8fac16ac7f5b/scratchpad/resultF/logs/report_$Q.log 2>&1; done
echo DONE

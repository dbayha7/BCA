#!/bin/sh
# Information-only XLA evidence for Stage 2 v7 (V7_SPEC.md C1 "Still to do", C7 "Cloud evidence", C7 step 2b).
# NOT v7 results: run on the cloud's Xeon outside the v7 wrapper, before anything is pinned.
# One run at a time, each on processor 0 at low priority (taskset -c 0 nice -n 10), so the cloud's v6 run keeps
# processors 2 and 3 and another agent keeps processor 1. Each run gets launch_v7.sh's eight variables (env -i) and,
# in emulated mode, the three emulation variables (xla_emul_common.EMULATION). Writes only under this folder and
# under the scratchpad folder v7_build2/ (model files); nothing under runs/.
# Usage: sh run_xla.sh <job> [<job> ...]
#   repro_<seed>_<default|emulated>              xla_repro.py for one gate seed
#   feas_<stall|no_stall|host>                     the feasibility review's run for seed 2, repeated on processor 0
#   fc_p1_<stall|no_stall>_<default|emulated>      forecast_seed2.py p1 (P1a, P1b, P1c) for seed 2
#   fc_p2_<default|emulated>                       forecast_seed2.py p2 for seed 2
#   dump_<default|emulated>                        xla_dump_check.py: does the cap change the training program's code?
set -u
ROOT=/home/user/bca
X=$ROOT/outputs/shift/beyond-the-data/stage2_v7/evidence/xla
S=/tmp/claude-0/-home-user/9e341792-006d-5a11-a8e9-8fac16ac7f5b/scratchpad/v7_build2
PY=/tmp/claude-0/bcavenv/bin/python
NPYOFF="AVX512F AVX512CD AVX512_SKX AVX512_CLX AVX512_CNL AVX512_ICL AVX512_SPR"
cd "$ROOT" || exit 1
mkdir -p "$X/logs" "$X/repro" "$X/forecast_seed2" "$S"

run() {  # run <name> <mode> <script> <args...>
  name=$1 mode=$2; shift 2
  echo "$(date -u +%FT%TZ) start $name ($mode, cpu 0)" >> "$X/driver.txt"
  if [ "$mode" = emulated ]; then
    taskset -c 0 nice -n 10 env -i PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin HOME=/root \
      LANG=C.UTF-8 LC_ALL=C.UTF-8 PYTHONPATH=$ROOT PYTHONDONTWRITEBYTECODE=1 JAX_PLATFORMS=cpu OMP_NUM_THREADS=1 \
      XLA_FLAGS=--xla_cpu_max_isa=AVX2 OPENBLAS_CORETYPE=Haswell NPY_DISABLE_CPU_FEATURES="$NPYOFF" \
      "$PY" "$@" > "$X/logs/$name.txt" 2>&1
  else
    taskset -c 0 nice -n 10 env -i PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin HOME=/root \
      LANG=C.UTF-8 LC_ALL=C.UTF-8 PYTHONPATH=$ROOT PYTHONDONTWRITEBYTECODE=1 JAX_PLATFORMS=cpu OMP_NUM_THREADS=1 \
      "$PY" "$@" > "$X/logs/$name.txt" 2>&1
  fi
  code=$?
  echo "$(date -u +%FT%TZ) end $name exit $code" >> "$X/driver.txt"
}

for job in "$@"; do
  case $job in
    repro_*_*)
      rest=${job#repro_}; seed=${rest%_*}; mode=${rest#*_}
      run "$job" "$mode" "$X/xla_repro.py" --seed "$seed" --mode "$mode" --out "$X/repro/seed${seed}_${mode}.json" ;;
    feas_*)
      what=${job#feas_}
      run "$job" default "$X/feas_review_seed2/cpu0_repeat/p_probe_cpu0.py" "$what" 2 --outdir "$S/feas_cpu0" \
        --record "$X/feas_review_seed2/cpu0_repeat/result_${what}_seed2.json" ;;
    fc_p1_*)
      rest=${job#fc_p1_}; mode=${rest##*_}; world=${rest%_*}
      run "$job" "$mode" "$X/forecast_seed2.py" p1 --world "$world" --mode "$mode" --r7 "$S/r7_$mode" \
        --outdir "$X/forecast_seed2/$mode" ;;
    fc_p2_*)
      mode=${job#fc_p2_}
      run "$job" "$mode" "$X/forecast_seed2.py" p2 --mode "$mode" --r7 "$S/r7_$mode" --outdir "$X/forecast_seed2/$mode" ;;
    dump_*)
      mode=${job#dump_}
      mkdir -p "$X/dump_check"
      run "$job" "$mode" "$X/xla_dump_check.py" --mode "$mode" --dump "$S/xladump_$mode" --out "$X/dump_check/$mode.json" ;;
    *) echo "$(date -u +%FT%TZ) unknown job $job" >> "$X/driver.txt" ;;
  esac
done
echo "$(date -u +%FT%TZ) jobs done: $*" >> "$X/driver.txt"

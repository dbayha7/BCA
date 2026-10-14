#!/bin/sh
# Stage 2 v7 evidence (V7_SPEC C1 "Evidence status", "Tested by (b)"): rerun pinned_constants.py, saving each JSON,
# for the default kernels and for each OpenBLAS kernel family the laptop could use, with numpy's AVX-512 dispatch off
# (the NPY_DISABLE_CPU_FEATURES string of the earlier probe scratchpad/v7_hw_probe/haswell_npyoff.json).
#   pinned:   the constants from world_constants.json substituted; gate load_prereg + guard, then load_seed for gate
#             seeds 0-4 in both worlds.
#   unpinned: nothing substituted (information: how many anchors and logs differ on that kernel; ST5's number);
#             load_seed runs even when the gate refuses (--seeds-anyway).
# Every process gets launch_v7.sh's eight variables (env -i) plus the probe's own, one processor (0 or 1, so the cloud's
# v6 run keeps processors 2 and 3) and nice 10. Writes only evidence/probe_*.json and evidence/probe_*.stderr.txt.
# Usage: taskset -c 0,1 nice -n 10 sh run_probes.sh
set -u
ROOT=/home/user/bca
EV=$ROOT/outputs/shift/beyond-the-data/stage2_v7/evidence
PY=/tmp/claude-0/bcavenv/bin/python
SHA=$(cut -d' ' -f1 "$ROOT/outputs/shift/beyond-the-data/stage2_v7/world_constants.json.sha256")
NPYOFF="AVX512F AVX512CD AVX512_SKX AVX512_CLX AVX512_CNL AVX512_ICL AVX512_SPR"
cd "$ROOT" || exit 1

one() {  # one <cpu> <family> <pinned|unpinned>
  cpu=$1 fam=$2 mode=$3
  name=probe_${fam}_${mode}
  if [ "$mode" = pinned ]; then flags=""; else flags="--no-pin --seeds-anyway"; fi
  echo "$(date -u +%FT%TZ) start $name (cpu $cpu)"
  if [ "$fam" = default ]; then
    env -i PATH=/usr/bin:/bin HOME=/root LANG=C.UTF-8 LC_ALL=C.UTF-8 PYTHONPATH=$ROOT PYTHONDONTWRITEBYTECODE=1 \
      JAX_PLATFORMS=cpu OMP_NUM_THREADS=1 \
      taskset -c "$cpu" "$PY" "$EV/pinned_constants.py" --out "$EV/$name.json" --constants-sha256 "$SHA" \
      --label "$name" $flags 2> "$EV/$name.stderr.txt"
  else
    env -i PATH=/usr/bin:/bin HOME=/root LANG=C.UTF-8 LC_ALL=C.UTF-8 PYTHONPATH=$ROOT PYTHONDONTWRITEBYTECODE=1 \
      JAX_PLATFORMS=cpu OMP_NUM_THREADS=1 OPENBLAS_CORETYPE="$fam" NPY_DISABLE_CPU_FEATURES="$NPYOFF" \
      taskset -c "$cpu" "$PY" "$EV/pinned_constants.py" --out "$EV/$name.json" --constants-sha256 "$SHA" \
      --label "$name" $flags 2> "$EV/$name.stderr.txt"
  fi
  echo "$(date -u +%FT%TZ) end $name exit $?"
}

lane() {  # lane <cpu> <mode>
  for fam in default Haswell Sandybridge Nehalem Katmai; do one "$1" "$fam" "$2"; done
}

lane 0 pinned &
lane 1 unpinned &
wait
echo "$(date -u +%FT%TZ) all done"

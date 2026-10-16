#!/bin/sh
# Re-run the toy notebooks (copies in RUN_DIR, made by get_sources.sh) with run_nb.py, one kernel at a time, and the
# three copies that go beyond the stored outputs: the discrete notebook with RUN_SWEEP = True (its only change) and the
# two copies with one information-only cell added (make_extra.py). Needs Python with numpy, matplotlib, jupyter_client
# and ipykernel (PYTHON). Outputs: RUN_DIR/out/*.json. Usage: PYTHON=... sh run_all.sh RUN_DIR
set -e
HERE=$(cd "$(dirname "$0")" && pwd)
RUN=$(cd "${1:?usage: run_all.sh RUN_DIR}" && pwd)
P=${PYTHON:-python3}
"$P" - "$RUN" <<'PY'
import json, sys
run = sys.argv[1]
d = json.load(open(run + "/bca_discrete_toy.ipynb", encoding="utf-8"))
j = "".join(d["cells"][1]["source"])
assert j.count("RUN_SWEEP = False") == 1
d["cells"][1]["source"] = [j.replace("RUN_SWEEP = False", "RUN_SWEEP = True")]
json.dump(d, open(run + "/bca_discrete_toy_sweep.ipynb", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
PY
"$P" "$HERE/make_extra.py" "$RUN" "$RUN"
cd "$RUN"
for pair in discrete:bca_discrete_toy discrete_sweep:bca_discrete_toy_sweep extra_discrete:extra_discrete \
            channel:bca_channel_toy extra_channel:extra_channel mean_gate:bca_mean_gate_toy weights:bca_weights_toy extra_weights:extra_weights \
            findings:bca_findings walkthrough:bca_findings_walkthrough; do
  k=${pair%%:*}; nb=${pair##*:}
  echo "== $k"; "$P" "$HERE/../run_nb.py" "$nb.ipynb" "out/$k.json" "$RUN" 1800 | tail -1
done
echo ALLDONE

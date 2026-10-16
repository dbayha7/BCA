#!/bin/sh
# Extract the committed toy notebooks (and the calibration package the weights check imports) from the repository's
# history into wf7/toy/src and a run folder, checking each notebook's SHA-256 against TOY.md. Read only: git show and
# git archive. Usage (from anywhere inside the repository): sh page_build/wf7/toy/rerun/get_sources.sh RUN_DIR
set -e
ROOT=$(git rev-parse --show-toplevel)
TOY="$ROOT/page_build/wf7/toy"
RUN=${1:?usage: get_sources.sh RUN_DIR}
mkdir -p "$TOY/src" "$RUN/out"
git -C "$ROOT" show 2c8a0e3:bca_discrete_toy.ipynb > "$TOY/src/bca_discrete_toy.ipynb"
for nb in bca_channel_toy bca_mean_gate_toy bca_weights_toy bca_findings; do
  git -C "$ROOT" show 6320224:$nb.ipynb > "$TOY/src/$nb.ipynb"
done
# the lesson version lives on the branch claude/bca-findings-walkthrough (fetch it first if 2467e29 is missing)
git -C "$ROOT" show 2467e29:bca_findings.ipynb > "$TOY/src/bca_findings_walkthrough.ipynb"
( cd "$TOY/src" && sha256sum -c - <<SUMS
bd6b8c927de23faae446c15ba624110b4cb262e838751f40c2b5a1e12505b200  bca_discrete_toy.ipynb
cf1ba0c5f1e09a115c1a903ec0641709add085d244880a9cc91e35d7e6386413  bca_channel_toy.ipynb
06da25617949943f8f4630b49dec130e87abb8a674c9f9fe8cd49fc674967f33  bca_mean_gate_toy.ipynb
73bf93449fe480f47c960745aee69183755ca137aa264ae1e07cb511d4a4e701  bca_weights_toy.ipynb
dbd12d2021c0a0c55679baea6c6e55f6406f63f7eef3494dc132ace6dfe1c7a7  bca_findings.ipynb
b2d8f44d9e51937bec39d07746d021b4f17ff7b100a257f5e3d7088d71e28ca5  bca_findings_walkthrough.ipynb
SUMS
)
cp "$TOY"/src/*.ipynb "$RUN/"
git -C "$ROOT" archive 6320224 calibration | tar -x -C "$RUN"
echo "sources in $TOY/src and $RUN"

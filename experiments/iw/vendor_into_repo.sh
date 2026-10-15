#!/bin/bash
# Copy the weighting-test package, its frozen specification and the result records into /root/bca-shift as real files
# (the package was a link to the Windows work folder). Code and spec bytes are unchanged, so every recorded hash still
# matches. Run results stay out of git except the small report, summary and approval records.
set -euo pipefail
W=/mnt/c/Users/David/dev/bca-shift-work
R=/root/bca-shift
cd "$R"

# 1. The package: replace the link with real files (no wheel, no caches).
test -L experiments/shift_iw
rm experiments/shift_iw
mkdir experiments/shift_iw
for f in "$W"/shift_iw/*.py "$W"/shift_iw/launch.sh; do cp -p "$f" experiments/shift_iw/; done

# 2. The frozen specification, next to the package (SPEC_JSON = experiments/iw/spec.json), and the orchestrator files.
mkdir -p experiments/iw/scout
for f in SPEC.md MARKS_FOR_USER.md spec.json spec.json.sha256 owner_go_bca_arms.txt run_v5_chain.sh write_approval_v5.py \
         write_approval_v5_explore.py results_brief.py results_table.py update_experiments_md.py vendor_into_repo.sh \
         scout_bca_weights.md scout_feasibility.md; do
  cp -p "$W/iw/$f" experiments/iw/
done
cp -rp "$W"/iw/scout/. experiments/iw/scout/
find experiments/iw/scout -name "__pycache__" -prune -exec rm -rf {} +

# 3. Records of the version-5 run (small JSON only) and the exploratory work.
O=outputs/shift_iw/v5
mkdir -p "$O"
cp -p runs/shift_iw_v5/approval.json runs/shift_iw_v5/approval.json.sha256 runs/shift_iw_v5/approval.band_coverage.json.bak "$O"/
cp -p runs/shift_iw_v5/summary/summary.json runs/shift_iw_v5/summary/summary.json.sha256 "$O"/
for q in random medium-replay medium medium-expert expert; do
  mkdir -p "$O/$q"
  cp -p runs/shift_iw_v5/$q/report/report.json runs/shift_iw_v5/$q/report/report.json.sha256 "$O/$q"/
done
E=outputs/shift_iw/explore
mkdir -p "$E/plain_twin"
for f in "$W"/iw_explore/*; do cp -p "$f" "$E"/; done
cp -p runs/shift_iw_v5_explore/plain_twin/*.json "$E/plain_twin"/
echo done

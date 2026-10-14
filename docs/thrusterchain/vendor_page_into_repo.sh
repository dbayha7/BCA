#!/bin/bash
# Copy the ThrusterChain page (version 16), the scripts that build and check it, the key design notes and the
# "critic in action" section's sources into /root/bca-shift/docs/thrusterchain. Screenshots, scratch folders, old
# page copies and node_modules stay in the work folder.
set -euo pipefail
W=/mnt/c/Users/David/dev/bca-shift-work
D=/root/bca-shift/docs/thrusterchain
mkdir -p "$D/scripts" "$D/notes" "$D/qviewer" "$D/mathbuild"
cp -p "$W/page/thrusterchain.html" "$D/"
cp -p "$W/page/vendor_page_into_repo.sh" "$W/page/update_experiments_md_v13.py" "$D/"
cp -p "$W/page/docs_README.md" "$D/README.md"
for f in "$W"/page/scripts/*.py "$W"/page/scripts/*.css "$W"/page/scripts/*.sh "$W"/page/scripts/*.js; do
  [ -e "$f" ] && cp -p "$f" "$D/scripts/"
done
for f in followup_queue.md p4_design.md p4_design.css p4_facts.md p4_facts.json notation.md structure3.md tooltips_audit.md; do
  cp -p "$W/page/notes/$f" "$D/notes/"
done
for f in "$W"/qviewer/*.py "$W"/qviewer/*.sh HANDOFF.md section.html script.js style.css qv_data.json qv_extra.json q_checks.json; do
  case "$f" in /*) [ -e "$f" ] && cp -p "$f" "$D/qviewer/";; *) cp -p "$W/qviewer/$f" "$D/qviewer/";; esac
done
cp -p "$W/mathbuild/render.js" "$W/mathbuild/package.json" "$W/mathbuild/package-lock.json" "$W/mathbuild/page_math.json" "$D/mathbuild/"
md5sum "$D/thrusterchain.html"
du -sh "$D"

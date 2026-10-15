#!/bin/bash
# Final screenshots of the critic-in-action section in test_page.html with the page's shot.py (CPU 2, nice 10),
# into shots/final-*.png, at 390 px light and 1200 px dark.
cd /mnt/c/Users/David/dev/bca-shift-work/qviewer || exit 1
SHOT="taskset -c 2 nice -n 10 /opt/shot-venv/bin/python /mnt/c/Users/David/dev/bca-shift-work/page/scripts/shot.py"
PAGE=/mnt/c/Users/David/dev/bca-shift-work/qviewer/test_page.html
OUT=/mnt/c/Users/David/dev/bca-shift-work/qviewer/shots
one() {   # width scheme eval shots...
  local w=$1 sc=$2 ev=$3; shift 3
  local args=()
  for s in "$@"; do args+=(--shot "$s"); done
  $SHOT --page $PAGE --out-dir $OUT --width $w --scheme $sc --eval "$ev" "${args[@]}" --split 1300 2>&1 |
    grep -E "overflow after actions|console errors|page errors|failed requests|FAILED" | sed "s/^/[$w $sc] /"
}
for ws in "390 light" "1200 dark"; do
  set -- $ws; w=$1; sc=$2; p="final-$w-$sc"
  one $w $sc 'qvSet({q:"expert",c:"gap",s:"strong",view:"rel",range:"hi",t:40})' "$p-section.png=#critic-in-action"
  one $w $sc 'qvSet({q:"expert",c:"gap",s:"strong",view:"rel",range:"hi",t:40})' "$p-c1-gap-strong-t40.png=#qv-c1" \
    "$p-now-gap-strong-t40.png=#qv-now" "$p-c2-gap-strong.png=#qv-c2" "$p-ep-gap-strong.png=#qv-ep"
  one $w $sc 'qvSet({q:"expert",c:"gap",s:"strong",view:"rel",range:"all",t:40})' "$p-c1-gap-strong-t40-all.png=#qv-c1"
  one $w $sc 'qvSet({q:"expert",c:"gap",s:"strong",view:"abs",range:"hi",t:40})' "$p-c1-gap-strong-t40-abs.png=#qv-c1"
  one $w $sc 'qvSet({q:"expert",c:"full",s:"strong",view:"rel",range:"hi",t:40})' "$p-c1-full-strong-t40.png=#qv-c1"
  one $w $sc 'qvSet({q:"expert",c:"gap",s:"strong",view:"rel",range:"hi",t:0})' "$p-strip-t0.png=.qv-stripbox"
  one $w $sc 'qvSet({q:"expert",c:"gap",s:"strong",view:"rel",range:"hi",t:78})' "$p-now-under-0.001.png=#qv-now" "$p-readouts-under-0.001.png=#qv-c1+.readouts"
  one $w $sc 'qvSet({q:"expert",c:"gap",s:"strong",t:40}); document.getElementById("qv-tables").open = true; document.getElementById("qv-qual").open = true' \
    "$p-tables.png=#qv-tables" "$p-qual.png=#qv-qual"
done

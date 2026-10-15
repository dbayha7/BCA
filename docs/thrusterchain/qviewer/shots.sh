#!/bin/bash
# Screenshots of the critic-in-action section in test_page.html, with the page's own shot.py, on CPU 2 at nice 10.
# Usage: bash shots.sh [390|1200|all]
cd /mnt/c/Users/David/dev/bca-shift-work/qviewer || exit 1
SHOT="taskset -c 2 nice -n 10 /opt/shot-venv/bin/python /mnt/c/Users/David/dev/bca-shift-work/page/scripts/shot.py"
PAGE=/mnt/c/Users/David/dev/bca-shift-work/qviewer/test_page.html
OUT=/mnt/c/Users/David/dev/bca-shift-work/qviewer/shots
one() {   # width scheme tag eval shots...
  local w=$1 sc=$2 tag=$3 ev=$4; shift 4
  local args=()
  for s in "$@"; do args+=(--shot "$s"); done
  $SHOT --page $PAGE --out-dir $OUT --width $w --scheme $sc --eval "$ev" "${args[@]}" --split 1300 2>&1 |
    grep -E "overflow after actions|console errors|page errors|failed requests|FAILED" | sed "s/^/[$tag] /"
}
run() {
  local w=$1 sc=$2 p="$1-$2"
  one $w $sc "$p section, the 80% hold removed, strong, t40" 'qvSet({q:"expert",c:"gap",s:"strong",view:"rel",range:"all",t:40})' \
    "$p-section-gap-strong-t40.png=#critic-in-action"
  for c in gap full; do
    for t in 0 40 99; do
      one $w $sc "$p $c strong t$t" "qvSet({q:\"expert\",c:\"$c\",s:\"strong\",view:\"rel\",range:\"all\",t:$t})" \
        "$p-strip-$c-strong-t$t.png=.qv-stripbox" "$p-now-$c-strong-t$t.png=#qv-now" \
        "$p-c1-$c-strong-t$t.png=#qv-c1" "$p-c2-$c-strong-t$t.png=#qv-c2" "$p-ep-$c-strong.png=#qv-ep"
    done
  done
  one $w $sc "$p gap strong t40, 60% to 100%" 'qvSet({q:"expert",c:"gap",s:"strong",view:"rel",range:"hi",t:40})' \
    "$p-c1-gap-strong-t40-hi.png=#qv-c1"
  one $w $sc "$p gap strong t40, values as computed" 'qvSet({q:"expert",c:"gap",s:"strong",view:"abs",range:"all",t:40})' \
    "$p-c1-gap-strong-t40-abs.png=#qv-c1"
  one $w $sc "$p random gap training, actor downstream" '(() => { const D = JSON.parse(document.getElementById("qv-data").textContent); const e = D.eps.find(x => x[0] === "random" && x[1] === "gap" && x[2] === "training"); const t = e[8].findIndex(c => c[0] > 0); return qvSet({q: "random", c: "gap", s: "training", view: "rel", range: "all", t}); })()' \
    "$p-c1-random-gap-training-down.png=#qv-c1" "$p-now-random-gap-training-down.png=#qv-now"
}
case "${1:-all}" in
  390) run 390 light ;;
  1200) run 1200 dark ;;
  *) run 390 light; run 1200 dark ;;
esac

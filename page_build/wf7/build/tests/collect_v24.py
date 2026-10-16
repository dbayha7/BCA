"""Test only: read every result file of the final run (run_final.sh) and print one line per check, with the time each
result file was written, so that nothing from an earlier run is counted. Writes out/results_v24.json (version 24;
collect_results.py is the same for version 20).
Usage: python -X utf8 collect_results.py"""
import json
import re
from datetime import datetime, timezone
from pathlib import Path

T = Path(__file__).resolve().parent
log = (T / "run_v24.log").read_text(encoding="utf-8")
m = re.search(r"== start (\d\d:\d\d:\d\d)", log)
start = m.group(1) if m else None
R, stale = {}, []


def load(p):
    p = T / p
    ts = datetime.fromtimestamp(p.stat().st_mtime, timezone.utc).strftime("%H:%M:%S")
    if start and ts < start:
        stale.append(str(p.relative_to(T)) + " " + ts)
    return json.loads(p.read_text(encoding="utf-8")) if p.suffix == ".json" else p.read_text(encoding="utf-8")


# static
st = load("out/static_v24.json")
R["static: page contract, ids, links, spelling, old navigation gone"] = st["ok"]
# lesson
rep = load("lesson/test/report_v24.json")
for k, c in rep["configs"].items():
    ok = (not c["errors"] and c["walkOk"] and c["page"]["render"] == "ok" and c["page"]["legends"] == 58 and c["page"]["formulas"] == 58
          and c["page"]["widest"] <= 42 and all(c["page"][x] for x in ("sim", "task", "critic", "lc", "tdModes")) and not c["failedRequests"])
    if "controls" in c:
        ok = ok and not c["controlsFailed"] and not c["d2"]["bad"] and not c["targets"]
        R["lesson %s: controls %d/%d, D2 lines %d (bad %d), touch targets" % (k, c["controlsPassed"], len(c["controls"]), c["d2"]["lines"], len(c["d2"]["bad"]))] = ok
    else:
        R["lesson %s: walk 39 steps and back, panels, formulas" % k] = ok
ex = json.loads(load("lesson/test/extra_v24_out.txt"))
R["lesson extra: random slider moves (%d line states, bad %d)" % (ex["random"]["lineStates"], ex["random"]["bad"])] = ex["random"]["bad"] == 0 and ex["random"]["lineStates"] > 1000 and not ex["randomErrors"]
for k in ("lesson_noUF", "lesson_reduced", "nav_noUF", "nav_reduced"):
    v = ex[k]
    R["lesson extra %s: links, walk, hiding, animation, chart width" % k] = (not v["errors"] and not v["links"] and v["walk"] and v["animation"] == "none"
                                                                        and v["chartW"] > 0 and not v["consoleErrors"])
# color
ver = load("color/work/verify_v24int.txt")
m2 = re.search(r"(\d+) checks passed, (\d+) failed", ver)
R["color: plan checks on 4 views (%s passed, %s failed)" % (m2.group(1), m2.group(2))] = m2.group(2) == "0"
tx = load("color/work/text_v24int.json")
okt = True
for n, v in tx.items():
    okt = okt and not v["console"] and all(isinstance(x, dict) and x["inside"] for x in v["tips"].values()) and all(v[j] != "missing" for j in ("statusend", "atsstatus", "claim1", "results"))
R["color: edited text found and new term tooltips open inside the screen (1280, 390)"] = okt
# check3
c3 = load("int/out/check3_v24.json")
R["old full check (check3, adapted): 4 views, 25 sections each, whole page"] = c3.get("ok") is True
# the regression test against version 23 (new for version 24)
rg = load("regression/regression_v23.json")
R["regression against version 23: (a) ids %s, (b) lines %s, (c) features in 4 views %s" % (rg["a"]["ok"], rg["b"]["ok"], all(x["ok"] for x in rg["c"].values()))] = rg["ok"] is True
tm = load("out/timing_v24.json")
R["(info) load at 390 px, 4x CPU, medians of %d loads: %s" % (tm["n"], json.dumps(tm["medians"]))] = True
# navigation
nav = load("nav/test/out/summary.json")
for k, v in nav.items():
    R["navigation " + k] = v
# perf
perf = load("lesson/test/perf_v24_out.txt").strip().splitlines()[-1]
R["(info) load at 390 px, unthrottled: " + perf[:300]] = True
out = {"start": start, "stale_files": stale, "results": R, "all_pass": all(R.values()) and not stale}
(T / "out").mkdir(exist_ok=True)
(T / "out" / "results_v24.json").write_text(json.dumps(out, indent=1, ensure_ascii=False), encoding="utf-8")
for k, v in R.items():
    print("PASS" if v else "FAIL", k)
print("stale result files:", stale)
print("all pass:", out["all_pass"])

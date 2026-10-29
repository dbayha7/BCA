"""Test only: read every result file of the final run (run_final.sh) and print one line per check, with the time each
result file was written, so that nothing from an earlier run is counted. Writes out/results_v27.json (version 27;
collect_results.py, collect_v24.py and collect_v26.py are the same for versions 20, 24 and 26; made by make_tests_v27.py).
Usage: python -X utf8 collect_results.py"""
import json
import re
from datetime import datetime, timezone
from pathlib import Path

T = Path(__file__).resolve().parent
log = (T / "run_v27.log").read_text(encoding="utf-8")
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
st = load("out/static_v27.json")
R["static: page contract, ids, links, spelling, old navigation gone, no status box, the live page's data and tooltip kept"] = st["ok"]
# lesson
rep = load("lesson/test/report_v27.json")
for k, c in rep["configs"].items():
    ok = (not c["errors"] and c["walkOk"] and c["page"]["render"] == "ok" and c["page"]["legends"] == 58 and c["page"]["formulas"] == 58
          and c["page"]["widest"] <= 42 and all(c["page"][x] for x in ("sim", "task", "critic", "lc", "tdModes")) and not c["failedRequests"])
    if "controls" in c:
        ok = ok and not c["controlsFailed"] and not c["d2"]["bad"] and not c["targets"]
        R["lesson %s: controls %d/%d, D2 lines %d (bad %d), touch targets" % (k, c["controlsPassed"], len(c["controls"]), c["d2"]["lines"], len(c["d2"]["bad"]))] = ok
    else:
        R["lesson %s: walk 39 steps and back, panels, formulas" % k] = ok
ex = json.loads(load("lesson/test/extra_v27_out.txt"))
R["lesson extra: random slider moves (%d line states, bad %d)" % (ex["random"]["lineStates"], ex["random"]["bad"])] = ex["random"]["bad"] == 0 and ex["random"]["lineStates"] > 1000 and not ex["randomErrors"]
for k in ("lesson_noUF", "lesson_reduced", "nav_noUF", "nav_reduced"):
    v = ex[k]
    R["lesson extra %s: links, walk, hiding, animation, chart width" % k] = (not v["errors"] and not v["links"] and v["walk"] and v["animation"] == "none"
                                                                        and v["chartW"] > 0 and not v["consoleErrors"])
# color
ver = load("color/work/verify_v27int.txt")
m2 = re.search(r"(\d+) checks passed, (\d+) failed", ver)
R["color: plan checks on 4 views (%s passed, %s failed)" % (m2.group(1), m2.group(2))] = m2.group(2) == "0"
tx = load("color/work/text_v27int.json")
okt = True
for n, v in tx.items():
    okt = okt and not v["console"] and all(v[j] != "missing" for j in ("atsstatus", "claim1", "results", "outcome", "plan")) and v["noStatusBox"] is True
R["color: the edited status text found (After the stop's box, the first claim, Results' opening, Stage 2's outcome, the plan), no status box (1280, 390)"] = okt
# check3
c3 = load("int/out/check3_v27.json")
R["old full check (check3, adapted): 4 views, 25 sections each, whole page"] = c3.get("ok") is True
# the regression test against the live page
rg = load("regression/regression_live.json")
R["regression against the live page: (a) ids %s, (b) lines %s, R10 toy section %s, (c) features in 4 views %s" % (rg["a"]["ok"], rg["b"]["ok"], rg["toy"]["ok"], all(x["ok"] for x in rg["c"].values()))] = rg["ok"] is True
# the toy section (the toy integration): six views on the integrated page, and inside the navigation
ty = load("toy/result_v27.json")
for v in ty["views"]:
    p = v["probe"]
    ok = (p["sectionFound"] and p["docScrollW"] <= p["vw"] and p["overflowCount"] == 0 and p["calcs"] == 0 and p["calcCount"] == 15
          and len(p["charts"]) == 4 and all(c["ok"] and c["paths"] >= 3 and c["axisTitles"] == 2 and c["hits"] > 0 for c in p["charts"])
          and not p["badLinks"] and not p["dupIds"] and all(t["unlabeled"] == 0 for t in p["tables"]) and not p["small"]
          and len(v["tips"]) == 4 and all(t["shown"] and t["keys"] >= 2 and t["inView"] for t in v["tips"]) and not v["errors"])
    R["toy section %s: no sideways scroll, 15 worked blocks fit, 4 charts drawn with both axes named, tooltips with a color key per line, links, labels, touch targets" % v["view"]] = ok
tn = load("toy/nav_v27.json")
R["toy section inside the navigation (1280 light, 390 dark): shown, In this section lists its 44 parts, links into it open their part"] = all(
    x["probe"]["shown"] and x["probe"]["railCount"] == 44 and not x["probe"]["badRail"] and x["probe"]["overflowCount"] == 0
    and x["probe"]["docScrollW"] <= x["probe"]["vw"] and x["meaning"]["visible"] and x["chart"]["visible"] and x["chart"]["drawn"] and not x["errors"] for x in tn)
tm = load("out/timing_v27.json")
R["(info) load at 390 px, 4x CPU, medians of %d loads: %s" % (tm["n"], json.dumps(tm["medians"]))] = True
# navigation
nav = load("nav/test/out/summary.json")
for k, v in nav.items():
    R["navigation " + k] = v
# perf
perf = load("lesson/test/perf_v27_out.txt").strip().splitlines()[-1]
R["(info) load at 390 px, unthrottled: " + perf[:300]] = True
out = {"start": start, "stale_files": stale, "results": R, "all_pass": all(R.values()) and not stale}
(T / "out").mkdir(exist_ok=True)
(T / "out" / "results_v27.json").write_text(json.dumps(out, indent=1, ensure_ascii=False), encoding="utf-8")
for k, v in R.items():
    print("PASS" if v else "FAIL", k)
print("stale result files:", stale)
print("all pass:", out["all_pass"])

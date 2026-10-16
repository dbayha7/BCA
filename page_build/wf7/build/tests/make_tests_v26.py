"""Test only: make the version-26 test scripts from the version-24 ones (page version 26 = version 25 + this build; the
reference page is version 25). Each replacement's old text must occur as often as stated, or nothing is written.

Made here (the version-24 scripts are kept as they were):
  setup_tests_v26.py   from setup_tests_v24.py: base and reference version 25 (live5/core_4933.html, live5/live_4933.html's
                       wrapper), stages in ../stages_v26, the navigation's expected lists rules/pass2_v25.json (made by
                       rules/run_rules.mjs on version 25 with the static edits; identical to pass2_v23's lists), T-7 and T-18
                       against version 25, the lesson's suite on core_v26.html, the color reference render_v25.json; the
                       text shots also find the first claim's status of 16 Oct, shoot the status box's new line on the BCA
                       and BCA IW rerun and open its BCA IW tooltip
  static_v26.py        from static_v24.py: base version 25, built page version 26; also checks the status box's rerun line
                       against the page's own data (script#sim-rerun-data), and the present parts of versions 24-25
  int/check3_v26.mjs   from int/check3_v24.mjs: version 26's page and entry, version 25 as the reference
  timing_v26.mjs       from timing_v24.mjs: version 25 against version 26
  collect_v26.py       from collect_v24.py: the version-26 result files (regression against version 25)
  run_v26.sh           from run_v24.sh
(regression_v25.mjs is written by hand.)
Usage: python -X utf8 make_tests_v26.py
"""
import sys
from pathlib import Path

T = Path(__file__).resolve().parent


def make(src, dst, pairs):
    s = (T / src).read_text(encoding="utf-8")
    for p in pairs:
        old, new = p[0], p[1]
        n = p[2] if len(p) > 2 else 1
        c = s.count(old)
        if c != n:
            sys.exit("make_tests_v26.py: %s: %r occurs %d times, expected %d" % (src, old[:100], c, n))
        s = s.replace(old, new)
    (T / dst).write_text(s, encoding="utf-8", newline="\n")
    print("made", dst, "from", src, len(pairs), "replacements")


# ------------------------------------------------------------------ setup
make("setup_tests_v24.py", "setup_tests_v26.py", [
    ('''page (version 24, built on version 23), and build the extra pages they compare against. The components' own folders
are only read. (setup_tests.py is the version-20 setup on version 19, kept as it was.)

Usage: python -X utf8 setup_tests_v24.py   (after apply_all.py has written ../core_v24.html, ../page_v24_preview.html
and ../stages/)''',
     '''page (version 26, built on version 25), and build the extra pages they compare against. The components' own folders
are only read. (setup_tests.py is the version-20 setup on version 19, setup_tests_v24.py the version-24 setup on version
23; both kept as they were. This file is made by make_tests_v26.py from setup_tests_v24.py.)

Usage: python -X utf8 setup_tests_v26.py   (after apply_all.py has written ../core_v26.html, ../page_v26_preview.html
and ../stages_v26/)'''),
    ("What changed from setup_tests.py, paths and reference pages only (the tests' checks are unchanged): the base and the\n"
     "reference page are version 23 (live4/core_6eea.html, with live4/live_6eea.html's wrapper) instead of version 19; the\n"
     "navigation's expected lists (T-2) are rules/pass2_v23.json, made by rules/run_rules.mjs on version 23 with the static\n"
     "edits (static_edits_v23.py); its T-7 redraws and T-18 startup compare with version 23 (v23_live.html); T-20 checks the\n"
     "static edits' anchors once in version 23; the lesson's suite runs on core_v24.html.",
     "What changed from setup_tests.py, paths and reference pages only (the tests' checks are unchanged): the base and the\n"
     "reference page are version 25 (live5/core_4933.html, with live5/live_4933.html's wrapper) instead of version 19; the\n"
     "navigation's expected lists (T-2) are rules/pass2_v25.json, made by rules/run_rules.mjs on version 25 with the static\n"
     "edits (rules/make_with_static.py); its T-7 redraws and T-18 startup compare with version 25 (v25_live.html); T-20 checks\n"
     "the static edits' anchors once in version 25; the lesson's suite runs on core_v26.html."),
    ("  nav/build/nav_lesson_live.html    the integrated page (= ../page_v24_preview.html)",
     "  nav/build/nav_lesson_live.html    the integrated page (= ../page_v26_preview.html)"),
    ("  nav/build/v23_live.html           version 23 (reference for redraws and startup time)",
     "  nav/build/v25_live.html           version 25 (reference for redraws and startup time)"),
    ('LIVE4 = SCR / "live4"\nWRAP = (LIVE4 / "live_6eea.html").read_bytes()[:531]',
     'LIVE4 = SCR / "live5"   # version 25 (the name LIVE4 is kept for the code below)\nWRAP = (LIVE4 / "live_4933.html").read_bytes()[:531]'),
    ("# the base is version 23: T-20's anchors are checked once in it\n"
     "sub(NT / \"test\" / \"static_checks.py\", 'base = (LIVE3 / \"core_9369.html\")', 'base = (LIVE3 / \"core_6eea.html\")', 1)",
     "# the base is version 25: T-20's anchors are checked once in it\n"
     "sub(NT / \"test\" / \"static_checks.py\", 'base = (LIVE3 / \"core_9369.html\")', 'base = (LIVE3 / \"core_4933.html\")', 1)"),
    ("# T-2's expected lists: the rules on version 23 with the static edits (pass2_v23.json)\n"
     "sub(NT / \"test\" / \"t_lists.mjs\", \"'/rules/pass2_v19.json'\", \"'/rules/pass2_v23.json'\", 1)\n"
     "# T-7 and T-18 compare with version 23\n"
     "sub(NT / \"test\" / \"t_panels.mjs\", \"file: 'v19_live.html'\", \"file: 'v23_live.html'\", 1)\n"
     "sub(NT / \"test\" / \"t_timing.mjs\", \"'v19_live.html'\", \"'v23_live.html'\")",
     "# T-2's expected lists: the rules on version 25 with the static edits (pass2_v25.json)\n"
     "sub(NT / \"test\" / \"t_lists.mjs\", \"'/rules/pass2_v19.json'\", \"'/rules/pass2_v25.json'\", 1)\n"
     "# T-7 and T-18 compare with version 25\n"
     "sub(NT / \"test\" / \"t_panels.mjs\", \"file: 'v19_live.html'\", \"file: 'v25_live.html'\", 1)\n"
     "sub(NT / \"test\" / \"t_timing.mjs\", \"'v19_live.html'\", \"'v25_live.html'\")"),
    ('sub(NT / "test" / "summarize.py", "DCL %d ms vs version 19 %d ms", "DCL %d ms vs version 23 %d ms", 1)',
     'sub(NT / "test" / "summarize.py", "DCL %d ms vs version 19 %d ms", "DCL %d ms vs version 25 %d ms", 1)'),
    ('stages = BUILD / "stages"\nshutil.copy2(BUILD / "page_v24_preview.html", NB / "nav_lesson_live.html")',
     'stages = BUILD / "stages_v26"\nshutil.copy2(BUILD / "page_v26_preview.html", NB / "nav_lesson_live.html")'),
    ('wrap(LIVE4 / "core_6eea.html", NB / "v23_live.html")', 'wrap(LIVE4 / "core_4933.html", NB / "v25_live.html")'),
    ('''sub(LT / "test" / "lib.mjs", "SCR + '/live3/live_9369.html'", "SCR + '/live4/live_6eea.html'", 1)   # version 23's wrapper''',
     '''sub(LT / "test" / "lib.mjs", "SCR + '/live3/live_9369.html'", "SCR + '/live5/live_4933.html'", 1)   # version 25's wrapper'''),
    ('shutil.copy2(BUILD / "core_v24.html", LT / "work" / "v24_core.html")', 'shutil.copy2(BUILD / "core_v26.html", LT / "work" / "v26_core.html")'),
    ('''    "const pages = { lesson: L + '/work/color_lesson_core.html', nav: L + '/work/v24_core.html' };   // integration", 1)''',
     '''    "const pages = { lesson: L + '/work/color_lesson_core.html', nav: L + '/work/v26_core.html' };   // integration", 1)'''),
    ('''wrap(LIVE4 / "core_6eea.html", CT / "work" / "v23_preview.html")   # rendered as the learning curves' reference (render_v23.json)''',
     '''wrap(LIVE4 / "core_4933.html", CT / "work" / "v25_preview.html")   # rendered as the learning curves' reference (render_v25.json)
# version 26: the first claim's status is dated 16 Oct; the status box's new line (version 25's rerun of BCA and BCA IW)
# is shot, and its BCA IW tooltip opened, with the other new tooltips
sub(CT / "tools" / "text_shots.mjs", "/^Status, as of 15 Oct/.test(p.textContent)", "/^Status, as of 1\\\\d Oct/.test(p.textContent)", 1)
sub(CT / "tools" / "text_shots.mjs",
    '  ["results", () => `[...document.querySelectorAll("#results > p")].slice(0, 1)`],\\n',
    '  ["results", () => `[...document.querySelectorAll("#results > p")].slice(0, 1)`],\\n'
    '  ["rerunline", () => `[...document.querySelectorAll("#head-status ul.g-status > li")].filter(li => /BCA and BCA IW/.test(li.textContent))`],   // version 26\\n', 1)
sub(CT / "tools" / "text_shots.mjs", 'for (const t of ["CQL", "IQL", "ReBRAC", "Unifloral", "Weighted BCA"]) {',
    'for (const t of ["CQL", "IQL", "ReBRAC", "Unifloral", "Weighted BCA", "BCA IW"]) {   // version 26: BCA IW', 1)'''),
])

# ------------------------------------------------------------------ static checks
make("static_v24.py", "static_v26.py", [
    ('''"""Test only: static checks of the integrated page (version 24) against its base, version 23 (static_v20.py did the same
for version 20 against version 19; only the pages and names changed).''',
     '''"""Test only: static checks of the integrated page (version 26) against its base, version 25 (static_v20.py did the same
for version 20 against version 19, static_v24.py for version 24 against version 23; made by make_tests_v26.py).'''),
    ('''    custom property those blocks define on :root is redefined in both dark blocks (or is defined from other tokens);''',
     '''    custom property those blocks define on :root is redefined in both dark blocks (or is defined from other tokens);
  - the status box's line on version 25's rerun (BCA and BCA IW) states the numbers of the page's own data block;'''),
    ("Usage: python -X utf8 static_v24.py  -> prints a summary and writes out/static_v24.json",
     "Usage: python -X utf8 static_v26.py  -> prints a summary and writes out/static_v26.json"),
    ('''v19 = (SCR / "live4" / "core_6eea.html").read_text(encoding="utf-8")   # the base: version 23 (the name v19 is kept for the code below)
v20 = (BUILD / "core_v24.html").read_text(encoding="utf-8")            # the built page: version 24''',
     '''v19 = (SCR / "live5" / "core_4933.html").read_text(encoding="utf-8")   # the base: version 25 (the name v19 is kept for the code below)
v20 = (BUILD / "core_v26.html").read_text(encoding="utf-8")            # the built page: version 26'''),
    ('''"<b>Version 24, 15 Oct 2026.</b>", "(page version 24)", 'id="simtog"', 'id="sim-learned-data"', 'id="lc-table5"', 'id="iw-plain-split"')}''',
     '''"<b>Version 26, 16 Oct 2026.</b>", "(page version 26)", 'id="simtog"', 'id="sim-learned-data"', 'id="lc-table5"', 'id="iw-plain-split"',
                                                   "<b>Version 25, 16 Oct 2026.</b>", "<b>Version 24, 15 Oct 2026.</b>", 'id="sim-rerun-data"', 'id="simtog-l-lr"', 'id="exLrnT"')}

# ------------------------------------------------------------------ version 26: the status box's rerun line against the page's data
import json as _json
_d = _json.loads(re.search(r'<script type="application/json" id="sim-rerun-data">(.*?)</script>', v20, re.S).group(1))
_Q = ["random", "medium-replay", "medium", "medium-expert", "expert"]
_p = {l: [_d["avg"][q + "|" + l + "|strong"]["pts"] for q in _Q] for l in ("td3bc", "bca", "bcaiw")}
_bt = [round(b - t, 1) for b, t in zip(_p["bca"], _p["td3bc"])]
_ib = [b - a for b, a in zip(_p["bcaiw"], _p["bca"])]
_li = re.search(r'<li><span class="tag">BCA and BCA IW: exploratory, one seed</span>(.*?)</li>', v20, re.S)
_txt = re.sub(r"\\s+", " ", re.sub(r"<[^>]+>", "", _li.group(1))) if _li else ""
res["rerun_line"] = {"bca_minus_td3bc": _bt, "bcaiw_minus_bca": [round(x, 2) for x in _ib], "text": _txt[:600],
                     "date": _d["notes"][0][:200]}
res["rerun_line_ok"] = bool(_li) and (
    _bt[:3] == [-10.6, -5.5, -3.5] and "3.5 to 10.6 weighting-test points below TD3+BC with random, medium-replay and medium data" in _txt
    and _bt[3:] == [0.1, 0.2] and "0.1 and 0.2 above it with medium-expert and expert data" in _txt
    and max(abs(x) for x in _ib) < 1 and "BCA IW was within 1 point of BCA" in _txt
    and "2026-10-15T22:32:50" in _d["notes"][0] and "(15 Oct, 22:32 UTC)" in _txt and "with the recorded commands around the 80% hold (thruster 1 held at 80% upstream) removed" in _txt
    and _d["learners"] == {"td3bc": "host_gap", "bca": "unweighted", "bcaiw": "exact_strong"})'''),
    ('''      and all(res["absent"].values()) and all(n == 1 for n in res["present_once"].values()))''',
     '''      and all(res["absent"].values()) and all(n == 1 for n in res["present_once"].values()) and res["rerun_line_ok"])'''),
    ('''(HERE / "out" / "static_v24.json").write_text(''', '''(HERE / "out" / "static_v26.json").write_text('''),
    ('''print("static_v24 ok:", ok)''', '''print("static_v26 ok:", ok)'''),
])

# ------------------------------------------------------------------ the old full check
make("int/check3_v24.mjs", "int/check3_v26.mjs", [
    ("// Test only: the old full check (final_look/check3.mjs) adapted to the new navigation, for page version 24 (check3_v20.mjs\n"
     "// did the same for version 20; here the pages, the reference and the version strings changed). For version 24 at 390x844",
     "// Test only: the old full check (final_look/check3.mjs) adapted to the new navigation, for page version 26 (check3_v20.mjs\n"
     "// and check3_v24.mjs did the same for versions 20 and 24; here the pages, the reference and the version strings\n"
     "// changed; made by make_tests_v26.py). For version 26 at 390x844"),
    ("// The same whole-page probes on version 23, the base (no navigation), are the reference. Writes out/check3_v24.json.",
     "// The same whole-page probes on version 25, the base (no navigation), are the reference. Writes out/check3_v26.json."),
    ("    v20entry: t.includes('Version 24, 15 Oct 2026.') && ['23', '22', '21', '20'].every(v => t.includes('Version ' + v + ', 15 Oct 2026.')),   // version 24's entry, and versions 20-23's kept",
     "    v20entry: t.includes('Version 26, 16 Oct 2026.') && t.includes('Version 25, 16 Oct 2026.') && ['24', '23', '22', '21', '20'].every(v => t.includes('Version ' + v + ', 15 Oct 2026.')),   // version 26's entry, and versions 20-25's kept"),
    ("launch({ port: 12791, logName: 'edge_check3_v24.log' })", "launch({ port: 12791, logName: 'edge_check3_v26.log' })"),
    ("    await navigate(cdp, pg, 'file:///' + B + '/page_v24_preview.html', { settle: 3500 });",
     "    await navigate(cdp, pg, 'file:///' + B + '/page_v26_preview.html', { settle: 3500 });"),
    ("    await navigate(cdp, pg, 'file:///' + B + '/tests/nav/build/v23_live.html', { settle: 3500 });",
     "    await navigate(cdp, pg, 'file:///' + B + '/tests/nav/build/v25_live.html', { settle: 3500 });"),
    ("fs.writeFileSync(B + '/tests/int/out/check3_v24.json', JSON.stringify(out, null, 1));", "fs.writeFileSync(B + '/tests/int/out/check3_v26.json', JSON.stringify(out, null, 1));", 2),
    ("&& r.whole.v20entry && /page version 24/.test(r.whole.head || '')", "&& r.whole.v20entry && /page version 26/.test(r.whole.head || '')"),
    ("console.log('check3_v24 ok:', ok);", "console.log('check3_v26 ok:', ok);"),
])

# ------------------------------------------------------------------ timing
make("timing_v24.mjs", "timing_v26.mjs", [
    ("// Load time of page version 24 against version 23 (test only): 390x844, touch, the CPU slowed 4 times, alternating\n"
     "// loads (version 23, then version 24, N times each, a fresh tab each time).",
     "// Load time of page version 26 against version 25 (test only; made by make_tests_v26.py from timing_v24.mjs): 390x844,\n"
     "// touch, the CPU slowed 4 times, alternating loads (version 25, then version 26, N times each, a fresh tab each time)."),
    ("the navigation's own start (window.__nvInitMs, version 24 only)", "the navigation's own start (window.__nvInitMs, version 26 only)"),
    ("// Usage: node timing_v24.mjs [N=5] -> out/timing_v24.json", "// Usage: node timing_v26.mjs [N=5] -> out/timing_v26.json"),
    ("const PAGES = { v23: 'file:///' + T + '/regression/v23_live.html', v24: 'file:///' + T + '/../page_v24_preview.html' };\n"
     "if (!fs.existsSync(T + '/regression/v23_live.html')) {",
     "const PAGES = { v25: 'file:///' + T + '/regression/v25_live.html', v26: 'file:///' + T + '/../page_v26_preview.html' };\n"
     "if (!fs.existsSync(T + '/regression/v25_live.html')) {"),
    ("  fs.writeFileSync(T + '/regression/v23_live.html', Buffer.concat([fs.readFileSync(SCR + '/live4/live_6eea.html').subarray(0, 531),\n"
     "    fs.readFileSync(SCR + '/live4/core_6eea.html'), Buffer.from('\\n</body></html>')]));",
     "  fs.writeFileSync(T + '/regression/v25_live.html', Buffer.concat([fs.readFileSync(SCR + '/live5/live_4933.html').subarray(0, 531),\n"
     "    fs.readFileSync(SCR + '/live5/core_4933.html'), Buffer.from('\\n</body></html>')]));"),
    ("const runs = { v23: [], v24: [] };", "const runs = { v25: [], v26: [] };"),
    ("launch({ port: 13961, logName: 'edge_timing_v24.log' })", "launch({ port: 13961, logName: 'edge_timing_v26.log' })"),
    ("  for (let k = 0; k < N; k++) for (const v of ['v23', 'v24']) {", "  for (let k = 0; k < N; k++) for (const v of ['v25', 'v26']) {"),
    ("medians: { v23: med('v23'), v24: med('v24') }, runs };", "medians: { v25: med('v25'), v26: med('v26') }, runs };"),
    ("out.medians.dclRatio = +(out.medians.v24.dcl / out.medians.v23.dcl).toFixed(2);\n"
     "out.medians.loadRatio = +(out.medians.v24.load / out.medians.v23.load).toFixed(2);",
     "out.medians.dclRatio = +(out.medians.v26.dcl / out.medians.v25.dcl).toFixed(2);\n"
     "out.medians.loadRatio = +(out.medians.v26.load / out.medians.v25.load).toFixed(2);"),
    ("fs.writeFileSync(T + '/out/timing_v24.json', JSON.stringify(out, null, 1));", "fs.writeFileSync(T + '/out/timing_v26.json', JSON.stringify(out, null, 1));"),
])

# ------------------------------------------------------------------ collect
make("collect_v24.py", "collect_v26.py", [
    ('''result file was written, so that nothing from an earlier run is counted. Writes out/results_v24.json (version 24;
collect_results.py is the same for version 20).''',
     '''result file was written, so that nothing from an earlier run is counted. Writes out/results_v26.json (version 26;
collect_results.py and collect_v24.py are the same for versions 20 and 24; made by make_tests_v26.py).'''),
    ('log = (T / "run_v24.log").read_text(encoding="utf-8")', 'log = (T / "run_v26.log").read_text(encoding="utf-8")'),
    ('st = load("out/static_v24.json")', 'st = load("out/static_v26.json")'),
    ('R["static: page contract, ids, links, spelling, old navigation gone"] = st["ok"]',
     'R["static: page contract, ids, links, spelling, old navigation gone, the rerun line against the page\'s data"] = st["ok"]'),
    ('rep = load("lesson/test/report_v24.json")', 'rep = load("lesson/test/report_v26.json")'),
    ('ex = json.loads(load("lesson/test/extra_v24_out.txt"))', 'ex = json.loads(load("lesson/test/extra_v26_out.txt"))'),
    ('ver = load("color/work/verify_v24int.txt")', 'ver = load("color/work/verify_v26int.txt")'),
    ('tx = load("color/work/text_v24int.json")', 'tx = load("color/work/text_v26int.json")'),
    ('''all(v[j] != "missing" for j in ("statusend", "atsstatus", "claim1", "results"))''',
     '''all(v[j] != "missing" for j in ("statusend", "atsstatus", "claim1", "results", "rerunline")) and "BCA IW" in v["tips"]'''),
    ('R["color: edited text found and new term tooltips open inside the screen (1280, 390)"] = okt',
     'R["color: edited text found (the rerun line too) and new term tooltips (BCA IW too) open inside the screen (1280, 390)"] = okt'),
    ('c3 = load("int/out/check3_v24.json")', 'c3 = load("int/out/check3_v26.json")'),
    ('''# the regression test against version 23 (new for version 24)
rg = load("regression/regression_v23.json")
R["regression against version 23: (a) ids %s, (b) lines %s, (c) features in 4 views %s" % (rg["a"]["ok"], rg["b"]["ok"], all(x["ok"] for x in rg["c"].values()))] = rg["ok"] is True
tm = load("out/timing_v24.json")''',
     '''# the regression test against version 25
rg = load("regression/regression_v25.json")
R["regression against version 25: (a) ids %s, (b) lines %s, (c) features in 4 views %s" % (rg["a"]["ok"], rg["b"]["ok"], all(x["ok"] for x in rg["c"].values()))] = rg["ok"] is True
tm = load("out/timing_v26.json")'''),
    ('perf = load("lesson/test/perf_v24_out.txt").strip().splitlines()[-1]', 'perf = load("lesson/test/perf_v26_out.txt").strip().splitlines()[-1]'),
    ('(T / "out" / "results_v24.json").write_text(', '(T / "out" / "results_v26.json").write_text('),
])

# ------------------------------------------------------------------ the run script
make("run_v24.sh", "run_v26.sh", [
    ("# Test only (page version 24, built on version 23): every component's suite on the integrated page, the old full check,\n"
     "# the regression test against version 23, then the timing tests last (nothing else running). Pages are set up by\n"
     "# setup_tests_v24.py. One headless Edge at a time (each test launches its own and closes it). Usage: sh run_v24.sh",
     "# Test only (page version 26, built on version 25; made by make_tests_v26.py from run_v24.sh): every component's suite on\n"
     "# the integrated page, the old full check, the regression test against version 25, then the timing tests last (nothing\n"
     "# else running). Pages are set up by setup_tests_v26.py. One headless Edge at a time (each test launches its own and\n"
     "# closes it). Usage: sh run_v26.sh"),
    ("sha256sum $T/../core_v24.html $T/../page_v24_preview.html $T/nav/build/nav_lesson_live.html $T/lesson/work/v24_core.html",
     "sha256sum $T/../core_v26.html $T/../page_v26_preview.html $T/nav/build/nav_lesson_live.html $T/lesson/work/v26_core.html"),
    ("cd $T && $PY -X utf8 static_v24.py > $T/out/static_v24.txt 2>&1; echo \"static exit $?\"",
     "cd $T && $PY -X utf8 static_v26.py > $T/out/static_v26.txt 2>&1; echo \"static exit $?\""),
    ("node test_full.mjs $T/lesson/work/v24_core.html v24 > $T/lesson/test/full_v24_out.txt 2>&1",
     "node test_full.mjs $T/lesson/work/v26_core.html v26 > $T/lesson/test/full_v26_out.txt 2>&1"),
    ("node test_extra.mjs > $T/lesson/test/extra_v24_out.txt 2>&1", "node test_extra.mjs > $T/lesson/test/extra_v26_out.txt 2>&1"),
    ("# 2. color: version 23 rendered as the learning curves' reference;", "# 2. color: version 25 rendered as the learning curves' reference;"),
    ("PAGE_FILE=$T/color/work/v23_preview.html TAG=v23 SHOTS=0 RUNS=d1280_light,d1280_dark,m390_light,m390_dark timeout 3600 node render_test.mjs > $T/color/work/log_v23.txt 2>&1; echo \"color render v23 exit $?\"",
     "PAGE_FILE=$T/color/work/v25_preview.html TAG=v25 SHOTS=0 RUNS=d1280_light,d1280_dark,m390_light,m390_dark timeout 3600 node render_test.mjs > $T/color/work/log_v25.txt 2>&1; echo \"color render v25 exit $?\""),
    ("PAGE_FILE=$T/../page_v24_preview.html TAG=v24int RUNS=d1280_light,d1280_dark,m390_light,m390_dark PRE=\"$PRE\" timeout 3600 node render_test.mjs > $T/color/work/log_v24int.txt 2>&1",
     "PAGE_FILE=$T/../page_v26_preview.html TAG=v26int RUNS=d1280_light,d1280_dark,m390_light,m390_dark PRE=\"$PRE\" timeout 3600 node render_test.mjs > $T/color/work/log_v26int.txt 2>&1"),
    ("node verify_render.mjs $T/color/work/render_v24int.json $T/color/work/render_v23.json > $T/color/work/verify_v24int.txt 2>&1",
     "node verify_render.mjs $T/color/work/render_v26int.json $T/color/work/render_v25.json > $T/color/work/verify_v26int.txt 2>&1"),
    ("PAGE_FILE=$T/../page_v24_preview.html TAG=v24int PRE=\"$PRE\" timeout 1800 node text_shots.mjs > $T/color/work/log_text_v24int.txt 2>&1",
     "PAGE_FILE=$T/../page_v26_preview.html TAG=v26int PRE=\"$PRE\" timeout 1800 node text_shots.mjs > $T/color/work/log_text_v26int.txt 2>&1"),
    ("cd $T/int && timeout 1800 node check3_v24.mjs > $T/int/out/check3_v24.log 2>&1", "cd $T/int && timeout 1800 node check3_v26.mjs > $T/int/out/check3_v26.log 2>&1"),
    ("# 4. the regression test against version 23 (ids, lines, features in 4 views)\n"
     "cd $T && timeout 2400 node regression_v23.mjs > $T/regression/regression_v23.log 2>&1; echo \"regression exit $?\"",
     "# 4. the regression test against version 25 (ids, lines, features in 4 views)\n"
     "cd $T && timeout 3600 node regression_v25.mjs > $T/regression/regression_v25.log 2>&1; echo \"regression exit $?\""),
    ("# 6. timing, alone: the navigation's T-18, the load-time comparison with version 23, the lesson's load cost",
     "# 6. timing, alone: the navigation's T-18, the load-time comparison with version 25, the lesson's load cost"),
    ("cd $T && timeout 1800 node timing_v24.mjs 5 > $T/out/timing_v24.log 2>&1", "cd $T && timeout 1800 node timing_v26.mjs 5 > $T/out/timing_v26.log 2>&1"),
    ("V23=C:/Users/David/AppData/Local/Temp/claude/C--Users-David-dev-BCA/3bdab4f0-86ee-439c-a05a-34828d6b0a71/scratchpad/live4/core_6eea.html\n"
     "V24=C:/Users/David/AppData/Local/Temp/claude/C--Users-David-dev-BCA/3bdab4f0-86ee-439c-a05a-34828d6b0a71/scratchpad/wf7/build/tests/lesson/work/v24_core.html",
     "V23=C:/Users/David/AppData/Local/Temp/claude/C--Users-David-dev-BCA/3bdab4f0-86ee-439c-a05a-34828d6b0a71/scratchpad/live5/core_4933.html\n"
     "V24=C:/Users/David/AppData/Local/Temp/claude/C--Users-David-dev-BCA/3bdab4f0-86ee-439c-a05a-34828d6b0a71/scratchpad/wf7/build/tests/lesson/work/v26_core.html"),
    ("RUNS=\"[[\\\"v23 a\\\",\\\"$V23\\\"],[\\\"v24 a\\\",\\\"$V24\\\"],[\\\"v23 b\\\",\\\"$V23\\\"],[\\\"v24 b\\\",\\\"$V24\\\"]]\" timeout 1800 node perf.mjs > $T/lesson/test/perf_v24_out.txt 2>&1",
     "RUNS=\"[[\\\"v25 a\\\",\\\"$V23\\\"],[\\\"v26 a\\\",\\\"$V24\\\"],[\\\"v25 b\\\",\\\"$V23\\\"],[\\\"v26 b\\\",\\\"$V24\\\"]]\" timeout 1800 node perf.mjs > $T/lesson/test/perf_v26_out.txt 2>&1"),
])

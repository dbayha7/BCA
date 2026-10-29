"""Test only: make the version-27 test scripts from the version-26 ones (page version 27 = the live page after version 25,
whose status box was removed at the owner's request, + this build; the reference page is that live page, service
version 1792156350-92ba, live6/core_92ba.html). Each replacement's old text must occur as often as stated, or nothing
is written.

Made here (the version-26 scripts are kept as they were):
  setup_tests_v27.py   from setup_tests_v26.py: base and reference the live page (live6/core_92ba.html, live6/live_92ba.html's
                       wrapper; the reference page is called l92ba_live.html), stages in ../stages_v27, the navigation's
                       expected lists rules/pass2_v27.json (made by rules/run_rules.mjs on the live page with the static
                       edits: version 25's lists without the status box), T-7 and T-18 against the live page, the lesson's
                       suite on core_v27.html, the color reference render_l92ba.json; the text shots no longer look for the
                       status box (gone): they shoot After the stop's box, the first claim, Results' opening, Stage 2's
                       outcome and "What happened to this plan" (the statements version 27 changed), and check that no part
                       of the status box is on the page; the navigation's T-15 clicks the header's process link (the link
                       to Results it used was in the status box) and expects that link's own section
  static_v27.py        from static_v26.py: base the live page, built page version 27; instead of the status box's rerun line:
                       no status box, version 25's data block and the live page's newer "Strong test mix" tooltip unchanged
  int/check3_v27.mjs   from int/check3_v26.mjs: version 27's page and entry (and no entry for version 26), no status heading
                       on either page, the live page as the reference
  timing_v27.mjs       from timing_v26.mjs: the live page against version 27
  collect_v27.py       from collect_v26.py: the version-27 result files (regression against the live page)
  run_v27.sh           from run_v26.sh
  regression_live.mjs  from regression_v25.mjs: against the live page; the status box's deliberate changes (R1-R3) are gone
                       with the box; new checks: no status box on either page, the header going from its opening paragraph
                       to "The process, step by step", the live page's "Strong test mix" tooltip the same, the navigation's
                       rails listing no status box
The toy integration (16 Oct 2026; the block at the end of this file) then adjusts setup_tests_v27.py, static_v27.py,
regression_live.mjs, run_v27.sh and collect_v27.py for the toy worlds' full results (apply_all.py's first stage, the toy
patch): T-20 on the page after the toy patch, the toy section's own suite, R10 in the regression test.
Usage: python -X utf8 make_tests_v27.py
"""
import re
import sys
from pathlib import Path

T = Path(__file__).resolve().parent
SP = "C:/Users/David/AppData/Local/Temp/claude/C--Users-David-dev-BCA/3bdab4f0-86ee-439c-a05a-34828d6b0a71/scratchpad"


def make(src, dst, pairs, regex=()):
    s = (T / src).read_text(encoding="utf-8")
    for p in pairs:
        old, new = p[0], p[1]
        n = p[2] if len(p) > 2 else 1
        c = s.count(old)
        if c != n:
            sys.exit("make_tests_v27.py: %s: %r occurs %d times, expected %d" % (src, old[:100], c, n))
        s = s.replace(old, new)
    for pat, new, n in regex:
        s, c = re.subn(pat, new, s)
        if c != n:
            sys.exit("make_tests_v27.py: %s: pattern %r matched %d times, expected %d" % (src, pat, c, n))
    (T / dst).write_text(s, encoding="utf-8", newline="\n")
    print("made", dst, "from", src, len(pairs) + len(regex), "replacements")


# ------------------------------------------------------------------ setup
make("setup_tests_v26.py", "setup_tests_v27.py", [
    ('''page (version 26, built on version 25), and build the extra pages they compare against. The components' own folders
are only read. (setup_tests.py is the version-20 setup on version 19, setup_tests_v24.py the version-24 setup on version
23; both kept as they were. This file is made by make_tests_v26.py from setup_tests_v24.py.)

Usage: python -X utf8 setup_tests_v26.py   (after apply_all.py has written ../core_v26.html, ../page_v26_preview.html
and ../stages_v26/)''',
     '''page (version 27, built on the live page after version 25, whose status box was removed), and build the extra pages
they compare against. The components' own folders are only read. (setup_tests.py, setup_tests_v24.py and
setup_tests_v26.py are the setups of versions 20, 24 and 26, kept as they were. This file is made by make_tests_v27.py
from setup_tests_v26.py.)

Usage: python -X utf8 setup_tests_v27.py   (after apply_all.py has written ../core_v27.html, ../page_v27_preview.html
and ../stages_v27/)'''),
    ("reference page are version 25 (live5/core_4933.html, with live5/live_4933.html's wrapper) instead of version 19; the\n"
     "navigation's expected lists (T-2) are rules/pass2_v25.json, made by rules/run_rules.mjs on version 25 with the static\n"
     "edits (rules/make_with_static.py); its T-7 redraws and T-18 startup compare with version 25 (v25_live.html); T-20 checks\n"
     "the static edits' anchors once in version 25; the lesson's suite runs on core_v26.html.",
     "reference page are the live page after version 25 (live6/core_92ba.html, with live6/live_92ba.html's wrapper) instead\n"
     "of version 19; the navigation's expected lists (T-2) are rules/pass2_v27.json, made by rules/run_rules.mjs on that page\n"
     "with the static edits (rules/make_with_static.py); its T-7 redraws and T-18 startup compare with it (l92ba_live.html);\n"
     "T-20 checks the static edits' anchors once in it; the lesson's suite runs on core_v27.html."),
    ("  nav/build/nav_lesson_live.html    the integrated page (= ../page_v26_preview.html)",
     "  nav/build/nav_lesson_live.html    the integrated page (= ../page_v27_preview.html)"),
    ("  nav/build/v25_live.html           version 25 (reference for redraws and startup time)",
     "  nav/build/l92ba_live.html         the live page (reference for redraws and startup time)"),
    ('LIVE4 = SCR / "live5"   # version 25 (the name LIVE4 is kept for the code below)\nWRAP = (LIVE4 / "live_4933.html").read_bytes()[:531]',
     'LIVE4 = SCR / "live6"   # the live page after version 25 (the name LIVE4 is kept for the code below)\nWRAP = (LIVE4 / "live_92ba.html").read_bytes()[:531]'),
    ("# the base is version 25: T-20's anchors are checked once in it\n"
     "sub(NT / \"test\" / \"static_checks.py\", 'base = (LIVE3 / \"core_9369.html\")', 'base = (LIVE3 / \"core_4933.html\")', 1)",
     "# the base is the live page: T-20's anchors are checked once in it\n"
     "sub(NT / \"test\" / \"static_checks.py\", 'base = (LIVE3 / \"core_9369.html\")', 'base = (LIVE3 / \"core_92ba.html\")', 1)"),
    ("# T-2's expected lists: the rules on version 25 with the static edits (pass2_v25.json)\n"
     "sub(NT / \"test\" / \"t_lists.mjs\", \"'/rules/pass2_v19.json'\", \"'/rules/pass2_v25.json'\", 1)\n"
     "# T-7 and T-18 compare with version 25\n"
     "sub(NT / \"test\" / \"t_panels.mjs\", \"file: 'v19_live.html'\", \"file: 'v25_live.html'\", 1)\n"
     "sub(NT / \"test\" / \"t_timing.mjs\", \"'v19_live.html'\", \"'v25_live.html'\")",
     "# T-2's expected lists: the rules on the live page with the static edits (pass2_v27.json)\n"
     "sub(NT / \"test\" / \"t_lists.mjs\", \"'/rules/pass2_v19.json'\", \"'/rules/pass2_v27.json'\", 1)\n"
     "# T-7 and T-18 compare with the live page\n"
     "sub(NT / \"test\" / \"t_panels.mjs\", \"file: 'v19_live.html'\", \"file: 'l92ba_live.html'\", 1)\n"
     "sub(NT / \"test\" / \"t_timing.mjs\", \"'v19_live.html'\", \"'l92ba_live.html'\")"),
    ('sub(NT / "test" / "summarize.py", "DCL %d ms vs version 19 %d ms", "DCL %d ms vs version 25 %d ms", 1)',
     'sub(NT / "test" / "summarize.py", "DCL %d ms vs version 19 %d ms", "DCL %d ms vs the live page %d ms", 1)'),
    # T-15's content link was the status box's link to Results, the header's only one: with the box gone the test fell
    # back to the header's first link, the start bar's "Learn step by step", and failed (first version-27 battery). Its
    # intent is kept: a content link in the header, clicked without hidden="until-found", shows its target's section.
    ("# T-13: every lesson page shown in Whole page (40 with the real lesson, 4 with the stub)\n",
     '''# T-15 (version 27): the header's link to Results was in the status box, which the live page removed; the content link
# clicked in the fallback is now the process's first link, and the check expects that link's own section
sub(NT / "test" / "t_deep.mjs",
    """    await ev(pg, `[...document.querySelectorAll('#top a[href="#results"]')].concat([...document.querySelectorAll('#top a[href^="#"]')])[0].click()`); await sleep(500);""",
    """    f.linkTo = await ev(pg, `(() => { const a = [...document.querySelectorAll('#top a[href="#results"]')].concat([...document.querySelectorAll('#top nav.pipe a[href^="#"]')])[0];
      const u = document.getElementById(a.getAttribute('href').slice(1)).closest('main > header#top, main > section[id], main > figure.panel[id]');
      a.click(); return u.id === 'top' ? 'glance' : u.id; })()`); await sleep(500);   // integration (version 27)""", 1)
sub(NT / "test" / "t_deep.mjs", "&& f.link.unit === 'results'", "&& f.linkTo !== 'glance' && f.link.unit === f.linkTo", 1)
# T-13: every lesson page shown in Whole page (40 with the real lesson, 4 with the stub)
'''),
    ('stages = BUILD / "stages_v26"\nshutil.copy2(BUILD / "page_v26_preview.html", NB / "nav_lesson_live.html")',
     'stages = BUILD / "stages_v27"\nshutil.copy2(BUILD / "page_v27_preview.html", NB / "nav_lesson_live.html")'),
    ('wrap(LIVE4 / "core_4933.html", NB / "v25_live.html")', 'wrap(LIVE4 / "core_92ba.html", NB / "l92ba_live.html")'),
    ('''sub(LT / "test" / "lib.mjs", "SCR + '/live3/live_9369.html'", "SCR + '/live5/live_4933.html'", 1)   # version 25's wrapper''',
     '''sub(LT / "test" / "lib.mjs", "SCR + '/live3/live_9369.html'", "SCR + '/live6/live_92ba.html'", 1)   # the live page's wrapper'''),
    ('shutil.copy2(BUILD / "core_v26.html", LT / "work" / "v26_core.html")', 'shutil.copy2(BUILD / "core_v27.html", LT / "work" / "v27_core.html")'),
    ('''    "const pages = { lesson: L + '/work/color_lesson_core.html', nav: L + '/work/v26_core.html' };   // integration", 1)''',
     '''    "const pages = { lesson: L + '/work/color_lesson_core.html', nav: L + '/work/v27_core.html' };   // integration", 1)'''),
    # the text shots: the status box is gone (no job, no section to show, no term tooltips of its own); version 27's
    # changed statements are shot instead, and the page must hold no part of the box
    ("""sub(CT / "tools" / "text_shots.mjs",
    "    out[name].tips = {};\\n",
    "    out[name].tips = {};\\n"
    "    await evaluate(pg, `(() => { if (window.nvGo) window.nvGo('head-status'); return 1; })()`); await sleep(900);   // integration: the status box's section\\n", 1)""",
     """sub(CT / "tools" / "text_shots.mjs",
    "    out[name].tips = {};\\n",
    "    out[name].tips = {};\\n"
    "    // version 27: no part of the status box is on the page (the owner asked for it to be removed)\\n"
    "    out[name].noStatusBox = await evaluate(pg, `!document.getElementById('head-status') && !document.querySelector('ul.g-status') && !document.querySelector('a[href=\\"#head-status\\"]')`);\\n", 1)"""),
    ('wrap(LIVE4 / "core_4933.html", CT / "work" / "v25_preview.html")   # rendered as the learning curves\' reference (render_v25.json)',
     'wrap(LIVE4 / "core_92ba.html", CT / "work" / "l92ba_preview.html")   # rendered as the learning curves\' reference (render_l92ba.json)'),
    ("""# version 26: the first claim's status is dated 16 Oct; the status box's new line (version 25's rerun of BCA and BCA IW)
# is shot, and its BCA IW tooltip opened, with the other new tooltips""",
     """# version 27: the first claim's status is dated 16 Oct; the status box is gone: its jobs and its term tooltips go, and
# Stage 2's outcome and "What happened to this plan" (changed by version 27) are shot"""),
    ("""sub(CT / "tools" / "text_shots.mjs",
    '  ["results", () => `[...document.querySelectorAll("#results > p")].slice(0, 1)`],\\n',
    '  ["results", () => `[...document.querySelectorAll("#results > p")].slice(0, 1)`],\\n'
    '  ["rerunline", () => `[...document.querySelectorAll("#head-status ul.g-status > li")].filter(li => /BCA and BCA IW/.test(li.textContent))`],   // version 26\\n', 1)
sub(CT / "tools" / "text_shots.mjs", 'for (const t of ["CQL", "IQL", "ReBRAC", "Unifloral", "Weighted BCA"]) {',
    'for (const t of ["CQL", "IQL", "ReBRAC", "Unifloral", "Weighted BCA", "BCA IW"]) {   // version 26: BCA IW', 1)""",
     """sub(CT / "tools" / "text_shots.mjs",
    '  ["statusend", () => `[...document.querySelectorAll("#head-status ul.g-status > li")].slice(-3)`],\\n',
    '', 1)
sub(CT / "tools" / "text_shots.mjs",
    '  ["results", () => `[...document.querySelectorAll("#results > p")].slice(0, 1)`],\\n',
    '  ["results", () => `[...document.querySelectorAll("#results > p")].slice(0, 1)`],\\n'
    '  ["outcome", () => `[...document.querySelectorAll("dd")].filter(d => /Seeds 5–9 were to decide/.test(d.textContent)).slice(0, 1)`],   // version 27\\n'
    '  ["plan", () => `[...document.querySelectorAll("p")].filter(p => /^What happened to this plan/.test(p.textContent.trim())).slice(0, 1)`],   // version 27\\n', 1)
sub(CT / "tools" / "text_shots.mjs", 'for (const t of ["CQL", "IQL", "ReBRAC", "Unifloral", "Weighted BCA"]) {',
    'for (const t of []) {   // version 27: the status box, and with it its term tooltips, is gone', 1)"""),
])

# ------------------------------------------------------------------ static checks
make("static_v26.py", "static_v27.py", [
    ('''"""Test only: static checks of the integrated page (version 26) against its base, version 25 (static_v20.py did the same
for version 20 against version 19, static_v24.py for version 24 against version 23; made by make_tests_v26.py).''',
     '''"""Test only: static checks of the integrated page (version 27) against its base, the live page after version 25, whose
status box was removed (static_v20.py, static_v24.py and static_v26.py did the same for versions 20, 24 and 26; made by
make_tests_v27.py).'''),
    ("  - the status box's line on version 25's rerun (BCA and BCA IW) states the numbers of the page's own data block;",
     "  - no part of the status box (removed at the owner's request); version 25's rerun data block and the live page's newer\n"
     "    \"Strong test mix\" tooltip are exactly as in the base;"),
    ("Usage: python -X utf8 static_v26.py  -> prints a summary and writes out/static_v26.json",
     "Usage: python -X utf8 static_v27.py  -> prints a summary and writes out/static_v27.json"),
    ('''v19 = (SCR / "live5" / "core_4933.html").read_text(encoding="utf-8")   # the base: version 25 (the name v19 is kept for the code below)
v20 = (BUILD / "core_v26.html").read_text(encoding="utf-8")            # the built page: version 26''',
     '''v19 = (SCR / "live6" / "core_92ba.html").read_text(encoding="utf-8")   # the base: the live page (the name v19 is kept for the code below)
v20 = (BUILD / "core_v27.html").read_text(encoding="utf-8")            # the built page: version 27'''),
    ('''res["absent"] = {s: s not in v20 for s in ('id="wt-open"', 'id="wt"', "Full walkthrough mode</button>", 'class="toc"', "sb-panel",
                                             'aside class="sb"', "wt-stage", 'id="ls-start"', 'id="sb"', "tc-choices-wide", 'id="sb-tab"', "sb-tab-away")}''',
     '''res["absent"] = {s: s not in v20 for s in ('id="wt-open"', 'id="wt"', "Full walkthrough mode</button>", 'class="toc"', "sb-panel",
                                             'aside class="sb"', "wt-stage", 'id="ls-start"', 'id="sb"', "tc-choices-wide", 'id="sb-tab"', "sb-tab-away",
                                             # version 27: the status box, which the owner asked to remove
                                             'id="head-status"', 'id="head-status-h"', '<ul class="g-status"', 'href="#head-status"',
                                             "Status as of ", "Status of every line of work", "<b>Version 26, ")}'''),
    ('''"<b>Version 26, 16 Oct 2026.</b>", "(page version 26)", 'id="simtog"',''',
     '''"<b>Version 27, 16 Oct 2026.</b>", 'id="simtog"','''),
])
# the rerun line's check (the status box) becomes the version-27 checks
s = (T / "static_v27.py").read_text(encoding="utf-8")
a = s.index("# ------------------------------------------------------------------ version 26: the status box's rerun line against the page's data")
b = s.index("ok = (not res[\"external_new\"]")
s = s[:a] + '''# ------------------------------------------------------------------ version 27: no status box; the live page's own parts kept
_rd = lambda x: re.search(r'<script type="application/json" id="sim-rerun-data">(.*?)</script>', x, re.S).group(1)
res["rerun_data_same"] = _rd(v20) == _rd(v19)
# the live page's "Strong test mix" tooltips (three: the switch bar's button, rewritten after version 25, and two older ones)
_st = re.findall(r'data-tip-title="Strong test mix" data-tip="[^"]*"', v19)
_new = [x for x in _st if "What makes it strong" in x]
res["strong_mix_tips"] = len(_st)
res["strong_mix_tip"] = _new[0][:400] if _new else None
res["strong_mix_tip_same"] = len(_new) == 1 and all(v20.count(x) == v19.count(x) for x in _st)
_g = lambda x: x[x.index("</p>", x.index('<p class="g-one">')) + 4:x.index('<p class="pipe-h" id="pipe-h">')]
res["header_as_live"] = _g(v20) == _g(v19) and not _g(v20).strip()

''' + s[b:]
s = s.replace('''      and all(res["absent"].values()) and all(n == 1 for n in res["present_once"].values()) and res["rerun_line_ok"])''',
              '''      and all(res["absent"].values()) and all(n == 1 for n in res["present_once"].values()) and res["rerun_data_same"]
      and res["strong_mix_tip_same"] and res["header_as_live"])''', 1)
s = s.replace('(HERE / "out" / "static_v26.json").write_text(', '(HERE / "out" / "static_v27.json").write_text(', 1)
s = s.replace('print("static_v26 ok:", ok)', 'print("static_v27 ok:", ok)', 1)
assert "rerun_line" not in s and "static_v26.json" not in s and 'print("static_v26 ok' not in s and 'res["header_as_live"])' in s
(T / "static_v27.py").write_text(s, encoding="utf-8", newline="\n")

# ------------------------------------------------------------------ the old full check
make("int/check3_v26.mjs", "int/check3_v27.mjs", [
    ("// Test only: the old full check (final_look/check3.mjs) adapted to the new navigation, for page version 26 (check3_v20.mjs\n"
     "// and check3_v24.mjs did the same for versions 20 and 24; here the pages, the reference and the version strings\n"
     "// changed; made by make_tests_v26.py). For version 26 at 390x844",
     "// Test only: the old full check (final_look/check3.mjs) adapted to the new navigation, for page version 27 (check3_v20.mjs,\n"
     "// check3_v24.mjs and check3_v26.mjs did the same for versions 20, 24 and 26; here the pages, the reference, the version\n"
     "// strings and the status heading (none: the status box is gone) changed; made by make_tests_v27.py). For version 27 at 390x844"),
    ("// The same whole-page probes on version 25, the base (no navigation), are the reference. Writes out/check3_v26.json.",
     "// The same whole-page probes on the live page, the base (no navigation), are the reference. Writes out/check3_v27.json."),
    ("    v20entry: t.includes('Version 26, 16 Oct 2026.') && t.includes('Version 25, 16 Oct 2026.') && ['24', '23', '22', '21', '20'].every(v => t.includes('Version ' + v + ', 15 Oct 2026.')),   // version 26's entry, and versions 20-25's kept",
     "    v20entry: t.includes('Version 27, 16 Oct 2026.') && !t.includes('Version 26, 16 Oct 2026.') && t.includes('Version 25, 16 Oct 2026.') && ['24', '23', '22', '21', '20'].every(v => t.includes('Version ' + v + ', 15 Oct 2026.')),   // version 27's entry (none for version 26), and versions 20-25's kept"),
    ("launch({ port: 12791, logName: 'edge_check3_v26.log' })", "launch({ port: 12791, logName: 'edge_check3_v27.log' })"),
    ("    await navigate(cdp, pg, 'file:///' + B + '/page_v26_preview.html', { settle: 3500 });",
     "    await navigate(cdp, pg, 'file:///' + B + '/page_v27_preview.html', { settle: 3500 });"),
    ("    await navigate(cdp, pg, 'file:///' + B + '/tests/nav/build/v25_live.html', { settle: 3500 });",
     "    await navigate(cdp, pg, 'file:///' + B + '/tests/nav/build/l92ba_live.html', { settle: 3500 });"),
    ("fs.writeFileSync(B + '/tests/int/out/check3_v26.json', JSON.stringify(out, null, 1));", "fs.writeFileSync(B + '/tests/int/out/check3_v27.json', JSON.stringify(out, null, 1));", 2),
    ("&& r.whole.v20entry && /page version 26/.test(r.whole.head || '')", "&& r.whole.v20entry && r.whole.head === null && out.v19[n].head === null"),
    ("console.log('check3_v26 ok:', ok);", "console.log('check3_v27 ok:', ok);"),
])

# ------------------------------------------------------------------ timing
make("timing_v26.mjs", "timing_v27.mjs", [
    ("// Load time of page version 26 against version 25 (test only; made by make_tests_v26.py from timing_v24.mjs): 390x844,\n"
     "// touch, the CPU slowed 4 times, alternating loads (version 25, then version 26, N times each, a fresh tab each time).",
     "// Load time of page version 27 against the live page (test only; made by make_tests_v27.py from timing_v26.mjs): 390x844,\n"
     "// touch, the CPU slowed 4 times, alternating loads (the live page, then version 27, N times each, a fresh tab each time)."),
    ("the navigation's own start (window.__nvInitMs, version 26 only)", "the navigation's own start (window.__nvInitMs, version 27 only)"),
    ("// Usage: node timing_v26.mjs [N=5] -> out/timing_v26.json", "// Usage: node timing_v27.mjs [N=5] -> out/timing_v27.json"),
    ("const PAGES = { v25: 'file:///' + T + '/regression/v25_live.html', v26: 'file:///' + T + '/../page_v26_preview.html' };\n"
     "if (!fs.existsSync(T + '/regression/v25_live.html')) {",
     "const PAGES = { live: 'file:///' + T + '/regression/l92ba_live.html', v27: 'file:///' + T + '/../page_v27_preview.html' };\n"
     "if (!fs.existsSync(T + '/regression/l92ba_live.html')) {"),
    ("  fs.writeFileSync(T + '/regression/v25_live.html', Buffer.concat([fs.readFileSync(SCR + '/live5/live_4933.html').subarray(0, 531),\n"
     "    fs.readFileSync(SCR + '/live5/core_4933.html'), Buffer.from('\\n</body></html>')]));",
     "  fs.writeFileSync(T + '/regression/l92ba_live.html', Buffer.concat([fs.readFileSync(SCR + '/live6/live_92ba.html').subarray(0, 531),\n"
     "    fs.readFileSync(SCR + '/live6/core_92ba.html'), Buffer.from('\\n</body></html>')]));"),
    ("const runs = { v25: [], v26: [] };", "const runs = { live: [], v27: [] };"),
    ("launch({ port: 13961, logName: 'edge_timing_v26.log' })", "launch({ port: 13961, logName: 'edge_timing_v27.log' })"),
    ("  for (let k = 0; k < N; k++) for (const v of ['v25', 'v26']) {", "  for (let k = 0; k < N; k++) for (const v of ['live', 'v27']) {"),
    ("medians: { v25: med('v25'), v26: med('v26') }, runs };", "medians: { live: med('live'), v27: med('v27') }, runs };"),
    ("out.medians.dclRatio = +(out.medians.v26.dcl / out.medians.v25.dcl).toFixed(2);\n"
     "out.medians.loadRatio = +(out.medians.v26.load / out.medians.v25.load).toFixed(2);",
     "out.medians.dclRatio = +(out.medians.v27.dcl / out.medians.live.dcl).toFixed(2);\n"
     "out.medians.loadRatio = +(out.medians.v27.load / out.medians.live.load).toFixed(2);"),
    ("fs.writeFileSync(T + '/out/timing_v26.json', JSON.stringify(out, null, 1));", "fs.writeFileSync(T + '/out/timing_v27.json', JSON.stringify(out, null, 1));"),
])

# ------------------------------------------------------------------ collect
make("collect_v26.py", "collect_v27.py", [
    ('''result file was written, so that nothing from an earlier run is counted. Writes out/results_v26.json (version 26;
collect_results.py and collect_v24.py are the same for versions 20 and 24; made by make_tests_v26.py).''',
     '''result file was written, so that nothing from an earlier run is counted. Writes out/results_v27.json (version 27;
collect_results.py, collect_v24.py and collect_v26.py are the same for versions 20, 24 and 26; made by make_tests_v27.py).'''),
    ('log = (T / "run_v26.log").read_text(encoding="utf-8")', 'log = (T / "run_v27.log").read_text(encoding="utf-8")'),
    ('st = load("out/static_v26.json")', 'st = load("out/static_v27.json")'),
    ('R["static: page contract, ids, links, spelling, old navigation gone, the rerun line against the page\'s data"] = st["ok"]',
     'R["static: page contract, ids, links, spelling, old navigation gone, no status box, the live page\'s data and tooltip kept"] = st["ok"]'),
    ('rep = load("lesson/test/report_v26.json")', 'rep = load("lesson/test/report_v27.json")'),
    ('ex = json.loads(load("lesson/test/extra_v26_out.txt"))', 'ex = json.loads(load("lesson/test/extra_v27_out.txt"))'),
    ('ver = load("color/work/verify_v26int.txt")', 'ver = load("color/work/verify_v27int.txt")'),
    ('tx = load("color/work/text_v26int.json")', 'tx = load("color/work/text_v27int.json")'),
    ('''    okt = okt and not v["console"] and all(isinstance(x, dict) and x["inside"] for x in v["tips"].values()) and all(v[j] != "missing" for j in ("statusend", "atsstatus", "claim1", "results", "rerunline")) and "BCA IW" in v["tips"]
R["color: edited text found (the rerun line too) and new term tooltips (BCA IW too) open inside the screen (1280, 390)"] = okt''',
     '''    okt = okt and not v["console"] and all(v[j] != "missing" for j in ("atsstatus", "claim1", "results", "outcome", "plan")) and v["noStatusBox"] is True
R["color: the edited status text found (After the stop's box, the first claim, Results' opening, Stage 2's outcome, the plan), no status box (1280, 390)"] = okt'''),
    ('c3 = load("int/out/check3_v26.json")', 'c3 = load("int/out/check3_v27.json")'),
    ('''# the regression test against version 25
rg = load("regression/regression_v25.json")
R["regression against version 25: (a) ids %s, (b) lines %s, (c) features in 4 views %s" % (rg["a"]["ok"], rg["b"]["ok"], all(x["ok"] for x in rg["c"].values()))] = rg["ok"] is True
tm = load("out/timing_v26.json")''',
     '''# the regression test against the live page
rg = load("regression/regression_live.json")
R["regression against the live page: (a) ids %s, (b) lines %s, (c) features in 4 views %s" % (rg["a"]["ok"], rg["b"]["ok"], all(x["ok"] for x in rg["c"].values()))] = rg["ok"] is True
tm = load("out/timing_v27.json")'''),
    ('perf = load("lesson/test/perf_v26_out.txt").strip().splitlines()[-1]', 'perf = load("lesson/test/perf_v27_out.txt").strip().splitlines()[-1]'),
    ('(T / "out" / "results_v26.json").write_text(', '(T / "out" / "results_v27.json").write_text('),
])

# ------------------------------------------------------------------ the run script
make("run_v26.sh", "run_v27.sh", [
    ("# Test only (page version 26, built on version 25; made by make_tests_v26.py from run_v24.sh): every component's suite on\n"
     "# the integrated page, the old full check, the regression test against version 25, then the timing tests last (nothing\n"
     "# else running). Pages are set up by setup_tests_v26.py. One headless Edge at a time (each test launches its own and\n"
     "# closes it). Usage: sh run_v26.sh",
     "# Test only (page version 27, built on the live page after version 25; made by make_tests_v27.py from run_v26.sh): every\n"
     "# component's suite on the integrated page, the old full check, the regression test against the live page, then the\n"
     "# timing tests last (nothing else running). Pages are set up by setup_tests_v27.py. One headless Edge at a time (each\n"
     "# test launches its own and closes it). Usage: sh run_v27.sh"),
    ("sha256sum $T/../core_v26.html $T/../page_v26_preview.html $T/nav/build/nav_lesson_live.html $T/lesson/work/v26_core.html",
     "sha256sum $T/../core_v27.html $T/../page_v27_preview.html $T/nav/build/nav_lesson_live.html $T/lesson/work/v27_core.html"),
    ("cd $T && $PY -X utf8 static_v26.py > $T/out/static_v26.txt 2>&1; echo \"static exit $?\"",
     "cd $T && $PY -X utf8 static_v27.py > $T/out/static_v27.txt 2>&1; echo \"static exit $?\""),
    ("node test_full.mjs $T/lesson/work/v26_core.html v26 > $T/lesson/test/full_v26_out.txt 2>&1",
     "node test_full.mjs $T/lesson/work/v27_core.html v27 > $T/lesson/test/full_v27_out.txt 2>&1"),
    ("node test_extra.mjs > $T/lesson/test/extra_v26_out.txt 2>&1", "node test_extra.mjs > $T/lesson/test/extra_v27_out.txt 2>&1"),
    ("# 2. color: version 25 rendered as the learning curves' reference;", "# 2. color: the live page rendered as the learning curves' reference;"),
    ("PAGE_FILE=$T/color/work/v25_preview.html TAG=v25 SHOTS=0 RUNS=d1280_light,d1280_dark,m390_light,m390_dark timeout 3600 node render_test.mjs > $T/color/work/log_v25.txt 2>&1; echo \"color render v25 exit $?\"",
     "PAGE_FILE=$T/color/work/l92ba_preview.html TAG=l92ba SHOTS=0 RUNS=d1280_light,d1280_dark,m390_light,m390_dark timeout 3600 node render_test.mjs > $T/color/work/log_l92ba.txt 2>&1; echo \"color render live exit $?\""),
    ("PAGE_FILE=$T/../page_v26_preview.html TAG=v26int RUNS=d1280_light,d1280_dark,m390_light,m390_dark PRE=\"$PRE\" timeout 3600 node render_test.mjs > $T/color/work/log_v26int.txt 2>&1",
     "PAGE_FILE=$T/../page_v27_preview.html TAG=v27int RUNS=d1280_light,d1280_dark,m390_light,m390_dark PRE=\"$PRE\" timeout 3600 node render_test.mjs > $T/color/work/log_v27int.txt 2>&1"),
    ("node verify_render.mjs $T/color/work/render_v26int.json $T/color/work/render_v25.json > $T/color/work/verify_v26int.txt 2>&1",
     "node verify_render.mjs $T/color/work/render_v27int.json $T/color/work/render_l92ba.json > $T/color/work/verify_v27int.txt 2>&1"),
    ("PAGE_FILE=$T/../page_v26_preview.html TAG=v26int PRE=\"$PRE\" timeout 1800 node text_shots.mjs > $T/color/work/log_text_v26int.txt 2>&1",
     "PAGE_FILE=$T/../page_v27_preview.html TAG=v27int PRE=\"$PRE\" timeout 1800 node text_shots.mjs > $T/color/work/log_text_v27int.txt 2>&1"),
    ("cd $T/int && timeout 1800 node check3_v26.mjs > $T/int/out/check3_v26.log 2>&1", "cd $T/int && timeout 1800 node check3_v27.mjs > $T/int/out/check3_v27.log 2>&1"),
    ("# 4. the regression test against version 25 (ids, lines, features in 4 views)\n"
     "cd $T && timeout 3600 node regression_v25.mjs > $T/regression/regression_v25.log 2>&1; echo \"regression exit $?\"",
     "# 4. the regression test against the live page (ids, lines, features in 4 views)\n"
     "cd $T && timeout 3600 node regression_live.mjs > $T/regression/regression_live.log 2>&1; echo \"regression exit $?\""),
    ("# 6. timing, alone: the navigation's T-18, the load-time comparison with version 25, the lesson's load cost",
     "# 6. timing, alone: the navigation's T-18, the load-time comparison with the live page, the lesson's load cost"),
    ("cd $T && timeout 1800 node timing_v26.mjs 5 > $T/out/timing_v26.log 2>&1", "cd $T && timeout 1800 node timing_v27.mjs 5 > $T/out/timing_v27.log 2>&1"),
    ("V23=C:/Users/David/AppData/Local/Temp/claude/C--Users-David-dev-BCA/3bdab4f0-86ee-439c-a05a-34828d6b0a71/scratchpad/live5/core_4933.html\n"
     "V24=C:/Users/David/AppData/Local/Temp/claude/C--Users-David-dev-BCA/3bdab4f0-86ee-439c-a05a-34828d6b0a71/scratchpad/wf7/build/tests/lesson/work/v26_core.html",
     "V23=C:/Users/David/AppData/Local/Temp/claude/C--Users-David-dev-BCA/3bdab4f0-86ee-439c-a05a-34828d6b0a71/scratchpad/live6/core_92ba.html\n"
     "V24=C:/Users/David/AppData/Local/Temp/claude/C--Users-David-dev-BCA/3bdab4f0-86ee-439c-a05a-34828d6b0a71/scratchpad/wf7/build/tests/lesson/work/v27_core.html"),
    ("RUNS=\"[[\\\"v25 a\\\",\\\"$V23\\\"],[\\\"v26 a\\\",\\\"$V24\\\"],[\\\"v25 b\\\",\\\"$V23\\\"],[\\\"v26 b\\\",\\\"$V24\\\"]]\" timeout 1800 node perf.mjs > $T/lesson/test/perf_v26_out.txt 2>&1",
     "RUNS=\"[[\\\"live a\\\",\\\"$V23\\\"],[\\\"v27 a\\\",\\\"$V24\\\"],[\\\"live b\\\",\\\"$V23\\\"],[\\\"v27 b\\\",\\\"$V24\\\"]]\" timeout 1800 node perf.mjs > $T/lesson/test/perf_v27_out.txt 2>&1"),
])

# ------------------------------------------------------------------ the regression test against the live page
REG_HEAD_OLD = (T / "regression_v25.mjs").read_text(encoding="utf-8").split("import fs from 'node:fs';")[0]
REG_HEAD_NEW = """// Regression test of page version 27 against the live page (version 25 with its status box removed at the owner's request,
// service version 1792156350-92ba; test only: it reads the pages and writes only under wf7/build/tests/regression/). Made
// by make_tests_v27.py from regression_v25.mjs (the test of version 26 against version 25); three parts:
//  (a) every element id of the live page exists in version 27, except the ones this build removes on purpose (REMOVED,
//      each with its reason);
//  (b) every line that versions 20-25 and the live page's later edits added (the lines of the live page that version 19
//      does not have, counted with their repeats) is in version 27 unchanged, once the attributes this build adds to an
//      opening tag are set aside (an id and a short label from the navigation's static edits, a color key from the color
//      patch); a line that is not is allowed only when it is one of the deliberate changes listed in REBASE.md (CHANGED,
//      each with its reason and a check that its replacement is there);
//  (c) in headless Edge at 390x844 and 1280x900, light and dark, the features of versions 20-25 work in version 27 as on
//      the live page: Who drives (the learned actor) for two data qualities and both test mixes; version 25's Learner
//      switch for all three learners (TD3+BC, BCA, BCA IW) on both mixes, through the panel and through the Choices,
//      with the same "Now playing", readouts, explanation (the Learner line included), legend tooltips and cost chart
//      label as the live page, the cost readout's numbers equal to the live page's with a color key per line, and with
//      all the data BCA and BCA IW disabled in both; the live page's newer "Strong test mix" tooltip (the same text and
//      lines); version 24's axis titles (the canvas titles drawn on the cost chart and the gauges, the SVG charts' titles
//      inside their charts, the same number of titles as the live page, each one of the page's own wordings); the switch
//      bar, the Observation Space range bars and their tooltips, the arm tables' tooltips beside their cells, the
//      learning curves' table and its selectors, version 23's notes and column tooltips, and the Choices rows (with "not
//      in use now"); no status box on either page, the header going from its opening paragraph to "The process, step by
//      step" on both, and the navigation's Contents and "In this section" listing no status box; no console errors, no
//      sideways page scroll.
// Usage: node regression_live.mjs   -> regression/regression_live.json, exit 0 when everything passes
"""
reg_pairs = [
    (REG_HEAD_OLD, REG_HEAD_NEW),
    ("const V19 = read(SCR + '/live3/core_9369.html'), V25 = read(SCR + '/live5/core_4933.html'), V26 = read(W7 + '/build/core_v26.html');",
     "const V19 = read(SCR + '/live3/core_9369.html'), VL = read(SCR + '/live6/core_92ba.html'), V27 = read(W7 + '/build/core_v27.html');"),
    # the status box's deliberate changes are gone with the box (R1-R3)
    ("""  { id: 'R1', why: 'B1: the status box gets one status time and page version 26', test: l => l.includes('<h2 id="head-status-h">Status as of 16 Oct 2026 (page version 25)</h2>'),
    repl: ['<h2 id="head-status-h">Status as of 16 Oct 2026, 12:39 UTC (page version 26)</h2>'] },
  { id: 'R2', why: 'B2: Stage 2 version 7 was approved at 15:45 UTC on 15 Oct and the approval checked and recorded at 02:27 UTC on 16 Oct; "waits for the owner\\'s approval" was out of date', test: l => l.includes("plan version 7</i>) waits for the owner's approval"),
    repl: ["plan version 7</i>) is cleared to start: the owner typed their approval at their own", 'and it was checked and recorded on 16 Oct at 02:27 UTC; no result of the run is in the repository yet'] },
  { id: 'R3', why: 'B20: the weighting test\\'s code, specification and exploratory results are in the shared repository', test: l => l === '          (<a href="#weighting-test">The weighting test</a>).</li>',
    repl: ['shared repository since 15 Oct at 17:18 UTC; its run records are not.'] },
""",
     """  // (R1-R3 of version 26 were changes to the status box, which the live page no longer has)
"""),
    ("const CHANGED = [   // version 25 lines that version 26 changes on purpose (REBASE.md, \"Deliberate changes to version 25\")",
     "const CHANGED = [   // live-page lines that version 27 changes on purpose (REBASE.md, part 3, \"Deliberate changes to the live page\")"),
    ("console.log('(a)', JSON.stringify(res.a).slice(0, 600));", "console.log('(a)', JSON.stringify(res.a).slice(0, 600));"),
    ("if (process.env.ONLY_AB) { fs.writeFileSync(OUT + '/regression_v25_ab.json', JSON.stringify(res, null, 1)); process.exit(res.a.ok && res.b.ok ? 0 : 1); }",
     "if (process.env.ONLY_AB) { fs.writeFileSync(OUT + '/regression_live_ab.json', JSON.stringify(res, null, 1)); process.exit(res.a.ok && res.b.ok ? 0 : 1); }"),
    ("const WRAP = fs.readFileSync(SCR + '/live5/live_4933.html').subarray(0, 531);\nconst page25 = OUT + '/v25_live.html';\n"
     "fs.writeFileSync(page25, Buffer.concat([WRAP, Buffer.from(V25), Buffer.from('\\n</body></html>')]));\n"
     "const PAGES = { v25: 'file:///' + page25, v26: 'file:///' + W7 + '/build/page_v26_preview.html' };",
     "const WRAP = fs.readFileSync(SCR + '/live6/live_92ba.html').subarray(0, 531);\nconst pageLive = OUT + '/l92ba_live.html';\n"
     "fs.writeFileSync(pageLive, Buffer.concat([WRAP, Buffer.from(VL), Buffer.from('\\n</body></html>')]));\n"
     "const PAGES = { live: 'file:///' + pageLive, v27: 'file:///' + W7 + '/build/page_v27_preview.html' };"),
    # the tooltip reader also keeps the lines as shown (innerText)
    ("    return { text: t.textContent.replace(/\\\\s+/g, ' ').trim(), read: (t.querySelector('.tt-read') || {}).textContent || null, keys: [...t.querySelectorAll('.key')].map(k => k.style.getPropertyValue('--c')), l: r.left, r: r.right, t: r.top, b: r.bottom }; })()`);",
     "    return { text: t.textContent.replace(/\\\\s+/g, ' ').trim(), inner: t.innerText, read: (t.querySelector('.tt-read') || {}).textContent || null, keys: [...t.querySelectorAll('.key')].map(k => k.style.getPropertyValue('--c')), l: r.left, r: r.right, t: r.top, b: r.bottom }; })()`);"),
    # version 27's own probes: the header, the status box, the Strong test mix tooltip, the rails
    ("""  // Who drives: the learned actor (TD3+BC), two data qualities, both test mixes
  r.learned = {};
  await ev(pg, press('who', 'learned')); await sleep(500);""",
     """  // version 27: no status box on either page; the header goes from its opening paragraph to "The process, step by step"
  r.head = await ev(pg, `(() => { const g = document.querySelector('header#top p.g-one'), n = g && g.nextElementSibling;
    return { next: n ? (n.id || n.tagName) : null, box: !!document.getElementById('head-status'), list: !!document.querySelector('ul.g-status'),
      link: !!document.querySelector('a[href="#head-status"]'), statusText: /Status as of/.test(document.body.textContent) }; })()`);
  // Who drives: the learned actor (TD3+BC), two data qualities, both test mixes
  r.learned = {};
  await ev(pg, press('who', 'learned')); await sleep(500);
  // the live page's newer "Strong test mix" tooltip (rewritten after version 25): its text and its lines as shown
  await show(pg, 'simtog');
  const stt = await hover(pg, '#simtog [data-tog-group="lm"][data-tog-value="strong"]', 0);
  r.strongTip = stt && stt.tip ? { text: stt.tip.text, inner: stt.tip.inner, inside: stt.tip.l >= 0 && stt.tip.r <= view.w } : null;"""),
    ("""  // the page: no sideways scroll, no console errors
  r.sideways = await ev(pg, `document.documentElement.scrollWidth - document.documentElement.clientWidth`);""",
     """  // version 27: the navigation's Contents and "In this section" (on the header's section) list no status box
  if (ver === 'v27') {
    await ev(pg, `window.nvGo && window.nvGo('glance')`); await sleep(700);
    r.rails = await ev(pg, `(() => { const a = [...document.querySelectorAll('#nv-list a, #nv-stops a')];
      return { n: a.length, status: a.filter(x => /head-status/.test(x.getAttribute('href') || '') || /^\\\\s*Status/.test(x.textContent)).map(x => x.textContent.trim()),
        glance: [...document.querySelectorAll('#nv-stops a')].map(x => x.textContent.trim()).slice(0, 8) }; })()`);
  }
  // the page: no sideways scroll, no console errors
  r.sideways = await ev(pg, `document.documentElement.scrollWidth - document.documentElement.clientWidth`);"""),
    ("const { proc, cdp } = await launch({ port: 13951, logName: 'edge_regression_v25.log' });",
     "const { proc, cdp } = await launch({ port: 13951, logName: 'edge_regression_live.log' });"),
    ("""      notes: same(r26.notes, r25.notes) && r26.notes.every(n => n && n.shown) && r26.covHeadTip === r25.covHeadTip
        && (view.w < 600 || !!r26.covHeadTip),   // a phone shows the table stacked, without its group headers (both versions)
    };""",
     """      notes: same(r26.notes, r25.notes) && r26.notes.every(n => n && n.shown) && r26.covHeadTip === r25.covHeadTip
        && (view.w < 600 || !!r26.covHeadTip),   // a phone shows the table stacked, without its group headers (both versions)
      // version 27
      noStatusBox: same(r26.head, r25.head) && r26.head.next === 'pipe-h' && !r26.head.box && !r26.head.list && !r26.head.link && !r26.head.statusText,
      strongTip: !!r26.strongTip && !!r25.strongTip && r26.strongTip.text === r25.strongTip.text && r26.strongTip.inner === r25.strongTip.inner
        && /What makes it strong: currents above 0\\.7 come in about 64% of its episodes/.test(r26.strongTip.text) && r26.strongTip.inside,
      rails: !!r26.rails && r26.rails.n > 20 && !r26.rails.status.length && r26.rails.glance.some(t => /The process, step by step/.test(t)),
    };"""),
    ("res.c[view.name] = { ok: Object.values(chk).every(Boolean), chk, v25: r25, v26: r26 };",
     "res.c[view.name] = { ok: Object.values(chk).every(Boolean), chk, live: r25, v27: r26 };"),
    ("fs.writeFileSync(OUT + '/regression_v25.json', JSON.stringify(res, null, 1));\nconsole.log('regression_v25 ok:', res.ok);",
     "fs.writeFileSync(OUT + '/regression_live.json', JSON.stringify(res, null, 1));\nconsole.log('regression_live ok:', res.ok);"),
]
# the names: version 25 -> the live page (VL, 'live'), version 26 -> version 27 (V27, 'v27'); the result keys r25 and r26
# keep their names inside the script (r25 = the live page's readings, r26 = version 27's)
reg_regex = [
    (r"\bV25\b", "VL", 2), (r"\bV26\b", "V27", 6),
    (r"'v25'", "'live'", 2), (r"'v26'", "'v27'", 12),
    (r"\bi25\b", "iL", 3), (r"\bi26\b", "i27", 5), (r"res\.a\.ids25", "res.a.idsLive", 1), (r"res\.a\.ids26", "res.a.ids27", 1),
    (r"\bdup26\b", "dup27", 2), (r"\bc25\b", "cL", 2), (r"\bL26\b", "L27", 2),
]
make("regression_v25.mjs", "regression_live.mjs", reg_pairs, reg_regex)


# ================================================================== page version 27 with the toy worlds' full results
# (the toy integration, 16 Oct 2026): apply_all.py's first stage is now the toy patch (wf7/toy/apply_toy.py), which
# replaces section#toy-worlds with the full results of the five toy notebooks. The scripts made above are adjusted in
# place, right after they are made (so this file still makes them all from the version-26 scripts in one run):
#   setup_tests_v27.py   the navigation's T-20 checks the static edits' anchors once in the live page after the toy patch
#                        (stages_v27/s0_toy.html; static_edits.json no longer has the five edits in the old toy section);
#                        the toy section's own suite is copied to tests/toy/ (test_toy.mjs, nav_smoke.mjs, cdp.mjs), its
#                        output folder moved there, and test_toy.mjs first shows the section through the navigation
#   static_v27.py        the toy section is the toy patch's fragment byte for byte; its color tokens are defined on :root
#                        and redefined in both dark blocks, and every token it uses is defined; its 13 old ids are kept
#   regression_live.mjs  R10: the live page's toy section (the one the toy patch knows) is replaced as a whole: part (b)
#                        counts the lines outside the section, and a new check finds the fragment in version 27
#   run_v27.sh           the toy section's suite on the integrated page (six views; inside the navigation)
#   collect_v27.py       their results, and R10
def post(dst, pairs):
    p = T / dst
    s = p.read_text(encoding="utf-8")
    for pair in pairs:
        old, new = pair[0], pair[1]
        n = pair[2] if len(pair) > 2 else 1
        c = s.count(old)
        if c != n:
            sys.exit("make_tests_v27.py (toy): %s: %r occurs %d times, expected %d" % (dst, old[:100], c, n))
        s = s.replace(old, new)
    p.write_text(s, encoding="utf-8", newline="\n")
    print("adjusted", dst, "for the toy integration,", len(pairs), "replacements")


TOY_IDS = ("toy-worlds", "toy-why", "toy-world", "toy-first", "toy-second", "toy-shelved", "toy-recheck", "toy-meaning",
           "toy-reward-table", "toy-pooled-table", "toy-shelved-width", "toy-shelved-posterior", "toy-more-detail")

post("setup_tests_v27.py", [
    ("T-20 checks the static edits' anchors once in it; the lesson's suite runs on core_v27.html.",
     "T-20 checks the static edits' anchors once in it (since the toy integration: in it after the toy patch,\n"
     "../stages_v27/s0_toy.html); the lesson's suite runs on core_v27.html; the toy section's own suite is copied to toy/\n"
     "and shows the section through the navigation first."),
    ("# the base is the live page: T-20's anchors are checked once in it\n"
     "sub(NT / \"test\" / \"static_checks.py\", 'base = (LIVE3 / \"core_9369.html\")', 'base = (LIVE3 / \"core_92ba.html\")', 1)",
     "# the base is the live page after the toy patch (apply_all.py's stage 0): T-20's anchors are checked once in it\n"
     "sub(NT / \"test\" / \"static_checks.py\", 'base = (LIVE3 / \"core_9369.html\")', 'base = Path(r\"%s\")' % (BUILD / \"stages_v27\" / \"s0_toy.html\"), 1)"),
    ('print("color suite ready")\n',
     'print("color suite ready")\n'
     '\n'
     '# ------------------------------------------------------------------ the toy section\'s suite (the toy integration)\n'
     'TC = W7 / "toy" / "test"\n'
     'TT = HERE / "toy"\n'
     'TT.mkdir(parents=True, exist_ok=True)\n'
     'for n in ("test_toy.mjs", "nav_smoke.mjs", "cdp.mjs"):\n'
     '    shutil.copy2(TC / n, TT / n)\n'
     'sub(TT / "cdp.mjs", "export const OUTDIR = \'" + SCR_S + "/wf7/toy/test\';", "export const OUTDIR = \'" + SCR_S + "/wf7/build/tests/toy\';", 1)\n'
     '# on the integrated page the navigation shows one section at a time: the test shows the toy section through it first\n'
     'sub(TT / "test_toy.mjs", "    await navigate(cdp, pg, URL, { settle: 3000 });\\n",\n'
     '    "    await navigate(cdp, pg, URL, { settle: 3000 });\\n"\n'
     '    "    await evaluate(pg, `(() => { if (window.nvGo) window.nvGo(\'toy-worlds\'); return 1; })()`); await sleep(1200);   // integration: show the section\\n", 1)\n'
     'print("toy suite ready")\n'),
])

post("static_v27.py", [
    ("  - the walkthrough's and the old navigation's markup is gone, the lesson's L4 and the navigation's start bar agree.\n",
     "  - the walkthrough's and the old navigation's markup is gone, the lesson's L4 and the navigation's start bar agree;\n"
     "  - the toy worlds (the toy integration): the section is the toy patch's fragment byte for byte, its color tokens are\n"
     "    defined on :root and redefined in both dark blocks, every token it uses is defined, and its 13 old ids are kept.\n"),
    ("ok = (not res[\"external_new\"]",
     "# ------------------------------------------------------------------ the toy worlds (the toy integration)\n"
     "_frag = (BUILD.parent / \"toy\" / \"toy_fragment.html\").read_text(encoding=\"utf-8\")\n"
     "_ta = v20.index('<section id=\"toy-worlds\">')\n"
     "_toy = v20[_ta:v20.index(\"</section>\", _ta) + len(\"</section>\")]\n"
     "res[\"toy_is_fragment\"] = _toy == _frag\n"
     "_css = re.search(r\"<style>(.*?)</style>\", _toy, re.S).group(1)\n"
     "_tok = lambda rx: set(re.findall(r\"(--toy-[\\w-]+)\\s*:\", re.search(rx, _css).group(1)))\n"
     "_root = _tok(r\"(?m)^:root \\{([^}]*)\\}\")\n"
     "_d1 = _tok(r'@media \\(prefers-color-scheme: dark\\) \\{ :root:not\\(\\[data-theme=\"light\"\\]\\) \\{([^}]*)\\}')\n"
     "_d2 = _tok(r':root\\[data-theme=\"dark\"\\] \\{([^}]*)\\}')\n"
     "res[\"toy_tokens\"] = {\"root\": sorted(_root), \"dark_media\": sorted(_d1), \"dark_theme\": sorted(_d2)}\n"
     "res[\"toy_tokens_ok\"] = bool(_root) and _root == _d1 == _d2\n"
     "res[\"toy_tokens_undefined\"] = sorted(set(re.findall(r\"var\\((--toy-[\\w-]+)\\)\", _toy)) - _root)\n"
     "res[\"toy_ids_kept\"] = {i: v20.count('id=\"%s\"' % i) for i in " + repr(TOY_IDS) + "}\n"
     "\n"
     "ok = (not res[\"external_new\"]"),
    ('      and res["strong_mix_tip_same"] and res["header_as_live"])',
     '      and res["strong_mix_tip_same"] and res["header_as_live"]\n'
     '      and res["toy_is_fragment"] and res["toy_tokens_ok"] and not res["toy_tokens_undefined"]\n'
     '      and all(n == 1 for n in res["toy_ids_kept"].values()))'),
])

post("regression_live.mjs", [
    ("//      sideways page scroll.\n// Usage: node regression_live.mjs",
     "//      sideways page scroll.\n"
     "// The toy integration (R10): the live page's section#toy-worlds is replaced as a whole by the toy patch (the full results\n"
     "// of the five toy notebooks): part (b) counts the lines outside that section on every page, and a check finds the live\n"
     "// section to be the one the patch knows (SHA-256 88505c7a...) and version 27's to be the patch's fragment byte for byte;\n"
     "// part (a) still asks for every id of the live section.\n"
     "// Usage: node regression_live.mjs"),
    ("const L19 = count(V19.split('\\n'));",
     "// the toy integration (R10): the toy section is replaced as a whole, so lines are counted outside it on every page\n"
     "const noToy = h => { const i = h.indexOf('<section id=\"toy-worlds\">'); return i < 0 ? h : h.slice(0, i) + h.slice(h.indexOf('</section>', i) + 10); };\n"
     "const L19 = count(noToy(V19).split('\\n'));"),
    ("const L27 = count(V27.split('\\n').map(strip));", "const L27 = count(noToy(V27).split('\\n').map(strip));"),
    ("  const cL = count(VL.split('\\n'));", "  const cL = count(noToy(VL).split('\\n'));"),
    ("if (process.env.ONLY_AB) { fs.writeFileSync(OUT + '/regression_live_ab.json', JSON.stringify(res, null, 1)); process.exit(res.a.ok && res.b.ok ? 0 : 1); }",
     "// ------------------------------------------------------------------------------------------------ R10: the toy section\n"
     "{\n"
     "  const crypto = await import('node:crypto');\n"
     "  const sec = h => { const i = h.indexOf('<section id=\"toy-worlds\">'); return h.slice(i, h.indexOf('</section>', i) + 10); };\n"
     "  const hL = crypto.createHash('sha256').update(sec(VL), 'utf8').digest('hex');\n"
     "  const frag = read(W7 + '/toy/toy_fragment.html');\n"
     "  const i27 = staticIds(V27);\n"
     "  res.toy = { id: 'R10', why: 'the toy patch (wf7/toy/apply_toy.py, the first stage of the build) replaces the live page\\'s Toy worlds section (two notebooks) by the full results of all five toy notebooks, at the owner\\'s request (REBASE.md part 4)',\n"
     "    liveSha256: hL.slice(0, 16), liveKnown: hL === '88505c7ac804b7c7e0dc2dc047aeeba396b780fb527bdd0549d7096a7505d05f',\n"
     "    isFragment: sec(V27) === frag, liveChars: sec(VL).length, chars27: sec(V27).length,\n"
     "    liveIdsKept: [...staticIds(sec(VL)).keys()].every(k => i27.has(k)) };\n"
     "  res.toy.ok = res.toy.liveKnown && res.toy.isFragment && res.toy.liveIdsKept;\n"
     "  console.log('(R10 toy section)', JSON.stringify(res.toy));\n"
     "}\n"
     "if (process.env.ONLY_AB) { fs.writeFileSync(OUT + '/regression_live_ab.json', JSON.stringify(res, null, 1)); process.exit(res.a.ok && res.b.ok && res.toy.ok ? 0 : 1); }"),
    ("res.ok = res.a.ok && res.b.ok && Object.values(res.c).every(x => x.ok);",
     "res.ok = res.a.ok && res.b.ok && res.toy.ok && Object.values(res.c).every(x => x.ok);"),
])

post("run_v27.sh", [
    ("# 5. the navigation (all but the timing test)\n",
     "# 4b. the toy section (the toy integration): its own test on the integrated page in six views, and inside the navigation\n"
     "mkdir -p $T/toy\n"
     "cd $T/toy && timeout 1800 node test_toy.mjs $T/../page_v27_preview.html $T/toy/result_v27.json > $T/toy/test_toy_v27.log 2>&1; echo \"toy exit $?\"\n"
     "cd $T/toy && timeout 900 node nav_smoke.mjs $T/../page_v27_preview.html $T/toy/nav_v27.json > $T/toy/nav_smoke_v27.log 2>&1; echo \"toy nav exit $?\"\n"
     "# 5. the navigation (all but the timing test)\n"),
])

post("collect_v27.py", [
    ('R["regression against the live page: (a) ids %s, (b) lines %s, (c) features in 4 views %s" % (rg["a"]["ok"], rg["b"]["ok"], all(x["ok"] for x in rg["c"].values()))] = rg["ok"] is True',
     'R["regression against the live page: (a) ids %s, (b) lines %s, R10 toy section %s, (c) features in 4 views %s" % (rg["a"]["ok"], rg["b"]["ok"], rg["toy"]["ok"], all(x["ok"] for x in rg["c"].values()))] = rg["ok"] is True\n'
     '# the toy section (the toy integration): six views on the integrated page, and inside the navigation\n'
     'ty = load("toy/result_v27.json")\n'
     'for v in ty["views"]:\n'
     '    p = v["probe"]\n'
     '    ok = (p["sectionFound"] and p["docScrollW"] <= p["vw"] and p["overflowCount"] == 0 and p["calcs"] == 0 and p["calcCount"] == 15\n'
     '          and len(p["charts"]) == 4 and all(c["ok"] and c["paths"] >= 3 and c["axisTitles"] == 2 and c["hits"] > 0 for c in p["charts"])\n'
     '          and not p["badLinks"] and not p["dupIds"] and all(t["unlabeled"] == 0 for t in p["tables"]) and not p["small"]\n'
     '          and len(v["tips"]) == 4 and all(t["shown"] and t["keys"] >= 2 and t["inView"] for t in v["tips"]) and not v["errors"])\n'
     '    R["toy section %s: no sideways scroll, 15 worked blocks fit, 4 charts drawn with both axes named, tooltips with a color key per line, links, labels, touch targets" % v["view"]] = ok\n'
     'tn = load("toy/nav_v27.json")\n'
     'R["toy section inside the navigation (1280 light, 390 dark): shown, In this section lists its 44 parts, links into it open their part"] = all(\n'
     '    x["probe"]["shown"] and x["probe"]["railCount"] == 44 and not x["probe"]["badRail"] and x["probe"]["overflowCount"] == 0\n'
     '    and x["probe"]["docScrollW"] <= x["probe"]["vw"] and x["meaning"]["visible"] and x["chart"]["visible"] and x["chart"]["drawn"] and not x["errors"] for x in tn)'),
])

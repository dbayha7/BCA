"""Test only: copy the three components' test suites into wf7/build/tests/<component>, point them at the integrated
page (version 27, built on the live page after version 25, whose status box was removed), and build the extra pages
they compare against. The components' own folders are only read. (setup_tests.py, setup_tests_v24.py and
setup_tests_v26.py are the setups of versions 20, 24 and 26, kept as they were. This file is made by make_tests_v27.py
from setup_tests_v26.py.)

Usage: python -X utf8 setup_tests_v27.py   (after apply_all.py has written ../core_v27.html, ../page_v27_preview.html
and ../stages_v27/)

What changed from setup_tests.py, paths and reference pages only (the tests' checks are unchanged): the base and the
reference page are the live page after version 25 (live6/core_92ba.html, with live6/live_92ba.html's wrapper) instead
of version 19; the navigation's expected lists (T-2) are rules/pass2_v27.json, made by rules/run_rules.mjs on that page
with the static edits (rules/make_with_static.py); its T-7 redraws and T-18 startup compare with it (l92ba_live.html);
T-20 checks the static edits' anchors once in it (since the toy integration: in it after the toy patch,
../stages_v27/s0_toy.html); the lesson's suite runs on core_v27.html; the toy section's own suite is copied to toy/
and shows the section through the navigation first.

Pages (all with the publish wrapper where a browser loads them):
  nav/build/nav_lesson_live.html    the integrated page (= ../page_v27_preview.html)
  nav/build/nav_live.html           color + navigation, without the lesson (the navigation's T-17 page)
  nav/build/l92ba_live.html         the live page (reference for redraws and startup time)
  nav/build/nav_core.html           color + navigation core (static checks)
  nav/build/nav_lesson_core.html    color + navigation + lesson core, before the integration edits
  nav/build/lesson_then_nav_core.html  color, then the lesson, then the navigation (order independence)
  lesson/work/color_lesson_core.html   color + lesson, without the navigation (the lesson's own pager)
"""
import html
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent            # wf7/build/tests
BUILD = HERE.parent
W7 = BUILD.parent
SCR = W7.parent
SCR_S = "C:/Users/David/AppData/Local/Temp/claude/C--Users-David-dev-BCA/3bdab4f0-86ee-439c-a05a-34828d6b0a71/scratchpad"   # as the tests spell it
LIVE4 = SCR / "live6"   # the live page after version 25 (the name LIVE4 is kept for the code below)
WRAP = (LIVE4 / "live_92ba.html").read_bytes()[:531]
TAIL = b"\n</body></html>"
PY = [sys.executable, "-X", "utf8"]


def wrap(core: Path, out: Path):
    b = core.read_bytes()
    assert b"\r" not in b
    out.write_bytes(WRAP + b + TAIL)


def copy_files(src: Path, dst: Path, exts):
    dst.mkdir(parents=True, exist_ok=True)
    for p in src.iterdir():
        if p.is_file() and p.suffix in exts:
            shutil.copy2(p, dst / p.name)


def sub(path: Path, old, new, count=None):
    s = path.read_text(encoding="utf-8")
    n = s.count(old)
    if n == 0 or (count is not None and n != count):
        sys.exit("setup_tests.py: %s: %r found %d times" % (path.name, old[:80], n))
    path.write_text(s.replace(old, new), encoding="utf-8", newline="\n")


def run(args):
    r = subprocess.run(PY + [str(a) for a in args], capture_output=True, text=True, encoding="utf-8",
                       env={**__import__("os").environ, "PYTHONDONTWRITEBYTECODE": "1"})
    if r.returncode:
        sys.exit("setup_tests.py: %s failed:\n%s%s" % (args[0], r.stdout, r.stderr))
    return r.stdout


# ------------------------------------------------------------------ the navigation's suite
NAVC = W7 / "nav"
NT = HERE / "nav"
for d in ("test", "tools", "rules", "src", "build", "shots", "test/out"):
    (NT / d).mkdir(parents=True, exist_ok=True)
copy_files(NAVC / "test", NT / "test", {".mjs", ".py", ".sh"})
copy_files(NAVC / "tools", NT / "tools", {".mjs"})
copy_files(NAVC / "rules", NT / "rules", {".json", ".js", ".mjs"})
copy_files(NAVC / "src", NT / "src", {".css", ".js", ".html", ".json"})
shutil.copy2(NAVC / "apply_nav.py", NT / "apply_nav.py")
old_root = SCR_S + "/wf7/nav"
new_root = SCR_S + "/wf7/build/tests/nav"
sub(NT / "test" / "lib.mjs", "export const ROOT = '" + old_root + "';", "export const ROOT = '" + new_root + "';", 1)
sub(NT / "tools" / "cdp.mjs", old_root + "/", new_root + "/")
sub(NT / "test" / "static_checks.py", 'LIVE3 = NAV.parents[1] / "live3"', 'LIVE3 = Path(r"%s")' % str(LIVE4))
# the base is the live page after the toy patch (apply_all.py's stage 0): T-20's anchors are checked once in it
sub(NT / "test" / "static_checks.py", 'base = (LIVE3 / "core_9369.html")', 'base = Path(r"%s")' % (BUILD / "stages_v27" / "s0_toy.html"), 1)
sub(NT / "test" / "static_checks.py", "static_edit_anchors_once_in_v19", "static_edit_anchors_once_in_base", 2)
# T-2's expected lists: the rules on the live page with the static edits (pass2_v27.json)
sub(NT / "test" / "t_lists.mjs", "'/rules/pass2_v19.json'", "'/rules/pass2_v27.json'", 1)
# T-7 and T-18 compare with the live page
sub(NT / "test" / "t_panels.mjs", "file: 'v19_live.html'", "file: 'l92ba_live.html'", 1)
sub(NT / "test" / "t_timing.mjs", "'v19_live.html'", "'l92ba_live.html'")
sub(NT / "test" / "t_timing.mjs", "v19_dcl_median", "base_dcl_median")
sub(NT / "test" / "summarize.py", 'tim["summary"]["v19_dcl_median"]', 'tim["summary"]["base_dcl_median"]', 1)
sub(NT / "test" / "summarize.py", "DCL %d ms vs version 19 %d ms", "DCL %d ms vs the live page %d ms", 1)
# version 21 capitalized the learning curves' quality buttons ("Medium", "Random"): T-9 finds them whatever the case
for w in ("medium", "random"):
    sub(NT / "test" / "t_panels.mjs", "replace(/^\\\\u2713/, '') === '%s').click();" % w,
        "replace(/^\\\\u2713/, '').toLowerCase() === '%s').click();" % w, 1)
# the lists test: the real lesson's 40 entries (About this lesson and the 39 steps) instead of the stub's 4
frag = (W7 / "lesson" / "lesson_fragment.html").read_text(encoding="utf-8")
steps = re.findall(r'<h3 id="(learn-[a-z0-9-]+)-h"[^>]*>(.*?)</h3>', frag, re.S)
exp_learn = [[1, "top", "About this lesson", "#learn"]] + [[1, "step", html.unescape(re.sub(r"<[^>]+>", "", t)).strip(), "#" + i]
                                                           for i, t in steps]
assert len(exp_learn) == 40, len(exp_learn)
(NT / "test" / "expected_learn.json").write_text(json.dumps(exp_learn, ensure_ascii=False, indent=1), encoding="utf-8")
sub(NT / "test" / "t_lists.mjs",
    "expected.learn = [[1, 'top', 'About this lesson', '#learn'], [1, 'step', '1.1 The channel and its parts', '#learn-world'],\n"
    "  [1, 'step', '1.2 Start 0: the 13 numbers', '#learn-state'], [1, 'step', '1.3 Four throttles, and clipping', '#learn-command']];",
    "expected.learn = JSON.parse(fs.readFileSync(ROOT + '/test/expected_learn.json', 'utf8'));   // the real lesson (integration)", 1)
# the run script: the pages are built here, not by build_test.py
(NT / "test" / "run_all.sh").write_text(
    "#!/bin/sh\n# Test only (integration): every test of the navigation on the integrated page; each writes out/<name>.json\n"
    "# and out/<name>.log. The pages are built by ../../setup_tests.py. Usage: sh run_all.sh [tests...]\n"
    "cd \"$(dirname \"$0\")\" || exit 1\nPY=/c/Users/David/dev/BCA/.venv/Scripts/python.exe\nexport PYTHONDONTWRITEBYTECODE=1\n"
    "$PY -X utf8 static_checks.py > out/static_checks.log 2>&1\n"
    "T=\"$*\"; [ -z \"$T\" ] && T=\"t_lists t_links t_walk t_history t_startbar t_deep t_kbd t_panels t_layout t_entries t_shots t_timing\"\n"
    "for t in $T; do\n  echo \"== $t\"\n  timeout 3600 node $t.mjs > out/$t.log 2>&1\n  echo \"exit $?\"\ndone\necho done\n",
    encoding="utf-8", newline="\n")
# T-16: with the real lesson, a page whose navigation did not start lets the lesson page itself (html.ls-own); the
# stub lesson had no script, so the class was empty
sub(NT / "test" / "t_deep.mjs",
    """      small: [...document.querySelector('main').children].filter(n => n.matches('header#top, section[id], figure.panel[id]')).filter(n => n.offsetHeight < 50).map(n => n.id) })`);
    t.neverErrors = errors(pg);""",
    """      small: [...document.querySelector('main').children].filter(n => n.matches('header#top, section[id], figure.panel[id]')).filter(n => n.offsetHeight < 50).map(n => n.id),
      // integration: with the real lesson, a page whose navigation did not start lets the lesson page itself (html.ls-own):
      // one of its pages shown, with the lesson's own Back and Next
      lessonPages: [...document.querySelectorAll('#learn .ls-intro, #learn .ls-step')].filter(p => !p.hasAttribute('hidden') && !p.classList.contains('ls-off') && !p.classList.contains('nv-off') && p.getClientRects().length).length,
      lessonFoot: !!document.querySelector('#learn .lsx-foot a, #learn .ls-foot a') })`);
    t.neverErrors = errors(pg);""", 1)
sub(NT / "test" / "t_deep.mjs",
    """    t.ok = t.forced.cls === '' && t.forced.hidden === 0 && !t.forced.heights.length && t.forced.nvDisplay === 'block' && t.forced.bar === 'none'
      && t.neverStarted.cls === '' && !t.neverStarted.small.length;""",
    """    // integration: '' (the stub lesson has no script) or 'ls-own' (the real lesson pages itself when the navigation is off)
    const plain = c => c === '' || c === 'ls-own';
    t.ok = plain(t.forced.cls) && t.forced.hidden === 0 && !t.forced.heights.length && t.forced.nvDisplay === 'block' && t.forced.bar === 'none'
      && plain(t.neverStarted.cls) && !t.neverStarted.small.length && (t.neverStarted.cls !== 'ls-own' || (t.neverStarted.lessonPages === 1 && t.neverStarted.lessonFoot));""", 1)
# T-15 (version 27): the header's link to Results was in the status box, which the live page removed; the content link
# clicked in the fallback is now the process's first link, and the check expects that link's own section
sub(NT / "test" / "t_deep.mjs",
    """    await ev(pg, `[...document.querySelectorAll('#top a[href="#results"]')].concat([...document.querySelectorAll('#top a[href^="#"]')])[0].click()`); await sleep(500);""",
    """    f.linkTo = await ev(pg, `(() => { const a = [...document.querySelectorAll('#top a[href="#results"]')].concat([...document.querySelectorAll('#top nav.pipe a[href^="#"]')])[0];
      const u = document.getElementById(a.getAttribute('href').slice(1)).closest('main > header#top, main > section[id], main > figure.panel[id]');
      a.click(); return u.id === 'top' ? 'glance' : u.id; })()`); await sleep(500);   // integration (version 27)""", 1)
sub(NT / "test" / "t_deep.mjs", "&& f.link.unit === 'results'", "&& f.linkTo !== 'glance' && f.link.unit === f.linkTo", 1)
# T-13: every lesson page shown in Whole page (40 with the real lesson, 4 with the stub)
sub(NT / "test" / "t_kbd.mjs",
    """      pages: await ev(pg, `[...document.querySelectorAll('#learn .ls-intro, #learn .ls-step')].filter(p => !p.hasAttribute('hidden')).length`),""",
    """      pages: await ev(pg, `[...document.querySelectorAll('#learn .ls-intro, #learn .ls-step')].filter(p => !p.hasAttribute('hidden')).length`),
      allPages: await ev(pg, `document.querySelectorAll('#learn .ls-intro, #learn .ls-step').length`),   // integration: 40 with the real lesson""", 1)
sub(NT / "test" / "t_kbd.mjs", "t.on.pages === 4 &&", "t.on.pages === t.on.allPages &&", 1)
# T-14: printing shows every lesson step, which builds its control and that control's collapsed blocks; "closed after
# printing" is counted over the blocks that existed before printing
sub(NT / "test" / "t_kbd.mjs",
    """    const closed0 = await ev(pg, `document.querySelectorAll('main details:not([open])').length`);""",
    """    const closed0 = await ev(pg, `document.querySelectorAll('main details:not([open])').length`);
    // integration: the real lesson builds a step's control (and its collapsed blocks) when the step is first shown, which
    // printing does; "closed after printing" is counted over the blocks that existed before it
    await ev(pg, `document.querySelectorAll('main details').forEach(d => { d.dataset.tPre = '1'; })`);""", 1)
sub(NT / "test" / "t_kbd.mjs",
    """    t.after = await ev(pg, `({ ...window.nvState(), closed: document.querySelectorAll('main details:not([open])').length })`);""",
    """    t.after = await ev(pg, `({ ...window.nvState(), closed: document.querySelectorAll('main details[data-t-pre]:not([open])').length, closedAll: document.querySelectorAll('main details:not([open])').length })`);""", 1)
(NT / "test" / "build_test.py").unlink(missing_ok=True)
(NT / "test" / "make_stub_lesson.py").unlink(missing_ok=True)

NB = NT / "build"
stages = BUILD / "stages_v27"
shutil.copy2(BUILD / "page_v27_preview.html", NB / "nav_lesson_live.html")
wrap(stages / "s2_color_nav.html", NB / "nav_live.html")
wrap(LIVE4 / "core_92ba.html", NB / "l92ba_live.html")
shutil.copy2(stages / "s2_color_nav.html", NB / "nav_core.html")
shutil.copy2(stages / "s3_color_nav_lesson.html", NB / "nav_lesson_core.html")
run([W7 / "lesson" / "apply_lesson.py", stages / "s1_color.html", NB / "lesson_first_core.html"])
run([W7 / "nav" / "apply_nav.py", NB / "lesson_first_core.html", NB / "lesson_then_nav_core.html"])
same = (NB / "lesson_then_nav_core.html").read_bytes() == (NB / "nav_lesson_core.html").read_bytes()
print("navigation suite ready; after the color patch, lesson then navigation == navigation then lesson:", same)

# ------------------------------------------------------------------ the lesson's suite
LC = W7 / "lesson"
LT = HERE / "lesson"
for d in ("test", "test/edge", "src", "work", "shots"):
    (LT / d).mkdir(parents=True, exist_ok=True)
copy_files(LC / "test", LT / "test", {".mjs", ".js"})
shutil.copy2(LC / "src" / "port.js", LT / "src" / "port.js")
sub(LT / "test" / "lib.mjs", "export const L = '" + SCR_S + "/wf7/lesson';", "export const L = '" + SCR_S + "/wf7/build/tests/lesson';", 1)
sub(LT / "test" / "cdp.mjs", SCR_S + "/wf7/lesson/", SCR_S + "/wf7/build/tests/lesson/")
sub(LT / "test" / "lib.mjs", "SCR + '/live3/live_9369.html'", "SCR + '/live6/live_92ba.html'", 1)   # the live page's wrapper
shutil.copy2(stages / "s1_color.html", LT / "work" / "color_core.html")
run([W7 / "lesson" / "apply_lesson.py", stages / "s1_color.html", LT / "work" / "color_lesson_core.html"])
shutil.copy2(BUILD / "core_v27.html", LT / "work" / "v27_core.html")
# test_extra's two pages: the lesson without the navigation (on the color stage) and the integrated page
sub(LT / "test" / "test_extra.mjs",
    "const pages = { lesson: L + '/work/v19_lesson_core.html', nav: L + '/work/lesson_then_nav_core.html' };",
    "const pages = { lesson: L + '/work/color_lesson_core.html', nav: L + '/work/v27_core.html' };   // integration", 1)
print("lesson suite ready")

# ------------------------------------------------------------------ the color component's suite
CC = W7 / "color" / "work" / "tools"
CT = HERE / "color"
for d in ("tools", "work", "shots"):
    (CT / d).mkdir(parents=True, exist_ok=True)
copy_files(CC, CT / "tools", {".mjs"})
for n in ("render_test.mjs", "text_shots.mjs"):
    sub(CT / "tools" / n, SCR_S + "/wf7/color", SCR_S + "/wf7/build/tests/color")
sub(CT / "tools" / "cdp.mjs", SCR_S + "/wf7/color/", SCR_S + "/wf7/build/tests/color/")
# the text shots look for blocks in several sections: with the navigation they need the Whole page switch on (PRE), as
# render_test.mjs already allows
sub(CT / "tools" / "text_shots.mjs",
    '    await navigate(cdp, pg, "file:///" + PAGE, { settle: 4000 });\n',
    '    await navigate(cdp, pg, "file:///" + PAGE, { settle: 4000 });\n'
    '    if (process.env.PRE) { await evaluate(pg, `(() => { ${process.env.PRE}; return 1; })()`); await sleep(1500); }   // integration: e.g. the Whole page switch\n', 1)
# ... but with the Whole page on and the lesson's 40 pages shown the page is about 297,000 px tall, and Edge's clipped
# screenshot stalls at the After the stop box (y about 222,000 px at device scale 2). So, one section at a time: each
# job first shows its block's section through the navigation's own link function, and the tooltips the header's.
sub(CT / "tools" / "text_shots.mjs",
    "      const r = await evaluate(pg, `(() => { const els = ${q()}.filter(Boolean); if (!els.length) return null;\n",
    "      await evaluate(pg, `(() => { const e = ${q()}.filter(Boolean)[0]; const u = e && e.closest('[id]'); if (u && window.nvGo) window.nvGo(u.id); return 1; })()`); await sleep(900);   // integration: show the block's section\n"
    "      const r = await evaluate(pg, `(() => { const els = ${q()}.filter(Boolean); if (!els.length) return null;\n", 1)
sub(CT / "tools" / "text_shots.mjs",
    "    out[name].tips = {};\n",
    "    out[name].tips = {};\n"
    "    // version 27: no part of the status box is on the page (the owner asked for it to be removed)\n"
    "    out[name].noStatusBox = await evaluate(pg, `!document.getElementById('head-status') && !document.querySelector('ul.g-status') && !document.querySelector('a[href=\"#head-status\"]')`);\n", 1)
for n in ("render_v19.json", "render_v20.json", "render_combined.json", "text_v20.json"):
    shutil.copy2(W7 / "color" / "work" / n, CT / "work" / n)
wrap(LIVE4 / "core_92ba.html", CT / "work" / "l92ba_preview.html")   # rendered as the learning curves' reference (render_l92ba.json)
# version 27: the first claim's status is dated 16 Oct; the status box is gone: its jobs and its term tooltips go, and
# Stage 2's outcome and "What happened to this plan" (changed by version 27) are shot
sub(CT / "tools" / "text_shots.mjs", "/^Status, as of 15 Oct/.test(p.textContent)", "/^Status, as of 1\\d Oct/.test(p.textContent)", 1)
sub(CT / "tools" / "text_shots.mjs",
    '  ["statusend", () => `[...document.querySelectorAll("#head-status ul.g-status > li")].slice(-3)`],\n',
    '', 1)
sub(CT / "tools" / "text_shots.mjs",
    '  ["results", () => `[...document.querySelectorAll("#results > p")].slice(0, 1)`],\n',
    '  ["results", () => `[...document.querySelectorAll("#results > p")].slice(0, 1)`],\n'
    '  ["outcome", () => `[...document.querySelectorAll("dd")].filter(d => /Seeds 5–9 were to decide/.test(d.textContent)).slice(0, 1)`],   // version 27\n'
    '  ["plan", () => `[...document.querySelectorAll("p")].filter(p => /^What happened to this plan/.test(p.textContent.trim())).slice(0, 1)`],   // version 27\n', 1)
sub(CT / "tools" / "text_shots.mjs", 'for (const t of ["CQL", "IQL", "ReBRAC", "Unifloral", "Weighted BCA"]) {',
    'for (const t of []) {   // version 27: the status box, and with it its term tooltips, is gone', 1)
print("color suite ready")

# ------------------------------------------------------------------ the toy section's suite (the toy integration)
TC = W7 / "toy" / "test"
TT = HERE / "toy"
TT.mkdir(parents=True, exist_ok=True)
for n in ("test_toy.mjs", "nav_smoke.mjs", "cdp.mjs"):
    shutil.copy2(TC / n, TT / n)
sub(TT / "cdp.mjs", "export const OUTDIR = '" + SCR_S + "/wf7/toy/test';", "export const OUTDIR = '" + SCR_S + "/wf7/build/tests/toy';", 1)
# on the integrated page the navigation shows one section at a time: the test shows the toy section through it first
sub(TT / "test_toy.mjs", "    await navigate(cdp, pg, URL, { settle: 3000 });\n",
    "    await navigate(cdp, pg, URL, { settle: 3000 });\n"
    "    await evaluate(pg, `(() => { if (window.nvGo) window.nvGo('toy-worlds'); return 1; })()`); await sleep(1200);   // integration: show the section\n", 1)
print("toy suite ready")

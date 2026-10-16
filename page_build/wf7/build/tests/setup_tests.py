"""Test only: copy the three components' test suites into wf7/build/tests/<component>, point them at the integrated
page (version 20), and build the extra pages they compare against. The components' own folders are only read.

Usage: python -X utf8 setup_tests.py   (after apply_all.py has written ../core_v20.html, ../page_v20_preview.html and
../stages/)

Pages (all with the publish wrapper where a browser loads them):
  nav/build/nav_lesson_live.html    the integrated page (= ../page_v20_preview.html)
  nav/build/nav_live.html           color + navigation, without the lesson (the navigation's T-17 page)
  nav/build/v19_live.html           version 19 (reference for redraws and startup time)
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
LIVE3 = SCR / "live3"
WRAP = (LIVE3 / "live_9369.html").read_bytes()[:531]
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
sub(NT / "test" / "static_checks.py", 'LIVE3 = NAV.parents[1] / "live3"', 'LIVE3 = Path(r"%s")' % str(LIVE3))
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
stages = BUILD / "stages"
shutil.copy2(BUILD / "page_v20_preview.html", NB / "nav_lesson_live.html")
wrap(stages / "s2_color_nav.html", NB / "nav_live.html")
wrap(LIVE3 / "core_9369.html", NB / "v19_live.html")
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
shutil.copy2(stages / "s1_color.html", LT / "work" / "color_core.html")
run([W7 / "lesson" / "apply_lesson.py", stages / "s1_color.html", LT / "work" / "color_lesson_core.html"])
shutil.copy2(BUILD / "core_v20.html", LT / "work" / "v20_core.html")
# test_extra's two pages: the lesson without the navigation (on the color stage) and the integrated page
sub(LT / "test" / "test_extra.mjs",
    "const pages = { lesson: L + '/work/v19_lesson_core.html', nav: L + '/work/lesson_then_nav_core.html' };",
    "const pages = { lesson: L + '/work/color_lesson_core.html', nav: L + '/work/v20_core.html' };   // integration", 1)
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
    "    await evaluate(pg, `(() => { if (window.nvGo) window.nvGo('head-status'); return 1; })()`); await sleep(900);   // integration: the status box's section\n", 1)
for n in ("render_v19.json", "render_v20.json", "render_combined.json", "text_v20.json"):
    shutil.copy2(W7 / "color" / "work" / n, CT / "work" / n)
wrap(LIVE3 / "core_9369.html", CT / "work" / "v19_preview.html")
print("color suite ready")

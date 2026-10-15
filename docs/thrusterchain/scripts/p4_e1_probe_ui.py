#!/usr/bin/env python3
"""Read-only browser probe for pass 4, edit 1: the shared tooltip on new terms and options, the drawing frame and
option colours, the strip's layout at several widths, and the glance inside the walkthrough pop-out (screenshot).
Run: /opt/shot-venv/bin/python scripts/p4_e1_probe_ui.py [--page PATH]
"""
import argparse
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from shot import wrap, DEFAULT_PAGE  # noqa: E402
from playwright.sync_api import sync_playwright  # noqa: E402

OUT = "/mnt/c/Users/David/dev/bca-shift-work/page/tmp/p4/edit1/shots"
ap = argparse.ArgumentParser()
ap.add_argument("--page", default=DEFAULT_PAGE)
a = ap.parse_args()
frag = open(a.page, encoding="utf-8").read()
tmp = os.path.join(tempfile.mkdtemp(prefix="ui-"), "page.html")
open(tmp, "w", encoding="utf-8").write(wrap(frag, "light"))
fails = []


def check(ok, msg):
    print(("  ok   " if ok else "  FAIL ") + msg)
    if not ok:
        fails.append(msg)


with sync_playwright() as p:
    br = p.chromium.launch()
    ctx = br.new_context(viewport={"width": 1200, "height": 900}, color_scheme="light")
    pg = ctx.new_page()
    errs = []
    pg.on("console", lambda m: errs.append(m.text) if m.type in ("error", "warning") else None)
    pg.on("pageerror", lambda e: errs.append(str(e)))
    pg.goto("file://" + tmp)
    pg.wait_for_timeout(900)
    # tooltips on new terms and on the options
    for sel, want in [('#glance i[data-tip-title="80% hold"]', "densest cluster"),
                      ('#glance i[data-tip-title="Capped pull"]', "constrains behaviour"),
                      ('#glance i[data-tip-title="Lower edge"]', "value-pessimism"),
                      ('#opt-sit-strong_current', "dynamics extrapolation"),
                      ('#opt-world-no_stall', "a control for step 2"),
                      ('#opt-ctrl-recording', "In the process:"),
                      ('#opt-ctrl-no_throttle', "no step of the process")]:
        pg.hover(sel)
        pg.wait_for_timeout(350)
        tt = pg.evaluate("() => { const t = document.querySelector('.tt'); return t && !t.hidden && t.classList.contains('on') ? t.textContent : ''; }")
        check(want in tt, f"tooltip on {sel}: {tt[:110]!r}")
        pg.mouse.move(5, 5)
        pg.wait_for_timeout(150)
    # keyboard focus on a new term shows its tooltip
    pg.focus('#glance i[data-tip-title="BCA&#x27;s band"], #glance i[data-tip-title="BCA\'s band"]')
    pg.wait_for_timeout(300)
    tt = pg.evaluate("() => { const t = document.querySelector('.tt'); return t && t.classList.contains('on') ? t.textContent : ''; }")
    check("interval around the critic" in tt, f"focus shows the band's tooltip: {tt[:80]!r}")
    # the frame and the option colours follow the choice
    for sit, var in [("strong_current", "--sit-strong"), ("big_waves", "--sit-waves"), ("big_kicks", "--sit-kicks"), ("recorded", "--sit-rec")]:
        pg.click(f"#opt-sit-{sit}")
        pg.wait_for_timeout(120)
        r = pg.evaluate(f"""() => {{ const c = getComputedStyle(document.getElementById('channel')).outlineColor;
            const p = document.createElement('span'); p.style.color = 'var({var})'; document.body.appendChild(p);
            const w = getComputedStyle(p).color; p.remove(); return [c, w]; }}""")
        check(r[0] == r[1], f"frame colour for {sit}: {r}")
    # the strip: five columns at 1200, one at 390; arrows inside the box
    cols = pg.evaluate("() => new Set([...document.querySelectorAll('#glance .pipe-step')].map(e => Math.round(e.getBoundingClientRect().top))).size")
    check(cols == 1, f"strip at 1200: steps in {cols} row(s)")
    pg.set_viewport_size({"width": 390, "height": 844})
    pg.wait_for_timeout(300)
    lefts = pg.evaluate("() => new Set([...document.querySelectorAll('#glance .pipe-step')].map(e => Math.round(e.getBoundingClientRect().left))).size")
    check(lefts == 1, f"strip at 390: one column ({lefts} distinct left edges)")
    for w in (700, 900, 1000, 1100):
        pg.set_viewport_size({"width": w, "height": 900})
        pg.wait_for_timeout(250)
        info = pg.evaluate("""() => { const n = document.querySelector('#glance nav.pipe');
            const rows = new Set([...n.querySelectorAll('.pipe-step')].map(e => Math.round(e.getBoundingClientRect().top))).size;
            return [Math.round(n.getBoundingClientRect().width), rows, document.documentElement.scrollWidth - innerWidth]; }""")
        print(f"   viewport {w}: strip width {info[0]} px, {info[1]} row(s), page overflow {info[2]} px")
        check(info[2] <= 0, f"no overflow at {w}")
    # the glance inside the walkthrough pop-out
    for w, h in ((1200, 800), (390, 844)):
        pg.set_viewport_size({"width": w, "height": h})
        pg.wait_for_timeout(200)
        pg.evaluate("document.getElementById('wt-open').click()")
        pg.wait_for_timeout(500)
        t = pg.evaluate("[document.getElementById('wt-title').textContent, document.getElementById('wt-stage').children[0].id]")
        check(t == ["At a glance", "glance"], f"walkthrough step 1 at {w}: {t}")
        pg.screenshot(path=os.path.join(OUT, f"wt-glance-{w}.png"))
        pg.keyboard.press("Escape")
        pg.wait_for_timeout(300)
    check(not errs, f"console errors: {errs[:3]}")
    br.close()
print("PASS" if not fails else f"{len(fails)} FAILS")
sys.exit(1 if fails else 0)

"""Browser test of Full walkthrough mode after the pass-3 structure edit (read-only on the page).
Copy of e1_wt_test.py with the step count N read from the page and the new reading order (the six specification
sections first, then the live panel and The task).
Run: /opt/shot-venv/bin/python s1_wt_test.py [--width N] [--scheme light|dark] [--reduced-motion] [--page PATH]"""
import argparse
import os
import sys
import tempfile

from playwright.sync_api import sync_playwright

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from shot import wrap, DEFAULT_PAGE  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--width", type=int, default=1200)
ap.add_argument("--scheme", default="light")
ap.add_argument("--reduced-motion", action="store_true")
ap.add_argument("--page", default=DEFAULT_PAGE)
a = ap.parse_args()

frag = open(a.page, encoding="utf-8").read()
tmp = os.path.join(tempfile.mkdtemp(prefix="wt-"), "page.html")
open(tmp, "w", encoding="utf-8").write(wrap(frag, "light"))
fails, errors = [], []


def check(cond, msg):
    print(("ok   " if cond else "FAIL ") + msg)
    if not cond:
        fails.append(msg)


with sync_playwright() as p:
    br = p.chromium.launch()
    ctx = br.new_context(viewport={"width": a.width, "height": 844 if a.width < 700 else 800}, color_scheme=a.scheme,
                         reduced_motion="reduce" if a.reduced_motion else "no-preference")
    pg = ctx.new_page()
    pg.on("console", lambda m: errors.append(m.text) if m.type in ("error", "warning") else None)
    pg.on("pageerror", lambda e: errors.append(str(e)))
    pg.goto("file://" + tmp, wait_until="load")
    pg.wait_for_timeout(600)
    W = 450 if not a.reduced_motion else 60
    order0 = pg.evaluate("[...document.querySelector('main').children].map(n => n.id)")
    N = pg.evaluate("1 + document.querySelectorAll('main > section[id], main > figure.panel[id]').length")
    print(f"   steps: {N}")
    pg.evaluate("window.scrollTo(0, 5000)")
    y0 = pg.evaluate("scrollY")
    pg.evaluate("document.getElementById('wt-open').click()")   # a real click would scroll the button into view
    pg.wait_for_timeout(300)
    st = pg.evaluate("({open: document.getElementById('wt').open, count: wt_count = document.getElementById('wt-count').textContent,"
                     " title: document.getElementById('wt-title').textContent, focus: document.activeElement.id,"
                     " lock: document.documentElement.classList.contains('wt-lock')})")
    check(st["open"] and st["count"] == f"Section 1 of {N}" and st["title"] == "At a glance", f"opens at step 1: {st}")
    check(st["focus"] == "wt-title" and st["lock"], "focus on the title, page scroll locked")
    if a.reduced_motion:
        anim = pg.evaluate("getComputedStyle(document.getElementById('wt')).animationName")
        check(anim == "none", f"reduced motion: no pop animation ({anim})")
    titles, widths = [], []
    sim_step = None
    for i in range(N):
        info = pg.evaluate("""() => { const s = document.getElementById('wt-stage'), b = document.getElementById('wt-body');
          return {t: document.getElementById('wt-title').textContent, n: s.children.length, id: s.children[0] && s.children[0].id,
                  over: b.scrollWidth - b.clientWidth, docOver: document.documentElement.scrollWidth - innerWidth,
                  // the pop-out itself must fit the screen (its text area only scrolls up and down)
                  dlgOver: Math.max(document.getElementById('wt').scrollWidth,
                                    Math.ceil(document.getElementById('wt').getBoundingClientRect().right)) - innerWidth,
                  footOver: Math.ceil(document.getElementById('wt-next').getBoundingClientRect().right) - innerWidth}; }""")
        titles.append(info["t"])
        if info["n"] != 1 or info["over"] > 1 or info["docOver"] > 0 or info["dlgOver"] > 0 or info["footOver"] > 0:
            check(False, f"step {i + 1} {info}")
        if info["id"] == "sim":
            sim_step = i
            cw =pg.evaluate("[document.getElementById('channel').clientWidth, document.getElementById('channel').width,"
                             " document.getElementById('wt-stage').clientWidth]")
            check(abs(cw[0] - (cw[2] - 2 * (16 if a.width > 560 else 12) - 2)) <= 2, f"panel canvas fits the pop-out {cw}")
            pg.click("#play")
            pg.wait_for_timeout(450)
            t1 = pg.evaluate("S.t")
            check(t1 > 1 and pg.evaluate("S.playing"), f"Play runs inside the pop-out (t={t1})")
            # arrow keys on the slider move the slider, not the section
            pg.focus("#tSlider")
            pg.keyboard.press("ArrowRight")
            pg.wait_for_timeout(W)
            check(pg.evaluate("document.getElementById('wt-title').textContent") == "Watch one episode",
                  "ArrowRight on the slider stays on the panel")
        if info["id"] == "weighting-test":
            ws = pg.evaluate("[...document.querySelectorAll('#weighting-test svg[id^=iw-svg]')].map(s => Math.round(s.getBoundingClientRect().width))")
            sw = pg.evaluate("document.getElementById('wt-stage').clientWidth")
            check(all(0 < w <= sw for w in ws), f"weighting-test charts drawn inside the pop-out {ws} <= {sw}")
        if i < N - 1:
            pg.click("#wt-next")
            pg.wait_for_timeout(W)
        if sim_step is not None and i == sim_step:
            check(not pg.evaluate("S.playing"), "leaving the panel pauses it")
    expect = ["At a glance", "Action Space", "Observation Space", "Starting State", "Transition Dynamics", "Rewards",
              "Episode End", "Watch one episode", "The task", "Summary of the environment", "The recording controller",
              "Reference Scores"]
    check(titles[:12] == expect, f"reading order {titles[:12]}")
    check(sim_step == 7, f"the live panel is step 8 (index {sim_step})")
    check("Version History" not in titles, "no Version History step")
    print("   all titles:", " | ".join(titles))
    nb = pg.evaluate("[document.getElementById('wt-next').textContent, document.getElementById('wt-next-name').textContent]")
    check(nb[0] == "Finish", f"last step shows Finish {nb}")
    pg.click("#wt-prev")
    pg.wait_for_timeout(W)
    check(pg.evaluate("document.getElementById('wt-count').textContent") == f"Section {N - 1} of {N}", "Back returns")
    # keys: arrows move, Escape closes
    pg.evaluate("document.getElementById('wt-title').focus()")
    pg.keyboard.press("ArrowLeft")
    pg.wait_for_timeout(W)
    check(pg.evaluate("document.getElementById('wt-count').textContent") == f"Section {N - 2} of {N}", "ArrowLeft moves back")
    pg.keyboard.press("ArrowDown")
    pg.wait_for_timeout(100)
    check(pg.evaluate("document.getElementById('wt-body').scrollTop") > 0, "ArrowDown on the title scrolls the text")
    pg.mouse.click(5, 5)  # the backdrop
    pg.wait_for_timeout(100)
    check(pg.evaluate("document.getElementById('wt').open"), "a click on the backdrop does not close")
    pg.keyboard.press("Escape")
    pg.wait_for_timeout(300)
    after = pg.evaluate("({open: document.getElementById('wt').open, order: [...document.querySelector('main').children].map(n => n.id),"
                        " focus: document.activeElement.id, y: scrollY, lock: document.documentElement.classList.contains('wt-lock'),"
                        " comments: [...document.querySelector('main').childNodes].filter(n => n.nodeType === 8 && /walkthrough/.test(n.data)).length})")
    check(not after["open"] and after["order"] == order0, "Esc closes and every section is back in place")
    check(after["focus"] == "wt-open" and abs(after["y"] - y0) <= 1 and not after["lock"] and after["comments"] == 0,
          f"focus back on the button, scroll restored ({after['y']} vs {y0}), no placeholders left")
    # reopen resumes; a link inside the pop-out opens the step that holds its target
    pg.click("#wt-open")
    pg.wait_for_timeout(300)
    check(pg.evaluate("document.getElementById('wt-count').textContent") == f"Section {N - 2} of {N}", "reopening resumes")
    for _ in range((N - 2) - (titles.index("The task") + 1)):
        pg.keyboard.press("ArrowLeft")
        pg.wait_for_timeout(W)
    check(pg.evaluate("document.getElementById('wt-title').textContent") == "The task", "back at The task")
    pg.evaluate("document.querySelector('#wt-stage a[href=\"#starting-state\"]').click()")
    pg.wait_for_timeout(W + 100)
    check(pg.evaluate("document.getElementById('wt-title').textContent") == "Starting State", "a link opens its section")
    pg.evaluate("document.querySelector('#wt-stage a[href=\"#task-situations\"]').click()")
    pg.wait_for_timeout(W + 100)
    r = pg.evaluate("""() => { const h = document.getElementById('task-situations'), b = document.getElementById('wt-body');
        return [document.getElementById('wt-title').textContent, Math.round(h.getBoundingClientRect().top - b.getBoundingClientRect().top)]; }""")
    check(r[0] == "The task" and 0 <= r[1] < 60, f"a link to a heading inside a section scrolls to it {r}")
    pg.click("#wt-close")
    pg.wait_for_timeout(300)
    check(pg.evaluate("[...document.querySelector('main').children].map(n => n.id)") == order0, "Close restores the page")
    print(f"console/page errors: {len(errors)}", errors[:5])
    br.close()
print("PASS" if not fails and not errors else f"{len(fails)} FAILS, {len(errors)} errors")
sys.exit(1 if fails or errors else 0)

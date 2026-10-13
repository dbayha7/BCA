"""Final screenshots after edit 4 (read-only on the page).

Run (WSL): /opt/shot-venv/bin/python e4_shots.py
For 390 px light and 1200 px dark, into page/shots/final-*.png:
  top            the first screen
  wt-first       Full walkthrough mode at its first step;  wt-last  at its last step
  sim            the live panel (and, at 390, its drawing at 2x: sim-channel-2x)
  demo           the task diagram's demo episode, mid-play
  starting-state, results, weighting-test   whole sections, cut into readable slices (-1, -2, ...)
Also prints console errors, page errors and horizontal overflow for each run, and the tick labels of the weighting
test's two charts.
"""
import os
import sys
import tempfile

from playwright.sync_api import sync_playwright

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from shot import wrap, DEFAULT_PAGE, OVERFLOW_JS  # noqa: E402

OUT = "/mnt/c/Users/David/dev/bca-shift-work/page/shots"
frag = open(DEFAULT_PAGE, encoding="utf-8").read()
status = 0


def sliced(pg, sel, stem, h):
    box = pg.locator(sel).first.evaluate(
        "e => { const r = e.getBoundingClientRect(); return {x: r.left + scrollX, y: r.top + scrollY, w: r.width, h: r.height}; }")
    pad = 8
    x0, y0, w, H = max(0, box["x"] - pad), max(0, box["y"] - pad), box["w"] + 2 * pad, box["h"] + 2 * pad
    vw = pg.evaluate("document.documentElement.clientWidth")
    w = min(w, vw - x0)
    names, y, k = [], y0, 1
    while y < y0 + H - 1:
        hh = min(h, y0 + H - y)
        p = os.path.join(OUT, f"{stem}-{k}.png")
        pg.screenshot(path=p, full_page=True, clip={"x": x0, "y": y, "width": w, "height": hh})
        names.append(os.path.basename(p))
        y += hh
        k += 1
    print(f"  {stem}: {round(box['h'])} css px -> {len(names)} slices")


for width, scheme in [(390, "light"), (1200, "dark")]:
    tag = f"{width}-{scheme}"
    tmp = os.path.join(tempfile.mkdtemp(prefix="e4s-"), "page.html")
    open(tmp, "w", encoding="utf-8").write(wrap(frag, scheme))
    errors = []
    with sync_playwright() as p:
        br = p.chromium.launch()
        H = 844 if width < 700 else 800
        ctx = br.new_context(viewport={"width": width, "height": H}, color_scheme=scheme)
        pg = ctx.new_page()
        pg.on("console", lambda m: errors.append(f"{m.type}: {m.text}") if m.type in ("error", "warning") else None)
        pg.on("pageerror", lambda e: errors.append(str(e)))
        pg.goto("file://" + tmp, wait_until="load")
        pg.wait_for_load_state("networkidle")
        pg.evaluate("document.fonts ? document.fonts.ready.then(() => true) : true")
        pg.wait_for_timeout(700)
        print(f"[{tag}]")
        pg.evaluate("window.scrollTo(0, 0)")
        pg.wait_for_timeout(200)
        pg.screenshot(path=os.path.join(OUT, f"final-top-{tag}.png"))

        # the walkthrough: first step, then the last
        pg.evaluate("document.getElementById('wt-open').click()")
        pg.wait_for_timeout(500)
        pg.screenshot(path=os.path.join(OUT, f"final-wt-first-{tag}.png"))
        first = pg.evaluate("[document.getElementById('wt-count').textContent, document.getElementById('wt-title').textContent,"
                            " document.getElementById('wt-next-name').textContent]")
        for _ in range(21):
            pg.click("#wt-next")
            pg.wait_for_timeout(330)
        pg.wait_for_timeout(300)
        pg.screenshot(path=os.path.join(OUT, f"final-wt-last-{tag}.png"))
        last = pg.evaluate("[document.getElementById('wt-count').textContent, document.getElementById('wt-title').textContent,"
                           " document.getElementById('wt-next').textContent, document.getElementById('wt-next-name').textContent,"
                           " Math.round(document.getElementById('wt').getBoundingClientRect().width)]")
        print("  walkthrough first:", first, "| last:", last)
        pg.click("#wt-close")
        pg.wait_for_timeout(300)

        # the live panel
        sliced(pg, "#sim", f"final-sim-{tag}", 1150 if width < 700 else 900)
        if width < 700:
            pg.locator("#channel").scroll_into_view_if_needed()
            pg.wait_for_timeout(200)
            ctx2 = br.new_context(viewport={"width": width, "height": H}, color_scheme=scheme, device_scale_factor=2)
            pg2 = ctx2.new_page()
            pg2.goto("file://" + tmp, wait_until="load")
            pg2.wait_for_load_state("networkidle")
            pg2.wait_for_timeout(600)
            pg2.locator("#channel").screenshot(path=os.path.join(OUT, f"final-sim-channel-2x-{tag}.png"))
            ctx2.close()

        # the demo, mid-play: autoplay starts once half the drawing is on screen
        if width < 700:
            # the drawing on show (the narrow one) near the top of the screen, so autoplay starts and the controls fit too
            pg.evaluate("(() => { const s = document.querySelector('#task-diagram svg.dg-narrow'); s.scrollIntoView({block: 'start'});"
                        " window.scrollBy(0, -12); })()")
            pg.wait_for_timeout(2600)
            st = pg.evaluate("window.taskDemo ? [window.taskDemo.S.t, window.taskDemo.S.playing] : null")
            pg.screenshot(path=os.path.join(OUT, f"final-demo-{tag}.png"))
            print("  demo state when shot (t, playing):", st)
            sliced(pg, "#task-diagram", f"final-demo-whole-{tag}", 1150)
        else:
            pg.evaluate("document.getElementById('task-diagram').scrollIntoView({block: 'start'})")
            pg.wait_for_timeout(2600)
            st = pg.evaluate("window.taskDemo ? [window.taskDemo.S.t, window.taskDemo.S.playing] : null")
            pg.locator("#task-diagram").screenshot(path=os.path.join(OUT, f"final-demo-{tag}.png"))
            print("  demo state when shot (t, playing):", st)

        for sec in ["starting-state", "results", "weighting-test"]:
            sliced(pg, "#" + sec, f"final-{sec}-{tag}", 1150 if width < 700 else 900)

        ticks = pg.evaluate("['iw-svg-dens', 'iw-svg-w'].map(id => [...document.querySelectorAll('#' + id + ' text.tick')]"
                            ".map(t => t.textContent).filter(s => s && !/^(0\\.1|0\\.3|0\\.5|0\\.7|0\\.9|1\\.1)$/.test(s) || s === '0'))")
        print("  weighting-test y ticks (density, weight):", ticks)
        sl = pg.evaluate("(() => { const l = document.querySelector('#iw-c-thrust .iw-sl'); return Math.round(l.getBoundingClientRect().height); })()")
        print("  throttle slider row height:", sl, "px")
        ov = pg.evaluate(OVERFLOW_JS)
        print(f"  horizontal overflow: {ov['overflow']} px; console errors/warnings and page errors: {len(errors)}")
        for e in errors:
            print("    ", e[:300])
        if ov["overflow"] > 0 or errors:
            status = 1
        br.close()
sys.exit(status)

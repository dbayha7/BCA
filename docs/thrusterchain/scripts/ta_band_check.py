"""Why does the task chart's band tooltip not show at 390 px? Read-only probe."""
import os, sys, tempfile
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from shot import wrap
from playwright.sync_api import sync_playwright
P = "/mnt/c/Users/David/dev/bca-shift-work/page/thrusterchain.html"
d = tempfile.mkdtemp(); f = os.path.join(d, "p.html")
open(f, "w", encoding="utf-8").write(wrap(open(P, encoding="utf-8").read(), "light"))
with sync_playwright() as p:
    b = p.chromium.launch()
    for w in (390, 1200):
        pg = b.new_page(viewport={"width": w, "height": 844})
        pg.goto("file://" + f); pg.wait_for_timeout(1200)
        loc = pg.locator("#task-chart .pt-hits rect").nth(7)
        print(w, "count", pg.locator("#task-chart .pt-hits rect").count())
        pg.evaluate("el => el.scrollIntoView({block: 'center'})", loc.element_handle())
        pg.wait_for_timeout(300)
        bx = loc.bounding_box()
        print(w, "box", bx)
        r = pg.evaluate("""([x, y]) => { const e = document.elementFromPoint(x, y); return e ? e.outerHTML.slice(0, 200) : null; }""",
                        [bx["x"] + bx["width"] / 2, bx["y"] + bx["height"] * 0.4])
        print(w, "at point:", r)
        print(w, pg.evaluate("() => { const s = document.querySelector('#task-chart svg'); const r = s.getBoundingClientRect(); return [r.width, r.height, getComputedStyle(s).display, s.getAttribute('viewBox')]; }"))
    b.close()

"""Read-only: step through the whole walkthrough with motion on and sample the pop-out's sideways overflow
(scrollWidth - clientWidth of #wt-body) during each change of section, at 0, 60, 120, 200 and 320 ms after Next."""
import os, sys, tempfile
from playwright.sync_api import sync_playwright
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from shot import wrap, DEFAULT_PAGE  # noqa: E402
page = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_PAGE
frag = open(page, encoding="utf-8").read()
f = os.path.join(tempfile.mkdtemp(), "p.html"); open(f, "w", encoding="utf-8").write(wrap(frag, "light"))
OV = "() => { const b = document.getElementById('wt-body'); return b.scrollWidth - b.clientWidth; }"
with sync_playwright() as p:
    br = p.chromium.launch()
    for w in (1200, 390):
        pg = br.new_page(viewport={"width": w, "height": 844 if w < 700 else 800})
        pg.goto("file://" + f); pg.wait_for_timeout(800)
        pg.click("#wt-open"); pg.wait_for_timeout(400)
        worst, bad = 0, []
        for step in range(30):
            title = pg.evaluate("document.getElementById('wt-title').textContent")
            if pg.evaluate("document.getElementById('wt-next').textContent").startswith("Finish"):
                break
            pg.evaluate("document.getElementById('wt-next').click()")
            last = 0
            for dt in (0, 60, 60, 80, 120):
                pg.wait_for_timeout(dt)
                o = pg.evaluate(OV)
                if o > 0: bad.append((title, last + dt, o))
                worst = max(worst, o); last += dt
            pg.wait_for_timeout(300)
        print(w, "steps", step, "largest sideways overflow during a change:", worst, "px", bad[:6])
        pg.close()
    br.close()

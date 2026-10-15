"""Read-only: step through the walkthrough as the test does and sample the pop-out body's horizontal overflow over time
on the given sections. Usage: /opt/shot-venv/bin/python s1_wt_over2.py PAGE WIDTH ID ..."""
import os, sys, tempfile
from playwright.sync_api import sync_playwright
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from shot import wrap  # noqa: E402
page, width, ids = sys.argv[1], int(sys.argv[2]), set(sys.argv[3:])
tmp = os.path.join(tempfile.mkdtemp(prefix="wto-"), "page.html")
open(tmp, "w", encoding="utf-8").write(wrap(open(page, encoding="utf-8").read(), "light"))
OV = "(() => { const b = document.getElementById('wt-body'); return b.scrollWidth - b.clientWidth; })()"
WIDE = """(() => { const b = document.getElementById('wt-body'), br = b.getBoundingClientRect(), lim = br.left + b.clientWidth, out = [];
  for (const el of document.getElementById('wt-stage').querySelectorAll('*')) { const r = el.getBoundingClientRect();
    if (r.width && r.right > lim + 0.5) out.push([el.tagName.toLowerCase(), el.id, String(el.className).slice(0,40), Math.round(r.right - lim), (el.textContent||'').trim().slice(0,60)]); }
  return out.slice(0, 10); })()"""
with sync_playwright() as p:
    br = p.chromium.launch()
    pg = br.new_context(viewport={"width": width, "height": 800}).new_page()
    pg.goto("file://" + tmp, wait_until="load"); pg.wait_for_timeout(600)
    pg.evaluate("window.scrollTo(0, 5000)")
    pg.evaluate("document.getElementById('wt-open').click()"); pg.wait_for_timeout(300)
    for i in range(25):
        cur = pg.evaluate("document.getElementById('wt-stage').children[0].id")
        if cur in ids:
            samples = []
            for t in (0, 100, 300, 600, 1200):
                samples.append((t, pg.evaluate(OV)))
                if t == 0: wide = pg.evaluate(WIDE)
                pg.wait_for_timeout(100 if t == 0 else (200 if t == 100 else (300 if t == 300 else 600)))
            print(f"#{cur}: overflow samples {samples}")
            for w in wide: print("    ", w)
        if pg.evaluate("document.getElementById('wt-next').textContent") == "Finish": break
        pg.click("#wt-next"); pg.wait_for_timeout(450)
    br.close()

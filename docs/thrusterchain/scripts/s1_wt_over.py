"""Read-only: open the walkthrough on a given step and list the elements that stick out of the pop-out's text area.
Usage: /opt/shot-venv/bin/python s1_wt_over.py PAGE WIDTH SECTION_ID [SECTION_ID ...]"""
import os, sys, tempfile
from playwright.sync_api import sync_playwright
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from shot import wrap  # noqa: E402
page, width, ids = sys.argv[1], int(sys.argv[2]), sys.argv[3:]
tmp = os.path.join(tempfile.mkdtemp(prefix="wto-"), "page.html")
open(tmp, "w", encoding="utf-8").write(wrap(open(page, encoding="utf-8").read(), "light"))
JS = """(id) => { const b = document.getElementById('wt-body'), br = b.getBoundingClientRect(), out = [];
  for (const el of document.getElementById(id).querySelectorAll('*')) {
    const r = el.getBoundingClientRect(); if (!r.width) continue;
    let clipped = false; for (let a = el.parentElement; a && a !== b; a = a.parentElement) {
      const ox = getComputedStyle(a).overflowX; if (ox !== 'visible') { if (a.getBoundingClientRect().right <= br.right - (b.offsetWidth - b.clientWidth) + 0.5) { clipped = true; break; } } }
    if (!clipped && r.right > br.left + b.clientWidth + 0.5) out.push([el.tagName.toLowerCase(), el.id, String(el.className).slice(0, 40), Math.round(r.right - br.left - b.clientWidth), (el.textContent || '').trim().slice(0, 70)]);
  }
  const keep = out.filter((x, i) => true); return [b.scrollWidth - b.clientWidth, keep.slice(0, 12)]; }"""
with sync_playwright() as p:
    br = p.chromium.launch()
    pg = br.new_context(viewport={"width": width, "height": 800}).new_page()
    pg.goto("file://" + tmp, wait_until="load"); pg.wait_for_timeout(500)
    pg.evaluate("document.getElementById('wt-open').click()"); pg.wait_for_timeout(300)
    for sid in ids:
        for _ in range(30):
            cur = pg.evaluate("document.getElementById('wt-stage').children[0].id")
            if cur == sid: break
            pg.evaluate("document.getElementById('wt-next').click()"); pg.wait_for_timeout(120)
        pg.wait_for_timeout(400)
        over, rows = pg.evaluate(JS, sid)
        print(f"#{sid}: body overflow {over}px")
        for r in rows: print("   ", r)
    br.close()

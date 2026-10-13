#!/usr/bin/env python3
"""Tooltip inventory: real-hover probes for the truncation questions. Read-only: never writes the page.

1. Hovers the "stall" cell of the Arguments table and a "Stalls" cell of the Action Space table with the mouse and
   records the tooltip exactly as shown (text, box, whether the box clips its content).
2. Repeats the hover on the tallest tooltips at short viewports (the box has max-height: calc(100vh - 16px) and
   overflow: hidden) and saves screenshots of the clipped box.
3. Works the weighting test's lean buttons and checks whether the redrawn bars bring back native <title> tooltips.
Screenshots go to shots/tt-inv-*.png. Run:
  cd /mnt/c/Users/David/dev/bca-shift-work/page && /opt/shot-venv/bin/python scripts/tt_probe.py
"""
import json
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from shot import wrap  # noqa: E402
from playwright.sync_api import sync_playwright  # noqa: E402

PAGE_DIR = "/mnt/c/Users/David/dev/bca-shift-work/page"
SHOTS = os.path.join(PAGE_DIR, "shots")
OUT = os.path.join(PAGE_DIR, "tmp", "inv", "probe.json")

TIP = r"""() => { const t = document.getElementById('tt'); const r = t.getBoundingClientRect();
  return { hidden: t.hidden, text: t.innerText, sh: t.scrollHeight, ch: t.clientHeight, top: r.top, bottom: r.bottom,
           vh: innerHeight, clipped: t.scrollHeight > t.clientHeight + 1 }; }"""

# the last line of text that is fully visible inside the box
VISIBLE_END = r"""() => { const t = document.getElementById('tt'); const box = t.getBoundingClientRect();
  const w = document.createTreeWalker(t, NodeFilter.SHOW_TEXT); let last = '';
  while (w.nextNode()) { const n = w.currentNode; const rg = document.createRange();
    for (let i = 0; i < n.length; i++) { rg.setStart(n, i); rg.setEnd(n, i + 1); const rr = rg.getBoundingClientRect();
      if (rr.height && rr.bottom <= box.bottom - 1) last += n.data[i]; } last += ' '; }
  return last.replace(/\s+/g, ' ').trim().slice(-140); }"""


def main():
    d = tempfile.mkdtemp(prefix="ttprobe-")
    f = os.path.join(d, "page.html")
    open(f, "w", encoding="utf-8").write(wrap(open(os.path.join(PAGE_DIR, "thrusterchain.html"), encoding="utf-8").read(), "light"))
    res = {}
    with sync_playwright() as p:
        b = p.chromium.launch()

        def page(w, h):
            ctx = b.new_context(viewport={"width": w, "height": h}, color_scheme="light")
            pg = ctx.new_page()
            pg.goto("file://" + f, wait_until="load")
            pg.wait_for_timeout(900)
            return pg

        def hover(pg, loc, name=None):
            loc.scroll_into_view_if_needed()
            pg.wait_for_timeout(200)
            box = loc.bounding_box()
            pg.mouse.move(1, 1)
            pg.wait_for_timeout(200)
            pg.mouse.move(box["x"] + min(12, box["width"] / 2), box["y"] + box["height"] / 2)
            pg.wait_for_timeout(600)
            t = pg.evaluate(TIP)
            t["visibleEnd"] = pg.evaluate(VISIBLE_END)
            if name:
                pg.screenshot(path=os.path.join(SHOTS, name))
                t["shot"] = "shots/" + name
            return t

        for w, h in ((1200, 800), (390, 844), (1200, 260), (390, 300)):
            pg = page(w, h)
            k = f"{w}x{h}"
            stall = pg.locator("#arguments table").first.locator("td").first
            stalls = pg.locator("#action-space table").first.locator("tbody tr").first.locator("td").last
            res[k + " arguments stall cell"] = hover(pg, stall, f"tt-inv-{k}-arguments-stall.png")
            res[k + " action-space Stalls cell"] = hover(pg, stalls, f"tt-inv-{k}-action-stalls.png")
            # the tallest tooltip: the legend of the dynamics formula
            leg = pg.locator("#dynamics div.math-display mjx-container > svg").first
            pg.evaluate("el => el.scrollIntoView({block: 'center'})", leg.element_handle())
            res[k + " dynamics legend"] = hover(pg, leg, f"tt-inv-{k}-dynamics-legend.png")
            pg.context.close()

        # weighting test: do redrawn bars bring back native <title> tooltips?
        pg = page(1200, 800)
        q = "() => ({ titles: [...document.querySelectorAll('#iw-svg-bars title')].map(t => t.textContent), " \
            "tips: [...document.querySelectorAll('#iw-svg-bars [data-tip]')].map(e => e.getAttribute('data-tip')) })"
        before = pg.evaluate(q)
        btns = pg.locator("#weighting-test button")
        labels = [btns.nth(i).inner_text().strip() for i in range(btns.count())]
        target = next((i for i, x in enumerate(labels) if x.startswith("Strong")), None)
        if target is not None:
            btns.nth(target).click()
            pg.wait_for_timeout(400)
        after = pg.evaluate(q)
        res["weighting bars"] = {"buttons": labels, "before": before, "after clicking Strong": after}
        pg.context.close()
        b.close()
    json.dump(res, open(OUT, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    for k, v in res.items():
        print(k, json.dumps(v, ensure_ascii=False)[:700])


if __name__ == "__main__":
    main()

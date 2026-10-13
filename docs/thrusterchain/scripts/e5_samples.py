"""Hover sample cells of special tables and print their tooltips (key-value, spans, math, ranges, groups, †)."""
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from shot import wrap  # noqa: E402
from playwright.sync_api import sync_playwright  # noqa: E402

PAGE = "/mnt/c/Users/David/dev/bca-shift-work/page/thrusterchain.html"
# (table index, CSS within the table, label)
SAMPLES = [
    (9, "tbody tr:nth-child(1) td", "summary table, key-value"),
    (9, "tbody tr:nth-child(4) th", "summary table, row header"),
    (1, "thead tr.grp th:nth-child(3)", "grouped header"),
    (1, "tbody tr:nth-child(6) td:nth-child(7)", "cell with dagger"),
    (7, "tbody tr:nth-child(3) td:nth-child(5)", "matrix A"),
    (7, "thead th:nth-child(1)", "matrix A, empty corner"),
    (11, "tbody tr:nth-child(1) td:nth-child(12)", "matrix K"),
    (18, "tbody tr:nth-child(18) td:nth-child(2)", "episode, reward row"),
    (20, "tbody tr:nth-child(11) td:nth-child(3)", "sum row"),
    (31, "tbody tr:nth-child(7) td:nth-child(2)", "row under a rowspan"),
    (32, "thead th:nth-child(2)", "math in header"),
    (33, "tbody tr:nth-child(8) td:nth-child(3)", "math in row label"),
    (36, "tbody tr:nth-child(5) td:nth-child(3)", "value with seed range, group"),
    (43, "tbody tr:nth-child(4) td:nth-child(2)", "weighting marks, group"),
    (42, "tbody tr:nth-child(2) td:nth-child(4)", "weight formula"),
]
JS = """([i, sel]) => { const t = document.querySelectorAll('table')[i]; const d = t.closest('details'); if (d) d.open = true;
  const el = t.querySelector(sel); el.scrollIntoView({block: 'center'}); return !!el; }"""
frag = open(PAGE, encoding="utf-8").read()
d = tempfile.mkdtemp()
f = os.path.join(d, "p.html")
open(f, "w", encoding="utf-8").write(wrap(frag, "light"))
with sync_playwright() as p:
    b = p.chromium.launch()
    pg = b.new_page(viewport={"width": 1200, "height": 800})
    pg.goto("file://" + f)
    pg.wait_for_timeout(800)
    for i, sel, label in SAMPLES:
        pg.mouse.move(2, 2)
        pg.wait_for_timeout(250)
        pg.evaluate(JS, [i, sel])
        pg.wait_for_timeout(150)
        el = pg.locator("table").nth(i).locator(sel).first
        box = el.bounding_box()
        pg.mouse.move(box["x"] + box["width"] / 2, box["y"] + box["height"] / 2)
        pg.wait_for_timeout(450)
        r = pg.evaluate("(() => { const t = document.getElementById('tt'); return t.hidden ? '(hidden)' : "
                        "[...t.children].map(c => c.textContent + ' || bold=' + [...c.querySelectorAll('b')].map(x => x.textContent).join('|')).join('\\n      '); })()")
        print(f"#{i} {label}:\n      {r}\n")
    b.close()

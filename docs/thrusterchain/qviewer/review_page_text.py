"""Reviewer: dump the rendered text of the critic-in-action section of test_page.html (read only) for a few episodes
and states, to compare every shown number with the data. Writes /root/bca-shift/runs/qviewer/review/page_text.json."""
import json
import os
import tempfile
from pathlib import Path

from playwright.sync_api import sync_playwright

PAGE = "/mnt/c/Users/David/dev/bca-shift-work/qviewer/test_page.html"
OUT = Path("/root/bca-shift/runs/qviewer/review/page_text.json")
CASES = [dict(q="expert", c="gap", s="strong", t=40), dict(q="expert", c="gap", s="strong", t=0),
         dict(q="medium", c="full", s="training", t=40), dict(q="random", c="full", s="strong", t=71),
         dict(q="medium-replay", c="gap", s="training", t=23), dict(q="expert", c="full", s="training", t=99)]

GRAB = r"""
() => {
  const R = document.getElementById('critic-in-action');
  const tx = id => { const n = document.getElementById(id); return n ? n.textContent : null; };
  const ro = {}; R.querySelectorAll('.ro .v').forEach(n => ro[n.id] = n.textContent);
  const rows = id => [...document.getElementById(id).tBodies[0].rows].map(r => [...r.cells].map(c => c.textContent));
  const nums = [...R.querySelectorAll('[data-qv-num]')].map(n => [n.dataset.qvNum, n.textContent]);
  const tips = [...R.querySelectorAll('[data-tip]')].map(n => [n.getAttribute('data-tip-title'), n.getAttribute('data-tip')]);
  document.getElementById('qv-tables').open = true;
  return { now: tx('qv-now'), ro, ep: tx('qv-ep-list'), eph: tx('qv-ep-h'), find: tx('qv-find'), capctx: tx('qv-cap-ctx'),
           ctx: rows('qv-tbl-ctx'), qual: rows('qv-tbl-qual'), nums, tips, y1: tx('qv-y1'),
           svg1: document.getElementById('qv-svg1').getAttribute('aria-label') };
}
"""


def main():
    frag = Path(PAGE).read_text(encoding="utf-8")
    d = tempfile.mkdtemp(prefix="qvrev-")
    p = os.path.join(d, "page.html")
    Path(p).write_text("<!doctype html>\n<html lang=\"en\">\n<head>\n<meta charset=\"utf-8\">\n<meta name=\"viewport\" "
                       "content=\"width=device-width, initial-scale=1\">\n<style>body{margin:0}</style>\n</head>\n<body>\n"
                       + frag + "\n</body>\n</html>\n", encoding="utf-8")
    out, errs = [], []
    with sync_playwright() as pw:
        b = pw.chromium.launch()
        pg = b.new_page(viewport={"width": 1200, "height": 900})
        pg.on("pageerror", lambda e: errs.append(str(e)))
        pg.goto("file://" + p, wait_until="load", timeout=60000)
        pg.wait_for_timeout(800)
        for cs in CASES:
            pg.evaluate("o => window.qvSet(o)", cs)
            pg.wait_for_timeout(150)
            g = pg.evaluate(GRAB)
            pg.wait_for_timeout(100)
            g["sweep"] = pg.evaluate("() => [...document.getElementById('qv-tbl-sweep').tBodies[0].rows].map(r => [...r.cells].map(c => c.textContent))")
            g["eptbl"] = pg.evaluate("() => [...document.getElementById('qv-tbl-ep').tBodies[0].rows].map(r => [...r.cells].map(c => c.textContent))")
            g["case"] = cs
            out.append(g)
        b.close()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(dict(cases=out, errors=errs), ensure_ascii=False, indent=1), encoding="utf-8")
    print("errors:", errs)
    for g in out:
        print("CASE", g["case"])
        print("  now:", g["now"])
        print("  ro:", g["ro"])
        print("  eph:", g["eph"])
        print("  ep:", g["ep"])
    print("FIND:", out[0]["find"])
    print("CAPCTX strong:", out[0]["capctx"])
    print("CTX strong:", out[0]["ctx"])
    print("CTX training:", out[2]["ctx"])
    print("QUAL:", out[0]["qual"])
    print("NUMS:", out[0]["nums"])
    print("Y1:", out[0]["y1"])
    for t in out[0]["tips"]:
        if t[0] in ("The 80% hold removed",):
            print("TIP gap:", t[1])


if __name__ == "__main__":
    main()

"""Read-only browser checks for the pass-3 final fixes: console errors, horizontal overflow, column descriptions of
every table, the weighting test's static readouts against what its script computes, the progress strip's row, the
status-chip dots, and the sticky group headings.

  cd /mnt/c/Users/David/dev/bca-shift-work/page && /opt/shot-venv/bin/python scripts/p3f_check.py [--page PATH]
"""
import argparse
import os
import re
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from shot import wrap, OVERFLOW_JS, DEFAULT_PAGE  # noqa: E402
from playwright.sync_api import sync_playwright  # noqa: E402

COVERAGE_JS = r"""
() => [...document.querySelectorAll('table')].map((t, i) => {
  const sec = t.closest('section[id]');
  const heads = [...t.querySelectorAll('[data-tt="h"]')];
  const missing = heads.filter(h => !h.hasAttribute('data-tt-desc') && h.closest('thead, tbody') &&
     (h.closest('thead') ? h.closest('thead').rows[h.closest('thead').rows.length - 1].contains(h) : true))
     .map(h => h.textContent.trim().slice(0, 60));
  return { i, sec: sec ? sec.id : '', heads: heads.length, missing };
})
"""
FAILS = []


def check(ok, msg):
    print(("  ok   " if ok else "  FAIL ") + msg)
    if not ok:
        FAILS.append(msg)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--page", default=DEFAULT_PAGE)
    a = ap.parse_args()
    frag = open(a.page, encoding="utf-8").read()
    static = {k: re.search(r'id="%s">([^<]*)<' % k, frag).group(1) for k in
              ("iw-t-mean", "iw-t-range", "iw-t-ess", "iw-ro-w", "iw-ro-d", "iw-ro-pt", "iw-ro-wc")}
    with sync_playwright() as p:
        br = p.chromium.launch()
        for width, scheme in ((390, "light"), (1200, "dark")):
            print(f"== {width} {scheme}")
            tmp = os.path.join(tempfile.mkdtemp(prefix="p3f-"), "page.html")
            open(tmp, "w", encoding="utf-8").write(wrap(frag, scheme))
            ctx = br.new_context(viewport={"width": width, "height": 844 if width < 700 else 900}, color_scheme=scheme,
                                 has_touch=width < 700, is_mobile=width < 700)
            pg = ctx.new_page()
            errs = []
            pg.on("console", lambda m: errs.append(f"{m.type}: {m.text}") if m.type in ("error", "warning") else None)
            pg.on("pageerror", lambda e: errs.append(f"pageerror: {e}"))
            pg.on("requestfailed", lambda r: errs.append(f"requestfailed: {r.url}"))
            pg.goto("file://" + tmp)
            pg.wait_for_timeout(1500)
            check(not errs, f"no console errors ({errs[:3]})")
            ov = pg.evaluate(OVERFLOW_JS)
            check(ov["overflow"] <= 0, f"no horizontal overflow ({ov['overflow']} px; {ov['offenders'][:3]})")
            cov = pg.evaluate(COVERAGE_JS)
            miss = [c for c in cov if c["missing"]]
            check(not miss, f"every table header has a description ({len(cov)} tables; missing: {miss[:3]})")
            live = pg.evaluate("ids => Object.fromEntries(ids.map(i => [i, document.getElementById(i).textContent]))",
                               list(static))
            for k, v in static.items():
                check(live[k] == v, f"static readout {k}: page {v!r}, script {live[k]!r}")
            steps = pg.evaluate("() => [...document.querySelectorAll('.iw-pipe .iw-step')].map(e => e.getBoundingClientRect().top)")
            rows = len({round(t) for t in steps})
            check(len(steps) == 6 and (rows == 1 if width >= 700 else rows <= 3), f"progress strip: {len(steps)} steps in {rows} row(s)")
            dots = pg.evaluate(r"""() => [...document.querySelectorAll('#weighting-test td .iw-chip')].map(c => {
                const r = c.getBoundingClientRect(), lh = parseFloat(getComputedStyle(c).lineHeight) || 16;
                const d = getComputedStyle(c, '::before');
                return { h: r.height, lh, mt: parseFloat(d.marginTop) }; })""")
            wrapped = [d for d in dots if d["h"] > 1.5 * d["lh"]]
            check(all(abs(d["mt"] - (d["lh"] / 2 - 4)) < 1.5 for d in dots),
                  f"chip dots sit on the first line ({len(dots)} chips, {len(wrapped)} wrapped; line {dots[0]['lh']:.1f}px, dot offset {dots[0]['mt']:.1f}px)")
            cl = pg.evaluate("() => [...document.querySelectorAll('.claims > .check')].map(e => [Math.round(e.getBoundingClientRect().left), Math.round(e.getBoundingClientRect().width)])")
            check(len(cl) == 2 and cl[0][0] == cl[1][0], f"the two claims stack: {cl}")
            # sticky group headings: scroll the first wide table of Stage 1 sideways and look at a heading's words
            g = pg.evaluate(r"""() => { const t = document.querySelector('#beyond-the-data table.rows8'); if (!t) return null;
                const box = t.closest('.tbl'); box.scrollLeft = 400;
                const grp = t.querySelector('tr.sub .grp'); const b = box.getBoundingClientRect(), r = grp.getBoundingClientRect();
                return { scroll: box.scrollLeft, boxL: b.left, boxR: b.right, grpL: r.left, grpR: r.right }; }""")
            if g and g["scroll"] > 0:
                check(g["grpL"] >= g["boxL"] - 1 and g["grpL"] < g["boxR"], f"group heading stays in view when scrolled ({g})")
            else:
                print(f"       (table not scrollable at this width: {g})")
            ctx.close()
        br.close()
    print("FAILED:" if FAILS else "all checks passed", *FAILS, sep="\n  ")
    return 1 if FAILS else 0


if __name__ == "__main__":
    sys.exit(main())

"""Final screenshots of pass 3 (read-only on the page): shots/p3-final-<what>-<width>-<scheme>[-k].png.

  cd /mnt/c/Users/David/dev/bca-shift-work/page && /opt/shot-venv/bin/python scripts/p3f_shots.py

Each shot is a clip of the page between the top of one element and the top of another (or the bottom of the first),
cut into slices of at most SLICE css px. Prints console errors and the horizontal overflow at each width.
"""
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from shot import wrap, OVERFLOW_JS, DEFAULT_PAGE  # noqa: E402
from playwright.sync_api import sync_playwright  # noqa: E402

OUT = "/mnt/c/Users/David/dev/bca-shift-work/page/shots"
SLICE = {390: 1100, 1200: 1000}

# (name, start selector, end selector or None for the start element's own bottom)
RANGES = {
    1200: [
        ("results", "#results", "#weighting-test"),
        ("weighting", "#weighting-test", "#iw-shift"),
        ("weighting-rest", "#iw-shift", "#arguments"),
        ("coupling", "#dynamics-coupling", "#dynamics details"),
        ("formula", "#p3f-formula", None),
        ("description-box", "#description .callout", None),
    ],
    390: [
        ("results-top", "#results", "#results-critic"),
        ("results-critic", "#results-critic", "#results-cost"),
        ("results-cost", "#results-cost", "#results-caution"),
        ("results-caution", "#results-caution", "#results-forecast"),
        ("results-world", "#results-world", "#beyond-the-data"),
        ("results-after", "#beyond-the-data", "#ats-bca"),
        ("results-claims", "#ats-claims", "#weighting-test"),
        ("weighting", "#weighting-test", "#iw-explore"),
        ("weighting-marks", "#p3f-marks", "#arguments"),
        ("coupling", "#dynamics-coupling", "#dynamics details"),
        ("formula", "#p3f-formula", None),
        ("description-box", "#description .callout", None),
    ],
}

MARK_FORMULA = r"""() => { for (const m of document.querySelectorAll('mjx-container[display="true"]')) {
  if ((m.getAttribute('aria-label') || '').includes('\\hat{a}}{m + \\mu}')) {
    const d = m.closest('.math-display'); d.id = 'p3f-formula';
    for (const h of document.querySelectorAll('#weighting-test h3')) if (h.textContent.startsWith('Pass marks')) h.id = 'p3f-marks';
    return true; } } return false; }"""


def box(pg, sel, end):
    return pg.evaluate("""([s, e]) => { const a = document.querySelector(s); if (!a) return null;
      const r = a.getBoundingClientRect(); const top = r.top + scrollY;
      let bottom = r.bottom + scrollY;
      if (e) { const b = document.querySelector(e); if (b) bottom = b.getBoundingClientRect().top + scrollY; }
      const left = Math.max(0, r.left + scrollX - 4);
      return { x: left, y: top - 6, w: Math.min(document.documentElement.clientWidth - left, r.width + 8), h: bottom - top + 6 }; }""",
                       [sel, end])


def main():
    frag = open(DEFAULT_PAGE, encoding="utf-8").read()
    made = []
    with sync_playwright() as p:
        br = p.chromium.launch()
        for width, scheme in ((390, "light"), (1200, "dark")):
            tmp = os.path.join(tempfile.mkdtemp(prefix="p3fs-"), "page.html")
            open(tmp, "w", encoding="utf-8").write(wrap(frag, scheme))
            ctx = br.new_context(viewport={"width": width, "height": 844 if width < 700 else 900}, color_scheme=scheme)
            pg = ctx.new_page()
            errs = []
            pg.on("console", lambda m: errs.append(f"{m.type}: {m.text}") if m.type in ("error", "warning") else None)
            pg.on("pageerror", lambda e: errs.append(f"pageerror: {e}"))
            pg.goto("file://" + tmp)
            pg.wait_for_timeout(1500)
            tag = f"{width}-{scheme}"
            # the top of the page
            path = f"{OUT}/p3-final-top-{tag}.png"
            pg.screenshot(path=path)
            made.append(path)
            assert pg.evaluate(MARK_FORMULA), "formula not found"
            for name, s, e in RANGES[width]:
                b = box(pg, s, e)
                assert b, f"{s} not found"
                y, k, end = b["y"], 1, b["y"] + b["h"]
                while y < end - 1:
                    h = min(SLICE[width], end - y)
                    path = f"{OUT}/p3-final-{name}-{tag}-{k}.png"
                    pg.screenshot(path=path, full_page=True, clip={"x": b["x"], "y": y, "width": b["w"], "height": h})
                    made.append(path)
                    y += h
                    k += 1
            # a wide table of Stage 1 scrolled sideways (phone): the group heading stays in view
            if width < 700:
                pg.evaluate("""() => { const t = document.querySelector('#beyond-the-data table.rows8');
                    t.closest('.tbl').scrollLeft = 300; t.scrollIntoView({block: 'start'}); window.scrollBy(0, -10); }""")
                pg.wait_for_timeout(300)
                path = f"{OUT}/p3-final-stage1-table-scrolled-{tag}.png"
                pg.screenshot(path=path)
                made.append(path)
            # the walkthrough: first step, then the Results step
            pg.evaluate("window.scrollTo(0, 0)")
            pg.click("#wt-open")
            pg.wait_for_timeout(900)
            path = f"{OUT}/p3-final-walkthrough-first-{tag}.png"
            pg.screenshot(path=path)
            made.append(path)
            for _ in range(40):
                t = pg.evaluate("() => document.getElementById('wt-title').textContent")
                if "Results" in t:
                    break
                nxt = pg.locator("#wt-next")
                if nxt.count() == 0 or nxt.is_disabled():
                    break
                nxt.click()
                pg.wait_for_timeout(450)
            pg.wait_for_timeout(700)
            path = f"{OUT}/p3-final-walkthrough-results-{tag}.png"
            pg.screenshot(path=path)
            made.append(path)
            ov = pg.evaluate("() => { const b = document.querySelector('.wt-body'); return b ? b.scrollWidth - b.clientWidth : null; }")
            print(f"{tag}: walkthrough at {t!r}, pop-out overflow {ov} px")
            pg.keyboard.press("Escape")
            pg.wait_for_timeout(400)
            o = pg.evaluate(OVERFLOW_JS)
            print(f"{tag}: console errors {len(errs)} {errs[:3]}; page overflow {o['overflow']} px")
            ctx.close()
        br.close()
    print(len(made), "shots")
    for m in made:
        print(" ", os.path.basename(m))


if __name__ == "__main__":
    main()

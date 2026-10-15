#!/usr/bin/env python3
"""The critic-in-action section inside the page's full walkthrough pop-out: open it, step to the section, check that
both charts drew at the pop-out's width, take a screenshot, close it, and check that the section is back in place."""
import os
import sys
import tempfile

from playwright.sync_api import sync_playwright

sys.path.insert(0, "/mnt/c/Users/David/dev/bca-shift-work/page/scripts")
import shot  # noqa: E402

PAGE = os.environ.get("QV_PAGE") or "/mnt/c/Users/David/dev/bca-shift-work/qviewer/test_page.html"   # QV_PAGE=path checks another page (read only)
OUT = "/mnt/c/Users/David/dev/bca-shift-work/qviewer/shots"


def main():
    frag = open(PAGE, encoding="utf-8").read()
    tmp = tempfile.mkdtemp(prefix="qvw-")
    path = os.path.join(tmp, "page.html")
    open(path, "w", encoding="utf-8").write(shot.wrap(frag, "light"))
    errs, ok = [], True
    with sync_playwright() as p:
        b = p.chromium.launch()
        for w, sc in ((1200, "dark"), (390, "light")):
            pg = b.new_context(viewport={"width": w, "height": 900 if w > 700 else 844}, color_scheme=sc).new_page()
            pg.on("console", lambda m: errs.append(f"{m.type}: {m.text}") if m.type in ("error", "warning") else None)
            pg.on("pageerror", lambda e: errs.append("pageerror: " + str(e)))
            pg.goto("file://" + path, wait_until="load")
            pg.wait_for_timeout(500)
            pg.click("#wt-open")
            pg.wait_for_timeout(400)
            for _ in range(40):
                if pg.text_content("#wt-title").strip() == "The critic in action":
                    break
                pg.click("#wt-next")
                pg.wait_for_timeout(260)
            title = pg.text_content("#wt-title").strip()
            count = pg.text_content("#wt-count").strip()
            pg.wait_for_timeout(400)
            inside = pg.evaluate("!!document.querySelector('#wt-stage #critic-in-action')")
            sw = pg.evaluate("[document.getElementById('qv-svg1').viewBox.baseVal.width, document.getElementById('qv-svg1').getBoundingClientRect().width,"
                             " document.getElementById('qv-svg2').viewBox.baseVal.width, document.getElementById('qv-svg2').getBoundingClientRect().width]")
            print(f"[{w} {sc}] title {title!r}, {count}; section inside the pop-out: {inside}; chart widths (viewBox, shown): {sw}")
            ok &= title == "The critic in action" and inside and abs(sw[0] - sw[1]) < 2 and abs(sw[2] - sw[3]) < 2
            pg.click("#qv-play")
            pg.wait_for_timeout(700)
            pg.screenshot(path=os.path.join(OUT, f"{w}-{sc}-walkthrough.png"))
            pg.click("#wt-next")
            pg.wait_for_timeout(500)
            st = pg.evaluate("window.qvState()")
            print(f"   after leaving the section: playing {st['playing']}")
            ok &= not st["playing"]
            pg.click("#wt-close")
            pg.wait_for_timeout(400)
            back = pg.evaluate("document.getElementById('critic-in-action').parentElement.tagName.toLowerCase() + ' / next: ' + document.getElementById('critic-in-action').nextElementSibling.id")
            print(f"   after closing: the section sits in {back}")
            ok &= back == "main / next: methods"
            pg.context.close()
        b.close()
    print(f"console errors/warnings: {len(errs)} {errs[:4]}")
    ok &= not errs
    print("WALKTHROUGH CHECKS PASSED" if ok else "WALKTHROUGH CHECKS FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())

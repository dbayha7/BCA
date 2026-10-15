#!/usr/bin/env python3
"""Final screenshots that need a pointer, a tap or a key (shot.py cannot hover): the chart readouts at 390 px (touch)
and 1200 px, a table cell's page tooltip, a header description, and the section inside the walkthrough pop-out.
Saves shots/final-*.png. Run (WSL): taskset -c 2 nice -n 10 /opt/shot-venv/bin/python final_shots_extra.py"""
import os
import sys
import tempfile

from playwright.sync_api import sync_playwright

sys.path.insert(0, "/mnt/c/Users/David/dev/bca-shift-work/page/scripts")
import shot  # noqa: E402

HERE = "/mnt/c/Users/David/dev/bca-shift-work/qviewer"
OUT = os.path.join(HERE, "shots")


def wrapped():
    tmp = tempfile.mkdtemp(prefix="qvx-")
    p = os.path.join(tmp, "page.html")
    open(p, "w", encoding="utf-8").write(shot.wrap(open(os.environ.get("QV_PAGE") or os.path.join(HERE, "test_page.html"), encoding="utf-8").read(), "light"))
    return p


def main():
    path = wrapped()
    with sync_playwright() as p:
        b = p.chromium.launch()
        # 390 px, touch: tap each chart; the readout sits under the chart
        ctx = b.new_context(viewport={"width": 390, "height": 844}, color_scheme="light", has_touch=True, is_mobile=True, device_scale_factor=2)
        pg = ctx.new_page()
        pg.goto("file://" + path)
        pg.wait_for_timeout(900)
        pg.evaluate("window.qvSet({q:'expert', c:'gap', s:'strong', view:'rel', range:'hi', t:40})")
        for n, frac in (("1", 0.62), ("2", 0.45)):
            pg.evaluate(f"document.getElementById('qv-c{n}').scrollIntoView({{block:'start'}}); window.scrollBy(0, -60)")
            pg.wait_for_timeout(250)
            bx = pg.locator(f"#qv-svg{n}").bounding_box()
            pg.touchscreen.tap(bx["x"] + bx["width"] * frac, bx["y"] + bx["height"] * 0.35)
            pg.wait_for_timeout(300)
            pg.screenshot(path=os.path.join(OUT, f"final-390-light-touch-c{n}-readout.png"))
            pg.touchscreen.tap(4, 4)
            pg.wait_for_timeout(200)
        ctx.close()
        # 1200 px dark: mouse readout, a table cell's page tooltip, a header description
        ctx = b.new_context(viewport={"width": 1200, "height": 800}, color_scheme="dark")
        pg = ctx.new_page()
        pg.goto("file://" + path)
        pg.wait_for_timeout(900)
        pg.evaluate("window.qvSet({q:'expert', c:'gap', s:'strong', view:'rel', range:'hi', t:40})")
        pg.evaluate("document.getElementById('qv-c1').scrollIntoView({block:'center'})")
        pg.wait_for_timeout(200)
        bx = pg.locator("#qv-svg1").bounding_box()
        pg.mouse.move(bx["x"] + bx["width"] * 0.6, bx["y"] + bx["height"] * 0.45)
        pg.wait_for_timeout(300)
        pg.screenshot(path=os.path.join(OUT, "final-1200-dark-c1-readout.png"))
        pg.mouse.move(2, 2)
        pg.evaluate("document.getElementById('qv-tables').open = true")
        pg.wait_for_timeout(300)
        for sel, name in (("#qv-tbl-ep tbody tr:nth-child(42) td:nth-child(5)", "tip-ep-cell"),
                          ("#qv-tbl-sweep thead th:nth-child(3) span", "tip-sweep-header"),
                          ("#qv-tbl-ctx tbody tr:nth-child(5) td:nth-child(3)", "tip-ctx-cell")):
            pg.mouse.move(2, 2)
            pg.wait_for_timeout(250)
            loc = pg.locator(sel).first
            loc.scroll_into_view_if_needed()
            pg.evaluate("window.scrollBy(0, 120)")
            pg.wait_for_timeout(150)
            loc.hover()
            pg.wait_for_timeout(500)
            pg.screenshot(path=os.path.join(OUT, f"final-1200-dark-{name}.png"))
        ctx.close()
        # the walkthrough pop-out, both widths
        for w, sc in ((1200, "dark"), (390, "light")):
            pg = b.new_context(viewport={"width": w, "height": 800 if w > 700 else 844}, color_scheme=sc).new_page()
            pg.goto("file://" + path)
            pg.wait_for_timeout(900)
            pg.click("#wt-open")
            pg.wait_for_timeout(300)
            for _ in range(40):
                if pg.text_content("#wt-title").strip() == "The critic in action":
                    break
                pg.click("#wt-next")
                pg.wait_for_timeout(220)
            pg.wait_for_timeout(500)
            pg.screenshot(path=os.path.join(OUT, f"final-{w}-{sc}-walkthrough.png"))
            pg.context.close()
        b.close()
    print("saved")


if __name__ == "__main__":
    main()

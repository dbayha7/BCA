#!/usr/bin/env python3
"""Tooltip text pass: screenshots of real tooltips, hovered with the mouse, at 390 px light and 1200 px dark.
Read-only on the page. For each sample it saves shots/ta-<sample>-<width>-<scheme>.png, clipped to the tooltip and the
element it explains (with a margin), and prints the tooltip's text, whether its box hides any text, console errors and
the page's horizontal overflow. Run:
  cd /mnt/c/Users/David/dev/bca-shift-work/page && /opt/shot-venv/bin/python scripts/ta_shots.py
"""
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from shot import wrap, OVERFLOW_JS  # noqa: E402
from playwright.sync_api import sync_playwright  # noqa: E402

PAGE_DIR = "/mnt/c/Users/David/dev/bca-shift-work/page"
SHOTS = os.path.join(PAGE_DIR, "shots")

# name, how to find the element, where on it to put the pointer (fractions of its box)
SAMPLES = [
    ("stall-cell", "#arguments table >> nth=0 >> td >> nth=0", (0.5, 0.5)),
    ("stalls-cell", "#action-space table >> nth=0 >> tbody tr >> nth=0 >> td >> nth=-1", (0.5, 0.5)),
    ("term-td3bc", '#glance [data-tip-title="TD3+BC"]', (0.5, 0.5)),
    ("legend-dynamics", "#dynamics div.math-display mjx-container > svg >> nth=0", (0.5, 0.5)),
    ("chart-band", "#task-chart .pt-hits rect >> nth=7", (0.5, 0.4)),
    ("channel-thruster", "#channel", None),          # the pointer goes to thruster 1 (computed below)
    ("tuple-notes", 'td:has-text("gapped critic")', (0.3, 0.5)),
]

TIP = r"""() => { const t = document.getElementById('tt'); const r = t.getBoundingClientRect();
  return { hidden: t.hidden, text: t.innerText, cls: t.className, hides: t.scrollHeight > t.clientHeight + 1,
           box: [r.left, r.top, r.right, r.bottom] }; }"""


def run(b, f, width, scheme):
    height = 844 if width < 700 else 800
    ctx = b.new_context(viewport={"width": width, "height": height}, color_scheme=scheme)
    pg = ctx.new_page()
    errs = []
    pg.on("pageerror", lambda e: errs.append("page error: " + str(e)))
    pg.on("console", lambda m: errs.append(f"console {m.type}: {m.text}") if m.type in ("error", "warning") else None)
    pg.goto("file://" + f, wait_until="load")
    pg.wait_for_timeout(1200)
    pg.evaluate("document.querySelectorAll('details').forEach(d => d.open = true)")
    pg.wait_for_timeout(300)
    over0 = pg.evaluate(OVERFLOW_JS)["overflow"]
    out = []
    for name, sel, at in SAMPLES:
        loc = pg.locator(sel).first
        loc.scroll_into_view_if_needed()
        pg.evaluate("el => el.scrollIntoView({block: 'center'})", loc.element_handle())
        pg.wait_for_timeout(250)
        bx = loc.bounding_box()
        pg.mouse.move(2, 2)
        pg.wait_for_timeout(250)
        if at is None:                                   # thruster 1 of the channel drawing
            narrow = bx["width"] < 560
            x = bx["x"] + bx["width"] / 5
            y = bx["y"] + (66 + 176 / 2 if narrow else 60 + 196 / 2)
        else:
            x, y = bx["x"] + bx["width"] * at[0], bx["y"] + bx["height"] * at[1]
        pg.mouse.move(x, y, steps=4)
        pg.wait_for_timeout(700)
        t = pg.evaluate(TIP)
        if t["hidden"]:
            out.append((name, None, "NOT SHOWN"))
            continue
        l, tp, r, bt = t["box"]
        pad = 16
        cx0 = max(0, min(l, bx["x"]) - pad)
        cy0 = max(0, min(tp, max(bx["y"], tp - 120)) - pad)
        cx1 = min(width, max(r, min(bx["x"] + bx["width"], r + 200)) + pad)
        cy1 = min(height, max(bt, min(bx["y"] + bx["height"], bt + 160)) + pad)
        path = os.path.join(SHOTS, f"ta-{name}-{width}-{scheme}.png")
        pg.screenshot(path=path, clip={"x": cx0, "y": cy0, "width": cx1 - cx0, "height": cy1 - cy0})
        out.append((name, path, t))
    over1 = pg.evaluate(OVERFLOW_JS)["overflow"]
    ctx.close()
    return out, errs, over0, over1


def main():
    d = tempfile.mkdtemp(prefix="tashots-")
    f = os.path.join(d, "page.html")
    open(f, "w", encoding="utf-8").write(wrap(open(os.path.join(PAGE_DIR, "thrusterchain.html"), encoding="utf-8").read(), "light"))
    with sync_playwright() as p:
        b = p.chromium.launch()
        for width, scheme in ((390, "light"), (1200, "dark")):
            out, errs, o0, o1 = run(b, f, width, scheme)
            print(f"\n===== {width} {scheme}: console errors and warnings {len(errs)}, page errors "
                  f"{sum(e.startswith('page error') for e in errs)}, horizontal overflow {o0} px after load, {o1} px after hovering")
            for e in errs:
                print("   ", e[:300])
            for name, path, t in out:
                if path is None:
                    print(f"[{name}] {t}")
                    continue
                print(f"[{name}] {os.path.basename(path)}  class={t['cls']!r} hides text={t['hides']}")
                print("    " + t["text"].replace("\n", "\n    ")[:1800])
        b.close()


if __name__ == "__main__":
    main()

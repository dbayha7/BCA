"""Read-only browser test of page pass 4, edit 2: the four-throttle widget and the stall chart.
Clicks the presets, switches the data, moves sliders with the keyboard and the pointer, toggles the histograms, and
checks the readouts against the formulas; screenshots the figures in a few states.

  cd /mnt/c/Users/David/dev/bca-shift-work/page && /opt/shot-venv/bin/python scripts/p4_e2_ui_test.py [--width 1200] [--scheme dark]
"""
import argparse
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from shot import wrap, DEFAULT_PAGE, OVERFLOW_JS  # noqa: E402
from playwright.sync_api import sync_playwright  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--width", type=int, default=1200)
ap.add_argument("--scheme", default="light")
ap.add_argument("--reduced-motion", action="store_true")
ap.add_argument("--page", default=DEFAULT_PAGE)
ap.add_argument("--out", default="tmp/p4/edit2/shots")
a = ap.parse_args()
T = f"{a.width}-{a.scheme}"
fails, errors = [], []


def check(ok, msg):
    print(("ok   " if ok else "FAIL ") + msg)
    if not ok:
        fails.append(msg)


frag = open(a.page, encoding="utf-8").read()
tmp = os.path.join(tempfile.mkdtemp(prefix="e2ui-"), "page.html")
open(tmp, "w", encoding="utf-8").write(wrap(frag, "light"))
with sync_playwright() as p:
    br = p.chromium.launch()
    ctx = br.new_context(viewport={"width": a.width, "height": 900}, color_scheme=a.scheme,
                         reduced_motion="reduce" if a.reduced_motion else "no-preference", has_touch=a.width < 700)
    pg = ctx.new_page()
    pg.on("console", lambda m: errors.append(m.text) if m.type in ("error", "warning") else None)
    pg.on("pageerror", lambda e: errors.append(str(e)))
    pg.goto("file://" + tmp, wait_until="load")
    pg.wait_for_timeout(800)
    txt = lambda sel: pg.evaluate(f"document.querySelector({sel!r}).textContent")

    # ---------------- the four-throttle widget ----------------
    pg.click('#as-widget button[data-preset="full"]')
    pg.wait_for_timeout(200)
    o1 = txt("#aw-o1")
    check("100% upstream" in o1 and "0.50" in o1 and "stalls" in o1 and "0.300" in o1, f"full-throttle preset, thruster 1: {o1}")
    cost = txt("#aw-cost")
    check("cost 0.458" in cost, f"full-throttle preset costs 0.458: {cost[:60]}")
    ball = txt("#aw-ball")
    check("0.242 from it" in ball and "inside" in ball, f"full-throttle preset inside the main ball: {ball[-90:]}")
    msg = txt("#aw-msg")
    check("60 of the 42,500" in msg and "Stall." in msg, f"past-80% message: {msg[:90]}")
    stalled = pg.evaluate("document.querySelector('.aw-thr[data-k=\"1\"]').classList.contains('is-stall')")
    check(stalled, "thruster 1's block is marked as stalling")
    pg.locator("#as-widget").screenshot(path=f"{a.out}/ui-aw-full-{T}.png")
    pg.click('#as-widget button[data-preset="capped"]')
    pg.wait_for_timeout(200)
    check("0.350" in txt("#aw-cost") and "0.069 from it" in txt("#aw-ball"), "capped preset: cost 0.350, 0.069 from the centre")
    check("hardest push" in txt("#aw-msg"), f"hold message: {txt('#aw-msg')[:80]}")
    pg.click('#as-widget button[data-preset="normal"]')
    pg.wait_for_timeout(200)
    o1 = txt("#aw-o1")
    check("41% upstream" in o1 and "0.41" in o1 and "stalls" not in o1, f"normal driving preset: {o1}")
    check("outside" in txt("#aw-ball") and "0.495" in txt("#aw-ball"), f"normal driving outside the ball: {txt('#aw-ball')[-80:]}")
    # keyboard: focus thruster 3, End, then the arrow keys
    pg.focus("#aw-a3")
    pg.keyboard.press("End")
    pg.wait_for_timeout(100)
    check(pg.evaluate("document.getElementById('aw-a3').value") == "1" and "100% downstream" in txt("#aw-o3"), "End puts thruster 3 at full downstream")
    check("thruster 3 is past 80%" in txt("#aw-msg"), f"other-thruster line: {txt('#aw-msg')[-160:]}")
    pg.keyboard.press("PageDown")
    pg.keyboard.press("ArrowLeft")
    pg.wait_for_timeout(100)
    v3 = pg.evaluate("document.getElementById('aw-a3').value")
    check(v3 == "0.89", f"Page Down and an arrow: {v3}")
    vt = pg.evaluate("document.getElementById('aw-a3').getAttribute('aria-valuetext')")
    check("89% downstream" in vt and "stalls" in vt, f"aria-valuetext: {vt}")
    # the weighting test's data
    pg.check('input[name="aw-data"][value="iw"]')
    pg.wait_for_timeout(200)
    pg.click('#as-widget button[data-preset="capped"]')
    pg.wait_for_timeout(200)
    b1 = txt("#aw-b1")
    check("5,378 of the 50,000" in b1 and "4,950 of them the 80% hold" in b1 and "414 of the 42,500" in b1, f"weighting test, hold bin: {b1}")
    check("99.8%" in txt("#aw-msg"), f"weighting-test hold message: {txt('#aw-msg')[:120]}")
    check(pg.evaluate("document.getElementById('aw-key-rm').dataset.k") == "hold", "legend key turns to the hold's neutral hatch")
    t11 = pg.evaluate("document.getElementById('aw-t1-1f').textContent")
    print("     weighting test, thruster 1, band −80% to −60%, all rows:", t11)
    pg.locator("#as-widget").screenshot(path=f"{a.out}/ui-aw-iw-{T}.png")
    # hover on a strip shows that band, leaving restores
    pg.evaluate("document.getElementById('aw-s1').scrollIntoView({block: 'center'})")
    pg.wait_for_timeout(200)
    box = pg.locator("#aw-s1").bounding_box()
    pg.mouse.move(box["x"] + box["width"] * 0.5 + 2, box["y"] + box["height"] * 0.5)
    pg.wait_for_timeout(150)
    hb = txt("#aw-b1")
    check("from 0% to 4% downstream" in hb, f"hover over the middle of the strip: {hb[:70]}")
    pg.mouse.move(5, 5)
    pg.wait_for_timeout(150)
    check("from 76% to 80% upstream" in txt("#aw-b1"), f"leaving restores the slider's band: {txt('#aw-b1')[:70]}")
    pg.check('input[name="aw-data"][value="main"]')
    pg.wait_for_timeout(150)

    # ---------------- the stall chart ----------------
    pg.locator("#stall-chart").scroll_into_view_if_needed()
    check(txt("#sc-vs").startswith("0.65") and "stalls" in txt("#sc-vs") and txt("#sc-vc") == "0.243", "stall chart starts at 90%")
    pg.focus("#sc-a")
    pg.keyboard.press("PageDown")
    pg.wait_for_timeout(100)
    check(txt("#sc-va") == "80%" and txt("#sc-vs") == "0.80" and "best throttle" in txt("#sc-val"), f"80%: {txt('#sc-va')} {txt('#sc-val')[-60:]}")
    pg.keyboard.press("End")
    pg.wait_for_timeout(100)
    check(txt("#sc-vs").startswith("0.50") and "0.297" in txt("#sc-val") and "60 of the 42,500" in txt("#sc-bin") or "59 of" in txt("#sc-bin"),
          f"100%: {txt('#sc-val')[-90:]} | {txt('#sc-bin')}")
    pg.keyboard.press("Home")
    pg.wait_for_timeout(100)
    check(txt("#sc-va") == "0%" and "0.691" in txt("#sc-val"), f"0%: {txt('#sc-val')[-110:]}")
    pg.evaluate("document.getElementById('sc-svg').scrollIntoView({block: 'center'})")
    pg.wait_for_timeout(200)
    sb = pg.locator("#sc-svg").bounding_box()
    pg.mouse.move(sb["x"] + sb["width"] * 0.6, sb["y"] + sb["height"] * 0.3)
    pg.wait_for_timeout(150)
    check("(pointer)" in txt("#sc-va"), f"hover shows the pointer's throttle: {txt('#sc-va')}")
    pg.mouse.click(sb["x"] + sb["width"] * 0.93, sb["y"] + sb["height"] * 0.3)
    pg.wait_for_timeout(150)
    v = pg.evaluate("Number(document.getElementById('sc-a').value)")
    check(0.85 <= v <= 0.97, f"a click sets the slider: {v}")
    pg.mouse.move(5, 5)
    pg.wait_for_timeout(100)
    pg.uncheck("#sc-full")
    pg.wait_for_timeout(100)
    n_hatch = pg.evaluate("document.querySelectorAll('#sc-svg rect[fill=\"url(#sc-hx)\"]').length")
    check(n_hatch == 0, f"full data off: no hatched bars ({n_hatch})")
    pg.check("#sc-full")
    pg.wait_for_timeout(100)
    n_hatch = pg.evaluate("document.querySelectorAll('#sc-svg rect[fill=\"url(#sc-hx)\"]').length")
    check(n_hatch >= 8, f"both on: the removed part is hatched ({n_hatch} bars)")
    pg.locator("#stall-chart").screenshot(path=f"{a.out}/ui-sc-{T}.png")
    ov = pg.evaluate(OVERFLOW_JS)
    check(ov["overflow"] <= 0, f"no horizontal overflow ({ov['overflow']})")
    # the formula tooltips of grouped legends hold every group
    tpl = pg.evaluate("[...document.querySelectorAll('template[id^=symkey-all-]')].map(t => [t.id, t.content.querySelectorAll('div').length, t.content.querySelectorAll('p').length])")
    print("     grouped-legend templates:", tpl)
    check(len(tpl) >= 4 and all(d > 0 and q > 1 for _, d, q in tpl), "every grouped legend beside a formula has a full template")
    br.close()
print(f"console/page errors: {len(errors)}", errors[:5])
print("PASS" if not fails and not errors else f"{len(fails)} FAILS, {len(errors)} errors")
sys.exit(1 if fails or errors else 0)

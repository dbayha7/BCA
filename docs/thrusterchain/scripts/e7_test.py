#!/usr/bin/env python3
"""Checks and final screenshots for the review fixes of the tooltip pass, at 1200 px dark (mouse and keyboard) and
390 px light (touch). Prints each check; exit status 1 on any failure.

  cd /mnt/c/Users/David/dev/bca-shift-work/page && /opt/shot-venv/bin/python scripts/e7_test.py [--page PATH] [--prefix tt-final]
"""
import argparse
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from shot import wrap, OVERFLOW_JS  # noqa: E402
from playwright.sync_api import sync_playwright  # noqa: E402

PAGE_DIR = "/mnt/c/Users/David/dev/bca-shift-work/page"
SHOTS = os.path.join(PAGE_DIR, "shots")
FAIL = []


def check(ok, msg):
    print(("  ok   " if ok else "  FAIL ") + msg)
    if not ok:
        FAIL.append(msg)


TIP_JS = r"""
() => { const t = document.getElementById('tt'); if (!t) return null;
  const r = t.getBoundingClientRect();
  return { hidden: t.hidden, text: t.textContent, wide: t.classList.contains('tt-wide'), html: t.innerHTML,
           parent: t.parentElement ? (t.parentElement.id || t.parentElement.tagName) : null,
           l: r.left, t: r.top, r: r.right, b: r.bottom, w: r.width, h: r.height,
           vw: document.documentElement.clientWidth, vh: innerHeight, sh: t.scrollHeight, ch: t.clientHeight }; }
"""
STOPS_JS = r"""() => { const F = 'a[href], button:not([disabled]), input:not([disabled]), select, textarea, summary, [tabindex]';
  const all = [...document.querySelectorAll(F)].filter(e => e.tabIndex >= 0 && e.getAttribute('tabindex') !== '-1'
    && e.getClientRects().length && !e.closest('[hidden], dialog:not([open])'));
  const sw = document.getElementById('tt-switch'), opt = document.querySelector('#sim .opt');
  const kinds = {}; all.forEach(e => { const k = e.matches('.sym') ? 'symbols' : e.matches('[data-tt]') ? 'tables'
    : e.matches('[data-tip]:not(button):not(a):not(input)') ? 'terms and readouts' : 'native'; kinds[k] = (kinds[k] || 0) + 1; });
  return { total: all.length, switchToPanel: all.indexOf(opt) - all.indexOf(sw), kinds }; }"""


def tip(pg):
    return pg.evaluate(TIP_JS)


def onscreen(t):
    return t and t["l"] >= 0 and t["t"] >= 0 and t["r"] <= t["vw"] + 0.5 and t["b"] <= t["vh"] + 0.5


def center(pg, sel, block="center"):
    pg.evaluate(f"el => {{ const d = el.closest('details'); if (d) d.open = true; el.scrollIntoView({{block: '{block}'}}); }}",
                pg.locator(sel).first.element_handle())
    pg.wait_for_timeout(700)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--page", default=os.path.join(PAGE_DIR, "thrusterchain.html"))
    ap.add_argument("--prefix", default="tt-final")
    a = ap.parse_args()
    d = tempfile.mkdtemp(prefix="e7-")
    f = os.path.join(d, "page.html")
    open(f, "w", encoding="utf-8").write(wrap(open(a.page, encoding="utf-8").read(), "light"))
    url = "file://" + f
    P = a.prefix

    with sync_playwright() as p:
        b = p.chromium.launch()
        errors = []

        def new(width, scheme, touch=False):
            ctx = b.new_context(viewport={"width": width, "height": 844 if width < 700 else 800}, color_scheme=scheme,
                                has_touch=touch, is_mobile=touch)
            pg = ctx.new_page()
            pg.on("console", lambda m: errors.append(f"[{width}] {m.type}: {m.text}") if m.type in ("error", "warning") else None)
            pg.on("pageerror", lambda e: errors.append(f"[{width}] pageerror: {e}"))
            pg.goto(url, wait_until="load")
            pg.wait_for_timeout(900)
            return ctx, pg

        def shot(pg, name):
            pg.screenshot(path=os.path.join(SHOTS, f"{P}-{name}.png"))

        def shot_el(pg, name, sel):
            pg.locator(sel).first.screenshot(path=os.path.join(SHOTS, f"{P}-{name}.png"))

        # ======================= 1200 px, dark, mouse and keyboard =======================
        print("== 1200 px dark (mouse and keyboard)")
        ctx, pg = new(1200, "dark")
        ov = pg.evaluate(OVERFLOW_JS)
        check(ov["overflow"] <= 0, f"no horizontal overflow at load ({ov['overflow']} px)")
        on = pg.evaluate(STOPS_JS)
        pg.click("#tt-switch")
        off = pg.evaluate(STOPS_JS)
        pg.click("#tt-switch")
        pg.mouse.move(5, 5)
        print(f"       Tab stops: on {on}, off {off}")
        check(on["total"] < 442 and on["switchToPanel"] <= 36, f"fewer Tab stops: {on['total']} (was 442), switch to panel {on['switchToPanel']} (was 36)")
        check(pg.evaluate("!document.querySelector('details.ctrl-guide').open"), "the controller guide starts closed")

        # finding 1: a cell clicked with the mouse leaves the arrow keys to the page
        cell = pg.locator("#observation-space td[data-tt]").filter(has_text="Position of float 2").first
        center(pg, "#observation-space table")
        cell.click()
        y0 = pg.evaluate("scrollY")
        f0 = pg.evaluate("document.activeElement.textContent.trim().slice(0, 30)")
        pg.keyboard.press("ArrowDown")
        pg.wait_for_timeout(300)
        y1 = pg.evaluate("scrollY")
        f1 = pg.evaluate("document.activeElement.textContent.trim().slice(0, 30)")
        check(y1 > y0 and f0 == f1, f"mouse-clicked cell: ArrowDown scrolls the page ({y0} -> {y1}), focus stays ({f1!r})")
        # keyboard: Tab into a table, then the arrows move between cells
        pg.mouse.move(5, 5)
        pg.evaluate("""() => { const t = document.querySelector('#observation-space table'); let p = t.previousElementSibling;
            while (p && !p.querySelector) p = p.previousElementSibling; const b = document.createElement('button'); b.id = 'probe-btn';
            b.textContent = 'probe'; t.before(b); b.focus(); }""")
        pg.keyboard.press("Tab")
        pg.wait_for_timeout(150)
        g0 = pg.evaluate("document.activeElement.textContent.trim().slice(0, 40)")
        pg.keyboard.press("ArrowDown")
        pg.wait_for_timeout(200)
        g1 = pg.evaluate("document.activeElement.textContent.trim().slice(0, 40)")
        t = tip(pg)
        check(g0 != g1 and pg.evaluate("!!document.activeElement.dataset.tt") and t and not t["hidden"],
              f"keyboard: Tab into the table, ArrowDown moves to the next cell ({g0!r} -> {g1!r}) and shows its tooltip")
        pg.evaluate("document.getElementById('probe-btn').remove()")
        shot(pg, "keyboard-cell-1200-dark")
        pg.keyboard.press("Escape")

        # finding 6: the whole tuple, nothing cut
        center(pg, "#weighting-test table")
        longest = pg.evaluate("""() => { let best = null, n = 0;
            document.querySelectorAll('td[data-tt="c"]').forEach(c => { const l = c.parentElement.textContent.length;
              if (l > n && c.getClientRects().length) { n = l; best = c; } });
            best.id = best.id || 'probe-long'; best.scrollIntoView({block: 'center'}); return [best.id, n, best.closest('section').id]; }""")
        pg.wait_for_timeout(200)
        pg.hover("#" + longest[0])
        pg.wait_for_timeout(450)
        t = tip(pg)
        check(t and not t["hidden"] and "…" not in t["text"] and onscreen(t) and t["sh"] <= t["ch"] + 1,
              f"longest row ({longest[1]} characters, {longest[2]}): whole tuple, no ellipsis, fits ({t and round(t['w'])}x{t and round(t['h'])}, wide {t and t['wide']})")
        shot(pg, "tuple-long-1200-dark")
        cell = pg.locator("#weighting-test td[data-tt]").filter(has_text="99%").first
        if cell.count():
            cell.scroll_into_view_if_needed()
            cell.hover()
            pg.wait_for_timeout(450)
            t = tip(pg)
            check(t and "…" not in t["text"], f"weighting test 'Passes when' row in full: {t and t['text'][:120]!r}")
            shot(pg, "tuple-wtest-1200-dark")
        pg.mouse.move(5, 5)

        # finding 5: the chart's columns
        center(pg, "#task-chart svg.chart")
        box = pg.locator("#task-chart svg.chart").first.bounding_box()
        rects = pg.evaluate("[...document.querySelectorAll('#task-chart .pt-hits rect')].map(r => { const b = r.getBoundingClientRect(); return [b.left, b.top, b.width, b.height]; })")
        check(len(rects) == 11, f"11 column targets ({len(rects)}), each {rects and round(rects[0][2])}x{rects and round(rects[0][3])} px")
        r = rects[7]
        pg.mouse.move(r[0] + r[2] / 2, r[1] + r[3] * 0.9)
        pg.wait_for_timeout(450)
        t = tip(pg)
        check(t and not t["hidden"] and t["text"].startswith("Water's push") and "Stall-aware" in t["text"]
              and "Rising-thrust thruster 1 push" in t["text"] and onscreen(t),
              f"a column away from its dots shows all three numbers: {t and t['text'][:150]!r}")
        shot(pg, "chart-column-1200-dark")
        ann = pg.locator("#task-chart text.ann").first
        ann.hover()
        pg.wait_for_timeout(300)
        check(tip(pg)["text"].startswith("The stall"), "the 80% label above the columns keeps its own tooltip")
        pg.mouse.move(5, 5)

        # finding 8: the channel's push when the wave pushes against the current
        center(pg, "#channel")
        found = None
        for k in range(100):
            w = pg.evaluate("(() => { const R = D.roll[key()]; return R.s[S.t][11]; })()")
            if w < -0.05:
                found = k
                break
            pg.click("#fwd")
        cb = pg.locator("#channel").bounding_box()
        pg.mouse.move(cb["x"] + cb["width"] / 2, cb["y"] + 30)
        pg.wait_for_timeout(450)
        t = tip(pg)
        check(found is not None and t and "c + w₁ = " in t["text"] and "+ −" not in t["text"] and " − " in t["text"].split("=")[1],
              f"channel: a wave against the current reads as a subtraction: {t and t['text'][:80]!r}")
        shot(pg, "channel-minus-1200-dark")
        pg.mouse.move(5, 5)

        # finding 3: the Play button's tooltip follows its label
        pg.hover("#play")
        pg.wait_for_timeout(400)
        pg.click("#play")
        pg.wait_for_timeout(300)
        t = tip(pg)
        lab = pg.evaluate("document.getElementById('play').textContent.trim()")
        check(lab == "Pause" and t and t["text"].startswith("Pause"), f"playing: the button reads {lab!r} and its tooltip {t and t['text'][:40]!r}")
        pg.click("#play")
        pg.wait_for_timeout(200)
        check(tip(pg)["text"].startswith("Play"), "paused again: the tooltip reads Play")
        pg.mouse.move(5, 5)

        # finding 24: a display formula shows its legend
        center(pg, "#methods div.math-display")
        f = pg.locator("#methods div.math-display mjx-container > svg").first
        f.hover()
        pg.wait_for_timeout(450)
        t = tip(pg)
        n_ref = pg.evaluate("document.querySelectorAll('div.math-display svg[data-tip-ref]').length")
        check(n_ref == 32 and t and not t["hidden"] and t["text"].startswith("What the symbols mean") and onscreen(t),
              f"display formulas show their legend ({n_ref} of 32): {t and t['text'][:90]!r}")
        shot(pg, "formula-legend-1200-dark")
        pg.mouse.move(5, 5)

        # texts that changed
        txt = pg.evaluate("document.body.innerText")
        for s in ("With no throttle the floats stop drifting further, but never come back to rest",
                  "shown rounded to whole steps under each float: 63 for float 1 (62.8)",
                  "Only the state number" if False else "all but the state number t = 1",
                  "An arrow after it (↓ or ↑)", "θ, ρ"):
            check(s in txt, f"page text: {s!r}")
        center(pg, "#sim .sim-keys")
        shot_el(pg, "leadin-1200-dark", ".sim-keys")
        pg.hover("#opt-world-no_stall")
        pg.wait_for_timeout(450)
        t = tip(pg)
        check(t and "only 60 of the 42,500" in t["text"], "the no-stall world's tooltip names the data after the removal")
        shot(pg, "world-nostall-1200-dark")
        pg.mouse.move(5, 5)
        center(pg, "details.ctrl-guide")
        pg.evaluate("document.querySelector('details.ctrl-guide').open = true")
        pg.wait_for_timeout(150)
        shot_el(pg, "guide-1200-dark", "details.ctrl-guide")
        pg.evaluate("document.querySelector('details.ctrl-guide').open = false")
        center(pg, "p.worked")
        shot_el(pg, "worked-1200-dark", "#arguments")

        # finding 1 in the walkthrough: after a click in a table the arrows still change sections
        pg.evaluate("window.scrollTo(0, 0)")
        pg.click("#wt-open")
        pg.wait_for_timeout(500)
        for _ in range(12):
            if pg.evaluate("document.getElementById('wt-title').textContent.trim()") == "Action Space":
                break
            pg.click("#wt-next")
            pg.wait_for_timeout(300)
        c = pg.locator("#wt-stage td[data-tt]").first
        pg.evaluate("el => el.scrollIntoView({block: 'center'})", c.element_handle())
        c.click()
        pg.keyboard.press("ArrowRight")
        pg.wait_for_timeout(600)
        ttl = pg.evaluate("document.getElementById('wt-title').textContent.trim()")
        check(ttl != "Action Space", f"walkthrough: after a click in a table, ArrowRight moves on ({ttl!r})")
        shot(pg, "walkthrough-1200-dark")
        pg.keyboard.press("Escape")
        pg.keyboard.press("Escape")
        pg.wait_for_timeout(300)
        ov = pg.evaluate(OVERFLOW_JS)
        check(ov["overflow"] <= 0, f"no horizontal overflow after the checks ({ov['overflow']} px)")
        ctx.close()

        # ======================= 390 px, light, touch =======================
        print("== 390 px light (touch)")
        ctx, pg = new(390, "light", touch=True)
        cdp = ctx.new_cdp_session(pg)

        def hold(sel, ms=700):
            el = pg.locator(sel).first
            bb = el.bounding_box()
            x, y = bb["x"] + bb["width"] / 2, bb["y"] + bb["height"] / 2
            cdp.send("Input.dispatchTouchEvent", {"type": "touchStart", "touchPoints": [{"x": x, "y": y}]})
            pg.wait_for_timeout(ms)
            cdp.send("Input.dispatchTouchEvent", {"type": "touchEnd", "touchPoints": []})
            pg.wait_for_timeout(250)

        ov = pg.evaluate(OVERFLOW_JS)
        check(ov["overflow"] <= 0, f"no horizontal overflow at load ({ov['overflow']} px)")
        shot(pg, "top-390-light")
        h = pg.evaluate("document.getElementById('sim').getBoundingClientRect().height")
        print(f"       live panel height at 390: {round(h)} px")
        # a tap on Play or Step works it and shows no tooltip
        center(pg, "#fwd")
        st0 = pg.evaluate("document.getElementById('tLabel').textContent")
        shown = []
        for _ in range(4):
            pg.locator("#fwd").tap()
            pg.wait_for_timeout(250)
            shown.append(not tip(pg)["hidden"])
        st1 = pg.evaluate("document.getElementById('tLabel').textContent")
        check(not any(shown) and st0 != st1, f"four taps on Step: no tooltip {shown}, state {st0!r} -> {st1!r}")
        pg.locator("#play").tap()
        pg.wait_for_timeout(600)
        t = tip(pg)
        check(t["hidden"] and pg.evaluate("document.getElementById('play').textContent.trim()") == "Pause",
              "a tap on Play plays, with no tooltip over the gauges")
        shot(pg, "playing-390-light")
        pg.locator("#play").tap()
        pg.wait_for_timeout(200)
        # press and hold an option: its tooltip shows and the option is not picked
        center(pg, "#opt-ctrl-stall_aware")
        before = pg.evaluate("document.getElementById('opt-ctrl-stall_aware').getAttribute('aria-pressed')")
        hold("#opt-ctrl-stall_aware")
        t = tip(pg)
        after = pg.evaluate("document.getElementById('opt-ctrl-stall_aware').getAttribute('aria-pressed')")
        check(t and not t["hidden"] and "Stall-aware controller" in t["text"] and onscreen(t) and before == after,
              f"press and hold a controller: its tooltip, and the option is not picked ({before} -> {after})")
        shot(pg, "hold-ctrl-390-light")
        pg.locator("#opt-ctrl-recording").tap()
        pg.wait_for_timeout(250)
        check(tip(pg)["hidden"] and pg.evaluate("document.getElementById('opt-ctrl-recording').getAttribute('aria-pressed')") == "true",
              "a tap on another option picks it and closes the tooltip")
        # finding 4: a tap on the State slider shows its tooltip, which goes after 2.5 s
        center(pg, "#tSlider")
        pg.locator("#tSlider").tap()
        pg.wait_for_timeout(250)
        t = tip(pg)
        check(t and not t["hidden"] and t["text"].startswith("State") and onscreen(t), f"a tap on the State slider shows its tooltip: {t and t['text'][:50]!r}")
        shot(pg, "slider-390-light")
        pg.wait_for_timeout(2700)
        check(tip(pg)["hidden"], "the slider's tooltip goes by itself 2.5 s later")
        center(pg, "#td-slider")
        pg.locator("#td-slider").tap()
        pg.wait_for_timeout(250)
        check(not tip(pg)["hidden"], "the demo's slider too")
        # a term, a table cell (cards), a chart column, a formula by tap
        center(pg, "#task i[data-tip]")
        pg.locator("#task i[data-tip]").first.tap()
        pg.wait_for_timeout(300)
        check(not tip(pg)["hidden"], "a tap on a term still shows its tooltip")
        center(pg, "#task-chart svg.chart")
        rects = pg.evaluate("[...document.querySelectorAll('#task-chart .pt-hits rect')].map(r => { const b = r.getBoundingClientRect(); return [b.left, b.top, b.width, b.height]; })")
        r = rects[3]
        pg.touchscreen.tap(r[0] + r[2] / 2, r[1] + 8)
        pg.wait_for_timeout(300)
        t = tip(pg)
        check(t and not t["hidden"] and t["text"].startswith("Water's push") and onscreen(t),
              f"a tap in a chart column ({round(r[2])}x{round(r[3])} px), away from its dots: {t and t['text'][:60]!r}")
        shot(pg, "chart-column-390-light")
        center(pg, "#rewards div.math-display")
        pg.locator("#rewards div.math-display mjx-container > svg").first.tap()
        pg.wait_for_timeout(300)
        t = tip(pg)
        check(t and not t["hidden"] and t["text"].startswith("What the symbols mean") and onscreen(t), "a tap on a formula shows its legend")
        shot(pg, "formula-legend-390-light")
        center(pg, "#weighting-test table")
        c = pg.locator("#weighting-test td[data-tt]").filter(has_text="99%").first
        c.scroll_into_view_if_needed()
        c.tap()
        pg.wait_for_timeout(300)
        t = tip(pg)
        check(t and not t["hidden"] and "…" not in t["text"] and onscreen(t) and t["sh"] <= t["ch"] + 1,
              f"a long weighting-test cell at phone width: whole row, fits ({t and round(t['h'])} px tall)")
        shot(pg, "tuple-wtest-390-light")
        center(pg, "details.ctrl-guide")
        shot(pg, "guide-closed-390-light")
        center(pg, "#task-diagram .symkey", "start")
        shot_el(pg, "diagram-legend-390-light", "#task-diagram")
        center(pg, "#dynamics .symkey")
        shot_el(pg, "legend-dynamics-390-light", "#dynamics .symkey")
        center(pg, "p.worked")
        shot(pg, "worked-390-light")
        ov = pg.evaluate(OVERFLOW_JS)
        check(ov["overflow"] <= 0, f"no horizontal overflow at the end ({ov['overflow']} px)")
        ctx.close()
        b.close()

    print(f"== console errors and warnings, page errors: {len(errors)}")
    for e in errors:
        print("   " + e[:300])
    check(not errors, "zero console errors")
    print(f"== {len(FAIL)} failures")
    return 1 if FAIL else 0


if __name__ == "__main__":
    sys.exit(main())

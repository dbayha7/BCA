#!/usr/bin/env python3
"""Checks and screenshots for the tooltip edit (edit 1 of 2): table coverage, hover, focus, tap, Esc, the switch,
storage, the walkthrough pop-out, overflow and console errors, at 1200 px dark and 390 px light (touch).

  cd /mnt/c/Users/David/dev/bca-shift-work/page && /opt/shot-venv/bin/python scripts/e5_test.py [--page PATH]
"""
import argparse
import json
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


COVERAGE_JS = r"""
() => [...document.querySelectorAll('table')].map((t, i) => {
  const sec = t.closest('section[id]');
  const heads = [...t.querySelectorAll('[data-tt="h"]')];
  const cells = [...t.querySelectorAll('[data-tt="c"]')];
  const missing = heads.filter(h => !h.hasAttribute('data-tt-desc') && h.closest('thead, tbody') &&
     (h.closest('thead') ? h.closest('thead').rows[h.closest('thead').rows.length - 1].contains(h) : true))
     .map(h => h.textContent.trim().slice(0, 50));
  return { i, sec: sec ? sec.id : '', heads: heads.length, described: heads.filter(h => h.hasAttribute('data-tt-desc')).length,
           cells: cells.length, missing, stop: t.querySelectorAll('[data-tt][tabindex="0"]').length };
})
"""

TIP_JS = r"""
() => { const t = document.getElementById('tt'); if (!t) return null;
  const r = t.getBoundingClientRect();
  return { hidden: t.hidden, on: t.classList.contains('on'), text: t.textContent, bold: [...t.querySelectorAll('b')].map(b => b.textContent),
           parent: t.parentElement ? (t.parentElement.id || t.parentElement.tagName) : null,
           l: r.left, t: r.top, r: r.right, b: r.bottom, vw: document.documentElement.clientWidth, vh: innerHeight }; }
"""


REVEAL = "el => { const d = el.closest('details'); if (d) d.open = true; el.scrollIntoView({block: 'center'}); }"


def tip(page):
    return page.evaluate(TIP_JS)


def onscreen(t):
    return t and t["l"] >= 0 and t["t"] >= 0 and t["r"] <= t["vw"] + 0.5 and t["b"] <= t["vh"] + 0.5


def nth_table(page, sel, n=0):
    return page.locator(sel).nth(n)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--page", default=os.path.join(PAGE_DIR, "thrusterchain.html"))
    ap.add_argument("--prefix", default="e5")
    a = ap.parse_args()
    frag = open(a.page, encoding="utf-8").read()
    d = tempfile.mkdtemp(prefix="e5-")
    f = os.path.join(d, "page.html")
    open(f, "w", encoding="utf-8").write(wrap(frag, "light"))
    url = "file://" + f
    P = a.prefix

    with sync_playwright() as p:
        b = p.chromium.launch()
        errors = []

        def new(width, scheme, touch=False):
            ctx = b.new_context(viewport={"width": width, "height": 844 if width < 700 else 800}, color_scheme=scheme,
                                has_touch=touch)
            pg = ctx.new_page()
            pg.on("console", lambda m: errors.append(f"[{width}] {m.type}: {m.text}") if m.type in ("error", "warning") else None)
            pg.on("pageerror", lambda e: errors.append(f"[{width}] pageerror: {e}"))
            pg.goto(url, wait_until="load")
            pg.wait_for_timeout(900)
            return ctx, pg

        # ---------------- 1200 px, dark, mouse ----------------
        print("== 1200 px dark (mouse and keyboard)")
        ctx, pg = new(1200, "dark")
        ov = pg.evaluate(OVERFLOW_JS)
        check(ov["overflow"] <= 0, f"no horizontal overflow at load ({ov['overflow']} px)")
        cov = pg.evaluate(COVERAGE_JS)
        json.dump(cov, open(os.path.join(PAGE_DIR, "tmp", "e5", "coverage.json"), "w"), indent=1)
        n_heads = sum(c["heads"] for c in cov)
        n_desc = sum(c["described"] for c in cov)
        n_cells = sum(c["cells"] for c in cov)
        miss = [(c["i"], c["sec"], c["missing"]) for c in cov if c["missing"]]
        check(len(cov) == 51, f"{len(cov)} tables found")  # 51 since edit 4 (48 after edit 3; +1 in Results, +2 in the weighting test)
        check(all(c["cells"] > 0 for c in cov), f"every table has tooltip cells ({n_cells} cells in all)")
        check(not miss, f"every column header has a description ({n_desc} of {n_heads} header cells described; missing: {miss})")
        check(all(c["stop"] == 1 for c in cov), "every table is exactly one Tab stop")
        sw = pg.evaluate("document.getElementById('tt-switch').getAttribute('aria-checked')")
        check(sw == "true", "switch on by default")
        check(pg.evaluate("getComputedStyle(document.querySelector('.wt-bar')).display") != "none", "top bar visible")

        # hover a column header of the task's chart table
        th = pg.locator("#task table").first.locator("thead th").nth(3)
        pg.evaluate(REVEAL, th.element_handle())
        pg.wait_for_timeout(200)
        th.hover()
        pg.wait_for_timeout(500)
        t = tip(pg)
        check(t and not t["hidden"] and "80% limit" in t["text"], f"header hover shows its description: {t and t['text'][:90]!r}")
        check(onscreen(t), "header tooltip on screen")
        pg.screenshot(path=os.path.join(SHOTS, f"{P}-header-1200-dark.png"))
        # hover a data cell: tuple with that cell's value in bold
        td = pg.locator("#task table").first.locator("tbody tr").nth(8).locator("td").nth(4)
        val = td.inner_text().strip()
        td.hover()
        pg.wait_for_timeout(400)
        t = tip(pg)
        check(t and not t["hidden"] and t["text"].startswith("(") and val in t["bold"],
              f"cell hover shows the row tuple with {val!r} in bold: {t and t['text'][:140]!r}")
        check(t and "Water's push c + w₁: " in t["text"] and "Rising-thrust: thruster 1 throttle:" in t["text"], "tuple names every column")
        pg.screenshot(path=os.path.join(SHOTS, f"{P}-cell-1200-dark.png"))
        pg.mouse.move(5, 5)
        pg.wait_for_timeout(400)
        check(tip(pg)["hidden"], "tooltip hides when the pointer leaves")

        # a wide, sideways-scrolling table and the shift table (group rows in the tuple)
        cell = pg.locator("#shift-card ~ .tblwrap table, #shift table.card").first.locator("tbody tr").nth(8).locator("td").nth(2)
        pg.evaluate(REVEAL, cell.element_handle())
        cell.hover()
        pg.wait_for_timeout(400)
        t = tip(pg)
        check(t and "Situation: Strong current" in t["text"] and "84.0%" in t["bold"],
              f"shift table tuple carries the situation heading: {t and t['text'][:120]!r}")
        check(onscreen(t), "shift-table tooltip on screen")
        pg.screenshot(path=os.path.join(SHOTS, f"{P}-cell-shift-1200-dark.png"))

        # keyboard: focus moves with the arrows, the tooltip follows, Esc closes it
        pg.mouse.move(5, 5)
        pg.wait_for_timeout(300)
        first = pg.locator("#action-space table").first.locator("thead th").first
        pg.evaluate(REVEAL, first.element_handle())
        first.focus()
        pg.keyboard.press("ArrowRight")
        pg.wait_for_timeout(250)
        foc = pg.evaluate("document.activeElement.textContent.trim().slice(0, 30)")
        t = tip(pg)
        check(foc.startswith("Action") and t and not t["hidden"] and "Which thruster" in t["text"],
              f"ArrowRight moves to the next header and shows its tooltip ({foc!r})")
        pg.keyboard.press("ArrowDown")
        pg.wait_for_timeout(250)
        t = tip(pg)
        check(t and not t["hidden"] and "Throttle of thruster 1" in t["bold"], f"ArrowDown reaches a cell: {t and t['text'][:80]!r}")
        pg.screenshot(path=os.path.join(SHOTS, f"{P}-keyboard-1200-dark.png"))
        pg.keyboard.press("Escape")
        pg.wait_for_timeout(150)
        check(tip(pg)["hidden"], "Esc hides the tooltip")
        # Tab leaves the table in one press (one Tab stop per table)
        pg.keyboard.press("Tab")
        pg.wait_for_timeout(100)
        out = pg.evaluate("!document.activeElement.closest('#action-space table')")
        check(out, "Tab leaves the table in one press")

        # elements given data-tip (what edit 2 adds): hover, keyboard focus, description, a canvas target
        pg.evaluate("""() => {
          const b = document.getElementById('opt-ctrl-no_throttle');
          b.removeAttribute('data-tip-ref');   // edit 2 gives this option a formatted tooltip; test plain text here
          b.setAttribute('data-tip-title', 'No throttle'); b.setAttribute('data-tip', 'Test text: all four throttles stay at 0.');
          const p = document.querySelector('#task > p');
          const s = document.createElement('span'); s.textContent = 'a term'; s.setAttribute('data-tip', 'Test text for a term.');
          s.id = 'tt-test-term'; p.prepend(s, ' ');
          document.getElementById('channel').setAttribute('data-tip', 'Test text for the channel drawing.');
        }""")
        pg.wait_for_timeout(150)
        prep = pg.evaluate("""() => { const s = document.getElementById('tt-test-term'), b = document.getElementById('opt-ctrl-no_throttle');
          const d = id => { const x = document.getElementById(id); return x ? x.textContent : null; };
          return { tab: s.getAttribute('tabindex'), sdesc: d((s.getAttribute('aria-describedby') || '').split(' ').pop()),
                   btab: b.getAttribute('tabindex'), bdesc: d((b.getAttribute('aria-describedby') || '').split(' ').pop()) }; }""")
        check(prep["tab"] == "0" and prep["btab"] is None, f"a plain element with data-tip becomes focusable, a button stays as it is ({prep['tab']}, {prep['btab']})")
        check(prep["sdesc"] == "Test text for a term." and prep["bdesc"] == "No throttle. Test text: all four throttles stay at 0.",
              f"data-tip text is the element's accessible description ({prep['bdesc']!r})")
        pg.evaluate("document.getElementById('sim').scrollIntoView({block: 'start'})")
        pg.wait_for_timeout(200)
        pg.hover("#opt-ctrl-no_throttle")
        pg.wait_for_timeout(450)
        t = tip(pg)
        check(t and not t["hidden"] and t["text"].startswith("No throttle") and onscreen(t), f"hover on a control shows its data-tip: {t and t['text'][:60]!r}")
        pg.screenshot(path=os.path.join(SHOTS, f"{P}-control-1200-dark.png"))
        cbox = pg.locator("#channel").bounding_box()
        pg.mouse.move(cbox["x"] + cbox["width"] * 0.7, cbox["y"] + cbox["height"] * 0.6)
        pg.wait_for_timeout(450)
        t = tip(pg)
        # edit 2 makes the drawing answer part by part, replacing the test text with the part under the pointer
        check(t and not t["hidden"] and ("channel drawing" in t["text"] or t["text"].startswith(("Float", "Thruster", "The water", "The channel")))
              and onscreen(t), f"hover on the canvas shows its tooltip near the pointer: {t and t['text'][:40]!r}")
        pg.mouse.click(cbox["x"] + cbox["width"] * 0.7, cbox["y"] + cbox["height"] * 0.6)
        pg.click("#fwd")
        check(pg.evaluate("document.getElementById('tLabel').textContent") != "", "the panel's controls still work with a tooltip open")
        pg.evaluate("el => el.scrollIntoView({block: 'center'})", pg.locator("#tt-test-term").element_handle())
        pg.focus("#tt-test-term")
        pg.keyboard.press("Shift+Tab")
        pg.keyboard.press("Tab")
        pg.wait_for_timeout(250)
        t = tip(pg)
        check(t and not t["hidden"] and "Test text for a term." in t["text"], "keyboard focus on a term shows its tooltip")
        pg.keyboard.press("Escape")
        pg.evaluate("""() => { document.getElementById('tt-test-term').remove();
          const b = document.getElementById('opt-ctrl-no_throttle'); b.removeAttribute('data-tip'); b.removeAttribute('data-tip-title');
          document.getElementById('channel').removeAttribute('data-tip'); }""")
        pg.mouse.move(5, 5)

        # the switch: off hides everything, is remembered, and comes back on
        pg.evaluate("window.scrollTo(0, 0)")
        pg.click("#tt-switch")
        pg.wait_for_timeout(200)
        check(pg.evaluate("document.getElementById('tt-switch').getAttribute('aria-checked')") == "false", "switch turns off")
        stored = pg.evaluate("(() => { try { return localStorage.getItem('thrusterchain-tooltips'); } catch (e) { return 'err:' + e; } })()")
        print(f"       stored choice: {stored!r}")
        pg.screenshot(path=os.path.join(SHOTS, f"{P}-switch-off-1200-dark.png"), clip={"x": 0, "y": 0, "width": 1200, "height": 420})
        td2 = pg.locator("#task table").first.locator("tbody td").nth(10)
        pg.evaluate(REVEAL, td2.element_handle())
        td2.hover()
        pg.wait_for_timeout(500)
        check(tip(pg)["hidden"], "switch off: hovering a cell shows nothing")
        sr = pg.evaluate("document.querySelectorAll('th .tt-sr').length")
        check(sr > 0, f"switch off: column descriptions stay as visually hidden text ({sr} headers)")
        stops = pg.evaluate("document.querySelectorAll('[data-tt][tabindex=\"0\"]').length")
        check(stops == 0, f"switch off: tables leave the Tab order ({stops} stops)")
        pg.reload(wait_until="load")
        pg.wait_for_timeout(800)
        if stored == "off":
            check(pg.evaluate("document.getElementById('tt-switch').getAttribute('aria-checked')") == "false",
                  "the choice survives a reload (stored in this browser)")
        else:
            print("       (storage unavailable on file:// in this browser; the page fell back to on)")
        if pg.evaluate("document.getElementById('tt-switch').getAttribute('aria-checked')") == "false":
            pg.click("#tt-switch")
        check(pg.evaluate("document.getElementById('tt-switch').getAttribute('aria-checked')") == "true", "switch back on")
        pg.evaluate("window.scrollTo(0, 0)")
        pg.wait_for_timeout(200)
        pg.screenshot(path=os.path.join(SHOTS, f"{P}-switch-on-1200-dark.png"), clip={"x": 0, "y": 0, "width": 1200, "height": 420})

        # the walkthrough pop-out: the tooltip sits inside the dialog; Esc closes the tooltip before the dialog
        pg.click("#wt-open")
        pg.wait_for_timeout(500)
        for _ in range(4):
            if pg.evaluate("document.getElementById('wt-title').textContent.trim()") == "The task":
                break
            pg.click("#wt-next")
            pg.wait_for_timeout(450)
        check(pg.evaluate("document.getElementById('wt-title').textContent.trim()") == "The task", "walkthrough at The task")
        cell = pg.locator("#wt-stage #task table").first.locator("tbody tr").nth(5).locator("td").nth(2)
        pg.evaluate(REVEAL, cell.element_handle())
        pg.wait_for_timeout(200)
        cell.hover()
        pg.wait_for_timeout(450)
        t = tip(pg)
        check(t and not t["hidden"] and t["parent"] == "wt" and onscreen(t), f"walkthrough: tooltip shows inside the dialog ({t and t['parent']})")
        pg.screenshot(path=os.path.join(SHOTS, f"{P}-walkthrough-1200-dark.png"))
        pg.keyboard.press("Escape")
        pg.wait_for_timeout(250)
        check(tip(pg)["hidden"] and pg.evaluate("document.getElementById('wt').open"), "walkthrough: first Esc closes only the tooltip")
        pg.keyboard.press("Escape")
        pg.wait_for_timeout(300)
        check(not pg.evaluate("document.getElementById('wt').open"), "walkthrough: second Esc closes the pop-out")

        # the live panel keeps working: play, step, slider, options
        pg.evaluate("document.getElementById('sim').scrollIntoView({block: 'start'})")
        pg.click("#opt-ctrl-no_throttle")
        pg.click("#fwd")
        pg.click("#play")
        pg.wait_for_timeout(400)
        lbl1 = pg.evaluate("document.getElementById('tLabel').textContent")
        pg.wait_for_timeout(500)
        lbl2 = pg.evaluate("document.getElementById('tLabel').textContent")
        pg.click("#play")
        check(lbl1 != lbl2, f"live panel still plays ({lbl1!r} -> {lbl2!r})")
        pg.mouse.move(600, 400)
        pg.locator("#channel").hover()
        pg.wait_for_timeout(300)
        ov = pg.evaluate(OVERFLOW_JS)
        check(ov["overflow"] <= 0, f"no horizontal overflow after the checks ({ov['overflow']} px)")
        ctx.close()

        # ---------------- 390 px, light, touch ----------------
        print("== 390 px light (touch)")
        ctx, pg = new(390, "light", touch=True)
        ov = pg.evaluate(OVERFLOW_JS)
        check(ov["overflow"] <= 0, f"no horizontal overflow at load ({ov['overflow']} px)")
        cov = pg.evaluate(COVERAGE_JS)
        check(all(c["stop"] == 1 for c in cov), "every table is one Tab stop at phone width")
        vis_stop = pg.evaluate("[...document.querySelectorAll('table')].every(t => { const s = t.querySelector('[data-tt][tabindex=\"0\"]'); return s && (s.getClientRects().length || !t.getClientRects().length || s.closest('details:not([open])')); })")
        check(vis_stop, "each table's Tab stop is a visible cell (card layout hides the header row)")
        pg.evaluate("window.scrollTo(0, 0)")
        pg.screenshot(path=os.path.join(SHOTS, f"{P}-top-390-light.png"))
        # tap a card cell of the task's chart table
        td = pg.locator("#task table").first.locator("tbody tr").nth(9).locator("td").nth(5)
        pg.evaluate(REVEAL, td.element_handle())
        pg.wait_for_timeout(200)
        val = td.locator(".cv").inner_text().strip()
        td.tap()
        pg.wait_for_timeout(350)
        t = tip(pg)
        check(t and not t["hidden"] and onscreen(t) and t["bold"] and t["bold"][0] in val,
              f"tap on a card cell shows the tuple, value in bold: {t and t['bold']} / {val!r}")
        check(t and "Rising-thrust: past 80%:" in t["text"], "card cell tooltip carries the column description")
        pg.screenshot(path=os.path.join(SHOTS, f"{P}-cell-390-light.png"))
        pg.locator("#task > p").first.tap()
        pg.wait_for_timeout(250)
        check(tip(pg)["hidden"], "a tap elsewhere closes it")
        # tap a column header of a table that keeps its header row at phone width
        th = pg.locator("#methods table").nth(1).locator("thead th").nth(1)
        pg.evaluate(REVEAL, th.element_handle())
        pg.wait_for_timeout(200)
        th.tap()
        pg.wait_for_timeout(350)
        t = tip(pg)
        check(t and not t["hidden"] and onscreen(t) and "rates highest" in t["text"], f"tap on a header shows its description")
        pg.screenshot(path=os.path.join(SHOTS, f"{P}-header-390-light.png"))
        th.tap()
        pg.wait_for_timeout(250)
        check(tip(pg)["hidden"], "a second tap on the same header closes it")
        # a wide table cell near the right edge stays on screen
        cell = pg.locator("#dynamics table.mat").first.locator("tbody tr").nth(2).locator("td").nth(10)
        pg.evaluate("el => { el.closest('details') && (el.closest('details').open = true); el.scrollIntoView({block: 'center', inline: 'end'}); }", cell.element_handle())
        pg.wait_for_timeout(300)
        cell.tap()
        pg.wait_for_timeout(350)
        t = tip(pg)
        check(t and not t["hidden"] and onscreen(t), "a cell at the right edge of a sideways table: tooltip on screen")
        pg.screenshot(path=os.path.join(SHOTS, f"{P}-cell-matrix-390-light.png"))
        ov = pg.evaluate(OVERFLOW_JS)
        check(ov["overflow"] <= 0, f"no horizontal overflow with a tooltip open ({ov['overflow']} px)")
        # the switch off at phone width
        pg.evaluate("window.scrollTo(0, 0)")
        pg.wait_for_timeout(200)
        pg.locator("#tt-switch").tap()
        pg.wait_for_timeout(250)
        check(pg.evaluate("document.getElementById('tt-switch').getAttribute('aria-checked')") == "false", "switch off by tap")
        pg.screenshot(path=os.path.join(SHOTS, f"{P}-switch-off-390-light.png"))
        td.tap()
        pg.wait_for_timeout(300)
        check(tip(pg)["hidden"], "switch off: a tap on a cell shows nothing")
        pg.locator("#tt-switch").tap()
        pg.wait_for_timeout(200)
        # the walkthrough at phone width
        pg.locator("#wt-open").tap()
        pg.wait_for_timeout(500)
        for _ in range(4):
            if pg.evaluate("document.getElementById('wt-title').textContent.trim()") == "Action Space":
                break
            pg.locator("#wt-next").tap()
            pg.wait_for_timeout(450)
        cell = pg.locator("#wt-stage #action-space table").first.locator("tbody tr").nth(0).locator("td").nth(4)
        pg.evaluate(REVEAL, cell.element_handle())
        pg.wait_for_timeout(250)
        cell.tap()
        pg.wait_for_timeout(350)
        t = tip(pg)
        check(t and not t["hidden"] and t["parent"] == "wt" and onscreen(t), "walkthrough at phone width: tap shows the tooltip in the dialog")
        pg.screenshot(path=os.path.join(SHOTS, f"{P}-walkthrough-390-light.png"))
        pg.locator("#wt-close").tap()
        pg.wait_for_timeout(300)
        ov = pg.evaluate(OVERFLOW_JS)
        check(ov["overflow"] <= 0, f"no horizontal overflow at the end ({ov['overflow']} px)")
        ctx.close()

        # ---------------- reduced motion, no storage ----------------
        print("== reduced motion, storage blocked")
        ctx = b.new_context(viewport={"width": 1200, "height": 800}, reduced_motion="reduce")
        pg = ctx.new_page()
        pg.add_init_script("Object.defineProperty(window, 'localStorage', { get() { throw new Error('blocked'); } });")
        pg.on("pageerror", lambda e: errors.append(f"[blocked storage] pageerror: {e}"))
        pg.goto(url, wait_until="load")
        pg.wait_for_timeout(700)
        check(pg.evaluate("document.getElementById('tt-switch').getAttribute('aria-checked')") == "true", "no storage: on by default")
        pg.click("#tt-switch")
        check(pg.evaluate("document.getElementById('tt-switch').getAttribute('aria-checked')") == "false", "no storage: the switch still works")
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

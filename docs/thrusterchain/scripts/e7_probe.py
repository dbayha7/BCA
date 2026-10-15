"""Probe: Tab stops with tooltips on/off, and how Chromium's touch emulation reports a long press on a button."""
import os, sys, tempfile
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from shot import wrap
from playwright.sync_api import sync_playwright
PAGE = sys.argv[1] if len(sys.argv) > 1 else "/mnt/c/Users/David/dev/bca-shift-work/page/thrusterchain.html"
d = tempfile.mkdtemp(); f = os.path.join(d, "p.html")
open(f, "w", encoding="utf-8").write(wrap(open(PAGE, encoding="utf-8").read(), "light"))
STOPS = r"""() => { const F = 'a[href], button:not([disabled]), input:not([disabled]), select, textarea, summary, [tabindex]';
  const all = [...document.querySelectorAll(F)].filter(e => e.tabIndex >= 0 && e.getAttribute('tabindex') !== '-1' && e.getClientRects().length && !e.closest('[hidden], dialog:not([open])'));
  const sw = document.getElementById('tt-switch'); const opt = document.querySelector('#sim .opt');
  const i0 = all.indexOf(sw), i1 = all.indexOf(opt);
  const kinds = {}; all.forEach(e => { const k = e.matches('.sym') ? 'sym' : e.matches('[data-tt]') ? 'table' : e.matches('[data-tip]:not(button):not(a):not(input)') ? 'tip-other' : 'native'; kinds[k] = (kinds[k] || 0) + 1; });
  return { total: all.length, switchToPanel: i1 - i0, kinds }; }"""
with sync_playwright() as p:
    b = p.chromium.launch()
    ctx = b.new_context(viewport={"width": 1200, "height": 800})
    pg = ctx.new_page(); pg.goto("file://" + f); pg.wait_for_timeout(800)
    print("on ", pg.evaluate(STOPS))
    pg.click("#tt-switch"); pg.wait_for_timeout(100)
    print("off", pg.evaluate(STOPS))
    pg.click("#tt-switch")
    ctx.close()
    ctx = b.new_context(viewport={"width": 390, "height": 844}, has_touch=True, is_mobile=True)
    pg = ctx.new_page(); pg.goto("file://" + f); pg.wait_for_timeout(800)
    pg.evaluate("""() => { window.EV = []; for (const t of ['pointerdown','pointerup','pointercancel','click','contextmenu','touchstart','touchend']) document.addEventListener(t, e => EV.push(t + ':' + (e.pointerType||'') + '@' + Math.round(performance.now())), true); }""")
    el = pg.locator("#fwd"); el.scroll_into_view_if_needed(); bb = el.bounding_box()
    cdp = ctx.new_cdp_session(pg)
    x, y = bb["x"] + bb["width"] / 2, bb["y"] + bb["height"] / 2
    cdp.send("Input.dispatchTouchEvent", {"type": "touchStart", "touchPoints": [{"x": x, "y": y}]})
    pg.wait_for_timeout(900)
    cdp.send("Input.dispatchTouchEvent", {"type": "touchEnd", "touchPoints": []})
    pg.wait_for_timeout(500)
    print("long press:", pg.evaluate("EV"), "state", pg.evaluate("document.getElementById('tLabel').textContent"))
    pg.evaluate("EV.length = 0")
    cdp.send("Input.dispatchTouchEvent", {"type": "touchStart", "touchPoints": [{"x": x, "y": y}]})
    pg.wait_for_timeout(60)
    cdp.send("Input.dispatchTouchEvent", {"type": "touchEnd", "touchPoints": []})
    pg.wait_for_timeout(500)
    print("tap:", pg.evaluate("EV"), "state", pg.evaluate("document.getElementById('tLabel').textContent"))
    b.close()

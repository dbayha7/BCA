#!/usr/bin/env python3
"""Final browser checks of the critic-in-action section in qviewer/test_page.html (headless Chromium, wrapped the way
the page's shot.py wraps it). Read only; prints a report; exit status 1 on a failed check.
  A. every number of the 41-row table and the readout "true value lost by taking a-hat", for all 20 episodes x 100
     states, against the 7-digit file (precise/q_data.json) formatted the page's way;
  B. the page's own table tooltips on the four tables: prepared, one Tab stop each, a cell's whole row, header text,
     the arrow keys, and values that follow a change of state;
  C. Esc inside the walkthrough pop-out closes a chart readout first and only it; a second Esc closes the pop-out;
  D. at 390 px (mouse and touch) a chart readout never covers the plot;
  E. chart labels against the curves and the actor's line (counts over 240 drawings at each width);
  F. the channel strip's throttles have one decimal, like the readouts.
Run (WSL): taskset -c 2-3 nice -n 10 /opt/shot-venv/bin/python final_check.py
"""
import json
import os
import sys
import tempfile
from decimal import Decimal, ROUND_HALF_UP

from playwright.sync_api import sync_playwright

sys.path.insert(0, "/mnt/c/Users/David/dev/bca-shift-work/page/scripts")
import shot  # noqa: E402

HERE = "/mnt/c/Users/David/dev/bca-shift-work/qviewer"
PAGE = os.environ.get("QV_PAGE") or os.path.join(HERE, "test_page.html")   # QV_PAGE=path checks another page (read only)
Q7 = json.load(open(os.path.join(HERE, "precise", "q_data.json")))
QV = json.load(open(os.path.join(HERE, "qv_data.json"), encoding="utf-8"))
F7 = {f: i for i, f in enumerate(Q7["episode_fields"])}
FV = {f: i for i, f in enumerate(QV["fields"])}
MINUS = "−"
OK = True


def bad(msg):
    global OK
    OK = False
    print("  FAIL:", msg)


# the page's number formats (script.js grp, fv, sg, fu), on the shortest decimal form of the double, half away from 0
def rnd(v, d):
    return Decimal(repr(abs(float(v)))).quantize(Decimal(1).scaleb(-d), rounding=ROUND_HALF_UP)


def grp(v, d):
    q = rnd(v, d)
    return (MINUS if v < 0 and q != 0 else "") + f"{q:,.{d}f}"


def fv(v):
    return grp(v, 1 if abs(v) >= 100 else 2)


def sg(v, d):
    s = grp(v, d)
    return s if rnd(v, d) == 0 else ("+" + s if v > 0 else s)


def fu(u):
    return (str(int(u)) if float(u).is_integer() else f"{u:.1f}") + "%"


def wrapped():
    tmp = tempfile.mkdtemp(prefix="qvf-")
    p = os.path.join(tmp, "page.html")
    open(p, "w", encoding="utf-8").write(shot.wrap(open(PAGE, encoding="utf-8").read(), "light"))
    return p


JS_EP = r"""
([q, c, s]) => {
  const out = [];
  document.getElementById('qv-tables').open = true;
  for (let t = 0; t < 100; t++) {
    window.qvSet({ q, c, s, t });
    const rows = [...document.querySelector('#qv-tbl-sweep tbody').rows].map(r => [...r.cells].map(x => x.textContent));
    out.push({ rows, cost: document.getElementById('qv-ro-cost').textContent, now: document.getElementById('qv-now').textContent });
  }
  return out;
}
"""


def check_numbers(pg):
    print("A. displayed numbers against the 7-digit file")
    u = QV["u"]
    n_cells = n_bad = n_cost_bad = n_under = n_states = 0
    examples = []
    for e, e7 in zip(QV["eps"], Q7["episodes"]):
        key = e[:3]
        assert key == e7[:3]
        got = pg.evaluate(JS_EP, key)
        for t in range(100):
            g = got[t]
            n_states += 1
            q7, p7 = e7[F7["q_sweep"]][t], e7[F7["qpi_sweep"]][t]
            qa, pa = e7[F7["q_actor"]][t], e7[F7["qpi_actor"]][t]
            byfirst = {r[0]: r for r in g["rows"]}
            for k in range(41):
                want = [fu(u[k]), fv(q7[k]), fv(p7[k]), sg(q7[k] - p7[k], 2)]
                have = byfirst.get(fu(u[k]), [None] * 4)[:4]
                n_cells += 3
                for a, b in zip(want[1:], have[1:]):
                    if a != b:
                        n_bad += 1
                        if len(examples) < 5:
                            examples.append((key, t, k, want, have))
            act = [r for r in g["rows"] if "stream" in r[0]]
            want = [fv(qa), fv(pa), sg(qa - pa, 2)]
            n_cells += 3
            if len(act) != 1 or act[0][1:4] != want:
                n_bad += 1
                if len(examples) < 5:
                    examples.append((key, t, "actor", want, act))
            h, b = e[FV["a_hat"]][t], e[FV["a_best_true"]][t]
            if h == b:
                wc = grp(0, 3)
            else:
                wc = grp(p7[b] - p7[h], 3)
                wc = "under 0.001" if wc == "0.000" else wc
                n_under += wc == "under 0.001"
                if ("would lose " + wc + " in true value") not in g["now"]:
                    n_cost_bad += 1
            if g["cost"] != wc:
                n_cost_bad += 1
                if len(examples) < 8:
                    examples.append((key, t, "cost", wc, g["cost"]))
    print(f"  {n_states} states; table cells compared {n_cells:,}, differing {n_bad}; 'true value lost' readouts and "
          f"sentences differing {n_cost_bad}; states shown as 'under 0.001': {n_under}")
    if examples:
        print("  examples:", examples)
    if n_bad or n_cost_bad:
        bad("displayed numbers differ from the 7-digit file")


JS_TBL = r"""
() => [...document.querySelectorAll('#critic-in-action table')].map(t => ({ id: t.id,
  cells: t.querySelectorAll('td').length, tt: t.querySelectorAll('td[data-tt="c"]').length,
  heads: t.querySelectorAll('th[data-tt="h"]').length, stops: t.querySelectorAll('[data-tt][tabindex="0"]').length }))
"""

TT = """(() => { const t = document.getElementById('tt'); return (!t || t.hidden) ? null : t.innerText; })()"""


def hover(pg, sel):
    pg.mouse.move(2, 2)
    pg.wait_for_timeout(250)
    loc = pg.locator(sel).first
    loc.scroll_into_view_if_needed()
    pg.wait_for_timeout(120)
    loc.hover()
    pg.wait_for_timeout(450)
    t = pg.evaluate(TT)
    return " ".join(t.split()) if t else t       # the tooltip puts each name and its value on lines of their own


def check_tables(pg, tag):
    print(f"B. the page's table tooltips [{tag}]")
    pg.evaluate("window.qvSet({q:'expert', c:'gap', s:'strong', t:40}); document.getElementById('qv-tables').open = true;"
                " document.getElementById('qv-qual').open = true")
    pg.wait_for_timeout(300)
    info = pg.evaluate(JS_TBL)
    for x in info:
        print(f"  {x['id']}: cells {x['cells']}, with the page's cell tooltip {x['tt']}, headers prepared {x['heads']}, Tab stops {x['stops']}")
        if x["tt"] != x["cells"] or x["stops"] != 1 or x["cells"] == 0:
            bad(f"{x['id']} not prepared by the page's table tooltips")
    probes = [("#qv-tbl-sweep tbody tr:nth-child(36) td:nth-child(2)", ["Thruster 1, upstream:", "Critic, Q(s, a):", "True value:", "Critic minus true value:"]),
              ("#qv-tbl-ep tbody tr:nth-child(42) td:nth-child(5)", ["State: 41", "Critic, Q(s, π(s)):", "True value:"]),
              ("#qv-tbl-qual tbody tr:nth-child(5) td:nth-child(4)", ["Dataset quality: expert", "Thruster 1 held at exactly 80%"]),
              ("#qv-tbl-ctx tbody tr:nth-child(5) td:nth-child(3)", ["Dataset quality: expert", "Critic rates 100% above 80%, 80% hold removed: 66.1%"])]
    if tag.startswith("390"):
        probes = probes[2:]      # at phone width the two long tables scroll inside their boxes; the card tables are checked
    for sel, need in probes:
        cell = pg.evaluate(f"document.querySelector('{sel}').textContent")
        tip = hover(pg, sel)
        ok = bool(tip) and all(n in tip for n in need) and cell in tip
        print(f"  hover {sel.split(' ')[0]} cell '{cell}': {'whole row shown' if ok else 'MISSING'} :: {(tip or '').replace(chr(10), ' | ')[:230]}")
        if not ok:
            bad(f"cell tooltip {sel}")
    if not tag.startswith("390"):
        for sel, need in (("#qv-tbl-sweep thead th:nth-child(3) span", "computed exactly by simulation"),
                          ("#qv-tbl-ep thead th:nth-child(2) span", "the current c plus the push"),
                          ("#qv-tbl-qual thead th:nth-child(5) span", "Only random commands go there"),
                          ("#qv-tbl-ctx thead tr.grp th:nth-child(3) span", "while the true value of 100%")):
            tip = hover(pg, sel)
            ok = bool(tip) and need in tip
            print(f"  hover header {sel.split(' ')[0]}: {'description shown' if ok else 'MISSING'} :: {(tip or '').replace(chr(10), ' | ')[:160]}")
            if not ok:
                bad(f"header tooltip {sel}")
        # values follow a change of state, and the arrow keys move between cells
        sel = "#qv-tbl-ep tbody tr:nth-child(42) td:nth-child(5)"
        pg.evaluate("window.qvSet({q:'random', c:'full', s:'training', t:41})")
        cell = pg.evaluate(f"document.querySelector('{sel}').textContent")
        tip = hover(pg, sel)
        print(f"  after changing the episode, the same cell reads '{cell}' and its tooltip has it: {bool(tip) and cell in tip}")
        if not (tip and cell in tip):
            bad("cell tooltip does not follow the new values")
        pg.mouse.move(2, 2)
        pg.keyboard.press("Shift+Tab")
        pg.evaluate("document.querySelector('#qv-tbl-ctx [data-tt][tabindex=\"0\"]').focus()")
        a = pg.evaluate("document.activeElement.textContent")
        pg.keyboard.press("ArrowDown")
        pg.keyboard.press("ArrowRight")
        b2 = pg.evaluate("[document.activeElement.tagName, document.activeElement.textContent, document.activeElement.closest('table').id]")
        tip = pg.evaluate(TT)
        print(f"  keyboard in the context table: from '{a}', ArrowDown then ArrowRight -> {b2}; tooltip {'shown' if tip else 'none'}")
        if b2[2] != "qv-tbl-ctx" or b2[1] == a:
            bad("arrow keys do not move between the context table's cells")


def check_escape(browser):
    print("C. Esc inside the walkthrough pop-out")
    for w in (390, 1200):
        pg = browser.new_page(viewport={"width": w, "height": 844 if w < 700 else 800})
        pg.goto("file://" + wrapped())
        pg.wait_for_timeout(900)
        for svg, tip in (("#qv-svg2", "#qv-tip2"), ("#qv-svg1", "#qv-tip1")):
            if not pg.evaluate("document.getElementById('wt').open"):
                pg.click("#wt-open")
                pg.wait_for_timeout(300)
                for _ in range(40):
                    if pg.text_content("#wt-title").strip() == "The critic in action":
                        break
                    pg.click("#wt-next")
                    pg.wait_for_timeout(220)
            pg.wait_for_timeout(300)
            pg.mouse.move(w - 3, 3)
            pg.evaluate(f"document.querySelector('{svg}').scrollIntoView({{block:'center'}})")
            pg.focus(svg)
            pg.keyboard.press("ArrowRight")
            pg.wait_for_timeout(200)
            s0 = [pg.evaluate(f"!document.querySelector('{tip}').hidden"), pg.evaluate("document.getElementById('wt').open")]
            pg.keyboard.press("Escape")
            pg.wait_for_timeout(250)
            s1 = [pg.evaluate(f"!document.querySelector('{tip}').hidden"), pg.evaluate("document.getElementById('wt').open")]
            pg.keyboard.press("Escape")
            pg.wait_for_timeout(250)
            s2 = pg.evaluate("document.getElementById('wt').open")
            print(f"  {w} {svg}: readout shown, pop-out open {s0} -> one Esc {s1} -> second Esc, pop-out open: {s2}")
            if s0 != [True, True] or s1 != [False, True] or s2:
                bad(f"Esc order at {w} on {svg}")
        pg.close()


def overlap(a, b):
    w = min(a["r"], b["r"]) - max(a["l"], b["l"])
    h = min(a["b"], b["b"]) - max(a["t"], b["t"])
    return max(0, w) * max(0, h)


RECTS = """([tip, svg]) => { const r = e => { const x = document.querySelector(e).getBoundingClientRect(); return {l: x.left, r: x.right, t: x.top, b: x.bottom}; };
  return document.querySelector(tip).hidden ? null : [r(tip), r(svg)]; }"""


def check_narrow(browser):
    print("D. 390 px: chart readouts against the plot")
    worst = 0
    for touch in (False, True):
        ctx = browser.new_context(viewport={"width": 390, "height": 844}, has_touch=touch, is_mobile=touch)
        pg = ctx.new_page()
        pg.goto("file://" + wrapped())
        pg.wait_for_timeout(900)
        pg.evaluate("window.qvSet({q:'expert', c:'gap', s:'strong', t:40})")
        for svg, tip, block in (("#qv-svg1", "#qv-tip1", "center"), ("#qv-svg1", "#qv-tip1", "end"),
                                ("#qv-svg2", "#qv-tip2", "center"), ("#qv-svg2", "#qv-tip2", "end")):
            pg.evaluate(f"document.querySelector('{svg}').scrollIntoView({{block:'{block}'}})")
            pg.wait_for_timeout(200)
            bx = pg.locator(svg).bounding_box()
            for fx_ in (0.2, 0.55, 0.92):
                x, y = bx["x"] + bx["width"] * fx_, bx["y"] + bx["height"] * 0.4
                if touch:
                    pg.touchscreen.tap(x, y)
                else:
                    pg.mouse.move(x, y)
                pg.wait_for_timeout(200)
                rr = pg.evaluate(RECTS, [tip, svg])
                if not rr:
                    bad(f"no readout on {svg} ({'touch' if touch else 'mouse'})")
                    continue
                ov = overlap(rr[0], rr[1])
                worst = max(worst, ov)
                inview = rr[0]["t"] >= 0 and rr[0]["b"] <= 844
                if ov > 0 or not inview:
                    print(f"  {svg} {'touch' if touch else 'mouse'} at {fx_:.2f}, chart {block}: overlap {ov:.0f} px², in the window {inview}")
                    bad("readout covers the plot or leaves the window")
            if touch:
                pg.touchscreen.tap(5, 5)
            else:
                pg.mouse.move(2, 2)
            pg.wait_for_timeout(150)
        ctx.close()
    print(f"  largest overlap of a readout with its chart: {worst:.0f} px²")


JS_LABELS = r"""
() => {
  const D = JSON.parse(document.getElementById('qv-data').textContent), res = { n: 0, curve: {}, actor: {}, lines: {}, plate: {}, ex: [] };
  const svg = document.getElementById('qv-svg1');
  for (const e of D.eps) for (const t of [0, 40, 99]) for (const range of ['hi', 'all']) for (const view of ['rel', 'abs']) {
    window.qvSet({ q: e[0], c: e[1], s: e[2], t, range, view });
    res.n++;
    const paths = [...svg.querySelectorAll('path.qv-lq, path.qv-lp')];
    const pts = paths.map(p => { const L = p.getTotalLength(), a = []; for (let l = 0; l <= L; l += 1.5) a.push(p.getPointAtLength(l)); return a; });
    const act = svg.querySelector('line.qv-act');
    const ax = act ? Number(act.getAttribute('x1')) : null;
    const vx = [...svg.querySelectorAll('line.qv-stall-l, line.qv-ball-l')].map(l => Number(l.getAttribute('x1')));
    for (const tx of svg.querySelectorAll('text.lab, text.qv-mlab')) {
      const name = tx.textContent;
      if (/^actor |^← actor/.test(name)) continue;
      const b = tx.getBBox(), x1 = b.x + 1, x2 = b.x + b.width - 1, y1 = b.y + 2, y2 = b.y + b.height - 2;
      const hit = pts.some(a => a.some(p => p.x > x1 && p.x < x2 && p.y > y1 && p.y < y2));
      const plate = tx.previousElementSibling && tx.previousElementSibling.classList.contains('qv-lbg');
      if (plate) { res.plate[name] = (res.plate[name] || 0) + 1; continue; }
      if (hit) { res.curve[name] = (res.curve[name] || 0) + 1; if (res.ex.length < 6) res.ex.push([e[0], e[1], e[2], t, range, view, name]); }
      if (ax != null && ax > x1 && ax < x2) res.actor[name] = (res.actor[name] || 0) + 1;
      if (vx.some(x => x > x1 && x < x2)) res.lines[name] = (res.lines[name] || 0) + 1;
    }
  }
  return res;
}
"""


def check_labels(pg, tag):
    r = pg.evaluate(JS_LABELS)
    print(f"E. labels [{tag}]: {r['n']} drawings; labels crossed by a curve {r['curve']}; labels crossed by the actor's line "
          f"{r['actor']}; labels crossed by the 80% line or a region edge {r['lines']}; labels set on a plate (no free spot) "
          f"{r['plate']}; examples {r['ex'][:4]}")
    if r["curve"] or r["actor"] or r["lines"]:
        bad("a line runs through a chart label")
    return r


def check_strip(pg, tag):
    pg.evaluate("window.qvSet({q:'expert', c:'gap', s:'strong', t:0})")
    vals = pg.evaluate("[...document.querySelectorAll('#qv-strip .qv-tval')].map(x => x.textContent)")
    ro = pg.text_content("#qv-ro-a1")
    ok = all(("%" not in v) or (v.count(".") == 1) for v in vals) and vals[0].startswith(ro.split("%")[0])
    print(f"F. strip [{tag}] at state 0: {vals}; readout '{ro}' -> {'one decimal, agrees' if ok else 'MISMATCH'}")
    if not ok:
        bad("strip throttles")


def main():
    with sync_playwright() as p:
        b = p.chromium.launch()
        for w, sc in ((1200, "dark"), (390, "light")):
            errs = []
            pg = b.new_context(viewport={"width": w, "height": 800 if w > 700 else 844}, color_scheme=sc).new_page()
            pg.on("console", lambda m: errs.append(f"{m.type}: {m.text}") if m.type in ("error", "warning") else None)
            pg.on("pageerror", lambda e: errs.append("pageerror: " + str(e)))
            pg.goto("file://" + wrapped(), wait_until="load")
            pg.wait_for_timeout(900)
            tag = f"{w} {sc}"
            if w == 1200:
                check_numbers(pg)
            check_tables(pg, tag)
            check_strip(pg, tag)
            check_labels(pg, tag)
            ov = pg.evaluate(shot.OVERFLOW_JS)
            print(f"  [{tag}] horizontal overflow {ov['overflow']} px; console errors/warnings {len(errs)} {errs[:3]}")
            if ov["overflow"] > 0 or errs:
                bad(f"overflow or console errors at {tag}")
            pg.context.close()
        check_escape(b)
        check_narrow(b)
        b.close()
    print("ALL FINAL CHECKS PASSED" if OK else "SOME FINAL CHECKS FAILED")
    return 0 if OK else 1


if __name__ == "__main__":
    sys.exit(main())

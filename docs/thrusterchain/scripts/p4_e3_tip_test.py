"""Read-only browser test of page pass 4, edit 3: the tooltips of Methods and the harm checks.

- every display formula in the two sections has a tooltip that points at its legend, and a grouped legend's tooltip
  holds every group heading;
- hovering a formula, a symbol, the new term and a table header shows the expected text;
- every cell of the new tables carries its column name (the phone cards), and the new step chips are labelled;
- no console errors, no horizontal overflow.

  cd /mnt/c/Users/David/dev/bca-shift-work/page && /opt/shot-venv/bin/python scripts/p4_e3_tip_test.py [--page P] [--width W] [--scheme S]
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
ap.add_argument("--page", default=DEFAULT_PAGE)
a = ap.parse_args()
fails, errors = [], []


def check(ok, msg):
    print(("ok   " if ok else "FAIL ") + msg)
    if not ok:
        fails.append(msg)


frag = open(a.page, encoding="utf-8").read()
tmp = os.path.join(tempfile.mkdtemp(prefix="e3tip-"), "page.html")
open(tmp, "w", encoding="utf-8").write(wrap(frag, "light"))
TT = "() => { const t = document.querySelector('.tt'); return t && !t.hidden && t.classList.contains('on') ? t.textContent : ''; }"
with sync_playwright() as p:
    br = p.chromium.launch()
    ctx = br.new_context(viewport={"width": a.width, "height": 900}, color_scheme=a.scheme)
    pg = ctx.new_page()
    pg.on("console", lambda m: errors.append(m.text) if m.type in ("error", "warning") else None)
    pg.on("pageerror", lambda e: errors.append(str(e)))
    pg.goto("file://" + tmp, wait_until="load")
    pg.wait_for_timeout(900)

    # 1. every display formula points at its legend
    info = pg.evaluate("""() => [...document.querySelectorAll('#methods div.math-display, #harm-checks div.math-display')].map(d => {
        const f = d.querySelector('mjx-container > svg'); const ref = f && f.getAttribute('data-tip-ref');
        const t = ref && document.getElementById(ref);
        const txt = t ? (t.content ? t.content.textContent : t.textContent) : '';
        return { label: d.querySelector('mjx-container').getAttribute('aria-label').slice(0, 40), ref, found: !!t, txt: txt.slice(0, 4000) };
    })""")
    check(len(info) == 11, f"display formulas in the two sections: {len(info)}")
    for i in info:
        check(i["found"], f"formula {i['label']!r} -> legend {i['ref']}")
    actor = [i for i in info if i["label"].startswith("\\begin{aligned} L_{")]
    check(actor and all(h in actor[0]["txt"] for h in ("What is lowered, and over what", "The commands", "The two forces and their weights")),
          "the actor's formula shows all three groups of its legend")
    seed = [i for i in info if i["label"].startswith("N_0")]
    check(seed and all(h in seed[0]["txt"] for h in ("The result", "What is measured", "Numbers in the formula")),
          "the seed-count formula shows all three groups of its legend")

    # 2. hover: a formula, a symbol, the new term, a reused term, a header of each new table
    def hover(sel, want, label):
        loc = pg.locator(sel).first
        loc.scroll_into_view_if_needed()
        loc.hover()
        pg.wait_for_timeout(350)
        tt = pg.evaluate(TT)
        check(all(w in tt for w in want), f"{label}: {tt[:120]!r}")
        pg.mouse.move(2, 2)
        pg.wait_for_timeout(150)

    hover('#methods div.math-display mjx-container[aria-label^="y = r"] > svg', ["one-step target"], "the target formula's tooltip")
    hover('#methods span.term[data-tip-title="Behaviour constraint"]', ["behaviour constraint", "Bayesian Conformal Aggregation"],
          "the new term")
    hover('#methods span.term[data-tip-title="Value-pessimism methods"]', ["value-pessimism method"], "a reused term")
    hover('#m-how-far ~ p.wex span.sym', ["critic"], "a symbol in a worked example")
    for sec, head, want in ([] if a.width < 700 else [("#methods table.p4-map", "Pipeline step", "offline reinforcement-learning pipeline"),
                            ("#methods table.p4-set", "Why it matters here", "ThrusterChain"),
                            ("#harm-checks table.p4-rules", "Pass mark", "error ratio"),
                            ("#harm-checks table.p4-stop", "Why", "rerun")]):
        hover(f'{sec} thead th:has-text("{head}")', [want], f"header {head!r} of {sec}")
    # a cell: label: value lines
    hover("#methods table.p4-set tbody tr:nth-child(2) td:nth-child(2)", ["Value"], "a settings cell")

    # 3. phone cards: every body cell of the new tables carries its column name
    lab = pg.evaluate("""() => ['#methods table.p4-map', '#methods table.p4-set', '#harm-checks table.p4-rules', '#harm-checks table.p4-stop']
        .map(s => { const t = document.querySelector(s); const cells = [...t.querySelectorAll('tbody tr:not(.sub) td')];
                    return [s, cells.length, cells.filter(c => !c.dataset.label).length]; })""")
    for s, n, miss in lab:
        check(n > 0 and miss == 0, f"{s}: {n} cells, {miss} without a column name")
    chips = pg.evaluate("""() => [...document.querySelectorAll('#methods .stepchip, #harm-checks .stepchip')].map(c => [c.dataset.step, c.textContent.trim()])""")
    names = {"collect": "Step 1", "remove": "Step 2", "train": "Step 3", "band": "Step 4", "test": "Step 5"}
    check(all(names.get(s) == t for s, t in chips), f"{len(chips)} step chips, each named for its step")
    ov = pg.evaluate(OVERFLOW_JS)
    check(ov["overflow"] <= 0, f"no horizontal overflow ({ov['overflow']} px)")
    br.close()
check(not errors, f"no console errors ({errors[:3]})")
print(f"{len(fails)} FAILS" if fails else "PASS")
sys.exit(1 if fails else 0)

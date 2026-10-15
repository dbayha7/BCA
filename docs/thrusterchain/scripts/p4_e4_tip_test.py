"""Read-only browser test of page pass 4, edit 4: Results, the weighting test's exploratory results and the critic
section's column descriptions.

- hovering the new terms, a reused term and a header of each new table shows the expected text;
- every column header of the critic section's four tables now has a description (keyboard focus shows it);
- every cell of the new tables carries its column name (the phone cards), and the step chips are labelled;
- situation colours: the ladder's rows, the scores table's and the Stage 1 tables' situation columns, the coverage
  table's group headings and the test-setting rows wear their situation's colour token;
- nothing says "running" any more, no warning-red tag is left in Results, no console errors, no horizontal overflow.

  cd /mnt/c/Users/David/dev/bca-shift-work/page && /opt/shot-venv/bin/python scripts/p4_e4_tip_test.py [--page P] [--width W] [--scheme S]
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
tmp = os.path.join(tempfile.mkdtemp(prefix="e4tip-"), "page.html")
open(tmp, "w", encoding="utf-8").write(wrap(frag, a.scheme))
TT = "() => { const t = document.querySelector('.tt'); return t && !t.hidden && t.classList.contains('on') ? t.textContent : ''; }"
with sync_playwright() as p:
    br = p.chromium.launch()
    ctx = br.new_context(viewport={"width": a.width, "height": 900}, color_scheme=a.scheme)
    pg = ctx.new_page()
    pg.on("console", lambda m: errors.append(m.text) if m.type in ("error", "warning") else None)
    pg.on("pageerror", lambda e: errors.append(str(e)))
    pg.goto("file://" + tmp, wait_until="load")
    pg.wait_for_timeout(900)

    def hover(sel, want, label):
        loc = pg.locator(sel).first
        loc.scroll_into_view_if_needed()
        loc.hover()
        pg.wait_for_timeout(350)
        tt = pg.evaluate(TT)
        sq = lambda x: "".join(x.split())     # the tooltip sets each name and its value as separate lines
        check(all(sq(w) in sq(tt) for w in want), f"{label}: {tt[:120]!r}")
        pg.mouse.move(2, 2)
        pg.wait_for_timeout(150)

    # 1. hover: the three new terms, reused terms, headers of each new table, a cell
    hover('#results span.term[data-tip-title="Dynamics extrapolation"]', ["dynamics extrapolation", "1.1"], "new term: dynamics extrapolation")
    hover('#results span.term[data-tip-title="Disturbance shift"]', ["disturbance shift", "0.7"], "new term: disturbance shift")
    hover('#results span.term[data-tip-title="Initial-state shift"]', ["initial-state shift", "three times"], "new term: initial-state shift")
    hover('#results span.term[data-tip-title="Behaviour constraint"]', ["behaviour constraint"], "reused term in Results")
    hover('#results span.term[data-tip-title="Points"]', ["score scale"], "the points term in Results")
    hover('#iw-arms span.term[data-tip-title="Capped pull"]', ["capped pull"], "reused term in the exploratory results")
    if a.width >= 700:
        for sec, head, want in (("#results table.p4-ladder", "Harm from the missing commands", "return-cost check"),
                                ("#results table.p4-world", "Passed", "every seed"),
                                ("#weighting-test table.p4-iwcov", "No shift", "held-out steps"),
                                ("#weighting-test table.p4-iwret", "Perfect lower edge", "perfect penalty")):
            hover(f'{sec} thead th:has-text("{head}")', [want], f"header {head!r} of {sec}")
        hover("#results table.p4-ladder tbody tr:nth-child(4) td:nth-child(6)", ["Level and situation", "98.9"], "a ladder cell")
        hover("#weighting-test table.p4-iwret tbody tr:nth-child(5) td:nth-child(8)",
              ["Quality: expert", "Perfect lower edge, for scale", "99.1"], "an exploratory-return cell")

    # 2. the critic section's four tables: every column header has a description now
    qv = pg.evaluate("""() => ['#qv-tbl-sweep', '#qv-tbl-ep', '#qv-tbl-qual', '#qv-tbl-ctx'].map(s => {
        const t = document.querySelector(s), last = t.tHead.rows[t.tHead.rows.length - 1];
        const hs = [...last.cells]; const grp = t.tHead.rows.length > 1 ? [...t.tHead.rows[0].cells].filter(c => c.textContent.trim()) : [];
        return [s, hs.length, hs.filter(h => !h.hasAttribute('data-tt-desc')).length, grp.length, grp.filter(h => !h.hasAttribute('data-tt-desc')).length]; })""")
    total = sum(x[1] for x in qv)
    check(total == 26 and all(x[2] == 0 for x in qv), f"critic section: {total} column headers, all described {qv}")
    check(all(x[4] == 0 for x in qv) and sum(x[3] for x in qv) == 3, "critic section: the 3 group headings are described")
    desc = pg.evaluate("() => { const th = document.querySelector('#qv-tbl-sweep thead th:nth-child(3)'); const sr = th.querySelector('.tt-sr'); return sr ? sr.textContent : ''; }")
    check("computed exactly by simulation" in desc and "600 steps" in desc, f"the sweep table's true-value description: {desc[:90]!r}")

    # 3. phone cards: every body cell of the new tables carries its column name; step chips
    lab = pg.evaluate("""() => ['#results table.p4-ladder', '#results table.p4-world', '#weighting-test table.p4-iwcov', '#weighting-test table.p4-iwret']
        .map(s => { const t = document.querySelector(s); const cells = [...t.querySelectorAll('tbody tr:not(.sub) td')];
                    return [s, cells.length, cells.filter(c => !c.dataset.label).length]; })""")
    for s, n, miss in lab:
        check(n > 0 and miss == 0, f"{s}: {n} cells, {miss} without a column name")
    chips = pg.evaluate("""() => [...document.querySelectorAll('#results .stepchip, #weighting-test .stepchip')].map(c => [c.dataset.step, c.textContent.trim()])""")
    names = {"collect": "Step 1", "remove": "Step 2", "train": "Step 3", "band": "Step 4", "test": "Step 5"}
    check(len(chips) == 13 and all(names.get(s) == t for s, t in chips), f"{len(chips)} step chips, each named for its step")

    # 4. situation colours carried into the tables
    col = pg.evaluate("""() => {
        const cs = getComputedStyle(document.documentElement);
        const tok = k => { const d = document.createElement('i'); d.style.color = `var(${k})`; document.body.appendChild(d);
                           const c = getComputedStyle(d).color; d.remove(); return c; };
        const T = { recorded: tok('--sit-rec'), strong_current: tok('--sit-strong'), big_waves: tok('--sit-waves'), big_kicks: tok('--sit-kicks') };
        const shadowHas = (el, key) => getComputedStyle(el).boxShadow.includes(T[key]);
        const ladder = [...document.querySelectorAll('#results table.p4-ladder tbody tr')].map(tr => shadowHas(tr.cells[0], tr.dataset.sit));
        const ths = [...document.querySelectorAll('#results th[data-sit], #weighting-test th[data-sit]')].map(th => shadowHas(th, th.dataset.sit));
        const rows = [...document.querySelectorAll('#weighting-test tr[data-sit], #results table.stack tr[data-sit]')].map(tr => shadowHas(tr.cells[0], tr.dataset.sit));
        return { ladder, nth: ths.length, ths: ths.every(Boolean), nrows: rows.length, rows: rows.every(Boolean) }; }""")
    check(len(col["ladder"]) == 4 and all(col["ladder"]), f"the ladder's four rows wear their situation colour {col['ladder']}")
    check(col["nth"] == 4 + 17 + 3 and col["ths"], f"{col['nth']} situation column headings wear their colour")
    check(col["nrows"] == 4 + 4 + 2 and col["rows"], f"{col['nrows']} situation rows wear their colour")

    # 5. nothing out of date, no warning red outside the stall
    txt = pg.evaluate("() => [document.getElementById('weighting-test').innerText, document.getElementById('results').innerText]")
    stale = ["exploratory, running", "shown here when finished", "Filled in when the runs finish", "are now running"]
    check(not [x for x in stale if x in txt[0]] and "running</" not in pg.content().split('id="weighting-test"')[1].split("</section>")[0]
          and "under way belong" not in txt[1], "no 'running' placeholder or 'now under way' left")
    check(pg.evaluate("() => document.querySelectorAll('#results .tag.warn').length") == 0, "no warning-red tag in Results")
    ov = pg.evaluate(OVERFLOW_JS)
    check(ov["overflow"] <= 0, f"no horizontal overflow ({ov['overflow']} px)")
    br.close()
check(not errors, f"no console errors ({errors[:3]})")
print(f"{len(fails)} FAILS" if fails else "PASS")
sys.exit(1 if fails else 0)

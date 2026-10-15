"""Browser proof for edit 3 (read-only on the page).

Loads the page in Chromium, lets every script run, then reads what a reader can see or hear: the rendered text of
every top-level block (innerText), every aria-label, title and aria-valuetext, the live panel's explanation box and
readouts for all 32 combinations of situation, world and controller, the walkthrough pop-out's titles, and the demo's
labels. It fails if a short code (H1-H4, T1-T9, D1-D3, C1-C3, a V-code, P2-P4, B1, A1, stage0) appears anywhere
outside the section Names used in the code. It also reports console errors and horizontal overflow.
Run: /opt/shot-venv/bin/python e3_browser.py [--width N] [--scheme light|dark]
"""
import argparse
import os
import re
import sys
import tempfile

from playwright.sync_api import sync_playwright

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from shot import wrap, DEFAULT_PAGE  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--width", type=int, default=390)
ap.add_argument("--scheme", default="light")
ap.add_argument("--page", default=DEFAULT_PAGE)
a = ap.parse_args()

CODES = re.compile(r"\b(?:H[1-4]|T[1-9]|D[1-3]|C[1-3]|V[0-9]|P[2-4]|B1|A1)\b|\bstage ?0\w*")
frag = open(a.page, encoding="utf-8").read()
tmp = os.path.join(tempfile.mkdtemp(prefix="e3-"), "page.html")
open(tmp, "w", encoding="utf-8").write(wrap(frag, a.scheme))
errors, fails = [], []

COLLECT = r"""() => {
  const out = [];
  const main = document.querySelector('main');
  for (const n of main.children) {
    const id = n.id || n.tagName.toLowerCase();
    out.push([id, n.innerText]);
    n.querySelectorAll('[aria-label],[title],[aria-valuetext]').forEach(el => {
      if (el.closest('mjx-container')) return;
      ['aria-label', 'title', 'aria-valuetext'].forEach(k => { const v = el.getAttribute(k); if (v) out.push([id, v]); });
    });
  }
  const nav = document.querySelector('nav.toc'); if (nav) out.push(['nav', nav.innerText]);
  const dlg = document.getElementById('wt'); if (dlg) out.push(['dialog', dlg.innerText + ' ' + (dlg.getAttribute('aria-label') || '')]);
  out.push(['title', document.title]);
  return out;
}"""

with sync_playwright() as p:
    br = p.chromium.launch()
    pg = br.new_page(viewport={"width": a.width, "height": 844 if a.width < 700 else 800}, color_scheme=a.scheme)
    pg.on("console", lambda m: errors.append(m.text) if m.type in ("error", "warning") else None)
    pg.on("pageerror", lambda e: errors.append(str(e)))
    pg.goto("file://" + tmp)
    pg.wait_for_load_state("networkidle")
    pg.wait_for_timeout(800)
    texts = pg.evaluate(COLLECT)
    # every combination of the live panel: explanation box, readouts and the step label at a few states
    combos = 0
    for sit in ["recorded", "strong_current", "big_waves", "big_kicks"]:
        for world in ["stall", "no_stall"]:
            for ctrl in ["recording", "stall_aware", "rising_thrust", "no_throttle"]:
                for sel in (f"#opt-sit-{sit}", f"#opt-world-{world}", f"#opt-ctrl-{ctrl}"):
                    pg.click(sel)
                pg.click("#fwd")
                t = pg.evaluate("() => document.getElementById('explain').innerText + ' ' + "
                                "document.querySelector('#sim .readouts').innerText + ' ' + "
                                "document.getElementById('tLabel').textContent")
                texts.append(["sim-run", t])
                combos += 1
    label = pg.evaluate("() => document.getElementById('tLabel').textContent")
    texts.append(["sim-run", label])
    # the walkthrough pop-out: every section title and counter
    if pg.evaluate("() => !!document.getElementById('wt-open') && !document.querySelector('.wt-bar').hidden"):
        pg.click("#wt-open")
        pg.wait_for_timeout(300)
        n = 0
        while True:
            texts.append(["dialog", pg.evaluate("() => document.getElementById('wt-title').textContent + ' ' + "
                                               "document.getElementById('wt-next-name').textContent")])
            n += 1
            if pg.evaluate("() => document.getElementById('wt-next').textContent.trim() === 'Finish'"):
                break
            pg.click("#wt-next")
            pg.wait_for_timeout(260)
        pg.keyboard.press("Escape")
        pg.wait_for_timeout(200)
        print(f"walkthrough sections read: {n}")
    ov = pg.evaluate("() => document.documentElement.scrollWidth - document.documentElement.clientWidth")
    br.close()

hits = 0
for where, t in texts:
    for m in CODES.finditer(t or ""):
        if where == "code-names":
            continue
        hits += 1
        print(f"FAIL code {m.group(0)!r} in {where}: …{t[max(0, m.start() - 60):m.end() + 60]!r}…")
allowed = sum(len(CODES.findall(t or "")) for w, t in texts if w == "code-names")
print(f"[{a.width} {a.scheme}] blocks read: {len(texts)}; live-panel combinations: {combos}; final step label: {label!r}")
print(f"short codes outside Names used in the code: {hits}; inside it (the mapping): {allowed}")
print(f"horizontal overflow: {ov} px; console errors/warnings and page errors: {len(errors)}")
for e in errors:
    print("   ", e[:300])
ok = hits == 0 and ov <= 0 and not errors
print("PASS" if ok else "FAIL")
sys.exit(0 if ok else 1)

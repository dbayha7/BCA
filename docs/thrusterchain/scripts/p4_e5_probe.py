#!/usr/bin/env python3
"""Read-only probes for page pass 4, edit 5. Loads the page at one width and scheme and reports:

 A. justified running text: how many visible p/li/dd/figcaption elements are justified, and for those, the share of
    their lines (last lines left out) whose widest gap between words is at least 2.5, 3 and 4 normal spaces;
 B. the walkthrough dialog: at every step, the dialog's own scrollHeight minus its clientHeight (hidden overflow);
 C. the stall chart: the crosshair is drawn before (under) the in-plot labels, and the hover readouts sit directly
    above the chart;
 D. the live panel's option cards: each colour key's vertical centre against the first line of its name;
 E. the learned-blue token and the live drawing's thruster colour.

Run: /opt/shot-venv/bin/python scripts/p4_e5_probe.py --page PATH --width 390 --scheme light
"""
import argparse
import json
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from shot import wrap, DEFAULT_PAGE  # noqa: E402
from playwright.sync_api import sync_playwright  # noqa: E402

JS_GAPS = r"""() => {
  const els = [...document.querySelectorAll('.wrap p, .wrap li, .wrap dd, .wrap figcaption')].filter(e => {
    const r = e.getBoundingClientRect(); if (!r.width || !r.height) return false;
    return getComputedStyle(e).textAlign === 'justify' && e.offsetParent !== null; });
  let lines = 0, g25 = 0, g3 = 0, g4 = 0, worst = [];
  for (const el of els) {
    const cs = getComputedStyle(el);
    const c = document.createElement('canvas').getContext('2d'); c.font = cs.font;
    const sp = c.measureText(' ').width || 4;
    const walker = document.createTreeWalker(el, NodeFilter.SHOW_TEXT);
    const byLine = new Map();
    let n;
    while ((n = walker.nextNode())) {
      if (n.parentElement.closest('code, mjx-container, .sym, svg, .nw')) continue;
      const t = n.textContent;
      for (let i = 0; i < t.length; i++) {
        if (t[i] !== ' ') continue;
        const r = document.createRange(); r.setStart(n, i); r.setEnd(n, i + 1);
        const rc = r.getClientRects()[0]; if (!rc || rc.width === 0) continue;
        const key = Math.round(rc.top);
        byLine.set(key, Math.max(byLine.get(key) || 0, rc.width));
      }
    }
    const keys = [...byLine.keys()].sort((a, b) => a - b); keys.pop();   // the last line is not stretched
    for (const k of keys) { const g = byLine.get(k) / sp; lines++; if (g >= 2.5) g25++; if (g >= 3) g3++; if (g >= 4) g4++;
      if (g >= 2.5 && worst.length < 6) worst.push([el.closest('section[id]') ? el.closest('section[id]').id : '-', +g.toFixed(1), el.textContent.trim().slice(0, 50)]); }
  }
  return { justified: els.length, lines, g25, g3, g4, worst };
}"""

JS_WT = r"""async () => {
  const sleep = ms => new Promise(r => setTimeout(r, ms));
  document.getElementById('wt-open').click(); await sleep(400);
  const out = []; const dlg = document.getElementById('wt');
  for (let i = 0; i < 40; i++) {
    out.push([document.getElementById('wt-title').textContent, dlg.scrollHeight - dlg.clientHeight]);
    const nx = document.getElementById('wt-next');
    if (/Finish/.test(nx.textContent)) break;
    nx.click(); await sleep(250);
  }
  document.getElementById('wt-close') ? document.getElementById('wt-close').click() : dlg.close();
  return out;
}"""

JS_SC = r"""() => {
  const svg = document.getElementById('sc-svg'); const kids = [...svg.children];
  const cross = kids.findIndex(e => e.classList && e.classList.contains('ix-cross'));
  const labs = kids.map((e, i) => [i, e]).filter(([i, e]) => e.tagName === 'text' && /stalling|stall region|stall$/.test(e.textContent)).map(([i]) => i);
  const ro = document.querySelector('#stall-chart .readouts').getBoundingClientRect(), sv = svg.getBoundingClientRect();
  const halo = [...svg.querySelectorAll('text.lab')].map(t => getComputedStyle(t).paintOrder + ' / ' + getComputedStyle(t).stroke).slice(0, 1);
  return { crossIndex: cross, labelIndexes: labs, readoutsBottomToChartTop: Math.round(sv.top - ro.bottom), halo };
}"""

JS_OPT = r"""() => [...document.querySelectorAll('#sim .opt')].map(o => {
  const b = o.querySelector('b'); const cs = getComputedStyle(b, '::before');
  const sw = b.querySelector('.sw'); const br = b.getBoundingClientRect();
  const lh = parseFloat(getComputedStyle(b).lineHeight);
  let keyMid = null;
  if (sw) { const r = sw.getBoundingClientRect(); keyMid = r.top + r.height / 2 - br.top; }
  else if (cs.content !== 'none') { keyMid = parseFloat(cs.marginTop) + parseFloat(cs.height) / 2; }
  return [o.id, keyMid == null ? null : +keyMid.toFixed(1), +(lh / 2).toFixed(1), Math.round(br.height / lh)];
})"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--page", default=DEFAULT_PAGE)
    ap.add_argument("--width", type=int, default=390)
    ap.add_argument("--scheme", default="light")
    a = ap.parse_args()
    frag = open(a.page, encoding="utf-8").read()
    tmp = tempfile.NamedTemporaryFile("w", suffix=".html", delete=False, encoding="utf-8")
    tmp.write(wrap(frag, "light dark"))
    tmp.close()
    with sync_playwright() as p:
        b = p.chromium.launch()
        pg = b.new_page(viewport={"width": a.width, "height": 844 if a.width < 700 else 900}, color_scheme=a.scheme)
        errs = []
        pg.on("pageerror", lambda e: errs.append(str(e)))
        pg.on("console", lambda m: errs.append(m.text) if m.type == "error" else None)
        pg.goto("file://" + tmp.name)
        pg.wait_for_timeout(1500)
        gaps = pg.evaluate(JS_GAPS)
        print("A. justified elements:", gaps["justified"], "lines:", gaps["lines"], ">=2.5 spaces:", gaps["g25"],
              ">=3:", gaps["g3"], ">=4:", gaps["g4"])
        for w in gaps["worst"]:
            print("   ", w)
        pg.evaluate("document.getElementById('stall-chart').scrollIntoView({block: 'center'})")
        pg.wait_for_timeout(300)
        print("C. stall chart:", json.dumps(pg.evaluate(JS_SC)))
        opt = pg.evaluate(JS_OPT)
        bad = [o for o in opt if o[1] is not None and abs(o[1] - o[2]) > 2.5]
        print("D. option keys off the first line's centre by more than 2.5 px:", bad if bad else "none", f"({len(opt)} cards)")
        tok = pg.evaluate("getComputedStyle(document.documentElement).getPropertyValue('--k-learn').trim()")
        print("E. --k-learn:", tok)
        wt = pg.evaluate(JS_WT)
        over = [w for w in wt if w[1] > 1]
        print("B. walkthrough steps:", len(wt), "with hidden dialog overflow > 1 px:", over if over else "none")
        print("console/page errors:", len(errs), errs[:5])
        b.close()
    os.unlink(tmp.name)


if __name__ == "__main__":
    main()

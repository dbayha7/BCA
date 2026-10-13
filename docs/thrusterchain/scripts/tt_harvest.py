#!/usr/bin/env python3
"""Tooltip inventory, step 1 of 2: harvest every tooltip text the page can show. Read-only: never writes the page.

Loads a temporary copy of thrusterchain.html in headless Chromium (wrapped as shot.py wraps it). The copy differs from
the page only in three exported hooks, so this script can call the page's own generators directly:
  window.__ttBuild    the shared tooltip's build(anchor)            (script "Tooltips: one shared tooltip")
  window.__ttChannel  the channel drawing's channelTip(x, y, canvas) (script "e6_tooltips")
  window.__ttCost     the cost chart's costTip(x, y, canvas)        (script "e6_tooltips")
Then, at 1200 px and at 390 px wide:
  - every element with data-tip or data-tip-ref (terms, symbols, readouts, controls, chart parts, chart bands,
    formula legends, controller guide), with its location and the text build() produces;
  - every table: column names and descriptions, header tooltips, and each cell's tuple (build() on the cell);
  - the Play buttons in both states; the channel drawing and cost chart over a grid of states, worlds, situations,
    controllers and pointer positions;
  - each tooltip's full height in the real tooltip box (to find the ones the box's max-height clips);
  - any <title> or title attribute left after the scripts ran (native browser tooltips).
Writes tmp/inv/harvest.json. Run:
  cd /mnt/c/Users/David/dev/bca-shift-work/page && /opt/shot-venv/bin/python scripts/tt_harvest.py
"""
import json
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from shot import wrap  # noqa: E402
from playwright.sync_api import sync_playwright  # noqa: E402

PAGE_DIR = "/mnt/c/Users/David/dev/bca-shift-work/page"
PAGE = os.path.join(PAGE_DIR, "thrusterchain.html")
OUT = os.path.join(PAGE_DIR, "tmp", "inv", "harvest.json")

HOOKS = [
    ("  window.ttRefresh = () =>",
     "  window.__ttBuild = build;\n  window.ttRefresh = () =>"),
    ('  wire(document.getElementById("channel"), channelTip);',
     '  window.__ttChannel = channelTip; window.__ttCost = costTip;\n  wire(document.getElementById("channel"), channelTip);'),
]

HARVEST_JS = r"""
() => {
  const build = window.__ttBuild;
  const tip = document.getElementById('tt');
  const clean = s => (s || '').replace(/\s+/g, ' ').trim();
  const secOf = e => { const s = e.closest('section[id]') || e.closest('#glance, #sim, #top, header, figure[id], div[id]'); return s ? s.id : ''; };
  const ancOf = e => { const ids = []; for (let x = e.parentElement; x && ids.length < 3; x = x.parentElement) if (x.id) ids.push(x.id); return ids.join(' < '); };
  const headOf = e => {             // the nearest heading before the element, in document order
    const hs = [...document.querySelectorAll('h1, h2, h3, h4, h5, summary, .sim-k, .optlbl, .g-k')];
    let best = '';
    for (const h of hs) { if (h.compareDocumentPosition(e) & Node.DOCUMENT_POSITION_FOLLOWING || h.contains(e)) best = clean(h.textContent).slice(0, 90); }
    return best;
  };
  const measure = nodes => {
    if (!nodes) return null;
    tip.replaceChildren(...nodes.map(n => n.cloneNode(true)));
    tip.classList.toggle('tt-wide', tip.textContent.length > 360);
    tip.hidden = false; tip.style.left = '0px'; tip.style.top = '0px';
    const r = { h: tip.scrollHeight + 2, w: tip.offsetWidth, vis: tip.offsetHeight };
    tip.hidden = true;
    return r;
  };
  const textOf = nodes => nodes ? nodes.map(n => n.textContent).join('\n') : null;
  const out = { vw: document.documentElement.clientWidth, vh: innerHeight, tips: [], tables: [], natives: [] };

  // ---- data-tip and data-tip-ref anchors ----
  document.querySelectorAll('[data-tip], [data-tip-ref]').forEach((a, i) => {
    if (tip.contains(a)) return;
    const nodes = build(a);
    const inSvg = a.closest('svg');
    const fig = a.closest('figure[id], .chartbox, .iw-chart, [id^="iw-"]');
    const det = a.closest('details');
    const ref = a.dataset.tipRef ? document.getElementById(a.dataset.tipRef) : null;
    out.tips.push({
      i, tag: a.localName, id: a.id || '', cls: String(a.getAttribute('class') || ''), anc: ancOf(a),
      section: secOf(a), heading: headOf(a),
      details: det ? clean(det.querySelector('summary') ? det.querySelector('summary').textContent : '').slice(0, 80) : '',
      svg: inSvg ? (inSvg.id || clean(inSvg.getAttribute('aria-label')).slice(0, 90)) : '',
      figure: fig ? (fig.id || fig.className) : '',
      own: clean(a.localName === 'rect' ? '' : a.textContent).slice(0, 90),
      title: a.dataset.tipTitle || '', tipAttr: a.dataset.tip == null ? null : a.dataset.tip,
      ref: a.dataset.tipRef || '', refHtml: ref ? (ref.content ? [...ref.content.childNodes].map(n => n.outerHTML || n.textContent).join('') : ref.innerHTML) : '',
      focus: a.dataset.tipFocus || '', desc: a.dataset.tipDesc || '',
      shown: textOf(nodes), shownHtml: nodes ? nodes.map(n => n.outerHTML).join('') : null,
      visible: a.getClientRects().length > 0, box: measure(nodes),
      isRect: a.localName === 'rect' && !!a.closest('.pt-hits'),
    });
  });

  // ---- tables ----
  const tabs = [...document.querySelectorAll('table')];
  tabs.forEach((t, ti) => {
    const sec = secOf(t);
    const idxInSec = [...document.querySelectorAll('section#' + CSS.escape(sec) + ' table')].indexOf(t);
    const cap = t.caption ? clean(t.caption.textContent) : '';
    const rec = { ti, section: sec, idxInSec, heading: headOf(t), caption: cap, cls: String(t.className),
                  headers: [], rows: [], nCells: 0 };
    t.querySelectorAll('[data-tt="h"]').forEach(h => {
      const nodes = build(h);
      rec.headers.push({ own: clean(h.childNodes ? [...h.childNodes].filter(n => !(n.classList && n.classList.contains('tt-sr'))).map(n => n.textContent).join('') : ''),
                         shown: textOf(nodes), box: measure(nodes) });
    });
    const rows = [...t.rows];
    rows.forEach((tr, ri) => {
      const cells = [...tr.cells].filter(c => c.dataset.tt === 'c');
      if (!cells.length) { if (tr.dataset.ttGroup !== undefined) rec.rows.push({ ri, group: clean(tr.textContent) }); return; }
      const row = { ri, cells: [] };
      cells.forEach(c => {
        const nodes = build(c);
        if (!nodes) { row.cells.push({ value: clean(c.textContent), shown: null }); return; }
        const tup = nodes[0], col = nodes[1];
        row.cells.push({
          value: tup.querySelector('.cur b') ? tup.querySelector('.cur b').textContent : null,
          key: tup.querySelector('.cur .k') ? tup.querySelector('.cur .k').textContent : '',
          tuple: tup.textContent, col: col ? col.textContent : '', colName: col && col.querySelector('b') ? col.querySelector('b').textContent : '',
          pairs: [...tup.querySelectorAll(':scope > .p')].map(p => [p.querySelector('.k') ? p.querySelector('.k').textContent : '', clean(p.textContent.slice(p.querySelector('.k') ? p.querySelector('.k').textContent.length : 0))]),
          raw: clean(c.textContent), box: measure(nodes),
        });
        rec.nCells++;
      });
      rec.rows.push(row);
    });
    out.tables.push(rec);
  });

  // ---- native tooltips left: <title> elements and title attributes ----
  document.querySelectorAll('title').forEach(t => { if (t.parentElement && t.parentElement.localName !== 'head') out.natives.push({ kind: '<title>', parent: t.parentElement.localName, text: clean(t.textContent), section: secOf(t) }); });
  document.querySelectorAll('[title]').forEach(e => out.natives.push({ kind: 'title attr', parent: e.localName, text: e.getAttribute('title'), section: secOf(e) }));
  return out;
}
"""

PLAY_JS = r"""
async () => {
  const res = [];
  const tip = document.getElementById('tt');
  for (const id of ['play', 'td-play']) {
    const b = document.getElementById(id);
    if (!b) continue;
    const t0 = window.__ttBuild(b).map(n => n.textContent).join('\n');
    const lab0 = b.textContent.trim();
    b.click(); await new Promise(r => setTimeout(r, 300));
    const t1 = window.__ttBuild(b).map(n => n.textContent).join('\n');
    const lab1 = b.textContent.trim();
    b.click(); await new Promise(r => setTimeout(r, 300));
    res.push({ id, states: [[lab0, t0], [lab1, t1]], section: (b.closest('section[id]') || {}).id || '' });
  }
  return res;
}
"""

DYN_JS = r"""
() => {
  const out = { channel: [], cost: [], meta: {} };
  if (!window.__ttChannel || typeof S === 'undefined') { out.meta.error = 'no hooks'; return out; }
  const cv = document.getElementById('channel'), cc = document.getElementById('cost');
  const W = cv.clientWidth, narrow = W < 560, chanTop = narrow ? 66 : 60, chanH = narrow ? 176 : 196, col = W / 5;
  const worlds = [...new Set(Object.keys(D.roll).map(k => k.split('|')[0]))];
  const sits = [...new Set(Object.keys(D.roll).map(k => k.split('|')[1]))];
  out.meta = { W, worlds, sits, ctrls: CTRL, keys: Object.keys(D.roll).length, Skeys: Object.keys(S) };
  const save = JSON.stringify(S);
  const tmpl = new Map();           // the text with its numbers replaced by #: one example each
  const note = (kind, ctx, r) => {
    const txt = r ? r.join('\n') : '(none: the general text stays)';
    const k = kind + '|' + txt.replace(/[−-]?\d+(\.\d+)?/g, '#');
    const e = tmpl.get(k);
    if (e) { e.n++; return; }
    tmpl.set(k, { kind, ctx, example: r, template: k.slice(kind.length + 1), n: 1 });
  };
  const pts = [];
  pts.push(['water', W / 2, chanTop / 2]);
  for (let j = 0; j < 5; j++) pts.push(['float ' + (j + 1), (j + 0.5) * col, chanTop + chanH / 2]);
  for (let k = 0; k < 4; k++) pts.push(['thruster ' + (k + 1), (k + 1) * col, chanTop + chanH / 2]);
  for (let k = 0; k < 4; k++) pts.push(['gauge ' + (k + 1), (k + 1) * col, chanTop + chanH + 20]);
  pts.push(['gauge gap', 2, chanTop + chanH + 20]);
  for (const w of worlds) for (const s of sits) for (const c of CTRL) {
    if (!D.roll[w + '|' + s + '|' + c]) continue;
    for (let t = 0; t < 100; t += 1) {
      S.world = w; S.sit = s; S.ctrl = c; S.t = t;
      for (const [part, x, y] of pts) {
        let r;
        try { r = window.__ttChannel(x, y, cv); } catch (e) { r = ['ERROR', String(e)]; }
        note('channel ' + part.replace(/\d/, 'k'), [w, s, c, t, part], r);
      }
    }
  }
  const cw = cc.clientWidth;
  for (const w of worlds) for (const s of sits) {
    S.world = w; S.sit = s; S.ctrl = CTRL[0];
    for (let x = 0; x <= cw; x += Math.max(4, cw / 120)) {
      let r;
      try { r = window.__ttCost(x, 50, cc); } catch (e) { r = ['ERROR', String(e)]; }
      note('cost', [w, s, S.ctrl, Math.round(x)], r);
    }
  }
  Object.assign(S, JSON.parse(save));
  out.templates = [...tmpl.values()];
  out.channel = out.templates.filter(t => t.kind.startsWith('channel')).length;
  out.cost = out.templates.filter(t => t.kind === 'cost').length;
  out.meta.base = { channel: [cv.getAttribute('data-tip-title'), cv.getAttribute('data-tip')], cost: [cc.getAttribute('data-tip-title'), cc.getAttribute('data-tip')] };
  return out;
}
"""


def main():
    s = open(PAGE, encoding="utf-8").read()
    for old, new in HOOKS:
        assert s.count(old) == 1, f"hook anchor not found exactly once: {old!r}"
        s = s.replace(old, new, 1)
    d = tempfile.mkdtemp(prefix="ttinv-")
    f = os.path.join(d, "page.html")
    open(f, "w", encoding="utf-8").write(wrap(s, "light"))
    res = {}
    with sync_playwright() as p:
        b = p.chromium.launch()
        for width in (1200, 390):
            ctx = b.new_context(viewport={"width": width, "height": 844 if width < 700 else 800}, color_scheme="light",
                                has_touch=width < 700, is_mobile=width < 700)
            pg = ctx.new_page()
            errs = []
            pg.on("pageerror", lambda e: errs.append(str(e)))
            pg.on("console", lambda m: errs.append(f"{m.type}: {m.text}") if m.type == "error" else None)
            pg.goto("file://" + f, wait_until="load")
            pg.wait_for_timeout(1200)
            # open every <details> so hidden tables and legends are measured with their real width
            pg.evaluate("document.querySelectorAll('details').forEach(d => d.open = true)")
            pg.wait_for_timeout(300)
            r = pg.evaluate(HARVEST_JS)
            r["play"] = pg.evaluate(PLAY_JS)
            r["dyn"] = pg.evaluate(DYN_JS)
            r["errors"] = errs
            res[str(width)] = r
            print(f"[{width}] tips {len(r['tips'])}, tables {len(r['tables'])}, cells {sum(t['nCells'] for t in r['tables'])}, "
                  f"natives {len(r['natives'])}, channel templates {r['dyn'].get('channel')}, cost templates {r['dyn'].get('cost')}, "
                  f"meta {r['dyn'].get('meta')}, errors {len(errs)}")
            for e in errs[:5]:
                print("   ", e[:200])
            ctx.close()
        b.close()
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    json.dump(res, open(OUT, "w", encoding="utf-8"), ensure_ascii=False)
    print("wrote", OUT, os.path.getsize(OUT), "bytes")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Read-only contrast check for pass 4 (edit 1). Loads the page in Chromium in both themes and computes WCAG
contrast ratios:

 A. the text tokens (ink, ink-2, muted, warn, accent-ink, region ink, ok) against every surface and tint the page
    defines, read from the live computed tokens (so it checks what the page really uses);
 B. every visible text element on the page: its computed colour against the first solid background behind it
    (walking up the tree), sorted into body text (at least 7:1 wanted) and short labels (at least 4.5:1). Text on a
    gradient or image background, text inside SVG, and disabled controls are skipped and counted.

Run: /opt/shot-venv/bin/python scripts/p4_e1_contrast.py [--page PATH] [--list N]
"""
import argparse
import json
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from shot import wrap, DEFAULT_PAGE  # noqa: E402
from playwright.sync_api import sync_playwright  # noqa: E402

TEXT = ["--ink", "--ink-2", "--muted", "--warn", "--accent-ink", "--k-region-ink", "--ok"]
FLOOR = {"--ink": 7, "--ink-2": 7, "--muted": 4.5, "--warn": 4.5, "--accent-ink": 4.5, "--k-region-ink": 4.5, "--ok": 4.5}
SURF = ["--surface", "--bg", "--surface-2", "--water", "--ok-soft", "--accent-soft", "--warn-soft", "--region",
        "--sit-rec-soft", "--sit-strong-soft", "--sit-waves-soft", "--sit-kicks-soft", "--k-stall-soft", "--k-rise-soft",
        "--k-region-soft", "--k-learn-soft"]

JS_TOKENS = """(names) => { const cs = getComputedStyle(document.documentElement);
  const probe = document.createElement('span'); document.body.appendChild(probe);
  const out = {}; for (const n of names) { probe.style.color = 'var(' + n + ')'; out[n] = getComputedStyle(probe).color; }
  probe.remove(); return out; }"""

JS_SCAN = r"""() => {
  const parse = c => { const m = c.match(/rgba?\(([^)]+)\)/); if (!m) return null;
    const p = m[1].split(/[ ,\/]+/).filter(Boolean).map(Number); return [p[0], p[1], p[2], p.length > 3 ? p[3] : 1]; };
  const lin = v => { v /= 255; return v <= 0.03928 ? v / 12.92 : Math.pow((v + 0.055) / 1.055, 2.4); };
  const L = c => 0.2126 * lin(c[0]) + 0.7152 * lin(c[1]) + 0.0722 * lin(c[2]);
  const ratio = (a, b) => { const x = L(a), y = L(b); return (Math.max(x, y) + 0.05) / (Math.min(x, y) + 0.05); };
  const over = (fg, bg) => [0, 1, 2].map(i => fg[i] * fg[3] + bg[i] * (1 - fg[3])).concat([1]);
  const res = { checked: 0, skipped: 0, low: [], hist: {} };
  const seen = new Set();
  for (const el of document.querySelectorAll('main *, nav.toc *, header *')) {
    if (el.closest('svg, mjx-container, canvas, template, script, style')) continue;
    const own = [...el.childNodes].some(n => n.nodeType === 3 && n.nodeValue.trim().length);
    if (!own) continue;
    const r = el.getBoundingClientRect(); if (!r.width || !r.height) continue;
    const st = getComputedStyle(el); if (st.visibility === 'hidden' || +st.opacity === 0) continue;
    if (el.closest('.tt-sr, .wt-sr, [aria-hidden="true"]') && !el.closest('.legend')) continue;
    let fg = parse(st.color); if (!fg) continue;
    let bg = null, a = el, img = false;
    while (a) { const s = getComputedStyle(a);
      if (s.backgroundImage && s.backgroundImage !== 'none' && a !== el && !a.classList.contains('km')) { img = true; break; }
      const b = parse(s.backgroundColor); if (b && b[3] > 0) { bg = b[3] >= 1 ? b : null; if (bg) break; }
      a = a.parentElement; }
    if (img || !bg) { res.skipped++; continue; }
    if (fg[3] < 1) fg = over(fg, bg);
    const k = ratio(fg, bg); res.checked++;
    const size = parseFloat(st.fontSize), bold = +st.fontWeight >= 600;
    const text = [...el.childNodes].filter(n => n.nodeType === 3).map(n => n.nodeValue).join('').trim().replace(/\s+/g, ' ');
    const label = text.length <= 40;
    const need = label ? 4.5 : 7;
    const bucket = k >= 7 ? '>=7' : k >= 4.5 ? '4.5-7' : '<4.5';
    res.hist[bucket] = (res.hist[bucket] || 0) + 1;
    if (k < need) { const key = st.color + '|' + getComputedStyle(a).backgroundColor + '|' + label;
      if (!seen.has(key)) { seen.add(key);
        const sec = el.closest('section[id], figure[id], header');
        res.low.push({ ratio: +k.toFixed(2), need, color: st.color, bg: getComputedStyle(a).backgroundColor,
          tag: el.tagName.toLowerCase(), cls: String(el.className).slice(0, 40), sec: sec ? (sec.id || sec.tagName) : '',
          size, text: text.slice(0, 70) }); } }
  }
  res.low.sort((x, y) => x.ratio - y.ratio);
  return res;
}"""


def rgb(s):
    p = s[s.index("(") + 1:s.index(")")].replace("/", " ").replace(",", " ").split()
    return [float(x) for x in p[:3]]


def lum(c):
    def f(v):
        v /= 255
        return v / 12.92 if v <= 0.03928 else ((v + 0.055) / 1.055) ** 2.4
    return 0.2126 * f(c[0]) + 0.7152 * f(c[1]) + 0.0722 * f(c[2])


def ratio(a, b):
    x, y = lum(a), lum(b)
    return (max(x, y) + 0.05) / (min(x, y) + 0.05)


def hexs(c):
    return "#%02x%02x%02x" % tuple(round(v) for v in c)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--page", default=DEFAULT_PAGE)
    ap.add_argument("--list", type=int, default=25)
    a = ap.parse_args()
    frag = open(a.page, encoding="utf-8").read()
    fails = 0
    with sync_playwright() as p:
        br = p.chromium.launch()
        for scheme in ("light", "dark"):
            tmp = os.path.join(tempfile.mkdtemp(prefix="con-"), "page.html")
            open(tmp, "w", encoding="utf-8").write(wrap(frag, scheme))
            for width, theme in ((1200, None), (390, None), (1200, scheme)):
                ctx = br.new_context(viewport={"width": width, "height": 900}, color_scheme=scheme)
                pg = ctx.new_page()
                pg.goto("file://" + tmp)
                pg.wait_for_timeout(800)
                label = f"{scheme} {width}" + (f" data-theme={theme}" if theme else "")
                if theme:   # the page's own theme toggle sets the same tokens through :root[data-theme]
                    pg.evaluate(f"document.documentElement.setAttribute('data-theme', '{theme}')")
                    pg.wait_for_timeout(200)
                tok = pg.evaluate(JS_TOKENS, TEXT + SURF)
                if width == 1200:
                    print(f"\n== A. text tokens against every surface ({label})")
                    print("   " + "token".ljust(15) + "".join(n.replace("--", "")[:11].rjust(12) for n in SURF) + "      lowest")
                    for t in TEXT:
                        rs = [ratio(rgb(tok[t]), rgb(tok[sf])) for sf in SURF]
                        lo = min(rs)
                        ok = lo >= FLOOR[t]
                        fails += 0 if ok else 1
                        print(("   " if ok else " ! ") + f"{t.replace('--', '')} {hexs(rgb(tok[t]))}".ljust(15) +
                              "".join(f"{r:12.2f}" for r in rs) + f"   {lo:6.2f} (needs {FLOOR[t]})")
                    print("   surfaces: " + ", ".join(f"{n.replace('--', '')} {hexs(rgb(tok[n]))}" for n in SURF))
                if theme:
                    ctx.close()
                    continue
                res = pg.evaluate(JS_SCAN)
                print(f"\n== B. every visible text element ({label}): {res['checked']} checked, {res['skipped']} skipped "
                      f"(gradient or image backgrounds); ratios: {res['hist']}")
                print(f"   below its floor (7:1 for text longer than 40 characters, 4.5:1 for short labels): {len(res['low'])} distinct colour pairs")
                for r in res["low"][:a.list]:
                    print(f"   {r['ratio']:5.2f} < {r['need']}  {r['color']} on {r['bg']}  <{r['tag']} .{r['cls']}> #{r['sec']} "
                          f"{r['size']}px  {r['text']!r}")
                fails += sum(1 for r in res["low"] if r["ratio"] < 4.5)
                ctx.close()
        br.close()
    print("\nFAILS (token floors missed, or any text under 4.5:1):", fails)
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()

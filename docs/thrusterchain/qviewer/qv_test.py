#!/usr/bin/env python3
"""Browser checks of the critic-in-action section in qviewer/test_page.html (headless Chromium, as shot.py loads it).

Checks, at 390 px (light) and 1200 px (dark): console errors and page errors, horizontal overflow, that the charts
draw for every one of the 20 episodes at states 0, 40 and 99 in both views and both ranges, that the readouts agree
with the data, keyboard reading of both charts, play/pause, no autoplay, and which elements inside the section carry a
tooltip the section did not write itself. Prints a report; exit status 1 on a failed check.
Run: /opt/shot-venv/bin/python qv_test.py
"""
import json
import os
import sys
import tempfile

from playwright.sync_api import sync_playwright

sys.path.insert(0, "/mnt/c/Users/David/dev/bca-shift-work/page/scripts")
import shot  # noqa: E402  (its wrap() and overflow check)

PAGE = os.environ.get("QV_PAGE") or "/mnt/c/Users/David/dev/bca-shift-work/qviewer/test_page.html"   # QV_PAGE=path checks another page (read only)

JS_ALL = r"""
() => {
  const out = { bad: [], n: 0 };
  const D = JSON.parse(document.getElementById('qv-data').textContent);
  for (const q of D.qualities) for (const c of D.critics) for (const s of D.situations) for (const v of ['rel', 'abs']) for (const rg of ['all', 'hi'])
    for (const t of [0, 40, 99]) {
      const st = window.qvSet({ q, c, s, view: v, range: rg, t });
      out.n++;
      const p1 = document.querySelectorAll('#qv-svg1 path.qv-lq, #qv-svg1 path.qv-lp').length;
      const p2 = document.querySelectorAll('#qv-svg2 path.qv-lq, #qv-svg2 path.qv-lp').length;
      const d1 = [...document.querySelectorAll('#qv-svg1 path.qv-lq, #qv-svg1 path.qv-lp')].map(p => p.getAttribute('d'));
      const nan = d1.some(d => /NaN|Infinity/.test(d)) || /NaN|undefined|Infinity/.test(document.getElementById('critic-in-action').innerText);
      if (p1 !== 2 || p2 !== 2 || nan || st.t !== t) out.bad.push([q, c, s, v, rg, t, p1, p2, nan]);
    }
  return out;
}
"""

JS_RO = r"""
() => {
  window.qvSet({ q: 'expert', c: 'gap', s: 'strong', view: 'rel', range: 'all', t: 40 });
  const g = id => document.getElementById(id).textContent;
  return { a1: g('qv-ro-a1'), hat: g('qv-ro-hat'), best: g('qv-ro-best'), err: g('qv-ro-err'), cost: g('qv-ro-cost'),
           ret: g('qv-ro-ret'), pts: g('qv-ro-pts'), now: g('qv-now'), lab: g('qv-tl') };
}
"""

JS_TIPS = r"""
() => {
  const sec = document.getElementById('critic-in-action');
  return [...sec.querySelectorAll('[data-tip], [data-tip-ref]')].map(e => [e.tagName.toLowerCase(), (e.id || ''),
    (e.getAttribute('data-tip-title') || ''), (e.textContent || '').trim().slice(0, 40)]);
}
"""


def run(width, scheme, report, src_tips):
    frag = open(PAGE, encoding="utf-8").read()
    tmp = tempfile.mkdtemp(prefix="qvt-")
    path = os.path.join(tmp, "page.html")
    open(path, "w", encoding="utf-8").write(shot.wrap(frag, "light"))
    errs = []
    ok = True
    with sync_playwright() as p:
        b = p.chromium.launch()
        ctx = b.new_context(viewport={"width": width, "height": 900}, color_scheme=scheme)
        pg = ctx.new_page()
        pg.on("console", lambda m: errs.append(f"{m.type}: {m.text}") if m.type in ("error", "warning") else None)
        pg.on("pageerror", lambda e: errs.append("pageerror: " + str(e)))
        pg.goto("file://" + path, wait_until="load")
        pg.wait_for_timeout(600)
        st = pg.evaluate("window.qvState()")
        report.append(f"[{width} {scheme}] initial state {st}")
        if st["playing"] or st["t"] != 0:
            ok = False; report.append("  FAIL: plays or moved on load")
        pg.wait_for_timeout(1200)
        if pg.evaluate("window.qvState().t") != 0:
            ok = False; report.append("  FAIL: the state moved without a press (autoplay)")
        r = pg.evaluate(JS_ALL)
        report.append(f"  episodes x views x states drawn: {r['n']}, bad: {r['bad'][:5]}")
        ok &= not r["bad"]
        ov = pg.evaluate(shot.OVERFLOW_JS)
        report.append(f"  horizontal overflow: {ov['overflow']} px {ov['offenders'][:3]}")
        ok &= ov["overflow"] <= 0
        ro = pg.evaluate(JS_RO)
        report.append("  readouts at expert/gap/strong t=40: " + json.dumps(ro, ensure_ascii=False)[:400])
        # keyboard: focus the upper chart, arrows move the readout
        pg.evaluate("document.getElementById('qv-svg1').scrollIntoView({block: 'center'})")
        pg.focus("#qv-svg1")
        pg.keyboard.press("ArrowLeft")
        pg.keyboard.press("ArrowLeft")
        tip1 = pg.evaluate("(() => { const t = document.getElementById('qv-tip1'); return t.hidden ? null : t.innerText; })()")
        report.append("  keyboard, upper chart: " + (tip1 or "NO TOOLTIP").replace("\n", " | ")[:300])
        ok &= bool(tip1)
        pg.focus("#qv-svg2")
        pg.keyboard.press("ArrowRight")
        t2 = pg.evaluate("window.qvState().t")
        tip2 = pg.evaluate("(() => { const t = document.getElementById('qv-tip2'); return t.hidden ? null : t.innerText; })()")
        report.append(f"  keyboard, lower chart: state now {t2}; " + (tip2 or "NO TOOLTIP").replace("\n", " | ")[:300])
        ok &= t2 == 41 and bool(tip2)
        # hover the upper chart with the mouse
        pg.evaluate("document.getElementById('qv-svg1').scrollIntoView({block: 'center'})")
        pg.wait_for_timeout(150)
        box = pg.locator("#qv-svg1").bounding_box()
        pg.mouse.move(box["x"] + box["width"] * 0.85, box["y"] + box["height"] * 0.5)
        pg.wait_for_timeout(200)
        tip1m = pg.evaluate("(() => { const t = document.getElementById('qv-tip1'); return t.hidden ? null : t.innerText; })()")
        report.append("  mouse, upper chart: " + (tip1m or "NO TOOLTIP").replace("\n", " | ")[:300])
        ok &= bool(tip1m)
        # play, then pause
        pg.evaluate("window.qvSet({t: 0})")
        pg.click("#qv-play")
        pg.wait_for_timeout(1500)
        st = pg.evaluate("window.qvState()")
        lab = pg.text_content("#qv-play")
        pg.click("#qv-play")
        st2 = pg.evaluate("window.qvState()")
        report.append(f"  play 1.5 s: t {st['t']} playing {st['playing']} button '{lab}'; after pause playing {st2['playing']}")
        ok &= st["t"] >= 2 and st["playing"] and not st2["playing"] and lab == "Pause"
        tips = pg.evaluate(JS_TIPS)
        extra = [x for x in tips if x[2] not in src_tips]
        report.append(f"  tooltips in the section: {len(tips)}; not written by the section: {extra[:8]}")
        report.append(f"  console errors/warnings: {len(errs)} {errs[:5]}")
        ok &= not errs
        b.close()
    return ok


def main():
    import re
    sec = open("/mnt/c/Users/David/dev/bca-shift-work/qviewer/section.html", encoding="utf-8").read()
    src_tips = set(m.replace("&#x27;", "'") for m in re.findall(r'data-tip-title="([^"]*)"', sec)) | {"Play", "Pause"}
    report = []
    ok = run(390, "light", report, src_tips)
    ok &= run(1200, "dark", report, src_tips)
    print("\n".join(report))
    print("ALL CHECKS PASSED" if ok else "SOME CHECKS FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())

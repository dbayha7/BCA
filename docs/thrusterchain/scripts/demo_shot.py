"""Browser test of the task-diagram demo on a scratch copy of the page (never the page itself).

Run in WSL with the screenshot venv:
  /opt/shot-venv/bin/python demo_shot.py
Steps: demo_apply.py (with --caption) makes a scratch copy; this wraps it in a document skeleton as the publish step
does; Chromium then checks autoplay, reduced motion, controls, phone width, label collisions (real text boxes), console
errors, and saves screenshots of the figure to page/notes/demo_shots/.
"""
import json
import os
import subprocess
import sys
from playwright.sync_api import sync_playwright

PAGE_DIR = "/mnt/c/Users/David/dev/bca-shift-work/page"
SCRIPTS = os.path.join(PAGE_DIR, "scripts")
SHOTS = os.path.join(PAGE_DIR, "notes", "demo_shots")
SCRATCH = sys.argv[1]   # a folder that persists between WSL calls, e.g. the session scratchpad under /mnt/c
os.makedirs(SHOTS, exist_ok=True)
os.makedirs(SCRATCH, exist_ok=True)
applied = os.path.join(SCRATCH, "applied.html")
subprocess.run([sys.executable, os.path.join(SCRIPTS, "demo_apply.py"), os.path.join(PAGE_DIR, "thrusterchain.html"),
                applied, "--caption"], check=True)
frag = open(applied, encoding="utf-8").read()
doc = os.path.join(SCRATCH, "doc.html")
open(doc, "w", encoding="utf-8").write(
    '<!doctype html><html lang="en"><head><meta charset="utf-8">'
    '<meta name="viewport" content="width=device-width, initial-scale=1">' + frag + "</html>")
URL = "file://" + doc

results, errors = {}, []

SET_T = """t => { const s = document.getElementById('td-slider'); s.value = String(t);
  s.dispatchEvent(new Event('input', { bubbles: true })); }"""

# real text boxes: every float circle against every visible text (except its own number) and every thruster arrow
COLLIDE = """() => {
  const svg = [...document.querySelectorAll('#task-diagram svg.dg')].find(s => s.getBoundingClientRect().width > 0);
  const vis = e => e.style.display !== 'none' && getComputedStyle(e).display !== 'none';
  // ink boxes: a text's real width, its height from the baseline (cap height above, a descender allowance below);
  // a circle with its 2 px stroke; other shapes as drawn
  const bb = e => { const b = e.getBBox();
    if (e.tagName === 'text') { const y = +e.getAttribute('y'), fs = parseFloat(getComputedStyle(e).fontSize);
      return { x0: b.x, x1: b.x + b.width, y0: y - 0.72 * fs, y1: y + 0.22 * fs }; }
    if (e.tagName === 'circle') { const cx = +e.getAttribute('cx'), cy = +e.getAttribute('cy'), r = +e.getAttribute('r') + 1;
      return { x0: cx - r, x1: cx + r, y0: cy - r, y1: cy + r }; }
    return { x0: b.x, x1: b.x + b.width, y0: b.y, y1: b.y + b.height }; };
  const ov = (a, b, pad) => a.x1 - pad > b.x0 && a.x0 + pad < b.x1 && a.y1 - pad > b.y0 && a.y0 + pad < b.y1;
  const texts = [...svg.querySelectorAll('text')].filter(t => vis(t) && !t.classList.contains('dg-n'));
  const floats = [...svg.querySelectorAll('circle.dg-float')], heads = [...svg.querySelectorAll('polygon.dg-thrarrow')].filter(vis);
  const hits = [];
  floats.forEach((c, j) => texts.forEach(t => { if (ov(bb(c), bb(t), 1)) hits.push('float ' + (j + 1) + ' / ' + t.textContent); }));
  heads.forEach((h, k) => texts.forEach(t => { if (ov(bb(h), bb(t), 1)) hits.push('arrow ' + (k + 1) + ' / ' + t.textContent); }));
  const ch = bb(svg.querySelector('rect.dg-chan'));
  floats.forEach((c, j) => { const b = bb(c); if (b.y0 < ch.y0 || b.y1 > ch.y1) hits.push('float ' + (j + 1) + ' outside channel'); });
  return { layout: svg.getAttribute('class'), hits };
}"""


def sweep(page, label):
    page.evaluate("() => window.taskDemo.setPlaying(false)")
    counts, examples = {}, []
    for t in range(100):
        page.evaluate(SET_T, t)
        got = page.evaluate("() => window.taskDemo.S.t")
        if got != t:
            counts["slider did not move the demo"] = counts.get("slider did not move the demo", 0) + 1
        r = page.evaluate(COLLIDE)
        for h in r["hits"]:
            counts[h] = counts.get(h, 0) + 1
            if len(examples) < 8:
                examples.append(f"step {t + 1}: {h}")
    results[f"collisions {label} ({r['layout']})"] = {"by pair (frames)": counts, "examples": examples}


with sync_playwright() as p:
    br = p.chromium.launch()

    def ctx(width, height, scheme="light", motion="no-preference"):
        c = br.new_context(viewport={"width": width, "height": height}, color_scheme=scheme, reduced_motion=motion,
                           device_scale_factor=2)
        pg = c.new_page()
        pg.on("console", lambda m: errors.append(f"[{width}/{scheme}/{motion}] console.{m.type}: {m.text}")
              if m.type in ("error", "warning") else None)
        pg.on("pageerror", lambda e: errors.append(f"[{width}/{scheme}/{motion}] pageerror: {e}"))
        pg.goto(URL, wait_until="load")
        return c, pg

    # wide, light: autoplay, then stills
    c, pg = ctx(1280, 900)
    results["wide: demo armed"] = pg.evaluate("() => !!window.taskDemo && !document.getElementById('td-ctrls').hidden")
    results["wide: playing before scrolled into view"] = pg.evaluate("() => window.taskDemo.S.playing")
    pg.locator("#task-diagram").scroll_into_view_if_needed()
    pg.wait_for_timeout(1700)
    results["wide: autoplay after 1.7 s in view (playing, step)"] = pg.evaluate("() => [window.taskDemo.S.playing, window.taskDemo.S.t + 1]")
    pg.locator("#td-play").click()
    results["wide: after Pause click (playing, user, button)"] = pg.evaluate(
        "() => [window.taskDemo.S.playing, window.taskDemo.S.user, document.getElementById('td-play').textContent]")
    for t in (0, 9, 24, 99):
        pg.evaluate(SET_T, t)
        pg.locator("#task-diagram").screenshot(path=os.path.join(SHOTS, f"wide_light_step{t + 1:03d}.png"))
    results["wide: readouts at step 100"] = pg.evaluate(
        "() => ['td-step','td-push','td-pushparts','td-thr','td-thrpush','td-ret'].map(i => document.getElementById(i).textContent)")
    pg.locator("#td-back").click()
    results["wide: step after Step back from 100"] = pg.evaluate("() => document.getElementById('td-step').textContent")
    pg.locator("#td-play").click()
    pg.wait_for_timeout(700)
    results["wide: user Play from 99 plays to the end and stops (playing, step)"] = pg.evaluate("() => [window.taskDemo.S.playing, window.taskDemo.S.t + 1]")
    pg.locator("#td-play").click()
    pg.wait_for_timeout(700)
    results["wide: user Play at 100 restarts from 1 (playing, step)"] = pg.evaluate("() => [window.taskDemo.S.playing, window.taskDemo.S.t + 1]")
    pg.locator("#td-play").click()
    sweep(pg, "wide")
    c.close()

    # wide, dark
    c, pg = ctx(1280, 900, scheme="dark")
    pg.evaluate("() => window.taskDemo.setPlaying(false)")
    pg.evaluate(SET_T, 2)
    pg.locator("#task-diagram").screenshot(path=os.path.join(SHOTS, "wide_dark_step003.png"))
    c.close()

    # phone width
    c, pg = ctx(400, 860)
    pg.locator("#task-diagram").scroll_into_view_if_needed()
    pg.wait_for_timeout(600)
    pg.evaluate("() => window.taskDemo.setPlaying(false)")
    results["narrow: page scroll width at 400 px"] = pg.evaluate("() => document.documentElement.scrollWidth")
    results["narrow: visible drawing"] = pg.evaluate(
        "() => [...document.querySelectorAll('#task-diagram svg.dg')].filter(s => s.getBoundingClientRect().width > 0).map(s => s.getAttribute('class'))")
    results["narrow: figure and controls widths"] = pg.evaluate(
        "() => ['task-diagram','td-ctrls'].map(i => Math.round(document.getElementById(i).getBoundingClientRect().width))")
    for t in (0, 24):
        pg.evaluate(SET_T, t)
        pg.locator("#task-diagram").screenshot(path=os.path.join(SHOTS, f"narrow_light_step{t + 1:03d}.png"))
    sweep(pg, "narrow")
    c.close()

    c, pg = ctx(400, 860, scheme="dark")
    pg.evaluate("() => window.taskDemo.setPlaying(false)")
    pg.evaluate(SET_T, 0)
    pg.locator("#task-diagram").screenshot(path=os.path.join(SHOTS, "narrow_dark_step001.png"))
    c.close()

    # reduced motion: no autoplay; Play still works, in whole steps
    c, pg = ctx(1280, 900, motion="reduce")
    pg.locator("#task-diagram").scroll_into_view_if_needed()
    pg.wait_for_timeout(1500)
    results["reduced motion: after 1.5 s in view (playing, step)"] = pg.evaluate("() => [window.taskDemo.S.playing, window.taskDemo.S.t + 1]")
    pg.locator("#td-play").click()
    pg.wait_for_timeout(900)
    results["reduced motion: Play clicked, after 0.9 s (playing, step)"] = pg.evaluate("() => [window.taskDemo.S.playing, window.taskDemo.S.t + 1]")
    results["reduced motion: float 1 cy is a whole-step value"] = pg.evaluate(
        """() => { const S = window.taskDemo.S, v = window.taskDemo.views[0];
                   return v.floats[0].getAttribute('cy'); }""")
    c.close()

    # no data object: the static drawing must stay untouched
    c = br.new_context(viewport={"width": 1280, "height": 900})
    pg = c.new_page()
    pg.on("pageerror", lambda e: errors.append(f"[no-data, expected from the live panel's own script] pageerror: {e}"))
    nodata = os.path.join(SCRATCH, "nodata.html")
    open(nodata, "w", encoding="utf-8").write(open(doc, encoding="utf-8").read().replace(
        "const D = {", "const D_unused = {", 1))
    pg.goto("file://" + nodata, wait_until="load")
    results["no data: demo kept static (taskDemo, controls hidden)"] = pg.evaluate(
        "() => [window.taskDemo, document.getElementById('td-ctrls').hidden]")
    c.close()
    br.close()

print(json.dumps(results, indent=1, ensure_ascii=False))
print("errors:" if errors else "no console errors or page errors")
for e in errors:
    print(" ", e)

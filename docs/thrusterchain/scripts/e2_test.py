"""Edit 2 browser test and screenshots: the task diagram's demo episode, on the page itself (read only).

Run in WSL:  /opt/shot-venv/bin/python /mnt/c/Users/David/dev/bca-shift-work/page/scripts/e2_test.py [--no-shots]
For 390 px and 1200 px, light and dark: the figure at states 0, 40 and 99 and once mid-play
(shots/e2-{state000,state040,state099,play}-{width}-{scheme}.png), the figure inside the walkthrough pop-out
(shots/e2-wt-{width}-{scheme}.png). Checks: the demo is armed and the controls shown, only one drawing shows, autoplay
starts only when the drawing is on screen, controls and readouts, overlaps over all 100 states (real text boxes),
the walkthrough (plays inside the pop-out, pauses when the section leaves or the pop-out opens over it, plays again),
reduced motion (no autoplay), no horizontal page scroll, and no console or page errors.
"""
import os
import sys
import tempfile

from playwright.sync_api import sync_playwright

PAGE_DIR = "/mnt/c/Users/David/dev/bca-shift-work/page"
PAGE = os.path.join(PAGE_DIR, "thrusterchain.html")
SHOTS = os.path.join(PAGE_DIR, "shots")
SHOOT = "--no-shots" not in sys.argv

frag = open(PAGE, encoding="utf-8").read()
tmp = os.path.join(tempfile.mkdtemp(prefix="e2-"), "page.html")
open(tmp, "w", encoding="utf-8").write(
    "<!doctype html>\n<html lang=\"en\">\n<head>\n<meta charset=\"utf-8\">\n"
    "<meta name=\"viewport\" content=\"width=device-width, initial-scale=1, viewport-fit=cover\">\n"
    "<meta name=\"color-scheme\" content=\"light\">\n<style>body{margin:0}</style>\n</head>\n<body>\n"
    + frag + "\n</body>\n</html>\n")
URL = "file://" + tmp

fails, errors = [], []


def check(cond, msg):
    print(("ok   " if cond else "FAIL ") + msg)
    if not cond:
        fails.append(msg)


SET_T = """t => { const s = document.getElementById('td-slider'); s.value = String(t);
  s.dispatchEvent(new Event('input', { bubbles: true })); }"""
STATE = "() => { const S = window.taskDemo.S; return { t: S.t, u: S.u, playing: S.playing, auto: S.auto, user: S.user }; }"
READ = "() => ['td-step','td-push','td-pushparts','td-thr','td-thrpush','td-ret'].map(i => document.getElementById(i).textContent)"
OVERFLOW = "() => Math.max(document.documentElement.scrollWidth, document.body.scrollWidth) - document.documentElement.clientWidth"
# real text boxes: every float circle against every visible text (except its own number) and every thruster arrow head
COLLIDE = """() => {
  const svg = [...document.querySelectorAll('#task-diagram svg.dg')].find(s => s.getBoundingClientRect().width > 0);
  const vis = e => e.style.display !== 'none' && getComputedStyle(e).display !== 'none';
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


def shot(pg, name):
    if SHOOT:
        path = os.path.join(SHOTS, name)
        pg.locator("#task-diagram").screenshot(path=path)
        print("     shot", path)


with sync_playwright() as p:
    br = p.chromium.launch()

    def open_page(width, scheme="light", motion="no-preference"):
        height = 844 if width < 700 else 800
        c = br.new_context(viewport={"width": width, "height": height}, color_scheme=scheme, reduced_motion=motion)
        pg = c.new_page()
        tag = f"[{width}/{scheme}/{motion}]"
        pg.on("console", lambda m: errors.append(f"{tag} console.{m.type}: {m.text}") if m.type in ("error", "warning") else None)
        pg.on("pageerror", lambda e: errors.append(f"{tag} pageerror: {e}"))
        pg.goto(URL, wait_until="load")
        pg.evaluate("document.fonts ? document.fonts.ready.then(() => true) : true")
        pg.wait_for_timeout(300)
        return c, pg

    for width in (390, 1200):
        for scheme in ("light", "dark"):
            tag = f"{width}-{scheme}"
            c, pg = open_page(width, scheme)
            print(f"--- {tag}")
            check(pg.evaluate("() => !!window.taskDemo && !document.getElementById('td-ctrls').hidden"), f"{tag}: demo armed, controls shown")
            shown = pg.evaluate("() => [...document.querySelectorAll('#task-diagram svg.dg')].filter(s => s.getBoundingClientRect().width > 0).map(s => s.getAttribute('class'))")
            check(shown == (["dg dg-narrow"] if width < 700 else ["dg dg-wide"]), f"{tag}: drawing on show {shown}")
            check(not pg.evaluate(STATE)["playing"], f"{tag}: not playing before the drawing is on screen")
            check(pg.evaluate(OVERFLOW) <= 0, f"{tag}: no horizontal page scroll (overflow {pg.evaluate(OVERFLOW)} px)")
            titles = pg.evaluate("() => [...document.querySelectorAll('#task-diagram svg.dg text')].map(t => t.textContent).filter(s => /^Thruster 1/.test(s))")
            check(len(titles) == 2, f"{tag}: bar titles name thruster 1: {titles}")
            # mid-play: autoplay once the drawing is on screen
            pg.evaluate("() => document.querySelector('#task-diagram svg.dg' + (innerWidth < 700 ? '.dg-narrow' : '.dg-wide')).scrollIntoView({ block: 'center' })")
            pg.wait_for_timeout(2600)
            st = pg.evaluate(STATE)
            check(st["playing"] and st["auto"] and 5 <= st["t"] <= 40, f"{tag}: autoplay on screen (state {st['t']}, u {st['u']:.2f})")
            shot(pg, f"e2-play-{tag}.png")
            st = pg.evaluate(STATE)
            print(f"     mid-play shot taken near state {st['t']}")
            for t in (0, 40, 99):
                pg.evaluate(SET_T, t)
                pg.wait_for_timeout(120)
                st = pg.evaluate(STATE)
                check(st["t"] == t and not st["playing"] and st["user"], f"{tag}: slider to state {t} pauses and shows it")
                print("     readouts", pg.evaluate(READ))
                shot(pg, f"e2-state{t:03d}-{tag}.png")
            r = pg.evaluate(READ)
            check(r[0] == "state 99 (last)" and r[5] == "−21.87", f"{tag}: last state reads {r[0]!r}, return {r[5]!r}")
            if scheme == "light":
                # controls
                pg.locator("#td-back").click()
                check(pg.evaluate(STATE)["t"] == 98, f"{tag}: Step back from 99 gives 98")
                pg.locator("#td-play").click()
                pg.wait_for_timeout(600)
                st = pg.evaluate(STATE)
                check(not st["playing"] and st["t"] == 99, f"{tag}: Play from 98 plays to the last state and stops")
                pg.locator("#td-play").click()
                pg.wait_for_timeout(500)
                st = pg.evaluate(STATE)
                check(st["playing"] and 1 <= st["t"] <= 6, f"{tag}: Play at the last state restarts from 0 (state {st['t']})")
                pg.locator("#td-play").click()
                check(not pg.evaluate(STATE)["playing"], f"{tag}: Pause")
                pg.locator("#td-fwd").click()
                # overlaps over all 100 states
                counts = {}
                for t in range(100):
                    pg.evaluate(SET_T, t)
                    for h in pg.evaluate(COLLIDE)["hits"]:
                        counts[h] = counts.get(h, 0) + 1
                check(not counts, f"{tag}: no overlaps over all 100 states {counts or ''}")
            c.close()

        # walkthrough: the section moves into the pop-out and back
        for scheme in ("light", "dark"):
            tag = f"{width}-{scheme}"
            c, pg = open_page(width, scheme)
            print(f"--- walkthrough {tag}")
            pg.click("#wt-open")
            pg.wait_for_timeout(400)
            for _ in range(2):
                pg.click("#wt-next")
                pg.wait_for_timeout(450)
            title = pg.evaluate("() => document.getElementById('wt-title').textContent")
            check(title == "The task", f"{tag}: pop-out shows {title!r}")
            check(pg.evaluate("() => document.getElementById('wt').contains(document.getElementById('task-diagram'))"), f"{tag}: the figure is inside the pop-out")
            pg.evaluate("() => document.querySelector('#task-diagram svg.dg' + (innerWidth < 700 ? '.dg-narrow' : '.dg-wide')).scrollIntoView({ block: 'center' })")
            pg.wait_for_timeout(1800)
            st = pg.evaluate(STATE)
            check(st["playing"] and st["auto"], f"{tag}: plays inside the pop-out (state {st['t']})")
            check(pg.evaluate(OVERFLOW) <= 0, f"{tag}: no horizontal page scroll with the pop-out open")
            fit = pg.evaluate("() => { const b = document.getElementById('wt-body').getBoundingClientRect(), f = document.getElementById('task-diagram').getBoundingClientRect(); return [Math.round(f.left - b.left), Math.round(b.right - f.right)]; }")
            check(fit[0] >= 0 and fit[1] >= 0, f"{tag}: the figure fits the pop-out's width {fit}")
            if SHOOT:
                path = os.path.join(SHOTS, f"e2-wt-{tag}.png")
                pg.screenshot(path=path)
                print("     shot", path)
            pg.click("#wt-next")
            pg.wait_for_timeout(450)
            st = pg.evaluate(STATE)
            check(not st["playing"], f"{tag}: pauses when the section leaves the pop-out")
            pg.wait_for_timeout(400)
            check(not pg.evaluate(STATE)["playing"], f"{tag}: stays paused while another section is shown")
            pg.click("#wt-prev")
            pg.wait_for_timeout(450)
            pg.evaluate("() => document.querySelector('#task-diagram svg.dg' + (innerWidth < 700 ? '.dg-narrow' : '.dg-wide')).scrollIntoView({ block: 'center' })")
            pg.wait_for_timeout(900)
            check(pg.evaluate(STATE)["playing"], f"{tag}: plays again when the section comes back")
            pg.keyboard.press("Escape")
            pg.wait_for_timeout(500)
            check(not pg.evaluate("() => document.getElementById('wt').open"), f"{tag}: Esc closes the pop-out")
            check(pg.evaluate("() => !document.getElementById('wt').contains(document.getElementById('task-diagram'))"), f"{tag}: the figure is back in the page")
            check(not pg.evaluate(STATE)["playing"], f"{tag}: not playing after closing (the page is back at its top)")
            c.close()
            # opening the pop-out (at its first section) over a playing drawing hides it, and closing shows it again
            c, pg = open_page(width, scheme)
            pg.evaluate("() => document.querySelector('#task-diagram svg.dg' + (innerWidth < 700 ? '.dg-narrow' : '.dg-wide')).scrollIntoView({ block: 'center' })")
            pg.wait_for_timeout(900)
            check(pg.evaluate(STATE)["playing"], f"{tag}: plays in the page when scrolled to")
            pg.evaluate("() => document.getElementById('wt-open').click()")
            pg.wait_for_timeout(600)
            t_now = pg.evaluate("() => document.getElementById('wt-title').textContent")
            check(t_now == "At a glance" and not pg.evaluate(STATE)["playing"], f"{tag}: pop-out opened at {t_now!r} over the drawing; the drawing behind it pauses")
            pg.keyboard.press("Escape")
            pg.wait_for_timeout(700)
            check(pg.evaluate(STATE)["playing"], f"{tag}: plays again once the pop-out closes over it")
            c.close()

    # reduced motion: no autoplay, Play steps through whole states
    c, pg = open_page(1200, "light", "reduce")
    pg.evaluate("() => document.getElementById('task-diagram').scrollIntoView({ block: 'center' })")
    pg.wait_for_timeout(1500)
    check(not pg.evaluate(STATE)["playing"], "reduced motion: no autoplay")
    pg.locator("#td-play").click()
    pg.wait_for_timeout(700)
    st = pg.evaluate(STATE)
    check(st["playing"] and st["t"] >= 2, f"reduced motion: Play works (state {st['t']})")
    pg.locator("#td-play").click()
    c.close()
    br.close()

print(f"[console errors/warnings and page errors] {len(errors)}")
for e in errors:
    print("   " + e[:300])
print("ALL CHECKS PASSED" if not fails and not errors else f"{len(fails)} FAILED")
sys.exit(1 if fails or errors else 0)

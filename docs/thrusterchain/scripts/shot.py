#!/usr/bin/env python3
"""Screenshot tool for the ThrusterChain page fragment.

Wraps the fragment (which starts at <title>) in a minimal publish-like skeleton
(doctype, meta charset, viewport-fit=cover, color-scheme meta, body margin 0),
loads it from a file URL in headless Chromium, optionally clicks selectors,
then takes one or more screenshots. Prints console errors, page errors, failed
requests and the horizontal page overflow in pixels.

Run with the venv python in WSL:
  /opt/shot-venv/bin/python /mnt/c/Users/David/dev/bca-shift-work/page/scripts/shot.py \
      --width 390 --scheme light \
      --shot base-top-390-light.png=@top --shot base-task-390-light.png=#task

--shot OUT=TARGET (repeatable). TARGET is a CSS selector, or one of
  @top       scroll to the top, then capture the viewport
  @viewport  capture the viewport as it is (after clicks)
  @full      capture the full page
OUT is a file name (saved into --out-dir) or an absolute path.
--split PX also saves tall selector shots as readable slices OUT-1.png, OUT-2.png ...
--click SEL (repeatable) clicks before the shots; --eval JS runs JS before the shots.
--reduced-motion, --data-theme light|dark, --anchor ID, --dpr N, --freeze, --outline: see --help.
Exit status 1 if a click, eval or shot failed.
"""
import argparse
import os
import sys
import tempfile
import time

from playwright.sync_api import sync_playwright, TimeoutError as PWTimeout

PAGE_DIR = "/mnt/c/Users/David/dev/bca-shift-work/page"
DEFAULT_PAGE = os.path.join(PAGE_DIR, "thrusterchain.html")
DEFAULT_OUT = os.path.join(PAGE_DIR, "shots")

OVERFLOW_JS = r"""
() => {
  const de = document.documentElement;
  const vw = de.clientWidth;
  const res = { scrollWidth: de.scrollWidth, bodyScrollWidth: document.body.scrollWidth,
                clientWidth: vw, innerWidth: window.innerWidth,
                overflow: Math.max(de.scrollWidth, document.body.scrollWidth) - vw, offenders: [] };
  if (res.overflow <= 0) return res;
  const clips = (el) => {
    for (let a = el.parentElement; a && a !== de; a = a.parentElement) {
      const ox = getComputedStyle(a).overflowX;
      if (ox !== 'visible') {
        const r = a.getBoundingClientRect();
        if (r.right <= vw + 0.5) return true;
      }
    }
    return false;
  };
  const off = [];
  for (const el of document.body.querySelectorAll('*')) {
    const r = el.getBoundingClientRect();
    if (r.width === 0 && r.height === 0) continue;
    if (r.right > vw + 0.5 && !clips(el)) off.push(el);
  }
  const set = new Set(off);
  for (const el of off) {
    if (el.parentElement && set.has(el.parentElement)) continue;  // keep outermost only
    const r = el.getBoundingClientRect();
    let sec = el.closest('section[id]');
    res.offenders.push({
      tag: el.tagName.toLowerCase(), id: el.id || '', cls: (el.className && el.className.baseVal === undefined) ? String(el.className).slice(0, 60) : '',
      section: sec ? sec.id : '', right: Math.round(r.right), width: Math.round(r.width),
      text: (el.textContent || '').trim().slice(0, 60)
    });
    if (res.offenders.length >= 12) break;
  }
  return res;
}
"""


def wrap(fragment, meta_scheme):
    return (
        "<!doctype html>\n<html lang=\"en\">\n<head>\n"
        "<meta charset=\"utf-8\">\n"
        "<meta name=\"viewport\" content=\"width=device-width, initial-scale=1, viewport-fit=cover\">\n"
        f"<meta name=\"color-scheme\" content=\"{meta_scheme}\">\n"
        "<style>body{margin:0}</style>\n"
        "</head>\n<body>\n" + fragment + "\n</body>\n</html>\n"
    )


def report_overflow(page, label):
    o = page.evaluate(OVERFLOW_JS)
    print(f"[overflow {label}] {o['overflow']} px (scrollWidth {o['scrollWidth']}, body {o['bodyScrollWidth']}, "
          f"clientWidth {o['clientWidth']}, innerWidth {o['innerWidth']})")
    for f in o["offenders"]:
        print(f"   offender <{f['tag']}{'#' + f['id'] if f['id'] else ''}"
              f"{'.' + f['cls'].replace(' ', '.') if f['cls'] else ''}> in #{f['section'] or '-'}: "
              f"right={f['right']} width={f['width']} text={f['text']!r}")
    return o["overflow"]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--page", default=DEFAULT_PAGE, help="page fragment to load (default: thrusterchain.html)")
    ap.add_argument("--out-dir", default=DEFAULT_OUT)
    ap.add_argument("--width", type=int, default=390)
    ap.add_argument("--height", type=int, default=0, help="viewport height (default 844 if width<700 else 800)")
    ap.add_argument("--scheme", choices=["light", "dark"], default="light", help="emulated prefers-color-scheme")
    ap.add_argument("--meta-scheme", default="light", help="content of the color-scheme meta (default light)")
    ap.add_argument("--data-theme", choices=["light", "dark"], help="set data-theme on <html> (page toggle)")
    ap.add_argument("--reduced-motion", action="store_true", help="emulate prefers-reduced-motion: reduce")
    ap.add_argument("--dpr", type=float, default=1, help="device scale factor")
    ap.add_argument("--anchor", default="", help="load with #anchor (without the #)")
    ap.add_argument("--click", action="append", default=[], help="CSS selector to click before shots (repeatable, in order)")
    ap.add_argument("--eval", action="append", default=[], help="JS expression to run before shots (repeatable)")
    ap.add_argument("--wait", type=int, default=500, help="ms to wait after load and after each click/eval")
    ap.add_argument("--freeze", action="store_true", help="screenshot with CSS animations disabled")
    ap.add_argument("--split", type=int, default=0, metavar="PX",
                    help="also save a tall selector shot as slices OUT-1.png, OUT-2.png ... of at most PX css px")
    ap.add_argument("--outline", action="store_true", help="print the page's top-level blocks (id, top, height)")
    ap.add_argument("--shot", action="append", default=[], metavar="OUT=TARGET",
                    help="OUT file = selector | @top | @viewport | @full (repeatable)")
    a = ap.parse_args()
    if not a.shot:
        a.shot = ["shot.png=@top"]
    height = a.height or (844 if a.width < 700 else 800)
    os.makedirs(a.out_dir, exist_ok=True)

    with open(a.page, encoding="utf-8") as fh:
        frag = fh.read()
    tmpdir = tempfile.mkdtemp(prefix="shot-")
    wrapped = os.path.join(tmpdir, "page.html")
    with open(wrapped, "w", encoding="utf-8") as fh:
        fh.write(wrap(frag, a.meta_scheme))
    url = "file://" + wrapped + (("#" + a.anchor) if a.anchor else "")

    console_errors, page_errors, failed = [], [], []
    status = 0
    with sync_playwright() as p:
        browser = p.chromium.launch()
        ctx = browser.new_context(viewport={"width": a.width, "height": height}, device_scale_factor=a.dpr,
                                  color_scheme=a.scheme,
                                  reduced_motion="reduce" if a.reduced_motion else "no-preference")
        page = ctx.new_page()
        page.on("console", lambda m: console_errors.append(f"{m.type}: {m.text}") if m.type in ("error", "warning") else None)
        page.on("pageerror", lambda e: page_errors.append(str(e)))
        page.on("requestfailed", lambda r: failed.append(f"{r.url[:100]} ({r.failure})"))
        t0 = time.time()
        page.goto(url, wait_until="load", timeout=60000)
        try:
            page.wait_for_load_state("networkidle", timeout=15000)
        except PWTimeout:
            print("[warn] network did not go idle within 15 s")
        page.evaluate("document.fonts ? document.fonts.ready.then(() => true) : true")
        if a.data_theme:
            page.evaluate(f"document.documentElement.setAttribute('data-theme', '{a.data_theme}')")
        page.wait_for_timeout(a.wait)
        print(f"[load] {a.width}x{height} scheme={a.scheme} reduced_motion={a.reduced_motion} in {time.time() - t0:.1f} s")
        report_overflow(page, "after load")
        if a.outline:
            rows = page.evaluate(
                "() => Array.from(document.querySelectorAll('main > *, body > *:not(main):not(style):not(script):not(link):not(title)'))"
                ".map(e => { const r = e.getBoundingClientRect(); return [e.tagName.toLowerCase(), e.id || '',"
                " String(e.className || '').slice(0, 30), Math.round(r.top + scrollY), Math.round(r.height)]; })")
            print("[outline] tag id class top height")
            for t, i, c, top, h in rows:
                print(f"   {t:8s} {('#' + i) if i else '-':24s} {c:30s} {top:7d} {h:6d}")

        for sel in a.click:
            try:
                page.click(sel, timeout=5000)
                print(f"[click] {sel}")
            except Exception as e:  # noqa: BLE001
                print(f"[click FAILED] {sel}: {str(e).splitlines()[0]}")
                status = 1
            page.wait_for_timeout(a.wait)
        for js in a.eval:
            try:
                r = page.evaluate(js)
                print(f"[eval] {js[:60]} -> {str(r)[:120]}")
            except Exception as e:  # noqa: BLE001
                print(f"[eval FAILED] {js[:60]}: {str(e).splitlines()[0]}")
                status = 1
            page.wait_for_timeout(a.wait)
        if a.click or a.eval:
            report_overflow(page, "after actions")

        anim = "disabled" if a.freeze else "allow"
        for spec in a.shot:
            if "=" not in spec:
                print(f"[shot FAILED] bad spec {spec!r}; use OUT=TARGET")
                status = 1
                continue
            out, target = spec.split("=", 1)
            path = out if os.path.isabs(out) else os.path.join(a.out_dir, out)
            try:
                if target == "@top":
                    page.evaluate("window.scrollTo(0, 0)")
                    page.wait_for_timeout(150)
                    page.screenshot(path=path, animations=anim)
                elif target == "@viewport":
                    page.screenshot(path=path, animations=anim)
                elif target == "@full":
                    page.screenshot(path=path, full_page=True, animations=anim)
                else:
                    loc = page.locator(target)
                    n = loc.count()
                    if n == 0:
                        print(f"[shot FAILED] {target}: no element matches")
                        status = 1
                        continue
                    if n > 1:
                        print(f"[shot note] {target}: {n} matches, using the first")
                    loc.first.screenshot(path=path, animations=anim, timeout=20000)
                    box = loc.first.evaluate(
                        "e => { const r = e.getBoundingClientRect();"
                        " return {x: r.left + scrollX, y: r.top + scrollY, width: r.width, height: r.height}; }")
                    print(f"[shot] {path}  ({target}, {round(box['width'])}x{round(box['height'])} css px)")
                    if a.split and box["height"] > a.split:
                        stem, ext = os.path.splitext(path)
                        y, k = box["y"], 1
                        while y < box["y"] + box["height"] - 1:
                            h = min(a.split, box["y"] + box["height"] - y)
                            sp = f"{stem}-{k}{ext}"
                            page.screenshot(path=sp, full_page=True, animations=anim,
                                            clip={"x": box["x"], "y": y, "width": box["width"], "height": h})
                            print(f"   slice {sp}  (y {round(y - box['y'])}..{round(y - box['y'] + h)})")
                            y += h
                            k += 1
                    continue
                print(f"[shot] {path}  ({target})")
            except Exception as e:  # noqa: BLE001
                print(f"[shot FAILED] {target}: {str(e).splitlines()[0]}")
                status = 1

        print(f"[console errors/warnings] {len(console_errors)}")
        for m in console_errors:
            print("   " + m[:300])
        print(f"[page errors] {len(page_errors)}")
        for m in page_errors:
            print("   " + m[:300])
        print(f"[failed requests] {len(failed)}")
        for m in failed:
            print("   " + m)
        browser.close()
    return status


if __name__ == "__main__":
    sys.exit(main())

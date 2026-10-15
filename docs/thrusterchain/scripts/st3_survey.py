"""Read-only survey for the structure plan (comments A, E, F).

Usage:
  st3_survey.py blocks                 top-level blocks of <main>: id, char span, heading
  st3_survey.py words ID [ID ...]      positional words ("above", "below", ...) in the visible text of the blocks
  st3_survey.py text ID [maxchars]     the visible text of one block (math shown as [M:tex])
  st3_survey.py find REGEX [w]         regex over the raw page (scripts cut), with context, math collapsed
"""
import re
import sys
import html as H

P = "/mnt/c/Users/David/dev/bca-shift-work/page/thrusterchain.html"
src = open(P, encoding="utf-8").read()
cut = src.find("\n<script>")
body = src[:cut]


def strip_mjx(s):
    return re.sub(r"<mjx-container[^>]*?aria-label=\"([^\"]*)\".*?</mjx-container>",
                  lambda m: "[M:" + m.group(1) + "]", s, flags=re.S)


def visible(s):
    s = strip_mjx(s)
    s = re.sub(r"<svg\b.*?</svg>", " [svg] ", s, flags=re.S)
    s = re.sub(r"<[^>]+>", "", s)
    s = H.unescape(s)
    return re.sub(r"\s+", " ", s).strip()


def blocks():
    m0 = body.find("<main>")
    m1 = body.find("</main>")
    out = []
    pat = re.compile(r"\n(  <header class=\"head\" id=\"top\">|  <section id=\"([^\"]+)\">|    <figure class=\"panel\" id=\"sim\")")
    starts = [(m.start() + 1, m.group(2) or ("top" if "header" in m.group(1) else "sim")) for m in pat.finditer(body, m0, m1)]
    for i, (s, bid) in enumerate(starts):
        e = starts[i + 1][0] if i + 1 < len(starts) else m1
        out.append((bid, s, e))
    return out


mode = sys.argv[1]
B = {b[0]: b for b in blocks()}
if mode == "blocks":
    for bid, s, e in blocks():
        seg = body[s:e]
        h = re.search(r"<h2[^>]*>(.*?)</h2>|<div class=\"panel-title\">(.*?)</div>|<h1>(.*?)</h1>", seg, re.S)
        ht = visible(next(g for g in h.groups() if g)) if h else ""
        print(f"{bid:22s} {s:>8d} {e:>8d} {e - s:>8d}  {ht}")
elif mode == "words":
    W = re.compile(r"\b(above|below|earlier|later|next|previous|preceding|following|comes|came|first|before this|after this|"
                   r"so far|again|already|as seen|we saw|was shown|is shown|this page's|at the top|further down|up the page|down the page|"
                   r"the live panel|the task|watch one episode|demo|drawing)\b", re.I)
    for bid in sys.argv[2:]:
        _, s, e = B[bid]
        t = visible(body[s:e])
        print(f"===== {bid} ({len(t)} chars) =====")
        for m in W.finditer(t):
            a = max(0, m.start() - 160)
            z = min(len(t), m.end() + 160)
            print(f"  [{m.group(0)}] ...{t[a:z]}...")
elif mode == "text":
    _, s, e = B[sys.argv[2]]
    t = visible(body[s:e])
    n = int(sys.argv[3]) if len(sys.argv) > 3 else 100000
    print(t[:n])
elif mode == "find":
    pat = re.compile(sys.argv[2])
    w = int(sys.argv[3]) if len(sys.argv) > 3 else 200
    for m in pat.finditer(body):
        bid = next((b for b, s, e in blocks() if s <= m.start() < e), "outside-main")
        ln = body.count("\n", 0, m.start()) + 1
        print(f"--- line {ln} char {m.start()} in {bid}")
        print(strip_mjx(body[max(0, m.start() - w): m.end() + w]))

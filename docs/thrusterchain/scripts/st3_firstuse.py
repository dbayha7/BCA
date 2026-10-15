"""Read-only: for each tooltip title (and some plain phrases), the block of its first use in the old and new order."""
import re, sys, html as H
P = "/mnt/c/Users/David/dev/bca-shift-work/page/thrusterchain.html"
src = open(P, encoding="utf-8").read()
body = src[:src.find("\n<script>")]
m0 = body.find("<main>"); m1 = body.find("</main>")
pat = re.compile(r"\n(  <header class=\"head\" id=\"top\">|  <section id=\"([^\"]+)\">|    <figure class=\"panel\" id=\"sim\")")
starts = [(m.start() + 1, m.group(2) or ("top" if "header" in m.group(1) else "sim")) for m in pat.finditer(body, m0, m1)]
blocks = {}
for i, (s, bid) in enumerate(starts):
    e = starts[i + 1][0] if i + 1 < len(starts) else m1
    blocks[bid] = body[s:e]
old = [b for _, b in starts]
six = ["action-space", "observation-space", "starting-state", "dynamics", "rewards", "episode-end"]
new = ["top"] + six + ["sim", "task"] + [b for b in old if b not in six + ["top", "sim", "task", "version-history"]]
def vis(s):
    s = re.sub(r"<mjx-container[^>]*?aria-label=\"([^\"]*)\".*?</mjx-container>", lambda m: "[M:" + m.group(1) + "]", s, flags=re.S)
    s = re.sub(r"<svg\b.*?</svg>", " ", s, flags=re.S)
    s = re.sub(r"\sdata-tip[a-z-]*=\"[^\"]*\"", "", s)
    return re.sub(r"\s+", " ", H.unescape(re.sub(r"<[^>]+>", " ", s)))
VIS = {b: vis(blocks[b]).lower() for b in blocks}
def first(order, phrase):
    for b in order:
        if phrase in VIS[b]:
            return b
    return None
titles = []
for b in old:
    for m in re.finditer(r"data-tip-title=\"([^\"]+)\"", blocks[b]):
        t = H.unescape(m.group(1))
        if t not in titles: titles.append(t)
extra = sys.argv[1:]
print("phrase | first block, old order | first block, new order")
for t in titles + extra:
    key = re.sub(r",.*$", "", t).strip().lower()
    if len(key) < 4: continue
    o, n = first(old, key), first(new, key)
    if o != n and (o in ("sim", "task") or n in six):
        print(f"{t!r:60s} old={o:18s} new={n}")

"""Read-only: every in-page link between the moved blocks (the six, sim, task, description) with its sentence."""
import re, html as H
P = "/mnt/c/Users/David/dev/bca-shift-work/page/thrusterchain.html"
src = open(P, encoding="utf-8").read()
body = src[:src.find("\n<script>")]
m0 = body.find("<main>"); m1 = body.find("</main>")
pat = re.compile(r"\n(  <header class=\"head\" id=\"top\">|  <section id=\"([^\"]+)\">|    <figure class=\"panel\" id=\"sim\")")
starts = [(m.start() + 1, m.group(2) or ("top" if "header" in m.group(1) else "sim")) for m in pat.finditer(body, m0, m1)]
def blk(i):
    b = None
    for s, bid in starts:
        if s <= i: b = bid
    return b
ids = {m.group(1): blk(m.start()) for m in re.finditer(r"\bid=\"([^\"]+)\"", body) if m0 <= m.start() < m1}
six = {"action-space", "observation-space", "starting-state", "dynamics", "rewards", "episode-end"}
grp = lambda b: "six" if b in six else b
def vis(s):
    s = re.sub(r"<mjx-container[^>]*?aria-label=\"([^\"]*)\".*?</mjx-container>", lambda m: "[M:" + m.group(1) + "]", s, flags=re.S)
    s = re.sub(r"\sdata-tip[a-z-]*=\"[^\"]*\"", "", s)
    return re.sub(r"\s+", " ", H.unescape(re.sub(r"<[^>]+>", "", s)))
for m in re.finditer(r"<a href=\"#([^\"]+)\"", body[m0:m1]):
    i = m.start() + m0
    src_b, tgt_b = blk(i), ids.get(m.group(1))
    pair = {grp(src_b), grp(tgt_b)}
    if src_b == tgt_b or not (("six" in pair and (pair & {"sim", "task", "top"})) or (src_b == "top")):
        continue
    ctx = vis(body[max(0, i - 500): i + 200])
    print(f"{src_b} -> #{m.group(1)} ({tgt_b}) line {body.count(chr(10), 0, i) + 1}: ...{ctx[-330:]}")

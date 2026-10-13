"""Read-only: the first use of each phrase within a given block, in visible text, with context."""
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
def vis(s):
    s = re.sub(r"<mjx-container[^>]*?aria-label=\"([^\"]*)\".*?</mjx-container>", lambda m: "[M:" + m.group(1) + "]", s, flags=re.S)
    s = re.sub(r"<svg\b.*?</svg>", " ", s, flags=re.S)
    s = re.sub(r"\sdata-tip[a-z-]*=\"[^\"]*\"", "", s)
    return re.sub(r"\s+", " ", H.unescape(re.sub(r"<[^>]+>", "", s)))
w = int(sys.argv[1])
for arg in sys.argv[2:]:
    b, ph = arg.split(":", 1)
    t = vis(blocks[b]); i = t.lower().find(ph.lower())
    print(f"== {b} / {ph}: ...{t[max(0, i - w): i + w]}...")

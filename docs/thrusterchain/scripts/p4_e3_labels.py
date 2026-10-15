"""Read-only: list every math container's TeX label on the page (display flag, count, first section)."""
import collections, html, os, re
PD = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
s = open(os.path.join(PD, "thrusterchain.html"), encoding="utf-8").read()
CONT = re.compile(r'<mjx-container role="img" aria-label="(.*?)" class="MathJax" jax="SVG"( display="true")?>', re.S)
secs = [(m.start(), m.group(1)) for m in re.finditer(r'<section id="([^"]+)"', s)]
def sec(i):
    r = None
    for a, n in secs:
        if a <= i: r = n
    return r
c = collections.OrderedDict()
for m in CONT.finditer(s):
    k = (html.unescape(m.group(1)), bool(m.group(2)))
    if k not in c: c[k] = [0, sec(m.start())]
    c[k][0] += 1
out = os.path.join(PD, "tmp", "p4", "edit3", "labels.txt")
with open(out, "w", encoding="utf-8") as f:
    for (t, d), (n, sc) in c.items():
        f.write(f"{'D' if d else 'I'}\t{n}\t{sc}\t{t}\n")
print(len(c), "distinct;", sum(v[0] for v in c.values()), "containers")

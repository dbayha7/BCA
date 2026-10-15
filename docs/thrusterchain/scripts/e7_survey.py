"""Survey: each display formula, its section, and the nearest legend (.symkey or other) after and before it."""
import re, html
P = "/mnt/c/Users/David/dev/bca-shift-work/page/thrusterchain.html"
s = open(P, encoding="utf-8").read()
end = s.index("<script")
secs = [(m.start(), m.group(1)) for m in re.finditer(r'<section id="([^"]+)"', s[:end])]
def sec_at(p):
    cur = "-"
    for a, i in secs:
        if a <= p: cur = i
    return cur
disp = [m.start() for m in re.finditer(r'<div class="math-display', s[:end])]
keys = [(m.start(), re.search(r'<p class="symkey-h">(.*?)</p>', s[m.start():m.start()+400]).group(1)) for m in re.finditer(r'<div class="symkey"', s[:end])]
print(len(disp), "display formulas;", len(keys), "symkeys")
for i, p in enumerate(disp):
    j = s.index("</div>", p)
    lab = re.search(r'aria-label="([^"]*)"', s[p:j+5000])
    lab = html.unescape(lab.group(1))[:70] if lab else "?"
    nxt = [k for k in keys if k[0] > p]
    prv = [k for k in keys if k[0] < p]
    n = nxt[0] if nxt else None
    nd = (n[0] - p) if n else None
    # other display formulas between this and the next symkey
    between = len([d for d in disp if p < d < (n[0] if n else 10**9)])
    print(f"{i:2d} {sec_at(p):22s} next+{nd} ({sec_at(n[0]) if n else '-'}: {n[1] if n else '-'}) after-me-before-it={between} prev-{p-prv[-1][0] if prv else None} | {lab}")

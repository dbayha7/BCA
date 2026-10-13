"""Read-only: every '*' (and star-like character) outside math containers, styles and script code: in text nodes,
in attribute values, and in script string literals. Complements nb_scan.py, whose pattern needs a letter before the star.
  wsl.exe -d Ubuntu-22.04 -u root -e bash -lc 'python3 /mnt/c/Users/David/dev/bca-shift-work/page/scripts/nb_anystar.py'
"""
import html as H
import re
import sys

sys.path.insert(0, '/mnt/c/Users/David/dev/bca-shift-work/page/scripts')
s = open('/mnt/c/Users/David/dev/bca-shift-work/page/thrusterchain.html', encoding='utf-8').read()
MJX = re.compile(r'<mjx-container role="img" aria-label="(.*?)" class="MathJax" jax="SVG"( display="true")?>.*?</mjx-container>', re.S)
spans = [(m.start(), m.end()) for m in MJX.finditer(s)]
spans += [(m.start(), m.end()) for m in re.finditer(r'<style\b[^>]*>.*?</style>', s, re.S)]
scripts = [(m.start(1), m.end(1)) for m in re.finditer(r'<script\b[^>]*>(.*?)</script>', s, re.S)]
STARCH = re.compile(r'[*⋆∗★☆⁎✱]')


def inside(p, sp):
    return any(a <= p < b for a, b in sp)


def ctx(p, w=160):
    t = s[max(0, p - w):p + w]
    t = MJX.sub(lambda m: '{MATH:' + m.group(1) + '}', t)
    t = re.sub(r'<svg\b.*?</svg>', '{SVG}', t, flags=re.S)
    return t.replace('\n', ' ')


n = 0
for m in STARCH.finditer(s):
    p = m.start()
    if inside(p, spans):
        continue
    if inside(p, scripts):
        # keep only stars inside string literals: crude test, a quote char opened earlier on the same line segment
        a = max(s.rfind('\n', 0, p), p - 400)
        seg = s[a:p]
        if seg.count('"') % 2 == 0 and seg.count("'") % 2 == 0 and seg.count('`') % 2 == 0:
            continue
        kind = 'script-string'
    else:
        lt, gt = s.rfind('<', 0, p), s.rfind('>', 0, p)
        kind = 'attribute' if lt > gt else 'text'
    n += 1
    print(n, kind, p, '|', ctx(p))
print('total', n)

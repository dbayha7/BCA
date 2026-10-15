"""Read-only: every star-like character on the page outside CSS and SVG paths, classified by where it sits.
  python3 nb2_stars.py [page]   (default: the working page)
Prints math labels holding a star (\\ast, \\star, ^*, *), visible-text and attribute stars, and script stars that sit
inside string literals (operators in code are counted, not listed)."""
import html as H
import re
import sys

p = sys.argv[1] if len(sys.argv) > 1 else '/mnt/c/Users/David/dev/bca-shift-work/page/thrusterchain.html'
s = open(p, encoding='utf-8').read()
STARS = '*∗⋆★✱✶✳﹡＊'
STAR_CLASS = '[' + re.escape(STARS) + ']'
MJX = re.compile(r'<mjx-container role="img" aria-label="(.*?)" class="MathJax" jax="SVG"( display="true")?>.*?</mjx-container>', re.S)
labels = [H.unescape(m.group(1)) for m in MJX.finditer(s)]
star_labels = [l for l in labels if re.search(r'\\ast|\\star|' + STAR_CLASS, l)]
print(f'math containers: {len(labels)}; with a star in the TeX: {len(star_labels)}')
for l in star_labels:
    print('   ', l)
b = MJX.sub(lambda m: ' ' * (m.end() - m.start()), s)
b = re.sub(r'<style\b.*?</style>|<svg\b.*?</svg>', lambda m: ' ' * (m.end() - m.start()), b, flags=re.S)
scripts = [(m.start(), m.end()) for m in re.finditer(r'<script\b.*?</script>', b, re.S)]
outside = list(b)
for a, e in scripts:
    outside[a:e] = ' ' * (e - a)
outside = ''.join(outside)
hits = [m.start() for m in re.finditer(STAR_CLASS, outside)]
print(f'stars in markup text and attributes (outside scripts, styles, svg, math): {len(hits)}')
for q in hits:
    print('   ', re.sub(r'\s+', ' ', outside[max(0, q - 90):q + 60]))
n_code, in_str = 0, []
BS = '\\'
STR = re.compile(
    '"(?:[^"' + BS + BS + '\\n]|' + BS + BS + '.)*"'
    + "|'(?:[^'" + BS + BS + "\\n]|" + BS + BS + ".)*'"
    + '|`(?:[^`' + BS + BS + ']|' + BS + BS + '.)*`'
    + '|/' + BS + '*.*?' + BS + '*/|//[^\\n]*', re.S)
for a, e in scripts:
    code = b[a:e]
    for m in STR.finditer(code):
        t = m.group(0)
        if t.startswith('/*') or t.startswith('//'):
            continue
        if re.search(STAR_CLASS, t):
            in_str.append(t[:200])
    stripped = STR.sub(lambda m: ' ' * len(m.group(0)), code)
    n_code += len(re.findall(STAR_CLASS, stripped))
print(f'stars used as operators in script code: {n_code}; inside script string literals: {len(in_str)}')
for t in in_str:
    print('   ', t)

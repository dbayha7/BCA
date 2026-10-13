"""Read-only: print raw HTML around page positions or around each match of a pattern, math containers shortened.
  python3 nb_show.py pos 1473347 [width]      |  python3 nb_show.py re "<sup>\*</sup>" [width]
"""
import re
import sys

s = open('/mnt/c/Users/David/dev/bca-shift-work/page/thrusterchain.html', encoding='utf-8').read()
MJX = re.compile(r'<mjx-container role="img" aria-label="(.*?)" class="MathJax" jax="SVG"( display="true")?>.*?</mjx-container>', re.S)


def short(h):
    h = MJX.sub(lambda m: '{MATH:' + m.group(1) + '}', h)
    return re.sub(r'<svg\b.*?</svg>', '{SVG}', h, flags=re.S)


mode, arg = sys.argv[1], sys.argv[2]
w = int(sys.argv[3]) if len(sys.argv) > 3 else 1500
if mode == 'pos':
    p = int(arg)
    print(short(s[max(0, p - w):p + w]))
else:
    for m in re.finditer(arg, s):
        print('== at', m.start())
        print(short(s[max(0, m.start() - w):m.end() + w]))

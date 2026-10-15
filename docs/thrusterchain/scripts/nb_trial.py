"""Trial run of the comment-B math swaps on a COPY of the page (tmp/nb/trial.html); the working page is only read.
Checks the renderer + mjx.py flow end to end and gives a file to screenshot.
  wsl.exe -d Ubuntu-22.04 -u root -e bash -lc 'python3 /mnt/c/Users/David/dev/bca-shift-work/page/scripts/nb_trial.py'
"""
import hashlib
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import mjx  # noqa: E402

PAGE = '/mnt/c/Users/David/dev/bca-shift-work/page/thrusterchain.html'
V11 = '/mnt/c/Users/David/dev/bca-shift-work/page/thrusterchain.v11.html'
OUT_DIR = '/mnt/c/Users/David/dev/bca-shift-work/page/tmp/nb'
md5 = lambda p: hashlib.md5(open(p, 'rb').read()).hexdigest()
before_page, before_v11 = md5(PAGE), md5(V11)

s = open(PAGE, encoding='utf-8').read()
n0 = s.count('<mjx-container')
jobs = json.load(open(f'{mjx.MATHBUILD}/jobs/notation_b.json', encoding='utf-8'))
R = mjx.load('notation_b')
for j in jobs:
    s = mjx.swap(s, j['old'], R[j['key']], count=j['count'])
assert s.count('<mjx-container') == n0, 'math count changed'
assert '^*' not in ''.join(mjx.label(m.group(0)) for m in mjx.CONTAINER.finditer(s)), 'a starred label is left'
print('glyph ids, all unique:', mjx.check(s))
print('new prefix numbers used: MJXC-1 ..', mjx.next_number(s) - 1)
os.makedirs(OUT_DIR, exist_ok=True)
open(f'{OUT_DIR}/trial.html', 'w', encoding='utf-8').write(s)
assert md5(PAGE) == before_page and md5(V11) == before_v11, 'page or backup changed'
print('wrote', f'{OUT_DIR}/trial.html; working page and v11 backup unchanged')

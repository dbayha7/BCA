"""Collapsed context around every display formula (or a given pattern list)."""
import re, sys
P = '/mnt/c/Users/David/dev/bca-shift-work/page/thrusterchain.html'
s = open(P, encoding='utf-8').read()
def coll(m):
    tag = m.group(0)[:m.group(0).index('>') + 1]
    lab = re.search(r'aria-label="([^"]*)"', tag)
    return '[[' + ('D ' if 'display="true"' in tag else '') + '$' + (lab.group(1) if lab else '?') + '$]]'
c = re.sub(r'<mjx-container.*?</mjx-container>', coll, s, flags=re.S)
c = re.sub(r'<svg\b.*?</svg>', lambda m: '<svg…' + (re.search(r'aria-label="([^"]*)"', m.group(0)[:600]) or re.search('x','x')).group(0)[:80] + '…/svg>', c, flags=re.S)
c = c[:c.index('<script')]
before = int(sys.argv[1]) if len(sys.argv) > 1 else 250
after = int(sys.argv[2]) if len(sys.argv) > 2 else 700
last = -1
for m in re.finditer(r'<div class="math-display">', c):
    if m.start() < last: continue
    a = max(0, m.start() - before); b = m.start() + after
    last = b
    print('=' * 30, m.start()); print(c[a:b])

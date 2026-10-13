"""View raw HTML of the page around a pattern, with MathJax containers collapsed to [[$label$]] and svg paths cut.
usage: e6_view.py PATTERN [CHARS_AFTER] [CHARS_BEFORE] [OCCURRENCE] [PAGE]"""
import re, sys
P = sys.argv[5] if len(sys.argv) > 5 else '/mnt/c/Users/David/dev/bca-shift-work/page/thrusterchain.html'
s = open(P, encoding='utf-8').read()
def coll(m):
    tag = m.group(0)[:m.group(0).index('>') + 1]
    lab = re.search(r'aria-label="([^"]*)"', tag)
    disp = 'display="true"' in tag
    return '[[' + ('DISPLAY ' if disp else '') + '$' + (lab.group(1) if lab else '?') + '$]]'
s = re.sub(r'<mjx-container.*?</mjx-container>', coll, s, flags=re.S)
s = re.sub(r'<path [^>]*>', '<path…>', s)
s = re.sub(r'd="[^"]{80,}"', 'd="…"', s)
pat = sys.argv[1]; after = int(sys.argv[2]) if len(sys.argv) > 2 else 3000
before = int(sys.argv[3]) if len(sys.argv) > 3 else 0
occ = int(sys.argv[4]) if len(sys.argv) > 4 else 0
ms = [m.start() for m in re.finditer(pat, s)]
print('matches:', len(ms), ms[:20])
if not ms: sys.exit()
p = ms[occ]
print(s[max(0, p - before): p + after])

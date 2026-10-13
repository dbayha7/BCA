"""Read-only: find a regex in the page (or a given file); print section, position and context. Usage: p3f_grep.py REGEX [WIDTH] [FILE]"""
import re, sys, bisect
P = '/mnt/c/Users/David/dev/bca-shift-work/page/'
pat = sys.argv[1]; w = int(sys.argv[2]) if len(sys.argv) > 2 else 80
f = sys.argv[3] if len(sys.argv) > 3 else P + 'thrusterchain.html'
s = open(f, encoding='utf-8').read()
secs = [(m.start(), m.group(1)) for m in re.finditer(r'<section id="([^"]+)"', s)]
starts = [a for a, _ in secs]
n = 0
for m in re.finditer(pat, s):
    i = bisect.bisect_right(starts, m.start()) - 1
    sec = secs[i][1] if i >= 0 else '-'
    ctx = s[max(0, m.start() - w):m.end() + w].replace('\n', ' ')
    ctx = re.sub(r'<mjx-container[^>]*aria-label="([^"]*)".*?(</mjx-container>|$)', r'[MJX:\1]', ctx)
    print(f'{sec:22s} {m.start():8d} | {ctx}')
    n += 1
print('matches:', n)

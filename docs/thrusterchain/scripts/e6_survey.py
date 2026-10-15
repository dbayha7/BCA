"""List display formulas, diagrams/charts and inline math labels with their section."""
import re, collections
P = '/mnt/c/Users/David/dev/bca-shift-work/page/thrusterchain.html'
s = open(P, encoding='utf-8').read()
body_end = s.index('<script')
secs = [(m.start(), m.group(1)) for m in re.finditer(r'<section[^>]*id="([^"]+)"', s)]
def sec(p):
    c = '-'
    for q, i in secs:
        if q < p: c = i
    return c
def strip(x):
    x = re.sub(r'<mjx-container[^>]*aria-label="([^"]*)".*?</mjx-container>', r'[$\1$]', x, flags=re.S)
    x = re.sub(r'<[^>]+>', '', x)
    return re.sub(r'\s+', ' ', x)
print('== DISPLAY')
for m in re.finditer(r'<mjx-container[^>]*display="true"[^>]*>', s[:body_end]):
    lab = re.search(r'aria-label="([^"]*)"', m.group(0)).group(1)
    end = s.index('</mjx-container>', m.end())
    # parent tag: look back for last opening tag
    pre = s[max(0, m.start()-300):m.start()]
    lastopen = re.findall(r'<(div|p|figure|span)([^>]*)>', pre)
    print(f'{m.start()} [{sec(m.start())}] parent={lastopen[-1] if lastopen else None} :: {lab[:150]}')
print('== FIGURES')
for m in re.finditer(r'<(svg|canvas|figure)\b([^>]*)>', s[:body_end]):
    a = m.group(2)
    if m.group(1) == 'svg' and ('mjx' in s[max(0,m.start()-200):m.start()] or 'focusable' in a and 'role="img"' not in a and 'class=' not in a):
        continue
    cls = re.search(r'class="([^"]*)"', a); idd = re.search(r'id="([^"]*)"', a); al = re.search(r'aria-label="([^"]*)"', a)
    print(f'{m.start()} [{sec(m.start())}] <{m.group(1)} id={idd and idd.group(1)} class={cls and cls.group(1)}> {al and al.group(1)[:90]}')
print('== INLINE LABELS')
cnt = collections.Counter(); where = collections.defaultdict(set)
for m in re.finditer(r'<mjx-container(?![^>]*display="true")[^>]*aria-label="([^"]*)"', s[:body_end]):
    cnt[m.group(1)] += 1; where[m.group(1)].add(sec(m.start()))
for k, v in sorted(cnt.items(), key=lambda kv: -kv[1]):
    print(f'{v:3d} {k!r} {sorted(where[k])}')

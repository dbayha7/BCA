import re, sys
P='/mnt/c/Users/David/dev/bca-shift-work/page/thrusterchain.html'
s=open(P,encoding='utf-8').read()
print(len(s), s.count('\n'))
print('tables', s.count('<table'))
print('scripts', [m.start() for m in re.finditer(r'<script', s)])
print('style', [m.start() for m in re.finditer(r'<style', s)], [m.start() for m in re.finditer(r'</style>', s)])
# per table: enclosing section id, class, caption, headers
secs=[(m.start(), m.group(1)) for m in re.finditer(r'<section id="([^"]+)"', s)]
def sec_of(pos):
    cur=''
    for p,i in secs:
        if p<pos: cur=i
    return cur
for k,m in enumerate(re.finditer(r'<table[^>]*>', s)):
    end=s.index('</table>', m.start())
    t=s[m.start():end]
    ths=re.findall(r'<th[^>]*>(.*?)</th>', t, re.S)
    ths=[re.sub(r'<[^>]+>','',h).strip()[:40] for h in ths]
    cap=re.search(r'<caption[^>]*>(.*?)</caption>', t, re.S)
    rows=t.count('<tr')
    # find nearest preceding id within 3000 chars
    pre=s[max(0,m.start()-1500):m.start()]
    ids=re.findall(r'id="([^"]+)"', pre)
    print(f'#{k} pos={m.start()} sec={sec_of(m.start())} tag={m.group(0)[:80]!r} rows={rows} nearid={ids[-1] if ids else ""}')
    print('    cap:', re.sub(r'<[^>]+>','',cap.group(1)).strip()[:100] if cap else None)
    print('    th:', ths[:30])

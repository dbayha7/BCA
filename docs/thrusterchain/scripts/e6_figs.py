import re
P = '/mnt/c/Users/David/dev/bca-shift-work/page/thrusterchain.html'
s = open(P, encoding='utf-8').read()
s = s[:s.index('<script')]
def coll(m):
    tag = m.group(0)[:m.group(0).index('>') + 1]
    lab = re.search(r'aria-label="([^"]*)"', tag)
    return '[[' + ('D ' if 'display="true"' in tag else '') + '$' + (lab.group(1) if lab else '?') + '$]]'
s = re.sub(r'<mjx-container.*?</mjx-container>', coll, s, flags=re.S)
# svg.chart / svg.dg: show tag + count of titles + texts, then the html after the svg until 1500 chars
for m in re.finditer(r'<svg class="(chart|dg)[^"]*"[^>]*>', s):
    end = s.index('</svg>', m.end())
    inner = s[m.end():end]
    titles = re.findall(r'<title>(.*?)</title>', inner)
    texts = re.findall(r'<text[^>]*>(.*?)</text>', inner)
    print('=' * 20, m.start(), m.group(0)[:160])
    print('  titles', len(titles), titles[:2])
    print('  texts', texts[:40])
    print('  BEFORE:', s[max(0, m.start() - 400):m.start()].replace('\n', ' ')[-400:])
    print('  AFTER:', s[end:end + 1400].replace('\n', ' '))

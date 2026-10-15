"""Read-only: readable dumps of the current page for the pass-3 final fixes (tmp/p3f/).
text.txt: whole page as text (math as $tex$, tips inline). sec_<id>.html: section HTML with MathJax collapsed."""
import re, sys, html
from html.parser import HTMLParser
P = '/mnt/c/Users/David/dev/bca-shift-work/page/'
src = sys.argv[1] if len(sys.argv) > 1 else P + 'thrusterchain.html'
tag = sys.argv[2] if len(sys.argv) > 2 else ''
s0 = open(src, encoding='utf-8').read()
def collapse(t):
    def rep(m):
        lab = re.search(r'aria-label="([^"]*)"', m.group(0)).group(1)
        disp = ' display' if 'display="true"' in m.group(0)[:300] else ''
        return '[MJX%s:%s]' % (disp, html.unescape(lab))
    return re.sub(r'<mjx-container[^>]*>.*?</mjx-container>', rep, t, flags=re.S)
ids = re.findall(r'<section id="([^"]+)"', s0)
print(ids)
for i, sid in enumerate(ids):
    a = s0.index('<section id="%s"' % sid)
    b = s0.index('</section>', a) + 10
    open(P + 'tmp/p3f/sec%s_%02d_%s.html' % (tag, i, sid), 'w', encoding='utf-8').write(collapse(s0[a:b]))
s = re.sub(r'<mjx-container[^>]*aria-label="([^"]*)"[^>]*>.*?</mjx-container>', lambda m: ' $' + m.group(1) + '$ ', s0, flags=re.S)
s = re.sub(r'<script.*?</script>', '', s, flags=re.S)
s = re.sub(r'<style.*?</style>', '', s, flags=re.S)
class Hp(HTMLParser):
    def __init__(s2):
        super().__init__(); s2.out = []
    def handle_starttag(s2, t, attrs):
        a = dict(attrs)
        if t == 'section' and a.get('id'): s2.out.append(f'\n\n######## SECTION {a["id"]}\n')
        elif a.get('id') and not a['id'].startswith('MJX'): s2.out.append(f' [#{a["id"]}] ')
        if t in ('h1', 'h2', 'h3', 'h4'): s2.out.append(f'\n\n== {t} ')
        if t == 'table': s2.out.append('\n<<TABLE>>\n')
        if t == 'tr': s2.out.append('\n| ')
        if t in ('p', 'li', 'div', 'figcaption', 'caption', 'br', 'dt', 'dd', 'pre'): s2.out.append('\n')
        if t == 'li': s2.out.append(' - ')
        if t == 'a' and a.get('href', '').startswith('#'): s2.out.append('{a' + a['href'] + '}')
        if a.get('data-tip'): s2.out.append(f' {{tip:{a["data-tip"]}}} ')
    def handle_endtag(s2, t):
        if t in ('td', 'th'): s2.out.append(' | ')
        if t in ('h1', 'h2', 'h3', 'h4'): s2.out.append('\n')
        if t == 'table': s2.out.append('\n<</TABLE>>\n')
    def handle_data(s2, d):
        s2.out.append(re.sub(r'[ \t\r\n]+', ' ', d))
h = Hp(); h.feed(s)
t = ''.join(h.out)
t = re.sub(r'\n[ \t]+', '\n', t); t = re.sub(r'\n{3,}', '\n\n', t)
open(P + 'tmp/p3f/text%s.txt' % tag, 'w', encoding='utf-8').write(t)
print(len(t))

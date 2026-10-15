"""Read-only: dump Results and weighting-test sections with MathJax collapsed to [MJX:label]."""
import re, sys, html
P = '/mnt/c/Users/David/dev/bca-shift-work/page/'
s = open(P + 'thrusterchain.html', encoding='utf-8').read()
def collapse(t):
    def rep(m):
        lab = re.search(r'aria-label="([^"]*)"', m.group(0)).group(1)
        disp = ' display' if 'display="true"' in m.group(0)[:300] else ''
        return '[MJX%s:%s]' % (disp, html.unescape(lab))
    t = re.sub(r'<mjx-container[^>]*>.*?</mjx-container>', rep, t, flags=re.S)
    return t
for sid, nxt in [('results', 'weighting-test'), ('weighting-test', 'arguments')]:
    a = s.index('<section id="%s"' % sid)
    b = s.index('<section id="%s"' % nxt)
    t = collapse(s[a:b])
    open(P + 'tmp/r4/cur_%s.html' % sid, 'w', encoding='utf-8').write(t)
    print(sid, a, b, b - a, len(t), t.count('\n'))

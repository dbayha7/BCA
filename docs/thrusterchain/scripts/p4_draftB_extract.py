"""Read-only: extract readable text of selected sections of thrusterchain.html.

Writes one text file per section to tmp/p4/draftB/<id>.txt. MathJax containers become $aria-label$.
SVG content is dropped. Block tags are shown as markers with id/class. data-tip text is kept in {tip: ...}
when --tips is given.
"""
import sys, os, re
from html.parser import HTMLParser

PAGE = os.path.join(os.path.dirname(__file__), '..', 'thrusterchain.html')
OUT = os.path.join(os.path.dirname(__file__), '..', 'tmp', 'p4', 'draftB')
TIPS = '--tips' in sys.argv
ids = [a for a in sys.argv[1:] if not a.startswith('--')]

BLOCK = {'section', 'h2', 'h3', 'h4', 'h5', 'p', 'li', 'ul', 'ol', 'table', 'tr', 'td', 'th', 'thead', 'tbody',
         'caption', 'dl', 'dt', 'dd', 'figure', 'figcaption', 'div', 'details', 'summary', 'blockquote', 'aside',
         'header', 'pre', 'label', 'button', 'fieldset', 'legend', 'output'}
VOID = {'br', 'img', 'input', 'meta', 'link', 'hr', 'source', 'wbr', 'col', 'area', 'base', 'embed', 'param', 'track'}


class P(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.out = []
        self.skip = 0  # inside svg / script / style
        self.mjx = 0
        self.depth = 0

    def handle_starttag(self, tag, attrs):
        a = dict(attrs)
        if self.skip:
            if tag in ('svg', 'script', 'style') or tag not in VOID:
                self.skip += 1
            return
        if tag == 'mjx-container':
            self.out.append('$' + (a.get('aria-label') or '?') + '$')
            self.skip = 1
            return
        if tag in ('svg', 'script', 'style'):
            lab = a.get('aria-label')
            self.out.append('[%s%s]' % (tag, (' "' + lab + '"') if lab else ''))
            self.skip = 1
            return
        if TIPS and a.get('data-tip'):
            self.out.append('{tip%s: %s}' % ((' ' + a['data-tip-title']) if a.get('data-tip-title') else '', a['data-tip']))
        if tag in BLOCK:
            bits = [tag]
            if a.get('id'):
                bits.append('#' + a['id'])
            if a.get('class'):
                bits.append('.' + '.'.join(a['class'].split()))
            for k in ('colspan', 'data-col', 'aria-label'):
                if a.get(k):
                    bits.append('%s=%s' % (k, a[k][:60]))
            self.out.append('\n' + '  ' * min(self.depth, 12) + '<' + ' '.join(bits) + '> ')
            if tag not in VOID:
                self.depth += 1
        elif tag == 'br':
            self.out.append(' / ')
        elif tag == 'a' and a.get('href'):
            self.out.append('[link ' + a['href'] + ']')
        elif tag in ('b', 'strong'):
            self.out.append('**')
        elif tag in ('i', 'em'):
            self.out.append('_')

    def handle_endtag(self, tag):
        if self.skip:
            self.skip -= 1
            return
        if tag in BLOCK:
            self.depth -= 1
        elif tag in ('b', 'strong'):
            self.out.append('**')
        elif tag in ('i', 'em'):
            self.out.append('_')

    def handle_data(self, d):
        if self.skip:
            return
        self.out.append(re.sub(r'\s+', ' ', d))


def section_html(s, sid):
    m = re.search(r'<section id="%s"[^>]*>' % re.escape(sid), s)
    if not m:
        raise SystemExit('no section ' + sid)
    i = m.start()
    # find matching </section>
    depth = 0
    for mm in re.finditer(r'<(/?)section\b', s[i:]):
        if mm.group(1):
            depth -= 1
            if depth == 0:
                return s[i:i + mm.end() + 1]
        else:
            depth += 1
    raise SystemExit('unclosed ' + sid)


def main():
    s = open(PAGE, encoding='utf-8').read()
    os.makedirs(OUT, exist_ok=True)
    for sid in ids:
        h = section_html(s, sid)
        p = P()
        p.feed(h)
        txt = ''.join(p.out)
        txt = re.sub(r'[ \t]+\n', '\n', txt)
        txt = re.sub(r'\n\s*\n+', '\n', txt)
        fn = os.path.join(OUT, sid + ('.tips' if TIPS else '') + '.txt')
        open(fn, 'w', encoding='utf-8').write(txt)
        print(sid, len(h), len(txt), fn)


main()

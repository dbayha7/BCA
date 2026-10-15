"""Read-only viewer: print a line range (or the block for an element id) of the page with every MathJax container
collapsed to <MJX "aria-label"> and long SVG path data shortened, so the markup can be read.

usage: e1_view.py FILE START END          (1-based, inclusive line numbers)
       e1_view.py FILE --id ID [--chars N] (from the tag carrying id=ID, N characters, default 6000)
       e1_view.py FILE --grep TEXT          (line numbers of lines containing TEXT, shortened)
"""
import re
import sys

MJX = re.compile(r'<mjx-container[^>]*?aria-label="([^"]*)"[^>]*>.*?</mjx-container>', re.S)
MJX2 = re.compile(r'<mjx-container.*?</mjx-container>', re.S)
PATHD = re.compile(r'\sd="[^"]{60,}"')
SVGB = re.compile(r'<svg\b([^>]*)>.*?</svg>', re.S)
ARIA = re.compile(r'\s+aria-label="[^"]*"')


def clean(s):
    s = MJX.sub(lambda m: '<MJX "%s">' % m.group(1), s)
    s = MJX2.sub('<MJX ?>', s)
    s = PATHD.sub(' d="..."', s)
    if KEEP_SVG:
        return s
    s = SVGB.sub(lambda m: '<SVG%s>[...]</SVG>' % ARIA.sub('', m.group(1))[:160], s)
    return s


KEEP_SVG = '--svg' in sys.argv
if KEEP_SVG:
    sys.argv.remove('--svg')


def main():
    path = sys.argv[1]
    with open(path, encoding='utf-8') as f:
        text = f.read()
    if sys.argv[2] == '--id':
        ident = sys.argv[3]
        n = int(sys.argv[5]) if len(sys.argv) > 5 and sys.argv[4] == '--chars' else 6000
        i = text.find('id="%s"' % ident)
        assert i >= 0, ident
        i = text.rfind('<', 0, i)
        line = text.count('\n', 0, i) + 1
        chunk = text[i:i + n * 6]
        out = clean(chunk)[:n]
        print('--- line %d ---' % line)
        print(out)
        return
    if sys.argv[2] == '--grep':
        needle = sys.argv[3]
        for k, line in enumerate(text.split('\n'), 1):
            if needle in line:
                c = clean(line)
                j = c.find(needle)
                print(k, ':', c[max(0, j - 200):j + 300])
        return
    a, b = int(sys.argv[2]), int(sys.argv[3])
    lines = text.split('\n')[a - 1:b]
    for k, line in enumerate(lines, a):
        c = clean(line)
        if len(c) > 4000:
            c = c[:4000] + ' …[%d more chars]' % (len(c) - 4000)
        print('%d: %s' % (k, c))


if __name__ == '__main__':
    main()

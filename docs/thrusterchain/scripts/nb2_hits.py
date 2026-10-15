"""Read-only: raw source around every hit of the given words (math collapsed to [$TeX$]), with the section and
whether the hit sits in a data-tip attribute, a script, or text.
  python3 nb2_hits.py <page> word [word ...]"""
import bisect
import html as H
import re
import sys

page = sys.argv[1]
words = sys.argv[2:]
s = open(page, encoding='utf-8').read()
MJX = re.compile(r'<mjx-container role="img" aria-label="(.*?)" class="MathJax" jax="SVG"( display="true")?>.*?</mjx-container>', re.S)
c = MJX.sub(lambda m: '[$' + H.unescape(m.group(1)) + '$]', s)
c = re.sub(r'<style\b.*?</style>', lambda m: '<style/>', c, flags=re.S)
c = re.sub(r'<svg\b.*?</svg>', lambda m: '<svg/>', c, flags=re.S)
marks = sorted([(m.start(), '#' + m.group(1)) for m in re.finditer(r'<section id="([^"]+)"', c)] +
               [(c.find('<figure class="panel" id="sim"'), '#sim'), (c.find('<header class="head" id="top"'), '#top'),
                (c.find('</main>'), 'after main')])
mp = [p for p, _ in marks]
scripts = [(m.start(), m.end()) for m in re.finditer(r'<script\b.*?</script>', c, re.S)]
for w in words:
    hits = [m.start() for m in re.finditer(r'\b' + re.escape(w) + r'\b', c, re.I)]
    print(f'==== {w!r}: {len(hits)}')
    for p in hits:
        sec = marks[bisect.bisect_right(mp, p) - 1][1]
        where = 'script' if any(a <= p < e for a, e in scripts) else 'text'
        tag_open = c.rfind('<', 0, p) > c.rfind('>', 0, p)
        if where == 'text' and tag_open:
            where = 'attribute'
        print(f'--- {sec} [{where}]')
        print('   ', repr(c[max(0, p - 260):p + 200]))

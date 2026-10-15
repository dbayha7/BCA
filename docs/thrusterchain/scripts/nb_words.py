"""Read-only: count candidate informal or figurative words in the page's visible text, attribute values and script
strings (math, styles and SVG removed), with the section of each hit.
  wsl.exe -d Ubuntu-22.04 -u root -e bash -lc 'python3 /mnt/c/Users/David/dev/bca-shift-work/page/scripts/nb_words.py [word ...]'
"""
import bisect
import html as H
import re
import sys

s = open('/mnt/c/Users/David/dev/bca-shift-work/page/thrusterchain.html', encoding='utf-8').read()
MJX = re.compile(r'<mjx-container role="img" aria-label="(.*?)" class="MathJax" jax="SVG"( display="true")?>.*?</mjx-container>', re.S)
marks = sorted([(m.start(), '#' + m.group(1)) for m in re.finditer(r'<section id="([^"]+)"', s)] +
               [(s.find('<figure class="panel" id="sim"'), '#sim'), (s.find('<header class="head" id="top"'), '#top'),
                (s.find('</main>'), 'after main (rail, scripts)')])
mp = [p for p, _ in marks]
WORDS = sys.argv[1:] or ['climb', 'climbs', 'climbing', 'hill', 'hills', 'favourite', 'favorite', 'nudge', 'nudges',
                         'believes', 'believe', 'knows', 'thinks', 'wants', 'likes', 'trusts', 'fooled', 'fools',
                         'toy', 'wobble', 'wobbles', 'kick', 'kicks', 'shove', 'bites', 'stalling world', 'lever',
                         'knob', 'dial', 'magic', 'trick', 'headline', 'punchline', 'sweet', 'blind', 'lures', 'chases',
                         'greedy', 'eager', 'happily', 'luckily', 'sadly', 'weird', 'odd', 'funny', 'nice', 'tiny', 'huge']
blank = MJX.sub(lambda m: ' ' * (m.end() - m.start()), s)
blank = re.sub(r'<style\b.*?</style>|<svg\b.*?</svg>', lambda m: ' ' * (m.end() - m.start()), blank, flags=re.S)
for w in WORDS:
    hits = [m.start() for m in re.finditer(r'\b' + re.escape(w) + r'\b', blank, re.I)]
    if not hits:
        continue
    secs = {}
    for p in hits:
        sec = marks[bisect.bisect_right(mp, p) - 1][1]
        secs[sec] = secs.get(sec, 0) + 1
    print(f'{w!r}: {len(hits)}  ', secs)
    for p in hits[:6]:
        t = re.sub(r'\s+', ' ', H.unescape(re.sub(r'<[^>]+>', '', blank[max(0, p - 120):p + 120])))
        print('     ', t)

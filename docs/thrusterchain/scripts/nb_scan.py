"""Read-only scan for the notation inventory (comment B): starred symbols and informal wording, everywhere on the page
(math labels, visible text, attributes such as data-tip, and script string literals) and in the tooltip text sources
ta_texts.py / ta_code.py. Writes notes/notation_scan.json and prints a summary.
  wsl.exe -d Ubuntu-22.04 -u root -e bash -lc 'python3 /mnt/c/Users/David/dev/bca-shift-work/page/scripts/nb_scan.py'
"""
import bisect
import html as H
import json
import re
import sys

PAGE = '/mnt/c/Users/David/dev/bca-shift-work/page/thrusterchain.html'
SCRIPTS = '/mnt/c/Users/David/dev/bca-shift-work/page/scripts/'
OUT = '/mnt/c/Users/David/dev/bca-shift-work/page/notes/notation_scan.json'
s = open(PAGE, encoding='utf-8').read()

MJX = re.compile(r'<mjx-container role="img" aria-label="(.*?)" class="MathJax" jax="SVG"( display="true")?>.*?</mjx-container>', re.S)

# ---- section map: section ids, the #sim figure, the header, the contents rail
marks = []
for m in re.finditer(r'<section id="([^"]+)"', s):
    marks.append((m.start(), '#' + m.group(1)))
for pat, name in [(r'<header class="head" id="top"', '#top (header)'), (r'<figure class="panel" id="sim"', '#sim'),
                  (r'<nav class="toc', 'contents rail'), (r'</main>', 'after main')]:
    m = re.search(pat, s)
    if m:
        marks.append((m.start(), name))
marks.sort()
mpos = [p for p, _ in marks]
heads = [(m.start(), re.sub(r'<[^>]+>', '', m.group(2)).strip()) for m in re.finditer(r'<h([1-4])\b[^>]*>(.*?)</h\1>', s, re.S)]
hpos = [p for p, _ in heads]


def where(pos):
    i = bisect.bisect_right(mpos, pos) - 1
    sec = marks[i][1] if i >= 0 else 'top of file'
    j = bisect.bisect_right(hpos, pos) - 1
    h = heads[j][1] if j >= 0 else ''
    h = MJX.sub(lambda m: H.unescape(m.group(1)), h)
    return sec, h[:80]


# ---- regions: scripts, styles, math containers
scripts = [(m.start(1), m.end(1)) for m in re.finditer(r'<script\b[^>]*>(.*?)</script>', s, re.S)]
styles = [(m.start(), m.end()) for m in re.finditer(r'<style\b[^>]*>.*?</style>', s, re.S)]
maths = [(m.start(), m.end(), m) for m in MJX.finditer(s)]


def inside(pos, spans):
    for a, b, *_ in spans:
        if a <= pos < b:
            return True
    return False


def plain(h):
    """Visible text: math kept as [TeX], tags removed, entities decoded, whitespace collapsed."""
    t = MJX.sub(lambda m: ' [' + H.unescape(m.group(1)) + '] ', h)
    t = re.sub(r'<[^>]+>', ' ', t)
    return re.sub(r'\s+', ' ', H.unescape(t)).strip()


# ---- JS string literals inside scripts (single, double, template), skipping comments and regex-free code
def js_strings(code, base):
    out, i, n = [], 0, len(code)
    while i < n:
        c = code[i]
        if c == '/' and code.startswith('//', i):
            j = code.find('\n', i)
            i = n if j < 0 else j
            continue
        if c == '/' and code.startswith('/*', i):
            j = code.find('*/', i + 2)
            i = n if j < 0 else j + 2
            continue
        if c in '\'"`':
            j = i + 1
            while j < n and code[j] != c:
                j += 2 if code[j] == '\\' else 1
            out.append((base + i + 1, base + j, code[i + 1:j]))
            i = j + 1
            continue
        i += 1
    return out


strings = []
for a, b in scripts:
    strings += js_strings(s[a:b], a)

STAR = re.compile(r'\^\{?\*\}?|[A-Za-zͰ-Ͽᴀ-ᶿ]\*(?![*/])|[⋆∗★☆]|\bstar(?:red|s)?\b|\ba-star\b', re.I)
INFORMAL_WORDS = [
    'favourite', 'favorite', 'favourites', 'favorites', 'favoured', 'favored', 'fave', 'darling', 'top pick', 'its pick',
    "critic's pick", 'best guess', 'sweet spot', 'knob', 'nudge', 'nudges', 'tug', 'yank', 'shove', 'trick', 'cheat',
    'magic', 'fancy', 'hack', 'toy', 'silly', 'weird', 'crazy', 'nice', 'lovely', 'basically', 'kind of', 'sort of',
    'a lot', 'lots of', 'huge', 'tiny', 'big deal', 'stuff', 'gonna', 'wanna', 'okay', 'hill', 'hilltop', 'peak',
    'likes', 'loves', 'wants', 'thinks', 'believes', 'hopes', 'fools', 'fooled', 'cheats', 'goofy', 'happy', 'unhappy',
    'blind spot', 'sweet', 'tame', 'wild', 'whopping', 'chase', 'chases', 'chasing', 'lured', 'lure', 'tempt', 'tempted',
    'over-eager', 'overeager', 'greedy', 'bets', 'bet on', 'gamble', 'jackpot', 'ghost', 'phantom', 'mirage', 'stalls out',
]
INFORMAL = re.compile(r'\b(' + '|'.join(re.escape(w) for w in INFORMAL_WORDS) + r')\b', re.I)

hits = []


def add(kind, pos, text, m, extra=None):
    sec, h = where(pos)
    a = max(0, m.start() - 110)
    b = min(len(text), m.end() + 110)
    rec = dict(kind=kind, pos=pos, section=sec, heading=h, match=m.group(0), context=text[a:b])
    if extra:
        rec.update(extra)
    hits.append(rec)


# 1. math labels
for a, b, m in maths:
    tex = H.unescape(m.group(1))
    for mm in STAR.finditer(tex):
        if mm.group(0).lower().startswith('star'):
            continue
        # the wrapping element's tooltip, when the math sits in a span with data-tip
        pre = s[max(0, a - 1500):a]
        k = pre.rfind('<span')
        tip = None
        if k >= 0 and '</span>' not in pre[k:]:
            tag = pre[k:pre.find('>', k) + 1]
            tt = re.search(r'data-tip-title="([^"]*)"', tag)
            td = re.search(r'data-tip="([^"]*)"', tag)
            tip = dict(title=H.unescape(tt.group(1)) if tt else None, tip=H.unescape(td.group(1)) if td else None)
        ctx = plain(s[max(0, a - 700):min(len(s), b + 700)])
        add('math', a, tex, mm, dict(display=bool(m.group(2)), tip=tip, near=ctx[:900]))
        break

# 2. attributes (outside math, scripts, styles): data-tip, data-tip-title, title, alt, aria-label, data-*
for m in re.finditer(r'\s([a-zA-Z-]+)="([^"]*)"', s):
    if inside(m.start(), maths) or inside(m.start(), scripts) or inside(m.start(), styles):
        continue
    name, val = m.group(1), H.unescape(m.group(2))
    if name in ('d', 'viewBox', 'transform', 'style', 'class', 'id', 'href', 'xlink:href', 'width', 'height', 'points'):
        continue
    for rx, kind in ((STAR, 'attr-star'), (INFORMAL, 'attr-word')):
        for mm in rx.finditer(val):
            add(kind, m.start(), val, mm, dict(attr=name))

# 3. visible text nodes (outside tags, math, scripts, styles)
for m in re.finditer(r'>([^<>]+)<', s):
    if inside(m.start(), maths) or inside(m.start(), scripts) or inside(m.start(), styles):
        continue
    val = H.unescape(m.group(1))
    if not val.strip():
        continue
    for rx, kind in ((STAR, 'text-star'), (INFORMAL, 'text-word')):
        for mm in rx.finditer(val):
            # show a wider plain-text context
            ctx = plain(s[max(0, m.start() - 600):min(len(s), m.end() + 600)])
            k = ctx.find(mm.group(0))
            add(kind, m.start(), val, mm, dict(wide=ctx[max(0, k - 250):k + 250]))

# 4. script string literals
for a, b, txt in strings:
    for rx, kind in ((STAR, 'js-star'), (INFORMAL, 'js-word')):
        for mm in rx.finditer(txt):
            add(kind, a, txt, mm)

# 5. tooltip text sources
for fn in ('ta_texts.py', 'ta_code.py'):
    src = open(SCRIPTS + fn, encoding='utf-8').read()
    for rx, kind in ((STAR, 'src-star'), (INFORMAL, 'src-word')):
        for mm in rx.finditer(src):
            line = src.count('\n', 0, mm.start()) + 1
            a = max(0, mm.start() - 120)
            hits.append(dict(kind=kind, file=fn, line=line, match=mm.group(0), context=src[a:mm.end() + 120]))

json.dump(hits, open(OUT, 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
from collections import Counter
print('hits', len(hits))
print(Counter(h['kind'] for h in hits))
print(Counter((h['kind'], h['match'].lower()) for h in hits).most_common(80))

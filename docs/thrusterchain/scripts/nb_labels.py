"""Extract every MathJax container on the page (TeX source, display flag, id prefix and number, full HTML) to
mathbuild/page_math.json, for the renderer's reproduction test and the notation inventory. Read-only on the page."""
import re, collections, json, html
P = '/mnt/c/Users/David/dev/bca-shift-work/page/thrusterchain.html'
s = open(P, encoding='utf-8').read()
out = []
for m in re.finditer(r'<mjx-container role="img" aria-label="(.*?)" class="MathJax" jax="SVG"( display="true")?>(.*?)</mjx-container>', s, re.S):
    blk = m.group(0)
    idm = re.search(r'id="(MJX[A-Z]?)-(\d+)-', blk)
    out.append(dict(pos=m.start(), label_raw=m.group(1), tex=html.unescape(m.group(1)), display=bool(m.group(2)),
                    prefix=idm.group(1) if idm else None, n=int(idm.group(2)) if idm else None, html=blk))
print('containers', len(out), 'of', s.count('<mjx-container'))
macros = collections.Counter()
for o in out:
    for mm in re.findall(r'\\[A-Za-z]+', o['tex']):
        macros[mm] += 1
print(sorted(macros))
json.dump(out, open('/mnt/c/Users/David/dev/bca-shift-work/mathbuild/page_math.json', 'w', encoding='utf-8'), ensure_ascii=False)
bad = [o['label_raw'] for o in out if o['label_raw'] != o['tex'].replace('&', '&amp;').replace('"', '&quot;').replace('<', '&lt;')]
print('escape mismatches', bad[:5])
print('prefixes', collections.Counter(o['prefix'] for o in out))

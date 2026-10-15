"""Read-only checks after edit 3 (comment C).
  wsl.exe -d Ubuntu-22.04 -u root -e bash -lc 'python3 /mnt/c/Users/David/dev/bca-shift-work/page/scripts/cc3_verify.py'
1. the v11 backup is unchanged; 2. tag balance against the page before edit 3; 3. every in-page link resolves; 4. every
new tooltip-source string (ta_texts.EDIT3_COUPLING and the A legend) is on the page word for word; 5. informal words and
stars in the new subsection; 6. every number in the new subsection's text is either on the page before edit 3 or one of
the computed coupling numbers listed here (from notes/coupling_facts.json and coupling_thrusters.json).
"""
import hashlib
import html as H
import json
import re
import sys

HERE = '/mnt/c/Users/David/dev/bca-shift-work/page/scripts'
sys.path.insert(0, HERE)
import mjx  # noqa: E402
import ta_texts as T  # noqa: E402

ROOT = '/mnt/c/Users/David/dev/bca-shift-work/page'
s = open(f'{ROOT}/thrusterchain.html', encoding='utf-8').read()
pre = open(f'{ROOT}/tmp/cc3/thrusterchain.pre-edit3.html', encoding='utf-8').read()
print('1. v11 backup md5:', hashlib.md5(open(f'{ROOT}/thrusterchain.v11.html', 'rb').read()).hexdigest(),
      '; page md5:', hashlib.md5(s.encode()).hexdigest())

print('2. tag balance (after minus before edit 3):')
for tag in ('p', 'span', 'div', 'dl', 'dt', 'dd', 'li', 'ul', 'table', 'thead', 'tbody', 'tr', 'td', 'th', 'h3', 'i',
            'b', 'a', 'mjx-container', 'svg', 'section', 'script'):
    o = len(re.findall(rf'<{tag}[\s>]', s)) - len(re.findall(rf'<{tag}[\s>]', pre))
    c = s.count(f'</{tag}>') - pre.count(f'</{tag}>')
    if o or c:
        print(f'   <{tag}>: opened {o:+d}, closed {c:+d}')
    assert o == c, tag

ids = set(re.findall(r'\bid="([^"]+)"', s))
links = [l for l in re.findall(r'href="#([^"]+)"', s) if not l.startswith('MJX')]
print(f'3. in-page links: {len(links)}, unresolved: {sorted({l for l in links if l not in ids}) or "none"}')

E = T.EDIT3_COUPLING
want = [H.escape(E['tip'][0], quote=True).replace('&#x27;', "'"), H.escape(E['tip'][1], quote=True).replace('&#x27;', "'"),
        E['legend']['C'], T.LEGENDS['tip-113'][1]] + [json.dumps(v, ensure_ascii=False) for v in E['cols'].values()] + \
       [json.dumps(k, ensure_ascii=False) for k in E['cols']]
miss = [w for w in want if w not in s]
print(f'4. new source strings: {len(want)}; not on the page word for word: {len(miss)}', miss[:3])

a = s.index('<h3 id="dynamics-coupling">')
b = s.index('<div class="tbl"><table class="stack" style="min-width:470px">', a)
sub = s[a:b]
text = H.unescape(re.sub(r'<[^>]+>', ' ', mjx.CONTAINER.sub(lambda m: ' ' + H.unescape(m.group(1)) + ' ', sub)))
text = re.sub(r'\s+', ' ', text)
words = ['favourite', 'favorite', 'hill', 'climbs', 'nudge', 'knows', 'trusts', 'trusted', 'wobble', 'jitter', 'blind',
         'toy', 'spectral', 'eigen', 'Gaussian', 'RMS', 'Version History']
print('5. informal words or codes in the new subsection:', [w for w in words if re.search(rf'\b{w}', text, re.I)] or 'none',
      '; stars:', text.count('*'))

pre_text = H.unescape(re.sub(r'<[^>]+>', ' ', pre))
pre_nums = set(re.findall(r'\d[\d,]*\.?\d*', pre_text))
COMPUTED = {  # coupling numbers new to the page, each from the facts files (checked by cc3_apply.py)
    '0.0106': 'largest entry of C', '0.003': 'mean |entry| of C (0.0027)', '0.004': 'smallest push transfer, 20 steps',
    '0.059': 'largest push transfer, 20 steps', '0.021': 'median push transfer, 20 steps',
    '0.051': 'float 1 to float 2 within 20 steps', '0.052': 'float 1 to float 2 within 100 steps',
    '0.018': 'largest one-step thrust transfer', '34': 'float 1 steps to halve without C', '17': 'float 5 without C',
    '2.37': 'float 2 wave response without C', '403.44': 'no-throttle return without C', '0.12': 'return change %',
    '5.6': 'largest single-episode change % (also a push-table percentage elsewhere)', '3.7': 'float 1 share %',
    '9.7': 'float 2 share %', '6.5': 'float 3 share %', '8.6': 'float 5 share %', '6.2': 'all floats share %',
    '0.02': 'size of C', '2': 'per cent of the float numbers', '6': 'per cent reach', '20': 'steps', '100': 'steps',
}
table_cells = {f'{x:.1f}' for x in (5.1, 2.8, 1.8, 2.7, 3.3, 1.4, 0.4, 0.9, 3.0, 5.9, 2.4, 2.2, 0.7, 0.6, 1.0, 2.1)}
nums = re.findall(r'\d[\d,]*\.?\d*', text)
unknown = sorted({n.rstrip('.') for n in nums} - pre_nums - set(COMPUTED) - table_cells)
print(f'6. numbers in the new subsection: {len(set(nums))} distinct; neither on the page before nor listed as computed: '
      f'{unknown or "none"}')
print('   formulas:', s.count('<mjx-container'), '; glyph ids unique:', mjx.check(s))

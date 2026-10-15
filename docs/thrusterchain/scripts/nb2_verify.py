"""Read-only checks after edit 2 (comment B).
  wsl.exe -d Ubuntu-22.04 -u root -e bash -lc 'python3 /mnt/c/Users/David/dev/bca-shift-work/page/scripts/nb2_verify.py'
1. "favourite"/"favorite" anywhere in the page (raw bytes, scripts and comments included), the tooltip sources and the
   Results draft; 2. tag balance against the page before edit 2; 3. every in-page link resolves; 4. every tooltip-source
   string that edit 2 changed is on the page, word for word; 5. the informal-word scan of the page.
"""
import html as H
import re
import sys

HERE = '/mnt/c/Users/David/dev/bca-shift-work/page/scripts'
sys.path.insert(0, HERE)
import mjx  # noqa: E402
import ta_code as C  # noqa: E402
import ta_texts as T  # noqa: E402

ROOT = '/mnt/c/Users/David/dev/bca-shift-work/page'
s = open(f'{ROOT}/thrusterchain.html', encoding='utf-8').read()
pre = open(f'{ROOT}/tmp/nb2/thrusterchain.pre-edit2.html', encoding='utf-8').read()

print('1. "favourite" / "favorite" (case-insensitive, raw text including scripts and comments):')
for name, path in (('page', f'{ROOT}/thrusterchain.html'), ('ta_texts.py', f'{HERE}/ta_texts.py'),
                   ('ta_code.py', f'{HERE}/ta_code.py'), ('results_draft.html', f'{ROOT}/notes/results_draft.html'),
                   ('weighting_results_draft.html', f'{ROOT}/notes/weighting_results_draft.html'),
                   ('coupling.md', f'{ROOT}/notes/coupling.md')):
    t = open(path, encoding='utf-8').read()
    print(f'   {name}: {len(re.findall(r"favou?rite", t, re.I))}')

print('2. tag balance (after minus before edit 2):')
for tag in ('p', 'span', 'div', 'dl', 'dt', 'dd', 'li', 'ul', 'ol', 'table', 'tr', 'td', 'th', 'h3', 'h4', 'i', 'b',
            'sup', 'sub', 'mjx-container', 'svg', 'section', 'figure', 'template', 'script'):
    o = len(re.findall(rf'<{tag}[\s>]', s)) - len(re.findall(rf'<{tag}[\s>]', pre))
    c = s.count(f'</{tag}>') - pre.count(f'</{tag}>')
    if o or c:
        print(f'   <{tag}>: opened {o:+d}, closed {c:+d}')
    assert len(re.findall(rf'<{tag}[\s>]', s)) - s.count(f'</{tag}>') == len(re.findall(rf'<{tag}[\s>]', pre)) - pre.count(f'</{tag}>'), tag

ids = set(re.findall(r'\bid="([^"]+)"', s))
links = re.findall(r'href="#([^"]+)"', s)
page_links = [l for l in links if not l.startswith('MJX')]
bad = sorted({l for l in page_links if l not in ids})
print(f'3. in-page links: {len(page_links)}, unresolved: {bad or "none"}')

def leaves(x):
    if isinstance(x, str):
        yield x
    elif isinstance(x, dict):
        for v in x.values():
            yield from leaves(v)
    elif isinstance(x, (list, tuple)):
        for v in x:
            yield from leaves(v)

MARKS = ['rates highest', 'random noise', 'trained to raise', 'trained against', 'without its placement',
         'strength of the cloning pull', "critic's value around", 'â']
srcs = list(leaves(T.TIPS)) + list(leaves(T.LEGENDS)) + list(leaves(T.GUIDE)) + list(leaves(T.COLS_NEW)) + \
       list(leaves(C.TT)) + list(leaves(C.DR)) + list(leaves(C.IW))
changed = [t for t in srcs if any(m in t for m in MARKS)]
missing = [t for t in changed if not any(v in s for v in (t, H.escape(t, quote=True), H.escape(t, quote=False),
                                                          t.replace('"', '\\"')))]
print(f'4. tooltip-source strings with edit 2 wording: {len(changed)}; not found on the page word for word: {len(missing)}')
for t in missing:
    print('   ', t[:200])
old_words = ['favourite', 'hill', 'climbs', 'nudge', 'knows', 'trusts', 'trusted', 'wobble', 'wobbles', 'jitter',
             'blindly', 'believes']
src_all = open(f'{HERE}/ta_texts.py', encoding='utf-8').read() + open(f'{HERE}/ta_code.py', encoding='utf-8').read()
print('   old words left in ta_texts.py + ta_code.py:', {w: len(re.findall(rf'\b{w}\b', src_all, re.I)) for w in old_words if re.search(rf'\b{w}\b', src_all, re.I)} or 'none')
b = mjx.CONTAINER.sub(' ', s)
b = re.sub(r'<style\b.*?</style>|<svg\b.*?</svg>', ' ', b, flags=re.S)
print('5. old words left on the page (text, attributes, scripts):',
      {w: len(re.findall(rf'\b{w}\b', b, re.I)) for w in old_words + ['blind'] if re.search(rf'\b{w}\b', b, re.I)} or 'none')
print('   formulas:', s.count('<mjx-container'), '; glyph ids unique:', mjx.check(s))

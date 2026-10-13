"""Read-only: print a stretch of the page as source with math collapsed to [$TeX$] (or [$$TeX$$] for displays).
  python3 nb2_peek.py <page> <start marker> [chars=3000] [--text]
--text drops tags (keeps data-tip text as {tip: ...}) for proofreading."""
import html as H
import re
import sys

page, marker = sys.argv[1], sys.argv[2]
n = int(sys.argv[3]) if len(sys.argv) > 3 and sys.argv[3].isdigit() else 3000
text_mode = '--text' in sys.argv
s = open(page, encoding='utf-8').read()
MJX = re.compile(r'<mjx-container role="img" aria-label="(.*?)" class="MathJax" jax="SVG"( display="true")?>.*?</mjx-container>', re.S)
a = s.find(marker)
assert a >= 0, f'marker not found: {marker!r}'
print('marker count:', s.count(marker), 'at', a)
seg = MJX.sub(lambda m: ('[$$' if m.group(2) else '[$') + H.unescape(m.group(1)) + ('$$]' if m.group(2) else '$]'), s[a:a + n * 40])
if text_mode:
    seg = re.sub(r'<[^>]*data-tip-title="([^"]*)"[^>]*data-tip="([^"]*)"[^>]*>', lambda m: '{' + m.group(1) + ': ' + m.group(2) + '}', seg)
    seg = re.sub(r'<[^>]+>', '', seg)
    seg = H.unescape(seg)
print(seg[:n])

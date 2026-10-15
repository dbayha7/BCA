"""Readable text of the page (copy of e6_text.py): writes tmp/ta/page_text_v10.txt. Read-only on the page."""
import re
from html.parser import HTMLParser
P='/mnt/c/Users/David/dev/bca-shift-work/page/thrusterchain.html'
OUT='/mnt/c/Users/David/dev/bca-shift-work/page/tmp/ta/page_text_v10.txt'
s=open(P,encoding='utf-8').read()
s=re.sub(r'<mjx-container[^>]*aria-label="([^"]*)"[^>]*>.*?</mjx-container>', lambda m:' $'+m.group(1)+'$ ', s, flags=re.S)
s=re.sub(r'<script.*?</script>','',s,flags=re.S)
s=re.sub(r'<style.*?</style>','',s,flags=re.S)
tabn=[0]
class H(HTMLParser):
    def __init__(s2):
        super().__init__(); s2.out=[]; s2.skip=0
    def handle_starttag(s2,tag,attrs):
        a=dict(attrs)
        if tag=='svg' and 'dg' not in (a.get('class') or '') and 'chart' not in (a.get('class') or ''): pass
        if tag in ('section',) and a.get('id'): s2.out.append(f'\n\n######## SECTION {a["id"]}\n')
        if a.get('id') and tag not in ('section',) and not a['id'].startswith('MJX'): s2.out.append(f' [#{a["id"]}] ')
        if tag in ('h1','h2','h3','h4'): s2.out.append(f'\n\n== {tag} ')
        if tag=='table': s2.out.append(f'\n<<TABLE {tabn[0]}>>\n'); tabn[0]+=1
        if tag=='tr': s2.out.append('\n| ')
        if tag in ('p','li','div','figcaption','caption','br','dt','dd','pre'): s2.out.append('\n')
        if tag=='li': s2.out.append(' - ')
        if a.get('data-tip'): s2.out.append(f' {{tip:{a["data-tip"]}}} ')
        if a.get('aria-label') and tag not in ('svg',): s2.out.append(f' {{aria:{a["aria-label"]}}} ')
    def handle_endtag(s2,tag):
        if tag in ('td','th'): s2.out.append(' | ')
        if tag in ('h1','h2','h3','h4'): s2.out.append('\n')
        if tag=='table': s2.out.append('\n<</TABLE>>\n')
    def handle_data(s2,d):
        s2.out.append(re.sub(r'[ \t\r\n]+',' ',d))
h=H(); h.feed(s)
t=''.join(h.out)
t=re.sub(r'\n[ \t]+','\n',t); t=re.sub(r'\n{3,}','\n\n',t)
open(OUT,'w',encoding='utf-8').write(t)
print(len(t))

"""Print each table's section, computed column labels and first data row, as the tooltip script will see them."""
import json, os, sys, tempfile
sys.path.insert(0, '/mnt/c/Users/David/dev/bca-shift-work/page/scripts')
from shot import wrap
from playwright.sync_api import sync_playwright
P='/mnt/c/Users/David/dev/bca-shift-work/page/thrusterchain.html'
JS = open('/mnt/c/Users/David/dev/bca-shift-work/page/scripts/e5_probe.js', encoding='utf-8').read()
frag=open(P,encoding='utf-8').read()
d=tempfile.mkdtemp(); f=os.path.join(d,'p.html'); open(f,'w',encoding='utf-8').write(wrap(frag,'light'))
with sync_playwright() as p:
    b=p.chromium.launch(); pg=b.new_page(viewport={'width':1200,'height':800})
    pg.goto('file://'+f); pg.wait_for_timeout(800)
    res=pg.evaluate(JS)
    b.close()
out='/mnt/c/Users/David/dev/bca-shift-work/page/tmp/e5/labels.json'
json.dump(res, open(out,'w',encoding='utf-8'), ensure_ascii=False, indent=1)
for i,t in enumerate(res):
    print(f"#{i} [{t['sec']}] kv={t['kv']} groups={t['groups']}")
    for l in t['labels']: print('    ', repr(l))
    print('    row:', t['row'][:3])

"""Read-only: open the walkthrough at a given section and list the elements that stick out of its text area.
  /opt/shot-venv/bin/python r4_wt_over.py [page] [section id ...]"""
import os, sys, tempfile
from playwright.sync_api import sync_playwright
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from shot import wrap  # noqa: E402
PAGE = sys.argv[1] if len(sys.argv) > 1 else '/mnt/c/Users/David/dev/bca-shift-work/page/thrusterchain.html'
IDS = sys.argv[2:] or ['results', 'weighting-test', 'methods']
frag = open(PAGE, encoding='utf-8').read()
d = tempfile.mkdtemp(); f = os.path.join(d, 'p.html'); open(f, 'w', encoding='utf-8').write(wrap(frag, 'light'))
with sync_playwright() as p:
    b = p.chromium.launch(); pg = b.new_page(viewport={'width': 1200, 'height': 800}); pg.goto('file://' + f); pg.wait_for_timeout(800)
    pg.evaluate("document.getElementById('wt-open').click()"); pg.wait_for_timeout(300)
    for _ in range(25):
        sid = pg.evaluate("(() => { const s = document.getElementById('wt-stage'); return s.children[0] && s.children[0].id; })()")
        if sid in IDS:
            r = pg.evaluate("""() => { const b = document.getElementById('wt-body'); const br = b.getBoundingClientRect();
              const out = []; for (const el of b.querySelectorAll('*')) { const r = el.getBoundingClientRect();
                if (r.width && r.right > br.left + b.clientWidth + 0.5) out.push([el.tagName + '.' + el.className, Math.round(r.right - br.left - b.clientWidth), (el.id || ''), el.textContent.slice(0, 50)]); }
              return {over: b.scrollWidth - b.clientWidth, out: out.slice(0, 12)}; }""")
            print(sid, r['over']); [print('   ', o) for o in r['out']]
        nxt = pg.query_selector('#wt-next')
        if not nxt: break
        pg.evaluate("document.getElementById('wt-next').click()"); pg.wait_for_timeout(250)
    b.close()

"""Read-only browser check after edit 2: the shared tooltip shows the new wording for the share column header in
Methods (a column description in the tooltip script) and for the a-hat symbol, at 390 px with touch.
  wsl.exe -d Ubuntu-22.04 -u root -e bash -lc 'cd /mnt/c/Users/David/dev/bca-shift-work/page && /opt/shot-venv/bin/python scripts/nb2_tipcheck.py'
"""
import os
import sys
import tempfile

from playwright.sync_api import sync_playwright

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import shot  # noqa: E402

PAGE = '/mnt/c/Users/David/dev/bca-shift-work/page/thrusterchain.html'
frag = open(PAGE, encoding='utf-8').read()
d = tempfile.mkdtemp(prefix='nb2tip-')
open(f'{d}/page.html', 'w', encoding='utf-8').write(shot.wrap(frag, 'light'))
TIP = "() => { const t = document.querySelector('.tt'); return t ? {hidden: t.hidden || getComputedStyle(t).display === 'none' || getComputedStyle(t).visibility === 'hidden', text: t.innerText} : null; }"
errors = []
with sync_playwright() as p:
    b = p.chromium.launch()
    ctx = b.new_context(viewport={'width': 390, 'height': 844}, has_touch=True, is_mobile=True)
    pg = ctx.new_page()
    pg.on('pageerror', lambda e: errors.append(str(e)))
    pg.on('console', lambda m: errors.append(m.text) if m.type == 'error' else None)
    pg.goto(f'file://{d}/page.html', wait_until='load')
    pg.wait_for_timeout(800)
    th = pg.locator('#methods table').nth(1).locator('thead th').nth(1)
    th.scroll_into_view_if_needed()
    th.tap()
    pg.wait_for_timeout(400)
    t = pg.evaluate(TIP)
    print('header:', th.inner_text().strip()[:80])
    print('  tooltip shown:', bool(t and not t['hidden']), '| has "rates highest":', bool(t and 'rates highest' in t['text']))
    print('  text:', (t or {}).get('text', '')[:300].replace('\n', ' / '))
    sym = pg.locator('#methods .sym[data-tip-title^="â"]').first
    sym.scroll_into_view_if_needed()
    sym.tap()
    pg.wait_for_timeout(400)
    t = pg.evaluate(TIP)
    print('a-hat symbol tooltip shown:', bool(t and not t['hidden']))
    print('  text:', (t or {}).get('text', '')[:300].replace('\n', ' / '))
    print('errors:', errors or 0)
    b.close()

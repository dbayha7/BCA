"""Screenshots for edit 3 (comment C): the new coupling subsection of Transition Dynamics, its push table (cards at
390 px, a grid at 1200 px), the legend with C, the C tooltip and a column-header tooltip of the push table.
  wsl.exe -d Ubuntu-22.04 -u root -e bash -lc 'cd /mnt/c/Users/David/dev/bca-shift-work/page && /opt/shot-venv/bin/python scripts/cc3_shots.py [page]'
Default page: tmp/cc3/applied.html. Shots go to tmp/cc3/shots.
"""
import subprocess
import sys

PAGE = sys.argv[1] if len(sys.argv) > 1 else '/mnt/c/Users/David/dev/bca-shift-work/page/tmp/cc3/applied.html'
OUT = '/mnt/c/Users/David/dev/bca-shift-work/page/tmp/cc3/shots'
SHOT = '/mnt/c/Users/David/dev/bca-shift-work/page/scripts/shot.py'

TOP = "const top = (el, pad) => { const r = el.getBoundingClientRect(); window.scrollTo(0, r.top + scrollY - pad); };\n"
# the subsection from its heading to the swings table: a tall element-free shot via a temporary wrapper box
SUB = TOP + r"""
(() => { const h = document.getElementById('dynamics-coupling'); top(h, 8); return !!h; })()
"""
PUSHTBL = TOP + r"""
(() => { const t = [...document.querySelectorAll('#dynamics table')].find(t => t.textContent.includes('Push given to'));
         top(t.closest('.tbl').previousElementSibling, -40); return !!t; })()
"""
WHY = TOP + r"""
(() => { const b = [...document.querySelectorAll('#dynamics p b')].find(b => b.textContent.startsWith('Why the coupling'));
         top(b.closest('p').previousElementSibling, 8); return !!b; })()
"""
LEGEND = TOP + r"""
(() => { const k = document.querySelector('#dynamics .symkey'); top(k, 8); return !!k; })()
"""
TIP_C = TOP + r"""
(() => { const m = [...document.querySelectorAll('#dynamics mjx-container')].find(m => m.getAttribute('aria-label') === 'C');
         const sym = m.closest('.sym'); top(sym.closest('p'), 140);
         const r = sym.getBoundingClientRect();
         sym.dispatchEvent(new PointerEvent('pointerover', {bubbles: true, pointerType: 'mouse', clientX: r.left + 3, clientY: r.top + 3}));
         return sym.dataset.tipTitle; })()
"""
TIP_HEAD = TOP + r"""
(() => { const t = [...document.querySelectorAll('#dynamics table')].find(t => t.textContent.includes('Push given to'));
         const th = t.querySelectorAll('thead th')[2]; top(t.closest('.tbl'), 260);
         const r = th.getBoundingClientRect();
         th.dispatchEvent(new PointerEvent('pointerover', {bubbles: true, pointerType: 'mouse', clientX: r.left + 3, clientY: r.top + 3}));
         return th.textContent; })()
"""
TIP_CELL = TOP + r"""
(() => { const t = [...document.querySelectorAll('#dynamics table')].find(t => t.textContent.includes('Push given to'));
         const td = t.querySelectorAll('tbody tr')[2].querySelectorAll('td')[2]; top(td, 240);
         const r = td.getBoundingClientRect();
         td.dispatchEvent(new PointerEvent('pointerover', {bubbles: true, pointerType: 'mouse', clientX: r.left + 3, clientY: r.top + 3}));
         return td.textContent; })()
"""
RUNS = [
    ('sub', SUB, [(390, 'light', 2400, []), (1200, 'dark', 1500, [])]),
    ('pushtable', PUSHTBL, [(390, 'light', 1300, []), (1200, 'dark', 700, []), (1200, 'light', 700, [])]),
    ('why', WHY, [(390, 'dark', 2000, ['--reduced-motion']), (1200, 'light', 1200, [])]),
    ('legend', LEGEND, [(390, 'light', 1100, []), (1200, 'dark', 600, [])]),
    ('tip-C', TIP_C, [(390, 'light', 844, []), (1200, 'dark', 700, [])]),
    ('tip-head', TIP_HEAD, [(1200, 'dark', 700, [])]),
    ('tip-cell', TIP_CELL, [(390, 'light', 844, []), (1200, 'light', 700, [])]),
]
for name, js, sizes in RUNS:
    for width, scheme, height, extra in sizes:
        args = ['/opt/shot-venv/bin/python', SHOT, '--page', PAGE, '--out-dir', OUT, '--width', str(width),
                '--height', str(height), '--scheme', scheme, '--wait', '700', '--eval', js, *extra,
                '--shot', f'{name}-{width}-{scheme}.png=@viewport']
        r = subprocess.run(args, capture_output=True, text=True)
        print(f'== {name} {width} {scheme} {" ".join(extra)}')
        print(r.stdout[-700:], r.stderr[-500:])

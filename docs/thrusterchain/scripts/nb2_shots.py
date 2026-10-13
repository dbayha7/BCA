"""Screenshots for edit 2: two re-typeset formulas with their legends, and the a-hat and delta-exact tooltips.
  wsl.exe -d Ubuntu-22.04 -u root -e bash -lc 'cd /mnt/c/Users/David/dev/bca-shift-work/page && /opt/shot-venv/bin/python scripts/nb2_shots.py [page]'
Default page: tmp/nb2/applied.html. Shots go to tmp/nb2/shots.
"""
import subprocess
import sys

PAGE = sys.argv[1] if len(sys.argv) > 1 else '/mnt/c/Users/David/dev/bca-shift-work/page/tmp/nb2/applied.html'
OUT = '/mnt/c/Users/David/dev/bca-shift-work/page/tmp/nb2/shots'
SHOT = '/mnt/c/Users/David/dev/bca-shift-work/page/scripts/shot.py'

FIND = r"""
const byLabel = (pred) => [...document.querySelectorAll('mjx-container')].filter(m => pred(m.getAttribute('aria-label') || ''));
const top = (el, pad) => { const r = el.getBoundingClientRect(); window.scrollTo(0, r.top + scrollY - pad); };
"""
# 1. Methods: the paragraph that introduces a-hat, the pi display and its legend
METHODS = FIND + r"""
(() => { const a = byLabel(l => l === '\\hat{a}')[0]; top(a.closest('p'), 12); return document.querySelectorAll('mjx-container').length; })()
"""
# 2. After the stop, Stage 1: the delta-exact definition and its legend
STAGE1 = FIND + r"""
(() => { const d = byLabel(l => l.startsWith('\\begin{aligned} \\delta^{\\text{exact}}'))[0];
         let p = d.closest('.math-display').previousElementSibling; top(p || d, 12); return !!d; })()
"""
# 3. Hover on the first a-hat: its tooltip
HOVER_AHAT = FIND + r"""
(() => { const a = byLabel(l => l === '\\hat{a}')[0]; const sym = a.closest('.sym'); top(sym.closest('p'), 120);
         const r = sym.getBoundingClientRect();
         sym.dispatchEvent(new PointerEvent('pointerover', {bubbles: true, pointerType: 'mouse', clientX: r.left + 3, clientY: r.top + 3}));
         return sym.dataset.tipTitle; })()
"""
# 4. Hover on the first inline delta-exact: its tooltip
HOVER_DELTA = FIND + r"""
(() => { const d = byLabel(l => l === '\\delta^{\\text{exact}}')[0]; const sym = d.closest('.sym'); top(sym.closest('p'), 160);
         const r = sym.getBoundingClientRect();
         sym.dispatchEvent(new PointerEvent('pointerover', {bubbles: true, pointerType: 'mouse', clientX: r.left + 3, clientY: r.top + 3}));
         return sym.dataset.tipTitle; })()
"""
RUNS = [
    ('methods', METHODS, {(390, 'light'): 1500, (1200, 'dark'): 1000}),
    ('stage1', STAGE1, {(390, 'light'): 1500, (1200, 'dark'): 900}),
    ('tip-ahat', HOVER_AHAT, {(390, 'light'): 844, (1200, 'dark'): 700}),
    ('tip-delta', HOVER_DELTA, {(390, 'light'): 844, (1200, 'dark'): 700}),
]
for name, js, sizes in RUNS:
    for (width, scheme), height in sizes.items():
        args = ['/opt/shot-venv/bin/python', SHOT, '--page', PAGE, '--out-dir', OUT, '--width', str(width),
                '--height', str(height), '--scheme', scheme, '--wait', '700', '--eval', js,
                '--shot', f'{name}-{width}-{scheme}.png=@viewport']
        r = subprocess.run(args, capture_output=True, text=True)
        print(f'== {name} {width} {scheme}')
        print(r.stdout[-900:], r.stderr[-600:])

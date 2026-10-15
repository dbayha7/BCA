"""Screenshots for edit 4: Results and the weighting test at 390 px light and 1200 px dark, as readable slices, plus the
header and a tooltip of a new table.
  wsl.exe -d Ubuntu-22.04 -u root -e bash -lc 'cd /mnt/c/Users/David/dev/bca-shift-work/page && /opt/shot-venv/bin/python scripts/r4_shots.py [page]'
Default page: tmp/r4/applied.html. Shots go to tmp/r4/shots.
"""
import subprocess
import sys

PAGE = sys.argv[1] if len(sys.argv) > 1 else '/mnt/c/Users/David/dev/bca-shift-work/page/tmp/r4/applied.html'
OUT = '/mnt/c/Users/David/dev/bca-shift-work/page/tmp/r4/shots'
SHOT = '/mnt/c/Users/David/dev/bca-shift-work/page/scripts/shot.py'
PY = '/opt/shot-venv/bin/python'
TOP = "const top = (el, pad) => { const r = el.getBoundingClientRect(); window.scrollTo(0, r.top + scrollY - pad); };\n"
TIP_HEAD = TOP + r"""
(() => { const t = document.querySelector('#results table');
         const th = t.querySelectorAll('thead th')[1]; top(t.closest('.tbl'), 260);
         const r = th.getBoundingClientRect();
         th.dispatchEvent(new PointerEvent('pointerover', {bubbles: true, pointerType: 'mouse', clientX: r.left + 3, clientY: r.top + 3}));
         return th.textContent; })()
"""
TIP_CELL = TOP + r"""
(() => { const t = [...document.querySelectorAll('#weighting-test table')].find(t => t.textContent.includes('Result on the five qualities'));
         const td = t.querySelectorAll('tbody tr')[13].querySelectorAll('td')[3]; top(td, 300);
         const r = td.getBoundingClientRect();
         td.dispatchEvent(new PointerEvent('pointerover', {bubbles: true, pointerType: 'mouse', clientX: r.left + 3, clientY: r.top + 3}));
         return td.textContent.slice(0, 60); })()
"""
RUNS = [
    (390, 'light', ['--split', '1600', '--shot', 'res-390-light.png=#results', '--shot', 'wt-390-light.png=#weighting-test',
                    '--shot', 'head-390-light.png=header.head']),
    (1200, 'dark', ['--split', '1400', '--shot', 'res-1200-dark.png=#results', '--shot', 'wt-1200-dark.png=#weighting-test',
                    '--shot', 'head-1200-dark.png=header.head']),
    (1200, 'dark', ['--eval', TIP_HEAD, '--wait', '700', '--shot', 'tip-head-1200-dark.png=@viewport']),
    (390, 'light', ['--eval', TIP_CELL, '--wait', '700', '--shot', 'tip-cell-390-light.png=@viewport']),
    (390, 'dark', ['--reduced-motion', '--shot', 'wt-pipe-390-dark.png=#weighting-test .iw-pipe']),
]
for w, sc, extra in RUNS:
    cmd = [PY, SHOT, '--page', PAGE, '--out-dir', OUT, '--width', str(w), '--scheme', sc] + extra
    r = subprocess.run(cmd, capture_output=True, text=True)
    print(f'== {w} {sc} exit {r.returncode}')
    print(r.stdout[-1500:], r.stderr[-1500:])

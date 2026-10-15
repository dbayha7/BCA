"""Screenshots of tmp/nb/b_applied.html (a copy with every comment-B edit; the working page is not touched).
  wsl.exe -d Ubuntu-22.04 -u root -e bash -lc 'cd /mnt/c/Users/David/dev/bca-shift-work/page && /opt/shot-venv/bin/python scripts/nb_copy_shots.py'
"""
import subprocess

PAGE = '/mnt/c/Users/David/dev/bca-shift-work/page/tmp/nb/b_applied.html'
OUT = '/mnt/c/Users/David/dev/bca-shift-work/page/tmp/nb'
MARK = r"""
(() => {
  const m = [...document.querySelectorAll('mjx-container')].find(x => x.getAttribute('aria-label') === '\\hat{a}');
  const p = m.closest('p'); p.id = 'nbP1';
  let k = p.nextElementSibling; let i = 2;
  while (k && i < 5) { k.id = 'nbP' + i; k = k.nextElementSibling; i++; }
  const d = [...document.querySelectorAll('dt')].find(x => x.textContent.startsWith('δexact'));
  d.closest('.symkey').id = 'nbK1';
  return i;
})()
"""
for width, scheme in ((390, 'light'), (1200, 'dark')):
    args = ['/opt/shot-venv/bin/python', '/mnt/c/Users/David/dev/bca-shift-work/page/scripts/shot.py', '--page', PAGE,
            '--out-dir', OUT, '--width', str(width), '--scheme', scheme, '--eval', MARK]
    for k in ('nbP1', 'nbP2', 'nbP3', 'nbP4', 'nbK1'):
        args += ['--shot', f'c-{k}-{width}-{scheme}.png=#{k}']
    r = subprocess.run(args, capture_output=True, text=True)
    print(r.stdout[-1200:], r.stderr[-800:])

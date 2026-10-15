"""Screenshots of the re-typeset formulas in tmp/nb/trial.html (a copy; the working page is not touched).
  wsl.exe -d Ubuntu-22.04 -u root -e bash -lc 'cd /mnt/c/Users/David/dev/bca-shift-work/page && /opt/shot-venv/bin/python scripts/nb_trial_shots.py'
"""
import subprocess

PAGE = '/mnt/c/Users/David/dev/bca-shift-work/page/tmp/nb/trial.html'
OUT = '/mnt/c/Users/David/dev/bca-shift-work/page/tmp/nb'
MARK = r"""
(() => {
  const by = l => [...document.querySelectorAll('mjx-container')].filter(m => m.getAttribute('aria-label') === l);
  const tag = (el, id) => { if (el) el.id = id; };
  tag(by('\\hat{a}')[0].closest('p'), 'nbA1');
  tag(by('\\pi = \\frac{m\\,a + \\mu\\,\\hat{a}}{m + \\mu}, \\qquad \\mu = \\frac{d\\,\\lambda\\,\\kappa}{2}')[0].closest('.math-display'), 'nbA2');
  tag(by('\\lambda\\,\\kappa\\,(\\hat{a} - a)')[0].closest('li'), 'nbA3');
  const d = [...document.querySelectorAll('mjx-container')].filter(m => (m.getAttribute('aria-label') || '').includes('\\delta^{\\text{exact}}'));
  tag(d[0].closest('.math-display, p'), 'nbD1');
  tag(d[1].closest('p'), 'nbD2');
  const w = [...document.querySelectorAll('mjx-container')].find(m => (m.getAttribute('aria-label') || '').startsWith('W(x_i, a) = \\big[\\delta^{\\text{exact}}'));
  tag(w.closest('.math-display, p'), 'nbD3');
  tag(by('[-\\delta^{\\text{exact}}(x_i, a)]_+')[0].closest('p, li'), 'nbD4');
  return d.length;
})()
"""
for width, scheme in ((390, 'light'), (1200, 'dark')):
    args = ['/opt/shot-venv/bin/python', '/mnt/c/Users/David/dev/bca-shift-work/page/scripts/shot.py', '--page', PAGE,
            '--out-dir', OUT, '--width', str(width), '--scheme', scheme, '--eval', MARK]
    for k in ('nbA1', 'nbA2', 'nbA3', 'nbD1', 'nbD2', 'nbD3', 'nbD4'):
        args += ['--shot', f'{k}-{width}-{scheme}.png=#{k}']
    print(subprocess.run(args, capture_output=True, text=True).stdout[-1500:])

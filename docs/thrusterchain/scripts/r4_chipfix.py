"""Edit 4, follow-up: in the weighting test's pass-marks table, put the colon after a status chip inside the chip, so
that at phone width it stays on the chip's line instead of starting the next one. Exact, asserted; the v11 backup is
only read. Applied to thrusterchain.html (md5 checked) and to the fragment source r4_weighting_src.html."""
import hashlib
import re
P = '/mnt/c/Users/David/dev/bca-shift-work/page/'
md5 = lambda p: hashlib.md5(open(p, 'rb').read()).hexdigest()
assert md5(P + 'thrusterchain.v11.html') == '79f00557ba8803625b9f36bc497f7cc6'
assert md5(P + 'thrusterchain.html') == '94d34bb5f72ad0bd41f3ca9fb94b1bef', md5(P + 'thrusterchain.html')
CHIP = re.compile(r'(<span class="iw-chip(?: pass| fail)?">)([^<]*)(</span>): ')
for f, n_exp in ((P + 'thrusterchain.html', 18), (P + 'scripts/r4_weighting_src.html', 18)):
    s = open(f, encoding='utf-8').read()
    a = s.index('<section id="weighting-test">') if 'thrusterchain' in f else s.index('<!-- FRAG:marks -->')
    b = s.index('</section>', a) if 'thrusterchain' in f else s.index('<!-- /FRAG -->', a)
    blk, n = CHIP.subn(r'\1\2:\3 ', s[a:b])
    assert n == n_exp, (f, n)
    s = s[:a] + blk + s[b:]
    open(f, 'w', encoding='utf-8', newline='').write(s)
    print(f, n, 'chips')
assert md5(P + 'thrusterchain.v11.html') == '79f00557ba8803625b9f36bc497f7cc6'
print('page md5', md5(P + 'thrusterchain.html'))

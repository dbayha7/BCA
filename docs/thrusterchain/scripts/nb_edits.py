"""Comment B (notation): the proposed edits as exact, counted replacements, plus the re-typeset math.

  check (default; reads only):
    wsl.exe -d Ubuntu-22.04 -u root -e bash -lc 'python3 /mnt/c/Users/David/dev/bca-shift-work/page/scripts/nb_edits.py'
  build a copy with every edit applied (tmp/nb/b_applied.html), for review and screenshots:
    ... nb_edits.py --copy
  apply to the working page (only when the orchestrator says so; the v11 backup is checked unchanged):
    ... nb_edits.py --write

Every old string must occur exactly `count` times in the page (in its section when one is named); otherwise nothing
is written and the failing edits are listed. Text edits run first, then the math swaps (mathjax: render.js output
mathbuild/out/notation_b.json, ids retagged by mjx.swap). SOURCES lists the matching edits in the tooltip text sources
(ta_texts.py), checked the same way and applied with --write-sources, so a later ta_apply.py run agrees with the page.
"""
import hashlib
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import mjx  # noqa: E402

PAGE = '/mnt/c/Users/David/dev/bca-shift-work/page/thrusterchain.html'
V11 = '/mnt/c/Users/David/dev/bca-shift-work/page/thrusterchain.v11.html'
TMP = '/mnt/c/Users/David/dev/bca-shift-work/page/tmp/nb'
TA_TEXTS = os.path.join(HERE, 'ta_texts.py')
V11_MD5 = None  # read at start; must be unchanged at the end

AHAT_TIP = ('This is the command the critic rates highest at this state, written â and read &quot;a-hat&quot;. '
            'It is the critic&#x27;s own choice, which need not be the truly best command.')

# (id, section, old, new, count). Text in attributes is HTML-escaped as on the page (&#x27; for ').
EDITS = [
    # ---- Methods, "How far the actor moves": a* becomes â, the hill picture goes
    ('B01', 'methods',
     'At one state, picture the critic as a hill over the possible commands. The top of\n'
     '      the hill is the command the critic rates best, its <i>favourite command</i> ',
     'At one state, the critic gives every possible command a value. Write the command it rates\n'
     '      highest as ', 1),
    ('B02', 'methods',
     'data-tip-title="a*, the favourite command" data-tip="This is the command the critic rates best at this state, '
     'the top of its hill (the critic&#x27;s value drawn as a hill over the possible commands)."',
     'data-tip-title="â, the command the critic rates highest" data-tip="' + AHAT_TIP + '"', 1),
    ('B03', 'methods',
     '</span>,</span> and the curvature <span class="sym" data-tip-title="κ, curvature"',
     '</span>,</span> read "a-hat". The curvature <span class="sym" data-tip-title="κ, curvature"', 1),
    ('B04', 'methods',
     '\n      says how steeply the hill falls away from the top. The cloning pull is a second hill, whose top is the recorded\n'
     '      command ',
     '\n      says how fast the critic\'s value falls as a command moves away from it. The cloning pull works the same\n'
     '      way around the recorded command ', 1),
    ('B05', 'methods',
     '</span> and whose steepness is set by the multiplier <span class="nw">',
     '</span>: it is zero at that command and grows with the squared distance from it, at a rate set by the multiplier '
     '<span class="nw">', 1),
    ('B06', 'methods',
     '<i>the critic\'s pull</i>: how hard the critic\'s hill draws the actor\n      toward the favourite command, on the same scale',
     '<i>the critic\'s pull</i>: how hard the critic draws the actor\n      toward the command it rates highest, on the same scale', 1),
    ('B07', 'methods',
     ' of the way from the recorded command toward the critic\'s favourite command. The share is a fraction of that '
     'distance, not a throttle: 0 means the actor\n      keeps the recorded command, 1 means it reaches the favourite.</p>',
     ' of the way from the recorded command toward the command the critic rates highest. The share is a fraction of '
     'that distance, not a throttle: 0 means the actor\n      keeps the recorded command, 1 means it reaches the command '
     'the critic rates highest.</p>', 1),
    ('B08', 'methods',
     '<div><dt><i>a</i>*</dt><dd>It is the critic\'s favourite command, the top of its hill (the critic\'s value drawn '
     'as a hill over the possible commands).</dd></div>',
     '<div><dt><i>â</i></dt><dd>It is the command the critic rates highest at this state, read "a-hat". It is the '
     'critic\'s own choice, which need not be the truly best command.</dd></div>', 1),
    ('B09', 'methods',
     '<dd>It is the curvature of the critic\'s hill: 2.36,',
     '<dd>It is the curvature of the critic\'s value around <i>â</i>: 2.36,', 1),
    ('B10', 'methods',
     '<dd>It is the share of the way toward <i>a</i>* for a critic that curves down past 80%',
     '<dd>It is the share of the way toward <i>â</i> for a critic that curves down past 80%', 1),
    ('B11', 'methods',
     '<i>curves down past 80%, as the exact value does,</i> keeps its hill: the actor goes the share',
     '<i>curves down past 80%, as the exact value does,</i> keeps its curvature there: the actor goes the share', 1),
    ('B12', 'methods',
     ' of the way, but never past the favourite: ',
     ' of the way, but never past the command the critic rates highest: ', 1),
    ('B13', 'methods',
     '<caption>Share of the way from the recorded command toward the critic\'s favourite command, at the critic\'s pull μ = 0.58</caption>',
     '<caption>Share of the way from the recorded command toward the command the critic rates highest, at the critic\'s pull μ = 0.58</caption>', 1),
    ('B14', 'methods',
     'move toward the critic\'s favourite command.</p>',
     'move toward the command the critic rates highest.</p>', 1),
    ('B15', 'methods',
     'which is the\n      favourite command of a critic that believes push keeps rising.',
     'which is the\n      command rated highest by a critic whose values treat push as rising all the way to full throttle.', 1),
    ('B16', 'methods',
     'the actor whose critic knows\n      the stall',
     'the actor whose critic values\n      throttle past 80% correctly', 1),
    # ---- tooltips (data-tip attributes, repeated wherever the symbol is)
    ('B17', None,
     'The curvature κ says how steeply the critic&#x27;s value falls away from its favourite command (the command it '
     'rates best) along each throttle.',
     'The curvature κ says how fast the critic&#x27;s value falls as a command moves away from â, the command the '
     'critic rates highest, along each throttle.', 2),
    ('B18', None,
     'The critic&#x27;s pull μ is how hard the critic draws the actor toward its favourite command (the command it '
     'rates best), on the same scale',
     'The critic&#x27;s pull μ is how hard the critic draws the actor toward â, the command the critic rates highest, '
     'on the same scale', 9),
    ('B19', None,
     'This share is how far the actor moves from the recorded command toward the critic&#x27;s favourite command, for '
     'a critic that curves down past 80%.',
     'This share is how far the actor moves from the recorded command toward â, the command the critic rates highest, '
     'for a critic that curves down past 80%.', 2),
    ('B20', None,
     'the actor moves μ / m of the way toward the favourite command, but never more than the whole way (a share of 1).',
     'the actor moves μ / m of the way toward â, the command the critic rates highest, but never more than the whole '
     'way (a share of 1).', 1),
    ('B21', None,
     'because the actor does not go past the critic&#x27;s favourite command.',
     'because the actor does not go past â, the command the critic rates highest.', 1),
    # ---- column description in the tooltip script (JSON string; apostrophe raw)
    ('B22', None,
     'toward the critic\'s favourite command (0 keeps the recorded command, 1 reaches the favourite)',
     'toward â, the command the critic rates highest (0 keeps the recorded command, 1 reaches â)', 1),
    # ---- Results: the one plain-text use (the Results rewrite may already have replaced this sentence)
    ('B23', 'results',
     'only 0.367 to 0.580 of the way toward its\n      critic\'s favourite command.',
     'only 0.367 to 0.580 of the way toward the command its\n      critic rates highest.', 1),
    # ---- Results, Stage 1: δ* (the critic's exact one-step error, not an optimal quantity) becomes δ^exact
    ('B24', 'results',
     'data-tip-title="δ*, the critic&#x27;s exact one-step error"',
     'data-tip-title="δᵉˣᵃᶜᵗ, the critic&#x27;s exact one-step error"', 3),
    ('B25', 'results',
     '<dt>δ*(<i>x</i>, <i>a</i>)</dt>',
     '<dt>δ<sup>exact</sup>(<i>x</i>, <i>a</i>)</dt>', 1),
]

# The same wording in the tooltip text sources, so that ta_apply.py and the page agree (Python source, raw ').
SOURCES = [
    ('S01', 'tip-176', '"This is the command the critic rates best at this state, the top of its hill (the critic\'s value drawn as a hill over the possible commands)."',
     '"This is the command the critic rates highest at this state, written â and read \\"a-hat\\". It is the critic\'s own choice, which need not be the truly best command."', 1),
    ('S02', 'tip-177', '"The curvature κ says how steeply the critic\'s value falls away from its favourite command (the "\n                      "command it rates best) along each throttle.',
     '"The curvature κ says how fast the critic\'s value falls as a command moves away from â, the "\n                      "command the critic rates highest, along each throttle.', 1),
    ('S03', 'tip-181', 'how hard the critic draws the actor toward its favourite command (the "\n                      "command it rates best), on the same scale',
     'how hard the critic draws the actor toward â, the command the "\n                      "critic rates highest, on the same scale', 1),
    ('S04', 'tip-182', 'toward the critic\'s favourite "\n                      "command, for a critic that curves down past 80%."',
     'toward â, the command "\n                      "the critic rates highest, for a critic that curves down past 80%."', 1),
    ('S05', 'tip-184', 'the actor moves μ / m of the way toward the favourite "\n                "command, but never more',
     'the actor moves μ / m of the way toward â, the command "\n                "the critic rates highest, but never more', 1),
    ('S06', 'tip-185', 'because the actor does not go past the critic\'s favourite "\n                "command."',
     'because the actor does not go past â, the command "\n                "the critic rates highest."', 1),
    ('S07', 'tip-179 legend', '"It is the critic\'s favourite command, the top of its hill (the critic\'s value drawn as a hill over the possible commands).",',
     '"It is the command the critic rates highest at this state, read \\"a-hat\\". It is the critic\'s own choice, which need not be the truly best command.",', 1),
    ('S08', 'tip-179 legend', '"It is the curvature of the critic\'s hill: 2.36,',
     '"It is the curvature of the critic\'s value around <i>â</i>: 2.36,', 1),
    ('S09', 'column "How far the actor goes"', 'toward the critic\'s favourite "\n     "command (0 keeps the recorded command, 1 reaches the favourite)',
     'toward â, the command the "\n     "critic rates highest (0 keeps the recorded command, 1 reaches â)', 1),
]


def md5(p):
    return hashlib.md5(open(p, 'rb').read()).hexdigest()


def section_span(s, sec):
    a = s.find(f'<section id="{sec}"')
    assert a >= 0, f'section {sec} not found'
    b = s.find('\n  </section>\n', a)
    return a, b


def run_edits(s, edits, sections=True):
    bad = []
    for eid, sec, old, new, count in edits:
        lo, hi = section_span(s, sec) if (sections and sec) else (0, len(s))
        n_in, n_all = s[lo:hi].count(old), s.count(old)
        if n_in != count or n_all != count:
            bad.append(f'{eid} ({sec}): expected {count}, found {n_in} in its range and {n_all} in all: {old[:90]!r}')
            continue
        s = s.replace(old, new)
    return s, bad


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else '--check'
    v11 = md5(V11)
    s = open(PAGE, encoding='utf-8').read()
    n_math = s.count('<mjx-container')
    out, bad = run_edits(s, EDITS)
    jobs = json.load(open(f'{mjx.MATHBUILD}/jobs/notation_b.json', encoding='utf-8'))
    R = mjx.load('notation_b')
    for j in jobs:
        try:
            out = mjx.swap(out, j['old'], R[j['key']], count=j['count'])
        except AssertionError as e:
            bad.append(f'math {j["key"]}: {e}')
    src = open(TA_TEXTS, encoding='utf-8').read()
    src_out, src_bad = run_edits(src, [(i, None, o, n, c) for i, _, o, n, c in SOURCES], sections=False)
    print(f'page edits: {len(EDITS)} text + {len(jobs)} math; problems: {len(bad)}')
    for b in bad:
        print('  ', b)
    print(f'ta_texts.py edits: {len(SOURCES)}; problems: {len(src_bad)}')
    for b in src_bad:
        print('  ', b)
    if not bad:
        assert out.count('<mjx-container') == n_math, 'math count changed'
        print('glyph ids unique:', mjx.check(out))
        left = [w for w in ('favourite', 'a*,', '<i>a</i>*', 'δ*', 'hill') if w in out]
        print('left on the page after the edits:', left or 'none of favourite / a* / δ* / hill')
    if mode == '--copy' and not bad:
        os.makedirs(TMP, exist_ok=True)
        open(f'{TMP}/b_applied.html', 'w', encoding='utf-8').write(out)
        print('wrote', f'{TMP}/b_applied.html')
    if mode == '--write' and not bad:
        open(PAGE, 'w', encoding='utf-8').write(out)
        print('wrote the working page')
    if mode == '--write-sources' and not src_bad:
        open(TA_TEXTS, 'w', encoding='utf-8').write(src_out)
        print('wrote ta_texts.py')
    assert md5(V11) == v11, 'v11 backup changed'


if __name__ == '__main__':
    main()

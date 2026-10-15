"""Edit 2 of 4, comment B (notation and informal words), on top of edit 1.

  check (default; reads only):  wsl.exe -d Ubuntu-22.04 -u root -e bash -lc 'python3 /mnt/c/Users/David/dev/bca-shift-work/page/scripts/nb2_apply.py'
  trial copy (tmp/nb2/applied.html):  ... nb2_apply.py --copy
  apply (page, ta_texts.py, ta_code.py, notes/results_draft.html):  ... nb2_apply.py --write

What it does, all as exact replacements with asserted counts (nothing is written unless every edit matches):
  1. nb_edits.EDITS (the notation agent's 25 text edits), with three wordings changed here (B04, B15, B16);
  2. EXTRA: the remaining hill wording, and the other informal words (climbs, nudge, knows, trusts/trusted, wobble,
     jitter, blind) in text, tooltips, legends, tables and the tooltip script;
  3. the 11 re-typeset formulas (mathbuild/out/notation_b.json, made by render.js) and one new copy of the a-hat
     formula (placeholder @@AHAT2@@), each with fresh glyph ids;
  4. the same wording in the tooltip text sources (ta_texts.py, ta_code.py), so the sources and the page agree;
  5. the Results draft for edit 3 (notes/results_draft.html): delta_exact as a superscript, and the same words.
The v11 backup is checked unchanged; the page must be edit 1's output (md5 below) or nothing is written.
"""
import hashlib
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import mjx  # noqa: E402
import nb_edits as NB  # noqa: E402

ROOT = '/mnt/c/Users/David/dev/bca-shift-work/page'
PAGE = f'{ROOT}/thrusterchain.html'
V11 = f'{ROOT}/thrusterchain.v11.html'
V11_MD5 = '79f00557ba8803625b9f36bc497f7cc6'
EDIT1_MD5 = '20e6e437aabf55a973619ed5d546b937'
TA_TEXTS = f'{HERE}/ta_texts.py'
TA_CODE = f'{HERE}/ta_code.py'
DRAFT = f'{ROOT}/notes/results_draft.html'
TMP = f'{ROOT}/tmp/nb2'

AHAT_TITLE = 'â, the command the critic rates highest'
AHAT_TIP = NB.AHAT_TIP

# ---------------------------------------------------------------- 1. the notation agent's edits, three reworded
OVERRIDE = {
    # "moves away from it" was ambiguous; the cloning pull is a penalty, not "the same way" as the critic's value
    'B04': ('\n      says how fast the critic\'s value falls as a command moves away from @@AHAT2@@ The cloning pull is a '
            'penalty\n      centred on the recorded command '),
    'B15': 'which is the\n      command a critic would rate highest if it treated push as rising all the way to full throttle.',
    'B16': 'the actor whose critic rates\n      throttle past 80% correctly',
}
EDITS = [(i, sec, old, OVERRIDE.get(i, new), n) for i, sec, old, new, n in NB.EDITS]

# ---------------------------------------------------------------- 2. the rest of the hill wording and informal words
# (id, section or None, old, new, count). Attribute text is escaped as on the page (&#x27; for ').
EXTRA = [
    # Methods: the last trace of the hill picture
    ('H01', 'methods', '</span>.</span> The actor settles where the two slopes\n      balance:</p>',
     '</span>.</span> The actor settles at the command where the critic\'s value and the pull\'s penalty change at the\n'
     '      same rate, so a small step either way gains nothing:</p>', 1),
    ('H02', 'methods', '<dd>It is the caution multiplier, the steepness of the cloning pull.</dd>',
     '<dd>It is the caution multiplier, which sets the strength of the cloning pull.</dd>', 1),
    # climbs: the actor is trained to raise the critic's value
    ('C01', None, 'Climbs its critic&#x27;s value of its own command while a pull keeps it near the recorded commands',
     'A network trained to raise its critic&#x27;s value of its own command, while a pull keeps it near the recorded commands', 1),
    ('C02', None, 'A network that climbs a critic&#x27;s value of its own command while staying near the recorded commands.',
     'A network trained to raise a critic&#x27;s value of its own command while staying near the recorded commands.', 1),
    ('C03', 'results', '<h4>Stage 1: can caution reach the harm through the value the actor climbs?</h4>',
     '<h4>Stage 1: can caution reach the harm through the value the actor is trained to raise?</h4>', 1),
    ('C04', 'results', 'the critic\'s value of the actor\'s command,\n      which the actor climbs, and the pull',
     'the critic\'s value of the actor\'s command,\n      which the actor is trained to raise, and the pull', 1),
    ('C05', 'results', 'The actor climbs the first estimate.', 'The actor is trained to raise the first estimate.', 2),
    ('C06', 'results', 'Can caution that lowers the value the actor climbs do\n        better?',
     'Can caution that lowers the value the actor is trained to raise do\n        better?', 1),
    ('C07', 'results', '<li><b>Act on the value the actor climbs.</b> In place of a dose on the pull, the actor climbs the critic\'s value',
     '<li><b>Act on the value the actor is trained to raise.</b> In place of a dose on the pull, the actor is trained to raise the critic\'s value', 1),
    # nudge: what the coupling does, in plain words (comment C's rewrite adds the measured size)
    ('N01', 'dynamics', 'data-tip="Each float&#x27;s swing turns a little and shrinks a little, and the floats nudge each other weakly."',
     'data-tip="Each float&#x27;s swing turns a little and shrinks a little, and each float&#x27;s motion moves every other float slightly."', 1),
    ('N02', 'dynamics', 'each float\'s swing turns a little and shrinks a little, and the\n        floats nudge each other weakly.</li>',
     'each float\'s swing turns a little and shrinks a little, and each\n        float\'s motion moves every other float slightly.</li>', 1),
    # knows
    ('K01', 'task', 'reads both wave numbers, so in\n        every state it knows how hard the wave pushes',
     'reads both wave numbers, so in\n        every state its command takes into account how hard the wave pushes', 1),
    ('K02', 'recording-controller', 'Everything it knows comes from a log', 'Everything it learns comes from a log', 1),
    # trusts / trusted
    ('T01', None, 'Does an actor that\n      trusts that critic lose return', 'Does an actor trained\n      against that critic lose return', 1),
    ('T02', 'methods', 'keeps two critics and trusts the\n      lower of their two estimates', 'keeps two critics and uses the\n      lower of their two estimates', 1),
    ('T03', 'results', '<dd>A wrong critic matters only if an actor that trusts it does worse.</dd>',
     '<dd>A wrong critic matters only if an actor trained against it does worse.</dd>', 1),
    ('T04', 'results', 'a critic that is wrong matters only if an actor that trusts it does worse.',
     'a critic that is wrong matters only if an actor trained against it does worse.', 1),
    ('T05', None, 'whether an actor that trusts the critic loses return', 'whether an actor trained against the critic loses return', 1),
    ('T06', None, 'would act if it trusted a critic that never saw the stall.', 'would act if it were trained against a critic that never saw the stall.', 1),
    ('T07', None, 'A learned actor that trusted a critic which never saw the stall would act like this.',
     'A learned actor trained against a critic that never saw the stall would act like this.', 1),
    ('T08', 'recording-controller', 'what a learned actor would do if it trusted a critic that never saw the stall.',
     'what a learned actor would do if it were trained against a critic that never saw the stall.', 1),
    ('T09', None, 'what a learned actor would do if it trusted a critic (its learned estimate',
     'what a learned actor would do if it were trained against a critic (its learned estimate', 1),
    ('T10', 'methods', 'differ only in which critic they trusted, so', 'differ only in which critic they were trained against, so', 1),
    ('T11', 'harm-checks', '<li>The actor that trusted the critic trained after the removal scores at least 5 points below the one\n'
                           '            that trusted the full-log critic',
     '<li>The actor that used the critic trained after the removal scores at least 5 points below the one\n'
     '            that used the full-log critic', 1),
    ('T12', 'results', 'the actor that trusted the critic trained after the removal scores 99\n'
                       '            points lower than the actor that trusted the full-log critic',
     'the actor that used the critic trained after the removal scores 99\n'
     '            points lower than the actor that used the full-log critic', 1),
    # wobble and jitter: the recording controller's random noise
    ('W01', None, 'and adding a random wobble;', 'and adding random noise;', 1),
    ('W02', None, 'and adds a random wobble.', 'and adds random noise.', 1),
    ('W03', None, 'normal driving without its wobble, and never the full-throttle command',
     'normal driving without its random noise, and never the full-throttle command', 1),
    ('W04', 'recording-controller', ' and wobbles: <span class="sym" data-tip-title="n, random numbers"',
     ' and adds random noise: <span class="sym" data-tip-title="n, random numbers"', 1),
    ('W05', None, 'Times 0.3 they are the recording controller&#x27;s wobble, and times 0.03 they are its jitter.',
     'Times 0.3 they are the random noise of the recording controller&#x27;s normal driving, and times 0.03 they are '
     'the small random noise on its full-throttle command.', 1),
    ('W06', 'recording-controller', 'so the wobble has\n      standard deviation 0.3.', 'so the noise has\n      standard deviation 0.3.', 1),
    ('W07', 'recording-controller', 'times 0.3 they are the wobble.</dd>', 'times 0.3 they are the random noise.</dd>', 1),
    ('W08', 'recording-controller', 'the starts, the wobble and the coin flips.', 'the starts, the random noise and the coin flips.', 1),
    ('W09', 'recording-controller', 'it uses fixed streams of wobble and coin flips', 'it uses fixed streams of random noise and coin flips', 1),
    ('W10', 'recording-controller', 'Normal driving without wobble; never the full-throttle command.',
     'Normal driving without random noise; never the full-throttle command.', 1),
    ('W11', 'shift', 'by simulation over its wobble and coin (4,096 draws', 'by simulation over its random noise and coin flip (4,096 draws', 1),
    ('W12', 'results', 'under the recording controller\'s wobble:', 'under the recording controller\'s random noise:', 1),
    ('W13', None, 'normal driving without its wobble and without the full-throttle command.',
     'normal driving without its random noise and without the full-throttle command.', 1),
    ('W14', None, 'damps the swinging weakly, wobbles, stays within 80%', 'damps the swinging weakly, adds random noise, stays within 80%', 1),
    ('J01', None, 'with a little jitter added.', 'with a little random noise added.', 2),
    ('J02', 'recording-controller', '</span> with a little jitter:</p>', '</span> with a little random noise:</p>', 1),
    ('J03', 'recording-controller', '<dd>It is a little jitter: four fresh random numbers', '<dd>It is a little random noise: four fresh random numbers', 1),
    # blind
    ('L01', 'bca-test', 'doses placed blindly and uniform caution of the same strength (the placement check). Winning over blind placements of the same caution\n',
     'doses with their placement removed (spread evenly or shuffled) and uniform caution of the same strength (the placement check). '
     'Winning over the same caution without its placement\n', 1),
    ('L02', None, 'also beats the same caution placed blindly: its own doses', 'also beats the same caution without its placement: its own doses', 1),
    ('L03', 'results', 'Each is tuned, blind to any return, to change', 'Each is tuned, without looking at any return, to change', 1),
    ('L04', 'results', 'so the choice is blind only if nobody works the marks out by\n      hand first.',
     'so the choice is made without knowing the results only if nobody works the marks out by\n      hand first.', 1),
]

# ---------------------------------------------------------------- 4. the tooltip text sources
SRC_TEXTS = [(i, None, o, n, c) for i, _, o, n, c in NB.SOURCES] + [
    ('S10', None, '"full throttle, with a little jitter added.', '"full throttle, with a little random noise added.', 1),
    ('S11', None, '"they are the recording controller\'s wobble, and times 0.03 they are its jitter."',
     '"they are the random noise of the recording controller\'s normal driving, and times 0.03 they are the "\n'
     '                      "small random noise on its full-throttle command."', 1),
    ('S12', None, 'The actor climbs the "\n                      "first estimate."',
     'The actor is trained to "\n                      "raise the first estimate."', 1),
    ('S13', None, '"the others; times 0.3 they are the wobble.",', '"the others; times 0.3 they are the random noise.",', 1),
    ('S14', None, '"It is a little jitter: four fresh', '"It is a little random noise: four fresh', 1),
    ('S15', None, '"It is the caution multiplier, the steepness of the cloning pull.",',
     '"It is the caution multiplier, which sets the strength of the cloning pull.",', 1),
    ('S16', None, 'and adding a random wobble; in the full-throttle "', 'and adding random noise; in the full-throttle "', 1),
    ('S17', None, 'would act if it trusted a critic that never saw the stall."',
     'would act if it were trained against a critic that never saw the stall."', 1),
    ('S18', None, '"recording controller\'s normal driving without its wobble and without the full-throttle command."',
     '"recording controller\'s normal driving without its random noise and without the full-throttle command."', 1),
]
SRC_CODE = [
    ('S19', None, 'also beats the same caution placed blindly: its own doses', 'also beats the same caution without its placement: its own doses', 1),
    ('S20', None, 'whether an actor that trusts the critic loses return', 'whether an actor trained against the critic loses return', 1),
]

# ---------------------------------------------------------------- 5. the Results draft (edit 3 applies it)
DRAFT_EDITS = [
    ('D01', None, r'\delta_{\text{exact}}', r'\delta^{\text{exact}}', 14),
    ('D02', None, '<!-- KEEP: the symbol key for delta and W, with "δ*(x, a)" -> "δ_exact(x, a)" in its first <dt> and in the tooltips\n'
                  '         "δ*, the critic\'s exact one-..." (3 occurrences on the page). -->',
     '<!-- KEEP: the symbol key for delta and W, unchanged (edit 2 already renamed the symbol on the page: its first <dt>\n'
     '         reads "δ<sup>exact</sup>(x, a)" and the three tooltip titles read "δᵉˣᵃᶜᵗ, the critic\'s exact one-step error"). -->', 1),
    ('D03', None, 'Wording change: "critic\'s favourite command" -> "the command its critic rates highest" (comment B).',
     'Wording change (comment B): the command the critic rates highest is named "the command its critic rates highest".', 1),
    ('D04', None, '<li>Does an actor that trusts that critic lose return,', '<li>Does an actor trained against that critic lose return,', 1),
    ('D05', None, 'A wrong critic matters only if an actor that trusts it does worse.',
     'A wrong critic matters only if an actor trained against it does worse.', 1),
    ('D06', None, 'differ only in which critic they trusted, so', 'differ only in which critic they were trained against, so', 1),
    ('D07', None, 'under the recording controller\'s wobble</td>', 'under the recording controller\'s random noise</td>', 1),
    ('D08', None, '<h4>Stage 1: can caution reach the harm through the value the actor climbs?</h4>',
     '<h4>Stage 1: can caution reach the harm through the value the actor is trained to raise?</h4>', 1),
    ('D09', None, 'command, which the actor climbs, and the pull', 'command, which the actor is trained to raise, and the pull', 1),
    ('D10', None, 'Each is tuned, blind to any return, to change', 'Each is tuned, without looking at any return, to change', 1),
    ('D11', None, 'lowers the value the\n          actor climbs do better?', 'lowers the value the\n          actor is trained to raise do better?', 1),
]

INFORMAL = ['favourite', 'favorite', 'hill', 'hills', 'climbs', 'climb', 'nudge', 'nudges', 'believes', 'knows',
            'trusts', 'trusted', 'wobble', 'wobbles', 'jitter', 'blindly', 'blind', 'toy']


def md5(p):
    return hashlib.md5(open(p, 'rb').read()).hexdigest()


def visible_words(s, words):
    """Counts of each word in the page outside math, styles and svg (text, attributes and script strings)."""
    b = mjx.CONTAINER.sub(' ', s)
    b = re.sub(r'<style\b.*?</style>|<svg\b.*?</svg>', ' ', b, flags=re.S)
    return {w: len(re.findall(r'\b' + re.escape(w) + r'\b', b, re.I)) for w in words}


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else '--check'
    assert md5(V11) == V11_MD5, 'v11 backup changed'
    page_md5 = md5(PAGE)
    s = open(PAGE, encoding='utf-8').read()
    n_math = s.count('<mjx-container')
    bad = []
    if page_md5 != EDIT1_MD5:
        bad.append(f'page md5 {page_md5} is not edit 1\'s {EDIT1_MD5}')
    out, b1 = NB.run_edits(s, EDITS)
    out, b2 = NB.run_edits(out, EXTRA)
    bad += b1 + b2
    jobs = json.load(open(f'{mjx.MATHBUILD}/jobs/notation_b.json', encoding='utf-8'))
    R = mjx.load('notation_b')
    for j in jobs:
        try:
            out = mjx.swap(out, j['old'], R[j['key']], count=j['count'])
        except AssertionError as e:
            bad.append(f'math {j["key"]}: {e}')
    if out.count('@@AHAT2@@') == 1:
        sym = (f'<span class="nw"><span class="sym" data-tip-title="{AHAT_TITLE}" data-tip="{AHAT_TIP}">'
               f'{mjx.retag(R["ahat"], out)}</span>.</span>')
        out = out.replace('@@AHAT2@@', sym)
    else:
        bad.append(f'placeholder @@AHAT2@@ found {out.count("@@AHAT2@@")} times')
    src = open(TA_TEXTS, encoding='utf-8').read()
    src_out, b3 = NB.run_edits(src, SRC_TEXTS, sections=False)
    code = open(TA_CODE, encoding='utf-8').read()
    code_out, b4 = NB.run_edits(code, SRC_CODE, sections=False)
    draft = open(DRAFT, encoding='utf-8').read()
    draft_out, b5 = NB.run_edits(draft, DRAFT_EDITS, sections=False)
    bad += [f'ta_texts.py {x}' for x in b3] + [f'ta_code.py {x}' for x in b4] + [f'results_draft.html {x}' for x in b5]
    print(f'page: {len(EDITS)} notation edits + {len(EXTRA)} wording edits + {len(jobs)} formula swaps + 1 new formula')
    print(f'sources: ta_texts.py {len(SRC_TEXTS)}, ta_code.py {len(SRC_CODE)}; draft: {len(DRAFT_EDITS)}')
    print(f'problems: {len(bad)}')
    for x in bad:
        print('   ', x)
    if bad:
        return
    assert out.count('<mjx-container') == n_math + 1, 'formula count is not old + 1'
    print('formulas:', n_math, '->', out.count('<mjx-container'), '; glyph ids unique:', mjx.check(out))
    left = {w: c for w, c in visible_words(out, INFORMAL).items() if c}
    print('informal words left on the page:', left or 'none')
    left_src = {w: c for w, c in visible_words(src_out + code_out, INFORMAL).items() if c}
    print('informal words left in ta_texts.py + ta_code.py:', left_src or 'none')
    d_plain = re.sub(r'<!--.*?-->', ' ', draft_out, flags=re.S)
    left_d = {w: c for w, c in visible_words(d_plain, INFORMAL).items() if c}
    print('informal words left in the Results draft (outside comments):', left_d or 'none')
    stars = [mjx.label(m.group(0)) for m in mjx.CONTAINER.finditer(out) if re.search(r'\*|\\ast|\\star', mjx.label(m.group(0)))]
    print('formulas with a star:', stars or 'none')
    if mode == '--copy':
        os.makedirs(TMP, exist_ok=True)
        open(f'{TMP}/applied.html', 'w', encoding='utf-8').write(out)
        print('wrote', f'{TMP}/applied.html')
    if mode == '--write':
        os.makedirs(TMP, exist_ok=True)
        open(f'{TMP}/thrusterchain.pre-edit2.html', 'w', encoding='utf-8').write(s)
        open(f'{TMP}/ta_texts.pre-edit2.py', 'w', encoding='utf-8').write(src)
        open(f'{TMP}/ta_code.pre-edit2.py', 'w', encoding='utf-8').write(code)
        open(f'{TMP}/results_draft.pre-edit2.html', 'w', encoding='utf-8').write(draft)
        open(PAGE, 'w', encoding='utf-8').write(out)
        open(TA_TEXTS, 'w', encoding='utf-8').write(src_out)
        open(TA_CODE, 'w', encoding='utf-8').write(code_out)
        open(DRAFT, 'w', encoding='utf-8').write(draft_out)
        print('wrote the page, ta_texts.py, ta_code.py and results_draft.html; new page md5', md5(PAGE))
    assert md5(V11) == V11_MD5, 'v11 backup changed'


if __name__ == '__main__':
    main()

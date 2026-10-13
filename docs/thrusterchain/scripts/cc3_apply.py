"""Edit 3 of 4, comment C ("this seems important. explain the coupling and why its weak"), on top of edits 1 and 2.

  check (default; writes only the trial copy tmp/cc3/applied.html):
      wsl.exe -d Ubuntu-22.04 -u root -e bash -lc 'python3 /mnt/c/Users/David/dev/bca-shift-work/page/scripts/cc3_apply.py'
  apply (page and ta_texts.py):  ... cc3_apply.py --write

What it does, all as exact replacements with asserted counts (nothing is written unless every edit matches):
  1. Transition Dynamics: the paragraph "The five swings of A ... Because the coupling is weak, at least 99.66% ..." is
     replaced by a subsection "The floats on their own, and the coupling between them": what the coupling C is, how
     it was drawn and how big it is, a push test with a 5 x 5 table, what it changes in the recorded conditions, that
     no float depends on it, why it is weak (no reason is stated in the records; what its weakness does is measured),
     and how it shifts the swings (leading into the existing swings table). Every number is from
     notes/coupling_facts.json and notes/coupling_thrusters.json, recomputed independently by cc3_facts_check.py.
  2. The symbol C is typeset by mathbuild/render.js (out/coupling_c.json) and gets its own tooltip; the A symbol of the
     old paragraph is reused as it is (same container, same glyph ids).
  3. The section's legend: the A entry names C, and a C entry follows it.
  4. The A bullet links to the new subsection.
  5. The table tooltips: a column-description entry for the new table (COLS["dynamics"]).
  6. ta_texts.py: the A legend text (LEGENDS["tip-113"]) and a record of the new texts (EDIT3_COUPLING).
The v11 backup is checked unchanged; the page must be edit 2's output (md5 below) or nothing is written.
"""
import hashlib
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import mjx  # noqa: E402

ROOT = '/mnt/c/Users/David/dev/bca-shift-work/page'
PAGE = f'{ROOT}/thrusterchain.html'
V11 = f'{ROOT}/thrusterchain.v11.html'
V11_MD5 = '79f00557ba8803625b9f36bc497f7cc6'
EDIT2_MD5 = 'eb6221254504607bf30965e59355151d'
TA_TEXTS = f'{HERE}/ta_texts.py'
TMP = f'{ROOT}/tmp/cc3'
WRITE = '--write' in sys.argv


def md5b(b):
    return hashlib.md5(b).hexdigest()


def once(s, old, new, what):
    n = s.count(old)
    assert n == 1, f'{what}: expected 1 match, found {n}: {old[:100]!r}'
    return s.replace(old, new, 1)


raw = open(PAGE, 'rb').read()
assert md5b(open(V11, 'rb').read()) == V11_MD5, 'the v11 backup has changed'
page_md5 = md5b(raw)
assert page_md5 == EDIT2_MD5, f'the page is not edit 2 output (md5 {page_md5})'
s = raw.decode('utf-8')
pre = s

# ------------------------------------------------------------------ the texts
C_TITLE = 'C, the coupling'
C_TIP = ('C is the coupling, the part of A that links the floats: a fixed 10 × 10 table of small numbers that adds, '
         'each step, at most 2% of the size of the ten float numbers. It is not the current, which is written with a '
         'small c.')
A_LEGEND_OLD = ('It is the 10 × 10 table of how the floats move on their own: each swing turns by θ and shrinks by ρ '
                'every step, and the floats are weakly coupled.')
A_LEGEND_NEW = ('It is the 10 × 10 table of how the floats move on their own: each swing turns by θ and shrinks by ρ '
                'every step, and the coupling <i>C</i> links the floats weakly.')
C_LEGEND = ('It is the coupling, the part of <i>A</i> that links the floats: a fixed 10 × 10 table of small numbers '
            'that adds, each step, at most 2% of the size of the ten float numbers. It is not the current <i>c</i>.')
PUSH_ROW = ('Each row is one test: the speed number of this float gets a single kick of 1, with the thrusters off and '
            'no current or wave, and every float is followed for 20 steps.')


def push_col(k):
    return (f'This is the farthest float {k} gets from rest within 20 steps of the push, as a percentage of the push; '
            f'its distance from rest is the length of its position and speed numbers. Without the coupling it would be '
            f'0%. A dash marks the row in which float {k} is the float pushed.')


COLS_ENTRY = {'cols': {'Push given to': PUSH_ROW, **{f'Float {k} moves': push_col(k) for k in range(1, 6)}}}

# largest swing size of each other float within 20 steps of a unit kick to the pushed float's speed, in % of the push
# (coupling_facts.json, impulse.table, max_size_20; recomputed by cc3_facts_check.py)
FACTS = json.load(open(f'{ROOT}/notes/coupling_facts.json'))
PUSH = [[None if i == j else round(100 * FACTS['impulse']['table'][f'push_on_float_{j + 1}'][i]['max_size_20'], 1)
         for i in range(5)] for j in range(5)]
assert PUSH == [[None, 5.1, 2.8, 1.8, 2.7], [3.3, None, 1.4, 0.4, 0.9], [3.0, 5.9, None, 2.4, 2.2],
                [1.4, 0.7, 2.4, None, 2.4], [0.6, 0.9, 1.0, 2.1, None]], PUSH
# the other numbers the text states, checked against the facts files
M = FACTS['matrices']
assert M['C_spectral_norm'] == 0.02 and M['coupling_attempt'] == 0 and M['C_nonzero_entries'] == 100
assert round(M['C_largest_entry_abs'], 4) == 0.0106 and round(M['C_mean_entry_abs'], 3) == 0.003
assert [round(x['steps_to_halve']) for x in M['swings_with_C']][::4] == [37, 15]
assert [round(x['steps_to_halve']) for x in M['swings_without_C']][::4] == [34, 17]
I = FACTS['impulse']
assert round(I['smallest_cross_within_20']['size'], 3) == 0.004 and round(I['largest_cross_within_20']['size'], 3) == 0.059
assert (I['largest_cross_within_20']['from_float'], I['largest_cross_within_20']['to_float']) == (3, 2)
assert round(I['median_cross_within_20'], 3) == 0.021
assert sorted(x for row in PUSH for x in row if x is not None)[9] <= 2.1     # half of the 20 pairs at 0.021 or less
t12 = I['table']['push_on_float_1'][1]
assert (round(t12['max_size_20'], 3), round(t12['max_size_100'], 3)) == (0.051, 0.052)
grow = [(j, i) for j in range(5) for i in range(5) if i != j and
        round(I['table'][f'push_on_float_{j + 1}'][i]['max_size_100'], 4) > round(I['table'][f'push_on_float_{j + 1}'][i]['max_size_20'], 4)]
assert grow == [(0, 1)], grow
TH = json.load(open(f'{ROOT}/notes/coupling_thrusters.json'))
assert round(max(TH[f'thruster_{k}']['largest_other_one_step_20'] for k in range(1, 5)), 3) == 0.018
rec = FACTS['no_throttle_episodes']['situations']['recorded']
assert [round(100 * f['change_share_of_motion'], 1) for f in rec['per_float']] == [3.7, 9.7, 6.5, 5.6, 8.6]
assert round(100 * rec['all_floats_change_share'], 1) == 6.2
assert (round(rec['mean_return_no_throttle_with_C'], 2), round(rec['mean_return_no_throttle_without_C'], 2)) == (-403.92, -403.44)
assert round(abs(rec['mean_return_change_pct']), 2) == 0.12 and round(rec['largest_episode_return_change_pct'], 1) == 5.6
W = FACTS['wave_per_unit_amplitude'][1]
assert (round(W['peak_with_C'], 2), round(W['peak_without_C'], 2)) == (2.60, 2.37)
R = FACTS['reachability']
assert R['controllable_rank_with_C'] == 10 and R['controllable_rank_without_C'] == 10
assert R['thrusters_pushing_each_float_directly'] == {'float_1': [1], 'float_2': [1, 2], 'float_3': [2, 3],
                                                      'float_4': [3, 4], 'float_5': [4]}

# ------------------------------------------------------------------ 1-2. the paragraph
P_START = '    <p>The five swings of <span class="nw"><span class="sym" data-tip-title="A, the floats on their own"'
P_END = 'how many steps it takes to\n      fade to half.</p>\n'
a = s.index(P_START)
assert s.count(P_START) == 1
b = s.index(P_END, a) + len(P_END)
assert s.count(P_END) == 1
old_par = s[a:b]
assert 'Because the coupling is weak, at least\n      99.66% of each swing' in old_par
A_SYM = re.search(r'<span class="sym" data-tip-title="A, the floats on their own"[^>]*>(<mjx-container.*?</mjx-container>)</span>',
                  old_par, re.S)
assert A_SYM and mjx.label(A_SYM.group(1)) == 'A'
a_sym = A_SYM.group(0)
assert 'data-tip-focus="no"' in a_sym

RC = mjx.load('coupling_c')
c_cont = RC['C']
assert mjx.label(c_cont) == 'C' and 'display="true"' not in c_cont.split('>', 1)[0]
# the renderer's A is byte-identical to the page's A apart from the glyph id number (a check on the settings)
norm = lambda x: re.sub(r'MJX[A-Z]*-\d+-', 'MJX-#-', x)
assert norm(RC['A']) == norm(A_SYM.group(1)), 'the renderer output differs from the page for A'
esc = lambda t: t.replace('&', '&amp;').replace('"', '&quot;').replace("'", '&#x27;').replace('<', '&lt;').replace('>', '&gt;')
c_sym_tmpl = f'<span class="sym" data-tip-title="{esc(C_TITLE)}" data-tip="{esc(C_TIP)}">@@CMJX@@</span>'

def cell(x):
    return '<td class="n">—</td>' if x is None else f'<td class="n">{x:.1f}%</td>'


table = ('<div class="tbl"><table class="stack keep1" style="min-width:520px">\n'
         '<thead><tr><th scope="col">Push given to</th>'
         + ''.join(f'<th class="n" scope="col">Float {k} moves</th>' for k in range(1, 6))
         + '</tr></thead><tbody>\n'
         + ''.join(f'<tr><td>Float {j + 1}</td>' + ''.join(cell(x) for x in PUSH[j]) + '</tr>\n' for j in range(5))
         + '</tbody></table></div>\n')

NEW = f'''    <h3 id="dynamics-coupling">The floats on their own, and the coupling between them</h3>
    <p>The table {a_sym} has two parts. The first part is five separate swings, one per float: every step, each
      float's position and speed numbers turn together by a fixed angle and shrink by a fixed factor. The project chose
      the angles 0.10, 0.20, 0.35, 0.55 and 0.80 radians, and the factors 0.98 for float 1, falling in even steps to
      0.96 for float 5. On its own, this part keeps the floats apart, because a float's next two numbers depend only on
      its own two numbers. The second part, the <b>coupling</b> <span class="nw">{c_sym_tmpl},</span> is what links the
      floats: a 10 × 10 table of small numbers added to the first part, so that every step each float number also
      changes by a small amount that depends on the numbers of every other float.</p>
    <p>The coupling was drawn once at random and then fixed. Its 100 numbers are draws from the bell-shaped (normal)
      distribution, made with seed 20261011 and then scaled together so that, in one step, the coupling adds at most 2%
      of the size of the floats' ten numbers (their length, taken as one list). The draw had one condition: a draw under
      which some motion would grow from step to step would have been replaced by the next one. The first draw met the
      condition, so it was kept. None of its 100 numbers is zero, so the coupling links every float directly to every
      other float, not only to its neighbours in the row. Its largest number is 0.0106, and a typical one is about
      0.003. By comparison, each float's own swing carries 0.96 to 0.98 of its size into the next step, so each step
      the coupling is about 2% of what a float keeps of its own motion.</p>
    <p>A simple test shows how little passes from one float to another. Give one float a single push, a kick of 1 to
      its speed number, with the thrusters off and no current or wave, and follow how far each other float gets from
      rest (the length of its position and speed numbers). Without the coupling, no other float would ever move. With
      it, every other float moves, but only a little: within 20 steps the farthest it gets is between 0.004 and 0.059
      times the push, depending on the pair, and for half of the 20 pairs it is 0.021 times the push or less. The
      largest reach is from float 3 to float 2. Waiting longer adds almost nothing, because the passed-on motion fades
      with the swings: over 100 steps only one pair goes further, float 1 to float 2, from 0.051 to 0.052 times the
      push. The thrusters behave the same way. A thrust of 1 for one step moves the thruster's own two floats by 0.3
      and every other float by at most 0.018, about 6% as much. The table gives every pair as a percentage of the
      push; a dash marks the pushed float itself.</p>
{table}    <p>In the recorded conditions the coupling changes little of what happens. Take the 1,000 fixed starts on which
      every controller is tested in the recorded conditions (that situation's evaluation bank), turn all thrusters off,
      and run each episode twice, once with the coupling and once without it. The floats then move only because of
      where they start, the current and the wave. The difference between the two runs is 3.7% of the motion for float
      1, 9.7% for float 2, 6.5% for float 3, 5.6% for float 4 and 8.6% for float 5, and 6.2% over all floats together.
      Each share divides the typical size of the difference by the typical size of the motion, where a typical size is
      the square root of the average squared size over the 100 steps and the 1,000 starts. Float 2 has the largest
      share, and it is also the float whose response to the wave the coupling changes most: per unit of wave
      amplitude, float 2 swings out to 2.60 from rest with the coupling and 2.37 without it. The return hardly
      changes. With no throttle, the average 100-step return is −403.92 with the coupling and −403.44 without it, a
      difference of 0.12%, and no single episode's return changes by more than 5.6%.</p>
    <p>No float depends on the coupling to be moved. The water pushes every float's speed number directly. Float 1 is
      pushed by thruster 1 and float 5 by thruster 4, and floats 2, 3 and 4 are each pushed by two thrusters. With or
      without the coupling, the four thrusters together can move the floats' ten numbers in any combination of
      directions. The only thing the coupling adds is a path from one float to another.</p>
    <p><b>Why the coupling is weak.</b> The design records state the coupling's size, 0.02 (the 2% above), but give no
      reason for it. The float dynamics, coupling included, come unchanged from the project's earlier test world, where
      0.02 was a default setting. The search that tuned that world's other settings, among them the swings' shrink
      factors, never changed the coupling's size; it only tried four other random draws of the coupling, to check that
      the world's planned properties did not depend on one particular draw. What the weakness does is measured above:
      a push on one float reaches any other float at no more than about 6% of the push, each thruster moves the floats
      it does not push by at most about 6% as much as its own two, and with the thrusters off in the recorded
      conditions the coupling changes each float's motion by 3.7% to 9.7% and the return by 0.12%.</p>
    <p>The coupling also shifts each float's own swing slightly. That is why the table below gives shrink factors from
      0.9817 to 0.9549 and turns from 0.100 to 0.801 radians, rather than the chosen 0.98 to 0.96 and 0.10 to 0.80:
      float 1's swing now fades to half in 37 steps instead of 34, and float 5's in 15 steps instead of 17. Even so, at
      least 99.66% of each swing is in its own float's two numbers. The table gives how much each swing shrinks and
      turns per step, how many steps one swing takes, and how many steps it takes to fade to half.</p>
'''
for bad in ('nudge', 'favourite', 'favorite', 'spectral', 'eigen', 'Gaussian', 'RMS', 'wobble', 'toy', 'jitter'):
    assert bad.lower() not in NEW.lower().replace('data-tip', ''), bad
s = s[:a] + NEW + s[b:]
s = s.replace('@@CMJX@@', mjx.retag(c_cont, s), 1)
assert '@@' not in s

# ------------------------------------------------------------------ 3. the legend
old_a = f'<div><dt><i>A</i></dt><dd>{A_LEGEND_OLD}</dd></div>'
s = once(s, old_a, f'<div><dt><i>A</i></dt><dd>{A_LEGEND_NEW}</dd></div><div><dt><i>C</i></dt><dd>{C_LEGEND}</dd></div>',
         'legend A + C')

# ------------------------------------------------------------------ 4. the A bullet links to the subsection
s = once(s, 'every step, plus a weak coupling\n        between all floats. The coupling was drawn',
         'every step, plus a weak coupling\n        between all floats (see "<a href="#dynamics-coupling">The floats on '
         'their own, and the coupling between them</a>"\n        in this section). The coupling was drawn', 'A bullet link')

# ------------------------------------------------------------------ 5. column descriptions for the new table
s = once(s, '"dynamics":[{"cols":{"Float":',
         '"dynamics":[' + json.dumps(COLS_ENTRY, ensure_ascii=False, separators=(',', ':')) + ',{"cols":{"Float":',
         'COLS dynamics')

# ------------------------------------------------------------------ checks
assert s.count('id="dynamics-coupling"') == 1
n_glyph = mjx.check(s)
assert s.count('<mjx-container') == pre.count('<mjx-container') + 1
for tag in ('p', 'span', 'div', 'table', 'tr', 'td', 'th', 'h3', 'b', 'i', 'a', 'dl', 'dt', 'dd', 'thead', 'tbody'):
    o = len(re.findall(rf'<{tag}[\s>]', s)) - s.count(f'</{tag}>')
    o0 = len(re.findall(rf'<{tag}[\s>]', pre)) - pre.count(f'</{tag}>')
    assert o == o0, f'<{tag}> balance changed: {o0} -> {o}'
ids = set(re.findall(r'\bid="([^"]+)"', s))
bad = sorted({l for l in re.findall(r'href="#([^"]+)"', s) if not l.startswith('MJX') and l not in ids})
assert not bad, bad
sec = s[s.index('<section id="dynamics">'):]
sec = sec[:sec.index('</section>')]
print('dynamics section: paragraphs', sec.count('<p>') + sec.count('<p '), '; tables', sec.count('<table'),
      '; formulas on the page', s.count('<mjx-container'), '; glyph ids', n_glyph)

# ------------------------------------------------------------------ the tooltip source (ta_texts.py)
src = open(TA_TEXTS, encoding='utf-8').read()
src_old = ('        "It is the 10 × 10 table of how the floats move on their own: each swing turns by θ and shrinks by ρ every "\n'
           '        "step, and the floats are weakly coupled.",\n')
src_new = ('        "It is the 10 × 10 table of how the floats move on their own: each swing turns by θ and shrinks by ρ every "\n'
           '        "step, and the coupling <i>C</i> links the floats weakly.",\n')
assert src.count(src_old) == 1 and 'EDIT3_COUPLING' not in src
src = src.replace(src_old, src_new, 1)
src += f'''

# ------------------------------------------------------------------------------------------------ edit 3 (comment C)
# New texts added by cc3_apply.py, already on the page (ta_apply.py does not use them): the tooltip of the symbol C in
# Transition Dynamics, its legend entry (after A's in that section's legend) and the column descriptions of the push
# table (COLS["dynamics"], first entry).
EDIT3_COUPLING = {{
    "tip": ({C_TITLE!r}, {C_TIP!r}),
    "legend": {{"C": {C_LEGEND!r}}},
    "cols": {COLS_ENTRY['cols']!r},
}}
'''
compile(src, TA_TEXTS, 'exec')

os.makedirs(TMP, exist_ok=True)
open(f'{TMP}/applied.html', 'w', encoding='utf-8').write(s)
print('trial copy written:', f'{TMP}/applied.html', md5b(s.encode('utf-8')))
if WRITE:
    if not os.path.exists(f'{TMP}/thrusterchain.pre-edit3.html'):
        open(f'{TMP}/thrusterchain.pre-edit3.html', 'wb').write(raw)
        open(f'{TMP}/ta_texts.pre-edit3.py', 'w', encoding='utf-8').write(open(TA_TEXTS, encoding='utf-8').read())
    open(PAGE, 'w', encoding='utf-8', newline='').write(s)
    open(TA_TEXTS, 'w', encoding='utf-8', newline='').write(src)
    assert md5b(open(V11, 'rb').read()) == V11_MD5
    print('WRITTEN: page md5', md5b(open(PAGE, 'rb').read()), '; ta_texts.py updated')
else:
    print('check only (add --write to apply)')

#!/usr/bin/env python3
"""Page pass 4, edit 2 of 4: the environment sections (draft A with the facts filled in), on top of edit 1.

- Action Space: a lead tying actions to the process, a Gymnasium-style specification table (with where the data sit
  in both tests), the interactive four-throttle widget ("Try a command") with a table view, and where the learning
  methods act on a command;
- Observation Space: a lead, the 13 observed numbers in one table (value at rest, bounds, the middle 90% in each
  situation, where the live panel draws it), and why the critic needs to see the water;
- Starting State: a lead tying the situations to steps 1 and 5, the situations with their usual names, a table with
  their place in the process, "Two ways to rank them" (raw difficulty against harm), and the legend in three groups;
- Transition Dynamics: the step formula's legend in four groups;
- Rewards: a lead, a worked example at one state in both worlds (exact values), and reward to return, true value and
  points;
- Summary of the environment: "Why the stall makes this a test of distribution shift" as its own part, a five-step
  argument with the interactive stall chart; the rest of the old box stays, in a neutral box;
- The task: the drawing's caption as five labelled, colour-matched parts tied to the steps (and the top arrows turn
  gold in the full-throttle region while it plays); a lead and takeaway for the "Thruster 1, state by state" chart,
  which now fills the column and marks the full-throttle region;
- One recorded episode: a lead on the deleted rows, a marker on each deleted column, and a reminder in each column's
  description (the cell tooltips);
- The recording controller: both legends in groups, with a worked example;
- column descriptions (COLS) for every new or rebuilt table; a formula's tooltip shows every group of a grouped legend.

Every replacement is exact and counted. Run from the page directory (WSL):
    python3 scripts/p4_e2_apply.py [--check]      (--check: build in memory, write nothing)
"""
import hashlib
import html
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import mjx  # noqa: E402

PAGE_DIR = os.path.dirname(HERE)
PAGE = os.path.join(PAGE_DIR, "thrusterchain.html")
BEFORE = os.path.join(PAGE_DIR, "tmp", "p4", "edit2.before.html")
CHECK = "--check" in sys.argv
E1_MD5 = "1098ef7267e620d8eebbf5602c7f7b1f"

s = open(PAGE, encoding="utf-8").read()
ORIG = s
assert hashlib.md5(open(BEFORE, "rb").read()).hexdigest() == E1_MD5, "backup changed"
assert hashlib.md5(s.encode("utf-8")).hexdigest() == E1_MD5, "the page is not the edit-1 page this edit was written for"
QV_BLOCKS = [m.group(0) for m in re.finditer(r"<!-- qviewer:begin (\w+) -->.*?<!-- qviewer:end \1 -->|/\* qviewer:begin style \*/.*?/\* qviewer:end style \*/|<script type=\"application/json\" id=\"qv-data\">.*?</script>", s, flags=re.S)]
F = json.load(open(os.path.join(PAGE_DIR, "notes", "p4_facts.json"), encoding="utf-8"))
NV = json.load(open(os.path.join(PAGE_DIR, "tmp", "p4", "edit2", "netvalue.json"), encoding="utf-8"))
N = {"n": 0}
MATH_BEFORE = s.count("<mjx-container")


def rep(old, new, count=1, lo=None, hi=None):
    """exact replacement, asserted count; optionally only inside s[lo:hi]"""
    global s
    if lo is None:
        n = s.count(old)
        assert n == count, f"expected {count} of {old[:100]!r}, found {n}"
        s = s.replace(old, new)
    else:
        part = s[lo:hi]
        n = part.count(old)
        assert n == count, f"expected {count} of {old[:100]!r} in range, found {n}"
        s = s[:lo] + part.replace(old, new) + s[hi:]
    N["n"] += 1


def section(sid):
    a = s.index(f'<section id="{sid}"')
    assert s.count(f'<section id="{sid}"') == 1
    return a, s.index("</section>", a)


def between(start, end, lo, hi, keep_end=True):
    """span of s from `start` up to (not including) `end`, both unique inside s[lo:hi]"""
    part = s[lo:hi]
    assert part.count(start) == 1, (start[:80], part.count(start))
    a = lo + part.index(start)
    assert s[a:hi].count(end) >= 1, end[:80]
    b = s.index(end, a)
    return a, b


def splice(a, b, new):
    global s
    s = s[:a] + new + s[b:]
    N["n"] += 1


def esc(text):
    return html.escape(text, quote=True)


def tip(title, contains=None):
    """the page's own data-tip-title + data-tip pair for a term, reused verbatim"""
    t = esc(title)
    texts = re.findall(r'data-tip-title="%s" data-tip="([^"]*)"' % re.escape(t), ORIG)
    if contains:
        texts = [x for x in texts if contains in html.unescape(x)]
    assert texts, f"no tooltip titled {title!r}" + (f" containing {contains!r}" if contains else "")
    return f'data-tip-title="{t}" data-tip="{texts[0]}"'


def newtip(title, text):
    assert text.endswith(".") and text[0].isupper(), text
    return f'data-tip-title="{esc(title)}" data-tip="{esc(text)}"'


def sym(block, title):
    """the first <span class="sym" ...> for `title` in `block`, with its formula, exactly as the page has it"""
    m = re.search(r'<span class="sym" data-tip-title="%s"[^>]*>\s*<mjx-container.*?</mjx-container></span>' % re.escape(esc(title)),
                  block, flags=re.S)
    assert m, title
    return m.group(0)


def chip(step, text=None):
    names = {"collect": "Step 1", "remove": "Step 2", "train": "Step 3", "band": "Step 4", "test": "Step 5"}
    return f'<span class="stepchip in" data-step="{step}">{text or names[step]}</span>'


def km_sit(key, text):
    return f'<span class="km" data-sit="{key}">{text}</span>'


MINUS = "−"


def num(x, d):
    t = f"{abs(x):.{d}f}"
    return (MINUS if x < 0 and float(t) != 0 else "") + t


def intc(n):
    return f"{int(round(n)):,}"


# =================================================================================================================
# 0. Data for the two figures, from the facts (each check below would stop the edit if a number changed)
# =================================================================================================================
SC = F["stall_chart"]
FW = F["four_throttle_widget"]
MAIN = FW["main_test_seed900"]
IW = FW["weighting_test_expert_seed0"]
E = SC["signed_edges"]
assert len(E) == 51 and abs(E[5] + 0.800000011920929) < 1e-15 and abs(E[45] - 0.800000011920929) < 1e-15
mainF = [MAIN["signed_hist_full"][f"thruster{k}"] for k in range(1, 5)]
mainK = [MAIN["signed_hist_kept"][f"thruster{k}"] for k in range(1, 5)]
iwF = [IW["signed_hist_all"][f"thruster{k}"] for k in range(1, 5)]
iwK = [IW["signed_hist_hold_removed"][f"thruster{k}"] for k in range(1, 5)]
for h, tot in [(mainF, 50000), (mainK, 42500), (iwF, 50000), (iwK, 42500)]:
    assert all(len(x) == 50 and sum(x) == tot for x in h), [sum(x) for x in h]
assert sum(mainF[0][:5]) == 6712 and sum(mainK[0][:5]) == 60 and all(sum(x[:5]) + sum(x[45:]) == 0 for x in mainF[1:])
assert sum(mainF[0][45:]) == 0 and all(sum(x[:5]) + sum(x[45:]) == 0 for x in iwF)
WT1 = SC["weighting_test_expert_seed0_thruster1"]
assert WT1["signed"]["all"] == iwF[0] and WT1["totals"]["hold_rows"] == 4950
IWBALL = F["weighting_test_ball"]["by_seed"]["0"]
MB = MAIN["removed_ball"]
mag = SC["main_test_seed900_thruster1"]["magnitude"]
magF = [mag["full"][2 * i] + mag["full"][2 * i + 1] for i in range(25)]
magK = [mag["kept"][2 * i] + mag["kept"][2 * i + 1] for i in range(25)]
ME = [SC["magnitude_edges"][2 * i] for i in range(26)]
assert sum(magF) == 50000 and sum(magK) == 42500 and sum(magF[20:]) == 6712 and sum(magK[20:]) == 60
assert abs(ME[20] - 0.800000011920929) < 1e-15 and magF[24] == 6080
assert IWBALL["hold_command"] == [-0.8, -0.1096, -0.4679, -0.325]
NET = NV["net_value"]["rows"]
assert len(NET) == 21 and NV["net_value"]["best"] == {"stall": 0.8, "no_stall": 0.8}
ND = NV["normal_driving_rest_c05"]["a"]
assert [round(x, 2) for x in ND] == [-0.41, -0.06, -0.26, -0.18]
PAT = NV["pattern"]
assert [round(x, 3) for x in PAT] == [-1.0, -0.137, -0.585, -0.406]
BALL = NV["ball"]["presets"]
assert round(BALL["capped"]["distance"], 3) == 0.069 and round(BALL["full"]["distance"], 3) == 0.242
assert round(BALL["capped"]["cost"], 3) == 0.350 and round(BALL["full"]["cost"], 3) == 0.458


def pct_pair(P, k):
    q = P[f"thruster{k}"]
    return [q["p5"], q["p95"]]


DATA = {
    "knee": 0.800000011920929,
    "signedEdges": E,
    "main": {"full": mainF, "kept": mainK, "ymax": round(max(max(x) for x in mainF) * 1.04),
             "ball": {"centre": MB["centre"], "radius": MB["radius"]},
             "pct": [pct_pair(MAIN["percentiles_kept"], k) for k in range(1, 5)],
             "past80": [[sum(mainF[k][:5]) + sum(mainF[k][45:]), sum(mainK[k][:5]) + sum(mainK[k][45:])] for k in range(4)],
             "cappedDist": round(BALL["capped"]["distance"], 4)},
    "iw": {"full": iwF, "kept": iwK, "ymax": round(max(max(x) for x in iwF) * 1.04),
           "ball": {"centre": IWBALL["centre"], "radius": IWBALL["radius"]},
           "pct": [pct_pair(IW["percentiles_hold_removed"], k) for k in range(1, 5)],
           "past80": [[0, 0]] * 4, "holds": 4950, "holdsRemoved": IWBALL["holds_removed_training_rows"]},
    "mag": {"full": magF, "kept": magK},
    "magEdges": ME,
    "net": [{"t": r["throttle"], "s": round(r["stall"]["value_minus_80"], 4), "n": round(r["no_stall"]["value_minus_80"], 4)} for r in NET],
    "presets": {"off": [0, 0, 0, 0], "normal": ND, "capped": [-0.8] + PAT[1:], "full": PAT, "hold": IWBALL["hold_command"]},
}

# =================================================================================================================
# 1. Action Space
# =================================================================================================================
lo, hi = section("action-space")
T_PULL = tip("Capped pull")
T_EDGE = tip("Lower edge")
T_BAND = tip("BCA's band")
AS_TOP = f'''<p>An <i>action</i> is the command a controller sends in one step: four throttles, one per thruster. The whole
      learning process turns on it. The <i {tip("Recording controller", "fixed controller that made the training data")}>recording
      controller</i>'s actions are what the data record {chip("collect")}, one cluster of them is what the main test's
      removal deletes {chip("remove")}, choosing them is what the <i {tip("Actor")}>actor</i> learns {chip("train")},
      and BCA's caution acts on them {chip("band")}. The one thing about actions that the data stop showing after the
      removal is what happens past 80% <i {tip("Throttle")}>throttle</i>.</p>
<div class="tbl"><table class="spec as-spec"><caption>The action space at a glance</caption><tbody>
<tr><th scope="row">Action space</th><td><code>Box(-1.0, 1.0, (4,), float32)</code>: 4 numbers, each from −1 to 1.
Gymnasium, maintained by the Farama Foundation, is the standard Python interface for environments in which controllers act
and learn, and <code>Box(low, high, shape, type)</code> is its description of a list of numbers with fixed bounds;
<code>float32</code> is how each number is stored (below).</td></tr>
<tr><th scope="row">Order</th><td>The throttles of thrusters 1, 2, 3 and 4, numbered 0 to 3 in the code.</td></tr>
<tr><th scope="row">Sign</th><td>A negative throttle pushes upstream, against the water, and 0 is off: −1 is full throttle
upstream and 1 full throttle downstream.</td></tr>
<tr><th scope="row">Push</th><td>Equal to the throttle up to 80%. Past 80%, in the <span class="km" data-world="stall">stalling
world</span>, the default and the world the data were recorded in, the propeller stalls: the push falls to 0.65 at 90% and
0.5 at full throttle. In the <span class="km" data-world="no_stall">no-stall comparison world</span> it keeps rising to 1;
that world shows what the same learning costs when the throttles missing from the data hide nothing.</td></tr>
<tr><th scope="row">Throttle cost</th><td>0.3 times the squared throttle of each thruster: 0.192 at 80% and 0.3 at full
throttle. It is charged on the throttle, even while the thruster stalls (<a href="#rewards">Rewards</a>).</td></tr>
<tr><th scope="row">Storage</th><td>Each command is rounded to float32 and then clipped to [−1, 1]; a command of exactly 80%
never stalls (details below).</td></tr>
<tr><th scope="row">The main test's data</th><td>Normal driving stays within ±80%. Only the full-throttle command goes past
it: thruster 1, upstream, in 13.42% of training rows. No other thruster passes 80% in any recorded row, and thruster 1
never goes further downstream than 73%.</td></tr>
<tr><th scope="row">After the main test's removal</th><td><i class="key removed" data-k="region"></i>The densest-15% removal
leaves 60 of the 42,500 kept training rows past 80% (0.141%), all with thruster 1 near full throttle upstream.</td></tr>
<tr><th scope="row">The weighting test's data</th><td>A different recording controller holds thruster 1 at exactly 80% when
the water pushes hard, the <i {tip("80% hold")}>80% hold</i>, and <i class="key removed" data-k="hold"></i>that test's
removal deletes it. Past 80% that test's data hold only uniform random commands: none in the expert data, and in the other
four dataset qualities (the weighting test records five, from random commands up to an expert controller) about 3% to 10%
of training rows push thruster 1 upstream past 80%, and about as many downstream.</td></tr>
</tbody></table></div>
    <p>Each thruster sits between two floats and pushes both: its push, times the push factor 0.3, is added to the speed
      numbers of those two floats in the next state. (Each float is described by two numbers, its position number and its
      speed number, the two coordinates of its swing; both are 0 at rest. See <a href="#observation-space">Observation
      Space</a>.) In the table, the index counts from 0, as the code does, and the observation items are places in the list
      of 13 observed numbers given in the next section.</p>
'''
a, b = between("<p>ThrusterChain is a Gymnasium environment.", '<div class="tbl"><table class="stack keep1" style="min-width:640px">', lo, hi)
splice(a, b, AS_TOP)

# the widget
lo, hi = section("action-space")
BANDS = ["past 80% upstream (−100% to −80%)", "−80% to −60% (−80% included)", "−60% to −40%", "−40% to −20%", "−20% to 0%",
         "0% to 20%", "20% to 40%", "40% to 60%", "60% to 80% (80% included)", "past 80% downstream (80% to 100%)"]


def band_counts(h, k, j):
    return sum(h[k][5 * j:5 * j + 5])


thr_blocks = []
for k in range(1, 5):
    thr_blocks.append(f'''      <div class="aw-thr" data-k="{k}">
        <div class="aw-head"><label for="aw-a{k}">Thruster {k}</label>
          <output id="aw-o{k}" for="aw-a{k}" class="aw-ro"><span class="aw-t">off</span><span>push <b>0.00</b> in the stalling world</span><span>push <b>0.00</b> in the no-stall world</span><span>throttle cost <b>0.000</b></span></output></div>
        <input type="range" id="aw-a{k}" min="-1" max="1" step="0.001" value="0" {newtip(f"Thruster {k}'s throttle", f"Drag to set thruster {k}'s throttle, from −1 (full throttle upstream, against the water) to 1 (full throttle downstream). The arrow keys move it by 1%, and Page Up and Page Down by 10%.")}>
        <svg class="chart aw-strip" id="aw-s{k}" role="img" aria-label="Where thruster {k}'s recorded throttles lie; the table below lists them."></svg>
        <p class="aw-bin" id="aw-b{k}"></p>
      </div>''')
heads2 = "".join(f'<th scope="colgroup" colspan="2">Thruster {k}</th>' for k in range(1, 5))
heads3 = "".join(f'<th class="n" scope="col" data-label="Thruster {k}: all rows">all rows</th>'
                 f'<th class="n" scope="col" data-label="Thruster {k}: rows kept">rows kept</th>' for k in range(1, 5))
rows = []
for j, lab in enumerate(BANDS):
    cells = "".join(f'<td class="n"><span id="aw-t{k + 1}-{j}f">{intc(band_counts(mainF, k, j))}</span></td>'
                    f'<td class="n"><span id="aw-t{k + 1}-{j}k">{intc(band_counts(mainK, k, j))}</span></td>' for k in range(4))
    rows.append(f"<tr><td>{lab}</td>{cells}</tr>")
assert band_counts(mainF, 0, 0) == 6712 and band_counts(mainK, 0, 0) == 60
PRESETS = [
    ("off", "Off", "Off", "All four throttles at 0: no push and no throttle cost."),
    ("normal", "Normal driving", "Normal driving",
     "This is the recording controller's normal driving with the floats at rest, a current of 0.5, no wave and no noise: "
     "throttles of −0.41, −0.06, −0.26 and −0.18, all well inside 80%."),
    ("capped", "Thruster 1 at 80%", "Thruster 1 at 80%",
     "This is the full-throttle pattern with thruster 1 held at exactly 80%: throttles of −0.800, −0.137, −0.585 and −0.406. "
     "It is the hardest push thruster 1 can give without stalling, and its throttle cost is 0.350. With the weighting test's "
     "data chosen, it sets that test's 80% hold instead: −0.800, −0.110, −0.468 and −0.325."),
    ("full", "Full-throttle command", "Full-throttle command",
     "This is the recording controller's full-throttle command without its small noise: throttles of −1.000, −0.137, −0.585 "
     "and −0.406. Thruster 1 stalls in the stalling world, and the throttle cost is 0.458."),
]
pre_btns = "\n".join(f'          <button type="button" class="btn ghost" data-preset="{p}" {newtip(t, d)}>{lab}</button>'
                     for p, lab, t, d in PRESETS)
WIDGET = f'''
    <h3 id="as-try">Try a command</h3>
    <p>Set the four throttles below, or start from one of the four commands on the buttons. For each thruster the readout
      gives the push it really gives in both worlds and its throttle cost. The bars under each slider show where that
      thruster's recorded commands lie, before and after the removal, so you can see which commands the learner still has
      evidence for; a red band marks the throttles past 80%, where a propeller stalls. The box under the sliders says what
      each part of the learning process does at the command you have set.</p>
    <figure class="ix aw" id="as-widget" aria-labelledby="aw-cap">
      <figcaption class="ix-cap" id="aw-cap">Four throttles: the push each gives, what it costs, and where the recorded commands lie</figcaption>
      <div class="ix-ctrls">
        <div class="ix-row" role="group" aria-labelledby="aw-pre-l">
          <span class="ix-gl" id="aw-pre-l">Set a command:</span>
{pre_btns}
        </div>
        <div class="ix-row" role="radiogroup" aria-labelledby="aw-data-l">
          <span class="ix-gl" id="aw-data-l">The bars show the recorded commands of</span>
          <label class="ix-radio"><input type="radio" name="aw-data" value="main" checked> the main test</label>
          <label class="ix-radio"><input type="radio" name="aw-data" value="iw"> the weighting test (expert data, data seed 0)</label>
        </div>
        <p class="stepnote"><span class="stepchip" data-step="collect">Step 1</span><span class="stepchip" data-step="remove">Step 2</span>The
          bars are the recorded data and the part of it that the removal deletes: the densest 15% of the commands in the main
          test, which is the full-throttle cluster, and the 80% hold in the weighting test. A slider sets the kind of command
          that the actor picks in step 3.</p>
      </div>
      <div class="legend aw-legend">
        <span><i class="key tile" style="--c: var(--sit-rec)"></i>recorded commands that the removal kept</span>
        <span><i class="key removed" data-k="region" id="aw-key-rm"></i><span id="aw-key-rm-t">recorded commands that the main test's densest-15% removal deleted</span></span>
        <span><i class="key tile" data-k="stall"></i>past 80%: the propeller stalls</span>
        <span><i class="key line" style="--c: var(--ink)"></i>the throttle you have set</span>
      </div>
      <div class="aw-thrs">
{chr(10).join(thr_blocks)}
      </div>
      <div class="aw-sum" id="aw-sum">
        <p><b>This command:</b> <span id="aw-cost">its four throttles cost 0.000 this step. For comparison, the full-throttle command costs 0.458, and the same pattern with thruster 1 held at 80% costs 0.350.</span></p>
        <p><b>The removed region:</b> <span id="aw-ball">set a command to see whether it lies in the region of commands that the removal deleted.</span></p>
        <p><b>What the learning process does here:</b> <span id="aw-msg">set a command to see which parts of the learning process act there.</span></p>
      </div>
      <p class="p4-sr" aria-live="polite" id="aw-live"></p>
      <details class="aw-tbl"><summary>The recorded throttles as a table</summary><div class="inner">
        <p class="note" id="aw-tnote">Rows of the main test's training data (design seed 900) with each thruster's throttle in each band: all 50,000 rows, and the 42,500 rows kept after the densest-15% removal. A negative throttle pushes upstream.</p>
<div class="tbl"><table class="grouped stack" style="min-width:820px">
<thead><tr class="grp"><th scope="colgroup" colspan="1" class="g0"></th>{heads2}</tr><tr><th scope="col" data-label="Throttle band">Throttle band</th>{heads3}</tr></thead><tbody>
{chr(10).join(rows)}
</tbody></table></div>
      </div></details>
    </figure>

    <h3 id="as-methods">Where the learning methods act on a command</h3>
    <ul class="leads as-methods">
      <li><b>TD3+BC's pull toward the recorded commands</b>{chip("train")} This pull, called behaviour cloning, acts at
        every kept training row: it draws the actor's command toward the command recorded at that row. It can only pull
        toward commands that are in the data.</li>
      <li><b>BCA's <i {T_PULL}>capped pull</i></b>{chip("band")} It is the same pull, made stronger, by a factor from 1 up
        to 1.5, at training rows where BCA's <i {T_BAND}>band</i> is wide. It is a constraint on behaviour, the same kind of
        setting as TD3+BC's own pull, and it acts only at recorded commands.</li>
      <li><b>BCA's <i {T_EDGE}>lower edge</i></b>{chip("band")} Tried in the weighting test's exploratory runs, it lowers the
        critic's value of the actor's own command where the band there is wider than at the recorded command. This is value
        pessimism, the kind of caution that methods such as conservative Q-learning apply (a standard offline learner that
        lowers the values of commands the data do not contain): it acts on the value the actor climbs, at the command the
        actor actually picks.</li>
      <li><b>The risk is the commands with no data.</b> The critic is trained only at recorded commands. Where none are
        left, here thruster 1 past 80% after the main test's removal, it can only extend what it learned nearby, and the
        actor, which picks the commands its critic scores highest, follows that guess.</li>
    </ul>
'''
part = s[lo:hi]
assert part.count("</details>") == 1
i = lo + part.index("</details>") + len("</details>")
splice(i, i, WIDGET)

# =================================================================================================================
# 2. Observation Space
# =================================================================================================================
lo, hi = section("observation-space")
a, b = between("<p>The observation space consists of the following parts (in order):</p>", "<p><b>The bounds are proven, not guessed.</b>", lo, hi)
old = s[a:b]
P_SYM, C_SYM, W1_SYM, W2_SYM = (sym(old, "The 10 float numbers"), sym(old, "c, the current"), sym(old, "w₁, the wave's push now"),
                                sym(old, "w₂, the wave's second number"))
OBS = F["observation_space"]["components"]
SITS = ["recorded", "strong_current", "big_waves", "big_kicks"]
SITN = {"recorded": "Recorded conditions", "strong_current": "Strong current", "big_waves": "Big waves", "big_kicks": "Big starting pushes"}
obs_rows = []
for j in range(10):
    c = OBS[j]
    assert c["index"] == j and c["bounds"] == [-300.0, 300.0] and c["at_rest"] == 0.0
    fl, kind = j // 2 + 1, ("Position" if j % 2 == 0 else "Speed")
    rng = [f'{num(c["ranges"][x]["recording_controller"]["p5"], 2)} to {num(c["ranges"][x]["recording_controller"]["p95"], 2)}' for x in SITS]
    live = (f"the number above float {fl}" + ("; the float is drawn below the dashed rest line when it is negative" if j == 0 else "")
            if j % 2 == 0 else f"the small arrow beside float {fl}")
    obs_rows.append(f'<tr><td>{kind} number of float {fl} (index {j})<span class="obs-live">In the live panel: {live}.</span></td><td class="n">0</td>'
                    + "".join(f'<td class="n">{r}</td>' for r in rng) + "</tr>")
assert OBS[0]["ranges"]["recorded"]["recording_controller"]["p5"] == -0.627
WATER = [
    (10, "Strength of the current, <i>c</i>; drawn at the start, then fixed for the whole episode", "none", "0 to 1.5",
     ["0.1 to 0.7", "0.7 to 1.1", "0.1 to 0.7", "0.1 to 0.7"], "part of the top bar, which shows the water's push <i>c</i> + <i>w</i><sub>1</sub>"),
    (11, "Wave, first number <i>w</i><sub>1</sub> = amplitude × cos(phase): the push the wave adds now; it turns every step",
     "none", "−1 to 1", ["−0.4 to 0.4", "−0.4 to 0.4", "−0.7 to 0.7", "−0.4 to 0.4"],
     "the rest of the top bar: the bar is <i>c</i> + <i>w</i><sub>1</sub>"),
    (12, "Wave, second number <i>w</i><sub>2</sub> = amplitude × sin(phase): the push the wave added a quarter cycle ago; "
     "positive while the push is falling", "none", "−1 to 1",
     ["−0.4 to 0.4", "−0.4 to 0.4", "−0.7 to 0.7", "−0.4 to 0.4"], "not drawn; it tells whether the top bar is about to grow or shrink"),
]
for idx, name, rest, bnd, rng, live in WATER:
    obs_rows.append(f'<tr class="water-row"><td>{name} (index {idx})<span class="obs-live">In the live panel: {live}.</span></td><td>{rest}</td>'
                    + "".join(f'<td class="n">{r}</td>' for r in rng) + "</tr>")
assert F["observation_space"]["situation_start_laws"]["strong_current"]["current"] == [0.7, 1.1]
assert F["observation_space"]["situation_start_laws"]["big_waves"]["wave"] == [0.4, 0.7]
OBS_NEW = f'''<p>An <i>observation</i> is what a controller reads before each step: 13 numbers that describe the whole
      state. Ten describe the five floats and three describe the water. Nothing about the state is hidden, and once the
      start is drawn nothing is random, so the next state follows exactly from these 13 numbers and the command. They are
      also what the learner sees: the <i {tip("Critic")}>critic</i> scores a command from these 13 numbers and the 4
      throttles, and the actor turns these 13 numbers into its 4 throttles {chip("train")}. What a learner cannot read
      from them is how hard a thruster pushes past 80%; it can learn that only from recorded rows past 80%.</p>
    <p>Each float is described by two numbers, its <i {tip("Position number")}>position number</i> and its
      <i {tip("Speed number")}>speed number</i>, interleaved float by float as <span class="nw">{P_SYM}.</span> They are
      the two coordinates of the float's swing, which turns and fades a little every step (<a href="#dynamics">Transition
      Dynamics</a>), and both are 0 when the float is at rest. They are not a literal position and its rate of change:
      under a steady downstream push a float settles with a negative position number and a positive speed number, even
      though it no longer moves. The water is described by the current's strength <span class="nw">{C_SYM},</span> fixed
      for the whole episode, and two wave numbers: <span class="nw">{W1_SYM},</span> the push the wave adds now, and
      <span class="nw">{W2_SYM},</span> the push it added a quarter cycle (5.8 steps) earlier. While <i>w</i><sub>2</sub>
      is positive the wave's push is falling, and while it is negative the push is rising, so together the two wave numbers
      mark the wave's place in its 23.27-step cycle, like the hands of a clock. The wave's <i>amplitude</i>, the largest
      push it adds to the current or takes away, and its <i>phase</i>, where it is in its cycle, are drawn at the start of
      each episode (<a href="#starting-state">Starting State</a>).</p>
    <p>The table gives, for each number, what it describes, its value at rest, and where it lies in each situation. For the float numbers that is the middle 90% of the values (from the 5th to the 95th percentile, the values
      that 5% and 95% of them stay under) over every state at which the recording controller gives a command on the
      situation's 1,000 fixed starts, in the stalling world. For the water numbers it is the exact range the start rule
      allows. Under each name, a line says where the live panel "<a href="#sim">Watch one episode</a>" draws the number.</p>
<div class="tbl"><table class="stack obs-tbl" style="min-width:820px">
<caption>The 13 observed numbers. The observation space is <code>Box(low, high, (13,), float32)</code>: its bounds are −300 to
300 for each of the 10 float numbers (they are proven to stay within 256.4 of rest), 0 to 1.5 for the current and −1 to 1 for
each wave number.</caption>
<thead><tr><th scope="col">Observation</th><th class="n" scope="col">At rest</th>'''
OBS_NEW += "".join(f'<th class="n" scope="col" data-sit="{x}">{SITN[x]}</th>' for x in SITS)
OBS_NEW += f'''</tr></thead><tbody>
{chr(10).join(obs_rows)}
</tbody></table></div>
    <p class="note">The water numbers' ranges follow from the start rule: the current is drawn from the situation's range,
      and each wave number stays within plus or minus the wave's amplitude, which is at most 0.4, or 0.7 in big waves. The
      float numbers' ranges are measured, so they depend on the controller driving; the recording controller is used
      because its states are the ones the data contain (in the recorded conditions the main test's training data agree
      with its column to about 0.02). The live panel draws the water's push as one bar, the current plus the wave.</p>
    <h3 id="obs-why-water">Why the critic needs to see the water</h3>
    <p>The critic estimates the discounted sum of all the rewards that follow a command, and much of each reward is the
      float cost, which depends on how hard the water will push in every later step. The current and the two wave numbers
      fix that push exactly: the current never changes, and the wave turns by the same 0.27 radians every step, so its
      cycle of 23.27 steps can be followed from any state. Without them, two states with the same floats but different
      water would look identical to the critic, although one needs thruster 1 near 80% and the other barely any throttle.
      This is also why the shift can be measured exactly: a state of {km_sit("strong_current", "strong current")} carries
      its current, above the recorded maximum of 0.7, in observation 10.</p>
'''
splice(a, b, OBS_NEW)

# =================================================================================================================
# 3. Starting State
# =================================================================================================================
lo, hi = section("starting-state")
a, b = between("<p>Every episode begins from a <i>start</i>: the 13 numbers of its first state.", '<ul class="leads">', lo, hi)
splice(a, b, f'''<p>Every episode begins from a <i>start</i>: the 13 numbers of its first state, drawn at random each time the
      environment is reset. The rule that draws them is the <i>situation</i>. One situation, the
      {km_sit("recorded", "recorded conditions")}, is the rule the training data were collected under {chip("collect")}.
      The other three, {km_sit("strong_current", "strong current")}, {km_sit("big_waves", "big waves")} and
      {km_sit("big_kicks", "big starting pushes")}, each push one setting beyond anything recorded; they exist only to test
      the learned actor {chip("test")}. A start sets four things, and each one matters for a different reason.</p>
    ''')
lo, hi = section("starting-state")
rep("""so
        in strong current holding often asks for 80% or more, where the stall is.</li>""",
    """so
        in strong current holding often asks for 80% or more, where the stall is. So strong current asks, in most states,
        for exactly the commands that the main test's removal took out of the data.</li>""", lo=lo, hi=hi)
lo, hi = section("starting-state")
rep('Near each crest it reaches further into the <i>full-throttle region</i>,',
    'Near each crest it reaches further into the <i class="km" data-k="region">full-throttle region</i>,', lo=lo, hi=hi)
lo, hi = section("starting-state")
rep("""<p>Each shifted situation changes exactly one of these and leaves the rest as recorded. What each change does to the task is told in the section "The task", under "<a href="#task-situations">How each situation changes the challenge</a>".</p>""",
    f"""<h3 id="ss-situations">The four situations</h3>
    <p>Each shifted situation changes exactly one of these four settings and leaves the rest as recorded. In the usual terms
      of offline reinforcement learning, the recorded conditions are the <i {tip("In distribution")}>in-distribution</i>
      test and the three shifted situations are three different kinds of <i {tip("Out of distribution")}>out-of-distribution</i>
      test. What each change does to the task is told in the section "The task", under "<a href="#task-situations">How each
      situation changes the challenge</a>".</p>""", lo=lo, hi=hi)
lo, hi = section("starting-state")
for old, new in [
    ("<li><b>The recorded conditions</b> change nothing.",
     f'<li><b>{km_sit("recorded", "The recorded conditions")}</b> change nothing, so they are the in-distribution test.'),
    ("<li><b>Strong current</b> changes only the current, to 0.7 to 1.1.",
     f'<li><b>{km_sit("strong_current", "Strong current")}</b> changes only the current, to 0.7 to 1.1.'),
    ("conditions, because its current is above the recorded maximum, 0.7. This is the main test.</li>",
     "conditions, because its current is above the recorded maximum, 0.7. In the usual terms it is a <i>dynamics\n"
     "        extrapolation</i>: a force on the floats beyond any recorded. This is the main test.</li>"),
    ("<li><b>Big waves</b> changes only the wave's size, to 0.4 to 0.7.",
     f'<li><b>{km_sit("big_waves", "Big waves")}</b> changes only the wave\'s size, to 0.4 to 0.7.'),
    ("""amplitude is above the recorded maximum, 0.4. It is a shift in the waves: the water's push ranges further into
        the full-throttle region.</li>""",
     """amplitude is above the recorded maximum, 0.4. In the usual terms it is a <i>disturbance shift</i>: pushes that
        come and go, larger than any recorded, so the water's push ranges further into the full-throttle region.</li>"""),
    ("<li><b>Big starting pushes</b> changes only where the floats start,",
     f'<li><b>{km_sit("big_kicks", "Big starting pushes")}</b> changes only where the floats start,'),
    ("""It is
        a shift in the floats: the starts lie far from the recorded states.</li>""",
     """In the
        usual terms it is an <i>initial-state shift</i>: the starts lie far from the recorded starts.</li>"""),
]:
    rep(old, new, lo=lo, hi=hi)
    lo, hi = section("starting-state")
a, b = between('<p class="note">The four situations side by side.', "<p><b>In symbols.</b>", lo, hi)
assert s[a:b].count("<table") == 1
SIT_ROWS = [
    ("recorded", "Recorded conditions", "0.1 to 0.7", "0.1 to 0.4", "0.5", "in distribution",
     "step 1, where the data were collected, and the in-distribution test of step 5"),
    ("strong_current", "Strong current", "<b>0.7 to 1.1</b>", "0.1 to 0.4", "0.5", "dynamics extrapolation: a force beyond any recorded",
     "step 5, the main test"),
    ("big_waves", "Big waves", "0.1 to 0.7", "<b>0.4 to 0.7</b>", "0.5", "disturbance shift: pushes that come and go, larger than recorded",
     "step 5"),
    ("big_kicks", "Big starting pushes", "0.1 to 0.7", "0.1 to 0.4", "<b>1.5</b>", "initial-state shift: starts unlike the recorded starts",
     "step 5"),
]
sit_tbl = "\n".join(f'<tr data-sit="{k}"><td>{km_sit(k, n)}</td><td>{c}</td><td>{w}</td><td>{sp}</td><td>{u}</td><td>{p}</td></tr>'
                    for k, n, c, w, sp, u, p in SIT_ROWS)
HARM = F["difficulty"]["by_situation"]
assert [round(HARM[x]["harm_points"], 1) for x in ("recorded", "big_kicks", "big_waves", "strong_current")] == [5.8, 11.8, 16.6, 98.9]
assert [HARM[x]["reference_raw_returns"]["stall_aware"] for x in SITS] == [-26.36, -103.79, -36.19, -163.5]
splice(a, b, f'''<p class="note">The four situations side by side. Bold marks the one setting each shifted situation changes.</p>
<div class="tbl"><table class="stack sit-tbl" style="min-width:860px">
<thead><tr><th scope="col">Situation</th><th scope="col">Current: any value from</th><th scope="col">Wave amplitude: any value from</th><th scope="col">Float spread at the start (standard deviation)</th><th scope="col">In the usual terms</th><th scope="col">Where it sits in the process</th></tr></thead><tbody>
{sit_tbl}
</tbody></table></div>
    <h3 id="ss-difficulty">Two ways to rank them</h3>
    <p>The situations can be ranked by how hard they are, or by how much the missing commands hurt in them, and the two
      rankings differ. That difference is the reason strong current is the main test.</p>
    <p><b>By raw difficulty</b>, measured by the best fixed controller's average return (closer to 0 is better), the order is
      {km_sit("recorded", "recorded conditions")} (−26.36), {km_sit("big_waves", "big waves")} (−36.19),
      {km_sit("strong_current", "strong current")} (−103.79) and {km_sit("big_kicks", "big starting pushes")} (−163.50). Big
      starting pushes are the hardest, because the floats start far out and the first steps are costly for any controller
      (<a href="#reference-scores">Reference Scores</a>).</p>
    <p><b>By the harm the missing commands cause</b>, measured as the points that an actor trained against the critic that
      never saw the stall loses against an actor trained against the critic of the full recording (the mean over five
      repeat runs; <a href="#results-cost">the return-cost check</a>), the order is {km_sit("recorded", "recorded conditions")}
      (5.8 points), {km_sit("big_kicks", "big starting pushes")} (11.8), {km_sit("big_waves", "big waves")} (16.6) and
      {km_sit("strong_current", "strong current")} (98.9). Big starting pushes are hard for every controller, but the
      stall-aware controller is at its 80% limit on only 15.5% of their states. Strong current puts it there in most
      states, 71.9% of them, so that is where holding needs exactly the commands the data lost.</p>
    ''')
lo, hi = section("starting-state")
a, b = between('<div class="symkey"><p class="symkey-h">What the symbols mean</p><dl><div><dt><i>s</i><sub>0,<i>i</i></sub></dt>',
               "<p><b>The evaluation banks.</b>", lo, hi)
assert s[a:b].rstrip().endswith("</dl></div>")
splice(a, b, f'''<div class="symkey"><p class="symkey-h">What the symbols mean</p>
      <p class="symkey-g">What a start draws</p>
      <dl><div><dt><i>s</i><sub>0,<i>i</i></sub></dt><dd>It is float number <i>i</i> of the starting state (state 0); the rule draws each of the ten float numbers this way.</dd></div>
      <div><dt><i>c</i></dt><dd>It is the current's strength, drawn once and then fixed for the whole episode.</dd></div>
      <div><dt>φ</dt><dd>It is the wave's phase, where the wave starts in its cycle: an angle from 0 up to, but not including, 2π, one full turn.</dd></div>
      <div><dt><i>w</i><sub>1</sub>, <i>w</i><sub>2</sub></dt><dd>They are the two wave numbers that follow from the phase: amplitude × cos φ, the push the wave adds now, and amplitude × sin φ, the push it added a quarter cycle earlier.</dd></div></dl>
      <p class="symkey-g">How each is drawn</p>
      <dl><div><dt>∼</dt><dd>It means that the value on its left is drawn at random from the distribution on its right.</dd></div>
      <div><dt>𝒩(0, σ<sup>2</sup>)</dt><dd>It is the normal distribution, the familiar bell-shaped spread, with mean 0 and standard deviation σ.</dd></div>
      <div><dt><i>U</i>[low, high]</dt><dd>It is the uniform distribution: every value from low to high is equally likely.</dd></div></dl>
      <p class="symkey-g">The settings that the situations change</p>
      <dl><div><dt>σ</dt><dd>It is the starting spread of the float numbers: 0.5, or 1.5 with {km_sit("big_kicks", "big starting pushes")}.</dd></div>
      <div><dt><i>c</i><sub>low</sub>, <i>c</i><sub>high</sub></dt><dd>They are the ends of the current's range: 0.1 and 0.7 as recorded, and 0.7 and 1.1 in {km_sit("strong_current", "strong current")}.</dd></div>
      <div><dt>amplitude</dt><dd>It is the largest push the wave adds or takes away, drawn uniformly: 0.1 to 0.4 as recorded, and 0.4 to 0.7 in {km_sit("big_waves", "big waves")}.</dd></div></dl></div>
    ''')

# =================================================================================================================
# 4. Transition Dynamics: the step formula's legend, in four groups
# =================================================================================================================
lo, hi = section("dynamics")
a, b = between('<div class="symkey"><p class="symkey-h">What the symbols mean</p><dl><div><dt><i>s</i><sub><i>t</i></sub>, <i>s</i><sub><i>t</i>+1</sub></dt>',
               "</dl></div>", lo, hi)
b += len("</dl></div>")
assert s[a:b].count("<dt>") == 12, s[a:b].count("<dt>")
splice(a, b, '''<div class="symkey"><p class="symkey-h">What the symbols mean</p>
      <p class="symkey-g">The state, now and one step later</p>
      <dl><div><dt><i>s</i><sub><i>t</i></sub></dt><dd>They are the 10 float numbers at step <i>t</i>, observations 0 to 9.</dd></div>
      <div><dt><i>s</i><sub><i>t</i>+1</sub></dt><dd>They are the same 10 numbers one step later.</dd></div></dl>
      <p class="symkey-g">The floats on their own</p>
      <dl><div><dt><i>A</i></dt><dd>It is the 10 × 10 table of how the floats move on their own: each float's swing turns by θ and shrinks by ρ every step, and the coupling <i>C</i> links the floats weakly.</dd></div>
      <div><dt>θ</dt><dd>It is a float's turn per step, in radians: 0.100 for float 1 up to 0.801 for float 5 (the table below lists all five).</dd></div>
      <div><dt>ρ</dt><dd>It is a float's shrink per step: 0.9817 for float 1 down to 0.9549 for float 5.</dd></div>
      <div><dt><i>C</i></dt><dd>It is the coupling, the part of <i>A</i> that links the floats: a fixed 10 × 10 table of small numbers that adds, each step, at most 2% of the size of the ten float numbers. It is not the current <i>c</i>.</dd></div></dl>
      <p class="symkey-g">The thrusters</p>
      <dl><div><dt><i>a</i><sub><i>t</i></sub></dt><dd>They are the four throttles at step <i>t</i>, each limited to −1 to 1.</dd></div>
      <div><dt><i>f</i>(<i>a</i><sub><i>t</i></sub>)</dt><dd>It is each thruster's push for its throttle: equal to the throttle up to 80%, and falling past it in the stalling world (<a href="#action-space">Action Space</a>).</dd></div>
      <div><dt><i>B</i></dt><dd>It is the 10 × 4 table of which floats each thruster pushes, with a 1 where thruster <i>k</i> pushes the speed number of float <i>k</i> or float <i>k</i> + 1.</dd></div>
      <div><dt>0.3</dt><dd>It is the push factor, the same for the thrusters and the water.</dd></div></dl>
      <p class="symkey-g">The water</p>
      <dl><div><dt><i>v</i></dt><dd>It has a 1 at each float's speed number and a 0 at each position number, so the water pushes every float's speed equally.</dd></div>
      <div><dt><i>c</i><sub><i>t</i></sub></dt><dd>It is the current; <i>c</i><sub><i>t</i>+1</sub> = <i>c</i><sub><i>t</i></sub> says that it never changes.</dd></div>
      <div><dt><i>w</i><sub>1,<i>t</i></sub></dt><dd>It is the push the wave adds at step <i>t</i>.</dd></div>
      <div><dt><i>w</i><sub><i>t</i></sub></dt><dd>They are the wave's two numbers (<i>w</i><sub>1</sub>, <i>w</i><sub>2</sub>) at step <i>t</i>, a point that turns around a circle.</dd></div>
      <div><dt><i>R</i>(θ)</dt><dd>It turns a point by any angle, in radians (2π is one full turn), and this angle is not a float's θ. The wave uses <i>R</i>(0.27), which turns it 0.27 radians per step, so one cycle takes 23.27 steps.</dd></div></dl></div>''')

# =================================================================================================================
# 5. Rewards
# =================================================================================================================
lo, hi = section("rewards")
rep("<p>The total reward is: <i>reward</i> = −(<i>float cost</i> + <i>throttle cost</i>).</p>",
    f'''<p>The <i>reward</i> of a step is minus what that step costs: the floats' squared distance from rest plus 0.3 times
      the squared throttles. So the best possible reward is 0, every other reward is negative, and a controller does well
      by keeping the floats near rest without spending much throttle. The reward is the only score the learner ever gets:
      each recorded row keeps it {chip("collect")}, and the critic learns to predict the rewards that follow a command
      {chip("train")}.</p>
    <p>The cost has two parts:</p>''', lo=lo, hi=hi)
lo, hi = section("rewards")
rep("<p>The best possible step scores 0. The dictionary <code>info</code>", '<p class="note">The dictionary <code>info</code>', lo=lo, hi=hi)
lo, hi = section("rewards")
a, b = between("<h3>Return and points</h3>", '<div class="math-display">', lo, hi)
old = s[a:b]
GAMMA = re.search(r'<span class="nw"><span class="sym" data-tip-title="γ, the discount".*?</mjx-container></span>,</span>', old, flags=re.S).group(0)
RW = {(r["throttle"], w): r[w] for r in NET for w in ("stall", "no_stall")}


def rw_row(t, w):
    r = RW[(t, w)]
    return (f'<tr><td>{round(t * 100)}% upstream</td><td class="n">{abs(r["push"]):.3f}</td><td class="n">{0.3 * t * t:.3f}</td>'
            f'<td class="n">{num(r["reward"], 4)}</td><td class="n">{r["float_cost_next"]:.4f}</td><td class="n">{num(r["value"], 4)}</td>'
            f'<td class="n">{"0" if t == 0.8 else num(r["value_minus_80"], 3)}</td></tr>')


assert num(RW[(1.0, "stall")]["value_minus_80"], 3) == "−0.297" and num(RW[(0.9, "stall")]["value_minus_80"], 3) == "−0.131"
assert num(RW[(1.0, "no_stall")]["value_minus_80"], 3) == "−0.029" and num(RW[(0.8, "stall")]["value"], 4) == "−12.0395"
assert num(RW[(1.0, "stall")]["reward"], 4) == "−0.4171" == num(RW[(1.0, "no_stall")]["reward"], 4)
assert [round(x, 3) for x in NV["net_value"]["others"]] == [-0.135, -0.519, -0.321] and round(NV["net_value"]["stall_aware_t1"], 3) == -0.793
EMPTY6 = "<td></td>" * 6
RW_NEW = f'''<h3 id="rw-example">Worked example: why pushing past 80% is a bad trade</h3>
    <p>Take one state inside the recorded conditions: the floats at rest, a current of 0.6, and a wave that adds 0.3 now, so
      the water pushes every float with 0.9. That is above 0.565, so the state is in the
      <span class="km" data-k="region">full-throttle region</span>. Thrusters 2, 3 and 4 give what the stall-aware
      controller gives here (−0.135, −0.519 and −0.321), and thruster 1 is set to 80%, 90% or full throttle upstream; the
      stall-aware controller itself would give 79.3%. The table follows each command in both worlds: the push it gives, this
      step's reward, the float cost of the next state, and its <i>true value</i>, the discounted sum of all the rewards that
      follow, with the stall-aware controller driving after the first step.</p>
<div class="tbl"><table class="stack rw-tbl" style="min-width:780px">
<thead><tr><th scope="col">Thruster 1's throttle</th><th class="n" scope="col">Its push</th><th class="n" scope="col">Its throttle cost</th><th class="n" scope="col">Reward this step</th><th class="n" scope="col">Float cost of the next state</th><th class="n" scope="col">True value of the command</th><th class="n" scope="col">True value against 80%</th></tr></thead><tbody>
<tr class="sub" data-world="stall"><td><span class="km" data-world="stall">Stalling world</span></td>{EMPTY6}</tr>
{rw_row(0.8, "stall")}
{rw_row(0.9, "stall")}
{rw_row(1.0, "stall")}
<tr class="sub" data-world="no_stall"><td><span class="km" data-world="no_stall">No-stall comparison world</span></td>{EMPTY6}</tr>
{rw_row(0.8, "no_stall")}
{rw_row(0.9, "no_stall")}
{rw_row(1.0, "no_stall")}
</tbody></table></div>
    <p>Read the stalling world first. Going from 80% to full throttle lowers this step's reward by 0.108, all of it throttle
      cost, and it lowers the push from 0.8 to 0.5, so the floats drift: the next state's float cost rises from 0.037 to
      0.057, and the drift keeps costing in the steps after. Added up, full throttle is worth 0.297 less than 80%, and 90% is
      worth 0.131 less. In the no-stall comparison world this step's reward is exactly the same, because the reward charges
      the throttle, not the push; but there the push keeps rising, so the floats hardly drift, and full throttle is worth
      only 0.029 less than 80%. One step's reward cannot show the stall; only the steps after it can. That is why the
      critic, which predicts the whole discounted sum, can learn the stall only from recorded rows that show what follows a
      stalled command.</p>
    <p>The same holds across the data. On 3,000 kept training rows in the full-throttle region, the median price of pushing
      past 80% is 0.3574 of true value in the stalling world, and full throttle is worse on 97.2% of the rows; in the no-stall
      comparison world the median price is −0.0065, slightly in favour of full throttle, and it is worse on only 48.2%
      (<a href="#shift">The distribution shift</a>). This is the fact the data lose: the recording controller's full-throttle
      commands are the rows that show it, and the main test's removal deletes them {chip("remove")}.</p>
    <h3 id="rw-return">From reward to return, true value and points</h3>
    <p><b>Return.</b> The <i {tip("Return")}>return</i> of an episode is the sum of its 100 rewards. It is the main score on
      this page, and like every reward it is negative, with closer to 0 better.</p>
    <p><b>Discounted return and true value.</b> Every learner on this page counts later rewards a little less: with the
      discount {GAMMA} a reward <i>t</i> steps ahead counts 0.98<sup><i>t</i></sup> times as much as one now, so a reward
      50 steps ahead counts about 0.36 and one 100 steps ahead about 0.13. The critic estimates such a discounted sum of all
      the rewards that follow a command, and the true value of a command is that sum computed exactly, by simulation, with a
      fixed controller driving afterwards (<a href="#episode-end">Episode End</a> explains how). Because every true value
      can be computed, every error the critic makes can be measured exactly. For one episode the two sums are</p>
    '''
splice(a, b, RW_NEW)
lo, hi = section("rewards")
rep("<p><b>Points.</b> Per situation, on the same 1,000 starts for every controller,</p>",
    "<p><b>Points.</b> A return on its own is hard to judge, because each situation has its own scale. Points place it\n"
    "      between two fixed controllers, per situation and on the same 1,000 starts for every controller:</p>", lo=lo, hi=hi)
lo, hi = section("rewards")
rep("D4RL's normalised score works the\n      same way with other reference levels.</p>",
    "D4RL's normalised score works the\n      same way with other reference levels (D4RL is a standard collection of recorded datasets for offline\n"
    "      reinforcement learning).</p>", lo=lo, hi=hi)
lo, hi = section("rewards")
rep("which scores 100 points.</dd></div></dl></div>\n  </section>",
    "which scores 100 points.</dd></div></dl></div>\n"
    "    <p>In the recorded conditions the two anchors are average returns of −79.80 (the recording controller, 0 points) and\n"
    "      −26.36 (the stall-aware controller, 100 points). So an average return of −53.08, halfway between them, is 50 points,\n"
    "      and doing nothing at all, no throttle, with an average return of −403.92, is −606.5 points. Each situation has its\n"
    "      own two anchors (<a href=\"#reference-scores\">Reference Scores</a>), so a point is worth a different amount of return\n"
    "      in each: 0.53 return units in the recorded conditions and 1.50 in strong current.</p>\n  </section>", lo=lo, hi=hi + 12)
assert abs((-53.08 + 79.80) / (-26.36 + 79.80) * 100 - 50) < 0.01 and round((-403.92 + 79.80) / 53.44 * 100, 1) == -606.5

# =================================================================================================================
# 6. Summary of the environment: why the stall makes this a test of distribution shift, with the stall chart
# =================================================================================================================
lo, hi = section("description")
a, b = between('<div class="callout">\n      <p><strong>Why the stall makes this a test of distribution shift.</strong>',
               "<p><strong>Why this environment, and not the earlier one.</strong>", lo, hi)


def tval(t, w):
    return RW[(t, w)]["value_minus_80"]


T1_ROWS = []
for t in [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0]:
    rs, rn = RW[(t, "stall")], RW[(t, "no_stall")]
    T1_ROWS.append(f'<tr><td>{round(t * 100)}%</td><td class="n">{abs(rs["push"]):.3f}</td><td class="n">{abs(rn["push"]):.3f}</td>'
                   f'<td class="n">{0.3 * t * t:.3f}</td><td class="n">{"0" if t == 0.8 else num(rs["value_minus_80"], 3)}</td>'
                   f'<td class="n">{"0" if t == 0.8 else num(rn["value_minus_80"], 3)}</td></tr>')
MBANDS = [("0% to 20%", 0, 5), ("20% to 40%", 5, 10), ("40% to 60%", 10, 15), ("60% to 76%", 15, 19), ("76% to 80% (80% included)", 19, 20),
          ("80% to 88%", 20, 22), ("88% to 96%", 22, 24), ("96% to 100%", 24, 25)]
T2_ROWS = []
for lab, i0, i1 in MBANDS:
    f, k = sum(magF[i0:i1]), sum(magK[i0:i1])
    T2_ROWS.append(f'<tr><td>{lab}</td><td class="n">{intc(f)}</td><td class="n">{100 * f / 50000:.2f}%</td><td class="n">{intc(k)}</td>'
                   f'<td class="n">{100 * k / 42500:.2f}%</td><td class="n">{intc(f - k)}</td></tr>')
# the words of the chart's text summary, checked against the bins
fine = mag["full"]
lowshare = [100 * c / 50000 for c in fine[:30]]
assert 2.2 <= min(lowshare) and max(lowshare) <= 2.75, (min(lowshare), max(lowshare))
assert all(fine[i] > fine[i + 1] for i in range(29, 39)), fine[29:40]   # thinning out from 60% to 80%
mid = sum(fine[40:48])
rm50 = sum(mag["full"][25:40]) - sum(mag["kept"][25:40])
assert mid == 632 and sum(fine[48:]) == 6080 and rm50 == 848 and sum(mag["kept"][:25]) == sum(mag["full"][:25])
SC_WORDS = (f"In the full data, thruster 1's throttle sizes up to 60% are spread almost evenly, {min(lowshare):.1f}% to "
            f"{max(lowshare):.1f}% of the rows in each step of 2%, and they thin out from 60% to 80%, because normal driving's "
            f"squashing keeps it inside 80%. Every row past 80% is a full-throttle command: {mid} lie between 80% and 96%, and "
            f"6,080 between 96% and 100%. After the removal nothing below 50% is lost, {rm50} rows are gone between 50% and 80%, "
            f"and the 60 rows left past 80% all lie between 96% and 100%.")
STALL = f'''<h3 id="why-stall">Why the stall makes this a test of distribution shift</h3>
    <p>A <i {tip("Distribution shift")}>distribution shift</i> is a difference between the data a controller learns from and
      the situations it meets when it is tested. ThrusterChain builds one on purpose, out of one physical fact, the stall, so
      that every step of the argument can be checked exactly. The argument has five steps, and the chart below shows the
      first two.</p>
    <p><b>1. Past 80%, more throttle buys less push and costs more.</b> Up to 80% a thruster pushes exactly as hard as its
      throttle. Past 80%, in the <span class="km" data-world="stall">stalling world</span>, its propeller
      <span class="term" {tip("The stall", "0.65 at 90%")}>stalls</span>: the push falls to 0.65 at 90% and 0.5 at full
      throttle, while the throttle cost keeps rising, from 0.192 at 80% to 0.243 at 90% and 0.3 at full throttle. So going
      from 80% to full throttle cuts the push by 37.5% and raises the throttle cost by 56%, and the lost push lets the floats
      drift, which raises the float cost of every later step too. The hardest push a thruster can give is 0.8, at exactly
      80%. In the <span class="km" data-world="no_stall">no-stall comparison world</span> the push keeps rising to 1, and
      nothing is lost.</p>
    <p><b>2. The full recording shows the stall; the data after the removal do not.</b> The recording controller's most
      common single command is the full-throttle command, which puts thruster 1 at 100%: 13.42% of the training rows, and
      every row past 80% is one of them {chip("collect")}. These rows are the only record of what happens past 80%. The main
      test's removal, the densest-15% removal, deletes the cluster around that command, 99.11% of the full-throttle commands
      {chip("remove")}. What is left holds 60 rows past 80% out of 42,500, 0.141%. The states themselves are not lost: 46.54%
      of the rows in the <span class="km" data-k="region">full-throttle region</span> stay, now with normal-driving
      commands. What is lost is one kind of command at states the data still cover. And because the two worlds are identical
      up to 80%, 99.859% of the kept rows follow the identical rule in both, so almost nothing in them can tell a stalling
      propeller from one whose push keeps rising.</p>
    <figure class="ix sc" id="stall-chart" aria-labelledby="sc-cap">
      <figcaption class="ix-cap" id="sc-cap">Push and throttle cost against the size of the throttle, in both worlds, with thruster 1's recorded throttles from the main test</figcaption>
      <div class="ix-ctrls">
        <div class="ix-row">
          <label class="ix-gl" for="sc-a">Throttle size</label>
          <input type="range" id="sc-a" min="0" max="1" step="0.01" value="0.9" {newtip("Throttle size", "Drag to set the size of one thruster's throttle, from off (0%) to full throttle (100%). The arrow keys move it by 1%, and Page Up and Page Down by 10%.")}>
        </div>
        <div class="ix-row" role="group" aria-labelledby="sc-tg-l">
          <span class="ix-gl" id="sc-tg-l">Show thruster 1's recorded throttles:</span>
          <label class="ix-check"><input type="checkbox" id="sc-full" checked> in the full data</label>
          <label class="ix-check"><input type="checkbox" id="sc-kept" checked> after the main test's removal</label>
        </div>
        <p class="stepnote"><span class="stepchip" data-step="collect">Step 1</span><span class="stepchip" data-step="remove">Step 2</span>The
          full data are what the recording controller collected; what is left after the removal is what the learner trains on.
          The main test's removal, the densest 15% of the commands, deletes the full-throttle cluster. With both boxes ticked,
          the part it deleted is hatched.</p>
      </div>
      <div class="readouts">
        <div class="ro"><span class="k">throttle size</span><span class="v" id="sc-va">90%</span></div>
        <div class="ro"><span class="k"><i class="key line" data-world="stall"></i>push, stalling world</span><span class="v" id="sc-vs">0.65<span class="st">stalls</span></span></div>
        <div class="ro"><span class="k"><i class="key dash" data-world="no_stall"></i>push, no-stall world</span><span class="v" id="sc-vn">0.90</span></div>
        <div class="ro"><span class="k"><i class="key dots"></i>throttle cost</span><span class="v" id="sc-vc">0.243</span></div>
      </div>
      <p class="ix-line" id="sc-cmp">Against 80%: in the stalling world the push is 0.15 lower, while in the no-stall world it is 0.10 higher; the throttle cost is 0.051 higher in both.</p>
      <p class="ix-line" id="sc-val">True value at one example state (floats at rest, a current of 0.6, the wave adding 0.3), computed every 5% of throttle: at 90% thruster 1 is worth 0.131 less than at 80% in the stalling world and 0.008 less in the no-stall world.</p>
      <p class="ix-line" id="sc-bin">Thruster 1's recorded throttles from 88% to 92% in size: {intc(magF[22])} of the 50,000 rows in the full data, and {intc(magK[22])} of the 42,500 rows kept after the main test's removal.</p>
      <svg class="chart sc-svg" id="sc-svg" role="img" aria-label="Two panels share one horizontal axis, the size of the throttle from 0% to 100%. Top: push and throttle cost of one thruster. Both worlds push as hard as the throttle up to 80%; past 80% the stalling world's push falls to 0.65 at 90% and 0.5 at full throttle, while the no-stall world's keeps rising to 1, and the throttle cost rises from 0 to 0.3 in both. Bottom: how many of the main test's training rows have thruster 1's throttle at each size. {esc(SC_WORDS)}"></svg>
      <div class="legend sc-legend">
        <span><i class="key line" data-world="stall"></i>push, stalling world</span>
        <span><i class="key dash" data-world="no_stall"></i>push, no-stall comparison world</span>
        <span><i class="key dots"></i>throttle cost</span>
        <span><i class="key tile" data-k="stall"></i>stall region: past 80%</span>
        <span><i class="key tile" style="--c: var(--sit-rec)"></i>thruster 1's recorded throttles</span>
        <span><i class="key removed" data-k="region"></i>deleted by the main test's removal (both boxes ticked)</span>
      </div>
      <p class="note">In words: the push equals the throttle up to 80% in both worlds. Past 80% the stalling world's push
        falls (0.65 at 90%, 0.5 at full throttle) while the throttle cost keeps rising (0.243 at 90%, 0.3 at full throttle).
        In the main test's full training data 13.42% of thruster 1's recorded throttles lie past 80%; after the removal,
        0.141% do (60 of 42,500 rows). {SC_WORDS}</p>
      <details class="sc-tbl"><summary>The numbers as tables</summary><div class="inner">
        <p class="note">Push, throttle cost and true value at each throttle. The true value is computed at one example state
          (floats at rest, a current of 0.6, the wave adding 0.3, thrusters 2 to 4 at the stall-aware controller's command),
          with the stall-aware controller driving afterwards, and given against thruster 1 at 80%.</p>
<div class="tbl"><table class="stack" style="min-width:720px">
<thead><tr><th scope="col">Throttle</th><th class="n" scope="col">Push, stalling world</th><th class="n" scope="col">Push, no-stall comparison world</th><th class="n" scope="col">Throttle cost</th><th class="n" scope="col">True value against 80%, stalling world</th><th class="n" scope="col">True value against 80%, no-stall comparison world</th></tr></thead><tbody>
{chr(10).join(T1_ROWS)}
</tbody></table></div>
        <p class="note">Thruster 1's recorded throttles in the main test's training data (design seed 900), by size, before and
          after the densest-15% removal.</p>
<div class="tbl"><table class="stack" style="min-width:680px">
<thead><tr><th scope="col">Size of the throttle</th><th class="n" scope="col">Full data: rows</th><th class="n" scope="col">Full data: share</th><th class="n" scope="col">After the removal: rows</th><th class="n" scope="col">After the removal: share</th><th class="n" scope="col">Rows deleted</th></tr></thead><tbody>
{chr(10).join(T2_ROWS)}
</tbody></table></div>
      </div></details>
    </figure>
    <p><b>3. The critic fills the gap with a guess, and the guess is wrong.</b> The critic is trained only at recorded commands
      {chip("train")}. Below 80%, in hard water, more throttle means more push and floats nearer rest, so a higher value. With
      almost nothing past 80% to correct it, the critic extends that trend: compared with thruster 1 at 80%, its value rises by
      0.04 at 90% and by 0.06 at full throttle, while the true value falls by 0.21 and by 0.46. The critic trained on the full
      recording, which saw the stall, does not prefer full throttle: its extra preference for full throttle over thruster 1 at
      80%, beyond what the true values say, is −0.06 to −0.01 across the five repeat runs, against +0.33 to +0.36 for the
      critic trained after the removal. The actor picks the commands its critic scores highest, so it learns to push past 80%
      at 42% of the training states in the full-throttle region, against 10% for the actor of the full-recording critic.</p>
    <p><b>4. Strong current asks for exactly those commands.</b> In {km_sit("strong_current", "strong current")} the current,
      from 0.7 to 1.1, is stronger than any recorded. Holding against the current alone asks thruster 1 for 80% once the
      current exceeds 0.879 with the floats at rest, or 0.844 once they have settled; 93.7% of strong-current states lie in
      the full-throttle region, and the stall-aware controller sits at its 80% limit on 71.9% of them
      (<a href="#task-strong">Why strong current is the hard case</a>). The question the data can no longer answer, what
      happens past 80%, is the question strong current asks in most states {chip("test")}.</p>
    <p><b>5. The result is large, measurable harm.</b> The actor trained against the critic that never saw the stall pushes
      past 80% on 73% of the strong-current states it visits and scores −14.3 points, against 84.6 for the actor of the
      full-recording critic. With its commands capped at 80%, the same actor scores 101.8, so almost all of the loss comes
      from that one wrong belief. A fixed controller that assumes the push keeps rising, the rising-thrust controller, scores
      −54.1 points there, below the recording controller.</p>
    <p>In the usual terms of offline reinforcement learning, this is <i {tip("Extrapolation error")}>extrapolation error</i>,
      a critic's error at commands the data do not contain, meeting a dynamics extrapolation, a test situation beyond the
      recorded ones. It is the error that BCA's band would have to catch {chip("band")}. The no-stall comparison world is the
      control: data made the same way, the same removal and the same learner, but there the missing commands hide nothing, so
      any harm that remains is not caused by the stall.</p>
    <div class="box env-history">
      '''
splice(a, b, STALL)

# =================================================================================================================
# 7. The task: the drawing's caption as five labelled parts; the chart's lead, width and region mark
# =================================================================================================================
lo, hi = section("task")
a, b = between('<figcaption>The water pushes every float with the same push', "</figcaption>", lo, hi)
b += len("</figcaption>")
old = s[a:b]
CW1, C1, W11 = sym(old, "c + w₁, the water's push"), sym(old, "c, the current"), sym(old, "w₁, the wave's push now")
splice(a, b, f'''<figcaption class="td-cap">
        <p class="td-k">The drawing has five parts. Each one feeds a step of the learning process, and each keeps its colour on the whole page.</p>
        <dl class="td-parts">
          <div style="--c: var(--water-ink)"><dt><i class="key line"></i>The water</dt><dd>It pushes every float with the same push <span class="nw">{CW1}:</span> a steady current {C1} (the solid part of each top arrow) plus a wave {W11} that rises and falls every 23.27 steps (the dashed part). The sign beside each arrow is + when the wave adds to the current's push and − when it takes from it; the faint dashed part is then the push it takes away. <b>In the process:</b> the water is what the situations change {chip("test")}. Strong current raises the current and big waves enlarge the wave, so this is where the test leaves the recorded conditions.</dd></div>
          <div data-k="region"><dt><i class="key tile" data-k="region"></i>The full-throttle region</dt><dd>These are the states where the water's push is above 0.565; while the drawing plays, the top arrows turn gold in them. <b>In the process:</b> there the recording controller flips a coin in each state and on half of them gives the full-throttle command {chip("collect")}, and those are the commands that the main test's removal deletes {chip("remove")}.</dd></div>
          <div style="--c: var(--ink)"><dt><i class="key ring"></i>The floats</dt><dd>Each float is drawn where it is in the state shown, all five to the same scale. Each swings back and forth around rest, a little less each time, with its <span class="term" {tip("Period")}>own period</span>, the steps one swing takes (rounded to whole steps under it). With the thrusters off, a steady current would carry float 1 furthest from rest, so thruster 1 works hardest. <b>In the process:</b> their squared distance from rest is the float cost, part of every step's cost, and the reward, minus that cost, is what every learner tries to raise (<a href="#rewards">Rewards</a>).</dd></div>
          <div style="--c: var(--accent)"><dt><i class="key tile"></i>The thrusters</dt><dd>Each sits between two floats and pushes both, so floats 1 and 5 have one thruster each and the middle floats two. A negative throttle pushes upstream, and each arrow is as long as the push it really gives. <b>In the process:</b> their four throttles are the action, the thing the actor learns to choose {chip("train")}.</dd></div>
          <div data-k="stall"><dt><i class="key tile" data-k="stall"></i>The stall</dt><dd>The bar at the bottom fills to thruster 1's throttle, and its dark tick marks the push it really gives; the end of the bar past 80% is drawn in red. Past 80% the propeller stalls: the push falls to 0.65 at 90% and 0.5 at full throttle, while the cost keeps rising. <b>In the process:</b> this is what the data after the removal barely show, so it is where the critic must guess {chip("train")} and where BCA's band would have to look {chip("band")}.</dd></div>
        </dl>
      </figcaption>''')
lo, hi = section("task")
rep("<div><dt>Bar and dark tick</dt><dd>Thruster 1's throttle (the bar's fill) and the push it really gives (the tick).</dd></div></dl></div>",
    "<div><dt>Bar and dark tick</dt><dd>Thruster 1's throttle (the bar's fill) and the push it really gives (the tick).</dd></div>"
    '<div><dt><i class="key tile" data-k="region"></i>gold arrows</dt><dd>While the drawing plays, the top arrows turn gold when the '
    "water's push is above 0.565: the state is then in the full-throttle region.</dd></div></dl></div>", lo=lo, hi=hi)
# the drawing script: the water's arrows wear the region's gold in the full-throttle region, and the readout says so
rep("""      set(wave, { y1: wy1, y2: wy2 }); show(wave, Math.abs(wy2 - wy1) > 1.5); cls(wave, "faint", faint);""",
    """      set(wave, { y1: wy1, y2: wy2 }); show(wave, Math.abs(wy2 - wy1) > 1.5); cls(wave, "faint", faint);
      [solid, wave, head].forEach(e => cls(e, "reg", push > 0.565));   // the full-throttle region, in its gold""")
rep("""        "wave " + Math.abs(push - WAT.c).toFixed(3));""",
    """        "wave " + Math.abs(push - WAT.c).toFixed(3) + (push > 0.565 ? ": full-throttle region" : ""));""")

# the chart: a lead and takeaway, a shorter caption, the region band and a wide drawing for wide columns
lo, hi = section("task")
TC_LEAD = f'''<h3 id="task-chart-h">Thruster 1, state by state</h3>
    <p>The chart below asks one question of two fixed controllers: as the water pushes harder, how hard does thruster 1 push
      back? It pools every state of the four evaluation banks in the stalling world, 400,000 states in all, and sorts them by
      the water's push at the state, <i>c</i> + <i>w</i><sub>1</sub>, into bands 0.2 wide. Each dot is one band's average for
      one controller. Across the chart is the water's push, from −0.6 (upstream) to 1.6. Up the chart is thruster 1's
      throttle or push, where negative means upstream, so higher on the chart means a harder push against the water. Two
      marks matter. The gold band from 0.565 on is the <span class="km" data-k="region">full-throttle region</span>, where the
      recording controller sends the full-throttle command in half of the states. The red dashed line at −0.8 is the 80%
      limit, past which the propeller <span class="km" data-k="stall">stalls</span>.</p>
    <p>The takeaway: the <span class="km" data-k="stall_aware">stall-aware controller</span> pushes harder as the water pushes
      harder and stops at the 80% line. The <span class="km" data-k="rising_thrust">rising-thrust controller</span> keeps
      asking for more, up to full throttle, but once it stalls the push it really gives falls back, to −0.51 at the strongest
      pushes. That second line is how a learned actor acts when its critic never saw the stall. The full-throttle region also
      starts well before holding needs 80%: where the water's push is 0.6 to 0.8, the stall-aware controller is at its limit on
      only 22% of states, and where it is 1.0 to 1.2, on 97%.</p>
    '''
rep('<figure class="chartbox" id="task-chart">', TC_LEAD + '<figure class="chartbox" id="task-chart">', lo=lo, hi=hi)
a = s.index('<figure class="chartbox" id="task-chart">')
svg_a = s.index("<svg", a)
svg_b = s.index("</svg>", svg_a) + len("</svg>")
NARROW = s[svg_a:svg_b]
assert NARROW.startswith('<svg class="chart" viewBox="0 0 440 290"')
X565 = 158.4 + 167.2 * 0.565
REG = (f'<rect class="tc-reg" x="{X565:.1f}" y="18" width="{426 - X565:.1f}" height="232"/>')
REGL = (f'<line class="k565" x1="{X565:.1f}" x2="{X565:.1f}" y1="18" y2="250"/>'
        f'<text class="ann" x="{X565 + 5:.1f}" y="244" text-anchor="start">full-throttle region →</text>')
narrow_new = NARROW.replace('<svg class="chart" viewBox="0 0 440 290"', '<svg class="chart tc-narrow" viewBox="0 0 440 290"', 1)
gt = narrow_new.index(">") + 1
narrow_new = narrow_new[:gt] + REG + narrow_new[gt:]
ax = '<line class="axis" x1="58" x2="58" y1="18" y2="250"/>'
assert narrow_new.count(ax) == 1
narrow_new = narrow_new.replace(ax, ax + REGL)


def wide_svg(src, size="wide"):
    """the same chart drawn 860 wide (or 360 wide for a phone): every plot coordinate mapped from the 440-wide drawing
    (x 58..426, y 18..250)"""
    phone = size == "phone"
    VW, VH = (360, 330) if phone else (860, 372)
    L, R, T, B = (62, 352, 22, 262) if phone else (66, 846, 22, 312)
    fx = lambda x: L + (x - 58) * (R - L) / (426 - 58)
    fy = lambda y: T + (y - 18) * (B - T) / (250 - 18)
    out = [f'<svg class="chart tc-{size}" viewBox="0 0 {VW} {VH}" role="img" aria-label="' +
           re.search(r'aria-label="([^"]*)"', src).group(1) + '">']
    out.append(f'<rect class="tc-reg" x="{fx(X565):.1f}" y="{T}" width="{R - fx(X565):.1f}" height="{B - T}"/>')
    for m in re.finditer(r'<line class="grid" x1="58" x2="426" y1="([\d.]+)" y2="[\d.]+"/><text class="tick" x="52" y="[\d.]+" text-anchor="end">([^<]*)</text>', src):
        y = fy(float(m.group(1)))
        out.append(f'<line class="grid" x1="{L}" x2="{R}" y1="{y:.1f}" y2="{y:.1f}"/><text class="tick" x="{L - 7}" y="{y + 4:.1f}" text-anchor="end">{m.group(2)}</text>')
    ticks = re.findall(r'<text class="tick" x="([\d.]+)" y="266" text-anchor="middle">([^<]*)</text>', src)
    assert len(ticks) == 5
    for x, t in ticks:
        out.append(f'<text class="tick" x="{fx(float(x)):.1f}" y="{B + 18}" text-anchor="middle">{t}</text>')
    out.append(f'<line class="axis" x1="{L}" x2="{R}" y1="{B}" y2="{B}"/><line class="axis" x1="{L}" x2="{L}" y1="{T}" y2="{B}"/>')
    out.append(f'<line class="k565" x1="{fx(X565):.1f}" x2="{fx(X565):.1f}" y1="{T}" y2="{B}"/>'
               f'<text class="ann" x="{fx(X565) + 6:.1f}" y="{B - 8}" text-anchor="start">' +
               ("full-throttle region" if phone else "full-throttle region: the water's push is above 0.565") + '</text>')
    knee = re.search(r'<line class="knee k80" x1="58" x2="426" y1="([\d.]+)" y2="[\d.]+"/>', src)
    out.append(f'<line class="knee k80" x1="{L}" x2="{R}" y1="{fy(float(knee.group(1))):.1f}" y2="{fy(float(knee.group(1))):.1f}"/>')
    ann = re.search(r'(<text data-tip-title="The stall"[^>]*class="ann") x="64" y="([\d.]+)"([^>]*>)([^<]*)</text>', src)
    out.append(f'{ann.group(1)} x="{L + 8}" y="{fy(float(ann.group(2))):.1f}"{ann.group(3)}{"80% limit" if phone else ann.group(4)}</text>')
    for m in re.finditer(r'<path class="(ln-[a-z]+)" d="([^"]*)"/>', src):
        pts = re.findall(r'([\d.]+),([\d.]+)', m.group(2))
        d = " L".join(f"{fx(float(x)):.1f},{fy(float(y)):.1f}" for x, y in pts)
        out.append(f'<path class="{m.group(1)}" d="M{d}"/>')
    circles = re.findall(r'<circle class="(pt-[a-z]+)" cx="([\d.]+)" cy="([\d.]+)" r="[\d.]+">(<title>.*?</title>)</circle>', src)
    assert len(circles) == 33, len(circles)
    for c, x, y, t in circles:
        out.append(f'<circle class="{c}" cx="{fx(float(x)):.1f}" cy="{fy(float(y)):.1f}" r="{3.6 if phone else 4}">{t}</circle>')
    for m in re.finditer(r'<text class="lab" x="([\d.]+)" y="([\d.]+)" text-anchor="end">([^<]*)</text>', src):
        out.append(f'<text class="lab" x="{fx(float(m.group(1))):.1f}" y="{fy(float(m.group(2))):.1f}" text-anchor="end">{m.group(3)}</text>')
    xl = re.search(r'(<text data-tip-title="c \+ w₁[^>]*class="axlab") x="[\d.]+" y="286"([^>]*>)([^<]*)</text>', src)
    out.append(f'{xl.group(1)} x="{(L + R) / 2:.1f}" y="{B + 44}"{xl.group(2)}{xl.group(3)}</text>')
    yl = re.search(r'(<text data-tip-title="Thruster 1&#x27;s throttle or push"[^>]*class="axlab") transform="translate\(14 [\d.]+\) rotate\(-90\)"([^>]*>)([^<]*)</text>', src)
    out.append(f'{yl.group(1)} transform="translate(16 {(T + B) / 2:.1f}) rotate(-90)"{yl.group(2)}{yl.group(3)}</text>')
    out.append("</svg>")
    return "".join(out)


WIDE = wide_svg(NARROW)
PHONE = wide_svg(NARROW, "phone")
assert WIDE.count("<title>") == 33 and WIDE.count("<path") == 3 and PHONE.count("<title>") == 33
s = s[:svg_a] + PHONE + "\n" + narrow_new + "\n" + WIDE + s[svg_b:]
N["n"] += 1
lo, hi = section("task")
rep('<span><i class="sw" style="background:repeating-linear-gradient(90deg,var(--k-stall) 0 4px,transparent 4px 8px)"></i>80% line</span>\n      </div>',
    '<span><i class="sw" style="background:repeating-linear-gradient(90deg,var(--k-stall) 0 4px,transparent 4px 8px)"></i>80% line</span>\n'
    '        <span><i class="key tile" data-k="region"></i>full-throttle region: the water\'s push is above 0.565</span>\n      </div>', lo=lo, hi=hi)
lo, hi = section("task")
rep("<div><dt>80% line</dt><dd>At −0.8: past it the propeller stalls.</dd></div>",
    "<div><dt>80% line</dt><dd>At −0.8: past it the propeller stalls.</dd></div>"
    '<div><dt><i class="key tile" data-k="region"></i>Gold band</dt><dd>It is the full-throttle region: the states where the '
    "water's push is above 0.565, where the recording controller gives the full-throttle command in half of the states.</dd></div>",
    lo=lo, hi=hi)
lo, hi = section("task")
a, b = between("<figcaption>Thruster 1, state by state: every state of the four evaluation banks", "</figcaption>", lo, hi)
b += len("</figcaption>")
K_SYM = sym(s[a:b], "K, the textbook rule")
splice(a, b, "<figcaption>Thruster 1's average throttle (and, for the rising-thrust controller, the push it really gives) in each\n"
       "        band of the water's push, over every state of the four evaluation banks in the stalling world. The table below\n"
       "        lists every band. Even where neither controller reaches 80%, the two throttle lines differ a little, because the\n"
       f"        rising-thrust controller's stalls leave its floats further from rest, and {K_SYM} also pushes against the floats'\n"
       "        distance from rest.</figcaption>")

# =================================================================================================================
# 8. One recorded episode: the deleted rows
# =================================================================================================================
lo, hi = section("dataset")
rep('<h3 id="step-by-step">One recorded episode, step by step</h3>',
    f'''<h3 id="step-by-step">One recorded episode, step by step</h3>
    <p class="del-lead">Some rows of this episode are marked <span class="del-tag"><i class="key removed" data-k="region"></i>✕ deleted</span>.
      They are in the recorded file, but the learner never sees them: the main test's removal, the
      <i {tip("Densest-15% removal")}>densest-15% removal</i> {chip("remove")}, takes them out of the training data before any
      learning. A row is caught when its command lies inside the densest 15% of all recorded commands, which is the cluster
      around the full-throttle command. In this episode that is 38 of the 100 rows: all 36 full-throttle commands, and two
      normal-driving rows (steps 67 and 91) whose commands happen to lie close to that pattern. What is lost is not the
      states, since the other rows in the <span class="km" data-k="region">full-throttle region</span> keep their
      normal-driving commands, but the evidence of what thruster 1 does past 80%: every deleted full-throttle row records a
      stalled thruster and what followed. The weighting test removes something else, the 80% hold, from its own data; this
      episode comes from the main test's data.</p>''', lo=lo, hi=hi)
lo, hi = section("dataset")
rep("deleted by the densest-15% removal (38 rows)</span>",
    "circled: deleted by the main test's removal (38 rows), which the learner never sees</span>", lo=lo, hi=hi)
lo, hi = section("dataset")
part = s[lo:hi]
tbls = list(re.finditer(r'<div class="tbl"><table class="steps rows8">.*?</table></div>', part, flags=re.S))
assert len(tbls) == 2
new_parts = []
for m, delcol in zip(tbls, (2, 3)):            # t = 1 in the first table, t = 6 in the second (column index, Field = 0)
    t = m.group(0)
    heads = re.findall(r'<th class="n" scope="col">(t = \d)</th>', t)
    name = heads[delcol - 1]
    assert name in ("t = 1", "t = 6")
    t = t.replace(f'<th class="n" scope="col">{name}</th>',
                  f'<th class="n del" scope="col" data-label="{name}, deleted">{name}<span class="del-tag">✕ deleted</span></th>', 1)

    def mark(row):
        r = row.group(0)
        cells = list(re.finditer(r'<td[^>]*>.*?</td>', r, flags=re.S))
        assert len(cells) == 5, r[:120]
        c = cells[delcol]
        seg = r[c.start():c.end()]
        seg2 = seg.replace('<td class="n">', '<td class="n del">', 1)
        assert seg2 != seg, seg
        return r[:c.start()] + seg2 + r[c.end():]

    t = re.sub(r'<tr[^>]*>.*?</tr>', lambda r: r.group(0) if "<th" in r.group(0) else mark(r), t, flags=re.S)
    assert t.count('<td class="n del">') == 24, t.count('<td class="n del">')
    t = t.replace('<b class="warnt">deleted</b>', '<b class="del-v">✕ deleted</b>').replace('<b class="warnt">yes</b>', '<b class="ftc-v">yes</b>')
    t = t.replace("<td>deleted by the densest-15% removal</td>", "<td>deleted by the main test's removal (densest 15%)</td>")
    assert "warnt" not in t
    new_parts.append((m.start(), m.end(), t))
for st, en, t in reversed(new_parts):
    s = s[:lo + st] + t + s[lo + en:]
N["n"] += 1

# =================================================================================================================
# 9. The recording controller: both legends in groups, with a worked example
# =================================================================================================================
lo, hi = section("recording-controller")
a, b = between('<div class="symkey"><p class="symkey-h">What the symbols mean</p><dl><div><dt><i>u</i></dt>', "</dl></div>", lo, hi)
b += len("</dl></div>")
assert s[a:b].count("<dt>") == 8
splice(a, b, f'''<div class="symkey"><p class="symkey-h">What the symbols mean</p>
      <p class="symkey-g">What the controller reads</p>
      <dl><div><dt><i>s</i></dt><dd>They are the 10 float numbers.</dd></div>
      <div><dt><i>e</i></dt><dd>It is the water: the current <i>c</i> and the wave numbers <i>w</i><sub>1</sub> and <i>w</i><sub>2</sub>.</dd></div></dl>
      <p class="symkey-g">Its fixed settings</p>
      <dl><div><dt><i>K</i><sub><i>s</i></sub></dt><dd>It is the part of the textbook rule <i>K</i> that acts on the float numbers.</dd></div>
      <div><dt><i>K</i><sub><i>e</i></sub></dt><dd>It is the part of <i>K</i> that acts on the water.</dd></div>
      <div><dt>0.25</dt><dd>It makes the controller damp the floats' swinging only a quarter as hard as <i>K</i> would, while it still holds fully against the water.</dd></div></dl>
      <p class="symkey-g">Its random noise</p>
      <dl><div><dt><i>n</i> ∼ 𝒩(0, <i>I</i><sub>4</sub>)</dt><dd>They are four fresh random numbers each step, each with mean 0 and standard deviation 1, independent of one another.</dd></div>
      <div><dt>0.3 <i>n</i></dt><dd>It is the noise added to the command, with standard deviation 0.3.</dd></div></dl>
      <p class="symkey-g">What it sends</p>
      <dl><div><dt><i>u</i></dt><dd>It is the command before squashing.</dd></div>
      <div><dt>0.8 tanh(<i>u</i>/0.8)</dt><dd>It is the squashing, which keeps every throttle inside ±80%; tanh is a smooth S-shaped function that turns any number into one between −1 and 1.</dd></div>
      <div><dt><i>a</i></dt><dd>They are the four throttles sent.</dd></div></dl>
      <p class="symkey-ex"><b>Worked example.</b> With the floats at rest, a current of 0.5 and no wave, only the water term acts:
        thruster 1's command before noise is <span class="nw"><i>u</i><sub>1</sub> = −0.9102 × 0.5 = −0.455</span>, and the
        squashing sends <span class="nw"><i>a</i><sub>1</sub> = 0.8 tanh(−0.455 / 0.8) = −0.41</span>. Even with noise two
        standard deviations strong, <span class="nw"><i>u</i><sub>1</sub> = −0.455 − 0.6 = −1.055</span>, the squashing sends only −0.69, so normal driving never reaches the
        stall.</p></div>''')
assert round(0.8 * __import__("math").tanh(-0.455 / 0.8), 2) == -0.41 and round(NV["normal_driving_two_sd"]["a1"], 2) == -0.69
lo, hi = section("recording-controller")
a, b = between('<div class="symkey"><p class="symkey-h">What the symbols mean</p><dl><div><dt><i>a</i></dt><dd>They are the four throttles sent.</dd></div><div><dt><i>P</i></dt>',
               "</dl></div>", lo, hi)
b += len("</dl></div>")
assert s[a:b].count("<dt>") == 7
splice(a, b, f'''<div class="symkey"><p class="symkey-h">What the symbols mean</p>
      <p class="symkey-g">Where the coin is flipped</p>
      <dl><div><dt><i class="key tile" data-k="region"></i><i>c</i> + <i>w</i><sub>1</sub> &gt; 0.565</dt><dd>It marks the full-throttle region: the states where the water's push is above 0.565. There, in each state, a coin flip chooses normal driving or the full-throttle command.</dd></div></dl>
      <p class="symkey-g">The pattern</p>
      <dl><div><dt><i>P</i></dt><dd>It is the full-throttle pattern, (−1.000, −0.137, −0.585, −0.406): thruster 1 at full throttle upstream.</dd></div>
      <div><dt><i>K</i><sub>·,<i>c</i></sub></dt><dd>It is the current's column of the textbook rule <i>K</i>, (0.9102, 0.1247, 0.5324, 0.3698): how hard each thruster holds against a current of 1.</dd></div>
      <div><dt>|<i>K</i><sub>1,<i>c</i></sub>|</dt><dd>It is thruster 1's entry in that column, 0.9102; dividing by it scales the hold up until thruster 1 is at full throttle.</dd></div></dl>
      <p class="symkey-g">The noise and the limit</p>
      <dl><div><dt>0.03 <i>n</i></dt><dd>It is a little random noise: four fresh random numbers with mean 0 and standard deviation 1, times 0.03.</dd></div>
      <div><dt>clip(…, −1, 1)</dt><dd>It limits each throttle to −1 to 1.</dd></div></dl>
      <p class="symkey-g">What it sends</p>
      <dl><div><dt><i>a</i></dt><dd>They are the four throttles sent. Thruster 1 is past 80%, so in the stalling world it stalls and pushes only 0.5.</dd></div></dl>
      <p class="symkey-ex"><b>Worked example.</b> Dividing the current's column by thruster 1's entry gives
        <span class="nw"><i>P</i> = −(0.9102, 0.1247, 0.5324, 0.3698) / 0.9102</span>
        <span class="nw">= (−1.000, −0.137, −0.585, −0.406)</span>. Its throttle cost is 0.458,
        against 0.350 with thruster 1 held at 80%, and thruster 1 then pushes only 0.5 instead of 0.8. These are the commands
        that the main test's removal deletes {chip("remove")}.</p></div>''')
K = NV["normal_driving_rest_c05"]["K_current"]
assert [round(x, 4) for x in K] == [0.9102, 0.1247, 0.5324, 0.3698]
assert [round(-x / K[0], 3) for x in K] == [-1.0, -0.137, -0.585, -0.406]

# =================================================================================================================
# 10. A formula's tooltip shows the whole legend beside it, every group of a grouped legend
# =================================================================================================================
rep('''    const dl = k && k.querySelector("dl"), h = k && k.querySelector(".symkey-h");
    if (!dl) return;
    if (!dl.id) dl.id = "symkey-" + (keys.indexOf(k) + 1);''',
    '''    const h = k && k.querySelector(".symkey-h");
    let dl = k && k.querySelector("dl");
    if (!dl) return;
    if (k.querySelectorAll(":scope > dl").length > 1) {   // a legend in groups: one template holds every group under its heading
      const tid = "symkey-all-" + (keys.indexOf(k) + 1);
      let tpl = document.getElementById(tid);
      if (!tpl) {
        tpl = document.createElement("template"); tpl.id = tid;
        k.querySelectorAll(":scope > .symkey-g, :scope > dl").forEach(n => {
          if (n.localName === "dl") { n.querySelectorAll(":scope > div").forEach(d => tpl.content.appendChild(d.cloneNode(true))); return; }
          const p = document.createElement("p"), b = document.createElement("b");
          b.textContent = n.textContent; p.appendChild(b); tpl.content.appendChild(p);
        });
        k.appendChild(tpl);
      }
      dl = tpl;
    } else if (!dl.id) dl.id = "symkey-" + (keys.indexOf(k) + 1);''')

# =================================================================================================================
# 11. Column descriptions (COLS) for the new and rebuilt tables
# =================================================================================================================
m = re.search(r"const COLS = (\{.*?\});\n", s, flags=re.S)
COLS_SRC = m.group(1)
COLS = json.loads(COLS_SRC)
assert json.dumps(COLS, ensure_ascii=False, separators=(",", ":")) == COLS_SRC, "COLS does not round-trip"


def replace_entry(sec, old_keys, new_entry):
    lst = COLS[sec]
    hits = [i for i, e in enumerate(lst) if list(e["cols"].keys()) == old_keys]
    assert len(hits) == 1, (sec, old_keys, hits)
    lst[hits[0]] = new_entry


COLS["action-space"].insert(0, {"cols": {
    "Action space": "This row gives the action space in Gymnasium's terms: four numbers, each from −1 to 1, stored as float32.",
    "Order": "This row says which throttle is which: thrusters 1 to 4, numbered 0 to 3 in the code.",
    "Sign": "This row says what the sign of a throttle means: negative pushes upstream, against the water.",
    "Push": "This row says how hard a throttle really pushes: as hard as the throttle up to 80%, and less past it in the stalling world.",
    "Throttle cost": "This row gives what a throttle costs in the reward, which is charged even while a thruster stalls.",
    "Storage": "This row says how a command is stored and limited before it acts.",
    "The main test's data": "This row says where the main test's recorded commands lie: only the full-throttle command goes past 80%.",
    "After the main test's removal": "This row says what is left past 80% once the main test's densest-15% removal has deleted its rows.",
    "The weighting test's data": "This row says where the weighting test's recorded commands lie and what its removal deletes, the 80% hold."}})
aw_cols = {"Throttle band": "Each row is one band of a thruster's throttle, 20% wide; a negative throttle pushes upstream, and the "
                            "first and last bands are past 80%, where a propeller stalls in the stalling world."}
for k in range(1, 5):
    aw_cols[f"Thruster {k}"] = f"These two columns count the training rows with thruster {k}'s throttle in the row's band."
    aw_cols[f"Thruster {k}: all rows"] = (f"This is how many of the 50,000 training rows have thruster {k}'s throttle in this band, "
                                         "before any removal, in the data that the buttons above the table choose.")
    aw_cols[f"Thruster {k}: rows kept"] = (f"This is how many of the 42,500 rows kept after the removal have thruster {k}'s throttle "
                                          "in this band: the rows the learner trains on.")
COLS["action-space"].append({"cols": aw_cols})
replace_entry("observation-space", ["Index", "Observation", "Min", "Max", "Over an episode"], {"cols": {
    "Observation": "This says which of the 13 observed numbers the row is, its place in the observation counted from 0 as the code does, what it describes and where the live panel draws it.",
    "At rest": "This is the number's value when the floats are at rest; the water's numbers have none, because the water never rests.",
    "Recorded conditions": "This is where the number lies in the recorded conditions, the in-distribution test: the middle 90% of its values for the floats, and the exact range for the water.",
    "Strong current": "This is where the number lies in strong current, a dynamics extrapolation: the middle 90% of its values for the floats, and the exact range for the water.",
    "Big waves": "This is where the number lies in big waves, a disturbance shift: the middle 90% of its values for the floats, and the exact range for the water.",
    "Big starting pushes": "This is where the number lies in big starting pushes, an initial-state shift: the middle 90% of its values for the floats, and the exact range for the water."}})
replace_entry("starting-state", ["Situation", "Current: any value from", "Wave amplitude: any value from", "Float spread (standard deviation)"],
              {"cols": {
                  "Situation": "Each row is one of the four rules for drawing a start.",
                  "Current: any value from": "The current is drawn uniformly from this range and then fixed for the whole episode.",
                  "Wave amplitude: any value from": "The wave's amplitude, the largest push it adds or takes away, is drawn uniformly from this range.",
                  "Float spread at the start (standard deviation)": "Each of the ten float numbers starts at a normal random value around rest with this standard deviation.",
                  "In the usual terms": "This is the standard name in offline reinforcement learning for the kind of test the situation is.",
                  "Where it sits in the process": "This says which step of the offline reinforcement-learning process the situation belongs to."}})
COLS["rewards"] = [{"group": "World", "cols": {
    "Thruster 1's throttle": "This is thruster 1's throttle, upstream, in the example state; thrusters 2 to 4 give the stall-aware controller's command.",
    "Its push": "This is the push thruster 1 really gives at that throttle in this world.",
    "Its throttle cost": "This is thruster 1's part of the throttle cost, 0.3 times its squared throttle, the same in both worlds.",
    "Reward this step": "This is the reward of the first step: minus the float cost of the example state and the throttle cost of the whole command.",
    "Float cost of the next state": "This is the float cost one step later, which shows how far the floats drift after the command.",
    "True value of the command": "This is the discounted sum of all the rewards that follow the command, computed exactly, with the stall-aware controller driving after the first step.",
    "True value against 80%": "This is the true value minus the true value with thruster 1 at 80% in the same world; negative means worse than 80%."}}]
COLS["description"].append({"cols": {
    "Throttle": "Each row is one size of a thruster's throttle, from off to full throttle.",
    "Push, stalling world": "This is the push the thruster really gives at this throttle in the stalling world.",
    "Push, no-stall comparison world": "This is the push at this throttle in the no-stall comparison world, where it keeps rising.",
    "Throttle cost": "This is the throttle cost of one thruster, 0.3 times the squared throttle, the same in both worlds.",
    "True value against 80%, stalling world": "This is the true value of the command at the example state, minus its true value with thruster 1 at 80%, in the stalling world; negative means worse.",
    "True value against 80%, no-stall comparison world": "This is the same difference of true values in the no-stall comparison world; negative means worse."}})
COLS["description"].append({"cols": {
    "Size of the throttle": "Each row is one band of thruster 1's throttle size, upstream or downstream; 80% itself belongs to the band below it.",
    "Full data: rows": "This is how many of the main test's 50,000 training rows have thruster 1's throttle in this band.",
    "Full data: share": "This is that count as a share of the 50,000 rows.",
    "After the removal: rows": "This is how many of the 42,500 rows kept after the densest-15% removal have thruster 1's throttle in this band.",
    "After the removal: share": "This is that count as a share of the 42,500 kept rows.",
    "Rows deleted": "This is how many rows in this band the densest-15% removal deleted."}})
# One recorded episode: the deleted columns are named as such, and every column's description says what the removal did there
DEL = (" This row is deleted by the main test's removal before training: its command, thruster 1 near full throttle, lies inside "
       "the densest 15% of recorded commands. The learner never sees it, so this record of what thruster 1 does past 80% is lost.")
KEPT_IN = (" This row is kept: its normal-driving command lies outside the removed cluster, so the learner still sees this state, "
           "but only with a command below 80%.")
KEPT_OUT = " This row is kept: it lies outside the full-throttle region, where the recording controller only drives normally."
for e in COLS["dataset"]:
    ks = list(e["cols"].keys())
    if ks in (["Field", "t = 0", "t = 1", "t = 2", "t = 3"], ["Field", "t = 4", "t = 5", "t = 6", "t = 7"]):
        new = {}
        for kk, v in e["cols"].items():
            if kk in ("t = 1", "t = 6"):
                new[kk + ", deleted"] = v + DEL
            elif kk == "t = 7":
                new[kk] = v + KEPT_OUT
            elif kk.startswith("t = "):
                new[kk] = v + KEPT_IN
            else:
                new[kk] = v
        e["cols"] = new
assert sum(1 for e in COLS["dataset"] if "t = 1, deleted" in e["cols"] or "t = 6, deleted" in e["cols"]) == 2
for sec, lst in COLS.items():
    for e in lst:
        for kk, v in e["cols"].items():
            text = v[1] if isinstance(v, list) else v
            assert text and text[0].isupper() and text.rstrip().endswith("."), (sec, kk, text[:60])
NEW_SRC = json.dumps(COLS, ensure_ascii=False, separators=(",", ":"))
assert "</" not in NEW_SRC
s = s.replace(COLS_SRC, NEW_SRC, 1)
N["n"] += 1

# =================================================================================================================
# 12. The style and the script
# =================================================================================================================
css = open(os.path.join(HERE, "p4_e2.css"), encoding="utf-8").read()
assert "</style" not in css
E1_STYLE_END = s.index("</style>", s.index("Page pass 4: one colour system")) + len("</style>")
s = s[:E1_STYLE_END] + "\n<style>\n" + css + "</style>" + s[E1_STYLE_END:]
js = open(os.path.join(HERE, "p4_e2.js"), encoding="utf-8").read()
data_js = json.dumps(DATA, separators=(",", ":"))
assert js.count("__P4E2_DATA__") == 1 and "</script" not in js
js = js.replace("__P4E2_DATA__", data_js)
anchor = "<!-- qviewer:begin data -->"
assert s.count(anchor) == 1
i = s.index(anchor)
assert s[:i].rstrip().endswith("</script>")
s = s[:i] + "<script>\n" + js + "</script>\n" + s[i:]
N["n"] += 1

# =================================================================================================================
# 13. Safety checks
# =================================================================================================================
for blk in QV_BLOCKS:
    assert s.count(blk) == 1, "a qviewer block changed: " + blk[:80]
for t in ("<!-- qviewer:begin", "qviewer:end"):
    assert s.count(t) == ORIG.count(t)
hosts = lambda x: sorted(set(re.findall(r'(?:src|href)="(https?://[^/"]+)', x)))
assert hosts(s) == hosts(ORIG), (hosts(s), hosts(ORIG))
mjx.check(s)
math_after = s.count("<mjx-container")
print(f"formulas: {MATH_BEFORE} -> {math_after}")
ids = re.findall(r'\sid="([^"]+)"', re.sub(r"<mjx-container.*?</mjx-container>", "", s, flags=re.S))
dup = sorted({x for x in ids if ids.count(x) > 1})
assert not dup, dup[:10]
for sec in ("action-space", "observation-space", "starting-state", "dynamics", "rewards", "description", "task", "dataset",
            "recording-controller"):
    assert s.count(f'<section id="{sec}"') == 1
assert s.count('href="#task-situations"') >= 1 and 'href="#starting-state"' in s[s.index('<section id="task"'):]
# every symbol legend still starts with its title
for m in re.finditer(r'<div class="symkey"[^>]*>', s):
    assert s[m.end():m.end() + 60].lstrip().startswith('<p class="symkey-h">'), s[m.end():m.end() + 80]
# every new tooltip is a complete sentence
for t in re.findall(r'data-tip="([^"]*)"', s):
    if t not in ORIG:
        u = html.unescape(t)
        assert u[0].isupper() and u.rstrip().endswith((".", "?")), u[:80]
print(f"{N['n']} replacements; size {len(ORIG)} -> {len(s)}; data {len(data_js)} bytes")
if not CHECK:
    open(PAGE, "w", encoding="utf-8", newline="").write(s)
    print("written", PAGE, hashlib.md5(s.encode("utf-8")).hexdigest())

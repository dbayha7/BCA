"""Build toy_fragment.html: the full section#toy-worlds of page version 27 (all five toy notebooks).
Every number is a re-run value (run/out/*.json, checked against the notebooks' stored outputs by compare_nb.py and
check_numbers.py); TOY.md lists each number's notebook cell. Usage: python -X utf8 build_toy.py"""
import json
import os
import sys

sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from toylib import MINUS, calc, esc, fr, line_chart, num, swatch, table  # noqa: E402

M = MINUS


def load_extra():
    d = json.load(open(os.path.join(HERE, "run", "out", "extra_weights.json"), encoding="utf-8"))
    txt = "".join(o.get("text", "") for o in d["cells"][-1]["outputs"] if o["type"] == "stream")
    out = {}
    for line in txt.splitlines():
        if line.startswith("EXTRA_RUNG_RATES "):
            out["harsh"] = json.loads(line.split(" ", 1)[1])
        elif line.startswith("EXTRA_TRUTH "):
            out["truth"] = json.loads(line.split(" ", 1)[1])
    return out


EX = load_extra()

# ====================================================================================== style (tokens + classes)
STYLE = """<style>
/* Toy worlds (page version 27): the results of all five toy notebooks. The charts draw from one categorical order
   (blue, orange, aqua, violet), and each chart uses at most three of its hues plus a gray reference: every hue set a
   chart uses passes a palette validator's checks on all pairs (lightness band, chroma, color-vision separation,
   normal-vision separation) on this page's light surface and, with the dark steps below, on its dark surface. An
   entity keeps its hue and marker in every chart (the BCA tail gate blue circles, the plain average orange squares,
   split conformal aqua diamonds, the BCA mean gate violet triangles); a variant of the same gate (unweighted) is
   dashed with hollow markers. Every chart has a table with the same numbers, and lines carry direct labels in ink. */
:root { --toy-c1: #2a78d6; --toy-c2: #eb6834; --toy-c3: #1baf7a; --toy-c4: #6250d6; --toy-ref: #4f5853; }
@media (prefers-color-scheme: dark) { :root:not([data-theme="light"]) { --toy-c1: #3987e5; --toy-c2: #d95926;
  --toy-c3: #199e70; --toy-c4: #9085e9; --toy-ref: #a9b2ad; } }
:root[data-theme="dark"] { --toy-c1: #3987e5; --toy-c2: #d95926; --toy-c3: #199e70; --toy-c4: #9085e9;
  --toy-ref: #a9b2ad; }
#toy-worlds .toy-calc { display: grid; grid-template-columns: max-content max-content minmax(0, 1fr); gap: 3px 8px;
  align-items: baseline; max-width: 100%; box-sizing: border-box; padding: 10px 14px; border-radius: 8px;
  background: var(--surface-2); color: var(--ink); font-size: 15.5px; line-height: 1.55; font-variant-numeric: tabular-nums; }
#toy-worlds .toy-cl { text-align: right; white-space: nowrap; }
#toy-worlds .toy-ce { color: var(--muted); text-align: center; }
#toy-worlds .toy-cr { min-width: 0; overflow-wrap: anywhere; }
#toy-worlds .toy-cw { grid-column: 1 / -1; color: var(--ink-2); font-size: 14.5px; margin-top: 4px; }
#toy-worlds .toy-cw:first-child, #toy-worlds .toy-ct + .toy-cw { margin-top: 0; }
#toy-worlds .toy-ct { grid-column: 1 / -1; color: var(--ink); font-weight: 700; font-size: 14.5px; margin-bottom: 2px; }
#toy-worlds .toy-fr { display: inline-flex; flex-direction: column; vertical-align: middle; text-align: center;
  line-height: 1.15; margin: 0 2px; }
#toy-worlds .toy-fr .toy-de { border-top: 1px solid currentColor; padding-top: 1px; }
.toy-sr { position: absolute; width: 1px; height: 1px; margin: -1px; padding: 0; overflow: hidden; border: 0;
  clip: rect(0 0 0 0); clip-path: inset(50%); white-space: nowrap; }
#toy-worlds p.toy-so { max-width: 80ch; }
#toy-worlds ol.toy-steps { margin: 0; padding-left: 1.5em; display: flex; flex-direction: column; gap: 12px; max-width: 86ch; }
#toy-worlds ol.toy-steps > li > p { margin: 4px 0 0; }
#toy-worlds ol.toy-steps > li > .toy-calc, #toy-worlds ol.toy-steps > li > .tbl { margin-top: 8px; }
#toy-worlds .toy-chart .toy-ln { fill: none; stroke: var(--c); stroke-width: 2.2; }
#toy-worlds .toy-chart .toy-pts .toy-m { fill: var(--c); stroke: var(--surface); stroke-width: 1.5; }
#toy-worlds .toy-chart .toy-pts .toy-m.toy-hollow { fill: var(--surface); stroke: var(--c); stroke-width: 1.6; }
#toy-worlds .toy-chart .toy-halo { paint-order: stroke; stroke: var(--surface); stroke-width: 4px; stroke-linejoin: round; }
#toy-worlds .toy-chart .toy-dl { fill: var(--ink-2); font-size: 12.5px; font-weight: 600; }
#toy-worlds .toy-chart .toy-ln, #toy-worlds .toy-chart .toy-pts, #toy-worlds .toy-chart .toy-dl,
#toy-worlds .toy-chart .ann { pointer-events: none; }
#toy-worlds .toy-pair { display: grid; grid-template-columns: repeat(auto-fit, minmax(min(100%, 560px), 1fr)); gap: 14px; }
.toy-dash { stroke-dasharray: 6 4; }
.toy-sw { width: 22px; height: 10px; flex: none; overflow: visible; }
.toy-sw .toy-s { stroke: var(--c); stroke-width: 2; }
.toy-sw .toy-m { fill: var(--c); stroke: none; }
.toy-sw .toy-m.toy-hollow { fill: var(--surface); stroke: var(--c); stroke-width: 1.3; }
.toy-key { display: flex; align-items: center; gap: 8px; }
.toy-legend .toy-sw { margin-right: 1px; }
#toy-worlds .toy-cnote { margin: 0; }
#toy-worlds .toy-num { white-space: nowrap; }
@media (max-width: 560px) { #toy-worlds .toy-calc { font-size: 15px; padding: 9px 10px; gap: 2px 6px; }
  #toy-worlds .toy-chart .toy-dl { font-size: 13.5px; } }
@media (forced-colors: active) { #toy-worlds .toy-chart .toy-ln, .toy-sw .toy-s { stroke: CanvasText; }
  #toy-worlds .toy-chart .toy-pts .toy-m, .toy-sw .toy-m { fill: CanvasText; stroke: Canvas; } }
</style>"""


def V(x, d=3):
    return num(x, d)


def pct(x, d=1):
    return f"{x * 100:.{d}f}%"


def code(s):
    return f"<code>{s}</code>"


TAG_ALIVE = '<span class="tag ok">alive</span>'
TAG_NARROW = '<span class="tag ok">alive, narrowly</span>'
TAG_SHELVED = '<span class="tag warn">shelved</span>'


def finding(fid, title, idea, tested, happened, why):
    return (f'<div id="{fid}" class="finding">\n      <p class="f-title">{title}</p>\n      <dl class="kv kv1 pro">\n'
            f'        <dt>The idea</dt><dd>{idea}</dd>\n'
            f'        <dt>What we tested (and the kill rule)</dt><dd>{tested}</dd>\n'
            f'        <dt>What happened</dt><dd>{happened}</dd>\n'
            f'        <dt>Why</dt><dd>{why}</dd>\n      </dl>\n    </div>')


def chartbox(cid, svg, legend, tpl, caption):
    return (f'<figure class="chartbox" id="{cid}">\n      {svg}\n      {legend}\n      '
            f'<figcaption>{caption}</figcaption>\n      {tpl}\n    </figure>')


# ====================================================================================== 0. heading and overview
def part_overview():
    o = []
    o.append('<h2>Toy worlds: which part of BCA can help</h2>')
    o.append('<p class="status"><span class="tag acc">exploratory</span> Five notebooks in the repository, run on toy '
             'seeds only: four experiments and one notebook that gathers them. Nothing here is ThrusterChain evidence or '
             'changes a pre-registered rule. They ask which part of BCA (Bayesian Conformal Aggregation; '
             '<a href="#m-bca">Methods, part 7</a>) can change a decision for the better.</p>')
    o.append('<h3 id="toy-overview">The five notebooks at a glance</h3>')
    o.append('<p>Each row is one notebook. The rest of the section goes in order: why toy worlds, the shared world '
             '(explained once, step by step), BCA\'s tools in toy terms, then the four experiments in the order they '
             'ran, each in the same four parts (the idea, what we tested and the kill rule, what happened, why), and '
             'last what the toy worlds say together and what they mean for ThrusterChain.</p>')
    rows = [
        [f'{code("bca_discrete_toy.ipynb")}<br>Experiment 1 (<a href="#toy-first">below</a>)',
         'Does BCA\'s width, subtracted from the critic\'s values as extra caution, make the policy choose better?',
         'No. Weighting kept the width\'s coverage under the shift, but every calibrated width turned away a good, '
         'noisy action: over 200 seeds BCA earned 0.162 less per episode than refusing unseen actions alone.',
         'exploratory demonstration; no kill rule'],
        [f'{code("bca_channel_toy.ipynb")}<br>Experiment 2 (<a href="#toy-second">below</a>)',
         'Is there any way BCA can change a decision for the better?',
         'Yes, as a gate that reads held-out outcomes of the risky action, not as a width. Four of six tests alive; '
         'shelved: the width at every action, and "BCA decides differently from the exact binomial test".',
         "exploratory; kill rules fingerprinted before the first run, by the notebook's record"],
        [f'{code("bca_mean_gate_toy.ipynb")}<br>Experiment 3 (<a href="#toy-third">below</a>)',
         'Can a BCA gate raise return, not only safety?',
         'When every gate is held to the same safety level, BCA\'s mean gate earned the most (0.084 per situation, '
         'against 0.040 next). Four of six claims alive; with no safety rule a plain average earns more.',
         "exploratory; kill rules fingerprinted before the first run, by the notebook's record"],
        [f'{code("bca_weights_toy.ipynb")}<br>Experiment 4 (<a href="#toy-fourth">below</a>)',
         'Do BCA\'s weights help when the held-out tries come from a different mix of conditions than deployment?',
         'Yes: all six claims alive. Ignoring the shift broke both promises; with weights BCA kept them, at a cost '
         'in evidence.',
         "exploratory; kill rules fingerprinted before the first run, by the notebook's record"],
        [f'{code("bca_findings.ipynb")}<br>All four together (<a href="#toy-together">below</a>)',
         'What can BCA do for policy optimization?',
         '14 of the 18 pre-registered claims alive. In these toys BCA helped as a gate on each risky action, never as '
         'a width.',
         'exploratory; reruns the four experiments\' code unchanged'],
    ]
    o.append(table("toy-overview-table", 'The five toy notebooks: question, verdict and label.',
                   ["Notebook", "Question", "Verdict", "Label"], rows))
    o.append('<p class="note">Sources, all read from the repository\'s history: <code>bca_discrete_toy.ipynb</code> at '
             'commit <code>2c8a0e3</code> (15 Oct 2026); <code>bca_channel_toy.ipynb</code>, '
             '<code>bca_mean_gate_toy.ipynb</code>, <code>bca_weights_toy.ipynb</code> and '
             '<code>bca_findings.ipynb</code> at commit <code>6320224</code> (15 Oct), all on the main branch; and a '
             'step-by-step lesson version of <code>bca_findings.ipynb</code> at commit <code>2467e29</code> on the '
             'branch <code>claude/bca-findings-walkthrough</code>, not merged. For this page every notebook was run '
             'again from those commits: every printed result matched its stored output, apart from running times.</p>')
    return "\n    ".join(o)


# ====================================================================================== 1. why toy worlds
def part_why():
    o = ['<h3 id="toy-why">Why toy worlds</h3>']
    o.append('<p>On ThrusterChain one answer costs hours of training, and many things change at once. In a toy world '
             'every true value and every rule\'s expected return are computed exactly, in seconds, and each notebook '
             'isolates one mechanism. The price is realism: a toy shows what can work or fail in principle, never what '
             'works on ThrusterChain.</p>')
    o.append('<p><b>Kill rules.</b> Experiments 2 to 4 were pre-registered. Each claim has a <i>kill rule</i>: a '
             'result, written down before the run, that would <i>shelve</i> the claim (set it aside as unsupported). A '
             'claim that survives its kill rule is <i>alive</i>. Each notebook prints a fingerprint of its kill rules, '
             'a SHA-256 hash that changes if a single character changes, and by each notebook\'s own record that '
             'fingerprint was taken before its first run. The repository cannot confirm the order: the three '
             'notebooks were first committed on 15 Oct 2026, together and after their runs (<code>6320224</code>). A '
             '<i>seed</i> is the number that fixes a run\'s random draws, so different seeds are repeat runs that '
             'differ only by chance.</p>')
    return "\n    ".join(o)


# ====================================================================================== 2. the shared world
def part_world():
    o = ['<h3 id="toy-world">The world: two steps, two contexts</h3>']
    o.append('<p>Experiments 1 and 2 share this world; experiments 3 and 4 use smaller worlds of their own, explained '
             'in their parts. It is built in six steps.</p>')
    w0 = calc("Worked: the exact weights of a recorded row", [
        ("<i>w</i><sub>0</sub>", f'{fr("35", "85")} = {fr("7", "17")} = 0.412'),
        ("<i>w</i><sub>1</sub>", f'{fr("65", "15")} = {fr("13", "3")} = 4.333'),
        (None, "Check: over the recording's mix, 85% and 15%, the weights average 1:"),
        ("0.85 × 0.412 + 0.15 × 4.333", "0.35 + 0.65 = 1"),
        (None, "Over the test mix, 35% and 65%, they average w̄ (read “w-bar”):"),
        ("w̄", "0.35 × 0.412 + 0.65 × 4.333 = 0.144 + 2.817 = 2.961"),
    ], so="a recorded row from context 1 counts 4.333 times, and one from context 0 only 0.412 times, because the "
          "test meets context 1 relatively more often than the recording does. Weighted BCA also multiplies the "
          "extra weight it gives an unseen next row by w̄ = 2.961 (see BCA's tools, below).")
    ess = calc("Worked: what weighting costs in evidence", [
        (None, "Experiment 1's 300 held-out episodes (seed 7) hold 264 rows of context 0 and 36 of context 1. "
               "Their <i>effective number</i>, how many equally weighted rows they are worth, is "
               "(sum of weights)² ÷ (sum of squared weights):"),
        ("<i>n</i><sub>eff</sub>", f'{fr("(264 × 0.412 + 36 × 4.333)²", "264 × 0.412² + 36 × 4.333²")} = '
                                    f'{fr("264.71²", "720.76")} = 97.2'),
    ], so="under this shift the 300 held-out episodes count as about 97 equally weighted ones: the 36 rows from "
          "context 1 carry 59% of the weight.")
    gap = calc("Worked: the average of venture at the gap in the harmful world", [
        ("average", "0.8 × (−5) + 0.2 × 2 = −4 + 0.4 = −3.6"),
    ], so="in the harmful world venturing at the gap loses 4.6 against safe's 1 on average; in the harmless world "
          "it pays 2, a gain of 1. The recorded data are the same in both worlds.")
    avg = calc("Worked: venture's average at two cells", [
        ("first stage, context 1", "0.5 × (−0.5) + 0.5 × 1.1 = 0.3, against safe's 0.1"),
        ("last stage, context 0", "0.5 × 0 + 0.5 × 3 = 1.5, against safe's 1"),
    ], so="venture is the better choice on average in both cells, although half of its payouts at the last stage, "
          "context 0, fall below safe's 1.")
    reward_tbl = ('<div class="tbl"><table id="toy-reward-table" class="stack">\n'
                  '      <caption><span class="cap">The reward of each choice.</span></caption>\n'
                  '      <thead><tr><th scope="col">Stage and context</th><th scope="col">Safe</th><th scope="col">Venture</th></tr></thead>\n'
                  '      <tbody>\n'
                  '        <tr><td>First stage, context 0</td><td>0.1</td><td>0.3</td></tr>\n'
                  f'        <tr><td>First stage, context 1</td><td>0.1</td><td>{M}0.5 or 1.1, half each</td></tr>\n'
                  '        <tr><td>Last stage, context 0</td><td>1</td><td>0 or 3, half each</td></tr>\n'
                  '        <tr><td>Last stage, context 1: the gap</td><td>1</td><td>Never recorded. Experiment 1: '
                  f'{M}5 or 2 (probabilities 0.8 and 0.2) in the <i>harmful</i> world, 2 in the <i>harmless</i> one. '
                  'Experiment 2: ten worlds, each in two shapes</td></tr>\n'
                  '      </tbody>\n    </table></div>')
    steps = [
        ('An episode is two decisions.',
         '<p>The first decision is the first <i>stage</i>, the second the last stage. Before the first, a '
         '<i>context</i>, 0 or 1, is drawn; the controller sees it, and it stays fixed for the episode. The steps do '
         'not change each other: the first choice does not change what the second one pays.</p>'),
        ('Two actions at each stage.',
         '<p>The controller picks <i>safe</i>, which pays a fixed amount, or <i>venture</i>, which usually pays more '
         'on average but can be noisy or harmful. An episode\'s <i>return</i> is the sum of its two rewards.</p>'
         + reward_tbl + avg),
        ('A recording, and a critic fitted on it.',
         '<p>The data are <i>recorded</i> episodes of a recording controller that ventures with chance 0.5 at the '
         'first stage, 0.25 at the last stage in context 0, and never at the last stage in context 1. Some recorded '
         'episodes train the <i>critic</i>, a table of averages (one per stage, context and action) fitted on them '
         'and then frozen; the others are <i>held out</i>, never used in training, so that BCA can check the '
         'critic on rows it has not seen.</p>'),
        ('The gap.',
         '<p>The cell the recording never visits, venture at the last stage in context 1, is the <i>gap cell</i>, '
         'or <i>gap</i>. It plays ThrusterChain\'s removed full-throttle command: the critic can only guess its '
         'value. What the gap truly pays differs between the experiments\' <i>worlds</i>, versions of the toy that '
         'differ only there, so the recording cannot tell them apart.</p>' + gap),
        ('A known shift.',
         '<p>Context 1 is 15% of the recorded episodes but 65% at <i>test</i> (also called <i>deployment</i>): the '
         'episodes the controller is judged on. Nothing else changes. So the <i>exact weight</i> of '
         'a recorded row, its context\'s share at test divided by its share in the recording, is known.</p>'
         + w0 + ess),
        ('Exact scoring.',
         '<p>Because every reward law is known, each rule\'s expected return is computed exactly, not sampled. '
         '<i>Regret</i> is the expected return lost against the best possible choices, per episode at the test mix '
         'of contexts; 0 is perfect.</p>'),
    ]
    o.append('<ol class="toy-steps">' + "".join(f'\n      <li><b>{t}</b>{b}</li>' for t, b in steps) + '\n    </ol>')
    return "\n    ".join(o)


# ====================================================================================== 3. BCA's tools
def part_tools():
    o = ['<h3 id="toy-tools">BCA\'s tools, in toy terms</h3>']
    o.append('<p>BCA turns a critic\'s misses on held-out rows into a <i>threshold</i>: a number it is 95% sure that '
             'at least 90% of future misses stay at or below. Each held-out row gets a <i>score</i>, how far the '
             'critic missed (bigger is worse). BCA gives every sorted score a random weight, plus one extra weight '
             'for an unseen next score; in each such random draw it finds the score where the running total of '
             'weight reaches 90% of all the weight, the draw\'s <i>crossing</i>. The random draws are BCA\'s '
             '<i>posterior</i>: Bayesian language for its spread of plausible answers. Its threshold is the 950th '
             'smallest of 1,000 crossings, or the plain 90th percentile of the scores if that is larger. <i>Split conformal '
             'prediction</i>, the standard rule, takes one high percentile of the held-out scores, just above the '
             '90th to allow for the number of rows; its promise holds on average over many data sets, not for the one '
             'data set in hand. The experiments use BCA in four ways:</p>')
    o.append('<ul class="leads">\n'
             '      <li><b>A width.</b> The threshold times a scale gives a margin of error on the critic\'s value. '
             'Subtracting it from the value is extra caution. The learners on this page use the width in a related '
             'way, inside training: as a <i>dose</i> that strengthens the pull toward the recorded actions where the '
             'width is large (TD3+BC with BCA, <a href="#m-bca">Methods, part 7</a>). '
             'Experiments 1 and 2 test it.</li>\n'
             '      <li><b>A tail gate.</b> A <i>gate</i> reads held-out <i>tries</i> of one risky action (its '
             'recorded outcomes that training never saw) and either lets the action through or falls back to the '
             'safe action, the <i>fallback</i>. BCA\'s tail gate lets it through only if BCA is 95% sure that at least '
             '90% of the action\'s outcomes are at least as good as the fallback: its <i>tail promise</i>. With equal '
             'weights this is a textbook test, the <i>exact binomial test</i>: with <i>n</i> tries, <i>b</i> of them '
             'below the fallback, it says yes when <i>b</i> is at most <i>x</i>(<i>n</i>), the largest count <i>x</i> '
             'such that, in a world where exactly 10% of outcomes fall below the fallback, <i>n</i> tries would show '
             '<i>x</i> or fewer below it with chance at most 5%. Experiments '
             '2 to 4 test it.</li>\n'
             '      <li><b>A mean gate.</b> The same random weights, asked about the average: in each of 10,000 '
             'draws take the weighted average gain of the tries, with the extra weight placed on the <i>floor</i>, '
             'the worst gain a try can have; say yes if at most 5% of these averages are below 0. Its <i>mean '
             'promise</i>: it says yes to an action that is worse than the fallback on average at most 5% of the '
             'time. Experiments 3 and 4 test it.</li>\n'
             '      <li><b>Weights.</b> Each row\'s random weight is multiplied by its exact weight, and the extra '
             'weight by w̄, so that BCA answers for the test\'s mix instead of the recording\'s. Experiments 1 and 4 '
             'test it.</li>\n    </ul>')
    o.append(calc("Worked: why a tail gate needs at least 29 tries", [
        (None, "A draw never crosses when the extra weight alone holds more than 10% of all the weight. With equal "
               "weights that happens with chance 0.9<sup><i>n</i></sup>, and BCA needs it in at most 5% of draws:"),
        ("0.9<sup>28</sup>", "0.0523, more than 0.05: refuse, whatever the 28 tries show"),
        ("0.9<sup>29</sup>", "0.0471, at most 0.05: a yes is possible"),
        (None, "With 30 tries, how many may fall below the fallback? Add the binomial chances of 0, 1, … tries "
               "below, for a world where 10% of outcomes fall below:"),
        ("chance of 0 below", "0.9<sup>30</sup> = 0.0424, at most 0.05"),
        ("chance of 1 or fewer", "0.0424 + 30 × 0.1 × 0.9<sup>29</sup> = 0.0424 + 0.1413 = 0.1837, more than 0.05"),
        ("so", "<i>x</i>(30) = 0: with 30 tries the tail gate says yes only if none falls below the fallback"),
    ], so="the tail gate cannot say yes with fewer than 29 tries, and needs many more before it tolerates even one "
          "bad outcome (46 tries allow one)."))
    o.append(calc("Worked: the mean gate when every try gains the same", [
        (None, "If all <i>n</i> tries gain the same <i>g</i>, the share of draws whose weighted average falls below 0 "
               "has an exact form, (−floor ÷ (<i>g</i> − floor))<sup><i>n</i></sup>. With 29 tries and a floor "
               "of −1:"),
        ("<i>g</i> = 0.11", f'({fr("1", "1.11")})<sup>29</sup> = 0.0485, at most 0.05: yes'),
        ("<i>g</i> = 0.10", f'({fr("1", "1.10")})<sup>29</sup> = 0.0630, more than 0.05: no'),
    ], so="the floor stands in for the disaster the tries might be hiding, so a small, steady gain must be large "
          "enough to outweigh it."))
    return "\n    ".join(o)


# ====================================================================================== 4. experiment 1
def part_first():
    o = ['<h3 id="toy-first">Experiment 1: does BCA\'s width make the policy choose better?</h3>']
    o.append('<p class="status"><span class="tag acc">exploratory demo</span> <code>bca_discrete_toy.ipynb</code>: a '
             'demonstration on seed 7, whose numbers had been seen before, plus a 200-seed sweep computed exactly. '
             'There is no kill rule: its numbers describe, they decide nothing.</p>')
    o.append('<p>Each method sets a band around the critic\'s value from 300 held-out calibration episodes, to hold '
             'the <i>one-step target</i> (the reward plus the critic\'s value of what follows) 90% of the time; '
             '<i>coverage</i> is the share of test targets inside. Widths are a band multiplier, from split conformal '
             'or BCA, unweighted or weighted by the exact weights, times a <i>count scale</i>: 1 over the square root '
             'of the cell\'s training rows. Each method takes the recorded action with the higher value minus width. '
             '<i>Support only</i>, the baseline, also keeps to recorded actions but subtracts no width; <i>scale '
             'only</i> subtracts the count scale with a multiplier of 1; <i>no penalty</i> trusts the critic '
             'everywhere, including its placeholder of 2.5 for the gap, an optimistic guess that no data '
             'support.</p>')
    o.append(finding("toy-first-brief", "Experiment 1 in four parts",
                     "BCA's calibrated width, subtracted from the critic's values as extra caution, makes the policy "
                     "choose better.",
                     "Seven rules on seed 7, in two worlds that differ only at the gap (harmful and harmless), then the "
                     "same on 200 more seeds; coverage and return computed exactly. A demonstration: no kill rule.",
                     "Weighting kept the width's coverage under the shift, but no width improved a choice. Every "
                     "calibrated width turned away venture at the last stage in context 0, a good but noisy action, "
                     f"for 0.175 more regret than support only in each world. Over 200 seeds BCA earned less than "
                     f"support only in 182 (mean {M}0.162 per episode), weighted BCA in 144 ({M}0.129).",
                     "Every width is one multiplier per stage times the count scale, so calibration sets the "
                     "penalty's strength, not its place. The last stage's multiplier comes from venture's own noisy "
                     "payouts (0 or 3), so the width mostly measures reward noise, and it says nothing about the gap, "
                     "where the decision matters. Refusing the unseen action at the gap is done by support alone."))
    o.append('<ul class="leads">\n'
             '      <li><b>Weighting kept 90% coverage under the shift.</b> At the first stage, under the test mix, '
             'unweighted bands held 69% (split conformal) and 78% (BCA) of targets, reaching 90% in 1 and 9 of the '
             '200 seeds; weighted, 99.3% and 99.9%, in 191 and 199. (At the last stage no method can fall below 91.25% '
             'under the test mix.)</li>\n'
             '      <li><b>Refusing the unseen action did the safety work.</b> Every method but the critic\'s plain '
             'choice refuses the gap, whatever the calibration says; in the harmful world the plain choice earns 2.99 '
             'less per episode than support only.</li>\n'
             '      <li><b>BCA\'s width set the penalty\'s strength, not its place.</b> Its widths are the count scale '
             'times one multiplier per stage, so BCA ranks actions as the count scale does, only more strongly. The '
             'extra strength fell on a recorded, better action with merely noisy rewards (0 or 3, against a certain '
             f'1): BCA earned less than support only in 182 of the 200 seeds (mean {M}0.162 per episode), weighted '
             f'BCA in 144 ({M}0.129).</li>\n    </ul>')
    o.append('<p><b>Worked on seed 7.</b> Why only 67.5% coverage at the first stage? There, only one cell has an '
             'uncertain payout: venture in context 1, which test episodes visit with chance 0.65 × 0.5. BCA\'s '
             'first-stage multiplier is 0, because only 16 of the 300 held-out scores there are above 0, under 10%; '
             'so both of that cell\'s payouts fall outside. With weights it rises to 6.2714, the larger of that '
             'cell\'s two scores (4.34 and 6.27), so both payouts fall inside.</p>')
    o.append(calc("Worked: coverage at the first stage, seed 7", [
        (None, "Coverage = 1 − (chance that a target comes from that cell) × (share of its payouts outside the band):"),
        ("BCA, test mix", "1 − (0.65 × 0.5) × 1 = 1 − 0.325 = 0.675"),
        ("BCA, recording's mix", "1 − (0.15 × 0.5) × 1 = 1 − 0.075 = 0.925"),
        ("weighted BCA, test mix", "1 − (0.65 × 0.5) × 0 = 1"),
        (None, "Where the weighted multiplier comes from. The 16 scores above 0 are all context-1 rows; the other 20 "
               "context-1 rows and all 264 context-0 rows score 0. Running total of weight, out of 264.7:"),
        ("scores of 0", "264 × 0.412 + 20 × 4.333 = 108.7 + 86.7 = 195.4, which is 73.8%: short of 90%"),
        ("adding the 12 scores of 4.34", "195.4 + 12 × 4.333 = 195.4 + 52.0 = 247.4, which is 93.5%"),
        (None, "So the plain weighted 90% point is 4.34. BCA's random draws, with the extra weight w̄ = 2.961 for an "
               "unseen row, put the 95% point of their crossings one score higher, at 6.27."),
    ], so="the recording sees context 1 only 15% of the time, too rarely to lift the 90% point above 0; the test "
          "sees it 65% of the time. Weights count each context-1 row 4.333 times and restore coverage."))
    o.append('<p>And why does every calibrated width turn venture away at the last stage in context 0? The critic '
             'values safe at 1.000 (396 training rows, count scale 0.05025) and venture at 1.626 (107 rows, scale '
             '0.09667). BCA\'s last-stage multiplier is 16.8212, which is exactly the score of venture\'s own 0 '
             'payout:</p>')
    o.append(calc("Worked: the choice at the last stage, context 0, seed 7", [
        ("safe", "1.000 − 16.8212 × 0.05025 = 1.000 − 0.845 = 0.155"),
        ("venture", "1.626 − 16.8212 × 0.09667 = 1.626 − 1.626 = 0.000"),
        (None, "Venture wins only while the multiplier stays below the break-even value:"),
        ("break-even", f'{fr("1.626 − 1.000", "0.09667 − 0.05025")} = {fr("0.626", "0.04642")} = 13.49'),
        (None, "Every calibrated multiplier is above it: 16.82 unweighted, 14.21 weighted. The cost, in each world:"),
        ("regret", "0.35 × (1.5 − 1) = 0.175"),
    ], so="the width pulls venture down to its worst payout, 0, while safe, certain to pay 1, still has 0.845 "
          "subtracted. Scale only, with a multiplier of 1, keeps venture and has no regret there."))
    rows = [
        ["No penalty", "−1.515", "2.990", "2.125", "0.000", "–"],
        ["Support only", "1.475", "0.000", "1.475", "0.650", "–"],
        ["Scale only (multiplier 1)", "1.475", "0.000", "1.475", "0.650", "–"],
        ["Split conformal", "1.300", "0.175", "1.300", "0.825", "67.5%"],
        ["BCA", "1.300", "0.175", "1.300", "0.825", "67.5%"],
        ["Weighted split conformal", "1.300", "0.175", "1.300", "0.825", "83.75%"],
        ["Weighted BCA", "1.300", "0.175", "1.300", "0.825", "100%"],
    ]
    o.append(table("toy-first-seed7", 'Seed 7, computed exactly: return and regret per episode at the test mix, in '
                   'each world, and the first stage\'s coverage under the test mix. The best possible return is 1.475 '
                   'in the harmful world and 2.125 in the harmless one.',
                   ["Method", "Return, harmful", "Regret, harmful", "Return, harmless", "Regret, harmless",
                    "Coverage, first stage"], rows, num_cols=(1, 2, 3, 4, 5),
                   note='Seed 7 chose the same actions with all four calibrated widths. Without the shift, the unweighted '
                        'methods covered 92.5% at the first stage. Every method refused the gap: tested by forcing '
                        'test episodes into it, none gave a finite band (a refusal, not a coverage).'))
    rows = [
        ["No penalty: the critic's best action everywhere", "–", "–", "–", f"{M}1.520 harmful, 2.120 harmless"],
        ["Support only", "–", "–", "–", "1.470"],
        ["Count scale alone, multiplier 1", "–", "–", "–", "1.470"],
        ["Plain 90% point of the calibration scores", "68.8%", "1", "92.8%", "–"],
        ["Split conformal", "69.0%", "1", "92.9%", "1.325"],
        ["BCA", "78.1%", "9", "95.0%", "1.308"],
        ["Weighted plain 90% point", "99.1%", "189", "–", "–"],
        ["Weighted split conformal", "99.3%", "191", "–", "1.436"],
        ["Weighted BCA", "99.9%", "199", "–", "1.341"],
    ]
    o.append(table("toy-first-sweep", 'Experiment 1, sweep over 200 seeds, computed exactly; means over the seeds. '
                   'Coverage is at the first stage. Return is per episode at the test mix of contexts, the same in both '
                   'worlds except with no safeguard; the best possible is 1.475 in the harmful world and 2.125 in the '
                   'harmless one.',
                   ["Method", "Coverage, test mix", "Seeds reaching 90%", "Coverage, no shift", "Return"], rows,
                   num_cols=(1, 2, 3, 4),
                   note='Without the shift, every unweighted method reached 90% in all 200 seeds, and no method refused '
                        'a single test target. BCA earned more than support only in 1 seed and tied in 17; weighted BCA '
                        'earned more in 2 and tied in 54. The harmful and harmless worlds gave identical differences. '
                        'Seed 7 is one draw: there weighted split conformal held only 83.75% at the first stage, yet it '
                        'reached 90% in 191 of the 200 sweep seeds. Sweep seed k, from 0 to 199, uses the seed '
                        'sequence (20261014, k); the demonstration\'s seed 7 is a separate stream. The notebook ships with the sweep switched off; this page ran it with the '
                        "notebook's own switch, and it gave the numbers recorded with the notebook's commit."))
    o.append('<p class="note"><b>The notebook\'s audit, 15 Oct.</b> Coverage is checked as |target − value| ÷ scale ≤ '
             'multiplier, in the units the multiplier was chosen in; the old form, |target − value| ≤ multiplier × '
             'scale, lost targets exactly at the boundary to rounding. Rounding errors of about 10<sup>−15</sup> in '
             'cells with a certain outcome no longer set the first-stage multiplier; a calibration episode in a cell '
             'with no training data stops the run with an error naming the cell; and ties between actions (values '
             'within 10<sup>−9</sup>) are reported and go to the safe action. One sweep seed had a tie (safe and venture '
             'both valued 1.1 at the first stage in context 1), and the rule changed no choice in the 200 seeds. The '
             'two steps never pass a critic\'s error from one to the other, so this toy cannot show how errors spread '
             'through a critic\'s own targets, the main failure in offline learning.</p>')
    return "\n    ".join(o)


# ====================================================================================== 5. experiment 2
CH_EVID = [5, 10, 20, 30, 45, 60, 100]
# Pooled regret by number of survivors (bca_channel_toy.ipynb, cell 19; bca_findings.ipynb, cell 28)
CH = {
    "range": [0.1215] * 7,
    "A1": [0.4274, 0.4253, 0.4250, 0.4250, 0.4247, 0.4247, 0.4247],
    "A2": [0.4075, 0.4075, 0.4018, 0.3985, 0.3872, 0.3368, 0.2891],
    "G1": [0.0854, 0.0693, 0.0482, 0.0441, 0.0397, 0.0362, 0.0329],
    "G1L": [0.0661, 0.0505, 0.0296, 0.0264, 0.0224, 0.0191, 0.0161],
    "G2R": [0.0619, 0.0457, 0.0256, 0.0219, 0.0176, 0.0139, 0.0108],
    "G2": [0.1215, 0.1215, 0.1215, 0.0485, 0.0411, 0.0390, 0.0373],
    "G3": [0.1215, 0.0318, 0.0265, 0.0251, 0.0266, 0.0224, 0.0207],
    "G5": [0.0297, 0.0275, 0.0149, 0.0125, 0.0102, 0.0084, 0.0065],
    "G6": [0.1600, 0.0944, 0.0612, 0.0443, 0.0341, 0.0289, 0.0208],
}
# Unsafe-deviation rate by number of survivors (the same cell)
CH_UNSAFE = {
    "G1": [0.602, 0.568, 0.520, 0.515, 0.491, 0.477, 0.448],
    "G2R": [0.392, 0.342, 0.272, 0.265, 0.239, 0.226, 0.196],
    "G2": [0.000, 0.000, 0.000, 0.003, 0.003, 0.002, 0.002],
    "G3": [0.000, 0.089, 0.076, 0.066, 0.050, 0.058, 0.057],
    "G4": [0.000, 0.000, 0.000, 0.004, 0.000, 0.001, 0.002],
    "G5": [0.272, 0.306, 0.261, 0.268, 0.265, 0.257, 0.260],
    "G6": [0.625, 0.555, 0.500, 0.463, 0.438, 0.423, 0.392],
}


def chart_channel():
    import math as _m
    lo, hi = _m.log(5), _m.log(100)
    xpos = lambda i: 0.04 + 0.92 * (_m.log(CH_EVID[i]) - lo) / (hi - lo)
    series = [
        {"name": "BCA tail certificate", "color": "--toy-c1", "values": CH["G2"], "mark": "circle"},
        {"name": "Split-conformal certificate", "color": "--toy-c3", "values": CH["G3"], "mark": "diamond"},
        {"name": "Plain average", "color": "--toy-c2", "values": CH["G5"], "mark": "square"},
        {"name": "Refuse thin data (reads no survivors)", "color": "--toy-ref", "values": CH["range"],
         "mark": "circle", "dash": True, "hollow": True},
    ]
    p29 = 0.04 + 0.92 * (_m.log(29) - lo) / (hi - lo)
    aria = ("Experiment 2, regret by number of survivors (line chart). Pooled regret against the number of "
            "survivors, 5 to 100, for three gates and for refusing thin data; means over 200 seeds, both harm shapes and ten worlds. Refusing thin data stays "
            "at 0.1215. The BCA tail certificate equals it up to 20 survivors, then drops to 0.0485 at 30 and 0.0373 "
            "at 100. The split-conformal certificate is 0.1215 at 5 survivors, 0.0318 at 10 and 0.0207 at 100. The "
            "plain average is lowest at every count, from 0.0297 down to 0.0065. Horizontal axis: survivors, a count, "
            "on a logarithmic scale. Vertical axis: pooled regret, in return per episode, 0 to 0.14. The table "
            "below the chart has every rule's numbers.")
    svg, leg, tpl = line_chart(
        "toy-ch-a", aria, CH_EVID, [str(x) for x in CH_EVID], xpos, series, 0.0, 0.14,
        [0, 0.02, 0.04, 0.06, 0.08, 0.10, 0.12, 0.14], lambda t: f"{t:.2f}",
        "Survivors: held-out tries of venture at the gap (log scale)",
        "Pooled regret (return per episode)",
        lambda i: f"{CH_EVID[i]} survivors: pooled regret", lambda v: f"{v:.4f}",
        guides=[(p29, "29 survivors", "start")],
        legend_note="Lower is better; 0 is the oracle, which knows the world. The dashed line at 29 survivors is the "
                    "fewest a tail certificate can use. The two width rules sit far above this chart, at 0.29 to 0.43, "
                    "and the other gates between these lines (table below).",
        labels=[{"series": 3, "text": "Refuse thin data", "cols": [6, 5, 4]},
                {"series": 0, "text": "BCA tail", "cols": [5, 4, 6, 3]},
                {"series": 1, "text": "Split conformal", "cols": [2, 1, 3, 4, 5]},
                {"series": 2, "text": "Plain average", "cols": [5, 6, 4, 3, 2, 1, 0]}])
    return chartbox("toy-second-chart", svg, leg, tpl,
                    "Experiment 2: regret falls as gates read more survivors; refusing thin data cannot use them, and "
                    "the tail certificate cannot use fewer than 29. Means over 200 seeds, both harm shapes and ten "
                    "worlds.")


def table_channel_regret():
    names = [("range", "Refuse thin data (reads no survivors)"), ("A1", "BCA's width as a penalty at every action"),
             ("A2", "BCA's width as the multiplier on the pull"), ("G1", "Step-1 gate"),
             ("G1L", "Step-1 gate, compared with context-1 rows only"), ("G2R", "BCA harm test"),
             ("G2", "BCA tail certificate"), ("G3", "Split-conformal certificate"), ("G5", "Plain average"),
             ("G6", "Harm test on the average (t-test)")]
    rows = [[nm] + [f"{v:.4f}" for v in CH[k]] for k, nm in names]
    return table("toy-second-regret", "The chart's numbers, and every other rule's: pooled regret by number of "
                 "survivors (both shapes, ten worlds, 200 seeds). Lower is better; the oracle's is 0.",
                 ["Rule"] + [f"{n} survivors" for n in CH_EVID], rows, num_cols=tuple(range(1, 8)))


def table_channel_case():
    rows = [
        ["Step-1 gate", "75,000 of the 1,000,000 pairs of draws differ by at most 0.10; 50,000 (5%) let it venture",
         "venture"],
        ["BCA harm test", "75 of 1,000 draws cross at or below the claimed advantage; 50 let it venture", "venture"],
        ["Harm test on the average (t-test)", "upper bound on the average outcome 1.659, above 1.004", "venture"],
        ["Plain average", "average outcome 0.827, below 1.004", "fall back"],
        ["Exact binomial certificate", "5 of 30 below 1.004; 30 tries allow 0", "fall back"],
        ["BCA tail certificate", "threshold 6.883 (the largest score), above the claimed advantage 0.443",
         "fall back"],
        ["Split-conformal certificate", "28th smallest of the 30 scores, 6.576, above 0.443", "fall back"],
    ]
    return table("toy-second-case", "One case, every gate: seed 0, 30 survivors, tail shape, advantage −0.25 (17.9% "
                 "of outcomes below 1.0, average 0.75; falling back is right). From the lesson version of "
                 "<code>bca_findings.ipynb</code>, which works each decision from scratch and checks it against the "
                 "run's record.", ["Gate", "The number it rests on", "Decision"], rows,
                 note="The three gates that ventured ask for proof of harm, and 30 tries were not enough to prove it; "
                      "the four that fell back ask for proof of safety, or (the plain average) look at the average.")


def part_second():
    o = ['<h3 id="toy-second">Experiment 2: where can BCA change a decision?</h3>']
    o.append('<p class="status"><span class="tag acc">exploratory</span> <code>bca_channel_toy.ipynb</code>: kill rules '
             'written down and fingerprinted (<code>bb9e0325…dbc3</code>) before the first run, by the notebook\'s own '
             'record; the repository cannot confirm the order, because the notebook was first committed on 15 Oct, '
             'after its runs. 200 toy seeds.</p>')
    o.append('<p>This notebook adds 5 to 100 held-out tries of venture at the gap, the <i>survivors</i> (named after '
             'ThrusterChain\'s surviving full-throttle steps), which the critic never trains on. The critic guesses '
             'about 1.5 there, against 1.0 for safe. Ten worlds differ only in the truth at the gap, one for each '
             '<i>advantage</i> (how much venture beats safe on average there): +1.0, +0.5, +0.25, +0.1, −0.1, −0.25, '
             '−0.5, −1.0, −2.0 and −4.6. Each comes in two shapes: <i>shortfall</i> (every outcome a little worse) and '
             '<i>tail</i> (a rare catastrophe of −5). Rewards carry noise (standard deviation 0.2). Every world and '
             'shape uses the same recording and the same random numbers behind the survivors.</p>')
    o.append(calc("Worked: the tail shape at advantage −0.25", [
        (None, "Venture pays −5 with chance <i>p</i> = (1 − advantage) ÷ 7, and 2 otherwise:"),
        ("<i>p</i>", f'{fr("1 − (−0.25)", "7")} = {fr("1.25", "7")} = 0.179'),
        ("average", "0.179 × (−5) + 0.821 × 2 = 2 − 7 × 0.179 = 0.75 = 1 + (−0.25)"),
    ], so="both shapes have the same average at each advantage; here it is 0.75, below safe's 1, so falling back is "
          "right. In the shortfall shape 89.4% of outcomes fall below 1.0, in the tail shape 17.9%."))
    o.append('<p><i>Regret</i> is the expected return lost against the best choices, per episode at the test mix; 0 is '
             'perfect. The <i>room</i> is the regret of the best rule that never reads the survivors, refusing choices '
             'with fewer than 20 training rows: 0.1215. A rule\'s <i>gain</i> is the share of the room it closes. A '
             'world is <i>unsafe</i> if more than 10% of venture\'s outcomes at the gap fall below safe\'s 1.0, and a '
             'rule\'s <i>unsafe-deviation rate</i> is how often it ventures at the gap in unsafe worlds.</p>')
    o.append(calc("Worked: the room", [
        (None, "Falling back at the gap loses the advantage whenever it is positive (+1, +0.5, +0.25 and +0.1 of the "
               "ten), in context 1, which the test meets 65% of the time:"),
        ("at the gap", f'0.65 × {fr("1 + 0.5 + 0.25 + 0.1", "10")} = 0.65 × 0.185 = 0.12025'),
        ("elsewhere", "0.00130 (in 2 of the 200 seeds the critic's noise flipped a first-stage choice)"),
        ("room", "0.12025 + 0.00130 = 0.12155, printed as 0.1215"),
    ], so="the room is what the <i>oracle</i>, a rule that knows the world, wins back. A gain of 1 means the oracle's "
          "choices, 0 means no better than refusing thin data, below 0 means worse. A gate decides only the gap, so "
          "it can close at most 0.12025 ÷ 0.12155 = 98.9% of the room."))
    o.append('<p><b>The rules.</b> BCA\'s width as built is subtracted from the critic\'s value at every action, or '
             'multiplies a pull toward the recorded actions, as in the learners BCA is added to, such as TD3+BC '
             '(<a href="#m-td3bc">Methods</a>). '
             '<i>Gates</i> read the survivors and decide only the gap: venture there, or fall back to safe. The '
             '<i>step-1 gate</i> copies the restructured BCA\'s first step on ThrusterChain (<a href="#toy-meaning">'
             'below</a>): fall back if BCA is 95% sure that the survivors\' over-predictions (the critic\'s value minus '
             'the outcome) exceed the other last-stage rows\' by over 0.10. The <i>harm test</i> falls back if BCA is '
             '95% sure that over 10% of outcomes fall below the critic\'s value of safe; the <i>tail certificate</i> '
             'ventures only if BCA is 95% sure that 90% beat it. Rivals: the same certificate by split conformal or by '
             'the <i>exact binomial test</i>, and the <i>plain average</i>, which ventures if the survivors\' average '
             'reward is at least the critic\'s value of safe. As controls, three gates are fed shuffled rows.</p>')
    o.append(finding("toy-second-brief", "Experiment 2 in four parts",
                     "Some way of using BCA's calibrated errors changes a decision for the better where simpler rules "
                     "cannot.",
                     "Every rule against refusing thin data, pooled over 200 seeds, both shapes, ten advantages and "
                     "seven survivor counts (28,000 cases per rule). Six tests, each with its kill rule (table below), "
                     "fixed before the run.",
                     "The widths did worse than refusing thin data (shelved); every gate that read the survivors closed "
                     "38% to 87% of the room, and shuffled rows erased the gain. With equal weights BCA decided as the "
                     "exact binomial test in 99.6% of cases (shelved). For return the plain average was best; for "
                     "safety only the tail certificates kept venturing rare in every unsafe world.",
                     "The width is measured at recorded actions and the gap has none, so it cannot see the gap's "
                     "outcomes. A gate compares the survivors' own outcomes with the value of safe, which the critic "
                     "knows well: the critic's guess drops out (worked below)."))
    tests = [
        ["1. BCA's width at every action helps", "It closes under a quarter of the room", f"{M}2.50 and {M}2.09 of the room", TAG_SHELVED],
        ["2. The step-1 gate helps by reading the survivors", "It closes under half the room, or shuffled rows keep over half its gain", "+0.58 of the room; shuffled rows keep none", TAG_ALIVE],
        ["3. Comparing with safe's value beats step 1's margin", "The harm test's regret is not below the step-1 gate's", "0.028 against 0.051; against 0.029 with a wider comparison group", TAG_ALIVE],
        ["4a. BCA's posterior decides differently from the exact binomial test", "They agree in 99% of cases or more", "99.6% agreement", TAG_SHELVED],
        ["4b. BCA makes venturing at the gap safer than split conformal", "Split conformal ventures in unsafe worlds under 5 percentage points more often", "5.7% against 0.2%; fragile", TAG_NARROW],
        ["4c. Safety needs a 90% share, not an average", "The plain average ventures in tail-shape unsafe worlds under 5 percentage points more often", "29.0% against 0.2%", TAG_ALIVE],
    ]
    o.append(table("toy-pooled-table", "Pooled over 200 seeds, both shapes, ten worlds and seven survivor counts: "
                   "28,000 cases per rule.", ["Test: the idea", "Shelved if", "Result", "Verdict"], tests))
    o.append(calc("Worked: why the critic drops out of a gate", [
        (None, "A survivor's score is the critic's value at the gap minus its outcome, <i>s</i> = <i>Q</i><sub>gap</sub> "
               "− <i>r</i>. The critic's claimed advantage is <i>Q</i><sub>gap</sub> − <i>Q</i><sub>safe</sub>. "
               "Subtract:"),
        ("claimed − <i>s</i>", "(<i>Q</i><sub>gap</sub> − <i>Q</i><sub>safe</sub>) − (<i>Q</i><sub>gap</sub> − <i>r</i>)"),
        ("", "<i>r</i> − <i>Q</i><sub>safe</sub>"),
        (None, "Seed 0: the critic values the gap at 1.447 and safe at 1.004, a claimed advantage of 0.443. In the "
               "tail shape at advantage −0.25, 5 of its 30 survivors fall below 1.004. With 30 tries the tail "
               "certificate allows none (<i>x</i>(30) = 0):"),
        ("so", "5 &gt; 0: the tail certificate falls back, as the run recorded"),
    ], so="every gate except the step-1 gate is really a test of the survivors' raw outcomes against safe's value, "
          "which the critic estimates well because safe is in the data."))
    o.append(table_channel_case())
    o.append(chart_channel())
    o.append(table_channel_regret())
    unsafe_rows = []
    names = [("G2", "BCA tail certificate"), ("G4", "Exact binomial certificate"), ("G3", "Split-conformal certificate"),
             ("G5", "Plain average"), ("G2R", "BCA harm test"), ("G1", "Step-1 gate"), ("G6", "Harm test on the average (t-test)")]
    for k, nm in names:
        unsafe_rows.append([nm] + [pct(v) for v in CH_UNSAFE[k]])
    o.append(table("toy-second-unsafe", "Unsafe-deviation rate by number of survivors: how often each gate ventured at "
                   "the gap in unsafe worlds (both shapes, 200 seeds). Lower is safer.",
                   ["Gate"] + [f"{n} survivors" for n in CH_EVID], unsafe_rows, num_cols=tuple(range(1, 8)),
                   note="The tail certificates cannot venture below 29 survivors, so their first three columns are 0. "
                        "In its single worst unsafe world, split conformal ventured in 32% to 38% of seeds with 10 or "
                        "more survivors; the BCA tail certificate in at most 3%, the exact binomial certificate in at "
                        "most 3%, the plain average in 100%."))
    o.append('<p><b>With equal weights, BCA is a textbook test.</b> Its multiplier is then simply one of the sorted '
             'held-out scores, picked by counting so that it holds for 90% of new scores with 95% confidence, for the '
             'data in hand. This long-known rule is Wilks\' (1941) tolerance limit; in conformal terms it is the '
             'training-conditional split-conformal bound (Vovk, 2012), and as a test it is the exact one-sided '
             'binomial test. It certifies nothing below 29 rows: with 28 rows, the chance that even the largest of '
             'them lies below the 90% point is 0.9<sup>28</sup> = 0.052, above 5%.</p>')
    o.append('<p><b>What survived: gates that read the held-out outcomes of venturing at the gap, compared with the '
             'value of safe, the fallback.</b> Every gate closed 38% to 87% of the room; fed shuffled rows, two did '
             'far worse than refusing thin data and one merely matched it. Compared with safe\'s value, the critic '
             'drops out, leaving the raw outcomes against a value the critic knows well, since safe is in the data. '
             'For return the plain average was best (regret 0.016); for safety only the tail certificate, by BCA or '
             'by the exact binomial test, kept venturing rare in every unsafe world (at most 3% of seeds), at 0.060 '
             'more regret.</p>')
    o.append('<h4 id="toy-shelved">Shelved</h4>')
    o.append(finding("toy-shelved-width", "BCA's width at every action (test 1)",
                     "BCA's calibrated width puts caution where the critic is wrong.",
                     "The width as a penalty, and as the multiplier on the pull, against refusing thin data; shelved "
                     "below a quarter of the room.",
                     "Regret 0.425 and 0.375, against 0.122, though the widths held 90.2% and 93.5% of fresh recorded "
                     "rows. No strength tried in the re-check rescued them.",
                     "The width is measured at recorded actions, and the gap has none. The survivors only move one "
                     "multiplier shared by every action, which also penalizes good but noisy actions."))
    o.append(finding("toy-shelved-posterior", "\"BCA's posterior decides differently from the exact binomial test\" (test 4a)",
                     "BCA's Bayesian posterior adds information that a classical test lacks.",
                     "The tail certificate by BCA's 1,000 random draws against the exact binomial test, case by case; "
                     "shelved at 99% agreement or more.",
                     "99.6% agreement; computed exactly, BCA agreed every time.",
                     "The identity above: with equal weights, BCA's rule is the exact binomial test. Both refuse in 91% "
                     "of cases, so a high agreement rate alone would prove little; the identity settles it."))
    o.append('<h4 id="toy-recheck">What an independent re-check found</h4>')
    o.append('<ul class="leads">\n'
             '      <li><b>Confirmed.</b> Three separate re-checks reproduced every printed result, matched the '
             'notebook\'s BCA and step-1 gate to the project\'s own BCA code, proved the identity, and found no verdict '
             'change on 200 fresh seeds.</li>\n'
             '      <li><b>Corrected.</b> Test 3 leans on the toy\'s comparison group: last-stage rows only, 10.7% of '
             'them a noisy cell\'s low outcomes, so the group\'s 90% point swung from 0.35 to 1.50 across seeds. With '
             'every other held-out row, as ThrusterChain\'s step 1 uses, test 3\'s margin shrinks from 0.023 to 0.001. '
             'And "nothing below 29" holds only for exact computation: with 1,000 random draws BCA sometimes certifies '
             'at 26 to 28 rows, which is why the project\'s BCA code requires 36.</li>\n'
             '      <li><b>Caveats.</b> Test 4b flips to shelved at other noise levels or a 95% coverage target. Test '
             '4c\'s rival had no confidence level; against a 95% certificate on the average, the tail certificate was '
             'still 11 percentage points safer.</li>\n    </ul>')
    o.append(details_second())
    return "\n    ".join(o)


def details_second():
    rule_rows = [
        ["No safeguard: the critic's best action everywhere", "greedy", "0.5505", f"{M}3.53", "100%", "100%"],
        ["Refuse thin data (fewer than 20 training rows)", "range limit", "0.1215", "0", "0%", "0%"],
        ["Count penalty, 1/√(training rows + 1)", "count penalty", "0.1222", f"{M}0.01", "0%", "0%"],
        ["Pull toward the recorded actions, no BCA", "host BC actor", "0.4939", f"{M}3.06", "86.5%", "86.5%"],
        ["BCA's width as a penalty at every action", "A1", "0.4253", f"{M}2.50", "0.3%", "0.3%"],
        ["BCA's width as the multiplier on the pull", "A2", "0.3755", f"{M}2.09", "64.9%", "65.3%"],
        ["Step-1 gate", "G1", "0.0508", "0.58", "51.7%", "42.4%"],
        ["Step-1 gate, compared with context-1 rows only", "G1 local reference", "0.0329", "0.73", "24.1%", "42.4%"],
        ["Harm test", "G2R", "0.0282", "0.77", "27.6%", "42.4%"],
        ["Tail certificate", "G2", "0.0758", "0.38", "0.2%", "0.2%"],
        ["Tail certificate by split conformal", "G3", "0.0392", "0.68", "5.7%", "7.8%"],
        ["Tail certificate by the exact binomial test", "G4", "0.0753", "0.38", "0.1%", "0.1%"],
        ["Plain average", "G5", "0.0157", "0.87", "27.0%", "29.0%"],
        ["Harm test on the average, by a one-sided 95% t-test", "G6", "0.0634", "0.48", "48.5%", "68.1%"],
        ["Step-1 gate, fed shuffled rows", "G1 shuffled", "0.4356", f"{M}2.58", "89.9%", "92.0%"],
        ["Harm test, fed shuffled rows", "G2R shuffled", "0.3543", f"{M}1.91", "71.0%", "79.1%"],
        ["Tail certificate, fed shuffled rows", "G2 shuffled", "0.1211", "0.00", "0.4%", "0.5%"],
        ["Oracle: knows the world", "oracle switch", "0", "1", "25.0%", "25.0%"],
    ]
    t = table("toy-second-all", "Experiment 2, every rule, pooled over all 28,000 cases. Gain is the share of the room closed (1 is "
              "the oracle, 0 is refusing thin data, below 0 is worse). Unsafe is how often the rule ventures at the gap "
              "in unsafe worlds: all of them, and the tail-shape ones.",
              ["Rule", "Label in the notebook", "Regret", "Gain", "Unsafe", "Unsafe, tail shape"], rule_rows,
              num_cols=(2, 3, 4, 5))
    inner = [t]
    inner.append('<p class="note">The oracle ventures in 25% of unsafe cases because two of the eight unsafe worlds of '
                 'each shape (where venture beats safe by 0.25 and 0.10 on average) are ones where venturing pays on '
                 'average. With 30 survivors, the step-1 gate ventured in 94% to 98% of seeds in the shortfall worlds '
                 'that lose 0.10 to 0.50 on average; the harm test stopped all of those.</p>')
    inner.append('<p><b>The re-check of experiment 2, in full.</b> Confirmed:</p>')
    inner.append('<ul>\n'
                 '          <li>A re-run reproduced 137 of the 139 printed lines; the other two are clock times. The kill '
                 'rules\' fingerprint matches their text, but whether that text was written before the first run cannot '
                 'be checked from the repository: the notebook was first committed on 15 Oct, after its runs. The '
                 're-check read an earlier, uncommitted copy; this page\'s own re-run of the committed version '
                 '(<code>6320224</code>) gave the same 137 lines, and every number this part quotes from the notebook '
                 'matches it. The re-check\'s own numbers below (fresh seeds, other settings) are the re-check\'s and '
                 'were not run again here.</li>\n'
                 '          <li>The notebook\'s BCA gives the same band multiplier and the same result in each of its '
                 '1,000 random draws as the project\'s <code>calibration/wbcp.py</code> on identical random numbers (400 '
                 'of 400 random score sets in one check, 3,000 of 3,000 in another). Its step-1 gate gives the same '
                 'decision as the project\'s step-1 code in 3,360 of 3,360 cases.</li>\n'
                 '          <li>The identity was proved, and the project\'s BCA code, run with up to a million draws, '
                 'landed on the predicted sorted score in 28 of 28 score sets. Computed exactly, BCA agreed with the exact '
                 'binomial test in all 28,000 cases, so test 4a\'s 110 disagreements were all sampling slips, each at the '
                 'boundary count. The critic\'s dropping out of the harm test held in 400,000 of 400,000 outcomes.</li>\n'
                 f'          <li>On fresh seeds 200–399: test 1, {M}2.47 and {M}2.05 of the room; test 2, +0.575; test 3, '
                 '0.029 against 0.052; test 4a, 99.5% agreement; test 4b, 5.8 percentage points; test 4c, 30.2 '
                 'percentage points. No verdict changed.</li>\n        </ul>')
    inner.append('<p>Corrected or qualified:</p>')
    inner.append('<ul>\n'
                 '          <li>Test 3: given every other held-out row of both stages, the step-1 gate\'s regret falls '
                 'from 0.051 to 0.029, against the harm test\'s 0.028, and its 94% to 98% harmful ventures in shortfall '
                 'worlds fall to 0% to 2%. Test 3 stays alive by its rule, by 0.001 instead of 0.023.</li>\n'
                 '          <li>"Nothing below 29" holds for exact computation. With the project\'s 1,000 random draws, '
                 'BCA gives a finite multiplier at 26 to 28 rows in 3% to 40% of cases, with only 93.5% to 94.8% real '
                 'confidence, and none at 29 and 30 rows in 30% and 10% of cases.</li>\n'
                 '          <li>The width does respond, weakly and at every action: up to about 45 survivors the shared '
                 'multiplier barely moves, but with 100 survivors in the worst worlds it rises from about 1.5 to 7–9, and '
                 'the gap decision of the width used as the multiplier on the pull changes across the worlds in 20% of '
                 'seed and shape pairs at 60 survivors and 35% at 100. Refusing the merely noisy ventures elsewhere cost '
                 'the width penalty about 0.30 of its regret.</li>\n'
                 '          <li>The tail certificate\'s worst unsafe world was 3% of seeds on seeds 0–199, but 3.5% (4.5% '
                 'for the same certificate by the exact binomial test) on fresh seeds. That is within the 5% the theory '
                 'allows, which holds against the critic\'s value of safe (0.96 to 1.04 across seeds) rather than the '
                 'true 1.0, and every such venture was in a world where venturing pays on average. Split conformal\'s '
                 'worst world (32% to 38% of seeds) is also one where venturing pays; where venturing loses on average, '
                 'it ventured in at most about 10% of seeds with 30 or more survivors.</li>\n'
                 '          <li>The regret of the width used as the multiplier on the pull is 0.375, not 0.376 as the '
                 'notebook\'s write-up says. The shuffled controls drew from the survivors and other rows pooled, so 1% '
                 'to 14% of each shuffled set were real survivors; that flattered the controls, so test 2 was '
                 'conservative. The oracle also fixes 0.0013 of regret away from the gap, so every gain is about 1% too '
                 'low. The notebook\'s check 5 compares the binomial formula with itself, so the identity rests on the '
                 'proof and on the project\'s code, not on that check.</li>\n        </ul>')
    inner.append('<p>Robustness, 100 seeds per setting:</p>')
    inner.append('<ul>\n'
                 '          <li>Tests 1, 2, 4a and 4c held at a reward noise of 0.1 and 0.4 (instead of 0.2), at a 95% '
                 'coverage target (instead of 90%), with 4,000 draws, at an even test mix of contexts and at every width '
                 'strength tried. Test 3 kept its sign everywhere, but its size fell to 0.003 at a noise of 0.4, or with '
                 'a quieter comparison group.</li>\n'
                 '          <li>Test 4b was shelved at a noise of 0.1 (4.8 percentage points) and 0.4 (4.6), and at a '
                 '95% coverage target (0.9), and dropping any one of three borderline worlds also shelves it. Resampling '
                 'the original seeds puts its margin anywhere from about 4.8 to 6.2 percentage points. Its margin comes '
                 'from "unsafe" worlds where venturing pays on average; where venturing loses on average, split '
                 'conformal ventured only 1.2 to 1.6 percentage points more often than the tail certificate.</li>\n'
                 '          <li>Test 4c: a 95% certificate on the average, added in the re-check, had regret 0.023 and '
                 'ventured in 11.6% of tail-shape unsafe cases, against 0.078 and 0.2% for the tail certificate and '
                 '0.018 and 30.4% for the plain average (fresh seeds). So about 60% of test 4c\'s difference comes from '
                 'asking for 95% confidence at all, not from asking about a 90% share.</li>\n        </ul>')
    return ('<details id="toy-more-detail">\n      <summary>More detail: every rule\'s numbers and the full re-check of '
            'experiment 2</summary>\n      <div class="inner">\n        ' + "\n        ".join(inner) +
            '\n      </div>\n    </details>')


# ====================================================================================== 6. experiment 3
MG_CAPS = ["1%", "2%", "5%", "10%", "20%", "no cap"]
# Gain per situation on seeds 100-199 after tuning on seeds 0-99 to each cap on worst-world harm
# (bca_mean_gate_toy.ipynb, cell 23; "no cap" is the notebook's cap 100%)
MG = {
    "mean": [0.030, 0.061, 0.084, 0.103, 0.137, 0.137],
    "avg": [0.000, 0.021, 0.040, 0.098, 0.136, 0.136],
    "shrunk": [0.001, 0.019, 0.019, 0.056, 0.087, 0.151],
    "plain": [0.001, 0.007, 0.007, 0.036, 0.078, 0.158],
    "tail": [0.003, 0.003, 0.017, 0.017, 0.027, 0.046],
    "ttest": [None, None, None, None, 0.040, 0.158],
    "hoeff": [0.005, 0.017, 0.017, 0.017, 0.017, 0.017],
    "bern": [0.006, 0.006, 0.006, 0.006, 0.006, 0.006],
    "split": [0.006, 0.006, 0.006, 0.019, 0.019, 0.062],
    "count": [0.000, 0.000, 0.000, 0.011, 0.011, 0.138],
}


def chart_mean_gate():
    xpos = lambda i: 0.05 + 0.9 * i / 5
    series = [
        {"name": "BCA mean gate", "color": "--toy-c4", "values": MG["mean"], "mark": "tri"},
        {"name": "BCA's cautious average", "color": "--toy-c3", "values": MG["avg"], "mark": "trid"},
        {"name": "Plain average", "color": "--toy-c2", "values": MG["plain"], "mark": "square"},
    ]
    aria = ("Experiment 3, gain at each cap on harm (line chart). Gain per situation after tuning each gate to a "
            "cap on its worst-world harm rate, seeds 100 to 199, for three gates. BCA's mean gate earns the most at every cap from 1% to 20%: "
            "0.030, 0.061, 0.084, 0.103 and 0.137. BCA's cautious average comes close at 10% and 20% (0.098 and "
            "0.136). The plain average earns 0.001 to 0.078 under a cap and the most with no cap, 0.158. Horizontal "
            "axis: the cap on the worst world's harm rate, the largest share of harmful proposals a gate may accept "
            "in any world. Vertical axis: gain per situation, in return, 0 to 0.16. The table below the chart has "
            "every gate's numbers.")
    svg, leg, tpl = line_chart(
        "toy-mg-b", aria, MG_CAPS, MG_CAPS, xpos, series, 0.0, 0.16,
        [0, 0.04, 0.08, 0.12, 0.16], lambda t: f"{t:.2f}",
        "Cap on the worst world's harm rate",
        "Gain per situation (return)",
        lambda i: f"Cap {MG_CAPS[i]}: gain per situation, seeds 100–199", lambda v: f"{v:.3f}",
        guides=[(0.05 + 0.9 * 2 / 5, "claim 5", "start")], h=310,
        legend_note="Higher is better. A gate's harm rate in a world is the share of harmful proposals it accepts; the "
                    "cap limits the highest harm rate over the 36 worlds. Each gate takes its best setting on seeds 0–99 "
                    "within the cap, then is scored on "
                    "seeds 100–199; the dashed line marks the 5% cap that claim 5 uses. Every other gate's numbers are "
                    "in the table below.",
        labels=[{"series": 0, "text": "BCA mean gate", "cols": [2, 3, 1]},
                {"series": 1, "text": "Cautious average", "cols": [3, 2, 1, 4]},
                {"series": 2, "text": "Plain average", "cols": [4, 3, 1, 2, 0]}])
    return chartbox("toy-third-chart", svg, leg, tpl,
                    "Experiment 3: once every gate is held to the same safety level, BCA's mean gate earns the most; "
                    "only without a safety level does the plain average earn more.")


def table_mean_caps():
    names = [("mean", "<b>BCA mean gate</b>"), ("avg", "BCA's cautious average"), ("shrunk", "Shrunk average"),
             ("plain", "Plain average"), ("ttest", "t-test"), ("hoeff", "Hoeffding"), ("bern", "Empirical Bernstein"),
             ("tail", "BCA tail gate"), ("split", "Split-conformal tail"), ("count", "Count rule")]
    rows = [[nm] + ["none qualifies" if v is None else f"{v:.3f}" for v in MG[k]] for k, nm in names]
    return table("toy-third-caps", "The chart's numbers, and every other gate's: gain per situation on seeds 100–199 "
                 "after tuning on seeds 0–99 to each cap on the worst world's harm rate. On seeds 100–199 every tuned "
                 "gate stayed within its cap.", ["Gate"] + [f"Cap {c}" if c != "no cap" else "No cap" for c in MG_CAPS], rows,
                 num_cols=tuple(range(1, 7)),
                 note="“None qualifies”: no setting of the t-test stayed within the cap on seeds 0–99.")


def part_third():
    o = ['<h3 id="toy-third">Experiment 3: can a BCA gate raise return? The mean gate</h3>']
    o.append('<p class="status"><span class="tag acc">exploratory</span> <code>bca_mean_gate_toy.ipynb</code>: kill rules '
             'written down and fingerprinted (<code>50a0f49d…b1d3</code>) before the first run, by the notebook\'s own '
             'record; 200 toy seeds of 20 situations, seeds 0–99 used only to tune settings and seeds 100–199 for every '
             'result.</p>')
    o.append('<p>Experiment 2\'s tail gate asks whether 90% of outcomes beat the fallback, the wrong question for '
             'return. This experiment tests the <i>mean gate</i>, which asks about the average, in a world of its own '
             'built in four steps:</p>')
    steps = [
        ('Situations and candidates.', '<p>In each <i>situation</i> the policy can take a safe action, worth 0, or one '
         'of several <i>candidate</i> actions. Each candidate has a true average gain over safe: above 0 it is '
         '<i>good</i>, below 0 <i>harmful</i>. Only the scoring knows these numbers.</p>'),
        ('Two sets of tries.', '<p>The recording holds 5 to 100 tries of each candidate, plus a separate held-out set of '
         'the same size.</p>'),
        ('The critic proposes, the gate decides.', '<p>Looking only at the first set, the critic proposes the candidate '
         'with the best average, if that average is above 0. With many candidates it is often fooled by one that got '
         'lucky. Looking only at the proposal\'s held-out tries, a gate accepts it or stays safe. The score is the true '
         'average gain of the final choice; staying safe scores 0.</p>'),
        ('Three shapes, 36 worlds.', '<p>A try\'s gain is <i>steady</i> (the true average plus noise of spread 1), a '
         '<i>rare disaster</i> (a small gain of 0.2 to 0.6, but sometimes a loss of 8) or a <i>rare jackpot</i> (a small '
         'loss of 0.1 to 0.4, but sometimes a gain of 8). Gains stay inside a known range: −3 to 3, −8 to 3 and −3 to 8; '
         'the bottom of the range is the floor. A <i>world</i> combines one shape, one share of good candidates (10%, '
         '30% or 50%) and one number of candidates per situation (1, 3, 10 or 30): 36 worlds.</p>'),
    ]
    o.append('<ol class="toy-steps">' + "".join(f'\n      <li><b>{t}</b>{b}</li>' for t, b in steps) + '\n    </ol>')
    o.append('<p>The rivals: the plain average of the held-out tries (yes if above 0); a <i>shrunk average</i>, pulled '
             'toward the average of all candidates in the seed\'s 20 situations; <i>BCA\'s cautious average</i>, (sum of '
             'tries + floor) ÷ (tries + 1), which counts the worst possible gain as one extra try; the classical 95% '
             'lower bounds on an average (the <i>t-test</i>, which assumes roughly bell-shaped outcomes; '
             '<i>Hoeffding</i>\'s bound, from the range alone; and <i>empirical Bernstein</i>\'s, from the range and the '
             'spread; the last two promise 5% for any outcomes inside the range); a count rule '
             '(yes with at least 30 held-out tries); BCA\'s tail gate and split conformal\'s; and the mean gate deciding '
             'all 20 situations together under a shared 10% harm budget (the Benjamini–Hochberg method). A gate\'s '
             '<i>harm rate</i> in a world is the share of harmful proposals it accepts; a world\'s <i>allowance</i> on a '
             '5% rate is 5% plus three standard errors for its number of harmful proposals.</p>')
    o.append(calc("Worked: why rare disasters fool the t-test", [
        (None, "A candidate gains 0.4 on 95% of tries and loses 10 on the rest. With 29 tries:"),
        ("true average", "0.95 × 0.4 + 0.05 × (−10) = 0.38 − 0.5 = −0.12: harmful"),
        ("chance of no loss", "0.95<sup>29</sup> = 0.23"),
        (None, "In those clean samples every try shows 0.4 and the spread is 0, so the t-test's bound is 0.4 and it "
               "says yes. Told that the floor is −10, the mean gate puts this share of its weighted averages below 0:"),
        ("clean sample", f'({fr("10", "10.4")})<sup>29</sup> = 0.32, far more than 0.05: no'),
    ], so="the t-test accepts this harmful action in about 23% of samples instead of 5%; the mean gate refuses every "
          "sample, because the extra weight on the floor stands in for the disaster a clean sample may be hiding."))
    o.append(finding("toy-third-brief", "Experiment 3 in four parts",
                     "BCA's mean gate keeps its 5% promise and, once every gate must be equally safe, earns the most.",
                     "Fourteen gates on 36 worlds; six claims, each with its kill rule (table below), fixed before the "
                     "run. For claim 5 each gate that has a knob takes its best setting on seeds 0–99 among those with "
                     "at most 5% harm in every world, then is scored on seeds 100–199.",
                     "Claims 1, 2, 3 and 5 alive: worst harm 0.3%; 17.4% of good proposals accepted against none for "
                     "Hoeffding and empirical Bernstein; the t-test accepted up to 31.3% of harmful ones; at equal harm "
                     "the mean gate earned 0.0845 per situation, against 0.0400 next. Claims 4 and 6 shelved. Without a "
                     "safety level the plain average earned more (0.158 against 0.045).",
                     "The mean gate uses the same random weights as BCA's tail gate, with the extra weight on the floor: "
                     "a guaranteed bound that still accepts (the same calculation as Gaffke's 2005 bound). But fresh "
                     "held-out tries already give an honest average, so extra caution costs return unless safety is "
                     "required."))
    claims = [
        ["1. BCA's mean gate keeps its promise", "Its harm rate is above the allowance in any world", "worst harm rate 0.3%; no world above its allowance", TAG_ALIVE],
        ["2. It is the tightest guaranteed mean gate", "It accepts under 5 points more of the good proposals than Hoeffding or empirical Bernstein", "17.4% against 0.0% and 0.0%", TAG_ALIVE],
        ["3. A t-test gate is not safe when disasters are rare", "Its harm rate is within the allowance in every rare-disaster world", "up to 31.3% of harmful proposals accepted", TAG_ALIVE],
        ["4. Where many candidates compete and few are good, the mean gate beats simple averaging", "Its gain over the 12 crowded worlds is not above both the plain and the shrunk average's", "0.0567 against 0.1829 and 0.1625", TAG_SHELVED],
        ["5. At equal harm, the mean gate earns the most", "Another tuned gate earns at least as much on seeds 100–199", "0.0845 against 0.0400 (BCA's cautious average)", TAG_ALIVE],
        ["6. Deciding all 20 situations together earns more", "It earns no more, or over 12% of its acceptances are harmful in some world", "0.0270 against 0.0453; harm 0.000", TAG_SHELVED],
    ]
    o.append(table("toy-third-claims", "Experiment 3's six claims, seeds 100–199. The crowded worlds have 10 or 30 "
                   "candidates and 10% or 30% good ones.", ["Claim", "Shelved if", "Result", "Verdict"], claims))
    res = [
        ["Always safe", "0.000", "0%", "0%", "0.0%", "0.0%"],
        ["No gate (the critic alone)", "0.138", "66%", "100%", "100.0%", "100.0%"],
        ["Count rule (30 tries or more)", "0.065", "31%", "32%", "43.2%", "12.4%"],
        ["Plain average", "0.158", "76%", "83%", "41.4%", "36.2%"],
        ["Shrunk average", "0.151", "73%", "69%", "90.0%", "16.4%"],
        ["BCA's cautious average", "0.136", "65%", "60%", "11.8%", "1.7%"],
        ["t-test", "0.080", "39%", "44%", "31.3%", "31.3%"],
        ["Hoeffding", "0.000", "0%", "0%", "0.0%", "0.0%"],
        ["Empirical Bernstein", "0.000", "0%", "0%", "0.0%", "0.0%"],
        ["<b>BCA mean gate</b>", "0.045", "22%", "17%", "0.3%", "0.0%"],
        ["BCA tail gate", "0.003", "2%", "1%", "0.0%", "0.0%"],
        ["Split-conformal tail", "0.019", "9%", "10%", "7.6%", "7.6%"],
        ["BCA mean gate, shared budget", "0.027", "13%", "10%", "0.0%", "0.0%"],
        ["Oracle gate (the proposal is truly good)", "0.208", "100%", "100%", "0.0%", "0.0%"],
    ]
    o.append(table("toy-third-results", "Every gate at its stated setting (5% where it has a level), seeds 100–199: "
                   "gain per situation pooled over the 36 worlds, its share of the oracle gate's gain, the share of good "
                   "proposals accepted, and the highest harm rate over all worlds and over the rare-disaster worlds.",
                   ["Gate", "Gain", "Of oracle", "Good accepted", "Worst harm", "Worst harm, rare disaster"], res,
                   num_cols=(1, 2, 3, 4, 5),
                   note="Picking the truly best candidate in every situation would gain 0.294: the critic's proposals "
                        "already lose part of that before any gate acts."))
    o.append(chart_mean_gate())
    o.append(table_mean_caps())
    tuned = [
        ["BCA mean gate", "level 0.2", "0.0864", "0.0845", "2.3%"],
        ["BCA's cautious average", "margin 0.5", "0.0410", "0.0400", "3.8%"],
        ["Shrunk average", "margin 0.5", "0.0165", "0.0187", "1.6%"],
        ["Hoeffding", "level 0.5", "0.0167", "0.0174", "0.1%"],
        ["BCA tail gate", "share 0.2", "0.0187", "0.0172", "2.9%"],
        ["Plain average", "margin 1.5", "0.0061", "0.0070", "0.8%"],
        ["Empirical Bernstein", "level 0.5", "0.0061", "0.0061", "0.0%"],
        ["Split-conformal tail", "share 0.05", "0.0064", "0.0056", "0.3%"],
        ["Count rule", "100 tries", "0.0001", "0.0003", "0.2%"],
        ["t-test", "none qualifies", "0", "0", "0.0%"],
    ]
    o.append(table("toy-third-tuned", "Claim 5: each gate's best setting with at most 5% harm in every world on seeds "
                   "0–99, scored on seeds 100–199. A level is the error share a bound allows (0.2 is 80% confidence), a "
                   "margin the gain an average must clear, a share the tail gates' allowed share below 0.",
                   ["Gate", "Chosen setting", "Gain, seeds 0–99", "Gain, seeds 100–199", "Worst harm, seeds 100–199"],
                   tuned, num_cols=(2, 3, 4)))
    o.append(finding("toy-third-shelved-crowded", "\"Where many candidates compete, the mean gate beats simple averaging\" (claim 4)",
                     "Where many candidates compete and few are good, the critic is often fooled by a lucky candidate, "
                     "so a cautious gate should earn more than simply checking the held-out average.",
                     "Pooled gain over the 12 crowded worlds; shelved unless the mean gate earned more than both the "
                     "plain and the shrunk average.",
                     "0.057 per situation, against 0.183 and 0.163.",
                     "The held-out tries are fresh, so their average is an honest estimate of a proposal's gain, however "
                     "lucky its training tries were. After that, extra caution mostly turns good candidates away: with "
                     "5–9, 10–19 and 20–39 tries the mean gate accepted 1%, 5% and 17% of good proposals, the plain "
                     "average 71%, 80% and 89%."))
    o.append(finding("toy-third-shelved-budget", "\"Deciding all situations together earns more\" (claim 6)",
                     "Deciding the 20 situations together, with a shared budget of 10% harmful acceptances, lets more "
                     "good candidates through than deciding each one alone at 5%.",
                     "Pooled gain against the mean gate deciding each situation alone; shelved unless it earned more, "
                     "with at most 12% harmful acceptances on average in every world.",
                     "0.027 per situation, against 0.045. Harm was not the problem: in every world under 0.1% of its "
                     "acceptances were harmful on average.",
                     "Deciding together lets more through only when many situations look clearly good at once. Here few "
                     "proposals pass at all, and then the shared bar is stricter than 5% per situation."))
    o.append('<p><b>Information only, written after the run.</b> BCA\'s cautious average has no knob. It earned 86% of '
             'the plain average\'s return (0.136 against 0.158) and lost return in none of the 36 worlds; the plain '
             'average and the t-test each lost return in 4 worlds, all of them the hardest (rare disasters, 10% good '
             'candidates), where the cautious average earned 0.008 to 0.048 per situation, the most of any real gate. '
             'The mean gate\'s promise rests on a July 2026 preprint by Vlassis and Thomas that is not yet peer '
             'reviewed; no world broke it here, which supports the proof but does not replace it.</p>')
    return "\n    ".join(o)


# ====================================================================================== 7. experiment 4
W_RUNGS = [0.0, 0.02, 0.05, 0.08, 0.10, 0.12, 0.15, 0.20]


def chart_weights():
    h = EX["harsh"]
    xpos = lambda i: 0.03 + 0.94 * W_RUNGS[i] / 0.20
    unsafe_p = (0.10 - EX["truth"]["usual_below"]) / (1 - EX["truth"]["usual_below"])   # share below = 0.10
    harm_p = 0.5 / 4.5                                                                  # average gain = 0
    s3 = lambda xs: ", ".join(f"{v:.3f}" for v in xs)
    tail = [
        {"name": "BCA tail gate, weighted", "color": "--toy-c1", "values": h["BCA tail, weighted"], "mark": "circle"},
        {"name": "BCA tail gate, unweighted", "color": "--toy-c1", "values": h["BCA tail, unweighted"],
         "mark": "circle", "dash": True, "hollow": True},
        {"name": "Weighted split conformal", "color": "--toy-c3", "values": h["weighted split conformal"],
         "mark": "diamond"},
    ]
    mean = [
        {"name": "BCA mean gate, weighted", "color": "--toy-c4", "values": h["BCA mean, weighted"], "mark": "tri"},
        {"name": "BCA mean gate, unweighted", "color": "--toy-c4", "values": h["BCA mean, unweighted"], "mark": "tri",
         "dash": True, "hollow": True},
        {"name": "Oracle (deviates when good on average)", "color": "--toy-ref", "values": h["oracle"],
         "mark": "square", "dash": True, "hollow": True},
    ]
    common = ("Experiment 4, harm in harsh conditions, strong shift, 120 held-out tries, 800 seeds. Horizontal axis: "
              "disaster chance at deployment (the rung), a probability, 0 to 0.20. Vertical axis: deviation rate, the "
              "share of seeds in which the gate deviates, 0 to 1. The table below the charts has the numbers.")
    aria_t = ("Experiment 4, the tail gates along the rungs (line chart). Deviation rate by disaster chance; unsafe "
              "from a disaster chance of 0.10. "
              f"The unweighted BCA tail gate: {s3(h['BCA tail, unweighted'])}. Weighted: {s3(h['BCA tail, weighted'])}. "
              f"Weighted split conformal: {s3(h['weighted split conformal'])}. " + common)
    aria_m = ("Experiment 4, the mean gates along the rungs (line chart). Deviation rate by disaster chance; "
              "harmful from a disaster chance of 0.12. "
              f"The unweighted BCA mean gate: {s3(h['BCA mean, unweighted'])}. Weighted: {s3(h['BCA mean, weighted'])}. "
              f"The oracle: {s3(h['oracle'])}. " + common)
    args = dict(xs=W_RUNGS, xlabels=[f"{r:.2f}" if r in (0.0, 0.05, 0.10, 0.15, 0.20) else "" for r in W_RUNGS], xpos=xpos, ylo=0.0, yhi=1.0,
                yticks=[0, 0.2, 0.4, 0.6, 0.8, 1.0], ytick_fmt=lambda t: f"{t:.1f}",
                xtitle="Disaster chance at deployment (the rung)", ytitle="Deviation rate (share of seeds)",
                tip_val=lambda v: f"{v:.3f}", hguides=[(0.05, "5%")], h=330)
    svg1, leg1, tpl1 = line_chart(
        "toy-wt-c", aria_t, series=tail,
        tip_title=lambda i: f"Disaster chance {W_RUNGS[i]:.2f}: deviation rate" + (" (unsafe)" if W_RUNGS[i] >= 0.10 else ""),
        guides=[(0.03 + 0.94 * unsafe_p / 0.20, "unsafe →", "start")],
        legend_note="Right of the dashed line the deviation is unsafe, and a tail gate that keeps its promise deviates "
                    "in at most about 5% of seeds.",
        labels=[{"series": 2, "text": "split conformal", "cols": [2, 3, 4, 5, 6], "optional": True},
                {"series": 1, "text": "unweighted", "cols": [4, 5, 3, 6]},
                {"series": 0, "text": "weighted", "cols": [2, 3, 1]}], **args)
    svg2, leg2, tpl2 = line_chart(
        "toy-wt-m", aria_m, series=mean,
        tip_title=lambda i: f"Disaster chance {W_RUNGS[i]:.2f}: deviation rate" + (" (harmful)" if W_RUNGS[i] >= 0.12 else ""),
        guides=[(0.03 + 0.94 * harm_p / 0.20, "harmful →", "start")],
        legend_note="Right of the dashed line the deviation is harmful on average, and a mean gate that keeps its "
                    "promise deviates in at most about 5% of seeds. Dashed lines with hollow markers ignore the shift.",
        labels=[{"series": 1, "text": "unweighted", "cols": [5, 6, 4]},
                {"series": 0, "text": "weighted", "cols": [2, 3, 1]},
                {"series": 2, "text": "oracle", "cols": [3, 2, 4]}], **args)
    a = chartbox("toy-fourth-chart", svg1, leg1, tpl1,
                 "The tail promise: when harm sits in conditions that deployment meets more often than the recording, "
                 "the unweighted tail gate keeps deviating into unsafe worlds; the weighted one stops.")
    b = chartbox("toy-fourth-chart-mean", svg2, leg2, tpl2,
                 "The mean promise: the same for the mean gates in harmful worlds. The oracle deviates exactly when "
                 "the deviation is good on average.")
    return '<div class="toy-pair">\n      ' + a + "\n      " + b + "\n    </div>"


def part_fourth():
    t = EX["truth"]
    h = EX["harsh"]
    o = ['<h3 id="toy-fourth">Experiment 4: do BCA\'s weights help under a shift?</h3>']
    o.append('<p class="status"><span class="tag acc">exploratory</span> <code>bca_weights_toy.ipynb</code>: kill rules '
             'written down and fingerprinted (<code>e32e04d7…f44e</code>) before the first run, by the notebook\'s own '
             'record; 800 toy seeds, seeds 0–399 used only to tune settings for claim 6 and seeds 400–799 to score it. '
             'One check imports the project\'s <code>calibration/wbcp.py</code> and confirms that the weighted tail gate '
             'decides as production BCA does, in 300 of 300 random cases.</p>')
    o.append('<p>With equal weights BCA is a textbook test (experiment 2); its one part that is not textbook is its '
             'weights. This experiment asks whether they keep BCA\'s promises when the held-out tries come from a '
             'different mix of conditions than deployment, in a world of its own:</p>')
    mix = [
        ["0 (mild)", "35%", "15%", "5%", "0.143"],
        ["1", "25%", "20%", "10%", "0.400"],
        ["2", "20%", "25%", "15%", "0.750"],
        ["3", "12%", "20%", "30%", "2.500"],
        ["4 (harsh)", "8%", "20%", "40%", "5.000"],
    ]
    mix_tbl = table("toy-fourth-mix", "The five conditions: their share among the recorded tries and at deployment "
                    "under each shift, and a try's exact weight under the strong shift.",
                    ["Condition", "Recorded tries", "Deployment, moderate shift", "Deployment, strong shift",
                     "Weight, strong shift"], mix, num_cols=(1, 2, 3, 4))
    steps = [
        ('One deviation, five conditions.', '<p>A policy wants to <i>deviate</i>, take a risky action instead of a '
         'well-recorded fallback, everywhere in a region of states, or fall back everywhere. The region has five '
         '<i>conditions</i>, from mild (0) to harsh (4).</p>'),
        ('A gain, usually small, sometimes a disaster.', f'<p>A try\'s gain over the fallback is usually about 0.5 '
         f'(0.5 plus noise of spread 0.2, kept between −1 and 2), but sometimes a <i>disaster</i> of −4. The '
         f'deployment\'s overall disaster chance, the <i>rung</i>, is one of 0, 0.02, 0.05, 0.08, 0.10, 0.12, 0.15 and '
         f'0.20; how it spreads over the conditions is the <i>profile</i>: everywhere alike, mostly in harsh conditions, '
         f'or mostly in mild ones.</p>'),
        ('A shift.', '<p>The recorded tries came mostly from mild conditions; deployment may meet harsh ones more often. '
         'A try\'s exact weight is its condition\'s share at deployment divided by its share among the recorded '
         'tries.</p>' + mix_tbl),
        ('288 worlds.', '<p>A <i>world</i> combines a profile (3), a rung (8), a shift (none, moderate or strong) and a '
         'number of held-out tries (30, 60, 120 or 240). A world is <i>unsafe</i> when more than 10% of outcomes at '
         'deployment fall below the fallback, and <i>harmful</i> when the average gain at deployment is below 0. A '
         'gate\'s <i>deviation rate</i> in a world is the share of seeds in which it deviates; in an unsafe or harmful '
         'world, that is how often it breaks its promise. The allowance on a 5% rate is 0.05 + 3 × √(0.05 × 0.95 ÷ '
         '800) = 0.073.</p>'),
    ]
    o.append('<ol class="toy-steps">' + "".join(f'\n      <li><b>{a}</b>{b}</li>' for a, b in steps) + '\n    </ol>')
    o.append(calc("Worked: which rungs are unsafe or harmful", [
        (None, "A usual gain falls below 0 with chance 0.0062 (0.5 is 2.5 spreads above 0). At deployment:"),
        ("rung 0.10, share below", "0.10 + 0.90 × 0.0062 = 0.1056, above 0.10: unsafe"),
        ("rung 0.10, average gain", "0.90 × 0.5 + 0.10 × (−4) = 0.45 − 0.40 = 0.05: still good"),
        ("rung 0.12, average gain", "0.88 × 0.5 + 0.12 × (−4) = 0.44 − 0.48 = −0.04: harmful"),
    ], so="rungs 0.10 and above are unsafe, rungs 0.12 and above harmful; at rung 0.10 a tail gate should refuse and "
          "a mean gate may accept."))
    o.append(calc("Worked: the weights under the strong shift", [
        ("<i>w</i><sub>0</sub> … <i>w</i><sub>4</sub>", f'{fr("5", "35")}, {fr("10", "25")}, {fr("15", "20")}, '
                                                        f'{fr("30", "12")}, {fr("40", "8")} = 0.143, 0.400, 0.750, 2.5, 5'),
        ("w̄", "0.05² ÷ 0.35 + 0.10² ÷ 0.25 + 0.15² ÷ 0.20 + 0.30² ÷ 0.12 + 0.40² ÷ 0.08"),
        ("", "0.0071 + 0.0400 + 0.1125 + 0.7500 + 2.0000 = 2.9096"),
        (None, "The average squared weight over the recording's mix is also w̄, so <i>n</i> recorded tries are "
               "worth about <i>n</i> ÷ w̄ equally weighted ones:"),
        ("effective tries, 120 recorded", "120 ÷ 2.9096 = 41.2"),
    ], so="under the strong shift each recorded try is worth about a third of an equally weighted one (the 800 seeds "
          "averaged 41.8 effective tries of 120), while rejection sampling, which discards tries until the rest follow "
          "deployment's mix, keeps only about 24."))
    o.append('<p>The gates: BCA\'s tail gate and mean gate, each unweighted (answering for the recording\'s mix) and '
             'weighted (BCA\'s weights, the extra weight at w̄; the weighted tail gate is the one built in '
             '<code>calibration/wbcp.py</code>, with 1,000 draws); weighted split conformal (Tibshirani and others, '
             '2019), whose promise holds only on average over data sets; two simple valid ways to handle a shift, both '
             'of which throw evidence away: <i>rejection sampling</i> (keep each try with chance weight ÷ largest '
             'weight, then run the exact test) and a <i>stratified</i> exact bound per condition; a weighted t-test; '
             'and plain and weighted averages. No proof covers the weighted mean gate.</p>')
    o.append(finding("toy-fourth-brief", "Experiment 4 in four parts",
                     "BCA's weights keep its tail and mean promises when the held-out tries come from a different mix "
                     "of conditions than deployment, and use the evidence better than the simple valid alternatives.",
                     "Fourteen gates, references included, on 288 worlds, 800 seeds; six claims with kill rules fixed before the run (table "
                     "below).",
                     "All six alive. Ignoring the shift, the tail gate deviated in up to 89.9% of seeds in an unsafe "
                     "world (17 worlds above the allowance), the mean gate in up to 93.6% in a harmful one. With weights: "
                     "at most 4.3% and 4.4%. In safe shifted worlds the weighted tail gate said yes in 33.6% of seeds, "
                     "against 19.0% for rejection sampling and 0% for the stratified bound. At equal harm the weighted "
                     "mean gate led narrowly: 0.0975 per decision against 0.0923.",
                     "Unweighted gates answer for the recording's mix, where the deviation looked safe. Weights make the "
                     "tries speak for deployment's mix while keeping every try, so they lose less evidence than "
                     "discarding tries. A shift still costs evidence: the effective number of tries shrinks."))
    claims = [
        ["1. Ignoring the shift breaks BCA's tail promise", "The unweighted tail gate stays within the allowance in every unsafe world", "up to 89.9%; 17 unsafe worlds above 7.3%", TAG_ALIVE],
        ["2. With weights, BCA's tail gate as built keeps its promise", "It is above the allowance in any unsafe world", "at most 4.3%", TAG_ALIVE],
        ["3. The weights use shifted evidence better than the simple valid alternatives", "In safe shifted worlds it says yes under 5 points more often than rejection sampling or the stratified bound", "33.6% against 19.0% and 0.0%", TAG_ALIVE],
        ["4. Weighted BCA's promise holds for the data in hand, unlike weighted split conformal's", "Split conformal's worst unsafe-world rate is under 5 points above weighted BCA's", "38.8% against 4.3%", TAG_ALIVE],
        ["5. With weights, BCA's mean gate keeps its promise", "It is above the allowance in any harmful world", "at most 4.4%", TAG_ALIVE],
        ["6. At equal harm, the weighted mean gate earns the most", "Another tuned gate earns at least as much on seeds 400–799", "0.0975 against 0.0923 (weighted split conformal)", TAG_NARROW],
    ]
    o.append(table("toy-fourth-claims", "Experiment 4's six claims, all 800 seeds (claim 6: tuned on seeds 0–399, scored "
                   "on seeds 400–799).", ["Claim", "Shelved if", "Result", "Verdict"], claims))
    res = [
        ["Never deviate", "0.000", "0.000", "0.000", "0.0000"],
        ["Always deviate", "1.000", "1.000", "1.000", "0.0950"],
        ["BCA tail gate, unweighted", "0.899", "0.550", "0.736", "0.1106"],
        ["<b>BCA tail gate, weighted</b>", "0.043", "0.336", "0.025", "0.0838"],
        ["Weighted split conformal", "0.388", "0.750", "0.266", "0.1423"],
        ["Rejection-sampled exact test", "0.036", "0.190", "0.021", "0.0644"],
        ["Stratified exact bound", "0.000", "0.000", "0.000", "0.0023"],
        ["BCA mean gate, unweighted", "0.986", "0.647", "0.936", "0.1253"],
        ["<b>BCA mean gate, weighted</b>", "0.119", "0.415", "0.044", "0.0982"],
        ["Rejection-sampled BCA mean gate", "0.119", "0.261", "0.026", "0.0789"],
        ["Weighted t-test", "0.440", "0.741", "0.381", "0.1395"],
        ["Weighted average", "0.730", "0.927", "0.549", "0.1536"],
        ["Plain average", "1.000", "0.888", "1.000", "0.1461"],
        ["Oracle (good on average at deployment)", "1.000", "1.000", "0.000", "0.1719"],
    ]
    o.append(table("toy-fourth-results", "Every gate at its stated setting, all 800 seeds: the highest deviation rate "
                   "over the 144 unsafe worlds (a tail gate keeping its promise stays near 0.05), the rate pooled over the "
                   "96 safe shifted worlds (higher uses more of the good evidence), the highest rate over the 108 harmful "
                   "worlds (a mean gate keeping its promise stays near 0.05), and the pooled gain per decision.",
                   ["Gate", "Worst unsafe", "Safe, shifted", "Worst harmful", "Gain"], res, num_cols=(1, 2, 3, 4),
                   note="The oracle answers the mean question, so it deviates at rung 0.10, unsafe but good; its worst "
                        "unsafe rate of 1 is expected. The mean gates are not judged on unsafe worlds."))
    o.append(chart_weights())
    rate_rows = []
    order = [("BCA tail, unweighted", "BCA tail gate, unweighted"), ("BCA tail, weighted", "BCA tail gate, weighted"),
             ("weighted split conformal", "Weighted split conformal"),
             ("rejection-sampled exact", "Rejection-sampled exact test"),
             ("BCA mean, unweighted", "BCA mean gate, unweighted"),
             ("BCA mean, weighted", "BCA mean gate, weighted"), ("weighted average", "Weighted average"),
             ("oracle", "Oracle")]
    for k, nm in order:
        rate_rows.append([nm] + [f"{v:.3f}" for v in h[k]])
    o.append(table("toy-fourth-rungs", "The charts' numbers, and two more gates': deviation rate by rung, harm in "
                   "harsh conditions, strong shift, 120 held-out tries, 800 seeds. Rungs 0.10 and above are unsafe, "
                   "0.12 and above harmful.",
                   ["Gate"] + [f"Rung {r:.2f}" for r in W_RUNGS], rate_rows, num_cols=tuple(range(1, 9)),
                   note="The notebook draws these rates as a chart without printing them. They were read for this page "
                        "by one cell added after the last cell of a copy of the notebook, which reads the run's "
                        "decisions and changes nothing; the copy's other outputs matched the notebook's."))
    o.append('<p><b>One decision, worked by hand.</b> Seed 1 of the world with harm in harsh conditions, rung 0.12 '
             '(unsafe and harmful), strong shift, 120 tries. Six tries fell below the fallback: 3 in condition 2 '
             '(weight 0.750) and 3 in condition 4 (weight 5).</p>')
    o.append(calc("Worked: the unweighted and the weighted tail gate on the same 120 tries", [
        ("unweighted: tries below", "6 of 120 = 5.0%; with 120 tries the exact test allows up to 6"),
        ("so", "6 ≤ 6: yes, deviate"),
        (None, "The tries by condition: 38, 33, 27, 14 and 8, with weights 0.143, 0.400, 0.750, 2.5 and 5:"),
        ("total weight", "38 × 0.143 + 33 × 0.400 + 27 × 0.750 + 14 × 2.5 + 8 × 5"),
        ("", "5.43 + 13.20 + 20.25 + 35.00 + 40.00 = 113.879"),
        ("weighted: weight below", "3 × 0.750 + 3 × 5 = 2.25 + 15 = 17.25"),
        ("weighted share at or above", "1 − 17.25 ÷ 113.879 = 0.849, short of the 0.90 needed"),
        ("draws that cross", "124 of 1,000, short of the 950 needed"),
        ("so", "no, fall back"),
    ], so="the six bad tries are 5% of the tries but carry 15.1% of the weight, because they come from conditions that "
          "deployment meets more often than the recording. Over all 800 seeds of this world the unweighted gate "
          "deviated in 42.6% and the weighted gate in 1.4%."))
    need = [
        ["None", "1.00", "1.00", "29", "29", "216"],
        ["Moderate", "0.73", "0.40", "39", "73", "287"],
        ["Strong", "0.34", "0.20", "83", "145", "382"],
    ]
    o.append(table("toy-fourth-evidence", "How many held-out tries each valid tail gate needs before it can say yes at "
                   "all, even when every try beats the fallback (about; from the notebook's closed forms).",
                   ["Shift", "Effective tries per try", "Tries kept by rejection", "Weighted BCA", "Rejection-sampled exact",
                    "Stratified exact"], need, num_cols=(1, 2, 3, 4, 5),
                   note="No world has more than 240 tries, so the stratified bound never says yes under a shift. "
                        "Unweighted BCA needs 29 in every case, but under a shift it answers for the wrong mix."))
    o.append('<p><b>Information only, added after the first run.</b> A version of the weighted mean gate that a July '
             '2026 proof appears to cover gives the extra worst-case weight the largest weight (2.5 under the moderate '
             'shift, 5 under the strong one) instead of the average one (1.37 and 2.91). At the stated 5% level it is '
             'more cautious: worst harmful-world rate 2.6% against 4.4%, yes in 27% of safe shifted cases against 42%, '
             'gain 0.080 against 0.098 per decision. At equal harm the gap nearly closes: 0.0954 against 0.0975, both '
             'above weighted split conformal\'s 0.0923. Held to the same safety level as claim 6, the unweighted tail '
             'gate earned 0.054; at its stated setting it earned 0.111 against 0.084 for the weighted one, but only by '
             'breaking the promise in 17 unsafe worlds. The weights here are known exactly; real weights must be '
             'estimated, and errors in them were not tested.</p>')
    return "\n    ".join(o)


# ====================================================================================== 8. together
def part_together():
    o = ['<h3 id="toy-together">What the toy worlds say together</h3>']
    o.append('<p class="status"><span class="tag acc">exploratory</span> <code>bca_findings.ipynb</code> gathers the four '
             'experiments in one notebook: each part\'s code is copied unchanged and reproduces every printed result '
             '(this page\'s rerun reproduced its 508 printed result lines; only 7 running times differ). Its lesson '
             'version on the branch <code>claude/bca-findings-walkthrough</code> (<code>2467e29</code>) adds a '
             'step-by-step lesson on BCA (Steps A.1 to A.10) and one decision worked by hand per part; its rerun here '
             'reproduced all 1,725 printed lines (only 7 running times differ), and every worked example checks itself '
             'against the experiments\' results.</p>')
    o.append('<p><b>One signal mattered: the risky action\'s own held-out outcomes, measured against the fallback\'s '
             'value.</b> A width built from errors at recorded actions never reached that signal; a gate that reads it '
             'did. With equal weights, each of BCA\'s gates is a known classical method: the exact binomial test for '
             'the tail promise, and Gaffke\'s bound for the mean promise. Under a shift, BCA\'s weights kept both '
             'promises and used the evidence better than the simple valid alternatives.</p>')
    rows = [
        ["1. Does the width make the policy choose better?", "none (demonstration)", "–", "–", "No: weighting fixed coverage, but no width improved a choice"],
        ["2. Where can BCA change a decision?", "6", "4", "2", "As a gate that reads the risky action's held-out outcomes, not as a width"],
        ["3. Can a BCA gate raise return?", "6", "4", "2", "Its mean gate earns the most at equal safety, not when safety is ignored"],
        ["4. Do BCA's weights help under a shift?", "6", "6", "0", "Yes: they keep both promises, at a cost in evidence"],
    ]
    o.append(table("toy-together-claims", "The four experiments: 18 pre-registered claims, 14 alive and 4 shelved.",
                   ["Experiment", "Claims", "Alive", "Shelved", "Answer"], rows, num_cols=(1, 2, 3)))
    o.append('<p><b>The four shelved claims say the most:</b></p>')
    o.append('<ul class="leads">\n'
             '      <li><b>BCA\'s width does not help as built</b> (experiment 2, test 1). It is measured where the '
             'recording has data, and the risky action has none; it also penalizes good actions that are merely '
             'noisy.</li>\n'
             '      <li><b>With equal weights, BCA adds no information over a textbook test</b> (test 4a). What it adds over '
             'split conformal is the kind of promise: one that holds for the data in hand.</li>\n'
             '      <li><b>Caution does not raise return when the held-out evidence is already honest</b> (experiment 3, '
             'claim 4): extra caution then mostly turns good candidates away.</li>\n'
             '      <li><b>A shared harm budget does not help when few actions pass at all</b> (experiment 3, claim '
             '6).</li>\n    </ul>')
    o.append('<p><b>What the notebook concludes for BCA</b> (toy evidence only):</p>')
    o.append('<ol>\n'
             '      <li>Use BCA as a gate on deviations, not as a width in training. The toys cannot judge a dose in '
             'training, because in them a critic\'s errors never spread through its own training targets; what they show '
             'is that neither way of using a width beat simply refusing thin data.</li>\n'
             '      <li>Measure a deviation against the fallback\'s value, not against other rows plus a margin: in '
             'experiment 2 this cut regret from 0.051 to 0.028 and drops the step-1 gate\'s hand-picked 0.10. A harm '
             'test still ventured in 27.6% of cases in unsafe worlds; a safety promise needs a gate that asks for proof '
             'of safety.</li>\n'
             '      <li>Match the gate to the promise: BCA\'s mean gate for the most return at a fixed safety level '
             '(0.0845 against 0.0400 in experiment 3, by a narrow margin in experiment 4, both with settings tuned on '
             'worlds whose truth is known); BCA\'s tail gate for "rarely worse than the fallback" (at most 3% of seeds '
             'in any unsafe world in experiment 2, against up to 38% for split conformal); the plain held-out average '
             'when no safety level is required.</li>\n'
             '      <li>Use weights when the held-out tries come from a different mix of states than deployment, and '
             'for the weighted mean gate prefer the version with the largest weight on the worst case.</li>\n'
             '      <li>Compute equal-weight BCA exactly: with 1,000 random draws only 4 of 8 score sets matched the '
             'exact answer at 28 and at 29 scores, and exact arithmetic lowers the evidence floor from the 36 rows that '
             'production asks for to 29.</li>\n'
             '      <li>Describe it plainly: the defensible claims are a safety promise for each decision that holds for '
             'the data in hand, and the most return at a fixed safety level; the claim that BCA\'s Bayesian posterior '
             'adds information over a textbook test is not supported.</li>\n'
             '      <li>Budget the evidence: the tail gate needs at least 29 held-out outcomes per decision with no '
             'shift, about 39 under the moderate shift and about 83 under the strong one.</li>\n    </ol>')
    o.append('<p><b>What the toys do not show.</b> Every gate was handed the deviation and read real, independent '
             'held-out tries of it; a real gate must find its deviations and group similar ones, and each split of the '
             'evidence costs tries. Floors and weights were known exactly: a wrong floor voids the mean gate\'s promise, '
             'and estimated weights are untested. Each decision stood alone. The mean gate\'s guarantee rests on an '
             'unreviewed 2026 preprint, and BCA\'s own weighted mean gate has no proof. None of the worlds lets a '
             'critic\'s errors spread through its own training targets, the main failure in offline learning. The '
             'notebook proposes three next steps, none of them run yet: a toy test with estimated weights (designed, with '
             'its kill rules to be fingerprinted before its first run); a ThrusterChain test of the mean gate at '
             'deviations, which would need a new pre-registration approved by the project owner; and exact '
             'equal-weight BCA in production, a small change to <code>calibration/wbcp.py</code> that needs the '
             'owner\'s approval.</p>')
    return "\n    ".join(o)


# ====================================================================================== 9. meaning for ThrusterChain
def part_meaning():
    o = ['<h3 id="toy-meaning">What it means for ThrusterChain</h3>']
    o.append('<p class="status"><span class="tag acc">exploratory</span> Toy evidence only: it can aim the next '
             'ThrusterChain tests, not replace them.</p>')
    o.append('<ul class="leads">\n'
             '      <li><b>It matches the record.</b> In a read-only diagnosis on seeds 0–4, BCA\'s own width was 0.31 to '
             '0.36 value units both at a harmful command past 80% and at the 80% cap, while the critic\'s false '
             'preference for that command was 0.63 to 0.89: as in experiment 2\'s test 1, the width did not react where '
             'the decision is made (see <a href="#ats-bca">After the stop</a>).</li>\n'
             '      <li><b>Reading outcomes worked there too.</b> The restructured BCA reads the critic\'s over-predictions '
             'on the 29 to 60 full-throttle steps per seed and world left after the removal, scored by critics that '
             'never trained on them. Its step 1 passed: caution on in 5 of 5 stalling seeds, off in 5 of 5 no-stall '
             'seeds, 106.1 points net across the two worlds against a bar of 90 (designed after seeing seeds 0–4; see '
             '<a href="#ats-restructure">Restructured BCA</a>). A later route search credited its fixed 0.10 margin, not '
             'the posterior: with 12 to 45 rows, BCA beat plain split conformal by only 1.78 points. Its step 2, which '
             'puts the same switch inside training, then met all three of its pass marks (GO, 15 Oct; exploratory); '
             'there too plain split conformal\'s switch made the same on/off decision as BCA\'s in all 10 runs, and '
             'nothing was credited to calibration or to BCA\'s Bayesian part.</li>\n'
             '      <li><b>Where the toy does not carry over.</b> ThrusterChain\'s comparison group was quiet (its 90% '
             'point stayed between −0.049 and −0.031), so the toy\'s step-1 weakness did not arise. ThrusterChain\'s '
             'scores use one-step targets, so the critic\'s value of the next state would stay in a harm test (the '
             'toy\'s last stage has no next state, so there the critic drops out). A tail certificate needs 29 rows; in '
             'step 1, no-stall seed 3 had 29, and BCA\'s random draws gave it no finite caution size (947 of BCA\'s 1,000 '
             'random draws cleared the bar; 950 were needed). And the stall is shortfall-shaped (past 80% every command '
             'pushes less, rather than failing rarely), the shape in which the toy\'s plain average served return '
             'best.</li>\n'
             '      <li><b>The evidence is thin for a shift.</b> Experiment 4\'s tail gate needs about 39 held-out '
             'outcomes per decision under its moderate shift and about 83 under its strong one, against 29 with no shift. '
             'If step 1\'s 29 to 60 full-throttle steps per seed and world had to be weighted for a shift like those, '
             'they would be enough for some seeds and worlds under the moderate shift and for none under the strong one. '
             'On ThrusterChain itself, the <a href="#weighting-test">weighting test</a> found for BCA\'s band what '
             'experiment 1 found for the width: weighting by the exact ratio restored 90% coverage under a known '
             'shift, and its exploratory runs found no gain in return from weighting as BCA uses its band.</li>\n'
             '      <li><b>No plan changes here.</b> The toys propose BCA as a gate that decides whether to venture at the '
             'gap, rather than as a width in training. Other learners that host BCA were not tested in them. Stage 2\'s '
             'version 7, which was to test the width on fresh seeds 5–9, has ended and will not run. Step 2 of the '
             'restructured BCA, its training pilot, was fixed in writing before any training and ran unchanged; any '
             'change to a pre-registered test needs the project owner\'s approval at their own terminal.</li>\n'
             '    </ul>')
    return "\n    ".join(o)


def build():
    parts = [STYLE, part_overview(), part_why(), part_world(), part_tools(), part_first(), part_second(), part_third(),
             part_fourth(), part_together(), part_meaning()]
    body = "\n\n    ".join(parts)
    return '<section id="toy-worlds">\n    ' + body + '\n  </section>'


if __name__ == "__main__":
    s = build()
    assert "\r" not in s
    open(os.path.join(HERE, "toy_fragment.html"), "w", encoding="utf-8", newline="\n").write(s)
    print("toy_fragment.html", len(s), "characters")

"""EDIT 1 of 4 (pass 3): the structure plan (notes/structure3.md) for the owner's comments A, E and F.

  A  the six specification sections (Action Space, Observation Space, Starting State, Transition Dynamics, Rewards,
     Episode End) move to right after the at-a-glance header, before "Watch one episode" and "The task"; the contents
     rail, the at-a-glance sentence that announces the order, the CSS comment and the walkthrough counter follow; eight
     terms that are now used before their definition get a short inline definition (wording from the page).
  F  Version History is deleted with its rail link; the Transition Dynamics link to it is reworded (link to the
     Summary's box); the facts that lived only there move to the Summary's box and to "Names used in the code".
  E  the "illustration, not a result" tag goes; the chart's caption becomes a full sentence that still says the error
     model is assumed; "illustration" also leaves the legend entry and the chart's accessible label.

Every replacement is exact and counted (in its block when one is named). The v11 backup's md5 is asserted unchanged.
  check (default; writes nothing):  python3 s1_structure.py
  trial copy (tmp/s1/s1_applied.html): python3 s1_structure.py --copy
  apply to the working page:         python3 s1_structure.py --write
"""
import hashlib
import os
import re
import sys

PAGE = "/mnt/c/Users/David/dev/bca-shift-work/page/thrusterchain.html"
V11 = "/mnt/c/Users/David/dev/bca-shift-work/page/thrusterchain.v11.html"
V11_MD5 = "79f00557ba8803625b9f36bc497f7cc6"
TMP = "/mnt/c/Users/David/dev/bca-shift-work/page/tmp/s1"


def md5(p):
    return hashlib.md5(open(p, "rb").read()).hexdigest()


assert md5(V11) == V11_MD5, "v11 backup changed"
s = open(PAGE, encoding="utf-8").read()
page_md5_before = hashlib.md5(s.encode("utf-8")).hexdigest()
n_mjx_before = s.count("<mjx-container")
TAGS = ("section", "div", "p", "ul", "li", "table", "tbody", "tr", "td", "nav", "figure", "header", "main", "dl", "i",
        "a", "code", "strong")
bal0 = {t: len(re.findall(r"<%s[\s>]" % t, s)) - s.count("</%s>" % t) for t in TAGS}
log = []


def once(old, new, label):
    global s
    n = s.count(old)
    assert n == 1, f"{label}: expected 1, found {n}: {old[:90]!r}"
    s = s.replace(old, new, 1)
    log.append(label)


BLOCK_RX = re.compile(r"\n(  <header class=\"head\" id=\"top\">|  <section id=\"([^\"]+)\">|    <figure class=\"panel\" id=\"sim\")")


def block_span(bid):
    m0, m1 = s.find("<main>"), s.find("</main>")
    st = [(m.start() + 1, m.group(2) or ("top" if "header" in m.group(1) else "sim")) for m in BLOCK_RX.finditer(s, m0, m1)]
    for i, (a, b) in enumerate(st):
        if b == bid:
            return a, (st[i + 1][0] if i + 1 < len(st) else m1)
    raise KeyError(bid)


def once_in(bid, old, new, label):
    """Replace old (exactly once inside block bid) with new."""
    global s
    a, e = block_span(bid)
    blk = s[a:e]
    n = blk.count(old)
    assert n == 1, f"{label}: expected 1 in #{bid}, found {n}: {old[:90]!r}"
    s = s[:a] + blk.replace(old, new, 1) + s[e:]
    log.append(label)


def main_order():
    m0, m1 = s.find("<main>"), s.find("</main>")
    return [m.group(2) or ("top" if "header" in m.group(1) else "sim") for m in BLOCK_RX.finditer(s, m0, m1)]


# ------------------------------------------------------------------ A: the block move
n0 = len(s)
a = s.index('\n    <figure class="panel" id="sim"') + 1
b = s.index('\n  <section id="action-space">') + 1
c = s.index('\n  <section id="description">') + 1
assert s.count('\n    <figure class="panel" id="sim"') == s.count('\n  <section id="action-space">') == \
    s.count('\n  <section id="description">') == 1 and a < b < c
assert s[:a].endswith("  </header>\n\n")
simtask, six = s[a:b], s[b:c]
assert simtask.endswith("  </section>\n\n") and six.endswith("  </section>\n\n")
assert six.count('<section id="') == 6 and simtask.count('<section id="') == 1
h_six, h_simtask = hashlib.md5(six.encode()).hexdigest(), hashlib.md5(simtask.encode()).hexdigest()
s = s[:a] + six + simtask + s[c:]
assert len(s) == n0
a2 = s.index('\n  <section id="action-space">') + 1
assert hashlib.md5(s[a2:a2 + len(six)].encode()).hexdigest() == h_six
assert hashlib.md5(s[a2 + len(six):a2 + len(six) + len(simtask)].encode()).hexdigest() == h_simtask
log.append("A0 move: the six sections now precede the live panel and The task")

# ------------------------------------------------------------------ A: contents rail (and F: its Version History link)
once('  <a href="#glance">At a glance</a>\n  <a href="#sim">Watch one episode</a>\n  <a href="#task">The task</a>\n',
     '  <a href="#glance">At a glance</a>\n', "A1 rail: live panel and The task leave the top")
once('  <a href="#episode-end">Episode End</a>\n',
     '  <a href="#episode-end">Episode End</a>\n  <a href="#sim">Watch one episode</a>\n  <a href="#task">The task</a>\n',
     "A2 rail: live panel and The task after Episode End")
once('  <a href="#version-history">Version History</a>\n', '', "F1 rail: Version History link removed")

# ------------------------------------------------------------------ A: the at-a-glance sentence that announces the order
once('The live panel "<a href="#sim">Watch one episode</a>" comes next, and right after it the section "<a href="#task">The task</a>" explains the whole task.',
     'The next six sections specify the environment: what a controller sets (<a href="#action-space">Action Space</a>), '
     'what it reads (<a href="#observation-space">Observation Space</a>), how an episode starts '
     '(<a href="#starting-state">Starting State</a>), how one step moves the floats '
     '(<a href="#dynamics">Transition Dynamics</a>), what each step costs (<a href="#rewards">Rewards</a>) and when an '
     'episode ends (<a href="#episode-end">Episode End</a>). After them, the live panel "<a href="#sim">Watch one '
     'episode</a>" replays real episodes state by state, and the section "<a href="#task">The task</a>" explains what '
     'makes the task hard.',
     "A3 at a glance: the sentence that announces the reading order")

# ------------------------------------------------------------------ A: the CSS comment that describes the reading order
once("/* Layout: one environment's reference page. Reading order: the summary and the live panel, the task, then the\n"
     "   sections of a Gymnasium environment page (Action Space first), the methods and the results. A reading column with a\n",
     "/* Layout: one environment's reference page. Reading order: the at-a-glance summary, the six specification sections\n"
     "   (Action Space first), the live panel, the task, the summary of the environment, the methods and the results. A reading column with a\n",
     "A4 CSS comment: reading order")

# ------------------------------------------------------------------ A: the walkthrough's static counter (overwritten on open)
once('<p class="wt-count" id="wt-count">Section 1 of 22</p>', '<p class="wt-count" id="wt-count">Section 1 of 21</p>',
     "A5 walkthrough: static counter 21 steps")

# ------------------------------------------------------------------ A: terms now used before their definition
# 1 Action Space: the two worlds (source: Summary, "Worlds (2)"; stalling-world and no-stall tooltips)
once_in("action-space",
        "past 80% the propeller stalls (below). In the table,",
        "past 80% the propeller stalls (below). The environment has two <i>worlds</i> that differ only here: in the "
        "<i>stalling world</i>, the default and the world the data were recorded in, the propeller stalls; in the "
        "<i>no-stall comparison world</i>, push keeps rising up to 100% throttle, which shows what the same learning "
        "costs when the throttles missing from the data hide nothing. In the table,",
        "G1 Action Space: the stalling world and the no-stall comparison world")
# 2 Observation Space: amplitude and phase (source: Starting State's amplitude and phase items, the phi tooltip)
once_in("observation-space",
        "mark the wave's place in its 23.27-step cycle, like the hands of a\n        clock.</li>",
        "mark the wave's place in its 23.27-step cycle, like the hands of a\n        clock. The wave's "
        "<i>amplitude</i>, the largest push it adds to the current or takes away, and its <i>phase</i>, where it is "
        "in its cycle, are drawn at the start of each episode (<a href=\"#starting-state\">Starting State</a>).</li>",
        "G2 Observation Space: amplitude and phase")
# 3 Starting State: situation and the names of the four (source: Summary, "Situations (4)"; Starting State's own list)
once_in("starting-state",
        "Each reset draws them at random,\n      by the rule of the chosen situation. A start sets four things,",
        "Each reset draws them at random,\n      by the rule of the chosen <i>situation</i>: either the recorded "
        "conditions, the rule the training data come from, or one of three shifted situations (strong current, big "
        "waves and big starting pushes), each of which changes one setting and is described after the list below. "
        "A start sets four things,",
        "G3 Starting State: situation and the names of the four situations")
# 4 Starting State: float cost (source: Rewards)
once_in("starting-state",
        "a starting state's float cost averages 2.5 in the recorded conditions",
        "a starting state's <i>float cost</i>, the floats' squared distance from rest that every step is charged "
        "(<a href=\"#rewards\">Rewards</a>), averages 2.5 in the recorded conditions",
        "G4 Starting State: float cost")
# 5 Starting State: the textbook command (source: the textbook-rule and stall-aware tooltips)
once_in("starting-state",
        "The textbook hold\n        against the current alone, with the floats at rest, asks thruster 1 for 80% once the current exceeds 0.879, so\n"
        "        in strong current holding often asks for 80% or more, where the stall is.</li>",
        "The <i>textbook\n        command</i> makes each throttle a fixed weighted sum of the 13 observed numbers, the best such rule if "
        "push always equalled throttle. Holding against the current alone, with the floats at rest, it asks thruster 1 "
        "for 80% once the current exceeds 0.879, so\n        in strong current holding often asks for 80% or more, where "
        "the stall is.</li>",
        "G5 Starting State: the textbook command")
# 6 + 7 Starting State: evaluation bank, the water's push, the full-throttle region
#   (sources: the evaluation-bank tooltip and the end of Starting State; the water's-push and full-throttle-region tooltips)
once_in("starting-state",
        "where each start of the evaluation bank has an amplitude 0.3 larger\n"
        "        than the same start in the recorded conditions' bank. This matters because a bigger wave swings the water's\n"
        "        push further every cycle: further into the full-throttle region near each crest, and further upstream near\n"
        "        each trough.</li>",
        "where each start of the <i>evaluation bank</i> (the fixed set of 1,000 starts per situation over which every "
        "reported return is averaged, described at the end of this section) has an amplitude 0.3 larger\n"
        "        than the same start in the recorded conditions' bank. This matters because a bigger wave swings the "
        "<i>water's push</i>, the current plus the push the wave adds now, the same on every float, further every "
        "cycle. Near each crest it reaches further into the <i>full-throttle region</i>, the states where the water's "
        "push is above 0.565, where the recording controller flips a coin in each state and on half of them gives "
        "the full-throttle command; near\n        each trough it reaches further upstream.</li>",
        "G6+G7 Starting State: evaluation bank, the water's push, the full-throttle region")
# 8 Rewards: the rising-thrust controller (source: its tooltip); "knows to stop" replaced by the header's wording
once_in("rewards",
        "the stall-aware controller's (in the no-stall comparison world, the rising-thrust controller's).\n"
        "      So 0 points is the level of the recording controller and 100 the level of a controller that knows to stop at 80%.",
        "the stall-aware controller's (in the no-stall comparison world, the <i>rising-thrust controller</i>'s: it "
        "gives the same textbook command but allows up to 100% throttle, as if push kept rising).\n"
        "      So 0 points is the level of the recording controller and 100 the level of a controller that never pushes "
        "past 80%.",
        "G8 Rewards: the rising-thrust controller; 'knows to stop' becomes 'never pushes past 80%'")

# ------------------------------------------------------------------ F: the Transition Dynamics link, then the section
once_in("dynamics",
        "These are the same\n        float dynamics as in the earlier world this one replaces "
        "(<a href=\"#version-history\">Version History</a>).</li>",
        "These float dynamics,\n        coupling included, are reused unchanged from the project's earlier test world, "
        "which had no stall (the box at the end of the section \"<a href=\"#description\">Summary of the environment</a>\" "
        "says why that world was replaced).</li>",
        "F2 Transition Dynamics: provenance sentence links to the Summary's box")
v0 = s.index('\n  <section id="version-history">') + 1
v1 = s.index('\n  <section id="code-names">') + 1
assert s.count('\n  <section id="version-history">') == 1 and v0 < v1
blk = s[v0:v1]
assert blk.count('<section id="') == 1 and blk.endswith("  </section>\n\n") and "<h2>Version History</h2>" in blk
s = s[:v0] + s[v1:]
log.append(f"F3 Version History section deleted ({len(blk)} characters)")

# F: the facts that lived only in Version History -> the Summary's box
once('(<a href="#task-strong">Why strong current is the hard case</a>).</p>\n    </div>',
     '(<a href="#task-strong">Why strong current is the hard case</a>).</p>\n'
     '      <p><strong>Why this environment, and not the earlier one.</strong> ThrusterChain replaces the project\'s\n'
     '        earlier test world, which had the same five floats and the same float dynamics but no stall. Its dynamics\n'
     '        were linear, meaning the next state is a fixed weighted sum of the state and the command, and its cost was\n'
     '        quadratic, meaning built from squares and products of the numbers, like the float cost. Its harm checks\n'
     '        rejected it twice: first on seeds 0–4, and again on fresh seeds 5–9 under a pre-registered amendment\n'
     '        whose pass marks were loosened after seeing seeds 0–4. Both times, the\n'
     '        return the learned actors lost did not come mainly from where the data were thin, and the second time\n'
     '        neither did the critics\' error, so there was no harm caused by the critic for BCA to find. BCA was never\n'
     '        run in it. The design work found two reasons to replace it. First, in a world with linear dynamics and a\n'
     '        quadratic cost every true value is itself a quadratic formula of the state and the command, so data from\n'
     '        any one region of states and commands pin down the true values everywhere, and missing data hide nothing.\n'
     '        Here the stall bends the true value at 80%, and the removed command is the only record of throttle past 80%\n'
     '        in the recorded data. Second, the removed command must sit at an edge of what is possible, not in the\n'
     '        middle, so that caution toward the recorded commands cannot pull the controller into the commands that\n'
     '        were removed. Full throttle, the end of the throttle range, is such an edge.</p>\n'
     '    </div>',
     "F4 Summary box: why this environment replaced the earlier one (moved from Version History)")

# F: package names that lived only in Version History -> Names used in the code
once('</tbody></table></div>\n  </section>\n</main>',
     '<tr><td>the floats\' own motion A, coupling included, reused unchanged from the project\'s earlier test world '
     '(linear dynamics, quadratic cost, no stall)</td><td><code>experiments/<wbr>signal/<wbr>gapped_lq.<wbr>'
     'transition_matrix(STAGE0_SPEC)</code>; the project\'s records call that world "Gapped LQ" (LQ for '
     'linear-quadratic), and the <code>glq</code> in <code>glq_match.<wbr>py</code> is the same name</td></tr>\n'
     '<tr><td>the weighting test</td><td><code>experiments/<wbr>shift_iw</code></td></tr>\n'
     '</tbody></table></div>\n  </section>\n</main>',
     "F5 Names used in the code: the earlier world's dynamics and the weighting test's package")

# ------------------------------------------------------------------ E: the illustration tag
once_in("weighting-test",
        '    <div class="panel">\n      <p><span class="tag warn">illustration, not a result</span></p>\n      <div class="iw-chart" id="iw-c-cov">',
        '    <div class="panel">\n      <div class="iw-chart" id="iw-c-cov">',
        "E1 weighting test: tag removed")
once_in("weighting-test",
        '<p class="note">Assumed error size 0.05 + 0.25·<i>c</i>, normally distributed; exact weights and an unlimited calibration set</p>',
        '<p class="note">This chart is computed, not measured: it assumes that the critic\'s error is normally '
        'distributed with a typical size of 0.05 + 0.25·<i>c</i>, where <i>c</i> is the current, and that the '
        'weights are exact and the calibration set unlimited.</p>',
        "E2 weighting test: caption is a full sentence and still says the error model is assumed")
once_in("weighting-test",
        "<dd>An assumption made for this illustration only: the typical error grows with the current.",
        "<dd>An assumption made for this chart only: the typical error grows with the current.",
        "E3 weighting test: legend entry says 'this chart'")
once_in("weighting-test",
        'aria-label="Illustrative coverage of unweighted and weighted bands against the test lean λ"',
        'aria-label="Coverage of unweighted and weighted bands against the test lean λ, computed from an assumed error size"',
        "E4 weighting test: accessible label")

# ------------------------------------------------------------------ checks
EXPECT = ["top", "action-space", "observation-space", "starting-state", "dynamics", "rewards", "episode-end", "sim",
          "task", "description", "recording-controller", "reference-scores", "dataset", "shift", "methods",
          "harm-checks", "bca-test", "results", "weighting-test", "arguments", "code-names"]
order = main_order()
assert order == EXPECT, order
nav = s[s.index('<nav class="toc"'):s.index("</nav>")]
rail = re.findall(r'href="#([^"]+)"', nav)
assert rail == ["glance"] + EXPECT[1:EXPECT.index("results") + 1] + ["beyond-the-data"] + EXPECT[EXPECT.index("results") + 1:], rail
for bad in ("version-history", "Version History", "illustration, not a result", "Illustrative coverage",
            "this illustration only", "knows to stop"):
    assert bad not in s, bad
assert s.count("<mjx-container") == n_mjx_before, "formula count changed"
# every in-page link resolves
ids = set(re.findall(r'\sid="([^"]+)"', s))
hrefs = set(re.findall(r'href="#([^"]+)"', s))
missing = sorted(h for h in hrefs if h not in ids)
assert not missing, missing
# tag balance for the elements touched: open minus close is the same as before the edits
for t in TAGS:
    o, cl = len(re.findall(r"<%s[\s>]" % t, s)), s.count("</%s>" % t)
    assert o - cl == bal0[t], (t, o, cl, bal0[t])

print("\n".join(log))
print(f"main order ({len(order)} blocks): {order}")
print(f"rail ({len(rail)} links): {rail}")
print(f"length {n0} -> {len(s)}; formulas {s.count('<mjx-container')}; internal links all resolve ({len(hrefs)})")

mode = sys.argv[1] if len(sys.argv) > 1 else "--check"
if mode == "--copy":
    os.makedirs(TMP, exist_ok=True)
    out = os.path.join(TMP, "s1_applied.html")
    open(out, "w", encoding="utf-8").write(s)
    print("wrote", out)
elif mode == "--write":
    cur = hashlib.md5(open(PAGE, "rb").read()).hexdigest()
    assert cur == page_md5_before, "page changed while this script ran"
    open(PAGE, "w", encoding="utf-8").write(s)
    print("wrote", PAGE, hashlib.md5(open(PAGE, "rb").read()).hexdigest())
else:
    print("check only; nothing written")
assert md5(V11) == V11_MD5, "v11 backup changed"

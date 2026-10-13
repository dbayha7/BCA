#!/usr/bin/env python3
"""Page pass 4, edit 5 (final): the final review's verified findings, on top of edits 1 to 4.

Every change is an exact, counted replacement (rep) or a splice of a block found by unique markers. The critic
section's data block and script are untouched (checked byte for byte); its section markup gets step lines under its
three control groups, colour marks on its two situation buttons and complete-sentence legend definitions.

Run from the page directory (WSL):
    python3 scripts/p4_e5_apply.py --check    (build in memory, write tmp/p4/edit5/applied.html only)
    python3 scripts/p4_e5_apply.py            (apply to thrusterchain.html)
"""
import hashlib
import html
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
PAGE_DIR = os.path.dirname(HERE)
PAGE = os.path.join(PAGE_DIR, "thrusterchain.html")
BEFORE = os.path.join(PAGE_DIR, "tmp", "p4", "edit5.before.html")
TRIAL = os.path.join(PAGE_DIR, "tmp", "p4", "edit5", "applied.html")
CSS = os.path.join(HERE, "p4_e5.css")
E4_CSS = os.path.join(HERE, "p4_e4.css")
E4_MD5 = "e24913a0187b041b7a279b10ed4b8a65"
CHECK = "--check" in sys.argv

s = open(PAGE, encoding="utf-8").read()
ORIG = s
assert hashlib.md5(open(BEFORE, "rb").read()).hexdigest() == E4_MD5, "backup changed"
assert hashlib.md5(s.encode("utf-8")).hexdigest() == E4_MD5, "the page is not the edit-4 page this edit was written for"
QV_LOCKED = re.compile(r'<!-- qviewer:begin (data|script) -->.*?<!-- qviewer:end \1 -->|/\* qviewer:begin style \*/.*?/\* qviewer:end style \*/', re.S)
QV_BEFORE = [m.group(0) for m in QV_LOCKED.finditer(s)]
assert len(QV_BEFORE) == 3, len(QV_BEFORE)
N = {"n": 0}


def region(name):
    if name == "all":
        return 0, len(s)
    if name == "glance":
        a = s.index('<div class="head-body" id="glance"')
        return a, s.index("</header>", a)
    if name == "sim":
        a = s.index('<figure class="panel" id="sim"')
        return a, s.index("\n    </figure>\n", a)
    if name == "script":
        a = s.index("</main>")
        return a, s.index("<!-- qviewer:begin data -->")
    if name == "style":
        return 0, s.index('<div class="wrap"')
    assert s.count(f'<section id="{name}"') == 1, name
    a = s.index(f'<section id="{name}"')
    return a, s.index("</section>", a) + len("</section>")


def rep(where, old, new, count=1):
    """exact replacement inside region `where`, asserted count"""
    global s
    lo, hi = region(where)
    part = s[lo:hi]
    n = part.count(old)
    assert n == count, f"[{where}] expected {count} of {old[:110]!r}, found {n}"
    s = s[:lo] + part.replace(old, new) + s[hi:]
    N["n"] += 1


def resub(where, pattern, new, count):
    global s
    lo, hi = region(where)
    part = s[lo:hi]
    out, n = re.subn(pattern, new, part)
    if n != count:
        for mm in re.finditer(pattern, part):
            print("   ", lo + mm.start(), repr(part[max(0, mm.start() - 90):mm.end() + 40]))
    assert n == count, f"[{where}] expected {count} of /{pattern}/, found {n}"
    s = s[:lo] + out + s[hi:]
    N["n"] += 1


def grab(where, start, end, include_end=True):
    lo, hi = region(where)
    part = s[lo:hi]
    assert part.count(start) == 1, (start[:80], part.count(start))
    a = lo + part.index(start)
    b = s.index(end, a)
    assert b < hi, end[:80]
    return a, b + (len(end) if include_end else 0)


def splice(a, b, new):
    global s
    s = s[:a] + new + s[b:]
    N["n"] += 1


def tip_el(title, tag="i", needle=None):
    """an existing tooltip element of the page with this title (first one, or the first whose text has needle)"""
    t = html.escape(title, quote=True)
    for m in re.finditer(r'<(i|span) (?:class="[^"]*" )?data-tip-title="%s" data-tip="([^"]*)">' % re.escape(t), ORIG):
        if needle is None or needle in html.unescape(m.group(2)):
            return f'<{tag} data-tip-title="{t}" data-tip="{m.group(2)}">'
    raise AssertionError(title)


TD3BC_LONG = "twin delayed deep deterministic policy gradient plus behaviour cloning"

# ===================================================================================================== 1. the top
# 1a. step 2: say what the densest 15% is, and what the 80% hold is, in the visible text
rep("glance", """densest-15% removal</i> deletes the recording controller's full-throttle
            commands, 99.11% of them.</p>""",
    """densest-15% removal</i> deletes the 15% of recorded commands that sit closest together, which here is
            the recording controller's full-throttle cluster (99.11% of those commands).</p>""")
rep("glance", """>80% hold</i>.</p>""",
    """>80% hold</i>: thruster 1 held at exactly 80%, the command that test's recording controller gives in hard
            water.</p>""")
# 1b. step 3: expand TD3+BC where it first appears
rep("glance", """>TD3+BC</span> trains a
            <i data-tip-title="Critic\"""",
    f""">TD3+BC</span> ({TD3BC_LONG}, a standard learner for recorded data) trains a
            <i data-tip-title="Critic\"""")
# 1c. step 4: what the band surrounds, and what the one-step error is
rep("glance", """<p class="pipe-d">BCA puts a <i data-tip-title="BCA&#x27;s band\"""",
    """<p class="pipe-d">BCA builds a <i data-tip-title="BCA&#x27;s band\"""")
rep("glance", """>band</i> around the critic's <i data-tip-title="One-step error\"""",
    """>band</i>, a range around the critic's estimate, sized on recorded rows kept out of training so that it
            contains the critic's <i data-tip-title="One-step error\"""")
rep("glance", """>one-step error</i>. Where the
            band is wide it adds caution:""",
    """>one-step error</i> (how far its estimate for a recorded row misses that row's reward plus 0.98 times its
            own value of the next state) 90% of the time. Where the band is wide it adds caution:""")
# 1d. the status: one line each, with links onward (the hash definition and the Stage 1 detail live in Results)
a, b = grab("glance", '<ul class="g-status" aria-label="Where the work stands">', "</ul>")
old = s[a:b]
POINTS = re.search(r'<i data-tip-title="Points" data-tip="[^"]*">', old).group(0)
HARM = re.search(r'<i data-tip-title="Harm checks" data-tip="[^"]*">', old).group(0)
IW = re.search(r'<i data-tip-title="Importance weighting" data-tip="[^"]*">', old).group(0)
COV = re.search(r'<i data-tip-title="Coverage" data-tip="[^"]*">', old).group(0)
RET = re.search(r'<i data-tip-title="Return" data-tip="[^"]*">', old).group(0)
SA = tip_el("Stall-aware controller", needle="textbook command")
splice(a, b, f"""<ul class="g-status" aria-label="Where the work stands">
      <li><span class="tag">Main test: stopped before BCA</span> Hiding the stall cost a learned controller 98.9
        {POINTS}points</i> on average in <span class="km" data-sit="strong_current">strong current</span> (95 to 103 over
        five repeat runs; 0 points is the recording controller, and 100 the {SA}stall-aware controller</i>, a fixed
        controller that follows the standard best linear rule but never goes past 80%). A harder pull toward the
        recorded commands, even one placed by the critic's exact error, could not win it back, so the
        {HARM}harm checks</i>, fixed before any learner ran, stopped the test before BCA ran
        (<a href="#results">Results</a>).</li>
      <li><span class="tag ok">Weighting test: pre-registered</span> On a second recording, made at five levels of
        quality, BCA's band worked as designed, and {IW}importance weighting</i> of its calibration rows toward the
        known test mix of currents restored its {COV}coverage</i> from 83.3–86.7% to 91.0–94.6%
        (<a href="#weighting-test">The weighting test</a>).</li>
      <li><span class="tag">Weighting test: exploratory</span> Weighting did not raise the {RET}return</i>, because the
        caution BCA's band gives at the actor's own commands is far too small: in BCA's exploratory training runs,
        weighting changed the return by −26.4 to +2.7 units of return (−2.07 to +0.21 points), with no gain both as
        large as the 1.8 units fixed in advance and clear of the spread between repeat runs
        (<a href="#iw-explore">Exploratory results</a>).</li>
    </ul>""")

# ===================================================================================================== 2. the live panel
rep("sim", " (the page also calls it the gapped critic)", "")
rep("sim", """"means over seeds 0–4" averages five repeat runs.</p>""",
    """"means over seeds 0–4" averages five repeat runs. Methods explains these; in short, the critic weight α
            sets how much the critic counts against the pull toward the recorded commands, and a dose multiplies that
            pull at one row.</p>""")
# the rising-thrust controller shows a direction, not what the learned actor does
rep("sim", """Step 3 gone wrong: it acts as an actor would whose critic never saw the stall.</span>""",
    """The direction step 3 can go wrong: an actor whose critic never saw the stall drifts this way, though only
              part of the way.</span>""")
rep("sim", """as if push kept rising with throttle, which is how a learned actor would act if it were trained against a critic that never saw the stall.</p>""",
    """as if push kept rising with throttle. It shows the direction a learned actor drifts when its critic never saw the stall; the learned actor goes only part of the way, past 80% on 73% of the strong-current states it visits, against 84.0% for this controller.</p>""")
rep("sim", """<b>In the process:</b> It shows step 3 gone wrong: an actor whose critic never saw the stall acts like this.</p>""",
    """<b>In the process:</b> It shows the direction step 3 can go wrong: an actor whose critic never saw the stall drifts this way, though only part of the way.</p>""")
rep("sim", """as if push kept rising with throttle. A learned actor trained against a critic that never saw the stall would act like this.</p>""",
    """as if push kept rising with throttle. A learned actor trained against a critic that never saw the stall drifts this way, though only part of the way: past 80% on 73% of the strong-current states it visits, against 84.0% for this controller.</p>""")
# the textbook rule K, defined where the panel first uses it
rep("sim", """swinging a quarter as hard as the textbook rule K and adding random noise""",
    """swinging a quarter as hard as the textbook rule K (the standard best linear rule, explained in The task) and adding random noise""")
rep("sim", """damps the floats&#x27; swinging a quarter as hard as the textbook rule K, and adds random noise.""",
    """damps the floats&#x27; swinging a quarter as hard as the textbook rule K (the standard best linear rule, explained in The task), and adds random noise.""")
# the drawing: un-stalled thrusters and gauge bars in ink and grey, so blue keeps meaning "learned"
rep("script", "const col = stalled ? warn : acc, len = f[k] * 34;", "const col = stalled ? warn : ink2, len = f[k] * 34;")
rep("script", """    g.fillStyle = stalled ? warn : acc;
    g.fillRect(gx - 5, Math.min(gc, GY(aa)), 10, Math.abs(GY(aa) - gc));""",
    """    g.fillStyle = stalled ? warn : muted;
    g.fillRect(gx - 5, Math.min(gc, GY(aa)), 10, Math.abs(GY(aa) - gc));""")
rep("sim", """<dd>It shows the push the thruster really gives, <i>f</i>(<i>a</i>); pointing up is upstream, against the water.</dd>""",
    """<dd>It shows the push the thruster really gives, <i>f</i>(<i>a</i>); pointing up is upstream, against the water. The thruster and its arrow are drawn in dark ink, and in red once the thruster stalls.</dd>""")
rep("sim", """<dd>It shows the thruster's throttle <i>a</i>, from −1 (full throttle against the water, at the top) to 1.</dd>""",
    """<dd>It shows the thruster's throttle <i>a</i>, from −1 (full throttle against the water, at the top) to 1. The bar is grey, and red once the thruster stalls.</dd>""")

# ===================================================================================================== 3. Action Space
rep("action-space", """<li><b>TD3+BC's pull toward the recorded commands</b><span class="stepchip in" data-step="train">Step 3</span> This pull, called behaviour cloning, acts at
        every kept training row:""",
    f"""<li><b>TD3+BC's pull toward the recorded commands</b><span class="stepchip in" data-step="train">Step 3</span> TD3+BC
        ({TD3BC_LONG}) is the standard learner for recorded data that this page uses. Its pull, called behaviour
        cloning, acts at every kept training row:""")
rep("action-space", """Past 80% that test's data hold only uniform random commands: none in the expert data, and in the other
four dataset qualities (the weighting test records five, from random commands up to an expert controller) about 3% to 10%
of training rows push thruster 1 upstream past 80%, and about as many downstream.</td>""",
    """Past 80% that test's data hold only uniform random commands: none in the expert data, and in the other
four dataset qualities (the weighting test records five, from random commands up to an expert controller) 3.25% to 10.11%
of all training steps push thruster 1 upstream past 80% (data seeds 0 to 4), and about as many downstream.</td>""")
rep("action-space", "a red band marks the throttles past 80%, where a propeller stalls.", "a red strip marks the throttles past 80%, where a propeller stalls.")
rep("action-space", """<th scope="col" data-label="Throttle band">Throttle band</th>""", """<th scope="col" data-label="Throttle range">Throttle range</th>""")
rep("action-space", """Rows of the main test's training data (design seed 900) with each thruster's throttle in each band:""",
    """Rows of the main test's training data (design seed 900, the recording this page shows) with each thruster's throttle in each 20%-wide range:""")
rep("script", """? "Rows of the main test's training data (design seed 900) with each thruster's throttle in each band:""",
    """? "Rows of the main test's training data (design seed 900, the recording this page shows) with each thruster's throttle in each 20%-wide range:""")
rep("script", """: "Rows of the weighting test's expert training data (data seed 0) with each thruster's throttle in each band:""",
    """: "Rows of the weighting test's expert training data (data seed 0) with each thruster's throttle in each 20%-wide range:""")
rep("script", '"up to " + int(fmax) + " rows in one band"', '"up to " + int(fmax) + " rows in one 4% bin"')
rep("script", '". The table below lists every band.";', '". The table below gives the rows in each 20%-wide range.";')
rep("script", """only uniform random commands here, about 3% to 10% of their training rows, so the critic guesses past 80% with or " +
          "without the hold removed. In the weighting test's exploratory runs BCA's lower edge, which reads the band at the " +
          "actor's own command, was far too small here: its penalty was about 1/30 to 1/160 of one built from the critic's " +
          "exact one-step error.");""",
    """only uniform random commands here, 3.25% to 10.11% of their training steps (data seeds 0 to 4), so the critic " +
          "guesses past 80% with or without the hold removed. BCA's lower edge reads the band at the actor's own command; " +
          "measured before training, at a no-caution actor's commands on the training rows, its penalty was about 1/30 to " +
          "1/160 of one built from the critic's exact one-step error.");""")

# ===================================================================================================== 4. Observation Space
rep("observation-space", """under a steady downstream push a float settles with a negative position number and a positive speed number, even
      though it no longer moves.""",
    """under a steady downstream push a float settles with a negative position number and a positive speed number, even
      though it no longer moves, because the two numbers are the coordinates of a turning swing, not a position and a
      velocity, and a steady push moves the centre of that swing in both coordinates.""")
for old, new in [("<dd>The 10 × 10 table of the floats' own motion", "<dd>It is the 10 × 10 table of the floats' own motion"),
                 ("<dd>How many steps ago a push was given.", "<dd>It is how many steps ago a push was given."),
                 ("<dd>The largest stretch a matrix can apply", "<dd>It is the largest stretch that a matrix can apply"),
                 ("<dd>Those stretches added over every", "<dd>It is the sum of those stretches over every"),
                 ("<dd>The largest push one step can add.", "<dd>It is the largest push that one step can add."),
                 ("<dd>The farthest an allowed start can put the floats from rest.", "<dd>It is the farthest from rest that an allowed start can put the floats.")]:
    rep("observation-space", old, new)

# ===================================================================================================== 5. Starting State
rep("starting-state", """measured as the points that an actor trained against the critic that
      never saw the stall loses against an actor trained against the critic of the full recording (the mean over five
      repeat runs;""",
    """measured as the points that an actor trained against the critic trained after the removal (which never saw the
      stall) loses against an actor trained against the full-recording critic (trained on the whole recording, nothing
      removed; the mean over five repeat runs;""")
rep("starting-state", """Big starting pushes are hard for every controller, but the
      stall-aware controller is at its 80% limit on only 15.5% of their states. Strong current puts it there in most
      states, 71.9% of them, so that is where holding needs exactly the commands the data lost.</p>""",
    """Big starting pushes are hard for every controller, but the
      stall-aware controller (the fixed 100-point controller, which never goes past 80%) has some thruster at its 80%
      limit on only 15.5% of their states. Strong current puts it there in most states, 73.2% of them (thruster 1 alone
      on 71.9%), so that is where holding needs exactly the commands the data lost.</p>""")

# ===================================================================================================== 6. Transition Dynamics, Rewards, Episode End, Reference Scores
rep("dynamics", """    <h2>Transition Dynamics</h2>
""", """    <h2>Transition Dynamics</h2>
    <p class="stepnote"><span class="stepchip" data-step="train">Step 3</span> The learner never sees these formulas:
      the critic must learn their effect from recorded rows, and the stall inside <i>f</i>(<i>a</i>) is what rows past
      80% would teach it.</p>
""")
rep("reference-scores", """    <h2>Reference Scores</h2>
""", """    <h2>Reference Scores</h2>
    <p class="stepnote"><span class="stepchip" data-step="test">Step 5</span> These fixed controllers set the 0 and the
      100 of the points on which every learned actor is scored.</p>
""")
rep("rewards", """Thrusters 2, 3 and 4 give what the stall-aware
      controller gives here""", """Thrusters 2, 3 and 4 give what the stall-aware
      controller (the fixed 100-point controller, which never goes past 80%) gives here""")
rep("episode-end", """(at the rows the harm checks score, for the price of pushing past 80%, and the exact over-preference values the detection checks are scored against)""",
    """(at the rows that the harm checks, the four checks that had to pass before BCA could run, score for the price of pushing past 80%, and the exact over-preference values, how much more the critic prefers a command than the exact values say, against which BCA's detection would be scored)""")
rep("episode-end", """Values that are computed many times over (during training, for the noise map and for the forecasts, all""",
    """Values that are computed many times over (during training, for the noise map, a map of where TD3+BC's noisy training target varies most, used as a control, and for the forecasts, the design plan's predictions made before any run, all""")
rep("episode-end", "<dd>How many steps are simulated: 1,000 or 600.</dd>", "<dd>It is how many steps are simulated: 1,000 or 600.</dd>")
rep("episode-end", "<dd>The most the rest of the never-ending sum can add after", "<dd>It is the most that the rest of the never-ending sum can add after")
rep("episode-end", "<dd>The never-ending discounted sum of rewards that a learner's critic estimates.</dd>",
    "<dd>It is the never-ending discounted sum of rewards that a learner's critic estimates.</dd>")

# ===================================================================================================== 7. The task
rep("task", """the water's push at the state, <i>c</i> + <i>w</i><sub>1</sub>, into bands 0.2 wide. Each dot is one band's average for""",
    """the water's push at the state, <i>c</i> + <i>w</i><sub>1</sub>, into bins 0.2 wide. Each dot is one bin's average for""")
rep("task", "marks matter. The gold band from 0.565 on is the", "marks matter. The gold strip from 0.565 on is the")
rep("task", """pushes. That second line is how a learned actor acts when its critic never saw the stall. The full-throttle region also""",
    """pushes. That second line shows the direction an actor drifts when its critic never saw the stall; the learned
      actor goes part of the way (past 80% on 73% of strong-current states, against 84.0% for the rising-thrust
      controller). The full-throttle region also""")
rep("task", """<figcaption>Thruster 1's average throttle (and, for the rising-thrust controller, the push it really gives) in each
        band of the water's push, over every state of the four evaluation banks in the stalling world. The table below
        lists every band.""",
    """<figcaption>Thruster 1's average throttle (and, for the rising-thrust controller, the push it really gives) in each
        0.2-wide bin of the water's push, over every state of the four evaluation banks in the stalling world. The table below
        lists every bin.""")
# the chart's legend: complete sentences, and "bin" / "strip" instead of "band"
for old, new in [
        ("<dd>The water's push <i>c</i> + <i>w</i><sub>1</sub> at the state (<i>c</i> the current, <i>w</i><sub>1</sub> the wave's push now), in bands 0.2 wide.</dd>",
         "<dd>It is the water's push <i>c</i> + <i>w</i><sub>1</sub> at the state (<i>c</i> the current, <i>w</i><sub>1</sub> the wave's push now), sorted into bins 0.2 wide.</dd>"),
        ("<dd>Thruster 1's average throttle or push. Negative is upstream,", "<dd>It is thruster 1's average throttle or push. Negative is upstream,"),
        ("<dd>At −0.8: past it the propeller stalls.</dd>", "<dd>It is the line at −0.8, and past it the propeller stalls.</dd>"),
        ("""<dt><i class="key tile" data-k="region"></i>Gold band</dt><dd>It is the full-throttle region:""",
         """<dt><i class="key tile" data-k="region"></i>Gold strip</dt><dd>It is the full-throttle region:"""),
        ("<dd>What the rising-thrust controller's thruster 1 really delivers: it falls back once it stalls,",
         "<dd>It is what the rising-thrust controller's thruster 1 really delivers, which falls back once it stalls,"),
        ("<dd>One band's average. Hover over or tap anywhere in a band's column for its three numbers;",
         "<dd>Each dot is one bin's average. Hover over or tap anywhere in a bin's column for its three numbers;")]:
    rep("task", old, new)
# the drawing's legend: complete sentences
for old, new in [
        ("<dd>The current's strength: drawn at the start, then steady all episode; it pushes every float downstream equally. 0.1 to 0.7 in the recorded conditions; 0.445 on this start. The solid part of each top arrow.</dd>",
         "<dd>It is the current's strength, drawn at the start and then steady all episode, and it pushes every float downstream equally. It is 0.1 to 0.7 in the recorded conditions and 0.445 on this start, and it is the solid part of each top arrow.</dd>"),
        ("<dd>The wave's push now: it runs from plus the amplitude down to minus the amplitude and back every 23.27 steps. This start's amplitude is 0.218. The dashed part of each top arrow.</dd>",
         "<dd>It is the wave's push now, which runs from plus the amplitude down to minus the amplitude and back every 23.27 steps. This start's amplitude is 0.218, and the wave is the dashed part of each top arrow.</dd>"),
        ("<dd>The water's push on every float in this state. Downstream unless the wave takes away more than the current gives.</dd>",
         "<dd>It is the water's push on every float in this state, which points downstream unless the wave takes away more than the current gives.</dd>"),
        ("<dd>+ when the wave adds to the current's push, − when it takes from it; the faint dashed part is then the push it takes away.</dd>",
         "<dd>The sign is + when the wave adds to the current's push and − when it takes from it; the faint dashed part is then the push it takes away.</dd>"),
        ("<dd>Where every float's two numbers are 0: the dashed line.</dd>",
         "<dd>It is where every float's two numbers are 0, and the dashed line marks it.</dd>"),
        ("<dd>A float's period, the steps one swing takes, shown rounded to whole steps under each float:",
         "<dd>It is a float's period, the steps one swing takes, shown rounded to whole steps under each float:"),
        ("<dd>Up to 80%, a thruster pushes exactly as hard as its throttle.</dd>",
         "<dd>It means that up to 80% a thruster pushes exactly as hard as its throttle.</dd>"),
        ("<dd>Past 80% the propeller stalls: push 0.65 at 90% and 0.5 at full throttle, while the throttle cost keeps rising.</dd>",
         "<dd>It means that past 80% the propeller stalls: the push is 0.65 at 90% and 0.5 at full throttle, while the throttle cost keeps rising.</dd>"),
        ("<dd>Thruster 1's throttle (the bar's fill) and the push it really gives (the tick).</dd>",
         "<dd>They show thruster 1's throttle (the bar's fill) and the push it really gives (the tick).</dd>"),
        ("<dd>While the drawing plays, the top arrows turn gold when the water's push is above 0.565: the state is then in the full-throttle region.</dd>",
         "<dd>While the drawing plays, the top arrows turn gold when the water's push is above 0.565, which means the state is in the full-throttle region.</dd>")]:
    rep("task", old, new)
# the list of situations wears their colours
for name, key in [("Recorded conditions", "recorded"), ("Strong current", "strong_current"), ("Big waves", "big_waves"),
                  ("Big starting pushes", "big_kicks")]:
    rep("task", f"<li><b>{name}</b> (", f'<li><b><span class="km" data-sit="{key}">{name}</span></b> (')

# ===================================================================================================== 8. Summary: why the stall makes this a test
rep("description", "2.3% to 2.7% of the rows in each step of 2%", "4.6% to 5.3% of the rows in each 4% bar", count=2)
rep("description", """With
      almost nothing past 80% to correct it, the critic extends that trend: compared with thruster 1 at 80%, its value rises by
      0.04 at 90% and by 0.06 at full throttle, while the true value falls by 0.21 and by 0.46. The critic trained on the full
      recording, which saw the stall, does not prefer full throttle: its extra preference for full throttle over thruster 1 at
      80%, beyond what the true values say, is −0.06 to −0.01 across the five repeat runs, against +0.33 to +0.36 for the
      critic trained after the removal.""",
    """With
      almost nothing past 80% to correct it, the critic extends that trend. Read at the actor's own commands at training
      states in the full-throttle region (After the stop), its value rises by 0.04 at 90% and by 0.06 at full throttle,
      compared with thruster 1 at 80%, while the true value falls by 0.21 and by 0.46. A second measure, the critic-error
      check's, takes the median over the kept rows of that region: there the full-recording critic, which saw the stall,
      does not prefer full throttle (its extra preference for full throttle over thruster 1 at 80%, beyond what the true
      values say, is −0.06 to −0.01 across the five repeat runs), against +0.33 to +0.36 for the critic trained after the
      removal, where the true loss from pushing past 80% is 0.33 to 0.35.""")
rep("description", """93.7% of strong-current states lie in
      the full-throttle region, and the stall-aware controller sits at its 80% limit on 71.9% of them""",
    """93.7% of strong-current states lie in
      the full-throttle region, and the stall-aware controller holds thruster 1 at its 80% limit on 71.9% of all
      strong-current states""")
rep("description", """The actor trained against the critic that never saw the stall pushes
      past 80%""", """The actor trained against the critic trained after the removal, which never saw the stall, pushes
      past 80%""")
# the hover readouts move directly above the chart, under the three sentences, so a tap shows its numbers on screen
a, b = grab("description", '<div class="readouts">\n        <div class="ro"><span class="k">throttle size</span>', "</div>\n      </div>\n")
ro_block = s[a:b]
assert ro_block.count('<div class="ro">') == 4
splice(a, b, "")
rep("description", """      <svg class="chart sc-svg" id="sc-svg\"""", "      " + ro_block.rstrip("\n").lstrip() + """
      <svg class="chart sc-svg" id="sc-svg\"""")
# the chart: the crosshair is drawn first, under the labels, which sit on a halo (CSS)
rep("script", """      hatch(mk("defs", {}, svg), "sc-hx", "hx-reg");
      // the stall region, in both panels
      mk("rect", { x: X(0.8), y: tA, width: X(1) - X(0.8), height: hA, class: "ix-zone" }, svg);
      mk("rect", { x: X(0.8), y: tB, width: X(1) - X(0.8), height: hB, class: "ix-zone" }, svg);""",
    """      hatch(mk("defs", {}, svg), "sc-hx", "hx-reg");
      // the stall region, in both panels
      mk("rect", { x: X(0.8), y: tA, width: X(1) - X(0.8), height: hA, class: "ix-zone" }, svg);
      mk("rect", { x: X(0.8), y: tB, width: X(1) - X(0.8), height: hB, class: "ix-zone" }, svg);
      // the crosshair at the slider's throttle, or at the pointer: under the curves and the labels, so it never cuts a label
      const a = hov != null ? hov : Number(inp.value), xa = X(a), bi = binOf(a);
      mk("rect", { x: X(ME[bi]) + 0.5, y: tB, width: X(ME[bi + 1]) - X(ME[bi]) - 1, height: hB, class: "ix-hl" }, svg);
      mk("line", { x1: xa, x2: xa, y1: tA, y2: bB, class: "ix-cross" }, svg);""")
rep("script", """      // the crosshair at the slider's throttle, or at the pointer
      const a = hov != null ? hov : Number(inp.value), xa = X(a), bi = binOf(a);
      mk("rect", { x: X(ME[bi]) + 0.5, y: tB, width: X(ME[bi + 1]) - X(ME[bi]) - 1, height: hB, class: "ix-hl" }, svg);
      mk("line", { x1: xa, x2: xa, y1: tA, y2: bB, class: "ix-cross" }, svg);
      mk("circle", { cx: xa, cy: YA(costOf(a)), r: 4.5, class: "sc-dc" }, svg);""",
    """      // the crosshair's three dots, on top
      mk("circle", { cx: xa, cy: YA(costOf(a)), r: 4.5, class: "sc-dc" }, svg);""")

# ===================================================================================================== 9. The recording controller, the recorded data
rep("recording-controller", "what a learned actor would do if it were trained against a critic that never saw the stall.",
    "the direction a learned actor drifts when it is trained against a critic that never saw the stall (the learned actor goes only part of the way).")
rep("dataset", "<td>never trained on; used only to measure distances and strengths</td>",
    "<td>never trained on; used only to measure how far a state lies from the data (the shift table) and how much caution changes the actor (its strength, Methods)</td>")
for old, new in [("<dd>The step within the episode, 0 to 99.</dd>", "<dd>It is the step within the episode, from 0 to 99.</dd>"),
                 ("<dd>The recorded throttle: −1 is full throttle upstream, against the water.", "<dd>It is the recorded throttle, where −1 is full throttle upstream, against the water."),
                 ("<dd>The full-throttle region: the water's push", "<dd>They are the steps in the full-throttle region, where the water's push")]:
    rep("dataset", old, new)

# ===================================================================================================== 10. The distribution shift: two removals kept apart
rep("shift", """<p><b>The 80% hold is inside the ball too.</b> The capped pattern (the full-throttle pattern with thruster 1 at 80%;
      see the table in""",
    """<p><b>The capped pattern is inside the ball too.</b> The capped pattern (the full-throttle pattern with thruster 1
      at 80%, which is not the weighting test's 80% hold, a different controller's command; see the table in""")
rep("shift", "deletes most recorded commands near the safe 80% hold, not only the full-throttle ones.",
    "deletes most recorded commands near the capped pattern, the safe command with thruster 1 at 80%, not only the full-throttle ones.")
rep("shift", "how much more the critic prefers full throttle over the 80% hold than the exact values do",
    "how much more the critic prefers full throttle over the capped pattern than the exact values do")
rep("shift", "both commands it compares, full throttle and the 80% hold, are missing",
    "both commands it compares, full throttle and the capped pattern, are missing")

# ===================================================================================================== 11. The critic in action (section markup only)
rep("critic-in-action", """<p>The critic's estimate of the return if the actor gives command <i>a</i> in state <i>s</i> and then keeps driving as it does. The return is the sum of all the rewards that follow, each step discounted by a factor of 0.98.""",
    """<p>The critic's estimate of the discounted return if the actor gives command <i>a</i> in state <i>s</i> and then keeps driving as it does: the sum of all the rewards that follow, each counting 0.98 times as much as the one before it. (The return used for scores adds an episode's 100 rewards with no discount.)""")
QV_STEPS = {
    "qv-lg-q": """<p class="stepnote"><span class="stepchip" data-step="collect">Step 1</span> Which recording the learner gets,
            from all random commands to the expert recording controller alone.</p>""",
    "qv-lg-c": """<p class="stepnote"><span class="stepchip" data-step="remove" data-k="hold">Step 2</span> With or without the
            weighting test's removal, which takes out the 80% hold.</p>""",
    "qv-lg-s": """<p class="stepnote"><span class="stepchip" data-step="test">Step 5</span> The training mix of currents (in
            distribution, the weighting test's no-shift setting) or the strong test mix (a dynamics extrapolation, where
            strong currents are common).</p>""",
}
for lid, note in QV_STEPS.items():
    a, b = grab("critic-in-action", f'<div class="qv-ctrl" role="group" aria-labelledby="{lid}">', "\n          </div>\n        </div>",
                include_end=False)
    end = b + len("\n          </div>")
    splice(end, end, "\n          " + note)
rep("critic-in-action", """">Training mix of currents</button>""", """"><span class="km" data-sit="recorded">Training mix of currents</span></button>""")
rep("critic-in-action", """">Strong test mix</button>""", """"><span class="km" data-sit="strong_current">Strong test mix</span></button>""")
for old, new in [
        ("<dd>The state on screen: the five floats' positions and speeds, the current <i>c</i> and the wave numbers w₁ and w₂.</dd>",
         "<dd>It is the state on screen, all 13 observed numbers: the five floats' positions and speeds, the current <i>c</i> and the wave numbers w₁ and w₂. Methods writes this whole observation <i>x</i> and keeps <i>s</i> for the 10 float numbers.</dd>"),
        ("<dd>A command: the four throttles. In the upper chart", "<dd>It is a command, the four throttles. In the upper chart"),
        ("<dd>The command the actor gives in state <i>s</i>; π is the trained actor.</dd>",
         "<dd>It is the command the actor gives in state <i>s</i>, where π is the trained actor.</dd>"),
        ("<dd>The critic's value: its estimate of the return from giving <i>a</i> in <i>s</i> and letting the actor drive after that. The smaller of the critic's two estimates, min(Q<sub>1</sub>, Q<sub>2</sub>).</dd>",
         "<dd>It is the critic's value: its estimate of the discounted return from giving <i>a</i> in <i>s</i> and letting the actor drive after that. It is the smaller of the critic's two estimates, min(Q<sub>1</sub>, Q<sub>2</sub>).</dd>"),
        ("<dd>The true value of the same thing, by exact simulation: 600 steps, discount 0.98.</dd>",
         "<dd>It is the true value of the same command, by exact simulation over 600 steps at discount 0.98. Methods writes this exact value Q<sup>true</sup>.</dd>"),
        ("<dd>The swept throttle the critic rates highest in this state.</dd>", "<dd>It is the swept throttle that the critic rates highest in this state.</dd>"),
        ("<dd>The swept throttle with the highest true value in this state.", "<dd>It is the swept throttle with the highest true value in this state."),
        ("<dd>The stall limit. Past it the push falls", "<dd>It is the stall limit. Past it the push falls"),
        ("<dd>Throttles whose command lies inside this section's removed ball,", "<dd>These are the throttles whose command lies inside this section's removed ball,"),
        ("<dd>This section's own scale: 100 ×", "<dd>It is this section's own scale: 100 ×")]:
    rep("critic-in-action", old, new)

# ===================================================================================================== 12. Methods
rep("methods", """<p><b>Which removal.</b> Everything in this section serves the <b>main test</b>. Its training data lose the""",
    """<p><b>Which removal.</b> Everything in this section serves the <b>main test</b> unless marked; part 2's worked
      example and part 7 also serve the weighting test, whose data lose the 80% hold instead. The main test's training data
      lose the""")
rep("methods", """<td>It has two consequences. The learnable-values check (<a href="#results-world">Results</a>) asks that this
            noise not swamp the differences between recorded commands (design seed 900: 0.298 against 0.382). And the""",
    """<td>It has two consequences. World check 8 in <a href="#results-world">Results (Already measured, before any
            learning)</a> asks that the spread of the exact value under this noise (0.298 on design seed 900) stay below
            its spread under the recording controller's own noise (0.382). And the""")
rep("methods", """<td>In the main test the return-cost check failed narrowly at 2.5 (1.46 against a required 1.5 on its
            past-80% rule) and passed at 5.""",
    """<td>In the main test the return-cost check failed narrowly at 2.5 (its rule that the actor of the critic trained
            after the removal push past 80% at least 1.5 times as often as the actor of the full-recording critic gave
            1.46) and passed at 5.""")
rep("methods", """is the median second derivative of that value along each
      throttle, at normal driving's own command.""",
    """is how sharply that value bends there: how fast its slope changes as
      one throttle changes (its second derivative), the median over the throttles, at normal driving's own command.""")
rep("methods", """state is the first one of its episode from the strong test's start: the water
        pushes with 0.94, and the actor holds thruster 1 at 77.5% upstream. The reward is −4.67. The critic's value is
        −44.71 here and −41.65 at the next state. Read with the critic itself and no target noise, the one-step target
        is −4.67 + 0.98 × (−41.65) = −45.48, so the one-step error is −45.48 − (−44.71) = −0.77. The exact value here
        is −45.34, so the error against the exact value is −44.71 − (−45.34) = +0.63: the critic rates this command
        0.63 too high, even though its one-step error points the other way. The exact value has no one-step error of
        its own: −4.67 + 0.98 × (−41.51) = −45.34.</p>""",
    """state is the first one of its episode from the strong test's start: the water
        pushes with 0.94, and the actor holds thruster 1 at 77.5% upstream. The reward is −4.666. The critic's value is
        −44.714 here and −41.651 at the next state. Read with the critic itself and no target noise, the one-step target
        is −4.666 + 0.98 × (−41.651) = −45.484, so the one-step error is −45.484 − (−44.714) = −0.770. The exact value
        here is −45.343, so the error against the exact value is −44.714 − (−45.343) = +0.629: the critic rates this
        command 0.63 too high, even though its one-step error points the other way. The exact value has no one-step
        error of its own: −4.666 + 0.98 × (−41.507) = −45.343.</p>""")
rep("methods", "<li><b>Reversed caution</b>, a sanity check reported beside", "<li><b>Reversed caution</b>, a check of the method reported beside")
rep("methods", """<td><span class="stepchip" data-step="band">Step 4</span> The lever of BCA's band, placed by the exact""",
    """<td><span class="stepchip" data-step="band">Step 4</span> BCA's kind of caution, placed by the exact""")
rep("methods", """<p class="stepnote"><span class="stepchip" data-step="band">Step 4</span> BCA's lever, with no BCA: the dose is
      placed by the exact error instead of by BCA's estimate, to see whether the lever can work at all.</p>
    <p><b>The problem.</b> BCA's only lever is the dose on the pull.""",
    """<p class="stepnote"><span class="stepchip" data-step="band">Step 4</span> BCA's kind of caution, without BCA: the
      dose is placed by the exact error instead of by BCA's estimate, to see whether this kind of caution can work at
      all.</p>
    <p><b>The problem.</b> BCA's only way to act is the dose on the pull.""")
rep("methods", """<p class="stepnote"><span class="stepchip" data-step="train">Step 3</span> Train the critic only. These critics
      measure what <span class="stepchip in" data-step="remove">Step 2</span>, the removal, did to the critic, with
      nothing else changing.</p>""",
    """<p class="stepnote"><span class="stepchip" data-step="train">Step 3</span><span class="stepchip" data-step="remove">Step 2</span>
      Only the critic is trained. These critics measure what the removal did to the critic, with nothing else
      changing.</p>""")
rep("methods", """<p class="stepnote"><span class="stepchip" data-step="train">Step 3</span> Train the actor only, then
      <span class="stepchip in" data-step="test">Step 5</span> test it. The swap isolates the critic as the cause of
      any lost return.</p>""",
    """<p class="stepnote"><span class="stepchip" data-step="train">Step 3</span><span class="stepchip" data-step="test">Step 5</span>
      Only the actor is trained, and then it is tested. The swap isolates the critic as the cause of any lost
      return.</p>""")

# ===================================================================================================== 13. The harm checks
rep("harm-checks", """<p class="stepnote"><span class="stepchip" data-step="remove">Step 2</span> seen through
          <span class="stepchip in" data-step="train">Step 3</span>: did the main test's removal of the densest 15% of
          commands make the critic wrong exactly there?</p>""",
    """<p class="stepnote"><span class="stepchip" data-step="remove">Step 2</span><span class="stepchip" data-step="train">Step 3</span>
          The removal, seen in the trained critic: did the main test's removal of the densest 15% of commands make the
          critic wrong exactly there?</p>""")
rep("harm-checks", """<p class="stepnote"><span class="stepchip" data-step="train">Step 3</span> Train the actor only, then
          <span class="stepchip in" data-step="test">Step 5</span> test it: does an actor trained against the wrong
          critic lose return where the missing commands matter?</p>""",
    """<p class="stepnote"><span class="stepchip" data-step="train">Step 3</span><span class="stepchip" data-step="test">Step 5</span>
          An actor is trained against each critic and then tested: does an actor trained against the wrong critic lose
          return where the missing commands matter?</p>""")
rep("harm-checks", """<p class="stepnote"><span class="stepchip" data-step="band">Step 4</span> BCA's lever, aimed perfectly and with
          no BCA: if it cannot help here, no result of BCA could show that it helps.</p>""",
    """<p class="stepnote"><span class="stepchip" data-step="band">Step 4</span> BCA's kind of caution, a harder pull,
          placed perfectly and without BCA: if even that cannot help, no result of BCA could show that it helps.</p>""")
rep("harm-checks", "beat the same caution placed blindly by 3 points", "beat the same caution placed without that knowledge by 3 points")

# ===================================================================================================== 14. Results
# 14a. reasoning before the verdict: a neutral opening; the verdict goes into the summary after the three questions
a, b = grab("results", '<p class="status"><span class="tag">stopped before BCA</span> The main test stopped at the third of its four', "</p>")
splice(a, b, """<p>This section reports the main test: what each of its checks asked, how it was measured, what was expected,
      what happened and what followed. The exploratory work after it is in <a href="#beyond-the-data">After the
      stop</a>, at the end of this section; the main line of work on this environment is now
      <a href="#weighting-test">the weighting test</a>, the next section.</p>""")
rep("results", """<h3 id="results-short">What was found, in short</h3>
    <ol>""", """<h3 id="results-short">What was found, in short</h3>
    <p class="status"><span class="tag">stopped before BCA</span> A summary of the three checks below. Every rule and
      pass mark had been written down before any learner ran and saved with a fingerprint, a code computed from the file
      that changes if any character of it changes; that written plan is the <i>pre-registration</i>, so the stop was
      decided by rules fixed in advance, not by judgement after the results were in. By its rules the main test stopped
      at the third of its four checks, before BCA (Bayesian Conformal Aggregation) was ever run.</p>
    <ol>""")
rep("results", "it scored 1.9 points <i>below</i> blind caution of the same strength in strong current, and",
    "it scored 1.9 points <i>below</i> caution of the same strength placed without that knowledge in strong current, and")
# 14b. step lines as plain sentences
rep("results", """<p class="stepnote"><span class="stepchip" data-step="remove">Step 2</span> seen through
      <span class="stepchip in" data-step="train">Step 3</span>: did the main test's removal make the trained critic
      wrong exactly at the removed commands?""",
    """<p class="stepnote"><span class="stepchip" data-step="remove">Step 2</span><span class="stepchip" data-step="train">Step 3</span>
      The removal, seen in the trained critic: did the main test's removal make the critic wrong exactly at the removed
      commands?""")
rep("results", """<p class="stepnote"><span class="stepchip" data-step="train">Step 3</span> Train an actor against each critic, then
      <span class="stepchip in" data-step="test">Step 5</span> test it: does the wrong critic cost return where the
      missing commands matter?""",
    """<p class="stepnote"><span class="stepchip" data-step="train">Step 3</span><span class="stepchip" data-step="test">Step 5</span>
      An actor is trained against each critic and then tested: does the wrong critic cost return where the missing
      commands matter?""")
rep("results", """<p class="stepnote"><span class="stepchip" data-step="band">Step 4</span> BCA's lever, aimed perfectly and with no
      BCA: if it cannot help here, no result of BCA could show that it helps.""",
    """<p class="stepnote"><span class="stepchip" data-step="band">Step 4</span> BCA's kind of caution, a harder pull,
      placed perfectly and without BCA: if even that cannot help, no result of BCA could show that it helps.""")
rep("results", """<p class="stepnote"><span class="stepchip" data-step="collect">Step 1</span> and
      <span class="stepchip in" data-step="remove">Step 2</span>, before any learning: do the recording and the main
      test's removal have the properties the test relies on?</p>""",
    """<p class="stepnote"><span class="stepchip" data-step="collect">Step 1</span><span class="stepchip" data-step="remove">Step 2</span>
      Before any learning: do the recording and the main test's removal have the properties the test relies on?</p>""")
# 14c. the main test's capped pattern is not the weighting test's 80% hold
rep("results", "full throttle over the 80% hold than the exact values say,",
    "full throttle over the capped pattern (the same command with thruster 1 at 80%; not the weighting test's 80% hold) than the exact values say,")
# 14d. the return-cost check: the two controls' numbers, and why critic weight 5 separates the actors
rep("results", """Removing the same number of rows
        at random, or the same removal in the no-stall world, explains almost none of it. The actor of the critic after
        the removal pushes past 80% 1.77 times as often, and a step-by-step accounting of the return puts the loss at
        the states where it does.</dd>""",
    """Removing the same number of rows
        at random changes the score by only −10 to +11 points, and the same removal in the no-stall comparison world by
        −1 to +2 points, so neither having less data nor the removal on its own explains the loss: the hidden stall does.
        The actor of the critic after the removal pushes past 80% 1.77 times as often, and a step-by-step accounting of
        the return puts the loss at the states where it does. With more freedom from the recorded commands, the actor of
        the full-log critic, whose data show the stall, learns to stay at or below 80%, while the actor of the critic
        trained after the removal, which never saw the stall, pushes further past it.</dd>""")
# 14e. loose words
rep("results", """The design plan forecast that perfectly placed caution would beat each of the blind
        controls by 4.7""", """The design plan forecast that perfectly placed caution would beat each of the controls
        placed without that knowledge by 4.7""")
rep("results", """harm-placed caution beat
        every blind control on all five seeds""", """harm-placed caution beat
        every control placed without that knowledge on all five seeds""")
rep("results", """caution scored 1.9 points <i>below</i> blind caution of the same strength. With every dose""",
    """caution scored 1.9 points <i>below</i> caution of the same strength placed without that knowledge. With every dose""")
rep("results", "(0.2 points\n        over blind caution of the same strength)", "(0.2 points\n        over caution of the same strength placed without that knowledge)")
rep("results", "<dt>What it asks</dt><dd>BCA has one lever, <span", "<dt>What it asks</dt><dd>BCA has one way to act, <span")
rep("results", """Whether BCA aims that
        lever at the right rows""", """Whether BCA aims that
        caution at the right rows""")
rep("results", "That pull is the only lever BCA has as built.", "That pull is the only way BCA as built can act.")
rep("results", "This is a statement about the lever, not about BCA's signal:", "This is a statement about the pull, not about BCA's signal:")
rep("results", "The follow-up tested the other lever directly,", "The follow-up tested the other channel directly,")
rep("results", "BCA's band can act through that lever too, by its", "BCA's band can act through that channel too, by its")
rep("results", "The stop is about the lever, not about BCA's signal.", "The stop is about the pull, not about BCA's signal.")
# 14f. the Stage 1 penalty: how much more the critic overrates the actor's own command than the recorded one
rep("results", """A penalty on the critic's value at the actor's own command, set by how much the critic's exact
      one-step error there exceeds its error at the recorded command, won back""",
    """A penalty on the critic's value at the actor's own command, set by how much more the critic overrates that
      command than the recorded one, measured with its exact one-step error, won back""")
rep("results", """penalty on the critic's value there, set by how much the critic's exact one-step error at that command exceeds
        its error at the recorded command, won back""",
    """penalty on the critic's value there, set by how much more the critic overrates that command than the recorded
        one, measured with its exact one-step error, won back""")
# 14g. difficulty levels: standard names, no claimed conventional ranking; the 11% to 17% range; the tie in return units
rep("results", """<p><b>The four situations as difficulty levels.</b> Reinforcement learning usually ranks test conditions by how far
      they are from the training data: in distribution, then shifts of the starting state, of the disturbances and of
      the dynamics. The table orders them instead by the harm the missing commands cause, because that is what this
      test is about.""",
    """<p><b>The four situations as difficulty levels.</b> In common reinforcement-learning terms the four situations
      are in distribution, an initial-state shift, a disturbance shift and a dynamics extrapolation. The table orders
      them by the harm the missing commands cause, because that is what this test is about.""")
rep("results", '<td class="n">11% to 18%</td>', '<td class="n">11% to 17%</td>')
rep("results", """    </table></div>
    <p><b>Reading the ladder.</b>""", """    </table></div>
    <p class="note">In units of return the middle two levels are tied (11.8 and 11.7); they differ in points only because
      a point is worth 1.0 unit of return with big starting pushes and 0.71 in big waves.</p>
    <p><b>Reading the ladder.</b>""")
rep("results", """only 11% to 18% of the visited states lie
      outside the recorded conditions, and the stall-aware controller needs a removed command at 14.3% of its states,
      fewer than in the recorded conditions: the loss is 11.8. <span class="km" data-sit="big_waves">Big waves</span>
      lie wholly outside the recording and need a removed command at 28.0% of the states: 16.6.""",
    """only 11% to 17% of the visited states lie
      outside the recorded conditions, and the stall-aware controller needs a removed command at 14.3% of its states,
      fewer than in the recorded conditions: the loss is 11.8 points. <span class="km" data-sit="big_waves">Big waves</span>
      lie wholly outside the recording and need a removed command at 28.0% of the states: 16.6 points, though in units
      of return these two levels lose about the same.""")
# 14h. the timeline in plain events; the test-fix note and the commit codes go into a collapsed sources block
rep("results", """<li>15:07: the code's automatic tests passed. A first attempt had failed on one test that wrongly required the
        harm checks' output folder not to exist. The faulty test was changed to check only that running the tests leaves that folder unchanged,
        before anything was written down or trained.</li>""", """<li>15:07: the code's automatic tests passed.</li>""")
a, b = grab("results", """<p class="note">Sources in the repository: the pre-registration (SHA-256 fingerprint""", "</p>")
src1 = s[a:b]
splice(a, b, """<details class="srcs"><summary>Sources, for the project team</summary><div class="inner">
      """ + src1.replace("</p>", """ A first attempt at the code's automatic tests had failed on one test that wrongly
        required the harm checks' output folder not to exist; the faulty test was changed to check only that running the
        tests leaves that folder unchanged, before anything was written down or trained.</p>""") + """
    </div></details>""")
# 14i. After the stop: undefined names and code-like detail
rep("results", """This breaks the
        published design rule that the actor's training never reaches back through the calibration, which is why it is
        declared as a new method version.</li>""",
    """So the actor
        also learns from how the band's width changes as its command changes, which BCA as published never lets the actor
        use; this is why it is declared as a new method version.</li>""")
rep("results", """is written
          as Stage 2's pre-registration (sixth version):""",
    """is written
          up as Stage 2's pre-registration, a plan fixed in advance (the sixth version of that written plan):""")
rep("results", "Third, the locked rehearsal of how detection will be scored has finished.",
    "Third, the rehearsal of how detection will be scored, a practice run of the scoring code on data with no BCA, locked before it ran, has finished.")
rep("results", """The
          <i>trace test bed</i> keeps 0.5% or 2% of the
          removed past-80% rows, in both worlds, seeds 0–2.
          It asks""",
    """The
          <i>trace test bed</i>, data that keep 0.5% or 2% of the
          removed past-80% rows, in both worlds, seeds 0–2,
          asks""")
a, b = grab("results", """<p class="note">Sources in the repository: the plan and its evidence in""", "</p>")
src2 = s[a:b]
splice(a, b, """<details class="srcs"><summary>Sources, for the project team</summary><div class="inner">
      """ + src2 + """
    </div></details>""")

# ===================================================================================================== 15. The weighting test
wa, wb = region("weighting-test")
W = s[wa:wb]
MARK = [("status", '<p class="status"><span class="tag ok">pre-registered results: complete</span>'),
        ("intro", "<p>The stop above did not end the work"),
        ("claims", '<div class="iw-claims">'),
        ("pipe", '<div class="iw-pipe"'),
        ("results", '<h3 id="iw-results">'),
        ("howto", "<p><b>How to read the numbers.</b>"),
        ("qual", "<h4>The five data qualities</h4>"),
        ("findings", '<div class="finding">\n      <p class="f-title">1. With no shift'),
        ("explore", '<h4 id="iw-explore">'),
        ("shift", '<h3 id="iw-shift">'),
        ("hold", '<h3 id="iw-hold">'),
        ("why", "<h3>Why weighting should matter</h3>"),
        ("ran", "<h3>What ran, in order</h3>"),
        ("marks", "<h3>Pass marks, and what each found</h3>")]
pos = []
for name, m in MARK:
    assert W.count(m) == 1, (name, W.count(m))
    pos.append(W.index(m))
assert pos == sorted(pos), pos
end = W.rindex("</section>")
B = {name: W[pos[i]:(pos[i + 1] if i + 1 < len(pos) else end)] for i, (name, _) in enumerate(MARK)}
head = W[:pos[0]]
tail = W[end:]


def brep(name, old, new, count=1):
    n = B[name].count(old)
    assert n == count, f"[iw {name}] expected {count} of {old[:100]!r}, found {n}"
    B[name] = B[name].replace(old, new)
    N["n"] += 1


# the status and the progress strip go to "What ran, in order"
brep("status", "(finding 4 below)", "(finding 4 above)")
brep("ran", "<h3>What ran, in order</h3>\n", "<h3>What ran, in order</h3>\n    " + B["status"].strip() + "\n    " + B["pipe"].strip() + "\n")
# the known shift: its step, the names of the three test settings, and the buttons' names
brep("shift", """    <p>Each distribution of the current leans""", """    <p class="stepnote"><span class="stepchip" data-step="test">Step 5</span> Which shifted test the band is weighted
      for. Why: a recorded step's weight is the test's chance of its current divided by training's chance.</p>
    <p>Each distribution of the current leans""")
brep("shift", """strong ones. Training uses ξ = −2.""", """strong ones. Training uses ξ = −2. The test comes in three settings, named by how they lean: the
      <i>training mix</i> (ξ = −2, <i>no shift</i>), the <i>mild test mix</i> (ξ = 0) and the <i>strong test mix</i> (ξ = +2,
      the <i>strong test</i> for short).""")
brep("shift", """data-l="-2">No shift</button>""", """data-l="-2">Training mix (no shift)</button>""")
brep("shift", """data-l="0">Mild</button>""", """data-l="0">Mild test mix</button>""")
brep("shift", """data-l="2" aria-pressed="true">Strong</button>""", """data-l="2" aria-pressed="true">Strong test mix</button>""")
brep("shift", """data-tip-title="No shift" data-tip="The test lean is ξ = −2""", """data-tip-title="Training mix (no shift)" data-tip="The test lean is ξ = −2""")
brep("shift", """data-tip-title="Mild" data-tip="With the lean ξ = 0""", """data-tip-title="Mild test mix" data-tip="With the lean ξ = 0""")
brep("shift", """data-tip-title="Strong" data-tip="With the lean ξ = +2""", """data-tip-title="Strong test mix" data-tip="With the lean ξ = +2""")
brep("shift", """<span id="iw-lg-test-t">Test (strong, ξ = +2)</span>""", """<span id="iw-lg-test-t">Test (strong test mix, ξ = +2)</span>""")
brep("shift", "<tr data-sit=\"recorded\"><td>No shift</td>", "<tr data-sit=\"recorded\"><td>Training mix (no shift)</td>")
brep("shift", "<tr data-iw=\"mild\"><td>Mild (uniform)</td>", "<tr data-iw=\"mild\"><td>Mild test mix (uniform)</td>")
brep("shift", "<tr data-sit=\"strong_current\"><td>Strong</td>", "<tr data-sit=\"strong_current\"><td>Strong test mix</td>")
brep("shift", """    <div class="panel">
      <div class="iw-controls">""", """    <div class="panel">
      <p class="stepnote"><span class="stepchip" data-step="test">Step 5</span> Pick the shifted test the band is weighted
        for; the weight of a recorded step is the test's chance of its current over training's chance.</p>
      <div class="iw-controls">""")
# the removed command: its steps
brep("hold", """<h3 id="iw-hold">The removed command: a common, good move</h3>
""", """<h3 id="iw-hold">The removed command: a common, good move</h3>
    <p class="stepnote"><span class="stepchip" data-step="collect">Step 1</span><span class="stepchip" data-step="remove" data-k="hold">Step 2</span>
      This test's recording controller holds thruster 1 at 80% in hard water, and this test's removal deletes that
      hold.</p>
""")
brep("hold", """<p class="note" id="iw-thr-read">Push 0.800 · throttle cost 0.192</p>""",
     """<p class="note" id="iw-thr-read">Push 0.800 · throttle cost 0.192</p>
          <p class="stepnote"><span class="stepchip" data-step="collect">Step 1</span> Why the hold is the best move in hard
            water: past 80% more throttle buys less push, so 80% gives the most push without a stall.</p>""")
brep("hold", """0.910·<i>c</i> + 0.822·w₁ − 0.320·w₂, is above 0.65.</p>""", """0.910·<i>c</i> + 0.822·w₁ − 0.320·w₂, is above 0.65.</p>
      <p class="stepnote"><span class="stepchip" data-step="collect">Step 1</span> How this test's recording controller
        decides to hold, which is the command step 2 then removes. Why: hard water is where the hold, and so the missing
        data, matter.</p>""")
# why weighting should matter: its steps
brep("why", """        <h4>Coverage at test as the shift grows</h4>
""", """        <h4>Coverage at test as the shift grows</h4>
        <p class="stepnote"><span class="stepchip" data-step="band">Step 4</span><span class="stepchip" data-step="test">Step 5</span>
          The band, checked on shifted tests: why weighting should keep its 90% promise as the test leans further from
          training.</p>
""")
# how to read the results, after the setup
brep("howto", """<p><b>How to read the numbers.</b> The test episodes come in three <i>test settings</i>, which differ only in how
      often each strength of current appears: the <i>strong test</i>, in which strong currents are the most common;
      the <i>mild test</i>, which makes every current from 0.1 to 1.1 equally likely; and <i>no shift</i>, which draws
      currents as training did (the section <a href="#iw-shift">The known shift</a>, below, gives the exact
      distributions).""",
     """<h3 id="iw-read">How to read the results</h3>
    <p>The test episodes come in the three <i>test settings</i> of <a href="#iw-shift">the known shift</a> above,
      which differ only in how often each strength of current appears: the <i>strong test mix</i> (the <i>strong
      test</i> for short), in which strong currents are the most common; the <i>mild test mix</i>, which makes every
      current from 0.1 to 1.1 equally likely; and the <i>training mix</i> (<i>no shift</i>), which draws currents as
      training did.""")
# the five data qualities: a heading of its own, and its step
brep("qual", "<h4>The five data qualities</h4>\n", """<h3 id="iw-qualities">The five data qualities</h3>
    <p class="stepnote"><span class="stepchip" data-step="collect">Step 1</span> Which recording the learner gets: five
      versions of the same recording, from all random commands to the expert recording controller alone.</p>
""")
brep("qual", """(the <i>removed region</i>; the section <a href="#iw-hold">The
      removed command</a>, below, explains it)""", """(the <i>removed region</i>; the section <a href="#iw-hold">The
      removed command</a>, above, explains it)""")
# the findings: a heading, and each finding's step
brep("findings", '<div class="finding">\n      <p class="f-title">1. With no shift', """<h3 id="iw-findings">The pre-registered findings</h3>
    <p>Each finding answers one claim of the specification, with its measurement and pass mark fixed before any
      run.</p>
    <div class="finding">
      <p class="f-title">1. With no shift""")
for title in ["1. With no shift, the band covers what it promises, and it is wider where the critic is more wrong",
              "2. Under a known shift, the unweighted band falls short, and exact weights restore it",
              "3. The command that training never saw is flagged"]:
    brep("findings", f"""<p class="f-title">{title}</p>
""", f"""<p class="f-title">{title}</p>
      <p class="stepnote"><span class="stepchip" data-step="band">Step 4</span><span class="stepchip" data-step="test">Step 5</span>
        The band, measured on held-out steps and on the test.</p>
""")
brep("findings", """<p class="f-title">4. Whether weighting raises return could not be tested under the pre-registered rules</p>
""", """<p class="f-title">4. Whether weighting raises return could not be tested under the pre-registered rules</p>
      <p class="stepnote"><span class="stepchip" data-step="train">Step 3</span><span class="stepchip" data-step="band">Step 4</span><span class="stepchip" data-step="test">Step 5</span>
        Whether the band's caution can change the trained actor enough to change its score on the test.</p>
""")
# the exploratory results: a heading at the same level, plain step lines, and the data fact stated exactly
brep("explore", """<h4 id="iw-explore">Exploratory results, outside the pre-registered claims</h4>""",
     """<h3 id="iw-explore">Exploratory results, outside the pre-registered claims</h3>""")
brep("explore", """<p class="stepnote"><span class="stepchip" data-step="band">Step 4</span> measured at
        <span class="stepchip in" data-step="test">Step 5</span>: whose error bar keeps its promise under the known
        shift?</p>""", """<p class="stepnote"><span class="stepchip" data-step="band">Step 4</span><span class="stepchip" data-step="test">Step 5</span>
        The band, checked on the shifted test: whose error bar keeps its promise under the known shift?</p>""")
brep("explore", """<p class="stepnote"><span class="stepchip" data-step="band">Step 4</span> used in training, then
        <span class="stepchip in" data-step="test">Step 5</span> on the
        <span class="km" data-sit="strong_current">strong test</span>: does the better-calibrated band become a better
        controller?</p>""", """<p class="stepnote"><span class="stepchip" data-step="band">Step 4</span><span class="stepchip" data-step="test">Step 5</span>
        The band used in training, and the trained actor tested on the <span class="km" data-sit="strong_current">strong
        test</span>: does the better-calibrated band become a better controller?</p>""")
brep("explore", """<p class="stepnote"><span class="stepchip" data-step="collect">Step 1</span> What the recording itself contains,
        before <span class="stepchip in" data-step="remove" data-k="hold">Step 2</span> takes out the 80% hold.</p>""",
     """<p class="stepnote"><span class="stepchip" data-step="collect">Step 1</span><span class="stepchip" data-step="remove" data-k="hold">Step 2</span>
        What the recording itself contains, before this test's removal takes out the 80% hold.</p>""")
brep("explore", """So the weighting test's
        removal takes away the 80% hold, the best move in hard water, but not the evidence of the
        <span class="km" data-k="stall">stall</span>: these data never contained it.</p>""",
     """So the weighting test's
        removal takes away the 80% hold, the best move in hard water, but it is not what hides the
        <span class="km" data-k="stall">stall</span>: past 80% the expert data hold nothing, and the other four qualities
        hold only sparse, uniform random commands (3% to 10% of the steps), too little for the critics to learn the stall,
        with or without the hold.</p>""")
brep("explore", "<dd><b>The capped pull itself costs return below the top.</b>",
     "<dd><b>The capped pull itself costs return on the three lower-quality datasets (random, medium-replay, medium).</b>")
brep("explore", "Below the top the\n          direction held but not the size.", "On the three lower qualities the\n          direction held but not the size.")
# what the results mean, after the findings: the data fact and the measured size of BCA's penalty, stated exactly
brep("results", """Past 80% these data show nothing about the stall (<a href="#iw-past80">exploratory result C</a>),
        and there the penalty that BCA's band gives is far too small, 1/30 to 1/160 of the penalty the exact error would
        give, and barely follows that error""",
     """Past 80% these data hold only sparse random commands, and the expert data none
        (<a href="#iw-past80">exploratory result C</a>), so the critics still guess there; and the penalty that BCA's band
        gives, measured before training at the actor's own commands on the training rows, is far too small, about 1/30 to
        1/160 of the penalty the exact error would give, and barely follows that error""")
new_W = (head + B["intro"] + B["claims"] + B["shift"] + B["hold"] + B["why"] + B["howto"] + B["qual"] + B["findings"]
         + B["explore"] + B["results"] + B["ran"] + B["marks"] + tail)
assert new_W.count('<div class="finding"') == W.count('<div class="finding"')
assert new_W.count("<table") == W.count("<table") and new_W.count("<svg") == W.count("<svg")
splice(wa, wb, new_W)
# the test settings' names in the weighting test's script
# (the bar chart keeps the short names No shift, Mild and Strong under its bars; the legend line uses the full names)
rep("script", """$("iw-lg-test-t").textContent = key ? `Test (${NAME[key].toLowerCase()}, ξ = ${l > 0 ? "+" : ""}${sgn(l, 0)})`""",
    """$("iw-lg-test-t").textContent = key ? `Test (${({ none: "training mix, no shift", mild: "mild test mix", strong: "strong test mix" })[key]}, ξ = ${l > 0 ? "+" : ""}${sgn(l, 0)})`""")

# ===================================================================================================== 16. one name per critic, page-wide
resub("all", r"full-log(\s+)critic", r"full-recording\1critic", 31)
rep("all", "(the critic trained on the full log, with nothing removed)", "(the critic trained on the full recording, with nothing removed)")
resub("all", r"the critic trained on the full(\s+)log", r"the full-recording critic", 7)
resub("all", r"the critic of the full(\s+)log", r"the full-recording critic", 2)
resub("all", r"(?<!\()[Tt]he critic trained on the full(\s+)recording", lambda m: ("The" if m.group(0)[0] == "T" else "the") + " full-recording critic", 1)
resub("all", r"the(\s+)critic after the(\s+)removal", r"the\1critic trained after the\2removal", 6)
assert "full-log critic" not in s and "critic after the removal" not in s.replace("critic trained after the removal", "")

# ===================================================================================================== 17. column descriptions: "bin" and "range" for histogram rows
m = re.search(r"const COLS = (\{.*?\});\n", s, flags=re.S)
COLS_SRC = m.group(1)
COLS = json.loads(COLS_SRC)
assert json.dumps(COLS, ensure_ascii=False, separators=(",", ":")) == COLS_SRC, "COLS does not round-trip"
BANDWORD = re.compile(r"\bband(s?)\b")
CHANGED_COLS = []


def unband(cols, word):
    for k in list(cols):
        v = cols[k]
        if isinstance(v, str) and BANDWORD.search(v):
            assert "BCA" not in v and "Bayesian" not in v, (k, v)
            cols[k] = BANDWORD.sub(lambda m: word + m.group(1), v)
            CHANGED_COLS.append((k, cols[k]))


t0 = COLS["task"][0]["cols"]
assert t0["Water's push c + w₁"].startswith("Each row is one band of the water's push")
unband(t0, "bin")
aw = [e for e in COLS["action-space"] if "Throttle band" in e["cols"]]
assert len(aw) == 1
aw[0]["cols"] = {("Throttle range" if k == "Throttle band" else k): v for k, v in aw[0]["cols"].items()}
aw[0]["cols"]["Throttle range"] = aw[0]["cols"]["Throttle range"].replace(
    "Each row is one band of a thruster's throttle, 20% wide;", "Each row is one 20%-wide range of a thruster's throttle;")
unband(aw[0]["cols"], "range")
st = [e for e in COLS["description"] if "Size of the throttle" in e["cols"]]
assert len(st) == 1
unband(st[0]["cols"], "range")
for k, v in CHANGED_COLS:
    print("  COLS:", k, "->", v[:150])
# the weighting test's test-setting table: its rows are now named training mix, mild test mix and strong test mix
ts = [e for e in COLS["weighting-test"] if "Test setting" in e["cols"]]
assert len(ts) == 1
splice(m.start(1), m.end(1), json.dumps(COLS, ensure_ascii=False, separators=(",", ":")))

# ===================================================================================================== 18. colour tokens and style
# the learned blue, re-stepped so that it passes the validator against all seven other colours (both modes)
rep("style", "--k-learn: #2346b8;", "--k-learn: #115096;")
rep("style", "--k-learn: #758ae0;", "--k-learn: #5957fe;", count=2)
# the water's ink as text: at least 4.5:1 on white and on the water tint in light
rep("style", "--water-ink: #4f7d96;", "--water-ink: #44708a;")
css = open(CSS, encoding="utf-8").read().rstrip("\n")
e4 = open(E4_CSS, encoding="utf-8").read().rstrip("\n")
anchor = e4 + "\n</style>\n"
rep("all", anchor, anchor + "<style>\n" + css + "\n</style>\n")

# ===================================================================================================== checks
QV_AFTER = [m.group(0) for m in QV_LOCKED.finditer(s)]
assert QV_AFTER == QV_BEFORE, "a locked qviewer block changed"
for p in ["thrusterchain.p3.html", "thrusterchain.v11.html"]:
    assert os.path.exists(os.path.join(PAGE_DIR, p))
assert s.count('id="critic-in-action"') == 1 and s.count('id="qv-data"') == 1
assert "gapped critic" not in s[:s.index("</main>")]
print(f"{N['n']} replacements; {len(ORIG):,} -> {len(s):,} characters")
out = TRIAL if CHECK else PAGE
os.makedirs(os.path.dirname(TRIAL), exist_ok=True)
open(TRIAL, "w", encoding="utf-8", newline="").write(s)
if not CHECK:
    open(PAGE, "w", encoding="utf-8", newline="").write(s)
print("wrote", out, hashlib.md5(s.encode("utf-8")).hexdigest())

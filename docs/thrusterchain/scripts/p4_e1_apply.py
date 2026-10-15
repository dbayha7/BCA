#!/usr/bin/env python3
"""Page pass 4, edit 1 of 4: the design applied to the whole page.

- the colour system's tokens and the readable text colours, in the three :root blocks of the main style;
- the components (scripts/p4_e1.css) in a new <style> after the main one: justified running text with lifted caps,
  aligned symbol legends, coloured option buttons and drawing frame, table group rows, chart re-points, the strip;
- lang="en" on .wrap and on the walkthrough dialog;
- #glance: the old at-a-glance block and both status paragraphs become one sentence, the five-step offline
  reinforcement-learning strip and a three-line status (draft A, facts checked against notes/p4_facts.md and
  iw_explore/RESULTS.md);
- the live panel: a compact strip with its note, one line per option on the pipeline step it stands for (and a
  sentence in each option's tooltip), the regrouped "Reading the drawing" legend with colour keys, colour marks in
  the lead-in, and the drawing script's full-throttle cues in the region's gold (red stays the stall);
- charts: the 80% lines get k80 (stall red), the inline legend swatches follow the re-pointed classes; the weighting
  test's exactly weighted band gets its own token, its mild test goes neutral, and its 80% hold is drawn as a removed
  (hollow) mark.

Every replacement is exact and counted. Run from the page directory (WSL):
    python3 scripts/p4_e1_apply.py [--check]      (--check: build in memory, write nothing)
"""
import hashlib
import html
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
PAGE_DIR = os.path.dirname(HERE)
PAGE = os.path.join(PAGE_DIR, "thrusterchain.html")
BEFORE = os.path.join(PAGE_DIR, "tmp", "p4", "edit1.before.html")
CSS = os.path.join(HERE, "p4_e1.css")
CHECK = "--check" in sys.argv

s = open(PAGE, encoding="utf-8").read()
ORIG = s
assert hashlib.md5(open(BEFORE, "rb").read()).hexdigest() == "f3736a56dc7698e0b57633acbb91de3c", "backup changed"
assert hashlib.md5(s.encode("utf-8")).hexdigest() == "f3736a56dc7698e0b57633acbb91de3c", "the page is not the version-12 working copy this edit was written for"
QV_BLOCKS = [m.group(0) for m in re.finditer(r"<!-- qviewer:begin (\w+) -->.*?<!-- qviewer:end \1 -->|/\* qviewer:begin style \*/.*?/\* qviewer:end style \*/|<script type=\"application/json\" id=\"qv-data\">.*?</script>", s, flags=re.S)]
N = {"n": 0}


def rep(old, new, count=1):
    global s
    n = s.count(old)
    assert n == count, f"expected {count} of {old[:90]!r}, found {n}"
    s = s.replace(old, new)
    N["n"] += 1


def esc(text):
    """attribute text, escaped as the page does (apostrophes as &#x27;)"""
    return html.escape(text, quote=True)


def tip(title, contains=None):
    """the page's own data-tip-title + data-tip attribute pair for a term, reused verbatim"""
    t = esc(title)
    texts = re.findall(r'data-tip-title="%s" data-tip="([^"]*)"' % re.escape(t), ORIG)
    if contains:
        texts = [x for x in texts if contains in x]
    assert texts, f"no tooltip titled {title!r}" + (f" containing {contains!r}" if contains else "")
    return f'data-tip-title="{t}" data-tip="{texts[0]}"'


def newtip(title, text):
    assert text.endswith(".") and text[0].isupper(), text
    return f'data-tip-title="{esc(title)}" data-tip="{esc(text)}"'


# =================================================================================================================
# 1. Tokens: the colour system and the readable text colours, in the main style's three :root blocks
# =================================================================================================================
rep("  --ink-2: #434c48;\n  --muted: #69726e;\n", "  --ink-2: #2b3330;\n  --muted: #4f5853;\n")
rep("  --warn: #b33f29;\n  --warn-soft: #f6e2dc;\n", "  --warn: #9f2818;\n  --warn-soft: #f3e3e1;\n")
rep("  --ok: #23794c;\n", "  --ok: #217447;\n")   # pass labels: at least 4.5:1 on every surface and tint (4.32 before)
rep("  --region: #f3e7c9;\n  --c-rec: #b07a12; --c-sa: #1f7a52; --c-rt: #6a4cc4; --c-none: #8a928e;\n",
    """  --region: #f0eace;
  /* the colour system (page pass 4; validated in notes/p4_design.md). One meaning per colour, page-wide; a colour is
     never the only cue, a label always goes with it. The four test situations: */
  --sit-rec: #00714f;          /* recorded conditions (in distribution); also the recorded data and the recording controller */
  --sit-strong: #d56c00;       /* strong current (dynamics extrapolation) */
  --sit-waves: #0092c1;        /* big waves (disturbance shift) */
  --sit-kicks: #ce4c8e;        /* big starting pushes (initial-state shift) */
  --sit-rec-soft: #e0eeea; --sit-strong-soft: #faede0; --sit-waves-soft: #e0f2f8; --sit-kicks-soft: #f9eaf1;
  --k-stall: #9f2818;          /* the stall and the stalling world, and nothing else (text in it: --warn) */
  --k-stall-soft: #f3e3e1;
  --k-rise: #7e53cc;           /* push keeps rising: the no-stall comparison world and the rising-thrust controller */
  --k-rise-soft: #f0eaf9;
  --k-region: #c6b042;         /* the full-throttle region and its commands: an area colour, always outlined in its ink */
  --k-region-ink: #735d10;     /* its outlines, thresholds and text */
  --k-region-soft: #f0eace;
  --k-learn: #2346b8;          /* the learned parts: critic, actor, BCA's band (the page's accent) */
  --k-learn-soft: #e3e8f7;
  --k-ref: #2b3330;            /* the stall-aware controller, the 100-point reference, in ink */
  --k-none: #8a928e;           /* no throttle, dashed */
  --hatch: repeating-linear-gradient(135deg, var(--ink-2) 0 1.4px, transparent 1.4px 5px);   /* data removed */
  --step-collect: var(--sit-rec); --step-remove: var(--k-region); --step-train: var(--k-learn); --step-band: var(--k-learn);
  --c-rec: var(--sit-rec); --c-sa: var(--k-ref); --c-rt: var(--k-rise); --c-none: var(--k-none);
""")
DARK_OLD = """--bg: #111412; --surface: #191d1b; --surface-2: #222724; --ink: #e7ebe8; --ink-2: #c2c9c5; --muted: #949d98;
    --rule: #2e3431; --rule-strong: #4a524e; --accent: #8199ec; --accent-ink: #aabcf3; --accent-soft: #1e2640;
    --warn: #ec8b74; --warn-soft: #3a201a; --ok: #6cc795; --ok-soft: #163024; --water: #16232a; --water-ink: #7fb0c9;
    --region: #2e2918; --c-rec: #bf8a2a; --c-sa: #36a06d; --c-rt: #8a7cec; --c-none: #8e9793;"""
DARK_NEW = """--bg: #111412; --surface: #191d1b; --surface-2: #222724; --ink: #e7ebe8; --ink-2: #d5dbd7; --muted: #a9b2ad;
    --rule: #2e3431; --rule-strong: #4a524e; --accent: #8199ec; --accent-ink: #aabcf3; --accent-soft: #1e2640;
    --warn: #ec8b84; --warn-soft: #442f2f; --ok: #6cc795; --ok-soft: #163024; --water: #16232a; --water-ink: #7fb0c9;
    --region: #3c3915;
    --sit-rec: #10a07b; --sit-strong: #a95507; --sit-waves: #1176a2; --sit-kicks: #b1397c;
    --sit-rec-soft: #17372e; --sit-strong-soft: #362817; --sit-waves-soft: #172f36; --sit-kicks-soft: #37232e;
    --k-stall: #dc6e76; --k-stall-soft: #442f2f; --k-rise: #b06bd8; --k-rise-soft: #372d41;
    --k-region: #ac9202; --k-region-ink: #c9b23a; --k-region-soft: #3c3915;
    --k-learn: #758ae0; --k-learn-soft: #1e2640; --k-ref: #d5dbd7; --k-none: #8e9793;"""
rep("    " + DARK_OLD, "    " + DARK_NEW)                                          # inside @media (prefers-color-scheme: dark)
rep("  " + DARK_OLD.replace("\n    ", "\n  "), "  " + DARK_NEW.replace("\n    ", "\n  "))  # :root[data-theme="dark"]
# the weighting test's series join the system: training and calibration data are recorded data, the strong test is
# strong current, the mild test is neutral, and the exactly weighted band (until now drawn with the mild test's
# colour) is BCA's, a learned part
rep("""/* the weighting test (built on this page's tokens; three series colours for training and the two test settings) */
:root { --iw-train: #2a78d6; --iw-strong: #eb6834; --iw-mild: #1baf7a; }
@media (prefers-color-scheme: dark) { :root:not([data-theme="light"]) { --iw-train: #3987e5; --iw-strong: #d95926; --iw-mild: #199e70; } }
:root[data-theme="dark"] { --iw-train: #3987e5; --iw-strong: #d95926; --iw-mild: #199e70; }
""", """/* the weighting test, in the page's colour system: training and calibration data are recorded data, the strong test is
   strong current, the mild test is neutral, and the exactly weighted band is BCA's (a learned part). Each token points
   at a theme token, so it follows both dark-mode blocks. */
:root { --iw-train: var(--sit-rec); --iw-strong: var(--sit-strong); --iw-mild: var(--ink-2); --iw-weighted: var(--k-learn); }
""")

# =================================================================================================================
# 2. The components, in a new style block after the main one; lang for hyphenation
# =================================================================================================================
css = open(CSS, encoding="utf-8").read()
assert "</style" not in css
i = s.index("</style>")
s = s[:i + len("</style>")] + "\n<style>\n" + css + "</style>" + s[i + len("</style>"):]
rep('<div class="wrap">', '<div class="wrap" lang="en">')
rep('<dialog class="wt" id="wt"', '<dialog class="wt" lang="en" id="wt"')

# =================================================================================================================
# 3. #glance: one sentence, the pipeline strip, a three-line status
# =================================================================================================================
T_HOLD = newtip("80% hold", "The 80% hold is thruster 1 at exactly 80% upstream, the hardest push it can give without "
                "stalling. The weighting test's own recording controller gives it often when the water pushes hard, so "
                "it is the densest cluster of that test's commands, and the densest-15% removal deletes almost all of it.")
T_BAND = newtip("BCA's band", "BCA's band is an interval around the critic's prediction, sized on recorded rows kept out "
                "of training so that the critic's one-step error falls inside it on 90% of them. A wide band marks a "
                "place where the critic may be wrong.")
T_ERR = newtip("One-step error", "The critic's one-step error is how far its prediction for a recorded row misses its "
               "own one-step target: the row's reward plus 0.98 times the critic's value of the next state, minus its "
               "value of this state.")
T_PULL = newtip("Capped pull", "The capped pull is BCA's own use of its band: at a training row where the band is wide, "
                "the pull toward the recorded command is made stronger by a factor from 1 up to, but not reaching, 1.5. "
                "It constrains behaviour, like TD3+BC's own pull; it is not a penalty on values.")
T_EDGE = newtip("Lower edge", "The lower edge is the critic's value minus the band's width. Used in training, it lowers "
                "the critic's value of the actor's own command where the band there is wider than at the recorded "
                "command, which is the kind of caution that value-pessimism methods apply.")
T_IN = newtip("In distribution", "In distribution means inside the conditions the training data were recorded in. Here "
              "that is the recorded conditions: a current of 0.1 to 0.7, a wave amplitude of 0.1 to 0.4 and a starting "
              "spread of 0.5.")
T_OUT = newtip("Out of distribution", "Out of distribution means outside the conditions the training data were recorded "
               "in. Every state of strong current and of big waves is out of distribution, and part of the states of "
               "big starting pushes.")

GLANCE = f'''    <p class="g-one">ThrusterChain is a small simulated task, built so that every true value in it can be computed
      exactly: four thrusters hold five floats at rest against moving water, and a propeller pushed past 80% throttle
      <span class="term km" data-k="stall" {tip("The stall", "0.65 at 90%")}>stalls</span>, pushing less while it
      costs more. It tests whether Bayesian Conformal Aggregation (BCA) helps a controller learned by
      <i {tip("Offline reinforcement learning")}>offline reinforcement learning</i>, that is, learned only from a fixed
      recording, when that recording lacks the commands that matter.</p>
    <p class="pipe-h" id="pipe-h">The process, step by step, as it runs in ThrusterChain. Each step, each test situation
      and the stall keep one colour on the whole page, and a label always goes with the colour.</p>
    <nav class="pipe" aria-labelledby="pipe-h">
      <ol>
        <li class="pipe-step" data-step="collect"><div class="pipe-card">
          <p class="pipe-head"><span class="pipe-n">1</span><span class="pipe-t">Collect the data</span></p>
          <p class="pipe-d">The <i {tip("Recording controller", "made the training data")}>recording controller</i>, a
            fixed controller, drives 100-step <i {tip("Episode")}>episodes</i> in the
            <span class="km" data-sit="recorded">recorded conditions</span> of the stalling world. Each step becomes
            one <i {tip("Row")}>row</i> of the data.</p>
          <p class="pipe-why"><b>Why:</b> its most common command, the
            <span class="term km" data-k="region" {tip("Full-throttle command")}>full-throttle command</span> (13.42% of
            training rows), puts thruster 1 past 80% <i {tip("Throttle")}>throttle</i>, so the full data show the
            stall.</p>
          <a class="pipe-go" href="#dataset">The recorded data</a></div></li>
        <li class="pipe-step" data-step="remove"><div class="pipe-card">
          <p class="pipe-head"><span class="pipe-n">2</span><span class="pipe-t">Remove commands</span></p>
          <p class="pipe-d"><span class="pipe-lab"><i class="key removed" data-k="region"></i>Main test:</span> the
            <i {tip("Densest-15% removal")}>densest-15% removal</i> deletes the recording controller's full-throttle
            commands, 99.11% of them.</p>
          <p class="pipe-d"><span class="pipe-lab"><i class="key removed" data-k="hold"></i>Weighting test:</span> the
            same rule, on that test's own data, deletes the <i {T_HOLD}>80% hold</i>.</p>
          <p class="pipe-why"><b>Why:</b> afterwards only 60 of the main test's 42,500 kept training rows go past 80%,
            so the data no longer show the stall. This is the
            <i {tip("Distribution shift")}>distribution shift</i>, built on purpose.</p>
          <a class="pipe-go" href="#shift">The distribution shift</a></div></li>
        <li class="pipe-step" data-step="train"><div class="pipe-card">
          <p class="pipe-head"><span class="pipe-n">3</span><span class="pipe-t">Train TD3+BC</span></p>
          <p class="pipe-d">On the rows that are left,
            <span class="term" {tip("TD3+BC", "standard learner for recorded data: twin")}>TD3+BC</span> trains a
            <i {tip("Critic")}>critic</i>, which scores every command in every state, and with it an
            <i {tip("Actor")}>actor</i>, the learned controller, which picks commands its critic scores highly while a
            pull keeps it near the recorded ones.</p>
          <p class="pipe-why"><b>Why:</b> the critic learns only where the data hold commands, so past 80% it can only
            guess, and a guess that is too high draws the actor into the stall.</p>
          <a class="pipe-go" href="#methods">Methods</a></div></li>
        <li class="pipe-step" data-step="band"><div class="pipe-card">
          <p class="pipe-head"><span class="pipe-n">4</span><span class="pipe-t">Add BCA's band</span></p>
          <p class="pipe-d">BCA puts a <i {T_BAND}>band</i> around the critic's <i {T_ERR}>one-step error</i>. Where the
            band is wide it adds caution: a harder pull toward the recorded command (its
            <i {T_PULL}>capped pull</i>), or a lower value for the actor's own command (its
            <i {T_EDGE}>lower edge</i>).</p>
          <p class="pipe-why"><b>Why:</b> caution should go where the critic may be wrong, and in ThrusterChain the
            critic is wrong past 80%.</p>
          <a class="pipe-go" href="#bca-test">The BCA test</a></div></li>
        <li class="pipe-step" data-step="test"><div class="pipe-card">
          <p class="pipe-head"><span class="pipe-n">5</span><span class="pipe-t">Test the actor</span></p>
          <p class="pipe-d">Every actor is scored exactly on each situation's 1,000 fixed starts: in the
            <span class="km" data-sit="recorded">recorded conditions</span>, which are <i {T_IN}>in distribution</i>,
            and in three <i {tip("Shifted situations")}>shifted situations</i>, which are
            <i {T_OUT}>out of distribution</i>: <span class="km" data-sit="strong_current">strong current</span>,
            <span class="km" data-sit="big_waves">big waves</span> and
            <span class="km" data-sit="big_kicks">big starting pushes</span>.</p>
          <p class="pipe-why"><b>Why:</b> the missing commands matter most where holding the floats needs 80% or more,
            and strong current asks for that in most states.</p>
          <a class="pipe-go" href="#starting-state">Starting State</a></div></li>
      </ol>
    </nav>
    <ul class="g-status" aria-label="Where the work stands">
      <li><span class="tag">Main test: stopped before BCA</span> The removal caused real harm. In
        <span class="km" data-sit="strong_current">strong current</span>, an actor trained against the critic that never
        saw the stall scored 95 to 103 <i {tip("Points", "page&#x27;s score scale")}>points</i> below one trained
        against the critic of the full recording (0 points is the recording controller, 100 the stall-aware
        controller). But <i {tip("Caution")}>caution</i> of BCA's kind, a harder pull toward the recorded commands at
        BCA's strengths, could not win that back even when it was placed by the critic's exact error. So the
        <i {tip("Harm checks")}>harm checks</i> stopped the test before any BCA run (<a href="#results">Results</a>).
        Their rules were pre-registered: written down before any result existed and fixed by a hash, a short
        fingerprint computed from a file that changes if any character of the file changes. Since then, with no BCA
        run, a penalty on the critic's
        value of the actor's own command, set by how far the critic's exact one-step error there exceeds its error at
        the recorded command, won back 93.9 of the 98.9 points lost in strong current
        (<a href="#beyond-the-data">After the stop</a>).</li>
      <li><span class="tag ok">Weighting test: pre-registered</span> The weighting test, the main line of work since
        the stop, has its own data, recorded by a second recording controller at five levels of quality, and every value
        in it is again known exactly. On all five, BCA's mechanics worked as designed, and
        <i {tip("Importance weighting")}>importance weighting</i> of its calibration rows toward the known test mix of
        currents restored the band's <i {tip("Coverage")}>coverage</i> under the shift: from 83.3–86.7% unweighted to
        91.0–94.6% weighted, all inside the required 89–96% (<a href="#weighting-test">The weighting test</a>).</li>
      <li><span class="tag">Weighting test: exploratory</span> Weighting did not raise the return. In exploratory
        training runs of BCA, outside the pre-registered claims, weighting changed the
        <i {tip("Return")}>return</i> by −26.4 to +2.7, and on no quality was a gain both as large as the 1.8 fixed in
        advance and clear of the spread between repeat runs. At the commands the data never contain, BCA's band is far
        too small: its penalty is about 1/30 to 1/160 of one built from the exact one-step error, and it barely moves
        with that error (row-by-row correlation −0.17 to 0.29).</li>
    </ul>
'''
a = s.index('    <div class="glance">')
assert s.count('<div class="glance">') == 1
b = s.index('\n    </div>\n  </header>', a)
old_glance = s[a:b]
assert old_glance.count('<p class="status">') == 2 and old_glance.rstrip().endswith("</p>"), old_glance[-200:]
s = s[:a] + GLANCE.rstrip("\n") + s[b:]
N["n"] += 1

# =================================================================================================================
# 4. The live panel
# =================================================================================================================
# 4a. the compact strip and its note, after the opening sentence
SMALL = '''
        <div class="pipe-where">
          <nav class="pipe pipe-sm" aria-label="Where this panel sits in the offline reinforcement-learning process">
            <ol>
              <li class="pipe-step is-here" data-step="collect"><a href="#dataset"><span class="pipe-n">1</span>Collect</a></li>
              <li class="pipe-step" data-step="remove"><a href="#shift"><span class="pipe-n">2</span>Remove</a></li>
              <li class="pipe-step" data-step="train"><a href="#methods"><span class="pipe-n">3</span>Train</a></li>
              <li class="pipe-step" data-step="band"><a href="#bca-test"><span class="pipe-n">4</span>Band</a></li>
              <li class="pipe-step is-here" data-step="test"><a href="#starting-state"><span class="pipe-n">5</span>Test</a></li>
            </ol>
          </nav>
          <p class="pipe-note">This panel replays the environment itself, with fixed controllers and no learning, so it
            shows steps 1 and 5 of the process (outlined above): the recording controller is the one that made the
            data, and the four situations are the ones every learned actor is tested in. Under each option, a line
            says which step it stands for, and why.</p>
        </div>'''
rep("through the episode's 100 states one at a time.</p>", "through the episode's 100 states one at a time.</p>" + SMALL)

# 4b. one line per option on its pipeline step, and the same in its tooltip
CHIP = {1: '<span class="stepchip" data-step="collect">Step 1</span>', 2: '<span class="stepchip" data-step="remove">Step 2</span>',
        3: '<span class="stepchip" data-step="train">Step 3</span>', 4: '<span class="stepchip" data-step="band">Step 4</span>',
        5: '<span class="stepchip" data-step="test">Step 5</span>', 0: '<span class="stepchip" data-step="none">No step</span>'}
OPTS = {
    "opt-sit-recorded": ((1, 5), "Where the data were collected, and the in-distribution test, where a learned actor must not lose.",
                         "In the process it is step 1, where the data were collected, and the in-distribution part of step 5, the test."),
    "opt-sit-strong_current": ((5,), "The main test, a dynamics extrapolation: the water itself goes beyond anything recorded, and holding asks thruster 1 for 80% or more in most states.",
                               "In the process it is part of step 5, the test: a dynamics extrapolation, because the current goes beyond any recorded one."),
    "opt-sit-big_waves": ((5,), "A disturbance shift: the wave's pushes are bigger than any recorded, and the water enters the full-throttle region on 39.6% of states, against 27.9% in the recorded conditions.",
                          "In the process it is part of step 5, the test: a disturbance shift, because the wave's pushes are larger than any recorded."),
    "opt-sit-big_kicks": ((5,), "An initial-state shift: the floats start three times as far from rest as recorded. It is the hardest situation by raw return, but not where the missing commands cost the most.",
                          "In the process it is part of step 5, the test: an initial-state shift, because the starts lie far outside the recorded starts."),
    "opt-world-stall": ((1,), "The world the data were recorded in. Its stall is what the removal in step 2 hides from the learner.",
                        "In the process it is the world of step 1, where the data were recorded, and its stall is what the removal of step 2 hides."),
    "opt-world-no_stall": ((2,), "A control: the same removal hides nothing here, which shows what caution costs when the missing commands are harmless.",
                           "In the process it is a control for step 2: the same removal, but here it hides nothing."),
    "opt-ctrl-recording": ((1,), "It made the data; its full-throttle commands are what the main test's removal deletes in step 2.",
                           "Step 1. Its log is the training data, and its full-throttle commands are what the main test&#x27;s removal deletes in step 2."),
    "opt-ctrl-stall_aware": ((5,), "The yardstick of the test, 100 points: it stops at 80%, which is what a good actor should learn.",
                             "It is the 100-point yardstick of step 5, the test."),
    "opt-ctrl-rising_thrust": ((3,), "Step 3 gone wrong: it acts as an actor would whose critic never saw the stall.",
                               "It shows step 3 gone wrong: an actor whose critic never saw the stall acts like this."),
    "opt-ctrl-no_throttle": ((0,), "The water acting alone, to show what the thrusters must undo.",
                             "It belongs to no step of the process; it shows what the water does alone."),
}
for oid, (steps, line, tipadd) in OPTS.items():
    m = re.search(r'<button type="button" class="opt" id="%s"[^>]*>.*?</button>' % re.escape(oid), s, flags=re.S)
    assert m and s.count(f'id="{oid}"') == 1, oid
    btn = m.group(0)
    chips = "".join(CHIP[k] for k in steps)
    new = btn[:-len("</button>")] + f'<span class="opt-step">{chips}{html.escape(line, quote=False)}</span></button>'
    if oid.startswith("opt-ctrl-"):
        tid = "tip-ctrl-" + oid[len("opt-ctrl-"):]
        tm = re.search(r'<template id="%s">.*?</template>' % re.escape(tid), s, flags=re.S)
        assert tm and s.count(f'"{tid}"') == 2, tid
        tpl = tm.group(0)
        s = s.replace(tpl, tpl[:-len("</template>")] + f' <p><b>In the process:</b> {tipadd}</p></template>')
    else:
        tm = re.match(r'(<button[^>]* data-tip=")([^"]*)(")', btn)
        assert tm, oid
        new = new.replace(tm.group(0), tm.group(1) + tm.group(2) + " " + esc(tipadd) + tm.group(3), 1)
    assert s.count(btn) == 1
    s = s.replace(btn, new)
    N["n"] += 1

# 4c. colour marks in the panel's lead-in, and the readouts' note
rep('recording controller gives the <span class="term" data-tip-title="Full-throttle command"',
    'recording controller gives the <span class="term km" data-k="region" data-tip-title="Full-throttle command"')
rep('Try strong current with\n          the <span class="term" data-tip-title="Rising-thrust controller"',
    'Try <span class="km" data-sit="strong_current">strong current</span> with\n          the <span class="term km" data-k="rising_thrust" data-tip-title="Rising-thrust controller"')
rep(', then switch to the <span class="term" data-tip-title="No-stall comparison world"',
    ', then switch to the <span class="term km" data-world="no_stall" data-tip-title="No-stall comparison world"')
rep("its gauge turns red and reads STALLED", 'its gauge <span class="km" data-k="stall">turns red</span> and reads STALLED')
rep('the state is in the <i data-tip-title="Full-throttle region"', 'the state is in the <i class="km" data-k="region" data-tip-title="Full-throttle region"')
rep("Above 0.565 the state is in the full-throttle region (shown in red), where",
    "Above 0.565 the state is in the full-throttle region (its number then turns gold, the region&#x27;s colour on this page), where")
rep('Hover over or tap a part for its numbers." role="img" aria-label="The channel:',
    'Hover over or tap a part for its numbers. The frame around the drawing has the colour of the chosen situation, and the top bar and its number turn gold when the state is in the full-throttle region." role="img" aria-label="The channel:')

# 4d. "Reading the drawing", regrouped by part of the drawing, with keys that match it
DRAWKEY = '''<div class="symkey" id="sim-drawkey"><p class="symkey-h">Reading the drawing</p>
      <p class="symkey-g">The water bar at the top</p>
      <dl><div><dt>Top bar</dt><dd>It shows the water's push on every float, <i>c</i> + <i>w</i><sub>1</sub>: the current <i>c</i> plus the push <i>w</i><sub>1</sub> that the wave adds in this state.</dd></div>
      <div><dt><i class="key tile" data-k="region"></i>Full-throttle region</dt><dd>The tinted end of the bar, past the line at 0.565, is the full-throttle region. When the state is in it, the bar and its number turn gold, and the recording controller gives the full-throttle command in about half of such states, by a coin flip.</dd></div></dl>
      <p class="symkey-g">The channel and the floats</p>
      <dl><div><dt><i class="key sitall"></i>Frame</dt><dd>The frame around the drawing has the colour of the situation you picked: <span class="km" data-sit="recorded">recorded conditions</span>, <span class="km" data-sit="strong_current">strong current</span>, <span class="km" data-sit="big_waves">big waves</span> or <span class="km" data-sit="big_kicks">big starting pushes</span>.</dd></div>
      <div><dt>Dashed line</dt><dd>It marks rest. A float is drawn below it when its position number is negative, which is where the water carries it.</dd></div>
      <div><dt>Number above a float</dt><dd>It is the float's position number <i>p</i>, one of its two numbers, which are both 0 at rest. An arrow after it (↓ or ↑) means the float is past the edge of the drawing in that direction.</dd></div>
      <div><dt>Small arrow by a float</dt><dd>It shows the float's speed number <i>v</i>, the other coordinate of its swing.</dd></div>
      <div><dt>Words at the bottom left</dt><dd>With the recording controller, they say which of its two behaviours gave this state's command: normal driving, or the full-throttle command, written in gold.</dd></div></dl>
      <p class="symkey-g">The thrusters and their gauges</p>
      <dl><div><dt>Arrow at a thruster</dt><dd>It shows the push the thruster really gives, <i>f</i>(<i>a</i>); pointing up is upstream, against the water.</dd></div>
      <div><dt>Gauge bar</dt><dd>It shows the thruster's throttle <i>a</i>, from −1 (full throttle against the water, at the top) to 1.</dd></div>
      <div><dt>Dark tick</dt><dd>It shows the push <i>f</i>(<i>a</i>), which equals the throttle up to 80%.</dd></div>
      <div><dt><i class="key tile" data-k="stall"></i>Red gauge, STALLED</dt><dd>In the stalling world the dashed 80% marks are red, and so are the ends of each gauge beyond them. A throttle past them stalls: the gauge bar and the thruster turn red, the gauge reads STALLED, and the dark tick falls back.</dd></div>
      <div><dt>Numbers under a gauge</dt><dd>They are the throttle first, then the push.</dd></div></dl>
      <p class="symkey-g">The chart under the drawing</p>
      <dl><div><dt>Chart</dt><dd>It shows the cost so far, which is minus the return so far, for all four controllers from this start; the dashed vertical line marks the state on screen.</dd></div>
      <div><dt><i class="key line" data-k="recording"></i>Recording controller</dt><dd>Its line wears the colour of the recorded data, because it made them.</dd></div>
      <div><dt><i class="key line" data-k="stall_aware"></i>Stall-aware controller</dt><dd>Its line is drawn in ink, because it is the 100-point reference.</dd></div>
      <div><dt><i class="key line" data-k="rising_thrust"></i>Rising-thrust controller</dt><dd>Its line is violet, the colour of push that keeps rising, which is what it assumes; the no-stall comparison world has the same colour.</dd></div>
      <div><dt><i class="key dash" data-k="no_throttle"></i>No throttle</dt><dd>Its dashed line is the water acting alone, with every thruster off.</dd></div></dl></div>'''
a = s.index('<div class="symkey"><p class="symkey-h">Reading the drawing</p>')
assert s.count('<p class="symkey-h">Reading the drawing</p>') == 1
b = s.index('</dl></div>', a) + len('</dl></div>')
assert s[a:b].count("<dt>") == 9, s[a:b].count("<dt>")
s = s[:a] + DRAWKEY + s[b:]
N["n"] += 1

# 4e. the drawing script: the full-throttle cues in the region's gold, the 80% marks red only in the stalling world,
#     and an off-edge float number in ink (red is kept for the stall)
rep('water = tok("--water"), waterInk = tok("--water-ink"), region = tok("--region"), body = tok("--body"), mono = tok("--mono");',
    'water = tok("--water"), waterInk = tok("--water-ink"), region = tok("--region"), body = tok("--body"), mono = tok("--mono"),\n'
    '    regInk = tok("--k-region-ink"), stallC = tok("--k-stall");')
rep('g.textAlign = "right"; g.font = "600 13px " + mono; g.fillStyle = inReg ? warn : ink2;',
    'g.textAlign = "right"; g.font = "600 13px " + mono; g.fillStyle = inReg ? regInk : ink2;')
rep('  g.fillStyle = inReg ? warn : waterInk;\n', '  g.fillStyle = inReg ? regInk : waterInk;\n')
rep('  g.strokeStyle = ink; g.lineWidth = 1.5;\n  g.beginPath(); g.moveTo(PX(D.trigger), my - 4);',
    '  g.strokeStyle = regInk; g.lineWidth = 1.5;\n  g.beginPath(); g.moveTo(PX(D.trigger), my - 4);')
rep('if (narrow && inReg) { g.font = "600 11px " + mono; g.fillStyle = warn;',
    'if (narrow && inReg) { g.font = "600 11px " + mono; g.fillStyle = regInk;')
rep('g.font = (narrow ? "10.5px " : "11.5px ") + mono; g.fillStyle = off ? warn : muted;',
    'g.font = (narrow ? "10.5px " : "11.5px ") + mono; g.fillStyle = off ? ink : muted;')
rep('    g.strokeStyle = muted; g.setLineDash([3, 3]); g.lineWidth = 1;\n'
    '    [-KNEE, KNEE].forEach(v => { g.beginPath(); g.moveTo(gx - 13, GY(v)); g.lineTo(gx + 13, GY(v)); g.stroke(); });\n'
    '    g.setLineDash([]);\n',
    '    g.strokeStyle = stallWorld ? stallC : muted; g.setLineDash([3, 3]); g.lineWidth = 1;\n'
    '    [-KNEE, KNEE].forEach(v => { g.beginPath(); g.moveTo(gx - 13, GY(v)); g.lineTo(gx + 13, GY(v)); g.stroke(); });\n'
    '    g.setLineDash([]); g.strokeStyle = muted;\n')
rep('      g.font = "10.5px " + mono; g.fillStyle = muted; g.textAlign = "left";\n      g.fillText("80%", gx + 15, GY(-KNEE) + 4);',
    '      g.font = "10.5px " + mono; g.fillStyle = stallWorld ? warn : muted; g.textAlign = "left";\n      g.fillText("80%", gx + 15, GY(-KNEE) + 4);')
rep('g.font = "600 12px " + body; g.textAlign = "left"; g.fillStyle = R.ft[t] ? warn : muted;',
    'g.font = "600 12px " + body; g.textAlign = "left"; g.fillStyle = R.ft[t] ? regInk : muted;')
rep('$("roW").style.color = D.water[S.sit].reg[t] ? "var(--warn)" : "";',
    '$("roW").style.color = D.water[S.sit].reg[t] ? "var(--k-region-ink)" : "";')

# =================================================================================================================
# 5. Charts across the page: the 80% lines, and the inline legend swatches that follow the re-pointed classes
# =================================================================================================================
rep('<line class="knee" x1="331.6" x2="331.6" y1="18" y2="230"/>', '<line class="knee k80" x1="331.6" x2="331.6" y1="18" y2="230"/>')
rep('<line class="knee" x1="58" x2="426" y1="58.0" y2="58.0"/>', '<line class="knee k80" x1="58" x2="426" y1="58.0" y2="58.0"/>')
rep('<line class="knee" x1="62" x2="548" y1="202.8" y2="202.8"/>', '<line class="knee k80" x1="62" x2="548" y1="202.8" y2="202.8"/>')
rep('<i class="sw" style="background:var(--accent)"></i>push, no-stall comparison world',
    '<i class="sw" style="background:var(--k-rise)"></i>push, no-stall comparison world')
rep('<i class="sw" style="background:repeating-linear-gradient(90deg,var(--muted) 0 4px,transparent 4px 8px)"></i>80% line',
    '<i class="sw" style="background:repeating-linear-gradient(90deg,var(--k-stall) 0 4px,transparent 4px 8px)"></i>80% line')
rep('<i class="mk" style="background:var(--warn)"></i>full-throttle command (36 steps)',
    '<i class="mk" style="background:var(--k-region);box-shadow:inset 0 0 0 1.2px var(--k-region-ink)"></i>full-throttle command (36 steps)')
rep('<i class="mk" style="background:var(--accent)"></i>normal driving', '<i class="mk" style="background:var(--sit-rec)"></i>normal driving')
rep('<i class="mk" style="background:var(--region);border-radius:2px"></i>full-throttle region (67 steps)',
    '<i class="mk" style="background:var(--region);border-radius:2px;box-shadow:inset 0 0 0 1px var(--k-region-ink)"></i>full-throttle region (67 steps)')
rep('<i class="sw" style="background:repeating-linear-gradient(90deg,var(--muted) 0 4px,transparent 4px 8px)"></i>−80% throttle: below this line thruster 1 stalls',
    '<i class="sw" style="background:repeating-linear-gradient(90deg,var(--k-stall) 0 4px,transparent 4px 8px)"></i>−80% throttle: below this line thruster 1 stalls')
# the weighting test: the exactly weighted band gets its own token; the no-stall push is violet; the 80% hold, which
# the removal deletes, is a hollow mark like the deleted rows elsewhere
rep('<i class="sw" style="background:var(--iw-mild)"></i>Exactly weighted band', '<i class="sw" style="background:var(--iw-weighted)"></i>Exactly weighted band')
rep('line(svg, [[-2, 90], [3, 90]], X, Y, "stroke:var(--iw-mild)");', 'line(svg, [[-2, 90], [3, 90]], X, Y, "stroke:var(--iw-weighted)");')
rep('<i class="sw dash"></i>Push if there were no stall',
    '<i class="sw" style="background:repeating-linear-gradient(90deg,var(--k-rise) 0 4px,transparent 4px 7px)"></i>Push if there were no stall')
rep('line(svg, [[0.8, 0.8], [1, 1]], X, Y, "stroke:var(--muted);', 'line(svg, [[0.8, 0.8], [1, 1]], X, Y, "stroke:var(--k-rise);')
rep('r: 6 }, svg, "fill:var(--iw-strong);stroke:var(--surface);stroke-width:2")', 'r: 6 }, svg, "fill:var(--surface);stroke:var(--ink);stroke-width:2")')

# 5b. tables: the situations' group rows and column heads carry their colour; the worlds' headings their mark
for key, label in [("recorded", "Recorded conditions"), ("strong_current", "Strong current"), ("big_waves", "Big waves"),
                   ("big_kicks", "Big starting pushes")]:
    rep(f'<tr class="sub"><td>{label}</td>', f'<tr class="sub" data-sit="{key}"><td><span class="km" data-sit="{key}">{label}</span></td>')
REFHEAD = ('<thead><tr><th scope="col">Controller</th><th class="n" scope="col">Recorded conditions</th><th class="n" scope="col">'
           'Strong current</th><th class="n" scope="col">Big waves</th><th class="n" scope="col">Big starting pushes</th></tr></thead>')
rep(REFHEAD, REFHEAD.replace('scope="col">Recorded', 'scope="col" data-sit="recorded">Recorded')
    .replace('scope="col">Strong', 'scope="col" data-sit="strong_current">Strong').replace('scope="col">Big waves', 'scope="col" data-sit="big_waves">Big waves')
    .replace('scope="col">Big starting', 'scope="col" data-sit="big_kicks">Big starting'), count=2)
rep('    <h3>Stalling world</h3>\n', '    <h3><span class="km" data-world="stall">Stalling world</span></h3>\n')
rep('    <h3>No-stall comparison world</h3>\n', '    <h3><span class="km" data-world="no_stall">No-stall comparison world</span></h3>\n')
rep('<p>Read the strong-current rows. Every visited state is outside the recorded conditions, and\n      93.7% are in the full-throttle region.',
    '<p>Read the <span class="km" data-sit="strong_current">strong-current</span> rows. Every visited state is outside the '
    '<span class="km" data-sit="recorded">recorded conditions</span>, and\n      93.7% are in the '
    '<span class="km" data-k="region">full-throttle region</span>.')

# =================================================================================================================
# 6. Safety checks
# =================================================================================================================
for blk in QV_BLOCKS:
    assert s.count(blk) == 1, "a qviewer block changed: " + blk[:80]
for t in ("<!-- qviewer:begin", "qviewer:end"):
    assert s.count(t) == ORIG.count(t)
assert s.count('id="glance"') == 1 and s.count('data-wt-title="At a glance"') == 1
assert s.count("<mjx-container") == ORIG.count("<mjx-container") - old_glance.count("<mjx-container"), \
    (s.count("<mjx-container"), ORIG.count("<mjx-container"), old_glance.count("<mjx-container"))
print("formulas removed with the old glance:", old_glance.count("<mjx-container"),
      re.findall(r'<mjx-container[^>]*aria-label="([^"]*)"', old_glance))
hosts = lambda x: sorted(set(re.findall(r'(?:src|href)="(https?://[^/"]+)', x)))
assert hosts(s) == hosts(ORIG), (hosts(s), hosts(ORIG))
# every symbol legend is the markup the grid CSS expects: div.symkey > p.symkey-h, then (p.symkey-g)? dl, ...
LEG = list(re.finditer(r'<div class="symkey"[^>]*>(.*?)</dl></div>', s, flags=re.S))
for m in LEG:
    assert m.group(1).lstrip().startswith('<p class="symkey-h">'), m.group(1)[:80]
print(f"{N['n']} replacements; legends: {len(LEG)}; size {len(ORIG)} -> {len(s)}")
if not CHECK:
    open(PAGE, "w", encoding="utf-8", newline="").write(s)
    print("written", PAGE, hashlib.md5(s.encode("utf-8")).hexdigest())

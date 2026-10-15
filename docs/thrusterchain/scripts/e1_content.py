"""Edit 1, part B: the new content that goes with the reading order (run after e1_moves.py).

usage: python3 e1_content.py IN.html OUT.html     (IN and OUT may be the same file; OUT may never be the backup)

1. Action Space opens by saying what Gymnasium and Box are (the quick facts that did so now come later), and names
   the position and speed numbers where it first needs them
2. the live panel: a plain lead-in in four labelled parts, the three choices as three numbered rows, and a caption
   that says what the position and speed numbers are
3. Starting State rewritten in plain sentences (comment 2): each starting number, how it is drawn and why it
   matters; then what each situation changes; a recap grid; the formula; the evaluation banks
4. Full walkthrough mode: the pop-out (a native dialog), its styles and its script
Every edit asserts that its anchor occurs exactly once; nothing is written unless all edits succeed.
"""
import os
import re
import sys

from e1_common import (MJX, LABEL, data_line_hash, once, replace_once, section_span, strip_text, sub_ws)

HERE = os.path.dirname(os.path.abspath(__file__))


def take_mjx(s, a, b, label, what):
    """Return the one MathJax container in s[a:b] whose aria-label is `label` (raw attribute text)."""
    hits = [m for m in MJX.finditer(s, a, b) if LABEL.search(m.group(0)).group(1) == label]
    assert len(hits) == 1, f"{what}: expected one container labelled {label!r}, found {len(hits)}"
    return hits[0].group(0)


def main():
    src, dst = sys.argv[1], sys.argv[2]
    assert not dst.endswith("thrusterchain.v5.html"), "never modify the backup"
    s = open(src, encoding="utf-8").read()
    assert 'id="glance"' in s, "run e1_moves.py first"
    assert 'id="wt"' not in s and "sim-lead" not in s, "the content edit is already applied"
    data0 = data_line_hash(s)

    # ---------- 1. Action Space ----------
    a, b = section_span(s, "action-space")
    s = sub_ws(s, '<p>The action space is a <code>Box(-1.0, 1.0, (4,), float32)</code> (<code>float32</code> is how '
                  'each number is stored; see below). An action is four throttles, one per thruster (in order):</p>',
               '<p>ThrusterChain is a Gymnasium environment. Gymnasium, maintained by the Farama Foundation, is the\n'
               '      standard Python interface for environments in which controllers act and learn, and it describes a list\n'
               '      of numbers with fixed bounds as <code>Box(low, high, shape, type)</code>. The action space is a\n'
               '      <code>Box(-1.0, 1.0, (4,), float32)</code>: 4 numbers, each from −1 to 1 (<code>float32</code> is how\n'
               '      each number is stored; see below). An action is four throttles, one per thruster (in order):</p>',
               "action space opening", a, b)
    a, b = section_span(s, "action-space")
    s = sub_ws(s, "A thruster's push enters the speed numbers of the two floats it sits between, scaled by the push "
                  "factor 0.3.",
               "A thruster's push enters the speed numbers of the two floats it sits between, scaled by\n"
               "      the push factor 0.3. (Each float is described by two numbers, its <i>position number</i> and its\n"
               "      <i>speed number</i>, the two coordinates of its swing; both are 0 at rest. See\n"
               '      <a href="#observation-space">Observation Space</a>.)', "action space: speed numbers", a, b)

    # ---------- 2. the live panel ----------
    sa = once(s, '<figure class="panel" id="sim"', "live panel")
    sb = s.index("</figure>", sa)
    n0 = once(s, '      <p class="note">Pick a situation, a world and a controller.', "panel note", sa, sb)
    n1 = s.index("</p>\n", n0) + len("</p>\n")
    t1 = take_mjx(s, n0, n1, "t = 1", "panel note")
    lead = """      <div class="sim-lead">
        <p class="sim-intro">One real episode of the environment, replayed state by state, every step of it, for the
          situation, world and controller you pick. Pick in the three numbered rows below, then press Play or step
          through the episode's 100 states one at a time.</p>
        <div class="sim-keys">
          <div class="sim-key">
            <p class="sim-k">What you are looking at</p>
            <p>Five floats sit in a row across a channel of water. The water flows down the screen and pushes every
              float downstream, below the dashed line that marks rest. Four thrusters sit between neighbouring floats,
              each pushing the two floats beside it. In every state a controller sets the four throttles, to hold the
              floats at rest with as little throttle as it can.</p>
          </div>
          <div class="sim-key">
            <p class="sim-k">What the three choices mean</p>
            <ul>
              <li><b>Situation:</b> how the episode starts: how strong the current is, how big the wave is, and how
                far from rest the floats begin.</li>
              <li><b>World:</b> whether a propeller pushed past 80% throttle stalls and pushes less.</li>
              <li><b>Controller:</b> who sets the throttles.</li>
            </ul>
          </div>
          <div class="sim-key">
            <p class="sim-k">What to watch for</p>
            <ul>
              <li><b>The floats drifting from rest,</b> below the dashed line, and how quickly the controller brings
                them back.</li>
              <li><b>Thruster 1 stalling.</b> Thruster 1, the leftmost, does the most work. In the stalling world,
                once its throttle goes past 80%, its gauge turns red and reads STALLED, and its arrow, which shows the
                push it really gives, gets shorter even though its throttle is higher.</li>
            </ul>
          </div>
          <div class="sim-key">
            <p class="sim-k">What the readouts tell you</p>
            <p>The first six describe the state on screen: its number (counted from 0), the water's push, the float
              cost and throttle cost, the reward (minus the two costs, so 0 is the best a state can score) and the
              return so far. The last two score the chosen controller over all 1,000 fixed starts of the situation:
              its exact average return and its points (0 is the recording controller, 100 the stall-aware
              controller). Each is explained just under the readouts.</p>
          </div>
        </div>
        <p class="note">The panel opens at <span class="nw">""" + t1 + """,</span> the first state at which the
          recording controller gives the full-throttle command, so the stall shows at once. Try strong current with
          the rising-thrust controller, then switch to the no-stall comparison world.</p>
      </div>
"""
    s = s[:n0] + lead + s[n1:]
    # the readouts' key, right under the readouts it explains
    sa = once(s, '<figure class="panel" id="sim"', "live panel")
    sb = s.index("</figure>", sa)
    c0 = once(s, '      <canvas id="cost"', "cost chart", sa, sb)
    s = s[:c0] + """      <p class="note ro-note"><b>Water's push:</b> how hard the water pushes every float in this state; above 0.565
        the state is in the <i>full-throttle region</i>, where the recording controller puts thruster 1 at full
        throttle in half of the states, by a coin flip. <b>Float cost:</b> the five floats' squared distances from rest,
        added up. <b>Throttle cost:</b> 0.3 times the squared throttles, added up. <b>Return so far:</b> the rewards
        added up to this state. <b>Points:</b> in the no-stall comparison world, 100 is the rising-thrust controller.
        The chart below compares the cost so far of all four controllers from this same start, and the box under it
        spells out your three choices and what this episode did.</p>
""" + s[c0:]

    groups = [("lg-sit", "1. Situation", "1", "Situation", "How does the episode start?"),
              ("lg-world", "2. World", "2", "World", "What happens past 80% throttle?"),
              ("lg-ctrl", "3. Controller", "3", "Controller", "Who sets the throttles?")]
    for gid, old, num, name, q in groups:
        sa = once(s, '<figure class="panel" id="sim"', "live panel")
        sb = s.index("</figure>", sa)
        old_h = f'          <h3 class="optlbl" id="{gid}">{old}</h3>\n'
        h0 = once(s, old_h, f"option label {gid}", sa, sb)
        g1 = s.index("\n        </div>\n", h0)
        buttons = s[h0 + len(old_h):g1 + 1]
        assert all(ln.strip().startswith("<button") for ln in buttons.strip("\n").split("\n")), f"{gid}: not only buttons"
        new_h = (f'          <h3 class="optlbl" id="{gid}"><span class="optn" aria-hidden="true">{num}</span> {name}'
                 f' <span class="q">{q}</span></h3>\n')
        s = s[:h0] + new_h + '          <div class="optbtns">\n' + buttons + "          </div>" + s[g1:]

    sa = once(s, '<figure class="panel" id="sim"', "live panel")
    sb = s.index("</figure>", sa)
    s = sub_ws(s, "A float is drawn below the line when its position number is negative, which is where the water "
                  "carries it; the small arrow beside it shows its speed number.",
               "Each float is described by two numbers, its <i>position number</i> and its <i>speed number</i>,\n"
               "        the two coordinates of its swing, both 0 at rest (<a href=\"#observation-space\">Observation Space</a>). A float is\n"
               "        drawn below the line when its position number is negative, which is where the water carries it; the\n"
               "        small arrow beside it shows its speed number.", "panel caption: float numbers", sa, sb)
    sa = once(s, '<figure class="panel" id="sim"', "live panel")
    sb = s.index("</figure>", sa)
    s = sub_ws(s, "<b>Chart:</b> the cost so far (minus the return so far) of all four controllers. Try strong current "
                  "with the rising-thrust controller, then switch to the no-stall comparison world.</figcaption>",
               "<b>Chart:</b> the cost so far (minus the return so far) of all four controllers.</figcaption>",
               "panel caption: try", sa, sb)

    # ---------- 3. Starting State ----------
    a, b = section_span(s, "starting-state")
    r0 = once(s, "    <p>Every reset draws a start from the situation's <i>start law</i>, in three parts:</p>",
              "starting state opening", a, b)
    r1 = once(s, "      under.</p>\n", "starting state: percentile note", a, b) + len("      under.</p>\n")
    m = {lab: take_mjx(s, r0, r1, lab, "starting state") for lab in [
        r"\sigma", "c", r"U[\text{low}, \text{high}]", "U[0.1, 0.7]", "U[0.1, 0.4]", r"\phi", r"[0, 2\pi)",
        "w_1", "w_2", r"\cos\phi", r"\sin\phi", r"\mathcal{N}", "U"]}
    disp = [mm.group(0) for mm in MJX.finditer(s, r0, r1) if "begin{aligned}" in LABEL.search(mm.group(0)).group(1)]
    assert len(disp) == 1
    used = set(m.values()) | set(disp)
    assert len(list(MJX.finditer(s, r0, r1))) == len(used), "a container in the old text would be dropped"
    new = """    <p>Every episode begins from a <i>start</i>: the 13 numbers of its first state. Each reset draws them at random,
      by the rule of the chosen situation. A start sets four things, and each one matters for a different reason.</p>
    <ul class="leads">
      <li><b>Where the five floats start (10 numbers).</b> Each float's position number and speed number is drawn at
        random around 0, its value at rest, from a normal distribution (the familiar bell-shaped spread) with mean 0
        and standard deviation @@sigma@@, the typical size of a random number's spread around its mean. The
        standard deviation is 0.5 in every situation except big starting pushes, where it is 1.5, so every float
        number starts three times as large. This matters because the floats begin out of place, and a controller's
        first job is to damp the starting swing: a starting state's float cost averages 2.5 in the recorded conditions
        and 22.8 with big starting pushes, and left alone, float 1 still keeps 16% of its starting swing after 100
        steps.</li>
      <li><b>How strong the current is (1 number, @@c@@).</b> The current is drawn once and then holds for the
        whole episode. Every value in the situation's range is equally likely (a <i>uniform</i> draw): from 0.1 to 0.7
        in the recorded conditions, big waves and big starting pushes, and from 0.7 to 1.1 in strong current. This
        matters because the current sets how hard thruster 1 must push upstream in every state. The textbook hold
        against the current alone, with the floats at rest, asks thruster 1 for 80% once the current exceeds 0.879, so
        in strong current holding often asks for 80% or more, where the stall is.</li>
      <li><b>How big the wave is (its amplitude).</b> The amplitude, the largest push the wave adds to the current or
        takes away, is drawn uniformly: from 0.1 to 0.4 in the recorded conditions, strong current and big starting
        pushes, and from 0.4 to 0.7 in big waves, where each start of the evaluation bank has an amplitude 0.3 larger
        than the same start in the recorded conditions' bank. This matters because a bigger wave swings the water's
        push further every cycle: further into the full-throttle region near each crest, and further upstream near
        each trough.</li>
      <li><b>Where the wave starts in its cycle (its phase @@phi@@).</b> The phase is an angle drawn uniformly
        from <span class="nw">@@turn@@,</span> that is, from 0 up to, but not including, one full turn, in
        every situation. The two wave numbers follow from it: @@w1@@ is the amplitude times
        <span class="nw">@@cos@@,</span> the push the wave adds now, and @@w2@@ is the amplitude times
        <span class="nw">@@sin@@,</span> which shows which way the push is heading
        (<a href="#observation-space">Observation Space</a>). This matters because the phase decides whether the
        episode opens near a crest, where holding needs the most push, or near a trough, where it needs the
        least.</li>
    </ul>
    <p>Each shifted situation changes exactly one of these and leaves the rest as recorded. What each change does to
      the task is told in The task, under <a href="#task-situations">How each situation changes the challenge</a>.</p>
    <ul class="leads">
      <li><b>The recorded conditions</b> change nothing. They are the recording controller's own rule for drawing a
        start, the rule the training data come from, so by design none of their starts lies outside the recorded
        conditions, and the learned actor must not lose here. Even so, about 1% of the states visited count as outside
        on the float test that The distribution shift uses to measure how far a state lies from the recorded data
        (<a href="#shift-card">How the shift is measured</a>), because that test's mark, the 99th percentile (the
        value that 99% of the numbers stay under), leaves out the farthest 1% of the recorded training states
        themselves.</li>
      <li><b>Strong current</b> changes only the current, to 0.7 to 1.1. Every start lies outside the recorded
        conditions, because its current is above the recorded maximum, 0.7. This is the main test.</li>
      <li><b>Big waves</b> changes only the wave's size, to 0.4 to 0.7. Every start lies outside, because its
        amplitude is above the recorded maximum, 0.4. It is a shift in the waves: the water's push ranges further into
        the full-throttle region.</li>
      <li><b>Big starting pushes</b> changes only where the floats start, with a standard deviation of 1.5. On part
        of the steps the floats lie farther from rest than the 99th percentile of the recorded training states. It is
        a shift in the floats: the starts lie far from the recorded states.</li>
    </ul>
    <p class="note">The four situations side by side. Bold marks the one setting each shifted situation changes.</p>
<div class="tbl" style="max-width:760px"><table class="stack">
<thead><tr><th scope="col">Situation</th><th scope="col">Current: any value from</th><th scope="col">Wave amplitude: any value from</th><th scope="col">Float spread (standard deviation)</th></tr></thead><tbody>
<tr><td>Recorded conditions</td><td>0.1 to 0.7</td><td>0.1 to 0.4</td><td>0.5</td></tr>
<tr><td>Strong current</td><td><b>0.7 to 1.1</b></td><td>0.1 to 0.4</td><td>0.5</td></tr>
<tr><td>Big waves</td><td>0.1 to 0.7</td><td><b>0.4 to 0.7</b></td><td>0.5</td></tr>
<tr><td>Big starting pushes</td><td>0.1 to 0.7</td><td>0.1 to 0.4</td><td><b>1.5</b></td></tr>
</tbody></table></div>
    <p><b>In symbols.</b> The same rule, where the first line is drawn for each of the ten float numbers at the start
      (state 0):</p>
    <div class="math-display">@@disp@@</div>
    <p>where @@N@@ is the normal distribution and @@U@@ the uniform distribution:
      <span class="nw">@@Ulh@@</span> means every value from low to high is equally
      likely, so the recorded conditions draw the current from @@U17@@ and the amplitude from
      <span class="nw">@@U14@@.</span> A reset draws by the rule of the situation named by the
      <code>start</code> argument (default <code>"recorded"</code>), and each registered id fixes one
      (<a href="#arguments">Arguments</a>).</p>
"""
    keys = {"sigma": r"\sigma", "c": "c", "Ulh": r"U[\text{low}, \text{high}]", "U17": "U[0.1, 0.7]",
            "U14": "U[0.1, 0.4]", "phi": r"\phi", "turn": r"[0, 2\pi)", "w1": "w_1", "w2": "w_2", "cos": r"\cos\phi",
            "sin": r"\sin\phi", "N": r"\mathcal{N}", "U": "U"}
    for k, lab in keys.items():
        assert new.count("@@" + k + "@@") == 1, k
        new = new.replace("@@" + k + "@@", m[lab])
    assert new.count("@@disp@@") == 1
    new = new.replace("@@disp@@", disp[0])
    assert "@@" not in new
    s = s[:r0] + new + s[r1:]
    a, b = section_span(s, "starting-state")
    s = sub_ws(s, "A reset with a seed draws from the same law, but not from the bank,",
               "A reset with a seed draws by the same rule, but not from the bank,", "starting state: details", a, b)
    # "start law" was defined only in the old Starting State text: every other use now says what it is
    for old, new_words in [
            ("each with its own law for drawing the start", "each with its own rule for drawing the start"),
            ("ThrusterChain's arguments choose the world, the start law, the text output",
             "ThrusterChain's arguments choose the world, the rule for drawing a start, the text output"),
            ("The start law of the default reset:", "The rule for drawing the start of the default reset:"),
            ("overrides the start law's ranges.", "overrides the ranges of that rule."),
            ("Within a world only the start law differs.", "Within a world only the rule for drawing a start differs."),
            ("Situation (start law: Starting State)", "Situation (rule for drawing a start: Starting State)"),
            ("with a new start law (the current runs", "with a new rule for drawing a start (the current runs"),
            ('recorded: "The start law of the recorded data: current between',
             'recorded: "The rule the recorded data were drawn from: current between')]:
        s = sub_ws(s, old, new_words, "start law: " + old[:40])
    assert "start law" not in s.replace("tilted current laws", "")

    # ---------- 4. full walkthrough mode ----------
    css_old = """.opts { display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 14px; }
@media (max-width: 820px) { .opts { grid-template-columns: minmax(0, 1fr); gap: 10px; } .opt { padding: 5px 9px 6px; } .opt span { font-size: 13px; } }
.optgroup { display: flex; flex-direction: column; gap: 6px; min-width: 0; }
.optlbl { font-family: var(--body); font-size: 13px; letter-spacing: .08em; text-transform: uppercase; color: var(--muted);
  font-weight: 600; margin: 0 0 2px; }
"""
    css_new = open(os.path.join(HERE, "e1_opts.css"), encoding="utf-8").read()
    s = replace_once(s, css_old, css_new, "options CSS")
    css_wt = open(os.path.join(HERE, "e1_walkthrough.css"), encoding="utf-8").read()
    s = replace_once(s, "\n</style>\n", "\n" + css_wt.rstrip("\n") + "\n</style>\n", "style end")

    dialog = open(os.path.join(HERE, "e1_dialog.html"), encoding="utf-8").read()
    s = replace_once(s, "</nav>\n</div>\n", "</nav>\n</div>\n\n" + dialog.rstrip("\n") + "\n", "wrap end")

    js = open(os.path.join(HERE, "e1_walkthrough.js"), encoding="utf-8").read()
    assert "</script" not in js
    assert s.endswith("</script>\n"), "the page should end with its last script"
    s = s + "\n<script>\n" + js.rstrip("\n") + "\n</script>\n"

    # ---------- checks ----------
    assert data_line_hash(s) == data0, "the data line changed"
    ids = re.findall(r'\bid="([^"]+)"', s)
    dup = sorted({i for i in ids if ids.count(i) > 1})
    assert not dup, f"duplicate ids: {dup[:10]}"
    idset = set(ids)
    for h in set(re.findall(r'href="#([^"]+)"', s)):
        assert h in idset, f"href #{h} does not resolve"
    for attr in ("aria-labelledby", "aria-describedby", "aria-controls"):
        for v in re.findall(attr + r'="([^"]+)"', s):
            for one in v.split():
                assert one in idset, f"{attr} {one} does not resolve"
    open(dst, "w", encoding="utf-8").write(s)
    print(f"content: wrote {dst} ({len(s)} chars)")


if __name__ == "__main__":
    main()

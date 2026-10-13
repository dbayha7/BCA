"""Edit 4: the reviewers' findings, as exact (old, new) replacements on the tokenized page (see e4_apply.py).
Each group names the finding it answers (F1 ... F39, in the order of the review list)."""

# math tokens that may disappear: the symbols in the live panel's readout labels (F8)
DROP = {"6", "7", "8", "9"}

E = []
add = lambda old, new: E.append((old, new))  # noqa: E731

# ---------------------------------------------------------------- F38: the layout note in the style sheet
add("/* Layout: one environment's reference page in the order of a Gymnasium environment page. A reading column with a",
    "/* Layout: one environment's reference page. Reading order: the summary and the live panel, the task, then the\n"
    "   sections of a Gymnasium environment page (Action Space first), the methods and the results. A reading column with a")

# ---------------------------------------------------------------- F31: the walkthrough pop-out never grows wider than the screen
add("dialog.wt[open] { display: grid; grid-template-rows: auto minmax(0, 1fr) auto; animation: wt-pop .22s ease-out; }",
    "dialog.wt[open] { display: grid; grid-template-rows: auto minmax(0, 1fr) auto; grid-template-columns: minmax(0, 1fr);\n"
    "  animation: wt-pop .22s ease-out; }")
add(".wt-foot { display: flex; align-items: center; gap: 12px; padding: 10px 20px;",
    ".wt-foot { display: flex; align-items: center; gap: 12px; padding: 10px 20px; min-width: 0;")

# ---------------------------------------------------------------- F33: the throttle slider sits under its chart
add('.iw-sl input[type="range"] { flex: 1 1 140px; max-width: 260px; }',
    '.iw-sl input[type="range"] { flex: 1 1 140px; max-width: 260px; }\n'
    '.iw-chart > .iw-sl { flex: none; }   /* inside a chart column the row rule\'s 260px basis would become a height */')

# ---------------------------------------------------------------- the glance: F1, F2/F19, F3, F27, F37
add("is an add-on to TD3+BC (twin delayed deep deterministic policy gradient plus behaviour cloning), a standard method "
    "that learns a controller only from recorded data.",
    'is an add-on to TD3+BC, a standard method that learns a controller only from recorded data (the section '
    '"<a href="#methods">Methods</a>" explains its name).')
add('The whole task is explained in <a href="#task">The task</a>, right after <a href="#sim">Watch one episode</a>, '
    'which comes next.</p>',
    'The live panel "<a href="#sim">Watch one episode</a>" comes next, and right after it the section '
    '"<a href="#task">The task</a>" explains the whole task.</p>')
add("The <i>densest-15% removal</i>, the rule this project used on D4RL (Datasets for Deep Data-Driven Reinforcement "
    "Learning, the standard benchmark of recorded datasets), deletes the 15% of recorded commands that sit closest "
    "together, and with them almost every full-throttle command.",
    "The <i>densest-15% removal</i> deletes the 15% of recorded commands that sit closest together, and with them "
    "almost every full-throttle command. The project used the same rule on D4RL (Datasets for Deep Data-Driven "
    "Reinforcement Learning), the standard benchmark of recorded datasets.")
add("A true value is the exact discounted sum of the rewards that follow a command in a state while a given controller "
    "drives afterwards; discounted means that later rewards count less: a reward ⟦M1|t⟧ steps ahead counts ⟦M2|0.98^t⟧ "
    "as much.",
    "A true value is the exact discounted sum of the rewards that follow a command in a state, with a given controller "
    "driving afterwards. Discounted means later rewards count less: a reward ⟦M1|t⟧ steps ahead counts ⟦M2|0.98^t⟧ as "
    "much.")
add("Does caution placed exactly at the harm win back more than the same caution placed without that knowledge",
    "Does caution placed exactly where the critic is wrong win back more than the same caution placed without that "
    "knowledge")
add('<b>The outcome:</b> the first two checks passed, so the shift causes real, large harm. The third did not: caution '
    'of the kind BCA applies cannot reach that harm, in any situation, even when placed exactly where the harm is. So '
    'the harm checks stopped the test before any BCA run. See <a href="#results">Results</a>; the checks are described '
    'in <a href="#harm-checks">The harm checks</a>.</p>',
    '<b>The outcome:</b> the first two checks passed, so the shift causes real, large harm (lost return). The third '
    'did not: caution of the kind BCA applies cannot win back that lost return, in any situation, even when placed '
    'exactly where the critic is wrong. So the fourth, the seed-count check, never ran, and the harm checks stopped '
    'the test before any BCA run. See the section "<a href="#results">Results</a>"; the checks are described in the '
    'section "<a href="#harm-checks">The harm checks</a>".</p>')
add('<span class="tag acc">next</span> The stop was not accepted. The next experiment on this environment is '
    '<a href="#weighting-test">The weighting test</a>.',
    '<span class="tag acc">next</span> The project owner chose not to end the work at this stop. The next experiment '
    'on this environment is the weighting test, described in the section '
    '"<a href="#weighting-test">The weighting test</a>".')

# ---------------------------------------------------------------- the live panel: F8, F9, F15, F17, F34, F35
add("In every state a controller sets the four throttles, to hold the floats at rest with as little throttle as it "
    "can.</p>",
    "In every state a controller sets the four throttles, to hold the floats at rest with as little throttle as it "
    "can. Each float is described by two numbers, its position number and its speed number, both 0 at rest. The "
    "number printed above each float is its position number, negative when the water has carried the float "
    "downstream.</p>")
add("<span class=\"k\">water's push ⟦M6|c + w_1⟧ (full-throttle region above 0.565)</span>",
    "<span class=\"k\">water's push on every float (full-throttle region above 0.565)</span>")
add(r'<span class="k">float cost ⟦M7|\sum s_i^2⟧</span>', '<span class="k">float cost</span>')
add(r'<span class="k">throttle cost ⟦M8|0.3\sum a_k^2⟧</span>', '<span class="k">throttle cost</span>')
add('<span class="k">reward ⟦M9|r_t⟧</span>', '<span class="k">reward (minus the two costs)</span>')
add('<figcaption>Every episode here was stepped through the registered environment. All 32 combinations start from '
    'the same start (start 11 of the evaluation bank, the fixed 1,000 starts of each situation described in '
    '<a href="#starting-state">Starting State</a>), changed only by the situation\'s own setting, so the differences '
    'come from the controller and the world. <b>Top:</b> the water\'s push on every float; past the 0.565 mark the '
    'state is in the full-throttle region.',
    '<figcaption>Every episode here was run through the environment\'s own code. All 32 combinations (4 situations '
    '× 2 worlds × 4 controllers) use the same random draws: start 11 of each situation\'s evaluation bank, the fixed '
    'set of 1,000 starts described in the section "<a href="#starting-state">Starting State</a>". So the four '
    'situations\' starts differ only in the one setting each situation changes, and within a situation the '
    'differences come from the controller and the world. <b>Top:</b> the water\'s push on every float, the current '
    'plus the wave\'s push in this state; past the 0.565 threshold the state is in the full-throttle region.')
# the panel's texts (JS)
add("In the full-throttle region, where the water's push c + w\\u2081 is above 0.565, it gives the full-throttle "
    "command on half of the steps: thruster 1 at full throttle.",
    "In the full-throttle region, where the water's push (the current plus the wave's push) is above 0.565, it gives "
    "the full-throttle command in half of the states: thruster 1 at full throttle.")
add("Every state is outside the recorded conditions and almost every step is in the full-throttle region. Holding "
    "against this current often asks thruster 1 for 80% or more, which is where the stall matters: the textbook hold "
    "against the current alone, with the floats at rest, does once the current exceeds 0.879.",
    "Every state is outside the recorded conditions and almost every state is in the full-throttle region. Holding "
    "against this current often asks thruster 1 for 80% or more, which is where the stall matters: the textbook hold "
    "against the current alone, with the floats at rest, asks for 80% once the current exceeds 0.879.")
add("Early steps reach float distances the data rarely show.",
    "Early states reach float distances the data rarely show.")
add("`Steps past 80%: ${sm.steps_past_80}; stalled thruster-steps (one thruster stalled for one step counts 1): "
    "${sm.stalled_thruster_steps}. ` +\n    `Steps in the full-throttle region: ${sm.steps_in_full_throttle_region} "
    "of 100.`",
    "`States past 80% (some throttle above 80%): ${sm.steps_past_80} of 100; stalled thrusters, counted once per "
    "thruster in each state: ${sm.stalled_thruster_steps}. ` +\n    `States in the full-throttle region: "
    "${sm.steps_in_full_throttle_region} of 100.`")
# the drawing: no symbols before they are defined; the region named in words at every width; float labels in words
add('g.fillText(narrow ? "Water\'s push c + w₁" : "Water\'s push on every float, c + w₁", 0, meterTop + 12);',
    'g.fillText(narrow ? "Water\'s push on every float" : "Water\'s push on every float (current + wave)", 0, meterTop + 12);')
add('g.fillText(fmt(push, 3) + (inReg ? (narrow ? "  region" : "  full-throttle region") : ""), W, meterTop + 12);',
    'g.fillText(fmt(push, 3) + (inReg && !narrow ? "  full-throttle region" : ""), W, meterTop + 12);')
add('if (!narrow) { g.textAlign = "right"; g.fillText("1.8", W - 2, my + mh + 13); g.textAlign = "left"; '
    'g.fillText("−0.6", 2, my + mh + 13); }',
    'if (!narrow) { g.textAlign = "right"; g.fillText("1.8", W - 2, my + mh + 13); g.textAlign = "left"; '
    'g.fillText("−0.6", 2, my + mh + 13); }\n'
    '  // at phone width the region\'s name goes under the meter, right-aligned, where the "1.8" end label would be\n'
    '  if (narrow && inReg) { g.font = "600 11px " + mono; g.fillStyle = warn; g.textAlign = "right"; '
    'g.fillText("full-throttle region", W, my + mh + 13); }')
add('g.fillText("p " + fmt(p, 2) + (off ? (p < 0 ? " ↓" : " ↑") : ""), xs[j], chanTop + 15);',
    'g.fillText((W / 5 >= 120 ? "position " : "") + fmt(p, 2) + (off ? (p < 0 ? " ↓" : " ↑") : ""), xs[j], chanTop + 15);')
add("""  const midY = k => (Y(x[2 * k]) + Y(x[2 * k + 2])) / 2;
  for (let k = 0; k < 4; k++) {
    const cx = W * (k + 1) / 5, my = midY(k);
    for (let yy = chanTop + 26 + off; yy < chanTop + chanH - 6; yy += sp) {
      if (Math.abs(yy - my) < 30) continue;""",
    """  const midY = k => (Y(x[2 * k]) + Y(x[2 * k + 2])) / 2;
  // keep the chevrons clear of the two labels at the bottom of the channel ("downstream", and the recording mode)
  const clear = [], yLab = chanTop + chanH - 22;
  g.font = "11px " + body; clear.push([W - 10 - g.measureText("downstream ↓").width, W]);
  if (S.ctrl === "recording" && R.ft) {
    g.font = "600 12px " + body;
    clear.push([0, 14 + g.measureText(R.ft[t] ? "full-throttle command" : "normal driving").width]);
  }
  for (let k = 0; k < 4; k++) {
    const cx = W * (k + 1) / 5, my = midY(k);
    for (let yy = chanTop + 26 + off; yy < chanTop + chanH - 6; yy += sp) {
      if (Math.abs(yy - my) < 30) continue;
      if (yy > yLab && clear.some(c => cx + 6 > c[0] && cx - 6 < c[1])) continue;""")

# ---------------------------------------------------------------- The task: F12, F13, F14, F17, F37, F39, F35
add('<p class="note">In this section a <i>share of states</i> counts every state met in the 100 steps from each of a '
    'situation\'s 1,000 fixed evaluation starts (its <i>evaluation bank</i>, <a href="#starting-state">Starting '
    'State</a>). A float\'s <i>distance from rest</i> is the length of its pair of numbers, '
    r'<span class="nw">⟦M13|\sqrt{p^2 + v^2}⟧;</span> the float cost adds up the squared distances of the five floats.</p>',
    '<p class="note">In this section, a percentage of states is taken over every state met in the 100 steps of each '
    'of the situation\'s 1,000 fixed starts (its <i>evaluation bank</i>; see <a href="#starting-state">Starting '
    'State</a>). A float\'s <i>distance from rest</i> is the length of its pair of numbers, '
    r'<span class="nw">⟦M13|\sqrt{p^2 + v^2}⟧,</span> where <i>p</i> is its position number and <i>v</i> its speed '
    'number; the float cost adds up the squared distances of the five floats.</p>')
add("(the mark 0.565 was chosen so that about 13% of recorded commands are full-throttle commands)",
    "(the 0.565 threshold was chosen so that about 13% of recorded commands are full-throttle commands)")
add('push at every later step; ⟦M20|w_2⟧ even tells which way the push is heading',
    'push at every later step; the second wave number, ⟦M20|w_2⟧, the push the wave added a quarter cycle earlier, '
    'even tells which way the push is heading')
add('pick <b>No throttle</b> as the controller in <a href="#sim">Watch one episode</a> above.',
    'pick <b>No throttle</b> as the controller in the live panel "<a href="#sim">Watch one episode</a>" above.')
add('one of the fixed reference controllers listed under <a href="#reference-controllers">Reference controllers</a>,',
    'one of the fixed reference controllers listed under "<a href="#reference-controllers">Reference controllers</a>" '
    'in the section "The recording controller",')
add("In the recorded conditions it brings the floats' cost from 4.04 down to 0.153 per state,",
    "In the recorded conditions it brings the floats' cost from 4.04 per state with no throttle down to 0.153 per "
    "state,")
add("The part of ⟦M34|K⟧ that reads the wave asks thruster 1 for 0.88 per unit of amplitude, timed 1.4 steps ahead of "
    "the wave's push; the part that pushes against the floats' own movement brings the throttle back into step.",
    "Reading the wave, ⟦M34|K⟧ asks thruster 1 for 0.88 per unit of amplitude, timed 1.4 steps ahead of the wave's "
    "push. Its response to the floats' own movement then delays the throttle, back into step with the push.")
add('settled under <span class="nw">⟦M40|K⟧,</span> it asks for these at currents of only 0.844 and 1.056.',
    'settled under <span class="nw">⟦M40|K⟧,</span> it already asks for 80% at a current of 0.844 and for full '
    'throttle at 1.056.')
add("The mark 0.565 is not where holding needs 80%:", "The 0.565 threshold is not where holding needs 80%:")
# F35: the wide drawing's thruster names in a fixed row under the floats, with dotted leaders, as in the narrow one;
# the float labels move down a little to make room
add('<text class="dg-t" x="170.0" y="211.2" font-size="12.5" text-anchor="middle">thruster 1</text>',
    '<line class="dg-leader" x1="170.0" x2="170.0" y1="196.2" y2="222"/>'
    '<text class="dg-t dg-halo" x="170.0" y="234.0" font-size="12.5" text-anchor="middle">thruster 1</text>')
add('<text class="dg-t" x="290.0" y="192.4" font-size="12.5" text-anchor="middle">thruster 2</text>',
    '<line class="dg-leader" x1="290.0" x2="290.0" y1="177.4" y2="222"/>'
    '<text class="dg-t dg-halo" x="290.0" y="234.0" font-size="12.5" text-anchor="middle">thruster 2</text>')
add('<text class="dg-t" x="410.0" y="182.9" font-size="12.5" text-anchor="middle">thruster 3</text>',
    '<line class="dg-leader" x1="410.0" x2="410.0" y1="167.9" y2="222"/>'
    '<text class="dg-t dg-halo" x="410.0" y="234.0" font-size="12.5" text-anchor="middle">thruster 3</text>')
add('<text class="dg-t" x="530.0" y="178.0" font-size="12.5" text-anchor="middle">thruster 4</text>',
    '<line class="dg-leader" x1="530.0" x2="530.0" y1="163.0" y2="222"/>'
    '<text class="dg-t dg-halo" x="530.0" y="234.0" font-size="12.5" text-anchor="middle">thruster 4</text>')
_fl = [("110.0", "only thruster 1", "63"), ("230.0", "two thrusters", "31"), ("350.0", "two thrusters", "18"),
       ("470.0", "two thrusters", "11"), ("590.0", "only thruster 4", "8")]
add("".join(f'<text class="dg-t" x="{x}" y="248.0" font-size="12.5" text-anchor="middle">{a}</text>'
            f'<text class="dg-t" x="{x}" y="266.0" font-size="12.5" text-anchor="middle">swing: {s} steps</text>'
            for x, a, s in _fl),
    "".join(f'<text class="dg-t" x="{x}" y="252.0" font-size="12.5" text-anchor="middle">{a}</text>'
            f'<text class="dg-t" x="{x}" y="270.0" font-size="12.5" text-anchor="middle">swing: {s} steps</text>'
            for x, a, s in _fl))
# F39: what the drawing shows if the replay cannot run
add('<div class="td-ctrls" id="td-ctrls" hidden>',
    '<p class="note" id="td-static">If this drawing does not play, it stays still and shows the floats, drawn to '
    'scale, where a steady current would leave them with the thrusters off: float 1 furthest from rest and float 5 '
    'nearest. Its arrows then show only which way a negative throttle pushes.</p>\n'
    '      <div class="td-ctrls" id="td-ctrls" hidden>')
add("box.hidden = false;",
    "box.hidden = false;\n    const stNote = ui(\"td-static\"); if (stNote) stNote.hidden = true;   // the drawing plays")
# F35: a short push is a bare shaft; a head sitting on the box would read as a roof
add("""const dir = up ? -1 : 1, base = up ? top : bot, tip = base + dir * L, hh = Math.min(v.headH, L);
      show(v.heads[k], L >= 1.5); show(v.shafts[k], L > hh + 0.5);
      set(v.shafts[k], { x1: x, x2: x, y1: base, y2: tip - dir * (hh - 1) });""",
    """const dir = up ? -1 : 1, base = up ? top : bot, tip = base + dir * L;
      const withHead = L >= 2 * v.headH, hh = withHead ? v.headH : 0;   // no head when the shaft would be shorter than it
      show(v.heads[k], withHead); show(v.shafts[k], L > 1.5);
      set(v.shafts[k], { x1: x, x2: x, y1: base, y2: withHead ? tip - dir * (hh - 1) : tip });""")

# ---------------------------------------------------------------- Action Space and Observation Space: F11
add("Up to 80% the push equals the throttle; past 80% the propeller stalls (below).</p>",
    "Up to 80% the push equals the throttle; past 80% the propeller stalls (below). In the table, the index counts "
    "from 0, as the code does, and the observation items are places in the list of 13 observed numbers given in the "
    'next section, <a href="#observation-space">Observation Space</a>.</p>')
add('<thead><tr><th scope="col">Num</th><th scope="col">Action</th>',
    '<thead><tr><th scope="col">Index</th><th scope="col">Action</th>')
for a, b in [("1 and 2", "1, 3"), ("2 and 3", "3, 5"), ("3 and 4", "5, 7"), ("4 and 5", "7, 9")]:
    add(f"<td>floats {a} (observations {b})</td>",
        f"<td>floats {a} (observation items {b.replace(', ', ' and ')})</td>")
add('for its clipped throttle <span class="nw">⟦M43|a⟧.</span>',
    'for its throttle <span class="nw">⟦M43|a⟧,</span> after it is limited to −1 to 1.')
add('<thead><tr><th scope="col">Num</th><th scope="col">Observation</th>',
    '<thead><tr><th scope="col">Index</th><th scope="col">Observation</th>')
add('the push the wave adds now</td><td class="n">−1</td><td class="n">1</td><td>turns 0.27 radians per step</td>',
    'the push the wave adds now</td><td class="n">−1</td><td class="n">1</td><td>turns 0.27 radians (about 15 '
    'degrees) per step</td>')
add("positive while the wave's push is falling</td><td class=\"n\">−1</td><td class=\"n\">1</td><td>turns 0.27 radians "
    "per step</td>",
    "positive while the wave's push is falling</td><td class=\"n\">−1</td><td class=\"n\">1</td><td>turns 0.27 radians "
    "(about 15 degrees) per step</td>")

# ---------------------------------------------------------------- Starting State: F37, F9, F4
add('What each change does to the task is told in The task, under <a href="#task-situations">How each situation '
    'changes the challenge</a>.</p>',
    'What each change does to the task is told in the section "The task", under "<a href="#task-situations">How each '
    'situation changes the challenge</a>".</p>')
add('Even so, about 1% of the states visited count as outside on the float test that The distribution shift uses to '
    'measure how far a state lies from the recorded data (<a href="#shift-card">How the shift is measured</a>), because '
    'that test\'s mark, the 99th percentile (the value that 99% of the numbers stay under), leaves out the farthest 1% '
    'of the recorded training states themselves.</li>',
    'Even so, about 1% of the states visited count as outside the recorded conditions. The reason is the float test, '
    'which measures how far a state\'s floats lie from the recorded data (described in the section "The distribution '
    'shift", under "<a href="#shift-card">How the shift is measured</a>"): its cut-off, the 99th percentile (the value '
    'that 99% of the numbers stay under), leaves out the farthest 1% of the recorded training states themselves.</li>')
add("On part of the steps the floats lie farther from rest than the 99th percentile",
    "In part of the states the floats lie farther from rest than the 99th percentile")
add("A reset can also restart from an exact state, which replay and the exact truth use",
    "A reset can also restart from an exact state, which replaying a recorded episode and the exact simulation use")

# ---------------------------------------------------------------- Transition Dynamics, Episode End: F4
add("the environment, the data generator and the exact truth all call it,",
    "the environment, the data generator and the exact simulation of true values all call it,")
add("<p><b>Exact values of the never-ending sum.</b> The truth on this page computes that sum by simulating far enough "
    "that the rest no longer matters.",
    "<p><b>Exact values of the never-ending sum.</b> This page computes that sum exactly by simulating far enough that "
    "the rest no longer matters.")
add("The fixed truth labels (the rows the harm checks score, the price of pushing past 80%, the detection labels) "
    "simulate 1,000 steps,",
    "The exact values that are computed once and then kept fixed (at the rows the harm checks score, for the price of "
    "pushing past 80%, and the exact over-preference values the detection checks are scored against) simulate 1,000 "
    "steps,")
add("Both are the same truth; the shorter one is only cheaper.",
    "Both are the same exact computation; the shorter one is only cheaper.")

# ---------------------------------------------------------------- Summary of the environment: F37
add('What the water does and how a controller counteracts it are explained in <a href="#task">The task</a>.</p>',
    'What the water does and how a controller counteracts it are explained in the section '
    '"<a href="#task">The task</a>".</p>')
add('The springs drawn in <a href="#sim">Watch one episode</a> are only a picture.',
    'The springs drawn in the live panel "<a href="#sim">Watch one episode</a>" are only a picture.')

# ---------------------------------------------------------------- The recording controller: F17, F26
add("found by the standard calculation for linear dynamics with a quadratic cost (the discounted linear-quadratic "
    "regulator):",
    "found by the standard calculation for linear dynamics with a quadratic cost, meaning a cost built from squares "
    "and products of the numbers, like the float cost (the calculation is called the discounted linear-quadratic "
    "regulator):")
add("<p>The trigger 0.565 was set by <i>the design plan</i>:", "<p>The 0.565 threshold was set by <i>the design plan</i>:")
add("design plan chose the trigger so that 13% of recorded commands",
    "design plan chose the threshold so that 13% of recorded commands")
add("The trigger depends only on the water, which no thruster changes.",
    "The threshold depends only on the water, which no thruster changes.")

# ---------------------------------------------------------------- The recorded data: F25, F37
add('listed in <a href="#code-names">Names used in the code</a>. In the Type column,',
    'listed in the section "<a href="#code-names">Names used in the code</a>". In the Type column,')
add("<td>the calibration set: 1,000 rows (5 per episode) that BCA uses to calibrate its band</td>",
    "<td>the calibration set: 1,000 rows (5 per episode) that BCA uses to set the width of its band (a range around "
    'the critic\'s estimate that should contain its error; see <a href="#methods">Methods</a>)</td>')

# ---------------------------------------------------------------- The distribution shift: F9, F19, F37
add('still sends commands inside the removed ball (the column "Command in the removed ball" of <a '
    'href="#shift-card">the shift table</a>): on 17.2% of its steps in the recorded conditions and 57.0% in strong '
    'current (normal driving only: 9.4% and 71.7%). For the critic-error check\'s pull-past-80% rule (<a '
    'href="#harm-checks">The harm checks</a>; the measure is defined in <a href="#methods">Methods</a>) this means that '
    'both commands it compares, full throttle and the 80% hold, are missing from the data after the removal.',
    'still sends commands inside the removed ball (the column "Command in the removed ball" of <a '
    'href="#shift-card">the shift table</a>): on 17.2% of the states it visits in the recorded conditions and 57.0% in '
    'strong current (normal driving only: 9.4% and 71.7%). One rule of the critic-error check measures the critic\'s '
    '<i>extra preference for full throttle</i>: how much more the critic prefers full throttle over the 80% hold than '
    'the exact values do (the rule is in the section "<a href="#harm-checks">The harm checks</a>", and the measure is '
    'defined in "<a href="#methods">Methods</a>"). For that rule, this means that both commands it compares, full '
    'throttle and the 80% hold, are missing from the data after the removal.')
add('On the four evaluation banks of 1,000 starts from <a href="#starting-state">Starting State</a>:',
    'On the four evaluation banks of 1,000 starts described in the section "<a href="#starting-state">Starting '
    'State</a>":')
add('The learner, TD3+BC, and every caution method are explained in <a href="#methods">Methods</a>.',
    'The learner, TD3+BC, and every caution method are explained in the section "<a href="#methods">Methods</a>".')
add("<li><b>Steps past 80%</b>: the share of steps that command more than 80% on any thruster.",
    "<li><b>States past 80%</b>: the share of visited states at which the command exceeds 80% on any thruster.")
add("<li><b>Command in the removed ball</b>: the share of steps whose command lies inside the removed ball,",
    "<li><b>Command in the removed ball</b>: the share of visited states whose command lies inside the removed ball,")
add("<li><b>Covered share</b>: of the steps past 80%, the share inside the recorded conditions.",
    "<li><b>Covered share</b>: of the states past 80%, the share inside the recorded conditions.")
add('<th class="n" scope="col">2. Steps past 80%</th>', '<th class="n" scope="col">2. States past 80%</th>')
add('84.0% of its steps there (the column "Steps past 80%"), none of them',
    '84.0% of the states it visits there (the column "States past 80%"), none of them')
add("9.7% of its steps and loses", "9.7% of the states it visits and loses")

# ---------------------------------------------------------------- Methods: F4, F9, F16, F19, F20, F37
add("In Methods and the checks that follow, ⟦M160|x⟧", "In this section and the checks that follow, ⟦M160|x⟧")
add("<li>A critic that <i>bends like the truth past 80%</i> (the <i>bending critic</i>) keeps its hill: the actor goes "
    "the share",
    "<li>A critic that <i>curves down past 80%, as the exact value does,</i> keeps its hill: the actor goes the share")
add("<li>A critic that <i>keeps its slope past the edge of the data</i> (the <i>straight-slope critic</i>) has no "
    "curvature beyond the recorded commands.",
    "<li>A critic that <i>keeps its slope past the recorded commands</i> has no curvature beyond them.")
add('<th class="n" scope="col">Bending critic: share μ / (m + μ)</th><th class="n" scope="col">Straight-slope critic: '
    'share min(μ / m, 1)</th>',
    '<th class="n" scope="col">Curving critic (bends past 80%): share μ / (m + μ)</th><th class="n" '
    'scope="col">Straight critic (keeps its slope past 80%): share min(μ / m, 1)</th>')
add("of the way for the bending critic, and ⟦M210|0.580 - 0.387 = 0.193⟧ for the straight-slope critic.",
    "of the way for a critic that curves down past 80%, and ⟦M210|0.580 - 0.387 = 0.193⟧ for one that keeps its "
    "slope.")
add("normal driving moved 0.367 (bending) or 0.580 (straight slope) of the way toward",
    "normal driving moved 0.367 (the share for a critic that curves down past 80%) or 0.580 (for one that keeps its "
    "slope) of the way toward")
add('<th class="n" scope="col">Forecast harm, bending critic (points)</th><th class="n" scope="col">Forecast harm, '
    'straight-slope critic (points)</th>',
    '<th class="n" scope="col">Forecast harm, curving critic (bends past 80%), points</th><th class="n" '
    'scope="col">Forecast harm, straight critic (keeps its slope past 80%), points</th>')
add("0.9% of them with the bending critic and 13.8% with the straight-slope critic, and in the recorded conditions "
    "capping",
    "0.9% of them for a critic that curves down past 80% and 13.8% for one that keeps its slope, and in the recorded "
    "conditions capping")
add("(the rising-thrust controller pushes past 80% on 84.0% of its steps there)",
    "(the rising-thrust controller pushes past 80% on 84.0% of the states it visits there)")
add("With a fixed controller, the exact truth ⟦M212|Q^{\\text{true}}⟧ of that controller is known at every row,",
    "With a fixed controller, the exact value ⟦M212|Q^{\\text{true}}⟧ of that controller, computed by exact "
    "simulation, is known at every row,")
add("<li><b>Pull past 80%</b>: how much more the critic prefers the full-throttle pattern over the capped pattern than "
    "the truth does:</li>",
    "<li><b>Extra preference for full throttle</b>: how much more the critic prefers the full-throttle pattern over "
    "the capped pattern than the exact values do:</li>")
add("the check compares its guesses with the truth and with the full-log critic's.",
    "the check compares its guesses with the exact values and with the full-log critic's guesses.")
add("whether the loss sits at the steps where the actor pushes past 80%. Minus a step's term is the return lost at "
    "that step.",
    "whether the loss sits at the states where the actor pushes past 80%. Minus a state's term is the return lost at "
    "that state.")
add("<p>Before BCA is ever run, the test asks whether per-row caution could help at all, using the exact "
    "over-preference in place of BCA's estimate. On the frozen critic of a plain TD3+BC run on the log after the "
    "removal (critic and actor trained together, then the critic frozen), actors are trained from the same cloned "
    "start with different caution multipliers <span class=\"nw\">⟦M229|m_i⟧.</span> The exact quantity they are placed "
    "by is each row's <i>over-preference</i> <span class=\"nw\">⟦M230|\\Delta_i⟧:</span> how much more the critic "
    "prefers the actor's command over the recorded one than the truth does, counted only when positive:</p>",
    "<p>Before BCA is ever run, the test asks whether per-row caution could help at all if it were placed perfectly. "
    "Perfect placement uses each training row's <i>over-preference</i> <span class=\"nw\">⟦M230|\\Delta_i⟧:</span> how "
    "much more the critic prefers the actor's command over the recorded one than the exact values do, counted only "
    "when positive:</p>")
add("<p>Extra cloning at row ⟦M232|i⟧ corrects exactly this. Here ⟦M233|Q^{\\text{true}}⟧ is the exact value",
    "<p>Extra cloning at row ⟦M232|i⟧ corrects exactly this. BCA has to estimate where its critic errs; these "
    "placements use the exact over-preference instead. On the frozen critic of a plain TD3+BC run on the log after "
    "the removal (critic and actor trained together, then the critic frozen), actors are trained from the same cloned "
    "start with different caution multipliers <span class=\"nw\">⟦M229|m_i⟧.</span> Here ⟦M233|Q^{\\text{true}}⟧ is "
    "the exact value")
add("<li><b>Harm-placed caution</b>, BCA's own dose form placed by the exact over-preference, the harm the critic's "
    "error does at that row: ⟦M234|m_i = 1 + 0.5\\,z_i / (1 + z_i)⟧ with",
    "<li><b>Harm-placed caution</b>, named for the harm (lost return) that over-preference causes: the same dose "
    "formula BCA uses (see BCA, below), placed by the exact over-preference: ⟦M234|m_i = 1 + 0.5\\,z_i / (1 + z_i)⟧ "
    "with")
add("every row gets the mean harm-placed dose of the same refresh.",
    "every row gets the mean harm-placed dose of the same recomputation (every 4,000 steps).")

# ---------------------------------------------------------------- The harm checks: F9, F19, F21, F22, F37
add("harm checks pass. <i>Pooled</i> means across the five seeds.",
    "harm checks pass. \"Pooled\" means combined across the five seeds.")
add("the median pull past 80% of the critic trained after the removal is positive and exceeds the full-log critic's "
    "by at least 25% of the median true price, in at least 4 of the 5 seeds.",
    "the median extra preference for full throttle of the critic trained after the removal is positive and exceeds "
    "the full-log critic's by at least 25% of the median true price of pushing past 80%, in at least 4 of the 5 "
    "seeds.")
add("(pooled share of steps)", "(pooled share of states)")
add("<li>The loss sits at its past-80% steps. The <i>loss rate</i> is the return lost per step by the step-by-step "
    "accounting (minus a step's term), averaged separately over the actor's steps past 80% and over its other steps, "
    "across all five seeds and all 1,000 strong-current starts. The loss rate at the past-80% steps must be positive "
    "and at least twice the loss rate at the other steps. When the other steps lose nothing at all, any positive loss "
    "rate at the past-80% steps meets this.</li>",
    "<li>The loss sits at the states where it pushes past 80%. The <i>loss rate</i> is the return lost in one state by "
    "the step-by-step accounting (minus that state's term), averaged separately over the states where the actor "
    "pushes past 80% and over its other states, across all five seeds and all 1,000 strong-current starts. The loss "
    "rate at the past-80% states must be positive and at least twice the loss rate at the other states. When the "
    "other states lose nothing at all, any positive loss rate at the past-80% states meets this.</li>")
add("<h3>The best-case caution check: could caution placed exactly at the harm help?</h3>",
    "<h3>The best-case caution check: could caution placed exactly where the critic is wrong help?</h3>")
add('a 5% false-alarm rate shared over the three tests of the gain check in <a href="#bca-test">The BCA test</a>, one '
    'per shifted situation (the worst case of the correction used there).',
    'a 5% false-alarm rate shared over three tests, one per shifted situation, of whether BCA beats plain TD3+BC (the '
    'gain check in the section "<a href="#bca-test">The BCA test</a>"; this is the strictest case of Holm\'s '
    'correction, explained there).')
add("the one pre-registered <i>steeper-stall rerun</i> runs: stall slope 3.0 (push 0.2 at full throttle), the nine "
    "world checks",
    "the one pre-registered <i>steeper-stall rerun</i> runs, with a steeper stall: past 80%, push falls by 3.0 per "
    "unit of throttle instead of 1.5 (push 0.2 at full throttle). It reruns the nine world checks")

# ---------------------------------------------------------------- The BCA test: F23
add("at a terminal. Every BCA run, calibration fit, band refresh and dose readout refuses to start without that "
    "approval file.",
    "at a terminal, which writes an approval file. Every BCA run, calibration fit, band refresh and dose readout "
    "refuses to start without that file.")
add("A row among the 10% most over-preferred gets a larger dose than another row at least 60% of the time on average "
    "(50% is chance), with the lower bound above 50%.",
    "Pick one row from the 10% most over-preferred and one from the rest: the first gets the larger dose at least 60% "
    "of the time on average (50% would be chance), with the lower bound above 50%.")

# ---------------------------------------------------------------- Results: F2, F4, F9, F10, F16, F19, F24, F28, F37
add("<b>The reasoning:</b> before BCA is judged, the best-case caution check gives that tool its best possible aim, "
    "the exact harm, and asks whether it then beats the same caution spread uniformly over every row. <b>The "
    "outcome:</b> it does not. In no shifted situation did caution placed by the exact harm, inside BCA's dose range, "
    "beat uniform caution, so no BCA result could have been informative.",
    "<b>The reasoning:</b> before BCA is judged, the best-case caution check gives that tool its best possible aim, "
    "each training row's exact over-preference (how much more the critic prefers the actor's command over the "
    "recorded one than the exact values do), and asks whether it then beats uniform caution of the same strength, "
    "spread over every row, by at least 3 points. <b>The outcome:</b> it does not. In no shifted situation did "
    "caution placed by the exact over-preference, inside BCA's dose range, beat uniform caution of the same strength "
    "by the required 3 points, so no BCA result could have been informative.")
add("and each was compared with the exact truth.", "and each was compared with the exact simulated values.")
add("The loss sits at the steps that push past 80%. The return-cost check passed all five of its rules at critic "
    "weight <span class=\"nw\">⟦M268|\\alpha = 5⟧.</span></dd>",
    "The loss sits at the states where the actor pushes past 80%. The return-cost check passed all five of its rules "
    "at critic weight <span class=\"nw\">⟦M268|\\alpha = 5⟧.</span> At the standard weight, 2.5, one rule fell just "
    "short (1.46 times as many states past 80%, against the required 1.5), and the pre-registered critic-weight "
    "step-up rule then reran the check at 5.</dd>")
add("<dt>Reasoning</dt><dd>So the pull was placed by the exact harm (each training row's over-preference: how much "
    "more the critic prefers the actor's command over the recorded one than the truth does; see",
    "<dt>Reasoning</dt><dd>So the pull was placed by each training row's exact over-preference (how much more the "
    "critic prefers the actor's command over the recorded one than the exact values do; see")
add("<dt>Outcome</dt><dd>Even placed by the exact harm, inside BCA's dose range, it beats uniform caution by under 1 "
    "point in every situation, and in strong current it does slightly worse.",
    "<dt>Outcome</dt><dd>Even placed this way, inside BCA's dose range, it beats uniform caution of the same strength "
    "by under 1 point in big waves and big starting pushes (+0.9 and +0.8), far short of the 3-point pass mark, and "
    "in strong current it does 1.9 points worse.")
add("<dt>Reasoning</dt><dd>The best-case caution check placed caution by the exact harm, which is better information",
    "<dt>Reasoning</dt><dd>The best-case caution check placed caution by the exact over-preference, which is better "
    "information")
add("in BCA's range, moves this harm, however well the harm is found.",
    "in BCA's range, wins back this lost return, however well the pull is aimed.")
add("The test was fixed to check only that the tests leave that folder unchanged,",
    "The faulty test was changed to check only that running the tests leaves that folder unchanged,")
add("as a fixed version (commit <code>bdb8c8a</code>) before any learner ran.",
    "as a fixed version (saved version <code>bdb8c8a</code> in the repository's history) before any learner ran.")
add('<tr><td>Pull past 80% in the full-throttle region: critic trained after the removal; full-log critic</td><td '
    'class="n">+0.33 to +0.36; −0.06 to −0.01</td><td>the critic trained after the removal exceeds the full-log '
    'critic by a quarter of the true price, in 4 of 5 seeds</td><td><span class="tag ok">pass</span></td></tr>',
    '<tr><td>Extra preference for full throttle in the full-throttle region: critic trained after the removal</td><td '
    'class="n">+0.33 to +0.36</td><td rowspan="2">the critic trained after the removal exceeds the full-log critic '
    'by at least a quarter of the median true price of pushing past 80%, in at least 4 of the 5 seeds</td><td '
    'rowspan="2"><span class="tag ok">pass</span></td></tr>\n'
    '        <tr><td>Extra preference for full throttle in the full-throttle region: full-log critic</td><td '
    'class="n">−0.06 to −0.01</td></tr>')
add("and the lost return sits at its past-80% steps.", "and the lost return sits at the states where it pushes past 80%.")
add("<h3>The best-case caution check: could caution placed at the harm help?</h3>",
    "<h3>The best-case caution check: could caution placed where the critic is wrong help?</h3>")
add("compare caution placed by the exact harm with the same caution placed without that knowledge;",
    "compare caution placed by the exact over-preference with the same caution placed without that knowledge;")
add("the largest on the rows where the exact harm is largest.</li>",
    "the largest on the rows where the exact over-preference is largest.</li>")
add("share of steps past 80% stays between 0.61 and 0.68", "share of states past 80% stays between 0.61 and 0.68")
add("pushes past 80% on 9% of strong-current steps and scores 54.",
    "pushes past 80% on 9% of the strong-current states it visits and scores 54.")
add('(worked out in <a href="#methods">Methods</a>, under "How far the actor moves",',
    '(worked out in the section "<a href="#methods">Methods</a>", under "How far the actor moves",')
add("in strong current with a critic that bends like the truth past 80% (the bending critic), and 40.4 points with "
    "one that keeps its slope past the edge of the data (the straight-slope critic), against",
    "in strong current with a critic that curves down past 80% as the exact value does, and 40.4 points with one that "
    "keeps its slope past the recorded commands, against")
add("and 5.4 points with the bending critic, and by 14.6, 21.5 and 24.0 with the straight-slope critic, all six above",
    "and 5.4 points with the critic that curves down, and by 14.6, 21.5 and 24.0 with the critic that keeps its "
    "slope, all six above")
add("two rules fail: the critic-error check's pull-past-80% rule and the return-cost check's 5-point mark.",
    "two rules fail: the critic-error check's rule on the extra preference for full throttle and the return-cost "
    "check's 5-point mark.")
add("differ by more than these 1.5-fold bands allow.", "differ by more than these 1.5-fold limits allow.")
add("<td>forecast harm in strong current at least 5 points with the bending critic and 20 with the straight-slope "
    "critic, at the seed's own critic's pull μ (Methods)</td>",
    "<td>forecast harm in strong current at least 5 points with a critic that curves down past 80% and 20 with one "
    "that keeps its slope past the recorded commands, at the seed's own critic's pull μ (see the section "
    "\"Methods\")</td>")
add("forecast in Methods, which uses the design plan's",
    'forecast in the section "<a href="#methods">Methods</a>", which uses the design plan\'s')

# ---------------------------------------------------------------- The weighting test: F5, F6, F7, F9, F17, F18, F25, F29, F32, F37
add("<p>Each distribution of the current is a simple tilt over its range 0.1 to 1.1: the <i>density</i>, a curve "
    "whose height says how likely each current is, is proportional to",
    "<p>Each distribution of the current leans toward weak or strong currents by a set amount λ over the range 0.1 "
    "to 1.1: its <i>density</i>, a curve whose height says how likely each current is, is proportional to")
add('for="iw-lam">Test tilt λ <input', 'for="iw-lam">Test lean λ <input')
add("<dt>λ</dt><dd>The tilt of a current distribution, density ∝ e<sup>λ<i>c</i></sup>.",
    "<dt>λ</dt><dd>How far a current distribution leans toward strong currents (negative values lean toward weak "
    "ones): density ∝ e<sup>λ<i>c</i></sup>.")
add("<dd>The constant that makes a density integrate to 1 over 0.1 to 1.1:",
    "<dd>The constant that makes the area under the density over 0.1 to 1.1 equal 1:")
add("<dt>Calibration set</dt><dd>1,000 held-out recorded steps that BCA uses to set how wide its band must be.</dd>",
    "<dt>Calibration set</dt><dd>1,000 recorded rows from the calibration split, never trained on, that BCA uses to "
    "set how wide its band must be.</dd>")
add("<h3>The removed action: a common, good move</h3>", "<h3>The removed command: a common, good move</h3>")
add('It drives normally like the one in <a href="#recording-controller">The recording controller</a>, but in hard '
    'water a coin flip each step chooses the 80% hold instead of full throttle. Because the hold repeats exactly, it '
    'forms the densest cluster of recorded commands, and the densest-15% rule deletes it from the training data and '
    'the calibration set.',
    'It drives normally like the one in the section "<a href="#recording-controller">The recording controller</a>", '
    'but in <i>hard water</i> (states where thruster 1\'s hold demand, defined below, exceeds 0.65; this replaces the '
    'original recording controller\'s full-throttle region, where the water\'s push exceeds the 0.565 threshold) a '
    'coin flip in each state chooses the 80% hold instead of full throttle. Because the hold repeats exactly, it '
    'forms the densest cluster of recorded commands, and the densest-15% removal deletes it from the training data '
    'and the calibration set.')
add('<p class="note">Share of test steps whose recorded command falls in the removed cluster</p>',
    '<p class="note">Share of test states whose recorded command falls in the removed cluster</p>')
add('aria-label="Share of test steps inside the removed cluster for each test setting"',
    'aria-label="Share of test states whose command is inside the removed cluster, for each test setting"')
add("The densest-15% rule removes 99.7–99.8% of the holds.",
    "The densest-15% removal takes out 99.7–99.8% of the holds.")
add('for="iw-ex-a">Wave amplitude A <input', 'for="iw-ex-a">Wave amplitude <input')
add("<dt>A</dt><dd>Wave amplitude: the largest push", "<dt>Amplitude</dt><dd>The wave's amplitude: the largest push")
add("<dd>The push the wave adds now, A·cos φ.", "<dd>The push the wave adds now, amplitude × cos φ.")
add("<dd>A·sin φ: the push the wave added", "<dd>Amplitude × sin φ: the push the wave added")
add("<dd>The share of steps whose error actually falls inside the band. The target is 90%.</dd>",
    "<dd>The share of rows whose critic error actually falls inside the band. The target is 90%.</dd>")
add("as well as exact ones. A test step whose command was removed has an infinite weight, so the weighted band must "
    "refuse to vouch for it. That refusal is BCA correctly flagging the missing common action.</p>",
    "as well as exact ones. The weight ω(<i>c</i>) assumes that training and test differ only in the current. The "
    "removal breaks that for one command: the 80% hold occurs in test episodes but has zero chance in the training "
    "data after the removal, so its ratio of test to training probability is a number divided by zero, which has no "
    "limit. No band can be calibrated for a test state with that command, so the weighted band must decline to give "
    "it one. That refusal is BCA correctly flagging the missing common command.</p>")
add("<li><b>Data with exact truth.</b>", "<li><b>Data with exact values.</b>")
add("which sees only observations from the two distributions; how sure it is gives the ratio. Steps inside the "
    "removed cluster are reported separately.</li>",
    "which sees only observations from the two distributions. Its estimated odds that an observation came from a "
    "test episode rather than from training, corrected for how many observations of each kind it learned from, give "
    "the ratio. Test states whose command lies inside the removed cluster are reported separately.</li>")
add("against no caution? If not, weighting cannot either, and coverage is the result.</li>",
    "against no caution? If not, weighting cannot raise the return either, and the coverage measurements become the "
    "main result.</li>")
add("Points run from 0 (this test's recording controller) to 100 (the stall-aware controller).</p>",
    "Points run from 0 (this test's recording controller) to 100 (the stall-aware controller). Coverage differences "
    "are in percentage points; returns are in the page's 0–100 points.</p>")
add("the removed share of test steps is within 3 points of the trial simulation's</td>",
    "the share of test states whose command falls in the removed cluster is within 3 percentage points of the trial "
    "simulation's</td>")
add("<td>The band is set to be 95% sure of at least 90%, so about 91.7% is expected</td>",
    "<td>BCA picks a width that is 95% likely to cover at least 90% of errors; that safety margin makes the expected "
    "coverage about 91.7%</td>")
add("<td>Bookkeeping</td>",
    "<td>The dose is computed directly from the width, so this only confirms that the code passes the width through "
    "correctly</td>")
add("<td>Exactly weighted, test steps outside the removed cluster: pooled 88–96% under mild and strong</td><td>About "
    "92–93% expected; the window allows for correlated rows</td>",
    "<td>Exactly weighted, test states whose command is outside the removed cluster: pooled 88–96% under mild and "
    "strong</td><td>About 92–93% is expected. The window is wide because rows from the same episode resemble each "
    "other, so coverage scatters more than it would for independent rows</td>")
add("<td>88 is 4 points below the no-shift level, beyond seed noise</td>",
    "<td>88% is 4 percentage points below the coverage expected with no shift, more than the seed-to-seed "
    "scatter</td>")
add("<tr><td>The missing action is flagged</td><td>The weighted band refuses every test step in the removed cluster "
    "and none outside; the unweighted band covers worse inside than outside</td><td>Shows over-confidence on the "
    "unseen common action</td>",
    "<tr><td>The missing command is flagged</td><td>The weighted band declines to give a band to every test state "
    "whose command is in the removed cluster, and to none outside it; the unweighted band covers worse inside than "
    "outside</td><td>Shows that the unweighted band is over-confident on the common command that training never "
    "saw</td>")
add("<td>Strong shift: within 2 points of exact-weight coverage</td><td>The classifier is the right shape and has 300 "
    "episodes a side</td>",
    "<td>Strong shift: within 2 percentage points of exact-weight coverage</td><td>The true weight depends on the "
    "current through a simple exponential, which the classifier's form can represent exactly, and it learns from "
    "300 episodes of each kind</td>")
# the weighting test's script
add("/* the weighting test: every number here is computed from closed forms (the tilted current laws), except the",
    "/* the weighting test: every number here is computed from closed forms (the leaning current laws), except the")
add("""  function niceMax(v) { const p = Math.pow(10, Math.floor(Math.log10(v))), f = v / p;
    return (f <= 1 ? 1 : f <= 2 ? 2 : f <= 2.5 ? 2.5 : f <= 5 ? 5 : 10) * p; }""",
    """  function niceMax(v) { const p = Math.pow(10, Math.floor(Math.log10(v))), f = v / p;
    return (f <= 1 ? 1 : f <= 2 ? 2 : f <= 2.5 ? 2.5 : f <= 5 ? 5 : 10) * p; }
  /* axis ticks that are round at the precision shown: a step of 1, 2, 2.5 or 5 times a power of ten that reaches max
     in 4 or 5 steps, printed with as many decimals as the step needs (so every label is the gridline's exact value) */
  function ticksTo(max) {
    const p = Math.pow(10, Math.floor(Math.log10(max / 4)));
    let s = 10 * p;
    for (const k of [1, 2, 2.5, 5, 10]) if (max / (k * p) <= 5 + 1e-9) { s = k * p; break; }
    let d = 0;
    while (d < 6 && Math.abs(Math.round(s * Math.pow(10, d)) - s * Math.pow(10, d)) > 1e-6) d++;
    const v = [];
    for (let i = 0, n = Math.round(max / s); i <= n; i++) v.push(i * s);
    return { v, d };
  }""")
add("""    const { X, Y } = frame(svg, W, H, m, [LO, HI], [0, ymax], CT, [0, ymax / 4, ymax / 2, 3 * ymax / 4, ymax],
      v => v.toFixed(1), v => v.toFixed(ymax < 2 ? 2 : 1));""",
    """    const tk = ticksTo(ymax);
    const { X, Y } = frame(svg, W, H, m, [LO, HI], [0, ymax], CT, tk.v, v => v.toFixed(1), v => v.toFixed(tk.d));""")
add("""    const { X, Y } = frame(svg, W, H, m, [LO, HI], [0, ymax], CT, [0, ymax / 4, ymax / 2, 3 * ymax / 4, ymax],
      v => v.toFixed(1), v => v.toFixed(ymax < 4 ? 1 : 0));""",
    """    const tk = ticksTo(ymax);
    const { X, Y } = frame(svg, W, H, m, [LO, HI], [0, ymax], CT, tk.v, v => v.toFixed(1), v => v.toFixed(tk.d));""")
add("el(\"title\", {}, r).textContent = `${NAME[k]}: ${GAP[k]}% of test steps use a removed command`;",
    "el(\"title\", {}, r).textContent = `${NAME[k]}: ${GAP[k]}% of test states use a removed command`;")
add('.textContent = "test tilt λ (training is −2)";', '.textContent = "test lean λ (training is −2)";')
add("l => `tilt ${sgn(l, 2)}<br>", "l => `lean ${sgn(l, 2)}<br>")
add('aria-label="Illustrative coverage of unweighted and weighted bands against the test tilt"',
    'aria-label="Illustrative coverage of unweighted and weighted bands against the test lean λ"')

# ---------------------------------------------------------------- Arguments, Version History, Names used in the code: F3, F17, F26, F29, F30, F37, F16, F19
add("Replay, the exact truth and the evaluation banks use it.",
    "Replaying a recorded episode, the exact simulation and the evaluation banks use it.")
add('<th scope="col">Situation (rule for drawing a start: Starting State)</th>',
    '<th scope="col">Situation (its rule for drawing a start is in the section "Starting State")</th>')
add("(the current runs from 0.1 to 1.1 with a known tilt) and a new recording controller whose common hard-water "
    "command is the 80% hold.",
    "(the current runs from 0.1 to 1.1, leaning toward weak or strong currents by a known amount) and a new recording "
    "controller whose common command in hard water (states where thruster 1 must push hard) is the 80% hold.")
add('Its four harm checks found that the shift causes real harm, but caution of BCA\'s kind cannot reach it (<a '
    'href="#results">Results</a>). It replaces the earlier linear-quadratic world (below) for two reasons found in the '
    'design work. In a world where every value is a quadratic, as in any world with linear dynamics and a quadratic '
    'cost, any patch of data fixes the critic everywhere, so missing data hide nothing;',
    'Three of its four harm checks ran: the first two found that the shift causes real harm, the third that caution '
    'of BCA\'s kind cannot reach it; the fourth never ran (<a href="#results">Results</a>). It replaces the earlier '
    'linear-quadratic world (below; linear dynamics with a quadratic cost, meaning a cost built from squares and '
    'products of the numbers, like this page\'s float cost) for two reasons found in the design work. In such a world '
    'every true value is itself a quadratic formula of the state and command, so any patch of data fixes the critic '
    'everywhere, and missing data hide nothing;')
add("Linear-quadratic means linear dynamics with a quadratic cost. The project's records call it",
    "The project's records call it")
add("<tr><td>pull past 80%</td>", "<tr><td>extra preference for full throttle</td>")
add("<tr><td>the recorded fields, in the order of The recorded data</td>",
    "<tr><td>the recorded fields, in the order of the table in the section \"The recorded data\"</td>")
add("900–902 (setting the 0.565 trigger),", "900–902 (setting the 0.565 threshold),")
add("<tr><td>the critic's pull μ (0.58), bending and straight-slope critics</td>",
    "<tr><td>the critic's pull μ (0.58), and the two critic shapes past 80% (curving down; keeping its slope)</td>")
add("(<code>B1</code>); <code>learners.<wbr>bca_band</code>",
    "(code label <code>B1</code> for the three detection checks together); <code>learners.<wbr>bca_band</code>")
add("<td>policy marks <code>P2, P3, P4</code> in <code>acceptance</code></td>",
    "<td>code labels <code>P2, P3, P4</code> in <code>acceptance</code> (unrelated to the full-throttle pattern "
    "P)</td>")
add("(short labels H1–H4 for the four harm checks, in the order of The harm checks)",
    "(short labels H1–H4 for the four harm checks, in the order of the section \"The harm checks\")")

# ---------------------------------------------------------------- found while reading the final screenshots
# the live panel's thruster arrows: a head only on a push long enough to carry one (a head on the box reads as a roof)
add("""      const dir = len < 0 ? -1 : 1;
      g.beginPath(); g.moveTo(mx - 5, yB - dir * 6); g.lineTo(mx, yB); g.lineTo(mx + 5, yB - dir * 6); g.stroke();""",
    """      const dir = len < 0 ? -1 : 1;
      if (Math.abs(len) >= 12) { g.beginPath(); g.moveTo(mx - 5, yB - dir * 6); g.lineTo(mx, yB); g.lineTo(mx + 5, yB - dir * 6); g.stroke(); }""")
# the weighting test's pass marks: one card per check at phone width, like the page's other key tables
add("<div class=\"tbl\"><table>\n      <thead><tr><th>Check</th><th>Passes when</th><th>Why this line</th><th>Status</th></tr></thead>",
    "<div class=\"tbl\"><table class=\"stack\">\n      <thead><tr><th>Check</th><th>Passes when</th><th>Why this line</th>"
    "<th>Status</th></tr></thead>")
add("<dd>The hard-water line. Above it the new recording controller flips a coin each step:",
    "<dd>The hard-water line. Above it the new recording controller flips a coin in each state:")

EDITS = E

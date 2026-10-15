r"""The plain-language edits for edit 3, as (old, new) pairs on the tokenised page (math and data are tokens).

Name table (one plain name per label; short codes survive only in Names used in the code, as a mapping):
  the gate                     -> the four harm checks
  H1 / H2 / H3 / H4            -> the critic-error check / the return-cost check / the best-case caution check /
                                  the seed-count check
  room-certified, certified    -> (a situation) where caution can help; "caution can help" / "caution cannot help"
  alpha ladder                 -> the critic-weight step-up rule;   the retune -> the steeper-stall rerun
  gate / confirmation seeds    -> harm-check seeds / BCA-test seeds;   gate report -> the harm-check report
  world card, T1..T9           -> the world checks: removal-target, stall-hidden, clear-choice, same-data, states-kept,
                                  copying-falls-short, stall-matters, learnable-values, forecast-harm
  D1 / D2 / D3                 -> the ranking / removal-region / beyond-noise check (the three detection checks)
  level 1 / level 2            -> the gain check / the placement check
  C1 / C2 / C3                 -> the no-loss / beats-copying / no-shift-harm check (and the harm-present check)
  verdicts 3 and 5             -> cited by their sentence
  gapped critic, full-data     -> the critic trained after the removal; the full-log critic; the log after the removal
  arm                          -> caution variant;  over-preference-placed -> harm-placed;  even -> uniform
  cap-2, factors up to 2       -> wider-range caution (doses up to 2);  edge-slope -> straight-slope critic
  shift card, column N         -> the shift table, the column by its name
  the design, scout, Spec      -> the design plan; a trial simulation; Specification
  the user, protocol           -> the project owner; the pre-registered plan
"""

EDITS = []


def e(old, new):
    EDITS.append((old, new))


# ---------------------------------------------------------------- style for the results written as problem/reasoning/outcome
e(r"""ol.marks { list-style: lower-alpha; }""",
  r"""ol.marks { list-style: lower-alpha; }
/* results written as problem, reasoning and outcome */
.finds .check { gap: 8px; }
.find-t { font-family: var(--display); font-size: 17px; font-weight: 700; color: var(--ink); line-height: 1.3; margin: 0; max-width: none; }
dl.pro dt { color: var(--accent-ink); }
dl.pro dd { color: var(--ink); }""")

# ---------------------------------------------------------------- header
e(r"""<p class="crumb">BCA test environments &nbsp;/&nbsp;""",
  r"""<p class="crumb">Test environments for Bayesian Conformal Aggregation (BCA) &nbsp;/&nbsp;""")

e(r"""<p class="status"><span class="tag warn">stopped before BCA</span> The <i>gate</i> has run on this environment:
      four harm checks, H1 to H4, that must pass before BCA is run at all. They were pre-registered: every rule and pass
      mark was written down and hashed before any result existed. A hash is a short fingerprint computed from a file;
      changing any character of the file changes it, so publishing the hash proves the file was not edited later. The
      shift causes real, large harm (H1 and H2 pass), but caution of the kind BCA applies cannot reach it (H3 certifies
      no situation), so the gate stopped before any BCA run. See <a href="#results">Results</a>; the checks are
      described in <a href="#harm-checks">The harm checks</a>.</p>""",
  r"""<p class="status"><span class="tag warn">stopped before BCA</span> <b>The problem:</b> a test of BCA means
      something only if the shift really harms the learned actor, and only if caution of the kind BCA applies could undo
      that harm at all. <b>The reasoning:</b> so before BCA is run at all, four <i>harm checks</i> must pass, each
      answering one question. Is the critic wrong where data are missing (the critic-error check)? Does an actor that
      trusts that critic lose return (the return-cost check)? Does caution placed exactly at the harm win back more than
      the same caution placed without that knowledge (the best-case caution check)? How many repeat runs, each with its
      own random seed, does a fair test need (the seed-count check)? The checks were pre-registered: every rule and pass
      mark was written down and hashed before any result existed. A hash is a short fingerprint computed from a file;
      changing any character of the file changes it, so publishing the hash proves the file was not edited later.
      <b>The outcome:</b> the first two checks passed, so the shift causes real, large harm. The third did not: caution
      of the kind BCA applies cannot reach that harm, in any situation, even when placed exactly where the harm is. So
      the harm checks stopped the test before any BCA run. See <a href="#results">Results</a>; the checks are described
      in <a href="#harm-checks">The harm checks</a>.</p>""")

e(r"""<p class="status"><span class="tag acc">next</span> The stop was not accepted. The next experiment on this environment is <a href="#weighting-test">The weighting test</a>: it checks that BCA's mechanics work on data where every value is known, and whether importance weighting toward a known test distribution improves on unweighted BCA.</p>""",
  r"""<p class="status"><span class="tag acc">next</span> The stop was not accepted. The next experiment on this
      environment is <a href="#weighting-test">The weighting test</a>. On data where every value is known, it checks
      that BCA's mechanics work. It also tests <i>importance weighting</i> toward a known test distribution: each
      recorded row that BCA calibrates on counts more or less according to how much more often its situation occurs at
      test than in training. The question is whether that improves on unweighted BCA.</p>""")

# ---------------------------------------------------------------- live panel (state is the measure, not time step)
e(r"""<figure class="panel" id="sim" aria-label="Live rollout of one episode">""",
  r"""<figure class="panel" id="sim" aria-label="Live replay of one episode">""")
e(r"""id="back" type="button" aria-label="Step back">""", r"""id="back" type="button" aria-label="Back one state">""")
e(r"""id="fwd" type="button" aria-label="Step forward">""", r"""id="fwd" type="button" aria-label="Forward one state">""")
e(r"""<input type="range" id="tSlider" min="0" max="99" value="0" aria-label="Time step">""",
  r"""<input type="range" id="tSlider" min="0" max="99" value="0" aria-label="State">""")
e(r"""<span class="tlabel" id="tLabel">t = 0</span>""", r"""<span class="tlabel" id="tLabel">state 0</span>""")
e(r"""<span class="k">time step ⟦M4|t⟧ (it counts from 0: ⟦M5|t = 0⟧ is the 1st of the 100 steps)</span>""",
  r"""<span class="k">state ⟦M4|t⟧ (counted from 0: ⟦M5|t = 0⟧ is the starting state, the 1st of the 100)</span>""")
e(r"""<span class="k">water's push ⟦M6|c + w_1⟧ (region above 0.565)</span>""",
  r"""<span class="k">water's push ⟦M6|c + w_1⟧ (full-throttle region above 0.565)</span>""")
e(r"""$("tLabel").textContent = "t = " + t;""", r"""$("tLabel").textContent = "state " + t;""")
e(r"""`Steps past 80%: ${sm.steps_past_80}; thruster-steps stalled: ${sm.stalled_thruster_steps}. ` +""",
  r"""`Steps past 80%: ${sm.steps_past_80}; stalled thruster-steps (one thruster stalled for one step counts 1): ${sm.stalled_thruster_steps}. ` +""")

# ---------------------------------------------------------------- the recording controller
e(r"""keeps every normal-driving throttle inside ±80%, so normal
      driving never stalls.</p>""",
  r"""keeps every normal-driving throttle inside ±80%, so normal
      driving never stalls (tanh is a smooth S-shaped function that turns any number into one between −1 and 1).</p>""")
e(r"""The trigger 0.565 was set by <i>the design</i>: the plan for this environment and its test, written before the
      pre-registration and checked""",
  r"""The trigger 0.565 was set by <i>the design plan</i>: the written plan for this environment and its test, made
      before the pre-registration and checked""")
e(r"""The
      design chose the trigger so that""", r"""The
      design plan chose the trigger so that""")
e(r"""the harm checks (gate seeds 0–9) and the BCA test (confirmation seeds from 100) never use them.""",
  r"""the harm checks (harm-check seeds 0–9) and the BCA test (BCA-test seeds from 100) never use them.""")

# ---------------------------------------------------------------- the distribution shift
e(r"""the gate's own seeds are measured the same way.""",
  r"""the seeds of the harm checks (<a href="#harm-checks">The harm checks</a>) are measured the same way.""")
e(r"""(<a href="#shift-card">the shift
      card</a>, column 5): on 17.2% of its""",
  r"""(the column "Command in the removed ball" of
      <a href="#shift-card">the shift table</a>): on 17.2% of its""")
e(r"""For H1's
      pull-past-80% check (<a href="#methods">Methods</a>) this means that both commands it compares, full throttle
      and the 80% hold, are missing from the data after the removal. The check still isolates the removal: it compares
      the critic after the removal with the full-log critic on the same two commands, and the full log holds commands
      near both.</p>""",
  r"""For the
      critic-error check's pull-past-80% rule (<a href="#harm-checks">The harm checks</a>; the measure is defined in
      <a href="#methods">Methods</a>) this means that both commands it compares, full throttle and the 80% hold, are
      missing from the data after the removal. The check still isolates the removal: it compares the critic trained
      after the removal with the <i>full-log critic</i> (trained on the full log, with nothing removed) on the same two
      commands, and the full log holds commands near both.</p>""")
e(r"""or even caution that changes the learned actor just as much""",
  r"""or uniform caution (the same dose at every row) that changes the learned actor just as much""")
e(r"""<h3 id="shift-card">How the shift is measured: the shift card</h3>""",
  r"""<h3 id="shift-card">How the shift is measured: the shift table</h3>""")
e(r"""<li><b>Outside the recorded conditions</b>, as defined in Setting 3: a state counts""",
  r"""<li><b>Outside the recorded conditions</b>, as defined above under "where the learned actor is tested": a state
        counts""")
e(r"""In the recorded conditions about 1% of visited states count as outside (column 4) even
      though""",
  r"""In the recorded conditions about 1% of visited states count as outside the recorded
      conditions (the column of that name) even though""")
e(r"""<p>In strong current every visited state is outside the recorded conditions (column 4) and
      93.7% are in the full-throttle region
      (column 3). The rising-thrust controller pushes past 80% on
      84.0% of its steps there (column 2), none
      of them inside the recorded conditions (column 7), and loses
      154.1 points by it (column 8).""",
  r"""<p>Read the strong-current rows. Every visited state is outside the recorded conditions, and
      93.7% are in the full-throttle region. The rising-thrust controller pushes past 80% on
      84.0% of its steps there (the column "Steps past 80%"), none of them inside the recorded conditions (its covered
      share is 0%), and loses 154.1 points by it (the column "Loss from pushing past 80%").""")

# ---------------------------------------------------------------- methods
e(r"""5 and 10 on the <a href="#alpha-ladder">alpha ladder</a>""",
  r"""5 and 10 under the <a href="#alpha-ladder">critic-weight step-up rule</a>""")
e(r"""The design measured the two ingredients on""", r"""The design plan measured the two ingredients on""")
e(r"""The <i>world card</i> (the checks T1–T9 on the world itself, which need no
      learning, run on design seeds 900–904 before the pre-registration; see <a href="#results">Results</a>) remeasures""",
  r"""The <i>world checks</i> (nine checks on the world itself, which need no
      learning, run on design seeds 900–904 before the pre-registration; see <a href="#results">Results</a>) remeasure""")
e(r"""so the values differ slightly from the design's.</p>""",
  r"""so the values differ slightly from the design plan's.</p>""")
e(r"""Two shapes bracket it, and the design uses both:</p>""", r"""Two shapes bracket it, and the design plan uses both:</p>""")
e(r"""<li>A critic that <i>bends like the truth past 80%</i> keeps its hill:""",
  r"""<li>A critic that <i>bends like the truth past 80%</i> (the <i>bending critic</i>) keeps its hill:""")
e(r"""<li>A critic that <i>keeps its slope past the edge of the data</i> has no curvature beyond the recorded commands.""",
  r"""<li>A critic that <i>keeps its slope past the edge of the data</i> (the <i>straight-slope critic</i>) has no
        curvature beyond the recorded commands.""")
e(r"""Edge-slope critic: share min(μ / m, 1)""", r"""Straight-slope critic: share min(μ / m, 1)""")
e(r"""for the edge-slope critic. Even BCA's strongest dose""", r"""for the straight-slope critic. Even BCA's strongest dose""")
e(r"""<b>From shares to points: the design's forecast.</b> For each critic shape the design built a stand-in for the""",
  r"""<b>From shares to points: the design plan's forecast.</b> For each critic shape the design plan built a stand-in
      for the""")
e(r"""0.580 (edge slope) of the way""", r"""0.580 (straight slope) of the way""")
e(r"""Forecast harm, edge-slope critic (points)""", r"""Forecast harm, straight-slope critic (points)""")
e(r"""13.8% with the edge-slope critic, and in the recorded conditions capping""",
  r"""13.8% with the straight-slope critic, and in the recorded conditions capping""")
e(r"""<span class="tag acc">used by harm check H1</span>""", r"""<span class="tag acc">used by the critic-error check</span>""")
e(r"""<span class="tag acc">used by harm check H2</span>""", r"""<span class="tag acc">used by the return-cost check</span>""")
e(r"""<span class="tag acc">used by harm check H3</span>""",
  r"""<span class="tag acc">used by the best-case caution check</span>""")
e(r"""lie inside the removed ball (<a href="#shift">Setting 1</a>), so the critic
      after the removal has seen neither near these states;""",
  r"""lie inside the removed ball (<a href="#shift">The distribution shift</a>,
      "which training rows are removed"), so the critic trained after the removal has seen neither near these states;""")
e(r"""<li><b>Over-preference-placed caution</b>, BCA's own dose form placed by the exact over-preference:""",
  r"""<li><b>Harm-placed caution</b>, BCA's own dose form placed by the exact over-preference, the harm the critic's
        error does at that row:""")
e(r"""<li><b>Cap-2 caution</b> (a diagnostic):""", r"""<li><b>Wider-range caution</b> (a diagnostic):""")
e(r"""<li><b>Even caution</b>: the same multiplier at every row,""", r"""<li><b>Uniform caution</b>: the same multiplier at every row,""")
e(r"""every row gets the mean over-preference-placed dose of the same refresh.""",
  r"""every row gets the mean harm-placed dose of the same refresh.""")
e(r"""<li><b>Shuffled caution</b>: the over-preference-placed doses put in a fixed random order""",
  r"""<li><b>Shuffled caution</b>: the harm-placed doses put in a fixed random order""")
e(r"""<li><b>Reversed caution</b> (a sanity check, reported beside H3 but not a pass rule): the over-preference-placed
        doses with their ranks flipped, so the largest dose lands on the row with the least over-preference. If where
        caution acts matters, reversed caution must score below over-preference-placed caution; H3 reports reversed minus
        over-preference-placed, which should be negative.</li>""",
  r"""<li><b>Reversed caution</b> (a sanity check, reported beside the best-case caution check but not a pass rule):
        the harm-placed doses with their ranks flipped, so the largest dose lands on the row with the least
        over-preference. If where caution acts matters, reversed caution must score below harm-placed caution; the check
        reports reversed minus harm-placed, which should be negative.</li>""")
e(r"""<li><b>Strength-matched even caution</b>: even caution exactly as strong as the over-preference-placed caution.""",
  r"""<li><b>Strength-matched uniform caution</b>: uniform caution exactly as strong as the harm-placed caution.""")
e(r"""compared with the run without caution, on 5,000 states of the held-out split. The even
        multiplier with the same""",
  r"""compared with the run without caution, on 5,000 states of the held-out split. The uniform
        multiplier with the same""")
e(r"""the over-preference-placed dose in BCA&#x27;s form rises from 1 toward 1.5, the cap-2 diagnostic from 1 toward 2.""",
  r"""the harm-placed dose in BCA&#x27;s form rises from 1 toward 1.5, the wider-range diagnostic from 1 toward 2.""")
e(r"""<text class="ann" x="422" y="22.1" text-anchor="end">cap 2</text>""",
  r"""<text class="ann" x="422" y="22.1" text-anchor="end">wider range, up to 2</text>""")
e(r"""over-preference-placed dose, BCA's form, [1, 1.5)</span>""", r"""harm-placed dose, BCA's form, [1, 1.5)</span>""")
e(r"""transparent 6px 10px)"></i>cap-2 diagnostic</span>""",
  r"""transparent 6px 10px)"></i>wider-range diagnostic, up to 2</span>""")
e(r"""<th class="n" scope="col">Over-preference-placed dose (BCA's form)</th><th class="n" scope="col">Cap-2 dose (diagnostic)</th>""",
  r"""<th class="n" scope="col">Harm-placed dose (BCA's form)</th><th class="n" scope="col">Wider-range dose (diagnostic, up to 2)</th>""")

# ---------------------------------------------------------------- the harm checks
e(r"""Before BCA is judged, the test must show three things: the shift makes the critic
      wrong, acting on that error costs return, and caution placed by the exact over-preference could help. Each check runs on gate
      seeds 0–4 (the data, the initial parameters and the batch stream all change with the seed), with pass rules fixed
      in the pre-registration. Together, H1 to H4 are the <i>gate</i>: BCA runs only if it passes.""",
  r"""Before BCA is judged, the test must show three things: the shift makes the critic
      wrong (the <i>critic-error check</i>), acting on that error costs return (the <i>return-cost check</i>), and
      caution placed by the exact over-preference could help (the <i>best-case caution check</i>). A fourth, the
      <i>seed-count check</i>, then sets how many repeat runs a fair BCA test needs. Each check runs on
      <i>harm-check seeds</i> 0–4 (a seed fixes every random draw of a run: the data, the initial parameters and the
      batch stream all change with the seed), with pass rules fixed in the pre-registration. BCA runs only if all four
      harm checks pass.""")
e(r"""<h3>H1. Is the critic wrong where data are missing?</h3>""",
  r"""<h3>The critic-error check: is the critic wrong where data are missing?</h3>""")
e(r"""At the removed commands, the critic from the log after the removal is at least twice as wrong as the critic
            from the full log:""",
  r"""At the removed commands, the critic trained after the removal is at least twice as wrong as the full-log
            critic:""")
e(r"""the ratio in (a) is at least 1.5 times""", r"""the ratio in the first rule is at least 1.5 times""")
e(r"""the median pull past 80% of the
            critic after the removal is positive""",
  r"""the median pull past 80% of the
            critic trained after the removal is positive""")
e(r"""<h3>H2. Does acting on that error cost return?</h3>""",
  r"""<h3>The return-cost check: does acting on that error cost return?</h3>""")
e(r"""The actor that trusted the critic from the log after the removal scores at least 5 points below""",
  r"""The actor that trusted the critic trained after the removal scores at least 5 points below""")
e(r"""<p id="alpha-ladder"><b>The alpha ladder rule</b> (an amendment to the design, fixed before any outcome). H2 runs first at TD3+BC's
          published <span class="nw">⟦M248|\alpha = 2.5⟧.</span> If it fails there, it reruns at <span class="nw">⟦M249|\alpha = 5⟧,</span> and if it fails again, at
          <span class="nw">⟦M250|\alpha = 10⟧.</span> H1 does not depend on <span class="nw">⟦M251|\alpha⟧.</span> The first ⟦M252|\alpha⟧ at which H1 and H2 both pass is used for
          H3, H4 and every later run, the same for every comparison. The choice never looks at any BCA output.</p>""",
  r"""<p id="alpha-ladder"><b>The critic-weight step-up rule</b> (an amendment to the design plan, fixed before any
          outcome). The critic weight sets how much the critic counts against the pull toward the recorded commands
          (<a href="#methods">Methods</a>). The return-cost check runs first at TD3+BC's published
          <span class="nw">⟦M248|\alpha = 2.5⟧.</span> If it fails there, it reruns at <span class="nw">⟦M249|\alpha = 5⟧,</span> and if it
          fails again, at <span class="nw">⟦M250|\alpha = 10⟧.</span> The critic-error check does not depend on
          <span class="nw">⟦M251|\alpha⟧.</span> The first ⟦M252|\alpha⟧ at which both checks pass is used for the best-case caution check,
          the seed-count check and every later run, the same for every comparison. The choice never looks at any BCA
          output.</p>""")
e(r"""<h3>H3. Could caution help?</h3>""",
  r"""<h3>The best-case caution check: could caution placed exactly at the harm help?</h3>""")
e(r"""<p>A shifted situation is <i>room-certified</i> when over-preference-placed caution beats each of strength-matched even
          caution, amount-matched caution and the shuffled doses by at least 3 points (pooled, lower bound above 0,
          ahead in at least 4 of 5 seeds), and costs at most 1 point against no caution in the recorded conditions. At
          least one shifted situation must be room-certified. The best even caution, the cap-2 result and reversed
          caution minus over-preference-placed caution are shown beside it. BCA is judged only on room-certified situations.</p>""",
  r"""<p>Caution <i>can help</i> in a shifted situation when harm-placed caution (placed by the exact over-preference)
          beats each of strength-matched uniform caution, amount-matched caution and the shuffled doses by at least 3
          points (pooled, lower bound above 0, ahead in at least 4 of 5 seeds), and costs at most 1 point against no
          caution in the recorded conditions. At least one shifted situation must pass. The best uniform caution, the
          wider-range result (doses up to 2) and reversed caution minus harm-placed caution are shown beside it. BCA is
          judged only on situations where caution can help.</p>""")
e(r"""<h3>H4. How many seeds?</h3>
        <p class="uses">Uses plain TD3+BC runs on the log after the removal in which critic and actor learn together
          (co-trained), unlike the actor-only runs of H2 and H3.</p>
        <p>Two paired differences are measured on the gate seeds.""",
  r"""<h3>The seed-count check: how many repeat runs does a fair test need?</h3>
        <p class="uses">Uses plain TD3+BC runs on the log after the removal in which critic and actor learn together,
          unlike the actor-only runs of the return-cost and best-case caution checks.</p>
        <p>Two paired differences are measured on the harm-check seeds.""")
e(r"""and even
          caution ×1.25 minus plain. Their larger standard deviation over the seeds, ⟦M254|\sigma⟧ in points on the
          room-certified situations, sets the number of confirmation seeds""",
  r"""and uniform
          caution ×1.25 minus plain. Their larger standard deviation over the seeds, ⟦M254|\sigma⟧ in points on the
          situations where caution can help, sets the number of BCA-test seeds""")
e(r"""over the three tests of level 1 in <a href="#bca-test">The BCA test</a>""",
  r"""over the three tests of the gain check in <a href="#bca-test">The BCA test</a>""")
e(r"""Confirmation seeds N</th>""", r"""BCA-test seeds N</th>""")
e(r"""<li>If H1 fails, or H2 fails at every ⟦M263|\alpha⟧ of the ladder, the one pre-registered retune runs: stall slope
          3.0 (push 0.2 at full throttle), the world card (the checks T1–T9 on the world itself, which need no learning;
          see <a href="#results">Results</a>) on design seeds 905–909, and the gate on fresh seeds 5–9 with the same
          ladder.""",
  r"""<li>If the critic-error check fails, or the return-cost check fails at every ⟦M263|\alpha⟧ of the step-up rule, the
          one pre-registered <i>steeper-stall rerun</i> runs: stall slope 3.0 (push 0.2 at full throttle), the nine
          world checks (checks on the world itself, which need no learning; see <a href="#results">Results</a>) on
          design seeds 905–909, and the four harm checks on fresh seeds 5–9 with the same step-up rule.""")
e(r"""<li>If no shifted situation is room-certified, the test stops and shows the cap-2 result, so the reader can see
          whether BCA's dose range is the limit.</li>""",
  r"""<li>If caution cannot help in any shifted situation, the test stops and shows the wider-range result (doses up
          to 2), so the reader can see whether BCA's dose range is the limit.</li>""")

# ---------------------------------------------------------------- the BCA test
e(r"""<p>Only after the user approves. The gate report is written, and the user approves it by typing the start of its
      hash""",
  r"""<p>The BCA test runs only after the project owner approves. The harm-check report is written, and the project
      owner approves it by typing the start of its hash""")
e(r"""Then, on fresh confirmation seeds 100 to 99 +""", r"""Then, on fresh BCA-test seeds 100 to 99 +""")
e(r"""<li>even caution ×1.25 and ×2, and even caution matched in strength to BCA;</li>""",
  r"""<li>uniform caution ×1.25 and ×2, and uniform caution matched in strength to BCA;</li>""")
e(r"""(no shift; the shift-specificity check D2 below uses this BCA run)""",
  r"""(no shift; the removal-region check below uses this BCA run)""")
e(r"""It is reported beside the ranking check D1, not judged""", r"""It is reported beside the ranking check, not judged""")
e(r"""command (the detection checks D1–D3, defined below)""", r"""command (the three detection checks, defined below)""")
e(r"""TD3+BC with BCA beats plain TD3+BC in a
          room-certified shifted situation, such as strong current (level 1), and also beats the same doses placed
          blindly and even caution of the same strength (level 2).""",
  r"""TD3+BC with BCA beats plain TD3+BC in a
          shifted situation where caution can help, such as strong current (the gain check), and also beats the same
          doses placed blindly and uniform caution of the same strength (the placement check).""")
e(r"""No loss in the recorded conditions (check C1), at least
          5 points better than behaviour cloning, which copies the recorded commands (C2), and no harm when nothing is
          removed (C3).""",
  r"""No loss in the recorded conditions (the no-loss check), at
          least 5 points better than behaviour cloning, which copies the recorded commands (the beats-copying check), and
          no harm when nothing is removed (the no-shift-harm check).""")
e(r"""in it means passing C2, beating a copy of the recorded commands.""",
  r"""in it means passing the beats-copying check, beating a copy of the recorded commands.""")
e(r"""<p>Three tests judged together under <i>Holm's correction</i>, a correction for testing several things at once that
      keeps the family's false-alarm rate at 5% (one-sided).""",
  r"""<p>The three detection checks are judged together under <i>Holm's correction</i>, a correction for testing several
      things at once that keeps the chance of any false alarm among them at 5% (one-sided).""")
e(r"""<li><b>D1, ranking.</b> A row among""",
  r"""<li><b>Ranking check: are BCA's largest doses on the most over-preferred rows?</b> A row among""")
e(r"""<li><b>D2, specific to the shift.</b> BCA's mean dose""",
  r"""<li><b>Removal-region check: does BCA's caution gather where the removal took commands?</b> BCA's mean dose""")
e(r"""<li><b>D3, beyond target noise.</b> The dose still follows""",
  r"""<li><b>Beyond-noise check: does the dose follow over-preference, not just target noise?</b> The dose still
        follows""")
e(r"""<p><b>Reported beside D1, not judged:</b>""", r"""<p><b>Reported beside the ranking check, not judged:</b>""")
e(r"""the same plain critic on which H3
      measured whether caution could help.</p>""",
  r"""the same plain critic on which the
      best-case caution check measured whether caution could help.</p>""")
e(r"""<li><b>Level 1.</b> BCA minus plain TD3+BC is at least 2 points on a room-certified shifted situation, significant
        within the Holm family of those situations.</li>
      <li><b>Level 2.</b> On each situation that passed level 1, BCA also beats its own doses spread evenly, its doses
        shuffled, and even caution of the same strength (one Holm family).</li>
      <li><b>C1, no loss in the recorded conditions:</b> the lower bound of BCA minus plain is above −2 points.</li>
      <li><b>C2, beats copying the recording:</b> in the recorded conditions, BCA minus behaviour cloning is at least 5
        points, lower bound above 0.</li>
      <li><b>C3, no harm without the shift:</b> on the full log, BCA minus plain has a lower bound above −2 points in the
        recorded conditions and in every room-certified situation.</li>
      <li><b>Harm present:</b> in every room-certified situation, averaged over the confirmation seeds, the plain TD3+BC
        actor still loses at least 3 points to pushing past 80%: the same actor with every throttle capped at 80% scores
        at least 3 points more (<a href="#shift-card">shift-card</a> measure 8).</li>""",
  r"""<li><b>Gain check: does BCA beat plain TD3+BC where caution can help?</b> BCA minus plain TD3+BC is at least 2
        points in a shifted situation where caution can help, significant after Holm's correction across those
        situations.</li>
      <li><b>Placement check: does the gain come from where BCA places its caution?</b> In each situation that passed
        the gain check, BCA also beats its own doses spread evenly, its doses shuffled, and uniform caution of the same
        strength (these comparisons corrected together, with Holm's correction).</li>
      <li><b>No-loss check: no loss in the recorded conditions.</b> The lower bound of BCA minus plain is above −2
        points.</li>
      <li><b>Beats-copying check: still better than copying the recording.</b> In the recorded conditions, BCA minus
        behaviour cloning is at least 5 points, lower bound above 0.</li>
      <li><b>No-shift-harm check: no harm without the shift.</b> On the full log, BCA minus plain has a lower bound above
        −2 points in the recorded conditions and in every situation where caution can help.</li>
      <li><b>Harm-present check: is there still harm to remove?</b> In every situation where caution can help, averaged
        over the BCA-test seeds, the plain TD3+BC actor still loses at least 3 points to pushing past 80%: the same
        actor with every throttle capped at 80% scores at least 3 points more (the loss-from-pushing-past-80% measure of
        <a href="#shift-card">the shift table</a>).</li>""")
e(r"""<thead><tr><th>Verdict</th><th>When</th></tr></thead>""",
  r"""<thead><tr><th>Verdict, in the order checked</th><th>When</th></tr></thead>""")
e(r"""<td>C1 or C3 fails, or BCA minus plain is at most −2 points with a 95% upper bound below 0 on a room-certified situation.</td>""",
  r"""<td>The no-loss or no-shift-harm check fails, or BCA minus plain is at most −2 points with a 95% upper bound below 0 in a situation where caution can help.</td>""")
e(r"""<td>The plain run loses less than 3 points to pushing past 80% on the confirmation seeds (the harm-present check fails), or a level-1 difference's 95% interval is wider than 5 points either side.</td>""",
  r"""<td>The plain run loses less than 3 points to pushing past 80% on the BCA-test seeds (the harm-present check fails), or a gain-check difference's 95% interval is wider than 5 points either side.</td>""")
e(r"""<td>Detection passes, levels 1 and 2 pass on at least one situation, and C1, C2 and C3 pass.</td>""",
  r"""<td>The detection checks pass, the gain and placement checks pass in at least one situation, and the no-loss, beats-copying and no-shift-harm checks pass.</td>""")
e(r"""<td>Levels 1 and 2 pass on at least one situation but detection fails (adding "it also does not improve on the recording by the required 5 points" when C2 fails too).</td>""",
  r"""<td>The gain and placement checks pass in at least one situation but the detection checks fail (adding "it also does not improve on the recording by the required 5 points" when the beats-copying check fails too).</td>""")
e(r"""<td>Levels 1 and 2 pass on at least one situation and detection passes, but C2 fails.</td>""",
  r"""<td>The gain and placement checks pass in at least one situation and the detection checks pass, but the beats-copying check fails.</td>""")
e(r"""<td>Level 1 passes somewhere, and level 2 fails wherever level 1 passed.</td>""",
  r"""<td>The gain check passes somewhere, and the placement check fails wherever the gain check passed.</td>""")
e(r"""<td>Detection passes and level 1 fails everywhere.</td>""",
  r"""<td>The detection checks pass and the gain check fails everywhere.</td>""")
e(r"""When placing the same doses by the noise map scores within 2 points of BCA, or better, verdicts 3
      and 5 (the two that say BCA finds the harm and that its placed caution improves returns out of distribution) add:""",
  r"""When placing the same doses by the noise map scores within 2 points of BCA, or better, the two
      verdicts that say BCA finds the harm and that its placed caution improves returns out of distribution add:""")

# ---------------------------------------------------------------- results
e(r"""<p class="status"><span class="tag warn">stopped before BCA</span> The gate stopped where its pre-registered rule
      says it must: before any BCA run. The reason is "no room": No shifted situation is room-certified: caution placed by the exact harm, inside BCA's dose range, does not beat even caution there, so no BCA result could be informative.</p>
    <p>In plain words, three things were found, in this order.</p>""",
  r"""<p class="status"><span class="tag warn">stopped before BCA</span> <b>The problem:</b> BCA's only tool is extra
      pull toward the recorded commands on chosen training rows, so a test of BCA means something only if that tool can
      help here at all. <b>The reasoning:</b> before BCA is judged, the best-case caution check gives that tool its best
      possible aim, the exact harm, and asks whether it then beats the same caution spread uniformly over every row.
      <b>The outcome:</b> it does not. In no shifted situation did caution placed by the exact harm, inside BCA's dose
      range, beat uniform caution, so no BCA result could have been informative. The test stopped before any BCA run,
      where its pre-registered rule says it must.</p>
    <p>In plain words, three things were found, in this order. Each answers one problem, by one line of
      reasoning.</p>""")
e(r"""<ol>
      <li><b>The environment works as built.</b> Removing the most common command makes the critic wrong exactly at the
        removed commands, and nowhere else. Random removal, another starting point and the no-stall comparison world
        explain none of it (H1, all six rules pass).</li>
      <li><b>Acting on that error costs a great deal under shift.</b> In strong current the actor of the gapped critic
        scores 99 points lower than the actor of the full-data critic: almost the whole distance from the stall-aware
        controller (100) to the recording controller (0). The loss sits at the steps that push past 80% (H2, all five
        rules pass at <span class="nw">⟦M268|\alpha = 5⟧).</span></li>
      <li><b>Caution of the kind BCA applies cannot recover it.</b> Even placed by the exact harm, a stronger pull
        toward the recorded commands on the training rows, inside BCA's dose range, beats even caution by under 1 point
        in every situation, and in strong current it does slightly worse (H3: no situation certified).</li>
    </ol>
    <p>H3 places caution by the exact harm, which is better information than any signal BCA could learn. So the stop
      says something about the mechanism, not about BCA's signal. No amount of extra pull toward the recorded commands
      on the training rows, in BCA's range, moves this harm, however well the harm is found. Whether BCA finds the
      harmful rows (its detection) was not tested: the protocol opens that only after the gate passes and the user
      approves.</p>""",
  r"""<div class="checks finds">
      <div class="check">
        <p class="find-t">1. The environment works as built.</p>
        <dl class="kv pro">
          <dt>Problem</dt><dd>If removing the most common command did not mislead the critic, there would be no error
            for BCA to find.</dd>
          <dt>Reasoning</dt><dd>Critics were trained on data that differ only in what was removed, and each was compared
            with the exact truth. Random removal, another starting point and the no-stall comparison world are the
            controls: each rules out one other cause.</dd>
          <dt>Outcome</dt><dd>Removing the most common command makes the critic wrong exactly at the removed commands,
            and nowhere else, and none of the controls explains it. The critic-error check passed all six of its
            rules.</dd>
        </dl>
      </div>
      <div class="check">
        <p class="find-t">2. Acting on that error costs a great deal under shift.</p>
        <dl class="kv pro">
          <dt>Problem</dt><dd>A wrong critic matters only if an actor that trusts it does worse.</dd>
          <dt>Reasoning</dt><dd>Actors were trained from the same start against frozen critics that differ only in their
            data, so any difference in their returns comes from the critic.</dd>
          <dt>Outcome</dt><dd>In strong current, the actor that trusted the critic trained after the removal scores 99
            points lower than the actor that trusted the full-log critic: almost the whole distance from the stall-aware
            controller (100) to the recording controller (0). The loss sits at the steps that push past 80%. The
            return-cost check passed all five of its rules at critic weight <span class="nw">⟦M268|\alpha = 5⟧.</span></dd>
        </dl>
      </div>
      <div class="check">
        <p class="find-t">3. Caution of the kind BCA applies cannot recover that loss.</p>
        <dl class="kv pro">
          <dt>Problem</dt><dd>BCA's only tool is a stronger pull toward the recorded commands on chosen training rows,
            with doses between 1 and 1.5 (BCA's dose range). Before testing whether BCA aims that pull well, the test
            must know whether the pull can help even when it is aimed perfectly.</dd>
          <dt>Reasoning</dt><dd>So the pull was placed by the exact harm (each training row's over-preference: how much
            more the critic prefers the actor's command over the recorded one than the truth does; see
            <a href="#methods">Methods</a>), a better aim than any signal BCA could learn, and compared with the same
            caution placed without that knowledge.</dd>
          <dt>Outcome</dt><dd>Even placed by the exact harm, inside BCA's dose range, it beats uniform caution by under 1
            point in every situation, and in strong current it does slightly worse. The best-case caution check found no
            situation where caution can help.</dd>
        </dl>
      </div>
      <div class="check">
        <p class="find-t">What the stop means</p>
        <dl class="kv pro">
          <dt>Problem</dt><dd>A stop before BCA could have two causes: BCA's signal could be poor, or caution of BCA's
            kind could be unable to work here.</dd>
          <dt>Reasoning</dt><dd>The best-case caution check placed caution by the exact harm, which is better information
            than any signal BCA could learn, so a poor signal cannot be the cause.</dd>
          <dt>Outcome</dt><dd>The stop says something about the mechanism, not about BCA's signal. No amount of extra
            pull toward the recorded commands on the training rows, in BCA's range, moves this harm, however well the
            harm is found. Whether BCA finds the harmful rows (its detection) was not tested: the pre-registered plan
            allows that test only after all four harm checks pass and the project owner approves.</dd>
        </dl>
      </div>
    </div>""")
e(r"""<p>Timeline (12 October 2026, UTC):</p>
    <ul>
      <li>15:07: the build checks passed. A first attempt had failed on one test that wrongly required the gate's
        output folder not to exist.""",
  r"""<p>Timeline (12 October 2026; times in Coordinated Universal Time, UTC):</p>
    <ul>
      <li>15:07: the code's automatic tests passed. A first attempt had failed on one test that wrongly required the
        harm checks' output folder not to exist.""")
e(r"""and a
        copy was committed to the repository (commit <code>bdb8c8a</code>) before any learner ran.</li>
      <li>16:51: <b>H1 passed</b>, all six rules, on gate seeds 0–4.</li>
      <li>19:24: <b>H2 failed at ⟦M269|\alpha = 2.5⟧ and passed at ⟦M270|\alpha = 5⟧</b>. By the pre-registered ladder rule,
        ⟦M271|\alpha = 5⟧ is used for every later check, the same for every method.</li>
      <li>20:39: <b>H3 certified no situation</b>, so H4 (how many seeds a fair test needs) refused to run, as its rule
        requires. The gate stopped.</li>
    </ul>
    <h3>H1: is the critic wrong where data are missing?</h3>
    <p>Critics that each score the stall-aware controller, trained on data that differ only in what was removed. The
      numbers are the critic's error on each set of rows, divided by the error of the critic trained on the full data
      (seeds 0–4).</p>""",
  r"""and a
        copy was saved in the repository as a fixed version (commit <code>bdb8c8a</code>) before any learner ran.</li>
      <li>16:51: <b>the critic-error check passed</b>, all six rules, on harm-check seeds 0–4.</li>
      <li>19:24: <b>the return-cost check failed at ⟦M269|\alpha = 2.5⟧ and passed at ⟦M270|\alpha = 5⟧</b>. By the pre-registered
        critic-weight step-up rule, ⟦M271|\alpha = 5⟧ is used for every later check, the same for every method.</li>
      <li>20:39: <b>the best-case caution check found no situation where caution can help</b>, so the seed-count check
        (how many seeds a fair test needs) refused to run, as its rule requires. The harm checks stopped the test.</li>
    </ul>
    <h3>The critic-error check: is the critic wrong where data are missing?</h3>
    <p><b>Problem:</b> the removal should make the critic wrong at the removed commands and only there, for a reason
      none of the controls explains. <b>Reasoning:</b> critics that each score the stall-aware controller were trained on
      data that differ only in what was removed, so any difference in their errors comes from the data. The numbers are
      the critic's error on each set of rows, divided by the error of the critic trained on the full data (seeds
      0–4).</p>""")
e(r"""<tr><td>Pull past 80% in the full-throttle region: gapped critic, full critic</td><td class="n">+0.33 to +0.36; −0.06 to −0.01</td><td>gapped exceeds full by a quarter of the true price, in 4 of 5 seeds</td>""",
  r"""<tr><td>Pull past 80% in the full-throttle region: critic trained after the removal; full-log critic</td><td class="n">+0.33 to +0.36; −0.06 to −0.01</td><td>the critic trained after the removal exceeds the full-log critic by a quarter of the true price, in 4 of 5 seeds</td>""")
e(r"""<h3>H2: does acting on the error cost return?</h3>
    <p>Actors trained from the same cloned start against frozen critics that differ only in their data, scored on the
      strong-current bank of 1,000 starts. Points: 0 is the recording controller, 100 the stall-aware controller.</p>""",
  r"""<p><b>Outcome:</b> every rule passes. At the removed commands the critic trained after the removal is more than
      twice as wrong as the full-log critic; far from the removed ball it is no more wrong; random removal and another
      starting point explain none of it; the stall makes it worse; and it wrongly prefers pushing past 80%.</p>
    <h3>The return-cost check: does acting on the error cost return?</h3>
    <p><b>Problem:</b> a critic that is wrong matters only if an actor that trusts it does worse. <b>Reasoning:</b>
      actors were trained from the same cloned start against frozen critics that differ only in their data, so any
      difference in return is caused by the critic. They are scored on the strong-current evaluation bank of 1,000
      starts. Points: 0 is the recording controller, 100 the stall-aware controller.</p>""")
e(r"""<th class="n">⟦M272|\alpha⟧ 2.5: full-data critic</th><th class="n">⟦M273|\alpha⟧ 2.5: gapped critic</th><th class="n">⟦M274|\alpha⟧ 5: full-data critic</th><th class="n">⟦M275|\alpha⟧ 5: gapped critic</th>""",
  r"""<th class="n">⟦M272|\alpha⟧ 2.5: actor of the full-log critic</th><th class="n">⟦M273|\alpha⟧ 2.5: actor of the critic trained after the removal</th><th class="n">⟦M274|\alpha⟧ 5: actor of the full-log critic</th><th class="n">⟦M275|\alpha⟧ 5: actor of the critic trained after the removal</th>""")
e(r"""<ul>
      <li><b>At ⟦M277|\alpha = 2.5⟧</b> the loss was already 75–93 points, and four of the five rules passed. The one that
        failed asks the gapped critic's actor to push past 80% at least 1.5 times as often as the full-data critic's
        actor: it did so 1.46 times as often, because at this pull toward the recorded commands both actors push past
        80% often in strong current.</li>
      <li><b>At ⟦M278|\alpha = 5⟧</b> every rule passes. The loss is 95–103 points. Removing the same number of rows at
        random changes the score by −10 to +11 points, and the same removal in the no-stall comparison world by −1 to
        +2 points. The gapped critic's actor pushes past 80% 1.77 times as often (pooled), and the lost return sits at
        its past-80% steps. With more freedom, the actor of the full-data critic, whose data show the stall, learns to
        stay at or below 80%, while the actor of the gapped critic, which never saw it, pushes further past it.</li>
      <li>In the other situations the loss is smaller: 4–7 points in the recorded conditions, 13–18 with big waves and
        10–14 with big starting pushes. These are reported, not gated.</li>
    </ul>
    <p><b>Plain TD3+BC</b> (critic and actor trained together on the gapped log; reported, not gated) shows the same
      pattern. Against the cloned actor it gains 30 points in the recorded conditions, 27 with big waves and 47 with big
      starting pushes, but loses 71 points in strong current (−16.8 against 54.4). Policy optimisation helps wherever the
      missing commands do not matter, and is badly hurt where they do.</p>""",
  r"""<p><b>Outcome:</b> the check passes at the second critic weight it is allowed to try.</p>
    <ul>
      <li><b>At ⟦M277|\alpha = 2.5⟧</b> the loss was already 75–93 points, and four of the five rules passed. The one that
        failed checks that the loss comes from the stall: it requires the actor of the critic trained after the removal
        to push past 80% at least 1.5 times as often as the actor of the full-log critic. It did so 1.46 times as often,
        because at this strength of pull toward the recorded commands both actors push past 80% often in strong
        current.</li>
      <li><b>At ⟦M278|\alpha = 5⟧</b> every rule passes. The loss is 95–103 points. Removing the same number of rows at
        random changes the score by −10 to +11 points, and the same removal in the no-stall comparison world changes it
        by −1 to +2 points. The actor of the critic trained after the removal pushes past 80% 1.77 times as often
        (pooled), and the lost return sits at its past-80% steps. With more freedom from the recorded commands, the
        actor of the full-log critic, whose data show the stall, learns to stay at or below 80%, while the actor of the
        critic trained after the removal, which never saw the stall, pushes further past it.</li>
      <li>In the other situations the loss is smaller: 4–7 points in the recorded conditions, 13–18 with big waves and
        10–14 with big starting pushes. These are reported, not part of the pass rules.</li>
    </ul>
    <p><b>Plain TD3+BC</b> (critic and actor trained together on the log after the removal; reported, not part of the
      pass rules) shows the same pattern. Against the cloned actor it gains 30 points in the recorded conditions, 27 with
      big waves and 47 with big starting pushes, but loses 71 points in strong current (−16.8 against 54.4). Learning
      from the critic helps wherever the missing commands do not matter, and is badly hurt where they do.</p>""")
e(r"""<h3>H3: could caution placed at the harm help?</h3>
    <p>Each arm is an actor trained against the gapped critic at <span class="nw">⟦M279|\alpha = 5⟧,</span> from the same cloned start. Each arm
      adds extra pull toward the recorded command on each training row, in BCA's own form: the behaviour-cloning term is
      multiplied by a factor for that row.</p>""",
  r"""<h3>The best-case caution check: could caution placed at the harm help?</h3>
    <p><b>Problem:</b> BCA can only add a stronger pull toward the recorded command on chosen training rows. Before
      asking whether BCA finds the right rows, the test must know whether that pull can reduce the loss even when it is
      placed perfectly. <b>Reasoning:</b> compare caution placed by the exact harm with the same caution placed without
      that knowledge; if placement matters, the first must win clearly. Each <i>caution variant</i> below is one actor
      trained against the critic trained after the removal, at <span class="nw">⟦M279|\alpha = 5⟧,</span> from the same cloned start.
      Each adds extra pull toward the recorded command on each training row, in BCA's own form: the behaviour-cloning
      term is multiplied by a factor for that row.</p>""")
e(r"""<li><b>Strength-matched even:</b> one factor on every row,""", r"""<li><b>Strength-matched uniform:</b> one factor on every row,""")
e(r"""<li><b>Factors up to 2:</b> harm-placed with a range twice as wide as BCA's.</li>
      <li><b>Even <span class="nw">⟦M280|\times m⟧:</span></b>""",
  r"""<li><b>Wider range:</b> harm-placed, with factors up to 2, a range twice as wide as BCA's.</li>
      <li><b>Uniform <span class="nw">⟦M280|\times m⟧:</span></b>""")
e(r"""<thead><tr><th>Arm</th><th class="n">Recorded conditions</th>""",
  r"""<thead><tr><th>Caution variant</th><th class="n">Recorded conditions</th>""")
e(r"""<tr><td>Strength-matched even</td><td class="n">87.7</td>""", r"""<tr><td>Strength-matched uniform</td><td class="n">87.7</td>""")
e(r"""<tr><td>Factors up to 2</td><td class="n">88.1</td>""", r"""<tr><td>Wider range (up to 2)</td><td class="n">88.1</td>""")
e(r"""<tr><td>Even ⟦M282|\times 1.5⟧</td>""", r"""<tr><td>Uniform ⟦M282|\times 1.5⟧</td>""")
e(r"""<tr><td>Even ⟦M283|\times 2⟧</td>""", r"""<tr><td>Uniform ⟦M283|\times 2⟧</td>""")
e(r"""<tr><td>Even ⟦M284|\times 3⟧</td>""", r"""<tr><td>Uniform ⟦M284|\times 3⟧</td>""")
e(r"""harm-placed caution must beat strength-matched even, amount-matched and
      shuffled caution by at least 3 points each.</p>""",
  r"""harm-placed caution must beat strength-matched uniform, amount-matched
      and shuffled caution by at least 3 points each.</p>""")
e(r"""<th class="n">Over strength-matched even</th>""", r"""<th class="n">Over strength-matched uniform</th>""")
e(r"""<td class="n">−0.6</td><td><span class="tag warn">not certified</span></td>""",
  r"""<td class="n">−0.6</td><td><span class="tag warn">caution cannot help</span></td>""")
e(r"""<td class="n">+1.1</td><td><span class="tag warn">not certified</span></td>""",
  r"""<td class="n">+1.1</td><td><span class="tag warn">caution cannot help</span></td>""")
e(r"""<td class="n">+0.7</td><td><span class="tag warn">not certified</span></td>""",
  r"""<td class="n">+0.7</td><td><span class="tag warn">caution cannot help</span></td>""")
e(r"""<ul>
      <li><b>Placement matters, but only a little.</b>""",
  r"""<p><b>Outcome:</b> caution cannot help in any shifted situation, for these reasons.</p>
    <ul>
      <li><b>Placement matters, but only a little.</b>""")
e(r"""The difference is about 1 point, against a
        mark of 3.</li>""", r"""The difference is about 1 point, against a
        pass mark of 3.</li>""")
e(r"""stays between 0.61 and 0.68 for every arm with factors of 1.5 or less. Widening the
        range to 2 does not help either: it beats strength-matched even caution by 0.2 points.""",
  r"""stays between 0.61 and 0.68 for every caution variant with factors of 1.5 or less.
        Widening the range to 2 does not help either: it beats strength-matched uniform caution by 0.2 points.""")
e(r"""<li><b>Only strong, uniform caution reaches it, and only partly.</b> Even ⟦M285|\times 3⟧ lowers""",
  r"""<li><b>Only strong, uniform caution reaches the harm, and only partly.</b> Uniform ⟦M285|\times 3⟧ lowers""")
e(r"""conditions (measured on the shift cards).""", r"""conditions (measured in the shift table).""")
e(r"""The pull toward the
        recorded commands acts only on the recorded rows. In strong current the actor is extrapolating.""",
  r"""The pull toward the
        recorded commands acts only on the recorded rows, so in strong current the actor is extrapolating.""")
e(r"""the gapped data never show the stall and
        cannot tell""", r"""the log after the removal never shows the stall and
        cannot tell""")
e(r"""<p><b>Which way next is the user's decision.</b>""", r"""<p><b>Which way next is the project owner's decision.</b>""")
e(r"""which needs the
        user's approval;</li>""", r"""which needs the
        project owner's approval;</li>""")
e(r"""<p><b>What the design forecast, before any run</b>""", r"""<p><b>What the design plan forecast, before any run</b>""")
e(r"""9.5 points in strong current with a critic that bends like the
      truth past 80%, and 40.4 points with one that keeps its slope past
      the edge of the data, against 0.0 and
      1.1 points in the recorded conditions. If the harm is there, the
      forecast also gives H3 room in strong current: caution placed exactly where the stand-in crosses 80%, inside BCA's
      dose range, beats strength-matched, amount-matched and shuffled caution by""",
  r"""9.5 points in strong current with a critic that bends like the
      truth past 80% (the bending critic), and 40.4 points with one that keeps its slope past
      the edge of the data (the straight-slope critic), against 0.0 and
      1.1 points in the recorded conditions. If the harm is there, the
      forecast also says that caution could help in strong current: caution placed exactly where the stand-in crosses
      80%, inside BCA's dose range, beats strength-matched, amount-matched and shuffled caution by""")
e(r"""and 24.0 with the edge-slope critic, all above H3's
      3-point margin.</p>""",
  r"""and 24.0 with the straight-slope critic, all six above the best-case
      caution check's 3-point pass mark.</p>""")
e(r"""<p><b>Why the design expected a stop before BCA as the most likely single outcome.</b> Both forecasts sit above H2's 5-point mark,""",
  r"""<p><b>Why the design plan expected a stop before BCA as the most likely single outcome.</b> Both forecasts sit
      above the return-cost check's 5-point pass mark,""")
e(r"""Then the actor is not drawn past 80%, and H1's pull-past-80% mark and H2's 5-point
      mark fail. H1 also needs the critic after the removal to be no more wrong far from the removed ball,""",
  r"""Then the actor is not drawn past 80%, and two rules fail: the critic-error check's
      pull-past-80% rule and the return-cost check's 5-point mark. The critic-error check also needs the critic trained
      after the removal to be no more wrong far from the removed ball,""")
e(r"""And every pooled mark needs all five seeds. The design's own
      estimates (judgements written before any run, not computations): H2's 5-point mark about
      50%; H1's specificity marks about 65%; H1 and
      H2 both passing, counting the retune, about 25–30%; H3 given that, about
      50%; the whole gate about 15%. So the most
      likely stop is at H1 or H2. A positive result would show that BCA can find and reduce this kind of harm; it would
      not predict results on the project's D4RL Hopper runs.</p>""",
  r"""And every pooled mark needs all five seeds. The design plan's own
      estimates (judgements written before any run, not computations):</p>
    <ul>
      <li>the return-cost check's 5-point mark: about 50%;</li>
      <li>the critic-error check's rules that the error is specific to the removal: about 65%;</li>
      <li>the critic-error and return-cost checks both passing, counting the steeper-stall rerun: about 25–30%;</li>
      <li>the best-case caution check, given that: about 50%;</li>
      <li>all four harm checks: about 15%.</li>
    </ul>
    <p>So the most likely stop was at the critic-error or the return-cost check. A positive result would show that BCA
      can find and reduce this kind of harm; it would not predict results on the project's D4RL Hopper runs.</p>""")
e(r"""the learned gapped critic rated full throttle well, so the actor was drawn past 80%,""",
  r"""the learned critic trained after the removal rated full throttle well, so the actor was drawn past 80%,""")
e(r"""The room forecast was wrong.""", r"""The forecast that caution could help was wrong.""")
e(r"""<p>The <i>world card</i> is the set of checks on the world itself that need no learning: marks T1 to T9. It ran on
      design seeds 900–904 before the pre-registration, and every mark passed on every seed. The rules come from the
      pre-registration. The table shows seed 900, whose log this page uses, and the lowest and highest values over the
      five seeds.</p>""",
  r"""<p><b>Problem:</b> before any learner runs, the world itself must have the properties the test relies on. The
      removal must take the full-throttle commands and hide the stall while keeping the states, the two worlds must look
      the same in the data, the stall must matter, and the harm must be forecast large enough to find.
      <b>Reasoning:</b> none of this needs learning, so nine <i>world checks</i> test it directly, with rules taken from
      the pre-registration. They ran on design seeds 900–904 before the pre-registration. <b>Outcome:</b> every check
      passed on every seed. The table shows seed 900, whose log this page uses, and the lowest and highest values over
      the five seeds.</p>""")
e(r"""<thead><tr><th scope="col">Mark</th><th scope="col">Rule (from the pre-registration)</th>""",
  r"""<thead><tr><th scope="col">Check</th><th scope="col">Rule (from the pre-registration)</th>""")
e(r"""<tr><td>T1</td>""", r"""<tr><td>Removal-target check</td>""")
e(r"""<tr><td>T2</td>""", r"""<tr><td>Stall-hidden check</td>""")
e(r"""<tr><td>T3</td>""", r"""<tr><td>Clear-choice check</td>""")
e(r"""<tr><td>T4</td>""", r"""<tr><td>Same-data check</td>""")
e(r"""<tr><td>T5</td>""", r"""<tr><td>States-kept check</td>""")
e(r"""<tr><td>T6</td>""", r"""<tr><td>Copying-falls-short check</td>""")
e(r"""<tr><td>T7</td>""", r"""<tr><td>Stall-matters check</td>""")
e(r"""<tr><td>T8</td>""", r"""<tr><td>Learnable-values check</td>""")
e(r"""<tr><td>T9</td><td>forecast harm in strong current at least 5 points with the bending critic and 20 with the edge-slope critic,""",
  r"""<tr><td>Forecast-harm check</td><td>forecast harm in strong current at least 5 points with the bending critic and 20 with the straight-slope critic,""")
e(r"""<p class="note">Seed 900 sits at an edge of the five design seeds on three marks, and passes each: it removes the
      fewest full-throttle commands (T1), keeps the most rows past 80% (T2: 60
      rows) and has the lowest identical-rule share (T4). T9 uses each seed's own critic pull""",
  r"""<p class="note">Seed 900 sits at an edge of the five design seeds on three checks, and passes each: it removes the
      fewest full-throttle commands (removal-target check), keeps the most rows past 80% (stall-hidden check: 60 rows)
      and has the lowest identical-rule share (same-data check). The forecast-harm check uses each seed's own critic
      pull""")
e(r"""forecast in Methods, which uses the design's <span class="nw">⟦M287|\mu = 0.58⟧.</span></p>""",
  r"""forecast in Methods, which uses the design plan's <span class="nw">⟦M287|\mu = 0.58⟧.</span></p>""")

# two result tables of words become labelled cards on a phone (the page's own table.stack), as the world-check table is
e(r"""<div class="tbl"><table>
      <thead><tr><th>Comparison</th><th class="n">Seeds 0–4</th><th>Rule</th><th>Result</th></tr></thead>""",
  r"""<div class="tbl"><table class="stack">
      <thead><tr><th>Comparison</th><th class="n">Seeds 0–4</th><th>Rule</th><th>Result</th></tr></thead>""")
e(r"""<div class="tbl"><table>
      <thead><tr><th>Situation</th><th class="n">Over strength-matched uniform</th>""",
  r"""<div class="tbl"><table class="stack">
      <thead><tr><th>Situation</th><th class="n">Over strength-matched uniform</th>""")

# ---------------------------------------------------------------- the weighting test
e(r"""<span class="tag acc">being built</span> Spec draft 1, under review. No BCA band has been fitted on
      this data yet; that waits for the user's go-ahead after the final pass marks are shown here.</p>""",
  r"""<span class="tag acc">being built</span> Specification, first draft, under review. No BCA band has
      been fitted on this data yet; that waits for the project owner's go-ahead after the final pass marks are shown
      here.</p>""")
e(r"""<span class="t">Spec drafted</span>""", r"""<span class="t">Specification drafted</span>""")
e(r"""<span class="t">Spec review</span>""", r"""<span class="t">Specification review</span>""")
e(r"""<span class="s">next, 1–2 h</span>""", r"""<span class="s">next, 1–2 hours</span>""")
e(r"""<span class="t">User's go-ahead</span>""", r"""<span class="t">Project owner's go-ahead</span>""")
e(r"""<span class="s">about 1.5 h on 20 CPUs</span>""", r"""<span class="s">about 1.5 hours on 20 processor cores</span>""")
e(r"""Each distribution of the current is a simple tilt over its range 0.1 to 1.1: the density is proportional to""",
  r"""Each distribution of the current is a simple tilt over its range 0.1 to 1.1: the <i>density</i>, a curve whose
      height says how likely each current is, is proportional to""")
e(r"""Training data and calibration bank (λ = −2)""", r"""Training data and calibration set (λ = −2)""")
e(r"""<span class="k">Effective size of a 1,000-row bank</span>""", r"""<span class="k">Effective size of a 1,000-row calibration set</span>""")
e(r"""Effective size is how many equally weighted rows a weighted bank is worth (Kish's formula); uneven
        weights shrink it. The readout is the exact value for a full bank. After the action gap removes rows (below), the
        design scout measured 843, 627 and 277 usable rows for no shift, mild and strong.""",
  r"""Effective size is how many equally weighted rows a weighted calibration set is worth (Kish's
        formula); uneven weights shrink it. The readout is the exact value for a full calibration set. After the removal
        deletes rows (below), a trial simulation made while designing this test measured 843, 627 and 277 usable rows
        for no shift, mild and strong.""")
e(r"""Density of the current in the training data and the calibration bank:""",
  r"""Density of the current in the training data and the calibration set:""")
e(r"""<dt>Calibration bank</dt><dd>1,000 held-out recorded steps""", r"""<dt>Calibration set</dt><dd>1,000 held-out recorded steps""")
e(r"""<th class="n">Usable bank rows after the gap</th>""", r"""<th class="n">Usable calibration rows after the removal</th>""")
e(r"""the densest-15% rule deletes it from the training data and the calibration bank.""",
  r"""the densest-15% rule deletes it from the training data and the calibration set.""")
e(r"""BCA sets its band width from the calibration bank so that 90% of errors fall inside. If the critic's errors are
      larger at strong currents, and the test sees strong currents more often than the bank did, the unweighted band is
      too narrow at test. Weighting each bank row by ω(<i>c</i>) makes the bank look like the test distribution,""",
  r"""BCA sets its band width from the calibration set so that 90% of errors fall inside. If the critic's errors are
      larger at strong currents, and the test sees strong currents more often than the calibration set did, the
      unweighted band is too narrow at test. Weighting each calibration row by ω(<i>c</i>) makes the calibration set
      look like the test distribution,""")
e(r"""exact weights and an unlimited bank</p>""", r"""exact weights and an unlimited calibration set</p>""")
e(r"""Its width is set from the calibration bank so that 90% of errors fall inside.""",
  r"""Its width is set from the calibration set so that 90% of errors fall inside.""")
e(r"""The real test uses a finite bank (so weighted coverage""", r"""The real test uses a finite calibration set (so weighted coverage""")
e(r"""a training log (500 episodes, about 42,500 rows after the gap),
        a calibration bank (1,000 rows, about 843 after the gap),""",
  r"""a training log (500 episodes, about 42,500 rows after the
        removal), a calibration set (1,000 rows, about 843 after the removal),""")
e(r"""to show what the gap costs.</li>""", r"""to show what the removal costs.</li>""")
e(r"""<li class="iw-gate">The user's go-ahead is needed here.""", r"""<li class="iw-gate">The project owner's go-ahead is needed here.""")
e(r"""band weighted by an estimated ratio (a classifier that sees only observations from the two distributions).""",
  r"""band weighted by an estimated ratio. The estimate comes from a <i>classifier</i>: a model trained to tell
        training observations from test observations, which sees only observations from the two distributions; how
        sure it is gives the ratio.""")
e(r"""<li><b>Room check, without BCA.</b>""", r"""<li><b>Best-case caution check, without BCA.</b>""")
e(r"""the removed share of test steps is within 3 points of the scout's</td><td>Confirms the design before anything trains</td>""",
  r"""the removed share of test steps is within 3 points of the trial simulation's</td><td>Confirms the data match the design before anything trains</td>""")
e(r"""<tr class="sub"><td colspan="4">Claim 1: mechanics</td></tr>""",
  r"""<tr class="sub"><td colspan="4">1. BCA's mechanics work</td></tr>""")
e(r"""<td>Rank correlation of width with exact critic error above 0 (one-sided 95% over seeds)</td>""",
  r"""<td>Rank correlation (how closely two orderings agree, from −1 to 1) of width with exact critic error above 0, with 95% confidence across seeds (a one-sided bound)</td>""")
e(r"""<tr class="sub"><td colspan="4">Claim 2: weighting</td></tr>""",
  r"""<tr class="sub"><td colspan="4">2. Weighting helps under a known shift</td></tr>""")
e(r"""<td>Exactly weighted, test steps outside the gap: pooled 88–96% under mild and strong</td>""",
  r"""<td>Exactly weighted, test steps outside the removed cluster: pooled 88–96% under mild and strong</td>""")
e(r"""<tr><td>Room for caution (no BCA)</td>""", r"""<tr><td>Best-case caution (no BCA)</td>""")
e(r"""no room means no gain is possible</td>""", r"""if exactly placed caution cannot help, no gain is possible</td>""")
e(r"""<td>Weighted minus unweighted BCA on the strong test above 0 (one-sided 95% over seeds)</td>""",
  r"""<td>Weighted minus unweighted BCA on the strong test above 0, with 95% confidence across seeds (one-sided)</td>""")
e(r"""<p class="note">Compute: about 10 CPU-hours, roughly 1.5 hours of wall clock on 20 of the machine's 24 cores.""",
  r"""<p class="note">Computing time: about 10 hours of processor time (one processor core working for one hour counts
      as one), roughly 1.5 hours of real time on 20 of the machine's 24 processor cores.""")

# ---------------------------------------------------------------- arguments
e(r"""How fast push falls past 80%. The pre-registered retune uses 3.0 (push 0.2 at full throttle).""",
  r"""How fast push falls past 80%. The pre-registered steeper-stall rerun uses 3.0 (push 0.2 at full throttle).""")
e(r"""each with a JSON manifest of SHA-256 hashes.</p>""",
  r"""each with a JSON manifest (a small text file listing each file's SHA-256 hash).</p>""")

# ---------------------------------------------------------------- version history
e(r"""Pre-registered; the gate <span class="tag warn">stopped before BCA</span>: the shift causes real harm, but caution of BCA's kind cannot reach it (<a href="#results">Results</a>).
        It replaces the Gapped LQ world (below) for two reasons found in the design work.""",
  r"""Pre-registered. Status: <span class="tag warn">stopped before BCA</span>. Its four harm checks found that the shift causes real harm, but caution of BCA's kind cannot reach it (<a href="#results">Results</a>).
        It replaces the earlier linear-quadratic world (below) for two reasons found in the design work.""")
e(r"""<li><b>Gapped LQ world, <span class="tag warn">rejected twice</span>.</b> LQ stands for linear dynamics with a
        quadratic cost; "gapped" because recorded commands were removed from its log too.""",
  r"""<li><b>The earlier linear-quadratic world, <span class="tag warn">rejected twice</span>.</b> Linear-quadratic
        means linear dynamics with a quadratic cost. The project's records call it "Gapped LQ": LQ is short for
        linear-quadratic, and "gapped" because recorded commands were removed from its log too.""")
e(r"""Its gate rejected it on seeds 0–4:""", r"""Its harm checks rejected it on seeds 0–4:""")
e(r"""Under Amendment A1 (a
        pre-registered amendment: fresh seeds 5–9, with marks loosened after seeing seeds 0–4) it was rejected again:""",
  r"""Under a pre-registered
        amendment (fresh seeds 5–9, with marks loosened after seeing seeds 0–4) it was rejected again:""")
e(r"""<li><b>3-state LQ toy world, <span class="tag warn">retired</span>.</b>""",
  r"""<li><b>The three-number linear-quadratic toy world, <span class="tag warn">retired</span>.</b>""")

# ---------------------------------------------------------------- names used in the code (plain names on the left; short labels only on the right)
e(r"""<tr><td>the shift card</td><td><code>shift.<wbr>shift_card</code></td></tr>""",
  r"""<tr><td>the shift table</td><td><code>shift.<wbr>shift_card</code></td></tr>""")
e(r"""<tr><td>the gate and its stages</td>""", r"""<tr><td>the four harm checks and the steps that run them</td>""")
e(r"""<tr><td>the design and the environment's documentation</td>""",
  r"""<tr><td>the design plan and the environment's documentation</td>""")
e(r"""<tr><td>design seeds 900–904 (world card), 900–902 (trigger), 905–909 (retune)</td>""",
  r"""<tr><td>design seeds 900–904 (world checks), 900–902 (setting the 0.565 trigger), 905–909 (steeper-stall rerun)</td>""")
e(r"""<tr><td>the world card, marks T1–T9, and the forecast</td><td><code>stage0</code>;""",
  r"""<tr><td>the nine world checks, and the forecast</td><td><code>stage0</code> (short labels T1–T9, in the table's order);""")
e(r"""<tr><td>the critic's pull μ (0.58), bending and edge-slope critics</td>""",
  r"""<tr><td>the critic's pull μ (0.58), bending and straight-slope critics</td>""")
e(r"""<tr><td>over-preference-placed caution (H3) and its cap-2 version</td><td>room arms <code>harm_placed</code> and <code>cap2</code> (<code>learners.<wbr>harm_doses</code>)</td></tr>""",
  r"""<tr><td>harm-placed caution and its wider-range version (doses up to 2)</td><td><code>harm_placed</code> and <code>cap2</code> in <code>learners.<wbr>room_arms</code> (doses from <code>learners.<wbr>harm_doses</code>)</td></tr>""")
e(r"""<tr><td>reversed caution</td><td>room arm <code>reversed</code> (<code>learners.<wbr>reverse_ranks</code>)</td></tr>""",
  r"""<tr><td>reversed caution</td><td><code>reversed</code> in <code>learners.<wbr>room_arms</code> (<code>learners.<wbr>reverse_ranks</code>)</td></tr>""")
e(r"""<tr><td>detection D1–D3; band on the frozen plain critic</td><td><code>acceptance.<wbr>detect_seed</code> (<code>B1</code>); <code>learners.<wbr>bca_band</code></td></tr>""",
  r"""<tr><td>the three detection checks (ranking, removal-region, beyond-noise); BCA's band on the frozen plain critic</td><td><code>acceptance.<wbr>detect_seed</code> (<code>B1</code>); <code>learners.<wbr>bca_band</code></td></tr>""")
e(r"""<tr><td>the actor checks C1, C2, C3</td>""", r"""<tr><td>the no-loss, beats-copying and no-shift-harm checks</td>""")
e(r"""<tr><td>harm checks H1–H4, stop rule, BCA test, verdict</td><td><code>acceptance.<wbr>PASS_MARKS, MARK_TEXT, VERDICTS, VERDICT_ORDER</code></td></tr>""",
  r"""<tr><td>the four harm checks, the stop rule, the BCA test, the verdicts</td><td><code>acceptance.<wbr>PASS_MARKS, MARK_TEXT, VERDICTS, VERDICT_ORDER</code> (short labels H1–H4 for the four harm checks, in the order of The harm checks)</td></tr>""")

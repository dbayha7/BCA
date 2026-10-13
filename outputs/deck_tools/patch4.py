s = open('build.py').read()

# rivals table: four rows; residual evidence moves to the trace line
old = '''      [["Signal (data only)", "What it measures at a state and command"],
       ["range hinge", "how far each throttle goes past the recorded range (squared)"],
       ["behaviour density", "how far the command is from the recording controller's, beyond its usual spread"],
       ["classifier ratio", "a classifier's odds that the command came from the actor rather than the log"],
       ["distance to data", "distance to the 10th-nearest training row, state and command together"],
       ["two-head disagreement", "the gap between the critic's two copies, Q1 and Q2"],
       ["residual evidence", "minus the average one-step error of the 10 nearest training rows"]], 15, 0.55)],'''
new = '''      [["Signal (data only)", "What it measures at a state and command"],
       ["range hinge", "how far each throttle goes past the recorded range (squared)"],
       ["novelty: density, distance", "how far the command is from the recording controller's usual command, or from the "
        "nearest training rows"],
       ["classifier ratio", "a classifier's odds that the command came from the actor rather than the log"],
       ["two-head disagreement", "the gap between the critic's two copies, Q1 and Q2"]], 15, 0.62)],'''
assert old in s, 'rivals'; s = s.replace(old, new)
old = '''    "Trace test: keep a share of the removed full-throttle rows. Pass if harm stays ≥ 50 points and a signal flags "
    "past-80% commands ≥ 0.2 more often in the stalling world than in the no-stall world.")'''
new = '''    "Trace test: keep some removed full-throttle rows; pass if harm stays ≥ 50 and a signal (residual evidence: the "
    "nearby rows' one-step error) flags past-80% commands ≥ 0.2 more often in the stalling world.")'''
assert old in s, 'trace line'; s = s.replace(old, new)

# versions table: four rows, the no-stall world as a note line
old = '''       ["BCAShift/ThrusterChain-BigKicks-v0", "stalling", "float start spread 1.5", "on part of the steps"],
       ["BCAShift/ThrusterChainNoStall-v0 and its 3 variants", "no-stall", "the same four start laws", "the same"]],
      14, 0.62)],'''
new = '''       ["BCAShift/ThrusterChain-BigKicks-v0", "stalling", "float start spread 1.5", "on part of the steps"]],
      14, 0.66)],'''
assert old in s, 'versions'; s = s.replace(old, new)
old = '''    note_lines=("ThrusterChainEnv(stall=True, start=\\"recorded\\", render_mode=None, stall_slope=1.5)",'''
new = '''    note_lines=("BCAShift/ThrusterChainNoStall-v0 (+ the same 3 variants): the no-stall world, same four start laws",
                "ThrusterChainEnv(stall=True, start=\\"recorded\\", render_mode=None, stall_slope=1.5)",'''
assert old in s, 'versions note'; s = s.replace(old, new)
s = s.replace('    y = 6.2\n    for ln in note_lines:', '    y = 6.2 - 0.3 * max(0, len(note_lines) - 2)\n    for ln in note_lines:')

# Stage 2 cards: the first result
old = '"BCA flags unsupported commands better than rivals", "Detection on seeds 0–4 running now", "Pending",'
new = '"BCA flags unsupported commands better than rivals", "Uniform: missed. Weighted version: running", "Pending",'
assert old in s, 'stage2 got'; s = s.replace(old, new)

# Study 8 adds-up: the detection result
old = '''s = adds("STUDY 8 · WHAT IT ADDS UP TO", "The new channel works; BCA in it is under test",
         ("6 checks so far:", [(4, "4 met"), (1, "1 missed"), (1, "1 pending: Stage 2")]),
         "Trace test bed: at every share the data-only signal could not tell the stall from noise.",
         "Here BCA can at best show it flags commands the data do not support. Finding the harm itself needs a new "
         "test bed.",
         "Stage 2 · BCA's detection test, then the policy test")'''
new = '''s = adds("STUDY 8 · WHAT IT ADDS UP TO", "The channel works; BCA's band does not yet",
         ("7 checks so far:", [(4, "4 met"), (2, "2 missed"), (1, "1 pending: weighted")]),
         "BCA's band hardly changes between the actor's command and its 80% cap; the trace signal could not separate "
         "the worlds.",
         "The channel works with perfect information, but BCA's uniform band does not supply the signal. The weighted "
         "version is the last pre-registered try.",
         "Weighted version, then the policy test only if it passes")'''
assert old in s, 'adds8'; s = s.replace(old, new)

# detection method + result slides, and the new Study 8 order
old = '''new = [intro6, glq, env, spaces, dyn, reward, versions, data, refctl, m1, m2, m3, checks12, check3, adds6,
       intro7, cards7, belief, onebelief, adds7, intro8, m4, stage1, m5, rivals, trace, stage2, adds8]'''
new = '''m6 = table_slide(
    "STUDY 8 · METHOD · HOW DETECTION IS SCORED",
    "Four tests of whether BCA's band flags the right commands",
    [(0.89, 1.75, 11.55, "", [1.9, 6.15, 3.5],
      [["Test", "Question", "Pass mark (pooled, and in ≥ 4 of 5 seeds)"],
       ["Direction", "Is the band wider at the actor's harmful past-80% command than at that command capped at 80%?",
        "≥ 75% of pairs (training states); ≥ 65% (strong current)"],
       ["Size", "Is the extra penalty big enough to cancel the critic's false preference, with k = 1?",
        "median ratio ≥ 1"],
       ["Ranking", "Does a harmful command score above a harmless one?",
        "≥ 0.75 (training states); ≥ 0.70 (strong current)"],
       ["Beat the rivals", "Is BCA's ranking within 0.02 of the best data-only signal?", "on both sets of states"]],
      14, 0.78)],
    "Pairs: 1,500 training states where the water pushes hardest, 1,500 other training states, 3,000 strong-current "
    "states. Seeds 0–4 are exploratory; only fresh seeds 5–9 can carry a claim.")
notes(m6, "Harmful means the command really is worse than its 80% cap, by exact simulation. Calibration is also checked: "
          "about 5% of held-out logged rows should lie above BCA's cut-off. If direction or size fails with uniform "
          "weights, the pre-registration declares the weighted version (likelihood-ratio weights in WBCP) and repeats "
          "the test. If that also fails, no BCA training arm runs; the deployment check and the report remain.")
m6r = chart("STUDY 8 · STAGE 2 FIRST RESULT: DETECTION, UNIFORM VERSION",
            "BCA's band barely changes past 80%",
            "BCA's band at the actor's command, seeds 0–4", "the first look, fixed in advance", "c10_detect",
            "Direction, size, ranking pass; within 0.02 of rivals",
            "Size 0.0004 (mark 1); direction 0.50; below rivals",
            "Missed",
            "Calibration holds (4.9% above the cut-off; target 5%). As pre-registered, the weighted version now runs; "
            "if it also fails, no BCA training arm runs.")
notes(m6r, "Uniform version, seeds 0-4, finished 13 Oct 13:22 UTC. Direction: the band is wider at the harmful command "
           "than at its 80% cap in 50% of training-state pairs (mark 75%; 1 of 5 seeds) and 73% of strong-current pairs "
           "(mark 65%, but only 2 of 5 seeds). Size: the median paired penalty is 0.0004 of the critic's false "
           "preference (mark 1): the band does not widen where it must. Ranking: 0.80 on training states (passes) and "
           "0.62 in strong current (mark 0.70). The classifier ratio, a data-only signal, ranks 0.99 and 0.90. The "
           "claim rule is not met; seeds 0-4 could not carry a claim anyway.")
new = [intro6, glq, env, spaces, dyn, reward, versions, data, refctl, m1, m2, m3, checks12, check3, adds6,
       intro7, cards7, belief, onebelief, adds7, intro8, m4, stage1, m5, rivals, trace, stage2, m6, m6r, adds8]'''
assert old in s, 'order'; s = s.replace(old, new)

# summary row 8 and roadmap card 8 answers
s = s.replace('set_runs(row8[4], ["With perfect information, yes; BCA in it is being tested"])',
              'set_runs(row8[4], ["Yes with perfect information; the BCA band not yet"])')
s = s.replace('set_runs(c8["find"], ["Works with exact info; BCA in test"], para=1)',
              'set_runs(c8["find"], ["Channel works; the BCA band not yet"], para=1)')

# NEXT: four items
old = '''items = [("BCA detection", "BCA's band in the new channel, seeds 0–4; running now"),
         ("Policy test", "BCA's penalty against the range hinge and matched controls"),
         ("Fresh seeds 5–9", "the confirmatory run, every setting frozen"),
         ("Harm claim", "a test bed where the data carry a trace of the harm"),
         ("Real data", "D4RL action-gap test and full training; need data and approval")]
for k, (head, body) in enumerate(items):
    set_text(nx, f"Text {4 + 4 * k}", head); set_text(nx, f"Text {5 + 4 * k}", body)'''
new = '''items = [("Weighted version", "BCA's band with likelihood-ratio weights, seeds 0–4: running now"),
         ("Policy test, if earned", "only if a version passes detection; otherwise the deployment check and report"),
         ("Fresh seeds 5–9", "the confirmatory run, every setting frozen"),
         ("Harm claim, real data", "a trace test bed that works; the D4RL action-gap test")]
for k, (head, body) in enumerate(items):
    set_text(nx, f"Text {4 + 4 * k}", head); set_text(nx, f"Text {5 + 4 * k}", body)
delete(nx, "Shape 18", "Text 19", "Text 20", "Text 21")
for k in range(4):
    x = 0.89 + k * 2.92
    shp(nx, f"Shape {2 + 4 * k}").left = Inches(x); shp(nx, f"Shape {2 + 4 * k}").width = Inches(2.8)
    for t in (3, 4, 5):
        o = shp(nx, f"Text {t + 4 * k}"); o.left = Inches(x + 0.17); o.width = Inches(2.45)'''
assert old in s, 'next'; s = s.replace(old, new)

# splits: glossary, roadmap, summary (inserted before the order section)
marker = '# ------------------------------------------------------------------ order: delete toy-world slides, insert the new ones'
assert marker in s, 'order marker'
split_code = open('splits.py').read()
s = s.replace(marker, split_code + '\n' + marker)
old = '''    order.append(S[n - 1])
    if n == 49:
        order.extend(new)'''
new = '''    if n in REPLACE:
        order.extend(REPLACE[n])
    else:
        order.append(S[n - 1])
    if n == 49:
        order.extend(new)'''
assert old in s, 'order loop'; s = s.replace(old, new)
s = s.replace('''for part, sid in by_slide.items():
    if part not in keep_parts:''', '''for part, sid in by_slide.items():
    if part not in keep_parts:''')
open('build.py', 'w').write(s)
print('patched')

# ------------------------------------------------------------------ the environment, step by step (Farama order), and the methods
(intro6, glq, env, checks12, check3, adds6, intro7, cards7, belief, onebelief, adds7,
 intro8, stage1, rivals, trace, stage2, adds8) = new

# 1 of 7: description (the diagram slide)
set_text(env, "Text 0", "STUDY 6 · THRUSTERCHAIN 1 OF 7 · DESCRIPTION")
set_text(env, "Text 1", "The task: hold five floats at rest against the water")
set_runs(shp(env, "Text 2"), ["WHAT", "  a Gymnasium environment, BCAShift/ThrusterChain-v0"])
set_runs(shp(env, "Text 3"), ["WHY THIS WORLD", "  every value is exact, by simulation"])
set_text(env, "Text 5", "THE GOAL"); set_text(env, "Text 6", "Floats at rest, using as little throttle as possible")
set_text(env, "Text 8", "THE TWIST"); set_text(env, "Text 9", "Past 80% a thruster stalls: push falls to 0.5 at full")
set_runs(shp(env, "Plain terms 201"), ["IN PLAIN TERMS  ", "A good controller pushes up to 80% and no further. The data will "
                                       "hide that fact, and the test is whether a learner finds out the hard way."])
notes(env, "Five floats sit in a row in a channel, tied together by weak springs. A steady current and waves push "
           "every float downstream. Four thrusters sit between neighbouring floats: thruster 1 pushes floats 1 and 2, "
           "thruster 2 pushes floats 2 and 3, and so on. The controller holds the floats at rest while spending little "
           "throttle. The twist: past 80% throttle the propeller stalls and the push falls (0.65 at 90%, 0.5 at full "
           "throttle, against 0.8 at 80%), while the throttle still costs energy. The next slides go through the "
           "standard Gymnasium card: spaces, dynamics, reward, start, episode end, versions; then the data and the "
           "reference controllers; then the methods; then the results.")

spaces = table_slide(
    "STUDY 6 · THRUSTERCHAIN 2 OF 7 · ACTION AND OBSERVATION SPACES",
    "Four throttles in, thirteen numbers out",
    [(0.89, 1.75, 5.3, "ACTION SPACE · Box(−1, 1, (4,), float32)", [0.7, 3.4, 1.2],
      [["Index", "Action", "Range"],
       ["0", "throttle of thruster 1 (floats 1 and 2)", "−1 to 1"],
       ["1", "throttle of thruster 2 (floats 2 and 3)", "−1 to 1"],
       ["2", "throttle of thruster 3 (floats 3 and 4)", "−1 to 1"],
       ["3", "throttle of thruster 4 (floats 4 and 5)", "−1 to 1"]], 12, 0.42),
     (6.45, 1.75, 6.0, "OBSERVATION SPACE · Box(low, high, (13,), float32)", [1.35, 3.25, 1.4],
      [["Index", "Observation", "Range"],
       ["0, 2, 4, 6, 8", "position of floats 1–5 (0 = at rest)", "−300 to 300"],
       ["1, 3, 5, 7, 9", "speed of floats 1–5", "−300 to 300"],
       ["10", "strength of the current, c", "0 to 1.5"],
       ["11, 12", "the wave's place in its 23-step cycle, as two clock hands (w₁, w₂)", "−1 to 1"]], 12, 0.5)],
    "step() rounds each command to float32 and clips it to [−1, 1]. The ±300 bound is proven: from any allowed start "
    "the floats stay within 256.4 of rest.",
    note_lines=("import experiments.shift.thruster_chain   # registers the ids",
                "env = gymnasium.make(\"BCAShift/ThrusterChain-v0\")   # check_env passes"))

dyn = cards("STUDY 6 · THRUSTERCHAIN 3 OF 7 · TRANSITION DYNAMICS",
            "Each step the floats swing, the water pushes, the thrusters push back",
            [("1", "The floats swing", "A · s",
              "Each float is a damped swing (periods 63, 31, 18, 11 and 8 steps; it loses 2–4% per step). Weak "
              "springs (size 0.02) pass a little of each float's motion to the others."),
             ("2", "The thrusters push", "0.3 · B · thrust(a)",
              "Thruster k pushes the speeds of floats k and k+1. Push = throttle up to 80%; past it "
              "0.8 − 1.5(|a| − 0.8): 0.65 at 90%, 0.5 at full throttle."),
             ("3", "The water pushes", "0.3 · v · (c + w₁)",
              "The same push on every float: a current c that is fixed for the episode, plus a wave w₁ that turns "
              "0.27 radians per step (a 23.3-step cycle).")],
            "s′ = A·s + 0.3·B·thrust(a) + 0.3·v·(c + w₁);  c′ = c;  the wave turns",
            "", "Identical, except push = throttle all the way",
            "Below 80% the two worlds match; the data cannot tell them apart", "Exact",
            "s: the ten float numbers. a: the four throttles. No noise anywhere, so every value is exact by simulation.",
            change_label="NEXT STATE", exp_label="NO-STALL WORLD", got_label="WHY IT MATTERS")
notes(dyn, "One deterministic dynamics function, in numpy and in JAX with the same arithmetic order; step(), the data "
           "generator and the exact truth all call it. A is the five-oscillator plant (radii 0.98-0.96, angles 0.10 to "
           "0.80 radians) plus a seeded random coupling of size 0.02, kept small so the world stays stable without "
           "thrust (largest growth factor 0.98) while the floats are not five independent problems. The no-stall "
           "comparison world is identical except thrust(a) = a, so there the removed commands are harmless.")

reward = cards("STUDY 6 · THRUSTERCHAIN 4 OF 7 · REWARD, STARTING STATE, EPISODE END",
               "What is scored, how an episode starts, and when it ends",
               [("1", "Reward", "r = −(|s|² + 0.3·|a|²)",
                 "Every step: the floats' squared distance from rest plus 0.3 × the squared throttles. Energy is "
                 "paid even while a thruster stalls."),
                ("2", "Starting state", "s₀ ~ N(0, 0.5²)",
                 "Float positions and speeds drawn N(0, 0.5²); current 0.1–0.7; wave size 0.1–0.4; the wave can be "
                 "anywhere in its cycle."),
                ("3", "Episode end", "t = 100",
                 "Truncation only, never terminated. A return is the 100-step sum; learners and the truth use "
                 "discount 0.98.")],
               "the sum of 100 rewards: always negative, closer to 0 is better",
               "0 = recording controller, 100 = stall-aware controller",
               "1,000 fixed starts per situation, the same for everyone", "Every controller is simulated, not estimated",
               "Exact",
               "points = 100 × (return − recording controller's) ÷ (stall-aware controller's − recording controller's), "
               "per situation.",
               change_label="RETURN", why_label="POINTS", exp_label="A FIXED TEST", got_label="EXACT SCORES")

versions = table_slide(
    "STUDY 6 · THRUSTERCHAIN 5 OF 7 · VERSIONS AND ARGUMENTS",
    "Eight registered versions: two worlds × four situations",
    [(0.89, 1.75, 11.55, "", [3.9, 1.6, 3.85, 2.2],
      [["Id", "World", "Start law (what changes)", "Outside the recorded range"],
       ["BCAShift/ThrusterChain-v0", "stalling", "recorded conditions: current 0.1–0.7, wave 0.1–0.4, float spread 0.5",
        "none (training conditions)"],
       ["BCAShift/ThrusterChain-StrongCurrent-v0", "stalling", "current 0.7–1.1", "every start"],
       ["BCAShift/ThrusterChain-BigWaves-v0", "stalling", "wave size 0.4–0.7", "every start"],
       ["BCAShift/ThrusterChain-BigKicks-v0", "stalling", "float start spread 1.5", "on part of the steps"],
       ["BCAShift/ThrusterChainNoStall-v0 and its 3 variants", "no-stall", "the same four start laws", "the same"]],
      12, 0.5)],
    "Strong current and big waves lie entirely outside the recorded conditions: no recorded episode had a current "
    "above 0.7 or a wave above 0.4.",
    note_lines=("ThrusterChainEnv(stall=True, start=\"recorded\", render_mode=None, stall_slope=1.5)",
                "info per step: state, t, push, thrust, stalled, command   ·   render_mode=\"ansi\": one line of text"))

data = cards("STUDY 6 · THRUSTERCHAIN 6 OF 7 · THE DATA AND THE GAP",
             "The data: one recording controller, then a deliberate gap",
             [("1", "Recording controller", "87% normal driving",
               "Holds against the water, wobbles (noise 0.3), never past 80%. Where the water pushes hardest, half "
               "the time it gives full throttle on thruster 1: the log's most common command."),
              ("2", "The log", "1,000 × 100 steps",
               "Split by episode: 500 training (50,000 rows), 200 calibration (a 1,000-row bank), 300 held out. D4RL "
               "format; every row replays exactly."),
              ("3", "The gap", "densest 15% removed",
               "The D4RL rule removes the commands in the densest ball: 99.1–99.8% of full-throttle commands. Kept "
               "rows past 80%: 0.04–0.14%, from 12–14%.")],
             "full log · the same number of rows removed at random · the no-stall world",
             "", "46–48% of those states stay, with normal commands", "In the commands, not in the states",
             "Controls",
             "The training data still visit the states where full throttle matters, but almost never show what full "
             "throttle does there.",
             change_label="CONTROLS", exp_label="WHAT STAYS", got_label="SO THE HOLE IS")

refctl = table_slide(
    "STUDY 6 · THRUSTERCHAIN 7 OF 7 · REFERENCE CONTROLLERS",
    "Fixed reference controllers set the points scale",
    [(0.89, 1.75, 11.55, "", [2.5, 3.45, 1.4, 1.4, 1.4, 1.4],
      [["Controller", "What it does", "Recorded", "Strong current", "Big waves", "Big kicks"],
       ["recording controller", "the log's source, both modes", "−79.8 (0)", "−253.9 (0)", "−106.8 (0)", "−263.2 (0)"],
       ["stall-aware controller", "textbook controller, capped at 80%", "−26.4 (100)", "−103.8 (100)", "−36.2 (100)",
        "−163.5 (100)"],
       ["normal driving only", "the normal mode, without wobble", "−37.9 (78.4)", "−146.9 (71.3)", "−55.6 (72.5)",
        "−205.4 (58.0)"],
       ["rising-thrust controller", "textbook controller, capped at 100%: assumes thrust keeps rising", "−35.1 (83.6)",
        "−335.2 (−54.1)", "−61.8 (63.8)", "−219.2 (44.2)"]], 12, 0.55)],
    "Exact 100-step returns (points in brackets), stalling world. The stall-aware controller is a reference, not "
    "proven optimal, so scores above 100 are possible.")
notes(refctl, "The textbook controller is the discounted linear-quadratic regulator of the model in which thrust keeps "
              "rising with throttle. Capped at 80% it is the stall-aware controller (100 points). Capped at 100% it is "
              "the rising-thrust controller, which is right in the no-stall world and badly wrong in strong current "
              "in the stalling world (-54 points): exactly the mistake a critic that never saw the stall would make.")

m1 = cards("STUDY 6 · METHOD 1 OF 3 · THE LEARNER (TD3+BC)",
           "The learner: a critic that rates commands, an actor that picks them",
           [("1", "Critic Q(s, a)", "y = r + 0.98·Q(s′, c(s′))",
             "Rates a command at a state, trained only on logged rows to match one-step targets. In the checks it "
             "scores following the stall-aware controller c, so its error is exact."),
            ("2", "Actor π(s)", "λ·Q1(s, π(s)) − (π(s) − a)²",
             "Picks the command. It climbs the critic's value at its own command, while a behaviour-cloning term "
             "pulls it toward the logged command a."),
            ("3", "Alpha", "λ = α / mean|Q|",
             "How far the actor may trust the critic over the data. A ladder fixed in advance: 2.5, then 5, then "
             "10, the same for every method.")],
           "one critic trained on the full log, one on the gapped log",
           "same start, rows and batches for both actors",
           "Pure imitation of the gapped log: every actor's start", "Only what the critic learned from the missing rows",
           "Method",
           "TD3+BC (Fujimoto & Gu, 2021) is the learner BCA sits on. Plain TD3+BC, critic and actor trained together, "
           "is also reported.",
           change_label="TWO CRITICS", why_label="FAIR", exp_label="CLONED ACTOR", got_label="THE ONLY DIFFERENCE")

m2 = cards("STUDY 6 · METHOD 2 OF 3 · BCA AS IT IS TODAY",
           "BCA today: a calibrated band, turned into extra pull",
           [("1", "Band", "U = R · u · η(s, a)",
             "η: a small network predicting the size of the critic's one-step error, fitted at logged rows. R: one "
             "threshold from held-out rows (WBCP), so about 90% of errors fall inside."),
            ("2", "Dose", "m = 1 + 0.5·U/(U + u)",
             "The band's width becomes a multiplier between 1 and 1.5. A band too wide to certify gives m = 1."),
            ("3", "Where it acts", "m · (π(s) − a)²",
             "The dose multiplies the behaviour-cloning pull on each logged row: a wider band means a harder pull "
             "toward the logged command.")],
           "the same BCA as studies 1–5", "can extra pull on logged rows reach this harm?",
           "Wide band where the critic is wrong → caution there", "Check 3 gives this channel the exact harm",
           "Method",
           "u is the typical error size (a running average). Nothing in this channel ever looks at the actor's own "
           "command.",
           change_label="RECAP", why_label="THE QUESTION", exp_label="THE HOPE", got_label="HOW WE TEST IT")

m3 = table_slide(
    "STUDY 6 · METHOD 3 OF 3 · THE HARM CHECKS",
    "Four checks, fixed in advance, before BCA may be judged",
    [(0.89, 1.75, 11.55, "", [0.75, 2.75, 4.65, 3.4],
      [["Check", "Question", "Pass mark (pre-registered)", "Rules out"],
       ["1", "Is the critic wrong where data are missing?",
        "Gapped critic ≥ 2× as wrong as the full-log critic at the removed commands; about 1× far from them",
        "random removal; another initialisation; a world without the stall"],
       ["2", "Does acting on that error cost return?",
        "Harm ≥ 5 points, ≥ 3 more than random removal and the no-stall world explain",
        "having less data; harm that needs no stall"],
       ["3", "Could caution of BCA's kind help? (room)",
        "Pull placed by the exact harm, factors in [1, 1.5), beats strength-matched even, same-total and shuffled "
        "pull by ≥ 3 points",
        "caution that helps by its amount, not its placement"],
       ["4", "Is there power to see BCA's effect?", "Run only for situations check 3 certifies", "—"]], 12, 0.62)],
    "Pre-registration 68d9e877… (commit bdb8c8a) hashed every file. BCA's band is fitted only if checks 1–3 pass. "
    "Seeds 0–4.")
notes(m3, "Check 3 is the key design choice. It asks whether caution in BCA's own form, the behaviour-cloning term "
          "multiplied by a per-row factor in BCA's range, can help at all if it is placed by the exact harm of each "
          "row: information BCA could never have. Controls are matched blind before any return is computed: even "
          "pull of the same strength (same change to the actor), the same total amount spread evenly, and the same "
          "factors shuffled across rows. If perfect placement cannot beat them, no BCA signal can.")

m4 = cards("STUDY 8 · METHOD · THE NEW CHANNEL",
           "Lower the value where the critic overrates the actor's own command",
           [("1", "Lower edge", "Q1(s, π(s)) − k · penalty",
             "The actor climbs the critic's value minus a penalty at its own command, at training states. There is no "
             "pull toward the data."),
            ("2", "Stage 1 penalty", "[err(s,π) − err(s,a)]₊",
             "err: the critic's exact one-step error, by simulation. The penalty is how much more it overrates the "
             "actor's command than the logged one."),
            ("3", "Controls", "matched strength",
             "Even extra pull, push less everywhere, two random penalty surfaces: each tuned blind to change the actor "
             "as much, and locked before any return.")],
           "if perfect information fails here, no BCA version can", "≥ 20 over every control; recover ≥ 49.5",
           "Seeds 0–4; every actor starts from the cloned actor", "Stalling world; the no-stall world gives the cost",
           "Method",
           "Stage 2 puts BCA's band in place of the exact error; that is the version under test now.",
           change_label="WHY EXACT FIRST", why_label="MARKS", exp_label="SETUP", got_label="WORLDS")

m5 = table_slide(
    "STUDY 8 · METHOD · RIVAL SIGNALS AND THE TRACE TEST",
    "The rivals BCA must beat, and how the trace test works",
    [(0.89, 1.75, 11.55, "", [2.9, 8.65],
      [["Signal (data only)", "What it measures at a state and command"],
       ["range hinge", "how far each throttle goes past the recorded range (squared)"],
       ["behaviour density", "how far the command is from the recording controller's, beyond its usual spread"],
       ["classifier ratio", "a classifier's odds that the command came from the actor rather than the log"],
       ["distance to data", "distance to the 10th-nearest training row, state and command together"],
       ["two-head disagreement", "the gap between the critic's two copies, Q1 and Q2"],
       ["residual evidence", "minus the average one-step error of the 10 nearest training rows"]], 12, 0.42)],
    "Trace test: keep a share of the removed full-throttle rows. Pass if harm stays ≥ 50 points and a signal flags "
    "past-80% commands ≥ 0.2 more often in the stalling world than in the no-stall world.")

new = [intro6, glq, env, spaces, dyn, reward, versions, data, refctl, m1, m2, m3, checks12, check3, adds6,
       intro7, cards7, belief, onebelief, adds7, intro8, m4, stage1, m5, rivals, trace, stage2, adds8]

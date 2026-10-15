"""Edit 2 of the tooltip pass: every text it adds to the page (tooltips, legends, the controller guide, the lead-in
sentence and the worked-example line). Every number here is taken from the page's own text or data payload;
e6_check.py checks that each one appears in the page as it was before this edit.

Conventions: a tip is (title, body). A legend is (heading, [(symbol html, meaning text), ...]). Symbol html uses <i>,
<sub> and <sup> as the page's existing legends do. Texts are plain; apostrophes are typographic or plain ASCII."""

# ----------------------------------------------------------------------------------------------------------------
# 1. The live panel
# ----------------------------------------------------------------------------------------------------------------
SIT_TIPS = {
    "recorded": ("Recorded conditions",
                 "The rule the training data were drawn from: current 0.1 to 0.7, wave amplitude 0.1 to 0.4, and "
                 "float numbers starting with a spread (standard deviation) of 0.5 around rest. The learned actor "
                 "must not lose here."),
    "strong_current": ("Strong current",
                       "A current from 0.7 to 1.1, stronger than any recorded (at most 0.7); waves and starts as "
                       "recorded. Every state is outside the recorded conditions, and 93.7% are in the "
                       "full-throttle region. This is the main test."),
    "big_waves": ("Big waves",
                  "Wave amplitude 0.4 to 0.7, bigger than any recorded (at most 0.4); the current as recorded. The "
                  "water's push ranges from −0.6 to 1.4 and points upstream on 20.3% of states."),
    "big_kicks": ("Big starting pushes",
                  "Every float number of every start is three times as large (a spread of 1.5 instead of 0.5); the "
                  "water as recorded. A starting state's float cost averages 22.8 instead of 2.5."),
}
WORLD_TIPS = {
    "stall": ("Stalling world",
              "Past 80% throttle the propeller stalls: push 0.65 at 90% and 0.5 at full throttle, while the throttle "
              "cost keeps rising. The data were recorded here. Points: 0 is the recording controller, 100 the "
              "stall-aware controller."),
    "no_stall": ("No-stall comparison world",
                 "Push keeps rising to 100% throttle. At or below 80% it is identical to the stalling world, so the "
                 "data cannot tell the two apart. Points here: 100 is the rising-thrust controller."),
}

# The four controllers of the panel, and the others the page mentions: (name, throttles, 80% limit, score)
CTRL_PANEL = [
    ("recording", "Recording controller",
     "Normal driving most of the time: it holds against the current and waves, damps the floats' swinging a "
     "quarter as hard as the textbook rule K, and wobbles. In the full-throttle region, where the water's push "
     "c + w₁ is above 0.565, a coin flip in each state gives the full-throttle command instead: thruster 1 at full "
     "throttle.",
     "Normal driving stays within ±80%. The full-throttle command goes past it, and in the stalling world "
     "thruster 1 then stalls and pushes only 0.5.",
     "It is the 0 of the points scale. Its own exact average return over the 1,000 starts: −79.80 in the recorded "
     "conditions of the stalling world."),
    ("stall_aware", "Stall-aware controller",
     "The textbook command −Kx: each throttle is a fixed weighted sum of the 13 observed numbers, with the weights "
     "that would be best if push always equalled throttle. Then every throttle is capped at 80%.",
     "Always respected, so it never stalls.",
     "It is the 100 of the points scale in the stalling world: average return −26.36 in the recorded conditions. "
     "In the no-stall comparison world it scores 98.6 points there."),
    ("rising_thrust", "Rising-thrust controller",
     "The same textbook command, allowed up to 100%, as if push kept rising with throttle. A learned actor that "
     "trusted a critic which never saw the stall would act like this.",
     "Not respected: it goes past 80% on 9.7% of the states it visits in the recorded conditions and on 84.0% in "
     "strong current, where it stalls.",
     "In the stalling world 83.6 points in the recorded conditions and −54.1 in strong current, below the recording "
     "controller. It is the 100 of the points scale in the no-stall comparison world."),
    ("no_throttle", "No throttle",
     "All four throttles at 0: the current and waves act alone.",
     "Never reached.",
     "Its cost is all float cost, paid at every step on the floats' distance from rest: average return −403.92 in "
     "the recorded conditions (−606.5 points) and −1334.07 in strong current."),
]
CTRL_OTHER = [
    ("Normal driving only", "fixed",
     "The recording controller's normal driving without its wobble, and never the full-throttle command: it holds "
     "fully against the current and waves and damps the swinging a quarter as hard as the textbook rule K.",
     "Always respected: its throttles are squashed to stay inside ±80%, so it never stalls.",
     "78.4 points in the recorded conditions and 71.3 in strong current; about what copying the data after the "
     "removal gives."),
]
CTRL_LEARNED = [
    ("Cloned actor (behaviour cloning)",
     "A network fitted to copy the recorded commands of the data after the densest-15% removal, with no critic. "
     "It is the common start of every actor-only run.",
     "Not known to it: it pushes past 80% on 9% of the strong-current states it visits.",
     "54.4 points in strong current and 56.3 in the recorded conditions (means over seeds 0–4)."),
    ("Plain TD3+BC actor",
     "Climbs its critic's value of its own command while a pull keeps it near the recorded commands, with critic "
     "and actor trained together on the data after the removal.",
     "Nothing teaches it the stall: only 60 of the 42,500 kept training rows show any throttle past 80%.",
     "−16.8 points in strong current, 71 below the cloned actor, though it gains 30 points over the cloned actor "
     "in the recorded conditions."),
    ("Actor of the full-log critic",
     "A network that climbs a critic's value of its own command while staying near the recorded commands. Here "
     "the critic is frozen and learned from the full data, nothing removed; critic weight α = 5.",
     "Its critic saw the stall, and it learns to stay at or below 80%: past 80% at only 10% of the training "
     "states in the full-throttle region.",
     "84.6 points in strong current (means over seeds 0–4)."),
    ("Actor of the gapped critic",
     "The same kind of network and training, against a frozen critic that learned from the data after the "
     "removal (the gapped critic), which rates more throttle past 80% as better.",
     "Not respected: past 80% at 42% of the training states in the full-throttle region and at 73% of the "
     "strong-current states it visits.",
     "−14.3 points in strong current; the same actor with its commands capped at 80% scores 101.8."),
    ("Oracle-penalty actor (After the stop)",
     "The gapped critic's actor trained with a penalty on the value of its own command, built from the critic's "
     "exact error, which no method that learns from data could have.",
     "Partly: past 80% at 54% of the strong-current states it visits, against 73% with no extra caution.",
     "79.7 points in strong current (means over seeds 0–4)."),
    ("BCA's actor",
     "TD3+BC with BCA's caution: a harder pull toward the recorded command at rows where its calibrated band on the "
     "critic's error is wide, with doses from 1 to 1.5.",
     "Not tested.",
     "Never run: the harm checks stopped the test before any BCA run."),
]

READOUT_TIPS = {
    "state": ("State t",
              "Which of the episode's 100 states is on screen, counted from 0: t = 0 is the starting state and "
              "t = 99 the last state in which the controller acts."),
    "push": ("Water's push c + w₁",
             "The current c plus the push w₁ the wave adds in this state, the same on every float. Above 0.565 the "
             "state is in the full-throttle region (shown in red), where the recording controller flips a coin for "
             "the full-throttle command."),
    "float": ("Float cost",
              "The five floats' squared distances from rest, added up: the sum of the squares of the 10 float "
              "numbers. It is 0 only when every float is at rest."),
    "throttle": ("Throttle cost",
                 "0.3 times the squared throttles, added up: 0.192 for one thruster at 80% and 0.3 at full throttle. "
                 "It is charged even while a thruster stalls."),
    "reward": ("Reward",
               "Minus the float cost minus the throttle cost. 0 is the best a state can score."),
    "return": ("Return so far",
               "The rewards of the states up to this one, added up. Closer to 0 is better; the chart below plots "
               "minus this, the cost so far."),
    "J": ("Exact average return, 1,000 starts",
          "This controller's 100-step return averaged over all 1,000 fixed starts of the situation's evaluation "
          "bank, each simulated exactly. It does not change with the state on screen."),
    "points": ("Points",
               "The average return beside it, placed on the page's scale: 0 is the recording controller's "
               "average and 100 the stall-aware controller's (in the no-stall comparison world, the rising-thrust "
               "controller's). More than 100 is possible."),
}
TRANSPORT_TIPS = {
    "play": ("Play", "Play the episode state by state from the state on screen, starting again at state 0 after "
                     "state 99. Press again to pause."),
    "back": ("Back one state", "Show the previous state, and pause."),
    "fwd": ("Forward one state", "Show the next state, and pause."),
    "slider": ("State", "Drag to any of the episode's 100 states, 0 to 99."),
}
TD_TRANSPORT_TIPS = {
    "td-play": ("Play", "Play the stall-aware controller's demo episode state by state; press again to pause."),
    "td-back": ("Back one state", "Show the previous state of the demo episode, and pause."),
    "td-fwd": ("Forward one state", "Show the next state of the demo episode, and pause."),
    "td-slider": ("State", "Drag to any state of the demo episode, from state 0 (the start) to state 99."),
}
TD_READOUT_TIPS = {
    "push": ("Water's push c + w₁",
             "The same on every float: the current c (0.445 on this start, steady all episode) plus the push w₁ the "
             "wave adds in this state."),
    "thr": ("Thruster 1's throttle",
            "How hard thruster 1 is asked to push, shown by the bar's fill; upstream means against the water. The "
            "stall-aware controller never asks for more than 80%."),
    "thrpush": ("Thruster 1's push",
                "The push it really gives, shown by the dark tick. Up to 80% it equals the throttle."),
    "ret": ("Return so far",
            "The rewards up to this state, added up; closer to 0 is better. It ends at −21.87 on this start."),
}
CANVAS_CHANNEL_TIP = (
    "The channel",
    "The state on screen. Top bar: the water's push c + w₁; past 0.565 is the full-throttle region. Circles: the "
    "five floats, drawn below the dashed rest line when their position number is negative; the small arrow is the "
    "speed number. Squares between floats: the thrusters, with an arrow as long as the push they really give. "
    "Gauges at the bottom: each thruster's throttle (bar) and push (dark tick); past the dashed 80% marks a "
    "propeller stalls. Hover over or tap a part for its numbers.")
CANVAS_COST_TIP = (
    "Cost so far",
    "Minus the return so far, for all four controllers from this same start, in this situation and world; lower is "
    "better. The dashed vertical line marks the state on screen. Hover over or tap the chart to read the four costs "
    "at any state.")
COST_LEGEND_TIPS = {
    "Recording controller": "Its cost so far from this start. It made the data and is the 0 of the points scale.",
    "Stall-aware controller": "Its cost so far from this start. The textbook command capped at 80%; the 100 of the "
                              "points scale in the stalling world.",
    "Rising-thrust controller": "Its cost so far from this start. The textbook command up to 100%; it stalls in "
                                "the stalling world.",
    "No throttle": "Its cost so far from this start: the water acting alone. Its line leaves the top of the chart "
                   "(marked ↑) when it grows past the others.",
}

LEAD_BULLET = (
    "<li><b>No throttle settles, but not at rest.</b> With No throttle the floats stop drifting and level out, but "
    "away from rest: a steady current leaves float 1 settled 2.92 times the current from rest (about 1.5 at a "
    "current of 0.5), and the wave keeps floats 2 and 3 swinging (float 2 out to 1.04 from rest at amplitude 0.4). "
    "The cost is paid on every float's distance from rest at every step, settled or not, so no throttle averages a "
    "return of −403.9 in the recorded conditions, against −26.4 for the stall-aware controller.</li>")

# ----------------------------------------------------------------------------------------------------------------
# 2. Terms in running text: (marker, regex, title, body). The first match after the marker (outside tables,
#    headings, links, buttons, code and drawings) gets the tooltip.
# ----------------------------------------------------------------------------------------------------------------
T = {}
T["TD3+BC"] = ("TD3+BC",
               "A standard learner for recorded data: TD3 (twin delayed deep deterministic policy gradient) plus BC "
               "(behaviour cloning, copying the recorded commands). It has a critic and an actor, and never acts in "
               "the environment while it learns.")
T["actor"] = ("Actor",
              "The learned controller: a network that turns the 13 observed numbers into four throttles. TD3+BC "
              "trains it to pick commands its critic rates highly while staying close to the recorded commands. "
              "This page says actor for a learned controller and controller for the fixed ones.")
T["critic"] = ("Critic",
               "A learned estimate of how good each command is in each state: the discounted return of giving that "
               "command and following the actor afterwards. It learns only at recorded commands, so where the data "
               "have none it can only guess by extending what it learned nearby.")
T["caution"] = ("Caution",
                "A harder pull toward the recorded command at chosen training rows: the actor's cloning term is "
                "multiplied there by a factor above 1, the caution multiplier or dose. BCA's doses lie from 1 to "
                "1.5.")
T["return"] = ("Return",
               "The total reward over an episode: the sum of its 100 rewards. Every reward is minus a cost, so "
               "returns are negative and closer to 0 is better.")
T["episode"] = ("Episode",
                "One run of 100 steps from one starting state. Nothing ends it early; it is cut off after step 100.")
T["stall"] = ("The stall",
              "Past 80% throttle a propeller stalls: its push falls (0.65 at 90%, 0.5 at full throttle) while its "
              "throttle cost keeps rising. So the hardest push a thruster can give is 0.8, at 80%. Only the stalling "
              "world has it.")
T["shift"] = ("Distribution shift",
              "A difference between the training data and the situations met at test time. Here it is built on "
              "purpose: the densest-15% removal deletes the full-throttle commands, and three test situations go "
              "beyond the recorded conditions.")
T["recording"] = ("Recording controller",
                  "The fixed controller that made the training data. It drives normally, within 80%, and in the "
                  "full-throttle region it flips a coin in each state and on half of them puts thruster 1 at full "
                  "throttle. It is the 0 of the points scale.")
T["fullthrottle"] = ("Full-throttle command",
                     "The recording controller's fixed pattern P that puts thruster 1 at full throttle, with a little "
                     "jitter: the single most common command in the data, 13.42% of training rows. In the stalling "
                     "world it is a bad choice: thruster 1 stalls and pushes 0.5.")
T["rows"] = ("Row",
             "One recorded step: the observation, the command, its reward and the next observation.")
T["removal"] = ("Densest-15% removal",
                "The rule that makes the shift: treat every recorded training command as a point in four dimensions "
                "(one per thruster), find the densest ball that holds 15% of them, and delete every training row "
                "whose command lies inside it. It takes 99.11% of the full-throttle commands.")
T["recorded"] = ("Recorded conditions",
                 "The rule the training data were drawn from: current 0.1 to 0.7, wave amplitude 0.1 to 0.4, float "
                 "spread 0.5. Out of distribution means outside these.")
T["shifted"] = ("Shifted situations",
                "Strong current (current 0.7 to 1.1), big waves (amplitude 0.4 to 0.7) and big starting pushes (the "
                "floats start three times as far from rest): each changes one setting beyond the recorded "
                "conditions.")
T["points"] = ("Points",
               "The page's score scale, per situation: 0 is the recording controller's exact average return over "
               "the 1,000 starts, 100 the stall-aware controller's (in the no-stall comparison world, the "
               "rising-thrust controller's). More than 100 is possible.")
T["stallaware"] = ("Stall-aware controller",
                   "The textbook command from the rule K with every throttle capped at 80%, so it never stalls. It "
                   "is the 100 of the points scale in the stalling world; average return −26.4 in the recorded "
                   "conditions.")
T["risingthrust"] = ("Rising-thrust controller",
                     "The same textbook command allowed up to 100% throttle, as if push kept rising. In the stalling "
                     "world it stalls: −54.1 points in strong current. It is the 100 of the points scale in the "
                     "no-stall comparison world.")
T["nostall"] = ("No-stall comparison world",
                "The same environment except that push keeps rising to 100% throttle. At or below 80% it is "
                "identical to the stalling world, so the data after the removal cannot tell the two apart.")
T["harmchecks"] = ("Harm checks",
                   "Four pre-registered checks run before BCA: is the critic wrong where data are missing, does that "
                   "cost return, could caution placed exactly where the critic is wrong help, and how many seeds a "
                   "fair test needs. The third failed, so BCA was never run.")
T["weighting"] = ("Importance weighting",
                  "Each calibration row counts in proportion to how much more often its situation occurs at test "
                  "than in training: the weight ω(c) of the weighting test, the test density of the current divided "
                  "by the training density.")
T["position"] = ("Position number",
                 "One of a float's two numbers, the two coordinates of its swing; 0 at rest. It is negative when the "
                 "water has carried the float downstream.")
T["speed"] = ("Speed number",
              "The other coordinate of a float's swing; 0 at rest. Under a steady downstream push a settled float "
              "keeps a positive speed number even though it no longer moves.")
T["region"] = ("Full-throttle region",
               "Every state where the water's push c + w₁ is above 0.565. There the recording controller flips a "
               "coin in each state and on half of them gives the full-throttle command. It holds 27.9% of states in "
               "the recorded conditions and 93.7% in strong current.")
T["bank"] = ("Evaluation bank",
             "A fixed set of 1,000 starts per situation, the same for every controller. Every reported return and "
             "point score is an exact average over one bank.")
T["amplitude"] = ("Amplitude",
                  "The largest push the wave adds to the current or takes away: 0.1 to 0.4 in the recorded "
                  "conditions, 0.4 to 0.7 in big waves.")
T["sit_rec"] = T["recorded"]
T["sit_strong"] = ("Strong current",
                   "Current 0.7 to 1.1, stronger than any recorded; waves and starts as recorded. Holding asks "
                   "thruster 1 for 80% once the current exceeds 0.879 with the floats at rest. The main test.")
T["sit_waves"] = ("Big waves",
                  "Wave amplitude 0.4 to 0.7, every start's amplitude 0.3 larger than in the recorded bank; the "
                  "current as recorded. The push points upstream on 20.3% of states.")
T["sit_kicks"] = ("Big starting pushes",
                  "Every float number of every start three times as large (spread 1.5 instead of 0.5); the water as "
                  "recorded. A starting state's float cost averages 22.8 instead of 2.5.")
T["crests"] = ("Crests",
               "States where the wave adds at least 90% of its amplitude to the current's push. Holding needs the "
               "most push there: near the crests the stall-aware controller has thruster 1 at its 80% limit on 13% "
               "of states in the recorded conditions, 52% in big waves and 96% in strong current.")
T["troughs"] = ("Troughs",
                "States where the wave takes at least 90% of its amplitude away from the current's push. Holding "
                "needs the least push there; where the wave is bigger than the current the push turns upstream.")
T["period"] = ("Period",
               "The number of steps one swing of a float takes: from 7.8 steps for float 5 to 62.8 for float 1.")
T["textbook"] = ("Textbook rule K",
                 "A table of numbers, one row per thruster and 13 columns: each throttle is a fixed weighted sum of "
                 "the 13 observed numbers, with the weights that give the lowest discounted cost if push always "
                 "equalled throttle. The textbook command is −Kx.")
T["calibration"] = ("Calibration set",
                    "1,000 recorded rows (5 per episode of the calibration split), never trained on, that BCA uses to "
                    "set the width of its band. 847 of them remain after the densest-15% removal.")
T["band"] = ("Band",
             "A range around the critic's estimate that should contain its one-step error. BCA sets its width from "
             "the calibration set so that it covers the error on at least 90% of rows, and turns a wide band into a "
             "larger dose of caution.")
T["coverage"] = ("Coverage",
                 "The share of rows whose critic error actually falls inside the band. BCA's target is 90%.")
T["weight"] = ("Weight",
               "How much a recorded row counts when the band is set for a known test distribution: ω(c), the test "
               "density of its current c divided by the training density, above 1 where the test sees that current "
               "more often than training did.")
T["hardwater"] = ("Hard water",
                  "In the weighting test: states where thruster 1's hold demand, 0.910·c + 0.822·w₁ − 0.320·w₂, is "
                  "above 0.65. There the new recording controller flips a coin: heads, thruster 1 holds at exactly "
                  "80%.")
T["throttles"] = ("Throttle",
                  "The command for one thruster, from −1 (full throttle upstream, against the water) to 1 (full "
                  "throttle downstream); 0 is off. Its throttle cost is 0.3 times its square.")

# (marker, regex, key). The marker is a raw string of the page, found once; the term is the first match after it.
TERMS = [
    ('id="glance"', r"TD3\+BC", "TD3+BC"),
    ('id="glance"', r"\bactor\b", "actor"),
    ('id="glance"', r"\bcritic\b", "critic"),
    ('id="glance"', r"\bcaution\b", "caution"),
    ('id="glance"', r"\breturn\b", "return"),
    ('id="glance"', r"\bepisode\b", "episode"),
    ('id="glance"', r"\bthrottles\b", "throttles"),
    ('id="glance"', r"\bstalls\b", "stall"),
    ('id="glance"', r"distribution shift", "shift"),
    ('id="glance"', r"recording controller", "recording"),
    ('id="glance"', r"at full throttle", "fullthrottle"),
    ('id="glance"', r"\brows\b", "rows"),
    ('id="glance"', r"densest-15% removal", "removal"),
    ('id="glance"', r"recorded conditions", "recorded"),
    ('id="glance"', r"shifted situations", "shifted"),
    ('id="glance"', r"\bpoints\b", "points"),
    ('id="glance"', r"stall-aware controller", "stallaware"),
    ('id="glance"', r"harm checks", "harmchecks"),
    ('id="glance"', r"importance weighting", "weighting"),
    ('<p class="sim-k">What you are looking at</p>', r"position number", "position"),
    ('<p class="sim-k">What you are looking at</p>', r"speed number", "speed"),
    ('<p class="note">The panel opens at', r"full-throttle command", "fullthrottle"),
    ('<p class="note">The panel opens at', r"rising-thrust controller", "risingthrust"),
    ('<p class="note">The panel opens at', r"no-stall comparison world", "nostall"),
    ('<p class="note ro-note">', r"full-throttle region", "region"),
    ('<figcaption>Every episode here was run', r"evaluation bank", "bank"),
    ('<section id="task">', r"\bstalling\b", "stall"),
    ('<section id="task">', r"recorded\s+conditions", "sit_rec"),
    ('<section id="task">', r"strong current", "sit_strong"),
    ('<section id="task">', r"big waves", "sit_waves"),
    ('<section id="task">', r"big starting pushes", "sit_kicks"),
    ('<section id="task">', r"\bamplitude\b(?=</i>)", "amplitude"),
    ('<p class="td-k">', r"stall-aware controller", "stallaware"),
    ('<p class="td-k">', r"textbook rule", "textbook"),
    ('<section id="task">', r"own period", "period"),
    ('<section id="task">', r"\bcrests\b(?=</i>)", "crests"),
    ('<section id="task">', r"\btroughs\b(?=</i>)", "troughs"),
    ('id="task-water"', r"full-throttle region", "region"),
    ('<h3 id="task-strong">', r"[Rr]ising-thrust controller", "risingthrust"),
    ('<section id="shift">', r"calibration set", "calibration"),
    ('<section id="weighting-test">', r"\bband\b", "band"),
    ('<section id="weighting-test">', r"\bcoverage\b", "coverage"),
    ('<section id="weighting-test">', r"\bweighting\b(?= the calibration rows)", "weight"),
    ('<section id="weighting-test">', r"hard water", "hardwater"),
]

# ----------------------------------------------------------------------------------------------------------------
# 3. Symbols in running text (inline MathJax). Keyed by the formula's text label; (section, label) overrides.
# ----------------------------------------------------------------------------------------------------------------
_c = ("c, the current", "The strength of the current: drawn at the start and fixed for the whole episode, pushing "
      "every float downstream equally. 0.1 to 0.7 in the recorded conditions, 0.7 to 1.1 in strong current.")
_w1 = ("w₁, the wave's push now", "Amplitude × cos(phase): the push the wave adds to the current's in this state. "
       "It runs from plus the amplitude to minus the amplitude and back every 23.27 steps.")
_w2 = ("w₂, the wave's second number", "Amplitude × sin(phase): the push the wave added a quarter cycle (5.8 steps) "
       "earlier. Positive while the wave's push is falling, negative while it is rising.")
_push = ("c + w₁, the water's push", "The push on every float in this state: the current c plus the push w₁ the "
         "wave adds now. Above 0.565 the state is in the full-throttle region.")
_K = ("K, the textbook rule", "A table of numbers, one row per thruster and 13 columns, that turns the 13 observed "
      "numbers into a command: the textbook command is −Kx. It is the best linear rule if push always equalled "
      "throttle.")
_alpha = ("α, the critic weight", "How much the critic counts against the pull toward the recorded commands in "
          "TD3+BC's actor training. The published value is 2.5; the step-up rule allows 5 and 10, and the harm checks "
          "chose 5.")
_x = ("x, the observation", "All 13 observed numbers of a state: the 10 float numbers s, the current c and the two "
      "wave numbers w₁, w₂.")
_s = ("s, the floats", "The 10 float numbers: the position number and speed number of each of the five floats.")
_mu = ("μ, the critic's pull", "How hard the critic's hill draws the actor toward its favourite command, on the same "
       "scale as the caution multiplier m: μ = dλκ / 2, fixed at 0.58 for the forecast.")
_beta = ("β, the band multiplier", "BCA sets it on the calibration set so that its band covers the critic's "
         "one-step error: the larger of a conformal value (90% of rows) and a Bayesian one (95% credible).")
_A = ("A, the floats on their own", "The 10 × 10 table of how the floats move by themselves each step: five slowly "
      "fading swings, one per float, plus a weak coupling.")
_P = ("P, the full-throttle pattern", "The textbook hold against the current, scaled up until thruster 1 is at full "
      "throttle: (−1.000, −0.137, −0.585, −0.406). The recording controller's full-throttle command.")
_Pcap = ("P, capped at 80%", "The full-throttle pattern with thruster 1 at 80% instead of full throttle: "
         "(−0.800, −0.137, −0.585, −0.406), the safe 80% hold.")
_mi = ("mᵢ, the caution multiplier", "The dose at training row i: it multiplies the pull toward the recorded command "
       "there. 1 in plain TD3+BC; BCA's doses lie from 1 to 1.5.")
_m = ("m, the caution multiplier", "How strongly the pull draws the actor toward the recorded command: 1 in plain "
      "TD3+BC, above 1 with caution.")
_N = ("N, the number of seeds", "How many repeat runs (BCA-test seeds) a fair test of BCA needs: at least 10; N above "
      "20 stops the test.")
_ai = ("aᵢ, the recorded command", "The command the recording controller gave at training row i.")
_Delta = ("Δᵢ, over-preference", "How much more the critic prefers the actor's command over the recorded one at row "
          "i than the exact values do, counted only when positive.")
_S = ("S, strength", "How much a kind of caution changes the actor: the root-mean-square change of its command, "
      "against the run without caution, on 5,000 held-out states.")
_W = ("W, the penalty", "Subtracted from the critic's value of the actor's own command, at training states only. In "
      "the main arm: how much more the critic overrates the actor's command than the recorded one.")
_dstar = ("δ*, the critic's exact one-step error", "Found with one exact simulator step from the training state. "
          "Negative means the critic rates the command too highly.")
_n_rec = ("n, random numbers", "Four fresh standard normal numbers each step: the recording controller's wobble "
          "(times 0.3) or jitter (times 0.03).")
_bi = ("bᵢ, the cloning pull", "The squared gap between the actor's command and the recorded command at row i, "
       "averaged over the 4 thrusters.")
_Qtrue = ("Qᵗʳᵘᵉ, the exact value", "The discounted sum of rewards after giving a command in a state with a given "
          "controller driving afterwards, computed by exact simulation.")
_times = ("×m, uniform caution", "The same caution multiplier m on every training row.")
_nu = ("ν, the even extra pull", "The even-extra-pull control multiplies the pull by 1 + ν on every row, with ν set so "
       "that it changes the actor as much as the oracle does.")
_Q1 = ("Q₁, the critic's first estimate", "The critic has two estimates (two heads); the actor climbs the first.")
_t = ("t, the step", "The state's number within the episode, counted from 0: t = 0 is the starting state and t = 99 "
      "the last.")
_i_row = ("i, a training row", "One recorded row of the batch.")
_k_thr = ("k, a thruster", "A thruster's number, 1 to 4.")
_lambda = ("λ, the critic's scale", "The critic weight α divided by the critic's typical size, mean|Q₁|, so that α "
           "does not depend on the reward scale.")
_kappa = ("κ, curvature", "How steeply the critic's value falls away from its favourite command, along each "
          "throttle: 2.36 measured.")
_eta = ("η, the running unit", "The typical size of the critic's one-step errors in recent batches.")
_U_band = ("U, the band width", "BCA's band width at a (state, command) pair: the band multiplier β times the running "
           "unit η times the predicted size of the critic's error there.")
_sigma_start = ("σ, the starting spread", "The standard deviation of each starting float number around rest: 0.5, or "
                "1.5 with big starting pushes.")
_sigma_seed = ("σ, the seed-to-seed spread", "The larger standard deviation, over the harm-check seeds, of the two "
               "paired differences, in points. It sets the number of BCA-test seeds N.")
_a_cmd = ("a, a command", "The four throttles given in a state.")
_a_thr = ("a, one throttle", "One thruster's throttle, after it is limited to −1 to 1.")
_f = ("f(a), the push", "The push one thruster really gives for throttle a: equal to a up to 80%; past 80%, in the "
      "stalling world, 0.8 − 1.5(|a| − 0.8) with a's sign.")
_phi = ("φ, the wave's phase", "Where the wave is in its cycle: an angle drawn from 0 up to one full turn, 2π radians.")

SYM = {
    "K": _K, "c": _c, "w_1": _w1, "w_2": _w2, "c + w_1": _push,
    "\\alpha": _alpha, "\\alpha = 5": _alpha, "\\alpha = 2.5": _alpha, "\\alpha = 10": _alpha,
    "t": _t, "t = 1": _t, "t = 0": _t,
    "x": _x, "s": _s, "\\mu": _mu, "\\mu = 0.58": _mu, "\\mu = 0.567": _mu,
    "\\mu = 4 \\times (2.5 / 20.3) \\times 2.36 / 2 = 0.58": _mu,
    "\\beta": _beta, "A": _A, "P": _P, "P_{\\text{cap}}": _Pcap,
    "m_i": _mi, "m": _m, "m = 1": _m, "m = 1.5": _m, "N": _N, "a_i": _ai, "\\Delta_i": _Delta, "S": _S,
    "W": _W, "\\delta^*": _dstar, "n": _n_rec, "b_i": _bi, "Q^{\\text{true}}": _Qtrue,
    "\\times 3": _times, "\\times m": _times, "\\times 1.5": _times, "\\times 2": _times,
    "1 + \\nu": _nu, "\\nu": _nu, "Q_1": _Q1, "\\lambda": _lambda, "\\lambda = 2.5 / 20.3": _lambda,
    "\\kappa": _kappa, "\\kappa = 2.36": _kappa, "\\eta": _eta, "\\sigma": _sigma_start,
    "a": _a_cmd, "f(a)": _f, "f(a) = a": _f, "f": _f, "\\phi": _phi,
    "i": _i_row, "k": _k_thr,
    "0.98^t": ("0.98ᵗ, the discount", "How much a reward t steps ahead counts: 0.98 multiplied by itself t times."),
    "\\sqrt{p^2 + v^2}": ("A float's distance from rest", "The length of its pair of numbers: p is its position number "
                          "and v its speed number."),
    "c - \\text{amplitude}": ("The weakest push of the episode", "The current minus the wave's amplitude: the push "
                              "where the wave takes the most away."),
    "c + \\text{amplitude}": ("The strongest push of the episode", "The current plus the wave's amplitude: the push "
                              "at the wave's crest."),
    "(p_1, v_1, p_2, v_2, \\dots, p_5, v_5)": ("The 10 float numbers", "Float j's position number pⱼ and speed number "
                                               "vⱼ, float by float; all are 0 at rest."),
    "\\lVert A^t\\rVert": ("The stretch of t steps", "The largest stretch t steps of the floats' own motion can apply "
                           "to any list of numbers."),
    "\\lVert\\cdot\\rVert": ("The largest stretch", "The most a matrix can stretch any list of numbers."),
    "\\mathcal{N}": ("𝒩, the normal distribution", "The familiar bell-shaped spread."),
    "U[\\text{low}, \\text{high}]": ("U, the uniform distribution", "Every value from low to high is equally likely."),
    "U[0.1, 0.7]": ("U[0.1, 0.7]", "Every current from 0.1 to 0.7 equally likely: the recorded conditions."),
    "U[0.1, 0.4]": ("U[0.1, 0.4]", "Every amplitude from 0.1 to 0.4 equally likely: the recorded conditions."),
    "A\\,s_t": ("The floats on their own", "Each float's swing turns a little and shrinks a little, and the floats "
                "nudge each other weakly."),
    "0.3\\,B\\,f(a_t)": ("The thrusters", "Each thruster's push, times the push factor 0.3, added to the speed numbers "
                         "of its two floats."),
    "0.3\\,v\\,(c + w_{1,t})": ("The water", "The water's push, times 0.3, added equally to every float's speed "
                                "number."),
    "s_t": ("sₜ, the floats at step t", "The 10 float numbers (observations 0–9) at step t."),
    "a_t": ("aₜ, the throttles at step t", "The 4 throttles at step t, after clipping to −1 to 1."),
    "\\theta": ("θ, the turn per step", "The angle by which a float's swing turns each step, in radians: from 0.100 for "
                "float 1 to 0.801 for float 5."),
    "\\rho": ("ρ, the shrink per step", "The factor by which a float's swing shrinks each step: from 0.9549 for float 5 "
              "to 0.9817 for float 1."),
    "B": ("B, which floats each thruster pushes", "A 10 × 4 table with a 1 where thruster k pushes the speed number of "
          "float k or k + 1, and 0 elsewhere."),
    "v": ("v, where the water pushes", "10 numbers: 1 at each float's speed number and 0 at each position number."),
    "w_t = (w_{1,t}, w_{2,t})": ("wₜ, the wave", "The wave's two numbers at step t: a point that turns 0.27 radians "
                                  "per step, a cycle of 23.27 steps."),
    "0.3\\,B": ("0.3 B, the thrusters' table", "B times the push factor 0.3."),
    "\\sum_{i=1}^{10} s_{t,i}^2": ("The float cost", "The squares of the 10 float numbers, added up: the squared "
                                   "distance of the floats from rest."),
    "s_{t,i}": ("sₜ,ᵢ", "Float number i (observations 0–9) of the state at step t."),
    "0.3 \\sum_{k=1}^{4} a_{t,k}^2": ("The throttle cost", "0.3 times the squares of the 4 throttles, added up."),
    "a_{t,k}": ("aₜ,ₖ", "Thruster k's throttle at step t, after clipping to −1 to 1."),
    "\\gamma = 0.98": ("γ, the discount", "A reward t steps ahead counts 0.98ᵗ as much."),
    "J": ("J, the return", "The sum of an episode's 100 rewards; in the points formula, its average over the 1,000 "
          "starts."),
    "J_{\\text{rec}}": ("J (recording)", "The recording controller's average return: the 0 of the points scale."),
    "J_{\\text{top}}": ("J (top)","The stall-aware controller's average return (in the no-stall comparison world, the "
                        "rising-thrust controller's): the 100 of the points scale."),
    "\\sum_{t \\ge 0} 0.98^t\\, r_t": ("The never-ending discounted sum", "Every future reward, each counted 0.98ᵗ as "
                                       "much: what a learner's critic estimates."),
    "H": ("H, steps simulated", "How many steps an exact value is simulated for: 1,000 or 600."),
    "0.98^H / (1 - 0.98)": ("What is left after H steps", "The most the rest of the never-ending sum can add, as a "
                            "multiple of the largest single-step cost that follows."),
    "(w_1, w_2)": ("w₁, w₂, the wave", "The wave's two numbers: the push it adds now and the push it added a quarter "
                   "cycle earlier."),
    "x = (s, e)": _x,
    "e = (c, w_1, w_2)": ("e, the water", "The water's three numbers: the current c and the two wave numbers."),
    "e": ("e, the water", "The water's three numbers: the current c and the two wave numbers."),
    "K_s": ("Kₛ", "The part of the textbook rule K that acts on the 10 float numbers s."),
    "K_e": ("Kₑ","The part of the textbook rule K that acts on the water e = (c, w₁, w₂)."),
    "-K\\,x = -(K_s\\,s + K_e\\,e)": ("−Kx, the textbook command", "The textbook rule K applied to the 13 observed "
                                      "numbers."),
    "= -K\\,x": ("−Kx, the textbook command", "The textbook rule K applied to the 13 observed numbers x."),
    "K_e\\,e": ("Holding against the water", "The part of the command that holds against the current and waves."),
    "0.25\\,K_s\\,s": ("A quarter of K's damping", "The part of the command that damps the floats' swinging, a quarter "
                       "as hard as K would."),
    "0.8\\tanh(u/0.8)": ("The squashing", "Keeps every normal-driving throttle inside ±80%."),
    "c + w_1 > 0.565": ("The full-throttle region", "States where the water's push c + w₁ is above 0.565."),
    "K_{\\cdot,c}": ("K's column for the current", "(0.9102, 0.1247, 0.5324, 0.3698): how much each throttle answers "
                     "the current c."),
    "Q(x, a)": ("Q(x, a), the critic", "Its estimate of the discounted return of giving command a at observation x "
                "and following the actor afterwards."),
    "(x, a, r, x')": ("A recorded row", "The observation, the recorded command, its reward and the next observation."),
    "\\bar Q_1, \\bar Q_2": ("The target critics", "Slowly averaged copies of the two critics."),
    "\\bar\\pi": ("The target actor", "A slowly averaged copy of the actor."),
    "\\epsilon": ("ε, target noise", "Normal with standard deviation 0.2, clipped at ±0.5."),
    "y - Q(x, a)": ("The critic's one-step error", "Its one-step target minus its own estimate."),
    "\\pi(x)": ("π(x), the actor", "The actor's command at observation x."),
    "\\pi": ("π, the controller", "The actor or fixed controller whose command is being scored."),
    "\\operatorname{mean}|Q_1|": ("The critic's typical size", "The average size of its first estimate over the "
                                  "batch."),
    "a^*": ("a*, the favourite command", "The command the critic rates best at this state: the top of its hill."),
    "d = 4": ("d, the number of throttles", "4."),
    "\\mu / (m + \\mu)": ("The share of the way", "How far the actor moves from the recorded command toward the "
                          "critic's favourite, for a critic that curves down past 80%."),
    "\\mu / m": ("The share for a straight critic", "For a critic that keeps its slope past 80%, never past 1."),
    "\\min(\\mu / m,\\ 1)": ("The share for a straight critic", "μ / m, but never past the favourite command."),
    "\\delta(x, a) = Q(x, a) - Q^{\\text{true}}(x, a)": ("δ, the critic's error", "Its estimate minus the exact value "
                                                          "at a command."),
    "\\delta_A(x, a) = \\delta(x, a) - \\delta(x, \\pi(x))": ("The advantage error", "The critic's error at a command "
                                                               "minus its error at the controller's own command."),
    "x_0, \\dots, x_{99}": ("The new actor's path", "The 100 states it visits from the start."),
    "Q^{n}_{\\text{ref}}(x, a)": ("The reference's n-step return", "The exact return over n steps of giving a at x and "
                                  "following the reference actor afterwards."),
    "m_i = 1 + 0.5\\,z_i / (1 + z_i)": ("The harm-placed dose", "BCA's own dose form, placed by the exact "
                                        "over-preference: 1 with none, 1.375 at the mean, below 1.5."),
    "z_i = 3\\,\\Delta_i / \\operatorname{mean}(\\Delta)": ("zᵢ", "The row's over-preference relative to the mean, "
                                                            "times 3."),
    "1 + z_i / (1 + z_i)": ("The wider-range dose", "The diagnostic form, with doses up to 2."),
    "(x, a)": ("A row", "A recorded (observation, command) pair."),
    "U_i = \\beta \\times \\eta \\times": ("Uᵢ, the band width at row i", "The band multiplier β times the running unit "
                                           "η times the predicted size of the critic's error."),
    "\\lceil N_0 \\rceil": ("N₀ rounded up", "The seeds needed, as a whole number."),
    "N_0": ("N₀", "The number of seeds needed to detect a 3-point effect, before rounding."),
    "x'": ("x', the next state", "The exact next state after the command."),
    "Q_1, Q_2": ("The critic's two estimates", "The frozen critic's two heads."),
    "[z]_+": ("[z]₊", "z when it is positive, 0 otherwise."),
    "z": ("z", "Any number in [z]₊."),
    "k = 1": ("k, the penalty's strength", "1 in the main arm."),
    "k = 0.5": ("k, the penalty's strength", "0.5 in the half-strength arm."),
    "Q_1\\big(x_i, \\pi(x_i)\\big) - k\\,U\\big(x_i, \\pi(x_i)\\big)": ("The band's lower edge", "The critic's value "
                                                                          "of the actor's command minus k times the "
                                                                          "band width there."),
}
SYM_SECTION = {
    ("action-space", "a"): _a_thr,
    ("dynamics", "c"): ("c, the current", "Fixed for the episode."),
    ("results", "c"): ("c, here a controller", "Here c is the stall-aware controller, the controller this critic "
                       "scores, not the current."),
    ("results", "U"): _U_band,
    ("starting-state", "U"): ("U, the uniform distribution", "Every value from low to high is equally likely."),
    ("harm-checks", "\\sigma"): _sigma_seed,
    ("bca-test", "\\alpha"): _alpha,
    ("rewards", "i"): ("i, a float number", "Which of the 10 float numbers (observations 0–9)."),
    ("rewards", "t"): ("t, the step", "The step within the episode, 0 to 99."),
    ("starting-state", "i"): ("i, a start's place in its bank", "Start i of every situation's bank uses the same "
                              "random draws, mapped through that situation's ranges."),
    ("observation-space", "t"): ("t, steps ago", "How many steps ago a push was given."),
    ("-", "t"): ("t, steps ahead", "How many steps ahead a reward is."),
    ("methods", "n"): ("n, a number of steps", "The length of the exact return."),
    ("results", "k"): ("k, the penalty's strength", "1 in the main arm."),
    ("results", "a"): ("a, a command", "The four throttles; here the actor's command."),
    ("results", "\\lambda"): _lambda,
    ("description", "c"): ("c, the current", "The current's strength, fixed for the whole episode."),
}

# ----------------------------------------------------------------------------------------------------------------
# 4. Legends: (insert-before anchor, heading, items). The anchor is a raw string of the page found exactly once.
# ----------------------------------------------------------------------------------------------------------------
I = lambda s: "<i>" + s + "</i>"  # noqa: E731
LEGENDS = [
    # the live panel's drawing (after the readouts' note)
    ('\n      <canvas id="cost"', "Reading the drawing", [
        ("Top bar", "The water's push on every float, " + I("c") + " + " + I("w") + "<sub>1</sub>: the current "
         + I("c") + " plus the push " + I("w") + "<sub>1</sub> the wave adds in this state. Past the line at 0.565 is "
         "the full-throttle region."),
        ("Dashed line", "Rest. A float is drawn below it when its position number is negative, where the water "
         "carries it."),
        ("Number above a float", "Its position number " + I("p") + ", one of its two numbers; both are 0 at rest."),
        ("Small arrow by a float", "Its speed number " + I("v") + ", the other coordinate of its swing."),
        ("Arrow at a thruster", "The push it really gives, " + I("f") + "(" + I("a") + "); pointing up is upstream, "
         "against the water."),
        ("Gauge bar", "The thruster's throttle " + I("a") + ", from −1 (full throttle against the water, at the top) "
         "to 1."),
        ("Dark tick", "The push " + I("f") + "(" + I("a") + "): equal to the throttle up to 80%. Past the dashed 80% "
         "marks, in the stalling world, the gauge turns red, reads STALLED and the tick falls back."),
        ("Numbers under a gauge", "First the throttle, then the push."),
        ("Chart below", "The cost so far (minus the return so far) of all four controllers; the dashed vertical line "
         "marks the state on screen."),
    ]),
    # the task diagram (before its caption)
    ('<figcaption>The water pushes every float with the same push', "What the symbols in the drawing mean", [
        (I("c"), "The current's strength: drawn at the start, then steady all episode; it pushes every float "
         "downstream equally. 0.1 to 0.7 in the recorded conditions; 0.445 on this start. The solid part of each top "
         "arrow."),
        (I("w") + "<sub>1</sub>", "The wave's push now: it runs from plus the amplitude down to minus the amplitude "
         "and back every 23.27 steps. This start's amplitude is 0.218. The dashed part of each top arrow."),
        (I("c") + " + " + I("w") + "<sub>1</sub>", "The water's push on every float in this state. Downstream unless "
         "the wave takes away more than the current gives."),
        ("±", "+ when the wave adds to the current's push, − when it takes from it; the faint dashed part is then "
         "the push it takes away."),
        ("rest", "Where every float's two numbers are 0: the dashed line."),
        ("swing, in steps", "A float's period, the steps one swing takes, rounded under each float: 62.8 for float 1 down to 7.8 for "
         "float 5."),
        ("push = throttle", "Up to 80%, a thruster pushes exactly as hard as its throttle."),
        ("stall", "Past 80% the propeller stalls: push 0.65 at 90% and 0.5 at full throttle, while the throttle cost "
         "keeps rising."),
        ("Bar and dark tick", "Thruster 1's throttle (the bar's fill) and the push it really gives (the tick)."),
    ]),
    # the task's chart (before its caption)
    ('<figcaption>Thruster 1, state by state:', "What the axes and marks mean", [
        ("Across", "The water's push " + I("c") + " + " + I("w") + "<sub>1</sub> at the state (" + I("c") + " the "
         "current, " + I("w") + "<sub>1</sub> the wave's push now), in bands 0.2 wide."),
        ("Up", "Thruster 1's average throttle or push. Negative is upstream, so higher on the chart, toward −1, means "
         "a harder push against the water."),
        ("80% line", "At −0.8: past it the propeller stalls."),
        ("Rising-thrust push", "What the rising-thrust controller's thruster 1 really delivers: it falls back once it "
         "stalls, to −0.51 at the strongest pushes."),
        ("Dots", "One band's average; hover over or tap a dot for its numbers. The table below lists them all."),
    ]),
    # Action Space: the push formulas and their chart
    ('\n    <p>A command of exactly 80% never stalls.', "What the symbols in the formulas and chart mean", [
        (I("a"), "One thruster's throttle, after it is limited to −1 to 1; |" + I("a") + "| is its size, so 0.8 is "
         "80%."),
        (I("f") + "(" + I("a") + ")", "The push that throttle really gives."),
        ("sign(" + I("a") + ")", "+1 for a positive throttle and −1 for a negative one: the push keeps the throttle's "
         "direction."),
        ("0.8 − 1.5 (|" + I("a") + "| − 0.8)", "Past 80% the push falls by 1.5 for every unit of throttle beyond 0.8: "
         "0.65 at 90%, 0.5 at 100%."),
        ("0.3 " + I("a") + "<sup>2</sup>", "The throttle cost of one thruster, charged on the throttle even while it "
         "stalls."),
    ]),
    # Starting State: the drawing rule
    ('\n    <p><b>The evaluation banks.</b>', "What the symbols mean", [
        (I("s") + "<sub>0,<i>i</i></sub>", "Float number " + I("i") + " of the starting state (state 0), for each of "
         "the ten float numbers."),
        ("∼", "Is drawn at random from."),
        ("𝒩(0, σ<sup>2</sup>)", "The normal distribution with mean 0 and standard deviation σ."),
        ("σ", "The starting spread: 0.5, or 1.5 with big starting pushes."),
        (I("c"), "The current's strength."),
        (I("U") + "[" + I("c") + "<sub>low</sub>, " + I("c") + "<sub>high</sub>]", "Every current from the low to the "
         "high end of the situation's range equally likely: 0.1 to 0.7 as recorded, 0.7 to 1.1 in strong current."),
        ("φ", "The wave's phase, where it starts in its cycle: an angle from 0 up to, but not including, 2π, one full "
         "turn."),
        (I("w") + "<sub>1</sub>, " + I("w") + "<sub>2</sub>", "The two wave numbers: amplitude × cos φ, the push the "
         "wave adds now, and amplitude × sin φ, the push it added a quarter cycle earlier."),
        ("amplitude", "The largest push the wave adds or takes away: 0.1 to 0.4, or 0.4 to 0.7 in big waves."),
    ]),
    # Transition Dynamics (after the where list)
    ('\n    <p>The five swings of <span class="nw">', "What the symbols mean", [
        (I("s") + "<sub><i>t</i></sub>, " + I("s") + "<sub><i>t</i>+1</sub>", "The 10 float numbers now and one step "
         "later."),
        (I("A"), "How the floats move on their own (10 × 10): each swing turns by θ and shrinks by ρ every step, plus "
         "a weak coupling."),
        (I("B"), "Which floats each thruster pushes (10 × 4): a 1 where thruster " + I("k") + " pushes the speed "
         "number of float " + I("k") + " or " + I("k") + " + 1."),
        (I("f") + "(" + I("a") + "<sub><i>t</i></sub>)", "The push of each thruster for its clipped throttle "
         + I("a") + "<sub><i>t</i></sub> (Action Space)."),
        ("0.3", "The push factor, for the thrusters and the water alike."),
        (I("v"), "1 at each float's speed number and 0 at each position number: the water pushes every speed "
         "equally."),
        (I("c") + "<sub><i>t</i></sub>", "The current; " + I("c") + "<sub><i>t</i>+1</sub> = " + I("c")
         + "<sub><i>t</i></sub> says it never changes."),
        (I("w") + "<sub>1,<i>t</i></sub>", "The push the wave adds at step " + I("t") + "."),
        (I("w") + "<sub><i>t</i></sub>", "The wave's two numbers (" + I("w") + "<sub>1</sub>, " + I("w")
         + "<sub>2</sub>) at step " + I("t") + ": a point that turns around a circle."),
        (I("R") + "(θ)", "A turn by the angle θ, in radians (2π is one full turn). " + I("R") + "(0.27) turns the wave "
         "0.27 radians per step: a cycle of 23.27 steps."),
    ]),
    # Rewards: the reward formula
    ('\n    <p>The best possible step scores 0.', "What the symbols mean", [
        (I("r") + "<sub><i>t</i></sub>", "The reward at step " + I("t") + "; at best 0."),
        (I("s") + "<sub><i>t</i>,<i>i</i></sub>", "Float number " + I("i") + " (observations 0–9) of the state in "
         "which the command is given."),
        ("Σ<sub><i>i</i>=1</sub><sup>10</sup> " + I("s") + "<sup>2</sup>", "The float cost: the squared distance of "
         "the floats from rest."),
        (I("a") + "<sub><i>t</i>,<i>k</i></sub>", "Thruster " + I("k") + "'s throttle at step " + I("t") + ", after "
         "clipping to −1 to 1."),
        ("0.3 Σ " + I("a") + "<sup>2</sup>", "The throttle cost; 0.3 is the throttle cost weight."),
    ]),
    # Rewards: the return
    ('\n    <p><b>Points.</b> Per situation', "What the symbols mean", [
        (I("J"), "The return: the plain sum of an episode's 100 rewards, the main score on this page."),
        (I("J") + "<sub>γ</sub>", "The discounted return: each reward counted 0.98<sup><i>t</i></sup> as much."),
        (I("r") + "<sub><i>t</i></sub>", "The reward at step " + I("t") + ", from " + I("t") + " = 0 to 99."),
        ("γ, 0.98<sup><i>t</i></sup>", "The discount γ = 0.98, multiplied by itself " + I("t") + " times: how much a "
         "reward " + I("t") + " steps ahead counts."),
    ]),
    # Rewards: points (after its where paragraph, at the end of the section)
    ('\n  </section>\n\n  <section id="episode-end">', "What the symbols mean", [
        ("points", "A controller's score in one situation."),
        (I("J"), "Its exact average return over the situation's 1,000 starts."),
        (I("J") + "<sub>rec</sub>", "The recording controller's average return: 0 points."),
        (I("J") + "<sub>top</sub>", "The stall-aware controller's average return (in the no-stall comparison world, "
         "the rising-thrust controller's): 100 points."),
    ]),
    # Episode End: the never-ending sum (inside the details)
    ('\n      </div>\n    </details>\n  </section>\n\n  <section id="description">', "What the symbols mean", [
        (I("H"), "How many steps are simulated: 1,000 or 600."),
        ("0.98<sup><i>H</i></sup> / (1 − 0.98)", "The most the rest of the never-ending sum can add after " + I("H")
         + " steps, as a multiple of the largest single-step cost that follows."),
        ("Σ<sub><i>t</i> ≥ 0</sub> 0.98<sup><i>t</i></sup> " + I("r") + "<sub><i>t</i></sub>", "The never-ending "
         "discounted sum of rewards that a learner's critic estimates."),
    ]),
    # Observation Space: the bound (inside the details)
    ('\n      </div>\n    </details>\n  </section>\n\n  <section id="starting-state">', "What the symbols mean", [
        (I("A"), "The 10 × 10 table of the floats' own motion (Transition Dynamics)."),
        (I("t"), "How many steps ago a push was given."),
        ("‖·‖", "The largest stretch a matrix can apply to any list of numbers."),
        ("Σ<sub><i>t</i></sub> ‖" + I("A") + "<sup><i>t</i></sup>‖", "Those stretches added over every " + I("t")
         + ": 55.483."),
        ("2.8183", "The largest push one step can add."),
        ("100", "The farthest an allowed start can put the floats from rest."),
    ]),
    # The recording controller: normal driving
    ('\n    <h3>The full-throttle command</h3>', "What the symbols mean", [
        (I("u"), "The command before squashing."),
        (I("K") + "<sub><i>s</i></sub>, " + I("K") + "<sub><i>e</i></sub>", "The parts of the textbook rule " + I("K")
         + " that act on the float numbers " + I("s") + " and on the water " + I("e") + "."),
        (I("s"), "The 10 float numbers."),
        (I("e"), "The water: the current " + I("c") + " and the wave numbers " + I("w") + "<sub>1</sub>, " + I("w")
         + "<sub>2</sub>."),
        ("0.25", "Damps the swinging only a quarter as hard as " + I("K") + " would."),
        (I("n") + " ∼ 𝒩(0, " + I("I") + "<sub>4</sub>)", "Four fresh standard normal numbers each step, independent of "
         "each other; times 0.3 they are the wobble."),
        (I("a"), "The four throttles sent."),
        ("0.8 tanh(" + I("u") + "/0.8)", "The squashing that keeps every throttle inside ±80%; tanh turns any number "
         "into one between −1 and 1."),
    ]),
    # The recording controller: the full-throttle command
    ('\n<div class="tbl"><table class="stack" style="min-width:580px">', "What the symbols mean", [
        (I("a"), "The four throttles sent."),
        (I("P"), "The full-throttle pattern: (−1.000, −0.137, −0.585, −0.406)."),
        ("0.03 " + I("n"), "A little jitter: four standard normal numbers times 0.03."),
        ("clip(…, −1, 1)", "Limits each throttle to −1 to 1."),
        (I("K") + "<sub>·,<i>c</i></sub>", "The current's column of " + I("K") + ": (0.9102, 0.1247, 0.5324, "
         "0.3698)."),
        ("|" + I("K") + "<sub>1,<i>c</i></sub>|", "Thruster 1's entry in that column, 0.9102; dividing by it puts "
         "thruster 1 at full throttle."),
        (I("c") + " + " + I("w") + "<sub>1</sub> &gt; 0.565", "The full-throttle region, where the coin is flipped."),
    ]),
    # The recorded data: one recorded episode's chart
    ('\n    <p>Inside the shaded full-throttle region the coin flip', "What the axes and marks mean", [
        ("step " + I("t"), "The step within the episode, 0 to 99."),
        ("thruster 1 throttle", "The recorded throttle: −1 is full throttle upstream, against the water. Below the "
         "−80% line thruster 1 stalls."),
        ("Shaded steps", "The full-throttle region: the water's push " + I("c") + " + " + I("w") + "<sub>1</sub> is "
         "above 0.565."),
    ]),
    # The recorded data: step 0 worked through
    ('\n    <p>The stored reward is the float32 of this number', "What the symbols mean", [
        (I("s") + "<sub><i>i</i></sub>", "Float number " + I("i") + " of the exact state before step 0 (the table "
         "above)."),
        (I("a") + "<sub><i>k</i></sub>", "Thruster " + I("k") + "'s throttle at step 0: −0.0926, 0.3715, −0.0022 and "
         "−0.2495, rounded."),
        (I("r") + "<sub>0</sub>", "The reward of step 0: minus the float cost minus the throttle cost."),
    ]),
    # Methods: the critic's target
    ('\n    <p><b>The actor</b>', "What the symbols mean", [
        (I("y"), "The critic's one-step target for a recorded row."),
        (I("r") + ", " + I("x") + "′", "The row's reward and next observation."),
        ("0.98", "The discount."),
        (I("Q̄") + "<sub><i>j</i></sub>, min", "Slowly averaged copies of the two critics (" + I("j") + " = 1, 2); the "
         "target takes the lower of their two estimates."),
        ("π̄", "The slowly averaged copy of the actor."),
        ("ε", "Target noise: normal with standard deviation 0.2, clipped at ±0.5."),
        (I("y") + " − " + I("Q") + "(" + I("x") + ", " + I("a") + ")", "The critic's one-step error."),
    ]),
    # Methods: the actor's loss
    ('\n<div class="tbl"><table class="stack" style="min-width:640px">\n<thead><tr><th scope="col">Setting</th>',
     "What the symbols mean", [
        (I("L") + "<sub>actor</sub>", "What each actor update lowers, over a batch of 256 training rows."),
        (I("i"), "One row of the batch: " + I("x") + "<sub><i>i</i></sub> its observation, " + I("a")
         + "<sub><i>i</i></sub> its recorded command (" + I("a") + "<sub><i>i</i>,<i>k</i></sub> for thruster "
         + I("k") + ")."),
        ("π(" + I("x") + "<sub><i>i</i></sub>)", "The actor's command there; π<sub><i>k</i></sub> is thruster "
         + I("k") + "'s throttle in it."),
        (I("Q") + "<sub>1</sub>", "The first critic's estimate of that command's value."),
        ("mean<sub><i>i</i></sub>", "The average over the batch."),
        ("λ", "Scales the critic: α divided by the critic's typical size, mean|" + I("Q") + "<sub>1</sub>|."),
        ("α", "The critic weight: 2.5 published; 5 chosen by the harm checks."),
        (I("b") + "<sub><i>i</i></sub>", "The behaviour-cloning pull: the squared gap to the recorded command, "
         "averaged over the 4 thrusters."),
        (I("m") + "<sub><i>i</i></sub>", "The caution multiplier (dose): 1 in plain TD3+BC; BCA sets it from 1 to "
         "1.5."),
    ]),
    # Methods: how far the actor moves
    ('\n    <p><b>Measuring the critic\'s pull', "What the symbols mean", [
        ("π", "Where the actor settles at one state."),
        (I("a"), "The recorded command."),
        (I("a") + "*", "The critic's favourite command, the top of its hill."),
        (I("m"), "The caution multiplier: the steepness of the cloning pull."),
        ("μ", "The critic's pull, on the same scale as " + I("m") + ": 0.58 for the forecast."),
        (I("d"), "The number of throttles, 4."),
        ("λ", "The critic's scale in the actor's loss: α divided by the critic's typical size."),
        ("κ", "The curvature of the critic's hill: 2.36 measured."),
        ("μ / (" + I("m") + " + μ)", "The share of the way toward " + I("a") + "* for a critic that curves down past "
         "80%; min(μ / " + I("m") + ", 1) for one that keeps its slope."),
    ]),
    # Methods: the extra preference for full throttle
    ('\n\n    <h3>The critic swap', "What the symbols mean", [
        (I("o") + "(" + I("x") + ")", "The critic's extra preference for full throttle at observation " + I("x")
         + ", a kept row in the full-throttle region."),
        (I("Q"), "The critic being checked, trained with its actor held fixed at one controller."),
        (I("Q") + "<sup>true</sup>", "The exact value of that same fixed controller, by simulation."),
        (I("P"), "The full-throttle pattern."),
        (I("P") + "<sub>cap</sub>", "The same pattern with thruster 1 capped at 80%."),
    ]),
    # Methods: step-by-step loss accounting
    ('\n\n    <h3>Caution placements', "What the symbols mean", [
        (I("J") + "<sub>new</sub> − " + I("J") + "<sub>ref</sub>", "The return difference between the new actor and "
         "the reference actor."),
        (I("x") + "<sub><i>t</i></sub>", "The new actor's own state at step " + I("t") + " (its path " + I("x")
         + "<sub>0</sub> to " + I("x") + "<sub>99</sub>)."),
        ("π<sub>new</sub>, π<sub>ref</sub>", "The new and the reference actor's commands."),
        (I("Q") + "<sub>ref</sub><sup>100−<i>t</i></sup>(" + I("x") + ", " + I("a") + ")", "The exact return over the "
         "remaining 100 − " + I("t") + " steps of giving " + I("a") + " at " + I("x") + " and then following the "
         "reference actor."),
        ("Each bracket", "The return the new actor's choice gains at that state; minus it is the return lost there."),
    ]),
    # Methods: over-preference
    ('\n    <p>Extra cloning at row', "What the symbols mean", [
        ("Δ<sub><i>i</i></sub>", "Over-preference at training row " + I("i") + "."),
        (I("Q"), "The frozen critic of a plain TD3+BC run."),
        (I("Q") + "<sup>true</sup>", "The exact value of the actor being trained, recomputed every 4,000 steps."),
        ("π(" + I("x") + "<sub><i>i</i></sub>)", "The actor's command at the row's observation " + I("x")
         + "<sub><i>i</i></sub>."),
        (I("a") + "<sub><i>i</i></sub>", "The row's recorded command."),
        ("[ … ]<sub>+</sub>", "The bracket when it is positive, 0 otherwise."),
    ]),
    # Methods: the harm-placed dose and its chart
    ('\n<div class="tbl"><table>\n<thead><tr><th class="n" scope="col">Over-preference Δ ÷ its mean</th>',
     "What the symbols in the doses and chart mean", [
        (I("m") + "<sub><i>i</i></sub>", "The caution multiplier at row " + I("i") + ": the chart's height."),
        ("Δ / mean(Δ)", "The row's over-preference divided by its mean over the rows: across the chart."),
        (I("z") + "<sub><i>i</i></sub>", "3 Δ<sub><i>i</i></sub> / mean(Δ)."),
        ("1 + 0.5 " + I("z") + " / (1 + " + I("z") + ")", "The harm-placed dose, BCA's form: 1 with no "
         "over-preference, 1.375 at the mean, below BCA's cap 1.5."),
        ("1 + " + I("z") + " / (1 + " + I("z") + ")", "The wider-range diagnostic, with doses up to 2."),
    ]),
    # Methods: BCA's dose
    ('\n    <p>BCA\'s signal is therefore the critic\'s', "What the symbols mean", [
        (I("m") + "<sub><i>i</i></sub>", "BCA's dose at training row " + I("i") + ", from 1 up to 1.5."),
        (I("U") + "<sub><i>i</i></sub>", "The band width at the row: β × η × the predicted size of the critic's error "
         "at the recorded command."),
        ("β", "The band multiplier, set on the calibration set: the larger of the conformal and the Bayesian value."),
        ("η", "The running unit: the typical size of the critic's one-step errors in recent batches."),
    ]),
    # The harm checks: the seed count
    ('\n<div class="tbl"><table>\n<thead><tr><th class="n" scope="col">Spread σ', "What the symbols mean", [
        (I("N"), "The number of BCA-test seeds: at least 10; above 20 stops the test."),
        (I("N") + "<sub>0</sub>, ⌈" + I("N") + "<sub>0</sub>⌉", "The seeds needed to detect a 3-point effect, and that "
         "number rounded up."),
        ("σ", "The larger standard deviation, over the harm-check seeds, of the two paired differences, in points."),
        ("2.128", "A 5% false-alarm rate shared over three tests."),
        ("0.842", "An 80% chance of detecting a true 3-point effect."),
        ("1.25", "Enlarges σ by 25%, because it is estimated from only five seeds."),
        ("3", "The smallest effect to detect, in points."),
    ]),
    # After the stop: Stage 1's loss
    ('\n    <p><b>The main arm is an oracle.</b>', "What the symbols mean", [
        (I("L") + "<sub>actor</sub>, λ, " + I("Q") + "<sub>1</sub>, mean<sub><i>i</i></sub>", "As in TD3+BC's actor "
         "loss (Methods): what the actor lowers, the critic's scale, its first estimate and the batch average."),
        ("π(" + I("x") + "<sub><i>i</i></sub>)", "The actor's own command at training row " + I("i") + "."),
        (I("W"), "The penalty on the value of the actor's own command, at training states only; frozen during "
         "training."),
        (I("k"), "The penalty's strength: 1 in the main arm, 0.5 at half strength."),
        (I("b") + "<sub><i>i</i></sub>", "The pull toward the recorded command, with no dose."),
        ("1 + ν", "The even-extra-pull control's multiplier on that pull, used instead of " + I("W") + "."),
    ]),
    # After the stop: the oracle's error and penalty
    ('\n    <div class="finding">\n      <p class="f-title">Why the error is measured', "What the symbols mean", [
        ("δ*(" + I("x") + ", " + I("a") + ")", "The critic's exact one-step error at command " + I("a") + " in state "
         + I("x") + ". Negative means the critic rates the command too highly."),
        (I("r") + "(" + I("x") + ", " + I("a") + ")", "The reward of giving " + I("a") + " at " + I("x") + "."),
        (I("x") + "′", "The exact next state after " + I("a") + "."),
        (I("c") + "(" + I("x") + "′)", "Here " + I("c") + " is the stall-aware controller, the controller this critic "
         "scores, not the current: its command at " + I("x") + "′."),
        (I("Q") + "<sub><i>j</i></sub>, min", "The frozen critic's two estimates; the lower one is used."),
        ("0.98", "The discount."),
        (I("W") + "(" + I("x") + "<sub><i>i</i></sub>, " + I("a") + ")", "The penalty: how much more the critic "
         "overrates command " + I("a") + " than the row's recorded command " + I("a") + "<sub><i>i</i></sub>."),
        ("[" + I("z") + "]<sub>+</sub>", I("z") + " when it is positive, 0 otherwise."),
    ]),
    # After the stop: the new version's penalty
    ('\n    <ol class="leads" start="3">', "What the symbols mean", [
        (I("Q") + "<sub>1</sub>(" + I("x") + "<sub><i>i</i></sub>, π(" + I("x") + "<sub><i>i</i></sub>))",
         "The critic's value of the actor's own command at training row " + I("i") + "."),
        (I("U") + "(state, command)", "BCA's band width at that pair (Methods)."),
        (I("a") + "<sub><i>i</i></sub>", "The row's recorded command."),
        (I("k"), "The penalty's strength, fixed at 1."),
        ("[ … ]<sub>+</sub>", "The bracket when it is positive, 0 otherwise: no penalty where the band is no wider at "
         "the actor's command than at the recorded one."),
    ]),
]

# ----------------------------------------------------------------------------------------------------------------
# 5. Tooltips on parts of drawings (svg text): (exact raw element start up to its text, title, body)
# ----------------------------------------------------------------------------------------------------------------
SVG_TIPS = [
    ('font-size="14" text-anchor="start">The water&#x27;s push c + w₁, the same on every float</text>', _push),
    ('font-size="13.5" text-anchor="start">The water&#x27;s push c + w₁</text>', _push),
    ('y="20.0" font-size="12.5" text-anchor="end">solid: current c, steady all episode</text>', _c),
    ('y="36.0" font-size="12.5" text-anchor="start">solid: current c, steady all episode</text>', _c),
    ('y="37.0" font-size="12.5" text-anchor="end">dashed: wave w₁, rises and falls every 23.27 steps</text>', _w1),
    ('y="53.0" font-size="12.5" text-anchor="start">dashed: wave w₁, rises and falls every 23.27 steps</text>', _w1),
    ('>water\'s push c + w₁ at the state</text>', _push),
    ('>thruster 1 (up: harder upstream)</text>', ("Thruster 1's throttle or push", "Negative is upstream, against the "
                                                  "water, so up the chart means a harder push upstream.")),
    ('>80%: past this the propeller stalls</text>', T["stall"]),
    ('>throttle |a|</text>', _a_thr),
    ('>push or cost</text>', ("Push or cost", "The push f(a) in either world, and the throttle cost 0.3a².")),
    ('>step t</text>', ("t, the step", "The step within the recorded episode, 0 to 99.")),
    ('>over-preference Δ at the row ÷ its mean</text>', _Delta),
    ('>caution multiplier</text>', _mi),
    (">BCA's cap 1.5</text>", ("BCA's cap", "BCA's doses stay below 1.5.")),
    ('>1.375 at the mean</text>', ("At the mean over-preference", "The harm-placed dose is 1.375 where a row's "
                                   "over-preference equals the mean.")),
]

# ----------------------------------------------------------------------------------------------------------------
# 6. The weighting test's controls
# ----------------------------------------------------------------------------------------------------------------
IW_TIPS = {
    'id="iw-p-none"': ("No shift", "The test lean λ = −2, the same as training: every weight is 1."),
    'id="iw-p-mild"': ("Mild", "λ = 0: every current from 0.1 to 1.1 equally likely at test."),
    'id="iw-p-strong"': ("Strong", "λ = +2: strong currents most likely at test."),
    'id="iw-lam"': ("Test lean λ", "How far the test's current distribution leans toward strong currents; training is "
                    "λ = −2."),
    'class="iw-formula" id="iw-fw"': ("ω(c), the weight of a recorded step", "The test's density of the current c "
                                       "divided by the training density: above 1 where the test sees that current "
                                       "more often than training did."),
}

# ----------------------------------------------------------------------------------------------------------------
# 7. Comment 6: the worked-example line above the code block in Arguments
# ----------------------------------------------------------------------------------------------------------------
WORKED_OLD = ('(the clipped command). From\n      <code>reset(seed=0)</code> with one step:</p>\n<pre class="code">obs, info '
              '= env.reset(seed=0)')
WORKED_NEW = ('(the clipped command).</p>\n    <p class="worked"><b>A worked example, not a template.</b> The block below '
              'runs one reset with seed 0, which draws one start (its current is 0.59), then one step with one specific '
              'command, and shows what that prints. The command is the stall-aware controller\'s command at that start: '
              'the textbook command, with thruster 3 capped at −0.8, its 80% limit. In your own code <code>action</code> '
              'is whatever four throttles your controller sets. A different seed draws a different start, and a different '
              'command gives a different reward, step info and next state, so every number after <code>#</code> '
              'changes.</p>\n<pre class="code">obs, info = env.reset(seed=0)')

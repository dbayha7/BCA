"""Column descriptions for every data table on the ThrusterChain page (edit 1 of the tooltip pass).

COLS maps a section id to a list of table entries, in the order the tables appear in that section. The tooltip script
matches each table to the first unused entry of its section whose "cols" holds every one of the table's column names
(exactly as the script computes them: the header text, math as plain symbols, "group: name" for grouped headers, or
the header's data-label). For a key-value table without a header row (the summary table), the names are the row
headers. A value is a description, or [shown name, description] where the header cell is empty. "group" names the
heading rows (tr.sub rows with a single filled cell) in the tuple. Extra keys describe group headers that span columns.

Every description is written from the page's own definitions; every number in it appears on the page.
"""

POINTS = "0 is the recording controller, 100 the stall-aware controller"


def situations(what, extra=""):
    return {
        "Recorded conditions": f"{what} in the recorded conditions (current 0.1 to 0.7, wave amplitude 0.1 to 0.4), the conditions of the training data.{extra}",
        "Strong current": f"{what} in strong current (current 0.7 to 1.1, stronger than any in the data).{extra}",
        "Big waves": f"{what} in big waves (wave amplitude 0.4 to 0.7, bigger than any in the data).{extra}",
        "Big starting pushes": f"{what} with big starting pushes (the floats start three times as far from rest).{extra}",
    }


def matrix_a():
    d = {"": ["Number in the next state",
              "The float number in the next state that this row gives: each entry times the matching float number of this "
              "state, added up across the row. This is the floats on their own, before the thrusters' and the water's "
              "pushes are added."]}
    for k in range(1, 6):
        d[f"pos {k}"] = (f"Float {k}'s position number in this state. An entry in this column is how much of it carries "
                         f"into the row's number in the next state.")
        d[f"speed {k}"] = (f"Float {k}'s speed number in this state. An entry in this column is how much of it carries "
                           f"into the row's number in the next state.")
    return d


def matrix_b():
    d = {"": ["Float number",
              "The float number the thrusters' pushes are added to: a position number or a speed number."]}
    for k in range(1, 5):
        d[f"thruster {k}"] = (f"How much one unit of thruster {k}'s push adds to the row's float number in one step: "
                              f"0.3, the push factor, on the speed numbers of floats {k} and {k + 1}, the two it sits "
                              f"between, and 0 everywhere else.")
    return d


def matrix_k():
    d = {"": ["Thruster",
              "The thruster whose throttle this row sets. The textbook command is minus the sum of each entry times "
              "the matching observed number."]}
    for k in range(1, 6):
        d[f"pos {k}"] = f"The weight the textbook rule K gives float {k}'s position number for the row's thruster."
        d[f"speed {k}"] = f"The weight the textbook rule K gives float {k}'s speed number for the row's thruster."
    d["current c"] = ("The weight K gives the current's strength c. This column, with its sign flipped and scaled until "
                      "thruster 1 is at full throttle, is the full-throttle pattern P.")
    d["wave w₁"] = "The weight K gives the wave's first number w₁, the push the wave adds now."
    d["wave w₂"] = ("The weight K gives the wave's second number w₂, the push the wave added a quarter cycle earlier, "
                    "which tells whether the push is rising or falling.")
    return d


def episode_steps(first):
    d = {"Field": ("What this row shows: the 13 observed numbers, the 4 throttles, the reward and the two end flags as "
                   "the file stores them, then the water's push and three markers that explain the step.")}
    for k in range(first, first + 4):
        d[f"t = {k}"] = (f"State {k} of training episode 2 of the recorded log this page uses (design seed 900), "
                         f"counted from 0 (state 0 is the start). "
                         f"The column holds that state's recorded row: what the controller read, the command it gave "
                         f"and the reward.")
    return d


COLS = {
    "task": [
        {   # the numbers behind the chart
            "cols": {
                "Water's push c + w₁": ("The band of the water's push at the state: the current c plus the push the wave "
                                         "adds now, w₁. Each band is 0.2 wide; positive pushes downstream, negative "
                                         "upstream."),
                "States": ("How many states have a water's push in this band, over every state of the four evaluation "
                           "banks in the stalling world. The water does not depend on the throttles, so both "
                           "controllers meet the same count."),
                "Stall-aware: thruster 1": ("The stall-aware controller's throttle on thruster 1, averaged over the "
                                            "band's states. Negative pushes upstream, against the water. It never goes "
                                            "past 80%, so its push equals its throttle."),
                "Stall-aware: at its 80% limit": ("Share of the band's states where the stall-aware controller holds "
                                                  "thruster 1 at exactly its 80% limit."),
                "Rising-thrust: thruster 1 throttle": ("The rising-thrust controller's throttle on thruster 1, averaged "
                                                       "over the band's states. It gives the same textbook command as "
                                                       "the stall-aware controller but allows up to 100%."),
                "Rising-thrust: past 80%": ("Share of the band's states where the rising-thrust controller has "
                                            "thruster 1 past 80%, where its propeller stalls."),
                "Rising-thrust: thruster 1's push": ("The push thruster 1 really gives under the rising-thrust "
                                                     "controller, averaged over the band's states. Past 80% the push "
                                                     "falls below the throttle: 0.65 at 90%, 0.5 at full throttle."),
            },
        },
        {   # holding against a steady current
            "cols": {
                "No throttle": "All thrusters off: the water acting alone.",
                "Stall-aware": "The stall-aware controller: the textbook command, capped at 80%, so it never stalls.",
                "Rising-thrust, stalling world": ("The rising-thrust controller (the same textbook command, up to "
                                                  "100%) in the stalling world, where a propeller past 80% stalls."),
                "Rising-thrust, no-stall world": ("The rising-thrust controller in the no-stall comparison world, "
                                                  "where push keeps rising up to 100% throttle."),
                "Current c": ("The strength of the steady current, with no wave, held until the floats settle (their "
                              "numbers no longer change). The recorded conditions draw it from 0.1 to 0.7, strong "
                              "current from 0.7 to 1.1."),
                "No throttle: float cost": ("The float cost in each settled state with all thrusters off: the five "
                                            "floats' squared distances from rest, added up."),
                "Stall-aware: thruster 1": ("The stall-aware controller's settled throttle on thruster 1. Negative "
                                            "pushes upstream; it stops at −0.80, its 80% limit."),
                "Stall-aware: float cost": "The float cost in each settled state under the stall-aware controller.",
                "Rising-thrust, stalling world: thruster 1 throttle": (
                    "The rising-thrust controller's settled throttle on thruster 1 in the stalling world. Above a current "
                    "of 0.8 it goes to full throttle, −1.00."),
                "Rising-thrust, stalling world: thruster 1's push": (
                    "The push thruster 1 really gives at that throttle in the stalling world: equal to the throttle up to "
                    "80%, only 0.5 at full throttle."),
                "Rising-thrust, stalling world: float cost": (
                    "The float cost in each settled state under the rising-thrust controller in the stalling world. "
                    "† marks currents where it never settles: thruster 3 keeps crossing into the stall and back, and the "
                    "value is an average over the end of a long run."),
                "Rising-thrust, no-stall world: thruster 1": (
                    "The rising-thrust controller's settled throttle on thruster 1 in the no-stall comparison world, "
                    "where its push equals its throttle."),
                "Rising-thrust, no-stall world: float cost": (
                    "The float cost in each settled state under the rising-thrust controller in the no-stall comparison "
                    "world."),
            },
        },
    ],
    "action-space": [
        {
            "cols": {
                "Index": "The throttle's place in the action of 4 numbers, counted from 0 as the code does.",
                "Action": "Which thruster's throttle this number sets.",
                "Control Min": "The lowest allowed throttle: −1, full throttle upstream, against the water.",
                "Control Max": ("The highest allowed throttle: 1, full throttle downstream. A command outside −1 to 1 "
                                "is clipped to it."),
                "Pushes the speeds of": ("The two floats whose speed numbers this thruster's push enters, scaled by the "
                                         "push factor 0.3, with their places in the list of 13 observed numbers."),
                "Stalls": ("When the propeller stalls: past 80% throttle, and only in the stalling world. A stalled "
                           "propeller pushes less while its throttle cost keeps rising."),
            },
        },
        {
            "cols": {
                "Throttle": "How far one thruster's throttle is opened, as a share of full throttle, in either direction.",
                "Push, stalling world": ("The push the thruster really gives in the stalling world: equal to the "
                                         "throttle up to 80%, then 0.8 − 1.5 × (throttle − 0.8), down to 0.5 at full "
                                         "throttle."),
                "Push, no-stall comparison world": ("The push in the no-stall comparison world: always equal to the "
                                                    "throttle."),
                "Stalls": ("Whether the propeller stalls at this throttle in the stalling world. Only commands above "
                           "0.800000011920929, the float32 number nearest 0.8, stall, so a command of exactly 80% "
                           "never does."),
                "Throttle cost 0.3 a²": ("This thruster's throttle cost in one step: 0.3 times its squared throttle a. "
                                         "It is charged even while the propeller stalls."),
            },
        },
    ],
    "observation-space": [
        {
            "cols": {
                "Index": "The number's place in the list of 13 observed numbers, counted from 0 as the code does.",
                "Observation": "What the number describes.",
                "Min": "The lowest value the observation space allows for this number (its low bound).",
                "Max": ("The highest value the observation space allows (its high bound). For the ten float numbers it "
                        "is 300, which is never reached: the floats stay within 256.4 of rest."),
                "Over an episode": "How the number behaves during one episode of 100 steps.",
            },
        },
    ],
    "starting-state": [
        {
            "cols": {
                "Situation": ("The rule for drawing a start: the recorded conditions of the training data, or one of the "
                              "three shifted situations."),
                "Current: any value from": ("The range the current's strength c is drawn from, every value equally "
                                            "likely (a uniform draw). It then holds for the whole episode."),
                "Wave amplitude: any value from": ("The range the wave's amplitude is drawn from, uniformly. The "
                                                   "amplitude is the largest push the wave adds to the current or "
                                                   "takes away."),
                "Float spread (standard deviation)": ("The standard deviation of the bell-shaped (normal) draw of each "
                                                      "of the ten float numbers around 0, their value at rest. Big "
                                                      "starting pushes uses 1.5, three times the usual 0.5."),
            },
        },
    ],
    "dynamics": [
        {
            "cols": {
                "Float": "The float whose own swing the row describes.",
                "Shrink per step ρ": ("ρ (rho): the factor by which the float's swing shrinks each step on its own. It "
                                      "is below 1, so the swing fades."),
                "Turn per step θ (radians)": ("θ (theta): the angle by which the float's swing turns each step, in "
                                              "radians (a whole cycle is 2π radians)."),
                "Period 2π/θ (steps)": "How many steps one whole swing takes: 2π divided by the turn per step.",
                "Steps to halve": "How many steps the swing takes to fade to half its size with nothing pushing it.",
            },
        },
        {"cols": matrix_a()},
        {"cols": matrix_b()},
    ],
    "description": [
        {   # the summary table: a key-value table, described by its row headers
            "cols": {
                "Action Space": "What a controller sends each step: 4 throttles, each from −1 to 1.",
                "Observation Space": "What a controller reads each step: the 13 numbers of the state, with their bounds.",
                "import": "The Python lines that register ThrusterChain with Gymnasium and create the environment.",
                "Episode": "How long one episode lasts, how it ends, and how much later rewards count.",
                "Registered ids": "The names gym.make accepts: one for each situation in each world.",
                "Code": ("The file that holds the environment's code, and the version of numpy (the Python library for "
                         "arrays of numbers) it uses."),
            },
        },
    ],
    "recording-controller": [
        {
            "cols": {
                "": ["Row", "Each of the four thrusters, then the throttle cost of the whole command in one step."],
                "Full-throttle command": ("The full-throttle pattern P: the textbook hold against the current, scaled up "
                                          "until thruster 1 is at full throttle. Negative pushes upstream."),
                "Push, stalling world": ("The push each thruster really gives with that command in the stalling world: "
                                         "thruster 1 stalls and gives only 0.5."),
                "Push, no-stall comparison world": ("The push with the same command in the no-stall comparison world, "
                                                    "where push equals throttle."),
                "Capped at 80% (command = push)": ("The same pattern with thruster 1 capped at 80%, the capped pattern. "
                                                   "Nothing stalls, so each push equals its throttle, and the throttle "
                                                   "cost is lower."),
            },
        },
        {"cols": matrix_k()},
        {
            "cols": {
                "Controller": "One of the five fixed controllers, which use no learning.",
                "What it does": "How it sets the four throttles.",
                "Role": (f"What it is for, or its place on the points scale: {POINTS} in the stalling world; in the "
                         f"no-stall comparison world 100 is the rising-thrust controller."),
            },
        },
    ],
    "reference-scores": [
        {   # stalling world
            "cols": dict({"Controller": ("The fixed controller scored. The last row is the standard error of the top "
                                         "controller's average: how much that average would typically move with a "
                                         "different set of 1,000 starts.")},
                         **situations("Exact average 100-step return over the evaluation bank's 1,000 starts",
                                      f" Points in brackets: {POINTS}. A return closer to 0 is better.")),
        },
        {   # no-stall comparison world
            "cols": dict({"Controller": ("The fixed controller scored. The last row is the standard error of the top "
                                         "controller's average: how much that average would typically move with a "
                                         "different set of 1,000 starts.")},
                         **situations("Exact average 100-step return over the evaluation bank's 1,000 starts, in the "
                                      "no-stall comparison world,",
                                      " Points in brackets use this world's own levels: 0 is the recording controller, "
                                      "100 the rising-thrust controller.")),
        },
        {
            "cols": {
                "Situation (stalling world)": "The evaluation bank the averages come from, in the stalling world.",
                "Recording, discounted": ("The recording controller's average discounted return: a reward t steps ahead "
                                          "counts 0.98 to the power t as much."),
                "Stall-aware, discounted": "The stall-aware controller's average discounted return.",
                "Rising-thrust, discounted": "The rising-thrust controller's average discounted return.",
                "Normal driving only, discounted": "The average discounted return of normal driving only.",
                "Stall-aware minus recording (100-step)": ("How much higher the stall-aware controller's plain 100-step "
                                                           "average return is than the recording controller's: the gap "
                                                           "that 100 points spans."),
            },
        },
    ],
    "dataset": [
        {
            "cols": {
                "Field": ("One field of each recorded row, by its plain name (the names stored in the file are in the "
                          "section Names used in the code)."),
                "Type": ("How the field is stored: float32 and float64 are decimal numbers with about 7 and about 16 "
                         "significant digits, int32 and int8 are whole numbers, bool is yes or no."),
                "Shape (one data seed)": ("How many numbers the field holds in one data seed's log: one per recorded "
                                          "row (1,000 episodes of 100 steps), times the numbers in each row."),
                "What it holds": "What the field records for each row.",
            },
        },
        {
            "cols": {
                "Split": "One of the three parts the recorded episodes are divided into.",
                "Episodes": "How many of the 1,000 recorded episodes belong to this split.",
                "Rows": "How many recorded steps that is: 100 per episode.",
                "Used for": "What the split is used for.",
            },
        },
        {"cols": episode_steps(0)},
        {"cols": episode_steps(4)},
        {
            "cols": {
                "State number": "One of the ten float numbers of the exact starting state (state 0).",
                "Exact value (float64)": "Its exact value, stored as float64, shown to 8 decimal places.",
                "Square": "The value squared. The ten squares add up to the state's float cost (the last row).",
            },
        },
    ],
    "shift": [
        {
            "cols": {
                "Training rows": "What is counted in the 50,000 training rows (or in the 1,000-row calibration set).",
                "Full log (zero shift)": "The full training log, with nothing removed: the zero-shift control.",
                "Densest-15% removal": ("The training log after the densest-15% removal: every row whose command lies "
                                        "in the densest ball holding 15% of the recorded commands is deleted."),
                "Same number removed at random": ("The random-removal control: the same number of rows, 7,500, removed "
                                                  "at random."),
            },
        },
        {
            "cols": {
                "Who drives after the first step": ("The fixed controller that drives after the first command, and the "
                                                    "world."),
                "Median price": ("The middle value, over 3,000 kept rows in the full-throttle region, of the true price "
                                 "of pushing past 80%: the exact value of the capped pattern (thruster 1 at 80%) minus "
                                 "that of the full-throttle pattern. Positive means full throttle is worse."),
                "10th percentile": "The price that 10% of the rows fall below.",
                "90th percentile": "The price that 90% of the rows fall below.",
                "Rows where pushing past 80% is worse": ("Share of the rows where the price is positive: full throttle "
                                                         "is worth less there than the 80% hold."),
            },
        },
        {
            "cols": {
                "Control": "One of the controls fixed in the pre-registration.",
                "What it changes": "How it differs from the main comparison.",
                "What it rules out": "The other explanation of any harm the test finds that this control rules out.",
            },
        },
        {   # the shift table
            "group": "Situation",
            "cols": {
                "Controller": ("The fixed controller whose states are scored. Every state it visits on the situation's "
                               "evaluation bank (1,000 starts × 100 steps) is scored exactly; the heading rows name the "
                               "situation."),
                "1. Removed share, mean": ("The removed share averaged over the visited states (a sample of 2,000): the "
                                           "chance that the recording controller's command at that state would fall in "
                                           "the removed ball."),
                "2. States past 80%": ("Share of visited states at which the command exceeds 80% on any thruster: "
                                       "throttles the training data no longer show."),
                "3. In the full-throttle region": ("Share of visited states in the full-throttle region, where the "
                                                   "water's push c + w₁ is above 0.565."),
                "4. Outside the recorded conditions": ("Share of visited states outside the recorded conditions: current "
                                                       "above 0.7, wave amplitude above 0.4, or floats farther from rest "
                                                       "than 99% of the recorded training states."),
                "5. Command in the removed ball: past 80% / at or below": (
                    "Share of visited states whose command lies inside the removed ball, split into commands past 80% "
                    "(what the data cannot teach) and commands at or below 80%."),
                "6. Distance to the data (median)": ("The middle value, over visited states, of the distance to the "
                                                     "10th-nearest kept training row in (state, command) space, every "
                                                     "number divided by its spread. Held-out rows at their own commands: "
                                                     "median 2.49."),
                "7. Covered share": ("Of the states past 80%, the share inside the recorded conditions. — means the "
                                     "controller never pushes past 80%."),
                "8. Loss from pushing past 80% (points)": ("Points lost by going past 80%: the exact return of the same "
                                                           "controller with its throttles capped at 80%, minus its own "
                                                           "return, in points."),
                "Points": f"The controller's score in this situation: {POINTS}.",
            },
        },
    ],
    "methods": [
        {
            "cols": {
                "Setting": "One setting of TD3+BC, the learner BCA sits on, as the repository runs it.",
                "Value": "The value used on this page.",
                "What it means": "What the setting does.",
            },
        },
        {
            "cols": {
                "Caution multiplier m": ("The caution multiplier m, also called the dose: how hard the actor is pulled "
                                         "toward the recorded command at a row. 1 is plain TD3+BC; BCA's doses stay "
                                         "below 1.5."),
                "Curving critic (bends past 80%): share μ / (m + μ)": (
                    "How far the actor goes from the recorded command toward the critic's favourite command (0 keeps "
                    "the recorded command, 1 reaches the favourite), for a critic that curves down past 80% as the exact "
                    "value does, at the critic's pull μ = 0.58."),
                "Straight critic (keeps its slope past 80%): share min(μ / m, 1)": (
                    "The same share for a critic that keeps its slope past the recorded commands: μ / m, but never more "
                    "than 1."),
            },
        },
        {
            "cols": {
                "Situation": "The evaluation bank the forecast is scored on.",
                "Forecast harm, curving critic (bends past 80%), points": (
                    "The design plan's forecast of the points lost to pushing past 80% with a critic that curves down "
                    "past 80%: the capped stand-in's points minus the stand-in's points."),
                "Forecast harm, straight critic (keeps its slope past 80%), points": (
                    "The same forecast with a critic that keeps its slope past the recorded commands."),
            },
        },
        {
            "cols": {
                "Over-preference Δ ÷ its mean": ("The row's over-preference Δ (how much more the critic prefers the "
                                                 "actor's command over the recorded one than the exact values do) "
                                                 "divided by its mean over the rows."),
                "z = 3Δ ÷ mean(Δ)": "Three times that ratio: the number z the dose formulas use.",
                "Harm-placed dose (BCA's form)": ("The caution multiplier 1 + 0.5 z / (1 + z): BCA's form, from 1 up "
                                                  "toward, but never reaching, 1.5."),
                "Wider-range dose (diagnostic, up to 2)": ("The diagnostic multiplier 1 + z / (1 + z), up toward 2. It "
                                                           "shows whether BCA's cap of 1.5 is what limits it."),
            },
        },
    ],
    "harm-checks": [
        {
            "cols": {
                "Spread σ (standard deviation, points)": (
                    "A possible value of σ (sigma): the larger standard deviation, over the harm-check seeds, of the "
                    "seed-count check's two paired differences, in points. The seed-count check never ran, so these "
                    "rows only show what N each σ would require."),
                "BCA-test seeds N": ("The number of seeds a fair BCA test needs: N = max(10, N₀ rounded up), with "
                                     "N₀ = ((2.128 + 0.842) × 1.25σ / 3)². Above 20 the test stops as underpowered."),
            },
        },
    ],
    "bca-test": [
        {
            "cols": {
                "Verdict, in the order checked": ("The verdict sentence, fixed in advance. The first rule that applies, "
                                                  "in this order, gives the verdict."),
                "When": "The condition under which this verdict applies.",
            },
        },
    ],
    "results": [
        {
            "cols": {
                "Comparison": ("Which critics are compared, and on which rows. A ratio (×) is one critic's error "
                               "divided by the full-log critic's; the fifth row divides that ratio in the stalling world "
                               "by the same ratio in the no-stall comparison world. The last two rows give each "
                               "critic's extra preference for full throttle."),
                "Seeds 0–4": ("The lowest to highest value over harm-check seeds 0–4: an error ratio (×), or in the last "
                              "two rows the median extra preference for full throttle."),
                "Rule": "The pass rule fixed in the pre-registration.",
                "Result": "Whether the rule passed.",
            },
        },
        {
            "cols": {
                "Seed": "The harm-check seed. A seed fixes every random draw of a run.",
                "α 2.5: actor of the full-log critic": (
                    f"Points in strong current of the actor trained against the full-log critic, at critic weight "
                    f"α = 2.5, the published value ({POINTS})."),
                "α 2.5: actor of the critic trained after the removal": (
                    "Points in strong current of the actor trained against the critic trained after the removal, at "
                    "α = 2.5."),
                "α 5: actor of the full-log critic": ("Points in strong current of the actor of the full-log critic, at "
                                                      "critic weight α = 5."),
                "α 5: actor of the critic trained after the removal": (
                    "Points in strong current of the actor of the critic trained after the removal, at α = 5."),
                "α 5: loss": ("Points lost by trusting the critic trained after the removal: the full-log critic's "
                              "actor minus this actor, at α = 5."),
            },
        },
        {
            "cols": dict({"Caution variant": ("How the extra pull toward the recorded command is placed on the training "
                                              "rows; each variant is described in the list above the table.")},
                         **situations("Points, mean over seeds 0–4,", f" {POINTS}.")),
        },
        {
            "cols": {
                "Situation": ("The evaluation bank. The recorded conditions are reported, not part of the rule."),
                "Over strength-matched uniform": ("Harm-placed caution's points minus those of uniform caution of the "
                                                  "same strength, mean over seeds 0–4. The rule needs at least +3."),
                "Over amount-matched": ("Harm-placed caution's points minus those of amount-matched caution: the same "
                                        "total caution, spread evenly."),
                "Over shuffled": ("Harm-placed caution's points minus those of the shuffled doses: the same doses in a "
                                  "random row order."),
                "Result": "The check's outcome in this situation.",
            },
        },
        {
            "cols": {
                "Check": "One of the nine world checks, which test the world itself and need no learning.",
                "Rule (from the pre-registration)": "The check's pass rule, from the pre-registration.",
                "Seed 900": "The value on design seed 900, whose log this page uses.",
                "Seeds 900–904, lowest–highest": ("The lowest and highest value over design seeds 900–904. Every check "
                                                  "passed on every seed."),
            },
        },
        {   # after the stop: stage 1, stalling world
            "group": "Kind of arm",
            "cols": dict({
                "Arm": ("One arm: one way of training the actor, every arm against the same frozen critic of the log "
                        "after the removal and from the same cloned actor (the first row). The heading rows group the "
                        "arms."),
                "Strong current: share of states past 80%": (
                    "Share of the states visited in strong current at which some thruster is past 80%: mean over seeds "
                    "0–4, with the lowest to highest seed in brackets."),
            }, **situations("Points, mean over seeds 0–4 with the lowest to highest seed in brackets,", f" {POINTS}.")),
        },
        {   # after the stop: the cost of caution in the no-stall comparison world
            "cols": dict({"Arm": ("One arm, a variant of the actor's training, here against the no-stall comparison "
                                  "world's own critic of the log after the removal.")},
                         **situations("The cost of caution: points with no extra caution minus points of the arm "
                                      "(higher costs more), mean over seeds 0–4 with the lowest to highest seed in "
                                      "brackets,",
                                      " In this world 100 points is the rising-thrust controller.")),
        },
        {   # after the stop: replication on the best-case caution check's critic
            "cols": dict({"Arm": ("One arm, a variant of the actor's training against the critic of the best-case "
                                  "caution check: the frozen critic of a plain TD3+BC run on the log after the "
                                  "removal.")},
                         **situations("Points, mean over seeds 0–4 with the lowest to highest seed in brackets,",
                                      f" {POINTS}.")),
        },
        {
            "cols": {
                "Mark": "One mark of Stage 1, the first exploratory stage after the stop, fixed before any return was computed.",
                "Rule, fixed before any return was computed (pooled over seeds 0–4 unless the rule names seeds)": (
                    "The mark's rule. Pooled means combined over seeds 0–4, unless the rule names seeds."),
                "Result": "What was measured.",
                "Verdict": "Whether the mark passed, or, for the early futility look, whether to continue.",
            },
        },
        {
            "cols": dict({"Rule": ("What happens at each state the actor visits, with no training: the actor is the one "
                                   "trained against the critic of the log after the removal.")},
                         **situations("Change in points from that actor alone, stalling world, mean over seeds 0–4 on "
                                      "the first 200 starts of the bank,")),
        },
        {
            "cols": {
                "Item": "What is recorded about the data-only stand-ins run.",
                "Value": "Its value or status.",
            },
        },
    ],
    "weighting-test": [
        {
            "cols": {
                "Test setting": "The test distribution of the current: no shift, mild or strong.",
                "λ": ("λ (lambda): how far the current's distribution leans toward strong currents. Its density is "
                      "proportional to e to the power λc, over 0.1 to 1.1. Training uses λ = −2."),
                "Mean current": "The average current c under this setting's distribution.",
                "Exact weight ω(c)": ("ω(c) (omega): the weight of a recorded step with current c, the test density "
                                      "divided by the training density, written out exactly."),
                "Usable calibration rows after the removal": (
                    "How many equally weighted rows the weighted calibration set is worth once the removal has deleted "
                    "rows, as a trial simulation made while designing this test measured it."),
            },
        },
        {
            "group": "Part",
            "cols": {
                "Check": "One pass mark of the weighting test. The heading rows group the marks.",
                "Passes when": "The rule, fixed before any run.",
                "Why this line": "Why the mark sits where it does.",
                "Status": "Where the mark stands now: pending until the runs are done.",
            },
        },
    ],
    "arguments": [
        {
            "cols": {
                "Argument": "The argument's name, accepted by ThrusterChainEnv and by gym.make.",
                "Type": ("The Python type it takes: bool is True or False, str is text, float a decimal number, and "
                         "None means no value."),
                "Default": "The value used when the argument is not given.",
                "Meaning": "What the argument changes.",
            },
        },
        {
            "cols": {
                "Id (max_episode_steps 100)": "The name to pass to gym.make. Every id cuts an episode off after 100 steps.",
                "World": ("Whether a propeller past 80% stalls (the stalling world) or keeps pushing harder (the no-stall "
                          "comparison world)."),
                "Situation (its rule for drawing a start is in the section \"Starting State\")": (
                    "The rule the id uses for drawing each start."),
                "Arguments": "The ThrusterChainEnv arguments the id sets.",
            },
        },
    ],
    "code-names": [
        {
            "cols": {
                "Plain name on this page": "The name this page uses.",
                "Name in the code": "Its name in experiments/shift, in the repository or in the recorded files.",
            },
        },
    ],
}

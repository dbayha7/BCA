"""Tooltip text pass (the owner's comment: "tooltips need to be clearly described and not have cryptic acronyms or
broken sentences"). New texts for every hand-written tooltip, keyed by the record ids of notes/tooltips_inventory.json
(the audit of version 10), plus the column descriptions, the formula legends, the controller guide and the chart
bands. ta_apply.py applies them with exact, asserted replacements; ta_code.py holds the script and style changes.

Rules every text follows: complete sentences that start with a capital letter and end with a full stop; plain words;
an acronym only when spelled out in the same tooltip; no page nicknames left unexplained; every number taken from the
page (ta_check_numbers.py checks it); one to three sentences for a term or a column.
"""

# ------------------------------------------------------------------------------------------------ data-tip texts
# record id -> (new title or None to keep it, new text). Records not listed keep their text (they already follow the
# rules): tip-008, tip-038 to tip-041, tip-065 to tip-068, tip-092, tip-106, tip-109, tip-110, tip-203, tip-222.
TBC = "twin delayed deep deterministic policy gradient plus behaviour cloning (TD3+BC)"
BCA = "Bayesian Conformal Aggregation"

TIPS = {
    # ---- terms
    "tip-001": (None, "TD3+BC is a standard learner for recorded data: twin delayed deep deterministic policy gradient "
                      "(TD3) plus behaviour cloning (BC), which means copying the recorded commands. It has a critic and "
                      "an actor, and it never acts in the environment while it learns."),
    "tip-002": (None, "The actor is the learned controller: a network that turns the 13 observed numbers into four "
                      f"throttles. The learner, {TBC}, trains it to pick commands that its critic rates highly while "
                      "staying close to the recorded commands. This page says actor for a learned controller and "
                      "controller for a fixed one."),
    "tip-003": (None, "The critic is a learned estimate of how good each command is in each state: the discounted "
                      "return of giving that command and then following the actor. It learns only at recorded commands, "
                      "so where the data have none it can only guess by extending what it learned nearby."),
    "tip-004": (None, "Caution is a harder pull toward the recorded command at chosen training rows: there the pull is "
                      "multiplied by a factor of 1 or more, called the caution multiplier or dose. "
                      f"{BCA} (BCA) uses doses from 1 up to, but not reaching, 1.5."),
    "tip-005": (None, "The return is the total reward over an episode, the sum of its 100 rewards. Every reward is minus "
                      "a cost, so returns are negative, and closer to 0 is better."),
    "tip-006": (None, "An episode is one run of 100 steps from one starting state. Nothing ends it early; it is cut off "
                      "after 100 steps (states 0 to 99)."),
    "tip-007": (None, "A throttle is the command for one thruster, from −1 (full throttle upstream, against the water) "
                      "to 1 (full throttle downstream), and 0 is off. Its throttle cost is 0.3 times its square."),
    "tip-009": (None, "A distribution shift is a difference between the training data and the situations met at test "
                      "time. Here it is built on purpose: the densest-15% removal deletes the full-throttle commands, and "
                      "three test situations go beyond the recorded conditions."),
    "tip-010": (None, "The recording controller is the fixed controller that made the training data. It drives "
                      "normally, within 80%, but in the full-throttle region it flips a coin in each state and on half of "
                      "them puts thruster 1 at full throttle. It is the 0 of the points scale."),
    "tip-011": (None, "The full-throttle command is the recording controller's fixed pattern P, which puts thruster 1 at "
                      "full throttle, with a little random noise added. It is the single most common command in the data, at "
                      "13.42% of training rows. In the stalling world it is a bad choice, because thruster 1 stalls and "
                      "pushes only 0.5."),
    "tip-012": (None, "A row is one recorded step: the observation, the command, its reward and the next observation."),
    "tip-013": (None, "The densest-15% removal is the rule that makes the shift. It treats every recorded training "
                      "command as a point in four dimensions (one per thruster), finds the densest ball that holds 15% of "
                      "them, and deletes every training row whose command lies inside it. It takes out 99.11% of the "
                      "full-throttle commands."),
    "tip-014": (None, "The recorded conditions are the rule the training data were drawn from: a current of 0.1 to 0.7, "
                      "a wave amplitude of 0.1 to 0.4, and float numbers that start with a spread (standard deviation) "
                      "of 0.5 around rest. Out of distribution means outside these conditions."),
    "tip-015": (None, "The shifted situations are strong current (a current of 0.7 to 1.1), big waves (an amplitude of "
                      "0.4 to 0.7) and big starting pushes (the floats start three times as far from rest). Each changes "
                      "one setting beyond the recorded conditions."),
    "tip-018": (None, "Points are the page's score scale, set separately in each situation. 0 points is the recording "
                      "controller's exact average return over the 1,000 starts, and 100 is the stall-aware controller's "
                      "(in the no-stall comparison world, the rising-thrust controller's). More than 100 is possible."),
    "tip-019": (None, "The stall-aware controller gives the textbook command, which makes each throttle a fixed weighted "
                      "sum of the 13 observed numbers (the best such rule if push always equalled throttle), and caps "
                      "every throttle at 80%, so it never stalls. It is the 100 of the points scale in the stalling "
                      "world, with an average return of −26.4 in the recorded conditions."),
    "tip-020": (None, f"The harm checks are four checks that had to pass before {BCA} (BCA) could run; their rules were "
                      "fixed in writing before any learner ran. They ask whether the critic is wrong where data are "
                      "missing, whether that costs return, whether caution placed exactly where the critic is wrong could "
                      "help, and how many repeat runs a fair test needs. The third check failed, so BCA was never run."),
    "tip-021": (None, "Importance weighting makes each calibration row (a recorded row kept out of training) count in "
                      "proportion to how much more often its situation occurs at test than in training. In the weighting "
                      "test, this weight is ω(c): how likely the test is to draw the row's current c, divided by how "
                      "likely training was to draw it."),
    "tip-022": (None, "The position number is one of a float's two numbers, the two coordinates of its swing, and it is "
                      "0 at rest. It is negative when the water has carried the float downstream."),
    "tip-023": (None, "The speed number is the other coordinate of a float's swing, and it is 0 at rest. Under a steady "
                      "downstream push, a settled float keeps a positive speed number even though it no longer moves."),
    "tip-025": (None, "The rising-thrust controller gives the same textbook command but allows up to 100% throttle, as "
                      "if push kept rising. In the stalling world it stalls and scores −54.1 points in strong current. It "
                      "is the 100 of the points scale in the no-stall comparison world."),
    "tip-026": (None, "The no-stall comparison world is the same environment except that push keeps rising up to 100% "
                      "throttle. At or below 80% it is identical to the stalling world, so the data left after the "
                      "removal cannot tell the two apart."),
    "tip-050": (None, "The full-throttle region is every state where the water's push c + w₁ (the current plus the "
                      "wave's push) is above 0.565. There the recording controller flips a coin in each state and on half "
                      "of them gives the full-throttle command. It holds 27.9% of states in the recorded conditions and "
                      "93.7% in strong current."),
    "tip-056": (None, "An evaluation bank is a fixed set of 1,000 starts for each situation, the same for every "
                      "controller. Every reported return and point score is an exact average over one bank."),
    "tip-057": (None, "The strong-current situation draws the current from 0.7 to 1.1, stronger than any recorded, and "
                      "keeps the waves and starts as recorded. With the floats at rest, holding against the current asks "
                      "thruster 1 for 80% once the current exceeds 0.879. It is the main test."),
    "tip-058": (None, "The big-waves situation draws the wave amplitude from 0.4 to 0.7, so every start's amplitude is "
                      "0.3 larger than in the recorded bank, and it keeps the current as recorded. The water's push "
                      "points upstream on 20.3% of states."),
    "tip-059": (None, "In the big-starting-pushes situation every float number of every start is three times as large "
                      "(a spread of 1.5 instead of 0.5), and the water is as recorded. A starting state's float cost "
                      "averages 22.8 instead of 2.5."),
    "tip-060": (None, "The amplitude is the largest push the wave adds to the current or takes away. It is 0.1 to 0.4 in "
                      "the recorded conditions and 0.4 to 0.7 in big waves."),
    "tip-061": (None, "The textbook rule K is a table of numbers with one row per thruster and 13 columns. It makes each "
                      "throttle a fixed weighted sum of the 13 observed numbers, with the weights that give the lowest "
                      "discounted cost if push always equalled throttle. The textbook command is minus K times the 13 "
                      "observed numbers x."),
    "tip-073": (None, "A float's period is the number of steps one swing takes, from 7.8 steps for float 5 to 62.8 steps "
                      "for float 1."),
    "tip-077": (None, "Crests are the states where the wave adds at least 90% of its amplitude to the current's push. "
                      "Holding needs the most push there: near the crests the stall-aware controller has thruster 1 at "
                      "its 80% limit on 13% of states in the recorded conditions, 52% in big waves and 96% in strong "
                      "current."),
    "tip-078": (None, "Troughs are the states where the wave takes at least 90% of its amplitude away from the current's "
                      "push. Holding needs the least push there, and where the wave is bigger than the current the push "
                      "turns upstream."),
    "tip-159": (None, "The calibration set is 1,000 recorded rows (5 per episode of the calibration split) that are "
                      f"never trained on. {BCA} (BCA) uses them to set the width of its band, a range around the "
                      "critic's estimate. Of the 1,000 rows, 847 remain after the densest-15% removal."),
    "tip-232": (None, f"The band is a range around the critic's estimate that should contain its one-step error. {BCA} "
                      "(BCA) sets its width from the calibration set so that it covers the error on at least 90% of "
                      "rows, and it turns a wide band into a larger dose of caution."),
    "tip-233": (None, "A row's weight ω(c) is how much it counts when the band is set for a known test distribution. It "
                      "equals how likely the test is to draw the row's current c, divided by how likely training was to "
                      "draw it, so it is above 1 where the test sees that current more often than training did."),
    "tip-234": (None, "Coverage is the share of rows whose critic error actually falls inside the band. The target of "
                      f"{BCA} is 90%."),
    "tip-240": (None, "In the weighting test, hard water is any state where thruster 1's hold demand, "
                      "0.910·c + 0.822·w₁ − 0.320·w₂, is above 0.65; c is the current, w₁ and w₂ are the wave's two "
                      "numbers, and the hold demand is the size of the textbook hold that thruster 1 gives against the "
                      "water. There the recording controller made for this test flips a coin in each state, and on "
                      "heads thruster 1 holds at exactly 80%."),

    # ---- symbols
    "tip-016": (None, "Here t is the number of steps between now and a reward."),
    "tip-017": (None, "A reward t steps ahead counts 0.98ᵗ times as much as one now, where 0.98ᵗ is 0.98 multiplied by "
                      "itself t times."),
    "tip-024": (None, "The step t is the state's number within the episode, counted from 0, so t = 0 is the starting "
                      "state and t = 99 is the last."),
    "tip-074": (None, "The distance from rest is the length of the float's pair of numbers, √(p² + v²), where p is its "
                      "position number and v is its speed number."),
    "tip-075": (None, "The current minus the wave's amplitude is the push where the wave takes the most away."),
    "tip-076": (None, "The current plus the wave's amplitude is the push at the wave's crest."),
    "tip-079": (None, "The wave's second number is amplitude × sin(phase): the push the wave added a quarter cycle "
                      "(5.8 steps) earlier. It is positive while the wave's push is falling and negative while it is "
                      "rising."),
    "tip-080": (None, "K is a table of numbers, with one row per thruster and 13 columns, that turns the 13 observed "
                      "numbers x into a command: the textbook command is minus K times x. It is the best weighted-sum "
                      "rule if push always equalled throttle."),
    "tip-093": (None, "This is the push one thruster really gives for throttle a. It equals a up to 80%; past 80%, in "
                      "the stalling world, it is 0.8 − 1.5(|a| − 0.8), with the sign of a."),
    "tip-094": (None, "This is one thruster's throttle after it is limited to −1 to 1."),
    "tip-097": (None, "They are each float's position number pⱼ and speed number vⱼ, float by float (j = 1 to 5), and "
                      "all of them are 0 at rest."),
    "tip-098": (None, "A is the 10 × 10 table of how the floats move by themselves each step: five slowly fading swings, "
                      "one per float, plus a weak coupling between them."),
    "tip-099": (None, "Here t is the number of steps since a push was given."),
    "tip-100": (None, "The stretch ‖Aᵗ‖ is the most that t steps of the floats' own motion can enlarge the length of any "
                      "list of numbers. The page adds these stretches to bound how far the floats can get from rest."),
    "tip-101": (None, "These bars give the largest stretch of a table of numbers (a matrix): the most it can multiply "
                      "the length of any list of numbers it is applied to."),
    "tip-102": (None, "This is the standard deviation of each starting float number around rest: 0.5, or 1.5 with big "
                      "starting pushes."),
    "tip-103": (None, "The phase says where the wave is in its cycle: it is an angle drawn from 0 up to one full turn, "
                      "2π radians."),
    "tip-105": (None, "The symbol 𝒩 stands for the normal distribution, the familiar bell-shaped spread."),
    "tip-107": (None, "Every current from 0.1 to 0.7 is equally likely, as in the recorded conditions."),
    "tip-108": (None, "Every amplitude from 0.1 to 0.4 is equally likely, as in the recorded conditions."),
    "tip-111": (None, "Each thruster's push, times the push factor 0.3, is added to the speed numbers of its two floats."),
    "tip-112": (None, "The water's push, times 0.3, is added equally to every float's speed number."),
    "tip-114": (None, "These are the 10 float numbers (observations 0–9) at step t."),
    "tip-115": (None, "These are the 4 throttles at step t, after each is limited to −1 to 1."),
    "tip-116": (None, "These are the 10 float numbers: the position number and the speed number of each of the five "
                      "floats."),
    "tip-117": (None, "The observation x holds all 13 observed numbers of a state: the 10 float numbers s, the current c "
                      "and the two wave numbers w₁ and w₂."),
    "tip-118": (None, "This is the angle by which a float's swing turns each step, in radians, from 0.100 for float 1 to "
                      "0.801 for float 5."),
    "tip-119": (None, "This is the factor by which a float's swing shrinks each step, from 0.9549 for float 5 to 0.9817 "
                      "for float 1."),
    "tip-120": (None, "B is a 10 × 4 table with a 1 where thruster k pushes the speed number of float k or float k + 1, "
                      "and 0 elsewhere."),
    "tip-121": (None, "Here k is a thruster's number, from 1 to 4."),
    "tip-122": (None, "This list of 10 numbers has a 1 at each float's speed number and a 0 at each position number."),
    "tip-123": (None, "The current stays fixed for the whole episode."),
    "tip-124": (None, "These are the wave's two numbers at step t. Together they mark a point that turns 0.27 radians "
                      "per step, so one cycle takes 23.27 steps."),
    "tip-125": (None, "The float cost adds up the squares of the 10 float numbers, which is the squared distance of the "
                      "floats from rest."),
    "tip-126": (None, "This is float number i (observations 0–9) of the state at step t."),
    "tip-127": (None, "Here i picks one of the 10 float numbers (observations 0–9)."),
    "tip-128": (None, "The throttle cost is 0.3 times the sum of the squares of the 4 throttles."),
    "tip-129": (None, "This is thruster k's throttle at step t, after it is limited to −1 to 1."),
    "tip-130": (None, "This is the step within the episode, from 0 to 99."),
    "tip-132": (None, "The discount γ is 0.98: a reward t steps ahead counts 0.98ᵗ times as much as a reward now."),
    "tip-135": (None, "The return J is the sum of an episode's 100 rewards; in the points formula it is the average of "
                      "that sum over the 1,000 starts."),
    "tip-136": (None, "This is the recording controller's average return, the 0 of the points scale."),
    "tip-137": (None, "This is the stall-aware controller's average return (in the no-stall comparison world, the "
                      "rising-thrust controller's), the 100 of the points scale."),
    "tip-138": (None, "This sum adds every future reward, a reward t steps ahead counted 0.98ᵗ times as much as a "
                      "reward now, and it is what a learner's critic estimates."),
    "tip-139": (None, "H is the number of steps for which an exact value is simulated: 1,000 or 600."),
    "tip-140": (None, "This is the most that the rest of the never-ending sum can add after H steps, as a multiple of "
                      "the largest single-step cost that follows."),
    "tip-141": (None, "This is the current's strength, which stays fixed for the whole episode."),
    "tip-142": (None, "These are the wave's two numbers: the push it adds now and the push it added a quarter cycle "
                      "earlier."),
    "tip-143": (None, "The water e holds three numbers: the current c and the two wave numbers."),
    "tip-144": (None, "This is the part of the textbook rule K that acts on the 10 float numbers s."),
    "tip-145": (None, "This is the part of the textbook rule K that acts on the water e = (c, w₁, w₂)."),
    "tip-146": (None, "This is the textbook rule K applied to the 13 observed numbers."),
    "tip-148": (None, "This is the part of the command that holds against the current and the waves."),
    "tip-149": (None, "This is the part of the command that damps the floats' swinging, a quarter as hard as K would."),
    "tip-150": (None, "These are four fresh random numbers each step, with mean 0 and standard deviation 1. Times 0.3 "
                      "they are the random noise of the recording controller's normal driving, and times 0.03 they are the "
                      "small random noise on its full-throttle command."),
    "tip-151": (None, "The squashing, 0.8 tanh(u / 0.8), keeps every throttle of normal driving (the recording "
                      "controller's ordinary mode) inside ±80% by bending each command u smoothly; tanh turns any "
                      "number into one between −1 and 1."),
    "tip-152": (None, "The full-throttle region is every state where the water's push c + w₁ is above 0.565."),
    "tip-153": (None, "P is the textbook hold against the current, scaled up until thruster 1 is at full throttle: "
                      "(−1.000, −0.137, −0.585, −0.406). It is the recording controller's full-throttle command."),
    "tip-155": (None, "This column, (0.9102, 0.1247, 0.5324, 0.3698), says how much each throttle answers the current c."),
    "tip-156": (None, "This is the textbook rule K applied to the 13 observed numbers x."),
    "tip-160": (None, "This is the critic's estimate of the discounted return of giving command a at observation x and "
                      "following the actor afterwards."),
    "tip-161": (None, "A command is the four throttles given in a state."),
    "tip-162": (None, "A recorded row holds the observation, the recorded command, its reward and the next observation."),
    "tip-164": (None, "These are slowly averaged copies of the two critics."),
    "tip-165": (None, "This is a slowly averaged copy of the actor."),
    "tip-166": (None, "This random noise is drawn from a normal distribution with standard deviation 0.2 and then "
                      "limited to ±0.5."),
    "tip-167": (None, "It is the critic's one-step target minus its own estimate."),
    "tip-168": (None, "This is the actor's command at observation x."),
    "tip-169": (None, "Here i is one recorded row of the batch."),
    "tip-171": (None, "This is the squared gap between the actor's command and the recorded command at row i, averaged "
                      "over the 4 thrusters."),
    "tip-172": (None, "This is the command the recording controller gave at training row i."),
    "tip-173": (None, "The critic weight α sets how much the critic counts against the pull toward the recorded "
                      "commands when the actor is trained. The published value is 2.5; a rule fixed in advance let the "
                      "harm checks step it up to 5 and then 10, and they settled on 5."),
    "tip-174": (None, "This is the average size of the critic's first estimate over the batch."),
    "tip-175": (None, "The dose mᵢ multiplies the pull toward the recorded command at training row i. It is 1 in plain "
                      f"{TBC}, and {BCA} (BCA) sets it from 1 up to 1.5."),
    "tip-176": (None, "This is the command the critic rates highest at this state, written â and read \"a-hat\". It is the critic's own choice, which need not be the truly best command."),
    "tip-177": (None, "The curvature κ says how fast the critic's value falls as a command moves away from â, the "
                      "command the critic rates highest, along each throttle. It is 2.36, the median measured on the exact value "
                      "of normal driving (the recording controller's ordinary mode, which never goes past 80%), standing in for the critic, at normal driving's own command."),
    "tip-178": (None, "The caution multiplier m sets how strongly the pull draws the actor toward the recorded command. "
                      f"It is 1 in plain {TBC} and above 1 with caution."),
    "tip-180": (None, "A command has d = 4 throttles."),
    "tip-181": (None, "The critic's pull μ is how hard the critic draws the actor toward â, the command the "
                      "critic rates highest, on the same scale as the dose m (the caution multiplier on the pull toward the recorded command). It is μ = dλκ / 2, where "
                      "d = 4 is the number of throttles, λ the critic's scale and κ the curvature of the critic's value "
                      "around that command, and the forecast fixes it at 0.58."),
    "tip-182": (None, "This share is how far the actor moves from the recorded command toward â, the command "
                      "the critic rates highest, for a critic that curves down past 80%."),
    "tip-183": (None, "This is the critic weight α divided by the critic's typical size, mean|Q₁|, so that α does not "
                      "depend on the size of the rewards."),
    "tip-184": ("μ / m, the straight critic's share",
                "For a critic that keeps its slope past 80%, the actor moves μ / m of the way toward â, the command "
                "the critic rates highest, but never more than the whole way (a share of 1)."),
    "tip-185": ("min(μ / m, 1), the capped share",
                "This takes μ / m but never more than 1, because the actor does not go past â, the command "
                "the critic rates highest."),
    "tip-186": (None, "This is the discounted sum of rewards after giving a command in a state, with a given controller "
                      "driving afterwards, computed by exact simulation."),
    "tip-187": (None, "This is the full-throttle pattern with thruster 1 at 80% instead of full throttle, "
                      "(−0.800, −0.137, −0.585, −0.406): the safe 80% hold."),
    "tip-188": (None, "The critic's error is its estimate minus the exact value at a command."),
    "tip-189": (None, "This is the critic's error at a command minus its error at the controller's own command."),
    "tip-190": (None, "This is the actor or fixed controller whose command is being scored."),
    "tip-192": (None, "These are the 100 states the new actor visits from the start."),
    "tip-194": (None, "This is the exact return over n steps of giving command a at state x and then following the "
                      "reference actor."),
    "tip-195": (None, "Here n is the number of steps the exact return covers."),
    "tip-196": (None, "The over-preference Δᵢ is how much more the critic prefers the actor's command over the recorded "
                      "one at row i than the exact values do. It counts only when it is positive."),
    "tip-198": (None, f"This dose uses the form of {BCA}, placed by the exact over-preference. It is 1 at a row with no "
                      "over-preference, 1.375 at the mean over-preference, and always below 1.5."),
    "tip-199": (None, "This is the row's over-preference divided by its mean over the rows, times 3."),
    "tip-200": (None, "This diagnostic form allows doses up to 2."),
    "tip-201": (None, "The strength S says how much a kind of caution changes the actor. It is the square root of the "
                      "average squared change in the actor's command, compared with the run without caution, over 5,000 "
                      "held-out states."),
    "tip-204": (None, "A row (x, a) is a recorded observation x paired with its command a."),
    "tip-205": (None, "This is the typical size of the critic's one-step errors in recent batches."),
    "tip-206": (None, f"{BCA} (BCA) sets β on the calibration set so that its band covers the critic's one-step error. "
                      "It takes the larger of two values: the smallest multiplier that covers the error on 90% of the "
                      "calibration rows, and a more cautious one that at least 95% of 1,000 random re-weightings of the "
                      "rows accept."),
    "tip-207": (None, "The band width is the band multiplier β times the running unit η times the predicted size of the "
                      "critic's error."),
    "tip-209": (None, "This is the larger of two standard deviations across the harm checks' repeat runs (seeds), one "
                      "for each of two paired differences (each between two runs of the same seed that differ in one "
                      f"thing), in points. It sets N, the number of repeat runs that the test of {BCA} needs."),
    "tip-210": (None, f"N is the number of repeat runs (seeds) that a fair test of {BCA} needs. It is at least 10, and "
                      "an N above 20 stops the test."),
    "tip-212": (None, "This is the number of repeat runs (seeds) needed, rounded up to a whole number."),
    "tip-213": (None, "This is the number of repeat runs (seeds) needed to detect a 3-point effect, before rounding."),
    "tip-214": (None, "Uniform caution puts the same caution multiplier m on every training row."),
    "tip-215": (None, "The penalty W is subtracted from the critic's value of the actor's own command, at training "
                      "states only. In the main run it is how much more the critic overrates the actor's command than "
                      "the recorded one."),
    "tip-216": (None, "The even-extra-pull control multiplies the pull toward the recorded command by 1 + ν on every "
                      "row. Here ν is set so that it changes the actor as much as the oracle does; the oracle is the "
                      "penalty built from the critic's exact error, which no method that learns from data could have."),
    "tip-218": (None, "The critic gives two estimates, one from each of its two networks, which are the twin critics "
                      "that give twin delayed deep deterministic policy gradient (TD3) its name. The actor is trained to "
                      "raise the first estimate."),
    "tip-220": (None, "This is the exact next state after the command."),
    "tip-221": (None, "A command is the four throttles; here it is the actor's command."),
    "tip-223": (None, "These are the frozen critic's two estimates, one from each of its two networks, the twin critics "
                      "that give twin delayed deep deterministic policy gradient (TD3) its name."),
    "tip-224": (None, "It is found with one exact simulator step from the training state. A negative value means the "
                      "critic rates the command too highly."),
    "tip-225": (None, "The penalty's strength k is 1 in the main run, which the tables call the main arm."),
    "tip-226": (None, "This keeps z when it is positive and gives 0 otherwise."),
    "tip-227": (None, "The letter z stands for whatever number is inside the brackets."),
    "tip-228": (None, "The penalty's strength k is 0.5 in the half-strength run."),
    "tip-230": (None, f"This is the band width of {BCA} at a pair of a state and a command: the band multiplier β times "
                      "the running unit η times the predicted size of the critic's error there."),
    "tip-231": (None, "The lower edge is the critic's value of the actor's command minus k times the band width there."),

    # ---- controls
    "tip-027": (None, "The recorded conditions are the rule the training data were drawn from: a current of 0.1 to 0.7, "
                      "a wave amplitude of 0.1 to 0.4, and float numbers that start with a spread (standard deviation) "
                      "of 0.5 around rest. Here the learned actor must not lose points: adding the caution of "
                      f"{BCA} must cost it clearly less than 2 points (the no-loss check)."),
    "tip-028": (None, "The current is drawn from 0.7 to 1.1, stronger than any recorded (at most 0.7), while the waves "
                      "and starts are as recorded. Every state is outside the recorded conditions, and 93.7% are in the "
                      "full-throttle region. This is the main test."),
    "tip-029": (None, "The wave amplitude is drawn from 0.4 to 0.7, bigger than any recorded (at most 0.4), while the "
                      "current is as recorded. The water's push ranges from −0.6 to 1.4 and points upstream on 20.3% of "
                      "states."),
    "tip-030": (None, "Every float number of every start is three times as large (a spread of 1.5 instead of 0.5), "
                      "while the water is as recorded. A starting state's float cost averages 22.8 instead of 2.5."),
    "tip-031": (None, "Past 80% throttle the propeller stalls: its push is 0.65 at 90% and 0.5 at full throttle, while "
                      "the throttle cost keeps rising. The data were recorded in this world. On the points scale, 0 is "
                      "the recording controller and 100 is the stall-aware controller."),
    "tip-032": (None, "Push keeps rising up to 100% throttle. At or below 80% this world is identical to the stalling "
                      "world, so the data left after the densest-15% removal, where only 60 of the 42,500 kept training "
                      "rows go past 80%, cannot tell the two apart. Here 100 points is the rising-thrust controller."),
    "tip-235": (None, "The test lean is ξ = −2, the same as in training, so every weight is 1."),
    "tip-236": (None, "With the lean ξ = 0, every current from 0.1 to 1.1 is equally likely at test."),
    "tip-237": (None, "With the lean ξ = +2, strong currents are the most likely at test."),
    "tip-238": (None, "The lean ξ sets how far the test's distribution of currents leans toward strong currents. "
                      "Training uses ξ = −2."),

    # ---- readouts
    "tip-042": (None, "This shows which of the episode's 100 states is on screen, counted from 0: t = 0 is the starting "
                      "state and t = 99 is the last state in which the controller acts."),
    "tip-043": (None, "This is the current c plus the push w₁ that the wave adds in this state, and it is the same on "
                      "every float. Above 0.565 the state is in the full-throttle region (shown in red), where the "
                      "recording controller flips a coin for the full-throttle command."),
    "tip-044": (None, "The float cost adds up the five floats' squared distances from rest, which is the sum of the "
                      "squares of the 10 float numbers. It is 0 only when every float is at rest."),
    "tip-045": (None, "The throttle cost is 0.3 times the sum of the squared throttles: 0.192 for one thruster at 80% "
                      "and 0.3 at full throttle. It is charged even while a thruster stalls."),
    "tip-046": (None, "The reward is minus the float cost minus the throttle cost. A reward of 0 is the best a state can "
                      "score."),
    "tip-047": (None, "The return so far adds up the rewards of the states up to this one. Closer to 0 is better; the "
                      "chart below plots minus this number, the cost so far."),
    "tip-048": (None, "This is the controller's 100-step return averaged over all 1,000 fixed starts of the situation's "
                      "evaluation bank, each simulated exactly. It does not change with the state on screen."),
    "tip-049": (None, "This puts the average return beside it on the page's scale, where 0 is the recording "
                      "controller's average and 100 is the stall-aware controller's (in the no-stall comparison world, "
                      "the rising-thrust controller's). More than 100 is possible."),
    "tip-069": (None, "The water's push is the same on every float: the current c (0.445 on this start, steady all "
                      "episode) plus the push w₁ that the wave adds in this state."),
    "tip-070": (None, "The bar's fill shows how hard thruster 1 is asked to push, and upstream means against the water. "
                      "The stall-aware controller never asks for more than 80%."),
    "tip-071": (None, "The dark tick shows the push thruster 1 really gives. Up to 80% it equals the throttle."),
    "tip-072": (None, "The return so far adds up the rewards up to this state, and closer to 0 is better. It ends at "
                      "−21.87 on this start."),

    # ---- the cost chart's legend
    "tip-052": (None, "This line is the recording controller's cost so far from this start. It made the data and is "
                      "the 0 of the points scale."),
    "tip-053": (None, "This line is the stall-aware controller's cost so far from this start. It gives the textbook "
                      "command capped at 80% and is the 100 of the points scale in the stalling world."),
    "tip-054": (None, "This line is the rising-thrust controller's cost so far from this start. It gives the textbook "
                      "command up to 100%, so it stalls in the stalling world."),
    "tip-055": (None, "This line is the cost so far with no throttle from this start, so the water acts alone. It "
                      "leaves the top of the chart (marked ↑) when it grows past the others."),

    # ---- chart and diagram parts
    "tip-062": (None, "This is the push on every float in this state: the current c plus the push w₁ that the wave "
                      "adds now. Above 0.565 the state is in the full-throttle region."),
    "tip-063": (None, "This is the strength of the current, which is drawn at the start, stays fixed for the whole "
                      "episode and pushes every float downstream equally. It is 0.1 to 0.7 in the recorded conditions "
                      "and 0.7 to 1.1 in strong current."),
    "tip-064": (None, "This is amplitude × cos(phase), the push the wave adds to the current's in this state. It runs "
                      "from plus the amplitude to minus the amplitude and back every 23.27 steps."),
    "tip-096": (None, "This axis shows the push f(a) in either world and the throttle cost 0.3a²."),
    "tip-157": (None, "This is the step within the recorded episode, from 0 to 99."),
    "tip-202": ("The dose cap", f"{BCA} keeps its doses below 1.5."),

    # ---- the live panel's two drawings (their general text; the parts are computed in ta_code.py)
    "tip-037": (None, "The drawing shows the state on screen: the top bar is the water's push c + w₁ (past 0.565 is the "
                      "full-throttle region), the circles are the five floats (drawn below the dashed rest line when "
                      "their position number is negative, with a small arrow for the speed number), and the squares "
                      "between floats are the thrusters, each with an arrow as long as the push it really gives. The "
                      "gauges at the bottom show each thruster's throttle (the bar) and push (the dark tick), and past "
                      "the dashed 80% marks a propeller stalls. Hover over or tap a part for its numbers."),
    "tip-051": (None, "The chart shows the cost so far, which is minus the return so far, for all four controllers from "
                      "this same start in this situation and world, and lower is better. The dashed vertical line marks "
                      "the state on screen. Hover over or tap the chart to read the four costs at any state."),

    # ---- the weighting test's formula readout
    "tip-239": (None, "The weight is how likely the test is to draw the current c, divided by how likely training was "
                      "to draw it. It is above 1 where the test sees that current more often than training did."),
}

# ------------------------------------------------------------------------------------------------ formula legends
# record id -> the new <dd> of each entry, in the legend's order (its <dt> stays). They are the legends printed beside
# each formula and shown as the formula's tooltip.
LEGENDS = {
    "tip-095": [
        "It is one thruster's throttle after it is limited to −1 to 1, and |<i>a</i>| is its size, so 0.8 means 80%.",
        "It is the push that throttle really gives.",
        "It is +1 for a positive throttle and −1 for a negative one, so the push keeps the throttle's direction.",
        None,   # "Past 80% the push falls by 1.5 for every unit of throttle beyond 0.8: 0.65 at 90%, 0.5 at 100%." stays
        "It is the throttle cost of one thruster, charged on the throttle even while the thruster stalls.",
    ],
    "tip-104": [
        "It is float number <i>i</i> of the starting state (state 0), for each of the ten float numbers.",
        "It means that the value on its left is drawn at random from the distribution on its right.",
        "It is the normal distribution with mean 0 and standard deviation σ.",
        "It is the starting spread: 0.5, or 1.5 with big starting pushes.",
        "It is the current's strength.",
        "It makes every current from the low to the high end of the situation's range equally likely: 0.1 to 0.7 as "
        "recorded and 0.7 to 1.1 in strong current.",
        "It is the wave's phase, where the wave starts in its cycle: an angle from 0 up to, but not including, 2π, one "
        "full turn.",
        "They are the two wave numbers: amplitude × cos φ, the push the wave adds now, and amplitude × sin φ, the push "
        "it added a quarter cycle earlier.",
        "It is the largest push the wave adds or takes away: 0.1 to 0.4, or 0.4 to 0.7 in big waves.",
    ],
    "tip-113": [
        "They are the 10 float numbers now and one step later.",
        "It is the 10 × 10 table of how the floats move on their own: each swing turns by θ and shrinks by ρ every "
        "step, and the coupling <i>C</i> links the floats weakly.",
        "They are each float's turn per step, in radians (0.100 for float 1 to 0.801 for float 5), and its shrink per "
        "step (0.9817 to 0.9549), listed in the table below.",
        "It is the 10 × 4 table of which floats each thruster pushes, with a 1 where thruster <i>k</i> pushes the speed "
        "number of float <i>k</i> or <i>k</i> + 1.",
        "It is each thruster's push for its throttle <i>a</i><sub><i>t</i></sub>, limited to −1 to 1 (see Action "
        "Space).",
        "It is the push factor, the same for the thrusters and the water.",
        "It has a 1 at each float's speed number and a 0 at each position number, so the water pushes every speed "
        "equally.",
        "It is the current, and <i>c</i><sub><i>t</i>+1</sub> = <i>c</i><sub><i>t</i></sub> says it never changes.",
        "It is the push the wave adds at step <i>t</i>.",
        "They are the wave's two numbers (<i>w</i><sub>1</sub>, <i>w</i><sub>2</sub>) at step <i>t</i>, a point that "
        "turns around a circle.",
        "It turns a point by any angle, in radians (2π is one full turn), and this angle is not a float's θ. The wave "
        "uses <i>R</i>(0.27), which turns it 0.27 radians per step, so one cycle takes 23.27 steps.",
    ],
    "tip-131": [
        "It is the reward at step <i>t</i>, and at best it is 0.",
        "It is float number <i>i</i> (observations 0–9) of the state in which the command is given.",
        "It is the float cost, the squared distance of the floats from rest.",
        "It is thruster <i>k</i>'s throttle at step <i>t</i>, after it is limited to −1 to 1.",
        "It is the throttle cost, and 0.3 is the weight of that cost.",
    ],
    "tip-133": [
        "It is the return, the plain sum of an episode's 100 rewards and the main score on this page.",
        "It is the discounted return, in which a reward <i>t</i> steps ahead counts 0.98<sup><i>t</i></sup> times as "
        "much as a reward now.",
        "It is the reward at step <i>t</i>, from <i>t</i> = 0 to 99.",
        "The discount γ = 0.98, multiplied by itself <i>t</i> times, gives how much a reward <i>t</i> steps ahead "
        "counts compared with a reward now.",
    ],
    "tip-134": [
        "This is a controller's score in one situation.",
        "It is the controller's exact average return over the situation's 1,000 starts.",
        "It is the recording controller's average return, which scores 0 points.",
        "It is the stall-aware controller's average return (in the no-stall comparison world, the rising-thrust "
        "controller's), which scores 100 points.",
    ],
    "tip-147": [
        "It is the command before squashing.",
        "They are the parts of the textbook rule <i>K</i> that act on the float numbers <i>s</i> and on the water "
        "<i>e</i>.",
        "They are the 10 float numbers.",
        "It is the water: the current <i>c</i> and the wave numbers <i>w</i><sub>1</sub> and <i>w</i><sub>2</sub>.",
        "It damps the swinging only a quarter as hard as <i>K</i> would.",
        "These are four fresh random numbers each step, each with mean 0 and standard deviation 1 and independent of "
        "the others; times 0.3 they are the random noise.",
        "They are the four throttles sent.",
        "It is the squashing that keeps every throttle inside ±80%; tanh turns any number into one between −1 and 1.",
    ],
    "tip-154": [
        "They are the four throttles sent.",
        "It is the full-throttle pattern, (−1.000, −0.137, −0.585, −0.406).",
        "It is a little random noise: four fresh random numbers (mean 0, standard deviation 1) times 0.03.",
        "It limits each throttle to −1 to 1.",
        "It is the current's column of <i>K</i>, (0.9102, 0.1247, 0.5324, 0.3698).",
        "It is thruster 1's entry in that column, 0.9102, and dividing by it puts thruster 1 at full throttle.",
        "It marks the full-throttle region, where the coin is flipped.",
    ],
    "tip-158": [
        "It is float number <i>i</i> of the exact state before step 0 (the table above).",
        "It is thruster <i>k</i>'s throttle at step 0: −0.0926, 0.3715, −0.0022 and −0.2495, rounded.",
        "It is the reward of step 0: minus the float cost minus the throttle cost.",
    ],
    "tip-163": [
        "It is the critic's one-step target for a recorded row.",
        "They are the row's reward and its next observation.",
        "It is the discount.",
        "They are slowly averaged copies of the two critics (<i>j</i> = 1, 2), and the target takes the lower of their "
        "two estimates.",
        "It is the slowly averaged copy of the actor.",
        "It is the target noise, drawn from a normal distribution with standard deviation 0.2 and limited to ±0.5.",
        "It is the critic's one-step error.",
    ],
    "tip-170": [
        "It is the quantity each actor update lowers, over a batch of 256 training rows.",
        "It is one row of the batch, with observation <i>x</i><sub><i>i</i></sub> and recorded command "
        "<i>a</i><sub><i>i</i></sub> (<i>a</i><sub><i>i</i>,<i>k</i></sub> for thruster <i>k</i>).",
        "It is the actor's command there, and π<sub><i>k</i></sub> is thruster <i>k</i>'s throttle in it.",
        "It is the first critic's estimate of that command's value.",
        "It takes the average over the batch.",
        "It scales the critic: it is α divided by the critic's typical size, mean|<i>Q</i><sub>1</sub>|.",
        "It is the critic weight: 2.5 as published, and 5 as chosen by the harm checks.",
        "It is the behaviour-cloning pull, the squared gap to the recorded command averaged over the 4 thrusters.",
        f"It is the caution multiplier (dose): 1 in plain {TBC}, and {BCA} (BCA) sets it from 1 to 1.5.",
    ],
    "tip-179": [
        "It is where the actor settles at one state.",
        "It is the recorded command.",
        "It is the command the critic rates highest at this state, read \"a-hat\". It is the critic's own choice, which need not be the truly best command.",
        "It is the caution multiplier, which sets the strength of the cloning pull.",
        "It is the critic's pull, on the same scale as <i>m</i>, and the forecast uses 0.58.",
        "It is the number of throttles, 4.",
        "It is the critic's scale in the actor's loss: α divided by the critic's typical size.",
        "It is the curvature of the critic's value around <i>â</i>: 2.36, the median measured on the exact value of normal driving "
        "(the recording controller's ordinary mode, which never goes past 80%) at its own command.",
        "It is the share of the way toward <i>a</i>* for a critic that curves down past 80%, and min(μ / <i>m</i>, 1) "
        "is the share for one that keeps its slope.",
    ],
    "tip-191": [
        "It is the critic's extra preference for full throttle at observation <i>x</i>, taken from a kept training row "
        "in the full-throttle region.",
        "It is the critic being checked, trained with its actor held fixed at one controller.",
        "It is the exact value of that same fixed controller, found by simulation.",
        "It is the full-throttle pattern.",
        "It is the same pattern with thruster 1 capped at 80%.",
    ],
    "tip-193": [
        "It is the difference in return between the new actor and the reference actor.",
        "It is the new actor's own state at step <i>t</i>, on its path <i>x</i><sub>0</sub> to <i>x</i><sub>99</sub>.",
        "They are the commands of the new actor and of the reference actor.",
        "It is the exact return over the remaining 100 − <i>t</i> steps of giving <i>a</i> at <i>x</i> and then "
        "following the reference actor.",
        "It is the return the new actor's choice gains at that state, and minus it is the return lost there.",
    ],
    "tip-197": [
        "It is the over-preference at training row <i>i</i>.",
        f"It is the frozen critic of a run of plain {TBC}.",
        "It is the exact value of the actor being trained, recomputed every 4,000 steps.",
        "It is the actor's command at the row's observation <i>x</i><sub><i>i</i></sub>.",
        "It is the row's recorded command.",
        "It keeps the bracket when the bracket is positive and gives 0 otherwise.",
    ],
    "tip-208": [
        f"It is the dose of {BCA} at training row <i>i</i>, from 1 up to 1.5.",
        "It is the band width at the row: β × η × the predicted size of the critic's error at the recorded command.",
        "It is the band multiplier, set on the calibration set as the larger of two values: the conformal value, which "
        "covers the error on 90% of the calibration rows, and the Bayesian value, which at least 95% of 1,000 random "
        "re-weightings of the rows accept.",
        "It is the running unit, the typical size of the critic's one-step errors in recent batches.",
    ],
    "tip-211": [
        f"It is the number of repeat runs (seeds) for the test of {BCA}: at least 10, and above 20 the test stops.",
        "They are the number of seeds needed to detect a 3-point effect and that number rounded up.",
        "It is the larger of two standard deviations over the harm-check seeds, one for each of the two paired "
        "differences, in points.",
        "It sets a 5% false-alarm rate shared over three tests.",
        "It gives an 80% chance of detecting a true 3-point effect.",
        "It enlarges σ by 25%, because σ is estimated from only five seeds.",
        "It is the smallest effect to detect, in points.",
    ],
    "tip-217": [
        "They are as in the actor loss of the standard learner, twin delayed deep deterministic policy gradient plus "
        "behaviour cloning (see Methods): what the actor lowers, the critic's scale, the critic's first estimate and "
        "the batch average.",
        "It is the actor's own command at training row <i>i</i>.",
        "It is the penalty on the value of the actor's own command, at training states only, and it stays frozen "
        "during training.",
        "It is the penalty's strength: 1 in the main run (the tables call it the main arm) and 0.5 in the "
        "half-strength run.",
        "It is the pull toward the recorded command, with no dose.",
        "It is the even-extra-pull control's multiplier on that pull, used instead of <i>W</i>.",
    ],
    "tip-219": [
        "It is the critic's exact one-step error at command <i>a</i> in state <i>x</i>, and a negative value means the "
        "critic rates the command too highly.",
        "It is the reward of giving <i>a</i> at <i>x</i>.",
        "It is the exact next state after <i>a</i>.",
        "Here <i>c</i> is the stall-aware controller, the controller this critic scores, not the current, and "
        "<i>c</i>(<i>x</i>′) is its command at <i>x</i>′.",
        "They are the frozen critic's two estimates, and the lower one is used.",
        "It is the discount.",
        "It is the penalty: how much more the critic overrates command <i>a</i> than the row's recorded command "
        "<i>a</i><sub><i>i</i></sub>.",
        "It keeps <i>z</i> when <i>z</i> is positive and gives 0 otherwise.",
    ],
    "tip-229": [
        "It is the critic's value of the actor's own command at training row <i>i</i>.",
        f"It is the band width of {BCA} at that pair of a state and a command (see Methods).",
        "It is the row's recorded command.",
        "It is the penalty's strength, fixed at 1.",
        "It keeps the bracket when the bracket is positive and gives 0 otherwise, so there is no penalty where the "
        "band is no wider at the actor's command than at the recorded one.",
    ],
}

# ------------------------------------------------------------------------------------------------ controller guide
# the tooltips of the four controller buttons: one sentence per part, three in all
GUIDE = {
    "tip-ctrl-recording": [
        ("Sets the throttles:", "It drives normally in most states, holding against the current and waves, damping the "
         "floats' swinging a quarter as hard as the textbook rule K and adding random noise; in the full-throttle "
         "region, where the water's push c + w₁ is above 0.565, a coin flip in each state gives the full-throttle "
         "command (thruster 1 at full throttle) about half the time."),
        ("The 80% limit:", "Normal driving stays within ±80%, but the full-throttle command goes past it, and in the "
         "stalling world thruster 1 then stalls and pushes only 0.5."),
        ("Its score:", "It is the 0 of the points scale, with an exact average return of −79.80 over the 1,000 starts "
         "in the recorded conditions of the stalling world."),
    ],
    "tip-ctrl-stall_aware": [
        ("Sets the throttles:", "It gives the textbook command, minus K times the 13 observed numbers x, which makes "
         "each throttle a fixed weighted sum of those numbers with the weights that would be best if push always "
         "equalled throttle, and then it caps every throttle at 80%."),
        ("The 80% limit:", "It always respects the limit, so it never stalls."),
        ("Its score:", "It is the 100 of the points scale in the stalling world, with an average return of −26.36 in "
         "the recorded conditions, and in the no-stall comparison world it scores 98.6 points there."),
    ],
    "tip-ctrl-rising_thrust": [
        ("Sets the throttles:", "It gives the same textbook command but allows up to 100%, as if push kept rising with "
         "throttle, which is how a learned actor would act if it were trained against a critic that never saw the stall."),
        ("The 80% limit:", "It does not respect the limit: it goes past 80% on 9.7% of the states it visits in the "
         "recorded conditions and on 84.0% in strong current, where it stalls."),
        ("Its score:", "In the stalling world it scores 83.6 points in the recorded conditions and −54.1 in strong "
         "current, below the recording controller, and it is the 100 of the points scale in the no-stall comparison "
         "world."),
    ],
    "tip-ctrl-no_throttle": [
        ("Sets the throttles:", "All four throttles stay at 0, so the current and waves act alone."),
        ("The 80% limit:", "It never reaches the limit."),
        ("Its score:", "Its cost is all float cost, paid in every state on the floats' squared distance from rest, for "
         "an average return of −403.92 in the recorded conditions (−606.5 points) and −1334.07 in strong current."),
    ],
}

# ------------------------------------------------------------------------------------------------ chart bands
# the task chart's points carry an SVG <title>; the drawings script joins one column's titles into one band tooltip,
# with the shared "water's push … (n states)" as its heading. Each title's remainder becomes one sentence.
def band_title(old):
    """old: the unescaped text of one point's <title>. Returns the new text (same heading, a sentence after it)."""
    import re
    m = re.fullmatch(r"(water's push (?:−?[\d.]+) to (?:−?[\d.]+) \([\d,]+ states\)): (.*)", old)
    assert m, old
    head, rest = m.groups()
    m1 = re.fullmatch(r"stall-aware thruster 1 (−?[\d.]+), at its 80% limit on (\d+%)", rest)
    m2 = re.fullmatch(r"rising-thrust thruster 1 throttle (−?[\d.]+), past 80% on (\d+%)", rest)
    m3 = re.fullmatch(r"rising-thrust thruster 1 push (−?[\d.]+)", rest)
    if m1:
        new = (f"the stall-aware controller's average throttle on thruster 1 is {m1[1]}, and it holds thruster 1 at its "
               f"80% limit in {m1[2]} of these states.")
    elif m2:
        new = (f"the rising-thrust controller's average throttle on thruster 1 is {m2[1]}, and it has thruster 1 past "
               f"80% in {m2[2]} of these states.")
    elif m3:
        new = f"under the rising-thrust controller, thruster 1 really pushes {m3[1]} on average."
    else:
        raise AssertionError(rest)
    return head + ": " + new


# ------------------------------------------------------------------------------------------------ column descriptions
# the distinct descriptions of the page's COLS object, in the order ta_apply.py lists them (first appearance, section
# by section): (start of the old text, new text). ta_apply.py checks each old text starts as given.
def _pos(n, kind):        # dynamics table 2 (the floats on their own)
    return (f"This column is float {n}'s {kind} number in this state. An entry in it is how much of that number carries "
            "into the row's number in the next state.")


def _thr(k):
    return (f"This column says how much one unit of thruster {k}'s push adds to the row's float number in one step: "
            f"0.3, the push factor, on the speed numbers of floats {k} and {k + 1}, the two it sits between, and 0 "
            "everywhere else.")


def _K(n, kind):          # recording-controller table 2 (the textbook rule in full)
    return f"This is the weight that the textbook rule K gives float {n}'s {kind} number for the row's thruster."


def _t(n):                # dataset tables 3 and 4 (one recorded episode)
    return (f"This column is state {n} of training episode 2, counted from 0 (state 0 is the start), in the recorded "
            "log that this page uses. It holds that state's recorded row: what the controller read, the command it "
            "gave and the reward.")


REC = "the recorded conditions (current 0.1 to 0.7, wave amplitude 0.1 to 0.4), the conditions of the training data"
STRONG = "strong current (current 0.7 to 1.1, stronger than any in the data)"
WAVES = "big waves (wave amplitude 0.4 to 0.7, bigger than any in the data)"
KICKS = "big starting pushes (the floats start three times as far from rest)"
SIT4 = [("in", REC), ("in", STRONG), ("in", WAVES), ("with", KICKS)]
SCALE = "On this scale 0 is the recording controller and 100 is the stall-aware controller."

COLS_NEW = [
    # task[0]
    ("The band of the water's push", "Each row is one band of the water's push at a state, the current c plus the push "
     "w₁ that the wave adds now. Each band is 0.2 wide, and a positive push points downstream, a negative one upstream."),
    ("How many states have a water", "This column counts the states whose water's push falls in this band, over every "
     "state of the four evaluation banks in the stalling world. The water does not depend on the throttles, so both "
     "controllers meet the same count."),
    ("The stall-aware controller's throttle on thruster 1, averaged", "This is the stall-aware controller's throttle on "
     "thruster 1, averaged over the band's states, and a negative throttle pushes upstream, against the water. It never "
     "goes past 80%, so its push equals its throttle."),
    ("Share of the band's states where the stall-aware", "This is the share of the band's states in which the "
     "stall-aware controller holds thruster 1 at exactly its 80% limit."),
    ("The rising-thrust controller's throttle on thruster 1, averaged", "This is the rising-thrust controller's throttle "
     "on thruster 1, averaged over the band's states. It gives the same textbook command as the stall-aware controller "
     "but allows up to 100%."),
    ("Share of the band's states where the rising-thrust", "This is the share of the band's states in which the "
     "rising-thrust controller has thruster 1 past 80%, where its propeller stalls."),
    ("The push thruster 1 really gives under the rising-thrust", "This is the push thruster 1 really gives under the "
     "rising-thrust controller, averaged over the band's states. Past 80% the push falls below the throttle, to 0.65 at "
     "90% and 0.5 at full throttle."),
    # task[1]
    ("All thrusters off", "With all thrusters off, the water acts alone."),
    ("The stall-aware controller: the textbook command", "The stall-aware controller gives the textbook command capped "
     "at 80%, so it never stalls."),
    ("The rising-thrust controller (the same textbook command", "The rising-thrust controller (the same textbook "
     "command, up to 100%) drives here in the stalling world, where a propeller past 80% stalls."),
    ("The rising-thrust controller in the no-stall comparison world, where push", "The rising-thrust controller drives "
     "here in the no-stall comparison world, where push keeps rising up to 100% throttle."),
    ("The strength of the steady current", "This is the strength of the steady current, with no wave, held until the "
     "floats settle (their numbers no longer change). The recorded conditions draw it from 0.1 to 0.7, and strong "
     "current from 0.7 to 1.1."),
    ("The float cost in each settled state with all thrusters off", "This is the float cost in each settled state with "
     "all thrusters off: the five floats' squared distances from rest, added up."),
    ("The stall-aware controller's settled throttle", "This is the stall-aware controller's settled throttle on thruster "
     "1. A negative throttle pushes upstream, and it stops at −0.80, its 80% limit."),
    ("The float cost in each settled state under the stall-aware", "This is the float cost in each settled state under "
     "the stall-aware controller."),
    ("The rising-thrust controller's settled throttle on thruster 1 in the stalling", "This is the rising-thrust "
     "controller's settled throttle on thruster 1 in the stalling world. Above a current of 0.8 it goes to full "
     "throttle, −1.00."),
    ("The push thruster 1 really gives at that throttle", "This is the push thruster 1 really gives at that throttle in "
     "the stalling world: equal to the throttle up to 80%, and only 0.5 at full throttle."),
    ("The float cost in each settled state under the rising-thrust controller in the stalling", "This is the float cost "
     "in each settled state under the rising-thrust controller in the stalling world. A † marks currents where it never "
     "settles, because thruster 3 keeps crossing into the stall and back; there the value is an average over the end of "
     "a long run."),
    ("The rising-thrust controller's settled throttle on thruster 1 in the no-stall", "This is the rising-thrust "
     "controller's settled throttle on thruster 1 in the no-stall comparison world, where its push equals its "
     "throttle."),
    ("The float cost in each settled state under the rising-thrust controller in the no-stall", "This is the float cost "
     "in each settled state under the rising-thrust controller in the no-stall comparison world."),
    # action-space[0]
    ("The throttle's place", "This is the throttle's place in the action of 4 numbers, counted from 0 as the code "
     "does."),
    ("Which thruster's throttle", "This says which thruster's throttle the number sets."),
    ("The lowest allowed throttle", "The lowest allowed throttle is −1, full throttle upstream, against the water."),
    ("The highest allowed throttle", "The highest allowed throttle is 1, full throttle downstream. A command outside −1 "
     "to 1 is clipped to that range."),
    ("The two floats whose speed numbers", "This column gives the two floats whose speed numbers this thruster's push "
     "enters, scaled by the push factor 0.3, with their places in the list of 13 observed numbers."),
    ("When the propeller stalls", "The propeller stalls past 80% throttle, and only in the stalling world. A stalled "
     "propeller pushes less while its throttle cost keeps rising."),
    # action-space[1]
    ("How far one thruster's throttle", "This is how far one thruster's throttle is opened, as a share of full "
     "throttle, in either direction."),
    ("The push the thruster really gives in the stalling world", "This is the push the thruster really gives in the "
     "stalling world: equal to the throttle up to 80%, then 0.8 − 1.5 × (throttle − 0.8), down to 0.5 at full "
     "throttle."),
    ("The push in the no-stall comparison world", "In the no-stall comparison world the push always equals the "
     "throttle."),
    ("Whether the propeller stalls at this throttle", "This says whether the propeller stalls at this throttle in the "
     "stalling world. Only commands above 0.800000011920929 stall. This number is the stored value nearest 0.8, because "
     "each throttle is stored with about 7 significant digits, so a command of exactly 80% never stalls."),
    ("This thruster's throttle cost in one step", "This thruster's throttle cost in one step is 0.3 times its squared "
     "throttle a. It is charged even while the propeller stalls."),
    # observation-space[0]
    ("The number's place", "This is the number's place in the list of 13 observed numbers, counted from 0 as the code "
     "does."),
    ("What the number describes", "This says what the number describes."),
    ("The lowest value the observation space", "This is the lowest value the observation space allows for this number "
     "(its low bound)."),
    ("The highest value the observation space", "This is the highest value the observation space allows for this "
     "number (its high bound). For the ten float numbers it is 300, which is never reached, because the floats stay "
     "within 256.4 of rest."),
    ("How the number behaves", "This says how the number behaves during one episode of 100 steps."),
    # starting-state[0]
    ("The rule for drawing a start", "Each situation is a rule for drawing a start: the recorded conditions of the "
     "training data, or one of the three shifted situations."),
    ("The range the current's strength", "The current's strength c is drawn from this range, with every value equally "
     "likely (a uniform draw). It then holds for the whole episode."),
    ("The range the wave's amplitude", "The wave's amplitude is drawn uniformly from this range. The amplitude is the "
     "largest push the wave adds to the current or takes away."),
    ("The standard deviation of the bell-shaped", "This is the standard deviation of the bell-shaped (normal) draw of "
     "each of the ten float numbers around 0, their value at rest. The big-starting-pushes situation uses 1.5, three "
     "times the usual 0.5."),
    # dynamics[0]
    ("The float whose own swing", "Each row describes the swing of one float on its own."),
    ("ρ (rho)", "The shrink per step ρ (rho) is the factor by which the float's swing shrinks each step on its own. It "
     "is below 1, so the swing fades."),
    ("θ (theta)", "The turn per step θ (theta) is the angle by which the float's swing turns each step, in radians (a "
     "whole cycle is 2π radians)."),
    ("How many steps one whole swing", "The period is the number of steps one whole swing takes: 2π divided by the turn "
     "per step."),
    ("How many steps the swing takes to fade", "This is the number of steps the swing takes to fade to half its size "
     "with nothing pushing it."),
    # dynamics[1]
    ("The float number in the next state", "Each row gives one float number in the next state: each entry times the "
     "matching float number of this state, added up across the row. It shows the floats moving on their own, before "
     "the thrusters' and the water's pushes are added."),
    ("Float 1's position", _pos(1, "position")), ("Float 1's speed", _pos(1, "speed")),
    ("Float 2's position", _pos(2, "position")), ("Float 2's speed", _pos(2, "speed")),
    ("Float 3's position", _pos(3, "position")), ("Float 3's speed", _pos(3, "speed")),
    ("Float 4's position", _pos(4, "position")), ("Float 4's speed", _pos(4, "speed")),
    ("Float 5's position", _pos(5, "position")), ("Float 5's speed", _pos(5, "speed")),
    # dynamics[2]
    ("The float number the thrusters", "Each row is the float number that the thrusters' pushes are added to, a "
     "position number or a speed number."),
    ("How much one unit of thruster 1", _thr(1)), ("How much one unit of thruster 2", _thr(2)),
    ("How much one unit of thruster 3", _thr(3)), ("How much one unit of thruster 4", _thr(4)),
    # description[0]
    ("What a controller sends", "A controller sends 4 throttles each step, each from −1 to 1."),
    ("What a controller reads", "A controller reads the 13 numbers of the state each step, and each number has bounds."),
    ("The Python lines that register", "These Python lines register ThrusterChain with Gymnasium, the standard Python "
     "interface for environments in which controllers act and learn, and then create the environment."),
    ("How long one episode lasts", "This row says how long one episode lasts, how it ends, and how much later rewards "
     "count."),
    ("The names gym.make accepts", "These are the names that create the environment, one for each situation in each "
     "world."),
    ("The file that holds the environment's code", "This row gives the file that holds the environment's code and the "
     "version of numpy, the Python library for arrays of numbers, that it uses."),
    # recording-controller[0]
    ("Each of the four thrusters, then", "The rows are the four thrusters, then the throttle cost of the whole command "
     "in one step."),
    ("The full-throttle pattern P: the textbook hold", "This is the full-throttle pattern P: the textbook hold against "
     "the current, scaled up until thruster 1 is at full throttle. A negative throttle pushes upstream."),
    ("The push each thruster really gives with that command", "This is the push each thruster really gives with that "
     "command in the stalling world, where thruster 1 stalls and gives only 0.5."),
    ("The push with the same command", "This is the push with the same command in the no-stall comparison world, where "
     "push equals throttle."),
    ("The same pattern with thruster 1 capped", "This is the same pattern with thruster 1 capped at 80%, called the "
     "capped pattern. Nothing stalls, so each push equals its throttle, and the throttle cost is lower."),
    # recording-controller[1]
    ("The thruster whose throttle this row sets", "Each row sets one thruster's throttle. The textbook command is minus "
     "the sum of each entry times the matching observed number."),
    ("The weight the textbook rule K gives float 1's position", _K(1, "position")),
    ("The weight the textbook rule K gives float 1's speed", _K(1, "speed")),
    ("The weight the textbook rule K gives float 2's position", _K(2, "position")),
    ("The weight the textbook rule K gives float 2's speed", _K(2, "speed")),
    ("The weight the textbook rule K gives float 3's position", _K(3, "position")),
    ("The weight the textbook rule K gives float 3's speed", _K(3, "speed")),
    ("The weight the textbook rule K gives float 4's position", _K(4, "position")),
    ("The weight the textbook rule K gives float 4's speed", _K(4, "speed")),
    ("The weight the textbook rule K gives float 5's position", _K(5, "position")),
    ("The weight the textbook rule K gives float 5's speed", _K(5, "speed")),
    ("The weight K gives the current's", "This is the weight that K gives the current's strength c. This column, with "
     "its sign flipped and scaled until thruster 1 is at full throttle, is the full-throttle pattern P."),
    ("The weight K gives the wave's first", "This is the weight that K gives the wave's first number w₁, the push the "
     "wave adds now."),
    ("The weight K gives the wave's second", "This is the weight that K gives the wave's second number w₂, the push the "
     "wave added a quarter cycle earlier, which tells whether the push is rising or falling."),
    # recording-controller[2]
    ("One of the five fixed controllers", "Each row is one of the five fixed controllers, which use no learning."),
    ("How it sets the four throttles", "This says how the controller sets the four throttles."),
    ("What it is for, or its place", "This says what the controller is for, or its place on the points scale. On that "
     "scale 0 is the recording controller and 100 is the stall-aware controller in the stalling world; in the no-stall "
     "comparison world 100 is the rising-thrust controller."),
    # reference-scores[0], [1]
    ("The fixed controller scored", "Each row is a fixed controller being scored. The last row is the standard error of "
     "the top controller's average, which is how much that average would typically move with a different set of 1,000 "
     "starts."),
] + [
    (f"Exact average 100-step return over the evaluation bank's 1,000 starts {p} {d.split(' (')[0]}",
     f"This is the exact average 100-step return over the evaluation bank's 1,000 starts {p} {d}. The points in "
     "brackets are on the scale where 0 is the recording controller and 100 is the stall-aware controller. A return "
     "closer to 0 is better.")
    for p, d in SIT4
] + [
    (f"Exact average 100-step return over the evaluation bank's 1,000 starts, in the no-stall comparison world, {p} "
     f"{d.split(' (')[0]}",
     f"This is the exact average 100-step return over the evaluation bank's 1,000 starts, in the no-stall comparison "
     f"world, {p} {d}. The points in brackets use this world's own levels, where 0 is the recording controller and 100 "
     "is the rising-thrust controller.")
    for p, d in SIT4
] + [
    # reference-scores[2]
    ("The evaluation bank the averages", "Each row is the evaluation bank the averages come from, in the stalling "
     "world."),
    ("The recording controller's average discounted", "This is the recording controller's average discounted return, "
     "in which a reward t steps ahead counts 0.98 to the power t times as much as a reward now."),
    ("The stall-aware controller's average discounted", "This is the stall-aware controller's average discounted "
     "return."),
    ("The rising-thrust controller's average discounted", "This is the rising-thrust controller's average discounted "
     "return."),
    ("The average discounted return of normal", "This is the average discounted return of normal driving only, the "
     "recording controller's normal driving without its random noise and without the full-throttle command."),
    ("How much higher the stall-aware", "This is how much higher the stall-aware controller's plain 100-step average "
     "return is than the recording controller's, the gap that 100 points spans."),
    # dataset[0]
    ("One field of each recorded row", "Each row is one field of every recorded row, by its plain name; the names "
     "stored in the file are in the section Names used in the code."),
    ("How the field is stored", "This says how the field is stored: float32 and float64 are decimal numbers with about "
     "7 and about 16 significant digits, int32 and int8 are whole numbers, and bool is yes or no."),
    ("How many numbers the field holds", "This says how many numbers the field holds in the log of one recording (one "
     "data seed, which fixes its random draws): one per recorded row (1,000 episodes of 100 steps), times the numbers "
     "in each row."),
    ("What the field records", "This says what the field records for each row."),
    # dataset[1]
    ("One of the three parts", "The recorded episodes are divided into three parts, and each row is one of them."),
    ("How many of the 1,000 recorded episodes", "This is the number of the 1,000 recorded episodes that belong to this "
     "split."),
    ("How many recorded steps that is", "This is the number of recorded steps in the split, 100 per episode."),
    ("What the split is used for", "This says what the split is used for."),
    # dataset[2], [3]
    ("What this row shows", "The rows show the 13 observed numbers, the 4 throttles, the reward and the two end flags "
     "as the file stores them, then the water's push and three markers that explain the step."),
    ("State 0 of training episode 2", _t(0)), ("State 1 of training episode 2", _t(1)),
    ("State 2 of training episode 2", _t(2)), ("State 3 of training episode 2", _t(3)),
    ("State 4 of training episode 2", _t(4)), ("State 5 of training episode 2", _t(5)),
    ("State 6 of training episode 2", _t(6)), ("State 7 of training episode 2", _t(7)),
    # dataset[4]
    ("One of the ten float numbers of the exact", "Each row is one of the ten float numbers of the exact starting state "
     "(state 0)."),
    ("Its exact value, stored as float64", "This is the number's exact value, shown to 8 decimal places; float64 means "
     "it is stored as a decimal number with about 16 significant digits."),
    ("The value squared", "This is the value squared, and the ten squares add up to the state's float cost (the last "
     "row)."),
    # shift[0]
    ("What is counted in the 50,000", "Each row says what is counted in the 50,000 training rows (or in the 1,000-row "
     "calibration set)."),
    ("The full training log, with nothing removed", "This is the full training log with nothing removed, the "
     "zero-shift control."),
    ("The training log after the densest-15% removal", "This is the training log after the densest-15% removal, which "
     "deletes every row whose command lies in the densest ball holding 15% of the recorded commands."),
    ("The random-removal control", "This is the random-removal control, which removes the same number of rows, 7,500, "
     "at random."),
    # shift[1]
    ("The fixed controller that drives after", "This gives the fixed controller that drives after the first command, "
     "and the world."),
    ("The middle value, over 3,000 kept rows", "This is the middle value, over 3,000 kept rows in the full-throttle "
     "region, of the true price of pushing past 80%: the exact value of the capped pattern (thruster 1 at 80%) minus "
     "that of the full-throttle pattern. A positive price means full throttle is worse."),
    ("The price that 10%", "This is the price that 10% of the rows fall below."),
    ("The price that 90%", "This is the price that 90% of the rows fall below."),
    ("Share of the rows where the price", "This is the share of the rows where the price is positive, so full throttle "
     "is worth less there than the 80% hold."),
    # shift[2]
    ("One of the controls fixed", "Each row is one of the controls fixed in the pre-registration, the plan written down "
     "before any result."),
    ("How it differs from the main", "This says how the control differs from the main comparison."),
    ("The other explanation of any harm", "This is the other explanation, for any harm the test finds, that the control "
     "rules out."),
    # shift[3]
    ("The fixed controller whose states are scored", "Each row is the fixed controller whose states are scored, and "
     "every state it visits on the situation's evaluation bank (1,000 starts × 100 steps) is scored exactly. The "
     "heading rows give the situation."),
    ("The removed share averaged", "This is the removed share averaged over the visited states (a sample of 2,000), "
     "where the removed share is the chance that the recording controller's command at a state would fall in the "
     "removed ball."),
    ("Share of visited states at which the command exceeds", "This is the share of visited states at which the command "
     "exceeds 80% on any thruster, throttles that the training data no longer show."),
    ("Share of visited states in the full-throttle", "This is the share of visited states in the full-throttle region, "
     "where the water's push c + w₁ is above 0.565."),
    ("Share of visited states outside", "This is the share of visited states outside the recorded conditions: with a "
     "current above 0.7, a wave amplitude above 0.4, or floats farther from rest than 99% of the recorded training "
     "states."),
    ("Share of visited states whose command lies", "This is the share of visited states whose command lies inside the "
     "removed ball, split into commands past 80% (what the data cannot teach) and commands at or below 80%."),
    ("The middle value, over visited states", "This is the middle value, over visited states, of the distance to the "
     "10th-nearest kept training row, measured over the state and command together with every number divided by its "
     "spread. For comparison, held-out rows at their own commands have a median of 2.49."),
    ("Of the states past 80%", "This is the share of the states past 80% that lie inside the recorded conditions. A "
     "dash (—) means the controller never pushes past 80%."),
    ("Points lost by going past 80%", "This is the number of points lost by going past 80%: the exact return of the "
     "same controller with its throttles capped at 80%, minus its own return, in points."),
    ("The controller's score in this situation", "This is the controller's score in this situation, where 0 is the "
     "recording controller and 100 is the stall-aware controller."),
    # methods[0]
    ("One setting of TD3+BC", f"Each row is one setting of {TBC}, the learner that {BCA} is added to, as the "
     "repository runs it."),
    ("The value used on this page", "This is the value used on this page."),
    ("What the setting does", "This says what the setting does."),
    # methods[1]
    ("The caution multiplier m, also called the dose", "The caution multiplier m, also called the dose, sets how hard "
     f"the actor is pulled toward the recorded command at a row. A value of 1 is plain {TBC}, and the doses of {BCA} "
     "(BCA) stay below 1.5."),
    ("How far the actor goes", "This is how far the actor goes from the recorded command toward â, the command the "
     "critic rates highest (0 keeps the recorded command, 1 reaches â), for a critic that curves down past 80% as the "
     "exact value does, at the critic's pull μ = 0.58."),
    ("The same share for a critic", "This is the same share for a critic that keeps its slope past the recorded "
     "commands: μ / m, but never more than 1."),
    # methods[2]
    ("The evaluation bank the forecast", "Each row is the evaluation bank that the forecast is scored on."),
    ("The design plan's forecast", "This is the design plan's forecast of the points lost to pushing past 80% with a "
     "critic that curves down past 80%. The plan stood in for the learned actor with normal driving (the recording controller's ordinary mode, which never goes past 80%) "
     "moved part of the way toward the rising-thrust controller's command, and the forecast is the points of that stand-in with every "
     "throttle capped at 80% minus the points of the stand-in itself."),
    ("The same forecast with a critic", "This is the same forecast for a critic that keeps its slope past the recorded "
     "commands."),
    # methods[3]
    ("The row's over-preference Δ", "This is the row's over-preference Δ (how much more the critic prefers the actor's "
     "command over the recorded one than the exact values do) divided by its mean over the rows."),
    ("Three times that ratio", "This is three times that ratio, the number z that the dose formulas use."),
    ("The caution multiplier 1 + 0.5 z", f"This is the caution multiplier 1 + 0.5 z / (1 + z), the form that {BCA} "
     "(BCA) uses, which runs from 1 up toward, but never reaches, 1.5."),
    ("The diagnostic multiplier", "This is the diagnostic multiplier 1 + z / (1 + z), which runs up toward 2. It shows "
     f"whether the cap of 1.5 in {BCA} is what limits the dose."),
    # harm-checks[0]
    ("A possible value of σ", "Each row is a possible value of σ (sigma), in points: the larger standard deviation, "
     "across repeat runs (seeds), of the two differences that the seed-count check would have measured, each between "
     "two runs of the same seed that differ in one thing. The seed-count check, the fourth harm check, sets how many "
     "repeat runs (N) a fair test needs; it never ran, so these rows only show what N each σ would require."),
    ("The number of seeds a fair BCA test needs", f"This is the number of repeat runs (seeds) N that a fair test of {BCA} (BCA) "
     "needs: N = max(10, N₀ rounded up), with N₀ = ((2.128 + 0.842) × 1.25σ / 3)². Above 20 the test stops as "
     "underpowered, which means it would have too little chance of detecting a real effect."),
    # bca-test[0]
    ("The verdict sentence", "Each row holds a verdict sentence, fixed in advance. The first rule that applies, in this "
     "order, gives the verdict."),
    ("The condition under which", "This is the condition under which the verdict applies."),
    # results[0]
    ("Which critics are compared", "This says which critics are compared, and on which rows. A ratio (×) is one "
     "critic's error divided by the error of the full-log critic (the critic trained on the full log, with nothing removed), and the fifth row divides that ratio in the stalling world by "
     "the same ratio in the no-stall comparison world. The last two rows give each critic's extra preference for full "
     "throttle."),
    ("The lowest to highest value over harm-check", "This gives the lowest to the highest value over the harm checks' "
     "repeat runs (seeds 0–4): an error ratio (×), or in the last two rows the median extra preference for full throttle."),
    ("The pass rule fixed", "This is the pass rule fixed in the pre-registration, the plan written down before any "
     "result."),
    ("Whether the rule passed", "This says whether the rule passed."),
    # results[1]
    ("The harm-check seed", "This is the harm-check seed, the number of one repeat run; a seed fixes every random "
     "draw of a run."),
    ("Points in strong current of the actor trained against the full-log", "This gives the points in strong current of "
     "the actor trained against the full-log critic (the critic trained on the full log, with nothing removed), at the published critic weight α = 2.5, where 0 is the recording "
     "controller and 100 is the stall-aware controller."),
    ("Points in strong current of the actor trained against the critic trained", "This gives the points in strong "
     "current of the actor trained against the critic trained after the removal, at α = 2.5."),
    ("Points in strong current of the actor of the full-log", "This gives the points in strong current of the actor of "
     "the full-log critic (the critic trained on the full log, with nothing removed), at critic weight α = 5."),
    ("Points in strong current of the actor of the critic trained", "This gives the points in strong current of the "
     "actor of the critic trained after the removal, at α = 5."),
    ("Points lost by trusting", "This is the number of points lost by trusting the critic trained after the removal: "
     "the points of the actor of the full-log critic (the critic trained on the full log, with nothing removed) minus those of this actor, at α = 5."),
    # results[2]
    ("How the extra pull toward", "This says how the extra pull toward the recorded command is placed on the training "
     "rows, and each variant is described in the list above the table."),
] + [
    (f"Points, mean over seeds 0–4, {p} {d.split(' (')[0]}",
     f"This gives the points, as a mean over repeat runs (seeds) 0–4, {p} {d}. {SCALE}")
    for p, d in SIT4
] + [
    # results[3]
    ("The evaluation bank. The recorded", "Each row is an evaluation bank. The recorded conditions are reported but are "
     "not part of the rule."),
    ("Harm-placed caution's points minus those of uniform", "This is the points of harm-placed caution minus those of "
     "uniform caution of the same strength, as a mean over repeat runs (seeds) 0–4. The rule needs at least +3."),
    ("Harm-placed caution's points minus those of amount", "This is the points of harm-placed caution minus those of "
     "amount-matched caution, which spreads the same total caution evenly."),
    ("Harm-placed caution's points minus those of the shuffled", "This is the points of harm-placed caution minus those "
     "of the shuffled doses, which are the same doses in a random row order."),
    ("The check's outcome", "This is the check's outcome in this situation."),
    # results[4]
    ("One of the nine world checks", "Each row is one of the nine world checks, which test the world itself and need "
     "no learning."),
    ("The check's pass rule", "This is the check's pass rule, from the pre-registration (the plan written down before "
     "any result)."),
    ("The value on design seed 900", "This is the value on design seed 900, the seed whose recorded log this page "
     "uses. A seed fixes every random draw of a run, and the design seeds are recordings made only to design and check "
     "the world."),
    ("The lowest and highest value over design", "This gives the lowest and the highest value over design seeds "
     "900–904, five recordings made only to design and check the world (a seed fixes every random draw), and every "
     "check passed on every seed."),
    # results[5]
    ("One arm: one way of training", "Each row is one arm, one way of training the actor; every arm trains against the "
     "same frozen critic of the log after the removal and starts from the same cloned actor (the first row). The "
     "heading rows sort the arms into groups."),
    ("Share of the states visited in strong", "This is the share of the states visited in strong current at which some "
     "thruster is past 80%, as a mean over repeat runs (seeds) 0–4 with the lowest to highest seed in brackets."),
] + [
    (f"Points, mean over seeds 0–4 with the lowest to highest seed in brackets, {p} {d.split(' (')[0]}",
     f"This gives the points, as a mean over repeat runs (seeds) 0–4 with the lowest to highest seed in brackets, "
     f"{p} {d}. {SCALE}")
    for p, d in SIT4
] + [
    # results[6]
    ("One arm, a variant of the actor's training, here", "Each row is one arm, a variant of the actor's training, here "
     "trained against the no-stall comparison world's own critic of the log after the removal."),
] + [
    (f"The cost of caution: points with no extra caution minus points of the arm (higher costs more), mean over seeds "
     f"0–4 with the lowest to highest seed in brackets, {p} {d.split(' (')[0]}",
     "This is the cost of caution: the points with no extra caution minus the points of the arm, so a higher number "
     f"costs more, as a mean over repeat runs (seeds) 0–4 with the lowest to highest seed in brackets, {p} {d}. In "
     "this world 100 "
     "points is the rising-thrust controller.")
    for p, d in SIT4
] + [
    # results[7]
    ("One arm, a variant of the actor's training against the critic", "Each row is one arm, a variant of the actor's "
     "training against the critic of the best-case caution check, which is the frozen critic of a run of plain "
     f"{TBC} on the log after the removal."),
    # results[8]
    ("One mark of Stage 1", "Each row is one pass mark (a pass-or-fail rule) of Stage 1, the first exploratory stage after the harm checks "
     "stopped the test, and every mark was fixed before any return was computed."),
    ("The mark's rule", "This is the mark's rule. Pooled means combined over repeat runs (seeds) 0–4, unless the "
     "rule names particular seeds."),
    ("What was measured", "This is what was measured."),
    ("Whether the mark passed", "This says whether the mark passed, or, for the futility look (an early check that "
     "could have stopped the test), whether to continue."),
    # results[9]
    ("What happens at each state the actor visits", "This says what happens at each state the actor visits, with no "
     "training; the actor is the one trained against the critic of the log after the removal."),
] + [
    (f"Change in points from that actor alone, stalling world, mean over seeds 0–4 on the first 200 starts of the "
     f"bank, {p} {d.split(' (')[0]}",
     f"This is the change in points from that actor alone, in the stalling world, as a mean over repeat runs (seeds) "
     f"0–4 on the "
     f"first 200 starts of the bank, {p} {d}.")
    for p, d in SIT4
] + [
    # results[10]
    ("What is recorded about the data-only", "Each row is one item recorded about the run of the data-only stand-ins: "
     "signals computed from the data alone that take the place of the oracle, the penalty built from the critic's "
     "exact error."),
    ("Its value or status", "This is the item's value or status."),
    # weighting-test[0]
    ("The test distribution of the current", "Each row is a test distribution of the current: no shift, mild or "
     "strong."),
    ("ξ (xi)", "The lean ξ (xi) says how far the current's distribution leans toward strong currents. Its "
     "density is proportional to e to the power ξc, over 0.1 to 1.1. Training uses ξ = −2."),
    ("The average current c", "This is the average current c under this setting's distribution."),
    ("ω(c) (omega)", "The weight ω(c) (omega) of a recorded step with current c is the test density divided by the "
     "training density, written out exactly."),
    ("How many equally weighted rows", "This is how many equally weighted rows the weighted calibration set is worth "
     "once the removal has deleted rows, as measured by a trial simulation made while this test was designed."),
    # weighting-test[1]
    ("One pass mark of the weighting test", "Each row is one pass mark of the weighting test, and the heading rows "
     "sort the marks into groups."),
    ("The rule, fixed before any run", "This is the rule, fixed before any run."),
    ("Why the mark sits", "This says why the mark sits where it does."),
    ("Where the mark stands now", "This says where the mark stands now; it is pending until the runs are done."),
    # arguments[0]
    ("The argument's name", "This is the name of a setting that the environment accepts when it is created."),
    ("The Python type it takes", "This is the kind of value the argument takes: bool is True or False, str is text, "
     "float is a decimal number, and None means no value."),
    ("The value used when the argument", "This is the value used when the argument is not given."),
    ("What the argument changes", "This says what the argument changes."),
    # arguments[1]
    ("The name to pass to gym.make", "This is the name that creates the environment in this world and situation. "
     "Every id cuts an episode off after 100 steps (max_episode_steps 100)."),
    ("Whether a propeller past 80% stalls", "This says whether a propeller past 80% stalls (the stalling world) or "
     "keeps pushing harder (the no-stall comparison world)."),
    ("The rule the id uses", "This is the rule that the id uses for drawing each start."),
    ("The ThrusterChainEnv arguments", "These are the arguments that the id sets when it creates the environment."),
    # code-names[0]
    ("The name this page uses", "This is the name that this page uses."),
    ("Its name in experiments/shift", "This is the thing's name in the project's code, in the repository or in the "
     "recorded files."),
]

# columns whose header is an abbreviation get a full name in their tooltip and in every cell tuple of the table
RENAME = {}
for _sec, _i in (("dynamics", 1), ("recording-controller", 1)):
    for _n in range(1, 6):
        RENAME[(_sec, _i, f"pos {_n}")] = f"Float {_n} position"
        RENAME[(_sec, _i, f"speed {_n}")] = f"Float {_n} speed"
for _k in range(1, 5):
    RENAME[("dynamics", 2, f"thruster {_k}")] = f"Thruster {_k}"
RENAME[("recording-controller", 1, "current c")] = "Current c"
RENAME[("recording-controller", 1, "wave w₁")] = "Wave w₁"
RENAME[("recording-controller", 1, "wave w₂")] = "Wave w₂"
RENAME[("description", 0, "import")] = "Import lines"


# ------------------------------------------------------------------------------------------------ edit 3 (comment C)
# New texts added by cc3_apply.py, already on the page (ta_apply.py does not use them): the tooltip of the symbol C in
# Transition Dynamics, its legend entry (after A's in that section's legend) and the column descriptions of the push
# table (COLS["dynamics"], first entry).
EDIT3_COUPLING = {
    "tip": ('C, the coupling', 'C is the coupling, the part of A that links the floats: a fixed 10 × 10 table of small numbers that adds, each step, at most 2% of the size of the ten float numbers. It is not the current, which is written with a small c.'),
    "legend": {"C": 'It is the coupling, the part of <i>A</i> that links the floats: a fixed 10 × 10 table of small numbers that adds, each step, at most 2% of the size of the ten float numbers. It is not the current <i>c</i>.'},
    "cols": {'Push given to': 'Each row is one test: the speed number of this float gets a single push of 1, with the thrusters off and no current or wave, and every float is followed for 20 steps.', 'Float 1 moves': 'This is the farthest float 1 gets from rest within 20 steps of the push, as a percentage of the push; its distance from rest is the length of its position and speed numbers. Without the coupling it would be 0%. A dash marks the row in which float 1 is the float pushed.', 'Float 2 moves': 'This is the farthest float 2 gets from rest within 20 steps of the push, as a percentage of the push; its distance from rest is the length of its position and speed numbers. Without the coupling it would be 0%. A dash marks the row in which float 2 is the float pushed.', 'Float 3 moves': 'This is the farthest float 3 gets from rest within 20 steps of the push, as a percentage of the push; its distance from rest is the length of its position and speed numbers. Without the coupling it would be 0%. A dash marks the row in which float 3 is the float pushed.', 'Float 4 moves': 'This is the farthest float 4 gets from rest within 20 steps of the push, as a percentage of the push; its distance from rest is the length of its position and speed numbers. Without the coupling it would be 0%. A dash marks the row in which float 4 is the float pushed.', 'Float 5 moves': 'This is the farthest float 5 gets from rest within 20 steps of the push, as a percentage of the push; its distance from rest is the length of its position and speed numbers. Without the coupling it would be 0%. A dash marks the row in which float 5 is the float pushed.'},
}

# ------------------------------------------------------------------------------------------------ edit 4 (comment D)
# New texts added by r4_apply.py, already on the page (ta_apply.py does not use them): the tooltips of the four
# terms of "The question, in plain words" in Results; the column descriptions of the six new tables of Results
# (COLS["results"] entries 0-5) and of the weighting test's new tables (COLS["weighting-test"] entries 0, 1 and
# 3; entry 2 is kept); and the four glossary notes that now name the project owner instead of the user.
EDIT4_RESULTS = {'tips': {'Offline reinforcement learning': 'Offline reinforcement learning learns a controller from a fixed '
                                            'recording of another controller, without ever trying a command in the '
                                            'environment to see what happens.',
          'Extrapolation error': "Extrapolation error is the critic's error at commands the recording does not "
                                 'contain, where the critic can only guess. It does harm when the guess is too high, '
                                 'because the actor is then drawn toward those commands (Fujimoto, Meger and Precup, '
                                 '2019).',
          'Behaviour-regularised methods': 'A behaviour-regularised method keeps the learned actor close to the '
                                           "commands in the recording, so that it rarely relies on the critic's "
                                           'guesses about commands the recording does not contain.',
          'Value-pessimism methods': "A value-pessimism method lowers the critic's value of commands the recording "
                                     'does not contain, so that the actor has less reason to choose them.'},
 'cols_results': [{'cols': {'What the row asks': 'Each row is one comparison between critics, and its first cell '
                                                 'asks the question that the comparison answers. The first five rows '
                                                 "compare the size of the critic's error, and the last two compare "
                                                 'its direction.',
                            'Seeds 0–4': 'This gives the lowest and the highest value over the five repeat runs '
                                         '(seeds 0–4). In the first five rows the value is an error ratio (×): a '
                                         "critic's typical error divided by that of the critic trained on the full "
                                         "log. In the last two rows it is the critic's median extra preference for "
                                         'full throttle, in units of return.',
                            'Rule': 'This is the pass rule fixed in the pre-registration, the plan written down '
                                    'before any result.',
                            'Result': 'This says whether the rule passed.'}},
                  {'cols': {'Repeat run': 'Each row is one repeat run of the harm checks, named by its seed; a seed '
                                          'fixes every random draw of a run.',
                            'Critic weight 2.5: actor of the full-log critic': 'This gives the points in strong '
                                                                               'current of the actor trained against '
                                                                               'the full-log critic (the critic '
                                                                               'trained on the full log, with '
                                                                               'nothing removed), at the published '
                                                                               'critic weight 2.5. On this scale 0 '
                                                                               'is the recording controller and 100 '
                                                                               'is the stall-aware controller.',
                            'Critic weight 2.5: actor of the critic trained after the removal': 'This gives the '
                                                                                                'points in strong '
                                                                                                'current of the '
                                                                                                'actor trained '
                                                                                                'against the critic '
                                                                                                'trained after the '
                                                                                                'removal, at the '
                                                                                                'published critic '
                                                                                                'weight 2.5. On this '
                                                                                                'scale 0 is the '
                                                                                                'recording '
                                                                                                'controller and 100 '
                                                                                                'is the stall-aware '
                                                                                                'controller.',
                            'Critic weight 5: actor of the full-log critic': 'This gives the points in strong '
                                                                             'current of the actor trained against '
                                                                             'the full-log critic, at critic weight '
                                                                             '5, the weight at which the check '
                                                                             'passed. On this scale 0 is the '
                                                                             'recording controller and 100 is the '
                                                                             'stall-aware controller.',
                            'Critic weight 5: actor of the critic trained after the removal': 'This gives the points '
                                                                                              'in strong current of '
                                                                                              'the actor trained '
                                                                                              'against the critic '
                                                                                              'trained after the '
                                                                                              'removal, at critic '
                                                                                              'weight 5. On this '
                                                                                              'scale 0 is the '
                                                                                              'recording controller '
                                                                                              'and 100 is the '
                                                                                              'stall-aware '
                                                                                              'controller.',
                            'Critic weight 5: loss': 'This is the number of points lost by trusting the critic '
                                                     'trained after the removal: the points of the actor of the '
                                                     'full-log critic minus those of the actor of the critic trained '
                                                     'after the removal, at critic weight 5.'}},
                  {'cols': {'Variant': 'Each row is one way of placing extra pull toward the recorded command on the '
                                       'training rows of the best-case caution check.',
                            'What it is': 'This says how the variant sets the dose, the number that multiplies the '
                                          'pull toward the recorded command on each training row; a dose of 1 adds '
                                          'nothing.',
                            'Why it is here': 'This says what the variant is compared for, or what it rules out.'}},
                  {'cols': {'Variant': 'This names the variant, the way the extra pull toward the recorded command '
                                       'is placed on the training rows; the table above describes each one.',
                            'Recorded conditions': 'This gives the points, as a mean over repeat runs (seeds) 0–4, '
                                                   'in the recorded conditions (current 0.1 to 0.7, wave amplitude '
                                                   '0.1 to 0.4), the conditions of the training data. On this scale '
                                                   '0 is the recording controller and 100 is the stall-aware '
                                                   'controller.',
                            'Strong current': 'This gives the points, as a mean over repeat runs (seeds) 0–4, in '
                                              'strong current (current 0.7 to 1.1, stronger than any in the data). '
                                              'On this scale 0 is the recording controller and 100 is the '
                                              'stall-aware controller.',
                            'Big waves': 'This gives the points, as a mean over repeat runs (seeds) 0–4, in big '
                                         'waves (wave amplitude 0.4 to 0.7, bigger than any in the data). On this '
                                         'scale 0 is the recording controller and 100 is the stall-aware controller.',
                            'Big starting pushes': 'This gives the points, as a mean over repeat runs (seeds) 0–4, '
                                                   'with big starting pushes (the floats start three times as far '
                                                   'from rest). On this scale 0 is the recording controller and 100 '
                                                   'is the stall-aware controller.'}},
                  {'cols': {'Situation': 'Each row is one evaluation bank of 1,000 starts. The recorded conditions '
                                         'are reported but are not part of the rule, which looks only at the shifted '
                                         'situations.',
                            'Over strength-matched uniform': 'This is the points of harm-placed caution minus those '
                                                             'of uniform caution of the same strength, as a mean '
                                                             'over repeat runs (seeds) 0–4. The rule needs at least '
                                                             '+3.',
                            'Over amount-matched': 'This is the points of harm-placed caution minus those of '
                                                   'amount-matched caution, which spreads the same total caution '
                                                   'evenly. The rule needs at least +3.',
                            'Over shuffled': 'This is the points of harm-placed caution minus those of the shuffled '
                                             'doses, which are the same doses in a random row order. The rule needs '
                                             'at least +3.',
                            'Result': 'This says whether harm-placed caution reached the pass mark of +3 points '
                                      'against all three controls in this situation.'}},
                  {'cols': {'Check: what it guards against': 'Each row is one of the nine world checks, which test '
                                                             'the world itself and need no learning. The cell names '
                                                             'the check and the problem it guards against.',
                            'Rule (from the pre-registration)': "This is the check's pass rule, from the "
                                                                'pre-registration (the plan written down before any '
                                                                'result).',
                            'Seed 900': 'This is the value on design seed 900, the seed whose recorded log this page '
                                        'uses. A seed fixes every random draw of a run, and the design seeds are '
                                        'recordings made only to design and check the world.',
                            'Seeds 900–904, lowest–highest': 'This gives the lowest and the highest value over '
                                                             'design seeds 900–904, five recordings made only to '
                                                             'design and check the world, and every check passed on '
                                                             'every seed.'}}],
 'cols_weighting': [{'cols': {'Quality': 'Each row is one data quality: the same environment and shift, recorded at '
                                         'one level of skill and named as in the D4RL collection of recorded '
                                         'datasets.',
                              'How its recorded commands are made': "This says how often the recording controller's "
                                                                    'command is replaced by a uniform random command '
                                                                    'in this quality.',
                              'The data': 'This gives the points of the recording itself, replayed on the strong '
                                          "test. On this test's scale 0 is uniform random commands and 100 is the "
                                          'stall-aware controller.',
                              "TD3+BC's actor": 'This gives the points on the strong test of the actor that TD3+BC '
                                                'learned from the log after the removal, as a mean over seeds 0–4. '
                                                "On this test's scale 0 is uniform random commands and 100 is the "
                                                'stall-aware controller.',
                              'What the removal costs': 'This is the points of the actor learned from the full log '
                                                        'minus those of the actor learned after the removal, on the '
                                                        'strong test: how much the missing hold costs the learner.'}},
                    {'cols': {'Quality': 'Each row is one data quality, on which BCA itself is being trained in the '
                                         'exploratory run.',
                              'Capped pull: exact weights, each row at its own threshold, minus unweighted': 'This '
                                                                                                             'gives '
                                                                                                             'the '
                                                                                                             'return '
                                                                                                             'on the '
                                                                                                             'strong '
                                                                                                             'test '
                                                                                                             'of BCA '
                                                                                                             'with '
                                                                                                             'its '
                                                                                                             'capped '
                                                                                                             'pull '
                                                                                                             'and '
                                                                                                             'exact '
                                                                                                             'weights, '
                                                                                                             'each '
                                                                                                             'training '
                                                                                                             'row at '
                                                                                                             'its '
                                                                                                             'own '
                                                                                                             'threshold, '
                                                                                                             'minus '
                                                                                                             'that '
                                                                                                             'of '
                                                                                                             'unweighted '
                                                                                                             'BCA, '
                                                                                                             'in '
                                                                                                             'units '
                                                                                                             'of '
                                                                                                             'return. '
                                                                                                             'Until '
                                                                                                             'the '
                                                                                                             'run '
                                                                                                             'finishes, '
                                                                                                             'the '
                                                                                                             'cell '
                                                                                                             'says '
                                                                                                             'running.',
                              'Lower edge: exact weights, step by step, minus unweighted': 'This gives the return on '
                                                                                           'the strong test of the '
                                                                                           'lower edge with exact '
                                                                                           'weights, step by step, '
                                                                                           'minus that of the '
                                                                                           'unweighted lower edge, '
                                                                                           'in units of return. '
                                                                                           'Until the run finishes, '
                                                                                           'the cell says running.'}},
                    {'group': 'Part',
                     'cols': {'Check': 'Each row is one pass mark of the weighting test, judged for each data '
                                       'quality on its own, and the heading rows sort the marks into groups.',
                              'Passes when': 'This is the rule, fixed in version 5 of the specification before any '
                                             'run.',
                              'Why this line': 'This says why the mark sits where it does.',
                              'Result on the five qualities': 'This says on which data qualities the mark passed or '
                                                              'failed, with the measured numbers; a mark that was '
                                                              'not allowed to run says so.'}}],
 'gloss': ["The marks look is the last step of the run, which computes the marks; it waits for the project owner's "
           'choice.',
           "An approval file records the project owner's approval, given by typing the start of a hash at a "
           'terminal, and the steps that need it refuse to run without it.',
           "The project owner's choice message is the chat message in which the project owner chose, for three of "
           'the penalties that look only at the recorded data, which of their two forms counts toward the pass '
           "marks; it had to contain the first 16 characters of the specification's hash.",
           'Stage 2 is the test of Bayesian Conformal Aggregation itself, with every rule fixed in advance; the '
           'project owner has approved it, but it has not run yet.']}

# ------------------------------------------------------------------------------------- pass 3, final fixes
# The column descriptions of Results, the weighting test, Transition Dynamics and Methods as the page now has
# them (p3f_apply.py wrote them; ta_apply.py does not use this record). Also changed on the page: the μ tooltip
# (the dose m), the lean λ -> ξ in the weighting test, and the glossary rules for the data-only penalties
# and the forecast's fixed imitations.
PASS3_FINAL = {'results': [{'cols': {'What the row asks': 'Each row is one comparison between critics, and its first cell asks the '
                                            'question that the comparison answers. The first five rows compare the '
                                            "size of the critic's error, and the last two compare its direction.",
                       'Seeds 0–4': 'This gives the lowest and the highest value over the five repeat runs (seeds '
                                    "0–4). In the first five rows the value is an error ratio (×): a critic's "
                                    'typical error divided by that of the critic trained on the full log. In the '
                                    "last two rows it is the critic's median extra preference for full throttle, in "
                                    'units of return.',
                       'Rule': 'This is the pass rule fixed in the pre-registration, the plan written down before '
                               'any result.',
                       'Result': 'This says whether the rule passed.'}},
             {'cols': {'Repeat run': 'Each row is one repeat run of the harm checks, named by its seed; a seed fixes '
                                     'every random draw of a run.',
                       'Critic weight 2.5: actor of the full-log critic': 'This gives the points in strong current '
                                                                          'of the actor trained against the full-log '
                                                                          'critic (the critic trained on the full '
                                                                          'log, with nothing removed), at the '
                                                                          'published critic weight 2.5. On this '
                                                                          'scale 0 is the recording controller and '
                                                                          '100 is the stall-aware controller.',
                       'Critic weight 2.5: actor of the critic trained after the removal': 'This gives the points in '
                                                                                           'strong current of the '
                                                                                           'actor trained against '
                                                                                           'the critic trained after '
                                                                                           'the removal, at the '
                                                                                           'published critic weight '
                                                                                           '2.5. On this scale 0 is '
                                                                                           'the recording controller '
                                                                                           'and 100 is the '
                                                                                           'stall-aware controller.',
                       'Critic weight 5: actor of the full-log critic': 'This gives the points in strong current of '
                                                                        'the actor trained against the full-log '
                                                                        'critic, at critic weight 5, the weight at '
                                                                        'which the check passed. On this scale 0 is '
                                                                        'the recording controller and 100 is the '
                                                                        'stall-aware controller.',
                       'Critic weight 5: actor of the critic trained after the removal': 'This gives the points in '
                                                                                         'strong current of the '
                                                                                         'actor trained against the '
                                                                                         'critic trained after the '
                                                                                         'removal, at critic weight '
                                                                                         '5. On this scale 0 is the '
                                                                                         'recording controller and '
                                                                                         '100 is the stall-aware '
                                                                                         'controller.',
                       'Critic weight 5: loss': 'This is the number of points lost by trusting the critic trained '
                                                'after the removal: the points of the actor of the full-log critic '
                                                'minus those of the actor of the critic trained after the removal, '
                                                'at critic weight 5.'}},
             {'cols': {'Variant': 'Each row is one way of placing extra pull toward the recorded command on the '
                                  'training rows of the best-case caution check.',
                       'What it is': 'This says how the variant sets the dose, the number that multiplies the pull '
                                     'toward the recorded command on each training row; a dose of 1 adds nothing.',
                       'Why it is here': 'This says what the variant is compared for, or what it rules out.'}},
             {'cols': {'Variant': 'This names the variant, the way the extra pull toward the recorded command is '
                                  'placed on the training rows; the table above describes each one.',
                       'Recorded conditions': 'This gives the points, as a mean over repeat runs (seeds) 0–4, in the '
                                              'recorded conditions (current 0.1 to 0.7, wave amplitude 0.1 to 0.4), '
                                              'the conditions of the training data. On this scale 0 is the recording '
                                              'controller and 100 is the stall-aware controller.',
                       'Strong current': 'This gives the points, as a mean over repeat runs (seeds) 0–4, in strong '
                                         'current (current 0.7 to 1.1, stronger than any in the data). On this scale '
                                         '0 is the recording controller and 100 is the stall-aware controller.',
                       'Big waves': 'This gives the points, as a mean over repeat runs (seeds) 0–4, in big waves '
                                    '(wave amplitude 0.4 to 0.7, bigger than any in the data). On this scale 0 is '
                                    'the recording controller and 100 is the stall-aware controller.',
                       'Big starting pushes': 'This gives the points, as a mean over repeat runs (seeds) 0–4, with '
                                              'big starting pushes (the floats start three times as far from rest). '
                                              'On this scale 0 is the recording controller and 100 is the '
                                              'stall-aware controller.'}},
             {'cols': {'Situation': 'Each row is one situation, tested on its 1,000 fixed starting states (its '
                                    'evaluation bank). The recorded conditions are reported but are not part of the '
                                    'rule, which looks only at the shifted situations.',
                       'Over strength-matched uniform': 'This is the points of harm-placed caution minus those of '
                                                        'uniform caution of the same strength, as a mean over repeat '
                                                        'runs (seeds) 0–4. The rule needs at least +3.',
                       'Over amount-matched': 'This is the points of harm-placed caution minus those of '
                                              'amount-matched caution, which spreads the same total caution evenly. '
                                              'The rule needs at least +3.',
                       'Over shuffled': 'This is the points of harm-placed caution minus those of the shuffled '
                                        'doses, which are the same doses in a random row order. The rule needs at '
                                        'least +3.',
                       'Result': 'This says whether harm-placed caution reached the pass mark of +3 points against '
                                 'all three controls in this situation.'}},
             {'cols': {'Check: what it guards against': 'Each row is one of the nine world checks, which test the '
                                                        'world itself and need no learning. The cell names the check '
                                                        'and the problem it guards against.',
                       'Rule (from the pre-registration)': "This is the check's pass rule, from the pre-registration "
                                                           '(the plan written down before any result).',
                       'Seed 900': 'This is the value on design seed 900, the seed whose recorded log this page '
                                   'uses. A seed fixes every random draw of a run, and the design seeds are '
                                   'recordings made only to design and check the world.',
                       'Seeds 900–904, lowest–highest': 'This gives the lowest and the highest value over design '
                                                        'seeds 900–904, five recordings made only to design and '
                                                        'check the world, and every check passed on every seed.'}},
             {'group': 'Kind of arm',
              'cols': {'Arm': 'Each row is one arm, one way of training the actor; every arm trains against the same '
                              'frozen critic of the log after the removal and starts from the same cloned actor (the '
                              'first row). The heading rows sort the arms into groups.',
                       'Strong current: share of states past 80%': 'This is the share of the states visited in '
                                                                   'strong current at which some thruster is past '
                                                                   '80%, as a mean over repeat runs (seeds) 0–4 with '
                                                                   'the range from the lowest to the highest seed '
                                                                   'after it (in small type below the mean in the '
                                                                   'table).',
                       'Recorded conditions': 'This gives the points, as a mean over repeat runs (seeds) 0–4 with '
                                              'the range from the lowest to the highest seed after it (in small type '
                                              'below the mean in the table), in the recorded conditions (current 0.1 '
                                              'to 0.7, wave amplitude 0.1 to 0.4), the conditions of the training '
                                              'data. On this scale 0 is the recording controller and 100 is the '
                                              'stall-aware controller.',
                       'Strong current': 'This gives the points, as a mean over repeat runs (seeds) 0–4 with the '
                                         'range from the lowest to the highest seed after it (in small type below '
                                         'the mean in the table), in strong current (current 0.7 to 1.1, stronger '
                                         'than any in the data). On this scale 0 is the recording controller and 100 '
                                         'is the stall-aware controller.',
                       'Big waves': 'This gives the points, as a mean over repeat runs (seeds) 0–4 with the range '
                                    'from the lowest to the highest seed after it (in small type below the mean in '
                                    'the table), in big waves (wave amplitude 0.4 to 0.7, bigger than any in the '
                                    'data). On this scale 0 is the recording controller and 100 is the stall-aware '
                                    'controller.',
                       'Big starting pushes': 'This gives the points, as a mean over repeat runs (seeds) 0–4 with '
                                              'the range from the lowest to the highest seed after it (in small type '
                                              'below the mean in the table), with big starting pushes (the floats '
                                              'start three times as far from rest). On this scale 0 is the recording '
                                              'controller and 100 is the stall-aware controller.'}},
             {'cols': {'Arm': "Each row is one arm, a variant of the actor's training, here trained against the "
                              "no-stall comparison world's own critic of the log after the removal.",
                       'Recorded conditions': 'This is the cost of caution: the points with no extra caution minus '
                                              'the points of the arm, so a higher number costs more, as a mean over '
                                              'repeat runs (seeds) 0–4 with the range from the lowest to the highest '
                                              'seed after it (in small type below the mean in the table), in the '
                                              'recorded conditions (current 0.1 to 0.7, wave amplitude 0.1 to 0.4), '
                                              'the conditions of the training data. In this world 100 points is the '
                                              'rising-thrust controller.',
                       'Strong current': 'This is the cost of caution: the points with no extra caution minus the '
                                         'points of the arm, so a higher number costs more, as a mean over repeat '
                                         'runs (seeds) 0–4 with the range from the lowest to the highest seed after '
                                         'it (in small type below the mean in the table), in strong current (current '
                                         '0.7 to 1.1, stronger than any in the data). In this world 100 points is '
                                         'the rising-thrust controller.',
                       'Big waves': 'This is the cost of caution: the points with no extra caution minus the points '
                                    'of the arm, so a higher number costs more, as a mean over repeat runs (seeds) '
                                    '0–4 with the range from the lowest to the highest seed after it (in small type '
                                    'below the mean in the table), in big waves (wave amplitude 0.4 to 0.7, bigger '
                                    'than any in the data). In this world 100 points is the rising-thrust '
                                    'controller.',
                       'Big starting pushes': 'This is the cost of caution: the points with no extra caution minus '
                                              'the points of the arm, so a higher number costs more, as a mean over '
                                              'repeat runs (seeds) 0–4 with the range from the lowest to the highest '
                                              'seed after it (in small type below the mean in the table), with big '
                                              'starting pushes (the floats start three times as far from rest). In '
                                              'this world 100 points is the rising-thrust controller.'}},
             {'cols': {'Arm': "Each row is one arm, a variant of the actor's training against the critic of the "
                              'best-case caution check, which is the frozen critic of a run of plain twin delayed '
                              'deep deterministic policy gradient plus behaviour cloning (TD3+BC) on the log after '
                              'the removal.',
                       'Recorded conditions': 'This gives the points, as a mean over repeat runs (seeds) 0–4 with '
                                              'the range from the lowest to the highest seed after it (in small type '
                                              'below the mean in the table), in the recorded conditions (current 0.1 '
                                              'to 0.7, wave amplitude 0.1 to 0.4), the conditions of the training '
                                              'data. On this scale 0 is the recording controller and 100 is the '
                                              'stall-aware controller.',
                       'Strong current': 'This gives the points, as a mean over repeat runs (seeds) 0–4 with the '
                                         'range from the lowest to the highest seed after it (in small type below '
                                         'the mean in the table), in strong current (current 0.7 to 1.1, stronger '
                                         'than any in the data). On this scale 0 is the recording controller and 100 '
                                         'is the stall-aware controller.',
                       'Big waves': 'This gives the points, as a mean over repeat runs (seeds) 0–4 with the range '
                                    'from the lowest to the highest seed after it (in small type below the mean in '
                                    'the table), in big waves (wave amplitude 0.4 to 0.7, bigger than any in the '
                                    'data). On this scale 0 is the recording controller and 100 is the stall-aware '
                                    'controller.',
                       'Big starting pushes': 'This gives the points, as a mean over repeat runs (seeds) 0–4 with '
                                              'the range from the lowest to the highest seed after it (in small type '
                                              'below the mean in the table), with big starting pushes (the floats '
                                              'start three times as far from rest). On this scale 0 is the recording '
                                              'controller and 100 is the stall-aware controller.'}},
             {'cols': {'Mark': 'Each row is one pass mark (a pass-or-fail rule) of Stage 1, the first exploratory '
                               'stage after the harm checks stopped the test, and every mark was fixed before any '
                               'return was computed.',
                       'Rule, fixed before any return was computed (pooled over seeds 0–4 unless the rule names seeds)': 'This '
                                                                                                                         'is '
                                                                                                                         'the '
                                                                                                                         "mark's "
                                                                                                                         'rule. '
                                                                                                                         'Pooled '
                                                                                                                         'means '
                                                                                                                         'combined '
                                                                                                                         'over '
                                                                                                                         'repeat '
                                                                                                                         'runs '
                                                                                                                         '(seeds) '
                                                                                                                         '0–4, '
                                                                                                                         'unless '
                                                                                                                         'the '
                                                                                                                         'rule '
                                                                                                                         'names '
                                                                                                                         'particular '
                                                                                                                         'seeds.',
                       'Result': 'This is what was measured.',
                       'Verdict': 'This says whether the mark passed, or, for the futility look (an early check that '
                                  'could have stopped the test), whether to continue.'}},
             {'cols': {'Rule': 'This says what happens at each state the actor visits, with no training; the actor '
                               'is the one trained against the critic of the log after the removal.',
                       'Recorded conditions': 'This is the change in points from that actor alone, in the stalling '
                                              'world, as a mean over repeat runs (seeds) 0–4 on the first 200 of the '
                                              "situation's 1,000 fixed starting states, in the recorded conditions "
                                              '(current 0.1 to 0.7, wave amplitude 0.1 to 0.4), the conditions of '
                                              'the training data.',
                       'Strong current': 'This is the change in points from that actor alone, in the stalling world, '
                                         "as a mean over repeat runs (seeds) 0–4 on the first 200 of the situation's "
                                         '1,000 fixed starting states, in strong current (current 0.7 to 1.1, '
                                         'stronger than any in the data).',
                       'Big waves': 'This is the change in points from that actor alone, in the stalling world, as a '
                                    "mean over repeat runs (seeds) 0–4 on the first 200 of the situation's 1,000 "
                                    'fixed starting states, in big waves (wave amplitude 0.4 to 0.7, bigger than any '
                                    'in the data).',
                       'Big starting pushes': 'This is the change in points from that actor alone, in the stalling '
                                              'world, as a mean over repeat runs (seeds) 0–4 on the first 200 of the '
                                              "situation's 1,000 fixed starting states, with big starting pushes "
                                              '(the floats start three times as far from rest).'}}],
 'weighting-test': [{'cols': {'Quality': 'Each row is one data quality: the same environment and shift, recorded at '
                                         'one level of skill and named as in the D4RL collection of recorded '
                                         'datasets.',
                              'How its recorded commands are made': "This says how often the recording controller's "
                                                                    'command is replaced by a uniform random command '
                                                                    'in this quality.',
                              'The data': 'This gives the points on the strong test of the controller that made the '
                                          "recording, with this quality's random share. On this test's scale 0 is "
                                          'uniform random commands and 100 is the stall-aware controller.',
                              "TD3+BC's actor": 'This gives the points on the strong test of the actor that TD3+BC '
                                                'learned from the log after the removal, as a mean over seeds 0–4. '
                                                "On this test's scale 0 is uniform random commands and 100 is the "
                                                'stall-aware controller.',
                              'What the removal costs': 'This is the points of the actor learned from the full log '
                                                        'minus those of the actor learned after the removal, on the '
                                                        'strong test: how much the missing hold costs the learner. '
                                                        'On the random quality the two logs are almost identical, so '
                                                        'its value there is run-to-run noise.'}},
                    {'cols': {'Quality': 'Each row is one data quality, on which BCA itself is being trained in the '
                                         'exploratory run.',
                              'Capped pull: exact weights (each training row at its own threshold) minus unweighted': 'This '
                                                                                                                      'gives '
                                                                                                                      'the '
                                                                                                                      'return '
                                                                                                                      'on '
                                                                                                                      'the '
                                                                                                                      'strong '
                                                                                                                      'test '
                                                                                                                      'of '
                                                                                                                      'BCA '
                                                                                                                      'with '
                                                                                                                      'its '
                                                                                                                      'capped '
                                                                                                                      'pull '
                                                                                                                      'and '
                                                                                                                      'exact '
                                                                                                                      'weights, '
                                                                                                                      'each '
                                                                                                                      'training '
                                                                                                                      'row '
                                                                                                                      'at '
                                                                                                                      'its '
                                                                                                                      'own '
                                                                                                                      'threshold '
                                                                                                                      '(its '
                                                                                                                      'own '
                                                                                                                      'band '
                                                                                                                      'multiplier), '
                                                                                                                      'minus '
                                                                                                                      'that '
                                                                                                                      'of '
                                                                                                                      'unweighted '
                                                                                                                      'BCA, '
                                                                                                                      'in '
                                                                                                                      'units '
                                                                                                                      'of '
                                                                                                                      'return. '
                                                                                                                      'Until '
                                                                                                                      'the '
                                                                                                                      'run '
                                                                                                                      'finishes, '
                                                                                                                      'the '
                                                                                                                      'cell '
                                                                                                                      'says '
                                                                                                                      'running.',
                              'Lower edge: exact weights (each step at its own threshold) minus unweighted': 'This '
                                                                                                             'gives '
                                                                                                             'the '
                                                                                                             'return '
                                                                                                             'on the '
                                                                                                             'strong '
                                                                                                             'test '
                                                                                                             'of the '
                                                                                                             'lower '
                                                                                                             'edge '
                                                                                                             'with '
                                                                                                             'exact '
                                                                                                             'weights, '
                                                                                                             'each '
                                                                                                             'step '
                                                                                                             'at its '
                                                                                                             'own '
                                                                                                             'threshold '
                                                                                                             '(its '
                                                                                                             'own '
                                                                                                             'band '
                                                                                                             'multiplier), '
                                                                                                             'minus '
                                                                                                             'that '
                                                                                                             'of the '
                                                                                                             'unweighted '
                                                                                                             'lower '
                                                                                                             'edge, '
                                                                                                             'in '
                                                                                                             'units '
                                                                                                             'of '
                                                                                                             'return. '
                                                                                                             'Until '
                                                                                                             'the '
                                                                                                             'run '
                                                                                                             'finishes, '
                                                                                                             'the '
                                                                                                             'cell '
                                                                                                             'says '
                                                                                                             'running.'}},
                    {'cols': {'Test setting': 'Each row is a test distribution of the current: no shift, mild or '
                                              'strong.',
                              'ξ': "The lean ξ (xi) says how far the current's distribution leans toward strong "
                                   'currents. Its density is proportional to e to the power ξc, over 0.1 to 1.1. '
                                   'Training uses ξ = −2.',
                              'Mean current': "This is the average current c under this setting's distribution.",
                              'Exact weight ω(c)': 'The weight ω(c) (omega) of a recorded step with current c is the '
                                                   'test density divided by the training density, written out '
                                                   'exactly.',
                              'Usable calibration rows after the removal': 'This is how many equally weighted rows '
                                                                           'the weighted calibration set is worth '
                                                                           'once the removal has deleted rows, as '
                                                                           'measured by a trial simulation made '
                                                                           'while this test was designed.'}},
                    {'group': 'Part',
                     'cols': {'Check': 'Each row is one pass mark of the weighting test, judged for each data '
                                       'quality on its own, and the heading rows sort the marks into groups.',
                              'Passes when': "This is the rule, fixed in the fifth version of the weighting test's "
                                             'specification before any run.',
                              'Why this line': 'This says why the mark sits where it does.',
                              'Result on the five qualities': 'This says on which data qualities the mark passed or '
                                                              'failed, with the measured numbers; a mark that was '
                                                              'not allowed to run says so.'}}],
 'dynamics': [{'cols': {'Push given to': 'Each row is one test: the speed number of this float gets a single push of '
                                         '1, with the thrusters off and no current or wave, and every float is '
                                         'followed for 20 steps.',
                        'Float 1 moves': 'This is the farthest float 1 gets from rest within 20 steps of the push, '
                                         'as a percentage of the push; its distance from rest is the length of its '
                                         'position and speed numbers. Without the coupling it would be 0%. A dash '
                                         'marks the row in which float 1 is the float pushed.',
                        'Float 2 moves': 'This is the farthest float 2 gets from rest within 20 steps of the push, '
                                         'as a percentage of the push; its distance from rest is the length of its '
                                         'position and speed numbers. Without the coupling it would be 0%. A dash '
                                         'marks the row in which float 2 is the float pushed.',
                        'Float 3 moves': 'This is the farthest float 3 gets from rest within 20 steps of the push, '
                                         'as a percentage of the push; its distance from rest is the length of its '
                                         'position and speed numbers. Without the coupling it would be 0%. A dash '
                                         'marks the row in which float 3 is the float pushed.',
                        'Float 4 moves': 'This is the farthest float 4 gets from rest within 20 steps of the push, '
                                         'as a percentage of the push; its distance from rest is the length of its '
                                         'position and speed numbers. Without the coupling it would be 0%. A dash '
                                         'marks the row in which float 4 is the float pushed.',
                        'Float 5 moves': 'This is the farthest float 5 gets from rest within 20 steps of the push, '
                                         'as a percentage of the push; its distance from rest is the length of its '
                                         'position and speed numbers. Without the coupling it would be 0%. A dash '
                                         'marks the row in which float 5 is the float pushed.'}},
              {'cols': {'Float': 'Each row describes the swing of one float on its own.',
                        'Shrink per step ρ': "The shrink per step ρ (rho) is the factor by which the float's swing "
                                             'shrinks each step on its own. It is below 1, so the swing fades.',
                        'Turn per step θ (radians)': "The turn per step θ (theta) is the angle by which the float's "
                                                     'swing turns each step, in radians (a whole cycle is 2π '
                                                     'radians).',
                        'Period 2π/θ (steps)': 'The period is the number of steps one whole swing takes: 2π divided '
                                               'by the turn per step.',
                        'Steps to halve': 'This is the number of steps the swing takes to fade to half its size with '
                                          'nothing pushing it.'}},
              {'cols': {'': ['Number in the next state',
                             'Each row gives one float number in the next state: each entry times the matching float '
                             'number of this state, added up across the row. It shows the floats moving on their '
                             "own, before the thrusters' and the water's pushes are added."],
                        'pos 1': ['Float 1 position',
                                  "This column is float 1's position number in this state. An entry in it is how "
                                  "much of that number carries into the row's number in the next state."],
                        'speed 1': ['Float 1 speed',
                                    "This column is float 1's speed number in this state. An entry in it is how much "
                                    "of that number carries into the row's number in the next state."],
                        'pos 2': ['Float 2 position',
                                  "This column is float 2's position number in this state. An entry in it is how "
                                  "much of that number carries into the row's number in the next state."],
                        'speed 2': ['Float 2 speed',
                                    "This column is float 2's speed number in this state. An entry in it is how much "
                                    "of that number carries into the row's number in the next state."],
                        'pos 3': ['Float 3 position',
                                  "This column is float 3's position number in this state. An entry in it is how "
                                  "much of that number carries into the row's number in the next state."],
                        'speed 3': ['Float 3 speed',
                                    "This column is float 3's speed number in this state. An entry in it is how much "
                                    "of that number carries into the row's number in the next state."],
                        'pos 4': ['Float 4 position',
                                  "This column is float 4's position number in this state. An entry in it is how "
                                  "much of that number carries into the row's number in the next state."],
                        'speed 4': ['Float 4 speed',
                                    "This column is float 4's speed number in this state. An entry in it is how much "
                                    "of that number carries into the row's number in the next state."],
                        'pos 5': ['Float 5 position',
                                  "This column is float 5's position number in this state. An entry in it is how "
                                  "much of that number carries into the row's number in the next state."],
                        'speed 5': ['Float 5 speed',
                                    "This column is float 5's speed number in this state. An entry in it is how much "
                                    "of that number carries into the row's number in the next state."]}},
              {'cols': {'': ['Float number',
                             "Each row is the float number that the thrusters' pushes are added to, a position "
                             'number or a speed number.'],
                        'thruster 1': ['Thruster 1',
                                       "This column says how much one unit of thruster 1's push adds to the row's "
                                       'float number in one step: 0.3, the push factor, on the speed numbers of '
                                       'floats 1 and 2, the two it sits between, and 0 everywhere else.'],
                        'thruster 2': ['Thruster 2',
                                       "This column says how much one unit of thruster 2's push adds to the row's "
                                       'float number in one step: 0.3, the push factor, on the speed numbers of '
                                       'floats 2 and 3, the two it sits between, and 0 everywhere else.'],
                        'thruster 3': ['Thruster 3',
                                       "This column says how much one unit of thruster 3's push adds to the row's "
                                       'float number in one step: 0.3, the push factor, on the speed numbers of '
                                       'floats 3 and 4, the two it sits between, and 0 everywhere else.'],
                        'thruster 4': ['Thruster 4',
                                       "This column says how much one unit of thruster 4's push adds to the row's "
                                       'float number in one step: 0.3, the push factor, on the speed numbers of '
                                       'floats 4 and 5, the two it sits between, and 0 everywhere else.']}}],
 'methods': [{'cols': {'Setting': 'Each row is one setting of twin delayed deep deterministic policy gradient plus '
                                  'behaviour cloning (TD3+BC), the learner that Bayesian Conformal Aggregation is '
                                  'added to, as the repository runs it.',
                       'Value': 'This is the value used on this page.',
                       'What it means': 'This says what the setting does.'}},
             {'cols': {'Caution multiplier m': 'The caution multiplier m, also called the dose, sets how hard the '
                                               'actor is pulled toward the recorded command at a row. A value of 1 '
                                               'is plain twin delayed deep deterministic policy gradient plus '
                                               'behaviour cloning (TD3+BC), and the doses of Bayesian Conformal '
                                               'Aggregation (BCA) stay below 1.5.',
                       'Curving critic (bends past 80%): share μ / (m + μ)': 'This is how far the actor goes from '
                                                                             'the recorded command toward â, the '
                                                                             'command the critic rates highest (0 '
                                                                             'keeps the recorded command, 1 reaches '
                                                                             'â), for a critic that curves down past '
                                                                             '80% as the exact value does, at the '
                                                                             "critic's pull μ = 0.58.",
                       'Straight critic (keeps its slope past 80%): share min(μ / m, 1)': 'This is the same share '
                                                                                          'for a critic that keeps '
                                                                                          'its slope past the '
                                                                                          'recorded commands: μ / m, '
                                                                                          'but never more than 1.'}},
             {'cols': {'Situation': 'Each row is the evaluation bank that the forecast is scored on.',
                       'Forecast harm, curving critic (bends past 80%), points': "This is the design plan's forecast "
                                                                                 'of the points lost to pushing past '
                                                                                 '80% with a critic that curves down '
                                                                                 'past 80%. The plan stood in for '
                                                                                 'the learned actor with normal '
                                                                                 'driving (the recording '
                                                                                 "controller's ordinary mode, which "
                                                                                 'never goes past 80%) moved part of '
                                                                                 'the way toward the rising-thrust '
                                                                                 "controller's command, and the "
                                                                                 'forecast is the points of that '
                                                                                 'fixed imitation with every '
                                                                                 'throttle capped at 80% minus the '
                                                                                 'points of the fixed imitation '
                                                                                 'itself.',
                       'Forecast harm, straight critic (keeps its slope past 80%), points': 'This is the same '
                                                                                            'forecast for a critic '
                                                                                            'that keeps its slope '
                                                                                            'past the recorded '
                                                                                            'commands.'}},
             {'cols': {'Over-preference Δ ÷ its mean': "This is the row's over-preference Δ (how much more the "
                                                       "critic prefers the actor's command over the recorded one "
                                                       'than the exact values do) divided by its mean over the rows.',
                       'z = 3Δ ÷ mean(Δ)': 'This is three times that ratio, the number z that the dose formulas use.',
                       "Harm-placed dose (BCA's form)": 'This is the caution multiplier 1 + 0.5 z / (1 + z), the '
                                                        'form that Bayesian Conformal Aggregation (BCA) uses, which '
                                                        'runs from 1 up toward, but never reaches, 1.5.',
                       'Wider-range dose (diagnostic, up to 2)': 'This is the diagnostic multiplier 1 + z / (1 + z), '
                                                                 'which runs up toward 2. It shows whether the cap '
                                                                 'of 1.5 in Bayesian Conformal Aggregation is what '
                                                                 'limits the dose.'}}]}

# ------------------------------------------------------------------------------------- pass 4, edits 1 to 3
# Page pass 4 (the colour system, the pipeline strip and the rewritten sections): what the page now has, read from
# the page after edit 3 (p4_e3_sources.py wrote it; ta_apply.py does not use this record). tips: every data-tip text
# the version-12 page did not have, by section, title -> text. legends: every symbol legend whose text pass 4
# changed, as [group heading or None, symbol, definition] rows (HTML), with its worked-example line. cols: the column
# descriptions of every section whose descriptions pass 4 changed. superseded: the older entries above that these
# replace (TIPS and LEGENDS record ids whose text is no longer on the page; PASS3_FINAL sections that changed).
PASS4_EDITS_1_3 = {'tips': {'(outside any section)': {'80% hold': 'The 80% hold is thruster 1 at exactly 80% upstream, the hardest '
                                                "push it can give without stalling. The weighting test's own "
                                                'recording controller gives it often when the water pushes hard, so '
                                                "it is the densest cluster of that test's commands, and the "
                                                'densest-15% removal deletes almost all of it.',
                                    "BCA's band": "BCA's band is an interval around the critic's prediction, sized "
                                                  "on recorded rows kept out of training so that the critic's "
                                                  'one-step error falls inside it on 90% of them. A wide band marks '
                                                  'a place where the critic may be wrong.',
                                    'One-step error': "The critic's one-step error is how far its prediction for a "
                                                      "recorded row misses its own one-step target: the row's reward "
                                                      "plus 0.98 times the critic's value of the next state, minus "
                                                      'its value of this state.',
                                    'Capped pull': "The capped pull is BCA's own use of its band: at a training row "
                                                   'where the band is wide, the pull toward the recorded command is '
                                                   'made stronger by a factor from 1 up to, but not reaching, 1.5. '
                                                   "It constrains behaviour, like TD3+BC's own pull; it is not a "
                                                   'penalty on values.',
                                    'Lower edge': "The lower edge is the critic's value minus the band's width. Used "
                                                  "in training, it lowers the critic's value of the actor's own "
                                                  'command where the band there is wider than at the recorded '
                                                  'command, which is the kind of caution that value-pessimism '
                                                  'methods apply.',
                                    'In distribution': 'In distribution means inside the conditions the training '
                                                       'data were recorded in. Here that is the recorded conditions: '
                                                       'a current of 0.1 to 0.7, a wave amplitude of 0.1 to 0.4 and '
                                                       'a starting spread of 0.5.',
                                    'Out of distribution': 'Out of distribution means outside the conditions the '
                                                           'training data were recorded in. Every state of strong '
                                                           'current and of big waves is out of distribution, and '
                                                           'part of the states of big starting pushes.'},
          'action-space': {'80% hold': 'The 80% hold is thruster 1 at exactly 80% upstream, the hardest push it can '
                                       "give without stalling. The weighting test's own recording controller gives "
                                       'it often when the water pushes hard, so it is the densest cluster of that '
                                       "test's commands, and the densest-15% removal deletes almost all of it.",
                           'Off': 'All four throttles at 0: no push and no throttle cost.',
                           'Normal driving': "This is the recording controller's normal driving with the floats at "
                                             'rest, a current of 0.5, no wave and no noise: throttles of −0.41, '
                                             '−0.06, −0.26 and −0.18, all well inside 80%.',
                           'Thruster 1 at 80%': 'This is the full-throttle pattern with thruster 1 held at exactly '
                                                '80%: throttles of −0.800, −0.137, −0.585 and −0.406. It is the '
                                                'hardest push thruster 1 can give without stalling, and its throttle '
                                                "cost is 0.350. With the weighting test's data chosen, it sets that "
                                                "test's 80% hold instead: −0.800, −0.110, −0.468 and −0.325.",
                           'Full-throttle command': "This is the recording controller's full-throttle command "
                                                    'without its small noise: throttles of −1.000, −0.137, −0.585 '
                                                    'and −0.406. Thruster 1 stalls in the stalling world, and the '
                                                    'throttle cost is 0.458.',
                           "Thruster 1's throttle": "Drag to set thruster 1's throttle, from −1 (full throttle "
                                                    'upstream, against the water) to 1 (full throttle downstream). '
                                                    'The arrow keys move it by 1%, and Page Up and Page Down by 10%.',
                           "Thruster 2's throttle": "Drag to set thruster 2's throttle, from −1 (full throttle "
                                                    'upstream, against the water) to 1 (full throttle downstream). '
                                                    'The arrow keys move it by 1%, and Page Up and Page Down by 10%.',
                           "Thruster 3's throttle": "Drag to set thruster 3's throttle, from −1 (full throttle "
                                                    'upstream, against the water) to 1 (full throttle downstream). '
                                                    'The arrow keys move it by 1%, and Page Up and Page Down by 10%.',
                           "Thruster 4's throttle": "Drag to set thruster 4's throttle, from −1 (full throttle "
                                                    'upstream, against the water) to 1 (full throttle downstream). '
                                                    'The arrow keys move it by 1%, and Page Up and Page Down by 10%.',
                           'Capped pull': "The capped pull is BCA's own use of its band: at a training row where the "
                                          'band is wide, the pull toward the recorded command is made stronger by a '
                                          'factor from 1 up to, but not reaching, 1.5. It constrains behaviour, like '
                                          "TD3+BC's own pull; it is not a penalty on values.",
                           "BCA's band": "BCA's band is an interval around the critic's prediction, sized on "
                                         "recorded rows kept out of training so that the critic's one-step error "
                                         'falls inside it on 90% of them. A wide band marks a place where the critic '
                                         'may be wrong.',
                           'Lower edge': "The lower edge is the critic's value minus the band's width. Used in "
                                         "training, it lowers the critic's value of the actor's own command where "
                                         'the band there is wider than at the recorded command, which is the kind of '
                                         'caution that value-pessimism methods apply.'},
          'starting-state': {'In distribution': 'In distribution means inside the conditions the training data were '
                                                'recorded in. Here that is the recorded conditions: a current of 0.1 '
                                                'to 0.7, a wave amplitude of 0.1 to 0.4 and a starting spread of '
                                                '0.5.',
                             'Out of distribution': 'Out of distribution means outside the conditions the training '
                                                    'data were recorded in. Every state of strong current and of big '
                                                    'waves is out of distribution, and part of the states of big '
                                                    'starting pushes.'},
          'episode-end': {'Recorded conditions': 'The recorded conditions are the rule the training data were drawn '
                                                 'from: a current of 0.1 to 0.7, a wave amplitude of 0.1 to 0.4, and '
                                                 'float numbers that start with a spread (standard deviation) of 0.5 '
                                                 'around rest. Here the learned actor must not lose points: adding '
                                                 'the caution of Bayesian Conformal Aggregation must cost it clearly '
                                                 'less than 2 points (the no-loss check). In the process it is step '
                                                 '1, where the data were collected, and the in-distribution part of '
                                                 'step 5, the test.',
                          'Strong current': 'The current is drawn from 0.7 to 1.1, stronger than any recorded (at '
                                            'most 0.7), while the waves and starts are as recorded. Every state is '
                                            'outside the recorded conditions, and 93.7% are in the full-throttle '
                                            'region. This is the main test. In the process it is part of step 5, the '
                                            'test: a dynamics extrapolation, because the current goes beyond any '
                                            'recorded one.',
                          'Big waves': 'The wave amplitude is drawn from 0.4 to 0.7, bigger than any recorded (at '
                                       "most 0.4), while the current is as recorded. The water's push ranges from "
                                       '−0.6 to 1.4 and points upstream on 20.3% of states. In the process it is '
                                       "part of step 5, the test: a disturbance shift, because the wave's pushes are "
                                       'larger than any recorded.',
                          'Big starting pushes': 'Every float number of every start is three times as large (a '
                                                 'spread of 1.5 instead of 0.5), while the water is as recorded. A '
                                                 "starting state's float cost averages 22.8 instead of 2.5. In the "
                                                 'process it is part of step 5, the test: an initial-state shift, '
                                                 'because the starts lie far outside the recorded starts.',
                          'Stalling world': 'Past 80% throttle the propeller stalls: its push is 0.65 at 90% and 0.5 '
                                            'at full throttle, while the throttle cost keeps rising. The data were '
                                            'recorded in this world. On the points scale, 0 is the recording '
                                            'controller and 100 is the stall-aware controller. In the process it is '
                                            'the world of step 1, where the data were recorded, and its stall is '
                                            'what the removal of step 2 hides.',
                          'No-stall comparison world': 'Push keeps rising up to 100% throttle. At or below 80% this '
                                                       'world is identical to the stalling world, so the data left '
                                                       'after the densest-15% removal, where only 60 of the 42,500 '
                                                       'kept training rows go past 80%, cannot tell the two apart. '
                                                       'Here 100 points is the rising-thrust controller. In the '
                                                       'process it is a control for step 2: the same removal, but '
                                                       'here it hides nothing.',
                          'The channel': "The drawing shows the state on screen: the top bar is the water's push c + "
                                         'w₁ (past 0.565 is the full-throttle region), the circles are the five '
                                         'floats (drawn below the dashed rest line when their position number is '
                                         'negative, with a small arrow for the speed number), and the squares '
                                         'between floats are the thrusters, each with an arrow as long as the push '
                                         "it really gives. The gauges at the bottom show each thruster's throttle "
                                         '(the bar) and push (the dark tick), and past the dashed 80% marks a '
                                         'propeller stalls. Hover over or tap a part for its numbers. The frame '
                                         'around the drawing has the colour of the chosen situation, and the top bar '
                                         'and its number turn gold when the state is in the full-throttle region.',
                          "Water's push c + w₁": 'This is the current c plus the push w₁ that the wave adds in this '
                                                 'state, and it is the same on every float. Above 0.565 the state is '
                                                 'in the full-throttle region (its number then turns gold, the '
                                                 "region's colour on this page), where the recording controller "
                                                 'flips a coin for the full-throttle command.'},
          'description': {'Throttle size': "Drag to set the size of one thruster's throttle, from off (0%) to full "
                                           'throttle (100%). The arrow keys move it by 1%, and Page Up and Page Down '
                                           'by 10%.'},
          'methods': {'Behaviour constraint': 'A behaviour constraint keeps the learned actor close to the commands '
                                              'in the recording by pulling it toward them. The weight on that pull '
                                              'in twin delayed deep deterministic policy gradient plus behaviour '
                                              'cloning (TD3+BC) works this way, and so does the dose of Bayesian '
                                              'Conformal Aggregation (BCA).',
                      "BCA's band": "BCA's band is an interval around the critic's prediction, sized on recorded "
                                    "rows kept out of training so that the critic's one-step error falls inside it "
                                    'on 90% of them. A wide band marks a place where the critic may be wrong.'}},
 'legends': {'starting-state': [{'heading': 'What the symbols mean',
                                 'rows': [['What a start draws',
                                           '<i>s</i><sub>0,<i>i</i></sub>',
                                           'It is float number <i>i</i> of the starting state (state 0); the rule '
                                           'draws each of the ten float numbers this way.'],
                                          ['What a start draws',
                                           '<i>c</i>',
                                           "It is the current's strength, drawn once and then fixed for the whole "
                                           'episode.'],
                                          ['What a start draws',
                                           'φ',
                                           "It is the wave's phase, where the wave starts in its cycle: an angle "
                                           'from 0 up to, but not including, 2π, one full turn.'],
                                          ['What a start draws',
                                           '<i>w</i><sub>1</sub>, <i>w</i><sub>2</sub>',
                                           'They are the two wave numbers that follow from the phase: amplitude × '
                                           'cos φ, the push the wave adds now, and amplitude × sin φ, the push it '
                                           'added a quarter cycle earlier.'],
                                          ['How each is drawn',
                                           '∼',
                                           'It means that the value on its left is drawn at random from the '
                                           'distribution on its right.'],
                                          ['How each is drawn',
                                           '𝒩(0, σ<sup>2</sup>)',
                                           'It is the normal distribution, the familiar bell-shaped spread, with '
                                           'mean 0 and standard deviation σ.'],
                                          ['How each is drawn',
                                           '<i>U</i>[low, high]',
                                           'It is the uniform distribution: every value from low to high is equally '
                                           'likely.'],
                                          ['The settings that the situations change',
                                           'σ',
                                           'It is the starting spread of the float numbers: 0.5, or 1.5 with <span '
                                           'class="km" data-sit="big_kicks">big starting pushes</span>.'],
                                          ['The settings that the situations change',
                                           '<i>c</i><sub>low</sub>, <i>c</i><sub>high</sub>',
                                           "They are the ends of the current's range: 0.1 and 0.7 as recorded, and "
                                           '0.7 and 1.1 in <span class="km" data-sit="strong_current">strong '
                                           'current</span>.'],
                                          ['The settings that the situations change',
                                           'amplitude',
                                           'It is the largest push the wave adds or takes away, drawn uniformly: 0.1 '
                                           'to 0.4 as recorded, and 0.4 to 0.7 in <span class="km" '
                                           'data-sit="big_waves">big waves</span>.']],
                                 'example': None}],
             'dynamics': [{'heading': 'What the symbols mean',
                           'rows': [['The state, now and one step later',
                                     '<i>s</i><sub><i>t</i></sub>',
                                     'They are the 10 float numbers at step <i>t</i>, observations 0 to 9.'],
                                    ['The state, now and one step later',
                                     '<i>s</i><sub><i>t</i>+1</sub>',
                                     'They are the same 10 numbers one step later.'],
                                    ['The floats on their own',
                                     '<i>A</i>',
                                     "It is the 10 × 10 table of how the floats move on their own: each float's "
                                     'swing turns by θ and shrinks by ρ every step, and the coupling <i>C</i> links '
                                     'the floats weakly.'],
                                    ['The floats on their own',
                                     'θ',
                                     "It is a float's turn per step, in radians: 0.100 for float 1 up to 0.801 for "
                                     'float 5 (the table below lists all five).'],
                                    ['The floats on their own',
                                     'ρ',
                                     "It is a float's shrink per step: 0.9817 for float 1 down to 0.9549 for float "
                                     '5.'],
                                    ['The floats on their own',
                                     '<i>C</i>',
                                     'It is the coupling, the part of <i>A</i> that links the floats: a fixed 10 × '
                                     '10 table of small numbers that adds, each step, at most 2% of the size of the '
                                     'ten float numbers. It is not the current <i>c</i>.'],
                                    ['The thrusters',
                                     '<i>a</i><sub><i>t</i></sub>',
                                     'They are the four throttles at step <i>t</i>, each limited to −1 to 1.'],
                                    ['The thrusters',
                                     '<i>f</i>(<i>a</i><sub><i>t</i></sub>)',
                                     "It is each thruster's push for its throttle: equal to the throttle up to 80%, "
                                     'and falling past it in the stalling world (<a href="#action-space">Action '
                                     'Space</a>).'],
                                    ['The thrusters',
                                     '<i>B</i>',
                                     'It is the 10 × 4 table of which floats each thruster pushes, with a 1 where '
                                     'thruster <i>k</i> pushes the speed number of float <i>k</i> or float <i>k</i> '
                                     '+ 1.'],
                                    ['The thrusters',
                                     '0.3',
                                     'It is the push factor, the same for the thrusters and the water.'],
                                    ['The water',
                                     '<i>v</i>',
                                     "It has a 1 at each float's speed number and a 0 at each position number, so "
                                     "the water pushes every float's speed equally."],
                                    ['The water',
                                     '<i>c</i><sub><i>t</i></sub>',
                                     'It is the current; <i>c</i><sub><i>t</i>+1</sub> = <i>c</i><sub><i>t</i></sub> '
                                     'says that it never changes.'],
                                    ['The water',
                                     '<i>w</i><sub>1,<i>t</i></sub>',
                                     'It is the push the wave adds at step <i>t</i>.'],
                                    ['The water',
                                     '<i>w</i><sub><i>t</i></sub>',
                                     "They are the wave's two numbers (<i>w</i><sub>1</sub>, <i>w</i><sub>2</sub>) "
                                     'at step <i>t</i>, a point that turns around a circle.'],
                                    ['The water',
                                     '<i>R</i>(θ)',
                                     'It turns a point by any angle, in radians (2π is one full turn), and this '
                                     "angle is not a float's θ. The wave uses <i>R</i>(0.27), which turns it 0.27 "
                                     'radians per step, so one cycle takes 23.27 steps.']],
                           'example': None}],
             'episode-end': [{'heading': 'Reading the drawing',
                              'rows': [['The water bar at the top',
                                        'Top bar',
                                        "It shows the water's push on every float, <i>c</i> + <i>w</i><sub>1</sub>: "
                                        'the current <i>c</i> plus the push <i>w</i><sub>1</sub> that the wave adds '
                                        'in this state.'],
                                       ['The water bar at the top',
                                        '<i class="key tile" data-k="region"></i>Full-throttle region',
                                        'The tinted end of the bar, past the line at 0.565, is the full-throttle '
                                        'region. When the state is in it, the bar and its number turn gold, and the '
                                        'recording controller gives the full-throttle command in about half of such '
                                        'states, by a coin flip.'],
                                       ['The channel and the floats',
                                        '<i class="key sitall"></i>Frame',
                                        'The frame around the drawing has the colour of the situation you picked: '
                                        '<span class="km" data-sit="recorded">recorded conditions</span>, <span '
                                        'class="km" data-sit="strong_current">strong current</span>, <span '
                                        'class="km" data-sit="big_waves">big waves</span> or <span class="km" '
                                        'data-sit="big_kicks">big starting pushes</span>.'],
                                       ['The channel and the floats',
                                        'Dashed line',
                                        'It marks rest. A float is drawn below it when its position number is '
                                        'negative, which is where the water carries it.'],
                                       ['The channel and the floats',
                                        'Number above a float',
                                        "It is the float's position number <i>p</i>, one of its two numbers, which "
                                        'are both 0 at rest. An arrow after it (↓ or ↑) means the float is past the '
                                        'edge of the drawing in that direction.'],
                                       ['The channel and the floats',
                                        'Small arrow by a float',
                                        "It shows the float's speed number <i>v</i>, the other coordinate of its "
                                        'swing.'],
                                       ['The channel and the floats',
                                        'Words at the bottom left',
                                        'With the recording controller, they say which of its two behaviours gave '
                                        "this state's command: normal driving, or the full-throttle command, written "
                                        'in gold.'],
                                       ['The thrusters and their gauges',
                                        'Arrow at a thruster',
                                        'It shows the push the thruster really gives, <i>f</i>(<i>a</i>); pointing '
                                        'up is upstream, against the water.'],
                                       ['The thrusters and their gauges',
                                        'Gauge bar',
                                        "It shows the thruster's throttle <i>a</i>, from −1 (full throttle against "
                                        'the water, at the top) to 1.'],
                                       ['The thrusters and their gauges',
                                        'Dark tick',
                                        'It shows the push <i>f</i>(<i>a</i>), which equals the throttle up to 80%.'],
                                       ['The thrusters and their gauges',
                                        '<i class="key tile" data-k="stall"></i>Red gauge, STALLED',
                                        'In the stalling world the dashed 80% marks are red, and so are the ends of '
                                        'each gauge beyond them. A throttle past them stalls: the gauge bar and the '
                                        'thruster turn red, the gauge reads STALLED, and the dark tick falls back.'],
                                       ['The thrusters and their gauges',
                                        'Numbers under a gauge',
                                        'They are the throttle first, then the push.'],
                                       ['The chart under the drawing',
                                        'Chart',
                                        'It shows the cost so far, which is minus the return so far, for all four '
                                        'controllers from this start; the dashed vertical line marks the state on '
                                        'screen.'],
                                       ['The chart under the drawing',
                                        '<i class="key line" data-k="recording"></i>Recording controller',
                                        'Its line wears the colour of the recorded data, because it made them.'],
                                       ['The chart under the drawing',
                                        '<i class="key line" data-k="stall_aware"></i>Stall-aware controller',
                                        'Its line is drawn in ink, because it is the 100-point reference.'],
                                       ['The chart under the drawing',
                                        '<i class="key line" data-k="rising_thrust"></i>Rising-thrust controller',
                                        'Its line is violet, the colour of push that keeps rising, which is what it '
                                        'assumes; the no-stall comparison world has the same colour.'],
                                       ['The chart under the drawing',
                                        '<i class="key dash" data-k="no_throttle"></i>No throttle',
                                        'Its dashed line is the water acting alone, with every thruster off.']],
                              'example': None}],
             'task': [{'heading': 'What the symbols in the drawing mean',
                       'rows': [[None,
                                 '<i>c</i>',
                                 "The current's strength: drawn at the start, then steady all episode; it pushes "
                                 'every float downstream equally. 0.1 to 0.7 in the recorded conditions; 0.445 on '
                                 'this start. The solid part of each top arrow.'],
                                [None,
                                 '<i>w</i><sub>1</sub>',
                                 "The wave's push now: it runs from plus the amplitude down to minus the amplitude "
                                 "and back every 23.27 steps. This start's amplitude is 0.218. The dashed part of "
                                 'each top arrow.'],
                                [None,
                                 '<i>c</i> + <i>w</i><sub>1</sub>',
                                 "The water's push on every float in this state. Downstream unless the wave takes "
                                 'away more than the current gives.'],
                                [None,
                                 '±',
                                 "+ when the wave adds to the current's push, − when it takes from it; the faint "
                                 'dashed part is then the push it takes away.'],
                                [None, 'rest', "Where every float's two numbers are 0: the dashed line."],
                                [None,
                                 'swing, in steps',
                                 "A float's period, the steps one swing takes, shown rounded to whole steps under "
                                 'each float: 63 for float 1 (62.8) down to 8 for float 5 (7.8).'],
                                [None,
                                 'push = throttle',
                                 'Up to 80%, a thruster pushes exactly as hard as its throttle.'],
                                [None,
                                 'stall',
                                 'Past 80% the propeller stalls: push 0.65 at 90% and 0.5 at full throttle, while '
                                 'the throttle cost keeps rising.'],
                                [None,
                                 'Bar and dark tick',
                                 "Thruster 1's throttle (the bar's fill) and the push it really gives (the tick)."],
                                [None,
                                 '<i class="key tile" data-k="region"></i>gold arrows',
                                 "While the drawing plays, the top arrows turn gold when the water's push is above "
                                 '0.565: the state is then in the full-throttle region.']],
                       'example': None},
                      {'heading': 'What the axes and marks mean',
                       'rows': [[None,
                                 'Across',
                                 "The water's push <i>c</i> + <i>w</i><sub>1</sub> at the state (<i>c</i> the "
                                 "current, <i>w</i><sub>1</sub> the wave's push now), in bands 0.2 wide."],
                                [None,
                                 'Up',
                                 "Thruster 1's average throttle or push. Negative is upstream, so higher on the "
                                 'chart, toward −1, means a harder push against the water.'],
                                [None, '80% line', 'At −0.8: past it the propeller stalls.'],
                                [None,
                                 '<i class="key tile" data-k="region"></i>Gold band',
                                 "It is the full-throttle region: the states where the water's push is above 0.565, "
                                 'where the recording controller gives the full-throttle command in half of the '
                                 'states.'],
                                [None,
                                 'Rising-thrust push',
                                 "What the rising-thrust controller's thruster 1 really delivers: it falls back once "
                                 'it stalls, to −0.51 at the strongest pushes.'],
                                [None,
                                 'Dots',
                                 "One band's average. Hover over or tap anywhere in a band's column for its three "
                                 'numbers; the table below lists them all, for keyboard and screen-reader use and '
                                 'with tooltips off.']],
                       'example': None}],
             'recording-controller': [{'heading': 'What the symbols mean',
                                       'rows': [['What the controller reads',
                                                 '<i>s</i>',
                                                 'They are the 10 float numbers.'],
                                                ['What the controller reads',
                                                 '<i>e</i>',
                                                 'It is the water: the current <i>c</i> and the wave numbers '
                                                 '<i>w</i><sub>1</sub> and <i>w</i><sub>2</sub>.'],
                                                ['Its fixed settings',
                                                 '<i>K</i><sub><i>s</i></sub>',
                                                 'It is the part of the textbook rule <i>K</i> that acts on the '
                                                 'float numbers.'],
                                                ['Its fixed settings',
                                                 '<i>K</i><sub><i>e</i></sub>',
                                                 'It is the part of <i>K</i> that acts on the water.'],
                                                ['Its fixed settings',
                                                 '0.25',
                                                 "It makes the controller damp the floats' swinging only a quarter "
                                                 'as hard as <i>K</i> would, while it still holds fully against the '
                                                 'water.'],
                                                ['Its random noise',
                                                 '<i>n</i> ∼ 𝒩(0, <i>I</i><sub>4</sub>)',
                                                 'They are four fresh random numbers each step, each with mean 0 and '
                                                 'standard deviation 1, independent of one another.'],
                                                ['Its random noise',
                                                 '0.3 <i>n</i>',
                                                 'It is the noise added to the command, with standard deviation '
                                                 '0.3.'],
                                                ['What it sends', '<i>u</i>', 'It is the command before squashing.'],
                                                ['What it sends',
                                                 '0.8 tanh(<i>u</i>/0.8)',
                                                 'It is the squashing, which keeps every throttle inside ±80%; tanh '
                                                 'is a smooth S-shaped function that turns any number into one '
                                                 'between −1 and 1.'],
                                                ['What it sends', '<i>a</i>', 'They are the four throttles sent.']],
                                       'example': '<b>Worked example.</b> With the floats at rest, a current of 0.5 '
                                                  "and no wave, only the water term acts: thruster 1's command "
                                                  'before noise is <span class="nw"><i>u</i><sub>1</sub> = −0.9102 × '
                                                  '0.5 = −0.455</span>, and the squashing sends <span '
                                                  'class="nw"><i>a</i><sub>1</sub> = 0.8 tanh(−0.455 / 0.8) = '
                                                  '−0.41</span>. Even with noise two standard deviations strong, '
                                                  '<span class="nw"><i>u</i><sub>1</sub> = −0.455 − 0.6 = '
                                                  '−1.055</span>, the squashing sends only −0.69, so normal driving '
                                                  'never reaches the stall.'},
                                      {'heading': 'What the symbols mean',
                                       'rows': [['Where the coin is flipped',
                                                 '<i class="key tile" data-k="region"></i><i>c</i> + '
                                                 '<i>w</i><sub>1</sub> &gt; 0.565',
                                                 "It marks the full-throttle region: the states where the water's "
                                                 'push is above 0.565. There, in each state, a coin flip chooses '
                                                 'normal driving or the full-throttle command.'],
                                                ['The pattern',
                                                 '<i>P</i>',
                                                 'It is the full-throttle pattern, (−1.000, −0.137, −0.585, −0.406): '
                                                 'thruster 1 at full throttle upstream.'],
                                                ['The pattern',
                                                 '<i>K</i><sub>·,<i>c</i></sub>',
                                                 "It is the current's column of the textbook rule <i>K</i>, (0.9102, "
                                                 '0.1247, 0.5324, 0.3698): how hard each thruster holds against a '
                                                 'current of 1.'],
                                                ['The pattern',
                                                 '|<i>K</i><sub>1,<i>c</i></sub>|',
                                                 "It is thruster 1's entry in that column, 0.9102; dividing by it "
                                                 'scales the hold up until thruster 1 is at full throttle.'],
                                                ['The noise and the limit',
                                                 '0.03 <i>n</i>',
                                                 'It is a little random noise: four fresh random numbers with mean 0 '
                                                 'and standard deviation 1, times 0.03.'],
                                                ['The noise and the limit',
                                                 'clip(…, −1, 1)',
                                                 'It limits each throttle to −1 to 1.'],
                                                ['What it sends',
                                                 '<i>a</i>',
                                                 'They are the four throttles sent. Thruster 1 is past 80%, so in '
                                                 'the stalling world it stalls and pushes only 0.5.']],
                                       'example': "<b>Worked example.</b> Dividing the current's column by thruster "
                                                  '1\'s entry gives <span class="nw"><i>P</i> = −(0.9102, 0.1247, '
                                                  '0.5324, 0.3698) / 0.9102</span> <span class="nw">= (−1.000, '
                                                  '−0.137, −0.585, −0.406)</span>. Its throttle cost is 0.458, '
                                                  'against 0.350 with thruster 1 held at 80%, and thruster 1 then '
                                                  'pushes only 0.5 instead of 0.8. These are the commands that the '
                                                  'main test\'s removal deletes <span class="stepchip in" '
                                                  'data-step="remove">Step 2</span>.'}],
             'methods': [{'heading': 'What the symbols mean',
                          'rows': [[None, '<i>y</i>', "It is the critic's one-step target for one recorded row."],
                                   [None, '<i>r</i>', "It is the row's reward."],
                                   [None, '<i>x</i>′', "It is the row's next observation."],
                                   [None, '0.98', "It is the discount, the weight of the next step's value."],
                                   [None,
                                    '<i>Q̄</i><sub><i>j</i></sub>',
                                    'It is the slowly averaged copy of critic <i>j</i>, for <i>j</i> = 1 and 2.'],
                                   [None, 'min', "It takes the lower of the two critics' estimates."],
                                   [None, 'π̄', 'It is the slowly averaged copy of the actor.'],
                                   [None,
                                    'ε',
                                    'It is the target noise: drawn from a normal distribution with standard '
                                    'deviation 0.2 and limited to ±0.5.'],
                                   [None,
                                    '<i>y</i> − <i>Q</i>(<i>x</i>, <i>a</i>)',
                                    "It is the critic's one-step error at the row."]],
                          'example': None},
                         {'heading': 'What the symbols mean',
                          'rows': [['What is lowered, and over what',
                                    '<i>L</i><sub>actor</sub>',
                                    "It is the actor's loss, the quantity each actor update lowers."],
                                   ['What is lowered, and over what',
                                    '<i>i</i>',
                                    'It is one recorded row of the batch of 256.'],
                                   ['What is lowered, and over what',
                                    'mean<sub><i>i</i></sub>',
                                    'It takes the average over the rows of the batch.'],
                                   ['The commands', '<i>x</i><sub><i>i</i></sub>', "It is the row's observation."],
                                   ['The commands',
                                    '<i>a</i><sub><i>i</i></sub>',
                                    "It is the row's recorded command."],
                                   ['The commands',
                                    '<i>a</i><sub><i>i</i>,<i>k</i></sub>',
                                    'It is the recorded throttle of thruster <i>k</i>.'],
                                   ['The commands',
                                    'π(<i>x</i><sub><i>i</i></sub>)',
                                    "It is the actor's command at the row's observation."],
                                   ['The commands',
                                    'π<sub><i>k</i></sub>(<i>x</i><sub><i>i</i></sub>)',
                                    "It is the actor's throttle for thruster <i>k</i>."],
                                   ['The two forces and their weights',
                                    '<i>Q</i><sub>1</sub>',
                                    "It is the first critic's estimate, which the actor climbs."],
                                   ['The two forces and their weights',
                                    'λ',
                                    "It is the critic's scale: α divided by the critic's typical size, "
                                    'mean|<i>Q</i><sub>1</sub>|.'],
                                   ['The two forces and their weights',
                                    'α',
                                    'It is the critic weight: 2.5 as published, and 5 in the main test, where the '
                                    "harm checks' step-up rule chose it; the weighting test keeps 2.5."],
                                   ['The two forces and their weights',
                                    '<i>b</i><sub><i>i</i></sub>',
                                    'It is the behaviour-cloning pull: the squared gap to the recorded command, '
                                    'averaged over the four thrusters.'],
                                   ['The two forces and their weights',
                                    '<i>m</i><sub><i>i</i></sub>',
                                    'It is the caution multiplier, or dose: 1 in plain twin delayed deep '
                                    'deterministic policy gradient plus behaviour cloning (TD3+BC), and from 1 up to '
                                    '1.5 with Bayesian Conformal Aggregation (BCA).']],
                          'example': None},
                         {'heading': 'What the symbols mean',
                          'rows': [[None, 'π', 'It is where the actor settles at one state.'],
                                   [None, '<i>a</i>', 'It is the recorded command.'],
                                   [None,
                                    '<i>â</i>',
                                    'It is the command the critic rates highest at this state, read "a-hat". It is '
                                    "the critic's own choice, which need not be the truly best command."],
                                   [None, '<i>m</i>', 'It is the dose, which sets the strength of the pull.'],
                                   [None,
                                    'μ',
                                    "It is the critic's pull, on the same scale as <i>m</i>; the forecast uses "
                                    '0.58.'],
                                   [None, '<i>d</i>', 'It is the number of throttles, 4.'],
                                   [None,
                                    'λ',
                                    "It is the critic's scale in the actor's loss: α divided by the critic's typical "
                                    'size.'],
                                   [None,
                                    'κ',
                                    "It is the curvature of the critic's value around <i>â</i>: 2.36, the median "
                                    "measured on the exact value of normal driving (the recording controller's "
                                    'ordinary mode, which never goes past 80%) at its own command.'],
                                   [None,
                                    'μ / (<i>m</i> + μ)',
                                    'It is the share of the way toward <i>â</i> for a critic that curves down past '
                                    '80%.'],
                                   [None,
                                    'min(μ / <i>m</i>, 1)',
                                    'It is the share of the way toward <i>â</i> for a critic that keeps its slope '
                                    'past 80%.']],
                          'example': None},
                         {'heading': 'What the symbols mean',
                          'rows': [[None,
                                    '<i>o</i>(<i>x</i>)',
                                    "It is the critic's extra preference for full throttle at observation <i>x</i>, "
                                    'a kept training row in the full-throttle region.'],
                                   [None,
                                    '<i>Q</i>',
                                    'It is the critic being checked, trained with the controller it scores held '
                                    'fixed.'],
                                   [None,
                                    '<i>Q</i><sup>true</sup>',
                                    'It is the exact value of that same fixed controller, found by simulation.'],
                                   [None, '<i>P</i>', 'It is the full-throttle pattern.'],
                                   [None,
                                    '<i>P</i><sub>cap</sub>',
                                    'It is the same pattern with thruster 1 capped at 80%.'],
                                   [None,
                                    'δ(<i>x</i>, <i>a</i>)',
                                    "It is the critic's error against the exact value at command <i>a</i>."],
                                   [None,
                                    'δ<sub>A</sub>(<i>x</i>, <i>a</i>)',
                                    'It is the advantage error: the error at <i>a</i> minus the error at the fixed '
                                    "controller's own command."]],
                          'example': None},
                         {'heading': 'What the symbols mean',
                          'rows': [[None,
                                    '<i>J</i><sub>new</sub> − <i>J</i><sub>ref</sub>',
                                    'It is the difference in return between the new actor and the reference actor.'],
                                   [None,
                                    '<i>x</i><sub><i>t</i></sub>',
                                    "It is the new actor's own state at step <i>t</i>, on its path from "
                                    '<i>x</i><sub>0</sub> to <i>x</i><sub>99</sub>.'],
                                   [None, 'π<sub>new</sub>', "It is the new actor's command."],
                                   [None, 'π<sub>ref</sub>', "It is the reference actor's command."],
                                   [None,
                                    '<i>Q</i><sub>ref</sub><sup>100−<i>t</i></sup>(<i>x</i>, <i>a</i>)',
                                    'It is the exact return over the remaining 100 − <i>t</i> steps of giving '
                                    '<i>a</i> at <i>x</i> and then following the reference actor.'],
                                   [None,
                                    '[ … ]',
                                    "Each bracket is the return the new actor's choice gains at that state, and "
                                    'minus the bracket is the return lost there.']],
                          'example': None},
                         {'heading': 'What the symbols mean',
                          'rows': [[None,
                                    'Δ<sub><i>i</i></sub>',
                                    'It is the over-preference at training row <i>i</i>.'],
                                   [None,
                                    '<i>Q</i>',
                                    'It is the frozen critic of a run of plain twin delayed deep deterministic '
                                    'policy gradient plus behaviour cloning (TD3+BC) on the log after the removal.'],
                                   [None,
                                    '<i>Q</i><sup>true</sup>',
                                    'It is the exact value of the actor being trained, recomputed every 4,000 '
                                    'steps.'],
                                   [None,
                                    'π(<i>x</i><sub><i>i</i></sub>)',
                                    "It is the actor's command at the row's observation."],
                                   [None, '<i>a</i><sub><i>i</i></sub>', "It is the row's recorded command."],
                                   [None,
                                    '[ … ]<sub>+</sub>',
                                    'It keeps the bracket when the bracket is positive and gives 0 otherwise.']],
                          'example': None},
                         {'heading': 'What the symbols in the doses and chart mean',
                          'rows': [[None,
                                    '<i>m</i><sub><i>i</i></sub>',
                                    "It is the dose at row <i>i</i>, the chart's height."],
                                   [None,
                                    'Δ / mean(Δ)',
                                    "It is the row's over-preference divided by its mean over the rows, the chart's "
                                    'horizontal axis.'],
                                   [None, '<i>z</i><sub><i>i</i></sub>', 'It is 3 Δ<sub><i>i</i></sub> / mean(Δ).'],
                                   [None,
                                    '1 + 0.5 <i>z</i> / (1 + <i>z</i>)',
                                    'It is the harm-placed dose in the form of Bayesian Conformal Aggregation (BCA): '
                                    '1 with no over-preference, 1.375 at the mean, and always below the cap of 1.5.'],
                                   [None,
                                    '1 + <i>z</i> / (1 + <i>z</i>)',
                                    'It is the wider-range diagnostic, with doses up to 2.']],
                          'example': None},
                         {'heading': 'What the symbols mean',
                          'rows': [[None,
                                    '<i>m</i><sub><i>i</i></sub>',
                                    'It is the dose of Bayesian Conformal Aggregation (BCA) at training row '
                                    '<i>i</i>, from 1 up to 1.5.'],
                                   [None,
                                    '<i>U</i><sub><i>i</i></sub>',
                                    "It is the band width at the row: β × η × the predicted size of the critic's "
                                    'error at the recorded command.'],
                                   [None,
                                    'β',
                                    'It is the band multiplier, the larger of the conformal value and the Bayesian '
                                    'value, both set on the calibration set.'],
                                   [None,
                                    'η',
                                    "It is the running unit, the typical size of the critic's one-step errors in "
                                    'recent batches.']],
                          'example': None}],
             'harm-checks': [{'heading': 'What the symbols mean',
                              'rows': [['The result',
                                        '<i>N</i>',
                                        'It is the number of repeat runs (seeds) that the test of Bayesian Conformal '
                                        'Aggregation (BCA) would use: at least 10, and above 20 the test stops.'],
                                       ['The result',
                                        '<i>N</i><sub>0</sub>',
                                        'It is the number of seeds needed to detect a 3-point effect.'],
                                       ['The result',
                                        '⌈<i>N</i><sub>0</sub>⌉',
                                        'It is that number rounded up to a whole number of seeds.'],
                                       ['What is measured',
                                        'σ',
                                        'It is the larger of two standard deviations over the harm-check seeds, one '
                                        'for each of the two paired differences, in points.'],
                                       ['Numbers in the formula',
                                        '2.128',
                                        'It sets a 5% false-alarm rate shared over three tests, one per shifted '
                                        'situation.'],
                                       ['Numbers in the formula',
                                        '0.842',
                                        'It gives an 80% chance of detecting a true 3-point effect.'],
                                       ['Numbers in the formula',
                                        '1.25',
                                        'It enlarges σ by 25%, because σ is estimated from only five seeds and may '
                                        'come out too small.'],
                                       ['Numbers in the formula',
                                        '3',
                                        'It is the smallest effect worth detecting, in points.']],
                              'example': '<b>Worked example.</b> If the spread were σ = 3 points, then '
                                         '<i>N</i><sub>0</sub> = ((2.128 + 0.842) × 1.25 × 3 / 3)² = 3.71² ≈ 13.8, '
                                         "which rounds up to 14 seeds: the table's row for σ = 3. At σ = 4 the test "
                                         'would need 25 seeds, more than 20, and would stop as underpowered. The '
                                         'check never ran, because the best-case caution check stopped the test '
                                         'first.'}]},
 'cols': {'action-space': [{'cols': {'Action space': "This row gives the action space in Gymnasium's terms: four "
                                                     'numbers, each from −1 to 1, stored as float32.',
                                     'Order': 'This row says which throttle is which: thrusters 1 to 4, numbered 0 '
                                              'to 3 in the code.',
                                     'Sign': 'This row says what the sign of a throttle means: negative pushes '
                                             'upstream, against the water.',
                                     'Push': 'This row says how hard a throttle really pushes: as hard as the '
                                             'throttle up to 80%, and less past it in the stalling world.',
                                     'Throttle cost': 'This row gives what a throttle costs in the reward, which is '
                                                      'charged even while a thruster stalls.',
                                     'Storage': 'This row says how a command is stored and limited before it acts.',
                                     "The main test's data": "This row says where the main test's recorded commands "
                                                             'lie: only the full-throttle command goes past 80%.',
                                     "After the main test's removal": 'This row says what is left past 80% once the '
                                                                      "main test's densest-15% removal has deleted "
                                                                      'its rows.',
                                     "The weighting test's data": "This row says where the weighting test's recorded "
                                                                  'commands lie and what its removal deletes, the '
                                                                  '80% hold.'}},
                           {'cols': {'Index': "This is the throttle's place in the action of 4 numbers, counted from "
                                              '0 as the code does.',
                                     'Action': "This says which thruster's throttle the number sets.",
                                     'Control Min': 'The lowest allowed throttle is −1, full throttle upstream, '
                                                    'against the water.',
                                     'Control Max': 'The highest allowed throttle is 1, full throttle downstream. A '
                                                    'command outside −1 to 1 is clipped to that range.',
                                     'Pushes the speeds of': 'This column gives the two floats whose speed numbers '
                                                             "this thruster's push enters, scaled by the push factor "
                                                             '0.3, with their places in the list of 13 observed '
                                                             'numbers.',
                                     'Stalls': 'The propeller stalls past 80% throttle, and only in the stalling '
                                               'world. A stalled propeller pushes less while its throttle cost keeps '
                                               'rising.'}},
                           {'cols': {'Throttle': "This is how far one thruster's throttle is opened, as a share of "
                                                 'full throttle, in either direction.',
                                     'Push, stalling world': 'This is the push the thruster really gives in the '
                                                             'stalling world: equal to the throttle up to 80%, then '
                                                             '0.8 − 1.5 × (throttle − 0.8), down to 0.5 at full '
                                                             'throttle.',
                                     'Push, no-stall comparison world': 'In the no-stall comparison world the push '
                                                                        'always equals the throttle.',
                                     'Stalls': 'This says whether the propeller stalls at this throttle in the '
                                               'stalling world. Only commands above 0.800000011920929 stall. This '
                                               'number is the stored value nearest 0.8, because each throttle is '
                                               'stored with about 7 significant digits, so a command of exactly 80% '
                                               'never stalls.',
                                     'Throttle cost 0.3 a²': "This thruster's throttle cost in one step is 0.3 times "
                                                             'its squared throttle a. It is charged even while the '
                                                             'propeller stalls.'}},
                           {'cols': {'Throttle band': "Each row is one band of a thruster's throttle, 20% wide; a "
                                                      'negative throttle pushes upstream, and the first and last '
                                                      'bands are past 80%, where a propeller stalls in the stalling '
                                                      'world.',
                                     'Thruster 1': "These two columns count the training rows with thruster 1's "
                                                   "throttle in the row's band.",
                                     'Thruster 1: all rows': 'This is how many of the 50,000 training rows have '
                                                             "thruster 1's throttle in this band, before any "
                                                             'removal, in the data that the buttons above the table '
                                                             'choose.',
                                     'Thruster 1: rows kept': 'This is how many of the 42,500 rows kept after the '
                                                              "removal have thruster 1's throttle in this band: the "
                                                              'rows the learner trains on.',
                                     'Thruster 2': "These two columns count the training rows with thruster 2's "
                                                   "throttle in the row's band.",
                                     'Thruster 2: all rows': 'This is how many of the 50,000 training rows have '
                                                             "thruster 2's throttle in this band, before any "
                                                             'removal, in the data that the buttons above the table '
                                                             'choose.',
                                     'Thruster 2: rows kept': 'This is how many of the 42,500 rows kept after the '
                                                              "removal have thruster 2's throttle in this band: the "
                                                              'rows the learner trains on.',
                                     'Thruster 3': "These two columns count the training rows with thruster 3's "
                                                   "throttle in the row's band.",
                                     'Thruster 3: all rows': 'This is how many of the 50,000 training rows have '
                                                             "thruster 3's throttle in this band, before any "
                                                             'removal, in the data that the buttons above the table '
                                                             'choose.',
                                     'Thruster 3: rows kept': 'This is how many of the 42,500 rows kept after the '
                                                              "removal have thruster 3's throttle in this band: the "
                                                              'rows the learner trains on.',
                                     'Thruster 4': "These two columns count the training rows with thruster 4's "
                                                   "throttle in the row's band.",
                                     'Thruster 4: all rows': 'This is how many of the 50,000 training rows have '
                                                             "thruster 4's throttle in this band, before any "
                                                             'removal, in the data that the buttons above the table '
                                                             'choose.',
                                     'Thruster 4: rows kept': 'This is how many of the 42,500 rows kept after the '
                                                              "removal have thruster 4's throttle in this band: the "
                                                              'rows the learner trains on.'}}],
          'observation-space': [{'cols': {'Observation': 'This says which of the 13 observed numbers the row is, its '
                                                         'place in the observation counted from 0 as the code does, '
                                                         'what it describes and where the live panel draws it.',
                                          'At rest': "This is the number's value when the floats are at rest; the "
                                                     "water's numbers have none, because the water never rests.",
                                          'Recorded conditions': 'This is where the number lies in the recorded '
                                                                 'conditions, the in-distribution test: the middle '
                                                                 '90% of its values for the floats, and the exact '
                                                                 'range for the water.',
                                          'Strong current': 'This is where the number lies in strong current, a '
                                                            'dynamics extrapolation: the middle 90% of its values '
                                                            'for the floats, and the exact range for the water.',
                                          'Big waves': 'This is where the number lies in big waves, a disturbance '
                                                       'shift: the middle 90% of its values for the floats, and the '
                                                       'exact range for the water.',
                                          'Big starting pushes': 'This is where the number lies in big starting '
                                                                 'pushes, an initial-state shift: the middle 90% of '
                                                                 'its values for the floats, and the exact range for '
                                                                 'the water.'}}],
          'starting-state': [{'cols': {'Situation': 'Each row is one of the four rules for drawing a start.',
                                       'Current: any value from': 'The current is drawn uniformly from this range '
                                                                  'and then fixed for the whole episode.',
                                       'Wave amplitude: any value from': "The wave's amplitude, the largest push it "
                                                                         'adds or takes away, is drawn uniformly '
                                                                         'from this range.',
                                       'Float spread at the start (standard deviation)': 'Each of the ten float '
                                                                                         'numbers starts at a normal '
                                                                                         'random value around rest '
                                                                                         'with this standard '
                                                                                         'deviation.',
                                       'In the usual terms': 'This is the standard name in offline reinforcement '
                                                             'learning for the kind of test the situation is.',
                                       'Where it sits in the process': 'This says which step of the offline '
                                                                       'reinforcement-learning process the situation '
                                                                       'belongs to.'}}],
          'description': [{'cols': {'Action Space': 'A controller sends 4 throttles each step, each from −1 to 1.',
                                    'Observation Space': 'A controller reads the 13 numbers of the state each step, '
                                                         'and each number has bounds.',
                                    'import': ['Import lines',
                                               'These Python lines register ThrusterChain with Gymnasium, the '
                                               'standard Python interface for environments in which controllers act '
                                               'and learn, and then create the environment.'],
                                    'Episode': 'This row says how long one episode lasts, how it ends, and how much '
                                               'later rewards count.',
                                    'Registered ids': 'These are the names that create the environment, one for each '
                                                      'situation in each world.',
                                    'Code': "This row gives the file that holds the environment's code and the "
                                            'version of numpy, the Python library for arrays of numbers, that it '
                                            'uses.'}},
                          {'cols': {'Throttle': "Each row is one size of a thruster's throttle, from off to full "
                                                'throttle.',
                                    'Push, stalling world': 'This is the push the thruster really gives at this '
                                                            'throttle in the stalling world.',
                                    'Push, no-stall comparison world': 'This is the push at this throttle in the '
                                                                       'no-stall comparison world, where it keeps '
                                                                       'rising.',
                                    'Throttle cost': 'This is the throttle cost of one thruster, 0.3 times the '
                                                     'squared throttle, the same in both worlds.',
                                    'True value against 80%, stalling world': 'This is the true value of the command '
                                                                              'at the example state, minus its true '
                                                                              'value with thruster 1 at 80%, in the '
                                                                              'stalling world; negative means worse.',
                                    'True value against 80%, no-stall comparison world': 'This is the same '
                                                                                         'difference of true values '
                                                                                         'in the no-stall comparison '
                                                                                         'world; negative means '
                                                                                         'worse.'}},
                          {'cols': {'Size of the throttle': "Each row is one band of thruster 1's throttle size, "
                                                            'upstream or downstream; 80% itself belongs to the band '
                                                            'below it.',
                                    'Full data: rows': "This is how many of the main test's 50,000 training rows "
                                                       "have thruster 1's throttle in this band.",
                                    'Full data: share': 'This is that count as a share of the 50,000 rows.',
                                    'After the removal: rows': 'This is how many of the 42,500 rows kept after the '
                                                               "densest-15% removal have thruster 1's throttle in "
                                                               'this band.',
                                    'After the removal: share': 'This is that count as a share of the 42,500 kept '
                                                                'rows.',
                                    'Rows deleted': 'This is how many rows in this band the densest-15% removal '
                                                    'deleted.'}}],
          'dataset': [{'cols': {'Field': 'Each row is one field of every recorded row, by its plain name; the names '
                                         'stored in the file are in the section Names used in the code.',
                                'Type': 'This says how the field is stored: float32 and float64 are decimal numbers '
                                        'with about 7 and about 16 significant digits, int32 and int8 are whole '
                                        'numbers, and bool is yes or no.',
                                'Shape (one data seed)': 'This says how many numbers the field holds in the log of '
                                                         'one recording (one data seed, which fixes its random '
                                                         'draws): one per recorded row (1,000 episodes of 100 '
                                                         'steps), times the numbers in each row.',
                                'What it holds': 'This says what the field records for each row.'}},
                      {'cols': {'Split': 'The recorded episodes are divided into three parts, and each row is one of '
                                         'them.',
                                'Episodes': 'This is the number of the 1,000 recorded episodes that belong to this '
                                            'split.',
                                'Rows': 'This is the number of recorded steps in the split, 100 per episode.',
                                'Used for': 'This says what the split is used for.'}},
                      {'cols': {'Field': 'The rows show the 13 observed numbers, the 4 throttles, the reward and the '
                                         "two end flags as the file stores them, then the water's push and three "
                                         'markers that explain the step.',
                                't = 0': 'This column is state 0 of training episode 2, counted from 0 (state 0 is '
                                         "the start), in the recorded log that this page uses. It holds that state's "
                                         'recorded row: what the controller read, the command it gave and the '
                                         'reward. This row is kept: its normal-driving command lies outside the '
                                         'removed cluster, so the learner still sees this state, but only with a '
                                         'command below 80%.',
                                't = 1, deleted': 'This column is state 1 of training episode 2, counted from 0 '
                                                  '(state 0 is the start), in the recorded log that this page uses. '
                                                  "It holds that state's recorded row: what the controller read, the "
                                                  'command it gave and the reward. This row is deleted by the main '
                                                  "test's removal before training: its command, thruster 1 near full "
                                                  'throttle, lies inside the densest 15% of recorded commands. The '
                                                  'learner never sees it, so this record of what thruster 1 does '
                                                  'past 80% is lost.',
                                't = 2': 'This column is state 2 of training episode 2, counted from 0 (state 0 is '
                                         "the start), in the recorded log that this page uses. It holds that state's "
                                         'recorded row: what the controller read, the command it gave and the '
                                         'reward. This row is kept: its normal-driving command lies outside the '
                                         'removed cluster, so the learner still sees this state, but only with a '
                                         'command below 80%.',
                                't = 3': 'This column is state 3 of training episode 2, counted from 0 (state 0 is '
                                         "the start), in the recorded log that this page uses. It holds that state's "
                                         'recorded row: what the controller read, the command it gave and the '
                                         'reward. This row is kept: its normal-driving command lies outside the '
                                         'removed cluster, so the learner still sees this state, but only with a '
                                         'command below 80%.'}},
                      {'cols': {'Field': 'The rows show the 13 observed numbers, the 4 throttles, the reward and the '
                                         "two end flags as the file stores them, then the water's push and three "
                                         'markers that explain the step.',
                                't = 4': 'This column is state 4 of training episode 2, counted from 0 (state 0 is '
                                         "the start), in the recorded log that this page uses. It holds that state's "
                                         'recorded row: what the controller read, the command it gave and the '
                                         'reward. This row is kept: its normal-driving command lies outside the '
                                         'removed cluster, so the learner still sees this state, but only with a '
                                         'command below 80%.',
                                't = 5': 'This column is state 5 of training episode 2, counted from 0 (state 0 is '
                                         "the start), in the recorded log that this page uses. It holds that state's "
                                         'recorded row: what the controller read, the command it gave and the '
                                         'reward. This row is kept: its normal-driving command lies outside the '
                                         'removed cluster, so the learner still sees this state, but only with a '
                                         'command below 80%.',
                                't = 6, deleted': 'This column is state 6 of training episode 2, counted from 0 '
                                                  '(state 0 is the start), in the recorded log that this page uses. '
                                                  "It holds that state's recorded row: what the controller read, the "
                                                  'command it gave and the reward. This row is deleted by the main '
                                                  "test's removal before training: its command, thruster 1 near full "
                                                  'throttle, lies inside the densest 15% of recorded commands. The '
                                                  'learner never sees it, so this record of what thruster 1 does '
                                                  'past 80% is lost.',
                                't = 7': 'This column is state 7 of training episode 2, counted from 0 (state 0 is '
                                         "the start), in the recorded log that this page uses. It holds that state's "
                                         'recorded row: what the controller read, the command it gave and the '
                                         'reward. This row is kept: it lies outside the full-throttle region, where '
                                         'the recording controller only drives normally.'}},
                      {'cols': {'State number': 'Each row is one of the ten float numbers of the exact starting '
                                                'state (state 0).',
                                'Exact value (float64)': "This is the number's exact value, shown to 8 decimal "
                                                         'places; float64 means it is stored as a decimal number '
                                                         'with about 16 significant digits.',
                                'Square': "This is the value squared, and the ten squares add up to the state's "
                                          'float cost (the last row).'}}],
          'methods': [{'cols': {'Method': 'Each row is one method used by the checks and tests, linked to its '
                                          'explanation below.',
                                'Pipeline step': 'This is the step of the offline reinforcement-learning pipeline '
                                                 'that the method stands for in ThrusterChain: collect the data, '
                                                 'remove commands, train the learner (twin delayed deep '
                                                 'deterministic policy gradient plus behaviour cloning (TD3+BC)), '
                                                 'build the band of Bayesian Conformal Aggregation (BCA), or test in '
                                                 'the shifted situations.',
                                'Used by': 'This says which check or test uses the method.'}},
                      {'group': 'What the settings control',
                       'cols': {'Setting': 'Each row is one setting of the learner, twin delayed deep deterministic '
                                           'policy gradient plus behaviour cloning (TD3+BC); the heading rows group '
                                           'the settings by what they control.',
                                'Value': "This is the setting's value in the runs of the main test and the weighting "
                                         'test.',
                                'What it does': 'This says in plain words what the setting does during learning.',
                                'Why it matters here': "This says why the setting matters for ThrusterChain's test "
                                                       'of the missing commands.'}},
                      {'cols': {'Caution multiplier m': 'The caution multiplier m, also called the dose, sets how '
                                                        'hard the actor is pulled toward the recorded command at a '
                                                        'row. A value of 1 is plain twin delayed deep deterministic '
                                                        'policy gradient plus behaviour cloning (TD3+BC), and the '
                                                        'doses of Bayesian Conformal Aggregation (BCA) stay below '
                                                        '1.5.',
                                'Curving critic (bends past 80%): share μ / (m + μ)': 'This is how far the actor '
                                                                                      'goes from the recorded '
                                                                                      'command toward â, the command '
                                                                                      'the critic rates highest (0 '
                                                                                      'keeps the recorded command, 1 '
                                                                                      'reaches â), for a critic that '
                                                                                      'curves down past 80% as the '
                                                                                      'exact value does, at the '
                                                                                      "critic's pull μ = 0.58.",
                                'Straight critic (keeps its slope past 80%): share min(μ / m, 1)': 'This is the same '
                                                                                                   'share for a '
                                                                                                   'critic that '
                                                                                                   'keeps its slope '
                                                                                                   'past the '
                                                                                                   'recorded '
                                                                                                   'commands: μ / m, '
                                                                                                   'but never more '
                                                                                                   'than 1.'}},
                      {'cols': {'Situation': 'Each row is the evaluation bank that the forecast is scored on.',
                                'Forecast harm, curving critic (bends past 80%), points': "This is the design plan's "
                                                                                          'forecast of the points '
                                                                                          'lost to pushing past 80% '
                                                                                          'with a critic that curves '
                                                                                          'down past 80%. The plan '
                                                                                          'stood in for the learned '
                                                                                          'actor with normal driving '
                                                                                          '(the recording '
                                                                                          "controller's ordinary "
                                                                                          'mode, which never goes '
                                                                                          'past 80%) moved part of '
                                                                                          'the way toward the '
                                                                                          'rising-thrust '
                                                                                          "controller's command, and "
                                                                                          'the forecast is the '
                                                                                          'points of that fixed '
                                                                                          'imitation with every '
                                                                                          'throttle capped at 80% '
                                                                                          'minus the points of the '
                                                                                          'fixed imitation itself.',
                                'Forecast harm, straight critic (keeps its slope past 80%), points': 'This is the '
                                                                                                     'same forecast '
                                                                                                     'for a critic '
                                                                                                     'that keeps its '
                                                                                                     'slope past the '
                                                                                                     'recorded '
                                                                                                     'commands.'}},
                      {'cols': {'Over-preference Δ ÷ its mean': "This is the row's over-preference Δ (how much more "
                                                                "the critic prefers the actor's command over the "
                                                                'recorded one than the exact values do) divided by '
                                                                'its mean over the rows.',
                                'z = 3Δ ÷ mean(Δ)': 'This is three times that ratio, the number z that the dose '
                                                    'formulas use.',
                                "Harm-placed dose (BCA's form)": 'This is the caution multiplier 1 + 0.5 z / (1 + '
                                                                 'z), the form that Bayesian Conformal Aggregation '
                                                                 '(BCA) uses, which runs from 1 up toward, but never '
                                                                 'reaches, 1.5.',
                                'Wider-range dose (diagnostic, up to 2)': 'This is the diagnostic multiplier 1 + z / '
                                                                          '(1 + z), which runs up toward 2. It shows '
                                                                          'whether the cap of 1.5 in Bayesian '
                                                                          'Conformal Aggregation is what limits the '
                                                                          'dose.'}}],
          'harm-checks': [{'cols': {'Rule': "This is the rule's letter, a to f, as in the pre-registration, the plan "
                                            'written down before any learner ran.',
                                    'The question': 'This is the question the rule answers, in plain words. Each '
                                                    "rule rules out one other explanation of the critic's error.",
                                    'Pass mark': "This is what the rule requires. An error ratio is a critic's "
                                                 'typical error divided by that of the critic trained on the full '
                                                 'log, so 2× means twice as wrong.',
                                    'Result': 'This says whether the rule passed, with the pooled value over seeds '
                                              '0–4, its 95% interval and the range of the five seeds.'}},
                          {'cols': {'Spread σ (standard deviation, points)': 'Each row is a possible value of σ '
                                                                             '(sigma), in points: the larger '
                                                                             'standard deviation, across repeat runs '
                                                                             '(seeds), of the two differences that '
                                                                             'the seed-count check would have '
                                                                             'measured, each between two runs of the '
                                                                             'same seed that differ in one thing. '
                                                                             'The seed-count check, the fourth harm '
                                                                             'check, sets how many repeat runs (N) a '
                                                                             'fair test needs; it never ran, so '
                                                                             'these rows only show what N each σ '
                                                                             'would require.',
                                    'BCA-test seeds N': 'This is the number of repeat runs (seeds) N that a fair '
                                                        'test of Bayesian Conformal Aggregation (BCA) needs: N = '
                                                        'max(10, N₀ rounded up), with N₀ = ((2.128 + 0.842) × 1.25σ '
                                                        '/ 3)². Above 20 the test stops as underpowered, which means '
                                                        'it would have too little chance of detecting a real '
                                                        'effect.'}},
                          {'cols': {'If': 'This is the failure the rule watches for, fixed in the pre-registration '
                                          'before any result existed.',
                                    'Then': 'This is what the test does when that failure happens.',
                                    'Why': 'This explains why that failure ends the test, or allows the one fixed '
                                           'rerun.'}}],
          'rewards': [{'group': 'World',
                       'cols': {"Thruster 1's throttle": "This is thruster 1's throttle, upstream, in the example "
                                                         "state; thrusters 2 to 4 give the stall-aware controller's "
                                                         'command.',
                                'Its push': 'This is the push thruster 1 really gives at that throttle in this '
                                            'world.',
                                'Its throttle cost': "This is thruster 1's part of the throttle cost, 0.3 times its "
                                                     'squared throttle, the same in both worlds.',
                                'Reward this step': 'This is the reward of the first step: minus the float cost of '
                                                    'the example state and the throttle cost of the whole command.',
                                'Float cost of the next state': 'This is the float cost one step later, which shows '
                                                                'how far the floats drift after the command.',
                                'True value of the command': 'This is the discounted sum of all the rewards that '
                                                             'follow the command, computed exactly, with the '
                                                             'stall-aware controller driving after the first step.',
                                'True value against 80%': 'This is the true value minus the true value with thruster '
                                                          '1 at 80% in the same world; negative means worse than '
                                                          '80%.'}}]},
 'superseded': {'TIPS': ['tip-016', 'tip-017', 'tip-043'],
                'LEGENDS': ['tip-104',
                            'tip-113',
                            'tip-147',
                            'tip-154',
                            'tip-163',
                            'tip-170',
                            'tip-179',
                            'tip-191',
                            'tip-193',
                            'tip-197',
                            'tip-208',
                            'tip-211',
                            'tip-217'],
                'PASS3_FINAL': ['methods']}}

# ------------------------------------------------------------------------------------- pass 4, edit 4
# Page pass 4, edit 4 (Results, the weighting test's exploratory results, the critic section's column
# descriptions): what the page now has, read from the page after edit 4 (p4_e4_sources.py wrote it; ta_apply.py
# does not use this record). tips: every data-tip text the edit-3 page did not have, by section, title -> text.
# cols: the column descriptions of every section whose descriptions edit 4 changed. superseded: the older entries
# above whose text edit 4 took off the page (TIPS ids; PASS3_FINAL sections; PASS4_EDITS_1_3 tips and sections).
# Edit 4 changed no symbol legend and nothing in the shared tooltip script (ta_code.py holds no edit-4 change).
PASS4_EDIT_4 = {'tips': {'results': {'Dynamics extrapolation': 'A dynamics extrapolation is a test in which the world itself '
                                                'behaves beyond anything in the training data, so a learner must '
                                                'extrapolate how it responds to commands. Here the current reaches '
                                                '1.1, above the recorded 0.7.',
                      'Disturbance shift': 'A disturbance shift is a test in which the outside pushes on the system '
                                           'are larger than any in the training data. Here the waves reach an '
                                           'amplitude of 0.7, above the recorded 0.4.',
                      'Initial-state shift': 'An initial-state shift is a test that starts from states farther out '
                                             'than any recorded start, while the world itself behaves as recorded. '
                                             'Here the floats start three times as far from rest.'}},
 'cols': {'results': [{'cols': {'What the row asks': 'Each row is one comparison between critics, and its first cell '
                                                     'asks the question that the comparison answers. The first five '
                                                     "rows compare the size of the critic's error, and the last two "
                                                     'compare its direction.',
                                'Seeds 0–4': 'This gives the lowest and the highest value over the five repeat runs '
                                             '(seeds 0–4). In the first five rows the value is an error ratio (×): a '
                                             "critic's typical error divided by that of the critic trained on the "
                                             "full log. In the last two rows it is the critic's median extra "
                                             'preference for full throttle, in units of return.',
                                'Rule': 'This is the pass rule fixed in the pre-registration, the plan written down '
                                        'before any result.',
                                'Result': 'This says whether the rule passed.'}},
                      {'cols': {'Repeat run': 'Each row is one repeat run of the harm checks, named by its seed; a '
                                              'seed fixes every random draw of a run.',
                                'Critic weight 2.5: actor of the full-log critic': 'This gives the points in strong '
                                                                                   'current of the actor trained '
                                                                                   'against the full-log critic (the '
                                                                                   'critic trained on the full log, '
                                                                                   'with nothing removed), at the '
                                                                                   'published critic weight 2.5. On '
                                                                                   'this scale 0 is the recording '
                                                                                   'controller and 100 is the '
                                                                                   'stall-aware controller.',
                                'Critic weight 2.5: actor of the critic trained after the removal': 'This gives the '
                                                                                                    'points in '
                                                                                                    'strong current '
                                                                                                    'of the actor '
                                                                                                    'trained against '
                                                                                                    'the critic '
                                                                                                    'trained after '
                                                                                                    'the removal, at '
                                                                                                    'the published '
                                                                                                    'critic weight '
                                                                                                    '2.5. On this '
                                                                                                    'scale 0 is the '
                                                                                                    'recording '
                                                                                                    'controller and '
                                                                                                    '100 is the '
                                                                                                    'stall-aware '
                                                                                                    'controller.',
                                'Critic weight 5: actor of the full-log critic': 'This gives the points in strong '
                                                                                 'current of the actor trained '
                                                                                 'against the full-log critic, at '
                                                                                 'critic weight 5, the weight at '
                                                                                 'which the check passed. On this '
                                                                                 'scale 0 is the recording '
                                                                                 'controller and 100 is the '
                                                                                 'stall-aware controller.',
                                'Critic weight 5: actor of the critic trained after the removal': 'This gives the '
                                                                                                  'points in strong '
                                                                                                  'current of the '
                                                                                                  'actor trained '
                                                                                                  'against the '
                                                                                                  'critic trained '
                                                                                                  'after the '
                                                                                                  'removal, at '
                                                                                                  'critic weight 5. '
                                                                                                  'On this scale 0 '
                                                                                                  'is the recording '
                                                                                                  'controller and '
                                                                                                  '100 is the '
                                                                                                  'stall-aware '
                                                                                                  'controller.',
                                'Critic weight 5: loss': 'This is the number of points lost by trusting the critic '
                                                         'trained after the removal: the points of the actor of the '
                                                         'full-log critic minus those of the actor of the critic '
                                                         'trained after the removal, at critic weight 5.'}},
                      {'cols': {'Variant': 'Each row is one way of placing extra pull toward the recorded command on '
                                           'the training rows of the best-case caution check.',
                                'What it is': 'This says how the variant sets the dose, the number that multiplies '
                                              'the pull toward the recorded command on each training row; a dose of '
                                              '1 adds nothing.',
                                'Why it is here': 'This says what the variant is compared for, or what it rules '
                                                  'out.'}},
                      {'cols': {'Variant': 'This names the variant, the way the extra pull toward the recorded '
                                           'command is placed on the training rows; the table above describes each '
                                           'one.',
                                'Recorded conditions': 'This gives the points, as a mean over repeat runs (seeds) '
                                                       '0–4, in the recorded conditions (current 0.1 to 0.7, wave '
                                                       'amplitude 0.1 to 0.4), the conditions of the training data. '
                                                       'On this scale 0 is the recording controller and 100 is the '
                                                       'stall-aware controller.',
                                'Strong current': 'This gives the points, as a mean over repeat runs (seeds) 0–4, in '
                                                  'strong current (current 0.7 to 1.1, stronger than any in the '
                                                  'data). On this scale 0 is the recording controller and 100 is the '
                                                  'stall-aware controller.',
                                'Big waves': 'This gives the points, as a mean over repeat runs (seeds) 0–4, in big '
                                             'waves (wave amplitude 0.4 to 0.7, bigger than any in the data). On '
                                             'this scale 0 is the recording controller and 100 is the stall-aware '
                                             'controller.',
                                'Big starting pushes': 'This gives the points, as a mean over repeat runs (seeds) '
                                                       '0–4, with big starting pushes (the floats start three times '
                                                       'as far from rest). On this scale 0 is the recording '
                                                       'controller and 100 is the stall-aware controller.'}},
                      {'cols': {'Situation': 'Each row is one situation, tested on its 1,000 fixed starting states '
                                             '(its evaluation bank). The recorded conditions are reported but are '
                                             'not part of the rule, which looks only at the shifted situations.',
                                'Over strength-matched uniform': 'This is the points of harm-placed caution minus '
                                                                 'those of uniform caution of the same strength, as '
                                                                 'a mean over repeat runs (seeds) 0–4. The rule '
                                                                 'needs at least +3.',
                                'Over amount-matched': 'This is the points of harm-placed caution minus those of '
                                                       'amount-matched caution, which spreads the same total caution '
                                                       'evenly. The rule needs at least +3.',
                                'Over shuffled': 'This is the points of harm-placed caution minus those of the '
                                                 'shuffled doses, which are the same doses in a random row order. '
                                                 'The rule needs at least +3.',
                                'Result': 'This says whether harm-placed caution reached the pass mark of +3 points '
                                          'against all three controls in this situation.'}},
                      {'cols': {'Level and situation': 'Each row is one test situation of the main test, as a '
                                                       'difficulty level: the levels are ordered by the harm the '
                                                       'missing full-throttle commands cause, not by raw difficulty.',
                                'Standard term': 'This is the usual reinforcement-learning name for how the '
                                                 'situation differs from the training data.',
                                'What differs from the recording': 'This says what changes from the conditions of '
                                                                   'the recorded training data.',
                                'States outside the recorded conditions': 'This is the share of the visited states '
                                                                          'that lie outside the recorded conditions: '
                                                                          'a current above 0.7, a wave amplitude '
                                                                          'above 0.4, or floats farther from rest '
                                                                          'than 99% of the recorded training states.',
                                'States where the stall-aware controller needs a removed command': 'This is the '
                                                                                                   'share of the '
                                                                                                   'states the '
                                                                                                   'stall-aware '
                                                                                                   'controller '
                                                                                                   'visits at which '
                                                                                                   'its own command '
                                                                                                   'lies inside the '
                                                                                                   "main test's "
                                                                                                   'removed ball, so '
                                                                                                   'the training '
                                                                                                   'data no longer '
                                                                                                   'show that '
                                                                                                   'command (the '
                                                                                                   'shift table, '
                                                                                                   'design seed '
                                                                                                   '900).',
                                'Harm from the missing commands, points': "This is the return-cost check's loss at "
                                                                          'critic weight 5: the actor trained '
                                                                          'against the critic of the full log minus '
                                                                          'the actor trained against the critic '
                                                                          'after the removal, in points, the mean '
                                                                          'over seeds 0 to 4.',
                                'Actor of the full-log critic, points': 'This is the score of the actor trained '
                                                                        'against the critic of the full log, which '
                                                                        'shows how hard the situation is for a '
                                                                        'learner when no command is missing; on this '
                                                                        'scale 0 is the recording controller and 100 '
                                                                        'the stall-aware controller.'}},
                      {'cols': {'Check': 'Each row is one of the nine world checks, run before any learning on '
                                         'design seeds 900 to 904, asked as the question it answers.',
                                'Pass mark': 'This is the pass mark fixed in the pre-registration, the plan written '
                                             'down before any learner ran.',
                                'Seed 900': 'This is the value on design seed 900, the recording this page shows. A '
                                            'seed fixes every random draw of a recording, and the design seeds are '
                                            'recordings made only to design and check the world.',
                                'All five': 'This gives the lowest and the highest value over the five design seeds, '
                                            '900 to 904.',
                                'Passed': 'This says whether the check passed, and whether it passed on every '
                                          'seed.'}},
                      {'group': 'Kind of arm',
                       'cols': {'Arm': 'Each row is one arm, one way of training the actor; every arm trains against '
                                       'the same frozen critic of the log after the removal and starts from the same '
                                       'cloned actor (the first row). The heading rows sort the arms into groups.',
                                'Strong current: share of states past 80%': 'This is the share of the states visited '
                                                                            'in strong current at which some '
                                                                            'thruster is past 80%, as a mean over '
                                                                            'repeat runs (seeds) 0–4 with the range '
                                                                            'from the lowest to the highest seed '
                                                                            'after it (in small type below the mean '
                                                                            'in the table).',
                                'Recorded conditions': 'This gives the points, as a mean over repeat runs (seeds) '
                                                       '0–4 with the range from the lowest to the highest seed after '
                                                       'it (in small type below the mean in the table), in the '
                                                       'recorded conditions (current 0.1 to 0.7, wave amplitude 0.1 '
                                                       'to 0.4), the conditions of the training data. On this scale '
                                                       '0 is the recording controller and 100 is the stall-aware '
                                                       'controller.',
                                'Strong current': 'This gives the points, as a mean over repeat runs (seeds) 0–4 '
                                                  'with the range from the lowest to the highest seed after it (in '
                                                  'small type below the mean in the table), in strong current '
                                                  '(current 0.7 to 1.1, stronger than any in the data). On this '
                                                  'scale 0 is the recording controller and 100 is the stall-aware '
                                                  'controller.',
                                'Big waves': 'This gives the points, as a mean over repeat runs (seeds) 0–4 with the '
                                             'range from the lowest to the highest seed after it (in small type '
                                             'below the mean in the table), in big waves (wave amplitude 0.4 to 0.7, '
                                             'bigger than any in the data). On this scale 0 is the recording '
                                             'controller and 100 is the stall-aware controller.',
                                'Big starting pushes': 'This gives the points, as a mean over repeat runs (seeds) '
                                                       '0–4 with the range from the lowest to the highest seed after '
                                                       'it (in small type below the mean in the table), with big '
                                                       'starting pushes (the floats start three times as far from '
                                                       'rest). On this scale 0 is the recording controller and 100 '
                                                       'is the stall-aware controller.'}},
                      {'cols': {'Arm': "Each row is one arm, a variant of the actor's training, here trained against "
                                       "the no-stall comparison world's own critic of the log after the removal.",
                                'Recorded conditions': 'This is the cost of caution: the points with no extra '
                                                       'caution minus the points of the arm, so a higher number '
                                                       'costs more, as a mean over repeat runs (seeds) 0–4 with the '
                                                       'range from the lowest to the highest seed after it (in small '
                                                       'type below the mean in the table), in the recorded '
                                                       'conditions (current 0.1 to 0.7, wave amplitude 0.1 to 0.4), '
                                                       'the conditions of the training data. In this world 100 '
                                                       'points is the rising-thrust controller.',
                                'Strong current': 'This is the cost of caution: the points with no extra caution '
                                                  'minus the points of the arm, so a higher number costs more, as a '
                                                  'mean over repeat runs (seeds) 0–4 with the range from the lowest '
                                                  'to the highest seed after it (in small type below the mean in the '
                                                  'table), in strong current (current 0.7 to 1.1, stronger than any '
                                                  'in the data). In this world 100 points is the rising-thrust '
                                                  'controller.',
                                'Big waves': 'This is the cost of caution: the points with no extra caution minus '
                                             'the points of the arm, so a higher number costs more, as a mean over '
                                             'repeat runs (seeds) 0–4 with the range from the lowest to the highest '
                                             'seed after it (in small type below the mean in the table), in big '
                                             'waves (wave amplitude 0.4 to 0.7, bigger than any in the data). In '
                                             'this world 100 points is the rising-thrust controller.',
                                'Big starting pushes': 'This is the cost of caution: the points with no extra '
                                                       'caution minus the points of the arm, so a higher number '
                                                       'costs more, as a mean over repeat runs (seeds) 0–4 with the '
                                                       'range from the lowest to the highest seed after it (in small '
                                                       'type below the mean in the table), with big starting pushes '
                                                       '(the floats start three times as far from rest). In this '
                                                       'world 100 points is the rising-thrust controller.'}},
                      {'cols': {'Arm': "Each row is one arm, a variant of the actor's training against the critic of "
                                       'the best-case caution check, which is the frozen critic of a run of plain '
                                       'twin delayed deep deterministic policy gradient plus behaviour cloning '
                                       '(TD3+BC) on the log after the removal.',
                                'Recorded conditions': 'This gives the points, as a mean over repeat runs (seeds) '
                                                       '0–4 with the range from the lowest to the highest seed after '
                                                       'it (in small type below the mean in the table), in the '
                                                       'recorded conditions (current 0.1 to 0.7, wave amplitude 0.1 '
                                                       'to 0.4), the conditions of the training data. On this scale '
                                                       '0 is the recording controller and 100 is the stall-aware '
                                                       'controller.',
                                'Strong current': 'This gives the points, as a mean over repeat runs (seeds) 0–4 '
                                                  'with the range from the lowest to the highest seed after it (in '
                                                  'small type below the mean in the table), in strong current '
                                                  '(current 0.7 to 1.1, stronger than any in the data). On this '
                                                  'scale 0 is the recording controller and 100 is the stall-aware '
                                                  'controller.',
                                'Big waves': 'This gives the points, as a mean over repeat runs (seeds) 0–4 with the '
                                             'range from the lowest to the highest seed after it (in small type '
                                             'below the mean in the table), in big waves (wave amplitude 0.4 to 0.7, '
                                             'bigger than any in the data). On this scale 0 is the recording '
                                             'controller and 100 is the stall-aware controller.',
                                'Big starting pushes': 'This gives the points, as a mean over repeat runs (seeds) '
                                                       '0–4 with the range from the lowest to the highest seed after '
                                                       'it (in small type below the mean in the table), with big '
                                                       'starting pushes (the floats start three times as far from '
                                                       'rest). On this scale 0 is the recording controller and 100 '
                                                       'is the stall-aware controller.'}},
                      {'cols': {'Mark': 'Each row is one pass mark (a pass-or-fail rule) of Stage 1, the first '
                                        'exploratory stage after the harm checks stopped the test, and every mark '
                                        'was fixed before any return was computed.',
                                'Rule, fixed before any return was computed (pooled over seeds 0–4 unless the rule names seeds)': 'This '
                                                                                                                                  'is '
                                                                                                                                  'the '
                                                                                                                                  "mark's "
                                                                                                                                  'rule. '
                                                                                                                                  'Pooled '
                                                                                                                                  'means '
                                                                                                                                  'combined '
                                                                                                                                  'over '
                                                                                                                                  'repeat '
                                                                                                                                  'runs '
                                                                                                                                  '(seeds) '
                                                                                                                                  '0–4, '
                                                                                                                                  'unless '
                                                                                                                                  'the '
                                                                                                                                  'rule '
                                                                                                                                  'names '
                                                                                                                                  'particular '
                                                                                                                                  'seeds.',
                                'Result': 'This is what was measured.',
                                'Verdict': 'This says whether the mark passed, or, for the futility look (an early '
                                           'check that could have stopped the test), whether to continue.'}},
                      {'cols': {'Rule': 'This says what happens at each state the actor visits, with no training; '
                                        'the actor is the one trained against the critic of the log after the '
                                        'removal.',
                                'Recorded conditions': 'This is the change in points from that actor alone, in the '
                                                       'stalling world, as a mean over repeat runs (seeds) 0–4 on '
                                                       "the first 200 of the situation's 1,000 fixed starting "
                                                       'states, in the recorded conditions (current 0.1 to 0.7, wave '
                                                       'amplitude 0.1 to 0.4), the conditions of the training data.',
                                'Strong current': 'This is the change in points from that actor alone, in the '
                                                  'stalling world, as a mean over repeat runs (seeds) 0–4 on the '
                                                  "first 200 of the situation's 1,000 fixed starting states, in "
                                                  'strong current (current 0.7 to 1.1, stronger than any in the '
                                                  'data).',
                                'Big waves': 'This is the change in points from that actor alone, in the stalling '
                                             'world, as a mean over repeat runs (seeds) 0–4 on the first 200 of the '
                                             "situation's 1,000 fixed starting states, in big waves (wave amplitude "
                                             '0.4 to 0.7, bigger than any in the data).',
                                'Big starting pushes': 'This is the change in points from that actor alone, in the '
                                                       'stalling world, as a mean over repeat runs (seeds) 0–4 on '
                                                       "the first 200 of the situation's 1,000 fixed starting "
                                                       'states, with big starting pushes (the floats start three '
                                                       'times as far from rest).'}}],
          'weighting-test': [{'cols': {'Quality': 'Each row is one data quality: the same environment and shift, '
                                                  'recorded at one level of skill and named as in the D4RL '
                                                  'collection of recorded datasets.',
                                       'How its recorded commands are made': 'This says how often the recording '
                                                                             "controller's command is replaced by a "
                                                                             'uniform random command in this '
                                                                             'quality.',
                                       'The data': 'This gives the points on the strong test of the controller that '
                                                   "made the recording, with this quality's random share. On this "
                                                   "test's scale 0 is uniform random commands and 100 is the "
                                                   'stall-aware controller.',
                                       "TD3+BC's actor": 'This gives the points on the strong test of the actor that '
                                                         'TD3+BC learned from the log after the removal, as a mean '
                                                         "over seeds 0–4. On this test's scale 0 is uniform random "
                                                         'commands and 100 is the stall-aware controller.',
                                       'What the removal costs': 'This is the points of the actor learned from the '
                                                                 'full log minus those of the actor learned after '
                                                                 'the removal, on the strong test: how much the '
                                                                 'missing hold costs the learner. On the random '
                                                                 'quality the two logs are almost identical, so its '
                                                                 'value there is run-to-run noise.'}},
                             {'cols': {'Quality': 'Each row is one data quality of the weighting test, from random '
                                                  'to expert.',
                                       'No shift': 'These three columns are measured on held-out steps of the '
                                                   'training distribution.',
                                       'Strong test, other steps': 'These three columns are measured on the strong '
                                                                   "test's steps whose command is not the removed "
                                                                   '80% hold.',
                                       'Strong test, steps that use the removed 80% hold': 'These three columns are '
                                                                                           'measured on the strong '
                                                                                           "test's steps whose "
                                                                                           'command is the removed '
                                                                                           '80% hold.',
                                       'No shift: plain': 'This is the coverage of plain twin delayed deep '
                                                          'deterministic policy gradient plus behaviour cloning '
                                                          "(TD3+BC)'s band, the lower critic's value plus or minus "
                                                          "the two critics' disagreement, on held-out steps of the "
                                                          'training distribution.',
                                       'No shift: BCA': 'This is the coverage of the unweighted band of Bayesian '
                                                        'Conformal Aggregation (BCA) on the same steps.',
                                       'No shift: weighted': 'This is the coverage of the band of Bayesian Conformal '
                                                             'Aggregation with exact weights on the same steps; with '
                                                             'no shift every weight is 1, so it equals the '
                                                             'unweighted band.',
                                       'Strong test, other steps: plain': "This is the coverage of plain TD3+BC's "
                                                                          "two-critic band on the strong test's "
                                                                          'steps that do not use the removed 80% '
                                                                          'hold.',
                                       'Strong test, other steps: BCA': 'This is the coverage of the unweighted band '
                                                                        'of Bayesian Conformal Aggregation (BCA) on '
                                                                        'those steps.',
                                       'Strong test, other steps: weighted': 'This is the coverage of the band of '
                                                                             'Bayesian Conformal Aggregation with '
                                                                             'exact weights, each step at its own '
                                                                             'threshold, on those steps.',
                                       'Strong test, removed hold: plain': "This is the coverage of plain TD3+BC's "
                                                                           "two-critic band on the strong test's "
                                                                           'steps that use the removed 80% hold.',
                                       'Strong test, removed hold: BCA': 'This is the coverage of the unweighted '
                                                                         'band of Bayesian Conformal Aggregation '
                                                                         '(BCA) on those steps.',
                                       'Strong test, removed hold: weighted': 'The weighted band refuses every one '
                                                                              'of these steps, because the removed '
                                                                              'hold has no chance in training, so '
                                                                              'its weight cannot be computed.'}},
                             {'cols': {'Quality': 'Each row is one data quality of the weighting test, from random '
                                                  'to expert.',
                                       'Capped pull: TD3+BC retrained as a whole': 'These three columns retrain the '
                                                                                   'whole learner, critic and actor, '
                                                                                   'for 100,000 updates.',
                                       'Lower edge: actor only, against the frozen critic': 'These four columns '
                                                                                            'train only the actor, '
                                                                                            'against the same frozen '
                                                                                            'critic.',
                                       'Plain TD3+BC': 'This is plain twin delayed deep deterministic policy '
                                                       'gradient plus behaviour cloning (TD3+BC) trained on the data '
                                                       'with the 80% hold removed, the reference for the capped '
                                                       'pull, in points on the strong test.',
                                       'Capped pull, unweighted': 'This is TD3+BC retrained with the dose of '
                                                                  'Bayesian Conformal Aggregation (BCA) on its pull '
                                                                  'toward the recorded commands, the band '
                                                                  'unweighted.',
                                       'Capped pull, weighted': 'This is the same run with the band weighted exactly '
                                                                'for the strong test, each training row at its own '
                                                                'threshold.',
                                       'Actor only, no extra caution': 'This is the actor trained alone against the '
                                                                       'frozen critic with no extra caution, the '
                                                                       'reference for the lower edge.',
                                       'Lower edge, unweighted': 'This is the same actor-only run with the '
                                                                 'lower-edge penalty of Bayesian Conformal '
                                                                 "Aggregation (BCA) on the value of the actor's own "
                                                                 'command, the band unweighted.',
                                       'Lower edge, weighted': 'This is the same run with the band weighted exactly '
                                                               'for the strong test.',
                                       'Perfect lower edge, for scale': 'This is the same actor-only run with a '
                                                                        "perfect penalty, built from the critic's "
                                                                        'exact one-step error; no method that learns '
                                                                        'from data has it.'}},
                             {'cols': {'Test setting': 'Each row is a test distribution of the current: no shift, '
                                                       'mild or strong.',
                                       'ξ': "The lean ξ (xi) says how far the current's distribution leans toward "
                                            'strong currents. Its density is proportional to e to the power ξc, over '
                                            '0.1 to 1.1. Training uses ξ = −2.',
                                       'Mean current': "This is the average current c under this setting's "
                                                       'distribution.',
                                       'Exact weight ω(c)': 'The weight ω(c) (omega) of a recorded step with current '
                                                            'c is the test density divided by the training density, '
                                                            'written out exactly.',
                                       'Usable calibration rows after the removal': 'This is how many equally '
                                                                                    'weighted rows the weighted '
                                                                                    'calibration set is worth once '
                                                                                    'the removal has deleted rows, '
                                                                                    'as measured by a trial '
                                                                                    'simulation made while this test '
                                                                                    'was designed.'}},
                             {'group': 'Part',
                              'cols': {'Check': 'Each row is one pass mark of the weighting test, judged for each '
                                                'data quality on its own, and the heading rows sort the marks into '
                                                'groups.',
                                       'Passes when': 'This is the rule, fixed in the fifth version of the weighting '
                                                      "test's specification before any run.",
                                       'Why this line': 'This says why the mark sits where it does.',
                                       'Result on the five qualities': 'This says on which data qualities the mark '
                                                                       'passed or failed, with the measured numbers; '
                                                                       'a mark that was not allowed to run says '
                                                                       'so.'}}],
          'critic-in-action': [{'cols': {'Thruster 1, upstream': 'Each row is one of the 41 throttles that thruster '
                                                                 '1 is swept through in the state on screen, from 0% '
                                                                 'to 100% upstream in steps of 2.5%, with thrusters '
                                                                 '2 to 4 kept where the actor put them. One more '
                                                                 "row, highlighted, holds the actor's own command.",
                                         'Critic, Q(s, a)': "This is the critic's value of the row's command: its "
                                                            'estimate of the discounted return of giving that '
                                                            'command in this state and letting the actor drive '
                                                            'afterwards, the smaller of its two estimates. Every '
                                                            'reward is a cost, so every value is negative and higher '
                                                            'is better.',
                                         'True value': "This is the true value of the row's command, computed "
                                                       'exactly by simulation: one step with the command, then the '
                                                       'actor drives for 600 steps, with each later reward '
                                                       'discounted by 0.98 per step.',
                                         'Critic minus true value': "This is the critic's value minus the true "
                                                                    'value. A positive number means the critic rates '
                                                                    'the command better than it is, which is the '
                                                                    'kind of error that draws the actor toward the '
                                                                    'command.',
                                         'Note': 'This marks the throttle the critic rates highest (â), the swept '
                                                 'throttle with the highest true value, the 80% stall limit, the '
                                                 "throttles inside this section's removed region, and the row of the "
                                                 "actor's own command."}},
                               {'cols': {'State': 'Each row is one state of the episode, numbered 0 to 99. The '
                                                  'highlighted row is the state on screen.',
                                         "Water's push c + w₁": "This is the water's push in the state: the current "
                                                                'c plus the push w₁ that the wave adds now. A '
                                                                'positive push points downstream.',
                                         'Thruster 1 (actor), upstream': 'This is the throttle the actor gives '
                                                                         'thruster 1 in the state, upstream unless '
                                                                         'marked down. Past 80% the propeller stalls '
                                                                         'and pushes less.',
                                         'Reward': "This is the state's reward: minus the sum of the squares of the "
                                                   "floats' ten numbers, minus 0.3 times the sum of the squared "
                                                   'throttles. Every reward is a cost, so closer to 0 is better.',
                                         'Critic, Q(s, π(s))': "This is the critic's value of the command the actor "
                                                               'gave in the state, the smaller of its two estimates.',
                                         'True value': 'This is the true value of the command the actor gave, '
                                                       'computed exactly by simulation with the actor driving '
                                                       'afterwards.',
                                         'Critic minus true value': "This is the critic's value minus the true value "
                                                                    "at the actor's command. A positive number means "
                                                                    'the critic rates the command better than it is.',
                                         'â': 'This is the swept throttle of thruster 1 that the critic rates '
                                              'highest in the state. The actor, which is trained to raise the '
                                              "critic's value, is drawn toward it.",
                                         'True best of the sweep': 'This is the swept throttle of thruster 1 with '
                                                                   'the highest true value in the state, with the '
                                                                   'other three thrusters kept where the actor put '
                                                                   'them.'}},
                               {'cols': {'Dataset quality': 'Each row is one of the five dataset qualities of the '
                                                            'weighting test, from random to expert.',
                                         'How it was recorded': "This says how the quality's commands were recorded, "
                                                                'and what share of its training steps carry a random '
                                                                "command instead of the recording controller's.",
                                         'Training steps kept, all data / hold removed': 'This is the number of '
                                                                                         'recorded training steps '
                                                                                         'each critic trained on: '
                                                                                         'all of them, and those '
                                                                                         'left after the densest-15% '
                                                                                         'removal, which in the '
                                                                                         'weighting test takes the '
                                                                                         '80% hold.',
                                         'Thruster 1 held at exactly 80%, all data / hold removed': 'This is the '
                                                                                                    'share of those '
                                                                                                    'training steps '
                                                                                                    'in which '
                                                                                                    'thruster 1 is '
                                                                                                    'held at exactly '
                                                                                                    '80% upstream, '
                                                                                                    'the best move '
                                                                                                    'in hard water, '
                                                                                                    'in each '
                                                                                                    "critic's data.",
                                         'Thruster 1 past 80%, all data / hold removed': 'This is the share of those '
                                                                                         'training steps in which '
                                                                                         'thruster 1 is more than '
                                                                                         '80% upstream, in each '
                                                                                         "critic's data. Only random "
                                                                                         'commands go there, so they '
                                                                                         'are the only evidence of '
                                                                                         'the stall the critic ever '
                                                                                         'sees.'}},
                               {'cols': {'Dataset quality': 'Each row is one dataset quality. The highlighted row is '
                                                            'the quality chosen above.',
                                         'Critic rates 100% above 80%': 'These two columns give the share of the '
                                                                        'states at which the critic rates thruster 1 '
                                                                        'at 100% upstream above 80%, with the other '
                                                                        'three thrusters as the actor set them.',
                                         'Critic rates 100% above 80%, all the data': 'This is that share for the '
                                                                                      'critic trained on all the '
                                                                                      'data, over the 1,000 states '
                                                                                      'its own actor visits from the '
                                                                                      'first ten starts of the '
                                                                                      'situation chosen above.',
                                         'Critic rates 100% above 80%, 80% hold removed': 'This is that share for '
                                                                                          'the critic trained with '
                                                                                          'the 80% hold removed, '
                                                                                          'over the states its own '
                                                                                          'actor visits.',
                                         'Critic rates 100% higher, true value lower': 'These two columns give the '
                                                                                       'share of the states at which '
                                                                                       'the critic rates 100% above '
                                                                                       '80% while the true value of '
                                                                                       '100% is below that of 80%: '
                                                                                       'the critic prefers the '
                                                                                       'stall, and the truth does '
                                                                                       'not.',
                                         'Critic rates 100% higher, true value lower, all the data': 'This is that '
                                                                                                     'share for the '
                                                                                                     'critic trained '
                                                                                                     'on all the '
                                                                                                     'data.',
                                         'Critic rates 100% higher, true value lower, 80% hold removed': 'This is '
                                                                                                         'that share '
                                                                                                         'for the '
                                                                                                         'critic '
                                                                                                         'trained '
                                                                                                         'with the '
                                                                                                         '80% hold '
                                                                                                         'removed.',
                                         'Actor gives thruster 1 more than 80%': 'These two columns give the share '
                                                                                 'of the states at which the actor '
                                                                                 'itself gives thruster 1 more than '
                                                                                 '80% upstream, into the stall.',
                                         'Actor gives thruster 1 more than 80%, all the data': 'This is that share '
                                                                                               'for the actor '
                                                                                               'trained with the '
                                                                                               'critic on all the '
                                                                                               'data.',
                                         'Actor gives thruster 1 more than 80%, 80% hold removed': 'This is that '
                                                                                                   'share for the '
                                                                                                   'actor trained '
                                                                                                   'with the 80% '
                                                                                                   'hold '
                                                                                                   'removed.'}}]},
 'superseded': {'TIPS': [],
                'PASS3_FINAL': ['results', 'weighting-test'],
                'PASS4_EDITS_1_3': {'tips': [], 'cols': []}}}

# ------------------------------------------------------------------------------------- pass 4, edit 5
# Page pass 4, edit 5 (the final review's findings): what the page now has, read from the page after edit 5
# (p4_e5_sources.py wrote it; ta_apply.py does not use this record). tips: every data-tip text the edit-4 page did
# not have, by section, title -> text. cols: the column descriptions of every section whose descriptions edit 5
# changed (histogram rows are bins or ranges, not bands; the critics are named full-recording critic and critic
# trained after the removal). legends: every legend definition edit 5 rewrote as a complete sentence, old -> new.
PASS4_EDIT_5 = {'tips': {},
 'cols': {'task': [{'cols': {"Water's push c + w₁": "Each row is one bin of the water's push at a state, the current "
                                                    'c plus the push w₁ that the wave adds now. Each bin is 0.2 '
                                                    'wide, and a positive push points downstream, a negative one '
                                                    'upstream.',
                             'States': "This column counts the states whose water's push falls in this bin, over "
                                       'every state of the four evaluation banks in the stalling world. The water '
                                       'does not depend on the throttles, so both controllers meet the same count.',
                             'Stall-aware: thruster 1': "This is the stall-aware controller's throttle on thruster "
                                                        "1, averaged over the bin's states, and a negative throttle "
                                                        'pushes upstream, against the water. It never goes past 80%, '
                                                        'so its push equals its throttle.',
                             'Stall-aware: at its 80% limit': "This is the share of the bin's states in which the "
                                                              'stall-aware controller holds thruster 1 at exactly '
                                                              'its 80% limit.',
                             'Rising-thrust: thruster 1 throttle': "This is the rising-thrust controller's throttle "
                                                                   "on thruster 1, averaged over the bin's states. "
                                                                   'It gives the same textbook command as the '
                                                                   'stall-aware controller but allows up to 100%.',
                             'Rising-thrust: past 80%': "This is the share of the bin's states in which the "
                                                        'rising-thrust controller has thruster 1 past 80%, where its '
                                                        'propeller stalls.',
                             "Rising-thrust: thruster 1's push": 'This is the push thruster 1 really gives under the '
                                                                 "rising-thrust controller, averaged over the bin's "
                                                                 'states. Past 80% the push falls below the '
                                                                 'throttle, to 0.65 at 90% and 0.5 at full '
                                                                 'throttle.'}},
                   {'cols': {'No throttle': 'With all thrusters off, the water acts alone.',
                             'Stall-aware': 'The stall-aware controller gives the textbook command capped at 80%, so '
                                            'it never stalls.',
                             'Rising-thrust, stalling world': 'The rising-thrust controller (the same textbook '
                                                              'command, up to 100%) drives here in the stalling '
                                                              'world, where a propeller past 80% stalls.',
                             'Rising-thrust, no-stall world': 'The rising-thrust controller drives here in the '
                                                              'no-stall comparison world, where push keeps rising up '
                                                              'to 100% throttle.',
                             'Current c': 'This is the strength of the steady current, with no wave, held until the '
                                          'floats settle (their numbers no longer change). The recorded conditions '
                                          'draw it from 0.1 to 0.7, and strong current from 0.7 to 1.1.',
                             'No throttle: float cost': 'This is the float cost in each settled state with all '
                                                        "thrusters off: the five floats' squared distances from "
                                                        'rest, added up.',
                             'Stall-aware: thruster 1': "This is the stall-aware controller's settled throttle on "
                                                        'thruster 1. A negative throttle pushes upstream, and it '
                                                        'stops at −0.80, its 80% limit.',
                             'Stall-aware: float cost': 'This is the float cost in each settled state under the '
                                                        'stall-aware controller.',
                             'Rising-thrust, stalling world: thruster 1 throttle': 'This is the rising-thrust '
                                                                                   "controller's settled throttle on "
                                                                                   'thruster 1 in the stalling '
                                                                                   'world. Above a current of 0.8 it '
                                                                                   'goes to full throttle, −1.00.',
                             "Rising-thrust, stalling world: thruster 1's push": 'This is the push thruster 1 really '
                                                                                 'gives at that throttle in the '
                                                                                 'stalling world: equal to the '
                                                                                 'throttle up to 80%, and only 0.5 '
                                                                                 'at full throttle.',
                             'Rising-thrust, stalling world: float cost': 'This is the float cost in each settled '
                                                                          'state under the rising-thrust controller '
                                                                          'in the stalling world. A † marks currents '
                                                                          'where it never settles, because thruster '
                                                                          '3 keeps crossing into the stall and back; '
                                                                          'there the value is an average over the '
                                                                          'end of a long run.',
                             'Rising-thrust, no-stall world: thruster 1': "This is the rising-thrust controller's "
                                                                          'settled throttle on thruster 1 in the '
                                                                          'no-stall comparison world, where its push '
                                                                          'equals its throttle.',
                             'Rising-thrust, no-stall world: float cost': 'This is the float cost in each settled '
                                                                          'state under the rising-thrust controller '
                                                                          'in the no-stall comparison world.'}}],
          'action-space': [{'cols': {'Action space': "This row gives the action space in Gymnasium's terms: four "
                                                     'numbers, each from −1 to 1, stored as float32.',
                                     'Order': 'This row says which throttle is which: thrusters 1 to 4, numbered 0 '
                                              'to 3 in the code.',
                                     'Sign': 'This row says what the sign of a throttle means: negative pushes '
                                             'upstream, against the water.',
                                     'Push': 'This row says how hard a throttle really pushes: as hard as the '
                                             'throttle up to 80%, and less past it in the stalling world.',
                                     'Throttle cost': 'This row gives what a throttle costs in the reward, which is '
                                                      'charged even while a thruster stalls.',
                                     'Storage': 'This row says how a command is stored and limited before it acts.',
                                     "The main test's data": "This row says where the main test's recorded commands "
                                                             'lie: only the full-throttle command goes past 80%.',
                                     "After the main test's removal": 'This row says what is left past 80% once the '
                                                                      "main test's densest-15% removal has deleted "
                                                                      'its rows.',
                                     "The weighting test's data": "This row says where the weighting test's recorded "
                                                                  'commands lie and what its removal deletes, the '
                                                                  '80% hold.'}},
                           {'cols': {'Index': "This is the throttle's place in the action of 4 numbers, counted from "
                                              '0 as the code does.',
                                     'Action': "This says which thruster's throttle the number sets.",
                                     'Control Min': 'The lowest allowed throttle is −1, full throttle upstream, '
                                                    'against the water.',
                                     'Control Max': 'The highest allowed throttle is 1, full throttle downstream. A '
                                                    'command outside −1 to 1 is clipped to that range.',
                                     'Pushes the speeds of': 'This column gives the two floats whose speed numbers '
                                                             "this thruster's push enters, scaled by the push factor "
                                                             '0.3, with their places in the list of 13 observed '
                                                             'numbers.',
                                     'Stalls': 'The propeller stalls past 80% throttle, and only in the stalling '
                                               'world. A stalled propeller pushes less while its throttle cost keeps '
                                               'rising.'}},
                           {'cols': {'Throttle': "This is how far one thruster's throttle is opened, as a share of "
                                                 'full throttle, in either direction.',
                                     'Push, stalling world': 'This is the push the thruster really gives in the '
                                                             'stalling world: equal to the throttle up to 80%, then '
                                                             '0.8 − 1.5 × (throttle − 0.8), down to 0.5 at full '
                                                             'throttle.',
                                     'Push, no-stall comparison world': 'In the no-stall comparison world the push '
                                                                        'always equals the throttle.',
                                     'Stalls': 'This says whether the propeller stalls at this throttle in the '
                                               'stalling world. Only commands above 0.800000011920929 stall. This '
                                               'number is the stored value nearest 0.8, because each throttle is '
                                               'stored with about 7 significant digits, so a command of exactly 80% '
                                               'never stalls.',
                                     'Throttle cost 0.3 a²': "This thruster's throttle cost in one step is 0.3 times "
                                                             'its squared throttle a. It is charged even while the '
                                                             'propeller stalls.'}},
                           {'cols': {'Throttle range': "Each row is one 20%-wide range of a thruster's throttle; a "
                                                       'negative throttle pushes upstream, and the first and last '
                                                       'ranges are past 80%, where a propeller stalls in the '
                                                       'stalling world.',
                                     'Thruster 1': "These two columns count the training rows with thruster 1's "
                                                   "throttle in the row's range.",
                                     'Thruster 1: all rows': 'This is how many of the 50,000 training rows have '
                                                             "thruster 1's throttle in this range, before any "
                                                             'removal, in the data that the buttons above the table '
                                                             'choose.',
                                     'Thruster 1: rows kept': 'This is how many of the 42,500 rows kept after the '
                                                              "removal have thruster 1's throttle in this range: the "
                                                              'rows the learner trains on.',
                                     'Thruster 2': "These two columns count the training rows with thruster 2's "
                                                   "throttle in the row's range.",
                                     'Thruster 2: all rows': 'This is how many of the 50,000 training rows have '
                                                             "thruster 2's throttle in this range, before any "
                                                             'removal, in the data that the buttons above the table '
                                                             'choose.',
                                     'Thruster 2: rows kept': 'This is how many of the 42,500 rows kept after the '
                                                              "removal have thruster 2's throttle in this range: the "
                                                              'rows the learner trains on.',
                                     'Thruster 3': "These two columns count the training rows with thruster 3's "
                                                   "throttle in the row's range.",
                                     'Thruster 3: all rows': 'This is how many of the 50,000 training rows have '
                                                             "thruster 3's throttle in this range, before any "
                                                             'removal, in the data that the buttons above the table '
                                                             'choose.',
                                     'Thruster 3: rows kept': 'This is how many of the 42,500 rows kept after the '
                                                              "removal have thruster 3's throttle in this range: the "
                                                              'rows the learner trains on.',
                                     'Thruster 4': "These two columns count the training rows with thruster 4's "
                                                   "throttle in the row's range.",
                                     'Thruster 4: all rows': 'This is how many of the 50,000 training rows have '
                                                             "thruster 4's throttle in this range, before any "
                                                             'removal, in the data that the buttons above the table '
                                                             'choose.',
                                     'Thruster 4: rows kept': 'This is how many of the 42,500 rows kept after the '
                                                              "removal have thruster 4's throttle in this range: the "
                                                              'rows the learner trains on.'}}],
          'description': [{'cols': {'Action Space': 'A controller sends 4 throttles each step, each from −1 to 1.',
                                    'Observation Space': 'A controller reads the 13 numbers of the state each step, '
                                                         'and each number has bounds.',
                                    'import': ['Import lines',
                                               'These Python lines register ThrusterChain with Gymnasium, the '
                                               'standard Python interface for environments in which controllers act '
                                               'and learn, and then create the environment.'],
                                    'Episode': 'This row says how long one episode lasts, how it ends, and how much '
                                               'later rewards count.',
                                    'Registered ids': 'These are the names that create the environment, one for each '
                                                      'situation in each world.',
                                    'Code': "This row gives the file that holds the environment's code and the "
                                            'version of numpy, the Python library for arrays of numbers, that it '
                                            'uses.'}},
                          {'cols': {'Throttle': "Each row is one size of a thruster's throttle, from off to full "
                                                'throttle.',
                                    'Push, stalling world': 'This is the push the thruster really gives at this '
                                                            'throttle in the stalling world.',
                                    'Push, no-stall comparison world': 'This is the push at this throttle in the '
                                                                       'no-stall comparison world, where it keeps '
                                                                       'rising.',
                                    'Throttle cost': 'This is the throttle cost of one thruster, 0.3 times the '
                                                     'squared throttle, the same in both worlds.',
                                    'True value against 80%, stalling world': 'This is the true value of the command '
                                                                              'at the example state, minus its true '
                                                                              'value with thruster 1 at 80%, in the '
                                                                              'stalling world; negative means worse.',
                                    'True value against 80%, no-stall comparison world': 'This is the same '
                                                                                         'difference of true values '
                                                                                         'in the no-stall comparison '
                                                                                         'world; negative means '
                                                                                         'worse.'}},
                          {'cols': {'Size of the throttle': "Each row is one range of thruster 1's throttle size, "
                                                            'upstream or downstream; 80% itself belongs to the range '
                                                            'below it.',
                                    'Full data: rows': "This is how many of the main test's 50,000 training rows "
                                                       "have thruster 1's throttle in this range.",
                                    'Full data: share': 'This is that count as a share of the 50,000 rows.',
                                    'After the removal: rows': 'This is how many of the 42,500 rows kept after the '
                                                               "densest-15% removal have thruster 1's throttle in "
                                                               'this range.',
                                    'After the removal: share': 'This is that count as a share of the 42,500 kept '
                                                                'rows.',
                                    'Rows deleted': 'This is how many rows in this range the densest-15% removal '
                                                    'deleted.'}}],
          'harm-checks': [{'cols': {'Rule': "This is the rule's letter, a to f, as in the pre-registration, the plan "
                                            'written down before any learner ran.',
                                    'The question': 'This is the question the rule answers, in plain words. Each '
                                                    "rule rules out one other explanation of the critic's error.",
                                    'Pass mark': "This is what the rule requires. An error ratio is a critic's "
                                                 'typical error divided by that of the full-recording critic, so 2× '
                                                 'means twice as wrong.',
                                    'Result': 'This says whether the rule passed, with the pooled value over seeds '
                                              '0–4, its 95% interval and the range of the five seeds.'}},
                          {'cols': {'Spread σ (standard deviation, points)': 'Each row is a possible value of σ '
                                                                             '(sigma), in points: the larger '
                                                                             'standard deviation, across repeat runs '
                                                                             '(seeds), of the two differences that '
                                                                             'the seed-count check would have '
                                                                             'measured, each between two runs of the '
                                                                             'same seed that differ in one thing. '
                                                                             'The seed-count check, the fourth harm '
                                                                             'check, sets how many repeat runs (N) a '
                                                                             'fair test needs; it never ran, so '
                                                                             'these rows only show what N each σ '
                                                                             'would require.',
                                    'BCA-test seeds N': 'This is the number of repeat runs (seeds) N that a fair '
                                                        'test of Bayesian Conformal Aggregation (BCA) needs: N = '
                                                        'max(10, N₀ rounded up), with N₀ = ((2.128 + 0.842) × 1.25σ '
                                                        '/ 3)². Above 20 the test stops as underpowered, which means '
                                                        'it would have too little chance of detecting a real '
                                                        'effect.'}},
                          {'cols': {'If': 'This is the failure the rule watches for, fixed in the pre-registration '
                                          'before any result existed.',
                                    'Then': 'This is what the test does when that failure happens.',
                                    'Why': 'This explains why that failure ends the test, or allows the one fixed '
                                           'rerun.'}}],
          'results': [{'cols': {'What the row asks': 'Each row is one comparison between critics, and its first cell '
                                                     'asks the question that the comparison answers. The first five '
                                                     "rows compare the size of the critic's error, and the last two "
                                                     'compare its direction.',
                                'Seeds 0–4': 'This gives the lowest and the highest value over the five repeat runs '
                                             '(seeds 0–4). In the first five rows the value is an error ratio (×): a '
                                             "critic's typical error divided by that of the full-recording critic. "
                                             "In the last two rows it is the critic's median extra preference for "
                                             'full throttle, in units of return.',
                                'Rule': 'This is the pass rule fixed in the pre-registration, the plan written down '
                                        'before any result.',
                                'Result': 'This says whether the rule passed.'}},
                      {'cols': {'Repeat run': 'Each row is one repeat run of the harm checks, named by its seed; a '
                                              'seed fixes every random draw of a run.',
                                'Critic weight 2.5: actor of the full-recording critic': 'This gives the points in '
                                                                                         'strong current of the '
                                                                                         'actor trained against the '
                                                                                         'full-recording critic (the '
                                                                                         'critic trained on the full '
                                                                                         'recording, with nothing '
                                                                                         'removed), at the published '
                                                                                         'critic weight 2.5. On this '
                                                                                         'scale 0 is the recording '
                                                                                         'controller and 100 is the '
                                                                                         'stall-aware controller.',
                                'Critic weight 2.5: actor of the critic trained after the removal': 'This gives the '
                                                                                                    'points in '
                                                                                                    'strong current '
                                                                                                    'of the actor '
                                                                                                    'trained against '
                                                                                                    'the critic '
                                                                                                    'trained after '
                                                                                                    'the removal, at '
                                                                                                    'the published '
                                                                                                    'critic weight '
                                                                                                    '2.5. On this '
                                                                                                    'scale 0 is the '
                                                                                                    'recording '
                                                                                                    'controller and '
                                                                                                    '100 is the '
                                                                                                    'stall-aware '
                                                                                                    'controller.',
                                'Critic weight 5: actor of the full-recording critic': 'This gives the points in '
                                                                                       'strong current of the actor '
                                                                                       'trained against the '
                                                                                       'full-recording critic, at '
                                                                                       'critic weight 5, the weight '
                                                                                       'at which the check passed. '
                                                                                       'On this scale 0 is the '
                                                                                       'recording controller and 100 '
                                                                                       'is the stall-aware '
                                                                                       'controller.',
                                'Critic weight 5: actor of the critic trained after the removal': 'This gives the '
                                                                                                  'points in strong '
                                                                                                  'current of the '
                                                                                                  'actor trained '
                                                                                                  'against the '
                                                                                                  'critic trained '
                                                                                                  'after the '
                                                                                                  'removal, at '
                                                                                                  'critic weight 5. '
                                                                                                  'On this scale 0 '
                                                                                                  'is the recording '
                                                                                                  'controller and '
                                                                                                  '100 is the '
                                                                                                  'stall-aware '
                                                                                                  'controller.',
                                'Critic weight 5: loss': 'This is the number of points lost by trusting the critic '
                                                         'trained after the removal: the points of the actor of the '
                                                         'full-recording critic minus those of the actor of the '
                                                         'critic trained after the removal, at critic weight 5.'}},
                      {'cols': {'Variant': 'Each row is one way of placing extra pull toward the recorded command on '
                                           'the training rows of the best-case caution check.',
                                'What it is': 'This says how the variant sets the dose, the number that multiplies '
                                              'the pull toward the recorded command on each training row; a dose of '
                                              '1 adds nothing.',
                                'Why it is here': 'This says what the variant is compared for, or what it rules '
                                                  'out.'}},
                      {'cols': {'Variant': 'This names the variant, the way the extra pull toward the recorded '
                                           'command is placed on the training rows; the table above describes each '
                                           'one.',
                                'Recorded conditions': 'This gives the points, as a mean over repeat runs (seeds) '
                                                       '0–4, in the recorded conditions (current 0.1 to 0.7, wave '
                                                       'amplitude 0.1 to 0.4), the conditions of the training data. '
                                                       'On this scale 0 is the recording controller and 100 is the '
                                                       'stall-aware controller.',
                                'Strong current': 'This gives the points, as a mean over repeat runs (seeds) 0–4, in '
                                                  'strong current (current 0.7 to 1.1, stronger than any in the '
                                                  'data). On this scale 0 is the recording controller and 100 is the '
                                                  'stall-aware controller.',
                                'Big waves': 'This gives the points, as a mean over repeat runs (seeds) 0–4, in big '
                                             'waves (wave amplitude 0.4 to 0.7, bigger than any in the data). On '
                                             'this scale 0 is the recording controller and 100 is the stall-aware '
                                             'controller.',
                                'Big starting pushes': 'This gives the points, as a mean over repeat runs (seeds) '
                                                       '0–4, with big starting pushes (the floats start three times '
                                                       'as far from rest). On this scale 0 is the recording '
                                                       'controller and 100 is the stall-aware controller.'}},
                      {'cols': {'Situation': 'Each row is one situation, tested on its 1,000 fixed starting states '
                                             '(its evaluation bank). The recorded conditions are reported but are '
                                             'not part of the rule, which looks only at the shifted situations.',
                                'Over strength-matched uniform': 'This is the points of harm-placed caution minus '
                                                                 'those of uniform caution of the same strength, as '
                                                                 'a mean over repeat runs (seeds) 0–4. The rule '
                                                                 'needs at least +3.',
                                'Over amount-matched': 'This is the points of harm-placed caution minus those of '
                                                       'amount-matched caution, which spreads the same total caution '
                                                       'evenly. The rule needs at least +3.',
                                'Over shuffled': 'This is the points of harm-placed caution minus those of the '
                                                 'shuffled doses, which are the same doses in a random row order. '
                                                 'The rule needs at least +3.',
                                'Result': 'This says whether harm-placed caution reached the pass mark of +3 points '
                                          'against all three controls in this situation.'}},
                      {'cols': {'Level and situation': 'Each row is one test situation of the main test, as a '
                                                       'difficulty level: the levels are ordered by the harm the '
                                                       'missing full-throttle commands cause, not by raw difficulty.',
                                'Standard term': 'This is the usual reinforcement-learning name for how the '
                                                 'situation differs from the training data.',
                                'What differs from the recording': 'This says what changes from the conditions of '
                                                                   'the recorded training data.',
                                'States outside the recorded conditions': 'This is the share of the visited states '
                                                                          'that lie outside the recorded conditions: '
                                                                          'a current above 0.7, a wave amplitude '
                                                                          'above 0.4, or floats farther from rest '
                                                                          'than 99% of the recorded training states.',
                                'States where the stall-aware controller needs a removed command': 'This is the '
                                                                                                   'share of the '
                                                                                                   'states the '
                                                                                                   'stall-aware '
                                                                                                   'controller '
                                                                                                   'visits at which '
                                                                                                   'its own command '
                                                                                                   'lies inside the '
                                                                                                   "main test's "
                                                                                                   'removed ball, so '
                                                                                                   'the training '
                                                                                                   'data no longer '
                                                                                                   'show that '
                                                                                                   'command (the '
                                                                                                   'shift table, '
                                                                                                   'design seed '
                                                                                                   '900).',
                                'Harm from the missing commands, points': "This is the return-cost check's loss at "
                                                                          'critic weight 5: the actor trained '
                                                                          'against the full-recording critic minus '
                                                                          'the actor trained against the critic '
                                                                          'trained after the removal, in points, the '
                                                                          'mean over seeds 0 to 4.',
                                'Actor of the full-recording critic, points': 'This is the score of the actor '
                                                                              'trained against the full-recording '
                                                                              'critic, which shows how hard the '
                                                                              'situation is for a learner when no '
                                                                              'command is missing; on this scale 0 '
                                                                              'is the recording controller and 100 '
                                                                              'the stall-aware controller.'}},
                      {'cols': {'Check': 'Each row is one of the nine world checks, run before any learning on '
                                         'design seeds 900 to 904, asked as the question it answers.',
                                'Pass mark': 'This is the pass mark fixed in the pre-registration, the plan written '
                                             'down before any learner ran.',
                                'Seed 900': 'This is the value on design seed 900, the recording this page shows. A '
                                            'seed fixes every random draw of a recording, and the design seeds are '
                                            'recordings made only to design and check the world.',
                                'All five': 'This gives the lowest and the highest value over the five design seeds, '
                                            '900 to 904.',
                                'Passed': 'This says whether the check passed, and whether it passed on every '
                                          'seed.'}},
                      {'group': 'Kind of arm',
                       'cols': {'Arm': 'Each row is one arm, one way of training the actor; every arm trains against '
                                       'the same frozen critic of the log after the removal and starts from the same '
                                       'cloned actor (the first row). The heading rows sort the arms into groups.',
                                'Strong current: share of states past 80%': 'This is the share of the states visited '
                                                                            'in strong current at which some '
                                                                            'thruster is past 80%, as a mean over '
                                                                            'repeat runs (seeds) 0–4 with the range '
                                                                            'from the lowest to the highest seed '
                                                                            'after it (in small type below the mean '
                                                                            'in the table).',
                                'Recorded conditions': 'This gives the points, as a mean over repeat runs (seeds) '
                                                       '0–4 with the range from the lowest to the highest seed after '
                                                       'it (in small type below the mean in the table), in the '
                                                       'recorded conditions (current 0.1 to 0.7, wave amplitude 0.1 '
                                                       'to 0.4), the conditions of the training data. On this scale '
                                                       '0 is the recording controller and 100 is the stall-aware '
                                                       'controller.',
                                'Strong current': 'This gives the points, as a mean over repeat runs (seeds) 0–4 '
                                                  'with the range from the lowest to the highest seed after it (in '
                                                  'small type below the mean in the table), in strong current '
                                                  '(current 0.7 to 1.1, stronger than any in the data). On this '
                                                  'scale 0 is the recording controller and 100 is the stall-aware '
                                                  'controller.',
                                'Big waves': 'This gives the points, as a mean over repeat runs (seeds) 0–4 with the '
                                             'range from the lowest to the highest seed after it (in small type '
                                             'below the mean in the table), in big waves (wave amplitude 0.4 to 0.7, '
                                             'bigger than any in the data). On this scale 0 is the recording '
                                             'controller and 100 is the stall-aware controller.',
                                'Big starting pushes': 'This gives the points, as a mean over repeat runs (seeds) '
                                                       '0–4 with the range from the lowest to the highest seed after '
                                                       'it (in small type below the mean in the table), with big '
                                                       'starting pushes (the floats start three times as far from '
                                                       'rest). On this scale 0 is the recording controller and 100 '
                                                       'is the stall-aware controller.'}},
                      {'cols': {'Arm': "Each row is one arm, a variant of the actor's training, here trained against "
                                       "the no-stall comparison world's own critic of the log after the removal.",
                                'Recorded conditions': 'This is the cost of caution: the points with no extra '
                                                       'caution minus the points of the arm, so a higher number '
                                                       'costs more, as a mean over repeat runs (seeds) 0–4 with the '
                                                       'range from the lowest to the highest seed after it (in small '
                                                       'type below the mean in the table), in the recorded '
                                                       'conditions (current 0.1 to 0.7, wave amplitude 0.1 to 0.4), '
                                                       'the conditions of the training data. In this world 100 '
                                                       'points is the rising-thrust controller.',
                                'Strong current': 'This is the cost of caution: the points with no extra caution '
                                                  'minus the points of the arm, so a higher number costs more, as a '
                                                  'mean over repeat runs (seeds) 0–4 with the range from the lowest '
                                                  'to the highest seed after it (in small type below the mean in the '
                                                  'table), in strong current (current 0.7 to 1.1, stronger than any '
                                                  'in the data). In this world 100 points is the rising-thrust '
                                                  'controller.',
                                'Big waves': 'This is the cost of caution: the points with no extra caution minus '
                                             'the points of the arm, so a higher number costs more, as a mean over '
                                             'repeat runs (seeds) 0–4 with the range from the lowest to the highest '
                                             'seed after it (in small type below the mean in the table), in big '
                                             'waves (wave amplitude 0.4 to 0.7, bigger than any in the data). In '
                                             'this world 100 points is the rising-thrust controller.',
                                'Big starting pushes': 'This is the cost of caution: the points with no extra '
                                                       'caution minus the points of the arm, so a higher number '
                                                       'costs more, as a mean over repeat runs (seeds) 0–4 with the '
                                                       'range from the lowest to the highest seed after it (in small '
                                                       'type below the mean in the table), with big starting pushes '
                                                       '(the floats start three times as far from rest). In this '
                                                       'world 100 points is the rising-thrust controller.'}},
                      {'cols': {'Arm': "Each row is one arm, a variant of the actor's training against the critic of "
                                       'the best-case caution check, which is the frozen critic of a run of plain '
                                       'twin delayed deep deterministic policy gradient plus behaviour cloning '
                                       '(TD3+BC) on the log after the removal.',
                                'Recorded conditions': 'This gives the points, as a mean over repeat runs (seeds) '
                                                       '0–4 with the range from the lowest to the highest seed after '
                                                       'it (in small type below the mean in the table), in the '
                                                       'recorded conditions (current 0.1 to 0.7, wave amplitude 0.1 '
                                                       'to 0.4), the conditions of the training data. On this scale '
                                                       '0 is the recording controller and 100 is the stall-aware '
                                                       'controller.',
                                'Strong current': 'This gives the points, as a mean over repeat runs (seeds) 0–4 '
                                                  'with the range from the lowest to the highest seed after it (in '
                                                  'small type below the mean in the table), in strong current '
                                                  '(current 0.7 to 1.1, stronger than any in the data). On this '
                                                  'scale 0 is the recording controller and 100 is the stall-aware '
                                                  'controller.',
                                'Big waves': 'This gives the points, as a mean over repeat runs (seeds) 0–4 with the '
                                             'range from the lowest to the highest seed after it (in small type '
                                             'below the mean in the table), in big waves (wave amplitude 0.4 to 0.7, '
                                             'bigger than any in the data). On this scale 0 is the recording '
                                             'controller and 100 is the stall-aware controller.',
                                'Big starting pushes': 'This gives the points, as a mean over repeat runs (seeds) '
                                                       '0–4 with the range from the lowest to the highest seed after '
                                                       'it (in small type below the mean in the table), with big '
                                                       'starting pushes (the floats start three times as far from '
                                                       'rest). On this scale 0 is the recording controller and 100 '
                                                       'is the stall-aware controller.'}},
                      {'cols': {'Mark': 'Each row is one pass mark (a pass-or-fail rule) of Stage 1, the first '
                                        'exploratory stage after the harm checks stopped the test, and every mark '
                                        'was fixed before any return was computed.',
                                'Rule, fixed before any return was computed (pooled over seeds 0–4 unless the rule names seeds)': 'This '
                                                                                                                                  'is '
                                                                                                                                  'the '
                                                                                                                                  "mark's "
                                                                                                                                  'rule. '
                                                                                                                                  'Pooled '
                                                                                                                                  'means '
                                                                                                                                  'combined '
                                                                                                                                  'over '
                                                                                                                                  'repeat '
                                                                                                                                  'runs '
                                                                                                                                  '(seeds) '
                                                                                                                                  '0–4, '
                                                                                                                                  'unless '
                                                                                                                                  'the '
                                                                                                                                  'rule '
                                                                                                                                  'names '
                                                                                                                                  'particular '
                                                                                                                                  'seeds.',
                                'Result': 'This is what was measured.',
                                'Verdict': 'This says whether the mark passed, or, for the futility look (an early '
                                           'check that could have stopped the test), whether to continue.'}},
                      {'cols': {'Rule': 'This says what happens at each state the actor visits, with no training; '
                                        'the actor is the one trained against the critic of the log after the '
                                        'removal.',
                                'Recorded conditions': 'This is the change in points from that actor alone, in the '
                                                       'stalling world, as a mean over repeat runs (seeds) 0–4 on '
                                                       "the first 200 of the situation's 1,000 fixed starting "
                                                       'states, in the recorded conditions (current 0.1 to 0.7, wave '
                                                       'amplitude 0.1 to 0.4), the conditions of the training data.',
                                'Strong current': 'This is the change in points from that actor alone, in the '
                                                  'stalling world, as a mean over repeat runs (seeds) 0–4 on the '
                                                  "first 200 of the situation's 1,000 fixed starting states, in "
                                                  'strong current (current 0.7 to 1.1, stronger than any in the '
                                                  'data).',
                                'Big waves': 'This is the change in points from that actor alone, in the stalling '
                                             'world, as a mean over repeat runs (seeds) 0–4 on the first 200 of the '
                                             "situation's 1,000 fixed starting states, in big waves (wave amplitude "
                                             '0.4 to 0.7, bigger than any in the data).',
                                'Big starting pushes': 'This is the change in points from that actor alone, in the '
                                                       'stalling world, as a mean over repeat runs (seeds) 0–4 on '
                                                       "the first 200 of the situation's 1,000 fixed starting "
                                                       'states, with big starting pushes (the floats start three '
                                                       'times as far from rest).'}}]},
 'legends': {'observation-space': [["The 10 × 10 table of the floats' own motion (Transition Dynamics).",
                                    "It is the 10 × 10 table of the floats' own motion (Transition Dynamics)."],
                                   ['How many steps ago a push was given.',
                                    'It is how many steps ago a push was given.'],
                                   ['The largest stretch a matrix can apply to any list of numbers.',
                                    'It is the largest stretch that a matrix can apply to any list of numbers.'],
                                   ['Those stretches added over every t: 55.483.',
                                    'It is the sum of those stretches over every t: 55.483.'],
                                   ['The largest push one step can add.',
                                    'It is the largest push that one step can add.'],
                                   ['The farthest an allowed start can put the floats from rest.',
                                    'It is the farthest from rest that an allowed start can put the floats.']],
             'episode-end': [['How many steps are simulated: 1,000 or 600.',
                              'It is how many steps are simulated: 1,000 or 600.'],
                             ['The most the rest of the never-ending sum can add after H steps, as a multiple of the '
                              'largest single-step cost that follows.',
                              'It is the most that the rest of the never-ending sum can add after H steps, as a '
                              'multiple of the largest single-step cost that follows.'],
                             ["The never-ending discounted sum of rewards that a learner's critic estimates.",
                              "It is the never-ending discounted sum of rewards that a learner's critic estimates."],
                             ['It shows the push the thruster really gives, f(a); pointing up is upstream, against '
                              'the water.',
                              'It shows the push the thruster really gives, f(a); pointing up is upstream, against '
                              'the water. The thruster and its arrow are drawn in dark ink, and in red once the '
                              'thruster stalls.'],
                             ["It shows the thruster's throttle a, from −1 (full throttle against the water, at the "
                              'top) to 1.',
                              "It shows the thruster's throttle a, from −1 (full throttle against the water, at the "
                              'top) to 1. The bar is grey, and red once the thruster stalls.']],
             'task': [["The current's strength: drawn at the start, then steady all episode; it pushes every float "
                       'downstream equally. 0.1 to 0.7 in the recorded conditions; 0.445 on this start. The solid '
                       'part of each top arrow.',
                       "It is the current's strength, drawn at the start and then steady all episode, and it pushes "
                       'every float downstream equally. It is 0.1 to 0.7 in the recorded conditions and 0.445 on '
                       'this start, and it is the solid part of each top arrow.'],
                      ["The wave's push now: it runs from plus the amplitude down to minus the amplitude and back "
                       "every 23.27 steps. This start's amplitude is 0.218. The dashed part of each top arrow.",
                       "It is the wave's push now, which runs from plus the amplitude down to minus the amplitude "
                       "and back every 23.27 steps. This start's amplitude is 0.218, and the wave is the dashed part "
                       'of each top arrow.'],
                      ["The water's push on every float in this state. Downstream unless the wave takes away more "
                       'than the current gives.',
                       "It is the water's push on every float in this state, which points downstream unless the wave "
                       'takes away more than the current gives.'],
                      ["+ when the wave adds to the current's push, − when it takes from it; the faint dashed part "
                       'is then the push it takes away.',
                       "The sign is + when the wave adds to the current's push and − when it takes from it; the "
                       'faint dashed part is then the push it takes away.'],
                      ["Where every float's two numbers are 0: the dashed line.",
                       "It is where every float's two numbers are 0, and the dashed line marks it."],
                      ["A float's period, the steps one swing takes, shown rounded to whole steps under each float: "
                       '63 for float 1 (62.8) down to 8 for float 5 (7.8).',
                       "It is a float's period, the steps one swing takes, shown rounded to whole steps under each "
                       'float: 63 for float 1 (62.8) down to 8 for float 5 (7.8).'],
                      ['Up to 80%, a thruster pushes exactly as hard as its throttle.',
                       'It means that up to 80% a thruster pushes exactly as hard as its throttle.'],
                      ['Past 80% the propeller stalls: push 0.65 at 90% and 0.5 at full throttle, while the throttle '
                       'cost keeps rising.',
                       'It means that past 80% the propeller stalls: the push is 0.65 at 90% and 0.5 at full '
                       'throttle, while the throttle cost keeps rising.'],
                      ["Thruster 1's throttle (the bar's fill) and the push it really gives (the tick).",
                       "They show thruster 1's throttle (the bar's fill) and the push it really gives (the tick)."],
                      ["While the drawing plays, the top arrows turn gold when the water's push is above 0.565: the "
                       'state is then in the full-throttle region.',
                       "While the drawing plays, the top arrows turn gold when the water's push is above 0.565, "
                       'which means the state is in the full-throttle region.'],
                      ["The water's push c + w1 at the state (c the current, w1 the wave's push now), in bands 0.2 "
                       'wide.',
                       "It is the water's push c + w1 at the state (c the current, w1 the wave's push now), sorted "
                       'into bins 0.2 wide.'],
                      ["Thruster 1's average throttle or push. Negative is upstream, so higher on the chart, toward "
                       '−1, means a harder push against the water.',
                       "It is thruster 1's average throttle or push. Negative is upstream, so higher on the chart, "
                       'toward −1, means a harder push against the water.'],
                      ['At −0.8: past it the propeller stalls.',
                       'It is the line at −0.8, and past it the propeller stalls.'],
                      ["What the rising-thrust controller's thruster 1 really delivers: it falls back once it "
                       'stalls, to −0.51 at the strongest pushes.',
                       "It is what the rising-thrust controller's thruster 1 really delivers, which falls back once "
                       'it stalls, to −0.51 at the strongest pushes.'],
                      ["One band's average. Hover over or tap anywhere in a band's column for its three numbers; the "
                       'table below lists them all, for keyboard and screen-reader use and with tooltips off.',
                       "Each dot is one bin's average. Hover over or tap anywhere in a bin's column for its three "
                       'numbers; the table below lists them all, for keyboard and screen-reader use and with '
                       'tooltips off.']],
             'dataset': [['The step within the episode, 0 to 99.',
                          'It is the step within the episode, from 0 to 99.'],
                         ['The recorded throttle: −1 is full throttle upstream, against the water. Below the −80% '
                          'line thruster 1 stalls.',
                          'It is the recorded throttle, where −1 is full throttle upstream, against the water. Below '
                          'the −80% line thruster 1 stalls.'],
                         ["The full-throttle region: the water's push c + w1 is above 0.565.",
                          "They are the steps in the full-throttle region, where the water's push c + w1 is above "
                          '0.565.']],
             'critic-in-action': [["The state on screen: the five floats' positions and speeds, the current c and "
                                   'the wave numbers w₁ and w₂.',
                                   "It is the state on screen, all 13 observed numbers: the five floats' positions "
                                   'and speeds, the current c and the wave numbers w₁ and w₂. Methods writes this '
                                   'whole observation x and keeps s for the 10 float numbers.'],
                                  ["A command: the four throttles. In the upper chart thruster 1's throttle takes 41 "
                                   'values from 0% to 100% upstream, 2.5% apart, and thrusters 2 to 4 stay where the '
                                   'actor put them.',
                                   "It is a command, the four throttles. In the upper chart thruster 1's throttle "
                                   'takes 41 values from 0% to 100% upstream, 2.5% apart, and thrusters 2 to 4 stay '
                                   'where the actor put them.'],
                                  ['The command the actor gives in state s; π is the trained actor.',
                                   'It is the command the actor gives in state s, where π is the trained actor.'],
                                  ["The critic's value: its estimate of the return from giving a in s and letting "
                                   "the actor drive after that. The smaller of the critic's two estimates, min(Q1, "
                                   'Q2).',
                                   "It is the critic's value: its estimate of the discounted return from giving a in "
                                   "s and letting the actor drive after that. It is the smaller of the critic's two "
                                   'estimates, min(Q1, Q2).'],
                                  ['The true value of the same thing, by exact simulation: 600 steps, discount 0.98.',
                                   'It is the true value of the same command, by exact simulation over 600 steps at '
                                   'discount 0.98. Methods writes this exact value Qtrue.'],
                                  ['The swept throttle the critic rates highest in this state.',
                                   'It is the swept throttle that the critic rates highest in this state.'],
                                  ['The swept throttle with the highest true value in this state. It is the best of '
                                   'the 41 throttles with the other three thrusters fixed, not the best possible '
                                   'command.',
                                   'It is the swept throttle with the highest true value in this state. It is the '
                                   'best of the 41 throttles with the other three thrusters fixed, not the best '
                                   'possible command.'],
                                  ['The stall limit. Past it the push falls (0.8 at 80%, 0.5 at 100%) and the '
                                   'throttle cost, 0.3 × throttle², keeps rising.',
                                   'It is the stall limit. Past it the push falls (0.8 at 80%, 0.5 at 100%) and the '
                                   'throttle cost, 0.3 × throttle², keeps rising.'],
                                  ["Throttles whose command lies inside this section's removed ball, which was taken "
                                   'out of the training data of the critic with "the 80% hold removed": every '
                                   'command within  (the distance across all four throttles, each from −1 to 1) of a '
                                   'centre command that has thruster 1 at  upstream. The ball takes in the 80% hold. '
                                   'It is shaded only for that critic, and only in states where the throttle sweep '
                                   'passes through the ball.',
                                   "These are the throttles whose command lies inside this section's removed ball, "
                                   'which was taken out of the training data of the critic with "the 80% hold '
                                   'removed": every command within  (the distance across all four throttles, each '
                                   'from −1 to 1) of a centre command that has thruster 1 at  upstream. The ball '
                                   'takes in the 80% hold. It is shaded only for that critic, and only in states '
                                   'where the throttle sweep passes through the ball.'],
                                  ["This section's own scale: 100 × (return − Jrandom) / (Jstall-aware − Jrandom), "
                                   'where Jrandom and Jstall-aware are the average returns of uniformly random '
                                   "commands and of the stall-aware controller over the mix's 1,000 starts:  and  in "
                                   'the training mix,  and  in the strong test mix. Elsewhere on this page 0 points '
                                   'is the recording controller.',
                                   "It is this section's own scale: 100 × (return − Jrandom) / (Jstall-aware − "
                                   'Jrandom), where Jrandom and Jstall-aware are the average returns of uniformly '
                                   "random commands and of the stall-aware controller over the mix's 1,000 starts:  "
                                   'and  in the training mix,  and  in the strong test mix. Elsewhere on this page 0 '
                                   'points is the recording controller.']]}}

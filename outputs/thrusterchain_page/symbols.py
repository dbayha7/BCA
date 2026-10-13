"""The page's symbols: one entry per symbol, used for the legend under every displayed formula and under the task
diagram, and for the tooltip of every formula in the text, so a symbol reads the same everywhere.

An entry: (TeX of the symbol, what it is, its range or value, where it comes from). Text is HTML with inline TeX
\\( \\) allowed (the page typesets it at build time) and numbers from the data as {{path#fmt}}, like the rest of the page.

symbol_keys(tex, ctx) reads a formula and returns the entries it uses, in order of first use. ctx is where the formula
stands (the id of its section, or 'beyond' for the After-the-stop subsection), because a few letters mean different
things in different places (a, c, i, k, n, U, v, z, sigma, pi); every such case is resolved here, not guessed.
"""

import re

C0 = "{{environment/start_laws/recorded/current/0#f1}}"
C1 = "{{environment/start_laws/recorded/current/1#f1}}"
SC0 = "{{environment/start_laws/strong_current/current/0#f1}}"
SC1 = "{{environment/start_laws/strong_current/current/1#f1}}"
A0 = "{{environment/start_laws/recorded/wave_amplitude/0#f1}}"
A1 = "{{environment/start_laws/recorded/wave_amplitude/1#f1}}"
BA0 = "{{environment/start_laws/big_waves/wave_amplitude/0#f1}}"
BA1 = "{{environment/start_laws/big_waves/wave_amplitude/1#f1}}"
PER = "{{environment/dynamics/wave_period_steps#f2}}"
QTR = "{{overview/waves/quarter_cycle_steps#f1}}"
SD0 = "{{environment/start_laws/recorded/float_start_sd#f1}}"
SD3 = "{{environment/start_laws/big_kicks/float_start_sd#f1}}"
BOUND = "{{environment/observation_bound/bound#f1}}"
GAIN = "{{environment/physics/stall/gain#f1}}"
TCOST = "{{environment/physics/stall/throttle_cost#f1}}"
TURN = "{{environment/physics/stall/wave_turn#f2}}"
GAMMA = "{{environment/physics/stall/gamma#f2}}"
SLOPE = "{{environment/physics/stall/stall_slope#f1}}"
P90 = "{{environment/thrust_curve/key_points/8/thrust_stall#f2}}"
P100 = "{{environment/thrust_curve/key_points/10/thrust_stall#f1}}"
TRIG = "{{recording_controller/trigger#f3}}"
ALPHA = "{{methods/td3_bc/settings/alpha#f1}}"
NOISE = "{{methods/td3_bc/settings/policy_noise#f1}}"
NCLIP = "{{methods/td3_bc/settings/noise_clip#f1}}"
BATCH = "{{methods/td3_bc/settings/batch_size#i}}"
MU = "{{methods/actor_fixed_point/critic_weight_k_reference#f2}}"
KAPPA = "{{sources/design_mu/curvature#f2}}"
SIZE = "{{sources/design_mu/size_with_noise#f1}}"
BLEND = "{{methods/bca_dose/settings/blend#f1}}"
MISS = "{{methods/bca_dose/settings/miscoverage#p0}}"
DAMP = "{{recording_controller/settings/damping#f2}}"
WOB = "{{recording_controller/settings/wobble#f1}}"
JIT = "{{recording_controller/settings/jitter#f2}}"
KCOL = ("({{recording_controller/K_current_column/0#f2}}, {{recording_controller/K_current_column/1#f2}}, "
        "{{recording_controller/K_current_column/2#f2}}, {{recording_controller/K_current_column/3#f2}})")
PAT = ("({{recording_controller/full_throttle_pattern/0#f2}}, {{recording_controller/full_throttle_pattern/1#f2}}, "
       "{{recording_controller/full_throttle_pattern/2#f2}}, {{recording_controller/full_throttle_pattern/3#f2}})")

# key: (TeX shown for the symbol, what it is, range or value, where it comes from)
SYMS = {
    # ---- the water and the floats
    "c": (r"c", "the current's strength: how hard the steady current pushes every float downstream. "
          r"\(c_t\) is its value at step \(t\); it never changes during an episode.",
          f"{C0} to {C1} in the recorded conditions, {SC0} to {SC1} in strong current",
          "drawn at the start of each episode from the situation's start law; observation 10"),
    "c_range": (r"c_{\text{low}},\ c_{\text{high}}", "the two ends of the situation's range of currents.",
                f"{C0} and {C1} in the recorded conditions; {SC0} and {SC1} in strong current",
                "the situation's start law (Starting State)"),
    "w1": (r"w_1", "the push the wave adds to the current's push now; negative when the wave takes push away. "
           r"\(w_{1,t}\) is its value at step \(t\).",
           f"from minus to plus the wave's amplitude; the amplitude is {A0} to {A1} in the recorded conditions and "
           f"{BA0} to {BA1} in big waves",
           f"observation 11; it rises and falls once every {PER} steps"),
    "w2": (r"w_2", "the push the wave added a quarter cycle earlier. It tells which way the push is heading: while it "
           "is positive the push is falling, while negative rising.",
           "from minus to plus the wave's amplitude", f"observation 12, {QTR} steps behind \\(w_1\\)"),
    "w": (r"w_t", r"the wave as one point with two numbers, \((w_1, w_2)\), at step \(t\): it turns "
          f"{TURN} radians every step and keeps its size.",
          "its size is the wave's amplitude", "observations 11 and 12"),
    "amplitude": (r"\text{amplitude}", "the largest push the wave adds to the current's push or takes away from it.",
                  f"{A0} to {A1} in the recorded conditions, {BA0} to {BA1} in big waves",
                  "drawn at the start of each episode, then fixed"),
    "phi": (r"\phi", "where in its cycle the wave starts (its phase), as an angle.",
            r"0 up to \(2\pi\) radians, one whole cycle", "drawn at the start, every value equally likely"),
    "theta": (r"\theta", "an angle in radians: how far a rotation turns in one step.",
              f"{TURN} for the wave; each float's swing turns by its own angle (the table of swings below)",
              "the fixed physics"),
    "rho": (r"\rho", "how much a float's swing shrinks in one step: its size is multiplied by this factor.",
            "just below 1, one value per float (the table of swings below)", "the fixed physics of \\(A\\)"),
    "R": (r"R(\theta)", r"a rotation: it turns the wave's point \((w_1, w_2)\) by the angle \(\theta\) without "
          "changing its size.", f"a 2 × 2 table of numbers; \\(R({TURN})\\) turns by {TURN} radians",
          "the fixed physics"),
    "const_pi": (r"\pi", r"the number pi, 3.14159…; \(2\pi\) radians is one whole turn.", "a constant",
                 "mathematics"),
    "s": (r"s", r"the 10 float numbers of a state: the position and speed numbers of the five floats, "
          r"\((p_1, v_1, \dots, p_5, v_5)\), all 0 at rest. \(s_t\) is the state at step \(t\), \(s_{t,i}\) its "
          r"number \(i\), \(s_{0,i}\) the starting value.",
          f"each stays within {BOUND} of rest (proven)", "observations 0 to 9"),
    "p": (r"p", r"a float's position number, \(p_1\) to \(p_5\); negative means carried downstream of rest.",
          "0 at rest", "observations 0, 2, 4, 6 and 8"),
    "v_speed": (r"v", r"a float's speed number, \(v_1\) to \(v_5\): the second coordinate of its swing.",
                "0 at rest", "observations 1, 3, 5, 7 and 9"),
    "v_vec": (r"v", "where the water's push lands: 10 numbers, 1 at each float's speed number and 0 at each "
              "position number, so the water pushes every float's speed equally.", "fixed",
              "the fixed physics"),
    "x": (r"x", r"the whole observation, all 13 numbers: \(x = (s, e)\). \(x_t\) is the observation at step "
          r"\(t\), \(x_i\) the one of recorded row \(i\), and \(x'\) the next observation, one step later.",
          "13 numbers", "what a controller reads at every step"),
    "e": (r"e", r"the water's three numbers, \(e = (c, w_1, w_2)\).", "see \\(c\\), \\(w_1\\) and \\(w_2\\)",
          "observations 10 to 12"),
    # ---- throttles, push, reward
    "a": (r"a", r"a command: the four throttles \((a_1, a_2, a_3, a_4)\), after clipping. \(a_t\) is the command "
          r"at step \(t\), \(a_{t,k}\) thruster \(k\)'s throttle in it, and \(a_i\) the recorded command of row "
          r"\(i\).",
          "each throttle from −1 to 1; negative pushes upstream; past ±0.8 a propeller stalls in the stalling world",
          "set by the controller or the actor at every step; for a recorded row, by the recording controller"),
    "a_thr": (r"a", "one thruster's throttle, after clipping.",
              "−1 to 1; negative pushes upstream; past ±0.8 the propeller stalls in the stalling world",
              "one of the four numbers of a command"),
    "a_star": (r"a^*", "the critic's favourite command at a state: the command its estimate rates best.",
               "four throttles, each −1 to 1", "the critic, at one state"),
    "f": (r"f(a)", r"the push a thruster really gives at throttle \(a\).",
          f"equal to the throttle up to 80%; past 80% in the stalling world it falls, to {P90} at 90% and {P100} at "
          "full throttle", "the stall curve (Action Space)"),
    "A": (r"A", "how the floats move on their own in one step: a 10 × 10 table of numbers that turns and shrinks "
          "each float's swing a little and adds a weak, fixed coupling between the floats.",
          "fixed (the table in Transition Dynamics)",
          "the fixed physics; the coupling was drawn once, from seed {{sources/coupling_seed/value#s}}"),
    "B": (r"B", r"which thruster pushes which float: a 10 × 4 table with a 1 where thruster \(k\) pushes the speed "
          r"numbers of floats \(k\) and \(k + 1\), and 0 elsewhere.", "fixed", "the fixed physics"),
    "r": (r"r", r"the reward of one step: minus the sum of the float cost and the throttle cost. \(r_t\) is the "
          r"reward at step \(t\); \(r(x, a)\) the reward of giving command \(a\) at \(x\).",
          "never above 0; 0 only with every float at rest and every throttle off",
          "returned by the environment after every step"),
    "J": (r"J", "the return: the sum of an episode's 100 rewards. In the points formula it is a controller's average "
          "over the 1,000 starts of a situation.", "never above 0; higher is better",
          "added up from the rewards"),
    "J_gamma": (r"J_\gamma", f"the discounted return: each reward counts {GAMMA} to the power of its step, so a "
                "reward t steps ahead counts less.", "never above 0; higher is better",
                "added up from the rewards"),
    "J_rec": (r"J_{\text{rec}}", "the recording controller's average return in the situation.",
              "one number per situation and world", "exact simulation of the 1,000 starts (Reference Scores)"),
    "J_top": (r"J_{\text{top}}", "the average return of the top reference: the stall-aware controller in the "
              "stalling world, the rising-thrust controller in the no-stall comparison world.",
              "one number per situation and world", "exact simulation of the 1,000 starts (Reference Scores)"),
    "points": (r"\text{points}", "the score of a controller: 0 is the recording controller's level and 100 the top "
               "reference's.", "any number; above 100 is possible", "computed from average returns"),
    "J_new_ref": (r"J_{\text{new}},\ J_{\text{ref}}", "the returns of the new actor and of the reference actor from "
                  "the same start.", "never above 0", "exact simulation"),
    "gamma": (r"\gamma", "the discount: how much a reward one step later counts against a reward now.", GAMMA,
              "every learner's setting on this page"),
    # ---- indexes
    "idx_t": (r"t", "the step: it counts the steps of an episode from 0.", "0 to 99 (t = 0 is the first step)", ""),
    "idx_i_float": (r"i", "which of the 10 float numbers.", "1 to 10", ""),
    "idx_i_row": (r"i", "which recorded row (one recorded step of the data).",
                  f"one of the {BATCH} rows of a training batch, or any recorded row", ""),
    "idx_k": (r"k", "which thruster.", "1 to 4", ""),
    "idx_j": (r"j", "which of the critic's two estimates (TD3+BC keeps two critics).", "1 or 2", ""),
    "n_steps": (r"n", "a number of steps: how far ahead an n-step return adds up rewards.",
                r"\(100 - t\) here: the steps left in the episode", ""),
    "H": (r"H", "the number of steps simulated before the rest of a never-ending sum is dropped.",
          "{{methods/td3_bc/settings/oracle_horizon#i}} or 1,000 on this page", "a setting of the exact truth"),
    # ---- the controllers' rules
    "K": (r"K", "the textbook rule: a table of numbers, one row per thruster and 13 columns, that turns the 13 "
          r"observed numbers \(x\) into the textbook command \(-K\,x\). It is the best linear rule if push always "
          "equalled throttle.", "fixed (4 × 13 numbers, The recording controller)",
          "computed once by the discounted linear-quadratic regulator"),
    "K_s": (r"K_s", r"the part of \(K\) that reads the 10 float numbers \(s\).", "a 4 × 10 table, fixed",
            r"the first 10 columns of \(K\)"),
    "K_e": (r"K_e", r"the part of \(K\) that reads the water's three numbers \(e\).", "a 4 × 3 table, fixed",
            r"the last 3 columns of \(K\)"),
    "K_col": (r"K_{\cdot,c}", r"the current's column of \(K\): how much each thruster's command answers the "
              r"current. \(K_{1,c}\) is its entry for thruster 1.", KCOL, r"the column of \(K\) for observation 10"),
    "u": (r"u", "the normal-driving command before it is squashed into ±80%.", "four numbers, any size",
          "the recording controller's normal driving"),
    "n": (r"n", "the wobble: four fresh random numbers each step, each from a standard normal distribution.",
          f"mean 0, spread (standard deviation) 1; scaled by {WOB} in normal driving and {JIT} in the "
          "full-throttle command", "the recording controller's random stream"),
    "I4": (r"I_4", "the 4 × 4 identity table: the four wobble numbers are independent, each with spread 1.",
           "fixed", "mathematics"),
    "P": (r"P", "the full-throttle pattern: the textbook hold against the current, scaled up until thruster 1 is "
          "at full throttle.", PAT, r"the current's column of \(K\)"),
    "P_cap": (r"P_{\text{cap}}", "the full-throttle pattern with thruster 1 cut back to 80%.",
              "thruster 1 at −0.8, the others as in the pattern", "the full-throttle pattern"),
    # ---- learning
    "Q": (r"Q", r"the critic: a learned estimate of the discounted return of giving command \(a\) at observation "
          r"\(x\) and letting the actor drive afterwards. \(Q_1\) and \(Q_2\) are its two estimates, "
          r"\(Q_j\) either one.", "on the scale of discounted returns (never above 0)",
          "learned from the recorded rows"),
    "Qbar": (r"\bar Q_j", "slowly averaged copies of the two critics (the target networks), which the critic's "
             "target uses so that it moves smoothly.", "as the critic", "updated a little after every update"),
    "Qtrue": (r"Q^{\text{true}}", "the exact true value of a command: the exact discounted return of giving it and "
              "then following the controller or actor in question.", "on the scale of discounted returns",
              "exact simulation"),
    "Qref": (r"Q^{n}_{\text{ref}}", r"the exact return over the next \(n\) steps of giving command \(a\) at \(x\) "
             "and then following the reference actor.", "never above 0", "exact simulation"),
    "pi": (r"\pi", r"the actor, the learned controller: \(\pi(x)\) is its command at observation \(x\) and "
           r"\(\pi_k(x)\) its throttle for thruster \(k\).", "each throttle from −1 to 1",
           "trained by TD3+BC (or one of the runs that change it)"),
    "pibar": (r"\bar\pi", "a slowly averaged copy of the actor (a target network), used inside the critic's target.",
              "as the actor", "updated a little after every update"),
    "pi_new_ref": (r"\pi_{\text{new}},\ \pi_{\text{ref}}", "the new actor and the reference actor being compared.",
                   "each throttle from −1 to 1", "two trained actors"),
    "y": (r"y", "the critic's one-step target: the row's reward plus the discounted, lower of the two target "
          "critics' estimates at the next state.", "on the scale of discounted returns",
          "computed from a recorded row"),
    "epsilon": (r"\epsilon", "target noise: a random change added to the next command inside the target.",
                f"normal with spread {NOISE}, cut off at ±{NCLIP}", "TD3+BC's setting"),
    "L_actor": (r"L_{\text{actor}}", "the actor's loss: the number every actor update makes smaller.",
                "any number", "computed on each batch"),
    "lambda": (r"\lambda", r"the critic's weight in the actor loss: \(\alpha\) divided by the critic's typical size, "
               "so the balance does not depend on the scale of the rewards.",
               f"about {ALPHA} / {SIZE} in the design's measurement", "computed on each batch"),
    "alpha": (r"\alpha", "the critic weight: how much the critic counts against the pull toward the recorded "
              "commands. Larger means the actor follows its critic more.",
              f"{ALPHA} (the published value); {{{{methods/gate/gate_config/alpha_ladder/1#f0}}}} and "
              f"{{{{methods/gate/gate_config/alpha_ladder/2#f0}}}} on the alpha ladder; the gate used "
              f"{{{{beyond/design/alpha#f0}}}}",
              "a setting of TD3+BC"),
    "m": (r"m", r"the caution multiplier (the dose): how hard the actor is pulled toward the recorded command. "
          r"\(m_i\) is its value at row \(i\).", "1 means no extra caution; BCA's doses lie from 1 up to 1.5",
          "1 in plain TD3+BC; set by the caution method in use"),
    "b": (r"b_i", r"the cloning pull at row \(i\): the squared gap between the actor's command and the recorded "
          "command, averaged over the 4 thrusters.", "0 or more; 0 when the actor copies the row",
          "computed on each batch"),
    "mu": (r"\mu", "the critic's pull: how hard the critic's hill draws the actor toward its favourite command, on the "
           "same scale as the multiplier.", f"{MU} in the design's measurement; "
           "{{world_card/range/kappa/0#f3}} to {{world_card/range/kappa/1#f3}} on the design seeds",
           r"\(\mu = d\,\lambda\,\kappa / 2\), measured (Methods)"),
    "kappa": (r"\kappa", "the curvature of the critic's hill: how steeply its value falls away from the favourite "
              "command.", f"{KAPPA} in the design's measurement", "measured on 1,000 training rows"),
    "d": (r"d", "the number of throttles.", "4", "the action space"),
    "delta": (r"\delta", r"the critic's error at a command: its estimate minus the exact truth. \(\delta_A\) is the "
              "advantage error: the error at a command minus the error at the controller's own command.",
              "any number; 0 means exact", "critic minus exact simulation"),
    "o": (r"o(x)", "the pull past 80%: how much more the critic prefers the full-throttle pattern over the same "
          "pattern cut to 80% than the truth does.", "any number; positive means a false preference for full "
          "throttle", "critic minus exact simulation"),
    "Delta": (r"\Delta_i", r"the over-preference at row \(i\): how much more the critic prefers the actor's command "
              "over the recorded one than the truth does, counted only when positive.", "0 or more",
              "critic minus exact simulation, recomputed every {{methods/td3_bc/settings/oracle_refresh#i}} steps"),
    "z": (r"z_i", r"the over-preference scaled by its mean: \(z_i = 3\,\Delta_i / \operatorname{mean}(\Delta)\).",
          "0 or more; 3 at a row of average over-preference", "computed from the over-preferences"),
    "S": (r"S", "the strength of a caution run: the root-mean-square change of its actor's command (the square root "
          "of the average squared change), compared with the run without caution.",
          "0 or more", "measured on {{methods/td3_bc/settings/strength_states#i}} held-out states"),
    "U": (r"U", r"BCA's band width: a range around the critic's estimate that should contain its one-step error. "
          r"\(U_i\) is the width at row \(i\), \(U(x, a)\) the width at command \(a\).", "0 or more",
          "BCA's error-size network, scaled by its calibration"),
    "beta": (r"\beta", "BCA's band multiplier, set by calibration so that the band covers the critic's error on all "
             f"but {MISS} of the calibration rows.",
             "0 or more; it can be infinite when no finite band covers the error", "BCA's calibration"),
    "eta": (r"\eta", "the running unit: the typical size of the critic's one-step errors in recent batches.",
            "more than 0", "kept up to date during training"),
    "nu": (r"\nu", r"the even extra pull's strength: the pull is multiplied by \(1 + \nu\) on every row.",
           "0 or more", "set by strength matching"),
    "W": (r"W", r"Stage 1's penalty on the value of the actor's command \(a\) at a training state; zero at the "
          "recorded command.", "0 or more", r"built from the exact one-step error \(\delta^*\)"),
    "delta_star": (r"\delta^*", "the critic's exact one-step error at a command: the reward plus the discounted "
                   "estimate at the exact next state, minus the estimate at the command. Negative means the critic "
                   "rates the command too highly.", "any number", "one exact simulator step"),
    "c_ctrl": (r"c", r"the stall-aware controller, whose values this critic estimates; \(c(x')\) is its command at the "
               r"next state \(x'\).", "each throttle within ±0.8", "the fixed stall-aware controller"),
    "k_pen": (r"k", "the penalty's weight.", "{{beyond/design/k_main#f0}} for the main arm, {{beyond/design/k_half#f1}} at "
              "half strength",
              "fixed in the plan before any training"),
    "k_stage2": (r"k", "the weight of the band penalty in the new version of BCA.",
                 "{{beyond/status/stage2_k#f0}} in the main version; {{beyond/status/stage2_k_secondary#f0}} in a "
                 "declared secondary arm",
                 "the Stage 2 pre-registration, which fixes it by its meaning (the band's lower edge); it is never tuned "
                 "on returns"),
    "z_any": (r"z", "any number.", "", ""),
    "sigma_start": (r"\sigma", "the spread (standard deviation) of each float number at the start: how far from rest "
                    "a start typically is.", f"{SD0}; {SD3} for big starting pushes", "the situation's start law"),
    "sigma_seeds": (r"\sigma", "the seed-to-seed spread (standard deviation), in points, of the larger of two paired "
                    "differences.", "not measured: the power check did not run; it would be the largest seed-to-seed standard "
                    "deviation of the two paired differences on the room-certified situations",
                    "the power check"),
    "N": (r"N", "the number of confirmation seeds a fair test of BCA needs.", "at least 10; above 20 stops the test",
          "the power check's formula"),
    "N0": (r"N_0", r"the seed count before rounding up.", "0 or more", r"computed from \(\sigma\)"),
    # ---- operations
    "op_sum": (r"\sum", "add up: the sum over the counter written below it, from the value below to the value above.",
               "", "arithmetic"),
    "op_min": (r"\min", "the smallest of the values listed (or of the values over the counter below it).", "",
               "arithmetic"),
    "op_max": (r"\max", "the largest of the values listed.", "", "arithmetic"),
    "op_mean": (r"\operatorname{mean}", "the average, over the counter written below it.", "", "arithmetic"),
    "op_abs": (r"|\cdot|", "the size of a number, without its sign.", "", "arithmetic"),
    "op_norm": (r"\lVert\cdot\rVert", "the largest stretch a table of numbers can apply to any list of numbers.", "",
                "arithmetic"),
    "op_ceil": (r"\lceil\cdot\rceil", "round up to the next whole number.", "", "arithmetic"),
    "op_pos": (r"[\cdot]_+", r"the positive part: \([z]_+\) is \(z\) when it is positive and 0 otherwise.", "",
               "arithmetic"),
    "op_tanh": (r"\tanh", r"a squashing curve: \(\tanh\) of any number lies between −1 and 1, so "
                r"\(0.8\tanh(u/0.8)\) stays inside ±0.8.", "", "arithmetic"),
    "op_clip": (r"\operatorname{clip}", r"cut to a range: \(\operatorname{clip}(v, -1, 1)\) raises anything below −1 "
                "to −1 and lowers anything above 1 to 1.", "", "arithmetic"),
    "op_sign": (r"\operatorname{sign}", r"the sign: +1 for a positive number, −1 for a negative one.", "",
                "arithmetic"),
    "op_cossin": (r"\cos,\ \sin", "cosine and sine: the two coordinates of the point at that angle on a circle of "
                  "size 1.", "−1 to 1", "arithmetic"),
    "op_sqrt": (r"\sqrt{\cdot}", "the square root.", "", "arithmetic"),
    "op_sim": (r"\sim", "is drawn at random from.", "", "probability"),
    "op_normal": (r"\mathcal{N}", r"the normal (bell-curve) distribution: \(\mathcal{N}(0, \sigma^2)\) has mean 0 and "
               r"spread (standard deviation) \(\sigma\).", "", "probability"),
    "op_unif": (r"U[\cdot]", r"the uniform distribution: \(U[\text{low}, \text{high}]\) makes every value between "
                "the two ends equally likely.", "", "probability"),
    "op_approx": (r"\approx", "is about equal to.", "", "arithmetic"),
    # ---- numbers in the formulas
    "num_knee": (r"0.8", "the 80% limit: up to here push equals throttle; past it the propeller stalls.", "", ""),
    "num_slope": (SLOPE, "how fast the push falls past 80% in the stalling world: 1.5 units of push for each unit "
                  "of throttle.", "", ""),
    "num_gain": (GAIN, "the push factor: every push, from a thruster or from the water, is scaled by it.", "", ""),
    "num_tcost": (TCOST, "the throttle cost weight: the throttle cost is this times the sum of the squared "
                  "throttles.", "", ""),
    "num_turn": (TURN, r"how far the wave turns per step, in radians (a whole turn is \(2\pi\)).", "", ""),
    "num_gamma": (GAMMA, "the discount: a reward one step later counts this much.", "", ""),
    "num_damp": (DAMP, "normal driving damps the floats' swinging only this share as hard as the textbook rule.",
                 "", ""),
    "num_wob": (WOB, "the size (spread) of the wobble in normal driving.", "", ""),
    "num_jit": (JIT, "the size (spread) of the jitter on the full-throttle pattern.", "", ""),
    "num_quarter": (r"\tfrac{1}{4}", "the average over the 4 thrusters.", "", ""),
    "num_blend": (BLEND, "the largest extra caution BCA adds: doses stay below 1 + 0.5.", "", ""),
    "num_power": (r"2.128,\ 0.842", "the standard normal values for a 5% false-alarm rate shared over three tests "
                  "and for an 80% chance of detecting a real effect.", "", ""),
    "num_125_3": (r"1.25,\ 3", "1.25 enlarges the measured spread by 25%; 3 points is the smallest effect to detect.",
                  "", ""),
    "num_10": (r"10", "the fewest confirmation seeds the test uses.", "", ""),
}

GREEK = set("alpha beta gamma delta Delta epsilon eta theta kappa lambda mu nu pi rho sigma phi omega".split())
OPS = {"sum": "op_sum", "min": "op_min", "max": "op_max", "tanh": "op_tanh", "cos": "op_cossin",
       "sin": "op_cossin", "lceil": "op_ceil", "lVert": "op_norm", "lvert": "op_abs", "sim": "op_sim",
       "approx": "op_approx", "sqrt": "op_sqrt"}
SKIP = {"rceil", "rVert", "rvert", "dots", "cdot", "frac", "tfrac", "times", "ge", "le", "geq", "leq", "to",
        "textstyle", "displaystyle", "scriptstyle", "underbrace", "overbrace", "phantom",
        "cdots", "ldots", "infty", "pm"}
WORDS = {"amplitude": "amplitude", "points": "points"}
FILLER = {"for", "low", "high", "rec", "top", "new", "ref", "true", "cap", "actor"}


def _group(s, i):
    """read one TeX argument at s[i]: a braced group, a command or one character; returns (text, next index)"""
    while i < len(s) and s[i] == " ":
        i += 1
    if i >= len(s):
        return "", i
    if s[i] == "{":
        depth, j = 0, i
        while j < len(s):
            depth += {"{": 1, "}": -1}.get(s[j], 0)
            if depth == 0:
                return s[i + 1:j], j + 1
            j += 1
        raise ValueError(f"unbalanced braces in {s!r}")
    if s[i] == "\\":
        m = re.match(r"\\([A-Za-z]+|.)", s[i:])
        return m.group(0), i + len(m.group(0))
    return s[i], i + 1


def _scripts(s, i):
    """the subscript, superscript and primes after a symbol"""
    sub = sup = None
    primes = 0
    while i < len(s):
        if s[i] == "'":
            primes, i = primes + 1, i + 1
        elif s[i] == "_" and sub is None:
            sub, i = _group(s, i + 1)
        elif s[i] == "^" and sup is None:
            sup, i = _group(s, i + 1)
        else:
            break
    return sub, sup, primes, i


def _text(g):
    m = re.fullmatch(r"\s*\\(?:text|mathrm|operatorname)\{([^}]*)\}\s*", g or "")
    return m.group(1).strip() if m else None


def atoms(tex):
    """the symbols and operations in a TeX formula: (kind, name, sub, sup, primes, followed_by_paren, preceded_by_digit)"""
    s = tex
    s = re.sub(r"\\(?:big|Big|bigg|Bigg)[lr]?(?![A-Za-z])", "", s)
    s = re.sub(r"\\(?:left|right)(?![A-Za-z])", "", s)
    s = re.sub(r"\\(?:qquad|quad)(?![A-Za-z])", " ", s)
    s = re.sub(r"\\[,;:! ]", " ", s)
    s = re.sub(r"\\(?:begin|end)\{[a-z*]+\}", " ", s)
    s = s.replace("\\\\", " ").replace("&", " ").replace("{}", " ")
    out, i = [], 0
    while i < len(s):
        ch = s[i]
        prev = s[:i].rstrip()[-1:] if s[:i].rstrip() else ""
        if ch == "\\":
            m = re.match(r"\\([A-Za-z]+|.)", s[i:])
            name, i = m.group(1), i + len(m.group(0))
            if name in ("operatorname", "text", "mathrm", "mathcal"):
                arg, i = _group(s, i)
                sub, sup, primes, i = _scripts(s, i)
                if name == "mathcal":
                    out.append(("op", "normal", sub, sup, 0, False, False))
                elif name == "operatorname":
                    out.append(("op", arg.strip(), sub, sup, 0, False, False))
                else:
                    for w in arg.split():
                        out.append(("word", w, None, None, 0, False, False))
                continue
            if name == "bar":
                arg, i = _group(s, i)
                sub, sup, primes, i = _scripts(s, i)
                out.append(("sym", "bar" + arg.strip().lstrip("\\"), sub, sup, primes, False, False))
                continue
            if name in GREEK:
                sub, sup, primes, i = _scripts(s, i)
                follow = s[i:].lstrip()[:1] == "("
                out.append(("sym", name, sub, sup, primes, follow, prev.isdigit()))
                continue
            if name in OPS:
                sub, sup, primes, i = _scripts(s, i)
                out.append(("op", name, sub, sup, 0, False, False))
                continue
            if name in SKIP or not name.isalpha():
                continue
            raise ValueError(f"unknown TeX command \\{name} in {tex!r}")
        if ch.isalpha():
            sub, sup, primes, j = _scripts(s, i + 1)
            follow = s[j:].lstrip()[:1] == "("
            out.append(("sym", ch, sub, sup, primes, follow, prev.isdigit()))
            i = j
            continue
        if ch == "|":
            out.append(("op", "abs", None, None, 0, False, False))
        if ch == "]" and s[i + 1:i + 3] == "_+":
            out.append(("op", "pos", None, None, 0, False, False))
            i += 3
            continue
        if ch == "_" or ch == "^":  # a script on a bracket or a number: read it for its symbols
            g, i = _group(s, i + 1)
            out.extend(atoms(g))
            continue
        i += 1
    return out


def _idx_keys(g, ctx):
    """keys of the counters inside a subscript or an exponent"""
    if not g:
        return []
    return [k for a in atoms(g) for k in _sym_key(a, ctx)]


def _sym_key(a, ctx):
    kind, name, sub, sup, primes, follow, after_digit = a
    if kind == "word":
        return [WORDS[name]] if name in WORDS else []
    if kind == "op":
        k = {"normal": "op_normal", "abs": "op_abs", "pos": "op_pos", "clip": "op_clip", "sign": "op_sign",
             "mean": "op_mean"}.get(name) or OPS.get(name)
        if k is None:
            raise ValueError(f"unknown operation {name}")
        return [k] + _idx_keys(sub, ctx)
    tsub, tsup = _text(sub), _text(sup)
    keys = []
    if name == "c":
        if tsub in ("low", "high"):
            return ["c_range"]
        keys = ["c_ctrl" if follow or ctx.startswith("beyond") else "c"]  # in After the stop, c names the controller
    elif name == "w":
        num = re.match(r"\s*([12])", sub or "")
        keys = ["w1" if num and num.group(1) == "1" else "w2" if num else "w"]
    elif name == "a":
        if sup == "*":
            return ["a_star"]
        keys = ["a_thr" if ctx == "action-space" else "a"]
    elif name == "s":
        keys = ["s"]
    elif name == "p":
        keys = ["p"]
    elif name == "v":
        keys = ["v_vec" if ctx == "dynamics" else "v_speed"]
    elif name == "x":
        keys = ["x"]
    elif name == "e":
        keys = ["e"]
    elif name == "f":
        keys = ["f"]
    elif name in ("A", "B"):
        keys = [name] + _idx_keys(sup, ctx)
        return keys
    elif name == "R":
        keys = ["R"]
    elif name == "r":
        keys = ["r"]
    elif name == "J":
        if tsub in ("rec",):
            return ["J_rec"]
        if tsub in ("top",):
            return ["J_top"]
        if tsub in ("new", "ref"):
            return ["J_new_ref"]
        if sub == "\\gamma":
            return ["J_gamma"]
        keys = ["J"]
    elif name == "t":
        keys = ["idx_t"]
    elif name == "i":
        keys = ["idx_i_float" if ctx in ("rewards", "starting-state", "sim") else "idx_i_row"]
    elif name == "k":
        keys = ["k_stage2" if ctx == "beyond-stage2" else "k_pen" if ctx == "beyond" else "idx_k"]
    elif name == "j":
        keys = ["idx_j"]
    elif name == "n":
        keys = ["n_steps" if ctx == "methods" else "n"]
    elif name == "H":
        keys = ["H"]
    elif name == "K":
        if sub in ("s", "e"):
            return ["K_" + sub]
        if sub and "c" in sub:
            return ["K_col"]
        keys = ["K"]
    elif name == "u":
        keys = ["u"]
    elif name == "I":
        return ["I4"]
    elif name == "P":
        return ["P_cap"] if tsub == "cap" else ["P"]
    elif name == "Q":
        if tsup == "true":
            return ["Qtrue"]
        if tsub == "ref":
            return ["Qref"] + _idx_keys(sup, ctx)
        keys = ["Q"]
    elif name == "barQ":
        keys = ["Qbar"]
    elif name == "pi":
        if after_digit:
            return ["const_pi"]
        if tsub in ("new", "ref"):
            return ["pi_new_ref"]
        keys = ["pi"]
    elif name == "barpi":
        return ["pibar"]
    elif name == "y":
        keys = ["y"]
    elif name == "epsilon":
        keys = ["epsilon"]
    elif name == "L":
        return ["L_actor"]
    elif name in ("lambda", "alpha", "mu", "kappa", "beta", "eta", "nu", "theta", "rho", "phi", "gamma"):
        keys = [name]
    elif name == "m":
        keys = ["m"]
    elif name == "b":
        keys = ["b"]
    elif name == "d":
        keys = ["d"]
    elif name == "delta":
        keys = ["delta_star" if sup == "*" else "delta"]
        if sub == "A":
            return keys
    elif name == "o":
        keys = ["o"]
    elif name == "Delta":
        keys = ["Delta"]
    elif name == "z":
        keys = ["z_any" if ctx.startswith("beyond") else "z"]
    elif name == "S":
        keys = ["S"]
    elif name == "U":
        keys = ["op_unif" if ctx == "starting-state" else "U"]
    elif name == "W":
        keys = ["W"]
    elif name == "sigma":
        keys = ["sigma_start" if ctx == "starting-state" else "sigma_seeds"]
    elif name == "N":
        return ["N0"] if (sub or "").strip() == "0" else ["N"]
    else:
        raise ValueError(f"unknown symbol {name!r} (sub {sub!r}, sup {sup!r}) in context {ctx}")
    # the counters in the subscript (t, i, k, j) and in a superscript that is not a plain power
    keys += _idx_keys(sub, ctx)
    if sup and sup not in ("*", "2") and not tsup:
        keys += _idx_keys(sup, ctx)
    return keys


def symbol_keys(tex, ctx):
    """the entries a formula uses, in order of first use"""
    seen = []
    for a in atoms(tex):
        for k in _sym_key(a, ctx):
            if k not in SYMS:
                raise ValueError(f"no symbol entry {k!r} (from {tex!r})")
            if k not in seen:
                seen.append(k)
    return seen

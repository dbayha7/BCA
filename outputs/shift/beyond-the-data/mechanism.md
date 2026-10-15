# How BCA works, read from the code, and why it cannot yet reach harm outside the data

EXPLORATORY NOTE. Code reading only: nothing was trained, fitted or read out, and no BCA band, calibrator or dose was
computed. It changes no pre-registered rule. Line numbers are those of the files as they are now.

Terms used below:
- **critic**: the network Q(s, a) that rates a command a in state s. **actor**: the network pi(s) that picks the command.
- **temporal-difference (TD) residual**: y - Q(s, a), where y = r + gamma * Q_target(s', next action) is the
  one-step target built from a recorded transition (s, a, r, s').
- **behaviour cloning (BC) term**: the actor's squared distance from the recorded command, (pi(s) - a)^2.
- **conservative Q-learning (CQL)**: a critic loss that pushes down Q at commands the data do not hold.
- **weighted Bayesian conformal prediction (WBCP)**: the method that turns held-out residual sizes into one threshold.
- **scale network eta(s, a)**: BCA's small network that predicts how large the residual is at (s, a).
- **band** and **width**: BCA's half-width U(s, a) = R * u * eta(s, a) around Q. **dose**: the multiplier
  1 + 0.5 * U / (U + u) that BCA puts on one loss term.
- **lower edge of the band**: Q(s, a) - k * U(s, a) (a lower confidence bound).
- **H1, check 3**: the pre-registered harm check 1 (how wrong the gapped critic is) and harm check 3 (the room check:
  extra pull placed by the exact harm), as in `outputs/shift/stop-report/README.md`.
- **ReLU**: the rectified linear unit, max(0, x), used inside the networks; such networks extrapolate along straight
  lines outside the inputs they were trained on.

## The short answer

1. BCA reads its width only at **recorded (state, command) rows**: training rows to fit the scale network, held-out
   bank rows to set the threshold, and the training batch rows to read the dose. Nothing in the TD3+BC or CQL code ever
   evaluates the width at the actor's own command pi(s) or at a state outside the data
   (`algorithms/td3_bc_bca.py:181,257,284`; `algorithms/cql_bca.py:197,251-253,285-287`; `ALGORITHMS.md:335,488-489`).
2. **Nothing in BCA, as configured, makes the band wider away from the data.** The scale network is a plain
   two-layer network trained only on residual sizes at recorded rows; the threshold is one number for all inputs;
   the threshold's weights are uniform; the calibration bank has the same hole as the training data. Away from the
   data, eta is whatever the network extrapolates to.
3. The **ordering** of (state, command) pairs by width comes entirely from eta: the threshold R and the unit u are
   single numbers per refresh, so they can stretch every width but cannot rank one command above another
   (`ALGORITHMS.md:96-97,712-713`).
4. The residual BCA calibrates cannot, by itself, see a critic that is **consistently** wrong. A critic that overrates
   pushing past 80% has that overrating on both sides of its own target (y is built from the same critic's target
   copy), so the residual at recorded rows stays small. The residual measures how well the critic fits its own
   one-step targets, not how far it is from the true value.
5. Two consumption rules work **against** caution off the data: (i) a width that cannot be certified (infinite) gives
   dose **1**, the native weight, i.e. no extra caution (`calibration/dose.py:52-60,68-69`); (ii) the dose is bounded
   below 1.5 whatever the width (`calibration/dose.py:57-60`).
6. In TD3+BC the dose multiplies the pull toward the recorded command on that recorded row
   (`algorithms/td3_bc.py:134-138`). Check 3 already put the **exact** harm into that channel and found no room. So
   for TD3+BC, changing where the width is read is not enough; the channel itself has to change.
7. The one part of BCA's existing code that was built for a shift is unused: `wbcp.calibrate` accepts
   likelihood-ratio weights dP_test/dP_cal and a test mass (`calibration/wbcp.py:13-14,57,70-77`). With a test mass
   per point, the threshold grows, and reaches "cannot certify", as a point leaves the calibration data. That is the
   one route that would make BCA's band distance-aware by its own construction.

What needs to happen, in one line: read BCA's width at the actor's own command, consume it as an unbounded penalty in
critic value units (or as a guard at deployment) instead of as a bounded pull on recorded rows, treat "cannot
certify" as maximum caution, and either show that eta already rises off the data or switch on WBCP's
likelihood-ratio weights so the threshold does.

## 1. Exactly what is computed, where, and from what

### 1.1 The residual target (TD3+BC)

- Target: y = r + gamma * (1 - done) * min over the two target critic heads of
  Q_target(s', clip(pi_target(s') + noise)), noise drawn from a normal with standard deviation 0.2, clipped at 0.5
  (`algorithms/td3_bc_bca.py:146-156`). For the scale fit the noise uses its own folded key
  (`algorithms/td3_bc_bca.py:162-169`); the host critic update draws its own noise (`algorithms/td3_bc.py:106-115`).
- Prediction: q = min over the two **online** critic heads at the **recorded** pair (s, a), detached
  (`algorithms/td3_bc_bca.py:170-174`). (The TD3+BC actor itself reads only head 1: `algorithms/td3_bc.py:130`.)
- Residual: y - q. Its inputs are a recorded transition (s, a, r, s', done). The only place the actor enters is the
  target's next command pi_target(s') + noise at the **next** state; the command at s is always the recorded one.
- CQL: the same residual with CQL's sampled target (`algorithms/cql_bca.py:142-173`): one command sampled from the
  actor at s' (or the best of several), optional entropy term; q = min of the two online critics at the recorded
  (s, a) (`algorithms/cql_bca.py:185-190`).

### 1.2 The scale network eta(s, a)

- `Calibrator` (`calibration/network.py:8-31`): input = normalised state concatenated with the command
  (`calibration/network.py:17-22`); two hidden layers of 64 ReLU units (`:24-26`); output softplus(z) + 0.01
  (`:27,30`), so eta > 0.01 everywhere. TD3+BC and CQL build it state- and command-dependent
  (`algorithms/td3_bc_bca.py:132`; `algorithms/cql_bca.py:132`; in the shift test `experiments/shift/learners.py:1097`,
  with inputs already normalised).
- It is a function of (s, a) only. It has no notion of where the data are.

### 1.3 The scale fit: what eta is trained on

- Every update, on the training batch of recorded rows (`algorithms/td3_bc_bca.py:159-231`; CQL
  `algorithms/cql_bca.py:176-224`):
  - masses w_i: a **Bayesian bootstrap** draw over the batch rows, Exponential(1) values normalised to sum 1
    (`calibration/network.py:44-46`; used at `algorithms/td3_bc_bca.py:175-177`). They are random resampling weights,
    not likelihood ratios; there is no importance tilt (`ALGORITHMS.md:229-230`).
  - loss = (sum_i w_i * soft_coverage_i - 0.9)^2 + 0.005 * sum_i w_i * eta_i^2
    (`algorithms/td3_bc_bca.py:180-190`), where soft_coverage_i is a smooth indicator that y_i lies within
    q_i +/- eta_i * u (`calibration/network.py:34-37`). So eta is pushed to cover 90% of batch residuals, and the
    width penalty pushes it down everywhere it is evaluated, which is only at recorded rows.
  - unit u: exponential moving average (0.99) of the batch standard deviation of y - q
    (`algorithms/td3_bc_bca.py:196-198`). u is in critic value units, so the width is too.
- Only eta's parameters move; the critic and actor are untouched by this fit (`ALGORITHMS.md:127-130,251`).

### 1.4 The calibration bank

- Rows withheld from training: whole episodes are withheld, and a few rows from each form the bank
  (`calibration/reference.py:49-121`; sampler `calibration/bank.py:45-102`).
- In the shift test: 5 rows from each of the 200 calibration episodes, 1,000 rows
  (`experiments/shift/dataset.py:12-14,215-220`). The gapped removal **also deletes bank rows inside the removed
  ball** (`experiments/shift/shift.py:145-146,178-179`), so the bank has the same hole past 80% as the training data,
  and its states are recorded conditions only. The learner reads it as `prepared.bank`
  (`experiments/shift/learners.py:257,264`).

### 1.5 The threshold and its weights

- At a refresh, on every bank row: score_j = |y_j - q_j| / (max(eta_f(s_j, a_j), 1e-6) * u_f), with the scale network
  and unit frozen at that moment (`algorithms/td3_bc_bca.py:244-270`; `calibration/reference.py:152-194`, score at
  `:175`).
- `wbcp.calibrate` is called **without weights** (`calibration/reference.py:176-179`), which is the exchangeable case:
  every weight and the test mass equal 1 (`calibration/wbcp.py:66-69`).
- Two quantities are taken (`calibration/wbcp.py:57-117`):
  - lambda_hat: the plain 90% quantile of the scores (`:89-102`);
  - lambda_hpd: for each of 1,000 draws, Dirichlet(1, ..., 1) masses are put on the n scores plus one extra
    "test atom" placed at the worst case; the draw's crossing is the first score where the mass reaches 90%
    (`:43-54,104-111`); lambda_hpd is the 95th percentile of those crossings (`:115`).
  - R = max(lambda_hat, lambda_hpd) (`:117`). R is +infinity when nothing can be certified.
- "Bayesian" here means a posterior over the **distribution of held-out scores** (finite-sample doubt about the 90%
  quantile). It is not a posterior over the critic, the actor or the dynamics (`ALGORITHMS.md:89-91,694-698`).
- What the weights could be: the code's docstring defines them as w(x) = dP_test/dP_cal at the bank points and the
  test mass as E_test[w] (`calibration/wbcp.py:13-14`); passing them is supported (`:70-77`). They are not used
  today: "WBCP weights are uniform in the current design; a likelihood-ratio study would supply them as a separately
  declared change" (`ALGORITHMS.md:88-89`). Consequence: the threshold targets the behaviour (recording controller's)
  distribution, "not the learned policy's state-action distribution" (`ALGORITHMS.md:709-710`).

### 1.6 Refresh schedule

- The scale is fitted from the first update; the frozen reference is first used after update 10,000 and refreshed
  every 5,000 updates (`ALGORITHMS.md:262-263,664-665`; shift test `experiments/shift/learners.py:104-105,1162-1164,
  1217-1222`). Between refreshes the dose reads the **frozen** scale parameters and unit, not the live ones
  (`algorithms/td3_bc_bca.py:283-285`; `INTEGRATION.md:65-67`). Before the first refresh the width is 0 and the dose
  is 1 (`calibration/dose.py:90`; `calibration/reference.py:139-145`).

### 1.7 Width and dose

- Width: U(s, a) = R * u_f * max(eta_f(s, a), 1e-6) (`calibration/dose.py:74-76`; `ALGORITHMS.md:121,329`).
- Dose: 1 + blend * U / (U + u_f) with blend 0.5 (`calibration/dose.py:52-60`; `experiments/shift/learners.py:104`),
  so in [1, 1.5). The dose depends on U / u_f = R * eta_f(s, a): u_f cancels (`ALGORITHMS.md:352-354`).
- Infinite width (R = +infinity) is treated as "unsupported": the row gets dose **1** (`calibration/dose.py:52-53,
  68-69`; `ALGORITHMS.md:338`).
- Everything is detached: no gradient flows into the critic or actor through the width (`calibration/dose.py:62`;
  `algorithms/td3_bc.py:135-137`).

### 1.8 Where the dose enters the loss

- **TD3+BC**: the actor loss is -lambda * mean_i Q1(s_i, pi(s_i)) + mean_i m_i * mean over commands of
  (pi(s_i) - a_i)^2, with lambda = 2.5 / mean |Q1| and m_i the dose of the recorded row i
  (`algorithms/td3_bc.py:128-139`; `algorithms/td3_bc_bca.py:295-311`). The critic loss is unchanged
  (`algorithms/td3_bc.py:117-119`). Per row, a dose m_i acts like using alpha / m_i instead of alpha, so BCA can only
  move each row's effective alpha between alpha / 1.5 and alpha.
- **CQL**: per row, gap_i = clip(logsumexp over proposed commands of Q(s_i, .) - Q(s_i, a_i)); the proposals are 10
  uniform commands, 10 commands sampled from the actor at s_i and 10 from the actor at s'_i, all evaluated at s_i
  (`algorithms/cql.py:176-180,261-274,279-299`). BCA replaces mean_i gap_i with mean_i m_i * gap_i
  (`algorithms/cql.py:300-312`), where m_i is read at the **recorded** (s_i, a_i) (`algorithms/cql_bca.py:285-288`).
  The automatic penalty-strength update reads the unweighted gap (`algorithms/cql.py:342-347`). So CQL+BCA already
  pushes down Q at the actor's own commands, but how hard is set by the width at the recorded command, at most 1.5x.

## 2. What can be evaluated where

| Piece | Inputs | At any (s, a), including pi(s) or a state outside the data? |
|---|---|---|
| Critic Q(s, a), actor pi(s) | state, command | Yes (networks) |
| Scale network eta_f(s, a) | normalised state, command | **Yes**: a small neural network; frozen parameters live in `FrozenReference.cal_params` (`calibration/reference.py:31-38`) |
| Unit u_f, threshold R (lambda_hat, lambda_hpd) | none at readout | Yes, but they are single numbers, the same everywhere |
| Width U = R * u_f * eta_f(s, a) | state, command | **Yes**: `frozen_level_dose` takes any vector of eta values (`calibration/dose.py:65-95`) |
| Dose 1 + 0.5 U / (U + u_f) | width | Yes |
| Target y and residual y - q | reward r and next state s' | **No**: recorded transitions only (no r or s' at an unrecorded command) |
| Scale-fit masses | batch rows | Recorded training rows only |
| Scores and threshold | bank rows | Recorded bank rows only |
| Coverage meaning (about 90% of residuals inside the band, with 95% credibility) | | Only for pairs drawn like the bank: recorded conditions, recording controller's commands, gap removed (`ALGORITHMS.md:700-710`) |

So every piece needed to compute a width at (s, pi(s)) or at a new state already exists and needs no new machinery.
What does not transfer is the **meaning**: there, the width is the network's extrapolation, not a calibrated band.

## 3. Does anything make the band wider away from the data?

No. In order:

- **Scale network**: trained only on residual sizes at recorded rows, with a penalty that shrinks it wherever it is
  evaluated (`algorithms/td3_bc_bca.py:180-190`). No term rewards a large eta where there are no data, and no input
  feature measures distance to the data. Its value past 80% is ReLU extrapolation along the command inputs from
  commands up to about 80%; it may go up, down or stay flat. Whether it rises there is an empirical question that
  needs a band fit (user approval).
- **Masses in the scale fit**: Bayesian bootstrap resampling of the batch (`calibration/network.py:44-46`), not
  covariate-shift likelihood ratios.
- **Threshold**: one number per refresh (`calibration/reference.py:31-38`); weights uniform
  (`calibration/reference.py:176-179`); the bank has the same hole as the training data
  (`experiments/shift/shift.py:178-179`). The test atom only guards against too few bank rows (with roughly 850
  rows, the 1,000 minus those in the removed ball, its mass is about 1/850, far below 0.1), so it never fires here.
- **Ordering**: since R and u_f are global, the ranking of (state, command) pairs by width is eta's ranking alone
  (`ALGORITHMS.md:712-713`).
- **The residual itself**: the critic is trained to match the same targets the residual is computed from, so a
  consistent overrating of commands past 80% (H1: about 0.34 too high relative to capping, which hides the stall's
  cost) leaves no large residual at recorded rows. The one indirect trace is the target noise: the residual at row i
  includes how much Q_target(s'_i, .) varies within +/-0.5 of the target actor's command at s'_i
  (`algorithms/td3_bc_bca.py:147-155`). That reflects how rough the critic is near the actor's commands, not whether
  it is biased.
- **Consumption**: "cannot certify" gives dose 1 (`calibration/dose.py:68-69`), and the dose is bounded below 1.5
  (`calibration/dose.py:60`). Even a band that did widen off the data would, in the current dose, add at most 1.5x
  pull, and a band too wide to certify would add nothing.

The one shift-aware piece is unused: with weights w(x) = dP_test/dP_cal and a per-point test mass w(x), WBCP's test
atom gets more mass the less the bank resembles the test point; a draw in which the test atom holds more than 10% of
the mass never crosses, and once more than 5% of the 1,000 draws fail to cross the threshold is +infinity
(`calibration/wbcp.py:43-54,115`). That is distance awareness built into BCA's own
threshold, but it needs a density-ratio estimate between the actor's (state, command) pairs and the bank's, and a
per-point call (the code currently takes one averaged test mass, `calibration/wbcp.py:13-14`).

## 4. Channels: the minimal change for each, and whether it stays BCA

"Stays BCA" below means: the signal is BCA's frozen width U from the same scale fit, bank, threshold and refresh
(Procedures A to C, `ALGORITHMS.md:215-319`); only its consumer (Procedure D, already different for every host,
`ALGORITHMS.md:43-48,321-363`) changes. Every new consumer is a declared scientific change with a new method identity
(`ALGORITHMS.md:718-721`), needs its own pre-registration for a confirmatory claim, and any run of it here needs the
user's typed approval. Where the width is read at the actor's command, its coverage meaning does not carry over
(section 2).

**(0) Read the existing dose at (s, pi(s)) and keep the BC pull.** One argument in `bc_readout`
(`algorithms/td3_bc_bca.py:284`: `batch.action` -> `pi(batch.obs)`). Stays BCA. **Not worth doing**: check 3's
harm-placed arm already used the exact over-preference of the actor's command at each training row, which is the best
any reading point could give this channel, and it beat strength-matched, amount-matched and shuffled caution by only
-1.9 to +1.1 points. The pull toward recorded commands on recorded rows cannot carry the cost to commands past 80%.

**(a) Pessimism about commands the actor proposes, on the critic side.**
- CQL host: change the reading point of the per-row multiplier from the recorded command to the actor's mean command,
  about two lines in `algorithms/cql_bca.py:285-287`
  (`frozen_predictions = models[3].apply(cal_params, batch.obs, max_action * tanh(actor_mean(batch.obs)))`). A finer
  version weights each proposed command inside the log-sum-exp by its own width; that edits the host
  (`algorithms/cql.py:285-299`) to accept per-proposal weights. Stays BCA in both versions. Caveats: the multiplier is
  still capped at 1.5 on a penalty CQL already applies to every unrecorded command, so the first question is whether
  plain CQL on the gapped log already avoids the harm (there is no CQL learner in `experiments/shift/` yet; building
  one with no BCA is allowed). If plain CQL already avoids it, BCA can add only by placing CQL's caution better, for
  example letting the dose also go below 1 where the band is narrow; that changes the dose rule.
- TD3+BC host: TD3+BC has no critic-side term to multiply. The smallest addition is a width-weighted gap in the critic
  loss, `+ c * mean_i U(s_i, pi(s_i)) * (Q(s_i, pi(s_i)) - Q(s_i, a_i))` (`algorithms/td3_bc.py:117-119`, plus passing
  U in from `algorithms/td3_bc_bca.py:295-311`). That is a CQL-style term added to TD3+BC: the signal is BCA's, the
  host is changed, and the coefficient c is new. A lower edge in the Bellman target instead,
  `target_q - k * U(s', a_next)` at `algorithms/td3_bc.py:110-114`, is simpler but lowers values of recorded commands
  that lead to risky next states; it does not lower Q(s, command past 80%) directly, which is where the actor's choice
  goes wrong.

**(b) A penalty in the actor objective at the actor's own command (lower edge of the band).**
- Change: in `_actor_loss_fn` (`algorithms/td3_bc.py:128-139`) use `Q1(s, pi(s)) - k * U(s, pi(s))` with
  U(s, a) = R * u_f * max(eta_f(s, a), 1e-6) from the frozen reference (`calibration/dose.py:74-76`), eta's
  parameters held fixed but the gradient allowed to flow through the command input. About ten lines: a new optional
  argument on `td3_bc_update` like the existing `bc_multiplier` (`algorithms/td3_bc.py:100-105`), set from
  `algorithms/td3_bc_bca.update`. Drop the dose's bound: U is already in critic value units, so k = 1 means "act on
  the lower edge of the band" and needs no tuning. Decide whether lambda's normaliser includes the penalty.
- Stays BCA as a signal, but it breaks one design sentence: "host optimization does not backpropagate through the
  calibrator" (`ALGORITHMS.md:127-130`). The parameters stay frozen; only the command input is differentiated.
- What it needs to work in this world: at full-throttle-region training states,
  k * [U(s, command past 80%) - U(s, command capped at 80%)] must exceed about 0.34 in critic value units (the
  critic's hidden overrating, H1). That is a sharp, testable requirement on eta's ordering.
- Infinite width must become maximum caution (for example the command is treated as worst-rated), the opposite of
  today's dose 1 (`calibration/dose.py:52-53`).

**(c) Caution at made-up states beyond the data.** Q, pi and eta can all be evaluated at made-up states; there is no
recorded command there, so only (b)'s penalty (not a BC pull) can act. Change: a second batch of perturbed states in
`algorithms/td3_bc_bca.update`, penalised with (b). Needs a rule for making states (generic noise in normalised space,
or domain knowledge of which inputs are the current, which would leak the test's design). Stays BCA as a signal plus
new machinery. **Lowest priority**: the full-log critic was trained on the same recorded states and its actor scores
84.6 in strong current, so what is missing is the critic's belief about commands past 80% at the training states, not
coverage of new states.

**(d) A deployment-time check that falls back to a cautious controller.** No training change. In evaluation, at each
step compute U(s, pi(s)); if it exceeds tau, use the cautious controller (the cloned actor, 54.4 in strong current,
or pi(s) clipped to the recorded commands' range). tau by a declared rule from the bank, for example the 90% quantile
of U(s_j, a_j) over bank rows ("the band at the actor's command is wider than at 90% of recorded pairs"). Code: a
wrapper around the float64 policy (`experiments/shift/learners.py:371-373`) passed to `evaluate_policy`
(`:925-943`), which already does the same for its 80% cap (`RC.CappedController`, `:933`). Stays BCA (a readout
only). Shows "identify the harm and prepare for shift" but does not by itself improve policy optimisation; pair it
with (a) or (b) for that. Same open question as (b): does eta rank the actor's past-80% commands above recorded ones?

**(e) A support constraint.** BCA version: a hinge `c * mean_i max(0, U(s_i, pi(s_i)) - tau)^2` in the actor loss,
same place as (b), or a projection of pi(s) onto commands with U <= tau. Stays BCA as a signal. **Warning for the test
design**: in this world the harmful commands are exactly the unrecorded ones (after the removal only 0.04-0.14% of
kept rows show any throttle past 80%, `experiments/shift/SHIFT_ENV.md:214-216`), so a constraint that needs no BCA at
all, such as clipping each throttle to the recorded range, would essentially reproduce the stall-aware controller.
Any BCA channel must be compared with that baseline and with a distance-to-data guard, or the test cannot show what
BCA adds.

**(f) Making the band itself distance-aware (a signal change, not a channel).** Switch on WBCP's weights: estimate
w(s, a) = dP_actor/dP_bank with a classifier between the actor's (s, pi(s)) pairs at training states and the bank's
(s, a) pairs; pass the bank weights to `wbcp.calibrate` (`calibration/reference.py:176-179`); since the threshold
depends on a point only through its test mass w(s, pi(s)), compute R on a grid of test masses at each refresh and
interpolate at readout (a vector field added to `FrozenReference`, `calibration/reference.py:31-38`). Far from the
bank, R(s, a) becomes +infinity: "cannot certify". This is the "W" of WBCP doing what it was built for, and
`ALGORITHMS.md:88-89` anticipates it as a declared change. It adds a density-ratio estimator, and it only helps if
consumption treats +infinity as maximum caution (see (b)).

| Channel | Code size | Same BCA signal? | Can reach commands past 80% at training states? | Main risk |
|---|---|---|---|---|
| (0) dose read at pi(s), BC pull | 1 line | yes | no (check 3) | none; it just cannot help |
| (a) CQL multiplier read at pi(s) | 2 lines (or a host edit) | yes | yes, through CQL's own push-down | plain CQL may already avoid the harm; cap 1.5 |
| (a) width-weighted gap in TD3+BC critic | ~10 lines, host edit | yes, host changed | yes | new coefficient |
| (b) lower edge Q - kU at pi(s) | ~10 lines | yes; gradient through eta's command input | yes | eta may not rise past 80% |
| (c) made-up states | new state generator | signal yes, plus machinery | indirectly | leaks test design; not needed per key observation |
| (d) fallback at deployment | wrapper, no training | yes | yes (at use time) | not an optimisation gain alone |
| (e) support hinge | ~10 lines | yes | yes | trivial range clip is a strong rival here |
| (f) likelihood-ratio threshold | estimator + grid | yes (WBCP's own weights) | makes the band widen by construction | ratio estimate quality |

## 5. What needs to happen, in order

1. **Change the channel, not only the placement.** For TD3+BC, use (b) (lower edge at the actor's command, unbounded,
   in critic value units) and/or (d); for CQL, (a). Treat infinite width as maximum caution.
2. **Measure channel capacity without BCA first (allowed now, exploratory).** Put stand-in signals into the chosen
   channel: exact over-preference or the full-log critic (upper bound), distance to the 10th nearest training row in
   standardised (state, command) space, disagreement between critic heads or an ensemble, a random signal, a uniform
   signal, and the throttle-range clip. If no signal can recover the strong-current harm through the channel, BCA
   cannot either. If the clip or plain distance already recovers it, the world rewards any support signal and BCA
   must at least match them.
3. **Then ask BCA's signal the one question that matters (needs the user's typed approval):** fit the band on the
   frozen gapped critic (`experiments/shift/learners.py:1253-1279` already does this on recorded rows) and read
   U(s, pi_gapped(s)) against U(s, capped command) at full-throttle-region training states. A change of a few lines
   (read at `pi(s)` instead of `batch.action`). If k * the width difference exceeds about 0.34 there, (b) has a
   chance; if not, only (f) can make BCA's band see this harm.
4. **If eta does not rise, switch on (f).** It is the only version in which BCA itself, not an added distance
   measure, says "outside what I can vouch for".
5. Any confirmatory claim needs a new pre-registration with the new channel named, the range-clip and
   distance-guard baselines included, and the report-card bug fixed.

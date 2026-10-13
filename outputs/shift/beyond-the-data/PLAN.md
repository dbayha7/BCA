# Plan: what BCA needs in order to help with harm that lies outside the data

**EXPLORATORY PLANNING DOCUMENT.** It changes no pre-registered rule.
- Stage 1 uses no part of BCA. It may run now.
- Stage 2 fits and reads BCA's band and trains with it. It needs your typed approval and a new, labelled pre-registration first.

Numbers are means over gate seeds 0–4 at alpha 5 unless stated. Truth is exact simulation.

**Words used**
- **BCA (Bayesian Conformal Aggregation):** the add-on under test. It puts a calibrated band around the critic's one-step error.
- **Critic:** the network Q(s, a) that rates a command a in state s.
- **Actor:** the network pi(s) that picks the command.
- **Host:** TD3+BC, the actor-critic learner BCA is added to here. It trains the actor to climb the critic, plus a pull toward the recorded command (the **cloning pull**).
- **One-step error:** r + gamma · Q(next state, next command) − Q(s, a). It is the quantity BCA's band is fitted to cover.
- **Width U(s, a):** the band's half-width, in critic value units.
- **Dose:** today's multiplier on the cloning pull, between 1 and 1.5.
- **Lower edge:** Q(s, a) − k · U(s, a).
- **Unsupported command:** any thruster past 80% throttle. The gapped data almost never show one.
- **Recorded range:** per thruster, the commands between the 0.5% and 99.5% quantiles of the gapped training data. For thruster 1 this is about −0.75 to 0.51.
- **Points:** 0 is the recording controller and 100 the stall-aware controller.
- **Process-hour:** one CPU busy for one hour. At most 2 run at once here.

---

## Scoring of the three designs

Each design is scored 1–10 on six criteria:
- **Diagnosis:** is it correct against the evidence?
- **Reach:** can the proposed change reach harm outside the data?
- **Still BCA:** does it keep BCA's own signal?
- **Rigour:** room test first, matched untargeted controls, no outcome leakage, exploratory work labelled.
- **Feasible:** can it run on the 4 shared CPUs?
- **Clarity:** is it clear to you?

| Design | Diagnosis | Reach | Still BCA | Rigour | Feasible | Clarity | Overall |
|---|---|---|---|---|---|---|---|
| 1. Read the band at the actor's own command and act on its lower edge (uniform weights first) | 9 | 8 | 9 | 8 | 8 | 7 | **8.2** |
| 2. Calibrate the band for the actor with likelihood-ratio weights and train on a "certified" lower edge | 9 | 8 | 6 | 7 | 5 | 6 | 6.8 |
| 3. Experiment first: an oracle room test, then families of data-only signals choose the variant | 8 | 8 | 7 | 8 | 7 | 7 | 7.5 |

**Design 1 is the backbone of this plan.** Its strengths:
- It is the smallest change that keeps BCA's signal as it is: same residual, scale network, bank, threshold rule and settings.
- It fixes k = 1 by meaning, not by tuning.
- It requires a reproduction check before anything else.
- It adds a "push less everywhere" control and a random-surface control.
- It locks results blind before any return is computed.
- It requires fresh seeds before any confirmatory claim.

**Grafted from Design 2:**
- The room test's main oracle. It is the critic's exact one-step error at any command, computed through the simulator's differentiable step. That is the band a perfect, perfectly sharp BCA would have, and it needs no fitted surface.
- The likelihood-ratio version of BCA's own threshold, with a smooth penalty beyond the "cannot certify" limit, as the declared second version.
- Its stand-in readouts, already on disk.
- Its point that only a held-out half of the bank can audit coverage once the actor trains against the threshold.

**Grafted from Design 3:**
- A futility look on seeds 0 and 1 before spending on seeds 2–4.
- Detection measured in pairs: the same state, the actor's command against that command capped at 80%. A signal driven by the state alone cannot then pass for detecting the command.
- Selectivity between the stalling and no-stall worlds.
- Its measurement of the stall's faint trace in the data, which leads to the optional trace test bed (section 4.6).

**Corrections found when checking the designs against the evidence**
- **Design 3 misstates a number.** It says the gapped critic "rates full throttle +0.51 above 80%". It rates full throttle +0.06 above 80%. The +0.51 is how much it overrates it, since the truth is −0.46.
- **Design 1 gives the wrong share for harm-placed caution.** It says the share past 80% in strong current "stayed at 0.68" under harm-placed caution. The stop report gives 0.65 for harm-placed caution and 0.68 with no extra caution. The conclusion is unchanged.
- **Design 2 states its guarantee too strongly.** It says "an actor that maximises the lower edge never prefers an uncertified command". That holds only for an infinite penalty, and three things stand in the way:
  - its smooth penalty is finite;
  - the actor also has a cloning pull;
  - under the gapped actor's weights, the bank's effective size is only 2–156 of about 850 rows (its own readout), so the weights are far from exact.

  The certificate is an aim, not a property the method can be expected to have.
- **Designs 2 and 3 gate on the wrong control.** They require the oracle to beat a state-shuffled copy of itself by 15–20 points. The harm sits mainly in the command: past 80%, thruster 1 gives less push at every state. A penalty borrowed from another state may therefore keep much of the benefit, and a working channel would fail that mark. Here state-shuffling is reported, not used to gate.
- **Design 1's exact-harm oracle is a fitted network.** Fit error at the 80% knee could understate the room. Design 2's one-step oracle is computed exactly, and the simulator step it uses (`thruster_chain.dynamics_jax`) can be differentiated. It is the main oracle here.
  - Caveat: it hands the actor the true first-step consequence of the stall. Passing with it shows only that the channel can carry the information.
- **Design 3's oracle is approximate.** It uses the full-log critic as truth. That critic is within 0.06 of the truth (check 1), so this is acceptable but weaker than an exact oracle.
- **The new exploratory numbers in Designs 2 and 3 hold.** They were checked against their output files:
  - At training states, the gapped critic's one-step error at its actor's own past-80% commands averages −0.67 (seeds −0.77 to −0.52). The typical size elsewhere is 0.105.
  - Classifier-ratio limit: catches 79% of harmful proposals at training states, with 1.3% false flags. In strong current it catches 47% (seeds 30–82%), with 0.9% false flags; ranking score 0.86.
  - Two-head disagreement ranks weakly: 0.65 at training states, 0.75 in strong current.
  - 34–59 past-80% rows survived the removal. Their mean signed one-step error is −0.10 in the stalling world and +0.12 in the no-stall world.
- **The room test uses a different critic from check 3.** All three designs run it on check 2's frozen fixed-controller critic. Check 3 used the frozen co-trained TD3+BC critic, whose no-caution actor scores −10.1 in strong current.
  - So the comparison is with check 2: −14.3 for the gapped critic's actor, 98.9 points of harm.
  - A replication on check 3's critic is included.

---

## 1. What we learned: why BCA as built cannot help with this harm

**The harm is one wrong belief, held at states that are in the data.**
- At training states in the full-throttle region, the gapped critic is right up to 80% throttle and wrong past it:
  - at 90% throttle it says +0.04 against 80%; the truth is −0.21;
  - at full throttle it says +0.06; the truth is −0.46.
- Its slope along thruster-1 throttle there is +1.1, against a true −3.9.
- The actor learns this at those training states. It pushes past 80% on 42% of region rows; the full-log critic's actor does so on 10%.
- It then carries the belief outward. In strong current it pushes past 80% on 73% of steps and scores −14.3, against 84.6 for the full-log critic's actor: 98.9 points of harm.
- The full-log critic was trained on the same states and saw no strong current either, yet its actor generalises. Capping the gapped critic's actor at 80% at evaluation gives 101.8.

So the fix must change what the actor is told about unsupported commands at the training states, or stop those commands when the actor is used. It does not need data from new states.

**BCA never looks where the harm is, and acts through a lever that cannot move it.**
- It reads its width only at recorded (state, command) rows. There the gapped critic is accurate: error 0.07, against 0.12 for the full-log critic.
- The harmful pairs were about 99% deleted, and nothing in the training code reads the band at the actor's own command.
- The quantity BCA calibrates is large exactly at the harmful pairs. At the actor's past-80% commands the critic's one-step error averages −0.67, against a typical 0.105. BCA simply never evaluates it there.
- Its dose multiplies the pull toward the recorded command, which sits at 0.42–0.51 throttle. That says "drive like the recorder", not "do not go past 80%".
- Check 3 put the exact harm into that lever. Against matched controls it moved strong current by −1.9 to +1.1 points; factors up to 2 did no better. Uniform ×3 caution recovers 17 of 99 points and costs 5–13 elsewhere.
- Two consumption rules work against caution: a row BCA "cannot certify" gets dose 1 (no caution), and the dose stays below 1.5.
- Nothing in BCA makes the band wider away from the data:
  - the scale network is fitted only at recorded rows, with a penalty that shrinks it;
  - the threshold is one number with equal weights;
  - the calibration bank has the same hole past 80%.

**The honest ceiling in this test bed.**
- Any limit to the recorded range is free here. Clipping every command into it scores 91.0 in strong current and 93.0 in recorded conditions.
- In the no-stall world, where the missing commands help, the same clip costs 61.8 points and a distance-to-data check costs 25.7.
- The gapped data are nearly identical in the two worlds. Only 34–59 surviving rows past 80% (0.1%) carry a trace of the stall: their signed one-step error is −0.10, against +0.12 in the no-stall world.

So from these data the most BCA can claim is this: it flags where the actor relies on commands the critic cannot vouch for, and acting on that flag repairs the harm while keeping the policy improvement. A claim that BCA "identifies the harm itself", rather than mere novelty, needs a test bed in which the data carry some trace of the harm.

---

## 2. What needs to happen

1. **Read the band where the actor acts.**
   - In training, at (training state, the actor's own command); at deployment, at the command about to be executed.
   - Why: the harm lives at those pairs, which the recorded-row readout never visits.
2. **Act on the value the actor climbs, not on the cloning pull.** The actor climbs the lower edge Q − k · U at its own command.
   - Why: only a term whose slope along the command can cancel the critic's false slope past 80% (+1.1 against −3.9) can steer the actor away. Check 3 showed that the pull cannot.
3. **Drop the 1.5 cap in this channel and fix k = 1 in advance.**
   - Why: the width difference must outweigh about 0.5 value units of false preference. Tuning k on returns would leak the outcome into the method.
4. **Treat "cannot certify" as the strongest caution.**
   - Why: today it means no caution, the opposite of what a refused certificate says.
5. **Let the gradient pass through the frozen scale network's command input, with its parameters still frozen.**
   - Why: without it the width cannot steer the actor's command.
   - This breaks the design sentence "host optimisation does not backpropagate through the calibrator", so it is declared as a new method version.
6. **Show that the width actually rises at unsupported commands before training with it.** If it does not, switch on the likelihood-ratio weights that BCA's own threshold rule already accepts, as a separately declared second version.
   - Why: nothing in BCA as fitted forces the band to widen off the data. The weighted threshold is BCA's own route to "cannot certify" there.
7. **Add a deployment check that reads the same band, with a fallback that clips the command into the recorded range.**
   - Why: it acts directly at unrecorded states. The clip keeps the policy's gains, whereas handing control to the cloned actor costs 37–51 points in big kicks.
8. **Score everything in both worlds, against a plain recorded-range limit and against data-only stand-ins.**
   - Why: this world rewards any range limit, and the data barely tell the two worlds apart. Without these rivals no test can show what BCA adds.
9. **Show room in the new channel without BCA first. Then write a new labelled pre-registration, get your typed approval, and confirm on fresh seeds 5–9.**
   - Why: this is the gate's own standard. The channel was also chosen after seeing seeds 0–4.
10. **Only for a "harm, not novelty" claim: use a test bed whose data carry a trace of the harm.**
    - Why: with 0.1% of rows carrying it, no signal from these data can separate harmful commands from merely unrecorded ones.

---

## 3. The candidate BCA variants (same signal, new channel)

**What every variant shares**
- The signal is BCA's, unchanged: the same one-step-error residual, the same scale network eta(s, a) and its fit, the same calibration bank, the same threshold rule (weighted Bayesian conformal prediction) and the same settings:
  - miscoverage 0.1, credibility 0.95, 1,000 draws;
  - scale learning rate 0.001, coverage sharpness 20, width penalty 0.005, unit averaging 0.99.
- On a frozen critic it is fitted as in `learners.bca_band`: 20,000 scale-fit steps, then one refresh on the bank. In the co-trained host it follows BCA's own schedule: first used after update 10,000, then refreshed every 5,000 updates.
- That fit gives:
  - a frozen scale network eta_f;
  - a unit u_f;
  - a threshold R = max(plain 90% quantile, credible quantile).
- The width, unchanged (`calibration/dose.py`):

  U(s, a) = R · u_f · max(eta_f(s, a), 10^-6)

**Variant A, primary: the lower edge in the actor's objective (TD3+BC host).**

  L(phi) = −lambda · (1/B) Σ_i [ Q1(s_i, pi_phi(s_i)) − k · U(s_i, pi_phi(s_i)) ] + (1/B) Σ_i (1/4) Σ_j (pi_phi(s_i)_j − a_ij)²

  lambda = alpha / stopgrad( (1/B) Σ_i |Q1(s_i, pi_phi(s_i))| )

- alpha = 5, the gate's value. k = 1, fixed: the lower edge of the band.
- The normaliser is computed on Q alone.
- The cloning pull stays at weight 1 on every row, and the critic loss is unchanged.
- eta_f's parameters get no gradient; the gradient flows through its command input.
- Requirement: at full-throttle-region training states, U(s, full throttle) − U(s, 80%) must be at least about 0.5. That is the critic's false preference there; it is 0.34 averaged over the region's past-80% pushes.
- Code: about 15 lines. A new optional argument sits next to the existing per-row cloning multiplier, and the frozen reference is reused unchanged.

**Variant A-weighted, the declared second version.** It is used only if the detection test (section 5.2) fails with uniform weights. The loss is the same, with U replaced by:

  U~(s, a) = R_c(w(s, a)) · u_f · max(eta_f(s, a), 10^-6) + U_max · softplus( log w(s, a) − l* )

- **The weights w(s, a):** the likelihood ratio between the actor's pairs and the bank's.
  - A classifier g separates target pairs (s, clip(pi(s) + a_rec − c(s))) from recorded pairs (s, a_rec) at training rows, where c is the cloned actor. The target pairs are the recorder's own scatter, re-centred on the actor's command.
  - w = g / (1 − g). It is refitted at every refresh.
- **The threshold:** R(t) is BCA's threshold rule with bank weights w_j = w(s_j, a_j) and test mass t. It is tabulated at each refresh on 48 log-spaced test masses.
  - R_c is R with +infinity replaced by its largest finite value.
  - l* is the log of the smallest test mass at which R becomes +infinity ("cannot certify").
  - U_max is the largest finite width over the bank.
- Beyond l* the penalty keeps growing, so refusal means maximum caution.
- The bank is split in two. One half sets the threshold and the other only audits coverage, because the actor trains against the threshold.

**Variant B: deployment check (no change to training).**

  a_exec(s) = pi(s)  if U(s, pi(s)) ≤ tau,  else  clip_j(pi(s)_j, lo_j, hi_j)

- tau is the 95% quantile of U at held-out recorded rows of the calibration split that are not bank rows.
- lo_j and hi_j are the recorded range.
- It is applied to Variant A's actor and to plain TD3+BC.

**Variant C: critic side.** It is used only if Variant A's room test fails.
- **TD3+BC host:** add to the critic loss

  beta · (1/B) Σ_i [U(s_i, a~_i) / u_f] · ( Q(s_i, a~_i) − Q(s_i, a_i) )

  where a~_i = clip(pi(s_i) + noise). The noise is normal with standard deviation 0.2, clipped at 0.5. beta = 1 is declared in advance.
- **Conservative Q-learning host** (a learner that pushes Q down at commands the data do not hold): read the multiplier at the actor's mean command instead of the recorded one, and replace the capped dose with 1 + U / u_f.

**What changes, and what follows.** All variants change only where the width is read, what it acts on, the cap, and what "cannot certify" means. Each is a new method version: its results do not speak for the current form of BCA, and the reverse holds too.

---

## 4. Stage 1, exploratory and without BCA

**Rules for Stage 1**
- Everything here is EXPLORATORY. No BCA band, calibrator, threshold or dose is fitted or read.
- Scripts go under `scratchpad/ood/room/` and outputs under `/home/user/bca/runs/shift_explore/room/`, each with a SHA-256 sidecar.
- Nothing under `experiments/shift` or `runs/shift` is written.
- Every arm, formula, ladder, matching rule, mark and seed goes into `spec.json`, which is hashed before any run.

### 4.0 Common protocol

**Setup.** Check 2's actor-only protocol, unchanged:
- gate seeds 0–4;
- the frozen fixed-controller gapped critic of the stalling world, whose scored policy is the stall-aware controller;
- start from the saved cloned actor;
- alpha 5;
- 20,000 actor steps in 4,000-step chunks;
- check 2's stream of batch rows.

**Actor loss for every arm.**

  −lambda · mean[ Q1(s, pi(s)) − k · W(s, pi(s)) ] + cloning pull (+ nu · extra pull, for the even-pull arms)

- lambda excludes W.
- W's parameters are frozen, so its gradient acts only through the command.

**Reproduction check.**
- The no-caution arm runs through the gate's own unchanged code. It must reproduce check 2's saved gapped-critic actor by parameter hash.
- The new program with W = 0 must then match that actor within 10^-5 on every parameter.
- If either fails, stop.

**The one-step oracle ("perfect band").** This is BCA's own residual, computed exactly at any command:

  delta*(s, a) = r(s, a) + gamma · min_h Q_h(s', c(s')) − min_h Q_h(s, a)

- (s', r) come from one exact simulator step from the row's state, in float64 and differentiable.
- c is the stall-aware controller and gamma = 0.98.
- Q_h are the frozen critic's two heads.

**Strength and lock.**
- Strength is the root-mean-square change of the command from the no-caution arm, on the 5,000 evaluation-split recorded states (check 3's rule).
- Every control is matched blind to the main oracle's strength with the gate's matcher, before any return is computed. Strength on region states is reported as a check.
- After training and matching, a lock file holding every actor's hash is written. Only then are exact returns computed, on the full 1,000-start banks of all four situations.

### 4.1 Room test for the lower-edge channel

This test decides whether the channel can carry the harm.

| Arm | Penalty W at the actor's command | Role |
|---|---|---|
| No caution | 0 | baseline (check 2's gapped critic's actor) |
| **Perfect band** | abs(delta*), k = 1 | **main oracle**: the band a perfect BCA would have |
| Perfect band, half strength | abs(delta*), k = 0.5 | dose-response, reported |
| Overrating half | max(0, −delta*), k = 1 | reported |
| Exact long-run harm | [e(s, a) − e(s, a_i)]_+, where e is Q1 minus the exact value of the stall-aware controller | check 3's exact harm, now read as a value; reported (see the note below the table) |
| Even extra pull | nu · mean_j (pi_j − a_ij)², ladder 0.1, 0.25, 0.5, 1, 2, then refined | **gating control**: check 3's caution at matched strength |
| Push less everywhere | c · mean_j pi_j², ladder, then refined | **gating control** |
| Random surface (two) | randomly initialised network of the scale network's shape, ladder, then refined | **gating control**: same kind of penalty, random placement |
| Underrating half | max(0, delta*) | the same information on the wrong side; reported |
| State-shuffled perfect band | abs(delta*(s_sigma(i), a)): another training state's penalty at this row's command | does state placement add anything? reported |
| Recorded-range hinge | 100 · Σ_j ([a_j − hi_j]_+² + [lo_j − a_j]_+²) | the data-only rival with no signal; reported |

The exact long-run harm is computed by a differentiable rollout of the stall-aware controller, truncated at 250 steps; at a discount of 0.98 the remainder weighs under 1%. If the seed-0 timing check shows that rollout is too slow, a fitted surface (as in Design 1) is used instead, with its fit error reported at past-80% pairs.

**Marks** (pooled over seeds 0–4, fixed in `spec.json` before any run):
- **Room:** in strong current, the perfect band beats each gating control (even extra pull, push less everywhere, each random surface) by at least 20 points, and is positive in at least 4 of 5 seeds.
- **Recovery:** the perfect band is at least 49.5 points above no caution in strong current, half of the 98.9-point harm.
- **Keeps the policy improvement:**
  - recorded conditions: at least no caution − 2, and at least the cloned actor + 25 (no caution scores 85.5, the cloned actor 56.3);
  - big waves and big kicks: at least no caution − 3.
- **Futility:** seeds 0 and 1 run first. If the perfect band beats the even extra pull by less than 5 points in strong current on both, stop and go to section 4.5.

**Other runs**
- **No-stall world:** no caution, perfect band (with that world's rising-thrust controller), even extra pull and recorded-range hinge, at the stalling world's locked strengths, not re-matched. This measures the price of caution where the missing commands help.
- **Replication on check 3's critic:** no caution, perfect band (next command from the host's own actor) and matched even pull. Check 3's saved arms are reused for comparison.

**Prediction, written now:**
- perfect band 60–100 points in strong current, against the cap-at-80% ceiling of 101.8 and the full-log actor's 84.6;
- matched controls within −5 to +10 of no caution.

### 4.2 Data-only stand-ins in the same channel

These set the bar BCA must meet. They run only if the room, recovery and cost marks pass. Each stand-in is a penalty W in the same slot, scaled to the perfect band's strength by blind matching (4-rung ladder, then refinement):

- **Recorded-range hinge:** as above.
- **Behaviour density:** [ ½ Σ_k ((a_k − c_k(s)) / sigma_k)² − q95 ]_+
  - c is the cloned actor;
  - sigma_k is a per-thruster spread fitted by maximum likelihood on training rows;
  - q95 is the 95% level on held-out recorded rows.
- **Classifier ratio:** U_max · softplus(log w_g(s, a) − l*). This is the score-free part of Variant A-weighted.
  - The limit l* is computed directly from the bank weights, without BCA's code.
  - The classifier is refitted every 4,000 steps with the arm's actor.
- **Distance to data:** [d10(s, a) − q95]_+, where d10 is the distance to the 10th-nearest training row in standard units, fitted as a surface.
- **Two-head disagreement:** abs(Q1 − Q2)(s, a) of the frozen critic.

**Mark, "captures the room".** A stand-in must meet all three:
- its matched gain over no caution in strong current is at least 50% of the perfect band's gain;
- it scores at least no caution − 3 in recorded conditions;
- it beats the even extra pull by at least 15 points in strong current.

**Reported:**
- each stand-in's no-stall loss at its locked scale;
- selectivity: stalling-world gain minus no-stall loss, compared with the range hinge.

**Expected:** the range hinge and behaviour density capture most of the room in the stalling world, and every stand-in costs points in the no-stall world. This sets the bar for BCA; it is not evidence for BCA.

### 4.3 The deployment-check channel (no training)

**What runs.** Extend the existing fallback diagnostic:
- actors: the saved gapped critic's actor and plain TD3+BC;
- both worlds;
- the same 200 starts;
- the clip fallback.

**Rules added, each against random switching at its own rate:**
- the perfect detector of past-80% proposals; this control is missing so far;
- the one-step oracle as a trigger: abs(delta*) at the actor's command above its 95% level on held-out recorded pairs;
- behaviour density;
- the classifier ratio;
- two-head disagreement.

**Mark, "the channel is open":**
- the perfect detector minus rate-matched random is at least 15 points, in strong current or in big kicks;
- the cost in recorded conditions is at most 3 points.

**Already known:**
- The distance check beats random by 20.2 ± 1.3 points in strong current and by 18.7 in big kicks.
- In the no-stall world it costs 21.2 points more than random.

**Caution.** With a clip fallback in the stalling world, clipping every step is free. A trigger therefore earns credit only by firing less where clipping costs, which is the no-stall world. Selectivity between the worlds is reported.

### 4.4 Detection rehearsal (no BCA)

Write and lock the detection script that will later score BCA, and debug it on stand-ins. Each design's existing readouts are reused where they match.

**Pair sets**, fixed and hashed now, with copies in the no-stall world:

| Set | States | Commands |
|---|---|---|
| T1 | 1,500 region training states | the gapped critic's actor's command, that command capped at 80%, the recorded command, the cloned command |
| T2 | 1,500 other training states | the same four |
| T3 | 1,500 strong-current states visited by the gapped critic's actor, and 1,500 visited by the stall-aware controller | the actor's command and its 80% cap |
| T4 | 5,000 held-out recorded rows | the recorded command |

**Labels.** A separate script that cannot see any signal computes them from exact simulation.
- **Harmful:** both hold:
  - the critic's preference for the command over its 80% cap exceeds the true preference by at least 0.1;
  - the cap is truly better by at least 0.05.
- **Unsupported:** any thruster is past 80%.

**Measures**
- **Paired widening:** the share of past-80% pairs where the signal at the command exceeds the signal at its cap.
- **Size,** for value-unit signals: the median of (signal difference) divided by (the critic's over-preference).
- **Ranking score:** the chance that a random harmful pair gets the larger signal than a random harmless one.
- **Catch rate and false flags** at the signal's 95% level on T4.
- **Selectivity:** flag rate at past-80% pairs in the stalling world minus the same rate in the no-stall world.

**Signals scored:**
- distance to data;
- behaviour density;
- classifier ratio;
- two-head disagreement;
- residual evidence: minus the mean one-step error of the 10 nearest training rows;
- the perfect band, as a sanity check.

**Bars carried forward** from the fallback diagnostic: the distance check catches 35% of strong-current past-80% proposals, the exact one-step check 67%.

### 4.5 Only if section 4.1 fails: the critic-side room test

- **The test:** retrain the gapped critic (TD3+BC host). Compare three versions:
  - a conservative gap weighted by the perfect band at sampled actor commands;
  - an even gap at matched strength;
  - no gap.

  Each critic is then scored with check 2's actor protocol. The marks are as in section 4.1.
- **A separate check:** run plain conservative Q-learning on the gapped log, with no BCA, to see whether it already avoids the harm. If it does, BCA in that host can add only better placement of caution that is already there.

### 4.6 Optional: a test bed in which the data carry a trace of the harm

**What it is for:** it makes "identifies harm, not novelty" testable.

**How it is built:** gapped logs that keep a small random share of the removed past-80% rows, 0.5% and 2%, in both worlds, seeds 0–2.
- The fixed-controller critic and check 2's actor are retrained on them.
- Scripts import the gate's code read-only.

**Questions asked:**
- Does the gapped critic still overrate pushing past 80%, with harm of at least 50 points in strong current?
- Does a residual-based stand-in (residual evidence) separate the two worlds with selectivity of at least 0.2, where the range hinge cannot?

**What follows.** If the answer to both is yes, this becomes the test bed for a later "harm" claim, under its own pre-registration.

### 4.7 Compute

| Part | Process-hours |
|---|---|
| Harness and seed-0 checks | about 0.7 |
| Room test, stalling world: about 25 arm-runs per seed with matching, plus exact evaluation of about 12 locked actors | about 1.5 per seed: 3 for seeds 0–1, then 4.5 for seeds 2–4 |
| Room test, no-stall world (4 arms) | about 2.5 |
| Replication on check 3's critic (3 arms) | about 2.5 |
| Data-only stand-ins (5 stand-ins × 4-rung ladders, both worlds) | about 10 |
| Deployment-check additions | about 2 |
| Detection rehearsal (exact labels for about 9,000 pairs per seed per world) | about 2.5 |
| **Core total** | **about 28**: about 14 hours on 2 processes. The decisive look (room test, seeds 0–1) takes about 2 hours. |
| Only if needed: critic-side test plus plain conservative Q-learning | about 23 + 5 |
| Optional trace test bed | about 30 |

These estimates scale the gate's own timings:
- check 2: about 37 minutes per seed;
- check 3: about 40 minutes per seed with 13 arms and matching;
- one critic: about 50 minutes.

Arms that take a simulator step per update are counted at about twice the cost of a plain arm.

### 4.8 What each outcome would mean

| Outcome | Meaning | Next |
|---|---|---|
| Room, recovery and cost marks pass, and a data-only stand-in captures the room | The lower-edge channel can carry this harm, and information in the data suffices in this world | Stage 2 detection test |
| The marks pass, but only the oracles capture the room | The channel is open, but nothing in the data points at the harm. BCA would have to beat every stand-in. | Stage 2 detection still worth running, with low expectations. Prepare the weighted version and consider section 4.6. |
| Recovery passes, the room mark fails | Caution helps, but where it is placed does not matter. A BCA result would be "more caution", not "identification". | No BCA training arm. Use the deployment check, and section 4.5. |
| Recovery fails: even the perfect band cannot recover half the harm | Value caution at training states cannot reach this harm in TD3+BC | Section 4.5. The deployment check remains. |
| **No channel has room:** lower edge, critic side and deployment check all fail against matched untargeted controls | In this test bed no placement of caution beats untargeted caution of the same strength. BCA cannot be credited with identifying this harm, however well it detects. | Report it plainly. The options are BCA as a reported alarm only (a detection test with no policy claim), or a test bed whose data carry a trace of the harm (section 4.6), under a new pre-registration. |

The evidence makes the last row unlikely. A perfect detector with the clip fallback already recovers 106 points. What is still unmeasured is how much of that comes from targeting rather than from the switching rate.

---

## 5. Stage 2, with BCA

Stage 2 needs your typed approval and a new, labelled pre-registration.

### 5.1 Before anything runs

**The pre-registration.** Its label is "BCA lower-edge consumer (new method version), exploratory on seeds 0–4, confirmatory on seeds 5–9". It names:
- the channel, k = 1 and uniform weights;
- the weighted version as a declared fallback, with its estimator fixed from section 4.4 and never chosen on returns;
- tau and the recorded range;
- every arm, control, mark and seed;
- the locked detection script;
- the fix for the report-card bug.

**Approval.** You give it at a terminal, in the gate's way.

### 5.2 Detection test

The band is fitted and read. No actor is trained with BCA.

**Fit.** BCA's band, settings unchanged, in both worlds:
- on each seed's frozen gapped fixed-controller critic. The residual target uses the stall-aware controller at the next state; this is adapted from `learners.bca_band`, which as written fits on a TD3+BC host critic;
- and, exactly as written, on check 3's frozen host critic.

**Readout.** Read the width only at the locked pair sets. Widths and labels are hashed separately and joined only afterwards.

**Marks** (pooled; each also in at least 4 of 5 seeds):
- **Direction:** on harmful past-80% pairs, the width at the command exceeds the width at its 80% cap in at least 75% of T1 pairs and at least 65% of T3 pairs. Chance is 50%.
- **Size:** the median over those T1 pairs of (width difference) / (the critic's over-preference) is at least 1, so that k = 1 would cancel the false preference.
- **Ranking:** a ranking score of at least 0.75 on T1 and T2 together, and at least 0.70 on T3.
- **Calibration sanity:** 5% ± 1.5% of T4 rows lie above tau.
- **Selectivity:** reported here. It is gated only in a trace test bed.

**Claim rule.** "BCA flags the actor's reliance on commands the critic cannot vouch for" may be claimed only if:
- direction, size and ranking hold;
- BCA's ranking score is no more than 0.02 below the best data-only stand-in's.

"Identifies the harm" may be claimed only with selectivity of at least 0.2 in a trace test bed.

**If direction or size fails:** declare the weighted version and repeat the detection test. No uniform-weight policy test runs.

### 5.3 Policy test

**Setup.** Actor-only, with section 4.0's protocol, in both worlds. Seeds 0–4 are exploratory.

**Arms:**
- no caution;
- **BCA lower edge, k = 1 (primary);**
- BCA lower edge, k = 2 (declared secondary);
- BCA's width taken from another training state (state-shuffled, reported);
- even extra pull, matched;
- random surface, matched;
- the best data-only stand-in from section 4.2, matched;
- the recorded-range hinge;
- the perfect band, as a reference.

At evaluation, Variant B runs on the no-caution and the k = 1 actors. It is compared with random switching at the same rate and with the distance check.

**Marks** (strong current unless stated):
- BCA (k = 1) beats the even extra pull by at least 15 points, and is positive in at least 4 of 5 seeds.
- BCA (k = 1) beats the random surface by at least 15 points.
- It keeps the policy improvement:
  - recorded conditions: at least no caution − 3, and at least the cloned actor + 25;
  - big waves and big kicks: at least no caution − 3.
- BCA is at least the best data-only stand-in − 5.
- The BCA deployment check beats rate-matched random by at least 10 points, with a cost of at most 2 in recorded conditions.

**Reported:**
- the no-stall cost of every arm;
- the share of steps past 80% on every thruster;
- the distance of visited pairs from the data.

### 5.4 The full host, and confirmation

**Full host.** TD3+BC is trained with the BCA lower edge for 100,000 updates, critic and actor together, with the band on BCA's own schedule. It is compared with plain TD3+BC and with TD3+BC plus the range hinge at matched strength, in both worlds.

**Confirmation.** A confirmatory claim needs fresh seeds 5–9, with new logs, critics and cloned actors made under the new pre-registration and every setting frozen. The marks are the detection marks plus the policy marks above.

**Compute:**
- detection test: about 3 process-hours (+2 for the weighted version);
- policy test on seeds 0–4: about 16;
- full host on seeds 0–4: about 22;
- fresh seeds 5–9: about 50.

In all, about 90 process-hours, about 45 hours on 2 processes.

### 5.5 Claim wording, fixed now

> "In strong current, BCA's lower edge recovers X of 99 points and keeps Y of TD3+BC's gain over cloning in the recorded conditions. A hand-set limit to the recorded range recovers Z. In the no-stall world, where the missing commands help, BCA costs W points and the range limit V."

---

## 6. Risks and open questions

**Risks**
- **The width may not rise past 80%.** The scale network is fitted only at recorded rows, shrunk by its width penalty, and extrapolates along straight lines. It may stay flat or even fall. The lower edge would then do nothing, or push the wrong way. That is why detection comes before any BCA-trained actor.
- **Coverage means nothing past 80%.** The band's coverage statement stops at the recorded distribution. With uniform weights, a width past 80% is extrapolation. The weighted version refuses instead, but that refusal comes from the ratio estimate, not from BCA's residuals. The share owed to the weights must be reported separately.
- **Ratio estimates are fragile at the edge of the data:**
  - under the gapped actor's weights the bank's effective size is 2–156 of about 850 rows;
  - the classifier's strong-current catch rate ranges from 30% to 82% by seed;
  - a mixture density caught 0–75%.
- **This world rewards any range limit.** BCA can at best match a hand-set hinge here, and every form of caution costs points in the no-stall world.
- **The actor may find gaps in the penalty.** Its gradient passes through a small network, so it may find unsupported commands where the width happens to be low: other thrusters, or the other sign. The share past 80% on every thruster and the actor's distance from the data are monitored.
- **k = 1 may be too weak.** If the width difference is small, nothing happens. k = 2 is only a declared secondary arm.
- **Frozen-critic results may not carry over to the co-trained host.** There the critic overrates by +1.05, and the actor's own commands enter the target. Section 5.4 tests this.
- **The perfect band uses information BCA never has:** the true first step. Passing with it only opens the channel.
- **Strength matching may be unfair in the region that matters.** Matched on recorded states, the controls may end up weaker or stronger in the region. Region-matched strengths are reported.
- **Multiplicity and leakage.** There are many arms, and seeds 0–4 have been seen. Everything on them is exploratory; only seeds 5–9 under the new pre-registration support a claim.
- **This is a new method version.** Differentiating through the scale network and turning refusal into caution change the method, so results do not speak for the current published form of BCA.
- **CPU contention** with the other job can stretch the times above.

**Open questions for you**
- **Which claim do you want?**
  - "BCA flags reliance on unsupported commands and acting on it repairs the harm": testable now.
  - "BCA identifies the harm, not just novelty": needs the trace test bed in section 4.6, which adds about 30 process-hours.
- Should the conservative Q-learning host be tested? Section 4.5 answers first whether plain conservative Q-learning already avoids the harm.
- The implicit Q-learning host may not show this harm at all, because its actor imitates recorded commands with weights and never proposes past-80% commands. Should it be dropped from this question?

---

## 7. The first concrete step

Build the room-test harness, then pass its checks on seed 0 before any return is computed.

1. **Write `/home/user/bca/runs/shift_explore/room/spec.json`**, labelled EXPLORATORY, with its SHA-256. It holds:
   - every arm and formula from section 4.1;
   - the ladders and the matching rule;
   - the room, recovery, cost and futility marks;
   - seeds 0–4.
2. **Write `scratchpad/ood/room/harness.py`.**
   - It copies, and does not edit, `learners.actor_step`, `_actor_only_loop` and `ActorOnly`, and adds the penalty slot W with per-arm coefficients.
   - It loads seed 0 with the pattern of `outputs/shift/stop-report/cards/cards_outside_gate.py`: `A.load_prereg(A.refuse_protected(A.RUNS_DEFAULT))`, `A.load_seed`, the saved cloned actor, and the gapped critic through `A.get_critic(..., 'gap', 'stall_aware', train=False)`.
   - It computes the one-step oracle with `thruster_chain.dynamics_jax` from the rows' float64 states.
3. **Run the seed-0 checks.** No return is computed in any of them.
   - (a) The no-caution arm reproduces check 2's saved gapped-critic actor by parameter hash.
   - (b) The oracle matches the existing readout: at the gapped critic's actor's past-80% commands at training states its mean is close to −0.73 (seed 0), and its median size elsewhere is close to 0.107.
   - (c) Its gradient along thruster-1 throttle is finite on both sides of the 80% knee.
   - (d) A 400-step run of every arm gives finite losses.
4. **Launch seeds 0 and 1,** one process each, at low priority. Each process:
   - trains every arm;
   - matches strengths;
   - writes the lock file;
   - only then computes exact returns.

   Then apply the futility rule. If it passes, run seeds 2–4 and the no-stall arms, and prepare the stand-ins and the detection script while they run.

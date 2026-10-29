# Placement full test: harm switch, safety certificate and value lower edge (test specification, draft 2)

**Draft 2 for the owner, 19 October 2026, revised after four reviews (section 13). Design only.** Nothing was
generated, trained, scored or run on ThrusterChain data for this design. The only computations were (i)
micro-benchmarks of a restated update step and of the rollout arithmetic on made-up arrays, outside the repository
(`placement_test/bench/`), and (ii) closed-form arithmetic on made-up numbers (`measure/floor_arith.py`,
`revise/revise_arith.py`, `revise/budget_arith.py`, and the reviewers' `review_cov/arith.py`,
`review_stats/marks_arith.py`, `review_feas/card_arith.out`). Every ThrusterChain number quoted below comes from an
earlier record, and every one of those is exploratory (data seeds 0-4). Draft 1 is kept as `revise/DESIGN_draft1.md`.

**How this document will bind.** Once you settle the decisions of section 14, this text becomes `PRESPEC.md` of a new
package `experiments/placement_test/`, with `prespec.json` beside it holding the same rules in machine-readable form.
It is frozen twice (section 9.4): `freeze_pilot.py` before the pilot, and `freeze_confirm.py` after the pilot report,
which may fill only the slots declared here (the margins, the size decision, the chosen caps, staged B's probe count,
C-RAND's coefficient, the budget). Every script refuses to run if a frozen file has changed. Runs on ThrusterChain data
wait for your approval at your own terminal, through the package's own gate built on the hosts' pattern (HOSTS
`experiments/shift_hosts/gate.py:1-34`). Pass marks, decision rules, the margin rule, seeds, banks and random streams
are fixed here, before any data are read. Nothing runs without your go.

**Labels.** The pilot on data seeds 0-4 is exploratory: those seeds were seen by every earlier study, and on the pilot
the margin is fitted from the pilot's own returns, so its A and B decisions are not blind (section 9.3). Only the
confirmatory stage, on fresh seeds and under a pre-registration you approve, can carry a claim.

**Source abbreviations.** MAIN = `origin/main`; CONN = `origin/claude/main-bca-connection-x4o46s`; HOSTS =
`origin/claude/shift-hosts`; HV2 = `origin/claude/shift-hosts-v2`; LIVE = `scratchpad/status/live_c264.html`; PLACE =
`scratchpad/placement/result.json`; REV = `placement_test/revise/` (`revise_arith.out`, `budget_arith.out`). Notes
beside this file: `infra_cost_notes.md`, `seed_registry.json`, `measure/MEASUREMENT_NOTES.md`,
`measure/floor_arith.out`, `bench/*.out`. Paths without a branch are on MAIN unless stated.

---

## 0. In one page

**What is tested.** Three places to use BCA, each tested in full, for return and for coverage, with online ThrusterChain
episodes that simulate deployment:

- **A, harm switch.** Leave the host alone by default. Switch on a plain caution action (clip to the recorded range, or
  a gated range hinge) only when the 5% lower bound of BCA's posterior gap, read on held-out residual scores of a
  declared group (full-throttle commands) against the rest, clears a margin fixed by a break-even rule.
- **B, safety certificate per deviation.** Fall back to the clip by default. Let a deviation (the host's own command
  past the recorded range, or one throttle rung) through only when a tail bound on that deviation's tries certifies
  that harmful tries are rare. Tested offline on held-out rows, and in a staged deployment that gathers its tries from
  a budget of online episodes.
- **C, value lower edge.** Train the actor on Q1 minus k times BCA's band width at its own smoothed command, with the
  band recalibrated on Q1, exact state-by-command weights, and "no value credit" where the band refuses. Tested always
  on and switched by A.

**Where.** Unifloral's TD3+BC (revision f2dc127) as the host, trained on the main study's gapped log in the stalling
world S and its no-stall twin N, plus two mixed worlds for A and a decision-level stall-slope ladder. Pilot on seeds
0-4 (exploratory); confirmation on fresh seeds 25-29 of the main generator, extended to 30-34 if the size rule asks.
Module W (recommended, decision D4) adds the known-shift weighting-test world and a new no-stall twin of it: the only
place where BCA's importance weights can earn credit, and the only place where C's band can act past 80% throttle.

**What decides.** Each option gets three separate verdicts (section 7.5): **placement works** (its primary return and
validity or coverage claims pass, and no plain alternative dominates it); **outcome reading earns its place** (the
score's outcomes matter; this credit is shared by every plain test on the same score); **BCA's own machinery earns
credit** (a BCA part, such as the Bayesian step, eta, the width or the weights, beats its matched plain rival on the
same score, after Holm's correction). All marks are in section 7, fixed now, with wording for both outcomes.

**What the record predicts (stated now so that surprises show).** A works, and its credit goes to outcome reading with
a break-even margin, not to the posterior. B can credit only the placement on the main log (with equal weights BCA's
tail gate is the exact binomial test); whether it releases the clip in N depends on N's harm rate, and offline it is
likely underpowered (section 8.5). On the main recorder C's band refuses every query past 80% by construction, so C is
refusal alone there; C is expected to be dominated by A's switched clip.

**Cost (section 9).** Main generator, per 5 seeds: about 21 card-hours on the laptop's graphics card for the pilot
(25 with a 20% allowance) and 19 (23) for a confirmatory block, or 170 (204) and 152 (183) CPU-hours where a training
environment exists; module W about 14 (17) card-hours per 5 seeds. Evaluation, replicates and online episodes add about
9 CPU-hours per seed. Stage 0 about 20-25 CPU-hours and 3-5 card-hours. Draft 1's 6-8 card-hours was too low: it
charged C's runs at the bare host rate and had fewer matched controls (section 9.2).

---

## 1. Words used here

Each term is defined here, at its first use in this document.

- **ThrusterChain**: the simulated system of five floats in a channel, pushed by a current and waves and held by four
  thrusters (MAIN `experiments/shift/thruster_chain.py:1-37`). Its dynamics are deterministic: the same start and the
  same commands give the same episode (`thruster_chain.py:13`).
- **Command**: the four thruster settings in [-1, 1]. **Throttle index** of a command: `t(a) = max_k |a_k|`.
- **Knee**: 80% throttle. **Stalling world S**: past the knee the thrust falls, `thrust(a) = sign(a)(0.8 - 1.5(|a| -
  0.8))`, so full throttle is harmful. **No-stall world N (twin)**: thrust keeps rising, so full throttle helps. At or
  below the knee the two worlds are identical (`thruster_chain.py:16-18`). **Stall slope sigma**: the 1.5 in that
  formula; `world_physics('stall', stall_slope=sigma)` sets it (`thruster_chain.py:143-149`); sigma = -1 is the
  no-stall world.
- **Episode**: one start state run for 100 steps. **Return J**: the sum of the 100 rewards (return units; closer to 0 is
  better). **Online episode**: an episode run through the environment's own arithmetic (`ThrusterChainEnv.step`,
  `thruster_chain.py:438-454`: the command is rounded to float32, then to float64, then clipped). Online episodes are
  how this test simulates deployment.
- **Situation**: the start law of a deployment. Four: recorded conditions, strong current (current 0.7-1.1, stronger
  than anything recorded; the scored situation), big waves, big kicks (`thruster_chain.py:87-92`).
- **Bank**: a fixed set of 1,000 start states of one situation. The **standard banks** come from
  `default_rng([20261012, 2, 9])` and share their random numbers across situations (`thruster_chain.py:331-336`).
  **Fresh banks** are new sets drawn the same way from a new stream (section 3.5).
- **Points**: `100 (J - J_rec) / (J_top - J_rec)`, with `J_rec` the recording controller's mean return on that bank (0
  points) and `J_top` the top controller's (100 points): the stall-aware controller in the stalling world, the
  rising-thrust controller in the no-stall world (MAIN `experiments/shift/truth.py:26-28, 339-354`). On any bank other
  than a standard one, the anchors come from `anchors_on` (section 11.2), never from `truth.anchors`, which rolls only
  the standard banks (`truth.py:330-349`).
- **Data seed** (or **seed**): one repeat of the whole experiment, with its own recorded log and its own trained networks.
  **Seed-world**: one seed in one world.
- **Recorded log**: 1,000 episodes of the recording controller, split by episode into training (0-499), calibration
  (500-699) and evaluation (700-999) (MAIN `experiments/shift/dataset.py:4-15`).
- **Recording controller**: normal driving `0.8 tanh(u / 0.8)`, which never passes 80% throttle; and, only where the
  water's push `c + w1` exceeds 0.565 (**hard water**), with probability 0.5 per step the full-throttle command
  `clip(P + 0.03 n, -1, 1)`, `P = (-1, -0.137, -0.585, -0.406)` (MAIN `experiments/shift/recorder.py:8-17`).
- **Gapped log**: the recorded log after the **ball** (the densest 15% of training commands, which holds almost all
  full-throttle commands) is cut from the training rows and the calibration bank (MAIN `experiments/shift/shift.py:52,
  111-123, 164-192`). **Kept rows**: rows whose logged command lies outside the ball. **Survivors**: full-throttle
  training rows left outside the ball (28-59 per seed-world). **Z(x)**: the recording controller's share of commands
  outside the ball at state x, by Monte Carlo (about 0.5 in hard water, about 1 elsewhere).
- **Gap rule, t_lo, t_up**: the outcome-free rule that finds the empty throttle band in the gapped training commands;
  `t_lo` (about 0.79) is its lower end (step 1 PRESPEC.md:51-63; code `step1lib.py:179-206`). **Full-throttle group**:
  rows with `t(a) > t_lo`. **Rest**: rows with `t(a) <= t_lo`.
- **Recorded range**: per thruster, the 0.5% and 99.5% quantiles of the gapped training commands (stage 1
  `harness.py:530-533`). **Clip** (the caution action and the fallback): when `t(a) > t_lo`, clip each thruster to the
  recorded range; otherwise leave the command alone (step 1 PRESPEC.md:192-199).
- **Host**: the offline learner whose actor is deployed and whose critic is read; here Unifloral's TD3+BC (section 3.1).
  **Actor** pi: the network that picks the command. **Critic**: the two networks Q1 and Q2 that predict the value of a
  command.
- **Residual score** (BCA's one-sided score): `s(x, a) = Q_h(x, a) - [r + 0.98 Q_h(x', pi(x'))]`, with
  `Q_h = min(Q1, Q2)` of the host's critics, `x'` the next state and `pi(x')` the host actor's command without noise.
  Positive means the critic predicted more than the step delivered (step 1 PRESPEC.md:133-144; `step1lib.py:216-221`).
  **Held-out**: scored by a critic that never trained on that row.
- **BCA**: Bayesian Conformal Aggregation. **WBCP**: its weighted Bayesian conformal calibration,
  `calibration/wbcp.calibrate` (MAIN `calibration/wbcp.py:57-117`): from scores and optional weights it draws 1,000
  posterior **crossings** (each a credible 90% upper quantile, with one extra atom for the unseen test point at the
  worst loss, `wbcp.py:43-54`) and returns a **threshold** R, the 95%-credible upper quantile; R = +inf (a
  **refusal**) means "cannot certify". Settings everywhere unless stated: miscoverage 0.1, credibility 0.95, 1,000
  draws. With equal weights a Dirichlet partial sum is Beta(k, n + 1 - k), so R is Wilks' (0.90, 0.95) tolerance limit:
  a **per-bank promise** (with probability at least 0.95 over the bank, coverage of a fresh score is at least 0.90),
  not only an average over banks (PLACE `theory.findings[3]`, `[9]`). With unequal weights credibility no longer
  equals confidence, so the per-bank promise must be measured (MAIN `experiments/wbcp/DEPENDENCE.md:10-12, 1114-1133`).
- **Posterior gap** and **L_gap**: for the harm switch, the 5% quantile of all 1,000 x 1,000 differences between the
  full-throttle group's crossings and the rest's (step 1 PRESPEC.md:146-154; `step1lib.py:226-276`). Shifting every
  group score by a constant c shifts every group crossing, and so L_gap, by exactly c.
- **Margin m**: the size the gap must clear, in units of the 90%-quantile gap. Its rule, a decision break-even (section
  3.6), is fixed now; its value is computed on the exploratory pilot and frozen before any confirmatory datum is read.
  **m_f**: the same break-even rule applied to another statistic f (the mean gap, the Hodges-Lehmann shift, the
  kernel-weighted mean gap), so that each rule switches at its own break-even. **m_hinge**: the rule applied with the
  gated hinge as the caution action. **m_on**: the rule applied to online tries (section 3.6).
- **Harm switch (A)**: on iff `L_gap > m`; when on, the caution action runs in that world; when off, the host runs
  untouched.
- **Fallback F**: clip, as above. **Deviation**: letting a command beyond what the fallback allows through. **Try**: a
  held-out row (offline) or an online step where that deviation was used, with its residual score. **Rung**: a
  0.05-wide throttle band above the recorded range, for one thruster and one sign.
- **Harm event of a try**: `s > tau`, with `tau = lambda_R + m`. Offline, `lambda_R` is the empirical 90% quantile of the
  rest's scores on the bank rows of calibration episodes that supply no try (so that tau does not depend on the tries);
  online, `tau_on = lambda_R,on + m_on`, with `lambda_R,on` from separate fallback episodes (section 5.3).
- **Tail gate**: certify a deviation iff, with 95% confidence, at most `alpha_B = 0.1` of its tries are harm events.
  **Exact tail gate** (primary on the main generator): BCA's tail posterior computed exactly in its Beta form, which with
  equal weights is the exact binomial (Clopper-Pearson) test (LIVE 10502-10517, 10533; PLACE `rank`, O8).
  **Shipped tail gate**: `wbcp.calibrate` with 1,000 Monte Carlo draws and BCA's own floor of 36 tries (HOSTS
  PRESPEC.md:266-268), reported beside it. **Evidence floor**: the fewest tries with which a gate can certify at all: 29
  for the exact gate at alpha 0.1 and 95%; 36 for the shipped gate.
- **Safety certificate (B)**: allow a deviation iff its tail gate certifies; otherwise the fallback runs.
- **Band**: BCA's interval `Q1 +- U` meant to hold the critic's one-step error `|y - Q1|` 90% of the time, `y` the host's
  own learning target. **Width** `U = R x u x max(eta, 1e-6)`: `eta` a small scale network, `u` a running unit, `R` the
  WBCP threshold (HOSTS PRESPEC.md:256-268). BCA refreshes the band 18 times, after outer steps 10,000, 15,000, ...,
  95,000; each band is then used for the next 5,000 updates while the networks keep moving (HOSTS PRESPEC.md:264-268).
- **Declared query law**: where the band's promise is made and where C uses it. States: kept-row states (steps of
  recorded-law episodes whose logged command lies outside the ball: the state law of the bank and of Unifloral's
  training batches). Command: `a = clip(pi(x) + xi, -1, 1)` with `xi ~ N(0, 0.2^2)` per thruster and no inner clip of
  xi (an inner clip at +-0.5 would put point masses at `pi(x) +- 0.5` that the recorder never has: 4.9% of queries would
  be refused for that alone, REV `revise_arith.out` section 7).
- **Exact state x command weights**: the density ratio of the declared query law to the kept bank rows' law,
  `w(x, a) = q_xi(a | x) Z(x) / beta(a | x)` for `a` outside the ball, with `beta` the recording controller's
  unrestricted conditional law and `Z(x)` as above; this equals `q_xi / beta_kept` with `beta_kept = beta / Z(x)`
  renormalized per state, and the state factor is 1 because the query states follow the kept rows' state law (the form
  of CONN `outputs/shift_iw/explore/channel/command_band_check.py:6-10, 112-116`). Point masses are matched component
  by component (the recorder's atom at -1 on thruster 1 in hard water; the query law's atoms at +-1 from the final
  clip). `w = +inf` (refusal) inside the ball or wherever `beta` has no mass at `a`.
- **Value lower edge (C)**: the actor maximizes `Q1 - k W`, with `W` built from the band width at its own smoothed
  command (section 4.3). **No value credit**: where the band refuses, the actor is credited no more than Q1 of the
  row's recorded command.
- **Covered-or-refused (COR)**: a draw is covered if `|y - Q1| <= U`, and counts as covered when refused; this is what
  the band promises. **Covered among certified (CAC)**: the covered share of the draws that were not refused.
  **Refused share (RS)**. **Soft refusal** (counted as refusal in RS and CAC only): a draw whose R equals the bank's
  largest score, or whose threshold-table lookup sits at the last finite grid mass, since R climbs to the largest score
  before it turns +inf (MAIN `experiments/shift_iw/coverage.py:20-24`).
- **Matched**: a control tuned, from strengths alone and before any return is computed, to move the actor as far as the
  primary arm does. **Strength**: the root-mean-square change of the actor's commands from the host's on 5,000 held-out
  recorded states (step 2 PRESPEC.md:47-50).
- **Plain rule**: any rule without BCA's posterior: split conformal, difference of means, rank test, exact binomial,
  Clopper-Pearson, Hoeffding, empirical Bernstein, HCOPE-style mean bounds (Thomas, Theocharous and Ghavamzadeh 2015),
  rejection sampling, support or range rules, plain caps.
- **World-blind rule**: a deployed rule that does the same thing in every world and reads no outcome. **W**, the
  world-blind set of this test: clip always; the gated hinge (k = 1) always; a cap at 0.80 always (labeled "knows the
  knee"); the best plain cap among 0.75, 0.85, 0.90 and 0.95 always, chosen on the pilot (section 4.1).
- **NET** of a deployed policy p: in strong current, per seed, `NET_k(p) = [P_S(p) - P_S(host)] + [P_N(p) - P_N(host)]`
  (the stalling gain minus the no-stall cost), and `NET = mean_k NET_k` (step 2 PRESPEC.md:207-214). A perfect world
  switch scored 106.1 in the record; clipping always 44.4 (ROUND1.md:13). **NET_mixed**: the same sum over the mixed
  worlds MW+ and MW-, each start scored in its own region's physics with that region's anchors (round 1's scoring,
  `routes/round1/scratch/mixed_waves/run_route.py:318-322`).
- **No-stall cost**: `P_N(host) - P_N(p)`.
- **Replicate**: a fresh draw of the evidence from fresh generator streams (same law, same episode structure, same
  frozen critic; its size varies as the actual evidence's would; section 6.4); used to measure error and coverage rates,
  which are probabilities over the draw of the evidence.
- **Boundary null**: replicates whose scores are shifted by one constant per seed-world so that the truth sits exactly
  at the edge of what the rule promises (A: the true 90%-quantile gap equals m; B: the true harm rate equals 1.1
  alpha_B). It is the hardest case a rule's error control must survive.
- **Design effect**: how much clustering of steps within episodes inflates the variance of a step-level rate,
  `1 + (m - 1) rho` for m steps per episode with within-episode correlation rho.
- **LB (statistics)**: a one-sided 95% lower bound over seeds, `mean - t(0.95, S - 1) x sd / sqrt(S)`; when sd = 0, LB =
  mean (HOSTS PRESPEC.md:540-542). **Holm's correction**: the step-down adjustment for several one-sided p-values, as
  the hosts apply it (HOSTS PRESPEC.md:543-548).
- **Dominated**: an option whose best variant is beaten by a plain alternative available to the same deployment by at
  least 5 NET points with LB > 0 (section 7.5).
- **Stop rule**: an integrity check that stops a phase; a stopped phase is never rerun or replaced. **Kill rule**: a
  mark, fixed in advance, that shelves a claim or an option.

---

## 2. Questions and claims (written before any data)

Every claim is judged on the confirmatory seeds; on the pilot the same rules are computed and labeled exploratory.
Every claim needs every seed: a missing, stopped or non-finite value fails every claim that needs it. Each claim names
the verdict it serves: **primary** (placement works), **outcome** (outcome reading earns its place) or **machinery**
(BCA's own machinery earns credit).

### 2.1 Option A: harm switch on a declared group or world

**Question.** When BCA's held-out residual score on a declared group (full-throttle commands past `t_lo`) is read against
the rest, and a plain caution action switches on only when the posterior gap's 5% lower bound clears the margin, does
the deployed system earn the return of a perfect world switch, beat every world-blind rule, keep its promise about
false switches and about the gap, and owe any of that to BCA's own parts rather than to plain tests at their own
break-even margins?

- **A-R1 (primary, return).** In online episodes, the switched clip beats the best world-blind rule and random
  switching at its own rate, and keeps the host's improvement in the other situations.
- **A-V1 (primary, validity and coverage).** At the boundary null, the switch turns on in at most 5% of replicate
  evidence sets. By the shift identity this is the same event as "L_gap lies at or below the true 90%-quantile gap", so
  A-V1 is also the coverage claim for L_gap.
- **A-V2 (reported).** Coverage of the quantity the posterior actually targets (the n-matched gap, section 6.4); the
  natural false-switch rate in N; an equal-shapes null; and every rival's size at its own boundary.
- **A-K1 (machinery: the Bayesian step).** BCA's posterior gap beats every plain test applied to the same scores, each
  at its own break-even margin.
- **A-K4 (machinery: weights that localize; mixed worlds).** A posterior weighted by closeness in wave size beats every
  plain or unweighted localizer, the world-level switch and the best plain cap, and loses most of its lead when the
  full-throttle scores are permuted among themselves.
- **A-K2 (outcome: the switch follows the harm).** On the stall-slope ladder, the switch beats the better of clipping
  always and never clipping, summed over the middle slopes.
- **A-K3 (outcome: held-out reading).** Reading held-out rows beats reading the in-sample survivors.

### 2.2 Option B: safety certificate per deviation

**Question.** When the fallback (clip) is the default and a deviation is let through only when a tail bound on its tries
certifies that harmful tries are rare, does the deployed system keep the fallback's safety where deviating harms,
release it where deviating helps, keep its promise at the boundary and in deployment, and owe anything to BCA rather
than to a textbook test?

- **B-R1 (primary, return and safety; offline evidence, declared group).** The certified map is no worse than the
  fallback in the stalling world and beats clipping always across worlds.
- **B-R2 (primary, return and safety; staged deployment, declared group).** The same, with safety judged on the
  episodes after the decision and release on the cumulative return of a 1,000-episode deployment that includes its
  reference and probe episodes.
- **B-V1 (primary, validity: the promise).** At the boundary null, offline and staged, certifications of a deviation
  whose true harm rate at the threshold actually used exceeds `alpha_B` occur in at most 5% of replicate try sets.
- **B-V2 (primary, validity: transfer to deployment).** Among deployed steps where the certified map departs from the
  fallback, the share whose residual exceeds tau (the certificate's own event) is at most `alpha_B` plus a small
  allowance.
- **B-K1 (outcome).** In the stalling world, the real tries hold more harm events than shuffled tries.
- **B-K2 and B-K3 (machinery: the weights; module W only).** With exact (B-K2) or estimated (B-K3) importance weights,
  BCA's weighted tail gate certifies more truly safe rungs than rejection sampling, weighted split conformal and a
  self-normalized weighted mean bound, each set to BCA's harm rate.

Declared now: with equal weights BCA's tail gate is the exact binomial test, so on the main generator (where tries are
either in the recorded law or come from the deployment law itself) no credit to BCA is possible; only the placement
can be credited (PLACE `red.challenges[11]` and `[13]`: 28,000 of 28,000 equal-weight decisions agreed with the
binomial test; `measure/MEASUREMENT_NOTES.md`, section B).

### 2.3 Option C: value lower edge at the actor's own command

**Question.** When the actor maximizes Q1 minus k times BCA's band width at its own smoothed command, with the band
recalibrated on Q1, exact weights and no value credit where it refuses, does the deployed actor recover return under
shift at a low no-stall cost and beat the plain actions a switch can deploy, does the band keep its coverage promise
where it is used (on average and bank by bank) without hiding behind refusal, and is any gain owed to the band's finite
width, eta, the Bayesian step or the weights rather than to refusal, support, strength or a plain hinge?

**Declared now: on the main recorder C is refusal alone past 80% throttle.** The recording controller never commands
past 80% except in its full-throttle mode, whose commands sit at the jittered point P, almost all inside the ball
(MAIN `recorder.py:9-17`; section 1). So `beta = 0` at every query past 80% except a thin shell beside the ball, where
the test mass is so large that WBCP returns +inf (`wbcp.py:43-54`). Past the knee C-BCA therefore equals C-F0 (no value
credit, no band) by construction, and at or below the knee the two worlds' physics are identical. On the main recorder
C-K1 to C-K4 can only be won by shaping below the knee; their expected result is "fail", and their wording is fixed
for that (section 7.4). Module W, whose recorder has a uniform random part that reaches past 80% (CONN
`outputs/shift_iw/explore/channel/command_le_train.py:73-84`), is where C's band can act past the knee.

- **C-R0 (kill: room).** An exact-error value penalty (oracle) beats the host by at least 20 points in the stalling
  world, strong current. Otherwise C's return claims are not judged.
- **C-R1 (primary, return).** C's best variant (always on, or switched by A) beats the best plain action that A's switch
  can deploy (the switched clip, the switched gated hinge, the switched matched hinge) by +5 NET.
- **C-R1b (reported, return, always on).** C beats the range hinge matched to its strength on NET.
- **C-V1 (primary, coverage: the promise where it is used).** At each declared refresh, the band of that refresh, on the
  networks of that refresh, is covered-or-refused at least 90% of the time at fresh draws from the declared law.
- **C-V2 (primary, coverage: usefulness).** The refused share (soft refusals included) is at most 50% and
  covered-among-certified at least 85%, overall and in the near-knee hard-water zone.
- **C-V3 (primary, coverage: the per-bank promise).** Over replicate banks, at most 5% give a band whose
  covered-or-refused share falls below 90%.
- **C-K1 (machinery: finite width).** C beats F0, the same refusal with no band, matched in strength.
- **C-K2 (machinery: eta).** C beats ratio-only, the same band with eta = 1, matched.
- **C-K3 (machinery: the Bayesian step).** C beats the same band with a weighted split-conformal threshold, matched.
- **C-K4 (machinery: beyond support).** C beats the matched support-only penalty (the exact log density ratio).
- **C-K6 (outcome).** C beats the same band built from shuffled outcomes, matched.
- **C-K5 and C-K1W to C-K4W (machinery; module W only).** Weighted C beats unweighted C, and C-K1 to C-K4 hold, under
  the known shift and past 80%.

---

## 3. Host, data, worlds and seeds

### 3.1 Host

- **Default host: Unifloral's TD3+BC**, imported unchanged after its SHA-256 is checked against `configs/unifloral.json`
  (revision `f2dc1278eae18ed3e22c92255119c54885414d17`; `td3_bc.py` SHA-256 `6f5ab6f9...`), through
  `experiments/shift_hosts/unifloral_import.py` (HOSTS: `prepare`, `args_for`, `make_transition`,
  `init_like_training_block`, `make_unifloral_step`, lines 215-285, 353-461; none of them calls a study gate).
  Unifloral's own defaults (alpha 2.5, batch 256, learning rate 3e-4, target noise 0.2 clipped at 0.5, two critic
  updates per outer step) with ThrusterChain's exceptions of the hosts study: discount 0.98, 100,000 outer steps in 40
  chunks of 2,500, the data as Unifloral's own `Transition`, done 0 on every row (HOSTS PRESPEC.md:138-159). The chunk
  loop restates `td3_bc.py:329-341` with the per-chunk split of the evaluation key, as the hosts do (HOSTS
  PRESPEC.md:211-213). Nothing is tuned.
- **What A and B read.** A and B are wrappers around the trained host. They read only its critic (forward passes) and
  wrap its actor at deployment; they never train anything. Their score uses the online critics and the online actor
  without noise, as step 1 did with its controller (step 1 PRESPEC.md:133-144). Reported beside it, for information:
  the score with Q1 alone, and the score with Unifloral's own noisy target (target networks, 32 fixed noise draws,
  `unifloral_td3_bc_bca.py:316-340`).
- **What C changes.** C restates Unifloral's `make_train_step` line for line, as the hosts did
  (HOSTS `unifloral_td3_bc_bca.py:11-18`), and changes only the actor loss line (`td3_bc.py:245`) to add the lower edge
  (section 4.3). There is no dose: the cloning term keeps multiplier 1 (the owner rejected the dose). BCA's per-step
  fit and its 18 refreshes follow the hosts' schedule (HOSTS PRESPEC.md:264-268), with Q1 in place of `min(Q1, Q2)` in
  the band's value (the `q = ... .min(axis=-1)` line, `unifloral_td3_bc_bca.py:348`, inside `_band_values` at :343-349;
  `band_parts` at :352-359 calls it) and the exact weights recomputed for the current actor at each refresh (as
  CONN `outputs/shift_iw/explore/channel/command_le_train.py:12` did).
- **Clone** (for the clone fallback, reported only): Unifloral's `bc.py` (SHA-256 `5f181ca7...`, pinned in
  `configs/unifloral.json`), loaded by a new `load_bc()` that restates `unifloral_import._import` and its XLA-flag repair
  (importing a Unifloral file resets `XLA_FLAGS`, `bc.py:19`) with the same fingerprint check, in a process of its own:
  `unifloral_import` refuses learners outside its list and prepares once per process (`unifloral_import.py:58,
  186-194, 218-227`).
- **Two environments.** Training in `/opt/bca-venv` (Python 3.10.12, numpy 1.26.4, JAX 0.6.2); data and scoring in
  `/opt/bca-shift-venv` (numpy 2.2.6, gymnasium 1.0.0). `experiments.shift` refuses to import under any other numpy
  (MAIN `experiments/shift/__init__.py:81-111`), and no process holds both a Unifloral network and the simulator; files
  cross by `handoff.py` with SHA-256 sidecars (HOSTS PRESPEC.md:191-197). **One declared exception**:
  `dynamics_train.py` restates `thrust_jax` and `dynamics_jax` (`thruster_chain.py:224-251`) in the training
  environment, from the A and B columns, knee and slope that the scoring environment hands off from `TC._columns` and
  `world_physics`. It is plain arithmetic only (no gymnasium, no import of `experiments.shift`), and a stage-0 test
  requires it to be bit-identical under x64 to `thruster_chain.dynamics_jax` on made-up states, compared across the two
  environments by hand-off files. Only the oracle arms use it (section 4.3).
- **Where training can run.** This container has no training environment (no `/opt/bca-venv`, no MuJoCo binaries,
  Python 3.13; `unifloral_import.prepare` refuses without the Dockerfile's variables, `unifloral_import.py:67-76,
  228-234`). Training runs on the laptop's card, or in a cloud training environment that is built first (decision D11).
- **Other Unifloral hosts (IQL, ReBRAC, CQL), added later, one pre-registered extension each, only after TD3+BC's
  verdicts.** A and B need per host only a score adapter: the host's own one-step target computed as its Unifloral
  file computes it, verbatim (IQL `r + 0.98 V(x')`; ReBRAC with its cloning term; CQL with its ten critics and entropy
  term; HOSTS PRESPEC.md:277-282), and its deterministic deployed command (CQL's mean action). C needs a restated step
  per host, with the lower edge in each host's own actor objective (ReBRAC: TD3's slot; IQL: inside the advantage of
  its weights; CQL: at sampled actions), the hosts' restating rules and a k = 0 reproduction test for each.

### 3.2 Data

- **Generator**: the main study's (master seed 20261012), the only one with a no-stall twin (seed_registry.json).
  `dataset.generate(rep, world)` builds 1,000 episodes of 100 steps through `env.step`; starts, noise and coin come from
  `default_rng([20261012, rep, 7])` and `[..., 8]`, so the two worlds share every random number (`dataset.py:152-158,
  176-226`). The calibration bank is 5 rows per calibration episode from `[..., 10]` (`dataset.py:214-220`).
- **Gapped log**: `shift.define_gap` and `make_removal('gap')`, as the gate does. The host trains on the gapped training
  rows (about 42,500 per seed-world).
- **Evidence for A (held-out rows).** Rows of the calibration and evaluation splits outside the ball and outside the
  bank. No critic trains on them. Full-throttle group: those rows past `t_lo`, plus the 0-2 bank rows past `t_lo`.
  Rest: the bank rows at or below `t_lo` (about 825-850), BCA's own calibration bank, as step 1 and the round-1 route
  (`routes/round1/scratch/heldout_calibration_rows/run_route.py:6-8, 102-109`).
  - On seeds 0-4 the calibration split alone held 14-28 full-throttle rows per seed-world (15-30 with the bank's; from
    13-25 episodes; MAIN `outputs/shift/beyond-the-data/restructure/routes/round1/scratch/_feasibility/count_rows_0.json`
    and `count_rows_1_2_3_4.json`). The evaluation split has 1.5 times as many episodes, so about 35-70 in all are
    expected [inferred; counted from commands alone, before any outcome is read, at the pilot's first step].
  - The calibration split alone (the round-1 route's evidence) is reported beside it for continuity.
- **Evidence for B.** B-GROUP's offline tries: the full-throttle evidence rows where the host's own command at the same
  state also passes `t_lo` with the same sign on the same largest thruster (so that the try stands for the deviation
  being certified, the host's command, not only the recorder's jittered P); the unconditioned rows and both counts are
  reported. B's rest for `lambda_R`: the bank rows at or below `t_lo` of calibration episodes that supply no try; the
  pooled form (all bank rows, as A) is reported beside it.
- **Survivors** (A-K3 only): the training rows past `t_lo`, scored by the host that trained on them.
- **Unifloral preparation**: a new `prepare` for the main gapped log in both worlds (and the mixed worlds), writing
  Unifloral `Transition` arrays, the bank, the evidence rows, their exact next states and the next commands' inputs,
  and, for C, every training and bank row's recorder mean `mu(x)`, hard-water flag and `Z(x)` (the training
  environment cannot compute them), with SHA-256 sidecars (the hosts' `prepare.py` covers only the weighting-test data,
  HOSTS `prepare.py:1-41`).
- **Leaks checked by code** (stop rule S9): no evidence or bank row is a training row; no evidence row lies in the ball;
  the gap rule reads commands only.

### 3.3 Worlds

| World | Physics | Log | Hosts trained | Used by |
|---|---|---|---|---|
| S, stalling | `world_physics('stall')`, sigma 1.5 | main gapped log | yes | A, B, C (primary) |
| N, no-stall twin | `world_physics('no_stall')` | main gapped log, same streams | yes | A, B, C (primary) |
| MW+, mixed by wave size | stall iff the start's wave amplitude > 0.25 | spliced log (below) | yes | A (A-K4) |
| MW-, mirror | stall iff amplitude <= 0.25 | spliced log | yes | A (A-K4) |
| Ladder | sigma in {-1, -0.5, -0.25, 0, 0.1, 0.2, 0.3, 0.5, 0.75, 1.0, 1.5, 3.0} | N's log, outcomes recomputed | no | margin rule (pilot), A-K2, B-V1 readouts (pilot and confirmatory) |
| M3, harm only beyond the log | N's host deployed in stalling physics | none | no | A, B information |
| M4, harm only inside the log | S's host deployed in no-stall physics | none | no | A, B information |

- **Why the mixed log is exact.** Wave amplitude never changes within an episode (`thruster_chain.py:11, 170-171`;
  largest jump 1.1e-16, `_where/feas_amp_0_1_2_3_4.json`) and both worlds' logs share every random number, so episode
  e of MW+ is episode e of the stalling log when its start's amplitude exceeds 0.25 and of the no-stall log otherwise.
  Its gap, ball and bank are recomputed on the spliced log by the same functions. An online episode in a mixed world
  uses its start's physics. Replicate blocks of a mixed world split their episodes by start amplitude across two
  `dataset.simulate` calls, one per physics (`simulate` takes one physics per call, `dataset.py:161-173`). (The round-1
  route spliced evidence rows in this way, `ROUND1.md:121-146`; this test also trains hosts on the spliced logs, which
  the route deferred.)
- **The ladder** is decision-level, as round 1's was (`ROUND1.md:66-96`): N's host and N's held-out rows are fixed; the
  one-step outcomes of the evidence rows (and of N's replicate blocks) are recomputed exactly under each sigma, and the
  value of caution is measured by online episodes of N's host in sigma-physics. No network is retrained. It runs in the
  pilot (margin rule) and again in the confirmatory stage (A-K2 and the readouts), where it needs no training.
- **M3 and M4** need no new training: they deploy existing hosts in the other physics (the shelved run-time monitor's
  worlds, `ROUND1.md:41-64`).

### 3.4 Seeds (an allow-list, enforced by `seeds.py`, stop rule S2)

- **Pilot: main-generator seeds 0-4, exploratory.** Seen by the gate, stage 2, steps 1 and 2 and round 1. Hosts are new
  (Unifloral), so nothing there is reused except the deterministic data, which are regenerated and checked against
  their records (copies in MAIN `outputs/shift/beyond-the-data/stage2_gpu/payload/files/runs__shift__pre__data__base__
  {stall,no_stall}__seed<k>__record.json`).
- **Confirmatory: main-generator seeds 25-29**, under a pre-registration you approve, with **30-34** admitted only if
  `size_decision.json` (written by the size rule of section 8.4 before any confirmatory datum exists) says 10 seeds.
- **Margin block 20-24**: admitted only if decision D3 chooses a fresh exploratory margin block.
- **Why 20-34 are free.** In the main generator, 0-4 are seen; 5-9 belong to version 6/7 (refused by
  CONN `near80/code/near80lib.py:70`); 10-14 are near-80 version 8's confirmatory seeds (`near80lib.py:69`); 15-19 are
  near-80's declared fallback for step 2's confirmation; 100-119 are the original gate's confirmation seeds; 900-909
  are stage 0's design seeds. A grep of all four branches finds no use of 20-34 in this namespace (seed_registry.json).
- **Module W (section 10)**: weighting-test generator (master 20261016) seeds 0-4 as pilot, 25-29 as confirmation
  (free: 10-14 are the four-learner study's, 15-19 arms A/B/C/F0's, 20-24 named and rejected; seed_registry.json). The
  same seed number gives different data in the two generators (HOSTS PRESPEC.md:613-615); every seed list in
  `prespec.json` names its generator and master seed.
- **Every other seed is refused**, whatever its number. `seed_registry.json` is committed with the package at the first
  freeze, so later studies can add these blocks to their own reserved lists (the `near80lib.py:68-70` pattern).
- **Every seed counts.** A seed whose run stops is reported and never replaced.

### 3.5 Random streams (new block, registered now)

The test's own randomness uses master seed **20261019** (unused on all four branches; 20261018 is used by HV2's
`probe_s7.py:300`). Every key starts with the master seed and a tag, has at least five words, and indexes worlds,
situations, replicates, permutations, decisions and ladder points from 1. numpy pads keys shorter than four words with
zeros, so `[20261019, 7, 25, 0]` and `[20261019, 7, 25]` give the same generator (checked in the scratch environment);
`seeds.py` asserts at import that all registered keys are distinct after padding to four words.

Codes: world w (S 1, N 2, MW+ 3, MW- 4, ladder 5); situation s (recorded 1, strong current 2, big waves 3, big kicks 4);
decision d (B-GROUP 1; B-RUNG level l = 1..5 as 10 + l); ladder point i = 1..12 in the order of section 3.3.

| Stream | Key |
|---|---|
| Unifloral `Args.seed` per seed k (every arm of seed k, in every world, shares it: same initial parameters and batches) | `default_rng([20261019, 11, k, 1, 1]).integers(2**31 - 1)` |
| BCA fit and refresh streams (C), then the hosts' fold-in rules (HOSTS PRESPEC.md:216-220) | s91 from `[20261019, 12, k, 1, 1]`, s39 from `[20261019, 13, k, 1, 1]`, each `.integers(2**31 - 1)` |
| WBCP posterior draws (A, B), group g (1 group, 2 rest): index 1 for the actual evidence, 1 + r for replicate block r = 1..1,000 | `[20261019, 1, k, w, 1 or 1 + r, g]` |
| Outcome shuffles, permutation j = 1..200 | `[20261019, 2, k, w, j]` |
| Random-switching coins (information only; the claim uses the expectation, section 7.2) | `[20261019, 3, k, w, s]` |
| Staged B: replicate reference-and-probe sets r = 1..1,000 of decision d (their union is the truth population) | `[20261019, 4, k, d, r]` (no world: S and N share starts) |
| Replicate held-out blocks (A, offline B) and C's replicate banks, replicate r: starts and noise, then coins | `[20261019, 5, k, r, 1]`, `[20261019, 5, k, r, 2]` (no world, as the generator, `dataset.py:152-158`) |
| Staged-B deployment stream of decision d (reference episodes, probes, then the rest) | `[20261019, 6, k, d, 1]` (no world) |
| C's declared-law draws (states, xi, target noise) at declared refresh j = 1..7 (in the order of section 5.1) | `[20261019, 7, k, w, j]` |
| Episode bootstrap | `[20261019, 8, k, 1, 1]` |
| Ladder point i | `[20261019, 9, k, i, 1]` |
| Tuning bank of B's oracle ceiling map | `[20261019, 10, k, w, 1]` |
| Recording controller's noise and coin on fresh banks and deployment starts (for `anchors_on`) | `[20261019, 14, k, s, 1]` |
| C-SHUF: bank permutation at refresh j = 1..18; eta-target shuffles by `fold_in(PRNGKey(seed), outer step)` with the seed from index 100 | `[20261019, 15, k, w, j]`; `default_rng([20261019, 15, k, w, 100]).integers(2**31 - 1)` |

**Scoring banks.** Pilot: the standard banks (comparable with every earlier number). Confirmatory: a fresh bank per seed,
`draw_starts(1000, default_rng([20261012, k, 9]), START_LAWS[situation])` for k in 25-34 (the standard bank's own rule at
the seed's index; tag 9 is used by the data streams only as the standard bank at k = 2, and no four-word key
`[20261012, k, 9, 0]` exists), plus the standard bank as a comparability readout. Points on a fresh bank use anchors
from `anchors_on` with the recorder stream above. Any try-gathering stream is disjoint from every scoring bank.

### 3.6 The margin rules (fixed now; values computed on the pilot and frozen before confirmation)

The 0.10 margin of the record was set after seeing seeds 0-2, for a different critic (step 1 PRESPEC.md:156-168,
366-368); it carried the false-alarm control (without it BCA netted 55.7, `ROUND1.md:34`). With a new host its scale
has no meaning. So the margin is set by a written break-even rule:

1. On each pilot seed k, take N's host, its held-out rows and its replicate blocks (generated before this step).
2. For each ladder sigma: the value of caution `V_k(sigma)` = mean over the strong-current bank of `J(clip wrapper) -
   J(host)` in sigma-physics (return units, online episodes); and the population gap `Gamma_k(sigma)` = 90% quantile of
   the scores of the replicate blocks' full-throttle rows minus that of their rest rows (one-step outcomes recomputed
   exactly under sigma; N's critic).
3. `sigma*_k`: the lowest sigma where `V_k` changes sign from at most 0 to above 0, by linear interpolation between ladder
   points. `m_k = Gamma_k(sigma*_k)`, interpolated the same way.
4. **m = median of m_k over the pilot seeds.** If fewer than 3 of 5 seeds have a crossing, the rule fails (kill K4).
5. **Per-statistic margins m_f.** For each rule whose statistic is not the 90%-quantile gap, the same steps with that
   statistic's population value in place of `Gamma`: the mean gap (A-MEAN; B's mean gates: mean of the group minus mean
   of the rest), the Hodges-Lehmann shift (A-HL), the kernel-weighted mean gap (A-K4's plain rival). `sigma*_k` is the
   same for every f (it comes from V alone). A-EXC needs no own margin: its statistic sits at 0.10 exactly when the
   quantile gap equals m.
6. **Hinge margin m_hinge.** The same steps with V taken from the gated hinge (`J(gated hinge actor) - J(host)` in
   sigma-physics); the gated hinge has about half the clip's no-stall cost (30.9 against 61.8: CONN
   `restructure/step2_results/results.md:29`; MAIN `fallback.md:174`), so its break-even sits lower. It is used when
   the gated hinge is the caution action, with its oracle `A-PERFECT_hinge = 1[V_hinge > 0]`.
7. **Online margin m_on** (for staged B and the information-only monitor): `Gamma_on(sigma)` = 90% quantile of first-use
   residuals (one per probe episode of N's host, unclipped, in sigma-physics, strong-current start law) minus the 90%
   quantile of the residuals at or below `t_lo` of separate fallback reference episodes, built exactly as staged B's
   tries and threshold are (section 5.3). The run-time monitor failed because its margin was set in recorded conditions
   (`ROUND1.md:60-63`).
8. Reported beside it: `m_k` from a ladder on S's host and S's rows (it changes no decision), and every A and B rule at
   multiples 0, 0.5, 1 and 2 of its own margin, and the quantile rules at the absolute 0.10.

---

## 4. Arms and controls

### 4.1 Option A arms

**Decision rules**, all on the same held-out scores and groups, one decision per seed-world (world-level), each at its
own margin (section 3.6):

| Arm | On iff |
|---|---|
| **A-BCA (primary)** | `L_gap > m`: BCA's two-posterior gap, uniform weights (`step1lib.certificate`) |
| A-BCA0 | the same at margin 0 (the margin's share) |
| A-BB | Bayesian bootstrap of the two 90% quantiles without WBCP's test atom, 5% lower bound of the gap > m |
| A-SC | plain split conformal: empirical 90% quantile of the group minus that of the rest > m (step 2's `conformal_k1` switch) |
| A-OS | exact order-statistic bound: the distribution-free 97.5% lower bound of the group's 90% quantile minus the 97.5% upper bound of the rest's > m |
| A-BOOT | percentile bootstrap (2,000 resamples) 5% lower bound of the quantile gap > m |
| A-MEAN | Welch 95% lower bound of the mean gap > m_mean; and the plain difference of means > m_mean (the round-1 verifier's rule) |
| A-HL | Hodges-Lehmann 95% lower bound of the shift > m_HL |
| A-RANK | one-sided Mann-Whitney p < 0.05, no margin (step 1's rank test, `step1lib.py:279-287`) |
| A-EXC | Clopper-Pearson 95% lower bound of the group's share above the rest's 90% quantile plus m > 0.10 |
| A-SHUF | A-BCA with outcomes shuffled within each seed-world (group and rest pooled and re-dealt), 200 permutations (`step1lib.py:290-305`) |
| A-RAND | random switching at A-BCA's own on-rate, pooled over the stage's S and N seed-worlds; scored by its exact expectation (section 7.2), one realized coin reported |
| A-NEVER | the host alone |
| A-INS | A-BCA on the in-sample survivors against the bank (A-K3) |
| A-PERFECT | on iff the true value of caution V > 0 (oracle reference); A-PERFECT_hinge for the hinge |

**World-blind set W** (deployment wrappers or existing actors; no outcome read): clip always (A-ALWAYS); the gated hinge
(k = 1) always; a cap at 0.80 always (every thruster clipped to [-0.80, 0.80]; labeled "knows the knee"); the best plain
cap among 0.75, 0.85, 0.90 and 0.95 always, chosen on the pilot by pooled NET and frozen in the confirmatory slot. In
the record the always-on hinge netted +76.5 and the always-on BCA ramp +77.7, against +44.4 for clipping always (CONN
`restructure/step2_results/results.md:20, 29`; ROUND1.md:13), so clipping always is not the best world-blind rule.

**Caution actions.** When on: (i) **clip** (primary): a deployment wrapper, no training; (ii) **gated hinge**: the host
retrained with stage 1's range hinge `100 sum_j([a_j - hi_j]_+^2 + [lo_j - a_j]_+^2)` at k = 1 in the actor loss
(`harness.py:373-376`), deployed in place of the host when on, switched at `m_hinge`; (iii) **clone fallback**
(reported): the clone's command replaces any command with `t(a) > t_lo`. With the same perfect detector the record has
clip +106.1 and +28.2 against the gated clone fallback's +73.8 and -30.4 in strong current and big kicks, and in the
no-stall world -61.6 for the clip against -109.5 for the clone (HV2
`outputs/shift/beyond-the-data/results-2026-10-13/deploy_tables.md:17, 33, 79, 95`), which is why the clone is reported
only.

**Mixed worlds (A-K4).** Declared groups by state, fixed now and not aligned with the stall cut: wave-amplitude cells
[0.1, 0.2), [0.2, 0.3), [0.3, 0.4] (the middle cell straddles 0.25). An episode's cell is its start's amplitude;
starts above 0.4 lie in no cell and run the host. Expected evidence: on seeds 0-4 each amplitude half held 6-18
calibration-split full-throttle rows per seed-world (`_where/feas_amp_0_1_2_3_4.json`), so about 15-45 per half with
both splits and about 8-30 per cell [inferred]. A cell with fewer than 10 full-throttle rows runs the host and is
reported "no evidence".

| Arm | On iff (per episode, by its start) |
|---|---|
| **A-KERN (primary of A-K4)** | A-BCA with weights from a Gaussian kernel in amplitude, Silverman bandwidth, test mass 1, as `mixed_waves/run_route.py:78-92` |
| A-CELL | A-BCA per cell (equal weights within the cell) |
| A-SC per cell | plain split conformal per cell at m |
| A-KQ | kernel-weighted plain quantile gap: the same kernel weights through WBCP's weighted empirical quantile `lambda_hat` (`wbcp.py:89-102`), no posterior, at m |
| A-KMEAN | kernel-weighted difference of means at its own margin `m_kmean` |
| world-level A-BCA | as above |
| caps | 0.75, 0.80 (labeled "knows the knee"), 0.85, 0.90, 0.95 always; the best non-knee cap chosen on the pilot |
| A-ORACLE-REGION | on iff the start's region stalls (reference; round 1's oracle scored 52.6, ROUND1.md:132) |
| A-KERN-CAP (reported) | A-KERN's own decisions with a cap at the pilot's best cap in place of the clip (the action, not the localizer) |
| A-KPERM | A-KERN with the full-throttle scores permuted among the full-throttle rows, each row keeping its amplitude, 200 permutations (round 1's decisive control, `mixed_waves/KILL_RULE.md:42-46`) |

**Information only (A-I).** Margin sweep by evidence size (subsamples of 5, 8, 12, 20, 30 rows and all rows, 400
replicates each, as round 1's small-evidence test); the online monitor in S, N, M3 and M4 (A-BCA re-read after each of
the first 30 strong-current episodes with margin `m_on` and a latch); the score with Q1 alone and with Unifloral's
noisy target.

### 4.2 Option B arms

**Units of decision.**
- **B-GROUP (primary unit)**: one declared deviation group per seed-world: "let the host's own command through
  unclipped" (that is, never apply the fallback). Tries: offline, the conditioned full-throttle evidence rows (section
  3.2); online, first uses (section 5.3).
- **B-RUNG (per deviation; reported)**: per thruster k and sign, rungs from the recorded range edge: rung 0 = (edge,
  0.80], rungs 1-4 = (0.80, 0.85], (0.85, 0.90], (0.90, 0.95], (0.95, 1.00] (float32 edges). A try of a rung: a row
  whose command for that thruster and sign lies in the rung and whose host command at the try's own state lies beyond
  the rung's lower edge (arm B's definition, CONN `arms_abc_f0/code/arm_b.py:8-12`). Walk rung 0, 1, ... in fixed
  order; stop at the first refusal; the map's ceiling for that thruster and sign is the top of the last certified rung
  (`arm_b.py:131-139`). The deployed map clips each thruster to [lo', hi'] with the certified ceilings in place of the
  recorded edges, when `t(a) > t_lo`.
- On the main log, rungs 1-3 hold no offline tries (the log has no commands between `t_lo` and `t_up`), so offline
  B-RUNG can only cap near 0.80 (declared now; reported as information). Online, probe episodes fill every rung the host
  reaches.

**Outcomes per try.**
- **Residual score s (primary)**, the same score as A, at the try's own command (the placement red team's fix, PLACE
  `red.challenges[4]` and `red.revised_recommendation`, test 3: step 1's same-command score carried the stall).
- Arm B's critic-scored gain `g = r + 0.98 Q_h(x', pi(x')) - Q_h(x, a~)` against one rung down (`arm_b.py:13-14`):
  the **negative control**; with shuffled outcomes it still certified 0.90/0.95/1.00 of rungs (`placement/O8_screen_seed0.json`
  via PLACE `red.challenges[4]`), and arm B stopped on its K1 (CONN `arms_abc_f0/RESULTS.md:44-47`).
- k-step segment outcome (k = 10, 25) against the critic at the fallback command: a matched contrast only, because the
  logged continuation follows the recording controller, not the host.
- Shuffled s (tries and rest pooled and re-dealt at the same sizes, 200 permutations, the same tau): the null of B-K1.
- Online only, simulator reference, not deployable: the **twin outcome** of a probe episode, `J(map with deviation) -
  J(map without)` from the same start, harm iff negative.

**Gates**, each with `alpha_B = 0.1` and level 0.95 unless stated, on the same tries and the same `tau`:

| Gate | Certify iff |
|---|---|
| **Exact tail gate (primary)** | BCA's tail posterior computed exactly (Beta form): the harm count is at most the largest count whose Clopper-Pearson 95% upper bound is <= alpha_B; must equal the exact binomial test on 100% of decisions (stop rule S6) |
| Shipped tail gate (reported) | `wbcp.calibrate(s_tries).threshold <= tau` with 1,000 draws and n >= 36 (BCA's own floor); every disagreement with the exact gate is attributed to the floor or to Monte Carlo noise (the floor exists because Monte Carlo BCA sometimes certified at 26-28 rows; LIVE 10533, 10540) |
| Split-conformal tail | the finite-sample-corrected empirical 90% quantile <= tau (+inf, refuse, when undefined) |
| Plain empirical rate | harm share <= alpha_B (no correction; an anti-conservative reference) |
| Hoeffding, empirical Bernstein | 95% upper bound of the harm rate <= alpha_B |
| BCA mean gate (Gaffke-type, Bayesian bootstrap with a ceiling atom) | 95%-credible upper bound of the mean of s <= mean of the rest + m_mean (lifted from MAIN `bca_mean_gate_toy.ipynb` cells 7, 9, 12-13) |
| t-test, empirical Bernstein and HCOPE-style mean bounds | 95% upper bound of the mean of s <= mean of the rest + m_mean; HCOPE-style: Thomas et al. (2015) bound with truncation at a value fixed from the bank's score range |
| Arm B's rule (negative control) | `wbcp.calibrate(-g)` at miscoverage 0.5 has threshold < 0, and arm B's plain tests (`arm_b.py:76-128`) |
| Rejection sampling, weighted split conformal, self-normalized weighted mean | module W only |

**Controls**: the fallback always (= A-ALWAYS); the host always (= A-NEVER); the rest of W; the range limit, each
thruster capped at its 99.5% percentile (`arm_b.py:152-155`); the matched global ceiling, one ceiling for all thrusters
with the certified map's clipped share (`arm_b.py:142-149`); the oracle ceiling map (per thruster and sign, the ceiling
that maximizes mean J on a separate tuning bank); A's switched clip (same score, same margin, the other default); the
clone fallback (reported).

### 4.3 Option C arms

All co-trained: Unifloral's TD3+BC restated with one change in the actor loss (`td3_bc.py:245`):

`actor_loss = -lambda x mean_i [Q1(x_i, pi(x_i)) - k W_i] + mean_i mean_j (pi(x_i)_j - a_ij)^2`,
`lambda = 2.5 / stopgrad(mean_i |Q1(x_i, pi(x_i))| + 1e-7)` (on Q1 alone, as `td3_bc.py:242-244`), k = 1, only the
actor's parameters differentiated through W.

- **Where the band is read.** At `a_i^q = clip(pi(x_i) + xi_i, -1, 1)`, one fresh `xi_i ~ N(0, 0.2^2)` per thruster per
  row per step from BCA's own key (never Unifloral's). Q1 stays at `pi(x_i)`, as Unifloral's line 238 has it, so that
  with k = 0 the step is exactly the host's. (The placement analysis wrote Q1 at the smoothed command too; that form is
  reported as information only, decision D6.) The training rows are kept rows, so the band is read at the declared
  query law's states.
- **The lower edge, paired form** (MAIN `experiments/shift_iw/lower_edge.py:1-17`):
  `W_i = [U_w(x_i, a_i^q) - U_w(x_i, a_i)]_+` where the band certifies the query;
  `W_i = [Q1(x_i, pi(x_i)) - Q1(x_i, a_i)]_+` where it refuses (no value credit, as CONN `command_le_train.py:105-112`).
- **The band**: `U_w(x, a) = R(w(x, a)) x u x max(eta(x, a), 1e-6)`; scores on the bank `|y - Q1| / (u x max(eta,
  1e-6))`, y = Unifloral's own target (`unifloral_td3_bc_bca.py:316-340`); R from WBCP with the bank rows' weights and
  the query's test mass `w(x, a)`, read from a threshold table on a grid of test masses (`iw.mass_grid`,
  `iw.threshold_table`, `iw.lookup`, HOSTS `experiments/shift_hosts/iw.py:136-192`, ungated, in the training
  environment).
- **Exact weights for the main recorder** (new code, section 1): `w = q_xi(a | x) Z(x) / beta(a | x)` outside the ball.
  The training environment evaluates `beta` from the per-row `mu(x)`, hard-water flag and `Z(x)` handed off by the
  scoring environment (section 3.2) and a JAX restatement of the recorder's density (normal driving by change of
  variables through `0.8 tanh(u / 0.8)`; the full-throttle command's density with thruster 1's atom at -1); a stage-0
  test requires the two environments to agree on the density, and that on a large made-up kept-row sample
  `mean(w) = 1 - P_query(w = +inf)` within Monte Carlo error (the query law's mass outside the recorder's support is
  exactly what the weights cannot carry).

**Arms** (in both S and N unless stated; "matched" = matched in S, coefficient locked in N):

| Arm | W |
|---|---|
| **C-BCA (primary)** | as above, k = 1 |
| C-k0 | C-BCA with k = 0: must reproduce the host bit for bit (stop rule S3; first seed-world of each stage) |
| C-F0_m (matched); C-F0 (k = 1, reported) | no value credit where C-BCA refuses, 0 elsewhere (the same refusal, no band; CONN `arms_abc_f0/code/arm_f0.py`); its strength set by scaling k on that term |
| C-RATIO_m; C-RATIO (k = 1, reported) | eta = 1 (the weighted threshold's scale only) |
| C-SC_m; C-SC (k = 1, reported) | weighted split-conformal threshold (Tibshirani et al. 2019) in place of WBCP's; same eta and weights |
| C-SHUF_m; C-SHUF (k = 1, reported) | the same band built from shuffled outcomes: at each refresh the bank scores are permuted across bank rows, and eta's fit targets are permuted within each batch (stream tag 15); same weights and refusal. With equal weights the bank permutation alone changes nothing, so the eta shuffle carries it there |
| C-SUP (matched) | support only: `c x [log w(x, a^q) - log w(x, a_i)]_+`; refusal as C-BCA |
| C-HINGE (matched) | range hinge (`harness.py:373-376`) |
| C-EVEN (matched) | even caution: the cloning term times (1 + nu) (the plain alpha retune the owner rejected) |
| C-RAND1, C-RAND2 | two fixed random-network penalty surfaces (stage 1's 64-wide, 2-deep); matched on the pilot, locked at the pilot's median coefficient for confirmation |
| C-ORACLE (new form) | co-trained one-step exact error on the host's own Q1 and actor: `[delta*(x_i, a_i) - delta*(x_i, a_i^q)]_+`, `delta*(x, a) = r + 0.98 Q1(x', pi(x')) - Q1(x, a)` from one exact float64 step by `dynamics_train.py`, traced under x64 (as `lower_edge.py:37-41` did); exempt from S3, because k = 0 cannot reproduce the host bit for bit under x64 |
| C-ORACLE-S1 (actor-only, reported) | stage 1's one-step oracle as it was run: `min_h Q_h` and the world's top controller at x', against the frozen host critic (`lower_edge.py:24-41`) |
| C-ORACLE-V (actor-only, reported) | stage 1's exact long-run harm penalty `e(s, a) = Q1(s, a) - Q^c_250(s, a)`, c the world's top controller, paired form, against the frozen host critic (stage 1 `harness.py:559, 791`; it scored 82.2 against the one-step oracle's 79.7, stage 1 `RESULTS.md:44, 48`) |
| C-BCA x A, C-HINGE x A, CLIP x A, HINGE1 x A | composites, no training: the arm where A-BCA is on, the host where it is off (HINGE1 = A's gated hinge, k = 1, switched at m_hinge) |

**Matching.** Target: C-BCA's strength in S. Each matched control: a 3-rung geometric ladder of coefficients, log-linear
interpolation to the target, one matched run; tolerance 1%; at most 5 more rounds; otherwise recorded as unmatched,
which counts against C (step 2's rule, PRESPEC.md:178-186). In N each control runs at its coefficient locked in S. All
strengths and fingerprints are locked before any return is computed (step 2 PRESPEC.md:188-194). Matching needs only
strengths, but in both stages it starts only after K3 is judged, so that no matched control is trained when there is no
room (section 9.3).

**Information only (C-I).** C-I1 eta's share of the variance of `log U_w` against `log R(w)`, and the partial Spearman
correlation of `U_w` with the oracle's error given w (placement test T1's readout, PLACE rank test_plan T1); it fixes
only the wording of C-K2 (expected: eta carries almost nothing, PLACE `red.challenges[11]`). C-I2 an actor-only variant
of C-BCA, C-F0 and C-HINGE against the host's frozen critic (the protocol of every earlier C result: 20,000 steps, the
`lower_edge.py` protocol), to separate placement from critic feedback. C-I3 the sharpness of certified widths (section
6.4).

### 4.4 Shared controls (summary)

| Control | A | B | C |
|---|---|---|---|
| Host alone | A-NEVER | host always | the host (and C-k0) |
| Plain rivals | A-SC, A-OS, A-BOOT, A-MEAN, A-HL, A-RANK, A-EXC, A-BB, each at its own margin | exact binomial / Clopper-Pearson, split-conformal tail, Hoeffding, empirical Bernstein, t, HCOPE-style | C-SC_m, C-RATIO_m, C-F0_m, C-SUP |
| Shuffled outcomes | A-SHUF (sanity), A-KPERM | shuffled s (B-K1) | C-SHUF_m (C-K6) |
| Matched-strength even caution | A-RAND (placement at its own rate) | matched global ceiling | C-EVEN |
| World-blind rules | W | W | W, through the switched composites |
| Support and range rules | caps (mixed) | range limit, knee clip | C-HINGE, C-SUP |
| Oracle references | A-PERFECT, A-ORACLE-REGION | oracle ceiling map, twin outcome | C-ORACLE, C-ORACLE-S1, C-ORACLE-V |
| Fallback variants | clip (primary), gated hinge, clone | clip (primary), clone | refusal = no value credit |

---

## 5. Deployment simulation

### 5.1 Offline computation, then deployment

1. **Train** the hosts (S, N, MW+, MW-), the clones (S, N), A's gated hinge (S, N) and C's arms (S, N). Lock every
   network's fingerprint. For C, also save Q1, Q2, the target networks, the actor, eta, u, the threshold tables and the
   bank weights at each declared refresh j in {10,000, 25,000, 40,000, 55,000, 70,000, 85,000, 95,000}, and the
   networks at the end of each such band's window (just before the next refresh; at the end of training for 95,000).
2. **Compute offline, read-only**: the gap rule on commands; the scores of evidence, bank and replicate rows; A's
   decisions for every rule; B's offline certificates and maps. Lock all decisions in `decisions.json` (write-once). In
   the confirmatory stage this happens before any return of any policy is read (section 9.3).
3. **Deploy**: online episodes of every deployed policy (host, wrappers, composites, trained arms) on every bank and
   world. Decisions never change after step 2 for the offline arms.

### 5.2 Online evaluation episodes

- **Runner**: one episode = 100 steps through `TC.dynamics` with the command rounded to float32 then float64 and clipped,
  exactly as `env.step` (`thruster_chain.py:443-446`). Spot check: 10 starts per bank also run through gymnasium
  `env.step`, bit for bit (stop rule S8). Mixed worlds: each episode's physics from its start's amplitude.
- **One arithmetic for every truth.** `truth.rollout` and `truth.attribution` do not round (`truth.py:85, 368-376`), and
  the clip wrapper's edges are float64 quantiles, so a clipped command differs between the two paths. Before any truth
  call, every deployed policy is therefore wrapped as `x -> float64(float32(p(float32(x))))`. Then `truth.rollout`,
  `truth.attribution` and the online runner coincide bit for bit, and S8 also checks attribution's `J_new` against the
  online J on the spot-check starts.
- **Counts, per seed-world and deployed policy**: 4 situations x 1,000 starts (fresh bank on confirmatory seeds;
  standard bank too); 100 steps each. Every arm runs on the same starts (common random numbers). Worlds: S and N for all
  options; MW+ and MW- for A; M3 and M4 for A and B (information); the ladder for the margin, A-K2 and the readouts.
- **Per start**: J, points, the same policy capped at 80%, the share of steps past the knee, F1, F2, late peak, and the
  visited states, commands, rewards and next states (for the step-level readouts and B-V2).

### 5.3 Staged deployment of B (the online variant)

Per seed-world, situation strong current (recorded conditions and big kicks reported), deployment horizon **H = 1,000
episodes** from the stream `[20261019, 6, k, d, 1]` (shared by S and N):

1. **Reference.** The first 50 episodes run the fallback F. The residuals of their steps at or below `t_lo` give
   `lambda_R,on`, their empirical 90% quantile; `tau_on = lambda_R,on + m_on`. These episodes are part of the
   deployment (F is the default anyway), so they carry no probing cost, and they share no episode with the tries, so
   tau does not depend on the tries. The first 10 also pick B-RUNG's families.
2. **Probe** decision d (B-GROUP: the host unclipped; B-RUNG: F with rung d opened). Each of the next N episodes runs the
   probe map; from it take **one try**: the first step where the deviation was used (`t(a) > t_lo` for the group; the
   rung used, for a rung), with its residual s from the episode's own reward and next state. One try per episode keeps
   tries independent.
3. **Decide** with the exact tail gate after the N probe episodes (primary: one batch decision; **N = 100**, unless the
   pilot's discriminability step of section 8.5 sets N to 150, 200 or 300 in the confirmatory slot). Reported: the
   shipped gate; three looks at 0.4N, 0.7N and N with Bonferroni level 0.05/3 (at N = 100: 40, 70 and 100 tries,
   allowing 0, 1 and 3 harm events; the zero-failure floor at that level is 39, REV `revise_arith.out` section 3); an
   anytime-valid test, the mixture-likelihood e-process `E_n = integral over q uniform on [0, alpha_B] of L_n(q) /
   L_n(alpha_B)`, rejecting "harm rate >= alpha_B" when `E_n >= 20` (zero-failure floor 43, `floor_arith.out` section
   3); and a plain exact decision at 30 tries (zero harm events only). Each reported variant is a separate
   counterfactual deployment on the same starts: identical to the primary up to its stopping time.
4. Certified: deploy the probe map for the rest of the horizon (B-RUNG: then probe the next rung). Refused: return to F.
5. **B-RUNG (reported)**: at most the 2 thruster-sign families whose host command exceeds the recorded edge most often
   in the first 10 reference episodes (at least 1% of steps). Families are probed in parallel at each rung level, one
   try per family per episode, with the confidence Bonferroni-split across families and a fixed-sequence walk within
   each; B-RUNG uses 100 probe episodes per level whatever B-GROUP's N, so 5 levels fit in H. B-V1's truth is computed
   for each realized probe map, cached by map.
6. **Score**: cumulative return over all H episodes, reference and probes included, per world, in points (anchors from
   `anchors_on` on the deployment stream's own starts); stall steps during probing; decision time; the final map's
   score on the scoring bank.
7. **Twin outcome** (reference only): the same protocol with the twin outcome as the try, to show what a perfect outcome
   would certify. **Realized mean variant** (reported): episodes randomized between F and the probe map, mean gates on
   the return difference (Welch t; HCOPE-style empirical Bernstein with declared truncation); about 580 episodes per arm
   would be needed for a 5-point effect (unpaired per-start SD about 34 points in strong current; LIVE 5104-5112 gives a
   bank standard error of 1.63 return units over 1,000 starts).

### 5.4 Crossing the two environments

Every residual needs the host critic, which lives only in the training environment; every episode needs the simulator,
which lives only in the scoring environment. The protocol, one hand-off round trip each:

- **Offline scores** (A, B, ladder, replicates): the scoring environment writes the rows (state, command, reward, next
  state, and the actor's next command from its float32 restatement, HOSTS `actors_f32.py`, `policies.MLPAdapter`);
  the training environment returns `Q_h(x, a)`, `Q_h(x', pi(x'))` and the residual.
- **Staged B**: the probe map is fixed for a decision, so the scoring environment runs all reference and probe episodes
  of the decision, hands off the reference steps and the tries, receives their residuals and decides: one round trip
  per decision (per rung level for B-RUNG). The looks and the e-process are computed from the same stored sequence.
- **Online monitor (information)**: run the 30 episodes with the host alone, score all their steps in one round trip,
  find the episode after which the latch would close, and rerun the later episodes with the clip (deterministic given
  the starts).
- **C's coverage draws**: the scoring environment draws states and commands from the declared law and steps the
  simulator once; the training environment computes y (target networks and target noise), Q1, eta, u, the weight and
  the band, and returns covered or refused per draw.

---

## 6. Metrics (exact formulas)

### 6.1 Return

- Per start i: `J_i`, points `P_i = 100 (J_i - J_rec) / (J_top - J_rec)` with the world's anchors on that bank.
- Per seed k, world w, situation s, policy p: `P_ksw(p) = mean_i P_i`; paired standard error over starts of
  `P_i(p) - P_i(q)` (HOSTS `evaluate.paired`, `evaluate.py:356-362`).
- `NET_k(p) = [P_k,S(p) - P_k,S(host)] + [P_k,N(p) - P_k,N(host)]`, strong current; `NET = mean_k NET_k`.
- `NET_mixed,k(p) = [P_k,MW+(p) - P_k,MW+(host_MW+)] + [P_k,MW-(p) - P_k,MW-(host_MW-)]`, strong current, every start
  in its own region's physics with that region's anchors on the same bank.
- No-stall cost `NSC_k(p) = P_k,N(host) - P_k,N(p)`, per situation.
- Keeps the improvement: `P_k,S,s(p) - P_k,S,s(host)` for s in recorded, big waves, big kicks.
- Every return is also reported in return units.

### 6.2 Harm events

Stall steps per episode (`TC.past_knee`, stalling world only); the stall loss `J(p capped at 80%) - J(p)`; F1 (return
below the zero-command controller's from the same start); F2 (late peak above the 99th percentile of the world's top
controller's late peak on the same bank) (HOSTS PRESPEC.md:486-490; `evaluate.py:23-35`); worst-10% mean J.

### 6.3 Decision accuracy (A and B)

- Value of caution `V_kw` = `P_kw(clip wrapper) - P_kw(host)`, strong current.
- A's decision is correct iff `on = 1[V_kw > 0]`; if `|V_kw| < 2` points both answers count as correct.
- B's group decision is correct iff `certified = 1[-V_kw >= 0]` with the same indifference band; a rung decision is
  correct iff `certified = 1[V_dev >= 0]`, `V_dev` the exact gain of opening that rung over the map without it.
- Expected NET of a rule over replicates: `E-NET_k = sum_w p_on(k, w) V_kw`, with `p_on` the rule's on-rate over the
  replicate evidence sets (section 6.4); realized NET uses the actual evidence. Regret against A-PERFECT; false-switch
  cost `sum over V < 0 of p_on |V|`; missed-harm cost `sum over V > 0 of (1 - p_on) V`.
- Ladder: `E-NET_lad,k(rule) = sum over the 9 middle ladder points i (all but sigma = -1, 1.5, 3.0) of p_on(k, i)
  V_k(i)`, with `p_on(k, i)` over N's replicate blocks with outcomes recomputed under sigma_i, and `V_k(i)` the value of
  caution of N's host in sigma_i-physics (points with `anchors_on`). Clipping always scores `sum_i V_k(i)`; never
  clipping scores 0.

### 6.4 Validity and coverage against simulator truth

The simulator gives three exact truths: the one-step outcome at any (state, command) (`thruster_chain.py:201-221`); the
value `Q^p` of any deterministic policy, wrappers included, by a 1,000-step float64 rollout (`truth.py:98-108`); and the
exact split of any return difference along the deployed path, `truth.attribution` (`truth.py:29-31, 359-383`), whose
terms are zero wherever two policies agree and sum to the difference within 1e-8.

**Replicates (A and offline B).** Replicate r of a seed is a fresh held-out block: 200 calibration and 300 evaluation
episodes of the recording controller, from the streams `[20261019, 5, k, r, 1..2]`, simulated in S and in N from the
same random numbers (and spliced for the mixed worlds), cut by the seed-world's own ball, with its own bank (5 rows per
calibration episode via `calibration/bank.stratified_bank`), its full-throttle rows and its rest, all scored by the
frozen host critic. A replicate is therefore drawn exactly as the actual evidence was, episode clustering included (the
14-28 full-throttle rows of a seed-world came from 13-25 episodes), and its group sizes vary as the actual ones would.
R = 1,000 blocks per seed (about 500,000 episodes per world); their union is the population that defines the truth
(about 35,000-70,000 full-throttle rows and 850,000 rest rows per seed-world expected).

**A (per seed-world, over the R replicates):**
- True gap `Delta_ev = q_0.9(pool_G) - q_0.9(pool_R)` over the union (inverted-cdf quantiles), with its Monte Carlo
  standard error. Also the mean gap, the Hodges-Lehmann shift, and the n-matched gap `q_{0.9/(1 - 1/(n_G + 1))}(pool_G)
  - q_0.9(pool_R)`: the quantity WBCP's posterior actually targets, because each crossing includes the test atom
  (`wbcp.py:43-54`), which for 35-70 rows moves the group's crossing to about its 91-92% quantile.
- **A-V1, boundary false-switch rate.** For each seed-world, shift every full-throttle score of every replicate (and of
  the union) by `c_kw = m - Delta_ev`, keeping episodes and groups. The true gap is then exactly m. `FSR_kw = (1/R)
  sum_r 1[L_gap,r + c_kw > m]`. Because L_gap moves exactly with the shift, `FSR_kw = 1 - COV_kw`, where `COV_kw = (1/R)
  sum_r 1[L_gap,r <= Delta_ev]` on the unshifted replicates: A-V1 is at once the error control at its hardest point and
  the coverage of the true gap by L_gap. The same posterior draws are used for the shifted and unshifted replicates.
- **Size table (reported, and used to read A-K1).** Every rule of section 4.1 at its own boundary (its own statistic's
  true value shifted to its own margin), the rate it switches on; and an equal-shapes null (group and rest pooled and
  re-dealt at the replicate's sizes, then m added to the re-dealt group scores).
- **Natural false-switch rate (reported).** `(1/R) sum_r 1[L_gap,r > m]` in seed-worlds with `Delta_ev <= m` (in practice
  N); round 1's no-stall lower bounds were -0.01 to +0.02 against a 0.10 margin (ROUND1.md:31), so this rate says little
  about the error control.
- **n-matched coverage (reported)**: the share of replicates with `L_gap,r` at or below the n-matched gap.
- Power: `(1/R) sum_r 1[L_gap,r > m]` where `Delta_ev > m`.
- Transfer (not promised): the same gap at the host's own visited steps in strong current (steps past `t_lo` against the
  rest, each with its episode's own outcome).
- Every rate with its Clopper-Pearson 95% interval.

**B:**
- **Truth of a decision.** `pi(t) = P_pop(s > t)`: the harm rate at threshold t over the truth population of its tries
  (offline: the union of the replicate blocks' conditioned full-throttle rows; staged: the union of the first uses of
  the replicate probe sets of section 5.3, `[20261019, 4, k, d, r]`, 1,000 sets of N).
- **B-V1, error of a replicate.** Replicate r computes its own `tau_r` (offline from its own decoupled rest; staged from
  its own reference episodes) and its own certificate. `error_r = certified_r AND pi(tau_r) > alpha_B`, so the error
  includes tau's randomness and the clustering of tries in episodes, which the exact binomial test ignores.
- **B-V1, boundary null (judged).** Per seed-world and per unit (B-GROUP offline; B-GROUP staged), shift every try's
  score by the constant `c_kw` that makes `pi(tau_pop) = 1.1 alpha_B`, `tau_pop` the population threshold (the 90%
  quantile of the population rest plus m, or m_on). Reported at `pi(tau_pop) = alpha_B`, `1.2 alpha_B` and
  `1.5 alpha_B`; the natural decisions and the ladder decisions with `pi` in `(alpha_B, 2 alpha_B]`; and the
  fixed-threshold variant (tau held at the actual evidence's value over all replicates), which isolates the gate.
- **B-V2, transfer of the certified event.** Run the deployed map p_B on the scoring bank (strong current). At every step
  where `p_B(x_t)` departs from `F(x_t)`, compute the exact residual `s_t` from the step's own reward and next state with
  the same critic and actor as the gate. `HR_k = #{departing steps with s_t > tau} / #{departing steps}` per seed-world
  (tau as the gate used it: offline tau, or tau_on). A world with fewer than 3 seeds having any departing step reads
  "vacuous" for B-V2.
- **B's safety readouts (reported with B-R).** From `truth.attribution(p_B, F)`: the share of departing steps with
  `term_t < 0`; the share with `term_t < -c`, c = 1% of F's mean absolute per-step reward on that bank; the harm mass
  `sum min(term_t, 0)`; per start `D_i = J_i(p_B) - J_i(F)`, `P(D_i < 0)`, its 10% tail mean, stall steps, F1, F2.
- Refusals: share refused, split into "too few tries" (n below the floor) and "too many harm events", priced against
  the oracle ceiling map: `P(oracle map) - P(p_B)`.

**C (per seed and world; refusal counts as covered for the promise):**
- **The place of use.** C's band acts only inside training: band j (refresh j's R, eta, u) shapes the actor for the 5,000
  updates after refresh j, at the training batches' kept-row states. The declared refreshes are j in {10,000, 25,000,
  40,000, 55,000, 70,000, 85,000, 95,000}.
- **C-V1 promise.** For each declared refresh j: M = 50,000 fresh draws from the declared law under actor j (kept-row
  states from fresh recorded-law episodes, `[20261019, 7, k, w, j]`); for each, `y = r + 0.98 min_k Qbar_k(x',
  clip(pibar(x') + eps))` with r and x' from one exact simulator step at the drawn command and one fresh target-noise
  draw (the host's own target rule), all networks those saved at refresh j; `COR_j = (1/M) sum 1[refused or |y - Q1| <=
  U_w]` with band j. `COR` = the mean of `COR_j` over the 7 declared refreshes. Reported beside it: each `COR_j`;
  staleness (band j on the networks at the end of its window, which is what the actor climbed); the band refreshed on
  the final networks (information only: nothing uses it).
- **C-V2 usefulness**, at the same draws, with soft refusals counted as refusals: `RS` = refused share; `CAC` = covered
  among certified. Judged overall and in the **near-knee hard-water zone** (states with `c + w1 > 0.565`, query throttle
  in (0.70, 0.80], query outside the ball), where the recorder has support and a finite band is possible. Past 80% on
  the main recorder RS is about 1 by construction (section 2.3): reported, not judged. Also reported inside and outside
  the ball.
- **C-V3 per-bank promise.** For each seed-world and declared refresh j: 200 replicate banks (the banks of the first 200
  replicate blocks: fresh calibration episodes, the same ball cut, 5 rows per episode), scored by the frozen networks of
  refresh j with its eta and u, weights recomputed, each bank's own threshold table, and each bank's `COR_b,j` on the
  same M draws. Per (seed-world, bank) unit, `f_b` = the share of the 7 refreshes with `COR_b,j < 0.90`; `F3` = the mean
  of `f_b` over all units of the stage. The same is computed for C-SC and C-RATIO; `F3(C-SC) - F3(C-BCA)` is reported as
  the coverage credit of the per-bank construction (shared with any training-conditional tolerance limit; with equal
  weights it is Wilks'). Also reported: whether the actual bank's COR is below 0.90, per seed-world.
- **C-I3 sharpness (reported)**: at certified draws in the near-knee zone, the median of `U_w` divided by the 90% quantile
  of `|y - Q1|` in the same zone.
- Information: on-policy coverage at every visited (state, executed command) of the 1,000 strong-current episodes, and
  at the same states with the smoothed command; value error `P(|Q1 - Q^pi| <= U)` and uncaught over-estimation
  `P(Q1 - k U > Q^pi)` at 2,000 visited pairs per bank (`Q^pi` by `truth.q_value_jax`); the share of hard-water states
  whose best throttle for `Q1 - W` lies past 80% (`lower_edge.py:49-58`).
- **Coverage arithmetic.** `coverage_c.py` computes COR as (covered + refused) / all. It does not reuse
  `experiments/shift_iw/coverage.py`'s coverage, where a refused step never counts as covered (`coverage.py:13-15`); a
  stage-0 unit test pins COR on a made-up table that contains refusals.

### 6.5 Uncertainty and dependence

- Starts are independent given a frozen policy; all arms share starts, so differences are paired by start.
- Steps are clustered in episodes: every step-level rate gets an episode bootstrap (2,000 resamples of episodes,
  `[20261019, 8, k, 1, 1]`) within each seed, and the measured design effect is reported. B-V2 uses a two-stage bootstrap
  (seeds, then episodes within seed). Precision at 90% coverage with 1,000 episodes of 100 steps: +-0.19 to +-1.45
  percentage points for within-episode correlation 0 to 0.6 (`floor_arith.out` section 6).
- Seeds are the unit of replication for every return and credit claim (section 8). Replicate-based rates (A-V1, B-V1,
  C-V3) pool independent replicates over seed-worlds; their marks are in section 7.

---

## 7. Marks, decision rules, kill rules and what each outcome would mean (fixed now)

All bars are in points (strong current) unless stated. "LB > 0" is the one-sided 95% bound over seeds of section 8.1.
"Holm p <= 0.05" is the Holm-adjusted one-sided p within the family named (section 8.3). Where a mark compares with
"the pooled-best" of several policies, that policy is fixed first by its pooled mean, and the per-seed differences are
then taken against it.

### 7.1 Kill and stop rules (read first, in this order)

**Stop rules** (each stops its phase; a stopped phase is never rerun or replaced; every claim that needs it fails):
- S1 fingerprints: Unifloral files against `configs/unifloral.json`; the frozen prespec and the code manifest (every
  placement module, every `experiments/shift_hosts` module it imports, `baselines/unifloral/algorithms/bc.py`,
  `configs/unifloral.json` and the hosts' `EXTERNAL_CODE` paths, HOSTS `gate.py:56-65`) against your approval.
- S2 seeds: the allow-list of section 3.4.
- S3 reproduction: C-k0 must equal the host bit for bit (restated step with k = 0; tolerance 1e-5 per parameter at
  most), on the first seed-world of each stage; the first host of each stage is trained twice and must match bit for
  bit. C-ORACLE is exempt (x64 step).
- S4 float32 check (the hosts' S7): open loop within 1e-7; closed loop within 0.05 points per start (declared now,
  before any data, for the reason HV2 PRESPEC.md:11-14 and 888-915 give).
- S5 non-finite: any non-finite network, score, threshold or return.
- S6 equal-weight identity: the exact tail gate equals the exact binomial test on 100% of decisions.
- S7 attribution identity: `|sum_t term_t - (J_new - J_ref)| <= 1e-8`.
- S8 runner: bit equality with gymnasium `env.step` on the spot-check starts, and attribution's `J_new` equal to the
  online J there.
- S9 leaks: as section 3.2.
- S10 budget: before a stage starts, its projected cost is checked against its cap (section 9.2); during the stage it
  stops between whole seeds only if measured spend would pass the cap.
- S11 bytecode: no `__pycache__` folder or `.pyc` file in the package or in `experiments/`, `experiments/shift`,
  `experiments/shift_iw`, `experiments/shift_hosts`, `calibration` or `baselines/unifloral/algorithms`, checked before
  and after every process (the hosts' gate refuses such files, HOSTS `gate.py:56-65, 124-134`).

**Kill rules** (judged on the pooled seeds of the stage):
- **K1 host learned**: the host's recorded-conditions points are at least the clone's + 10 in S and in N. Otherwise "did
  not learn": no claim is judged on that host.
- **K2 room for A and B**: `V_S >= +20` and `V_N <= -10` (strong current, pooled). Otherwise A's and B's return claims
  read "no room" and are not judged.
- **K3 room for C (= C-R0)**: the largest of `P_S(oracle) - P_S(host)` over C-ORACLE (co-trained), C-ORACLE-S1 and
  C-ORACLE-V (actor-only) is at least +20, strong current, pooled. Otherwise C's return claims are not judged and C's
  matched controls are not trained; C-BCA's coverage claims are still judged.
- **K4 margin**: the margin rule finds a crossing in at least 3 of 5 pilot seeds. Otherwise no confirmatory stage for A
  and B without a new version.

### 7.2 Option A marks

| Claim | Pass iff |
|---|---|
| **A-R1** | (i) `NET(A-BCA, clip) - NET(W*)`, W* the pooled-best member of W: mean >= +10 and LB > 0; (ii) `NET(A-BCA) - NET(A-RAND)` with the exact expectation `NET_k(A-RAND) = r [P_k,S(clip) - P_k,S(host)] + r [P_k,N(clip) - P_k,N(host)]`, r = A-BCA's on-rate pooled over the stage's S and N seed-worlds: mean >= +10 and LB > 0; (iii) in S, recorded / big waves / big kicks each >= host - 3 (pooled mean, step 2's G2 form) |
| **A-V1** | the boundary false-switch count pooled over every S and N seed-world of the stage (R = 1,000 each) has a one-sided Clopper-Pearson 95% upper bound <= 0.065, and no seed-world has more than 73 switches in 1,000 |
| A-K1 (machinery) | per seed, `E-NET(A-BCA) - E-NET(rival*)`, rival* the pooled-best of A-SC, A-OS, A-BOOT, A-BB, A-MEAN, A-HL, A-RANK and A-EXC, each at its own margin: mean >= +5 and Holm p <= 0.05 in family {A-K1, A-K4}; and A-BCA leads rival* at least at 3 of the 4 margin multiples 0, 0.5, 1 and 2 |
| A-K4 (machinery) | per seed, `NET_mixed(A-KERN) - NET_mixed(rival*)`, rival* the pooled-best of world-level A-BCA, A-CELL, per-cell A-SC, A-KQ, A-KMEAN, clip always and the pilot's best cap: mean >= +5 and Holm p <= 0.05 in family {A-K1, A-K4}; and under A-KPERM A-KERN keeps less than half of that lead (mean over the 200 permutations) |
| A-K2 (outcome) | per seed, `E-NET_lad(A-BCA) - E-NET_lad(R*)`, R* the pooled-better of clipping always and never clipping: mean >= +10 and Holm p <= 0.05 in family {A-K2, A-K3}; confirmatory seeds only (on the pilot m is fitted on the same ladder) |
| A-K3 (outcome) | per seed, realized `NET(A-BCA, held-out) - NET(A-INS)`: mean >= +5 and Holm p <= 0.05 in family {A-K2, A-K3} |
| Shuffle sanity (outcome verdict) | A-SHUF's on-rate in S is at most 0.05 (pooled over seeds and the 200 permutations) while A-BCA's real on-rate in S is at least 0.95 |

Reported, decides nothing: the gated hinge in place of the clip, `NET(A-BCA, gated hinge at m_hinge) - NET(gated hinge
always)`; `NET(A-BCA) - NET(clip always)` against draft 1's +20; A-V2's items (n-matched coverage, natural
false-switch rate, equal-shapes null, rivals' size table); one realized A-RAND coin; a Bayesian-bootstrap E-NET for
A-K3 (resampling the fixed survivors, labeled resampling, not replication); decision accuracy (10 or 20 seed-worlds);
the margin and evidence-size sweeps; the ladder; M3 and M4 with the online monitor; the clone fallback; A-KERN-CAP and
the 0.80 cap; A-ORACLE-REGION's regret.

**Rationale.** A-R1's bar is the best world-blind rule, as step 2's G1 was ("the best world-blind rival + 10", CONN
`step2_results/results.md:9`). With the record's values a switch that false-alarms in 1 of 5 no-stall seeds fails it
(per-seed differences against the hinge always 29.6 four times and -32.2 once: mean 17.2, LB -9.1), and a perfect
switch passes whenever the seed sd of its difference stays below 31 (REV `revise_arith.out` section 9). A-RAND is the
placement control of step 1 (PRESPEC.md:201-202); with one realized coin per seed-world a perfect switch failed it about
21% of the time (`review_stats/marks_arith.out` line 1), so its exact expectation is used (about 83.9 per seed with
record values, REV section 10). A-V1 is the error the switch controls, at its hardest point; by the shift identity its
count is also the coverage of the true gap by L_gap, which rests only on stretched theory for the two-posterior gap
(PLACE rank, O8), so it is measured. Its mark pools independent replicates: a rule at exactly 5% passes with
probability 1.000 (pooled) and 0.994 (guard) over 10 seed-worlds, a rule at 6% about 0.66, at 6.5% about 0.05 (REV
section 1); draft 1's "every seed-world at 0.073" failed an exact 5% rule 60% of the time over 10 seed-worlds
(`review_cov/arith.out`). The +5 credit bar is round 1's (`ROUND1.md:107`); expected NET over replicates removes the
luck of one evidence draw (`ROUND1.md:101-107`). A-K2 is round 1's middle-strength rule (+23.0 over clipping always
against a bar of 10, `ROUND1.md:74-96`); plain split conformal came within 2.4 points of BCA there, so its credit goes to
outcome reading, not to BCA.

**What the outcomes mean (wording fixed for both outcomes; with K2 passed).**

| A-R1 | A-V1 | Meaning |
|---|---|---|
| pass | pass | The harm switch works on a Unifloral host: it beats every world-blind rule and random switching, and at the boundary it fires in at most 5% of evidence draws (L_gap covers the true gap). Credit is read from the outcome and machinery verdicts. |
| pass | fail | Return earned on these seeds, but at the boundary the switch fires more than 5% of the time: L_gap is not a valid 95% bound on the gap; the return rests on the evidence lying far from the margin. If the n-matched coverage passes: "the posterior bounds the test atom's gap, not the margin's". |
| fail, dominated | any | A world-blind rule does as well or better: the switch is not needed here. |
| fail, not dominated | any | Not distinguishable from the best world-blind rule on these seeds. |

A-K1 passing: BCA's posterior beats every plain test at its own break-even margin (credit to the Bayesian step);
failing: plain tests at their own margins match it (expected; the credit stays with outcome reading and the margin).
A-K4 passing (with A-KPERM): credit to localization by weights; failing: credit to the action (a plain cap) or the
world-level switch. A-K2 passing: the switch follows the harm as it grows (outcome reading, shared by plain tests);
failing: on the ladder it does no better than a fixed rule. A-K3 failing: held-out rows are not needed (in-sample
survivors suffice, as round 1 saw, `ROUND1.md:39`).

### 7.3 Option B marks

`alpha_B = 0.1`, level 0.95, the score s, `tau = lambda_R + m` offline (decoupled rest) and `tau_on = lambda_R,on +
m_on` online, the exact tail gate.

| Claim | Pass iff |
|---|---|
| **B-R1** (B-GROUP, offline) | safety: in S, for every situation, `P(p_B) - P(F)` has LB >= -3; and release: `NET(p_B) - NET(F always)`: mean >= +10 and LB > 0 |
| **B-R2** (B-GROUP, staged, batch at N) | safety: in S, the episodes after the decision have LB >= -3 against F (the Seldonian promise is about the policy returned, not about probing); release: on the cumulative points of the whole H = 1,000-episode deployment, reference and probe episodes included, `NET - NET(F always)`: mean >= +10 and LB > 0. The probing cost in S (F always minus the cumulative points) is reported beside it |
| **B-V1** | at the boundary null, separately for B-GROUP offline and B-GROUP staged: the error count pooled over every S and N seed-world of the stage (R = 1,000 each) has a one-sided Clopper-Pearson 95% upper bound <= 0.065, and no seed-world has more than 73 errors in 1,000 |
| **B-V2** | per world with uses (offline map and staged map separately): the two-stage bootstrap 95% upper bound of the seed mean of `HR_k` is <= alpha_B + 0.02; it must pass in at least one world; a world with fewer than 3 seeds having uses reads "vacuous" |
| B-K1 (outcome) | B-GROUP offline in S: per seed, the one-sided permutation p-value of the real harm-event count among the tries against its 200 shuffles (same tau) is <= 0.05 in at least 4 of 5 seeds (8 of 10) |
| B-K2, B-K3 (machinery) | module W only (section 10) |

Reported beside B-R1 and B-R2: the best world-blind rule's NET and A's switched clip on the same seeds; B-RUNG offline
and staged; the shipped gate and every other gate; the looks and the e-process; the twin outcome; the realized mean
variant; the clone fallback; refusal prices; B's safety readouts (section 6.4); the certification rates with real and
shuffled scores (both over the 1,000 replicate try sets); arm B's gain g against B-K1's statistic; the confusion table
of A's and B's decisions on the same evidence.

**Rationale.** B's default is the fallback, so its safety mark is non-inferiority to F in the world where deviation
harms. It uses a bound (LB >= -3), not a mean, because safety is B's own promise; A's and C's "keeps the improvement"
is a side condition copied from step 2's G2 (pooled mean >= -3, step 2 PRESPEC.md:223-224). When B refuses everywhere,
`p_B = F`, every difference is 0, sd = 0 and LB = 0: safety passes. The release mark is the release of F where
deviation helps (+10 over clipping always, a sixth of the no-stall clip cost). In the stalling world 100 probe episodes
of 1,000 cost about a tenth of the value of caution, which is why B-R2's safety is judged after the decision and its
release on the whole deployment. B-V1 is the promise, judged where it can fail: with a fixed threshold and independent
tries the exact gate's size near the boundary is only 2.3-3.4% at 36-100 tries (`review_cov/arith.out`); what can push
it past 5% is the clustering of tries in episodes and a threshold computed from data, which only replicates near the
boundary show. B-V2 transfers the certificate's own event to the deployed map; the share of negative attribution terms
is a readout, because one negative term has no fixed relation to `s > tau` (in N a full-throttle step can lose a little
to the energy cost `0.3 |a|^2`, `thruster_chain.py:22`, while the map gains overall). The +0.02 allowance covers the
bootstrap's resolution at the expected number of uses. B-K1's draft-1 form (a certification-rate difference) could fail
even when the score carries the harm, since at 50 tries the exact gate certifies only 28% of the time at a harm rate of
0.05 (`review_cov/arith.out`); the permutation test does not depend on certifying, and with no information it passes
with probability 3e-5 (REV section 8). Release power is set in section 8.5.

**What the outcomes mean.**

| B-R | B-V1 | B-V2 | Meaning |
|---|---|---|---|
| pass | pass | pass | A safety certificate per deviation works here: it keeps the fallback where deviation harms and releases it where it helps. Credit to the placement only (equal weights make BCA the exact binomial test). |
| safety pass, release fails, marked "underpowered by design" (8.5) | pass | any | Safe; it refuses because the evidence is too small for the no-stall harm rate: evidence size, not the method. |
| safety pass, release fails, not underpowered | pass | pass or vacuous | Safe but refuses where it should release. |
| release passes, but dominated by W or by A's switched clip | any | any | It releases F, but a world-blind rule or the harm switch does as well: the certificate is not needed. |
| safety fails | any | fail | The certificate lets harm through: the score does not carry the harm at deployment. |
| pass | pass | fail | Promise kept on the tries, broken in deployment: the try law does not represent the deployed map. |
| any | fail | any | The tail gate's promise breaks near the boundary on clustered tries with a threshold computed from data. |

B-R2 passing and B-R1 failing: online tries rescue the per-deviation certificate at the probing cost reported. B-K1
failing: the score's tries are not worse than shuffled ones; the score does not carry the harm.

### 7.4 Option C marks

| Claim | Pass iff |
|---|---|
| C-R0 (kill K3) | the largest `P_S(oracle) - P_S(host)` over C-ORACLE, C-ORACLE-S1 and C-ORACLE-V is >= +20, pooled, strong current |
| **C-R1** | per seed, `NET(C*) - NET(P*)`, C* the pooled-best of C-BCA and C-BCA x A, P* the pooled-best of CLIP x A, HINGE1 x A and C-HINGE x A: mean >= +5 and LB > 0; and in S, recorded / big waves / big kicks of C* each >= host - 3 (pooled) |
| **C-V1** | `COR` (mean over the 7 declared refreshes) >= 0.90 as the seed mean with LB >= 0.88, in S and in N |
| **C-V2** | `RS <= 0.50` and `CAC >= 0.85` (seed means, soft refusals counted as refusals), overall and in the near-knee hard-water zone, in S and in N |
| **C-V3** | `F3`, pooled over S and N, has a one-sided 95% upper bound <= 0.073 (bootstrap over (seed-world, bank) units, 2,000 resamples) |
| C-K1 (machinery) | `NET(C-BCA) - NET(C-F0_m)`: mean >= +5 and Holm p <= 0.05 in family {C-K1, C-K2, C-K3, C-K4} |
| C-K2 (machinery) | `NET(C-BCA) - NET(C-RATIO_m)`: the same |
| C-K3 (machinery) | `NET(C-BCA) - NET(C-SC_m)`: the same |
| C-K4 (machinery) | `NET(C-BCA) - NET(C-SUP)`: the same |
| C-K6 (outcome) | `NET(C-BCA) - NET(C-SHUF_m)`: mean >= +5 and one-sided p <= 0.05 |

Reported: C-R1b, `NET(C-BCA) - NET(C-HINGE)` (draft 1's primary; mean >= +5 and LB > 0 shown against it); every k = 1
version; C-EVEN and C-RAND; the three oracles; C-I1 to C-I3; each `COR_j`, staleness and the final-refresh band; RS and
CAC past 80%, inside and outside the ball; `F3(C-SC) - F3(C-BCA)`; the actual bank's COR per seed-world.

**Rationale.** The record puts C's bars at the switched clip (106.1) and the switched hinge (107.4), not at the host
(PLACE `red.challenges[6]`): within the stalling world a no-signal clip beat the exact-error oracle (93.1 against 79.7,
stage 1 `RESULTS.md:44, 55`), and the lower edge lost to a matched range hinge by 45.6 (results-2026-10-14
`README.md:25`). So C-R1 compares C's best variant with the best plain action a switch can deploy; a C that beats only
the always-on matched hinge while far below A's switched clip is not a useful placement. C-V1 is the band's own promise
where the band acts: the certificate is about the critic at the refresh (PLACE `theory.findings[5]`), and nothing uses a
band refreshed on the final networks. C-V2 blocks meeting the promise by refusing, overall and where a finite band is
possible (48-75% of draws were refused before, PLACE `red.challenges[11]`); at RS = 0.5 the promise alone implies only
CAC >= 0.80 (`review_cov/arith.out`), so CAC is a usefulness bar. C-V3 is the per-bank promise that the Bayesian step
adds (DEPENDENCE.md:10-12; PLACE `theory.findings[9]`); over 2,000 independent units a rule at exactly 5% passes with
probability 0.996 and a rule at 8% with 0.002 (REV section 2), and averaging each unit over refreshes only narrows the
bound. The credit claims compare at matched strength, because step 2 could
not tell "Bayesian" from "larger" (CONN `step2_results/results.md:34, 38`). C-I1 is expected to fail (eta's coefficient
of variation 0.065-0.111 against a ratio spanning orders of magnitude, PLACE `red.challenges[11]`).

**What the outcomes mean.**

| C-R1 | C-V1 | C-V2 | C-V3 | Meaning |
|---|---|---|---|---|
| pass | pass | pass | pass | C beats every plain action a switch can deploy, and its band keeps its promise where it is used, bank by bank, without hiding behind refusal. |
| fail, dominated | any | any | any | A's switched clip or hinge does better: once the switch is right, the lower edge is not needed (expected). |
| fail, not dominated | any | any | any | Not distinguishable from the switched plain actions on these seeds. |
| any | pass | fail | any | The promise is met by refusing; where a finite band is possible it is refused or too wide to use. |
| any | fail | any | any | The band does not keep its promise at the place of use on a Unifloral critic. |
| any | pass | any | fail | The average promise holds but not bank by bank: the per-bank promise fails under these weights and this dependence. |

C-K1 failing: the gain is refusal (no value credit past the data), not the band (as F - F0 = +0.14/+0.08 before, CONN
`arms_abc_f0/RESULTS.md:122-123`); on the main recorder this is expected past 80% by construction. C-K2 failing: eta
carries nothing (wording also by C-I1). C-K3 failing: at equal strength the Bayesian threshold shapes the actor no
better than split conformal. C-K4 failing: the band acts as a support rule. C-K6 failing: the band's shape does not
depend on outcomes.

### 7.5 Across options: verdicts, domination and the credit map

**Three verdicts per option** (for B, separately offline and staged; for C, separately on the main recorder and in
module W):

1. **Placement works**: every primary claim passes (A: A-R1, A-V1; B offline: B-R1, B-V1 offline, B-V2 of the offline
   map; B staged: B-R2, B-V1 staged, B-V2 of the staged map; C: C-R1, C-V1, C-V2, C-V3), its kill rules passed, and the
   option is not dominated.
2. **Outcome reading earns its place**: A: at least one of A-K2, A-K3 passes in its Holm family and the shuffle sanity
   holds; B: B-K1; C: C-K6. The verdict says that this credit is shared by every plain test that reads the same score.
3. **BCA's own machinery earns credit**: at least one machinery claim passes in its Holm family: A {A-K1, A-K4}; C on the
   main recorder {C-K1, C-K2, C-K3, C-K4}; module W {C-K1W, C-K2W, C-K3W, C-K4W, C-K5} and {B-K2, B-K3}. The verdict
   names the family and the part.

**Dominated** (computed for every option): the best plain alternative available to the same deployment (A: the
pooled-best of W; B: the pooled-best of W and A's switched clip; C: the pooled-best of CLIP x A, HINGE1 x A, C-HINGE x A
and W) beats the option's pooled-best variant by at least 5 NET points with LB > 0 over the per-seed differences. A
dominated option cannot read "placement works". At a true difference of 0 this reading arises at most about 4% of the
time at a seed sd of 8.8 (REV section 11).

"At least one placement works" uses Holm's correction over the four primary return claims (A-R1, B-R1, B-R2, C-R1).

A and B read the same score with the same margin; they differ in the default. Where the evidence is thin, A leaves the
host alone and B falls back. The paired table of their decisions per seed-world is reported, with the returns of each
disagreement.

**Credit map: which comparison separates which part of BCA from which plain rule.**

| BCA's part | Separated from | By | Verdict |
|---|---|---|---|
| The residual score as an outcome reader | rules that read no outcome; shuffled outcomes | A-R1 (against W and random switching), A-K2, the shuffle sanity, B-K1, C-K6 | outcome (shared with plain tests) |
| Reading held-out rows | in-sample survivors | A-K3 | outcome |
| The Bayesian step (WBCP's posterior and test atom) | split conformal, order statistics, bootstrap, means, rank test, exact binomial, each at its own margin | A-K1, C-K3, C-K3W; for B none: with equal weights it is the exact binomial test (stop rule S6); C-V3 (per-bank promise, reported credit shared with Wilks' tolerance limits) | machinery |
| eta (the scale network) | the same band with eta = 1, matched | C-K2, C-K2W (wording also by C-I1) | machinery |
| The band's finite width | refusal alone, matched | C-K1, C-K1W | machinery |
| The band beyond support | a support-only penalty and a range hinge of the same strength | C-K4, C-K4W, C-R1b | machinery |
| The weights | unweighted posteriors and localizers, rejection sampling, weighted split conformal, weighted means | A-K4 (localizing weights, mixed worlds); B-K2, B-K3, C-K5 (importance weights, module W) | machinery |
| The margin (not BCA's, but carried the record) | the same rules at margin 0 | reported: A-BCA0 and the margin sweeps | none |

---

## 8. Statistics

### 8.1 Pairing and seeds

- For each claim's difference D (between two policies, or two rules), per seed: the mean over the bank's 1,000 starts
  (paired by start), with its paired standard error.
- Across seeds: the mean and sd (n - 1) of the per-seed values; `LB = mean - t(0.95, S - 1) x sd / sqrt(S)`
  (t = 2.1318 for S = 5, 1.8331 for S = 10). One-sided p = `P(T_{S-1} >= mean / (sd / sqrt(S)))`; when sd = 0, p = 0 if
  the mean is above 0, else 1 (HOSTS PRESPEC.md:540-542).
- **Nothing is shown until a stage is complete.** Until every seed of a stage is complete, only the integrity checks
  (S1-S11) and the spend are shown; no per-seed value, pooled value or bound. (Draft 1 showed per-seed values as they
  arrived; together with a stop between whole seeds that would have let early seeds steer a stop.)

### 8.2 Coverage and error rates

Per seed: episode bootstrap for step-level rates; Clopper-Pearson for replicate rates. Replicate rates that are judged
(A-V1, B-V1, C-V3) pool independent replicates over seed-worlds with a per-unit guard (section 7). Across seeds: seed
mean with the t bound, plus the count of seeds below the bar.

### 8.3 Multiplicity

- Within an option, a verdict needs all its primary claims: no correction (all must pass).
- Across the three options: each verdict at one-sided 5%; "at least one placement works" uses Holm's correction over
  the four primary return claims (A-R1, B-R1, B-R2, C-R1).
- **Families** (Holm within each, as the hosts apply it: sort the one-sided p-values ascending, ties by claim id; the
  i-th smallest passes if it and every smaller one are at most `0.05 / (k - i + 1)`; the adjusted p is the running
  maximum of `min(1, (k - i + 1) p)`, HOSTS PRESPEC.md:543-548): A machinery {A-K1, A-K4}; A outcome {A-K2, A-K3}; C
  machinery on the main recorder {C-K1, C-K2, C-K3, C-K4}; module W C {C-K1W, C-K2W, C-K3W, C-K4W, C-K5}; module W B
  {B-K2, B-K3}. C-K6 and B-K1 are alone. LB is also reported at the Holm level. Each family keeps the chance of any false
  pass at 5%; the verdicts name their family, so no correction is made across families.
- Fixed-rule checks without a p-value: the shuffle sanity, domination, the per-unit guards.
- Information items: no correction; pointers, never evidence.

### 8.4 Power and the size rule

- **Planning seed spreads, per claim, from the nearest record** (sd of per-seed differences):
  - A-R1: NET seed sd about 9.4 (step 2: seed SE 4.20, CONN `addon_report/report.md:143`); a perfect switch's lead over
    the record's hinge always is about 29.6, so power is about 1 at 5 seeds unless a false switch occurs.
  - A-K1, A-K3: round 1's small-evidence leads (BCA over split conformal 1.78, over the rank test 4.29, half of that from
    one seed, `ROUND1.md:101-113`): skewed by rare false alarms; expected fail.
  - A-K2: round 1's ladder lead +23.0, +15.7 to +33.1 when any one seed is dropped, one seed at -17.2 (`ROUND1.md:74-96`):
    seed sd about 15-20, so at Holm's first step for two claims (t = 2.776 at 5 seeds) the mean must reach about 19-25.
  - C-K1: 0.1-1.9 (A2 - F0 per seed 0.04-0.33 return units, CONN `arms_abc_f0/RESULTS.md:122-123`; step 2's A1 1.9,
    `addon_report/report.md:185`). C-K2, C-K3: 1.9-6.4 (`report.md:185-187`). C-K4, C-K6, C-R1, C-R1b: 8.8
    (`report.md:161`).
  - At Holm's first step for four claims (level 0.0125, t = 3.495 at 5 seeds), the mean must reach 3.0 at sd 1.9, 8.6 at
    5.5 and 13.8 at 8.8 (`review_stats/marks_arith.out` line 5). With a five-claim family at 5 seeds, a true 10 at sd 5.5
    passes 64% of the time, at 10 seeds 99% (REV section 6).
  - The switch's error rate from seed-worlds alone is weak (10 of 10 correct bounds it at 74%), which is why A-V1 and
    B-V1 use 1,000 replicates per seed-world.
- **Size rule (fixed now).** After the pilot, for every primary and credit claim, compute its power at its own bar (the
  smallest effect of interest) and its own level (Holm's first-step level for a family claim), using the upper 80%
  confidence limit of the pilot's seed sd. Confirm on 5 seeds (25-29); add 30-34 if any primary return claim has power
  below 0.8 at 5 seeds and at least 0.8 at 10. Any claim still below 0.8 at the chosen size is labeled "underpowered"
  in the confirmatory registration, before any confirmatory datum exists. The decision is written to
  `size_decision.json`.

### 8.5 B's release power (a pre-registered discriminability step on the pilot)

Release in N needs N's true harm rate `pi_N` well below alpha_B, while the margin rule places m at break-even. With the
exact gate, release passes iff at least 3 of 5 seeds certify (per-seed values about 61.8 or 0). Pass probabilities at 5
seeds (REV `revise_arith.out` section 4):

| Tries per seed | pi_N = 0.02 | 0.03 | 0.05 | 0.07 |
|---|---|---|---|---|
| 50 | 0.88 | 0.60 | 0.14 | 0.02 |
| 70 | 0.97 | 0.76 | 0.18 | 0.02 |
| 100 | 1.00 | 0.95 | 0.38 | 0.03 |
| 200 | 1.00 | 1.00 | 0.94 | 0.24 |
| 300 | 1.00 | 1.00 | 1.00 | 0.61 |

Per-seed certification of at least 0.9 needs 76 tries at pi_N = 0.02, 103 at 0.03, 154 at 0.04 and 239 at 0.05.

The step, fixed now: on each pilot seed-world in N, compute `pi_N` from the truth population (offline and staged).
Offline: if `P(release passes)` at the pilot's try counts and median `pi_N` is below 0.8, B-R1's release is marked
"underpowered by design" in the confirmatory registration. Staged: N is the smallest of 100, 150, 200 and 300 whose
per-seed certification is at least 0.9 at the pilot's median `pi_N`; if none, N = 300 and B-R2's release is marked
"underpowered by design". The probing cost in S grows with N (about a tenth of the value of caution per 100 probe
episodes) and is reported.

---

## 9. Stages and compute

### 9.1 Training runs per seed (100,000 outer steps each unless stated)

| Runs | S | N | MW+ | MW- | Cost class |
|---|---|---|---|---|---|
| Host (Unifloral TD3+BC) | 1 | 1 | 1 | 1 | host |
| Clone (Unifloral BC) | 1 | 1 | | | light |
| A's gated hinge (k = 1) | 1 | 1 | | | host |
| C-BCA | 1 | 1 | | | with BCA refreshes |
| k = 1 references C-SC, C-RATIO, C-F0, C-SHUF | 4 | 4 | | | with BCA refreshes |
| Matched C-F0_m, C-SC_m, C-RATIO_m, C-SHUF_m: 3-rung ladder + matched run + about 1.5 rounds in S; locked in N | about 22 | 4 | | | with BCA refreshes |
| Matched C-HINGE, C-EVEN, C-SUP | about 16.5 | 3 | | | C-family |
| C-RAND1, C-RAND2: matched on the pilot; locked coefficient on confirmation | about 11 (pilot) or 2 | 2 | | | C-family |
| C-ORACLE (co-trained, x64 step) | 1 | 1 | | | oracle |
| Actor-only, 20,000 steps: C-I2 (C-BCA, C-F0, C-HINGE), C-ORACLE-S1, C-ORACLE-V | 5 | 5 | | | actor-only |
| Once per stage: C-k0 and the second training of the first host (S3) | 2 | | | | |
| **Total per seed** (besides the 2 runs once per stage) | | | | | **about 88.5 (pilot), 79.5 (confirmatory)** |

Extra matching rounds beyond the expected 1.5 (at most 5 per control) are covered by a 20% allowance. With K3 failed,
the matched controls are not trained (section 9.3).

### 9.2 Compute per stage

Measured speeds: on this container (4 CPUs, no card), Unifloral's TD3+BC step restated on made-up arrays runs 13.8-14.5
ms per outer step with four runs side by side (about 23.5 minutes per 100,000 steps), 16.3-16.8 ms with a BCA-like
per-step fit (about 27.5 minutes), and 14.0-14.1 ms with the lower edge (`bench/conc_*.out`); that benchmark left out
C's real per-step work (the recorder density, the threshold-table lookup, eta and Q1 at the recorded command). On the
laptop's card the hosts measured 0.58 ms alone and 1.26 ms with BCA, and 3.5-5.2 s per threshold table on a 900-row
bank (HOSTS PRESPEC.md:601-604): on the card BCA's work more than doubles the step, while on this CPU it adds 18%. Stage
1's card runs implied about 0.5 minutes of fixed cost per run (HV2 PRESPEC.md:880-915). An evaluation of one policy on
one bank with its capped twin takes about 2 s; one online episode 11-17 ms; `Q^pi` for 1,000 pairs about 9 s
(`bench/rollout_*.out`); one A-BCA decision about 35 ms (`review_feas/bench_wbcp.out`).

**Cost rule (fixed now).** Stage 0 benchmarks every C penalty kind on the card with its full per-step work and the 18
refresh tables. Every C-family run is costed at the larger of its measured rate and 1.26 ms per outer step, plus 0.5
minutes of compilation, plus the tables for runs with BCA refreshes. Card-hours are card wall time (HOSTS `budget.py:8-14`).

| Stage | What | Laptop card for training, CPU for evaluation | CPU only (needs a built training environment, D11) |
|---|---|---|---|
| 0, code tests on made-up data | unit and reproduction tests, the two-environment tests, smoke runs of 2,000 steps, the card benchmark, freeze | about 3-5 card-hours and 20-25 CPU-hours, booked separately (the hosts' smaller stage 0 took 3.8-4.5 and 15-19, HOSTS PRESPEC.md:574-586) | about 20-25 CPU-hours |
| Pilot, main, seeds 0-4 (exploratory) | about 88.5 runs x 5; margins; replicates; online episodes; coverage | about 20.9 card-hours (25.1 with 20%) + about 42 CPU-hours | about 170 (204) + 42 CPU-hours |
| Confirmatory, main, per 5 seeds | the same, frozen, after approval | about 18.9 (22.7) card-hours + about 42 CPU-hours | about 152 (183) + 42 CPU-hours |
| Module W, per 5 seeds (section 10) | about 51.5 runs per seed | about 14.1 (17.0) card-hours + about 15 CPU-hours | about 112 (134) + 15 CPU-hours |
| Trimmed main (decision D14) | C-RATIO_m and C-SHUF_m reported at k = 1 instead of matched | pilot 16.8 (20.1), confirmatory 14.8 (17.8) card-hours | 140 (168), 123 (147) CPU-hours |

(All from REV `budget_arith.out`.) Draft 1's 6-8 card-hours per 5 seeds charged C's runs at the bare host rate, left out
the refresh tables, and had 29 fewer runs per seed; the review's re-costing of draft 1's own run list already gave
7.9-11.3 (`review_feas/card_arith.out`).

Evaluation per seed (CPU, about 8-9 hours): replicate held-out blocks (about 1 hour); A's replicates, shuffles, sweeps
and the size table (about 0.7); the ladder decisions (about 1.4); staged deployments with their replicate probe sets
(about 0.7); online scoring (about 0.4); C-V1 on the refresh bands (about 0.5); C-V3's replicate banks (about 3.2: 200
banks x 7 refreshes x 2 worlds x 2 bands at about 4 s per table); truths and attribution (about 0.5).

**Budget caps (proposed for your approval):** card route: pilot main 26 card-hours + 50 CPU-hours; confirmatory main 24
card-hours + 50 CPU-hours per 5 seeds; module W 18 card-hours + 18 CPU-hours per 5 seeds; stage 0 booked separately.
CPU route: 260, 240 and 155 CPU-hours. Stop rule S10 checks the projected cost against the cap before a stage starts.

**Where and when.** Card runs go one at a time: about 42 card-hours per stage with module W. The laptop card is shared
with the hosts' version 2 restart (HV2 PRESPEC.md:11-16): the placement pilot runs after the hosts' stages, or under a
time-share you set, and laptop evaluation never overlaps card training (keep heavy processor work off the laptop while
card runs go, HOSTS PRESPEC.md:598-604), or evaluation runs in the cloud scoring environment. On the CPU route a 25-29
hour job cannot run as one background command here (two-hour limit), so the driver is resumable per whole run with a
crash-safe ledger (HOSTS `budget.py:21-30`), under an environment signature like the hosts' S9: Unifloral's Dockerfile,
Python 3.10, JAX 0.6.2, numpy 1.26.4, d4rl at its pinned revision, mujoco-py with the MuJoCo 2.1 binaries, plus the
scoring environment with gymnasium 1.0.0.

### 9.3 Order

1. **Stage 0** on made-up data; `freeze_pilot.py`; your approval of stage 0's summary and of the pilot.
2. **Pilot:**
   1. data and counts (commands only: `t_lo`, group sizes, B's conditioned tries, rung tries, cell counts);
   2. hosts, clones and gated hinges (S and N first, then MW+ and MW-); lock;
   3. replicate blocks (scoring environment), then every offline score (one hand-off round trip);
   4. the margin rules (they read the ladder's returns): m, m_f, m_hinge, m_on. The pilot's A and B decisions are
      therefore not blind; they are written to `decisions.json` right after the margins, before any other return;
   5. K1, K2, K4;
   6. C-ORACLE and the actor-only oracles for every seed; K3 pooled;
   7. C's arms, matching, lock (matched controls only if K3 passed);
   8. all online episodes, staged deployments, truths and ladder readouts;
   9. the pilot report, then the declared slots: the size decision, B's discriminability step, the best plain caps, the
      C-RAND coefficients, the budget.
3. **Confirmatory registration**: `freeze_confirm.py` with the filled slots; your approval at your terminal, before any
   confirmatory datum exists.
4. **Confirmatory seeds:** data and counts; hosts, clones and gated hinges, lock; replicate blocks and offline scores;
   `decisions.json` with the frozen margins, **before any return of any policy is read**; then K1 and K2; C-ORACLE and
   the actor-only oracles for every seed, K3 pooled (C-BCA is trained in any case, since its coverage claims stand
   without K3; matched controls only if K3 passes); C's arms, matching, lock; online episodes, staged deployments,
   truths and the ladder; verdicts only when every seed is complete.

### 9.4 Freezes and changes

- `freeze_pilot.py` writes `prespec_pilot.json` and the code manifest, and refuses to run again once any pilot output
  exists (the pattern of step 1, MAIN `outputs/shift/beyond-the-data/restructure/step1/PRESPEC.md:5-8`).
- `freeze_confirm.py` runs only once the pilot report exists, hash-chains to `prespec_pilot.json`, and may fill only the
  declared slots: m, the m_f, m_hinge, m_on; the size decision; staged B's N and the "underpowered by design" labels;
  the best plain cap (W and A-K4); C-RAND's coefficients; the budget caps.
- Any other change after the pilot goes into `CHANGES.md` with its reason and the pilot result that prompted it, and
  needs its own approval fingerprint (the pattern of HOSTS `gate.py:7-22`). Results pages flag every claim whose mark
  changed after the pilot.

---

## 10. Module W: the known shift (recommended; decision D4)

**Why.** BCA's weights can earn credit only where the evidence law and the deployment law overlap with a known ratio.
On the main log the recorded current is U[0.1, 0.7] and strong current U[0.7, 1.1], so the supports are disjoint and no
weight exists (`thruster_chain.py:87-92`). The weighting test's world has an exact current weight `w(c)` on [0.1, 1.1]
(MAIN `experiments/shift_iw/world.py:7-16`), and its recorder's random-command part (`eps / 16`, CONN
`channel/command_le_train.py:73-84`) fills the rungs past 80% (on seed 0 arm B had 42, 54 and 78 thruster-1 tries in
the 80-85% rung on medium-replay, medium and medium-expert, and 35-67 in each higher rung on medium-expert; CONN
`outputs/shift_iw/explore/arms_abc_f0/SCREEN_seed0.md:41, 46-52`). So module W is also the only place where C's band can
act past the knee (section 2.3). It is stalling-only (`world.py:69`; `data.py` hard-codes the stalling world at :80,
336, 454, 646 and 652), so a no-stall twin must be built: the generator with `PHYSICS` passed in (`world.simulate` and
`HoldRecorder` already accept physics, `world.py:218, 324`).

**Design.** Quality medium-expert (the hosts' primary quality); weighting-test seeds 0-4 (pilot) and 25-29
(confirmatory); Unifloral TD3+BC hosts in S and the new twin N; tries = training-law log rows of the calibration split
outside the bank, or the evaluation split, outside the ball (`arm_b.py:8-9`); deployment = the strong-mix law; weights
exact `w(c)` (B-K2, C-K5) or estimated by a classifier on the weighting test's classifier samples (B-K3). NET in module
W is defined as in section 6.1, under the strong-mix law.

- **B arms**: B-RUNG with the residual score and BCA's weighted tail gate (the shipped Monte Carlo gate with its floor of
  36 is primary here, because with unequal weights no exact Beta form exists; a 10,000-draw version is reported)
  against rejection sampling (accept each try with probability `w / max w`, then the exact binomial test), weighted
  split conformal (Tibshirani et al. 2019), a self-normalized weighted mean with an empirical-Bernstein bound, the
  unweighted gate, and shuffled outcomes.
- **C arms** (S and N): C-BCA with exact state x command weights for the strong mix (primary); matched in S and locked
  in N: unweighted C-BCA, C-F0_m, C-RATIO_m, C-SC_m, C-SUP, C-HINGE, C-EVEN; C-ORACLE. About 51.5 runs per seed (9.2).
- **B-K2 / B-K3 pass iff**: in the stalling world BCA's weighted gate makes harmful certifications (true harm rate above
  alpha_B, from 1,000 replicate try sets per decision) at most 5% of the time (B-V1's pooled mark: upper bound <= 0.065,
  at most 73 in 1,000 per decision); every
  rival above that rate is reported invalid and dropped; each remaining rival's level is set on the pilot so that its
  stalling-world rate of harmful certifications equals BCA's (matched harm), frozen in a confirmatory slot; and in the
  no-stall twin BCA's weighted gate certifies at least 10 percentage points more truly safe rungs than the best
  remaining rival at matched harm (pooled over seeds, Holm p <= 0.05 in {B-K2, B-K3}). With exact weights the toys gave
  33.6% against 19.0% for rejection sampling (LIVE 10811); evidence floors are about 29 / 39 / 83 tries for no, moderate
  and strong shift (LIVE 10987-10992).
- **C in module W**: C-R1W, `NET(C-BCA) - NET(pooled-best of C-HINGE, C-EVEN and clip always)`: mean >= +5 and LB > 0;
  C-V1W, C-V3W as on the main recorder; C-V2W judged overall, in the near-knee zone and **past 80%** (RS <= 0.50 and CAC
  >= 0.85 there; no certified draw past 80% fails it). Machinery family {C-K1W vs C-F0_m, C-K2W vs C-RATIO_m, C-K3W vs
  C-SC_m, C-K4W vs C-SUP, C-K5 weighted vs unweighted C-BCA}: each mean >= +5 and Holm p <= 0.05; C-K5 also needs C-V1W
  met by both arms.
- The weighted promise is empirical, not proven, at these sizes (PLACE `red.challenges[3]`; MAIN
  `experiments/wbcp/DEPENDENCE.md:1114-1133`); module W measures it, bank by bank (C-V3W).

---

## 11. Code plan

### 11.1 Base branch, reuse and what is not reused

- **Base.** A new branch from `origin/claude/shift-hosts` at `608eefa` (the hosts' approved version 1 bytes), with
  `origin/main` merged. HOSTS does not contain MAIN (7 MAIN commits are missing), but between their merge base and MAIN
  no file under the hosts' `EXTERNAL_CODE` paths differs (checked with `git diff`), so the hosts' approval still
  verifies there. HV2 edits `actors_f32.py` and `evaluate.py` (`git diff` HOSTS..HV2), so it is not the base; if the
  hosts' version 2 is frozen later, S1 catches the change through the code manifest.
- **Clone of the repository.** The test runs from a clone separate from the hosts' `/root/bca-hosts`, and every entry
  point sets `sys.dont_write_bytecode` first (stop rule S11).
- **Test references only (never imported)**: CONN `outputs/shift_iw/explore/arms_abc_f0/code/arm_b.py` and `arm_f0.py`;
  MAIN `step1lib.py`; MAIN `bca_mean_gate_toy.ipynb` and `bca_weights_toy.ipynb`. They are read with `git show
  <commit>:<path>`, their blob SHA-256 pinned in `prespec.json`, and the package's restatements are tested against them.

**Reused and imported (none of these calls a study's gate):**

| Need | Where |
|---|---|
| Simulator, start laws, banks, env (scoring environment) | MAIN `experiments/shift/thruster_chain.py` (`dynamics`, `dynamics_jax`, `draw_starts`, `bank_starts`, `world_physics`, `ThrusterChainEnv`) |
| Exact truth | MAIN `experiments/shift/truth.py` (`rollout`, `q_value`, `q_value_jax`, `points`, `attribution`); not `anchors`, which takes no starts |
| Data, gap, bank | MAIN `experiments/shift/dataset.py` (`generate`, `simulate`, `draw_streams`), `shift.py` (`define_gap`, `make_removal`), `recorder.py`, `calibration/bank.py` |
| WBCP | MAIN `calibration/wbcp.py` |
| Unifloral import | HOSTS `experiments/shift_hosts/unifloral_import.py` (`prepare`, `args_for`, `make_transition`, `init_like_training_block`, `make_unifloral_step`); MAIN `baselines/unifloral/algorithms/td3_bc.py`, `bc.py` |
| BCA arithmetic | HOSTS `bca_core.py` core functions (seed and key rules :96-157, `fit_scale_core` :282, `width` :364, `bank_scores` :387, `calibrate_bank` :399, `apply_refresh` :417); `unifloral_td3_bc_bca.py` `_own_target` :316, `_band_values` :343-349 and `band_parts` :352-359 as templates for the Q1 version |
| Threshold tables (training environment) | HOSTS `iw.py` (`calibrate` :94, `mass_grid` :136, `threshold_table` :157, `lookup` :180) |
| Policies and hand-off | HOSTS `policies.py` (`MLPAdapter`), `actors_f32.py`, `handoff.py` |
| Pure helpers | HOSTS `band_readout.py:85-157` (`width`, `refusals`, `outcomes`, `episode_bootstrap`, `summarize`, `groups`, `in_ball`); `evaluate.py` `paired` :356, `late_peak` :275, `costs` :282 |
| Coverage arithmetic (scoring environment only; it imports `shift_iw.world`, `coverage.py:42`) | MAIN `experiments/shift_iw/coverage.py`, with COR computed in `coverage_c.py` |
| Module W data | MAIN `experiments/shift_iw/data.py`, `world.py` |

**Not reused** (each calls a study's gate, or does not fit): `make_train_step` with BCA present, `refresh` and
`recompute_bank_scores` (HOSTS `unifloral_td3_bc_bca.py:243, 392, 408`); `iw.threshold_tables` and `iw.iw_refresh`
(`iw.py:266, 485`); `band_readout`'s `band_values`, `readout` and `coverage` (:296, 331, 354); `evaluate`'s
`score_actor` and `closed_loop_check`, and `score`, which knows only the standard and weighting banks with the base and
stall anchors and needs the laptop's scoring cache (`evaluate.py:171-263, 313`); `runner` and `stages`; `lower_edge.py`'s
band readers and `Runner`, which call the weighting test's gate (`lower_edge.py:19-22`). The hosts' gate opens only on
registered made-up arrays or on a hosts' Approval (HOSTS `guard.py:127-136`). A stage-0 test fails if any placement
module calls `guard.require_bca`, `gate.require_run_approval`, `gate.verify` or `learners_iw.require_gate`, or builds a
hosts' Approval. The package lives at `experiments/placement_test/`, outside the hosts' fingerprint list (HOSTS
`gate.py:56-65`); nothing is added under `experiments/shift`, `experiments/shift_iw` or `calibration`, and the hosts'
approval is not reused for this study.

### 11.2 To write

| Module | Does |
|---|---|
| `PRESPEC.md`, `prespec_pilot.json`, `freeze_pilot.py`, `freeze_confirm.py`, `CHANGES.md`, `gate.py`, `approve_placement.py` | the frozen rules, the two freezes, recorded changes and your terminal approval |
| `seeds.py` | the seed allow-list (S2); the stream registry with the padding check (section 3.5) |
| `data_main.py` (scoring) | gapped logs in S and N; spliced MW+ and MW-; evidence; B's conditioned tries and decoupled rest; per-row `mu(x)`, hard-water flags and `Z(x)`; Unifloral prepare and hand-off; leak checks |
| `hosts.py` (training) | the chunk loop (restating `td3_bc.py:329-341`); TD3+BC runs; `load_bc()` and BC runs in their own process; run records; a resumable per-run ledger |
| `dynamics_train.py` (training) | the restated dynamics from handed-off columns, under x64 (section 3.1) |
| `scores.py` (training) | the residual score on a Unifloral critic (min head, Q1, noisy-target variants) |
| `harm_switch.py` | A's rules at their own margins, shuffles, A-RAND by expectation, cells, kernel, A-KQ, A-KPERM, boundary shifts |
| `safety_cert.py` | B's exact and shipped gates, mean gates, HCOPE-style, looks, e-process, rung walk, maps, decoupled tau, boundary shifts |
| `recorder_density.py` (both) | the recorder's density, `Z(x)` by Monte Carlo (scoring), the query law's density and the weights (training), with the cross-environment test |
| `lower_edge_unifloral.py` (training) | the restated TD3+BC step with the lower-edge slot; Q1 band parts; its own refresh with per-query tables; C's penalties, C-SHUF and the oracles; matching; saving the networks of each declared refresh |
| `online.py` (scoring) | the episode runner (env.step arithmetic); the float32 wrapper for truth calls; mixed physics; fresh banks; `anchors_on(starts, physics, recorder_stream)`, tested to reproduce `truth.anchors` on the standard bank bit for bit; the no-stall scorer; the staged-B loop and the monitor replay with hand-offs; attribution |
| `truth_pop.py` | replicate blocks (split by physics for mixed worlds), staged-B replicate sets, populations, boundary shifts |
| `coverage_c.py` | COR, RS, CAC with soft refusals, the zones, C-V3, sharpness |
| `s4_checks.py` | the package's own open-loop and closed-loop float32 checks (stop rule S4) |
| `stats.py` | paired t bounds, Holm (the hosts' procedure), Clopper-Pearson, pooled marks with guards, bootstraps (episode, two-stage, unit), design effect, the size rule, the discriminability step |
| `report.py` | `results.json` and `results.md`, write-once |
| `tests/` | unit tests on made-up data for every module (stage 0): the gate-call test, the cross-environment tests, the weight identity, COR with refusals, stream distinctness, `anchors_on`'s reproduction, k = 0 reproduction |
| Module W only | the weighting-test no-stall twin; its prepare; the weighted gates |

### 11.3 Outputs and report

- Official outputs only under `runs/placement_test/<stage>/`, write-once, each with a SHA-256 sidecar; run records hold
  the fingerprint, seeds, streams, CPU or card, software and timings; every process pinned to one CPU with
  `OMP_NUM_THREADS=1` and no bytecode (step 1 PRESPEC.md:107-113).
- `decisions.json` (locked before any return in the confirmatory stage), `lock.json` (network fingerprints and
  strengths), `values.json` per seed-world, `results.json`, `results.md`. Until a stage is complete, only integrity checks
  and spend are shown.
- `results.md`: per claim a badge (passed, shelved, flagged, pending, not judged, no room, did not learn, vacuous,
  underpowered by design, dominated), the numbers against the mark, per-seed values, and the fixed wording; the three
  verdicts per option; then per option the write-up in your format: The idea / What we tested (and the kill rule) / What
  happened / Why (HOSTS PRESPEC.md:515-516). Every pilot page carries "EXPLORATORY: seeds 0-4 were seen; the margin was
  fitted on these seeds".

---

## 12. What this design rests on (verified for this document)

- Step 1's certificate, rules and process: MAIN `outputs/shift/beyond-the-data/restructure/step1/PRESPEC.md:5-8, 51-63,
  133-168, 192-268, 246-251, 340-347`; code `step1lib.py:174-352`.
- Round 1's routes (held-out rows 106.1; 55.7 without the margin; plain tests at the margin also 106.1; ladder +23.0
  over the middle strengths; small evidence; mixed world 43.2 against a cap at the knee 41.5 and a region oracle 52.6;
  run-time monitor): MAIN `restructure/routes/round1/ROUND1.md:13-146`.
- Step 2's arms, matching (1%, at most 5 rounds), marks and results (NET 102.9; best world-blind rival 77.7; conformal
  97.3; range hinge 76.5 with no-stall cost 30.9; "Bayesian or only larger cannot be told"): MAIN
  `restructure/step2/PRESPEC.md:97-257`; CONN `restructure/step2_results/results.md:9-38` and
  `addon_report/report.md:138-230`.
- Fallback: deployment clip +106.0 in strong current, -61.8 in the no-stall world; every step to the cloned actor +68.9:
  MAIN `outputs/shift/beyond-the-data/fallback.md:83-88, 167-178`. Gated clone fallback +73.8 / -30.4 and no-stall
  -109.5 against the clip's -61.6: HV2 `outputs/shift/beyond-the-data/results-2026-10-13/deploy_tables.md:17, 33, 79,
  95`.
- Stage 1's room (oracle 79.7, exact long-run harm 82.2, no-signal clip 93.1, no-stall costs 1.3 and 30.9; actor-only
  against a frozen critic): MAIN `stage1/RESULTS.md:28-65`; `stage1/code/harness.py:373-376, 530-533, 559, 791`.
- Stage 2 v6 (lower edge 24.9 against the matched hinge 70.5; no-stall cost 14.8; co-trained 28.0 against -16.8):
  MAIN `results-2026-10-14/README.md:20-36, 69-77`.
- ThrusterChain, truth, data, recorder: MAIN `experiments/shift/thruster_chain.py:1-53, 83-92, 143-149, 195-251,
  308-336, 389-481`; `truth.py:1-35, 77-108, 330-383`; `dataset.py:4-15, 50-65, 152-226`; `recorder.py:8-60, 210-270`;
  `shift.py:1-60`; `experiments/shift/__init__.py:81-111`.
- WBCP and its promise: MAIN `calibration/wbcp.py:1-117`; `experiments/wbcp/DEPENDENCE.md:1-16, 1110-1135`;
  `experiments/shift_iw/coverage.py:1-60`.
- Unifloral and the hosts: MAIN `baselines/unifloral/algorithms/td3_bc.py:23-46, 174-279, 326-342`, `bc.py:1-22`;
  `configs/unifloral.json`; HOSTS `experiments/shift_hosts/PRESPEC.md:138-226, 254-282, 466-606, 613-621`,
  `unifloral_td3_bc_bca.py:1-40, 102-413`, `bca_core.py:90-420`, `iw.py:60-490`, `band_readout.py:40-465`,
  `evaluate.py:1-125, 302-362`, `unifloral_import.py:55-76, 186-234`, `gate.py:1-70`, `guard.py:115-136`; HV2
  `PRESPEC.md:1-46, 866-930`; `git diff` HOSTS..HV2 and the merge base of HOSTS and MAIN.
- Arm B and the lower edge: CONN `arms_abc_f0/code/arm_b.py:1-155`, `arms_abc_f0/RESULTS.md:40-48, 118-126`;
  `channel/command_le_train.py:1-18, 73-118`; `channel/command_band_check.py:1-130`; MAIN
  `experiments/shift_iw/lower_edge.py:1-64, 196-224, 293-312`.
- Weighting-test world: MAIN `experiments/shift_iw/world.py:1-70, 218, 324`.
- Seeds: CONN `near80/code/near80lib.py:64-72`; `seed_registry.json`; mixed-world counts
  `routes/round1/scratch/_where/feas_amp_0_1_2_3_4.json`.
- Floors, power, marks and precision: `measure/floor_arith.out`; REV `revise_arith.out`, `budget_arith.out`;
  `review_cov/arith.out`; `review_stats/marks_arith.out`; `review_feas/card_arith.out`, `bench_wbcp.out`; LIVE 5104-5112
  (bank standard errors), 10502-10560 (tail-gate floor; 36 with 1,000 draws), 10985-10996 (weighted floors).
- Placement analysis and its red team: PLACE (theory findings 3, 5, 9; rank test_plan T1-T6; red challenges 1-13,
  missing options, revised recommendation).
- Benchmarks: `bench/conc_alone_*.out`, `conc_bca_*.out`, `conc_lower_*.out`, `rollout_1cpu.out`, `rollout_4cpu.out`.
- numpy key padding: checked in the scratch environment (numpy 2.2.6): `default_rng([20261019, 7, 25, 0])` and
  `default_rng([20261019, 7, 25])` give the same first integer; with five or more words a trailing word changes the
  stream.

---

## 13. Review round

Four reviews of draft 1 were read: coverage and validity (R1, 18 findings), methods and credit (R2, 18), feasibility
and code (R3, 17), statistics (R4, 17). Each finding was re-checked against the sources before it was applied. "Applied"
means as proposed; "applied in part" names what was changed or left out and why; "rejected" gives the reason.

**R1, coverage and validity**

| # | Finding | Status and where |
|---|---|---|
| R1.1 | C-V1 judged only average coverage; BCA promises coverage per bank | Applied: C-V3, primary (6.4, 7.4), also for C-SC and C-RATIO, with the per-bank credit reported. Part (ii) as a mark rejected (kept as a readout): an exact-95% rule fails "at most 1 of 10 seed-worlds" 8.6% of the time (REV section 2), and it adds ten draws to the 2,000 units of part (i). |
| R1.2 | C-V1's band is one that nothing uses | Applied: judged on the bands of 7 declared refreshes with that refresh's networks; staleness and the final band reported (5.1, 6.4). |
| R1.3 | Staged B's threshold and tries share episodes; B-V1 assumed a fixed tau | Applied in part: per-replicate error at the replicate's own tau (6.4); offline tau from calibration episodes that supply no try (3.2). Changed: staged tau comes from 50 fallback reference episodes, not 50 probe-map episodes; they are equally independent of the tries, cost no probing, and F runs first anyway (5.3). |
| R1.4 | B-V2 judged an event the certificate never promised | Applied: B-V2 is the share of departing steps with `s_t > tau`; the attribution shares and harm mass are readouts (6.4, 7.3). |
| R1.5 | The query law's states give the weights a spurious Z(x) | Applied: kept-row states, `w = q Z / beta` (section 1, 4.3). Corrected: the proposed check "mean bank weight is 1" fails whenever the query law has mass the recorder never covers; the identity is `mean(w) = 1 - P_query(w = +inf)`. |
| R1.6 | C-V2 can be passed by soft refusal and easy draws | Applied: soft refusals counted; judged overall and in the near-knee hard-water zone; sharpness readout (6.4, 7.4). Past 80%: reported on the main recorder, where refusal is the support rule by construction, and judged in module W (with R2.4). |
| R1.7 | B-V1 is vacuous far from the boundary | Applied in part: a judged boundary null at `pi(tau_pop) = 1.1 alpha_B` (6.4, 7.3); alpha_B, 1.2 and 1.5 alpha_B and the in-band ladder decisions are reported, not judged, since the judged shift sits next to the hardest point and more conjunctions would only fail exact rules more often. "Vacuous" badge applied to B-V2; a boundary null cannot be vacuous. |
| R1.8 | B-K1 confused "no outcome information" with the evidence floor | Applied: the permutation p-value test (7.3). |
| R1.9 | m_on was set on a different population from its tries | Applied: first uses against fallback reference episodes, as the tries and tau are built (3.6 step 7). |
| R1.10 | Looks at 10/30/100 cannot certify before the last | Applied: looks at 0.4N, 0.7N, N (40/70/100 at N = 100) (5.3). |
| R1.11 | "A true 5% passes" is false for every-seed-world marks | Applied in part: pooled bound <= 0.065 with a per-unit guard (7.2, 7.3). The guard is 73 in 1,000 (99.9th percentile), not 66 (99th): over 10 units a 66 guard alone fails an exact-5% rule 10.1% of the time, 73 only 0.65% (REV sections 1, 12). |
| R1.12 | A-K3's expected NET for A-INS cannot be computed | Applied: realized NET; a bootstrap E-NET reported, labeled (7.2). |
| R1.13 | Attribution runs on other arithmetic than the episodes | Applied: float32 wrapper before every truth call; S8 extended (5.2, 7.1). |
| R1.14 | Staged B-RUNG horizon, parallelism and truth unspecified | Applied: parallel probing, at most 2 families (with R3.10), truth per realized map (5.3). |
| R1.15 | The reused coverage code counts refusals as misses | Applied: COR in `coverage_c.py` with a stage-0 test (6.4). |
| R1.16 | Fresh-bank anchors cannot come from `truth.anchors` | Applied: `anchors_on`, with a registered recorder stream and a reproduction test (3.5, 11.2). |
| R1.17 | World in the stream tags breaks shared random numbers | Applied: no world in tags 4, 5 and 6 (3.5). |
| R1.18 | Per-seed values appear before a stage completes | Applied (8.1, 11.3). |

**R2, methods and credit**

| # | Finding | Status and where |
|---|---|---|
| R2.1 | "BCA earns credit" could rest on checks that pass almost automatically | Applied: three verdicts (7.5); A-K2 is now the ladder claim on confirmatory seeds; the shuffle is a sanity check; C-K6 counts as outcome reading. |
| R2.2 | Clipping always is not the best world-blind rule | Applied: the set W (section 1, 4.1); A-R1 against its pooled-best; W reported beside B; domination rows (7.2-7.5). |
| R2.3 | C's primary bar let C "work" while far below A's switched clip | Applied: C-R1 against the best switched plain action; draft 1's C-R1 kept as C-R1b (reported); domination rule (7.4, 7.5). |
| R2.4 | On the main recorder C is refusal alone past 80%; module W lacked controls | Applied in part: the construction declared (2.3); module W's C arms extended to the matched set, one Holm family, recommended (10, D4). C-V2 past 80% is judged in module W but only reported on the main recorder: there it fails by construction, which the declaration already states. |
| R2.5 | Mean and shift rivals used a quantile-unit margin | Applied: per-statistic margins m_f (3.6, 4.1, 4.2). |
| R2.6 | C-K1 to C-K3 compared arms of different strength | Applied: matched C-F0_m, C-RATIO_m, C-SC_m; k = 1 versions reported (4.3, 9.1). |
| R2.7 | A-K4 lacked rivals, a metric, a reference and the decisive control | Applied in part: NET_mixed, A-ORACLE-REGION, A-KQ, A-KMEAN at its own margin, A-CELL and per-cell A-SC in the rival set, the A-KPERM condition (4.1, 6.1, 7.2). Rejected: the local cap (a cap where A-KERN is on) as a rival: it reuses A-KERN's own decisions, so it compares actions, not localizers; it is reported as A-KERN-CAP. |
| R2.8 | Validity checks far from the hardest case; A-V2's target | Applied: boundary nulls, the rivals' size table; A-V2's target declared (the 90%-quantile gap judged through A-V1, because the margin is in that unit; the n-matched gap reported with a wording row) (6.4, 7.2). |
| R2.9 | C had no shuffled-outcome control | Applied: C-SHUF_m and C-K6 (4.3, 7.4). |
| R2.10 | C's weights dropped the per-state factor | Applied (with R1.5). |
| R2.11 | A-K3's E-NET for A-INS | Applied (with R1.12). |
| R2.12 | B's primary gate was the Monte Carlo one | Applied on the main generator: the exact gate is primary, the shipped gate reported (4.2). Not in module W: with unequal weights no exact form exists, so the shipped weighted gate stays primary there (10). |
| R2.13 | B-K1 can fail for too few tries | Superseded by R1.8: the permutation test does not depend on certifying. Both certification rates are reported. |
| R2.14 | The margin depends on the action and the critic | Applied: m_hinge and A-PERFECT_hinge; S-host ladder readout (3.6). |
| R2.15 | B-GROUP's offline tries were the recorder's commands | Applied: tries conditioned on the host's command at the same state; unconditioned reported (3.2). |
| R2.16 | Clone-fallback numbers mis-cited | Applied: deploy_tables.md:17, 33, 79, 95 (4.1, 12). |
| R2.17 | Module W's mark failed BCA when a rival was unsafe, and was not at equal harm | Applied (10). |
| R2.18 | C-ORACLE changed stage 1's oracle; the value-error oracle was missing | Applied in part: relabeled; stage 1's one-step form (C-ORACLE-S1) and long-run form (C-ORACLE-V) added actor-only; K3 uses the largest (4.3, 7.1). Changed: C-ORACLE-V uses stage 1's own `Q1 - Q^c_250` (`harness.py:559`), not `Q1 - Q^pi` inside co-training, which would need a 1,000-step rollout per batch row per update for a moving actor. |

**R3, feasibility and code**

| # | Finding | Status and where |
|---|---|---|
| R3.1 | C-ORACLE needed the simulator inside training | Applied: `dynamics_train.py` as a declared exception with a cross-environment test, x64, S3 exemption; the actor-only alternative also run (3.1, 4.3). |
| R3.2 | The card estimate was low; the cap would trip S10 | Applied: cost rule, stage-0 card benchmark, caps re-derived (9.2). The suggested trimming taken in part: 3-rung ladders and C-RAND matched on the pilot only; C-EVEN stays matched, since it is the plain alpha retune you rejected; a trimmed scope is offered (D14). |
| R3.3 | No single branch held the code | Applied: base branch and test references (11.1). |
| R3.4 | Reused host modules refuse without the four-learner approval | Applied: the reuse and not-reused lists, the gate-call test, own entry points (11.1, 11.2). |
| R3.5 | This container cannot train Unifloral | Applied (3.1, 9.2, D11). |
| R3.6 | The stage order read returns before the lock | Applied (9.3). |
| R3.7 | Fresh anchors, mixed points and NET_mixed undefined | Applied (with R1.16, R2.7); NET_mixed is a sum, as NET is (6.1). |
| R3.8 | Online residuals need the critic across the split | Applied: hand-off protocol (5.4). |
| R3.9 | The weights' form and the smoothing law's clip atoms | Applied: the two ratios agree once the state law is fixed; the kept-row law is chosen because the band is used on training batches (section 1). The inner clip is dropped from the declared law. The density is computed in the training environment from handed-off inputs, with a cross-environment test (4.3). |
| R3.10 | Evaluation cost missed staged-B replicates; runs missing | Applied (9.1, 9.2; B-RUNG staged capped at 2 families). |
| R3.11 | Bytecode in `experiments/` would block the hosts | Applied: S11, a separate clone, `sys.dont_write_bytecode` (7.1, 11.1). |
| R3.12 | The laptop card is shared with the hosts' restart | Applied (9.2, D11). |
| R3.13 | Keys with a zero last word collide | Applied (3.5). |
| R3.14 | The clone needs its own loader and process | Applied (3.1). |
| R3.15 | Mixed-world tries per cell are thin | Applied: expected counts, a minimum of 10 per cell, replicates split by physics (3.3, 4.1). |
| R3.16 | Stage 0 was under-budgeted | Applied (9.2). |
| R3.17 | Citation slip on the Q1 line | Applied (3.1). |

**R4, statistics**

| # | Finding | Status and where |
|---|---|---|
| R4.1 | A realized random coin fails a perfect switch 21% of the time | Applied: exact expectation (7.2). |
| R4.2 | Validity checks could not fail; every-seed-world marks fail exact rules | Applied in part: judged boundary nulls and pooled marks (6.4, 7.2, 7.3); ladder and nulls in the confirmatory stage too (3.3). A-V1 shifts the group to the margin (R2.8's form) rather than re-dealing pooled scores, because the shift keeps the group's own shape and clustering and gives the exact identity with coverage; the re-dealt null is reported. B-V1 shifts the tries to `1.1 alpha_B` rather than fixing tau at the union's 88% quantile, because a fixed tau would hide the threshold's randomness; the fixed form is reported. |
| R4.3 | The verdict did not separate BCA from plain rules | Applied (with R2.1). |
| R4.4 | B's release had no power analysis | Applied: discriminability step, staged N slot, "underpowered by design" (8.5, 7.3). |
| R4.5 | "L > 0" marks beside Holm; A-K4 outside every family | Applied: Holm-adjusted p in the marks; A-K4 in A's machinery family (7, 8.3). |
| R4.6 | One seed spread for every claim; observed-power size rule | Applied (8.4). |
| R4.7 | C-R2's non-inferiority was underpowered | Applied option (ii), merged with R2.3: replaced by the domination check (7.5). |
| R4.8 | A-K3 undefined | Applied (with R1.12). |
| R4.9 | B-K1 could not pass with few tries; mixed sources; unnamed cell | Applied (with R1.8): cell named, both rates over the replicate sets (7.3). |
| R4.10 | B-V2's event, and steps pooled across seeds | Applied (with R1.4): per seed, two-stage bootstrap (6.4, 7.3). |
| R4.11 | tau not fixed across replicates | Applied in part: the per-replicate threshold is primary (with R1.3) because the deployed procedure computes tau from data; the fixed-tau form is reported; staged tau from separate fallback episodes. |
| R4.12 | The freeze contradicted the two-stage plan | Applied (9.4). |
| R4.13 | A-V2 judged against a quantile L does not target | Applied in part (with R2.8): the functional declared; the 90%-quantile gap judged, the n-matched gap reported. |
| R4.14 | Stream layouts collide; replicate and decision indices missing | Applied (3.5). |
| R4.15 | The seed guard was a deny-list kept in the scratchpad | Applied (3.4). |
| R4.16 | Safety marks and matching inconsistent | Matching applied (1%, at most 5 rounds, step 2 PRESPEC.md:181-182). One safety form rejected: B's safety is its own primary promise and needs a bound, while A's and C's side condition is step 2's G2 copied verbatim; the sd = 0 rule is stated (7.3, 8.1). |
| R4.17 | Undefined or overloaded quantities | Applied: NET_mixed, L_gap and LB, the pooled-best rule, A-RAND's pooled rate, percentage points (sections 1, 6, 7). |

---

## 14. Decisions for the owner

| # | Decision | Recommended |
|---|---|---|
| D1 | Generator for the primary test | the main study's (only one with a no-stall twin, the gap and every earlier A and C result) |
| D2 | Confirmatory seeds | main 25-29, with 30-34 by the size rule (8.4) |
| D3 | Margin source | the break-even rules on the pilot (3.6); alternatives: a fresh exploratory block 20-24 (the pilot's A and B results are then not fitted to their own margin), or the record's 0.10 (no meaning on a new host; not recommended) |
| D4 | Module W | include, medium-expert: the only place for the weights' credit and for C's band past 80%; about 14 card-hours per 5 seeds (17 with the allowance) |
| D5 | Evidence pool for A and B | calibration and evaluation splits; the calibration split alone reported |
| D6 | Where Q1 is read in C's objective | at `pi(x)` (k = 0 reproduces the host); Q1 at the smoothed command as information |
| D7 | C's protocol | co-trained as primary; actor-only arms as references |
| D8 | Mixed worlds | train hosts on the spliced logs (2 runs per seed); alternative: drop A-K4 |
| D9 | Staged B | 50 fallback reference episodes, then a batch decision at N = 100, raised to 150-300 by the pilot's discriminability step; H = 1,000 |
| D10 | alpha_B | 0.1 (arm B's 0.5 reported only) |
| D11 | Compute site and timing | the laptop card for training, after the hosts' version 2 stages or under a time-share you set; evaluation never overlapping card training (or in the cloud scoring environment); the CPU route only if a cloud training environment is built, on its own budget line |
| D12 | A pilot kill rule fires | that option's test ends at the pilot, with its write-up |
| D13 | Near-80 version 8 | it runs on main seeds 10-14; if it reports first, A here is partly a replication with new rivals on a Unifloral host |
| D14 | Scope of C's matched controls on the main recorder | full (recommended); trimmed (C-RATIO_m and C-SHUF_m at k = 1, reported) saves about 4 card-hours per 5 seeds, but then C-K2 is judged only in module W and C's outcome verdict is unavailable on the main recorder |
| D15 | Budget caps | card route: pilot 26 card-hours + 50 CPU-hours; confirmatory 24 + 50 per 5 seeds; module W 18 + 18 per 5 seeds; or the CPU route's 260 / 240 / 155 CPU-hours |
| D16 | Stage 0 | booked separately, about 3-5 card-hours and 20-25 CPU-hours |

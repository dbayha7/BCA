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
| WBCP posterior draws (A, B), replicate r (1 = the actual evidence, 2..1,001 replicates), group g (1 group, 2 rest) | `[20261019, 1, k, w, r, g]` |
| Outcome shuffles, permutation j = 1..200 | `[20261019, 2, k, w, j]` |
| Random-switching coins (information only; the claim uses the expectation, section 7.2) | `[20261019, 3, k, w, s]` |
| Staged B: replicate reference-and-probe sets r = 1..1,000 of decision d (their union is the truth population) | `[20261019, 4, k, d, r]` (no world: S and N share starts) |
| Replicate held-out blocks (A, offline B) and C's replicate banks, replicate r: starts and noise, then coins | `[20261019, 5, k, r, 1]`, `[20261019, 5, k, r, 2]` (no world, as the generator, `dataset.py:152-158`) |
| Staged-B deployment stream of decision d (reference episodes, probes, then the rest) | `[20261019, 6, k, d, 1]` (no world) |
| C's declared-law draws (states, xi, target noise) at declared refresh j | `[20261019, 7, k, w, j]` |
| Episode bootstrap | `[20261019, 8, k, 1, 1]` |
| Ladder point i | `[20261019, 9, k, i, 1]` |
| Tuning bank of B's oracle ceiling map | `[20261019, 10, k, w, 1]` |
| Recording controller's noise and coin on fresh banks and deployment starts (for `anchors_on`) | `[20261019, 14, k, s, 1]` |
| C-SHUF's permutations at refresh j and its eta-target shuffles | `[20261019, 15, k, w, j]` |

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

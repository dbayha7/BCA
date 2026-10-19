# Placement full test: harm switch, safety certificate and value lower edge (test specification, draft)

**Draft for the owner, 19 October 2026. Design only.** Nothing was generated, trained, scored or run on ThrusterChain
data for this design. The only computations were (i) micro-benchmarks of a restated update step and of the rollout
arithmetic on made-up arrays, outside the repository (`placement_test/bench/`), and (ii) closed-form arithmetic for
evidence floors and power (`placement_test/measure/floor_arith.py`, output `floor_arith.out`). Every ThrusterChain
number quoted below comes from an earlier record, and every one of those is exploratory (data seeds 0-4).

**How this document will bind.** Once you settle the open decisions of section 12, this text becomes `PRESPEC.md` of a
new package `experiments/placement_test/`, with `prespec.json` beside it holding the same rules in machine-readable
form. `freeze.py` writes the SHA-256 of both and a code manifest. Every script refuses to run if a frozen file has
changed, and `freeze.py` refuses to run again once any official output exists (the pattern of step 1,
MAIN `outputs/shift/beyond-the-data/restructure/step1/PRESPEC.md:5-8`). Runs on ThrusterChain data wait for your
approval at your own terminal, as the Unifloral hosts do (HOSTS `experiments/shift_hosts/gate.py:1-34`). Pass marks,
decision rules, the margin rule, seeds, banks and random streams are fixed here, before any data are read. Nothing runs
without your go.

**Labels.** The pilot on data seeds 0-4 is exploratory: those seeds were seen by every earlier study. Only the
confirmatory stage, on fresh seeds and under a pre-registration you approve, can carry a claim.

**Source abbreviations.** MAIN = `origin/main`; CONN = `origin/claude/main-bca-connection-x4o46s`; HOSTS =
`origin/claude/shift-hosts`; HV2 = `origin/claude/shift-hosts-v2`; LIVE = `scratchpad/status/live_c264.html`; PLACE =
`scratchpad/placement/result.json`. Notes beside this file: `infra_cost_notes.md` (infrastructure and cost),
`seed_registry.json` (seed namespaces), `measure/MEASUREMENT_NOTES.md` (measurement design), `measure/floor_arith.out`
(arithmetic), `bench/*.out` (benchmarks). Paths without a branch are on MAIN unless stated.

---

## 0. In one page

**What is tested.** Three places to use BCA, each tested in full, for return and for coverage, with online ThrusterChain
episodes that simulate deployment:

- **A, harm switch.** Leave the host alone by default. Switch on a plain caution action (clip to the recorded range, or
  a gated range hinge) only when BCA's held-out residual score shows, with 95% credibility, that the critic over-predicts
  full-throttle steps more than the rest by more than a margin fixed in advance.
- **B, safety certificate per deviation.** Fall back to the clip by default. Let a deviation (a declared group of
  commands past the recorded range, or one throttle rung) through only when BCA's tail bound on that deviation's
  held-out tries certifies that harmful tries are rare. Tested offline, and in a staged deployment that gathers tries
  from a budget of online episodes.
- **C, value lower edge.** Train the actor on Q1 minus k times BCA's band width at its own smoothed command, with the
  band recalibrated on Q1, exact state-by-command weights, and "no value credit" where the band refuses. Tested always
  on, and switched by A.

**Where.** Unifloral's TD3+BC (revision f2dc127) as the host, trained on the main study's gapped log in the stalling
world and its no-stall twin, plus a mixed world for A. Pilot on seeds 0-4 (exploratory). Confirmation on fresh seeds
25-29 of the main generator, extended to 30-34 if the power rule asks for it. An optional module W adds the known-shift
weighting-test world, the only place where BCA's importance weights can earn credit.

**What decides.** For each option: a primary return claim, a primary coverage or validity claim, and credit claims that
ask whether BCA's own part (the residual score, eta, the Bayesian step, the weights) beats the matched plain rule. All
marks are in section 7, fixed now. Every claim has its wording fixed for both outcomes.

**What the record predicts (stated now so that surprises show).** A's return claim passes and its credit to the
posterior fails (plain tests at the same margin match it). B certifies in the no-stall world and refuses in the stalling
world only where evidence reaches about 30 tries; with equal weights BCA is the exact binomial test, so B can credit
only the placement, not BCA. C's band keeps its covered-or-refused promise mostly by refusing, and loses to a matched
range hinge and to the switched clip; its gain over refusal alone is near zero.

**Cost.** About 50 Unifloral runs of 100,000 steps per seed: about 19.5 CPU-hours per seed on this kind of processor
(about 98 CPU-hours per 5 seeds, 117 with a 20% allowance for extra matching rounds; about 25-29 hours of wall time on
4 CPUs), or about 1.1-1.3 card-hours per seed on the laptop's graphics card (about 6-8 card-hours per 5 seeds).
Evaluation, replicates and online episodes add about 3 CPU-hours per seed (section 9).

---

## 1. Words used here

Each term is defined here, at its first use in this document.

- **ThrusterChain**: the simulated system of five floats in a channel, pushed by a current and waves and held by four
  thrusters (MAIN `experiments/shift/thruster_chain.py:1-37`). Its dynamics are deterministic: the same start and the
  same commands give the same episode (`thruster_chain.py:13`).
- **Command**: the four thruster settings in [-1, 1]. **Throttle index** of a command: `t(a) = max_k |a_k|`.
- **Knee**: 80% throttle. **Stalling world**: past the knee the thrust falls, `thrust(a) = sign(a)(0.8 - 1.5(|a| - 0.8))`,
  so full throttle is harmful. **No-stall world (twin)**: thrust keeps rising, so full throttle helps. At or below the
  knee the two worlds are identical (`thruster_chain.py:16-18`). **Stall slope sigma**: the 1.5 in that formula;
  `world_physics('stall', stall_slope=sigma)` sets it (`thruster_chain.py:143-149`); sigma = -1 is the no-stall world.
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
  rising-thrust controller in the no-stall world (MAIN `experiments/shift/truth.py:26-28, 339-354`).
- **Data seed** (or **seed**): one repeat of the whole experiment, with its own recorded log and its own trained networks.
  **Seed-world**: one seed in one world.
- **Recorded log**: 1,000 episodes of the recording controller, split by episode into training (0-499), calibration
  (500-699) and evaluation (700-999) (MAIN `experiments/shift/dataset.py:4-15`).
- **Gapped log**: the recorded log after the **ball** (the densest 15% of training commands, which holds almost all
  full-throttle commands) is cut from the training rows and the calibration bank (MAIN `experiments/shift/shift.py:52,
  111-123, 164-192`). **Survivors**: full-throttle training rows left outside the ball (28-59 per seed-world).
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
  posterior **crossings** (each a credible 90% upper quantile, with one extra atom for the unseen test point,
  `wbcp.py:43-54`) and returns a **threshold** R, the 95%-credible upper quantile; R = +inf (a **refusal**) means
  "cannot certify". Settings everywhere unless stated: miscoverage 0.1, credibility 0.95, 1,000 draws.
- **Posterior gap** and **L**: for the harm switch, the 5% quantile of all 1,000 x 1,000 differences between the
  full-throttle group's crossings and the rest's (step 1 PRESPEC.md:146-154; `step1lib.py:226-276`).
- **Margin m**: the size the gap must clear. Its rule, a decision break-even (section 3.6), is fixed now; its value is
  computed by that rule on the exploratory pilot and frozen before any confirmatory datum is read.
- **Harm switch (A)**: on iff `L > m`; when on, the caution action runs in that world; when off, the host runs untouched.
- **Fallback F**: clip, as above. **Deviation**: letting a command beyond what the fallback allows through. **Try**: a
  held-out row (offline) or an online step where that deviation was used, with its residual score. **Rung**: a 0.05-wide
  throttle band above the recorded range, for one thruster and one sign.
- **Harm event of a try**: `s > tau`, with `tau = lambda_R + m`, `lambda_R` the rest's empirical 90% quantile.
- **Tail gate**: certify a deviation iff, with 95% credibility (BCA) or confidence (plain), at most `alpha_B = 0.1` of
  its tries are harm events. With equal weights BCA's tail gate is the exact binomial (Wilks / Clopper-Pearson) test
  (LIVE 10502-10517; PLACE rank, O8). **Evidence floor**: the fewest tries with which a gate can certify at all: 29 for
  the exact binomial test at alpha 0.1 and 95%; BCA's own code certifies nothing below 36 (HOSTS PRESPEC.md:266-268).
- **Safety certificate (B)**: allow a deviation iff its tail gate certifies; otherwise the fallback runs.
- **Band**: BCA's interval `Q1 +- U` meant to hold the critic's one-step error `|y - Q1|` 90% of the time, `y` the host's
  own learning target. **Width** `U = R x u x max(eta, 1e-6)`: `eta` a small scale network, `u` a running unit, `R` the
  WBCP threshold (HOSTS PRESPEC.md:256-268). **eta**: the scale network.
- **Declared query law**: where the band's promise is made: states drawn as the recorded log draws them, and the command
  `pi(x) + xi` with `xi = clip(N(0, 0.2^2), +-0.5)` per thruster, then clipped to [-1, 1].
- **Exact state x command weights**: the density ratio of the declared query law to the law of the kept bank rows,
  `w(x, a) = q_xi(a | x) / beta_kept(a | x)` (the state factor is 1, because the declared states follow the log's own
  state law). `w = +inf` (refusal) inside the ball or wherever the recording controller never commands `a`.
- **Value lower edge (C)**: the actor maximizes `Q1 - k W`, with `W` built from the band width at its own smoothed
  command (section 4.3). **No value credit**: where the band refuses, the actor is credited no more than Q1 of the
  row's recorded command.
- **Covered-or-refused**: a draw is covered if `|y - Q1| <= U`, and counts as covered when refused; this is what the
  band promises. **Covered among certified**: the covered share of the draws that were not refused. **Refused share**.
- **Matched**: a control tuned, from strengths alone and before any return is computed, to move the actor as far as the
  primary arm does. **Strength**: the root-mean-square change of the actor's commands from the host's on 5,000 held-out
  recorded states (step 2 PRESPEC.md:47-50).
- **Plain rule**: any rule without BCA's posterior: split conformal, difference of means, rank test, exact binomial,
  Clopper-Pearson, Hoeffding, empirical Bernstein, HCOPE-style mean bounds (Thomas, Theocharous and Ghavamzadeh 2015),
  rejection sampling, support or range rules, plain caps.
- **NET** of a deployed policy p: in strong current, the mean over seeds of
  `[P_stall(p) - P_stall(host)] - [P_nostall(host) - P_nostall(p)]` (step 2 PRESPEC.md:207-214). A perfect world switch
  scored 106.1 in the record; clipping always 44.4 (ROUND1.md:13).
- **No-stall cost**: `P_nostall(host) - P_nostall(p)`.
- **Replicate**: a fresh draw of the evidence from fresh generator streams (same law, same episode structure, same
  frozen critic; its size varies as the actual evidence's would; section 6.4); used to measure error and coverage rates
  of certificates, which are probabilities over the draw of the evidence.
- **Design effect**: how much clustering of steps within episodes inflates the variance of a step-level rate,
  `1 + (m - 1) rho` for m steps per episode with within-episode correlation rho.
- **L (statistics)**: a one-sided 95% lower bound over seeds, `mean - t(0.95, S - 1) x sd / sqrt(S)` (HOSTS
  PRESPEC.md:540-542). **Holm's correction**: the step-down adjustment for several p-values (HOSTS PRESPEC.md:543-548).
- **Stop rule**: an integrity check that stops a phase; a stopped phase is never rerun or replaced. **Kill rule**: a
  mark, fixed in advance, that shelves a claim or an option.

---

## 2. Questions and claims (written before any data)

Every claim is judged on the confirmatory seeds; on the pilot the same rules are computed and labeled exploratory.
Every claim needs every seed: a missing, stopped or non-finite value fails every claim that needs it.

### 2.1 Option A: harm switch on a declared group or world

**Question.** When BCA's held-out residual score on a declared group (full-throttle commands past `t_lo`) is read against
the rest, and a plain caution action switches on only when the posterior gap's 5% lower bound clears the margin, does
the deployed system earn the return of a perfect world switch, keep its promise about the gap and about false switches,
and owe any of that to BCA's own parts rather than to plain tests at the same margin?

- **A-R1 (primary, return).** In online episodes, the switched clip beats clipping always (the best world-blind rule)
  and random switching at its own rate.
- **A-V1 (primary, validity: the error A controls).** In seed-worlds where the true gap is at most the margin, the
  switch turns on in at most 5% of replicate evidence sets.
- **A-V2 (primary, coverage).** The posterior gap's lower bound L lies below the true gap in at least 95% of replicate
  evidence sets.
- **A-K1 (credit: the Bayesian step).** BCA's posterior gap beats every plain test applied to the same scores at the same
  margin.
- **A-K2 (credit: outcome information).** With outcomes shuffled within each seed-world, the switch keeps at most half
  its NET.
- **A-K3 (credit: held-out reading).** Reading held-out rows beats reading the in-sample survivors.
- **A-K4 (credit: weights that localize; mixed world, secondary).** A posterior weighted by closeness in wave size
  beats the world-level switch, the best plain cap and a weighted difference of means at the same margin.

### 2.2 Option B: safety certificate per deviation

**Question.** When the fallback (clip) is the default and a deviation is let through only when BCA's tail bound on its
held-out tries certifies that harmful tries are rare, does the deployed system keep the fallback's safety where
deviating harms, release it where deviating helps, keep its promise in deployment, and owe anything to BCA rather than
to a textbook test?

- **B-R1 (primary, return and safety; offline evidence, declared group).** The certified map is no worse than the
  fallback in the stalling world and beats clipping always across worlds.
- **B-R2 (primary, return and safety; staged deployment, declared group).** The same, with safety judged on the
  episodes after the decision and release on the cumulative return of a 1,000-episode deployment that includes its
  probe episodes.
- **B-V1 (primary, validity: the promise).** Certified decisions whose true harm rate exceeds `alpha_B` occur in at most
  5% of replicate try sets.
- **B-V2 (primary, validity: transfer to deployment).** Among deployed steps where the certified map departs from the
  fallback, the share whose exact effect on return is negative is at most `alpha_B`.
- **B-K1 (credit: outcome information).** The residual score carries the harm: in the stalling world, shuffling the
  outcomes raises the certification rate by at least 0.3. Arm B's critic-scored gain (the negative control, which
  certified even with shuffled outcomes before) is reported against the same mark.
- **B-K2 and B-K3 (credit: the weights; module W only).** With exact (B-K2) or estimated (B-K3) importance weights,
  BCA's weighted tail gate certifies more truly safe rungs than rejection sampling, weighted split conformal and a
  self-normalized weighted mean bound, at equal harm.

Declared now: with equal weights BCA's tail gate is the exact binomial test, so on the main generator (where tries are
either in the recorded law or come from the deployment law itself) no credit to BCA is possible; only the placement can
be credited (PLACE `red.challenges[11]` and `[13]`: 28,000 of 28,000 equal-weight decisions agreed with the binomial
test; `measure/MEASUREMENT_NOTES.md`, section B).

### 2.3 Option C: value lower edge at the actor's own command

**Question.** When the actor maximizes Q1 minus k times BCA's band width at its own smoothed command, with the band
recalibrated on Q1, exact weights and no value credit where it refuses, does the deployed actor recover return under
shift at a low no-stall cost, does the band keep its coverage promise, and is any gain owed to the band's finite width,
eta, the Bayesian step or the weights rather than to refusal, support or a plain hinge?

- **C-R0 (kill: room).** The exact-error value penalty (oracle) beats the host by at least 20 points in the stalling
  world, strong current. Otherwise C's return claims are not judged.
- **C-R1 (primary, return, always on).** C beats the range hinge matched to its strength on NET, and keeps the host's
  improvement in the other situations.
- **C-R2 (return, switched by A).** C switched by A is no worse than A's switched clip and the switched hinge (non-
  inferiority).
- **C-V1 (primary, coverage: the promise).** Covered-or-refused is at least 90% at fresh draws from the declared law.
- **C-V2 (primary, coverage: usefulness).** The refused share is at most 50% and covered-among-certified at least 85%
  at the same draws.
- **C-K1 (credit: finite width).** C beats F0, the same refusal with no band.
- **C-K2 (credit: eta).** C beats ratio-only, the same band with eta = 1.
- **C-K3 (credit: the Bayesian step).** C beats the same band with a weighted split-conformal threshold in place of WBCP.
- **C-K4 (credit: beyond support).** C beats the matched support-only penalty (the exact log density ratio as the shape).
- **C-K5 (credit: the weights; module W only).** Weighted C beats unweighted C under the known shift.

---

## 3. Host, data, worlds and seeds

### 3.1 Host

- **Default host: Unifloral's TD3+BC**, imported unchanged after its SHA-256 is checked against `configs/unifloral.json`
  (revision `f2dc1278eae18ed3e22c92255119c54885414d17`; `td3_bc.py` SHA-256 `6f5ab6f9...`), through
  `experiments/shift_hosts/unifloral_import.py` (HOSTS: `prepare`, `args_for`, `make_transition`,
  `init_like_training_block`, `make_unifloral_step`, lines 215-285, 353-461). Unifloral's own defaults (alpha 2.5,
  batch 256, learning rate 3e-4, target noise 0.2 clipped at 0.5, two critic updates per outer step) with ThrusterChain's
  exceptions of the hosts study: discount 0.98, 100,000 outer steps in 40 chunks of 2,500, the data as Unifloral's own
  `Transition`, done 0 on every row (HOSTS PRESPEC.md:138-159). Nothing is tuned.
- **What A and B read.** A and B are wrappers around the trained host. They read only its critic (forward passes) and
  wrap its actor at deployment; they never train anything. Their score uses the online critics and the online actor
  without noise, as step 1 did with its controller (step 1 PRESPEC.md:133-144). Reported beside it, for information:
  the score with Q1 alone, and the score with Unifloral's own noisy target (target networks, 32 fixed noise draws,
  `unifloral_td3_bc_bca.py:316-340`).
- **What C changes.** C restates Unifloral's `make_train_step` line for line, as the hosts did
  (HOSTS `unifloral_td3_bc_bca.py:11-18`), and changes only the actor loss line (`td3_bc.py:245`) to add the lower edge
  (section 4.3). There is no dose: the cloning term keeps multiplier 1 (the owner rejected the dose). BCA's per-step
  fit and its 18 refreshes follow the hosts' schedule (after outer steps 10,000, 15,000, ..., 95,000; HOSTS
  PRESPEC.md:264-268), with Q1 in place of `min(Q1, Q2)` in the band's value (`band_parts`,
  `unifloral_td3_bc_bca.py:348`) and the exact weights recomputed for the current actor at each refresh (as
  CONN `outputs/shift_iw/explore/channel/command_le_train.py:12` did).
- **Clone** (for the clone fallback, reported only): Unifloral's `bc.py` (pinned in `configs/unifloral.json`), loaded
  with the same fingerprint check by a new loader (`unifloral_import.LEARNERS` lists only four learners,
  `unifloral_import.py:58`).
- **Two environments.** Training in `/opt/bca-venv` (Python 3.10.12, numpy 1.26.4, JAX 0.6.2); data and scoring in
  `/opt/bca-shift-venv` (numpy 2.2.6). No process holds both a Unifloral network and the simulator; files cross by
  `handoff.py` with SHA-256 sidecars (HOSTS PRESPEC.md:191-197).
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
- **Evidence for A and B (held-out rows).** Rows of the calibration and evaluation splits outside the ball and outside
  the bank. No critic trains on them. Full-throttle group: those rows past `t_lo`, plus the 0-2 bank rows past `t_lo`.
  Rest: the bank rows at or below `t_lo` (about 825-850), BCA's own calibration bank, as step 1 and the round-1 route
  (`routes/round1/scratch/heldout_calibration_rows/run_route.py:6-8, 102-109`).
  - On seeds 0-4 the calibration split alone held 14-28 full-throttle rows per seed-world (15-30 with the bank's;
    MAIN `outputs/shift/beyond-the-data/restructure/routes/round1/scratch/_feasibility/count_rows_0.json` and
    `count_rows_1_2_3_4.json`). The evaluation split has 1.5 times as many episodes, so about 35-70 in all are
    expected [inferred; counted from commands alone, before any outcome is read, at the pilot's first step].
  - The calibration split alone (the round-1 route's evidence) is reported beside it for continuity.
- **Survivors** (A-K3 only): the training rows past `t_lo`, scored by the host that trained on them.
- **Unifloral preparation**: a new `prepare` for the main gapped log in both worlds (and the mixed worlds), writing
  Unifloral `Transition` arrays, the bank, the evidence rows, their exact next states and the next commands' inputs,
  with SHA-256 sidecars (the hosts' `prepare.py` covers only the weighting-test data, HOSTS `prepare.py:1-41`).
- **Leaks checked by code** (stop rule S9): no evidence or bank row is a training row; no evidence row lies in the ball;
  the gap rule reads commands only.

### 3.3 Worlds

| World | Physics | Log | Hosts trained | Used by |
|---|---|---|---|---|
| S, stalling | `world_physics('stall')`, sigma 1.5 | main gapped log | yes | A, B, C (primary) |
| N, no-stall twin | `world_physics('no_stall')` | main gapped log, same streams | yes | A, B, C (primary) |
| MW+, mixed by wave size | stall iff the start's wave amplitude > 0.25 | spliced log (below) | yes | A (secondary) |
| MW-, mirror | stall iff amplitude <= 0.25 | spliced log | yes | A (secondary) |
| Ladder | sigma in {-1, -0.5, -0.25, 0, 0.1, 0.2, 0.3, 0.5, 0.75, 1.0, 1.5, 3.0} | N's log, outcomes recomputed | no | margin rule, A-V (secondary) |
| M3, harm only beyond the log | N's host deployed in stalling physics | none | no | A, B information |
| M4, harm only inside the log | S's host deployed in no-stall physics | none | no | A, B information |

- **Why the mixed log is exact.** Wave amplitude never changes within an episode (`thruster_chain.py:11, 170-171`) and
  both worlds' logs share every random number, so episode e of MW+ is episode e of the stalling log when its start's
  amplitude exceeds 0.25 and of the no-stall log otherwise. Its gap, ball and bank are recomputed on the spliced log by
  the same functions. An online episode in a mixed world uses its start's physics. (The round-1 route spliced evidence
  rows in this way, `ROUND1.md:121-146`; this test also trains hosts on the spliced logs, which the route deferred.)
- **The ladder** is decision-level, as round 1's was (`ROUND1.md:66-96`): N's host and N's held-out rows are fixed; the
  one-step outcomes of the evidence rows are recomputed exactly under each sigma, and the value of caution is measured by
  online episodes of N's host in sigma-physics. No network is retrained.
- **M3 and M4** need no new training: they deploy existing hosts in the other physics (the shelved run-time monitor's
  worlds, `ROUND1.md:41-64`).

### 3.4 Seeds

- **Pilot: main-generator seeds 0-4, exploratory.** Seen by the gate, stage 2, steps 1 and 2 and round 1. Hosts are new
  (Unifloral), so nothing there is reused except the deterministic data, which are regenerated and checked against
  their records (copies in MAIN `outputs/shift/beyond-the-data/stage2_gpu/payload/files/runs__shift__pre__data__base__
  {stall,no_stall}__seed<k>__record.json`).
- **Confirmatory: main-generator seeds 25-29**, under a pre-registration you approve, with **30-34** declared now as the
  extension if the power rule of section 8.4 asks for 10 seeds.
- **Why 25-34 are free.** In the main generator, 0-4 are seen; 5-9 belong to version 6/7 (refused by
  CONN `near80/code/near80lib.py:70`); 10-14 are near-80 version 8's confirmatory seeds, now running
  (`near80lib.py:69`); 15-19 are near-80's declared fallback for step 2's confirmation; 100-119 are the original gate's
  confirmation seeds; 900-909 are stage 0's design seeds. Seeds 20-99 and 120-899 were never generated or named; a grep
  of all four branches finds no use of 20-34 in this namespace (seed_registry.json). 20-24 stay free as an optional fresh
  margin-calibration block (decision D3).
- **Module W (optional, section 10)**: weighting-test generator (master 20261016) seeds 0-4 as pilot, 25-29 as
  confirmation (free: 10-14 are the four-learner study's, 15-19 arms A/B/C/F0's, 20-24 named and rejected;
  seed_registry.json). The same seed number gives different data in the two generators (HOSTS PRESPEC.md:613-615); every
  seed list in `prespec.json` names its generator and master seed.
- **Every seed counts.** A seed whose run stops is reported and never replaced.

### 3.5 Random streams (new block, registered now)

The test's own randomness uses master seed **20261019** (unused on all four branches; 20261018 is used by HV2's
`probe_s7.py:300`). Tags:

| Stream | Rule |
|---|---|
| Unifloral `Args.seed` per seed k (every arm of a seed-world shares it, so arms share initial parameters and batches) | `default_rng([20261019, k, 31]).integers(2**31 - 1)` |
| BCA fit and refresh keys (C) | as the hosts' rule (HOSTS PRESPEC.md:216-220) with `[20261019, k, 91]` and `[20261019, k, 39]` |
| WBCP posterior draws (A, B): replicate r (0 = the actual evidence), group g (1 group, 2 rest) | `[20261019, 1, k, w, r, g]` |
| Outcome shuffles, permutation j = 1..200 | `[20261019, 2, k, w, j]` |
| Random-switching coins, situation index s | `[20261019, 3, k, w, s]` |
| Fresh probe-episode populations (truth for staged B) | `[20261019, 4, k, w, d]` |
| Replicate held-out blocks (A, offline B; their union is the truth population) and C's replicate banks | `[20261019, 5, k, w, r]`, split as the generator splits: starts and noise, then coins |
| Staged-B deployment stream, decision d | `[20261019, 6, k, w, d]` |
| C's declared-law draws (states, xi, target noise) | `[20261019, 7, k, w]` |
| Episode bootstrap | `[20261019, 8, k]` |
| Ladder, stall slope index | `[20261019, 9, k, i]` |
| Tuning bank of B's oracle ceiling map | `[20261019, 10, k, w]` |

**Scoring banks.** Pilot: the standard banks (comparable with every earlier number). Confirmatory: a fresh bank per seed,
`draw_starts(1000, default_rng([20261012, k, 9]), START_LAWS[situation])` for k in 25-34 (the standard bank's own rule at
the seed's index; tag 9 is used by the data streams only as the standard bank at k = 2), plus the standard bank as a
comparability readout. Points on a fresh bank use anchors recomputed on that bank with `truth.anchors` (no learned
policy). Any try-gathering stream is disjoint from every scoring bank.

### 3.6 The margin rule (fixed now; its value is computed on the pilot and frozen before confirmation)

The 0.10 margin of the record was set after seeing seeds 0-2, for a different critic (step 1 PRESPEC.md:156-168,
366-368); it carried the false-alarm control (without it BCA netted 55.7, `ROUND1.md:34`). With a new host its scale
has no meaning. So the margin is set by a written break-even rule:

1. On each pilot seed k, take N's host and its held-out rows.
2. For each ladder sigma: the value of caution `V_k(sigma)` = mean over the strong-current bank of `J(clip wrapper) -
   J(host)` in sigma-physics (return units, online episodes); and the population gap `Gamma_k(sigma)` = 90% quantile of
   the scores of fresh full-throttle rows minus that of fresh rest rows (at least 10,000 and 100,000 fresh rows, drawn
   exactly like the evidence; one-step outcomes recomputed exactly under sigma; N's critic).
3. `sigma*_k`: the lowest sigma where `V_k` changes sign from at most 0 to above 0, by linear interpolation between ladder
   points. `m_k = Gamma_k(sigma*_k)`, interpolated the same way.
4. **m = median of m_k over the pilot seeds.** If fewer than 3 of 5 seeds have a crossing, the rule fails (kill K4).
5. **Online margin m_on** (for staged B and the information-only monitor): the same steps with the population read at
   the deployment's own visited steps (N's host in sigma-physics, strong current, steps past `t_lo` against steps at or
   below it). The run-time monitor failed because its margin was set in recorded conditions (`ROUND1.md:60-63`).
6. The populations of step 2 are the union of the replicate blocks of section 6.4 (offline) and fresh probe episodes
   (online), with one-step outcomes recomputed exactly under each sigma.

Reported beside it for every A and B rule: margins 0, m/2, m, 2m and the absolute 0.10.

---

## 4. Arms and controls

### 4.1 Option A arms

**Decision rules**, all on the same held-out scores, groups and margin m, one decision per seed-world (world-level):

| Arm | On iff |
|---|---|
| **A-BCA (primary)** | `L > m`: BCA's two-posterior gap, uniform weights (`step1lib.certificate`) |
| A-BCA0 | the same at margin 0 (the margin's share) |
| A-BB | Bayesian bootstrap of the two 90% quantiles without WBCP's test atom, 5% lower bound of the gap > m |
| A-SC | plain split conformal: empirical 90% quantile of the group minus that of the rest > m (step 2's `conformal_k1` switch) |
| A-OS | exact order-statistic bound: the distribution-free 97.5% lower bound of the group's 90% quantile minus the 97.5% upper bound of the rest's > m |
| A-BOOT | percentile bootstrap (2,000 resamples) 5% lower bound of the quantile gap > m |
| A-MEAN | Welch 95% lower bound of the mean gap > m; and the plain difference of means > m (the round-1 verifier's rule) |
| A-HL | Hodges-Lehmann 95% lower bound of the shift > m |
| A-RANK | one-sided Mann-Whitney p < 0.05, no margin (step 1's rank test, `step1lib.py:279-287`) |
| A-EXC | Clopper-Pearson 95% lower bound of the group's share above the rest's 90% quantile plus m > 0.10 |
| A-SHUF | A-BCA with outcomes shuffled within each seed-world, 200 permutations (`step1lib.py:290-305`) |
| A-RAND | random switching at A-BCA's own on-rate (step 1 PRESPEC.md:201-202) |
| A-ALWAYS / A-NEVER | clip always (world-blind) / the host alone |
| A-INS | A-BCA on the in-sample survivors against the bank (A-K3) |
| A-PERFECT | on iff the true value of caution V > 0 (oracle reference) |

**Caution actions.** When on: (i) **clip** (primary): a deployment wrapper, no training; (ii) **gated hinge**: the host
retrained with stage 1's range hinge `100 sum_j([a_j - hi_j]_+^2 + [lo_j - a_j]_+^2)` at k = 1 in the actor loss
(`harness.py:373-376`), deployed in place of the host when on; (iii) **clone fallback** (reported): the clone's command
replaces any command with `t(a) > t_lo`.

**Mixed worlds (A-K4).** Declared groups by state, fixed now and not aligned with the stall cut: wave-amplitude cells
[0.1, 0.2), [0.2, 0.3), [0.3, 0.4] (the middle cell straddles 0.25). Arms: A-CELL (A-BCA per cell; an episode's cell
is its start's amplitude; starts above 0.4 lie in no cell and run the host); A-KERN (A-BCA with weights from a
Gaussian kernel in amplitude, Silverman bandwidth, test mass 1, as `mixed_waves/run_route.py:78-92`); world-level A-BCA;
plain caps at 0.75, 0.80, 0.85, 0.90 and 0.95 (no outcomes; 0.80 labeled "knows the knee"); kernel-weighted difference of
means at margin m; per-cell A-SC.

**Information only (A-I).** Margin sweep (0, m/2, m, 2m, 0.10) by evidence size (subsamples of 5, 8, 12, 20, 30 rows
and all rows, 400 replicates each, as round 1's small-evidence test); the ladder decisions of every rule; the online
monitor in S, N, M3 and M4 (A-BCA re-read after each of the first 30 strong-current episodes with margin `m_on` and a
latch); the score with Q1 alone and with Unifloral's noisy target.

### 4.2 Option B arms

**Units of decision.**
- **B-GROUP (primary unit)**: one declared deviation group per seed-world: "let the host's own command through
  unclipped" (that is, never apply the fallback). Tries: the full-throttle evidence rows (offline) or first uses
  (online).
- **B-RUNG (per deviation)**: per thruster k and sign, rungs from the recorded range edge: rung 0 = (edge, 0.80], rungs
  1-4 = (0.80, 0.85], (0.85, 0.90], (0.90, 0.95], (0.95, 1.00] (float32 edges). A try of a rung: a row whose command for
  that thruster and sign lies in the rung and whose host command at the try's own state lies beyond the rung's lower
  edge (arm B's definition, CONN `arms_abc_f0/code/arm_b.py:8-12`). Walk rung 0, 1, ... in fixed order; stop at the
  first refusal; the map's ceiling for that thruster and sign is the top of the last certified rung (`arm_b.py:131-139`).
  The deployed map clips each thruster to [lo', hi'] with the certified ceilings in place of the recorded edges, when
  `t(a) > t_lo`.
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
- Shuffled s (tries and rest pooled and re-dealt, 200 permutations): the null.
- Online only, simulator reference, not deployable: the **twin outcome** of a probe episode, `J(map with deviation) -
  J(map without)` from the same start, harm iff negative.

**Gates**, each with `alpha_B = 0.1` and level 0.95 unless stated, on the same tries and the same `tau`:

| Gate | Certify iff |
|---|---|
| **BCA tail gate (primary)** | `wbcp.calibrate(s_tries).threshold <= tau` and n >= 36 (BCA's own minimum) |
| BCA tail gate, exact | the same posterior computed exactly (Beta form); must equal the exact binomial test on 100% (stop rule S6) |
| Exact binomial / Clopper-Pearson | Clopper-Pearson 95% upper bound of the harm rate <= alpha_B |
| Split-conformal tail | the finite-sample-corrected empirical 90% quantile <= tau (+inf, refuse, when undefined) |
| Plain empirical rate | harm share <= alpha_B (no correction; an anti-conservative reference) |
| Hoeffding, empirical Bernstein | 95% upper bound of the harm rate <= alpha_B |
| BCA mean gate (Gaffke-type, Bayesian bootstrap with a ceiling atom) | 95%-credible upper bound of the mean of s <= mean of the rest + m (lifted from MAIN `bca_mean_gate_toy.ipynb` cells 7, 9, 12-13) |
| t-test, empirical Bernstein and HCOPE-style mean bounds | 95% upper bound of the mean of s <= mean of the rest + m; HCOPE-style: Thomas et al. (2015) bound with truncation at a value fixed from the bank's score range |
| Arm B's rule (negative control) | `wbcp.calibrate(-g)` at miscoverage 0.5 has threshold < 0, and arm B's plain tests (`arm_b.py:76-128`) |
| Rejection sampling, weighted split conformal, self-normalized weighted mean | module W only |

**Controls**: the fallback always (= A-ALWAYS); the host always (= A-NEVER); the range limit, each thruster capped at
its 99.5% percentile (`arm_b.py:152-155`); the matched global ceiling, one ceiling for all thrusters with the certified
map's clipped share (`arm_b.py:142-149`); the knee clip at 0.80 (labeled "knows the physics"); the oracle ceiling map
(per thruster and sign, the ceiling that maximizes mean J on a separate tuning bank, `[20261019, 10, k, w]`); the
clone fallback (reported).

### 4.3 Option C arms

All co-trained: Unifloral's TD3+BC restated with one change in the actor loss (`td3_bc.py:245`):

`actor_loss = -lambda x mean_i [Q1(x_i, pi(x_i)) - k W_i] + mean_i mean_j (pi(x_i)_j - a_ij)^2`,
`lambda = 2.5 / stopgrad(mean_i |Q1(x_i, pi(x_i))|)` (on Q1 alone, as Unifloral), k = 1, only the actor's parameters
differentiated through W.

- **Where the band is read.** At `a_i^q = pi(x_i) + xi_i`, one fresh `xi_i` per row per step from BCA's own key (never
  Unifloral's). Q1 stays at `pi(x_i)`, as Unifloral's line 238 has it, so that with k = 0 the step is exactly the host's.
  (The placement analysis wrote Q1 at `pi(x) + xi` too; that form is reported as information only, decision D6.)
- **The lower edge, paired form** (MAIN `experiments/shift_iw/lower_edge.py:1-17, 204-208`):
  `W_i = [U_w(x_i, a_i^q) - U_w(x_i, a_i)]_+` where the band certifies the query;
  `W_i = [Q1(x_i, pi(x_i)) - Q1(x_i, a_i)]_+` where it refuses (no value credit, as CONN `command_le_train.py:105-112`).
- **The band**: `U_w(x, a) = R(w(x, a)) x u x max(eta(x, a), 1e-6)`; scores on the bank `|y - Q1| / (u x max(eta,
  1e-6))`, y = Unifloral's own target (`unifloral_td3_bc_bca.py:316-340`); R from WBCP with the bank rows' weights and
  the query's test mass `w(x, a)`, read from a threshold table on a grid of test masses (MAIN
  `experiments/shift_iw/coverage.py:153-208`).
- **Exact weights for the main recorder** (new code): normal driving `0.8 tanh(u / 0.8)` with `u ~ N(mu(x), 0.3^2)`, and in
  the full-throttle region (water push above 0.565) the full-throttle command with probability 0.5,
  `clip(P + 0.03 n, -1, 1)`, whose first thruster has an atom at -1 (MAIN `experiments/shift/recorder.py:8-17`);
  restricted to outside the ball. The query law's atoms at +-1 (from the final clip) are matched component by component,
  as `command_band_check.py` did for the weighting test.

**Arms** (in both S and N unless stated):

| Arm | W |
|---|---|
| **C-BCA (primary)** | as above |
| C-k0 | C-BCA with k = 0: must reproduce the host bit for bit (stop rule S3) |
| C-F0 | no value credit where C-BCA refuses; 0 elsewhere (the same refusal, no band; CONN `arms_abc_f0/code/arm_f0.py`) |
| C-RATIO | eta = 1 (the weighted threshold's scale only) |
| C-SC | weighted split-conformal threshold (Tibshirani et al. 2019) in place of WBCP's; same eta and weights |
| C-SUP | support only: `c x [log w(x, a^q) - log w(x, a_i)]_+`, c matched; refusal as C-BCA |
| C-ORACLE | exact-error value penalty `[delta*(x_i, a_i) - delta*(x_i, a_i^q)]_+` with `delta*(x, a) = r + 0.98 Q1(x', pi(x')) - Q1(x, a)` from one exact float64 step (stage 1's main oracle on Q1; MAIN `lower_edge.py:32-41`) |
| C-HINGE | range hinge (`harness.py:373-376`), coefficient matched |
| C-EVEN | even caution: the cloning term times (1 + nu), nu matched (the plain alpha retune) |
| C-RAND1, C-RAND2 | two fixed random-network penalty surfaces (stage 1's 64-wide, 2-deep), coefficient matched |
| C-BCA x A, C-HINGE x A, CLIP x A | composites, no training: the arm where A-BCA is on, the host where it is off |

**Matching.** Target: C-BCA's strength in S. Each matched control: a 4-rung geometric ladder of coefficients, log-linear
interpolation to the target, one matched run; tolerance 5%; at most 2 more rounds; otherwise recorded as unmatched,
which counts against C (as step 2, PRESPEC.md:178-186). In N each control runs at its coefficient locked in S. All
strengths and fingerprints are locked before any return is computed (step 2 PRESPEC.md:188-194).

**Information only (C-I).** C-I1 eta's share of the variance of `log U_w` against `log R(w)`, and the partial Spearman
correlation of `U_w` with the oracle's error given w (placement test T1's readout, PLACE rank test_plan T1); it fixes
only the wording of C-K2. C-I2 an actor-only variant of C-BCA, C-F0 and C-HINGE against the host's frozen critic (the
protocol of every earlier C result), to separate placement from critic feedback.

### 4.4 Shared controls (summary)

| Control | A | B | C |
|---|---|---|---|
| Host alone | A-NEVER | host always | the host (and C-k0) |
| Plain rivals | A-SC, A-OS, A-BOOT, A-MEAN, A-HL, A-RANK, A-EXC, A-BB | binomial / Clopper-Pearson, split-conformal tail, Hoeffding, empirical Bernstein, t, HCOPE-style | C-SC, C-RATIO, C-F0, C-SUP |
| Shuffled outcomes | A-SHUF | shuffled s | C-I: eta shuffled within state (reported) |
| Matched-strength even caution | A-RAND (placement at its own rate) | matched global ceiling | C-EVEN |
| Support and range rules | A-ALWAYS, plain caps (mixed) | range limit, knee clip | C-HINGE, C-SUP |
| Oracle references | A-PERFECT | oracle ceiling map, twin outcome | C-ORACLE |
| Fallback variants | clip (primary), gated hinge, clone | clip (primary), clone | refusal = no value credit |

---

## 5. Deployment simulation

### 5.1 Offline computation, then deployment

1. **Train** the hosts (S, N, MW+, MW-), the clones (S, N), A's gated hinge (S, N) and C's arms (S, N). Lock every
   network's fingerprint.
2. **Compute offline, read-only, before any return**: the gap rule on commands; the scores of evidence and bank rows; A's
   decisions for every rule; B's offline certificates and maps; C's final band (refresh at the end of training on the
   final critic and actor, as deployment would use it). Lock all decisions in `decisions.json` (write-once).
3. **Deploy**: online episodes of every deployed policy (host, wrappers, composites, trained arms) on every bank and
   world. Decisions never change after step 2 for the offline arms.

### 5.2 Online evaluation episodes

- **Runner**: one episode = 100 steps through `TC.dynamics` with the command rounded to float32 then float64 and clipped,
  exactly as `env.step` (`thruster_chain.py:443-446`). (`truth.rollout` does not round, `truth.py:85`; its value is
  reported beside it for continuity; a difference above 0.01 points on any bank is reported.) Spot check: 10 starts per
  bank also run through gymnasium `env.step`, bit for bit (stop rule S8). Mixed worlds: each episode's physics from its
  start's amplitude.
- **Counts, per seed-world and deployed policy**: 4 situations x 1,000 starts (fresh bank on confirmatory seeds;
  standard bank too); 100 steps each. Every arm runs on the same starts (common random numbers). Worlds: S and N for all
  options; MW+ and MW- for A; M3 and M4 for A and B (information); the ladder for the margin and A's ladder decisions.
- **Per start**: J, points, the same policy capped at 80%, the share of steps past the knee, F1, F2, late peak, and the
  visited states, commands, rewards and next states (for C2 and B-V2).

### 5.3 Staged deployment of B (the online variant)

Per seed-world, situation strong current (recorded conditions and big kicks reported), deployment horizon **H = 1,000
episodes** from the stream `[20261019, 6, k, w, d]`:

1. Start with the fallback F.
2. **Probe** decision d (B-GROUP: the host unclipped; B-RUNG: F with rung d opened). Each probe episode runs the probe
   map; from it take **one try**: the first step where the deviation was used (`t(a) > t_lo` for the group; the rung
   used, for a rung), with its residual s from the episode's own reward and next state. One try per episode keeps tries
   independent. The rest reference `lambda_R,on` is the empirical 90% quantile of the residuals of all steps at or below
   `t_lo` in the same probe episodes; `tau_on = lambda_R,on + m_on`.
3. **Decide** after N probe episodes with the tail gate (primary: one batch decision at N = 100; reported: N = 30; three
   looks at 10/30/100 with Bonferroni level 0.05/3; and an anytime-valid test, the mixture-likelihood e-process
   `E_n = integral over q uniform on [0, alpha_B] of L_n(q) / L_n(alpha_B)`, rejecting the null "harm rate >= alpha_B"
   when `E_n >= 20`). Evidence floors (floor_arith.out sections 1-3): zero-failure minimum 29 (batch), 39 (three
   looks), 43 (anytime); failures allowed at N = 100: 4, 3, 2. At N = 30 BCA cannot certify at all (below its own
   minimum of 36); only the plain exact test can, and only with zero harm events.
4. Certified: deploy the probe map for the rest of the horizon (B-RUNG: then probe the next rung). Refused: return to F.
5. **B-RUNG families**: thruster-sign pairs whose host command exceeds the recorded edge on at least 1% of steps in the
   first 10 fallback episodes (read from the host's raw command, which the wrapper sees); confidence Bonferroni-split
   across those families; fixed-sequence walk within each family.
6. **Score**: cumulative return over all H episodes, probes included, per world, in points (anchors recomputed on the
   deployment stream's own starts); stall steps during probing; decision time; the final map's score on the scoring
   bank.
7. **Twin outcome** (reference only): the same protocol with the twin outcome as the try, to show what a perfect outcome
   would certify. **Realized mean variant** (reported): episodes randomized between F and the probe map, mean gates on
   the return difference (Welch t; HCOPE-style empirical Bernstein with declared truncation); about 580 episodes per arm
   would be needed for a 5-point effect (unpaired per-start SD about 34 points in strong current; LIVE 5104-5112 gives
   a bank standard error of 1.63 return units over 1,000 starts).

---

## 6. Metrics (exact formulas)

### 6.1 Return

- Per start i: `J_i`, points `P_i = 100 (J_i - J_rec) / (J_top - J_rec)` with the world's anchors on that bank.
- Per seed k, world w, situation s, policy p: `P_ksw(p) = mean_i P_i`; paired standard error over starts of
  `P_i(p) - P_i(q)` (HOSTS `evaluate.paired`, `evaluate.py:356-362`).
- `NET_k(p) = [P_k,S(p) - P_k,S(host)] - [P_k,N(host) - P_k,N(p)]`, strong current; `NET = mean_k NET_k`.
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
- Expected NET of a rule over replicates: `E-NET = mean_k sum_w p_on(k, w) V_kw`, with `p_on` the rule's on-rate over
  the replicate evidence sets (section 6.4); realized NET uses the actual evidence. Regret against A-PERFECT; false-switch
  cost `sum over V < 0 of p_on |V|`; missed-harm cost `sum over V > 0 of (1 - p_on) V`.

### 6.4 Validity and coverage against simulator truth

The simulator gives three exact truths: the one-step outcome at any (state, command) (`thruster_chain.py:201-221`); the
value `Q^p` of any deterministic policy, wrappers included, by a 1,000-step float64 rollout (`truth.py:98-108`); and the
exact split of any return difference along the deployed path, `truth.attribution` (`truth.py:29-31, 359-383`), whose
terms are zero wherever two policies agree and sum to the difference within 1e-8.

**Replicates (A and offline B).** Replicate r of a seed-world is a fresh held-out block: 200 calibration and 300
evaluation episodes of the recording controller in that world, from the streams `[20261019, 5, k, w, r]`, cut by the
seed-world's own ball, with its own bank (5 rows per calibration episode), its full-throttle rows and its rest, all
scored by the frozen host critic. A replicate is therefore drawn exactly as the actual evidence was, episode clustering
included (the 14-28 full-throttle rows of a seed-world came from 13-25 episodes, `count_rows_*.json`), and its group
sizes vary as the actual ones would. R = 1,000 blocks per seed-world (about 500,000 episodes); their union is the
population that defines the truth (about 35,000-70,000 full-throttle rows and 850,000 rest rows expected).

**A (per seed-world, over the R replicates):**
- True gap `Delta_ev = q_0.9(pool_G) - q_0.9(pool_R)` over the union (inverted-cdf quantiles), with its Monte Carlo
  standard error.
  Also the mean gap, and the n-matched functional `q_{0.9 + 1/(n_G + 1)}(pool_G) - q_0.9(pool_R)` (how much of L the
  test atom moves).
- **A-V1 false-switch rate**: `FSR_kw = (1/R) sum_r 1[L_r > m]` for seed-worlds with `Delta_ev <= m`.
- **A-V2 lower-bound coverage**: `COV_kw = (1/R) sum_r 1[L_r <= Delta_ev]`.
- Power: `(1/R) sum_r 1[L_r > m]` where `Delta_ev > m`. Size under an exact null: pools G and R merged and re-dealt
  (true gap 0), at margins 0 and m.
- Transfer (not promised): the same gap at the host's own visited steps in strong current (steps past `t_lo` against the
  rest, each with its episode's own outcome).
- Every rate with its Clopper-Pearson 95% interval. Every rule of section 4.1 on the same replicates.

**B:**
- **B-V1**: per decision d, the true harm rate `pi_d = P(s > tau)` under the try law (offline: over the union of the
  replicate blocks' full-throttle rows; staged: first uses of at least 10,000 fresh probe episodes, streams
  `[20261019, 4, k, w, d]`). For each decision with `pi_d > alpha_B`, the error rate is its certification rate over the
  R = 1,000 replicates (offline: the blocks above; staged: fresh sets of N probe episodes). Offline tries are clustered
  in episodes, which the exact binomial test ignores; the block replicates measure what that costs.
- **B-V2**: run the deployed map p_B on the scoring bank; `term_t` from `truth.attribution(p_B, F)`;
  `HR = #{t: p_B(x_t) != F(x_t), term_t < 0} / #{t: p_B(x_t) != F(x_t)}` per seed-world; per rung by the
  single-thruster counterfactual (only that thruster deviates). Harm mass `sum min(term_t, 0)`. If the map never departs
  from F, HR is reported as "no uses".
- B-V3 (reported): per start `D_i = J_i(p_B) - J_i(F)`; `P(D_i < 0)`, its 10% tail mean, stall steps, F1, F2.
- Refusals: share refused, split into "too few tries" (n below the floor) and "too many harm events", priced against
  the oracle ceiling map: `P(oracle map) - P(p_B)`.

**C (per seed and world; refusal counts as covered for the promise):**
- **C-V1 promise**: M = 50,000 fresh draws from the declared law; for each, `y = r + 0.98 min_k Qbar_k(x', clip(pibar(x')
  + eps))` with r and x' from one exact simulator step at the drawn command and one fresh target-noise draw (the host's
  own target rule); `COR = (1/M) sum_j 1[refused_j or |y_j - Q1(x_j, a_j)| <= U_w(x_j, a_j)]`. Three band states:
  (i) the last training refresh, (ii) a final refresh on the final networks (primary), (iii) the refresh-time actor's
  own law. (i) against (ii) measures staleness.
- **C-V2 usefulness**: `RS` = refused share; `CAC` = covered among certified; each overall, past 80% and within 80%,
  inside and outside the ball.
- C-V1 bank-conditional check (reported): 200 replicate banks; share with coverage below 0.90, Clopper-Pearson bound.
- Information: C2 on-policy coverage at every visited (state, executed command) of the 1,000 strong-current episodes,
  and at the same states with the smoothed command; C3 value error `P(|Q1 - Q^pi| <= U)` and uncaught over-estimation
  `P(Q1 - k U > Q^pi)` at 2,000 visited pairs per bank (`Q^pi` by `truth.q_value_jax`); C4 the share of hard-water
  states whose best throttle for `Q1 - W` lies past 80% (`lower_edge.py:49-58`).

### 6.5 Uncertainty and dependence

- Starts are independent given a frozen policy; all arms share starts, so differences are paired by start.
- Steps are clustered in episodes: every step-level rate (C2, B-V2, transfer gaps) gets an episode bootstrap (2,000
  resamples of episodes, `[20261019, 8, k]`) within each seed, and the measured design effect is reported. Precision at
  90% coverage with 1,000 episodes of 100 steps: +-0.19 to +-1.45 points for within-episode correlation 0 to 0.6
  (floor_arith.out section 6).
- Seeds are the unit of replication for every claim (section 8).
- Replicate-based rates: Clopper-Pearson; R = 1,000 shows a true 5% as at most about 6.3% (floor_arith.out section 7).

---

## 7. Marks, decision rules, kill rules and what each outcome would mean (fixed now)

All bars are in points (strong current) unless stated; "L > 0" is the one-sided 95% bound over seeds of section 8.1.

### 7.1 Kill and stop rules (read first, in this order)

**Stop rules** (each stops its phase; a stopped phase is never rerun or replaced; every claim that needs it fails):
- S1 fingerprints: Unifloral files against `configs/unifloral.json`; `prespec.json` and the code manifest against the
  approval.
- S2 seeds: the code refuses main seeds 5-19, 100-119 and 900-909, weighting seeds 10-24 and 1000 and above, and any
  confirmatory seed before approval.
- S3 reproduction: C-k0 must equal the host bit for bit (restated step with k = 0; tolerance 1e-5 per parameter at
  most); the first host of the pilot is trained twice and must match bit for bit.
- S4 float32 check (the hosts' S7): open loop within 1e-7; closed loop within 0.05 points per start (declared now, before
  any data, for the reason HV2 PRESPEC.md:11-14 and 888-915 give).
- S5 non-finite: any non-finite network, score, threshold or return.
- S6 equal-weight identity: BCA's exact tail gate equals the exact binomial test on 100% of decisions.
- S7 attribution identity: `|sum_t term_t - (J_new - J_ref)| <= 1e-8`.
- S8 runner: bit equality with gymnasium `env.step` on the spot-check starts.
- S9 leaks: as section 3.2.
- S10 budget caps (section 9).

**Kill rules** (judged on the pooled seeds of the stage):
- **K1 host learned**: the host's recorded-conditions points are at least the clone's + 10 in S and in N. Otherwise "did
  not learn": no claim is judged on that host.
- **K2 room for A and B**: `V_S >= +20` and `V_N <= -10` (strong current, pooled). Otherwise A's and B's return claims
  read "no room" and are not judged.
- **K3 room for C (= C-R0)**: `P_S(C-ORACLE) - P_S(host) >= +20`, strong current. Otherwise C's return claims are not
  judged.
- **K4 margin**: the margin rule finds a crossing in at least 3 of 5 pilot seeds. Otherwise no confirmatory stage for A
  and B without a new version.

### 7.2 Option A marks

| Claim | Pass iff |
|---|---|
| **A-R1** | `NET(A-BCA, clip) - NET(A-ALWAYS)`: mean >= +20 and L > 0; and `NET(A-BCA) - NET(A-RAND)`: mean >= +10 and L > 0; and in S, recorded / big waves / big kicks each >= host - 3 (pooled) |
| **A-V1** | in every seed-world with `Delta_ev <= m`, `FSR` has Clopper-Pearson upper bound <= 0.073 (a true 5% passes; R = 1,000); if no seed-world has `Delta_ev <= m`, A-V1 reads "not judged" and A cannot read "works"; the size under the exact null (section 6.4) is then the reported check |
| **A-V2** | pooled `COV` has Clopper-Pearson lower bound >= 0.927, and no seed-world below 0.90 |
| A-K1 | `E-NET(A-BCA) - max over A-SC, A-OS, A-BOOT, A-MEAN, A-HL, A-RANK, A-EXC, A-BB of E-NET`: mean >= +5 and L > 0 at the primary evidence; and BCA leads at 4 of 5 margins (0, m/2, m, 2m, 0.10) |
| A-K2 | `NET(A-BCA) - mean shuffled NET >= 0.5 NET(A-BCA)` (step 1's C3) |
| A-K3 | `E-NET(A-BCA, held-out) - E-NET(A-INS)`: mean >= +5 and L > 0 |
| A-K4 (MW+, MW-) | `NET_mixed(A-KERN) - max(NET_mixed(world-level A-BCA), NET_mixed(kernel difference of means), best cap among 0.75 / 0.85 / 0.90 / 0.95 chosen on the pilot)`: mean >= +5 and L > 0; the 0.80 cap reported, labeled |

Reported, decides nothing: the gated hinge and the clone fallback in place of the clip; decision accuracy table (10 or
20 seed-worlds); the margin and evidence-size sweep; the ladder; M3 and M4 with the online monitor.

**Rationale.** The +20 against clipping always is a third of the cost of one no-stall false switch (about 61.8 points,
`fallback.md:172-174`); one false switch in five seeds still passes, one missed stall (about -106 in that seed) fails.
A-RAND at the switch's own rate is the placement control of step 1 (PRESPEC.md:201-202). +5 for credit is round 1's bar
(`ROUND1.md:107`); expected NET over replicates removes the luck of one evidence draw, as round 1's small-evidence test
did (`ROUND1.md:101-107`). A-V1 is the error the harm switch controls (false alarms); A-V2 is its coverage promise,
which rests only on stretched theory for the two-posterior gap (PLACE rank, O8), so it must be measured, not assumed.

**What the outcomes mean (wording fixed for both outcomes).**

| A-R1 | A-V1 | A-V2 | A-K1 | Meaning |
|---|---|---|---|---|
| pass | pass | pass | pass | The harm switch works on a Unifloral host, keeps its promise, and BCA's posterior earns credit beyond plain tests at the same margin. |
| pass | pass | pass | fail | The placement works; the credit goes to reading held-out outcomes with a margin fixed by break-even; BCA's posterior adds nothing measurable (expected). |
| pass | pass | fail | any | It works and controls false switches, but L is not a valid 95% bound on the gap: the margin, not the posterior, carries the error control. |
| pass | fail | any | any | Return earned on these seeds, but false switches exceed 5% at the margin: not reliable; the return rests on the luck of the evidence. |
| fail | any | any | any | With K2 passed: the harm switch does not carry over to a Unifloral host on fresh seeds; shelve it. |

A-K2 failing means the switch reads support, not outcomes. A-K3 failing means held-out rows are not needed (in-sample
survivors suffice, as round 1 saw, `ROUND1.md:39`). A-K4 passing credits localization by weights in the mixed world;
failing credits the action (a plain cap) or the world-level switch.

### 7.3 Option B marks

`alpha_B = 0.1`, level 0.95, the score s, `tau = lambda_R + m` offline and `tau_on = lambda_R,on + m_on` online.

| Claim | Pass iff |
|---|---|
| **B-R1** (B-GROUP, offline, BCA tail gate) | safety: in S, for every situation, `P(p_B) - P(F)` has L >= -3; and release: `NET(p_B) - NET(F always)`: mean >= +10 and L > 0 |
| **B-R2** (B-GROUP, staged, N = 100 batch) | safety: in S, the episodes after the decision have L >= -3 against F (the Seldonian promise is about the policy returned, not about probing); release: on the cumulative points of the whole H = 1,000-episode deployment, probes included, `NET - NET(F always)`: mean >= +10 and L > 0. The probing cost in S (F always minus the cumulative points) is reported beside it |
| **B-V1** | for every decision whose true harm rate exceeds alpha_B, its certification rate over the replicates has Clopper-Pearson upper bound <= 0.073 |
| **B-V2** | pooled over seeds, per world: HR's episode-bootstrap 95% upper bound <= alpha_B + 0.02; "no uses" passes |
| B-K1 | in S, the certification rate of the primary cell with shuffled s (200 permutations) minus that with real s (over the 1,000 replicate try sets) is at least 0.3, pooled over seeds; arm B's gain g is reported against the same mark |
| B-K2, B-K3 | module W only (section 10) |

Reported: B-RUNG offline and staged; all gates; looks and e-process; twin outcome; realized mean variant; clone fallback;
refusal prices; the confusion table of A's and B's decisions on the same evidence.

**Rationale.** B's default is the fallback, so its safety mark is non-inferiority to F in the world where deviation
harms (-3, step 2's G2 tolerance, PRESPEC.md:223-224), and its return mark is the release of F where deviation helps
(+10 over clipping always, a sixth of the no-stall clip cost). In the stalling world 100 probe episodes of 1,000 cost
about a tenth of the value of caution (about 10 points if that value is near the record's 106), which is why B-R2's
safety is judged after the decision and its release on the whole deployment. B-V1 is the promise (bank-conditional,
exact under equal weights for independent tries); B-V2 asks whether that promise transfers to the deployed map, which
nothing guarantees (the try law differs from the deployed map's visited states). The +0.02 allowance covers the
bootstrap's resolution at the expected number of uses.

**What the outcomes mean.**

| B-R | B-V1 | B-V2 | Meaning |
|---|---|---|---|
| pass | pass | pass | A safety certificate per deviation works here: it keeps the fallback where deviation harms and releases it where it helps. Credit to the placement only (equal weights make BCA the exact binomial test). |
| fail (release fails), safety pass | pass | pass or no uses | It is safe but refuses too much: evidence below the floor turns it into clipping always (expected offline if fewer than about 30 tries). |
| fail (safety fails) | any | fail | The certificate lets harm through: the score does not carry the harm at deployment. |
| pass | pass | fail | Promise kept on the tries, broken in deployment: the try law does not represent the deployed map. |

B-R2 passing and B-R1 failing: online tries rescue the per-deviation certificate (the floor drops to 29 without weights)
at the probing cost reported. B-K1 failing: the score's certificates do not depend on outcomes.

### 7.4 Option C marks

| Claim | Pass iff |
|---|---|
| C-R0 (kill) | `P_S(C-ORACLE) - P_S(host) >= +20` pooled, strong current |
| **C-R1** | `NET(C-BCA) - NET(C-HINGE matched)`: mean >= +5 and L > 0; and in S, recorded / big waves / big kicks each >= host - 3 (pooled) |
| C-R2 | `NET(C-BCA x A) - max(NET(CLIP x A), NET(C-HINGE x A))`: L >= -5 (non-inferiority) |
| **C-V1** | `COR` (final refresh) >= 0.90 as the seed mean with L >= 0.88, in S and in N |
| **C-V2** | `RS <= 0.50` and `CAC >= 0.85` (seed means), in S and in N |
| C-K1 | `NET(C-BCA) - NET(C-F0)`: mean >= +5 and L > 0 |
| C-K2 | `NET(C-BCA) - NET(C-RATIO)`: mean >= +5 and L > 0 |
| C-K3 | `NET(C-BCA) - NET(C-SC)`: mean >= +5 and L > 0 |
| C-K4 | `NET(C-BCA) - NET(C-SUP matched)`: mean >= +5 and L > 0 |

**Rationale.** The record puts C's bars at the switched clip (106.1) and the switched hinge (107.4), not at the host
(PLACE `red.challenges[6]`): within the stalling world a no-signal clip beat the exact-error oracle (93.1 against 79.7,
stage 1 `RESULTS.md:44, 55`), and the lower edge lost to a matched range hinge by 45.6 (results-2026-10-14
`README.md:25`). So C-R1 compares with the matched hinge and C-R2 with the switched plain actions. C-V1 is the band's
own promise at the declared law (PLACE theory, Uses 2-4); C-V2 blocks meeting it by refusing (the red team's fix; 48-75%
of draws refused before, PLACE `red.challenges[11]`). The credit bars are the same +5 as A's. C-I1 is expected to fail
(eta's coefficient of variation 0.065-0.111 against a ratio spanning orders of magnitude, PLACE `red.challenges[11]`).

**What the outcomes mean.**

| C-R1 | C-V1 | C-V2 | C-K1 to C-K4 | Meaning |
|---|---|---|---|---|
| pass | pass | pass | pass | BCA's band, at the actor's own command, beats a matched support rule, keeps its promise usefully, and its width, eta and Bayesian step each earn credit. |
| pass | pass | pass | K1 fail | The gain is refusal: no value credit past the data (support), not the band (as F - F0 = +0.14/+0.08 before). |
| pass | pass | pass | K2 fail | The band's shape is the weighted threshold's scale; eta carries nothing (wording fixed by C-I1's readout either way). |
| fail | pass | fail | any | The promise is met by refusing; the band does not shape the actor usefully, and a plain hinge takes the room (expected). |
| fail | fail | any | any | The band does not keep its promise at the declared law on a Unifloral critic. |

C-R2 failing with A-R1 passing: once the switch is right, a plain action takes all the room and the lower edge is not
needed.

### 7.5 Across options

- A and B read the same score with the same margin; they differ in the default. Where the evidence is thin, A leaves the
  host alone and B falls back. The paired table of their decisions per seed-world is reported, with the returns of each
  disagreement.
- Verdict per option: "placement works" iff its primary return and primary validity claims all pass. "BCA earns credit"
  iff, in addition, at least one of its credit claims passes (after Holm's correction where section 8.3 applies it).
- **Credit map: which comparison separates which part of BCA from which plain rule.**

| BCA's part | Separated from | By |
|---|---|---|
| The residual score as an outcome reader | support and range rules that read no outcome | A-R1 (against clipping always and random switching), A-K2 and B-K1 (shuffled outcomes) |
| Reading held-out rows | in-sample survivors | A-K3 |
| The Bayesian step (WBCP's posterior and test atom) | split conformal, order statistics, bootstrap, means, rank test, exact binomial | A-K1, C-K3; for B, none: with equal weights it is the exact binomial test (stop rule S6) |
| eta (the scale network) | the same band with eta = 1 | C-K2 (wording also by C-I1) |
| The band's finite width | refusal alone (no value credit, no band) | C-K1 |
| The band beyond support | a support-only penalty and a range hinge of the same strength | C-K4, C-R1 |
| The weights | unweighted posteriors, rejection sampling, weighted split conformal, weighted means | A-K4 (localizing weights, mixed world); B-K2, B-K3, C-K5 (importance weights, module W) |
| The margin (not BCA's, but carried the record) | the same rules at margin 0 | reported: A-BCA0 and the margin sweep |

---

## 8. Statistics

### 8.1 Pairing and seeds

- For each claim's difference D (between two policies, or two rules), per seed: the mean over the bank's 1,000 starts
  (paired by start), with its paired standard error.
- Across seeds: the mean and sd (n - 1) of the per-seed values; `L = mean - t(0.95, S - 1) x sd / sqrt(S)`
  (t = 2.1318 for S = 5, 1.8331 for S = 10). One-sided p = `P(T_{S-1} >= mean / (sd / sqrt(S)))` (HOSTS
  PRESPEC.md:540-542).
- No pooled statistic is shown until every seed of the stage is complete; per-seed values are shown as they arrive
  (HOSTS PRESPEC.md:549-550).

### 8.2 Coverage and error rates

Per seed: episode bootstrap for step-level rates, Clopper-Pearson for replicate rates. Across seeds: seed mean with the
t bound, plus the count of seeds below the bar.

### 8.3 Multiplicity

- Within an option, the verdict needs all its primary claims: no correction (all must pass).
- Across the three options: each verdict at one-sided 5%; a statement that "at least one placement works" uses Holm's
  correction over the three primary return claims.
- Credit families judged by a one-sided t over seeds, Holm within each: A {A-K1, A-K3} (A-K4, in the secondary worlds,
  alone); C {C-K1, C-K2, C-K3, C-K4}; module W {B-K2, B-K3, C-K5}. A-K2 and B-K1 are fixed-rule comparisons without a
  p-value (as step 1's C3) and are not corrected.
- Information items: no correction; pointers, never evidence.

### 8.4 Power from past variance and the size rule

- Seed spreads on record (step 2's add-on report, CONN `restructure/step2_results/addon_report/report.md:138-188`):
  NET seed SE 4.20 (sd about 9.4); BCA vs conformal NET seed SE 0.85 (sd about 1.9); range hinge vs the BCA ramp about
  3.9 (sd about 8.8). Start noise is small: NET's start SE 1.58 against 4.20 from seeds, so 1,000 starts per bank are
  enough.
- Smallest effect detected with 80% power (one-sided 5%, noncentral t): 1.36 sd with 5 seeds, 0.85 sd with 10
  (floor_arith.out section 5).
  - A-R1 (expected difference about 61.8 if the switch is right): power about 1 at 5 seeds; one missed stall fails it.
  - Credit claims at sd 1.9: a true difference of about 5.7 passes 80% of the time at 5 seeds (the +5 bar binds).
  - C-R1 at sd 8.8: about 12 points detectable at 5 seeds, 7.5 at 10.
  - The switch's error rate from seed-worlds alone is weak: 10 of 10 correct bounds it at 74% (Clopper-Pearson 95%
    lower bound), 20 of 20 at 86%. That is why A-V1 and B-V1 use 1,000 replicates per seed-world.
- **Size rule (fixed now).** Confirm on 5 seeds (25-29). Add 30-34 if, for any primary return claim, the pilot's mean and
  sd give power below 0.8 at 5 seeds and at least 0.8 at 10. Decided and recorded before any confirmatory datum exists.

---

## 9. Stages and compute

### 9.1 Training runs per seed (100,000 outer steps each)

| Runs | S | N | MW+ | MW- | Cost class |
|---|---|---|---|---|---|
| Host (Unifloral TD3+BC) | 1 | 1 | 1 | 1 | plain |
| Clone (Unifloral BC) | 1 | 1 | | | light |
| A's gated hinge (k = 1) | 1 | 1 | | | plain |
| C-BCA, C-SC, C-k0 | 3 | 3 | | | with BCA fit |
| C-F0, C-RATIO, C-ORACLE (unmatched by design) | 3 | 3 | | | plain |
| Matched controls C-HINGE, C-EVEN, C-RAND1, C-RAND2, C-SUP: 4-rung ladder + matched run in S, locked coefficient in N | 25 | 5 | | | plain |
| **Total** | 34 | 14 | 1 | 1 | **50** |

Extra matching rounds (at most 2 per control) are covered by a 20% allowance.

### 9.2 Compute per stage

Measured speeds: on this container (4 CPUs, no card), Unifloral's TD3+BC step restated on made-up arrays runs 13.8-14.5
ms per outer step with four runs side by side (about 23.5 minutes per 100,000 steps; about 10 runs per hour on 4 CPUs),
16.3-16.8 ms with a BCA-like per-step fit (about 27.5 minutes), and 14.0-14.1 ms with the lower edge (no measurable
extra) (`bench/conc_*.out`). On the laptop's card the hosts measured 0.58 ms alone and 1.26 ms with BCA (about 1.0 and
2.1 minutes per run; HOSTS PRESPEC.md:601-604). An evaluation of one policy on one bank with its capped twin takes about
2 s; one online episode 11-17 ms; `Q^pi` for 1,000 pairs about 9 s (`bench/rollout_*.out`).

| Stage | What | Cloud CPU (4 cores) | Laptop card |
|---|---|---|---|
| 0, code tests on made-up data | unit tests, reproduction tests, smoke runs of 2,000 steps, freeze | about 5-10 CPU-hours (the hosts' stage 0 took about 17 processor-hours and 4.2 card-hours, HOSTS PRESPEC.md:583-586: budget risk) | same, plus about 0.5 card-hours |
| Pilot, seeds 0-4 (exploratory) | data regeneration (minutes); 50 runs x 5 = 250 runs; margin rule; replicates; online episodes; coverage | training about 98 CPU-hours (+20%: 117), about 25-29 hours of wall time; evaluation about 15 CPU-hours | training about 6-8 card-hours; evaluation about 15 CPU-hours on the laptop's processor |
| Confirmatory, seeds 25-29 (+30-34) | the same, frozen, after approval (no margin rule) | about 117 + 15 CPU-hours per 5 seeds | about 6-8 card-hours + 15 CPU-hours per 5 seeds |
| Module W (optional) | section 10: about 17 runs per seed | about 35 CPU-hours per 5 seeds | about 1.6 card-hours per 5 seeds |

Evaluation detail per seed (CPU): the replicate held-out blocks (about 500,000 recorded-law episodes per seed-world, by
the vectorized `dataset.simulate`, about 10-15 minutes each, so about 1 hour for 4 worlds) and their forward passes;
A's replicates (about 0.1 s per BCA decision; 1,000 replicates x 4 worlds x about 12 rules, plus 200 shuffles and the
sweeps: about 40 minutes); B's gates (seconds) and staged
deployments (H = 1,000 episodes per decision, seconds); C-V1 (50,000 draws x 3 band states, minutes, by hand-off
files), C3 (about 13 minutes); online scoring of about 60 deployed policies x 4 situations x 2 banks (fresh and
standard) at about 2 s each (about 16 minutes).

Arithmetic per seed, CPU: 6 runs with BCA's fit x 27.5 min + 42 plain runs x 23.5 min + 2 clones x about 10 min = about
1,170 CPU-minutes (19.5 CPU-hours). Card: 6 x 2.1 + 42 x 1.0 + 2 x 0.5 = about 56 minutes, plus about 0.5 minutes of
compilation per run (about 25 minutes): about 1.1-1.3 card-hours.

**Budget caps (proposed for your approval):** pilot 145 CPU-hours, or 9 card-hours plus 20 CPU-hours; confirmatory the
same per 5 seeds; module W 45 CPU-hours, or 2 card-hours plus 8 CPU-hours. A stage stops between whole seeds when its
cap would be passed (stop rule S10).

### 9.3 Order

1. Stage 0 on made-up data; freeze; your approval of stage 0's summary and of the pilot.
2. Pilot: data and counts (commands only: `t_lo`, group sizes, rung tries), then hosts (seed by seed, S and N first, then
   MW+ and MW-), then K1; then the margin rule (N's hosts and the ladder) and K2, K4; then A's and B's offline decisions,
   locked; then C's arms, matching, lock, K3; then all online episodes and truths; then the pilot report.
3. The confirmatory pre-registration: this document with the margin values, the size decision, the budget and any change
   you make, frozen and approved at your terminal, before any confirmatory datum exists.
4. Confirmatory seeds in the same order; verdicts only when every seed is complete.

---

## 10. Module W: the known shift (optional; decision D4)

**Why.** BCA's weights can earn credit only where the evidence law and the deployment law overlap with a known ratio.
On the main log the recorded current is U[0.1, 0.7] and strong current U[0.7, 1.1], so the supports are disjoint and no
weight exists (`thruster_chain.py:87-92`). The weighting test's world has an exact current weight `w(c)` on [0.1, 1.1]
(MAIN `experiments/shift_iw/world.py:7-16`), and its random-command rows fill the rungs past 80% (on seed 0 arm B had
42, 54 and 78 thruster-1 tries in the 80-85% rung on medium-replay, medium and medium-expert, and 35-67 in each higher
rung on medium-expert; CONN `outputs/shift_iw/explore/arms_abc_f0/SCREEN_seed0.md:41, 46-52`). It is stalling-only (`world.py:69`), so a
no-stall twin must be built: the generator with `PHYSICS` passed in (`world.simulate` and `HoldRecorder` already accept
physics, `world.py:218, 324`).

**Design.** Quality medium-expert (the hosts' primary quality); weighting-test seeds 0-4 (pilot) and 25-29
(confirmatory); Unifloral TD3+BC hosts in S and the new twin N; tries = training-law log rows of the calibration split
outside the bank, or the evaluation split, outside the ball (`arm_b.py:8-9`); deployment = the strong-mix law; weights
exact `w(c)` (B-K2, C-K5) or estimated by a classifier on the weighting test's classifier samples (B-K3).

- **B arms**: B-RUNG with the residual score and the weighted BCA tail gate against rejection sampling (accept each try
  with probability `w / max w`, then the exact binomial test), weighted split conformal (Tibshirani et al. 2019), a
  self-normalized weighted mean with an empirical-Bernstein bound, the unweighted BCA gate, and shuffled outcomes.
- **C arms**: C-BCA with exact state x command weights for the strong mix, C-BCA unweighted (w = 1), C-F0, C-HINGE
  matched (about 15 runs per seed in S and N).
- **B-K2 / B-K3 pass iff**: in the stalling world, harmful certifications (true harm rate above alpha_B, from fresh
  tries) at most 5% (Clopper-Pearson upper <= 0.073) for every gate compared; and in the no-stall twin, BCA's weighted
  gate certifies at least 10 percentage points more truly safe rungs than the best of rejection sampling, weighted split
  conformal and the weighted mean bound (pooled over seeds, L > 0). With exact weights the toys gave 33.6% against 19.0%
  for rejection sampling (LIVE 10811); evidence floors are about 29 / 39 / 83 tries for no, moderate and strong shift
  (LIVE 10987-10992).
- **C-K5 pass iff**: `NET(C-BCA weighted) - NET(C-BCA unweighted)`: mean >= +5 and L > 0, with C-V1 met by both.
- The weighted promise is empirical, not proven, at these sizes (PLACE `red.challenges[3]`; MAIN
  `experiments/wbcp/DEPENDENCE.md:1114-1133`); module W measures it.

---

## 11. Code plan

### 11.1 Reused, read-only (imported, never edited)

| Need | Where |
|---|---|
| Simulator, start laws, banks, env | MAIN `experiments/shift/thruster_chain.py` (`dynamics`, `draw_starts`, `bank_starts`, `world_physics`, `ThrusterChainEnv`) |
| Exact truth | MAIN `experiments/shift/truth.py` (`rollout`, `q_value`, `q_value_jax`, `anchors`, `points`, `attribution`) |
| Data, gap, bank | MAIN `experiments/shift/dataset.py` (`generate`, `simulate`, `draw_streams`), `shift.py` (`define_gap`, `make_removal`), `recorder.py`, `calibration/bank.py` |
| WBCP | MAIN `calibration/wbcp.py` |
| Unifloral import and step | HOSTS `experiments/shift_hosts/unifloral_import.py`; `baselines/unifloral/algorithms/td3_bc.py`, `bc.py` |
| BCA arithmetic, band parts, refresh | HOSTS `bca_core.py` (`fit_scale_core`, `calibrate_bank`, `apply_refresh`), `unifloral_td3_bc_bca.py` (`_own_target`, `_band_values` as the template for a Q1 version) |
| Policies, actor files, hand-off | HOSTS `policies.py` (`MLPAdapter`), `actors_f32.py`, `handoff.py` |
| Scoring metrics | HOSTS `evaluate.py` (`score`, `paired`, F1/F2, late peak; stalling-world only as written, `evaluate.py:88`) |
| Coverage arithmetic | MAIN `experiments/shift_iw/coverage.py`; HOSTS `band_readout.py` |
| Step 1's rules | MAIN `outputs/.../restructure/step1/step1lib.py` (restated in the package with tests that compare against it) |
| Arm B's walk, plain tests, ceilings | CONN `outputs/shift_iw/explore/arms_abc_f0/code/arm_b.py` (restated, tested against it) |
| Lower-edge objective and penalties | MAIN `experiments/shift_iw/lower_edge.py`; stage 1 `harness.py` (hinge, random surfaces); CONN `command_le_train.py`, `arm_f0.py` |
| Toy gates (mean gate, tail gates, rejection sampling, weighted split conformal) | MAIN `bca_mean_gate_toy.ipynb`, `bca_weights_toy.ipynb` (lifted into a module, tested against the notebooks' numbers) |

The new package lives at `experiments/placement_test/`, outside the hosts' fingerprint list (`EXTERNAL_CODE`, HOSTS
`gate.py:56-65`), so the hosts' approval stays valid; nothing is added under `experiments/shift`,
`experiments/shift_iw` or `calibration`. The hosts' approval is not reused for this study.

### 11.2 To write

| Module | Does |
|---|---|
| `prespec.json`, `PRESPEC.md`, `freeze.py`, `gate.py`, `approve_placement.py` | the frozen rules, hashes and your terminal approval |
| `seeds.py` | generator-and-seed guard (S2); stream registry (section 3.5) |
| `data_main.py` (scoring env) | main gapped log in S, N; spliced MW+, MW-; evidence rows; Unifloral prepare and hand-off; leak checks |
| `hosts.py` (training env) | Unifloral TD3+BC and BC runs; run records (fingerprints, environment, timings) |
| `scores.py` | residual score on a Unifloral critic (min head, Q1, noisy-target variants) |
| `harm_switch.py` | A's rules, shuffles, random switching, cells and kernel |
| `safety_cert.py` | B's gates (tail, exact, mean, HCOPE-style, looks, e-process), rung walk, maps |
| `recorder_density.py` | the main recorder's exact command density outside the ball, the query law's density, weights |
| `lower_edge_unifloral.py` (training env) | the restated TD3+BC step with the lower-edge slot, Q1 band parts and refresh, C's penalties, matching |
| `online.py` (scoring env) | episode runner (env.step arithmetic), mixed physics, fresh banks and anchors, no-stall scorer, staged-B loop, attribution |
| `truth_pop.py` | fresh populations, replicate evidence sets and banks |
| `coverage_c.py` | C-V1, C-V2 and the information readouts |
| `stats.py` | paired t bounds, Holm, Clopper-Pearson, episode bootstrap, design effect, size rule |
| `report.py` | `results.json` and `results.md`, write-once |
| `tests/` | unit tests on made-up data for every module (stage 0) |
| Module W only | the weighting-test no-stall twin; its prepare; weighted gates |

### 11.3 Outputs and report

- Official outputs only under `runs/placement_test/<stage>/`, write-once, each with a SHA-256 sidecar; run records hold
  the fingerprint, seeds, streams, CPU or card, software and timings; every process pinned to one CPU with
  `OMP_NUM_THREADS=1` and no bytecode (step 1 PRESPEC.md:107-113).
- `decisions.json` (locked before any return), `lock.json` (network fingerprints and strengths), `values.json` per
  seed-world, `results.json`, `results.md`.
- `results.md`: per claim a badge (passed, shelved, flagged, pending, not judged, no room, did not learn), the numbers
  against the mark, per-seed values, and the fixed wording; then per option the write-up in your format: The idea / What
  we tested (and the kill rule) / What happened / Why (HOSTS PRESPEC.md:515-516). Every pilot page carries
  "EXPLORATORY: seeds 0-4 were seen".

---

## 12. Owner decisions still open (flagged)

| # | Decision | Recommended |
|---|---|---|
| D1 | Generator for the primary test | main study (only one with a no-stall twin, the gap and every earlier A/C result) |
| D2 | Confirmatory seeds | main 25-29, with 30-34 by the size rule (section 8.4) |
| D3 | Margin | break-even rule on the pilot (section 3.6). Alternatives: a fresh exploratory block 20-24 for it (then the pilot's A results are not fitted to their own margin), or the record's 0.10 (no meaning on a new host; not recommended) |
| D4 | Module W | include, medium-expert only: the only place BCA's weights can earn credit (B-K2, B-K3, C-K5); needs the weighting-test no-stall twin |
| D5 | Evidence pool for A and B | calibration and evaluation splits (about 35-70 full-throttle rows expected); calibration split alone reported |
| D6 | Where Q1 is read in C's objective | at `pi(x)` (k = 0 reproduces the host); the placement analysis's form (Q1 at `pi(x) + xi`) as information only |
| D7 | C's protocol | co-trained restated Unifloral step as primary; actor-only as information (C-I2) |
| D8 | Mixed worlds | train hosts on the spliced logs (2 more runs per seed); alternative: drop A-K4 |
| D9 | Staged B | N = 100 batch as primary, H = 1,000, twin outcome as reference only |
| D10 | alpha_B | 0.1 (tail gate); arm B's 0.5 (median) reported only |
| D11 | Compute site and caps | the laptop card for training, the cloud or the laptop's processor for evaluation; caps of section 9.2 |
| D12 | Continue to confirmation if a pilot kill rule fires | no: a fired K1-K4 ends that option's test at the pilot, with its write-up |
| D13 | Near-80 version 8 | it runs on main seeds 10-14 now; its claim 4 confirms step 2's switched ramp. If it reports first, A here is partly a replication with new rivals (plain tests at the margin, caps, online episodes) on a Unifloral host |

---

## 13. What this design rests on (verified for this document)

- Step 1's certificate, rules and process: MAIN `outputs/shift/beyond-the-data/restructure/step1/PRESPEC.md:51-63,
  133-168, 192-268`; code `step1lib.py:174-352`.
- Round 1's routes (held-out rows 106.1; 55.7 without the margin; plain tests at the margin also 106.1; ladder; small
  evidence; mixed world; run-time monitor): MAIN `restructure/routes/round1/ROUND1.md:23-146`.
- Step 2's arms, matching, marks and results (NET 102.9; conformal 97.3; range hinge 76.5 with no-stall cost 30.9):
  MAIN `restructure/step2/PRESPEC.md:97-257`; CONN `restructure/step2_results/results.md:9-38` and
  `addon_report/report.md:138-188, 240-245`.
- Fallback: deployment clip +106.0 in strong current against the clone's +68.9; -61.8 in the no-stall world:
  MAIN `outputs/shift/beyond-the-data/fallback.md:83-88, 167-178`.
- Stage 1's room (oracle 79.7, no-signal clip 93.1, no-stall costs 1.3 and 30.9): MAIN `stage1/RESULTS.md:28-65`.
- Stage 2 v6 (lower edge 24.9 against the matched hinge 70.5; no-stall cost 14.8; co-trained 28.0 against -16.8):
  MAIN `results-2026-10-14/README.md:20-36, 69-77`.
- ThrusterChain, truth, data, recorder: MAIN `experiments/shift/thruster_chain.py:1-53, 83-92, 143-149, 308-336,
  389-481`; `truth.py:1-35, 77-108, 330-383`; `dataset.py:4-15, 152-226`; `recorder.py:8-17`.
- WBCP: MAIN `calibration/wbcp.py:43-117`.
- Unifloral and the hosts: MAIN `baselines/unifloral/algorithms/td3_bc.py:23-46, 174-279`; `configs/unifloral.json`;
  HOSTS `experiments/shift_hosts/PRESPEC.md:138-226, 254-282, 466-555, 601-621`, `unifloral_td3_bc_bca.py:1-40,
  316-413`, `evaluate.py:1-125, 302-362`, `unifloral_import.py:58`, `gate.py:50-70`; HV2 `PRESPEC.md:1-46, 866-930`.
- Arm B and the lower edge: CONN `arms_abc_f0/code/arm_b.py:1-155`; `channel/command_le_train.py:1-18, 92-118`;
  MAIN `experiments/shift_iw/lower_edge.py:1-64, 196-224, 293-312`; stage 1 `code/harness.py:373-376, 530-533`.
- Weighting-test world: MAIN `experiments/shift_iw/world.py:1-70`.
- Seeds: CONN `near80/code/near80lib.py:68-70`; `seed_registry.json`.
- Floors, power and precision: `measure/floor_arith.out`; LIVE 5104-5112 (bank standard errors), 10502-10560 (tail-gate
  floor; 36 with 1,000 draws), 10985-10996 (weighted floors).
- Placement analysis and its red team: PLACE (rank test_plan T1-T6; red challenges, missing options, revised
  recommendation).
- Benchmarks: `bench/conc_alone_*.out`, `conc_bca_*.out`, `conc_lower_*.out`, `rollout_1cpu.out`, `rollout_4cpu.out`.

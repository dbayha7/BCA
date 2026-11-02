# OOD definition v4, revision 4 (DRAFT r4, not approved): the exploratory native screen

Written 2026-11-01 after the user chose the leaner route ("Option B") with corrections. This file is a versioned
revision of `DRAFT_DEFINITION.md` (r3). That file is hashed into the sealed formal G2 of candidate knee_s10_v2 and is
not edited; everything in it stands except what this file changes. r4 changes no sealed rule and no G2 check, so
candidate knee_s10_v2's formal G2 PASS (attempt 1, 2026-11-01, `../knee_s10_candidate_v2/formal_g2/`) is a PASS under
r4 as well. Nothing here is approved, sealed or run; the user adopts r4 in their own approval words, candidate by
candidate.

## 1. What r4 changes, and why

r3 reaches a BCA test only after a policy-evaluation (PE) chain: G1, a variance pilot, G3 learnability and G4
withholding, then a native stage. That chain earns an error-controlled claim about critic over-valuation, but it
spends about 40 GPU runs before anyone learns whether the learning actor even enters the withheld region on this
bed, and BCA appears in none of its gates.

r4 keeps G1 and moves one native, exploratory screen ahead of the PE chain. The screen asks the study's actual
question on this bed, as a go/no-go: **does BCA, as built, change the learning actor's decisions where the critic's
value is unsupported, beyond a paired native baseline?** The PE chain is deferred, not deleted (section 8): it stays
the route to an error-controlled over-valuation claim, and it remains a valid critic-error diagnostic.

Three corrections from the user shape how the question is stated. They are part of the definition:

1. **Nothing here says over-valuation is built in.** G2 established an exact value difference between two physics
   worlds under a fixed continuation: rho_j = Q_nostall(s10) - Q_stall(s10), median 1.765 on 96 fresh anchors. That is
   a property of the bed and pi_ref. Whether this host's learned critic over-values the past-knee contrast is a
   hypothesis the screen tests (reads O1 and O2 below). A screen that finds no over-valuation is a finding about the
   host on this bed, not a failed premise.
2. **A three-seed null does not identify BCA as the problem.** The dose (kappa), the training time (20,000 updates)
   and whether the labelled control learned the departure stay confounded with "BCA does nothing". The screen's
   negative reading is therefore always "no pre-registered effect at this dose and budget on this bed", never
   "BCA does not work". Separating those causes is the confirmation stage's job (section 6).
3. **Frozen-actor evaluation cannot measure this BCA's policy effect, and that is specific to this implementation.**
   BCA as built enters only the actor's objective (the actor maximizes Q1 - kappa U; `KNEE_BED_CONCLUSION_20261031.md`,
   Terms). Its critic is the host's. So a frozen-actor run shows no BCA effect by construction. PE mode still
   diagnoses critic errors exactly (that is what the deferred chain does), and a BCA variant that entered the critic
   could be tested there.

## 2. Vocabulary added to section 2 of r3

| Term | Meaning | Established by |
|---|---|---|
| **Retained-supported** | The scored states lie inside the state distribution of the rows the withheld arm actually trains on (CUTOFF's retained rows), not only inside all fitting rows. | RS1, exact computation |
| **Over-valued natively** (finding) | In native training on this bed, CUTOFF's critic over-values the past-knee contrast at the scored slot, against the exact value under its own continuation, by the pre-registered margin, in every screen seed. | The native screen, O1 and O2 |
| **Screen GO / NO-GO** | The screen's go/no-go reading for the confirmation stage. GO is not a claim; it is permission to spend confirmation seeds. | The native screen |

## 3. Preconditions: nothing trains until all four hold

1. **G2 PASS** for the candidate (exact; unchanged from r3).
2. **RS1 HOLDS: retained-state support.** G2's D3 measured state support against all fitting rows (its sealed rule).
   CUTOFF's critic trains on the retained rows only, so RS1 repeats the check against the retained population: the
   64-nearest-neighbor standardized distance of the 128 starts and the 96 gated anchors to CUTOFF's retained rows,
   against the 0.99 quantile of the leave-one-episode-out distances of the retained deep-load rows at the same time
   step; bars unchanged (starts 95%, anchors 90%). Rule, code and result: `../knee_s10_candidate_v2/retained_support/`.
   Under the widened recorder, retention within a load band is independent of the state (a0 depends only on the band
   and the sealed uniforms), so RS1 is expected to agree with D3 up to noise; it is required because the expectation
   is an argument and the check is a measurement. A RS1 miss stops the native screen on that candidate.
3. **G1 recorded.** G1 is unchanged in substance from r3: the host's 2x256 critic, fit by supervised mean-squared
   error to exact Q^{pi_ref} labels on the twin's (s, a) rows, must capture the departure at the scored slot (L1 at
   s10). It fixes q* (the per-anchor share bar for the labelled arm's learnability read) and the in-support
   resolution band. Its rule: `../knee_s10_candidate_v2/g1/G1_RULE.md`. NOT REPRESENTABLE stops the candidate: if the
   host's critic cannot hold the departure with exact labels, no learning run on this host is informative about it.
4. **The screen's frozen protocol and budget reviewed by the user.** The screen's package writes its rule, seeds,
   manifest and caps first; the user reviews them and approves at their terminal; the BCA arm is approved in the
   user's own words (the standing rule for any BCA training arm). No agent's message counts.

## 4. The native screen: design

**Host.** The pinned Unifloral TD3+BC (`MODEL_IDENTITY.md`: revision f2dc1278, upstream `algorithms/td3_bc.py`
sha256 6f5ab6f9..., loaded by `payload/unifloral_td3bc_validation_v1/native_loader.py`), with its native settings:
target-policy smoothing on, the actor learning, two critic updates per actor update, gamma 0.98, batch 256,
Adam 3e-4. No CORL-derived learner anywhere in the package.

**BCA.** The uncertainty-only bridge (`payload/unifloral_bca_pipeline_validation_v1/bca_bridge.py`): the actor
maximizes Q1 - kappa U with U = R x eta(s, a) x u, exactly as the leverage readout and the dose screen used it
(`../bca_dose_screen_v1/PROTOCOL.md`). The dose kappa is fixed before the run. Two doses are on record: the
configured 3 and the dose screen's kappa_A = 63; the user chooses one or both at review (section 9). Each dose is its
own branch set with the same seeds.

**Pipeline.** knee_edge_ood_v3's Stage 1 (`stage1.py`, `stage1_worker.py`, `stage1_eval.py`, `stage1_math.py`,
`units.py`), reused as the dose screen reused it: the widened recorder substituted, the datasets by digest, each GPU
job a separate process under its cap, the BCA branch paired with the native branch of the same fork by shared random
numbers. Two adapter changes are required for the labelled arm and must be built and tested before sealing (as
pi_ref's runtime hook was): the superset twin's update budget scaled by its row ratio (76,800 / 47,978, so 32,000
updates where CUTOFF gets 20,000) and CUTOFF's normalizer shared by both arms (r3 section 3.3). If the user prefers
Stage 1's count-matched RANDOM control instead (no adapter change), the confound is stated in the package rule: the
two arms then differ in which in-support rows they hold.

**Data.** The candidate's formal G2 datasets, by digest (knee_s10_v2: stall3 c2a6d616..., no-stall e3549869...).
Arms: CUTOFF (retained rows) and the labelled twin (all fitting rows). Worlds: stall3 for every arm; no-stall for the
two native arms (the placebo for O1 and O2) and for BCA on CUTOFF (the over-correction read U1).

**Seeds.** Three screen seeds from a new namespace, `knee_s10_candidate_v2/native_screen/`, derived and audited like
the formal G2 seeds. A confirmation namespace, `knee_s10_candidate_v2/confirmation/`, is reserved and never drawn by
anything in r4. Screen seeds are never confirmatory and are never reused by the confirmation stage or by the deferred
PE chain.

**Schedule and units.** Warmup to 10,000 updates per (seed, world, arm), native; R0 at 10,000; branches from the
saved fork to 20,000: native, and BCA at each chosen dose with the fork's own branch seeds and random-number stream;
one BC-only actor per arm per seed in stall3; evaluation at 20,000 on the sealed E2 banks and the 96 gated anchors,
exact simulator values under each arm's own clipped continuation (the Stage 1 primary) with the pi_ref reference
reported beside it.

| Unit | Count (one dose) | Count (two doses) |
|---|---|---|
| Warmups (3 seeds x 2 worlds x 2 arms) | 12 | 12 |
| Native branches | 12 | 12 |
| BCA branches (CUTOFF stall3, CUTOFF no-stall, twin stall3; 3 seeds each) | 9 | 18 |
| BC-only actors (2 arms x 3 seeds, stall3) | 6 | 6 |
| Evaluations (one per seed index) | 3 | 3 |

Order, fail-fast: warmups, R0, native branches, the reach read R1 on the native CUTOFF branches, then the BCA
branches only if R1 holds in at least one seed. A bed the actor never reaches spends no BCA compute.

## 5. Frozen reads and criteria

Every read is reported with its per-seed values whether or not it decides. "Every seed" means all three. KC is
Stage 1's kink-capture error, KC_slot = [Q1(slot) - Q1(-KNEE)] - [Q^pi(slot) - Q^pi(-KNEE)], per anchor, against the
exact value under the arm's own clipped continuation (`stage1_math.kc`); its decomposition against the pi_ref
reference is reported (`kc_decomposition`). The scored slot is s10; tol = 0.15 and c = 0.3 as in r3. rho_own,j is
the exact gap at s10 under the arm's own clipped continuation, computed by the evaluator in both worlds.

**Validity (any miss: INCONCLUSIVE, no BCA reading).**

| id | Read | Bar |
|---|---|---|
| V0(a) | adequacy: J_E1(20k) - J_E1(untrained), CUTOFF and twin, stall3 | >= +50 each (sealed) |
| V0(c) | valid anchors common to both arms | >= 32 (sealed) |
| V0(e) | a1..a3 past the knee on E2 steps | <= 0.005 (sealed) |
| V0(f) | twin learnability, one-sided, as sealed: median KC_L(s10) <= 0.3 (median rho + 0.15), median rho >= 0.30 | sealed form |
| L1n | twin learnability, two-sided per anchor: -0.15 <= KC_L,j <= 0.3 rho_own,j + 0.15 at a share of valid anchors >= q* (from G1), and the median inside the same band | q* from G1 |
| BAL | balance in support: median_j \|KC_C,j - KC_L,j\| at edge_700 (0.1 inside the knee) | <= the resolution band from G1 |
| PLC | placebo: in the no-stall world, median_j (KC_C - KC_L)(s10), pooled over seeds | within +-0.30 (2 tol) |

**Over-valuation (findings about the host on this bed; not validity).**

| id | Read | Bar |
|---|---|---|
| O1 | CUTOFF over-values the contrast: median_j KC_C(s10) >= 0.5 and >= 0.5 x median_j rho_own,j (the sealed K1c form, at s10) | every seed |
| O2 | the over-valuation is carried by withholding: median_j (KC_C - KC_L)(s10) >= 0.5, and (stall3 minus no-stall) of that median > 0 | every seed |
| U0 | underestimation, reported: the share of anchors with KC <= -0.15 per arm and world; the signed KC quantiles; the twin's under-capture share (KC_L,j < -0.15) | reported |

**Reach and harm (the bed).**

| id | Read | Bar |
|---|---|---|
| R1 | CUTOFF's R_deep- entry rate on E2 (stall3) minus its BC-only actor's | >= 0.25 in every seed (sealed K2 bar); the twin's excess <= 0.10, else INCONCLUSIVE-reach |
| R2 | harm from the hidden structure: K3a, K3b, K3c | >= +2.0 each (sealed) |
| R0 | at 10,000: adequacy >= +50 (else INCONCLUSIVE); CUTOFF reach < 0.05 in every seed -> STOP "not reached at 10k", no branch trains | sealed |

**BCA (the go/no-go; each paired by seed with the native branch of the same fork).**

| id | Read | Bar |
|---|---|---|
| B1 | reach: entry(BCA-C) - entry(HOST-C) on E2, stall3 | <= -0.10 in every seed |
| B2 | harmful over-estimation, on the Q values: among gated anchors where the past-knee slot is truly worse (C_true(s10) < 0), the share ranked correctly by the penalized value Q1 - kappa U minus the share ranked correctly by raw Q1 (the dose screen's q_rank_gain_s10) | >= +0.25 in every seed |
| B3 | return: J(BCA-C) - J(HOST-C) on E2 in stall3 | >= +2.0 in every seed |
| B3s | selectivity: [J(BCA-C) - J(HOST-C)](stall3) - [J(BCA-C) - J(HOST-C)](no-stall) | >= +2.0 in every seed |
| U1 | over-correction (underestimation harm): J(BCA-C) - J(HOST-C) in the no-stall world; and the penalized value's under-ranking share (penalized contrast < 0 where C_true > 0) minus raw Q1's | >= -2.0 in every seed; <= +0.25 in every seed |

**Readings.** Precedence: INCOMPLETE > INCONCLUSIVE > the stop readings > NO-GO readings > GO.

| Reading | When | Leads to |
|---|---|---|
| INCOMPLETE | a job failed, a cap bound, or the integrity check failed | one rerun of the incomplete units only, under a new approval |
| INCONCLUSIVE | a validity read failed | fix the named instrument cause; one rerun with fresh screen seeds, at most |
| STOP: not reached | R0 or R1 fails in every seed | the actor does not enter W on this bed; a bed finding; no BCA unit trains |
| NO-GO: nothing to repair | R1 holds, O1 fails in every seed | the native critic does not over-value the contrast here; BCA has nothing to act on; a host-on-bed finding, recorded, no rerun |
| NO-GO: no effect at this dose and budget | O1 and R1 hold, a B read fails in every seed | exactly that sentence in the ledger; a different dose or budget is a new screen |
| NO-GO: over-correction | the B reads hold, U1 fails | BCA costs return where caution is unneeded; recorded |
| MIXED | seeds disagree on a deciding read | no reading; the screen is exploratory, and a MIXED result is a design question for the confirmation stage, not a rerun |
| GO | validity, R1, every B read and U1 hold | the confirmation stage (section 6) may spend confirmation seeds |

**Kill rules, written before any run.** Every cap above fails closed. Any non-finite loss, Q or U in a training job
ends that job INCOMPLETE. The screen never changes a bar, a dose, a slot or a seed after its approval; a change is a
new screen with its own rule.

## 6. What the screen can and cannot say

It can say, for this host on this bed with three seeds: whether the actor enters W and whether that costs return;
whether the critic over-values the past-knee contrast natively, with its sign and size; whether BCA at the chosen
dose changes entry, Q-value ranking and return beyond its paired native baseline; and whether it over-corrects in the
smooth world.

It cannot say that over-valuation holds with a controlled error rate (the deferred PE chain does that); it cannot
separate BCA's mechanism from dose, training time or the twin's learnability on a null; and it cannot say that BCA
beats spreading caution evenly or a data-only rule, because the screen has no strength-matched even-caution arm and
no data-only arm. GO therefore leads to a confirmation stage that must carry, before any claim: fresh confirmation
seeds; a strength-matched even-caution control (a penalty of the same mean action-gradient size that does not depend
on the critic's error); a data-only control; and, for the over-valuation claim, the r3 PE chain (pilot, G3, G4) on its
own seeds.

## 7. Budget, for the user's review

CPU, exact arithmetic, user-run: RS1 (minutes; about 10,000 simulator steps for the anchors and starts); G1 (labels:
two worlds x 76,800 rows x about 50.5 remaining steps, about 7.8 million exact simulator steps, under a 9 million cap;
three supervised fits per world of 20,000 updates at batch 256 on the CPU JAX runtime, minutes).

GPU, user-run in WSL: per training job at most 1,800 s (Stage 1's cap); one dose, 39 training jobs (12 warmups,
12 native branches, 9 BCA branches, 6 BC-only), two doses 48; evaluation at most 2 h and 36 million exact
transitions. Expected wall clock at the exploratory Stage 1's pace: 6 to 8 GPU hours for one dose, 8 to 10 for two,
plus evaluation. Nothing runs before the user's review of the sealed package.

## 8. The gate table under r4

| Gate | Status under r4 |
|---|---|
| G0 instrument | unchanged |
| G2 exact | unchanged; knee_s10_v2's formal PASS stands |
| RS1 retained support | new precondition (section 3) |
| G1 representation positive control | unchanged in substance; now also fixes q* for L1n and the band for BAL |
| Native screen | new, exploratory, go/no-go (sections 4 to 6) |
| Variance pilot, G3, G4 (PE mode) | deferred to the confirmation stage; the route to an error-controlled over-valuation claim and a critic-error diagnostic; fresh seeds |
| G5 native | its reads (N1, R1, R2) are carried inside the screen; a confirmation on fresh seeds repeats them |
| G6 ledger | unchanged; the screen and every reading go in the ledger |

## 9. Decisions for the user at review

1. Adopt r4 for candidate knee_s10_v2 (in the G1 approval words, and again in the screen's).
2. The dose: 3, 63, or both.
3. The labelled control: the superset twin with the two adapter changes (recommended; labels are then the only
   difference) or Stage 1's count-matched RANDOM.
4. The proposed bars in section 5: B1 at -0.10, B2 at +0.25, B3 and B3s at +2.0, U1 at -2.0 and +0.25, PLC at
   +-0.30. They are proposals until the approval freezes them.
5. The GPU budget in section 7.

## Change log

- **r4 (2026-11-01, after the user's choice of the leaner route):** the native exploratory screen placed before the
  PE chain; the three corrections in section 1 made part of the definition; RS1 added as a precondition; G1 kept and
  given the two calibration roles; the PE chain deferred, not deleted; the screen's reads, readings, kill rules and
  budget frozen for review; confirmation seeds reserved and untouched; the screen's claims bounded (section 6).

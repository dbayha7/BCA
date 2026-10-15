# Importance weighting for BCA on ThrusterChain: specification

Version 5, 2026-10-13, not frozen. Version 5 adds five dataset qualities (part A) and a second use of BCA's band, the
lower edge (part B), and replaces version 4's run (version 4 never ran); its expert column is version 4's experiment,
data bit for bit (see "Changes in version 5"). It was revised after a review of version 5 (see "After a review of
version 5"); sections 5-8 are version 5's. Version 4 was revised after two reviews, the owner's decisions of 2026-10-13 and a review of version 4 (see "Changes in version 4").
Everything is still written before any learning or BCA run on ThrusterChain data. The numbers come
from the two scout reports in this folder and one synthetic calculation (`scout/r2_per_step_expectation.py`);
`spec.json` is the machine-readable copy, and every pass-mark number is read from it (`pass_marks[i].thresholds`),
not from the code. BCA's band runs on real data only after the user has read these marks and said go, and only after
the first report says the design check passed. Training any BCA arm needs a further go from the user for that kind of
arm and that quality. The gate checks for an approval file holding the SHA-256 of
`spec.json` and of every code file the run loads from this repository: the experiment's own package with its launch
script, and every reused module the stage runner imports (32 files; a test checks in a fresh process that none is
missing). So neither the marks nor that code can change after approval. Installed libraries (JAX, NumPy, SciPy,
Flax, Optax) are not hashed; every record holds their versions.

**Goal.** (1) Check that BCA's parts work on data where every value is known. (2) With the test distribution known,
check that weighting BCA's calibration restores coverage. (3) Check that it raises test return, if weighting moves
BCA's caution enough to matter: through BCA's own capped pull (the mark "can weighting move return?") or through the
lower edge (part B, the mark "can weighting move return through the lower edge?").

**Terms.**
- **TD3+BC**: the offline learner, an actor-critic pulled toward the recorded commands. Its **critic** is
  q = min(Q1, Q2).
- **BCA's band**: q ± U(s, a), meant to hold the critic's one-step error |y − q| 90% of the time, where
  y = reward + 0.98 × the target critic's next value.
- **Width** U = R × unit × scale network(s, a). The threshold R is set on a held-out **calibration bank** by weighted
  Bayesian conformal prediction (`wbcp.calibrate`, miscoverage 0.1, credibility 0.95).
- **Test mass**: the weight that weighted conformal prediction gives the unseen test point. **Per-step threshold**
  (the method, the owner's choice): each step x uses its own weight w(x) as the test mass, so its threshold is R(w(x))
  (Tibshirani, Barber, Candès and Ramdas, 2019). **Single threshold** (version 3's, kept as a reported comparison):
  one test mass, the mean weight, for every step.
- **Dose** = 1 + 0.5 U / (U + unit) = 1 + 0.5 x / (1 + x), with x = R × scale network(s, a). It multiplies the pull
  toward the recorded command: BCA's own use of its band, the **capped pull**.
- **Lower edge** (version 5, part B): a second use of the band, a penalty k [U(s, π(s)) − U(s, a_i)]_+ in the actor's
  objective, which grows linearly with the width.
- **Coverage**: the share of steps with |y − q| ≤ U. A **refused** step (R = +∞, "cannot certify") never counts as
  covered.
- **Points** (version 5): 0 = uniform random commands, 100 = the stall-aware controller (the textbook gain capped at
  80%), on a **start bank** of 1,000 fixed starts (D4RL's scale). Marks on return read the raw 100-step return
  (**return units**); points are shown beside them. In version 4, 0 points was the recording controller.
- **Dataset quality** (version 5): one of random, medium-replay, medium, medium-expert and expert, as in D4RL. Every
  quality is the recording controller with a **random share**: on each step, with that chance, the command is replaced
  by a uniform random command.
- **v1**: the earlier ThrusterChain study in `experiments/shift`; its code is reused unchanged.

## 1. Distributions and exact weights

ThrusterChain, stalling world, unchanged (`experiments/shift`, read-only). The current c is fixed per episode and
observed. Only its law changes:

p_λ(c) = e^(λc) / M(λ) on [0.1, 1.1], with M(λ) = (e^(1.1λ) − e^(0.1λ)) / λ and M(0) = 1.

| Law | λ | Mean c | Exact weight w(c) = p_test / p_train | Effective bank size after the gap |
|---|---|---|---|---|
| Training, bank, test "none" | −2 | 0.443 | 1 | 843 rows |
| Test "mild" (uniform) | 0 | 0.600 | 0.35396 e^(2c) | 627 |
| Test "strong" | +2 | 0.757 | 0.090718 e^(4c) | 277 |

Given c, everything else is the same in training and test: float starts N(0, 0.5²), waves (amplitude U[0.1, 0.4],
phase U[0, 2π)), recorder and physics.

**The gap.**
- v1's densest-15% ball (`gap.find_gap`) is cut from training and bank rows, never from test steps.
- The exact density ratio of a test step to a kept training row is Z · w(c) outside the ball and +∞ inside it.
- Z = P_train(command outside the ball) ≈ 0.843, by quadrature per seed. It cancels.
- Both weighted arms, exact and estimated, refuse every test step whose command lies inside the ball, before
  `calibrate` is called. The ball is public, so this rule needs no estimation.

**Test mass** (the weight of the unseen test point).
- Per step (the method): a test step's own weight w(c), so a step whose current is common at test and rare in training
  gets a wider band. Weighted conformal prediction prescribes this.
- R(t) is tabulated on a log-spaced grid of test masses, t = 1.01^k for integer k, because calling `calibrate` once per
  step (1,000 posterior draws each) is too slow.
  - The grid covers every mass looked up: the kept training rows and the tilt's certified test steps.
  - A step uses the smallest grid mass at least its own weight. This is conservative, because R(t) never falls as t
    grows when every t uses the same random stream (a heavier test atom only moves each draw's crossing up).
  - Each table is checked for that, and a running maximum is applied if it ever falls; the record says whether it was.
  - The record holds the grid, the largest mass gap (at most 1%) and the largest threshold step between the grid points
    used.
  - A step's "own threshold" means this looked-up, rounded-up threshold everywhere below (coverage, refusals, mark W3,
    mark R2 and the per-row doses).
  - R becomes +∞ ("cannot certify") beyond some mass. A step whose own R is +∞ is refused and never covered; its share
    is reported. The record also counts the steps that may have been refused only because the lookup rounded their mass
    up: every step whose looked-up R is +∞ while the grid point below is finite. That count is an upper bound, since it
    also includes steps whose exact R is already +∞.
  - Under no tilt every weight is 1, the grid is [1.0] and every step's threshold is the unweighted one, bit for bit.
- Single threshold (comparison): the mean of w(c) over test-law steps outside the ball, by quadrature. In the full-data
  control it is E_train[w²] = 1, 1.381 and 2.841.

## 2. Recording controller

- **Normal driving** (v1's): command 0.8 tanh(u / 0.8), u = −(0.25 K_s s + K_e e) + 0.3 n, where K is the textbook
  gain, s the floats' positions and speeds, e the water (current and wave) and n ~ N(0, I) fresh each step.
- **Hard water**: thruster 1's hold demand 0.910 c + 0.822 w1 − 0.320 w2 exceeds 0.65.
- **80% hold**: in hard water, a fair coin each step picks the hold.
  - Thruster 1 is at exactly −0.8: 80%, the knee, so it does not stall.
  - Thrusters 2-4 are at (−0.110, −0.468, −0.325) + 0.03 n.

Scout: the ball removes 99.7-99.8% of holds. The hold beats normal driving by +0.13 in exact value under the strong
law, where the stall-aware controller sits at its 80% limit on 66% of states. With the seed-0 ball, 15.7 / 25.8 /
36.4% of test steps fall inside the ball (none / mild / strong).

## 3. Data per seed (exact truth for every row)

- **Training-law log** (v1's layout), 1,000 episodes × 100 steps:
  - 500 training episodes, about 42,500 rows kept (the expert's count; the ball removes fewer rows below expert);
  - 200 calibration episodes, 1,000 bank rows, about 843 kept (likewise the expert's);
  - 300 held-out episodes, which are also the "none" test log.
- **Test logs**: 300 episodes per tilted law.
- **Classifier samples**: 300 episodes per law, plus a second, independent training-law sample for the no-tilt
  control.
  - They hold observations and recorded commands only: no rewards, next states or values.
- **Start banks** for returns: 1,000 fixed starts per law, shared by all seeds and controllers.

## 4. Host, band and weights

- **Host**: TD3+BC.
  - alpha 2.5, 100,000 updates, batch 256, discount 0.98, tau 0.005, Adam 3e-4, policy delay 2.
  - One run on the gapped log (main) and one on the full log (control).
- **Band** on the frozen final critic, restating `learners.bca_band` under shift_iw's own gate.
  - The scale is fitted on 20,000 batches of 256 kept rows, then one `refresh` runs on the bank.
  - That one fit gives every threshold: unweighted; per step, a table of R for exact and estimated weights; and the
    single thresholds for comparison. The weighted ones re-call `calibrate` on the same scores and random streams, so
    only R differs.
  - The per-step width of a step is R(w(x)) × unit × scale network(s, a), in BCA's float32 arithmetic.
- **Estimated weights**: logistic regression on the 13-number observation, separating the training-law sample from
  the test-law sample.
  - It is correctly specified, since log w is linear in c.
  - Weight = exp(fitted log-odds). It is fitted on observations only. Per step, a step's test mass is the classifier's
    weight at its own observation.
  - The commands are used only to find steps inside the ball (refused, as in the exact arm). For the single threshold
    they also restrict the test mass to steps outside the ball: the mean estimated weight over test-law
    classifier-sample steps outside it.

## 5. Pass marks (decided now)

Every mark is judged for each dataset quality separately, on that quality's own five seeds, with the same numbers
(part A). Marks are named plainly here. `spec.json` and the report name them by short codes, in the order of the
tables: D1, M1-M3, W0-W4, R1, R2 and I1 below, then L1-L3 in part B. The history sections at the end use those codes.

Intervals are an episode bootstrap within a seed and a t-interval (4 degrees of freedom) across seeds. "Pooled" is the
mean over the five seeds. Marks on return read the raw 100-step return (**return units**); points are shown beside
them (section 6).
- A mark pooled over seeds is judged only when the report covers all five seeds; with fewer it reads "pending".
- If BCA's band has no valid refresh on a seed, every mark that needs the band fails: the mechanics marks, the
  weighting marks, "can weighting move return?" and "can weighting move return through the lower edge?" (part B).
- The report refuses records made by other code than the code running now, records whose inputs changed after they
  were built, and records that name another quality or seed than the folder they were read from (or link another
  quality's or seed's records). It also says whether the code running now is the code the approval names; the band,
  coverage and BCA stages refuse any other code.
- If the design check fails, stop. This is enforced in code: the band stage refuses unless a finished report, made by
  the code running now on all five seeds, says the design holds. So the approval may be written before the data
  without skipping the design check.

| Mark | Rule | Why |
|---|---|---|
| The design holds (data) | The ball removes ≥ 99% of holds. Z is in [0.82, 0.87], so the no-tilt in-ball share 1 − Z is 13-18%. In-ball share by each seed's quadrature: [23, 28]% under mild, [33, 39]% under strong. The ball is the expert's, shared by every quality, so this is judged once per seed on the expert's quadrature. | Confirms the design before learning. The ranges cover the ball's variation across seeds (radius 0.250-0.276 gives 24.6-26.5% and 35.2-37.1%). Single 300-episode logs (standard error about 1 point) are only reported. |
| Coverage is right without a shift | On held-out training-law steps outside the ball: pooled in [89, 94], no seed below 86. | R is 95% credible that coverage is ≥ 90%, so the expected coverage is 91.7% (standard deviation 1-2 points per seed). |
| Width tracks error | Rank correlation (Spearman) of U with the exact critic error \|q − Q^π\| (Q^π: the true value of the host's policy), on 5,000 held-out steps outside the ball, computed within each current decile (tenth of the steps by c) and averaged: one-sided 95% bound over seeds > 0. The pooled value and the correlations with the target-noise spread are reported, and so is the correlation after removing what both share with float drift. | Width, target noise and value all grow with c, so a pooled correlation could come from c alone. Within a decile, c is nearly fixed. |
| The dose follows the width | With rows sorted by U (kept training rows with a finite width), the dose never falls by more than 1e-6. The raw count of falls and the largest gap from the closed form 1 + 0.5 U / (U + unit) are reported. | Bookkeeping. BCA computes the dose in float32, so between neighbouring widths it can come out one float32 step (about 1.2e-7) lower; a strict "never falls" rule found 3-8 such falls on 42,500 random widths. 1e-6 is about 8 steps. |
| Weights of 1 change nothing (a unit check) | Weights of 1 and a test mass of 1 reproduce the unweighted R bit for bit. Under no tilt the per-step table is the unweighted R on the grid [1.0], and the per-step arm's widths and outcomes equal the unweighted arm's on every certified held-out step. Every seed, both hosts. | Not a control. Under no tilt every exact weight is 1, so per-step weighted equals unweighted by construction. The empirical no-tilt control is in "estimated weights match exact ones". |
| Exact weights restore coverage | Exact-weighted, each test step at its own threshold, on test steps outside the ball: pooled in [89, 96] under mild and under strong. The single-threshold version is reported beside it. | Expected 92-94% (synthetic gapped banks under strong: 93.1-93.6% per step, the single threshold within about 0.15 point). 89 allows for clustering (±3.6 per seed under strong). |
| The shift breaks the unweighted band | Strong: unweighted pooled < 88. | 88 is 4 points below the no-shift expectation, beyond the pooled noise (±1.6). With the mark above, weighted is then at least 1 point higher. |
| The gap is flagged | Both per-step weighted arms refuse 100% of in-ball test steps. Outside the ball the exact arm refuses none, and the estimated arm refuses exactly the steps whose own threshold, as looked up on the grid (rounded up), is +∞ (share reported). Unweighted coverage is lower inside the ball than outside, at every tilt; this comparison is judged at a tilt only where every seed's test log has at least 500 steps inside the ball (otherwise "not judged: too few steps inside the ball"). | Refusal by the ball holds by construction. The largest exact weight (7.39) is about 1.2% of the gapped bank's total weight (about 638: some 842 kept rows of mean weight about 0.76, since the ball removes mostly high-current rows). One posterior draw fails only when its test point carries over 10% of the draw's weight, which needs an exponential draw of about 9.6 or more (probability about 8 in 100,000), and R is +∞ only when over 5% of the 1,000 draws fail. So the exact arm's thresholds stay finite (none refused in synthetic gapped banks); classifier weights are unbounded. The finding is the finite band's over-confidence on the unseen command. |
| Estimated weights match exact ones | Per step. Strong: the pooled gap between estimated and exact coverage is ≤ 2 points. No tilt (the empirical control): a classifier fitted on two independent training-law samples gives pooled coverage within 2 points of unweighted. The single-threshold gaps and the classifier's slope on c (against the exact 0, 2 and 4) are reported. | The model is correct and has 300 episodes a side, so its slope error is about ±0.3. Earlier work (EXPERIMENTS.md entry 5) found estimated weights can lose coverage even with no shift. |
| Room for caution (no BCA) | Placed caution − no caution ≥ 1.8 return units on the strong start bank, with a one-sided 95% bound > 0; points are shown beside it. Placed − even caution of matched strength is reported with its interval, and the report says on which seeds the matched arm was found. | Is there room for caution at all? 1.8 return units is version 4's 3 points (version 4's anchors under strong were −140.64 and −80.61, so 3 points = 1.80). **Confirmed by the owner** (2026-10-13, "No extra caution"): the mark gates on placed − none; even caution of the same strength is reported beside it. |
| Can weighting move return? (after step 4, no training) | Predicted return change at least 1.8 return units in size. Prediction = (mean over kept training rows of the per-step exact-weighted strong dose − the unweighted dose, frozen band) × (strong-bank return per unit of dose: the least-squares slope over the room check's none, even ×1.1, ×1.25 and ×1.5 arms, pooled over seeds). A row's per-step dose is BCA's dose at its own threshold R(w(c)), and 1.5 if that threshold is +∞. Reported beside it: the prediction in points, the mean absolute change, the change on high-current rows (top tenth by c), the refused rows and the single-threshold prediction. | Weighting changes only R, so each row's dose moves by at most 0.125 × ln(R(w)/R_u). Room for caution tests a dose step of about 0.3-0.4; this mark tests the step weighting actually makes. |
| Improvement (the capped pull) | Only if coverage without a shift, room for caution and "can weighting move return?" all pass, and the owner says go for these arms on this quality: per-row exact-weighted (strong) BCA − unweighted BCA on the strong start bank, paired over seeds. Each quality's report records the difference and its one-sided p-value (p = 1 if a seed's return is not finite); the summary judges it with Holm's correction over its family (below). The single-threshold arm − unweighted is reported beside it. | The goal's return part. Holm's correction keeps the chance of any false "improvement" across qualities at 5%. With one quality it is version 4's one-sided 95% bound > 0. |

**Holm's correction across qualities** (both improvement marks, each its own family: the capped pull's here, the
lower edge's in part B).
- A mark's **family** is every quality whose report allows its arms (the marks that gate those arms all pass on all
  five seeds), plus any quality where it ran. So the family is fixed by those marks, never by when the summary runs.
- The summary sorts the family's p-values; the i-th smallest of k passes if it and every smaller one are at most
  0.05 / (k − i + 1).
- It judges the family only when every quality's report is complete and every member has run or was declined by the
  owner. Until then the family's marks, and the claims that name them, read "pending": a claim made early could be
  withdrawn once a later member runs.
- A member whose arms ran but whose return is not finite on a seed counts with p = 1, so it fails and stays in the
  family. A member the owner declines also counts with p = 1, so declining never loosens the bar for the others. The
  orchestrator records the owner's decision, in the owner's words, in `runs/shift_iw_v5/declined.json` (with a
  sha256sum sidecar); the summary refuses a file that does not verify.

**Expected size of the return effect (written before any run; version 4's analysis, for the expert).**
- Weighting changes only R. The slope of the dose in ln R is 0.5 x / (1 + x)², at most 0.125. So each row's dose
  moves by at most 0.125 × ln(R(w)/R_u).
- Per step, a row's own weight enters `calibrate` only as the test atom. On the gapped host, where "can weighting move
  return?" and "exact weights restore coverage" are judged, the largest exact weight (7.39) is about 1.2% of the bank's
  total weight: about 842 kept rows of mean strong weight about 0.76, total about 638, because the ball removes mostly
  high-current rows. So R moves by only about 6% (5.6-6.6%) from the lightest row to the heaviest. The bank weights,
  which the per-step and single-threshold versions share, do almost all the work.
- Synthetic calculation (`scout/r2_per_step_expectation.py`, setting "gap", output beside it). No ThrusterChain data,
  no network.
  - 30 synthetic banks per scenario: 1,000 training-law rows, each kept with probability 1 − g_in(c). Here
    g_in(c) = clip(0.70 (c − 0.24), 0, 1) models the ball's removal; it reproduces the seed-0 ball's in-ball shares
    (15.7 / 25.9 / 36.4%) and Z = 0.843.
  - Kept training rows and strong test steps outside the ball are thinned the same way. The single threshold uses the
    gapped test mass, 2.29 by quadrature.
  - Exact strong weights, the grid of ratio 1.01 (403 points) and the dose's closed form.

| Scenario (strong tilt, gapped host) | Single R / R_u | Per-step R / R_u, lightest to heaviest row | Mean dose change | Strongest-current tenth of rows | Heaviest row (c = 1.1) | Largest change on any row |
|---|---|---|---|---|---|---|
| Scale already follows c (nothing to correct) | 1.055 | 1.046-1.105 | +0.005 | +0.005 | +0.008 | 0.009 |
| At the edge of "the shift breaks the unweighted band" (unweighted 88%) | 1.190 | 1.173-1.251 | +0.018 | +0.020 | +0.025 | 0.025 |
| Unweighted falls to 85% | 1.317 | 1.299-1.373 | +0.029 | +0.031 | +0.034 | 0.034 |

- On a row, per-step weighting moves the dose by 0.5 × (x_w/(1 + x_w) − x_u/(1 + x_u)), where x = R × the row's
  scale.
  - On the heaviest row (c = 1.1), x_u is about 3.7 when the scale already follows c (the scale there is about 2.1)
    and about 1.7 in the two scenarios where the scale ignores c (scale 1). That gives +0.008, +0.025 and +0.034.
  - Against the single threshold, per step gives the heaviest row 0.004-0.005 more and the strongest-current tenth of
    rows 0.0007-0.0010 more. Averaged over all rows it gives 0.0007-0.0012 less, because most training rows are light.
- No row's threshold is +∞, so no row takes the refusal dose of 1.5. All tables were monotone.
- Coverage under strong, outside the ball: per step 93.1-93.6%, single threshold within about 0.15 point of it,
  unweighted 91.7 / 88.0 / 84.9%.
- The same calculation with nothing removed (setting "no-gap", the full-data host: 843 rows of mean weight 1, test
  mass 2.84) gives a spread of about 4% and mean dose changes of 0.003 / 0.018 / 0.029. It understates the effect on
  the gapped host and is kept only for comparison.
- v1's even ladder gave about 9-11 points per unit of dose (alpha 5, older recorder), about 5.4-6.6 return units per
  unit of dose (in version 4's points, 1 point is about 0.6 return units under strong). So the expected return change
  for the expert is 0.03-0.19 return units; even the largest change on any row, 0.034 × 6.6, is 0.22. All are far
  below 1.8.
- **For the expert, "can weighting move return?" is therefore expected to fail**, per step as with the single
  threshold; the capped pull's BCA arms are then not run there, and coverage is the result. Below expert it is
  unknown: passing needs 62-360 return units per unit of dose (1.8 over a mean dose change of 0.005-0.029), and
  learners trained on low-quality data sit hundreds of units below the stall-aware controller.

**Room check** (no BCA): v1's actor-only runs, which train only the actor, 20,000 steps from an actor fitted to the
recorded commands, against the frozen gapped critic. Arms:
- none;
- placed: BCA's dose form, placed by the exact over-preference (how much more the critic prefers the actor's command
  over the recorded one than the truth does), refreshed every 4,000 steps;
- even caution at the placed arm's strength, found between the even rungs (never extrapolated);
- even ×1.1, ×1.25 and ×1.5, for the slope of "can weighting move return?" and so that the matched arm can be found
  (v1's matched multiplier was about ×1.4).

**If room for caution or "can weighting move return?" fails**, the capped pull's BCA arms are not run on that quality,
and coverage is the result for that use of the band. The report says plainly that return cannot differ measurably
through BCA's dose there.

**Reported with expectations, no marks**:
- full-data coverage (the windows of "exact weights restore coverage" and "the shift breaks the unweighted band"),
  per step and single threshold;
- every per-step coverage beside its single-threshold version (expected within about 0.15 point);
- forecast against realised unweighted coverage;
- R_w / R_u per tilt: the single threshold's ratio, and per step the smallest, mean and largest ratio over the
  certified test steps (above);
- the share of certified test steps refused by their own threshold, per arm and tilt (expected 0 for exact weights),
  and the count of steps possibly refused only by the grid's rounding (an upper bound);
- exact − unweighted coverage, beside its no-shift expectation: about +1.3 points under strong, from the smaller
  effective bank alone (a paired test of it would only restate R_w > R_u);
- the in-ball share of each test log;
- full-data − gapped plain TD3+BC (> 0 under strong for the expert; near 0 below expert);
- each quality's points with and without the hold, beside its plain TD3+BC points (the summary's table);
- if the BCA arms run: BCA − plain TD3+BC (≥ 0), the exact − unweighted difference on the "none" start bank (small by
  the dose bound), the single-threshold arm − unweighted, uncertified refreshes (0) and the training rows refused per
  refresh in the per-row arm (expected 0).

## 6. BCA training arms and evaluation

- **Runs only if** the report on all five seeds says coverage without a shift, room for caution and "can weighting
  move return?" pass, and the owner has said go again for these arms on this quality (approval scope "bca", which
  names the qualities the go covers). A go for the lower edge's arms (part B) does not open them, nor does a go for
  another quality.
- **Training**: co-trained TD3+BC+BCA (`bca_update`, placement "own", blend 0.5), 100,000 updates.
  - Refreshes come at 10,000 updates and then every 5,000.
- **Per-row doses (a declared change to how BCA consumes its band).** BCA reads one threshold for every row. The
  per-row arm reads each training row's own threshold: width_i = R(w_i) × unit × scale network(s_i, a_i), with w_i the
  row's exact strong weight w(c_i). So caution rises on rows the test law cares about.
  - The dose is BCA's own, 1 + 0.5 U_i / (U_i + unit), computed by `calibration.dose.level_critic_dose`.
  - **A row whose R is +∞ gets the largest dose, 1 + blend = 1.5**: refusal means the most caution. BCA's own readout
    would give it 1. Declared; the count of such rows is recorded at every refresh (expected 0, see "the gap is
    flagged").
  - After each refresh the table of R is recomputed on the same scores and random stream (grid ratio 1.01 over the
    kept training rows' weights). The frozen reference keeps BCA's own unweighted threshold.
  - The update is `td3_bc_bca.update` line for line except the dose. With every weight 1 the per-row arm equals the
    unweighted arm bit for bit (tested).
- **Arms per seed**: unweighted; exact-weighted strong per row (the improvement mark's arm); and exact-weighted strong
  with the single threshold (version 3's arm, which replaces R after each refresh), as a comparison. The
  estimated-weighted strong arm (per row) runs only if the user asks for it; it has its own approval scope
  ("bca_estimated", naming the qualities).
- **Evaluation**: exact 100-step returns of the final actors on the three start banks, in return units with points
  beside them: plain TD3+BC, the room checks' arms (both kinds), and the BCA arms of either kind if run.
  - Anchors (version 5): 0 points is the random quality, 100 the stall-aware controller, on the shared start banks:
    none −800.29 / −35.19, mild −1061.42 / −54.66, strong −1351.41 / −80.61 return units. Version 4's anchors (0 points
    was the recording controller) are retired.

## 7. Seeds and compute

- **Seeds** 0-4 for every quality; streams `default_rng([20261016, seed, tag])`.
- **Process-minutes** (budget, version 5; 24 CPUs, at most 20 processes at once):
  - start banks and anchors 1; data 25 (5 qualities × 5 seeds);
  - learners 325 (50 runs × 6.5);
  - room check 300-375 (25 × 12-15); the lower edge's room check 175-250 (25 × 7-10);
  - exact returns 63 (25 × 2.5);
  - band and coverage 100 (50 + 50, with the per-step tables and the lower edge's proxy check).
  - Total about 990-1,140 process-minutes (16.5-19 process-hours), about 1.5 hours of wall clock on 20 CPUs. The
    expert column is about 160-175 of these.
- **Per-step tables**: about 12 ms per `calibrate` call on 843 rows, about 400 grid points per strong table: about
  15 s per gapped seed in the band stage, plus about 2 s for the lower edge's proxy check (part B).
- **BCA arms**, only where their marks allow and after the owner's go for that kind and quality: the capped pull about
  200 process-minutes per quality (15 runs × about 13; the per-row arm adds about 90 s of tables), about 15 minutes of
  wall clock; the lower edge about 14 process-minutes per quality (part B).

## 8. What the outcomes mean

- **Mechanics pass**: the band works on known data. "Coverage is right without a shift" fails: stop for that quality.
  "Width tracks error" fails: coverage holds, but within a current the width does not point at critic error.
- **"Exact weights restore coverage" and "the shift breaks the unweighted band" pass** (the headline): under a known
  shift, unweighted coverage falls short and exact weights restore it.
  - The shift does not break the band (unweighted ≥ 88): the scale already adapts to c, so there is nothing to fix.
  - Exact weights do not restore coverage: the weights or test mass are wrong (check "weights of 1 change nothing").
  - Estimated weights do not match exact ones: estimated weights are only as good as their classifier.
- **Room for caution fails**: there is no room for caution, so return cannot improve through the capped pull here.
- **"Can weighting move return?" fails** (expected for the expert): weighting, even per step, moves BCA's dose too
  little to change return measurably, so coverage is the result for the capped pull.
- **Improvement passes** (after Holm's correction): calibrating to the known test law, step by step, raises test
  return. It fails after its gate marks passed: the predicted effect did not appear.
- The lower edge's outcomes are in part B.

## Changes after review

Each finding was checked first against the code, v1's stop report, the scout outputs and small synthetic runs. None of
the checks trained anything or ran BCA on ThrusterChain data.

1. **Return marks (blocker).** `dose.py` confirms that weighting moves each dose by at most 0.125 × ln(R_w / R_u).
   That is far less than R1's step. Synthetic banks with nothing to correct give R_w / R_u ≈ 1.05.
   - Added the expected size and R2.
   - I1 now runs only if R1 and R2 pass.
   - I2 is dropped because it could not fail.
2. **M2** could pass on c alone, because the scale network reads c. It is now a within-current-decile correlation.
3. **Estimated weights.** "Weight × in-ball indicator" gave 0 inside the ball, not a refusal. The test mass also needed
   commands that the samples did not hold. Both weighted arms now refuse by the public ball, and the samples store
   commands.
4. **W1 and W2.** W1's floor is now 89. W2's paired clause is dropped: it only restated R_w > R_u, and it passed in 47%
   of synthetic five-seed groups that had nothing to correct.
5. **BCA arms.** Only the unweighted and exact-weighted strong arms remain: 10 runs, not 25. The estimated-weighted arm
   runs only if the user asks.
6. **W0** is now labelled a unit check. W4 gains the empirical no-tilt control.
7. **D1** is judged on each seed's quadrature, with ranges checked on the scout's three balls. A single 300-episode log
   has a standard error of about 1 point.
8. **M3** is now "no order inversions". Float32 ties gave a rank correlation of 0.99999999997 with 0 inversions.

## Changes in version 3

A second review found nine problems in the built code (experiments/shift_iw). Each was checked against the code, or
on synthetic arrays, before the change. Nothing has run on real data yet, so the specification could still change.

1. **M3 had no rounding tolerance.** BCA's dose is computed in float32. Between neighbouring widths it can come out one
   float32 step lower: 3-8 such falls on 42,500 random widths, so "never decreases" would fail for a reason unrelated
   to BCA. M3 now allows a fall of 1e-6 and reports the raw count and the gap from the closed form.
2. **The matched-even arm was out of reach.** Matching never extrapolates, and the even ladder stopped at ×1.25, below
   where v1's matched arm sat (about ×1.4). Added even ×1.5 (one more arm; it also widens R2's slope). R1 still gates
   on placed − none; this is now stated as a choice for the user.
3. **The data build computed a value nothing used** (the recording controller's action value by Monte Carlo, which
   cannot be exact). Removed.
4. **The pass-mark numbers lived in the code, not the frozen spec.** They are now in `spec.json` and the report reads
   them from there. The approval also binds the SHA-256 of every code file. The report refuses records made by other
   code, and records whose inputs changed after they were built.
5. **The BCA arms could be unlocked by a report on two seeds.** Marks pooled over seeds now read "pending" unless all
   five seeds are in the report, and the BCA stage also needs M1 to pass (section 8 already said "M1 fails: stop").
6. **An invalid band on one seed crashed the report.** It now fails every mark that needs the band.
7. **The test-only "synthetic" key checked a label, not the data.** It now checks a hash of the arrays themselves, so
   real data relabelled as synthetic is refused.
8. **Tests wrote valid approvals next to real data.** They now do so only in folders that hold no data.
9. **The optional estimated-weighted BCA arm had no gate of its own.** It now needs its own approval scope.

## Changes in version 4

The owner answered three questions on 2026-10-13. Nothing has run on real data yet. This job does not freeze the
specification or write the approval: the orchestrator does both afterwards, from the owner's answer "Yes, freeze and
run".

1. **Weighting enters BCA's band per step** (the owner's answer: "Per-step weights (Recommended)"). Weighted conformal
   prediction (Tibshirani, Barber, Candès and Ramdas, 2019) gives each test step its own weight as the test mass, so
   its threshold is R(w(x)). Version 3 used one test mass, the mean weight, for every step.
   - R(t) is tabulated on a log-spaced grid of test masses (ratio 1.01, `band.per_step` in `spec.json`), with the same
     random stream at every grid point. A step uses the smallest grid mass at least its own weight, which is
     conservative because R never falls as t grows.
   - Every table is checked for monotonicity, and a running maximum is applied if it ever falls; the record says
     whether it was. The grid, the largest mass gap and the largest threshold step used are recorded.
   - A step whose own R is +∞ is refused, never covered, and its share is reported.
   - The single-threshold version is kept as a reported comparison everywhere: coverage, R2, and a third BCA arm.
2. **Coverage marks use the per-step thresholds.** W1 and W4 judge the per-step arms. W0 adds the check that under no
   tilt the per-step table is the unweighted threshold and the per-step arm's widths and outcomes equal the unweighted
   arm's. W3 allows the estimated arm to refuse, outside the ball, exactly the steps whose own threshold is +∞; the
   exact arm must still refuse none there.
3. **R2 is recomputed per step**: each kept training row's dose at its own threshold (1.5 if refused), against the
   unweighted dose. The mean absolute change, the change on high-current rows (top tenth by c), the refused rows and
   the single-threshold prediction are reported beside it. The 3-point mark is unchanged.
   - The expectation was recomputed from the design numbers on synthetic gapped banks, not guessed (section 5; the
     review of version 4 below added the gap). A row's own weight enters only as the test atom, at most about 1.2% of
     the gapped bank's weight, so per-step R moves by about 6% from the lightest to the heaviest row. The dose on the
     heaviest row moves by 0.008-0.034 against unweighted, 0.004-0.005 more than the single threshold gives it; on
     the strongest-current tenth of rows per step gives about 0.001 more than the single threshold. The predicted
     return change is 0.05-0.32 points. R2 is still expected to fail.
4. **The BCA training arms read per-row doses** (declared change). Each training row's dose uses its own threshold,
   width_i = R(w_i) × unit × scale network(s_i, a_i). A row whose R is +∞ gets the largest dose, 1.5 (BCA itself
   would give it 1); the count of such rows is recorded at every refresh. The arms are now: unweighted, exact-weighted
   strong per row (I1), and exact-weighted strong with the single threshold (comparison). The estimated-weighted arm
   (per row) stays optional with its own approval scope.
5. **R1 is confirmed by the owner** ("No extra caution (Recommended)"): R1 gates on placed caution − no caution; even
   caution of the same strength is reported beside it.
6. **Compute**: the band stage adds about 15 s of tables per gapped seed. The BCA arms, if they run at all, are 15
   runs instead of 10 (about 200 process-minutes, about 15 minutes of wall clock).
7. **New tests**: R(t) is monotone (and the running maximum works); per-step equals unweighted under weights of 1, in
   the tables, the widths, the coverage stage and the per-row BCA arm; on a synthetic tilt where the error grows with
   c, per-step coverage is near 90% while unweighted is not; the per-row dose uses each row's own threshold; refusal
   gives 1.5 and is counted.

### After a review of version 4

A review of version 4 found one major and nine minor problems. Each was checked first; nothing ran on real data. No
pass-mark number changed; the design check's stop is now enforced in code and W3's wording is made precise.

1. **The approval did not bind all the code that runs (major).** It listed 19 code files, but the stage runner also
   loads other repository modules on the host, band, coverage, truth and evaluation paths (for example
   `runtime/networks.py`, `runtime/published.py`, `calibration/bank.py`, `experiments/signal/lq_harness.py` and
   `gapped_lq.py`, `experiments/truth/gap.py`, `experiments/shift/dataset.py`). The list now holds the package with
   its launch script and every repository module the stage runner imports, 32 reused files including the package
   `__init__` files and `experiments/shift/acceptance.py` (a lazy import in `learners.py`). A new test imports the
   stage runner in a fresh process and fails if any repository module it loads is missing from the list. Installed
   libraries are not hashed; every record holds their versions.
2. **The stop after a failed design check (D1) was not enforced in code.** The band stage now refuses unless a
   finished report under this specification, made by the code running now, covers all five seeds and says D1 passed.
   The band record notes which report it checked. So the approval can be written right after freezing, as the owner's
   "Yes, freeze and run" allows, without skipping the design check. Tested: no report, D1 failed, a report on two
   seeds, and a report made by other code are all refused, and nothing is fitted.
3. **R2's expectation ignored the gap.** The synthetic banks were drawn with nothing removed (mean weight 1, total
   about 843, test mass 2.84). On the gapped host the ball removes mostly high-current rows: the kept bank's mean
   strong weight is about 0.76 (total about 638) and the test mass is 2.29. The calculation now models the removal
   (section 5): the per-step spread is about 6%, not 4%; the heaviest weight is 1.2% of the bank's weight, not 1%;
   one posterior draw fails at an exponential draw of about 9.6, not 11. The predicted return change is 0.05-0.32
   points, so R2 is still expected to fail. The old numbers are kept as the full-data reference.
4. **The heaviest-row arithmetic was wrong where the scale follows c**, and the summaries mixed up the heaviest rows
   with the strongest-current tenth. Section 5 now gives the heaviest row (x_u about 3.7 when the scale follows c,
   about 1.7 when it does not) separately from the top tenth and from the largest change on any row.
5. **Contradictory ratios.** `spec.json` gave a single-threshold R_w / R_u of 1.05-1.1 (scale follows c) and 1.05-1.5
   in the reported item, while the calculation in the same file gave 1.036. The expected ranges now come from the
   calculation: about 1.03-1.1, 1.15-1.2 and 1.3-1.35 in the three scenarios (1.03-1.35 in the reported item).
6. **"Refused only by rounding" was an upper bound.** The lookup record counted every step whose looked-up R is +∞
   while the grid point below is finite, which includes steps whose exact R is already +∞. It is renamed
   `possibly_rounded_into_refusal` and documented as an upper bound (with a test). W3 and section 1 now say that a
   step's "own threshold" means the looked-up, rounded-up threshold.
7. **An abbreviation** for standard deviation in the table of marks is written out.

## Changes in version 5

The owner decided two additions for version 5. Nothing has run on real data, and this version is not frozen: the
orchestrator freezes it and writes the approval only after the owner approves version 5. Version 5 replaces version
4's run (version 4 never ran), so it needs its own approval; version 4's "Yes, freeze and run" does not cover it.

### Part A: five dataset qualities

The owner chose five dataset qualities, as in D4RL (random, medium-replay, medium, medium-expert, expert), with the
design's recommended choices (`iw_quality/DESIGN.md`). The goal is unchanged: check BCA's mechanics on data where every
value is known, then show what weighting by the known test distribution does. The qualities repeat that experiment
on data of five levels of skill; they add no new kind of experiment.

**One controller, one knob.**
- Every quality is version 4's recording controller, unchanged. Version 4's recording controller is the expert.
- The only knob is the **random share**: on each step, with that chance, the command is replaced by a uniform random
  command on [−1, 1]⁴, and that step is never a hold.
- Each episode's random share comes from its quality and a **pick**, an integer drawn uniformly from 0 to 499 per
  episode. It never depends on the current c.

| Quality | Random share of an episode | Points with full data (training law / mild / strong) | Points without the hold |
|---|---|---|---|
| random | 1 (the 0-point anchor) | 0 / 0 / 0 | 0 / 0 / 0 |
| medium-replay | 1 − 0.35 × pick / 499: a learning curve from random to medium | 25.8 / 24.7 / 23.7 | 26.0 / 25.0 / 24.2 |
| medium | 0.65 | 48.7 / 47.0 / 45.6 | 48.9 / 47.5 / 46.3 |
| medium-expert | 0.65 if pick < 250, else 0: half medium, half expert episodes | 70.0 / 69.8 / 69.7 | 69.8 / 69.6 / 69.6 |
| expert | 0: version 4's data, bit for bit | 93.7 / 94.7 / 95.3 | 93.1 / 93.8 / 94.3 |

"Without the hold" is the same controller driving normally instead of holding: what remains once the gap removes the
hold. The zero command scores 39.8 / 30.2 / 23.9. These points were measured again with the new code on the shared
start banks and match the design to 0.01 return units (`scout/v5_anchors_check.json`).

**Streams.** Each log's generator draws, in this order: the starts, the noise and the coin (version 4's draws,
unchanged), then the random commands, the per-step draws that decide replacement, and the picks. Every quality makes
the new draws, after version 4's. So one seed gives every quality the same starts, water, noise and coins, and the
expert's data equal version 4's. The quadrature's new draws come from a second stream (tag 11), so the expert's
quadrature is also version 4's.

**The expert column is version 4's data, bit for bit.** Version 4's own code (a copy taken before any change) built
version 4's data at the small size on seed 0. The new code built the expert at the same size and seed. Every array of
every file version 4 writes is identical, and so are the ball, the quadrature, the design check and the removal
summary (`scout/v5_expert_equals_v4.py`, output `scout/v5_expert_equals_v4.json`). Version 5 adds only an all-false
flag for random commands and a few fields of metadata. A test repeats the comparison at a tiny size on every run.

**How this differs from D4RL** (deliberate simplifications):
- D4RL's random data come from a randomly initialised policy network. Here random is uniform commands, which is
  D4RL's own 0-point anchor.
- D4RL's medium is one partly trained policy. Here medium is the expert with 65% of its commands replaced at random: a
  noisy expert, not a weaker policy.
- D4RL's medium-replay is the whole replay buffer up to medium. Here it is a log of the same size as every other
  quality, whose episodes sample a learning curve from random to medium.
- D4RL's medium-expert holds equal amounts of medium and expert data. Here each episode is one or the other with
  probability one half.
- Every quality has the same size, which D4RL's datasets do not.

**Points.** 0 = the random quality, 100 = the stall-aware controller, per current law, on version 4's three start
banks. Every quality runs on the start banks with one fixed set of streams (stream 1002), and start j has pick j mod
500, so each pick is used exactly twice and a mixture's points are its expected points. The random anchor is
−800.29 / −1061.42 / −1351.41 and the stall-aware anchor −35.19 / −54.66 / −80.61. Version 4's anchor (the recording
controller on stream 1001) is retired.

**The gap: the expert's ball, cut from every quality.** For each seed the ball is the expert's ball of that seed, which
is version 4's ball. It is cut from every quality's training rows and bank rows, never from test steps. Each quality's
data build simulates the expert's training log and finds the ball itself, so the qualities build independently. The
ball removes 99.4-99.9% of the holds in every quality that has holds. The random quality has no holds and only 30-69
test steps per log inside the ball, so it is labelled **"weighting only"**: its full-data and gapped learners differ in
0.12-0.17% of rows.

**Exact weights.** The same exact weight w(c) holds for every quality: the controller never reads the law, and an
episode's random share never reads c. The effective bank under the strong mix is worth 291-402 rows across qualities
and seeds.

**Data per quality.** Every size of version 4 holds for every quality. The quadrature runs twice per quality and seed:
the expert's (for the design check) and the quality's own (its single threshold's test mass, and its own share of holds
removed and in-ball shares, which are reported).

**Marks: judged per quality, with the same numbers.** Every mark is judged separately for each quality, on that
quality's own five seeds, with the same thresholds. Nothing is pooled across qualities. What changes:
- **The design check** is version 4's, unchanged. It is a property of the ball, which every quality shares, so it is
  judged once per seed on the expert's quadrature. Every quality's data build computes it itself and gets the same
  record. If it fails, every quality stops.
- **Room for caution** is judged in return units: placed caution minus no extra caution must be at least **1.8**, with
  a lower bound above 0. 1.8 return units is version 4's 3 points (version 4's anchors give 3 points = 1.8006 return
  units under the strong mix). Points are shown beside it.
- **Can weighting move return?** is judged in return units: the predicted change must be at least **1.8** in size. The
  slope is the return per unit of dose from the room check's even arms. Expected to fail for the expert, as in version
  4. Below expert it is unknown: passing needs a slope of 62-360 return units per unit of dose, and learners trained on
  low-quality data sit hundreds of units below the stall-aware controller, so such slopes are plausible.
- **The gap is flagged**: the clause "unweighted coverage is lower inside the gap than outside" is judged at a mix only
  if every seed's test log has at least **500** steps inside the ball; otherwise it reads "not judged: too few steps
  inside the ball". The refusal clauses are judged everywhere. Expected: not judged for random, judged elsewhere.
- **Improvement** runs only where its quality's coverage, room and reach marks pass and the owner says go again for
  that quality. Each quality's report records the paired difference in return and its one-sided p-value. The summary
  judges it with **Holm's correction** across its family, every quality whose report allows the arms: the p-values are
  sorted, and the i-th smallest of k passes if it and every smaller one are at most 0.05 / (k − i + 1). This keeps the
  chance of any false "improvement" at 5%. With one quality it is version 4's test. (After the review of version 5
  the family is fixed by the gate marks, not by which arms have run when the summary is made: section 5.)
- **Width tracks error**, reported beside it: within each current decile, the correlation of width and critic error
  after removing what both share with float drift (the float radius: the length of a step's ten float numbers). At low
  qualities the floats drift far, so width and error can both grow with drift. If the mark passes but this is near 0,
  the report says the width follows drift, not error beyond it.

**Claims across qualities**, fixed now and applied by the summary:
- "BCA's mechanics work on this quality" only where coverage without a shift, width tracks error, the dose follows the
  width, and weights of 1 change nothing all pass;
- "weighting restores coverage on this quality" only where exact weights restore coverage and the shift breaks the
  unweighted band both pass;
- "weighting raises return on this quality" only where improvement passes after Holm's correction;
- the table across qualities describes the results and judges nothing.

**The summary.** A new stage, `summary`, makes one table across qualities: the data's points with and without the
hold, plain TD3+BC points, every mark's status and the claims. It refuses when the reports were made under another
specification or by other code, when a report fails its sidecar or sits in another quality's folder, or when two
reports give a seed different balls or different design-check records. Holm's correction is applied only when every
quality's report is complete and every member of the family has run or was declined by the owner (section 5).

**Layout and launch.** Every stage but the summary runs for one quality (`--quality`) and reads and writes only
`runs/shift_iw_v5/<quality>/`. The start banks, the approval and the summary are shared. `launch.sh` multiplies the job
list by the qualities (default: all five) and builds the start banks once.

**Stage order.**
1. The owner approves version 5; the orchestrator freezes it and writes one approval for every quality.
2. Start banks and anchors: one job.
3. Data: 25 jobs (5 qualities × 5 seeds).
4. Learners: 50 jobs (gapped and full data).
5. Room check: 25 jobs. Then exact returns: 25 jobs.
6. A report per quality: the design check and room for caution are judged. If the design check fails, stop.
7. BCA's band and coverage: 50 + 50 jobs (each quality's band waits for its report to say the design check passed).
8. A report per quality, then the summary.
9. BCA training arms, per quality and per kind, only where their marks allow and the owner says go again for that
   kind and quality (or declines it); then the summary judges improvement with Holm's correction.

**Compute** (24 CPUs, at most 20 processes at once). Start banks about 1 process-minute; data about 25; learners
325 (50 × 6.5); room check 300-375 (25 × 12-15); exact returns 50; band and coverage 100. Total about 800-875
process-minutes (13.4-14.6 process-hours), about 1 hour of wall clock on 20 CPUs. The expert column is about 160-175
of these, version 4's own run, not done twice. BCA arms, only where allowed: about 200 process-minutes per quality.

**Code.** `world.py`: the qualities, the random-share rule, the new draws, the quality's controller on the start banks
and the points rule. `data.py`: the quality in every log, the expert's ball, both quadratures, the new anchors, the
random-command flag (a diagnostic no learner can read) and the removal summary's new fields. `run.py`: the
per-quality layout, return units, the 500-step rule, the partial correlation, the p-value, the summary with Holm's
correction and the claims. `learners_iw.py`: points from the new anchors. `launch.sh`: the qualities. Every number is
in `spec.json` (`qualities`, `pass_marks`, `pass_marks_rules.claims`, `evaluation`, `compute`).

**New tests.** The expert equals version 4 bit for bit (arrays, curves, the ball, against hashes recorded from version
4's own code); the random-share rule at shares 0 and 1; each quality's share per pick and the same under every law;
every quality's holds are a subset of the expert's; on steps that are not random, exactly 80% means a hold; the start
banks' controller is the logs' controller; the ball and the design check are shared by all five qualities; every
quality replays through the environment with no mismatch; the anchors are exactly 0 and 100 points and the random anchor
is −800.29 / −1061.42 / −1351.41 at 1,000 starts; the random-command flag is guarded; records stay under their
quality; the return marks read 1.8 from the specification (1.79 fails, 1.81 passes); the 500-step rule; the
p-value; Holm's correction (0.010, 0.020, 0.030 all pass; 0.010, 0.030, 0.040 only the first); the claims; and the
summary's refusals.

### Part B: the lower-edge use of BCA's band

The owner chose to add a second use of BCA's band ("Add the lower-edge use (Recommended)"), next to BCA's own capped
pull. The goal is unchanged: check BCA's mechanics on data where every value is known, then show what weighting by the
known test distribution does. Part B gives weighting a second way to reach return. Numbers are in `spec.json`
(`lower_edge`, and the lower edge's three marks in `pass_marks`).

**Why.** Weighting changes only the band's threshold R. BCA uses its band through the dose 1 + 0.5 U / (U + unit), a
pull toward the recorded command that is capped at 1.5. The dose saturates, so per-step weighting moves it by only
about 0.005-0.03 (section 5), and "can weighting move return?" is expected to fail for the expert. In the lower
edge the width enters the actor's objective linearly, so per-step weighting scales the penalty at each state by
R(w(c)) / R_u: 1.05-1.37 per row on version 4's synthetic gapped banks.

**The lower edge.** Actor-only training against the frozen gapped critic, from the cloned actor, with the room check's
protocol (20,000 steps, the room check's stream of batch rows, Adam 3e-4, batch 256, alpha 2.5):

L = −λ · mean[ Q1(s, π(s)) − k · W(s, π(s)) ] + the usual pull toward the recorded command,
λ = alpha / stopgrad(mean |Q1(s, π(s))|), computed on Q1 alone; k = 1.

- **W(s, a) = [U(s, a) − U(s, a_i)]_+**, the paired form: the width at the actor's command minus the width at the
  row's recorded command at the same state, kept only where positive.
- **U(s, a) = R × unit × scale network(s, a)**, in BCA's float32 arithmetic, from the frozen gapped band.
- **R** is BCA's own threshold R_u in the unweighted arm. In a per-step weighted arm it is each row's own R(w(c)), read
  from version 4's table of R(t). So at each state the penalty is the unweighted one times R(w(c)) / R_u.
- **Gradient.** Only the actor's parameters are trained. The penalty reaches the actor only through the scale network's
  command input; the network's parameters, the unit, R and U(s, a_i) are constants.
- **"Cannot certify"** (declared). A row whose R is +∞ uses the largest finite threshold of its arm's table: the widest
  band the bank still certifies, the most caution that stays finite. "The largest finite width over the bank" is read
  as capping the threshold, not the width, so that the penalty still depends on the command. Such rows are counted;
  for exact weights none are expected (see "the gap is flagged" in section 5). An arm whose table holds no finite
  threshold does not run.
- **Why paired.** The cloud's exploratory Stage 1 review (`outputs/shift/beyond-the-data/stage1/spec.json`, change A1)
  found that an unpaired penalty can reward going just past 80%: measured from zero, the critic's exact error first
  shrinks past the knee. Measured from the recorded command, the penalty is 0 there and never rewards a command for
  being underrated.
- With k = 0, one step is `learners.actor_step`'s, and a one-arm program equals a one-arm `learners.ActorOnly`
  no-caution run, bit for bit (tested). The room check's own no-caution arm, trained five arms side by side in one
  vectorised program, differs from it by lane rounding; no mark depends on it.
- **Gate.** Every function that reads the band's width for the lower edge passes BCA's gate on the band first: the band
  stage's scope for the proxy check (below), the lower edge's own scope for its BCA arms. It also refuses a band fitted
  on other data.

**Arms.** They use BCA's band on real data, so they wait for the owner's second go, like version 4's BCA arms, and
that go is their own: approval scope "bca_lower_edge", naming the qualities it covers. A go for the capped pull ("bca")
does not open them, nor does a go for another quality. Lower edge unweighted; per-step exact-weighted strong (the
improvement mark's arm); per-step exact-weighted mild (reported, on the mild start bank); per-step estimated-weighted
strong only if the user asks (scope "bca_lower_edge_estimated"). A seed's arms run one after another in one process,
each in its own program: on the CPU, lanes of one vectorised program can round differently, so separate programs make
equal thresholds give equal actors, bit for bit. The shape check (below) runs on each arm's penalty before training and
is reported: does BCA's width rise past 80%?

**Room check for the lower edge (no BCA).** It runs after the room check and before any BCA, one job per quality and
seed (stage `room-lower-edge`).
- The same objective, with W from the exact one-step oracle in the paired form:
  W(s, a) = [δ*(s, a_i) − δ*(s, a)]_+, with δ*(s, a) = r + 0.98 · min_h Q_h(s′, c(s′)) − min_h Q_h(s, a). Here c is
  the stall-aware controller, and (s′, r) is one exact float64 simulator step from the row's state, differentiable in
  the command.
- **Float64 inside training.** The training program runs in JAX's 64-bit mode; the actor, the critic and Adam stay
  32-bit. Adam's bias correction is then rounded from 64-bit, so the k = 0 arm differs from the room check's no-caution
  arm. Training amplifies such rounding: on synthetic arrays the largest parameter difference reached 0.13 after 20,000
  steps, and `learners.ActorOnly` itself differs this way between runs with different numbers of arms. That difference
  is reported beside the lower edge's room mark, and the mark compares arms of this one program.
- **Arms**, side by side: no penalty (k = 0), the oracle at k = 1, and the oracle at k = 1.5 (for the reach mark's
  slope). The even pull matched in strength to the k = 1 arm is reported. It is matched on the room check's rungs ×1.1,
  ×1.25 and ×1.5 and two more, ×2 and ×3 (the cloud's matched pull sat at ×1.8-1.9), never extrapolated.
- **The shape check** (check (e) in `spec.json` and the cloud harness), on every seed before training: on 2,000
  hard-water training states, thruster 1 of the no-caution actor's command is set to each throttle of a grid from 70%
  to 100%, along the actor's own push. The oracle penalty must rise past 80%, in four parts:
  1. it falls from 80% to 81% on at most a quarter of the states;
  2. its mean rises from 80% to 81%;
  3. its mean slope at 80.5% is positive and every slope is finite;
  4. the best grid throttle for Q1 − W lies past 80% on fewer states than for the critic alone.
  If the critic alone never prefers a throttle past 80% on these states, there is no over-preference there to remove,
  and part 4 as the harness writes it could never pass (0 is not fewer than 0), whatever the penalty does. Part 4 is
  then recorded as not applicable and passes only if Q1 − W never prefers past 80% either; parts 1-3 still check the
  shape. This is a choice made after the review of version 5, which the owner confirms by approving version 5. A seed
  that fails stops; its record and the report name the parts that failed, and the lower edge's room and reach marks
  fail.
- The oracle, the actor step and the shape check restate the cloud harness
  (`outputs/shift/beyond-the-data/stage1/code/harness.py`, read from the repository, never imported by the run); the
  tests compare them with the harness's own functions.

**Marks** (judged per quality, on its own five seeds, with the same thresholds):

| Mark | Rule | Why |
|---|---|---|
| Room for the lower edge (no BCA) | Strong start bank: the oracle at k = 1 minus no penalty ≥ 1.8 return units, one-sided 95% bound > 0. A seed stopped by the shape check fails it (the report names the parts that failed). The matched even pull is reported beside it, and so are the seeds where the shape check's fourth part was not applicable. | Is there room for a lower edge, given a perfect width? The same 1.8 as room for caution. |
| Can weighting move return through the lower edge? (no BCA training) | Predicted change = (strong-bank return per unit of k, from k = 1 to k = 1.5, mean over seeds) × (mean over kept training rows of R(w_strong(c)) / R_u − 1 on the frozen band, mean over seeds). Its size must be ≥ 1.8 return units. A stopped seed or an invalid band fails it. Reported: the slope in points, high-current rows, the mild arm, refused rows, and the proxy check (below). | A proxy: BCA's unweighted lower edge is treated as the oracle at k = 1, and weighting as raising k on each row to R(w) / R_u. The slope is taken above k = 1, where the unweighted lower edge sits and in the direction weighting moves it, because the response can saturate. |
| Improvement through the lower edge | Only if coverage without a shift, room for the lower edge and "can weighting move return through the lower edge?" all pass and the owner says go for these arms on this quality: per-step exact-weighted strong minus unweighted lower edge on the strong starts, paired by seed. Each report records the difference and its one-sided p-value (p = 1 if a seed's return is not finite); the summary applies Holm's correction across its own family (section 5), separate from the capped pull's: every quality whose report allows these arms. | The goal's return part, through the lower edge. |

The claim "weighting raises return through the lower edge on this quality" is made only where improvement through the
lower edge passes after Holm's correction.

**The proxy check** (reported beside "can weighting move return through the lower edge?", no rule). The prediction
assumes that BCA's unweighted lower edge acts like the oracle at k = 1, and a review of version 5 found that nothing
measured this before the mark gates the arms. So the band stage, on the frozen gapped band (its own scope, no
training), compares the two penalties on every kept training row at the room check's no-caution actor's command: BCA's
paired penalty at R_u and the oracle's paired penalty. It reports the ratio of their means (BCA's effective k against
the oracle's 1), their correlations, the share of rows where each is positive, and the shape check on BCA's own
penalty, with a note naming any seed where that check fails. The owner sees this with the reach mark before saying go
for the lower edge's BCA arms. A ratio far from 1, a weak correlation or a failed shape check means the prediction may
over- or understate what BCA's lower edge can do. About 2 s per gapped seed.

**Expected (written before any run): unknown.** Weighting raises the penalty's scale by 0.05-0.30 on average over
training rows. So the reach mark passes if return rises by about 6-36 return units or more per unit of k above k = 1;
the capped pull needed 62-360 per unit of dose. The cloud's exploratory room test (alpha 5, another critic, two seeds)
found that the oracle lower edge recovered about 90 of 99 lost points under strong current. Its half-strength arm did
about as well, though (81.1 against 77.6, and 75.7 against 82.4 points). If return has already saturated at k = 1, the
slope above it is near 0 and the reach mark fails.

**Stage order.** Room check, then the room check for the lower edge (25 jobs), then evaluation of hosts, room arms and
the lower edge's room arms. First report: the design check, room for caution and room for the lower edge are judged.
Band (with the proxy check; the gapped band reads the room check of its seed) and coverage, then the full report (both
reach marks). BCA arms of either kind run only where their marks allow and the owner says go for that kind and
quality: the capped pull's where coverage without a shift, room for caution and "can weighting move return?" pass; the
lower edge's (`bca-lower-edge`) where coverage without a shift, room for the lower edge and its reach mark pass. Then
evaluation, reports and the summary (both improvement marks, each with Holm's correction over its own family).

**Compute** (measured on synthetic arrays of the full size, one CPU; no ThrusterChain data). The room check for the
lower edge took 397 s per seed without matching: the clone 20 s, the shape check 3 s, the three oracle arms side by side
286 s (4.8 s per 1,000 steps each), the two extra even rungs 88 s. Matching adds 0-5 rounds of one even arm, about 45 s
each. So it is about 7-10 minutes per seed, 175-250 process-minutes for 25 jobs, plus about 13 for evaluating the
extra actors. The proxy check adds about 2 s per gapped seed to the band stage (1.8 s measured with compilation at the
full size). The run's total becomes about 990-1,140 process-minutes (16.5-19 process-hours), about 1.5 hours of wall
clock on 20 CPUs. The lower edge's BCA arms, only where allowed: about 160 s per seed for the three arms (137 s of
training), about 14 process-minutes per quality.

**Code.** `lower_edge.py` (new): the two penalties, the actor step, the side-by-side runner, the shape check, the proxy
check, the room check and the BCA arms. `run.py`: stages `room-lower-edge` and `bca-lower-edge`, the band stage's scale
change and proxy check (for the lower edge's reach mark), evaluation groups, the lower edge's three marks and the new
claim. `learners_iw.py`: the new file is fingerprinted with the code, and the approval's per-kind, per-quality scopes.
`launch.sh`: the two stages.

**New tests** (`test_lower_edge.py`, `test_learners_iw.py`):
- the float64 step equals a float64 reference computed outside training, and the paired penalty is 0 at the recorded
  command;
- the gradient reaches the actor only through the scale network's command input: with that input's weights at 0, k = 1
  trains exactly like k = 0, and the band is unchanged by training;
- the per-step R scales the penalty state by state;
- "cannot certify" uses the largest finite threshold and is counted, and a table with none refuses;
- k = 0 is `learners.actor_step`'s step, and a one-arm program is a one-arm `learners.ActorOnly` no-caution run, bit
  for bit; per-step thresholds all equal to R_u give the unweighted arm bit for bit;
- the shape check's verdict on built surfaces (rising passes; falling, too gentle or not finite fails; the failing
  parts are named; part 4 not applicable when the critic alone never prefers past 80%), and a failed check stops a
  seed;
- the oracle, the actor step and the shape check against the cloud harness's own functions;
- every band reader of the lower edge refuses a band whose approval does not cover its scope, no band, and a band
  fitted on other data; the proxy check's arithmetic;
- the lower edge's BCA arms refuse without their own scope (a go for the capped pull is not enough, and the optional
  arm needs its own), and on relabelled or unregistered data, using only synthetic arrays opened by the synthetic token;
- the stages' refusals, including a go for one kind or one quality opening no other; the marks' arithmetic (1.79
  fails, 1.81 passes); a stopped seed fails the lower edge's room and reach marks; an invalid band fails the reach
  mark; the improvement mark's p-value; its own Holm family; the end-to-end synthetic pipeline through both new stages.

### After a review of version 5

A review of version 5 reported twenty findings, six of them major; several were the same problem found twice. They
come down to the eleven changes below. Each was checked against the code or on synthetic arrays first; nothing ran on
real data, and no pass-mark number changed.

1. **Holm's family depended on when the summary ran (major).** A quality whose report allowed the arms but whose arms
   had not yet run was left out, so an early summary used a smaller family and could make a claim that a later one
   withdrew. Each improvement mark's family is now every quality whose report allows its arms (plus any where it ran).
   The summary judges a family only when every member has run or the owner has declined it, and the claims wait too.
   A declined member counts as a fail (p = 1), so declining never loosens the bar; the decision is recorded from the
   owner's words in `declined.json`. Tested with a member still waiting, a decline, and a broken decline file.
2. **A failed seed dropped its quality from the family (major).** A return that is not finite on one seed gave no
   p-value, so that quality left the family and stayed "pending". It now counts with p = 1, fails and stays in the
   family (tested on both improvement marks).
3. **One go opened both kinds of BCA arm, on every quality (major).** Both kinds were opened by the scope "bca". The
   lower edge now has its own scope ("bca_lower_edge", and "bca_lower_edge_estimated" for its optional arm), and every
   BCA-arm scope names the qualities its go covers; the stage runner refuses any other quality. Tested both ways.
4. **The lower edge's reach mark rested on an unmeasured assumption (major).** Its prediction treats BCA's unweighted
   lower edge as the oracle at k = 1, and nothing measured that before the mark gates the arms. The band stage now
   reports the proxy check (above) beside the mark, so the owner sees it before the go. It is reported only: the
   owner's rule for the mark is unchanged.
5. **Sections 5-7 were still version 4's (major).** They now read 1.8 return units with the slope in return units,
   Holm's correction with its family, the random and stall-aware anchors and version 5's compute. The expected size
   is converted to return units (0.03-0.19 for the expert, against 1.8).
6. **The shape check's fourth part could never pass on a critic that never prefers past 80%.** It is now recorded as
   not applicable there (above), a choice the owner confirms by approving version 5; and a stopped seed's record names
   the parts that failed.
7. **Records from another quality's or seed's folder were accepted.** Every stage record now names its quality and
   seed; the report refuses a record that names another, or links another quality's or seed's records; the summary
   refuses a report in the wrong folder or one that fails its sidecar.
8. **The improvement marks read "pending" forever when coverage without a shift failed.** They now read "not run", as
   their stages refuse.
9. **The lower edge's band readers had no gate of their own.** They now pass BCA's gate on the band and refuse a band
   fitted on other data.
10. **Wording.** The k = 0 claim is now "a one-arm `learners.ActorOnly` run"; the plain marks say the design check's
    82-87% is the expert's, that "tenths" are tenths of the steps sorted by current, and that the report refuses
    results made by other code than the code running now (the band, coverage and BCA stages refuse code other than the
    approved code; the report now says whether the code is the approved code). Part B names marks plainly. In
    `spec.json`, the reach mark's slope is in return units, the retired reference row is gone, and the kept-row counts
    are labelled as the expert's.
11. **Bytecode caches** written into the repository's protected folders by an earlier run without
    PYTHONDONTWRITEBYTECODE were deleted (19 files, none of them source).

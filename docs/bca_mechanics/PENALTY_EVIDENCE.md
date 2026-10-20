# Is BCA's penalty right? The evidence on record, case by case

Written 2026-10-20 for the owner to carry to the advisor. Everything here is read from committed records, from run
files the owner's machine holds, or from two scripts in `docs/bca_mechanics/scripts/` that rebuild the numbers from
those records. Nothing was trained and no band was fitted on study data. The companion module map is
`docs/bca_mechanics/MODULES.md`; the module numbers below (M1 to M11) are its.

The advisor's two questions, in his words: *"Evidence on the penalty if it's constantly wrong/right. IW: is the weighting
of the penalty correct or not. IID = low pen, but if IW has higher pen for IID then I need to see why. Does it always
happen? Is this an exceptional case?"*

## The short answer

1. **Standard BCA's penalty is not a direction signal.** It is the same number whether the test is IID or shifted,
   because it is read on the recorded training rows from a threshold fixed at the last refresh. On every host, data
   quality and benchmark on record the dose sits between 1.29 and 1.39 (110 readings), and the band at the actor's
   unseen commands is only 1.10 to 1.29 times its width at seen commands (25 of 25 seeds) while coverage there falls
   by 0.05 to 0.64. What drops under a shift is **coverage**, not the penalty: 91-93% (IID) to 80-87% (strong shift)
   on 25 of 25 seeds for TD3+BC's critic, and to 63-75% for CQL's. The one penalty that is right in direction is the
   harm switch's height (high where full throttle is harmful, low where it is harmless, 32 of 32 finite cases), and
   it reads held-out outcomes, not where the states are.
2. **The weighting is correct in the sense that matters: with the IID test declared exactly (weights of 1) the weighted
   penalty is the standard penalty to the last bit.** 50 of 50 seed-hosts in the weighting test, 4 of 4 learners in the
   four-learner study, 30 of 30 recorded held-out banks recomputed in the pinned environment. IW's penalty is higher
   than standard's under IID in three settings, each with a known cause:
   - *exceptional*: weights **estimated** by a classifier for the IID test move the threshold by -8.7% to +5.0% on
     single seeds (higher on 14 of 25, lower on 9, equal on 2; beyond 5% on 2 of 25). The sign follows the classifier's
     spurious slope on the current in 20 of 23 moved cases. Classifier noise, not a mechanism of BCA;
   - *consistent, by construction*: weights for the **strong shift**, applied while the learner trains on IID rows, raise
     the penalty on every seed (threshold 1.11-2.25x on 25 of 25; dose +0.011 to +0.075 for TD3+BC on 25 of 25, +0.125 to
     +0.156 for CQL on 10 of 10; the IW arm's dose above the standard arm's in 90 of 90 training chunks). The band is
     built for the declared test law, so this is intended; it costs at most about 1 point of return without a shift;
   - *exceptional, a refusal rather than a wider band*: when the effective number of held-out rows falls below about 29
     the threshold is infinite. That happened in 8 of 40 harm-switch certificates (groups of 18-29 rows) and in the
     command-weighted band (24-100% of the actor's within-range commands refused when the effective bank fell to 0.3-21%
     of its size). It never happened in the IW arms with exact weights (0 refused rows on 35 of 35 seeds).
3. **Why IW's penalty is higher under a declared shift**, from the recorded held-out scores with one input changed at a
   time: the smaller effective sample plus the heavier test atom add about +9% to the threshold on their own (mean over
   600 random reweightings of 30 recorded banks; range 0.81-1.51x); the rest of the recorded 1.11-2.25x comes from the
   weights pointing at the high-current rows, where the critic's errors are larger. The error-size network and the
   refresh timing play no part (they are shared by both arms).

What to say to the advisor, one line each, is at the end (section 9).

## 1. Words used

- **BCA**: Bayesian Conformal Aggregation, the project's method for putting an error band around an offline critic.
- **Critic**: the learned value network Q(s, a) of the host algorithm (TD3+BC, CQL, ReBRAC); the host trains two and
  reads q = min(Q1, Q2).
- **Residual** (M1): y - q, where y = r + gamma min Q(s', pi(s') + noise) is the host's own training target.
- **Error-size network** eta(s, a) and **unit** u (M2): a small network that predicts how large the residual tends to be
  at a row, and a running average of the residual's spread.
- **Score** (M3): |y - q| / (u eta(s, a)), computed on the **held-out rows** (rows kept out of training; about 820-850 in
  every band on record; the code calls them the bank).
- **Weights** w (M4): importance weights p_test(c) / p_train(c) on the held-out rows, where c is the water current.
  *Exact* weights come from the known laws, *estimated* weights from a classifier, *weights of 1* declare the IID test.
  **IW** = importance-weighted BCA. **Standard** = BCA with no weights.
- **Threshold** R (M5): the 90% quantile of the held-out scores, read from 1,000 Bayesian-bootstrap draws at 95%
  credibility, clamped below by the plain weighted quantile: R = max(lambda_hat, lambda_hpd). R = +inf means
  **refusal**: the band cannot be certified.
- **Band width** U(s, a) = R u eta(s, a) (M6). **Coverage**: the share of test rows whose target y lies inside q +/- U;
  the band promises at least 90%.
- **Penalty**: how the host uses U. Four forms are on record:
  - the **dose** m = 1 + 0.5 U / (U + u) on TD3+BC's behavior-cloning term, read on the recorded training rows (M7);
  - the **band width at the actor's own command** (M6 read off the data; the channel check);
  - the **lower edge** Q1 - k U at the actor's command (M9; Stage 2 and the weighting test's lower-edge arms);
  - the **harm switch's height** R_tail - R_rest, the ramp's size when the switch is on (M10, M11; restructured BCA).
- **IID**: test states drawn from the training law (the weighting test's "none" mix, the four-learner study's no-shift
  bank, Stage 2's training states and recorded rows). **Shift**: the mild and strong current mixes (currents still
  inside the recorded range).
- **Test atom** and **test mass**: the extra point the Bayesian quantile adds for the unseen test row; its mass is the
  mean weight under the test law, E_test[w] (1 with no weights, 2.84 for the strong shift on the full training law).
- **Kish effective size**: (sum w)^2 / sum w^2, how many equally weighted rows the weighted bank is worth.
- **Labels**: *pre-registered* (pass marks frozen before the run), *exploratory* (seen seeds, or run outside a
  pre-registration), *description only* (reported, never judged).

## 2. Where the penalty is computed, as numbered lines

Each line carries its module and purpose; the real code is cited beside each block. The blocks are the path a number
takes from the held-out rows to the penalty.

**Block A. The standard threshold at a refresh** (`algorithms/td3_bc_bca.py:244-270`, `calibration/reference.py:152-194`,
`calibration/wbcp.py:62-117`)

```
1  [M1.1  residual: y = r + gamma min Q(s', pi(s') + noise), q = min(Q1, Q2)(s, a) on the held-out rows]   td3_bc_bca.py:244-256
2  [M2.1  scale: eta(s, a) from the frozen error-size network; u = the running unit]                       td3_bc_bca.py:257, reference.py:164
3  [M3.1  score_i = |y_i - q_i| / (max(eta_i, 1e-6) u)  for every held-out row i]                           reference.py:175
4  [M5.1  weights: none given, so w_i = 1 and test mass = 1 (the exchangeable case)]                        wbcp.py:65-66
5  [M5.2  lambda_hat = the smallest score whose weighted share of all scores is at least 0.90]              wbcp.py:90-101
6  [M5.3  draw 1,000 Bayesian-bootstrap weightings (standard exponentials) of the scores plus one test atom] wbcp.py:103-108
7  [M5.4  each draw's crossing = the smallest score at which that draw's mass reaches 0.90 of its total;
          a draw whose test atom alone exceeds 0.10 of the total never crosses (+inf)]                       wbcp.py:43-56
8  [M5.5  lambda_hpd = the 950th smallest crossing (95% credibility); +inf if fewer than 950 are finite]    wbcp.py:110-114
9  [M5.6  R = max(lambda_hat, lambda_hpd); stored in float32 with u and eta's parameters]                    wbcp.py:116, reference.py:185-188
```

**Block B. The weighted threshold** (`experiments/shift_iw/world.py:128-135`, `experiments/shift_iw/coverage.py:59-97,
173-205`; the four-learner restatement is `experiments/shift_hosts/iw.py` on branch `claude/shift-hosts`)

```
1  [M3.2  the same scores as block A, recomputed from the stored state with the refresh's own key (checked bit for bit)]  learners_iw.py:541-569
2  [M4.1  exact weight of held-out row i: w_i = p_test(c_i) / p_train(c_i) = (M(-2)/M(lam)) exp((lam + 2) c_i);
          lam = -2 (IID, so w = 1), 0 (mild), +2 (strong)]                                                    world.py:128-135
3  [M4.2  or estimated weight: a logistic classifier on the 13-number observation, class sizes corrected]     classifier.py:27-128
4  [M4.3  test mass t = the test row's own weight (per-step arms) or the mean weight (single threshold)]      coverage.py:153-170, SPEC.md:33
5  [M5.7  R(w, t) = calibrate(scores, same random stream, w, t): the weighted lambda_hat, the weighted draws,
          R = max(lambda_hat, lambda_hpd)]                                                                    coverage.py:59-62, wbcp.py:84-116
6  [M5.8  unit check: calibrate(scores, 1, 1) must equal the unweighted call in threshold, lambda_hat,
          lambda_hpd and all 1,000 crossings]                                                                 coverage.py:78-86
7  [M5.9  per-step table R(t_k) on the grid t_k = 1.01^k, running maximum if it ever falls]                   coverage.py:173-186
8  [M5.10 a step's threshold = R at the smallest grid mass at least its own test mass; +inf = refused]        coverage.py:196-205
```

**Block C. From R to the penalty** (`calibration/dose.py:65-95` and `:23-62`, `experiments/shift_iw/learners_iw.py:655-683`,
`experiments/shift_iw/lower_edge.py:196-224`)

```
1  [M6.1  width U(s, a) = R max(eta(s, a), 1e-6) u, float32; +inf when R = +inf]                              dose.py:74-76
2  [M7.1  dose m = 1 + 0.5 U / (U + u), between 1 and 1.5; read at the recorded (s, a) of each training batch] dose.py:59-60, td3_bc_bca.py:283-288
3  [M7.2  refused row: dose 1 in BCA's own rule (native multiplier); 1.5 in the per-row IW arms]              dose.py:52-53, learners_iw.py:681-682
4  [M9.1  lower edge: U at the actor's command a minus U at the recorded command, floored at 0]                lower_edge.py:196-208
5  [M9.2  per-row thresholds R_i capped at the largest finite one; refuse only when none is finite]            lower_edge.py:211-224
```

**Block D. The harm switch** (`outputs/shift/beyond-the-data/restructure/step1/step1lib.py:244-276`)

```
1  [M10.1 split the held-out rows into the full-throttle group (tail) and the rest by the throttle gap rule]   step1lib.py:174-215
2  [M10.2 one-sided overestimates s = Q(s, a) - [r + gamma (1 - d) Q(s', c(s'))] in each group]              step1lib.py:216-225
3  [M10.3 posterior of each group: block A lines 5-9 on that group's overestimates, its own random tags]      step1lib.py:237-241, 257-258
4  [M10.4 switch on iff the 5% quantile of (crossing_tail - crossing_rest) over all draw pairs > 0.10]         step1lib.py:226-235, 260, 267
5  [M11.1 height = R_tail - R_rest (both deployed thresholds); +inf when the tail has too few rows]           step1lib.py:271
```

Two facts follow from the blocks and decide the rest of this document. First, nothing in block A or C looks at the
test condition: the standard penalty cannot differ between IID and shifted tests. Second, the weights enter only block
B line 5: eta, u and the refresh are shared by the standard and weighted arms, so any difference between them is the
threshold.

## 3. Question 1: is the penalty low where the data cover the state and command, higher where they do not?

Measured per seed and condition. Figure: `figures/penalty_iid_vs_shift.svg` (panels A to F). Every number below is
printed by `scripts/penalty_cases.py` (cases 1 to 5) from the sources named.

### 3.1 The dose (M7): the same number in every condition — consistent, 110 of 110 readings

| Where read | Standard dose | Source |
|---|---|---|
| TD3+BC, weighting test, seed 0, mean per 5,000 updates after the first refresh (18 chunks x 5 qualities) | 1.293 to 1.343; last chunk 1.343 / 1.336 / 1.341 / 1.316 / 1.318 (random to expert) | `outputs/shift_iw/explore/routes2/diagnose/dose_chunks.json` (exploratory) |
| the same band read at the recorded command, the actor's command, a 60% hold, an 80% hold and full throttle (seed 0) | spread at most 0.0041 across the five commands | `.../diagnose/diag_band_at_commands.json` (exploratory) |
| four-learner study, last refresh, per seed: CQL medium / CQL medium-expert / ReBRAC medium / TD3+BC medium-expert | 1.347-1.373 / 1.336-1.385 / 1.299-1.354 / 1.357-1.368; single rows 1.008-1.458 | `origin/claude/shift-hosts:outputs/shift_hosts_v3/results.json` (exploratory) |
| D4RL hosts and the linear-quadratic harness (signal study) | 1.26-1.33 and 1.29-1.35 | `experiments/signal/SIGNAL_STUDY.md:215-227, 336-340` (exploratory, audited) |

The dose is read on the recorded training rows (block C line 2), so a shift in the test states never reaches it. It
is not "low" under IID either: calibration puts R eta near the 90% point of a unit-spread score, so U is about 2 to 2.5
units and the dose about 1.33 everywhere.

**Verdict for the dose as a direction signal: does not work.** Takeaway: "Standard BCA's penalty is the same 1.33 under
IID and under shift; it never looks at the test condition."

### 3.2 Coverage, not the penalty, moves with the condition — consistent, 25 of 25 seeds

Weighting test, TD3+BC's critic, standard band, per seed (pre-registered marks M1 and W2,
`outputs/shift_iw/v5/<quality>/report.json`): IID coverage 0.889-0.940, strong-shift coverage 0.797-0.873; lower under
the shift on 25 of 25 seed-qualities, by 0.046 to 0.112. Four-learner study, standard band under the strong law at
recorded rows: CQL 0.630-0.752, TD3+BC 0.833-0.870, ReBRAC 0.904-0.961 (its critic did not learn, so its errors do
not grow with the current).

### 3.3 The band width at the actor's own command (M6): flat where coverage collapses — consistent, 25 of 25 seeds

Channel check (exploratory; `outputs/shift_iw/explore/channel/results/<quality>.json`), standard band, per seed:

| Quality | Width past 80% throttle / width within 80% (5 seeds) | Coverage within 80%, training law | Coverage past 80%, training law | Width multiplier that would have been needed past 80% (strong law) |
|---|---|---|---|---|
| random | 1.10-1.17 | 0.961-0.967 | 0.864-0.902 | 1.10-1.25 |
| medium-replay | 1.14-1.27 | 0.951-0.974 | 0.846-0.919 | 0.99-1.31 |
| medium | 1.15-1.21 | 0.958-0.979 | 0.790-0.889 | 1.19-1.71 |
| medium-expert | 1.19-1.29 | 0.970-0.985 | 0.559-0.664 | 3.21-3.77 |
| expert | 1.11-1.25 | 0.953-0.979 | 0.316-0.446 | 6.41-9.27 |

Past 80% throttle is the removed region: no training row has such a command. There the band is 10-29% wider and wrong
2 to 20 times as often. With exact strong weights the band is 1.1-2.3 times wider everywhere (within and past 80%
alike; figure panel D), which restores 90-95% coverage on the test steps as a whole but still leaves 21-43%
(medium-expert) and 56-72% (expert) of the past-80% commands uncovered under the strong law.

**Verdict for the width at the actor's command: does not work** as a direction signal. Takeaway: "Where the data stop,
the band does not widen; it is simply wrong more often."

### 3.4 Stage 2's lower edge (M9): flat between the harmful command, its cap and the recorded command — consistent, 5 of 5 seeds

From the recorded width scores (`runs/shift_explore/stage2/detect/<version>/stall/fixed/seed<K>/scores/`; exploratory
seeds 0-4; case 3): the uniform (standard) band at training states in the full-throttle region is 0.308-0.345 at the
actor's harmful command, 0.307-0.341 at that command capped at 80%, and 0.295-0.329 at the recorded command. Command
minus cap on harmful pairs: +0.0003 to +0.0122 on every seed and set. The band at strong-current rollout states
(shifted) is 0.92 to 1.18 times the band at training states. The false preference the lower edge had to cancel was
0.63-0.89 (`outputs/shift/beyond-the-data/results-2026-10-13/README.md`). The classifier-weighted version is 0.8-2.4 at
the harmful command (it adds a refusal term built from the classifier), but also 0.23-0.57 at recorded rows, higher
than the uniform band on 3 of 5 seeds.

**Verdict: does not work** (standard band). Takeaway: "The lower edge with BCA's band is a constant; the detection
passed only with the classifier's weights, which carry the signal themselves."

### 3.5 The harm switch's height (M10, M11): right in direction — consistent, 32 of 32 finite cases; refused in 8 of 40

Every certificate on record (near-80 run, 30 seed-worlds; step 1, 10; case 4): where full throttle stalls the height is
1.41 to 6.02; where it is harmless, 0.04 to 0.77; every stall height exceeds every no-stall height. The height is
infinite in 8 of 40 seed-worlds, all with 18-29 rows in the full-throttle group (section 5.4). This penalty reads
held-out *outcomes* of the full-throttle group, not where the states are, which is why it can have a direction.

**Verdict: works with a caveat.** Takeaway: "The one BCA penalty that points the right way is the one that reads
outcomes in the risky group; it refuses when the group has fewer than about 29 rows."

### 3.6 The penalty rises under the shift only when the shift is declared through weights — consistent, 25 of 25

Weighting test, same held-out scores, exact weights (shown values, single threshold; case 5): R_IW / R_standard is
1.03-1.35 for the mild mix and 1.11-2.25 for the strong mix, above 1 on 25 of 25 seeds for each, strong above mild on
25 of 25; the dose change on kept training rows (mark R2) +0.011 to +0.075 on 25 of 25; the lower edge's scale change
(mark L2) +0.11 to +1.25 on 25 of 25; coverage under the strong shift restored to at least 0.89 on 21 of 25 seeds
(0.877-0.963). CQL's dose rises from 1.336-1.385 to 1.472-1.497 on 10 of 10 seeds (+0.125 to +0.156) and its coverage at
recorded rows from 0.63-0.75 to 0.89-0.95.

**Verdict for the weighted penalty under a declared shift: works.** Takeaway: "Tell BCA the test law and its penalty
rises everywhere by 1.1-2.3x; it never finds the shift on its own."

## 4. Question 2: is the importance weighting of the penalty correct? When is IW's penalty higher than standard's under IID?

Figure: `figures/penalty_iw_minus_standard.svg`. Cases 6 to 10 of `scripts/penalty_cases.py`.

### 4.1 The count, per study, seed and refresh

"IID" here means the held-out rows and the test follow the training law. "Higher" means IW's threshold, dose or width is
above standard's on the same rows.

| Study | Weights | What was compared | IW higher | IW lower | Equal | Label |
|---|---|---|---|---|---|---|
| Weighting test (TD3+BC's critic) | weights of 1 (exact for the IID mix) | 25 seed-qualities x 2 hosts, frozen band: threshold, lambda_hat, lambda_hpd, all 1,000 crossings, per-step table and the per-step arm's coverage | 0 | 0 | 50 | pre-registered mark W0 |
| Weighting test | exact weights for the IID mix, per-step table | per-step ratio R_IW / R_standard on 25 seed-qualities (mean, min and max over test steps) | 0 | 0 | 25 (table [1.0]) | shown value |
| Weighting test | **estimated** weights fitted for the IID mix, single threshold | 25 seed-qualities | **14** (6 by more than 1%, 1 by more than 5%) | 9 (6 by more than 1%, 1 by more than 5%) | 2 | shown value; pre-registered W4 judged the coverage gap (0.0001-0.002, pass) |
| Weighting test | exact weights for the **strong** mix, learner trained on IID rows | 25 seed-qualities (frozen band, R2 dose change); seed 0: 18 chunks x 5 qualities; 10 recorded refresh tables | **25; 90 of 90; 10 of 10** | 0 | 0 | R2 pre-registered; chunks exploratory |
| Four-learner study | weights of 1 (stop rule S2) | every bit of the run after the refresh, and R(1) against the stored standard threshold, 4 learners | 0 | 0 | 4 | pre-specification version 3 |
| Four-learner study | exact strong weights (CQL) | mean dose on kept training rows at the last refresh, 10 seeds; thresholds reproduced at 18 of 18 refreshes (S3) | **10 of 10** (+0.125 to +0.156) | 0 | 0 | exploratory |
| Four-learner study | **shuffled** strong weights (placebo: same sizes, no link to the rows), CQL | 10 seeds | 6 (at most +0.017) | 4 (at least -0.008) | 0 | exploratory control |
| Four-learner study | exact strong weights, ReBRAC (critic did not learn) | 5 seeds | 2 (+0.003) | 3 (down to -0.022) | 0 | description only |
| Recorded held-out banks (near-80 and Stage 2), pinned environment | weights of 1 | 30 banks | 0 | 0 | 30 | new recompute |
| Recorded held-out banks | strong weights given to rows **at random** (no link to the score), 20 draws per bank | 600 draws | **522** (mean +9%, up to +51%) | 68 (down to -19%) | 10 | new recompute, synthetic currents |
| Recorded held-out banks | the 25 recorded IID-classifier slopes applied to random currents | 3,000 cases | within 0.930-1.033; beyond 5% in 2 | | | new recompute, synthetic currents |
| Stage 2 | classifier-weighted band (weights for the actor's commands + refusal term) at recorded rows | 5 seeds | 3 (0.32-0.57 against 0.28-0.31) | 2 | 0 | exploratory; a different construction |
| Channel check | command-weighted band at the actor's within-80% commands | 25 seeds | **25** (refuses 24-100% of these commands) | 0 | 0 | exploratory; a refusal, not a width |

The first-order answer: **with the IID test declared exactly, IW never gives a higher penalty; it gives the same one,
bit for bit, in every study.** Higher penalties under IID come from three sources, treated in turn.

### 4.2 Exceptional: estimated weights for the IID test (case 7)

The classifier is fitted to tell test states from training states. Under the IID test there is nothing to tell, the
exact weights are 1, and the fitted slope on the current should be 0. The recorded slopes are -0.28 to +0.42 (mark W4,
25 seeds). Fed to the threshold, these weights move R_IW / R_standard between 0.913 (medium-expert seed 2) and 1.050
(medium-expert seed 4):

| Quality | R_IW / R_standard per seed | Fitted slope on the current per seed |
|---|---|---|
| random | 1.023, 0.981, 1.000, 1.001, 1.042 | +0.19, -0.12, -0.28, +0.08, +0.39 |
| medium-replay | 1.003, 0.990, 0.990, 1.013, 1.002 | +0.29, -0.02, -0.12, +0.10, +0.42 |
| medium | 1.010, 0.988, 0.986, 1.006, 1.038 | +0.11, -0.08, -0.28, +0.14, +0.29 |
| medium-expert | 1.000, 0.973, 0.913, 1.031, 1.050 | +0.28, +0.11, +0.00, +0.10, +0.33 |
| expert | 1.001, 0.994, 0.995, 1.003, 1.005 | +0.20, -0.23, +0.01, +0.15, +0.37 |

Does it always happen? No: higher on 14 of 25, lower on 9, equal on 2; the means by quality are within 1% of 1; the
coverage of the estimated band differs from the exact one by -0.016 to +0.011 per seed (pass under W4). Why does it
happen? The sign of the move follows the sign of the fitted slope in 20 of the 23 moved cases: a positive slope tilts
the weights toward the high-current rows, where the critic's scores are larger (mark M2), so the weighted 90% quantile
rises; a negative slope lowers it. The three disagreements are medium-expert seeds 1 and 2 and expert seed 2, where the
slope on the current is near 0 but the classifier used the other twelve state numbers. The smaller effective sample is
not the cause: the same 25 slopes applied to random currents on the 30 recorded banks (no link between a row's weight
and its score) move R by only 0.930 to 1.033, beyond 5% in 2 of 3,000 cases. **Exceptional; the fix is a better
classifier or exact weights, not a change to BCA.**

### 4.3 Consistent and intended: strong-shift weights while training on IID rows (case 8)

This is how the IW arm is actually run. Its band is calibrated for the strong test law, and the dose it reads on the
recorded (IID) training rows is therefore higher than the standard arm's: at every one of the 18 refreshes (seed 0,
five qualities, 90 of 90 chunks; per-row threshold ratios 1.16-2.37 at the recorded refreshes), on the frozen band (R2,
25 of 25), and for CQL (10 of 10). The band is for the declared test law; raising the penalty on the training rows is
the price of covering the shifted rows. The price on record (exploratory): the capped pull loses return on the IID
starts on 18 of 25 seed-qualities, by -9.1 return units on medium-replay (-1.2 points), -4.6 on medium, within 1.3
units elsewhere; the lower edge within 1.7 units either way on every seed; CQL's claim 3 ("IW costs nothing without the
shift") passed on both qualities (-0.75 and +0.45 return units). **Consistent, by construction, and the cost is about a
point at most.**

### 4.4 Exceptional: refusal (case 9)

The credible quantile needs at least 950 of the 1,000 draws to cross, and a draw fails with probability 0.9^n, so
about 29 held-out rows are needed (`outputs/shift/beyond-the-data/near80_results_check/REPORT.md:24-27`). On record:

- harm-switch certificates: every full-throttle group of 28 or fewer rows refused (n = 18, 18, 19, 21, 22, 28), every
  group of 30 or more certified, and at exactly 29 rows 2 of 3 refused (a 70/30 draw: the share of finite draws was
  0.947, 0.947 and 0.950 against the 0.950 needed). Three of the five fresh stalling seeds (10, 12, 13) had the switch
  on with an infinite height, and the approved code refused to build the arm;
- before it refuses, the same arithmetic widens: the Bayesian margin lambda_hpd / lambda_hat on random subsets of the
  recorded scores is 1.09 at n = 805-851, 1.17 at n = 300, 1.50 at n = 100;
- the IW arms with exact weights never refused a row: R2's refused rows 0 on 25 of 25 seeds, CQL's 0 on 10 of 10 (18 of
  18 refreshes certified). The largest exact strong weight is 7.39, about 1.2% of the bank's total weight
  (`experiments/iw/SPEC.md:189-192`), too small to sink a draw;
- the command-weighted band (weights for the actor's own commands; exploratory) is the one place IW's penalty is far
  above standard's under IID-like commands: it refuses 24-100% of the actor's within-80% commands on 25 of 25 seeds,
  because the Kish effective size of the bank falls to 0.3-21% of its rows. A refusal, not a wider band.

### 4.5 Control: weights change nothing when the errors do not grow with the current (case 10)

ReBRAC's critic did not learn; its standard band already covered 90-96% under the shift. The strong weights change its
dose by -0.022 to +0.003 (2 of 5 above 0). CQL's shuffled-weight placebo changes the dose by -0.008 to +0.017 (6 of 10
above 0) against +0.125 to +0.156 with the real weights. The rise needs weights that point at rows where the critic's
error is larger.

## 5. The mechanism: why the weighted threshold is higher, from the recorded scores

Script `scripts/penalty_recompute.py`, run in the pinned environment (Python 3.10.20, NumPy 2.2.6, JAX 0.6.2) on the 30
recorded held-out banks (20 of the near-80 run, 10 of Stage 2; 805-851 scores each); output
`data/penalty_recompute.json`. The scores, eta and u are the recorded ones; only the weights, the test mass and the
random stream are varied, one at a time. The currents given to the rows in cases B to F are synthetic (a bank's own
currents are not stored beside its scores), so those cases describe the arithmetic of the threshold on real scores, not
a recorded study result; case A is the recorded reproduction.

| Case | What changed | R_weighted / R_unweighted (30 banks) |
|---|---|---|
| A | weights of 1, test mass 1, the refresh's own stream | 1.0000 on 30 of 30; equals the recorded threshold (float64 recompute and float32 record) and the explicit unweighted call bit for bit on 30 of 30 |
| B | test atom alone: mass 2.84 (the strong law's mean weight) with weights of 1 | mean 1.015 (1.002-1.047) |
| C | strong weights given to rows at random (Kish 0.35 n), test mass 2.84, 20 draws per bank | mean 1.094, range 0.806-1.513; above 1 in 522 of 600 draws; weighted lambda_hat / unweighted 1.013; Bayesian margin lambda_hpd / lambda_hat 1.089 -> 1.175 |
| C-mild | mild weights at random (test mass 1.38) | mean 1.023 |
| D | strong weights sorted onto the scores: heaviest on the largest scores / on the smallest | 2.91 / 0.43 (the most the tilt can add or take away) |
| E | the 25 recorded IID-classifier slopes on random currents (4 draws each, 3,000 cases) | 0.930-1.033; beyond 5% in 2 |
| F | weights of 1 on random subsets of 300 and 100 rows (margin lambda_hpd / lambda_hat) | 1.17 and 1.50 against 1.09 at full size; refusal below about 29 rows |

Reading the table against the records:

1. **The reweighting toward high-error rows is the main part.** The recorded strong ratios, 1.11-2.25 per seed, sit far
   above case C (no link between weight and score, mean 1.09) and below case D's upper bound (2.9). Under the strong law
   the heavy rows are the high-current rows, where TD3+BC's and CQL's scores are larger (mark M2; CQL's standard band
   covers only 63-75% there). The weighted 90% quantile therefore moves up. ReBRAC, whose errors do not grow with the
   current, shows no rise (section 4.5).
2. **The smaller effective sample and the heavier test atom add about +9%** to the threshold on their own (case C), of
   which the test atom is +1.5% (case B) and the rest is the Bayesian margin growing from 9% to 17% as the Kish size
   falls from about 830 to about 290. In dose terms this is about +0.01, the size of the four-learner shuffled placebo
   (-0.008 to +0.017). This is the one part that can make a weighted band wider than the standard one on exchangeable
   data, and it is also what makes the threshold refuse below about 29 effective rows (case F, section 4.4).
3. **eta and u play no part.** Weighting changes R only (block B line 5); the standard and weighted arms share the
   error-size network, the unit, the held-out scores and the refresh (stop rule S3 reproduced the standard threshold at
   18 of 18 refreshes in every four-learner IW run; the weighting test's IW arm recomputes the bank scores after each
   unchanged refresh, `experiments/shift_iw/learners_iw.py:866-870`). The dose's sensitivity to R is at most 0.125 per
   unit of ln R (`experiments/iw/SPEC.md:186-187`), which is why a 1.2-1.8x threshold moves TD3+BC's dose by only
   0.02-0.05; CQL's larger move (+0.14) comes from its larger R eta (its standard dose is already 1.35 with R 1.6-2.8).
4. **Refresh timing plays no part** in the comparisons on record: both arms certified every refresh, and the per-refresh
   comparison (section 4.3) is above 1 at every refresh.

## 6. The case list for debugging and for the advisor

| # | Case | Consistent or exceptional | Numbers | Source | Rebuild |
|---|---|---|---|---|---|
| 1 | Standard dose is the same number under IID and shift | consistent | 1.293-1.385 on 110 readings; 0.004 spread across five commands | dose_chunks.json, diag_band_at_commands.json, shift_hosts_v3 results | `penalty_cases.py --case 1` |
| 2 | Coverage drops under the shift while the width barely moves | consistent | 25 of 25 seeds; width past 80% 1.10-1.29x, coverage down 0.05-0.64 | report.json M1, W2; channel results | `--case 2` |
| 3 | Stage 2 band flat between harmful command, cap and recorded command | consistent | +0.0003 to +0.012 on 5 of 5 seeds | stage2 width scores | `--case 3 --stage2-copies <dir>` |
| 4 | Harm-switch height high where harmful, low where harmless | consistent | 32 of 32 finite; 1.41-6.02 against 0.04-0.77 | switch.json x 30, step1 results | `--case 4` |
| 5 | Declared weights raise the penalty on every seed | consistent | R 1.11-2.25x (25 of 25), dose +0.011 to +0.075 (25 of 25), CQL +0.125 to +0.156 (10 of 10) | report.json shown values, R2, L2; shift_hosts_v3 | `--case 5` |
| 6 | Weights of 1 = standard bit for bit | consistent | 50 of 50, 4 of 4, 30 of 30 | W0, S2, penalty_recompute.json | `--case 6`; `--runs <runs/shift_explore>` recomputes live |
| 7 | Estimated IID weights move R by -8.7% to +5.0% | exceptional | higher 14 of 25, lower 9, equal 2; beyond 5% on 2; sign follows the slope 20 of 23 | report.json shown values, W4 | `--case 7` |
| 8 | Strong-shift weights on IID training rows: always higher | consistent, by construction | 25 of 25; 90 of 90 chunks; 10 of 10 refresh records; CQL 10 of 10; cost at most about 1 point | report.json, dose_chunks.json, RESULTS.json, shift_hosts_v3 | `--case 8` |
| 9 | Refusal below about 29 effective rows | exceptional, tied to group size | 8 of 40 certificates (n 18-29; 2 of 3 at n = 29); 0 refused rows in the IW arms; command-weighted band refuses 24-100% | switch.json, step1 results, command_band results | `--case 9` |
| 10 | No rise when the error does not grow with the current | control | ReBRAC -0.022 to +0.003; shuffled placebo -0.008 to +0.017 | shift_hosts_v3 | `--case 10` |

The three recorded exceptional cases, in the advisor's sense of "IW has a higher penalty under IID", are case 7
(estimated weights, both directions, classifier noise), case 9 (refusal, group size) and, by construction, case 8.

## 7. What this means for whether the penalty can be trusted

- **As a measure of where the critic is wrong, the standard penalty cannot be trusted** in these worlds, and the records
  say so consistently rather than occasionally: it is one number per band (about 1.33 in dose terms, 0.3 in Stage 2's
  value units), read on training rows, flat across the actor's commands and across the IID-to-shift boundary. Its
  coverage promise holds where the data are (89-94% IID on every seed) and fails where they are not (80-87%, 63-75%, and
  25-45% at the actor's unseen commands). This is what the placement, Stage 2 and near-80 studies found by other means.
- **The weighting is mechanically right.** It changes only the threshold; with the IID test declared it is the standard
  band to the last bit; with a declared shift it raises the band everywhere by the amount the declared law demands and
  restores coverage. Its higher penalty on IID rows is the declared shift, not an error, and costs about a point.
- **The two places IW's penalty is higher under IID without a declared shift are both understood and bounded:**
  classifier noise (within +/-5% on 23 of 25 seeds, +/-9% at worst) and refusal when the effective bank falls below about
  29 rows. Neither is a bug in BCA's arithmetic; the first is a weakness of estimated weights, the second a property of
  the Bayesian quantile that the switch and the height handle differently (the switch can be on while the height is
  infinite, which stopped the near-80 run on 3 of 5 fresh seeds).
- **The one penalty that points the right way is the harm switch's height**, and it does so by reading held-out
  outcomes of the risky group, not by reading the band's width where the data stop. That is the design lesson: a
  conformal band calibrated on recorded rows describes the recorded rows.

## 8. How to reproduce every number here

From the repository root (the worktree), with Python 3 and NumPy (the project venv or the pinned environment):

```
# the case list (prints before/after for cases 1-10; the four-learner record is read with git show unless --hosts is given)
python docs/bca_mechanics/scripts/penalty_cases.py --root . --case all \
    --hosts <copy of origin/claude/shift-hosts:outputs/shift_hosts_v3/results.json> \
    --stage2-copies <folder with pairs/seedK/{pairs,labels}.npz and {uniform,weighted}/stall/seedK/scores/bca_width_*_fixed.npz> \
    --write-json docs/bca_mechanics/data/penalty_cases.json
# saved output of that command: docs/bca_mechanics/data/penalty_cases.out

# the mechanism on the 30 recorded banks (pinned environment, through the WSL bridge; 90 seconds)
PYTHONPATH=/home/user/bca JAX_PLATFORMS=cpu /tmp/claude-0/bcavenv/bin/python docs/bca_mechanics/scripts/penalty_recompute.py \
    --runs /home/user/bca/runs/shift_explore --out docs/bca_mechanics/data/penalty_recompute.json
# the same banks are on branch claude/near80-run-files (near80/learners/*/*/seed*/v6_bank_scores.npz + v6_band.json);
# case A reproduces the recorded thresholds only with NumPy 2.2.6 and JAX 0.6.2 (the recorded random stream)

# a single live case on one recorded bank (weights of 1; test atom; random strong weights; sorted weights)
python docs/bca_mechanics/scripts/penalty_cases.py --root . --case 6,8 --runs <runs/shift_explore> --banks 1

# the two figures
python docs/bca_mechanics/scripts/penalty_figures.py --cases docs/bca_mechanics/data/penalty_cases.json \
    --recompute docs/bca_mechanics/data/penalty_recompute.json --out docs/bca_mechanics/figures
```

Which inputs are where: `outputs/shift_iw/v5/<quality>/report.json` and `outputs/shift_iw/explore/**` are on this
branch; the four-learner record is `origin/claude/shift-hosts:outputs/shift_hosts_v3/results.json`; the near-80
certificates are under `outputs/shift/beyond-the-data/near80/results-20261020T115246Z/files/learners/`; Stage 2's width
scores and pair labels are run files (`runs/shift_explore/stage2/detect/...` and `runs/shift_explore/detect/seed<K>/stall/`
in the owner's WSL clone; copies were used here). The weighting test's and the four-learner study's raw arrays (held-out
scores, per-row weights, per-row doses) are not in the repository, which is why the mechanism was recomputed on the
near-80 and Stage 2 banks.

## 9. What to say to the advisor

- "Standard BCA's penalty is the same number, about 1.33, whether the test is IID or shifted. It is read on the training
  rows from a fixed threshold, so it cannot know where the data stop. What drops under a shift is coverage: 91-93% to
  80-87% for TD3+BC's critic, 63-75% for CQL's, on every seed."
- "The weighting is mechanically correct: with the IID test declared, weighted BCA is the standard band to the last bit,
  checked in two studies and on 30 recorded banks."
- "Weighted BCA's penalty is higher on IID rows only when the weights declare a shift, and then it is higher on every
  seed: 1.1 to 2.3 times in the threshold, 0.02 to 0.16 in the dose. That is intended; it restores 90-95% coverage
  under the shift and costs about a point without it."
- "The exceptions are two: classifier-estimated weights add +/-5% of noise (14 seeds up, 9 down, 2 beyond 5%, sign set
  by the classifier's spurious slope), and the threshold refuses when the effective number of held-out rows falls below
  about 29 (8 of 40 harm-switch certificates; never in the weighted arms with exact weights)."
- "The reason a declared shift raises the penalty is mostly that it points the band at the high-current rows, where the
  critic is wrong by more; the smaller effective sample adds about 9% to the threshold and about 0.01 to the dose."
- "The one BCA penalty that points the right way, high where full throttle harms and low where it does not on 32 of 32
  finite cases, is the one that reads held-out outcomes in the risky group, not the band's width."

## 10. Labels and gaps

Pre-registered: the weighting test's marks M1, W0, W1, W2, W4, R2, L2 (spec version 5; the BCA arms' returns are
exploratory); the near-80 run's seeds 10-14 (claim 4, failed by the refusal rule). Exploratory: the four-learner study
(seeds 0-4, pre-specification version 3 with stop rules S2 and S3), Stage 2 (seeds 0-4), step 1 and the near-80 seeds
0-4, the channel checks, the dose-per-chunk diagnosis (seed 0 only), the return costs. Description only: the "shown"
threshold ratios of the weighting test, ReBRAC's band readouts, the single-threshold estimated-weight ratios. New here:
the pinned recompute on recorded banks (cases B to F use synthetic currents on real scores).

Gaps. No record reads BCA's penalty under the main study's big-waves or big-kicks situations. The weighting test's
per-seed dose levels are not in the repository (only the change); the four-learner study and the signal study give
the level. The per-refresh comparison of IW against standard is on record for one seed per quality (dose per chunk)
and for two refresh tables per quality; the frozen-band comparison covers all 25 seeds. Nothing on record separates
the reweighting part from the sample-size part on the weighting test's own banks (their arrays were not kept); the
separation in section 5 is on the near-80 and Stage 2 banks with synthetic currents.

# Step 1: calibration-bank dependence

This note answers, for every change made in this step: what changed, why, why it makes
sense, what was expected (written down before the result), what happened, whether it met
the expectation, and if not, why. All numbers come from the semi-synthetic D4RL benchmark
([README](README.md#semi-synthetic-benchmark-on-d4rl)). TD3+BC with BCA's scale fit was
trained for 100k updates on half of hopper-medium-v2's episodes, frozen, and scored on the
other half (500,285 transitions, 1,094 episodes, mean length 457).

Target miscoverage is alpha = 0.1 and the posterior credibility is beta = 0.95. A
calibration rule is therefore valid if at most 5% of calibration banks give a threshold
whose realized miscoverage exceeds 10%. "Failure" below is that frequency. Changes 1-9 and 11 use
that hopper pool; Change 10 adds walker2d-medium-replay and pen-cloned. Every pool comes from
the TD3+BC host (see Limits).

This is the second version of the note. Two independent reviews of the first version found
a wrong design-effect input (Change 2), a selectively reported pre-registration (Change 5),
several overstated verdicts and an inaccurate sampler description. All are corrected below.
Changes 6-11 are new. Change 9 implements the recommended bank in BCA, Change 10 tests the
dependence on two more datasets, and Change 11 explains why WBCP exceeds 5% under strong
shift even with independent rows.

## Terms

- **Bank**: the calibration set a threshold is computed from. n = 1,103 is the bank size
  BCA reserves on hopper.
- **The paper**: Lou and Luo, Weighted Bayesian Conformal Prediction (arXiv:2604.06464).
- **BQ-CP** (Snell and Griffiths) is WBCP with uniform weights. It is today's
  uniform-weight BCA, called "uniform BCA" below.
- **RCPS**: risk-controlling prediction sets, a frequentist, deliberately conservative
  baseline.
- **lambda\***: the population threshold that exactly 10% of test rows (under the test law)
  exceed.
- **Tilt, gamma**: the shifted test law reweights the held-out pool by exp(gamma z), where
  z is a standardized feature of (s, a). gamma = 0 means no shift. The three tilts:
  - policy: closeness of the logged action to the actor's action;
  - density: sparseness, the log distance to the 10th nearest neighbour;
  - state: the norm of the observation.
- **Oracle vs estimated weights**: the exact tilt ratio, or a ratio fit by a logistic
  discriminator on the feature. "WBCP" means estimated weights unless marked oracle.
- **n_eff**: effective sample size, (sum w)^2 / sum w^2.
- **Abstain**: WBCP cannot certify any threshold and returns none. Abstentions count as
  passes.
- **Design effect D**: how much clustering inflates the variance of a bank's exceedance
  rate over independent rows.
- **rho** (intra-class correlation, ICC): the correlation between two rows of the same
  episode.
- **Normalized / raw score**: the nonconformity score with / without BCA's learned scale
  sigma(s, a).
- **K, random / stratified**: a thinned bank takes K rows from each episode. Random spacing
  puts them anywhere in the episode; stratified spacing puts one in each of K equal
  segments.
- **Host**: the offline RL algorithm BCA wraps (here TD3+BC).
- **IW**: importance-weighted.
- **Pre-registered**: written to a timestamped file before the run (see the next section).

## Artifacts and timing

- **Changes 3-5.** In `runs/wbcp_dependence/hopper-u100000/`, `predictions.json/.md`
  (computed from 14:42:52 UTC on 2026-09-30, written 14:43:09 UTC) and
  `expectations_shift.md` (14:43:35 UTC) precede every run behind Changes 3-5 (first result
  file 14:44:28 UTC).
- **Changes 6-8.** In `runs/wbcp_dependence/hopper-u100000-spacing/`,
  `predictions.json/.md` (computed from 15:12:18 UTC, written 15:12:36 UTC),
  `expectations_shift.md` (15:12:54 UTC) and
  `expectations_strat5.md` (15:22:49 UTC) precede Changes 6-8 (result files 15:13:41 to
  15:25:39 UTC). The Change 8 expectation was written after the Change 7 results and before
  the Change 8 run.
- **Change 9.** In `runs/wbcp_dependence/hopper-u100000-reservation/`, `expectations.md`
  (17:15:10 UTC, before any code; addendum 17:26:55 UTC, before any benchmark run) precedes
  the benchmark results (first result file 17:27:54 UTC). Its SHA-256 values are in
  Change 9, so they can be checked later.
- **Changes 10 and 11.** `runs/wbcp_dependence/other-datasets/expectations.md` (stages at
  18:23:07, 18:49:01 and 19:24:10 UTC, each before the data it predicts) and
  `runs/wbcp_dependence/hopper-u100000-weighting/expectations.md` (18:23:28 UTC, before
  any code or run for Change 11). Their SHA-256 values are given in each Change. The
  post-hoc analysis in Change 10 is labelled as such.
- **Not pre-registered.** The Change 1 result and the iid shift baseline
  (`runs/wbcp_bench/`, 12:08-12:10 UTC) came earlier.
- **How much the timestamps prove.** The expectation files are not under version control,
  so their timestamps are self-reported. From the next step on, expectations will be
  committed (or their hash sent to the advisor) before running.
- **Code versions.** Changes 3-5 were produced by the benchmark code as of 15:03 UTC;
  Change 1 and the iid baseline came from an earlier version, before `--per-episode`
  existed. The scripts then gained `--spacing` / `--stratified` for Changes 6-8. Rerunning two of the
  earlier experiments (`e2_k10` and `e3_k10`) with the current code reproduces them
  exactly: every failure count and mean threshold is identical
  (`runs/wbcp_dependence/repro_check/`).
- **Post-hoc scripts.** The two post-hoc analyses in Change 5 are
  `experiments/wbcp/posthoc/`.

## Summary

- **BCA's bank breaks the guarantee.** BCA reserves its calibration bank as whole episodes;
  on hopper, its 1,103-row bank is about 3 episodes. Rows inside an episode are correlated,
  but the Bayesian posterior (like the paper's theory) assumes they are not. So the bank is
  overconfident: about 28% of banks fail with no distribution shift at all, instead of at
  most 5%. RCPS is hit too (1.5% to 22%), so the problem is not specific to the Bayesian
  posterior.
- **A bigger bank of whole episodes does not help** (26-28% up to 11,739 rows). It only
  makes the threshold tighter and more confident.
- **Taking K rows from each of many episodes restores validity for small K**, close to the
  design-effect predictions made in advance: 4.9% (K = 1), 5.2% (K = 5), 5.7% (K = 10).
  K = 25 still fails 8.0%. The threshold stays at its iid value. The cost is training data,
  because every contributing episode must be withheld from training: K = 5 withholds about
  10-11% of the dataset, K = 10 about 5%.
- **Spacing the K rows apart lowers the dependence further, at no cost.** Without shift,
  stratified K = 10 fails 5.4% and stratified K = 25 fails 6.4% (random spacing: 5.7% and
  8.0%).
- **The uniform-versus-WBCP picture survives thinning under distribution shift.** But under
  the density tilt, K = 10 banks fail more often than K = 5 banks in every comparison run
  (pooled, about 1.7 points), whether or not the rows are spaced apart. So the number of
  distinct episodes matters beyond the no-shift design effect. A design effect recomputed
  under the tilt accounts for part of the excess; the rest is not explained yet.
- **Stratified K = 5 (221 episodes) met its pre-registered ranges under shift** on the
  normalized score; on the raw score one value was 0.2 points over, within noise. Uniform
  BCA still fails 26-98% under the density and state tilts. WBCP fails 5.2-8.5% (except
  state gamma 1: 0.6%, abstaining in 31% of banks), which is above 5% at density gamma 1
  and state gamma 0.5, as it is with iid banks.
- **For BCA: fix the bank before any IW comparison.** The old bank broke the guarantee for
  vanilla and importance-weighted BCA alike. The recommended design on hopper was
  stratified K = 5 from about 220 held-out episodes chosen in proportion to their length.
- **Hopper's K = 5 does not transfer (Change 10).** On walker2d-medium-replay and
  pen-cloned, rho is about 0.11-0.12, seven times hopper's. There K = 5 fails 6-8%, and the
  configured walker2d banks fail 18-30%. K has to be chosen from each dataset's measured
  rho, and on these datasets that means about one row per episode and smaller banks.
- **WBCP's excess under strong shift is small and finite-sample (Change 11).** It is
  strongest when heavy weights sit on the exceedances. A bank that misses them reports a
  spread about a third too small. On hopper with iid banks the excess shrinks with n and
  failing banks exceed alpha by under a point. The paper reports the same size of excess
  and attributes it to the same finite-sample term, which its certificate cannot bound at
  these sizes.
- **Estimated weights can hurt without any shift (Change 10).** When the score depends on
  the covariates, a weight model fit on 1,000 samples makes WBCP fail 6% at 1,024 rows and
  15% at 8,192. Ten times more fitting data brings it down to 5.7% [5.0, 6.4]; exact
  weights give 4.2%.
- **BCA now uses that bank (Change 9).** In the hopper benchmark, BCA's own sampler
  reproduces the validated behaviour (5.0% without shift; all shift ranges met). Under the
  declared rule, K is 5-6 for most TD3+BC / ReBRAC / CQL datasets. It is raised further where
  the withholding cap forces it (walker2d, pen-human, and most IQL datasets, whose banks
  are 8,192 rows). Those banks carry a `dependence_validated: false` flag.

## Change 1 (motivating observation): calibrate on whole episodes

**What changed.** The benchmark gained `--blocks`. A calibration bank is drawn as whole
episodes (uniformly, with replacement, until at least n rows) instead of n independent
rows. BCA's `reserve_calibration` draws episodes without replacement, but with about 3
episodes per 1,103-row bank, repeats are rare (about 0.3% of banks).

**Why.** BCA's `reserve_calibration` holds out whole episodes, so the benchmark's earlier
iid banks did not match how BCA actually calibrates.

**Why it makes sense.** Whole-episode reservation keeps calibration rows out of training
trajectories, which is correct, but it means the bank's rows are not independent draws. The
paper's guarantee and the posterior both treat rows as independent units, so the benchmark
must test the bank BCA really uses.

**Expectation.** This is the only experiment here run without a written expectation. The
expectation at the time was only "some loss of validity"; the size was not predicted.
Change 2 retrodicts it. Other choices were also made after seeing results: the rerun and
the analyses marked post hoc in Change 5, and the "inside the 95% interval" criterion in
Change 4.

**Result** (`runs/wbcp_bench/blocks.json` and `main.json`, 2,000 trials each). With no
shift, every Bayesian rule failed about 28% of the time: uniform BCA 28.0% [26.0, 30.0],
WBCP 28.1%. iid banks of the same size failed 4.9% [4.0, 5.9]. The frequentist RCPS rose
from 1.5% to 21.6%. So the problem is not specific to the Bayesian posterior: any rule that
treats rows as exchangeable is affected. The 4,000-trial reruns in Changes 3 and 4 give
27.8% and 4.7% (RCPS 22.0% and 1.7%).

**Met?** Not applicable (no quantitative expectation). The size of the effect prompted the
next changes.

## Change 2: measure within-episode dependence and predict from it

**What changed.** A new script, `experiments/wbcp/dependence.py`, measures the dependence
and turns it into predictions:

- The intra-episode correlation rho of the exceedance indicator I = 1{score > lambda*}, the
  quantity a miscoverage guarantee depends on, and how that correlation decays with time
  lag.
- The design effect D of each bank design, two ways: from rho (D = 1 + (m - 1) rho for m
  rows per episode) and by Monte Carlo with the benchmark's own sampler.
- A predicted failure rate, 1 - Phi(1.645 / sqrt(D)), and the share of the dataset each
  design would withhold.

**Why.** To explain the 28% with a mechanism rather than a guess, and to turn the mechanism
into numbers the next experiments can confirm or refute.

**Why it makes sense.** For clustered samples, the variance of a sample proportion is
inflated by the design effect. WBCP's posterior assumes D = 1. So the safety margin between
its beta-credible threshold and the true quantile is sized for a standard deviation sqrt(D)
times smaller than the real one. In a normal approximation the failure rate becomes
1 - Phi(1.645 / sqrt(D)) instead of 5%. The approximation makes three simplifications: it
treats the credible bound as a z-bound, it uses the indicator at the population lambda\*,
and it ignores discreteness.

A small correlation matters because episodes are long. At rho = 0.0115 across about 480
rows per episode, D is about 6.5. So a 1,362-row whole-episode bank carries about as much
information as 210 independent rows.

**Expectation.** If dependence is the mechanism, the design effect of whole-episode banks
should reproduce the 28% already observed.

**Result.** rho = 0.0155 for the normalized score (0.0196 raw), by the one-way ANOVA
estimator. The correlation between rows t and t + lag of the same episode decays with lag:

| Lag | 1 | 5 | 10 | 50 | 100 | 200 |
|---|---|---|---|---|---|---|
| Correlation | 0.19 | 0.11 | 0.075 | 0.022 | 0.007 | about 0 |

Whole-episode banks have D = 6.6 by Monte Carlo, which predicts 26.1% failure against
28.0% observed (raw: D = 8.2, predicting 28.3% against 31.8%).

Which rho feeds the formula depends on the design, because exceedance is more common in
short episodes (12.8% against 8.6%, averaged over episodes shorter / longer than the
median; 12.0% against 8.7% pooled over rows):

- The ANOVA rho (0.0155) is the right input for thinned banks, which pick episodes in
  proportion to their length.
- For whole episodes picked uniformly, the right input is the pair-weighted correlation,
  0.0115 (raw 0.0143). It gives D = 6.5 (raw 7.9), which matches the Monte Carlo value.

The first version of this note plugged the ANOVA rho into the whole-episode formula,
giving D = 8.5 and a predicted 28.6%. That was wrong, and its closeness to 28% was a
coincidence.

**Met?** Mostly, as a retrodiction:

- The mechanism accounts for about 21 of the 23 excess points. The model under-predicts by
  about 2 points: 26.1% against 28.0% [26.0, 30.0], and against 27.8% [26.4, 29.2] in the
  4,000-trial rerun. A plausible reason is that a bank of about 3 episodes is far from the
  normal approximation.
- The 28% was known when the model was written. So this shows the mechanism is large
  enough to explain the failure, not that it is the only one.
- Post hoc, anchored at RCPS's own iid failure rate, the same D values predict 20-23% for
  RCPS with whole-episode banks; observed 20.6-22.0%.

Changes 3 and 4 are the real tests.

## Change 3: whole-episode banks of increasing size

**What changed.** Whole-episode banks at 1,103, 2,300, 4,600 and 11,500 target rows (about
3, 6, 11 and 26 episodes), no shift, 4,000 trials each.

**Why.** The obvious fix is "reserve a bigger bank". The design effect depends on rows per
episode, not on the number of episodes, which predicts that a bigger bank of whole episodes
does not help. This experiment tests that prediction and rules the obvious fix in or out.

**Why it makes sense.** With whole episodes, the true sampling variance and the posterior's
assumed variance both shrink as the bank grows. But their ratio (D) stays fixed at about
6.5-6.7 (Monte Carlo), so the failure rate should not move.

**Expectation (pre-registered).** Flat at every size. `predictions.md` gives point
predictions: 28.6% (ICC) and 26.0-26.3% (Monte Carlo) for the normalized score, 30.6% and
27.9-28.3% for the raw score. The ranges used below (26-29% and 28-31%) were read from
those points. The upper ends came from the ANOVA-rho formula that Change 2 now corrects.

**Result.**

| Target rows (mean bank) | Share of dataset | Uniform BCA | WBCP | Uniform BCA, raw score | Mean threshold |
|---|---|---|---|---|---|
| 1,103 (1,362) | 0.1% | 27.8% [26.4, 29.2] | 28.0% | 31.3% [29.9, 32.8] | 1.134 |
| 2,300 (2,542) | 0.3% | 26.7% [25.3, 28.1] | 26.7% | 29.4% [28.0, 30.8] | 1.112 |
| 4,600 (4,842) | 0.5% | 26.4% [25.1, 27.8] | 26.4% | 29.1% [27.6, 30.5] | 1.099 |
| 11,500 (11,739) | 1.2% | 27.0% [25.6, 28.4] | 27.2% | 29.0% [27.5, 30.4] | 1.085 |

**Met?** Yes for the normalized score: 26.4-27.8% at every size, all inside 26-29%. Mostly
for the raw score:

- At 1,103 rows the raw score starts just above its range: 31.3% [29.9, 32.8] against
  28-31%.
- It then falls to 29.0% at 11,500 rows, a 2.4-point decline (about 2.3 standard errors).
  It is roughly flat, not exactly flat.
- The 1,103-row result was already known from Change 1, so only the three larger sizes
  test the prediction.

A bigger whole-episode bank buys nothing. As the bank grows, the threshold tightens toward
the true quantile (1.064) while the failure rate stays put: the posterior gets more
confident without getting more reliable.

## Change 4: thinned banks, K rows from each of many episodes

**What changed.** The benchmark gained `--per-episode K`. A bank of n rows makes ceil(n/K)
episode draws, with replacement and with probability proportional to episode length, and
takes K positions uniformly inside each (also with replacement). K was swept over 1, 2, 5,
10, 25, 50 and 100 at n = 1,103, no shift, 4,000 trials each.

Because draws repeat, a bank spans fewer distinct episodes than ceil(n/K): about 683 at
K = 1 (the pool has only 1,094 episodes), 199 at K = 5 and 105 at K = 10.

- **Effect on the results.** The tested banks are slightly more dependent than a real
  reservation, which would use ceil(n/K) distinct episodes, so the results are
  conservative for it.
- **Size of the effect.** It is small at K ≤ 10: D is 1.07 against 1.04 at K = 5, and 1.16
  against 1.14 at K = 10. At large K it explains the gap between the ICC and Monte Carlo
  design effects (K = 100: 2.52 without replacement, 2.75 with).
- **Predictions.** The Monte Carlo predictions use the same sampler.
- **Table columns.** "Episodes needed" in the table below is for a real bank drawn without
  replacement.

**Why.** Thinning keeps the property that matters for training (every calibration row
belongs to an episode withheld from training) while cutting the rows per episode, the m in
D = 1 + (m - 1) rho. It is a simple change to BCA's reservation that attacks the dependence
directly. Alternatives are to space the rows apart within each episode (Change 6) or to
correct the posterior's sample size for D (not tested).

**Why it makes sense.** With K rows per episode, D is about 1 + (K - 1) rho, which is close
to 1 for small K. The price is that ceil(n/K) whole episodes leave training, so K trades
validity against training data. The sweep measures that trade-off.

**A detail that matters.** Episodes are drawn with probability proportional to their
length, and rows uniformly inside each:

- Otherwise short episodes would be over-represented. The bank's rows would no longer be
  distributed like the test rows, and a failure could not be attributed to dependence
  alone.
- With this rule every calibration row is marginally uniform over the pool in every design,
  so the designs differ only in dependence. A unit test checks both stages.
- BCA's `reserve_calibration` currently picks episodes uniformly, so adopting this design
  means changing that too.

**Expectation (pre-registered).** Failure falls toward 5% as K shrinks, following the
design-effect prediction below.

**Result.** "Share withheld" assumes length-proportional selection (mean selected length
484).

| K | Episodes needed | Share withheld | Predicted (ICC / MC) | Uniform BCA | WBCP | Mean threshold |
|---|---|---|---|---|---|---|
| 1 | 1,103 | about 53% | 5.0% / 5.1% | 4.9% [4.3, 5.6] | 5.0% | 1.141 |
| 2 | 552 | 26.7% | 5.1% / 5.1% | 4.6% [3.9, 5.3] | 4.8% | 1.141 |
| 5 | 221 | 10.7% | 5.5% / 5.7% | 5.2% [4.5, 5.9] | 5.4% | 1.139 |
| 10 | 111 | 5.4% | 6.2% / 6.3% | 5.7% [5.0, 6.4] | 5.6% | 1.140 |
| 25 | 45 | 2.2% | 8.0% / 8.5% | 8.0% [7.1, 8.8] | 7.9% | 1.139 |
| 50 | 23 | 1.1% | 10.7% / 11.4% | 10.8% [9.9, 11.8] | 10.6% | 1.141 |
| 100 | 12 | 0.6% | 15.1% / 16.2% | 14.7% [13.6, 15.8] | 14.6% | 1.143 |

The iid reference at n = 1,103 is 4.7% [4.0, 5.4]. The raw score (uniform BCA) gives 4.8%,
4.5%, 5.2%, 5.7%, 8.9%, 11.8% and 17.4% across the same K.

**Met?** Largely:

- **Normalized score.** All seven ICC predictions and six of seven Monte Carlo predictions
  fall inside the observed 95% interval. The exception is K = 100 (16.2% predicted, 14.7%
  [13.6, 15.8] observed).
- **Raw score.** K = 2 and K = 10 fall just outside: 5.2% against 4.5% [3.8, 5.1], and
  6.47% and 6.59% against 5.70% [5.00, 6.46].
- **Offset.** The Monte Carlo predictions run 0.1-1.5 points high at every K. The ICC
  predictions run 0.1-0.8 points high at K ≤ 10 and sit within ±0.5 points above that.
  Part of the offset is the anchor: the formula puts D = 1 at the nominal 5%, while iid banks fail 4.7% (4.4% raw).
- **Re-anchoring (post hoc).** Re-anchored at the observed iid rate, all normalized
  predictions fall inside the interval (K = 100 Monte Carlo: 15.7%). But the ICC version
  then under-predicts the raw score at K = 25 and 100: 7.9% against 8.9% [8.0, 9.8], and
  15.9% against 17.4% [16.2, 18.6].
- **Overall accuracy.** The model is accurate to about ±1 point here. The "inside the 95%
  interval" criterion was chosen after seeing the results.
- **Band width.** The mean threshold does not change with K (1.139-1.143, the iid value).
  Thinning costs no band width, only withheld training data.

## Change 5: re-check the distribution-shift results under thinned banks

**What changed.** The shift sweep was repeated with thinned banks: policy, density and
state tilts at gamma 0.5 and 1, K = 5 and K = 10 (random spacing), n = 1,103, 2,000 trials.

**Why.** The case for importance weighting rests on the shift results measured with iid
banks: uniform BCA fails under shift, while WBCP fails far less (though above 5% at the
strongest tilts). If fixing dependence changed those results, the IW story would have to
change too.

**Why it makes sense.** A thinned bank is the design we would actually recommend, so the
shift conclusions must hold for it.

**Expectation (pre-registered, `expectations_shift.md`, quoted).** The note covered
"normalized and raw scores":

> thinning adds a design effect of about 1.07 (K = 5) or 1.15 (K = 10) on top of the iid
> benchmark, so every Bayesian arm should fail roughly 0.5-1.5 points more often than in
> the iid sweep, and the qualitative picture should not change: uniform BCA (BQ-CP) still
> fails badly under density/state shift (about 40-90%); WBCP stays near its iid values plus
> about a point: density gamma 0.5 about 5-6%, density gamma 1 about 8-9%, state gamma 0.5
> about 9%; policy shift about 5-6%. Failure would mean dependence interacts with the tilt
> (for example, if tilted mass concentrates in a few episodes), which the design-effect
> model does not capture.

**Result** (2,000 trials per cell; the iid column is `runs/wbcp_bench/main.json`).

WBCP, normalized score:

| Tilt | iid banks | K = 5 | K = 10 |
|---|---|---|---|
| policy 0.5 | 4.7% [3.8, 5.7] | 4.9% [4.0, 5.9] | 5.6% [4.6, 6.6] |
| policy 1 | 4.4% [3.5, 5.3] | 4.5% [3.6, 5.5] | 5.7% [4.7, 6.8] |
| density 0.5 | 4.2% [3.4, 5.2] | 5.7% [4.7, 6.8] | 8.2% [7.0, 9.5] |
| density 1 | 7.4% [6.2, 8.6] | 9.0% [7.7, 10.3] | 11.1% [9.7, 12.5] |
| state 0.5 | 8.0% [6.8, 9.2] | 7.8% [6.7, 9.1] | 9.6% [8.3, 11.0] |
| state 1 | 1.0% [0.6, 1.5] (abstains 31.2%) | 0.7% [0.4, 1.2] (abstains 31.9%) | 0.7% [0.4, 1.2] (abstains 32.5%) |

WBCP, raw score:

| Tilt | iid banks | K = 5 | K = 10 |
|---|---|---|---|
| policy 0.5 | 4.9% [4.0, 5.9] | 4.4% [3.5, 5.3] | 5.5% [4.5, 6.5] |
| policy 1 | 4.2% [3.4, 5.2] | 5.0% [4.0, 6.0] | 5.8% [4.8, 6.9] |
| density 0.5 | 4.5% [3.6, 5.4] | 5.6% [4.6, 6.6] | 7.7% [6.5, 8.9] |
| density 1 | 7.9% [6.8, 9.2] | 9.5% [8.3, 10.9] | 11.1% [9.7, 12.5] |
| state 0.5 | 8.5% [7.3, 9.8] | 8.0% [6.8, 9.2] | 10.7% [9.4, 12.1] |
| state 1 | 0.9% [0.5, 1.4] (abstains 29.9%) | 0.9% [0.5, 1.4] (abstains 31.4%) | 1.1% [0.7, 1.6] (abstains 31.5%) |

Uniform BCA, normalized score:

| Tilt | iid banks | K = 5 | K = 10 |
|---|---|---|---|
| policy 0.5 | 4.5% [3.6, 5.4] | 4.6% [3.7, 5.6] | 5.8% [4.8, 6.9] |
| policy 1 | 5.5% [4.5, 6.5] | 5.5% [4.5, 6.5] | 6.9% [5.8, 8.1] |
| density 0.5 | 42.6% [40.4, 44.8] | 42.8% [40.6, 45.0] | 42.4% [40.2, 44.6] |
| density 1 | 89.5% [88.1, 90.8] | 89.1% [87.6, 90.4] | 88.8% [87.3, 90.1] |
| state 0.5 | 24.9% [23.0, 26.9] | 25.3% [23.4, 27.2] | 27.7% [25.7, 29.7] |
| state 1 | 98.4% [97.7, 98.9] | 97.9% [97.1, 98.4] | 97.9% [97.1, 98.4] |

Uniform BCA, raw score:

| Tilt | iid banks | K = 5 | K = 10 |
|---|---|---|---|
| policy 0.5 | 4.9% [4.0, 5.9] | 5.4% [4.5, 6.5] | 7.4% [6.2, 8.6] |
| policy 1 | 6.2% [5.2, 7.3] | 7.6% [6.5, 8.8] | 9.4% [8.2, 10.8] |
| density 0.5 | 51.7% [49.5, 53.9] | 52.2% [50.0, 54.4] | 51.1% [48.9, 53.3] |
| density 1 | 96.2% [95.2, 96.9] | 94.9% [93.8, 95.8] | 94.2% [93.0, 95.1] |
| state 0.5 | 39.8% [37.6, 42.0] | 40.6% [38.4, 42.7] | 41.3% [39.1, 43.5] |
| state 1 | 99.9% [99.6, 100.0] | 99.7% [99.3, 99.9] | 99.8% [99.5, 99.9] |

**Met?** Partly.

**K = 5.** WBCP met or beat its ranges on both scores, with one borderline exception:

- Density: 5.7% and 9.0% (raw 5.6% and 9.5%), against 5-6% and 8-9%. Raw density 1 is
  borderline.
- Policy: 4.9% and 4.5% (raw 4.4-5.0%) came in at or below "about 5-6%".
- State 0.5: 7.8% (raw 8.0%) came in below "about 9%".

Two parts of the expectation were not met:

- The predicted 0.5-1.5-point rise over iid could not be resolved. On the normalized
  score the observed changes ranged from -0.5 to +1.6 points, and rises of 0.5 points or
  more appeared only for WBCP under density (+1.5, +1.6). On the raw score they ranged
  from -1.3 to +1.6 points, with rises of at least 1 point for WBCP under density (+1.1,
  +1.6) and for uniform BCA under policy gamma 1 (+1.4). The 95% interval on a
  2,000-trial rate is about ±1 point; the expectation should have stated a Monte Carlo
  tolerance.
- The 40-90% range for uniform BCA was wrong for state shift (25% at gamma 0.5, 98% at
  gamma 1) and for raw-score density gamma 1 (94.9%), as it already was with iid banks.
  The reading "fails badly" holds.

**K = 10.** For the normalized score, WBCP met its ranges under the policy and state
tilts, but not under the density tilt: 8.2% and 11.1%, against 5-6% and 8-9%. The
predicted rise over iid (0.5-1.5 points) was exceeded in several arms: WBCP rose 4.0 and
3.7 points under density and 1.65 under state 0.5, and uniform BCA rose 2.8 points under
state 0.5. The raw score, which the expectation also covered, missed more widely:

- WBCP: 7.7% and 11.1% under density (iid 4.5% and 7.9%), and 10.7% under state 0.5 (iid
  8.5%).
- Uniform BCA under the policy tilt rose 2.5-3.2 points (7.4% and 9.4%, against iid 4.9%
  and 6.2%).

**Rerun (post hoc).** After seeing the density miss, we reran the density tilt with a
different random stream (oracle-weight WBCP, `posthoc/dominant_episode.py`, seed 11). The
rerun also built the density feature with a different k-NN seed (0 instead of the
benchmark's 20260930), so it is a slightly different tilt, and the pooled figures mix two
nearly identical tilts. Failure rates at gamma 0.5 / 1:

| Bank | Rerun | Pooled with the main run's oracle-weight values (4,000 banks) |
|---|---|---|
| iid | 4.6% / 8.0% | |
| K = 5 | 6.0% / 8.9% | 5.7% / 8.9% |
| K = 10 | 6.5% / 9.6% | 7.3% / 10.2% |

In the rerun alone, the K = 10 excess is small (0.5 and 0.7 points). Pooled, it is about
2-3 standard errors. Changes 7 and 8 later replicated it with a
pre-registered, 4,000-trial pair (stratified K = 10 against stratified K = 5).

**Why not (post hoc, not pre-registered).**

1. **The tilt feature is episode-clustered, which raises the design effect under the
   tilt.**
   - The density feature's intra-episode correlation is 0.050, and 0.051 with the
     benchmark's k-NN seed (the analysis used seed 0, the function default). That is
     3.4-3.9 times the policy and state features (0.013-0.015).
   - The quantity whose variance matters under this tilt is w (I - R), the weighted miss
     term. It is about 1.5 times as clustered as without shift (ICC 0.022-0.024 against
     0.0155; computed in an independent review).
   - Recomputing the design effect under the tilt gives about 1.20 at K = 10, against
     1.15 without shift (`posthoc_tilted_design_effect.json`; a 40,000-bank check with the
     benchmark's feature gave 1.205 and 1.196 ± 0.012). That moves the prediction to 5.9%
     and 9.4%, part of the way.
   - At K = 5 the same recomputation (6,000 banks) gave 1.10 and 1.03, and below 1 for
     state gamma 1, which positive correlation cannot produce. So those K = 5 entries are
     noise-dominated.
2. **A "dominant episode" explanation was tested and contradicted.** The explanation was
   that failures come from banks dominated by one heavy episode, one version of the
   concentration failure the expectation named. But failures are most common when the
   largest single episode holds the smallest share of the bank's weight. For K = 10 at
   gamma 1, banks in the lowest quartile of that share fail 16.4%, against 3.2% in the
   highest (500 banks per quartile). The same gradient appears with iid banks (11.4%
   against 5.0%) and with K = 5 (12.6% against 6.2%). So it is largely a generic property
   of weighted calibration under shift, only steeper at K = 10.
3. **Remaining hypothesis (untested at this point).** Failing banks miss the few sparse,
   high-risk episodes, and a K = 10 bank (about 105 distinct episodes, against 199 at
   K = 5) misses them more often. This is another version of an episode-level effect.
   Change 7 tests one of its implications; the direct test (failure rate against how many
   high-risk episodes each bank contains) has not been run.

## Change 6: space the K rows apart inside each episode

**What changed.** The benchmark gained `--spacing stratified`. Each drawn episode is cut
into K equal segments and one row is drawn uniformly inside each, so rows sit about L/K
steps apart (L is the episode length).

- **Episode draws are unchanged** (with replacement, in proportion to length). So every row
  is still marginally uniform over the pool, and a stratified bank spans the same number of
  distinct episodes as a random one with the same K.
- **Predictions.** `dependence.py --stratified` gives the Monte Carlo design effect for
  these banks. The ICC formula averages over random pairs, so it does not apply.
- **Sweep.** K = 2, 5, 10, 25 and 50 at n = 1,103, no shift, 4,000 trials each.

**Why.** Change 2 found the within-episode correlation is mostly short-range: 0.19 at lag
1, 0.022 at lag 50, about 0 by lag 200. Random positions put some rows close together, and
those close pairs carry most of the dependence. Spacing the rows apart should lower D at
the same K, that is, at the same number of withheld episodes, which is what costs training
data.

**Why it makes sense.** It changes only where inside an episode the rows come from, not
which episodes leave training, so it is free. Its limit is also clear in advance: it cannot
reduce dependence that is shared by a whole episode (an episode-level effect), because rows
of one episode share it whatever their spacing.

**Expectation (pre-registered, `hopper-u100000-spacing/predictions.md`, written 15:12:36
UTC).**

- Monte Carlo predictions for stratified banks at K = 2, 5, 10, 25 and 50: 5.1%, 5.1%,
  5.7%, 7.3% and 10.1% for the normalized score (raw 5.1%, 5.4%, 5.7%, 7.9% and 11.2%).
  Random banks in the same file: 5.1%, 5.6%, 6.4%, 8.4% and 11.4%.
- That implies no difference at K = 2, a small gain at K = 5-10, and a gain of about 1
  point at K = 25-50.
- `expectations_shift.md` (15:12:54 UTC) added: "Without shift, stratified K = 10 should
  fail about 5.7% (random K = 10: 6.4%)."

**Result** (uniform BCA; WBCP is within 0.4 points of it in every cell; random observed
values are from Change 4).

| K | Episodes | Random: predicted | Random: observed | Stratified: predicted | Stratified: observed |
|---|---|---|---|---|---|
| 2 | 552 | 5.1% | 4.6% [3.9, 5.3] | 5.1% | 4.4% [3.8, 5.1] |
| 5 | 221 | 5.6% | 5.2% [4.5, 5.9] | 5.1% | 5.2% [4.5, 5.9] |
| 10 | 111 | 6.4% | 5.7% [5.0, 6.4] | 5.7% | 5.4% [4.7, 6.1] |
| 25 | 45 | 8.4% | 8.0% [7.1, 8.8] | 7.3% | 6.4% [5.7, 7.2] |
| 50 | 23 | 11.4% | 10.8% [9.9, 11.8] | 10.1% | 9.2% [8.3, 10.2] |

Raw score, stratified: 4.4%, 5.1%, 5.3%, 6.7% and 10.1% (random: 4.5%, 5.2%, 5.7%, 8.9% and
11.8%). The mean threshold is unchanged (1.139-1.141).

**Met?** Mostly:

- **Direction and growth with K came out as predicted.**
  - Stratified banks fail less than random ones at K = 25 (6.4% against 8.0%, about 2.7
    standard errors) and at K = 50 (9.2% against 10.8%, about 2.4 standard errors).
  - At K = 10 the gain is 0.3 points, inside the noise. At K = 2 and 5 there is none.
  - Stratified K = 10 (5.4% [4.7, 6.1]) met its explicit 5.7% expectation.
- **The predicted small gain at K = 5 did not show:** random 5.15%, stratified 5.20%,
  against a predicted 5.6% to 5.1%. Random K = 5 came in about 0.5 points under its
  prediction.
- **As in Change 4, the predictions run high.** For the normalized score, 4 of 5 fall
  inside the observed 95% interval (K = 25 does not: 7.3% against 6.4% [5.7, 7.2]). For
  the raw score only K = 5 and 10 do. Re-anchored post hoc at the observed iid rate (4.7%;
  raw 4.4%), all ten fall inside.

Practical reading: without shift, stratified K = 10 (111 episodes, about 5% of the data) is
as valid as random K = 5 (221 episodes, about 10-11%).

## Change 7: spaced rows under shift at K = 10, a test of the episode-level explanation

**What changed.** The Change 5 shift sweep was run with stratified K = 10, plus random
K = 10 at the same seed as a paired control. Both used 4,000 trials, twice the Change 5
count.

**Why.** There were two reasons:

- **Practical.** If stratified K = 10 held under shift, BCA could withhold half as many
  episodes as with K = 5.
- **Diagnostic.** It tests an implication of the episode-level explanation for the K = 10
  density miss. Spacing removes short-range dependence but not episode-level dependence,
  and it does not add distinct episodes. If the miss is an episode-level effect, spacing
  should not fix it.

**Why it makes sense.** The test could fail in a useful direction. If stratified K = 10
had recovered the K = 5 result under the density tilt, the episode-level explanation would
have been wrong or incomplete, and the cheaper design would have been acceptable.

**Expectation (pre-registered, `hopper-u100000-spacing/expectations_shift.md`, 15:12:54
UTC).**

- Without shift, stratified K = 10 should fail about 5.7% (random K = 10: 6.4%).
- Under the density tilt, stratified K = 10 should not recover the K = 5 result. WBCP
  should fail about 6.5-8% at gamma 0.5 and 9-11% at gamma 1, at most about 1 point better
  than random K = 10.
- Under the policy and state tilts, stratified K = 10 should behave like random K = 5: WBCP
  about 4.5-5.5% (policy) and 7.5-9% (state 0.5).
- "If stratified K = 10 fixes the density case, the rare-episode explanation is wrong or
  incomplete."

**Result** (4,000 trials per cell, same seed for both designs).

| Tilt | WBCP, random K = 10 | WBCP, stratified K = 10 | Uniform BCA, random K = 10 | Uniform BCA, stratified K = 10 |
|---|---|---|---|---|
| policy 0.5 | 5.7% [5.0, 6.5] | 5.7% [5.0, 6.4] | 5.3% [4.6, 6.0] | 5.3% [4.6, 6.0] |
| policy 1 | 5.4% [4.7, 6.1] | 5.2% [4.5, 5.9] | 6.1% [5.4, 6.9] | 6.3% [5.5, 7.0] |
| density 0.5 | 7.1% [6.3, 7.9] | 6.7% [5.9, 7.5] | 41.4% [39.8, 42.9] | 42.6% [41.1, 44.2] |
| density 1 | 10.4% [9.5, 11.4] | 10.1% [9.1, 11.0] | 88.3% [87.2, 89.2] | 89.3% [88.3, 90.2] |
| state 0.5 | 8.5% [7.7, 9.4] | 8.3% [7.4, 9.2] | 25.5% [24.2, 26.9] | 25.9% [24.6, 27.3] |
| state 1 | 0.8% [0.5, 1.1] (abstains 32.9%) | 0.7% [0.4, 1.0] (abstains 32.6%) | 98.1% [97.7, 98.5] | 98.3% [97.9, 98.7] |

WBCP, raw score, random / stratified:

| Tilt | Random K = 10 | Stratified K = 10 |
|---|---|---|
| policy 0.5 | 5.7% | 5.7% |
| policy 1 | 5.9% | 4.9% |
| density 0.5 | 7.8% | 7.5% |
| density 1 | 9.9% | 10.4% |
| state 0.5 | 8.1% | 8.9% |

**Met?** Yes on the part that matters:

- **Density.**
  - WBCP failed 6.7% [5.9, 7.5] and 10.1% [9.1, 11.0], inside 6.5-8% and 9-11%.
  - That is 0.4 and 0.3 points better than random K = 10 at the same seed; the expectation
    allowed at most about 1 point.
  - On the raw score, stratified was no better (7.5% and 10.4%, against 7.8% and 9.9%).
  - Spacing did not fix the density case, so the episode-level explanation survived a test
    it could have failed.
- **Policy and state.**
  - State 0.5 (8.3%) and policy gamma 1 (5.2%) are inside their ranges.
  - Policy gamma 0.5 is 5.7% [5.0, 6.4], 0.2 points above the range. The same-seed random
    K = 10 also gives 5.7% there, so this is not a spacing effect.
- **Uniform BCA still fails badly:** 42.6% and 89.3% under density, 25.9% and 98.3% under
  state.
- **Without shift:** stratified K = 10 failed 5.4% (Change 6), against the expected 5.7%.

**What it shows, and what it does not.**

- **The K = 10 density excess is now replicated.** Together with Change 8 (same seed,
  4,000 trials), it is a pre-registered result rather than a one-run finding. Stratified
  K = 10 fails 6.7% and 10.1%, against 5.4% and 8.4% for stratified K = 5 (about 2.5
  standard errors each).
- **Every comparison points the same way.** K = 10 banks fail more than K = 5 under the
  density tilt at gamma 0.5 / 1:
  - Change 5: +2.5 / +2.1 points.
  - Post-hoc rerun (oracle weights): +0.5 / +0.7.
  - Changes 7-8: +1.3 / +1.7.
  - Pooled over 10,000 K = 10 and 6,000 K = 5 banks (estimated weights): about 1.7 / 1.8.
- **The excess is episode-level.** Spacing removes only about 0.4 points of it. So the
  excess goes with how many distinct episodes the bank holds, not with rows sitting close
  together.
- **The exact mechanism is still open.** "Sparse, high-risk episodes" is one version of an
  episode-level effect. Its direct test has not been run.

## Change 8: the candidate design, stratified K = 5, under shift

**What changed.** The shift sweep was run with stratified K = 5 (221 episode draws), 4,000
trials, same seed as Change 7.

**Why.** Change 7 ruled out stratified K = 10 as the cheaper design under shift, which
leaves K = 5. Random K = 5 met Change 5's WBCP ranges, but on 2,000 trials and without spacing. Before
a design is recommended for the IW study, it has to be run as recommended.

**Why it makes sense.** Stratified K = 5 withholds the same number of episodes as random
K = 5, with lower predicted dependence (D 1.01 against 1.07), so it should do at least as
well. The run checks that spacing does not interact badly with the tilts.

**Expectation (pre-registered, `expectations_strat5.md`, 15:22:49 UTC, quoted).**

> it has the distinct-episode count of random K = 5 (which met all shift expectations) and
> no-shift validity at least as good (5.2% vs 5.1%), so WBCP should match random K = 5
> within noise: density about 5-6% (gamma 0.5) and 8-9% (gamma 1), state 0.5 about 7.5-9%,
> policy about 4.5-5.5%; uniform BCA still fails badly under density and state shift.

The parenthesis overstated Change 5 (see its Met?). The ranges were updated from the
observed random K = 5 values, which is why they differ slightly from Change 5's.

**Result** (stratified K = 5: 4,000 trials; random K = 5 from Change 5: 2,000 trials).

| Tilt | WBCP, stratified K = 5 | WBCP, random K = 5 | WBCP raw, stratified K = 5 | Uniform BCA, stratified K = 5 |
|---|---|---|---|---|
| policy 0.5 | 5.3% [4.6, 6.0] | 4.9% [4.0, 5.9] | 5.5% [4.8, 6.2] | 4.8% [4.2, 5.5] |
| policy 1 | 5.2% [4.5, 5.9] | 4.5% [3.6, 5.5] | 5.7% [5.0, 6.5] | 6.0% [5.3, 6.8] |
| density 0.5 | 5.4% [4.7, 6.1] | 5.7% [4.7, 6.8] | 5.6% [4.9, 6.4] | 42.1% [40.6, 43.7] |
| density 1 | 8.4% [7.5, 9.3] | 9.0% [7.7, 10.3] | 8.4% [7.6, 9.3] | 89.4% [88.4, 90.3] |
| state 0.5 | 8.5% [7.7, 9.4] | 7.8% [6.7, 9.1] | 8.3% [7.4, 9.2] | 26.1% [24.7, 27.4] |
| state 1 | 0.6% [0.4, 0.9] (abstains 31.4%) | 0.7% [0.4, 1.2] (abstains 31.9%) | 0.8% [0.5, 1.1] (abstains 33.1%) | 98.4% [98.0, 98.8] |

**Met?** Yes:

- **Normalized score.** Every WBCP value is inside its stated range: 5.3% and 5.2%
  (policy), 5.4% and 8.4% (density), 8.5% (state 0.5).
- **Uniform BCA still fails badly:** 42.1% and 89.4% under density, 26.1% and 98.4% under
  state.
- **Raw score.** All values are inside the ranges except policy gamma 1, at 5.7% [5.0,
  6.5]: 0.2 points above 5.5%, within noise.
- **Against random K = 5,** the differences are within noise. The largest are 0.7 points,
  with stratified worse, at state 0.5 (8.5% against 7.8%) and policy gamma 1 (5.2% against
  4.5%); at density gamma 1 stratified is 0.6 points better (8.4% against 9.0%). At K = 5, spacing makes no measurable difference on hopper,
  with or without shift.

WBCP stays above 5% at density gamma 1 (8.4% [7.5, 9.3]) and state gamma 0.5 (8.5% [7.7,
9.4]), as it does with iid banks (7.4% and 8.0%). Oracle weights show the same (8.2% and
8.0%). So this is not a bank problem. Change 11 explains it.

## Change 9: implement the thinned bank in BCA

**What changed.** BCA's reservation now builds the Change 8 design instead of whole
episodes.

- **Sampler.** A new NumPy-only module, `calibration/bank.py`, holds `stratified_bank`,
  used by both BCA and the benchmark:
  - It chooses m = ceil(n/K) distinct episodes, with inclusion probability exactly
    m L / N. The episodes are laid end to end in a random order, and the episode under each
    of m equally spaced points is withheld (systematic sampling).
  - It takes one uniform row from each of K equal segments of every withheld episode, or
    every row when L ≤ K.
- **Reservation.** `calibration/reference.py` `reserve_calibration` returns training,
  withheld and calibration rows separately:
  - Every row of a withheld episode leaves training; only the K thinned rows are
    calibrated.
  - The withholding cap now counts all withheld rows.
  - The metadata records K, the number of withheld episodes, the withheld share and a flag,
    `dependence_validated`. It is true when K ≤ 10 and at least 100 episodes are withheld,
    the range this study validated without shift.
- **Hosts.** In TD3+BC, ReBRAC, CQL and IQL, the bank is the calibration rows. Each host's
  partition and leakage checks now test training against all withheld rows. Where a host
  re-verifies the reservation, it now re-derives the new selection.
- **Configs.** Every dataset declares `rows_per_episode`, and the config loader refuses a
  dataset that does not.
- **Tools.** The benchmark gained `--spacing reservation` (BCA's own sampler), and
  `dependence.py` gained `--reservation`.
- **Population split kept.** `rows_per_episode=None` keeps the old whole-episode split,
  only for `freeze_scores.py`'s population split, so existing frozen pools still
  reproduce.

**Why.** Changes 1-8 showed that BCA's whole-episode bank fails about 28% of the time.
Stratified K = 5 from length-weighted episodes was the design that passed both without
shift and under all six tilts. Validating a design in the benchmark means little until BCA
actually uses it.

**Why it makes sense.** Four design choices, each forced by a requirement:

1. **Distinct episodes, exactly proportional to length.** A real reservation cannot repeat
   an episode; the benchmark's sampler could. Exact length-proportional inclusion keeps
   every calibration row marginally distributed like the data, and systematic sampling
   gives both properties in a few lines. Its one limit is that no episode may be longer
   than N / m; the sampler raises an error if one is.
2. **Rotating the segment grid.** Segment edges fall on whole rows, so segments differ in
   size by one row. Without rotation, rows in shorter segments would be picked slightly
   more often (1/18 against 1/19 on a 93-step episode). A uniform rotation of the grid per
   episode makes every row's probability exactly K m / N. This differs from the
   benchmark's stratified sampler (continuous positions), and both are exactly uniform.
3. **The cap counts withheld rows.** The cap protects training data, and a withheld
   episode's unused rows are lost to training just like its calibration rows.
4. **K per dataset, by a declared rule.** Where K = 5 does not fit a config's cap, K is
   raised until it fits. That trades some validity for bank size; the alternative was to
   shrink the bank. On pen-human K is raised anyway and the bank is flagged. The rule: K is
   the smallest integer ≥ 5 whose reservation fits the cap for every one of 1,000
   simulated seeds (`choose_rows_per_episode.py`).

**Expectation (pre-registered, `runs/wbcp_dependence/hopper-u100000-reservation/expectations.md`).**
The file was written at 17:15:10 UTC, before any code for this change (SHA-256
`12c0ac27d6edc6b5e003dadec6aeaf779288fe152e885b1705af71081a842aec`). An addendum at
17:26:55 UTC followed, after the unit tests and before any benchmark run (file SHA-256
with the addendum: `566034c5238635459b2c23d2a5030f0cfaa2c78e6ffb3951c0c5a7587e4e98e3`).

1. The unit tests should show the exact properties: the partition, the thinned subset,
   min(K, L) rows per episode, inclusion probability m L / N, and a per-row probability of
   K m / N.
2. The rule's K values should be roughly:
   - TD3+BC / ReBRAC / CQL: halfcheetah 6, hopper 5-6, maze2d and pen 5, walker2d about
     21, pen-human 124.
   - IQL: halfcheetah about 19, hopper about 18, maze2d 5, pen-cloned about 11,
     pen-expert about 8, walker2d about 64, pen-human 199.
   - Flags false for walker2d and pen-human everywhere, and for IQL halfcheetah, hopper
     and pen-cloned.
3. BCA's sampler in the hopper benchmark (K = 5, n = 1,103) should fail within noise of
   the stratified results (Changes 6 and 8) or lower: 4.2-5.9% without shift, and the
   Change 8 WBCP ranges under shift.
4. IQL's raised K should cost validity: about 5.5-7% at K = 18, n = 8,192.

The addendum records a problem noticed before running Expectations 3 and 4. BCA's sampler
draws distinct episodes from the benchmark's finite pool of 1,094 episodes. A bank that
covers a large share of that pool varies less around the pool's risk than a real bank
varies around a new test law, which makes the benchmark optimistic.

- At K = 5 the effect is small: the Monte Carlo design effect is 0.99, predicting 4.9%.
- At K = 18, n = 8,192, where a bank covers 44% of the pool, BCA's sampler predicts 5.7%.
  The with-replacement stratified sampler, which overstates dependence by repeating
  episodes, predicts 6.6%.
- The addendum set ranges of 4.9-6.5% and 5.8-7.4% for those two runs, with a real
  deployment in between.

**Result.**

1. **Tests and real-data checks.** All pass.
   - `test_bank` (6) checks the exact probabilities with z-tests over 6,000-30,000 seeds.
   - `test_reference` (14) checks the partition, the thinning, the cap, determinism and the
     population-split mode.
   - The host tests (TD3+BC 12, ReBRAC 12, CQL 6, IQL 12) check each host: training and
     withheld partition the data, no withheld row trains, and the bank is a strict subset
     with K rows per episode. They also check that the host's re-verification rejects an
     altered reservation.
   - The config tests (4) check that every dataset's K reaches its host and that a dataset
     without K is refused. The remaining suites (benchmark 23, WBCP 15, freeze_scores 12)
     also pass: 116 tests in total.
   - Real data: all 28 host-dataset configurations prepare on the actual D4RL files, and
     the withheld shares equal the rule's values below.
   - `freeze_scores.py` still reproduces the frozen hopper pool row for row (500,285 rows).
   - `check.py` and `check.py --runtime` accept. With the new K fields removed, the
     280-run experiment matrix hashes to the previously recorded value, so K is the only
     change to the matrix.
2. **Config values** (`rows_per_episode.json`; withheld share at the configured seed
   911; ✗ = `dependence_validated` false):

| Dataset | TD3+BC / ReBRAC / CQL (1,024 rows, cap 10%) | IQL (8,192 rows, cap 25%) |
|---|---|---|
| halfcheetah-medium-expert | K = 6: 171 episodes, 8.6% withheld | K = 17: 482 episodes, 24.1% ✗ |
| hopper-medium | K = 6: 171 episodes, 8.3% | K = 17: 482 episodes, 23.3% ✗ |
| maze2d-large | K = 5: 205 episodes, 1.6% | K = 5: 1,639 episodes, 12.6% |
| pen-cloned | K = 5: 205 episodes, 6.1% | K = 11: 745 episodes, 22.2% ✗ |
| pen-expert | K = 5: 205 episodes, 4.1% | K = 7: 1,171 episodes, 23.5% |
| pen-human | K = 124: 2 episodes, 8.0% ✗ | K = 199 (whole episodes): 6 episodes, 24.0% ✗ |
| walker2d-medium-replay | K = 23: 45 episodes, 7.0% ✗ | K = 67: 123 episodes, 21.2% ✗ |

3. **BCA's sampler in the hopper benchmark** (K = 5, n = 1,103, 4,000 trials; the bank
   holds 1,105 rows):

| Tilt | WBCP, BCA sampler | WBCP, stratified (Changes 6, 8) | Uniform BCA, BCA sampler |
|---|---|---|---|
| none | 5.0% [4.3, 5.7] | 5.1% [4.4, 5.8] | 5.1% [4.4, 5.8] |
| policy 0.5 | 4.8% [4.1, 5.5] | 5.3% [4.6, 6.0] | 4.4% [3.8, 5.1] |
| policy 1 | 4.6% [3.9, 5.3] | 5.2% [4.5, 5.9] | 5.1% [4.5, 5.9] |
| density 0.5 | 5.5% [4.8, 6.3] | 5.4% [4.7, 6.1] | 41.3% [39.7, 42.8] |
| density 1 | 8.6% [7.8, 9.5] | 8.4% [7.5, 9.3] | 89.6% [88.6, 90.6] |
| state 0.5 | 7.9% [7.1, 8.8] | 8.5% [7.7, 9.4] | 24.2% [22.8, 25.5] |
| state 1 | 0.8% [0.5, 1.1] (abstains 31.5%) | 0.6% [0.4, 0.9] (abstains 31.4%) | 98.6% [98.1, 98.9] |

Raw score, BCA sampler (WBCP): 5.0% with no shift, 4.5% and 4.6% under policy, 5.8% and
9.1% under density, 9.0% under state 0.5.

4. **IQL's raised K on hopper** (K = 18, n = 8,192, no shift, 4,000 trials):
   - BCA's sampler: WBCP 4.9% [4.2, 5.6], uniform 5.0% [4.4, 5.7] (raw 5.1%).
   - The stratified sampler with replacement: WBCP 6.8% [6.0, 7.6], uniform 6.7% [5.9, 7.5]
     (raw 7.0-7.2%).
   - Independent rows fail 4.7% [4.0, 5.4] for comparison.

**Met?** Mostly:

1. **Yes.** Every test passes, and the real-data checks agree with the simulated rule.
2. **Mostly.**
   - Every K is within 2 of the rough expectation. Walker2d came out at 23 and 67 (expected
     about 21 and 64); IQL halfcheetah and hopper at 17 (expected about 19 and 18);
     pen-expert at 7 (expected about 8).
   - The expectation estimated K from 90% of the cap. The rule simulates the actual
     sampler against the full cap, which lands slightly lower where episode lengths are
     uniform and slightly higher on walker2d, whose long episodes make the withheld share
     variable.
   - Every flag matched.
3. **Yes.**
   - Without shift: 5.0% and 5.1%, inside 4.2-5.9%.
   - Under shift, every normalized WBCP value is inside its range. Uniform BCA still fails
     24-99% under density and state.
   - On the raw score, density gamma 1 is 9.1% [8.2, 10.0], 0.1 points above 8-9%, within
     noise of the stratified 8.4%.
   - The implemented bank behaves like the design the benchmark validated.
4. **Yes for both addendum ranges, but the answer to the question is a range.**
   - BCA's sampler failed 5.0%, inside 4.9-6.5%; the stratified sampler failed 6.7%,
     inside 5.8-7.4%.
   - The BCA-sampler run does not separate from independent rows. The stratified run
     does, by about 2 points.
   - Reading: on hopper, IQL's K = 17-18 bank fails somewhere between 5% and 7% in a real
     deployment. The benchmark cannot narrow that further, because its pool is only half
     of hopper. The original expectation (5.5-7%) is consistent with that range but was not
     sharply tested.

**What this does not cover.**

- The rule's K values outside hopper are chosen to fit the cap, not validated. No dataset
  other than hopper has a measured rho or a frozen score pool.
- walker2d (K = 23 and 67) and pen-human carry `dependence_validated: false` in their
  metadata. So do IQL halfcheetah, hopper and pen-cloned. Their calibration claims should
  be read as unverified.

## Change 10: dependence on two more datasets

**What changed.** Frozen score pools for two more datasets, built with the same recipe as
hopper (`freeze_scores.py`: TD3+BC with BCA's scale fit, 100k updates on an episode half,
scored on the other half; 6 and 11 minutes of CPU):

- walker2d-medium-replay: 151,195 rows in 528 episodes.
- pen-cloned: 248,165 rows in 1,887 episodes.

On each pool the step measured the dependence (`dependence.py`) and benchmarked the
configured designs (`run_benchmarks.sh` in the run directory). Both samplers were run
because of the finite-pool effect (Change 9).

**Why.**

- walker2d-medium-replay is, among the datasets with enough episodes for a thinned bank,
  the one whose K the withholding cap forced up most (only pen-human was forced higher).
- pen-cloned has both a validated configuration (K = 5 for TD3+BC, ReBRAC and CQL) and a
  flagged one (IQL, K = 11).
- Change 9's K values outside hopper were chosen to fit the caps, not validated, and K = 5
  itself rested on hopper's rho.

**Why it makes sense.** The design effect is 1 + (K - 1) rho, and rho is a property of the
dataset. Measuring rho on two contrasting datasets tests whether hopper's answer
transfers.

**Expectation (pre-registered, `runs/wbcp_dependence/other-datasets/expectations.md`).**
The file was written in stages, each before the data it predicts. Hashes after each
stage: A `91858219…785d`, B `ae716ef7…74dd`, D `add84dee…44c6`.

- **Stage A** (18:23:07 UTC, before the pools): rho above hopper's 0.016, about 0.02-0.10
  on both datasets, with the failure rates that implies. Whole-episode banks fail at least
  as badly as on hopper.
- **Stage B** (18:49:01 UTC, after the pools, before any benchmark): Monte Carlo
  predictions for each design, with each thinned design expected within about ±2 points of
  its prediction (±5 at very large D). Even K = 5 should fail about 8-9%. Under shift, the
  configured TD3+BC-family design should fail at least as often as without shift, and
  uniform BCA should still fail badly under density and state. The stage covered both
  scores; the raw-score results fall within the same bands and are not tabulated here.
- **Stage D** (19:24:10 UTC): a test of an unplanned observation, described in the Result.

**Result.**

*Dependence* (normalized score; raw score in brackets):

| Dataset | rho | Lag correlation at 1 / 50 / 200 steps |
|---|---|---|
| hopper-medium (Change 2) | 0.016 | 0.19 / 0.022 / about 0 |
| walker2d-medium-replay | 0.120 (0.109) | 0.30 / 0.15 / 0.17 |
| pen-cloned | 0.108 (0.133) | 0.27 / 0.087 / 0.008 at 100 |

On walker2d the correlation never decays: episodes differ as whole units, as expected of
a replay buffer that mixes policies from different stages of training.

*Designs without shift* (uniform BCA, which is what the Monte Carlo predicts; 4,000
trials; "BCA" and "strat." are the two samplers):

| Pool, design | Predicted (BCA / strat.) | Uniform BCA, BCA sampler | Uniform BCA, stratified |
|---|---|---|---|
| walker2d, whole episodes, 1,024 rows | 42.6% | | 48.0% [46.4, 49.6] |
| walker2d, K = 5, 1,024 rows | - / 8.4% | | 7.5% [6.7, 8.4] |
| walker2d, K = 23 (TD3-family config) | 17.0% / 19.2% | 17.7% [16.5, 18.9] | 19.5% [18.2, 20.7] |
| walker2d, K = 67 (IQL config), 8,192 rows | 20.3% / 28.8% | 23.4% [22.1, 24.7] | 29.7% [28.3, 31.2] |
| pen-cloned, whole episodes, 1,024 rows | 33.0% | | 34.7% [33.2, 36.2] |
| pen-cloned, K = 5 (TD3-family config) | 7.7% / 8.2% | 6.3% [5.5, 7.1] | 7.0% [6.3, 7.9] |
| pen-cloned, K = 11 (IQL config), 8,192 rows | 9.3% / 12.2% | 9.5% [8.6, 10.4] | 13.0% [11.9, 14.0] |

WBCP with exact weights matches uniform BCA in every one of these (within 0.4 points).

*Configured TD3-family design under shift* (BCA sampler, WBCP with exact weights; uniform
BCA in brackets):

| Tilt | walker2d, K = 23 (no shift: 17.7%) | pen-cloned, K = 5 (no shift: 6.1%) |
|---|---|---|
| policy 0.5 | 9.5% (0.0%) | 5.4% (0.0%) |
| policy 1 | 6.1% (0.0%) | 5.3% (0.0%) |
| density 0.5 | 11.2% (85.7%) | 6.9% (73.8%) |
| density 1 | 10.3% (100%) | 6.4% (98.4%) |
| state 0.5 | 17.7% (99.3%) | 9.8% (98.6%) |
| state 1 | abstains in every bank | abstains in every bank |

*An unplanned observation, then tested (Stage D).* On both datasets WBCP with estimated
weights fails more than with exact weights even without shift:

- 6.2% against 4.6% (walker2d) and 6.3% against 4.6% (pen-cloned) at n = 1,024.
- At n = 8,192: 14.9% against 4.2%, and 15.7% against 4.9%.

The estimated weights are nearly uniform (n_eff 8,175 of 8,192). The hypothesis written
before the test was Theorem 4's product term: a small weight error matters in proportion
to how much the score depends on the covariates. On walker2d the policy tilt moves lambda\*
from 1.26 to 0.97, where on hopper it barely moved (1.064 to 1.066). As n grows, the
posterior concentrates on the slightly wrong risk curve.

The test fit the weight model on 10× the samples (walker2d, n = 8,192, no shift):

- Estimated-weight WBCP fell from 14.9% to 5.7% [5.0, 6.4]; the prediction was 5-9%.
- Exact weights stayed at 4.2%.

**Met?** Partly. The dependence results match their predictions; the shift expectation
did not hold.

- **Stage A: mostly.**
  - rho is above hopper's on both datasets, as expected, but at 0.120 and 0.108 it lies
    just above the pre-registered 0.02-0.10.
  - The implied failure ranges hold for the BCA-sampler results. Three stratified results
    fall above them: walker2d K = 23 at 19.5% (range 8-18%), K = 67 at 29.7% (14-28%), and
    pen-cloned K = 11 at 13.0% (6.5-12%).
  - Whole-episode banks fail more than on hopper (48% and 35%).
- **Stage B, designs without shift: mostly.**
  - Every thinned design with K ≤ 23 is within ±2 points of its prediction for the same
    sampler.
  - walker2d K = 67 with the BCA sampler is 3.1 points above its prediction, inside the ±5
    allowed for large D.
  - walker2d whole episodes is 5.4 points above its prediction, just outside ±5.
  - The claim that "even K = 5 fails about 8-9%" missed in most cells. Uniform BCA and
    exact-weight WBCP fail 6.1-7.7%: above 5%, but below the stated range. Estimated-weight
    WBCP fails 7.7-9.1%, partly from the weight-noise effect below.
- **Stage B, shift: no on walker2d, mixed on pen-cloned.** Expected: WBCP fails at least
  as often under shift as without.
  - walker2d: under the policy and density tilts it fails less (6.1-11.2% against 17.7%).
    Only the state tilt matches the no-shift rate.
  - pen-cloned: the policy tilt fails less (5.3-5.4% against 6.1%). Density (6.4-6.9%) and
    state (9.8%) are at or above it.
  - Uniform BCA still fails badly under density and state (74-100%), as expected.
  - With estimated weights, the state tilt is much worse on pen-cloned: 16.7% against 9.8%
    with exact weights. Elsewhere the two arms are within 1.3 points.
- **Stage D: yes.**

**Why the shift expectation failed (post hoc,
`experiments/wbcp/posthoc/tilted_rho.py`).**

- Under a tilt, the dependence that matters is that of the weighted miss term w (I - R),
  not of I. On walker2d its episode correlation drops from 0.120 to 0.020-0.052 under the
  policy and density tilts: the tilts move weight toward rows whose exceedances are less
  tied to their episode.
- The resulting design effects predict 8.5-13.1% against 6.1-11.2% observed. The
  direction is right, and the predictions run a little high, as expected with the BCA
  sampler on a finite pool.
- The state tilt keeps more of the correlation (0.055). Its prediction, 13.4%, falls short
  of the observed 17.7%; a possible reason is that it also carries Change 11's heavy-weight
  effect, which was not tested here.
- On pen-cloned the same computation gives 6.5-8.5% against 5.3-6.9% observed for policy
  and density, and 8.0% against 9.8% for state.
- The pre-registered expectation assumed the no-shift dependence carries over unchanged;
  it does not.

**What this means.**

1. **K = 5 does not transfer from hopper.** With rho around 0.11-0.12, keeping the design
   effect near hopper's 1.07 needs K of about 1 to 2.
   - On walker2d even that is limited: its dependence is episode-level, so spacing cannot
     help, and only the number of distinct episodes counts.
   - Under the configs' caps that means much smaller banks. At one row each, about 46
     episodes (walker2d) or 316 (pen-cloned) fit the 10% cap for every one of 1,000
     simulated seeds. The alternative is the posterior correction n / D (Change 4,
     untested).
   - pen-cloned's K = 5 banks still carry `dependence_validated: true`, although they fail
     6-7% here. The flag's rule rests on hopper's rho and must become rho-based too.
   - K must be chosen from each dataset's measured rho, and the current rule (raise K
     until the cap fits) is not validated on these datasets.
2. **Estimated weights may need a much larger fit set on these datasets.** Otherwise WBCP
   can be worse than uniform BCA even without shift, and more so the larger the bank. The
   likely reason is that the score is not locally adaptive there. That is inferred from how
   far lambda\* moves under the policy tilt; adaptivity was not varied directly, and the
   10× test is a single cell. This matters for the IW study: the weight model's sample size (and
   the score's locality) must be part of the design.

## Change 11: why WBCP exceeds 5% under strong shift

**What changed.** Four tools, and no change to WBCP itself:

- The benchmark now reports the 95th and 99th percentiles of realized risk, and the mean
  excess over alpha among failing banks. Until now failure was counted but never sized.
- `--shuffle-tilt SEED` is a control. It reassigns a tilt's weights to pool rows by a fixed
  permutation, which keeps the same weights and n_eff but removes any link to the score.
- `weighted_mechanism.py` gives a per-bank view. It evaluates the WBCP posterior at the
  true threshold lambda\* for each benchmark bank, using the benchmark's own random
  streams, so every bank and threshold is the benchmark's.
- `certificate_slack.py` computes the slack eta_n of the paper's Theorem 4 for our weights.

**Why.** WBCP fails 7-9% at density gamma 1 and state gamma 0.5 with every bank design,
including iid banks and oracle weights (Changes 5, 8 and 9; 8.6% with BCA's own bank). The IW study will rely on
WBCP, so it has to be known first whether this is a defect, an artifact of the benchmark,
or behaviour the method allows.

**Why it makes sense.** Three questions, each with its own test:

1. What does the paper promise? Checked against its theorems.
2. Is the excess finite-sample or systematic? A finite-sample effect shrinks as the bank
   grows; a bias does not.
3. What drives it? A control that keeps the weights but breaks their alignment with the
   score, and a per-bank look at the posterior.

**Expectation (pre-registered, `runs/wbcp_dependence/hopper-u100000-weighting/expectations.md`,
written 18:23:28-18:24:12 UTC, before the first code change and result, SHA-256 `ee65c8458807ef4c580fc1f3e2ed188a58faabc631c2f077f9cd9c98a97bdf73`).**
Written before any code or run:

- **Theory.** Theorem 4's slack exceeds alpha here, so 8% failure breaks no theorem.
- **Hypothesis H.** Under a tilt aligned with the score, a few heavy-weight exceedances
  dominate the risk. A bank holding few of them reports both a low risk and a too-small
  posterior spread, and certifies too low a threshold. On average the spread is right; the
  error is finite-sample and shrinks about as 1 / sqrt(n).
- **A.** Oracle-WBCP failure falls with n, to about 5.9% (density gamma 1) and 6.1% (state
  gamma 0.5) at n = 8,824, each within about ±0.8 points of a stated path. The policy tilt
  stays at 4.2-5.8%. Exceedances are small: at n = 1,103, a 95th percentile of about
  9.8-10.3%, a 99th of about 10.3-11.2%, and a mean excess among failing banks of about
  0.3-1.0 points. All shrink with n.
- **B.** With shuffled weights, 4.0-6.0% failure and lambda\* at the uniform value.
- **C.** (i) The failure rate read off the posterior at lambda\* matches the benchmark's.
  (ii) Across-bank SD over average posterior SD is 0.9-1.1. (iii) Among failing banks the
  posterior SD is at most 0.8 of the across-bank SD, and the correlation between a bank's
  estimate and its posterior SD is clearly stronger than in the two controls.

**Result.**

*What the paper promises* (Lou and Luo, §3.2-3.3 and Appendix B):

- Corollary 3 (beta-credibility conditional on the bank) is a statement about the
  posterior.
- §1: under weighting, "credibility no longer coincides with confidence". Remark 1: the
  identity "breaks twice" under weighting; it holds only for uniform weights, which is why
  BQ-CP's ~5% without shift is exact.
- The frequentist statement is Theorem 4: with probability 1 - delta, the deployed risk is
  at most alpha + eta_n (oracle weights, pure covariate shift).
- For our weights, eta_n (Bernstein form, delta = 0.05, n = 1,103) is:

| Tilt | Largest weight ÷ mean weight (B) | eta_n |
|---|---|---|
| none | 1.0 | 0.32 |
| policy 1 | 3.1 | 0.42 |
| density 0.5 | 6.5 | 0.44 |
| density 1 | 33 | 1.19 |
| state 0.5 | 39 | 1.17 |

  The slack exceeds alpha = 0.1 in every case, even without shift, so the certificate says
  nothing at these sizes. The paper's "certifies at the target failure rate" under shift is
  an empirical finding from its own experiments, not a theorem.

*A. Bank size* (iid banks, normalized score, 4,000 trials per cell, oracle weights):

| Tilt | n = 1,103 | 2,206 | 4,412 | 8,824 |
|---|---|---|---|---|
| density 1 | 7.6% [6.8, 8.4] | 6.3% [5.5, 7.1] | 6.7% [6.0, 7.5] | 6.0% [5.3, 6.8] |
| state 0.5 | 7.9% [7.1, 8.8] | 7.0% [6.2, 7.8] | 6.7% [6.0, 7.5] | 6.0% [5.3, 6.8] |
| policy 1 (control) | 4.8% [4.2, 5.5] | 5.1% [4.4, 5.8] | 5.2% [4.5, 5.9] | 4.9% [4.3, 5.6] |

How far failing banks exceed alpha (oracle weights):

| Tilt | n | 99th percentile of realized risk | Mean excess among failing banks |
|---|---|---|---|
| density 1 | 1,103 → 8,824 | 11.5% → 10.5% | 0.76 → 0.29 points |
| state 0.5 | 1,103 → 8,824 | 11.3% → 10.4% | 0.68 → 0.23 points |
| policy 1 | 1,103 → 8,824 | 10.8% → 10.3% | 0.47 → 0.18 points |

Estimated weights track the oracle within noise up to n = 4,412. At n = 8,824 they are
higher (density 6.7%, state 6.5%), which suggests weight-estimation error no longer
vanishes against the shrinking finite-sample term.

*B. Shuffled weights* (density gamma 1 weights on random rows, n = 1,103):

- Oracle WBCP fails 5.7% [5.0, 6.5], against 7.6% with the same weights aligned.
- lambda\* = 1.061, near the uniform value of 1.064.

*C. Per bank* (n = 1,103, oracle weights, 4,000 banks and 4,000 posterior draws each):

| | Aligned density 1 | Shuffled | Policy 1 |
|---|---|---|---|
| Failure (benchmark) | 7.6% | 5.7% | 4.8% |
| Failure read off the posterior at lambda\* | 7.5% | 5.8% | 4.8% |
| Across-bank SD ÷ average posterior SD | 1.03 | 1.03 | 1.01 |
| Failing banks: posterior SD ÷ across-bank SD (median) | 0.63 | 0.70 | 0.87 |
| Correlation of estimate and posterior SD | 0.81 | 0.80 | 0.93 |
| Skewness of the bank estimate | 0.40 | 0.31 | 0.16 |

**Met?** Mostly:

- **Theory: yes.** eta_n > alpha in every case.
- **A: mostly.**
  - All twelve oracle failure rates are inside their pre-registered ranges.
  - Excess over 5% at n = 1,103 and 8,824 (points): density 2.6 and 1.0, state 2.9 and 1.0;
    policy stays near 5%. That is consistent with 1 / sqrt(8) and rejects S (no
    shrinkage). The rate itself is loosely pinned: at 8,824 the excess is 1.0 point with a
    95% interval of about 0.3-1.8.
  - The mean excess among failing banks is inside its range and shrinks at about the same
    rate.
  - Two magnitude predictions were slightly too low at n = 1,103: the 95th percentile of
    realized risk was 10.37% (predicted at most 10.3%) and the 99th was 11.46% (at most
    11.2%).
- **B: yes.** 5.7%, inside 4.0-6.0% though near the top.
- **C: partly.**
  - (i) Yes: 99.3% of banks agree.
  - (ii) Yes.
  - (iii) Half. Failing banks' posterior SD is 0.63 of the true spread, well under 0.8. But
    the correlation between a bank's estimate and its own spread is not stronger in the
    aligned case (0.81, against 0.80 and 0.93).

**Why not (iii), second half.** The correlation is generic: a miss rate's variance grows
with the rate itself, as for any binomial proportion, so the correlation is high in every
case. What separates the cases is how far the spread collapses in the banks that fail
(0.63, 0.70, 0.87), and that follows how heavy the weights are and how much they sit on
exceedances.

**Explanation.**

- **The mechanism.** WBCP's posterior has the right spread on average. But a bank that
  happens to hold few of the heavy-weight exceedances reports both a low risk and a spread
  about a third too small, so it certifies too tight a threshold. This is a one-sided,
  Wald-type undercoverage of a skewed, self-normalized weighted estimate.
- **Why only these tilts.** It is strongest when heavy weights sit on the exceedances
  (density and state tilts, largest weight 33-39× the mean). The same weights on random
  rows give 5.7%; a mild, unaligned tilt gives 4.8%.
- **It is finite-sample.** At 8× the bank size the excess falls to about 1 point, and the
  exceedances fall with it. The shrinkage is clear; its exact rate is not pinned down.
- **It is small in size here.** On hopper with iid banks, failing banks exceed 10% by
  0.2-0.8 points on average, and the mean realized risk stays well below alpha (7.2-9.1%
  for oracle WBCP under the aligned tilts). With dependent banks it is larger: walker2d's
  configured bank under the state tilt fails 17.7% with a mean excess of 1.3 points
  (Change 10).
- **Not a bug, and the paper reports the same.** WBCP behaves as its theory allows, and
  its frequentist certificate is loose at these sizes.
  - In the paper's §4.2 experiment (an unbounded exponential tilt), WBCP "certifies every
    trial at 7.5-7.9%". At n = 250 that is 7.9% with estimated weights and 5.6% with
    oracle weights, and the authors attribute the residual to eta_n.
  - Its published Table 1 oracle row is 6.7%. Our reproduction of Table 1, whose tilt has a
    largest weight about 4× the mean, gave 4.9%.
  - What this step adds is the mechanism, the dependence on bank size, and the role of
    alignment with the score.

## What this means for BCA

- **Vanilla BCA's shift failure persists with the recommended bank, in this benchmark.**
  - With a stratified K = 5 bank, uniform BCA fails 42% and 89% under the density tilt and
    26% under the state tilt (gamma 0.5).
  - WBCP fails 5.2-8.5% against a target of at most 5% (state gamma 1: 0.6%, abstaining
    in 31% of banks). That is far better, but above target under strong density and
    moderate state shift, as it already was with iid banks. Oracle weights share this.
  - This supports importance weighting when the weights are well specified. With a
    misspecified (raw-feature) discriminator, WBCP failed 17% and 36% under the density
    tilt and 17% and 28% under the state tilt (normalized score,
    `runs/wbcp_bench/rawdisc.json`).
- **There is a second limitation that weighting cannot fix.**
  - Calibrating on whole episodes, as BCA does, makes its nominal 95% credibility hold only
    about 72% of the time on hopper-medium, even without shift.
  - It hits both arms. With whole-episode banks, WBCP fails 21-33% under the policy,
    density and state tilts (gamma 0.5 and 1, except state gamma 1, where it fails 2.2%
    and abstains in 25% of banks).
  - Uniform BCA's density-shift failure drops from 89.5% to 75%, because the extra variance
    sometimes lands on a conservative threshold (`runs/wbcp_bench/blocks.json`).
  - An IW comparison on the current bank would mix the two failures. Fix the bank first.
- **Recommended bank design (hopper-medium evidence), now implemented (Change 9).**
  - Draw ceil(n/5) distinct held-out episodes with probability proportional to their
    length, and take one row uniformly from each fifth of each episode (stratified K = 5).
  - At n = 1,103 that is 221 episodes, about 10-11% of hopper-medium withheld from
    training.
  - It is the only design tested that was both valid without shift (5.2%) and inside its
    pre-registered ranges under shift (normalized score; on the raw score one value was 0.2
    points over, within noise). Stratified K = 10 also met its own expectation, but that
    expectation was that it would fail under the density tilt.
  - BCA's configs reserve 1,024 rows under a 10% cap for TD3+BC, ReBRAC and CQL. On hopper
    that gives K = 6 (171 episodes, 8.3% withheld), because K = 5 would sit at the cap.
    That is still inside the validated range.
- **The cheaper option and why it is not recommended.** Stratified K = 10 (111 episodes,
  about 5% withheld) is valid without shift (5.4%) but fails 6.7% and 10.1% under the
  density tilt (Change 7). It would be enough only if no-shift validity were all that
  mattered.
- **Other datasets.**
  - K = 5 follows from hopper's rho of 0.016. Change 10 measured rho of 0.12 on
    walker2d-medium-replay and 0.11 on pen-cloned, where K = 5 fails 6-8%. Choose K from each
    dataset's measured rho (`dependence.py`) so that D stays near 1.07. That is about one
    row per episode on these two datasets, so the bank size is set by how many episodes the
    cap allows.
  - Then check the choice under a density tilt, since that is where K = 10 failed here.
  - Episode counts and caps limit what is possible. On pen-human (25 episodes) the cap
    allows 2 episodes (TD3+BC / ReBRAC / CQL) or 6 (IQL), so its bank is nearly whole
    episodes and flagged. walker2d-medium-replay's long episodes force K = 23 (67 for
    IQL).
  - Change 10 did this for walker2d and pen-cloned. The rule "raise K until the cap fits"
    fails there: walker2d's configured banks fail 18% (K = 23) and 23-30% (K = 67). The rule
    should be replaced by one based on measured rho, with smaller banks where needed.
- **For the IW study.** WBCP's weights must be fit on enough data when the score is not
  locally adaptive. Otherwise the weight model's own noise makes WBCP fail more than
  uniform BCA even without shift (Change 10). Under strong, score-aligned shift, hopper
  with iid banks gives about 92-94% frequentist credibility with exceedances under a point
  (Change 11). BCA's own hopper bank gives 91% (Change 9). Dependent banks on other
  datasets do worse: walker2d's configured bank under the state tilt gives 82%, with a mean
  excess of 1.3 points (Change 10).
- **Untested alternatives and costs.**
  - Keep whole episodes and shrink the posterior's sample size to n / D. This costs band
    width instead of training data, and D has to be estimated per dataset.
  - The training cost of withholding 10-11% of episodes has not been measured. The model
    here is frozen and trained on a fixed half.

## Limits of this evidence

- **Scope.** Three datasets (hopper for Changes 1-9 and 11; walker2d-medium-replay and
  pen-cloned for Change 10), one host (TD3+BC), one frozen 100k-update model per dataset,
  one seed per experiment.
- **Frozen scores.** The adaptive loop, where the same bank is rescored at every refresh,
  is not tested.
- **Sampling with replacement.** The benchmark draws episodes and rows with replacement.
  This is conservative relative to a real reservation, and the effect is small at K ≤ 10.
- **The design-effect model.** It is a normal approximation, and it is rough with about 3
  episodes per whole-episode bank. Under the density tilt it under-predicts the K = 10
  failure even when recomputed under the tilt.
- **Frequency and size.** Changes 1-9 count only how often a bank's miscoverage exceeds
  10%. Changes 10 and 11 also record by how much.
- **Training cost.** The training cost of withholding episodes was not measured.
- **Finite pool.** The benchmark's pool is half of hopper. For BCA's distinct-episode
  sampler that makes the benchmark slightly optimistic, noticeably so when a bank covers a
  large share of the pool (IQL-size banks; Change 9).
- **Other datasets.** Only hopper, walker2d-medium-replay and pen-cloned have score pools,
  all from the TD3+BC host. The other four datasets and the other three hosts are untested.
- **Change 11 is on hopper only**, with its tilts; the mechanism is shown for oracle weights.
- **WBCP above target.** WBCP is above 5% at density gamma 1 and state gamma 0.5 with every
  bank design, including iid, and with oracle weights. Change 11 explains it on hopper; it
  is not corrected.
- **Post-hoc analyses.** The Change 5 analyses, including the rerun's failure rates, used
  a density feature built with a different k-NN seed from the benchmark (0 instead of
  20260930). An independent recomputation with the benchmark's feature confirmed the
  K = 10 design effects; the quartile analysis and the rerun were not recomputed.

## Appendix: supporting changes to the benchmark tooling

Each change was prompted by an automated review of the benchmark code (independent Claude
agents). The first five were made before the Change 1 run. `main.json` and `blocks.json`
record three of them directly (the log density transform, the unpenalized feature
discriminator and `risk_certified`); the other two leave no trace in the artifacts. The
last two were made while building Changes 4 and 6.

| Change | Why | Expectation | Result | Met? |
|---|---|---|---|---|
| Density tilt uses log k-NN distance | The raw distance is heavy-tailed (9-17 SD), so exp(gamma z) put almost all test mass on a handful of rows (n_eff 0.9% of the pool at gamma 1) | Tilts usable up to gamma 1 | n_eff 78% (gamma 0.5) and 35% (gamma 1) of the pool | Yes |
| Feature discriminator unpenalized (ridge 0; raw keeps 1e-3) | The ridge shrank the well-specified slope (1.98 instead of 2.0 at gamma 2), biasing estimated weights low | Estimated and oracle WBCP agree | Density gamma 1: 7.35% estimated vs 7.55% oracle | Yes |
| One BLAS thread per process (the script re-executes itself) | Forked workers oversubscribed the CPU (a load of about 130 on 24 cores, observed but not saved). A site hook imports NumPy at startup, so setting the variables inside the script had no effect | Much faster, same results | The raw-discriminator sweep runs at 0.94 s per trial (`rawdisc.json`); feature-discriminator sweeps run at 0.009-0.08 s per trial. No before/after comparison was saved | Faster, after a first attempt that did not work; "same results" was not checked |
| `risk_certified` reported next to `risk` | Mean risk counted abstentions as 0, unlike `reproduce_table1.py` | Comparable numbers | Both conventions reported | Yes |
| NaN thresholds rejected | A NaN would silently count as a pass | Loud failure | Unit-tested | Yes |
| Per-episode sampler test rewritten | The first test used a chi-square on row counts. Counts are over-dispersed because a draw adds K rows of one episode, so the test failed even for a correct sampler (p = 1e-28) | Valid test | Tests episode draws and within-episode positions separately; passes | Yes |
| Stratified sampler test fixed | The first version assumed segment boundaries at integer multiples of L/K | Valid test | Checks floor(jL/K) ≤ step ≤ floor((j+1)L/K); passes | Yes |

## Reproduce

Run from the BCA root. Each benchmark command also takes `--output <new file>` (it refuses
to overwrite). The default seed is 2026093001.

```bash
F=runs/wbcp_frozen/hopper-medium-v2-s202609171-u100000
python experiments/wbcp/dependence.py --frozen $F --output runs/wbcp_dependence/<new>          # predictions, Changes 2-5
python experiments/wbcp/dependence.py --frozen $F --output runs/wbcp_dependence/<new> \
  --block-sizes 1103 --per-episode 2 5 10 25 50 --stratified 2 5 10 25 50                       # predictions, Change 6
B="python experiments/wbcp/d4rl_benchmark.py --frozen $F --score both --workers 16"
$B --tilt policy density state --gamma 0 0.5 1 2 --n 200 1103 --trials 2000                     # iid shift baseline (main.json)
$B --tilt policy density state --gamma 0 0.5 1 --n 1103 --trials 2000 --blocks                   # Change 1 (blocks.json)
$B --tilt policy density state --gamma 0.5 1 --n 1103 --trials 2000 --discriminator raw         # misspecified weights (rawdisc.json)
$B --tilt policy --gamma 0 --n 1103 --trials 4000                                                # iid reference (e0)
$B --tilt policy --gamma 0 --n 1103 2300 4600 11500 --trials 4000 --blocks                       # Change 3 (e1)
$B --tilt policy --gamma 0 --n 1103 --trials 4000 --per-episode K                                # Change 4 (e2), K = 1 ... 100
$B --tilt policy density state --gamma 0.5 1 --n 1103 --trials 2000 --per-episode K              # Change 5 (e3), K = 5, 10
$B --tilt policy --gamma 0 --n 1103 --trials 4000 --per-episode K --spacing stratified           # Change 6 (e6), K = 2 ... 50
S="--tilt policy density state --gamma 0.5 1 --n 1103 --trials 4000 --seed 2026093002"
$B $S --per-episode 10 --spacing stratified                                                      # Change 7 (e7_stratified_k10)
$B $S --per-episode 10 --spacing random                                                          # Change 7 control (e7_random_k10)
$B $S --per-episode 5 --spacing stratified                                                       # Change 8 (e8_stratified_k5)
# The post-hoc scripts write fixed paths and refuse to overwrite: move the existing JSONs first.
R=runs/wbcp_dependence/hopper-u100000-reservation
python experiments/wbcp/choose_rows_per_episode.py --output $R/rows_per_episode.json          # Change 9 config K
python experiments/wbcp/dependence.py --frozen $F --output $R/predictions_k5 --n 1103 --block-sizes 1103 \n  --per-episode 5 --stratified 5 --reservation 5                                              # Change 9 predictions
python experiments/wbcp/dependence.py --frozen $F --output $R/predictions_k18 --n 8192 --block-sizes 8192 \n  --per-episode 18 --stratified 18 --reservation 18
$B --tilt policy --gamma 0 --n 1103 --trials 4000 --per-episode 5 --spacing reservation          # r1
$B $S --per-episode 5 --spacing reservation                                                      # r2
$B --tilt policy --gamma 0 --n 8192 --trials 4000 --per-episode 18 --spacing reservation         # r3
$B --tilt policy --gamma 0 --n 8192 --trials 4000 --per-episode 18 --spacing stratified          # r4
python experiments/wbcp/posthoc/tilted_design_effect.py    # posthoc_tilted_design_effect.json (seed 7)
python experiments/wbcp/posthoc/dominant_episode.py        # posthoc_dominant_episode.json (seed 11), the rerun
# Change 10
for d in walker2d pen-cloned; do
  python experiments/wbcp/freeze_scores.py --dataset $d --seed 202609171 --updates 100000 \
    --output runs/wbcp_frozen/$d-s202609171-u100000; done
O=runs/wbcp_dependence/other-datasets
python experiments/wbcp/dependence.py --frozen runs/wbcp_frozen/walker2d-s202609171-u100000 --output $O/predictions_walker2d_n1024 \
  --n 1024 --block-sizes 1024 --per-episode 5 23 --stratified 5 23 --reservation 23   # and n8192 / pen, see the run directory
bash $O/run_benchmarks.sh                                                              # Stage C runs
$B --frozen runs/wbcp_frozen/walker2d-s202609171-u100000 --tilt policy --gamma 0 --n 8192 --trials 4000 \
  --weight-fit 10000 --test-mass-size 10000                                            # Stage D
python experiments/wbcp/posthoc/tilted_rho.py --frozen runs/wbcp_frozen/walker2d-s202609171-u100000 --k 23 --output ...
# Change 11 (normalized score, 4,000 trials; W=runs/wbcp_dependence/hopper-u100000-weighting)
$B --tilt density --gamma 1 --n 1103 2206 4412 8824 --score normalized --trials 4000     # also state 0.5, policy 1
$B --tilt density --gamma 1 --n 1103 --score normalized --trials 4000 --shuffle-tilt 20261001
python experiments/wbcp/weighted_mechanism.py --frozen $F --tilt density --gamma 1 --output $W/c_density1.json
python experiments/wbcp/certificate_slack.py --frozen $F --output $W/certificate_slack.json
```

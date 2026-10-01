# Weighted Bayesian conformal prediction (WBCP)

[calibration/wbcp.py](../../calibration/wbcp.py) implements Algorithm 1 of Lou and Luo,
*Weighted Bayesian Conformal Prediction*, arXiv:2604.06464v3, for the miscoverage loss:
nonconformity scores, normalized weighted Bayesian bootstrap masses with one extra test
atom (the exact posterior of a Dirichlet-process prior pushed through the likelihood
ratio), the beta-credible crossing, clamped at the weighted empirical quantile. With
uniform weights it is BQ-CP (Snell and Griffiths, 2025). BCA consumes it through
[calibration/reference.py](../../calibration/reference.py) at every refresh.

`test_wbcp.py`, `reproduce_table1.py` and `reproduce_table2.py` import no hosts. The `test_*_host.py` suites drive
each host runtime with synthetic data and stub environments; nothing here trains a full run
or steps a simulator.

## Checks

```bash
python -m unittest experiments.wbcp.test_wbcp        # NumPy + SciPy only
JAX_PLATFORMS=cpu python -m unittest experiments.wbcp.test_reference experiments.test_wbcp_configs
JAX_PLATFORMS=cpu python -m unittest experiments.wbcp.test_td3_bc_host experiments.wbcp.test_rebrac_host experiments.wbcp.test_cql_host experiments.wbcp.test_iql_host
python experiments/wbcp/reproduce_table1.py --gamma 1 # about two minutes on a laptop CPU
python -m unittest experiments.wbcp.test_reproduce_table2  # count-loss posterior, exact checks
```

`test_reference.py` covers the frozen reference, the dose readout and the bank
certifiability gate. The host suites check native behavior before the first refresh,
an eager refresh that matches an independent WBCP recomputation, consumption of the
frozen threshold, invalid-refresh aborts, checkpoints and, for TD3+BC and ReBRAC,
bitwise host/BCA pairing before the first refresh.

`test_wbcp.py` checks exact identities rather than smoke behavior: the crossing law
under uniform weights equals the Beta law of Dirichlet(1,...,1) partial sums; a single
weighted atom matches its closed form; the exponential sampler matches the tilted
Dirichlet definition (Theorem 6); Eq. (1) keeps exact ties; Eq. (7) is the smallest
beta-credible grid point by direct count; the clamp binds when it should; weight scale
and input order cancel; huge weights do not overflow. During development an independent
implementation written from the paper text alone (not committed) agreed with this module bit
for bit on 1,400 random cases.

## Reproduction of Table 1 (section 4.1)

Synthetic heteroskedastic regression, X ~ U[0,4], Y | X ~ N(0, X^2), score |Y|,
n = 200, alpha = 0.1, beta = 0.95, test covariates tilted by exp(x) (this tilt gives the
paper's shortest valid length 10.6), weights from a logistic discriminator on covariate
samples disjoint from the scored calibration set, 10,000 trials, M = 1000 draws:

| Rule | Failure frequency [95% CI] | Mean risk | Mean length | Paper |
|---|---|---|---|---|
| BQ-CP | 92.8% [92.2, 93.3] | 14.6% | 9.30 | 92.3% / 14.4% / 9.35 |
| RCPS | 85.7% [85.0, 86.3] | 13.5% | 9.62 | 85.2% / 13.3% / 9.67 |
| W-CRC | 41.2% [40.2, 42.2] | 9.6% | 10.95 | 42.8% / 9.5% / 10.98 |
| **WBCP** | **5.1% [4.7, 5.5]** | **4.9%** | **13.44** | **5.0% / 5.0% / 13.41** |
| WBCP (oracle w), Eq. (6) mass | 4.9% [4.5, 5.4] | 4.9% | 13.42 | 6.7% / 5.4% / 13.09 |
| WBCP (oracle w), test mass 1 | 7.0% [6.5, 7.5] | 5.4% | 13.06 | 6.7% / 5.4% / 13.09 |

Two published numbers do not follow from the paper's own definitions. The oracle row is
reproduced only with test-atom mass 1 = E_cal[w*], not E_test[w*] = 2.07 as Eq. (6)
specifies. The exchangeable control's 2.6% failure (page 7) is below what BQ-CP can
achieve at n = 200: the exact infinite-draw value is 3.20% (3.7% with 1000 draws). The
settings the paper does not state (|Cw| = |Tw| = |Ts| = 200, M = 1000) did not move the
WBCP row materially in sensitivity runs.

With a locally adaptive score (|Y|/X here), estimated weights become harmless: WBCP
fails 3.5% (no shift) and 3.3% (shift) of trials, the either-arm property of the
paper's Theorem 4. This is why BCA keeps its learned scale in the score.

## Reproduction of Table 2 (section 4.2)

`reproduce_table2.py` adds the count loss that calibration/wbcp.py does not cover. x ~ N(0,1);
each unit has K = 4 outcomes T_k | x ~ Exp(rate(x)); its loss is the share of its outcomes
above lambda. alpha = 0.4, beta = 0.95, test covariates N(gamma, 1), 10,000 trials, M = 1000
draws. `test_reproduce_table2.py` checks the posterior against `calibrate` bit for bit at
K = 1, and against a literal evaluation of Eqs. (6)-(7) at K > 1.

The paper does not state rate(x). With rate(x) = exp(b - a x), only the shift-blind rows are
used to fit it (BQ-CP and RCPS see calibration data alone). Their mean-threshold ratios point to
a = 0.69-0.70 and their n = 10 failure rates to about 0.67. We use a = ln 2 and b = -0.019, so the
blind rows are fits, not tests. Every W-CRC and WBCP number below is a prediction. W-CRC uses the
plug-in test mass wbar (the paper's section 3.1). Appendix C's literal CRC Prop. 2 form, with the
test point's own weight, fails about 33% at n = 10, far from the paper's row.

Results at gamma = 1 (paper in brackets):

| Rule | n = 10 fail | n = 10 abstain | n = 250 fail | n = 250 mean lambda |
|---|---|---|---|---|
| BQ-CP (fitted) | 29.4% (28.2%) | 0% (0%) | 100% (100%) | 1.05 (1.05) |
| RCPS, published bound | 2.4% (5.7%) | 0% | 100% (100%) | 1.17 (1.16) |
| RCPS, floor(n R_hat) (fitted) | 6.2% (5.7%) | 0% | 100% (100%) | 1.16 (1.16) |
| W-CRC | 11.8% (13.2%) | 3.3% (4%) | 46.3% (43.7%) | 1.94 (1.92) |
| **WBCP** | **0.0% (0.0%)** | **91.8% (93%)** | **10.2% (7.9%)** | **2.42 (2.36)** |
| WBCP, samples of 1,000 for the classifier and test mass | 0.0% | 93.2% | 7.6% (7.9%) | 2.36 (2.36) |
| WBCP (oracle w), Eq. (6) mass | 0.0% (0.0%) | 93.3% (about 96%) | 7.0% (5.6%) | 2.37 (2.35) |
| WBCP (oracle w), test mass 1 | 1.4% (0.0%) | 9.2% (about 96%) | 9.2% (5.6%) | 2.33 (2.35) |

The paper's oracle abstention at n = 10 is not printed. "About 96%" is what its interval [0.0, 0.9]
for 0 failures implies (about 410 certified trials).

**What reproduces.**
- At n = 10, WBCP abstains instead of failing: 91.8% abstention and no failures, against the paper's
  93% and none.
- W-CRC at n = 10 is close: 11.8% failing and 3.3% abstaining, against 13.2% and 4%.
- At n = 100 and 250, WBCP certifies every trial, failing 7.8% and 10.2% (paper: 7.5-7.9%).
- The blind rules fail in essentially every trial by n = 100 (BQ-CP 100%, RCPS 99.95%). That
  follows from the fit.
- At gamma = 2, BQ-CP fails 92-100% with risk up to 0.74, as published.

**Deviations.**
- **n = 250 does not reproduce quantitatively at the pre-registered settings.** WBCP fails 10.2%
  [9.6, 10.8] against 7.9% [7.4, 8.5]. W-CRC (46.3% against 43.7%) and the oracle row (7.0% against
  5.6%) also fall outside the paper's intervals.
  - About 1.4 points of WBCP's 2.3-point gap appear in the exact-weight row too. The oracle row
    stays at 6.5-7.2% under every slope and sample size tried, and the cause is not identified.
  - With samples of 1,000 for the classifier and the test mass, WBCP fails 7.6%, inside the
    paper's interval. W-CRC and the oracle row do not move closer, so this does not show that the
    paper used a larger classifier.
- **RCPS rounding.** The paper's RCPS numbers need floor(n R_hat) in the Bentkus term. The published
  Hoeffding-Bentkus bound uses ceil, which gives 2.4% and a mean lambda of 4.22 at n = 10. The two
  agree for Table 1's miscoverage loss.
- **Oracle test-atom mass.**
  - Table 2's oracle row rules out mass 1. Mass 1 certifies 91% of n = 10 trials and fails 1.4% of
    them.
  - The row is closer to the Eq. (6) mass E_test[w*] but not matched by it. Eq. (6) abstains 93.3%;
    abstention does not depend on rate(x), and about 96% needs a mass near 3.0.
  - Table 1's oracle row matches mass 1 on failure, risk and length.
- **Mean lambda at n = 10.** Over the few certifying trials it is 11.2, against 7.61. This
  conditional mean depends on the tail of T, which the fit does not pin down.
- **Appendix C's other regimes.**
  - At gamma = 2, WBCP abstains in 100%, 97% and 55% of trials at n = 10, 100 and 250 (paper:
    every trial up to n = 100, 75% at n = 250). It fails 0.4% of certified trials at n = 250
    (paper: never).
  - At gamma = 0, BQ-CP fails 0.4%, 4.0% and 3.9%, against the paper's 0.3-2.2%.
- **Estimated weights with no shift.** At gamma = 0, WBCP with estimated weights fails 0.6%, 6.7% and
  11.3%. The classifier's noise slope tilts a non-pivotal score's risk curve, and the posterior
  concentrates on the tilted curve as n grows. This is the walker2d mechanism of Change 10.

The full scorecard against pre-registered expectations, the gamma sweep and the sensitivity
runs (slope 0.65 / 0.75, classifier samples 1,000) are in section 9 of `results.ipynb`.

```bash
python experiments/wbcp/reproduce_table2.py --n 250   # about eight minutes; --n 10 takes under a minute
```

## What changed relative to the archived BCA radius

The archived radius (tag `bca-bayesmax-archive`, `calibration/posterior.py`) took the
maximum of a finite-rank conformal radius and a Bayesian bootstrap quantile drawn over
the n observed scores only. Without the test atom its nominal 95% credibility is really
88.8% at n = 50, 94.3% at n = 200 and 93.9% at n = 1000; exchangeable failure rates were
9.7% / 6.8% / 5.6% against 3.4% / 3.6% / 4.4% for WBCP, and it certified every trial even
where the published rule abstains. Adding only the atom closed the gap.

## Semi-synthetic benchmark on D4RL

`freeze_scores.py` trains TD3+BC with BCA's scale fit on half of a D4RL dataset's episodes
(through the repo's own preparation, training and scoring code) and freezes one score per
transition of the other half. `d4rl_benchmark.py` then resamples that fixed pool under
known tilts of (s, a) and scores each rule against exact ground truth: oracle ratio
a / mean(a), oracle test mass mean(a^2) / mean(a)^2, realized risk as a tilt-weighted tail
sum over the pool. Resampling never changes a row's own reward, next state or score, so
the shift is purely covariate. `--checkpoint RUN_DIR` scores a finished train.py run instead.

```bash
python experiments/wbcp/freeze_scores.py --dataset hopper --updates 100000 \
  --output runs/wbcp_frozen/hopper-medium-v2-s202609171-u100000        # about 4.5 min on CPU
python experiments/wbcp/d4rl_benchmark.py --frozen runs/wbcp_frozen/hopper-medium-v2-s202609171-u100000 \
  --tilt policy density state --gamma 0 0.5 1 2 --n 200 1103 --trials 2000 --score both \
  --workers 16 --output runs/wbcp_bench/main.json                     # about 3 min
python -m unittest experiments.wbcp.test_d4rl_benchmark                # NumPy + SciPy only
JAX_PLATFORMS=cpu python -m unittest experiments.wbcp.test_freeze_scores
```

Tilts: `policy` up-weights rows near the frozen actor's action; `density` up-weights
sparse rows (log k-NN distance in standardized (s, a)); `state` tilts on ||obs||.
`--blocks` draws whole episodes for calibration; `--per-episode K` draws K rows from each
of ceil(n/K) length-weighted episodes (`--spacing stratified` puts one in each K-th of the
episode; `--spacing reservation` runs BCA's own reservation sampler); `--discriminator raw`
misspecifies the estimated ratio; `--shuffle-tilt SEED` reassigns a tilt's weights to random
rows (a control with the same weights and no link to the score). Each arm reports its failure
frequency and also the 95th / 99th percentiles of realized risk and the mean excess over alpha
among failing trials. `weighted_mechanism.py` evaluates the WBCP posterior at the true
threshold per bank, and `certificate_slack.py` computes Theorem 4's slack for the tilts. An independent reimplementation matched the benchmark's lambda*, oracle
test mass and n_eff exactly, and 100M Monte Carlo draws confirmed lambda* and R.

Hopper-medium-v2, 100k updates, 500,285 held-out transitions, normalized score,
n = 1103 (BCA's bank size), 2000 trials. Failure frequency (realized miscoverage above
alpha = 0.1; the target is at most 5%); BQ-CP is today's uniform BCA:

| Tilt | gamma | tilt n_eff / pool | BQ-CP | WBCP | WBCP (oracle w) |
|---|---|---|---|---|---|
| none | 0 | 100% | 4.9% | 4.8% | 4.6% |
| policy | 1 / 2 | 67% / 42% | 5.5% / 7.7% | 4.3% / 4.3% | 4.5% / 4.7% |
| density | 0.5 / 1 | 78% / 35% | 42.6% / 89.5% | 4.2% / 7.3% | 4.3% / 7.5% |
| density | 2 | 3.2% | 100% | 2.6% (abstains 12%) | 2.4% |
| state | 0.5 / 1 | 47% / 2.8% | 24.9% / 98.4% | 8.0% / 1.0% (abstains 31%) | 8.1% / 0.9% |

- Policy shift barely moves the shortest valid threshold (1.064 to 1.076): the learned
  scale makes the normalized score close to pivotal where the actor acts. With the raw
  score |y - q| uniform BCA fails 12.2% at policy gamma 2.
- Under density and state shift uniform BCA fails badly; WBCP holds or abstains. Where
  the shift is strong and aligned with the score (calibration n_eff about 400) WBCP
  fails 7-8% with oracle weights too: a finite-sample limit, not weight estimation.
- A misspecified discriminator costs a lot: WBCP fails 17% (density 0.5) and 36%
  (density 1) with raw-feature weights, against 4.2% and 7.3% with well-specified ones.
- Episode dependence dominates: drawing the ~1100 calibration rows as whole episodes
  (about 3 hopper episodes, as BCA's block reservation does; 1362 rows on average) makes
  every Bayesian rule fail about 28% of trials (RCPS 22%) with no shift at all. Reweighting
  cannot repair that; a bank of 5 spaced rows from each of 221 episodes does. See
  [DEPENDENCE.md](DEPENDENCE.md) for the full step-1 study.

# Weighted Bayesian conformal prediction (WBCP)

[calibration/wbcp.py](../../calibration/wbcp.py) implements Algorithm 1 of Lou and Luo,
*Weighted Bayesian Conformal Prediction*, arXiv:2604.06464v3, for the miscoverage loss:
nonconformity scores, normalized weighted Bayesian bootstrap masses with one extra test
atom (the exact posterior of a Dirichlet-process prior pushed through the likelihood
ratio), the beta-credible crossing, clamped at the weighted empirical quantile. With
uniform weights it is BQ-CP (Snell and Griffiths, 2025). BCA consumes it through
[calibration/reference.py](../../calibration/reference.py) at every refresh.

`test_wbcp.py` and `reproduce_table1.py` import no hosts. The `test_*_host.py` suites drive
each host runtime with synthetic data and stub environments; nothing here trains a full run
or steps a simulator.

## Checks

```bash
python -m unittest experiments.wbcp.test_wbcp        # NumPy + SciPy only
JAX_PLATFORMS=cpu python -m unittest experiments.wbcp.test_reference experiments.test_wbcp_configs
JAX_PLATFORMS=cpu python -m unittest experiments.wbcp.test_td3_bc_host experiments.wbcp.test_rebrac_host experiments.wbcp.test_cql_host experiments.wbcp.test_iql_host
python experiments/wbcp/reproduce_table1.py --gamma 1 # about two minutes on a laptop CPU
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

### What this shows

Under **covariate shift**, test rows come from a different law than the calibration bank, but the score of a given state-action keeps its law. Let $w(x) = dP_{\text{test}}/dP_{\text{cal}}(x)$. Then the true test miscoverage of a threshold $\lambda$ is a *weighted* tail mass of the calibration law:

$$L_{\text{test}}(\lambda) = \Pr_{\text{test}}(S > \lambda) = \frac{\mathbb{E}_{\text{cal}}\big[w(X)\,\mathbf 1\{S>\lambda\}\big]}{\mathbb{E}_{\text{cal}}[w(X)]}.$$

The benchmark applies this shift to a real frozen score pool: hopper-medium-v2, 500,285 held-out rows. The scores come from a TD3+BC model trained for 100k updates with BCA's scale fit: $s = |y - q|/\sigma$, where $q = \min(Q_1, Q_2)$. Row $i$ gets test probability $\propto a_i = e^{\gamma z_i}$. Here $z$ is the standardized log distance to the 10th nearest neighbour in standardized $(s,a)$ space, measured against a seeded 10,000-row reference subset of the pool. This is the **density** tilt: rows in sparse regions gain mass. The exact ratio and test mass are

$$w^*_i = \frac{a_i}{\operatorname{mean}(a)}, \qquad \bar w = \mathbb{E}_{\text{test}}[w^*] = \frac{\operatorname{mean}(a^2)}{\operatorname{mean}(a)^2}.$$

**WBCP** (Lou & Luo, Algorithm 1) draws Bayesian-bootstrap variables $E_1,\dots,E_{n+1}\sim\text{Exp}(1)$. Score $s_i$ gets mass $w_iE_i$. A test atom of mass $\bar w E_{n+1}$ sits at the worst-case loss:

$$L^+(\lambda) = \frac{\sum_{i:\,s_i>\lambda} w_i E_i + \bar w E_{n+1}}{\sum_j w_j E_j + \bar w E_{n+1}}, \qquad \lambda_{\text{hpd}} = \min\{\lambda : \Pr(L^+(\lambda)\le\alpha) \ge \beta\}.$$

Here $\lambda$ ranges over the bank's scores. WBCP deploys $\max(\hat\lambda_w, \lambda_{\text{hpd}})$, where $\hat\lambda_w$ is the weighted empirical $(1-\alpha)$ quantile. With $w_i = \bar w = 1$ this is exactly **BQ-CP**, the uniform calibration BCA runs today (`calibration/reference.py:176-179` calls `calibrate()` without weights and deploys the clamped threshold). The benchmark's BQ-CP arm records $\lambda_{\text{hpd}}$. For the bank in panel b the clamp does not bind, so the two agree. Uniform BQ-CP certifies the *calibration* law, so it can under-cover the test law.

Weighting costs information. Heuristically, the posterior behaves like a sample of Kish size

$$n_{\text{eff}} = \frac{(\sum_i w_i)^2}{\sum_i w_i^2} \le n,$$

so it is wider and its threshold is higher. With very uneven weights, the test atom's share alone can exceed $\alpha$ in more than $1-\beta$ of the draws. WBCP then abstains ($\lambda=\infty$).

Settings: $\alpha = 0.1$ and $\beta = 0.95$. A bank **fails** when its certified threshold misses more than 10% of test rows. A valid rule fails in at most 5% of banks.

### How to read it

- **a: The tail grows.**
  - x is a threshold λ on the normalized score |y − q|/σ.
  - y is the share of rows whose score exceeds λ, which is λ's true miscoverage.
  - Orange dashed: the calibration law, where every pool row is equally likely.
  - Blue: the tilted test law at γ = 1. It uses the same rows, reweighted by $w^*$. The largest weight is 33× the mean.
  - The orange dot is the threshold that is exact for the calibration law (λ = 1.06). The blue dot above it shows that this threshold misses 13.0% of test rows.
  - The black dot is the test law's own 10% point, λ* = 1.20.
- **b: One bank, two threshold posteriors.**
  - This is one real 1,103-row bank, drawn with the benchmark's own random streams. It is the first bank in benchmark order with the majority outcome; bank 0, where both rules fail, was skipped.
  - Each curve is a threshold posterior drawn as a CDF: at each λ, the posterior probability $\Pr(L^+(\lambda)\le\alpha)$ over 1,000 bootstrap draws. Both curves are step functions because each draw crosses at one of the bank's scores.
  - Each method deploys where its curve reaches β = 0.95 (the dots). The vertical line is λ*; any threshold to its left fails.
  - Uniform BQ-CP is steep: all 1,103 rows count fully, and the posterior SD is 0.038. It sits on the calibration law, certifies 1.07 and misses 12.8%.
  - WBCP is flatter (n_eff = 344, SD 0.119) and shifted right. It certifies 1.33 and misses 7.8%.
- **c: The benchmark.**
  - For each shift strength γ: the share of 2,000 iid banks (n = 1,103) whose certified threshold misses more than 10% of tilted-test rows.
  - Error bars are exact 95% intervals. The y axis is log scale.
  - Orange dashed: uniform BQ-CP.
  - Blue solid, filled circles: WBCP with exact weights.
  - Blue dotted, open diamonds: WBCP with weights from a logistic discriminator. The two WBCP series are offset slightly sideways so that neither hides the other.
  - The dotted horizontal line is the 5% budget (1 − β).
  - The second row of x labels is the mean Kish n_eff of a bank's exact weights.

### Takeaway

- **No shift.** All three rules fail about 5% of banks (4.85-5.0%), as they should.
- **Uniform BQ-CP is shift-blind.** It fails 42.6% of banks at γ = 0.5, 89.5% at γ = 1 and 100% at γ = 2.
- **WBCP nearly restores the guarantee.**
  - At γ = 0.5 it fails 4.3% of banks with exact weights and 4.2% with estimated weights.
  - At γ = 1 it fails 7.55% and 7.35%. This excess is finite-sample (DEPENDENCE.md Change 11). In a 4,000-bank sweep it went 7.6% → 6.3% → 6.7% → 6.0% [5.3, 6.8] as the bank grew from 1,103 to 8,824 rows: the excess shrinks but is still above 5% at 8×. Failing banks exceed α by 0.76 points on average.
  - The paper's frequentist slack η_n (Theorem 4) is 1.19 here, far above α, so the paper does not promise a 5% failure rate at this size. Under weighting, credibility is not frequentist confidence. DEPENDENCE.md reports that the paper's own §4.2 experiment gives 7.5-7.9%.
  - At γ = 2 it fails 2.35% / 2.6%, partly because it abstains in 6.3% / 12.1% of banks, and abstentions count as passes.
- **The price is effective sample size.**
  - n_eff falls from 1,103 to 857, 391 and 47 as γ grows.
  - At γ = 1 the mean certified threshold rises from 1.14 (uniform) to 1.40 (WBCP, exact weights), above λ* = 1.20. WBCP is conservative on average, yet 7.55% of banks still fail.
- **Estimated weights** track the exact ones within 0.3 points. Here the discriminator is fitted on the tilt feature itself, so the weight model is correct by construction.

### Sources

- `experiments/wbcp/viz/fig_shift_weighting.py`: this figure. Panels a-b are recomputed from the pool, and the recomputed tilt is checked against main.json.
- `calibration/wbcp.py`: `calibrate()` implements Algorithm 1: crossings at lines 43-54, λ_hpd at 115, n_eff at 116, the deployed max(λ̂, λ_hpd) at 117.
- `calibration/reference.py:176-179`: BCA's uniform calibration.
- `experiments/wbcp/d4rl_benchmark.py`: `load_pool`, `tilt_feature` (193), `standardize`, `make_tilt`, `ExactRisk`, `common_scale` (320), `draw_calibration`, `stream`, `run_trial` (439-475).
- `runs/wbcp_frozen/hopper-medium-v2-s202609171-u100000/frozen.npz` and `frozen.json`: the score pool and its score definition.
- `runs/wbcp_bench/main.json`: panel c and the tilt checks. In results.ipynb this is section 5, "Distribution shift" (the "Independent rows" column); in DEPENDENCE.md it is the iid column of Change 5.
- `runs/wbcp_dependence/hopper-u100000-weighting/a_density1.json` and `certificate_slack.json`: DEPENDENCE.md Change 11 (results.ipynb section 8).
- `experiments/wbcp/DEPENDENCE.md`: Changes 5, 8, 10 and 11.

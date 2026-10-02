### What this shows

At each refresh, BCA calibrates one number: a threshold $\lambda$ on the normalized critic residual score

$$s = \frac{|y - q|}{\sigma},\qquad y = \text{host Bellman target},\quad q = \min(Q_1, Q_2),\quad \sigma = \max(\eta, 10^{-6})\,u,$$

where $\eta$ is the learned scale network and $u$ the frozen residual unit (`calibration/reference.py:148-175`). The deployed threshold times $\sigma$ is the width that the dose reads (`calibration/dose.py:65-76`).

WBCP (Lou & Luo, arXiv:2604.06464v3, Algorithm 1) turns a bank of $n$ held-out scores into that threshold. BCA runs it with uniform weights ($w_i = 1$, $\bar w = 1$; `calibration/reference.py:176`). That is exactly BQ-CP (Snell & Griffiths, 2025).

The figure runs `calibration/wbcp.calibrate` on real banks. The scores come from the frozen TD3+BC hopper-medium-v2 pool: a critic trained for 100k updates, then frozen, scores every row of the held-out half of the dataset. Each bank is drawn with BCA's own reservation sampler, which takes $K = 6$ rows from each reserved episode. The settings are $\alpha = 0.1$, $\beta = 0.95$ and $M = 1000$ posterior draws, as in `configs/td3_bc.yaml`.

**Step 1: point estimate (Eq. 1).** $\hat\lambda$ is the smallest score whose weighted empirical miscoverage is at most $\alpha$:
$$\hat\lambda = \min\Big\{\lambda:\ \frac{\sum_i w_i\,\mathbf 1\{s_i > \lambda\}}{\sum_i w_i} \le \alpha\Big\}.$$

**Step 2: posterior over the calibration law (Eqs. 3, 6).** Each draw $m$ puts random masses on the $n$ scores. It also puts one mass on a *test atom*. The test atom stands for the unseen test point and is placed at the worst-case loss, so it is never covered:
$$E_1,\dots,E_{n+1}\overset{iid}{\sim}\mathrm{Exp}(1),\qquad V_i = \frac{w_i E_i}{\sum_j w_j E_j + \bar w E_{n+1}},\qquad V_{n+1} = \frac{\bar w E_{n+1}}{\sum_j w_j E_j + \bar w E_{n+1}}.$$
With uniform weights, $(V_1,\dots,V_{n+1})\sim\mathrm{Dirichlet}(1,\dots,1)$. The posterior miscoverage at $\lambda$ is $L^+(\lambda) = \sum_i V_i\,\mathbf 1\{s_i>\lambda\} + V_{n+1}$.

**Step 3: each draw's crossing.** $L^+(\lambda)\le\alpha$ exactly when the posterior mass of the scores at or below $\lambda$ reaches $1-\alpha$:
$$\lambda^{(m)} = \min\Big\{s_{(j)}:\ \sum_{i\le j} V^{(m)}_{(i)} \ge 1-\alpha\Big\}.$$
The crossing is $+\infty$ if the test atom alone exceeds $\alpha$. That did not happen in any draw shown here.

**Step 4: credible threshold (Eq. 7), then deploy.**
$$\lambda_{\rm hpd} = \min\{\lambda:\ \Pr(L^+(\lambda)\le\alpha\mid\text{bank})\ge\beta\} = \text{the } \lceil\beta M\rceil\text{-th smallest } \lambda^{(m)},\qquad \lambda_{\rm dep} = \max(\hat\lambda,\ \lambda_{\rm hpd}).$$

### How to read it

**Panel a: posterior CDFs for one bank of $n = 256$ scores from 43 episodes.**
- The orange step is the empirical CDF. It crosses the $1-\alpha$ line at $\hat\lambda$ (the diamond).
- The light-blue steps are the first 30 of `calibrate()`'s 1,000 Bayesian-bootstrap CDFs. The module regenerates them from the same random stream and checks that they reproduce `calibrate()`'s crossings exactly.
- Each blue dot on the dashed line is one draw's crossing $\lambda^{(m)}$.
- No blue curve ever reaches 1, because each draw keeps the mass $V_{n+1}$ on the test atom at $+\infty$.
- The x-axis is cut at 2.05. Six of the 256 scores lie beyond the cut (the largest is 5.31), so the gap to 1 at the right edge also holds those scores, not only the test atom.
- The y-axis starts at 0.6 to zoom in on the crossings.

**Panel b: all 1,000 crossings of that bank, which together are the threshold posterior.** Every crossing is one of the bank's own scores, so the posterior is drawn as stems: the number of draws that cross at each score.
- $\hat\lambda = 1.008$ (orange dashed line) sits in the bulk of the posterior.
- $\lambda_{\rm hpd} = 1.289$ (blue line) is the smallest score with at least 95% of draws at or below it. 31 draws (grey) cross above it. It exceeds $\hat\lambda$, so it is the deployed threshold.
- The dotted line is the pool's true 90% point, $\lambda^* = 1.064$, computed from all 500,285 pool scores.
- "misses" is each threshold's true miscoverage on the pool rows outside the bank. $\hat\lambda$ misses 11.3%, more than $\alpha$, so used alone it would make this a failed bank. The deployed threshold misses 6.2%.

**Panel c: one bank each of $n$ = 64, 256 and 1,024.** 1,024 is the configured TD3+BC hopper size.
- The thin line spans all draws.
- The bar spans the middle 90%, from the 50th to the 950th smallest crossing, so its top is $\lambda_{\rm hpd}$.
- The diamond is $\hat\lambda$. The blue dot is the deployed threshold, labelled with its true miscoverage.
- The posterior sd falls from 0.408 to 0.120 to 0.054.
- The deployed threshold moves down toward $\lambda^*$, and its miscoverage rises toward $\alpha$ without passing it: 1.1%, 6.2%, 8.3%.

### Takeaway

- $\hat\lambda$ alone is the bank's empirical 90% point. It scatters around $\lambda^*$, and it misses more than $\alpha$ whenever it lands below $\lambda^*$. In two of the three banks shown, it lands below: it misses 11.3% at $n=256$ and 10.5% at $n=1{,}024$.
- WBCP deploys the $\beta$-credible crossing rather than the centre of the posterior. This adds the margin that the bank's own uncertainty calls for. The margin is large for a small bank (the $n=64$ threshold misses only 1.1%) and shrinks as the bank grows.
- The guarantee is about frequency across banks: at most $1-\beta = 5\%$ of banks should fail. One bank per $n$, as shown here, illustrates the mechanism but does not measure that rate.
- On this pool, the benchmark measured 5.1% [4.4, 5.8] failed banks (204 of 4,000) for uniform BCA with BCA's sampler at $K=5$, $n=1{,}103$ (DEPENDENCE.md, Change 9). Two qualifications apply:
  - That run predates the remainder trim, so its banks held 1,105 rows. `dependence_evidence.json` notes that a rerun is needed to confirm it.
  - The benchmark scores $\lambda_{\rm hpd}$ (`d4rl_benchmark.py:453`), not the deployed $\max(\hat\lambda, \lambda_{\rm hpd})$. The deployed threshold is never lower, so its failure rate can only be equal or smaller.
- The configured design ($K=6$, $n=1{,}024$) has no TD3+BC entry in `calibration/dependence_evidence.json` yet.

### Sources

- `calibration/wbcp.py`:
  - `calibrate()`: Eq. 1 at lines 89-102, posterior draws at 104-111, Eq. 7 at 115, deployed max at 117.
  - `crossings()` at lines 43-54, including the test atom in the total at line 52.
- `calibration/reference.py:148-179`: the score and BCA's call with uniform weights.
- `calibration/dose.py:65-76`: the width is the deployed threshold times the scale.
- `calibration/bank.py`: `stratified_bank`.
- `experiments/wbcp/d4rl_benchmark.py`: `load_pool`, `episode_groups`, `draw_calibration` (spacing `reservation`), `stream` (seed 2026093001, trial 0), `ExactRisk.lambda_star`, and the BQ-CP arm at line 453.
- `configs/td3_bc.yaml`: `bca.alpha`, `credibility`, `draws`; `datasets.hopper.reservation` (size 1024, `rows_per_episode` 6).
- `runs/wbcp_frozen/hopper-medium-v2-s202609171-u100000/frozen.npz` and `frozen.json` (`score_definition`, `rows.heldout` = 500,285, `population_episodes` = 1,094).
- `experiments/wbcp/DEPENDENCE.md`: $\lambda^* = 1.064$ and the Change 9 table.
- `calibration/dependence_evidence.json`: the td3_bc hopper entry at K = 5, n = 1,103 (204/4000, `bank_trim` null).
- Figure module: `experiments/wbcp/viz/fig_wbcp_threshold.py`.

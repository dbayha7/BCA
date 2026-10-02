### What this shows

BCA calibrates a bank of critic-residual scores $s_1,\dots,s_n$. For the TD3+BC host each score is $s = |y - \min(Q_1,Q_2)(s,a)|/\sigma(s,a)$, where $y$ is TD3+BC's own target and $\sigma$ is BCA's learned scale. The threshold comes from WBCP (`calibration/wbcp.py`). With uniform weights, WBCP is BQ-CP.

The posterior is a Bayesian bootstrap. Each draw $m$ takes $E_1,\dots,E_{n+1} \sim \mathrm{Exp}(1)$ independently. Row $i$ gets the mass $E_i / \big(\sum_{j\le n} E_j + E_{n+1}\big)$. The extra atom $E_{n+1}$ stands for the unseen test point, placed at the worst-case loss. Each draw gives a crossing $\lambda^{(m)}$, the smallest score at which the posterior mass of misses is at most $\alpha$. The deployed threshold is

$$\lambda = \max\big(\hat\lambda,\ \lambda_{\text{hpd}}\big), \qquad \Pr_{\text{post}}\big(L^+(\lambda_{\text{hpd}}) \le \alpha\big) \ge \beta, \qquad \alpha = 0.1,\ \beta = 0.95,$$

where $\hat\lambda$ is the empirical $(1-\alpha)$ quantile and $\lambda_{\text{hpd}}$ is the $\beta$-quantile of the crossings (`wbcp.py:43-117`). A bank **fails** when more than 10% of test rows miss its threshold. A valid rule fails in at most 5% of banks.

The bootstrap treats rows as **exchangeable, independent units**. Rows from one episode are not independent. Let $I = \mathbf{1}\{s > \lambda^*\}$ mark a miss, where $\lambda^*$ is the true 90% quantile. Two rows of the same episode have correlated misses, with intra-class correlation $\rho$ (one-way ANOVA, `experiments/wbcp/dependence.py:42-52`). With $m$ rows per episode, clustering inflates the variance of the bank's miss rate by the **design effect**

$$D \approx 1 + (m-1)\rho, \qquad \operatorname{Var}(\hat p) \approx D\,\frac{p(1-p)}{n}, \qquad n_{\text{eff}} = \frac{n}{D}.$$

The posterior assumes $D = 1$, so its safety margin is $\sqrt{D}$ times too narrow. In a normal approximation (`dependence.py:102-103`) the failure rate becomes

$$\Pr(\text{fail}) \approx 1 - \Phi\!\left(\frac{z_\beta}{\sqrt{D}}\right), \qquad z_{0.95} = 1.645,$$

rather than 5%.

A small $\rho$ is enough to matter because episodes are long. On hopper-medium, a whole-episode bank has a Monte Carlo $D = 6.6$. Its mean of 1,362 rows carries about as much information as 206 independent rows, and 27.8% of such banks fail.

Which $\rho$ goes into the formula depends on how episodes are picked (DEPENDENCE.md Change 2):
- The ANOVA $\rho = 0.016$ shown in panel b is the right input for banks that pick episodes in proportion to their length.
- For whole episodes picked uniformly, the right input is the pair-weighted 0.0115. That gives $D \approx 6.5$, matching the Monte Carlo value.

A bigger bank of whole episodes does not help either. It still fails 26–28% with mean banks of up to 11,739 rows (Change 3), because $D$ depends on the number of rows per episode, not on the number of episodes.

**BCA's fix is a thinned bank** (`calibration/bank.py:45-102`, `stratified_bank`):
- Withhold $m = \lceil n/K\rceil$ distinct episodes. Each episode's inclusion probability is exactly proportional to its length (systematic sampling).
- Cut each episode into $K$ equal segments and take one row from each. Any surplus over $n$ is removed at random.

This gives $D \approx 1 + (K-1)\rho$. Spacing the rows apart lowers $D$ further, because the correlation decays with lag. On hopper it is 0.19 at lag 1, 0.022 at lag 50 and about 0 at lag 200.

The cost is training data, since every row of a withheld episode leaves training. On hopper, K = 5 at n = 1,103 withholds about 10–11% of the data.

### How to read it

- **a**: one bank of each kind, drawn with fixed seed 911 on the real lengths of the 1,094 held-out hopper-medium episodes in the benchmark pool.
  - Orange bars: a whole-episode bank, using the old rule (now kept only for population splits) at `calibration/reference.py:83-86`. Every row of 3 episodes is a calibration row.
  - Blue dots: `stratified_bank` with K = 5 and n = 1,103. It picks 221 episodes, and 6 are drawn here.
    - The whole grey bar leaves training. Only the dots are calibrated.
    - The top episode has 4 dots because 2 surplus rows were dropped across the bank.
  - The large numbers are benchmark failure rates over 4,000 banks each, against 5% allowed.
    - 27.8%: whole episodes.
    - 5.1%: BCA's own sampler. That run predates the surplus trim, so its banks held 1,105 rows.
- **b**: the measured $\rho$ of the miss indicator on the five datasets whose TD3+BC critic was stable. It ranges from 0.016 on hopper to 0.120 on walker2d-medium-replay.
- **c**: every no-shift benchmark run on those five pools (uniform-weight arm, normalized score), placed at its Monte Carlo design effect $D$. Each point is 4,000 banks.
  - Orange: whole episodes.
  - Blue: K = 2–50 stratified (spaced) rows per episode.
  - Grey ring: independent rows, where $D = 1$.
  - The curve is the normal approximation above. The dashed line is the 5% budget.

### Takeaway

On all five datasets, whole-episode banks fail 27–48% of the time. They sit roughly where the design-effect theory puts them, within about 6 points; with only 1–3 episodes per bank the normal approximation is rough.

Thinning brings $D$ back near 1 and failures back near 5%. On hopper that is 5.1% with BCA's own sampler and 5.2% with stratified K = 5. Thinned designs land within about ±2 points of the curve.

How small K must be depends on $\rho$. K = 5 is enough on hopper. On walker2d ($\rho = 0.12$) it fails 7.5%, and keeping $D$ near hopper's level needs about 1–2 rows per episode (Changes 10 and 12).

This fix is separate from importance weighting. It applies to uniform and weighted BCA alike, and it has to be in place before any comparison under shift means anything.

### Sources

- `experiments/wbcp/DEPENDENCE.md`: Summary; Changes 1–4, 6, 9, 10 and 12; "What this means for BCA".
- Code:
  - `calibration/wbcp.py:43-117` (the posterior and threshold).
  - `calibration/bank.py:45-102` (`stratified_bank`) and `:28` (the 5% budget).
  - `calibration/reference.py:83-86` (the whole-episode rule).
  - `experiments/wbcp/dependence.py:42-52` (ICC) and `:102-103` (the failure model).
- Run data:
  - `runs/wbcp_frozen/hopper-medium-v2-s202609171-u100000/frozen.npz` (episode lengths).
  - `runs/wbcp_dependence/hopper-u100000/` (`e0_iid`, `e1_blocks`, `predictions.json`).
  - `runs/wbcp_dependence/hopper-u100000-spacing/` (`e6_strat_k*`, `predictions.json`).
  - `runs/wbcp_dependence/hopper-u100000-reservation/` (`r1_resv_k5`, `predictions_k5`).
  - `runs/wbcp_dependence/all-datasets/` (`*_iid`, `*_blocks`, `*_strat*`, `predictions_*_small`).
  - `runs/wbcp_dependence/other-datasets/` (`w_*_n1024`, `predictions_walker2d_n1024`).
- Figure module: `experiments/wbcp/viz/fig_bank_dependence.py`.

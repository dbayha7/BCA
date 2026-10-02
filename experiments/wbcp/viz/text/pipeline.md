### What this shows

BCA leaves the host algorithm (TD3+BC, ReBRAC, CQL or IQL) as it is and changes **one detached term of its loss**. The size of that change comes from a calibrated band on the critic's one-step Bellman residual. The diagram has three bands:
- **Training** is the host's own machinery.
- **Calibration** is what BCA adds.
- **Testing** lists the four studies that check BCA, each with its status.

**1. Episode reservation** (`calibration/reference.py` `reserve_calibration`, `calibration/bank.py` `stratified_bank`). BCA withholds $m=\lceil n/K\rceil$ episodes, each drawn with probability proportional to its length. The bank takes one row from each of $K$ equal segments of every withheld episode. Any surplus over $n$ is then removed at random, which leaves $n$ rows: 248 to 8,192, depending on host and dataset. The host trains only on the episodes that were not withheld. Thinning matters: a bank of whole episodes fails in about 28% of banks on hopper-medium, with no shift at all (DEPENDENCE.md).

**2. Scale fit, every update** (`fit_scale`; ALGORITHMS.md Procedure B). On each training minibatch BCA updates the live scale $\sigma(s,a)=u\cdot\max(\eta_\psi(s,a),10^{-6})$. The network $\eta_\psi$ takes one Adam step on

$$\mathcal L_\text{scale}=\Big(\textstyle\sum_i w_i\,\mathrm{cov}_i-(1-\alpha)\Big)^2+\lambda_w\sum_i w_i\eta_i^2,$$

where $\mathrm{cov}_i$ is a soft indicator that the residual lies inside the band and $w$ are Bayesian-bootstrap masses. The unit $u$ is not trained by this loss. It follows an exponential moving average of the residual spread, $u\leftarrow0.99\,u+0.01\,\mathrm{std}_B(y-q)$; IQL averages its per-batch unit instead. The host's parameters are never touched.

**3. Refresh: freeze the scale, score the bank, run WBCP** (`freeze_reference`, `calibration/wbcp.py`). The first refresh comes after update 10,000, and its result is used from update 10,001. Later refreshes run every 5,000 updates up to 995,000, so a run has 198 refreshes. Each refresh freezes $(\psi_f,u_f)$ and scores the bank with the current critic:

$$\rho_j=\frac{|y_j-\min_k Q_k(s_j,a_j)|}{\sigma_f(s_j,a_j)}.$$

Here $y_j$ is the host's own Bellman target. IQL takes the min over its target heads; the other hosts take it over their online heads. WBCP with uniform weights (exactly BQ-CP) then draws $M=1{,}000$ Dirichlet masses over the $n$ scores plus one test atom at the worst case. For each draw $m$, $\lambda^{(m)}$ is the first sorted score whose cumulative mass reaches $1-\alpha$. The threshold is

$$\lambda_\text{HPD}=\lambda^{(\lceil\beta M\rceil)},\qquad R=\max(\hat\lambda,\lambda_\text{HPD}),\qquad \alpha=0.1,\ \beta=0.95,$$

where $\hat\lambda$ is the empirical $(1-\alpha)$ quantile. Equivalently, $R$ is the smallest score that is $\beta$-credible and is not below $\hat\lambda$. The diagram's box states it this way.

**4. Width and dose** (`calibration/dose.py`, `calibration/advantage.py`). Between refreshes, each training row $(s,a)$, at its logged action, gets

$$U(s,a)=R\,\sigma_f(s,a),\qquad m=1+0.5\,\frac{U}{U+u_f}\in[1,1.5).$$

What the dose touches depends on the host:
- **TD3+BC and ReBRAC:** $m$ multiplies the actor BC term (ReBRAC's critic BC term stays fixed).
- **CQL:** $m$ multiplies each row's conservative gap.
- **IQL:** for rows with $A>0$, the excess of the capped actor weight above 1 is shrunk: $w=1+\frac{A}{A+U}\,(\min(e^{\beta A},100)-1)$.

Everything is detached from the gradient. Before the first refresh there is no dose, and every host keeps its native weighting.

### How to read it

- **Pipeline flow.** Follow the arrows along the top row: dataset → reservation → host training. Then go down into calibration. Scale fit → score the bank → WBCP threshold run right to left. The frozen $R$ and $\sigma$ travel along the bottom lane into width → dose, which feeds back up into one host loss term.
- **Colours and borders.**
  - Ink borders and arrows are the host's own code.
  - Blue borders and arrows are BCA's additions.
  - The dashed blue bracket marks the steps that run only at a refresh. Everything else in the calibration band runs every update.
- **Tags.** Numbered tags (1–4) on the pipeline boxes match the four tests below. Each test checks one thing:
  1. whether the threshold is valid;
  2. whether the dose tracks the true Q error and changes the true value;
  3. where the dose should be applied;
  4. return after full training.
- **Status chips.** Solid blue means **done**, dashed light blue **running**, and dotted grey **next**.
  - **Test 1** is done on TD3+BC's pools (DEPENDENCE.md, Change 12) and in the ReBRAC, CQL and IQL host matrices (IQL finished on 2026-10-02: 55 of 55 runs, `results_iql.ipynb`). TD3+BC's host matrix started after IQL finished and is running.
  - **Test 3**, the step-4 placement study, is in its pilot and has no outcomes yet.
  - **Test 4**, full D4RL training, has not been run and needs approval.

### Takeaway

BCA's intervention is narrow and fully detached: one scalar per row, multiplying one loss term per host. What it calibrates is a normalised one-step Bellman residual on held-out rows of the behaviour data, not the policy's return.

The testing ladder therefore checks the pieces separately:
- **Is the threshold valid?** Test 1 checks it on frozen pools, with and without imposed shift.
- **Does the dose carry useful signal?** Test 2 uses a known MDP with exact $Q$.
  - The dose tracks the true Q1 error, but it stays below 1.5 and varies by only about 0.02 from row to row.
  - Against a constant dose with the same mean, BCA's change in true value differs by a median 6% of its effect (SIGNAL_STUDY.md).
- **Where should the signal be applied?** Test 3, the step-4 placement study, is still running.
- **Does any of it improve return?** Test 4, full D4RL training, has not been run.

### Sources

- Code:
  - `calibration/reference.py` (`reserve_calibration`, `freeze_reference`, `initial_reference`)
  - `calibration/bank.py`, `calibration/wbcp.py`, `calibration/dose.py`, `calibration/advantage.py`, `calibration/network.py`
  - `algorithms/{td3_bc,rebrac,cql}_bca.py` (`fit_scale`, `refresh`)
  - `calibration/iql_reference.py`, `calibration/iql_scale.py`
  - `runtime/{td3_bc,rebrac,cql,iql_pair}.py` (refresh call sites)
- Configs:
  - `configs/experiment.yaml`: 1M updates, warmup 10k, refresh every 5k to 995k.
  - `configs/{td3_bc,rebrac,cql,iql}.yaml`: α, β, M, blend 0.5, bank sizes.
- Docs: `ALGORITHMS.md` (Procedures A–D, Algorithms 1–4, Schedules) and `INTEGRATION.md`.
- Tests and their status:
  - Test 1: `experiments/wbcp/d4rl_benchmark.py`, `experiments/wbcp/host_matrix.py`, `experiments/wbcp/README.md`, `experiments/wbcp/DEPENDENCE.md`, and the file counts in `runs/wbcp_hosts/<host>/`.
  - Test 2: `experiments/signal/lq_harness.py`, `experiments/signal/SIGNAL_STUDY.md`.
  - Test 3: `experiments/signal/run_step4.py`, `experiments/signal/STEP4_DESIGN.md`.
  - Test 4: `train.py`.
- Figure module: `experiments/wbcp/viz/fig_pipeline.py`.

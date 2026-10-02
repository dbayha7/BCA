### What this shows

TD3+BC has two critic heads, but its actor follows only the first one. The actor loss is

$$\mathcal L_\pi = -\lambda\,\overline{Q_1\big(s,\pi(s)\big)} + \overline{d(s,a)\,\big(\pi(s)-a\big)^2},\qquad \lambda = \frac{\alpha_{\text{TD3+BC}}}{\operatorname{mean}|Q_1(s,\pi(s))|}$$

(`algorithms/td3_bc.py:128-139`; `[..., 0]` at line 130 selects $Q_1$). Here $d(s,a)$ is BCA's per-row dose on the behaviour-cloning term, and $d \equiv 1$ is plain TD3+BC. In the recorded baseline, BCA's code calibrates the residual of the *minimum* of the two online heads, both in the scale fit and in the refresh (`algorithms/td3_bc_bca.py:173`, `:255`). The current code still does this:

$$s_{\min} = \frac{|t - \min(Q_1,Q_2)|}{\sigma(s,a)} \quad\text{rather than}\quad s_1 = \frac{|t - Q_1|}{\sigma(s,a)},$$

where $t = r + \gamma(1-d_{\text{done}})\min(Q_1',Q_2')(s',\tilde a')$ is TD3's target from the Polyak target heads (`td3_bc_bca.py:146-156`). Because $\min(Q_1,Q_2)\le Q_1$,

$$t - Q_1 = \big(t - \min(Q_1,Q_2)\big) - \big(Q_1 - \min(Q_1,Q_2)\big).$$

On rows where $Q_1 > Q_2$, $Q_1$'s residual is shifted by the head gap, and $s_{\min}$ never sees that shift. WBCP gives a threshold $\hat\lambda$ such that, with credibility $\beta = 0.95$, at most $\alpha = 0.1$ of the score it was calibrated on lies above it. That guarantee holds only for the calibrated score. A bank **fails** for $Q_1$ when its min-calibrated threshold, applied under the min band's own $\sigma$, misses more than 10% of $Q_1$'s residuals in the population:

$$\Pr_{\text{pop}}\!\Big(\tfrac{|t-Q_1|}{\sigma_{\min}} > \hat\lambda_{\min}\Big) > \alpha .$$

A valid rule fails in at most $1-\beta = 5\%$ of banks.

**Panel a (Illustration)** is computed exactly from the closed forms of the linear-quadratic harness (`experiments/signal/lq_harness.py`), re-implemented in numpy. The system, $K^*$ and the direction $D$ are read from `runs/wbcp_signal/lq/results.json`, and $K^*$ is checked against the recorded value. The panel uses the harness's `q1_optimistic` case: $Q_1 = Q^\pi + \kappa\,\|a-\beta(s)\|^2$ and $Q_2 = Q^\pi$, so $\min(Q_1,Q_2) = Q^\pi$ exactly. The settings are poor behaviour data ($K_b = 0$, so $\beta(s)=0$), $\kappa = 1$ and π offset 1, all cells of the step-3 grid. The state $s=(0.5,0.5,0.5)$ was chosen for display. "True Q" means $Q^\pi$, the value of the policy being evaluated.

**Panel b (real data)** comes from signal-study step 2. It uses the five frozen TD3+BC critics that converged. Each pool has 4,000 banks of 1,024 independent held-out rows at logged actions, calibrated with uniform-weight WBCP at α = 0.1, β = 0.95 and BCA's deployed threshold $\max(\hat\lambda,\lambda_{\text{HPD}})$. σ is refit separately for each signal.

### How to read it

- **a.** The orange curve is the true $Q^\pi$. Here it also equals $Q_2$ and $\min(Q_1,Q_2)$. The dashed blue curve is $Q_1$.
  - Inside the grey band of logged actions ($\beta(s)\pm 2\sigma_b$, $\sigma_b = 0.2$), the two curves almost agree.
  - Away from the data, $Q_1$ rises above the truth (blue shading). The actor's Q term pulls toward $Q_1$'s peak, 1.25 from $\beta(s)$. The true $Q^\pi$ peaks at 0.63 on this slice.
  - At $Q_1$'s peak, $Q_1$ is wrong by 1.57. Because $\min(Q_1,Q_2)=Q_2$ everywhere, none of $Q_1$'s error, there or anywhere else, appears in $t-\min(Q_1,Q_2)$.
  - The panel does **not** claim that a band calibrated on $Q_1$ would cover the error at the peak. Banks see only logged actions (see Takeaway, Limits).
- **b.** Each pool has two bars, each with its 95% interval:
  - orange: the share of banks whose min-calibrated threshold misses more than 10% of $Q_1$'s residuals;
  - blue: the same share for a threshold calibrated on $Q_1$.
  - The dashed line is the 5% budget. Pools are sorted by the orange bar.
  - The note "Q1 > Q2 on 46–54% of rows" is a warning. The real heads are **not** one-sidedly optimistic as in panel a. They disagree in both directions, by about one residual (median $|Q_1-Q_2|$ ÷ median $|r_{\min}|$ = 0.52–1.15). That disagreement alone is enough to move $Q_1$'s residual outside the min band.

### Takeaway

- **The min-calibrated band fails for $Q_1$ on all five converged D4RL pools.** Between 10.5% (pen-expert) and 82.4% (walker2d) of banks fail, against a 5% budget.
- **Calibrating $Q_1$ restores validity at little cost.** 4.0–5.1% of banks fail, and the band is 2–10% wider. In the same files, the $Q_1$ band still covers the min residual: 0.0–2.9% of banks fail.
- **A small shift in mean miscoverage produces many failures.** Under the min band, $Q_1$'s mean miscoverage is 8.9–10.9%, against 8.5% for the min's own residual. My reading, not separately tested: WBCP's threshold sits about 1.5 points inside the 10% target, and this shift uses up most of that slack.
- **Decision (a judgment, not yet implemented).** SIGNAL_STUDY.md adopted $Q_1$ as the measurement signal **for step 4** under decision rule 1, whose precondition was only partly met. CRITIC_ALIGNMENT.md recommends calibrating $Q_1$ on TD3+BC. BCA's TD3+BC code still calibrates the min (`td3_bc_bca.py:173`, `:255`).
- **What it does not show.** Switching the signal changes what the band certifies, but barely changes the dose the actor receives: the signals' mean doses differ by at most 0.006. Whether calibrating $Q_1$ improves the actor is untested. The step-4 placement study is still running and has no outcomes yet.
- **Limits.**
  - The banks use logged actions and independent rows, not BCA's thinned per-episode banks.
  - The earlier alignment study (online σ, 1,000 banks) found 8.2–70.2% against 4.2–5.9%, with the same conclusion. Its run folder is not on disk, so those numbers are cited from the write-up only.
  - pen-cloned and pen-human are excluded because their critics did not converge.

### Sources

- `experiments/wbcp/CRITIC_ALIGNMENT.md`
- `experiments/signal/SIGNAL_STUDY.md` (steps 2–3, decision rules)
- `runs/wbcp_signal/frozen/<pool>/evaluation_b4000_s2026100201.json` (design `iid`; fields `signals.{min,q1}.failure.{q1,min}`, `half_width_mean`, `head_disagreement`; pools hopper-medium-v2, walker2d, halfcheetah, maze2d, pen-expert)
- `runs/wbcp_signal/lq/results.json` (meta: system, `k_opt`, `direction`, settings)
- `experiments/signal/lq_harness.py:175-195, 227-231, 267-292, 516-517, 582, 657`
- `algorithms/td3_bc.py:128-139`
- `algorithms/td3_bc_bca.py:146-156, 173, 255`
- Figure module: `experiments/wbcp/viz/fig_critic_alignment.py`

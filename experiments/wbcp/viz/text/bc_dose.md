### What this shows

WBCP certifies one threshold $R$ on the normalized score $|r|/\sigma(s,a)$. Here $\sigma(s,a) = u\,\eta(s,a)$ is the fitted scale and $u$ is the residual unit. BCA deploys $R = \max(\hat\lambda, \lambda_{\text{hpd}})$ with uniform weights, at $\alpha = 0.1$ and $\beta = 0.95$. It turns the certified band into a width per row, then into a multiplier on the host's behaviour-cloning (BC) term, called the **dose** (`calibration/dose.py`, `frozen_level_dose` and `level_critic_dose`):

$$U(s,a) = R\,u\,\max(\eta(s,a),10^{-6}), \qquad m(s,a) = 1 + b\,\frac{U(s,a)}{U(s,a)+u} = 1 + b\,\frac{R\,\eta(s,a)}{1+R\,\eta(s,a)}, \qquad b = 0.5 .$$

The unit $u$ cancels, so the dose depends only on the dimensionless width $U/u = R\,\eta$. The blend $b = 0.5$ is set in `configs/td3_bc.yaml`, and the CQL and ReBRAC configs use the same value. So $m$ runs from 1 at zero width toward a cap of 1.5, which no finite width reaches.

In TD3+BC the dose multiplies each transition's BC term, at the recorded action $a$ (`algorithms/td3_bc.py` l.127-138):

$$\mathcal{L}_\pi = -\lambda\,\mathbb{E}\big[Q_1(s,\pi(s))\big] + \mathbb{E}\Big[\,\mathrm{sg}(m(s,a))\,\tfrac{1}{d_A}\|\pi(s)-a\|^2\Big], \qquad \lambda = \frac{\alpha_{\text{TD3+BC}}}{\mathbb{E}|Q_1|},\ \ \alpha_{\text{TD3+BC}} = 2.5 .$$

$\alpha_{\text{TD3+BC}}$ is TD3+BC's own weight. It is not the calibration miscoverage $\alpha = 0.1$.

The same map multiplies ReBRAC's actor BC term (its critic BC term is unchanged) and CQL's conservative gap. IQL does not use this map: it shrinks the capped advantage weight instead (ALGORITHMS.md, Procedure D).

**Panel a: real data.** Each of the five healthy TD3+BC pools has a frozen critic (signal study, step 2). For every scored row of the pool, the dose was computed from the Q1 signal's fitted $\eta$ and unit. The threshold is the pool's mean deployed WBCP threshold over 4,000 independent-row banks of 1,024 rows. The computation is a float32 NumPy port of `dose.py`. The module checks it against the run's own dose statistics: it reproduces the exact statistics at $\lambda^*$ to $10^{-6}$, and the bank-averaged statistics to $2\times10^{-4}$.

**Panel b: post hoc, not pre-registered.** In the step-3 linear-quadratic (LQ) harness, the true value $J$ and every gradient are exact. Step 3 compared each dose against a *constant dose with the same mean*. The panel shows the BC term's actual gradient norm at the starting actor, relative to that constant dose, for the Q1 signal.
- Orange is the oracle dose, built from the true Q1 error. Blue is BCA's dose.
- Each bar spans every replicate of every $\kappa$/offset cell of the case: 15 values for independent errors and 45 for the others. The markers are medians.

Matching the mean weight did not match strength. The write-up's reading (not separately tested) is this: rows with a large true error or a wide band tend to have large per-row gradient leverage (large $\|s\|$ or $\|\pi(s)-a\|$). Weighting them up enlarges the BC gradient even at the same mean weight.

### How to read it

**Panel a, curve**
- The x-axis is the width in residual units. The y-axis is the dose.
- The dashed line is the cap at 1.5.
- The dotted line is dose 1, the host's own BC weight.

**Panel a, shading**
- The vertical band is the range of $U/u$ that contains each pool's middle 90% of rows: 0.49 to 2.66, the union of the per-pool 5th to 95th percentiles.
- The horizontal band carries that range across to the doses those rows receive, 1.16 to 1.36.

**Panel a, strip on the right**
- There is one violin per pool, showing the per-row doses between the 0.5th and 99.5th percentiles. The black tick is the pool mean.
- Pool means are 1.26 to 1.33 and row SDs are 0.02 to 0.05. So the doses fill only a small slice of $[1, 1.5)$.

**Panel b, rows**
- Each row is one LQ error case with good or poor behaviour data.
- The dashed line at 1× is the constant dose with the same mean.
- The orange bars sit far to the right only in *independent errors*. There the oracle's BC gradient is 2.3 to 3.1× the constant dose's with good data, and 1.1 to 2.8× with poor data.
- The right-hand column is the oracle's change in true $J$ against the constant dose, averaged over the case's cells (Q1 signal). The only large means are in independent errors, the one case with a large strength ratio: +0.14 with good data and −1.01 with poor. Elsewhere the means stay between −0.041 and +0.008.
- BCA's bars stay within 1.00 to 1.03× in every row.

### Takeaway

On real pools, BCA's dose is close to uniform extra BC: about 1.3× the host's BC weight, with a row-to-row SD of 0.02 to 0.05. The map is bounded below 1.5, and most rows sit at $R\,\eta \approx 0.5$ to $2.7$, past the curve's steep start. That leaves little room for rows to differ. The write-up suggests the cap is why targeting is weak, but that is untested.

Post hoc, most of the oracle dose's apparent "targeting" advantage in step 3 is **strength**. In independent errors, at the same mean dose, its BC gradient was 2 to 3× the constant dose's, and its large effects on $J$ appear only there.

BCA's own small departures from the constant dose also follow its slightly higher strength. A strength-only model correlates 0.85 with them over all 216 step-3 cells (three signals).

So a control matched on mean dose does not isolate targeting, and step 4 is designed to match **realized strength** instead. The step-4 placement study is still running and has no results yet. Nothing in this figure comes from it.

### Sources

- `calibration/dose.py` l.53-60 (`level_critic_dose`) and l.74-76 (`frozen_level_dose`): the dose map
- `algorithms/td3_bc.py` l.127-138 and `algorithms/td3_bc_bca.py` l.285-300: where the dose enters TD3+BC
- `algorithms/rebrac.py` l.229-235: actor BC only
- `configs/td3_bc.yaml` l.20, `configs/cql.yaml` l.38, `configs/rebrac.yaml` l.21: blend $b = 0.5$
- `ALGORITHMS.md`: "What BCA changes" (l.41-48) and Procedure D (l.320-357)
- `runs/wbcp_signal/frozen/<pool>/heads.npz` and `evaluation_b4000_s2026100201.json`, for hopper-medium-v2, walker2d, halfcheetah, maze2d and pen-expert
- `runs/wbcp_signal/strength_reanalysis.json` and `experiments/signal/strength_reanalysis.py`
- `runs/wbcp_signal/lq/results.json`
- `experiments/signal/SIGNAL_STUDY.md`: step 2 table (l.217-223), oracle against constant (l.312-318), and "Post hoc: same mean, different strength" (l.392-417)
- Figure code: `experiments/wbcp/viz/fig_bc_dose.py`

# Making offline caution reach harm outside the data: a literature and methods review for BCA

EXPLORATORY. This is a reading of published work plus a reading of BCA's own code. Nothing was trained or run. It
changes no pre-registered rule, and it makes no claim about how BCA itself behaves: that needs a BCA readout, which
needs the user's typed approval.

Source note: the full-text fetcher could not reach any paper host during this review (name lookup failed). Venues,
authors and core claims were checked with web search against proceedings pages and abstracts. A few formula-level
details come from the reviewer's memory of the papers. Those are marked "(detail from memory)".

Terms used throughout:
- **Offline reinforcement learning (offline RL):** learning a controller from a fixed log, with no new trials.
- **Critic:** the learned estimate of long-run value Q(s, a) for a state s and command a.
- **Actor:** the learned controller.
- **Behaviour cloning (BC):** imitating the logged commands.
- **Temporal-difference (TD) residual:** the critic's one-step error on a logged row.
- **Clone:** the cloned actor.
- **Band:** BCA's conformal band.
- **Width:** the band's half-width at a (state, command) pair.
- **Unsupported command:** a command the log almost never shows at that kind of state. Here that means past 80%
  throttle.

---

## 1. What has to change, in one page

The gate found two different kinds of "outside the data":

1. **Unsupported commands at recorded states.** The gapped critic overrates pushing past 80% by about 0.34. That is
   almost exactly the true price, so the critic sees no cost in it (check 1, mark f).
2. **Unrecorded states.** Every strong-current start lies outside the recorded conditions. At those states the actor
   only extrapolates.

The key observation decides which kind is the lever. The full-log critic saw no strong-current states either, yet its
actor scores 84.6 there. So the information that prevents the harm can be given at recorded states, and it concerns
unsupported commands. Reading BCA's code shows three separate reasons why BCA, as built, cannot deliver that
information:

- **(a) Where it acts.** The dose multiplies the BC term on the logged row (s, a_logged) (`algorithms/td3_bc_bca.py`,
  `bc_readout`: the scale network is evaluated at `batch.obs, batch.action`). The harmful pairs, full-throttle
  commands in the full-throttle region, were about 99% deleted. So there is almost no row there to dose. BC caution
  can only pull toward the commands that remain, at the states that remain.
- **(b) What it measures.** The width is the threshold times the scale network eta(s, a). eta is fitted to TD residuals
  on logged rows, and the threshold is calibrated on held-out logged rows with equal weights
  (`calibration/reference.py`: "unit weights, the exchangeable case"). The band therefore covers the residual on
  average over logged rows only. At the actor's own command past 80%, two things are missing:
  - there is no TD residual to calibrate against (no transition was recorded);
  - eta there is whatever the small network extrapolates to. Networks built from rectified-linear units become linear
    along any direction leaving the data (Xu et al., ICLR 2021), so eta can grow or shrink there.
- **(c) How strong, and what "uncertified" does.** The dose is 1 + 0.5 · width / (width + unit), which is below 1.5 by
  construction. Worse, when the threshold is infinite (nothing can be certified), `frozen_level_dose` deliberately
  maps that row to "the native multiplier 1" (`calibration/dose.py`). The one honest verdict a conformal band can give
  about a point outside its calibration support is "I cannot certify this". BCA currently turns that verdict into no
  caution at all.

Check 3 showed that widening the range (factors up to 2, even ×3) does not fix (a) and (b). The rest of this review
asks which mechanisms in the literature act on unsupported commands or on unrecorded states, and what a conformal band
would need in order to drive them.

---

## 2. Conformal prediction when the test point is not like the calibration data

### 2.1 Weighted conformal prediction under covariate shift

**Paper:** Tibshirani, Barber, Candès and Ramdas, *Conformal prediction under covariate shift*, NeurIPS 2019.

**Method.** The test inputs may follow a different law from the calibration inputs, while the label given the input
stays the same. Each calibration score is reweighted by the likelihood ratio w(x) = dP_test/dP_cal. The test point's
own weight sits as a point mass at +∞. With p_i = w(X_i) / (Σ_j w(X_j) + w(x)) and p_test = w(x) / (Σ_j w(X_j) + w(x)),
the threshold is the (1 − α) quantile of Σ_i p_i δ_{V_i} + p_test δ_{+∞} (detail from memory).

**What it means outside the calibration support.** The guarantee requires the test law to be absolutely continuous
with respect to the calibration law. As a test point moves into a region where calibration data are rare, w(x) grows,
p_test passes α, and the quantile becomes +∞, so the interval is the whole real line. This follows directly from the
construction. At a point truly outside the support, the ratio is unbounded and the only valid answer is "infinite".
The method does not extrapolate. It refuses.

**Follow-up results.**
- Barber, Candès, Ramdas and Tibshirani, *Conformal prediction beyond exchangeability*, Annals of Statistics 2023:
  bound the coverage loss from weights that are not exactly right.
- Ying, Guo, Khamaru and Hung, *Informativeness of weighted conformal prediction*, arXiv 2024: show directly that
  weighted conformal returns infinite intervals more and more often as the overlap shrinks. Coverage is kept only
  because of those infinite intervals; coverage on the finite intervals falls below target.
- Lei and Candès, *Conformal inference of counterfactuals and individual treatment effects*, Journal of the Royal
  Statistical Society Series B 2021: use estimated weights and need an overlap assumption.

When weights are estimated by a classifier, a classifier that extrapolates smoothly can report a moderate ratio far
from the data. The band is then finite and wrong. That is the silent failure mode.

**Related designs that localise the guarantee.**
- Guan, *Localized conformal prediction*, Biometrika 2023: weights calibration points by their closeness to the query.
  With no neighbours, the band is infinite.
- Gibbs, Cherian and Candès, *Conformal prediction with conditional guarantees*, Journal of the Royal Statistical
  Society Series B 2025: coverage holds over a chosen class of shifts. One could ask for coverage within the group
  "throttle past 80%". With almost no rows there, the honest threshold is again infinite.

**BCA connection.** BCA's threshold rule (weighted Bayesian conformal prediction, `calibration/wbcp.py`, after Lou and
Luo, arXiv 2026) has exactly this structure: an extra test atom with mass w̄·E carries the unseen point at the worst
loss, and a draw "whose test atom alone exceeds alpha never crosses (+inf)". BCA already has the machinery to say
"uncertified". It is used today with equal weights, so the actor's shift never enters. And its "uncertified" output is
mapped to no caution.

**What BCA would need.**
- Weights that describe the distribution the actor creates. The natural choice is the command ratio π(a|s) / π̂_b(a|s)
  at recorded states, with π a smoothed actor and π̂_b a behaviour density, such as the clone with a learned spread or
  a variational autoencoder as in SPOT.
- A rule that treats an infinite or near-infinite width as the strongest caution, not the weakest.

### 2.2 Feedback covariate shift: the model chooses its own test points

These are the closest statistical analogues of an actor that picks commands by climbing a critic fitted on the same
log.

- Fannjiang, Bates, Angelopoulos, Listgarten and Jordan, *Conformal prediction under feedback covariate shift for
  biomolecular design*, PNAS 2022. The test input (a designed protein) is chosen by a model trained on the calibration
  data. That is the "feedback" case, which plain weighted conformal does not cover. They give valid sets for it.
- Prinster, Stanton, Liu and Saria, *Conformal validity guarantees exist for any data distribution (and how to find
  them)*, ICML 2024. This extends validity to multi-step shifts that an agent's own actions induce (black-box
  optimisation, active learning).
- Stanton, Maddox and Wilson, *Bayesian optimization with conformal prediction sets*, AISTATS 2023. Conformal sets
  steer queries toward regions where the model's predictions have guaranteed validity. In their tests, coverage of the
  queried points improved without loss of sample efficiency. This is the optimisation-side twin of "keep the actor
  where the critic is certified".
- Prinster, Fannjiang and others, *Conformal policy control*, arXiv 2026 (ICML 2026 poster). A safe reference policy
  acts as a regulator for an optimised but untested policy. Their method:
  - interpolates between the two with a likelihood-ratio threshold;
  - calibrates that threshold conformally on data from the safe policy, so a declared risk tolerance holds in finite
    samples;
  - needs no knowledge of how the optimised policy was trained.

  This is the most direct published template for BCA as a certificate of how far the actor may deviate from the clone.

**What BCA would need.** Calibrate the band, or a deviation threshold, on held-out logged rows weighted by the
actor-to-behaviour ratio. Then let the actor deviate from the clone only as far as the certificate allows (channel 3
below).

### 2.3 Conformal off-policy evaluation

- Taufiq, Ton, Cornish, Teh and Doucet, *Conformal off-policy prediction in contextual bandits*, NeurIPS 2022.
  Weighted conformal with policy ratios gives intervals for outcomes under a new policy from logged data, with
  finite-sample guarantees.
- Zhang, Shi and Luo, *Conformal off-policy prediction*, AISTATS 2023. Return intervals in sequential decisions, using
  a "pseudo policy" to build subsamples that resemble the target policy, plus weighted conformal.
- Foffano, Russo and Proutiere, *Conformal off-policy evaluation in Markov decision processes*, arXiv 2023 (venue not
  confirmed). Finite-horizon version, with several ways to handle the behaviour-to-target shift.

All of them need overlap: the target policy may only choose commands the logging policy could have chosen. A
deterministic actor in continuous commands has to be smoothed before any ratio exists.

**What this means for BCA.** These methods can certify "how wrong is the critic for the actor's commands" only where
the actor stays inside the logged support. Past 80%, their honest answer is again "infinite".

### 2.4 Conformal prediction for decisions and action sets

- Ren et al., *Robots that ask for help: uncertainty alignment for large language model planners*, CoRL 2023. Conformal
  sets of candidate actions. When the set holds more than one option, the robot asks for help. Abstention is driven by
  set size, with a guarantee on task success.
- Lekeufack, Angelopoulos, Bajcsy, Jordan and Malik, *Conformal decision theory: safe autonomous decisions from
  imperfect predictions*, ICRA 2024. Calibrates decisions directly for low risk instead of sets for coverage. The
  guarantees hold even for observations that are neither independent nor identically distributed. Examples include
  choosing at run time between a nominal policy and a safe backup.
- Kiyani, Pappas, Roth and Hassani, *Decision theoretic foundations for conformal prediction*, ICML 2025. For a decision
  maker who cares about a high-probability floor on utility (value at risk), prediction sets followed by a max-min
  action rule are optimal. This is the decision-theoretic support for acting on a lower bound Q − width rather than on
  Q.
- Huang, Sharma, Loquercio, Angelopoulos, Goldberg and Malik, *Conformal policy learning for sensorimotor control under
  distribution shifts*, arXiv 2023. Conformal quantiles detect shift and switch between a safe and a fast base policy,
  or are fed to the policy as an input.
- Zhao, Simmons, Admoni, Ramdas and Bajcsy, *Conformalized interactive imitation learning*, ICLR 2025. Online
  conformal intervals that widen under deployment shift and trigger requests for help. Ensemble disagreement and dropout
  were overconfident under shift.
- Gibbs and Candès, *Adaptive conformal inference under distribution shift*, NeurIPS 2021. Online threshold updates
  that keep long-run coverage under arbitrary shift. It needs feedback at deployment.
- Upadhyay, *Conformal calibration for adaptive conservatism in offline reinforcement learning*, arXiv 2026 (ICML 2026
  workshop). This is the closest prior art to BCA in offline RL. It builds on implicit Q-learning (IQL, Kostrikov et
  al. 2022). Split conformal prediction on a value ensemble gives per-state uncertainty, which sets a per-state
  conservatism level: looser where data are dense, stricter at the data frontier. The gains it reports are modest and
  mixed. It still acts only through the in-sample value fit, at logged states.

### 2.5 Conformal detection of points outside the data

- Bates, Candès, Lei, Romano and Sesia, *Testing for outliers with conformal p-values*, Annals of Statistics 2023.
  Conformal p-values give false-alarm control for "is this new point unlike the reference set?", for any score.
- Cai and Koutsoukos, *Real-time out-of-distribution detection in learning-enabled cyber-physical systems*, ICCPS
  2020. Inductive conformal anomaly detection with calibrated false alarms, run online on a braking system and a
  self-driving controller.
- Kaur et al., *iDECODe*, AAAI 2022 (detail from memory). Conformal detection with an equivariance-based score.
- Prashant et al., AAAI 2025, per search summary. Conformal detection of unfamiliar transitions in deep RL.
- Contreras et al., *Safe, out-of-distribution-adaptive model predictive control with conformalized neural network
  ensembles*, L4DC 2025. A conformal monitor hands control to a reachability-based fallback. The paper also argues that
  raw ensemble disagreement is not a calibrated signal.

**What BCA would need.** A nonconformity score that is command-aware, not only state-aware. Strong current and big waves
lie entirely outside the recorded conditions for every controller (share 1.00 on the shift cards). So a detector on
the state alone would fire on every step, including the big-wave steps where the gapped actor is 26 points better than
the clone. A useful score asks whether this command is unusual for what the behaviour would do here, for example the
clone-density ratio or the distance from (s, a) to logged rows in command space.

---

## 3. Offline RL mechanisms: what each penalises, where, and whether it reaches outside the data

"Reaches unsupported commands" means the mechanism acts on commands the log does not show, at recorded states.
"Reaches unrecorded states" means it acts on states the log does not contain, during training or at deployment.

### 3.1 Policy constraints

| Method | What it constrains | At which (state, command) pairs | Reaches unsupported commands? | Reaches unrecorded states? | What BCA's band would need to play this role |
|---|---|---|---|---|---|
| Batch-constrained Q-learning (BCQ; Fujimoto, Meger and Precup, ICML 2019) | Actor picks among commands drawn from a generative model of the log, each nudged by at most Φ | Recorded states in training. At deployment, the generative model is queried at the test state | Yes: forbids them by construction | Only through the generative model's extrapolation (like the clone) | The band would set the per-state nudge limit Φ(s) instead of a fixed one |
| Bootstrapping error accumulation reduction (BEAR; Kumar, Fu, Soh, Tucker and Levine, NeurIPS 2019) | Maximum mean discrepancy between actor and logged commands kept under ε (support matching) | Recorded states | Yes | No | ε(s) set by the band; needs the band to be shift-aware |
| Behaviour-regularised actor-critic (BRAC; Wu, Tucker and Nachum, arXiv 2019) | Divergence from the behaviour policy, either in the actor loss or as a value penalty inside the critic's target | Recorded states, and next states in the value-penalty version | Yes, softly | Next states only | The value-penalty version is the template for channel 2 |
| Supported policy optimisation (SPOT; Wu et al., NeurIPS 2022) | Penalty −log π̂_b(π(s) given s), from a variational autoencoder density of the log, added to the actor loss of twin delayed deep deterministic policy gradient (TD3) | Recorded states, at the actor's own command | Yes: the penalty grows as the command leaves the support | No | Replace the density with a width that grows off-support (shift-weighted) |
| TD3 plus behaviour cloning (TD3+BC; Fujimoto and Gu, NeurIPS 2021) | Squared distance to the logged command, weighted against α·Q/mean abs Q | Logged rows only | Only indirectly: pulls toward the logged command at that row | No | This is BCA's current channel; check 3 shows it is too weak here |
| Revisited behaviour-regularised actor-critic (ReBRAC; Tarasov et al., NeurIPS 2023) | Behaviour-cloning penalty in both the actor loss and the critic's target (the next command's distance from the logged next command) | Logged rows and logged next states | Yes, softly, in the critic | No | A band-weighted critic penalty (channel 2) fits naturally in BCA's ReBRAC host |
| Distance-sensitive offline RL with better generalisation (DOGE; Li et al., ICLR 2023) | Learned distance from (s, a) to the data kept under a threshold. Rationale: deep critics interpolate well inside the data's convex hull and badly outside it | Recorded states | Yes, by distance rather than density, so it allows interpolation between logged commands | No | The band's width could replace or calibrate DOGE's distance threshold |

### 3.2 Value pessimism

| Method | What it penalises | At which pairs | Reaches unsupported commands? | Reaches unrecorded states? | What BCA's band would need |
|---|---|---|---|---|---|
| Conservative Q-learning (CQL; Kumar, Zhou, Tucker and Levine, NeurIPS 2020) | Pushes Q down at commands from the actor and at random commands; pushes up at logged commands; gives a lower bound on value | Recorded states, sampled commands | Yes | Only through bootstrapping from logged next states | BCA's CQL host multiplies the whole per-row gap. To reach unsupported commands the weight must be per sampled command, larger where the band is wide (channel 2) |
| Soft actor-critic with N critics, and its diversified version (SAC-N / EDAC; An, Moon, Kim and Song, NeurIPS 2021) | Minimum over N critics, close to mean − k·spread, so penalty grows with disagreement. Used in the target and in the actor loss | Actor's commands at recorded and next states | Yes: disagreement grows off-support | No | A width at the actor's command that grows off-support, like the ensemble spread does (channel 1) |
| Pessimistic bootstrapping (PBRL; Bai et al., ICLR 2022) | Ensemble-spread penalty on the target. Also samples the actor's commands at logged states and fits them to a pseudo-target Q − β·spread. The authors argue that penalising in-distribution targets alone does not control behaviour outside the data | Recorded states with the actor's commands, plus next states | Yes, explicitly | No | Exactly channel 2 with width instead of ensemble spread; the width must be meaningful off-support |
| Mildly conservative Q-learning (MCQ; Lyu et al., NeurIPS 2022) | Gives commands outside the data a pseudo-value close to the best supported command's value, not a fixed push-down | Recorded states, actor's commands | Yes, mildly | No | A "mild" variant of channel 2: replace pseudo-targets for wide-band commands by the best certified value |
| Anti-exploration by random network distillation, on soft actor-critic (SAC-RND; Nikulin et al., ICML 2023) | A novelty bonus at the actor's command, subtracted in actor and critic, with no ensemble | Actor's commands | Yes | No | A conformal-calibrated novelty score in place of random network distillation (channel 1) |
| Uncertainty-weighted actor-critic (UWAC; Wu et al., ICML 2021) | Down-weights pairs with high dropout uncertainty in actor and critic losses | Actor's commands at logged states | Yes | No | Per-command trust weight on the Q term from the band |
| Implicit Q-learning (IQL; Kostrikov, Nair and Levine, ICLR 2022) | Never queries commands outside the data: in-sample expectile fit; the actor is weighted imitation of logged commands | Logged rows only | Avoids them rather than penalising them | No: the actor extrapolates like a weighted clone | Prediction: BCA's IQL host may show little of this harm, because weighted imitation never proposes past-80% commands that the log lacks. Park et al. 2024 find this extraction also leaves improvement unused |
| Pessimistic value iteration (Jin, Yang and Wang, ICML 2021); pessimism and imitation (Rashidinejad et al., NeurIPS 2021) | A lower confidence bound bonus subtracted from the value | All pairs, in theory | Yes | Yes, in theory, if the bound is valid there | This theory is why channel 1 is principled: act on a valid lower bound. A conformal band is a lower bound only where it is calibrated |

### 3.3 Model-based uncertainty penalties

| Method | What it penalises | At which pairs | Reaches unrecorded states? | What BCA's band would need |
|---|---|---|---|---|
| Model-based offline policy optimisation (MOPO; Yu et al., NeurIPS 2020) | Reward minus λ · model-ensemble uncertainty, on short model rollouts | States the model generates near the data | Yes, near the data | A conformal band on model error could calibrate λ·uncertainty |
| Model-based offline RL (MOReL; Kidambi et al., NeurIPS 2020) | An unknown-pair detector (ensemble disagreement over a threshold) sends the rollout to an absorbing halt state with a large negative reward | Model-generated pairs | Yes, near the data | The detector's threshold calibrated conformally |
| Conservative offline model-based policy optimisation (COMBO; Yu et al., NeurIPS 2021) | CQL-style push-down on pairs from model rollouts | Model-generated pairs | Yes, near the data | As channel 2, on generated states |

A caution specific to ThrusterChain: the stall is absent from the gapped log, so a learned dynamics model would also
extrapolate the thrust curve linearly past 80% (Xu et al. 2021). Only the model's disagreement, not its prediction,
would flag the region.

### 3.4 State-side robustness

| Method | What it does | Reaches unrecorded states? | What BCA's band would need |
|---|---|---|---|
| Robust offline RL via conservative smoothing (RORL; Yang et al., NeurIPS 2022) | Perturbs logged states within a small ball. Smooths actor and critic over the ball and makes the critic extra pessimistic at the perturbed states | Yes, within the ball | Use width at perturbed states as the pessimism. The band is uncalibrated there, so this must count as "uncertified means cautious" |
| Simple self-supervision for offline RL (S4RL; Sinha, Mandlekar and Garg, CoRL 2021) | State augmentation (noise, mixup, adversarial) with consistent Q targets | Yes, within the augmentation radius, but with smoothing rather than pessimism | Same as RORL |
| State deviation correction (SDC; Zhang et al., AAAI 2022) | Perturbs states. Trains the actor so the model-predicted next state moves back toward the data | Yes: teaches recovery toward the data | A band-defined "certified region" as the target to return to |
| State correction and action suppression (SCAS; Mao et al., NeurIPS 2024) | Pulls the agent from unfamiliar states toward high-value familiar states, and suppresses commands outside the data | Yes | As SDC |

In ThrusterChain, strong current is far from the data: current 0.7–1.1 against 0.1–0.7. Generic small perturbations
are unlikely to reach it. These methods help mainly with big kicks, where 11–18% of steps are outside the data.

### 3.5 Falling back to the behaviour where data are scarce

- Laroche, Trichelair and Tachet des Combes, *Safe policy improvement with baseline bootstrapping* (SPIBB), ICML 2019.
  On state-command pairs seen fewer than a set number of times, the new policy must copy the logging policy's
  probabilities. It may only improve on well-supported pairs. This guarantees, with high probability, no loss against
  the logging policy (detail from memory).
- Nadjahi, Laroche and Tachet des Combes, *Safe policy improvement with soft baseline bootstrapping*, ECML-PKDD 2019.
  Replaces the hard rule with a budget: the deviation from the logging policy, weighted by an error bound per pair
  (larger where counts are low), must stay below ε. Scholl et al. (2022) argue that the original safety proof has a
  gap and give a corrected variant.

**Where they act.** At every state, including at deployment. The fallback (the logging policy) is defined everywhere.

**What BCA would need.** The per-pair error bound e(s, a) in Soft-SPIBB is exactly the role a calibrated width could
play. It must be finite and small where the actor's command is supported, and large or infinite where it is not.
Continuous commands need a behaviour density in place of counts.

**Deployment-time fallback.** Conformal policy learning, conformal decision theory, conformal policy control and
KnowNo (section 2.4) are run-time versions of the same idea, with conformal calibration of the switching rule.

---

## 4. Evidence on actor and critic extrapolation, and how it maps onto the gate

**The critic overrates commands it never saw.**
- Fujimoto, Meger and Precup (ICML 2019) name this extrapolation error.
- Kumar et al. (NeurIPS 2019) show it accumulates through bootstrapping.
- Xu et al. (*How neural networks extrapolate*, ICLR 2021) show that networks built from rectified-linear units become linear
  along directions leaving the data. A critic that saw thrust rising with throttle up to 80% will keep it rising.
- The gate's check 1 is consistent with this: the overrating is about 0.34, almost exactly the true price of the
  stall.

**Deep critics interpolate well inside the data and badly outside it** (Li et al., DOGE, ICLR 2023). The shift cards
fit: the gapped critic's actor sits at median distance 3.93 from the data, against 2.49 for held-out logged rows.

**The actor's behaviour at new states is the bottleneck.**
- Park, Frans, Levine and Kumar, *Is value learning really the main bottleneck in offline RL?*, NeurIPS 2024: offline
  RL performance is often limited by how the policy generalises to states met at test time, not by performance on
  training states. They propose test-time policy extraction: improving the action at test states with the critic.
  Here that would amplify the gapped critic's error unless it is gated by a valid band.
- Mediratta, You, Jiang and Raileanu, *The generalization gap in offline reinforcement learning*, ICLR 2024: behaviour
  cloning often generalises to new environments better than offline RL.
- The gate's numbers match: in strong current the clone scores 54.4 and plain TD3+BC on the gapped log scores −16.8.

**Adaptivity is the alternative to pessimism, and it needs feedback.** Ghosh, Ajay, Agrawal and Levine, *Offline RL
policies should be trained to be adaptive*, ICML 2022: a log that leaves the dynamics partly unidentified makes
evaluation a partially observed problem. The best policy conditions on what it observes during deployment.

**Hard limit.** Foster, Krishnamurthy, Simchi-Levi and Xu, *Offline reinforcement learning: fundamental barriers for
value function approximation*, COLT 2022, give lower bounds showing that coverage and correct model form are not enough
on their own. Where the log has no coverage, no offline method can learn the true value. In ThrusterChain the stall
past 80% is almost unobserved in the gapped log, so no method, BCA included, can learn its price. What is possible:
- recognise that the critic cannot be trusted there, and act cautiously (pessimism, fallback);
- adapt once deployed.

Any claim for BCA has to be "identifies where the critic cannot be trusted", not "detects the stall".

**Why the full-log run implies a training-state fix can work.** The full-log critic carried a correct belief about
past-80% commands at recorded states. The actor trained on it turned that belief into a mapping that also holds in
strong current (84.6). A pessimistic belief, "unsupported commands are costly", pushes the same way here: the
stall-aware controller never passes 80% and scores 100. The actor learns it at recorded states and carries it outward.

**Control for the cost of caution.** In the no-stall comparison world, pushing past 80% is genuinely good. The same
pessimism should cost return there. That world is the built-in control.

---

## 5. Ranked channels for BCA

Each channel below changes where BCA acts or what its "uncertified" verdict does. None is the pre-registered
mechanism, so each would need its own pre-registration. Each also has a room check that uses only stand-in signals:
- exact or region harm;
- k-nearest-neighbour distance to logged rows;
- critic-ensemble disagreement;
- shuffled or uniform placement.

These checks can be run now as exploratory work, in the existing harness: frozen critics, actor trained from the
clone, exact returns.

### Rank 1: put the band on the actor's own command, as a value penalty (a lower bound the actor climbs)

**Mechanism.** Train the actor on Q(s, π(s)) − κ · W(s, π(s)) instead of Q(s, π(s)), keeping TD3+BC's own BC term. W is
a width that is large where the command is unsupported. The penalty is evaluated at the command the actor proposes, at
every batch state, and its gradient flows through the command. It therefore changes the direction the actor moves,
along the throttle axis. A per-row BC multiplier only changes how hard the actor is pulled toward a logged command that
may not even exist there.

**Precedent.**
- SAC-N/EDAC (minimum over an ensemble), SPOT (density penalty at the actor's command) and SAC-RND (novelty at the
  actor's command).
- Pessimistic value iteration, for why a lower bound is the right object.
- Kiyani et al. 2025: max-min over a calibrated set is optimal for a decision maker who wants a high-probability floor
  on utility.
- UNIQ: conformal calibration of the conservatism level.

**Why it can reach outside.** It installs at recorded states the belief that made the full-log actor generalise. It
also acts at states whose full-throttle rows were deleted, because it needs only the state and the actor's command,
not a logged row.

**What BCA must change.**
1. Evaluate eta at (s, π(s)), not at the logged command.
2. Make the width shift-aware. Either calibrate with actor-to-clone command-ratio weights (section 2.1), so that
   unsupported commands come out uncertified, or feed eta a command-space distance-to-data feature, so that it cannot
   shrink off-support.
3. Map "uncertified" to a large, capped penalty, and keep a smooth surrogate so the actor still gets a gradient.
4. Scale κ in units of the residual unit.

Inside the support the width is small and calibrated, so the native policy improvement is largely kept. That is the
"mild conservatism" argument of MCQ, DOGE and UNIQ, and it bears on the user's second aim.

**Exploratory room check (no BCA).** Use the frozen gapped critic, train actors from the clone at alpha 5, and compare
these arms:
- (i) region oracle: penalty κ·max(0, throttle − 0.8), with κ sized to the measured overrating of about 0.34;
- (ii) k-nearest-neighbour distance of (s, π(s)) to logged rows;
- (iii) disagreement of a small critic ensemble trained on the gapped log;
- (iv) strength-matched even penalty: squared distance to the clone's command at the same strength;
- (v) shuffled placement.

The channel is open only if (i) recovers a large share of the 99 strong-current points and keeps most of TD3+BC's +27
to +47 points elsewhere, while (iv) and (v) do not. Report the no-stall world as the cost of caution. If even (i)
fails, value-side caution at recorded states cannot reach the harm either.

### Rank 2: band-weighted conservative critic at sampled unsupported commands

**Mechanism.** In the critic update, sample commands at each logged state and at each logged next state, both from the
actor and uniformly. Then either:
- push their Q down in proportion to their width (CQL with per-command weights); or
- fit them to a pseudo-target Q − κ·W (PBRL), or to the best certified value (MCQ, the mild version).

The critic itself then carries "unsupported is costly". Bootstrapping spreads that belief along trajectories, so every
actor trained on that critic inherits it. This is the most literal match to the key observation: give the gapped
critic the full-log critic's belief about past-80% commands.

**Precedent.** PBRL (explicitly argues that penalties on in-distribution targets alone are not enough), CQL, COMBO,
ReBRAC's critic penalty and BRAC's value-penalty version.

**What BCA must change.**
- Today BCA's CQL host doses the whole per-row gap. The weight must instead be per sampled command, which needs the
  width at sampled commands.
- In the TD3+BC and ReBRAC hosts, the critic needs a new regularising term.
- The width must be meaningful off-support, the same requirement as rank 1.

**Why second.** It is a larger change: critics are retrained per arm. And it needs the same off-support width as rank
1. Rank 1 is cheaper and easier to attribute; rank 2 is the full-host version.

**Exploratory room check.** The rank 1 placement signals inside a critic regulariser, trained on the gapped log.
Then the H2 protocol: frozen critic, actor from the clone, exact returns.

### Rank 3: a certified deviation from the clone (baseline bootstrapping with a conformal budget), at training and at deployment

**Mechanism.** At any state, training or deployment, let the actor deviate from the clone only as far as the band stays
certified. Concretely: use a(s) = clone(s) + t(s)·(π(s) − clone(s)), with t(s) the largest step whose command keeps a
finite, small width. Calibrate with actor-to-clone ratio weights, as in conformal policy control. In training this is
Soft-SPIBB with the width as the per-pair error bound. At deployment it is a run-time fallback, as in conformal policy
learning and conformal decision theory.

**Why it reaches outside.** It acts directly at unrecorded states. The fallback, the clone, is defined everywhere and
holds up comparatively well in strong current (54.4, against −14.3 for the gapped critic's actor).

**Bounds already known from the gate.**
- Full fallback in strong current would recover 68.7 points.
- Full fallback in big waves would cost 26.4 points (70.0 to 43.6).

So the gate must be command-aware, not state-aware: weights must use the command ratio given the state, with the clone
as the behaviour model, and never the state ratio. The state ratio is infinite at every strong-current and big-wave
start, and would throw away the improvement in big waves.

**What BCA must change.**
- A behaviour density, for example the clone with a learned spread.
- A smoothed actor, so the ratio exists.
- The weighted threshold already in `wbcp.py`, used with non-uniform weights.
- "Uncertified" read as "fall back", not "no caution".

**Why third.** It needs density modelling. It is a policy constraint rather than an improvement of the critic, so it
caps how much optimisation can gain. Its guarantee is about deviation risk measured on logged data, which says nothing
about the stall itself.

**Exploratory room check.** Deployment-time projection toward the clone, gated by:
- (i) region oracle;
- (ii) clone-density ratio of the actor's command;
- (iii) k-nearest-neighbour command distance;
- (iv) a state-only distance gate, as the negative control expected to lose in big waves.

All are applied to the saved gapped and TD3+BC actors in `harm_return/.../actors.msgpack`. This needs no training at
all.

### Rank 4: extend the penalty to nearby unrecorded states by perturbation

**Mechanism.** Evaluate the rank 1 or rank 2 penalty also at logged states perturbed within a ball (RORL, S4RL), or add
recovery toward the data (SDC, SCAS). The band is uncalibrated at perturbed states, so their width counts as
uncertified, meaning cautious.

**Why fourth.** In ThrusterChain the harm sits far from the data, so a generic small ball will not reach strong current.
Perturbing along the known shift direction would use knowledge of the test. It is most useful for big kicks.

**Room check.** Isotropic state noise of growing radius on top of the best rank 1 arm, reporting gains per situation.

### Rank 5: deployment-time identification (a conformal alarm), as a report rather than a fix

**Mechanism.** Turn the command-aware nonconformity score from rank 3 into conformal p-values (Bates et al. 2023), with
a controlled false-alarm rate on held-out logged rows. Report the alarm rate at past-80% steps against other steps, per
situation. With deployment feedback, update the threshold online (adaptive conformal inference; Gibbs and Candès 2021).

**Why it matters.** It is the cleanest evidence for the user's first aim, "BCA helps identify the harm", and it is a
by-product of rank 3.

**Why last as a stand-alone channel.** On its own it detects but does not improve the actor.

**Room check.** With stand-in scores only: for each situation, compute the share of the gapped actor's past-80% steps
that the alarm flags, and the share of the full-log actor's and the stall-aware controller's steps that it flags (false
alarms in the useful sense).

### Summary of the ranking

| Rank | Channel | Acts at | Reaches unsupported commands | Reaches unrecorded states | Keeps policy improvement | Size of change to BCA |
|---|---|---|---|---|---|---|
| 1 | Width at the actor's command as a value penalty | Recorded states, actor's command | Yes, directly | Through the actor's generalisation (proven possible by the full-log run) | Yes, if the width is small inside the support | Small: evaluation point and loss term |
| 2 | Width-weighted conservative critic | Recorded and next states, sampled commands | Yes, in the critic | Through bootstrapping and the actor's generalisation | Yes, if mild | Medium: critic regulariser per host |
| 3 | Certified deviation from the clone | Every state, including deployment | Yes | Yes, directly | Capped by the constraint | Medium: behaviour density and weighted calibration |
| 4 | Perturbed-state penalty | Near-data states | Yes | Near the data only | Mostly | Medium |
| 5 | Conformal alarm | Deployment | Detects | Detects | Not by itself | Small |

**Common requirement for every channel.** BCA must read its own "uncertified" verdict as the strongest caution, and its
band must be calibrated for the distribution the actor creates, not only for logged rows. Without those two changes, no
channel above can be driven by BCA. With them, BCA's distinctive contribution over ensembles and densities is a
calibrated, finite-sample statement of where the critic can be trusted. Inside the support that permits strong
optimisation; outside it, the honest answer is "not certified".

---

## 6. References

- An, G., Moon, S., Kim, J.-H. and Song, H. O. Uncertainty-based offline reinforcement learning with diversified
  Q-ensemble. NeurIPS 2021.
- Bai, C. et al. Pessimistic bootstrapping for uncertainty-driven offline reinforcement learning. ICLR 2022.
- Barber, R. F., Candès, E. J., Ramdas, A. and Tibshirani, R. J. Conformal prediction beyond exchangeability. Annals of
  Statistics 51(2), 2023.
- Bates, S., Candès, E., Lei, L., Romano, Y. and Sesia, M. Testing for outliers with conformal p-values. Annals of
  Statistics 51(1), 2023.
- Cai, F. and Koutsoukos, X. Real-time out-of-distribution detection in learning-enabled cyber-physical systems. ICCPS
  2020.
- Contreras, P. et al. Safe, out-of-distribution-adaptive MPC with conformalized neural network ensembles. L4DC 2025.
- Fannjiang, C., Bates, S., Angelopoulos, A. N., Listgarten, J. and Jordan, M. I. Conformal prediction under feedback
  covariate shift for biomolecular design. PNAS 2022.
- Foffano, D., Russo, A. and Proutiere, A. Conformal off-policy evaluation in Markov decision processes. arXiv 2023.
- Foster, D. J., Krishnamurthy, A., Simchi-Levi, D. and Xu, Y. Offline reinforcement learning: fundamental barriers for
  value function approximation. COLT 2022.
- Fujimoto, S. and Gu, S. S. A minimalist approach to offline reinforcement learning (TD3+BC). NeurIPS 2021.
- Fujimoto, S., Meger, D. and Precup, D. Off-policy deep reinforcement learning without exploration (BCQ). ICML 2019.
- Ghosh, D., Ajay, A., Agrawal, P. and Levine, S. Offline RL policies should be trained to be adaptive. ICML 2022.
- Gibbs, I. and Candès, E. Adaptive conformal inference under distribution shift. NeurIPS 2021.
- Gibbs, I., Cherian, J. J. and Candès, E. J. Conformal prediction with conditional guarantees. Journal of the Royal
  Statistical Society Series B, 2025.
- Guan, L. Localized conformal prediction: a generalized inference framework for conformal prediction. Biometrika 2023.
- Huang, H., Sharma, S., Loquercio, A., Angelopoulos, A., Goldberg, K. and Malik, J. Conformal policy learning for
  sensorimotor control under distribution shifts. arXiv 2023.
- Jin, Y., Yang, Z. and Wang, Z. Is pessimism provably efficient for offline RL? ICML 2021.
- Kaur, R. et al. iDECODe: in-distribution equivariance for conformal out-of-distribution detection. AAAI 2022.
- Kidambi, R., Rajeswaran, A., Netrapalli, P. and Joachims, T. MOReL: model-based offline reinforcement learning.
  NeurIPS 2020.
- Kiyani, S., Pappas, G., Roth, A. and Hassani, H. Decision theoretic foundations for conformal prediction: optimal
  uncertainty quantification for risk-averse agents. ICML 2025.
- Kostrikov, I., Nair, A. and Levine, S. Offline reinforcement learning with implicit Q-learning. ICLR 2022.
- Kumar, A., Fu, J., Soh, M., Tucker, G. and Levine, S. Stabilizing off-policy Q-learning via bootstrapping error
  reduction (BEAR). NeurIPS 2019.
- Kumar, A., Zhou, A., Tucker, G. and Levine, S. Conservative Q-learning for offline reinforcement learning. NeurIPS
  2020.
- Laroche, R., Trichelair, P. and Tachet des Combes, R. Safe policy improvement with baseline bootstrapping. ICML 2019.
- Lei, L. and Candès, E. J. Conformal inference of counterfactuals and individual treatment effects. Journal of the
  Royal Statistical Society Series B, 2021.
- Lekeufack, J., Angelopoulos, A. N., Bajcsy, A., Jordan, M. I. and Malik, J. Conformal decision theory: safe autonomous
  decisions from imperfect predictions. ICRA 2024.
- Li, J., Zhan, X. et al. When data geometry meets deep function: generalizing offline reinforcement learning (DOGE).
  ICLR 2023.
- Lou and Luo. Weighted Bayesian conformal prediction. arXiv 2604.06464, 2026 (BCA's threshold rule).
- Lyu, J., Ma, X., Li, X. and Lu, Z. Mildly conservative Q-learning for offline reinforcement learning. NeurIPS 2022.
- Mao, Y., Wang, Q., Chen, C., Qu, Y. and Ji, X. Offline reinforcement learning with OOD state correction and OOD action
  suppression. NeurIPS 2024.
- Mediratta, I., You, Q., Jiang, M. and Raileanu, R. The generalization gap in offline reinforcement learning. ICLR
  2024.
- Nadjahi, K., Laroche, R. and Tachet des Combes, R. Safe policy improvement with soft baseline bootstrapping.
  ECML-PKDD 2019.
- Nikulin, A., Kurenkov, V., Tarasov, D. and Kolesnikov, S. Anti-exploration by random network distillation. ICML
  2023.
- Park, S., Frans, K., Levine, S. and Kumar, A. Is value learning really the main bottleneck in offline RL? NeurIPS
  2024.
- Prinster, D., Stanton, S., Liu, A. and Saria, S. Conformal validity guarantees exist for any data distribution (and
  how to find them). ICML 2024.
- Prinster, D., Fannjiang, C. et al. Conformal policy control. arXiv 2603.02196, 2026 (ICML 2026).
- Rashidinejad, P., Zhu, B., Ma, C., Jiao, J. and Russell, S. Bridging offline reinforcement learning and imitation
  learning: a tale of pessimism. NeurIPS 2021.
- Ren, A. Z. et al. Robots that ask for help: uncertainty alignment for large language model planners. CoRL 2023.
- Sinha, S., Mandlekar, A. and Garg, A. S4RL: surprisingly simple self-supervision for offline reinforcement learning in
  robotics. CoRL 2021.
- Stanton, S., Maddox, W. and Wilson, A. G. Bayesian optimization with conformal prediction sets. AISTATS 2023.
- Taufiq, M. F., Ton, J.-F., Cornish, R., Teh, Y. W. and Doucet, A. Conformal off-policy prediction in contextual
  bandits. NeurIPS 2022.
- Tarasov, D., Kurenkov, V., Nikulin, A. and Kolesnikov, S. Revisiting the minimalist approach to offline reinforcement
  learning (ReBRAC). NeurIPS 2023.
- Tibshirani, R. J., Barber, R. F., Candès, E. J. and Ramdas, A. Conformal prediction under covariate shift. NeurIPS
  2019.
- Upadhyay, A. UNIQ: conformal calibration for adaptive conservatism in offline reinforcement learning. arXiv
  2606.07592, 2026 (ICML 2026 workshop).
- Wu, J., Wu, H., Qiu, Z., Wang, J. and Long, M. Supported policy optimization for offline reinforcement learning
  (SPOT). NeurIPS 2022.
- Wu, Y., Tucker, G. and Nachum, O. Behavior regularized offline reinforcement learning (BRAC). arXiv 2019.
- Wu, Y. et al. Uncertainty weighted actor-critic for offline reinforcement learning. ICML 2021.
- Xu, K., Zhang, M., Li, J., Du, S. S., Kawarabayashi, K. and Jegelka, S. How neural networks extrapolate: from
  feedforward to graph neural networks. ICLR 2021.
- Yang, R., Bai, C. et al. RORL: robust offline reinforcement learning via conservative smoothing. NeurIPS 2022.
- Ying, M., Guo, W., Khamaru, K. and Hung, Y. Informativeness of weighted conformal prediction. arXiv 2024.
- Yu, T. et al. MOPO: model-based offline policy optimization. NeurIPS 2020.
- Yu, T., Kumar, A., Rafailov, R., Rajeswaran, A., Levine, S. and Finn, C. COMBO: conservative offline model-based policy
  optimization. NeurIPS 2021.
- Zhang, H. et al. State deviation correction for offline reinforcement learning. AAAI 2022.
- Zhang, Y., Shi, C. and Luo, S. Conformal off-policy prediction. AISTATS 2023.
- Zhao, M., Simmons, R., Admoni, H., Ramdas, A. and Bajcsy, A. Conformalized interactive imitation learning: handling
  expert shift and intermittent feedback. ICLR 2025.

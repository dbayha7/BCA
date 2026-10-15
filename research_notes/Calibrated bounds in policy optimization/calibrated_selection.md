# Calibrated selection and decision-making with statistical guarantees: choosing policies, hyperparameters, and the states where a policy may deviate from a baseline

Compiled 14 Oct 2026. "BCA" below means the user's method: Weighted Bayesian Conformal Prediction (Lou & Luo, arXiv:2604.06464v3). With uniform weights it is BQ-CP (Snell & Griffiths, 2025): Dirichlet(1,…,1) masses on n held-out losses plus one extra atom at the worst-case loss B. Sources were read from full text (arXiv HTML or author PDFs) unless a finding is marked otherwise.

## 1. Learn then Test (LTT) and conformal risk control (CRC): how they calibrate a hyperparameter of a decision procedure with risk guarantees, and any use in RL, control or robotics to tune conservatism or a fallback threshold

### Takeaway
LTT turns "choose λ from a finite grid" into multiple hypothesis testing. Every λ it returns satisfies P(risk ≤ α) ≥ 1−δ over the draw of the calibration set, for any dimension of λ and any (non-monotone) risk. CRC gives the cheaper in-expectation guarantee E[loss] ≤ α for monotone losses, and its rule is algebraically the posterior mean of the BQ-CP/BCA posterior. BCA's upper quantile is the training-conditional, LTT/RCPS-style analogue. Robotics uses exist for ask-for-help and fallback thresholds and for conservatism weights. One ICML 2025 paper applies LTT to the behavior-cloning weight of TD3+BC, but it tests candidates with online episodes. I found no purely offline use that also carries a guarantee on deployed return.

### Cited Findings
**Learn then Test (LTT)**
- Citation: Angelopoulos, Bates, Candès, Jordan & Lei, "Learn then test: Calibrating predictive algorithms to achieve risk control," arXiv:2110.01052 (2021); *Annals of Applied Statistics* 19(2):1641–1662, June 2025, DOI 10.1214/24-AOAS1998 — [Project Euclid](https://projecteuclid.org/journals/annals-of-applied-statistics/volume-19/issue-2/Learn-then-test--Calibrating-predictive-algorithms-to-achieve-risk/10.1214/24-AOAS1998.short)
- Target guarantee, an (α,δ)-risk-controlling prediction: P(R(T_λ̂) ≤ α) ≥ 1−δ. The probability is over the sampling of the n calibration points. When a risk cannot be controlled at level α, LTT abstains and returns nothing — [arXiv:2110.01052](https://arxiv.org/abs/2110.01052)
- "Our main insight is to reframe the risk-control problem as multiple hypothesis testing." Earlier distribution-free methods (conformal prediction, RCPS) "required that λ is one-dimensional and that the risk function be monotonic in λ"; LTT drops both assumptions — [arXiv:2110.01052](https://arxiv.org/abs/2110.01052)
- Mechanics, as summarized by a later paper: each candidate λ_i gets the null hypothesis "λ_i does not meet the reliability requirement." A p-value is computed from calibration data, e.g. with the Hoeffding–Bentkus inequality. A family-wise error rate (FWER)-controlling procedure then selects the certified subset, and an FDR-controlling variant is "readily obtained" — [Zecchin, Park & Simeone, arXiv:2409.15844](https://arxiv.org/abs/2409.15844)
- For an ordered threshold, LTT uses fixed-sequence testing. Example: certifying a binarization threshold so that multilabel predictions have FDR ≤ 10% — [arXiv:2110.01052](https://arxiv.org/abs/2110.01052)
- Worked examples include FDR in multilabel classification, IoU in instance segmentation (detectron2), and simultaneous control of outlier type-1 error and set coverage — [Project Euclid abstract](https://projecteuclid.org/journals/annals-of-applied-statistics/volume-19/issue-2/Learn-then-test--Calibrating-predictive-algorithms-to-achieve-risk/10.1214/24-AOAS1998.short)

**Conformal risk control (CRC)**
- Citation: Angelopoulos, Bates, Fisch, Lei & Schuster, "Conformal risk control," arXiv:2208.02814. Published at ICLR 2024 (the Twelfth ICLR), per the reference list of Snell & Griffiths — [arXiv:2208.02814](https://arxiv.org/abs/2208.02814); [Snell & Griffiths ref. list](https://arxiv.org/abs/2502.13228)
- Rule: λ̂ = inf{λ : (n/(n+1))·R̂_n(λ) + B/(n+1) ≤ α}, for losses that are non-increasing in λ and bounded by B. Guarantee: E[L_{n+1}(λ̂)] ≤ α. For i.i.d. continuous losses it is tight: E[L_{n+1}(λ̂)] ≥ α − 2B/(n+1). For non-monotone risks, the algorithm "does not control the risk" — [arXiv:2208.02814](https://arxiv.org/abs/2208.02814)
- The guarantee holds in expectation, and the authors contrast it with high-probability risk control (RCPS/LTT). Extensions cover distribution shift, quantile risk control, multiple and adversarial risks, and U-statistics — [arXiv:2208.02814](https://arxiv.org/abs/2208.02814)

**Link to BCA (Snell & Griffiths, "Conformal prediction as Bayesian quadrature," arXiv:2502.13228, 2025)**
- Split conformal prediction and CRC "can both be recovered as special cases" of a Bayesian-quadrature procedure. Quantile spacings of n continuous losses satisfy (u_1,…,u_{n+1}) ~ Dir(1,…,1). "The Conformal Risk Control decision rule … is simply the infimum over λ for which E(L⁺) ≤ α." Marginal guarantees "correspond to the posterior mean" — [arXiv:2502.13228](https://arxiv.org/abs/2502.13228)
- Experiments with 10,000 random calibration splits each:
  - Synthetic: the CRC rule exceeded the target risk in 21.20% of trials, versus 0.03% for their 95% highest-posterior-density (HPD) rule. Mean incurred risk was 0.3363 for CRC and 0.1758 for HPD.
  - Heteroskedastic regression (n = 200, α = 0.1, target failure rate 5%): both RCPS (Hoeffding) and their method stayed below 5% failure, with their method giving "significantly smaller prediction intervals."
  - MS-COCO false-negative-rate control: CRC exceeded the threshold even more often, while RCPS controlled it at the cost of larger sets.
  - Source: [arXiv:2502.13228](https://arxiv.org/abs/2502.13228)

**Uses in robotics, control and RL (conservatism, fallback, ask-for-help)**
- Conformal Decision Theory (CDT, Section 2 below): its batch version is CRC applied to decisions, and the paper names offline RL as a batch use case — [arXiv:2310.05921](https://arxiv.org/abs/2310.05921)
- RCIP (Lidard, Pham, Bachman, Boateng & Majumdar, "Risk-Calibrated Human-Robot Interaction via Set-Valued Intent Prediction," RSS 2024, arXiv:2403.15959) builds on LTT.
  - It jointly selects a prediction-set threshold λ and model parameters θ (e.g., temperature) from finite grids. Calibration uses 400 scenarios, and it controls several risks at once (e.g., task-failure probability and help-request rate). The robot asks for help when the calibrated action set is not a singleton.
  - Reported: "RCIP reduces the amount of human help by 5–30% versus baseline approaches."
  - Source: [arXiv:2403.15959](https://arxiv.org/abs/2403.15959); [RSS 2024 PDF](https://www.roboticsproceedings.org/rss20/p027.pdf)
- Luo, Zhao, Kuck, Ivanovic, Savarese, Schmerling & Pavone, "Sample-efficient safety assurances using conformal prediction" (WAFR 2022; journal version in IJRR, DOI 10.1177/02783649231221580).
  - Conformal calibration of a warning-system threshold gives false-negative rate ≤ ε + 1/(1+|A|), where |A| is the number of unsafe calibration examples, "using as few as 1/ε data points."
  - With too few unsafe examples (|A| ≤ 1/ε* − 1) the calibrated system must always warn. In practice, |A| ≈ 1.5/ε* − 1 sufficed for a low false-positive rate. Applications: driver warnings and robotic grasping.
  - Source: [arXiv:2109.14082](https://arxiv.org/abs/2109.14082); [WAFR PDF](https://wafr2022.github.io/proceedings/WAFR_2022_Final_24.pdf); [IJRR listing](https://dl.acm.org/doi/10.1177/02783649231221580)
- Adaptive LTT (aLTT; Zecchin, Park & Simeone, ICML 2025, arXiv:2409.15844) replaces LTT's p-values with e-processes, which allows sequential, data-adaptive testing with early stopping under FWER or FDR control.
  - Offline RL experiment: N = 20 TD3+BC policies with the behavior-cloning regularization weight λ on an even grid in [0.25, 5] (HalfCheetah), tested through online episodes, with target reliability α = 0.57 and δ = 0.1.
  - As acquisition becomes more evidence-driven, the true-positive rate (TPR, the fraction of truly reliable λ certified) rises from 0.32 to 0.85 under FWER control and from 0.40 to 0.85 under FDR control. "The TPR of the schemes under FDR control is larger than that obtained under FWER control."
  - Source: [arXiv:2409.15844](https://arxiv.org/abs/2409.15844); [ICML 2025 listing](https://dl.acm.org/doi/abs/10.5555/3780338.3783303)
- UNIQ (Upadhyay, ICML 2026 Workshop Demo, arXiv:2606.07592) uses split conformal calibration of value-ensemble disagreement to set a per-state IQL expectile τ(s): relax conservatism in well-covered states and tighten it at the data frontier.
  - D4RL MuJoCo average normalized score: 85.2 versus 81.6 for IQL. UNIQ uses 3 seeds; baseline numbers are taken from published reports.
  - A separate pessimism multiplier κ is task-specific: κ = 1.0 gives 77.4 on walker2d-medium but 13.7 on hopper-medium-replay. It is chosen "using held-out validation returns," with no stated guarantee.
  - Source: [arXiv:2606.07592](https://arxiv.org/abs/2606.07592)

### Inferences
- **BCA's posterior mean is a CRC bound, and its quantile is an LTT/RCPS-style certificate.** With uniform weights, the expected BCA risk is (n·R̂_n + B)/(n+1). That is exactly CRC's left-hand side, so a BCA-mean gate inherits CRC's marginal guarantee, while BCA's upper quantile gives a data-conditional (PAC-style) certificate.
  - The user's single-decision finding (the plain mean beat the 90% quantile for expected return) is what this predicts. The quantile pays in return for protecting the calibration set in hand. Snell & Griffiths' 21.2% versus 0.03% exceedance rates show what the mean gives up.
- **To calibrate an offline conservatism ladder with LTT,** every candidate λ (SPIBB's N∧, TD3+BC's α, CQL's penalty weight, IQL's τ) needs a held-out loss that can be evaluated offline.
  - Outcome-based losses exist only for logged actions. So the per-λ test is computable only where the λ-policy's decisions are covered by held-out data, or via importance-weighted or OPE estimates with valid uncertainty. CDT names this same bottleneck: counterfactual evaluation of every λ.
- **Ordered ladders avoid a multiplicity penalty.** If risk is monotone in conservatism, fixed-sequence testing from the most to the least conservative λ (stop at the first failure) needs no Bonferroni penalty. An unordered grid of K candidates pays roughly log K in a union-bound/Bonferroni version.
- **A second selection step reintroduces the optimizer's curse.** LTT/CRC certify risk, not return. Picking the best-return λ among the certified ones (aLTT's "post-calibration selection" step) is a point-estimate selection and is exposed to the optimizer's curse (Section 5) unless it uses separate data.

### Gaps
- I found no paper that uses LTT or CRC to pick an offline RL conservatism hyperparameter from offline data alone with a guarantee on deployed return. aLTT tests online; UNIQ's conformal step calibrates uncertainty, not the choice of κ.
- I found no head-to-head evidence that LTT- or CRC-selected hyperparameters yield higher return than point-estimate selection; the papers measure risk control and TPR or set size.
- I did not verify the publication venue of Snell & Griffiths (2025), or the publication year of the IJRR version of Luo et al.
- I did not extract numeric results from LTT's own experiments.
- Ren et al.'s "Robots that ask for help" (KnowNo, CoRL 2023) is cited by CDT as a conformal ask-for-help method for LLM planners; I did not verify its numbers.

## 2. Conformal decision theory (Lekeufack, Angelopoulos, Bajcsy, Jordan & Malik, ICRA 2024): calibrating decisions directly, what it guarantees, and reported results

### Takeaway
CDT calibrates a decision's conservatism parameter λ on the realized decision loss, rather than calibrating prediction sets.
- **Online:** a one-line controller guarantees average realized loss ≤ ε + O(1/t) on any sequence, even an adversarial one, provided a safe fallback exists.
- **Batch:** the exchangeable version is CRC applied to decisions, with E[loss] ≤ ε.
- **Results:** robot navigation reached the goal about 29% faster than an adaptive conformal prediction (ACI) prediction-set planner while staying collision-free. No guarantee on utility is given, and every experiment uses the online controller.

### Cited Findings
- Citation: Lekeufack, Angelopoulos, Bajcsy, Jordan & Malik, "Conformal Decision Theory: Safe Autonomous Decisions from Imperfect Predictions," ICRA 2024 (Yokohama), pp. 11668–11675; arXiv:2310.05921 — [researchr](https://researchr.org/publication/LekeufackABJM24); [arXiv:2310.05921](https://arxiv.org/abs/2310.05921)
- Key idea: "instead of calibrating prediction sets for coverage, we directly calibrate decisions for low risk." A family of decisions D_t^λ is indexed by a "conformal control variable" λ that runs from least to most conservative, with bounded loss L ∈ [0,1]. The claimed benefit is the "ability to ignore irrelevant uncertainty, only accounting for that which affects decisions" — [arXiv:2310.05921v3](https://arxiv.org/html/2310.05921v3)
- Assumption, "eventually safe": some λ^safe and K exist such that any K consecutive steps with λ ≤ λ^safe have average loss ≤ ε^safe. The authors note this is weaker than the K = 1 assumption used in prior online conformal proofs — [arXiv:2310.05921v3](https://arxiv.org/html/2310.05921v3)
- Theorem 1 (conformal controller): update λ_{t+1} = λ_t + η(ε − ℓ_t). If ε^safe ≤ ε and λ_1 ≥ λ^safe − η, then for any realization, R̂_t ≤ ε + ((λ_1 − λ^safe)/η + K)/t for all t ≥ K. The update resembles ACI but acts on λ rather than on a quantile — [arXiv:2310.05921v3](https://arxiv.org/html/2310.05921v3)
- Batch CDT: with n+1 exchangeable decision functions and monotone loss, λ̂ = sup{λ : (1/n)Σ_i L(D_i(λ)) ≤ ε − (1−ε)/n}. Corollary 2 (via CRC): E[L(D_{n+1}(λ̂))] ≤ ε.
  - The paper names offline RL as a batch setting. Computing λ̂ requires iterating "through a sequence of counterfactual decisions … and evaluat[ing] what the loss would have been," which "restricts the applications."
  - Source: [arXiv:2310.05921v3](https://arxiv.org/html/2310.05921v3)
- Robot navigation, Stanford Drone Dataset (nexus_4 video, risk threshold ε = 2 m):

  | Method | Reached goal? | Time (s) | Safe? |
  | --- | --- | --- | --- |
  | Aggressive | Yes | 8.57 | No (min distance 0.16 m) |
  | Conservative | Never | — | Yes |
  | ACI, η = 0 / 0.01 / 0.1 | Yes | 27.17 / 26.67 / 24.73 | No |
  | Conformal controller, η = 50 | Yes | 20.03 | No |
  | Conformal controller, η = 100 / 500 / 1000 | Yes | 17.4 / 17.33 / 16.17 | Yes |

  The authors summarize this as "substantially (~29%) faster at reaching the goal than the ACI algorithm" — [arXiv:2310.05921v3](https://arxiv.org/html/2310.05921v3)
- Manufacturing (conveyor speed, target ε = 0.05 failed-grasp rate, 1,000 runs, horizon T = 2000): the conformal controller "performs as well as the Oracle-Loss," never exceeds ε, and keeps high throughput. Bandit baselines (UCB on utility, LCB on loss) both violate the risk threshold — [arXiv:2310.05921v3](https://arxiv.org/html/2310.05921v3)
- Stock trading (simulated geometric Brownian motion, 100 five-year runs, annual loss threshold ε = 25%): the controller stays at the threshold. With a good predictor (ρ = 0.1) its returns are second only to Greedy, which violates the loss threshold. ACI is "highly conservative," with loss well below threshold and limited gains — [arXiv:2310.05921v3](https://arxiv.org/html/2310.05921v3)
- Stated limitations: "no guarantees on utility are provided." For constrained utility maximization (maximize predicted utility subject to predicted loss ≤ λ_t), the method "will work as long as we revert to a safe decision if λ_t ≤ λ^safe" — [arXiv:2310.05921v3](https://arxiv.org/html/2310.05921v3)

### Inferences
- **CDT's batch rule is the decision-level analogue of the user's deviation gate.** λ is deviation aggressiveness and the loss is a harm indicator (deviation outcome below the fallback's value). With a 0/1 harm loss, batch CDT gives a CRC-type marginal guarantee. That matches BCA's posterior mean, not BCA's quantile certificate.
- **Its empirical advantage matches the user's finding.** CDT gains by calibrating directly on decision loss instead of on prediction-set coverage. This is the same lesson as the user's result that a decision-referenced gate (outcome versus fallback value) beats width-based penalties.
- **An online version fits deployment.** CDT's eventual-safety assumption needs a fallback that brings the loss under ε; the user's setting has a safe fallback action. A deployment-time conformal controller on the deviation threshold would therefore carry a finite-time bound on the realized harm rate, even under drift.

### Gaps
- The paper does not empirically evaluate the batch (offline) variant; all three experiments use the online update.
- It has no offline RL experiments and no return guarantees.
- I found no follow-up applying batch CDT to offline policy selection.

## 3. Conformal selection with FDR control (Jin & Candès 2023, JMLR; Jin & Candès, weighted conformal p-values): could it select the set of states or actions where a policy may deviate, while controlling the fraction of harmful deviations? How much more powerful is FDR control than per-decision or family-wise control?

### Takeaway
cfBH (Jin & Candès 2023) selects the test units whose unobserved outcome exceeds a unit-specific threshold c_j, with finite-sample false discovery rate (FDR) ≤ q. Its weighted version (WCS) does the same under covariate shift. This is a ready template for "deviate at state s only if the deviation's outcome beats the fallback's value, keeping the expected fraction of harmful deviations among those allowed ≤ q."
- **Versus per-decision control:** naive per-unit conformal selection does not control the harmful fraction; about 80% false discoveries at 90% confidence on an HIV dataset.
- **Versus family-wise control:** Bonferroni "nearly has no power" in their simulations.

I found no RL application.

### Cited Findings
**Selection by prediction with conformal p-values (cfBH)**
- Citation: Jin & Candès, "Selection by Prediction with Conformal p-values," *JMLR* 24(244):1–41, 2023; arXiv:2210.01408 — [Candès publications](https://candes.su.domains/publications/); [arXiv:2210.01408](https://arxiv.org/abs/2210.01408)
- Goal: find as many test units with Y_{n+j} > c_j as possible while FDR = E[#{selected with Y ≤ c} / max(1, #selected)] ≤ q. Hypotheses H_j: Y_{n+j} ≤ c_j are random, because they concern unobserved outcomes — [arXiv:2210.01408](https://arxiv.org/abs/2210.01408)
- Algorithm: use a nonconformity score V that is monotone in y. Compute p_j = [#{i : V_i < V̂_{n+j}} + U_j·(1 + #{ties})]/(n+1), with V̂_{n+j} = V(X_{n+j}, c_j), then run Benjamini–Hochberg (BH) at level q.
  - Theorem 2.3: FDR ≤ q for i.i.d. calibration and test data with a stated mutual-independence condition; the paper extends this to exchangeable data. The p-values need not satisfy P(p_j ≤ α | H_j true) ≤ α, because the nulls are random, yet BH still controls FDR.
  - "In many cases, the procedure selects candidates whose predictions are above a data-dependent threshold."
  - Source: [arXiv:2210.01408](https://arxiv.org/abs/2210.01408)
- Remark 2.2: p_j is the smallest α at which a one-sided split-conformal interval excludes c_j. That interval is the quantile of n atoms at the calibration scores plus one atom at −∞, each with weight 1/(n+1) — [arXiv:2210.01408](https://arxiv.org/abs/2210.01408)
- **Per-decision (naive) selection fails.** On HIV drug screening (41,127 molecules, about 3% active), selecting units whose conformal prediction set is {1} gave "an error rate of about 80%" at 1−α = 0.90 ("4 out of 5 'discoveries' are false"), and FDR above 35% even at 1−α = 0.99. Marginal coverage does not carry over to the selected units — [arXiv:2210.01408](https://arxiv.org/abs/2210.01408)
- **FDR versus FWER.** A Bonferroni-corrected version (select if C_{1−q/m}(X) = {1}) controls the probability of any false selection, which upper-bounds FDR. On HIV, cfBH yields "almost exact FDR control and much higher power than Bonferroni's" (the numbers are in a figure). In simulations, "the Bonferroni correction nearly has no power (even if we set V(x,y) to be the same as the most powerful BH_clip)" — [arXiv:2210.01408](https://arxiv.org/abs/2210.01408)
- HIV results for BH_clip, averaged over 100 splits, about 8,000 test molecules:

  | Target q | Realized FDR | Power | Units selected |
  | --- | --- | --- | --- |
  | 0.1 | 0.0957 | 0.0788 | 26.5 |
  | 0.2 | 0.196 | 0.174 | 64.2 |
  | 0.5 | 0.495 | 0.410 | 240 |

  At q = 0.1, a predicted score of about 0.8 was needed to be selected — [arXiv:2210.01408](https://arxiv.org/abs/2210.01408)
- Practical guidance: power depends on the prediction model and the noise level. A sensible choice of q is "moderately larger than the marginal proportion of positive Y's" — [arXiv:2210.01408](https://arxiv.org/abs/2210.01408)

**Weighted conformal p-values under covariate shift (WCS)**
- Citation: Jin & Candès, "Model-free selective inference under covariate shift via weighted conformal p-values," *Biometrika* 113(1), asaf066, DOI 10.1093/biomet/asaf066. Candès's publication list dates it 2025; the MaRDI portal lists 29 April 2026. arXiv:2307.09291 (2023) — [Candès publications](https://candes.su.domains/publications/); [MaRDI](https://portal.mardi4nfdi.de/wiki/Model-free_selective_inference_under_covariate_shift_via_weighted_conformal_%5C(p%5C)-values); [arXiv:2307.09291](https://arxiv.org/abs/2307.09291)
- Setting: calibration data come from P and test data from Q, with dQ/dP(x,y) = w(x) (pure covariate shift).
  - The weighted conformal p-values "cannot be passed onto classical multiple testing procedures since they may not obey a well-known positive dependence property." The new procedure WCS "controls false discovery rate (FDR) in finite samples."
  - It also covers inference on multiple individual treatment effects (counterfactual outcomes) and outlier detection under inlier shift.
  - Source: [arXiv:2307.09291](https://arxiv.org/abs/2307.09291)
- Empirically, the FDR stayed below nominal levels in drug-discovery tasks over 200 runs. A clipped score used the error budget almost fully and had "much higher power" than a residual score — [arXiv:2307.09291](https://arxiv.org/abs/2307.09291)

**Related selection results**
- Bai & Jin, "Optimized Conformal Selection" (OptCS), arXiv:2411.17983 (2024). Choosing among models on the same data ("Greedy" double-dipping) produced a "drastic violation of FDR." OptCS keeps finite-sample FDR control after data-driven model selection or full-data training. "For all 8 settings and across all nominal FDR levels," OptCS-MSel_homo "consistently outperforms all competing methods that maintain valid FDR control" — [arXiv:2411.17983](https://arxiv.org/abs/2411.17983)
- aLTT (hyperparameter version): FDR-controlled certification gave a higher TPR than FWER on the TD3+BC λ grid: 0.40 versus 0.32 at the least adaptive acquisition setting, rising to 0.85 for both with evidence-driven acquisition. "The TPR of the schemes under FDR control is larger than that obtained under FWER control" — [arXiv:2409.15844](https://arxiv.org/abs/2409.15844)
- Gu & Koenker, "Invidious Comparisons: Ranking and Selection as Compound Decisions," *Econometrica* 91(1):1–41 (2023), arXiv:2012.12550. They select "most meritorious" units from noisy estimates "of potentially heterogeneous precision" subject to constraints on the expected number selected and the false discovery proportion.
  - "For fixed α and homogeneous variance, posterior mean, posterior tail probability and posterior tail expectation all yield the same ranking and therefore the same selection."
  - "When variances are heterogeneous, the linear shrinkage estimator provides a different ranking than the posterior tail probability rules."
  - FDR estimation requires knowing the mixing distribution G, and a misplaced Gaussian assumption "may lead to … procedures that fail to control for false discovery."
  - Source: [arXiv:2012.12550](https://arxiv.org/abs/2012.12550); [Econometrica](https://www.econometricsociety.org/publications/econometrica/2023/01/01/Invidious-Comparisons-Ranking-and-Selection-as-Compound-Decisions)

**RL baseline for "where to deviate": SPIBB**
- Citation: Laroche, Trichelair & Tachet des Combes, "Safe Policy Improvement with Baseline Bootstrapping," ICML 2019, PMLR 97; arXiv:1712.06924 — [PMLR](https://proceedings.mlr.press/v97/laroche19a.html)
- Rule: the policy must follow the baseline on the "bootstrapped set," meaning state–action pairs with counts below N∧. Elsewhere it optimizes freely. Π_b-SPIBB is a ζ-approximate safe policy improvement with probability 1−δ.
  - "N∧ is not a property of the dataset, but a hyper-parameter," and the method allows safe improvement "if N∧ is chosen large enough to ensure safety and small enough to ensure improvement."
  - Source: [arXiv:1712.06924](https://arxiv.org/abs/1712.06924)
- Results over 100,000 repetitions, reporting mean and 1%-CVaR:
  - Grid search was used for all baselines' hyperparameters.
  - SPIBB was safe except Π_b-SPIBB with low N∧ on 10-trajectory datasets, and "the bigger the N∧, the more conservative SPIBB gets."
  - Basic RL failed because pairs that were "observed only once or twice" let the maximum-likelihood MDP produce "policies falsely taking advantage of those chimaeras."
  - RaMDP was "very efficient" only at κ_adj = 0.003. It acts like Basic RL at ≤ 0.002 and degenerates at ≥ 0.004, and its good mean "hides some catastrophic runs."
  - Robust MDP and RaMDP safety tests "always failed."
  - Source: [arXiv:1712.06924](https://arxiv.org/abs/1712.06924)

### Inferences
- **Mapping to the user's gate.**
  - A unit is a (state, candidate deviation action) pair where the actor wants to leave the fallback. The outcome Y_j is the realized return-to-go after deviating, or the deviation's advantage. The threshold c_j is the fallback's value at that state, which the brief says the critic estimates well. Calibration data are held-out logged transitions in which deviation actions were taken.
  - cfBH then returns the set of allowed deviations with E[harmful fraction among allowed] ≤ q. It typically reduces to a calibrated, data-dependent cutoff on predicted advantage. That is a calibrated replacement for SPIBB's count rule N∧ and for a hand-set margin.
- **The required shift is a covariate shift.** Logged deviation states are not exchangeable with the states where the learned actor deviates.
  - If the outcome distribution given (state, action) does not change, the shift from logging to deployment is a covariate shift with w = d_target(s,a)/d_log(s,a). This is the setting WCS was built for.
  - The action part of w is known if the behavior policy is known; the state part needs estimation (for example, DICE-style distribution-correction ratios), and validity then depends on that estimate.
  - Rarely logged actions mean large, unstable weights, which will cost power.
- **Per-deviation certificates are not enough with many candidates.** Applying a separate BCA certificate to each of m candidate deviations controls each deviation's own error rate, not the harmful fraction among those allowed. When most candidates are harmful and few are good, the allowed set can be dominated by false passes, the same mechanism as Jin & Candès's 80% naive FDR. BH-style selection fixes this at little power cost; Bonferroni fixes it at a large one.
- **FDR's power advantage is the clearest documented case of calibrated selection choosing better.** It allows many more truly good deviations at a fixed harm budget. That raises return only when the harm budget is the binding constraint.
- **SPIBB's results are optimizer's-curse evidence.** Its Basic-RL failure on rarely observed pairs shows that unregularized optimization over sparsely logged actions selects lucky outliers. Its sensitivity to N∧ and κ_adj shows why choosing such hyperparameters is itself the hard part.

### Gaps
- I found no RL or offline RL paper applying cfBH or WCS to choose deviation states or actions.
- The exact Bonferroni power numbers on HIV are only in a figure, which I could not read.
- I did not find quantified WCS power loss under extreme importance weights.
- Open question: is FDR (an expectation over selections) the right safety notion for deviations, compared with a per-episode or per-state harm bound?

## 4. Offline policy and hyperparameter selection (Paine et al. 2020; Yang, Dai, Nachum, Tucker & Schuurmans, AISTATS 2022; Konyushkova et al. 2021; pessimistic or high-confidence selection): choosing by point estimates versus lower confidence bounds

### Takeaway
Large empirical studies select by point estimates (FQE) and find this works when candidates stay close to the behavior policy. Failures concentrate in the least-regularized algorithm, which looks best and is worst.
- **The one direct mean-versus-bound comparison** (Yang et al.) finds lower-bound ranking "almost always" worse than the mean for top-k metrics. The best rule there is posterior-based selection matched to the downstream metric.
- **Theory** (Xiao et al.; Rashidinejad et al.) says no index rule dominates. LCB wins when the data cover a good option heavily and the tempting candidates are rarely sampled; optimism or greedy wins when the good options are the rarely sampled ones.
- **Practical safeguards:** held-out single-candidate safety tests (HCPI) and repeated data splitting (Nie et al.).

### Cited Findings
- **Paine et al. 2020**, "Hyperparameter Selection for Offline Reinforcement Learning," arXiv:2007.09055 (DeepMind).
  - Setup: 256 policies per task (64 BC, 128 CRR, 64 D4PG), with ground truth from 100 online episodes per policy.
  - Critics learned during offline RL overestimate on all tasks; D4PG (no behavior regularization) overestimates most. "D4PG deceptively looks to produce the best policy, while actually being the worst" on humanoid tasks.
  - FQE re-evaluation "significantly reduces over-estimation." For BC and CRR, FQE's V̂(s₀) rank correlation is "above 0.9 for most tasks," and regret@5 is "often close to zero." "Better performing policies overestimate less." Only point estimates were compared.
  - Source: [arXiv:2007.09055](https://arxiv.org/abs/2007.09055)
- **Yang, Dai, Nachum, Tucker & Schuurmans**, "Offline Policy Selection under Uncertainty," arXiv:2012.06919.
  - BayesDICE estimates posteriors over policy values; OfflineSelect then chooses the ranking that maximizes the expected downstream score (top-k precision, accuracy, correlation or regret) by sampling from those posteriors.
  - "Policy ranking based on the lower bound estimates does not always lead to lower (top-k) regret." The Figure 3 caption says: "Interestingly, the lower-bound point estimate almost always performs worse than the mean or the upper bound."
  - Matching the selection criterion to the evaluation metric "is empirically the best approach." It is near Bayes-optimal in a two-armed bandit, and BayesDICE+OfflineSelect beats DualDICE point estimates and high-confidence lower bounds on top-k regret and correlation (Bandit, Reacher).
  - Source: [arXiv:2012.06919](https://arxiv.org/abs/2012.06919)
- **Konyushkova et al.**, "Active Offline Policy Selection," arXiv:2106.10251.
  - Method: a Gaussian process over policy values, using OPE estimates as noisy prior observations and a kernel based on action distances between policies, then UCB-style choice of which policy to test online. Candidate sets: K = 50 (dm-control) and K = 200 (manipulation playground and Atari), 100 repeats.
  - On the manipulation tasks, A-ops "cut[s] the OPE regret by more than half before executing 100 trajectories."
  - Pure online uniform evaluation was worse than OPE alone at these budgets. The kernel helped in 11 of 12 settings. On a real robot, selected checkpoints improved on the previous best by 1.55%–4.95%.
  - Source: [arXiv:2106.10251](https://arxiv.org/abs/2106.10251)
- **Lee, Tucker, Nachum & Dai**, "Model Selection in Batch Policy Optimization," arXiv:2112.12320.
  - Three error sources (approximation, statistical complexity, coverage): "no batch policy optimization algorithm can achieve a guarantee addressing all three simultaneously," but relaxing any one gives near-oracle inequalities for the other two.
  - One rule is "pessimistic regarding values but optimistic across model classes."
  - In experiments (20 trials), an 80/20 hold-out "performs consistently better than the SLOPE method with small data."
  - Source: [arXiv:2112.12320](https://arxiv.org/abs/2112.12320)
- **Nie, Flet-Berliac, Jordan, Steenbergen & Brunskill**, "Data-Efficient Pipeline for Offline RL with Limited Data," NeurIPS 2022, arXiv:2210.08642.
  - "A single partition can have a notable failure rate in identifying the best algorithm-hyperparameter."
  - SSR-RRS: repeated random subsampling with K splits, OPE on each validation set, averaging, then retraining on all data. With K = 1, "policies are picked quite uniformly; many of which are performing poorly"; with K = 5, "higher-performing policies are selected much more frequently."
  - The gain shrinks as data grow (Sepsis simulator with 200, 1,000 and 5,000 patients) but remains at 5,000.
  - Source: [NeurIPS 2022](https://proceedings.neurips.cc/paper_files/paper/2022/hash/5ee7ed60a7e8169012224dec5fe0d27f-Abstract-Conference.html); [arXiv:2210.08642](https://arxiv.org/abs/2210.08642)
- **High-confidence selection: HCPI** (Thomas, Theocharous & Ghavamzadeh, "High Confidence Policy Improvement," ICML 2015, PMLR 37:2380–2388).
  - Testing many candidates on the same data inflates the error: with two policies "its error rate can be as large as 2δ," and more candidates "further increase the error rate."
  - Fix: split into D_train, used to pick a single candidate, and D_test, used for one safety test that returns the candidate only if its lower bound clears ρ₋. The candidate is picked to maximize predicted performance subject to a predicted lower bound computed for the actual test size.
  - Without regularization this "has a tendency to overfit": it picks a policy predicted to pass easily that "actually fails the subsequent safety test." A k-fold-cross-validated mixing constraint toward the initial policy mitigates this.
  - Mountain Car: the confidence-interval version achieved safe improvement with 900 trajectories, 400 with k-fold.
  - Empirical error rates (a policy wrongly declared safe) were zero in gridworld and digital marketing and "below 0.6%" in Mountain Car, versus 5% allowed.
  - Source: [PMLR](https://proceedings.mlr.press/v37/thomas15.html); [ML Anthology](https://mlanthology.org/icml/2015/thomas2015icml-high/)
- **Theory on LCB versus greedy versus UCB: Xiao, Wu, Lattimore, Dai, Mei, Li, Szepesvári & Schuurmans**, "On the Optimality of Batch Policy Optimization Algorithms," arXiv:2104.02293.
  - Minimax regret is Ω(1/√(min_i n_i)), where n_i is arm i's sample count. "Any confidence-adjusted index algorithm is minimax optimal, whether it be optimistic, pessimistic or neutral." Instance-dependent optimality is impossible.
  - LCB is weighted-minimax optimal when instances are weighted by the difficulty of predicting the optimal value. "The pessimistic choice is beneficial when the batch dataset contains enough information that is good for predicting the optimal value."
  - Illustration: if the heavily played set "contains at least one good arm but no bad arm, LCB will select a good played arm … while UCB will select a bad unplayed arm"; the reverse holds when only bad arms were played.
  - Simulations with 100 arms: LCB wins when an ε-greedy behavior policy plays the best or 10th-best arm often; UCB wins when good arms are played rarely.
  - Source: [arXiv:2104.02293](https://arxiv.org/abs/2104.02293)
- **Rashidinejad, Zhu, Ma, Jiao & Russell**, "Bridging Offline RL and Imitation Learning: A Tale of Pessimism," arXiv:2103.12021 (part published at NeurIPS 2021).
  - Proposition 1: the empirical best arm has constant error "even in the favorable case when C* ≈ 1," because "a less-explored sub-optimal arm might have high empirical mean just by chance." LCB secures Õ(√(1/N)) there. (C* measures how far the data distribution is from the optimal policy; C* = 1 means expert data.)
  - In multi-armed bandits with C* ∈ [1,2), the most-played arm achieves an exp(−N·KL) rate, and LCB "cannot be adaptively optimal" in both regimes.
  - In contextual bandits with at least two contexts, LCB is adaptively optimal: rate S/N for near-expert data and √(S(C*−1)/N) otherwise. Value iteration with LCB is nearly optimal in MDPs.
  - Source: [arXiv:2103.12021](https://arxiv.org/abs/2103.12021)

### Inferences
- **Evidence that LCB selection raises the chosen policy's value is mostly theoretical and depends on the instance.** The empirical paper that compares directly (Yang et al.) points the other way for ranking metrics.
- **The user's setting is the regime where theory predicts pessimism wins.** The fallback is the heavily logged, well-evaluated option, and deviations are sparsely logged arms. That is Rashidinejad's failure instance for the empirical best arm and Xiao's "played set contains a good arm" case. A quantile or LCB gate should beat plain means here when many deviation candidates compete, not when one deviation is compared with the fallback.
- **HCPI suggests a two-stage design for a deviation gate.** Propose deviations on a training split using a lower bound predicted for the held-out test size, then certify on the untouched held-out split. Gating many deviations on the same held-out outcomes used to choose them would reintroduce the multiple-comparisons inflation HCPI warns about.
- **Split randomness is a separate source of mis-selection.** Nie et al. show that a single split can mis-select by itself. A BCA posterior on one held-out split does not capture variability across splits, so repeated splitting (or cross-fitting) is complementary.
- **Most of the selection gain may come from independent evaluation, not from a lower bound.** Paine et al.'s D4PG case is a selection-by-point-estimate failure caused by evaluation errors that differ across candidates and are largest for policies that deviate most. A separate evaluator (FQE) fixed most of it.

### Gaps
- Sources I did not fetch: "Pessimistic Model Selection for Offline Deep RL" (arXiv:2111.14346); Tang & Wiens 2021 (model selection for offline RL in healthcare); Kumar et al. 2021 (offline robotic RL workflow); Zhang & Jiang 2021 (BVFT); Liu et al. 2023, "When is offline policy selection sample efficient?" (arXiv:2312.02355).
- Venues I did not verify from the text I read: Yang et al. (AISTATS 2022, per the brief), Konyushkova et al. (NeurIPS 2021), Lee et al. (ICML 2022), Xiao et al. (ICML 2021). Paine et al. appears to be an arXiv preprint only.
- Yang et al.'s regret and correlation numbers appear only in figures.

## 5. The optimizer's curse (Smith & Winkler 2006) and related results: when does selecting by a lower confidence bound or a shrunk estimate raise the true expected value of the chosen option compared with selecting by the point estimate?

### Takeaway
Choosing the argmax of unbiased estimates is biased upward: 0.56σ for 2 equal options and 1.54σ for 10. The Bayes fix is to choose by posterior mean.
- **Equal precision:** the fix changes which option is chosen only when candidates differ in precision (or prior mean). With equal precision and an exchangeable prior, shrinkage, posterior tail probability and equal-width lower bounds rank exactly like the raw estimates, so they cannot raise the chosen option's value.
- **Unequal precision:** the raw argmax over-selects noisy, rarely sampled candidates, and pessimism or shrinkage can raise true value substantially (constant versus vanishing regret in Rashidinejad's example). It is not uniformly better (Xiao et al.; Yang et al.).

### Cited Findings
**Smith & Winkler**
- Citation: Smith & Winkler, "The Optimizer's Curse: Skepticism and Postdecision Surprise in Decision Analysis," *Management Science* 52(3):311–322, 2006, DOI 10.1287/mnsc.1050.0451 — [INFORMS](https://pubsonline.informs.org/doi/10.1287/mnsc.1050.0451); [author PDF](https://jimsmith.host.dartmouth.edu/wp-content/uploads/2022/04/The_Optimizers_Curse.pdf)
- Proposition 1: for conditionally unbiased estimates V_i, choosing i* = argmax V_i gives E[μ_{i*} − V_{i*}] ≤ 0, strictly negative if there is any chance of choosing a wrong alternative, with no normality assumption — [author PDF](https://jimsmith.host.dartmouth.edu/wp-content/uploads/2022/04/The_Optimizers_Curse.pdf)
- Expected disappointment for n alternatives with equal true values and i.i.d. N(0,1) estimation errors, in units of σ: 0.56 (n=2), 0.85 (3), 1.03 (4), 1.16 (5), 1.54 (10) — [author PDF](https://jimsmith.host.dartmouth.edu/wp-content/uploads/2022/04/The_Optimizers_Curse.pdf)
- Separation Δ between true values, 3 alternatives (corrected numbers from the authors' 2007 erratum, which supersede the paper's Figure 3):

  | Separation Δ (σ) | Expected disappointment (σ) | P(correct choice) |
  | --- | --- | --- |
  | 0 | 0.85 | 0.33 |
  | 1 | 0.54 | 0.73 |
  | 2 | 0.21 | 0.92 |
  | 3 | 0.07 | 0.98 |

  Source: [correction](https://jimsmith.host.dartmouth.edu/wp-content/uploads/2022/04/pds_correction.pdf)
- Correlation: positive correlation among the estimates reduces disappointment, while positive correlation among the true values increases it. With 4 alternatives and no correlation, disappointment is 0.73σ — [author PDF](https://jimsmith.host.dartmouth.edu/wp-content/uploads/2022/04/The_Optimizers_Curse.pdf)
- Real-data simulation on Eastman Kodak's 38 decision analyses (true value added $487M): estimate errors of 5%, 10% and 25% inflate the claimed value added to $555M, $678M and $1.111B (+14%, +39%, +128%) — [author PDF](https://jimsmith.host.dartmouth.edu/wp-content/uploads/2022/04/The_Optimizers_Curse.pdf)
- Fix: posterior means v_i = α_i·V_i + (1−α_i)·μ̄_i, with α_i = 1/(1 + σ²_{V_i}/σ²_{μ_i}). Proposition 2: choosing argmax v_i gives zero expected disappointment, conditionally and unconditionally — [author PDF](https://jimsmith.host.dartmouth.edu/wp-content/uploads/2022/04/The_Optimizers_Curse.pdf)
- When the fix changes the choice:
  - With a common prior mean and variance ratio, "the ranking of alternatives is not changed"; only the magnitudes shrink (e.g., the top estimate of $80.0M shrinks to $66.7M at a 20% variance ratio).
  - With differing variance ratios (20% versus 50%), "the alternative with the highest value estimate V_i is not preferred," and "alternatives whose values are hard to estimate may be passed over by alternatives with lower, but more accurate, value estimates." Differences in prior means can also reorder alternatives.
  - The hierarchical version shrinks toward the mean of all estimates.
  - Source: [author PDF](https://jimsmith.host.dartmouth.edu/wp-content/uploads/2022/04/The_Optimizers_Curse.pdf)
- Shrinkage differs from ambiguity aversion: shrinkage "will adjust low values upward toward the prior mean," whereas ambiguity aversion "leads only to values being adjusted down" — [author PDF](https://jimsmith.host.dartmouth.edu/wp-content/uploads/2022/04/The_Optimizers_Curse.pdf)

**Related results**
- Gu & Koenker (2023): with homogeneous variance, posterior mean, posterior tail probability and posterior tail expectation give identical rankings and selections. With heterogeneous variances, shrinkage and tail-probability rules rank differently — [arXiv:2012.12550](https://arxiv.org/abs/2012.12550)
- Rashidinejad et al.: under unequal sample counts the empirical best arm has constant error even with near-expert data, while LCB's regret vanishes — [arXiv:2103.12021](https://arxiv.org/abs/2103.12021)
- Xiao et al.: whether LCB, greedy or UCB is best depends on whether the heavily sampled set contains a good arm — [arXiv:2104.02293](https://arxiv.org/abs/2104.02293)
- Yang et al.: with posteriors available, lower-bound ranking "almost always performs worse than the mean or the upper bound" for top-k selection metrics in their Bandit and Reacher experiments — [arXiv:2012.06919](https://arxiv.org/abs/2012.06919)
- Paine et al.: with candidate-specific evaluation errors, the point-estimate statistic picked D4PG, which "deceptively looks to produce the best policy, while actually being the worst" — [arXiv:2007.09055](https://arxiv.org/abs/2007.09055)

### Inferences
**When does pessimistic or shrunk selection raise the true value of the chosen option?**
1. **Candidates differ in precision.** Unequal counts or unequal noise are the necessary condition: with equal widths, an LCB or shrinkage rule picks the same argmax as the mean.
2. **There are many candidates.** The order-statistic inflation grows with n.
3. **True values are close together.** Disappointment is largest at Δ = 0.
4. **Errors are independent across candidates.** Correlated errors cancel.
5. **The well-estimated options include a good one** (a good, heavily logged fallback). This makes rejecting a good-but-noisy candidate cheap, while accepting a lucky bad one is costly; it is Xiao's LCB-favorable regime.

Pessimism hurts when the good options are the noisy ones (UCB's regime), or when the metric rewards finding the best rather than avoiding the worst (Yang's top-k results).

**BCA-specific reading**
- **Mean versus quantile as penalties.** BCA's uniform-weight posterior mean is (n·mean + B)/(n+1). That is a count-dependent pessimistic shrinkage toward the worst case, of size about (B − mean)/(n+1), i.e. O(1/n). The 90% quantile adds a penalty of order 1/√n, like an LCB.
- **Both only adjust downward.** Neither is Smith & Winkler's shrinkage toward a prior mean; both behave like ambiguity aversion in Smith & Winkler's terms. For maximizing expected return across many candidates with unequal n, the principled rival is empirical-Bayes or hierarchical shrinkage of held-out means toward the pooled mean of comparable candidates. A BCA claim of improving return should be tested against that rival as well as against plain means.
- **Why the single-decision result holds.** With one deviation candidate against a well-estimated fallback value, there is no maximum over noisy estimates, hence no optimizer's curse. Deviating iff the posterior-mean outcome exceeds the fallback value is the Bayes rule for expected return, which is consistent with the user's finding that the plain mean beats the quantile.
- **Where the curse re-enters.** It returns as soon as the gate chooses among several deviation actions per state, ranks many states, or tunes the threshold, margin or hyperparameter on the same held-out outcomes it then uses to certify.

#### Synthesis across questions: setups where calibrated selection has a real advantage over point estimates
- **Many candidates with sparse, unequal evidence and a good, well-covered fallback.** This is the gate across many rarely logged deviations. The argmax of plain means selects lucky outliers (Rashidinejad's constant error; SPIBB's "chimaeras"; Paine's D4PG); an LCB or quantile gate, or count-dependent shrinkage, avoids this.
- **A harm-budget constraint on a set of deviations.** Here FDR-controlled selection (cfBH or WCS) admits far more good deviations than family-wise rules ("Bonferroni nearly has no power") at the same harm rate. Per-deviation certificates without multiplicity control do not bound the harmful fraction (about 80% false discoveries in the naive conformal example).
- **Choosing a conservatism hyperparameter or threshold that must satisfy a risk constraint.** LTT, CRC and batch CDT certify the constraint (e.g., help-rate or harm-rate budgets); aLTT shows FDR or adaptive testing certifies more good candidates.
  - Efficient calibration keeps more performance at the same risk: CDT about 29% faster than ACI; Snell & Griffiths' HPD tighter than RCPS at below 5% failure; RCIP 5–30% less human help.
  - On held-out data, a single-candidate held-out test (HCPI) avoids multiple-comparison inflation.
- **Where calibrated selection does not beat point estimates:** a single decision against a known fallback when maximizing expected return; equal evidence per candidate; and ranking objectives that reward finding the best (Yang et al.). In these cases the posterior mean, or the plain held-out mean, is the right rule. A calibrated quantile then buys only a data-conditional risk guarantee.

### Gaps
- Smith & Winkler do not quantify the gain in realized value from Bayesian re-ranking; they show only that the ranking changes and that the disappointment disappears. Choosing by posterior mean maximizing expected true value under the model is standard decision theory, not a numbered result I read in the paper.
- I found no offline RL study that measures the chosen policy's true value under LCB versus mean versus empirical-Bayes selection with deliberately unequal held-out sample sizes. Yang et al. (figures only) and Xiao et al. (synthetic bandits) come closest.
- I found no source that quantifies the threshold in candidate count or imbalance in sample size at which quantile selection starts to beat mean selection. That would need a targeted simulation in the user's toy settings.

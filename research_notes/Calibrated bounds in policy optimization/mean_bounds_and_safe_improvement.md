# Small-sample confidence bounds on a bounded mean (Gaffke / Learned-Miller–Thomas / BQ-CP) and their use in high-confidence policy improvement

Notes current to 14 Oct 2026. Conventions: α or δ is the allowed failure probability of a frequentist bound; β is the credibility level used by BQ-CP / WBCP (β = 1 − δ). "Gaffke/LMT bound" means the construction that puts uniform Dirichlet (uniform-spacing) weights on the n observations plus one extra atom at the known worst-case value. "My calculation" marks numbers I computed from the stated formulas; they are not quoted from any paper.

## Q1. What are the main small-sample bounds on the mean of a bounded variable, how are they built, which are proven, and how tight are they at n ≈ 10–100?

### Takeaway
The Gaffke/LMT bound is the tightest of the guaranteed bounds reviewed here at small n. For every sample it is at least as tight as Anderson's bound, and Anderson's is at least as tight as Hoeffding's (α ≤ 0.5). It was an open conjecture for about 20 years (proved only for Bernoulli and half-Bernoulli laws). A July 2026 preprint by Vlassis & Thomas claims a full proof. A second July 2026 preprint (Ming et al., including Ramdas and Waudby-Smith) explicitly relies on it, and a third (Bissias & Learned-Miller) proves an optimality property of the same bound. Phan et al. (ICML 2021) give a provably valid family that dominates Anderson. Betting confidence sequences (Waudby-Smith & Ramdas) are valid, anytime-valid, and tighter than Phan et al., but did not compare against Gaffke.

### Cited Findings

**Classical guaranteed bounds**
- Hoeffding upper confidence bound (UCB) for data in [a, b]: x̄ + (b − a)·√(ln(1/α)/(2n)). It does not need identically distributed samples. — [Phan, Thomas & Learned-Miller, ICML 2021 (PDF)](https://proceedings.mlr.press/v139/phan21a/phan21a.pdf); [Learned-Miller & Thomas, arXiv:1905.06208](https://arxiv.org/abs/1905.06208)
- Maurer–Pontil empirical Bernstein UCB: x̄ + √(2σ̂² ln(2/α)/n) + 7(b − a) ln(2/α)/(3(n − 1)). The range enters at rate 1/n instead of 1/√n. It needs n ≥ 2 and beats Hoeffding mainly when n is large and the variance is small relative to the range. — [Phan et al. 2021](https://proceedings.mlr.press/v139/phan21a/phan21a.pdf)
- Anderson (1969) builds a (1 − α) lower confidence band on the CDF from the one-sided Kolmogorov–Smirnov / DKW statistic. The bound is the mean of the stair-step CDF that puts mass as far right as the band allows. — [Phan et al. 2021](https://proceedings.mlr.press/v139/phan21a/phan21a.pdf)
- LMT's version of Anderson uses Massart's tight DKW constant, uᵢ = max{0, i/n − √(ln(1/α)/2n)}, valid for α ≤ 0.5. — [LMT](https://arxiv.org/abs/1905.06208)
- Anderson's UCB needs only an upper support bound, not a lower one. — [LMT §7](https://arxiv.org/abs/1905.06208)
- In off-policy evaluation, Anderson's lower bound on the mean depends on the largest *observed* sample rather than the worst-case range b. However, it requires i.i.d. data. — [Thomas, Theocharous & Ghavamzadeh, HCOPE, AAAI 2015](https://people.cs.umass.edu/~pthomas/papers/Thomas2015.pdf)
- Anderson ≤ Hoeffding for every sample when α ≤ 0.5 (LMT Theorem 2), strictly when n ≥ 3. Neither Anderson nor Maurer–Pontil dominates the other. — [LMT](https://arxiv.org/abs/1905.06208); [Phan et al. 2021](https://proceedings.mlr.press/v139/phan21a/phan21a.pdf)
- LMT also derive a sharpened Hoeffding variant, x̄ + (1 − z₁)·√(ln(1/α)/2n), that replaces the lower range bound with the smallest observation. It holds for i.i.d. data with α ≤ 0.5. — [LMT eq. 64](https://arxiv.org/abs/1905.06208)

**Gaffke / Learned-Miller–Thomas construction**
- **LMT form.** Sort the sample, z₁ ≤ … ≤ zₙ, and set z_{n+1} = 1 (the known upper bound). Draw U, the order statistics of n Uniform(0,1) variables. The "induced mean" is m(z, U) = Σ_{i=1}^{n+1} zᵢ(Uᵢ − U_{i−1}) = 1 − Σᵢ Uᵢ(z_{i+1} − zᵢ), with U₀ = 0 and U_{n+1} = 1. The bound m_α(z) is the (1 − α)-quantile of m(z, U). — [LMT eqs. 2–4, 8–10](https://arxiv.org/abs/1905.06208)
- **Interpretation.** m(z, u) is the mean of the "conservative completion": the largest-mean CDF passing through the points (zᵢ, uᵢ). — [LMT §3](https://arxiv.org/abs/1905.06208)
- **Computation.** Monte Carlo (Algorithm 1). LMT recommend about 10,000 draws for α = 0.05. Monte Carlo error can break coverage unless the number of draws is increased or the error is bounded explicitly. Closed-form simplex-volume formulas exist but are numerically unstable when values are tied. — [LMT §6](https://arxiv.org/abs/1905.06208)
- **Gaffke form.** K(x) = P{Σ_{i=1}^n xᵢDᵢ ≤ 1} with (D₀, D₁, …, Dₙ) ~ Dir(1, …, 1). In words: "augment the sample with a zero, average the resulting n + 1 values using uniform Dirichlet weights, and take the conditional probability that this average is at most 1". This is a test of H₀: E Xᵢ ≤ 1 for independent nonnegative Xᵢ; reject when K(X) ≤ α. Equivalently, K(x) = P{Σ(xᵢ − 1)Eᵢ ≤ E₀} with Eᵢ i.i.d. Exp(1). — [Vlassis & Thomas, arXiv:2607.08415](https://arxiv.org/abs/2607.08415)
- **Same bound, two forms.** Inverting Gaffke's test for data in [0,1] gives exactly the LMT interval; "the original order-statistic construction and the Gaffke inversion are two representations of the same bound". Its endpoints are quantiles of Dirichlet averages, and for Bernoulli data it reduces to Clopper–Pearson. — [Ming, Ramdas, Shen, Wang & Waudby-Smith, arXiv:2607.18661](https://arxiv.org/html/2607.18661)
- LMT learned of Gaffke (2005) after first posting; Aaditya Ramdas pointed it out. Citation: N. Gaffke, "Three test statistics for a nonparametric one-sided hypothesis on the mean of a nonnegative variable", *Mathematical Methods of Statistics* 14(4):451–467, 2005. — [LMT](https://arxiv.org/abs/1905.06208)

**Validity status: conjectured until July 2026**
- LMT only *conjectured* guaranteed coverage. They proved it for Bernoulli and "half-Bernoulli" laws (point masses at k and 1) for all n and α, and reported extensive simulations consistent with coverage. They also conjectured extensions to (−∞, 1] for upper bounds and [0, ∞) for lower bounds. — [LMT §§2, 5, 9](https://arxiv.org/abs/1905.06208)
- Phan et al. (ICML 2021): Gaffke's bound "gives remarkably tight bounds on the mean, but has eluded a proof of guaranteed coverage". — [Phan et al. 2021](https://proceedings.mlr.press/v139/phan21a/phan21a.pdf)
- Thomas, Learned-Miller & Phan, in their discussion of Waudby-Smith & Ramdas (JRSS-B, 2024): whether Gaffke's interval has guaranteed coverage "remains an open question". They note "substantial effort … finding distributions for which it does not exhibit coverage (with no success)". — [JRSS-B discussion contribution](https://academic.oup.com/jrsssb/article-abstract/86/1/51/7307586)
- Waudby-Smith & Ramdas likewise said the bound "has not been proved to be valid in general". — [WSR, JRSS-B PDF incl. discussion](https://academic.oup.com/jrsssb/article-pdf/doi/10.1093/jrsssb/qkad119/56961803/qkad119.pdf)
- Gaffke himself proved asymptotic behavior and reduced the problem to independent mean-one two-point laws. He proved n = 2 within that family and reported numerical checks up to n = 15. — [Vlassis & Thomas](https://arxiv.org/abs/2607.08415)
- H. Michael's preprint separates Gaffke's "weaker" conjecture (coverage for all n) from a "stronger" one (non-identically distributed variables, valid simultaneously for all means). It gives partial results but no proof. — [Michael, preprint](https://haben-michael.github.io/preprints/gaffke.pdf)
- **Claimed proof (July 2026).** Vlassis (Adobe Research) & Thomas (UMass), Theorem 1: if X₁, …, Xₙ are independent and nonnegative with E Xᵢ ≤ 1, then P{K(X) ≤ α} ≤ α for every α and every n ≥ 1. The variables need not be identically distributed or continuous.
  - Proof route: reduce to mean-one two-point mixtures (Lemma 7), build an inductively constructed dominating chain measure (Lemma 6), and use an exponential-transfer lemma based on log-concavity (Lemma 5).
  - Footnote: "AI tools assisted with the development of this proof, including ideation, derivations, and writing."
  - — [Vlassis & Thomas, arXiv:2607.08415](https://arxiv.org/html/2607.08415)
- Ming et al. (Waterloo / CMU / Berkeley) state that LMT's upper bound "has finite-sample coverage at least 1−α under iid sampling from every distribution on [0,1]". They add that Vlassis & Thomas's theorem "resolves the general coverage conjecture in Learned-Miller and Thomas (2020)". — [arXiv:2607.18661](https://arxiv.org/html/2607.18661)

**Optimality and efficiency (2026 preprints)**
- Bissias & Learned-Miller prove that Gaffke's bound is Buehler-optimal for the order it induces on samples. In words, "no other valid LCB that orders samples in the same way as Gaffke's bound can improve on it" for the (max marginal) mean. v1 posted 25 Jul 2026, v2 28 Jul 2026. — [arXiv:2607.22971](https://arxiv.org/abs/2607.22971)
- Ming et al.:
  - Gaffke's interval is first-order asymptotically efficient: √n·Width → 2σ·z_{1−α/2} almost surely, the same as the normal (Wald) interval.
  - Gaffke's p-value is *inadmissible* for every n ≥ 2. The known improvements change it only on exceptional configurations, and "we do not currently obtain a generally useful tighter bounded-mean interval from the inadmissibility result".
  - Gaffke pointwise dominates their own SymPol test under independence.
  - — [arXiv:2607.18661](https://arxiv.org/html/2607.18661)

**Dominance results**
- LMT ≤ Anderson for every sample and α ∈ [0, 0.5] (Theorem 1). This is presented as a proof *sketch*.
- LMT ≤ Hoeffding (Corollary 1), strictly when α ≤ 0.5.
- — [LMT §8](https://arxiv.org/abs/1905.06208)

**Phan, Thomas & Learned-Miller (ICML 2021, PMLR 139:8567–8576; arXiv:2106.03163)**
- Construction: choose any statistic T. The bound is the (1 − α)-quantile over U of sup_{z : T(z) ≤ T(x)} m_D(z, U). Guaranteed coverage is *proven* for every T (Theorem 2.7). — [Phan et al. 2021](https://proceedings.mlr.press/v139/phan21a/phan21a.pdf)
- With T = Anderson's bound, the result is ≤ Anderson for every sample (Theorem 4.3) and ≤ Hoeffding, strictly when n ≥ 3 (Theorem 4.4).
- With only an upper support bound (the "one-ended" setting), this version *equals* Anderson's. It gains only when a lower support bound is informative.
- With T = sample mean and support {0,1}, it recovers Clopper–Pearson.
- Each Monte Carlo draw requires a linear program; n = 50 takes "a few seconds" with 10,000 draws. Code: github.com/myphan9/small_sample_mean_bounds.
- — [Phan et al. 2021](https://proceedings.mlr.press/v139/phan21a/phan21a.pdf)

**Betting-based confidence sequences (Waudby-Smith & Ramdas, JRSS-B 86(1), 2024; DOI 10.1093/jrsssb/qkad119; arXiv:2010.09686)**
- Construction: composite nonnegative (super)martingales ("capital processes") combined with Ville's maximal inequality. The confidence set is {m : K_t(m) < 1/α}, where K_t is the "hedged capital" process that bets in both directions. The bounds adapt to the unknown variance and include a closed-form empirical-Bernstein confidence sequence. They are anytime-valid, and the paper also covers sampling without replacement. — [WSR, JRSS-B](https://academic.oup.com/jrsssb/article/86/1/1/7043257)
- Hedged capital CSs and CIs "significantly outperform past works even for small sample sizes" (details in the online supplement, not accessed). — [WSR PDF](https://academic.oup.com/jrsssb/article-pdf/doi/10.1093/jrsssb/qkad119/56961803/qkad119.pdf)
- WSR compared against Phan et al. ("PTL21") in the supplement. The PTL21 authors found "the further increase in tightness compelling". WSR did *not* compare against Gaffke, calling that comparison "interesting". — [discussion](https://academic.oup.com/jrsssb/article-abstract/86/1/51/7307586); [WSR PDF incl. reply](https://academic.oup.com/jrsssb/article-pdf/doi/10.1093/jrsssb/qkad119/56961803/qkad119.pdf)
- Ming et al. name WSR (2024), Orabona & Jun (2024), Voráček & Orabona (2025) and Austern & Mackey (2022) as modern alternatives. — [arXiv:2607.18661](https://arxiv.org/html/2607.18661)

**Reported tightness at small n**
- LMT:
  - Their bound frequently reaches mean upper bounds "that previous methods require an order of magnitude more samples to achieve".
  - It is comparable to the Student-t interval. It is tighter than t for left-skewed data (β(10,1)), and looser for right-skewed data (β(1,5), β(1,10)), where the t interval under-covers.
  - On U.S. census ages (0–84) with n < 30, the other guaranteed bounds were "too loose to provide useful high-confidence bounds", while LMT was comparable to t.
  - — [LMT §9](https://arxiv.org/abs/1905.06208)
- Phan et al. (α = 0.05, data in [0,1]; β(1,5), U(0,1), β(5,1)):
  - With T = Anderson, their bound beats Anderson, Hoeffding and Maurer–Pontil; with T = the ℓ2 norm it is "substantially tighter".
  - Student-t fails coverage for β(1,5) and U(0,1) at small n.
  - Only figures are reported, no tables.
  - — [Phan et al. 2021](https://proceedings.mlr.press/v139/phan21a/phan21a.pdf)
- Ming et al.:
  - Gaffke was the shortest interval "in every tested cell", including against the Austern–Mackey empirical Berry–Esseen interval, which has the same first-order target. They state this is "empirical and restricted to the designs considered".
  - Gaffke's empirical coverage was 0.940–1.000; the lowest value came from a cell with only 250 replications. The empirical Berry–Esseen interval covered 0.979–1.000.
  - One width table averages over 5 replications per cell.
  - — [arXiv:2607.18661](https://arxiv.org/html/2607.18661)
- In risk control, the BQ-CP rule (the same bound; see Q2) at n = 200 gave mean interval length 9.50 versus 14.29 for Hoeffding-based RCPS. Failure rates were 3.42% and 0%, against a 5% target. — [Snell & Griffiths, arXiv:2502.13228](https://arxiv.org/html/2502.13228)

### Inferences
- **Ordering.** For every sample with α ≤ 0.5: Gaffke/LMT ≤ Anderson ≤ Hoeffding. Gaffke/LMT is also ≤ *every* Phan-family bound. This follows from Phan's definition: the sample x itself satisfies T(x) ≤ T(x), so the supremum is at least m_D(x, U) for every U, and the quantile can only be larger. Phan's family is therefore a provably valid fallback that is never tighter than Gaffke.
- **Why it is tight at small n.** The posterior mean is E[L⁺] = (Σℓᵢ + B)/(n + 1) (Snell & Griffiths eq. 30). The worst-case atom therefore shifts the bound by only about (B − x̄)/(n + 1). The range is paid at rate 1/n, like the Bernstein range term but with a constant near 1 instead of 7·ln(2/δ)/3. The spread term then behaves like a t interval.
- **Worked example for a deviation gate (my calculation).** Take a 95% lower bound on a mean advantage with known worst case a = −1 (range [−1, 1]) and n identical observed values x, the most favorable case. The Gaffke lower bound is x − (x − a)·(1 − δ^{1/n}), because the extra atom's mass D₀ ~ Beta(1, n). Minimum x needed to certify a positive mean:

  | n | Gaffke/LMT | Hoeffding | Maurer–Pontil |
  |---|---|---|---|
  | 10 | 0.349 | 0.774 | 1.91 (vacuous) |
  | 29 | 0.109 | 0.455 | 0.615 |
  | 100 | 0.030 | 0.245 | 0.174 |

  For the 0/1 loss with zero failures, the same formula gives 1 − δ^{1/n}. That is Clopper–Pearson / Wilks: about 0.098 at n = 29, which is the source of the "29 points" rule.
- **Rare catastrophic outcomes (my calculation).** Suppose the advantage is −1 with probability 0.1 and +0.1 otherwise, so the true mean is −0.01. With n = 29, all draws equal +0.1 with probability 0.9²⁹ ≈ 0.047. On such a sample the t-test sees zero variance and certifies a mean of 0.1. Gaffke gives 0.1 − 1.1 × 0.098 ≈ −0.008 and does not certify. Two-point laws are exactly the worst cases Gaffke's analysis handles; this is the failure mode a deviation gate most needs to avoid.
- **Monte Carlo error.** With 1,000 Dirichlet draws (BCA's setting), the realized tail probability at the 5% level has standard error about √(0.05·0.95/1000) ≈ 0.007. Use ≥ 10,000 draws, exact Beta forms where available, or a conservative Monte Carlo correction (LMT §6.2; Phan et al. App. D).
- **Fixed n vs. sequential.** Betting confidence sequences buy validity under optional stopping and repeated testing. At a single fixed n, an exact fixed-n method like Gaffke should be at least competitive, and Ming et al.'s simulations agree. Betting is preferable if the gate will be re-checked as data arrive.

### Gaps
- I could not access Gaffke (2005). Its content is reported second-hand from Vlassis & Thomas and Michael. A preprint copy is cited in the WSR reply (math.uni-magdeburg.de …/pp1304.pdf) but I did not fetch it.
- The Vlassis & Thomas proof is a July 2026 arXiv preprint with an AI-assisted proof. I found no peer-reviewed version. I read the proof structure, which looks coherent, but did not verify every step. So far Ming et al. (July 2026) explicitly rely on it. From the abstract alone I could not tell whether Bissias & Learned-Miller's optimality result relies on Vlassis & Thomas or establishes validity independently.
- Exact small-n widths in LMT, Phan and WSR are in figures or the online supplement; I did not extract numbers.
- I could not confirm which methods (for example, hedged betting) were included in Ming et al.'s width comparison table.
- Michael's preprint is undated in the text I saw; it cites WSR 2024, so it is 2024 or later.

## Q2. Is BQ-CP with a general bounded loss the same as the Gaffke / LMT bound? Do Snell & Griffiths discuss the link, and what guarantees do they claim?

### Takeaway
Yes. For a fixed threshold, BQ-CP's posterior bound on expected loss is *exactly* the Gaffke/LMT bound, with B in place of the upper support bound 1. Snell & Griffiths do not cite Gaffke, Learned-Miller–Thomas, Phan et al. or Anderson. They claim only a prior-robust *posterior* ("data-conditional") guarantee, show that conformal risk control is the posterior *mean* of the same object, and present empirical failure rates. Frequentist validity of their general-loss rule now rests on the July 2026 Vlassis & Thomas proof.

### Cited Findings
- **BQ-CP construction.**
  - Theorem 4.3: L⁺ = Σ_{i=1}^{n+1} Uᵢ ℓ_(i) with U ~ Dir(1, …, 1) and ℓ_(n+1) = B. For any b ≤ B, inf over priors π of Pr(L ≤ b | ℓ_{1:n}) ≥ Pr(L⁺ ≤ b).
  - Corollary 4.4: b*_β = inf{b : Pr(L⁺ ≤ b | ℓ_{1:n}) ≥ β}, computed by Monte Carlo.
  - Experiments use the one-sided highest-posterior-density rule λ_hpd = inf{λ : Pr(L⁺(λ) ≤ α | ℓ) ≥ β} with 1,000 Dirichlet samples.
  - — [Snell & Griffiths, arXiv:2502.13228](https://arxiv.org/html/2502.13228)
- The Dirichlet law comes from the quantile spacings uᵢ = F(ℓ_(i)) − F(ℓ_(i−1)) ~ Dir(1, …, 1) for continuous F (their Lemma 4.2, citing Aitchison & Dunsmore 1975). A footnote claims Dirichlet sums stochastically dominate discrete spacing sums but restricts the paper to continuous distributions "due to space constraints". — [Snell & Griffiths](https://arxiv.org/html/2502.13228)
- Losses only need an upper bound: ℓᵢ ∈ (−∞, B]. — [Snell & Griffiths, App. A.1](https://arxiv.org/html/2502.13228)
- **Relation to conformal methods.** E[L⁺] = (Σℓᵢ + B)/(n + 1). The conformal risk control (CRC) rule is "simply the infimum over λ for which E(L⁺) ≤ α". Split conformal is the 0/1 loss case, giving k* = ⌈(n + 1)(1 − α)⌉. — [Snell & Griffiths §4.6](https://arxiv.org/html/2502.13228)
- **Type of guarantee.** They call it a "data-conditional guarantee", conditioned on the observed losses, in contrast to marginal (CRC) or input-conditional guarantees. Stated limitations: i.i.d. calibration and deployment, a known bound B, and conservatism because the bound "hold[s] for any choice of prior". — [Snell & Griffiths §6](https://arxiv.org/html/2502.13228)
- **Citations.** The reference list has no Gaffke, Learned-Miller–Thomas, Phan et al. or Anderson. Their distribution-free lineage is Wilks (1941), Tukey (1947/48), Guttman (1970) and Aitchison & Dunsmore (1975). — [Snell & Griffiths, references](https://arxiv.org/html/2502.13228)
- **Experiments** (failure = fraction of trials with true risk above α; target ≤ 5%):

  | Setting | CRC | RCPS (Hoeffding) | BQ-CP (β = 0.95) |
  |---|---|---|---|
  | Binomial, n = 10, α = 0.4 | 21.20% | 0.00% | 0.03% (mean risk 0.176 vs CRC 0.336) |
  | Heteroskedastic regression, n = 200 | 46.19%, length 7.99 | 0.0%, length 14.29 | 3.42%, length 9.50 |
  | MS-COCO false-negative rate, n = 1,000 | 45.05%, set size 2.92 | 0.0%, 3.57 | 5.43%, 3.04 |

  — [Snell & Griffiths §5](https://arxiv.org/html/2502.13228)
- **Third-party descriptions.** LMT's interval endpoints are quantiles of Dirichlet averages, and the order-statistic and Gaffke forms are one bound. — [Ming et al.](https://arxiv.org/html/2607.18661) WBCP restates BQ-CP's L⁺ identically and says it "controls the risk at credibility β conditionally on the observed calibration set". — [WBCP §2.2](https://arxiv.org/abs/2604.06464v3)
- Venue: ICML 2025, Outstanding Paper Award, according to WBCP's bibliography. — [WBCP references](https://arxiv.org/abs/2604.06464v3)

### Inferences
- **Exact identity.** Spacings of n sorted uniforms are Dir(1, …, 1) over n + 1 cells. So L⁺ (Snell & Griffiths eq. 27) and LMT's m(z, U) (LMT eq. 8) are the same random variable, with B in place of z_{n+1} = 1. Therefore b*_β(λ) equals the LMT bound at confidence β (that is, m_{1−β}), which also equals the inverted Gaffke test. They are identical, not merely close.
- **Frequentist reading.** By Vlassis & Thomas, P_data(R(λ) ≤ b*_β(λ)) ≥ β for each fixed λ, under i.i.d. data and losses bounded above by B; continuity and lower bounds are not needed. Snell & Griffiths did not prove this; their theorem is a posterior statement whose key step treats the quantile levels as Dirichlet *given* the observed losses.
- **Threshold selection.** Losses that are nonincreasing in λ make L⁺(λ) stochastically nonincreasing, so {λ : Pr(L⁺(λ) ≤ α) ≥ β} is an upper set. Snell & Griffiths' λ_hpd then coincides with an RCPS-style scan using the Gaffke UCB. With the standard RCPS argument (Bates et al., JACM 2021; not fetched here) this should give P(R(λ_hpd) ≤ α) ≥ β for i.i.d. data. This is my inference, not stated by any source.
- The MS-COCO failure rate of 5.43% (about 2 standard errors above 5% over 10,000 trials) is probably due to estimating each trial's risk from a finite test set (3,952 examples) and to using only 1,000 Monte Carlo draws. It is not a validity counterexample. This is an inference.
- **Applied to a deviation's advantage A ≥ a** (only the lower bound a is needed): use ℓ = −A and B = −a. The β-credible lower bound on E[A] is the (1 − β)-quantile of Σ_{i=1}^n DᵢAᵢ + D₀a with D ~ Dir(1, …, 1). That is precisely the Gaffke/LMT lower confidence bound.

**Assessment of the hypothesis** ("BQ-CP on a bounded return loss gives a credible lower bound on a deviation's expected advantage, coincides with Gaffke/LMT, and could beat Hoeffding / t-test certificates for high-confidence policy improvement"):
1. *Coincidence:* **supported.** It is an exact identity with uniform weights, i.i.d. held-out outcomes, and the extra atom at the known worst-case advantage.
2. *Validity as a frequentist certificate:* **supported, conditionally** on the July 2026 Vlassis & Thomas preprint (not yet peer-reviewed; AI-assisted proof). Before that, only the 0/1 loss case was proven (Bernoulli; Clopper–Pearson / Wilks / Vovk).
3. *Better than Hoeffding:* **supported** for every sample (α ≤ 0.5). Better than the t-test: **mixed.** The t-test is not valid. Gaffke is comparable in width; it is wider when the advantage has a heavy *lower* tail, where t lower bounds under-cover, and narrower when there is a heavy upper tail, where t lower bounds are conservative.
4. *Better deployed policies:* **unknown.** No RL study plugs a Gaffke-type bound into HCPI, Seldonian or SPI pipelines (see Q3). Indirect evidence says tighter bounds certify improvements with 2–30× less data. Any lower-confidence gate is still more conservative than the plain mean. On expected return alone it can only beat the plain mean if wrongly certified deviations are costly. Its distinctive value is a valid bound on the probability of harm.

### Gaps
- The fetched HTML did not show which arXiv version of Snell & Griffiths it was. The venue and award come only from WBCP's bibliography.
- I found no paper that explicitly says "BQ-CP equals Gaffke/LMT". The identity is my derivation, consistent with Ming et al.'s description of LMT.

## Q3. Which bounds do HCOPE, HCPI and Seldonian algorithms use? Is there evidence that tighter bounds improve how often safe improvements are returned or the deployed return? Are there importance-weighted Gaffke-type bounds?

### Takeaway
HCOPE introduced a truncated empirical-Bernstein bound for nonnegative, heavy-tailed importance-weighted returns. HCPI used that bound ("safe") alongside the Student-t and BCa bootstrap bounds ("semi-safe"). Seldonian algorithms use the t-test or Hoeffding, and the published RL example is "quasi-Seldonian" (t-test). In every comparison, tighter bounds returned improvements with much less data, and realized error rates stayed far below δ. No one has evaluated a Gaffke-type bound in these pipelines. However, Gaffke's theorem covers independent, nonnegative, non-identically distributed variables, so in principle it applies directly to ordinary importance-weighted returns.

### Cited Findings
**HCOPE (Thomas, Theocharous & Ghavamzadeh, AAAI 2015)**
- Importance-weighted (IS) returns are R(τ)·w, with returns normalized to [0, 1]. They are bounded below by zero and can be very large: a worst-case range of about 10^9.4 versus a largest *observed* value of about 316 on 100,000 Mountain Car trajectories. — [HCOPE](https://people.cs.umass.edu/~pthomas/papers/Thomas2015.pdf)
- New bound (Thm. 1): collapse the upper tail, Yᵢ = min{Xᵢ, cᵢ}, normalize, and apply Maurer–Pontil. It allows independent, non-identically distributed variables (multiple behavior policies) with the same mean. The threshold c is chosen on 1/20 of the data and the bound computed on the remaining 19/20. — [HCOPE](https://people.cs.umass.edu/~pthomas/papers/Thomas2015.pdf)
- 95% lower bounds on 100,000 trajectories (sample mean 0.191): Thm. 1 gives 0.154; Anderson (AM) 0.055; Maurer–Pontil −129,703; Hoeffding −5,831,000. Anderson "can be applied only if the random variables are i.i.d.". — [HCOPE Table 1](https://people.cs.umass.edu/~pthomas/papers/Thomas2015.pdf)
- Digital marketing simulator (behavior policy 0.0765, evaluation policy 0.086): Thm. 1 certified improvement at 95% with 2 million trajectories (lower bound 0.077). With 5 million trajectories the bound was 0.0803; with 6 million it is printed as "0.81", apparently 0.081. — [HCOPE Figs. 3–4](https://people.cs.umass.edu/~pthomas/papers/Thomas2015.pdf)

**HCPI (Thomas, Theocharous & Ghavamzadeh, ICML 2015, PMLR 37:2380–2388)**
- Three lower bounds: CI (the HCOPE Thm. 1 bound, "safe"), TT (one-sided Student-t, "semi-safe"), and BCa bootstrap with B = 2,000 ("semi-safe"). A candidate policy is chosen on 1/5 of the data and given a single safety test on the remaining 4/5. This avoids multiple comparisons; testing two policies on the same data can push the error rate to 2δ. — [HCPI](https://proceedings.mlr.press/v37/thomas15.pdf)
- Mountain Car (batch):
  - "the CI approach requires more data to ensure safe policy improvement than the TT approach, and BCa requires the least data".
  - CI with n = 50 "never resulted in a change to the policy".
  - CI "is able to produce safe policy improvement using 900 trajectories" (tested sizes were 50, 200, 900 and 5,000); unshown results with k-fold candidate selection reduce this to 400.
  - All three bounds reach the optimal policy with 5,000 trajectories.
  - — [HCPI §6, Fig. 3](https://proceedings.mlr.press/v37/thomas15.pdf)
- Incremental algorithm (Daedalus): BCa certified improvements "with as few as 50 trajectories", and TT fell between CI and BCa. Empirical error rates were zero in the gridworld and digital-marketing domains and below 0.6% in Mountain Car (5% allowed). — [HCPI §6](https://proceedings.mlr.press/v37/thomas15.pdf)
- On a heavy-upper-tailed Gamma(k = 2, θ = 50) test, CI never erred, the t-test was conservative at first and approached 5% as n grew, and BCa stayed near 5%. "The t-test tends to produce overly conservative lower bounds for distributions with heavy upper tails." — [HCPI Fig. 1](https://proceedings.mlr.press/v37/thomas15.pdf)

**Seldonian algorithms (Thomas et al., Science 366:999–1004, 2019)**
- The Safety Test "uses standard statistical tools such as Student's t test and Hoeffding's inequality". "Quasi-Seldonian" algorithms rely on "reasonable but false assumptions, such as appeals to the central limit theorem". — [Science (author PDF)](https://people.cs.umass.edu/~bsilva/papers/preventing_undesirable_behavior_ML_SCIENCE2019.pdf); [Science](https://www.science.org/doi/10.1126/science.aag3311)
- Hoeffding-based Seldonian algorithms "tend to be overly conservative" and "might require an impractical amount of data", which motivates t-test or bootstrap bounds. — [Supplement](https://people.cs.umass.edu/~bsilva/papers/preventing_undesirable_behavior_ML_supplementary_SCIENCE2019.pdf)
- In the regression example, the Hoeffding version (NDLR) needs "around m = 300,000" points before it returns a solution more often than not. The t-test version (QNDLR) needs "around m = 10,000" and "still maintains an error probability less than δ". — [Supplement §4.5](https://people.cs.umass.edu/~bsilva/papers/preventing_undesirable_behavior_ML_supplementary_SCIENCE2019.pdf)
- The RL diabetes example is labeled quasi-Seldonian. It applies a Student-t bound to importance-weighted returns and assumes their sum is normally distributed. Over 200 trials it never returned a policy that increased low-blood-sugar prevalence. It improved on the initial policy "with just 1 to 5 months of data", and with about 180 days of data it always returned a new policy. — [Supplement §5](https://people.cs.umass.edu/~bsilva/papers/preventing_undesirable_behavior_ML_supplementary_SCIENCE2019.pdf); [Science Fig. 4](https://people.cs.umass.edu/~bsilva/papers/preventing_undesirable_behavior_ML_SCIENCE2019.pdf)

**Importance-weighted / off-policy tight bounds**
- Karampatziakis, Mineiro & Ramdas (ICML 2021, PMLR 139:5301–5310) give off-policy confidence sequences for contextual bandits. The wealth process is K_t(v) = Π[1 + λ₁(wᵢ − 1) + λ₂(wᵢrᵢ − v)], which exploits E[w] = 1, with hedged bets. The bounds are non-asymptotic and valid at stopping times, and the paper applies them to "gated deployment" of a production bandit. — [ICML 2021](https://proceedings.mlr.press/v139/karampatziakis21a.html); [arXiv:2102.09540](https://arxiv.org/html/2102.09540v1)
- Vlassis & Thomas's theorem allows independent, not identically distributed, nonnegative variables with means ≤ 1. — [arXiv:2607.08415](https://arxiv.org/abs/2607.08415)
- Phan et al. name importance-weighted safe RL as a motivating use, because the IS sample mean stays skewed even well beyond n = 30, but run no RL experiments. — [Phan et al. 2021](https://proceedings.mlr.press/v139/phan21a/phan21a.pdf)
- Thomas, Learned-Miller & Phan said a tighter interval "could result in improved data-efficiency of safe and fair machine learning methods". This is an expectation, not evidence. — [JRSS-B discussion](https://academic.oup.com/jrsssb/article-abstract/86/1/51/7307586)

### Inferences
- **Gaffke applies directly to ordinary importance sampling.** With returns in [0, 1], the importance-weighted returns Xᵢ = wᵢRᵢ ≥ 0 are unbiased. Testing H₀: ρ(π_e) ≤ ρ₀ with K(X/ρ₀) gives a valid lower confidence bound on the evaluation policy's value, even with several behavior policies. That is more general than Anderson (i.i.d. only) and needs no truncation hyperparameter. It does *not* directly cover weighted (self-normalized) or per-decision importance sampling. I found no empirical study of it; whether it beats HCOPE's 0.154 in Mountain Car is untested.
- **Evidence on tightness is consistent but indirect.**
  - HCPI: earlier certified improvement with t/BCa than with CI, and equal final performance given abundant data.
  - Seldonian regression: about 30× less data when switching from Hoeffding to t.
  - SPIBB: about 2× less data with tighter bounds (Q4).
  - In each case the tighter bound was either invalid (t/BCa) or applied to a different part of the pipeline. No study measures deployed return with a *valid* Gaffke-level bound.
- **Alignment with BCA's finding.** HCPI's and Seldonian's safety tests bound the *mean* return, and candidates are chosen by predicted mean subject to passing. A 90%-quantile (0/1 miscoverage) certificate targets a tail functional, not the expected advantage. That fits BCA's observation that the plain mean beat the quantile certificate on expected return. A Gaffke lower bound targets the right functional. It would still fall back to the baseline ("No Solution Found") more often than the plain mean at small n.
- **Multiple testing.** Gating many deviation types or states, each at δ, on overlapping data inflates family-wise error (HCPI's argument). Per-gate δ needs splitting, or the guarantee should be stated per decision.

### Gaps
- I found no RL paper using Gaffke, LMT or Phan bounds inside HCOPE, HCPI, Seldonian or SPI. I did not check which bounds the Seldonian Toolkit currently offers, and I did not consult Thomas's 2015 thesis.
- No quantitative curve of "probability of returning an improvement vs. bound tightness" was extracted beyond the cited figures and text.

## Q4. How do SPIBB, Soft-SPIBB and later safe-policy-improvement work decide where the new policy may deviate from the baseline? Have outcome-based or calibrated bounds been used to allow more deviation safely?

### Takeaway
The SPIBB family decides where to deviate by *visit counts*, not by observed outcomes. SPIBB copies the baseline on state-action pairs seen fewer than N∧ times. Soft-SPIBB gives each state a deviation budget weighted by a Hoeffding-type error ∝ 1/√N. Tighter concentration bounds (Wienhöft et al. 2023) shrink N∧ and roughly halve the data needed. Valid values of N∧ remain astronomically large, though, so N∧ is tuned as a hyperparameter. I found no SPI work that gates deviations using outcome-based or conformal bounds; outcome-based gating is the HCPI/Seldonian approach, applied at the level of the whole policy.

### Cited Findings
- **SPIBB (Laroche, Trichelair & Tachet des Combes, ICML 2019, PMLR 97).**
  - The bootstrapped set B = {(x, a) : N_D(x, a) < N∧}.
  - Π_b-SPIBB copies the baseline π_b on B and optimizes elsewhere in the maximum-likelihood MDP. Π≤b-SPIBB never puts more probability than π_b on B.
  - Theorem 2 gives a ζ-approximate safe-improvement guarantee, generalizing Petrik et al. 2016.
  - — [SPIBB](https://proceedings.mlr.press/v97/laroche19a/laroche19a.pdf)
- The standard SPI analysis would require every pair to have N_D(x, a) ≥ N∧ = 8V²max/(ζ²(1 − γ)²)·log(2|X||A|2^|X|/δ). SPIBB relaxes this by allowing violations on B. "N∧ is not a property of the dataset, but a hyper-parameter". In their experiments Petrik et al.'s bound would be trivial (‖e‖∞ = 2). — [SPIBB](https://proceedings.mlr.press/v97/laroche19a/laroche19a.pdf)
- SPIBB is described as superior to existing algorithms "not only in safety but also in mean performance", with a deep-RL variant, SPIBB-DQN. — [SPIBB abstract](https://proceedings.mlr.press/v97/laroche19a.html)
- **Soft-SPIBB (Nadjahi, Laroche & Tachet des Combes, arXiv:1907.05079, 2019).**
  - A policy must satisfy Σ_a e(x, a)·|π(a|x) − π_b(a|x)| ≤ ε in every state.
  - Error functions from Hoeffding: e_P(x, a) = √((2/N_D(x, a))·log(2|X||A|2^|X|/δ)) and e_Q(x, a) = √((2/N_D(x, a))·log(2|X||A|/δ)).
  - Π_b-SPIBB is the special case e = ∞ on B and ε/2 elsewhere.
  - The safety theorem relies on an "Assumption 1" and has cubic horizon dependence, but "allow[s] us to safely search over more policies".
  - Variants: Exact-Soft-SPIBB (linear program) and Approx-Soft-SPIBB (greedy: move mass to the action with the best Q improvement per unit of error budget).
  - It "empirically outperforms SPIBB both in mean performance and in safety".
  - — [Soft-SPIBB (ar5iv)](https://ar5iv.labs.arxiv.org/html/1907.05079)
- **Wienhöft, Suilen, Simão, Dubslaff, Baier & Jansen (IJCAI 2023, pp. 4406–4415; arXiv:2305.07958).**
  - An implicit "two-successor MDP" transformation plus either the Weissman L1 bound or an inverse-incomplete-beta interval (a Beta(k₁ + 1, k₂ + 1) posterior under a uniform prior). The required N∧ grows logarithmically rather than linearly in |S|.
  - Even for |S| = 100 (δ = 0.1, ζ = 0.1), they still need "over 10 million samples per state-action pair", so N∧ stays a hyperparameter in practice.
  - The Beta variant "starts to improve over the behavior policy for data sets about half the size compared to SPIBB" on Gridworld. The ratio between N∧ values roughly carries over to dataset size on all three benchmarks. Basic RL was unreliable (CVaR below the baseline).
  - — [IJCAI 2023](https://www.ijcai.org/proceedings/2023/0490.pdf)
- Related work cited there but not read: Satija, Thomas, Pineau & Laroche, "Multi-objective SPIBB: Seldonian offline policy improvement with safety constraints in finite MDPs" (NeurIPS 2021); Scholl et al., "Safe policy improvement approaches on discrete MDPs" (ICAART 2022); Simão, Laroche & Tachet des Combes (AAMAS 2020, estimated baseline). — [Wienhöft et al. references](https://www.ijcai.org/proceedings/2023/0490.pdf)

### Inferences
- SPIBB-style gating uses *how much* data exists for a deviation, not *what happened* when the deviation was taken. BCA's gate reads held-out outcomes of rarely logged deviations, which makes it an outcome-based, local counterpart of HCPI's safety test.
- A Gaffke lower bound on a deviation's mean advantage could act as an outcome-based replacement for SPIBB's count threshold, with a valid per-test guarantee. It certifies only the mean advantage under the logged state distribution for that deviation. It does not cover compounding over trajectories or deployment-time state shift, which SPIBB's whole-policy guarantee does address in tabular MDPs.
- The SPIBB evidence (about half the data with a tighter bound) points the same way as HCPI and Seldonian: tighter valid bounds mostly buy earlier or more frequent permission to deviate.

### Gaps
- I found no SPI or offline-RL paper that uses conformal, calibrated or other outcome-based bounds to decide per-state deviations. The search was limited, so such work may exist.
- I did not verify Soft-SPIBB's publication venue (often cited as ECML-PKDD 2019) or the later critiques of Soft-SPIBB's Assumption 1 (e.g., Scholl et al. 2022).
- I did not read MO-SPIBB (Satija et al. 2021), which combines Seldonian constraints with SPIBB.

## Q5. What does Weighted Bayesian Conformal Prediction (Lou & Luo, arXiv:2604.06464v3) claim beyond split conformal and BQ-CP, and what can be verified?

### Takeaway
WBCP's main additions are covariate shift handling and a frequentist certificate:
- an exact (Bayesian-bootstrap) posterior under covariate shift, obtained by pushing the Dirichlet process posterior through the likelihood ratio (the law is a scaled Dirichlet);
- a posterior-only "data-conditional" guarantee;
- a two-sided, finite-sample, Hoeffding-rate frequentist certificate that accounts for weight-estimation error and conditional shift;
- strong shift experiments.

With uniform weights it is exactly BQ-CP, and hence the Gaffke/LMT bound. Its claim that in the i.i.d. case "credibility equals confidence" is, for general bounded losses, exactly the Gaffke conjecture, which WBCP does not cite. Its own frequentist certificate is vacuous at small n.

### Cited Findings
- Authors: Xiayin Lou (TU Munich) and Peng Luo (MIT). — [WBCP](https://arxiv.org/abs/2604.06464v3)
- **Claims (abstract):** "pushing a Dirichlet process prior on the calibration distribution through the likelihood ratio yields the exact Bayesian posterior over the deployed risk", plus "finite-sample upper and lower bounds on the risk the selected threshold actually incurs, in terms of the weight-estimation error and the conditional shift". — [WBCP](https://arxiv.org/abs/2604.06464v3)
- **Posterior (Prop. 1 / Thm. 6, with proof).**
  - Masses Ṽᵢ = wᵢEᵢ/Σⱼ wⱼEⱼ: the normalized weighted Bayesian bootstrap of Newton & Raftery (1994).
  - Density Γ(n)(Πwᵢ)⁻¹(Σⱼ ṽⱼ/wⱼ)⁻ⁿ on the simplex; this is Dirichlet only when the weights are equal.
  - Kish's n_eff = (Σw)²/Σw² is the unique leading-order two-moment Dirichlet match. Thm. 7 shows no Dirichlet matches all cumulative variances, so the algorithm samples the exact posterior.
  - — [WBCP §3.1, App. A](https://arxiv.org/abs/2604.06464v3)
- **Decision rule (eqs. 6–7, Algorithm 1).**
  - (Ṽ₁, …, Ṽ_{n+1}) ∝ (w₁E₁, …, wₙEₙ, w̄E_{n+1}), where w̄ = E_{P̃_X}[ŵ] and the last atom sits at the worst-case loss B_ℓ.
  - λ_HPD = inf{λ : Pr(L⁺_w(λ) ≤ α | ℓ) ≥ β}. The deployed threshold is λ_dep = max(λ̂, λ_HPD), so the marginal guarantee of weighted conformal prediction is inherited.
  - Taking the posterior mean recovers weighted CRC (exactly at uniform weights).
  - — [WBCP §3.1](https://arxiv.org/abs/2604.06464v3)
- **Data-conditional guarantee (Thm. 2 / Cor. 3).** inf over priors π of Pr(L(λ_dep) ≤ α | ℓ_{1:n}) ≥ β. "The argument uses no property of the mass law beyond its living on the simplex." — [WBCP §3.2](https://arxiv.org/abs/2604.06464v3)
- **Remark 1 and §1.** "Under exchangeability with continuous scores, Dir(1,…,1) is the exact frequentist law of the masses … credibility equals confidence, and the i.i.d. construction obtains valid risk control from a posterior for free." Also: "BQ-CP's Bayesian credibility and frequentist confidence coincide by construction". — [WBCP](https://arxiv.org/abs/2604.06464v3)
- **Cor. 5(ii).** In the i.i.d. case, Thm. 4 gives R ≤ α + B_ℓη_n. "We do not claim it reproduces BQ-CP's theorem, which is a posterior-credibility statement." — [WBCP App. B](https://arxiv.org/abs/2604.06464v3)
- **Deployment certificate (Thm. 4).**
  - With probability ≥ 1 − δ: α − B_ℓ(Δ_wΔ_s + Δ_shift + η_n + w̃_max) − M̂_β < R_test(λ_dep) ≤ α + B_ℓ(Δ_wΔ_s + Δ_shift + η_n).
  - Here η_n = (2ε + 1/m)/(1 − ε), ε = B·√(log((4m + 6)/δ)/(2n)), and B = ess sup ŵ / E ŵ.
  - Assumption 1 requires ŵ to be fixed, i.e. estimated on data independent of the calibration and test sets.
  - Prop. 11: a two-point construction attains the product term Δ_wΔ_s.
  - — [WBCP §3.3, App. B](https://arxiv.org/abs/2604.06464v3)
- **Experiments** (β = 0.95; 10,000 trials for synthetic data):
  - Regression under shift, n = 200, α = 0.1: failure rates BQ-CP 92.3%, RCPS 85.2%, W-CRC 42.8%, WBCP 5.0% (oracle weights 6.7%). Mean lengths 9.35, 9.67, 10.98 and 13.41.
  - Exchangeable control: failure 2.6%, with intervals 3% shorter than RCPS.
  - Multilabel count loss, α = 0.4:
    - n = 10 (n_eff ≈ 7): WBCP abstains in 93% of trials and has 0.0% failures when it certifies.
    - n = 250: BQ-CP and RCPS fail 100%, W-CRC 43.7%, WBCP 7.9% (oracle 5.6%).
    - With no shift (γ = 0), WBCP violates in 0.3–2.2% of trials.
  - PovertyMap: WBCP 1.6% vs. 18.7–29.0% for the other rules. Fashion-MNIST: 6.5% vs. 45–100%. A mis-specified discriminator raises WBCP's rate to 28.0%.
  - — [WBCP §4, App. C](https://arxiv.org/abs/2604.06464v3)
- **Stated limitations:** the weight function must be fixed and estimated on disjoint data; Δ_w and Δ_shift cannot be estimated in the field; the loss must be bounded and monotone; tied scores inflate the granularity term; calibration points must be i.i.d. (no autocorrelation). — [WBCP §5](https://arxiv.org/abs/2604.06464v3)
- The bibliography does not cite Gaffke, LMT or Phan. — [WBCP references](https://arxiv.org/abs/2604.06464v3)

### Inferences
- Thm. 6 is an algebraic property of the Bayesian-bootstrap limit (Dirichlet process with τ → 0). It is "exact" relative to that improper limiting prior and says nothing about frequentist coverage. Thm. 2 holds for any mass law on the simplex by WBCP's own statement, so the "data-conditional guarantee" is a property of the posterior model, not evidence of calibration.
- **Remark 1.** For the 0/1 loss with thresholds at order statistics, "credibility = confidence" is classical (Clopper–Pearson / Wilks). For general bounded losses it is exactly the Gaffke/LMT conjecture: unproven when WBCP v3 was written (v3 cites June 2026 work but not Vlassis & Thomas) and now supported by the July 2026 preprint. It is not "by construction". WBCP is also internally hedged: Remark 1 asserts the equivalence, while Cor. 5(ii) declines to claim it.
- **WBCP's own frequentist certificate is loose at RL-scale n** (my calculation, uniform weights, δ = 0.05, m = ⌈√n⌉): η_n ≈ 1.24 at n = 29 (vacuous), 0.58 at n = 100, 0.39 at n = 200, and 0.17 at n = 1,000. By contrast, the Gaffke / Clopper–Pearson route certifies risk ≤ 0.098 with 29 zero-failure points. For a frequentist statement in the uniform-weight case, Vlassis & Thomas plus Gaffke/LMT is far stronger than WBCP's Thm. 4.
- **Weighted case.** No Gaffke-type finite-sample validity theorem exists for WBCP's weighted-mass posterior; only the Hoeffding-rate certificate does. If BCA needs importance weighting with a tight valid mean bound, Vlassis & Thomas suggest an alternative: put the weights on the *outcomes*, Xᵢ = wᵢ·(Aᵢ − a) ≥ 0, keep uniform Dirichlet masses, and require independent samples with known weights. That is valid but higher-variance.
- For BCA with uniform weights: BCA equals BQ-CP. With the 0/1 loss it reproduces the Wilks / Vovk (2012) bound. Switching to a bounded advantage loss gives the Gaffke/LMT lower bound on expected advantage, valid for i.i.d. held-out outcomes.

### Gaps
- I did not find WBCP v3's posting date or peer-review status. I did not reproduce its experiments or check every constant in Thm. 4; the proofs of Lemmas 8–10 are standard and looked correct on reading.
- I did not determine whether a peer-reviewed version of WBCP, or a version citing Vlassis & Thomas, exists.

# Aim BCA at a deviation's average gain

Yes, but only through one channel, and its most promising form is still untested. In offline reinforcement learning, a policy (a rule for choosing actions) learns from a fixed log of past decisions. BCA (Bayesian Conformal Aggregation, your method) helps such a policy only as a **gate**: a yes-or-no check, on held-out outcomes (logged results kept out of training), of whether the policy may leave a safe, well-logged **fallback** action. As a gate, it already beat count-based caution at the same safety level in your toy. It also kept a per-data-set safety promise that split conformal prediction (the standard held-out cutoff method) cannot make. As an uncertainty width fed into training, it measures noise rather than ignorance, and it failed. The best next step keeps BCA's random-weight sampler but averages its draws, instead of reading a cutoff that 90% of outcomes must clear. With equal weights, that average gives **Gaffke's bound on a mean**, a classic small-sample bound. It is provably tighter than Hoeffding's range-based bound and stays valid where the t-test breaks on rare disasters. But its validity was proven only in an unreviewed July 2026 preprint. It should earn the most return of any rule that validly promises a deviation will not lower average return. It can beat the plain mean on raw return only when many thinly logged deviations compete, not on a single decision. So test it next on a many-candidate version of your channel toy, compared at equal harm rates, with kill rules fixed before the first run.

## Widths fail; gates that read a deviation's own outcomes work

Production BCA turns held-out errors of the critic (the learned value estimate) into a width. The width then strengthens the pull toward logged actions, or reshapes a penalty or weight ([README](../README.md)). It comes from conformal prediction, which turns held-out errors into an interval that covers a new outcome at a stated rate.

Theory says this is the wrong signal. Pessimism, the standard cure for over-optimistic critics, lowers values where data are thin. Its guarantees need a bound on the error of the *average* predicted value at every state and action, including actions the log never took ([Jin, Yang & Wang](https://ar5iv.labs.arxiv.org/html/2012.15085)). A conformal width covers single noisy outcomes, and only at logged actions. So it charges actions for being noisy, not for being unknown. Worse, without assumptions, no method can bound an average at a single point more tightly as data grow ([Barber](https://arxiv.org/abs/2004.09477)).

Your channel toy confirmed this. Its ground-truth scenarios ("worlds") run from harmless to harmful deviations. Both width rules lost to a simple range limit that falls back wherever the critic has under 20 training rows: **pooled regret 0.425 and 0.376, against 0.122** ([channel toy](../bca_channel_toy.ipynb)). Regret is the best possible average return minus the policy's; pooled means averaged over all worlds and data sizes. The widths still covered 90.2% and 93.5% of outcomes where measured, against a 90% target. They just could not respond at the deviation, and they punished good but noisy actions. Your earlier discrete toy found the same: its gains came from the support mask (refusing actions the log never took), not the width.

The literature offers no rescue for widths. The one reported gain, from a method called UNIQ, uses a conformal quantile (a cutoff below which a set share of values falls) only to scale the disagreement among several critics. It reports 85.2 against 81.6 on a standard robot benchmark. But it rests on three random restarts and a remove-one-part test that cannot be squared with its main table ([Upadhyay](https://arxiv.org/html/2606.07592)). The penalties that do work on benchmarks come from critic disagreement itself, which is not a conformal quantity ([Ghasemipour, Gu & Nachum](https://proceedings.neurips.cc/paper_files/paper/2022/file/7423902b5534e2b267438c85444a54b1-Paper-Conference.pdf)).

Gates have better support. Using uncertainty to decide where to leave the behavior policy (the policy that produced the log) beat pessimism when both used the same imperfect uncertainty ([Brandfonbrener et al.](https://ar5iv.labs.arxiv.org/html/2206.01085)). A 2026 method calibrates, on a safe policy's data, how far a new policy may stray, with guarantees that hold at any sample size ([Prinster et al.](https://arxiv.org/abs/2603.02196)). In one robot-navigation scenario, calibrating the decision itself, rather than a prediction set, reached the goal **about 29% faster** while staying safe ([Lekeufack et al.](https://ar5iv.labs.arxiv.org/html/2310.05921)).

Your toy agrees. Every rule that read the deviation's own held-out outcomes (the toy's "survivors") closed **38% to 87%** of the gap between the best data-blind rule and an oracle that knows the true world. Shuffling those outcomes erased the gain. Comparing outcomes with the fallback's value beat comparing them with the rest of the data plus a margin, at regret 0.028 against 0.051. This works because the critic's value at the deviation cancels out. The gate needs only the fallback's value, which the critic knows well because the fallback is in the log.

BCA's tail gate, as built, is a textbook test. With equal weights, the threshold BCA is 95% sure of equals the exact binomial (Clopper–Pearson) bound, also called Wilks' tolerance limit. Your toy found 99.6% agreement, and the misses were sampling noise. That is why it certifies nothing below 29 outcomes: 0.9²⁹ is the first power of 0.9 below 0.05. Its edge over split conformal is the kind of promise. Split conformal promises 90% coverage only on average across calibration sets (the held-out data that set a threshold). For one set of 100 points, real coverage falls below 90% about **45% of the time** (my arithmetic from the coverage law in [Tabbara, Yang & Sibai](https://arxiv.org/html/2511.07899v2)). The same arithmetic says split conformal approves a world with 10.6% bad outcomes, just past your 10% line, 28% to 38% of the time. Your toy saw 32–38% in its worst world, against at most 3% for BCA. So that gap is the expected cost of an average-only promise, not a hidden flaw. BCA's tail gate also beat the range limit, at regret 0.076 against 0.122, with unsafe deviations at 0.2% against 0%. Its price was 0.060 more regret than the plain mean.

## Averaging BCA's draws gives the tightest valid bound on a mean

At its core, BCA's sampler gives the held-out scores random shares that sum to one (uniform Dirichlet weights). It adds one more share for a worst-case value that stands in for the unseen point. Production reads where each draw crosses 90%. Instead, put the extra share on a declared worst outcome, take each draw's weighted average, and read the 5% quantile across draws. The result is a lower bound on the deviation's average gain. Snell and Griffiths define exactly this random average for bounded losses ([Snell & Griffiths](https://arxiv.org/html/2502.13228)). Ming and colleagues describe the Learned-Miller–Thomas bound as quantiles of the same averages, and show it equals Gaffke's 2005 test ([Ming et al.](https://arxiv.org/html/2607.18661)). No paper states the identity in one line; it follows from matching the definitions.

It is the tightest guaranteed small-sample bound on a mean in the sources reviewed. On every sample, at confidence above 50%, it is never looser than Hoeffding's bound, which uses only the range of possible values. Nor is it looser than Anderson's bound, which is built from the sorted data, though that step is shown only as a proof sketch ([Learned-Miller & Thomas](https://arxiv.org/abs/1905.06208)). In 2026 simulations it gave the shortest interval in every design tried, with coverage of 0.940 to 1.000 ([Ming et al.](https://arxiv.org/html/2607.18661)). No valid bound that orders samples the same way can beat it ([Bissias & Learned-Miller](https://arxiv.org/abs/2607.22971)). One modern rival family, betting-based bounds, has not been compared with it directly ([Waudby-Smith & Ramdas](https://academic.oup.com/jrsssb/article/86/1/1/7043257)). In a risk-control test with 200 points, the same construction gave intervals of **9.50 against 14.29** for a Hoeffding-based rule, with 3.42% failures against a 5% target ([Snell & Griffiths](https://arxiv.org/html/2502.13228)).

The table shows the evidence it saves. Take a deviation whose gain over the fallback lies between −1 and +1, and suppose every held-out outcome shows the same gain. Each entry is the smallest such gain that a bound certifies as positive at 95%. Empirical Bernstein is a guaranteed bound that also uses the observed spread. The numbers are my arithmetic from the formulas in [Learned-Miller & Thomas](https://arxiv.org/abs/1905.06208) and [Phan et al.](https://proceedings.mlr.press/v139/phan21a/phan21a.pdf).

| Outcomes | Gaffke | Hoeffding | Empirical Bernstein |
|---|---|---|---|
| 10 | 0.35 | 0.77 | never |
| 29 | 0.11 | 0.45 | 0.61 |
| 100 | 0.03 | 0.24 | 0.17 |

It also stays honest where the t-test does not. The t-test assumes the sample average is roughly bell-shaped. Suppose a deviation gains 0.4 on 95% of tries and loses 10 on the rest. Its true average is −0.12, so it is harmful. With 29 held-out outcomes, **23% of samples contain no loss at all**. On those samples the t-test sees no spread and approves, so its real error rate is about 23%, not 5%. Gaffke's bound knows a loss of 10 is possible. It gives −0.62 and refuses (my arithmetic). Published simulations show the t-test missing its coverage target on skewed data at small samples ([Phan et al.](https://proceedings.mlr.press/v139/phan21a/phan21a.pdf)).

Every valid promise about an average pays for the worst case, and the declared floor sets the price. At 95%, beyond a spread term much like the t-test's, the extra share costs roughly 3/n of the distance between the observed average and the floor, for n outcomes. So with no losses seen, the gate needs about 3 × (1 + worst loss ÷ typical gain) clean outcomes before it approves. In the example above, the exact count is 77. The tail gate avoids this cost because it bounds a frequency, which always lies between 0 and 1. The two gates therefore make different promises, and neither dominates. The mean gate accepts deviations with frequent small losses, which the tail gate must refuse. It refuses deviations that might hide a large unseen loss until clean evidence piles up. And it cannot stand in for the tail gate: in your toy's rare-catastrophe worlds, the plain mean let through 29.0% unsafe deviations, against 0.2% for the tail gate.

Its validity is new. For about 20 years it was a conjecture. A 2024 journal discussion still called it "an open question" after failed hunts for counterexamples ([Thomas, Learned-Miller & Phan](https://academic.oup.com/jrsssb/article-abstract/86/1/51/7307586)). A July 2026 preprint claims a full proof for independent, nonnegative variables that need not share a distribution. Its authors say artificial-intelligence tools helped, and it is not yet peer reviewed ([Vlassis & Thomas](https://arxiv.org/abs/2607.08415)). If the proof fails, Phan and colleagues' family of bounds is provably valid and never tighter than Gaffke's, so it is the safe fallback ([Phan et al.](https://proceedings.mlr.press/v139/phan21a/phan21a.pdf)).

WBCP (Weighted Bayesian Conformal Prediction, the method BCA implements) does not settle the question. It claims that its Bayesian 95% equals a long-run 95% error rate "by construction." For averages, that claim is Gaffke's conjecture restated. Its own guarantee is useless at small samples. With equal weights, its error allowance is about 1.24 at 29 points and 0.58 at 100 (my arithmetic from its Theorem 4) ([Lou & Luo](https://arxiv.org/abs/2604.06464v3)).

Two practical rules follow. Use at least 10,000 Monte Carlo draws (the random draws that compute the bound), as Learned-Miller and Thomas recommend for a 5% level ([Learned-Miller & Thomas](https://arxiv.org/abs/1905.06208)). With 1,000 draws, the estimated 5% tail has a typical error of about 0.007, which explains your toy's boundary slips. And keep the sampler. This bound has no simple, stable closed form, so here BCA's sampler is the real computing engine.

## The plain mean wins on return until many deviations compete

Your toy covered three of the four ways to build a gate. The empty cell, a strict gate on the average, is where the sources point. Entries are pooled regret. The plain mean, which makes no promise, scored 0.016 ([channel toy](../bca_channel_toy.ipynb)).

| Gate style | Tail promise | Average promise |
|---|---|---|
| Lenient: benefit of the doubt | BCA harm test, 0.028 | t-test, 0.063 |
| Strict: burden of proof | BCA tail gate, 0.076 | Untested |

Every strict bound is more cautious than the plain mean. For one deviation against a well-estimated fallback, deviating whenever the held-out average beats the fallback is the right rule for average return, and your toy found it best at every data size. So on a single decision, a strict mean gate buys a guarantee and costs some return. The guarantee is real, though. In your toy, the plain mean approved deviations that lower average return in **up to 46% of seeds** (random repetitions). That was the rare-catastrophe world with a true gain of −0.10 and 30 outcomes. Each such mistake was cheap there, which is why the plain mean still won.

Many candidates change the arithmetic. Picking the best of several estimates that are each right on average still overstates the winner, which is called the **optimizer's curse**. With equal true values and noise of one unit, the winner looks 0.56 units too good among 2 options and 1.54 among 10 ([Smith & Winkler](https://jimsmith.host.dartmouth.edu/wp-content/uploads/2022/04/The_Optimizers_Curse.pdf)). Offline learners hit this constantly. A rarely tried option can look best by luck. So picking the empirical best keeps a constant error even with near-expert data, while a lower-bound rule's error vanishes ([Rashidinejad et al.](https://arxiv.org/abs/2103.12021)). In a study of 256 policies per task, the learner with the weakest pull toward logged actions "deceptively looks to produce the best policy, while actually being the worst" ([Paine et al.](https://arxiv.org/abs/2007.09055)).

Pessimism is still not a free win. No ranking rule is best on every problem ([Xiao et al.](https://arxiv.org/abs/2104.02293)). A lower bound helps when the well-logged option is good and the tempting ones are thinly logged, which is your fallback setting. It hurts when the good options are the thin ones. In one direct test, ranking by lower bounds "almost always" did worse than ranking by means when picking top policies ([Yang et al.](https://arxiv.org/abs/2012.06919)). With equal evidence per candidate, a lower bound ranks exactly like the mean ([Gu & Koenker](https://arxiv.org/abs/2012.12550)).

The principled rival is shrinkage: pulling each estimate toward the average of similar candidates. Shrinkage moves low estimates up as well as high ones down ([Smith & Winkler](https://jimsmith.host.dartmouth.edu/wp-content/uploads/2022/04/The_Optimizers_Curse.pdf)). BCA's posterior mean (the average over its draws) only moves down. With equal weights, it is (sum of scores + worst case) ÷ (n + 1) ([Snell & Griffiths](https://arxiv.org/html/2502.13228)). Any claim that BCA raises return must beat shrinkage, not just the plain mean.

Many candidates also break per-decision promises. A 5% error per deviation does not cap the share of approved deviations that are harmful. In drug screening, about 3% of molecules were active. Approving molecules one at a time with conformal sets gave about **80% false approvals** at 90% confidence ([Jin & Candès](https://arxiv.org/abs/2210.01408)). Their Benjamini–Hochberg version, a standard procedure that caps the expected share of false approvals, gave near-exact control. Splitting the error budget equally across tests (Bonferroni's correction) "nearly has no power." Reusing held-out data inflates errors too: testing two policies on one held-out set can double the error rate ([Thomas et al., policy improvement](https://proceedings.mlr.press/v37/thomas15.pdf)).

Here is where a strict mean gate should beat each rival you named.

| Rival | Mean gate's edge | Rival's edge |
|---|---|---|
| Plain mean | A guarantee; many thin candidates | More return on one decision |
| t-test | Valid under rare disasters | Tighter only where it fails |
| Hoeffding | Tighter on every sample | None at the same promise |
| Split conformal | Targets the average; promise per data set | Approves sooner; promise only on average |
| Count rule | Reads outcomes, not counts | Needs no held-out outcomes |

## Other routes cost return or remain blocked

| Route | Evidence | Verdict |
|---|---|---|
| Strict mean gate | Strong math, no offline test | Test next |
| Tail gate, as built | Works in your toy | Keep for tail promises |
| Capping the false share | Strong outside this field | Add with many candidates |
| Bound on reweighted policy returns | Untested | Stochastic policies only |
| Calibrating a caution knob | Online tests only | Blocked offline |
| Conformal scale for critic disagreement | One weak paper | Low priority |
| Width penalty | Fails in theory and toy | Retire |

A policy-level bound is the strongest untested idea after the mean gate. Gaffke's theorem allows independent, nonnegative outcomes that need not share a distribution. That fits importance-weighted returns, where each logged return is reweighted to mimic a new policy. The bound could drop into high-confidence policy improvement, which ships a new policy only if a held-out lower bound beats the old one. On 100,000 runs of the Mountain Car control task averaging 0.191, the 95% lower bounds were 0.154 for a Bernstein-type bound that caps extreme values, 0.055 for Anderson's and −5,831,000 for Hoeffding's ([Thomas et al., evaluation](https://people.cs.umass.edu/~pthomas/papers/Thomas2015.pdf)). Whether Gaffke's beats 0.154 is untested.

Tighter bounds have reliably bought earlier approvals. One Seldonian algorithm (a method that ships a solution only if a held-out safety test passes) switched from Hoeffding to a t-test. That cut the data it needed from about 300,000 points to about 10,000 ([Thomas et al., supplement](https://people.cs.umass.edu/~bsilva/papers/preventing_undesirable_behavior_ML_supplementary_SCIENCE2019.pdf)). The catch for your hosts (your four base offline algorithms): importance weights are undefined for deterministic policies with continuous actions.

Calibrating a caution knob is principled but blocked offline. Learn then Test turns "pick a knob value" into a set of statistical tests, and returns only values whose risk is certified ([Angelopoulos et al.](https://arxiv.org/abs/2110.01052)). Conformal risk control guarantees risk only on average over data sets, and its rule is exactly BCA's posterior mean ([Snell & Griffiths](https://arxiv.org/html/2502.13228)). The bottleneck is that each knob value must be scored on data that cover its decisions. The one offline use tuned the behavior-cloning weight of TD3+BC, a method that pulls the policy toward logged actions, but scored it with online test episodes. Adaptive testing raised the share of truly good settings it certified from 0.32 to 0.85 ([Zecchin, Park & Simeone](https://arxiv.org/abs/2409.15844)).

Weighting is the only part of WBCP that is not a textbook test, and it is weakest where a policy strays furthest. In weighted conformal prediction, the unseen point carries its own weight. When that weight is large, the threshold becomes infinite and the method refuses ([Tibshirani et al.](https://ar5iv.labs.arxiv.org/html/1904.06019)). Estimated weights also cost coverage, by half the average weight error, and need overlap between the two policies ([Taufiq et al.](https://proceedings.neurips.cc/paper_files/paper/2022/file/cc84bfabe6389d8883fc2071c848f62a-Paper-Conference.pdf)). Under real shift, though, weighting matters. In WBCP's shifted regression test, the unweighted version failed in 92.3% of trials, against 5.0% for WBCP ([Lou & Luo](https://arxiv.org/abs/2604.06464v3)). For a gate, test weighting only where survivors come from a different mix of states than deployment will visit. If a mean gate needs weights, put them on the outcomes and keep equal shares. The July 2026 theorem's conditions appear to cover that case when the weights are known.

## Test the strict mean gate on many candidates at equal harm

Start cheap. Add a strict mean gate to the existing channel toy to fill the empty cell, with a strict t-test and a strict Hoeffding gate beside it. The toy's Gaussian noise has no floor. Clip it, for example at three standard deviations, and declare the floor before running. This step checks validity and cost on one decision, where the plain mean should still win on return.

Then build the real test: a many-candidate toy. Give the policy many tempting deviations, each with its own held-out outcomes. Make the evidence uneven, because equal evidence makes every bound rank like the mean. Make every candidate look attractive to the critic, since that is how offline critics fail. Then vary how many are truly good, from a few to half.

| Factor | Settings |
|---|---|
| Candidates | 1, 3, 10, 30 |
| Outcomes per candidate | Uneven, 5 to 100 |
| Outcome shapes | Shortfall, rare big loss, rare big gain |
| Truly good candidates | 10%, 30%, 50% |
| Draws per bound | At least 10,000 |

The rule is a few lines on top of BCA's sampler. For each candidate, draw equal random shares over its outcomes, plus one share at the floor. Take each draw's weighted average. Approve only if the 5% quantile of those averages beats the fallback's value. Choose candidates on training rows and certify them on held-out rows, never on the same outcomes. Repeat the split several times, because a single split can pick the wrong option by itself ([Nie et al.](https://arxiv.org/abs/2210.08642)).

Run every rival on the same draws. That means the plain mean, the plain mean plus a margin, a strict t-test, Hoeffding, empirical Bernstein, split conformal, BCA's tail gate, a count rule, shrinkage toward similar candidates, and BCA's posterior mean. Add one arm that turns each candidate's bound into a p-value (how surprising its outcomes would be if it were harmful), then applies Benjamini–Hochberg across candidates. Score return as exact regret, as in the toy. Score harm three ways. First, the share of seeds that approve a deviation whose true average gain is negative, in each world. Second, the share of approved deviations that are harmful. Third, the chance that the gated policy does worse than always falling back.

Compare the rules twice. First, at their stated 5% level: does each keep its promise, and what return does it earn? Then at equal harm: tune each rule's knob until all reach the same worst-world harm rate, and compare return. The second comparison is the honest test of "higher return at a fixed safety level." With one candidate and fixed evidence, Hoeffding is just the plain mean minus a constant, so their tuned curves coincide. Only uneven evidence and mixed shapes can separate the rules.

Hash the kill rules before the first run, as you did for the channel toy. Three follow from the sources. Shelve the mean gate if its harm rate tops 5%, plus Monte Carlo error, in any world. Shelve its tightness claim if, at the same stated promise, it does not approve clearly more good deviations than Hoeffding. Shelve BCA's return claim if, in many-candidate worlds, it never beats both the plain mean and shrinkage on pooled regret. The sources predict a win only when harmful look-alikes outnumber good deviations, and harm arrives as rare large losses.

If it survives, promote it in two steps. First, run it on the small discrete benchmarks used for count-based safe improvement, replacing the count threshold with the mean gate ([Laroche et al.](https://proceedings.mlr.press/v97/laroche19a/laroche19a.pdf)). We found no published work that gates deviations on outcomes there. Second, move to ThrusterChain and one host, where the gate must find deviations and pool similar ones. Your step-1 gate there had only 28–59 outcomes per seed and world, right at the tail gate's floor of 29 ([channel toy](../bca_channel_toy.ipynb)). Finally, check the whole deployed policy. A gate changes which states the policy visits. In one safety-filter study, a more cautious switch caused more violations, 22%, because the fallback landed in unfamiliar states ([Tabbara, Yang & Sibai](https://arxiv.org/html/2511.07899v2)).

## Conclusion

The useful question is not whether calibrated uncertainty helps, but which quantity it bounds and how many decisions share the evidence. A width on residuals bounds noise at logged actions, so it cannot steer the policy where it errs. A tail bound buys a per-decision safety promise at a known cost in return. An average bound buys the cheapest valid promise that return will not fall. It can raise return only when thin evidence is spread across many tempting options. With equal weights, BCA's real asset is not its Bayesian reading. It is a fast engine for the tightest valid small-sample bounds we found, and an honest description names them: Wilks for tails, Gaffke for averages.

Two dependencies deserve a flag. The mean gate's guarantee rests on an unreviewed July 2026 proof, though Phan's family offers a valid fallback. And none of the sources we found compares calibrated with fixed thresholds at equal violation rates in offline learning. So the many-candidate test pays off either way. A win gives BCA its first defensible return claim. A loss settles that its value is the guarantee alone.

## Update after the toy test (written after it ran)

The many-candidate test is now [bca_mean_gate_toy.ipynb](../bca_mean_gate_toy.ipynb), with kill rules fingerprinted before the first run. The answer falls between the two outcomes above: BCA earns the most return at a fixed safety level, but not the most raw return. Four of six claims survived:

- **The promise held.** At a stated 5%, BCA's mean gate accepted at most 0.3% of harmful proposals in any of 36 worlds.
- **It is far tighter than the classic bounds.** It accepted 17% of good proposals. Hoeffding and empirical Bernstein accepted almost none.
- **The t-test is unsafe with rare disasters.** It accepted up to 31% of harmful proposals.
- **At equal harm, it earned the most.** Every gate was tuned to at most 5% harm in every world. The mean gate then earned 0.084 per situation: about twice BCA's own posterior mean (0.040), and about 4.5 times the best gate not built from BCA (shrinkage, 0.019). It also led at every harm cap from 1% to 20%.

Two claims were shelved. The first contradicts this report's prediction:

- **No return win in crowded worlds.** There, the mean gate earned 0.057 per situation, against 0.183 for the plain held-out mean and 0.163 for shrinkage. Choosing on training rows and certifying on fresh held-out rows already removes the critic's luck; after that, caution mostly costs return. The claim above that a bound "can raise return only when thin evidence is spread across many tempting options" did not hold here, even in that setting.
- **Benjamini–Hochberg earned less** than deciding each situation alone (0.027 against 0.045), because few proposals pass at all.

One lead, not one of the claims: BCA's posterior mean, (sum of outcomes + floor) ÷ (outcomes + 1), kept 86% of the plain mean's return and never lost return in any world. In the hardest worlds, it earned the most of any real gate; the plain mean lost return there.

The toy departs from the plan above in two ways. It used one training/held-out split per seed rather than several. It scored harm as the per-world share of harmful proposals accepted, plus the harmful share of acceptances for the Benjamini–Hochberg arm. It did not score the chance that the gated policy does worse than always falling back.

## Update after the weighting test (written after it ran)

The weighting question above is now [bca_weights_toy.ipynb](../bca_weights_toy.ipynb). There, the held-out outcomes came from a different mix of conditions than deployment. Its kill rules were fingerprinted before the first run, and all six claims survived:

- **Under shift, weighting is needed.** Ignoring the shift, BCA's tail gate deviated in up to 90% of seeds in an unsafe world. With weights, as built in `calibration/wbcp.py`, it stayed at or below 4.3% in every unsafe world. Weighted split conformal reached 39%.
- **Weights beat the simple valid alternatives.** In safe shifted worlds, weighted BCA said yes in 34% of seeds. Rejection sampling said yes in 19%, and per-condition exact bounds never did.
- **At equal harm, the weighted mean gate earned the most, narrowly:** 0.0975 per decision, against 0.0923 for a tuned weighted split conformal.
- **Weights cost evidence.** Under the strong shift, the tail gate needed about 83 outcomes before it could say yes at all, against 29 with no shift.

The advice above held up: put a mean gate's weights on the outcomes. That version is the same gate, except the worst-case point gets the largest weight rather than the average one. It earned less at the stated 5% level (0.080 against 0.098), but nearly as much at equal harm (0.0954 against 0.0975). This comparison was added after the run, as information only. The toy's weights are exact, so estimated weights are the next test.

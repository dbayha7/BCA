# Knee/stall study: figures and reproducible saved-data analysis

This is the completed `knee_s10_candidate_v2/confirmation_v1` study, attempt 3, not Path A. The figures present all three penalty strengths and all three training seeds. They preserve the original result: overall **INCONCLUSIVE**, C1 **NOT CONFIRMED**, C2 **OVER-CORRECTION**. Two Q-pi/rollout identity checks narrowly exceeded the frozen relative tolerance of 1e-6; no result is relabeled. The returns are descriptive observations from completed rollouts. The outcome does not erase the useful exploratory strength profile.

## What to show in a meeting

Start with **01_paired_return_change.png**. It directly answers whether BCA improved policy return over paired native TD3+BC. Then show **03_exposure_return_tradeoff.png** and **05_candidate_ranking.png** to distinguish avoidance from beneficial choice. Use **04_raw_and_adjusted_q_errors.png** to explain why a lower score is not automatically a more accurate critic. **02_policy_return_context.png** supplies the absolute return scale. Each PNG is 300 dpi; matching PDFs are vector exports. `Knee_Study_Figures.pdf` combines all five pages.

## Captions and interpretation limits

1. **Paired return change.** Mean discounted return difference between each BCA policy and the native policy with the same seed, world, warmup and minibatch stream. Points identify training seeds; the dark line is their mean. At kappa 189 all three stall seed means improve (+11.245, +121.479, +9.760), while all three no-stall means worsen (-41.421, -98.321, -20.525). Seed 1 supplies approximately 85% of the summed stall improvement, so the mean is not a typical seed effect. Kappa 63 was the prespecified confirmation dose; 21 and 189 are the prespecified exploratory dose profile. Three training seeds are the independent training replications. The 128 common starts per policy are nested evaluations, not 128 independent training runs. No confidence intervals or significance claims are inferred from those episode counts.

2. **Absolute return context.** Each point is one policy's mean across 128 starts. Higher (less negative) is better. Both worlds use the same vertical scale. Return is the discounted 100-step sum with gamma 0.98, not normalized benchmark score. The large stall improvement of seed 1 at kappa 189 is genuine descriptive policy evidence; its generality is limited by three seeds and the fixed development-informed benchmark. Kappa 189 improved on 81/128, 128/128 and 71/128 stall starts, respectively, but on 0/128 no-stall starts in every seed. These episode-level facts are available in `policy_metrics.csv` and exact per-start differences can be reconstructed from `data/returns.csv`.

3. **Exposure versus return.** Horizontal position is the percentage-point change in the fraction of executed steps with first-thruster action beyond the withheld knee, relative to paired native. Vertical position is paired return change. Every BCA policy reduced this exposure; the value of avoiding it depends on the world. Lines connect doses within a seed for readability, not as a fitted relation or causal claim. This plot does not establish action-only distribution shift, withholding-specific harm, or avoidance as the cause of return changes. World-specific dynamics and the remaining action components still matter.

4. **Raw Q and adjusted score errors.** Top: mean signed error. Positive denotes overestimation and negative underestimation. Bottom: mean absolute error. Open marks/dashed lines use raw Q1 minus own-policy Q-pi; filled marks/solid lines use Q1 minus kappa U minus Q-pi. Points are three seeds and lines their means. Each policy has 48 own-action probes (16 at each time 0, 25 and 50), drawn from its own visited states and followed by its own unclipped continuation. Across policies, both those states and continuations can differ. Therefore a reduction in raw error does not isolate improved critic accuracy on matched inputs. The adjusted score is the actor's training score, not a revised critic prediction; its large negative offset can coexist with better action preferences. The Bayesian residual calibration is not a full-return true-Q or conditional OOD coverage guarantee. No new near-zero tolerance is introduced.

5. **Candidate-bank ranking.** Top: mean regret of choosing the bank's raw-Q maximizer versus its penalized-score maximizer; regret is best bank Q-pi minus chosen bank Q-pi. Each bank has four complete actions: the logged command and its three replacements of a0 by same-sign magnitudes 0.85, 0.9 and 1.0, with other action components preserved. Bottom: fractions of the 192 banks per seed where penalization improves or worsens chosen true return. Bars average three seeds, points show each seed, and labels give counts across 576 bank evaluations. Unchanged choices account for the remaining banks. The same 192 states/actions are reused across all models within a world, so 576 is not an independent-state sample size. Each trained model has its own continuation Q-pi. No ties occurred. This is a retrospective saved-data score diagnostic: the deployed actor does not select these four-action argmaxes, and these switches are not proven causes of policy-return differences. The top panels have different, explicitly labeled vertical scales. At kappa 189, mean regret decreases in every stall seed (overall 1.68946 to 1.17188) and increases in every no-stall seed (0.06051 to 0.27764).

## Reproduce without access to the repository or a GPU

The archive includes every numeric array used to reproduce the figures, as CSV. It does not contain a learner, simulator or model checkpoints. With Python, NumPy and matplotlib already available:

```text
python -B reproduce_figures.py
```

This validates included CSV hashes against `SOURCE_MANIFEST.json`, recomputes metrics, and rewrites the five PNGs, five vector PDFs, combined PDF and metric tables locally. It performs no training, calibration, model inference or simulator call. No new package installation is performed by this artifact.

For an authorized source re-extraction in the original checkout only:

```text
python -B reproduce_figures.py --source-repo C:/Users/David/dev/BCA
```

The extractor requires final result SHA256 `b136b39c670e7224c22fe1b8f38062a22d30f608d22c9b4253c406194b53d197`. `SOURCE_MANIFEST.json` records all 24 source report paths, SHA256 hashes and sizes, the final receipt and decision/evaluator source hashes. `VALIDATION.json` records count/identity checks, output hashes and plotting-library versions. `policy_metrics.csv/json` includes per-policy returns, paired differences and start wins, exposure, signed/absolute Q errors, sign counts and candidate regret/switch counts. Source extract counts: 24 policies, 3,072 returns, 1,152 own-action probes, 18,432 candidate values.

## Scope and method

Native and BCA share a 10k native warmup within each seed/world, then each receives 10k further critic updates. BCA changes the actor objective through Q1 minus kappa U with a live action gradient; BC strength is unchanged and the critic retains the native learning rule. Kappa is penalty strength, not head count. No simple penalty or ordinary-quantile ablation in this package isolates a uniquely Bayesian benefit. No missing-connector claim is tested here. Full rollout state trajectories and return learning curves are not available in these final reports.

The applicable repository skills actually read during figure preparation were `.agents/skills/offline-rl-experiment-design/SKILL.md` and `.agents/skills/bca-code-and-calibration-audit/SKILL.md`. Their pairing, independent-unit, own-policy reference and raw-score distinction checks are reflected in the figures and captions. All new calculations were pure arithmetic on existing saved arrays.

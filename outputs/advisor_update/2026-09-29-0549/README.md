# BCA advisor update

**Status snapshot: September 29, 2026, 1:49 a.m. EDT / 05:49 UTC.**

[Open the updated PowerPoint](BCA_Advisor_Update_2026-09-29_0549.pptx) · [Speaker notes](SPEAKER_NOTES.md)

## How the experiment is going

- **Cluster training:** 68/140 successful physical closures, split 34 ReBRAC and 34 IQL. This adds four since the previous deck. The original controller and worker identities matched. The active third-seed ReBRAC Maze host was at 425,000 updates. Its final exit and the controller's final exit remain pending.
- **Local training:** two successful TD3+BC Hopper closures. CQL remains stopped after its final evaluation because the saved result field names differ from the validator's expectations. Original exit 1 remains preserved.
- **Independent acceptance:** 14/280 physical runs, forming seven pairs. ReBRAC Hopper and Walker each have 3/5 accepted paired seeds. TD3+BC Hopper has 1/5. Fifty-six cluster closures await full audits. Successful execution alone does not close those audits.
- **OOD evidence:** still one completed TD3+BC Hopper training seed, with 5,120 simulator rollouts. No prospective v2 outcomes yet. There is no complete five-seed OOD comparison.
- **Current blocker:** the real v2 worker, checkpoint and simulator execution path is not accepted. The latest isolated runtime component passed 41 synthetic fixtures, but the integrated production dispatcher remains unfinished. These are engineering tests, not new OOD outcomes. No supported completion ETA is available.

## Advisor question: why do this study?

Average return curves do not tell us whether BCA recognizes harmful unfamiliar actions or helps the policy behave better after them. This study adds controlled action interventions at the same saved simulator state.

It separates two questions: do larger BCA bands rank harmful actions, and does a BCA-trained continuation earn more return after the same imposed action? This helps identify which part needs investigation instead of changing coverage or weights based only on a weak score.

## Advisor question: did the result meet expectations?

| Expectation | Evidence so far | Meaning |
| --- | --- | --- |
| Wider bands warn about harmful actions | First Hopper seed: within-state AUROC 0.529, versus 0.500 chance and 0.566 for support distance | Weak separation in this panel |
| BCA handles unfamiliar actions better | Post-hoc distant-action continuation advantage -5.046 raw return, degradation advantage -0.019 against a common anchor | No demonstrated benefit in this one-seed comparison |
| Correctly weighted thresholds address population shift under their assumptions | Known-ratio toy coverage 90.70% at nominal 90%, versus ordinary shifted coverage 61.05% | The controlled arithmetic test meets its limited expectation |
| The new action panels clarify the limitation | No prospective v2 outcomes yet | This expectation remains untested |

The OOD result does not establish why BCA underperformed. Bootstrapped target errors may fail to track behavioral harm, the host may use widths ineffectively, or the calibration and tested populations may differ. These are hypotheses. Weak harm ranking does not itself refute a conformal coverage theorem. Correct toy coverage does not establish real RL coverage or a policy benefit.

## Advisor question: why does each change make sense?

1. Near, moderate and distant action bands test how the result varies with dataset support.
2. A common recorded-action anchor makes degradation comparisons use the same reference action.
3. Separate familiar-state and joint-shift panels distinguish action novelty from simultaneous state/action novelty.
4. Five declared training seeds test whether the finding repeats across independently trained policies.
5. The approved Walker candidate-seed correction removes overlap with an existing evaluation stream without tuning outcomes.
6. The intended IW correction puts justified target/source ratios into the held-out threshold and query mass. The earlier fitting-IW route weighted the scale objective but did not implement this complete threshold calculation.

The new experiment can yield useful evidence even if BCA does poorly. Favorable warning quality without policy improvement would direct attention to how the host consumes width. Weak warning quality would direct attention to the score and its relationship to harm. A reliable policy improvement would require the declared repetitions and uncertainty analysis. IW remains a hypothesis-driven later comparison, not a guaranteed cure for the baseline result.

## Next milestones

Finish and review the actual v2 execution path, pass the real checkpoint/simulator checks, then collect the precommitted action panels. Handle the local CQL schema recovery separately using saved artifacts without repeating completed training. Continue auditing cluster results. Keep coverage unchanged. Validate the real sampling and ratio contract before claiming a corrected IW implementation for the hosts.

## Slide guide

Slides 1–6 explain the research objective, experiment, host attachment points and fresh status. Slides 7–11 show accepted training and OOD results. Slides 12–18 justify each change, explain favorable/unfavorable outcomes, describe the IW correction, and compare expectations with observations. Slides 19–20 explain calibration and weighted-threshold arithmetic.

## Evidence

The new live snapshots provide the execution counts. `data.json`, the seven-pair CSV, all chart series, and the five embedded chart workbooks are unchanged from the previous accepted deck. All 23 underlying result-source hashes still match. The earlier two-seed cluster endpoint table remains labeled preliminary where full audits are pending. It is separate from the new execution count.

This publication adds no training, model query, simulator step, queue change or scientific acceptance. Original results, failures, prior slides and native inventories remain unchanged.

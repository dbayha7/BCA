---
name: offline-rl-evidence-and-visualization
description: Analyze and visualize saved offline RL results with traceable empirical Q errors, decision outcomes, policy returns and uncertainty.
---

# Evidence and visualization for offline RL

Start from source arrays and the frozen protocol, not a remembered summary. Respect existing test-data boundaries and preserve raw files. Record file, checkpoint, config, seed, sample counts, units, filters and the formula used for each figure; verify array indexing and joins, including whether a field uses zero-based indexing.

1. State the question and estimand for each plot. Distinguish reference Q, raw predicted Q and penalty-adjusted Q on the identical state/action samples. Define signed error as prediction minus reference: positive is overestimation, negative is underestimation. Negative Q values require a clearly labeled axis: -30 is higher than -60.
2. Show correction and possible harm: positive error removed, remaining positive error, overshoot, introduced or worsened underestimation, and absolute error. Aggregate mixed signs can obscure individual outcomes. A mean score reduction is not an overestimation detector.
3. For decisions, show unchanged, improved and harmed choices, action-bank regret and the appropriate raw and simple controls. State the bank size and candidate scope; do not extrapolate to global optimum or policy-return improvement. Pairwise Q margins versus penalty contrasts can reveal why a choice changes or remains unchanged.
4. Show nominal/shifted conditions and support labels separately. State whether an action was forced or freely selected and whether local support is actually known. Use an own-policy reference for each checkpoint; values from different continuation policies have different estimands.
5. Use independent training seeds or state/episode clusters for uncertainty, preserve pairing, and report dependent action counts honestly. No invented error bars; show values and sample sizes if no defensible interval exists. Do not call a post-hoc threshold a calibrated detector.
6. Make one principal question per figure with large labels and readable units, especially on mobile. Provide a plain-language conclusion with negative or inconclusive findings and numerical effect sizes. Inspect rendered output before delivery. Visual selection must not hide counterexamples.

**Deliverable:** readable figure(s), concise empirical interpretation, data provenance and reproducible calculations, with precise limitations. A visualization task does not authorize additional simulation, fitting, confirmation-data reuse or publication.

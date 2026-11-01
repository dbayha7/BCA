---
name: offline-rl-experiment-design
description: Design or review a bounded offline reinforcement-learning experiment, including matched baselines, data splits, uncertainty, selection and independent evaluation.
---

# Offline RL experiment design

Turn the user's actual research question into an executable, falsifiable comparison. Check the repository's instructions and source before assuming what a method or baseline implements.

1. Write the claim, primary estimand, comparison and practical effect of interest. Separate an implementation check, a frozen-action mechanism diagnostic, and a trained-policy return comparison; none automatically proves the others.
2. Identify dataset provenance and splits by independent episode or collection unit. Freeze fitting, calibration, development/selection and untouched evaluation roles, seeds, candidate methods and any decision rule before examining the corresponding outcomes. Record any previously inspected data as development. Never tune on final evaluation labels.
3. Pin the baseline source revision, configs, normalization, optimizer, randomization, batch reuse, target update and actual optimizer-update counters. Match environment, data, budget, architecture and evaluation as appropriate, and state unavoidable differences. Do not rename a different implementation as the requested baseline.
4. Specify what is paired and the independent unit: training seeds for variation across training runs, with episode/state families nested within runs. Shared states across methods are paired, not independent replications. Report per-seed effects and cluster intervals at the relevant independent unit; small sample uncertainty remains substantial.
5. Define discount, horizon, terminal versus truncation, bootstrap mask, action scaling, and exact evaluation reference. A candidate action followed by its own frozen actor estimates a policy-specific Q, not Q-star; common-policy values and smoothed TD targets are different estimands.
6. Prespecify metrics, controls, sample/compute limits, stopping and inconclusive criteria. Include simple and strength-matched controls when the scientific question requires them; compare actual choices and policy returns, not just lower scores. State whether support or distribution shift was measured, not assumed.
7. Inspect saved receipts to avoid duplicate runs; report work already spent and remaining. Execute only within the user's authorization and bounded budget. Do not retry a denied access route, create credentials, publish, delete, or start expensive jobs merely because this skill is present.

**Deliverable:** protocol with the hypothesis, source/data identity, arms and budget, independent units, frozen selection rule, primary/secondary metrics, uncertainty and failure/inconclusive conditions. Report missing prerequisites and evidence rather than inventing them.


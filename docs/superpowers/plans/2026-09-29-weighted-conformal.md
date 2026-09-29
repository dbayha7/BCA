# Weighted conformal correction implementation plan

> Execute in this session without subagents, following David's standing instruction. His confirmation that weighted conformal prediction is the intended IW method authorizes this correction and its CPU mathematical checks. Preserve existing experiments. This plan does not release a new RL training run.

**Goal:** Implement and verify a separate weighted split-conformal reference, and specify the remaining requirements for a scientifically justified IW-BCA integration.

**Architecture:** Add an isolated package under `experiments/weighted_conformal/`. Its explicit calibration and query density ratios determine the held-out threshold. Retain a separately identified weighted Bayesian-bootstrap component and take the maximum. Do not modify the four existing host paths, training configs, frozen manifests, or accepted results. This is an additive development package in the existing clean checkout, not a replacement of a running worker's sources.

**Tech stack:** Python, NumPy, exact rational arithmetic for the conformal mass boundary, standard-library unittest, SciPy for binomial uncertainty, CPU only.

## Task 1: Pin the starting point and mathematical contract

- [x] Save hashes of all pre-existing tracked repository files and the two canonical native-readiness files, plus the current git revision.
- [x] Document `X=(s,a)`, the frozen host-specific Bellman target `Y`, the normalized residual score, and source versus target populations.
- [x] Cite Tibshirani et al. (2019), equations (5)-(7), and the September 22 audit; distinguish the supplied Bayesian paper from the finite-sample weighted conformal floor.
- [x] State the deterministic-action support issue, adaptive calibration feedback issue, unknown behavior density, and trajectory dependence. Do not convert metadata assertions into coverage guarantees.

## Task 2: Test and implement the isolated reference

**Create:** `experiments/weighted_conformal/reference.py` and `test_reference.py`.

- [x] First write tests for the finite-sample query atom, equal weights, unequal weights, an infinite radius, ties, zero calibration weights, common rescaling, invalid data, support violations, and interval arithmetic. Run and retain the expected missing-module failure.
- [x] Implement ratios from explicit source/target densities; reject missing support and numerical overflow instead of clipping or silently normalizing different populations separately.
- [x] Implement the conformal threshold from calibration masses plus query mass at infinity. Use exact rational comparisons for the represented float weights and decimal alpha.
- [x] Implement the tilted observed-support Bayesian bootstrap with explicitly supplied draws; keep its Monte Carlo credibility distinct from a frequentist risk certificate.
- [x] Return the query-specific conformal radius, Bayesian radius, full maximum, query mass, and effective sample size. Preserve infinity; require explicit finite positive scales for interval construction.
- [x] Check the independent finite-distribution coverage law by exhaustive enumeration, along with all arithmetic cases.

## Task 3: Freeze and execute a synthetic validation

**Create:** `experiments/weighted_conformal/protocol.yaml`, `validate.py`, and `README.md`.

- [x] Comment every YAML field. Pin the seed, sample size, trial count, alpha, bootstrap draws, and known source/target laws before execution.
- [x] Run the same frozen score function under a known covariate shift, comparing ordinary conformal, weighted conformal, and weighted BCA's maximum. Use independent trials and report uncertainty plus infinite-width rates.
- [x] Write a new results directory with actual test/validation exits, source/protocol hashes and synthetic-only labels. Refuse to overwrite an existing result directory.
- [x] Reconcile outputs and verify all pre-existing tracked files and protected inventories are unchanged.

## Task 4: Deliver the corrected design and next integration gates

**Create:** `docs/WEIGHTED_CONFORMAL_DESIGN.md`.

- [x] Explain the code-to-math gap, correction, arithmetic example, and data flow in plain language.
- [x] Separate established known-ratio mathematics from estimated-ratio experiments and adaptive training heuristics.
- [x] Define a future frozen-checkpoint validation and the host-specific integration boundaries; list unfinished ratio, sampling, and training-feedback decisions explicitly.
- [ ] Publish only owned new code, design, and synthetic evidence after checks. Exclude weights, supplied PDFs and all existing scientific outputs.

## Completion boundary

This stage is complete when the additive reference and its tests are verified and the design is reviewable. It does not establish real-policy coverage, fix all host integrations, select a best IW rule, or authorize relabeling prior fitting-IW results. Full training integration remains pending the declared target population, defensible ratios/support, independent calibration design, query-specific width consumption, checkpoint contracts, and a separately frozen run matrix.

## Execution notes

The first synthetic attempt and two reader failures are archived. The corrected decimal-ratio calculation uses exactly 0.25/4; all generated scalar inputs and random streams are unchanged. Fifteen unit checks and 2,000 independent trial reconstructions pass. The broader repository checker has pre-existing source-hash mismatches, documented without changing frozen manifests. Real-policy ratios and host training integration remain unfinished as stated above.

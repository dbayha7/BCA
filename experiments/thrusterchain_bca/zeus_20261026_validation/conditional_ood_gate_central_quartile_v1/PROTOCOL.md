# Central-quartile support design: one development gate

This separately versioned construction preserves the failed negative-band gate and all previous reservations. The user's 28 October approval authorizes one replacement design and a bounded validity screen, not training. `contract.json` and `DECISION_FROZEN.json` are the decision authority. No new outcome was evaluated before their hash was saved.

## Why this single design

The pinned recorder draws actions independently of state and time. Its general two-region mixture is uniform over the four-dimensional action cube only at epsilon = 0.25; runtime must verify that dataset metadata. The first thruster acts on the first float; command zero is neutral thrust. Remove the central recorder quartile, **abs(a0) <= 0.25 when observed current + first wave >= 0.4**. Retain the same band at lower load and both adjacent equal-probability bands [-0.75,-0.25) and (0.25,0.75]. All are below the 0.8 stall knee in magnitude.

The width is a recorder-mass rule, not the unfinished zero-slab memo's fitted-value rule. The shape is nevertheless baseline-informed development: prior actor ranges, memo value fits and the failed gate are already known. This is not independent benchmark discovery. The 0.4 split denotes an upper load side, not extreme current or a change in hidden dynamics. No alternative region or width is tested in this gate.

For the ideal requested-action law, high-side retention is 0.75 and low-side retention is 1. Thus population state support is unchanged, but state density changes. Symmetric removal preserves the population conditional action mean, not its exact finite-sample mean. Executed float32 rounding slightly changes boundary masses; the explicit executed-command mask governs every recorded count. Typical overall deletion is about one eighth if the load sides balance, not a promised measured count.

Known recorder factorization and the mask define the conditional action-support hole. Finite counts in nearby states are only marginal a0 occupancy; they do not prove dense 13D state coverage or matched four-action trials at an identical state. If that stronger empirical claim is required, the minimum next collection design is repeated state anchors with independent randomized one-step branches and explicit episode grouping. That collection is not part of this gate.

## Data, controls and leakage boundaries

Decode the pinned 768 training, original 128 calibration and last 64 of the remaining 128 tuning-validation episodes: 960 episodes, 96,000 tuples. All have previously been consumed in development. These 64 probe episodes differ from the previous gate's first 64, but are not untouched data. Probe times cycle 0, 25, 50, 75. Free trajectories use the first 16 of these probe start seeds; these are nested samples, not an independent confirmation set. Protected selection/final-calibration transition arrays are never decoded. Provenance metadata and whole-file hash bytes are read.

Full replay, hole replay and once-sampled random removal are defined prospectively. Random removal exactly matches hole retained counts within source episode, ten-step bin and load side. It may retain hole labels. Hole versus random is the structured-label-removal contrast; full versus hole also changes sample count and state density. Continuous-state imbalance can remain despite exact coarse matching. Report physical-coordinate mean differences and conditional mean actions; do not match rewards or successors, which change systematically with removed commands.

Later learners must use condition-retained tuples for replay, BC and uncertainty fitting. The common model normalization must be fitted only to the intersection of hole/random retained training rows, then shared across conditions. This diagnostic preserves the old broad-trained checkpoint and its existing normalization; it is not a removal-trained model. The audit-only distance standardization uses original training observations, without labels, calibration or probe fitting.

Preserve original one-row-per-episode calibration choices when eligible; replace an in-hole row uniformly within the same original calibration episode using the frozen replacement seed. All prospective conditions share the same hole-excluding plan. An episode with no eligible row fails. This gate performs no calibration updates and makes no hole-conditional or selected-action coverage claim. A small count of removed calibration rows is exposure accounting, not evidence of conditional validity.

This is tuple-label withholding, not recollection under a restricted behavior policy. Retained states and successors can reflect earlier removed actions in the original trajectory. No history features or sequence learner may expose removed action/reward tuples. Actor and bootstrap-target queries inside the hole are unlabeled model queries, not observed-label leakage; later training must log their rates. Independent later confirmation requires new collision-audited seeds and a separately frozen design.

## Frozen state-overlap check

Use standardized RMS distance over 13 physical coordinates: sqrt(mean(((x-x_ref)/std_train)^2)). Fit population mean/std on original training observations; replace std below 1e-12 with 1. Compare only the same ten-step time bin and load side, excluding the whole reference/query episode.

In each of 20 time/load strata, choose exactly 32 reference rows from distinct training episodes by SHA256("central-quartile-reference-v1:" + source_row_id), then row ID; keep the first row per episode. The 95th `higher` quantile of their leave-episode-out 64th-neighbor distances is the fixed-rule typicalness threshold. No anchor, action or outcome selects it. Insufficient references or neighbors makes the stratum unavailable and fails the gate. All 20 strata are required.

A probe passes if its 64th distance is at most this threshold, its fixed original 64-neighbor set has at least 32 rows/eight episodes before and after hole removal, original band evidence has at least two rows/two episodes, and each retained flank has at least two rows/two episodes. No refill occurs after removal. At least 80% of anchors on each load side must pass, with at least 16 anchors per side. This is relative training-state typicalness, not a relaxed claim that the old absolute-radius criterion passed. The 32 references, 95th quantile and 80% hurdle are declared engineering conventions, not a coverage or precision guarantee.

## Value, exposure and decision

Reuse the authenticated evaluator: restore exact float64 state by recorded-prefix replay, preserving time; check every float32 observation. Compare direct dynamics with the environment at t = 0, 25, 50, 99, including reward, horizon, termination and discount. No private time mutation.

At each of 64 anchors evaluate six actions: actor; a0=-0.5; the float32 point immediately inside -0.25; zero; immediately inside +0.25; +0.5. Hold the other three commands at actor values. Continue the same frozen deterministic actor to t=100 with gamma=0.98 and no terminal bootstrap. These are Q-pi references, not optimal values or smoothed TD targets.

All inherited global coverage gates remain: at least 128 hole rows/32 episodes; eight hole rows per time bin; 32 per load stratum [0.4,0.5), [0.5,0.6), [0.6,infinity); 128 low-load band rows; zero forbidden retained/calibration labels; exact random counts; maximum hole-random mean-state difference <=0.1 original SD.

At least 25% of high-load anchors must have max(abs(Qpi(center)-Qpi(left)), abs(Qpi(center)-Qpi(right))) >=1 return unit. Separately, free actor trajectories must enter the hole on at least 1% of high-load steps, in at least two episodes, with at least 100 high-load steps observed. Forced candidate inclusion does not satisfy free-policy exposure. Report within-band spread and all low-load results without using them to rescue high-load failure.

Report every raw-Q signed error: over/under/near-zero using tau = 1e-4 + 1e-5*abs(Qpi), counts and magnitudes, MAE, raw-Q choice regret and harmful switches. These are descriptions of one broad-trained native critic. There is no BCA comparison or missing-label causal estimate in this gate.

**Pass requires every integrity and scientific criterion. Any scientific failure shelves this exact design; no threshold, width, radius or seed changes, automatic retry or training launch.** All fixed scientific components finish once even on a scientific failure; an integrity/runtime error stops immediately. A pass permits a separate training proposal only.

## Budget and later efficacy boundary

One reservation; 900 seconds total; 28,104 simulator transitions = 2,400 prefix + 24,000 continuation + 1,600 free + 104 parity; 81 resets; 384 candidate returns; at most 30,000 actor rows, 386 critic rows and two loader width rows. CPU support arithmetic is capped at 640 references, 64 query anchors and 54,067,200 distance pairs. Zero actor, critic, auxiliary or calibration updates. Save attempted/completed counts and uncertainty on interruption. No automatic resume or retry.

Later efficacy, if separately approved, must extend the existing support pilot's complete 10k fork, use at least three independent initialization seeds, and compare unchanged native UniFloral, original direct-Q BCA kappa=3 and the existing training-gradient-strength-matched simple control across full/hole/random data. Primary interpretation is the hole-minus-random interaction in paired native/BCA own-policy error; full data and retained low-load cells are controls, not substitutes. Report fixed learning curves, separate same-critic raw/adjusted-score errors from separately trained critic errors, over/under/near-zero transitions, decision regret, free returns and unsupported query rates. A global calibration multiplier alone cannot establish action-specific Bayesian benefit; that needs matched quantile/raw-scale effective-strength controls in a separately scoped test. No such training or additional method ablation is authorized here.

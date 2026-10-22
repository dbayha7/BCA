# Fresh-bank mean alignment is negative; ranking scarcely changes

On this fresh bank, the mean width/error relationship runs opposite to the intended relative-error penalty direction, while the change in action-ranking concordance is close to zero. No incremental benefit from adaptive width is demonstrated on this bank. This is not an impossibility result or a test of all importance-weighted BCA variants; no broad training follows from this report.

The width changes 5 commands: 4 improve, 1 is harmed and 27 are unchanged versus raw Q. Mean selected-Q change is 0.044886, interval [-0.367432, 0.421841]. The direct width-minus-distance selected-Q difference is -0.283109, interval [-0.652308, -0.009318].

The shuffled-width control has mean selected-Q change 0.370231 versus raw, despite mean concordance change -0.004298. Average pair ordering and the value of the single maximum-score choice are different endpoints. One fixed permutation is a descriptive placement control, not a randomization test.

Fresh-bank mean width/signed-error correlation: -0.111290; width-minus-distance correlation: -0.125563 [-0.242948, -0.002111]; width-minus-shuffled correlation: -0.132611 [-0.280195, 0.014854].

Completed 384/384 candidate continuations, 21696/21696 planned plant transitions. Valid states: 32/32; invalid states: 0. New training updates and new validation transitions: 0.

This separate fresh bank was specified after the earlier 32-state bank was inspected, then frozen before its own return labels. It replicates a frozen-model diagnostic, not independent training seeds or a confirmatory learner comparison. Earlier results remain unchanged and are not pooled.

## What is being checked

BCA means the existing Bayesian calibration procedure. Its native use changes behavior-cloning strength. This diagnostic instead ranks fixed commands by Q1 minus the exported uncertainty width. The earlier mechanical tests verified the proposed live-width gradient path; the present run performs no gradient step.

The primary error is signed e=Q1-Qpi, so a larger penalty should track larger relative overestimation within a state. This differs from the earlier positive-error Spearman summary. Pearson correlation of all indexed candidates equals the normalized dot product of all i<j differences, without recentering those differences. The equivalent calculation is an identity check, not a second independent result.

## Source and target laws

The historical band uses 128 pooled calibration rows: one uniformly selected transition per reserved behavior episode. Under full retention, time is uniform over 0..99 and physical states follow behavior trajectories. The 625-command product categorical behavior law gives each coordinate probability 0.84 at its mode and 0.04 at each other atom. These rows were adaptively reused; the frozen reference was last refreshed at update 95000. Its target is a noisy one-step residual for the current twin-min critic.

The new bank has 32 fresh independent physical-state draws, eight per stall/monotone and low/high-load context. Times 0, 50, 99 have counts 3, 3, 2 in each context. Ten position/velocity coordinates are uniform on [-0.5, 0.5]; load is 0.3 or 1.1 and wave amplitude 0.2 has uniform phase. These reset-law states crossed with assigned times are designed queries, not states visited by the behavior or learned policy. Each state has six distinct grid commands: include the nearest-grid actor command and behavior mode when distinct, then fill by uniform grid IDs without replacement. Append the original continuous actor command and five Uniform[-1,1]^4 commands. All 12 indices remain, including cross-slot duplicates.

A true-Q label executes the candidate once, then follows the unchanged unsmoothed original actor to horizon 100 with discount 1 and no terminal bootstrap. It is the deterministic numerical return of this one-action deviation, not Q-star or a new policy that repeatedly selects corrected-score actions.

The historical target, evaluator inputs and long-return labels differ. A finite width is not a long-horizon Q-coverage certificate. The nominal 90% one-step calibration setting is frozen metadata, not a coverage result of this replication.

## Signed-error alignment

| Endpoint | Mean | 95% interval | Defined / planned states | Defined states per bootstrap draw | Undefined draws / 10000 |
| --- | --- | --- | --- | --- | --- |
| Width versus signed error | -0.111290 | [-0.231189, 0.002985] | 32/32 | 32..32 | 0 |
| Distance versus signed error | 0.014273 | [-0.099091, 0.131228] | 32/32 | 32..32 | 0 |
| Shuffled width versus signed error | 0.021321 | [-0.065533, 0.107426] | 32/32 | 32..32 | 0 |
| Width minus distance correlation | -0.125563 | [-0.242948, -0.002111] | 32/32 | 32..32 | 0 |
| Width minus shuffled correlation | -0.132611 | [-0.280195, 0.014854] | 32/32 | 32..32 | 0 |

Each standalone mean uses its own defined states. Paired differences use the pair-specific common intersection; paired arm means and all masks/denominators remain in paired_results.json.

## Ranking concordance

Concordance is the fraction of comparable true action pairs ordered correctly. True differences at most 1e-6 are excluded; predicted differences at most 1e-6 earn half credit. The aggregate is an equal-state mean, not a pooled-pair estimate.

| Endpoint | Mean | 95% interval | Defined / planned states | Defined states per bootstrap draw | Undefined draws / 10000 |
| --- | --- | --- | --- | --- | --- |
| Raw Q1 concordance | 0.715814 | [0.678278, 0.753309] | 32/32 | 32..32 | 0 |
| Raw Q1 concordance change | 0.000000 | [0.000000, 0.000000] | 32/32 | 32..32 | 0 |
| Q1 minus width concordance | 0.716339 | [0.676493, 0.755567] | 32/32 | 32..32 | 0 |
| Q1 minus width concordance change | 0.000524 | [-0.005653, 0.006214] | 32/32 | 32..32 | 0 |
| Q1 minus matched distance concordance | 0.725852 | [0.686050, 0.765683] | 32/32 | 32..32 | 0 |
| Q1 minus matched distance concordance change | 0.010038 | [0.005303, 0.014307] | 32/32 | 32..32 | 0 |
| Q1 minus shuffled width concordance | 0.711517 | [0.673258, 0.750134] | 32/32 | 32..32 | 0 |
| Q1 minus shuffled width concordance change | -0.004298 | [-0.008129, -0.000473] | 32/32 | 32..32 | 0 |
| Q1 minus state-constant offset concordance | 0.715814 | [0.678278, 0.753309] | 32/32 | 32..32 | 0 |
| Q1 minus state-constant offset concordance change | 0.000000 | [0.000000, 0.000000] | 32/32 | 32..32 | 0 |

## Selected action consequences

| Endpoint | Mean | 95% interval | Defined / planned states | Defined states per bootstrap draw | Undefined draws / 10000 |
| --- | --- | --- | --- | --- | --- |
| Raw Q1 selected true Q | -107.269059 | [-112.134291, -102.161364] | 32/32 | 32..32 | 0 |
| Raw Q1 change versus raw | 0.000000 | [0.000000, 0.000000] | 32/32 | 32..32 | 0 |
| Raw Q1 advantage versus actor | -0.318841 | [-1.500248, 0.663064] | 32/32 | 32..32 | 0 |
| Q1 minus width selected true Q | -107.224174 | [-112.118974, -102.061378] | 32/32 | 32..32 | 0 |
| Q1 minus width change versus raw | 0.044886 | [-0.367432, 0.421841] | 32/32 | 32..32 | 0 |
| Q1 minus width advantage versus actor | -0.273956 | [-1.412998, 0.658034] | 32/32 | 32..32 | 0 |
| Q1 minus matched distance selected true Q | -106.941065 | [-111.814667, -101.831016] | 32/32 | 32..32 | 0 |
| Q1 minus matched distance change versus raw | 0.327995 | [0.036081, 0.637949] | 32/32 | 32..32 | 0 |
| Q1 minus matched distance advantage versus actor | 0.009154 | [-1.152665, 0.962543] | 32/32 | 32..32 | 0 |
| Q1 minus shuffled width selected true Q | -106.898828 | [-111.773597, -101.871811] | 32/32 | 32..32 | 0 |
| Q1 minus shuffled width change versus raw | 0.370231 | [0.006064, 0.808070] | 32/32 | 32..32 | 0 |
| Q1 minus shuffled width advantage versus actor | 0.051390 | [-1.311780, 1.203034] | 32/32 | 32..32 | 0 |
| Q1 minus state-constant offset selected true Q | -107.269059 | [-112.134291, -102.161364] | 32/32 | 32..32 | 0 |
| Q1 minus state-constant offset change versus raw | 0.000000 | [0.000000, 0.000000] | 32/32 | 32..32 | 0 |
| Q1 minus state-constant offset advantage versus actor | -0.318841 | [-1.500248, 0.663064] | 32/32 | 32..32 | 0 |
| Original actor true Q | -106.950218 | [-111.209363, -102.343174] | 32/32 | 32..32 | 0 |
| Width minus distance: selected true Q | -0.283109 | [-0.652308, -0.009318] | 32/32 | 32..32 | 0 |
| Width minus shuffled: selected true Q | -0.325345 | [-0.820429, 0.113812] | 32/32 | 32..32 | 0 |

Each chosen command is executed once and then control returns to the original actor. Changes of candidate index and actual command are counted separately. Selecting the original actor after raw Q chose another command is restoration, not evidence that a newly trained policy improved.

| Score | Changed indices | Changed commands | Selected actor command | Actor restorations | Improved / harmed / unchanged versus raw |
| --- | --- | --- | --- | --- | --- |
| Raw Q1 | 0 | 0 | 8 | 0 | 0 / 0 / 32 |
| Q1 minus width | 5 | 5 | 8 | 0 | 4 / 1 / 27 |
| Q1 minus matched distance | 4 | 4 | 8 | 0 | 4 / 0 / 28 |
| Q1 minus shuffled width | 6 | 6 | 8 | 1 | 4 / 2 / 26 |
| Q1 minus state-constant offset | 0 | 0 | 8 | 0 | 0 / 0 / 32 |

## Controls, support and refusal

Width coefficient remains 1; inherited distance coefficient remains 0.519506430786272. Neither is retuned. One fixed within-state permutation preserves the 12 indexed widths while changing their placement; it is a control, not a permutation test. The state-constant offset must preserve raw-Q ordering.

Conditional-action support counts: {"common_on_grid": 47, "rare_on_grid": 145, "unsupported": 192}. Native width refusals: 0/384. Evaluated failure events: 0/384; nonfinite evaluated returns: 0/384.

Common grid actions have beta>=0.01; rare grid actions have 0<beta<0.01; unsupported means beta=0 under exact float32 grid equality. These labels do not certify state/calibration overlap. A finite but unusable native width falls back to raw scoring; refusal does not reject an action. Invalid numeric input or return invalidates the whole state across arms without replacement. Physical threshold events remain measured outcomes.

Invalid-state reasons: {}.

## What the intervals mean

One common saved set of 10,000 bootstrap draws resamples whole states within the original context/time cells, retaining the 3/3/2 counts. Actions and action pairs are not independent replications. Each mean correlation uses its own defined-state mask; each correlation contrast uses its pair-specific intersection. The same eligibility mask is applied after drawing from all 32 original states; missing states are not replaced or resampled away. Every draw denominator is retained. Any draw with no defined contributions makes that endpoint interval unavailable. These percentile intervals are descriptive and conditional on this one checkpoint, designed state law and candidate realization; there are only 2-3 states per context/time cell. They do not quantify variation across independently trained models.

No hypothesis test, policy-benefit conclusion, Q coverage test, checkpoint selection or extra variants were introduced.

Rebuild this report and V5 without model calls: `python report_assets/build_report.py`. Reproduce the numerical analysis from saved arrays with the frozen `reproduce_analysis.py` command documented in README.md. The report checks source/input/result hashes before writing. V5 preserves V4 apart from this new section and its navigation links. Static HTML/source-link checks passed; rendered browser, navigation and runtime JavaScript QA was not performed for V5.

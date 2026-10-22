# Independent review of the completed V2 diagnostic

Reviewed 2026-10-21. **PASS for the stated bounded numerical diagnostic:** no material arithmetic, selection, or provenance defect remains in the completed V2 artifacts. This is not evidence that the width provides calibrated Q uncertainty or improves a learned policy. Review used source inspection and independent arithmetic on saved arrays and records, with a second reviewer checking exports. Neither reviewer ran model inference, policy rollouts, or training. This report is the only review-authored file.

## Verification

- All 384 candidate records, 2,725 paired decision records, 48 subgroup summaries, 128 correlation records, and 48 error summaries reconcile with the saved NPZ arrays. No candidate action duplicates occurred in this realized bank.
- Independently recomputed all 240 group-by-method summaries and reproduced their 10,000-draw bootstrap intervals exactly, including unavailable intervals. Ties, eligibility, harm, candidate-set opportunity loss, and actor-relative advantages reconcile. All-candidate concordance uses 2,106 comparable pairs, excluding six truth ties.
- Verified all eight output-file hashes listed in `results/status.json`, three frozen source hashes, five prepared-artifact hashes, freeze/execution links, and eight model/state hashes before and after. Prepared arrays and score readouts are unchanged in the raw results. Preserved V1 files retain their recorded hashes.
- All 384 returns completed: 21,696 main plant transitions, zero nonfinite returns, and zero failure events. Runtime fixtures used 608 transitions; batch/scalar error was 8.526512829121202e-14, policy-action Q versus V error 5.684341886080802e-14, and terminal99 error zero. Original tolerances were retained.
- The test receipt records 15 pure tests passed and four gated tests skipped. The runtime runner performed the authenticated numerical fixtures separately. Review did not rerun tests or fixtures.

## Measured results

All-candidate comparisons use 32 paired states and candidate-set Q values from one forced action followed by the original actor.

| Score | Mean Q gain versus raw Q1 | State-bootstrap 95% interval | Improved / harmed / unchanged | Mean candidate-set opportunity loss |
| --- | ---: | --- | --- | ---: |
| Raw Q1 | 0 | [0, 0] | 0 / 0 / 32 | 3.136473 |
| Q1 minus width | 0.153817 | [0, 0.461452] | 1 / 0 / 31 | 2.982655 |
| Matched distance penalty | 0.405389 | [0.077511, 0.809574] | 4 / 0 / 28 | 2.731083 |

The entire pooled width gain comes from state 6 (stall-low, time 0): the score switches from rare-grid candidate 2 to the original off-grid actor, candidate 6, gaining 4.92215307783642. The distance control also makes this change and three other beneficial changes. Its larger point estimate is descriptive; separate intervals versus raw Q1 do not test the direct distance-versus-width contrast.

**The support-restricted negative result must be retained.** Common-grid ranking has 13 eligible states containing 28 actions; 19 states with only one common candidate are excluded. Width harms one restricted decision by 0.30000000000000004, giving mean delta -0.02307692307692308. Its interval is correctly unavailable because eligible context/time strata are sparse. At state 18 (monotone-low, time 99), raw Q1 selects zero action, whose true Q is -0.7940484735204976 and width 2.0013201236724854. Subtracting width selects [0, -1, 0, 0], whose true Q is -1.0940484735204976 and narrower width 1.8938686847686768. This is lower return, not a physical failure event.

Rare-grid restricted width ranking improves two of 32 states, mean +0.0467710323326741; unsupported-action restricted ranking improves one, mean +0.013895891836385055. Both intervals include zero. These are separate rerankings within each action class, not decompositions of the pooled selected actions.

Width versus positive Q1-error Spearman correlation is defined in only 10 of 32 states: eight negative, two positive; mean -0.15488226393662008 and median -0.2875075790000986 among defined states. The 22 undefined values must not be replaced by zeros or omitted from the denominator disclosure. Pooled mean ranking concordance changes only from 0.6618687627476689 to 0.6618762747668998; matched distance reaches 0.6770639841928905. Constant-offset choices match raw Q1, and detached-width scores and choices match live width throughout.

## Numerical repair and interpretation limits

V1 stopped before any main-bank labels. A separate saved trace establishes identical observations producing different native actor outputs at batch sizes two and one, followed by amplified trajectory differences. V2 consistently uses one-row native float32 actor inference for candidate generation and continuation. It changes the numerical deployment convention, not model weights; no bitwise equivalence with prior batched deployment is claimed. Physical states, grid candidates, random continuous candidates, and support masses match V1 exactly. All-candidate support remains 47 common, 145 rare, and 192 unsupported; no width refusal occurred. The adaptation-only matched coefficient is 0.519506430786272.

V1's failed fixture execution used 304 transitions despite its stale zero counter; the preserved trace used another 304. V2 used 22,304 transitions including fixtures, giving 22,912 across these executions. V2 persists completed fixture-phase counts; an exception within a phase would still require accounting for partial work separately. Terminal checks use the declared reward expression and add no plant calls.

Inference is conditional on one trained checkpoint, one coefficient, this designed state/time law, and this candidate realization. The independent unit is the state, not 384 actions; full context/time strata contain only two or three states. Error-summary tables are candidate-row summaries, while ranking effects and intervals use states. Off-grid action mass zero and finite width do not establish joint support or calibration. The frozen95k width concerns one-step residuals; Q1 and the terminal100k deterministic continuation are different targets. No actor optimization, repeated score-maximizing policy, Qstar comparison, across-training-seed evidence, or coverage guarantee is tested. The result does not establish a benefit of adaptive width over the matched distance control.

## Exact SHA-256 identities

```text
diagnostic.py
6c0ed9257b54e10c3c44850bb05274cdb8289b9659524e26f089d3472f8b2fc9
run_diagnostic.py
a7e419494e2a4f90a3e783050efb10900bf536c19ed9bf7e118514cc547748d7
protocol.json
f4ee162725aef69df0a04ca4e13586b07d341b423b9ce18f4a67280d7e08d52c
tests/test_diagnostic.py
c6b40ca462400ca27adb518e03c727284f473aae1ada78a4ce8ad837e6b04a20
prepared/preparation_manifest.json
e7d627bcea6af292af132ea09f197fadcaa2fa521d5aece86585cdf236f48778
freeze.json
5e41f9a11fcf392ce0c42677993eaaa034e20d9a3284ff396cad1a12da2922de
results/raw_results.npz
623ec4c5f8bf5f9b7ec9277a4d8b98d82973f162f8847c79292a792370014e35
results/paired_results.json
929ecbf628158f5c11a8083d5d332c0182a9fb83f8a50f34196484e68db9e51e
results/fixture_results.json
a6bd8c0cc5246070f7dfa8b325cdf07ac499c30dcaf1bf196d7805d4611d783a
results/status.json
7b459716cbf48946bd9e2355a8aac2537cc28e3cf59d059bfa4f4d3e6011baef
```

# What the OOD calculations measure

**20 calculation/figure tests pass.** This is engineering validation using known
answers and constructed examples. It adds no training, model query, simulator
transition, checkpoint acceptance or scientific result. The annotated OOD YAML,
host/BCA training sources and original Unifloral sources are unchanged.

Open the [synthetic reader](ood-demo/index.html), [all figures PDF](ood-demo/all-figures.pdf)
or individual PNG/SVG/PDF/CSV exports in that folder. Every figure says synthetic.
All ten static figures were visually inspected; artist coordinates and uncertainty
envelopes are checked against the exported CSV. Missing assets and changed numbers
are deliberately tested and rejected. Final PNG/SVG/PDF hashes are recorded.

**Remaining visual check:** live browser interaction and narrow/wide layout are
unverified. Browser Use rejected local `file://` navigation under its URL policy.
No serving, alternate-browser or browser-command workaround was attempted.
The static exports and their data are independently available and checked.

| Calculation | Purpose | Arithmetic or interpretation |
| --- | --- | --- |
| Fixed-horizon return | Measure what an action actually achieves under one named later policy | Add raw simulator rewards over the declared horizon; stop at termination or the original time limit. No learned tail value. |
| Harm | Compare first actions from the same state under that same later policy | Reference return − candidate return. In the toy example, `−3 − (−12.1) = 9.1` lost reward. Positive loss is worse. The primary harmful label is strictly `loss > 1`; loss exactly 1 is not harmful under that threshold. |
| Within-state AUROC | Check whether larger widths identify worse actions | Compare every harmful/nonharmful pair in one state: correct order = 1, tie = 0.5, reversed = 0. With pair results `0.5, 0, 1, 1`, AUROC is `2.5/4 = 0.625`. A single-class state gives N/A. |
| Width minus support AUROC | Test whether width ranks harm better than a simple distance proxy | Compute both scores on identical actions/outcomes and subtract within the paired analysis. Distance is not true OOD ground truth. |
| Average precision and loss ranks | Describe harmful-action retrieval and ordering of continuous reward loss | Tied scores enter together. These are secondary measures; they do not replace the declared AUROC. |
| Risk versus retained fraction | Ask whether excluding high-score actions leaves less harmful alternatives | Keep complete score-tie groups. For losses `[0,2,4]`, retaining the lowest-risk two gives mean loss `(0+2)/2 = 1`, versus 2 for all three. This is an offline selection diagnostic, not a policy the agent executed. |
| Residual coverage | Check whether the band contains the frozen host's prediction error | Illustrative one-critic target: `1 + 0.9×8 = 8.2`; predicted Q = 6, so error `|8.2−6| = 2.2`. Width `1.5×1×2 = 3` covers that error. The host target is estimated; it is not known true Q. Exact multi-critic/target conventions remain in the adapters. |
| Radius rules | Choose how wide the residual band is | Ordinary conformal uses ordered score `ceil((n+1)×(1−alpha))`. At alpha 0.1, nine scores give rank 9; eight scores require infinity. Bayesian bootstrap and conformal components are retained, and the deployed radius is their maximum. |
| IID tolerance-rank reference | Separate a stronger conditional-risk reference from marginal conformal coverage | At 90% coverage and 95% credibility, the maximum of 28 IID scores is insufficient (`1−0.9^28 < .95`); 29 can suffice (`1−0.9^29 > .95`). This diagnostic does not establish IID or a weighted-shift theorem. |
| Coverage uncertainty | Show precision of the test-bank failure fraction | Exact binomial intervals are conditional on the frozen model/band and independent test episodes. Zero failures in ten episodes still gives a nonzero upper 95% failure bound of `1−0.025^(1/10) ≈ 0.3085`. Infinite bands are explicitly counted as vacuous. |
| Paired seed/reset bootstrap | Avoid pretending that many correlated actions are independent experiments | Resample training seeds, then whole paired reset blocks. Keep scores/collectors/captures together, using common block draws for shared resets. The two continuation analyses use the same declared resampling stream. Five seeds limit precision. |
| Whole-policy comparisons | Ask whether BCA actually improves the learned policy | Keep reserved final means separate from averages over all 200 common 5k-to-1M evaluations. Pair by seed. Neither local harm ranking nor residual coverage alone proves a policy benefit. |
| Actor-update averages | Avoid counting skipped actor steps as zero-loss observations | Logged `[2,0,4,0]` with updates on rows 0 and 2 has actor-update mean `(2+4)/2 = 3`, not 1.5. A genuine zero on an update row remains included. |

The unfamiliar action in the toy oracle is beneficial (`−3 − (−0.1) = −2.9`
loss), while a familiar action is harmful. This prevents a distance proxy or a
large prediction error from being treated as a ground-truth harm label.

## Statistical safeguards that were exercised

- Duplicated applied actions retain aliases but get one ranking vote; aliases of
  the reference are excluded. Conflicting duplicate outcomes/scores are refused.
- Shared reset states count once in unique-state denominators and pooled AUC.
  Collector/capture strata keep their declared equal weighting. Missing strata
  have a separately labeled available-strata estimate; the complete estimate is N/A.
- The fixture contains 48 collector/capture rows per seed but only 36 unique
  states. It does not claim 48 independent resets or treat actions as seeds.
- States with only harmful or only nonharmful alternatives are N/A. Sparse
  estimates remain visible but cannot support broad claims.
- Both rank calculations are independently checked; explicit saved Bayesian
  draws agree with the unchanged production JAX radius function.
- Pooled AUROC can be 0.75 while each state's AUROC is 1.00. Separate columns
  prevent population-level score shifts from being mistaken for within-state skill.
- A positive global radius preserves scale ordering and ties. Increasing the
  radius can widen coverage while leaving harmful-action ranking unchanged.
- Bootstrap rows stay paired. Missing strata, an unestimable bootstrap draw or
  one training seed withhold intervals. Partially crossed reset designs are
  refused pending an explicit design. Eight-contrast family intervals use the
  declared approximate Bonferroni bootstrap correction, never an exact-test claim.

## Reproduction and preserved attempts

Use the existing Python 3.10 WSL runtime and unchanged root requirements, plus
`experiments/ood/requirements.txt` for Matplotlib 3.10.9. NumPy 2.2.6, SciPy 1.15.3
and the existing CPU JAX installation produced the accepted result. No fresh
environment installation or GPU reproduction is claimed.

```bash
JAX_PLATFORMS=cpu python -m unittest experiments.ood.test_analysis -v
python -m unittest experiments.ood.test_protocol -v
python check.py
# A NEW directory is required; existing evidence is never overwritten.
python -m experiments.ood.report --output runs/ood/my-calculation-reader-01
python -m experiments.ood.report --output docs/ood-demo --verify-only
```

The first red test (`analysis-tests-v1`) correctly failed because `analyze.py`
did not yet exist; its log and actual exit 1 remain. Version 2 passed 17 numeric
tests. Version 3 passed 20 tests using a temporary report fixture. Version 4
passed all 20 tests and retains the accepted reader plus deliberate missing-asset
and changed-CSV copies in separate ignored directories. Deliberate refusal cases
are successful tests, not scientific failures.
Version 5 repeats the 20 tests after fixing CSV line endings to LF so Git cannot
change a verified exported-table hash. The numeric tables remain identical.

Initial render v1 and its exact source copy remain archived. Render v2 improves
colors, axis separation and title wrapping. Final rendering additionally shows
the single-point tied-score curve, distinguishes an overlapping maximum radius
with a dashed line, and supplies one combined PDF. Prior renders and receipts
remain unchanged. Render v3 and its exact source were archived before the final
LF-only CSV format correction in v4. The [machine-readable receipt](../experiments/ood/analysis_validation.json)
records actual exits, hashes and the browser-review limitation.

A final verification using Windows Python 3.14.2 / NumPy 2.4.1 / SciPy 1.17.0
failed exact table equality: 13 values differed by at most `2.220446049250313e-16`.
The unchanged report verifies exactly in the documented WSL runtime. This is not
a claim of byte-identical reproduction across dependency versions; no equality
gate was relaxed. A separate WSL path-quoting failure occurred before its verifier
started. Both failed checks and the successful pinned-runtime check are recorded.

The reader currently accepts only its synthetic fixture. Ingestion of actual
checkpoint/collector artifacts, missing real training seeds, fresh coverage
banks and final scientific reports still require Task D's validation. ReBRAC's
live recorded-next-action contract remains unresolved. No old native recovery,
training queue, original outcome or thesis PDF was changed.

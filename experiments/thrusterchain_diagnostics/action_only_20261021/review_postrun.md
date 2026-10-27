# Independent post-run scientific review

**POST-RUN SCIENTIFIC REVIEW PASSED.** No remaining material finding in the completed controlled evaluation, numerical aggregation, or final HTML's scientific claims. This conclusion applies to the identities below. Browser rendering and interaction remain unverified.

Reviewer: independent scientific/code review agent, 2026-10-21 UTC. The frozen implementation, protocol, preflight review, checkpoint, and run records were not edited. No training or main experiment was launched by this reviewer. Only saved evidence was inspected and calibration was replayed.

## Completed evidence and independent checks

The frozen protocol completed all **200/200** registered replications, IDs 0–199, with no missing, failed, or historical-failure records. There are **600** paired mixture records, **1,200** method summaries, and **8,400** context/time subgroup summaries. All 200 calibration-bank identities are distinct. The execution status is terminal and the frozen source/preflight hashes still match.

Independently checked every saved NPZ checksum and every record's freeze, checkpoint, state-bank, support-audit, replicate, sample-size, and mixture identities. Reconstructed all seven saved calibration/evaluation sampling streams from the declared seeds: state IDs, categorical source actions, calibration noise, evaluation state IDs, base actions, mixture uniforms, and evaluation noise. Posterior seed bindings were checked for all records. These checks verify the declared fresh source law, independent source/query streams, and paired target construction, without rerunning the model.

From all saved arrays, independently recomputed the exact calibration/query ratios, normalized residual scores, ESS, positive-weight and actor-hit counts, overall coverage numerators/denominators, refusal counts, finite conditional coverage, finite-and-covered fraction, finite widths, neutral-fallback multiplier means, subgroup counts/ESS, and paired differences. All matched. The lambda-zero methods agree exactly in all 200 replications. Uniform subgroup ESS and positive-weight counts correctly equal their calibration row counts; the earlier preflight bug was fixed before freezing and requires no post-hoc correction.

For fixed **replicate 000**, called the unchanged preserved native `calibration/wbcp.py` directly on the saved calibration scores, weights, exact distinct query masses, and declared posterior seed. All three mixtures and both methods reproduced the saved threshold arrays exactly, including infinite thresholds; all stored posterior-crossing hashes also matched. This was calibration replay, not another training or evaluation run.

Independently recomputed **309** aggregate estimate objects, including means, standard deviations, standard errors, and Student-t intervals; all aggregate counts and CSV row totals also matched. Fully defined quantities use 200 independent calibration replications and df=199, including paired differences. Four focused analysis tests also passed. Final regeneration changed provenance/export fields, not any mixture statistics; the final numerical aggregate was checked against the independently verified version.

## What the observations support

Means across the 200 replications:

| Mixture | Method | Coverage including infinite bands | Finite and covered / all queries | Refusal | Finite conditional coverage |
| --- | --- | ---: | ---: | ---: | ---: |
| 0 | Both, exactly equal | 94.11% | 94.11% | 0.00% | 94.11% |
| 0.5 | Uniform | 93.48% | 93.48% | 0.00% | 93.48% |
| 0.5 | Exact categorical IW, per query | 95.48% | 78.82% | 16.66% | 94.57% |
| 1 | Uniform | 92.84% | 92.84% | 0.00% | 92.84% |
| 1 | Exact categorical IW, per query | 97.61% | 64.79% | 32.82% | 96.44% |

All observed weighted refusals were **posterior-infinite**: 136,470 of 819,200 query rows at mixture 0.5, and 268,864 of 819,200 at mixture 1. There were **zero no-hit/all-zero-weight banks** and no structural support failures for the specified grid targets. The deployed actor was off-grid at all 64 fixed states; its separate support audit does not assign a finite ratio or coverage claim to those commands. Explicit support outcomes are retained in all 600 paired export rows.

The weighted procedure trades higher interval containment for more vacuous intervals and fewer finite covering bands. Pooled uniform coverage remains above the nominal 90% level at all three mixtures; this does not imply per-context coverage. Finite conditional coverage and each method's finite-width mean use method-specific accepted subsets. The reported jointly-finite width contrast compares the same subset. The paired finite-and-covered difference is −14.67 percentage points at mixture 0.5 (95% interval −14.98 to −14.35) and −28.05 points at mixture 1 (−28.41 to −27.69). These results do not establish policy benefit, dominance under a specified decision loss, or a defect in the exact categorical ratio.

## Final report and HTML review

Reviewed `analysis/RESULTS.md`, the aggregation implementation, and `presentation/ThrusterChain_diagnostics_results_v3_delivery.html`. The A1–A8 stage descriptions, formula, tested/untested labels, and conclusions agree with the implementation and evidence. The final prose distinguishes the per-query Bayesian variant, one-step residual versus true policy-value error, common fixed state law versus occupancy, same state probabilities versus realized sample counts, no-hit versus posterior-infinite refusal, and the frozen post-hoc control versus a preregistered training study.

Independently verified **10** displayed source excerpts against their actual file hashes, line ranges, and text; **3,824** unique IDs and **780** resolved internal anchors; unchanged historical scripts; and exact preservation of the previous guide after removing the new section and two explicit historical-plan notices. Analysis/build receipts match the final artifacts. The records-only portable aggregation mode explicitly disclaims raw-NPZ verification; the local review above did verify all raw files.

Browser rendering, navigation, and JavaScript-runtime QA were blocked when the parent's browser tool denied the local-file URL under its security policy. No workaround was attempted by this reviewer, and the HTML records that limitation. Static inspection does not establish live rendering or interaction.

## Final identities

| Artifact | SHA-256 |
| --- | --- |
| `freeze.json` | `1fc66520f2715c1e5c2396c1fcc7db9279bfecb20dedf61d58227fd575d7a18f` |
| `analysis/aggregate_results.py` | `b97d83e4760bcd09e5332346be2fda80c594d8494aabd45eb70e4f61c0902af6` |
| `analysis/aggregate.json` | `e3e548446f30587d03a14ebb2145e0320d9ee36c0f80399140dc99df6cd4acd8` |
| `analysis/RESULTS.md` | `48a82e86d1a1ffdea71f392bbd6abf75567c6fd2ff812ce889716a35f648a895` |
| `presentation/ThrusterChain_diagnostics_results_v3_delivery.html` | `9e3ddabd4bfd6a1d3f3bf7f801c6b6f3310ab9843e722e2726212f563f86fe38` |
| `presentation/build_v3_receipt.json` | `01399168f908041f8d589a4c065392beb879124c1c782cba2af4b54e23c9077f` |

The HTML is 11,470,144 bytes. Scope remains conditional on one trained checkpoint and one fixed 64-state bank. There is no across-trained-seed or new-episode population inference, no borrowed frequentist coverage theorem, no off-grid/occupancy/true-policy-Q coverage claim, and no actor update or policy-improvement test. Independent neural forward inference for every saved query was not repeated in this review; native scorer parity and state immutability were checked by the previously reviewed checkpoint fixtures, while the post-run replay audited saved scores and the actual native calibration calculation.

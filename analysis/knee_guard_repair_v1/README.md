# Knee experiment guard repairs — local, isolated amendment

The four scoped repairs are implemented and **82 selected CPU tests pass**. Retained-state support remains **INCONCLUSIVE**. This is not evidence of BCA efficacy and does not authorize another scientific run.

## What changed

| Defect | Repair | Verification |
|---|---|---|
| Integrity-check exception left a completed worker successful | The shared Stage 1 / Stage 2 finalizer fails the job when verification raises; it retains prior errors, outputs and counters. | Real worker entrypoints with mocked jobs/counters write failed receipts for integrity errors. |
| Legacy transition correction could be applied twice or to another producer | Restrict it to the authenticated completed v2 attempt, its 70 source files, 98 receipts and 32 unit reports. New, changed, incomplete and already corrected records are refused. | Original reconstructed count remains 10,994,000; evaluation including R0 remains 7,538,000; decision remains INCONCLUSIVE and V0(f) fails all seeds. |
| Missing timing became a zero upper bound | Missing/invalid elapsed time stays null, with unbounded units carried through attempt, prior and cumulative summaries. Nominal timeout is separate configuration. | Missing, negative, nonfinite, string/bool timing; valid zero; propagation; old ledgers with misleading zero. A kill wait and bookkeeping can exceed the configured worker timeout. |
| D3 used all fitting rows instead of retained training support | Apply the CUTOFF mask to observations, episode IDs, times and deep flags before normalization and L1.5. Insufficient or malformed evidence is inconclusive. | Actual D3 adapter wiring; removed-outlier invariance; 60-row rejection; 64-row boundary; malformed IDs/shapes; original numerical bars. |

Independent code review found one additional malformed-identifier issue (NaN episode IDs could defeat leave-episode-out exclusion). It was fixed and regression-tested before the final test run.

## Isolation and adoption boundary

The working copies are:

- `experiments/thrusterchain_bca/zeus_20261026_validation/knee_edge_ood_v3_guard_repair_v1/`
- `experiments/thrusterchain_bca/zeus_20261026_validation/knee_s10_candidate_guard_repair_v1/`

The original v3 and candidate sources, manifests, approval files, reservations and outcomes were not edited. No approvals, seals or output directories were copied into these working copies. Candidate scientific `main()` is explicitly disabled. These are repaired source copies for review and future incorporation, **not a sealed production launcher**. Existing runners still reference historical source; they do not automatically gain these repairs. Do not reuse an old reservation or reseal an old attempt to adopt them.

This isolation preserves byte-authenticated historical evidence and avoids modifying files an unobserved worker could import. Saved records establish that the known dose screen is terminal (12 branches and three evaluations completed); filesystem reads alone are not a global process-liveness check. WSL access remains denied and was not retried.

Initial HEAD was `af30c346d35b2fd13a3c0f8b3b9d1ae49962f582`. Concurrent commit `d703134637d607e57862256cd70268311b7429e6` corrected `KNEE_BED_CONCLUSION_20261031.md` and its earlier claim that the user stopped the entire bed. That work was left alone. Only the Q-error method-change track is described there as shelved; this repair does not expand or cancel any research authorization. All 52 snapshotted source/fixture files remained byte-identical at the final check.

## Saved evidence and scientific interpretation

The first candidate report records retained deep-episode reference counts **65, 70, 60** at **t=0,25,50**, in both worlds. The unchanged minimum is **64**. The final time group is insufficient. These are saved report counts; raw distances were not recomputed.

The r3 PASS used **all fitting rows**: start share 127/128 in both worlds, anchor share 48/48 in stall and 44/48 in no-stall. Those values do not establish support relative to CUTOFF-retained data. The historical PASS file remains unchanged, and the repaired check does not reinterpret it as a retained-only pass.

The inspected candidate producer persists summary JSON, not its observation/action arrays or anchors. No raw array artifact exists in that candidate directory. This is a scoped availability finding, not a claim about every file on the computer. Therefore empirical retained-only distances cannot be recovered from the current summaries alone.

The repaired metric retains **64th-nearest-row standardized Euclidean distance**, the **0.99** reference quantile, and **95% starts / 90% anchors** thresholds. Reference queries exclude their own episode. The 64 neighbors are rows, not 64 distinct episodes; distinct episode counts are reported for the same-time reference groups. It is a typicality diagnostic, not proof of equal state distributions or joint conditional action coverage. Its reference normalization now comes only from retained fitting rows. This is an explicit population correction, not radius or pass-mark tuning.

`G2_RULE_r3.md` in the copy is historical: it explicitly says all fitting rows. `RETAINED_SUPPORT_AMENDMENT.md` defines the corrected population for this repair. No past rule or outcome was silently rewritten.

The actual candidate `pi_ref` is a deterministic analytical, projected LQR continuation, with no learned actor or checkpoint. Its gains and action calculation were not changed. The G2 code computes deterministic Q-pi references without target-policy noise and never updates a learner. The draft's proposed later policy-evaluation learner freezes its actor and turns target smoothing off, but that learner path has **not** been implemented or runtime-validated here. It would be a policy-evaluation diagnostic, not the unchanged native TD3+BC training operator or proof that a fitted critic is exact. Projection into recorder coordinate ranges does not prove empirical joint support; the copied docstring now says so.

## Checks actually run

`TEST_RECEIPT.json` records commands and EST observation time. `guard_and_pure_regression.log` records **69 passed**; `retained_support.log` records **13 passed**. All **53 copied/new Python files** also passed non-executing AST syntax parsing. These cover the new guards plus selected existing objective/decision math, fake training/fork control flow and terminal/accounting regressions. No real optimizer, calibration, model inference, simulator transition or GPU call was made. Real simulator tests and full CUDA integration were deliberately excluded. This is not a claim that every copied historical test was run.

Two old fixture expectations were corrected in the isolated copies: arbitrary reports no longer qualify for the historical correction, and a bare timeout no longer supplies a verified physical wall-time bound. Original tests remain untouched.

## Smallest proposed next evidence step

First prefer recovering an already saved, authenticated raw-array export if one is located; that would cost zero simulator transitions. The inspected producer did not persist one.

If collection is approved, the bounded proposal is **support-only**, with unchanged recorder/physics/mask and no Q-pi continuation evaluation:

- 2,048 episodes per world; 1,231 fitting episodes per world under the existing split rule.
- 128 independent start queries and 48 anchors, 16 each at t=0,25,50.
- At most **820,400 transitions**: 409,600 collection + 409,600 full lineage-validation replay + 1,200 anchor-prefix steps. At most 4,272 resets.
- **600-second global ceiling**, zero actor/critic/auxiliary updates, zero calibration, zero model calls.
- Keep k=64, quantile 0.99, 95%/90% shares. Insufficient data, invalid evidence or a cap hit is INCONCLUSIVE; sufficient data below a share threshold is FAIL. Stop without top-ups or threshold changes.
- Persist full arrays, identities, retention masks and query states/times. New samples are independent development evidence, not confirmation. A pass establishes only this retained-state typicality check, not full G2 admission or BCA efficacy.

The sample size addresses the known reference-count bottleneck by design, not by looking for a successful BCA outcome: P(deep and retained)=0.15×(0.98×0.5+0.02×0.8)=0.0759. With 1,231 fitting episodes the expected count is 93.43 per time group; a nominal binomial calculation gives P(count<64)=0.0003503, with a conservative six-group union bound 0.002102. This assumes independent episode streams/splits and does not predict a distance-check pass. The two worlds are paired, not independent replications.

Exact proposed limits and provisional fresh streams are in `PROPOSED_SUPPORT_ONLY_CHECK.json`. Adoption of this sample-count change, full stream-collision checks, an immutable executable contract and a run reservation are still required before execution. No such collection, approval, reservation or launch was performed here.

## Evidence files

- `SOURCE_SNAPSHOT.json`: original-to-copy SHA256 mapping.
- `SAVED_EVIDENCE.json`: source/report hashes and retained-support limitations.
- `AUTHENTICATED_SAVED_ANALYSIS.json`: read-only recheck of the historical accounting and unchanged scientific result.
- `TEST_RECEIPT.json` and test logs: actual local verification.
- `REPAIR_DIFF.patch`: reviewable changes relative to preserved source.
- `OUTPUT_MANIFEST.json`: hashes of the repaired sources and deliverables; not a research authorization.

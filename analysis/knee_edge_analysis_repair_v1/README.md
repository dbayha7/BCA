# Isolated knee-edge analysis repair v1

This package implements two already specified analysis repairs without importing or changing the running experiment. It reads completed saved JSON only. It cannot train, calibrate, simulate, restart a reservation, or unlock Stage 2. The original v1 A2 and v2 A3 INCONCLUSIVE results remain unchanged. The current Stage 1 run is explicitly exploratory because the preceding gate did not pass.

## Current evidence

At October 30, 2026, 11:56:59 p.m. EST, saved receipts showed 12/12 warmups and 5/12 continuations complete, zero completed evaluations, and no failed supervisor receipts. The sixth continuation, `branch_s1_random_stall3`, had saved additional step 3,500 at 11:56:40 p.m. EST. No terminal attempt report existed. This is a saved-file observation, not a process-liveness guarantee.

The real-input readiness command returned `waiting_for_terminal_saved_arrays`, with `empirical: false`. There are **no new empirical V0(f) or decomposition results**. `RUN_SNAPSHOT.json` records the observation and receipt hashes; `outputs/readiness_2/RESULT.json` records the missing final-report prerequisite after the final lineage-check fix. The first readiness receipt is also preserved. The implementation's successful fixture outputs are not scientific evidence.

## Exact analysis

For each seed 0, 1 and 2 independently, use the final 20k native CUTOFF and RANDOM models in stall3, on the shared E2 anchor panel. Only its first 48 anchors, 16 each at times 0, 25 and 50, enter the gated analysis; time 75 is excluded. The adapter validates the complete 64-by-14 arrays, fixed slot order and reference receipts. Invalid or nonfinite arrays are rejected; missing arrays are never imputed.

**V0(f), the missing RANDOM control check.** Common validity is the intersection of CUTOFF and RANDOM validity. R-valid additionally requires all three RANDOM item-5 checks, with maximum prediction errors against the no-stall references at s05/s10/s20 no greater than 0.10/0.15/0.30 respectively. CUTOFF item-5 failure does not remove a RANDOM-eligible anchor. The existing three deterministic polynomial summaries are reconstructed from cached values in the original arithmetic order; this does not fit an uncertainty model or use new evaluator calls. Slot coordinates preserve the producer's literal float32 bits, including s10 `0xBF666667`, which is not the ordinary cast of decimal -0.9.

On R-valid anchors, V0(f) requires both:

- median rho-hat at s10 >= 0.30;
- median KC at s10 <= 0.3 * (median rho-hat + 0.15).

These inclusive bounds are unchanged and one-sided: a negative KC may pass. An empty eligible set is inconclusive. Every seed must pass; pooling cannot rescue a failed seed. V0(c)'s separate 32-anchor requirement is not silently inserted into V0(f), and a V0(f)-only pass is not a full validity pass. CUTOFF companion values are descriptive only.

**Reference-policy decomposition.** For every common anchor and each of s05/s10/s20, subtract that same anchor's edge-knee value first:

```
KC_clip = delta Q1 - delta Qpi_clip
KC_own  = delta Q1 - delta Qpi_unclipped
D       = delta Qpi_unclipped - delta Qpi_clip
KC_clip = KC_own + D
```

The same identity is checked for the paired CUTOFF-minus-RANDOM differences. D need not cancel across arms. Per-anchor components, masks, denominators, per-seed medians and means are retained. Medians are not assumed additive. The original clipped reference remains primary; the deployed-policy reference is an explanatory diagnostic, not a replacement criterion or true-Q guarantee. Own-policy references differ between arms and also differ from the native smoothed target operator.

## Input and output safeguards

`SOURCE_CONTRACT.json` pins the source definitions, protocol, allowed attempt, A4 manifest and the analysis thresholds before new evaluation outcomes were available. Input is restricted to the resolved approved attempt and its resolved JSON files; no receipt-supplied arbitrary file paths are followed. All three evaluations require normal worker and supervisor completion, matching job/seed identity, no smoke or seed overrides, and no nonreproduction marker. Recorded evaluation source digests must match the contract. Inputs and pinned sources are hashed again before writing. Every output uses a new directory inside this package; existing outputs are not overwritten.

The three Q-pi receipts must match the prescribed worlds, clipping, discount, horizon and termination. Anchor pairing is supported by the shared source-pinned evaluation job and fixed times/slots/order. The reports do not store independent anchor IDs or complete probe hashes; this limitation cannot be repaired from absent data. Producer AST fingerprints are recorded but not independently authenticated by this tool. Source-file checks are not a complete transitive dependency or runtime-seal audit.

## Verification performed

Windows `.venv` Python ran 45 focused tests successfully, covering boundaries, negative one-sided values, masks, empty sets, exact float32 slots, all reference receipts, polynomial fixtures, decomposition signs, nonadditive medians, paired summaries, malformed inputs, source/lineage and in-read mutation, incomplete/failed/smoke reports, path confinement, output isolation and a complete three-seed file fixture. No live experiment module was imported. The real startup command was also exercised and safely reported the missing final arrays. See `TEST_RESULTS.md`.

Run the focused tests from the repository root:

```powershell
.\.venv\Scripts\python.exe -B -m unittest discover -s analysis/knee_edge_analysis_repair_v1 -p 'test_*.py' -v
```

After the same attempt has completed its three final evaluations and terminal report, the read-only analysis command is:

```powershell
.\.venv\Scripts\python.exe -B analysis/knee_edge_analysis_repair_v1/analyze_saved.py --out analysis/knee_edge_analysis_repair_v1/outputs/final_saved_analysis_1
```

This launches only saved-array analysis, not an experiment. A source/receipt mismatch must be reviewed; do not edit the contract to force acceptance. If that output directory already exists, preserve it and use a distinct analysis identity. No empirical result or Stage 2 authorization is implied by this prepared command.

## Remaining audit issues and interpretation

This separate package addresses R1 (missing V0(f)) and the analysis portion of R2 (clipped/unclipped reference confounding). It does not fix the live runner or establish that any scientific gate passes. Five other audit repairs remain outside this authorization: R3 batched transition accounting; R4 transitive dependency sealing; R5 timeout/interruption finalization; R6 physical retry-spend accounting; R7 exploratory-wrapper/smoke approval binding. See the preserved `analysis/ood_progress_audit_20261030_v1/AUDIT_AND_REPAIR_PLAN.md` for the full repair plan.

Current native learning and withheld-region exposure do not establish harmful overestimation, CUTOFF-versus-RANDOM withholding harm, BCA detection, improved action choice or policy benefit. Complete final arrays are pending. Three seeds remain exploratory, and anchor rows are not independent training replicates. This analysis does not provide new BCA efficacy evidence or reverse the earlier conditional-quartile study's failed policy-rescue screen.

Only files inside this analysis directory were created or edited. No live source, protocol, reservation, approval, experiment receipt, model or dataset was modified. No commit, push, training, calibration, simulator call or WSL call was performed.

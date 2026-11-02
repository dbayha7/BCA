# knee_edge_ood_v1: step 0.2 (approval A1) package

Approved by the user's commit `3115ed35` (2026-10-30 10:30 EDT): code, schema, seeds and synthetic tests only; no
simulator collection, training or checkpoint access. The protocol is `PROTOCOL.md`, a byte copy of
`knee_edge_ood_design_v1/DRAFT_PROTOCOL.md` revision 13; its sha256 is in `MANIFEST.json`.

## What is here (protocol Part 4)

| Module | Protocol | Role |
|---|---|---|
| `knee_region.py` | 4.1 | The only region predicate: past_knee_a0, depth, in_deep, load_g, load_band, slot table; import-time check against canonical `stalled()` |
| `region_independent.py` | 4.2 | Independent second implementation (bit test, exact Fractions, canonical check), written by a separate agent that never read `knee_region.py` |
| `recorder.py`, `dataset_knee.py` | 3.2-3.3, 4.3 | knee_mix15 reset law, five-uniform recorder, schema `thrusterchain.knee_edge_trajectories.v1` with bit-for-bit lineage replay; cross-rejection with the core schema tested |
| `masks.py`, `reconcile.py` | 3.7, 3.9, 4.7 | CUTOFF / three-band RANDOM masks, calibration rows, normalization intersection, plumbing (INCOMPLETE) and expected-value (INCONCLUSIVE) checks, twin check |
| `bindings.py` | 4.4 | Stream receipts and assertions on the exact arrays every learner receives; knee-edge data provider for rerun1 `_initialize_data` (imported read-only) |
| `probes.py` | 4.4 | The single probe builder: Part 0 slots by name (a swap is caught), plus sealed point sets for G0.5/F, the reach forecast, T-images and the 4.8 known answers; every array checked by both region implementations |
| `qpi.py` | 4.5 | Environment-based and raw-`dynamics()` Q^pi evaluators sharing only the raw actor forward pass |
| `worlds.py`, `thrust_check.py` | 4.5b | World dicts and assertions, literal T3/T15 identities, data-level thrust reconstruction |
| `frozen_checks.py` | 4.8 | Comparison rules, tolerances, preflight and the narrow 5.0 exception for the user-run step 0.3 |
| `seeds.py`, `seeds.json` | 3.7, 3.9, 4.7, Stage 1-2 | Name-derived seeds (Stage 1 rows; Stage 2 per-fork seeds keyed by seed index, arm and world), sealed draw rules for every bank and anchor set, and a clean collision audit: 3,478 prior seed integers, raw and derived reset seeds under stream keys 1701, 1801 and 1901 |
| `expected_values.json` | 4.7(ii) | 6 SD bounds sealed from `screens/random_control_mc_output.txt` (sha256 2dc0c383...f716) |

Tests (`test_*.py`, unittest, 77 tests): run from the repository root with
`PYTHONDONTWRITEBYTECODE=1 .venv/Scripts/python.exe -B -m unittest discover -s experiments/thrusterchain_bca/zeus_20261026_validation/knee_edge_ood_v1 -p "test_*.py"`.

## Interpretations and limits, disclosed

- **"No simulator" in A1.** The tests call the canonical `dynamics()` and environment on tiny smoke datasets (12
  episodes, test-only entropy 77001, and a 10-episode core-schema dataset for cross-rejection). No benchmark seed is
  used and nothing is collected for the study. Mask and reconciliation tests use dynamics-free synthetic loads.
- **Load edge 0.4 is unreachable from float32 observations.** No float32 pair sums in float64 to exactly 0.4 or the
  float64 just below it (found by the 4.2 author's whole-range search). The 4.6 edge test is therefore done exactly at
  1.0, and at 0.4 on the float64 values and the nearest float32 pairs either side.
- **Not in A1; prepared with each later approval:** the step 0.3 driver that builds rerun1 `FrozenPolicy` objects
  (A2, user-run, WSL GPU), the step 0.4 collection and anchor driver (A3), and the Stage 1 training and evaluation
  driver (A4). Each reuses these modules; none may change a threshold, seed or bound sealed here.
- `bindings.create` and the 4.8 preflight need the recorded WSL GPU stack; on Windows they stop at the runtime receipt,
  by design.
- **Actor identity (4.5) belongs to the drivers.** Each evaluator's separate check of the loaded actor (the actor file's
  sha256 against `actor_params.sha256` in the cell's manifest, plus `assert_exact_state` against the export's
  `actor_descriptor`) is called by the A2/A3 drivers, which own model loading; A1 never loads a model.
- **Validation decodes every role.** `dataset_knee.validate` replays every row of all four roles, as 4.7's replay requires;
  the learner receipt field is therefore `protected_transition_rows_bound_to_learner` (0), not "decoded".
- **Review.** An independent three-lens review (with skeptics and a critic) found 2 high and 7 medium items, all in
  safeguards rather than computed values; all 17 items were fixed, and one fix exposed a real bug: the 4.8 lambda
  comparison had used the Q-value floor max(1, |v|) instead of a pure relative tolerance. A bounded re-check confirmed
  13 of the 17 fixes and found one regression (the probe builder refused three of step 0.3's G0.5 points) and four
  partial fixes; all five were then fixed and covered by tests (not re-reviewed a third time).
- **Probe point sets and protocol 4.4.** 4.4 says every gated probe array holds "the Part 0 slot value", but step 0.3
  names point sets outside the Part 0 table (G0.5/F: -0.5, -0.60000002, -0.69999999, -KNEE, s05, -0.89999998, -1; the
  reach forecast adds -0.25, 0, 0.25, 0.5). The sealed protocol is not edited after approval: each of those sets is
  sealed here as its own probe kind with its exact float32 bits, and Part 0 slots are taken by name.

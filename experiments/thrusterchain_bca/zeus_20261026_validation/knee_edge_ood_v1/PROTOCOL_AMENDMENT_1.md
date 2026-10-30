# Protocol amendment 1: L1.5 calibrated within group

**Status.** Decided by the user in session on 2026-10-30 (design decision); it takes effect only through the user's
A3 approval, which names this amendment. It amends `PROTOCOL.md` (revision 13, sealed with A1) and corresponds to
revision 14 of `knee_edge_ood_design_v1/DRAFT_PROTOCOL.md`. No study data existed when it was made: the evidence
below comes from a smoke dataset drawn with a test-only entropy, never from the sealed dataset entropy.

## The problem

L1.5 (Part 2) required at least 95% of E2 start states and 90% of gated E2 anchors to lie within the training 0.99
quantile of 64-nearest-neighbor distance, the quantile taken over training rows in general. E2 states sit in a small
group of the training data: deep-load episodes (15%) at a fixed time step. Their nearest neighbors are farther apart
than those of a typical training row, so the general quantile flags them even when they are fully inside the data.

On a smoke dataset of 1,280 episodes (same recorder, reset law and stall world; test entropy):

| Rows | Share within the general 0.99 quantile |
|---|---|
| Training rows that are deep-load episode starts (t = 0) | 0.805 |
| Training rows at t = 0, all loads | 0.918 |
| Training deep-load rows, all t | 0.970 |
| The 128 Stage 1 E2 bank starts | 0.820 |

States that are in the training data fail the 95% bar at the same rate as the E2 starts, so the test as written does
not detect state shift. It would have ended step 0.4 INCONCLUSIVE for a reason unrelated to OOD.

## The amendment

L1.5 compares each E2 state with training rows of its own group:
- **Group:** fitting rows from deep-load episodes (reset mixture component 1, current in [0.9, 1.1]) at the same time
  step t as the E2 state (t = 0 for bank starts; t in {0, 25, 50} for the gated anchors).
- **Threshold:** the 0.99 quantile ("higher" method) of the group's own leave-one-episode-out 64-nearest-neighbor
  distances, measured against all of the world's fitting rows. Distances use the same 14 normalized inputs.
- **Query distance:** each E2 state's 64-nearest-neighbor distance to all of the world's fitting rows.
- **Bars unchanged:** at least 95% of the 128 Stage 1 E2 bank starts and 90% of the 48 gated anchors within their
  group's threshold, in each world. A miss is still INCONCLUSIVE.

This tests what Part 2 already says about E2, "a sub-population of the collection law's deep component, with
overlapping support": an E2 state passes when it is as close to the data as the data's own deep-load states at that
time step. Nothing else changes.

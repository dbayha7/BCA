# Reach screen sensitivity (written after the rule ran)

**Status.** This file records a sensitivity check. It does not change the selection rule in `../REACH_SCREEN_RULE.md`, which stays exactly as written.

**Source.** An independent review of the draft protocol (2026-10-29) wrote `reach_variants.py`. It was run here unchanged, and its output is in `reach_variants_output.txt`.

## What was found

The rule's visited-row model was a_new = scale * a_old + 0.87 * m.
- The constant 0.87 is 1/(1 + 4 lambda) evaluated at the old lambda of about 0.037; it comes from the design panel's dataset design.
- The screen exists to correct lambda for the new reset law, so the constant is stale under that same correction.

Selections under three readings of the model, at d = 0.10:

| Reading | Feasible at d = 0.10 (both depths) | Selected by the written rule | mix15/C visited minimum |
|---|---|---|---|
| As written (0.87) | 2 (8) | mix15/C | 0.317 |
| Factor corrected to 1/(1 + 4 lambda'), post hoc | 5 (15) | wide/D | 0.389 |
| Visited rows limited to current >= 0.9 (72 per cell), post hoc | 3 (12) | mix25/D | 0.417 |

- The second and third readings were written after the rule ran. They are **post-hoc sensitivity readings**, not part of the rule.
- Only **mix15/C** and **mix15/D** are feasible under all three. The selection itself comes from the as-written rule alone.
- **Binding cell for mix15/C as written:** hole, seed 2, with 40 of 126 visited rows forecast in R_deep-, against 38 needed for 0.30. A lambda scale of 0.72 instead of 0.751, or an anchor factor of 0.80 instead of 0.87, would make it fail.

## Decision recorded in the draft

- mix15/C remains the proposal, selected by the as-written rule. It is also feasible under both post-hoc readings. No new candidate was added.
- The draft makes three changes:
  - step 0.3 now fixes one consistent equilibrium model with no 0.87 constant, fitted on measured Q1 at the supported points only (revision 2);
  - the thin margin is disclosed;
  - the alternative mix15/D is discussed in DRAFT_PROTOCOL.md Part 8. It has larger margins but leaves the labelled control no labelled depth.

## Limits shared by every reading

The edge slopes come from old critics:
- trained on the uniform recorder at stall_slope 1.5;
- evaluated at currents >= 0.7, which lay outside their training data;
- extrapolated from fits on a0 in [-0.5, 0.5].

Only Stage 1's R0 check measures the real learner.

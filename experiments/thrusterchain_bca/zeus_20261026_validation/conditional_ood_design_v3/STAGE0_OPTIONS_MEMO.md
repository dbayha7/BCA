# Stage 0 options memo: a replacement conditional region

Date: 28 October 2026. This memo supports section 0 of `DECISION_BLOCK.md` in this folder. It chooses no number.
Every region below is written with named parameters for you to set. It was produced read-only. No repository file
other than this memo was written, no simulator or model ran, and no `.npz` array was decoded.

**Throughout this memo:**
- "First thruster" means command `a0`, the thruster between floats 1 and 2.
- "Load" `g` is current plus the first wave coordinate.
- **Memo tally:** a figure computed from the saved development bank in `docs/thrusterchain/site/data/results.json`.
  That bank holds 128 states reached by the saved penalty-off 20k actor, with 9 candidate commands each. A tally
  becomes a stage 0 entry only after you approve that read.
- **Model:** a figure from a fitted or linear-quadratic approximation, not a measured Q^pi.

## Later update, 28 October 2026, 13:35 EDT: a replacement passed

You approved one replacement, the central-quartile gate in `../conditional_ood_gate_central_quartile_v1/`.
- **Region:** the first thruster's commands within 0.25 of zero, at load at least 0.4. It is candidate A's shape, with a
  width set as the central quartile of the uniform recorder's command range.
- **Result:** it passed its development design gate.
  - **Support:** 22 of 23 high-load and 40 of 41 low-load anchors passed.
  - **Value:** 87% of high-load anchors had a gap of at least 1 unit.
  - **Exposure:** the free actor entered the hole on 74.5% of 773 high-load steps, in 14 of 16 episodes.
  - **Coverage:** 9,692 hole rows across 676 episodes.
- **What the pass permits:** only proposing a separately reviewed training study, not BCA efficacy.

A parallel draft, `../conditional_ood_gate_zero_slab_v1/`, was never frozen and is superseded. It used the same anchors.

## Update, 28 October 2026, after the gate ran

The corrected gate ran as attempt 2 in `../conditional_ood_gate_v1_retry_20261028/`. It completed, and the old band
failed. The record is `SHELVED_V2_BAND_20261028.md` in this folder. Four measured results change this memo.

1. **Local support failed first on a row count that does not depend on the region.**
   - **Required versus found:** no anchor had the required 32 same-side rows within RMS distance 0.5. The best had 17.
   - **Forecast versus measured:** the analytic receipt forecast 1.4 to 2.3 such rows on average, and 1.7 were measured.
     The median distance to the 64th neighbor was 0.75, or 0.79 at high load. That is slightly above the receipt's
     forecast of about 0.72 to 0.73, so real neighborhoods are a little sparser than the model.
   - **Consequence:** any region judged by radius 0.5 would fail the same way. A replacement gate needs a different
     support rule, fixed by a written derivation before its data is read.
   - **Not settled:** whether the old band would have met its own band-row criteria under an adequate radius was not
     tested.
2. **The saved actor never entered the band in free play: 0 of 927 high-load steps.**
   - **Off-trajectory entries:** its own command did fall inside the band at 2 of the 29 recorded high-load anchor
     states. So it can command there, but its own trajectories do not go there.
   - **Saved evidence:** the 16 free episodes hold the actor's commands at every step. That is direct item 0.4 evidence
     for any candidate.
3. **The value test passed at its minimum.** It needed 8 of 29 high-load anchors and got exactly 8, with a median gap of
   0.70. These are exact returns under this one actor's continuation.
4. **The evaluator ran to completion and passed its built-in checks.** Those checks were the exact replay of every
   recorded prefix observation, and dynamics-to-environment parity at four times on one path.
   - **What a candidate A gate can reuse:** the replay, parity and continuation code.
   - **What it cannot reuse:** the candidate bank, the free-rollout hole test, the region constants in `geometry.py` and
     the contract. Those are hard-coded around −0.8 and need a new, separately frozen version.

Several statements below describe the situation before attempt 2:
- that the gate "failed on two bugs" (only one stopped it; the other hid its message);
- that a corrected gate "would fill item 0.8";
- that "no practical value difference was shown";
- that "the saved actor never chose those commands".

Attempt 2 passed the value test and found the actor's command inside the band at 2 recorded anchors.

## What is being replaced

The v2 region withheld the first thruster's hardest upstream push at the stall point: `a0` from −0.85 to −0.75 at
states with load `g >= 0.4`. The v3 addendum shelved it for four reasons:
- the saved actor never chose those commands;
- no practical value difference was shown;
- the load threshold is a median split;
- its support check could not pass.

## Bottom line

1. **No candidate can pass stage 0 from saved evidence today.**
   - **What item 0.8 needs:** a Q^pi comparison that changes one command while the other three stay fixed.
   - **What the saved bank has:** no such pairs, so every separation figure below is a model fit.
   - **The only exact identities are too small.** The stall-twin gap is at most 0.225 return units, and at the last
     step only the energy cost differs.
2. **The calculation that would supply item 0.8 already exists, but it failed on two software bugs before reading any
   data.** That is the "conditional OOD design gate v1", prepared by another agent this morning and launched at 11:00.
   Details are in the next section.
3. **The strongest replacement shape is a slab centered on zero thrust for the first thruster, at high load.** Of 11
   candidates from three independent angles, it is the only one rated viable with changes. The other ten are not
   viable today.
4. **One corrected gate could fill item 0.8 for every candidate at once.** It would sweep the first thruster's command
   across its range at fixed anchor states, with the other three commands held at the actor's values. That needs a new
   gate version and your manual launch.

## The gate that ran this morning

`../conditional_ood_gate_v1/` screens the original v2 band. It runs the frozen 20k actor from 64 development anchors
and evaluates six first-thruster commands with the other three held at the actor's values. It then continues the actor
to step 100 and measures free-rollout entry over 16 episodes. Its ceiling is 900 seconds and 28,104 simulator steps.

Attempt 1 ran at 15:00 UTC and stopped after 37 seconds, in its source-identity phase. It made zero simulator steps,
decoded no data and reached no decision. The cause is two code defects, not a scientific result:

- **The real failure.** `conditional_ood_gate_v1/reference.py` line 46 reads `core.canonical.__file__`. The pinned loader
  (`core_v4/thrusterchain/canonical/__init__.py` lines 88–110) builds that module in memory with `types.ModuleType`, so it
  has no `__file__`. The access raises an AttributeError.
- **Why the message was lost.** `conditional_ood_gate_v1/worker.py` line 289 calls `type(exc).__name()` instead of
  `type(exc).__name__`. The crash handler itself crashed, which is the only error the run recorded.

The gate's 32 tests used synthetic arrays and toy dynamics, so neither path ran before launch. The gate's rules forbid
an automatic retry and editing its prepared files. A fix therefore needs a new, labeled gate version and your launch.
Attempt 1 stays on record as a software failure.

## The candidates

The three angles were where the actor goes, where values differ, and what the plant makes consequential. Each
candidate was checked by a separate refuter.

| Candidate | Region | Verdict | Deciding reason |
|---|---|---|---|
| **A. Zero-centered slab, high load** | `abs(a0) <= h_A` at `g >= g_A` | **Viable with changes** | The actor operates there, the behavior-cloning target is unchanged, and value is steep in `a0`. It still needs item 0.8. |
| R1. Zero-centered slab, high drive | `abs(a0) < b1` where the next-step speed of floats 1 and 2 is at least `s1` | Not viable today | Same strengths as A, but the drive split is a median split again. It still needs item 0.8. |
| C1. Counter slab, high load | `a0` in `[-w_out, -w_in)` at `g >= g_s` | Not viable today | Strong reach: 52 of 59 high-load states. It shifts the behavior-cloning target, and its range contains every negative command of the one saved actor, which risks fitting the region to that actor. |
| C2. Counter slab, high drift | as C1, on a drift condition | Not viable | The drift condition depends on the actor's own past commands, so reach is not independent evidence. |
| B. Full counter tail | `a0 <= -c_B` at high load | Not viable | The actor's lowest command is −0.483. Only a shallow tail is reachable, where value evidence is model-only. |
| R2. Deep counter and stall | `a0` in `[-1, -b2]` at high drive | Not viable | The actor never commands below −0.483. |
| C. Load-assisting half | `a0 >= c_C` at high load | Not viable | Reached at only 7 of 59 high-load states, where the separation is smallest. |
| C3. Assist slab, light load | `a0` in `(w_in, w_out]` at `g < g_l` | Not viable | It reverses the behavior-cloning shift, and its value gaps are small. |
| R4. Shallow assist, high drive | `0 <= a0 < b4` at high drive | Not viable | Weak reach: 13 of 70 states. |
| C4. Counter slab on the second thruster | `a1` in `[-w_out, -w_in)` at wave crests | Not viable | Its value slope is about 40% of the first thruster's. |
| R3. Last steps of the episode | `t >= 100 - H3` | Not viable | Value-flat: at the final step only the energy cost differs, at most 0.3 return units. |

## Candidate A in detail

**Region.**
- **State condition:** load `g >= g_A`.
- **Command condition:** `abs(a0) <= h_A` on the executed float32 command, promoted to float64. The other three commands
  are unrestricted.
- **Alternative bands:** the equal-width neighbors `[-3h_A, -h_A)` and `(h_A, 3h_A]`.
- **Far set:** `a0 = -k`, plus a load-assisting band near +0.6 to +k.
- **Knee constraint:** keep `3h_A < k`, so no alternative band reaches the stall knee.

**Why it fits.**
- **Mechanics (exact on the frozen plant).** The first thruster is the only actuator on float 1, the slowest float. Zero
  thrust is the point where the command switches from opposing the water to assisting it. Value falls nearly linearly
  in `a0` at high load. Within states, a linear fit gives a median R² of 0.985 to 0.991 (memo tally plus model).
- **Behavior-cloning neutrality.** The recorder's commands are uniform on [−1, 1]. A hole symmetric about zero leaves
  the mean logged command at in-region states exactly where it was. So the behavior-cloning pull does not move the
  actor out of the hole by itself. The one-sided candidates shift that mean by about +0.08 at width 0.5 (model).
- **Policy reach (memo tally).** At the 59 high-load bank states, the saved actor's `a0` ran from −0.483 to +0.252. It
  fell within `abs(a0) <= 0.15` at 21 of them and within `0.3` at 47. Its median sits on the load-opposing side, from
  −0.09 to −0.29 across load depths, so the actor is inside the slab but off its center.
- **Value structure (model, other commands fixed).** The fitted slope in first-thruster thrust is negative at all 59
  high-load states, about −5 return units per unit thrust. Adjacent band centers differ by roughly 10 × `h_A` units. The
  same fit gives only +0.18 across the v2 band's edges.

  | Pair | Fitted gap (return units) |
  |---|---|
  | Q(−0.2) − Q(0) | about +1.0 |
  | Q(0) − Q(+0.4) | about +2.1 |
  | Q(−k) − Q(0) | about +3.9 |

- **Existing critic errors (memo tally plus model).** The full-data critic already flattens this slope, to about −3.5
  against about −5. It orders the 9 candidates correctly in only 74% to 82% of within-state pairs.

**Risks.**
- **The critic may simply interpolate.** Labels remain on both sides of the slab, and value is near-linear there. A
  native critic may bridge the gap, so the error gates may fail. That outcome would be inconclusive, not a BCA result.
- **Every figure comes from one weak actor.** Its return is −74.2, against about −69.5 for continued native actors and
  about −67 for BCA. Better continuations recover more from a poor first command, which shrinks single-step gaps.
- **The hole is larger.** It would remove `76,800 × P(g >= g_A) × h_A` rows, for example 3,840 at `g_A = 0.4` and
  `h_A = 0.1`. That is 2 to 4 times the v2 band.
- **The width must not be chosen from the actor counts above.** The counts are now known, and using them would fit the
  region to one saved actor.
- **A median split at 0.4 carries over the v3 objection.** A deeper threshold cuts the actor-visited states: at
  `g >= 0.7`, only 10 of 16 states fall within `abs(a0) <= 0.3`, and none within 0.15.

**Numbers for you to set.**
- `g_A`, the load threshold.
- `h_A`, the half-width.
- The far-set commands.
- The boundary convention for item 0.5.

Each needs a rule that uses no outcomes. One proposed rule is the smallest `h_A` whose predicted neighbor gap meets your
item 7.1 practical size. It would rely on the saved actor's value fit, so you would need to judge whether that counts as
outcome-independent.

## Facts that apply to every candidate

These are memo tallies from the one saved penalty-off 20k actor.

- **Where the saved actor commands.** It never commands past the knee. Its commands track the logged mean plus a small
  multiple of the critic's action slope, which is how TD3+BC balances its two terms. So a hole is relevant only inside
  the narrow ranges below.

  | Thruster | Command range |
  |---|---|
  | `a0` | −0.483 to +0.521 |
  | `a1` | −0.492 to +0.377 |
  | `a2` | −0.345 to +0.314 |
  | `a3` | −0.703 to +0.376 |

- **The first thruster matters most.** Its fitted value slope, about −3.6 per unit thrust, is about 3, 5 and 23 times
  those of the second, third and fourth thrusters, at −1.07, −0.79 and −0.16 (model).
- **Bellman targets query the hole.** Targets use the target actor's command plus clipped noise. A hole that contains
  the actor's own commands is queried by about 13% to 26% of all targets. The v2 band was queried by about 0.3%.
- **Value separation depends on the continuation.** Under the optimal linear-quadratic continuation, the curvature in
  `a0` is only 0.672 per unit squared (model). Gaps measured under one actor do not carry over to a better actor.

## Decisions for you

This section was revised after attempt 2. The old band is shelved, and the corrected gate has already run. Decide in
this order, so no number is set after seeing data that bears on it.

1. **Which region shape to carry into section 0:** A, or another row above.
2. **The setting rule for every number of that candidate.** For A, these are `g_A`, `h_A`, the far-set commands and the
   boundary convention. Write how each is set before any of the reads in decision 4.
3. **Which support rule replaces radius 0.5,** written before the next gate reads data. Two options:
   - **A radius derived from the analytic state model.** It must target the pass criterion actually used. For example,
     choose the model radius at which at least 80% of anchors would have 32 rows from 8 episodes. The receipt's 0.66 to
     0.73 gives the expected count for a typical anchor, so it would pass only about half the anchors. Attempt 2's
     measured distances, 0.58 to 1.00, are now known. Record that they were known when the rule was written.
   - **Low-dimensional cells:** ten-step bin × load bins × current bins, labeled as occupancy only, never as full-state
     support.
4. **Whether to approve two reads of saved arrays,** used only to fill stage 0 evidence items:
   - the 1,600 free-actor steps from attempt 2, for item 0.4;
   - the 128-state development bank, for the memo tallies above.

   Neither read may revise `g_A`, `h_A`, the far-set commands, the boundary convention or any other number. The bank
   tallies in this memo were seen before any number was set, and that is recorded here.
5. **Whether to prepare a gate for candidate A.**
   - **Reused unchanged:** attempt 2's replay, parity and continuation code.
   - **New:** a separately frozen candidate bank, hole test, region constants and contract. Six commands would sit around
     zero with the other three commands held at the actor's own.
   - **Support:** the rule from decision 3.
   - **Numbers and launch:** every number left blank for you to set, and it runs only on your manual launch.

Attempt 2's 64 anchor episodes have now been used. Its 16 free episodes started from the seeds of the first 16 of them.
A new gate needs anchors from other episodes, and none of these anchors can later serve as test probes.

## Provenance

- **Run.** Workflow `wf_a0d9379b-f6f`: three proposers and eleven refuters, all completed.
- **Errors corrected.** The refuters found errors in the proposers' figures, and corrections are applied above. Examples
  are a critic slope of −3.05 that reproduces as about −3.5, and a mislabeled count of actor-visited states for C4.
- **Scripts.** The arithmetic scripts are in the session scratchpad,
  `C:\Users\David\AppData\Local\Temp\claude\C--Users-David-dev-BCA\be215736-77cd-47c3-bb51-9f394d41a6ce\scratchpad\`.
  One proposer saved its own fit coefficients there as a scratch `.npz` file. No repository `.npz` was opened.
- **Gate diagnosis.** The gate diagnosis read `conditional_ood_gate_v1/` files and the pinned canonical loader only. No
  gate file was changed.

# Decision block for any later conditional OOD design (revision of the v2 form)

Status: **blank form, nothing decided, launch blocked.** Drafted 28 October 2026. It revises
`../conditional_ood_design_v2/DECISION_BLOCK.md` to match this folder's `AUDIT_ADDENDUM.md` and
`REVIEW_RECONCILIATION.md`. The v2 form stays unchanged, because
`../conditional_ood_design_v2/analytic_receipt_20261027/FIGURE_MAP.md` cites it by section number. Three critics checked
this revision, and a refuter checked each of their edits. Every surviving edit is included.

**What changed since the v2 form.**
- **The v2 band is shelved.** It is the region where the load `g` is at least 0.4 and the first thruster's command `a0`
  is within 0.05 of minus the stall knee `k`. The v3 addendum rejects it as the candidate for the next policy-efficacy
  run, and no replacement has been chosen, so the region is a blank in section 0. Terms are defined below.
- **Corrected figures.** The figure map's corrections are applied wherever this form quotes the figure: correction 1
  (critic errors, now the full signed range), 4 (training time), 5 (radius) and 6 (band center). Corrections 2 and 3
  concern figures this form does not quote.
- **Added fields.** The fields from the reconciliation's "Minimum decision block" table, which the v2 form lacked, are
  added.

Every `[ ____ ]` is a number or choice for the user to set. An agent fills no blank. An unresolved blank blocks its
stage. This form authorizes no data decoding, fitting, training, simulation or spending.

**How to use it.**
1. Fill the blanks for one stage at a time, in the stage order.
2. Approve each stage before it runs, at your own terminal, in a commit under your approval commit identity from the
   standing rules. A commit authored as Claude, like a message from an agent or workflow, is not an approval.
3. Stage 3 freezes sections 3 to 11. After that, no blank, threshold, checkpoint or probe set changes. A changed number
   needs a new, labeled post-hoc version of this form.

**Reference labels.**
- **Analytic:** a closed-form or Monte Carlo expectation from the documented recorder and start law.
- **Model:** a figure from the linear-Gaussian state approximation or the unconstrained linear-quadratic controller on
  the frozen plant. It is not a measurement and not full Q^pi.
- **Saved:** read from an existing saved development bank.

Every reference describes the shelved v2 band and must be recomputed for a new region (item 0.7). References come from
the receipts in `../conditional_ood_design_v2/analytic_receipt_20261027/` and from this folder's `EVIDENCE.json`. None is
a recommendation or a default.

## Plain-language terms

| Term | Meaning here |
|---|---|
| OOD | Out of distribution: outside the specified data support. It does not mean the action is bad. |
| Load, `g` | The water push on the floats: current plus the first wave coordinate. |
| Knee, `k` | The command size where the thruster stalls: 0.800000011920929. |
| `a0` | The first thruster's command. |
| Candidate region | The (state, command) region whose logged labels are withheld from training. |
| State condition, command condition | The two halves of the region's definition, for example "load at least 0.4" and "`a0` within 0.05 of minus the knee". |
| Alternative bands, far set | Comparison commands at the same states: bands next to the region's command condition, and commands far from it. |
| Hole | The candidate region once its logged rows are withheld. |
| Data condition | One of three training pools: full data (all 768 training episodes), hole, or random removal. |
| Random removal | The training pool with the same number of rows removed in each (episode, ten-step bin, side of the state condition) stratum as the hole, drawn uniformly at random instead of by the region test. |
| Q^pi reference, `v` | The discounted return of taking one command, then following the snapshot's own actor to step 100. |
| Raw score, `q` | The first critic's value (Q1) for that command. |
| Adjusted score, `z` | The raw score minus kappa times `U`. Kappa is BCA's penalty multiplier, 3 here. `U` is BCA's calibrated nonnegative residual half-width, which the code calls width. It is not a verified bound on the Q^pi error. |
| Absolute error, MAE | `abs(q − v)` or `abs(z − v)` in return units; MAE is its mean. |
| Regret | The best candidate's `v` minus the chosen candidate's `v`, within one fixed candidate bank. |
| RMS distance | The square root of the mean squared coordinate difference. Over 13 coordinates, multiply by about 3.6 to get the Euclidean distance. |
| Standardized mean difference | A difference in means, in standard-deviation units. |
| Native | Unmodified TD3+BC from the Unifloral library. TD3 is an actor-critic learner, and its BC (behavior cloning) term keeps the actor near logged commands. |
| BCA | The original BCA actor penalty at kappa 3. |
| Distance control | The existing simple penalty, strength-matched once on training probes at the 10k fork. |
| Arm | One training method: native, BCA or distance control. |
| Checkpoint, 10k, 20k | A training point named by its critic-update count: 10k is 10,000 updates. At 10k the shared native warmup forks into the three arms. |
| Snapshot | A saved copy of an arm's actor and critics at one checkpoint. |
| Probe state | A fixed test state at which every arm scores the same candidate commands. |
| Forced probe | A candidate command imposed at a probe state, as opposed to a command the actor chooses in a free rollout. |
| Test family | One independently seeded set of test episodes and their probe states. |
| Recorded current, strong current | Starts whose current is drawn from 0.1 to 0.7, as in training, or from 0.7 to 1.1, never seen in training. |
| Readiness | BCA has a valid, finite calibrated reference from its first calibration at 10k onward. A missing or infinite reference is a refused case. It is counted with its ID and never replaced by a zero penalty. |
| RNG, RNG address | Random number generator. Its address is the fixed seed and stream label that let a draw be repeated exactly. |
| Trained seed | One independent training initialization. It is the unit of replication. |
| Kill rule | A result, written before a stage runs, that shelves the stage or the design. |
| Inconclusive | The stage could not test the question. It is not evidence for or against BCA. |

## Stage order

| Order | Stage | Reads or runs | Approved before it runs |
|---|---|---|---|
| 0 | Candidate proposal | Saved receipts only, such as `EVIDENCE.json` and the analytic receipt. A new calculation for 0.7, or a read of saved arrays for 0.4 or 0.8, needs its own approval first. | Section 0 |
| 1 | Seed, role and evidence ledger | Local JSON receipts only | Section 1 |
| 2 | Synthetic fixtures | Hand-built cases, plus the bounded simulator steps in 2.1 for the reference-time case | Section 2 |
| 3 | Freeze sections 3 to 11 | Nothing | Sections 3 to 11 |
| 4 | Geometry-only coverage audit | Training and tuning-validation arrays only, no model | A run approval; sections 4 and 5 as frozen |
| 5 | Calibration and random-removal row plans | The same arrays, no model | A run approval; section 6 as frozen |
| 6 | Budget sheet | Nothing | The budget sheet, checked against section 10 as frozen |
| 7 | Training and evaluation | Models and simulator, within the ceiling | Its own run authorization |
| 8 | Unsealing and analysis | Saved outputs, in the order of section 8 | Nothing new |

**Holds and reservations.** Earlier holds and reservations, including `../titan_handoff_20261027/ATTEMPT_HOLD.json`, stay
intact. They are never cleared, reset or reused for this design. A new run needs its own immutable protocol, reservation
and accounting.

## 0. Candidate region (stage 0)

The v3 addendum requires any replacement to be justified by mechanics and pre-intervention coverage. It also requires
practical value exposure and policy relevance, both established independently of BCA outcomes.

| Item | Entry |
|---|---|
| 0.1 Region definition: state condition and command condition | `[ ____ ]` |
| 0.2 Alternative bands and far set, with the other three commands held fixed at each probe state | `[ ____ ]` |
| 0.3 Mechanical rationale, from the plant equations only | `[ ____ ]` |
| 0.4 Policy relevance: why a trained actor is expected to choose commands in the region, citing saved actor evidence | `[ ____ ]` |
| 0.5 Boundary convention. Load: summed as a float32 observation promoted to float64, as a float32 sum, or from the carried float64 state. Command: how the float32 executed value is tested. | `[ ____ ]` |
| 0.6 Width or severity, one level only for the first comparison | `[ ____ ]` |
| 0.7 Recomputed references, one for each analytic or model reference below: the float32 edges in 2.3, items 4.2 to 4.7, the section 5 radius figures, the section 6 calibration-row figure and 7.1. The saved references in 0.9, 7.3 and 7.5 need saved arrays and are marked not recomputed unless approved. | `[ ____ ]` |
| 0.8 Value separation: the spread of Q^pi, in return units, between region commands and the 0.2 alternatives at in-region states, with the other commands held fixed. State its source: a full-return argument with its continuation, state and time assumptions, or a valid saved comparison with the other commands held fixed. One-step physics bounds and linear-quadratic model figures do not count. | `[ ____ ]` return units, from `[ ____ ]` |

**0.9 Kill rule.** The proposal is shelved before stage 1 in either of these cases:
- 0.4 cannot cite actor evidence that reaches the region;
- 0.8 cites neither a full-return argument nor a valid saved fixed-other-commands comparison.

The region is never tuned to intersect one saved actor's commands.

**Reference (saved).** The single saved penalty-off 20k actor never reached the v2 band:
- 0 of its 59 high-load states and 0 of its 128 states in all fell in the band;
- its first command ranged from −0.483 to +0.521;
- its nearest command was 0.32 from the band center.

The saved bank also has no same-state pairs with the other three commands fixed. So it cannot supply 0.8 for the v2 band.

## 1. Seed, role and evidence ledger (stage 1)

**1.1 Ledger scope.** Ranges are collected from every local receipt. The standing reservations of seeds 5 to 9 and 10 to
14 count as known ranges. The ledger is labeled "Zeus-visible only; Titan collisions unverified" until the Titan
manifests are imported.

**1.2 Episode roles.** These are fixed; no blank.

| Episodes | Role |
|---|---|
| 768 training | Development data, already inspected. |
| 128 calibration | The first 128 tuning-validation groups sorted by group ID, with times from seed 2026102362. Development data, reused. |
| Other 128 tuning-validation | Development data, already consumed as validation in the earlier one-sided BCA study (`bca_one_sided_v1`). |
| Selection and final calibration | Sealed. |

No result from this design is confirmatory.

**1.3 Evidence imports.** Status of each item and what it blocks:

| Item | Status | Blocks |
|---|---|---|
| Stall-boundary or fair assay | Reported completed on Titan, locally unverified; version equivalence unresolved | `[ ____ ]` |
| Explicit OOD diagnostic | Reported on Titan, absent here | `[ ____ ]` |
| Direct held-out BCA | Reported on Titan, absent here | `[ ____ ]` |
| Timeout recovery | Reported on Titan, absent here | `[ ____ ]` |
| Adaptive-signal recovery | Reported on Titan, absent here | `[ ____ ]` |

Imports are not rerun to fill a transfer gap.

**1.4 Kill rule.** Freezing any new seed is blocked if any of these happens:
- a proposed range overlaps a known range;
- any episode's role is unresolved, or any episode holds two roles;
- an import marked as blocking is still missing.

## 2. Fixtures (stage 2)

**2.1 Scope ceiling.** `[ ____ ]` hand-built cases at most, with no dataset or model. The reference-time case may use
at most `[ ____ ]` simulator transitions. Nothing else in stage 2 uses the simulator.

**2.2 Kill rule.** The work stops on any of these:
- a disagreement between the two independent mask implementations;
- a disagreement with a hand label;
- a failed accounting identity;
- a failed reference-time case.

Each failure is recorded. The specification is fixed. If a failure is traced to a fixture defect, the failed result is
kept, and stage 2 reruns only with the corrected fixture under a new, labeled version of this form approved at your
terminal.

**2.3 Required cases.** These are fixed; no blank.
- **Load threshold:** a load exactly at the region's threshold.
- **Load precision:** a load where the summing conventions in item 0.5 fall on opposite sides of the threshold.
- **Command boundary:** commands exactly at the region's edges as synthetic float64 requests, plus the adjacent float32
  executed values. For the v2 band, those values are −0.84999996 and −0.75000006 (inside) and −0.85000002 and −0.75
  (outside).
- **Control cell:** region commands at states outside the state condition, which must stay eligible.
- **Kappa zero:** gives equal raw and adjusted errors.
- **No sign flip:** subtracting a nonnegative `U` never turns underestimation into overestimation.
- **Underestimation grows:** an underestimated score's absolute error grows by exactly kappa times `U`.
- **Reference time:** the reference evaluator must match environment-driven sequences, with matching states, rewards
  and observations, at an initial, an interior and a final time. The canonical reset always sets time to zero. A later
  time is reached by replaying the saved prefix or by the direct-dynamics continuation, and each route used gets its
  own case.

## 3. Primary comparison (fixed in stage 3, read in stage 8)

One comparison decides efficacy. Every other outcome is secondary or descriptive.

| Item | Choice |
|---|---|
| 3.1 Contrast | `[ BCA minus native / other: ____ ]`, paired within each trained seed |
| 3.2 Data condition | `[ hole / interaction: the 3.1 contrast in the hole condition minus the same contrast under random removal ]` |
| 3.3 Probe stratum | `[ ____ ]`, for example in-region probes under recorded current |
| 3.4 Checkpoint | `[ 20k, the v2 proposal / other: ____ ]` |
| 3.5 BCA's score in the contrast | `[ raw Q1 / adjusted Q1 − 3U ]`. Each arm is scored against its own actor's `v`, so this compares separately trained critics. It is not a correction against one common target. |
| 3.6 Statistic | `[ decrease in MAE: the comparison arm's MAE minus BCA's MAE, so a positive value favors BCA, using the aggregation weights in ../conditional_ood_design_v2/DRAFT_PROTOCOL.md line 146 / other: ____ ]` |
| 3.7 Practical effect size | `[ ____ ]` return units |
| 3.8 Trained seeds S | `[ ____ ]` |
| 3.9 Gates that apply to the primary | `[ every gate in section 7 / 7.1, 7.2, 7.3 and 7.4, with 7.5 governing only the return claim as in 7.6 ]` |

**3.10 Readiness and learning checks.**
- **Readiness.** An invalid, unavailable or infinite required BCA reference stops `[ that branch / the run ]`, as v2
  `DRAFT_PROTOCOL.md` line 128 requires. Refused or invalid rows may be at most `[ ____ ]` of the primary stratum's rows
  in every seed and arm. The statistic uses only rows valid in both arms of each pair.
- **Learning.** The native arm must improve mean paired return over its own untrained actor by more than `[ ____ ]`
  return units in every seed and data condition.

If either check fails, the run is an inadequate-readiness or inadequate-learning screen and is inconclusive for BCA.
Training is not extended.

**3.11 Decision rule.** The three results never overlap.
- **Pass.** Both checks in 3.10 passed, every gate in 3.9 passed for the primary stratum, and the decrease meets this
  rule:
  - For S up to `[ ____ ]`: the decrease exceeds the 3.7 effect size in every seed. This is an exploratory screen, not
    a p-value. With no effect, independent seeds and each seed equally likely to favor either arm, "every seed
    positive" has probability 0.5 to the power S. Requiring each seed to exceed a positive effect size can only lower
    that probability.
  - For larger S: the decrease exceeds the 3.7 effect size in at least `[ ____ ]` of S seeds, and a seed-level
    `[ sign test / paired t-test ]` gives one-sided p below `[ ____ ]`.
- **Fail.** Both checks in 3.10 passed, every gate in 3.9 passed for the primary stratum, and the decrease does not meet
  the pass rule.
- **Inconclusive.** A check in 3.10 failed, or a gate in 3.9 failed for the primary stratum. Pass and fail are then not
  read.

**3.12 Secondary outcomes and multiplicity.** All other outcomes are `[ descriptive only / tested with ____ correction ]`.
They are listed in section 11.

## 4. Coverage thresholds (fixed in stage 3, checked in stage 4)

The stage 4 audit counts rows and never moves these marks.

| Item | Threshold | Reference, v2 band (analytic) |
|---|---|---|
| 4.1 Forbidden in-region labels in any fitting, calibration or matching bank | 0, fixed | none |
| 4.2 Minimum original in-region rows in the training pool | `[ ____ ]` | about 1,920 |
| 4.3 Minimum training episodes with at least one in-region row | `[ ____ ]` of 768 | about 606 |
| 4.4 Minimum in-region rows per ten-step time bin | `[ ____ ]` | about 192 |
| 4.5 Minimum retained rows in the within-model control cell, meaning region commands at states outside the state condition | `[ ____ ]` | about 1,920 |
| 4.6 Minimum in-region rows per state-depth stratum, with strata `[ ____ ]` | `[ ____ ]` each | For load strata [0.4, 0.5), [0.5, 0.6) and at least 0.6: about 29%, 25% and 46% of high-load rows. The recorder ignores the state, so these are also the shares of the 4.2 rows. |
| 4.7 Band counts in each anchor's or probe's neighborhood or cell (section 5), counted `[ per neighborhood / pooled over the neighborhoods in a cell ]`. Minimum original region rows before removal, and minimum retained rows in the 0.2 alternative bands, counted `[ each band separately / bands pooled ]`. | Original region rows: `[ ____ ]`. Retained alternative rows: `[ ____ ]`. | 3.2 expected per band and 6.4 for two bands pooled, per 64 neighbors. A given band is empty by chance in about 4% of 64-row and 19% of 32-row neighborhoods. Requiring at least 1 row in each of three bands separately fails by chance in 11% of 64-row and 48% of 32-row neighborhoods. |

The chance that a neighborhood is labeled unknown at the chosen 4.7 minimums follows from binomial arithmetic. It is
recorded before stage 4.

**4.8 Kill rule.** If any of 4.1 to 4.6 fails, the region is shelved as written. Item 4.7 is checked per anchor or probe:
one that fails is labeled support unknown under 5.13, and the design is shelved only through 5.11. There is no redraw
and no search over regions.

## 5. Local support definition (fixed in stage 3, checked in stage 4)

| Item | Choice |
|---|---|
| 5.0 Anchor set for the stage 4 audit, which runs before any test probe exists | `[ ____ ]` anchor states meeting the state condition, drawn from `[ training / tuning-validation ]` episodes by `[ ____ ]`, with RNG address `[ ____ ]`, never by outcomes. The section 9 probes are labeled later under the same frozen rule. |
| 5.1 Method | `[ neighbor radius / low-dimensional cells ]` |
| 5.2 Normalization source | `[ original training rows, 13 physical coordinates / other: ____ ]`. It is never fed to the learner. |
| 5.3 Distance convention | `[ RMS / Euclidean ]` |
| 5.4 Neighbor pool restrictions | Same ten-step bin: `[ yes / no ]`. Same side of the state condition: `[ yes / no ]`. |
| 5.5 Anchor-episode exclusion | The whole episode of a training-derived anchor is excluded from its neighbors: `[ yes / no ]` |
| 5.6 Row cap | `[ ____ ]` neighbors |
| 5.7 Radius, or the rule that derives it from quantities fixed before stage 4, never from stage 4 distances, counts or any outcome | `[ ____ ]` |
| 5.8 Minimum for a populated neighborhood or cell | `[ ____ ]` rows from `[ ____ ]` distinct episodes |
| 5.9 Sensitivity settings | `[ ____ ]` neighbors at `[ ____ ]`, and `[ ____ ]` at `[ ____ ]` |
| 5.10 Maximum share of support labels that may change across 5.9 | `[ ____ ]` |
| 5.11 Minimum share of anchors in stage 4, and of in-region probes later, that must be populated and pass 4.7 | `[ ____ ]` |
| 5.12 Cross-boundary reporting | Fixed, no blank: report retained region-command rows just across the state boundary near each anchor or probe. |

**Cells.** If cells are used, fill these instead of 5.6, 5.7 and 5.9; 5.8 applies as written. Cells are ten-step bin ×
`[ ____ ]` bins of width `[ ____ ]` × `[ ____ ]` bins of width `[ ____ ]`. A cell label describes low-dimensional
occupancy and is never relabeled as full-state support.

**Reference, v2 band (model; RMS over 13 standardized coordinates).**
- **Same ten-step bin and load side, radius 0.5:** about 1.4 to 2.3 of the roughly 3,840 rows in that bin and side are
  within reach.
- **Same bin and side, 32 to 64 rows:** a radius of about 0.66 to 0.73 is needed.
- **Same time step, other episodes, 8 distinct episodes:** about 0.67 to 0.71 over the whole pool, or about 0.72 to 0.77
  after halving the pool for one load side.

v3 treats these as a feasibility warning, not a measured count. The radius is never scanned.

The recorder ignores the state, so a band count inside a state neighborhood is binomial. An empty band in a populated
neighborhood is mostly chance.

**5.13 Kill rule.**
- **Probes and anchors.** One that fails 5.8 or 4.7, or whose label changes across 5.9, is labeled support unknown. The
  conditional-hole claim is not made for it.
- **Unstable labels.** If the share of changed labels exceeds the 5.10 maximum,
  `[ the design is shelved / every probe is labeled support unknown ]`.
- **Too few supported.** If 5.11 fails, the design is shelved as unable to show a conditional hole.

The radius is never relaxed to rescue labels. Strong-current probes are outside recorded current support by definition.

## 6. Row plans (fixed in stage 3, run in stage 5)

| Item | Choice |
|---|---|
| 6.1 Calibration row selector | `[ existing uniform eligible-row selector with fixed RNG (support pilot design.py lines 43–56) / next eligible step cyclically / other: ____ ]`. These rules select times with different probabilities. |
| 6.2 Selector RNG address | `[ ____ ]` |
| 6.3 Calibration rows across data conditions | `[ common to all conditions / specific to each condition ]` |
| 6.4 Random-removal quotas | Per (episode, ten-step bin, side of the state condition) stratum, with RNG address `[ ____ ]` |
| 6.5 Maximum standardized mean difference on any state coordinate, hole pool versus random pool | `[ ____ ]` |

Reward and successor differences are consequences of removing commands. They are reported with no bound and never
matched away.

**Reference, v2 band (analytic).** About 3.2 of the 128 existing calibration rows would fall in the hole.

**6.6 Kill rule.**
- **No eligible row.** A calibration episode with no eligible row shelves the design as written, with no substitution.
- **State imbalance.** An imbalance above the 6.5 bound `[ shelves the region as written / is reported as an imbalanced
  control ]`. The reconciliation's geometry stopping rule says to shelve. v2 said to report.
- **No redraws.** No plan is redrawn.

## 7. Exposure gates (fixed in stage 3, read first in stage 8)

These are read on native outputs before any BCA or other distance-control outcome is unsealed. The distance-control
actor's entry share is also read here, but only if 7.5 counts it.

| Item | Threshold | Reference, v2 band |
|---|---|---|
| 7.1 Value gap: minimum spread of native `v` across the candidate families at an in-region probe state, with the other commands held fixed | `[ ____ ]` return units | Model, 100 high-load starts, first commands −0.9, −0.85, −0.8, −0.75 and −0.7. Mean spread 0.86 under a load-opposing rule on the first thruster, with a largest single-state spread of 2.77. Mean spread 1.47 under zero later commands, with a largest single-state spread of 4.13. Adding the far command 0 raises the means to 4.81 and 8.33. Across the band alone, the one-step reward spread is at most 0.048. No full-return bound is verified. |
| 7.2 Value-gap prevalence: minimum share of in-region probe states meeting 7.1 | `[ ____ ]` | none |
| 7.3 Error opportunity: minimum native MAE on in-region probes | `[ ____ ]` return units | Saved: full-data penalty-off 20k critic, 25 in-band high-load candidates from 21 states. Signed errors ranged from −6.5 to +82.0, with MAE 47.5. Band membership was tested on the saved float32 actions without a float64 cast, so it is not certified against the exact v2 boundary rule. |
| 7.4 Removal effect on native: minimum increase in in-region MAE, hole minus full data | `[ ____ ]` return units | none |
| 7.5 Free-policy entry: among steps at states meeting the state condition, the minimum share where the executed command of `[ the native actor / each of the native and distance-control actors / either actor ]` meets the command condition, widened by a tolerance of `[ ____ ]` | Steps: `[ ____ ]`. Episodes: `[ at least ____ of episodes enter the region at least once / the step share holds in at least ____ of episodes ]`. | Saved actor: 0 of 59 high-load bank states, nearest command 0.32 from the band center |

A large shared Q bias can coexist with correct ranking. So 7.3 never replaces 7.1 and 7.2.

**7.6 Kill rules.**
- **Accuracy and decision claims.** If 7.1, 7.2, 7.3 or 7.4 fails, these claims are inconclusive for the mechanism. BCA
  outcomes for them stay sealed.
- **Return claim.** If 7.5 fails, the whole-policy return claim is untested for the mechanism. Forced-probe accuracy may
  still be reported, labeled as forced.
- **BCA's own entry share.** It is reported in every case. Avoidance is itself a result.

## 8. Unsealing order (fixed in stage 3)

During stage 7, only progress, accounting and failure records are read. Every outcome file stays sealed until its step.

1. **Checks.** Readiness and the learning check (3.10), with refusal and invalid counts.
2. **Gates.** The exposure gates (section 7): native outputs, plus the distance-control entry share only if 7.5 counts
   it.
3. **Primary.** The primary comparison (section 3), only if step 1 passed and the 3.9 gates passed for its stratum.
4. **Secondary.** Secondary outcomes (section 11), following 7.6:
   - if 7.1 to 7.4 failed, BCA outcomes for accuracy and decisions stay sealed;
   - if 7.5 failed, return outcomes are labeled untested for the mechanism, and forced-probe accuracy is labeled forced.

## 9. Seeds and evaluation counts (fixed in stage 3)

| Item | Value |
|---|---|
| 9.1 Trained seeds | S from 3.8, no separate blank |
| 9.2 Training seed values, from a range disjoint from the ledger | `[ ____ ]` |
| 9.3 Test-family seed range | `[ ____ ]` |
| 9.4 Test families | `[ separate per trained seed / shared across seeds ]` |
| 9.5 Probe states per family, recorded current | `[ ____ ]` at times `[ ____ ]` |
| 9.6 Probe states per family, strong current | `[ ____ ]` |
| 9.7 Behavior that generates the probe states | `[ original uniform recorder from independent starts / other method-neutral behavior: ____ ]`. Never a trained actor. |
| 9.8 Route to the probe state and time for the reference `v` | `[ replay of the authenticated saved prefix / existing authenticated direct-dynamics continuation ]`, under the contract in `AUDIT_ADDENDUM.md`, "Exact reference-state/time contract" |
| 9.9 Candidates per probe state, in this order: region commands, each alternative band, far set, plus the actor's own command | `[ ____ ]`, `[ ____ ]`, `[ ____ ]`, plus 1 |
| 9.10 Whole-policy return episodes per policy, recorded current | `[ ____ ]` |
| 9.11 Whole-policy return episodes per policy, strong current | `[ ____ ]` |
| 9.12 Seeds 10 to 14, which the standing rules reserve for a new version | `[ stay reserved and excluded from 9.2 and 9.3 / used by this design as that new version ]`. Seeds 5 to 9, reserved for versions 6 and 7, are excluded either way. |

More test episodes do not add trained replicas.

## 10. Runtime ceiling (fixed in stage 3, checked in stage 6)

**10.0 Design shape and schedule.** Choose one:
- **The v2 proposal:**
  - C = 3 data conditions: full, hole and random removal;
  - A = 3 branches: native, BCA and distance control;
  - a shared native warmup to 10k, then the branches run to 20k;
  - calibration refreshes at 10k shared, plus 15k and 20k in each branch;
  - actor snapshots at 0 and 10k shared, plus 11k, 15k and 20k in each branch.
- **Other:** `[ ____ ]`, with its own formulas `[ ____ ]`.

Section 11's learning curves use these snapshots. For the v2 proposal, the counts follow from S. They are arithmetic,
not separate choices.

| Quantity | Formula | With the chosen S |
|---|---|---|
| Critic updates | 120,000 × S | `[ ____ ]` |
| Actor updates | 60,000 × S | `[ ____ ]` |
| Auxiliary updates | 120,000 × S | `[ ____ ]` |
| Calibration refreshes | 21 × S | `[ ____ ]` |
| Unique actor snapshots | 33 × S | `[ ____ ]` |

**Training-time projection.** This is an extrapolation, not a measured runtime. It comes from two Zeus 10k warmups of
637.6 s and 708.3 s. Training would take about 2.1 to 2.4 hours per seed, or about 4.3 to 4.7 hours for two seeds. That
is 1.7 to 1.9 times the support pilot's 9,000-second global cap, counting training only. That cap also covered setup,
compilation and evaluation. Reference rollouts and state banks would come on top.

| Item | Ceiling |
|---|---|
| 10.1 Global wall-clock, everything included | `[ ____ ]` seconds |
| 10.2 Per warmup job | `[ ____ ]` seconds |
| 10.3 Per branch job | `[ ____ ]` seconds |
| 10.4 Reference rollouts, state banks and evaluation | `[ ____ ]` seconds |
| 10.5 Simulator transitions: the sum of `(100 − t)` over every candidate and snapshot, plus 100 × return episodes, plus state banks and verification | `[ ____ ]` |
| 10.6 Machine | `[ Zeus / Titan ]` |

**10.7 Kill rule.**
- **Over the ceiling.** A projection above the 10.1 ceiling needs a reviewed redesign. That redesign is recorded as a
  new, labeled version of this form and approved at your terminal. It never triggers an automatic runtime extension or
  an automatic cut in seeds or in the section 9 counts.
- **Interruptions.** An interrupted or timed-out job keeps its evidence and stops. There is no automatic restart,
  retuning or reuse of an old reservation. A failed job counts against the ceiling.

## 11. Secondary and descriptive outcomes (reported, not decided)

- **Same-critic panel:** `q` against `z` at the same checkpoint.
- **Error classes:** over, near-zero and under counts, using `tau(v)` below, and the 3 × 3 table of how each probe's
  class changes from `q` to `z`.
- **Paired change:** improved, worsened and tied counts, using `eta` below.
- **Decisions:** regret, gain, harm and choice switches within the fixed bank, read against 11.1.
- **Learning curves:** at the snapshots in 10.0.
- **Factor cells:** the four recorded-current cells in the v3 addendum, plus the strong-current cells.
- **Interaction:** the hole-minus-random interaction, if it is not the primary.

| Band or size | v2 proposal | Confirm or set |
|---|---|---|
| `tau(v)`, the near-zero error band | `1e-4 + 1e-5 × abs(v)` | `[ confirm / ____ ]` |
| `eta`, the tie band for paired change | `tau(v)` | `[ confirm / ____ ]` |
| 11.1 Practical regret or gain size | none | `[ ____ ]` return units |
| 11.2 Practical return gain | none | `[ ____ ]` return units |

These sizes read the secondary outcomes. They are not gates, and no kill rule uses them.

## 12. Approval record

| Stage | What is approved | Approver (your approval commit identity) | Commit | Date |
|---|---|---|---|---|
| 0 | Section 0 | `[ ____ ]` | `[ ____ ]` | `[ ____ ]` |
| 1 | Section 1 | | | |
| 2 | Section 2 | | | |
| 3 | Sections 3 to 11, frozen | | | |
| 4 | Running stage 4 against sections 4 and 5 as frozen | | | |
| 5 | Running stage 5 against section 6 as frozen | | | |
| 6 | The budget sheet, against section 10 as frozen | | | |
| 7 | Run authorization | | | |

When a stage is shelved, write it up in the standing format: **The idea / What we tested (and the kill rule) / What
happened / Why**.

# Decision block for the conditional OOD design, v2

Status: **blank form, nothing decided.** Drafted 27 October 2026 for `DRAFT_PROTOCOL.md` in this folder. Every
`[ ____ ]` is a number or choice for the user to set. An agent fills no blank, and a message from an agent or workflow
never counts as approval. This form authorizes no data decoding, fitting, training, simulation or spending.

**How to use it.**
1. Fill the blanks for one stage at a time, starting at the top of the stage order.
2. Approve each stage's blanks at your own terminal before that stage runs, under your approval commit identity.
3. After a stage has run, its blanks never change. A changed number needs a new, labeled post-hoc version of this form.

Where a blank has an **analytic reference**, that number comes from the documented recorder and start law. It is a
closed-form expectation, not a measurement and not a recommendation. The sources are `REVIEW_20261027.md` and v2.

## Plain-language terms

| Term | Meaning here |
|---|---|
| OOD | Out of distribution: commands or states the training data do not contain. |
| Load, `g` | The water push on the floats: current plus the first wave coordinate, read from the observation. |
| Knee, `k` | The command size where the thruster stalls: 0.800000011920929. |
| Hole | The withheld region: load `g >= 0.4` and first-thruster command within `h = 0.05` of `−k`. |
| Band | The same first-thruster interval, `−k ± h`, at any load. |
| Adjacent bands | The two intervals of the same width on either side of the band: `[−k−3h, −k−h)` and `(−k+h, −k+3h]`. |
| Q^pi reference, `v` | The discounted return of taking one command and then following the snapshot's own actor to step 100. |
| Raw score, `q` | The first critic's value for that command. |
| Adjusted score, `z` | The raw score minus kappa times the BCA uncertainty width `U`. |
| Absolute error | `abs(q − v)` or `abs(z − v)`, in return units. |
| Regret | The best candidate's `v` minus the chosen candidate's `v`, within one fixed candidate bank. |
| Native | Unmodified Unifloral TD3+BC. |
| BCA | The original BCA actor penalty at kappa 3. |
| Distance control | The existing simple penalty, strength-matched once on training probes at the 10k fork. |
| Trained seed | One independent training initialization. It is the unit of replication. |
| Kill rule | A result, written before a stage runs, that shelves the stage or the design. |
| Inconclusive | The stage could not test the question. That is not evidence for or against BCA. |

## Stage order

| Order | Stage | Decodes or runs | Approval needed before it runs |
|---|---|---|---|
| 1 | Seed and role ledger | Reads local JSON receipts only | Sections 1 and 9 |
| 2 | Synthetic mask and arithmetic fixtures | Hand-built cases only, no dataset | Section 2 |
| 3 | Freeze this whole form | Nothing | Every remaining blank |
| 4 | Geometry-only coverage audit | Training and tuning-validation arrays only, no model | Sections 4 and 5 |
| 5 | Calibration and random-removal row plans | The same arrays, no model | Section 6 |
| 6 | Budget sheet | Nothing | Section 10 |
| 7 | Training and evaluation | Models and simulator, within the ceiling | Sections 3, 7, 8, 9 and 10, plus a separate run authorization |
| 8 | Unsealing and analysis | Saved outputs, read in the order of section 8 | Nothing new; the rules are already frozen |

Stages 4 to 7 also need the Zeus execution hold resolved by the user. That hold is recorded in
`../titan_handoff_20261027/ATTEMPT_HOLD.json`.

## 1. Seed and role ledger (stage 1)

**1.1 Ledger scope.** Ranges are collected from every local receipt. The ledger is labeled "Zeus-visible only; Titan
collisions unverified" until the Titan manifests are imported.

**1.2 Kill rule.** Freezing any new seed is blocked if either of these happens:
- a proposed range overlaps any range already in the ledger;
- any of the 1,280 episodes has an unresolved role.

**1.3 Episode roles.** These are fixed; no blank.

| Episodes | Role |
|---|---|
| 768 training | Development data, already inspected. |
| 128 calibration | The first 128 tuning-validation groups sorted by group ID. Development data, reused. |
| Other 128 tuning-validation | Development data, used in the one-sided work. |
| Selection and final calibration | Sealed. |

No result from this design is confirmatory.

## 2. Fixtures (stage 2)

**2.1 Kill rule.** The work stops on any of these:
- a disagreement between the two independent mask implementations;
- a disagreement with a hand label;
- a failed accounting identity.

The specification is fixed. Only a fixture defect may be corrected.

**2.2 Required cases.** These are fixed; no blank.
- **Load threshold:** load exactly 0.4.
- **Load precision:** a load where the float32 observation sum and the float64 state fall on opposite sides of 0.4.
- **Command boundary:** commands exactly at `−k ± h` as synthetic float64 requests.
- **Executed edges:** the nearest float32 executed values `−0.84999996`, `−0.85000002`, `−0.75000006` and `−0.75`.
- **Low load:** in-band commands at low load, which must stay eligible.
- **Kappa zero:** gives equal raw and adjusted errors.
- **No sign flip:** subtracting a nonnegative width never turns underestimation into overestimation.
- **Underestimation grows:** an underestimated score's absolute error grows by exactly kappa times `U`.

## 3. Primary comparison (fixed in stage 3, read in stage 8)

One comparison decides efficacy. Every other outcome is secondary or descriptive.

| Item | Choice |
|---|---|
| 3.1 Contrast | BCA minus native, paired within each trained seed. Alternative to choose instead: `[ ____ ]`. |
| 3.2 Data condition | `[ hole / hole minus random-removal interaction ]` |
| 3.3 Probe stratum | In-band probes at high load under recorded current. Alternative: `[ ____ ]`. |
| 3.4 Checkpoint | `[ 20k / other: ____ ]` |
| 3.5 Statistic | Decrease in mean absolute error, BCA's own `z` against BCA's own `v`, minus native's `q` against native's `v`. Use the frozen aggregation weights in v2 line 146. Alternative: `[ ____ ]`. |
| 3.6 Practical effect size | `[ ____ ]` return units of mean absolute error |
| 3.7 Trained seeds S | `[ ____ ]` (also section 9) |

**3.8 Decision rule for the chosen S.**
- **Pass.** For S of 4 or fewer: the decrease exceeds 3.6 in every seed. For S of 5 or more: the decrease exceeds 3.6 in
  at least `[ ____ ]` of S seeds, and the seed-level `[ sign test / paired t ]` gives one-sided p below `[ ____ ]`.
- **Fail.** Every exposure gate in section 7 passed, and the pass rule is not met.
- **Inconclusive.** Any exposure gate in section 7 failed, or a readiness or learning check in 3.9 failed.

**3.9 Learning check.** The native arm must improve mean paired return over its own untrained actor by more than
`[ ____ ]` in every seed and condition. If not, the run is an inadequate-learning screen and is inconclusive for BCA.
Training is not extended.

**3.10 Secondary outcomes and multiplicity.** All other contrasts, cells, checkpoints and the same-critic panel are
reported as `[ descriptive only / tested with ____ correction ]`. Secondary outcomes are listed in section 11.

## 4. Coverage thresholds (fixed in stage 3, checked in stage 4)

These are frozen before the audit decodes any array. The audit counts rows. It does not move these marks.

| Item | Threshold | Analytic reference |
|---|---|---|
| 4.1 Forbidden hole labels in any fitting, calibration or matching bank | 0, fixed | none |
| 4.2 Minimum original hole rows in the training pool | `[ ____ ]` | about 1,920 |
| 4.3 Minimum training episodes with at least one hole row | `[ ____ ]` of 768 | about 606 (79%) |
| 4.4 Minimum hole rows in each ten-step time bin | `[ ____ ]` | about 192 |
| 4.5 Minimum retained in-band rows at low load (the within-model control cell) | `[ ____ ]` | about 1,920 |
| 4.6 Minimum hole rows in each load-depth stratum: [0.4, 0.5), [0.5, 0.6), at least 0.6 | `[ ____ ]` each | shares of high-load rows about 29%, 25%, 46% |

**4.7 Kill rule.** If any of 4.1 to 4.6 fails, the region is shelved as written, and a new version may be proposed.
The new region is chosen without outcomes, and there is no search over regions.

## 5. Local support definition (fixed in stage 3, checked in stage 4)

**5.1 Method.** `[ neighbor radius / low-dimensional cells ]`

**Option A: neighbor radius.** Standardization uses the 13 physical coordinates fitted on original training rows.
Neighbors come from the same ten-step bin and the same load side. The neighbor set is fixed before any filtering.

| Item | Threshold | Analytic reference |
|---|---|---|
| 5.2 Radius, standardized RMS distance | `[ ____ ]` | 0.5 gives about 1 to 2 rows; about 0.67 to 0.77 gives 32 to 64 |
| 5.3 Maximum neighbors kept | `[ ____ ]` | v2 proposes 64 |
| 5.4 Minimum rows from distinct episodes for a populated neighborhood | `[ ____ ]` rows from `[ ____ ]` episodes | v2 proposes 32 from 8 |
| 5.5 Sensitivity settings | `[ ____ ]` neighbors at radius `[ ____ ]`, and `[ ____ ]` at `[ ____ ]` | v2 proposes 32 at 0.4 and 128 at 0.6 |

**Option B: low-dimensional cells.** Cells are ten-step bin × load bins of width `[ ____ ]` × current bins of width
`[ ____ ]`. A cell is populated with at least `[ ____ ]` rows from `[ ____ ]` episodes.

**Band counts.** The recorder ignores the state, so a band count is a binomial count. An empty band in a populated
neighborhood is mostly chance.

| Item | Threshold | Analytic reference |
|---|---|---|
| 5.6 How band counts are summarized | `[ per neighborhood / pooled over neighborhoods in a cell ]` | none |
| 5.7 Minimum original rows in the excluded band | `[ ____ ]` | 3.2 expected per 64 neighbors |
| 5.8 Minimum rows in the two adjacent bands pooled | `[ ____ ]` | 6.4 expected per 64 neighbors |
| 5.9 Minimum share of probe neighborhoods or cells that must be populated and pass 5.7 and 5.8 | `[ ____ ]` | none |
| 5.10 Maximum share of support labels that may change between the main and sensitivity settings | `[ ____ ]` | none |

**5.11 Kill rule.** If 5.9 or 5.10 fails, empirical support is labeled unknown for the affected probes, and the
conditional-hole claim is not made for them. If more than `[ ____ ]` of in-band high-load probes are unknown, the
design is shelved as unable to show a conditional hole. The radius is never relaxed to rescue labels.

Strong-current probes are outside recorded current support by definition, whatever their neighbor distance.

## 6. Row plans (fixed in stage 3, run in stage 5)

**6.1 Calibration row rule.** Use the existing calibration row from seed 2026102362 when it is outside the hole.
Otherwise use `[ next eligible step cyclically / a fresh draw from RNG address ____ ]`. About 3.2 of the 128 rows are
expected to need replacing.

**6.2 Shared calibration rows.** Do all three data conditions use one shared calibration row set? `[ yes / no ]`

**6.3 Random-removal stream.** The RNG stream is addressed per (episode, ten-step bin, load side) stratum. Address:
`[ ____ ]`.

**6.4 Imbalance bound.** Maximum standardized mean difference, hole pool versus random pool, on any state coordinate:
`[ ____ ]`. On reward: `[ ____ ]`. Reward differences are expected and systematic, because reward depends on the
command.

**6.5 Kill rule.** A calibration episode with no eligible row is a design failure, with no substitution. An imbalance
above 6.4 is reported as an imbalanced control. The plan is never redrawn.

## 7. Exposure gates (fixed in stage 3, read first in stage 8)

These are read on native and distance-control outputs before any BCA outcome is unsealed.

| Item | Threshold | Analytic or development reference |
|---|---|---|
| 7.1 Value gap: minimum spread of native `v` between the best and worst candidate families at an in-band high-load probe state | `[ ____ ]` return units | about 1 to 1.5 units across band and adjacent bands under simple continuations |
| 7.2 Value-gap prevalence: minimum share of in-band high-load probe states meeting 7.1 | `[ ____ ]` | none |
| 7.3 Error opportunity: minimum native mean absolute error on in-band high-load probes | `[ ____ ]` return units | full-data development critic: +19 to +82 at such commands |
| 7.4 Hole effect on native: minimum increase of native in-band high-load mean absolute error, hole condition minus full-data condition | `[ ____ ]` return units | none |
| 7.5 Free-policy entry: minimum share of high-load steps where the native or distance-control actor's executed command is in the band (`abs(a0 + k) <= h + 1e-6`) | `[ ____ ]` | development actor: 0 of 128 states, nearest 0.32 away |

**7.6 Kill rules.**
- **Accuracy and decision claims.** If 7.1, 7.2, 7.3 or 7.4 fails, these claims are inconclusive for the hole
  mechanism. BCA outcomes stay sealed for those claims.
- **Return claim.** If 7.5 fails, the whole-policy return claim is untested for this mechanism. Forced-probe accuracy
  may still be reported.
- **BCA's own entry share.** It is reported separately. Avoidance is itself a result.

## 8. Unsealing order (fixed in stage 3)

1. Readiness and learning checks (3.9), with refusal and invalid counts.
2. Exposure gates (section 7), native and distance-control outputs only.
3. The primary comparison (section 3), only if step 2 passed for the primary stratum.
4. Secondary outcomes (section 11).

No blank, threshold, checkpoint or probe set may change after step 1 begins.

## 9. Seeds and evaluation counts (fixed in stage 3)

| Item | Value |
|---|---|
| 9.1 Trained seeds S | `[ ____ ]` |
| 9.2 Training seed values, from a range disjoint from the ledger | `[ ____ ]` |
| 9.3 Test-family seed range, per trained seed | `[ ____ ]` |
| 9.4 Probe states per family, at times 0, 25, 50 and 75, recorded current | `[ ____ ]` |
| 9.5 Probe states per family, strong current | `[ ____ ]` |
| 9.6 Candidates per probe state: in band, each adjacent band, far set, plus the actor's own command | `[ ____ ]`, `[ ____ ]`, `[ ____ ]`, plus 1 |
| 9.7 Whole-policy return episodes per policy, recorded current | `[ ____ ]` |
| 9.8 Whole-policy return episodes per policy, strong current | `[ ____ ]` |
| 9.9 Test families | `[ separate per trained seed / shared across seeds ]` |

The standing rules reserve seeds 10 to 14 for a new version. Whether that reservation applies here is `[ yes / no ]`.

## 10. Runtime ceiling (fixed in stage 6)

With C = 3 conditions, A = 3 branches, warmup to 10k and an end at 20k, v2's formulas give:

| Quantity | Formula | With the chosen S |
|---|---|---|
| Critic updates | 120,000 × S | `[ ____ ]` |
| Actor updates | 60,000 × S | `[ ____ ]` |
| Auxiliary updates | 120,000 × S | `[ ____ ]` |
| Calibration refreshes | 21 × S | `[ ____ ]` |
| Unique actor snapshots | 33 × S | `[ ____ ]` |

**Training-time estimate.** It is extrapolated from two Zeus 10k warmups and is not a budget: about 2.1 to 2.4 hours per
seed of training only.

| Item | Ceiling |
|---|---|
| 10.1 Global wall-clock, everything included | `[ ____ ]` seconds |
| 10.2 Per warmup job | `[ ____ ]` seconds |
| 10.3 Per branch job | `[ ____ ]` seconds |
| 10.4 Reference rollouts and evaluation | `[ ____ ]` seconds |
| 10.5 Simulator transitions | `[ ____ ]`: the sum of `(100 − t)` over every candidate and snapshot, plus 100 × return episodes, plus state-bank generation |
| 10.6 Machine | `[ Zeus / Titan ]` |

**10.7 Kill rule.** A projected total above 10.1 reduces S or the counts in section 9 before the approval request. An
interrupted or timed-out job keeps its evidence and stops. There is no automatic retry or extension. A failed job counts
against the ceiling.

## 11. Secondary and descriptive outcomes (reported, not decided)

- **Same-critic panel:** `q` against `z` at the same checkpoint.
- **Error classes:** over, near-zero and under counts with `tau(v)`, and the 3 × 3 sign-transition table.
- **Paired change:** improved, worsened and tied counts with `eta`.
- **Decisions:** regret, gain, harm and choice switches within the fixed bank.
- **Learning curves:** at 0, 10k, 11k, 15k and 20k.
- **Other cells:** current-shift and combined cells, and the hole-minus-random interaction if it is not the primary.

The numerical bands are already proposed in v2. Each needs confirming or replacing:

| Band | v2 proposal | Confirm or replace |
|---|---|---|
| `tau(v)` | `1e-4 + 1e-5 × abs(v)` | `[ confirm / ____ ]` |
| `eta` | `tau(v)` | `[ confirm / ____ ]` |
| Practical regret size | none | `[ ____ ]` return units |
| Practical return gain | none | `[ ____ ]` return units |

## 12. Approval record

| Stage | Blanks approved | Approver | Commit | Date |
|---|---|---|---|---|
| 1 | 1, 9 | `[ ____ ]` | `[ ____ ]` | `[ ____ ]` |
| 2 | 2 | | | |
| 3 | all remaining | | | |
| 4 | 4, 5 | | | |
| 5 | 6 | | | |
| 6 | 10 | | | |
| 7 | run authorization | | | |

When a stage is shelved, write it up in the standing format: **The idea / What we tested (and the kill rule) / What
happened / Why**.

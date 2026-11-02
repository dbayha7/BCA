# Reconciliation of the five critical findings

Status: **reject the present band as the next policy-efficacy pilot candidate; keep launch blocked**. This is a design-priority decision, not proof that the band is mathematically value-flat or that BCA fails. No replacement band, larger support radius or new experiment is selected here. Versions 1 and 2 remain unchanged; this addendum supersedes v2's recommendation to retain the band provisionally for the next policy experiment.

The five-component structure remains useful. The current candidate lacks demonstrated practical value separation and policy exposure, and its local state-support criterion has not been measured. Those are substantive deficiencies for the advisor's decision/return objective. Forced diagnostics could still be valid for a separately specified question, but do not justify the proposed multi-condition training matrix by themselves.

## Verified versus unverified claims

| User finding | Verification and scope | Required correction |
|---|---|---|
| The band is value-flat, with differences bounded to tenths while Q errors are 20-80 | **Partially supported, full-return bound unverified.** For fixed state and the other three actions, source algebra gives immediate reward spread at most 0.0480000007153, effective-thrust spread 0.075 and next-state L2 spread at most 0.0318198052. These do not bound later actor feedback or full Q^pi for t<99. Saved high-load band candidates have absolute Q1 errors 6.5421-81.9592, MAE 47.48894; 20-80 is an approximate scale, not an all-row range. | Require a stated full-return proof or a valid saved fixed-other-actions value comparison before calling the band value-flat. Do not compare shared absolute bias directly with ranking gaps as a rejection proof. Practical value separation remains missing. |
| Radius 0.5 fails almost everywhere; 0.7-0.75 is needed for eight episodes | **Unverified numerical claim.** No matching saved count/normalization/anchor receipt was found in the scoped artifacts. V2 uses standardized RMS distance in 13 coordinates, not Euclidean distance: 0.5 RMS equals about 1.803 Euclidean; 0.7-0.75 RMS equals about 2.524-2.704 Euclidean. | Keep empirical support unknown. Obtain the exact distance/normalization/anchor/stratum/row-cap/episode-exclusion specification and count receipt. Do not adopt 0.7-0.75 or search radii automatically. Exclude the entire anchor episode when measuring other-episode support for a training-derived anchor. |
| No trained actor has approached the band; all 128 development actor actions lie around -0.48 to +0.52 | **Verified for the single saved penalty-off 20k actor.** Its exact first-command range is -0.4828205704689026 to +0.52145916223526, with 0/128 in the proposed band and 0/59 among high-load actor states. This does not establish the behavior of every later trained actor or future fresh initialization. | The available actor evidence does not support a deployed-policy exposure claim for this band. Do not promote a forced-probe result to policy relevance or tune the band merely to intersect this actor's actions. |
| Environment reset cannot restore time greater than zero | **Verified.** Public reset accepts state/start and assigns time zero. Resetting to x_t and then evaluating another 100 steps would produce the wrong time observation and horizon. The existing frozen evaluator already handles this correctly with authenticated direct dynamics and an explicit remaining horizon. | Replace generic state/time restoration wording with the exact reference contract below. Existing saved parity receipts validate the old evaluator, not a new unimplemented adapter. |
| The factor table omits the low-load retained-band within-model control | **Verified omission.** V2 prose says the band remains eligible at low load, but its factor table lacks that named comparison cell. | Add the explicit four-cell recorded-current table below. Keep the same model and report the contrast, while acknowledging that low-load versus high-load states are different populations. |

## What the saved development bank can establish

Source: `titan_20261026/payload/unifloral_operator_alignment_v1/results_20261026T221700_762125Z/raw_banks.{json,npz}`. NPZ SHA-256: `cb87c6fe809e94d38715fc45df07605a386222fb9bc2c9a33059c708463d576b`. The bank is explicitly two development batches, 128 states and 1,152 candidate rows. Candidate zero is the frozen actor action, as defined in `unifloral_frozen_usefulness_v1/simulation.py:candidate_bank`.

The proposed geometry uses `k=0.800000011920929`, `h=0.05`, action interval `[-0.850000011920929,-0.750000011920929]`, and observed load at least 0.4. The earlier cached-mask calculation selected 25 high-load candidates from 21 states. It promoted the two saved observation coordinates separately to float64 for the load test, matching v2, but evaluated `abs(actions[:,0] + k) <= h` on the saved float32 action array without an explicit float64 cast. Therefore its candidate subset and error summaries are labeled by that arithmetic, not certified as byte-exact v2 boundary membership. No new arrays were loaded to resolve boundary equivalence. The actor range lies far outside either boundary convention. Their signed Q1 error ranges from approximately -6.5421 to +81.9592. They were scored by the earlier **broad-data, penalty-off** checkpoint; these are geometric locations, not evidence that the newly proposed training hole has already been implemented.

There are **zero same-state candidate pairs holding all three other action coordinates fixed** in the saved bank. Consequently its values do not measure an isolated first-thruster sweep across the band. Random candidates differ in multiple coordinates. The saved actor range, error summaries and absence of suitable pairs can be checked without model calls or simulator steps; narrow recomputation and hashes are recorded in `EVIDENCE.json`.

### Why the local physics bounds are not a full-return proof

For the first action alone, with the other coordinates fixed, reward is `-||x_physical||^2 - 0.3*||a||^2`. Across `a0 in [-k-h,-k+h]`, its range is `0.3*((k+h)^2-(k-h)^2) = 1.2*k*h`. At t=99 this also bounds the complete remaining Q^pi difference because there is no continuation.

The stalled thrust reaches its most negative value at the knee; across this band the thrust spread is `1.5*h`. The first input column affects two velocity coordinates with gain 0.3, giving next-state difference at most `0.3*sqrt(2)*1.5*h` for the paired one-step calculation. At earlier times, a full-return bound also needs the carried state/time and sensitivity of future states, rewards and actor actions. A small first-step perturbation alone does not supply that bound.

Likewise, a common Q bias of tens of units can cancel when comparing actions within a state. Large absolute error can coexist with correct ranking. It therefore cannot replace measurements of within-state value gaps, ranking errors and practical regret. Conversely, a numerically correct ordering of nearly tied actions may have little practical value. These questions must stay separate.

## Required within-model control, if a later design is justified

For each fixed checkpoint, use the same parameters, normalization, calibration reference, candidate construction and Q^pi continuation policy in all cells. Under recorded-current starts, report:

| Observed load | First action in the candidate band | First action outside the band |
|---|---|---|
| High: g >= 0.4 | Proposed **withheld conditional-label** cell | Retained alternative-action cell at the high-load states |
| Low: g < 0.4 | **Retained-band within-model control**: the same action interval retains labels | Retained low-load reference cell |

Require original and retained counts, distinct episodes, state-support status, per-probe signed errors, MAE, adjustment benefit/harm and decisions for each cell. At a given state, candidate comparisons must keep the other action coordinates fixed. Never classify every low-load band action as empirically supported merely because its mask permits it.

This control holds the model fixed, not the state distribution. High/low-load contrasts can differ because physics, occupancy, remaining time and baseline errors differ. Report the band-versus-outside contrast within each load stratum and any cross-load difference descriptively; it is not a pure causal effect of withholding. The full-data and count-matched random-removal comparisons remain necessary for that broader question and still have the limitations recorded in v2.

Strong-current tests remain a separate state-shift factor. Apply the same load/band labels and report empty cells without manufacturing balance, but do not call strong-current pairs previously covered and then selectively removed. No current-shift outcomes enter model selection.

## Exact reference-state/time contract

The canonical environment's `reset` always sets `_t=0`; it exposes no arbitrary-time restore option. A new evaluator must use one of two validated routes: replay the authenticated saved prefix to reach x_t and time t, or use the existing authenticated direct-dynamics continuation contract. Do not simply mutate a private time field or silently run a fresh horizon after reset-to-state.

For direct continuation:

1. Bind the exact carried float64 state, t, source/physics hashes and checkpoint actor to the probe. Verify the saved observation equals the state rounded to float32 with time coordinate `(100-t)/100`.
2. Execute the candidate using the same float32 action conversion and canonical reward/transition arithmetic. Subsequent actions use that same frozen actor and reconstructed remaining-time observations.
3. Execute exactly `100-t` transitions. Discount starts at exponent zero at the query. The final pre-transition reward is included; no reward or bootstrap is added after t=99 to 100.
4. Require parity/accounting checks against environment-driven sequences at initial, interior and final times. Bind any reused fixtures to the exact new caller/source; stop if they fail. Do not claim old receipts validate new code automatically.

Both saved development value receipts already report complete references at times `{0,25,50,75}`, 576 continuations and 36,000 continuation transitions each, plus 104 verification transitions. Their verification records cover initial and final-step state/reward/observation parity, remaining time and terminal semantics. These are **existing completed counts**, not work performed by this audit.

## What the previous audit missed and the corrected decision

The previous audit correctly kept execution blocked, distinguished Q^pi from the TD operator, and marked region/radius parameters provisional. It nevertheless missed using the available actor-range evidence before recommending this band for a policy experiment. It also failed to state the reset-time limitation explicitly and omitted the low-load retained-band cell from the factor table. Its four-action sparsity calculation did **not** validate the separate state-neighborhood radius. Those omissions are corrected here.

The full-return flatness bound and numerical neighborhood claims still need provenance; they are not accepted as measured facts. A radius receipt needs dataset/normalization hashes, anchor identities, RMS-versus-L2 definition, time/load filters, row cap, whole-episode exclusion and per-anchor distinct-episode counts. A full-Q bound needs the exact continuation policy/state/time assumptions and proof or eligible saved comparisons. No such evidence is created by relaxing thresholds.

**Decision:** do not spend the proposed policy-training budget on this candidate. Reject its readiness and priority for the next efficacy test; retain it only as a documented unvalidated geometry. This is neither a scientific rejection of all conditional-hole designs nor a reason to modify BCA. Any later replacement needs an explicit new proposal justified by mechanics and pre-intervention coverage, with practical value exposure and policy relevance established independently of favorable BCA outcomes. No automatic radius/band tuning, new rollouts, or launch follows this addendum.

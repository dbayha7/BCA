# OOD actions: current experiment overview

**Our priority is to test what happens when the agent takes an unfamiliar action.**
We already have training-performance logs. We will use those as supporting context
and concentrate the new analysis on actual action consequences.

The main question is: **from the same state, do actions with larger BCA widths
actually lead to worse outcomes?** A second question is whether the BCA policy
chooses better actions than the plain host from those same states.

## What we will do

1. **Load completed models.** Start with the first accepted 1M TD3+BC Hopper
   host/BCA pair, seed 202609171. Use standard BCA without importance weighting;
   retain both Bayesian and conformal components. Select by the declared queue
   order and readiness, never by the scores. Keep all five declared seeds.
2. **Save identical starting situations.** Let each frozen policy collect states
   under paired reset seeds. Save the full simulator state so every candidate
   action starts from the same situation. Missing captures stay missing.
3. **Choose actions before observing their consequences.** Compare the host
   action, BCA action, a nearby recorded action, a random action, and six fixed
   perturbations. Save BCA widths and a separate training-data distance score.
4. **Try each action in the simulator.** Restore the same state, take the chosen
   first action, then follow a fixed continuation policy for at most 250 total
   transitions. Repeat with the other continuation policy as a separate comparison.
   The networks and calibrator stay frozen; these tests do not train them.
5. **Measure the consequence and compare it with the warning.** Sum actual raw
   rewards and compare with the host first action under the same continuation.
   Larger loss means greater harm. Ask whether BCA width identifies these losses
   better than chance and better than the dataset-distance baseline.

`harm = return after host first action - return after candidate first action`

Example: from one saved state, the host first action produces return 40 and a
candidate produces 25 under the same later policy. That candidate's harm is
`40 - 25 = 15`. It is harmful under the declared threshold of more than 1 raw
reward unit. These numbers are illustrative, not results.

## What each measurement means

| Measurement | What it tells us | What it does not establish |
| --- | --- | --- |
| Distance from recorded actions near that state | A reproducible proxy for unfamiliarity | That every distant action is bad |
| Frozen BCA width | BCA's residual-scale warning for that state/action | The true Q-value or a guaranteed bound on action harm |
| Matched simulator return loss | Actual finite-horizon consequences under the named continuation policy | Infinite-horizon value or the cause of a training difference |
| Within-state AUROC | Whether larger widths rank harmful actions above nonharmful alternatives in each state | Broad OOD detection from a small or sparse panel |
| Host-action versus BCA-action return | Which first action worked better under the same later policy | That calibration caused the policy difference |
| Existing final scores and learning curves | Whole-policy performance context | Which individual actions caused the result |

**If BCA ranks harmful actions well:** this supports the usefulness of its width
as a warning signal in the tested states, candidates and continuation policy.
Beating the distance baseline provides stronger evidence than beating chance alone.

**If it ranks near chance, backwards, or ties all actions:** the width is not
usefully distinguishing harm in that panel, even if policy performance improved.
Changing one positive global radius cannot fix the action ordering.

**If BCA warns about harmless unfamiliar actions:** it may be overly cautious for
those actions. If it gives narrow widths to harmful ones, it misses their risk.
Include unfamiliar beneficial and familiar harmful actions in the interpretation.

States with only harmful or only nonharmful alternatives have AUROC **N/A**.
They are not proof of safety or perfect detection. Keep pooled AUROC separate
from within-state results, and show valid-state counts and seed uncertainty.

## What is ready, and what comes next

As checked on September 27, 2026, the first TD3+BC Hopper host/BCA pair has accepted 1M training audits and separate CPU checkpoint/simulator gates. Its production bank now contains **all 256 declared state captures** and **2,560 precommitted candidate slots**, plus the training-complement support bank and actual JAX continuation keys. Both production workers and their supervisors exited 0, and independent saved-record audits passed.

**The first real action-harm experiment has resumed after David’s desktop restart.** At September 28 01:54 UTC (September 27, 9:54 p.m. Eastern), it had advanced to 144/512 state/continuation panels. All 143 previously closed panels and 1,438 completed action rollouts are retained; all 237 saved records from the one interrupted action matched exactly during replay. The original attempt and unknown exits remain preserved. The recovery worker’s actual closure and independent outcome audit are still pending; no harm-ranking result is accepted. All five training seeds and both continuations remain required. See the [current collection status](../../docs/OOD_ACTION_COLLECTION_STATUS.md) and [recovery evidence](../../outputs/ood/td3_bc/hopper/s202609171/desktop-restart-v1/README.md).

Fresh residual coverage is a separate endpoint. ReBRAC's missing recorded-next-action
coverage contract stays unavailable; we will not invent a target or call action-harm
measurements coverage. Any executable subset must be explicitly declared and keep
the original full protocol and missing endpoints visible.

The local queue has since stopped on a CQL evaluation-schema validation failure;
its original exit 1 is preserved and no retry was launched. The cluster continues.
No queue-hold answer has arrived. This priority update neither interrupts an
active worker nor resumes an old halted queue. No model weights, training recipe,
scientific settings, gates or outcomes are changed by this document.

- [Accepted first-pair identities and remaining gates](../../docs/validation/ood-actions-first-readiness.json)
- [Trained CPU connection and collector validation](../../docs/validation/ood-trained-td3-hopper-connection.json)
- [Next implementation steps](../../docs/superpowers/plans/2026-09-27-ood-actions-first.md)
- [Full fixed scientific design](../../docs/superpowers/plans/2026-09-24-ood-experiment.md)
- [Every annotated experiment setting](../../configs/ood.yaml)
- [Adapter engineering evidence](../../docs/OOD_ADAPTER_TESTS.md)
- [Calculation and synthetic-figure checks](../../docs/OOD_CALCULATION_TESTS.md)
- [Existing training status and performance readouts](../../docs/STANDARD_BCA_STATUS.md)

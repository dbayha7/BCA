# Proposed v2 correction to one candidate-generation stream

Status: **PROPOSAL ONLY — awaiting explicit direction. Not applied.**

The original protocol and all accepted component sources stay immutable. Its
scientific design, training seeds, checkpoints, support thresholds, candidate
pool size/selection, endpoints, inference and resource caps remain as declared.

The complete current-campaign inventory contains597,032 declared seed occurrences
and83,875 distinct seed integers, from all280 standard training configurations,
all40 original OOD checkpoint configurations and all72,777 V1 OOD allocations.
Exactly one of the16,640 frozen v2 stream integers overlaps that inventory:

| Field | Value |
|---|---|
| Host/environment/training seed | TD3+BC / Walker2d /202609171 |
| Purpose | Candidate proposal generation |
| Nominal state index |54: host collector, reset block27, capture step100 |
| Original candidate-generation seed |404735174 |
| Proposed candidate-generation seed |404735181 |
| Other v2 stream integers changed |0 |

The original value is already declared for training-evaluation episode index3
of evaluation event126 in42 standard configurations and eight original resolved
OOD checkpoint configurations. This is an integer-identifier collision under
the declared gate; it does not by itself demonstrate identical random draws
across different libraries, biased performance or a failed scientific run.

The proposed rule is the first larger uint32 integer absent from the union of
all predecessor seeds and all original v2 seeds. Values404735175 through
404735180 are also reserved;404735181 is the first free value. An independent
stdlib implementation checked the entire proposed16,640-stream map: exactly
one change, zero predecessor overlaps and zero internal duplicates. No action
pool, JAX key, model output or v2 outcome was generated to choose it.

If approved, implement this **one named exception only** in separately versioned
stream/precommit/driver bindings, bind this amendment and the original protocol
by hash, and independently test full-map equality outside the one exception and
collision refusal. Preserve all original sources and the failed gate. This is
not permission for automatic repair of later collisions, new training seeds,
candidate selection by score, scientific retries or bypassing any remaining
runtime, storage, engineering or outcome gate.

The current v2 stream-acceptance gate stays failed, and no v2 scientific work is
dispatched while the proposal is pending. Unaffected engineering preparation and
read-only training audits can continue. Active training queues remain unchanged.

Reason approval is requested: the frozen v2 protocol says, “any collision is an
engineering failure, not silently resampled.” The request explicitly changes one
frozen stream assignment rather than treating the failure as automatic reseeding
permission. The prospective amendment is after v1 results, before any v2 results;
it must not be described as part of the original preregistration.

Evidence: monitor_20260928T200658Z/real_stream_inventory.json and actual gate exit1;
independent_stream_review.json and independent_inventory_v2_actual_exit.json0.
The initial independent-review script's extra assumption that the immediately
following integer was free failed before producing a proposal. That actualexit1
and original script remain; the separate v2 review applies the stated first-free
rule without that assumption. No original stream was changed in either review.

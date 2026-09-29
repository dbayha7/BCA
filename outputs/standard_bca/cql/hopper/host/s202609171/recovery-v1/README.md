# CQL Hopper host: saved result recovered

The first CQL host finished one million updates and its final evaluation, but
the common result validator then read the wrong episode field names. CQL writes
`score` and `return`; the validator expected `normalized_score` and `raw_return`.
This stopped the local training queue on September 27. It was a post-training
validation bug, not evidence that the CQL learner failed to train.

On September 29, David explicitly requested the fix. The validator now selects
the schema using the declared host. Only that function and its call site changed.
Every scientific source, configuration, seed, numerical gate and saved original
outcome remains unchanged. No completed training or simulator evaluation was rerun.

| Recovered result | Value |
|---|---:|
| Host / actor / updates per critic | 1,000,000 / 1,000,000 / 1,000,000 |
| Final normalized mean, 20 reserved episodes | 62.9825183 |
| Final normalized median | 59.6367230 |
| Same checkpoint, last periodic bank mean | 62.0635469 |
| Mean of 200 periodic bank means | 56.5803372 |
| Verified banks / episodes | 201 / 2,020 |
| Verified saved checkpoints | 10k, 50k, 1M |

This is one host training seed. Its BCA comparison and action-level OOD results
are not complete. The 1,000 journal metric rows are sparse scan summaries,
not one million per-update records.

## Acceptance and provenance

[audit.json](audit.json) checks all 108 original source files, original settings,
raw data and episode mappings, whole-episode reservations, normalization and
reward units, all scan/refresh/evaluation schedules, finite checkpoint leaves,
optimizer counts and normalized-score arithmetic. An independent saved-file
review checked the original hashes, journal arithmetic, checkpoint counters and
137-row continuation manifest. Nine regression/lifecycle tests passed. The first
unstarted CQL+BCA typed protocol retains 198 refreshes and 201 evaluation banks.

[recovered_result.json](recovered_result.json) is a derivative saved-result
acceptance. **The original worker and controller really exited 1.** Their exit
receipts, missing original result/exit files, failed validator and original
checkpoint/journal bytes remain unchanged. No original exit 0 is manufactured.
Future OOD adapters must consume this explicit recovery provenance and pass
their own checkpoint/simulator gates; `ood_ready` remains false.

The original saved-only audit attempt incorrectly assumed target counters and
the disabled Lagrange temperature optimizer had zero steps. Source inspection
showed CQL increments target steps during Polyak updates (optimizer counts stay
zero), and still advances the zero-gradient alpha-prime optimizer. The corrected
audit verifies those exact source semantics. The initial assertion failure and
source are archived in [execution_attempts](execution_attempts/). A lifecycle
fixture import failure was corrected only by running from the frozen source
directory; both attempts remain. Neither event executed a scientific retry.

## Untouched training continuation

The original 137 unstarted TD3+BC/CQL rows are frozen in an execution-only
[continuation package](../../../../../../../experiments/standard_bca/cql_validation_recovery/README.md).
The two completed TD3 runs and this recovered CQL host are excluded.
Cluster job **27068529** now owns this queue exclusively; its predecessor
27068516 stopped before training at a separately documented data-check wrapper
mismatch. [Execution v2](../../../../../../../experiments/standard_bca/cql_validation_recovery/v2/README.md)
corrects that wrapper and reuses the seven passed TD3 checks. The existing ReBRAC/IQL
job **27045057** continues unchanged. Both have separate A100 allocations.
The local launch guard was safely blocked by the running CPU OOD worker's shared
lock. That lock and worker were left intact, and the local queue now has a durable
delegation claim preventing a duplicate launch.

Moving the unstarted rows changes their execution hardware from RTX 5070 Ti to
A100. Record that change explicitly. In particular, the first recovered CQL host
and its future BCA counterpart are a mixed-hardware pair; four later CQL seeds
will have both methods on the cluster. No cross-device bitwise equivalence or
hardware-independent causal effect is claimed. No seed is replaced or excluded.

Original manifest: `13ae3e6693d1cc71228e7b676243232c52ac6766c6a2c9b0623aaac1145f97fe`.
Continuation manifest: `9ecf480c560871832ec6384045e5646de3238de8697bb56d6ee9dec85172b3cf`.
Recovered result: `77c29d79adbef813c5176a8e0aea130c35beefa4fa7e57a8fd476907c106f9fe`.
Original audit: `5edea41fdb7bb738a51a479f45da742b853eef4e6320f9c43222347c4787294e`.

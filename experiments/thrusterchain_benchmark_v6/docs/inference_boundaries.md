# Inference instrumentation and provenance boundaries

The supported benchmark evaluator uses a fixed built-in prediction implementation,
immutable parameter arrays, matching receipts and before/after parameter checks.
`InferenceGuard` adds cooperative Python instrumentation around that execution.
Its counters report calls observed through specific instrumented paths. Zero
counters do not prove that arbitrary code performed no I/O, accessed no held-out
data, or temporarily adapted its state.

## What the guard observes

| Counter | Observed path |
|---|---|
| `training_calls` | Calls through the patched `regular.train_behavior_cloning` attribute |
| `dataset_collection_calls` | Calls through the patched `datasets.generate_dataset` attribute |
| `write_attempts` | Write-mode calls through patched `builtins.open`, `io.open`, `os.open` and `pathlib.Path.open` lookups |
| `forbidden_api_calls` | Built-in benchmark hooks using this guard's active context-local ledger |

The context checks inside supported fitting, identification, collection,
calibration and persistence functions also reject aliases imported before entry,
when those aliases run in the active evaluation context. These are hooks in
benchmark functions. They do not extend to an alias of an uninstrumented external
function. With `activate_context=False`, this guard does not activate its own
context ledger. A nested built-in evaluator can own a separate ledger; its counts
must be read from that evaluator's result and are not merged into this receipt.

## What remains outside the claim

Reads are permitted and unmonitored. Reading a file containing a synthetic
final-only canary produces zero attempted-write and forbidden-API counts. The
guard therefore does not establish final-data confidentiality or prove absence
of held-out-data access.

A `builtins.open` callable saved before entering the guard can write a file
without incrementing `write_attempts`. Existing open file handles, uninstrumented
native calls, subprocesses and other file mutation APIs are also outside the
complete-monitoring claim. Some alternate Python APIs happen to reach patched
lookups, but this is not a promise of exhaustive interception.

Threads have no comprehensive isolation guarantee: monkey-patched attributes are
process-global, while evaluation state is context-local. Calls in another thread
may not share the benchmark context. The guard does not certify thread safety,
context propagation, or arbitrary concurrent inference purity.

The guard does not observe arbitrary parameter updates or temporary state
changes. Changing a parameter and restoring it can leave both these counters and
a before/after hash unchanged. Purity claims for the supported immutable built-in
policy depend on its restricted implementation and its own tests, not on a zero
guard counter alone. No OS sandbox or exhaustive I/O isolation is implemented.

`tests/test_revision_planner_guard.py` deliberately demonstrates the read, saved
file-open alias and transient-state limitations using only temporary synthetic
fixtures. Those tests also require the receipt to disclose the limitations.
Existing `tests/test_inference_guard.py` checks supported training aliases and
write refusals; built-in evaluator purity checks remain in `tests/test_regular.py`.

## Planner matrix provenance

`BoundedPlanner` derives `provenance` from its constructor inputs:

| Inputs | Derived receipt provenance |
|---|---|
| Omitted `A` and `B` | `privileged_true_model`; matrices loaded through `environment.matrices` |
| Explicit `A` and `B` | `caller_supplied_model`; matrices came from constructor arguments |

A conflicting declaration with omitted matrices, including
`provenance="train_only_identified_model"`, raises `ValueError` before loading the
true model. With explicit matrices, a supplied label is retained separately as
`declared_provenance`. It cannot turn a caller assertion into verified training
history, even if the values happen to equal the true model. The receipt and solve
certificate both state `training_provenance_verified=False`.

The existing runner supplies learned `A` and `B` and retains the separate
`identify_from_training` receipt under `identification`. That receipt records the
training split, groups, fit details and a matrix hash that can be compared to the
planner's matrix hash. The planner constructor itself does not authenticate that
history. This separates observed matrix input from the caller's claimed origin
while preserving the genuine train-only identification workflow.

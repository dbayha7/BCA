# Host training and evaluation contract, v1

The public boundary is `thrusterchain.training_protocol`. Its module-level code
imports NumPy and standard-library code only and never constructs an environment.
Checkpoint persistence lazily imports the shared evaluation guard; the existing
package initializer may separately import the environment module. The machine-readable companion is
`contracts/training_modes_v1.json`.

## Budgets and actual work

`OfflineBudget(updates=1_000_000, batch_size=256)` describes a requested offline
budget. The default is **1,000,000 primary optimizer updates**, and the offline
maximum is **3,000,000 primary optimizer updates**. For TD3-family hosts, the
declared primary optimizer is the critic optimizer; delayed actor and auxiliary
steps are reported separately. Counts must be Python
`int` values: booleans, floats (including `3.0`), strings and NumPy integer scalars
are rejected. Updates may be zero; batch size must be positive. The default and
maximum are an execution contract, not a claim that either training run has been
performed or is sufficient for convergence.

An **optimizer step is exactly one physical `optimizer.step()` invocation** (or
one equivalent parameter update in the NumPy reference optimizer). Calling the
critic once and the actor once consumes one primary update and two total optimizer
steps. Gradient accumulation,
minibatch sampling, target-network Polyak updates, RNG advancement, validation,
and inference do not themselves consume optimizer steps. An adapter's `update`
must expose exactly one primary optimizer step, together with the auxiliary
optimizer steps prescribed by the host's schedule. Record every optimizer's
monotonic count; their sum must equal the total. The runner rejects a call that
advances the primary counter by zero or more than one. It must stop before any
primary update would exceed the requested budget; auxiliary steps do not reduce
the primary budget.

Requested and actual work must be separate in receipts: requested offline primary
updates, actual offline primary updates, actual offline total optimizer steps,
requested online environment interactions,
actual online interactions, requested online primary updates, actual online
primary updates, actual online total optimizer steps, and per-optimizer counts.
The requested primary count is at `budget.requested.updates`; common receipt
fields are `primary_optimizer`, `requested_primary_optimizer_updates`, `executed_primary_optimizer_updates`,
`executed_optimizer_steps_total`, and `optimizer_step_counts`.
Online budgets are per scenario family; the top-level requested and executed
counters sum across families. `requested_primary_optimizer_updates_per_family`
and `budget.executed_per_family` preserve the per-family scope.
For resumption, retain both cumulative
checkpoint counts and the actual counts added by the current invocation. A
requested budget never substitutes for a measured actual count.

The existing behavior-cloning and identification fits are closed-form fits. Their
optimizer budget is inapplicable and they report zero optimizer steps. They must
not run repeated solves or meaningless loops to imitate a million-step budget.
Those built-ins do not support online adaptation. The affine SGD behavior-cloning
reference adapter supports real atomic updates and online plumbing; it is not a
neural BCA implementation, a neural reinforcement-learning baseline, or evidence
that a host neural learner has been integrated.

## Explicit online settings and strict frozen default

```python
OfflineBudget(updates=1_000_000, batch_size=256)
OnlineBudget(interactions, updates, offline_fraction, batch_size, learning_settings)
EvaluationConfig(frozen_policy=True, episodes=2, seed=20361022, online=None)
```

All five `OnlineBudget` arguments are mandatory. Interactions and updates are
nonnegative Python integers, batch size is a positive Python integer, and
`offline_fraction` is finite and within `[0, 1]`. Positive online updates require
positive online interactions. `learning_settings` must be an explicit, finite,
JSON-compatible dictionary; an adapter validates which settings it accepts. An
explicit empty dictionary is structurally valid but may be rejected by the host.
The requested mixture and actual sampled origins belong in the receipt; rounding
at finite batch size may change the realized mixture. Origin labels identify
offline transitions as `0` and newly collected online transitions as `1`.

Frozen evaluation is ON by default. It rejects any online budget, even an all-zero
budget. Switching `frozen_policy=False` requires an explicit `OnlineBudget`.
`interactions=0, updates=0` requests no adaptation; receipts must say so rather than
call that fine-tuning. Positive interactions with zero updates permit collection
without parameter learning and must report that distinction. Online budgets and
capability checks must be validated before collecting transitions or writing
results. The offline maximum does not silently create an online limit or default.

Budget/configuration classes expose `to_dict()` for JSON-safe receipts.
`OnlineBudget.learning_settings` returns a fresh deep copy, so mutating an input
or a returned dictionary does not change the approved configuration.

## Host adapter API

```python
class LearnerAdapter(Protocol):
    @property
    def capabilities(self) -> AdapterCapabilities: ...
    @property
    def optimizer_steps(self) -> int: ...
    @property
    def optimizer_step_counts(self) -> Mapping[str, int]: ...
    def initialize(self, train_batch, seed: int, learning_settings: dict) -> None: ...
    def update(self, batch) -> dict: ...  # one primary step; count auxiliary steps
    def act(self, observation, explore: bool): ...
    def checkpoint(self) -> LearnerCheckpoint: ...
    def restore(self, checkpoint: LearnerCheckpoint) -> None: ...
    def freeze(self) -> FrozenAdapterPolicy: ...

class OnlineLearnerAdapter(LearnerAdapter, Protocol):
    def configure_learning_settings(self, learning_settings: dict) -> None: ...

class FrozenAdapterPolicy(Protocol):
    def act(self, observation): ...
    def state_bytes(self) -> bytes: ...
```

`AdapterCapabilities(adapter_id, fit_semantics, supports_online, primary_optimizer=None)` declares
`fit_semantics` as either `optimizer_steps` or `closed_form`, with an actual Python
boolean for online support. Optimizer-based hosts must name their nonempty primary
optimizer counter key; closed-form hosts must use `None`. The checkpoint metadata
must repeat this name at `metadata['primary_optimizer']`, and the runner must check
it agrees with the capability declaration. `OnlineLearnerAdapter` is a capability-gated extension:
adapters advertising online support must implement it. The protocol types are
structural typing aids; the runtime cannot prove arbitrary implementations obey
their declared semantics.

The two read-only count properties must be cheap and return defensive values.
Validate their increments after each primary update; take full checkpoints at
phase boundaries and verify their counts agree. A million-step run must not
serialize/hash a full neural checkpoint after every step or retain a giant
per-step receipt. Count declarations remain cooperative: a host audit must verify
that each increment corresponds to an actual optimizer call. Both lightweight
`optimizer_steps` and checkpoint `optimizer_steps` are total optimizer steps,
including auxiliary optimizers; they are not the primary budget counter.

Initialization fits preprocessing using the training split only. It initializes
model, optimizer, targets and all RNG streams without consuming optimizer steps.
The runner controls atomic updates, uses executed actions in its replay, and
retains sampler RNG state. Restoring a checkpoint restores training state before
any online settings change. `configure_learning_settings` is an explicit phase
boundary: it changes declared learning hyperparameters while preserving model
weights, optimizer accumulators and counters, targets, preprocessing statistics,
and RNG state. Adaptation records retain `learning_settings_before`,
`learning_settings_after`, `learning_settings_before_sha256`, and
`learning_settings_after_sha256`.

The runner may add checkpoint metadata such as `runner_state`, dataset lineage,
phase and parent digests. Host adapters must preserve unknown metadata when
restoring and taking a subsequent checkpoint, so sampler resumption state is not
lost. A no-update restore/checkpoint round trip must preserve the checkpoint
digest unless an explicit, recorded settings change occurs.

## Immutable transition batches

`TransitionBatch` contains `observations`, `actions`, `rewards`,
`next_observations`, `terminated`, `truncated`, `bootstrap_mask`, `origin` and
`group_ids`. Every array shares a leading transition dimension. The six scalar
fields from `rewards` onward, excluding `next_observations`, have shape `(N,)`.
Observations and next observations have identical shapes. Actions are the actual
actions executed after clipping or any other action transform, never an
unexecuted proposal. Numeric transition values must be finite; terminal flags
are boolean arrays. Group IDs are integers or fixed-width strings and retain
dataset group provenance through sampling.

`bootstrap_mask` equals `1 - terminated`: a time-limit truncation alone still
bootstraps. Both flags may be true if the transition actually terminates at a
time limit. Origin is integer `0` or `1`. Object arrays are refused. Construction
detaches every array into immutable bytes-backed storage, so caller mutation and
`setflags(write=True)` cannot change the batch. Every field access returns a fresh
read-only view, so reshaping a retrieved array or changing its ndarray header does
not change the stored batch layout. `len(batch)`,
`batch.select(indices)` (integer vector, boolean mask, or slice), and
`batch.concat(*others)` return or preserve the same contract.

## Full-state checkpoints and safe serialization

```python
LearnerCheckpoint(
    adapter_id, components, optimizer_steps, optimizer_step_counts, metadata
)
```

Components map names to immutable `bytes`; all checkpoints require the names
`model`, `optimizer`, `target_networks`, `rng` and `preprocessing`. Store all
resumption state: model parameters, every optimizer's accumulators and step
counters, target networks, learning settings and schedules, all random streams,
normalization statistics, and action transforms. Additional named components
are permitted. A genuinely inapplicable component still needs explicit bytes
describing the reason and its name in `metadata['not_applicable_components']`.
For example, affine SGD behavior cloning has no target networks. Empty or
placeholder bytes must never be used to disguise an omitted applicable state.

The initial checkpoint metadata must retain the training configuration and
dataset lineage: content/file digest as applicable, training split identity and
digest, grouping provenance, seed, and host algorithm/version identity. Runner
sampler state and cumulative counts must support identical continuation from an
offline checkpoint. Online checkpoints retain their parent checkpoint digest,
phase settings and actual adaptation counts. A host is responsible for selecting
an appropriate portable representation of its own component state.

Component mappings and per-optimizer count mappings are defensively copied and
read-only. `optimizer_steps` is the sum of all per-optimizer counts.
`.primary_optimizer` and `.primary_optimizer_steps` derive the primary name and
count from metadata (both are `None` for an undeclared generic checkpoint).
Optimizer-based host checkpoints must declare that name, even at zero steps.
Metadata is kept internally as canonical JSON; `.metadata` returns a
fresh deep-copied dictionary. `.component_hashes` returns SHA-256 values and
`.digest` hashes the canonical full payload. `.to_bytes()` / `.from_bytes()` use
a versioned JSON/base64 envelope, reject duplicate JSON keys and non-finite JSON
numbers, and verify payload and component hashes. They never invoke pickle or
deserialize arbitrary Python objects. `.save(path)` creates a new file
exclusively and refuses overwrite; `.load(path)` validates the same envelope.
The save method checks the shared evaluation context before opening a file,
including when a caller saved an alias before entering evaluation.

Presence checks and hashes provide **cooperative integrity**, not proof of
checkpoint completeness, scientific validity, adapter correctness or honesty.
A malicious or mistaken adapter can omit state from its declarations or lie
about optimizer calls. Independent host tests should compare uninterrupted
training with checkpointed/resumed training, exercise multiple optimizers, and
audit the host's actual optimizer call sites.

## Frozen evaluation boundary

`run_mode(..., checkpoint_directory=path)` saves every scenario endpoint to a new
directory, reloads each checkpoint through the validated serializer, verifies its
digest, and restores the reloaded state before any frozen evaluation begins. The
CLI and shipped example supply this directory. Receipts report
`checkpoint_persistence="saved_and_reloaded"`,
`persisted_before_frozen_evaluation=True`, and
`evaluation_checkpoint_source="reloaded_verified_checkpoint"` for this path.
The optional Python-only `checkpoint_directory=None` path supports lightweight
tests and explicitly reports `in_memory_snapshot`; it must not be described as
evaluation from a saved checkpoint.

A frozen policy receives only observations. It has no update, replay insertion,
exploration, model-mode change, or normalization-fitting path. Its
`state_bytes()` must include every inference parameter, preprocessing statistic,
action transformation and relevant RNG state. Freeze by detached immutable state,
and compare its state digest before and after evaluation. Evaluation receives no
training batches and does not perform adaptation, fitting, or data collection.
`act` performs no checkpoint loading, saving, or other I/O.

The benchmark's existing `InferenceGuard` refuses supported training, collection
and Python file-write entry points during rollout. Host training entry points
must participate in its context checks. Endpoint checkpoint writes, reloads and
restoration finish before the frozen guard starts; result artifacts are written
after the guard exits. These hooks and before/after digests cannot establish purity of arbitrary
Python/native code, capture every alias or thread, detect state omitted from
`state_bytes()`, or detect a transient mutation that is reverted before hashing.
Zero counters mean no forbidden call was observed through the instrumented
paths; they are not a security sandbox or proof of no I/O/adaptation.

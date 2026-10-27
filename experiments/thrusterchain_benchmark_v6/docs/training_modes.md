# Training budgets and the frozen-policy toggle

Strict offline evaluation is the default. The policy is frozen before complete
simulator episodes begin. Switching frozen policy off selects a separate
offline-to-online experiment with explicitly declared interaction and update
budgets. It does not weaken or relabel an existing strict result.

The reusable [host adapter interface](host_adapter_interface.md) is the extension
point for externally developed neural methods. The bundled iterative example is
an affine policy trained by SGD on action imitation. It demonstrates actual
gradient updates, replay, resumption and phase separation; it is not a neural
offline RL host or evidence of beneficial online control learning.

## Three different budgets

| Quantity | Meaning | Default and scope |
|---|---|---|
| Offline primary optimizer updates | Steps of the adapter's declared primary optimizer using fixed Train data | 1,000,000; configurable through 3,000,000, including resumed offline work |
| Online interactions and primary updates | Newly collected transitions and primary optimizer steps during adaptation | Both explicitly supplied; no implicit allowance |
| Evaluation episodes | Complete episodes of a frozen checkpoint | 2 per scenario unless specified |

An adapter update performs exactly one step of its declared primary optimizer.
For TD3-family hosts the primary optimizer is the critic: one million primary
critic updates may also execute scheduled actor and auxiliary updates. Those
steps have separate counters, as does the total across all optimizers; they do
not consume the critic-update budget. The affine SGD example declares its
imitation optimizer as primary. Minibatch samples, target-network averaging, and
environment interactions are not silently counted as primary updates.
Configuration values must be integers, not Boolean
or floating-point aliases. Execution receipts report the requested and actual
counts; passing a large allowed budget in a test does not mean it was executed.

Closed-form BC fits a small declared set of least-squares candidates. Train-only
system identification fits least squares and then plans with frozen coefficients.
Neither has a gradient-update budget or online adaptation implementation. The
existing BC command reports the default budget as inapplicable and executes zero
optimizer updates; an explicitly supplied optimizer budget is rejected. The
planner remains an inference-only baseline. These methods never run empty loops
to match a neural learner's nominal update count.

## A small runnable example

These commands deliberately override the million-update iterative default for a
protocol smoke check. Choose fresh output paths. A full-budget study requires its
own registered configuration, compute budget and multiple training seeds.

```sh
python -B -m thrusterchain generate outputs/data.npz --episodes 40
python -B -m thrusterchain train outputs/data.npz outputs/pretrained --adapter affine-sgd --offline-updates 8 --batch-size 16
python -B -m thrusterchain evaluate outputs/pretrained outputs/strict.json --dataset outputs/data.npz --episodes 2
python -B -m thrusterchain evaluate outputs/pretrained outputs/adaptation.json --dataset outputs/data.npz --no-frozen-policy --online-interactions 12 --online-updates 6 --offline-fraction 0.5 --batch-size 16 --learning-settings '{}' --episodes 2
```

PowerShell users should pass the JSON settings as a single quoted argument.
`--frozen-policy` is on by default. The legacy `--learning disabled` option remains
available for strict evaluation and is rejected with adaptation. Switching the
toggle off without the full online configuration is an error. Zero interactions
and zero updates means **no online learning**, not a successful adaptation run.

The iterative checkpoint contains the full adapter state, including optimizer
momentum, action RNG and fixed preprocessing; its target-network component is
explicitly not applicable for this affine SGD method. Runner replay-sampling state
is preserved for supported offline resumption. `train --resume-checkpoint PATH`
continues with an additional declared primary budget and retains lineage; the
cumulative offline primary count must remain at most 3,000,000. Resumption
support and its exact limits are documented in the adapter interface.

## Reading adaptation output

Every scenario starts from a newly restored instance of the same pretrained
checkpoint. Each gets its own reset and replay-sampling streams and a fresh online
replay buffer. The current reference runner retains all collected transitions in
memory and exports their records; large online budgets have not been performance
tested. The explicit replay mixture determines how many sampled batch rows
come from the unchanged offline Train split versus newly collected transitions.
Learning settings change only at the declared phase boundary. Optimizer and RNG
state are preserved when supported; a neural adapter must also preserve its target
networks and any other inference/training buffers.

Interaction returns describe data collected while the policy may be changing.
Incomplete episodes are labeled partial. CLI and example runs save every endpoint
checkpoint, verify its readback, and restore from those files before any frozen
scoring episodes begin. They use separate seeds; only those complete episodes enter its clean
frozen-evaluation result. Neither set of returns feeds into the finished offline
selection. Using them to choose later configurations constitutes development
feedback and needs a new held-out study. Python callers may omit the optional
`checkpoint_directory` for an in-memory test; its receipt explicitly labels the
checkpoint as an in-memory snapshot, not a saved-file evaluation.

Supported adapters expose complete immutable inference state and exact update
counters. The runner checks those declarations, state hashes and instrumented
calls. This is a cooperative API contract, not an OS sandbox or execution
attestation for arbitrary external code. The original strict BC evaluator and
its tested guards remain unchanged. No filesystem or security permissions change.

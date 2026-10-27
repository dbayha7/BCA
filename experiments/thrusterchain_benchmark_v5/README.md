# ThrusterChain benchmark v5 — validation build

This is a separately versioned development copy of the v4 benchmark, package
version 0.3.0. The v4 transfer bundle and its historical studies are unchanged.
V5 is not yet a completed multi-algorithm research benchmark. Strict-offline
adapters for the repository-native TD3+BC, IQL, CQL and ReBRAC hosts have
bounded tests; unified UniFloral is not integrated.
The runnable standalone example remains behavior cloning. None of the checks
establishes algorithm performance.

The default finite task has 14 observations including remaining time and ends
after 100 decisions. The continuing task instead exposes 13 physical features
and a constant zero padding coordinate; it truncates at the collection boundary
and bootstraps from the true post-step physical state. V5 datasets have a v3
schema and are not silently interchangeable with v4 datasets. V5 transition
batches include episode-local `next_actions` and `next_action_valid`. A true
termination, time-limit truncation, and missing recorded next action are distinct
conditions. Adapters must explicitly implement their own required masking.

The shifts vary initial-condition laws of a deterministic synthetic oscillator.
The larger-kick law overlaps the recorded law. The controlled action-box mixture
can produce rare or absent action regions; its known requested-action density is
not a trajectory occupancy ratio. These tests do not establish performance on
stochastic dynamics, real robots or every offline RL failure mode.

`thrusterchain/comparison.py` checks declared seed partitions and summarizes
per-policy means across declared training seeds; it cannot establish independence
of fits or whether final results were previously inspected. Freeze policy selection, budgets and
evaluation before collecting final results. State separately any simulator
episodes used for selection, and report raw returns and all attempted runs.

## Validation

With Python 3.10+ and the dependencies in `requirements-tested.txt`, run:

```text
python -B validate_v5.py
```

This runs the inherited and new tests applicable to v5 and explicitly skips the
classes whose assertions require missing, immutable v4 historical smoke artifacts
and their original v4 loader. The test count and skipped classes are printed;
this is not a pass for the unavailable archival tests. Historical evidence should
be checked against its registered v4 source and exact artifacts, not recreated as
v5 evidence. For unfiltered discovery use `python -B -m unittest discover -s
tests -v`; it currently requires those original files and source contracts.

`host_integration/` contains bounded validation probes; inspect its documentation
for exact host, task adaptation, update counts and limitations. No full training,
sweep, scientific comparison or final holdout result is implied by those probes.
The four host adapters require this BCA repository and a compatible JAX runtime;
they are not part of the standalone wheel's default BC command. Their eager
updates have not been profiled at research-scale budgets.

For a bounded strict-offline integration check, with the repository root and v5
directory on `PYTHONPATH`, a caller can use a validated v5 dataset as follows:

```python
from thrusterchain.datasets import load_dataset
from thrusterchain.training_protocol import OfflineBudget, EvaluationConfig
from thrusterchain.training_runner import train_offline, run_mode
from host_integration.td3bc_adapter import TD3BCAdapter

data = load_dataset("path/to/v5_dataset.npz")
checkpoint, receipt = train_offline(data, TD3BCAdapter,
    budget=OfflineBudget(updates=2, batch_size=16), seed=123)
result = run_mode(data, checkpoint, TD3BCAdapter,
    config=EvaluationConfig(frozen_policy=True, episodes=1, seed=456),
    families=("recorded",))
```

The other fixed adapters are `IQLAdapter`, `CQLAdapter`, and `ReBRACAdapter` in
their matching modules. This two-update example is a plumbing check. The code
does not impose a research study's full pre-registration or prove holdout status.
The inherited `package_release.py` is disabled because its v4 release claims do
not describe this development copy. The inherited audit receipts and `REVISION_V4.md`
describe their historical snapshots, not validation of v5.

Before a research-scale comparison, validate each actual learner and its target
and update schedule, select dataset/action/reward transforms and budgets,
record all development feedback, use independent trained seeds, and establish a
genuinely unexposed final evaluation. Preserve raw returns and dataset/host hashes.
The scenario and budget template is [benchmark_protocol_v5.md](docs/benchmark_protocol_v5.md).
The checks executed on this source are recorded in
[VALIDATION_20261026.md](validation/VALIDATION_20261026.md).

# ThrusterChain v4: training budgets and explicit adaptation

Package **0.2.0** adds a reusable training interface while preserving v3 physics,
14-dimensional observations, strict BC evaluation and prior results. The default
iterative offline budget is **1,000,000 primary optimizer updates**, with a
**3,000,000 cumulative offline maximum**, including resumed work. For a TD3-family
adapter, the primary optimizer is the critic. Actor and auxiliary steps are
reported separately, together with their total; they do not reduce the critic
budget.

This is a verified software extension with small-budget demonstrations. No
million-update run, neural BCA integration, newly trained nine-cell benchmark,
adaptation benefit or new coverage guarantee is claimed.

## Use and extend

- [Training modes and runnable CLI](docs/training_modes.md)
- [External host adapter interface](docs/host_adapter_interface.md)
- [Exact fields and machine-readable contract](contracts/training_modes_v1.json)
- [Current source manifest](CURRENT_SOURCE_MANIFEST.json)
- [Preserved v3 scientific smoke report](MEETING_REPORT_V3.md)

The default CLI remains fast closed-form BC. Its budget receipt explicitly marks
optimizer updates inapplicable, and an explicitly supplied gradient budget is
rejected. The identified-model planner also has no gradient-training or online
adaptation implementation. Neither runs repeated solves to imitate a large
optimizer budget.

`--adapter affine-sgd` selects a genuine NumPy SGD/momentum imitation example. It
uses fixed Train normalization and exposes the same protocol expected of a future
neural host. Its online update imitates replay actions; it does not learn a reward
maximization objective. External neural adapters remain a separate integration.

Frozen policy is ON by default. OFF requires explicit online interactions,
primary updates, replay mixture, batch size and learning settings. Every scenario
restores a fresh instance of the same pretrained checkpoint. The CLI and example
save all endpoint checkpoints, verify their readback, and restore them before
any clean frozen scoring episode. Adaptive interaction returns and partial
episodes remain separate. Zero updates are labeled **no online learning**;
positive interaction budgets can collect without parameter updates.

Checkpoint resumption preserves full learner state and the offline sampling RNG.
The reference SGD adapter has no target network and explicitly declares that
component inapplicable. The independently implemented toy host probe includes
target state. Online endpoint checkpoints preserve learner state, but resuming
an interrupted online trajectory/replay buffer exactly is not implemented.

## Evidence on the delivered source

Core source SHA-256:

`6f36e2714f6012fbc71a9315b19ab995f0b7d57d96026a1e2d97ef66d66d6596`

| Check | Actual result | Receipt |
|---|---|---|
| Full pinned suite, Python 3.10 / NumPy 2.2.6 | 209 passed; zero failures, errors or skips | [Full suite](audits/revision_v4_final/full_suite_attempt02/receipt.json) |
| Alternative runtime, Python 3.12 / NumPy 2.3.5 | 38 CLI/protocol/adapter tests passed | [Compatibility](audits/revision_v4_integration_audit/numpy235_persisted_release/receipt.json) |
| Independent runner/protocol review | 25 checks passed | [Independent review](audits/revision_v4_independent_review/final_attempt02/receipt.json) |
| Independently implemented host | 5 critic + 2 actor steps; exact serialized 2+3 resume; online 2 critic + 1 actor after 3 interactions | [Host probe](audits/revision_v4_integration_audit/external_host_persisted_release/receipt.json) |
| Ten topical acceptance passes | All ten passed against this source | [Ten passes](audits/ten_pass_revision_v4/ten_pass_review.json) |
| Offline wheel build and isolated target install | Package 0.2.0, source matched, both original notices retained | [Wheel receipt](audits/revision_v4_wheel/receipt.json) |

These are software and numerical checks, not independent scientific training
replicates. The alternative-runtime check does not cover the full SciPy suite.
The independent toy host establishes interface behavior, not neural RL efficacy.

## End-to-end records

The final [training example](results/revision_v4_training_example_final/completion.json)
fits an 8-update pretrained affine SGD policy. A separate 3-update resume matches
an uninterrupted 11-update run byte for byte, including optimizer, normalization,
RNG and sampler cursor. This comparison executes 22 offline optimizer calls across
the original, resumed and uninterrupted verification branches, with each branch
recorded separately.

For each of four scenarios, adaptation restores that same 8-update checkpoint,
collects 12 transitions and executes 6 additional primary updates. That is 48
online interactions and 24 online updates in total. Each 12-step interaction
segment is explicitly partial. Strict, adapted and zero-budget branches each
receive 2 clean frozen episodes per scenario: 24 complete evaluation episodes
across the three branches. The zero-budget endpoints retain the original 8-update
state. Configurations were saved before fitting/scoring; no score selected a
checkpoint or changed a budget.

- [Registered configuration](results/revision_v4_training_example_final/registered_config.json)
- [Offline update receipt](results/revision_v4_training_example_final/offline_training_receipt.json)
- [Exact resume comparison](results/revision_v4_training_example_final/resume_comparison.json)
- [Strict frozen records](results/revision_v4_training_example_final/strict/results.json)
- [Adaptive interactions and separate frozen records](results/revision_v4_training_example_final/adaptation/results.json)
- [Zero-budget control](results/revision_v4_training_example_final/zero_online_budget/results.json)

The fresh [original baseline workflow](results/revision_v4_smoke_final/benchmark_results.json)
uses 40 fixed offline episodes and evaluates four methods on four families with
two episodes each. All 32 control CSV records, all 16 method/family summaries,
the dataset bytes and the terminal diagnostic match v3 exactly on this host.
[Closed-form budget receipts](results/revision_v4_smoke_final/training_budget_receipts.json)
separate fitting semantics from optimizer budgets. All 48 earlier result files
and the v1, v2 and v3 archives retain their original hashes:
[preservation receipt](audits/revision_v4_final/preservation_and_results.json).

The earlier v4 folders without `_final` are retained candidate evidence from
before saved-file evaluation was added. Their receipts identify that earlier
source; use the `_final` folders above for the delivered implementation. The
two-step sanity appendix and historical 45-run study remain unchanged, with
their own earlier source-bound evidence.

## Reproduce

With the dependencies in `pyproject.toml`, from this folder:

```sh
python -B -m unittest discover -s tests -v
python -B run_training_example.py --output results/my_training_example --dataset results/smoke_v2/offline_dataset.npz
python -B run_benchmark.py --output results/my_strict_smoke --episodes 40 --evaluation-episodes 2 --planner-horizon 12
python -B audit_benchmark.py --output audits/my_acceptance --run results/my_strict_smoke
```

Use new output destinations. Full commands, runtimes, logs and source identities
are retained in the linked receipts. The example intentionally overrides the
million-update default with a small smoke budget.

The execution boundary remains cooperative instrumentation and declared state
checks. It does not authenticate arbitrary adapter code or create an OS sandbox.
No permissions, historical artifacts, public sharing or release approvals were
changed. Existing ownership and publication approval questions remain open.

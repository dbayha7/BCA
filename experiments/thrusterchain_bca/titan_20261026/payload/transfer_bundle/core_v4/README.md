# ThrusterChain benchmark

**Revision v4 (package 0.2.0):** explicit optimizer budgets and separate strict-frozen versus offline-to-online modes. See [training modes](docs/training_modes.md), the [external host adapter interface](docs/host_adapter_interface.md), and [revision evidence](REVISION_V4.md). The previous v3 implementation and every prior result remain preserved. Earlier audit receipts certify only their recorded source snapshots.

The [current source manifest](CURRENT_SOURCE_MANIFEST.json) identifies the core implementation. The previous archive manifest is preserved under [v3 basis evidence](audits/revision_v4_basis/BASIS_ARCHIVE_MANIFEST_V3.json); it does not describe the revised source.

**Separate evaluator appendix:** [Two-step decision sanity check](docs/two_step_decision_sanity.md) evaluates 18 synthetic fixed-bank queries with a fixed zero continuation. It shows state-dependent nonzero best bank actions, evaluates zero learned methods, and leaves terminal coverage unchanged. Its source and checks have separate receipts from the core suite.

An inspectable offline control benchmark with a real simulator, explicit data protocols, and separate state/action coverage diagnostics. This release prioritizes correct contracts and useful baselines over any particular algorithm winning.

**Start here:** [current budget and mode extension](REVISION_V4.md), [v3 scientific smoke results](MEETING_REPORT_V3.md), [Regular mode](docs/regular_mode.md), [environment contract](docs/environment.md), [coverage/calibration](docs/coverage_and_calibration.md), [bounded references](docs/reference_controllers.md), [historical actor audit](docs/historical_actor_audit.md). The [v3 explorer delta](audits/revision_v3_final/explorer_delta.json) remains valid for those preserved scientific results; the [original interface](contracts/explorer_interface.json) remains an explicitly v1 compatibility artifact.

## Run the example

Use Python 3.10+ with the dependencies in requirements-tested.txt. This folder is self-contained; it does not import the original repository at runtime. With dependencies available:

    python -B -m unittest discover -s tests -v
    python -B run_benchmark.py --output results/my_smoke --episodes 40 --evaluation-episodes 2 --planner-horizon 12

On Zeus, the verified existing interpreter and Gymnasium path are:

    $env:PYTHONPATH='C:\Users\David\.cache\bca-shift\py'
    & 'C:\Users\David\dev\BCA\.venv\Scripts\python.exe' -B -m unittest discover -s tests -v
    & 'C:\Users\David\dev\BCA\.venv\Scripts\python.exe' -B run_benchmark.py --output results/my_smoke

Choose a new output directory each time. Existing datasets, policies, receipts and runs are never overwritten. The package can also be installed in a new isolated environment with pip install ., but fresh-host installation is not part of the verified evidence.

For separate collection, offline training and evaluation processes:

    python -B -m thrusterchain generate results/staged/data.npz --episodes 40
    python -B -m thrusterchain validate results/staged/data.npz
    python -B -m thrusterchain train results/staged/data.npz results/staged/frozen
    python -B -m thrusterchain calibrate results/staged/data.npz results/staged/frozen results/staged/calibration.json
    python -B -m thrusterchain evaluate results/staged/frozen results/staged/evaluation.json --learning disabled

The default train command fits, selects and freezes closed-form BC without simulator rollouts. Its optimizer-update budget is inapplicable. Use `--adapter affine-sgd` for the separate iterative protocol example; its default is 1,000,000 primary optimizer steps and its maximum is 3,000,000. For a short runnable test, explicitly pass `--offline-updates 8`. Frozen evaluation is on by default. See [the complete toggle and adaptation example](docs/training_modes.md) before switching it off.

## Two distinct uses

**Regular mode** creates a fixed episode dataset, fits behavior cloning with train-only preprocessing, selects its ridge using an offline validation split, saves a freeze receipt, performs fresh final calibration, and evaluates the frozen policy in the simulator. Random control, a privileged true-model constrained planner, and a train-only identified-model planner use paired evaluation starts. Three shifted families are absent from offline collection. Live scores are for reporting; using them to retune changes the selection regime to simulator-assisted.

**Controlled diagnostics** draw independent state/action queries from nine broad/rare/absent designs. Their exact densities describe these designed queries. They do not describe trajectory occupancy. Terminal deterministic Q calibration is narrowly scoped to the final decision of the finite objective. It is not a general neural Q or posterior guarantee.

## What the results mean

The bundled run is an integration smoke profile, not a completed nine-cell policy benchmark. A shorter MPC horizon is labeled as a receding-horizon reference. The bounded planner reports KKT residuals and float32 rounding, without claiming the exact float-lattice optimum. Returns are raw and discounted, not fabricated normalized scores.

The historical 45-run study is preserved as development evidence, with its exact actor equation and inference audit reported separately. Artifact provenance and transfer history are retained in the audit receipts.

## Navigation

- thrusterchain/: reusable environment, dataset, policies, bounded planning and diagnostic code.
- contracts/: machine-readable environment, physics and explorer interfaces.
- results/: immutable generated datasets, receipts and smoke reports.
- tests/: invariant, schema, protocol, numerical and calibration checks.
- audits/: exact source hashes, historical checks and review passes.
- docs/: definitions, interpretation, limitations and research positioning.

No existing studies were overwritten. A manifest establishes traceability, not an institutional compliance certificate. [Research positioning and remaining publication gates](docs/research_positioning.md).


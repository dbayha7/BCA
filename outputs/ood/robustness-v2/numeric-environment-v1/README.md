# Exact numeric CPU environment: narrowly accepted

The new versioned numeric profile passed one independently observed real
Python3.10 / NumPy2.2.6 / JAX0.6.2 CPU import route. The child exited0; separate
stdlib review of its saved evidence also exited0. This closes the observed
environment additions for these numeric imports. It does not accept native
simulator imports or the full scientific execution route.

The profile explicitly supplies eight fields at launch:

| Field | Exact value |
|---|---|
| CUDA_VISIBLE_DEVICES | empty |
| JAX_PLATFORMS | cpu |
| XLA_PYTHON_CLIENT_PREALLOCATE | false |
| OMP_NUM_THREADS | 1 |
| OPENBLAS_NUM_THREADS | 1 |
| LC_CTYPE | C.UTF-8 |
| TF_CPP_MIN_LOG_LEVEL | 1 |
| TPU_SKIP_MDS_QUERY | 1 |

No inherited environment is merged. No field is stripped or changed after
launch. The kernel environment and effective Python environment exactly matched
the profile at all four phases: Python startup, NumPy import, JAX import and CPU
backend initialization. The parent separately observed argv, environment and
PID/parent/start/session/group identity before and after imports. Those
observations match. Runtime/source pins were checked before and after the route.

The new NumericEnvironmentGuard refuses missing/extra/changed fields, policy
mutation, fork/process changes, skipped/repeated phases and premature/repeated
completion. Refusal stays poisoned after a field is restored. Sixteen new pure
guard tests passed with no skips; no preimplementation red was captured. Phase
labels alone do not prove imports: acceptance here also binds the exact observed
probe source and saved process receipts. This is not a Python sandbox.

The contract was reviewed and saved before the single import-only probe. The
original five-field engineering capsule, worker route and supervisor remain
frozen and unchanged. This numeric profile has not been integrated into them;
using it there requires a separately versioned and reviewed declaration/route.
It is not an unrestricted environment override or a native-runtime acceptance.

The earlier key-producing process remains actualexit1, and all saved keys remain
held from execution. Its closed generator, independent key review and failed
gate were not rerun. This probe generated no keys, candidate pools or outcomes.
It does not retroactively supply the failed producer's missing final environment
or turn that exit into0. A separate documented disposition is still required
before those independently checked key values may be consumed by real execution.

A read-only installed-source/cache inspection also pinned the existing Python3.10
Linux CPU mujoco_py extension. Ordinary mujoco_py import checks LD_LIBRARY_PATH,
chooses CPU/GPU builder based on environment, acquires a package build lock and
can rebuild the extension. Discovery additionally depends on the explicit
MuJoCo path or home-directory semantics. None of those paths was executed here.
The ELF dependency graph, exact native import route and any rebuild refusal need
independent acceptance; a cached filename/hash alone is insufficient.

No gym/mujoco_py/d4rl/flax/torch scientific stack, checkpoint, model, simulator,
shared resource lease, archive or extension ledger was imported or constructed
by the probe. JAX's optional CUDA plugin discovery stderr retains
CUDA_ERROR_NO_DEVICE; the selected backend is CPU and the probe exited0.
The full native/runtime/storage/ownership/reviewer and scientific gates remain
pending. Every later physical engineering call still needs prior independent
engineering acceptance and durable reservation; action1e-6/reward1e-7/full-state
repeat and separate global precommit/scientific acceptance remain unchanged.

All prior closed v2 sources/receipts, accepted training artifacts, approved
amendment and key-diagnostic evidence remain unchanged. The local training
mirror matches107/108 original manifest files; only the separately reviewed
advisor-only .gitattributes append differs. No scientific source exception is
granted. No closed test suite, training audit or key review was rerun.

This package retains exact policy bytes (including their original line endings),
source, all four child observations, two parent observations, intent, stdout,
stderr, actual exits and independent reviews. It contains no weights, key-table
binary arrays, SQLite, NPZ or native libraries. The numeric profile and its tests
are published separately as new code; original closed components remain frozen.

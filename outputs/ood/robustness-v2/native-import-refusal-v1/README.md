# Cached native import route: stopped during NumPy startup

The single real import-only child exited1 on September29 at00:15:53.051757UTC.
The parent transport also exited1. The new guard stopped NumPy import before its
phase completed; JAX and mujoco_py imports were never reached. No model, simulator,
key, physics, original shared resource lease, extension ledger or package build
lock was used. The probe has not been rerun, and its raw failure evidence remains.

A separate stdlib pre-review exited0 before dispatch. It permitted only one
instrumented import observation under the explicitly declared12-field environment.
The parent independently observed the child's PID, parent, start ticks, session,
group, isolated argv and exact launch environment once, before release. The child
recorded the same12 kernel/effective fields at Python startup. There is no second
parent observation or completed NumPy/JAX/native stage, and none is inferred.

The new `native_import_guard_v1.py` passed24 new tests with no skips. Nine isolated
fixtures exercised actual Python audit/profile hooks using fake source/cache files
and temporary locks. They refused rebuilding before its function body, changed live
builder code, subprocess launch, unauthorized file creation/deletion, environment
mutation and lost profiling. Pure cases covered path identity, mutation, alias,
fork and phase-order refusals. No preimplementation red was captured. All fixture
commands, actual exits and raw output are retained in the test stderr receipt.

The guard binds exact installed builder/cache/package-lock identities and compiles
the builder source without executing it to recognize its code objects. It refuses
known rebuild/discovery function calls before the body and blocks process launches,
socket operations, unauthorized Python file mutations and environment changes.
The allowed package-lock route is append-open of the existing file, a mkdir check
of its existing parent, and whole-file nonblocking lock/unlock on that exact file.
All guard bindings,91 runtime pins and the source bytes were checked before launch;
the saved child result reports all91 runtime hashes unchanged afterward.

This is cooperative instrumentation, not a Python or native-code security sandbox.
Native C syscalls, arbitrary interpreter tampering and complete transitive dependency
semantics are outside its guarantee. Passing the temporary fixtures does not accept
the real package-lock, native loader, simulator or scientific execution route.

The new failure exposed a limit of the earlier numeric environment evidence.
Installed NumPy2.2.6 `numpy/_core/__init__.py` iterates over OPENBLAS_MAIN_FREE and
GOTOBLAS_MAIN_FREE. For each absent name it temporarily sets1 before importing
multiarray, then removes only the fields it added. Both names were absent from the
declared12-field profile. The actual traceback points to that assignment at line19;
the guard refused it. A subsequent source-only read retained the exact5612 bytes,
SHA256 `1fde7ecd24f4087ea99e739b8d75175e23e0b6e6fd337e4804669813eab8c2b5`.
No diagnostic reimport was performed.

The guard's profile poison check replaced the original refusal message with
`Previous import refusal.` The exact rejected environment-field argument was not
saved. The ordered two-field source behavior is established by the pinned source;
it must not be described as two captured mutation events or a captured first-field
argument. A future separately versioned guard must preserve its first refusal
reason before applying poison checks. The current source and receipts stay frozen.

The earlier numeric probe still proves equality of its eight environment fields
at four recorded stages. It did not observe every intermediate instruction and
cannot establish that no temporary environment mutation occurred between those
snapshots. This new scope clarification supersedes any broad reading of the earlier
"no field changed" wording. The old report, successful exit and sources are retained
unchanged; it was not rerun or retroactively turned into a failed process.

Independent stdlib review of the saved failure exited0 at00:18:05.746199UTC. It checked
the prereview/source relationship, actual child/transport exits, initial parent
identity, two journal records, unchanged initial/final effective environment,
saved maps,24-test receipt and exact NumPy source behavior. It confirmed that the
builder-call and permitted-write logs are empty, JAX/mujoco_py/GLFW are absent from
the saved module inventory, and cymj/GLFW binaries are absent from the saved maps.
This accepts the stopped-process evidence; it does not accept native imports.

The next implementation must explicitly account for these temporary NumPy fields
in a separately reviewed launch/transition contract. Source suggests predeclaring
both as1 would avoid this particular add/remove branch, but this has not been
executed or accepted, and additional import behavior remains unknown. No field
has been silently added to any accepted profile and no failed probe was retried.

The original scientific capsule/supervisor/worker remain unchanged. Native imports,
full source/runtime graph, native fields/contextmanagers, actual storage paths,
worker-owned original shared lease and the only-route integration remain pending.
No extension or physical engineering may start before independent engineering
acceptance; durable reservation for every physical call, action1e-6/reward1e-7,
full-state repeat and separate pairwide precommit/science gates remain required.

The prior key producer remains exit1, with all40 saved tables held from execution.
No closed key generator/review, test suite or training audit was rerun. Accepted
training remains10 runs/five pairs. No cluster probe or other-chat access occurred.
The approved single-stream amendment stands; no scientific parameters were tuned.
All prior frozen evidence and original plan remain unchanged. The local training
mirror is107/108, with only the previously reviewed advisor .gitattributes append.

The publication contains new guard/test source and small exact-byte evidence,
including partial journal, maps, actual exits and independent review. It contains
no native binaries, weights, key arrays, SQLite or NPZ. The runtime refusal warrants
one notification; it is an engineering failure, not a new scientific comparison.

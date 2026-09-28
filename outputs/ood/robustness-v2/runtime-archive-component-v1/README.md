# V2 runtime archive integration

Status:24 synthetic storage/integration tests passed, actual exit0/no skips.
Tests use temporary Linux SQLite files, synthetic data/dynamics and a synthetic
resource ancestor. No original resource DB, real shared lock, weights, model or
native simulator was opened. This is not actual execution acceptance.

`runtime_archive.py` wraps the frozen ArtifactArchive through injection. It
requires a fresh absolute unaliased path, a held lifetime lease, pinned source
bytes and explicit file/free-space bounds. It refuses existing archives and
sidecars rather than recovering them. Every operation checks process ownership,
lease, source identity, file identity and SQLite FULL/DELETE durability settings.
The underlying numerical and archive sources are unchanged.

An append returns only after SQLite commits, explicit file and parent-directory
fsyncs complete, and exact raw readback agrees. This same wrapper serves native
ArchiveDirectory input/applied/output writes and the new EncodedStore full-state
and completion records. Partial raw streams remain visibly incomplete. Any
failed append, fsync, readback or ownership check poisons the instance and prevents
further writes. Committed or partial bytes stay intact; there is no retry,
deletion, repair or reopen interface. A failed call remains charged and pending
through the separate extension ledger.

The cap check conservatively allows raw record size plus64KiB of SQLite overhead
before each append, also checking resulting file size. This may refuse early
near a declared file cap. It is not a quota extension or a guarantee that the
whole future campaign fits on the final device. The free-space floor and cap
must be set in the future independently accepted execution declaration.

The tests execute exact AST-extracted bodies of the three frozen archive classes,
externally bound to their existing source hash, with only the required pure
serialization dependencies. This avoids scientific module imports. It exercises
the SQLite implementation, not a memory substitute, but does not establish that
the eventual imported runtime factory has all accepted globals and identities.
That binding remains an explicit runtime gate.

Integration checks establish that a separate read-only SQLite connection sees
both constructor input and intercepted-control artifacts before the synthetic
physics callback. Complete returned native contents are stored before the fake
ledger acknowledgment. Another integration combines the real closed extension
component with a temporary synthetic ancestor and lock: one synthetic recorded
step commits seven raw/typed artifacts and one1env/4physics completion while the
ancestor stays byte-identical. An fsync failure before a synthetic step leaves
its reservation pending and prevents the callback.

Tests also cover exact typed values, failed/partial writes, mismatched readback,
source/file replacement, hardlinks/symlinks/sidecars, forked or lost ownership,
mutated SQLite durability settings, space limits and refusal to reopen. A child
process deliberately exits23 after an acknowledged append; its exact saved
bytes are visible through a read-only connection and execution reopen is refused.
This is a process-death fixture, not a host power-loss test or a scientific exit.
No existing test suites were rerun; prior synthetic fixture helpers were reused.
The expected missing-module failure and18-/21-test development passes are kept.

Remaining work: actual runtime/declaration/checkpoint/data/support/native-schema/
fresh-stream/path binding, global state/candidate/key/warning precommit barrier,
supervisor and independent engineering acceptance. Actual filesystem capacity and
storage performance must be measured for the selected execution path; synthetic
fsync success is not proof of device power-loss behavior. All real engineering
calls must be separately accepted and reserved before physics. Full native
action1e-6/reward1e-7/restore/repeat gates precede scientific dispatch. No real
extension was created and no OOD comparison or global readiness is claimed.

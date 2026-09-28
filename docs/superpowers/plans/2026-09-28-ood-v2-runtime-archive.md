# OOD v2 durable runtime archive integration

Goal: integrate the closed native archive interface with the new constructor and
recorded-step evidence stores under an existing exclusive lease. Preserve frozen
numerics and archive source. This is synthetic storage integration, not permission
to create scientific resources or call a model/simulator.

Create runtime_archive.py as an injected wrapper over the frozen ArtifactArchive.
Require a fresh absolute non-redirected path, an externally pinned archive class
source, a held lifetime lease, explicit storage bounds and free-space floor.
Refuse reopen/recovery, sidecars, modified ownership and archive replacement.
Check SQLite DELETE/FULL settings; fsync file and parent directory and verify
exact appended readback before returning to the simulator or ledger callback.
Any storage failure poisons the wrapper and leaves committed/partial bytes intact.

Test the exact pinned archive class definitions extracted by AST with only their
required pure serialization dependencies, in temporary Linux files. Exercise
ArchiveDirectory raw pre-physics writes and EncodedStore full-state writes through
the same wrapper, plus injected constructor and recorded-step integration. No
actual native imports/physics, model, original ledger or shared resource lock.
Save and retain expected missing-module failure before implementation, actual
tests, source pins and failures. Publish owned code/evidence after integrity checks.

Explicit limits: SQLite/fsync ordering on a synthetic temporary filesystem is not
proof of final-device power-loss durability or complete campaign storage capacity.
Actual path/device/declaration/runtime/source/class binding, global precommit
barrier, supervisor and independent engineering/execution acceptance remain.

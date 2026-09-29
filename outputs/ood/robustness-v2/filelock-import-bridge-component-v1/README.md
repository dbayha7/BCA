# Filelock import-entry bridge: component fixtures and current source aliases

The new bridge observes a capability function called during ordinary importlib
module execution, without invoking or replaying that function. Twenty-five new
isolated fixture tests passed with actual exit 0 on September 29, 2026 at
01:36:56.452117 UTC. A separate stdlib-only process reviewed the saved evidence
without replaying fixtures and exited 0 at 01:39:06.521658 UTC. This closes the
component fixture boundary only. No new real filelock/JAX/MuJoCo import probe ran.

## What the bridge observes

`filelock_import_bridge_v1.py` composes the unchanged, closed
`filelock_scratch_v1.py`. It compiles the declared module source without executing
it, requires one capability function and its direct module-level assignment,
and installs profile/audit observation before loading the fixture module. It
binds the registered module namespace, full module code, callback code, direct
callsite, defaults, selected globals, source identities, process, TMPDIR and
scratch root. It also checks the closed parent class's live method objects and
code against its pinned source. This is not a transitive global-graph check.

On the actual callback entry, the bridge attaches the existing scratch observer
to the function already executing. It never calls the callback or `scratch.run`
and never substitutes a capability result. The successful fixture has four
boundaries: module entry, callback entry, callback return, module return. It
observes the exact 12-event filesystem lifecycle: default-tempdir probe and
four-byte `blat` readback, one temporary directory, empty probe source, one hard
link with matching device/inode and link count 2, then exact descriptor-bound
pair deletion and directory removal. The empty root is retained.

The successful module contains the exact installed
`filelock._strict._probe_link_follow_symlinks` source text and imports real
os/tempfile/Path helpers with actual os.link availability. It is still a synthetic
module loaded through importlib, not the complete filelock package or its real
imported namespace. Negative fixtures intentionally alter code or state to test
refusal. No installed dependency was edited or pre-imported outside the guard.

The component refuses extra/repeated entry or completion, changed module,
function, code, defaults, globals, parent method, source, callsite or contract,
outside writes, environment/process/network/lock routes, incomplete cleanup,
changed capability result, and swallowed refusals. The first refusal survives
later checks, with audit-event arguments retained as strings when available.
New audit/profile and at-fork registrations are refused. For at-fork registration,
the profile event saves the caller filename/line only; it does not expose the
actual callback keyword arguments. The sys.addaudithook audit event has empty
arguments. Neither diagnostic is a callback/state binding.

## Tests and retained evidence

Final source SHA256:

- `filelock_import_bridge_v1.py`: `fb18f9d069858ddc5f6ec2ce8624974b6a0d65f39a2f1dcfc918056493029425`
- `test_filelock_import_bridge_v1.py`: `e844c31508c9aed464313ddffa3995e457e78922f35c17f408cb55f2bd826e38`

The final 25 children comprise 24 actual-zero expected pass/refusal cases and
one deliberately actual-23 process death inside TemporaryDirectory. Independent
review found that child's retained empty temporary directory. Failed fixtures
keep the hooks and use os._exit to avoid unguarded finalizer cleanup. Persistent
paths are under
`/home/dbayha/bca-work/ood-import-bridge-fixtures-v1/monitor_20260929T012803Z/fixture-*`.
Raw commands, child exits, stdout/stderr, exact fixture sources and source pins
are preserved in the package. No closed suite, prior native probe or key review
was rerun. There was no preimplementation red run and none is claimed.

The earlier 21-test development pass (actual 0 at 01:35:21.222136 UTC) and exact
code bytes remain saved. Subsequent review tightened ongoing contract,
function/default and parent-method checks and added four cases before the final
25-test pass. No failed test or scientific retry occurred in this component.

The test harness creates the root before the bridge and supplies its inherited
environment plus TMPDIR. This does not establish an accepted 15-field launch
profile or a real worker's scratch creation/ownership contract.

## Concrete shared-source provenance

Source-only inspection exited 0 at 01:30:55.732617 UTC. It examined 24 bounded
candidate source paths and four Conda metadata records, without importing the
target packages. For each of os.py, pathlib.py, shutil.py and tempfile.py, all
three observed hard links share one current device/inode, exact bytes, link
count 3, uid/gid 1000 and mode 0644. They are in:

- `/home/dbayha/miniconda3/envs/corl-orig-local/lib/python3.10/`
- `/home/dbayha/miniconda3/envs/sdbca/lib/python3.10/`
- `/home/dbayha/miniconda3/pkgs/python-3.10.20-h741d88c_0/lib/python3.10/`

Both environment records identify that package-cache source and hardlink mode;
package entries match the four exact byte hashes and sizes. The alternative
Python package candidate has matching bytes but a different inode and is not
counted as a fourth alias. Independent review rehashed/restatted all 12 alias
paths and all four metadata records. This establishes current provenance at
inspection/review, not immutability, absence of races or protection from later
changes through another alias. The generic unaliased guard remains unchanged;
no exception for scientific execution is accepted.

The installed filelock `_read_write.py` remains SHA256
`3c194b83f7db92a0dca51c89e4d91982858b47d0c450733ed6c43306766c383f`.
Saved source binds its audit callback, at-fork callback, transition context,
database registry and connection escrow declarations. The source indicates that
CPython makes `_IS_PYPY` false, so the shown `note_sqlite_use` mutation branch is
inactive; this is source inference, not live registration/state acceptance.
The fork callback exits 70 if the thread-local SQLite transition depth is
nonzero. No SQLite connection or real callback registration was exercised here.

Inspection source SHA256:
`ab6892112f17be154ff264dd7016459d49c0d3b752453ace1c00e1119dae5f3c`.
Independent review source SHA256:
`3588ea99a8c5c62e0f0f5a881e0d8d907e77617403f1c0de80ded82e2ee8fde1`.

## Remaining execution boundary

The bridge is cooperative Python instrumentation, not a security sandbox or
native-C syscall monitor. Full live stdlib code/global dependencies, all imported
filelock modules and audit/at-fork callback state still need independent binding.
It has not been composed with the closed native-import guard or scientific
supervisor. The old fourteen-field contract remains unchanged and passed only
Python/NumPy snapshots; no fifteen-field real profile was executed or accepted.

Before one separately versioned import-only observation, bind those live
dependencies and registrations, compose one observing route without replay,
declare the exact environment plus TMPDIR and fresh worker-owned scratch
creation/cleanup/failure retention, and independently review the complete new
contract. Do not globally allow temp writes, fake a capability, change installed
sources or retry either closed native probe.

All prior source/receipt, approved amendment and training evidence stays frozen.
The failed key producer remains actual 1 and its 40 saved tables remain held.
No scientific shared lease, extension ledger, archive, model, simulator, physics
or new keys were used. Training acceptance remains 12 runs/six pairs, with all
five paired seeds required per cell. No new scientific comparison is reported.
Full native/storage/worker ownership engineering gates and subsequent reserved
physical checks and independent pair-wide science gates remain required.

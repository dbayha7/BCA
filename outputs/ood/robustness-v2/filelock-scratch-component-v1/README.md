# Filelock capability-probe scratch lifecycle

The new filelock_scratch_v1.py component passed29 tests at September29,
00:59:44.819500UTC. Independent saved-fixture/source review exited0 at
00:59:51.542915UTC. This closes a narrow filesystem fixture prerequisite for the
previously refused filelock import. No filelock, JAX or MuJoCo package import was
performed; neither closed native import probe was rerun. Native environment and
scientific execution acceptance remain false.

The successful fixture executes the exact compiled function code from installed
filelock._strict._probe_link_follow_symlinks, with explicitly injected real
os/tempfile/Path globals and the actual os.link availability. It does not execute
the package module or pretend that this namespace is the real imported namespace.
The callback's code, globals, defaults, source path and bytes are bound. The
component also checks supplied tempfile/shutil/pathlib/os source bytes/stat/link
counts, selected helper/API object identities, owner PID and a fresh empty0700
scratch directory. Complete live stdlib code/global dependency binding is still
required; helper object identity alone does not establish it.

Only the observed lifecycle is permitted: a random default-tempdir probe file,
its four bytes `blat` checked before unlink; one random temporary directory; an
empty probe-source; one probe-link hard link with shared device/inode and nlink2;
then descriptor-bound cleanup of exactly those two empty files and that directory.
The directory's owner/mode/identity is checked throughout. Twelve audit events
cover this successful route. The caller-owned scratch root remains empty afterward.
Unexpected operations/order, outside paths, aliases, extra probes, subprocesses,
network/environment/lock operations, changed source/callback/globals/API/ownership,
reentry, repeated completion and swallowed refusal poison further use. First
refusal event/arguments are retained as strings. Native C writes, arbitrary
interpreter mutation and unaudited operations are outside this cooperative
instrumentation's guarantee; it is not a security sandbox.

All29 final tests run isolated children. Twenty-eight exit0 after checking success
or expected refusal. One deliberately exits23 while inside a newly created
TemporaryDirectory. A separate reviewer observed that directory still present,
empty and uncompleted. Failed fixtures retain their hooks and exit directly to
avoid a later unguarded finalizer cleanup. Fixture files are isolated under
/home/dbayha/bca-work/ood-import-scratch-fixtures-v1/monitor_20260929T004802Z.
These are nonscientific fixtures, not the original scientific lease/resource
archive or a real v2 execution directory. Tests inherit their harness environment
plus an explicit TMPDIR; they do not accept any complete scientific launch profile.

Development evidence is retained. The initial23-test run exited1 because the
draft expected shutil.rmtree audit arguments `(path,None)`; installed Python3.10
actually emits `(path,)`. The corrected exact tuple passed23 tests. Identity-check
exceptions were then made persistently poisonous, with six new cases covering
missing/changed source, helper replacement, reentry, swallowed refusal and a hard
link escaping the temporary directory. The expanded29 tests passed. No
preimplementation red was captured and no closed suite was rerun.

The first separate review then exited1 because a prior /tmp fixture directory
was unavailable, although its test parent had observed it. Its disappearance
cause is not established. All old sources/logs/exits remain. The test fixture was
separately revised to use the named persistent workspace above; a new29-test run
passed, and the separate reviewer verified its retained paths without replaying
any fixture. This accepts that observed retention, not power-loss durability or
a guarantee about arbitrary /tmp contents. Two read-only PowerShell lookup/parser
errors were also retained; neither executed a test or target import.

Source-only inspection recorded27 installed files and five exact copies,
including the strict callback plus tempfile, pathlib, shutil and os. Four stdlib
files have three hard links each. Their current bytes/stat/link counts were
rechecked by the independent review; their alias provenance and acceptance for
scientific execution remain unresolved. The component does not weaken the old
unaliased source guard. The module-level call inventory explicitly excludes
function/class definitions and is not a complete transitive execution graph.

That inventory found another concrete integration obligation: filelock._read_write
registers an audit hook and an at-fork callback at module scope. A separate
source-only read exited0 at01:00:31.231274UTC and saved the exact module bytes,
SHA2563c194b83f7db92a0dca51c89e4d91982858b47d0c450733ed6c43306766c383f.
The audit handler calls _FORKED_DATABASES.note_sqlite_use() on sqlite3.connect;
the fork handler exits if the SQLite transition depth is nonzero. No live hook
registration or SQLite behavior was tested or accepted here. These callbacks and
their referenced state need binding before later ledger/archive integration.

Next, independently review a separately versioned native-import bridge using this
exact scratch lifecycle in the real imported callback namespace. Bind source/live
stdlib and filelock hook semantics, explicit launch TMPDIR and worker-owned fresh
path, diagnostic arguments and failure retention before any new single import
observation. Do not preimport filelock outside instrumentation, fake its capability
result, globally allow temporary writes or alter installed dependencies. No new
14/15-field real launch has been accepted. Full native fields/context managers,
storage, worker-owned original shared lease and only-route integration remain.

All prior closed v2 sources/receipts, original plan, approved one-stream amendment,
failed native probes and failed key producer remain unchanged. All40 key tables
stay held. Training remains10 accepted runs/five pairs; no cluster probe, other-chat
read, messages or subagents occurred. The local training mirror remains107/108
with only the reviewed advisor .gitattributes append. No model, simulator, physics,
scientific lock/ledger/archive or new key ran. Publication contains only new small
source/evidence; no native binaries, weights, key arrays, SQLite or NPZ. Routine
component preparation does not warrant another failure notification.

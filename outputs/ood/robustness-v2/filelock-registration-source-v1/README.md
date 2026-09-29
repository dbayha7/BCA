# Filelock registration conditions and relative source graph

The complete saved registration statements correct an overbroad reading of the
earlier call-name inventory: on the declared CPython route, `_read_write.py`
skips its SQLite audit-hook registration because it is inside `if _IS_PYPY`.
A different audit hook in `_api.py` still needs binding. Following relative
imports also found three `_soft_rw` subpackage sources and an atexit cleanup
registration absent from the earlier top-level file inventory.

These are independently reviewed source findings, not a real import observation.
No filelock/JAX/MuJoCo import, callback registration, SQLite connection, fork,
scratch probe, native probe or scientific operation was performed. No production
guard changed and no closed test or audit was rerun.

## Exact source-predicted registrations on CPython/Linux

| Source route | Registration and required binding |
|---|---|
| `_api.py` module calls `_register_fork_hooks()` at line1801. Its early return applies only when `_REGISTER_AT_FORK is None`. | `sys.addaudithook(_audit_fork_safety)` plus one `_REGISTER_AT_FORK` call with before=`_pin_fork_objects`, after_in_parent=`_resume_parent_after_fork`, after_in_child=`_reset_child_after_fork`. |
| `_read_write.py` line775 `if _IS_PYPY` | The `sys.addaudithook(_track_sqlite_use)` call is skipped for the saved CPython implementation. This does not mean all filelock audit hooks are skipped. |
| `_read_write.py` line777 `if hasattr(os, "register_at_fork")` | One `os.register_at_fork(after_in_child=_abort_forked_sqlite_transition)` call. The callback exits70 if its thread-local SQLite transition depth is nonzero. |
| `_soft_rw/_sync.py` line979 | `atexit.register(_cleanup_all_instances)`. The callback iterates `_ALL_INSTANCES` and calls `instance.release(force=True)` under exception suppression. |

The source predicts one audit registration, two at-fork registration calls and
one atexit registration along this completed route. These are not captured event
counts from either earlier failed native process. Their raw journals and actual
exits remain unchanged; no previously unrecorded registration or absence is
invented retroactively.

`_audit_fork_safety` captures `_FORK_AUDIT_EVENTS` and the exact `_FORK_STATE`
object in keyword defaults. Checking only the current module globals would miss
those captured references. Its event set is `os.fork`/`os.forkpty`; it refuses
when transition depth or fork-owner depths are nonzero, and separately checks
`_posixsubprocess.fork_exec` with nonzero transition depth. It does not provide
the import observer's blanket no-process policy.

The at-fork callbacks refer to live conditions/RLocks, thread-local transition
state, weak object/class registries, owned descriptors, token counters, pinned
objects/classes and parameter-model state. `_register_fork_object` and
`_register_fork_class` enter the contextmanager `_fork_transition`; class
creation also calls registration through `BaseFileLock.__init_subclass__`.
Therefore the final registries must not simply be assumed empty. `_read_write`
explicitly registers its connection escrow and database registry objects and
ReadWriteLock class. The soft-read/write class and atexit registry also require
live binding. The saved review records exact relevant function bodies and state
initializers as concrete obligations, without claiming the complete runtime
dependency graph is already accepted.

## Saved inspections and independent review

The first new source-only inspection completed with actual0 at
01:58:26.222329 UTC on September29,2026. It saved 18 top-level filelock files and
five relevant stdlib sources, with enclosing conditional/class/function contexts
for registration calls. A separate follow-up at02:00:28.390941 UTC followed the
new `_soft_rw` relative-import lead and added its three sources. The combined
inventory has26 source files:21 filelock files and five stdlib files. All64
syntactic relative ImportFrom module edges resolve in those21 files, including
deferred and type-checking paths. This is not observed import order, dynamic
import closure or complete external/transitive runtime acceptance.

The separate saved-source reviewer completed with actual0 at02:03:22.072811 UTC.
It rehashed/restatted all26 current files against saved exact bytes and checked
the conditional registrations, callback keyword defaults, early-return route,
state initializers and reported relative edges. A supplementary saved-byte
review independently reconstructs all64 edges and compares the complete edge
multiset. Neither reviewer imports or executes target definitions.

The first supplementary edge reviewer returned actual1 at02:05:36.507486 UTC:
Windows Path rendered candidate Linux metadata paths with backslashes, which
did not match the saved POSIX path keys. A separately versioned reviewer uses
PurePosixPath for those metadata paths, preserving local Path handling and the
failed source/stdout/stderr/receipt. The correction does not change any target source,
edge, registration finding or installed file, and does not replay an inspection.

Key source hashes:

- `_api.py`: `5568ae5b6a161eb33d172df2465b31adef9d6978e99b309b24ee37885b989091`
- `_read_write.py`: `3c194b83f7db92a0dca51c89e4d91982858b47d0c450733ed6c43306766c383f`
- `_soft_rw/_sync.py`: `5cc0b490ca1104ca02009ab12bd6d4fd45b9a3429543952671d62aa69e63f243`
- Initial inspector: `0f2cc1c12b8dbb1186e741fab1a2e05b5998548a605e9d4216d125f54ce09ac1`
- Relative-edge inspector: `7d50046d217fe2709dc97517e617d0a55c67f577b19971ba0255f089a5ed426c`
- Separate source reviewer: `2aa8cfbbd77df306962316ce14de21b5d3ec114651b5da75be7f818f1a897535`

One initial helper invocation omitted the required `.py` suffix. The helper's
allowlist assertion returned shell1 before WSL dispatch; target execution/exit
is null, not a failed source inspection or scientific probe. The exact command
and observed traceback are retained. Correctly named new inspections then ran
once each and returned0. No test suite or preimplementation red is claimed.

## Next actual binding boundary

Bind the live `_api` audit callback including captured keyword defaults, its
three at-fork callbacks and referenced state; the `_read_write` at-fork callback;
and the soft-read/write atexit callback and registries. Retain the distinction
between source-predicted and actually captured registration arguments/order.
Bind the relevant stdlib live code/global graph and the explicitly observed
shared-source aliases. Compose the closed scratch observation with the native
build/cache/lock refusal route without replaying the capability function.

Only after a separately declared exact15-field environment including TMPDIR,
fresh worker-owned scratch lifecycle/failure retention, and independent review
of that actual composed route may one new versioned import-only observation run.
No such profile or observation ran here. Do not globally allow temp writes or
registrations, pre-import filelock, fake capability results, edit dependencies,
retry closed native probes or silently relax any old guard.

All prior source/receipt, scientific plan, approved one-stream amendment, twelve
accepted training runs/six pairs and failed producer/40 held key tables remain
unchanged. Local training-source comparison remains107/108 solely because of the
previously reviewed publication-only .gitattributes append. No cluster probe,
queue change, new keys, resource lease, ledger, archive, model, simulator or
physics was performed. Full engineering and pair-wide science gates remain.

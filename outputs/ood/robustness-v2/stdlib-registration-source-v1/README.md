# Stdlib registrations and finalizer state on the proposed native import route

Source inspection on September 29, 2026 identified concrete additional bindings
needed before composing the real native import observer. The four registration
requests in the closed synthetic filelock fixture are not a complete declaration
for the actual import route. No real native import or new test suite ran here.

The exact saved `filelock._async_read_write` source imports `ThreadPoolExecutor`
from `concurrent.futures`. The installed `concurrent.futures.__getattr__` resolves
that name by importing `.thread` and caching the result. Actual timing depends on
whether an earlier import has already resolved it. Source inspection does not
establish the observed import order or a live registration count.

`concurrent/futures/thread.py` (SHA256
`b06f8899881193efc72cfc3ebf2836dce4e668b3856ad35f4016616d643a519e`)
has two module-level registration routes:

- Line 37 calls `threading._register_atexit(_python_exit)`. This is an internal
  Python registration route distinct from `atexit.register`. Its helper creates
  `functools.partial(func, *arg, **kwargs)` and appends it to `_threading_atexits`;
  `_shutdown` calls that list in reverse before joining threads. Bind the actual
  partial's function/arguments/keywords, list contents and `_SHUTTING_DOWN` state.
- Lines 41â€“43 call `os.register_at_fork` under `hasattr`, with ordered keywords
  `before=_global_shutdown_lock.acquire`,
  `after_in_child=_global_shutdown_lock._at_fork_reinit`, then
  `after_in_parent=_global_shutdown_lock.release`. These are native bound lock
  methods, not plain Python callback functions. Bind the same owning lock, native
  type/descriptors and actual registration argument objects; do not replace them
  with wrappers or synthetic functions.

`_python_exit` sets `_shutdown` under `_global_shutdown_lock`, snapshots the
`_threads_queues` WeakKeyDictionary, sends `None` to each queue and joins its
threads. Its source initializes that registry empty, but this is not a live or
future emptiness guarantee. The callback, registry, queue/thread ownership and
shared lock all need live bindings. No executor, queue work or thread was started
by these source inspection scripts.

The inspected `threading.py` also contains its own module-level
`os.register_at_fork(after_in_child=_after_fork)` under `hasattr`, at line 1645.
The actual worker must account for bootstrap timing and any registration before
the composed guard starts. This finding does not retrospectively count callbacks
from earlier failed native probes.

The exact saved filelock capability callback uses `tempfile.TemporaryDirectory`.
Installed `tempfile.py` line 837 constructs `weakref.finalize` with the directory
object, the bound `_cleanup` classmethod and directory name, plus `warn_message`
and `ignore_errors`. Installed `weakref.py` line 574 registers
`self._exitfunc` with `atexit` only when `not self._registered_with_atexit`.
`_exitfunc` is itself a classmethod. The source initializes the flag false and
sets it true after the registration request; an existing finalizer may have
already done so before this capability probe. The flag alone does not establish
observed registration arguments or the current shutdown callback inventory.

The finalizer stores a weak reference, callback, arguments, keyword arguments,
exit flag and index in the shared `finalize._registry`. Explicit
`TemporaryDirectory.cleanup` calls `self._finalizer.detach()`; the saved
`detach` body removes the matching registry entry when the weak target is live.
This is a concrete lifecycle binding beyond filesystem deletion. A future
observer must bind the actual classmethod `__self__`/`__func__`, source/globals,
registry entry, target, callback arguments, index and registration flag, then
verify the exact detach/removal. Other existing finalizers must remain visible.
An empty scratch directory does not by itself establish finalizer state or
harmless future shutdown. No finalizer or shutdown callback was invoked here.

The closed registration-argument component only accepts plain FunctionType
callbacks and wraps three native APIs. It therefore does not yet cover these
classmethods, native bound lock methods or `threading._register_atexit`. Its
40-test evidence remains valid within its synthetic scope and was not rerun or
edited. The closed scratch/import bridge/native guards also remain unchanged.
Any composition must be separately versioned, preserve actual argument/order
evidence and first refusal, and bind these additional source/state routes.

Inspection finished at 03:05:23.183532 UTC with actual exit 0. Seven exact stdlib
sources were saved: `concurrent/__init__.py`, `concurrent/futures/__init__.py`,
`concurrent/futures/_base.py`, `concurrent/futures/thread.py`, `threading.py`,
`weakref.py` and `tempfile.py`. The extractor records four hook-registration
source sites and one finalizer-construction site. These are syntactic source
sites, not five observed registrations or a complete external dependency graph.
The two filelock leads reuse their previously saved exact bytes. No extracted
definition was executed, and the inspector's recorded module inventory contains
none of filelock, concurrent.futures, NumPy, JAX, mujoco_py or GLFW.

A separate bounded alias inspection finished at 03:06:46.123479 UTC, actual 0.
All seven files have exactly three current hard links across the corl-orig-local
environment, sdbca environment and package cache `python-3.10.20-h741d88c_0`.
All 21 named paths match device/inode/bytes/stat, link count 3, uid/gid 1000 and
mode 0644. Two environment records and the matching package path metadata agree;
the separately retained alternate package metadata is not a fourth alias claim.
These files remain writable and shared. This is current provenance, not an
execution alias exception or protection against later changes/races.

A separate stdlib review finished at 03:09:25.300544 UTC, actual 0. It independently
reconstructed the five source call records from AST parent chains, checked exact
helper/callback bodies and the lazy import branch, and rehashed/restatted all
21 alias paths plus four metadata records. It imported neither the target
packages nor the closed components and replayed no fixture. The review supplies
explicit actual-route binding obligations, not live object acceptance. No test
suite, preimplementation red or native observation occurred.

A subsequent concrete lead from saved `filelock._api` imports and the inspected
futures base source added `logging/__init__.py`, `secrets.py` and `random.py`.
Their source inspection completed at 03:12:49.913114 UTC with actual 0. Separate
supplemental review v2 completed at 03:15:28.622846 UTC with actual 0. It checked
the three sources and nine additional hard-link paths against the previously
reviewed Conda metadata; the initial seven-source review was not rerun. Combined
coverage is ten named source files, thirty current alias paths and seven hook
registration source sites, plus the separate finalizer construction site. It is
still not the complete external/runtime graph or an observed registration count.

`logging` registers three fork callbacks (before `_acquireLock`, child
`_after_at_fork_child_reinit_locks`, parent `_releaseLock`) and registers `shutdown`
with `atexit`. **`shutdown(handlerList=_handlerList)` captures the original mutable
list in a positional default.** Bind that exact list reference, weak handler
references, `_handlers`, each handler's relevant methods/state, the module RLock
and `_at_fork_reinit_lock_weakset`. Its shutdown body acquires, flushes, closes and
releases live handlers; its child callback reinitializes their locks and the
module lock. Empty-state or harmless-shutdown assumptions are not justified.

`secrets` imports `SystemRandom` from `random`. At module scope `random` creates
`_inst = Random()` and, under `hasattr(_os, "fork")`, registers the bound Python
method `_inst.seed` as its child callback. Bind the original owner, `Random.seed`
code/defaults/globals and its implicit **`__class__` closure cell**, confirmed by
compile-without-execution (`co_freevars == ('__class__',)`). The native
`_random.Random` dependency remains an additional runtime obligation. Keep
`random._inst` distinct from `secrets._sysrand = SystemRandom()`. No seed method,
RNG draw or callback was invoked by this inspection/review, and no scientific key
was generated. These method/default/closure shapes also require a separately
versioned actual binding route; the closed plain-function fixture is insufficient.

The first supplemental reviewer returned actual 1 at 03:14:53.802227 UTC because
its unique-name selector encountered two conditional definitions of logging's
`_register_at_fork_reinit_lock` (fork-absent stub and fork-present implementation).
The exact failed source/output/exit are retained. Separate v2 explicitly retains
both definitions for that one named helper; other selectors remain unique.
No inspected source, callback or existing test changed, and no target execution
was retried. Two document-only patch context mismatches also made no changes;
corrected literal edits and their tool-error descriptions are retained. These
are routine preparation corrections, not new scientific failure notifications.

All previous closed sources/evidence, original plan, approved one-stream amendment
and accepted training artifacts were verified unchanged at 03:10:02 UTC. Training
remains fourteen runs/seven pairs, both ReBRAC cells three of five paired seeds.
Local training files remain 107/108 exact with only the previously reviewed
publication metadata append. The latest own cluster snapshot is still the dated
02:24:54 UTC record of 62 closures and 50 pending audits; no cluster probe ran.
Queues, local CQL stop and resource hold remain unchanged.

No exact 15-field real import profile, fresh worker-owned scratch lifecycle,
full live dependency graph or native acceptance exists yet. Bind these newly
identified routes alongside the real filelock callbacks/state and stdlib graph,
then independently prereview the actual composed route and explicit TMPDIR
creation/cleanup/ownership/diagnostics/failure-retention contract before one new
versioned import-only observation. Source findings alone grant no execution.
Do not preimport filelock, fake a capability result, edit installed dependencies,
permit hooks/writes broadly or retry the closed failed probes.

The failed key producer remains actual 1; all forty saved key tables remain held
for explicit independent execution disposition. No regeneration or closed key
review occurred. No scientific lease, ledger, archive, model, simulator, learner
or physics ran. Full engineering and separate science gates, durable per-call
reservations, action 1e-6/reward 1e-7/full-first-repeat checks and pairwide precommit
remain required. No other chat was read or messaged and no subagent was used.

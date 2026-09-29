# Filelock registration arguments: synthetic component evidence

Forty new isolated tests passed with actual exit 0 and no skips on September 29,
2026 at 02:46:28 UTC. A separate stdlib saved-evidence review passed at 02:56:12 UTC
without importing this component or replaying fixtures. This closes the argument
capture gap in synthetic registration fixtures. It does not accept the real
filelock import or the native runtime.

The separately versioned `filelock_registration_args_v1.py` wraps
`sys.addaudithook`, `os.register_at_fork` and `atexit.register`. Before forwarding,
it checks the pinned caller source/code, registered module namespace, declared
callsite/order, positional arguments, ordered keyword arguments and callback
object/code/default/global references. The audit callback's keyword defaults bind
the exact captured `_FORK_AUDIT_EVENTS` and `_FORK_STATE` objects, and later checks
reject substitutions even when a replacement looks equal. A supplied state
verifier runs around registration and at completion; its object/code/defaults
are checked. The verifier's complete dependency graph is not thereby accepted.

Successful fixtures contain the exact saved installed `_register_fork_hooks`
helper and six callback definitions: `_audit_fork_safety`, `_pin_fork_objects`,
`_resume_parent_after_fork`, `_reset_child_after_fork`,
`_abort_forked_sqlite_transition` and `_cleanup_all_instances`. They use a
registered synthetic module, synthetic state and inert helper substitutions.
Neither the real package namespace nor its actual import order is reproduced.

The declared fixture sequence forwards one audit registration, one three-callback
fork registration, one child-only fork registration and one shutdown registration
to the real native APIs. Each request saves actual callback metadata and ordered
keyword arguments, followed by observed native entry and return: twelve events
in each successful case. The wrappers forward the original callback objects
exactly once and never invoke, replay or replace them. Direct native bypass is
refused by the profile observer; its unavailable arguments remain null.

Native audit registration returning normally does **not** prove the target hook
was installed. An explicit fixture has an earlier hook suppress that registration
while native entry/return still completes. The component checks its own audit
hook with a private canary and refuses suppression or an exception during its own
installation. This canary does not certify the target callback. Automatic audit
deliveries are not counted; their count remains null.

The final run contains two successful observations (ordinary registration and
the explicit target-suppression case) and 38 expected refusals. All forty children
exit 0 after checking their expected result. Refusal cases include changed
callback/default/captured-state/global/metadata, argument order/shape, source or
namespace, wrapper/native API/verifier replacement, missing/repeated operations,
swallowed refusal, an outside write, environment mutation and profile replacement.
The first refusal persists on a subsequent check. All forty saved child inventories
contain no filelock, NumPy, JAX, MuJoCo or GLFW target modules. These are saved
snapshots, not a general sandbox claim.

The initial 31-test development run also passed (02:44:15 UTC). Review then added
nine cases and tightened wrapper inventory, callback metadata, namespace identity,
keyword-default dictionary identity and observer-installation exception handling
before the final forty-test run. Both exact code versions, logs and actual exits
are retained. No preimplementation red or failed test run was captured. One
read-only guessed exporter filename was missing; the enclosing shell returned 0,
no target was dispatched, and the corrected name was found by listing files.

All children keep installed observers and exit through `os._exit`. Persistent
fixture source directories remain under
`/home/dbayha/bca-work/ood-registration-args-fixtures-v1/monitor_20260929T023304Z`.
No fork, shutdown callback, SQLite connection, capability scratch callback or
scientific resource operation ran. This does not establish shutdown safety,
continuous containment after arbitrary tampering, native-C syscall coverage or
power-loss durability. The state verifier accepts only the declared synthetic
state; it does not validate real Condition/RLock/thread-local/weak/class/singleton
registries or future state transitions.

Independent review checked all forty saved exits, retained source bytes, six exact
callback bodies, the exact successful helper, twelve-event argument/order records,
refusal diagnostics and explicit scope flags. It used the closed source graph's
saved `_api.py`, `_read_write.py` and `_soft_rw/_sync.py` bytes, without a new
installed-source inspection. The previous CPython source-level correction stands:
`_read_write`'s PyPy-only SQLite audit registration is source-predicted skipped;
this turn supplies no live full-package hook count or absence observation.

The old scratch, import bridge and native refusal guards are unchanged. This new
observer is not yet composed with them. The actual route must reconcile its
profile/wrapper observers, bind the full relevant live stdlib graph and explicit
current shared-source aliases, and bind real filelock callback/state lifetimes.
An exact separately versioned 15-field environment including TMPDIR, fresh
worker-owned scratch creation/cleanup/ownership/diagnostics/failure retention and
independent prereview must precede one new import-only observation. No such
profile or observation ran here. No blanket hook or temporary-write permission
was added; no installed dependency was edited.

All prior closed components, source graph, native failure receipts, accepted
training artifacts, original plan and approved single-stream amendment were
verified unchanged. Local training files remain 107/108 exact, with only the
previously reviewed publication-only `.gitattributes` append. Training acceptance
remains fourteen runs/seven pairs, with both ReBRAC cells at three of five paired
seeds. The latest own cluster snapshot remains the dated 02:24:54 UTC record of
62 closures and 50 pending audits; no cluster probe ran this turn. The already
reported third Walker comparison is not a new notification.

The failed key producer remains actual exit 1 and all forty saved tables remain
held pending explicit independent execution disposition. No real extension,
shared scientific lease, archive, model, simulator, learner or physics ran.
Full engineering and separate science gates, durable per-call reservations,
action 1e-6/reward 1e-7/full-first-repeat checks and pairwide precommit remain
required. Queues, local CQL stop and resource hold are unchanged. No other chat
was read or messaged, and no subagent was used.

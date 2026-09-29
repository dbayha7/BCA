# Unified bootstrap, capability and finalizer composition

The new dispatcher now observes the saved filelock capability's ordinary module
entry, natural single callback invocation, temporary storage and actual weakref
finalizer construction/detach through one audit/profile owner. This closes a
specific composition gap between the previous independent bootstrap and scratch
fixtures. **The full actual filelock/native route is still unimplemented and real
dispatch remains held.** This is a registered synthetic module containing the
exact saved capability definition with real stdlib objects, not full filelock.

`native_import_dispatch_v4.py` is separately versioned from the closed v3 source;
it does not stack old observers or invoke `scratch.run`. Final source SHA256:
`9380e8fceae651015a6207dbbc579a45bfb54d913f52d88a437403feba6a31db`.
Test SHA256:
`6ecac363b2546524f8ba6a38f8befc07e98a86da75a6bf455685fef4dacb7414`.
These sources are now closed. Later integration needs a separately versioned
derivative; no routine rerun or silent edit of these sources.

## Observed composition

The fixture parent observes the child's actual Linux identity and exact fifteen
environment fields before releasing it. Only fixture TMPDIR differs from the
closed production declaration. The child exclusively creates diagnostics then
scratch with retained parent directory descriptors, no-follow and name/descriptor
checks, mode0700. Four exclusive mode0600 diagnostic files retain the existing
8MiB each/32MiB total limits and fsync before acknowledgment. First refusal is
shared and immutable; identity tampering blocks even subsequent diagnostics.

The ordinary module loader executes the exact saved `_probe_link_follow_symlinks`
definition once. The observer records four entry/return boundaries and twelve
scratch audit events: default temp probe with `blat` readback, temporary directory,
empty source and verified hardlink pair, then exact descriptor-linked cleanup.
It never invokes that capability itself or substitutes its return. Thirteen
native entry/return pairs have source/line evidence but **C arguments remain
unavailable/null**. Separate audit arguments carry filesystem paths/order.

The actual TemporaryDirectory creates a weakref.finalize entry. The observer binds
the live target, classmethod owner/function, actual callback/argument objects,
ordered keyword arguments, weak reference, atexit flag and index. It observes the
natural cleanup's detach entry and exact removal/returned tuple while the target
is alive. This is registry-lifecycle evidence in addition to filesystem cleanup.
The observer retains the current target during inspection; it does not establish
uninstrumented garbage-collection behavior.

Two successful cases cover different **observed** initial states:

- With empty registry and registration flag false, the original weakref exit
  classmethod is passed once to the original native atexit API. Actual request,
  native entry and return are saved, including source line574. The new entry has
  index0 and is removed by detach; registration flag becomes true.
- With one unrelated live sentinel finalizer and flag true before the observer,
  the new entry has index1; the unrelated entry remains exactly visible after
  detach. No additional atexit registration occurs. The pre-boundary sentinel
  registration is not counted as an observed registration.

No shutdown/finalizer callback is executed. Children keep hooks and wrappers and
terminate with `os._exit`. The observer's own canary does not establish target
audit-hook installation. This remains cooperative Python instrumentation, not a
security sandbox or complete native syscall observer.

## Source and state scope

Seven actual TemporaryDirectory/finalize functions are matched against compiled
full saved source, exact defining classes, defaults, absence of closures,
registered namespace and direct global references. Native weakref symbol object
identities and the actual registry are checked. Five stdlib roles are bound to
fifteen explicit current alias paths (corl-orig-local, sdbca and Python package
cache), with equal bytes/stat and nlink3. These writable aliases are not accepted
as a scientific exception or protected against future changes/races. Direct
global/native identities do not close the full transitive implementation graph.

Stdlib modules load before this observer boundary; their earlier registrations
are not retroactively observed or inferred absent. The synthetic namespace has
only the exact capability definition and required real globals. Full filelock
registration/state behavior is not established.

`sys.dont_write_bytecode=True` is an explicit fixture runtime choice before dynamic
source loading. It changes no environment field and is not a production runtime
acceptance. Production use requires a separately bound implementation/prereview.

## Tests and separate review

Final22 new isolated Linux fixtures passed with actual process exit0 and no skips
at2026-09-29T04:28:04.074477UTC: two successful compositions and twenty expected
refusals, all22 child exits0. Nine cases completed the capability lifecycle before
any deliberate later refusal. Negative cases cover class/method/default/global/
native symbol identity, registry replacement/flag/entry mutation, restored detached
entry, wrapper replacement, repeated import/binding, outside mutation, undeclared
atexit requests/direct aliases, no-bytecode flag mutation and full-route refusal.
Target-module inventories are empty snapshots; all final profile snapshots are
present. Neither fact proves continuous absence or native containment.

Separate stdlib saved-evidence review actual0 at04:38:44.097668UTC, reviewer SHA256
`327aa64f4f9b4b434b2ee5e21f0f31bbbca2a049fa7a6fa4cad5c6f77b29af5d`,
checked22 receipts, retained source/config/diagnostic bytes/stat and identity
releases,372 bootstrap/diagnostic native operation triples, nine complete
capability/finalizer lifecycles, conditional registration evidence, fifteen current
source alias paths, exact saved capability AST and source predicates. It imported
neither component nor target definitions and replayed no fixture. Disposition:
`ACCEPT_SAVED_CAPABILITY_FINALIZER_COMPOSITION_FIXTURES_ONLY_HOLD_REAL_DISPATCH`.

Preserve the first fourteen-case development attempt: suite actual1 at04:26:08UTC,
first child actual3, first refusal from importlib's bytecode-cache open before
capability module entry. No capability/finalizer ran in that child. Its original
source/test/rawlogs/partial bytes, including any local compiled cache, remain
retained. A separate version explicitly disabled bytecode writes and passed14
tests at04:26:40UTC. Review then tightened defaults/direct globals/defining owners/
native symbols and added eight tests before the final22-test pass. There was no
preimplementation red and no closed test/native probe/key-review rerun. The
initial shell preparation quoting failure dispatched no Python or file mutation;
its receipt is retained. These are fixture preparation events, not new scientific
failures. Compiled fixture caches are not published.

## Remaining execution hold

The closed fifteen-field production contract remains byte-identical SHA256
`5f922e3d0d27116139fd99293f537427f6d369f702b7fd685dbbe9de67567763`,
declared/unlaunched/not dispatchable, with planned actual entry/parent hashes null.
Both actual v3 roots were absent at the dated04:38:40UTC review snapshot; they were
not created or claimed. This is not future freshness acceptance.

Next compose the actual filelock `_api` audit callback and captured keyword
defaults/three fork callbacks/shared state, read-write fork state and soft-rw
shutdown registries, plus native bound lock owners/descriptors, Random.seed's
owner/__class__ cell, logging's original mutable default/handlers/locks and
threading's internal partial/list/shutdown state. Preserve this natural finalizer
lifecycle and unrelated entries. Bind the relevant full code/global/native/state
graph, current aliases and pre-boundary provenance; implement the exact cached
builder/package-lock refusal and production parent release/completion identity
acknowledgment. Independently prereview the **full actual route** before one new
versioned import-only observation. No preimport of filelock outside the guard,
fake capability, installed edits, old probe retry or blanket hook/write exception.

Fresh immutable-input review actual0 at04:38:38.890298UTC verifies all prior closed
sources/receipts including v3 bootstrap, original plan, approved one-stream
amendment and accepted training evidence unchanged. Local training retains the
reviewed107/108 publication-metadata-only context. Forty saved key tables remain
held pending explicit independent execution disposition; no regeneration or
closed review rerun. No native probe, lease, extension, model, simulator or physics
was dispatched. Training remains14runs/seven pairs and both ReBRAC cells3/5;
the02:24:54UTC62-closure/50-pending-audit cluster snapshot is historical. No cluster
probe, queue change, subagent, other-chat read or message occurred.

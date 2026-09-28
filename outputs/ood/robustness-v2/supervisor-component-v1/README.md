# One-shot engineering supervisor and worker route

The separate supervisor now observes an actual cooperating child process with
the exact isolated command and CPU environment from the closed engineering
capsule. Thirty-four new synthetic tests passed, actual exit 0, with no skips.
These sources are now closed. No real scientific entrypoint, model, simulator,
resource lock, ledger, archive, fresh stream or candidate pool was executed.

The supervisor consumes the independently reviewed capsule once, exclusively
claims a fixed sibling evidence directory, and durably saves launch intent before
starting its child. That claim survives failures and prevents reopening through
another supervisor instance. The evidence path cannot replace or contain the
declared original ledger or shared lock. Unknown spawn/exit observations remain
null; a missing observer receipt or unresolved launch intent is not success.

The worker independently checks the capsule, actual command, initial and
effective environment, parent PID/start ticks, own PID/start ticks, session and
process group. It saves readiness, then waits. The observer saves the actual
process identity and release intent with file/directory fsync before sending
release. Only then may the cooperating worker invoke its one engineering
callback. Pinned inputs, original ancestor identity/no-sidecars, ownership and
environment are checked before and after the callback; the callback also receives
the check function for eventual per-operation use. Mutation, swallowed reentry,
fork, parent loss or callback errors poison the route. No retry exists.

Raw child stdout/stderr and actual return codes are retained. Time and log-size
ceilings are explicit. On observer-detected failure the still-identified private
child process group is signaled, with actual negative signal return codes saved.
The synthetic timeout produced actual -15; deliberate process death produced
actual 23. Neither is a scientific failure. A child exiting 0 without the protocol
is rejected with actual 0 retained, rather than mislabeling the route as accepted.
Failed spawn retains actual exit null. Storage failures retain partial evidence;
no delete-and-retry or repair path exists.

This is a cooperating process route, not a Python sandbox or complete descendant
containment. It neither proves the truth/provenance of the supplied review nor
accepts the actual native entrypoint. It does not acquire the shared lease or
create the extension. Different declarations still require the original shared
lease in the worker for global exclusivity. Parent death cannot be assumed to
kill arbitrary descendants; the actual worker must use the ownership check on
every physical route. Those integrations and actual storage durability remain
unaccepted. The original production.construct/reserved_step writer remains unused.

An actual standard-library inspection found that the pinned Python 3.10 process
adds LC_CTYPE=C.UTF-8 to its effective environment even when launched with only
the five frozen CPU fields. The unmodified synthetic entrypoint therefore fails
the exact effective-environment check before readiness, actual exit 1 and no
release. The closed capsule environment was not extended or edited. Successful
synthetic fixtures explicitly remove this locale field in their temporary
entrypoints only. This isolates the process protocol for tests; it is not an
accepted real compatibility fix and does not prove native-library behavior.
An explicitly versioned and independently reviewed real environment binding is
still required before dispatch, in addition to the unresolved stream amendment.

Tests reuse only the closed capsule fixture helper. They use fake checkpoint,
runtime and non-SQLite ancestor bytes, invented passing reviewer receipts and
temporary copies of the exact scientific plan. No prior test suite or closed
audit reran. The first 28-test development pass exited 0. Review then added resource
path containment and consumed-capsule mutation refusal, six tests, and preservation
of the small synthetic child receipts. The final 34-test pass exited 0. No missing-
module red was captured. Temporary scientific-looking paths are fixture data.

The separate standard-library review checks the saved actual process receipts,
intent/release/completion relationships and source hashes without launching any
child again. Exact raw small child evidence and temporary entrypoint bytes are
packaged, including the expected locale refusal and deliberate exit 23.

Remaining before real engineering: stream amendment authorization and independent
acceptance; exact native/local-import/field/contextmanager dependency bindings;
actual checkpoint/data/support/restore schema; accepted environment and final
device storage; accepted reviewer/observer provenance; worker-owned shared lease,
original AncestorGuard and new ExtensionLedger integration covering every physical
call. Then action 1e-6, reward 1e-7, full-state repeat, global precommit and separate
science acceptance are required before outcomes. Component tests authorize none
of those operations. Accepted training remains six runs/three first-seed pairs.

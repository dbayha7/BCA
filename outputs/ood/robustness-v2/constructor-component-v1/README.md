# V2 native constructor evidence component

Status:21 synthetic tests passed, actual exit0/no skips. A separate read-only
installed-source review also exited0. No real model, simulator, resource lock or
extension ledger was opened by these tests or the source review. Runtime and
scientific execution acceptance remain false.

The installed Gym constructor samples its native action space and calls the
native step once before the outer wrappers exist. The source review verified
this ordering directly; assuming a zero constructor action would be incorrect.
The new observer preserves the sampled action and original native methods. It
does not seed, sample, replace controls or change numerical source. Sampling has
already happened when the native step is observed: the actual action and existing
after-sampling RNG state are saved, but pre-sampling entropy is not reconstructed.

`native_constructor.py` requests one engineering reservation of1 environment and
4 physics steps through the closed extension interface before invoking its
factory. It temporarily observes exactly the bound native step and defining
physics method, saves the actual input action and native prestate, then durably
saves the intercepted control and another native prestate BEFORE the original
physics callback. A second native step, second physics call, changed control or
wrong frame count is refused before that additional call. The original methods
are restored on success and failure.

Native evidence copies data/model fields, native time and user state, model-byte
hash, bounds, frame skip and existing RNG state. Access uses the existing internal
RNG field rather than the lazy `np_random` property, which could initialize an
absent RNG. Wrapper elapsed/reset/action-RNG/adapter termination fields are
explicitly unavailable during the constructor; no initialized wrapper restore
snapshot is invented. Full native contents are saved, not just their hashes.

The returned observation/reward/done/info and full native poststate are saved
before validation. The component checks native controls, unchanged static model
and RNG, the four-frame time increment, observation and health arithmetic, and
position-derived forward reward plus alive bonus and action cost under the
unchanged absolute reward1e-7 gate. A completion artifact binds the input,
intercepted control, output and native poststate before the ledger acknowledges
the charge. This producer completion is not independent scientific acceptance.

Any exception leaves a charged pending reservation and blocks further calls via
the ledger interface. If native code throws after some or all physics, available
poststate and exception details are retained without inventing a return. Input,
output or completion storage failure, invalid returned arithmetic, or a factory
failure receives no refund or retry. Cleanup closes only the newly created
adapter/native object where available; the original exception stays authoritative.

Tests use synthetic Hopper/Walker classes, memory stores and ledger doubles.
They cover nonzero native actions, data copying and honest missing RNG fields,
ordering, exact one-call guards, state/control/reward/observation/health refusal,
storage and factory failures, method restoration and poststate preservation on a
native exception. The expected missing-module failure, first20-test pass and
final21-test pass are preserved. No closed prior test suite was rerun; only its
synthetic fixture helpers were reused.

The read-only review opened installed source bytes and parsed their AST without
importing the modules. The three MuJoCo source hashes match the frozen simulator
registry. `core.py`, `spaces/space.py` and `spaces/box.py` are additional current
source pins absent from that registry; their `matches_frozen_simulator_source`
false fields mean no prior registry binding, not a detected source change.
Their exact runtime acceptance remains pending along with the other bindings.

Still required: an actual bound factory/classes/runtime and durable archive,
checkpoint/data/support/native-schema/fresh-stream/actual-path declarations,
storage feasibility, CPU/shared-lease supervisor, global state/candidate/key/
warning precommit barrier, and independently reviewed engineering execution
acceptance. Any real engineering physical call must first be accepted and charged
under the new envelope. Actual native action1e-6 parity, reward1e-7 and full-state
repeat/restore gates must pass before scientific dispatch. This component does
not authorize creation of a real extension ledger or claim a new OOD result.

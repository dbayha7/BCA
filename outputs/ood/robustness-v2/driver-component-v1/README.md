# V2 injected collector and outcome drivers

Status:25 synthetic tests passed, actual exit0/no skips. The actual extension
journal component was also exercised with a temporary synthetic ancestor, lock
and journal. **No real scientific resource, model or simulator was opened.**
These drivers are not a supervisor, accepted execution declaration or ready
scientific executor. Constructor/runtime integration remains pending.

`recorded_step.py` maps each call to exactly one declared pair/phase/token and
invokes the frozen extension's reserve-before-callback interface. The callback
saves the full supplied input capture, calls the injected step once, then saves
the output and full supplied after-state contents. It binds all three native raw
input/applied/output payload hashes to that simulator transition number. Only
after durable archive append/readback and control/reward/counter/flag checks does
it write completion evidence and allow the ledger to acknowledge the reservation.

Input evidence failure, physical callback error, missing raw payload, output or
after-state failure, control/reward mismatch or failed completion write leaves
the charge pending. No refund, replacement, retry or recovery is attempted. All
available evidence is saved before semantic refusal. A missing earlier artifact
or exception still stops immediately; the component cannot save evidence that
the callback never returned. Reward1e-7 remains unchanged, and expected float32
applied controls are compared exactly. Native actor-query action1e-6 parity is a
separate engineering gate, not established by these injected callbacks.

`EncodedStore` is an adapter to a separately accepted durable append/read archive;
it checks exact readback bytes. Typed encoding retains array dtype/shape/bytes,
signed zero, RNG tuples and inactive None fields under bounded artifact limits.
The tests' archive is an in-memory double; the physical adapter must separately
prove its raw intercepted-action archive is durable BEFORE physics. A synthetic
raw byte string cannot establish that ordering or native physics correctness.

`collector_driver.py` executes the fixed64 paired reset blocks under each frozen
collector, capturing at100 and300. Early native termination/truncation marks
unreached captures missing, including termination exactly at the scheduled step.
There are always256 nominal result rows, no replacement state or score selection.
Both collectors share each block's declared reset seed. Actual paired-reset RNG
initialization still needs the accepted runtime adapter; the injected reset
callback does not prove that native state restoration is complete.

`outcome_driver.py` binds the supplied precommit and snapshot content hashes,
capture step and actual saved keys before any step. It retains all14 nominal rows,
including missing slots and aliases referring to completed owner-result hashes.
The common recorded-action anchor stays slot0 and host-first reference stays12.
Every unique first action is restored and repeated once. Both entire records and
entire supplied first-end captures must match exactly before continuation. The
driver restores the first end state and follows the frozen deterministic policy
for at most250 TOTAL transitions including the first; it stops at the original
termination/time limit and computes the sequential raw undiscounted sum without
a tail value. Declared keys are retained but deterministic actors do not consume
them; do not describe them as stochastic rollout draws.

Each completed action saves the full final capture, its content hash and the last
call's evidence hash. The step bridge has already saved that full after-state
before acknowledging its call. This addresses the v1 missing-final-contents issue
at the interface level. **Actual complete restore-schema/native-field coverage
is still unverified for v2** until the real adapter and independent audit pass.
Externally accepted file hashes must bind the loaded input files in the runtime
layer; driver content hashes alone do not establish file/checkpoint provenance.

Tests cover the full38400-call nominal collection control flow using inexpensive
synthetic transitions, fixed captures and aliases/missing slots, 250-step horizon,
first-step and later termination/truncation, record and hidden-state repeat
mismatch, wrong hashes/keys/capture step, failed restore, durable ordering and
all recorded failure paths. The temporary SQLite integration confirms one1/4
completed reservation followed by a failed-evidence1/4 reservation left pending,
while the synthetic ancestor bytes remain unchanged. No prior closed test suite
was rerun: only its fixture helper was reused. Initial missing-module failure and
the19-test first pass are preserved; review then added exact capture-step refusal,
a defensive key copy and six focused tests before the25-test final pass.

Remaining integration is substantial: native constructor evidence with truthful
pre-wrapper field availability; actual source/runtime/checkpoint/support/stream
and path/declaration bindings; complete restore-schema/native-action/repeat gates;
global state/candidate/key/warning precommit barrier before any outcome; bounded
durable archive/storage checks; CPU/one-worker/shared-lease supervisor and actual
exit receipts. Pass and independently accept those engineering gates before
creating a real extension ledger or dispatching v2 science. Existing v1 sources,
results and unknown historical calls/exits remain preserved. Scientific and
execution acceptance remain false; this component yields no new OOD comparison.

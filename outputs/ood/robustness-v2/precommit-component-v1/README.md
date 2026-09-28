# V2 precommit component

Status:28 synthetic tests passed, actual exit0/no skips. This is a separately
versioned collector prerequisite. No real v2 state/candidate bank, JAX key table,
extension ledger, resource lock, model query or simulator call was created.
Execution and scientific acceptance remain false.

`precommit_bank.py` gives every captured state a fixed nominal index: host then
BCA collector,64 paired reset blocks each, captures100 then300. Both collectors
share each block's reset seed. Candidate/continuation/random-score streams retain
distinct state indices. All20 declared host/environment/seed identities are
checked against the frozen namespace; seed collisions with a supplied nonempty
predecessor inventory stop preparation without resampling.

Actual supplied continuation keys must be256x250x2 uint32 per pair. Duplicate
keys or collisions with supplied old/prior-v2 keys are refused, and every state's
250 keys are hash-bound into its precommit. This is array/inventory consistency,
**not verification of JAX key derivation or completeness of the old inventory**.
Those require independently bound runtime/stream inputs during actual integration.
The tests use synthetic integer key arrays and invented receipt hashes only.

The frozen candidate generator/selector is reused unchanged. Fixed slots are
anchor0, other near1..3, moderate4..7, strong8..11, host-policy12, BCA-policy13.
The original selector's compact output is expanded into these nominal slots.
If a fixed8,192-proposal pool misses a quota, those exact slots are marked missing;
the policy slots and other support groups never shift. No new proposal, replacement
or score-based choice is made. Missing captures must also be explicitly recorded
by the forthcoming collector; a nominal schedule is not a claim of collected data.

Each populated slot preserves exact proposed/sent/applied coordinates, clipping,
proposal index, support distance/band and the earliest equal applied float32 owner.
Aliases reuse one trajectory while retaining every nominal attribution. Signed
zero is preserved in coordinate bytes, while numerical signed-zero equality uses
the same alias rule as the frozen candidate core. The anchor must equal the first
distance-ordered neighboring recorded action. Native choices have their own
measured support band; their role does not establish in-distribution membership.

State-near/distant status, support thresholds, source/runtime/checkpoint/training/
support/state/execution/stream hashes and actual keys are bound into the artifact.
`check_precommit` rebuilds it from independently supplied bound inputs and compares
canonical bytes. Hash strings inside an artifact cannot certify their own provenance.
The later integration must read and validate the actual accepted files, verify
neighbor ordering/normalization and training-only thresholds, and independently
check native unit bounds before any physical execution.

`precommit_file.py` writes exclusively, flushes/fsyncs and checks file identity;
existing or partially written artifacts cannot be overwritten/repaired. Reads
require an external exact SHA256, bounded regular files and JSON depth/nodes/bytes,
and reject duplicate keys, nonfinite values and symlink paths. A durability failure
preserves the unacknowledged file and stops. The execution layer must own the
directory and maintain the lifetime shared lease; these helpers do not replace
that ownership or authorize science.

Tests cover complete/deficient banks, fixed native slot indices, exact aliases,
Hopper/Walker dimensions, wrong anchor/key rows, counterfeit component acceptance,
missing/extra pins, externally rebuilt equality, score/outcome argument refusal,
stream collisions, signed-zero bytes, exclusive writes, failed durability and
bounded/redirected/corrupt reads. The initial missing-module failure and22-test
development pass are preserved. Review then added explicit nearest-anchor and
key-receipt-schema refusal plus six tests; all28 final tests passed once.

Existing33 preparation tests and29 resource-component tests/sources remain frozen;
none was rerun. The prospective scientific plan is unchanged at
f11ebe8f8e3ccc4af10511ed1a241e8f7c5f0aa9f84770103dd6928e59b60756.

Next: integrate the actual collector/outcome driver and supervisor with the tested
resource extension; bind each physical call to its scope/token and durable full
input/applied/output/final-state evidence, including the separate constructor.
Pass native action1e-6/reward1e-7/full-first-repeat/source/runtime/checkpoint gates
and independent execution acceptance before creating a real extension or dispatch.
No accepted old outcome, ledger/archive audit, report or failed queue is reopened.

# NumPy provenance, portability and collector lineage

The dataset manifest's `generator.numpy_version` records the NumPy version that
generated its content. Loading preserves this value exactly. It need not equal
the current runtime's NumPy version. A version string is required and must contain
`major.minor.patch`, optionally followed by an `a`, `b` or `rc` prerelease number,
`.postN`, `.devN`, or a `+local.build` suffix. Missing values, non-string values,
whitespace and unrecognized suffixes are rejected. This is provenance syntax
validation, not a claim that every recorded version has been tested.

The sampler version, bit generator, seed-sequence configuration, named stream
addresses and each episode's reset seed still match the declared collector
exactly, including JSON types: integer fields cannot be Boolean or floating-point
aliases with equal numeric values. The source provenance, canonical-source digest, adapter-source digest,
canonical-loader digest and environment configuration also still match exactly.
The v5 dataset schema is `thrusterchain.continuous_trajectories.v3`; the v2
dataset and its recorded source hashes remain attached to the preserved release.
The v3 loader does not silently reinterpret v2 artifacts.

Revision v3 makes that JSON-type rule explicit. Earlier review demonstrated a
Boolean address accepted where the collector API requires an integer. No seeded
stream or lineage bypass resulted. Rejecting integer-valued JSON floats is a
deliberate stricter v3 contract choice; their prior acceptance was not independently
established as a defect. Object-key ordering remains irrelevant. The released
collector already emits integer addresses, so its existing artifacts are unchanged.

This schema represents the specific `continuous_box_sampler_v2` seeded collector.
Validation reproduces the canonical recorded-condition initial-state draw from
each declared reset seed and the requested-action draws from each episode's
declared action stream. Comparing these values detects a training trajectory
copied into a calibration row even when its trajectory ID, group ID, reset seed
and split label are left distinct. The check does not impose a general ban on
identical trajectories in externally collected datasets; a different collector
requires an explicitly supported contract instead of claiming this collector's
lineage.

Lineage checks use `rtol=1e-13` and `atol=1e-13` for float64 initial states and
requested actions. Existing transition and reward replay checks retain the same
tolerances. These permit small platform arithmetic differences while detecting
different seeded draws. Validation uses the pure canonical start sampler and
independent random generators; it does not construct an environment, call
`reset` or `step`, generate a dataset, fit a policy or write an artifact. The
inference guard remains active. Existing split grouping and leakage checks
remain in force; the validator does not reproduce a particular split permutation.

All recorded metadata, including the original NumPy version, remains part of
the canonical content digest. Loading one unchanged NPZ on another runtime keeps
its stored content digest. Editing the version field while retaining the old
digest is refused, even when the new version string is valid. Independently
regenerating data can produce different float64 bytes on another platform, so
equal seeds do not imply byte-identical files or identical content digests.
NPZ ZIP timestamps remain outside the content digest. The digest is an integrity
check, not an authenticated assertion that a collector actually ran.

The inherited v4 `tests/test_numpy_compatibility.py` refers to the archived NumPy 2.2.6
artifact and its original digest; it is historical, not a v3-loader compatibility test.
It also writes a synthetic NPZ with a
different recorded version to test the validation rule without changing
`np.__version__`. That synthetic fixture is not evidence of execution under a
different NumPy runtime. Actual runtime compatibility must be established by
running the tests and loading the original artifact with the other interpreter.

The focused test evidence is in `audits/revision_v2_numpy/`. The pre-fix failing
run is retained as `regressions_before.log`; the first passing run is
`regressions_after_attempt01.log`. The release's separate runtime audit records
actual interpreter versions and their compatibility results.

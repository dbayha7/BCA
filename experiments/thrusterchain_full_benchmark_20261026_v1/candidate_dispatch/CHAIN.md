# Mixed-engine on-disk chain verifier draft

`chain.py` verifies one cell's complete published checkpoint chain. It is
separate from `study.py` and is **not connected to the full benchmark**. It
does not validate GPU parity, evaluate policies, create a study, resume a run,
or establish the global train/evaluate barrier.

The draft expects a separately created manifest with exactly these keys:

```text
schema = "thrusterchain.mixed_study_manifest.v1"
spec = the validated frozen study specification
sources = study.source_identity()
runtime = study.environment()
dispatch_binding = candidate_dispatch.dispatcher.binding()
chain_verifier_sha256 = candidate_dispatch.chain.source_sha256()
```

The manifest is saved using `study.write_new()` as `manifest.json`, with its
exact SHA-256 in `manifest.sha256`. The caller must independently freeze and
provide the expected manifest digest to `chain.verify`; the adjacent sidecar
alone does not independently attest a mutable manifest. The dispatch binding
freezes the method routes, candidate and native sources, and JAX/runtime
configuration. GPU production integration must additionally validate the
runtime and the candidates on the actual GPU before creating such a manifest.

The output layout reuses `training/<case>/<method>/round_<rounds>/` with a
`checkpoint.json` and a `thrusterchain.study_chunk.v1` `receipt.json`. Its
`native_receipt` field holds the dispatcher-published receipt unchanged; the
CQL route's receipt is the distinct wrapper around its native receipt.
`chain.verify(root, frozen_manifest_sha256, case_id, method, terminal=True)`
requires the full declared budget. An explicit `terminal=False` permits only
a contiguous prefix of complete chunks. Either mode refuses terminal failure
markers or `.pending-*` artifacts until they are separately inspected; it
never recommends retrying a recorded failure.

The verifier checks the manifest, actual dataset behavior, seed, episode count,
physics, horizon and task semantics against the frozen case, the saved dataset
and train-split identity,
adapter ID, initialized learner seed and resolved configuration, all declared
optimizer schedules, each full checkpoint and receipt, predecessor digest,
route identity, and an independently advanced PCG64 sampling cursor. It
rechecks source, runtime, checkpoint/receipt bytes and dataset identity at the
end of the audit. For the full study, replaying sampled indices and hashing
all checkpoint files twice can incur substantial time and I/O. These checks
detect inconsistencies in the recorded chain; they are not protection against
an adversary who controls the externally supplied expected digest or a proof
of scientific correctness.

The bounded CPU-only test suite is `test_chain.py`; its current output is
`../validation/candidate_chain_cpu_20261027_v5.log` (11 tests). The tests stub
the GPU runtime check to a fixed CPU-only identity and include complete and
partial chains, recovery markers, actual mixed-engine CQL continuation,
rehashed receipt and sampler tampering, pinned optimizer schedules, and data or
receipt changes during verification. GPU validation and an integrated
collect/train/evaluate/report smoke remain required before any full launch.

# Disconnected mixed-engine dispatcher draft

This is an isolated CPU-validation draft. It is **not connected to the full
study launcher**, is not a frozen study manifest, and has not passed GPU
validation. Do not use it to launch the benchmark. The original compiled-scan
TD3+BC and CQL parity failures remain recorded and unresolved.

The fixed draft routes the separate TD3+BC and CQL cached-branch candidates and
uses the existing compiled engine for IQL, ReBRAC, and unified UniFloral. It
captures a prospective route/source/runtime binding, rejects checkpoint
identities from other routes before training continuation, checks independent
PCG64 cursor advancement, validates per-method critic-round counts, and checks
the actual engine-specific receipt rather than changing its schema. For CQL it
marks the published checkpoint's runner state and wraps the unchanged native
receipt in a separately named dispatcher receipt. The wrapper binds both the
native and published checkpoint digests, the frozen dispatcher identity, and
the predecessor digest. Resuming requires both the checkpoint and its prior
published receipt. An intervening native-runner training update erases the
dispatcher marker and therefore cannot resume through this draft.

The mixed-chain guard applies **only when this dispatcher is used**. The native
base runner can still accept a candidate checkpoint because adapter IDs are
shared; a future frozen study dispatcher must enforce its identity on every
training entry point and verify the entire predecessor chain. The immediate
predecessor check in this draft does not replace complete chain verification.
Native adapter restoration for frozen evaluation is separate and must first
verify the candidate's recorded identity.

Before integration, independently review the dispatcher and verifier, freeze
an explicit method-to-engine mapping and all candidate sources/runtime in a
new manifest, test failure and tamper handling in the actual study checkpoint
layout, validate evaluator identity and the global train/evaluate barrier,
and rerun full GPU parity, exact continuation, memory and representative
throughput checks. Then run a new isolated end-to-end collect/train/evaluate/
report smoke against the final integrated source. None of those steps is
established by the draft's CPU unit tests.

Bounded CPU evidence is recorded in
`../validation/candidate_dispatch_cpu_20261027_v5.log`.

# Integration requirements (not implemented)

This candidate remains unvalidated on GPU and is not used by the full study.
Do not change the full launcher until the unchanged original strict gate has
been evaluated on the candidate GPU path, resume and sampled-data checks pass,
and memory and representative throughput have been measured with GPU headroom.
The existing compiled-scan failure remains a separate recorded result.

If those checks pass, integrating this candidate requires a new, explicitly
versioned study configuration and manifest frozen before data collection:

1. Extend `study.source_identity()` to bind the candidate engine and any new
   dispatcher or verifier sources. It currently includes `compiled/` sources
   but does not include `candidate_td3/`. Freeze the engine choice for each
   method in the spec/manifest; do not silently select a backend at runtime.
2. `study.train_all` currently imports `compiled.engine.train_chunk` and calls
   it with `(dataset, method, seed, batch_size, steps, checkpoint)`. The
   candidate accepts `(dataset, seed, batch_size, steps, checkpoint)` for
   TD3+BC only. A future explicit dispatcher must route only the declared TD3+BC
   cells to it and preserve the other method routes, fixed update budgets,
   global training/evaluation barrier, and non-retry failure handling.
3. `study.verify_native_receipt` requires the original
   `thrusterchain.compiled_chunk.v1` schema, `method`,
   `compiled-native-scan-v1` identity, compiled-engine digest and
   `executed_optimizer_steps_total`. The candidate intentionally emits
   `thrusterchain.candidate_td3_chunk.v1`, its own engine identity and digest,
   dependency/runtime identities, and optimizer counts/deltas. It does not
   claim to be the original compiled engine. A separate explicit verifier must
   check its schema, all identities and digest, dataset/split, batch/draws,
   primary and total optimizer deltas, zero online interactions, and complete
   predecessor/checkpoint lineage. It must not weaken the original verifier.
4. Candidate checkpoints use the same `LearnerCheckpoint` and TD3+BC adapter
   identity, which allows the existing policy restore/evaluation factory to be
   exercised in a bounded integration test. Their runner metadata names the
   candidate engine and requires matching candidate source, dependency hashes,
   XLA/JAX/vendor settings, resolved JAX flags, versions, devices and GPU
   inventory for resume. The frozen training dispatcher must reject a
   checkpoint unless its engine identity matches the route in **both**
   directions, including candidate-to-base/native and base/native-to-candidate
   resume. The base adapter shares the same adapter ID and can restore the
   model for read-only evaluation; that ID alone does not authorize training
   continuation. In particular, the base offline runner can otherwise accept
   candidate metadata and produce a new checkpoint without the candidate engine
   marker. Preserve read-only native adapter restore for frozen evaluation,
   while enforcing the engine restriction at every training entry point used by
   the study. Test both reversed-chain refusals explicitly.
5. Before any full run, test the actual dispatcher and manifest with a new
   isolated collect/train/evaluate/report smoke, multiple chunks, saved/reloaded
   checkpoints, source and receipt tampering, failures and nonfinite updates,
   RNG and PCG64 sampling, and GPU memory/throughput. Confirm that the existing
   study remains untouched and its integrity checks still reject changed files.

CPU validation is recorded in `../validation/candidate_td3_cpu_tests_v5.log`.
It is not evidence that the GPU path or million-step trajectories are equivalent.

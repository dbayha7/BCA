# CQL candidate integration review

This is a review of remaining work, not authorization to modify the frozen
study or claim that the candidate has passed GPU validation. The candidate is
isolated and the full-study launcher is unchanged.

1. `study.py:source_identity` currently binds `compiled/` sources, not this
   candidate. A future, separately versioned study manifest must bind every
   candidate implementation and its native dependencies, runtime, precision,
   and per-method execution variant before any training. The candidate stores
   the adapter and native-update hashes and runtime identity in checkpoint
   metadata; the manifest and verifier must independently require those values at training,
   resume, reporting, and evaluation.
2. `study.py:train_all` imports `compiled.engine.train_chunk` unconditionally.
   `train_offline(..., CachedCQLAdapter, ...)` instead emits the standard
   `thrusterchain.offline_training.v4` receipt and stores sampler state in the
   native runner checkpoint. It is not a drop-in compiled receipt. A future
   dispatch must explicitly select a candidate per method and verify that
   receipt's own schema, digest, previous-checkpoint digest, train dataset and
   split, adapter identity, batch size, requested and executed updates, all
   optimizer counts and deltas, PCG64 stream, and zero online interactions.
   It must not relabel native receipts as `compiled-native-scan-v1` or weaken
   the existing verifier's lineage checks.
3. The candidate uses the native CQL adapter ID for checkpoint compatibility,
   but has an additional `adapter.execution_identity`. The current checkpoint
   verifier checks the native adapter ID and compiled engine only; it must
   separately require the candidate identity and reject native/candidate or
   cross-version resume. The original native adapter shares the ID and does
   not itself enforce the candidate marker, so direct native-runner resume
   must be guarded by the integration. The native runner can accept a candidate
   checkpoint and replace its runner state, erasing an engine marker. Never
   allow candidate-to-native or cross-variant training resume, even when the
   shared adapter ID makes restore succeed. The standard PCG64 runner validates seed,
   dataset, split, batch size, settings, cursor and continuation state, but the
   study-level verifier must check them too.
4. The existing evaluator's native checkpoint factory can restore candidate
   numerical state and preserve metadata in a bounded CPU test. Evaluation
   must nevertheless check the execution identity against the frozen manifest
   before restoring or interpreting scores. Native restore is permitted only
   for this read-only evaluation after identity validation; it must not be used
   to continue training. No changed-source checkpoint may be accepted merely
   because its adapter ID matches.
5. After GPU memory is available, require the unchanged strict state, metrics,
   keys and counters on representative seeds, byte-identical continuation,
   finite-failure rollback, real batch-256 timing, full-sized chunk resource
   checks, and a fresh collect/train/evaluate/report smoke. The currently
   measured CPU eager timings imply a long serialized study and do not predict
   GPU runtime. Validation artifacts must record exact source/runtime hashes.

The existing evaluator's source inspection, receipt checks, and study manifest
do not currently certify this candidate. No full training has been started.

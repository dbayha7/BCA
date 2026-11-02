# Prospective bounded same-process cache and memory probe

This is isolated validation, not benchmark training or evaluation. Generate
12 independent 1,000-episode smoke datasets using the four study behavior
families and paired data seeds 62081, 62082 and 62083. Each dataset has 60,000
training transitions and distinct provenance from the planned study data. Keep
all twelve datasets loaded in host memory simultaneously, as `study.train_all`
does. Use distinct paired training seeds 62091, 62092 and 62093.

In one GPU process, traverse the 12 datasets in behavior-then-replicate order
and the five methods in `spec.full.json` order. For each of the 60 cells run a
single two-critic-round chunk at batch size 256: two primary steps except the
unified method, which takes one. Verify the native receipt and optimizer counts,
write the checkpoint to a new directory, and confirm its loaded digest. Record
the compiled program-cache count by method after every cell, process resident
and peak host memory, JAX allocator statistics, whole-device sampled memory from
`nvidia-smi`, wall time, the per-cell dataset and checkpoint identities, and the
source/runtime fingerprint. Execute `source_identity()` and `environment()`
before and after each cell, as the study runner does. The output directories,
source files and dataset files are new and never replace prior evidence.

The GPU and allocator samples are point-in-time observations, not guaranteed
peaks. This checks whether the compiled cache grows across twelve distinct
normalizations in one process; two-round chunks cannot validate the time or
peak memory of repeated million-round training. Other GPU users can affect the
observed device baseline. A failure or partial result must be reported, with
no silent retries or alterations of completed outputs. No final evaluation or
scientific performance comparison is permitted.

# Full-size GPU resource and continuation probe, 2026-10-27

This is a bounded, isolated validation, not the scientific benchmark. Preserve
all existing studies and previous smoke/diagnostic artifacts. Run only after
the GPU is released by the other validations and the candidate compiled engine
source is stable. Record the exact plan, probe, compiled engine, source/runtime,
and dataset identities before the GPU work. Do not use benchmark dataset or
learner seeds, and do not evaluate or compare learned policies in this probe.

Generate a separate `mixed_random_strong`, finite-horizon, `smoke`-provenance
dataset of exactly 1,000 episodes with data seed 62077. Save it inside a new
validation output directory. Verify dataset identity, 100-step horizon and
the whole-episode training split (expected 60,000 training transitions). Use
training seed 62078, batch size 256 and the candidate study's current compiled
runner. For each of the five methods, launch a fresh process and execute two
successive chunks of 10,000 critic rounds apiece. Thus each method consumes
20,000 critic rounds, with 10,000 primary updates per chunk except unified
UniFloral, which takes 5,000 primary updates and 10,000 critic rounds per chunk.
The maximum is 100,000 critic rounds across all methods. No evaluation episodes
or online parameter updates are permitted. Stop a method on an exception and
record failure without reporting its missing work as successful. Preserve all
completed checkpoints, lineage and diagnostic output.

For both chunks, record elapsed wall time (includes initialization/restore,
sampling, device transfer, compilation if any, synchronized execution and
checkpoint construction), optimizer deltas, dataset draws, and native receipt.
The runner's per-update device-side finite check must succeed. Independently
restore each serialized checkpoint and verify its hash, optimizer counts,
sampler continuation, predecessor identity and native receipt with the study's
verification functions. Verify the GPU backend and runtime match the study
manifest rules. Preserve source and environment hashes and record starting and
peak device allocation from JAX `memory_stats()` if available; separately
sample whole-device `nvidia-smi` used/total memory at one-second intervals and
record baseline and peak. Whole-device memory includes any unrelated graphical
usage, is sampled rather than guaranteed true peak, and is not allocator-only
memory. Missing memory metrics are recorded as unavailable, not zero.

The two full chunks check both initial and resumed resource behavior; the
existing exact short `2+2` versus `4` compiled continuation test remains the
determinism check. These measurements do not establish general full-study
throughput, final policy quality, scientific validity or equivalence to eager
execution. A partial or failed probe does not clear full-study readiness.

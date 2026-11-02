# TD3+BC cached-branch candidate

This is an isolated execution candidate. The existing compiled scan, study
runner, and prior studies remain unchanged. The candidate evaluates the native
critic update eagerly and creates the delayed actor branch closures once per
learner; the minibatch is a conditional operand. The candidate source SHA-256,
hashes of adapter, host, runtime, data and runner dependencies, and numerical
execution settings, device and package information are recorded in each
checkpoint and receipt. Resume requires the same identity. The runtime record
includes the XLA, JAX, visibility and vendor math environment overrides used by
the study manifest, the resolved JAX flags, package versions, Python version,
and GPU/driver inventory when on GPU. A missing GPU inventory fails closed.

The candidate is **not cleared for GPU use or full experiments**. The original
compiled scan failed the retained strict parity gate. CPU-only bounded checks
are in `test_engine.py`; GPU parity, throughput, memory and end-to-end runner
integration remain pending. Do not substitute this candidate into a study
without those checks and an explicit, documented integration.

Run the bounded CPU tests from the parent study directory in the compatible
environment with `JAX_PLATFORMS=cpu` and `JAX_DEFAULT_MATMUL_PRECISION=highest`:

```text
python -B -m unittest candidate_td3.test_engine -v
```

The latest test evidence is `../validation/candidate_td3_cpu_tests_v5.log`.
See `INTEGRATION.md` for the manifest, dispatcher and receipt work needed if
the pending GPU validation succeeds.

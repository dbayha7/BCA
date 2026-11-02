# Unified UniFloral bounded CPU/GPU validation

The completed isolated-runtime and Zeus GPU results are recorded in
[RESULTS.md](RESULTS.md). The CPU/GPU receipts and the separate elementwise
follow-up retain their exact source and runtime identities.

This additive harness tests the existing ThrusterChain v6 unified TD3+BC
composition; it does not change the v6 benchmark, the pinned upstream source,
or previous studies. No full training, sweep, or scientific comparison occurs.
The benchmark and pinned source must retain their expected v6 source hashes.

## Fixed protocol

Use the same isolated environment for the CPU reference and GPU test. The CPU
run saves an initialized immutable learner checkpoint and fixed-batch numerical
reference. The GPU run restores those exact CPU checkpoint bytes, so model,
normalization and random state start identically. Each run also executes the
existing four v6 unified tests (native sampler vs selected batch, independently
computed actor and critic targets, terminal/time-limit masks, optimizer counts,
exact resume, frozen evaluation). The harness performs one additional actor
update and two critic updates on the same five actual finite-task training rows.
The validation dataset contains ten smoke episodes and is never used for model
selection or a research claim.

The fixed critic action-gradient test uses direction `[1, 0, 0, 0]`,
central differences at `h=0.001` and `h=0.002`, and compares each directional
finite difference to autodiff with `atol=0.001`, `rtol=0.03`. These are
predeclared float32 diagnostic tolerances, not a claim of exact derivatives at
ReLU or action-clipping boundaries. CPU/GPU paired checks use `atol=0.0001`,
`rtol=0.005` for Q, loss metrics, and update norms; and `atol=0.001`,
`rtol=0.03` for the gradient quantities. No tolerance is adjusted after seeing
results. A failure is reported rather than silently relaxed. Tests compare
conditional numerical behavior, not entire training trajectories or returns.

GPU runs require `JAX_PLATFORMS=cuda`, `JAX_DEFAULT_MATMUL_PRECISION=highest`,
and a matching CPU reference. The adapter's runtime gate must confirm its pinned
JAX/NumPy upper bounds, unless `--allow-incompatible-for-debug` is explicitly
provided, in which case the result is labeled debug-only. The flag checks only
JAX and NumPy upper bounds; the caller must separately check all dependency
constraints (`pip check` is useful but not a performance or semantic test).
Actual JAX backend/device, precision and all dependency versions are recorded.

The first CPU-only debug self-check used direction `[0.4, -0.3, 0.2, 0.5]`,
which nearly canceled the four component derivatives. The axis direction above
was fixed before GPU testing; the tolerances were not relaxed.

```text
JAX_PLATFORMS=cpu JAX_DEFAULT_MATMUL_PRECISION=highest python run_validation.py --mode cpu --baseline-root /path/to/experiments/thrusterchain_benchmark_v6 --benchmark-root /path/to/experiments/thrusterchain_unifloral_gpu_compat_v1 --output /new/path/cpu
JAX_PLATFORMS=cuda JAX_DEFAULT_MATMUL_PRECISION=highest python run_validation.py --mode gpu --baseline-root /path/to/experiments/thrusterchain_benchmark_v6 --benchmark-root /path/to/experiments/thrusterchain_unifloral_gpu_compat_v1 --cpu-reference /new/path/cpu --output /new/path/gpu
```

Outputs use exclusive new directories. The receipt identifies source, fixture,
runtime, devices, optimizer counts, numerical results and individual tests.
The CPU reference is required to match source and dependency identities and the
actual deterministic dataset. A CUDA test must fail if JAX falls back to CPU.

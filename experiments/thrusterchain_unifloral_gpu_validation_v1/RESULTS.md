# Unified UniFloral: isolated runtime and bounded GPU validation

## Outcome

On 2026-10-26 the bounded checks passed on Zeus's NVIDIA GeForce RTX 5070 Ti
Laptop GPU (`cuda:0`, compute capability 12.0, driver 610.62). The isolated WSL
Ubuntu-22.04 interpreter is
`/home/david/.cache/thrusterchain-unifloral-gpu-v1/bin/python`. It has Python
3.10.12, JAX/JAXLIB and CUDA plugin/PJRT 0.5.3, NumPy 1.26.4, SciPy 1.15.3,
Flax 0.10.4, Optax 0.2.4, Chex 0.1.89, actual Distrax 0.1.5, TensorFlow
Probability 0.25.0 and Gymnasium 1.0.0. The final installed distributions are
in [pip-freeze-final.txt](results/zeus_jax053_v1/pip-freeze-final.txt);
`pip check` reported no broken installed requirements.

This satisfies the pinned upstream **JAX <0.6 and NumPy <2 upper bounds** and
the separately versioned compatibility package's requirements. It is a
computational integration runtime, not an installation of the full upstream
D4RL/MuJoCo/CLI stack. The freeze is an environment record and contains a
machine-local source path; it is not a portable lock file.

## Preserved source and compatibility copy

The v4, v5 and v6 source manifests were rechecked: respectively all 52, 71 and
78 recorded files matched. The 37 functional Python/JSON files in the separate
`../thrusterchain_unifloral_gpu_compat_v1` copy match v6 byte for byte. The
copy changes only package name/version and NumPy dependency metadata, the
corresponding requirements file and wheel-version assertion, plus a README
explaining the copy. The source gate verifies the original v6 manifest and the
three explicitly hashed source-manifest/test deltas. The pinned UniFloral source
remains hash `14a9bcdbc6d96fad85544a9d98957e98a7e46db13b89d74d54e3c5ce246517c8`.
These checks establish source identity, not a byte-by-byte audit of every
historical result or untracked asset.

## Executed checks

- The compatibility copy's `validate_v6.py` ran 241 tests: 178 passed and 63
  historical v4 artifact tests were explicitly skipped. An initial run failed
  its wheel-build setup because the new environment lacked `wheel`; the initial
  log is retained. Installing `setuptools==80.9.0` and `wheel==0.45.1` in the
  isolated environment resolved it; the subsequent complete applicable suite
  passed and `pip check` remained clean. See
  [initial log](results/zeus_jax053_v1/contract-tests.log) and
  [final log](results/zeus_jax053_v1/contract-tests-final.log).
- The [fixed bounded protocol](README.md) passed four unified integration tests
  on CPU and four on GPU. They cover the selected deterministic unified TD3+BC
  composition, actual native fixed-batch update parity, terminal versus timeout
  targets, actor-first one-actor/two-critic cadence, checkpoint resume and frozen
  evaluation. The GPU run restored the exact SHA-verified CPU initial checkpoint,
  used the same dataset and batch, explicitly required `JAX_PLATFORMS=cuda` and
  `JAX_DEFAULT_MATMUL_PRECISION=highest`, and did not fall back to CPU. See the
  [CPU reference](results/zeus_jax053_v1/cpu_final/reference.json) and
  [GPU receipt](results/zeus_jax053_v1/gpu_final/receipt.json).
- The GPU critic action-gradient directional derivative was approximately
  `-0.008698175`; central finite differences were approximately `-0.008698552`
  and `-0.008676200`. Both passed the predeclared tolerances. CPU/GPU critic
  losses, actor losses, gradients and update norms also passed their declared
  comparisons. These are fixed-fixture checks, not policy-quality results.
- A separate predeclared [elementwise check](ELEMENTWISE_PLAN.md) compared 51
  leaves after one matched outer update, including actor, critic, both targets,
  optimizer state and RNG. All passed `atol=1e-4, rtol=0.005`; integer and RNG
  leaves matched exactly. The largest absolute difference was about `7.15e-7`
  in critic optimizer state. See the
  [elementwise receipt](results/zeus_jax053_v1/elementwise_gpu/receipt.json).
- A further predeclared [feedback-data integration check](END_TO_END_PLAN.md)
  generated ten finite-horizon feedback episodes, trained the unified adapter on
  the GPU for exactly two actor and four critic updates with no out-of-constraint
  override, reloaded its checkpoint, and evaluated one complete episode in each
  of the four v6 dynamics scenarios. Offline training used zero environment
  interactions; the four evaluations totaled 400 steps and left the frozen
  policy identity unchanged. The [GPU smoke receipt](results/zeus_jax053_v1/feedback_gpu/receipt.json)
  records the actual dataset, source, checkpoint, runtime and results. These are
  wiring checks, not selection, holdout, or performance evidence.

No full training run, tuning sweep, scientific performance experiment or other
study was started. The checks validate one unified configuration and bounded
CPU/GPU execution, not all configurations, full-trajectory numerical identity,
research-scale throughput, or policy performance. The current v6 runner uses a
different sampling stream from the native host; numerical parity is conditional
on the matched batch.

## Reuse

Run the documented commands in [README.md](README.md) with the isolated
interpreter above and fresh output directories. Keep `JAX_PLATFORMS=cuda` and
`JAX_DEFAULT_MATMUL_PRECISION=highest` for the validated GPU path. The existing
v6 package still declares NumPy 2 or newer; do not install it into this runtime
as though it were the NumPy-1 copy. Full experiments require their own resolved
protocol, budgets, and user authorization.

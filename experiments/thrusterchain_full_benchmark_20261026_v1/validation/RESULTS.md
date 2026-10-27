# Full benchmark readiness, 2026-10-26

**Status: blocked; full benchmark not started.** The selected design is in
`../spec.full.json` (SHA-256
`74315c19bbf0e7ee42c02b9eb09366485341e6029be5d9df4fac419db755ab3b`).
The study manifest has not been prepared for a full run. Existing studies were
not used as write targets. The bounded smoke outputs are not benchmark results.

The planned matrix is four behavior datasets by three independently generated,
paired dataset/training-seed replicates and five methods, for 60 policies. Each
policy has a budget of one million critic-update rounds. Unified UniFloral's
TD3+BC composition takes 500,000 primary updates, each with two critic rounds;
this does not match replay draws or compute across methods. The final checkpoint
is evaluated on four starting-condition families crossed with four dynamics
scenarios and five episode seeds: 4,800 episodes if every cell succeeds. These
scenario families were previously inspected and are not unseen tasks.

## Validation evidence

- The five-method GPU suite in `compiled_gpu_suite.log` used the predeclared
  eager-versus-compiled `atol=1e-6, rtol=1e-5` check. Fixed terminal batches,
  exact compiled checkpoint continuation, and first/resumed receipts passed
  for all five methods. IQL, ReBRAC, and unified UniFloral passed the tested
  numerical cases. TD3+BC failed on a real-sampling fixture, with a maximum
  difference of `2.5108457e-6`; CQL failed a cross-dataset fixture, with a
  maximum difference of `1.2535602e-6`. Each failed array had one element
  outside the tolerance. The tolerance was not changed.
- `td3bc_parity_cuda.json` and `td3bc_parity_cuda.log` preserve the independent
  TD3+BC diagnosis. Tiny critic and actor-gradient differences were observed,
  but the cause is not established and the effect on full training is unknown.
  The alternative XLA settings in `xla_no_fusion_gpu.log` and
  `xla_alternate_gpu.log` did not clear either failed case.
- `smoke_20261026_v2/report.json` records an isolated end-to-end plumbing test:
  all five methods, four critic rounds each in two chunks, 16 scenarios, 80
  evaluation episodes, and no failed cells. The first isolated smoke stopped
  because a test source was changed after its manifest was created; its
  artifacts are preserved in `smoke_20261026_v1`. The second smoke used a
  stable snapshot. Neither smoke establishes benchmark performance or resolves
  the parity gate.
- The bounded eager fallback in `eager_finite_gpu.json` preserved bitwise state,
  metrics, keys and counters in four updates and detected injected nonfinite
  state and metrics. Twelve timed batch-256 updates averaged 0.524 seconds
  for the native adapter and 0.479 seconds for the prototype. A naive
  extrapolation of the latter to TD3+BC's twelve million-update runs is about
  66 days, excluding other methods and evaluation; it is not a reliable ETA.
  The prototype is not integrated into the study runner.
- `compiled_throughput_cuda.json` contains bounded timings for the faster
  compiled path. Because the strict parity gate failed, those timings do not
  clear it for the study.

The next technical step is to resolve the failed parity cases or establish and
validate a separately specified compiled implementation with an appropriate
scientific equivalence protocol. Do not interpret this readiness report as a
completed benchmark or launch the existing compiled study as if it had passed.

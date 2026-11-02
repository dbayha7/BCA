# Benchmark readiness, 2026-10-27

**Status: original strict parity gate failed; full benchmark not started.** The
user chose to continue pursuing the original gate, rather than proceed using a
separately specified compiled implementation. The agreed `../spec.full.json`
remains unchanged, SHA-256
`74315c19bbf0e7ee42c02b9eb09366485341e6029be5d9df4fac419db755ab3b`.
No completed benchmark results are available. The earlier readiness record and
all previous studies and validation outputs remain preserved.

## Original gate

The current-source GPU suite is `strict_suite_20261027.log`, with process exit
status recorded in `strict_suite_20261027.status`. At the unchanged elementwise
`atol=1e-6, rtol=1e-5`, TD3+BC real-runner sampling and CQL cross-dataset tests
still fail, with maximum array differences of approximately `2.51e-6` and
`1.25e-6`, respectively. The same TD3+BC diagnostic also fails on CPU
(`td3bc_parity_cpu_20261027.json`). RNG and optimizer counters match in the
recorded bounded trajectory checks, but their model and optimizer states are
not identical. In the three-seed CQL 64-step diagnostic the largest observed
state-leaf differences were approximately 0.00118, 0.00316, and 0.000747;
these are descriptive observations, not long-run equivalence claims.

Attempts using strict floating-point/compiler settings and optimization
barriers did not resolve the TD3+BC failures. A CQL random-barrier candidate and
a TD3+BC cached-branch candidate were stopped as bounded probes before
producing parity results; neither can be reported as a passing or failing
parity test. A separate CQL fixed-key random-array check found exact agreement
for all six checked arrays, which does not rule out other sources of error.

Two separate cached-branch candidates subsequently passed bounded **CPU**
comparisons at the original tolerance. TD3+BC kept eager critic arithmetic
and cached its delayed-actor branches; CQL kept native eager arithmetic and
cached its target-update branches. Both passed two 16-step, batch-256 CPU
fixtures, including state, metrics, random keys, counters and finiteness.
The isolated candidates and further sampling/resume tests are documented in
`../candidate_td3/` and `../candidate_cql/`. Their GPU parity, long-run
throughput, and integration into the frozen study launcher remain untested.
They do not make the compiled-scan gate pass.

The TD3 candidate's final CPU suite is
`candidate_td3_cpu_tests_v5.log` (7 tests); the CQL candidate's is
`candidate_cql_cpu_runtime_identity_v2_20261027.log` (3 tests). Each now records
compiler and runtime settings and refuses tested runtime drift. Independent
review found that the base adapters share IDs with the candidates: future
training dispatch must reject mixed candidate/native checkpoint chains rather
than rely on adapter IDs. Native restoration for frozen evaluation is a
separate operation. See each candidate's `INTEGRATION.md`.

`../candidate_dispatch/` contains a disconnected mixed-engine dispatcher
draft. Independent review exposed a candidate-to-native-to-candidate CQL
checkpoint bypass in its earlier version; the revision adds a training marker
and a wrapper binding the unmodified native receipt to the published checkpoint.
An independent CPU retest rejected the reproduced bypass. The final CPU suite
`candidate_dispatch_cpu_20261027_v5.log` passed 3 tests covering all five
routes, continuation, receipt tampering and cross-engine resume refusal. Full
on-disk predecessor-chain validation remains required. This draft is not
integrated into the full launcher or validated on GPU.

A separate on-disk chain verifier has since been drafted and its final-source
CPU suite passed 11/11 (`candidate_chain_cpu_20261027_v5.log`). It checks the
predecessor chain, dataset case metadata, engine-specific receipts, sampler
state, pinned optimizer schedules, terminal failures, interrupted publications
and file changes during verification. The verifier remains disconnected from
the full-study launcher and has not been tested on GPU.

## Other bounded evidence

- `td3bc_independent_gpu_20261027.json` and
  `cql_reference_gpu_20261027.json` contain independently specified toy-model
  objective/gradient/optimizer checks. The CQL full-network local-gradient and
  entropy-branch checks are in `cql_network_gradient_gpu_20261027.json` and
  `cql_entropy_gpu_20261027.json`. They passed their separately declared
  criteria; they do not replace the original gate. The full-network CQL
  fixture uses the same numerical roots as the planned third dataset/training
  replicate, and is not an independent-seed fixture.
- `resource_20261027_v1/summary.json` records all five methods completing two
  full-size, batch-256, 10,000-critic-round chunks each on a separate smoke
  dataset, including finite, sampler, optimizer and receipt checks. It is a
  resource and resume probe, not a benchmark performance evaluation.
- `smoke_20261027_v3/report.json` records the updated source/runtime's isolated
  end-to-end smoke: five methods, four critic rounds each, 80 evaluation
  episodes and zero failed cells. It is plumbing evidence only.
- `cache_memory_20261027_v1/summary.json` records 60 small cells in one process
  across 12 separate full-size smoke datasets. The sampled peak was 3,450 MiB
  whole-device GPU use, including graphics, and about 2,104.5 MiB host RSS;
  sampled peaks can miss transients. This is not a full-duration stress test.
- `../study.py` now binds compiler flags, device visibility, vendor math
  overrides, and resolved JAX settings into the frozen runtime identity, and
  rejects disabled JIT and enabled x64. The updated study contract tests passed
  12/12. A running study cannot silently resume with changed recorded settings.
- A later independent review found that dataset bytes and receipts were not
  bound to their planned behavior/seed/episode metadata. `../study.py` now
  validates that metadata and the task configuration during collection and
  loading; the updated bounded CPU study suite passed 14/14. The declared
  metadata checks do not prove the behavior policy generated every action.
  This source change requires a fresh integrated GPU smoke before launch.

After the GPU probes stopped, `nvidia-smi` initially reported approximately
11.5 GiB of 12.2 GiB in use, with the allocation source not established.
Later memory use decreased, but utilization remained high from another active
workload. Further GPU probes were paused to avoid interference; recheck the
device before resuming.

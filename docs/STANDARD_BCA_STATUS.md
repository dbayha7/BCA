# Standard BCA: execution status

The September 27 study compares each plain host with BCA **without an importance
tilt**, retaining Bayesian bootstrap masses and both radius components. Each
run has 1M host updates. The frozen matrix contains 280 runs / 315 actor
trajectories across four hosts, seven datasets and five seeds.

| Execution lane | Hosts | Status |
|---|---|---|
| Local RTX 5070 Ti | TD3+BC, CQL | First TD3+BC Hopper host verified complete at 1M; its no-IW BCA counterpart is running. |
| Cluster A100 | ReBRAC, IQL | Job 27045057 passed all 14 mandatory data cells with actual gate exit 0. First ReBRAC Hopper host verified complete at 1M; its no-IW BCA counterpart is running on str-gpu13. |

Two of 280 declared physical runs (two of 315 actor trajectories) are independently verified complete. The first TD3+BC Hopper host final mean is **65.2157311**; its periodic-curve mean is **57.1475157**. See the [complete host readout and plots](../outputs/standard_bca/td3_bc/hopper/host/s202609171/README.md) and [audit receipts](validation/standard-first-td3-host.json). Four further host seeds and all paired BCA comparisons remain pending for this cell.

The first ReBRAC Hopper host final mean is **102.0239558**; its periodic-curve mean is **92.5215207**. See its [host readout and plots](../outputs/standard_bca/rebrac/hopper/host/s202609171/README.md) and [audit receipts](validation/standard-first-rebrac-host.json). Its paired BCA comparison and four further host seeds remain pending. The two verified results are different hosts and do not supply a paired method comparison.

A checkpoint below 1M is progress, not completion. No completed paired seed comparison,
standard-BCA advantage, or real OOD result is claimed here.

## Accepted preparation

- 280 typed configurations and 47 exact integration-map snippets verified.
- Eight host/BCA CPU numerical comparisons match preserved implementations
  exactly, including BCA radius refreshes and decoded checkpoint counters.
- All 28 host/dataset preparations match the saved data identities; paired
  host/BCA data agree. IQL Pen reservation feasibility passes.
- Four no-IW known-answer tests, three harmless launcher lifecycle tests,
  and 39 OOD protocol/arithmetic/figure regression tests pass.
- IQL host and shared-Q/V BCA execution/checkpoint fixtures pass eight mock
  updates each, with zero simulator outcomes.
- Local TD3+BC/CQL GPU fixtures pass. All four plain host files and all 40
  bundled Unifloral files remain unchanged.

## Preserved cluster preflight attempts

1. Job 27044796: missing `GL/osmesa.h`; exit 1 before learner/simulator work.
2. Job 27044823: headers fixed; installed `patchelf` absent from PATH; exit 1
   before learner/simulator work.
3. Job 27044828: older JAX plugin appeared through inherited packages; typed
   validation exited by signal 11 (Slurm wrapper 245). The exact crash cause
   is unresolved. No learner/simulator phase was reached.

4. Job 27045031: **all five phases passed**, actual wrapper exit 0: typed
   protocols, no-IW/lifecycle tests, seven explicit simulator test steps, and
   ReBRAC/IQL GPU fixtures. This supplies engineering validation, not OOD results.

The initial isolated runtime inherited the old environment's packages. A second,
fully isolated runtime now passes with the repository's pinned versions and
MJRL commit `3871d93763d3b49c4741e6daeaebbc605fe140dc`. Gym, D4RL and MJRL
Python source files are byte-identical across the local and cluster environments.
The original runtime and all failed attempts remain intact. Graphics installation
finished and its files were verified, but its first shell wrapper had a quoting
error and exited 1; its original JSON must not be read as proof of installer exit.
These are environment/setup attempts, not scientific repetitions.

## Reproduction and next gates

Scientific source commit: `6e912b0ec3e3346ba6c628b6eef6fddedc74ffc0`.
All 108 frozen source files match that commit. Frozen manifest SHA256:
`13ae3e6693d1cc71228e7b676243232c52ac6766c6a2c9b0623aaac1145f97fe`.
See [manifest and runner](../experiments/standard_bca/README.md) and
[validation evidence](validation/standard-bca-preflight.json).

Cluster runtime, simulator identity and GPU checks passed. Its mandatory
14-cell data gate passed in job 27045057 before the cluster training controller started. Both methods matched the accepted local fingerprints in every cell. Live command/start/group identities were checked on the compute node; a login-node PID lookup is insufficient. See [data-gate and live-process evidence](validation/standard-bca-live-gate.json). Real OOD collection follows
verified 1M checkpoints and adapter checks. TD3+BC/ReBRAC Hopper/Walker adapters
have prior engineering evidence; IQL/CQL and remaining environments still need
their own gates. ReBRAC's missing recorded-next-action coverage target remains
unavailable unless a valid target contract is established.

Importance weighting is a later study. Its recipe will be chosen using separate,
predeclared development evidence rather than these final test outcomes. Old 1M
and 100k results retain their original identities; the halted old queue and two
old behavioral recoveries remain untouched. At most two GPUs total are used.

The compressed scientific manifest and implementation are published. See the
[cluster attempt receipts](validation/standard-bca-cluster-preflight.json) for
the original failures, separately documented corrections and actual exits.

The frozen `configs/sources.json` records its creation-time preflight state.
The completed 28-cell acceptance is in the newer validation receipt linked above;
the frozen snapshot is deliberately preserved.

## OOD process-receipt preparation

`experiments/ood/standard_receipt.py` binds the standard runner's separate actual
worker exit to its dispatch, declared row and result. It rejects failed, interrupted,
stale and mismatched metadata, and refuses to overwrite a receipt. Its current
scope is TD3+BC/ReBRAC Hopper/Walker. This bridge performs no model query or
simulator step and leaves scientific acceptance pending: checkpoints, journals,
source/data contents, paired preparation and simulator parity still need independent
verification. See [synthetic receipt tests](validation/standard-ood-receipts.json).

The original login-node process observation and direct compute-node SSH timeout
are retained in the monitoring snapshot. Process identity was subsequently verified
with a read-only Slurm step in the existing allocation, requesting no GPU. Neither
inspection attempt changed scientific execution.

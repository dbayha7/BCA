# OOD v2 injected collector/outcome driver implementation plan

**Goal:** Implement the fixed state-capture and14-slot finite-horizon control flow with durable reservation/evidence ordering, without running science.

**Architecture:** A separate typed evidence encoder and recorded-step bridge calls the already tested extension ledger. Pure injected collector/outcome drivers accept an initialized simulator interface, sealed deterministic policy interfaces and durable stores. They never construct or import a simulator/model/resource writer themselves. Synthetic doubles and a temporary synthetic extension journal exercise the integration. Real execution acceptance remains separate.

**Tech Stack:** Python3.10, NumPy and standard-library helpers; frozen v2 candidate/precommit/resource components reused without edits. No subagents under standing direction.

## Files and checks

- Create work/ood_robustness_v2/recorded_step.py: bounded typed bytes, append/read-bound evidence adapter, strict token/scope mapping, reserve-before-input/step and full-post-state evidence before completion.
- Create work/ood_robustness_v2/collector_driver.py: fixed100/300 captures for64 paired blocks under each collector; explicit terminal/missing rows; no replacement.
- Create work/ood_robustness_v2/outcome_driver.py:14 fixed slots, missing rows, exact applied aliases,250-total-step raw returns, full first/repeat equality and saved full final states.
- Create work/ood_robustness_v2/test_drivers.py and preserve expected missing-module failure. Use only temporary synthetic files/callbacks, no actual model, simulator, lock or ledger.
- Save actual exits and source hashes in monitor_20260928T190657Z; publish owned small code/evidence after frozen-source checks.

## Checklist

- [ ] Test reservation happens before evidence and physical callback, and completion only after input/raw-triplet/output/full-after-state records are durably acknowledged. Wrong scopes/tokens/actions, missing raw outputs and write failure must stop with charged pending calls and no automatic retry.
- [ ] Test collection's exact capture indices, early termination at/before captures, paired reset seeds, policy seals and no state replacement.
- [ ] Test outcome first restore, first applied action, repeated record/full-state equality, restored first end state, horizon including first, native termination/time-limit stopping and sequential return arithmetic.
- [ ] Preserve all14 rows including missing/aliases, bind aliases to saved owner results, prevent duplicate trajectories and keep host reference12/common anchor0 distinct.
- [ ] Require bound precommit/snapshot/key inputs before starting, detect source mutation, and save actual full final-state contents before a completed action row.
- [ ] Review and freeze only the new components. Real constructors, source/runtime/checkpoint/stream/path gates, native parity, supervisor and independent engineering/execution acceptance remain pending.

An injected double cannot establish real physics or prove that an arbitrary callback is trustworthy. Actual integration must bind the accepted native adapter and durable archive to these interfaces, including exact before-physics applied-action recording. Constructor evidence needs a distinct native pre-wrapper schema and remains unimplemented here. Do not create a real extension journal or run physical engineering calls from these synthetic tests.

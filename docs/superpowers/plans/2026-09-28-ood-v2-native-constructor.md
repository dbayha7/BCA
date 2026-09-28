# OOD v2 native constructor evidence implementation plan

**Goal:** Archive the actual native constructor call without inventing wrapper state, and charge it through the new extension before physics.

**Architecture:** A new injected observer wraps exactly the declared native step and physics methods for the duration of one constructor factory call. It copies native data/model/RNG availability before and after, saves actual sampled action and intercepted control before physics, records returned observation/reward/done/info, and validates saved native arithmetic before ledger completion. It never changes the action, random stream or native implementation. Pure synthetic classes test failure and ordering; no actual simulator construction this step.

**Tech Stack:** Python3.10, NumPy, standard-library context-managed method observation and existing closed evidence/ledger interfaces. No subagents under standing direction.

Read-only installed-source inspection established that this Gym constructor calls
`action_space.sample()` and then `self.step(action)` after setting its native
action space. It does NOT impose a zero action. The native environment RNG may
not yet exist; its presence must be observed without invoking a lazy RNG property.
The wrapper's time-limit/reset/termination state is unavailable at that native
point. Actual random action bytes and after-sampling RNG state are evidence;
pre-sampling entropy must never be invented or claimed reconstructed.

## Files and checks

- Create work/ood_robustness_v2/native_constructor.py: native-only snapshot, truthful RNG availability, one step/four physics guard, saved arithmetic checks and reserve-before-factory integration.
- Create work/ood_robustness_v2/test_native_constructor.py: synthetic native data/model, inherited physics interface and constructor factories. No real imports/physics/lock/ledger.
- Preserve initial missing-module failure, then test Hopper and Walker dimensions, actual nonzero sampled controls, full native captures, reward/observation/health checks, no/extra/wrong-physics calls, archive/factory failures, changed model, missing fields and restored methods.
- Save actual receipts/read-only source-review evidence in monitor_20260928T192657Z; publish owned code/small evidence after frozen-source checks.

## Checklist

- [ ] Write tests and preserve expected missing-module actual exit.
- [ ] Implement no-side-effect native capture: data/model restore fields, time/udd_state/model bytes hash, existing RNG state or explicit absence. Never read a lazy native RNG property.
- [ ] Reserve1environment/4physics before factory invocation. Reject a second native step or physics call before additional physics. Save control/prestate before the original physics method and save native output/poststate before validation/completion.
- [ ] Preserve native random action and returned values exactly. Reconstruct forward/action-cost/reward under unchanged1e-7, observation and health from saved states; full-state values are not replaced with hashes.
- [ ] On error retain pending charge and all available evidence, restore patched methods, and close only the newly created adapter/native object where available. No retry/refund/resume or scientific acceptance.
- [ ] Freeze and publish after tests. Real source/path/runtime/checkpoint/stream/resource binding, engineering-only acceptance, supervisor, global precommit and independent science acceptance remain pending.

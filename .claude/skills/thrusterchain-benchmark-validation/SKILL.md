---
name: thrusterchain-benchmark-validation
description: Validate ThrusterChain or another offline RL benchmark for a specified state or action failure mode using method-neutral probes and explicit reference values.
---

# Benchmark validity for the intended question

Find the smallest independently justified way to measure the user's stated mechanism. Preserve frozen protocols and negative results; version any benchmark change separately.

1. Inspect simulator dynamics, rewards, reset laws, observation and hidden-state representation, dataset generation, splits and filtering. Define state distribution shift separately from global action cutoff and state-conditioned action support. Report the actual exposure and mark unsupported local-density claims unknown.
2. Predeclare starts and feasible candidate actions using environment semantics, not a tested method's outcomes. Give every method the same state/action bank, retain every case and record any raw-Q error opportunity. An environment can create meaningful action-value gaps but cannot guarantee that every learned model overestimates.
3. Restore full simulator state/time and specify first action, continuation policy, discount, horizon, terminal/truncation, reward timing and action scaling. A model's own-policy Q reference differs from a shared-policy reference. Label smoothing, target lag and twin-min differences; do not silently equate them to deployment-policy Q or Q-star. Quantify rollout uncertainty where transitions are stochastic.
4. Before efficacy, test a fixed physics and value-gap exposure criterion independent of BCA performance. Record candidate bank spread, raw baseline mistakes, known support labels, per-stratum counts, and outcome-independent failure or inconclusive rules. Forced probes characterize a mechanism; they are not freely selected policy performance.
5. Compare all prespecified methods and simple matched controls without selective filtering. Freeze independent held-out conditions before efficacy evaluation; report shifts and failures, not only successful cases. Avoid using privileged simulator labels to train or tune an offline learner unless the user explicitly defines that distinct task.
6. Give exact model, simulator and wall-clock budgets; inspect existing attempts before execution. This skill does not authorize training, expensive runs, access-denial bypass or external publication.

**Deliverable:** a claim-to-test table, fixed probe and support definitions, reference semantics, validity gates, exact budget, results by stratum, and limitations. A failed exposure gate is inconclusive about the method, not evidence of equivalence.


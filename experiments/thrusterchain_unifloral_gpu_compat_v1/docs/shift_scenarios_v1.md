# Dynamics shift scenarios, v1

This is an additive, versioned evaluation extension. It does not change the
historical environment, data, policies, or v5 benchmark. It uses the finite
100-decision task and the existing 14-component float32 observation. Scenario
identity and parameters are not supplied to the policy. The shifted cases are
therefore generally partially observed when the scenario is unknown to the
policy; this extension does not claim a fully observed universal MDP.

| Scenario | Effect after policy command is rounded to float32 and clipped | Transition randomness |
|---|---|---|
| `baseline` | Original action and unchanged plant | Deterministic |
| `attenuated_thruster` | Multiply the first command by 0.55 before canonical thrust | Deterministic |
| `bounded_disturbance` | Original action | Add independent uniform noise in [-0.02, 0.02] to each of ten physical coordinates after every deterministic step |
| `combined` | First command multiplier 0.55 | The same bounded process disturbance law |

The other three action coordinates, current, wave dynamics, horizon and gamma
remain unchanged. Attenuation is a **command mapping**, not an assertion that
the physical thrust is always reduced by the same fraction: the actuator's stall
curve is nonlinear. The reward is recomputed from the pre-step physical state
and the policy's original clipped command, so effort cost remains in the
policy-command coordinates in every scenario. `info` distinguishes `command`,
`actuator_command`, the actual canonical `thrust`, and post-step `disturbance`.
These diagnostic fields are not passed to the evaluated policy.

Disturbances are additive to positions and velocities in the simulator's native
discrete coordinates. They are a synthetic stress test, not a calibrated noise
model or a claim of physical realism. The wave and current coordinates are not
noised. For a 100-step task, the test suite numerically checks an independent
conservative operator-norm bound including the 0.02 coordinate disturbance;
its bound is below 250, leaving over 50 of margin to the existing norm-300
state bound. The runtime additionally rejects nonfinite or out-of-bound states.

## Evaluation

`evaluate_frozen_shifts` evaluates a frozen policy with `act(observation)` and
`state_bytes()` methods, including the existing v6 runner's frozen adapters.
It uses actual shifted environment steps and records raw and discounted returns,
both source and reset seeds, full scenario specifications, policy hashes,
environment and canonical-source identities, and NumPy/PRNG provenance. It
checks policy state around each action and uses the existing **cooperative**
inference guard; this is not a sandbox or proof that arbitrary policy code is
pure. Episodes, not gradient steps, are counted separately from prior training.
The caller must explicitly declare `policy_task_semantics="finite_horizon"`;
continuing-time-limit policies are refused because their stationary padded
observation is incompatible with the finite task's countdown. This is a caller
declaration, not an independently verified policy or checkpoint attestation.

The caller supplies complete declared `dataset`, `selection`, and `final` seed
sets. The function refuses duplicate/overlapping declared seed roles, and also
checks derived environment-reset seeds against the dataset and one another.
It reuses each evaluation seed across scenarios for paired starts and, where
applicable, the same disturbance draws. This checks declarations; it cannot
prove dataset provenance, that the plan was frozen before observing outcomes,
or that a final scenario/seed was never used in previous development. The
scenario family can be any existing start law; a new actuator or disturbance
case is distinct from the existing current, wave, or kick start-law shifts.
Episode variation for one frozen policy is not training-seed uncertainty.

```python
from thrusterchain.shift_scenarios_v1 import SCENARIOS, evaluate_frozen_shifts

results = evaluate_frozen_shifts(
    frozen_policy,
    scenarios=tuple(SCENARIOS),
    policy_task_semantics="finite_horizon",
    partitions={"dataset": dataset.arrays["episode_seeds"].tolist(),
                "selection": [101, 102], "final": [201, 202]},
    purpose="final",
)
```

The numbers in this snippet are examples, not a registered holdout. Run the
bounded regression suite from the v6 directory with
`python -B -m unittest discover -s tests -p test_shift_scenarios_v1.py -v`.
The suite covers baseline canonical reset/step parity, independently calculated
altered transitions and rewards, stochastic replay and bounds, seed separation,
policy-state mutation refusal, and a small trained-policy evaluation. It does
not constitute a research-scale performance comparison.

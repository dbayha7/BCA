# Behavior-diverse datasets, version 1

`thrusterchain.quality_datasets` collects a separately named schema,
`thrusterchain.behavior_trajectories.v1`. It preserves the original v5 schema
and files. Choices are:

| Behavior | Collection policy |
|---|---|
| `random` | Each requested command is uniform on the 4D cube `[-1, 1]^4`. |
| `feedback_weak` | Public-model feedback with gain multiplier 0.25 and Gaussian demand noise SD 0.30, passed through `0.8*tanh(demand/0.8)`. |
| `feedback_strong` | The same feedback gain with multiplier 1 and noise SD 0.10, passed through the same bounded map. |
| `mixed_random_strong` | A seeded, balanced assignment of whole episodes to random or strong feedback; no episode is selected by its realized return. Odd episode counts give the extra episode to strong feedback. |

The gain is the discounted LQR solution for the public rising-thrust relaxation
of the plant. It is privileged model knowledge used to construct a behavior
policy; the learner receives only recorded observations and actions. Feedback
uses the public float32 observation, not the hidden float64 simulator state.
These names describe controller settings. They do not certify expert quality,
optimality, or a return ordering; measure returns and uncertainty for the exact
dataset and intended task before making such claims.

Reset, action/noise and split streams are separately addressed by trajectory;
the mixture allocation has its own stream. Reusing a seed across behavior
datasets intentionally reuses their episode start seeds, so these data are
paired and should not be described as independent dataset replications. A
manifest records the stream hierarchy, component ID per episode, gain and
source hashes, physics, task semantics and package versions. A loader validates
the actual behavior draws against the saved public observations and streams,
replays every transition and reward through the recorded model, checks shapes,
actions, time coordinates, masks, split membership and digests, and refuses
unknown schema/source changes. This is an integrity check, not proof of policy
quality or a claim of cross-platform bitwise portability.

Each episode stores 101 observations and 100 actions/rewards, with actual
float32 executed commands and true post-step observations. Groups and seeded
splits are episode-local; the final transition has no recorded next action.
For a finite task it is terminated and has bootstrap zero. For a continuing
time limit it is truncated, has bootstrap one, and has unavailable next action;
ReBRAC's current adapter refuses such a dataset because its objective requires
that action on all bootstrapped rows.

No action density, executed-action mass or state/trajectory occupancy ratio is
supplied. In particular, the uniform density of one component does not define
the feedback-mixture occupancy or authorize importance weighting. The balanced
mixture's fixed allocation also does not establish exchangeability between a
calibration episode and a new independently sampled episode. The built-in
imitation calibration labels mixture output **descriptive only**; it is not a
return or distribution-shift certificate.

For a small plumbing check, from the v6 directory:

```text
python -B -m thrusterchain generate-behavior data.npz --behavior mixed_random_strong --episodes 10 --seed 822
python -B -m thrusterchain validate data.npz
python -B -m thrusterchain train data.npz frozen_bc
python -B -m thrusterchain calibrate data.npz frozen_bc imitation_calibration.json
python -B -m thrusterchain evaluate frozen_bc evaluation.json --episodes 1
```

Generation and artifacts are write-once. The original `generate` command
continues to produce its separate v3 random-box dataset. Python APIs are
`generate_behavior_dataset`, `save_behavior_dataset`, and
`load_behavior_dataset`; the optimizer runner and closed-form BC use the same
validated object. The CLI generates `smoke` provenance only. These example
counts test plumbing and do not constitute an offline RL benchmark result.

# ThrusterChain v6 — bounded-validation build

V6 is a separate copy of the validated v5 benchmark; it preserves the existing
studies and adds behavior datasets, dynamics shifts, and one explicit unified
UniFloral composition. Package version is 0.4.0. The standalone wheel contains
the environment and BC paths; the neural host adapters are used from this source
checkout with the BCA repository and compatible JAX dependencies available.

## Data and scenarios

`generate-behavior` produces full-episode datasets using a random recorder,
weak feedback, strong feedback, or an episode-level balanced random/strong
mixture. The fixed feedback controller uses public float32 observations and a
gain derived from the public model: this is privileged model access in data
collection, not a guarantee that a named policy has better return. Complete
trajectories, train/tuning/selection/calibration groups, source and gain hashes,
reset seeds, executed actions and terminal/timeout masks are preserved. The
schema does not provide a behavior-policy or occupancy density. Fixed-quota
mixture imitation calibration is descriptive, not a future-coverage guarantee.
See [dataset contract](docs/behavior_datasets_v6.md).

New finite-task scenarios are the canonical baseline, one attenuated actuator
command, bounded random physical-state disturbances, and both changes together.
The original initial-condition families remain available independently. The
new scenario identity is not part of the observation; unknown regimes can make
the problem partially observed. Attenuating the command before nonlinear stall
is not necessarily a uniform reduction in thrust. Costs use the original policy
command. The CLI binds a frozen checkpoint to its dataset and refuses continuing
tasks for these finite-time scenarios. The lower-level API requires an explicit
task declaration but cannot independently attest it. See
[shift contract](docs/shift_scenarios_v1.md).

## Unified UniFloral

`unifloral-unified` is one explicit deterministic TD3+BC-style composition of
the separately pinned unified implementation, not the standalone TD3+BC host
or an arbitrary UniFloral configuration. One primary runner update is one actor
step followed by two critic steps on the same batch; the other native adapters
use their declared optimizer as primary. The runner's PCG64 sampler replaces
the native sampler boundary, so update parity is conditional on a matched batch,
not an assertion that the entire training trajectory matches upstream.

The bounded tests use real Distrax with JAX 0.6.2 and NumPy 2.2.6, outside the
pinned upstream JAX and NumPy bounds. An explicit learning-setting opt-in is
required and recorded; it does not certify upstream support. A local alias
maps the removed `jax.tree_map` to `jax.tree_util.tree_map` without modifying the
pinned file. Only this composition and CPU paths were tested. See
[unified integration](host_integration/unifloral_README.md).

## Staged CLI and bounded use

From this directory in a compatible environment, with the BCA repository root
and this directory on `PYTHONPATH`:

```text
python -B -m thrusterchain generate-behavior data.npz --behavior feedback_strong --episodes 10 --seed 62001
python -B -m thrusterchain validate data.npz
python -B -m thrusterchain train data.npz frozen --adapter unifloral-unified --offline-updates 2 --batch-size 8 --seed 62003 --learning-settings '{"allow_outside_upstream_constraints":true}'
python -B -m thrusterchain evaluate-shifts frozen shifts.json --dataset data.npz --scenarios baseline attenuated_thruster bounded_disturbance combined --final-seeds 62004
```

These sizes are plumbing checks, not a performance experiment. Neural adapters
require an explicit update budget; commands create outputs exclusively. The
compatibility opt-in above is appropriate only when deliberately testing an
out-of-constraint runtime. Other neural choices are `td3bc`, `iql`, `cql`, and
`rebrac`; their separate implementations and settings are documented in
[host integration](host_integration/README.md). ReBRAC refuses a bootstrapped
row without an observed next action, including current continuing-data cutoffs.
The copied standalone adapters have v6 checkpoint identities and dataset labels;
v5 neural checkpoints remain usable with their original v5 source rather than
being silently reinterpreted as v6 checkpoints.

## Evidence and limitations

Run `python -B validate_v6.py` for v6 contract tests. It explicitly skips archived
v4 evidence classes whose original artifacts and loader are unavailable; skips
are not successful historical checks. See [bounded validation](validation/VALIDATION.md)
for actual commands, runtime, source identities, and results. The inherited
`package_release.py` remains disabled because its old release claims do not
describe v6. Historical audit receipts copied into this directory refer only to
their original source snapshots.

These deterministic and stochastic synthetic cases do not establish performance
on real hardware, every distribution shift, or an independently unexposed test.
Seed partitions and source hashes do not prove independence or precommitment.
No full training, tuning sweep, or scientific method comparison is implied.

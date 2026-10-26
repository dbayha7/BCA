# Regular mode: fixed data, offline selection, frozen evaluation

Regular mode uses `ThrusterChainEnv` as a live Gymnasium environment for dataset
collection and final reporting. Between those stages, behavior cloning receives a
fixed trajectory artifact. Its fitting and selection code never calls the
environment. The public physical law is not needed to fit the BC policy; it is
used by the artifact validator to check replay integrity. Privileged and fitted
model controllers are separate baselines, with their model access recorded.

## Task and data contract

The default observation is the thirteen physical coordinates followed by
`remaining_steps / 100`. The objective is the discounted finite-horizon return
with gamma 0.98 and exactly 100 decisions. Default `finite_horizon` semantics mark
the final transition `terminated=True, truncated=False`; its bootstrap is zero.
The explicit `continuing_time_limit` compatibility option preserves the original
canonical simulator's `terminated=False, truncated=True` flags. Such artifacts
are labeled separately and must not be confused with the finite-horizon target.
Neither mode changes physical dynamics or rewards.

`generate_dataset` draws recorded-condition episodes with independent seeds,
using the same physics for the whole artifact. The continuous recorder is
state-independent and samples directly from two bounded four-dimensional boxes:
the region `a0 > 0.5` has volume 4 and mixture mass epsilon; its complement has
volume 12 and mass `1 - epsilon`. The requested-action density is therefore
`epsilon / 4` in the region and `(1 - epsilon) / 12` outside it. The default
`epsilon=0.25` is uniform on the complete action cube. Alternatives `0.01` and `0`
provide a rare region and an exact support gap. All other action coordinates
are uniform on `[-1, 1]`. This recorder is deliberately competence-poor; it is not
tuned to improve the BC baseline. There is no Gaussian clipping or feedback mean.

An NPZ artifact contains a scalar JSON manifest and the following arrays. The
trajectory axis is always first, `E` is the number of episodes, and each trajectory
includes its final observation and exact simulator state.

| Array | Shape | Type |
|---|---|---|
| observations | E × 101 × 14 | float32 |
| states | E × 101 × 13 | float64 |
| actions | E × 100 × 4 | float32 |
| requested_actions | E × 100 × 4 | float64 |
| rewards | E × 100 | float64 |
| terminated, truncated | E × 100 | bool |
| trajectory_ids, group_ids, episode_seeds | E | int64 |
| split_codes | E | uint8 |
| behavior_component, action_region | E × 100 | uint8 |
| behavior_log_density | E × 100 | float64 |

The schema is `thrusterchain.continuous_trajectories.v2`. The manifest records the
two regions, epsilon, sampler version, explicit PCG64 generator, NumPy version
and full named SeedSequence hierarchy. Reset seeds use spawn keys
`(1701, trajectory_id)`, the split permutation uses `(1702,)`, and each episode
gets a fresh action generator at `(1703, trajectory_id)`, all under the recorded
root seed. The manifest enumerates every reset seed and episode action address.
Adding episodes or consuming more split-stream draws leaves existing reset and
action sequences unchanged; split memberships may change when its size changes.
Region
and component IDs refer to requested commands, since float32 rounding may move
commands near `a0=0.5` onto the boundary. `actions` are the executed float32 values;
replay always uses these, never unquantized requests.

Loading refuses nonfinite values, wrong dtypes/shapes, inconsistent requested-action
densities, wrong time/terminal fields, changed hashes, broken replay, or duplicate
groups or episode seeds anywhere in the artifact. This schema requires one
independent seeded episode per group, including within calibration. NPZ loading disables pickled objects.
Dataset arrays and policy parameters are backed by immutable bytes. Saving uses
exclusive creation and refuses to overwrite an existing artifact. SHA-256 covers
canonical JSON metadata, array names, shapes, dtypes and exact array bytes; ZIP
timestamps are deliberately excluded. This identifies content reproducibly, but
does not provide an authenticated signature or make files physically undeletable.
The manifest also binds the canonical source provenance and its SHA-256, adapter
source hash, canonical-loader source hash and full environment configuration.
Loading compares those fields against the current hash-verified sources and
configuration, and checks every reset seed against the named hierarchy. Canonical
source and sampler-revision mismatches are refused. The recorded NumPy version is
retained as provenance and may differ from the loading runtime; reset and action
lineage is checked with documented numerical tolerances. See [NumPy compatibility
and collector lineage](numpy_compatibility.md). Earlier dataset-schema v1 artifacts
are not accepted as schema v2. This is separate from the benchmark folder's revision
number: the preserved v1 benchmark's schema-v2 data remain supported.

## Density statements

The logged density is the exact analytic *requested-action density* of the
continuous box-mixture law with respect to Lebesgue measure. Float64 pseudorandom
sampling numerically approximates that continuous law. Execution then quantizes
the request to float32; the executed law is atomic and is not an ordinary
continuous density. Its quantization mass function is not evaluated here. The
transition is deterministic, so its conditional law is a Dirac measure. The
closed-loop state/time occupancy and joint state-action density are **unknown**.
No joint-density estimate, importance-weighted return estimate, or coverage
certificate is manufactured from the logged requested-action density.

Known requested-action support does not establish state occupancy coverage for a
new policy. Under epsilon zero, the rare region has no requested-action support;
`behavior_log_density` correctly returns negative infinity there, while all
actually sampled rows have finite log density. Offline action MSE is imitation
evidence, not an estimate of the BC policy's return. Only subsequent live reporting
rollouts measure that return here.

## Four independent uses of data

Groups are complete episodes. A seeded permutation assigns approximately 60/20/10/10
percent to the following splits, with at least ten episodes required overall.
No individual transition is moved independently of its trajectory.

1. `train`: fit observation mean/standard deviation and every affine ridge BC
   candidate. Constant coordinates get scale one; rewards are not normalized.
2. `tuning_validation`: choose the ridge coefficient by episode-averaged action
   MSE. The ordered candidate grid is declared before fitting and resolves ties.
3. `selection`: report the already chosen policy's imitation error once, without
   refitting. This BC baseline has one selected model; benchmark-level selection
   among controller families needs its own declared rule and receipt.
4. `final_calibration`: reserved for a fresh post-freeze residual calculation.
   Fitting and selection only hash its contents to bind artifact identity.

`train_behavior_cloning` returns a `FrozenLinearPolicy` and `SelectionReceipt`.
The receipt binds all split hashes, dataset hash, normalization and coefficient
hash, fixed preprocessing/action transform and source implementation hash,
candidate grid, selected ridge, offline scores, task semantics, physics,
episode seeds and interaction budgets. `freeze_policy` writes one complete NPZ
transaction; `load_frozen_policy` verifies both receipt and policy hashes.
The receipt is tamper evidence, not proof that a researcher never viewed a split
or used earlier simulator results. Prior development feedback remains unverified.

`calibrate_frozen_policy` optionally uses the fresh split to calculate a split-
conformal quantile of each episode's maximum absolute imitation residual. Its
exchangeability unit is a trajectory, not a correlated transition. If the required
order statistic exceeds the number of calibration episodes, the radius is `null`
and the result explicitly vacuous. It makes no return bound or shifted-family
coverage claim, and changes no policy or normalization parameter. The alpha and
calibration procedure should be fixed before accessing that split.

## Final reporting and simulator-assisted work

`evaluate_frozen_policy` accepts the immutable prediction-only BC policy and its
matching receipt. It runs newly seeded episodes in recorded conditions and three
families excluded from collection: strong current, big waves and big kicks. It
checks the policy hash before and after reporting and uses a source-hashed pure
inference function. It refuses replaced prediction implementations and arbitrary
policy subclasses. Parameters and preprocessing arrays are immutable; this affine
policy has no mutable running buffers. An evaluation context instruments and
refuses built-in collection, training, calibration and persistence APIs. It counts
actual inference calls and forbidden-API attempts; tests install fail-on-call
sentinels for training, updates and file-writing APIs. This enforces the built-in
interface, not an operating-system sandbox against deliberately modified code.
The integrated run uses one process; staged CLI invocations share filesystem
permissions. Dataset objects contain all offline roles, although the supported
fitting functions optimize only on permitted arrays. Final returns are generated
after the freeze receipt. Reads, saved I/O aliases, native calls and arbitrary
transient adaptation are not comprehensively intercepted. Zero counters mean
zero observed intercepted calls, not universal no-I/O isolation. See the tested
[inference boundaries](inference_boundaries.md). No OS permissions were changed.
It performs zero learning or normalization updates and records every return, discounted return, start seed,
stall fraction and total interaction count. Start seeds are shared across families
for paired comparisons and exclude dataset seeds. Their outcomes are therefore
not independent across families. A single policy's rollouts do not measure
variation across independently fitted training seeds.

Any use of these outcomes to choose configurations is simulator-assisted selection.
Call with `purpose="simulator_assisted_tuning"` to label this explicitly; that
report is ineligible as strict-offline final reporting. A new held-out reporting
bank is then needed. This original closed-form API does not implement online
fine-tuning. Revision v4 adds a separate [adapter and adaptation runner](training_modes.md),
with explicit budgets and fresh checkpoint restoration per scenario. It does not
silently treat tuning feedback as free offline evidence. The local library cannot
detect external tuning; the human experiment record must preserve that distinction.

`provenance="smoke"` is the default everywhere and remains attached to the dataset,
receipt and result. Such artifacts cannot be promoted by relabeling evaluation as
research. Explicit `"research"` enables a research run label, not a statistical
validity guarantee. Sample counts, independent fitted seeds, candidate budgets,
prior development feedback, full raw outcomes and uncertainty methodology must
still justify a research claim. Do not put smoke output into a research result table.

## Minimal Python workflow

```python
from thrusterchain.datasets import generate_dataset, load_dataset
from thrusterchain.regular import (
    train_behavior_cloning, freeze_policy, load_frozen_policy,
    calibrate_frozen_policy, evaluate_frozen_policy,
)

generate_dataset("outputs/smoke/trajectories.npz", episodes=10, seed=123,
                 provenance="smoke")
data = load_dataset("outputs/smoke/trajectories.npz")
policy, receipt = train_behavior_cloning(data)
freeze_policy(policy, receipt, "outputs/smoke/frozen_bc")
policy, receipt = load_frozen_policy("outputs/smoke/frozen_bc")
calibration = calibrate_frozen_policy(data, policy, receipt)
result = evaluate_frozen_policy(policy, receipt, episodes_per_family=2, seed=456)
```

Run the focused checks with `python -B -m unittest discover -s tests -p test_regular.py -v`.
They cover trajectory leakage, replay/density corruption, mixture normalization,
the support gap and quantization boundary, train-only statistics,
independent random-stream prefixes and canonical source identity,
offline fitting without simulator access, immutability, artifact tampering, small-
sample calibration refusal, inference transforms, forbidden calls, reproducible
held-out reporting, and explicit tuning labels.

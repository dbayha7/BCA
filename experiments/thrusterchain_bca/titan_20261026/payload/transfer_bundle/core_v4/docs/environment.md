# ThrusterChain environment contract

The benchmark uses the existing deterministic ThrusterChain plant, with a finite task of 100 steps and discount 0.98. The adapter adds strict inputs, observed remaining time and explicit task-end flags. It does not alter transition or reward arithmetic. The machine-readable contract is [environment_contract.json](../contracts/environment_contract.json), and ten physical transition fixtures are in [dynamics_fixtures.json](../contracts/dynamics_fixtures.json).

## Physical state, actions and units

The exact state is `x=(p1,v1,...,p5,v5,c,w1,w2)`, 13 float64 values. Position and speed are labels for the two coordinates of each discrete oscillator. The canonical source assigns no metres, seconds or SI calibration: all coordinates and throttle commands are dimensionless model units, and one transition is one model step. The spring/current narrative describes the fixed linear plant; the code is not a calibrated fluid or mechanical simulator.

Four commands control adjacent pairs of velocity coordinates. The Gymnasium action space is `Box(-1,1,(4,),float32)`. `step` accepts an exact `(4,)` finite real numeric array, converts it to float32 and back to float64, then clips it into `[-1,1]`. This matches the upstream deployment boundary. Malformed shapes, strings, complex numbers, booleans, nonfinite values and values overflowing float32 are rejected. Low-level `dynamics` retains the original float64 interface; explicitly quantize commands first when replaying environment steps.

Let `s=x[:10]`, `c=x[10]`, and `w=x[11:13]`. The unchanged equations are

```text
s_next = A10 s + 0.3 B1 thrust(a) + 0.3 v (c + w1)
c_next = c
w_next = Rot(0.27) w
r = -(sum(s**2) + 0.3 sum(a**2))
```

`v=(0,1,0,1,0,1,0,1,0,1)`. Column `k` of `B1` is one at rows `2k+1` and `2(k+1)+1`, zero elsewhere. `A10` is the canonical Stage0 oscillator plant: radii 0.98 through 0.96, angles 0.10, 0.20, 0.35, 0.55 and 0.80, plus the seed-20261011 coupling of spectral norm 0.02, accepted attempt zero. The full matrices are saved in the fixtures.

With `k=float32(0.8)=0.800000011920929`, stalling thrust is `clip(a,-k,k)-slope*sign(a)*max(abs(a)-k,0)`. Default slope is 1.5; the upstream documented retune 3.0 is also accepted. No-stall thrust equals the clipped command. Exactly float32 0.8 is at the knee and does not stall. Energy uses the command magnitude even when thrust falls. Reward uses the state before the transition. There is no transition or reward noise.

The original fixed order of column sums is executed unchanged. Batch and scalar NumPy dynamics therefore agree bit for bit for the same float64 inputs. JAX functions remain optional and require x64 for float64 comparisons; compiled backends can differ by floating-point rounding.

## Controllability and exogenous forcing

For the canonical matrices in `x_next=A*x+B*thrust(a)`, the augmented physical
state is **not fully controllable**. The controllability matrix
`[B, A B, ..., A**12 B]` has rank **10 of 13**. The actuated oscillator subsystem
`(A[:10,:10], B[:10,:])` has rank **10 of 10**, using powers zero through nine.
These are linear controllability statements in effective-thrust coordinates;
they do not imply global reachability under bounded, stalled, float32 commands.

The last three rows of `B` and `A[10:,:10]` are exactly zero. Current obeys
`c_next=c`, and waves obey the autonomous rotation `w_next=Rot(0.27)*w`.
Actuation cannot stop, set, or damp these exogenous coordinates: current remains
constant, and wave amplitude remains constant in exact arithmetic. A controller
can counter their effect on the ten oscillator coordinates within its actuator
limits. Adding observed remaining time does not add an actuated state.

Independent SVD checks use NumPy's default tolerance
`sigma_max * max(matrix.shape) * eps(float64)` and absolute tolerances
`1e-14, 1e-12, 1e-10, 1e-8, 1e-6`; all give the ranks above. The smallest
nonzero singular values are approximately `0.16547317` (augmented) and
`0.09990576` (actuated), while the augmented matrix has three exactly zero
singular values. The full numerical receipt and unchanged source/fixture hashes
are in [revision_v2_claims](../audits/revision_v2_claims/claims_receipt.json).
This correction changes the stated claim only; the plant and reward are unchanged.

## Observation and episode semantics

Default observation is float32 `[x[13], (100-t)/100]`, shape `(14,)`. Its first ten bounds are `[-300,300]`, current is `[0,1.5]`, each wave coordinate is `[-1,1]`, and remaining fraction is `[0,1]`. Internal carried state remains float64 and is returned by copy in `info['state']`. Observations are rounded: time augmentation resolves hidden horizon, but does not eliminate float32 state aliasing. Stall/no-stall is fixed per configured environment; these are distinct tasks, with no world bit in the observation.

The default actor and learner interface is the public observation. Exact `info['state']`, water push, thrust and stall flags are diagnostic and replay channels. They must not be used as additional actor inputs or training features in the observation-only benchmark. A method using exact logged states needs a separately declared information tier; the identified-model baseline fits the public float32 observation coordinates instead. Live controllers receive `act(obs)`, while saved diagnostic states may support plotting or independent replay.

The default objective is `G_t=sum_{k=0}^{99-t} 0.98**k r_(t+k)`, with zero terminal reward and continuation. The hundredth transition returns `terminated=True`, `truncated=False`, `bootstrap_mask=0`, and `continuation_discount=0`. This follows Gymnasium's rule that an inherent finite horizon is termination and must be represented by time in the observation. [Gymnasium time-limit guidance](https://gymnasium.farama.org/tutorials/gymnasium_basics/handling_time_limits/).

For explicit upstream replay, `task_semantics='continuing_time_limit'` returns the original `terminated=False`, `truncated=True` at step 100; its bootstrap mask stays one. Only this mode permits `observation_mode='physical'`, the upstream 13-vector. The finite-task and continuing-task objectives must not share labels. Both modes expose `canonical_terminated` and `canonical_truncated` in info, and both reject stepping after the collection episode ends until reset. Step 100 physically advances the state and yields the last reward before ending.

## Reset validity and randomness

`reset(seed=None, options=None)` returns `(observation, info)`. An explicit state must have exact shape `(13,)`, finite values, `||s||2<=100`, current in `[0,1.5]`, and wave amplitude `<=1+1e-9`. The tiny wave allowance is inherited from the canonical source for floating-point rotation drift. Observation bounds are checked separately after reset and each step.

Options accept either `{'state': x}` or `{'start': overrides}`, never both. Start overrides are finite current and wave-amplitude ranges plus nonnegative Gaussian standard deviation `kick`; their domains are checked. Without options the named start law applies:

| Law | Current | Wave amplitude | Kick standard deviation |
| --- | --- | --- | --- |
| recorded | 0.1–0.7 | 0.1–0.4 | 0.5 |
| strong_current | 0.7–1.1 | 0.1–0.4 | 0.5 |
| big_waves | 0.1–0.7 | 0.4–0.7 | 0.5 |
| big_kicks | 0.1–0.7 | 0.1–0.4 | 1.5 |

The per-instance Gymnasium random generator draws ten Gaussian float coordinates, current, wave amplitude, and uniform phase, in that order. The phase forms `(amp*cos(phase),amp*sin(phase))`. `reset(seed=...)` starts that sequence; `reset(seed=None)` continues it. Explicit-state reset consumes no draw. `action_space.seed(...)` is independent. Gaussian starts have mathematically unbounded tails: a draw outside the start set raises, and is neither clipped nor resampled. This is upstream behaviour; ordinary default draws are far inside the boundary.

## Source identity and portability

`thrusterchain/canonical/thrusterchain.py` and `gapped_lq.py` are byte-identical source archives. SHA-256 values are respectively `f22f2e214ad9231bee4ce60ef55e64409534c2654c4e184406553ba8ed67dde1` and `f8ec7cc6c7d9d013d3e509cd3aba0a6dedf1f371480b4e1977f51e2b5dd923aa`.

The original source was first imported normally with its repository dependency gate intact to export fixtures and `plant.json`. The canonical matrix SHA-256 is `289afb581b4fdbc97fd68566c0b42ff6244b20686e9e49ba177e7e6bd57454a1`, hashing full A followed by B in contiguous little-endian float64. The adapter authenticates the archived files and artifact before loading, then checks that full matrix hash. It requires no original repository path and never imports `experiments.shift`.

The dependency bridge executes an explicit allowlist of original AST definitions. It supplies normal NumPy/Gymnasium dependencies and the frozen Stage0 `A10` export, runs the original `base_input`, and suppresses upstream global Gym registration. Freezing the export avoids platform-dependent SVD/eigenvalue differences when reconstructing the seeded coupling. Dynamics, thrust, reward, reset draws and the original environment class are unchanged source definitions. This is a documented packaging adaptation; the archived repository dependency gate itself is not executed by the portable package.

Beyond packaging, wrapper changes are exact-shape/type/finiteness validation, finite supported slope settings, time observation, and the explicit terminal-semantics choice. The adapter supports ANSI rendering, and returns defensive state copies. Importing it does not install packages, edit old studies or register the legacy `BCAShift/*` IDs.

## Reference values and historical compatibility

The environment exposes exact deterministic transitions and immediate rewards. A specified policy can be rolled out under the same float32 action boundary, world, discount and remaining horizon to obtain its finite-policy return. This capability does not establish global optimality of a controller. Capped linear feedback is a baseline unless a separate reference solver provides a certificate.

Earlier continuous diagnostics used 15 coordinates `[physical13,regime,time]` and default discount 1.0. Their actor observations, fitted critics and returns cannot be fed into this 14-coordinate, discount-0.98 task without an explicit audited adapter. The common physical matrix identity does not make those task contracts interchangeable.

## Use and checks

```python
from thrusterchain import ThrusterChainEnv

env = ThrusterChainEnv()
obs, info = env.reset(seed=123)
obs, reward, terminated, truncated, info = env.step([0., 0., 0., 0.])
```

From the bundle root with NumPy and Gymnasium available, run `python -B -m unittest discover -s tests -p test_environment.py -v`. Tests check authenticated source/matrix identity, exact fixtures against the normally imported original, scalar/batch identity, Gymnasium's environment checker, seeded RNG behaviour, strict input validation, time and flags, observation bounds and current/wave invariants through boundary rollouts. Optional JAX parity is checked when JAX is already installed.

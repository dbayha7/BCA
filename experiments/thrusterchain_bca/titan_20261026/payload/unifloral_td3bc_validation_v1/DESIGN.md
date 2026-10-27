# Isolated UniFloral TD3+BC microstep candidate

This folder prepares a bounded comparison against **UniFloral's standalone
`algorithms/td3_bc.py`**, not its distinct unified `algorithms/unifloral.py`
variant and not CORL. The upstream commit is
`f2dc1278eae18ed3e22c92255119c54885414d17`; the standalone source SHA256 is
`6f5ab6f98d706d902a18f11e1c81e99e2691dc4c8a53843bf6521c9f95d00eee`.
The companion loader verifies pinned sources and supplies their unchanged
computational definitions. This candidate is not yet a production adapter.

## Native behavior retained

The candidate receives native initialized actor, critic, target and optimizer
TrainStates. Network architecture, initialization, observation normalization
`(obs - mean)/(std + 0.001)`, Adam epsilon `1e-5`, learning rate and all other
native arguments remain the caller's unchanged native choices. No optimizer
or network is reconstructed here.

For each native outer step:

1. Split the native RNG and sample one batch with replacement.
2. Update both critic heads once, using native target noise and target networks.
3. Update the critics a second time using **the same batch**, with the next
   native target-noise draws.
4. Update the actor once from the newly updated first critic head.
5. Update actor and critic targets once by native Polyak interpolation.

Critic loss averages over batch and heads. BC averages action coordinates and
then batch rows. The actor's deterministic distribution is sampled with
`seed=None`; it consumes no RNG. Targets preserve the native done mask, min of
critic heads, noise scaling/clipping and action clipping. The original native
`num_updates` counts outer steps; one outer step performs **two** critic
optimizer updates. This candidate explicitly counts each critic update.

## Sole proposed objective change

For the Q-value used by the actor, let

`c = stop_gradient(td3_alpha / (mean(abs(Q1)) + 1e-7))`.

The native actor loss is `-c * mean(Q1) + mean(BC)`.
The candidate's positive-penalty path adds `c * kappa_penalty * mean(U)`.
The coefficient `c` is always calculated from **unpenalized Q1**. Diagnostics
call it `q_scale_coefficient`, to distinguish it from `kappa_penalty`.
The BC coefficient, critic loss, TD targets, update order and target cadence
are unchanged. No new normalization or small-Q rejection is introduced.

For exactly zero kappa, a Python-static branch retains the native loss
expression and never calls the uncertainty function. For positive kappa,
`width_fn(reference, observations, live_actions)` returns one width per row
with the native Q dtype. Every reference leaf is passed through
`stop_gradient`; the action input remains differentiable. The function must
be deterministic and side-effect free. The prototype receives an already
constructed frozen reference and does not refresh or fit uncertainty.
The validation harness must verify gradient identities and numerical
finiteness; this interface alone does not establish either.

## Phase-aware API and exact continuation data

`make_microsteps(args, actor_apply_fn, q_apply_fn, dataset, width_fn=None,
kappa_penalty=0.0)` returns `initial_state`, `step`, and `actor_loss` callables.
`step(state, unused=None)` is a pure function suitable for `lax.scan` and
returns `(next_state, metrics)`. It calls the critic optimizer exactly once.

| Entry phase | Batch operation | Optimizer and target operations | Exit phase |
| --- | --- | --- | --- |
| 0 | Sample one native batch | First critic only | 1 |
| 1 | Reuse held batch | Second critic, actor, both targets | 0 |

`MicrostepState` retains the native RNG, every native TrainState, held batch,
phase, first-critic loss, frozen reference and explicit counters for critic,
actor, actor-target, critic-target and completed outer steps. The initializer
creates concrete zero-shaped batch storage without drawing RNG and is intended
for a fresh native state. Wrapping a nonfresh state requires the caller to
establish an outer-step boundary. Resuming an arbitrary microstep restores
the entire serialized state, not the initializer or weights alone.

Phase-0 actor diagnostics are zeros and `outer_completed` is false; these
zeros are not an observed actor loss. On completing phase 1, `q_loss` is the
mean of both critic losses and native actor diagnostics are available.
`critic_loss_this_call` is always the actual current critic loss. Returned
state must be accepted by the caller before its counters describe committed
progress. A failed or interrupted call does not itself persist a checkpoint.

The harness must bind checkpoints to source/config/dataset identities and
restore with matching pytree templates. This module performs no filesystem
writes and defines no persistent checkpoint format. Flax serialization of
the complete namedtuple is used by the proposed bounded parity harness.

## Production integration remains a separate decision

The current ThrusterChain core owns batch sampling and requires one primary
critic update per adapter call. Preserving UniFloral's two-critic reuse of
the *same* batch conflicts with accepting a newly sampled core batch on every
call. This candidate therefore owns the native sampler and cannot be dropped
into that core as a renamed CORL adapter. Ignoring alternate core batches or
counting an outer step as a single critic update would conceal a mismatch.

Two future options require a separate explicit design:

- A dedicated native runner keeps outer steps intact and converts budgets to
  actual critic updates, with explicit checkpoint and evaluation integration.
- A versioned microstep runner API makes phase, held-batch ownership, sampler
  state and progress receipts explicit across the core/adapter boundary.

The microstep form is chosen here only to test exact native parity and odd
critic-count continuation before choosing a production integration. The
existing core, existing adapters, source snapshot and paused pilot are not
changed by this candidate.

## Bounded validation and limits

The companion harness is planned to compare six native outer steps (12
critics) against 12 candidate critics at zero kappa, exercise four positive
kappa critic calls, and replay three critic calls from an odd-phase saved
state: **31 total critic optimizer calls**, on predetermined synthetic data.
Checks must include native state/optimizer/target/RNG parity, counters,
same-batch reuse, exact full-state continuation, frozen-reference gradients,
live-action gradients and the added actor-term identity.

Static review cannot establish runtime equality; JIT boundaries may expose
numerical differences that must be reported rather than hidden with looser
tolerances. A successful synthetic check would not establish BCA efficacy,
ThrusterChain compatibility, collector validity, evaluation correctness or
performance under shifted data. It would also not make an environment outside
the upstream JAX/NumPy constraints supported. No new full training, benchmark,
dataset generation or evaluation is performed while preparing these files.

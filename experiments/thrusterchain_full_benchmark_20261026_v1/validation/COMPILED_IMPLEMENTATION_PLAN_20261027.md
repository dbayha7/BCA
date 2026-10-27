# Separately specified compiled implementation: bounded validation plan

This plan is additive. The original eager-versus-compiled float32 state gate
(`atol=1e-6, rtol=1e-5`) and its recorded TD3+BC and CQL failures remain
unchanged. Passing the checks below does not make that gate pass, establish
identical optimization trajectories, or prove equal benchmark outcomes. A
compiled implementation passing these checks must be identified as such, with
the original failure and limits reported. This plan specifies checks before
their execution and does not change the study's existing seeds or budgets.

## Exact contracts

For all five methods retain the existing fixed terminal-batch, per-leaf shape and
dtype, integer and random-key, sampler, update-cadence, predecessor receipt,
and byte-identical compiled continuation tests. Retain the original strict
eager-versus-compiled comparison and report its outcome without changing its
threshold or selecting passing fixtures. Also test fresh independent datasets
and seeds, cache reuse, resumed versus uninterrupted chunks, finite state and
metrics, and a full batch-size-256 bounded end-to-end GPU study. Re-run these
checks for each candidate engine and frozen runtime configuration.

## Independent TD3+BC scalar and gradient fixture

The fixture `td3bc_independent_oracle_20261027.py` supplies an analytically
tractable policy `tanh(obs @ W + b)` and two critics linear in observation and
action. It passes them to the real pinned `algorithms.td3_bc.td3_bc_update`
in both native and JIT forms. The oracle independently computes in NumPy
float64: clipped smoothed target actions, each target critic, their minimum,
terminal/nonterminal Bellman labels, the two-critic mean-square loss and its
gradient, post-Adam critic parameters, the actor's detached Q-normalization,
behavior-cloning term, actor gradients, post-Adam actor parameters, and target
Polyak updates. The RNG noise tensor itself is obtained once from the same
explicit JAX key, then passed as immutable data to the NumPy oracle; this is a
matched-input objective check, not an independent test of JAX's PRNG.

Use three fixed rows, two state dimensions, two action dimensions and two
critic heads. Use terminal mask `[0,1,0]`, `discount=0.9`, `alpha=1.7`,
`tau=0.2`, policy noise `0.17`, noise clip `0.22`, action bound `1`, and
`policy_freq=2`. The fixture source contains the fixed numeric matrices. Test
both `it=1` (critic only, targets untouched) and `it=2` (actor after critic,
then both targets); each case starts from the same initial parameters and zero
Adam moments. Use Adam learning rates 0.002 for critic and 0.001 for actor,
with beta1=0.9, beta2=0.999, eps=1e-8 and eps_root=0. Compare metrics,
parameters and independently recovered optimizer gradients to the NumPy
oracle with `atol=1e-6, rtol=1e-5`. Check counters and unchanged state exactly.
The fixture is designed away from zero Q, clipping boundaries, and zero
gradients; assert those conditions rather than silently selecting another case.

Additionally, compare analytic actor and critic directional gradients against
NumPy float64 central differences at steps `0.001` and `0.002` in fixed
directions with `atol=1e-6, rtol=1e-5`. For the actor finite difference, freeze
the detached lambda coefficient and post-critic parameters at their base
values; recomputing lambda would test a different gradient. Report the full
component and derivative errors, masks, dtypes, runtime and source hashes.

## Independent CQL fixture

Before running any CQL oracle, separately record its fixed fixture values and
thresholds. Check both critics' detached terminal-aware TD target, policy loss,
entropy coefficient, proposal densities, candidate-axis logsumexp, data-Q
subtraction and fixed conservative weight against independent numerical
calculations, plus gradient ownership and fixed-noise directional derivatives.
The existing configured branch is authoritative: inactive BC, optional dual,
or backup variants must not be silently substituted for it. Existing parity
failures and the source/counter/PRNG contracts remain visible.

The references test selected algebra and numerical behavior, not arbitrary
states, distributions, optimizer history, or full training trajectories.

# Every BCA entry point in the host

This is the code map for the **current four CORL-derived host/BCA pairs**.
All Python blocks below are literal source excerpts. `python check.py` checks
them against their source lines, so a code change cannot silently leave the
map's snippets stale. The [complete algorithms](ALGORITHMS.md) give the equations
and full update order; this map shows where those equations enter the code.

There are **four direct loss interventions**, plus the preparation, calibration,
reference-state and validity boundaries needed to supply them. The table is
about direct equations; it does not claim that other learned parameters remain
identical after training.

| Host | Direct BCA input | Exact loss location | Downstream effect |
|---|---|---|---|
| IQL | Detached, post-cap actor regression weights and support/validity masks | Guarded `actor_update` in `calibration/iql_state.py` | Changes the BCA actor. Both paired actors share one Q/V system; their actor changes do not feed the offline IQL Q/V update. |
| TD3+BC | `bc_multiplier` | BC part of `td3_bc_update`'s actor loss | Changed actor/target actor can change subsequent Bellman targets. |
| ReBRAC | `actor_bc_multiplier` | BC part of `rebrac_update`'s actor loss | Changed target actions can change subsequent critic targets; the critic BC coefficient itself stays fixed. |
| CQL | `conservative_multiplier` | Conservative gap in both critic losses | Changed critics can later affect the SAC actor, targets and dual trajectory. |

## Reading order

1. [TD3+BC: actor BC](#td3bc-actor-bc)
2. [ReBRAC: actor BC](#rebrac-actor-bc)
3. [CQL: conservative critic gap](#cql-conservative-critic-gap)
4. [IQL: actor regression weights](#iql-actor-regression-weights)
5. [Preparation and added state](#preparation-and-added-state)
6. [How the host feeds scale fitting](#how-the-host-feeds-scale-fitting)
7. [Refresh, frozen width and gradients](#refresh-frozen-width-and-gradients)
8. [Acceptance, scheduling and evaluation](#acceptance-scheduling-and-evaluation)
9. [Unifloral and the scope of this map](#unifloral-and-the-scope-of-this-map)

## TD3+BC: actor BC

**Call boundary.** The extension fits the live scale, reads the *previous frozen*
reference from `state`, then passes one multiplier per recorded `(s,a)` row into
the host. It does not use the newly fitted scale immediately. `None` selects the
plain host branch. `consumed_bc_multiplier` currently returns the dose unchanged.

[algorithms/td3_bc_bca.py, lines 464–480](algorithms/td3_bc_bca.py#L464-L480)

<!-- source: algorithms/td3_bc_bca.py:464:480 -->
```python
def update(args, config, models, state, batch, it, rng, max_action=1.0):
    fitted, diag, multiplier = (state, {}, None)
    if config.arm == "bca":
        fitted, diag = fit_scale(args, config, models, state, batch, rng, max_action)
        dose = bc_readout(models, state, batch, config.blend)
        multiplier = consumed_bc_multiplier(config, dose.dose, batch, it, rng)
    native, metrics = BASE.td3_bc_update(
        args,
        models[0].apply,
        models[1].apply,
        state.native,
        batch,
        it,
        rng,
        max_action,
        bc_multiplier=multiplier,
    )
```

**Loss boundary.** BCA multiplies each row's mean action-coordinate BC error.
The Q term and its normalization coefficient keep their host equations.
The multiplier is detached even at this final host boundary.

[algorithms/td3_bc.py, lines 128–139](algorithms/td3_bc.py#L128-L139)

<!-- source: algorithms/td3_bc.py:128:139 -->
```python
def _actor_loss_fn(params, critic_params):
    pi = actor_apply_fn(params, batch.obs)
    q = critic_apply_fn(critic_params, batch.obs, pi)[..., 0]
    lmbda = args.alpha / jax.lax.stop_gradient(jnp.abs(q).mean())
    if bc_multiplier is None:
        bc = jnp.square(pi - batch.action).mean()
    else:
        bc = (
            jax.lax.stop_gradient(bc_multiplier)
            * jnp.square(pi - batch.action).mean(axis=-1)
        ).mean()
    return (-lmbda * q.mean() + bc, (q.mean(), lmbda, bc))
```

The host's critic update and delayed actor/target schedule are reused. The
multiplier does not multiply Bellman errors, rewards, Q values or target actions
directly. Later target actions can differ because the actor has learned differently.

| Component | Exact function entry points |
|---|---|
| Direct hook | [`td3_bc_update`](algorithms/td3_bc.py#L91) |
| Width readout and connection | [`bc_readout`](algorithms/td3_bc_bca.py#L447), [`consumed_bc_multiplier`](algorithms/td3_bc_bca.py#L460), [`update`](algorithms/td3_bc_bca.py#L464), [`make_train_step`](algorithms/td3_bc_bca.py#L496) |

## ReBRAC: actor BC

**Call boundary.** The multiplier is passed only as `actor_bc_multiplier`.
The host returns its own advanced RNG; the extension carries it forward after
validating the combined state.

[algorithms/rebrac_bca.py, lines 561–577](algorithms/rebrac_bca.py#L561-L577)

<!-- source: algorithms/rebrac_bca.py:561:577 -->
```python
def update(args, config, models, state, batch, it, rng):
    valid = transition_valid(batch)
    fitted, diag, multiplier = (state, {}, None)
    if config.is_posterior:
        fitted, diag = fit_scale(args, config, models, state, batch, rng)
        dose = bc_readout(models, state, batch, config.blend)
        multiplier = consumed_bc_multiplier(config, dose.dose, batch, it, rng)
    native, returned_rng, metrics = BASE.rebrac_update(
        args,
        models[0].apply,
        models[1].apply,
        state.native,
        batch,
        it,
        rng,
        actor_bc_multiplier=multiplier,
    )
```

**Loss boundary.** ReBRAC sums the action-coordinate BC errors, unlike TD3+BC's
coordinate mean. BCA multiplies the existing `actor_bc_coef`; it does not replace
that coefficient or Q normalization.

[algorithms/rebrac.py, lines 220–237](algorithms/rebrac.py#L220-L237)

<!-- source: algorithms/rebrac.py:220:237 -->
```python
def actor_loss_fn(params):
    actions = actor_apply_fn(params, batch.obs)
    bc_penalty = ((actions - batch.action) ** 2).sum(-1)
    q_values = critic_apply_fn(
        agent_state.critic.params, batch.obs, actions
    ).min(0)
    lmbda = 1
    if args.normalize_q:
        lmbda = jax.lax.stop_gradient(1 / jnp.abs(q_values).mean())
    if actor_bc_multiplier is None:
        loss = (args.actor_bc_coef * bc_penalty - lmbda * q_values).mean()
    else:
        loss = (
            args.actor_bc_coef
            * jax.lax.stop_gradient(actor_bc_multiplier)
            * bc_penalty
            - lmbda * q_values
        ).mean()
```

**Critic boundary.** This is the unchanged target-side BC term. There is no
BCA multiplier here. Its numerical value may change later through target actions.

[algorithms/rebrac.py, lines 193–199](algorithms/rebrac.py#L193-L199)

<!-- source: algorithms/rebrac.py:193:199 -->
```python
next_actions = jnp.clip(next_actions + noise, -1, 1)
bc_penalty = ((next_actions - batch.next_action) ** 2).sum(-1)
next_q = critic_apply_fn(
    agent_state.critic.target_params, batch.next_obs, next_actions
).min(0)
next_q = next_q - args.critic_bc_coef * bc_penalty
target_q = batch.reward + (1 - batch.done) * args.gamma * next_q
```

| Component | Exact function entry points |
|---|---|
| Direct hook | [`rebrac_update`](algorithms/rebrac.py#L163) |
| Width readout and connection | [`bc_readout`](algorithms/rebrac_bca.py#L544), [`consumed_bc_multiplier`](algorithms/rebrac_bca.py#L557), [`update`](algorithms/rebrac_bca.py#L561), [`make_train_step`](algorithms/rebrac_bca.py#L602) |

## CQL: conservative critic gap

**Call boundary.** Frozen width is queried on recorded dataset actions.
`models[:3]` are the actor and twin critics; `models[3]` is the added scale network.

[algorithms/cql_bca.py, lines 352–367](algorithms/cql_bca.py#L352-L367)

<!-- source: algorithms/cql_bca.py:352:367 -->
```python
fitted, fit_diag = fit_scale(args, config, models, state, batch, rng, max_action)
frozen_predictions = models[3].apply(
    state.posterior.cal_params, batch.obs, batch.action
)
dose = frozen_level_dose(state.posterior, frozen_predictions, "full", config.blend)
native, metrics = BASE.cql_update(
    args,
    *(m.apply for m in models[:3]),
    state.native,
    batch,
    it,
    rng,
    max_action,
    -float(batch.action.shape[-1]),
    conservative_multiplier=dose.dose
)
```

**Loss boundary.** Each critic uses this helper. BCA weights the clipped
per-transition conservative gap *before averaging*. The Bellman `td_loss`
is added separately. With Lagrange CQL, the target gap is subtracted after
weighting; it is not multiplied by BCA.

[algorithms/cql.py, lines 296–322](algorithms/cql.py#L296-L322)

<!-- source: algorithms/cql.py:296:322 -->
```python
q_diff = jnp.mean(
    jnp.clip(q_ood - q_pred, args.cql_clip_diff_min, args.cql_clip_diff_max)
)
critic_gap = q_diff
if conservative_multiplier is not None:
    dose = jnp.asarray(conservative_multiplier)
    if dose.shape != q_pred.shape:
        raise ValueError("conservative_multiplier must have shape (batch,)")
    dose = jax.lax.stop_gradient(
        jnp.where(jnp.all(jnp.isfinite(dose) & (dose >= 0)), dose, jnp.nan)
    )
    critic_gap = jnp.mean(
        dose
        * jnp.clip(
            q_ood - q_pred, args.cql_clip_diff_min, args.cql_clip_diff_max
        )
    )
if args.cql_lagrange:
    min_q_loss = (
        alpha_prime * args.cql_alpha * (critic_gap - args.cql_target_action_gap)
    )
else:
    min_q_loss = critic_gap * args.cql_alpha
return (
    td_loss + min_q_loss,
    (td_loss, q_diff, min_q_loss, q_pred, q_rand, q_cur, q_nxt, std_q),
)
```

**Dual boundary.** The auxiliary return above retains `q_diff`, the unweighted
gap. The alpha-prime optimizer consumes that value. It does not consume
`critic_gap`. This distinction matters even though later learned Q values can
make the dual trajectory different.

[algorithms/cql.py, lines 342–352](algorithms/cql.py#L342-L352)

<!-- source: algorithms/cql.py:342:352 -->
```python
if args.cql_lagrange:

    def _alpha_prime_loss_fn(params):
        ap = jnp.clip(jnp.exp(params["constant"]), 0.0, 1000000.0)
        l1 = ap * args.cql_alpha * (qf1_diff - args.cql_target_action_gap)
        l2 = ap * args.cql_alpha * (qf2_diff - args.cql_target_action_gap)
        return (-l1 - l2) * 0.5

    alpha_prime_loss, alpha_prime_grad = jax.value_and_grad(_alpha_prime_loss_fn)(
        agent_state.log_alpha_prime.params
    )
```

There is no direct BCA multiplier in the SAC actor loss, entropy-alpha loss,
Bellman target or target-network update. All current-step CQL losses are computed
from the host's pre-update state. The BCA-modified critics affect later steps.

| Component | Exact function entry points |
|---|---|
| Direct hook and both critic losses | [`cql_update`](algorithms/cql.py#L151) |
| Scale/width and host call | [`update`](algorithms/cql_bca.py#L339), [`make_train_step`](algorithms/cql_bca.py#L380) |

## IQL: actor regression weights

IQL is structurally different: the extension runs **two actors with one shared
Q/V state**. It calls the host's `nuisance_update`, then routes each actor through
a guarded regression update. It does not pass a multiplier into `iql_update`.

**Declared pair.** One scale fitter (no importance tilt by default), the same host beta for both actors,
and a fixed decision gain of one. The `host` actor has mode `off`.

[algorithms/iql_bca.py, lines 74–85](algorithms/iql_bca.py#L74-L85)

<!-- source: algorithms/iql_bca.py:74:85 -->
```python
def default_design(published_beta, *, fitting_mode="awr"):
    """The only paired design: shared Q/V, identical beta, fixed gain1, one scale fitter."""
    _positive(published_beta, "host beta")
    variant = ScaleVariant(
        "noiw" if fitting_mode == "off" else "awr",
        W.ScaleIWConfig(mode=fitting_mode, beta=published_beta), True
    )
    actors = (
        IWArm("host", -1, "off", published_beta),
        IWArm("bca", 0, "full", published_beta),
    )
    return ((variant,), actors)
```

**Q/V boundary.** The shared nuisance update reads the recorded minibatch. It
has no BCA actor-weight argument. Its returned target and advantage feed fitting.

[algorithms/iql_bca.py, lines 236–246](algorithms/iql_bca.py#L236-L246)

<!-- source: algorithms/iql_bca.py:236:246 -->
```python
it = it + 1
rng, batch_key, dropout_key = jax.random.split(rng, 3)
idx = jax.random.randint(batch_key, (args.batch_size,), 0, n)
batch = jax.tree_util.tree_map(lambda x: x[idx], dataset)
prior_valid = P.finite_tree((state.qf, state.qf_target, state.vf))
state, adv, target, loss = H.nuisance_update(
    args, state.qf.apply_fn, state.vf.apply_fn, state, batch
)
nuisance_valid = prior_valid & P.finite_tree(
    (state.qf, state.qf_target, state.vf, loss, adv, target)
)
```

**Weight-selection boundary.** The BCA actor reads the frozen reference; the
host actor uses capped native AWR. Until a reference is ready, the BCA path
selects the native weights as well. This is equality of the intended weights,
not a claim of bitwise identical trajectories across distinct execution paths.

[algorithms/iql_bca.py, lines 194–215](algorithms/iql_bca.py#L194-L215)

<!-- source: algorithms/iql_bca.py:194:215 -->
```python
def arm_weights(args, arms, extras, predictions, advantage):
    rows = []
    for arm in arms:
        native = S.native_weights(advantage, arm.beta)
        if arm.mode == "off":
            row = native
        else:
            ref = extras[arm.variant_index].posterior
            config = replace(args.posterior, mode=arm.mode, decision_gain=arm.gain)
            row = P.weights_at_reference(
                config,
                ref,
                predictions[arm.variant_index],
                advantage,
                arm.beta,
                H.EXP_ADV_MAX,
            )
            row = jax.tree_util.tree_map(
                lambda new, old: jnp.where(ref.ready, new, old), row, native
            )
        rows.append(row)
    return jax.tree_util.tree_map(lambda *xs: jnp.stack(xs), *rows)
```

**Weight formula.** For positive advantage `A`, the final weight is
`1 + A/(A+U) * (min(exp(beta*A), cap)-1)`; for nonpositive advantage it remains
the native capped weight. The implementation uses overflow-safe ratios and
support masks. It shrinks the excess *after* capping, not the exponent beforehand.

[calibration/advantage.py, lines 44–63](calibration/advantage.py#L44-L63)

<!-- source: calibration/advantage.py:44:63 -->
```python
support = base.support_mask & valid
safe_adv = jnp.where(support, advantages, 0.0)
safe_width = jnp.where(support, widths, 0.0)
safe_beta = jnp.where(valid, beta, 0.0)
safe_cap = jnp.where(valid, max_weight, 1.0)
positive = jnp.maximum(safe_adv, 0.0)
scale = jnp.maximum(positive, safe_width)
safe_scale = jnp.where(scale > 0.0, scale, 1.0)
rescale = jnp.where(safe_scale > jnp.finfo(dtype).max / 4.0, 0.25, 1.0)
denominator_scale = safe_scale * rescale
a = positive * rescale / denominator_scale
b = safe_width * rescale / denominator_scale
fraction = jnp.clip(a / jnp.where(a + b > 0.0, a + b, 1.0), 0.0, 1.0)
native = jnp.minimum(
    jnp.exp(jnp.minimum(safe_beta * safe_adv, jnp.log(safe_cap))), safe_cap
)
bonus = fraction * (native - 1.0)
positive_weight = jnp.minimum(jnp.maximum(1.0 + bonus, 1.0), native)
bounded = jnp.where(safe_adv > 0.0, positive_weight, native)
weights = jnp.where(support, bounded, 0.0)
```

**Actor-loss boundary.** The same Gaussian negative log likelihood or
deterministic squared regression error is used. `weights.weights` replaces the
native capped AWR multiplier. Inputs outside support are masked; a row with zero
weight contributes zero to the batch mean. The guarded helper lives in
`calibration/iql_state.py`; the standalone host's corresponding regression loss
is in `algorithms/iql.py`. They are distinct execution paths using the same host
network/log-probability definitions.

[calibration/iql_state.py, lines 45–79](calibration/iql_state.py#L45-L79)

<!-- source: calibration/iql_state.py:45:79 -->
```python
def actor_update(args, actor_state, batch, advantage, dropout_key, weights):
    del advantage
    safe_obs = jnp.where(weights.support_mask[:, None], batch.obs, 0.0)
    safe_action = jnp.where(weights.support_mask[:, None], batch.action, 0.0)

    def propose(_):

        def loss(params):
            out = actor_state.apply_fn(
                params, safe_obs, deterministic=False, rngs={"dropout": dropout_key}
            )
            if args.iql_deterministic:
                terms = jnp.sum((out - safe_action) ** 2, axis=1)
            else:
                mean, std = out
                terms = -gaussian_log_prob(mean, std, safe_action).sum(-1)
            return jnp.mean(weights.weights * terms)

        value, grad = jax.value_and_grad(loss)(actor_state.params)
        proposed = actor_state.apply_gradients(grads=grad)
        valid = jnp.isfinite(value) & P.finite_tree(grad) & P.finite_tree(proposed)
        return (
            jax.lax.cond(valid, lambda _: proposed, lambda _: actor_state, None),
            value,
            valid,
        )

    def skip(_):
        return (
            actor_state,
            jnp.asarray(0.0),
            weights.inputs_valid & P.finite_tree(actor_state),
        )

    return jax.lax.cond(weights.has_support & weights.inputs_valid, propose, skip, None)
```

**Actor acceptance boundary.** A BCA actor can abstain if no usable reference
support exists. This is an additional control-flow boundary, not just a scalar
coefficient. A numerically invalid proposed actor update is also rejected.

[calibration/iql_state.py, lines 79–79](calibration/iql_state.py#L79-L79)

<!-- source: calibration/iql_state.py:79:79 -->
```python
return jax.lax.cond(weights.has_support & weights.inputs_valid, propose, skip, None)
```

**Shared execution boundary.** Both actors receive the same minibatch,
advantage and dropout key. The state keeps a representative actor for its data
structure, but Q/V updates use neither actor's actions.

[algorithms/iql_bca.py, lines 302–322](algorithms/iql_bca.py#L302-L322)

<!-- source: algorithms/iql_bca.py:302:322 -->
```python
weights = arm_weights(args, arms, extras, predictions, adv)
dependency_valid = jnp.stack(
    [
        (
            nuisance_valid
            if a.mode == "off"
            else nuisance_valid
            & schedule_valid
            & detail["numerical_valid"][a.variant_index]
        )
        for a in arms
    ]
)
weights = weights._replace(
    inputs_valid=weights.inputs_valid & dependency_valid,
    has_support=weights.has_support & dependency_valid,
)
actors, actor_loss, actor_valid = S.update_actors(
    args, actors, batch, adv, dropout_key, weights
)
state = state._replace(actor=S.actor_at(actors, representative))
```

| Component | Exact function entry points |
|---|---|
| Unmodified shared Q/V learning | [`nuisance_update`](algorithms/iql.py#L258) |
| Paired connection | [`initialize_shared`](algorithms/iql_bca.py#L172), [`arm_weights`](algorithms/iql_bca.py#L194), [`make_shared_train_step`](algorithms/iql_bca.py#L218) |
| Frozen width to weights | [`posterior_width`](calibration/iql_reference.py#L156), [`weights_at_reference`](calibration/iql_reference.py#L184), [`gain_actor_weights`](calibration/iql_reference.py#L99) |
| Post-cap formula | [`postcap_level_actor_weights`](calibration/advantage.py#L32) |
| Actor vectorization | [`update_actors`](calibration/iql_actors.py#L44) |
| Guarded regression | [`actor_update`](calibration/iql_state.py#L45) |

## Preparation and added state

**Dataset boundary.** Calibration reserves blocks before learning. Both active
`host` and `bca` methods train on the same complement and share the same host
preprocessing. Thus reservation is part of the paired experimental protocol,
not a hidden BCA-only reduction in data. Held-out rows are excluded from scale
gradient fitting as well as host gradient updates. A separate training subset
is used to construct the frozen reference. For example, TD3+BC does:

[runtime/td3_bc.py, lines 667–673](runtime/td3_bc.py#L667-L673)

<!-- source: runtime/td3_bc.py:667:673 -->
```python
training_ids, heldout_ids, inherited = P.reserve_pool(
    data,
    r.target_size,
    r.seed,
    max_fraction=r.max_fraction,
    episode_ids=np.asarray(maps["effective_components"]),
)
```

[runtime/td3_bc.py, lines 740–745](runtime/td3_bc.py#L740-L745)

<!-- source: runtime/td3_bc.py:740:745 -->
```python
take = lambda ids: jax.tree_util.tree_map(lambda x: x[ids], all_data)
training = P.select_training_pool(all_data, cfg.arm, training_ids)
heldout, reference = (
    take(heldout_ids) if len(heldout_ids) else None,
    take(reference_ids) if len(reference_ids) else None,
)
```

| Component | Exact function entry points |
|---|---|
| TD3+BC data preparation | [`prepare`](runtime/td3_bc.py#L643) |
| ReBRAC data preparation | [`prepare`](runtime/rebrac.py#L593) |
| CQL data preparation | [`prepare`](runtime/cql.py#L358) |
| IQL data preparation | [`prepare_dataset`](calibration/iql_reference.py#L275) |
| Whole-block reservation primitive | [`reserve_calibration`](calibration/reference.py#L32) |

**Initialization boundary.** BCA reuses the host initialization and adds a
separate calibrator optimizer, residual unit and frozen-reference state. It
uses folded keys for the extra randomness. The host baseline carries no scale
optimizer. These are the four actual initialization sites:

[algorithms/td3_bc_bca.py, lines 217–234](algorithms/td3_bc_bca.py#L217-L234)

<!-- source: algorithms/td3_bc_bca.py:217:234 -->
```python
rng, native, (actor, critic) = BASE.initialize(
    args, obs_dim, action_dim, max_action
)
obs, action = jnp.zeros(obs_dim), jnp.zeros(action_dim)
if config.arm != "bca":
    return (rng, State(native, None, None, None), (actor, critic, None))
cal = Calibrator(jnp.zeros(obs_dim), jnp.ones(obs_dim), state_dep=True)
cal_state = TrainState.create(
    apply_fn=cal.apply,
    params=cal.init(jax.random.fold_in(rng, 1128352841), obs[None], action[None]),
    tx=optax.adam(config.cal_lr),
)
posterior = initialize_posterior(cal_state.params, 1, config.posterior.draws)
return (
    rng,
    State(native, cal_state, jnp.asarray(1.0), posterior),
    (actor, critic, cal),
)
```

[algorithms/rebrac_bca.py, lines 261–279](algorithms/rebrac_bca.py#L261-L279)

<!-- source: algorithms/rebrac_bca.py:261:279 -->
```python
obs, action = (training.obs[:1], training.action[:1])
rng, native, (actor, critic) = BASE.initialize(args, training)
if not config.is_posterior:
    return (rng, State(native, None, None, None, None, None), (actor, critic, None))
mean, std = (jnp.mean(training.obs, axis=0), jnp.std(training.obs, axis=0, ddof=0))
if not bool(np.asarray(tree_finite((mean, std)))):
    raise ValueError("calibrator training-pool statistics must be finite")
cal = Calibrator(mean, std, state_dep=True)
cal_state = TrainState.create(
    apply_fn=cal.apply,
    params=cal.init(jax.random.fold_in(rng, KEY_CAL_INIT), obs, action),
    tx=optax.adam(config.cal_lr),
)
posterior = initialize_posterior(cal_state.params, 1, config.posterior.draws)
return (
    rng,
    State(native, cal_state, jnp.asarray(1.0), posterior, mean, std),
    (actor, critic, cal),
)
```

[algorithms/cql_bca.py, lines 127–140](algorithms/cql_bca.py#L127-L140)

<!-- source: algorithms/cql_bca.py:127:140 -->
```python
rng, native, (actor, c1, c2) = BASE.initialize(
    args, obs_dim, action_dim, max_action
)
obs, action = jnp.zeros(obs_dim), jnp.zeros(action_dim)
if config.arm != "bca":
    return (rng, State(native, None, None, None), (actor, c1, c2, None))
cal = Calibrator(jnp.zeros(obs_dim), jnp.ones(obs_dim), state_dep=True)
cal_state = TrainState.create(
    apply_fn=cal.apply,
    params=cal.init(jax.random.fold_in(rng, 1128352841), obs[None], action[None]),
    tx=optax.adam(config.cal_lr),
)
post = initialize_posterior(cal_state.params, 1, config.posterior.draws)
return (rng, State(native, cal_state, jnp.asarray(1.0), post), (actor, c1, c2, cal))
```

[algorithms/iql_bca.py, lines 177–191](algorithms/iql_bca.py#L177-L191)

<!-- source: algorithms/iql_bca.py:177:191 -->
```python
_validate_arms(args, arms, len(fitters))
extras = tuple(
    (
        H.PosteriorTrainState(
            f.initial,
            P.POST.initialize_posterior(
                f.initial.calibrator.params, 1, args.posterior.draws
            ),
        )
        for f in fitters
    )
)
return SharedIWCarry(
    rng, state, jnp.int32(0), extras, S.stack_actors(state.actor, len(arms))
)
```

IQL's fitter is created by `make_fitters`; its initialized host actor is stacked
into the paired state. ReBRAC's scale network has training-pool normalization
statistics even when the host's observation normalization is disabled. These
statistics belong to the calibrator; they do not silently normalize the host.

## How the host feeds scale fitting

This direction is **host → BCA**. Fitting reads detached host predictions and
targets, updates only scale parameters, and never differentiates the fitting
loss into the actor or critics. Three distinct weight concepts stay separate:
fitting importance weights, bootstrap masses, and consumed host weights.

The current experiment setting is `calibration_weighting: none`. TD3+BC, ReBRAC
and CQL initialize fitting masses to the Bayesian bootstrap prior and skip the
`config.iw.mode != "off"` branch. The conditional IW snippets below are retained
to map that separate path; they are inactive in the first-stage comparison.

**TD3+BC residual and optional affinity.** The target uses target-policy smoothing with a
folded fitting key; Q is the online twin minimum on recorded actions.

[algorithms/td3_bc_bca.py, lines 272–287](algorithms/td3_bc_bca.py#L272-L287)

<!-- source: algorithms/td3_bc_bca.py:272:287 -->
```python
target = native_target(
    args,
    models,
    state.native,
    batch,
    jax.random.fold_in(rng, 1179210836),
    max_action,
)
q = jax.lax.stop_gradient(
    models[1]
    .apply(state.native.critic.params, batch.obs, batch.action)
    .min(axis=-1)
)
prior = jax.lax.stop_gradient(
    bayesian_bootstrap_weights(jax.random.fold_in(rng, 1128352850), len(batch.obs))
)
```

[algorithms/td3_bc_bca.py, lines 291–309](algorithms/td3_bc_bca.py#L291-L309)

<!-- source: algorithms/td3_bc_bca.py:291:309 -->
```python
signal = affinity_log_weights(
    models[0].apply,
    state.native.actor.params,
    batch.obs,
    batch.action,
    config.iw.bandwidth,
    max_action,
)
tilted = fit_weighting(
    signal.log_weights,
    prior,
    jax.random.fold_in(rng, 1413761367),
    config.iw.canonical(),
)
weights, valid = (
    tilted.product.weights,
    signal.inputs_valid & tilted.inputs_valid,
)
feasible = tilted.fit_feasible
```

**ReBRAC residual and optional affinity.** Its fitting target retains ReBRAC's recorded
next-action critic BC penalty. Standard BCA uses Bayesian bootstrap masses directly;
the optional IW branch below tilts those masses using action affinity.

[algorithms/rebrac_bca.py, lines 402–412](algorithms/rebrac_bca.py#L402-L412)

<!-- source: algorithms/rebrac_bca.py:402:412 -->
```python
target = native_target(
    args, models, state.native, batch, jax.random.fold_in(rng, KEY_FIT_NOISE)
)
q = jax.lax.stop_gradient(
    models[1].apply(state.native.critic.params, batch.obs, batch.action).min(0)
)
prior = jax.lax.stop_gradient(
    bayesian_bootstrap_weights(
        jax.random.fold_in(rng, KEY_BOOTSTRAP), len(batch.obs)
    )
)
```

[algorithms/rebrac_bca.py, lines 415–432](algorithms/rebrac_bca.py#L415-L432)

<!-- source: algorithms/rebrac_bca.py:415:432 -->
```python
signal = affinity_log_weights(
    models[0].apply,
    state.native.actor.params,
    batch.obs,
    batch.action,
    config.iw.bandwidth,
)
tilted = fit_weighting(
    signal.log_weights,
    prior,
    jax.random.fold_in(rng, KEY_AFFINITY),
    config.iw.canonical(),
)
weights, valid = (
    tilted.product.weights,
    valid & signal.inputs_valid & tilted.inputs_valid,
)
feasible = tilted.fit_feasible
```

**CQL residual and policy density.** The target retains the configured CQL
backup; Q is the online twin minimum. Policy log density is evaluated on recorded
actions and becomes scale-fitting weights, not actor or Bellman-loss weights.

[algorithms/cql_bca.py, lines 211–224](algorithms/cql_bca.py#L211-L224)

<!-- source: algorithms/cql_bca.py:211:224 -->
```python
target = native_target(
    args,
    models,
    state.native,
    batch,
    jax.random.fold_in(rng, 1179210836),
    max_action,
)
q = jax.lax.stop_gradient(
    jnp.minimum(
        models[1].apply(state.native.critic1.params, batch.obs, batch.action),
        models[2].apply(state.native.critic2.params, batch.obs, batch.action),
    )
)
```

[algorithms/cql_bca.py, lines 230–243](algorithms/cql_bca.py#L230-L243)

<!-- source: algorithms/cql_bca.py:230:243 -->
```python
if config.iw.mode != "off":
    signal = policy_log_weights(
        models[0].apply,
        state.native.actor.params,
        batch.obs,
        batch.action,
        max_action,
    )
    weighted = fit_weighting(
        signal.log_weights, prior, jax.random.fold_in(rng, 1129531735), config.iw
    )
    weights = weighted.product.weights
    valid = signal.inputs_valid & weighted.inputs_valid
    feasible = weighted.fit_feasible
```

**IQL residual and optional AWR fitting.** The target uses pre-update V; the scale reads
post-update online Q. The adapter explicitly detaches both. Standard BCA uses
equal importance factors and unchanged Bayesian bootstrap masses. In the optional
IW branch, AWR fitting weights use the pre-BCA advantage. The complete weighting
result is detached; the native actor still uses its original AWR weights.

[calibration/iql_targets.py, lines 22–37](calibration/iql_targets.py#L22-L37)

<!-- source: calibration/iql_targets.py:22:37 -->
```python
def calibration_inputs(self, state, batch, rng, step, host=None):
    if host is None or "target" not in host:
        raise ValueError("Supply the host pre-update V target.")
    heads = self.q_apply_fn(state.qf.params, batch.obs, batch.action)
    return CalibrationInputs(
        targets=jax.lax.stop_gradient(host["target"]),
        q_at_data=jax.lax.stop_gradient(jnp.min(heads, axis=-1)),
        q_heads=jax.lax.stop_gradient(heads),
        obs=batch.obs,
        action=batch.action,
        vintage={
            "targets": "pre_update_value_net",
            "q_at_data": "post_update_online",
            "q_heads": "post_update_online",
        },
    )
```

[calibration/iql_scale.py, lines 60–86](calibration/iql_scale.py#L60-L86)

<!-- source: calibration/iql_scale.py:60:86 -->
```python
if not isinstance(config, ScaleIWConfig):
    raise ValueError("scale_weighting requires a typed ScaleIWConfig")
advantage = jnp.asarray(advantage)
if advantage.ndim != 1 or advantage.size == 0:
    raise ValueError("advantage must be a nonempty vector")
# Uniform importance factors preserve the SAME Bayesian bootstrap prior.
# The actor still uses native AWR; only calibration importance tilting is off.
raw = (IW.IWLogWeights(jnp.zeros_like(advantage), jnp.ones(advantage.shape, dtype=bool),
                      jnp.all(jnp.isfinite(advantage)))
       if config.mode == "off" else
       IW.capped_awr_log_weights(advantage, config.beta, config.cap, config.mixing))
iw = IW.stabilize_log_weights(
    raw.log_weights,
    support=raw.support_mask,
    ess_floor=config.ess_floor,
    tau_min=config.tau_min,
    iterations=config.iterations,
)
n = raw.log_weights.size
prior = BF.bayesian_bootstrap_weights(
    jax.random.fold_in(key, BF._FOLD_BOOTSTRAP), n
)
permutation = jnp.arange(n)
product = IW.tilt_bootstrap_weights(prior, iw)
return jax.tree_util.tree_map(
    jax.lax.stop_gradient, ScaleWeighting(prior, raw, iw, product, permutation)
)
```

**Scale optimizer boundary.** TD3+BC's objective below is representative of
TD3+BC/ReBRAC/CQL's weighted soft-coverage plus width objectives. IQL retains its
own coverage-balance term and batch residual-unit convention; it is not silently
replaced by this loss. See the complete pseudocode for those differences.

[algorithms/td3_bc_bca.py, lines 317–335](algorithms/td3_bc_bca.py#L317-L335)

<!-- source: algorithms/td3_bc_bca.py:317:335 -->
```python
def objective(params):
    eta = models[2].apply(params, batch.obs, batch.action)
    if eta.shape != (len(batch.obs),):
        raise ValueError("scale predictor must emit one positive value per row")
    coverage = soft_coverage(target / unit, q / unit, eta, config.cal_beta)
    width = jnp.sum(weights * jnp.square(eta))
    return (
        jnp.square(jnp.sum(weights * coverage) - (1.0 - config.posterior.alpha))
        + config.width_penalty * width,
        eta,
    )

(loss, eta), grad = jax.value_and_grad(objective, has_aux=True)(
    state.calibrator.params
)
proposed = state.calibrator.apply_gradients(grads=grad)
unit_new = config.scale_ema * unit + (1.0 - config.scale_ema) * jnp.maximum(
    jnp.std(target - q), 1e-06
)
```

| Component | Exact function entry points |
|---|---|
| TD3+BC fitting target/optimizer | [`native_target`](algorithms/td3_bc_bca.py#L256), [`affinity_log_weights`](algorithms/td3_bc_bca.py#L237), [`fit_scale`](algorithms/td3_bc_bca.py#L269) |
| ReBRAC fitting target/optimizer | [`native_target`](algorithms/rebrac_bca.py#L383), [`affinity_log_weights`](algorithms/rebrac_bca.py#L365), [`fit_scale`](algorithms/rebrac_bca.py#L398) |
| CQL fitting target/optimizer | [`native_target`](algorithms/cql_bca.py#L176), [`policy_log_weights`](algorithms/cql_bca.py#L143), [`fit_scale`](algorithms/cql_bca.py#L210) |
| IQL fitting target | [`CorlIQLAdapter`](calibration/iql_targets.py#L11) |
| IQL weighting and scale optimizer | [`scale_weighting`](calibration/iql_scale.py#L59), [`IWScaleFitter`](calibration/iql_scale.py#L89) |
| IQL scale objective | [`make_cal_loss_fn`](calibration/iql_network.py#L52) |

## Refresh, frozen width and gradients

**Reference boundary.** Each host refresh computes held-out residuals using
its current host state, then snapshots the current scale parameters and unit.
These are all four calls into the shared radius builder. No fitting importance
weights are passed into these calls.

[algorithms/td3_bc_bca.py, lines 417–426](algorithms/td3_bc_bca.py#L417-L426)

<!-- source: algorithms/td3_bc_bca.py:417:426 -->
```python
post = fit_posterior(
    state.calibrator.params,
    state.residual_scale,
    fit,
    cal,
    target - q,
    jax.random.fold_in(rng, 1347375956),
    config.posterior,
    groups=1,
)
```

[algorithms/rebrac_bca.py, lines 514–523](algorithms/rebrac_bca.py#L514-L523)

<!-- source: algorithms/rebrac_bca.py:514:523 -->
```python
post = fit_posterior(
    state.calibrator.params,
    state.residual_scale,
    fit,
    cal,
    target - q,
    jax.random.fold_in(rng, KEY_POSTERIOR),
    config.posterior,
    groups=1,
)
```

[algorithms/cql_bca.py, lines 316–325](algorithms/cql_bca.py#L316-L325)

<!-- source: algorithms/cql_bca.py:316:325 -->
```python
post = fit_posterior(
    state.calibrator.params,
    state.residual_scale,
    fit_predictions,
    cal_predictions,
    target - q,
    jax.random.fold_in(rng, 1347375956),
    config.posterior,
    groups=1,
)
```

[calibration/iql_reference.py, lines 212–221](calibration/iql_reference.py#L212-L221)

<!-- source: calibration/iql_reference.py:212:221 -->
```python
state = POST.fit_posterior(
    cal_state.calibrator.params,
    cal_state.resid_scale,
    pred_fit,
    pred_cal,
    residual,
    key,
    PosteriorConfig(args.alpha, args.credibility, args.draws),
    groups=1,
)
```

**Score boundary.** Residuals are divided by the positive frozen scale/unit;
the active designs use one global group. Bayesian and conformal components
operate on these unweighted held-out scores.

[calibration/reference.py, lines 132–147](calibration/reference.py#L132-L147)

<!-- source: calibration/reference.py:132:147 -->
```python
edges = jnp.quantile(fit_scale, jnp.arange(1, groups) / groups)
labels = jnp.searchsorted(edges, cal_scale, side="right")
radii = partitioned_posterior(
    jnp.abs(residuals) / cal_scale, labels, rng, config, num_groups=groups
)
valid = jnp.all(jnp.isfinite(fit_scale) & (fit_scale > 0)) & jnp.all(
    jnp.isfinite(cal_scale) & (cal_scale > 0)
)
radii = radii._replace(
    radius=jnp.where(valid, radii.radius, jnp.inf),
    finite=radii.finite & valid,
    inputs_valid=radii.inputs_valid & valid,
)
return IQLPosteriorState(
    cal_params, residual_scale, edges, radii, jnp.asarray(True)
)
```

**Radius boundary.** Both radii are retained; the consumed radius is their maximum.

[calibration/posterior.py, lines 143–148](calibration/posterior.py#L143-L148)

<!-- source: calibration/posterior.py:143:148 -->
```python
qs = sorted_scores[jnp.minimum(indices, n - 1)]
posterior_rank = _rank(config.draws, Decimal(str(config.credibility))) - 1
bayes = jnp.sort(qs)[posterior_rank]
floor = jnp.where(valid, floor, jnp.inf)
bayes = jnp.where(valid, bayes, jnp.inf)
radius = jnp.maximum(floor, bayes)
```

**Width and warmup boundary.** For TD3+BC/ReBRAC/CQL, the width uses frozen
parameters, frozen unit and full radius. Before readiness it is zero, yielding
the unit host multiplier. An unusable infinite width is handled by the support
mask, not blindly inserted into a loss. IQL uses the parallel `posterior_width`
and the explicit native-weight fallback shown earlier.

[calibration/dose.py, lines 77–80](calibration/dose.py#L77-L80)

<!-- source: calibration/dose.py:77:80 -->
```python
radius = (state.radii.radius if mode == "full" else state.radii.conformal_radius)[0]
unit = state.residual_scale
scale = jnp.maximum(predictions, 1e-06) * unit
width = radius * scale
```

[calibration/dose.py, lines 95–100](calibration/dose.py#L95-L100)

<!-- source: calibration/dose.py:95:100 -->
```python
width = jnp.where(state.ready, width, jnp.zeros_like(width))
valid = jnp.where(state.ready, valid, True)
unit = jnp.where(state.ready, unit, 1.0)
return level_critic_dose(
    width, jnp.ones_like(width, dtype=bool), unit, blend, valid
)
```

**Consumption and gradient boundary.** The overflow-safe implementation below
computes `1 + blend * U/(U + frozen_unit)` and detaches its result. This multiplies
the specific loss term shown for each host. It does not optimize the scale through
the host loss. The IQL weighting result is also detached at its return boundary.

[calibration/dose.py, lines 52–62](calibration/dose.py#L52-L62)

<!-- source: calibration/dose.py:52:62 -->
```python
support = valid & usable & jnp.isfinite(widths)
safe_u, safe_c = (jnp.where(support, widths, 0.0), jnp.where(valid_unit, unit, 1.0))
normalizer = jnp.maximum(safe_u, safe_c)
rescale = jnp.where(normalizer > jnp.finfo(dtype).max / 4.0, 0.25, 1.0)
denominator_scale = normalizer * rescale
u_norm = safe_u * rescale / denominator_scale
c_norm = safe_c * rescale / denominator_scale
ratio = u_norm / (u_norm + c_norm)
dose = 1.0 + jnp.where(valid_blend, blend, 0.0) * ratio
result = CQLLevelDose(widths, dose, support, valid, jnp.any(support))
return jax.tree_util.tree_map(jax.lax.stop_gradient, result)
```

[calibration/advantage.py, lines 69–70](calibration/advantage.py#L69-L70)

<!-- source: calibration/advantage.py:69:70 -->
```python
result = LevelActorWeights(weights, used, support, valid, jnp.any(support))
return jax.tree_util.tree_map(jax.lax.stop_gradient, result)
```

| Component | Exact function entry points |
|---|---|
| Shared positive scale/reference | [`positive_scale`](calibration/reference.py#L104), [`fit_posterior`](calibration/reference.py#L108) |
| Bayesian/conformal radius | [`posterior_radius`](calibration/posterior.py#L45), [`partitioned_posterior`](calibration/posterior.py#L167) |
| BC/CQL frozen width and detached dose | [`frozen_level_dose`](calibration/dose.py#L65), [`level_critic_dose`](calibration/dose.py#L23) |
| IQL frozen width/weight | [`posterior_width`](calibration/iql_reference.py#L156), [`weights_at_reference`](calibration/iql_reference.py#L184) |
| Per-host refresh | [`refresh`](algorithms/td3_bc_bca.py#L379) |
| Per-host refresh | [`refresh`](algorithms/rebrac_bca.py#L487) |
| Per-host refresh | [`refresh`](algorithms/cql_bca.py#L291) |
| IQL refresh | [`refresh`](calibration/iql_reference.py#L194) |

## Acceptance, scheduling and evaluation

**ESS abstention is not a whole-host stop.** A finite but infeasible fitting
weight distribution leaves the live calibrator and unit unchanged. A previously
valid frozen reference can still be consumed. Numerical invalidity is a separate
failure path. Here are the scale-acceptance boundaries for the common pattern
and IQL:

[algorithms/td3_bc_bca.py, lines 360–366](algorithms/td3_bc_bca.py#L360-L366)

<!-- source: algorithms/td3_bc_bca.py:360:366 -->
```python
accepted = valid & feasible
result = jax.lax.cond(
    accepted,
    lambda _: state._replace(calibrator=proposed, residual_scale=unit_new),
    lambda _: state,
    None,
)
```

[calibration/iql_scale.py, lines 185–191](calibration/iql_scale.py#L185-L191)

<!-- source: calibration/iql_scale.py:185:191 -->
```python
numerical_valid = (
    raw_valid & P.finite_tree(proposed) & (proposed.resid_scale > 0)
)
has_support = (weighting.iw.supported_count > 0) & jnp.any(weights > 0)
feasible = weighting.iw.ess_feasible
accepted = numerical_valid & feasible & has_support
out = jax.lax.cond(accepted, lambda _: proposed, lambda _: state, operand=None)
```

**Whole-step numerical guards.** TD3+BC and CQL reject a proposed combined state
when their validity checks fail; ReBRAC also rolls back its returned RNG.
The runtime then raises on invalid metrics instead of silently counting success.
IQL's actor masks/acceptance were shown above, and its runtime checks all returned
validity flags. These guards can stop progress and therefore are part of the map.

[algorithms/td3_bc_bca.py, lines 489–493](algorithms/td3_bc_bca.py#L489-L493)

<!-- source: algorithms/td3_bc_bca.py:489:493 -->
```python
result = jax.lax.cond(
    valid, lambda _: fitted._replace(native=native), lambda _: state, None
)
metrics.update(diag, inputs_valid=valid)
return (result, metrics)
```

[algorithms/rebrac_bca.py, lines 592–599](algorithms/rebrac_bca.py#L592-L599)

<!-- source: algorithms/rebrac_bca.py:592:599 -->
```python
result, key = jax.lax.cond(
    valid,
    lambda _: (fitted._replace(native=native), returned_rng),
    lambda _: (state, rng),
    None,
)
metrics.update(diag, inputs_valid=valid)
return (result, key, metrics)
```

[algorithms/cql_bca.py, lines 368–370](algorithms/cql_bca.py#L368-L370)

<!-- source: algorithms/cql_bca.py:368:370 -->
```python
valid = fit_diag["scale_inputs_valid"] & dose.inputs_valid & tree_finite(native)
proposed = fitted._replace(native=native)
result = jax.lax.cond(valid, lambda _: proposed, lambda _: state, operand=None)
```

**Scheduling boundary.** Scale fitting is attempted each host update. Refreshes
occur after updates 10k, 15k, ..., 995k; their new reference is consumed on later
updates. The runtime replaces only the extra reference state at refresh. TD3's
actual call is shown here; the table links every runner's corresponding entry.

[runtime/td3_bc.py, lines 1127–1143](runtime/td3_bc.py#L1127-L1143)

<!-- source: runtime/td3_bc.py:1127:1143 -->
```python
def refresh_at(event):
    nonlocal carry
    key = jax.random.fold_in(jax.random.PRNGKey(event.seed), event.step)
    frozen_identity = _tree_hash(carry[1])
    proposal, metrics = P.refresh(
        args,
        cfg,
        models,
        carry[1],
        prepared.reference,
        prepared.heldout,
        key,
        prepared.max_action,
        training_ids=prepared.reference_ids,
        heldout_ids=prepared.heldout_ids,
    )
    _accept(proposal, metrics)
```

| Component | Exact function entry points |
|---|---|
| TD3+BC refresh scheduler and validity | [`run_prepared`](runtime/td3_bc.py#L1092), [`_accept`](runtime/td3_bc.py#L1080) |
| ReBRAC refresh scheduler and validity | [`run_prepared`](runtime/rebrac.py#L812), [`_accept`](runtime/rebrac.py#L803) |
| CQL refresh scheduler and validity | [`run_prepared`](runtime/cql.py#L724), [`_accept`](runtime/cql.py#L714) |
| IQL reference scheduler and validity | [`IWRuntime`](runtime/iql_pair.py#L271) |

**Evaluation boundary.** Evaluation uses the resulting actor parameters. There
is no extra BCA action search, rejection filter, width penalty or calibration
network call in action selection. The paired IQL evaluator explicitly receives
each actor's parameters with common episode IDs:

[runtime/iql_pair.py, lines 445–452](runtime/iql_pair.py#L445-L452)

<!-- source: runtime/iql_pair.py:445:452 -->
```python
self.checkpoint("training_end")
self.phase = "final_evaluation"
start = time.perf_counter()
ids = D.episode_ids(True, 0, self.o.final_episodes)
rows = {
    a.name: self.evaluator.evaluate(s.params, ids)
    for a, s in zip(self.arms, self.actors())
}
```

For the other hosts, see `runtime/common.py:eval_policy` and the host-specific
runner's evaluator. Deterministic actor/tanh-mean semantics remain as declared.
Component-engagement diagnostics compute alternative component readouts on a
fixed panel; they do not create additional trained actors or change the consumed
full-radius recipe. Their validity checks can still stop an invalid run.

**Saved state boundary.** The runtime saves calibrator parameters/optimizer,
residual unit, both frozen radius components, host state and RNGs. Logging and
checkpoint observers do not add learning objectives. Accepted fitting counts,
actor counts, reference refreshes and actual evaluations are recorded separately.

## Unifloral and the scope of this map

The exact Unifloral standalone scripts are pinned in
[configs/unifloral.json](configs/unifloral.json). **They have no BCA entry points.**
They are included as unmodified baseline references in [baselines/unifloral/](baselines/unifloral/), as selected
for the minimal repository. This document does not claim that our CQL twin-critic
hook is already implemented in Unifloral's ten-critic source, or that its TD3/ReBRAC
outer-step counts match our critic-step budget.

This map covers the active `host`/`bca` design. Historical Bellman weighting,
permuted controls, floor-only actors, fixed-strength arms and smooth-level IQL
recipes are not additional active methods. Original historical configurations
and results keep their identities in the archived reproduction matrix.
The map documents code behavior, not a proof of improved returns, useful OOD-harm
ranking or a conformal guarantee for the adaptive training loop.

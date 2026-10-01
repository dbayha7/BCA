# Every BCA entry point in the host

This is the code map for the **current four CORL-derived host/BCA pairs** (run
identity `bca-wbcp`). All Python blocks below are literal source excerpts.
`python check.py` checks them against their source lines, so a code change cannot
silently leave the map's snippets stale. The [complete algorithms](ALGORITHMS.md)
give the equations and full update order; this map shows where those equations
enter the code.

There are **four direct loss interventions**, plus the preparation, scale-fitting,
refresh and validity boundaries needed to supply them. The frozen threshold is
weighted Bayesian conformal prediction (WBCP; Lou and Luo, arXiv:2604.06464v3,
Algorithm 1) with uniform weights. The table is about direct equations; it does
not claim that other learned parameters remain identical after training.

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

[algorithms/td3_bc_bca.py, lines 295–311](algorithms/td3_bc_bca.py#L295-L311)

<!-- source: algorithms/td3_bc_bca.py:295:311 -->
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

**Readout boundary.** The width is evaluated with the frozen reference's own
scale parameters, `state.posterior.cal_params`, not the live calibrator. A stored
reference that fails its storage check makes the whole step invalid.

[algorithms/td3_bc_bca.py, lines 283–288](algorithms/td3_bc_bca.py#L283-L288)

<!-- source: algorithms/td3_bc_bca.py:283:288 -->
```python
def bc_readout(models, state, batch, blend):
    predictions = models[2].apply(state.posterior.cal_params, batch.obs, batch.action)
    dose = frozen_level_dose(state.posterior, predictions, blend)
    return dose._replace(
        inputs_valid=dose.inputs_valid & reference_valid(state.posterior)
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
| Width readout and connection | [`bc_readout`](algorithms/td3_bc_bca.py#L283), [`consumed_bc_multiplier`](algorithms/td3_bc_bca.py#L291), [`update`](algorithms/td3_bc_bca.py#L295), [`make_train_step`](algorithms/td3_bc_bca.py#L327) |

## ReBRAC: actor BC

**Call boundary.** The multiplier is passed only as `actor_bc_multiplier`.
The host returns its own advanced RNG; the extension carries it forward after
validating the combined state. ReBRAC's `bc_readout` has the same form as
TD3+BC's, but validates the whole calibration storage, including the fixed
calibrator statistics, through `calibration_storage_valid`.

[algorithms/rebrac_bca.py, lines 394–410](algorithms/rebrac_bca.py#L394-L410)

<!-- source: algorithms/rebrac_bca.py:394:410 -->
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
| Width readout and connection | [`bc_readout`](algorithms/rebrac_bca.py#L382), [`consumed_bc_multiplier`](algorithms/rebrac_bca.py#L390), [`update`](algorithms/rebrac_bca.py#L394), [`make_train_step`](algorithms/rebrac_bca.py#L435) |
| Calibration storage check | [`calibration_storage_valid`](algorithms/rebrac_bca.py#L244) |

## CQL: conservative critic gap

**Call boundary.** Frozen width is queried on recorded dataset actions with the
frozen reference's scale parameters. `models[:3]` are the actor and twin critics;
`models[3]` is the added scale network. The `host` arm returns earlier in the same
function, calling `cql_update` without `conservative_multiplier`.

[algorithms/cql_bca.py, lines 277–292](algorithms/cql_bca.py#L277-L292)

<!-- source: algorithms/cql_bca.py:277:292 -->
```python
fitted, fit_diag = fit_scale(args, config, models, state, batch, rng, max_action)
frozen_predictions = models[3].apply(
    state.posterior.cal_params, batch.obs, batch.action
)
dose = frozen_level_dose(state.posterior, frozen_predictions, config.blend)
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
| Scale/width and host call | [`update`](algorithms/cql_bca.py#L264), [`make_train_step`](algorithms/cql_bca.py#L311) |

## IQL: actor regression weights

IQL is structurally different: the extension runs **two actors with one shared
Q/V state**. It calls the host's `nuisance_update`, then routes each actor through
a guarded regression update. It does not pass a multiplier into `iql_update`.

**Declared pair.** One scale fitter (`wbcp_uniform`: Bayesian bootstrap masses on
both the coverage and width terms, no importance factors), the same host beta for
both actors, and a fixed decision gain of one. The `host` actor has mode `off`.

[algorithms/iql_bca.py, lines 73–81](algorithms/iql_bca.py#L73-L81)

<!-- source: algorithms/iql_bca.py:73:81 -->
```python
def default_design(published_beta):
    """The only paired design: shared Q/V, identical beta, fixed gain1, one scale fitter."""
    _positive(published_beta, "host beta")
    variant = ScaleVariant(VARIANT, True)
    actors = (
        PairArm("host", -1, "off", published_beta),
        PairArm("bca", 0, "full", published_beta),
    )
    return ((variant,), actors)
```

**Q/V boundary.** The shared nuisance update reads the recorded minibatch. It
has no BCA actor-weight argument. Its returned target and advantage feed fitting.

[algorithms/iql_bca.py, lines 216–226](algorithms/iql_bca.py#L216-L226)

<!-- source: algorithms/iql_bca.py:216:226 -->
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

[algorithms/iql_bca.py, lines 174–195](algorithms/iql_bca.py#L174-L195)

<!-- source: algorithms/iql_bca.py:174:195 -->
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

**Actor acceptance boundary.** A BCA actor abstains if no row has usable
reference support, for example when the frozen threshold is `+inf`. This is an
additional control-flow boundary, not just a scalar coefficient. A numerically
invalid proposed actor update is also rejected.

[calibration/iql_state.py, line 79](calibration/iql_state.py#L79)

<!-- source: calibration/iql_state.py:79:79 -->
```python
return jax.lax.cond(weights.has_support & weights.inputs_valid, propose, skip, None)
```

**Shared execution boundary.** Both actors receive the same minibatch,
advantage and dropout key. The state keeps a representative actor for its data
structure, but Q/V updates use neither actor's actions.

[algorithms/iql_bca.py, lines 264–284](algorithms/iql_bca.py#L264-L284)

<!-- source: algorithms/iql_bca.py:264:284 -->
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
| Paired connection | [`initialize_shared`](algorithms/iql_bca.py#L155), [`arm_weights`](algorithms/iql_bca.py#L174), [`make_shared_train_step`](algorithms/iql_bca.py#L198) |
| Frozen width to weights | [`posterior_width`](calibration/iql_reference.py#L157), [`weights_at_reference`](calibration/iql_reference.py#L174), [`gain_actor_weights`](calibration/iql_reference.py#L102) |
| Post-cap formula | [`postcap_level_actor_weights`](calibration/advantage.py#L32) |
| Actor vectorization | [`update_actors`](calibration/iql_actors.py#L44) |
| Guarded regression | [`actor_update`](calibration/iql_state.py#L45) |

## Preparation and added state

**Dataset boundary.** Before learning, reservation withholds whole blocks (episodes or
effective raw-dependency components), chosen with probability proportional to length,
and keeps K = `rows_per_episode` rows of each, one per K equal segments, as the
calibration bank. Whole blocks as the bank make the posterior overconfident, because
rows of one block are correlated (experiments/wbcp/DEPENDENCE.md). Both active `host`
and `bca` methods train on the same complement of the withheld blocks and share the
same host preprocessing. Thus reservation is part of the paired experimental protocol,
not a hidden BCA-only reduction in data. Withheld rows are excluded from scale gradient
fitting as well as host gradient updates; the bank rows are the only rows a refresh
scores, and the rest of each withheld block is used for nothing. The cap
(`max_fraction`) counts every withheld row. For example, TD3+BC does:

[runtime/td3_bc.py, lines 537–544](runtime/td3_bc.py#L537-L544)

<!-- source: runtime/td3_bc.py:537:544 -->
```python
training, withheld, heldout, inherited = P.reserve_pool(
    data,
    r.target_size,
    r.seed,
    r.rows_per_episode,
    max_fraction=r.max_fraction,
    episode_ids=components,
)
```

[runtime/td3_bc.py, lines 704–708](runtime/td3_bc.py#L704-L708)

<!-- source: runtime/td3_bc.py:704:708 -->
```python
training_ids, withheld_ids, heldout_ids, reservation = _reserve(
    maps, protocol.reservation, data
)
if cfg.arm == "bca" and not P.certifiable(len(heldout_ids), cfg.posterior):
    raise ValueError("held-out bank is too small for WBCP to certify a finite threshold")
```

[runtime/td3_bc.py, lines 711–713](runtime/td3_bc.py#L711-L713)

<!-- source: runtime/td3_bc.py:711:713 -->
```python
take = lambda ids: jax.tree_util.tree_map(lambda x: x[ids], all_data)
training = P.select_training_pool(all_data, cfg.arm, training_ids)
heldout = take(heldout_ids) if len(heldout_ids) else None
```

**Bank-size gate.** For `bca`, preparation rejects a held-out bank that is too
small for the WBCP test atom to be certified. With uniform weights a posterior
draw crosses only when the test atom's mass, Beta(1, n), is at most alpha, so the
crossing draws are Binomial(M, 1-(1-alpha)^n) and a finite threshold needs
ceil(credibility·M) of them. The gate requires that failure to have probability at
most 1e-6, which at the declared 0.1 / 0.95 / 1000 means n ≥ 36. The `host` arm
has no gate.

[calibration/reference.py, lines 123–135](calibration/reference.py#L123-L135)

<!-- source: calibration/reference.py:123:135 -->
```python
def certifiable(n, config, tolerance=1e-6):
    """Whether a uniform-weight bank of n held-out scores certifies a finite threshold.

    A posterior draw crosses iff the test atom's mass, Beta(1, n), is at most alpha,
    so the crossing draws are Binomial(M, 1 - (1 - alpha)^n) and lambda_hpd needs
    ceil(credibility * M) of them. The bank qualifies when failing that has
    probability at most `tolerance` (n >= 36 at the declared 0.1 / 0.95 / 1000).
    """
    if isinstance(n, bool) or not isinstance(n, Integral) or n < 1:
        return False
    need = math.ceil(Fraction(str(config.credibility)) * config.draws)
    crossing = -math.expm1(n * math.log1p(-config.alpha))
    return float(stats.binom.cdf(need - 1, config.draws, crossing)) <= tolerance
```

ReBRAC and CQL apply the same check right after reservation, as TD3+BC does above.
IQL applies it inside `PosteriorArgs.validate`, which `prepare_dataset` calls with
the reserved bank size and `refresh` calls again with the bank it scores:

[calibration/iql_reference.py, lines 75–80](calibration/iql_reference.py#L75-L80)

<!-- source: calibration/iql_reference.py:75:80 -->
```python
if calibration_size is not None and not POST.certifiable(
    int(calibration_size), self.config()
):
    raise ValueError(
        "calibration bank is too small for the WBCP test atom to be certified"
    )
```

| Component | Exact function entry points |
|---|---|
| TD3+BC data preparation | [`prepare`](runtime/td3_bc.py#L686), [bank-size gate](runtime/td3_bc.py#L707) |
| ReBRAC data preparation | [`prepare`](runtime/rebrac.py#L549), [bank-size gate](runtime/rebrac.py#L558) |
| CQL data preparation | [`prepare`](runtime/cql.py#L355), [bank-size gate](runtime/cql.py#L381) |
| IQL data preparation | [`prepare_dataset`](calibration/iql_reference.py#L263), [bank-size gate](calibration/iql_reference.py#L295) |
| Thinned reservation primitive | [`reserve_calibration`](calibration/reference.py#L49), [`stratified_bank`](calibration/bank.py#L21) |
| Certifiability of a uniform-weight bank | [`certifiable`](calibration/reference.py#L123) |

**Initialization boundary.** BCA reuses the host initialization and adds a
separate calibrator optimizer, a residual unit of one and a not-ready frozen
reference. It uses folded keys for the extra randomness. The host baseline
carries no calibration state. These are the four actual initialization sites:

[algorithms/td3_bc_bca.py, lines 126–143](algorithms/td3_bc_bca.py#L126-L143)

<!-- source: algorithms/td3_bc_bca.py:126:143 -->
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
posterior = initial_reference(cal_state.params)
return (
    rng,
    State(native, cal_state, jnp.asarray(1.0), posterior),
    (actor, critic, cal),
)
```

[algorithms/rebrac_bca.py, lines 210–228](algorithms/rebrac_bca.py#L210-L228)

<!-- source: algorithms/rebrac_bca.py:210:228 -->
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
posterior = initial_reference(cal_state.params)
return (
    rng,
    State(native, cal_state, jnp.asarray(1.0), posterior, mean, std),
    (actor, critic, cal),
)
```

[algorithms/cql_bca.py, lines 119–132](algorithms/cql_bca.py#L119-L132)

<!-- source: algorithms/cql_bca.py:119:132 -->
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
post = initial_reference(cal_state.params)
return (rng, State(native, cal_state, jnp.asarray(1.0), post), (actor, c1, c2, cal))
```

[algorithms/iql_bca.py, lines 159–171](algorithms/iql_bca.py#L159-L171)

<!-- source: algorithms/iql_bca.py:159:171 -->
```python
_validate_fitters(args, fitters, variants)
_validate_arms(args, arms, len(fitters))
extras = tuple(
    (
        H.PosteriorTrainState(
            f.initial, P.POST.initial_reference(f.initial.calibrator.params)
        )
        for f in fitters
    )
)
return SharedPairCarry(
    rng, state, jnp.int32(0), extras, S.stack_actors(state.actor, len(arms))
)
```

**Not-ready reference.** All four sites start from the same reference: infinite
thresholds, unit one, `n_eff` zero and `ready` false. Until the first refresh the
hosts keep their native weighting (see the width boundary below).

[calibration/reference.py, lines 138–144](calibration/reference.py#L138-L144)

<!-- source: calibration/reference.py:138:144 -->
```python
def initial_reference(cal_params):
    """Not ready: hosts keep their native weighting until the first refresh."""
    inf = jnp.asarray(jnp.inf, jnp.float32)
    return FrozenReference(
        cal_params, jnp.asarray(1.0, jnp.float32), inf, inf, inf,
        jnp.asarray(0.0, jnp.float32), jnp.asarray(False),
    )
```

IQL's fitter is created by `make_fitters`; its initialized host actor is stacked
into the paired state. ReBRAC's scale network has training-pool normalization
statistics even when the host's observation normalization is disabled. These
statistics belong to the calibrator; they do not silently normalize the host.

## How the host feeds scale fitting

This direction is **host → BCA**. Fitting reads detached host predictions and
targets, updates only scale parameters, and never differentiates the fitting
loss into the actor or critics. Three kinds of weights stay separate: the
Bayesian bootstrap masses of this scale fit, the WBCP posterior masses drawn
inside a refresh, and the consumed host multipliers or actor weights. The scale
fit has no importance factors, tempering search or effective-sample-size
acceptance gate.

**TD3+BC residual.** The target uses target-policy smoothing with a folded
fitting key; Q is the online twin minimum on recorded actions. The fitting masses
are one Bayesian bootstrap draw per minibatch, from a separate folded key.

[algorithms/td3_bc_bca.py, lines 162–177](algorithms/td3_bc_bca.py#L162-L177)

<!-- source: algorithms/td3_bc_bca.py:162:177 -->
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

**ReBRAC residual.** Its fitting target retains ReBRAC's recorded next-action
critic BC penalty inside `native_target`.

[algorithms/rebrac_bca.py, lines 287–297](algorithms/rebrac_bca.py#L287-L297)

<!-- source: algorithms/rebrac_bca.py:287:297 -->
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

**CQL residual.** The target retains the configured CQL backup; Q is the online
twin minimum. The masses weight only the scale objective, not actor or Bellman
losses.

[algorithms/cql_bca.py, lines 170–186](algorithms/cql_bca.py#L170-L186)

<!-- source: algorithms/cql_bca.py:170:186 -->
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
weights = jax.lax.stop_gradient(
    bayesian_bootstrap_weights(jax.random.fold_in(rng, 1128352850), len(batch.obs))
)
```

**IQL residual.** The target uses pre-update V; the scale reads the post-update
Polyak target Q, the copy IQL's actor advantage reads (its weights are
exp(beta * (min Qbar - V))), not the online heads. The adapter explicitly detaches both. The fitter divides both by the
current-batch residual spread and draws one Bayesian bootstrap simplex per
minibatch; the native actor still uses its original AWR weights.

[calibration/iql_targets.py, lines 24–39](calibration/iql_targets.py#L24-L39)

<!-- source: calibration/iql_targets.py:24:39 -->
```python
def calibration_inputs(self, state, batch, rng, step, host=None):
    if host is None or "target" not in host:
        raise ValueError("Supply the host pre-update V target.")
    heads = self.q_apply_fn(state.qf_target.params, batch.obs, batch.action)
    return CalibrationInputs(
        targets=jax.lax.stop_gradient(host["target"]),
        q_at_data=jax.lax.stop_gradient(jnp.min(heads, axis=-1)),
        q_heads=jax.lax.stop_gradient(heads),
        obs=batch.obs,
        action=batch.action,
        vintage={
            "targets": "pre_update_value_net",
            "q_at_data": "post_update_target",
            "q_heads": "post_update_target",
        },
    )
```

[calibration/iql_scale.py, lines 40–53](calibration/iql_scale.py#L40-L53)

<!-- source: calibration/iql_scale.py:40:53 -->
```python
unit_now = jax.lax.stop_gradient(
    jnp.std(inputs.targets - inputs.q_at_data) + BF._EPS
)
weights = bootstrap_masses(key, len(batch.reward))
(loss, (eta_mean, cov, cov_var, width)), grad = jax.value_and_grad(
    self.loss_fn, has_aux=True
)(
    state.calibrator.params,
    inputs.targets / unit_now,
    inputs.q_at_data / unit_now,
    weights,
    batch.obs,
    batch.action,
)
```

**Scale optimizer boundary.** TD3+BC's objective below is representative of
TD3+BC/ReBRAC/CQL's weighted soft-coverage plus width objectives. IQL retains its
own coverage-balance term and batch residual-unit convention; it is not silently
replaced by this loss. See the complete pseudocode for those differences.

[algorithms/td3_bc_bca.py, lines 180–198](algorithms/td3_bc_bca.py#L180-L198)

<!-- source: algorithms/td3_bc_bca.py:180:198 -->
```python
def objective(params):
    eta = models[2].apply(params, batch.obs, batch.action)
    if eta.shape != (len(batch.obs),):
        raise ValueError("scale predictor must emit one positive value per row")
    coverage = soft_coverage(target / unit, q / unit, eta, config.cal_beta)
    width = jnp.sum(prior * jnp.square(eta))
    return (
        jnp.square(jnp.sum(prior * coverage) - (1.0 - config.posterior.alpha))
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
| TD3+BC fitting target/optimizer | [`native_target`](algorithms/td3_bc_bca.py#L146), [`fit_scale`](algorithms/td3_bc_bca.py#L159) |
| ReBRAC fitting target/optimizer | [`native_target`](algorithms/rebrac_bca.py#L268), [`fit_scale`](algorithms/rebrac_bca.py#L283) |
| CQL fitting target/optimizer | [`native_target`](algorithms/cql_bca.py#L135), [`fit_scale`](algorithms/cql_bca.py#L169) |
| Bayesian bootstrap masses | [`bayesian_bootstrap_weights`](calibration/network.py#L44), [`bootstrap_masses`](calibration/iql_scale.py#L7) |
| IQL fitting target | [`CorlIQLAdapter`](calibration/iql_targets.py#L11) |
| IQL scale optimizer | [`BootstrapScaleFitter`](calibration/iql_scale.py#L14), [`BootstrapScaleFitter.update`](calibration/iql_scale.py#L36) |
| IQL scale objective | [`make_cal_loss_fn`](calibration/iql_network.py#L52) |

## Refresh, frozen width and gradients

**Reference boundary.** Each host refresh scores the held-out bank with its
*current* host state and freezes the *current* scale parameters and unit together
with the selected threshold. These are all four calls into `freeze_reference`.
Each runs eagerly between jitted scan blocks and passes no calibration weights.
TD3+BC, ReBRAC and CQL fold their target-noise and posterior-draw keys from a
per-refresh key; IQL folds its posterior-draw key from the current training key
without consuming it.

TD3+BC scores its smoothed target-policy backup against the online twin minimum
at the recorded held-out actions. An invalid refresh keeps the previous state and
reports `posterior_inputs_valid=False`, which the runtime treats as a failure:

[algorithms/td3_bc_bca.py, lines 244–272](algorithms/td3_bc_bca.py#L244-L272)

<!-- source: algorithms/td3_bc_bca.py:244:272 -->
```python
target = native_target(
    args,
    models,
    state.native,
    heldout,
    jax.random.fold_in(rng, 1213156420),
    max_action,
)
q = (
    models[1]
    .apply(state.native.critic.params, heldout.obs, heldout.action)
    .min(axis=-1)
)
cal = models[2].apply(state.calibrator.params, heldout.obs, heldout.action)
stored = (
    tree_finite((state.native, state.calibrator, heldout, target, q))
    & reference_valid(state.posterior)
    & jnp.all(jnp.isfinite(cal) & (cal > 0))
)
reference, frozen, diagnostics = freeze_reference(
    state.calibrator.params,
    state.residual_scale,
    cal,
    target - q,
    jax.random.fold_in(rng, 1347375956),
    config.posterior,
)
valid = bool(stored) and frozen
result = state._replace(posterior=reference) if valid else state
```

ReBRAC's refresh target keeps the critic BC penalty on the recorded next action;
the refresh also checks the fixed calibrator statistics and the new reference's
storage:

[algorithms/rebrac_bca.py, lines 351–371](algorithms/rebrac_bca.py#L351-L371)

<!-- source: algorithms/rebrac_bca.py:351:371 -->
```python
target = native_target(
    args, models, state.native, heldout, jax.random.fold_in(rng, KEY_REFRESH_NOISE)
)
q = models[1].apply(state.native.critic.params, heldout.obs, heldout.action).min(0)
cal = models[2].apply(state.calibrator.params, heldout.obs, heldout.action)
valid = (
    valid
    & tree_finite((state.native, target, q))
    & calibration_storage_valid(models, state)
    & jnp.all(jnp.isfinite(cal) & (cal > 0))
)
reference, frozen, diagnostics = freeze_reference(
    state.calibrator.params,
    state.residual_scale,
    cal,
    target - q,
    jax.random.fold_in(rng, KEY_POSTERIOR),
    config.posterior,
)
valid = bool(valid & _reference_storage_valid(reference)) and frozen
result = state._replace(posterior=reference) if valid else state
```

CQL scores its configured sampled target-policy backup:

[algorithms/cql_bca.py, lines 232–256](algorithms/cql_bca.py#L232-L256)

<!-- source: algorithms/cql_bca.py:232:256 -->
```python
target = native_target(
    args,
    models,
    state.native,
    heldout,
    jax.random.fold_in(rng, 1213156420),
    max_action,
)
q = jnp.minimum(
    models[1].apply(state.native.critic1.params, heldout.obs, heldout.action),
    models[2].apply(state.native.critic2.params, heldout.obs, heldout.action),
)
cal_predictions = models[3].apply(
    state.calibrator.params, heldout.obs, heldout.action
)
reference, valid, diagnostics = freeze_reference(
    state.calibrator.params,
    state.residual_scale,
    cal_predictions,
    target - q,
    jax.random.fold_in(rng, 1347375956),
    config.posterior,
)
valid = bool(valid) and bool(tree_finite(state.calibrator))
result = state._replace(posterior=reference) if valid else state
```

IQL scores `r + (1-d)*gamma*V(s')` with the current V against the current target
twin minimum (the Polyak copy IQL's actor advantage reads), using the live
calibrator's predictions. It raises instead of
returning a validity flag:

[calibration/iql_reference.py, lines 193–218](calibration/iql_reference.py#L193-L218)

<!-- source: calibration/iql_reference.py:193:218 -->
```python
key = jax.random.fold_in(training_key, POSTERIOR_FOLD)
pred_cal = jax.lax.stop_gradient(
    fitter.predictions(cal_state, cal_state.calibrator.params, cal)
)
target = cal.reward + (1.0 - cal.done) * discount * agent_state.vf.apply_fn(
    agent_state.vf.params, cal.next_obs
)
q = jnp.min(
    agent_state.qf_target.apply_fn(agent_state.qf_target.params, cal.obs, cal.action),
    axis=-1,
)
residual = jax.lax.stop_gradient(target - q)
state, valid, wbcp = POST.freeze_reference(
    cal_state.calibrator.params,
    cal_state.resid_scale,
    pred_cal,
    residual,
    key,
    args.config(),
)
if (
    not valid
    or not bool(POST.reference_valid(state))
    or not bool(finite_tree(cal_state))
):
    raise FloatingPointError("invalid frozen WBCP reference")
```

**Score boundary.** `freeze_reference` divides absolute residuals by the positive
frozen scale `max(eta, 1e-6)*u`, rejects nonfinite or nonpositive inputs without
producing a threshold, and calls WBCP without weights. It stores the threshold
and both of its components as float32, and refuses a finite threshold that would
overflow when stored.

[calibration/reference.py, lines 163–187](calibration/reference.py#L163-L187)

<!-- source: calibration/reference.py:163:187 -->
```python
scale = np.asarray(positive_scale(predictions, residual_scale), np.float64)
residuals = np.asarray(residuals, np.float64)
valid = bool(
    tree_valid(cal_params)
    and np.isfinite(float(residual_scale)) and float(residual_scale) > 0
    and predictions.size > 0
    and np.all(np.isfinite(scale) & (scale > 0))
    and np.all(np.isfinite(residuals))
)
if not valid:
    return initial_reference(cal_params), False, dict(certified=False)
scores = np.abs(residuals) / scale
result = wbcp.calibrate(
    scores, np.random.default_rng(np.asarray(key, np.uint32).ravel()),
    alpha=config.alpha, beta=config.credibility, draws=config.draws,
)
with np.errstate(over="ignore"):
    stored = np.float32(result.threshold)
if result.certified and not np.isfinite(stored):  # a finite threshold must stay finite when stored
    return initial_reference(cal_params), False, dict(certified=False)
as32 = lambda value: jnp.asarray(value, jnp.float32)
reference = FrozenReference(
    cal_params, residual_scale.astype(jnp.float32), as32(result.threshold),
    as32(result.lambda_hat), as32(result.lambda_hpd), as32(result.n_eff), jnp.asarray(True),
)
```

**Threshold boundary.** Omitting weights is the exchangeable case: unit weights
and a test mass of one.

[calibration/wbcp.py, lines 66–69](calibration/wbcp.py#L66-L69)

<!-- source: calibration/wbcp.py:66:69 -->
```python
if weights is None:
    if test_mass is not None:
        raise ValueError("test_mass requires explicit calibration weights")
    weights, test_mass = np.ones_like(scores), 1.0
```

Each posterior draw puts exponential masses on the sorted scores plus one test
atom and crosses at the first score whose cumulative mass reaches `1-alpha` of the
total. A draw whose test atom alone exceeds alpha never crosses (`+inf`).

[calibration/wbcp.py, lines 51–54](calibration/wbcp.py#L51-L54)

<!-- source: calibration/wbcp.py:51:54 -->
```python
mass = exponentials[:, :-1] * sorted_weights
total = mass.sum(axis=1) + exponentials[:, -1] * test_mass
covered = np.cumsum(mass, axis=1) >= (1.0 - alpha) * total[:, None]
return np.where(covered.any(axis=1), sorted_scores[covered.argmax(axis=1)], np.inf)
```

The consumed threshold is the `ceil(beta*M)`-th smallest crossing, clamped from
below by the empirical selection `lambda_hat`. A threshold of `+inf` means
nothing is certifiable.

[calibration/wbcp.py, lines 113–117](calibration/wbcp.py#L113-L117)

<!-- source: calibration/wbcp.py:113:117 -->
```python
# Eq. (7): Pr(L+ <= alpha) at lambda is the fraction of draws crossing at or
# below lambda, so the smallest beta-credible grid point is an order statistic.
lambda_hpd = float(posterior[math.ceil(Fraction(str(beta)) * int(draws)) - 1])
n_eff = float(weights.sum() ** 2 / np.dot(weights, weights))  # weights rescaled above
return Calibration(max(lambda_hat, lambda_hpd), lambda_hat, lambda_hpd, posterior, n_eff)
```

**Width and warmup boundary.** For TD3+BC/ReBRAC/CQL, the width is the frozen
threshold times the frozen scale and unit at the recorded rows. Before readiness
it is zero, yielding the unit host multiplier. An infinite width (threshold
`+inf`) is handled by the support mask, which gives that row the native
multiplier 1; it is not inserted into a loss.

[calibration/dose.py, lines 65–76](calibration/dose.py#L65-L76)

<!-- source: calibration/dose.py:65:76 -->
```python
def frozen_level_dose(reference, predictions, blend):
    """Width = WBCP threshold x frozen scale at the recorded rows; not ready means width 0.

    A threshold of +inf (nothing certifiable) gives infinite widths, which
    level_critic_dose treats as unsupported rows with the native multiplier 1.
    """
    predictions = jnp.asarray(predictions)
    if predictions.ndim != 1 or reference.threshold.ndim != 0 or reference.residual_scale.ndim != 0:
        raise ValueError("level dose requires vector predictions, one threshold and a scalar unit")
    threshold, unit = reference.threshold, reference.residual_scale
    scale = jnp.maximum(predictions, 1e-06) * unit
    width = threshold * scale
```

[calibration/dose.py, lines 90–95](calibration/dose.py#L90-L95)

<!-- source: calibration/dose.py:90:95 -->
```python
width = jnp.where(reference.ready, width, jnp.zeros_like(width))
valid = jnp.where(reference.ready, valid, True)
unit = jnp.where(reference.ready, unit, 1.0)
return level_critic_dose(
    width, jnp.ones_like(width, dtype=bool), unit, blend, valid
)
```

IQL uses the parallel `posterior_width`. A `+inf` threshold leaves every row
unsupported, so its BCA actor weight is zero and the actor abstains; before
readiness the width is zero with full support, which reproduces the native
weights, and `arm_weights` also selects the native weights explicitly.

[calibration/iql_reference.py, lines 157–171](calibration/iql_reference.py#L157-L171)

<!-- source: calibration/iql_reference.py:157:171 -->
```python
def posterior_width(reference, predictions):
    """Width = WBCP threshold x frozen positive scale; a +inf threshold leaves no support."""
    threshold, unit = reference.threshold, reference.residual_scale
    scale = POST.positive_scale(predictions, unit)
    valid = (
        POST.reference_valid(reference)
        & jnp.all(jnp.isfinite(predictions) & (predictions > 0))
        & jnp.all(jnp.isfinite(scale) & (scale > 0))
    )
    width = threshold * scale
    valid = valid & (
        jnp.isposinf(threshold)
        | jnp.all(jnp.isfinite(width) & ((threshold == 0) | (width > 0)))
    )
    return (width, jnp.isfinite(width) & valid, valid)
```

[calibration/iql_reference.py, lines 174–181](calibration/iql_reference.py#L174-L181)

<!-- source: calibration/iql_reference.py:174:181 -->
```python
def weights_at_reference(args, posterior, predictions, advantage, beta, cap):
    width, support, valid = posterior_width(posterior, predictions)
    width = jnp.where(posterior.ready, width, 0.0)
    support = jnp.where(posterior.ready, support, True)
    valid = jnp.where(posterior.ready, valid, True)
    return gain_actor_weights(
        advantage, width, support, beta, cap, args.decision_gain, valid
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
| Frozen reference | [`WBCPConfig`](calibration/reference.py#L17), [`FrozenReference`](calibration/reference.py#L31), [`initial_reference`](calibration/reference.py#L138), [`positive_scale`](calibration/reference.py#L147), [`freeze_reference`](calibration/reference.py#L151), [`reference_valid`](calibration/reference.py#L200) |
| WBCP threshold | [`calibrate`](calibration/wbcp.py#L57), [`crossings`](calibration/wbcp.py#L43) |
| BC/CQL frozen width and detached dose | [`frozen_level_dose`](calibration/dose.py#L65), [`level_critic_dose`](calibration/dose.py#L23) |
| IQL frozen width/weight | [`posterior_width`](calibration/iql_reference.py#L157), [`weights_at_reference`](calibration/iql_reference.py#L174) |
| TD3+BC refresh | [`refresh`](algorithms/td3_bc_bca.py#L234) |
| ReBRAC refresh | [`refresh`](algorithms/rebrac_bca.py#L340) |
| CQL refresh | [`refresh`](algorithms/cql_bca.py#L220) |
| IQL refresh | [`refresh`](calibration/iql_reference.py#L184) |

## Acceptance, scheduling and evaluation

**Scale-fit acceptance is numerical only.** There is no feasibility or ESS gate:
a numerically valid proposal is always committed, so `scale_fit_accepted` equals
`scale_inputs_valid`. A numerically invalid proposal leaves the live calibrator
and unit unchanged and marks the step invalid; the runtimes stop on that flag
rather than counting a silent skip. A previously frozen reference is never
changed by fitting. Here are the scale-acceptance boundaries for the common
pattern and IQL:

[algorithms/td3_bc_bca.py, lines 222–231](algorithms/td3_bc_bca.py#L222-L231)

<!-- source: algorithms/td3_bc_bca.py:222:231 -->
```python
result = jax.lax.cond(
    valid,
    lambda _: state._replace(calibrator=proposed, residual_scale=unit_new),
    lambda _: state,
    None,
)
return (
    result,
    dict(scale_inputs_valid=valid, scale_fit_accepted=valid, scale_loss=loss),
)
```

[calibration/iql_scale.py, lines 90–91](calibration/iql_scale.py#L90-L91)

<!-- source: calibration/iql_scale.py:90:91 -->
```python
accepted = raw_valid & P.finite_tree(proposed) & (proposed.resid_scale > 0)
out = jax.lax.cond(accepted, lambda _: proposed, lambda _: state, operand=None)
```

**Whole-step numerical guards.** TD3+BC and CQL reject a proposed combined state
when their validity checks fail; ReBRAC also rolls back its returned RNG. The
checks include the stored reference, through `reference_valid` directly or
through the readout. The runtime then raises on invalid metrics instead of
silently counting success. IQL's actor masks/acceptance were shown above, and its
runtime checks all returned validity flags. These guards can stop progress and
therefore are part of the map.

[algorithms/td3_bc_bca.py, lines 312–324](algorithms/td3_bc_bca.py#L312-L324)

<!-- source: algorithms/td3_bc_bca.py:312:324 -->
```python
valid = tree_finite((batch, state.native, native, metrics))
if config.arm == "bca":
    valid = valid & diag["scale_inputs_valid"] & dose.inputs_valid
    metrics.update(
        bc_multiplier_mean=jnp.mean(multiplier),
        bc_support_fraction=jnp.mean(dose.support_mask.astype(jnp.float32)),
        posterior_ready=state.posterior.ready,
    )
result = jax.lax.cond(
    valid, lambda _: fitted._replace(native=native), lambda _: state, None
)
metrics.update(diag, inputs_valid=valid)
return (result, metrics)
```

[algorithms/rebrac_bca.py, lines 425–432](algorithms/rebrac_bca.py#L425-L432)

<!-- source: algorithms/rebrac_bca.py:425:432 -->
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

[algorithms/cql_bca.py, lines 293–300](algorithms/cql_bca.py#L293-L300)

<!-- source: algorithms/cql_bca.py:293:300 -->
```python
valid = (
    fit_diag["scale_inputs_valid"]
    & dose.inputs_valid
    & reference_valid(state.posterior)
    & tree_finite(native)
)
proposed = fitted._replace(native=native)
result = jax.lax.cond(valid, lambda _: proposed, lambda _: state, operand=None)
```

**Scheduling boundary.** Scale fitting is attempted each host update. Refreshes
occur after updates 10k, 15k, ..., 995k (198 in total); their new reference is
consumed on later updates. Because `freeze_reference` runs NumPy, each runner
calls `refresh` eagerly between jitted `lax.scan` blocks, and a refresh replaces
only the frozen reference in the state. TD3's actual call is shown here; the table
links every runner's corresponding entry.

[runtime/td3_bc.py, lines 1064–1079](runtime/td3_bc.py#L1064-L1079)

<!-- source: runtime/td3_bc.py:1064:1079 -->
```python
def refresh_at(event):
    nonlocal carry
    key = jax.random.fold_in(jax.random.PRNGKey(event.seed), event.step)
    frozen_identity = _tree_hash(carry[1])
    proposal, metrics, diagnostics = P.refresh(
        args,
        cfg,
        models,
        carry[1],
        prepared.heldout,
        key,
        prepared.max_action,
        heldout_ids=prepared.heldout_ids,
    )
    _accept(proposal, metrics)
    carry = (carry[0], proposal, carry[2])
```

| Component | Exact function entry points |
|---|---|
| TD3+BC refresh scheduler and validity | [`run_prepared`](runtime/td3_bc.py#L1029), [`_accept`](runtime/td3_bc.py#L1017) |
| ReBRAC refresh scheduler and validity | [`run_prepared`](runtime/rebrac.py#L769), [`_accept`](runtime/rebrac.py#L760) |
| CQL refresh scheduler and validity | [`run_prepared`](runtime/cql.py#L717), [`_accept`](runtime/cql.py#L705) |
| IQL reference scheduler and validity | [`run_schedule`](runtime/iql.py#L407), [`PairRuntime`](runtime/iql_pair.py#L224), [`PairRuntime.refresh`](runtime/iql_pair.py#L286) |

**Evaluation boundary.** Evaluation uses the resulting actor parameters. There
is no extra BCA action search, rejection filter, width penalty or calibration
network call in action selection. The paired IQL evaluator explicitly receives
each actor's parameters with common episode IDs:

[runtime/iql_pair.py, lines 387–394](runtime/iql_pair.py#L387-L394)

<!-- source: runtime/iql_pair.py:387:394 -->
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

For the other hosts, see [`evaluate_episodes`](runtime/td3_bc.py#L914) (TD3+BC and
ReBRAC) and [`evaluate`](runtime/cql.py#L569) (CQL). Deterministic actor/tanh-mean
semantics remain as declared.

**Saved state boundary.** Checkpoints save the calibrator parameters/optimizer,
the live residual unit, the frozen reference (its scale parameters, unit,
threshold, `lambda_hat`, `lambda_hpd`, `n_eff` and `ready`), host state and RNGs.
Each refresh record logs the WBCP diagnostics: threshold, both components,
posterior spread, `n_eff`, whether the clamp binds, whether the threshold is
certified, and the number of scores. The individual posterior draws are not
saved. Logging and checkpoint observers do not add learning objectives. Accepted
fitting counts, actor counts, reference refreshes and actual evaluations are
recorded separately.

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
and results keep their identities in the archived reproduction matrix. The
previous Bayesian/conformal-maximum radius, fitting importance weights with their
ESS gate, the training-reference subset and quantile groups are preserved only at
the local `bca-bayesmax-archive` tag. The map documents code behavior, not a proof
of improved returns, useful OOD-harm ranking or a conformal guarantee for the
adaptive training loop.

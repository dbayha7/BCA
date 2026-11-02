"""Q1-aligned variant of the pinned BCA auxiliary fitting and refresh.

Only the online-critic center changes: Q1 replaces min(Q1,Q2).  In particular,
the supplied, pinned ``pipeline.native_target`` still uses the smoothed target
actor, target twin-min, the original discount and the original bootstrap mask.
This module never updates actor, critic, targets, or their optimizer states.

The held-out calibration is about a one-step TD residual, not true Q^pi error.
The caller owns data splits, the fresh initialization, PRNG streams and budget.
"""
from types import SimpleNamespace


def fresh_auxiliary(context, saved_agent, aux_seed):
    """Make a new scale network and optimizer; never recycle the min-Q fit.

    The supplied context must come from the verified central-quartile runtime,
    with ``data.normalization_fixture`` from its common retained intersection.
    Native model states are passed through unchanged as read-only views.
    """
    norm = context.training._transitions(
        context.native, context.data.normalization_fixture,
        "common retained-intersection normalization", len(context.data.normalization_fixture.obs))
    return context.pipeline.initialize_pipeline(saved_agent, norm, context.args, aux_seed)


def fresh_auxiliary_exact_normalization(context, saved_agent, aux_seed, *, expected_mean, expected_std):
    """Initialize a new auxiliary on the authenticated checkpoint statistics.

    The ordinary bridge insists that a newly reduced fixture match the model
    statistics byte-for-byte. CPU and GPU reductions need not agree at the last
    bits, so the development runner first authenticates the saved checkpoint
    manifest and binds all four native apply functions to those exact stats.
    This function checks every binding against the caller-supplied manifest;
    authentication of that manifest remains the runner's responsibility.

    Apart from the source of the already-frozen normalizers, this is the
    pinned bridge's Calibrator architecture, seed domain, optimizer, state,
    reference initialization, and native model views. It fits no parameters.
    """
    p = context.pipeline
    jax, jnp, np = p.jax, p.jnp, p.np
    fixture = context.data.normalization_fixture
    if fixture.obs.ndim != 2 or fixture.action.ndim != 2 or len(fixture.obs) == 0:
        raise ValueError("Expected nonempty raw observation/action matrices")
    if len(fixture.obs) != len(fixture.action):
        raise ValueError("Observation/action row counts differ")
    if isinstance(aux_seed, bool) or not isinstance(aux_seed, int) or not 0 <= aux_seed < 2**32:
        raise ValueError("Auxiliary initialization seed must be a uint32-range integer")
    mean = np.asarray(expected_mean, dtype=np.float32)
    std = np.asarray(expected_std, dtype=np.float32)
    if (mean.shape != (fixture.obs.shape[1],) or std.shape != mean.shape
            or not np.all(np.isfinite(mean)) or not np.all(np.isfinite(std))
            or np.any(std < 0)):
        raise ValueError("Invalid authenticated normalization")

    native_states = (saved_agent.actor, saved_agent.actor_target,
                     saved_agent.dual_q, saved_agent.dual_q_target)
    for ts in native_states:
        model = getattr(ts.apply_fn, "__self__", None)
        if model is None or not hasattr(model, "obs_mean") or not hasattr(model, "obs_std"):
            raise TypeError("Native apply methods must carry their observation statistics")
        if (not np.array_equal(np.asarray(model.obs_mean), mean)
                or not np.array_equal(np.asarray(model.obs_std), std)):
            raise ValueError("Native model statistics differ from the authenticated manifest")
    actor_model = saved_agent.actor.apply_fn.__self__
    if type(actor_model).__name__ != "DeterministicTanhActor":
        raise TypeError("Expected the pinned UniFloral DeterministicTanhActor")

    config = p.Config(arm="bca", posterior=p.WBCPConfig(), blend=1.0)
    args = SimpleNamespace(discount=float(context.args.gamma),
                           policy_noise=float(context.args.policy_noise),
                           noise_clip=float(context.args.noise_clip))
    if not np.isfinite(args.discount) or not 0 <= args.discount <= 1:
        raise ValueError("Invalid native discount")
    if any(not np.isfinite(x) or x < 0 for x in (args.policy_noise, args.noise_clip)):
        raise ValueError("Invalid native target-noise settings")
    cal = p.Calibrator(actor_model.obs_mean, actor_model.obs_std, state_dep=True)
    obs = jnp.zeros((1, fixture.obs.shape[1]), dtype=jnp.asarray(fixture.obs).dtype)
    action = jnp.zeros((1, fixture.action.shape[1]), dtype=jnp.asarray(fixture.action).dtype)
    key = jax.random.fold_in(jax.random.PRNGKey(aux_seed), 1128352841)
    cal_state = p.TrainState.create(apply_fn=cal.apply, params=cal.init(key, obs, action),
                                    tx=p.optax.adam(config.cal_lr))
    aux = p.State(p.native_view(saved_agent), cal_state, jnp.asarray(1.0, jnp.float32),
                  p.initial_reference(cal_state.params))
    # These adapter classes are part of the verified bridge; the native view
    # contains the bound original TrainStates and does not copy their params.
    models = (context.models[0].__class__(saved_agent.actor.apply_fn),
              context.models[1].__class__(saved_agent.dual_q.apply_fn), cal)
    return models, aux, config, args


def make_q1_pipeline(pipeline):
    """Bind the Q1 routines to the hash-verified original uncertainty pipeline.

    Interfaces intentionally match original ``fit_scale`` and ``refresh``;
    pass the fresh models/auxiliary returned by ``fresh_auxiliary``. Native
    target, bootstrap weights, soft coverage and WBCP freeze are unchanged.
    """
    jax, jnp, np = pipeline.jax, pipeline.jnp, pipeline.np

    def q1_and_target(args, models, native, batch, noise_key, max_action=1.0):
        if batch.obs.ndim != 2 or batch.action.ndim != 2:
            raise ValueError("observations and actions must be batch matrices")
        n = batch.obs.shape[0]
        if n == 0 or batch.action.shape[0] != n:
            raise ValueError("nonempty aligned minibatch required")
        # Preserve every native operator detail, including stochastic key, two
        # target heads, action clipping, discount, mask and reward convention.
        target = pipeline.native_target(args, models, native, batch, noise_key, max_action)
        heads = models[1].apply(native.critic.params, batch.obs, batch.action)
        if target.shape != (n,):
            raise ValueError("target must have exactly one scalar per row")
        if heads.shape != (n, 2):
            raise ValueError("online critic must return exactly [batch, 2] heads")
        # Both the actor-used online Q1 and target are constants for scale fit.
        return jax.lax.stop_gradient(heads[:, 0]), jax.lax.stop_gradient(target)

    def fit_scale(args, config, models, state, batch, rng, max_action=1.0):
        if config.arm != "bca":
            raise ValueError("only posterior arms fit a scale")
        q, target = q1_and_target(args, models, state.native, batch,
                                  jax.random.fold_in(rng, 1179210836), max_action)
        prior = jax.lax.stop_gradient(pipeline.bayesian_bootstrap_weights(
            jax.random.fold_in(rng, 1128352850), len(batch.obs)))
        unit = state.residual_scale

        def objective(params):
            eta = models[2].apply(params, batch.obs, batch.action)
            if eta.shape != (len(batch.obs),):
                raise ValueError("scale predictor must emit one scalar per row")
            coverage = pipeline.soft_coverage(target / unit, q / unit, eta, config.cal_beta)
            width = jnp.sum(prior * jnp.square(eta))
            return (jnp.square(jnp.sum(prior * coverage) - (1.0 - config.posterior.alpha))
                    + config.width_penalty * width, eta)

        (loss, eta), grad = jax.value_and_grad(objective, has_aux=True)(state.calibrator.params)
        proposed = state.calibrator.apply_gradients(grads=grad)
        unit_new = config.scale_ema * unit + (1.0 - config.scale_ema) * jnp.maximum(
            jnp.std(target - q), 1e-6)
        valid = (pipeline.tree_finite((batch, state.native, state.calibrator, target, q, prior,
                                       unit, loss, grad, proposed, unit_new))
                 & jnp.all(jnp.isfinite(eta) & (eta > 0))
                 & jnp.all(prior >= 0) & jnp.any(prior > 0)
                 & jnp.isclose(prior.sum(), 1.0, rtol=1e-5, atol=1e-6)
                 & (unit > 0) & (unit_new > 0))
        result = jax.lax.cond(valid,
            lambda _: state._replace(calibrator=proposed, residual_scale=unit_new),
            lambda _: state, None)
        return result, dict(scale_inputs_valid=valid, scale_fit_accepted=valid, scale_loss=loss)

    def refresh(args, config, models, state, heldout, rng, max_action=1.0, *, heldout_ids):
        if config.arm != "bca" or not len(heldout.obs):
            raise ValueError("posterior refresh requires a nonempty held-out reference")
        ids = np.asarray(heldout_ids)
        if (ids.ndim != 1 or len(ids) != len(heldout.obs)
                or not np.issubdtype(ids.dtype, np.integer) or np.any(ids < 0)
                or len(np.unique(ids)) != len(ids)):
            raise ValueError("held-out IDs must be unique aligned nonnegative integer row IDs")
        q, target = q1_and_target(args, models, state.native, heldout,
                                  jax.random.fold_in(rng, 1213156420), max_action)
        cal = models[2].apply(state.calibrator.params, heldout.obs, heldout.action)
        if cal.shape != (len(heldout.obs),):
            raise ValueError("scale predictor must emit one scalar per row")
        stored = (pipeline.tree_finite((state.native, state.calibrator, heldout, target, q))
                  & pipeline.reference_valid(state.posterior)
                  & jnp.all(jnp.isfinite(cal) & (cal > 0)))
        reference, frozen, diagnostics = pipeline.freeze_reference(
            state.calibrator.params, state.residual_scale, cal, target - q,
            jax.random.fold_in(rng, 1347375956), config.posterior)
        valid = bool(stored) and frozen
        result = state._replace(posterior=reference) if valid else state
        return result, dict(posterior_inputs_valid=valid,
                            posterior_certified=valid and bool(diagnostics["certified"])), diagnostics

    return SimpleNamespace(fit_scale=fit_scale, refresh=refresh, q1_and_target=q1_and_target,
                           native_target=pipeline.native_target, live_width=pipeline.live_width,
                           center="online_q1", fit="fresh_q1_symmetric_td_residual")

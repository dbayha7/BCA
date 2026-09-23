"""BCA iql shared.

Scientific definitions extracted from the recorded BCA source; see configs/sources.json.
"""

from dataclasses import dataclass, replace
from typing import NamedTuple
import _iql_agent as H
from _iql_agent import jax, jnp, P
from _bca_level import LevelActorWeights
import numpy as np


@dataclass(frozen=True)
class Arm:
    name: str
    mode: str
    beta: float
    gain: float


def tuning_arms(published_beta, multipliers=(0.25, 0.5, 1.0, 2.0), gains=(0.25, 0.5, 1.0, 2.0)):
    if not np.isfinite(published_beta) or published_beta <= 0:
        raise ValueError("positive finite base beta required")
    for choices in (multipliers, gains):
        if (
            not choices
            or len(set(choices)) != len(choices)
            or any((not np.isfinite(x) or x <= 0 for x in choices))
        ):
            raise ValueError("candidate choices must be distinct positive finite values")
    arms = [Arm(f"baseline_beta_{m:g}", "off", published_beta * m, 1.0) for m in multipliers]
    arms += [
        Arm(f"{mode}_gain_{g:g}", mode, published_beta, g)
        for mode in ("full", "floor")
        for g in gains
    ]
    for arm in arms:
        if not np.isfinite(np.float32(arm.beta)) or np.float32(arm.beta) <= 0:
            raise ValueError("candidate beta must be representable as positive float32")
        if not np.isfinite(np.float32(arm.gain)) or np.float32(arm.gain) <= 0:
            raise ValueError("candidate gain must be representable as positive float32")
    return tuple(arms)


class SharedCarry(NamedTuple):
    rng: object
    nuisance: object
    step: object
    extra: object
    actors: object


def stack_actors(actor, count):
    if count < 1:
        raise ValueError("at least one actor required")
    return jax.tree_util.tree_map(lambda x: jnp.stack([x] * count), actor)


def actor_at(actors, index):
    return jax.tree_util.tree_map(lambda x: x[index], actors)


def native_weights(advantage, beta):
    adv = jax.lax.stop_gradient(advantage)
    values = jnp.minimum(jnp.exp(beta * adv), H.EXP_ADV_MAX)
    valid = jnp.all(jnp.isfinite(adv)) & jnp.all(jnp.isfinite(values))
    return LevelActorWeights(values, adv, jnp.ones_like(adv, dtype=bool), valid, valid)


def arm_weights(args, arms, posterior, predictions, advantage):
    rows = []
    for arm in arms:
        native = native_weights(advantage, arm.beta)
        if arm.mode == "off":
            row = native
        else:
            config = replace(args.posterior, mode=arm.mode, decision_gain=arm.gain)
            row = P.weights_at_reference(
                config, posterior, predictions, advantage, arm.beta, H.EXP_ADV_MAX
            )
            row = jax.tree_util.tree_map(
                lambda new, old: jnp.where(posterior.ready, new, old), row, native
            )
        rows.append(row)
    return jax.tree_util.tree_map(lambda *xs: jnp.stack(xs), *rows)


def update_actors(args, actors, batch, advantage, dropout_key, weights):
    return jax.vmap(
        lambda actor, weight: H.actor_update(args, actor, batch, advantage, dropout_key, weight)
    )(actors, weights)


def make_shared_train_step(args, dataset, fitter, arms):
    args.posterior.validate()
    if args.posterior.mode != "full":
        raise ValueError("shared reference is always the full Bayesian/conformal posterior")
    if not arms or any((a.mode not in ("off", "full", "floor") for a in arms)):
        raise ValueError("explicit baseline/full/floor arms required")
    representative = next(
        (i for i, arm in enumerate(arms) if arm.mode == "off" and arm.beta == args.beta), None
    )
    if representative is None:
        raise ValueError("include the published-beta baseline as representative actor")
    n = len(dataset.reward)
    if n < 1:
        raise ValueError("nonempty training pool required")

    def step(carry, unused):
        del unused
        rng, state, it, extra, actors = carry
        it = it + 1
        rng, batch_key, dropout_key = jax.random.split(rng, 3)
        idx = jax.random.randint(batch_key, (args.batch_size,), 0, n)
        batch = jax.tree_util.tree_map(lambda x: x[idx], dataset)
        state, adv, target, loss = H.nuisance_update(
            args, state.qf.apply_fn, state.vf.apply_fn, state, batch
        )
        calibration, readout, cal_valid = fitter.update(
            extra.calibration, state, batch, target, adv, dropout_key, it
        )
        predictions = fitter.predictions(extra.calibration, extra.posterior.cal_params, batch)
        weights = arm_weights(args, arms, extra.posterior, predictions, adv)
        nuisance_valid = P.finite_tree((state.qf, state.qf_target, state.vf, loss))
        valid = cal_valid & nuisance_valid
        weights = weights._replace(
            inputs_valid=weights.inputs_valid & valid, has_support=weights.has_support & valid
        )
        actors, actor_loss, actor_valid = update_actors(
            args, actors, batch, adv, dropout_key, weights
        )
        state = state._replace(actor=actor_at(actors, representative))
        extra = H.PosteriorTrainState(calibration, extra.posterior)
        mass = jnp.sum(weights.weights, axis=1)
        diagnostics = {
            **loss,
            "inputs_valid": valid & jnp.all(weights.inputs_valid & actor_valid),
            "nuisance_valid": nuisance_valid,
            "calibration_valid": cal_valid,
            "actor_inputs_valid": weights.inputs_valid,
            "actor_proposal_valid": actor_valid,
            "actor_loss": actor_loss,
            "actor_updated": weights.has_support & actor_valid,
            "actor_step": actors.step,
            "weight_mean": jnp.mean(weights.weights, axis=1),
            "weight_ess_fraction": mass**2
            / (len(adv) * jnp.sum(weights.weights**2, axis=1) + 1e-30),
            "weight_cap_fraction": jnp.mean(
                (weights.weights >= H.EXP_ADV_MAX).astype(jnp.float32), axis=1
            ),
            "supported_fraction": jnp.mean(weights.support_mask.astype(jnp.float32), axis=1),
            "posterior_ready": extra.posterior.ready,
            "scale_loss": readout.diagnostics["cal_loss"],
            "scale_coverage": readout.coverage,
        }
        return (SharedCarry(rng, state, it, extra, actors), diagnostics)

    return step

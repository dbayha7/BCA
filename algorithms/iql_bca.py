"""IQL + BCA: two actors sharing Q/V; one AWR-weighted scale fitter."""

from dataclasses import asdict, dataclass, replace
from numbers import Real
from typing import NamedTuple
import numpy as np
import calibration.iql_state as H
import calibration.iql_actors as S
import calibration.iql_scale as W
from calibration.iql_state import P, jax, jnp


def _name(value):
    if not isinstance(value, str) or not value.strip() or value != value.strip():
        raise ValueError("names must be nonempty strings without outer whitespace")


def _positive(value, name):
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(name + " must be a positive finite float32 scalar")
    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        valid = np.isfinite(np.float32(value)) and np.float32(value) > 0
    if not valid:
        raise ValueError(name + " must be a positive finite float32 scalar")


@dataclass(frozen=True)
class ScaleVariant:
    name: str
    iw: W.ScaleIWConfig = W.ScaleIWConfig()
    weight_width: bool = False

    def __post_init__(self):
        _name(self.name)
        if not isinstance(self.iw, W.ScaleIWConfig):
            raise ValueError("iw must be a ScaleIWConfig")
        if not isinstance(self.weight_width, bool):
            raise ValueError("weight_width must be an explicit boolean")


@dataclass(frozen=True)
class IWArm:
    name: str
    variant_index: int
    mode: str
    beta: float
    gain: float = 1.0

    def __post_init__(self):
        _name(self.name)
        _positive(self.beta, "actor beta")
        _positive(self.gain, "actor gain")
        if (
            not isinstance(self.variant_index, int)
            or isinstance(self.variant_index, bool)
            or self.variant_index < -1
            or (self.mode not in ("off", "full"))
        ):
            raise ValueError("invalid variant index or actor mode")
        if (self.mode == "off") != (self.variant_index == -1):
            raise ValueError("only an off baseline uses variant_index=-1")
        if self.gain != 1.0:
            raise ValueError("baseline gain is unused and must be one")


class SharedIWCarry(NamedTuple):
    rng: object
    nuisance: object
    step: object
    extras: tuple
    actors: object


def default_design(published_beta):
    """The only paired design: shared Q/V, identical beta, fixed gain1, one scale fitter."""
    _positive(published_beta, "host beta")
    variant = ScaleVariant(
        "awr", W.ScaleIWConfig(mode="awr", beta=published_beta), True
    )
    actors = (
        IWArm("host", -1, "off", published_beta),
        IWArm("bca", 0, "full", published_beta),
    )
    return ((variant,), actors)


def _validate_variants(args, variants):
    args.posterior.validate()
    if (
        len(variants) != 1
        or variants[0].iw.mode != "awr"
        or not variants[0].weight_width
    ):
        raise ValueError("Exactly one AWR-weighted full BCA fitter is supported.")
    if args.posterior.mode != "full":
        raise ValueError(
            "shared references retain the full Bayesian/conformal posterior"
        )
    _positive(args.beta, "published beta")
    if not variants or any((not isinstance(v, ScaleVariant) for v in variants)):
        raise ValueError("explicit scale variants required")
    if len({v.name for v in variants}) != len(variants):
        raise ValueError("scale variant names must be unique")
    common = variants[0].iw
    if common.beta != args.beta or any((v.iw != common for v in variants)):
        raise ValueError(
            "all variants require the same fixed IW settings and published IW beta"
        )


def _validate_arms(args, arms, count):
    if (
        count != 1
        or len(arms) != 2
        or tuple(a.name for a in arms) != ("host", "bca")
        or any(a.beta != args.beta for a in arms)
    ):
        raise ValueError("Only the host/BCA pair with the same beta is supported.")
    if not arms or any((not isinstance(a, IWArm) for a in arms)):
        raise ValueError("explicit IWArm actors required")
    if len({a.name for a in arms}) != len(arms):
        raise ValueError("actor names must be unique")
    if any((a.variant_index >= count for a in arms)):
        raise ValueError("actor variant index is out of range")
    if {a.variant_index for a in arms if a.mode != "off"} != set(range(count)):
        raise ValueError("each scale variant must have a consuming actor")
    representative = next(
        (i for i, a in enumerate(arms) if a.mode == "off" and a.beta == args.beta), None
    )
    if representative is None:
        raise ValueError("include a published-beta same-pool baseline actor")
    return representative


def _validate_fitters(args, fitters, variants):
    _validate_variants(args, variants)
    if len(fitters) != len(variants):
        raise ValueError("one separately constructed fitter per scale variant required")
    if len({id(f) for f in fitters}) != len(fitters):
        raise ValueError("scale variants cannot reuse one fitter instance")
    for fitter, variant in zip(fitters, variants):
        if (
            not isinstance(fitter, W.IWScaleFitter)
            or fitter.iw != variant.iw
            or fitter.weight_width != variant.weight_width
            or (asdict(fitter.args) != asdict(args.posterior))
        ):
            raise ValueError(
                "fitter does not match its declared variant/posterior configuration"
            )


def make_fitters(args, state, obs_dim, action_dim, variants):
    _validate_variants(args, variants)
    return tuple(
        (
            W.IWScaleFitter(
                args.posterior,
                state.qf.apply_fn,
                obs_dim,
                action_dim,
                args.seed,
                iw=v.iw,
                weight_width=v.weight_width,
            )
            for v in variants
        )
    )


def initialize_shared(args, state, rng, fitters, arms):
    variants = tuple(
        (ScaleVariant(str(i), f.iw, f.weight_width) for i, f in enumerate(fitters))
    )
    _validate_fitters(args, fitters, variants)
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


def make_shared_train_step(args, dataset, fitters, variants, arms):
    _validate_fitters(args, fitters, variants)
    representative = _validate_arms(args, arms, len(variants))
    n = len(dataset.reward)
    if n < 1 or any((len(x) != n for x in dataset)):
        raise ValueError("aligned nonempty training pool required")
    if (
        not isinstance(args.batch_size, int)
        or isinstance(args.batch_size, bool)
        or args.batch_size < 1
    ):
        raise ValueError("positive integer batch size required")

    def step(carry, unused):
        del unused
        rng, state, it, extras, actors = carry
        if len(extras) != len(fitters) or actors.step.shape != (len(arms),):
            raise ValueError("carry does not match scale/actor dimensions")
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
        ready = jnp.stack([extra.posterior.ready for extra in extras])
        if ready.dtype != jnp.bool_ or ready.shape != (len(fitters),):
            raise ValueError("posterior readiness must be scalar boolean per variant")
        schedule_valid = jnp.all(ready == ready[0])
        new_extras, predictions, variant_logs = ([], [], [])
        for f, extra in zip(fitters, extras):
            calibration, readout, accepted = f.update(
                extra.calibration, state, batch, target, adv, dropout_key, it
            )
            d = readout.diagnostics
            numerical = (
                d.get("numerical_valid", accepted)
                & P.finite_tree(extra.calibration)
                & (extra.calibration.resid_scale > 0)
            )
            accepted = accepted & numerical & nuisance_valid
            calibration = jax.lax.cond(
                accepted, lambda _: calibration, lambda _: extra.calibration, None
            )
            new_extras.append(H.PosteriorTrainState(calibration, extra.posterior))
            predictions.append(
                f.predictions(extra.calibration, extra.posterior.cal_params, batch)
            )
            variant_logs.append(
                {
                    "accepted": accepted,
                    "numerical_valid": numerical,
                    "ess_feasible": d.get("iw_ess_feasible", jnp.asarray(True)),
                    "ess_infeasible": d.get("iw_ess_infeasible", jnp.asarray(False)),
                    "iw_inputs_valid": d.get("iw_inputs_valid", jnp.asarray(True)),
                    "has_support": d.get("iw_has_support", jnp.asarray(True)),
                    "step": calibration.calibrator.step,
                    "loss": d["cal_loss"],
                    "coverage": readout.coverage,
                    "width_loss": d["width_loss"],
                    "eta_mean": d["eta_mean"],
                    "resid_scale": calibration.resid_scale,
                    "iw_tau": d.get("iw_tau", jnp.asarray(1.0)),
                    "iw_raw_ess_fraction": d.get(
                        "iw_raw_ess_fraction", jnp.asarray(1.0)
                    ),
                    "iw_ess_fraction": d.get("iw_ess_fraction", jnp.asarray(1.0)),
                    "prior_ess_fraction": d.get(
                        "bootstrap_prior_ess_fraction", d["bootstrap_ess_fraction"]
                    ),
                    "product_ess_fraction": d["bootstrap_ess_fraction"],
                    "iw_supported_count": d.get(
                        "iw_supported_count", jnp.asarray(len(adv), jnp.int32)
                    ),
                    "iw_positive_count": d.get(
                        "iw_positive_count", jnp.asarray(len(adv), jnp.int32)
                    ),
                }
            )
        detail = jax.tree_util.tree_map(lambda *xs: jnp.stack(xs), *variant_logs)
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
        mass = jnp.sum(weights.weights, axis=1)
        calibration_valid = jnp.all(detail["numerical_valid"])
        diagnostics = {
            **loss,
            "inputs_valid": nuisance_valid
            & calibration_valid
            & schedule_valid
            & jnp.all(weights.inputs_valid & actor_valid),
            "nuisance_valid": nuisance_valid,
            "calibration_valid": calibration_valid,
            "posterior_schedule_valid": schedule_valid,
            "posterior_ready": jnp.all(ready),
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
            "supported_fraction": jnp.mean(
                weights.support_mask.astype(jnp.float32), axis=1
            ),
            "scale_loss": jnp.mean(detail["loss"]),
            "scale_coverage": jnp.mean(detail["coverage"]),
            **{"cal_variant_" + k: v for k, v in detail.items()},
        }
        return (SharedIWCarry(rng, state, it, tuple(new_extras), actors), diagnostics)

    return step

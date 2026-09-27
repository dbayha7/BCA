"""One IQL scale fitter: explicit uniform or AWR importance factors, same Bayesian masses."""

from dataclasses import dataclass
from numbers import Real
from typing import Literal, NamedTuple
import numpy as np
import calibration.weights as IW
import calibration.iql_reference as P
from calibration.iql_reference import BF, jax, jnp

ASSIGNMENT_FOLD = 1230458957


@dataclass(frozen=True)
class ScaleIWConfig:
    mode: Literal["off", "awr"] = "awr"
    beta: float = 3.0
    cap: float = 100.0
    mixing: float = 1.0
    ess_floor: float = 0.25
    tau_min: float = 0.05
    iterations: int = 32

    def __post_init__(self):
        if self.mode not in ("off", "awr"):
            raise ValueError("IQL fitting importance factors must be off or aligned AWR.")
        for name in ("beta", "cap", "mixing", "ess_floor", "tau_min"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, Real):
                raise ValueError(name + " must be a finite scalar")
            with np.errstate(over="ignore", under="ignore", invalid="ignore"):
                value32 = np.float32(value)
            if not np.isfinite(value32):
                raise ValueError(name + " must be finite and representable in float32")
            if name == "beta" and value32 <= 0:
                raise ValueError("beta must be positive")
            if name == "cap" and value < 1:
                raise ValueError("cap must be at least one")
            if name in ("mixing", "tau_min") and (not 0 <= value <= 1):
                raise ValueError(name + " must be in [0,1]")
            if name == "ess_floor" and (not (0 < value32 and value <= 1)):
                raise ValueError("ess_floor must be in (0,1]")
        if (
            not isinstance(self.iterations, int)
            or isinstance(self.iterations, bool)
            or self.iterations < 1
        ):
            raise ValueError("iterations must be a positive static integer")


class ScaleWeighting(NamedTuple):
    prior: jax.Array
    raw: IW.IWLogWeights
    iw: IW.IWWeights
    product: IW.BootstrapIW
    permutation: jax.Array


def scale_weighting(advantage, key, config):
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


class IWScaleFitter(P.ScaleFitter):

    def __init__(
        self,
        args,
        q_apply_fn,
        obs_dim,
        action_dim,
        seed,
        *,
        iw=ScaleIWConfig(),
        weight_width=False
    ):
        if not weight_width:
            raise ValueError("BCA uses the same fitting masses on coverage and width.")
        if not isinstance(iw, ScaleIWConfig):
            raise ValueError("iw must be a ScaleIWConfig")
        if not isinstance(weight_width, bool):
            raise ValueError("weight_width must be an explicit boolean")
        super().__init__(args, q_apply_fn, obs_dim, action_dim, seed)
        self.iw = iw
        self.weight_width = weight_width
        if weight_width:
            a = self.framework_args
            self.loss_fn = BF.make_cal_loss_fn(
                cal_apply_fn=self.initial.calibrator.apply_fn,
                cal_mode="soft",
                cal_beta=a.cal_beta,
                cal_alpha=a.cal_alpha,
                cal_lambda_width=a.cal_lambda_width,
                cal_cond_mode=a.cond_mode,
                balance_coverage=a.balance_coverage,
                balance_weight=a.balance_weight,
                weight_width=True,
            )

    def update(self, state, agent_state, batch, target, advantage, key, step):
        advantage = jnp.asarray(advantage)
        if advantage.shape != batch.reward.shape or advantage.ndim != 1:
            raise ValueError("advantage must be a vector aligned with batch.reward")
        inputs = self.adapter.calibration_inputs(
            agent_state, batch, key, step, host={"target": target, "adv": advantage}
        )
        unit_now = jax.lax.stop_gradient(
            jnp.std(inputs.targets - inputs.q_at_data) + BF._EPS
        )
        weighting = scale_weighting(advantage, key, self.iw)
        weights = weighting.product.weights
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
        eta = jax.lax.stop_gradient(
            self.predictions(state, state.calibrator.params, batch)
        )
        iw_valid = (
            weighting.raw.inputs_valid
            & weighting.iw.inputs_valid
            & weighting.product.inputs_valid
        )
        raw_valid = (
            P.finite_tree(
                (
                    state,
                    inputs.q_heads,
                    inputs.targets,
                    grad,
                    loss,
                    eta_mean,
                    cov,
                    cov_var,
                    width,
                    eta,
                )
            )
            & jnp.all(eta > 0)
            & jnp.isfinite(unit_now)
            & (unit_now > 0)
            & (state.resid_scale > 0)
            & iw_valid
        )
        cal = jax.lax.cond(
            raw_valid,
            lambda _: state.calibrator.apply_gradients(grads=grad),
            lambda _: state.calibrator,
            operand=None,
        )
        proposed = state._replace(
            calibrator=cal, resid_scale=0.99 * state.resid_scale + 0.01 * unit_now
        )
        numerical_valid = (
            raw_valid & P.finite_tree(proposed) & (proposed.resid_scale > 0)
        )
        has_support = (weighting.iw.supported_count > 0) & jnp.any(weights > 0)
        feasible = weighting.iw.ess_feasible
        accepted = numerical_valid & feasible & has_support
        out = jax.lax.cond(accepted, lambda _: proposed, lambda _: state, operand=None)
        # Preserve the original no-IW ESS arithmetic as well as its model updates.
        prior_ess = (1.0 / (len(batch.reward) * jnp.sum(weighting.prior**2))
                     if self.iw.mode == "off" else weighting.product.prior_ess_fraction)
        product_ess = (prior_ess if self.iw.mode == "off"
                       else weighting.product.product_ess_fraction)
        diagnostics = {
            "cal_loss": loss,
            "eta_mean": eta_mean,
            "cov": cov,
            "cov_var": cov_var,
            "width_loss": width,
            "resid_scale": state.resid_scale,
            "inputs_valid": accepted,
            "update_accepted": accepted,
            "numerical_valid": numerical_valid,
            "iw_inputs_valid": iw_valid,
            "iw_ess_feasible": feasible,
            "iw_ess_infeasible": weighting.iw.inputs_valid & ~feasible,
            "iw_has_support": has_support,
            "iw_supported_count": weighting.iw.supported_count,
            "iw_positive_count": weighting.iw.positive_count,
            "product_positive_count": jnp.sum(weights > 0),
            "iw_tau": weighting.iw.tau,
            "iw_raw_ess_fraction": weighting.iw.raw_ess_fraction,
            "iw_ess_fraction": weighting.iw.ess_fraction,
            "bootstrap_prior_ess_fraction": prior_ess,
            "bootstrap_ess_fraction": product_ess,
            "weight_width": jnp.asarray(self.weight_width),
            "iw_permuted": jnp.asarray(self.iw.mode == "awr_permuted"),
        }
        readout = BF.CalibratorReadout(
            eta,
            state.resid_scale,
            eta,
            cov,
            jnp.asarray(0.0),
            cov - (1.0 - self.args.alpha),
            jnp.asarray(1.0),
            jnp.asarray(1.0),
            diagnostics,
        )
        return (out, readout, accepted)

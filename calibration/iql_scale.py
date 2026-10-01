"""The IQL scale fitter: Bayesian bootstrap masses on coverage and width, no importance factors."""

import calibration.iql_reference as P
from calibration.iql_reference import BF, jax, jnp


def bootstrap_masses(key, n):
    """One Dirichlet(1,...,1) simplex draw per minibatch, from a fold of the step key."""
    return jax.lax.stop_gradient(
        BF.bayesian_bootstrap_weights(jax.random.fold_in(key, BF._FOLD_BOOTSTRAP), n)
    )


class BootstrapScaleFitter(P.ScaleFitter):

    def __init__(self, args, q_apply_fn, obs_dim, action_dim, seed, *, weight_width=False):
        if not isinstance(weight_width, bool):
            raise ValueError("weight_width must be an explicit boolean")
        if not weight_width:
            raise ValueError("BCA uses the same fitting masses on coverage and width.")
        super().__init__(args, q_apply_fn, obs_dim, action_dim, seed)
        self.weight_width = weight_width
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

    def update(self, state, agent_state, batch, target, key, step):
        inputs = self.adapter.calibration_inputs(
            agent_state, batch, key, step, host={"target": target}
        )
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
        eta = jax.lax.stop_gradient(
            self.predictions(state, state.calibrator.params, batch)
        )
        masses_valid = jnp.all(jnp.isfinite(weights) & (weights >= 0)) & jnp.any(
            weights > 0
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
            & masses_valid
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
        accepted = raw_valid & P.finite_tree(proposed) & (proposed.resid_scale > 0)
        out = jax.lax.cond(accepted, lambda _: proposed, lambda _: state, operand=None)
        diagnostics = {
            "cal_loss": loss,
            "eta_mean": eta_mean,
            "cov": cov,
            "cov_var": cov_var,
            "width_loss": width,
            "resid_scale": state.resid_scale,
            "inputs_valid": accepted,
            "update_accepted": accepted,
            "numerical_valid": accepted,
            "bootstrap_ess_fraction": 1.0 / (len(batch.reward) * jnp.sum(weights**2)),
            "weight_width": jnp.asarray(self.weight_width),
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

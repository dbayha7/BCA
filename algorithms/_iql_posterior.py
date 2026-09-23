"""BCA iql posterior.

Scientific definitions extracted from the recorded BCA source; see configs/sources.json.
"""

from dataclasses import dataclass
from decimal import Decimal
import hashlib
from typing import Literal, NamedTuple
import _bca_host as H
from _bca_host import BF, C, jax, jnp
import numpy as np
import _iql_adapter as BA
import _bca_iql_adapter as POST
from _bca_posterior import PosteriorConfig, _rank
from _bca_level import LevelActorWeights, postcap_level_actor_weights

POSTERIOR_FOLD = 1347375956


@dataclass
class PosteriorArgs:
    mode: Literal["off", "full", "floor"] = "off"
    reserve_size: int = 0
    reserve_seed: int = 911
    reserve_max_fraction: float = 0.25
    alpha: float = 0.1
    credibility: float = 0.95
    draws: int = 128
    warmup: int = 10000
    refresh_interval: int = 5000
    fit_size: int = 4096
    decision_gain: float = 1.0
    cond_mode: Literal["scalar", "state", "state_action"] = "state_action"
    use_bootstrap: bool = True
    cal_beta: float = 20.0
    cal_lambda_width: float = 0.005
    cal_lr: float = 0.001
    cal_hidden_dim: int = 64
    balance_coverage: bool = True
    balance_weight: float = 0.5

    def validate(self, calibration_size=None):
        if self.mode not in ("off", "full", "floor"):
            raise ValueError("mode must be off/full/floor")
        if self.reserve_size < 0 or not 0 < self.reserve_max_fraction < 1:
            raise ValueError("invalid reservation configuration")
        if self.mode == "off":
            return
        PosteriorConfig(self.alpha, self.credibility, self.draws)
        if not isinstance(self.draws, int):
            raise ValueError("posterior draws must be an integer")
        if self.reserve_size <= 0 or not self.use_bootstrap:
            raise ValueError(
                "full/floor require a held-out bank and Bayesian bootstrap scale fitting"
            )
        gain = np.float32(self.decision_gain)
        if not np.isfinite(gain) or gain <= 0:
            raise ValueError(
                "decision_gain must be positive finite and representable in float32"
            )
        if self.warmup < 0 or self.refresh_interval < 1 or self.fit_size < 1:
            raise ValueError("invalid posterior schedule or fitting size")
        if self.cond_mode not in ("scalar", "state", "state_action") or self.cal_hidden_dim < 1:
            raise ValueError("invalid scale conditioning")
        for name in ("cal_beta", "cal_lr"):
            if not np.isfinite(getattr(self, name)) or getattr(self, name) <= 0:
                raise ValueError(name + " must be positive finite")
        for name in ("cal_lambda_width", "balance_weight"):
            if not np.isfinite(getattr(self, name)) or getattr(self, name) < 0:
                raise ValueError(name + " must be nonnegative finite")
        if calibration_size is not None:
            n = int(calibration_size)
            if n < 1 or _rank(n + 1, Decimal(1) - Decimal(str(self.alpha))) > n:
                raise ValueError(
                    "global calibration bank cannot provide a finite rank at configured alpha"
                )


def finite_tree(tree):
    leaves = jax.tree_util.tree_leaves(tree)
    return (
        jnp.all(jnp.stack([jnp.all(jnp.isfinite(x)) for x in leaves]))
        if leaves
        else jnp.asarray(True)
    )


def fingerprint(tree):
    leaves, structure = jax.tree_util.tree_flatten(tree)
    h = hashlib.sha256(str(structure).encode())
    for leaf in leaves:
        a = np.asarray(leaf)
        h.update(str((a.shape, a.dtype.str)).encode())
        h.update(a.tobytes())
    return h.hexdigest()


def gain_actor_weights(adv, width, usable, beta, cap, gain=1.0, inputs_valid=True):
    if not np.isfinite(np.float32(gain)) or np.float32(gain) <= 0:
        raise ValueError("decision gain must be positive finite float32")
    if gain == 1.0:
        return postcap_level_actor_weights(adv, width, usable, beta, cap, inputs_valid)
    check = postcap_level_actor_weights(adv, width, usable, beta, cap, inputs_valid)
    dtype = check.weights.dtype
    adv, width, beta = (
        jnp.asarray(adv, dtype),
        jnp.asarray(width, dtype),
        jnp.asarray(beta, dtype),
    )
    a = jnp.where(check.support_mask, adv, 0.0)
    u = jnp.where(check.support_mask, width, 0.0)
    p = jnp.maximum(a, 0.0)
    fraction = jax.nn.sigmoid(
        jnp.log(jnp.where(p > 0, p, 1.0))
        - jnp.log(jnp.where(u > 0, u, 1.0))
        - jnp.log(jnp.asarray(gain, dtype))
    )
    fraction = jnp.where(u == 0, 1.0, fraction)
    fraction = jnp.where(p > 0, fraction, 0.0)
    native = postcap_level_actor_weights(
        adv, jnp.zeros_like(width), usable, beta, cap, inputs_valid
    )
    weight = jnp.minimum(
        jnp.maximum(1.0 + fraction * (native.weights - 1.0), 1.0), native.weights
    )
    weight = jnp.where(a > 0, weight, native.weights)
    weight = jnp.where(check.support_mask, weight, 0.0)
    effective = jnp.minimum(
        p, jnp.log1p(jnp.maximum(weight - 1.0, 0.0)) / jnp.where(beta > 0, beta, 1.0)
    )
    used = jnp.where(check.support_mask, jnp.where(a > 0, effective, a), 0.0)
    return jax.tree_util.tree_map(
        jax.lax.stop_gradient,
        LevelActorWeights(
            weight, used, check.support_mask, check.inputs_valid, check.has_support
        ),
    )


class ScaleFitter:

    def __init__(self, args, q_apply_fn, obs_dim, action_dim, seed):
        args.validate()
        self.args = args
        a = BF.BCAFrameworkArgs(
            use_bca=True,
            bca_channel="",
            bca_source="conformal",
            cond_mode=args.cond_mode,
            cal_alpha=args.alpha,
            cal_beta=args.cal_beta,
            cal_lambda_width=args.cal_lambda_width,
            cal_lr=args.cal_lr,
            cal_hidden_dim=args.cal_hidden_dim,
            balance_coverage=args.balance_coverage,
            balance_weight=args.balance_weight,
            use_bootstrap=True,
            num_heads=2,
            cal_iw=False,
            use_eat=False,
        )
        self.adapter = BA.CorlIQLAdapter(q_apply_fn=q_apply_fn, num_heads=2, iw_source=None)
        mean = jnp.zeros(obs_dim)
        std = jnp.full(obs_dim, 1.0 - 0.001)
        self.framework_args = a
        key = jax.random.fold_in(jax.random.PRNGKey(seed), H.FOLD_BCA_INIT)
        cal = BF.build_calibrator(key, a, mean, std, jnp.zeros(obs_dim), jnp.zeros(action_dim))
        self.initial = BF.BCAState(cal, jnp.asarray(1.0), jnp.asarray(0.0))
        self.loss_fn = BF.make_cal_loss_fn(
            cal_apply_fn=cal.apply_fn,
            cal_mode="soft",
            cal_beta=a.cal_beta,
            cal_alpha=a.cal_alpha,
            cal_lambda_width=a.cal_lambda_width,
            cal_cond_mode=a.cond_mode,
            balance_coverage=a.balance_coverage,
            balance_weight=a.balance_weight,
            weight_width=False,
        )

    def predictions(self, cal_state, params, batch):
        apply = cal_state.calibrator.apply_fn
        if self.args.cond_mode == "scalar":
            eta = apply(params)
        elif self.args.cond_mode == "state":
            eta = apply(params, batch.obs)
        else:
            eta = apply(params, batch.obs, batch.action)
        return jnp.broadcast_to(eta, batch.reward.shape)

    def update(self, state, agent_state, batch, target, advantage, key, step):
        inputs = self.adapter.calibration_inputs(
            agent_state, batch, key, step, host={"target": target, "adv": advantage}
        )
        unit_now = jax.lax.stop_gradient(jnp.std(inputs.targets - inputs.q_at_data) + BF._EPS)
        weights = jax.lax.stop_gradient(
            BF.bayesian_bootstrap_weights(
                jax.random.fold_in(key, BF._FOLD_BOOTSTRAP), len(batch.reward)
            )
        )
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
        eta = jax.lax.stop_gradient(self.predictions(state, state.calibrator.params, batch))
        raw_valid = (
            finite_tree(
                (inputs.q_heads, inputs.targets, grad, loss, eta_mean, cov, cov_var, width, eta)
            )
            & jnp.all(eta > 0)
            & jnp.isfinite(unit_now)
            & (unit_now > 0)
        )
        cal = jax.lax.cond(
            raw_valid,
            lambda _: state.calibrator.apply_gradients(grads=grad),
            lambda _: state.calibrator,
            None,
        )
        proposed = state._replace(
            calibrator=cal, resid_scale=0.99 * state.resid_scale + 0.01 * unit_now
        )
        valid = raw_valid & finite_tree(proposed) & (proposed.resid_scale > 0)
        out = jax.lax.cond(valid, lambda _: proposed, lambda _: state, operand=None)
        readout = BF.CalibratorReadout(
            eta,
            state.resid_scale,
            eta,
            cov,
            jnp.asarray(0.0),
            cov - (1.0 - self.args.alpha),
            jnp.asarray(1.0),
            jnp.asarray(1.0),
            {
                "cal_loss": loss,
                "eta_mean": eta_mean,
                "cov": cov,
                "cov_var": cov_var,
                "width_loss": width,
                "resid_scale": state.resid_scale,
                "inputs_valid": valid,
                "bootstrap_ess_fraction": 1.0 / (len(batch.reward) * jnp.sum(weights**2)),
            },
        )
        return (out, readout, valid)


def posterior_width(state, predictions, mode):
    if mode not in ("full", "floor", "bayes"):
        raise ValueError("unknown diagnostic component")
    radius = {
        "full": state.radii.radius,
        "floor": state.radii.conformal_radius,
        "bayes": state.radii.bayesian_radius,
    }[mode][0]
    unit = state.residual_scale
    scale = POST.positive_scale(predictions, unit)
    valid = (
        jnp.all(state.radii.inputs_valid)
        & finite_tree(state.cal_params)
        & jnp.all(jnp.isfinite(predictions) & (predictions > 0))
        & jnp.isfinite(unit)
        & (unit > 0)
        & jnp.all(jnp.isfinite(scale) & (scale > 0))
        & (radius >= 0)
        & ~jnp.isnan(radius)
    )
    width = radius * scale
    valid = valid & (
        jnp.isposinf(radius) | jnp.all(jnp.isfinite(width) & ((radius == 0) | (width > 0)))
    )
    return (width, jnp.isfinite(width) & valid, valid)


def weights_at_reference(args, posterior, predictions, advantage, beta, cap):
    width, support, valid = posterior_width(posterior, predictions, args.mode)
    width = jnp.where(posterior.ready, width, 0.0)
    support = jnp.where(posterior.ready, support, True)
    valid = jnp.where(posterior.ready, valid, True)
    return gain_actor_weights(advantage, width, support, beta, cap, args.decision_gain, valid)


def refresh(args, fitter, cal_state, agent_state, train, cal, training_key, step, *, discount):
    args.validate(len(cal.reward))
    key = jax.random.fold_in(training_key, POSTERIOR_FOLD)
    idx = np.linspace(
        0, len(train.reward) - 1, min(args.fit_size, len(train.reward)), dtype=np.int32
    )
    fit = jax.tree_util.tree_map(lambda x: x[idx], train)
    pred_fit = fitter.predictions(cal_state, cal_state.calibrator.params, fit)
    pred_cal = fitter.predictions(cal_state, cal_state.calibrator.params, cal)
    target = cal.reward + (1.0 - cal.done) * discount * agent_state.vf.apply_fn(
        agent_state.vf.params, cal.next_obs
    )
    q = jnp.min(agent_state.qf.apply_fn(agent_state.qf.params, cal.obs, cal.action), axis=-1)
    residual = target - q
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
    if not bool(jnp.all(state.radii.inputs_valid)) or not bool(finite_tree(cal_state)):
        raise FloatingPointError("invalid frozen posterior reference")
    components = {}
    for name, network in [
        ("actor", agent_state.actor),
        ("online_q", agent_state.qf),
        ("target_q", agent_state.qf_target),
        ("value", agent_state.vf),
        ("calibrator", cal_state.calibrator),
    ]:
        components[name + "_params"] = fingerprint(network.params)
        components[name + "_optimizer"] = fingerprint(network.opt_state)
    components.update(
        residual_unit=fingerprint(cal_state.resid_scale),
        posterior_key=fingerprint(key),
        posterior_quantile_draws=fingerprint(state.radii.posterior_quantiles),
        fit_indices=fingerprint(idx),
        cal_residuals=fingerprint(residual),
        cal_predictions=fingerprint(pred_cal),
        fit_predictions=fingerprint(pred_fit),
        radii=fingerprint(state.radii),
    )
    record = {
        "step": int(step),
        "posterior_key": np.asarray(key).tolist(),
        "training_key": np.asarray(training_key).tolist(),
        "component_sha256": components,
        "residual_unit": float(cal_state.resid_scale),
        "full_radius": float(state.radii.radius[0]),
        "bayesian_radius": float(state.radii.bayesian_radius[0]),
        "conformal_radius": float(state.radii.conformal_radius[0]),
        "calibration_rows": len(cal.reward),
        "fit_rows": len(idx),
        "decision_gain": args.decision_gain,
        "reference_vintage": "current online-Q min over twins and current V; frozen current scale/unit",
        "scale_training_vintage": "post-Q-update online twins versus pre-V-update target",
        "bootstrap_weights_saved": False,
        "posterior_quantile_draws_hashed": True,
    }
    return (state, record)


class PreparedData(NamedTuple):
    train: object
    calibration: object
    obs_mean: object
    obs_std: object
    train_indices: object
    calibration_indices: object
    episode_ids: object
    metadata: dict


def prepare_dataset(args, converted, raw):
    a = args.posterior
    a.validate()
    n = len(converted["rewards"])
    ids = POST.qlearning_episode_ids(raw)
    if ids is not None and len(ids) != n:
        raise ValueError("raw terminal/timeout IDs do not align with default D4RL converter")
    obs = np.asarray(converted["observations"], np.float32)
    nxt = np.asarray(converted["next_observations"], np.float32)
    done = np.asarray(converted["terminals"], np.float32)
    if ids is None:
        ends = done != 0
        ends[:-1] |= np.any(nxt[:-1] != obs[1:], axis=1)
        ids = np.r_[0, np.cumsum(ends[:-1])].astype(np.int64)
        boundary = "inferred terminal or exact observation discontinuity; no raw timeouts"
    else:
        boundary = "raw terminal/timeout IDs through default D4RL filtering"
    if a.reserve_size:
        train_idx, cal_idx, meta = POST.reserve_calibration(
            obs,
            nxt,
            done,
            a.reserve_size,
            a.reserve_seed,
            a.reserve_max_fraction,
            episode_ids=ids,
        )
        a.validate(len(cal_idx))
    else:
        train_idx = np.arange(n, dtype=np.int32)
        cal_idx = np.empty(0, np.int32)
        meta = {
            "training_size": n,
            "calibration_size": 0,
            "reserved_blocks": 0,
            "total_blocks": len(np.unique(ids)),
        }
    meta["boundary_rule"] = boundary
    rewards = np.asarray(converted["rewards"], np.float64).copy()
    reward_info = {"normalize_reward": args.normalize_reward, "return_range": None}
    if args.normalize_reward and any(
        (s in args.dataset for s in ("halfcheetah", "hopper", "walker2d"))
    ):
        if a.reserve_size:
            starts = np.r_[0, np.flatnonzero(np.diff(ids[train_idx])) + 1]
            block_returns = np.add.reduceat(rewards[train_idx], starts)
            lo, hi = (min(block_returns), max(block_returns))
            reward_info["range_rule"] = (
                "return sums within retained original transition blocks; includes partial blocks"
            )
        else:
            lo, hi = C.return_reward_range({"rewards": rewards, "terminals": done}, 1000)
            reward_info["range_rule"] = "native CORL full-data return_reward_range"
        if not np.isfinite(hi - lo) or hi <= lo:
            raise ValueError("training reward return range must be positive finite")
        rewards = rewards / (hi - lo) * 1000
        reward_info["return_range"] = [lo, hi]
    transformed = C.apply_reward_transform(
        {"rewards": rewards},
        args.dataset,
        args.reward_transform,
        args.reward_scale,
        args.reward_bias,
        legacy_print=False,
    )
    rewards = transformed["rewards"]
    if args.normalize:
        mean, std = C.compute_mean_std(obs[train_idx], eps=0.001)
        obs = C.normalize_states(obs, mean, std)
        nxt = C.normalize_states(nxt, mean, std)
    else:
        mean = np.zeros(obs.shape[1], np.float32)
        std = np.ones(obs.shape[1], np.float32)
    values = C.Transition(
        obs,
        np.asarray(converted["actions"], np.float32),
        np.asarray(rewards, np.float32),
        nxt,
        done,
    )
    if any((not np.all(np.isfinite(x)) for x in values)):
        raise ValueError("nonfinite dataset/preprocessing")

    def subset(index):
        return jax.tree_util.tree_map(lambda x: jnp.asarray(x[index]), values)

    meta.update(
        dataset_rows=n,
        training_blocks=len(np.unique(ids[train_idx])),
        calibration_blocks=len(np.unique(ids[cal_idx])),
        train_indices_sha256=fingerprint(train_idx),
        calibration_indices_sha256=fingerprint(cal_idx),
        episode_ids_sha256=fingerprint(ids),
        converted_numeric_data_sha256=fingerprint(
            tuple((np.asarray(converted[k]) for k in sorted(converted)))
        ),
        raw_terminal_timeout_sha256=fingerprint(
            {k: raw[k] for k in ("terminals", "timeouts") if k in raw}
        ),
        obs_mean=mean.tolist(),
        obs_std_with_epsilon=std.tolist(),
        normalization_fit="training rows only",
        scale_internal_normalization="identity after host normalization",
        reward=reward_info,
    )
    return PreparedData(
        subset(train_idx),
        subset(cal_idx),
        jnp.asarray(mean),
        jnp.asarray(std),
        train_idx,
        cal_idx,
        ids,
        meta,
    )

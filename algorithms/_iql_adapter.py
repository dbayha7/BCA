"""BCA iql adapter.

Scientific definitions extracted from the recorded BCA source; see configs/sources.json.
"""

import math
from dataclasses import dataclass
from typing import Callable
import jax
import jax.numpy as jnp
from _iql_calibration import (
    C3_PER_SAMPLE_TERM_REWEIGHT,
    C4_ADVANTAGE_PENALTY,
    CalibrationInputs,
    ChannelSite,
    iw_gauss_logp,
)

_LOG_SQRT_2PI = float(math.log(math.sqrt(2.0 * math.pi)))
_IW_EPS = 1e-12


def _implied_gauss_logw(mean, action):
    e = jnp.square(action - mean).sum(axis=-1)
    da = float(action.shape[-1])
    sigma = jnp.sqrt(jnp.mean(e) / da + _IW_EPS)
    return iw_gauss_logp(mean, action, sigma)


def _gauss_logw(mean, std, action):
    return (
        -jnp.square(action - mean) / (2.0 * jnp.square(std)) - jnp.log(std) - _LOG_SQRT_2PI
    ).sum(axis=-1)


def _clipped_exp_weight_logw(score, log_clip):
    return jnp.minimum(score, log_clip)


def _to_bk(q, head_axis):
    return q if head_axis == -1 else jnp.swapaxes(q, 0, 1)


def _min_agg(q_bk):
    return jnp.min(q_bk, axis=-1)


def _need(host, key, adapter, where):
    if host is None or key not in host:
        raise KeyError(
            f"{adapter}.calibration_inputs requires host[{key!r}], the host's own tensor from {where}. The adapter names the host's variable rather than recomputing it, so that the calibration VINTAGE is an explicit declaration."
        )
    return host[key]


def _sg(x):
    return jax.lax.stop_gradient(x)


@dataclass(frozen=True)
class _Base:
    AGGREGATOR = "min"
    SUPPORTED_LOSS_MODES = frozenset({"coverage"})
    IW_SOURCES = frozenset()
    IW_SOURCE_REFUSALS = {}
    SUPPLIES_IW_LOGW = False

    def __init_subclass__(cls, **kw):
        super().__init_subclass__(**kw)
        cls.SUPPLIES_IW_LOGW = bool(cls.IW_SOURCES)

    def __post_init__(self):
        src = getattr(self, "iw_source", None)
        if src is None:
            return
        if src not in self.IW_SOURCES:
            why = self.IW_SOURCE_REFUSALS.get(src)
            raise ValueError(
                f"{type(self).__name__} (host {self.HOST!r}) cannot build the {src!r} IW numerator; it declares IW_SOURCES={sorted(self.IW_SOURCES)}."
                + (f"\n  {why}" if why else "")
            )


@dataclass(frozen=True)
class CorlIQLAdapter(_Base):
    q_apply_fn: Callable
    num_heads: int = 2
    actor_apply_fn: Callable | None = None
    iql_deterministic: bool = False
    beta: float = 3.0
    exp_adv_max: float = 100.0
    iw_source: str | None = None
    HOST = "corl_iql"
    CHANNELS = frozenset({C4_ADVANTAGE_PENALTY, C3_PER_SAMPLE_TERM_REWEIGHT})
    IW_SOURCES = frozenset({"policy", "consumption"})

    def calibration_inputs(self, state, batch, rng, step, host=None) -> CalibrationInputs:
        targets = _need(host, "target", self.HOST, "corl_iql.py:443")
        q_heads = _to_bk(self.q_apply_fn(state.qf.params, batch.obs, batch.action), -1)
        vintage = {
            "targets": "pre_update_value_net",
            "q_at_data": "post_update_online",
            "q_heads": "post_update_online",
        }
        iw_logw = None
        if self.iw_source == "policy":
            out = self.actor_apply_fn(state.actor.params, batch.obs, deterministic=True)
            if self.iql_deterministic:
                iw_logw = _implied_gauss_logw(out, batch.action)
                vintage["iw_logw"] = "policy/implied_gaussian_batch_mle@pre_update_actor"
            else:
                mean, std = out
                iw_logw = _gauss_logw(mean, std, batch.action)
                vintage["iw_logw"] = "policy/host_gaussian_log_prob@pre_update_actor"
            iw_logw = _sg(iw_logw)
        elif self.iw_source == "consumption":
            adv = _need(
                host,
                "adv",
                self.HOST,
                "corl_iql.py:450 / corl_iql_bca.py:102, the PRE-PENALTY advantage",
            )
            iw_logw = _sg(
                _clipped_exp_weight_logw(self.beta * _sg(adv), math.log(self.exp_adv_max))
            )
            vintage["iw_logw"] = (
                "consumption/exp_adv@pre_penalty_pre_update_target_critic_and_pre_update_value_net"
            )
        return CalibrationInputs(
            targets=_sg(targets),
            q_at_data=_sg(_min_agg(q_heads)),
            q_heads=_sg(q_heads),
            obs=batch.obs,
            action=batch.action,
            iw_logw=iw_logw,
            vintage=vintage,
        )

    def channel_sites(self) -> dict:
        return {
            C4_ADVANTAGE_PENALTY: ChannelSite(
                channel=C4_ADVANTAGE_PENALTY,
                enclosing_form="multiplier",
                eval_action="dataset",
                reads_params="pre_update_target_critic",
                note="`adv` at corl_iql.py:434-435 becomes `(target_q - pen) - value(s)` and `exp_adv` at :463-465 keeps its EXP_ADV_MAX clip. That clip is what makes the dial's LEVEL live: at pen=0, 7.4% of the batch sits at the clip with ess=0.181; at pen=2.0 nothing is clipped and ess=0.051.",
            ),
            C3_PER_SAMPLE_TERM_REWEIGHT: ChannelSite(
                channel=C3_PER_SAMPLE_TERM_REWEIGHT,
                enclosing_form="multiplier",
                term="value_regression",
                eval_action="dataset",
                reads_params="post_update_online_critic",
                note="`bc_losses` is already (B,) at corl_iql.py:471/:474, so `mean(exp_adv * bc_losses)` at :475 becomes `mean(w * exp_adv * bc_losses)` with no restructuring. SELECTING BOTH C4 AND C3 PUTS TWO CHANNELS ON THE SAME TERM -- permitted, but it writes bca_confound into the arm identity.",
            ),
        }

    def step_order(self) -> tuple:
        return ("value", "critic", "polyak", "calibrator", "penalty", "actor")

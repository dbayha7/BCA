"""IQL scale-fitting records and calibrator construction."""

from dataclasses import dataclass, field
from typing import Any, Literal, NamedTuple
import jax
from calibration.iql_network import (
    ScalarCalibrator,
    StateDependentCalibrator,
    bayesian_bootstrap_weights,
    create_cal_train_state,
    make_cal_loss_fn,
)

_FOLD_BOOTSTRAP = 1111704112
_EPS = 1e-06


@dataclass
class BCAFrameworkArgs:
    cond_mode: Literal["scalar", "state", "state_action"] = "state"
    cal_alpha: float = 0.1
    cal_beta: float = 20.0
    cal_lambda_width: float = 0.005
    cal_lr: float = 0.001
    cal_hidden_dim: int = 64
    cal_mode: Literal["soft", "hard"] = "soft"
    balance_coverage: bool = True
    balance_weight: float = 0.5
    use_bootstrap: bool = True
    num_heads: int = 2


@dataclass(frozen=True)
class CalibrationInputs:
    targets: jax.Array
    q_at_data: jax.Array
    q_heads: jax.Array
    obs: jax.Array
    action: jax.Array
    vintage: dict
    residual_norm: bool = True
    iw_logw: jax.Array | None = None
    adv_scale: jax.Array | None = None

    def batch_size(self) -> int:
        return int(self.targets.shape[0])


class CalibratorReadout(NamedTuple):
    eta: jax.Array
    resid_scale: jax.Array
    source_raw: jax.Array
    coverage: jax.Array
    coverage_hard: jax.Array
    coverage_gap: jax.Array
    blend: jax.Array
    beta_scale: jax.Array
    diagnostics: dict


class BCAState(NamedTuple):
    calibrator: Any
    resid_scale: jax.Array
    log_beta_bias: jax.Array
    beta_density: Any = None


def build_calibrator(
    rng, a: BCAFrameworkArgs, obs_mean, obs_std, dummy_obs, dummy_action
):
    if a.cond_mode == "scalar":
        net, dummy = (ScalarCalibrator(), [])
    elif a.cond_mode == "state":
        net, dummy = (
            StateDependentCalibrator(obs_mean, obs_std, a.cal_hidden_dim),
            [dummy_obs[None]],
        )
    else:
        net, dummy = (
            StateDependentCalibrator(obs_mean, obs_std, a.cal_hidden_dim),
            [dummy_obs[None], dummy_action[None]],
        )
    return create_cal_train_state(rng, net, dummy, a.cal_lr)

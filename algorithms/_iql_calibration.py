"""BCA iql calibration.

Scientific definitions extracted from the recorded BCA source; see configs/sources.json.
"""

from dataclasses import dataclass, field
from typing import Any, Literal, NamedTuple
import jax
from _iql_scale_network import (
    ScalarCalibrator,
    StateDependentCalibrator,
    bayesian_bootstrap_weights,
    create_cal_train_state,
    make_cal_loss_fn,
)
from _iw_module import iw_gauss_logp

C1_TARGET_SHIFT = "target_shift"
C2_OBJECTIVE_MIXTURE_GATE = "objective_mixture_gate"
C3_PER_SAMPLE_TERM_REWEIGHT = "per_sample_term_reweight"
C4_ADVANTAGE_PENALTY = "advantage_penalty"
C5_ADMISSION_GATE = "admission_gate"
_FOLD_BOOTSTRAP = 1111704112
_EPS = 1e-06


@dataclass
class BCAFrameworkArgs:
    use_bca: bool = False
    bca_channel: str = ""
    bca_source: Literal["conformal", "head_std", "rankgauss"] = "conformal"
    bca_source_permute: bool = False
    rankgauss_plot_a: float = -6.0
    mod_gain: float = 1.0
    mod_clip: float = 3.0
    awr_dial: Literal["raw", "center", "zscore", "cap"] = "center"
    awr_pen_frac: float = 0.25
    eta_max: float = 10.0
    gate_floor: float = 0.0
    allow_full_bc: bool = False
    pess_gamma_scale: float = 1.0
    filter_temp: float = 1.0
    cond_mode: Literal["scalar", "state", "state_action"] = "state"
    cal_alpha: float = 0.1
    cal_beta: float = 20.0
    cal_lambda_width: float = 0.005
    cal_lr: float = 0.001
    cal_hidden_dim: int = 64
    cal_mode: Literal["soft", "hard"] = "soft"
    report_hard_coverage: bool = False
    balance_coverage: bool = True
    balance_weight: float = 0.5
    use_bootstrap: bool = True
    num_heads: int = 2
    cal_iw: bool = False
    cal_iw_source: Literal["policy", "consumption"] = "policy"
    cal_iw_stab: Literal["clip", "psis", "temper"] = "clip"
    cal_iw_clip_lo: float = 0.1
    cal_iw_clip_hi: float = 10.0
    cal_iw_ess_floor: float = 0.25
    cal_iw_tau_min: float = 0.05
    cal_iw_weight_width: bool = False
    cal_iw_permute: bool = False
    iw_health: bool = False
    cal_warmup: int = 0
    ramp_all_sources: bool = False
    use_eat: bool = False
    probe_max_step: int = 2048

    def channels(self) -> frozenset:
        raw = [c.strip() for c in self.bca_channel.split(",") if c.strip()]
        return frozenset(raw)


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


@dataclass(frozen=True)
class ChannelSite:
    channel: str
    enclosing_form: str
    term: str | None = None
    eval_action: str = "dataset"
    reads_params: str = ""
    gamma_scaled: bool = False
    note: str = ""


@dataclass(frozen=True)
class ChannelOutput:
    channel: str
    value: jax.Array
    neutral: jax.Array
    dose_matched_null: jax.Array
    stats: dict = field(default_factory=dict)


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


def build_calibrator(rng, a: BCAFrameworkArgs, obs_mean, obs_std, dummy_obs, dummy_action):
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

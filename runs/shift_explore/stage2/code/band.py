"""Stage 2: BCA's band and its lower-edge consumer (method version bca-lower-edge-1).

THE SIGNAL IS BCA'S, UNCHANGED. The band is fitted by algorithms/td3_bc_bca.py's own functions, imported read-only and
never edited: fit_scale (the scale network eta(s, a), the unit u, Bayesian-bootstrap masses, coverage sharpness 20,
width penalty 0.005, unit averaging 0.99, learning rate 0.001) and refresh (the frozen scale network eta_f, the frozen
unit u_f and the threshold R = max(plain 90% quantile, 95%-credible quantile) of weighted Bayesian conformal
prediction, miscoverage 0.1, 1,000 draws, uniform weights), with the gate's settings (experiments/shift/learners.
BCA_SETTINGS). The width is calibration/dose.py's:

    U(s, a) = R * u_f * max(eta_f(s, a), 1e-6)

WHAT IS NEW (the consumer; this is why it is a new method version):
  1. the width is read at ANY (s, a), in particular at the actor's own command, and is differentiable in the command:
     the gradient flows through the frozen scale network's command input (its parameters get no gradient);
  2. "cannot certify" (R = +infinity) means maximum caution: the width is U_max, the largest finite width over the
     bank (computed with R_c, the threshold with +infinity replaced by the largest finite candidate), never 0 (with
     uniform weights this is all or nothing and inert: see below);
  3. no 1.5 cap: the width is used in critic value units, unbounded;
  4. the primary penalty is paired and one-sided (change A1 of the room test: a penalty is read relative to the
     recorded command at the same state):

        P(s_i, a) = [ U(s_i, a) - U(s_i, a_i) ]_+          (primary, "paired lower edge")
        U(s_i, a)                                           (declared secondary arm, "unpaired lower edge")

     and the actor climbs Q1(s, pi(s)) - k * P(s, pi(s)) with k = 1 (k = 2 a declared secondary arm).

THE WEIGHTED VERSION (declared fallback, used only if detection direction or size fails with uniform weights): the
threshold uses likelihood-ratio weights w(s, a) (weights.py) through BCA's own threshold rule (calibration/wbcp.
calibrate with bank weights w_j and test mass t), tabulated on 48 log-spaced test masses; one half of the bank sets the
threshold, the other half only audits coverage. Its width:

    U~(s, a) = R_c(w(s, a)) * u_f * max(eta_f(s, a), 1e-6) + U_max * softplus(log w(s, a) - l*)

with R_c(t) the tabulated threshold, +infinity replaced by its largest finite value, interpolated linearly in log t;
l* the log of the smallest tabulated test mass at which R is +infinity ("cannot certify"); U_max the largest finite
width over the threshold half of the bank.

GRADIENT PATHS (declared in the pre-registration). Uniform: the actor's gradient reaches U only through eta_f's
command input. Weighted: also through the frozen classifier's command input log w(s, a), in BOTH terms: the
tabulated threshold R_c(w(s, a)) (piecewise linear in log w, so host optimisation backpropagates through the
threshold calculation too, which ALGORITHMS.md:127-130 forbids for BCA as written) and the refusal term
U_max softplus(log w - l*). Only the classifier's and eta_f's parameters are frozen (stop_gradient).

UNIFORM "CANNOT CERTIFY" IS ALL OR NOTHING: one threshold for all inputs, so a band that cannot certify gives U_max at
every pair. Then the paired penalty is 0 with zero gradient, the unpaired penalty is a constant with zero gradient, and
the deployment check's tau (the split-conformal level of a constant) equals U_max, so U <= tau everywhere and the
check never fires. "Maximum caution" acts per command only in the weighted version. With about 850 bank rows the
uniform case cannot happen (calibration.reference.certifiable holds for n >= 36).

ENTRY POINTS. Functions that take ThrusterChain data (fit_on_context, WidthReader, load_band) call
stage2_common.require_approval() first and refuse without a valid approval of the current prereg.json. The pure functions
(width_*, paired, fit_band_arrays, band_from_reference, tabulate_threshold, ...) are the method's arithmetic; the unit
tests call them on small random arrays only.
"""

import math
import sys
import time
from pathlib import Path
from typing import Any, NamedTuple

import numpy as np

ROOT = Path("/home/user/bca")
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
from flax.training.train_state import TrainState  # noqa: E402

import algorithms.td3_bc as BASE  # noqa: E402
import algorithms.td3_bc_bca as P  # noqa: E402
from calibration import wbcp  # noqa: E402
from calibration.network import Calibrator  # noqa: E402
from calibration.reference import WBCPConfig, initial_reference, positive_scale  # noqa: E402
import optax  # noqa: E402

import stage2_common as C  # noqa: E402
import weights as Wt  # noqa: E402

EPS = 1e-6
GRID = 48
TAG_BAND = 41               # learners.TAG_BAND: the band's batch rows and keys (as learners.bca_band)
KEY_CALIBRATOR = 1128352841  # td3_bc_bca.initialize's calibrator fold_in constant (learners.KEY_CALIBRATOR)
TAG_SPLIT = 71               # weighted version: the bank's split into a threshold half and an audit half
TAG_TABLE = 72               # weighted version: the posterior draws of the tabulated threshold
BAND_FIT_STEPS = 20_000

# BCA's settings exactly as the gate declares them (experiments/shift/learners.BCA_SETTINGS); blend is unused by the
# lower-edge consumer (no dose), kept only so that td3_bc_bca.Config accepts the arm.
SETTINGS = dict(miscoverage=0.1, credibility=0.95, draws=1000, blend=0.5, cal_lr=0.001, cal_beta=20.0,
                width_penalty=0.005, scale_ema=0.99, warmup=10_000, refresh_every=5_000)


def bca_config():
    s = SETTINGS
    return P.Config(arm="bca", posterior=WBCPConfig(s["miscoverage"], s["credibility"], s["draws"]), blend=s["blend"],
                    cal_lr=s["cal_lr"], cal_beta=s["cal_beta"], width_penalty=s["width_penalty"],
                    scale_ema=s["scale_ema"])


def TC_DX():
    import experiments.shift.thruster_chain as TC
    return TC.DX


def TC_DA():
    import experiments.shift.thruster_chain as TC
    return TC.DA


def calibrator(dx):
    """BCA's scale network as the gate builds it (inputs already normalised: mean 0, SD 1, plus 0.001)."""
    return Calibrator(jnp.zeros(dx), jnp.ones(dx), state_dep=True)


# --------------------------------------------------------------------------------------------- the band

class Band(NamedTuple):
    """A frozen band. Scalars are float32 0-d arrays so that a Band can be passed into compiled programs."""

    cal_params: Any        # eta_f's parameters (frozen)
    unit: Any              # u_f
    R: Any                 # threshold (uniform weights); +inf when nothing can be certified
    R_c: Any               # R with +inf replaced by the largest finite candidate (finite)
    U_max: Any             # the largest finite width over the bank (threshold half when weighted)
    certified: Any         # bool
    ready: Any             # bool; False before the co-trained host's first refresh: width 0 (BCA's own rule)
    log_t: Any             # weighted: (GRID,) tabulated log test masses (same units as the classifier's logit)
    R_tab: Any             # weighted: (GRID,) R_c at each tabulated test mass (finite)
    l_star: Any            # weighted: log of the smallest tabulated test mass at which R is +inf
    cls_params: Any        # weighted: the classifier's parameters; uniform: None


def f32(x):
    return jnp.asarray(x, jnp.float32)


def uniform_band(cal_params, unit, R, R_c, U_max, ready=True):
    R = float(R)
    return Band(cal_params, f32(unit), f32(R), f32(R_c), f32(U_max), jnp.asarray(math.isfinite(R)),
                jnp.asarray(bool(ready)), jnp.zeros(GRID, jnp.float32), jnp.zeros(GRID, jnp.float32),
                f32(1e30), None)


def eta(cal, band, obs, a):
    """eta_f(s, a) with eta_f's parameters frozen: gradients reach the command, never the parameters."""
    return cal.apply(jax.lax.stop_gradient(band.cal_params), obs, a)


def width_uniform(cal, band, obs, a):
    """U(s, a) = R u_f max(eta_f, 1e-6); U_max where the band cannot certify; 0 before the first refresh."""
    e = jnp.maximum(eta(cal, band, obs, a), EPS)
    R_safe = jnp.where(band.certified, band.R, band.R_c)       # finite in both branches (no inf * 0 in gradients)
    u = jnp.where(band.certified, R_safe * band.unit * e, band.U_max * jnp.ones_like(e))
    return jnp.where(band.ready, u, jnp.zeros_like(u))


def width_weighted(cal, band, obs, a):
    """U~(s, a) = R_c(w) u_f max(eta_f, 1e-6) + U_max softplus(log w - l*); 0 before the first refresh. The command a
    stays differentiable in log w = logit(s, a): the gradient flows through the classifier's command input into both
    R_c(w) and the softplus refusal term (declared); the classifier's parameters are frozen."""
    e = jnp.maximum(eta(cal, band, obs, a), EPS)
    lw = Wt.logit(jax.lax.stop_gradient(band.cls_params), obs, a)
    Rw = jnp.interp(lw, band.log_t, band.R_tab)
    u = Rw * band.unit * e + band.U_max * jax.nn.softplus(lw - band.l_star)
    return jnp.where(band.ready, u, jnp.zeros_like(u))


def width(cal, band, obs, a, weighted):
    return width_weighted(cal, band, obs, a) if weighted else width_uniform(cal, band, obs, a)


def paired(u_at_a, u_at_recorded):
    """The primary penalty: [U(s, a) - U(s, a_i)]_+ (zero at the recorded command; never rewards a command)."""
    return jnp.maximum(u_at_a - jax.lax.stop_gradient(u_at_recorded), 0.0)


def lower_edge_objective(q1, u_at_a, u_at_recorded, k_paired, k_unpaired=0.0):
    """Per row: Q1(s, a) - k_paired [U(s, a) - U(s, a_i)]_+ - k_unpaired U(s, a)."""
    return q1 - k_paired * paired(u_at_a, u_at_recorded) - k_unpaired * u_at_a


_ROWS = {}


def width_rows(cal, band, obs, actions, weighted, chunk=8192):
    """U at many (normalised observation, command) pairs; numpy float64 out; one compiled program per shape."""
    key = (id(cal), bool(weighted), int(chunk), np.shape(obs)[-1], np.shape(actions)[-1])
    if key not in _ROWS:
        _ROWS[key] = jax.jit(lambda b, o, a: width(cal, b, o, a, weighted))
    obs, actions = np.asarray(obs, np.float32), np.asarray(actions, np.float32)
    out = []
    for i in range(0, len(obs), chunk):
        o, a = obs[i:i + chunk], actions[i:i + chunk]
        pad = chunk - len(o)
        if pad:
            o = np.concatenate([o, np.repeat(o[-1:], pad, 0)])
            a = np.concatenate([a, np.repeat(a[-1:], pad, 0)])
        out.append(np.asarray(_ROWS[key](band, jnp.asarray(o), jnp.asarray(a)), np.float64)[:chunk - pad])
    return np.concatenate(out) if out else np.zeros(0)


def width_grad_rows(cal, band, obs, actions, weighted):
    """dU/da per row (n, da): the readout's gradient through the frozen scale network's command input."""
    g = jax.vmap(jax.grad(lambda a, o: width(cal, band, o[None], a[None], weighted)[0]), in_axes=(0, 0))
    return np.asarray(jax.jit(g)(jnp.asarray(actions, jnp.float32), jnp.asarray(obs, jnp.float32)), np.float64)


# --------------------------------------------------------------------------------------------- the fit (BCA's own)

def fixed_native(critic_params, fixed, lr=3e-4):
    """The frozen fixed-controller critic as a td3_bc AgentTrainState: its 'actor' is the fixed controller (the
    stall-aware controller), so td3_bc_bca.native_target builds the residual target with the stall-aware controller's
    next command plus TD3's smoothing noise (as the critic was trained). The gate saved only the critic's online
    parameters, so the target critic is the same frozen critic."""
    tx = BASE.C.torch_adam(lr)
    actor = TrainState.create(apply_fn=fixed.apply, params=fixed.params(), tx=tx)
    critic = TrainState.create(apply_fn=fixed_critic_apply(), params=critic_params, tx=tx)
    return BASE.AgentTrainState(actor, actor, critic, critic)


def fixed_critic_apply():
    import experiments.shift.learners as L
    return L.CRITIC.apply


def initial_state(args, config, native, dx, da):
    """td3_bc_bca.initialize's calibrator state without its D4RL checks (learners.bca_initial_state's rule): the
    calibrator initialised from fold_in(rng, 1128352841), rng from td3_bc.initialize(args); Adam (cal_lr); unit 1;
    the initial (not ready) reference."""
    rng, _, _ = BASE.initialize(args, dx, da, 1.0)
    cal = calibrator(dx)
    cal_state = TrainState.create(apply_fn=cal.apply,
                                  params=cal.init(jax.random.fold_in(rng, KEY_CALIBRATOR), jnp.zeros(dx)[None],
                                                  jnp.zeros(da)[None]),
                                  tx=optax.adam(config.cal_lr))
    return rng, P.State(native, cal_state, jnp.asarray(1.0), initial_reference(cal_state.params)), cal


def band_rows_and_keys(rng, n, steps, batch, rep):
    """learners.bca_band's batch rows default_rng([20261015, rep, 41]), its scale-fit keys and its refresh key."""
    rows = np.random.default_rng([20261015, int(rep), TAG_BAND]).integers(n, size=(steps, batch))
    keys = jax.random.split(jax.random.fold_in(rng, TAG_BAND), steps)
    refresh_key = jax.random.fold_in(
        jax.random.PRNGKey(int(np.random.default_rng([20261015, int(rep), TAG_BAND, 1]).integers(2 ** 31 - 1))), steps)
    return rows, keys, refresh_key


def fit_band_arrays(args, config, models, state, train, bank, rows, keys, refresh_key):
    """learners.bca_band's procedure on a frozen host, restated: td3_bc_bca.fit_scale on the batches `rows` (steps x
    batch) of `train`, then one td3_bc_bca.refresh on the bank. models = (next-command actor, critic, calibrator). The
    host's networks never change. Returns (state, accepted share, refresh diagnostics)."""
    def body(st, inp):
        r, k = inp
        batch = jax.tree_util.tree_map(lambda x: x[r], train)
        st, diag = P.fit_scale(args, config, models, st, batch, k, 1.0)
        return st, diag["scale_fit_accepted"]

    native_sha = _tree_sha(state.native)
    state, accepted = jax.jit(lambda st, r, k: jax.lax.scan(body, st, (r, k)))(state, jnp.asarray(rows, jnp.int32),
                                                                                keys)
    state, diag, wdiag = P.refresh(args, config, models, state, bank, refresh_key, 1.0,
                                   heldout_ids=np.arange(len(bank.obs)))
    if _tree_sha(state.native) != native_sha:
        raise AssertionError("the band must not change the frozen host")
    return state, float(np.mean(np.asarray(accepted))), dict(diag, **{k: _plain(v) for k, v in wdiag.items()})


def _plain(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return v


def _tree_sha(tree):
    """SHA-256 of a pytree's values, each leaf first made a JAX array (so a Python int step counter and the int32 it
    becomes inside a compiled program hash alike)."""
    import hashlib
    h = hashlib.sha256()
    for x in jax.tree_util.tree_leaves(tree):
        a = np.ascontiguousarray(np.asarray(jnp.asarray(x)))
        h.update(str(a.dtype).encode() + str(a.shape).encode())
        h.update(a.tobytes())
    return h.hexdigest()


def band_from_reference(reference, cal, bank_obs, bank_actions):
    """The uniform band from BCA's frozen reference: R, u_f, eta_f; R_c = R when finite, else the largest finite of
    (plain 90% quantile, credible quantile); U_max = R_c u_f max_j max(eta_f(s_j, a_j), 1e-6) over the bank."""
    if not bool(reference.ready):
        raise ValueError("the reference is not ready: the refresh did not run or its inputs were invalid")
    R = float(reference.threshold)
    cands = [v for v in (R, float(reference.lambda_hat), float(reference.lambda_hpd)) if math.isfinite(v)]
    if not cands:
        raise ValueError("no finite threshold candidate")
    R_c = R if math.isfinite(R) else max(cands)
    e = np.maximum(np.asarray(cal.apply(reference.cal_params, jnp.asarray(bank_obs), jnp.asarray(bank_actions)),
                              np.float64), EPS)
    U_max = R_c * float(reference.residual_scale) * float(e.max())
    return uniform_band(reference.cal_params, float(reference.residual_scale), R, R_c, U_max)


def bank_scores(args, models, state, bank, refresh_key, cal_params, unit):
    """BCA's own scores at the bank rows, as td3_bc_bca.refresh computes them with the same key:
    |y_j - q_j| / (max(eta_f(s_j, a_j), 1e-6) u_f), y from native_target, q = min over the online heads."""
    target = P.native_target(args, models, state.native, bank, jax.random.fold_in(refresh_key, 1213156420), 1.0)
    q = models[1].apply(state.native.critic.params, bank.obs, bank.action).min(axis=-1)
    e = models[2].apply(cal_params, bank.obs, bank.action)
    scale = np.asarray(positive_scale(e, jnp.asarray(unit)), np.float64)   # freeze_reference's own arithmetic
    return np.abs(np.asarray(target - q, np.float64)) / scale, np.asarray(e, np.float64)


def uniform_threshold_check(scores, refresh_key, config, R_bca):
    """The uniform threshold recomputed from bank_scores with refresh's own posterior key; must equal BCA's R
    (a plumbing check of bank_scores, reported)."""
    rng = np.random.default_rng(np.asarray(jax.random.fold_in(refresh_key, 1347375956), np.uint32).ravel())
    res = wbcp.calibrate(scores, rng, alpha=config.posterior.alpha, beta=config.posterior.credibility,
                         draws=config.posterior.draws)
    return dict(recomputed=res.threshold, bca=float(R_bca),
                equal=bool(np.float32(res.threshold) == np.float32(R_bca)))


def split_bank(n, rep):
    """The weighted version's bank halves: a fixed permutation, first half sets the threshold, second half audits."""
    perm = np.random.default_rng([20261015, int(rep), TAG_SPLIT]).permutation(n)
    return np.sort(perm[: n // 2]), np.sort(perm[n // 2:])


def tabulate_threshold(scores, bank_logw, rep, config, grid=GRID):
    """BCA's threshold rule (wbcp.calibrate) with likelihood-ratio weights w_j = exp(log w_j) on the bank's threshold
    half, tabulated at `grid` log-spaced test masses t (common random numbers across t, so R(t) is nondecreasing).
    The grid runs from the smallest bank log-weight minus 1 to the closed-form refusal point
    log(alpha / (1 - alpha) * sum_j w_j) plus 3. Returns (log_t, R(t) with +inf, R_c(t), l*)."""
    scores = np.asarray(scores, np.float64)
    lw = np.asarray(bank_logw, np.float64)
    shift = float(lw.max())
    w = np.exp(lw - shift)
    a = config.posterior.alpha
    l_ref = math.log(a / (1 - a)) + math.log(float(w.sum()))
    lo, hi = float(np.log(w.min())) - 1.0, l_ref + 3.0
    grid_lt = np.linspace(lo, hi, int(grid))
    R = np.empty(len(grid_lt))
    for i, lt in enumerate(grid_lt):
        rng = np.random.default_rng([20261015, int(rep), TAG_TABLE])
        R[i] = wbcp.calibrate(scores, rng, weights=w, test_mass=float(np.exp(lt)), alpha=a,
                              beta=config.posterior.credibility, draws=config.posterior.draws).threshold
    finite = np.isfinite(R)
    if not finite.any():
        raise ValueError("the weighted threshold is +inf at every tabulated test mass")
    R_c = np.where(finite, R, R[finite].max())
    inf_at = np.flatnonzero(~finite)
    l_star = (grid_lt[inf_at[0]] if len(inf_at) else grid_lt[-1] + 50.0) + shift
    return grid_lt + shift, R, R_c, float(l_star)


def weighted_band(uniform, log_t, R_c_tab, l_star, cls_params, cal, bank_obs_A, bank_act_A, bank_logw_A):
    """The weighted band from the uniform fit's eta_f and u_f and the tabulated threshold. U_max = the largest finite
    width over the threshold half: max_j R_c(w_j) u_f max(eta_f(s_j, a_j), 1e-6) over rows with log w_j < l*."""
    e = np.maximum(np.asarray(cal.apply(uniform.cal_params, jnp.asarray(bank_obs_A), jnp.asarray(bank_act_A)),
                              np.float64), EPS)
    lw = np.asarray(bank_logw_A, np.float64)
    Rw = np.interp(lw, np.asarray(log_t), np.asarray(R_c_tab))
    fin = lw < l_star
    U = Rw * float(uniform.unit) * e
    U_max = float(U[fin].max()) if fin.any() else float(U.max())
    return Band(uniform.cal_params, uniform.unit, f32(np.max(R_c_tab)), f32(np.max(R_c_tab)), f32(U_max), jnp.asarray(True),
                jnp.asarray(True), f32(log_t), f32(R_c_tab), f32(l_star), cls_params)


def coverage_audit(residuals_abs, widths):
    """Share of audit-half bank rows whose |y - q| lies inside the band (target 0.9); reported only."""
    r, w = np.asarray(residuals_abs, np.float64), np.asarray(widths, np.float64)
    return dict(rows=int(len(r)), covered_share=float(np.mean(r <= w)), target=0.9)


# --------------------------------------------------------------------------------------------- guarded drivers

def band_template(dx, da, weighted):
    cal = calibrator(dx)
    p = cal.init(jax.random.PRNGKey(0), jnp.zeros(dx)[None], jnp.zeros(da)[None])
    cls = Wt.cls_init(0, dx, da)[0] if weighted else None
    return Band(p, f32(1.0), f32(1.0), f32(1.0), f32(1.0), jnp.asarray(True), jnp.asarray(True),
                jnp.zeros(GRID, jnp.float32), jnp.zeros(GRID, jnp.float32), f32(1.0), cls)


def fit_on_context(ctx, world, critic_kind, approval, out_dir, *, log_fn=C.log):
    """The detection test's fit (plan 5.2) for one seed and world. Refuses without approval.

    critic_kind 'fixed': the frozen fixed-controller gapped critic (check 2's, the one the policy test trains
    against), residual target with the stall-aware controller's next command (fixed_native). 'host': check 3's
    frozen co-trained plain TD3+BC critic, exactly as written (learners.bca_band's native: the host's own actor
    target and critic target). Uniform weights (BCA as written). Saves band.msgpack, band.json and bank_scores.npz
    (BCA's own scores at the bank rows, for the weighted version) under out_dir; returns (Band, record). No width is
    read at any pair other than the bank's recorded rows."""
    C.recheck(approval)
    out_dir = C.guard_out(out_dir)
    if (out_dir / "band.json").exists():
        raise SystemExit(f"refused: {out_dir}/band.json exists; a band is fitted once")
    out_dir.mkdir(parents=True, exist_ok=True)
    import experiments.shift.learners as L
    t0 = time.perf_counter()
    learn = ctx.learn
    pg = ctx.prepared(world)
    args = L.make_args(learn, ctx.rep)
    config = bca_config()
    if critic_kind == "fixed":
        critic = ctx.critic(world, "gap")[0]
        fixed = L.FixedController("stall_aware", pg.physics, pg.obs_mean, pg.obs_std)
        native = fixed_native(critic, fixed, learn.lr)
        rng, state, cal = initial_state(args, config, native, L.DX, L.DA)
        models = (fixed, L.CRITIC, cal)
    elif critic_kind == "host":
        if world != "stall":
            raise ValueError("check 3's host critic exists in the stalling world only")
        host = ctx.host()[0]
        rng, state, cal = initial_state(args, config, host, L.DX, L.DA)
        models = (L.ACTOR, L.CRITIC, cal)
    else:
        raise ValueError(critic_kind)
    steps = BAND_FIT_STEPS
    rows, keys, refresh_key = band_rows_and_keys(rng, pg.n, steps, learn.batch_size, ctx.rep)
    state, accepted, diag = fit_band_arrays(args, config, models, state, pg.train, pg.bank, rows, keys, refresh_key)
    ref = state.posterior
    record = dict(label=C.LABEL_LONG, kind="stage2-band", rep=int(ctx.rep), world=world, critic=critic_kind,
                  weighted=False, steps=steps, accepted_share=accepted, refresh=diag,
                  approval_sha256=approval.file_sha256, prereg_sha256=approval.prereg_sha256,
                  settings=SETTINGS, native_sha256=_tree_sha(state.native), data=pg.describe())
    if not bool(ref.ready):
        record.update(failed=True, reason="the refresh did not produce a ready reference (invalid inputs)")
        C.write_json(Path(out_dir) / "band.json", record)
        return None, record
    band = band_from_reference(ref, cal, np.asarray(pg.bank.obs), np.asarray(pg.bank.action))
    scores, e_bank = bank_scores(args, models, state, pg.bank, refresh_key, ref.cal_params, ref.residual_scale)
    np.savez(C.guard_out(Path(out_dir) / "bank_scores.npz"), scores=scores, eta=e_bank)
    record["bank_scores_sha256"] = C.sha256_bytes(np.ascontiguousarray(scores).tobytes())
    record["uniform"] = dict(R=float(ref.threshold), lambda_hat=float(ref.lambda_hat),
                             lambda_hpd=float(ref.lambda_hpd), unit=float(ref.residual_scale),
                             R_c=float(band.R_c), U_max=float(band.U_max), certified=bool(band.certified),
                             bank_rows=int(len(scores)),
                             threshold_recomputed=uniform_threshold_check(scores, refresh_key, config, ref.threshold))
    record["band_file_sha256"] = C.save_tree(Path(out_dir) / "band.msgpack", band)
    record["failed"] = False
    record["seconds"] = time.perf_counter() - t0
    C.write_json(Path(out_dir) / "band.json", record)
    log_fn(f"band fitted: seed {ctx.rep} {world} {critic_kind} R={record['uniform']['R']:.4g}")
    return band, record


def load_band(out_dir, approval, dx, da):
    """A saved band, verified against its record (sidecar) and the record's file SHA-256. Refuses without approval."""
    C.recheck(approval)
    rec = C.read_json(Path(out_dir) / "band.json")
    if rec.get("failed"):
        return None, rec
    if rec.get("prereg_sha256") != approval.prereg_sha256:
        raise SystemExit(f"refused: {out_dir}/band.json was fitted under another pre-registration")
    band = C.load_tree(Path(out_dir) / "band.msgpack", band_template(dx, da, rec["weighted"]), rec["band_file_sha256"])
    return band, rec


class WidthReader:
    """U(s, a) at float64 states and commands, for the detection script and the deployment check: the state is
    rounded to float32 and normalised as the learners see it ((float32(s) - mean) / std). Refuses without approval."""

    def __init__(self, band, record, obs_mean, obs_std, approval, dx):
        C.recheck(approval)
        self.band, self.record, self.weighted = band, record, bool(record["weighted"])
        self.mean, self.std = np.asarray(obs_mean, np.float32), np.asarray(obs_std, np.float32)
        self.cal = calibrator(dx)
        self.name = f"bca_width_{'weighted' if self.weighted else 'uniform'}_{record['critic']}"
        self.sha256 = record["band_file_sha256"]

    def normalise(self, states):
        return (np.asarray(states, np.float32) - self.mean) / self.std

    def __call__(self, states, actions):
        return width_rows(self.cal, self.band, self.normalise(states), np.asarray(actions, np.float32), self.weighted)


def weighted_refitter(ctx, world, pg, band_u, brec, d_band, approval):
    """The weighted version built on a saved uniform fit (same eta_f and u_f): refit(actor_params) -> (weighted Band,
    U at every training row's recorded command, diagnostics). The classifier (weights.py) is fitted with that actor;
    the bank's log-weights are its logits at the bank pairs; the threshold is BCA's rule tabulated on the bank's
    threshold half with BCA's own scores (bank_scores.npz from the uniform fit); the audit half's coverage is
    reported. Refuses without approval."""
    C.recheck(approval)
    import experiments.shift.learners as L
    with np.load(Path(d_band) / "bank_scores.npz") as z:
        scores, e_bank = z["scores"], z["eta"]
    if C.sha256_bytes(np.ascontiguousarray(scores).tobytes()) != brec["bank_scores_sha256"]:
        raise SystemExit("refused: bank_scores.npz differs from the band record")
    A_idx, B_idx = split_bank(len(scores), ctx.rep)
    obs = np.asarray(pg.train.obs)
    cl = L.actor_rows(ctx.clone(world)[0], obs)
    rows = Wt.cls_rows(ctx.rep, pg.n)
    p0, cseed = Wt.cls_init(ctx.rep, L.DX, L.DA)
    b_obs, b_act = np.asarray(pg.bank.obs), np.asarray(pg.bank.action)
    config = bca_config()
    cal = calibrator(L.DX)
    resid = scores * np.maximum(e_bank, EPS) * float(band_u.unit)

    def refit(actor_params):
        cls_params, cdiag = Wt.fit_for_actor(L.ACTOR.apply, actor_params, obs, pg.actions, cl, rows, p0)
        lw = np.asarray(Wt.logit(cls_params, jnp.asarray(b_obs), jnp.asarray(b_act)), np.float64)
        log_t, R_raw, R_c, l_star = tabulate_threshold(scores[A_idx], lw[A_idx], ctx.rep, config)
        band = weighted_band(band_u, log_t, R_c, l_star, cls_params, cal, b_obs[A_idx], b_act[A_idx], lw[A_idx])
        u_rec = width_rows(cal, band, obs, pg.actions, True)
        wB = width_rows(cal, band, b_obs[B_idx], b_act[B_idx], True)
        diag = dict(cdiag, init_seed=cseed, l_star=l_star, U_max=float(band.U_max), log_t=log_t, R=R_raw, R_c=R_c,
                    threshold_half=int(len(A_idx)), audit_half=int(len(B_idx)), audit=coverage_audit(resid[B_idx], wB),
                    bank_logw_quantiles={q: float(np.quantile(lw, q / 100)) for q in (5, 50, 95, 99)},
                    finite=bool(cdiag["finite"] and np.all(np.isfinite(u_rec))))
        return band, u_rec, diag

    return refit


def fit_weighted_on_context(ctx, world, critic_kind, approval, uniform_dir, out_dir, log_fn=C.log):
    """The detection test's weighted version (declared fallback) for one seed and world: the saved uniform fit plus
    the classifier fitted with the gapped critic's actor (the actor whose commands the pair sets hold). Saves
    band.msgpack and band.json under out_dir."""
    C.recheck(approval)
    out_dir = C.guard_out(out_dir)
    if (out_dir / "band.json").exists():
        raise SystemExit(f"refused: {out_dir}/band.json exists; a band is fitted once")
    out_dir.mkdir(parents=True, exist_ok=True)
    import experiments.shift.learners as L
    t0 = time.perf_counter()
    band_u, brec = load_band(uniform_dir, approval, L.DX, L.DA)
    if band_u is None:
        raise SystemExit("refused: the uniform fit of this seed failed; no weighted version can be built on it")
    pg = ctx.prepared(world)
    refit = weighted_refitter(ctx, world, pg, band_u, brec, uniform_dir, approval)
    band, _, diag = refit(ctx.none(world))
    record = dict(label=C.LABEL_LONG, kind="stage2-band", rep=int(ctx.rep), world=world, critic=critic_kind,
                  weighted=True, uniform_band_record_sha256=C.file_sha256(Path(uniform_dir) / "band.json"),
                  uniform=brec.get("uniform"), weighted_fit=diag, approval_sha256=approval.file_sha256,
                  prereg_sha256=approval.prereg_sha256, bank_scores_sha256=brec["bank_scores_sha256"],
                  actor="the gapped critic's actor (check 2's saved actor, or the fresh seed's)")
    record["band_file_sha256"] = C.save_tree(out_dir / "band.msgpack", band)
    record.update(failed=False, seconds=time.perf_counter() - t0)
    C.write_json(out_dir / "band.json", record)
    log_fn(f"weighted band: seed {ctx.rep} {world} {critic_kind} l*={diag['l_star']:.3g} U_max={diag['U_max']:.3g}")
    return band, record

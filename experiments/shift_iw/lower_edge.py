"""The lower-edge use of BCA's band (spec version 5, part B), and its room check with an exact oracle (no BCA).

BCA's band q +- U(s, a) is consumed in two ways in this experiment:
  capped pull   BCA's own: the dose 1 + 0.5 U / (U + unit) multiplies the pull toward the recorded command
                (learners_iw.bca_cotrain). The dose saturates, so weighting moves it by only about 0.005-0.03.
  lower edge    this module: a penalty in the actor's objective that grows linearly with the width,
                    L = -lambda * mean_i[ Q1(s_i, pi(s_i)) - k W(s_i, pi(s_i)) ] + mean_i mean_j (pi(s_i)_j - a_ij)^2
                    lambda = alpha / stopgrad(mean_i |Q1(s_i, pi(s_i))|)   (on Q alone; W never enters it), k = 1,
                with W in the paired form W(s_i, a) = [U_i(s_i, a) - U_i(s_i, a_i)]_+: the width at the actor's
                command minus the width at the row's recorded command at the same state, kept only where positive.
                U_i(s, a) = R_i x (max(eta(s, a), 1e-6) x unit) in BCA's float32 arithmetic (calibration/dose's
                frozen_level_dose), eta the frozen scale network. R_i is BCA's own threshold R_u on every row
                (unweighted) or the row's own per-step threshold R(w(c_i)) from the band's table (weighted), so the
                weighted penalty is the unweighted one times R(w(c_i)) / R_u, row by row. A row whose R_i is +inf
                ("cannot certify") uses the largest finite threshold of its table: the widest band the bank still
                certifies, the most caution that stays finite (declared in SPEC.md).
                The gradient reaches the actor only through the scale network's command input: only the actor's
                parameters are differentiated; the scale network, the unit, R_i and U_i(s_i, a_i) are constants.
                Every reader of the band's width here (band_shared, widths_at, logged_widths, band_shape, proxy_check,
                Runner of kind 'band', band_arms) calls learners_iw.require_gate on the band first: scope 'band' for
                the band stage's proxy check, 'bca_lower_edge' (and 'bca_lower_edge_estimated' for the optional arm)
                for the BCA arms.

Protocol: the room check's (learners.ActorOnly). Actor-only training against the frozen gapped host critic, from the
cloned actor (learners.clone), 20,000 steps in chunks of 4,000, one shared stream of batch rows
default_rng([20261016, seed, 35]), Adam 3e-4, batch 256, alpha 2.5; the oracle arms vmapped side by side, the BCA arms
one after another (Runner.run's separate: lanes of one vmapped program can round differently). The cloning term is
learners.actor_step's per-row form with every weight 1, as learners.ActorOnly runs it, so with k = 0 the step is
learners.actor_step's, and a one-arm program equals a one-arm learners.ActorOnly no-caution run bit for bit (tested);
the room check's own 'none' arm, trained five arms side by side, differs from it by lane rounding.

Room check for the lower edge (no BCA): the same objective with W from the exact one-step oracle in the paired form
    W(s_i, a) = [delta*(s_i, a_i) - delta*(s_i, a)]_+,
    delta*(s, a) = r(s, a) + 0.98 min_h Q_h(s', c(s')) - min_h Q_h(s, a),
(s', r) one exact float64 step of thruster_chain.dynamics_jax from the row's state (the float64 of its float32
observation), differentiable in a; c the stall-aware controller; Q_h the frozen critic's two heads at the
float32-rounded normalised state. delta*(s_i, a_i) is computed once per row with the same function. The training
program is traced and run under jax enable_x64 so that this step is float64, while the actor, the critic and Adam stay
float32. (Adam's bias correction is then computed in float64 and rounded, so the k = 0 arm can differ from
learners.ActorOnly's 'none' arm by rounding: the largest parameter difference is recorded, and the lower edge's room
mark compares arms of this one program.)

Restated from outputs/shift/beyond-the-data/stage1/code/harness.py (the cloud's exploratory Stage 1, read from the
repository, never imported by the run): one_step (here with a float64 step), band_terms' main form [d(a_i) - d(a)]_+
(its change A1: an unpaired penalty rewards going just past 80%), actor_step_w (with check 3's per-row cloning form)
and check (e) (oracle_shape, here for any penalty, with this specification's master seed). test_lower_edge compares
them with the harness's own functions.

Check (e) (the harness's change A2), before any arm is trained: on full-throttle-region training states (the hard-water
rows, default_rng([20261016, seed, 65])), thruster 1 of the no-caution actor's command is set along the actor's own
push to each throttle of a grid, the other thrusters staying at the actor's command. The penalty passes when (e1) it
falls from 80% to 81% on at most a given share of the states, (e2) its mean rises from 80% to 81%, (e3) its mean slope
along the push at 80.5% is positive and every slope is finite, and (e4) the best grid throttle for Q1 - W lies past 80%
on fewer states than for Q1 alone. e4 is recorded as not applicable when Q1 alone never prefers past 80% on these
states (nothing to remove there); it then passes only if Q1 - W never does either (version 5, after review: the
harness's strict e4 is kept beside it). Every number is spec.json's lower_edge.check_e (passed in by run.py). A seed
whose oracle penalty fails stops before training, and the record names the sub-checks that failed; on the band's
penalty the check is reported only.

Proxy check (reported beside L2, spec.json lower_edge.proxy_check): at the band stage, before any lower-edge BCA
training, BCA's unweighted paired penalty against the oracle's on every kept training row at the room check's
no-caution actor's command (their ratio of means, BCA's effective k, and their correlations), and check (e) on BCA's
penalty. L2's prediction assumes BCA's unweighted lower edge acts like the oracle at k = 1; this shows how far it does.
"""

import contextlib
import math
import time
from dataclasses import dataclass
from functools import partial

import numpy as np
from scipy import stats

import experiments.shift  # noqa: F401  (gymnasium path and version checks)

import jax
import jax.numpy as jnp
from flax.training.train_state import TrainState
from jax.experimental import enable_x64

from experiments.shift import learners as LR
from experiments.shift import recorder as RC
from experiments.shift import thruster_chain as TC
from experiments.shift_iw import learners_iw as LI
from experiments.shift_iw import world as W

HARNESS = "outputs/shift/beyond-the-data/stage1/code/harness.py"
TAG_SHAPE = 65                 # check (e)'s rows: default_rng([20261016, seed, 65]) (the harness's tag)
KINDS = ("oracle", "band")
METRIC_NAMES = ("actor_loss", "q_mean", "lambda", "bc", "W_mean")
ROOM_ARMS = ("le_none", "le_oracle", "le_oracle_x1.5")   # penalty scales k from spec.json lower_edge.room_check.arms
MATCHED = "le_matched_even"
BAND_ARMS = ("unweighted", "exact_strong", "exact_mild")
BAND_OPTIONAL_ARMS = ("estimated_strong",)          # per step, only when the user asks (scope 'bca_lower_edge_estimated')
SCOPE = "bca_lower_edge"                             # the approval scope of the lower edge's BCA arms (its own go)
SCOPE_OPTIONAL = "bca_lower_edge_estimated"          # and of its optional estimated-weighted arm
SCALE = LI.models()[2]                               # BCA's scale network (calibration/network.Calibrator)
_PROGRAMS = {}
_CONTROLLERS = {}


# ------------------------------------------------------------------------------------------------ the oracle (no BCA)

def controller(physics):
    """The next command of the one-step oracle: the stall-aware controller (recorder.stall_aware)."""
    if physics not in _CONTROLLERS:
        _CONTROLLERS[physics] = RC.stall_aware(physics)
    return _CONTROLLERS[physics]


def one_step(critic_params, shared, idx, a, physics):
    """delta*(s_idx, a) = r + 0.98 min_h Q_h(s', c(s')) - min_h Q_h(s_idx, a), float64 (call under enable_x64): one
    exact float64 step of thruster_chain.dynamics_jax from the row's float64 state, differentiable in a; the frozen
    float32 critic at the float32-rounded normalised state (prepared.normalize's rule) and the float32 command."""
    x = shared["x"][idx]
    a32 = jnp.asarray(a, jnp.float32)
    x1, r = TC.dynamics_jax(x, a32.astype(jnp.float64), physics)
    o1 = (x1.astype(jnp.float32) - shared["mean"]) / shared["std"]
    a1 = controller(physics).jax(x1).astype(jnp.float32)
    qn = LR.CRITIC.apply(critic_params, o1, a1).min(axis=-1)
    q0 = LR.CRITIC.apply(critic_params, shared["obs"][idx], a32).min(axis=-1)
    return r + physics.gamma * qn.astype(jnp.float64) - q0.astype(jnp.float64)


def oracle_penalty(physics, critic_params, shared, idx, obs, a, arm):
    """The paired oracle W(s_i, a) = [delta*(s_i, a_i) - delta*(s_i, a)]_+ (float64): how much more the critic
    overrates the command a than the row's recorded command, by its exact one-step error (the harness's main form)."""
    return jnp.maximum(shared["d_logged"][idx] - one_step(critic_params, shared, idx, a, physics), 0.0)


def _rows_program(name, fn):
    if name not in _PROGRAMS:
        _PROGRAMS[name] = jax.jit(fn)
    return _PROGRAMS[name]


def _in_chunks(fn, n, chunk, *arrays):
    """fn on padded chunks of row-aligned arrays (the last row repeated), concatenated (float64)."""
    out = []
    for i in range(0, n, chunk):
        part = [np.asarray(x)[i:i + chunk] for x in arrays]
        pad = chunk - len(part[0])
        if pad:
            part = [np.concatenate([x, np.repeat(x[-1:], pad, 0)]) for x in part]
        out.append(np.asarray(fn(*part), np.float64)[:chunk - pad])
    return np.concatenate(out) if out else np.zeros(0)


def oracle_shared(prepared, critic_params, physics=W.PHYSICS, chunk=8192):
    """The arrays the oracle reads: the rows' float64 states (prepared.states, exactly float32 values), their normalised
    observations, the normalisation and delta*(s_i, a_i) at every training row's recorded command (float64, the same
    function). Returns (shared, info)."""
    x = np.asarray(prepared.states, np.float64)
    if not np.array_equal(x.astype(np.float32).astype(np.float64), x):
        raise ValueError("training states must be exactly float32 values")
    with enable_x64():
        shared = dict(x=jnp.asarray(x, jnp.float64), obs=jnp.asarray(prepared.train.obs, jnp.float32),
                      mean=jnp.asarray(prepared.obs_mean, jnp.float32), std=jnp.asarray(prepared.obs_std, jnp.float32))
    d = oracle_delta_rows(critic_params, shared, np.arange(prepared.n), prepared.actions, physics, chunk)
    if not np.all(np.isfinite(d)):
        raise ValueError("delta* at the recorded commands is not finite")
    with enable_x64():
        shared["d_logged"] = jnp.asarray(d, jnp.float64)
    return shared, dict(d_logged_sha256=LR.tree_sha256(d), d_logged_mean=float(d.mean()), d_logged_sd=float(d.std()),
                        next_command=controller(physics).name, step="float64 (jax enable_x64)")


def oracle_delta_rows(critic_params, shared, rows, actions, physics=W.PHYSICS, chunk=8192):
    """delta*(s_rows, actions) (float64) with the in-program function."""
    fn = _rows_program(("delta", physics), lambda c, sh, i, a: one_step(c, sh, i, a, physics))
    rows = np.asarray(rows, np.int32)
    with enable_x64():
        return _in_chunks(lambda i, a: fn(critic_params, shared, jnp.asarray(i, jnp.int32),
                                          jnp.asarray(a, jnp.float32)), len(rows), chunk, rows,
                          np.asarray(actions, np.float32))


def oracle64(critic_params, prepared, states, actions, physics=W.PHYSICS):
    """delta* computed outside the training program, as the harness's oracle64 does (dynamics_jax under enable_x64,
    the stall-aware controller, learners.critic_rows at prepared.normalize of s'): the reference the in-program
    function is tested against."""
    x = np.asarray(states, np.float64)
    a = np.asarray(actions, np.float32).astype(np.float64)
    with enable_x64():
        x1, r = TC.dynamics_jax(jnp.asarray(x), jnp.asarray(a), physics)
        a1 = controller(physics).jax(x1)
        x1, r, a1 = np.asarray(x1), np.asarray(r), np.asarray(a1)
    qn = LR.critic_rows(critic_params, prepared.normalize(x1), a1.astype(np.float32))[0].min(1)
    q0 = LR.critic_rows(critic_params, prepared.normalize(x), a.astype(np.float32))[0].min(1)
    return r + physics.gamma * qn - q0


# ------------------------------------------------------------------------------------------------ the band's penalty

def band_width(cal_params, unit, obs, a, R):
    """U = R x (max(eta(s, a), 1e-6) x unit), float32: calibration/dose.frozen_level_dose's arithmetic (as
    learners_iw.per_row_dose), with the frozen scale network eta read at the command a."""
    eta = SCALE.apply(jax.lax.stop_gradient(cal_params), obs, a)
    scale = jnp.maximum(eta, 1e-06) * jax.lax.stop_gradient(unit)
    return jax.lax.stop_gradient(R) * scale


def band_penalty(critic_params, shared, idx, obs, a, arm):
    """The paired lower edge W(s_i, a) = [U_i(s_i, a) - U_i(s_i, a_i)]_+ (float32); U_i(s_i, a_i) is the row's
    constant u_logged."""
    U = band_width(shared["cal_params"], shared["unit"], obs, a, arm["R"][idx])
    return jnp.maximum(U - jax.lax.stop_gradient(arm["u_logged"][idx]), 0.0)


def capped_thresholds(R_rows, finite_pool):
    """Each row's threshold for the lower edge: R_i where it is finite; where R_i is +inf ('cannot certify'), the
    largest finite threshold of finite_pool (the arm's table, or [R_u] for the unweighted arm). Refuses when the pool
    holds no finite threshold (the band certifies nothing). Returns (R per row, the cap, refused rows)."""
    R = np.asarray(R_rows, np.float64)
    if np.any(np.isnan(R)) or np.any(R < 0):
        raise ValueError("thresholds must be non-negative numbers or +inf")
    pool = np.asarray(finite_pool, np.float64).ravel()
    pool = pool[np.isfinite(pool)]
    if not pool.size:
        raise ValueError("the band certifies nothing here: no finite threshold to read a width from")
    cap = float(pool.max())
    refused = np.isinf(R)
    return np.where(refused, cap, R), cap, refused


def scale_change(R_rows, pool, R_u, high=None):
    """Mark L2's scale change of a per-step arm's lower edge against the unweighted one, on kept training rows: the
    penalty at row i is the unweighted one times R_i / R_u (R_i capped as in capped_thresholds), so the change is
    R_i / R_u - 1: its mean, its mean on high-current rows (mask `high`), its extremes, the refused rows and the cap.
    None when R_u is not finite and positive or the pool holds no finite threshold (the band certifies nothing)."""
    R_u = float(R_u)
    if not (math.isfinite(R_u) and R_u > 0):
        return None
    try:
        R, cap, refused = capped_thresholds(R_rows, pool)
    except ValueError:
        return None
    ratio = R / R_u
    out = dict(scale_change=float(ratio.mean() - 1.0), ratio_min=float(ratio.min()), ratio_max=float(ratio.max()),
               refused_rows=int(refused.sum()), largest_finite_threshold=cap, rows=int(len(R)))
    if high is not None:
        high = np.asarray(high, bool)
        out["high_current_scale_change"] = float(ratio[high].mean() - 1.0) if high.any() else None
    return out


def as_float32_thresholds(R):
    R = np.asarray(R, np.float64)
    with np.errstate(over="ignore"):
        R32 = R.astype(np.float32)
    if not np.all(np.isfinite(R32)):
        raise ValueError("a lower-edge threshold must be finite in float32")
    return R32


def band_shared(band, stage=SCOPE):
    """The band's frozen parts the lower edge reads: the scale network's parameters and the unit. Refuses unless the
    band's approval covers `stage` (learners_iw.require_gate: 'band' at the band stage, 'bca_lower_edge' for the BCA
    arms), or it is the synthetic token on registered synthetic arrays."""
    LI.require_gate(band.approval if isinstance(band, LI.Band) else None, band, stage)
    return dict(cal_params=band.state.posterior.cal_params,
                unit=jnp.asarray(band.state.posterior.residual_scale, jnp.float32))


def _band_of(band, prepared, stage):
    """The gate for a band reader: the band's approval covers `stage`, and the band was fitted on `prepared`."""
    shared = band_shared(band, stage)
    if band.data_sha256 != LI.data_sha256(prepared):
        raise ValueError("the band was fitted on other data")
    return shared


def widths_at(band, prepared, actions, R32, *, stage=SCOPE, shared=None, chunk=8192):
    """U_i(s_i, a) at every training row's state and the given commands (float32, the in-program function), after
    the gate (_band_of). `shared` replaces the band's parts (a test's altered scale network); the gate is the band's."""
    own = _band_of(band, prepared, stage)
    shared = own if shared is None else shared
    fn = _rows_program("logged_width", lambda sh, o, a, r: band_width(sh["cal_params"], sh["unit"], o, a, r))
    out = _in_chunks(lambda o, a, r: fn(shared, jnp.asarray(o), jnp.asarray(a), jnp.asarray(r)), prepared.n, chunk,
                     np.asarray(prepared.train.obs, np.float32), np.asarray(actions, np.float32),
                     np.asarray(R32, np.float32))
    return out.astype(np.float32)


def logged_widths(band, prepared, R32, *, stage=SCOPE, shared=None, chunk=8192):
    """U_i(s_i, a_i) at every training row's recorded command (widths_at, with its gate)."""
    return widths_at(band, prepared, prepared.train.action, R32, stage=stage, shared=shared, chunk=chunk)


# ------------------------------------------------------------------------------------------------ the actor step

def actor_step(args, penalty, actor, critic_params, batch, idx, shared, arm):
    """The harness's actor_step_w (learners.actor_step expression by expression, then apply_gradients) with the penalty
    slot: -lambda * mean(Q1 - k W) + mean(pull_i x mean_j (pi - a)^2), lambda on Q1 alone; the cloning term in
    learners.actor_step's per-row form, its weights read from a per-row table of ones as learners.ActorOnly reads its
    table (a constant 1 would let the compiler fold it and round differently). W is cast to the critic's float32
    before it enters the loss. Only the actor's parameters are differentiated. Returns (actor, (loss, Q1 mean,
    lambda, cloning term, W mean))."""

    def loss_fn(params):
        pi = LR.ACTOR.apply(params, batch.obs)
        q = LR.CRITIC.apply(critic_params, batch.obs, pi)[..., 0]
        lmbda = args.alpha / jax.lax.stop_gradient(jnp.abs(q).mean())
        w = penalty(critic_params, shared, idx, batch.obs, pi, arm).astype(q.dtype)
        kw = arm["k"] * w
        bc_w = arm["pull"][idx]
        bc = (jax.lax.stop_gradient(bc_w) * jnp.square(pi - batch.action).mean(axis=-1)).mean()
        return -lmbda * (q - kw).mean() + bc, (q.mean(), lmbda, bc, w.mean())

    (loss, aux), grad = jax.value_and_grad(loss_fn, has_aux=True)(actor.params)
    return actor.apply_gradients(grads=grad), (loss, *aux)


def _loop(args, penalty, critic_params, actor, train, shared, rows, arm):
    def body(a, r):
        batch = jax.tree_util.tree_map(lambda x: x[r], train)
        a, aux = actor_step(args, penalty, a, critic_params, batch, r, shared, arm)
        return a, jnp.stack(aux)

    return jax.lax.scan(body, actor, rows)


def _penalty(kind, physics):
    """The penalty of a kind as penalty(critic params, shared, rows, observations, commands, arm) -> W per row."""
    if kind == "oracle":
        return partial(oracle_penalty, physics)
    if kind == "band":
        return band_penalty
    raise ValueError(f"unknown penalty kind {kind!r}")


class Runner:
    """One data seed's lower-edge arms of one penalty kind ('oracle': float64 step under enable_x64; 'band': float32) on
    the frozen critic, from one start actor and learners.ActorOnly's stream of batch rows, vmapped over the arms.
    arms: [dict(name, k, and for 'band' R (float32, one per training row) and u_logged)]. Kind 'band' needs the Band
    itself and passes only the gate (its approval covers `stage`, 'bca_lower_edge' by default, and it was fitted on
    `prepared`); `shared` is then the band's parts (band_shared), or None to read them from the band."""

    def __init__(self, prepared, start_params, critic_params, settings, rep, kind, shared, *, physics=W.PHYSICS,
                 steps=None, chunk=None, band=None, stage=SCOPE):
        if kind not in KINDS:
            raise ValueError(f"unknown penalty kind {kind!r}")
        if kind == "band":
            own = _band_of(band, prepared, stage)
            shared = own if shared is None else shared
        self.prepared, self.settings, self.rep, self.kind = prepared, settings, int(rep), kind
        self.args = LR.make_args(settings, rep, settings.alpha)
        self.steps = int(settings.actor_only_steps if steps is None else steps)
        self.chunk = int(settings.actor_only_chunk if chunk is None else chunk)
        if self.steps % self.chunk:
            raise ValueError("actor-only steps must be a multiple of the chunk")
        rows = LR.rng_np(settings, rep, LR.TAG_ACTOR_ONLY).integers(prepared.n, size=(self.steps,
                                                                                       settings.batch_size))
        self.rows_sha256 = LR.tree_sha256(rows)
        self.rows = jnp.asarray(rows.reshape(self.steps // self.chunk, self.chunk, settings.batch_size), jnp.int32)
        self.start_params = start_params
        self.start = TrainState.create(apply_fn=LR.ACTOR.apply, params=start_params, tx=LR.adam(settings.lr))
        self.critic = critic_params
        self.shared = shared
        self.physics = physics

    def context(self):
        return enable_x64() if self.kind == "oracle" else contextlib.nullcontext()

    def program(self, n_arms):
        key = ("lower_edge", self.kind, LR._args_key(self.args), int(n_arms), self.chunk, self.physics)
        if key not in _PROGRAMS:
            loop = partial(_loop, self.args, _penalty(self.kind, self.physics))
            _PROGRAMS[key] = jax.jit(jax.vmap(loop, in_axes=(None, 0, None, None, None, 0)))
        return _PROGRAMS[key]

    def arm_tree(self, arms):
        tree = dict(k=jnp.asarray(np.array([float(a["k"]) for a in arms], np.float32)),
                    pull=jnp.ones((len(arms), self.prepared.n), jnp.float32))
        if self.kind == "band":
            for f in ("R", "u_logged"):
                v = np.stack([np.asarray(a[f], np.float32) for a in arms])
                if v.shape != (len(arms), self.prepared.n) or not np.all(np.isfinite(v)):
                    raise ValueError(f"every band arm needs a finite {f} per training row")
                tree[f] = jnp.asarray(v)
        return tree

    def run(self, arms, log=None, separate=False):
        """Train every arm: side by side in one vmapped program, or (separate) one after another, each in its own
        one-arm program. On the CPU, lanes of one vmapped program can round differently (one float32 step), which
        training amplifies; separate arms with equal inputs give equal actors bit for bit, at the same cost. Returns
        ({name: actor params}, info: per-chunk mean metrics, finite flags, the rows' and the start's SHA-256,
        seconds)."""
        names = [a["name"] for a in arms]
        if len(set(names)) != len(names):
            raise ValueError("arm names must be unique")
        if separate and len(arms) > 1:
            params, infos = {}, []
            for a in arms:
                p, i = self.run([a], log=log)
                params.update(p)
                infos.append(i)
            return params, dict(infos[0], arms=[x for i in infos for x in i["arms"]], separate=True,
                                chunk_metrics={i["arms"][0]["name"]: i["chunk_metrics"] for i in infos},
                                finite={k: v for i in infos for k, v in i["finite"].items()},
                                seconds=sum(i["seconds"] for i in infos))
        n = len(arms)
        states = jax.tree_util.tree_map(lambda x: jnp.broadcast_to(x, (n,) + jnp.shape(x)), self.start)
        tree = self.arm_tree(arms)
        metrics, t0 = [], time.perf_counter()
        with self.context():
            program = self.program(n)
            for c in range(self.steps // self.chunk):
                states, aux = program(self.critic, states, self.prepared.train, self.shared, self.rows[c], tree)
                metrics.append(np.asarray(aux, np.float64).mean(1).tolist())
                if log:
                    log(f"lower edge ({self.kind}) rep {self.rep}: {n} arms, step {(c + 1) * self.chunk}/{self.steps}")
            jax.block_until_ready(states.params)
        params = {a["name"]: jax.tree_util.tree_map(lambda x, i=i: x[i], states.params) for i, a in enumerate(arms)}
        info = dict(kind=self.kind, arms=[dict(name=a["name"], k=float(a["k"])) for a in arms], separate=False,
                    chunk_metrics=metrics, metric_names=list(METRIC_NAMES), rows_sha256=self.rows_sha256,
                    start_sha256=LR.tree_sha256(self.start_params), steps=self.steps, chunk=self.chunk,
                    finite={k: bool(LR._finite_tree(p)) for k, p in params.items()},
                    seconds=time.perf_counter() - t0)
        return params, info


def max_param_diff(a, b):
    """The largest absolute difference over every parameter of two actors."""
    return max(float(np.max(np.abs(np.asarray(x, np.float64) - np.asarray(y, np.float64))))
               for x, y in zip(jax.tree_util.tree_leaves(a), jax.tree_util.tree_leaves(b)))


# ------------------------------------------------------------------------------------------------ check (e)

def shape_rows(prepared, rep, n_rows, settings=None):
    """Check (e)'s rows: n_rows full-throttle-region training rows (prepared.region: the hard-water rows),
    default_rng([20261016, seed, 65]).choice, sorted."""
    reg = np.flatnonzero(np.asarray(prepared.region, bool))
    if not len(reg):
        return np.zeros(0, np.int64)
    settings = settings or LI.settings()
    return np.sort(LR.rng_np(settings, rep, TAG_SHAPE).choice(reg, min(int(n_rows), len(reg)), replace=False))


def judge_shape(grid, W_at, Q1_at, slopes, rule):
    """Check (e)'s verdict from the penalty W and Q1 at each grid throttle ({throttle: per-row values}) and the
    penalty's slope along the push at the slope points ({throttle: per-row values}). rule: spec.json
    lower_edge.check_e (knee, just_past, slope_point, max_fall_share). Returns the record (e1-e4 and 'passed')."""
    grid = [float(m) for m in grid]
    knee, past, at = float(rule["knee"]), float(rule["just_past"]), float(rule["slope_point"])
    g = np.asarray(grid)
    best_q = g[np.argmax(np.stack([np.asarray(Q1_at[m], np.float64) for m in grid], 1), 1)]
    best = g[np.argmax(np.stack([np.asarray(Q1_at[m], np.float64) - np.asarray(W_at[m], np.float64)
                                 for m in grid], 1), 1)]
    w_knee, w_past = np.asarray(W_at[knee], np.float64), np.asarray(W_at[past], np.float64)
    fall, rise = float(np.mean(w_past < w_knee)), float(np.mean(w_past > w_knee))
    mean_w = {f"{m:g}": float(np.mean(W_at[m])) for m in grid}
    sl = {f"{m:g}": dict(mean=float(np.mean(v)), negative_share=float(np.mean(np.asarray(v) < 0)),
                         finite_share=float(np.mean(np.isfinite(v)))) for m, v in slopes.items()}
    rec = dict(rows=int(len(w_knee)), grid=grid, mean_W=mean_w, fall_share_80_to_81=fall, rise_share_80_to_81=rise,
               unchanged_share_80_to_81=float(np.mean(w_past == w_knee)), mean_slope_along_push=sl,
               best_past80_share=float(np.mean(best > knee)), q1_alone_best_past80_share=float(np.mean(best_q > knee)),
               mean_W_full_minus_80=float(np.mean(np.asarray(W_at[grid[-1]]) - w_knee)))
    rec["e1_fall_share_ok"] = bool(fall <= float(rule["max_fall_share"]))
    rec["e2_mean_rises"] = bool(np.mean(w_past) > np.mean(w_knee))
    rec["e3_slope_past_knee_positive"] = bool(sl[f"{at:g}"]["mean"] > 0 and
                                              all(v["finite_share"] == 1.0 for v in sl.values()))
    best, alone = rec["best_past80_share"], rec["q1_alone_best_past80_share"]
    rec["e4_fewer_best_past80_than_q1"] = bool(best < alone)          # the harness's e4, kept for comparison
    # e4 is not applicable when the critic alone never prefers past 80% here (nothing to remove): it then passes only
    # if Q1 - W never does either, so the penalty creates no preference past 80%; e1-e3 still check the shape
    rec["e4_not_applicable"] = bool(alone == 0.0)
    rec["e4_passed"] = bool(best < alone or (alone == 0.0 and best == 0.0))
    if rec["e4_not_applicable"]:
        rec["note"] = ("the critic alone never prefers a throttle past 80% on these rows: there is no over-preference "
                       "past 80% to remove (e4 not applicable; it passes only if Q1 - W never prefers past 80% either)")
    rec["failed"] = [why for ok, why in (
        (rec["e1_fall_share_ok"], f"e1: the penalty falls from 80% to 81% on {fall:.3f} of the rows (more than "
                                  f"{float(rule['max_fall_share']):g})"),
        (rec["e2_mean_rises"], "e2: its mean does not rise from 80% to 81%"),
        (rec["e3_slope_past_knee_positive"], "e3: its mean slope at 80.5% is not positive, or a slope is not finite"),
        (rec["e4_passed"], f"e4: Q1 - W prefers a throttle past 80% on {best:.3f} of the rows, not fewer than the "
                           f"critic alone ({alone:.3f})")) if not ok]
    rec["passed"] = not rec["failed"]
    return rec


def penalty_shape(values, slope, critic_params, prepared, actor_params, rows, rule):
    """Check (e) for one penalty: values(rows, actions) -> W per row, slope(rows, actions) -> dW/da per row (n, 4).
    The no-caution actor's command at each row, thruster 1 set along the actor's own push to each grid throttle (the
    knee itself at 80%: thruster_chain.KNEE)."""
    rows = np.asarray(rows, np.int64)
    if not len(rows):
        return dict(rows=0, passed=False, why="no full-throttle-region training rows",
                    failed=["no full-throttle-region training rows to check on"])
    obs = np.asarray(prepared.train.obs)[rows]
    base = LR.actor_rows(actor_params, obs).astype(np.float64)
    sign = np.where(base[:, 0] < 0, -1.0, 1.0)
    knee = float(rule["knee"])

    def at(m):
        aa = base.copy()
        aa[:, 0] = sign * (TC.KNEE if float(m) == knee else float(m))
        return aa.astype(np.float32)

    grid = [float(m) for m in rule["grid"]]
    W_at, Q1_at = {}, {}
    for m in grid:
        aa = at(m)
        W_at[m] = np.asarray(values(rows, aa), np.float64)
        Q1_at[m] = np.asarray(LR.critic_rows(critic_params, obs, aa)[0][:, 0], np.float64)
    slopes = {float(m): np.asarray(slope(rows, at(m)), np.float64)[:, 0] * sign for m in rule["slope_points"]}
    rec = judge_shape(grid, W_at, Q1_at, slopes, rule)
    rec.update(actor_thruster1_past80_share=float(np.mean(np.abs(base[:, 0]) > TC.KNEE)),
               push_negative_share=float(np.mean(sign < 0)))
    return rec


def oracle_shape(critic_params, shared, prepared, actor_params, rep, rule, physics=W.PHYSICS, chunk=8192, rows=None):
    """Check (e) on the oracle penalty [delta*(s_i, a_i) - delta*(s_i, a)]_+ (float64), on shape_rows (or given
    rows)."""
    rows = shape_rows(prepared, rep, rule["rows"]) if rows is None else np.asarray(rows, np.int64)
    val = _rows_program(("oracle_W", physics), lambda c, sh, i, a: oracle_penalty(physics, c, sh, i, None, a, None))
    grad = _rows_program(("oracle_dW", physics), lambda c, sh, i, a: jax.vmap(jax.grad(
        lambda aa, ii: oracle_penalty(physics, c, sh, ii[None], None, aa[None], None)[0]))(a, i))

    def values(r, a):
        with enable_x64():
            return _in_chunks(lambda i, x: val(critic_params, shared, jnp.asarray(i, jnp.int32),
                                               jnp.asarray(x, jnp.float32)), len(r), chunk, r, a)

    def slope(r, a):
        with enable_x64():
            return _in_chunks(lambda i, x: grad(critic_params, shared, jnp.asarray(i, jnp.int32),
                                                jnp.asarray(x, jnp.float32)), len(r), chunk, r, a)

    t0 = time.perf_counter()
    rec = penalty_shape(values, slope, critic_params, prepared, actor_params, rows, rule)
    rec.update(penalty="[delta*(s_i, a_i) - delta*(s_i, a)]_+ (exact one-step oracle, float64)",
               rows_rule=f"default_rng([{W.MASTER_SEED}, {int(rep)}, {TAG_SHAPE}]) choice of {rule['rows']} hard-water "
                         "rows",
               seconds=time.perf_counter() - t0)
    return rec


def band_shape(band, arm, critic_params, prepared, actor_params, rep, rule, *, stage=SCOPE, chunk=8192):
    """Check (e) on one band arm's penalty [U_i(s_i, a) - U_i(s_i, a_i)]_+ (reported only), after the gate
    (_band_of: the band's approval covers `stage`)."""
    shared = _band_of(band, prepared, stage)
    rows = shape_rows(prepared, rep, rule["rows"])
    obs_all = np.asarray(prepared.train.obs, np.float32)
    val = _rows_program("band_W", lambda sh, o, a, R, u: jnp.maximum(
        band_width(sh["cal_params"], sh["unit"], o, a, R) - u, 0.0))
    grad = _rows_program("band_dW", lambda sh, o, a, R, u: jax.vmap(jax.grad(
        lambda aa, oo, rr, uu: jnp.maximum(band_width(sh["cal_params"], sh["unit"], oo[None], aa[None], rr) - uu,
                                           0.0)[0]))(a, o, R, u))
    R = np.asarray(arm["R"], np.float32)
    u = np.asarray(arm["u_logged"], np.float32)

    def values(r, a):
        return _in_chunks(lambda o, x, rr, uu: val(shared, o, x, rr, uu), len(r), chunk, obs_all[r], a, R[r], u[r])

    def slope(r, a):
        return _in_chunks(lambda o, x, rr, uu: grad(shared, o, x, rr, uu), len(r), chunk, obs_all[r], a, R[r], u[r])

    rec = penalty_shape(values, slope, critic_params, prepared, actor_params, rows, rule)
    rec.update(penalty="[U_i(s_i, a) - U_i(s_i, a_i)]_+ (BCA's width, paired)", reported_only=True)
    return rec


def proxy_check(band, prepared, critic_params, none_params, R_u, rep, rule, *, physics=W.PHYSICS, stage="band",
                chunk=8192):
    """Reported beside L2 (spec.json lower_edge.proxy_check), at the band stage on the frozen gapped band, before any
    lower-edge BCA training: does BCA's unweighted lower edge act like the oracle penalty at k = 1, as L2's prediction
    assumes? On every kept training row, at the room check's no-caution actor's command a = pi_none(s_i): BCA's paired
    penalty at R_u, [U(s_i, a) - U(s_i, a_i)]_+ (float32), and the oracle's, [delta*(s_i, a_i) - delta*(s_i, a)]_+
    (float64 step); the ratio of their means (BCA's effective k against the oracle's 1), their Pearson and Spearman
    correlations and the share of rows where each is positive; and check (e) on BCA's penalty with the same actor and
    rule. The gate is the band's (scope 'band'). No training, no return."""
    t0 = time.perf_counter()
    _band_of(band, prepared, stage)
    R_u = float(R_u)
    if not (math.isfinite(R_u) and R_u > 0):
        return dict(skipped=True, why="the unweighted threshold is not finite and positive: the band certifies nothing")
    R32 = as_float32_thresholds(np.full(prepared.n, R_u))
    obs = np.asarray(prepared.train.obs, np.float32)
    a = np.asarray(LR.actor_rows(none_params, obs), np.float32)
    u_rec = logged_widths(band, prepared, R32, stage=stage, chunk=chunk)
    u_pi = widths_at(band, prepared, a, R32, stage=stage, chunk=chunk)
    w_band = np.maximum(u_pi - u_rec, np.float32(0.0)).astype(np.float64)
    o_shared, info = oracle_shared(prepared, critic_params, physics, chunk)
    d_pi = oracle_delta_rows(critic_params, o_shared, np.arange(prepared.n), a, physics, chunk)
    w_oracle = np.maximum(np.asarray(o_shared["d_logged"], np.float64) - d_pi, 0.0)
    mb, mo = float(w_band.mean()), float(w_oracle.mean())
    varied = bool(w_band.std() > 0 and w_oracle.std() > 0)
    check_e = band_shape(band, dict(R=R32, u_logged=u_rec), critic_params, prepared, none_params, rep, rule,
                         stage=stage, chunk=chunk)
    return dict(skipped=False, rows=int(prepared.n), actor="the room check's no-caution ('none') actor",
                R="R_u on every row (the unweighted lower edge at k = 1)", R_u=R_u, band_mean=mb, oracle_mean=mo,
                band_over_oracle=mb / mo if mo > 0 else None,
                pearson=float(np.corrcoef(w_band, w_oracle)[0, 1]) if varied else None,
                spearman=float(stats.spearmanr(w_band, w_oracle)[0]) if varied else None,
                band_positive_share=float(np.mean(w_band > 0)), oracle_positive_share=float(np.mean(w_oracle > 0)),
                band_check_e=check_e, oracle=info, seconds=time.perf_counter() - t0)


# ------------------------------------------------------------------------------------------------ the room check

@dataclass(frozen=True)
class RoomRule:
    """spec.json lower_edge.room_check and check_e, as run.py reads them: the penalty scale k of each oracle arm
    ({name: k} over ROOM_ARMS), the even rungs added above the room check's ladder, and check (e)'s rule."""

    k: dict
    even_extra: tuple
    check_e: dict


def room_check(prepared, critic_params, room_params, room_record, settings, rep, rule, *, physics=W.PHYSICS,
               log=None):
    """The lower edge's room check (no BCA), one seed: clone the actor (learners.clone, the room check's start), check
    (e) on the oracle penalty with the room check's no-caution actor (a failure stops the seed: no arm is trained),
    then the oracle arms side by side in one float64-step program ('le_none' k = 0, 'le_oracle' k = 1 and
    'le_oracle_x1.5' k = 1.5, the scales from spec.json), then even pull matched blind in strength to 'le_oracle'
    (learners.match_even on the room check's rungs x1.1, x1.25, x1.5 and this check's extra rungs, never
    extrapolated; reported only). Strengths are measured against the room check's
    'none' actor. No return is computed here. Returns ({name: actor params}, record)."""
    t0 = time.perf_counter()
    clone_params, clone_rec = LR.clone(prepared, settings, rep, log=log)
    reproduced = clone_rec["actor_sha256"] == (room_record.get("clone") or {}).get("actor_sha256")
    shared, sinfo = oracle_shared(prepared, critic_params, physics)
    none = room_params["none"]
    check_e = oracle_shape(critic_params, shared, prepared, none, rep, rule.check_e, physics)
    record = dict(kind="shift-iw-room-lower-edge", rep=int(rep), clone=clone_rec,
                  clone_reproduces_room_check=reproduced, oracle=sinfo, check_e=check_e,
                  k={k: float(v) for k, v in rule.k.items()}, even_extra=[float(m) for m in rule.even_extra])
    if not check_e["passed"]:
        failed = "; ".join(check_e.get("failed") or []) or "its verdict"
        record.update(stopped=True, why=f"check (e) on the oracle penalty failed ({failed}); no arm is trained",
                      names=[], seconds=time.perf_counter() - t0)
        return {}, record
    runner = Runner(prepared, clone_params, critic_params, settings, rep, "oracle", shared, physics=physics)
    if sorted(rule.k) != sorted(ROOM_ARMS):
        raise ValueError(f"the oracle arms are {ROOM_ARMS}")
    params, info = runner.run([dict(name=n, k=rule.k[n]) for n in ROOM_ARMS], log=log)
    check_b = max_param_diff(params["le_none"], none)
    S = {n: LR.strength(p, none, prepared.eval_obs) for n, p in params.items()}
    # even pull: the room check's rungs (their strengths from its record) and this check's extra rungs
    ao = LR.ActorOnly(prepared, clone_params, settings, rep, alpha=settings.alpha)
    extra = [LR.Arm(f"even_x{m:g}", "even", float(m)) for m in rule.even_extra]
    stacked, ainfo = ao.run(extra, [critic_params], log=log)
    for i, a in enumerate(extra):
        params[f"le_{a.name}"] = ao.arm_params(stacked, i)
        S[f"le_{a.name}"] = LR.strength(params[f"le_{a.name}"], none, prepared.eval_obs)
    room_S = room_record.get("strengths") or {}
    ladder = sorted([(float(n[len("even_x"):]), float(room_S[n])) for n in room_S if n.startswith("even_x")] +
                    [(a.value, S[f"le_{a.name}"]) for a in extra])
    target = S["le_oracle"]
    if not info["finite"]["le_oracle"] or not (math.isfinite(target) and target > 0):
        match = dict(multiplier=None, knob=None, ladder_multiplier=None, skipped=True,
                     reason="the oracle arm failed or has no finite positive strength")
    else:
        match, evaluated = LR.match_even(ao, [critic_params], none, prepared.eval_obs, ladder, target, settings,
                                         log=log)
        m = match["multiplier"]
        if m is not None:
            rung = match["ladder_multiplier"]
            if rung is None:
                params[MATCHED] = evaluated[m]
            elif f"le_even_x{rung:g}" in params:
                params[MATCHED] = params[f"le_even_x{rung:g}"]
            else:
                params[MATCHED] = room_params[f"even_x{rung:g}"]
            S[MATCHED] = LR.strength(params[MATCHED], none, prepared.eval_obs)
    record.update(stopped=False, names=sorted(params), strengths=S, ladder=ladder, match=match,
                  check_b=dict(rule="the k = 0 arm of this float64-step program against the room check's 'none' arm "
                                    "(learners.ActorOnly, float32): largest parameter difference (reported)",
                               max_param_diff=check_b),
                  training=info, even_training=dict(rows_sha256=ainfo["rows_sha256"],
                                                    start_sha256=ainfo["start_sha256"], failed=ainfo["failed"]),
                  seconds=time.perf_counter() - t0)
    return params, record


def room_template(names):
    return {k: LR.actor_template() for k in names}


# ------------------------------------------------------------------------------------------------ the BCA arms

def band_arms(prepared, band, critic_params, arms, k, settings, rep, approval, *, none_params=None, check_e_rule=None,
              stage=SCOPE, log=None):
    """The lower edge's BCA arms (approval scope 'bca_lower_edge', its own go, separate from the capped pull's 'bca';
    'bca_lower_edge_estimated' also for the optional arm), one seed: actor-only against the frozen gapped critic from
    the cloned actor, the arms one after another, each in its own float32 program (Runner.run separate: equal
    thresholds give equal actors bit for bit). arms: [dict(name, R (one threshold per kept training row, +inf
    allowed), pool (the finite thresholds of its table, or [R_u]))]; k: the penalty scale (spec.json lower_edge.k, 1).
    Each arm's R is capped (capped_thresholds), its U_i(s_i, a_i) computed, and (with none_params and check_e_rule)
    check (e) is run on its penalty and reported. Returns ({name: actor params}, record)."""
    LI.require_gate(approval, prepared, SCOPE)
    if stage != SCOPE:
        LI.require_gate(approval, prepared, stage)
    if band.data_sha256 != LI.data_sha256(prepared):
        raise ValueError("the band was fitted on other data")
    t0 = time.perf_counter()
    shared = _band_of(band, prepared, SCOPE)
    specs, per_arm = [], {}
    for a in arms:
        R, cap, refused = capped_thresholds(a["R"], a["pool"])
        R32 = as_float32_thresholds(R)
        u = logged_widths(band, prepared, R32, stage=SCOPE)
        specs.append(dict(name=a["name"], k=float(k), R=R32, u_logged=u))
        per_arm[a["name"]] = dict(refused_rows=int(refused.sum()), largest_finite_threshold=cap,
                                  R_min=float(R.min()), R_mean=float(R.mean()), R_max=float(R.max()),
                                  u_logged_mean=float(u.mean()), R_sha256=LR.tree_sha256(R32))
    clone_params, clone_rec = LR.clone(prepared, settings, rep, log=log)
    if none_params is not None and check_e_rule is not None:
        for s in specs:
            per_arm[s["name"]]["check_e"] = band_shape(band, s, critic_params, prepared, none_params, rep,
                                                       check_e_rule, stage=SCOPE)
    runner = Runner(prepared, clone_params, critic_params, settings, rep, "band", shared, band=band, stage=SCOPE)
    params, info = runner.run(specs, log=log, separate=True)
    record = dict(kind="shift-iw-bca-lower-edge", rep=int(rep), k=float(k), arms=per_arm, clone=clone_rec,
                  training=info,
                  unit=float(shared["unit"]), refused_rule="a row whose R is +inf uses the largest finite threshold of "
                                                           "its table (most caution that stays finite)",
                  seconds=time.perf_counter() - t0)
    return params, record

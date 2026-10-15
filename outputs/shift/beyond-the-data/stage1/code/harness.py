"""EXPLORATORY (Stage 1 of outputs/shift/beyond-the-data/PLAN.md, sections 4.0, 4.1 and 7). Outside the gate; changes no
pre-registered rule. NO BCA: no band, calibrator, threshold, dose or BCA update is called. Nothing from
algorithms/td3_bc_bca.py or learners.bca_* is called; from calibration/ only calibration/bank.stratified_bank runs,
because the gate's data loader (experiments/shift/dataset.py, through acceptance.load_seed) calls it to regenerate the
log's calibration-bank split. It is a NumPy row sampler, not a BCA signal. Every penalty here is a stand-in or an
oracle.

The room test of the lower-edge channel. Check 2's actor-only protocol (the frozen fixed-controller gapped critic of the
stalling world, whose scored policy is the stall-aware controller; start from the saved cloned actor; alpha 5; 20,000
actor steps in 4,000-step chunks; check 2's stream of batch rows), with one new slot in the actor's loss:

    loss = -lambda * mean_i[ Q1(s_i, pi(s_i)) - kW(s_i, pi(s_i)) ] + (1 + nu) * mean_ij (pi(s_i)_j - a_ij)^2
    lambda = alpha / stopgrad( mean_i |Q1(s_i, pi(s_i))| )        (on Q alone; kW never enters it)

kW is the arm's penalty times its coefficient. The penalty's own parameters (the frozen critic, a random network, the
simulator) get no gradient: only the actor's parameters are differentiated, so the penalty acts through the command
pi(s) alone. With kW = 0 and nu = 0 the loss is learners.actor_step's unweighted expression term for term.

What is copied from the gate (experiments/shift/learners.py, unchanged in substance): actor_step (as
actor_step_w, with the slot added), _actor_only_loop (as _room_loop), ActorOnly (as RoomRunner: the same batch-row
stream default_rng([20261015, rep, 35]), the same start state and optimiser, the arms vmapped side by side). The gate's
own code is imported read-only and only its public functions are called, unchanged: the networks (learners.ACTOR,
learners.CRITIC), make_args, adam, rng_np, tree_sha256, strength, actor_rows, critic_rows, save_tree, load_tree,
ActorOnly / Arm (for the reproduction check only), the matcher experiments/signal/glq_match.match_parallel, and the
gate's loaders and evaluation in acceptance.py (load_prereg, guard, load_seed, get_critic(train=False), get_host
(train=False), finished, anchors_of, _evaluate). Gate artifacts are read with the pattern of
outputs/shift/stop-report/cards/cards_outside_gate.py. Nothing is written under experiments/ or runs/shift/.

The one-step oracle, BCA's own residual computed exactly at any command:
    delta*(s, a) = r(s, a) + 0.98 * min_h Q_h(s', c(s')) - min_h Q_h(s, a)
(s', r) from one step of thruster_chain.dynamics_jax from the row's state (float64 of the float32 observation, which
is exactly representable in float32), differentiable in a; c the critic's scored controller (recorder.stall_aware);
Q_h the frozen critic's two heads at the float32-rounded, normalised state. Inside the float32 training program the
step runs in float32 (see SPEC deviations); oracle64() gives the float64 version for the checks.

The main oracle ("perfect band", amended before any run; spec.json changes_before_any_run) measures delta* from the
row's own recorded command a_i:
    W(s_i, a) = [delta*(s_i, a_i) - delta*(s_i, a)]_+
the critic's extra overrating of a over the recorded command, by its exact one-step error (the one-step counterpart of
the plan's long-run harm arm [e(s, a) - e(s, a_i)]_+). delta*(s_i, a_i) is a constant per row (shared['d_logged']).
The plan's literal |delta*(s, a)| runs beside it as a reported arm: it falls just past the 80% knee at most
full-throttle-region states (delta* is about +0.17 at 80% and crosses zero near 86%), so it rewards the first points
of over-throttle.
"""

import hashlib
import json
import math
import os
import sys
import time
from dataclasses import dataclass, field, asdict
from functools import partial
from pathlib import Path

import numpy as np

ROOT = Path("/home/user/bca")
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import experiments.shift.acceptance as A  # noqa: E402  (puts gymnasium on the path)
import experiments.shift.learners as L  # noqa: E402
import experiments.shift.recorder as RC  # noqa: E402
import experiments.shift.thruster_chain as TC  # noqa: E402
import experiments.shift.truth as TR  # noqa: E402
from experiments.signal import glq_match as M  # noqa: E402

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import flax.linen as nn  # noqa: E402
from flax.training.train_state import TrainState  # noqa: E402

OUT = ROOT / "runs" / "shift_explore" / "room"
SPEC_PATH = OUT / "spec.json"
HERE = Path(__file__).resolve().parent
PLAN_PATH = ROOT / "outputs" / "shift" / "beyond-the-data" / "PLAN.md"
LABEL = ("EXPLORATORY: Stage 1 room test of the lower-edge channel (plan sections 4.0, 4.1, 7), computed outside the "
         "gate on saved gate artifacts; not a gated output; changes no pre-registered rule; no BCA band, calibrator, "
         "threshold, dose or BCA update was used (stand-ins and oracles only)")

ALPHA = 5.0
GAMMA = 0.98
SEEDS = (0, 1, 2, 3, 4)
FUTILITY_SEEDS = (0, 1)
STEPS = 20_000
CHUNK = 4_000
CHECK_STEPS = 400
W0_TOL = 1e-5                 # check (b): the new program with W = 0 against check 2's actor, every parameter
ORACLE_PRECISION_TOL = 1e-3   # check (c): |float32 in-program delta* - float64 delta*| at every training row
HARM_HORIZON = 250
SLOPE_ROWS = 4096
TAG_PERM, TAG_RANDOM, TAG_SLOPE = 61, 62, 63
EVEN_LADDER = (0.1, 0.25, 0.5, 1.0, 2.0)                  # nu (plan 4.1)
SLOPE_LADDER = (0.1, 0.3, 1.0, 3.0, 10.0)                  # slope-normalised knobs (push less, random surfaces)
EXTEND_FACTOR = {"even_pull": 2.0, "push_less": math.sqrt(10.0), "random_1": math.sqrt(10.0),
                 "random_2": math.sqrt(10.0)}
MAX_EXTENSIONS = 3
GATING = ("even_pull", "push_less", "random_1", "random_2")
HINGE_SCALE = 100.0
RANDOM_HIDDEN, RANDOM_DEPTH = 64, 2
PREVIOUS_SPEC_SHA256 = "7a3abb1ebd2a553085228be6463f36de4ae3d0ee94e83d0c08989f917e229782"   # spec v1 (superseded)

# The band kind's terms, with dd = delta*(s, a) - delta*(s, a_i) and d = delta*(s, a):
#   over_rel  [-dd]_+   the main oracle (the critic's extra overrating of a over the recorded command)
#   abs_rel   |dd|      the symmetric band measured from the recorded command (reported)
#   under_rel [dd]_+    the same information on the wrong side (reported)
#   abs_lit   |d|       the plan's literal perfect band (reported)
#   over_lit  [-d]_+    the plan's literal overrating half (reported)
BAND_TERMS = ("over_rel", "abs_rel", "under_rel", "abs_lit", "over_lit")
MAIN_BAND = (1.0, 0.0, 0.0, 0.0, 0.0)

# Check (e), the shape of the main oracle's penalty past the knee (critic surface only, no return):
SHAPE_ROWS = 2000                                  # full-throttle-region training states, default_rng([.., seed, 65])
TAG_SHAPE = 65
SHAPE_GRID = (0.70, 0.75, 0.78, 0.79, 0.80, 0.805, 0.81, 0.82, 0.83, 0.84, 0.85, 0.86, 0.88, 0.90, 0.95, 1.00)
SHAPE_MAX_FALL_SHARE = 0.25     # (e1) W(81%) < W(80%) on at most this share of the rows
SHAPE_GRAD_POINTS = (0.79, 0.80, 0.805, 0.81, 0.85, 0.90)   # reported: mean dW/d(push) of the in-program penalty

# The code manifest (runs/shift_explore/room/code_manifest.json): every file listed is enforced by check_code().
MANIFEST_NAME = "code_manifest.json"
CODE_FILES = ("harness.py", "run_seed.py", "launch.sh", "write_spec.py", "write_manifest.py", "checks_seed0.py",
              "summarise_checks.py", "dryrun.py")

# Every repository file whose code the harness imports (the gate's own hashed set) plus the pattern it follows.
GATE_FILES = tuple(A.NEW_FILES) + tuple(A.REUSED_FILES) + (
    "outputs/shift/stop-report/cards/cards_outside_gate.py", "runs/shift/preregistration.json",
    "runs/shift/preregistration.json.sha256")


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


# ------------------------------------------------------------------------------------------------ JSON (gate pattern)

def guard_out(path):
    """Refuse any output outside runs/shift_explore/room (never runs/shift, never experiments/)."""
    p = Path(path).resolve()
    if OUT.resolve() not in p.parents and p != OUT.resolve():
        raise SystemExit(f"refused: {p} is outside {OUT}")
    for protected in (ROOT / "runs" / "shift", ROOT / "experiments"):
        if p == protected.resolve() or protected.resolve() in p.parents:
            raise SystemExit(f"refused: {p} is inside {protected}")
    return p


def write_json(path, payload):
    """acceptance.write_json's format (strict sorted JSON, '<name>.sha256' sidecar), restricted to OUT."""
    path = guard_out(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    text = A.canonical(payload)
    path.write_text(text, encoding="utf8")
    digest = hashlib.sha256(text.encode("utf8")).hexdigest()
    path.with_name(path.name + ".sha256").write_text(f"{digest}  {path.name}\n", encoding="utf8")
    return digest


def read_json(path):
    """A JSON written by write_json; None when missing; ValueError when its sidecar does not match."""
    path = Path(path)
    if not path.exists():
        return None
    text = path.read_text(encoding="utf8")
    side = path.with_name(path.name + ".sha256")
    if not side.exists() or side.read_text(encoding="utf8").split()[0] != hashlib.sha256(text.encode("utf8")).hexdigest():
        raise ValueError(f"{path}: its SHA-256 differs from its sidecar")
    return json.loads(text)


def file_sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def gate_file_hashes():
    return {rel: file_sha256(ROOT / rel) for rel in GATE_FILES}


def code_hashes(names=CODE_FILES):
    return {n: file_sha256(HERE / n) for n in names}


def check_code(spec_sha):
    """Refuse unless the code manifest exists, matches its sidecar, names the current spec, and every file it lists
    (here and in its copy under OUT/code/) still has the SHA-256 it recorded. Returns (manifest, its SHA-256)."""
    path = OUT / MANIFEST_NAME
    man = read_json(path)
    if man is None:
        raise SystemExit(f"refused: no code manifest at {path}; write it (write_manifest.py) before any phase")
    if man.get("spec_sha256") != spec_sha:
        raise SystemExit("refused: the code manifest names another spec; write a new, labelled manifest")
    now = code_hashes(tuple(man["files"]))
    bad = [n for n, h in man["files"].items() if now[n] != h]
    copies = [n for n, h in man["files"].items()
              if not (OUT / "code" / n).exists() or file_sha256(OUT / "code" / n) != h]
    if bad or copies:
        raise SystemExit("refused: the code differs from the manifest (" + ", ".join(bad + [f"code/{n}" for n in copies])
                         + "); any change needs a new, labelled manifest written before the next phase")
    return man, file_sha256(path)


def save_tree(path, tree):
    return L.save_tree(guard_out(path), tree)


# ------------------------------------------------------------------------------------------------ gate artifacts

class SeedContext:
    """One gate seed's verified gate artifacts, read through the gate's own loaders (cards_outside_gate.py pattern):
    A.load_prereg(A.refuse_protected(A.RUNS_DEFAULT)), A.load_seed, the saved cloned actor, the fixed-controller
    critics through A.get_critic(..., train=False) and check 2's / check 3's saved actors with their SHA-256."""

    def __init__(self, rep, worlds=("stall",)):
        t = time.perf_counter()
        self.prereg = A.load_prereg(A.refuse_protected(A.RUNS_DEFAULT))
        A.guard(self.prereg)
        self.cfg = self.prereg.config
        self.learn = self.cfg.learn
        outcome = A.gate_outcome(self.prereg)
        self.variant = outcome["variant"]
        last = outcome.get("alpha") or A._last_alpha_run(self.prereg, self.variant)
        if float(last) != ALPHA:
            raise SystemExit(f"refused: the gate's last alpha is {last}, the room test is fixed at alpha {ALPHA}")
        self.rep = int(rep)
        self.sd = {w: A.load_seed(self.prereg.runs, self.cfg, self.variant, w, rep) for w in worlds}
        self._prepared = {}
        self.load_seconds = time.perf_counter() - t

    def prepared(self, world="stall"):
        if world not in self._prepared:
            self._prepared[world] = self.sd[world].prepared("gap", self.learn)
        return self._prepared[world]

    def critic(self, world="stall", removal="gap"):
        rec, params = A.get_critic(self.prereg, self.sd[world], removal, "stall_aware", train=False)
        if params is None:
            raise A.GateError(f"the {world}/{removal} critic of seed {self.rep} failed in the gate")
        return params, rec

    def clone(self, world="stall"):
        """get_clone's read path only (never its training path)."""
        d = A._variant_root(self.prereg, "clone", self.variant) / world / f"seed{self.rep}"
        rec = A.finished(d / "record.json", self.prereg)
        if rec is None:
            raise A.GateError(f"no saved cloned actor at {d}")
        return L.load_tree(d / "actor.msgpack", L.actor_template(), rec["result"]["file_sha256"]), rec

    def h2(self):
        d = A._variant_root(self.prereg, "harm_return", self.variant) / f"alpha{ALPHA:g}" / f"seed{self.rep}"
        rec = A.finished(d / "values.json", self.prereg)
        if rec is None:
            raise A.GateError(f"check 2 has no verified values at {d}")
        acts = L.load_tree(d / "actors.msgpack", {k: L.actor_template() for k in rec["actor_names"]},
                           rec["actors_sha256"])
        return acts, rec, d

    def h3(self):
        d = A._variant_root(self.prereg, "room", self.variant) / f"alpha{ALPHA:g}" / f"seed{self.rep}"
        blind = A.finished(d / "room_blind.json", self.prereg)
        vals = A.finished(d / "values.json", self.prereg)
        if blind is None:
            raise A.GateError(f"check 3 has no verified lock at {d}")
        acts = L.load_tree(d / "actors.msgpack", {k: L.actor_template() for k in blind["names"]},
                           blind["actors_sha256"])
        return acts, blind, vals, d

    def host(self):
        rec, state = A.get_host(self.prereg, self.sd["stall"], "gap", ALPHA, "plain", train=False)
        if state is None:
            raise A.GateError(f"check 3's host of seed {self.rep} failed in the gate")
        return state, rec

    def anchors(self, world="stall"):
        return A.anchors_of(self.prereg, self.variant, world)


# ------------------------------------------------------------------------------------------------ penalty surfaces

class RandomSurface(nn.Module):
    """A random network of the shape of BCA's scale network (calibration/network.Calibrator with state_dep=True,
    hidden 64, depth 2: Dense-ReLU-Dense-ReLU-Dense(1), softplus + 0.01), restated here (calibration/ is not imported
    or called). Inputs as td3_bc_bca feeds it: the normalised observation / (1 + 0.001) and the command."""

    hidden: int = RANDOM_HIDDEN
    depth: int = RANDOM_DEPTH

    @nn.compact
    def __call__(self, obs, action):
        x = jnp.concatenate([obs / (1.0 + 0.001), action], axis=-1)
        for _ in range(self.depth):
            x = nn.relu(nn.Dense(self.hidden)(x))
        return jax.nn.softplus(nn.Dense(1)(x).squeeze(-1)) + 0.01


RANDOM_NET = RandomSurface()


def random_surface_params(rep, j):
    seed = int(L.rng_np(L.LearnSettings(), rep, TAG_RANDOM, j).integers(2 ** 31 - 1))
    return RANDOM_NET.init(jax.random.PRNGKey(seed), jnp.zeros(TC.DX), jnp.zeros(TC.DA)), seed


def controller_jax(name, physics):
    if name == "stall_aware":
        return RC.stall_aware(physics).jax
    if name == "rising_thrust":
        return RC.rising_thrust(physics).jax
    raise ValueError(name)


@dataclass(frozen=True)
class ProgramConfig:
    """Static parts of a training program: the world's physics, the next command of the one-step oracle
    ('stall_aware', 'rising_thrust' or 'host_actor'), the long-run horizon and the form of the cloning term (the
    gate's unweighted expression, as check 2 ran, or its per-row weighted expression, as check 3 ran)."""

    physics: object = TC.STALL
    next_command: str = "stall_aware"
    horizon: int = HARM_HORIZON
    bc_form: str = "unweighted"   # 'unweighted': check 2's cloning expression; 'weighted': check 3's per-row form


def one_step(critic, shared, cfg, idx, pi):
    """delta*(s_idx, pi) = r + 0.98 min_h Q_h(s', c(s')) - min_h Q_h(s_idx, pi) (float32 in the training program).
    shared['raw32'][idx] is the row's state (float32, exact), shared['obs'][idx] its normalised observation."""
    x = shared["raw32"][idx]
    o = shared["obs"][idx]
    x1, rew = TC.dynamics_jax(x, pi, cfg.physics)
    o1 = (x1 - shared["mean"]) / shared["std"]
    if cfg.next_command == "host_actor":
        a1 = L.ACTOR.apply(shared["host_actor"], o1)
    else:
        a1 = controller_jax(cfg.next_command, cfg.physics)(x1)
    qn = L.CRITIC.apply(critic, o1, a1).min(axis=-1)
    q0 = L.CRITIC.apply(critic, o, pi).min(axis=-1)
    return rew + cfg.physics.gamma * qn - q0


def controller_value(x, a, cfg):
    """Q^c_H(x, a) of the stall-aware controller by an H-step rollout through dynamics_jax (truth._jax_q's rule, in
    the dtype of x), differentiable in a."""
    ctrl = controller_jax("stall_aware", cfg.physics)
    gamma = cfg.physics.gamma
    x1, r0 = TC.dynamics_jax(x, a, cfg.physics)

    def body(carry, _):
        xx, acc, disc = carry
        disc = disc * gamma
        xx2, rr = TC.dynamics_jax(xx, ctrl(xx), cfg.physics)
        return (xx2, acc + disc * rr, disc), None

    (_, acc, _), _ = jax.lax.scan(body, (x1, r0, jnp.asarray(1.0, x.dtype)), None, length=int(cfg.horizon) - 1)
    return acc


def penalty(kind, cfg, critic, shared, arm, rows, pi, q):
    """kW(s_rows, pi) per row (B,), the coefficient included. q = Q1(s_rows, pi) of the main term."""
    c = arm["coef"]
    if kind == "plain":
        return jnp.zeros_like(q)
    if kind == "band":
        return band_terms(c, one_step(critic, shared, cfg, rows, pi), shared["d_logged"][rows])
    if kind == "band_shuffled":
        j = shared["perm"][rows]
        return c[0] * jnp.maximum(shared["d_logged"][j] - one_step(critic, shared, cfg, j, pi), 0.0)
    if kind == "harm":
        e = q - controller_value(shared["raw32"][rows], pi, cfg)
        return c[0] * jnp.maximum(e - shared["e_logged"][rows], 0.0)
    if kind == "push_less":
        return c[0] * jnp.square(pi).mean(axis=-1)
    if kind == "random":
        return c[0] * RANDOM_NET.apply(arm["net"], shared["obs"][rows], pi)
    if kind == "hinge":
        hi, lo = shared["hi"], shared["lo"]
        return c[0] * HINGE_SCALE * (jnp.square(jnp.maximum(pi - hi, 0.0)) +
                                     jnp.square(jnp.maximum(lo - pi, 0.0))).sum(axis=-1)
    raise ValueError(f"unknown kind {kind!r}")


def band_terms(c, d, d_rec):
    """The band kind's penalty (BAND_TERMS order) from d = delta*(s, a) and d_rec = delta*(s, a_i); works on jax
    and numpy arrays."""
    xp = np if isinstance(d, np.ndarray) else jnp
    dd = d - d_rec
    return (c[0] * xp.maximum(-dd, 0.0) + c[1] * xp.abs(dd) + c[2] * xp.maximum(dd, 0.0) + c[3] * xp.abs(d)
            + c[4] * xp.maximum(-d, 0.0))


KINDS = ("plain", "band", "band_shuffled", "harm", "push_less", "random", "hinge")
N_COEF = {"plain": 1, "band": len(BAND_TERMS), "band_shuffled": 1, "harm": 1, "push_less": 1, "random": 1,
          "hinge": 1}


# ------------------------------------------------------------------------------------------------ the actor step

def actor_step_w(args, kind, cfg, actor, critic_params, batch, rows, shared, arm):
    """learners.actor_step (td3_bc's _actor_loss_fn expression by expression, then apply_gradients) with the penalty
    slot: -lambda * mean(q - kW) + (1 + nu) * mean (pi - a)^2, lambda on q alone. Returns (actor, (loss, q mean,
    lambda, bc term, kW mean))."""

    def _actor_loss_fn(params, critic_params):
        pi = L.ACTOR.apply(params, batch.obs)
        q = L.CRITIC.apply(critic_params, batch.obs, pi)[..., 0]
        lmbda = args.alpha / jax.lax.stop_gradient(jnp.abs(q).mean())
        kw = penalty(kind, cfg, critic_params, shared, arm, rows, pi, q)
        if cfg.bc_form == "unweighted":
            bc = jnp.square(pi - batch.action).mean() * arm["pull"]
        else:
            bc_w = jnp.broadcast_to(arm["pull"], q.shape)
            bc = (jax.lax.stop_gradient(bc_w) * jnp.square(pi - batch.action).mean(axis=-1)).mean()
        return (-lmbda * (q - kw).mean() + bc, (q.mean(), lmbda, bc, kw.mean()))

    grad_fn = jax.value_and_grad(_actor_loss_fn, has_aux=True)
    (a_loss, aux), a_grad = grad_fn(actor.params, critic_params)
    return actor.apply_gradients(grads=a_grad), (a_loss, *aux)


def _room_loop(args, kind, cfg, critic_params, actor, train, shared, rows, arm):
    def body(a, r):
        batch = jax.tree_util.tree_map(lambda x: x[r], train)
        a, aux = actor_step_w(args, kind, cfg, a, critic_params, batch, r, shared, arm)
        return a, jnp.stack(aux)

    return jax.lax.scan(body, actor, rows)


METRIC_NAMES = ["actor_loss", "q_mean", "lambda", "bc", "kW_mean"]
_PROGRAMS = {}


@dataclass(frozen=True)
class RoomArm:
    """One arm. kind: one of KINDS. coef: the kind's coefficients (band: one k per BAND_TERMS entry; others: (k,)).
    pull: 1 + nu. surface: random surface index (1 or 2). family / knob: the matched family and its ladder knob, for
    the record."""

    name: str
    kind: str
    coef: tuple = (0.0,)
    pull: float = 1.0
    surface: int = 0
    family: str = None
    knob: float = None

    def record(self):
        return asdict(self)


class RoomRunner:
    """learners.ActorOnly with the penalty slot: one data seed's arms on one frozen critic, from one start actor and
    one shared stream of batch rows (default_rng([20261015, rep, 35]), check 2's), the arms of each kind vmapped side
    by side. shared: the arrays every penalty reads (not vmapped)."""

    def __init__(self, prepared, start_params, critic, settings, rep, *, cfg=ProgramConfig(), shared=None,
                 random_nets=None, alpha=ALPHA, steps=STEPS, chunk=CHUNK):
        self.prepared, self.settings, self.rep = prepared, settings, int(rep)
        self.args = L.make_args(settings, rep, alpha)
        self.steps, self.chunk = int(steps), int(min(chunk, steps))
        if self.steps % self.chunk:
            raise ValueError("steps must be a multiple of the chunk")
        rows = L.rng_np(settings, rep, L.TAG_ACTOR_ONLY).integers(prepared.n, size=(STEPS, settings.batch_size))
        self.rows_sha256 = L.tree_sha256(rows)   # the full 20,000-step stream, as check 2's ActorOnly hashes it
        rows = rows[:self.steps]
        self.rows = jnp.asarray(rows.reshape(self.steps // self.chunk, self.chunk, settings.batch_size), jnp.int32)
        self.start_params = start_params
        self.start = TrainState.create(apply_fn=L.ACTOR.apply, params=start_params, tx=L.adam(settings.lr))
        self.critic = critic
        self.cfg = cfg
        self.shared = shared
        self.random_nets = random_nets or {}
        self.seconds = {}

    def program(self, kind, n_arms):
        key = ("room", L._args_key(self.args), kind, self.cfg, int(n_arms), self.chunk)
        if key not in _PROGRAMS:
            loop = partial(_room_loop, self.args, kind, self.cfg)
            _PROGRAMS[key] = jax.jit(jax.vmap(loop, in_axes=(None, 0, None, None, None, 0)))
        return _PROGRAMS[key]

    def arm_tree(self, kind, arms):
        coef = np.zeros((len(arms), N_COEF[kind]), np.float32)
        for i, a in enumerate(arms):
            if len(a.coef) != N_COEF[kind]:
                raise ValueError(f"arm {a.name}: kind {kind} takes {N_COEF[kind]} coefficients")
            coef[i] = a.coef
        tree = dict(coef=jnp.asarray(coef), pull=jnp.asarray(np.array([a.pull for a in arms], np.float32)))
        if kind == "random":
            tree["net"] = jax.tree_util.tree_map(lambda *xs: jnp.stack(xs),
                                                 *[self.random_nets[a.surface] for a in arms])
        return tree

    def run_kind(self, kind, arms, log_fn=None):
        n = len(arms)
        states = jax.tree_util.tree_map(lambda x: jnp.broadcast_to(x, (n,) + jnp.shape(x)), self.start)
        tree = self.arm_tree(kind, arms)
        program = self.program(kind, n)
        metrics, finite_steps = [], np.ones(n, bool)
        t0 = time.perf_counter()
        for c in range(self.steps // self.chunk):
            states, aux = program(self.critic, states, self.prepared.train, self.shared, self.rows[c], tree)
            aux = np.asarray(aux, np.float64)          # (arms, chunk, 5)
            finite_steps &= np.all(np.isfinite(aux), axis=(1, 2))
            metrics.append(aux.mean(1).tolist())
            if log_fn:
                log_fn(f"  {kind} x{n}: step {(c + 1) * self.chunk}/{self.steps} ({time.perf_counter() - t0:.0f}s)")
        jax.block_until_ready(states.params)
        sec = time.perf_counter() - t0
        out = {}
        for i, a in enumerate(arms):
            p = jax.tree_util.tree_map(lambda x: x[i], states.params)
            out[a.name] = dict(params=p, finite=bool(finite_steps[i] and L._finite_tree(p)),
                               chunk_metrics=[m[i] for m in metrics], arm=a.record(), seconds=sec / n)
        self.seconds[kind] = self.seconds.get(kind, 0.0) + sec
        return out

    def run(self, arms, log_fn=None):
        names = [a.name for a in arms]
        if len(set(names)) != len(names):
            raise ValueError("arm names must be unique")
        out = {}
        for kind in KINDS:
            group = [a for a in arms if a.kind == kind]
            if group:
                out.update(self.run_kind(kind, group, log_fn))
        return out


# ------------------------------------------------------------------------------------------------ shared arrays

def hinge_bounds(actions):
    """The recorded range: per thruster, the 0.5% and 99.5% quantiles of the gapped training commands."""
    a = np.asarray(actions, np.float64)
    return np.quantile(a, 0.005, axis=0), np.quantile(a, 0.995, axis=0)


_E_PROGRAMS = {}


def controller_value_rows(states32, actions32, cfg, chunk=8192):
    """Q^c_H at rows in float32 with the in-program function (jit, chunked, padded)."""
    key = (cfg,)
    if key not in _E_PROGRAMS:
        _E_PROGRAMS[key] = jax.jit(lambda x, a: controller_value(x, a, cfg))
    fn = _E_PROGRAMS[key]
    x, a = np.asarray(states32, np.float32), np.asarray(actions32, np.float32)
    out = []
    for i in range(0, len(x), chunk):
        xs, as_ = x[i:i + chunk], a[i:i + chunk]
        pad = chunk - len(xs)
        if pad:
            xs = np.concatenate([xs, np.repeat(xs[-1:], pad, 0)])
            as_ = np.concatenate([as_, np.repeat(as_[-1:], pad, 0)])
        out.append(np.asarray(fn(jnp.asarray(xs), jnp.asarray(as_)), np.float64)[:chunk - pad])
    return np.concatenate(out)


def build_shared(prepared, critic, rep, *, cfg=ProgramConfig(), host_actor=None, need_harm=True, log_fn=None):
    """The arrays the penalties read: raw float32 states, normalised observations, the normalisation, the
    state-shuffle permutation, the recorded range, e(s_i, a_i) = Q1(s_i, a_i) - Q^c_250(s_i, a_i) for the long-run
    harm arm (float32, the in-program function) and, for the replication, the host's actor target."""
    t0 = time.perf_counter()
    raw32 = np.asarray(prepared.states, np.float32)
    if not np.array_equal(raw32.astype(np.float64), prepared.states):
        raise ValueError("training states are not exactly float32")
    perm = L.rng_np(L.LearnSettings(), rep, TAG_PERM).permutation(prepared.n)
    lo, hi = hinge_bounds(prepared.actions)
    shared = dict(raw32=jnp.asarray(raw32), obs=prepared.train.obs, mean=jnp.asarray(prepared.obs_mean),
                  std=jnp.asarray(prepared.obs_std), perm=jnp.asarray(perm, jnp.int32),
                  lo=jnp.asarray(lo, jnp.float32), hi=jnp.asarray(hi, jnp.float32))
    info = dict(perm_sha256=L.tree_sha256(perm), perm_fixed_points=int(np.sum(perm == np.arange(prepared.n))),
                recorded_range=dict(lo=lo.tolist(), hi=hi.tolist()))
    if need_harm:
        q1 = L.critic_rows(critic, np.asarray(prepared.train.obs), prepared.actions)[0][:, 0]
        qc = controller_value_rows(raw32, prepared.actions, cfg)
        e = (q1 - qc).astype(np.float32)
        shared["e_logged"] = jnp.asarray(e)
        info.update(e_logged_sha256=L.tree_sha256(e), e_logged_mean=float(e.mean()))
    else:
        shared["e_logged"] = jnp.zeros(prepared.n, jnp.float32)
    shared["host_actor"] = host_actor if host_actor is not None else L.actor_template()
    shared, dinfo = with_logged_delta(shared, critic, cfg, prepared)
    info.update(dinfo)
    info["seconds"] = time.perf_counter() - t0
    if log_fn:
        log_fn(f"shared arrays built ({info['seconds']:.0f}s)")
    return shared, info


def with_logged_delta(shared, critic, cfg, prepared):
    """A copy of `shared` with d_logged = delta*(s_i, a_i) at every training row's recorded command, computed once with
    the in-program float32 function for this program's configuration (its next command). Returns (shared, info)."""
    shared = dict(shared, d_logged=jnp.zeros(prepared.n, jnp.float32))
    d = oracle32(critic, shared, cfg, np.arange(prepared.n), prepared.actions).astype(np.float32)
    if not np.all(np.isfinite(d)):
        raise ValueError("delta* at the recorded commands is not finite")
    shared["d_logged"] = jnp.asarray(d)
    return shared, dict(d_logged_sha256=L.tree_sha256(d), d_logged_mean=float(d.mean()),
                        d_logged_next_command=cfg.next_command)


# ------------------------------------------------------------------------------------------------ oracle readouts

_ORACLE = {}


def oracle32(critic, shared, cfg, rows_idx, actions, shuffled=False):
    """delta* (float32, the in-program function) at training rows rows_idx and commands `actions`."""
    key = ("o32", cfg, bool(shuffled))
    if key not in _ORACLE:
        def f(critic, shared, idx, a):
            return one_step(critic, shared, cfg, shared["perm"][idx] if shuffled else idx, a)
        _ORACLE[key] = jax.jit(f)
    out = []
    for i in range(0, len(rows_idx), 8192):
        out.append(np.asarray(_ORACLE[key](critic, shared, jnp.asarray(rows_idx[i:i + 8192], jnp.int32),
                                           jnp.asarray(np.asarray(actions[i:i + 8192], np.float32))), np.float64))
    return np.concatenate(out)


def oracle_grad32(critic, shared, cfg, rows_idx, actions):
    """d delta* / d a (float32, in-program function) per row (n, 4)."""
    key = ("g32", cfg)
    if key not in _ORACLE:
        def g(critic, shared, idx, a):
            return jax.vmap(jax.grad(lambda aa, i: one_step(critic, shared, cfg, i[None], aa[None])[0]),
                            in_axes=(0, 0))(a, idx)
        _ORACLE[key] = jax.jit(g)
    out = []
    for i in range(0, len(rows_idx), 8192):
        out.append(np.asarray(_ORACLE[key](critic, shared, jnp.asarray(rows_idx[i:i + 8192], jnp.int32),
                                           jnp.asarray(np.asarray(actions[i:i + 8192], np.float32))), np.float64))
    return np.concatenate(out)


def oracle64(critic, prepared, states, actions, physics, next_command="stall_aware", q_head="min"):
    """delta* with the simulator step in float64 (dynamics_jax under enable_x64) and the frozen float32 critic at the
    float32-rounded normalised states: r + 0.98 min_h Q_h(s', c(s')) - Q(s, a), Q = min_h Q_h (the plan's oracle) or
    Q1 (q_head='q1', the earlier readout's form)."""
    from jax.experimental import enable_x64
    x = np.asarray(states, np.float64)
    a = np.asarray(actions, np.float64)
    with enable_x64():
        x1, r = TC.dynamics_jax(jnp.asarray(x), jnp.asarray(a), physics)
        ctrl = RC.stall_aware(physics) if next_command == "stall_aware" else RC.rising_thrust(physics)
        a1 = ctrl.jax(x1)
        x1, r, a1 = np.asarray(x1), np.asarray(r), np.asarray(a1)
    qn = L.critic_rows(critic, prepared.normalize(x1), a1.astype(np.float32))[0].min(1)
    q0 = L.critic_rows(critic, prepared.normalize(x), a.astype(np.float32))[0]
    q0 = q0.min(1) if q_head == "min" else q0[:, 0]
    return r + physics.gamma * qn - q0


# ------------------------------------------------------------------------------------------------ slope normalisation

_SLOPE = {}


def slope_rms(kind, cfg, critic, shared, rows_idx, actions, net=None):
    """RMS over rows of ||grad_a W(s, a)|| for W = the main oracle [delta*(s, a_i) - delta*(s, a)]_+ ('band'),
    mean_j a_j^2 ('push_less') or a random surface ('random'), at training rows and the given commands (float32)."""
    key = ("slope", kind, cfg)
    if key not in _SLOPE:
        def w_row(a, i, critic, shared, net):
            if kind == "band":
                return band_terms(MAIN_BAND, one_step(critic, shared, cfg, i[None], a[None])[0],
                                  shared["d_logged"][i])
            if kind == "push_less":
                return jnp.square(a).mean()
            if kind == "random":
                return RANDOM_NET.apply(net, shared["obs"][i][None], a[None])[0]
            raise ValueError(kind)

        def g(critic, shared, idx, a, net):
            gr = jax.vmap(jax.grad(w_row), in_axes=(0, 0, None, None, None))(a, idx, critic, shared, net)
            return jnp.sum(jnp.square(gr), axis=-1)

        _SLOPE[key] = jax.jit(g)
    net = net if net is not None else RANDOM_NET.init(jax.random.PRNGKey(0), jnp.zeros(TC.DX), jnp.zeros(TC.DA))
    sq = np.asarray(_SLOPE[key](critic, shared, jnp.asarray(rows_idx, jnp.int32),
                                jnp.asarray(np.asarray(actions, np.float32)), net), np.float64)
    return float(np.sqrt(sq.mean()))


def slope_sample(prepared, rep):
    return np.sort(L.rng_np(L.LearnSettings(), rep, TAG_SLOPE).choice(prepared.n, min(SLOPE_ROWS, prepared.n),
                                                                     replace=False))


# ------------------------------------------------------------------------------------------------ check (e)

def penalty_grad32(critic, shared, cfg, rows_idx, actions, coef):
    """dW/da of the band kind's in-program penalty (coefficients coef, BAND_TERMS order) per row (n, 4), float32."""
    coef = tuple(float(c) for c in coef)
    key = ("pg32", cfg, coef)
    if key not in _ORACLE:
        def g(critic, shared, idx, a):
            def w(aa, i):
                return band_terms(coef, one_step(critic, shared, cfg, i[None], aa[None])[0], shared["d_logged"][i])
            return jax.vmap(jax.grad(w), in_axes=(0, 0))(a, idx)
        _ORACLE[key] = jax.jit(g)
    out = []
    for i in range(0, len(rows_idx), 8192):
        out.append(np.asarray(_ORACLE[key](critic, shared, jnp.asarray(rows_idx[i:i + 8192], jnp.int32),
                                           jnp.asarray(np.asarray(actions[i:i + 8192], np.float32))), np.float64))
    return np.concatenate(out)


def shape_rows(prepared, rep):
    """SHAPE_ROWS full-throttle-region training states, default_rng([20261015, seed, 65]) (check (c)'s rows)."""
    reg = np.flatnonzero(prepared.region)
    return np.sort(L.rng_np(L.LearnSettings(), rep, TAG_SHAPE).choice(reg, min(SHAPE_ROWS, len(reg)), replace=False))


SHAPE_FORMS = {"main (perfect_band)": MAIN_BAND,
               "abs_rel (perfect_band_centred)": (0.0, 1.0, 0.0, 0.0, 0.0),
               "under_rel (underrating_half)": (0.0, 0.0, 1.0, 0.0, 0.0),
               "abs_lit (perfect_band_literal)": (0.0, 0.0, 0.0, 1.0, 0.0),
               "over_lit (overrating_half_literal)": (0.0, 0.0, 0.0, 0.0, 1.0)}


def oracle_shape(critic, shared, cfg, prepared, actor_params, rep, forms=SHAPE_FORMS):
    """Check (e): the shape of each band penalty along thruster-1 throttle at full-throttle-region training states,
    from the frozen critic's surface alone (no return, no training). At SHAPE_ROWS region states the no-caution actor's
    command has its thruster 1 set to each SHAPE_GRID throttle in the actor's own push direction (the other thrusters
    stay at the actor's command). The main form passes when (e1) W falls from 80% to 81% on at most
    SHAPE_MAX_FALL_SHARE of the rows, (e2) mean W rises from 80% to 81%, (e3) the in-program penalty's mean slope along
    the push at 80.5% is positive (and every slope is finite), and (e4) the best grid throttle for Q1 - W lies past 80%
    on fewer rows than for Q1 alone. The other forms are measured the same way and reported."""
    t0 = time.perf_counter()
    rows = shape_rows(prepared, rep)
    obs = np.asarray(prepared.train.obs)[rows]
    base = L.actor_rows(actor_params, obs).astype(np.float64)
    sign = np.where(base[:, 0] < 0, -1.0, 1.0)
    dl = np.asarray(shared["d_logged"], np.float64)[rows]

    def at(m):
        aa = base.copy()
        aa[:, 0] = sign * (TC.KNEE if m == 0.80 else m)
        return aa

    D, Q1 = {}, {}
    for m in SHAPE_GRID:
        aa = at(m)
        D[m] = oracle32(critic, shared, cfg, rows, aa)
        Q1[m] = np.asarray(L.critic_rows(critic, obs, aa.astype(np.float32))[0][:, 0], np.float64)
    grid = np.array(SHAPE_GRID)
    q_best = grid[np.argmax(np.stack([Q1[m] for m in SHAPE_GRID], 1), 1)]
    out = dict(rows=int(len(rows)), rows_rule=f"default_rng([20261015, {int(rep)}, {TAG_SHAPE}]) choice of the region rows",
               actor_thruster1_past80_share=float(np.mean(np.abs(base[:, 0]) > TC.KNEE)),
               push_negative_share=float(np.mean(sign < 0)), delta_at_recorded_mean=float(dl.mean()),
               q1_alone_best_past80_share=float(np.mean(q_best > 0.8)), grid=list(SHAPE_GRID), forms={})
    for name, coef in forms.items():
        c = np.asarray(coef, np.float64)
        W = {m: band_terms(c, D[m], dl) for m in SHAPE_GRID}
        best = grid[np.argmax(np.stack([Q1[m] - W[m] for m in SHAPE_GRID], 1), 1)]
        grads = {}
        for m in SHAPE_GRAD_POINTS:
            g = penalty_grad32(critic, shared, cfg, rows, at(m), coef)[:, 0] * sign
            grads[f"{m:g}"] = dict(mean=float(g.mean()), negative_share=float(np.mean(g < 0)),
                                   finite_share=float(np.isfinite(g).mean()))
        fall, rise = float(np.mean(W[0.81] < W[0.80])), float(np.mean(W[0.81] > W[0.80]))
        rec = dict(coef=list(coef), mean_W={f"{m:g}": float(W[m].mean()) for m in SHAPE_GRID},
                   fall_share_80_to_81=fall, rise_share_80_to_81=rise, unchanged_share_80_to_81=1.0 - fall - rise,
                   best_past80_share=float(np.mean(best > 0.8)),
                   mean_W_full_minus_80=float(np.mean(W[1.0] - W[0.80])), mean_dW_dpush=grads)
        rec["e1_fall_share_ok"] = bool(fall <= SHAPE_MAX_FALL_SHARE)
        rec["e2_mean_rises"] = bool(rec["mean_W"]["0.81"] > rec["mean_W"]["0.8"])
        rec["e3_slope_past_knee_positive"] = bool(grads["0.805"]["mean"] > 0 and
                                                  all(v["finite_share"] == 1.0 for v in grads.values()))
        rec["e4_fewer_best_past80_than_q1"] = bool(rec["best_past80_share"] < out["q1_alone_best_past80_share"])
        rec["passed"] = bool(rec["e1_fall_share_ok"] and rec["e2_mean_rises"] and rec["e3_slope_past_knee_positive"]
                             and rec["e4_fewer_best_past80_than_q1"])
        out["forms"][name] = rec
    out["passed"] = out["forms"]["main (perfect_band)"]["passed"]
    out["seconds"] = time.perf_counter() - t0
    return out


# ------------------------------------------------------------------------------------------------ arms

def fixed_arms():
    """Section 4.1's arms at fixed coefficients (no matching)."""
    return [RoomArm("w0_check", "plain", (0.0,), 1.0),
            RoomArm("perfect_band", "band", MAIN_BAND),
            RoomArm("perfect_band_half", "band", (0.5, 0.0, 0.0, 0.0, 0.0)),
            RoomArm("perfect_band_centred", "band", (0.0, 1.0, 0.0, 0.0, 0.0)),
            RoomArm("underrating_half", "band", (0.0, 0.0, 1.0, 0.0, 0.0)),
            RoomArm("perfect_band_literal", "band", (0.0, 0.0, 0.0, 1.0, 0.0)),
            RoomArm("overrating_half_literal", "band", (0.0, 0.0, 0.0, 0.0, 1.0)),
            RoomArm("state_shuffled_band", "band_shuffled", (1.0,)),
            RoomArm("long_run_harm", "harm", (1.0,)),
            RoomArm("range_hinge", "hinge", (1.0,))]


def family_arm(family, knob, scale, tag=None):
    """The arm of a gating-control family at a ladder knob. scale: {family: coefficient per knob} (slope
    normalisation; 1 for the even pull, whose knob is nu)."""
    name = f"{family}@{knob:.10g}" if tag is None else f"{family}@{tag}"
    if family == "even_pull":
        return RoomArm(name, "plain", (0.0,), 1.0 + float(knob), family=family, knob=float(knob))
    if family == "push_less":
        return RoomArm(name, "push_less", (float(knob) * scale[family],), family=family, knob=float(knob))
    if family in ("random_1", "random_2"):
        return RoomArm(name, "random", (float(knob) * scale[family],), surface=int(family[-1]), family=family,
                       knob=float(knob))
    raise ValueError(family)


def ladder(family):
    return EVEN_LADDER if family == "even_pull" else SLOPE_LADDER


# ------------------------------------------------------------------------------------------------ strengths

def region_strength(params, params_none, prepared):
    obs = np.asarray(prepared.train.obs)[prepared.region]
    a = L.actor_rows(params, obs).astype(np.float64)
    b = L.actor_rows(params_none, obs).astype(np.float64)
    return float(np.sqrt(np.mean(np.sum((a - b) ** 2, axis=1))))


def max_param_diff(a, b):
    return max(float(np.max(np.abs(np.asarray(x, np.float64) - np.asarray(y, np.float64))))
               for x, y in zip(jax.tree_util.tree_leaves(a), jax.tree_util.tree_leaves(b)))


# ------------------------------------------------------------------------------------------------ the spec

CHANGES_BEFORE_ANY_RUN = [
    dict(id="A1 main oracle re-referenced to the recorded command",
         finding="review, oracle lens (blocker)",
         status=("NEEDS THE USER'S APPROVAL before seeds 0 and 1 are launched: it changes the plan's main oracle. "
                 "The plan's literal form still runs, as the reported arm perfect_band_literal."),
         change=("The main oracle ('perfect band', k = 1) is W(s_i, a) = [delta*(s_i, a_i) - delta*(s_i, a)]_+ instead "
                 "of the plan's |delta*(s_i, a)|: the critic's extra overrating of the actor's command a over the row's "
                 "own recorded command a_i, by its exact one-step error. The same change is made in every arm built "
                 "on the oracle: the half-strength arm (k = 0.5), the state-shuffled arm (the same form at another "
                 "training state, with that state's own recorded command), the two no-stall perfect bands and the "
                 "replication's perfect band. The plan's overrating and underrating halves become the two sides of the "
                 "re-referenced error: the overrating side is the main oracle itself; the underrating side "
                 "[delta*(s, a) - delta*(s, a_i)]_+ is the reported arm underrating_half. Added, reported: the "
                 "symmetric band measured from the recorded command |delta*(s, a) - delta*(s, a_i)| "
                 "(perfect_band_centred), the plan's literal |delta*(s, a)| (perfect_band_literal) and the plan's "
                 "literal overrating half max(0, -delta*(s, a)) (overrating_half_literal)."),
         reason=("The literal |delta*| does not rise past the 80% knee; it rewards the first points of over-throttle. "
                 "At full-throttle-region training states delta* is about +0.17 at 80% throttle and crosses zero only "
                 "near 86%, so |delta*| first falls as the critic's false preference grows. On 2,000 region states "
                 "(seed 0 / seed 1), with thruster 1 set along the no-caution actor's own push: |delta*| falls from "
                 "80% to 81% on 81.8% / 84.1% of them; the in-program penalty's mean slope along the push at 80.5% is "
                 "-1.97 / -2.44; and the throttle that maximises Q1 - |delta*| lies past 80% on 71.3% / 75.0% of them, "
                 "against 57.0% / 61.8% for Q1 alone. Part of the cause is an offset that comes from the oracle's "
                 "definition, not from the harm: the critic was trained toward r + 0.98 min_h Q_h(s', clip(c(s') + "
                 "smoothing noise)), the oracle reads the noiseless c(s'), and the minimum over heads sits below Q1; "
                 "at recorded commands delta* averages +0.149 (seed 0). Measuring from the recorded command removes "
                 "that offset (W = 0 at a = a_i) and keeps what changes with the command. The one-sided form is chosen "
                 "for its meaning: a lower edge exists to remove overrating, and this form removes exactly the "
                 "critic's excess overrating at the command and never rewards a command for being underrated. It is "
                 "the one-step counterpart of the plan's own long-run harm arm [e(s, a) - e(s, a_i)]_+. On the same "
                 "states it falls from 80% to 81% on 5.1% / 4.6% (on 45% / 49% it is zero at both points: caution "
                 "starts a little further out), its slope at 80.5% is +1.54 / +1.46, it rises by 0.54 / 0.59 from 80% "
                 "to full throttle, and the best throttle for Q1 - W lies past 80% on 25.8% / 33.0%. The two-sided "
                 "centred form falls on 46.5% / 52.3% and is reported. On check 3's critic (the replication) the "
                 "chosen form falls on 7.5% / 7.1%, the literal form on 80.1% / 82.1%. The choice was made before "
                 "any room-test training, from the frozen critics' surfaces only: no return, points, score or "
                 "strength of any arm was computed. What is lost: the main oracle is no longer literally the width "
                 "of a perfectly sharp two-sided BCA band; it is the overrating part of that width, measured from "
                 "the data."),
         evidence=("scratchpad/review_oracle/v2_knee.py, v6_centered.py (reviewer); scratchpad/ood/room/verify/"
                   "v_knee.py and knee_<seed>_<critic>.log (an independent float64 numpy step); check (e) below")),
    dict(id="A2 check (e) on the penalty itself",
         finding="review, oracle lens (blocker)",
         change=("New per-seed check (e), run before any arm is trained: the main oracle's penalty, not delta*, must "
                 "not fall past the knee and must not move the static optimum past 80%. Rule in checks.e. A failure "
                 "stops that seed (stop.json), like checks (a) and (b), in the stalling world and in the replication. "
                 "In the no-stall world, where the missing commands help, the same measurement is reported only."),
         reason="check (c) measured the slope of delta*, not of the penalty the actor feels"),
    dict(id="A3 check (c) relabelled",
         finding="review, fidelity and oracle and rigour lenses (minor, three findings)",
         change=("Check (c)'s readout match is reported for what it is: with the readout's Q1 form on the Q(s, a) side "
                 "it reproduces the readout (-0.7298, 0.1066) on the readout's own 3,000 rows, which checks the "
                 "simulator step, the critic's inputs and the row alignment (the plumbing). The trained form "
                 "(min over the two heads on both sides, as BCA's own residual: algorithms/td3_bc_bca.fit_scale and "
                 "refresh take q = min_h Q_h(s, a)) does not reproduce the plan's 7.3(b) reference numbers: -0.589 "
                 "and 0.145 on those rows, an offset of +0.149 at recorded commands. That is reported as 'not met', "
                 "not as a pass. The plan's 7.3(b) reference values apply only to the Q1 form. The min form is kept "
                 "because it is BCA's own residual. The amended oracle's own readout is reported beside it. The "
                 "float32-against-float64 bound stays the check on the trained form."),
         reason="the earlier check certified a form that no arm trains with"),
    dict(id="A4 slope normalisation on the main oracle",
         change=("g_band, which scales the push-less and random-surface ladders, is computed for the amended main "
                 "oracle (was |delta*|)"),
         reason="knob 1 is meant to have the main oracle's average command slope"),
    dict(id="A5 the no-BCA statement corrected",
         finding="review, rigour lens (minor)",
         change=("'Nothing from calibration/ is called' was false: the gate's data loader calls "
                 "calibration/bank.stratified_bank (experiments/shift/dataset.py line 215, through "
                 "acceptance.load_seed) to regenerate the log's calibration-bank split. It is a NumPy row sampler; "
                 "no band, calibrator, threshold, dose or BCA update is called."),
         reason="the claim as written was wrong; no BCA signal was touched"),
    dict(id="A6 code pinned by a manifest",
         finding="review, rigour lens (major)",
         change=("runs/shift_explore/room/code_manifest.json (+ .sha256) holds the SHA-256 of every room-test script "
                 "(harness.py, run_seed.py, launch.sh, write_spec.py, write_manifest.py, checks_seed0.py, "
                 "summarise_checks.py, dryrun.py) and the spec's SHA-256; copies of the files are in "
                 "runs/shift_explore/room/code/. Every phase, futility and pooled included, and the seed-0 checks "
                 "refuse when a file or its copy differs from the manifest. Every lock, values file, stop file and "
                 "check records the manifest's SHA-256. A later change needs a new manifest written by "
                 "write_manifest.py with a stated reason; the old one is kept in code_manifest_history/."),
         reason="nothing pinned the code between the futility look and seeds 2-4"),
    dict(id="A7 values, eval and pooled refusals",
         finding="review, rigour lens (minor)",
         change=("The futility and pooled looks accept a seed's values.json only when it is labelled EXPLORATORY, its "
                 "kind is room-values, its spec_sha256 is the current spec and its lock_sha256 is the SHA-256 of that "
                 "seed's verified lock.json; a stop.json is accepted only when labelled and naming the current spec. "
                 "Every eval phase refuses when its values.json already exists or the seed has stopped. Futility "
                 "refuses until seeds 0 and 1 each have values.json or stop.json. Pooled refuses until futility.json "
                 "says continue and every seed has values.json or stop.json. Both looks check the gate files and the "
                 "code manifest. The futility and pooled launch lines run at nice 5, pinned."),
         reason="synthetic values files were accepted; eval could overwrite values; pooled could run early"),
    dict(id="A8 futility with a missing difference",
         finding="review, rigour lens (minor)",
         change=("A missing difference counts as less than 5 points in the futility look: the perfect band not "
                 "finite, the even pull with no matched arm, or the seed stopped by a failed check (a), (b) or (e). "
                 "futility.json names the reason for each seed."),
         reason=("the plan did not say; a missing difference means the room mark cannot pass for that seed, and a "
                 "control that cannot be matched already fails every mark that needs it")),
    dict(id="A9 logs",
         change="the launch script logs to runs/shift_explore/room/logs/ (was the scratchpad)",
         reason="the logs should sit beside the outputs"),
]


def spec_payload(timing=None):
    """Everything fixed before any room-test training (written to SPEC_PATH by write_spec.py). timing: the pre-spec
    timing record (timing.py: a random start actor, results discarded) that decided the long-run harm arm."""
    main_w = "[delta*(s_i, a_i) - delta*(s_i, a)]_+"
    return dict(
        pre_spec_timing=timing,
        label=LABEL,
        kind="room-test-spec",
        version=2,
        previous_spec=dict(sha256=PREVIOUS_SPEC_SHA256, archived="spec_history/spec_v1.json",
                           note=("version 1, written before any room-test training; only the seed-0 checks ran under "
                                 "it (archived with their outputs in superseded/v1/)")),
        changes_before_any_run=CHANGES_BEFORE_ANY_RUN,
        user_approval=("Change A1 alters the plan's main oracle. Seeds 0 and 1 should be launched only after the user "
                       "approves it. Nothing in the code enforces this; it is recorded here and in the job report."),
        plan=dict(path=str(PLAN_PATH), sha256=file_sha256(PLAN_PATH), sections=["4.0", "4.1", "7"]),
        no_bca=("No BCA band, calibrator, threshold, dose, BCA update or BCA training arm. Nothing from "
                "algorithms/td3_bc_bca.py or learners.bca_* is called (learners.py imports them; they are never "
                "called). From calibration/ only calibration/bank.stratified_bank runs: the gate's data loader "
                "(experiments/shift/dataset.py, through acceptance.load_seed) calls it to regenerate the log's "
                "calibration-bank split. It is a NumPy row sampler, not a BCA signal."),
        seeds=list(SEEDS), futility_seeds=list(FUTILITY_SEEDS), variant="base", world="stall",
        setup=dict(
            protocol="check 2's actor-only protocol (acceptance.h2_seed's critic swap), unchanged",
            critic=("check 2's frozen fixed-controller critic of the stalling world on the gapped log "
                    "(runs/shift/<prereg>/critics/base/stall/stall_aware_gap/seed<k>, read with "
                    "acceptance.get_critic(prereg, sd, 'gap', 'stall_aware', train=False)); its scored policy is the "
                    "stall-aware controller (recorder.stall_aware: the textbook gain clipped at +-0.8)"),
            start_actor="the saved cloned actor (runs/shift/<prereg>/clone/base/stall/seed<k>/actor.msgpack)",
            alpha=ALPHA, actor_steps=STEPS, chunk=CHUNK, batch_size=256, optimiser="Adam (runtime.networks."
            "torch_adam), learning rate 3e-4, as check 2",
            batch_rows=("check 2's stream: np.random.default_rng([20261015, seed, 35]).integers(n_train_rows, "
                        "size=(20000, 256)), the same rows for every arm"),
            arms_side_by_side="the arms of one penalty kind run vmapped in one program, as the gate runs its arms"),
        actor_loss=dict(
            formula=("-lambda * mean_i[Q1(s_i, pi(s_i)) - k * W(s_i, pi(s_i))] + (1 + nu) * mean_ij (pi(s_i)_j - "
                     "a_ij)^2"),
            lam="lambda = alpha / stopgrad(mean_i |Q1(s_i, pi(s_i))|), computed on Q1 alone (W never enters it)",
            frozen=("W's own parameters (frozen critic, random network, simulator, constants such as delta*(s_i, a_i)) "
                    "get no gradient: only the actor's parameters are differentiated, so W acts on the actor only "
                    "through the command pi(s)"),
            w0=("with k W = 0 and nu = 0 the loss is learners.actor_step's unweighted expression term for term "
                "((q - 0).mean(), and the cloning term times 1.0)")),
        oracle=dict(
            residual="delta*(s, a) = r(s, a) + 0.98 * min_h Q_h(s', c(s')) - min_h Q_h(s, a)",
            parts=("(s', r) = thruster_chain.dynamics_jax(s, a, physics): one exact simulator step from the row's "
                   "state (prepared.states: the float64 of the float32 observation, exactly representable in "
                   "float32), reward -(|s|^2 + 0.3 |clip(a)|^2) on the clipped command, thrust stalling past the "
                   "knee as thruster_chain.thrust_jax; c = recorder.stall_aware (the critic's scored controller) "
                   "on s'; Q_h the frozen critic's two heads at the float32-rounded normalised state "
                   "((float32(s) - mean) / std, learners' normalisation) and the command. Min over the heads on both "
                   "sides, as BCA's own residual (td3_bc_bca.fit_scale and refresh)."),
            main_penalty=dict(
                W=main_w,
                meaning=("the critic's extra overrating of the command a over the row's own recorded command a_i, by "
                         "its exact one-step error; zero at a = a_i; the one-step counterpart of the long-run harm arm "
                         "[e(s, a) - e(s, a_i)]_+"),
                logged=("delta*(s_i, a_i) is computed once per training row with the same in-program float32 function "
                        "and the same next command as the arm (shared['d_logged']); a constant in training"),
                amended="change A1 (changes_before_any_run): the plan's literal |delta*(s, a)| is the reported arm "
                        "perfect_band_literal"),
            band_terms=dict(over_rel="[delta*(s, a_i) - delta*(s, a)]_+ (the main oracle)",
                            abs_rel="|delta*(s, a) - delta*(s, a_i)|", under_rel="[delta*(s, a) - delta*(s, a_i)]_+",
                            abs_lit="|delta*(s, a)|", over_lit="[-delta*(s, a)]_+"),
            gamma=GAMMA,
            precision=("float32 inside the training program (the same dynamics_jax function); float64 version "
                       "(dynamics_jax under enable_x64) computed for check (c); see deviations"),
            gradient="full derivative in a: through r, through s' (critic input and c(s')) and through Q(s, a)"),
        arms=[
            dict(name="none", role="baseline (check 2's gapped critic's actor)", W="0", k=0, nu=0,
                 how=("trained by the gate's own unchanged code path (learners.ActorOnly.run, unweighted, the three "
                      "critics of check 2 side by side) and checked against check 2's saved 'gapped' actor by "
                      "parameter hash; the locked 'none' actor is check 2's saved actor")),
            dict(name="w0_check", role="check (b) only, not scored", W="0", k=0, nu=0,
                 how="the new program with kW = 0, nu = 0; must match 'none' within 1e-5 on every parameter"),
            dict(name="perfect_band", role="main oracle (amended, change A1)", W=main_w, k=1.0),
            dict(name="perfect_band_half", role="dose-response, reported", W=main_w, k=0.5),
            dict(name="perfect_band_centred", role="the symmetric band measured from the recorded command; reported",
                 W="|delta*(s, a) - delta*(s, a_i)|", k=1.0),
            dict(name="underrating_half", role="the same information on the wrong side; reported",
                 W="[delta*(s, a) - delta*(s, a_i)]_+", k=1.0),
            dict(name="perfect_band_literal", role="the plan's literal perfect band; reported", W="|delta*(s, a)|",
                 k=1.0),
            dict(name="overrating_half_literal", role="the plan's literal overrating half; reported",
                 W="max(0, -delta*(s, a))", k=1.0),
            dict(name="state_shuffled_band", role="does state placement add anything? reported",
                 W=("[delta*(s_sigma(i), a_sigma(i)) - delta*(s_sigma(i), a)]_+ at this row's command a = pi(s_i): "
                    "the main oracle at another training state with that state's own recorded command; sigma = "
                    "default_rng([20261015, seed, 61]).permutation(n_train_rows), fixed"), k=1.0),
            dict(name="long_run_harm", role="check 3's exact harm read as a value; reported",
                 W=("[e(s, a) - e(s, a_i)]_+, e(s, a) = Q1(s, a) - Q^c_250(s, a), Q^c_250 the stall-aware "
                    "controller's value by a differentiable 250-step rollout through dynamics_jax (float32 in the "
                    "program); e(s_i, a_i) at the recorded command precomputed once with the same function"),
                 k=1.0),
            dict(name="even_pull", role="gating control: check 3's caution at matched strength",
                 W="none (cloning pull multiplied by 1 + nu)", ladder_nu=list(EVEN_LADDER),
                 matched=True),
            dict(name="push_less", role="gating control", W="mean_j pi_j^2",
                 coefficient="c = knob * g_band / g_push", ladder_knobs=list(SLOPE_LADDER), matched=True),
            dict(name="random_1", role="gating control: same kind of penalty, random placement",
                 W=("random network of the scale network's shape (Dense 64 - ReLU - Dense 64 - ReLU - Dense 1, "
                     "softplus + 0.01; flax default initialisation; inputs obs / 1.001 and the command), frozen"),
                 init_seed="default_rng([20261015, seed, 62, 1]).integers(2^31 - 1) as PRNGKey",
                 coefficient="k = knob * g_band / g_random_1", ladder_knobs=list(SLOPE_LADDER), matched=True),
            dict(name="random_2", role="gating control (second random surface)", W="as random_1",
                 init_seed="default_rng([20261015, seed, 62, 2]).integers(2^31 - 1) as PRNGKey",
                 coefficient="k = knob * g_band / g_random_2", ladder_knobs=list(SLOPE_LADDER), matched=True),
            dict(name="range_hinge", role="the data-only rival with no signal; reported",
                 W="100 * sum_j ([a_j - hi_j]_+^2 + [lo_j - a_j]_+^2)",
                 range="lo_j, hi_j: the 0.5% and 99.5% quantiles of the gapped training commands per thruster",
                 k=1.0)],
        slope_normalisation=(
            "g_X = sqrt(mean over 4,096 training rows (default_rng([20261015, seed, 63]).choice(n, 4096, "
            "replace=False)) of ||grad_a W_X(s, a)||^2) at a = the cloned actor's command (before any training); "
            "g_band for the main oracle W = " + main_w + " (change A4). A push-less or random-surface knob of 1 gives "
            "the same average command slope as the perfect band at k = 1. Outcome-free: no return, no strength enters "
            "it."),
        strength=dict(
            definition=("S = root-mean-square change of the command from the no-caution arm on the 5,000 "
                        "evaluation-split recorded states (learners.strength on prepared.eval_obs; check 3's rule)"),
            region_check="the same on the full-throttle-region training states (prepared.region), reported"),
        matching=dict(
            rule=("every gating control (even_pull, push_less, random_1, random_2) is matched blind to the perfect "
                  "band's strength with the gate's matcher experiments/signal/glq_match.match_parallel (knob 0 = no "
                  "caution, S = 0, leads every ladder; first rung within tol taken; else log-linear interpolation "
                  "(linear from 0) then geometric bisection), tol 0.01, at most 5 rounds, strengths only, before any "
                  "return is computed; the families run in parallel rounds"),
            extension=("blind ladder extension, declared now: when the target exceeds every rung's strength of a "
                       "family, rungs are appended at the top knob times the family's factor (even pull x2: nu = 4, "
                       "8, 16; the others x sqrt(10)) until a rung's strength reaches the target, at most 3 "
                       "rungs; still unreachable -> that control has no matched arm and every mark that needs it "
                       "fails"),
            chosen_arm=("the rung or evaluated proposal with the best |S/S* - 1| (glq_match's record); its actor is "
                        "the one locked")),
        lock=("after every arm is trained and matched, lock.json (+ .sha256) is written with every actor's "
              "parameter SHA-256 (learners.tree_sha256), the SHA-256 of actors.msgpack, the strengths, matches, "
              "coefficients, checks (a), (b) and (e), spec.json's SHA-256 and the code manifest's SHA-256. No return, "
              "points or score of any trained arm is computed before it exists; the evaluation phase verifies it "
              "first"),
        code=("runs/shift_explore/room/code_manifest.json (change A6): every phase and every seed-0 check refuses "
              "unless each listed script and its copy under code/ have the SHA-256 the manifest records and the "
              "manifest names this spec"),
        evaluation=("acceptance._evaluate (learners.evaluate_actor): exact 100-step returns on the full 1,000-start "
                    "banks of all four situations (recorded, strong current, big waves, big kicks), points against "
                    "the gate's anchors (0 = recording controller, 100 = stall-aware controller), the same actor "
                    "capped at 80%, the share of steps past 80%; plus, reported, the share past 80% per thruster "
                    "from one more exact rollout per situation. Scored actors: none (check 2's saved actor), the "
                    "cloned actor, every fixed arm except w0_check, and the matched arm of every gating control. "
                    "An eval phase refuses when its values.json exists or its seed has stopped."),
        marks=dict(
            pooled=("means over seeds 0-4; every mark needs every seed (a missing or failed seed fails the mark). The "
                    "pooled look refuses until futility.json says continue and every seed has a verified values.json "
                    "or stop.json"),
            room=("in strong current the perfect band beats each gating control (even_pull, push_less, random_1, "
                  "random_2, each at its matched strength) by at least 20 points (pooled mean of the paired "
                  "difference), and that difference is positive in at least 4 of 5 seeds, for each control"),
            recovery=("in strong current the perfect band is at least 49.5 points above no caution (pooled mean "
                      "of the paired difference; half of check 2's 98.9-point harm)"),
            keeps_improvement=("recorded conditions: perfect band >= no caution - 2 and >= cloned actor + 25 "
                               "(pooled means); big waves and big kicks: perfect band >= no caution - 3 (pooled "
                               "means)"),
            futility=("seeds 0 and 1 run first; if the perfect band beats the matched even pull by less than 5 "
                      "points in strong current on both seeds, stop and go to plan section 4.5 (seeds 2-4 are not "
                      "run). A missing difference counts as less than 5 points: the perfect band not finite, the even "
                      "pull with no matched arm, or the seed stopped by a failed check (a), (b) or (e); futility.json "
                      "names the reason per seed (change A8). The look refuses until both seeds have a verified "
                      "values.json or stop.json"),
            prediction_not_a_mark=("perfect band 60-100 points in strong current (cap-at-80% ceiling 101.8, "
                                   "full-log actor 84.6); matched controls within -5 to +10 of no caution. Written "
                                   "by the plan for its literal oracle; kept unchanged")),
        other_runs=dict(
            no_stall=("no-stall comparison world, after the stalling world's lock of the same seed, not re-matched: "
                      "none (check 2's 'no_stall_gapped' actor, reproduced by the gate's code path and checked by "
                      "hash), perfect band k = 1 (the amended form) with the critic's own scored controller "
                      "(stall-aware) for c(s'), perfect band k = 1 (the amended form) with the rising-thrust "
                      "controller for c(s') (the plan's literal wording; reported), even pull at the stalling world's "
                      "locked nu, range hinge (that world's own recorded range); critic: check 2's no-stall gapped "
                      "critic; start: the no-stall cloned actor; the W = 0 check (b) against 'none' as well; check (e)'s "
                      "measurement reported, not a stop condition (the missing commands help there)"),
            replication=("check 3's critic (the frozen co-trained plain TD3+BC critic on the gapped log, alpha 5): "
                         "none (check 3's saved 'none' actor; the new program with W = 0 must match it within 1e-5), "
                         "perfect band k = 1 (the amended form) with c(s') = the host's own actor (its target copy, "
                         "the actor the critic's own Bellman target reads, no smoothing noise), and the even pull "
                         "matched blind to that perfect band's strength (ladder nu 0.1-2, same rule); the cloning term "
                         "in check 3's per-row weighted form, as check 3 ran it; check (e) on this critic with check "
                         "3's 'none' actor (stop on failure); check 3's saved arms (harm_placed, "
                         "strength_matched_even, cap2) are reused from its values.json for comparison")),
        checks=dict(
            a="the no-caution arm through the gate's unchanged code reproduces check 2's saved gapped-critic actor by "
              "parameter hash (every seed; stop on failure)",
            b="the new program with W = 0 matches that actor within 1e-5 on every parameter (every seed; stop on "
              "failure)",
            c=("seed 0. Plumbing: on the earlier readout's own 3,000 training rows (readout_seed0.json) the readout's "
               "Q1 form (Q1 on the Q(s, a) side) reproduces its mean at the gapped critic's actor's past-80% commands "
               "(-0.7298) and its median size elsewhere (0.1066) within 0.01; this checks the simulator step, the "
               "critic's inputs and the row alignment. The trained form (min over heads, BCA's residual) against the "
               "same reference numbers: reported as met or not met, not a stop condition (change A3; the plan gives "
               "no stop rule for this check). The amended main oracle's readout on the same rows and on every row: "
               "reported. Precision: |float32 in-program - float64| <= 1e-3 for delta* at every training row, at the "
               "actor's and at the recorded commands. Gradients of delta* and of the main penalty along thruster-1 "
               "throttle finite on both sides of the 80% knee"),
            d="a 400-step run of every arm kind gives finite losses; timing per 1,000 steps per kind",
            e=("every seed, before any arm is trained (change A2); the frozen critic's surface only, no return. On "
               "2,000 full-throttle-region training states (default_rng([20261015, seed, 65])) thruster 1 of the "
               "no-caution actor's command is set along the actor's own push to each throttle of the grid "
               + ", ".join(f"{m:g}" for m in SHAPE_GRID) + " (the other thrusters stay at the actor's command). The "
               "main oracle passes when (e1) W falls from 80% to 81% on at most "
               f"{SHAPE_MAX_FALL_SHARE:.0%} of the states, (e2) the mean of W rises from 80% to 81%, (e3) the "
               "in-program penalty's mean slope along the push at 80.5% is positive and every slope is finite, and "
               "(e4) the best grid throttle for Q1 - W lies past 80% on fewer states than for Q1 alone. The other band "
               "forms are measured the same way and reported. Stalling world and replication: stop that seed on "
               "failure. No-stall world: reported only")),
        deviations=[
            ("Precision of the in-program simulator step: the plan asks for float64. A float64 step inside the float32 "
             "training program needs the program traced under jax enable_x64, which changes the gate's float32 actor "
             "arithmetic (measured before any room-test training, on random networks and random data: the gate's "
             "own actor-only program traced under enable_x64 differs by 1 ulp after one update). The step therefore "
             "runs in float32 with the same dynamics_jax function, as the gate's own check-3 oracle did "
             "(learners.LearnSettings.oracle_x64 = False). Its inputs are exact in float32; check (c) bounds the "
             "difference from the float64 step at every training row (tolerance 1e-3 value units; stop if "
             "exceeded)."),
            ("Long-run harm arm: the 250-step differentiable rollout is used as written; no fitted-surface "
             "fallback. Pre-spec timing on seed 0 (pre_spec_timing below; random start actor, results discarded): "
             "58 s per 1,000 steps against 3 s for a plain arm, about 19 minutes for its 20,000 steps, which keeps "
             "one seed near 1.3-1.5 process-hours, inside the plan's 1.5. (Unrolling the rollout's scan was slower.) "
             "e uses Q1 as the plan says; both e terms use the same float32 250-step function."),
            ("No-stall world: check 2's no-stall critics are stall-aware-controller critics (the gate trained no "
             "rising-thrust critic in that world), so the critic's own one-step error uses the stall-aware "
             "controller. That version is the no-stall perfect band; the plan's literal 'rising-thrust controller' "
             "version is run beside it and reported."),
            ("Ladders for push less and the random surfaces are not given in the plan: they are slope-normalised "
             "knobs 0.1, 0.3, 1, 3, 10 (see slope_normalisation). All ladders get the declared blind extension rule."),
            ("Replication: 'the host's own actor' is taken as its target copy (the one its critic's Bellman target "
             "read)."),
            ("The no-caution actor that is locked and scored is check 2's saved actor; check (a) shows the gate's "
             "code reproduces it bit for bit on every seed."),
            ("Room mark wording made exact: 'beats each gating control by at least 20 points' is the pooled mean of "
             "the paired strong-current difference; 'positive in at least 4 of 5 seeds' is applied to each such "
             "difference."),
            ("Plan step 7.3(b)'s reference values (about -0.73 at the actor's past-80% commands, about 0.107 "
             "elsewhere) were computed with Q1 on the Q(s, a) side. They apply only to that form. The trained "
             "residual takes the minimum over the two heads on both sides, as BCA's own residual does, and gives "
             "-0.589 and 0.145 on the readout's rows (seed 0), with a mean of +0.149 at the recorded commands "
             "(change A3). The amended main oracle measures from the recorded command, so that offset does not enter "
             "it (change A1); the literal arm keeps it."),
            ("The main oracle is the plan's residual measured from the recorded command and one-sided (change A1).")],
        run_order=("0. the user approves change A1; 1. seeds 0 and 1: train (lock) then eval, one pinned process "
                   "each; 2. futility on seeds 0 and 1; 3. only if it says continue: seeds 2-4 (train, eval), then "
                   "the no-stall arms (no_stall_train, no_stall_eval) and the replication on check 3's critic "
                   "(replication_train, replication_eval) on seeds 0-4; 4. pooled marks over seeds 0-4. Every train "
                   "phase writes its lock before its eval phase may run. Logs: runs/shift_explore/room/logs/."),
        gate_files_sha256=gate_file_hashes(),
        gate_prereg_sha256=file_sha256(ROOT / "runs" / "shift" / "preregistration.json"),
        outputs=str(OUT),
    )


def load_spec():
    spec = read_json(SPEC_PATH)
    if spec is None:
        raise SystemExit(f"refused: no spec at {SPEC_PATH}; it must be written before any training")
    return spec, file_sha256(SPEC_PATH)


def check_gate_files(spec):
    now = gate_file_hashes()
    bad = [k for k, v in spec["gate_files_sha256"].items() if now.get(k) != v]
    if bad:
        raise SystemExit("refused: gate files changed since the spec: " + ", ".join(bad))

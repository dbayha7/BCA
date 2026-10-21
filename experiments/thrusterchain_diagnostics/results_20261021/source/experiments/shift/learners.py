"""Learners for the ThrusterChain shift test. Every learner here is trained on one recorded log; nothing here judges a
pass mark (acceptance.py does that, after its pre-registration check).

What is trained (all with the repository's own TD3+BC networks: algorithms/td3_bc.Actor, a 13-256-256-4 ReLU network
with a tanh output, and algorithms/td3_bc.DualCritic, two 256-256 ReLU critics):

  plain TD3+BC (cotrain)   algorithms/td3_bc.make_train_step, unchanged: two critic heads, clipped double-Q target, target
                           smoothing noise 0.2 clipped at 0.5, discount 0.98, tau 0.005, Adam 3e-4, batch 256, actor
                           and target updates every 2 updates, alpha 2.5 (or the alpha the harm checks chose).
                           Initialisation is td3_bc.initialize's own key scheme from one seed per data seed. With a
                           per-row caution table (even caution, for example) the same update is called with its
                           bc_multiplier argument, which td3_bc_update already has; nothing in td3_bc.py is edited.
                           stream > 0 keeps the initial parameters and changes only the random stream of batch rows
                           and target noise (the "null pair" of the power check).
  fixed-controller critics  (fixed_critic) the same critic update, with the actor frozen at a fixed controller: the
                           stall-aware controller (the textbook gain capped at 80%) or the rising-thrust controller
                           (capped at 100%). After each td3_bc_update call the actor and its target are put back, so
                           the critic follows td3_bc exactly while its targets always read the fixed controller. All
                           critics of one data seed share their initial parameters and their random stream; only the
                           rows differ. The placebo critic changes only its initial parameters.
  behaviour cloning (clone) the same actor network fitted to the recorded commands by Adam on the mean squared error,
                           from the same initial actor as the plain TD3+BC run.
  actor-only runs (ActorOnly) from the cloned actor, TD3+BC's own actor loss (-lambda Q1(s, pi(s)) + caution x
                           (pi(s) - a)^2, lambda = alpha / mean|Q1|) against a frozen critic, every step an actor step,
                           one shared stream of batch rows for every arm, the arms run side by side (jax.vmap). Arms
                           differ only in their frozen critic (the critic swap of harm check H2) or only in their
                           per-row caution (the room check H3).

Per-row caution in the room check (a multiplier on the behaviour-cloning term, as BCA's):
  even m                every row m.
  harm-placed           1 + 0.5 x / (1 + x) with x = 3 D_i / mean(D): BCA's own dose form, so it stays in [1, 1.5).
                        D_i is the exact over-preference of the arm's current actor at training row i (truth.py):
                        how much more the frozen critic prefers the actor's command over the recorded one than the
                        truth does. It is recomputed every 4,000 steps (float32 exact rollouts of 600 steps).
  cap-2                 1 + x / (1 + x) (BCA with blend 1.0; a diagnostic of whether BCA's dose cap is the limit).
  amount-matched        every row gets the mean harm-placed dose of the same refresh (same amount, no placement).
  shuffled k            the harm-placed doses of the same refresh, permuted by a fixed seeded permutation.
  reversed              the harm-placed doses with their ranks flipped (the largest dose on the smallest D_i).
  strength-matched even even caution whose strength equals the harm-placed arm's. Strength S is the root-mean-square
                        change of the command from the 'none' arm on 5,000 evaluation-split states. The multiplier is
                        found by experiments/signal/glq_match.match_parallel (interpolation, then bisection) from
                        strengths alone, before any return is computed.

BCA (switched off until the user approves). The BCA arms use algorithms/td3_bc_bca unchanged: its scale fit
(fit_scale), its frozen threshold refresh on the calibration bank (refresh; first after a 10,000-update warm-up, then
every 5,000 updates), its dose readout (bc_readout, blend 0.5, doses in [1, 1.5)) and td3_bc's update with the dose as
bc_multiplier. bca_update restates td3_bc_bca.update line for line and only passes the dose through a placement
transform: 'own' (unchanged: its trace is identical to td3_bc_bca.update's, tested), 'flattened' (each batch's doses
replaced by their mean), 'shuffled' (permuted within the batch by a separate key, fold_in of the update key, so the batch
stream is unchanged) and 'noise_map' (the same doses re-assigned so the largest dose goes to the row with the largest
noise map). Every function that would fit a calibrator, refresh a threshold, read a dose or run a BCA update calls
require_approval first and refuses (ApprovalError) without a valid approval file: a JSON file that holds the SHA-256 of
the gate report, written only by `acceptance.py approve`. trace_bca builds and lowers the BCA program without running
it; that is all the tests do with BCA.

Seeds. np.random.default_rng([settings.seed, rep, TAG]) for every stream of data seed `rep`; settings.seed is
20261015. td3_bc.Args.seed of rep is default_rng([settings.seed, rep, TAG_INIT]).integers(2^31 - 1).

Normalisation. Every learner of one (data seed, world) normalises observations by one rule: the mean and SD (+ 0.001)
of the float32 observations of the gapped log's kept training rows (the runtime's compute_mean_std), so learners on the
full, gapped and randomly thinned logs see identical inputs for identical rows. Rewards are raw.

Tests: JAX_PLATFORMS=cpu python -m unittest experiments.shift.test_learners
"""

import hashlib
import json
import math
import time
from dataclasses import asdict, dataclass, field, replace
from functools import partial
from pathlib import Path

import numpy as np

import jax
import jax.numpy as jnp
import optax
from flax import serialization
from flax.training.train_state import TrainState

import algorithms.td3_bc as BASE
import algorithms.td3_bc_bca as P
from calibration.network import Calibrator
from calibration.reference import WBCPConfig, initial_reference
from experiments.shift import recorder as RC
from experiments.shift import thruster_chain as TC
from experiments.shift import truth as TR
from experiments.signal import glq_match as M
from runtime.networks import Transition

DX, DA = TC.DX, TC.DA
LEARNER_SEED = 20261015
KEY_TD3_INIT = 1414676809          # td3_bc.initialize's fold_in constant
KEY_CALIBRATOR = 1128352841        # td3_bc_bca.initialize's calibrator fold_in constant
TAG_INIT, TAG_PLACEBO, TAG_STREAM, TAG_CLONE, TAG_ACTOR_ONLY = 31, 32, 33, 34, 35
TAG_SHUFFLE, TAG_TIE, TAG_STRENGTH, TAG_REFRESH, TAG_PLACEMENT, TAG_BAND = 36, 37, 38, 39, 40, 41
NORM_EPS = 0.001
CONTROLLERS = ("stall_aware", "rising_thrust")
CONTROLLER_LABELS = {"stall_aware": "stall-aware controller", "rising_thrust": "rising-thrust controller"}
ROOM_EVEN_LADDER = (1.1, 1.25, 1.5, 2.0, 3.0)
DOSE_GAIN = 3.0                    # x = 3 D_i / mean(D)
PLACEMENTS = ("own", "flattened", "shuffled", "noise_map")
BCA_SETTINGS = dict(miscoverage=0.1, credibility=0.95, draws=1000, blend=0.5, cal_lr=0.001, cal_beta=20.0,
                    width_penalty=0.005, scale_ema=0.99, warmup=10_000, refresh_every=5_000)
BCA_REFUSAL = ("refused: BCA is switched off. No BCA arm, calibrator fit, threshold refresh or dose readout may run "
               "before the user approves the gate report: run `python -m experiments.shift.acceptance approve` "
               "(the user, at a terminal) and pass the approval file it writes")

ACTOR = BASE.Actor(DX, DA, 1.0)
CRITIC = BASE.DualCritic(DX, DA)


@dataclass(frozen=True)
class LearnSettings:
    """Every learner number. updates: co-trained TD3+BC and critic updates (100,000). snapshots: fractions of the
    updates at which the whole state is kept (50% reported, 100% primary). clone_steps: behaviour-cloning steps that
    start every actor-only run (20,000); clone_arm_steps: the cloning-only comparison arm (50,000: as many actor
    updates as a TD3+BC run with policy delay 2). actor_only_steps: steps of every actor-only arm (20,000).
    oracle_refresh: steps between exact over-preference refreshes (4,000), with float32 rollouts of oracle_horizon
    (600) steps. strength_states: evaluation-split states for the strength S (5,000). chunk: updates per compiled
    call. band_fit_steps: scale-fit steps of BCA's band on a frozen critic (20,000)."""

    seed: int = LEARNER_SEED
    updates: int = 100_000
    gamma: float = 0.98
    tau: float = 0.005
    lr: float = 3e-4
    batch_size: int = 256
    policy_freq: int = 2
    policy_noise: float = 0.2
    noise_clip: float = 0.5
    alpha: float = 2.5
    snapshots: tuple = (0.5, 1.0)
    chunk: int = 5_000
    clone_steps: int = 20_000
    clone_arm_steps: int = 50_000
    actor_only_steps: int = 20_000
    actor_only_chunk: int = 4_000
    oracle_refresh: int = 4_000
    oracle_horizon: int = 600
    oracle_x64: bool = False
    strength_states: int = 5_000
    match_tol: float = M.MATCH_TOL
    match_rounds: int = M.MATCH_ROUNDS
    band_fit_steps: int = 20_000
    bca_warmup: int = BCA_SETTINGS["warmup"]
    bca_refresh: int = BCA_SETTINGS["refresh_every"]
    row_chunk: int = 8_192

    def as_dict(self):
        return asdict(self)


def make_args(settings, rep, alpha=None):
    """td3_bc.Args of data seed rep: the settings' TD3+BC numbers, the given alpha and the rep's seed."""
    seed = int(np.random.default_rng([settings.seed, int(rep), TAG_INIT]).integers(2 ** 31 - 1))
    return replace(BASE.Args(), seed=seed, dataset="bca-shift-thrusterchain", num_updates=settings.updates,
                   discount=settings.gamma, tau=settings.tau, lr=settings.lr, batch_size=settings.batch_size,
                   policy_freq=settings.policy_freq, policy_noise=settings.policy_noise,
                   noise_clip=settings.noise_clip, alpha=float(settings.alpha if alpha is None else alpha))


def _args_key(args):
    return tuple(sorted(asdict(args).items()))


_TX = {}


def adam(lr):
    """One shared runtime.networks.torch_adam per learning rate, so that train states stack under vmap."""
    if lr not in _TX:
        _TX[lr] = BASE.C.torch_adam(lr)
    return _TX[lr]


def rng_np(settings, rep, *tags):
    return np.random.default_rng([int(settings.seed), int(rep), *[int(t) for t in tags]])


# ------------------------------------------------------------------------------------------------ data

def tree_sha256(tree):
    h = hashlib.sha256()
    for x in jax.tree_util.tree_leaves(tree):
        a = np.ascontiguousarray(np.asarray(x))
        h.update(str(a.dtype).encode() + str(a.shape).encode())
        h.update(a.tobytes())
    return h.hexdigest()


def normalization(log, removal):
    """(mean, std) float32 of the float32 observations of the removal's kept training rows; std + 0.001 (the
    runtime's compute_mean_std)."""
    obs = np.asarray(log["observations"][removal.kept_train_rows(log)], np.float32)
    return obs.mean(0), obs.std(0) + np.float32(NORM_EPS)


@dataclass
class Prepared:
    """One log, one removal and one normalisation, ready for the learners.

    rows: the kept training rows (log indices), in order; states: float64 of their float32 observations (what the
    critic sees, the state every exact value of a row starts from); actions: their float32 commands; region: the
    full-throttle region (push above the trigger) at each row; train: the normalised transitions (runtime.networks.
    Transition, float32; done = terminals, all False, so TD3+BC bootstraps through the time limit); bank_rows / bank:
    the calibration bank after the removal, normalised the same way; eval_obs: normalised evaluation-split states for
    the strength S; physics and spec of the log."""

    rep: int
    world: str
    removal: str
    physics: object
    spec: object
    rows: np.ndarray
    states: np.ndarray
    actions: np.ndarray
    region: np.ndarray
    train: Transition
    bank_rows: np.ndarray
    bank: Transition
    obs_mean: np.ndarray
    obs_std: np.ndarray
    eval_rows: np.ndarray
    eval_obs: jnp.ndarray
    sha256: dict = field(default_factory=dict)

    @property
    def n(self):
        return int(len(self.rows))

    def normalize(self, states):
        """Normalised float32 observations of float64 states (rounded to float32 first, as the environment does)."""
        return (np.asarray(states, np.float32) - self.obs_mean) / self.obs_std

    def describe(self):
        return dict(rep=self.rep, world=self.world, removal=self.removal, rows=self.n, bank_rows=int(len(self.bank_rows)),
                    eval_states=int(len(self.eval_rows)), obs_mean=self.obs_mean.tolist(),
                    obs_std=self.obs_std.tolist(), sha256=self.sha256)


def _transition(log, rows, mean, std):
    obs = (np.asarray(log["observations"][rows], np.float32) - mean) / std
    nxt = (np.asarray(log["next_observations"][rows], np.float32) - mean) / std
    return Transition(jnp.asarray(obs), jnp.asarray(np.asarray(log["actions"][rows], np.float32)),
                      jnp.asarray(np.asarray(log["rewards"][rows], np.float32)), jnp.asarray(nxt),
                      jnp.asarray(np.asarray(log["terminals"][rows], np.float32)))


def prepare(log, removal, norm, settings=None):
    """Prepared data of a log under a removal with the normalisation norm = (mean, std) (normalization())."""
    settings = settings or LearnSettings()
    mean, std = (np.asarray(v, np.float32) for v in norm)
    rows = removal.kept_train_rows(log)
    obs32 = np.asarray(log["observations"][rows], np.float32)
    bank_rows = removal.bank_rows(log)
    ev = log.split_rows("evaluation")
    pick = np.sort(rng_np(settings, log.meta["rep"], TAG_STRENGTH).choice(
        len(ev), min(settings.strength_states, len(ev)), replace=False))
    eval_rows = ev[pick]
    recorder = RC.Recorder(log.spec, log.physics)
    train = _transition(log, rows, mean, std)
    bank = _transition(log, bank_rows, mean, std)
    eval_obs = jnp.asarray((np.asarray(log["observations"][eval_rows], np.float32) - mean) / std)
    sha = dict(rows=tree_sha256(rows), train=tree_sha256(train), bank_rows=tree_sha256(bank_rows),
               bank=tree_sha256(bank), normalization=tree_sha256((mean, std)), eval_rows=tree_sha256(eval_rows))
    return Prepared(rep=int(log.meta["rep"]), world=log.meta["world"], removal=removal.kind, physics=log.physics,
                    spec=log.spec, rows=rows, states=obs32.astype(np.float64), actions=np.asarray(log["actions"][rows],
                                                                                                np.float32),
                    region=recorder.in_region(obs32.astype(np.float64)), train=train, bank_rows=bank_rows, bank=bank,
                    obs_mean=mean, obs_std=std, eval_rows=eval_rows, eval_obs=eval_obs, sha256=sha)


# ------------------------------------------------------------------------------------------------ snapshots

def save_tree(path, tree):
    """Flax msgpack of a pytree; returns its SHA-256."""
    blob = serialization.to_bytes(tree)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_bytes(blob)
    return hashlib.sha256(blob).hexdigest()


def load_tree(path, template, sha256=None):
    blob = Path(path).read_bytes()
    if sha256 is not None and hashlib.sha256(blob).hexdigest() != sha256:
        raise ValueError(f"{path}: SHA-256 differs from the recorded one")
    return serialization.from_bytes(template, blob)


def actor_template():
    return ACTOR.init(jax.random.PRNGKey(0), jnp.zeros(DX))


def critic_template():
    return CRITIC.init(jax.random.PRNGKey(0), jnp.zeros(DX), jnp.zeros(DA))


def agent_template(settings=None):
    settings = settings or LearnSettings()
    _, state, _ = BASE.initialize(make_args(settings, 0), DX, DA, 1.0)
    return state


# ------------------------------------------------------------------------------------------------ rows

_ROW_PROGRAMS = {}


def _row_program(grad):
    if grad not in _ROW_PROGRAMS:
        if grad:
            def program(p, o, a):
                q, vjp = jax.vjp(lambda u: CRITIC.apply(p, o, u), a)
                return q, vjp(jnp.zeros_like(q).at[:, 0].set(1.0))[0]
        else:
            def program(p, o, a):
                return CRITIC.apply(p, o, a), jnp.zeros_like(a)
        _ROW_PROGRAMS[grad] = jax.jit(program)
    return _ROW_PROGRAMS[grad]


def critic_rows(critic_params, obs_norm, action, grad=False, chunk=8192):
    """Both critic heads Q (n, 2) and, with grad, grad_a Q1 (n, 4), at float32 normalised observations and float32
    commands (the critic's own inputs); float64 out; padded chunks so that one program serves every size."""
    program = _row_program(bool(grad))
    obs, act = np.asarray(obs_norm, np.float32), np.asarray(action, np.float32)
    qs, gs = [], []
    for i in range(0, len(obs), chunk):
        o, a = obs[i:i + chunk], act[i:i + chunk]
        pad = chunk - len(o)
        if pad:
            o = np.concatenate([o, np.zeros((pad, DX), np.float32)])
            a = np.concatenate([a, np.zeros((pad, DA), np.float32)])
        q, g = program(critic_params, jnp.asarray(o), jnp.asarray(a))
        qs.append(np.asarray(q, np.float64)[:chunk - pad])
        gs.append(np.asarray(g, np.float64)[:chunk - pad])
    if not qs:
        return np.zeros((0, 2)), np.zeros((0, DA))
    return np.concatenate(qs), np.concatenate(gs)


_ACTOR_ROWS = {}


def actor_rows(actor_params, obs_norm, chunk=8192):
    """The float32 actor's commands at normalised observations (float32 out)."""
    if "fn" not in _ACTOR_ROWS:
        _ACTOR_ROWS["fn"] = jax.jit(ACTOR.apply)
    obs = np.asarray(obs_norm, np.float32)
    out = []
    for i in range(0, len(obs), chunk):
        o = obs[i:i + chunk]
        pad = chunk - len(o)
        if pad:
            o = np.concatenate([o, np.zeros((pad, DX), np.float32)])
        out.append(np.asarray(_ACTOR_ROWS["fn"](actor_params, jnp.asarray(o)), np.float32)[:chunk - pad])
    return np.concatenate(out) if out else np.zeros((0, DA), np.float32)


def strength(params_arm, params_none, eval_obs):
    """S: the root-mean-square change of the command from the 'none' arm on the evaluation states."""
    a = actor_rows(params_arm, eval_obs).astype(np.float64)
    b = actor_rows(params_none, eval_obs).astype(np.float64)
    return float(np.sqrt(np.mean(np.sum((a - b) ** 2, axis=1))))


# ------------------------------------------------------------------------------------------------ exact values of an MLP

def mlp_policy(actor_params, prepared, name="TD3+BC actor (float64 copy)"):
    """truth.MLPPolicy of a learned actor: the same network in float64, seeing float32-rounded observations."""
    return TR.MLPPolicy(actor_params, prepared.obs_mean, prepared.obs_std, name=name)


_EXACT = {}


def _mlp_layers(actor_params, dtype):
    p = actor_params.get("params", actor_params)
    names = sorted(p, key=lambda k: int(str(k).rsplit("_", 1)[1]))
    Ws = tuple(jnp.asarray(np.asarray(p[k]["kernel"], np.float64), dtype) for k in names)
    bs = tuple(jnp.asarray(np.asarray(p[k]["bias"], np.float64), dtype) for k in names)
    return Ws, bs


def _exact_program(physics, horizon, gamma, x64):
    key = (physics, int(horizon), float(gamma), bool(x64))
    if key not in _EXACT:
        def q(Ws, bs, mean, std, x, a):
            def pi(z):
                o = z.astype(jnp.float32).astype(z.dtype)
                h = (o - mean) / std
                h = jnp.maximum(h @ Ws[0] + bs[0], 0.0)
                h = jnp.maximum(h @ Ws[1] + bs[1], 0.0)
                return 1.0 * jnp.tanh(h @ Ws[2] + bs[2])

            x1, r0 = TC.dynamics_jax(x, a, physics)

            def body(carry, _):
                xx, acc, disc = carry
                disc = disc * gamma
                xx2, rr = TC.dynamics_jax(xx, pi(xx), physics)
                return (xx2, acc + disc * rr, disc), None

            (_, acc, _), _ = jax.lax.scan(body, (x1, r0, jnp.asarray(1.0, x.dtype)), None, length=int(horizon) - 1)
            return acc

        _EXACT[key] = jax.jit(q)
    return _EXACT[key]


def exact_q(actor_params, prepared, states, actions, physics=None, horizon=TR.HORIZON, x64=True, chunk=8192):
    """Q^pi(s, a) of a learned actor by one rollout through thruster_chain.dynamics_jax (truth.py's rule), with the
    actor's weights as arguments so that refreshing them never recompiles. x64=True: float64 (exact); x64=False: the
    float32 fast path for repeated refreshes (relative error about 1e-6)."""
    physics = prepared.physics if physics is None else physics
    context, dtype = TR._precision(x64)
    x, a = np.asarray(states, dtype), np.asarray(actions, dtype)
    out = []
    with context:
        jdt = jnp.float64 if x64 else jnp.float32
        Ws, bs = _mlp_layers(actor_params, jdt)
        mean = jnp.asarray(np.asarray(prepared.obs_mean, np.float64), jdt)
        std = jnp.asarray(np.asarray(prepared.obs_std, np.float64), jdt)
        fn = _exact_program(physics, horizon, physics.gamma, x64)
        for i in range(0, len(x), chunk):
            xs, as_ = x[i:i + chunk], a[i:i + chunk]
            pad = chunk - len(xs)
            if pad:
                xs = np.concatenate([xs, np.repeat(xs[-1:], pad, 0)])
                as_ = np.concatenate([as_, np.repeat(as_[-1:], pad, 0)])
            out.append(np.asarray(fn(Ws, bs, mean, std, jnp.asarray(xs), jnp.asarray(as_)), np.float64)[:chunk - pad])
    return np.concatenate(out) if out else np.zeros(0)


def over_preference_rows(actor_params, critic_params, prepared, rows=None, horizon=TR.HORIZON, x64=True,
                         q_logged=None, chunk=8192):
    """Exact over-preference D_i of an actor and a critic at training rows (all by default), at the inputs the critic
    sees: D = [(Q1(s, pi(s)) - Q1(s, a)) - (Q^pi(s, pi(s)) - Q^pi(s, a))]_+. Also returns its parts. q_logged:
    Q1 at the recorded commands, when already known (a frozen critic)."""
    idx = np.arange(prepared.n) if rows is None else np.asarray(rows, np.int64)
    obs = np.asarray(prepared.train.obs)[idx]
    a32 = prepared.actions[idx]
    pi32 = actor_rows(actor_params, obs)
    q_pi = critic_rows(critic_params, obs, pi32, chunk=chunk)[0][:, 0]
    q_a = critic_rows(critic_params, obs, a32, chunk=chunk)[0][:, 0] if q_logged is None else q_logged[idx]
    s = prepared.states[idx]
    t_pi = exact_q(actor_params, prepared, s, pi32.astype(np.float64), horizon=horizon, x64=x64, chunk=chunk)
    t_a = exact_q(actor_params, prepared, s, a32.astype(np.float64), horizon=horizon, x64=x64, chunk=chunk)
    D = TR.over_preference(q_pi, q_a, t_pi, t_a)
    return dict(D=D, q1_policy=q_pi, q1_logged=q_a, true_policy=t_pi, true_logged=t_a, policy=pi32)


# ------------------------------------------------------------------------------------------------ plain TD3+BC

def _host_step_weighted(args, train, table, runner, _):
    """td3_bc.make_train_step's step with a per-row caution table passed as td3_bc_update's bc_multiplier."""
    rng, agent_state, it = runner
    it = it + 1
    rng, rng_batch, rng_noise = jax.random.split(rng, 3)
    idx = jax.random.randint(rng_batch, (args.batch_size,), 0, train.obs.shape[0])
    batch = jax.tree_util.tree_map(lambda x: x[idx], train)
    agent_state, loss = BASE.td3_bc_update(args, ACTOR.apply, CRITIC.apply, agent_state, batch, it, rng_noise, 1.0,
                                           bc_multiplier=table[idx])
    return (rng, agent_state, it), loss


_PROGRAMS = {}


def host_program(args, length, weighted):
    """Compiled lax.scan of `length` plain TD3+BC updates: td3_bc.make_train_step itself (weighted=False) or the same
    step with a caution table."""
    key = ("host", _args_key(args), int(length), bool(weighted))
    if key not in _PROGRAMS:
        if weighted:
            def run(runner, train, table):
                return jax.lax.scan(partial(_host_step_weighted, args, train, table), runner, None, length=length)
        else:
            def run(runner, train, table):
                step = BASE.make_train_step(args, ACTOR.apply, CRITIC.apply, train, 1.0)
                return jax.lax.scan(step, runner, None, length=length)
        _PROGRAMS[key] = jax.jit(run)
    return _PROGRAMS[key]


def _events(total, chunk, snapshots, extra=()):
    ev = set(range(chunk, total + 1, chunk)) | set(snapshots) | {total} | set(extra)
    return sorted(e for e in ev if 0 < e <= total)


def snapshot_steps(settings, total):
    steps = sorted({int(round(f * total)) for f in settings.snapshots})
    if any(s <= 0 for s in steps):
        raise ValueError("snapshot fractions must give positive update counts")
    return steps


def _finite_tree(tree):
    return all(bool(np.all(np.isfinite(np.asarray(x)))) for x in jax.tree_util.tree_leaves(tree)
               if np.asarray(x).dtype.kind == "f")


def _chunk_summary(metrics, policy_freq=None):
    out = {}
    for k, v in metrics.items():
        v = np.asarray(v)
        if v.dtype.kind in "fc":
            out[k] = float(np.mean(v))
        elif v.dtype == bool:
            out[k] = float(np.mean(v))
    return out


BATCH_STREAM_CHECK = 64   # batch draws hashed per co-trained run to show that the arms of a seed share their stream


def batch_stream(rng, n, batch_size, steps=BATCH_STREAM_CHECK):
    """The first `steps` batch-row draws of a co-trained run that starts from `rng`: the split pattern of
    td3_bc.make_train_step (and of td3_bc_bca.make_train_step and bca_train_step): rng -> (rng, rng_batch, rng_noise),
    rows = randint(rng_batch, (batch_size,), 0, n). Returns an (steps, batch_size) int array."""
    out = []
    for _ in range(int(steps)):
        rng, kb, _ = jax.random.split(rng, 3)
        out.append(np.asarray(jax.random.randint(kb, (int(batch_size),), 0, int(n))))
    return np.stack(out).astype(np.int64)


def batch_stream_sha256(rng, n, batch_size, steps=BATCH_STREAM_CHECK):
    return hashlib.sha256(np.ascontiguousarray(batch_stream(rng, n, batch_size, steps)).tobytes()).hexdigest()


def cotrain(prepared, settings, rep, *, alpha=None, table=None, stream=0, updates=None, out_dir=None, log=None,
            name="plain TD3+BC"):
    """One plain TD3+BC run (module docstring). table: per-row caution (n,) or None. stream: 0 is the run's own
    random stream; k > 0 keeps the initial parameters and draws batch rows and target noise from fold_in(rng, 33 + k).
    Returns (record, final AgentTrainState, {update: AgentTrainState} at the snapshots)."""
    started = time.perf_counter()
    args = make_args(settings, rep, alpha)
    total = int(settings.updates if updates is None else updates)
    rng, state, _ = BASE.initialize(args, DX, DA, 1.0)
    init_sha = tree_sha256((state.actor.params, state.critic.params))
    if stream:
        rng = jax.random.fold_in(rng, TAG_STREAM + int(stream))
    weighted = table is not None
    tab = jnp.asarray(np.asarray(table, np.float32)) if weighted else jnp.zeros(1, jnp.float32)
    if weighted and tab.shape != (prepared.n,):
        raise ValueError("a caution table holds one value per kept training row")
    snaps = snapshot_steps(settings, total)
    runner, done, chunks, states = (rng, state, jnp.int32(0)), 0, [], {}
    record = dict(kind="shift-cotrain", name=name, rep=int(rep), args=asdict(args), stream=int(stream),
                  weighted=weighted, table_sha256=tree_sha256(np.asarray(table, np.float32)) if weighted else None,
                  updates=total, data=prepared.describe(), initial_sha256=init_sha,
                  batch_stream_sha256=batch_stream_sha256(rng, prepared.n, args.batch_size), snapshots=[],
                  failed=False)
    for step in _events(total, settings.chunk, snaps):
        runner, metrics = host_program(args, step - done, weighted)(runner, prepared.train, tab)
        done = step
        chunks.append(dict(_chunk_summary(metrics), update=step))
        if not _finite_tree(metrics) or not _finite_tree(runner[1].actor.params):
            record.update(failed=True, failed_at=step)
            break
        if step in snaps:
            states[step] = runner[1]
            snap = dict(update=step, actor_sha256=tree_sha256(runner[1].actor.params),
                        critic_sha256=tree_sha256(runner[1].critic.params))
            if out_dir is not None:
                snap["file_sha256"] = save_tree(Path(out_dir) / f"snapshot_{int(round(100 * step / total)):03d}.msgpack",
                                                runner[1])
            record["snapshots"].append(snap)
        if log:
            log(f"{name} rep {rep}: update {step}/{total}, critic loss {chunks[-1].get('critic_loss', float('nan')):.4g}")
    record.update(chunks=chunks, final_update=done, seconds=time.perf_counter() - started)
    return record, runner[1], states


# ------------------------------------------------------------------------------------------------ fixed controllers

class FixedController:
    """A fixed controller as an actor of the normalised observation, for the critic's targets: the observation is
    un-normalised (o x std + mean, float32) and the command is clip(-K x, -cap, cap), K the textbook gain
    (recorder.textbook_gain): cap 0.8 for the stall-aware controller, 1.0 for the rising-thrust controller. Its
    params {'K': K} take td3_bc_update's (discarded) actor step."""

    def __init__(self, name, physics, mean, std):
        if name not in CONTROLLERS:
            raise ValueError(f"controller must be one of {CONTROLLERS}")
        self.name = name
        self.cap = 0.8 if name == "stall_aware" else 1.0
        self.K = np.asarray(RC.textbook_gain(physics), np.float32)
        self.mean = jnp.asarray(np.asarray(mean, np.float32))
        self.std = jnp.asarray(np.asarray(std, np.float32))

    def params(self):
        return {"K": jnp.asarray(self.K)}

    def apply(self, params, obs):
        x = obs * self.std + self.mean
        return jnp.clip(-(x @ params["K"].T), -self.cap, self.cap)

    def numpy(self, physics):
        """The same controller on float64 states (recorder.LinearController): the truth's policy."""
        return RC.stall_aware(physics) if self.name == "stall_aware" else RC.rising_thrust(physics)


def _fixed_step(args, fixed, train, runner, _):
    """td3_bc.make_train_step's layout around td3_bc_update, then the actor and its target are put back."""
    rng, state, it = runner
    it = it + 1
    rng, rng_batch, rng_noise = jax.random.split(rng, 3)
    idx = jax.random.randint(rng_batch, (args.batch_size,), 0, train.obs.shape[0])
    batch = jax.tree_util.tree_map(lambda x: x[idx], train)
    new, loss = BASE.td3_bc_update(args, fixed.apply, CRITIC.apply, state, batch, it, rng_noise, 1.0)
    new = new._replace(actor=state.actor, actor_target=state.actor_target)
    return (rng, new, it), dict(critic_loss=loss["critic_loss"])


def fixed_critic(prepared, controller, settings, rep, *, placebo=False, updates=None, out_dir=None, log=None):
    """A fixed-controller critic (module docstring). placebo=True changes only the critic's initial parameters
    (default_rng([seed, rep, 32]) seeds them); the random stream of rows and target noise stays the run's own.
    Returns (record, final AgentTrainState)."""
    started = time.perf_counter()
    args = make_args(settings, rep)
    total = int(settings.updates if updates is None else updates)
    fixed = FixedController(controller, prepared.physics, prepared.obs_mean, prepared.obs_std)
    rng = jax.random.fold_in(jax.random.PRNGKey(args.seed), KEY_TD3_INIT)
    rng, _, kc = jax.random.split(rng, 3)
    if placebo:
        other = int(rng_np(settings, rep, TAG_PLACEBO).integers(2 ** 31 - 1))
        kc = jax.random.split(jax.random.fold_in(jax.random.PRNGKey(other), KEY_TD3_INIT), 3)[2]
    critic = TrainState.create(apply_fn=CRITIC.apply, params=CRITIC.init(kc, jnp.zeros(DX), jnp.zeros(DA)),
                               tx=adam(settings.lr))
    actor = TrainState.create(apply_fn=fixed.apply, params=fixed.params(), tx=adam(settings.lr))
    state = BASE.AgentTrainState(actor, actor, critic, critic)
    key = ("fixed", _args_key(args), controller, tree_sha256((fixed.mean, fixed.std)))
    runner, done, chunks = (rng, state, jnp.int32(0)), 0, []
    record = dict(kind="shift-fixed-critic", controller=controller, label=CONTROLLER_LABELS[controller],
                  placebo=bool(placebo), rep=int(rep), args=asdict(args), updates=total, data=prepared.describe(),
                  initial_sha256=tree_sha256(critic.params), failed=False)
    for step in _events(total, settings.chunk, ()):
        length = step - done
        pkey = key + (length,)
        if pkey not in _PROGRAMS:
            _PROGRAMS[pkey] = jax.jit(lambda r, t, length=length: jax.lax.scan(
                partial(_fixed_step, args, fixed, t), r, None, length=length))
        runner, metrics = _PROGRAMS[pkey](runner, prepared.train)
        done = step
        chunks.append(dict(critic_loss=float(np.mean(np.asarray(metrics["critic_loss"]))), update=step))
        if not _finite_tree(metrics) or not _finite_tree(runner[1].critic.params):
            record.update(failed=True, failed_at=step)
            break
        if log:
            log(f"{CONTROLLER_LABELS[controller]} critic rep {rep}: update {step}/{total}, "
                f"critic loss {chunks[-1]['critic_loss']:.4g}")
    record.update(chunks=chunks, final_update=done, critic_sha256=tree_sha256(runner[1].critic.params),
                  seconds=time.perf_counter() - started)
    if out_dir is not None:
        record["file_sha256"] = save_tree(Path(out_dir) / "critic.msgpack", runner[1].critic.params)
    return record, runner[1]


# ------------------------------------------------------------------------------------------------ cloning

def initial_actor(settings, rep):
    """The actor's initial parameters of td3_bc.initialize for data seed rep (the plain TD3+BC run's own)."""
    rng = jax.random.fold_in(jax.random.PRNGKey(make_args(settings, rep).seed), KEY_TD3_INIT)
    _, ka, _ = jax.random.split(rng, 3)
    return ACTOR.init(ka, jnp.zeros(DX))


def _clone_loop(train, actor, rows):
    def body(a, r):
        def loss_fn(p):
            return jnp.square(ACTOR.apply(p, train.obs[r]) - train.action[r]).mean()

        loss, g = jax.value_and_grad(loss_fn)(a.params)
        return a.apply_gradients(grads=g), loss

    return jax.lax.scan(body, actor, rows)


def clone(prepared, settings, rep, steps=None, *, chunk=5_000, log=None):
    """Behaviour cloning on the prepared rows: mean squared error to the recorded commands, Adam (torch_adam, lr),
    batch 256, rows default_rng([seed, rep, 34]). Starts from the plain TD3+BC run's initial actor. Returns (actor
    params, record)."""
    steps = int(settings.clone_steps if steps is None else steps)
    actor = TrainState.create(apply_fn=ACTOR.apply, params=initial_actor(settings, rep), tx=adam(settings.lr))
    rows = rng_np(settings, rep, TAG_CLONE).integers(prepared.n, size=(steps, settings.batch_size))
    key = ("clone",)
    if key not in _PROGRAMS:
        _PROGRAMS[key] = jax.jit(_clone_loop)
    losses = []
    for i in range(0, steps, chunk):
        actor, loss = _PROGRAMS[key](prepared.train, actor, jnp.asarray(rows[i:i + chunk], jnp.int32))
        losses.append(float(np.mean(np.asarray(loss))))
        if log:
            log(f"cloning rep {rep}: step {min(i + chunk, steps)}/{steps}, loss {losses[-1]:.4g}")
    record = dict(kind="shift-clone", rep=int(rep), steps=steps, losses=losses, data=prepared.describe(),
                  rows_sha256=tree_sha256(rows), actor_sha256=tree_sha256(actor.params),
                  failed=not _finite_tree(actor.params))
    return actor.params, record


# ------------------------------------------------------------------------------------------------ actor-only arms

def actor_step(args, actor, critic_params, batch, bc_w=None):
    """td3_bc's actor step against a frozen critic: its _actor_loss_fn expression by expression, then
    apply_gradients. bc_w: per-row caution (None: td3_bc's unweighted path). Returns (actor, (loss, q mean, lambda,
    bc term))."""

    def _actor_loss_fn(params, critic_params):
        pi = ACTOR.apply(params, batch.obs)
        q = CRITIC.apply(critic_params, batch.obs, pi)[..., 0]
        lmbda = args.alpha / jax.lax.stop_gradient(jnp.abs(q).mean())
        if bc_w is None:
            bc = jnp.square(pi - batch.action).mean()
        else:
            bc = (jax.lax.stop_gradient(bc_w) * jnp.square(pi - batch.action).mean(axis=-1)).mean()
        return (-lmbda * q.mean() + bc, (q.mean(), lmbda, bc))

    grad_fn = jax.value_and_grad(_actor_loss_fn, has_aux=True)
    (a_loss, aux), a_grad = grad_fn(actor.params, critic_params)
    return actor.apply_gradients(grads=a_grad), (a_loss, *aux)


def _actor_only_loop(args, weighted, critic_params, actor, train, rows, table):
    def body(a, r):
        batch = jax.tree_util.tree_map(lambda x: x[r], train)
        a, aux = actor_step(args, a, critic_params, batch, table[r] if weighted else None)
        return a, jnp.stack(aux)

    return jax.lax.scan(body, actor, rows)


@dataclass(frozen=True)
class Arm:
    """One actor-only arm. kind: 'even' (value = the multiplier; 1.0 is 'none'), 'harm' (harm-placed), 'cap2',
    'amount', 'shuffled' (index = which fixed permutation), 'reversed' (these three reuse the doses of the 'harm' arm
    named by source at each refresh). critic: index into the list of critics (critic swap) or 0."""

    name: str
    kind: str = "even"
    value: float = 1.0
    source: str = None
    index: int = 0
    critic: int = 0


ORACLE_KINDS = ("harm", "cap2")
DERIVED_KINDS = ("amount", "shuffled", "reversed")


def harm_doses(D, kind="harm"):
    """BCA's dose form on the exact over-preference: x = 3 D / mean(D) (0 when D is all zero); 'harm' gives
    1 + 0.5 x / (1 + x) in [1, 1.5), 'cap2' gives 1 + x / (1 + x) in [1, 2)."""
    D = np.asarray(D, np.float64)
    m = D.mean() if len(D) else 0.0
    x = DOSE_GAIN * D / m if m > 0 else np.zeros_like(D)
    scale = 0.5 if kind == "harm" else 1.0
    return 1.0 + scale * x / (1.0 + x)


def reverse_ranks(doses, D, tie):
    """The same dose multiset with its ranks flipped against D: the largest dose on the smallest D (ties of D broken
    by the fixed permutation tie)."""
    order = np.lexsort((tie, np.asarray(D)))
    out = np.empty(len(doses))
    out[order] = np.sort(np.asarray(doses, np.float64))[::-1]
    return out


class ActorOnly:
    """One data seed's actor-only arms on frozen critics, from one start actor and one shared stream of batch rows
    (default_rng([seed, rep, 35])), vmapped over the arms."""

    def __init__(self, prepared, start_params, settings, rep, *, alpha=None, steps=None, chunk=None):
        self.prepared, self.settings, self.rep = prepared, settings, int(rep)
        self.args = make_args(settings, rep, alpha)
        self.steps = int(settings.actor_only_steps if steps is None else steps)
        self.chunk = int(settings.actor_only_chunk if chunk is None else chunk)
        if self.steps % self.chunk:
            raise ValueError("actor-only steps must be a multiple of the chunk (the oracle refresh interval)")
        rows = rng_np(settings, rep, TAG_ACTOR_ONLY).integers(prepared.n, size=(self.steps, settings.batch_size))
        self.rows_sha256 = tree_sha256(rows)
        self.rows = jnp.asarray(rows.reshape(self.steps // self.chunk, self.chunk, settings.batch_size), jnp.int32)
        self.start_params = start_params
        self.start = TrainState.create(apply_fn=ACTOR.apply, params=start_params, tx=adam(settings.lr))
        n = prepared.n
        self.tie = rng_np(settings, rep, TAG_TIE).permutation(n)
        self.perms = {k: rng_np(settings, rep, TAG_SHUFFLE, k).permutation(n) for k in range(4)}
        self.oracle_calls = 0
        self._q_logged = {}

    def program(self, n_arms, shared, weighted):
        key = ("actor_only", _args_key(self.args), int(n_arms), bool(shared), bool(weighted), self.chunk)
        if key not in _PROGRAMS:
            loop = partial(_actor_only_loop, self.args, weighted)
            _PROGRAMS[key] = jax.jit(jax.vmap(loop, in_axes=(None if shared else 0, 0, None, None, 0)))
        return _PROGRAMS[key]

    def oracle(self, actor_params, critic_params):
        """Exact over-preference D at every training row for one arm's current actor (float32 fast path unless
        settings.oracle_x64)."""
        self.oracle_calls += 1
        critic_key = tree_sha256(critic_params)
        if critic_key not in self._q_logged:
            self._q_logged[critic_key] = critic_rows(critic_params, self.prepared.train.obs, self.prepared.actions)[0][:, 0]
        return over_preference_rows(actor_params, critic_params, self.prepared, horizon=self.settings.oracle_horizon,
                                    x64=self.settings.oracle_x64, q_logged=self._q_logged[critic_key],
                                    chunk=self.settings.row_chunk)["D"]

    def tables(self, arms, D):
        """Every arm's per-row multipliers given the latest exact D of each oracle arm ({arm name: D})."""
        n = self.prepared.n
        doses = {a.name: harm_doses(D[a.name], a.kind) for a in arms if a.kind in ORACLE_KINDS}
        out = []
        for a in arms:
            if a.kind == "even":
                out.append(np.full(n, float(a.value)))
            elif a.kind in ORACLE_KINDS:
                out.append(doses[a.name])
            elif a.kind == "amount":
                out.append(np.full(n, float(doses[a.source].mean())))
            elif a.kind == "shuffled":
                out.append(doses[a.source][self.perms[a.index]])
            elif a.kind == "reversed":
                out.append(reverse_ranks(doses[a.source], D[a.source], self.tie))
            else:
                raise ValueError(f"unknown arm kind {a.kind!r}")
        return np.stack(out).astype(np.float32)

    def run(self, arms, critics, *, weighted=True, log=None):
        """Train every arm. critics: a list of frozen critic params (arm.critic indexes it). Returns (final actor
        params stacked over arms, dict(history of dose summaries per refresh, per-chunk mean metrics, oracle
        calls))."""
        names = [a.name for a in arms]
        if len(set(names)) != len(names):
            raise ValueError("arm names must be unique")
        for a in arms:
            if a.kind in DERIVED_KINDS and (a.source not in names or
                                            next(b for b in arms if b.name == a.source).kind not in ORACLE_KINDS):
                raise ValueError(f"arm {a.name}: its source must be a harm-placed or cap-2 arm of the same run")
        if not weighted and any(a.kind != "even" or a.value != 1.0 for a in arms):
            raise ValueError("the unweighted path runs plain arms only")
        n_arms = len(arms)
        shared = len({a.critic for a in arms}) == 1
        critic_arg = critics[arms[0].critic] if shared else jax.tree_util.tree_map(
            lambda *xs: jnp.stack(xs), *[critics[a.critic] for a in arms])
        states = jax.tree_util.tree_map(lambda x: jnp.broadcast_to(x, (n_arms,) + jnp.shape(x)), self.start)
        oracle_arms = [i for i, a in enumerate(arms) if a.kind in ORACLE_KINDS]
        refresh = self.settings.oracle_refresh
        if oracle_arms and refresh % self.chunk:
            raise ValueError("the oracle refresh interval must be a multiple of the chunk")
        program = self.program(n_arms, shared, weighted)
        history, metrics, D = [], [], {}
        tables = None
        for c in range(self.steps // self.chunk):
            step0 = c * self.chunk
            if tables is None or (oracle_arms and step0 % refresh == 0):
                for i in oracle_arms:
                    p_i = jax.tree_util.tree_map(lambda x: x[i], states.params)
                    D[arms[i].name] = self.oracle(p_i, critics[arms[i].critic])
                tables_np = self.tables(arms, D)
                tables = jnp.asarray(tables_np)
                history.append(dict(step=step0, **{a.name: dict(mean=float(t.mean()), sd=float(t.std()),
                                                                min=float(t.min()), max=float(t.max()))
                                                   for a, t in zip(arms, tables_np)}))
            states, aux = program(critic_arg, states, self.prepared.train, self.rows[c], tables)
            metrics.append(np.asarray(aux, np.float64).mean(1).tolist())
            if log:
                log(f"actor-only rep {self.rep}: {n_arms} arms, step {step0 + self.chunk}/{self.steps}")
        info = dict(history=history, chunk_metrics=metrics, metric_names=["actor_loss", "q_mean", "lambda", "bc"],
                    oracle_calls=self.oracle_calls, rows_sha256=self.rows_sha256,
                    start_sha256=tree_sha256(self.start_params),
                    failed=[not _finite_tree(jax.tree_util.tree_map(lambda x: x[i], states.params))
                            for i in range(n_arms)])
        return states.params, info

    def arm_params(self, stacked, i):
        return jax.tree_util.tree_map(lambda x: x[i], stacked)


def room_arms(n_shuffled=2):
    """The room check's arms before matching: none, the even ladder, harm-placed, amount-matched, the shuffled
    versions, reversed and cap-2."""
    arms = [Arm("none", "even", 1.0)] + [Arm(f"even_x{m:g}", "even", m) for m in ROOM_EVEN_LADDER]
    arms += [Arm("harm_placed", "harm"), Arm("amount_matched", "amount", source="harm_placed")]
    arms += [Arm(f"shuffled_{k}", "shuffled", source="harm_placed", index=k) for k in range(n_shuffled)]
    arms += [Arm("reversed", "reversed", source="harm_placed"), Arm("cap2", "cap2")]
    return arms


def match_even(runner, critics, none_params, eval_obs, ladder, target_S, settings, log=None):
    """Blind strength matching of even caution to a target strength: glq_match.match_parallel on the knob m - 1 (knob
    0 is 'none', S = 0), evaluating proposals as vmapped even arms. Uses strengths only (no return is computed).
    ladder: [(multiplier, S)]. Returns (public match record, {multiplier: actor params} of evaluated proposals)."""
    knobs = [0.0] + [float(m) - 1.0 for m, _ in ladder]
    S = [0.0] + [float(s) for _, s in ladder]
    evaluated = {}  # {1 + knob: actor params} of every evaluated proposal

    def evaluate(proposals):
        arms = [Arm(f"match_{i}", "even", 1.0 + k) for i, (_, k) in enumerate(proposals)]
        stacked, _ = runner.run(arms, critics)
        out = []
        for i, (_, k) in enumerate(proposals):
            p = runner.arm_params(stacked, i)
            evaluated[1.0 + float(k)] = p
            out.append(strength(p, none_params, eval_obs))
        return out

    states, rounds = M.match_parallel({"even": (knobs, S)}, {"target": float(target_S)}, evaluate,
                                      settings.match_tol, settings.match_rounds, log=log)
    rec = M.public(states[("even", "target")])
    rung = next((float(m) for m, _ in ladder if rec["knob"] is not None and float(m) - 1.0 == rec["knob"]), None)
    rec["ladder_multiplier"] = rung
    rec["multiplier"] = None if rec["knob"] is None else (rung if rung is not None else 1.0 + rec["knob"])
    rec["rounds"] = rounds
    rec["glq_match_sha256"] = M.source_sha256()
    return rec, evaluated


# ------------------------------------------------------------------------------------------------ evaluation

def evaluate_policy(policy, physics, anchors, situations=TC.SITUATIONS, n=TC.BANK_SIZE):
    """Exact 100-step returns of a deterministic policy on each situation's fixed bank, with the same policy capped at
    80% at evaluation: J, its bank SE, the discounted return, points, the unrecorded-throttle share (steps past 80%
    on any thruster) and the loss from unrecorded throttle (capped minus uncapped) in points."""
    out = {}
    for s in situations:
        x0 = TC.bank_starts(s, n)
        run = TR.rollout(policy, x0, physics, record=True)
        capped = TR.rollout(RC.CappedController(policy), x0, physics)
        an = anchors[s]
        pts, cpts = TR.points(run.J, an), TR.points(capped.J, an)
        out[s] = dict(J=float(run.J.mean()), J_se=float(run.J.std() / math.sqrt(len(run.J))),
                      J_discounted=float(run.Jd.mean()), points=float(pts.mean()),
                      points_se=float(pts.std() / math.sqrt(len(pts))), capped_points=float(cpts.mean()),
                      loss_from_unrecorded_throttle=float((cpts - pts).mean()),
                      unrecorded_throttle_share=float(TC.past_knee(run.commands, physics).mean()),
                      finite=bool(np.all(np.isfinite(run.J))))
    return out


def evaluate_actor(actor_params, prepared, anchors, physics=None, situations=TC.SITUATIONS, n=TC.BANK_SIZE):
    return evaluate_policy(mlp_policy(actor_params, prepared), prepared.physics if physics is None else physics,
                           anchors, situations, n)


def loss_rate_ratio(attr, physics):
    """From truth.attribution(new, ref, ...): the loss rate (minus the mean term) at steps where the new policy
    commands past 80% and at the other steps, and their ratio."""
    loss = -np.asarray(attr["terms"])
    past = TC.past_knee(attr["commands"], physics)
    lp = float(loss[past].mean()) if past.any() else None
    lo = float(loss[~past].mean()) if (~past).any() else None
    return dict(loss_rate_past_80=lp, loss_rate_other=lo, steps_past_80=int(past.sum()), steps_other=int((~past).sum()),
                sum_loss_past_80=float(loss[past].sum()), sum_loss_other=float(loss[~past].sum()),
                max_sum_error=float(attr["max_sum_error"]))


# ------------------------------------------------------------------------------------------------ BCA (switched off)

class ApprovalError(PermissionError):
    """Raised by every BCA path without a valid approval file."""


@dataclass(frozen=True)
class Approval:
    """A verified approval: the approval file, the gate report it approves and that report's SHA-256."""

    path: str
    report_path: str
    report_sha256: str
    prereg_sha256: str
    file_sha256: str


def _sha_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


OFFICIAL_RUNS = Path(__file__).resolve().parents[2] / "runs" / "shift"   # the one official pre-registration's root


def _bound_to_official_prereg(path, body):
    """The approval must belong to the one official pre-registration: it sits at
    runs/shift/<first 16 hex of the pre-registration's SHA-256>/approval.json, runs/shift/preregistration.json exists
    with exactly that SHA-256 (and its sidecar agrees), is a real-preset ThrusterChain pre-registration, and the
    report it approves is that directory's report/report.json, written by the gate's report stage. Raises
    ApprovalError otherwise (a pair of hand-written JSON files anywhere else is refused)."""
    approval = Path(path).resolve()
    prereg_sha = str(body.get("prereg_sha256") or "")
    root, runs = approval.parent, approval.parent.parent
    if len(prereg_sha) != 64 or approval.name != "approval.json" or root.name != prereg_sha[:16]:
        raise ApprovalError(f"{BCA_REFUSAL} (the approval file is not <runs>/<first 16 hex of the pre-registration's "
                            "SHA-256>/approval.json)")
    if runs != OFFICIAL_RUNS.resolve():
        raise ApprovalError(f"{BCA_REFUSAL} (the approval does not belong to the official pre-registration in "
                            f"{OFFICIAL_RUNS})")
    prereg = runs / "preregistration.json"
    side = runs / "preregistration.json.sha256"
    if not prereg.exists() or not side.exists():
        raise ApprovalError(f"{BCA_REFUSAL} (no official pre-registration at {prereg})")
    text = prereg.read_bytes()
    if hashlib.sha256(text).hexdigest() != prereg_sha or side.read_text(encoding="utf8").split()[0] != prereg_sha:
        raise ApprovalError(f"{BCA_REFUSAL} (the pre-registration's SHA-256 differs from the one the approval holds)")
    try:
        payload = json.loads(text)
    except ValueError as error:
        raise ApprovalError(f"{BCA_REFUSAL} (the pre-registration is not JSON)") from error
    if payload.get("kind") != "bca-shift-preregistration" or (payload.get("config") or {}).get("preset") != "real":
        raise ApprovalError(f"{BCA_REFUSAL} (not the real-preset ThrusterChain pre-registration)")
    if Path(body.get("report_path", "")).resolve() != root / "report" / "report.json":
        raise ApprovalError(f"{BCA_REFUSAL} (the approval names a report outside its pre-registration's directory)")
    _official_prereg_verifies(prereg_sha)


_PREREG_VERIFIED = set()


def _official_prereg_verifies(digest):
    """acceptance.load_prereg must accept the official pre-registration (sidecar, every hashed file, the dependency
    directory, the software, the body against what the hashed code writes) and its SHA-256 must be `digest`. Checked
    once per process and digest (every gate stage re-checks the hashes itself through acceptance.guard)."""
    if digest in _PREREG_VERIFIED:
        return
    from experiments.shift import acceptance as A

    try:
        prereg = A.load_prereg(OFFICIAL_RUNS)
    except A.GateError as error:
        raise ApprovalError(f"{BCA_REFUSAL} (the official pre-registration does not verify: {error})") from error
    if prereg.digest != digest:
        raise ApprovalError(f"{BCA_REFUSAL} (the approval names another pre-registration than the official one)")
    _PREREG_VERIFIED.add(digest)


def verify_approval(path):
    """Read and verify an approval file (written by `acceptance.py approve`): its sidecar matches; it belongs to the
    one official pre-registration (_bound_to_official_prereg: runs/shift, real preset, the approval and the report
    inside that pre-registration's directory); the gate report it names exists, has the SHA-256 the approval holds,
    was written by the report stage of the same pre-registration and says that the gate allows the BCA stage.
    Raises ApprovalError otherwise."""
    path = Path(path)
    if not path.exists():
        raise ApprovalError(f"{BCA_REFUSAL} (no approval file at {path})")
    text = path.read_text(encoding="utf8")
    side = path.with_name(path.name + ".sha256")
    digest = hashlib.sha256(text.encode("utf8")).hexdigest()
    if not side.exists() or side.read_text(encoding="utf8").split()[0] != digest:
        raise ApprovalError(f"{BCA_REFUSAL} (the approval file differs from its sidecar)")
    try:
        body = json.loads(text)
    except ValueError as error:
        raise ApprovalError(f"{BCA_REFUSAL} (the approval file is not JSON)") from error
    if body.get("kind") != "bca-shift-approval":
        raise ApprovalError(f"{BCA_REFUSAL} (not an approval file)")
    _bound_to_official_prereg(path, body)
    report = Path(body.get("report_path", ""))
    if not report.exists() or _sha_file(report) != body.get("report_sha256"):
        raise ApprovalError(f"{BCA_REFUSAL} (the gate report is missing or differs from the approved SHA-256)")
    try:
        rep = json.loads(report.read_text(encoding="utf8"))
    except ValueError as error:
        raise ApprovalError(f"{BCA_REFUSAL} (the gate report is not JSON)") from error
    if (rep.get("kind") != "shift-report" or rep.get("prereg_sha256") != body.get("prereg_sha256")
            or not rep.get("bca_allowed", False)):
        raise ApprovalError(f"{BCA_REFUSAL} (the report belongs to another pre-registration or does not allow BCA)")
    return Approval(str(path), str(report), body["report_sha256"], body["prereg_sha256"], digest)


def require_approval(approval):
    """Refuse unless approval is a verified Approval whose files are unchanged since verification."""
    if not isinstance(approval, Approval):
        raise ApprovalError(BCA_REFUSAL)
    again = verify_approval(approval.path)
    if again != approval:
        raise ApprovalError(f"{BCA_REFUSAL} (the approval or its report changed since verification)")
    return approval


def bca_config():
    """td3_bc_bca.Config with configs/td3_bc.yaml's BCA settings (blend 0.5: doses in [1, 1.5))."""
    s = BCA_SETTINGS
    return P.Config(arm="bca", posterior=WBCPConfig(s["miscoverage"], s["credibility"], s["draws"]), blend=s["blend"],
                    cal_lr=s["cal_lr"], cal_beta=s["cal_beta"], width_penalty=s["width_penalty"],
                    scale_ema=s["scale_ema"])


def bca_initial_state(args, config, native=None):
    """td3_bc_bca.initialize's state without its D4RL config checks: td3_bc.initialize's native state (or a given
    one), the Calibrator (state-dependent, inputs already normalised) initialised from fold_in(rng, 1128352841), its
    Adam (cal_lr), residual scale 1 and the initial (not ready) reference. Returns (rng, State, models)."""
    rng, base_native, _ = BASE.initialize(args, DX, DA, 1.0)
    native = base_native if native is None else native
    cal = Calibrator(jnp.zeros(DX), jnp.ones(DX), state_dep=True)
    cal_state = TrainState.create(apply_fn=cal.apply,
                                  params=cal.init(jax.random.fold_in(rng, KEY_CALIBRATOR), jnp.zeros(DX)[None],
                                                  jnp.zeros(DA)[None]),
                                  tx=optax.adam(config.cal_lr))
    state = P.State(native, cal_state, jnp.asarray(1.0), initial_reference(cal_state.params))
    return rng, state, (ACTOR, CRITIC, cal)


def place(multiplier, placement, rng, signal=None):
    """Where the doses go. 'own': unchanged. 'flattened': every row the batch mean. 'shuffled': permuted by
    fold_in(rng, 40) (the update key is not consumed). 'noise_map': the same doses re-assigned so that the largest
    dose goes to the row with the largest noise map (signal)."""
    if placement == "own":
        return multiplier
    if placement == "flattened":
        return jnp.full_like(multiplier, jnp.mean(multiplier))
    if placement == "shuffled":
        return jax.random.permutation(jax.random.fold_in(rng, TAG_PLACEMENT), multiplier)
    if placement == "noise_map":
        if signal is None:
            raise ValueError("noise-map placement needs the noise map of the batch rows")
        order = jnp.argsort(signal)
        return jnp.zeros_like(multiplier).at[order].set(jnp.sort(multiplier))
    raise ValueError(f"placement must be one of {PLACEMENTS}")


def bca_update(args, config, models, state, batch, it, rng, max_action=1.0, placement="own", signal=None):
    """algorithms/td3_bc_bca.update line for line, with the dose passed through place() before td3_bc_update."""
    fitted, diag = P.fit_scale(args, config, models, state, batch, rng, max_action)
    dose = P.bc_readout(models, state, batch, config.blend)
    multiplier = P.consumed_bc_multiplier(config, dose.dose, batch, it, rng)
    multiplier = place(multiplier, placement, rng, signal)
    native, metrics = BASE.td3_bc_update(args, models[0].apply, models[1].apply, state.native, batch, it, rng,
                                         max_action, bc_multiplier=multiplier)
    valid = P.tree_finite((batch, state.native, native, metrics))
    valid = valid & diag["scale_inputs_valid"] & dose.inputs_valid
    metrics.update(bc_multiplier_mean=jnp.mean(multiplier),
                   bc_support_fraction=jnp.mean(dose.support_mask.astype(jnp.float32)),
                   posterior_ready=state.posterior.ready)
    result = jax.lax.cond(valid, lambda _: fitted._replace(native=native), lambda _: state, None)
    metrics.update(diag, inputs_valid=valid)
    return (result, metrics)


def bca_train_step(args, config, models, dataset, max_action=1.0, placement="own", signal=None):
    """algorithms/td3_bc_bca.make_train_step line for line, calling bca_update; signal (the noise map per training
    row) is read at the batch rows for 'noise_map' placement only."""
    if not len(dataset.obs):
        raise ValueError("training pool must be nonempty")

    def step(carry, _):
        rng, state, it = carry
        kr, kb, kn = jax.random.split(rng, 3)
        idx = jax.random.randint(kb, (args.batch_size,), 0, len(dataset.obs))
        batch = jax.tree_util.tree_map(lambda x: x[idx], dataset)
        sig = None if placement != "noise_map" else signal[idx]
        result, metrics = bca_update(args, config, models, state, batch, it + 1, kn, max_action, placement, sig)
        proposed = (kr, result, it + 1)
        accepted = jax.lax.cond(metrics["inputs_valid"], lambda _: proposed, lambda _: carry, None)
        return (accepted, metrics)

    return step


def refresh_steps(settings, total):
    """BCA's threshold refreshes: after the warm-up (10,000), every 5,000 updates, before the last update."""
    return list(range(settings.bca_warmup, total, settings.bca_refresh))


def trace_bca(prepared, settings, rep, placement="own", alpha=None, signal=None, length=2):
    """Build and lower (trace) the BCA program of one placement without running it: no calibrator fit, threshold
    refresh or dose readout is executed. Returns the lowered program."""
    args = make_args(settings, rep, alpha)
    config = bca_config()
    rng, state, models = bca_initial_state(args, config)
    sig = None if signal is None else jnp.asarray(np.asarray(signal, np.float32))
    step = bca_train_step(args, config, models, prepared.train, 1.0, placement, sig)
    program = jax.jit(lambda carry: jax.lax.scan(step, carry, None, length=length))
    return program.lower((rng, state, jnp.int32(0)))


def bca_cotrain(prepared, settings, rep, approval, *, placement="own", noise_map=None, alpha=None, updates=None,
                out_dir=None, log=None):
    """The co-trained BCA arm of one placement (module docstring). Refuses without approval. Returns (record, final
    State, {update: State} at the snapshots)."""
    require_approval(approval)
    if placement not in PLACEMENTS:
        raise ValueError(f"placement must be one of {PLACEMENTS}")
    started = time.perf_counter()
    args = make_args(settings, rep, alpha)
    config = bca_config()
    rng, state, models = bca_initial_state(args, config)
    total = int(settings.updates if updates is None else updates)
    signal = None if noise_map is None else jnp.asarray(np.asarray(noise_map, np.float32))
    if placement == "noise_map" and (signal is None or signal.shape != (prepared.n,)):
        raise ValueError("noise-map placement needs the noise map of every kept training row")
    step = bca_train_step(args, config, models, prepared.train, 1.0, placement, signal)
    refreshes = refresh_steps(settings, total)
    snaps = snapshot_steps(settings, total)
    refresh_seed = int(rng_np(settings, rep, TAG_REFRESH).integers(2 ** 31 - 1))
    carry, done, chunks, states, refresh_log = (rng, state, jnp.int32(0)), 0, [], {}, []
    programs = {}
    record = dict(kind="shift-bca", placement=placement, rep=int(rep), args=asdict(args), updates=total,
                  bca=BCA_SETTINGS, data=prepared.describe(), approval=asdict(require_approval(approval)),
                  initial_sha256=tree_sha256((state.native.actor.params, state.native.critic.params)),
                  batch_stream_sha256=batch_stream_sha256(rng, prepared.n, args.batch_size),
                  refresh_seed=refresh_seed, snapshots=[], failed=False)
    for target in _events(total, settings.chunk, snaps, refreshes):
        length = target - done
        if length not in programs:
            programs[length] = jax.jit(lambda c, length=length: jax.lax.scan(step, c, None, length=length))
        carry, metrics = programs[length](carry)
        done = target
        try:
            P.require_valid(metrics)
        except FloatingPointError as error:
            record.update(failed=True, failed_at=target, failure=str(error))
            break
        chunks.append(dict(_chunk_summary(metrics), update=target))
        if target in refreshes:
            key = jax.random.fold_in(jax.random.PRNGKey(refresh_seed), target)
            new, diag, _ = P.refresh(args, config, models, carry[1], prepared.bank, key, 1.0,
                                     heldout_ids=np.arange(len(prepared.bank.obs)))
            carry = (carry[0], new, carry[2])
            refresh_log.append(dict(step=target, **{k: bool(v) for k, v in diag.items()}))
        if target in snaps:
            states[target] = carry[1]
            snap = dict(update=target, actor_sha256=tree_sha256(carry[1].native.actor.params))
            if out_dir is not None:
                snap["file_sha256"] = save_tree(
                    Path(out_dir) / f"snapshot_{int(round(100 * target / total)):03d}.msgpack", carry[1])
            record["snapshots"].append(snap)
        if log:
            log(f"BCA ({placement}) rep {rep}: update {target}/{total}")
    record.update(chunks=chunks, refreshes=refresh_log, final_update=done, seconds=time.perf_counter() - started)
    return record, carry[1], states


def bca_doses(state, prepared, approval, rows=None, chunk=8192):
    """BCA's doses (blend 0.5) read out at training rows from a BCA state's frozen reference (td3_bc_bca.bc_readout).
    Refuses without approval."""
    require_approval(approval)
    config = bca_config()
    cal = Calibrator(jnp.zeros(DX), jnp.ones(DX), state_dep=True)
    models = (ACTOR, CRITIC, cal)
    idx = np.arange(prepared.n) if rows is None else np.asarray(rows, np.int64)
    out = []
    readout = jax.jit(lambda st, b: P.bc_readout(models, st, b, config.blend).dose)
    for i in range(0, len(idx), chunk):
        r = idx[i:i + chunk]
        batch = jax.tree_util.tree_map(lambda x: x[r], prepared.train)
        out.append(np.asarray(readout(state, batch), np.float64))
    return np.concatenate(out) if out else np.zeros(0)


def bca_band(host_state, prepared, settings, rep, approval, *, steps=None, log=None):
    """BCA's band on a frozen plain TD3+BC critic: td3_bc_bca.fit_scale for `steps` (20,000) batches of the kept rows
    (default_rng([seed, rep, 41])), one refresh on the calibration bank, then the doses at every kept row. The host's
    networks never change. Refuses without approval. Returns (doses, record)."""
    require_approval(approval)
    steps = int(settings.band_fit_steps if steps is None else steps)
    args = make_args(settings, rep)
    config = bca_config()
    rng, state, models = bca_initial_state(args, config, native=host_state)
    rows = rng_np(settings, rep, TAG_BAND).integers(prepared.n, size=(steps, settings.batch_size))
    keys = jax.random.split(jax.random.fold_in(rng, TAG_BAND), steps)

    def body(st, inp):
        r, k = inp
        batch = jax.tree_util.tree_map(lambda x: x[r], prepared.train)
        st, diag = P.fit_scale(args, config, models, st, batch, k, 1.0)
        return st, diag["scale_fit_accepted"]

    state, accepted = jax.jit(lambda st, r, k: jax.lax.scan(body, st, (r, k)))(state, jnp.asarray(rows), keys)
    key = jax.random.fold_in(jax.random.PRNGKey(int(rng_np(settings, rep, TAG_BAND, 1).integers(2 ** 31 - 1))), steps)
    state, diag, _ = P.refresh(args, config, models, state, prepared.bank, key, 1.0,
                               heldout_ids=np.arange(len(prepared.bank.obs)))
    doses = bca_doses(state, prepared, approval)
    if tree_sha256(state.native) != tree_sha256(host_state):
        raise AssertionError("the band must not change the frozen host")
    return doses, dict(kind="shift-bca-band", rep=int(rep), steps=steps, accepted_share=float(np.mean(accepted)),
                       refresh={k: bool(v) for k, v in diag.items()}, host_sha256=tree_sha256(host_state))


def identity_placement_check(prepared, settings, rep, approval, updates=50):
    """After approval only: the 'own' placement of bca_train_step against td3_bc_bca.make_train_step, run for
    `updates` updates from the same state; True when every leaf is bit for bit equal."""
    require_approval(approval)
    args = make_args(settings, rep)
    config = bca_config()
    rng, state, models = bca_initial_state(args, config)
    ours = bca_train_step(args, config, models, prepared.train, 1.0, "own")
    theirs = P.make_train_step(args, config, models, prepared.train, 1.0)
    a = jax.jit(lambda c: jax.lax.scan(ours, c, None, length=updates))((rng, state, jnp.int32(0)))[0]
    b = jax.jit(lambda c: jax.lax.scan(theirs, c, None, length=updates))((rng, state, jnp.int32(0)))[0]
    return all(np.array_equal(np.asarray(x), np.asarray(y)) for x, y in
               zip(jax.tree_util.tree_leaves(a), jax.tree_util.tree_leaves(b)))

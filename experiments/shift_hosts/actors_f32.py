"""Float32 restatements of the four Unifloral actors, the actor parameter file, and the learner's own float32 commands
on saved states (plan sections 3.9 and 4.5, stop rule S7, test (h)).

An experiment-local attachment (plan section 3.1): it changes none of the repository's declared methods and edits
nothing under baselines/unifloral/. It imports in both Python environments (JAX, flax, numpy and handoff.py only).
The Unifloral files are imported only by unifloral_commands and open_loop_commands, lazily and only after
unifloral_import.prepare(), so only in the training environment.

Why it exists. The scoring environment (/opt/bca-shift-venv) has the simulator but cannot import the Unifloral files
(no distrax, d4rl, gym or wandb). The training environment (/opt/bca-venv) has the files but not the simulator
(experiments/shift refuses numpy 1.26.4). So S7 has two halves (plan sections 3.9 and 6.1):
  open loop, by file   the training environment computes each Unifloral actor's float32 commands on saved states
                       (open_loop_commands, here, with the learner's own network classes); this restatement, in the
                       scoring environment, must match them within 1e-7 (evaluate.open_loop_compare). Both run on the
                       processor; CQL uses the same noise on both sides.
  closed loop          in the scoring environment, this restatement (Float32Policy) and the float64 adapter
                       (policies.py) must agree within 0.05 points on every start (evaluate.closed_loop_check).

The four actors, restated in plain flax with Unifloral's layer order, so the parameter names are Unifloral's own:
  TD3+BC  td3_bc.py:98-114     z = (o - obs_mean) / (obs_std + 1e-3); 2 x [Dense(256), relu]; tanh(Dense(4)). The
                               command is the tanh output (distrax.Transformed(Deterministic, Tanh) samples its loc
                               through tanh).
  IQL     iql.py:110-132       the same body, nn.tanh on the output; the scoring action is that mean (eval=True,
                               iql.py:123-126, 157). The state-independent logstd is not used.
  ReBRAC  rebrac.py:118-139    raw o (norm_obs False, rebrac.py:52); 3 x [Dense(256), relu] with no layer norm
                               (actor_ln False, rebrac.py:50); tanh(Dense(4)).
  CQL     cql.py:101-120       raw o; 3 x [Dense(256), relu]; Dense_3 is the log-spread head (created first),
                               Dense_4 the mean head; std = exp(clip(log_std, -5, 2)). The scored command is
                               tanh(std * eps + mean): distrax's Normal._sample_n order (scale * rnd + loc), then
                               the Tanh bijector, with eps the declared noise (policies.cql_noise, choice 7) in place
                               of distrax's own draw. The readout is tanh(mean) (shown, not judged).
Every command goes through nan_to_num, as Unifloral's evaluation does (_policy_step, td3_bc.py:135-140,
cql.py:153-158, iql.py:153-158, rebrac.py:162-167). Dense's initializers (bias_init=constant(0.1) and the like) are not
restated: the parameters are always given.

The actor parameter file (the runner writes it with write_actor_file in the training environment; read_actor_file
reads it anywhere). One handoff.py file, <name>.npz with its manifest and both sidecars. Every array is float32:
  actor/params/Dense_0/kernel [13, 256]   actor/params/Dense_0/bias [256]
  actor/params/Dense_1/kernel [256, 256]  actor/params/Dense_1/bias [256]
  TD3+BC, IQL   actor/params/Dense_2/kernel [256, 4], bias [4]          (the output layer)
                norm/obs_mean [13], norm/obs_std [13]                     (the network's own fields: the training
                                                                           rows' mean and nan_to_num(std, nan=1),
                                                                           without the 1e-3 the network adds)
  IQL also      actor/params/logstd [4]                                   (carried, unused by the scoring action)
  ReBRAC        actor/params/Dense_2/kernel [256, 256], bias [256]; actor/params/Dense_3/kernel [256, 4], bias [4]
  CQL           actor/params/Dense_2/kernel [256, 256], bias [256];
                actor/params/Dense_3/kernel [256, 4], bias [4]            (log-spread head)
                actor/params/Dense_4/kernel [256, 4], bias [4]            (mean head)
These are handoff.flatten_dict(agent_state.actor.params, "actor") (INTERFACES.md section 12). The manifest's meta holds
schema "shift-hosts-actor-v1", learner, made_up (true only for test parameters), and for ReBRAC actor_ln and norm_obs
(both must be false); the runner adds its own keys (quality, seed, setup, step, run record, actor tree hash).
"""

import os
from collections import namedtuple
from typing import Any

import numpy as np

from experiments.shift_hosts import handoff as H

SCHEMA = "shift-hosts-actor-v1"
LEARNERS = ("td3_bc", "cql", "iql", "rebrac")
OBS_DIM, NUM_ACTIONS, HIDDEN = 13, 4, 256
NORM_EPS = 1e-3                            # the networks' own "obs_std + 1e-3" (td3_bc.py:105, iql.py:118)
CQL_LOG_STD = (-5.0, 2.0)                  # TanhGaussianActor's log_std_min, log_std_max (cql.py:103-104)
HIDDEN_LAYERS = dict(td3_bc=2, iql=2, rebrac=3, cql=3)
NORMALIZED = ("td3_bc", "iql")             # normalize inside the network; CQL and ReBRAC take raw observations
READOUTS = dict(td3_bc=("scored",), iql=("scored",), rebrac=("scored",), cql=("scored", "tanh_mean"))
# The processor flags and matrix precision of a training-environment process after unifloral_import.prepare("cpu")
# (unifloral_import.chosen_flags("cpu") and MATMUL_PRECISION); configure() sets the same in the scoring environment so
# both halves of S7's open loop compile the same programs.
CPU_XLA_FLAGS = "--xla_gpu_deterministic_ops=true --xla_cpu_multi_thread_eigen=false"
MATMUL_PRECISION = "highest"

ActorFile = namedtuple("ActorFile", "learner params norm meta path npz_sha256 manifest")
ActorFile.__doc__ = """An actor parameter file as read: params is the flax tree {"params": {...}} of float32 numpy
arrays, norm is {"obs_mean", "obs_std"} (float32 [13]; TD3+BC and IQL) or None, meta the manifest's meta."""


class ActorFileError(ValueError):
    """An actor parameter file that does not hold exactly its learner's layout."""


# ------------------------------------------------------------------------------------------------ the file layout

def _dense(name, n_in, n_out):
    return {f"actor/params/{name}/kernel": (n_in, n_out), f"actor/params/{name}/bias": (n_out,)}


def layout(learner):
    """{array name: shape} of a learner's actor file (module docstring); every array is float32."""
    if learner not in LEARNERS:
        raise ValueError(f"learner must be one of {LEARNERS}, not {learner!r}")
    hidden = HIDDEN_LAYERS[learner]
    out, n_in = {}, OBS_DIM
    for i in range(hidden):
        out.update(_dense(f"Dense_{i}", n_in, HIDDEN))
        n_in = HIDDEN
    out.update(_dense(f"Dense_{hidden}", HIDDEN, NUM_ACTIONS))
    if learner == "cql":
        out.update(_dense(f"Dense_{hidden + 1}", HIDDEN, NUM_ACTIONS))
    if learner == "iql":
        out["actor/params/logstd"] = (NUM_ACTIONS,)
    if learner in NORMALIZED:
        out["norm/obs_mean"] = (OBS_DIM,)
        out["norm/obs_std"] = (OBS_DIM,)
    return out


def _check_layout(learner, arrays, where):
    want = layout(learner)
    if sorted(arrays) != sorted(want):
        missing = sorted(set(want) - set(arrays))
        extra = sorted(set(arrays) - set(want))
        raise ActorFileError(f"{where}: not a {learner} actor (missing {missing}, unexpected {extra})")
    for name, shape in want.items():
        a = np.asarray(arrays[name])
        if a.dtype != np.float32 or tuple(a.shape) != tuple(shape):
            raise ActorFileError(f"{where}: {name} is {a.dtype} {list(a.shape)}, expected float32 {list(shape)}")


def _check_meta(learner, meta, where):
    if not isinstance(meta, dict):
        raise ActorFileError(f"{where}: meta must be a dict")
    if not isinstance(meta.get("made_up"), bool):
        raise ActorFileError(f"{where}: meta must say made_up true or false")
    if learner == "rebrac":
        for flag in ("actor_ln", "norm_obs"):
            if meta.get(flag) is not False:
                raise ActorFileError(f"{where}: ReBRAC's {flag} must be recorded and false (rebrac.py:50-52); the "
                                     "adapters restate only Unifloral's default actor")


def write_actor_file(path, learner, actor_params, *, obs_mean=None, obs_std=None, meta):
    """Write a learner's actor parameter file (module docstring) with handoff.write. actor_params: the flax tree
    agent_state.actor.params (JAX or numpy leaves, float32). obs_mean, obs_std: Built.obs_mean and Built.obs_std
    (TD3+BC and IQL only). meta: a dict with at least made_up (bool), and for ReBRAC actor_ln and norm_obs (both
    false). Returns the manifest."""
    if learner not in LEARNERS:
        raise ValueError(f"learner must be one of {LEARNERS}, not {learner!r}")
    arrays = {k: np.asarray(v) for k, v in H.flatten_dict(actor_params, "actor").items()}
    if learner in NORMALIZED:
        if obs_mean is None or obs_std is None:
            raise ActorFileError(f"{learner}'s actor file carries the network's obs_mean and obs_std")
        arrays["norm/obs_mean"] = np.asarray(obs_mean)
        arrays["norm/obs_std"] = np.asarray(obs_std)
    elif obs_mean is not None or obs_std is not None:
        raise ActorFileError(f"{learner}'s actor takes raw observations; give no normalization constants")
    _check_layout(learner, arrays, str(path))
    meta = dict(meta)
    _check_meta(learner, meta, str(path))
    full = dict(meta, schema=SCHEMA, learner=learner, obs_dim=OBS_DIM, num_actions=NUM_ACTIONS,
                hidden_layers=HIDDEN_LAYERS[learner],
                finite=bool(all(np.all(np.isfinite(a)) for a in arrays.values())))
    if learner == "cql":
        full.update(log_std_min=CQL_LOG_STD[0], log_std_max=CQL_LOG_STD[1])
    return H.write(path, arrays, meta=full)


def read_actor_file(path, learner=None):
    """An ActorFile after handoff.read's checks (sidecars, manifest, every array's hash) and the layout check."""
    arrays, manifest = H.read(path)
    meta = manifest.get("meta") or {}
    if meta.get("schema") != SCHEMA:
        raise ActorFileError(f"{path}: not a {SCHEMA} file")
    found = meta.get("learner")
    if found not in LEARNERS or (learner is not None and found != learner):
        raise ActorFileError(f"{path}: holds learner {found!r}, expected {learner!r}")
    _check_layout(found, arrays, str(path))
    _check_meta(found, meta, str(path))
    params = H.unflatten_dict({k: v for k, v in arrays.items() if k.startswith("actor/")}, "actor")
    norm = None
    if found in NORMALIZED:
        norm = dict(obs_mean=arrays["norm/obs_mean"], obs_std=arrays["norm/obs_std"])
    return ActorFile(found, params, norm, meta, str(path), manifest["npz_sha256"], manifest)


# ------------------------------------------------------------------------------------------------ JAX set-up

def backend_started():
    """Whether JAX's backend has started in this process (the same test as unifloral_import.backend_started)."""
    import sys

    if "jax" not in sys.modules:
        return False
    from jax._src import xla_bridge

    return bool(xla_bridge.backends_are_initialized())


def configure():
    """Processor settings of the scoring environment's float32 restatement, equal to a training-environment process's
    after unifloral_import.prepare("cpu"): XLA_FLAGS = CPU_XLA_FLAGS, jax_platforms "cpu", matrix precision "highest".
    Call it before JAX's backend starts. Once started, it only checks the flags (RuntimeError if they differ).
    Returns the settings in effect."""
    import jax

    if not backend_started():
        os.environ["XLA_FLAGS"] = CPU_XLA_FLAGS
        jax.config.update("jax_platforms", "cpu")
    elif os.environ.get("XLA_FLAGS") != CPU_XLA_FLAGS:
        raise RuntimeError(f"JAX's backend started with XLA_FLAGS={os.environ.get('XLA_FLAGS')!r}, not "
                           f"{CPU_XLA_FLAGS!r}; call actors_f32.configure() first in the process")
    jax.config.update("jax_default_matmul_precision", MATMUL_PRECISION)
    return settings()


def settings():
    """The JAX settings in effect (for records)."""
    import jax

    return dict(xla_flags=os.environ.get("XLA_FLAGS"), jax_platforms=jax.config.jax_platforms,
                matmul_precision=jax.config.jax_default_matmul_precision, jax=jax.__version__,
                backend=jax.default_backend() if backend_started() else None)


# ------------------------------------------------------------------------------------------------ the restatement

def _modules():
    """The flax modules of the restatement (built on first use, so importing this file creates nothing)."""
    if "modules" in _CACHE:
        return _CACHE["modules"]
    import flax.linen as nn
    import jax.numpy as jnp

    class NormalizedTanhActor32(nn.Module):
        """TD3+BC's DeterministicTanhActor (td3_bc.py:98-114) and IQL's TanhGaussianActor mean (iql.py:110-126), whose
        bodies are the same: normalize, two hidden layers, tanh of the output layer."""

        obs_mean: Any
        obs_std: Any

        @nn.compact
        def __call__(self, x):
            x = (x - self.obs_mean) / (self.obs_std + 1e-3)
            for _ in range(2):
                x = nn.Dense(256)(x)
                x = nn.relu(x)
            x = nn.Dense(NUM_ACTIONS)(x)
            return jnp.tanh(x)

    class RawTanhActor32(nn.Module):
        """ReBRAC's DeterministicTanhActor with actor_ln and norm_obs off (rebrac.py:118-139): three hidden layers on
        the raw observation, tanh of the output layer."""

        @nn.compact
        def __call__(self, x):
            for _ in range(3):
                x = nn.Dense(256)(x)
                x = nn.relu(x)
            action = nn.Dense(NUM_ACTIONS)(x)
            return jnp.tanh(action)

    class CQLActor32(nn.Module):
        """CQL's TanhGaussianActor (cql.py:101-120) up to its distribution: (mean, std). Dense_3 (created first) is
        the log-spread head, Dense_4 the mean head."""

        log_std_max: float = CQL_LOG_STD[1]
        log_std_min: float = CQL_LOG_STD[0]

        @nn.compact
        def __call__(self, x):
            for _ in range(3):
                x = nn.Dense(256)(x)
                x = nn.relu(x)
            log_std = nn.Dense(NUM_ACTIONS)(x)
            std = jnp.exp(jnp.clip(log_std, self.log_std_min, self.log_std_max))
            mean = nn.Dense(NUM_ACTIONS)(x)
            return mean, std

    _CACHE["modules"] = (NormalizedTanhActor32, RawTanhActor32, CQLActor32)
    return _CACHE["modules"]


_CACHE = {}


def _program(learner, readout):
    """The compiled (jax.jit) float32 command program of (learner, readout): run(params, obs, obs_mean, obs_std,
    eps). The normalization constants are arguments, as the learner's own side passes them (unifloral_commands), so
    both halves of S7's open loop compile the same arithmetic."""
    key = ("program", learner, readout)
    if key in _CACHE:
        return _CACHE[key]
    import jax
    import jax.numpy as jnp

    Normalized, Raw, CQL = _modules()
    if readout not in READOUTS[learner]:
        raise ValueError(f"{learner} has readouts {READOUTS[learner]}, not {readout!r}")
    if learner in NORMALIZED:
        def run(params, obs, obs_mean, obs_std, eps):
            return jnp.nan_to_num(Normalized(obs_mean, obs_std).apply(params, obs))
    elif learner == "rebrac":
        def run(params, obs, obs_mean, obs_std, eps):
            return jnp.nan_to_num(Raw().apply(params, obs))
    else:
        def run(params, obs, obs_mean, obs_std, eps):
            mean, std = CQL().apply(params, obs)
            action = jnp.tanh(mean) if readout == "tanh_mean" else jnp.tanh(std * eps + mean)
            return jnp.nan_to_num(action)
    _CACHE[key] = jax.jit(run)
    return _CACHE[key]


def _restated_params(actor):
    """The parameters the restatement applies: IQL's mean layers only (logstd is not part of its scoring action)."""
    if actor.learner == "iql":
        p = actor.params["params"]
        return {"params": {k: p[k] for k in ("Dense_0", "Dense_1", "Dense_2")}}
    return actor.params


def commands(actor, obs, eps=None, readout="scored"):
    """float32 [n, 4] commands of the restated actor (an ActorFile) at float32 observations obs [n, 13]. CQL's
    "scored" readout needs eps float32 [n, 4] (the declared noise of each row); "tanh_mean" is its readout."""
    import jax.numpy as jnp

    obs = np.asarray(obs)
    if obs.dtype != np.float32 or obs.ndim != 2 or obs.shape[1] != OBS_DIM:
        raise ValueError(f"obs must be float32 [n, {OBS_DIM}], not {obs.dtype} {list(obs.shape)}")
    if actor.learner == "cql" and readout == "scored":
        eps = np.asarray(eps)
        if eps.dtype != np.float32 or eps.shape != (len(obs), NUM_ACTIONS):
            raise ValueError("CQL's scored command needs eps float32 [n, 4], one row per observation")
        eps = jnp.asarray(eps)
    else:
        eps = None
    norm = actor.norm or {}
    mean = None if actor.learner not in NORMALIZED else jnp.asarray(norm["obs_mean"])
    std = None if actor.learner not in NORMALIZED else jnp.asarray(norm["obs_std"])
    out = _program(actor.learner, readout)(_restated_params(actor), jnp.asarray(obs), mean, std, eps)
    return np.asarray(out)


class Float32Policy:
    """The restated float32 actor as a controller for experiments/shift/truth.rollout (stop rule S7, closed loop):
    act(x, t) rounds the float64 state to the float32 observation the environment returns
    (thruster_chain.ThrusterChainEnv._obs), computes the float32 command and returns it as float64, which is exactly
    what env.step receives (the command is already float32). CQL's scored readout uses eps[:, t] (eps float32
    [n, 100, 4], rows aligned with the starts being rolled)."""

    def __init__(self, actor, eps=None, readout="scored"):
        if readout not in READOUTS[actor.learner]:
            raise ValueError(f"{actor.learner} has readouts {READOUTS[actor.learner]}, not {readout!r}")
        self.actor, self.readout = actor, readout
        self.eps = None
        if actor.learner == "cql" and readout == "scored":
            if eps is None:
                raise ValueError("CQL's scored command needs its declared noise eps [n, 100, 4]")
            self.eps = np.asarray(eps)
            if self.eps.dtype != np.float32 or self.eps.ndim != 3 or self.eps.shape[2] != NUM_ACTIONS:
                raise ValueError("eps must be float32 [n, steps, 4]")
        self.name = f"{actor.learner} actor, float32 restatement ({readout})"

    def act(self, x, t):
        obs = np.asarray(x, np.float64).astype(np.float32)
        eps = None
        if self.eps is not None:
            if len(obs) != len(self.eps):
                raise ValueError(f"{len(obs)} states but noise for {len(self.eps)} starts")
            eps = self.eps[:, int(t)]
        return commands(self.actor, obs, eps, self.readout).astype(np.float64)

    def __call__(self, x):
        if self.eps is not None:
            raise TypeError("CQL's sampled command needs the step index; call act(x, t)")
        return self.act(x, 0)


# ------------------------------------------------------------------------------------------ the learner's own network

def unifloral_commands(actor, obs, eps=None, readout="scored"):
    """float32 [n, 4] commands of the learner's OWN network (the imported Unifloral class) at float32 observations,
    on the processor: Unifloral's _policy_step (td3_bc.py:135-140; cql.py:153-158; iql.py:153-158 with eval=True;
    rebrac.py:162-167), compiled with jax.jit and mapped over rows with jax.vmap as there, with nan_to_num. The
    normalization constants (TD3+BC and IQL) are passed as arguments to the compiled program. For CQL the
    distribution's own sample is replaced by the declared noise: pi.bijector.forward(pi.distribution.scale * e +
    pi.distribution.loc), distrax's Normal._sample_n arithmetic (scale * rnd + loc) and Tanh; the readout is
    pi.bijector.forward(loc). Training environment only, after unifloral_import.prepare(device="cpu")."""
    import jax
    import jax.numpy as jnp

    from experiments.shift_hosts import unifloral_import as UI

    learner = actor.learner
    U = UI.load(learner)
    if jax.default_backend() != "cpu":
        raise RuntimeError("S7's open loop runs on the processor (plan section 3.9); prepare with device='cpu'")
    if readout not in READOUTS[learner]:
        raise ValueError(f"{learner} has readouts {READOUTS[learner]}, not {readout!r}")
    obs = np.asarray(obs)
    if obs.dtype != np.float32 or obs.ndim != 2 or obs.shape[1] != OBS_DIM:
        raise ValueError(f"obs must be float32 [n, {OBS_DIM}]")
    n = len(obs)
    if learner == "cql" and readout == "scored":
        eps = np.asarray(eps)
        if eps.dtype != np.float32 or eps.shape != (n, NUM_ACTIONS):
            raise ValueError("CQL's scored command needs eps float32 [n, 4]")
    else:
        eps = np.zeros((n, NUM_ACTIONS), np.float32)          # unused
    flags = (actor.meta.get("actor_ln"), actor.meta.get("norm_obs")) if learner == "rebrac" else None
    key = ("unifloral", learner, readout, flags)
    if key not in _CACHE:

        def run(params, obs, obs_mean, obs_std, eps, keys):
            if learner == "td3_bc":                                       # td3_bc.py:314
                net = U.DeterministicTanhActor(NUM_ACTIONS, obs_mean, obs_std)
            elif learner == "rebrac":                                     # rebrac.py:345-346
                net = U.DeterministicTanhActor(NUM_ACTIONS, obs_mean, obs_std, flags[0], flags[1])
            elif learner == "iql":                                        # iql.py:303
                net = U.TanhGaussianActor(NUM_ACTIONS, obs_mean, obs_std)
            else:                                                         # cql.py:358
                net = U.TanhGaussianActor(NUM_ACTIONS)

            def _policy_step(rng, o, e):
                if learner == "iql":
                    pi = net.apply(params, o, eval=True)
                else:
                    pi = net.apply(params, o)
                if learner == "cql" and readout == "tanh_mean":
                    action = pi.bijector.forward(pi.distribution.loc)
                elif learner == "cql":
                    action = pi.bijector.forward(pi.distribution.scale * e + pi.distribution.loc)
                else:
                    action = pi.sample(seed=rng)
                return jnp.nan_to_num(action)

            return jax.vmap(_policy_step)(keys, obs, eps)

        _CACHE[key] = jax.jit(run)
    if learner in NORMALIZED:
        obs_mean, obs_std = jnp.asarray(actor.norm["obs_mean"]), jnp.asarray(actor.norm["obs_std"])
    else:                                                                 # unused (norm_obs False; CQL has none)
        obs_mean, obs_std = jnp.zeros(OBS_DIM, jnp.float32), jnp.ones(OBS_DIM, jnp.float32)
    keys = jax.random.split(jax.random.PRNGKey(0), n)                     # unused by deterministic actors
    out = _CACHE[key](actor.params, jnp.asarray(obs), obs_mean, obs_std, jnp.asarray(eps), keys)
    return np.asarray(out)


def open_loop_commands(states_path, actor_path, out_path):
    """S7's open loop, the training environment's half: read the saved states (evaluate.open_loop_states wrote them:
    obs float32 [n, 13], and for CQL eps float32 [n, 4]) and the actor file, compute the learner's own float32
    commands with unifloral_commands, and write them to out_path (handoff.py): commands float32 [n, 4], and for CQL
    also commands_tanh_mean. The meta names both input files' SHA-256 and the import record. Returns the manifest."""
    from experiments.shift_hosts import unifloral_import as UI

    states, smanifest = H.read(states_path)
    actor = read_actor_file(actor_path)
    if smanifest["meta"].get("actor_npz_sha256") != actor.npz_sha256:
        raise ActorFileError(f"{states_path} was written for another actor file")
    out = {"commands": unifloral_commands(actor, states["obs"], states.get("eps"), "scored")}
    if actor.learner == "cql":
        out["commands_tanh_mean"] = unifloral_commands(actor, states["obs"], None, "tanh_mean")
    rec = UI.record()
    meta = dict(kind="s7-open-loop-commands", learner=actor.learner, states_npz_sha256=smanifest["npz_sha256"],
                actor_npz_sha256=actor.npz_sha256, network="the learner's own Unifloral class",
                device=rec.get("device"), xla_flags=rec.get("xla_flags", {}).get("in_effect"),
                matmul_precision=rec.get("matmul_precision"), unifloral_sha256=rec.get("sha256", {}).get(actor.learner),
                packages={k: rec.get("packages", {}).get(k) for k in ("jax", "jaxlib", "flax", "distrax", "numpy")})
    return H.write(out_path, out, meta=meta)

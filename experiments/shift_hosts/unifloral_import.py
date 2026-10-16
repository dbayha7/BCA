"""Unifloral's four vendored learner files, imported unchanged after a fingerprint check, with the XLA_FLAGS repair.

An experiment-local attachment (plan section 3.1, PLAN_UNIFLORAL.md): it changes none of the repository's declared
methods and edits nothing under baselines/unifloral/. Training environment only (/opt/bca-venv): the files import
distrax, d4rl, gym, tyro and wandb at the top (td3_bc.py:7-17), which /opt/bca-shift-venv lacks.

prepare() does the following once per process, in this order (stop rules S1 and S9; test (f)):
  1. environment  The variables of Unifloral's Dockerfile (baselines/unifloral/Dockerfile:42-45, 57-58) must already
                  be in the process environment when Python starts: without LD_LIBRARY_PATH=/root/.mujoco/mujoco210/bin
                  every import fails inside mujoco-py (environment report, section 3.2). launch_env() gives them to a
                  launcher; prepare() refuses without them (LaunchEnvironmentError).
  2. fingerprint  Each file's SHA-256 must equal configs/unifloral.json's source_sha256 entry (revision at line 5,
                  hashes from line 6), else PinnedCodeError (stop rule S1). Checked again just before each import.
  3. import       importlib by path under the module name "shift_hosts_unifloral_<learner>", never "__main__", so the
                  training block (td3_bc.py:282, cql.py:328, iql.py:271, rebrac.py:312) never runs. Bytecode writing is
                  off (sys.dont_write_bytecode), so no __pycache__ appears in baselines/.
  4. XLA_FLAGS    Every file assigns os.environ["XLA_FLAGS"] = "--xla_gpu_triton_gemm_any=True" when imported
                  (td3_bc.py:19, cql.py:21, iql.py:20, rebrac.py:20), replacing any earlier value. After the imports,
                  and only if JAX's backend has not started yet (else BackendStartedError: the flags could no longer
                  take effect), XLA_FLAGS is set to choice 8's flags (--xla_gpu_deterministic_ops=true; on the
                  processor also --xla_cpu_multi_thread_eigen=false, as the environment report's timings used) and
                  jax_default_matmul_precision to "highest". The record keeps XLA_FLAGS before the imports, as the
                  imports left it, and in effect.
  5. backend      Optionally starts JAX's backend on the requested device and records it (devices, platform version).

The learner's device comes from prepare(device=...) or the environment variable SHIFT_HOSTS_DEVICE ("cpu", the
default, or "gpu"). On the card XLA_PYTHON_CLIENT_PREALLOCATE must be "false" (set when absent). On the processor
jax_platforms is set to "cpu".

Rule for every process of the training environment: call prepare() before anything can start JAX's backend (creating
any JAX array does), and before importing a package module that needs Unifloral's pieces (load()).

Also here, restated with the line numbers of each training block, for every caller (the runner and the tests):
make_transition (td3_bc.py:300-306; rebrac.py:330-337 with Unifloral's roll) and init_like_training_block (the
network creation and key splits of td3_bc.py:285, 308-327; cql.py:331, 354-375; iql.py:274, 297-319;
rebrac.py:315, 339-359). No environment, D4RL dataset, logger or evaluation is created.
"""

import dataclasses
import hashlib
import importlib.metadata
import importlib.util
import json
import os
import platform
import sys
import time
from collections import namedtuple
from pathlib import Path

sys.dont_write_bytecode = True

PACKAGE = Path(__file__).resolve().parent
REPO = PACKAGE.parents[1]
ALGORITHMS = REPO / "baselines" / "unifloral" / "algorithms"
MANIFEST = REPO / "configs" / "unifloral.json"
REVISION = "f2dc1278eae18ed3e22c92255119c54885414d17"          # configs/unifloral.json:5
LEARNERS = ("td3_bc", "cql", "iql", "rebrac")
MODULE_PREFIX = "shift_hosts_unifloral_"
TRAINING_BLOCK_LINE = dict(td3_bc=282, cql=328, iql=271, rebrac=312)   # `if __name__ == "__main__":`
UNIFLORAL_XLA_FLAGS = "--xla_gpu_triton_gemm_any=True"                    # td3_bc.py:19 and the other three
DETERMINISTIC_FLAGS = ("--xla_gpu_deterministic_ops=true",)              # choice 8
CPU_FLAGS = ("--xla_cpu_multi_thread_eigen=false",)                      # one processor thread per process
MATMUL_PRECISION = "highest"                                              # choice 8: full float32
DEVICE_ENV = "SHIFT_HOSTS_DEVICE"
DEVICES = ("cpu", "gpu")
MUJOCO_BIN = "/root/.mujoco/mujoco210/bin"
# baselines/unifloral/Dockerfile:42-45 and 57-58 (LD_LIBRARY_PATH must contain MUJOCO_BIN; other entries may follow)
DOCKERFILE_ENV = {
    "D4RL_SUPPRESS_IMPORT_ERROR": "1",
    "MUJOCO_GL": "osmesa",
    "PYOPENGL_PLATFORM": "osmesa",
    "DISPLAY": "",
    "MUJOCO_PY_MUJOCO_PATH": "/root/.mujoco/mujoco210",
    "LD_LIBRARY_PATH": MUJOCO_BIN,
}
PACKAGES = ("jax", "jaxlib", "jax-cuda12-plugin", "jax-cuda12-pjrt", "numpy", "scipy", "flax", "optax", "distrax",
            "chex", "gym", "d4rl", "tyro", "wandb", "mujoco-py")


class PinnedCodeError(RuntimeError):
    """A Unifloral file's SHA-256 differs from configs/unifloral.json (stop rule S1)."""


class BackendStartedError(RuntimeError):
    """JAX's backend started before XLA_FLAGS was repaired, so choice 8's flags could not take effect."""


class LaunchEnvironmentError(RuntimeError):
    """The process was started without the Dockerfile's variables, or with a device setting that conflicts."""


_STATE = dict(device=None, modules={}, record=None)


# ------------------------------------------------------------------------------------------------ fingerprints

def sha256_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def manifest_sha256(manifest=MANIFEST):
    """{"algorithms/<file>.py": sha256, ...}: the source_sha256 block of configs/unifloral.json, after checking that the
    manifest names the pinned revision."""
    body = json.loads(Path(manifest).read_text(encoding="utf8"))
    if body.get("revision") != REVISION:
        raise PinnedCodeError(f"{manifest} names revision {body.get('revision')!r}, expected {REVISION}")
    return dict(body["source_sha256"])


def verify_file(path, expected):
    """The file's SHA-256, after checking that it equals `expected` (stop rule S1)."""
    path = Path(path)
    if not path.is_file():
        raise PinnedCodeError(f"stop rule S1: {path} does not exist")
    digest = sha256_file(path)
    if not isinstance(expected, str) or digest != expected:
        raise PinnedCodeError(f"stop rule S1: {path} has SHA-256 {digest}, but the pinned value is {expected}")
    return digest


def check_pinned(learner, *, algorithms=ALGORITHMS, manifest=MANIFEST):
    """SHA-256 of Unifloral's <learner>.py after checking it against configs/unifloral.json."""
    if learner not in LEARNERS:
        raise ValueError(f"learner must be one of {LEARNERS}, not {learner!r}")
    expected = manifest_sha256(manifest).get(f"algorithms/{learner}.py")
    return verify_file(Path(algorithms) / f"{learner}.py", expected)


# ------------------------------------------------------------------------------------------------ environment

def launch_env(base=None):
    """A copy of `base` (default os.environ) with the Dockerfile's variables, PYTHONDONTWRITEBYTECODE=1 and
    OMP_NUM_THREADS=1, for starting a training-environment process. LD_LIBRARY_PATH keeps MuJoCo's folder first."""
    env = dict(os.environ if base is None else base)
    for name, value in DOCKERFILE_ENV.items():
        if name == "LD_LIBRARY_PATH":
            rest = [p for p in env.get(name, "").split(":") if p and p != MUJOCO_BIN]
            env[name] = ":".join([MUJOCO_BIN] + rest)
        else:
            env[name] = value
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env.setdefault("OMP_NUM_THREADS", "1")
    return env


def missing_launch_env(env=None):
    """Names of the Dockerfile's variables that `env` (default os.environ) lacks or holds with another value."""
    env = os.environ if env is None else env
    bad = []
    for name, value in DOCKERFILE_ENV.items():
        if name == "LD_LIBRARY_PATH":
            if MUJOCO_BIN not in env.get(name, "").split(":"):
                bad.append(name)
        elif env.get(name) != value:
            bad.append(name)
    return bad


def backend_started():
    """Whether any JAX backend has been initialized in this process."""
    if "jax" not in sys.modules:
        return False
    from jax._src import xla_bridge

    return bool(xla_bridge.backends_are_initialized())


def _device(device):
    device = device or os.environ.get(DEVICE_ENV) or "cpu"
    if device not in DEVICES:
        raise LaunchEnvironmentError(f"device must be one of {DEVICES} (argument or {DEVICE_ENV}), not {device!r}")
    return device


def chosen_flags(device, extra=()):
    """The XLA_FLAGS value set after the imports: choice 8's deterministic flag, the processor's one-thread flag when
    device is "cpu", then any extra flags."""
    flags = list(DETERMINISTIC_FLAGS) + (list(CPU_FLAGS) if device == "cpu" else []) + list(extra)
    return " ".join(flags)


# ------------------------------------------------------------------------------------------------ import

def _import(learner, *, algorithms=None, manifest=MANIFEST):
    """Import Unifloral's <learner>.py under its own module name (never __main__), executing exactly the bytes whose
    SHA-256 was checked against configs/unifloral.json (stop rule S1): the file is read once, hashed, compiled from those
    bytes and executed, so a change between the check and the import cannot slip in, and no bytecode cache (.pyc) is
    read in place of the source."""
    path = Path(ALGORITHMS if algorithms is None else algorithms) / f"{learner}.py"
    if learner not in LEARNERS:
        raise ValueError(f"learner must be one of {LEARNERS}, not {learner!r}")
    expected = manifest_sha256(manifest).get(f"algorithms/{learner}.py")
    if not path.is_file():
        raise PinnedCodeError(f"stop rule S1: {path} does not exist")
    source = path.read_bytes()
    digest = hashlib.sha256(source).hexdigest()
    if not isinstance(expected, str) or digest != expected:
        raise PinnedCodeError(f"stop rule S1: {path} has SHA-256 {digest}, but the pinned value is {expected}")
    name = MODULE_PREFIX + learner
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    code = compile(source, str(path), "exec", dont_inherit=True)
    sys.modules[name] = module
    try:
        exec(code, module.__dict__)
    except BaseException:
        sys.modules.pop(name, None)
        raise
    if module.__name__ == "__main__":
        raise AssertionError("a Unifloral file must never be imported as __main__")
    return module


def prepare(learners=LEARNERS, device=None, *, start_backend=True, extra_xla_flags=()):
    """Check the launch environment, fingerprint-check and import the requested learner files, repair XLA_FLAGS, set
    full float32 matrix products and (by default) start JAX's backend on `device`. Call once per process, first.
    Returns the import record (record()). A second call with the same device and learners returns the same record."""
    learners = tuple(learners)
    device = _device(device)
    if _STATE["record"] is not None:
        if _STATE["device"] != device or not set(learners) <= set(_STATE["modules"]):
            raise LaunchEnvironmentError("prepare() already ran in this process with "
                                         f"device={_STATE['device']!r} and learners {sorted(_STATE['modules'])}")
        return _STATE["record"]
    if any(l not in LEARNERS for l in learners) or not learners:
        raise ValueError(f"learners must be a non-empty subset of {LEARNERS}")
    bad = missing_launch_env()
    if bad:
        raise LaunchEnvironmentError(
            f"the Dockerfile's variables {bad} are missing or differ (baselines/unifloral/Dockerfile:42-45, 57-58); "
            "start the process with unifloral_import.launch_env(), for example: env D4RL_SUPPRESS_IMPORT_ERROR=1 "
            "MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa DISPLAY= MUJOCO_PY_MUJOCO_PATH=/root/.mujoco/mujoco210 "
            "LD_LIBRARY_PATH=/root/.mujoco/mujoco210/bin ...")
    if backend_started():
        raise BackendStartedError("JAX's backend started before prepare(); choice 8's XLA flags could not take effect")
    flags_before = os.environ.get("XLA_FLAGS")
    started = time.perf_counter()
    sha, after = {}, {}
    for learner in learners:
        sha[learner] = check_pinned(learner)
        _STATE["modules"][learner] = _import(learner)
        after[learner] = os.environ.get("XLA_FLAGS")
    import_seconds = time.perf_counter() - started
    flags_after_import = os.environ.get("XLA_FLAGS")
    import jax

    started_before_repair = backend_started()
    if started_before_repair:
        raise BackendStartedError("importing the Unifloral files started JAX's backend; the repair cannot take effect")
    os.environ["XLA_FLAGS"] = chosen_flags(device, extra_xla_flags)
    platforms = jax.config.jax_platforms
    if device == "cpu":
        if platforms not in (None, "", "cpu"):
            raise LaunchEnvironmentError(f"device 'cpu' but jax_platforms is {platforms!r}")
        jax.config.update("jax_platforms", "cpu")
    else:
        if platforms not in (None, "", "cuda", "gpu", "cuda,cpu", "gpu,cpu"):
            raise LaunchEnvironmentError(f"device 'gpu' but jax_platforms is {platforms!r} (unset JAX_PLATFORMS)")
        prealloc = os.environ.get("XLA_PYTHON_CLIENT_PREALLOCATE")
        if prealloc not in (None, "false"):
            raise LaunchEnvironmentError("XLA_PYTHON_CLIENT_PREALLOCATE must be 'false' on the shared card")
        os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    jax.config.update("jax_default_matmul_precision", MATMUL_PRECISION)
    record = dict(
        unifloral_revision=REVISION, manifest=str(MANIFEST.relative_to(REPO)), manifest_sha256=sha256_file(MANIFEST),
        learners=list(learners), sha256={l: sha[l] for l in learners},
        module_names={l: MODULE_PREFIX + l for l in learners}, import_seconds=round(import_seconds, 3),
        device=device,
        xla_flags=dict(before_import=flags_before, after_each_import=after, after_import=flags_after_import,
                       in_effect=os.environ["XLA_FLAGS"]),
        unifloral_flags_overwrote=flags_after_import == UNIFLORAL_XLA_FLAGS,
        backend_started_before_repair=started_before_repair,
        matmul_precision=jax.config.jax_default_matmul_precision,
        jax_platforms=jax.config.jax_platforms,
        env={k: os.environ.get(k) for k in sorted(set(DOCKERFILE_ENV) | {
            "XLA_PYTHON_CLIENT_PREALLOCATE", "JAX_PLATFORMS", "OMP_NUM_THREADS", "PYTHONDONTWRITEBYTECODE",
            "CUDA_VISIBLE_DEVICES", DEVICE_ENV})},
        python=sys.version.split()[0], executable=sys.executable, platform=platform.platform(),
        packages=package_versions(), affinity=sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity")
        else None, dont_write_bytecode=bool(sys.dont_write_bytecode))
    _STATE.update(device=device, record=record)
    if start_backend:
        start(device)
    return record


def start(device=None):
    """Start JAX's backend (after prepare()) and add the devices to the record. Refuses a device mismatch."""
    if _STATE["record"] is None:
        raise RuntimeError("call unifloral_import.prepare() first")
    import jax

    device = _device(device or _STATE["device"])
    if device != _STATE["device"]:
        raise LaunchEnvironmentError(f"prepared for {_STATE['device']!r}, asked to start {device!r}")
    backend = jax.default_backend()
    if backend != device:
        raise LaunchEnvironmentError(f"asked for device {device!r}, JAX's default backend is {backend!r}")
    devs = jax.devices()
    if os.environ.get("XLA_FLAGS") != _STATE["record"]["xla_flags"]["in_effect"]:
        raise LaunchEnvironmentError("XLA_FLAGS changed after the repair")
    _STATE["record"].update(backend=backend, devices=[str(d) for d in devs],
                            device_kind=[getattr(d, "device_kind", None) for d in devs],
                            platform_version=getattr(getattr(devs[0], "client", None), "platform_version", None),
                            card=card_record() if device == "gpu" else None)
    return _STATE["record"]


def card_record():
    """The card's name, driver version and memory as nvidia-smi reports them (read-only query; stop rule S9)."""
    import subprocess

    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=name,driver_version,memory.total",
                              "--format=csv,noheader"], capture_output=True, text=True, timeout=30, check=True)
    except (OSError, subprocess.SubprocessError) as error:
        return dict(error=repr(error))
    name, driver, memory = [s.strip() for s in out.stdout.strip().splitlines()[0].split(",")]
    return dict(name=name, driver_version=driver, memory_total=memory)


def load(learner):
    """The imported Unifloral module of `learner` (prepare() must have imported it)."""
    if learner not in _STATE["modules"]:
        raise RuntimeError(f"Unifloral's {learner}.py is not imported: call unifloral_import.prepare() first, "
                           f"with {learner!r} among its learners")
    return _STATE["modules"][learner]


def record():
    """The import record of prepare() (for run records: hashes, flags, environment, devices)."""
    if _STATE["record"] is None:
        raise RuntimeError("call unifloral_import.prepare() first")
    return json.loads(json.dumps(_STATE["record"]))


def package_versions():
    out = {}
    for name in PACKAGES:
        try:
            out[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            out[name] = None
    return out


# ------------------------------------------------------------------------------------------------ training blocks

Built = namedtuple("Built", "learner rng agent_state dataset obs_mean obs_std nets apply_fns dummy_obs dummy_action")


def args_for(learner, **overrides):
    """Unifloral's own Args defaults (td3_bc.py:23-46, cql.py:25-49, iql.py:24-46, rebrac.py:24-52) with `overrides`
    (the plan's are seed, gamma=0.98 and num_updates=100_000; section 3.2)."""
    return dataclasses.replace(load(learner).Args(), **overrides)


def make_transition(learner, rows, *, next_action="roll"):
    """Unifloral's Transition of `rows` (a mapping with obs, action, reward, next_obs, done; numpy or JAX arrays), as
    the training block builds it from D4RL's dictionary (td3_bc.py:300-306, cql.py:346-352, iql.py:289-295). ReBRAC
    adds next_action (rebrac.py:330-337): next_action="roll" takes Unifloral's jnp.roll(actions, -1, axis=0)
    (rebrac.py:335, for training rows); an array is used as given (bank rows: the bank rule of plan section 3.8)."""
    import jax.numpy as jnp

    U = load(learner)
    fields = dict(obs=jnp.array(rows["obs"]), action=jnp.array(rows["action"]), reward=jnp.array(rows["reward"]),
                  next_obs=jnp.array(rows["next_obs"]))
    if learner == "rebrac":
        if isinstance(next_action, str):
            if next_action != "roll":
                raise ValueError("next_action must be 'roll' or an array")
            fields["next_action"] = jnp.roll(rows["action"], -1, axis=0)
        else:
            fields["next_action"] = jnp.array(next_action)
    fields["done"] = jnp.array(rows["done"])
    return U.Transition(**fields)


def init_like_training_block(learner, args, rows, *, obs_dim=13, action_dim=4, next_action="roll"):
    """Each training block's set-up, restated without the gym environment, D4RL, logger or evaluation:
    rng = PRNGKey(args.seed) (td3_bc.py:285, cql.py:331, iql.py:274, rebrac.py:315); the Transition
    (make_transition); obs_mean = dataset.obs.mean(axis=0), obs_std = nan_to_num(dataset.obs.std(axis=0), nan=1.0)
    (td3_bc.py:310-311, iql.py:299-300, rebrac.py:341-342; CQL's block computes none, but the scale network needs them:
    choice 3); dummy inputs of 13 and 4 zeros (env.single_observation_space.shape, num_actions); the networks; the key
    split (3 ways td3_bc.py:318, rebrac.py:350; 4 ways cql.py:363, iql.py:308) and the train states, target networks
    sharing their online network's key. Returns Built; Built.rng is the key after that split, the one the chunk loop
    starts from, and apply_fns is in the order of make_train_step's apply-function arguments."""
    import jax
    import jax.numpy as jnp

    U = load(learner)
    rng = jax.random.PRNGKey(args.seed)
    dataset = make_transition(learner, rows, next_action=next_action)
    num_actions = action_dim
    obs_mean = dataset.obs.mean(axis=0)
    obs_std = jnp.nan_to_num(dataset.obs.std(axis=0), nan=1.0)
    dummy_obs = jnp.zeros((obs_dim,))
    dummy_action = jnp.zeros(num_actions)
    if learner == "td3_bc":                                             # td3_bc.py:314-324
        actor_net = U.DeterministicTanhActor(num_actions, obs_mean, obs_std)
        q_net = U.DualQNetwork(obs_mean, obs_std)
        rng, rng_actor, rng_q = jax.random.split(rng, 3)
        agent_state = U.AgentTrainState(
            actor=U.create_train_state(args, rng_actor, actor_net, [dummy_obs]),
            actor_target=U.create_train_state(args, rng_actor, actor_net, [dummy_obs]),
            dual_q=U.create_train_state(args, rng_q, q_net, [dummy_obs, dummy_action]),
            dual_q_target=U.create_train_state(args, rng_q, q_net, [dummy_obs, dummy_action]),
        )
        nets = dict(actor=actor_net, q=q_net)
        apply_fns = (actor_net.apply, q_net.apply)
    elif learner == "rebrac":                                           # rebrac.py:345-356
        actor_cls = U.DeterministicTanhActor
        actor_net = actor_cls(num_actions, obs_mean, obs_std, args.actor_ln, args.norm_obs)
        q_net = U.DualQNetwork(obs_mean, obs_std, args.critic_ln, args.norm_obs)
        rng, rng_actor, rng_q = jax.random.split(rng, 3)
        agent_state = U.AgentTrainState(
            actor=U.create_train_state(args, rng_actor, actor_net, [dummy_obs]),
            actor_target=U.create_train_state(args, rng_actor, actor_net, [dummy_obs]),
            dual_q=U.create_train_state(args, rng_q, q_net, [dummy_obs, dummy_action]),
            dual_q_target=U.create_train_state(args, rng_q, q_net, [dummy_obs, dummy_action]),
        )
        nets = dict(actor=actor_net, q=q_net)
        apply_fns = (actor_net.apply, q_net.apply)
    elif learner == "cql":                                              # cql.py:358-370
        actor_net = U.TanhGaussianActor(num_actions)
        q_net = U.VectorQ(args.num_critics)
        alpha_net = U.EntropyCoef()
        rng, rng_actor, rng_q, rng_alpha = jax.random.split(rng, 4)
        actor_lr = args.actor_lr if args.actor_lr is not None else args.lr
        agent_state = U.AgentTrainState(
            actor=U.create_train_state(args, rng_actor, actor_net, [dummy_obs], actor_lr),
            vec_q=U.create_train_state(args, rng_q, q_net, [dummy_obs, dummy_action]),
            vec_q_target=U.create_train_state(args, rng_q, q_net, [dummy_obs, dummy_action]),
            alpha=U.create_train_state(args, rng_alpha, alpha_net, []),
        )
        nets = dict(actor=actor_net, q=q_net, alpha=alpha_net)
        apply_fns = (actor_net.apply, q_net.apply, alpha_net.apply)
    elif learner == "iql":                                              # iql.py:303-314
        actor_net = U.TanhGaussianActor(num_actions, obs_mean, obs_std)
        q_net = U.DualQNetwork(obs_mean, obs_std)
        value_net = U.StateValueFunction(obs_mean, obs_std)
        rng, rng_actor, rng_q, rng_value = jax.random.split(rng, 4)
        agent_state = U.AgentTrainState(
            actor=U.create_train_state(args, rng_actor, actor_net, [dummy_obs]),
            dual_q=U.create_train_state(args, rng_q, q_net, [dummy_obs, dummy_action]),
            dual_q_target=U.create_train_state(args, rng_q, q_net, [dummy_obs, dummy_action]),
            value=U.create_train_state(args, rng_value, value_net, [dummy_obs]),
        )
        nets = dict(actor=actor_net, q=q_net, value=value_net)
        apply_fns = (actor_net.apply, q_net.apply, value_net.apply)
    else:
        raise ValueError(f"learner must be one of {LEARNERS}")
    return Built(learner, rng, agent_state, dataset, obs_mean, obs_std, nets, apply_fns, dummy_obs, dummy_action)


def make_unifloral_step(built, args):
    """Unifloral's own (imported, unchanged) make_train_step for a Built: the step of the "alone" setups and room
    references (td3_bc.py:327, cql.py:373-375, iql.py:317-319, rebrac.py:359)."""
    U = load(built.learner)
    return U.make_train_step(args, *built.apply_fns, built.dataset)

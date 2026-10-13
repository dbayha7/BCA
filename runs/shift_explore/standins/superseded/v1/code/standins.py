"""EXPLORATORY (Stage 1 of outputs/shift/beyond-the-data/PLAN.md, section 4.2): data-only stand-ins in the same penalty
slot as the room test's perfect band. Outside the gate; changes no pre-registered rule. NO BCA: no band, calibrator,
threshold, dose or BCA update is fitted, read or called. Nothing from algorithms/td3_bc_bca.py or learners.bca_* is
called; from calibration/ only calibration/bank.stratified_bank runs, because the gate's data loader
(experiments/shift/dataset.py, through acceptance.load_seed) calls it to regenerate the log's calibration-bank split.

The room harness (runs/shift_explore/room/code/harness.py) is imported read-only: its bytes are hashed, checked
against the room's code manifest (and the manifest against its sidecar and the SHA-256 pinned here), and only then
executed. Its loaders (SeedContext), the range-hinge definition (hinge_bounds, HINGE_SCALE), the room's main-oracle
slope (slope_rms, for the ladder scale), the check-(e) rows and grid (shape_rows, SHAPE_GRID, ...) and small helpers
are used unchanged. Nothing under runs/shift_explore/room, runs/shift or experiments/ is written.

Every arm trains the actor against check 2's frozen gapped critic of the stalling world (its no-stall twin in the
no-stall world), from the saved cloned actor, alpha 5, 20,000 actor steps in 4,000-step chunks, check 2's stream of
batch rows, with the loss of the room test:

    loss = -lambda * mean_i[ Q1(s_i, pi(s_i)) - k * W(s_i, pi(s_i)) ] + mean_ij (pi(s_i)_j - a_ij)^2
    lambda = alpha / stopgrad( mean_i |Q1(s_i, pi(s_i))| )

W is a data-only stand-in. Its own parameters (the critic, the classifier, the distance surface, the constants) get
no gradient: only the actor's parameters are differentiated, so W acts through the command pi(s) alone.

Notation: s_i a training state, a_i its recorded command, a = pi(s_i) the actor's command, c(s) the cloned actor's
command, [x]_+ = max(x, 0).

  range_hinge       100 * sum_j ([a_j - hi_j]_+^2 + [lo_j - a_j]_+^2): the room's arm, unchanged
  density           [N(s_i, a) - N(s_i, a_i)]_+,  N(s, a) = 1/2 sum_k ((a_k - c_k(s)) / sigma_k)^2
  classifier_ratio  softplus(log w(s_i, a) - l*), w = g / (1 - g), g a classifier of target pairs against recorded
                    pairs, refitted every 4,000 steps with the arm's own actor; l* = log(0.1 / 0.9 * sum_bank w)
  distance          [f(s_i, a) - f(s_i, a_i)]_+, f a smooth surface fitted to d10 (the distance to the 10th-nearest
                    training row in standard units)
  two_head          [D(s_i, a) - D(s_i, a_i)]_+, D = |Q1 - Q2| of the frozen critic
  reported beside them, the plan's literal forms:
  density_plan      [N(s_i, a) - q95_N]_+
  distance_plan     [f(s_i, a) - q95_d]_+
  two_head_plan     |Q1 - Q2|(s_i, a)
"""

import hashlib
import json
import math
import sys
import time
import types
from dataclasses import asdict, dataclass
from pathlib import Path

sys.dont_write_bytecode = True

import numpy as np  # noqa: E402

ROOT = Path("/home/user/bca")
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

HERE = Path(__file__).resolve().parent
OUT = ROOT / "runs" / "shift_explore" / "standins"
SPEC_PATH = OUT / "spec.json"
MANIFEST_NAME = "code_manifest.json"
ROOM_OUT = ROOT / "runs" / "shift_explore" / "room"
PLAN_PATH = ROOT / "outputs" / "shift" / "beyond-the-data" / "PLAN.md"

# The room test's code as pinned when this component was written (its manifest v1, spec v2). Any change refuses.
ROOM_PINNED = dict(manifest_sha256="469ada26b9af7eff33be17b8241867e83e44c45a30d2d137dffe7deca3f32391",
                   spec_sha256="56bbda24cdbaa10cc0de0261450de5d9034e08e188c88e9432be83d8cf136bf4",
                   harness_sha256="97f547b14d55c3afa392cbdf99508e45c1b862d8a285030808c5783a067ef2cf")


def _sha_bytes(b):
    return hashlib.sha256(b).hexdigest()


def file_sha256(path):
    return _sha_bytes(Path(path).read_bytes())


def load_room_harness():
    """The room harness, executed from the exact bytes whose SHA-256 the room's code manifest records. Refuses when
    the manifest differs from its sidecar or from the pinned SHA-256, when the manifest names another room spec, or
    when the pinned copy (runs/shift_explore/room/code/harness.py) or the room's source file differs."""
    man_path = ROOM_OUT / "code_manifest.json"
    text = man_path.read_bytes()
    side = (ROOM_OUT / "code_manifest.json.sha256").read_text(encoding="utf8").split()[0]
    if _sha_bytes(text) != side:
        raise SystemExit("refused: the room's code manifest differs from its sidecar")
    if side != ROOM_PINNED["manifest_sha256"]:
        raise SystemExit("refused: the room's code manifest is not the one this component was pinned to")
    man = json.loads(text)
    if man["spec_sha256"] != ROOM_PINNED["spec_sha256"] or file_sha256(ROOM_OUT / "spec.json") != man["spec_sha256"]:
        raise SystemExit("refused: the room's spec is not the one this component was pinned to")
    want = man["files"]["harness.py"]
    if want != ROOM_PINNED["harness_sha256"]:
        raise SystemExit("refused: the room manifest's harness.py is not the pinned one")
    copy = ROOM_OUT / "code" / "harness.py"
    src = copy.read_bytes()
    if _sha_bytes(src) != want:
        raise SystemExit("refused: the room's pinned copy of harness.py differs from its manifest")
    source = Path(man["source_dir"]) / "harness.py"
    if source.exists() and file_sha256(source) != want:
        raise SystemExit("refused: the room's source harness.py differs from its manifest")
    mod = types.ModuleType("room_harness")
    mod.__file__ = str(copy)
    sys.modules["room_harness"] = mod
    exec(compile(src, str(copy), "exec"), mod.__dict__)
    return mod, dict(manifest_sha256=side, harness_sha256=want, copy=str(copy), source=str(source))


H, ROOM_IMPORT = load_room_harness()
A, L, TC = H.A, H.L, H.TC
M = H.M

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import flax.linen as nn  # noqa: E402
from flax.training.train_state import TrainState  # noqa: E402

LABEL = ("EXPLORATORY: Stage 1 data-only stand-ins in the lower-edge channel (plan section 4.2), computed outside the "
         "gate on saved gate artifacts and the room test's locked strengths; not a gated output; changes no "
         "pre-registered rule; no BCA band, calibrator, threshold, dose or BCA update was used (data-only stand-ins "
         "only)")

SEEDS = H.SEEDS
ALPHA = H.ALPHA
STEPS, CHUNK = H.STEPS, H.CHUNK
CHECK_STEPS = 400
W0_TOL = H.W0_TOL
LADDER = (0.3, 1.0, 3.0, 10.0)               # 4 rungs of slope-normalised knobs (plan 4.2: "4-rung ladder")
EXTEND_FACTOR = math.sqrt(10.0)
MAX_EXTENSIONS = 3
HELD_ROWS = 5000
LEVEL = 95.0
MISCOVERAGE = 0.1
K_NN = 10
KNN_CANDIDATES = 16
CLS_STEPS, CLS_BATCH, CLS_LR, CLS_HIDDEN = 2000, 512, 1e-3, 64
SURF_PER_TYPE, SURF_TYPES, SURF_VALID = 20_000, 6, 20_000
SURF_STEPS, SURF_BATCH, SURF_LR, SURF_LR_END, SURF_HIDDEN = 12_000, 1024, 1e-3, 5e-5, 128
SURF_NOISE = 0.25
TAG_HELD, TAG_CLS_ROWS, TAG_SURF_QUERIES, TAG_CLS_INIT, TAG_SURF_INIT, TAG_SURF_ROWS = 81, 82, 83, 84, 85, 86

FAMILIES = ("range_hinge", "density", "classifier_ratio", "distance", "two_head",
            "density_plan", "distance_plan", "two_head_plan")
MAIN_FAMILIES = FAMILIES[:5]
REPORTED_FAMILIES = FAMILIES[5:]
FAMILY_KIND = {"range_hinge": ("hinge", 0), "density": ("density", 0), "density_plan": ("density", 1),
               "classifier_ratio": ("classifier", 0), "distance": ("distance", 0), "distance_plan": ("distance", 1),
               "two_head": ("two_head", 0), "two_head_plan": ("two_head", 1)}
KINDS = ("plain", "hinge", "density", "classifier", "distance", "two_head")
N_COEF = {"plain": 1, "hinge": 1, "density": 2, "classifier": 1, "distance": 2, "two_head": 2}
FORMULAS = {
    "range_hinge": "100 * sum_j ([a_j - hi_j]_+^2 + [lo_j - a_j]_+^2), lo_j, hi_j the 0.5% and 99.5% quantiles of the "
                   "gapped training commands per thruster (the room's range_hinge, unchanged)",
    "density": "[N(s_i, a) - N(s_i, a_i)]_+, N(s, a) = 1/2 sum_k ((a_k - c_k(s)) / sigma_k)^2, c the cloned actor, "
               "sigma_k^2 = mean_i (a_ik - c_k(s_i))^2 over the training rows (maximum likelihood)",
    "density_plan": "[N(s_i, a) - q95_N]_+, q95_N the split-conformal 95% level of N on 5,000 held-out recorded rows",
    "classifier_ratio": "softplus(log w(s_i, a) - l*), log w = the logit of a classifier g of target pairs "
                        "(s_i, clip(pi(s_i) + a_i - c(s_i), -1, 1)) against recorded pairs (s_i, a_i), refitted every "
                        "4,000 steps with the arm's own actor; l* = log(0.1 / 0.9 * sum_{j in bank} w(s_j, a_j))",
    "distance": "[f(s_i, a) - f(s_i, a_i)]_+, f a smooth surface (MLP) fitted to d10(s, a), the distance to the "
                "10th-nearest training row in standard units of (state, command)",
    "distance_plan": "[f(s_i, a) - q95_d]_+, q95_d the split-conformal 95% level of the exact d10 on 5,000 held-out "
                     "recorded rows",
    "two_head": "[D(s_i, a) - D(s_i, a_i)]_+, D(s, a) = |Q1(s, a) - Q2(s, a)| of the frozen critic",
    "two_head_plan": "|Q1(s_i, a) - Q2(s_i, a)| of the frozen critic (the plan's literal form)",
}


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


# ------------------------------------------------------------------------------------------------ JSON

def guard_out(path, out=None):
    """Refuse any output outside runs/shift_explore/standins (never runs/shift, the room test or experiments/)."""
    out = Path(out or OUT).resolve()
    p = Path(path).resolve()
    if out not in p.parents and p != out:
        raise SystemExit(f"refused: {p} is outside {out}")
    for protected in (ROOT / "runs" / "shift", ROOT / "experiments", ROOM_OUT):
        if p == protected.resolve() or protected.resolve() in p.parents:
            raise SystemExit(f"refused: {p} is inside {protected}")
    return p


def write_json(path, payload, out=None):
    """acceptance.write_json's format (strict sorted JSON, '<name>.sha256' sidecar), restricted to OUT."""
    path = guard_out(path, out)
    path.parent.mkdir(parents=True, exist_ok=True)
    text = A.canonical(payload)
    path.write_text(text, encoding="utf8")
    digest = _sha_bytes(text.encode("utf8"))
    path.with_name(path.name + ".sha256").write_text(f"{digest}  {path.name}\n", encoding="utf8")
    return digest


def read_json(path):
    """A JSON written with a sidecar; None when missing; SystemExit when its sidecar does not match."""
    path = Path(path)
    if not path.exists():
        return None
    text = path.read_text(encoding="utf8")
    side = path.with_name(path.name + ".sha256")
    if not side.exists() or side.read_text(encoding="utf8").split()[0] != _sha_bytes(text.encode("utf8")):
        raise SystemExit(f"refused: {path}: its SHA-256 differs from its sidecar")
    return json.loads(text)


def save_tree(path, tree, out=None):
    return L.save_tree(guard_out(path, out), tree)


CODE_FILES = ("standins.py", "run_standins.py", "launch.sh", "write_spec.py", "write_manifest.py", "checks_seed0.py",
              "dryrun.py", "test_marks.py")


def code_hashes(names=CODE_FILES):
    return {n: file_sha256(HERE / n) for n in names}


def load_spec():
    spec = read_json(SPEC_PATH)
    if spec is None:
        raise SystemExit(f"refused: no spec at {SPEC_PATH}; it must be written before any run")
    return spec, file_sha256(SPEC_PATH)


def check_code(spec_sha, out=None):
    """Refuse unless the manifest exists, matches its sidecar, names the current spec, and every file it lists (here
    and in its copy under OUT/code/) has the SHA-256 it recorded. Returns (manifest, its SHA-256)."""
    out = Path(out or OUT)
    path = out / MANIFEST_NAME
    man = read_json(path)
    if man is None:
        raise SystemExit(f"refused: no code manifest at {path}; write it (write_manifest.py) before any phase")
    if man.get("spec_sha256") != spec_sha:
        raise SystemExit("refused: the code manifest names another spec; write a new, labelled manifest")
    now = code_hashes(tuple(man["files"]))
    bad = [n for n, h in man["files"].items() if now[n] != h]
    copies = [n for n, h in man["files"].items()
              if not (out / "code" / n).exists() or file_sha256(out / "code" / n) != h]
    if bad or copies:
        raise SystemExit("refused: the code differs from the manifest (" + ", ".join(bad + [f"code/{n}" for n in copies])
                         + "); any change needs a new, labelled manifest")
    return man, file_sha256(path)


def check_gate_files(spec):
    now = H.gate_file_hashes()
    bad = [k for k, v in spec["gate_files_sha256"].items() if now.get(k) != v]
    if bad:
        raise SystemExit("refused: gate files changed since the spec: " + ", ".join(bad))


def begin(out=None):
    """Every phase and every check starts here: the spec, the gate files, the room import and the code manifest."""
    spec, spec_sha = load_spec()
    check_gate_files(spec)
    if spec["room"]["manifest_sha256"] != ROOM_IMPORT["manifest_sha256"] or \
            spec["room"]["harness_sha256"] != ROOM_IMPORT["harness_sha256"]:
        raise SystemExit("refused: the spec names another room harness")
    _, man_sha = check_code(spec_sha, out)
    return spec, spec_sha, man_sha


# ------------------------------------------------------------------------------------------------ the room test

def room_gate():
    """Plan 4.2: the stand-ins run only if the room test's room, recovery and cost (keeps-improvement) marks pass.
    Reads only runs/shift_explore/room/pooled.json's pass flags (sidecar-verified, labelled, naming the pinned spec)."""
    pooled = read_json(ROOM_OUT / "pooled.json")
    if pooled is None:
        raise SystemExit("refused: the room test's pooled look has not run (runs/shift_explore/room/pooled.json)")
    if not str(pooled.get("label", "")).startswith("EXPLORATORY") or pooled.get("kind") != "room-pooled" \
            or pooled.get("spec_sha256") != ROOM_PINNED["spec_sha256"]:
        raise SystemExit("refused: room pooled.json is not the room harness's labelled output for the pinned spec")
    ok = dict(complete=bool(pooled.get("complete")), room=bool(pooled["room"]["passed"]),
              recovery=bool(pooled["recovery"]["passed"]),
              keeps_improvement=bool(pooled["keeps_improvement"]["passed"]))
    if not all(ok.values()):
        raise SystemExit(f"refused: the room test's pooled marks did not all pass ({ok}); plan 4.2's stand-ins run only "
                         "if the room, recovery and cost marks pass")
    return dict(ok, pooled_sha256=file_sha256(ROOM_OUT / "pooled.json"))


def room_lock_target(rep, none_params):
    """The perfect band's strength for this seed, from the room's lock.json only (no return). The lock must match its
    sidecar, be the room harness's labelled stalling-world lock for this seed under the pinned spec and code
    manifest, and its 'none' actor must be the one this component scores against. Returns (target, record)."""
    path = ROOM_OUT / f"seed{int(rep)}" / "lock.json"
    lock = read_json(path)
    if lock is None:
        raise SystemExit(f"refused: no room lock at {path}")
    if not str(lock.get("label", "")).startswith("EXPLORATORY") or lock.get("kind") != "room-lock" \
            or lock.get("world") != "stall" or lock.get("rep") != int(rep):
        raise SystemExit(f"refused: {path} is not the room harness's stalling-world lock of seed {rep}")
    if lock.get("spec_sha256") != ROOM_PINNED["spec_sha256"] or \
            lock.get("code_manifest_sha256") != ROOM_PINNED["manifest_sha256"]:
        raise SystemExit(f"refused: {path} names another room spec or code manifest")
    if lock["actor_sha256"].get("none") != L.tree_sha256(none_params):
        raise SystemExit("refused: the room lock's 'none' actor is not check 2's saved gapped-critic actor")
    target = lock.get("target_strength")
    target = float(target) if target is not None and np.isfinite(target) and target > 0 else None
    return target, dict(lock_file=str(path), lock_sha256=file_sha256(path), target_strength=target,
                        rows_sha256=lock["checks"]["a"]["rows_sha256"], read="target_strength only")


# ------------------------------------------------------------------------------------------------ networks

class ClassifierNet(nn.Module):
    """The classifier g of the classifier ratio: the shape of BCA's scale network (Dense 64 - ReLU - Dense 64 - ReLU -
    Dense 1), restated here (calibration/ is not imported); its output is the logit, log w = log(g / (1 - g)). Inputs
    as BCA's scale network gets them: the normalised observation / (1 + 0.001) and the command."""

    hidden: int = CLS_HIDDEN

    @nn.compact
    def __call__(self, obs, action):
        x = jnp.concatenate([obs / (1.0 + 0.001), action], axis=-1)
        x = nn.relu(nn.Dense(self.hidden)(x))
        x = nn.relu(nn.Dense(self.hidden)(x))
        return nn.Dense(1)(x).squeeze(-1)


class SurfaceNet(nn.Module):
    """The distance surface f(s, a) ~ d10(s, a): Dense 128 - softplus - Dense 128 - softplus - Dense 1 - softplus, on
    the standardised state (13) and the standardised command (4). Smooth, so its gradient reaches the command."""

    hidden: int = SURF_HIDDEN

    @nn.compact
    def __call__(self, zs, za):
        x = jnp.concatenate([zs, za], axis=-1)
        x = nn.softplus(nn.Dense(self.hidden)(x))
        x = nn.softplus(nn.Dense(self.hidden)(x))
        return jax.nn.softplus(nn.Dense(1)(x).squeeze(-1))


CLS = ClassifierNet()
SURF = SurfaceNet()


def cls_init(rep):
    seed = int(L.rng_np(L.LearnSettings(), rep, TAG_CLS_INIT).integers(2 ** 31 - 1))
    return CLS.init(jax.random.PRNGKey(seed), jnp.zeros(TC.DX), jnp.zeros(TC.DA)), seed


def surf_init(rep):
    seed = int(L.rng_np(L.LearnSettings(), rep, TAG_SURF_INIT).integers(2 ** 31 - 1))
    return SURF.init(jax.random.PRNGKey(seed), jnp.zeros(TC.DX), jnp.zeros(TC.DA)), seed


# ------------------------------------------------------------------------------------------------ raw signals

def density_N(shared, rows, a):
    """N(s_rows, a) = 1/2 sum_k ((a_k - c_k(s)) / sigma_k)^2."""
    return 0.5 * jnp.sum(jnp.square((a - shared["c_tr"][rows]) / shared["sigma"]), axis=-1)


def surface_f(shared, rows, a):
    """f(s_rows, a): the fitted d10 surface at the standardised state and command."""
    zs = (shared["raw32"][rows] - shared["s_mu"]) / shared["s_sd"]
    za = (a - shared["a_mu"]) / shared["a_sd"]
    return SURF.apply(shared["surf"], zs, za)


def two_head_D(critic, shared, rows, a):
    """D(s_rows, a) = |Q1 - Q2| of the frozen critic."""
    q = L.CRITIC.apply(critic, shared["obs"][rows], a)
    return jnp.abs(q[..., 0] - q[..., 1])


def classifier_logit(net, shared, rows, a):
    return CLS.apply(net, shared["obs"][rows], a)


def hinge_W(shared, a):
    hi, lo = shared["hi"], shared["lo"]
    return H.HINGE_SCALE * (jnp.square(jnp.maximum(a - hi, 0.0)) + jnp.square(jnp.maximum(lo - a, 0.0))).sum(axis=-1)


def penalty(kind, critic, shared, arm, rows, pi):
    """k * W(s_rows, pi) per row (B,), the coefficients included (one slot per form of the kind)."""
    c = arm["coef"]
    if kind == "plain":
        return jnp.zeros(pi.shape[:-1], pi.dtype)
    if kind == "hinge":
        return c[0] * hinge_W(shared, pi)
    if kind == "density":
        n = density_N(shared, rows, pi)
        return c[0] * jnp.maximum(n - shared["N_rec"][rows], 0.0) + c[1] * jnp.maximum(n - shared["q95_N"], 0.0)
    if kind == "classifier":
        return c[0] * jax.nn.softplus(classifier_logit(arm["net"], shared, rows, pi) - arm["lstar"])
    if kind == "distance":
        f = surface_f(shared, rows, pi)
        return c[0] * jnp.maximum(f - shared["f_rec"][rows], 0.0) + c[1] * jnp.maximum(f - shared["q95_f"], 0.0)
    if kind == "two_head":
        d = two_head_D(critic, shared, rows, pi)
        return c[0] * jnp.maximum(d - shared["D_rec"][rows], 0.0) + c[1] * d
    raise ValueError(f"unknown kind {kind!r}")


# ------------------------------------------------------------------------------------------------ the actor step

def actor_step_s(args, kind, actor, critic_params, batch, rows, shared, arm):
    """The room harness's actor_step_w (learners.actor_step with the penalty slot, unweighted cloning term) with the
    stand-in penalty: -lambda * mean(q - kW) + pull * mean (pi - a)^2, lambda on q alone, pull = 1."""

    def _actor_loss_fn(params, critic_params):
        pi = L.ACTOR.apply(params, batch.obs)
        q = L.CRITIC.apply(critic_params, batch.obs, pi)[..., 0]
        lmbda = args.alpha / jax.lax.stop_gradient(jnp.abs(q).mean())
        kw = penalty(kind, critic_params, shared, arm, rows, pi)
        bc = jnp.square(pi - batch.action).mean() * arm["pull"]
        return (-lmbda * (q - kw).mean() + bc, (q.mean(), lmbda, bc, kw.mean()))

    grad_fn = jax.value_and_grad(_actor_loss_fn, has_aux=True)
    (a_loss, aux), a_grad = grad_fn(actor.params, critic_params)
    return actor.apply_gradients(grads=a_grad), (a_loss, *aux)


def _loop(args, kind, critic_params, actor, train, shared, rows, arm):
    def body(a, r):
        batch = jax.tree_util.tree_map(lambda x: x[r], train)
        a, aux = actor_step_s(args, kind, a, critic_params, batch, r, shared, arm)
        return a, jnp.stack(aux)

    return jax.lax.scan(body, actor, rows)


METRIC_NAMES = H.METRIC_NAMES
_PROGRAMS = {}


def fit_classifier(actor_params, shared):
    """One refit of the classifier with one actor (vmapped over arms by the runner): logistic loss, target pairs
    (s_i, clip(pi(s_i) + a_i - c(s_i), -1, 1)) labelled 1 against recorded pairs (s_i, a_i) labelled 0 at the same
    rows, CLS_STEPS Adam steps (lr CLS_LR) of CLS_BATCH rows each from a fixed row stream, from a fixed initialisation.
    Returns (params, l*, diagnostics)."""
    obs, act = shared["obs"], shared["act"]
    pi = L.ACTOR.apply(actor_params, obs)
    tgt = jnp.clip(pi + act - shared["c_tr"], -1.0, 1.0)
    tx = L.adam(CLS_LR)
    p0 = shared["cls_init"]

    def loss(p, r):
        lt = CLS.apply(p, obs[r], tgt[r])
        lr_ = CLS.apply(p, obs[r], act[r])
        return jnp.mean(jax.nn.softplus(-lt)) + jnp.mean(jax.nn.softplus(lr_))

    def body(carry, r):
        p, s = carry
        val, g = jax.value_and_grad(loss)(p, r)
        u, s = tx.update(g, s, p)
        return (jax.tree_util.tree_map(lambda x, y: x + y, p, u), s), val

    (p, _), ls = jax.lax.scan(body, (p0, tx.init(p0)), shared["cls_rows"])
    bank_logit = CLS.apply(p, shared["bank_obs"], shared["bank_act"])
    lstar = jnp.log(MISCOVERAGE / (1.0 - MISCOVERAGE)) + jax.scipy.special.logsumexp(bank_logit)
    probe = shared["cls_probe"]
    diag = jnp.stack([ls[:100].mean(), ls[-100:].mean(), lstar, bank_logit.mean(), bank_logit.max(),
                      CLS.apply(p, obs[probe], act[probe]).mean(), CLS.apply(p, obs[probe], tgt[probe]).mean(),
                      jnp.mean(jnp.abs(pi[probe]) > TC.KNEE)])
    return p, lstar, diag


CLS_DIAG_NAMES = ["loss_first100", "loss_last100", "l_star", "bank_logit_mean", "bank_logit_max",
                  "logit_recorded_probe_mean", "logit_target_probe_mean", "actor_past80_share_probe"]


@dataclass(frozen=True)
class StandinArm:
    """One arm. kind: one of KINDS; coef: the kind's coefficients (N_COEF[kind] slots: the stand-in's main form first,
    the plan's literal form second); family / knob: the matched family and its ladder knob."""

    name: str
    kind: str
    coef: tuple = (0.0,)
    family: str = None
    knob: float = None

    def record(self):
        return asdict(self)


class StandinRunner:
    """The room harness's RoomRunner (learners.ActorOnly with the penalty slot) for the stand-in kinds: one data seed's
    arms on one frozen critic, from one start actor and check 2's stream of batch rows (default_rng([20261015, rep,
    35])), the arms of each kind vmapped side by side; for the classifier kind the classifier of every arm is refitted
    with that arm's own actor before every chunk."""

    def __init__(self, prepared, start_params, critic, settings, rep, *, shared, alpha=ALPHA, steps=STEPS,
                 chunk=CHUNK):
        self.prepared, self.settings, self.rep = prepared, settings, int(rep)
        self.args = L.make_args(settings, rep, alpha)
        self.steps, self.chunk = int(steps), int(min(chunk, steps))
        if self.steps % self.chunk:
            raise ValueError("steps must be a multiple of the chunk")
        rows = L.rng_np(settings, rep, L.TAG_ACTOR_ONLY).integers(prepared.n, size=(STEPS, settings.batch_size))
        self.rows_sha256 = L.tree_sha256(rows)
        rows = rows[:self.steps]
        self.rows = jnp.asarray(rows.reshape(self.steps // self.chunk, self.chunk, settings.batch_size), jnp.int32)
        self.start = TrainState.create(apply_fn=L.ACTOR.apply, params=start_params, tx=L.adam(settings.lr))
        self.critic = critic
        self.shared = shared
        self.seconds, self.cpu_seconds, self.refit_cpu_seconds = {}, {}, 0.0

    def program(self, kind, n_arms):
        key = ("standin", L._args_key(self.args), kind, int(n_arms), self.chunk)
        if key not in _PROGRAMS:
            loop = lambda *a: _loop(self.args, kind, *a)  # noqa: E731
            _PROGRAMS[key] = jax.jit(jax.vmap(loop, in_axes=(None, 0, None, None, None, 0)))
        return _PROGRAMS[key]

    def refit_program(self, n_arms):
        key = ("refit", int(n_arms))
        if key not in _PROGRAMS:
            _PROGRAMS[key] = jax.jit(jax.vmap(fit_classifier, in_axes=(0, None)))
        return _PROGRAMS[key]

    def arm_tree(self, kind, arms):
        coef = np.zeros((len(arms), N_COEF[kind]), np.float32)
        for i, a in enumerate(arms):
            if len(a.coef) != N_COEF[kind]:
                raise ValueError(f"arm {a.name}: kind {kind} takes {N_COEF[kind]} coefficients")
            coef[i] = a.coef
        tree = dict(coef=jnp.asarray(coef), pull=jnp.ones(len(arms), jnp.float32))
        if kind == "classifier":
            tree["net"] = jax.tree_util.tree_map(lambda x: jnp.broadcast_to(x, (len(arms),) + x.shape),
                                                 self.shared["cls_init"])
            tree["lstar"] = jnp.zeros(len(arms), jnp.float32)
        return tree

    def run_kind(self, kind, arms, log_fn=None):
        n = len(arms)
        states = jax.tree_util.tree_map(lambda x: jnp.broadcast_to(x, (n,) + jnp.shape(x)), self.start)
        tree = self.arm_tree(kind, arms)
        program = self.program(kind, n)
        metrics, refits, finite_steps = [], [], np.ones(n, bool)
        t0, c0 = time.perf_counter(), time.process_time()
        for c in range(self.steps // self.chunk):
            if kind == "classifier":
                r0 = time.process_time()
                nets, lstar, diag = self.refit_program(n)(states.params, self.shared)
                tree = dict(tree, net=nets, lstar=lstar)
                diag = np.asarray(diag, np.float64)
                finite_steps &= np.all(np.isfinite(diag), axis=1)
                refits.append(diag.tolist())
                self.refit_cpu_seconds += time.process_time() - r0
            states, aux = program(self.critic, states, self.prepared.train, self.shared, self.rows[c], tree)
            aux = np.asarray(aux, np.float64)
            finite_steps &= np.all(np.isfinite(aux), axis=(1, 2))
            metrics.append(aux.mean(1).tolist())
            if log_fn:
                log_fn(f"  {kind} x{n}: step {(c + 1) * self.chunk}/{self.steps} ({time.perf_counter() - t0:.0f}s)")
        jax.block_until_ready(states.params)
        sec, cpu = time.perf_counter() - t0, time.process_time() - c0
        out = {}
        for i, a in enumerate(arms):
            p = jax.tree_util.tree_map(lambda x: x[i], states.params)
            out[a.name] = dict(params=p, finite=bool(finite_steps[i] and L._finite_tree(p)),
                               chunk_metrics=[m[i] for m in metrics], arm=a.record(), seconds=sec / n,
                               refits=[r[i] for r in refits] if refits else None)
        self.seconds[kind] = self.seconds.get(kind, 0.0) + sec
        self.cpu_seconds[kind] = self.cpu_seconds.get(kind, 0.0) + cpu
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


# ------------------------------------------------------------------------------------------------ data-side fits

def conformal_level(scores, level=LEVEL):
    """The split-conformal level: the ceil((n + 1) level / 100)-th smallest score (fallback diagnostic's rule)."""
    s = np.sort(np.asarray(scores, np.float64))
    r = math.ceil((len(s) + 1) * level / 100.0)
    return float(s[r - 1]) if r <= len(s) else math.inf


def held_out_rows(sd, rep):
    """5,000 held-out recorded rows: the calibration split's rows with the gap removed (the gapped log's removal rule)
    and the calibration-bank rows left out, sampled by default_rng([20261015, seed, 81])."""
    lg = sd.log
    cal = lg.split_rows("calibration")
    cal = cal[~sd.gap.contains(lg["actions"][cal])]
    held = np.setdiff1d(cal, lg.bank_rows)
    pick = np.sort(L.rng_np(L.LearnSettings(), rep, TAG_HELD).choice(len(held), min(HELD_ROWS, len(held)),
                                                                     replace=False))
    rows = held[pick]
    return rows, np.asarray(lg["observations"][rows], np.float32).astype(np.float64), \
        np.asarray(lg["actions"][rows], np.float32), int(len(held))


class KNN:
    """Exact 10th-nearest-neighbour distance to the training rows in standard units of (state, command): candidates
    by a float32 search (JAX top-k of 16), then their float64 distances; the 10th smallest is returned."""

    def __init__(self, states, actions):
        X = np.concatenate([np.asarray(states, np.float64), np.asarray(actions, np.float32).astype(np.float64)], 1)
        self.mu, self.sd = X.mean(0), X.std(0)
        self.Z = (X - self.mu) / self.sd
        Z32 = jnp.asarray(self.Z.astype(np.float32))
        z2 = jnp.sum(Z32 * Z32, 1)

        def cand(q):
            d2 = jnp.sum(q * q, 1)[:, None] + z2[None, :] - 2.0 * q @ Z32.T
            return jax.lax.top_k(-d2, KNN_CANDIDATES)[1]

        self._cand = jax.jit(cand)

    def d10(self, states, actions, chunk=1024):
        Xq = np.concatenate([np.asarray(states, np.float64), np.asarray(actions, np.float32).astype(np.float64)], 1)
        Zq = (Xq - self.mu) / self.sd
        out = np.empty(len(Zq))
        for i in range(0, len(Zq), chunk):
            q = Zq[i:i + chunk]
            pad = chunk - len(q)
            qq = np.concatenate([q, np.zeros((pad, q.shape[1]))]) if pad else q
            idx = np.asarray(self._cand(jnp.asarray(qq.astype(np.float32))))[:len(q)]
            dd = np.sqrt(np.sum((self.Z[idx] - q[:, None, :]) ** 2, -1))
            out[i:i + len(q)] = np.sort(dd, 1)[:, K_NN - 1]
        return out


def surface_queries(pg, a_clone, a_none, rep):
    """The fit set of the distance surface: SURF_TYPES x SURF_PER_TYPE queries (training state, command), the training
    row drawn uniformly; commands: (0) the recorded one, (1) the cloned actor's, (2) the no-caution actor's, (3)
    uniform in [-1, 1]^4, (4) the recorded one plus normal noise (sd 0.25), clipped, (5) the no-caution actor's with
    one random thruster set to a random throttle in [0.6, 1] on its own side. Then SURF_VALID validation queries drawn
    the same way. default_rng([20261015, seed, 83]). Returns (row index, command float32, type)."""
    rng = L.rng_np(L.LearnSettings(), rep, TAG_SURF_QUERIES)
    total = SURF_TYPES * SURF_PER_TYPE + SURF_VALID
    ri = rng.integers(pg.n, size=total)
    kind = np.concatenate([np.repeat(np.arange(SURF_TYPES), SURF_PER_TYPE), rng.integers(SURF_TYPES, size=SURF_VALID)])
    acts = np.asarray(pg.actions, np.float64)
    a = np.empty((total, TC.DA))
    a[kind == 0] = acts[ri[kind == 0]]
    a[kind == 1] = a_clone[ri[kind == 1]]
    a[kind == 2] = a_none[ri[kind == 2]]
    a[kind == 3] = rng.uniform(-1.0, 1.0, size=((kind == 3).sum(), TC.DA))
    a[kind == 4] = np.clip(acts[ri[kind == 4]] + rng.normal(0.0, SURF_NOISE, size=((kind == 4).sum(), TC.DA)), -1, 1)
    k5 = np.flatnonzero(kind == 5)
    a5 = a_none[ri[k5]].astype(np.float64).copy()
    j = rng.integers(TC.DA, size=len(k5))
    side = np.where(a5[np.arange(len(k5)), j] < 0, -1.0, 1.0)
    a5[np.arange(len(k5)), j] = side * rng.uniform(0.6, 1.0, size=len(k5))
    a[k5] = a5
    return ri, a.astype(np.float32), kind


def fit_surface(pg, knn, a_clone, a_none, rep, log_fn=None):
    """The distance surface: exact d10 at the query set, then SURF_STEPS Adam steps (cosine decay SURF_LR ->
    SURF_LR_END) of SURF_BATCH queries on the mean squared error, from a fixed initialisation. Returns (params, the
    standardisation constants, report)."""
    t0, c0 = time.perf_counter(), time.process_time()
    ri, a, kind = surface_queries(pg, a_clone, a_none, rep)
    y = knn.d10(pg.states[ri], a)
    t_knn = (time.perf_counter() - t0, time.process_time() - c0)
    s_mu, s_sd = knn.mu[:TC.DX].astype(np.float32), knn.sd[:TC.DX].astype(np.float32)
    a_mu, a_sd = knn.mu[TC.DX:].astype(np.float32), knn.sd[TC.DX:].astype(np.float32)
    zs = (np.asarray(pg.states[ri], np.float32) - s_mu) / s_sd
    za = (a - a_mu) / a_sd
    ntr = SURF_TYPES * SURF_PER_TYPE
    import optax
    sched = optax.cosine_decay_schedule(SURF_LR, SURF_STEPS, alpha=SURF_LR_END / SURF_LR)
    tx = optax.adam(sched)
    p0, init_seed = surf_init(rep)
    rr = jnp.asarray(L.rng_np(L.LearnSettings(), rep, TAG_SURF_ROWS).integers(ntr, size=(SURF_STEPS, SURF_BATCH)),
                     jnp.int32)
    Zs, Za, Y = jnp.asarray(zs[:ntr]), jnp.asarray(za[:ntr]), jnp.asarray(y[:ntr], jnp.float32)

    def loss(p, r):
        return jnp.mean(jnp.square(SURF.apply(p, Zs[r], Za[r]) - Y[r]))

    def body(carry, r):
        p, s = carry
        val, g = jax.value_and_grad(loss)(p, r)
        u, s = tx.update(g, s, p)
        return (optax.apply_updates(p, u), s), val

    (p, _), ls = jax.jit(lambda q: jax.lax.scan(body, (q, tx.init(q)), rr))(p0)
    ls = np.asarray(ls, np.float64)
    fv = np.asarray(SURF.apply(p, jnp.asarray(zs[ntr:]), jnp.asarray(za[ntr:])), np.float64)
    yv = y[ntr:]
    past = np.any(np.abs(a[ntr:]) > TC.KNEE, axis=1)
    err = fv - yv
    report = dict(queries=int(len(y)), fit_queries=int(ntr), validation_queries=int(len(yv)), init_seed=init_seed,
                  query_types=["recorded", "cloned actor", "no-caution actor", "uniform", "recorded + noise 0.25",
                               "no-caution actor with one thruster at 0.6-1.0"],
                  d10_quantiles={q: float(np.quantile(y, q / 100)) for q in (1, 50, 99)},
                  train_mse_first100=float(ls[:100].mean()), train_mse_last500=float(ls[-500:].mean()),
                  validation=dict(rmse=float(np.sqrt(np.mean(err ** 2))), r2=float(1 - np.mean(err ** 2) / np.var(yv)),
                                  median_abs=float(np.median(np.abs(err))), bias=float(err.mean())),
                  validation_past80=dict(n=int(past.sum()), rmse=float(np.sqrt(np.mean(err[past] ** 2))),
                                         bias=float(err[past].mean()), median_abs=float(np.median(np.abs(err[past])))),
                  knn_seconds=t_knn[0], knn_cpu_seconds=t_knn[1],
                  seconds=time.perf_counter() - t0, cpu_seconds=time.process_time() - c0)
    if log_fn:
        log_fn(f"distance surface: validation rmse {report['validation']['rmse']:.4f}, R2 {report['validation']['r2']:.4f}"
               f", past-80% rmse {report['validation_past80']['rmse']:.4f} ({report['cpu_seconds']:.0f} cpu s)")
    return p, dict(s_mu=s_mu, s_sd=s_sd, a_mu=a_mu, a_sd=a_sd), report


_ROWFN = {}


def rows_eval(name, fn, *args, rows_idx, actions, chunk=8192):
    """fn(*args, rows, actions) (jitted, cached by name) over row chunks; float64 out."""
    if name not in _ROWFN:
        _ROWFN[name] = jax.jit(fn)
    out = []
    for i in range(0, len(rows_idx), chunk):
        out.append(np.asarray(_ROWFN[name](*args, jnp.asarray(rows_idx[i:i + chunk], jnp.int32),
                                           jnp.asarray(np.asarray(actions[i:i + chunk], np.float32))), np.float64))
    return np.concatenate(out) if out else np.zeros((0,))


def build_shared(ctx, world, pg, critic, clone, none, rep, log_fn=None):
    """Everything the stand-in penalties read, for one world, fitted on that world's gapped log only (no return):
    the recorded range, the density's sigma and q95_N, the distance surface and q95_d, the per-row values at the
    recorded commands (N, f, D), the classifier's row stream, initialisation and bank. Returns (shared, info)."""
    t0, c0 = time.perf_counter(), time.process_time()
    sd = ctx.sd[world]
    obs = np.asarray(pg.train.obs)
    raw32 = np.asarray(pg.states, np.float32)
    if not np.array_equal(raw32.astype(np.float64), pg.states):
        raise ValueError("training states are not exactly float32")
    acts = np.asarray(pg.actions, np.float32)
    a_clone = L.actor_rows(clone, obs)
    a_none = L.actor_rows(none, obs)
    lo, hi = H.hinge_bounds(pg.actions)
    sigma = np.sqrt(np.mean((acts.astype(np.float64) - a_clone.astype(np.float64)) ** 2, axis=0))
    hrows, hstates, hacts, n_held = held_out_rows(sd, rep)
    hobs = pg.normalize(hstates)
    h_clone = L.actor_rows(clone, hobs).astype(np.float64)
    N_held = 0.5 * np.sum(((hacts.astype(np.float64) - h_clone) / sigma) ** 2, axis=1)
    q95_N = conformal_level(N_held)
    knn = KNN(pg.states, acts)
    d_held = knn.d10(hstates, hacts)
    q95_d = conformal_level(d_held)
    surf, std, srep = fit_surface(pg, knn, a_clone, a_none, rep, log_fn)
    cinit, cseed = cls_init(rep)
    cls_rows = L.rng_np(L.LearnSettings(), rep, TAG_CLS_ROWS).integers(pg.n, size=(CLS_STEPS, CLS_BATCH))
    probe = np.sort(L.rng_np(L.LearnSettings(), rep, TAG_CLS_ROWS, 1).choice(pg.n, min(4096, pg.n), replace=False))
    shared = dict(obs=pg.train.obs, act=jnp.asarray(acts), raw32=jnp.asarray(raw32), lo=jnp.asarray(lo, jnp.float32),
                  hi=jnp.asarray(hi, jnp.float32), c_tr=jnp.asarray(a_clone), sigma=jnp.asarray(sigma, jnp.float32),
                  q95_N=jnp.asarray(q95_N, jnp.float32), surf=surf,
                  **{k: jnp.asarray(v) for k, v in std.items()}, q95_f=jnp.asarray(q95_d, jnp.float32),
                  cls_init=cinit, cls_rows=jnp.asarray(cls_rows, jnp.int32), cls_probe=jnp.asarray(probe, jnp.int32),
                  bank_obs=pg.bank.obs, bank_act=pg.bank.action,
                  N_rec=jnp.zeros(pg.n, jnp.float32), f_rec=jnp.zeros(pg.n, jnp.float32),
                  D_rec=jnp.zeros(pg.n, jnp.float32))
    idx = np.arange(pg.n)
    N_rec = rows_eval("N", density_N, shared, rows_idx=idx, actions=acts).astype(np.float32)
    f_rec = rows_eval("f", surface_f, shared, rows_idx=idx, actions=acts).astype(np.float32)
    D_rec = rows_eval("D", two_head_D, critic, shared, rows_idx=idx, actions=acts).astype(np.float32)
    shared.update(N_rec=jnp.asarray(N_rec), f_rec=jnp.asarray(f_rec), D_rec=jnp.asarray(D_rec))
    # surface against the exact distance at the held-out rows (not in the fit set) and at the training rows
    f_held = np.asarray(SURF.apply(surf, jnp.asarray((np.asarray(hstates, np.float32) - std["s_mu"]) / std["s_sd"]),
                                   jnp.asarray((hacts - std["a_mu"]) / std["a_sd"])), np.float64)
    info = dict(world=world, n_train_rows=int(pg.n), recorded_range=dict(lo=lo.tolist(), hi=hi.tolist()),
                density=dict(sigma=sigma.tolist(), q95_N=q95_N, N_held_quantiles={q: float(np.quantile(N_held, q / 100))
                                                                                 for q in (50, 90, 95, 99)},
                             N_recorded_train_mean=float(N_rec.mean()), chi2_4_half_95=4.7438646),
                held_out=dict(rows=int(len(hrows)), pool=n_held, rows_sha256=L.tree_sha256(hrows),
                              rule="calibration split, gap removed, bank rows left out; default_rng([20261015, seed, 81])"),
                distance=dict(q95_d=q95_d, d10_held_quantiles={q: float(np.quantile(d_held, q / 100))
                                                               for q in (50, 90, 95, 99)},
                              surface=srep,
                              surface_vs_exact_held_out=dict(rmse=float(np.sqrt(np.mean((f_held - d_held) ** 2))),
                                                             bias=float(np.mean(f_held - d_held)),
                                                             q95_of_surface=conformal_level(f_held)),
                              f_recorded_train_mean=float(f_rec.mean())),
                two_head=dict(D_recorded_train_mean=float(D_rec.mean()),
                              D_recorded_train_quantiles={q: float(np.quantile(D_rec, q / 100)) for q in (50, 95, 99)}),
                classifier=dict(init_seed=cseed, steps=CLS_STEPS, batch=CLS_BATCH, lr=CLS_LR, bank_rows=int(len(pg.bank_rows)),
                                rows_sha256=L.tree_sha256(cls_rows)),
                seconds=time.perf_counter() - t0, cpu_seconds=time.process_time() - c0)
    if log_fn:
        log_fn(f"{world} shared arrays built ({info['cpu_seconds']:.0f} cpu s)")
    return shared, info, dict(a_clone=a_clone, a_none=a_none)


def classifier_fit_for(actor_params, shared):
    """A single classifier fit with a given actor (check (e), the slope scale): (params, l*, diag)."""
    key = ("refit1",)
    if key not in _PROGRAMS:
        _PROGRAMS[key] = jax.jit(fit_classifier)
    p, lstar, diag = _PROGRAMS[key](actor_params, shared)
    return p, float(lstar), dict(zip(CLS_DIAG_NAMES, np.asarray(diag, np.float64).tolist()))


# ------------------------------------------------------------------------------------------------ per-form readouts

def family_arm(family, knob, scale, tag=None):
    """The arm of a stand-in family at a ladder knob; scale: {family: coefficient per knob}."""
    kind, slot = FAMILY_KIND[family]
    coef = [0.0] * N_COEF[kind]
    coef[slot] = float(knob) * float(scale[family])
    name = f"{family}@{knob:.10g}" if tag is None else f"{family}@{tag}"
    return StandinArm(name, kind, tuple(coef), family=family, knob=float(knob))


def locked_arm(family, coef, name=None):
    kind, _ = FAMILY_KIND[family]
    return StandinArm(name or family, kind, tuple(float(c) for c in coef), family=family)


def form_arm(family, net=None, lstar=None):
    """The arm dict (coefficient 1 on the family's form) for readouts outside training."""
    kind, slot = FAMILY_KIND[family]
    coef = np.zeros(N_COEF[kind], np.float32)
    coef[slot] = 1.0
    arm = dict(coef=jnp.asarray(coef), pull=jnp.asarray(1.0, jnp.float32))
    if kind == "classifier":
        arm["net"], arm["lstar"] = net, jnp.asarray(lstar, jnp.float32)
    return kind, arm


def W_rows(family, critic, shared, rows_idx, actions, net=None, lstar=None):
    kind, arm = form_arm(family, net, lstar)
    return rows_eval(("W", kind), lambda cr, sh, ar, r, a: penalty(kind, cr, sh, ar, r, a), critic, shared, arm,
                     rows_idx=rows_idx, actions=actions)


def W_grad_rows(family, critic, shared, rows_idx, actions, net=None, lstar=None):
    """dW/da per row (n, 4) of the in-program penalty (coefficient 1)."""
    kind, arm = form_arm(family, net, lstar)

    def g(cr, sh, ar, r, a):
        return jax.grad(lambda aa: penalty(kind, cr, sh, ar, r, aa).sum())(a)

    name = ("dW", kind)
    if name not in _ROWFN:
        _ROWFN[name] = jax.jit(g)
    out = []
    for i in range(0, len(rows_idx), 8192):
        out.append(np.asarray(_ROWFN[name](critic, shared, arm, jnp.asarray(rows_idx[i:i + 8192], jnp.int32),
                                           jnp.asarray(np.asarray(actions[i:i + 8192], np.float32))), np.float64))
    return np.concatenate(out)


def slope_scales(pg, critic, shared, room_shared, a_none, rep, cls_none, cfg):
    """Ladder scale (spec 'ladder'): kappa_X = g_band / g_X with g_X = sqrt(mean over the room's 4,096 slope rows
    (default_rng([20261015, seed, 63])) of ||grad_a W_X(s, a)||^2) at a = the no-caution actor's command, and g_band
    the same for the room's main oracle [delta*(s, a_i) - delta*(s, a)]_+ (harness.slope_rms, unchanged) at the same
    commands (cfg: the room's program configuration whose d_logged room_shared holds). The classifier ratio is read
    with the classifier fitted with the no-caution actor. Outcome-free."""
    idx = H.slope_sample(pg, rep)
    a = np.asarray(a_none)[idx]
    g_band = H.slope_rms("band", cfg, critic, room_shared, idx, a)
    g, scale = {}, {}
    for f in FAMILIES:
        net, lstar = (cls_none[0], cls_none[1]) if f == "classifier_ratio" else (None, None)
        gr = W_grad_rows(f, critic, shared, idx, a, net, lstar)
        g[f] = float(np.sqrt(np.mean(np.sum(gr ** 2, axis=1))))
        scale[f] = g_band / g[f] if np.isfinite(g[f]) and g[f] > 0 else 1.0
    return dict(g_band=g_band, g=g, scale=scale, rows=int(len(idx)),
                fallback={f: bool(not (np.isfinite(g[f]) and g[f] > 0)) for f in FAMILIES})


def check_e(family, pg, critic, shared, none, rep, scale, net=None, lstar=None):
    """Check (e) on one stand-in penalty, the room harness's rule (harness.oracle_shape) restated for any penalty: on
    the room's 2,000 full-throttle-region training states (default_rng([20261015, seed, 65])) thruster 1 of the
    no-caution actor's command is set along the actor's own push to each throttle of the room's grid (the other
    thrusters stay at the actor's command). (e1) W falls from 80% to 81% on at most 25% of the states, (e2) the mean
    of W rises from 80% to 81%, (e3) the in-program penalty's mean slope along the push at 80.5% is positive and
    every slope is finite, (e4) the best grid throttle for Q1 - k W lies past 80% on fewer states than for Q1 alone,
    with k the coefficient of ladder knob 1 (also reported at every rung). Frozen surfaces only; no return."""
    t0 = time.perf_counter()
    rows = H.shape_rows(pg, rep)
    obs = np.asarray(pg.train.obs)[rows]
    base = L.actor_rows(none, obs).astype(np.float64)
    sign = np.where(base[:, 0] < 0, -1.0, 1.0)

    def at(m):
        aa = base.copy()
        aa[:, 0] = sign * (TC.KNEE if m == 0.80 else m)
        return aa

    W, Q1 = {}, {}
    for m in H.SHAPE_GRID:
        aa = at(m)
        W[m] = W_rows(family, critic, shared, rows, aa, net, lstar)
        Q1[m] = np.asarray(L.critic_rows(critic, obs, aa.astype(np.float32))[0][:, 0], np.float64)
    grid = np.array(H.SHAPE_GRID)
    q_best = grid[np.argmax(np.stack([Q1[m] for m in H.SHAPE_GRID], 1), 1)]
    q1_share = float(np.mean(q_best > 0.8))

    def best_share(k):
        best = grid[np.argmax(np.stack([Q1[m] - k * W[m] for m in H.SHAPE_GRID], 1), 1)]
        return float(np.mean(best > 0.8))

    grads = {}
    for m in H.SHAPE_GRAD_POINTS:
        g = W_grad_rows(family, critic, shared, rows, at(m), net, lstar)[:, 0] * sign
        grads[f"{m:g}"] = dict(mean=float(g.mean()), negative_share=float(np.mean(g < 0)),
                               finite_share=float(np.isfinite(g).mean()))
    fall, rise = float(np.mean(W[0.81] < W[0.80])), float(np.mean(W[0.81] > W[0.80]))
    k1 = float(scale[family])
    rec = dict(family=family, formula=FORMULAS[family], rows=int(len(rows)),
               actor_thruster1_past80_share=float(np.mean(np.abs(base[:, 0]) > TC.KNEE)),
               q1_alone_best_past80_share=q1_share, mean_W={f"{m:g}": float(W[m].mean()) for m in H.SHAPE_GRID},
               active_share={f"{m:g}": float(np.mean(W[m] > 0)) for m in (0.7, 0.8, 0.81, 0.9, 1.0)},
               fall_share_80_to_81=fall, rise_share_80_to_81=rise, unchanged_share_80_to_81=1.0 - fall - rise,
               mean_W_full_minus_80=float(np.mean(W[1.0] - W[0.80])), mean_dW_dpush=grads,
               k_knob1=k1, best_past80_share_knob1=best_share(k1),
               best_past80_share_by_knob={f"{kn:g}": best_share(kn * k1) for kn in LADDER})
    rec["e1_fall_share_ok"] = bool(fall <= H.SHAPE_MAX_FALL_SHARE)
    rec["e2_mean_rises"] = bool(rec["mean_W"]["0.81"] > rec["mean_W"]["0.8"])
    rec["e3_slope_past_knee_positive"] = bool(grads["0.805"]["mean"] > 0 and
                                              all(v["finite_share"] == 1.0 for v in grads.values()))
    rec["e4_fewer_best_past80_than_q1"] = bool(rec["best_past80_share_knob1"] < q1_share)
    rec["passed"] = bool(rec["e1_fall_share_ok"] and rec["e2_mean_rises"] and rec["e3_slope_past_knee_positive"]
                         and rec["e4_fewer_best_past80_than_q1"])
    rec["all_W_finite"] = bool(all(np.all(np.isfinite(W[m])) for m in H.SHAPE_GRID))
    rec["seconds"] = time.perf_counter() - t0
    return rec


def finite_check(family, pg, critic, shared, a_rows, net=None, lstar=None):
    """Each penalty is finite and nonnegative on every training row at the recorded, cloned, no-caution and
    uniform-random commands; its gradient is finite there."""
    idx = np.arange(pg.n)
    out = {}
    for name, a in a_rows.items():
        w = W_rows(family, critic, shared, idx, a, net, lstar)
        g = W_grad_rows(family, critic, shared, idx, a, net, lstar)
        out[name] = dict(finite=bool(np.all(np.isfinite(w))), nonnegative=bool(np.all(w >= 0)),
                         grad_finite=bool(np.all(np.isfinite(g))), mean=float(np.mean(w)),
                         share_positive=float(np.mean(w > 0)), max=float(np.max(w)))
    out["passed"] = bool(all(v["finite"] and v["nonnegative"] and v["grad_finite"] for v in out.values()
                             if isinstance(v, dict)))
    return out


# ------------------------------------------------------------------------------------------------ matching

def match_standins(runner, trained, none, prepared, target, scale, families, log_fn=log):
    """Blind matching of every stand-in family to the perfect band's strength: the 4-rung ladder (knob 0 = no
    caution, S = 0, leads it), the declared extension (x sqrt(10), at most 3 rungs), then the gate's matcher
    experiments/signal/glq_match.match_parallel (tol 0.01, 5 rounds). The room's run_seed.match_gating, restated for
    these families. Strengths only."""
    by_knob = {f: {} for f in families}
    for name, v in trained.items():
        f = v["arm"].get("family")
        if f in by_knob and v["arm"].get("knob") is not None:
            by_knob[f][float(v["arm"]["knob"])] = name

    def S_of(name):
        return float(L.strength(trained[name]["params"], none, prepared.eval_obs)) if trained[name]["finite"] \
            else float("nan")

    extensions = {f: [] for f in families}
    for _ in range(MAX_EXTENSIONS):
        need = []
        for f in families:
            knobs = sorted(by_knob[f])
            fin = [s for s in (S_of(by_knob[f][k]) for k in knobs) if np.isfinite(s)]
            if fin and target > max(fin):
                need.append(family_arm(f, knobs[-1] * EXTEND_FACTOR, scale))
        if not need:
            break
        log_fn(f"ladder extension: {[a.name for a in need]}")
        trained.update(runner.run(need, log_fn))
        for a in need:
            by_knob[a.family][a.knob] = a.name
            extensions[a.family].append(a.knob)
    ladders = {f: ([0.0] + sorted(by_knob[f]), [0.0] + [S_of(by_knob[f][k]) for k in sorted(by_knob[f])])
               for f in families}

    def evaluate(proposals):
        arms = [family_arm(f, float(k), scale) for f, k in proposals if float(k) not in by_knob[f]]
        if arms:
            trained.update(runner.run(arms, log_fn))
            for a in arms:
                by_knob[a.family][a.knob] = a.name
        return [S_of(by_knob[f][float(k)]) for f, k in proposals]

    states, rounds = M.match_parallel(ladders, {"perfect_band": float(target)}, evaluate, M.MATCH_TOL,
                                      M.MATCH_ROUNDS, log=log_fn)
    out = {}
    for f in families:
        rec = M.public(states[(f, "perfect_band")])
        rec.update(rounds=rounds, extensions=extensions[f], glq_match_sha256=M.source_sha256())
        name = None if rec["knob"] is None else by_knob[f].get(float(rec["knob"]))
        strongest = None
        fin = [(S_of(n), n) for n in by_knob[f].values() if trained[n]["finite"]]
        if fin:
            strongest = max(fin)[1]
        out[f] = dict(record=rec, arm=name, matched=name is not None, strongest_finite_rung=strongest)
    return out


# ------------------------------------------------------------------------------------------------ evaluation

def thruster_shares(params, prepared, physics):
    """Reported: the share of steps past 80% per thruster, from one exact rollout per situation's full bank."""
    pol = L.mlp_policy(params, prepared)
    out = {}
    for s in TC.SITUATIONS:
        run = H.TR.rollout(pol, TC.bank_starts(s, TC.BANK_SIZE), physics, record=True)
        out[s] = (np.abs(run.commands) > physics.knee).mean(axis=(0, 1)).tolist()
    return out


def max_param_diff(a, b):
    return H.max_param_diff(a, b)


# ------------------------------------------------------------------------------------------------ the spec

FORM_CHOICES = [
    dict(id="F1 behaviour density read from the row's recorded command",
         family="density", plan_form="[1/2 sum_k ((a_k - c_k(s)) / sigma_k)^2 - q95]_+", main_form=FORMULAS["density"],
         plan_form_runs_as="density_plan (reported; the same matching, evaluation and marks, labelled reported)",
         reason=("Meaning. N(s, a) is how unusual the command is for the recorder at this state, in units of the "
                 "recorder's spread. The plan's threshold q95 is one number for the whole log, so the penalty asks "
                 "'is this command less usual than 95% of recorded commands anywhere'. Measuring from the row's own "
                 "recorded command asks 'is this command less usual for the recorder at this state than the command "
                 "the recorder actually gave here', which is change A1's reading (W = 0 at a = a_i, one-sided, it "
                 "removes exactly the extra unusualness and never rewards a command for being usual). Why the global "
                 "level fails here: the recorder's spread is wide and heavy-tailed across the log (sigma 0.21-0.23 per "
                 "thruster on seed 0; q95 of N on held-out recorded rows 7.46 against 4.74 for a Gaussian), so on "
                 "seed 0 the plan's form is positive at only 1.1% of the 2,000 region states with thruster 1 at 80% "
                 "and 2.4% at full throttle; the read-from-the-recorded-command form is positive at 56% and 86%, "
                 "falls from 80% to 81% at 0.1% of them and rises on 58%. Measured on frozen surfaces before any "
                 "stand-in training (scratchpad/ood/standins/explore/shapes.py); no return, points or strength.")),
    dict(id="F2 distance to data read from the row's recorded command",
         family="distance", plan_form="[d10(s, a) - q95]_+ (fitted as a surface)", main_form=FORMULAS["distance"],
         plan_form_runs_as="distance_plan (reported)",
         reason=("Meaning. d10 measures how far the (state, command) pair is from the data; at training states its "
                 "level is set mostly by where the state is, not by the command (the fallback diagnostic found the "
                 "same: swapping a harmful command for its clipped version moved the median distance from 4.26 to 4.06 "
                 "standard units against a threshold of 4.30). That state-set level is an offset unrelated to the "
                 "command, the reason change A1 measured the oracle from the recorded command. [f(s, a) - f(s, a_i)]_+ "
                 "keeps what the command adds to the distance at this state. On seed 0 (exact kNN, 2,000 region "
                 "states): the plan's form is positive at 6.9% of them at 80% and 8.0% at full throttle (mean d10 "
                 "2.73 and 2.98 against q95 4.27); the chosen form is positive at 63% and 95%, falls from 80% to 81% "
                 "at 0.4% and rises on 65% (on the fitted surface: falls at 0%, rises on 66%). Frozen surfaces only.")),
    dict(id="F3 two-head disagreement read from the row's recorded command",
         family="two_head", plan_form="|Q1 - Q2|(s, a)", main_form=FORMULAS["two_head"],
         plan_form_runs_as="two_head_plan (reported)",
         reason=("Meaning. |Q1 - Q2| at the recorded command is an offset set by the state and by the two heads' "
                 "random initialisation, not by the command, and as an absolute value of a signed gap it can fall as "
                 "the command moves (as the literal |delta*| did). [D(s, a) - D(s, a_i)]_+ keeps the extra "
                 "disagreement the command causes at this state. On seed 0 the literal form falls from 80% to 81% "
                 "on 43.9% of the region states (check (e1) allows 25%), the chosen form on 23.6%. The signal is "
                 "weak either way (mean D 0.107 at 70% throttle, 0.123 at full). Frozen surfaces only.")),
    dict(id="F4 range hinge and classifier ratio kept in the plan's form",
         family="range_hinge, classifier_ratio",
         reason=("The range hinge is the room's arm, reused unchanged: it is zero inside the recorded range, so it "
                 "already reads relative to the recorded commands (and is zero at 99% of them). The classifier ratio "
                 "is a density ratio of the actor's pairs against the recorded pairs at the same states, with l* set "
                 "by the bank's recorded pairs; at recorded commands log w is about 0, far below l* (about 4.3-4.5 "
                 "on seed 0), so softplus(log w - l*) is already near 0 there (0.01) and re-referencing would change "
                 "almost nothing while adding a kink.")),
]

DEVIATIONS = [
    ("Stand-in forms (F1-F3): density, distance to data and two-head disagreement are read from the row's recorded "
     "command, [x(s, a) - x(s, a_i)]_+, following change A1 of the room spec; the plan's literal forms run beside them "
     "as reported arms (density_plan, distance_plan, two_head_plan) with the same matching, evaluation and marks, "
     "labelled reported. The choice was made from frozen surfaces only, before any stand-in training. NEEDS THE "
     "USER'S APPROVAL before launch: it changes the plan's stand-in formulas. Nothing in the code enforces this."),
    ("Range hinge: matched in strength like every stand-in (plan 4.2), with the room's definition unchanged; the room "
     "test ran it unmatched at k = 1 as a reported arm."),
    ("Classifier ratio. The classifier g has the shape of BCA's scale network (64-64 ReLU, restated; calibration/ is "
     "not imported) with the logit as output, so log w = logit. Target pairs (s_i, clip(pi(s_i) + a_i - c(s_i), -1, "
     "1)) against recorded pairs (s_i, a_i) at the same training rows, balanced, logistic loss, 2,000 Adam steps (lr "
     "1e-3) of 512 rows from a fixed row stream (default_rng([20261015, seed, 82])) and a fixed initialisation "
     "(tag 84), refitted before every 4,000-step chunk (steps 0, 4,000, ..., 16,000) with the arm's own actor; the "
     "first fit uses the start actor (the cloned actor), so its target pairs equal the recorded pairs and the "
     "penalty starts near zero. l* is computed directly from the bank weights, without BCA's code, as the plain "
     "weighted split-conformal limit at miscoverage 0.1: the threshold becomes +infinity when the test weight "
     "exceeds 0.1 / 0.9 times the bank's weight sum, so l* = log(0.1 / 0.9) + log sum_bank w(s_j, a_j); BCA's "
     "credible (Bayesian) layer is not restated. The bank is the gapped log's calibration bank (prepared.bank, the "
     "rows the gate's loader regenerates). U_max, BCA's largest finite width, does not exist without BCA; the "
     "matched coefficient takes its place."),
    ("Distance surface (the plan says only 'fitted as a surface'). Standard units: per-dimension mean and SD of "
     "the training rows' (state, command), 13 + 4 dimensions, as the fallback diagnostic. Exact d10 (float32 "
     "candidate search, float64 distances) at 120,000 fit queries and 20,000 validation queries: training states "
     "drawn uniformly with commands of six kinds (recorded; cloned actor; no-caution actor; uniform; recorded + "
     "normal noise 0.25; no-caution actor with one thruster at 0.6-1.0 on its own side), default_rng([20261015, "
     "seed, 83]). Surface: MLP 17-128-128-1, softplus activations and output; 12,000 Adam steps of 1,024 queries, "
     "cosine decay 1e-3 -> 5e-5, fixed initialisation (tag 85) and row stream (tag 86). Fit error is reported "
     "overall, at past-80% queries and at held-out recorded rows. q95_d of the plan's form is the exact d10's "
     "level on the held-out rows; the surface's own level is reported."),
    ("Held-out recorded rows (plan: 'held-out recorded rows'): 5,000 rows of the calibration split with the gap "
     "removed and the calibration-bank rows left out, default_rng([20261015, seed, 81]); q95 is the split-conformal "
     "level ceil((n + 1) 0.95) / n, as the fallback diagnostic."),
    ("Ladder (plan: '4-rung ladder, then refinement', rungs not given): knobs 0.3, 1, 3, 10, slope-normalised: the "
     "coefficient of knob 1 is g_band / g_X, the RMS command slope of the room's main oracle over that of the "
     "stand-in, both at the no-caution actor's commands on the room's 4,096 slope rows. The room normalised at the "
     "cloned actor's commands, where most stand-ins here are exactly zero (they are zero at or near the recorded "
     "command), so that point gives no scale. The room's blind extension rule (x sqrt(10), at most 3 rungs) and the "
     "gate's matcher (glq_match.match_parallel, tol 0.01, at most 5 rounds, knob 0 = no caution leading the ladder) "
     "are used unchanged; the chosen arm is the evaluated knob with the best |S/S* - 1|."),
    ("Check (e) is run on every penalty before any training and reported; it is not a stop condition for the "
     "stand-ins. Reason: these are rivals that set the bar BCA must meet; dropping one that does not rise past the "
     "knee would lower that bar, and whether a data-only penalty rises past 80% is itself what the check asks. (e4) "
     "needs a coefficient: the one of ladder knob 1 (reported at every rung too). The classifier ratio is read with "
     "its classifier fitted with the no-caution actor (and, reported, with the cloned actor)."),
    ("Marks wording made exact, as in the room spec: each condition is on the pooled mean over seeds 0-4 of the "
     "paired per-seed difference; every seed is needed (a missing value or an unmatched stand-in fails the mark). "
     "The perfect band, the matched even pull and the room's no-caution points come from the room's verified "
     "values.json, read only in the marks phase; the stand-ins' no-caution points come from this component's own "
     "evaluation of the same saved actor, and the difference from the room's is reported (it should be 0)."),
    ("A stand-in whose target strength cannot be reached (no bracket after the extensions) has no matched arm and "
     "fails every mark, as in the room; its strongest finite rung is evaluated and reported, labelled unmatched."),
    ("No-stall world: every stand-in runs at its stalling-world locked coefficient (absolute, not re-matched), with "
     "its data side refitted on the no-stall world's own gapped log (range, sigma, q95, distance surface, classifier "
     "refits with the no-stall arm's own actor, the no-stall critic's two heads); critic: check 2's no-stall gapped "
     "critic; start: the no-stall cloned actor; check (b) against check 2's no-stall gapped actor (stop on failure)."),
    ("The no-stall loss is read in strong current, the situation of the stalling-world gain, and reported in every "
     "situation; selectivity = stalling-world gain minus no-stall loss (both strong current), compared with the "
     "matched range hinge by the paired per-seed difference."),
]


def spec_payload():
    """Everything fixed before any stand-in run (written to SPEC_PATH by write_spec.py)."""
    stand = {f: dict(W=FORMULAS[f], role=("stand-in (mark)" if f in MAIN_FAMILIES else
                                          "the plan's literal form (reported; same mark computed, labelled reported)"),
                     kind=FAMILY_KIND[f][0], coefficient_slot=FAMILY_KIND[f][1]) for f in FAMILIES}
    return dict(
        label=LABEL, kind="standins-spec", version=1,
        plan=dict(path=str(PLAN_PATH), sha256=file_sha256(PLAN_PATH), sections=["4.0", "4.2"]),
        no_bca=("No BCA band, calibrator, threshold, dose, BCA update or BCA training arm. Nothing from "
                "algorithms/td3_bc_bca.py or learners.bca_* is called. From calibration/ only "
                "calibration/bank.stratified_bank runs, inside the gate's data loader. The classifier and the "
                "distance surface are new data-only networks; the classifier's l* is a closed-form count over the "
                "bank's classifier weights, not BCA's threshold rule."),
        room=dict(manifest_sha256=ROOM_PINNED["manifest_sha256"], spec_sha256=ROOM_PINNED["spec_sha256"],
                  harness_sha256=ROOM_PINNED["harness_sha256"],
                  imported=("runs/shift_explore/room/code/harness.py, executed from the bytes whose SHA-256 the room's "
                            "code manifest records (the manifest checked against its sidecar and the pinned SHA-256; "
                            "the room's source file checked too); refuse on any mismatch"),
                  reused=("SeedContext (the gate's loaders), hinge_bounds and HINGE_SCALE (the range hinge), "
                          "build_shared and slope_rms (the room's main-oracle slope for the ladder scale), "
                          "shape_rows, SHAPE_GRID, SHAPE_GRAD_POINTS, SHAPE_MAX_FALL_SHARE (check (e)), region_strength, "
                          "max_param_diff, gate_file_hashes; RoomRunner only in the seed-0 check (d)"),
                  read=("train: runs/shift_explore/room/pooled.json's pass flags (the run condition) and "
                        "seed<k>/lock.json's target_strength (the perfect band's strength), its check-(a) row-stream "
                        "SHA-256 and its 'none' actor SHA-256; marks: seed<k>/values.json's points of perfect_band, "
                        "the matched even pull and none. Nothing else of the room's outputs is read.")),
        run_condition=("plan 4.2: the stand-ins run only if the room test's room, recovery and cost (keeps-improvement) "
                       "marks pass: every train phase refuses unless runs/shift_explore/room/pooled.json matches its "
                       "sidecar, is the room harness's labelled pooled look for the pinned spec, is complete and its "
                       "room, recovery and keeps_improvement marks passed"),
        seeds=list(SEEDS), variant="base",
        setup=dict(protocol="the room test's (plan 4.0): check 2's actor-only protocol, unchanged",
                   critic="check 2's frozen fixed-controller gapped critic of the stalling world (no-stall twin)",
                   start_actor="the saved cloned actor of the world", alpha=ALPHA, actor_steps=STEPS, chunk=CHUNK,
                   batch_size=256, optimiser="Adam (runtime.networks.torch_adam), learning rate 3e-4",
                   batch_rows="check 2's stream default_rng([20261015, seed, 35]).integers(n, size=(20000, 256))",
                   arms_side_by_side="the arms of one kind run vmapped in one program"),
        actor_loss=dict(formula="-lambda * mean_i[Q1(s_i, pi(s_i)) - k * W(s_i, pi(s_i))] + mean_ij (pi(s_i)_j - a_ij)^2",
                        lam="lambda = alpha / stopgrad(mean_i |Q1(s_i, pi(s_i))|), on Q1 alone",
                        frozen=("W's own parameters (critic, classifier, surface, constants) get no gradient; W acts on "
                                "the actor only through the command"),
                        w0="with k W = 0 the loss is the room's and learners.actor_step's unweighted expression"),
        standins=stand, form_choices=FORM_CHOICES,
        data_side=dict(
            cloned_actor="c(s): the saved cloned actor of the world (check 2's start actor)",
            density=("sigma_k^2 = mean_i (a_ik - c_k(s_i))^2 over the training rows (maximum likelihood with mean c); "
                     "N(s_i, a_i) precomputed per row; q95_N on the held-out rows"),
            distance="see deviations (standard units, exact d10, surface); f(s_i, a_i) precomputed per row",
            two_head="D(s_i, a_i) precomputed per row with the frozen critic",
            classifier="see deviations; refitted every 4,000 steps per arm",
            held_out_rows="see deviations (tag 81)",
            tags=dict(held_out=TAG_HELD, classifier_rows=TAG_CLS_ROWS, surface_queries=TAG_SURF_QUERIES,
                      classifier_init=TAG_CLS_INIT, surface_init=TAG_SURF_INIT, surface_rows=TAG_SURF_ROWS),
            settings=dict(cls_steps=CLS_STEPS, cls_batch=CLS_BATCH, cls_lr=CLS_LR, cls_hidden=CLS_HIDDEN,
                          miscoverage=MISCOVERAGE, k_nn=K_NN, surf_per_type=SURF_PER_TYPE, surf_types=SURF_TYPES,
                          surf_valid=SURF_VALID, surf_steps=SURF_STEPS, surf_batch=SURF_BATCH, surf_lr=SURF_LR,
                          surf_lr_end=SURF_LR_END, surf_hidden=SURF_HIDDEN, surf_noise=SURF_NOISE,
                          held_rows=HELD_ROWS, level=LEVEL)),
        check_e=("every seed and world, before any arm is trained, on every penalty (the 8 families): the room's rule "
                 "(harness.oracle_shape) for any penalty: on the room's 2,000 full-throttle-region training states "
                 "(default_rng([20261015, seed, 65])) thruster 1 of the no-caution actor's command is set along the "
                 "actor's own push to each throttle of the room's grid; (e1) W falls from 80% to 81% on at most 25% "
                 "of them, (e2) mean W rises from 80% to 81%, (e3) the in-program penalty's mean slope along the push "
                 "at 80.5% is positive and every slope finite, (e4) the best grid throttle for Q1 - k W lies past 80% "
                 "on fewer states than for Q1 alone, k the coefficient of ladder knob 1 (also reported per rung). "
                 "Reported; not a stop condition (deviations)."),
        checks=dict(b="the stand-in program with W = 0 matches check 2's actor within 1e-5 on every parameter (each "
                      "seed and world; stop on failure)",
                    e="see check_e",
                    row_stream="the batch-row stream's SHA-256 equals the room lock's check-(a) record (stop otherwise)",
                    seed0_f=("seed 0: each penalty finite and nonnegative with finite gradients on every training row at "
                             "the recorded, cloned, no-caution and uniform commands, both worlds"),
                    seed0_d=("seed 0: a 400-step run of every arm (each family at knob 1, and W = 0) in both worlds, "
                             "200-step chunks (two classifier refits), finite; W = 0 against the room's RoomRunner "
                             "(parameter hash); timing")),
        ladder=dict(knobs=list(LADDER), scale="kappa_X = g_band / g_X at the no-caution actor's commands (deviations)",
                    extension=f"x sqrt(10), at most {MAX_EXTENSIONS} rungs", matcher="glq_match.match_parallel",
                    tol=M.MATCH_TOL, rounds=M.MATCH_ROUNDS,
                    target=("the room's perfect band's strength of the same seed, lock.json target_strength (S = RMS "
                            "change of the command from no caution on the 5,000 evaluation-split states)"),
                    chosen_arm="the evaluated knob with the best |S/S* - 1| (glq_match's record)",
                    unmatched="no matched arm: every mark fails; the strongest finite rung is evaluated and reported"),
        strength=dict(definition="learners.strength on prepared.eval_obs against check 2's saved actor (the room's)",
                      region_check="the same on the full-throttle-region training states, reported"),
        lock=("after every arm is trained and matched, lock.json (+ .sha256) holds every actor's parameter SHA-256, "
              "the actors file's SHA-256, strengths, matches, coefficients, checks (b) and (e), the data-side fit "
              "reports, the spec's and the code manifest's SHA-256 and the room files read. No return, points or "
              "score of any trained arm is computed before it exists; every evaluation verifies it first"),
        evaluation=("acceptance._evaluate (learners.evaluate_actor): exact 100-step returns on the full 1,000-start "
                    "banks of the four situations, points against the gate's anchors, capped at 80%, share past 80%; "
                    "plus, reported, the share past 80% per thruster. Scored: none (check 2's saved actor), the cloned "
                    "actor, the matched arm of every family and, for an unmatched family, its strongest finite rung "
                    "(reported). No-stall world: none, the cloned actor and every family at its locked coefficient."),
        marks=dict(
            captures_the_room=("a stand-in captures the room when all three hold: (1) its matched gain over no caution "
                               "in strong current is at least 50% of the perfect band's gain; (2) it scores at least no "
                               "caution - 3 in recorded conditions; (3) it beats the matched even pull by at least 15 "
                               "points in strong current"),
            pooled=("each condition on the pooled mean over seeds 0-4 of the paired per-seed difference; every seed "
                    "needed; computed for the five stand-ins (the mark) and for the three literal forms (reported)"),
            reported=("each stand-in's no-stall loss at its locked scale (no caution minus the stand-in, no-stall "
                      "world, strong current; every situation reported); selectivity = stalling-world gain minus "
                      "no-stall loss; selectivity minus the range hinge's (paired)"),
            expected_by_the_plan=("the range hinge and behaviour density capture most of the room in the stalling "
                                  "world, and every stand-in costs points in the no-stall world; this sets the bar "
                                  "for BCA; it is not evidence for BCA")),
        deviations=DEVIATIONS,
        evidence=dict(note=("scratch scripts that measured the frozen surfaces behind F1-F3 before any stand-in "
                            "training (no return, points or strength); hashed here for provenance"),
                      files={n: file_sha256(HERE / "explore" / n) for n in ("cache_seed.py", "shapes.py", "proto.py")}),
        user_approval=("F1-F3 change the plan's stand-in formulas (the literal forms still run, reported). Launch only "
                       "after the user approves them. Nothing in the code enforces this."),
        code=("runs/shift_explore/standins/code_manifest.json: every phase and check refuses unless each listed "
              "script and its copy under code/ have the recorded SHA-256 and the manifest names this spec"),
        run_order=("0. the room test's pooled look passes (room, recovery, keeps improvement) and the user approves "
                   "F1-F3; 1. per seed 0-4: train, eval, no_stall_train, no_stall_eval (one pinned process per seed); "
                   "2. marks over seeds 0-4. Logs: runs/shift_explore/standins/logs/."),
        gate_files_sha256=H.gate_file_hashes(),
        gate_prereg_sha256=file_sha256(ROOT / "runs" / "shift" / "preregistration.json"),
        outputs=str(OUT))

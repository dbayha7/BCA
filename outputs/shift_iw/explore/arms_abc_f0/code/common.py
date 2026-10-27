"""EXPLORATORY: arms A, B, C and result F's no-band control F0 -- the pieces every job shares.

Pass marks and kill rules: routes2/arms/PASSMARKS_arms_ABC_and_F_control.md (frozen with a sha256sum sidecar before
this code was written). Every job, check, scorer and wording step calls require_passmarks() first and refuses on any
difference between that file, its sidecar and the digest below.

Runs from <sandbox>/repo with PYTHONPATH=<sandbox>/repo:<sandbox>/explore (the sandbox's byte-identical copy of the
weighting test's code); this folder is put on the path because every job is started as a file (python -B <file>).
The package and result F's scripts are imported unchanged (result F's from the repository's channel/ folder, read in
place, never copied); reused_files() records the SHA-256 of every reused file in each record.
"""
import sys

sys.dont_write_bytecode = True

import hashlib  # noqa: E402
import importlib.util  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import os  # noqa: E402
import platform  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

CODE = Path(__file__).resolve().parent
ARMS_DIR = CODE.parent
PASSMARKS = ARMS_DIR / "PASSMARKS_arms_ABC_and_F_control.md"
PASSMARKS_SHA256 = "a073be97003938c112e9ba77581a311e812fe5851f7e64211a11fb4bf47d9ecf"
SRC = Path(os.environ.get("ARMS_SRC", "/home/user/bca"))
IW = Path(os.environ.get("ARMS_IW", str(ARMS_DIR.parent.parent / "iw_rerun")))
CHAN = SRC / "outputs" / "shift_iw" / "explore" / "channel"
STATE = ARMS_DIR / "state"
LOGS = ARMS_DIR / "logs"
SEEDS = (0, 1, 2, 3, 4)
QUALITIES = ("random", "medium-replay", "medium", "medium-expert", "expert")
F0_QUALITIES = ("medium-expert", "expert")
K0_QUALITIES = ("medium-replay", "medium", "medium-expert")      # arm A's K0 (section 4.5)
P1_QUALITIES = ("medium-replay", "medium", "medium-expert")      # arm A's P1
WEAK = ("random", "medium-replay", "medium")                     # arm C's B1 and KB0
MASTER = 20261016
TAG_UNTARGETED, TAG_SPREAD = 4301, 4302                          # arm A (section 4.1)
TAG_CERT, TAG_SHUFFLE = 4311, 4312                               # arm B (section 5.1-5.2)
KNEE32 = np.float32(0.8)
LABEL = ("EXPLORATORY: arms A/B/C and result F's no-band control; seen seeds 0-4; pass marks frozen in "
         "routes2/arms/PASSMARKS_arms_ABC_and_F_control.md (SHA-256 {sha}) before this code was written")
OWN_CODE = ("common.py", "arm_a.py", "arm_b.py", "arm_c.py", "arm_f0.py", "gates.py", "checks.py", "screen.py", "kills.py",
            "score.py", "rt.py", "test_arms.py")


class Refused(RuntimeError):
    """A job refused (frozen file, gate, inputs or a kill rule that already fired)."""


class InputFailed(Refused):
    """An input failed for good (a TD3+BC host that went non-finite, an invalid band, a non-finite A2 actor): rerunning
    cannot change it, so the job writes a stopped record instead of refusing (section 3.2: the number is missing and
    every mark that needs it is not met)."""


# ------------------------------------------------------------------------------------------------ the frozen file

def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def require_passmarks(path=PASSMARKS, digest=PASSMARKS_SHA256):
    """The frozen pass marks: the file, its sha256sum sidecar and the digest frozen before any code agree. Returns
    the digest; raises Refused otherwise."""
    path = Path(path)
    side = Path(str(path) + ".sha256")
    if not path.is_file() or not side.is_file():
        raise Refused(f"refused: the frozen pass marks or their sidecar are missing ({path})")
    got = sha256_file(path)
    words = side.read_text(encoding="utf8").split()
    if not words or words[0] != got:
        raise Refused(f"refused: {path.name} differs from its sidecar (the frozen pass marks changed)")
    if got != digest:
        raise Refused(f"refused: {path.name} is not the file frozen before this code ({got} != {digest})")
    return got


def label(digest=PASSMARKS_SHA256):
    return LABEL.format(sha=digest)


def this_code():
    return {n: sha256_file(CODE / n) for n in OWN_CODE if (CODE / n).is_file()}


def channel_module(name, file):
    """One of result F's scripts, loaded in place from the repository's channel/ folder (never copied; not entered in
    sys.modules)."""
    spec = importlib.util.spec_from_file_location(name, CHAN / file)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def reused_files():
    """SHA-256 of every reused file: the package and the repository modules it loads (learners_iw.code_hashes, the
    files the committed approval names), result F's three scripts and their frozen pass marks, and the four-situation
    script whose arithmetic score.py restates."""
    from experiments.shift_iw import learners_iw as LI
    out = {f"sandbox:{k}": v for k, v in LI.code_hashes().items()}
    for f in ("command_le_train_f1.py", "command_le_train.py", "command_band_check.py",
              "PASSMARKS_fallback_rerun.md"):
        out[f"repository:outputs/shift_iw/explore/channel/{f}"] = sha256_file(CHAN / f)
    sit = ARMS_DIR.parent.parent / "sit4" / "situations.py"
    if sit.is_file():
        out["scratchpad:sit4/situations.py (restated in score.py)"] = sha256_file(sit)
    return out


# ------------------------------------------------------------------------------------------------ records

def plain(x):
    """Strict-JSON form (the package's run.plain: numpy to Python, inf and nan to strings)."""
    if isinstance(x, dict):
        return {str(k): plain(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [plain(v) for v in x]
    if hasattr(x, "__array__") and not isinstance(x, (str, bytes)):
        a = np.asarray(x)
        return plain(a.tolist()) if a.ndim else plain(a.item())
    if isinstance(x, (bool, np.bool_)):
        return bool(x)
    if isinstance(x, (int, np.integer)):
        return int(x)
    if isinstance(x, (float, np.floating)):
        v = float(x)
        return v if math.isfinite(v) else ("inf" if v > 0 else "-inf" if v < 0 else "nan")
    return x


def num(v):
    return None if v is None else float(v)


def write_atomic(path, text):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp{os.getpid()}")
    with open(tmp, "w", encoding="utf-8") as f:
        f.write(text)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def write_record(path, rec):
    """A JSON record, then its sha256sum sidecar (record first, sidecar last: a crash leaves no valid pair)."""
    path = Path(path)
    write_atomic(path, json.dumps(plain(rec), indent=1, sort_keys=False))
    write_atomic(Path(str(path) + ".sha256"), f"{sha256_file(path)}  {path.name}\n")
    return sha256_file(path)


def read_record(path):
    """A record that exists and matches its sidecar, or None."""
    path = Path(path)
    side = Path(str(path) + ".sha256")
    if not path.is_file() or not side.is_file():
        return None
    words = side.read_text(encoding="utf8").split()
    if not words or words[0] != sha256_file(path):
        return None
    try:
        return json.loads(path.read_text(encoding="utf8"))
    except ValueError:
        return None


def arm_record(path, digest=PASSMARKS_SHA256):
    """An arm record this run wrote: matches its sidecar, carries the label and this frozen file's digest; else
    Refused (the scorer and the aggregators never read another record)."""
    rec = read_record(path)
    if rec is None:
        raise Refused(f"refused: no verified record at {path}")
    if rec.get("label") != label(digest) or rec.get("passmarks_sha256") != digest or rec.get("exploratory") is not True:
        raise Refused(f"refused: {path} lacks the label or names another pass-mark file")
    return rec


def software():
    import flax
    import jax
    import optax
    import scipy
    return dict(python=platform.python_version(), numpy=np.__version__, jax=jax.__version__, flax=flax.__version__,
                optax=optax.__version__, scipy=scipy.__version__, machine=platform.node(),
                affinity=sorted(os.sched_getaffinity(0)), omp=os.environ.get("OMP_NUM_THREADS"),
                xla_flags=os.environ.get("XLA_FLAGS"), openblas=os.environ.get("OPENBLAS_CORETYPE"))


def base_record(arm, quality, seed, digest):
    from experiments.shift_iw import learners_iw as LI
    return dict(label=label(digest), exploratory=True, passmarks_sha256=digest, arm=arm, quality=quality,
                seed=int(seed), smoke_test=None if smoke() is None else "SMOKE TEST: not results",
                spec_sha256=LI.spec_sha256(), code_sha256=LI.code_hashes(), this_code=this_code(),
                reused_files_sha256=reused_files(), software=software(),
                started_at=time.strftime("%Y-%m-%dT%H:%M:%S%z"))


def smoke():
    """ARMS_SMOKE=<scratch folder>: a smoke test or the aggregation unit test; every output of this run goes there
    instead (never set by run_arms.sh start/continue)."""
    v = os.environ.get("ARMS_SMOKE")
    return Path(v) if v else None


def smoke_settings():
    """ARMS_SMOKE_SETTINGS=1 (only with ARMS_SMOKE): a few steps, 50 starts per bank, 3,000 tries; not results."""
    return smoke() is not None and os.environ.get("ARMS_SMOKE_SETTINGS") == "1"


def out_root():
    if smoke() is not None:
        return smoke() / "shift_iw_v5_explore" / "arms_abc_f0"
    from experiments.shift_iw import run as R
    return R.ROOT_DEFAULT.parent / "shift_iw_v5_explore" / "arms_abc_f0"


def docs():
    """Where SCREEN_seed0, KILLS and RESULTS go: routes2/arms (a smoke test: <ARMS_SMOKE>/docs)."""
    return smoke() / "docs" if smoke() is not None else ARMS_DIR


def state():
    return smoke() / "state" if smoke() is not None else STATE


def f1_out():
    """Result F's script's output folder (channel/train_f1: A2's records and evaluations)."""
    return out_root().parent / "channel" / "train_f1"


def v5_root():
    if smoke() is not None:
        return smoke() / "shift_iw_v5"
    from experiments.shift_iw import run as R
    return R.ROOT_DEFAULT


def arm_dir(arm, quality, seed):
    """<sandbox>/repo/runs/shift_iw_v5_explore/arms_abc_f0/<arm>/<quality>/seed<k> (F0: .../F0/<quality>/F0_no_band/
    seed<k>, section 7.2)."""
    if arm == "F0":
        return out_root() / "F0" / quality / "F0_no_band" / f"seed{int(seed)}"
    return out_root() / arm / quality / f"seed{int(seed)}"


def log(m):
    print(time.strftime("%H:%M:%S ") + str(m), flush=True)


# ------------------------------------------------------------------------------------------------ one seed's inputs

def host_failed(ctx, seed, spec):
    """The reason when the seed's gapped TD3+BC host finished but failed (non-finite; the package never retrains it), or
    None (finished and usable, or not finished yet)."""
    from experiments.shift_iw import run as R
    rec = R.finished(R.host_dir(ctx, "gap", seed) / "record.json", spec)
    if rec is not None and (rec.get("result") or {}).get("failed"):
        return (f"the gapped TD3+BC host of {ctx.quality} seed {int(seed)} failed (non-finite at update "
                f"{rec['result'].get('failed_at')})")
    return None


def band_invalid(ctx, seed, spec):
    """The reason when the seed's gapped band finished with no valid refresh (band_valid false), or None."""
    from experiments.shift_iw import run as R
    rec = R.finished(R.band_dir(ctx, "gap", seed) / "record.json", spec)
    if rec is not None and not rec.get("band_valid"):
        return f"the gapped band of {ctx.quality} seed {int(seed)} is invalid (band_valid false)"
    return None


def write_stop(path, rec, rule, reason, started=None, **extra):
    """A stopped record (a kill rule fired, or an input failed for good): written whole with its sidecar, so the kill
    checks can complete and every mark that needs the seed is not met."""
    rec.update(status="stopped", stopped_by=rule, reason=reason, finished_at=time.strftime("%Y-%m-%dT%H:%M:%S%z"),
               **extra)
    if started is not None:
        rec["seconds"] = time.perf_counter() - started
    write_record(path, rec)
    log(f"{rec.get('arm')} {rec.get('quality')} seed {rec.get('seed')}: STOPPED by {rule}: {reason}")
    return 0


def files_fresh(hashes):
    """Every file a summary read still exists with the SHA-256 it had then (a summary built on older inputs is stale)."""
    for p, h in (hashes or {}).items():
        try:
            if sha256_file(p) != h:
                return False
        except OSError:
            return False
    return True


class Seed:
    """One quality and data seed: context, data, the gapped data prepared as every learner sees it, the gapped TD3+BC
    host (frozen) and the shared start banks with their anchors."""

    def __init__(self, quality, seed):
        from experiments.shift_iw import data as D
        from experiments.shift_iw import run as R
        if quality not in QUALITIES or int(seed) not in SEEDS:
            raise Refused(f"refused: quality {quality!r} and seed {seed!r} must be one of {QUALITIES} and {SEEDS}")
        self.quality, self.seed = quality, int(seed)
        self.ctx = R.Context(quality=quality, log=log)
        self.cfg = self.ctx.cfg
        try:
            self.spec = R.spec_digest(self.ctx)
            R.require(R.banks_record_path(self.ctx), self.spec, "the start banks")
            self.data = R.load_data(self.ctx, seed, self.spec)
            failed = host_failed(self.ctx, seed, self.spec)
            if failed:
                raise InputFailed(failed)
            self.host_rec = R.require_host(self.ctx, "gap", seed, self.spec)
            self.host = R.load_host_state(self.ctx, "gap", seed, self.spec)
        except R.Refused as error:
            raise Refused(str(error)) from error
        if smoke_settings():
            from experiments.shift_iw import learners_iw as LI
            self.cfg = LI.settings(clone_steps=200, actor_only_steps=500, actor_only_chunk=100)
        self.prep = D.prepare(self.data, "gap", self.cfg)
        self.banks, sb = D.load_start_banks(self.ctx.banks_root)
        if smoke_settings():
            self.banks = {k: v[:50] for k, v in self.banks.items()}
        self.anchors = sb["anchors"]
        self.obs = np.asarray(self.prep.train.obs, np.float32)
        self.a_rec = np.asarray(self.prep.actions, np.float32)

    def links(self, paths):
        from experiments.shift_iw import run as R
        return R.links(self.ctx, paths)

    def host_record_path(self):
        from experiments.shift_iw import run as R
        return R.host_dir(self.ctx, "gap", self.seed) / "record.json"


# ------------------------------------------------------------------------------------------------ critic and actor rows

def q_min(critic_params, obs, actions):
    """min over the frozen critic's two heads at float32 normalised observations and commands (float64 out)."""
    from experiments.shift import learners as LR
    return LR.critic_rows(critic_params, obs, actions)[0].min(axis=1)


def actor_out(params, obs):
    from experiments.shift import learners as LR
    return LR.actor_rows(params, obs)


_PROGRAMS = {}


def plain_spread(host, prep, cfg, seed, chunk=8192):
    """Arm A's plain residual spread s (section 4.1, step 2; no BCA): over every kept training row,
    y_i = r_i + 0.98 (1 - done_i) min_h Q'_h(s'_i, clip(pi'(s'_i) + e_i, -1, 1)) with the host's target networks and
    target-policy noise e_i ~ N(0, 0.2^2) clipped to +-0.5 (td3_bc_bca.native_target's arithmetic, in float32),
    e_i keyed from numpy default_rng([20261016, seed, 4302]); q_i = min_h Q_h(s_i, a_i); s = population SD of y - q.
    Returns (s, details)."""
    import jax
    import jax.numpy as jnp
    from experiments.shift import learners as LR
    args = LR.make_args(cfg, seed)
    rng = np.random.default_rng([MASTER, int(seed), TAG_SPREAD])
    z = rng.standard_normal((prep.n, LR.DA)).astype(np.float32)
    noise = np.clip(z * np.float32(args.policy_noise), -np.float32(args.noise_clip), np.float32(args.noise_clip))
    key = ("spread", float(args.discount), int(chunk))
    if key not in _PROGRAMS:
        disc = float(args.discount)

        def target(nat, nxt, rew, done, e):
            a = LR.ACTOR.apply(nat.actor_target.params, nxt)
            a = jnp.clip(a + e, -1.0, 1.0)
            q = LR.CRITIC.apply(nat.critic_target.params, nxt, a).min(axis=-1)
            return rew + (1.0 - done) * disc * q
        _PROGRAMS[key] = jax.jit(target)
    fn = _PROGRAMS[key]
    tr = prep.train
    nxt, rew, done = (np.asarray(tr.next_obs, np.float32), np.asarray(tr.reward, np.float32),
                      np.asarray(tr.done, np.float32))
    ys = []
    for i in range(0, prep.n, chunk):
        sl = slice(i, i + chunk)
        m = len(rew[sl])
        pad = chunk - m

        def p(x):
            return np.concatenate([x, np.repeat(x[-1:], pad, 0)]) if pad else x
        y = fn(host, jnp.asarray(p(nxt[sl])), jnp.asarray(p(rew[sl])), jnp.asarray(p(done[sl])),
               jnp.asarray(p(noise[sl])))
        ys.append(np.asarray(y, np.float64)[:m])
    y = np.concatenate(ys)
    q = q_min(host.critic.params, np.asarray(tr.obs, np.float32), np.asarray(prep.actions, np.float32))
    resid = y - q
    s = float(np.std(resid))
    return s, dict(s=s, residual_mean=float(resid.mean()), rows=int(prep.n), finite=bool(np.all(np.isfinite(resid))),
                   noise_stream=[MASTER, int(seed), TAG_SPREAD], noise_sha256=hashlib.sha256(noise.tobytes()).hexdigest())


# ------------------------------------------------------------------------------------------------ acting and scoring

def ceiling_policy(base, t, name=None):
    """clip(pi(x), -t_k, +t_k) per thruster inside the policy, before truth.rollout's own clip to [-1, 1]."""
    from experiments.shift import recorder as RC

    class Ceiling(RC.Controller):
        def __init__(self):
            self.base = base
            self.t = np.asarray(t, np.float64).reshape(-1)
            if self.t.shape != (4,) or not np.all(np.isfinite(self.t) & (self.t > 0)):
                raise ValueError("a ceiling map holds one finite positive ceiling per thruster")
            self.name = name or f"{getattr(base, 'name', 'actor')} with ceilings {self.t.tolist()}"

        def __call__(self, x):
            return np.clip(self.base(x), -self.t, self.t)
    return Ceiling()


def actor_policy(params, prep, name="learned actor (float64 copy)"):
    from experiments.shift import learners as LR
    return LR.mlp_policy(params, prep, name=name)


def evaluate_policy(policy, banks, anchors):
    """learners_iw.evaluate_actor's arithmetic for any policy: exact float64 100-step returns (truth.rollout) from
    every fixed start of each bank, points = world.points with the bank's anchors."""
    from experiments.shift import truth as TR
    from experiments.shift_iw import world as W
    out = {}
    for law, x0 in banks.items():
        J = TR.rollout(policy, x0, W.PHYSICS).J
        pts = W.points(J, anchors[law]["anchor"])
        n = len(J)
        out[law] = dict(J=float(J.mean()), J_se=float(J.std(ddof=1) / math.sqrt(n)) if n > 1 else None,
                        points=float(pts.mean()), points_se=float(pts.std(ddof=1) / math.sqrt(n)) if n > 1 else None,
                        finite=bool(np.all(np.isfinite(J))), starts=int(n))
    return out


def strong_commands(policy, banks):
    """The applied commands (1,000 starts x 100 steps x 4) of a policy's unclipped rollouts from the strong bank's
    fixed starts. Only commands leave this function: no return is computed for a record here."""
    from experiments.shift import truth as TR
    from experiments.shift_iw import world as W
    return np.asarray(TR.rollout(policy, banks["strong"], W.PHYSICS, record=True).commands, np.float64)


def past_share(commands, t, thruster=None):
    """Share of entries with |a| > t (per thruster t; thruster 0-based or None for all four)."""
    a = np.abs(np.asarray(commands, np.float64))
    t = np.asarray(t, np.float64).reshape(-1)
    if thruster is None:
        return float(np.mean(a > t))
    return float(np.mean(a[..., thruster] > t[thruster]))


SITUATIONS = ("recorded", "big_kicks", "big_waves", "strong_current")


def situation_banks():
    """The main test's four situations as /scratchpad/sit4/situations.py builds them: thruster_chain.bank_starts (1,000
    fixed starts each) and data.anchors (0 = the random quality, 100 = the stall-aware controller)."""
    from experiments.shift import thruster_chain as TC
    from experiments.shift_iw import data as D
    n = 50 if smoke_settings() else TC.BANK_SIZE
    banks = {s: TC.bank_starts(s, n) for s in SITUATIONS}
    return banks, D.anchors(banks)


def evaluate_situations(policy, sbanks, sanchors):
    """situations.py's arithmetic: mean J, its SE, the discounted mean and points of the mean."""
    from experiments.shift import truth as TR
    from experiments.shift_iw import world as W
    out = {}
    for s in SITUATIONS:
        run = TR.rollout(policy, sbanks[s], W.PHYSICS)
        Jm = float(run.J.mean())
        se = float(run.J.std(ddof=1) / math.sqrt(len(run.J)))
        a = sanchors[s]["anchor"]
        out[s] = dict(J=Jm, se=se, Jd=float(run.Jd.mean()), points=float(W.points(Jm, a)),
                      points_se=100.0 * se / (a["top"] - a["random"]), finite=bool(np.all(np.isfinite(run.J))))
    return out


def finite_tree(tree):
    from experiments.shift import learners as LR
    return bool(LR._finite_tree(tree))

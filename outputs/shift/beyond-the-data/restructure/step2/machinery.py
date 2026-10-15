"""Step 2: the heavy machinery (loaders, the outcome-ramp runner, helpers shared by the phases). EXPLORATORY; see
PRESPEC.md.

Read-only imports, each executed from bytes whose SHA-256 prespec.json pins, WITHOUT version 6's approval checks.
Step 2 lies outside the gate and outside version 6: it calls none of version 6's phase functions (they all require
version 6's approval) and none of its writers (they write only under runs/shift_explore/stage2). It uses:
  - version 6's Stage 2 code (stage2_common.py, weights.py, band.py, fresh.py, policy.py) from
    outputs/shift/beyond-the-data/stage2/code (byte-identical to the copies version 6's code manifest pins):
    fresh.context (the gate's verified loaders through the room harness's SeedContext), policy.room_scales,
    policy.match_controls, policy.strengths_of, policy.arms_record, policy.standins_module, policy.MATCHED,
    policy.W0_TOL. weights.py is loaded only because band.py imports it (`import weights as Wt`, which then binds to
    the verified module); step 2 calls none of its functions. check_imports() refuses if any module from that folder
    or step 1's was loaded without its hash checked (at import, before every lock and before every values file);
  - through it, the room test's harness (stage2_common.load_room_harness checks its manifest and bytes): RoomRunner,
    RoomArm, ProgramConfig, build_shared, family_arm, random_surface_params, max_param_diff, shape_rows, the ladders;
  - the stand-ins module (policy.standins_module checks version 6's pre-registration pins, the stand-ins manifest and
    the module's bytes): build_shared, classifier_fit_for, slope_scales, StandinRunner, family_arm, locked_arm, LADDER,
    match_standins;
  - step 1's step1lib.py: the gap rule (gap_rule, throttle), to re-derive t_lo and t_up from the data.

The new piece is RampRunner: the room's RoomRunner with one more penalty kind, 'ramp' (coefficients (h, k)), whose
actor step is version 6's actor_step_bca with the outcome ramp in place of BCA's width (PRESPEC.md, THE ACTOR LOSS).
"""
import sys

sys.dont_write_bytecode = True

import time  # noqa: E402
from functools import partial  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import step2lib as S  # noqa: E402

SPEC, SPEC_SHA = S.load_prespec()
_IN = SPEC["inputs"]


_exec_verified = S.exec_verified

V6_DIR = S.ROOT / _IN["v6_code"]["dir"]
_V6 = _IN["v6_code"]["sha256"]
C = _exec_verified("stage2_common", V6_DIR / "stage2_common.py", _V6["stage2_common.py"])
Wt = _exec_verified("weights", V6_DIR / "weights.py", _V6["weights.py"])     # before band.py, which imports it
B = _exec_verified("band", V6_DIR / "band.py", _V6["band.py"])
F = _exec_verified("fresh", V6_DIR / "fresh.py", _V6["fresh.py"])
P = _exec_verified("policy", V6_DIR / "policy.py", _V6["policy.py"])
S1 = _exec_verified("step1lib", S.ROOT / _IN["step1lib"]["path"], _IN["step1lib"]["sha256"])
H, A, L, TC, TR, M = P.H, P.A, P.L, P.TC, P.TR, P.M
VERIFIED_FOLDERS = (V6_DIR, (S.ROOT / _IN["step1lib"]["path"]).parent)


def check_imports():
    """Refuses if any module from version 6's code folder or step 1's folder was loaded through the normal import
    system (no hash check), or if band.py's `weights` is not the verified module. Returns {module: SHA-256} of the
    modules loaded from verified bytes (recorded in every lock and values file)."""
    bad = S.unverified_modules(VERIFIED_FOLDERS)
    if bad:
        raise SystemExit(f"refused: modules loaded from a pinned folder without a hash check: {bad}")
    if B.Wt is not Wt:
        raise SystemExit("refused: band.py's weights module is not the verified weights.py")
    return {n: m.__step2_sha256__ for n, m in sorted(sys.modules.items())
            if getattr(m, "__step2_sha256__", None) is not None}


check_imports()
if C.ROOM_PINNED["harness_sha256"] != _IN["room"]["harness_sha256"] or \
        C.ROOM_PINNED["manifest_sha256"] != _IN["room"]["manifest_sha256"]:
    raise SystemExit("refused: version 6's room-harness pins differ from prespec.json's")

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
from flax import serialization  # noqa: E402

W0_TOL = float(P.W0_TOL)
if W0_TOL != float(SPEC["checks"]["w0_tolerance"]):
    raise SystemExit("refused: version 6's check (b) tolerance differs from prespec.json's")


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def v6_prereg():
    """Version 6's pre-registration (read with its sidecar check; its pins locate the stand-ins module)."""
    pr = C.load_prereg()
    if pr.sha256 != _IN["v6_prereg_sha256"]:
        raise SystemExit("refused: version 6's prereg.json is not the one prespec.json names")
    return pr


def best_standin():
    rec = C.read_json(C.OUT / "best_standin.json")
    if rec["best"] != _IN["best_standin"]:
        raise SystemExit(f"refused: version 6's best stand-in is {rec['best']}, prespec.json says {_IN['best_standin']}")
    return rec["best"]


def context(rep, world):
    """The gate's verified artifacts of one gate seed (version 6's fresh.context -> the room harness's SeedContext),
    for one world only."""
    S.check_seed(rep)
    return F.context(int(rep), worlds=(S.check_world(world),))


def gap_rule_check(pg, row):
    """Step 1's gap rule on this world's gapped training commands must give the arm table's t_lo and t_up exactly."""
    g = S1.gap_rule(S1.throttle(pg.actions))
    ok = bool(g["defined"] and g["t_lo"] == row["t_lo"] and g["t_up"] == row["t_up"])
    return dict(passed=ok, t_lo=g["t_lo"], t_up=g["t_up"], n_tail=g["n_tail"], table_t_lo=row["t_lo"],
                table_t_up=row["t_up"])


# --------------------------------------------------------------------------------------------- the actor step

_PROGRAMS = {}


def actor_step_ramp(args, actor, critic_params, batch, ramp_shared, arm):
    """Version 6's actor_step_bca (the room harness's actor_step_w with the cloning term unweighted) with the outcome
    ramp in place of BCA's width:
        loss = -lambda * mean_i[ Q1(s_i, pi(s_i)) - k [U(pi(s_i)) - U(a_i)]_+ ] + pull * mean_ij (pi(s_i)_j - a_ij)^2
        U(a) = h * clip((max_j |a_j| - t_lo) / (t_up - t_lo), 0, 1),   lambda = alpha / stopgrad(mean_i |Q1|)
    a_i is the row's recorded command (batch.action). Only the actor's parameters get a gradient."""

    def _actor_loss_fn(params, critic_params):
        pi = L.ACTOR.apply(params, batch.obs)
        q = L.CRITIC.apply(critic_params, batch.obs, pi)[..., 0]
        lmbda = args.alpha / jax.lax.stop_gradient(jnp.abs(q).mean())
        kw = S.ramp_penalty(pi, batch.action, arm["coef"][0], arm["coef"][1], ramp_shared["t_lo"],
                            ramp_shared["t_up"], xp=jnp)
        bc = jnp.square(pi - batch.action).mean() * arm["pull"]
        return (-lmbda * (q - kw).mean() + bc, (q.mean(), lmbda, bc, kw.mean()))

    grad_fn = jax.value_and_grad(_actor_loss_fn, has_aux=True)
    (a_loss, aux), a_grad = grad_fn(actor.params, critic_params)
    return actor.apply_gradients(grads=a_grad), (a_loss, *aux)


def _ramp_loop(args, critic_params, actor, train, ramp_shared, rows, arm):
    def body(a, r):
        batch = jax.tree_util.tree_map(lambda x: x[r], train)
        a, aux = actor_step_ramp(args, a, critic_params, batch, ramp_shared, arm)
        return a, jnp.stack(aux)

    return jax.lax.scan(body, actor, rows)


def ramp_arm(name, h, k):
    return H.RoomArm(name, "ramp", (float(h), float(k)))


class RampRunner(H.RoomRunner):
    """The room's RoomRunner with the 'ramp' kind (coefficients (h, k)); t_lo and t_up are this seed-world's, shared
    by every ramp arm. Every other kind runs through the room's own unchanged program. The arms of one kind run
    vmapped side by side, from the same start actor and the same stream of batch rows."""

    def __init__(self, *a, t_lo, t_up, **kw):
        super().__init__(*a, **kw)
        self.ramp_shared = dict(t_lo=jnp.asarray(t_lo, jnp.float32), t_up=jnp.asarray(t_up, jnp.float32))
        self.chunk_seconds = []

    def ramp_program(self, n_arms):
        key = ("step2-ramp", L._args_key(self.args), int(n_arms), self.chunk)
        if key not in _PROGRAMS:
            _PROGRAMS[key] = jax.jit(jax.vmap(partial(_ramp_loop, self.args), in_axes=(None, 0, None, None, None, 0)))
        return _PROGRAMS[key]

    def run_ramp(self, arms, log_fn=None):
        n = len(arms)
        for a in arms:
            if len(a.coef) != 2:
                raise ValueError(f"arm {a.name}: the ramp kind takes two coefficients (h, k)")
        states = jax.tree_util.tree_map(lambda x: jnp.broadcast_to(x, (n,) + jnp.shape(x)), self.start)
        tree = dict(coef=jnp.asarray(np.array([a.coef for a in arms], np.float32)),
                    pull=jnp.asarray(np.array([a.pull for a in arms], np.float32)))
        program = self.ramp_program(n)
        metrics, finite_steps = [], np.ones(n, bool)
        t0 = time.perf_counter()
        for c in range(self.steps // self.chunk):
            states, aux = program(self.critic, states, self.prepared.train, self.ramp_shared, self.rows[c], tree)
            aux = np.asarray(aux, np.float64)
            finite_steps &= np.all(np.isfinite(aux), axis=(1, 2))
            metrics.append(aux.mean(1).tolist())
            self.chunk_seconds.append(("ramp", n, c, time.perf_counter() - t0))
            if log_fn:
                log_fn(f"  ramp x{n}: step {(c + 1) * self.chunk}/{self.steps} ({time.perf_counter() - t0:.0f}s)")
        jax.block_until_ready(states.params)
        sec = time.perf_counter() - t0
        out = {}
        for i, a in enumerate(arms):
            p = jax.tree_util.tree_map(lambda x: x[i], states.params)
            out[a.name] = dict(params=p, finite=bool(finite_steps[i] and L._finite_tree(p)),
                               chunk_metrics=[m[i] for m in metrics], arm=a.record(), seconds=sec / n)
        self.seconds["ramp"] = self.seconds.get("ramp", 0.0) + sec
        return out

    def run(self, arms, log_fn=None):
        names = [a.name for a in arms]
        if len(set(names)) != len(names):
            raise ValueError("arm names must be unique")
        out = {}
        rest = [a for a in arms if a.kind != "ramp"]
        if rest:
            out.update(H.RoomRunner.run(self, rest, log_fn))
        group = [a for a in arms if a.kind == "ramp"]
        if group:
            out.update(self.run_ramp(group, log_fn))
        return out


# --------------------------------------------------------------------------------------------- the stand-in rival

def standin_matched(SM, ctx, world, pg, critic, clone, none, family, target, room_shared, cfg, steps, chunk, log_fn):
    """Version 6's policy.standin_matched, restated only to pass the step count (scratch runs); the calls are the
    same: the stand-ins' build_shared, classifier_fit_for, slope_scales, StandinRunner, the 4-rung ladder LADDER and
    match_standins (blind to the target strength). Returns (trained, match record, data-side info)."""
    shared_s, info_s, extra = SM.build_shared(ctx, world, pg, critic, clone, none, ctx.rep, log_fn)
    cls_none = SM.classifier_fit_for(none, shared_s)
    sc = SM.slope_scales(pg, critic, shared_s, room_shared, extra["a_none"], ctx.rep, cls_none[:2], cfg)
    runner = SM.StandinRunner(pg, clone, critic, ctx.learn, ctx.rep, shared=shared_s, steps=steps, chunk=chunk)
    trained = runner.run([SM.family_arm(family, k, sc["scale"]) for k in SM.LADDER], log_fn)
    m = SM.match_standins(runner, trained, none, pg, target, sc["scale"], (family,), log_fn)[family]
    return trained, m, dict(info=info_s, scale=sc, seconds=runner.seconds)


def standin_locked(SM, ctx, world, pg, critic, clone, none, family, coefficient, steps, chunk, log_fn):
    """Version 6's train_no_stall stand-in step: the stand-in at the stalling world's locked coefficient."""
    shared_s, info_s, _ = SM.build_shared(ctx, world, pg, critic, clone, none, ctx.rep, log_fn)
    sr = SM.StandinRunner(pg, clone, critic, ctx.learn, ctx.rep, shared=shared_s, steps=steps, chunk=chunk)
    st = sr.run([SM.locked_arm(family, coefficient, name="standin_locked")], log_fn)
    return st["standin_locked"], dict(info=info_s, seconds=sr.seconds)


# --------------------------------------------------------------------------------------------- reported readouts

def info_only(what, fn, log_fn=log):
    """An INFORMATION-ONLY item: an error inside it is recorded and never stops a phase."""
    try:
        return fn()
    except (Exception, SystemExit) as e:
        log_fn(f"INFORMATION-ONLY ITEM FAILED ({what}): {e!r}; recorded, the phase continues")
        return dict(error=f"{type(e).__name__}: {e}")


def ramp_shape(pg, critic, none, rep, t_lo, t_up, arms):
    """Reported (the room's check (e) measurement, restated for the ramp): on the room's 2,000 full-throttle-region
    training states, thruster 1 of the no-caution actor's command set along its push to each throttle of the room's
    grid; the share of states whose best grid throttle for Q1 - k [U(a) - U(a_i)]_+ lies past 80%, against Q1
    alone. arms: {name: (h, k)}."""
    rows = H.shape_rows(pg, rep)
    obs = np.asarray(pg.train.obs)[rows]
    base = L.actor_rows(none, obs).astype(np.float64)
    sign = np.where(base[:, 0] < 0, -1.0, 1.0)
    a_rec = np.asarray(pg.actions, np.float64)[rows]
    grid = np.array(H.SHAPE_GRID)
    acts, q1 = {}, {}
    for m in H.SHAPE_GRID:
        aa = base.copy()
        aa[:, 0] = sign * (TC.KNEE if m == 0.80 else m)
        acts[m] = aa
        q1[m] = np.asarray(L.critic_rows(critic, obs, aa.astype(np.float32))[0][:, 0], np.float64)
    q_best = grid[np.argmax(np.stack([q1[m] for m in H.SHAPE_GRID], 1), 1)]
    out = dict(rows=int(len(rows)), q1_alone_best_past80_share=float(np.mean(q_best > 0.8)), arms={})
    for name, (h, k) in arms.items():
        W = {m: S.ramp_penalty(acts[m], a_rec, h, k, t_lo, t_up) for m in H.SHAPE_GRID}
        best = grid[np.argmax(np.stack([q1[m] - W[m] for m in H.SHAPE_GRID], 1), 1)]
        out["arms"][name] = dict(h=h, k=k, best_past80_share=float(np.mean(best > 0.8)),
                                 mean_W={f"{m:g}": float(W[m].mean()) for m in H.SHAPE_GRID})
    return out


def visit_shares(params, prepared, t_lo):
    """Where the actor goes (reported): one exact rollout per situation's full 1,000-start bank (as version 6's
    thruster_shares): the share of visited steps past 80% per thruster, and the share of visited steps whose throttle
    index max_j |a_j| (the clipped command) lies above this seed-world's t_lo."""
    pol = L.mlp_policy(params, prepared)
    out = {}
    for s in TC.SITUATIONS:
        run = TR.rollout(pol, TC.bank_starts(s, TC.BANK_SIZE), prepared.physics, record=True)
        c = np.asarray(run.commands)
        out[s] = dict(past_80_share_per_thruster=(np.abs(c) > prepared.physics.knee).mean(axis=(0, 1)).tolist(),
                      past_t_lo_share=float((np.abs(c).max(-1) > float(t_lo)).mean()))
    return out


# --------------------------------------------------------------------------------------------- lock and stop

def save_and_lock(d, actors, payload, official):
    """actors.msgpack, then lock.json (both write-once): every actor's parameter SHA-256, written before any return,
    points or score of any trained arm is computed."""
    verified = check_imports()
    names = sorted(actors)
    blob = serialization.to_bytes({k: actors[k] for k in names})
    file_sha = S.write_bytes_once(Path(d) / "actors.msgpack", blob, official)
    payload = dict(payload, names=names, actor_sha256={k: L.tree_sha256(actors[k]) for k in names},
                   actors_file="actors.msgpack", actors_file_sha256=file_sha, verified_imports=verified,
                   written=time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                   about=("the lock: every actor trained and every control matched in strength, written before any "
                          "return, points or score of any trained arm is computed"))
    return S.write_json_once(Path(d) / "lock.json", payload, official)


def verify_lock(d, spec_sha, man_sha, seed, world, official, steps):
    """(lock, actors, lock SHA-256): sidecar, record checks (step2lib.lock_problems) and every actor's SHA-256."""
    d = Path(d)
    lock = S.read_json_checked(d / "lock.json")
    problems = S.lock_problems(lock, spec_sha, man_sha, seed, world, official, steps)
    if problems:
        raise SystemExit(f"refused: {d}/lock.json: {problems}")
    acts = L.load_tree(d / lock["actors_file"], {k: L.actor_template() for k in lock["names"]},
                       lock["actors_file_sha256"])
    bad = [k for k in lock["names"] if L.tree_sha256(acts[k]) != lock["actor_sha256"][k]]
    if bad:
        raise SystemExit(f"refused: actors differ from the lock: {bad}")
    return lock, acts, S.file_sha256(d / "lock.json")


def stop(d, payload, official, reason):
    S.write_json_once(Path(d) / "stop.json", dict(payload, kind="step2-stop", stopped=True, reason=reason,
                                                  written=time.strftime("%Y-%m-%dT%H:%M:%S%z")), official)
    log(f"STOP: {reason}")
    raise SystemExit(2)


def machine_record():
    import platform
    return dict(cpus=S.cpus(), machine=platform.node(), processor=_cpu_model(), software=A.software_state())


def _cpu_model():
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return None

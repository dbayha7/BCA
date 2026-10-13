"""Stage 2, the policy test (plan section 5.3): actor-only training with BCA's lower edge, on the room test's protocol.

PROTOCOL (the room test's, runs/shift_explore/room/code/harness.py imported from its pinned bytes and reused):
check 2's actor-only protocol: the frozen fixed-controller gapped critic of the world (its scored policy the
stall-aware controller), start from the saved cloned actor, alpha 5, 20,000 actor steps in 4,000-step chunks, check
2's stream of batch rows default_rng([20261015, seed, 35]), Adam 3e-4, the arms of one kind vmapped side by side. The
room's RoomRunner is subclassed (BCARunner) to add two penalty kinds; every other kind (plain, band, hinge,
push_less, random) runs through the room's own unchanged program.

THE ACTOR LOSS of a BCA arm (coefficients c = (k_paired, k_unpaired, k_shuffled)):

    loss = -lambda * mean_i[ Q1(s_i, pi(s_i)) - kW_i ] + mean_ij (pi(s_i)_j - a_ij)^2
    kW_i = k_paired [U(s_i, pi(s_i)) - U(s_i, a_i)]_+ + k_unpaired U(s_i, pi(s_i))
           + k_shuffled [U(s_sigma(i), pi(s_i)) - U(s_sigma(i), a_sigma(i))]_+
    lambda = alpha / stopgrad( mean_i |Q1(s_i, pi(s_i))| )         (on Q1 alone)

U is band.width (uniform: kind 'bca_u'; weighted: 'bca_w', whose classifier and threshold table are refitted with
the arm's own actor before every chunk). Only the actor's parameters are differentiated: the critic, eta_f and the
classifier are frozen, so the penalty acts through the command alone. GRADIENT PATHS (declared in the
pre-registration): uniform, through eta_f's command input only; weighted, also through the frozen classifier's command
input log w(s, a), both in the tabulated threshold R_c(w(s, a)) (so host optimisation backpropagates through the
threshold calculation as well, ALGORITHMS.md:127-130) and in the refusal term U_max softplus(log w - l*). sigma is the
room's fixed permutation default_rng([20261015, seed, 61]).

ARMS (stalling world): none (check 2's saved gapped-critic actor), w0_check (the BCA program with c = 0; must match
'none' within 1e-5 on every parameter, else the seed stops), bca_k1 = (1, 0, 0) PRIMARY, bca_k2 = (2, 0, 0) declared
secondary, bca_unpaired = (0, 1, 0) declared secondary, bca_shuffled = (0, 0, 1) reported, the room's perfect band
(reference), the room's range hinge at k = 1 (reported), and the gating controls matched blind to bca_k1's strength:
even extra pull, push less everywhere (reported), the two random surfaces; plus the best data-only stand-in matched
blind to bca_k1's strength with the stand-ins' own ladder.

LOCK BEFORE RETURNS: every arm is trained and every control matched from strengths alone; then lock.json holds every
actor's parameter SHA-256; only the eval phase computes returns, after verifying the lock.

Every phase function here requires a verified approval (stage2_common.recheck) before it loads any data.
"""

import sys
import time
from functools import partial
from pathlib import Path

import numpy as np

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))
import stage2_common as C  # noqa: E402
import band as B  # noqa: E402

H = C.load_room_harness()
A, L, TC, TR, M, RC = H.A, H.L, H.TC, H.TR, H.M, H.RC

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402

BCA_KINDS = ("bca_u", "bca_w")
N_COEF_BCA = 3
CAL = B.calibrator(TC.DX)
W0_TOL = H.W0_TOL
GATING = ("even_pull", "random_1", "random_2")           # the marks' controls (plan 5.3)
MATCHED = ("even_pull", "push_less", "random_1", "random_2")   # every matched family (push less reported)
SITUATIONS = TC.SITUATIONS
_PROGRAMS = {}


# --------------------------------------------------------------------------------------------- the actor step

def bca_penalty(kind, shared, arm, rows, pi):
    """kW per row (B,). Uniform: the band and U(s_i, a_i) are shared; weighted: per arm (refitted each chunk)."""
    weighted = kind == "bca_w"
    band = arm["band"] if weighted else shared["band"]
    u_rec = arm["u_rec"] if weighted else shared["u_rec"]
    c = arm["coef"]
    u = B.width(CAL, band, shared["obs"][rows], pi, weighted)
    j = shared["perm"][rows]
    us = B.width(CAL, band, shared["obs"][j], pi, weighted)
    return c[0] * B.paired(u, u_rec[rows]) + c[1] * u + c[2] * B.paired(us, u_rec[j])


def actor_step_bca(args, kind, actor, critic_params, batch, rows, shared, arm):
    """The room harness's actor_step_w (learners.actor_step, unweighted cloning term) with the BCA penalty."""

    def _actor_loss_fn(params, critic_params):
        pi = L.ACTOR.apply(params, batch.obs)
        q = L.CRITIC.apply(critic_params, batch.obs, pi)[..., 0]
        lmbda = args.alpha / jax.lax.stop_gradient(jnp.abs(q).mean())
        kw = bca_penalty(kind, shared, arm, rows, pi)
        bc = jnp.square(pi - batch.action).mean() * arm["pull"]
        return (-lmbda * (q - kw).mean() + bc, (q.mean(), lmbda, bc, kw.mean()))

    grad_fn = jax.value_and_grad(_actor_loss_fn, has_aux=True)
    (a_loss, aux), a_grad = grad_fn(actor.params, critic_params)
    return actor.apply_gradients(grads=a_grad), (a_loss, *aux)


def _bca_loop(args, kind, critic_params, actor, train, shared, rows, arm):
    def body(a, r):
        batch = jax.tree_util.tree_map(lambda x: x[r], train)
        a, aux = actor_step_bca(args, kind, a, critic_params, batch, r, shared, arm)
        return a, jnp.stack(aux)

    return jax.lax.scan(body, actor, rows)


def bca_arm(name, kind, coef):
    if len(coef) != N_COEF_BCA or kind not in BCA_KINDS:
        raise ValueError("a BCA arm takes kind bca_u or bca_w and three coefficients")
    return H.RoomArm(name, kind, tuple(float(c) for c in coef))


class BCARunner(H.RoomRunner):
    """The room's RoomRunner with the two BCA kinds. refit(actor_params) -> (Band, U at the recorded commands,
    diagnostics) refits the weighted band with one arm's actor (before every chunk; weighted kind only)."""

    def __init__(self, *a, refit=None, **kw):
        super().__init__(*a, **kw)
        self.refit = refit
        self.refit_log = {}

    def bca_program(self, n_arms, kind):
        key = ("stage2", L._args_key(self.args), kind, int(n_arms), self.chunk)
        if key not in _PROGRAMS:
            loop = partial(_bca_loop, self.args, kind)
            _PROGRAMS[key] = jax.jit(jax.vmap(loop, in_axes=(None, 0, None, None, None, 0)))
        return _PROGRAMS[key]

    def run_bca(self, kind, arms, log_fn=None):
        n = len(arms)
        states = jax.tree_util.tree_map(lambda x: jnp.broadcast_to(x, (n,) + jnp.shape(x)), self.start)
        tree = dict(coef=jnp.asarray(np.array([a.coef for a in arms], np.float32)),
                    pull=jnp.asarray(np.array([a.pull for a in arms], np.float32)))
        program = self.bca_program(n, kind)
        metrics, finite_steps = [], np.ones(n, bool)
        t0 = time.perf_counter()
        for c in range(self.steps // self.chunk):
            if kind == "bca_w":
                if self.refit is None:
                    raise ValueError("the weighted kind needs a refit function")
                got = [self.refit(jax.tree_util.tree_map(lambda x: x[i], states.params)) for i in range(n)]
                tree = dict(tree, band=jax.tree_util.tree_map(lambda *xs: jnp.stack(xs), *[g[0] for g in got]),
                            u_rec=jnp.stack([jnp.asarray(g[1], jnp.float32) for g in got]))
                for a, g in zip(arms, got):
                    self.refit_log.setdefault(a.name, []).append(g[2])
                    finite_steps[arms.index(a)] &= bool(g[2].get("finite", True))
            states, aux = program(self.critic, states, self.prepared.train, self.shared, self.rows[c], tree)
            aux = np.asarray(aux, np.float64)
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
                               chunk_metrics=[m[i] for m in metrics], arm=a.record(), seconds=sec / n,
                               refits=self.refit_log.get(a.name))
        self.seconds[kind] = self.seconds.get(kind, 0.0) + sec
        return out

    def run(self, arms, log_fn=None):
        names = [a.name for a in arms]
        if len(set(names)) != len(names):
            raise ValueError("arm names must be unique")
        out = {}
        rest = [a for a in arms if a.kind not in BCA_KINDS]
        if rest:
            out.update(H.RoomRunner.run(self, rest, log_fn))
        for kind in BCA_KINDS:
            group = [a for a in arms if a.kind == kind]
            if group:
                out.update(self.run_bca(kind, group, log_fn))
        return out


# --------------------------------------------------------------------------------------------- matching (restated)

def match_controls(runner, trained, none, prepared, target, scale, families, log_fn=C.log):
    """The room's run_seed.match_gating, restated with the target named 'bca_k1': the ladder (knob 0 = no caution, S
    = 0, leads it), the room's declared blind extension (even pull x2, others x sqrt(10), at most 3 rungs), then the
    gate's matcher glq_match.match_parallel (tol 0.01, 5 rounds). Strengths only; no return."""
    by_knob = {f: {} for f in families}
    for name, v in trained.items():
        f = v["arm"].get("family")
        if f in by_knob and v["arm"].get("knob") is not None:
            by_knob[f][float(v["arm"]["knob"])] = name

    def S_of(name):
        return float(L.strength(trained[name]["params"], none, prepared.eval_obs)) if trained[name]["finite"] \
            else float("nan")

    extensions = {f: [] for f in families}
    for _ in range(H.MAX_EXTENSIONS):
        need = []
        for f in families:
            knobs = sorted(by_knob[f])
            fin = [s for s in (S_of(by_knob[f][k]) for k in knobs) if np.isfinite(s)]
            if fin and target > max(fin):
                need.append(H.family_arm(f, knobs[-1] * H.EXTEND_FACTOR[f], scale))
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
        arms = [H.family_arm(f, float(k), scale) for f, k in proposals if float(k) not in by_knob[f]]
        if arms:
            trained.update(runner.run(arms, log_fn))
            for a in arms:
                by_knob[a.family][a.knob] = a.name
        return [S_of(by_knob[f][float(k)]) for f, k in proposals]

    states, rounds = M.match_parallel(ladders, {"bca_k1": float(target)}, evaluate, M.MATCH_TOL, M.MATCH_ROUNDS,
                                      log=log_fn)
    out = {}
    for f in families:
        rec = M.public(states[(f, "bca_k1")])
        rec.update(rounds=rounds, extensions=extensions[f], glq_match_sha256=M.source_sha256())
        name = None if rec["knob"] is None else by_knob[f].get(float(rec["knob"]))
        out[f] = dict(record=rec, arm=name, matched=name is not None)
    return out


# --------------------------------------------------------------------------------------------- the stand-in rival

def standins_module(prereg):
    """The stand-ins module, executed from the bytes the Stage 2 pre-registration pins (its whole code manifest
    verified first)."""
    C.verify_component(C.STANDINS_OUT, prereg.payload["pins"]["standins_manifest_sha256"])
    pin = prereg.payload["pins"]["standins_py"]
    return C.load_pinned_module("stage2_standins", pin["path"], pin["sha256"])


def standin_matched(S, ctx, world, pg, critic, clone, none, family, target, room_shared, cfg, log_fn=C.log):
    """The best data-only stand-in (identity fixed by best_standin.json before any BCA run), trained with the
    stand-ins' own runner and 4-rung ladder and matched blind to bca_k1's strength with their own matcher.
    Interface used (standins.py at its pinned SHA-256): build_shared, classifier_fit_for, slope_scales,
    StandinRunner, family_arm, LADDER, match_standins. Returns (trained, match record, data-side info)."""
    shared_s, info_s, extra = S.build_shared(ctx, world, pg, critic, clone, none, ctx.rep, log_fn)
    cls_none = S.classifier_fit_for(none, shared_s)
    sc = S.slope_scales(pg, critic, shared_s, room_shared, extra["a_none"], ctx.rep, cls_none[:2], cfg)
    runner = S.StandinRunner(pg, clone, critic, ctx.learn, ctx.rep, shared=shared_s)
    trained = runner.run([S.family_arm(family, k, sc["scale"]) for k in S.LADDER], log_fn)
    m = S.match_standins(runner, trained, none, pg, target, sc["scale"], (family,), log_fn)[family]
    return trained, m, dict(info=info_s, scale=sc, runner=runner, shared=shared_s)


# --------------------------------------------------------------------------------------------- shape (reported)

def bca_shape(band, weighted, pg, critic, none, u_rec, rep):
    """The room's check (e) measurements on the paired BCA penalty (reported, not a stop: the detection marks gate the
    policy test): on the room's 2,000 full-throttle-region training states, thruster 1 of the no-caution actor's
    command set along its push to each throttle of the room's grid."""
    rows = H.shape_rows(pg, rep)
    obs = np.asarray(pg.train.obs)[rows]
    base = L.actor_rows(none, obs).astype(np.float64)
    sign = np.where(base[:, 0] < 0, -1.0, 1.0)
    ur = np.asarray(u_rec, np.float64)[rows]
    Wv, Q1 = {}, {}
    for m in H.SHAPE_GRID:
        aa = base.copy()
        aa[:, 0] = sign * (TC.KNEE if m == 0.80 else m)
        Wv[m] = np.maximum(B.width_rows(CAL, band, obs, aa, weighted) - ur, 0.0)
        Q1[m] = np.asarray(L.critic_rows(critic, obs, aa.astype(np.float32))[0][:, 0], np.float64)
    grid = np.array(H.SHAPE_GRID)
    q_best = grid[np.argmax(np.stack([Q1[m] for m in H.SHAPE_GRID], 1), 1)]
    best = grid[np.argmax(np.stack([Q1[m] - Wv[m] for m in H.SHAPE_GRID], 1), 1)]
    return dict(rows=int(len(rows)), mean_W={f"{m:g}": float(Wv[m].mean()) for m in H.SHAPE_GRID},
                fall_share_80_to_81=float(np.mean(Wv[0.81] < Wv[0.80])),
                rise_share_80_to_81=float(np.mean(Wv[0.81] > Wv[0.80])),
                mean_W_full_minus_80=float(np.mean(Wv[1.0] - Wv[0.80])),
                best_past80_share=float(np.mean(best > 0.8)), q1_alone_best_past80_share=float(np.mean(q_best > 0.8)))


# --------------------------------------------------------------------------------------------- phases

def seed_dir(rep, sub=None, paths=None):
    d = (paths or C.PATHS).out / "policy" / f"seed{int(rep)}"
    return d / sub if sub else d


def detect_dir(rep, world, critic_kind, version, paths=None):
    return (paths or C.PATHS).out / "detect" / version / world / critic_kind / f"seed{int(rep)}"


def stop(path, rep, reason, approval, **fields):
    C.write_json(path, dict(label=C.LABEL_LONG, kind="stage2-stop", rep=int(rep), stopped=True, reason=reason,
                            approval_sha256=approval.file_sha256, prereg_sha256=approval.prereg_sha256,
                            written=time.strftime("%Y-%m-%dT%H:%M:%S%z"), **fields))
    C.log(f"STOP: {reason}")
    raise SystemExit(2)


def save_and_lock(d, actors, payload):
    names = sorted(actors)
    file_sha = C.save_tree(d / "actors.msgpack", {k: actors[k] for k in names})
    payload = dict(payload, names=names, actor_sha256={k: L.tree_sha256(actors[k]) for k in names},
                   actors_file="actors.msgpack", actors_file_sha256=file_sha,
                   written=time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                   about=("the lock: every actor trained and every control matched in strength, written before any "
                          "return, points or score of any trained arm is computed"))
    return C.write_json(d / "lock.json", payload)


def verify_lock(d, approval):
    lock = C.read_json(d / "lock.json")
    if lock.get("prereg_sha256") != approval.prereg_sha256:
        raise SystemExit("refused: the lock was written under another pre-registration")
    acts = C.load_tree(d / lock["actors_file"], {k: L.actor_template() for k in lock["names"]},
                       lock["actors_file_sha256"])
    bad = [k for k in lock["names"] if L.tree_sha256(acts[k]) != lock["actor_sha256"][k]]
    if bad:
        raise SystemExit(f"refused: actors differ from the lock: {bad}")
    return lock, acts, C.file_sha256(d / "lock.json")


def arms_record(trained):
    return {k: dict(arm=v["arm"], finite=v["finite"], chunk_metrics=v["chunk_metrics"], refits=v.get("refits"),
                    metric_names=H.METRIC_NAMES) for k, v in trained.items()}


def strengths_of(trained, none, prepared):
    S, SR = {}, {}
    for name, v in trained.items():
        S[name] = L.strength(v["params"], none, prepared.eval_obs) if v["finite"] else float("nan")
        SR[name] = H.region_strength(v["params"], none, prepared) if v["finite"] else float("nan")
    return S, SR


def room_scales(pg, critic, shared, clone, rep, cfg, nets):
    """The room's own ladder scales (slope normalisation at the cloned actor's commands, room harness functions):
    the controls' ladders do not depend on BCA."""
    idx = H.slope_sample(pg, rep)
    a_clone = L.actor_rows(clone, np.asarray(pg.train.obs)[idx])
    g = dict(band=H.slope_rms("band", cfg, critic, shared, idx, a_clone),
             push_less=H.slope_rms("push_less", cfg, critic, shared, idx, a_clone),
             random_1=H.slope_rms("random", cfg, critic, shared, idx, a_clone, net=nets[1][0]),
             random_2=H.slope_rms("random", cfg, critic, shared, idx, a_clone, net=nets[2][0]))
    scale = {"even_pull": 1.0, "push_less": g["band"] / g["push_less"], "random_1": g["band"] / g["random_1"],
             "random_2": g["band"] / g["random_2"]}
    return g, scale


def train(rep, approval, version, run_condition, best_standin, log_fn=C.log, paths=None):
    """Stalling world, one seed: train every arm, check (b), match every control and the best stand-in blind to
    bca_k1's strength, lock. No return."""
    C.recheck(approval)
    prereg = C.load_prereg(paths)
    t0 = time.perf_counter()
    d = seed_dir(rep, paths=paths)
    if (d / "lock.json").exists() or (d / "stop.json").exists():
        raise SystemExit(f"refused: {d} is already locked or stopped")
    weighted = version == "weighted"
    kind = "bca_w" if weighted else "bca_u"
    from fresh import context
    ctx = context(rep)
    pg = ctx.prepared()
    critic = ctx.critic()[0]
    clone = ctx.clone()[0]
    none = ctx.none()
    d_band = detect_dir(rep, "stall", "fixed", "uniform", paths)
    band_u, brec = B.load_band(d_band, approval, TC.DX, TC.DA)
    if band_u is None:
        stop(d / "stop.json", rep, "the uniform band fit failed on this seed", approval)
    cfg = H.ProgramConfig()
    shared, sinfo = H.build_shared(pg, critic, rep, cfg=cfg, need_harm=False, log_fn=log_fn)
    refit = None
    if weighted:
        refit = B.weighted_refitter(ctx, "stall", pg, band_u, brec, d_band, approval)
        shared["band"], shared["u_rec"] = band_u, jnp.zeros(pg.n, jnp.float32)   # unused by bca_w
        shape_band, u_rec_none, _ = refit(none)
    else:
        u_rec_none = B.width_rows(CAL, band_u, np.asarray(pg.train.obs), pg.actions, False)
        shared["band"], shared["u_rec"] = band_u, jnp.asarray(u_rec_none, jnp.float32)
        shape_band = band_u
    shape = bca_shape(shape_band, weighted, pg, critic, none, u_rec_none, rep)
    nets = {j: H.random_surface_params(rep, j) for j in (1, 2)}
    g, scale = room_scales(pg, critic, shared, clone, rep, cfg, nets)
    runner = BCARunner(pg, clone, critic, ctx.learn, rep, cfg=cfg, shared=shared,
                       random_nets={j: nets[j][0] for j in (1, 2)}, refit=refit)
    trained = runner.run([bca_arm("w0_check", kind, (0, 0, 0))], log_fn)
    w0 = H.max_param_diff(trained["w0_check"]["params"], none)
    check_b = dict(max_abs_diff=w0, tol=W0_TOL, passed=bool(trained["w0_check"]["finite"] and w0 <= W0_TOL))
    if not check_b["passed"]:
        stop(d / "stop.json", rep, "check (b) failed: the BCA program with k = 0 differs from check 2's actor",
             approval, check_b=check_b)
    arms = [bca_arm("bca_k1", kind, (1, 0, 0)), bca_arm("bca_k2", kind, (2, 0, 0)),
            bca_arm("bca_unpaired", kind, (0, 1, 0)), bca_arm("bca_shuffled", kind, (0, 0, 1)),
            H.RoomArm("perfect_band", "band", H.MAIN_BAND), H.RoomArm("range_hinge", "hinge", (1.0,))]
    arms += [H.family_arm("even_pull", nu, scale) for nu in H.EVEN_LADDER]
    arms += [H.family_arm(f, k, scale) for f in ("push_less", "random_1", "random_2") for k in H.SLOPE_LADDER]
    trained.update(runner.run(arms, log_fn))
    target = float(L.strength(trained["bca_k1"]["params"], none, pg.eval_obs)) if trained["bca_k1"]["finite"] \
        else float("nan")
    if np.isfinite(target) and target > 0:
        matches = match_controls(runner, trained, none, pg, target, scale, MATCHED, log_fn)
    else:
        matches = {f: dict(record=None, arm=None, matched=False, reason="bca_k1 failed or has no positive strength")
                   for f in MATCHED}
    standin = dict(family=best_standin, matched=False, arm=None)
    s_actors = {}
    if np.isfinite(target) and target > 0 and best_standin:
        try:
            S = standins_module(prereg)
            s_trained, s_match, s_info = standin_matched(S, ctx, "stall", pg, critic, clone, none, best_standin,
                                                         target, shared, cfg, log_fn)
            standin.update(record=s_match["record"], matched=s_match["matched"],
                           arm=None if s_match["arm"] is None else f"standin:{s_match['arm']}",
                           coefficient=None if s_match["arm"] is None else
                           list(s_trained[s_match["arm"]]["arm"]["coef"]),
                           data_side=s_info["info"], scale=s_info["scale"])
            s_actors = {f"standin:{k}": v["params"] for k, v in s_trained.items() if v["finite"]}
        except Exception as error:  # a rival that cannot run fails every mark that needs it (counts against BCA)
            standin.update(error=repr(error))
    S_, SR = strengths_of(trained, none, pg)
    actors = {k: v["params"] for k, v in trained.items()}
    actors.update(s_actors)
    actors.update(none=none, clone=clone, full_none=ctx.full_none())
    scored = ["none", "clone", "full_none", "bca_k1", "bca_k2", "bca_unpaired", "bca_shuffled", "perfect_band",
              "range_hinge"] + [m["arm"] for m in matches.values() if m["arm"]] + \
        ([standin["arm"]] if standin.get("arm") else [])
    payload = dict(label=C.LABEL_LONG, kind="stage2-policy-lock", rep=int(rep), world="stall", version=version,
                   role=ctx.role, prereg_sha256=approval.prereg_sha256, approval_sha256=approval.file_sha256,
                   run_condition=run_condition, band_record_sha256=C.file_sha256(d_band / "band.json"),
                   checks=dict(b=check_b), bca_penalty_shape=shape, arms=arms_record(trained), strengths=S_,
                   region_strengths=SR, target_strength=target, matches=matches,
                   matched_names={f: m["arm"] for f, m in matches.items()}, standin=standin, scored_names=scored,
                   slope_rms=g, coefficient_per_knob=scale, random_surface_seeds={j: nets[j][1] for j in (1, 2)},
                   shared=sinfo, seconds=dict(total=time.perf_counter() - t0, by_kind=runner.seconds))
    save_and_lock(d, actors, payload)
    log_fn(f"policy seed {rep}: locked ({time.perf_counter() - t0:.0f}s)")


def thruster_shares(params, prepared, physics):
    pol = L.mlp_policy(params, prepared)
    out = {}
    for s in SITUATIONS:
        run = TR.rollout(pol, TC.bank_starts(s, TC.BANK_SIZE), physics, record=True)
        out[s] = (np.abs(run.commands) > physics.knee).mean(axis=(0, 1)).tolist()
    return out


def shift_cards(ctx, world, names, acts, prepared):
    """Reported: the gate's shift cards (where each actor goes: share of time outside the recorded conditions, share
    past 80%, distance of visited pairs from the data) with the report-card bug fixed: shift.shift_card receives the
    anchors of ONE situation (anchors[situation]), not the table of all four (the gate's acceptance.report_cards
    passed the whole table and raised KeyError: 'top')."""
    import experiments.shift.shift as SH
    sd = ctx.sd[world]
    recorder = RC.Recorder(sd.spec, sd.physics)
    ref = SH.card_reference(sd.log, sd.removals["gap"], recorder)
    an = ctx.anchors(world)
    cfg = ctx.cfg
    out = {}
    for n in names:
        pol = L.mlp_policy(acts[n], prepared, n)
        out[n] = {s: SH.shift_card(pol, s, sd.physics, ref, an[s], ms_states=cfg.card_ms_states,
                                   knn_states=cfg.card_knn_states) for s in SITUATIONS}
    return out


def evaluate(rep, approval, world="stall", log_fn=C.log, paths=None):
    """Verify the lock, then exact returns of the scored actors (acceptance._evaluate: full 1,000-start banks of the
    four situations, points against the gate's anchors, capped at 80%, share past 80%), per-thruster shares and
    (reported) shift cards of none and bca_k1 -> values.json, once."""
    C.recheck(approval)
    d = seed_dir(rep, None if world == "stall" else "no_stall", paths)
    if (d / "values.json").exists() or (d / "stop.json").exists():
        raise SystemExit(f"refused: {d} has values.json or stop.json; an evaluation is computed once")
    lock, acts, lock_sha = verify_lock(d, approval)
    from fresh import context
    ctx = context(rep, worlds=("stall", "no_stall"))
    prep = ctx.prepared(world)
    an = ctx.anchors(world)
    t0 = time.perf_counter()
    ev, shares = {}, {}
    for name in lock["scored_names"]:
        finite = name in ("none", "clone", "full_none") or lock["arms"].get(name, {}).get("finite", True)
        ev[name] = A._evaluate(acts[name], prep, an, ctx.cfg, failed=not finite)
        shares[name] = thruster_shares(acts[name], prep, prep.physics) if finite else None
        log_fn(f"evaluated {name}")
    cards = shift_cards(ctx, world, [n for n in ("none", "bca_k1") if n in acts], acts, prep)
    values = {s: dict(points={k: (None if ev[k] is None else ev[k][s]["points"]) for k in ev},
                      unrecorded_throttle_share={k: None if ev[k] is None else ev[k][s]["unrecorded_throttle_share"]
                                                 for k in ev},
                      capped_points={k: None if ev[k] is None else ev[k][s]["capped_points"] for k in ev},
                      past_80_share_per_thruster={k: None if shares[k] is None else shares[k][s] for k in shares})
              for s in SITUATIONS}
    C.write_json(d / "values.json", dict(label=C.LABEL_LONG, kind="stage2-policy-values", rep=int(rep), world=world,
                                         prereg_sha256=approval.prereg_sha256, lock_sha256=lock_sha,
                                         matched_names=lock.get("matched_names"), standin=lock.get("standin"),
                                         evaluations=ev, situations=values, shift_cards=cards,
                                         seconds=time.perf_counter() - t0,
                                         about="exact returns of the locked actors, computed after the lock"))


def train_no_stall(rep, approval, version, log_fn=C.log, paths=None):
    """The no-stall comparison world at the stalling world's locked coefficients (not re-matched): none (check 2's
    no-stall gapped actor), bca_k1 with the no-stall world's own band (fitted on its gapped critic), the even pull at
    the locked nu, the range hinge, the perfect band (stall-aware next command) and the best stand-in at its locked
    coefficient. The price of caution where the missing commands help."""
    C.recheck(approval)
    prereg = C.load_prereg(paths)
    t0 = time.perf_counter()
    d = seed_dir(rep, "no_stall", paths)
    if (d / "lock.json").exists() or (d / "stop.json").exists():
        raise SystemExit(f"refused: {d} is already locked or stopped")
    stall_lock, _, stall_lock_sha = verify_lock(seed_dir(rep, paths=paths), approval)
    weighted = version == "weighted"
    kind = "bca_w" if weighted else "bca_u"
    from fresh import context
    ctx = context(rep, worlds=("stall", "no_stall"))
    pt = ctx.prepared("no_stall")
    critic = ctx.critic("no_stall", "gap")[0]
    clone = ctx.clone("no_stall")[0]
    none = ctx.none("no_stall")
    d_band = detect_dir(rep, "no_stall", "fixed", "uniform", paths)
    band_u, brec = B.load_band(d_band, approval, TC.DX, TC.DA)
    if band_u is None:
        stop(d / "stop.json", rep, "the no-stall band fit failed on this seed", approval)
    cfg = H.ProgramConfig(physics=pt.physics, next_command="stall_aware")
    shared, sinfo = H.build_shared(pt, critic, rep, cfg=cfg, need_harm=False, log_fn=log_fn)
    refit = None
    if weighted:
        refit = B.weighted_refitter(ctx, "no_stall", pt, band_u, brec, d_band, approval)
        shared["band"], shared["u_rec"] = band_u, jnp.zeros(pt.n, jnp.float32)
    else:
        shared["band"] = band_u
        shared["u_rec"] = jnp.asarray(B.width_rows(CAL, band_u, np.asarray(pt.train.obs), pt.actions, False),
                                      jnp.float32)
    runner = BCARunner(pt, clone, critic, ctx.learn, rep, cfg=cfg, shared=shared, refit=refit)
    em = stall_lock["matches"]["even_pull"]
    nu = None if not em["matched"] else float(stall_lock["arms"][em["arm"]]["arm"]["knob"])
    arms = [bca_arm("w0_check", kind, (0, 0, 0)), bca_arm("bca_k1", kind, (1, 0, 0)),
            H.RoomArm("perfect_band", "band", H.MAIN_BAND), H.RoomArm("range_hinge", "hinge", (1.0,))]
    if nu is not None:
        arms.append(H.RoomArm("even_pull_locked", "plain", (0.0,), 1.0 + nu, family="even_pull", knob=nu))
    trained = runner.run(arms, log_fn)
    w0 = H.max_param_diff(trained["w0_check"]["params"], none)
    check_b = dict(max_abs_diff=w0, tol=W0_TOL, passed=bool(trained["w0_check"]["finite"] and w0 <= W0_TOL))
    if not check_b["passed"]:
        stop(d / "stop.json", rep, "check (b) failed in the no-stall world", approval, check_b=check_b)
    s_actors, standin = {}, dict(stall_lock["standin"])
    if standin.get("coefficient") is not None:
        try:
            S = standins_module(prereg)
            shared_s, info_s, extra = S.build_shared(ctx, "no_stall", pt, critic, clone, none, rep, log_fn)
            sr = S.StandinRunner(pt, clone, critic, ctx.learn, rep, shared=shared_s)
            st = sr.run([S.locked_arm(standin["family"], standin["coefficient"], name="standin_locked")], log_fn)
            s_actors = {"standin_locked": st["standin_locked"]["params"]} if st["standin_locked"]["finite"] else {}
            standin["no_stall_data_side"] = info_s
        except Exception as error:
            standin["no_stall_error"] = repr(error)
    S_, SR = strengths_of(trained, none, pt)
    actors = {k: v["params"] for k, v in trained.items()}
    actors.update(s_actors, none=none, clone=clone)
    scored = ["none", "clone", "bca_k1", "perfect_band", "range_hinge"] + (["even_pull_locked"] if nu is not None
                                                                          else []) + list(s_actors)
    payload = dict(label=C.LABEL_LONG, kind="stage2-policy-lock-no-stall", rep=int(rep), world="no_stall",
                   version=version, role=ctx.role, prereg_sha256=approval.prereg_sha256,
                   approval_sha256=approval.file_sha256, stall_lock_sha256=stall_lock_sha, locked_nu=nu,
                   band_record_sha256=C.file_sha256(d_band / "band.json"), checks=dict(b=check_b),
                   arms=arms_record(trained), strengths=S_, region_strengths=SR, scored_names=scored, standin=standin,
                   shared=sinfo, seconds=dict(total=time.perf_counter() - t0, by_kind=runner.seconds))
    save_and_lock(d, actors, payload)


# --------------------------------------------------------------------------------------------- pooled marks

def _mean(xs):
    return None if not xs or any(x is None for x in xs) else float(np.mean(xs))


def _d(a, b):
    return None if a is None or b is None else a - b


def load_values(rep, world, approval, paths=None):
    """('values', v) | ('stopped', s) | ('missing', None), each verified: sidecar, label, kind, pre-registration and
    the lock it names. A fresh seed (5-9) with neither file but a recorded data or learner failure
    (fresh.recorded_failure) is 'stopped': it fails every mark and is never replaced (pre-registration,
    confirmation.failed_seed), so the confirmatory pooled look can still be written."""
    d = seed_dir(rep, None if world == "stall" else "no_stall", paths)
    v = C.read_json(d / "values.json", required=False)
    st = C.read_json(d / "stop.json", required=False)
    if v is not None and st is not None:
        raise SystemExit(f"refused: {d} has both values.json and stop.json")
    if v is not None:
        if v.get("kind") != "stage2-policy-values" or v.get("prereg_sha256") != approval.prereg_sha256 \
                or v.get("lock_sha256") != C.file_sha256(d / "lock.json"):
            raise SystemExit(f"refused: {d}/values.json is not this pre-registration's output for its lock")
        return "values", v
    if st is not None:
        if st.get("prereg_sha256") != approval.prereg_sha256:
            raise SystemExit(f"refused: {d}/stop.json names another pre-registration")
        return "stopped", st
    from fresh import recorded_failure
    why = recorded_failure(rep, paths)
    if why is not None:
        return "stopped", dict(stopped=True, reason=why, recorded_failure=True)
    return "missing", None


def policy_marks(seeds, approval, deploy_values, paths=None):
    """Plan 5.3's marks (strong current unless stated), each on the pooled mean over `seeds` of the paired per-seed
    difference; every seed is needed (a missing value, a stopped seed or an unmatched control fails the mark):
      M1 bca_k1 - matched even pull >= 15, and positive in at least 4 of 5 seeds
      M2 bca_k1 - each matched random surface >= 15
      M3 keeps the improvement: recorded bca_k1 - none >= -3 and bca_k1 - clone >= 25; big waves and big kicks
         bca_k1 - none >= -3
      M4 bca_k1 - the matched best stand-in >= -5
      M5 the deployment check on the no-caution actor beats random switching at its own rate by >= 10, and costs at
         most 2 in recorded conditions (check - never >= -2)
    Reported: every arm's no-stall cost, the claim sentence's numbers."""
    res = {rep: load_values(rep, "stall", approval, paths) for rep in seeds}
    if any(s == "missing" for s, _ in res.values()):
        raise SystemExit("refused: the pooled look needs every seed's values.json or stop.json")
    vals = {rep: (v if s == "values" else None) for rep, (s, v) in res.items()}

    def pts(rep, name, sit):
        v = vals[rep]
        if v is None or name is None:
            return None
        return v["situations"][sit]["points"].get(name)

    def matched(rep, fam):
        v = vals[rep]
        return None if v is None else (v.get("matched_names") or {}).get(fam)

    def col(fn):
        return [fn(rep) for rep in seeds]

    sc = "strong_current"
    out = dict(seeds=list(seeds), status={rep: s for rep, (s, _) in res.items()})
    e = col(lambda r: _d(pts(r, "bca_k1", sc), pts(r, matched(r, "even_pull"), sc)))
    out["M1_even_pull"] = dict(values=e, mean=_mean(e), positive=sum(1 for x in e if x is not None and x > 0),
                               passed=bool(_mean(e) is not None and _mean(e) >= 15 and
                                           sum(1 for x in e if x is not None and x > 0) >= 4))
    m2 = {}
    for f in ("random_1", "random_2"):
        x = col(lambda r, f=f: _d(pts(r, "bca_k1", sc), pts(r, matched(r, f), sc)))
        m2[f] = dict(values=x, mean=_mean(x), passed=bool(_mean(x) is not None and _mean(x) >= 15))
    out["M2_random_surfaces"] = dict(m2, passed=all(v["passed"] for v in m2.values()))
    rn = col(lambda r: _d(pts(r, "bca_k1", "recorded"), pts(r, "none", "recorded")))
    rc = col(lambda r: _d(pts(r, "bca_k1", "recorded"), pts(r, "clone", "recorded")))
    keeps = dict(recorded_vs_none=dict(values=rn, mean=_mean(rn)), recorded_vs_clone=dict(values=rc, mean=_mean(rc)))
    ok = _mean(rn) is not None and _mean(rc) is not None and _mean(rn) >= -3 and _mean(rc) >= 25
    for s in ("big_waves", "big_kicks"):
        x = col(lambda r, s=s: _d(pts(r, "bca_k1", s), pts(r, "none", s)))
        keeps[f"{s}_vs_none"] = dict(values=x, mean=_mean(x))
        ok = ok and _mean(x) is not None and _mean(x) >= -3
    out["M3_keeps_improvement"] = dict(keeps, passed=bool(ok))
    sd = col(lambda r: _d(pts(r, "bca_k1", sc),
                          pts(r, (vals[r] or {}).get("standin", {}).get("arm") if vals[r] else None, sc)))
    out["M4_best_standin"] = dict(values=sd, mean=_mean(sd), passed=bool(_mean(sd) is not None and _mean(sd) >= -5))
    dv = deploy_values or {}
    gain = [dv.get(rep, {}).get("none", {}).get(sc, {}).get("check_minus_random") for rep in seeds]
    cost = [dv.get(rep, {}).get("none", {}).get("recorded", {}).get("check_minus_never") for rep in seeds]
    out["M5_deployment_check"] = dict(gain=gain, gain_mean=_mean(gain), cost=cost, cost_mean=_mean(cost),
                                      passed=bool(_mean(gain) is not None and _mean(gain) >= 10 and
                                                  _mean(cost) is not None and _mean(cost) >= -2))
    out["passed"] = bool(all(out[k]["passed"] for k in ("M1_even_pull", "M2_random_surfaces",
                                                         "M3_keeps_improvement", "M4_best_standin",
                                                         "M5_deployment_check")))
    ns = {rep: load_values(rep, "no_stall", approval, paths) for rep in seeds}

    def nspts(rep, name):
        s, v = ns[rep]
        return None if s != "values" else v["situations"][sc]["points"].get(name)

    harm = col(lambda r: _d(pts(r, "full_none", sc), pts(r, "none", sc)))
    X = col(lambda r: _d(pts(r, "bca_k1", sc), pts(r, "none", sc)))
    Yn = col(lambda r: _d(pts(r, "bca_k1", "recorded"), pts(r, "clone", "recorded")))
    Yd = col(lambda r: _d(pts(r, "none", "recorded"), pts(r, "clone", "recorded")))
    Z = col(lambda r: _d(pts(r, "range_hinge", sc), pts(r, "none", sc)))
    W_ = col(lambda r: _d(nspts(r, "none"), nspts(r, "bca_k1")))
    V = col(lambda r: _d(nspts(r, "none"), nspts(r, "range_hinge")))
    cl = dict(harm=_mean(harm), X=_mean(X), Y=None if _mean(Yn) is None or not _mean(Yd) else _mean(Yn) / _mean(Yd),
              Z=_mean(Z), W=_mean(W_), V=_mean(V))
    out["claim_numbers"] = cl
    out["claim_sentence"] = claim_sentence(cl)
    out["no_stall_cost"] = {name: _mean(col(lambda r, n=name: _d(nspts(r, "none"), nspts(r, n))))
                            for name in ("bca_k1", "perfect_band", "range_hinge", "even_pull_locked", "standin_locked")}
    return out


def claim_sentence(cl):
    """Plan 5.5's wording, fixed in the pre-registration; filled only with pooled numbers."""
    def f(x):
        return "n/a" if x is None else "%.1f" % x

    y = "n/a" if cl["Y"] is None else "%.0f%%" % (100.0 * cl["Y"])
    return ("In strong current, BCA's lower edge recovers %s of %s points and keeps %s of TD3+BC's gain over cloning "
            "in the recorded conditions. A hand-set limit to the recorded range recovers %s. In the no-stall world, "
            "where the missing commands help, BCA costs %s points and the range limit %s."
            % (f(cl["X"]), f(cl["harm"]), y, f(cl["Z"]), f(cl["W"]), f(cl["V"])))

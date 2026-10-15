"""Return test of the command-weighted lower edge (EXPLORATORY; pass marks frozen in
channel/PASSMARKS_training_test.md, SHA-256 in its sidecar; the owner's approval: "Approve as drafted (Recommended)").

Arms, each actor-only against the frozen gapped critic from the cloned actor, with version 5's lower-edge protocol
(lower_edge._loop and lower_edge.actor_step unchanged; only the penalty differs):
  cmd_strong   W = [Q1(s, a) - Q1(s, a_i)]_+ where the band refuses the command (no value credit over the recorded
               command), else [U(s, a) - U(s, a_i)]_+ with U = R(omega(s, a)) x unit x scale(s, a); omega the exact
               ratio of command_band_check.py (strong law), the actor's spread centred on the current actor
  cmd_train    the same with w(c) = 1 (the training law)
  untargeted   no value credit on random rows, as many per chunk as cmd_strong refused at the same chunk and seed;
               no band penalty otherwise
The bank weights, the table of R and U(s, a_i) are recomputed from the current actor at the start of every chunk.

Usage (from /root/bca-shift, PYTHONPATH=/root/bca-shift, JAX_PLATFORMS=cpu):
    python command_le_train.py checks <quality> <seed>
    python command_le_train.py train <quality> <seed>
    python command_le_train.py evaluate <quality> <seed>
"""
import hashlib
import importlib.util
import json
import math
import sys
import time
from functools import partial
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from flax.training.train_state import TrainState

from experiments.shift import learners as LR
from experiments.shift import thruster_chain as TC
from experiments.shift_iw import data as D
from experiments.shift_iw import learners_iw as LI
from experiments.shift_iw import lower_edge as LE
from experiments.shift_iw import run as RN
from experiments.shift_iw import world as W

HERE = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location("cbc", HERE / "command_band_check.py")
CB = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(CB)

OUT = RN.ROOT_DEFAULT.parent / "shift_iw_v5_explore" / "channel" / "train"
LABEL = ("EXPLORATORY: outside the version-5 pre-registration; pass marks frozen in "
         "channel/PASSMARKS_training_test.md before this code was written")
PASSMARKS = HERE / "PASSMARKS_training_test.md"
ARMS = ("cmd_strong", "cmd_train", "untargeted")
LAWS = {"cmd_strong": "strong", "cmd_train": "train"}
MAXG = 4096
TAG_UNTARGETED = 4243
LOGK0 = float(TC.DA * (CB.LOG_PHI0 - math.log(CB.SPREAD)))   # log pi_h at its own centre
STAGE = "bca_lower_edge"


def passmarks_ok():
    want = (HERE / "PASSMARKS_training_test.md.sha256").read_text().split()[0]
    got = hashlib.sha256(PASSMARKS.read_bytes()).hexdigest()
    if want != got:
        raise SystemExit(f"the frozen pass marks changed: {got} != {want}")
    return got


def this_code():
    return {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in (Path(__file__).resolve(), HERE / "command_band_check.py")}


# ------------------------------------------------------------------------------------------------ in-graph pieces

def recorder_logdensity_jnp(a, mu, eps, hard):
    """log beta_eps(a | s) in float32, as command_band_check.recorder_logdensity (a, mu: (n, 4); eps, hard: (n,))."""
    cap, s = REC_CAP, REC_WOBBLE
    inside = jnp.all(jnp.abs(a) < cap, axis=-1)
    r = jnp.clip(a / cap, -1.0 + 1e-6, 1.0 - 1e-6)
    z = (cap * jnp.arctanh(r) - mu) / s
    lp = jnp.sum(CB.LOG_PHI0 - 0.5 * z * z - math.log(s) - jnp.log1p(-r * r), axis=-1)
    lp = jnp.where(inside, lp, -jnp.inf)
    p_norm = (1.0 - eps) * (1.0 - HOLD_P * hard)
    t1 = jnp.where(p_norm > 0, jnp.log(jnp.maximum(p_norm, 1e-30)) + lp, -jnp.inf)
    t2 = jnp.where(eps > 0, jnp.log(jnp.maximum(eps, 1e-30) / 16.0), -jnp.inf)
    return jnp.logaddexp(t1, t2)


REC_CAP = CB.REC.spec.normal_driving_cap
REC_WOBBLE = CB.REC.spec.wobble
HOLD_P = CB.REC.spec.hold_probability


def band_read(arm, idx, a):
    """(R, refused) at rows idx and commands a, in-graph (no gradient)."""
    a0 = jax.lax.stop_gradient(a)
    lb = recorder_logdensity_jnp(a0, arm["mu"][idx], arm["eps"][idx], arm["hard"][idx])
    lm = arm["log_w"][idx] + LOGK0 + arm["log_Z"][idx] - lb
    in_ball = jnp.sum(jnp.square(a0 - arm["centre"]), axis=-1) <= arm["rho2"]
    m = jnp.exp(jnp.clip(lm, arm["log_floor"], 80.0))
    gi = jnp.searchsorted(arm["grid"], m, side="left")
    R = arm["thresholds"][jnp.minimum(gi, MAXG - 1)]
    refused = in_ball | jnp.isposinf(lm) | (gi >= arm["n_grid"]) | ~jnp.isfinite(R)
    return jnp.where(refused, 1.0, R), refused


def penalty_band(critic_params, shared, idx, obs, a, arm):
    """Arms cmd_strong / cmd_train: no value credit where refused, else the paired band penalty."""
    q1 = LR.CRITIC.apply(critic_params, obs, a)[..., 0]
    w_ref = jnp.maximum(q1 - arm["q1_rec"][idx], 0.0)
    R, refused = band_read(arm, idx, a)
    U = LE.band_width(shared["cal_params"], shared["unit"], obs, a, R)
    w_cert = jnp.maximum(U - jax.lax.stop_gradient(arm["u_logged"][idx]), 0.0)
    return jnp.where(refused, w_ref, w_cert)


def penalty_untargeted(critic_params, shared, idx, obs, a, arm):
    """Arm untargeted: no value credit on the chunk's randomly chosen rows, nothing elsewhere."""
    q1 = LR.CRITIC.apply(critic_params, obs, a)[..., 0]
    return jnp.where(arm["no_credit"][idx], jnp.maximum(q1 - arm["q1_rec"][idx], 0.0), 0.0)


_PROGRAMS = {}


def program(args, kind):
    key = (kind, LR._args_key(args))
    if key not in _PROGRAMS:
        pen = penalty_band if kind == "band" else penalty_untargeted
        _PROGRAMS[key] = jax.jit(partial(LE._loop, args, pen))
    return _PROGRAMS[key]


# ------------------------------------------------------------------------------------------------ one seed's world

class World:
    """One quality and seed: data, gapped host, frozen band, bank scores, the exact-ratio ingredients."""

    def __init__(self, quality, seed):
        self.ctx = RN.Context(quality=quality)
        self.spec = RN.spec_digest(self.ctx)
        self.seed = int(seed)
        ctx, spec = self.ctx, self.spec
        self.data = RN.load_data(ctx, seed, spec)
        self.prep = D.prepare(self.data, "gap", ctx.cfg)
        self.host = RN.load_host_state(ctx, "gap", seed, spec)
        self.brec = RN.require_band(ctx, "gap", seed, spec)
        self.approval = RN.approval_for(ctx, STAGE)
        self.band = LI.load_band(RN.band_dir(ctx, "gap", seed) / "band.msgpack", self.brec["band_sha256"], self.host,
                                 self.prep, ctx.cfg, seed, self.approval, self.brec["band"], stage=STAGE)
        self.R_u = RN.num(self.brec["thresholds"]["unweighted"]["threshold"])
        self.shared = LE.band_shared(self.band, STAGE)
        log, gap = self.data.train_log, self.data.gap
        self.gap = gap
        rng = lambda part: np.random.default_rng([W.MASTER_SEED, self.seed, CB.TAG, part])
        consts = CB.ball_constants(gap, rng(0))
        self.bank = CB.Rows(log, self.prep.bank_rows, gap, consts, rng(1))
        self.train = CB.Rows(log, self.prep.rows, gap, consts, rng(2))
        self.a_bank = np.asarray(log["actions"][self.prep.bank_rows], np.float32).astype(np.float64)
        self.hold_bank = np.asarray(log.diagnostic("infos/hold")[self.prep.bank_rows], bool)
        self.scores, self.stream = LI.bank_scores(self.band, self.prep.bank, ctx.cfg)
        self.obs = np.asarray(self.prep.train.obs, np.float32)
        self.a_rec = np.asarray(self.prep.actions, np.float64)
        self.q1_rec = np.asarray(LR.critic_rows(self.host.critic.params, self.obs,
                                                self.prep.actions.astype(np.float32))[0][:, 0], np.float32)
        self.args = LR.make_args(ctx.cfg, seed, ctx.cfg.alpha)

    def log_ratio(self, rows, idx, a, centre, law):
        """log omega under a law ('strong' or 'train') at rows idx of a Rows object."""
        lw = rows.log_ratio(idx, a, centre)
        if law == "train":
            lw = lw - np.log(W.ratio(rows.c[idx], CB.TILT))
        return lw

    def refresh(self, actor_params, law):
        """The arm's numbers from the current actor: bank weights, the table, U(s, a_i), and the refusal share at the
        actor's own commands over every kept training row (numpy, as command_band_check)."""
        pi_bank = np.asarray(LR.actor_rows(actor_params, np.asarray(self.prep.bank.obs, np.float32)), np.float64)
        lw_bank = self.log_ratio(self.bank, np.arange(len(self.a_bank)), self.a_bank, pi_bank, law)
        w_bank = np.where(self.hold_bank | ~np.isfinite(lw_bank), 0.0, np.exp(np.clip(lw_bank, -700, 700)))
        th = CB.Thresholds(self.scores, self.stream, w_bank)
        if len(th.table.grid) > MAXG:
            raise SystemExit(f"the table has {len(th.table.grid)} grid points, more than {MAXG}")
        n = self.prep.n
        idx = np.arange(n)
        a_pi = np.asarray(LR.actor_rows(actor_params, self.obs), np.float64)
        lm_rec = np.where(self.gap.contains(self.a_rec), np.inf, self.log_ratio(self.train, idx, self.a_rec, a_pi, law))
        R_rec, _ = th.read(lm_rec)
        u_rec = LE.logged_widths(self.band, self.prep, LE.as_float32_thresholds(R_rec), stage=STAGE)
        a32 = a_pi.astype(np.float32).astype(np.float64)
        lm_pi = np.where(self.gap.contains(a32), np.inf, self.log_ratio(self.train, idx, a32, a32, law))
        _, ref_pi = th.read(lm_pi)
        grid = np.full(MAXG, np.inf, np.float32)
        grid[:len(th.table.grid)] = th.table.grid
        thr = np.full(MAXG, np.inf, np.float32)
        thr[:len(th.table.grid)] = th.table.thresholds
        log_w = np.log(W.ratio(self.train.c, CB.TILT)) if law == "strong" else np.zeros(n)
        tree = dict(k=jnp.float32(1.0), pull=jnp.ones(n, jnp.float32),
                    mu=jnp.asarray(CB.recorder_mean(self.train.x), jnp.float32),
                    eps=jnp.asarray(self.train.eps, jnp.float32), hard=jnp.asarray(self.train.hard, jnp.float32),
                    log_w=jnp.asarray(log_w, jnp.float32),
                    log_Z=jnp.asarray(np.log(np.maximum(self.train.Z, 1e-30)), jnp.float32),
                    centre=jnp.asarray(np.asarray(self.gap.centre), jnp.float32),
                    rho2=jnp.float32(float(self.gap.rho) ** 2), log_floor=jnp.float32(math.log(th.floor)),
                    grid=jnp.asarray(grid), thresholds=jnp.asarray(thr), n_grid=jnp.int32(len(th.table.grid)),
                    u_logged=jnp.asarray(u_rec, jnp.float32), q1_rec=jnp.asarray(self.q1_rec))
        past = np.asarray(TC.stalled(a_pi.astype(np.float32)), bool).any(1)
        info = dict(kish_share=CB.CV.kish(w_bank) / len(w_bank), cap_over_R_u=th.cap / self.R_u,
                    no_atom_over_R_u=th.no_atom / self.R_u, grid_points=int(len(th.table.grid)),
                    refused_share=float(ref_pi.mean()), past_80_share=float(past.mean()),
                    refused_share_past_80=float(ref_pi[past].mean()) if past.any() else None,
                    refused_share_within_80=float(ref_pi[~past].mean()))
        return tree, info, ref_pi

    def untargeted_tree(self, share, chunk):
        n = self.prep.n
        pick = np.random.default_rng([W.MASTER_SEED, self.seed, TAG_UNTARGETED, chunk]).random(n) < share
        return dict(k=jnp.float32(1.0), pull=jnp.ones(n, jnp.float32), no_credit=jnp.asarray(pick),
                    q1_rec=jnp.asarray(self.q1_rec)), float(pick.mean())

    def start(self):
        """The cloned actor every lower-edge arm starts from (learners.clone, as lower_edge.band_arms), made once."""
        if not hasattr(self, "_clone"):
            self._clone = LR.clone(self.prep, self.ctx.cfg, self.seed)
        return self._clone

    def rows(self):
        cfg = self.ctx.cfg
        steps, chunk = int(cfg.actor_only_steps), int(cfg.actor_only_chunk)
        rows = LR.rng_np(cfg, self.seed, LR.TAG_ACTOR_ONLY).integers(self.prep.n, size=(steps, cfg.batch_size))
        return jnp.asarray(rows.reshape(steps // chunk, chunk, cfg.batch_size), jnp.int32), LR.tree_sha256(rows)


# ------------------------------------------------------------------------------------------------ training

def arm_dir(quality, arm, seed):
    return OUT / quality / arm / f"seed{int(seed)}"


def train_arm(wd, arm, log):
    cfg = wd.ctx.cfg
    rows, rows_sha = wd.rows()
    clone_params, clone_rec = wd.start()
    state = TrainState.create(apply_fn=LR.ACTOR.apply, params=clone_params, tx=LR.adam(cfg.lr))
    chunks, metrics, t0 = [], [], time.perf_counter()
    match = None
    if arm == "untargeted":
        rec = json.loads((arm_dir(wd.ctx.quality, "cmd_strong", wd.seed) / "record.json").read_text())
        match = [c["refused_share"] for c in rec["chunks"]]
    for c in range(rows.shape[0]):
        if arm == "untargeted":
            tree, picked = wd.untargeted_tree(match[c], c)
            info = dict(target_share=match[c], picked_share=picked)
            prog = program(wd.args, "untargeted")
        else:
            tree, info, _ = wd.refresh(state.params, LAWS[arm])
            prog = program(wd.args, "band")
        state, aux = prog(wd.host.critic.params, state, wd.prep.train, wd.shared, rows[c], tree)
        aux = np.asarray(aux, np.float64)
        metrics.append(aux.mean(0).tolist())
        chunks.append(dict(chunk=c, **info, W_mean=float(aux[:, 4].mean()), q_mean=float(aux[:, 1].mean())))
        log(f"{wd.ctx.quality} seed {wd.seed} {arm}: chunk {c + 1}/{rows.shape[0]} " +
            ", ".join(f"{k} {v:.3f}" for k, v in info.items() if isinstance(v, float)) +
            f", W mean {chunks[-1]['W_mean']:.4f}")
    jax.block_until_ready(state.params)
    finite = bool(LR._finite_tree(state.params))
    d = arm_dir(wd.ctx.quality, arm, wd.seed)
    d.mkdir(parents=True, exist_ok=True)
    sha = LR.save_tree(d / "actor.msgpack", state.params)
    rec = dict(label=LABEL, exploratory=True, quality=wd.ctx.quality, seed=wd.seed, arm=arm,
               passmarks_sha256=passmarks_ok(), spec_sha256=wd.spec, code_sha256=LI.code_hashes(),
               this_code=this_code(), settings=dict(spread=CB.SPREAD, clip=CB.CLIP, grid_ratio=CB.GRID_RATIO,
                                                    floor=CB.FLOOR, law=LAWS.get(arm)),
               rows_sha256=rows_sha, clone=clone_rec, chunks=chunks, metric_names=list(LE.METRIC_NAMES),
               chunk_metrics=metrics, finite=finite, actor_sha256=sha, seconds=time.perf_counter() - t0,
               finished_at=time.strftime("%Y-%m-%dT%H:%M:%S%z"))
    (d / "record.json").write_text(json.dumps(rec, indent=1, default=float), encoding="utf-8")
    log(f"{wd.ctx.quality} seed {wd.seed} {arm}: finished in {rec['seconds']:.0f}s (finite {finite})")


# ------------------------------------------------------------------------------------------------ checks

def checks(quality, seed, log):
    """The pass marks' checks before training, on one quality and seed."""
    wd = World(quality, seed)
    out = {}
    # (1) unweighted, no refusals: the band penalty equals lower_edge.band_penalty, and a short run equals Runner's
    R32 = LE.as_float32_thresholds(np.full(wd.prep.n, wd.R_u))
    u_v5 = LE.logged_widths(wd.band, wd.prep, R32, stage=STAGE)
    n = wd.prep.n
    tree = dict(k=jnp.float32(1.0), pull=jnp.ones(n, jnp.float32), mu=jnp.zeros((n, TC.DA), jnp.float32),
                eps=jnp.ones(n, jnp.float32), hard=jnp.zeros(n, jnp.float32), log_w=jnp.zeros(n, jnp.float32),
                log_Z=jnp.zeros(n, jnp.float32), centre=jnp.full(TC.DA, 9.0, jnp.float32), rho2=jnp.float32(0.0),
                log_floor=jnp.float32(-80.0), grid=jnp.asarray(np.r_[np.float32(3e38), np.full(MAXG - 1, np.inf)],
                                                               jnp.float32),
                thresholds=jnp.asarray(np.r_[np.float32(wd.R_u), np.full(MAXG - 1, np.inf)], jnp.float32),
                n_grid=jnp.int32(1), u_logged=jnp.asarray(u_v5), q1_rec=jnp.asarray(wd.q1_rec))
    rows, _ = wd.rows()
    short = rows[0][:300]
    clone_params, _ = wd.start()
    st = TrainState.create(apply_fn=LR.ACTOR.apply, params=clone_params, tx=LR.adam(wd.ctx.cfg.lr))
    mine, _ = program(wd.args, "band")(wd.host.critic.params, st, wd.prep.train, wd.shared, short, tree)
    v5_arm = dict(k=jnp.float32(1.0), pull=jnp.ones(n, jnp.float32), R=jnp.asarray(R32), u_logged=jnp.asarray(u_v5))
    ref, _ = jax.jit(partial(LE._loop, wd.args, LE.band_penalty))(wd.host.critic.params, st, wd.prep.train, wd.shared,
                                                                  short, v5_arm)
    out["check1_max_param_diff_vs_v5_unweighted_300_steps"] = LE.max_param_diff(mine.params, ref.params)
    # (2) the in-graph recorder density against the numpy one
    rng = np.random.default_rng([W.MASTER_SEED, int(seed), 4244])
    pick = rng.choice(n, 4000, replace=False)
    a_try = np.concatenate([wd.a_rec[pick[:2000]], rng.uniform(-0.85, 0.85, size=(2000, TC.DA))])
    lb_np = CB.recorder_logdensity(a_try, wd.train.x[pick], wd.train.eps[pick], wd.train.hard[pick])
    lb_j = np.asarray(recorder_logdensity_jnp(jnp.asarray(a_try, jnp.float32),
                                              jnp.asarray(CB.recorder_mean(wd.train.x[pick]), jnp.float32),
                                              jnp.asarray(wd.train.eps[pick], jnp.float32),
                                              jnp.asarray(wd.train.hard[pick], jnp.float32)), np.float64)
    both = np.isfinite(lb_np) & np.isfinite(lb_j)
    out["check2_logdensity"] = dict(rows=int(len(a_try)), same_finiteness=float(np.mean(np.isfinite(lb_np) ==
                                                                                       np.isfinite(lb_j))),
                                    max_abs_diff_finite=float(np.max(np.abs(lb_np[both] - lb_j[both]))),
                                    p99_abs_diff_finite=float(np.quantile(np.abs(lb_np[both] - lb_j[both]), 0.99)))
    # (3) refusal shares at the no-caution actor's commands: in-graph against numpy (command_band_check's)
    pi_none = RN.controllers_of(wd.ctx, "room_lower_edge", seed, wd.spec)["le_none"]
    tree3, info3, ref_np = wd.refresh(pi_none, "strong")
    a_pi = jnp.asarray(LR.actor_rows(pi_none, wd.obs), jnp.float32)
    _, ref_j = jax.jit(band_read)(tree3, jnp.arange(n), a_pi)
    ref_j = np.asarray(ref_j, bool)
    saved = json.loads((CB.OUT / f"{quality}.json").read_text())["seeds"][int(seed)]
    out["check3_refusals"] = dict(numpy_share=float(ref_np.mean()), in_graph_share=float(ref_j.mean()),
                                  disagree_share=float(np.mean(ref_np != ref_j)),
                                  check_run_all_refused_share_strong_law=saved["at_actor"]["all"]["refused"],
                                  info=info3)
    log(json.dumps(out, indent=1, default=float))
    d = OUT / quality
    d.mkdir(parents=True, exist_ok=True)
    (d / f"checks_seed{int(seed)}.json").write_text(json.dumps(dict(label=LABEL, **out), indent=1, default=float),
                                                    encoding="utf-8")
    return out


# ------------------------------------------------------------------------------------------------ evaluation

def evaluate(quality, seed, log):
    ctx = RN.Context(quality=quality)
    spec = RN.spec_digest(ctx)
    banks, sb = D.load_start_banks(ctx.banks_root)
    data = RN.load_data(ctx, seed, spec)
    prep = D.prepare(data, "gap", ctx.cfg)
    ev = {}
    for arm in ARMS:
        d = arm_dir(quality, arm, seed)
        rec = json.loads((d / "record.json").read_text())
        params = LR.load_tree(d / "actor.msgpack", LR.actor_template(), rec["actor_sha256"])
        ev[arm] = LI.evaluate_actor(params, prep, banks, sb["anchors"])
        log(f"{quality} seed {seed} {arm}: strong J {ev[arm]['strong']['J']:.2f} ({ev[arm]['strong']['points']:.2f} "
            f"points), none J {ev[arm]['none']['J']:.2f}")
    (OUT / quality / f"evaluate_seed{int(seed)}.json").write_text(
        json.dumps(dict(label=LABEL, quality=quality, seed=int(seed), evaluations=ev,
                        start_banks_json_sha256=LI.sha256_file(ctx.banks_root / "start_banks.json")),
                   indent=1, default=float), encoding="utf-8")


def main(argv):
    mode, quality, seed = argv[0], argv[1], int(argv[2])
    log = lambda m: print(time.strftime("%H:%M:%S ") + str(m), flush=True)
    passmarks_ok()
    if mode == "checks":
        checks(quality, seed, log)
    elif mode == "train":
        wd = World(quality, seed)
        for arm in ARMS:
            if (arm_dir(quality, arm, seed) / "record.json").exists():
                log(f"{quality} seed {seed} {arm}: already finished")
                continue
            train_arm(wd, arm, log)
    elif mode == "evaluate":
        evaluate(quality, seed, log)
    else:
        raise SystemExit(f"unknown mode {mode!r}")


if __name__ == "__main__":
    main(sys.argv[1:])

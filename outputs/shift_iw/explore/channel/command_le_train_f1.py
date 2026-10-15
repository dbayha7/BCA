"""Rerun of the return test with the refusal fallback fixed (EXPLORATORY; pass marks frozen in
channel/PASSMARKS_fallback_rerun.md before this code was written; the owner's go: "Fix the fallback, then rerun
(Recommended)").

The one change from command_le_train.py: a refused command gets "no value credit over the recorded command" only where
the exact ratio is infinite (the command is inside the removed ball, or the recording process's density is 0 at the
row's state and episode); where the ratio is finite but its test mass is past the onset of refusal (the bank is merely
too small), it gets version 5's rule (a), the table's largest finite threshold. Arms A2 (strong law), B2 (training law),
C2 (untargeted: "no value credit" on random rows, as many as A2 gives it per chunk). Everything else is
command_le_train.py's (World, rows, clone, protocol, evaluation).

Usage (from /root/bca-shift, PYTHONPATH=/root/bca-shift, JAX_PLATFORMS=cpu):
    python command_le_train_f1.py checks|train|evaluate <quality> <seed>
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
_spec = importlib.util.spec_from_file_location("cle", HERE / "command_le_train.py")
T = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(T)
CB = T.CB

OUT = T.OUT.parent / "train_f1"
FIRST = T.OUT
LABEL = ("EXPLORATORY: outside the version-5 pre-registration; pass marks frozen in "
         "channel/PASSMARKS_fallback_rerun.md before this code was written")
PASSMARKS = HERE / "PASSMARKS_fallback_rerun.md"
ARMS = ("A2_cmd_strong", "B2_cmd_train", "C2_untargeted")
LAWS = {"A2_cmd_strong": "strong", "B2_cmd_train": "train"}


def passmarks_ok():
    want = (HERE / "PASSMARKS_fallback_rerun.md.sha256").read_text().split()[0]
    got = hashlib.sha256(PASSMARKS.read_bytes()).hexdigest()
    if want != got:
        raise SystemExit(f"the frozen pass marks changed: {got} != {want}")
    return got


def this_code():
    return {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in (Path(__file__).resolve(), HERE / "command_le_train.py", HERE / "command_band_check.py")}


# ------------------------------------------------------------------------------------------------ in-graph

def band_read_f1(arm, idx, a):
    """(R to use, no-credit flag, size-refused flag) at rows idx and commands a (no gradient)."""
    a0 = jax.lax.stop_gradient(a)
    lb = T.recorder_logdensity_jnp(a0, arm["mu"][idx], arm["eps"][idx], arm["hard"][idx])
    lm = arm["log_w"][idx] + T.LOGK0 + arm["log_Z"][idx] - lb
    in_ball = jnp.sum(jnp.square(a0 - arm["centre"]), axis=-1) <= arm["rho2"]
    inf_ref = in_ball | jnp.isposinf(lm)
    m = jnp.exp(jnp.clip(lm, arm["log_floor"], 80.0))
    gi = jnp.searchsorted(arm["grid"], m, side="left")
    R = arm["thresholds"][jnp.minimum(gi, T.MAXG - 1)]
    size_ref = ~inf_ref & ((gi >= arm["n_grid"]) | ~jnp.isfinite(R))
    R_use = jnp.where(inf_ref, 1.0, jnp.where(size_ref, arm["cap"], R))
    return R_use, inf_ref, size_ref


def penalty_band_f1(critic_params, shared, idx, obs, a, arm):
    q1 = LR.CRITIC.apply(critic_params, obs, a)[..., 0]
    w_ref = jnp.maximum(q1 - arm["q1_rec"][idx], 0.0)
    R, inf_ref, _ = band_read_f1(arm, idx, a)
    U = LE.band_width(shared["cal_params"], shared["unit"], obs, a, R)
    w_cert = jnp.maximum(U - jax.lax.stop_gradient(arm["u_logged"][idx]), 0.0)
    return jnp.where(inf_ref, w_ref, w_cert)


_PROGRAMS = {}


def program(args, kind):
    key = (kind, LR._args_key(args))
    if key not in _PROGRAMS:
        pen = penalty_band_f1 if kind == "band" else T.penalty_untargeted
        _PROGRAMS[key] = jax.jit(partial(LE._loop, args, pen))
    return _PROGRAMS[key]


# ------------------------------------------------------------------------------------------------ refresh

def refresh(wd, actor_params, law):
    """T.World.refresh's numbers, plus the cap and the split of refusals at the actor's own commands (numpy)."""
    pi_bank = np.asarray(LR.actor_rows(actor_params, np.asarray(wd.prep.bank.obs, np.float32)), np.float64)
    lw_bank = wd.log_ratio(wd.bank, np.arange(len(wd.a_bank)), wd.a_bank, pi_bank, law)
    w_bank = np.where(wd.hold_bank | ~np.isfinite(lw_bank), 0.0, np.exp(np.clip(lw_bank, -700, 700)))
    th = CB.Thresholds(wd.scores, wd.stream, w_bank)
    if len(th.table.grid) > T.MAXG:
        raise SystemExit(f"the table has {len(th.table.grid)} grid points, more than {T.MAXG}")
    n = wd.prep.n
    idx = np.arange(n)
    a_pi = np.asarray(LR.actor_rows(actor_params, wd.obs), np.float64)
    lm_rec = np.where(wd.gap.contains(wd.a_rec), np.inf, wd.log_ratio(wd.train, idx, wd.a_rec, a_pi, law))
    R_rec, _ = th.read(lm_rec)       # refused recorded commands (kept holds) read the cap, as before
    u_rec = LE.logged_widths(wd.band, wd.prep, LE.as_float32_thresholds(R_rec), stage=T.STAGE)
    a32 = a_pi.astype(np.float32).astype(np.float64)
    lm_pi = np.where(wd.gap.contains(a32), np.inf, wd.log_ratio(wd.train, idx, a32, a32, law))
    _, ref_pi = th.read(lm_pi)
    inf_pi = np.isposinf(lm_pi)
    size_pi = ref_pi & ~inf_pi
    grid = np.full(T.MAXG, np.inf, np.float32)
    grid[:len(th.table.grid)] = th.table.grid
    thr = np.full(T.MAXG, np.inf, np.float32)
    thr[:len(th.table.grid)] = th.table.thresholds
    log_w = np.log(W.ratio(wd.train.c, CB.TILT)) if law == "strong" else np.zeros(n)
    tree = dict(k=jnp.float32(1.0), pull=jnp.ones(n, jnp.float32),
                mu=jnp.asarray(CB.recorder_mean(wd.train.x), jnp.float32),
                eps=jnp.asarray(wd.train.eps, jnp.float32), hard=jnp.asarray(wd.train.hard, jnp.float32),
                log_w=jnp.asarray(log_w, jnp.float32),
                log_Z=jnp.asarray(np.log(np.maximum(wd.train.Z, 1e-30)), jnp.float32),
                centre=jnp.asarray(np.asarray(wd.gap.centre), jnp.float32),
                rho2=jnp.float32(float(wd.gap.rho) ** 2), log_floor=jnp.float32(math.log(th.floor)),
                grid=jnp.asarray(grid), thresholds=jnp.asarray(thr), n_grid=jnp.int32(len(th.table.grid)),
                cap=jnp.float32(th.cap), u_logged=jnp.asarray(u_rec, jnp.float32), q1_rec=jnp.asarray(wd.q1_rec))
    past = np.asarray(TC.stalled(a_pi.astype(np.float32)), bool).any(1)
    info = dict(kish_share=CB.CV.kish(w_bank) / len(w_bank), cap_over_R_u=th.cap / wd.R_u,
                grid_points=int(len(th.table.grid)), no_credit_share=float(inf_pi.mean()),
                capped_share=float(size_pi.mean()), past_80_share=float(past.mean()),
                no_credit_share_past_80=float(inf_pi[past].mean()) if past.any() else None,
                capped_share_past_80=float(size_pi[past].mean()) if past.any() else None,
                no_credit_share_within_80=float(inf_pi[~past].mean()),
                capped_share_within_80=float(size_pi[~past].mean()))
    return tree, info, inf_pi, size_pi


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
    if arm == "C2_untargeted":
        rec = json.loads((arm_dir(wd.ctx.quality, "A2_cmd_strong", wd.seed) / "record.json").read_text())
        match = [c["no_credit_share"] for c in rec["chunks"]]
    for c in range(rows.shape[0]):
        if arm == "C2_untargeted":
            tree, picked = wd.untargeted_tree(match[c], c)
            info = dict(target_share=match[c], picked_share=picked)
            prog = program(wd.args, "untargeted")
        else:
            tree, info, _, _ = refresh(wd, state.params, LAWS[arm])
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
                                                    floor=CB.FLOOR, law=LAWS.get(arm),
                                                    fallback="no value credit where the ratio is infinite; "
                                                             "version 5's capped threshold where the bank is too "
                                                             "small"),
               rows_sha256=rows_sha, clone=clone_rec, chunks=chunks, metric_names=list(LE.METRIC_NAMES),
               chunk_metrics=metrics, finite=finite, actor_sha256=sha, seconds=time.perf_counter() - t0,
               finished_at=time.strftime("%Y-%m-%dT%H:%M:%S%z"))
    (d / "record.json").write_text(json.dumps(rec, indent=1, default=float), encoding="utf-8")
    log(f"{wd.ctx.quality} seed {wd.seed} {arm}: finished in {rec['seconds']:.0f}s (finite {finite})")


def checks(quality, seed, log):
    wd = T.World(quality, seed)
    out = {}
    n = wd.prep.n
    R32 = LE.as_float32_thresholds(np.full(n, wd.R_u))
    u_v5 = LE.logged_widths(wd.band, wd.prep, R32, stage=T.STAGE)
    tree = dict(k=jnp.float32(1.0), pull=jnp.ones(n, jnp.float32), mu=jnp.zeros((n, TC.DA), jnp.float32),
                eps=jnp.ones(n, jnp.float32), hard=jnp.zeros(n, jnp.float32), log_w=jnp.zeros(n, jnp.float32),
                log_Z=jnp.zeros(n, jnp.float32), centre=jnp.full(TC.DA, 9.0, jnp.float32), rho2=jnp.float32(0.0),
                log_floor=jnp.float32(-80.0),
                grid=jnp.asarray(np.r_[np.float32(3e38), np.full(T.MAXG - 1, np.inf)], jnp.float32),
                thresholds=jnp.asarray(np.r_[np.float32(wd.R_u), np.full(T.MAXG - 1, np.inf)], jnp.float32),
                n_grid=jnp.int32(1), cap=jnp.float32(wd.R_u), u_logged=jnp.asarray(u_v5),
                q1_rec=jnp.asarray(wd.q1_rec))
    rows, _ = wd.rows()
    short = rows[0][:300]
    clone_params, _ = wd.start()
    st = TrainState.create(apply_fn=LR.ACTOR.apply, params=clone_params, tx=LR.adam(wd.ctx.cfg.lr))
    mine, _ = program(wd.args, "band")(wd.host.critic.params, st, wd.prep.train, wd.shared, short, tree)
    v5_arm = dict(k=jnp.float32(1.0), pull=jnp.ones(n, jnp.float32), R=jnp.asarray(R32), u_logged=jnp.asarray(u_v5))
    ref, _ = jax.jit(partial(LE._loop, wd.args, LE.band_penalty))(wd.host.critic.params, st, wd.prep.train, wd.shared,
                                                                  short, v5_arm)
    out["check1_max_param_diff_vs_v5_unweighted_300_steps"] = LE.max_param_diff(mine.params, ref.params)
    pi_none = RN.controllers_of(wd.ctx, "room_lower_edge", seed, wd.spec)["le_none"]
    tree3, info3, inf_np, size_np = refresh(wd, pi_none, "strong")
    a_pi = jnp.asarray(LR.actor_rows(pi_none, wd.obs), jnp.float32)
    _, inf_j, size_j = jax.jit(band_read_f1)(tree3, jnp.arange(n), a_pi)
    inf_j, size_j = np.asarray(inf_j, bool), np.asarray(size_j, bool)
    out["check2_refusal_types"] = dict(no_credit_numpy=float(inf_np.mean()), no_credit_in_graph=float(inf_j.mean()),
                                       capped_numpy=float(size_np.mean()), capped_in_graph=float(size_j.mean()),
                                       disagree_no_credit=float(np.mean(inf_np != inf_j)),
                                       disagree_capped=float(np.mean(size_np != size_j)), info=info3)
    log(json.dumps(out, indent=1, default=float))
    d = OUT / quality
    d.mkdir(parents=True, exist_ok=True)
    (d / f"checks_seed{int(seed)}.json").write_text(json.dumps(dict(label=LABEL, **out), indent=1, default=float),
                                                    encoding="utf-8")


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
        wd = T.World(quality, seed)
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

"""EXPLORATORY: result F's no-band control F0 (frozen pass marks, section 7; the owner's accepted change 2).

F0 is result F's arm A2 ("fixed fallback, exact weights", channel/command_le_train_f1.py) with the band's finite part
deleted. The one code difference is the penalty's last line:
    A2: return jnp.where(inf_ref, w_ref, w_cert)        (command_le_train_f1.penalty_band_f1)
    F0: return jnp.where(inf_ref, w_ref, 0.0)           (penalty_f0 below; every other line is A2's)
Everything else is A2's code, imported unchanged from the repository's channel/ folder and called as A2 calls it:
T.World(quality, seed) (data, the frozen host critic, the clone from World.start(), the rows from World.rows()),
command_le_train_f1.refresh(wd, actor, "strong") at each chunk's start (so the no-credit set comes from A2's own
code), jax.jit(partial(lower_edge._loop, args, penalty)), 5 chunks of 4,000 steps, alpha 2.5, k = 1, pull 1 on every
row, Adam 3e-4, batch 256, result F's settings, and learners_iw.evaluate_actor on the three banks.

Checks before training (section 7.3; a failure stops that seed and quality, never replaced):
  KF1  F0's in-graph no-credit flags (command_le_train_f1.band_read_f1) equal refresh's numpy inf_pi on every kept row
       at the clone's commands (seed 0: also at the result F rerun's no-caution actor's commands); disagreement 0
  KF2  (a) with every no-credit flag forced false, 300 steps of F0's program from the clone equal 300 steps of
       lower_edge._loop with a zero penalty bit for bit; (b) 300 steps of F0's program are unchanged bit for bit when
       the tree's grid, thresholds, cap and u_logged and the band's cal_params are replaced by other finite values
  KF3  F0's clone SHA-256 equals A2's clone SHA-256 (A2's record) for the same seed and quality
  KF4  a non-finite actor stops the seed and quality
F0's return is computed right after training (no later rule guards it).

  python -B arm_f0.py --quality {medium-expert,expert} --seed K [--dry]
Writes <sandbox>/repo/runs/shift_iw_v5_explore/arms_abc_f0/F0/<Q>/F0_no_band/seed<K>/{actor.msgpack, record.json
(+ .sha256)}; never writes under channel/train_f1/.
"""
import sys

sys.dont_write_bytecode = True

import argparse  # noqa: E402
import json  # noqa: E402
import time  # noqa: E402
from functools import partial  # noqa: E402

import numpy as np  # noqa: E402

import common as C  # noqa: E402

F1 = C.channel_module("cle_f1", "command_le_train_f1.py")
T = F1.T
KF2_STEPS = 300

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
from flax.training.train_state import TrainState  # noqa: E402

from experiments.shift import learners as LR  # noqa: E402
from experiments.shift_iw import learners_iw as LI  # noqa: E402
from experiments.shift_iw import lower_edge as LE  # noqa: E402


band_read_f1 = F1.band_read_f1        # A2's own function, so penalty_f0's lines read exactly as A2's


def penalty_f0(critic_params, shared, idx, obs, a, arm):
    q1 = LR.CRITIC.apply(critic_params, obs, a)[..., 0]
    w_ref = jnp.maximum(q1 - arm["q1_rec"][idx], 0.0)
    R, inf_ref, _ = band_read_f1(arm, idx, a)
    U = LE.band_width(shared["cal_params"], shared["unit"], obs, a, R)
    w_cert = jnp.maximum(U - jax.lax.stop_gradient(arm["u_logged"][idx]), 0.0)  # noqa: F841  (A2's line, unused)
    return jnp.where(inf_ref, w_ref, 0.0)


def zero_penalty(critic_params, shared, idx, obs, a, arm):
    """KF2(a)'s reference: no penalty anywhere."""
    return jnp.zeros(obs.shape[:-1], jnp.float32)


_PROGRAMS = {}


def program(args, kind="f0"):
    key = (kind, LR._args_key(args))
    if key not in _PROGRAMS:
        pen = {"f0": penalty_f0, "zero": zero_penalty, "a2": F1.penalty_band_f1}[kind]
        _PROGRAMS[key] = jax.jit(partial(LE._loop, args, pen))
    return _PROGRAMS[key]


def flags_off(tree):
    """The same tree with every no-credit flag forced false (as command_le_train_f1.checks builds its tree): epsilon 1
    and no hold, so the recording density is finite everywhere; the ball centred at 9 with radius 0."""
    n = len(np.asarray(tree["q1_rec"]))
    return dict(tree, mu=jnp.zeros((n, 4), jnp.float32), eps=jnp.ones(n, jnp.float32), hard=jnp.zeros(n, jnp.float32),
                log_w=jnp.zeros(n, jnp.float32), log_Z=jnp.zeros(n, jnp.float32),
                centre=jnp.full(4, 9.0, jnp.float32), rho2=jnp.float32(0.0), log_floor=jnp.float32(-80.0))


def band_parts_replaced(tree, shared):
    """Other finite values for every band-derived part F0's loss must not read (KF2(b))."""
    n = len(np.asarray(tree["q1_rec"]))
    grid = np.linspace(0.01, 50.0, T.MAXG).astype(np.float32)
    tree2 = dict(tree, grid=jnp.asarray(grid), thresholds=jnp.full(T.MAXG, 2.75, jnp.float32),
                 cap=jnp.float32(7.5), u_logged=jnp.asarray(np.linspace(0.1, 3.0, n).astype(np.float32)))
    shared2 = dict(shared, cal_params=jax.tree_util.tree_map(lambda x: x * 0.5 + 0.125, shared["cal_params"]))
    return tree2, shared2


def same_params(a, b):
    return bool(LR.tree_sha256(a) == LR.tree_sha256(b)), float(LE.max_param_diff(a, b))


def kf1(wd, params, label):
    tree, info, inf_np, _ = F1.refresh(wd, params, "strong")
    a_pi = jnp.asarray(LR.actor_rows(params, wd.obs), jnp.float32)
    _, inf_j, _ = jax.jit(F1.band_read_f1)(tree, jnp.arange(wd.prep.n), a_pi)
    inf_j = np.asarray(inf_j, bool)
    dis = float(np.mean(inf_np != inf_j))
    return dict(at=label, no_credit_numpy=float(inf_np.mean()), no_credit_in_graph=float(inf_j.mean()),
                disagreement=dis, rows_disagreeing=int(np.sum(inf_np != inf_j)), passes=dis == 0.0, info=info), tree


def kf2(wd, clone_params, tree0):
    rows, _ = wd.rows()
    short = rows[0][:KF2_STEPS]
    st = TrainState.create(apply_fn=LR.ACTOR.apply, params=clone_params, tx=LR.adam(wd.ctx.cfg.lr))
    critic = wd.host.critic.params
    off = flags_off(tree0)
    _, inf_off, _ = jax.jit(F1.band_read_f1)(off, jnp.arange(wd.prep.n),
                                             jnp.asarray(LR.actor_rows(clone_params, wd.obs), jnp.float32))
    mine_a, _ = program(wd.args, "f0")(critic, st, wd.prep.train, wd.shared, short, off)
    ref_a, _ = program(wd.args, "zero")(critic, st, wd.prep.train, wd.shared, short, off)
    same_a, diff_a = same_params(mine_a.params, ref_a.params)
    mine_b, _ = program(wd.args, "f0")(critic, st, wd.prep.train, wd.shared, short, tree0)
    tree2, shared2 = band_parts_replaced(tree0, wd.shared)
    alt_b, _ = program(wd.args, "f0")(critic, st, wd.prep.train, shared2, short, tree2)
    same_b, diff_b = same_params(mine_b.params, alt_b.params)
    forced = float(np.mean(np.asarray(inf_off)))
    same_a = bool(same_a and forced == 0.0)
    return dict(steps=KF2_STEPS, a=dict(passes=same_a, max_param_diff=diff_a, forced_flags_true_share_at_clone=forced),
                b=dict(passes=same_b, max_param_diff=diff_b,
                       replaced=["grid", "thresholds", "cap", "u_logged", "cal_params"]),
                passes=bool(same_a and same_b))


def input_failure(quality, seed):
    """The reason when an input of A2 and F0 failed for good, or None: a failed gapped host or an invalid band (A2's
    World refuses both on every retry, so A2's record never appears), or A2's own actor finished non-finite."""
    from experiments.shift_iw import run as R
    ctx = R.Context(quality=quality, log=lambda m: None)
    spec = R.spec_digest(ctx)
    why = C.host_failed(ctx, seed, spec) or C.band_invalid(ctx, seed, spec)
    if why:
        return f"{why}; result F's A2 cannot run on it"
    p = F1.arm_dir(quality, "A2_cmd_strong", seed) / "record.json"
    if p.is_file():
        try:
            rec = json.loads(p.read_text())
        except ValueError:
            return None
        if rec.get("finite") is False:
            return f"A2's actor of {quality} seed {seed} is not finite ({p})"
    return None


def a2_record(quality, seed):
    p = F1.arm_dir(quality, "A2_cmd_strong", seed) / "record.json"
    if not p.is_file():
        raise C.Refused(f"refused: A2's record for {quality} seed {seed} does not exist ({p}); F0 runs after A2")
    rec = json.loads(p.read_text())
    if rec.get("passmarks_sha256") != F1.passmarks_ok() or rec.get("finite") is not True:
        raise C.Refused(f"refused: A2's record {p} names other pass marks or a non-finite actor")
    return p, rec


def pretraining_checks(quality, seed):
    """KF3, KF1 and KF2 on real data without training an arm or writing a record (section 8 step 1's unit checks of
    F0; the F0 job runs them again before it trains). Returns the checks."""
    C.require_passmarks()
    if quality not in C.F0_QUALITIES:
        raise C.Refused(f"refused: F0 runs on {C.F0_QUALITIES} only, not {quality!r}")
    from experiments.shift_iw import run as R
    _, a2 = a2_record(quality, seed)
    wd = T.World(quality, seed)
    clone_params, clone_rec = wd.start()
    checks = dict(KF3=dict(clone_sha256=clone_rec["actor_sha256"], a2_clone_sha256=a2["clone"]["actor_sha256"],
                           passes=clone_rec["actor_sha256"] == a2["clone"]["actor_sha256"]))
    checks["KF1"], tree0 = kf1(wd, clone_params, "the clone")
    if int(seed) == 0:
        le_none = R.controllers_of(wd.ctx, "room_lower_edge", 0, wd.spec)["le_none"]
        checks["KF1_no_caution"], _ = kf1(wd, le_none, "the result F rerun's no-caution actor (le_none)")
    checks["KF2"] = kf2(wd, clone_params, tree0)
    checks["passes"] = bool(checks["KF3"]["passes"] and checks["KF1"]["passes"] and
                            checks.get("KF1_no_caution", {"passes": True})["passes"] and checks["KF2"]["passes"])
    return checks


def run_job(quality, seed, dry=False):
    from experiments.shift_iw import data as D
    from experiments.shift_iw import run as R
    digest = C.require_passmarks()
    if quality not in C.F0_QUALITIES:
        raise C.Refused(f"refused: F0 runs on {C.F0_QUALITIES} only, not {quality!r}")
    started = time.perf_counter()
    F1.passmarks_ok()
    d = C.arm_dir("F0", quality, seed)
    why = input_failure(quality, seed)
    if why and not dry:     # stopped, never replaced: the quality is "not decided" (section 7.3)
        rec = C.base_record("F0", quality, seed, digest)
        return C.write_stop(d / "record.json", rec, "input", why, started)
    a2_path, a2 = a2_record(quality, seed)
    try:
        wd = T.World(quality, seed)
    except (R.Refused, LI.ApprovalError) as error:
        raise C.Refused(f"refused: result F's World for {quality} seed {seed}: {error}") from error
    if C.smoke_settings():       # a few steps per chunk; the clone keeps its 20,000 steps (KF3 compares it with A2's)
        from dataclasses import replace
        wd.ctx.cfg = replace(wd.ctx.cfg, actor_only_steps=500, actor_only_chunk=100)
    if dry:
        C.log(f"F0 {quality} seed {seed} DRY: World constructed (data, host, band, approval 'bca_lower_edge'), A2's "
              f"record present; nothing trained or written")
        return 0
    rec = C.base_record("F0", quality, seed, digest)
    rec.update(a2_record=str(a2_path), a2_record_sha256=C.sha256_file(a2_path), this_penalty="penalty_f0",
               differs_from_a2="the penalty's last line: jnp.where(inf_ref, w_ref, 0.0) for jnp.where(inf_ref, "
                               "w_ref, w_cert)",
               settings=dict(spread=T.CB.SPREAD, clip=T.CB.CLIP, grid_ratio=T.CB.GRID_RATIO, floor=T.CB.FLOOR,
                             law="strong", k=1.0, pull=1.0))

    def stop(rule, reason, **extra):
        rec.update(status="stopped", stopped_by=rule, reason=reason, seconds=time.perf_counter() - started,
                   finished_at=time.strftime("%Y-%m-%dT%H:%M:%S%z"), **extra)
        C.write_record(d / "record.json", rec)
        C.log(f"F0 {quality} seed {seed}: STOPPED by {rule}: {reason}")
        return 0

    clone_params, clone_rec = wd.start()
    checks = {}
    checks["KF3"] = dict(clone_sha256=clone_rec["actor_sha256"], a2_clone_sha256=a2["clone"]["actor_sha256"],
                         passes=clone_rec["actor_sha256"] == a2["clone"]["actor_sha256"])
    if not checks["KF3"]["passes"]:
        return stop("KF3", "F0's clone differs from A2's clone", checks=checks)
    checks["KF1"], tree0 = kf1(wd, clone_params, "the clone")
    if int(seed) == 0:
        le_none = R.controllers_of(wd.ctx, "room_lower_edge", 0, wd.spec)["le_none"]
        checks["KF1_no_caution"], _ = kf1(wd, le_none, "the result F rerun's no-caution actor (le_none)")
    if not checks["KF1"]["passes"] or not checks.get("KF1_no_caution", {"passes": True})["passes"]:
        return stop("KF1", "F0's in-graph no-credit flags differ from refresh's numpy flags", checks=checks)
    checks["KF2"] = kf2(wd, clone_params, tree0)
    if not checks["KF2"]["passes"]:
        return stop("KF2", "F0's program is not A2's loop with no penalty off the no-credit set, or a band input "
                           "reaches it", checks=checks)
    rows, rows_sha = wd.rows()
    state = TrainState.create(apply_fn=LR.ACTOR.apply, params=clone_params, tx=LR.adam(wd.ctx.cfg.lr))
    prog = program(wd.args, "f0")
    chunks, metrics = [], []
    for c in range(rows.shape[0]):
        tree, info, _, _ = F1.refresh(wd, state.params, "strong")
        state, aux = prog(wd.host.critic.params, state, wd.prep.train, wd.shared, rows[c], tree)
        aux = np.asarray(aux, np.float64)
        metrics.append(aux.mean(0).tolist())
        chunks.append(dict(chunk=c, **info, W_mean=float(aux[:, 4].mean()), q_mean=float(aux[:, 1].mean())))
        C.log(f"F0 {quality} seed {seed}: chunk {c + 1}/{rows.shape[0]}, no-credit share "
              f"{info['no_credit_share']:.4f}, W mean {chunks[-1]['W_mean']:.4f}")
    jax.block_until_ready(state.params)
    finite = C.finite_tree(state.params)
    sha = LR.save_tree(d / "actor.msgpack", state.params)
    rec.update(checks=checks, rows_sha256=rows_sha, clone=clone_rec, chunks=chunks, metric_names=list(LE.METRIC_NAMES),
               chunk_metrics=metrics, finite=finite, actor_sha256=sha,
               a2_no_credit_share_per_chunk=[c.get("no_credit_share") for c in a2.get("chunks", [])])
    if not finite:
        return stop("KF4", "the F0 actor is not finite")
    banks, sb = D.load_start_banks(wd.ctx.banks_root)
    if C.smoke_settings():
        banks = {k: v[:50] for k, v in banks.items()}
    ev = LI.evaluate_actor(state.params, wd.prep, banks, sb["anchors"])
    rec.update(status="finished", stopped_by=None, evaluation=ev,
               start_banks_json_sha256=LI.sha256_file(wd.ctx.banks_root / "start_banks.json"),
               seconds=time.perf_counter() - started, finished_at=time.strftime("%Y-%m-%dT%H:%M:%S%z"))
    C.write_record(d / "record.json", rec)
    C.log(f"F0 {quality} seed {seed}: strong {ev['strong']['points']:.2f} points (J {ev['strong']['J']:.2f}); "
          f"{rec['seconds']:.0f}s -> {d / 'record.json'}")
    return 0


def main(argv=None):
    p = argparse.ArgumentParser(description="EXPLORATORY: result F's no-band control F0, one quality and seed")
    p.add_argument("--quality", required=True, choices=C.QUALITIES)
    p.add_argument("--seed", type=int, required=True, choices=C.SEEDS)
    p.add_argument("--dry", action="store_true")
    a = p.parse_args(argv)
    try:
        return run_job(a.quality, a.seed, a.dry)
    except C.Refused as error:
        print(str(error), file=sys.stderr, flush=True)
        return 2


if __name__ == "__main__":
    sys.exit(main())

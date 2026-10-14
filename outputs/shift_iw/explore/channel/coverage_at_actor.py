"""Coverage of BCA's band where the lower edge reads it: at the no-caution actor's own commands (EXPLORATORY, post hoc;
see EXPECTATIONS_coverage_at_actor.md, written before this ran).

Reads (never writes) runs/shift_iw_v5: per quality and seed, the gapped host, its band and band record, the data and the
lower-edge room check's no-caution actor ('le_none'). On every kept training row it compares the band at the recorded
command and at the actor's command a = pi(s), with the truth from one exact float64 simulator step at (s, a) and BCA's
own target rule (learners_iw.critic_parts). Writes runs/shift_iw_v5_explore/channel/<quality>.json.

Usage (from /root/bca-shift, PYTHONPATH=/root/bca-shift): python coverage_at_actor.py <quality> [<quality> ...]
"""
import json
import sys
import time

import jax.numpy as jnp
import numpy as np
from jax.experimental import enable_x64

from experiments.shift import learners as LR
from experiments.shift import thruster_chain as TC
from experiments.shift_iw import data as D
from experiments.shift_iw import learners_iw as LI
from experiments.shift_iw import lower_edge as LE
from experiments.shift_iw import run as RN
from experiments.shift_iw import world as W
from runtime.networks import Transition

OUT = RN.ROOT_DEFAULT.parent / "shift_iw_v5_explore" / "channel"
LABEL = ("EXPLORATORY, post hoc: outside the version-5 pre-registration; expectations in "
         "channel/EXPECTATIONS_coverage_at_actor.md, written before this ran")
STAGE = "coverage"


def wmean(x, w):
    w = np.asarray(w, np.float64)
    return float(np.sum(np.asarray(x, np.float64) * w) / np.sum(w)) if w.sum() > 0 else None


def wquantile(x, w, q):
    x, w = np.asarray(x, np.float64), np.asarray(w, np.float64)
    if not len(x) or w.sum() <= 0:
        return None
    order = np.argsort(x)
    cw = np.cumsum(w[order]) / w.sum()
    return float(x[order][min(np.searchsorted(cw, q), len(x) - 1)])


def exact_step(states, actions):
    """One exact float64 step of the stalling world from float64 states at float32 commands: (next states, rewards)."""
    with enable_x64():
        x1, r = TC.dynamics_jax(jnp.asarray(states, jnp.float64),
                                jnp.asarray(np.asarray(actions, np.float32).astype(np.float64)), W.PHYSICS)
        return np.asarray(x1, np.float64), np.asarray(r, np.float64)


def group_stats(err, U, w_train, w_strong, mask):
    """Coverage (training law and strong law), the needed multiplier and the mean signed error on the rows in mask."""
    m = np.asarray(mask, bool)
    if not m.any():
        return dict(rows=0)
    cov = (np.abs(err) <= U)[m]
    ratio = (np.abs(err) / np.maximum(U, 1e-12))[m]
    return dict(rows=int(m.sum()), share_of_rows=float(m.mean()),
                strong_law_share=float(w_strong[m].sum() / w_strong.sum()),
                coverage_train_law=wmean(cov, w_train[m]), coverage_strong_law=wmean(cov, w_strong[m]),
                needed_multiplier_strong_law=wquantile(ratio, w_strong[m], 0.9),
                signed_error_mean_strong_law=wmean(err[m], w_strong[m]),
                abs_error_mean_strong_law=wmean(np.abs(err[m]), w_strong[m]),
                width_mean_strong_law=wmean(U[m], w_strong[m]))


def one(ctx, spec, seed):
    t0 = time.perf_counter()
    data = RN.load_data(ctx, seed, spec)
    prep = D.prepare(data, "gap", ctx.cfg)
    host = RN.load_host_state(ctx, "gap", seed, spec)
    brec = RN.require_band(ctx, "gap", seed, spec)
    approval = RN.approval_for(ctx, STAGE)
    band = LI.load_band(RN.band_dir(ctx, "gap", seed) / "band.msgpack", brec["band_sha256"], host, prep, ctx.cfg,
                        seed, approval, brec["band"], stage=STAGE)
    R_u = RN.num(brec["thresholds"]["unweighted"]["threshold"])
    pi = RN.controllers_of(ctx, "room_lower_edge", seed, spec)["le_none"]

    obs = np.asarray(prep.train.obs, np.float32)
    a_pi = np.asarray(LR.actor_rows(pi, obs), np.float32)
    c = np.asarray(data.train_log.current[prep.rows], np.float64)
    w_train = np.ones(prep.n)
    w_strong = W.ratio(c, "strong")

    # sanity: the exact step at the recorded commands reproduces the recorded transitions
    x1_rec, r_rec = exact_step(prep.states, prep.actions)
    sanity = dict(reward_max_abs_diff=float(np.max(np.abs(r_rec.astype(np.float32) - np.asarray(prep.train.reward)))),
                  next_obs_max_abs_diff=float(np.max(np.abs(prep.normalize(x1_rec) - np.asarray(prep.train.next_obs)))))

    x1, r = exact_step(prep.states, a_pi)
    trans_pi = Transition(jnp.asarray(obs), jnp.asarray(a_pi), jnp.asarray(r.astype(np.float32)),
                          jnp.asarray(prep.normalize(x1)), jnp.zeros(prep.n, jnp.float32))
    y_pi, q_pi = LI.critic_parts(host, trans_pi, ctx.cfg, seed, LI.coverage_key(ctx.cfg, seed, 90))
    y_rec, q_rec = LI.critic_parts(host, prep.train, ctx.cfg, seed, LI.coverage_key(ctx.cfg, seed, 91))
    err_pi, err_rec = y_pi - q_pi, y_rec - q_rec

    R32u = LE.as_float32_thresholds(np.full(prep.n, R_u))
    arm = RN.lower_edge_arm(data, prep, brec, "exact_strong", R_u)
    Rw, cap, refused = LE.capped_thresholds(arm["R"], arm["pool"])
    R32w = LE.as_float32_thresholds(Rw)
    U = {("unweighted", "pi"): LE.widths_at(band, prep, a_pi, R32u, stage=STAGE),
         ("unweighted", "rec"): LE.logged_widths(band, prep, R32u, stage=STAGE),
         ("exact_strong", "pi"): LE.widths_at(band, prep, a_pi, R32w, stage=STAGE),
         ("exact_strong", "rec"): LE.logged_widths(band, prep, R32w, stage=STAGE)}
    U = {k: np.asarray(v, np.float64) for k, v in U.items()}

    stall_pi = np.asarray(TC.stalled(a_pi), bool).any(axis=1)
    groups = dict(all=np.ones(prep.n, bool), actor_past_80=stall_pi, actor_within_80=~stall_pi,
                  hard_water=np.asarray(prep.region, bool))
    out = dict(seed=int(seed), rows=int(prep.n), R_u=R_u, R_exact_strong=dict(mean=float(Rw.mean()),
               min=float(Rw.min()), max=float(Rw.max()), refused_rows=int(refused.sum()), cap=cap),
               sanity=sanity, actor_past_80_share=float(stall_pi.mean()),
               actor_past_80_share_strong_law=wmean(stall_pi, w_strong),
               actor_past_80_share_hard_water=float(stall_pi[groups["hard_water"]].mean())
               if groups["hard_water"].any() else None)
    for (arm_name, where), width in U.items():
        err = err_pi if where == "pi" else err_rec
        gs = groups if where == "pi" else dict(all=groups["all"], hard_water=groups["hard_water"])
        out[f"{arm_name}@{'actor' if where == 'pi' else 'recorded'}"] = {
            g: group_stats(err, width, w_train, w_strong, m) for g, m in gs.items()}
    out["width_actor_over_recorded_mean"] = float(np.mean(U[("unweighted", "pi")] / U[("unweighted", "rec")]))
    out["seconds"] = time.perf_counter() - t0
    return out


def main(argv):
    OUT.mkdir(parents=True, exist_ok=True)
    for quality in argv:
        ctx = RN.Context(quality=quality)
        spec = RN.spec_digest(ctx)
        seeds = []
        for seed in RN.SEEDS:
            r = one(ctx, spec, seed)
            ctx.log(f"{quality} seed {seed}: actor past 80% on {r['actor_past_80_share']:.3f} of rows; "
                    f"coverage at actor (strong law) unweighted "
                    f"{r['unweighted@actor']['all']['coverage_strong_law']:.3f}, exact "
                    f"{r['exact_strong@actor']['all']['coverage_strong_law']:.3f}; at recorded (train law) "
                    f"{r['unweighted@recorded']['all']['coverage_train_law']:.3f} ({r['seconds']:.0f}s)")
            seeds.append(r)
        body = dict(label=LABEL, quality=quality, spec_sha256=spec, code_sha256=LI.code_hashes(),
                    finished_at=time.strftime("%Y-%m-%dT%H:%M:%S%z"), seeds=seeds)
        (OUT / f"{quality}.json").write_text(json.dumps(body, indent=1, default=float), encoding="utf-8")


if __name__ == "__main__":
    main(sys.argv[1:])

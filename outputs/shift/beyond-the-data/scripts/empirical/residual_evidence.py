"""EXPLORATORY (outside the gate; changes no pre-registered rule). NO BCA: no scale network, calibrator, threshold or
dose. Data-side question only: do the few training rows that survived the removal past 80% throttle carry evidence of
the stall in the frozen gapped critic's one-step (temporal-difference) residual, and is that evidence absent in the
no-stall world? If it is absent, no residual-based band (BCA's kind) can tell harmful unsupported commands from
harmless ones in this world, and BCA can only act as a calibrated 'unsupported' detector.

Residual at a kept training row: r + gamma * min(Q1, Q2)(s', c(s')) - Q1(s, a), c = the stall-aware controller that
the fixed-controller critic scores (no target noise). Also the critic's exact error at the surviving rows.

Usage: python residual_evidence.py <seed>
"""
import json
import sys
import time

import numpy as np

sys.path.insert(0, "/tmp/claude-0/-home-user/9e341792-006d-5a11-a8e9-8fac16ac7f5b/scratchpad/ood/anatomy")
from common import Seed, jsonable, log  # noqa: E402
import experiments.shift.learners as L  # noqa: E402
import experiments.shift.recorder as RC  # noqa: E402
import experiments.shift.thruster_chain as TC  # noqa: E402
import experiments.shift.truth as TR  # noqa: E402
import jax.numpy as jnp  # noqa: E402

OUT = "/home/user/bca/runs/shift_explore/empirical"


def residuals(critic, pg, physics):
    fixed = L.FixedController("stall_aware", physics, pg.obs_mean, pg.obs_std)
    nxt = np.asarray(pg.train.next_obs)
    a_next = np.asarray(fixed.apply(fixed.params(), jnp.asarray(nxt)), np.float32)
    qn = L.critic_rows(critic, nxt, a_next)[0]
    q = L.critic_rows(critic, np.asarray(pg.train.obs), pg.actions)[0]
    y = np.asarray(pg.train.reward, np.float64) + physics.gamma * qn.min(1)
    return y - q[:, 0], q[:, 0]


def groups(pg, physics):
    a = np.abs(pg.actions.astype(np.float64))
    past = TC.past_knee(pg.actions, physics)
    near = (~past) & (a.max(1) > 0.7)
    reg = pg.region
    return {"past80_all": past, "past80_region": past & reg, "near80_region(0.7-0.8)": near & reg,
            "region_other": reg & ~past & ~near, "outside_region": ~reg & ~past}


def main(rep):
    t = time.perf_counter()
    S = Seed(rep, worlds=("stall", "no_stall"))
    out = dict(rep=rep, label="EXPLORATORY, no BCA: TD residual evidence at surviving past-80% rows", worlds={})
    for world, cname in (("stall", "gapped"), ("no_stall", "no_stall_gapped")):
        pg = S.prepared(world)
        physics = S.sd[world].physics
        critic = S.h2_critic(cname)
        d, q1 = residuals(critic, pg, physics)
        ad = np.abs(d)
        q90 = float(np.quantile(ad, 0.9))
        g = groups(pg, physics)
        rec = dict(n_rows=int(pg.n), abs_residual_q90_all=q90, abs_residual_median_all=float(np.median(ad)), groups={})
        for k, m in g.items():
            if m.sum() == 0:
                rec["groups"][k] = dict(n=0)
                continue
            rec["groups"][k] = dict(n=int(m.sum()), share_rows=float(m.mean()), median_abs=float(np.median(ad[m])),
                                    mean_signed=float(d[m].mean()), q90_abs=float(np.quantile(ad[m], 0.9)),
                                    share_above_overall_q90=float((ad[m] > q90).mean()))
        # exact critic error at the surviving past-80% rows: Q1 - Q^{stall-aware}(s, a)
        m = g["past80_all"]
        if m.sum():
            pol = RC.stall_aware(physics)
            tq = TR.q_value_jax(pol, pg.states[m], pg.actions[m].astype(np.float64), physics, chunk=256)
            tq_ref_a = np.clip(pg.actions[m].astype(np.float64), -0.8, 0.8)
            tq_ref = TR.q_value_jax(pol, pg.states[m], tq_ref_a, physics, chunk=256)
            qc_ref = L.critic_rows(critic, np.asarray(pg.train.obs)[m], tq_ref_a.astype(np.float32))[0][:, 0]
            err = q1[m] - tq
            rec["surviving_rows_exact"] = dict(
                critic_error_median=float(np.median(err)), critic_abs_error_median=float(np.median(np.abs(err))),
                true_price_past80_vs_clip_median=float(np.median(tq - tq_ref)),
                critic_price_past80_vs_clip_median=float(np.median(q1[m] - qc_ref)))
        out["worlds"][world] = rec
        log(world, json.dumps(jsonable(rec))[:1500])
    out["seconds"] = time.perf_counter() - t
    with open(f"{OUT}/residual_evidence_seed{rep}.json", "w") as f:
        json.dump(jsonable(out), f, indent=1, sort_keys=True)
    log("done", out["seconds"])


if __name__ == "__main__":
    main(int(sys.argv[1]))

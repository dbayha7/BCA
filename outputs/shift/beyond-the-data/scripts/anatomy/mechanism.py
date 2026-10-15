"""EXPLORATORY (outside the gate, no BCA, no training). Part 4: where, in what the actor actually reads, the gapped
and full-log critics differ.

The H2 actors were trained only on the gapped log's training rows: every actor update reads the critic at
(training state s, the actor's own command pi(s)) and nowhere else (learners.actor_step). So the two actors (same
start, same rows, same batches) differ only through Q_gapped - Q_full at training states, at the commands the actor
emits there. This script splits that difference into commands inside the recorded range (every thruster at most
80%) and commands past 80%, against exact truth (Q of the stall-aware controller, the policy these critics score).

Usage: python mechanism.py <seed>
"""
import sys
import time

import numpy as np

from common import *  # noqa: F401,F403

N_ROWS = 8000
T_BINS = ((-1.0, 0.6), (0.6, 0.75), (0.75, 0.8 + 1e-7), (0.8 + 1e-7, 1.01))


def main(rep):
    t0 = time.perf_counter()
    S = Seed(rep)
    pg, ph = S.pg, S.physics
    sa = RC.stall_aware(ph)
    rng = np.random.default_rng([20261012, int(rep), 4404])
    idx = np.sort(rng.choice(pg.n, N_ROWS, replace=False))
    X = pg.states[idx]
    X32 = np.asarray(np.asarray(X, np.float32), np.float64)
    obs = np.asarray(pg.train.obs)[idx]
    region = pg.region[idx]
    push = TC.push(X)
    critics = {k: S.h2_critic(k) for k in ("gapped", "full", "random")}
    h2, rec = S.h2_actors()
    actors = {k: h2[k] for k in ("clone", "gapped", "full", "random")}
    cmds = {k: L.actor_rows(p, obs).astype(np.float64) for k, p in actors.items()}
    cmds["stall_aware"] = sa(X)
    cmds["recorded"] = pg.actions[idx].astype(np.float64)
    res = dict(n_rows=N_ROWS, region_share=float(region.mean()), share_push_gt_0p8=float((push > 0.8).mean()),
               h2_points={s: v["points"] for s, v in rec["situations"].items()},
               h2_share_past80={s: v["unrecorded_throttle_share"] for s, v in rec["situations"].items()})

    # actor behaviour at training states
    beh = {}
    for k, a in cmds.items():
        t1 = -a[:, 0]
        beh[k] = dict(throttle1_mean_all=float(t1.mean()), throttle1_mean_region=float(t1[region].mean()),
                      throttle1_mean_push_gt_0p8=float(t1[push > 0.8].mean()),
                      share_past80_all=float(TC.past_knee(a, ph).mean()),
                      share_past80_region=float(TC.past_knee(a[region], ph).mean()),
                      share_past80_push_gt_0p8=float(TC.past_knee(a[push > 0.8], ph).mean()),
                      share_throttle1_gt_0p75_region=float((t1[region] > 0.75).mean()))
    res["actor_at_training_states"] = beh
    log(f"seed {rep}: training-state behaviour " + "  ".join(
        f"{k}: thr1 region {v['throttle1_mean_region']:.3f} past80 region {v['share_past80_region']:.3f}"
        for k, v in beh.items()))

    # critic slopes along thruster 1 at the commands each actor emits at training states, against truth
    slopes = {}
    for who in ("clone", "gapped", "full", "random", "stall_aware"):
        a = cmds[who]
        a32 = np.asarray(np.asarray(a, np.float32), np.float64)
        g_true = TR.action_slope(sa, X32, a32, ph, chunk=4096)
        out = {}
        g = {}
        q = {}
        for cname, cp in critics.items():
            q[cname], g[cname] = critic_q_grad(cp, pg, X, a)
        t1 = -a[:, 0]
        for lo, hi in T_BINS:
            m = (t1 > lo) & (t1 <= hi)
            key = f"throttle1_{max(lo, 0):.2f}-{min(hi, 1.0):.2f}"
            if not m.any():
                out[key] = dict(share_rows=0.0)
                continue
            # slope along throttle up for thruster 1: dQ/dt1 = -dQ/da0
            out[key] = dict(share_rows=float(m.mean()), share_region=float(region[m].mean()),
                            dQ_dthrottle1_true=float(-g_true[m, 0].mean()),
                            **{f"dQ_dthrottle1_{c}": float(-g[c][m, 0].mean()) for c in critics},
                            grad_gap_gapped_minus_full_norm=float(np.linalg.norm(g["gapped"][m] - g["full"][m], axis=1).mean()),
                            grad_error_gapped_norm=float(np.linalg.norm(g["gapped"][m] - g_true[m], axis=1).mean()),
                            grad_error_full_norm=float(np.linalg.norm(g["full"][m] - g_true[m], axis=1).mean()))
        out["abs_Q_mean"] = {c: float(np.abs(q[c]).mean()) for c in critics}
        out["lambda_at_alpha5"] = {c: float(5.0 / np.abs(q[c]).mean()) for c in critics}
        slopes[who] = out
        log(f"  slopes at {who}'s commands: " + "; ".join(
            f"{k}: n {v['share_rows']:.3f} true {v.get('dQ_dthrottle1_true', float('nan')):+.2f} gap "
            f"{v.get('dQ_dthrottle1_gapped', float('nan')):+.2f} full {v.get('dQ_dthrottle1_full', float('nan')):+.2f}"
            for k, v in out.items() if k.startswith("throttle1")))
    res["slopes_at_actor_commands"] = slopes

    # value differences: inside the recorded range against past 80%
    def adv(c_params, a, b):
        return critic_q(c_params, pg, X, a) - critic_q(c_params, pg, X, b)

    def adv_true(a, b):
        return true_q(sa, X, a, ph) - true_q(sa, X, b, ph)

    pairs = {}
    # inside the range: recorded command vs the stall-aware command; the clone's command vs the stall-aware command
    for name, a, b in (("recorded_vs_stall_aware", cmds["recorded"], cmds["stall_aware"]),
                       ("clone_vs_stall_aware", cmds["clone"], cmds["stall_aware"])):
        t = adv_true(a, b)
        d = {c: adv(cp, a, b) for c, cp in critics.items()}
        pairs[name] = dict(true_mean=float(t.mean()),
                           **{f"{c}_error_mean": float((d[c] - t).mean()) for c in critics},
                           **{f"{c}_error_abs_mean": float(np.abs(d[c] - t).mean()) for c in critics},
                           gapped_minus_full_abs_mean=float(np.abs(d["gapped"] - d["full"]).mean()),
                           gapped_minus_full_abs_mean_region=float(np.abs(d["gapped"] - d["full"])[region].mean()))
    # past 80%: each actor's command against the same command capped at 80% (rows where it is past 80% only),
    # and thruster 1 pushed to full throttle against capped at 80% on top of the stall-aware command (all rows)
    for who in ("gapped", "full"):
        a = cmds[who]
        m = TC.past_knee(a, ph)
        if m.sum() < 5:
            pairs[f"{who}_actor_past80_vs_capped"] = dict(n=int(m.sum()))
            continue
        ac = np.clip(a, -0.8, 0.8)
        Xm = X[m]
        t = true_q(sa, Xm, a[m], ph) - true_q(sa, Xm, ac[m], ph)
        d = {c: critic_q(cp, pg, Xm, a[m]) - critic_q(cp, pg, Xm, ac[m]) for c, cp in critics.items()}
        pairs[f"{who}_actor_past80_vs_capped"] = dict(
            n=int(m.sum()), share_region=float(region[m].mean()), true_mean=float(t.mean()),
            **{f"{c}_mean": float(d[c].mean()) for c in critics},
            **{f"{c}_error_mean": float((d[c] - t).mean()) for c in critics},
            gapped_minus_full_abs_mean=float(np.abs(d["gapped"] - d["full"]).mean()))
    a_hi = cmds["stall_aware"].copy()
    a_hi[:, 0] = -1.0
    a_cap = cmds["stall_aware"].copy()
    a_cap[:, 0] = -0.8
    t = adv_true(a_hi, a_cap)
    d = {c: adv(cp, a_hi, a_cap) for c, cp in critics.items()}
    for nm, m in (("all", np.ones(N_ROWS, bool)), ("region", region), ("push_gt_0p8", push > 0.8)):
        pairs[f"thruster1_full_vs_80_{nm}"] = dict(
            n=int(m.sum()), true_mean=float(t[m].mean()),
            **{f"{c}_mean": float(d[c][m].mean()) for c in critics},
            **{f"{c}_error_mean": float((d[c] - t)[m].mean()) for c in critics},
            gapped_minus_full_abs_mean=float(np.abs(d["gapped"] - d["full"])[m].mean()))
    res["value_differences"] = pairs
    for k, v in pairs.items():
        log(f"  {k}: " + " ".join(f"{a}={b:+.3f}" if isinstance(b, float) else f"{a}={b}" for a, b in v.items()))
    write(f"mechanism_seed{rep}.json", dict(kind="anatomy-mechanism", rep=int(rep),
                                            seconds=time.perf_counter() - t0, **res))


if __name__ == "__main__":
    main(int(sys.argv[1]))

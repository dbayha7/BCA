"""EXPLORATORY (outside the gate, no BCA). Part 1: what each critic believes about pushing thruster 1 past 80%,
against the exact truth, at training states and at strong-current states.

Critics: the fixed-controller critics of H2 (they score the stall-aware controller; truth = Q of the stall-aware
controller in that critic's world): gapped, full, random removal, full-log placebo (another initialisation), and the
no-stall world's gapped and full critics. Optionally the co-trained plain TD3+BC critics (H3 used the gapped one),
scored against their own actor.

Throttle sweep: thruster 1 set to -v (v = throttle, pushing against the current), v in V_GRID, the other thrusters
either from the full-throttle pattern (H1's c_hi / c_cap family) or from the stall-aware controller's own command at
that state. Preference past 80% = Q(v = 1.0) - Q(v = 0.8).

Usage: python critic_beliefs.py <seed> [--host]
"""
import sys
import time

import numpy as np

from common import *  # noqa: F401,F403

V_GRID = np.array([0.5, 0.6, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0])
I80 = int(np.where(V_GRID == 0.8)[0][0])
N_SET = 1500
CURRENT_PROBE = (0.3, 0.5, 0.7, 0.8, 0.9, 1.0, 1.1)
N_PROBE = 300
HOST_SETS = ("train_region", "train_other", "strong_visit_stall_aware", "strong_visit_gapped_actor", "probe_c0.5",
             "probe_c0.9")


def sweep_actions(base, v):
    a = np.repeat(np.asarray(base, np.float64)[:, None, :], len(v), 1).copy()
    a[:, :, 0] = -v[None, :]
    return a  # (N, V, 4)


def state_sets(S, rng):
    pg, ph = S.pg, S.physics
    sa = RC.stall_aware(ph)
    acts, _ = S.h2_actors()
    out = {}
    reg = np.where(pg.region)[0]
    oth = np.where(~pg.region)[0]
    out["train_region"] = pg.states[np.sort(rng.choice(reg, min(N_SET, len(reg)), replace=False))]
    out["train_other"] = pg.states[np.sort(rng.choice(oth, N_SET, replace=False))]
    out["recorded_visit_stall_aware"] = visited_states(sa, "recorded", ph, 200, take=N_SET, rng=rng)[0]
    out["strong_visit_stall_aware"] = visited_states(sa, "strong_current", ph, 200, take=N_SET, rng=rng)[0]
    for k in ("gapped", "full"):
        pol = L.mlp_policy(acts[k], pg, k)
        out[f"strong_visit_{k}_actor"] = visited_states(pol, "strong_current", ph, 200, take=N_SET, rng=rng)[0]
    # current probe: training full-throttle-region states with the current replaced (floats and wave kept)
    base = pg.states[np.sort(rng.choice(reg, N_PROBE, replace=False))]
    for c in CURRENT_PROBE:
        x = base.copy()
        x[:, TC.CURRENT] = c
        out[f"probe_c{c:.1f}"] = x
    return out


def summarise(qc, qt, v=V_GRID):
    """qc, qt: (N, V) critic and true Q along the sweep. Returns summary dict."""
    pc = qc[:, -1] - qc[:, I80]
    pt = qt[:, -1] - qt[:, I80]
    err = pc - pt
    lo_c = qc[:, I80] - qc[:, 1]  # 0.6 -> 0.8 (inside the recorded range)
    lo_t = qt[:, I80] - qt[:, 1]
    am_c = v[np.argmax(qc, 1)]
    am_t = v[np.argmax(qt, 1)]
    return dict(
        pref_critic_mean=float(pc.mean()), pref_true_mean=float(pt.mean()),
        pref_error_mean=float(err.mean()), pref_error_median=float(np.median(err)),
        pref_error_abs_median=float(np.median(np.abs(err))),
        share_critic_prefers_past80=float((pc > 0).mean()), share_true_prefers_past80=float((pt > 0).mean()),
        rise_0p6_to_0p8_critic=float(lo_c.mean()), rise_0p6_to_0p8_true=float(lo_t.mean()),
        rise_0p6_to_0p8_error_mean=float((lo_c - lo_t).mean()),
        argmax_critic_mean=float(am_c.mean()), argmax_true_mean=float(am_t.mean()),
        share_argmax_critic_past80=float((am_c > 0.8).mean()), share_argmax_true_past80=float((am_t > 0.8).mean()),
        curve_critic=(qc - qc[:, I80:I80 + 1]).mean(0).tolist(), curve_true=(qt - qt[:, I80:I80 + 1]).mean(0).tolist(),
        n=int(len(qc)))


def main(rep, host=False):
    t0 = time.perf_counter()
    rng = np.random.default_rng([20261012, int(rep), 4401])
    S = Seed(rep, worlds=("stall", "no_stall"))
    log(f"seed {rep}: loaded in {S.load_seconds:.0f}s")
    sets = state_sets(S, rng)
    log(f"state sets built ({time.perf_counter() - t0:.0f}s)")
    critics = {k: S.h2_critic(k) for k in H2_CRITICS}
    critics["full_placebo"] = A.get_critic(S.prereg, S.st, "none", "stall_aware", placebo=True, train=False)[1]
    worlds = {k: ("no_stall" if k.startswith("no_stall") else "stall") for k in critics}
    phys = {w: S.sd[w].physics for w in ("stall", "no_stall")}
    sa = {w: RC.stall_aware(phys[w]) for w in phys}
    res = {}
    for sname, X in sets.items():
        res[sname] = dict(current_mean=float(X[:, TC.CURRENT].mean()),
                          push_mean=float(TC.push(X).mean()),
                          region_share=float(S.recorder.in_region(X).mean()))
        for bname in ("pattern", "controller"):
            base = np.broadcast_to(S.pattern, (len(X), 4)) if bname == "pattern" else sa["stall"](X)
            acts = sweep_actions(base, V_GRID)
            N, V = acts.shape[:2]
            Xr = np.repeat(X, V, 0)
            Ar = acts.reshape(-1, 4)
            truth = {w: true_q(sa[w], Xr, Ar, phys[w]).reshape(N, V) for w in phys}
            for cname, cp in critics.items():
                if cp is None:
                    continue
                w = worlds[cname]
                qc = critic_q(cp, S.prepared(w), Xr, Ar).reshape(N, V)
                res[sname][f"{bname}|{cname}"] = summarise(qc, truth[w])
        log(f"  {sname}: done ({time.perf_counter() - t0:.0f}s)  "
            + "  ".join(f"{c}: err {res[sname][f'pattern|{c}']['pref_error_mean']:+.3f}"
                        for c in ("gapped", "full", "random")))
    if host:
        hosts = {k: S.host(r) for k, r in (("host_gapped", "gap"), ("host_full", "none"))}
        vh = np.array([0.6, 0.8, 0.9, 1.0])
        for sname, X in sets.items():
            if sname not in HOST_SETS:
                continue
            X = X[:300]
            base = np.broadcast_to(S.pattern, (len(X), 4))
            acts = sweep_actions(base, vh)
            N, V = acts.shape[:2]
            Xr, Ar = np.repeat(X, V, 0), acts.reshape(-1, 4)
            Xr32 = np.asarray(np.asarray(Xr, np.float32), np.float64)
            Ar32 = np.asarray(np.asarray(Ar, np.float32), np.float64)
            for hname, hs in hosts.items():
                if hs is None:
                    continue
                qt = L.exact_q(hs.actor.params, S.pg, Xr32, Ar32, horizon=1000, x64=True).reshape(N, V)
                qc = critic_q(hs.critic.params, S.pg, Xr, Ar).reshape(N, V)
                pc, pt = qc[:, -1] - qc[:, 1], qt[:, -1] - qt[:, 1]
                res[sname][f"pattern|{hname}"] = dict(
                    pref_critic_mean=float(pc.mean()), pref_true_mean=float(pt.mean()),
                    pref_error_mean=float((pc - pt).mean()), pref_error_median=float(np.median(pc - pt)),
                    rise_0p6_to_0p8_critic=float((qc[:, 1] - qc[:, 0]).mean()),
                    rise_0p6_to_0p8_true=float((qt[:, 1] - qt[:, 0]).mean()), n=int(N),
                    note="truth = Q of the host's own actor (the policy this co-trained critic scores)")
            log(f"  host {sname}: done ({time.perf_counter() - t0:.0f}s)")
    write(f"critic_beliefs_seed{rep}.json", dict(
        kind="anatomy-critic-beliefs", rep=int(rep), v_grid=V_GRID, seconds=time.perf_counter() - t0,
        about="critic preference for thruster 1 past 80% (Q at v=1.0 minus Q at v=0.8) against exact truth",
        sets=res))


if __name__ == "__main__":
    main(int(sys.argv[1]), host="--host" in sys.argv)

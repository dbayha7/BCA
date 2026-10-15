"""READ-ONLY exact rollouts of the SAVED Stage 2 'none' actor. The GATED stratified band's deployment check:
gate = the 5% WBCP-posterior lower bound of the tail stratum's 90% overestimate quantile minus the bulk's plain 90%
quantile is > 0 (e11, score B). Gate on: the paired width U(s, a) - U(s, a_clip) > 0 exactly when t(a) = max_k |a_k|
> t_lo (the bulk's edge), so the check clips to the recorded range (Stage 2's fallback) every command past t_lo.
Gate off: the band is flat past the edge and the check never fires (no cost by construction).
Rules: gated (fallback = the recorded range, as Stage 2), random:gated (own rate), gated_tlo (fallback = clip to
[-t_lo, t_lo], a variant fallback), random:gated_tlo. Stalling and no-stall world, strong current and recorded."""
import json
import math
import sys
import time
import numpy as np
sys.path.insert(0, "/tmp/claude-0/-home-user/9e341792-006d-5a11-a8e9-8fac16ac7f5b/scratchpad/restructure/consumer")
import common as C  # noqa: E402
TC, TR, L = C.TC, C.TR, C.L
OUT = "/tmp/claude-0/-home-user/9e341792-006d-5a11-a8e9-8fac16ac7f5b/scratchpad/restructure/calibration"


def run(rule, actor, physics, anchor, x0, lo, hi, t_lo, gate, rate=None, coin=None):
    x = np.asarray(x0, np.float64).copy()
    n, T = len(x), physics.length
    J = np.zeros(n)
    fired = np.zeros((n, T), bool)
    pp = np.zeros((n, T), bool)
    rng = np.random.default_rng(coin) if coin is not None else None
    for t in range(T):
        a_act = np.clip(actor(x), -1.0, 1.0)
        fb = np.clip(a_act, lo, hi) if rule.endswith("gated") else np.clip(a_act, -t_lo, t_lo)
        if rule.startswith("random:"):
            fire = rng.uniform(size=n) < rate
        else:
            fire = (np.abs(a_act).max(1) > t_lo) if gate else np.zeros(n, bool)
        a = np.where(fire[:, None], fb, a_act)
        fired[:, t] = fire
        pp[:, t] = TC.past_knee(a_act, physics)
        x, r = TC.dynamics(x, a, physics)
        J = J + r
    pts = TR.points(J, anchor)
    return dict(points=float(pts.mean()), points_se=float(pts.std() / math.sqrt(n)), fire_share=float(fired.mean()),
                recall_past80=float((fired & pp).sum() / max(pp.sum(), 1)),
                flag_rate_not_past80=float((fired & ~pp).sum() / max((~pp).sum(), 1)))


gates = json.load(open(f"{OUT}/e11_credible_gap.json"))
e4 = {(r["seed"], r["world"]): r for r in json.load(open(f"{OUT}/e4_tailband.json"))}
allr = {}
t0 = time.perf_counter()
for rep in range(5):
    for wi, world in enumerate(("stall", "no_stall")):
        gate = gates[f"{rep}_{world}"]["B"]["post_q05_gap"] > 0
        t_lo = e4[(rep, world)]["t_lo"]
        sd, pg = C.seed_data(world, rep)
        lo, hi = C.recorded_range(pg)
        acts, _, _ = C.policy_actors(world, rep)
        pol = L.mlp_policy(acts["none"], pg, "none")
        an = C.anchors(world)
        for si, s in enumerate(("strong_current", "recorded")):
            x0 = TC.bank_starts(s, TC.BANK_SIZE)[:200]
            r = {"never": run("never_gated", pol, pg.physics, an[s], x0, lo, hi, t_lo, False)}
            for rule in ("gated", "gated_tlo"):
                r[rule] = run(rule, pol, pg.physics, an[s], x0, lo, hi, t_lo, gate)
                r[f"random:{rule}"] = run(f"random:{rule}", pol, pg.physics, an[s], x0, lo, hi, t_lo, gate,
                                          rate=r[rule]["fire_share"], coin=[20261014, rep, wi, si, 993])
                r[f"{rule}_minus_random"] = r[rule]["points"] - r[f"random:{rule}"]["points"]
                r[f"{rule}_minus_never"] = r[rule]["points"] - r["never"]["points"]
            allr[f"{rep}_{world}_{s}"] = dict(gate=bool(gate), t_lo=t_lo, **r)
            print(rep, world, s, "gate", gate, {k: (round(v["points"], 1), round(v["fire_share"], 2)) if isinstance(v, dict) else round(v, 1)
                                                for k, v in r.items()}, f"{time.perf_counter()-t0:.0f}s", flush=True)
json.dump(allr, open(f"{OUT}/e12_gated_deploy.json", "w"), indent=1)
for w in ("stall", "no_stall"):
    for s in ("strong_current", "recorded"):
        rs = [allr[f"{k}_{w}_{s}"] for k in range(5)]
        print(w, s, {k: round(float(np.mean([r[k]["points"] if isinstance(r[k], dict) else r[k] for r in rs])), 1)
                     for k in ("never", "gated", "random:gated", "gated_minus_random", "gated_minus_never",
                               "gated_tlo", "random:gated_tlo", "gated_tlo_minus_random", "gated_tlo_minus_never")})

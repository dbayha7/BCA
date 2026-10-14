"""READ-ONLY forward passes. Variant C2 of the throttle-stratified band: a NORMALISED score in the tail stratum, the
critic's excess overestimate as a share of its own claimed gain past the edge. No training, no fitting.

At each TAIL training row j (e4's strata; IN-SAMPLE rows of the frozen gapped critic):
  g_j  = Q1(s_j, a_j) - Q1(s_j, clip(a_j, -t_lo, t_lo))     the critic's claimed gain past the bulk's edge t_lo
  x_j  = -delta_j - median_bulk(-delta)                     its overestimate beyond the bulk's typical level
  score_j = x_j / max(g_j, g_floor), g_floor = the 10% quantile of |g| over TAIL rows (guards tiny denominators)
R_ratio = BCA's threshold rule (wbcp.calibrate, alpha 0.1, credibility 0.95, uniform weights) on the TAIL scores.
Width at any (s, a): U(s, a) = max(R_ratio, 0) [Q1(s, a) - Q1(s, clip(a, -t_lo, t_lo))]_+ ; paired against the
reference command as e6. The actor's objective Q1 - k U past the edge becomes (1 - k R) Q1(s, a) + k R Q1(s, cap):
with k R >= 1 the critic's claimed gain past the edge is cancelled at EVERY state (it scales with the claim, so it
extrapolates to strong current), with R ~ 0 nothing changes. Best response read-out as e6 (T1, T3a; k = 1, 2)."""
import hashlib
import json
import sys

import numpy as np
import jax
import jax.numpy as jnp
from flax import serialization

sys.path.insert(0, "/home/user/bca")
import algorithms.td3_bc as BASE  # noqa: E402
from calibration import wbcp  # noqa: E402

G = "/home/user/bca/runs/shift/68d9e8779e148b7b/critics/base"
ST2 = "/home/user/bca/runs/shift_explore/stage2/detect"
DET = "/home/user/bca/runs/shift_explore/detect"
OUT = "/tmp/claude-0/-home-user/9e341792-006d-5a11-a8e9-8fac16ac7f5b/scratchpad/restructure/calibration"
CRITIC = BASE.DualCritic(13, 4)
TEMPLATE = CRITIC.init(jax.random.PRNGKey(0), jnp.zeros(13), jnp.zeros(4))
Q = jax.jit(CRITIC.apply)
GRID = (0.70, 0.75, 0.78, 0.79, 0.80, 0.82, 0.84, 0.86, 0.88, 0.90, 0.95, 1.00)


def load_critic(world, seed):
    d = f"{G}/{world}/stall_aware_gap/seed{seed}"
    rec = json.load(open(f"{d}/record.json"))
    blob = open(f"{d}/critic.msgpack", "rb").read()
    assert hashlib.sha256(blob).hexdigest() == rec["result"]["file_sha256"] and not rec["result"]["failed"]
    return serialization.from_bytes(TEMPLATE, blob)


def q1(c, o, a):
    return np.asarray(Q(c, jnp.asarray(o, jnp.float32), jnp.asarray(a, jnp.float32)), np.float64)[:, 0]


def run(seed, world, E):
    d = np.load(f"{OUT}/data_seed{seed}_{world}.npz")
    dl = np.load(f"{OUT}/delta_seed{seed}_{world}.npz")["train_delta"]
    crit = load_critic(world, seed)
    t = np.abs(d["train_act"]).max(1)
    t_lo = E["t_lo"]
    tail = t >= E["t_up"] if E["t_up"] else np.zeros(len(t), bool)
    o_t, a_t = d["train_obs"][tail], d["train_act"][tail]
    g = q1(crit, o_t, a_t) - q1(crit, o_t, np.clip(a_t, -t_lo, t_lo))
    x = -dl[tail] - np.median(-dl[~tail])
    floor = float(np.quantile(np.abs(g), 0.1))
    sc = x / np.maximum(g, floor)
    rng = np.random.default_rng([20261015, 9300, seed, 0 if world == "stall" else 1])
    cal = wbcp.calibrate(sc, rng, alpha=0.1, beta=0.95, draws=1000)
    R = cal.threshold
    out = dict(seed=seed, world=world, n_tail=int(tail.sum()), g_median=float(np.median(g)),
               g_share_positive=float(np.mean(g > 0)), x_median=float(np.median(x)), score_median=float(np.median(sc)),
               R_ratio=float(R), R_ratio_plain=float(cal.lambda_hat))
    Rc = max(float(R) if np.isfinite(R) else float(cal.lambda_hat), 0.0)
    out["R_used"] = Rc
    out["R_used_note"] = "finite WBCP threshold" if np.isfinite(R) else "cannot certify: plain quantile shown (refusal reading = cap everything past the edge)"
    pz = np.load(f"{DET}/seed{seed}/{world}/pairs.npz")
    lab = np.load(f"{DET}/seed{seed}/{world}/labels.npz")
    rec = json.load(open(f"{ST2}/uniform/{world}/fixed/seed{seed}/band.json"))
    mean, std = np.asarray(rec["data"]["obs_mean"], np.float32), np.asarray(rec["data"]["obs_std"], np.float32)
    for setname, src in (("T1", None), ("T3a", "actor")):
        st = "T1" if setname == "T1" else "T3"
        m = (pz["set"] == st) & (pz["role"] == "command")
        if src:
            m &= pz["t3_source"] == src
        rows = np.flatnonzero(m)
        s = np.asarray(pz["states"][rows])
        o = (s.astype(np.float32) - mean) / std
        base = np.asarray(pz["commands"][rows], np.float32)
        ref = np.asarray(pz["refs"][rows], np.float32)
        cur = s[:, 10]
        harm = lab["harmful"][rows].astype(bool)

        def U(a):
            return Rc * np.maximum(q1(crit, o, a) - q1(crit, o, np.clip(a, -t_lo, t_lo)), 0.0)
        Uref = U(ref)
        Qg = np.stack([q1(crit, o, np.clip(base, -mm, mm)) for mm in GRID])
        Pg = np.stack([np.maximum(U(np.clip(base, -mm, mm)) - Uref, 0) for mm in GRID])
        gg = np.asarray(GRID)
        r = dict(q1_alone=float(np.mean(gg[Qg.argmax(0)] > 0.8)))
        for k in (1, 2):
            best = gg[(Qg - k * Pg).argmax(0)]
            r[f"k{k}_best_past80"] = float(np.mean(best > 0.8))
            if harm.any():
                r[f"k{k}_best_past80_on_harmful"] = float(np.mean(best[harm] > 0.8))
            if setname == "T3a":
                for clo, chi in ((0.7, 0.9), (0.9, 1.0), (1.0, 1.2)):
                    mm_ = (cur >= clo) & (cur < chi)
                    r[f"k{k}_best_past80_cur[{clo},{chi})"] = float(np.mean(best[mm_] > 0.8))
        out[setname] = r
    return out


if __name__ == "__main__":
    e4 = {(r["seed"], r["world"]): r for r in json.load(open(f"{OUT}/e4_tailband.json"))}
    res = []
    for world in ("stall", "no_stall"):
        for seed in range(5):
            r = run(seed, world, e4[(seed, world)])
            res.append(r)
            print(seed, world, "n", r["n_tail"], "g>0", round(r["g_share_positive"], 2), "g med", round(r["g_median"], 3),
                  "x med", round(r["x_median"], 3), "R", round(r["R_ratio"], 3), "plain", round(r["R_ratio_plain"], 3),
                  "T1", {k: round(v, 3) for k, v in r["T1"].items()}, "T3a", {k: round(v, 3) for k, v in r["T3a"].items() if "cur" not in k}, flush=True)
    pooled = {}
    for world in ("stall", "no_stall"):
        rs = [r for r in res if r["world"] == world]
        for sn in ("T1", "T3a"):
            pooled[f"{world}_{sn}"] = {k: float(np.mean([r[sn][k] for r in rs])) for k in rs[0][sn]}
    json.dump(dict(per=res, pooled=pooled), open(f"{OUT}/e7_ratio.json", "w"), indent=1)
    for k, v in pooled.items():
        print(k, {kk: round(vv, 3) for kk, vv in v.items()})

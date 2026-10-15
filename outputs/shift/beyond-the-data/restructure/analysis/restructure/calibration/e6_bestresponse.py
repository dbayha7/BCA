"""READ-ONLY forward passes. Best response of the actor's objective Q1 - k P along a throttle cap, at the locked T1
states (full-throttle-region training states) and T3a states (strong-current states the no-caution actor visits),
seeds 0-4, both worlds. No training, no fitting.

At each state: base = the no-caution actor's command (pair role 'command'); a_m = clip(base, -m, m) for m on GRID
(m = 1 is the command itself); a_ref = the pair's reference command (recorded command at T1, the cloned actor's
command at T3: pairs.npz 'refs'). P(s, a) = [U(s, a) - U(s, a_ref)]_+ for each width U:
  tail_B   R_B(t(a)) u_f                                   (e4: one-sided, throttle-stratified BCA threshold)
  tail_A   R_A(t(a)) u_f max(eta_f(s, a), 1e-6)            (e4: BCA's own score, throttle-stratified)
  weighted the saved Stage 2 weighted band (eta_f, classifier, threshold table, refusal term), as impact/exact.py
  hinge    100 sum_k [a_k - hi_k]_+^2 + [lo_k - a_k]_+^2 (hand-set strength; unpaired, as the room runs it)
Q1 = the frozen gapped fixed-controller critic's first head (SHA-256 checked). Reports, per k in (1, 2): the share
of states whose best grid cap is past 80%, and (T3a) by current bin."""
import hashlib
import json
import sys

import numpy as np
import jax
import jax.numpy as jnp
import flax.linen as nn
from flax import serialization

sys.path.insert(0, "/home/user/bca")
import algorithms.td3_bc as BASE  # noqa: E402
from calibration.network import Calibrator  # noqa: E402

G = "/home/user/bca/runs/shift/68d9e8779e148b7b/critics/base"
ST2 = "/home/user/bca/runs/shift_explore/stage2/detect"
DET = "/home/user/bca/runs/shift_explore/detect"
OUT = "/tmp/claude-0/-home-user/9e341792-006d-5a11-a8e9-8fac16ac7f5b/scratchpad/restructure/calibration"
CRITIC = BASE.DualCritic(13, 4)
TEMPLATE = CRITIC.init(jax.random.PRNGKey(0), jnp.zeros(13), jnp.zeros(4))
Q = jax.jit(CRITIC.apply)
CAL = Calibrator(jnp.zeros(13), jnp.ones(13), state_dep=True)
CALA = jax.jit(CAL.apply)
GRID = (0.70, 0.75, 0.78, 0.79, 0.80, 0.82, 0.84, 0.86, 0.88, 0.90, 0.95, 1.00)
KS = (1, 2)


class ClassifierNet(nn.Module):  # stage2/code/weights.py ClassifierNet, restated
    @nn.compact
    def __call__(self, obs, action):
        x = jnp.concatenate([obs / (1.0 + 0.001), action], axis=-1)
        x = nn.relu(nn.Dense(64)(x))
        x = nn.relu(nn.Dense(64)(x))
        return nn.Dense(1)(x).squeeze(-1)


CLSA = jax.jit(ClassifierNet().apply)


def load_critic(world, seed):
    d = f"{G}/{world}/stall_aware_gap/seed{seed}"
    rec = json.load(open(f"{d}/record.json"))
    blob = open(f"{d}/critic.msgpack", "rb").read()
    assert hashlib.sha256(blob).hexdigest() == rec["result"]["file_sha256"] and not rec["result"]["failed"]
    return serialization.from_bytes(TEMPLATE, blob)


def r_of_t(t, t_lo, t_hi, Rb, Rt):
    if t_hi is None or not np.isfinite(Rt):
        return np.full(t.shape, Rb)
    f = np.clip((t - t_lo) / (t_hi - t_lo), 0, 1)
    return Rb + f * (Rt - Rb)


def run(seed, world, e4):
    pz = np.load(f"{DET}/seed{seed}/{world}/pairs.npz")
    lab = np.load(f"{DET}/seed{seed}/{world}/labels.npz")
    rec = json.load(open(f"{ST2}/uniform/{world}/fixed/seed{seed}/band.json"))
    mean, std = np.asarray(rec["data"]["obs_mean"], np.float32), np.asarray(rec["data"]["obs_std"], np.float32)
    bw = serialization.msgpack_restore(open(f"{ST2}/weighted/{world}/fixed/seed{seed}/band.msgpack", "rb").read())
    bu = serialization.msgpack_restore(open(f"{ST2}/uniform/{world}/fixed/seed{seed}/band.msgpack", "rb").read())
    crit = load_critic(world, seed)
    d = np.load(f"{OUT}/data_seed{seed}_{world}.npz")
    a_tr = d["train_act"].astype(np.float64)
    lo, hi = np.quantile(a_tr, 0.005, axis=0), np.quantile(a_tr, 0.995, axis=0)
    unit = float(bu["unit"])
    E = e4[(seed, world)]
    out = {}
    for setname, src in (("T1", None), ("T3a", "actor")):
        st = "T1" if setname == "T1" else "T3"
        m = (pz["set"] == st) & (pz["role"] == "command")
        if src:
            m &= pz["t3_source"] == src
        rows = np.flatnonzero(m)
        s = np.asarray(pz["states"][rows])
        o = jnp.asarray((s.astype(np.float32) - mean) / std)
        base = np.asarray(pz["commands"][rows], np.float32)
        ref = np.asarray(pz["refs"][rows], np.float32)
        cur = s[:, 10]
        harmful = lab["harmful"][rows].astype(bool)

        def widths(a):
            a32 = jnp.asarray(a, jnp.float32)
            t = np.abs(np.asarray(a, np.float64)).max(1)
            eta = np.maximum(np.asarray(CALA(bu["cal_params"], o, a32), np.float64), 1e-6)
            lw = np.asarray(CLSA(bw["cls_params"], o, a32), np.float64)
            Rw = np.interp(lw, np.asarray(bw["log_t"], np.float64), np.asarray(bw["R_tab"], np.float64))
            Uw = Rw * float(bw["unit"]) * eta + float(bw["U_max"]) * np.logaddexp(0.0, lw - float(bw["l_star"]))
            aa = np.asarray(a, np.float64)
            hinge = 100 * (np.maximum(aa - hi, 0) ** 2 + np.maximum(lo - aa, 0) ** 2).sum(1)
            return dict(tail_B=r_of_t(t, E["t_lo"], E["t_hi"], E["R_B"]["bulk"], E["R_B"]["tail"]) * unit,
                        tail_A=r_of_t(t, E["t_lo"], E["t_hi"], E["R_A"]["bulk"], E["R_A"]["tail"]) * unit * eta,
                        weighted=Uw, hinge=hinge)
        Wref = widths(ref)
        q1 = np.zeros((len(GRID), len(rows)))
        P = {k: np.zeros((len(GRID), len(rows))) for k in Wref}
        for i, mm in enumerate(GRID):
            a = np.clip(base.astype(np.float64), -mm, mm).astype(np.float32)
            q1[i] = np.asarray(Q(crit, o, jnp.asarray(a)), np.float64)[:, 0]
            Wm = widths(a)
            for k in Wref:
                P[k][i] = Wm[k] if k == "hinge" else np.maximum(Wm[k] - Wref[k], 0)
        g = np.asarray(GRID)
        res = dict(n=int(len(rows)), q1_alone_best_past80=float(np.mean(g[q1.argmax(0)] > 0.8)),
                   actor_past80=float(np.mean(np.abs(base).max(1) > np.float32(0.8))))
        for k in P:
            for kk in KS:
                best = g[(q1 - kk * P[k]).argmax(0)]
                res[f"{k}_k{kk}_best_past80"] = float(np.mean(best > 0.8))
                res[f"{k}_k{kk}_best_ge95"] = float(np.mean(best >= 0.95))
                if harmful.any():
                    res[f"{k}_k{kk}_best_past80_on_harmful"] = float(np.mean(best[harmful] > 0.8))
                if setname == "T3a":
                    for clo, chi in ((0.7, 0.9), (0.9, 1.0), (1.0, 1.2)):
                        mm_ = (cur >= clo) & (cur < chi)
                        res[f"{k}_k{kk}_best_past80_cur[{clo},{chi})"] = float(np.mean(best[mm_] > 0.8))
            res[f"{k}_P_at_1.0_mean"] = float(P[k][-1].mean())
            res[f"{k}_P_at_0.80_mean"] = float(P[k][GRID.index(0.80)].mean())
        if setname == "T3a":
            for clo, chi in ((0.7, 0.9), (0.9, 1.0), (1.0, 1.2)):
                mm_ = (cur >= clo) & (cur < chi)
                res[f"q1_alone_best_past80_cur[{clo},{chi})"] = float(np.mean(g[q1.argmax(0)][mm_] > 0.8))
                res[f"q1_gain_1.0_minus_0.8_cur[{clo},{chi})"] = float(np.mean((q1[-1] - q1[GRID.index(0.80)])[mm_]))
        res["q1_gain_1.0_minus_0.8_mean"] = float(np.mean(q1[-1] - q1[GRID.index(0.80)]))
        out[setname] = res
    return out


if __name__ == "__main__":
    e4 = {(r["seed"], r["world"]): r for r in json.load(open(f"{OUT}/e4_tailband.json"))}
    allr = []
    for world in ("stall", "no_stall"):
        for seed in range(5):
            r = run(seed, world, e4)
            r.update(seed=seed, world=world)
            allr.append(r)
            for sn in ("T1", "T3a"):
                x = r[sn]
                print(seed, world, sn, "Q1 alone", round(x["q1_alone_best_past80"], 3),
                      {k: round(x[f"{k}_k1_best_past80"], 3) for k in ("tail_B", "tail_A", "weighted", "hinge")},
                      "k2", {k: round(x[f"{k}_k2_best_past80"], 3) for k in ("tail_B", "tail_A", "weighted")}, flush=True)
    json.dump(allr, open(f"{OUT}/e6_bestresponse.json", "w"), indent=1)
    # pooled
    pooled = {}
    for world in ("stall", "no_stall"):
        rs = [r for r in allr if r["world"] == world]
        for sn in ("T1", "T3a"):
            keys = rs[0][sn].keys()
            pooled[f"{world}_{sn}"] = {k: float(np.mean([r[sn][k] for r in rs])) for k in keys}
    json.dump(pooled, open(f"{OUT}/e6_bestresponse_pooled.json", "w"), indent=1)
    for k, v in pooled.items():
        print(k, {kk: round(vv, 3) for kk, vv in v.items() if "past80" in kk and "ge95" not in kk})

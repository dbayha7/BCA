"""READ-ONLY. Penalty shapes along thruster-1 throttle at the 1,500 T1 states (full-throttle-region training states),
seeds 0-4, both worlds -- the room's check-(e) protocol restated on the locked T1 pairs (no shape_rows: no data loader).

At each T1 state: base = the no-caution actor's command (T1 'command' role), sign = its thruster-1 push direction,
a_m = base with thruster 1 set to sign*m for m in GRID; a_rec = the row's recorded command (T1 'recorded').
Penalty forms (paired as BCA's lower edge: P(a) = [W(s, a) - W(s, a_rec)]_+), all from SAVED artifacts, no fitting:
  E2      W = sample std of the gapped critic's two heads (= |Q1-Q2|/sqrt 2)
  EI      W = sample std of 4 heads of the full + full_placebo critics (stall only; same data, two inits)
  bca_w   W = the saved Stage 2 weighted detection band (eta_f, classifier fitted with this same no-caution actor,
              threshold table, refusal term) -- exactly as impact/exact.py restates band.width
  bca_u   W = the saved uniform band R u_f eta_f
  slope   W = sum_k g_k(s) e_k(a): e_k = excess of a_k beyond the recorded range [lo_k, hi_k] (the hinge's own
              bounds, room lock.json:shared.recorded_range), g_k = |dQ1/da_k| at the projection of a onto the box
              (the critic's own slope where it still has data). Candidate 'extrapolation-slope prior' term.
  hinge   W = 100 * sum_k [a_k - hi_k]_+^2 + [lo_k - a_k]_+^2 (the room's range hinge, unpaired as the room runs it)
Per form and k in (1, 2, 4): share of states whose best grid throttle under Q1 - k P is past 80% (Q1 alone for
reference), mean P along the grid, and mean slope dP/dm on [0.80, 0.85] and [0.85, 0.95] relative to P(1.0)-P(0.8).
"""
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
ROOM = "/home/user/bca/runs/shift_explore/room"
OUT = "/tmp/claude-0/-home-user/9e341792-006d-5a11-a8e9-8fac16ac7f5b/scratchpad/restructure/action_aware"
DX, DA = 13, 4
CRITIC = BASE.DualCritic(DX, DA)
TEMPLATE = CRITIC.init(jax.random.PRNGKey(0), jnp.zeros(DX), jnp.zeros(DA))
APPLY = jax.jit(CRITIC.apply)
GRADQ1 = jax.jit(jax.grad(lambda p, o, a: CRITIC.apply(p, o, a)[..., 0].sum(), argnums=2))
GRID = (0.70, 0.75, 0.78, 0.79, 0.80, 0.805, 0.81, 0.82, 0.83, 0.84, 0.85, 0.86, 0.88, 0.90, 0.95, 1.00)
KS = (1, 2, 4)
CAL = Calibrator(jnp.zeros(13), jnp.ones(13), state_dep=True)


class ClassifierNet(nn.Module):  # copy of stage2/code/weights.py ClassifierNet (as impact/exact.py)
    hidden: int = 64

    @nn.compact
    def __call__(self, obs, action):
        x = jnp.concatenate([obs / (1.0 + 0.001), action], axis=-1)
        x = nn.relu(nn.Dense(self.hidden)(x))
        x = nn.relu(nn.Dense(self.hidden)(x))
        return nn.Dense(1)(x).squeeze(-1)


CLS = ClassifierNet()
CAL_APPLY, CLS_APPLY = jax.jit(CAL.apply), jax.jit(CLS.apply)


def load_critic(world, name, seed):
    d = f"{G}/{world}/{name}/seed{seed}"
    rec = json.load(open(f"{d}/record.json"))
    blob = open(f"{d}/critic.msgpack", "rb").read()
    assert hashlib.sha256(blob).hexdigest() == rec["result"]["file_sha256"]
    return serialization.from_bytes(TEMPLATE, blob)


def idx(pz, st, role):
    m = (pz["set"] == st) & (pz["role"] == role)
    rows = np.flatnonzero(m)
    return rows[np.argsort(pz["pair"][rows])]


def bounds(world, seed):
    f = f"{ROOM}/seed{seed}/lock.json" if world == "stall" else f"{ROOM}/seed{seed}/no_stall/lock.json"
    rr = json.load(open(f))["shared"]["recorded_range"]
    return np.asarray(rr["lo"], np.float64), np.asarray(rr["hi"], np.float64)


def one(seed, world):
    pz = np.load(f"{DET}/seed{seed}/{world}/pairs.npz")
    lab = np.load(f"{DET}/seed{seed}/{world}/labels.npz")
    data = json.load(open(f"{ST2}/uniform/{world}/fixed/seed{seed}/band.json"))["data"]
    mean, std = np.asarray(data["obs_mean"], np.float32), np.asarray(data["obs_std"], np.float32)
    ic, ir = idx(pz, "T1", "command"), idx(pz, "T1", "recorded")
    obs = ((np.asarray(pz["states"][ic], np.float32) - mean) / std).astype(np.float32)
    base = np.asarray(pz["commands"][ic], np.float64)
    arec = np.asarray(pz["commands"][ir], np.float32)
    assert np.allclose(pz["states"][ic], pz["states"][ir])
    sign = np.where(base[:, 0] < 0, -1.0, 1.0)
    lo, hi = bounds(world, seed)
    crit = {"gap": load_critic(world, "stall_aware_gap", seed)}
    if world == "stall":
        crit["full"] = load_critic(world, "stall_aware_full", seed)
        crit["plac"] = load_critic(world, "stall_aware_full_placebo", seed)
    bw = serialization.msgpack_restore(open(f"{ST2}/weighted/{world}/fixed/seed{seed}/band.msgpack", "rb").read())
    bu = serialization.msgpack_restore(open(f"{ST2}/uniform/{world}/fixed/seed{seed}/band.msgpack", "rb").read())
    log_t, R_tab = np.asarray(bw["log_t"], np.float64), np.asarray(bw["R_tab"], np.float64)

    def widths(a):
        a32 = np.asarray(a, np.float32)
        o, aj = jnp.asarray(obs), jnp.asarray(a32)
        Qg = np.asarray(APPLY(crit["gap"], o, aj), np.float64)
        W = {"E2": Qg.std(1, ddof=1)}
        if world == "stall":
            Qf = np.asarray(APPLY(crit["full"], o, aj), np.float64)
            Qp = np.asarray(APPLY(crit["plac"], o, aj), np.float64)
            W["EI"] = np.concatenate([Qf, Qp], 1).std(1, ddof=1)
        eta_w = np.asarray(CAL_APPLY(bw["cal_params"], o, aj), np.float64)
        lw = np.asarray(CLS_APPLY(bw["cls_params"], o, aj), np.float64)
        W["bca_w"] = np.interp(lw, log_t, R_tab) * float(bw["unit"]) * np.maximum(eta_w, 1e-6) + \
            float(bw["U_max"]) * np.logaddexp(0.0, lw - float(bw["l_star"]))
        eta_u = np.asarray(CAL_APPLY(bu["cal_params"], o, aj), np.float64)
        W["bca_u"] = float(bu["R"]) * float(bu["unit"]) * np.maximum(eta_u, 1e-6)
        proj = np.clip(np.asarray(a, np.float64), lo, hi).astype(np.float32)
        g = np.abs(np.asarray(GRADQ1(crit["gap"], o, jnp.asarray(proj)), np.float64))
        e = np.maximum(np.asarray(a, np.float64) - hi, 0) + np.maximum(lo - np.asarray(a, np.float64), 0)
        W["slope"] = (g * e).sum(1)
        W["hinge"] = 100.0 * (np.maximum(np.asarray(a, np.float64) - hi, 0) ** 2 +
                              np.maximum(lo - np.asarray(a, np.float64), 0) ** 2).sum(1)
        return Qg[:, 0], W

    _, Wrec = widths(arec)
    Q1, P = {}, {}
    for m in GRID:
        a = base.copy()
        a[:, 0] = sign * m
        q1, W = widths(a)
        Q1[m] = q1
        P[m] = {f: (W[f] if f == "hinge" else np.maximum(W[f] - Wrec[f], 0.0)) for f in W}
    grid = np.array(GRID)
    Qm = np.stack([Q1[m] for m in GRID], 1)
    out = dict(seed=seed, world=world, n=int(len(ic)), lo=lo.tolist(), hi=hi.tolist(),
               actor_thr1_past80_share=float(np.mean(np.abs(base[:, 0]) > 0.8)),
               q1_alone_best_past80_share=float(np.mean(grid[np.argmax(Qm, 1)] > 0.8)),
               q1_gain_1p0_minus_0p8_mean=float(np.mean(Q1[1.0] - Q1[0.8])),
               q1_gain_0p8_minus_0p75_mean=float(np.mean(Q1[0.8] - Q1[0.75])), forms={})
    harm = lab["harmful"][ic].astype(bool)
    for f in P[GRID[0]]:
        Pm = np.stack([P[m][f] for m in GRID], 1)
        span = float(np.mean(P[1.0][f] - P[0.8][f]))
        rec = dict(mean_P={f"{m:g}": float(P[m][f].mean()) for m in GRID}, span_0p8_to_1p0=span,
                   slope_80_85=float(np.mean(P[0.85][f] - P[0.8][f]) / 0.05),
                   slope_85_95=float(np.mean(P[0.95][f] - P[0.85][f]) / 0.10),
                   slope_70_80=float(np.mean(P[0.8][f] - P[0.7][f]) / 0.10),
                   ratio_span_over_q1gain=float(span / max(out["q1_gain_1p0_minus_0p8_mean"], 1e-9)))
        for k in KS:
            best = grid[np.argmax(Qm - k * Pm, 1)]
            rec[f"best_past80_share_k{k}"] = float(np.mean(best > 0.8))
            rec[f"best_ge95_share_k{k}"] = float(np.mean(best >= 0.95))
            if harm.any():
                rec[f"best_past80_share_on_harmful_k{k}"] = float(np.mean(best[harm] > 0.8))
        out["forms"][f] = rec
    return out


if __name__ == "__main__":
    res = []
    for world in ("stall", "no_stall"):
        for seed in range(5):
            r = one(seed, world)
            res.append(r)
            print(world, seed, "q1-alone past80", r["q1_alone_best_past80_share"], "actor past80",
                  round(r["actor_thr1_past80_share"], 3), "Q1 gain 0.8->1.0", round(r["q1_gain_1p0_minus_0p8_mean"], 3),
                  flush=True)
            for f, d in r["forms"].items():
                print(f"   {f:6s} span={d['span_0p8_to_1p0']:.4f} ratio_to_q1gain={d['ratio_span_over_q1gain']:.3f} "
                      f"slope70-80/80-85/85-95={d['slope_70_80']:.3f}/{d['slope_80_85']:.3f}/{d['slope_85_95']:.3f} "
                      f"best>80 k1/2/4={d['best_past80_share_k1']:.3f}/{d['best_past80_share_k2']:.3f}/"
                      f"{d['best_past80_share_k4']:.3f} best>=95 k1={d['best_ge95_share_k1']:.3f}", flush=True)
    json.dump(res, open(f"{OUT}/shape_sweep.json", "w"), indent=1)

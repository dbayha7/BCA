"""READ-ONLY counterfactual. BCA's band with a THROTTLE-STRATIFIED (Mondrian) threshold, from saved files only.

No network is trained or fitted. Inputs: the SAVED uniform fixed-critic band (eta_f, u_f), the one-step errors of the
SAVED gapped fixed-controller critic at the kept training rows (e3_delta.py, forward passes), the locked detection
pairs and labels, and the saved rival scores. Thresholds are BCA's own rule (calibration/wbcp.calibrate: alpha 0.1,
credibility 0.95, 1,000 draws, uniform weights) applied per stratum: a closed-form quantile computation, no fit.

Design (angle C). Throttle index t(a) = max_k |a_k|. The kept training commands have a bulk (t up to ~0.8) and a
saturated atom at full throttle (t >= 0.99, 28-59 rows per seed and world) separated by an empty gap. Strata, by a
rule that reads only the training commands: sort t over rows above the 99th percentile; the largest gap between
consecutive values splits BULK (below) from TAIL (above) if the gap is >= 0.05 and TAIL has >= 9 rows; else no TAIL.
  R_bulk = WBCP threshold of the scores of BULK rows, R_tail = that of TAIL rows (+inf if not certifiable).
  R(t) = R_bulk for t <= t_lo (the gap's lower end); linear from (t_lo, R_bulk) to (t_hi, R_tail) across the gap
         (t_hi = median t of TAIL); R_tail beyond. If R_tail is +inf: R_c(t) = the gap ramp to max finite candidate,
         and the pair is marked 'cannot certify' past t_lo (BCA's refusal).
Two score forms:
  A  (BCA's own score): |delta| / (u_f max(eta_f(s, a), 1e-6));  width U_A(s, a) = R_A(t(a)) u_f max(eta_f(s, a), 1e-6)
  B  (one-sided lower edge): -delta / u_f (the critic's overestimate);  width U_B(s, a) = R_B(t(a)) u_f
delta = r + gamma min_h Q(s', c(s')) - min_h Q(s, a) at kept training rows (IN-SAMPLE for the critic; the bank keeps
0-2 full-throttle rows, too few for a tail stratum). Penalty, as Stage 2: W(s, a) = [U(s, a) - U(s, a_ref)]_+.
Controls: SHUFFLED (scores permuted across all training rows, 20 permutations: any ramp then comes only from the
TAIL's small n under BCA's credibility rule, not from the residuals there).
Writes e4_tailband.json and per-pair widths e4_widths_seed<K>_<world>.npz to this folder only."""
import json
import sys

import numpy as np
import jax
import jax.numpy as jnp
from flax import serialization
from scipy.stats import rankdata

sys.path.insert(0, "/home/user/bca")
from calibration import wbcp  # noqa: E402
from calibration.network import Calibrator  # noqa: E402

ST2 = "/home/user/bca/runs/shift_explore/stage2/detect"
DET = "/home/user/bca/runs/shift_explore/detect"
OUT = "/tmp/claude-0/-home-user/9e341792-006d-5a11-a8e9-8fac16ac7f5b/scratchpad/restructure/calibration"
IMP = "/tmp/claude-0/-home-user/9e341792-006d-5a11-a8e9-8fac16ac7f5b/scratchpad/impact"
CAL = Calibrator(jnp.zeros(13), jnp.ones(13), state_dep=True)
APPLY = jax.jit(CAL.apply)
ALPHA, BETA, DRAWS = 0.1, 0.95, 1000
PR = {"T1": ("command", "cap", "recorded", "cloned"), "T2": ("command", "cap", "recorded", "cloned"),
      "T3": ("command", "cap")}
RIVALS = ("classifier_ratio", "distance", "range_hinge", "perfect_band", "density", "two_head", "residual_evidence")
N_SHUF = 20


def auroc(s, p):
    p = np.asarray(p, bool)
    n1, n0 = int(p.sum()), int((~p).sum())
    if n1 == 0 or n0 == 0:
        return None
    r = rankdata(np.asarray(s, np.float64))
    return float((r[p].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def view(arr):
    out = {}
    for st in ("T1", "T2", "T3"):
        m = arr["set"] == st
        out[st] = {}
        for r in np.unique(arr["role"][m]):
            rows = np.flatnonzero(m & (arr["role"] == r))
            out[st][r] = rows[np.argsort(arr["pair"][rows])]
    return out


def thr(scores, seed_tag):
    """BCA's threshold rule (wbcp.calibrate), uniform weights; returns (R, lambda_hat)."""
    rng = np.random.default_rng([20261015, 9100, *seed_tag])
    c = wbcp.calibrate(np.asarray(scores, np.float64), rng, alpha=ALPHA, beta=BETA, draws=DRAWS)
    return float(c.threshold), float(c.lambda_hat)


def strata(t):
    hi_rows = np.sort(t[t > np.quantile(t, 0.99)])
    g = np.diff(hi_rows)
    i = int(np.argmax(g))
    t_lo, t_up = float(hi_rows[i]), float(hi_rows[i + 1])
    tail = t >= t_up
    ok = (t_up - t_lo) >= 0.05 and tail.sum() >= 9
    return (t_lo, t_up, tail) if ok else (float(t.max()), None, np.zeros(len(t), bool))


def r_of_t(tq, t_lo, t_hi, Rb, Rt):
    """Piecewise-linear threshold in the throttle index (module docstring)."""
    Rt_c = Rt if np.isfinite(Rt) else None
    if t_hi is None or Rt_c is None:
        return np.full(len(tq), Rb), tq > t_lo
    f = np.clip((tq - t_lo) / max(t_hi - t_lo, 1e-9), 0.0, 1.0)
    return Rb + f * (Rt_c - Rb), np.zeros(len(tq), bool)


def paired(U, v, st, role, ref="recorded"):
    return np.maximum(U[v[st][role]] - U[v[st][ref]], 0.0)


def metrics(U, v, H, past, over, arr, cur, t3src, tau_pool):
    """Detection read-outs of one width U at the pairs (stall-world marks need harmful pairs)."""
    o = {}
    ic, ip, ir = v["T1"]["command"], v["T1"]["cap"], v["T1"]["recorded"]
    sel = H[ic] if H[ic].any() else past[ic]
    Wc, Wp = np.maximum(U[ic] - U[ir], 0), np.maximum(U[ip] - U[ir], 0)
    o["T1_direction_share"] = float(np.mean((U[ic] > U[ip])[sel]))
    o["T1_size_median"] = float(np.median(((Wc - Wp) / over[ic])[sel])) if H[ic].any() else None
    o["T1_penalty_cmd_mean_past80"] = float(np.mean(Wc[past[ic]]))
    o["T1_penalty_minus_cap_mean_past80"] = float(np.mean((Wc - Wp)[past[ic]]))
    r12 = np.concatenate([v[st][r] for st in ("T1", "T2") for r in PR[st]])
    r3 = np.concatenate([v["T3"][r] for r in PR["T3"]])
    o["rank_T12"] = auroc(U[r12], H[r12])
    o["rank_T3"] = auroc(U[r3], H[r3])
    # T3a (strong-current states visited by the no-caution actor): past-80% detection among the actor's commands,
    # by current bin (the M5 situation)
    c3 = v["T3"]["command"]
    a_src = t3src[c3] == "actor"
    for lo, hi in ((0.7, 0.9), (0.9, 1.0), (1.0, 1.2)):
        m = a_src & (cur[c3] >= lo) & (cur[c3] < hi)
        o[f"T3a_auc_past80_cur[{lo},{hi})"] = auroc(U[c3][m], past[c3][m])
    m = a_src
    o["T3a_auc_past80_all"] = auroc(U[c3][m], past[c3][m])
    # deployment-check proxy: tau = 95% level of U on T4 (held-out recorded rows) -> fire share at T3a commands
    tau = float(np.quantile(U[tau_pool], 0.95))
    fire = U[c3][m] > tau
    o["T3a_fire_share"] = float(np.mean(fire))
    o["T3a_recall_past80"] = float(np.mean(fire[past[c3][m]])) if past[c3][m].any() else None
    o["T3a_false_flag"] = float(np.mean(fire[~past[c3][m]])) if (~past[c3][m]).any() else None
    o["T3a_past80_share"] = float(np.mean(past[c3][m]))
    o["T1_fire_share_cmd_past80"] = float(np.mean(U[ic][past[ic]] > tau))
    o["T1_fire_share_cap"] = float(np.mean(U[ip] > tau))
    return o


def one(seed, world):
    d = np.load(f"{OUT}/data_seed{seed}_{world}.npz")
    dl = np.load(f"{OUT}/delta_seed{seed}_{world}.npz")["train_delta"]
    brec = json.load(open(f"{ST2}/uniform/{world}/fixed/seed{seed}/band.json"))
    bu = serialization.msgpack_restore(open(f"{ST2}/uniform/{world}/fixed/seed{seed}/band.msgpack", "rb").read())
    unit = float(bu["unit"])
    eta_tr = np.maximum(np.asarray(APPLY(bu["cal_params"], jnp.asarray(d["train_obs"]), jnp.asarray(d["train_act"])),
                                   np.float64), 1e-6)
    t_tr = np.abs(d["train_act"]).max(1).astype(np.float64)
    sA = np.abs(dl) / (unit * eta_tr)
    sB = -dl / unit
    t_lo, t_up, tail = strata(t_tr)
    t_hi = float(np.median(t_tr[tail])) if tail.any() else None
    rec = dict(seed=seed, world=world, t_lo=t_lo, t_up=t_up, t_hi=t_hi, n_tail=int(tail.sum()), n_bulk=int((~tail).sum()),
               unit=unit, R_uniform_bank=float(bu["R"]))
    R = {}
    for nm, s in (("A", sA), ("B", sB)):
        Rb, lb = thr(s[~tail], (seed, 0 if world == "stall" else 1, ord(nm), 0))
        Rt, lt = thr(s[tail], (seed, 0 if world == "stall" else 1, ord(nm), 1)) if tail.any() else (np.inf, np.inf)
        rng = np.random.default_rng([20261015, 9200, seed, ord(nm)])
        sh = []
        Rall = thr(s, (seed, 7, ord(nm), 99))[0]   # under a permutation the bulk is ~all rows (n_tail << n)
        for j in range(N_SHUF):
            ps = s[rng.permutation(len(s))]
            sh.append((Rall, thr(ps[tail], (seed, 7, ord(nm), j, 1))[0] if tail.any() else np.inf))
        sh = np.asarray(sh)
        R[nm] = (Rb, Rt)
        rec[f"R_{nm}"] = dict(bulk=Rb, tail=Rt, bulk_plain=lb, tail_plain=lt, tail_over_bulk=Rt / Rb if nm == "A" else None,
                             tail_minus_bulk_value=(Rt - Rb) * unit * (float(np.median(eta_tr)) if nm == "A" else 1.0),
                             shuffled_tail_minus_bulk_value=float(np.median(sh[:, 1] - sh[:, 0])) * unit *
                             (float(np.median(eta_tr)) if nm == "A" else 1.0),
                             shuffled_tail_minus_bulk_value_q90=float(np.quantile(sh[:, 1] - sh[:, 0], 0.9)) * unit *
                             (float(np.median(eta_tr)) if nm == "A" else 1.0),
                             shuffled_certified_share=float(np.mean(np.isfinite(sh[:, 1]))))
    # pairs
    pz = np.load(f"{DET}/seed{seed}/{world}/pairs.npz")
    arr = {k: pz[k] for k in ("set", "pair", "role", "t3_source")}
    lab = np.load(f"{DET}/seed{seed}/{world}/labels.npz")
    H, past, over = lab["harmful"].astype(bool), lab["unsupported"].astype(bool), lab["over_pref"]
    T = np.load(f"{IMP}/terms_seed{seed}_{world}.npz")
    eta_p = np.maximum(T["eta"], 1e-6)
    t_p = np.abs(pz["commands"]).max(1).astype(np.float64)
    cur = np.asarray(pz["states"])[:, 10]
    v = view(arr)
    t4 = arr["set"] == "T4"
    widths = {}
    RA, refA = r_of_t(t_p, t_lo, t_hi, *R["A"])
    RB, refB = r_of_t(t_p, t_lo, t_hi, *R["B"])
    widths["tail_A"] = RA * unit * eta_p
    widths["tail_B"] = RB * unit
    rec["cannot_certify_share_T1cmd_A"] = float(np.mean(refA[v["T1"]["command"]]))
    widths["weighted_bca"], widths["uniform_bca"] = T["Uw"], T["Uu"]
    widths["eta_only"] = eta_p
    for r in RIVALS:
        widths[r] = np.load(f"{DET}/seed{seed}/{world}/scores/{r}.npz")["scores"]
    np.savez(f"{OUT}/e4_widths_seed{seed}_{world}.npz", **widths)
    rec["metrics"] = {}
    for nm, U in widths.items():
        if nm == "perfect_band":   # already paired against the reference (needs_ref); read as a width-like score
            continue
        rec["metrics"][nm] = metrics(np.asarray(U, np.float64), v, H, past, over, arr, cur, arr["t3_source"], t4)
    return rec


if __name__ == "__main__":
    out = []
    for world in ("stall", "no_stall"):
        for seed in range(5):
            r = one(seed, world)
            out.append(r)
            mA, mB = r["metrics"]["tail_A"], r["metrics"]["tail_B"]
            print(seed, world, "gap", round(r["t_lo"], 3), r["t_up"] and round(r["t_up"], 3), "n_tail", r["n_tail"],
                  "A bulk/tail", round(r["R_A"]["bulk"], 3), round(r["R_A"]["tail"], 3),
                  "A tail-bulk val", round(r["R_A"]["tail_minus_bulk_value"], 3),
                  "shuf", round(r["R_A"]["shuffled_tail_minus_bulk_value"], 3),
                  "B tail-bulk val", round(r["R_B"]["tail_minus_bulk_value"], 3),
                  "shuf", round(r["R_B"]["shuffled_tail_minus_bulk_value"], 3), flush=True)
            for nm in ("tail_A", "tail_B", "weighted_bca", "classifier_ratio", "distance", "range_hinge"):
                m = r["metrics"][nm]
                print("    ", nm, {k: (round(x, 3) if isinstance(x, float) else x) for k, x in m.items()}, flush=True)
    json.dump(out, open(f"{OUT}/e4_tailband.json", "w"), indent=1)

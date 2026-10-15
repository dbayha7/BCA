"""Feasibility scout for the importance-weighting experiment (exact simulation only, no learning, no BCA).

Only the current's strength c changes between training and test. Every quantity that differs between the two
distributions is therefore an integral over c of one fixed curve f(c) against a closed-form density:
  share under a tilt = integral of p(c) f(c) dc.
This script measures the curves by simulation on a grid of currents (common random numbers for every controller),
integrates them against the truncated-exponential densities, and checks the integrals by direct sampling.

Usage: python scout.py <spec name> [<spec name> ...]   (names in SPECS below; 'all' runs every one)
Outputs: /root/bca-shift/runs/shift_iw/scout/<spec>.json
"""

import json
import math
import os
import sys
import time

import numpy as np

import experiments.shift  # noqa: F401  (dependency path and version checks)
from experiments.shift import recorder as RC
from experiments.shift import shift as SH
from experiments.shift import thruster_chain as TC
from experiments.shift import truth as TR
from experiments.truth import gap as GAP

OUT = "/root/bca-shift/runs/shift_iw/scout"
SEED = 20261013
LO, HI = 0.1, 1.1
WAVE, KICK, L = (0.1, 0.4), 0.5, 100
K = np.array(RC.textbook_gain())
KNEE32 = np.float32(0.8)
HOLD80 = -0.8 * K[:, TC.CURRENT] / K[0, TC.CURRENT]
HOLD80[0] = -0.8
V1_FULL = RC.full_throttle_pattern(K)
SA = RC.stall_aware()


# ------------------------------------------------------------------ closed-form densities on [0.1, 1.1]

def M(lam):
    return (HI - LO) if lam == 0 else (math.exp(lam * HI) - math.exp(lam * LO)) / lam


def pdf(c, lam):
    return np.exp(lam * np.asarray(c, np.float64)) / M(lam)


def inv_cdf(u, lam):
    u = np.asarray(u, np.float64)
    if lam == 0:
        return LO + u * (HI - LO)
    return np.log(math.exp(lam * LO) + u * (math.exp(lam * HI) - math.exp(lam * LO))) / lam


def kish_closed(lam0, lam, n=1000):
    """n / E_train[w^2], w = p_test / p_train, both truncated exponentials: E = M(l0) M(2 l - l0) / M(l)^2."""
    return n / (M(lam0) * M(2 * lam - lam0) / M(lam) ** 2)


# ------------------------------------------------------------------ recording controllers

def hold_demand(x):
    """The textbook hold against the water alone on thruster 1: (K_e e)_1 = 0.910 c + 0.822 w1 - 0.320 w2."""
    _, ex = RC._linear(np.asarray(x, np.float64), K)
    return ex[..., 0]


def hard_water(x, trigger=0.8):
    """Hard water: the textbook hold on thruster 1 above the trigger (0.8: the textbook hold alone would stall)."""
    return hold_demand(x) > trigger


class ClipRecorder:
    """Spec A: u = -(damping K_s s + hold K_e e) + wobble n, command = clip(u, -0.8, 0.8) per thruster.
    Its 'special' command: thruster 1 exactly at -80% (the clip binds)."""

    def __init__(self, hold=1.0, damping=0.25, wobble=0.3):
        self.hold, self.damping, self.wobble = hold, damping, wobble
        self.name = f"clip recorder (hold x{hold}, damping {damping}, wobble {wobble})"

    def parts(self, x, noise=None, coin=None):
        fl, ex = RC._linear(np.asarray(x, np.float64), K)
        u = -(self.damping * fl + self.hold * ex)
        if noise is not None:
            u = u + self.wobble * noise
        cmd = np.clip(u, -0.8, 0.8)
        return cmd, hard_water(x), cmd[..., 0] == -0.8

    def clean(self, x):
        return self.parts(x)[0]


class HoldRecorder:
    """Spec B: v1's normal driving (0.8 tanh(u / 0.8), u = -(0.25 K_s s + K_e e) + 0.3 n). In hard water, with
    probability p per step (a coin), the stall-aware hold instead: thruster 1 exactly at -80%, thrusters 2-4 at the
    hold pattern scaled to that point, -0.8 K_e[:, current] / K_e[0, current] = (-0.110, -0.468, -0.325), plus
    jitter n. v1 with its full-throttle command (-1, -0.137, -0.585, -0.406) replaced by this 80% hold."""

    def __init__(self, p=0.5, jitter=0.03, damping=0.25, wobble=0.3, trigger=0.8):
        self.p, self.jitter, self.damping, self.wobble, self.trigger = p, jitter, damping, wobble, trigger
        self.name = (f"80% hold recorder (p {p}, jitter {jitter}, damping {damping}, wobble {wobble}, "
                     f"trigger {trigger})")

    def parts(self, x, noise=None, coin=None):
        fl, ex = RC._linear(np.asarray(x, np.float64), K)
        u = -(self.damping * fl + ex)
        if noise is not None:
            u = u + self.wobble * noise
        normal = 0.8 * np.tanh(u / 0.8)
        hold = np.broadcast_to(HOLD80, normal.shape).copy()
        if noise is not None:
            hold[..., 1:] = hold[..., 1:] + self.jitter * noise[..., 1:]
        hold[..., 0] = -0.8
        region = hard_water(x, self.trigger)
        press = region & (np.asarray(coin) < self.p) if coin is not None else np.zeros(region.shape, bool)
        return np.where(press[..., None], hold, normal), region, press

    def clean(self, x):
        """Normal driving only, no wobble, never the hold (about what cloning the gapped log gives)."""
        return self.parts(x)[0]


SPECS = {
    "A_hold1.0": lambda: ClipRecorder(1.0),
    "A_hold1.3": lambda: ClipRecorder(1.3),
    "B_p0.5": lambda: HoldRecorder(0.5),
    "B_p0.7": lambda: HoldRecorder(0.7),
    "B_p0.5_t0.6": lambda: HoldRecorder(0.5, trigger=0.6),
    "B_p0.5_t0.65": lambda: HoldRecorder(0.5, trigger=0.65),
}


# ------------------------------------------------------------------ simulation

def draw_starts(n, rng, c):
    """Float starts kick * N(0, I10), then the given currents c (n,), wave amplitude U[0.1, 0.4], phase U[0, 2 pi)."""
    s = KICK * rng.standard_normal((n, TC.DS))
    amp = rng.uniform(*WAVE, size=n)
    ph = rng.uniform(0.0, 2 * math.pi, size=n)
    return np.concatenate([s, np.asarray(c, np.float64)[:, None], (amp * np.cos(ph))[:, None],
                           (amp * np.sin(ph))[:, None]], 1)


def record_log(rec, starts, rng):
    """Episodes of the recorder (commands rounded to float32 as env.step does). Returns states (E, L, 13), commands
    (E, L, 4) float32, special (E, L), region (E, L)."""
    x = starts.copy()
    S, C, P, G = [], [], [], []
    for _ in range(L):
        noise = rng.standard_normal((len(x), TC.DA))
        coin = rng.uniform(size=len(x))
        cmd, region, special = rec.parts(x, noise, coin)
        a32 = cmd.astype(np.float32)
        S.append(x), C.append(a32), P.append(special), G.append(region)
        x, _ = TC.dynamics(x, a32.astype(np.float64))
    st = lambda v: np.stack(v, 1)
    return st(S), st(C), st(P), st(G)


def exact80(a32):
    return np.abs(np.asarray(a32)[..., 0]) == KNEE32


def curves(rec, gap, cgrid, per, rng_seed):
    """Per-episode statistics on a grid of currents, common random numbers across controllers and grid points:
    the recorder (stochastic), its clean form, and the stall-aware controller."""
    rng = np.random.default_rng(rng_seed)
    base = draw_starts(per, rng, np.zeros(per))
    starts = np.tile(base, (len(cgrid), 1))
    starts[:, TC.CURRENT] = np.repeat(cgrid, per)
    noise_seed = rng.integers(2**63)
    centre, rho = np.asarray(gap.centre), gap.rho
    n = len(starts)
    acc = {k: np.zeros(n) for k in ("special", "exact80", "in_gap", "region", "sa_limit_on_rec", "J_rec",
                                    "in_gap_and_special", "J_clean", "J_sa", "sa_limit_own", "J_v1_hand")}
    # recorder, stochastic
    r2 = np.random.default_rng(noise_seed)
    x = starts.copy()
    for _ in range(L):
        noise = r2.standard_normal((n, TC.DA))
        coin = r2.uniform(size=n)
        cmd, region, special = rec.parts(x, noise, coin)
        a = cmd.astype(np.float32).astype(np.float64)
        ing = GAP.in_gap(a, centre, rho)
        acc["special"] += special
        acc["exact80"] += exact80(a.astype(np.float32))
        acc["in_gap"] += ing
        acc["in_gap_and_special"] += ing & special
        acc["region"] += region
        acc["sa_limit_on_rec"] += np.abs(SA(x)[:, 0]) >= 0.8
        x, r = TC.dynamics(x, a)
        acc["J_rec"] += r
    for key, ctrl in (("J_clean", rec.clean), ("J_sa", SA), ("J_v1_hand", RC.hand_only())):
        x = starts.copy()
        for _ in range(L):
            a = np.asarray(ctrl(x)).astype(np.float32).astype(np.float64)
            if key == "J_sa":
                acc["sa_limit_own"] += np.abs(a[:, 0]) >= KNEE32
            x, r = TC.dynamics(x, a)
            acc[key] += r
    out = {}
    for k, v in acc.items():
        v = v.reshape(len(cgrid), per)
        out[k] = v if k.startswith("J") else v / L
    return out


def simpson_weights(grid):
    n = len(grid) - 1
    assert n % 2 == 0
    h = (grid[-1] - grid[0]) / n
    w = np.ones(n + 1)
    w[1:-1:2], w[2:-1:2] = 4, 2
    return w * h / 3


def integrate(raw, grid, lam):
    """raw (grid, per): per base start j, I_j = sum_i simpson_i p(c_i) f_ij; mean and SE over the base starts
    (every grid point shares the base starts, so this SE counts the correlation)."""
    wq = simpson_weights(grid) * pdf(grid, lam)
    I = wq @ raw
    return float(I.mean()), float(I.std(ddof=1) / math.sqrt(raw.shape[1]))


def bank_kish(lam0, lam, reps=2000, episodes=200, per_episode=5, seed=(SEED, 77)):
    """Kish effective size of a 1,000-row bank (200 episodes x 5 rows sharing a current), no removal."""
    rng = np.random.default_rng(list(seed))
    c = inv_cdf(rng.uniform(size=(reps, episodes)), lam0)
    w = np.repeat(pdf(c, lam) / pdf(c, lam0), per_episode, axis=1)
    k = w.sum(1) ** 2 / (w ** 2).sum(1)
    we = pdf(c, lam) / pdf(c, lam0)
    ke = we.sum(1) ** 2 / (we ** 2).sum(1)
    return dict(rows_mean=float(k.mean()), rows_p05=float(np.quantile(k, 0.05)),
                episodes_mean=float(ke.mean()), episodes_p05=float(np.quantile(ke, 0.05)))


def goodness(rec, lam, n=1500, seed=(SEED, 55)):
    """At the states where the recorder gives its special command under a current law: the stall-aware controller's
    exact discounted action value (gamma 0.98) of (a) the special command (the 80% hold without jitter), (b) v1's
    clean normal-driving command at that state, (c) the stall-aware command, (d) v1's full-throttle command.
    Differences against (b), overall and by the textbook hold demand on thruster 1."""
    rng = np.random.default_rng(list(seed))
    starts = draw_starts(4 * n, rng, inv_cdf(rng.uniform(size=4 * n), lam))
    S, C, P, G = record_log(rec, starts, rng)
    xs, cs, ps = S.reshape(-1, TC.DX), C.reshape(-1, TC.DA), P.reshape(-1)
    pick_from = np.flatnonzero(ps)
    if len(pick_from) < 10:
        return None
    pick = rng.choice(pick_from, min(n, len(pick_from)), replace=False)
    x = xs[pick]
    if isinstance(rec, HoldRecorder):
        special = np.broadcast_to(HOLD80, (len(x), TC.DA)).astype(np.float32).astype(np.float64)
    else:
        special = np.asarray(rec.clean(x)).astype(np.float32).astype(np.float64)
        special[:, 0] = -float(KNEE32)
    clean_normal = np.asarray(HoldRecorder().clean(x)).astype(np.float32).astype(np.float64)
    q = lambda a: TR.q_value(SA, x, a)
    q_special, q_normal = q(special), q(clean_normal)
    q_sa = q(np.asarray(SA(x)).astype(np.float32).astype(np.float64))
    q_v1 = q(np.broadcast_to(V1_FULL.astype(np.float32).astype(np.float64), special.shape))
    d = lambda v: dict(mean=float(v.mean()), se=float(v.std(ddof=1) / math.sqrt(len(v))),
                       share_positive=float(np.mean(v > 0)), n=int(len(v)))
    hd = hold_demand(x)
    bins = {}
    for lo, hi in ((0.0, 0.7), (0.7, 0.8), (0.8, 0.9), (0.9, 9.0)):
        m = (hd > lo) & (hd <= hi)
        if m.sum() >= 20:
            bins[f"{lo}-{hi}"] = dict(special_minus_normal=d((q_special - q_normal)[m]),
                                      special_minus_stall_aware=d((q_special - q_sa)[m]),
                                      stall_aware_at_limit=float(np.mean(np.abs(SA(x[m])[:, 0]) >= 0.8)))
    return dict(states=int(len(pick)), current_mean=float(x[:, TC.CURRENT].mean()),
                special_minus_normal=d(q_special - q_normal), stall_aware_minus_normal=d(q_sa - q_normal),
                v1_full_throttle_minus_normal=d(q_v1 - q_normal), special_minus_stall_aware=d(q_special - q_sa),
                stall_aware_at_limit=float(np.mean(np.abs(SA(x)[:, 0]) >= 0.8)), by_hold_demand=bins)


def run(name, lam0=-2.0, tilts=(("none", -2.0), ("mild", 0.0), ("strong", 2.0)), data_seeds=(0, 1, 2),
        grid_points=41, per=2000):
    t0 = time.time()
    rec = SPECS[name]()
    res = dict(spec=name, recorder=rec.name, lam_train=lam0, tilts=dict(tilts), hold80=HOLD80.tolist())
    # 1. training logs under p_train: 500 episodes x 100 steps, the gap, and what it removes
    logs = []
    for ds in data_seeds:
        rng = np.random.default_rng([SEED, ds, 1])
        starts = draw_starts(500, rng, inv_cdf(rng.uniform(size=500), lam0))
        S, C, P, G = record_log(rec, starts, rng)
        xs, a32, ps, gs = S.reshape(-1, TC.DX), C.reshape(-1, TC.DA), P.reshape(-1), G.reshape(-1)
        a = a32.astype(np.float64)
        t1 = time.time()
        record = GAP.find_gap(a, SH.gap_settings())
        tg = time.time() - t1
        e80 = exact80(a32)
        row = dict(data_seed=ds, rows=int(len(a)), gap_seconds=round(tg, 1),
                   exact80_share=float(e80.mean()), special_share=float(ps.mean()), region_share=float(gs.mean()),
                   exact80_share_in_region=float(e80[gs].mean()) if gs.any() else None,
                   gap_defined=bool(record["defined"]), gap_reason=record.get("reason"))
        if record["defined"]:
            centre, rho = np.asarray(record["centre"]), float(record["rho"])
            ing = GAP.in_gap(a, centre, rho)
            kept = ~ing
            a1k = a[kept, 0]
            row.update(
                centre=[round(v, 4) for v in record["centre"]], rho=rho, mass=float(record["mass_train"]),
                rule=record["rule"],
                exact80_removed=float(ing[e80].mean()) if e80.any() else None,
                special_removed=float(ing[ps].mean()) if ps.any() else None,
                removed_that_are_exact80=float(e80[ing].mean()),
                removed_that_are_special=float(ps[ing].mean()),
                removed_other_rows_share_of_log=float((ing & ~e80).mean()),
                removed_other_thruster1_quantiles=np.quantile(a[ing & ~e80, 0], [0.05, 0.5, 0.95]).tolist()
                if (ing & ~e80).any() else None,
                removed_current_quantiles=np.quantile(xs[ing, TC.CURRENT], [0.05, 0.5, 0.95]).tolist(),
                kept_thruster1=dict(min=float(a1k.min()), max=float(a1k.max()),
                                    q001=float(np.quantile(a1k, 0.001)), q01=float(np.quantile(a1k, 0.01)),
                                    share_le_minus_075=float(np.mean(a1k <= -0.75)),
                                    share_le_minus_079=float(np.mean(a1k <= -0.79)),
                                    exact80_left=int(e80[kept].sum())),
                region_rows_kept=float(kept[gs].mean()) if gs.any() else None,
                kept_share=float(kept.mean()))
            if ds == data_seeds[0]:
                gap0 = SH.Gap(centre=tuple(float(v) for v in centre), rho=rho, mass=float(record["mass_train"]),
                              rule=record["rule"], q=0.15, competing_ratio=float("nan"), record_sha256="")
        logs.append(row)
        print(name, "seed", ds, json.dumps({k: row[k] for k in row if k in (
            "exact80_share", "special_share", "centre", "rho", "mass", "exact80_removed", "removed_that_are_exact80",
            "kept_thruster1")}), f"{time.time() - t0:.0f}s", flush=True)
    res["training_logs"] = logs
    if not logs[0]["gap_defined"]:
        return res
    # 2. curves over the current on a grid, with the seed-0 gap
    grid = np.linspace(LO, HI, grid_points)
    cv = curves(rec, gap0, grid, per, [SEED, 3])
    res["grid"] = grid.tolist()
    res["curves"] = {k: dict(mean=v.mean(1).tolist(), se=(v.std(1, ddof=1) / math.sqrt(per)).tolist())
                     for k, v in cv.items()}
    print(name, "curves done", f"{time.time() - t0:.0f}s", flush=True)
    # 3. integrals per tilt
    Z, Zse = integrate(1 - cv["in_gap"], grid, lam0)
    res["Z"] = dict(value=Z, se=Zse, training_log_kept_share=logs[0]["kept_share"])
    tab = {}
    for tname, lam in tilts:
        rowt = {}
        for key in ("special", "exact80", "in_gap", "in_gap_and_special", "region", "sa_limit_on_rec",
                    "sa_limit_own", "J_rec", "J_clean", "J_sa", "J_v1_hand"):
            m, s = integrate(cv[key], grid, lam)
            rowt[key] = dict(mean=m, se=s)
        span = rowt["J_sa"]["mean"] - rowt["J_rec"]["mean"]
        rowt["points"] = dict(clean=100 * (rowt["J_clean"]["mean"] - rowt["J_rec"]["mean"]) / span,
                              v1_hand=100 * (rowt["J_v1_hand"]["mean"] - rowt["J_rec"]["mean"]) / span,
                              span=span)
        rowt["kish_closed_1000"] = kish_closed(lam0, lam)
        # with the gap: kept bank rows, weights Z p_test / p_train
        wq = simpson_weights(grid)
        keep = 1 - cv["in_gap"].mean(1)
        e_w = float(wq @ (pdf(grid, lam) * keep))
        e_w2 = Z * float(wq @ (pdf(grid, lam) ** 2 * keep / pdf(grid, lam0)))
        rowt["kish_gapped_bank"] = 1000 * Z * e_w ** 2 / e_w2
        rowt["kish_bank_sim"] = bank_kish(lam0, lam)
        rowt["share_c_above_0.9"] = float((math.exp(lam * HI) - math.exp(lam * 0.9)) / (lam * M(lam))) if lam \
            else 0.2
        rowt["mean_c"] = float(wq @ (grid * pdf(grid, lam)))
        tab[tname] = rowt
    res["tilts_table"] = tab
    # 4. direct-sampling check of the integrals (strong tilt): 2,000 episodes drawn from p_test
    lam = dict(tilts)["strong"]
    rng = np.random.default_rng([SEED, 4])
    starts = draw_starts(2000, rng, inv_cdf(rng.uniform(size=2000), lam))
    S, C, P, G = record_log(rec, starts, rng)
    a = C.reshape(-1, TC.DA).astype(np.float64)
    res["direct_check_strong"] = dict(
        in_gap=float(GAP.in_gap(a, np.asarray(gap0.centre), gap0.rho).mean()),
        exact80=float(exact80(C.reshape(-1, TC.DA)).mean()),
        sa_limit_on_rec=float(np.mean(np.abs(SA(S.reshape(-1, TC.DX))[:, 0]) >= 0.8)))
    # 5. is the special command good? exact action values under the stall-aware controller
    res["goodness_strong"] = goodness(rec, lam)
    res["goodness_train"] = goodness(rec, lam0)
    res["seconds"] = round(time.time() - t0, 1)
    print(name, "done", f"{res['seconds']}s", flush=True)
    return res


def main(argv):
    os.makedirs(OUT, exist_ok=True)
    names = list(SPECS) if argv == ["all"] else argv
    for name in names:
        res = run(name)
        with open(os.path.join(OUT, f"{name}.json"), "w") as f:
            json.dump(res, f, indent=1, default=float)


if __name__ == "__main__":
    main(sys.argv[1:])

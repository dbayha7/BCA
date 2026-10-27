import json, math, sys
import numpy as np

ENV = r"C:/Users/David/dev/BCA/docs/thrusterchain/site/data/environment.json"
RES = r"C:/Users/David/dev/BCA/docs/thrusterchain/site/data/results.json"
env = json.load(open(ENV))
res = json.load(open(RES))
A = np.array(env["A"]); B = np.array(env["B"])
ph = env["physics"]
gamma, k, H = ph["gamma"], ph["knee"], ph["length"]
print("knee", repr(k), "gamma", gamma, "H", H)
print("B rows 1,3 col 0:", B[1, 0], B[3, 0], " A[1,10],A[1,11]:", A[1, 10], A[1, 11])
print("A[10,10]", A[10, 10], "wave block", A[11:13, 11:13])

# ---------- finite-horizon LQ: cost = sum gamma^t (|s|^2 + 0.3|a|^2), terminal 0
Q = np.zeros((13, 13)); Q[:10, :10] = np.eye(10)
R = 0.3 * np.eye(4)
P = np.zeros((13, 13))
Ks, Ss = [None] * H, [None] * H
for t in range(H - 1, -1, -1):
    S = R + gamma * B.T @ P @ B
    K = np.linalg.solve(S, gamma * B.T @ P @ A)      # a* = -K x
    Ks[t], Ss[t] = K, S
    P = Q + gamma * A.T @ P @ A - gamma * A.T @ P @ B @ K
S0 = Ss[0]
print("S_0[0,0] curvature:", S0[0, 0], " delta-cost for 0.05:", S0[0, 0] * 0.05**2, " for 0.10:", S0[0, 0] * 0.1**2)
print("slope one unit off optimum (2*S00):", 2 * S0[0, 0])
K0 = Ks[0]
print("a0* coefficients on (c,w1,w2) at t=0:", -K0[0, 10], -K0[0, 11], -K0[0, 12])
print("||K0 row0 floats||:", np.linalg.norm(K0[0, :10]), " SD of float term at t=0 with kick 0.5:", 0.5 * np.linalg.norm(K0[0, :10]))
print("gain at t=99 (c,w1,w2):", -Ks[99][0, 10:13], " norm floats", np.linalg.norm(Ks[99][0]))
for t in (10, 50, 90, 98):
    print(f"t={t} a0* coeffs (c,w1,w2):", -Ks[t][0, 10:13])
# floats-at-rest a0* = -k when
cw = -K0[0, 10]; ww = -K0[0, 11]
print("g where floats-at-rest a0*=-k if all g is current:", -k / cw, " if g split c=0.4 rest wave:", (-k - cw * 0.4) / ww + 0.4)

# ---------- stationary float covariance under uniform recorder (Lyapunov recursion with the exogenous forcing)
rng = np.random.default_rng(0)
a = rng.uniform(-1, 1, size=2_000_000)
th = np.clip(a, -k, k) - 1.5 * np.sign(a) * np.maximum(np.abs(a) - k, 0)
var_th = th.var()
print("Var thrust(a), a~U[-1,1]:", var_th)
# per-time covariance of full 13-state under recorder with recorded start law (Monte Carlo on the linear plant is exact; do it with covariance recursion including mean)
# Monte Carlo of starts then linear propagation with random thrust: use covariance recursion directly
def start_moments(n=400000):
    s = 0.5 * rng.standard_normal((n, 10))
    c = rng.uniform(0.1, 0.7, n); amp = rng.uniform(0.1, 0.4, n); phs = rng.uniform(0, 2 * np.pi, n)
    x = np.concatenate([s, c[:, None], (amp * np.cos(phs))[:, None], (amp * np.sin(phs))[:, None]], 1)
    return x
X = start_moments()
Sig = np.cov(X.T); mu = X.mean(0)
Bth = B @ (var_th * np.eye(4)) @ B.T
sd_t = {}
for t in range(H):
    sd_t[t] = math.sqrt(Ks[t][0] @ Sig @ Ks[t][0])  # SD of a0* (all coordinates) at time t
    sd_floats = math.sqrt(Ks[t][0, :10] @ Sig[:10, :10] @ Ks[t][0, :10])
    if t in (0, 10, 25, 50, 75, 90, 99):
        print(f"t={t}: SD float-driven a0* {sd_floats:.3f}; per-coordinate float SD range {np.sqrt(np.diag(Sig)[:10]).min():.2f}-{np.sqrt(np.diag(Sig)[:10]).max():.2f}")
    Sig = A @ Sig @ A.T + Bth
    mu = A @ mu

# ---------- load probabilities under recorded start law (Monte Carlo, large n)
n = 4_000_000
c = rng.uniform(0.1, 0.7, n); amp = rng.uniform(0.1, 0.4, n); phs = rng.uniform(0, 2 * np.pi, n)
g = c + amp * np.cos(phs)
for thr in (0.4, 0.5, 0.7, 0.8, 0.9, 1.0):
    print(f"P(g>={thr}) recorded = {np.mean(g >= thr):.4f}")
print("P(0.9375<=g<=1.0625) recorded:", np.mean((g >= 0.9375) & (g <= 1.0625)))
cs = rng.uniform(0.7, 1.1, n); gs = cs + amp * np.cos(phs)
print("P(g>=0.4 | strong current):", np.mean(gs >= 0.4), " P(g>=0.9375|strong):", np.mean(gs >= 0.9375))
print("P(episode always g>=0.4)=P(c-A>=0.4):", np.mean(c - amp >= 0.4), " P(always g<0.4)=P(c+A<0.4):", np.mean(c + amp < 0.4))

# ---------- P(a0* <= -k | g) with floats SD (t=0 kick and stationary); a0* = cw*c + ww*w1 + w2w*w2 + N(0, sd^2)
w2w = -K0[0, 12]
sd0 = 0.5 * np.linalg.norm(K0[0, :10])
from math import erf
def Phi(z): return 0.5 * (1 + erf(z / math.sqrt(2)))
for sd in (sd0, 1.15):
    print(f"-- float SD {sd:.3f}")
    for gt in (0.4, 0.6, 0.8, 1.0):
        sel = np.abs(g - gt) < 0.01
        w1 = amp[sel] * np.cos(phs[sel]); w2 = amp[sel] * np.sin(phs[sel]); cc = c[sel]
        m = cw * cc + ww * w1 + w2w * w2
        p_le = np.mean([Phi((-k - mm) / sd) for mm in m[:20000]])
        p_band = np.mean([Phi((-k + 0.05 - mm) / sd) - Phi((-k - 0.05 - mm) / sd) for mm in m[:20000]])
        print(f"g={gt}: P(a0*<=-k)={p_le:.3f}  P(|a0*+k|<=0.05)={p_band:.3f}")

# ---------- per-episode hole-row counts: n_e high-load steps over 100 steps, hole ~ Binomial(n_e, h)
ne_n = 200000
c = rng.uniform(0.1, 0.7, ne_n); amp = rng.uniform(0.1, 0.4, ne_n); phs = rng.uniform(0, 2 * np.pi, ne_n)
tt = np.arange(100)
gmat = c[:, None] + amp[:, None] * np.cos(phs[:, None] + 0.27 * tt[None, :])
ne = (gmat >= 0.4).sum(1)
print("P(g>=0.4) marginal by t (first 5):", (gmat >= 0.4).mean(0)[:5], " mean n_e:", ne.mean())
print("P(n_e=0):", np.mean(ne == 0), " P(n_e=100):", np.mean(ne == 100))
for h in (0.025, 0.05, 0.10):
    print(f"h={h}: expected hole rows {h * 0.5 * 76800:.0f}; P(episode zero hole rows)={np.mean((1 - h) ** ne):.3f}")
# strata (episode, ten-step bin, side): count of high-load rows per bin; P(all removed) = h^n
hi = (gmat >= 0.4).reshape(ne_n, 10, 10).sum(2)   # per bin
lo = 10 - hi
for h in (0.05,):
    for name, arr in (("high", hi), ("low", lo)):
        nz = arr[arr > 0]
        p_all = np.mean(np.where(arr > 0, h ** arr.astype(float), 0.0))
        print(f"h={h} {name}-side strata: mean rows {arr.mean():.2f}; P(n=1)={np.mean(arr == 1):.4f}; P(n=2)={np.mean(arr == 2):.4f}; "
              f"expected zero-retention strata among 7680 = {7680 * p_all:.2f} (high side only matters)")
    print("P(bin has n_high=1):", np.mean(hi == 1), " expected number of such strata in 7680 bins:", 7680 * np.mean(hi == 1))
    print("expected removed rows per high-side stratum with n>0:", (hi * h).sum() / (hi > 0).sum())
    print("expected zero-retention high strata =", 7680 * np.mean(np.where(hi > 0, h ** hi.astype(float), 0.0)))

# ---------- float32 lattice at band edges
f32 = np.float32
for h in (0.025, 0.05, 0.10):
    for edge in (-k - h, -k + h):
        cand = [np.float32(edge)]
        cand += [np.nextafter(cand[0], f32(-2)), np.nextafter(cand[0], f32(2))]
        vals = sorted(set(float(x) for x in cand))
        ds = [(v, abs(v + k)) for v in vals]
        print(f"h={h} edge {edge:.9f}: lattice |a0+k| ->", ["%.9f" % d for v, d in ds])
print("float32 ulp near 0.8:", float(np.spacing(np.float32(0.8))))
dens = 0.5  # a0 density
ulp = float(np.spacing(np.float32(0.8)))
print("expected rows within one ulp of either edge (all rows):", 76800 * dens * 2 * 2 * ulp, " joint with g>=0.4:", 76800 * 0.5 * dens * 2 * 2 * ulp)

# ---------- least-squares all-thruster counter
B1 = B[1:10:2, :] / 0.3
u, *_ = np.linalg.lstsq(B1, -np.ones(5), rcond=None)
print("LS counter per unit g:", u)
print("thruster 1 at -k on v1,v2:", B[1, 0] * -k, " water push per unit g on each speed:", A[1, 10])

# ---------- development bank checks
states = res["development_banks"]["states"]
rows = []
for si, s in enumerate(states):
    x = np.array(s["state"]); gg = x[10] + x[11]; t = s["time"]
    for ci in range(9):
        a0 = s["actions"][ci][0]
        rows.append((si, t, gg, ci, a0, s["q1"][ci], s["v"][ci], s["u"][ci]))
rows = np.array(rows)
inband = np.abs(rows[:, 4] + k) <= 0.05
print("candidates in band (|a0+k|<=0.05):", inband.sum(), " of", len(rows), " random-only:", (inband & (rows[:, 3] >= 2)).sum())
ib = rows[inband]
hi_ib = ib[ib[:, 2] >= 0.4]
err = hi_ib[:, 5] - hi_ib[:, 6]
print("in-band at g>=0.4: n", len(hi_ib), " signed error q1-v min/max:", err.min(), err.max(), " negatives:", (err < 0).sum())
lo_ib = ib[ib[:, 2] < 0.4]
err_lo = lo_ib[:, 5] - lo_ib[:, 6]
print("in-band at g<0.4: n", len(lo_ib), " err min/max", err_lo.min(), err_lo.max(), " negatives at t:", lo_ib[err_lo < 0][:, 1])
for si in (49, 73):
    s = states[si]; x = np.array(s["state"]); gg = x[10] + x[11]
    for ci in range(9):
        if abs(s["actions"][ci][0] + k) <= 0.05:
            print(f" state {si} group {s['group_id']} t={s['time']} g={gg:.3f} a0={s['actions'][ci][0]:.4f} q1-v={s['q1'][ci]-s['v'][ci]:.1f}")
# actor a0 stats
act = rows[rows[:, 3] == 0]
a0 = act[:, 4]; gact = act[:, 2]; tact = act[:, 1]
print("actor a0 min/mean/max:", a0.min(), a0.mean(), a0.max(), " within 0.05 of -k:", np.sum(np.abs(a0 + k) <= 0.05), " <=-0.7:", np.sum(a0 <= -0.7))
print("corr(a0,g)", np.corrcoef(a0, gact)[0, 1], " corr(a0,t)", np.corrcoef(a0, tact)[0, 1])
sel = gact >= 0.7
print("g>=0.7 states:", sel.sum(), " mean a0", a0[sel].mean(), " min", a0[sel].min(), " nearest actor a0 to -k:", np.min(np.abs(a0 + k)))
# best-v candidate a0 by g bin
best = []
for si, s in enumerate(states):
    x = np.array(s["state"]); gg = x[10] + x[11]
    bi = int(np.argmax(s["v"]))
    best.append((gg, s["actions"][bi][0], max(s["v"]) - min(s["v"])))
best = np.array(best)
for lo_, hi_ in ((-9, 0.4), (0.4, 0.7), (0.7, 9)):
    m = (best[:, 0] >= lo_) & (best[:, 0] < hi_)
    print(f"g in [{lo_},{hi_}): n={m.sum()} mean best a0={best[m, 1].mean():.3f}")
print("v range across 9 candidates per state: median", np.median(best[:, 2]), " max", best[:, 2].max(), " min", best[:, 2].min())
# g distribution of the bank
gb = np.array([np.array(s["state"])[10] + np.array(s["state"])[11] for s in states])
print("bank g>=0.4 share:", np.mean(gb >= 0.4), " g>=0.7 share:", np.mean(gb >= 0.7))
# thrust values
def thrust(a): return np.clip(a, -k, k) - 1.5 * np.sign(a) * np.maximum(np.abs(a) - k, 0)
print("thrust(-0.75),(-0.8),(-0.85):", thrust(-0.75), thrust(-0.8), thrust(-0.85), " midpoint interp:", (thrust(-0.75) + thrust(-0.85)) / 2)
# counter_current demo forcing
traj = env["trajectories"][1]
print("demo", traj["name"], "start state c,w:", traj["states"][0][10:13], " return", traj["discounted_return"])

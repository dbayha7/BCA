"""Closed-form / quadrature arithmetic only. Reads environment.json matrices and data_receipt std.
No simulator code, no dataset arrays."""
import json, math, sys
import numpy as np

env = json.load(open(sys.argv[1]))
rec = json.load(open(sys.argv[2]))
A = np.array(env["A"]); B = np.array(env["B"])
std = np.array(rec["normalization"]["std"][:13]); mean = np.array(rec["normalization"]["mean"][:13])
k = 0.800000011920929; h = 0.05

rng = np.random.default_rng(0)  # quadrature sampler over the documented start law formula only
N = 400000
c = rng.uniform(0.1, 0.7, N); amp = rng.uniform(0.1, 0.4, N); ph = rng.uniform(0, 2*math.pi, N)
t = np.arange(100)
g = c[:, None] + amp[:, None] * np.cos(ph[:, None] + 0.27 * t[None, :])
hi = g >= 0.4
print("P(g>=0.4) per step", hi.mean())
nh = hi.sum(1)
print("P(episode never high)", (nh == 0).mean(), " P(always high)", (nh == 100).mean())
print("E[0.95^Nh] = P(no hole row in episode)", (0.95 ** nh).mean())
print("mean hole rows/episode", 0.05 * nh.mean(), " x768 =", 0.05 * nh.mean() * 768)
# hole rows per episode distribution quantiles (expected)
# per ten-step bin x load side: episodes with >=1 row on that side
for b in range(10):
    s = hi[:, 10*b:10*b+10]
    pass
side_any_hi = np.array([(hi[:, 10*b:10*b+10].any(1)).mean() for b in range(10)])
side_any_lo = np.array([((~hi[:, 10*b:10*b+10]).any(1)).mean() for b in range(10)])
print("frac episodes with >=1 high row per time bin", side_any_hi.round(3))
print("frac episodes with >=1 low row per time bin", side_any_lo.round(3))
# strong current
cs = rng.uniform(0.7, 1.1, N)
gs = cs[:, None] + amp[:, None] * np.cos(ph[:, None] + 0.27 * t[None, :])
print("strong current P(g>=0.4)", (gs >= 0.4).mean(), "min g", gs.min())

# neighbour band counts Binomial(64, 0.05)
from math import comb
p = 0.05
pmf = [comb(64, j) * p**j * (1-p)**(64-j) for j in range(65)]
cdf = np.cumsum(pmf)
print("Bin(64,.05): mean", 64*p, "P0", pmf[0], "P(>=3)", 1-cdf[2], "P(>=5)", 1-cdf[4], "P(>=8)", 1-cdf[7])
for n in (32, 128):
    print(f"Bin({n},.05) mean", n*p, "P0", (1-p)**n)

# float32 edge check: can float64(a0_f32)+k equal +-h exactly?
a = np.arange(-0.86, -0.74, 1e-9)  # dense float64 grid near edges
a32 = a.astype(np.float32).astype(np.float64)
d = np.abs(a32 + k)
print("any exact |a0+k|==h among float32 values near edges:", bool(np.any(d == h)))
lo_edge = -k - h; hi_edge = -k + h
f = np.float32(lo_edge); print("edges", repr(lo_edge), repr(hi_edge), "nearest f32", repr(float(f)), repr(float(np.float32(hi_edge))))
# requested float64 rows flipping when cast: width of flip interval ~ half ulp f32 at 0.85/0.75
ulp85 = float(np.spacing(np.float32(0.85))); ulp75 = float(np.spacing(np.float32(0.75)))
print("f32 ulp near edges", ulp85, ulp75, "expected flipping rows in 38400 hi rows:", 38400*0.5*(ulp85/2+ulp75/2))
# also does stored k as float64(float32(0.8)) matter vs 0.8: difference
print("k-0.8", k-0.8)

# ---- linear covariance of state at time t under uniform behaviour (closed form)
A10 = A[:10, :10]; B1 = B[:10, :]; v = A[:10, 10] / 0.3 if A[1, 10] != 0 else None
v = A[:10, 10]  # already gain*v
w1col = A[:10, 11]
var_th = 0.5 * (2 * 0.8**3 / 3 + 2 * 0.086)
print("Var thrust (uniform a)", var_th)
Ec2 = (0.7**3 - 0.1**3) / (3*0.6); Ec = 0.4; varc = Ec2 - Ec**2
EA2 = (0.4**3 - 0.1**3) / (3*0.3)
print("var c", varc, "E A^2", EA2)

def cov_at(T):
    # state x_T (13): s part linear in s0, thrusts, c, wave(A cos, A sin)
    # s_T = A10^T s0 + sum_j A10^(T-1-j) [B1 u_j + v c + w1col * A cos(ph + .27 j)]
    P = np.eye(10)
    pw = [np.linalg.matrix_power(A10, m) for m in range(T+1)]
    S = 0.25 * pw[T] @ pw[T].T
    for j in range(T):
        M = pw[T-1-j] @ B1
        S = S + var_th * M @ M.T
    gc = sum(pw[T-1-j] @ v for j in range(T)) if T > 0 else np.zeros(10)
    # wave: A*(cos(ph+.27j)) -> represent as A*(cos ph cos .27j - sin ph sin .27j)
    gcos = sum(pw[T-1-j] @ w1col * math.cos(0.27*j) for j in range(T)) if T > 0 else np.zeros(10)
    gsin = sum(-pw[T-1-j] @ w1col * math.sin(0.27*j) for j in range(T)) if T > 0 else np.zeros(10)
    # w1_T = A cos(ph+.27T), w2_T = A sin(ph+.27T); (A cos ph, A sin ph) has cov EA2/2 I
    C = np.zeros((13, 13))
    C[:10, :10] = S + varc * np.outer(gc, gc) + EA2/2 * (np.outer(gcos, gcos) + np.outer(gsin, gsin))
    C[10, 10] = varc; C[:10, 10] = C[10, :10] = varc * gc
    R = np.array([[math.cos(0.27*T), -math.sin(0.27*T)], [math.sin(0.27*T), math.cos(0.27*T)]])
    # s depends on (Acosph, Asinph) via [gcos, gsin]; w_T = R (Acosph, Asinph)
    G = np.stack([gcos, gsin], 1)  # 10x2
    C[:10, 11:13] = EA2/2 * G @ R.T; C[11:13, :10] = C[:10, 11:13].T
    C[11:13, 11:13] = EA2/2 * np.eye(2)
    return C

zs = 0
for T in (5, 25, 50, 75, 95):
    C = cov_at(T)
    Cz = C / np.outer(std, std)
    ev = np.linalg.eigvalsh(2 * Cz)
    ev = np.clip(ev, 0, None)
    # P(sum ev_i chi2_1 <= 13*0.25) by Gaussian sampling (not a simulator)
    zz = rng.standard_normal((400000, 13))
    q = (zz**2 * ev).sum(1)
    for r in (0.4, 0.5, 0.6):
        pr = (q <= 13 * r * r).mean()
        print(f"T={T} r={r}: P(pair within RMS r) ~ {pr:.2e}; x768 episodes ~ {pr*768:.3f} per-time-slice; x7680 bin rows ~ {pr*7680:.2f}")
    print(f"T={T} median pair RMS", math.sqrt(np.median(q)/13), "per-coord std ratio within-time/global", np.sqrt(np.diag(Cz)).round(2))

# ---- finite-horizon LQ (linear thrust, no stall, no box) optimal a0 at s = 0 and s = data mean
Aa = A.copy(); Bb = B.copy()
Q = np.zeros((13, 13)); Q[:10, :10] = np.eye(10); R = 0.3 * np.eye(4); gam = 0.98
P = np.zeros((13, 13)); K = [None]*100
for tt in range(99, -1, -1):
    H = R + gam * Bb.T @ P @ Bb
    Kt = np.linalg.solve(H, gam * Bb.T @ P @ Aa)
    K[tt] = Kt
    Acl = Aa - Bb @ Kt
    P = Q + Kt.T @ R @ Kt + gam * Acl.T @ P @ Acl
for tt in (0, 25, 50, 75, 90):
    for cc in (0.4, 0.55, 0.7, 0.9, 1.1):
        for w in ((0.0, 0.0), (0.3, 0.0), (-0.3, 0.0)):
            x0 = np.zeros(13); x0[10] = cc; x0[11:13] = w
            u0 = -K[tt] @ x0
            xm = mean.copy(); xm[10] = cc; xm[11:13] = w
            um = -K[tt] @ xm
            if w == (0.0, 0.0) or tt == 25:
                print(f"t={tt} c={cc} w={w} g={cc+w[0]:.2f}: a*(s=0)={u0.round(3)}  a*(s=mean)={um.round(3)}")
print("feedback gain row0 on s at t=25", K[25][0, :10].round(3), "on c,w1,w2", K[25][0, 10:].round(3))

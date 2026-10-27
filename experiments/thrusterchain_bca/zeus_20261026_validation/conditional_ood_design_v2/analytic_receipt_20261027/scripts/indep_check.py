"""Independent re-derivation (closed-form linear-Gaussian approximation + sampling of that approximation only).
Reads environment.json A,B and data_receipt normalization std. No simulator/experiment code, no dataset arrays."""
import json, math, sys
import numpy as np
from math import comb

env = json.load(open(sys.argv[1])); rec = json.load(open(sys.argv[2]))
A = np.array(env["A"]); B = np.array(env["B"])
std = np.array(rec["normalization"]["std"][:13])
k = 0.800000011920929
rng = np.random.default_rng(12345)

# ---- g distribution (exact law, sampled)
N = 2_000_000
c = rng.uniform(0.1, 0.7, N); amp = rng.uniform(0.1, 0.4, N); ph = rng.uniform(0, 2*math.pi, N)
g = c + amp*np.cos(ph)
H = g >= 0.4
print("P(g>=.4)", H.mean(), "P(g>=.6)", (g>=.6).mean(), "P(g>=.7)", (g>=.7).mean())
gh = g[H]
print("among high: P(g<.45)", (gh<.45).mean(), "P(g<.5)", (gh<.5).mean(), "P(g<.6)", (gh<.6).mean())
cs = rng.uniform(0.7, 1.1, N); gs = cs + amp*np.cos(ph)
print("strong: min g", gs.min(), "P(g>=.4)", (gs>=.4).mean())

# ---- per-episode
t = np.arange(100)
M = 200000
c2 = c[:M]; a2 = amp[:M]; p2 = ph[:M]
G = c2[:, None] + a2[:, None]*np.cos(p2[:, None] + 0.27*t[None, :])
nh = (G >= 0.4).sum(1)
print("P never high", (nh==0).mean(), "P always high", (nh==100).mean(), "P no hole row", (0.95**nh).mean(),
      "E hole rows/ep", 0.05*nh.mean(), "x768", 0.05*nh.mean()*768)

# ---- binomial
p = .05
pm = [comb(64, j)*p**j*(1-p)**(64-j) for j in range(65)]
print("Bin64: P0", pm[0], "P>=5", 1-sum(pm[:5]))

# ---- thrust push values
def thrust(a):
    return np.clip(a, -k, k) - 1.5*np.sign(a)*np.maximum(np.abs(a)-k, 0)
for a in (-1, -0.95, -0.85, -0.8333333, -0.8, -0.75, -0.65):
    print("thrust", a, thrust(np.float64(a)))
var_th = 2*(0.8**3/3) / 2 + 0.086  # E th^2 = int_0^1 th^2
print("var_th", var_th)

# ---- linear Gaussian state at time T given (c, amp, ph)
A10 = A[:10, :10]; B10 = B[:10]; vcol = A[:10, 10]; wcol = A[:10, 11]
pw = [np.eye(10)]
for m in range(110): pw.append(A10 @ pw[-1])
Sig = {}
S = 0.25*np.eye(10)
for T in range(101):
    Sig[T] = S.copy()
    S = A10 @ S @ A10.T + var_th * B10 @ B10.T
gcur = {0: np.zeros(10)}; gc = np.zeros(10)
for T in range(1, 101):
    gc = A10 @ gc + vcol; gcur[T] = gc.copy()
# wave contribution: sum_j A^{T-1-j} wcol * amp cos(ph + .27 j)
def wave_mean(T, amp, ph):
    m = np.zeros((len(amp), 10))
    for j in range(T):
        m += (amp*np.cos(ph + 0.27*j))[:, None] * (pw[T-1-j] @ wcol)[None, :]
    return m
Lch = {T: np.linalg.cholesky(Sig[T] + 1e-12*np.eye(10)) for T in range(101)}
def sample(T, n, lo=0.1, hi=0.7):
    cc = rng.uniform(lo, hi, n); aa = rng.uniform(0.1, 0.4, n); pp = rng.uniform(0, 2*math.pi, n)
    s = cc[:, None]*gcur[T][None, :] + wave_mean(T, aa, pp) + rng.standard_normal((n, 10)) @ Lch[T].T
    return np.concatenate([s, cc[:, None], (aa*np.cos(pp+.27*T))[:, None], (aa*np.sin(pp+.27*T))[:, None]], 1)

# probe at time T (bin b), neighbor at uniform time within the same ten-step bin, same load side as probe
for b in (0, 2, 5, 9):
    n = 400000
    Tp = rng.integers(10*b, 10*b+10)
    xp = sample(int(Tp), n)
    # neighbors: mixture over times in bin
    xs = np.concatenate([sample(T, n//10) for T in range(10*b, 10*b+10)])
    rng.shuffle(xs)
    xs = xs[:n]
    gp = xp[:, 10]+xp[:, 11]; gn = xs[:, 10]+xs[:, 11]
    hp = gp >= .4
    same = (gp >= .4) == (gn >= .4)
    d = np.sqrt(np.mean(((xp-xs)/std)**2, 1))
    out = []
    for r in (0.4, 0.5, 0.6):
        # high-load probes, high-load neighbors; expected count = P(within r | both high) * rows on high side (768*10*0.5)
        ph_ = np.mean(d[hp & same] <= r)
        out.append((r, ph_, ph_*3840))
    dq = d[hp & same]
    print(f"bin {b} probeT={Tp}: (r,P,expected rows of 3840)", [(r, f"{x:.2e}", round(y, 2)) for r, x, y in out],
          "median", round(float(np.median(dq)), 3), "q32", round(float(np.quantile(dq, 32/3840)), 3), "q64", round(float(np.quantile(dq, 64/3840)), 3))

# ---- unconstrained finite-horizon LQ, linear thrust
Q = np.zeros((13, 13)); Q[:10, :10] = np.eye(10); R = 0.3*np.eye(4); gam = 0.98
P = np.zeros((13, 13)); K = [None]*100
for tt in range(99, -1, -1):
    Hm = R + gam*B.T@P@B
    Kt = np.linalg.solve(Hm, gam*B.T@P@A); K[tt] = Kt
    P = Q + gam*A.T@P@A - gam*A.T@P@B@Kt
for lab, lo, hi in (("rec", .1, .7), ("strong", .7, 1.1)):
    for T in (5, 25, 50, 75, 95):
        x = sample(T, 200000, lo, hi)
        gg = x[:, 10]+x[:, 11]; Hh = gg >= .4
        a0 = -(x @ K[T].T)[:, 0]
        print(f"{lab} T={T} hi={Hh.mean():.3f} high: P(a0*<=-.85)={np.mean(a0[Hh]<=-.85):.3f} P(a0*<=-k)={np.mean(a0[Hh]<=-k):.3f} "
              f"P(a0*>-.75)={np.mean(a0[Hh]>-.75):.3f} | low: P(a0*<=-k)={np.mean(a0[~Hh]<=-k):.3f} P(a0*<=-.75)={np.mean(a0[~Hh]<=-.75):.3f}")

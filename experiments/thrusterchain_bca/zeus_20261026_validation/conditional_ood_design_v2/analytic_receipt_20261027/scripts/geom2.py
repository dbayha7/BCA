"""Closed-form linear-Gaussian approximation only (no simulator code, no dataset arrays)."""
import json, math, sys
import numpy as np
env = json.load(open(sys.argv[1])); rec = json.load(open(sys.argv[2]))
A = np.array(env["A"]); B = np.array(env["B"])
std = np.array(rec["normalization"]["std"][:13])
k = 0.800000011920929
A10 = A[:10, :10]; B1 = B[:10, :]; vcol = A[:10, 10]; w1col = A[:10, 11]
var_th = 0.2566666666666667
pw = [np.linalg.matrix_power(A10, m) for m in range(101)]
# LQ gains (linear thrust, no box)
Q = np.zeros((13, 13)); Q[:10, :10] = np.eye(10); R = 0.3*np.eye(4); gam = 0.98
P = np.zeros((13, 13)); K = [None]*100
for tt in range(99, -1, -1):
    H = R + gam*B.T@P@B; Kt = np.linalg.solve(H, gam*B.T@P@A); K[tt] = Kt
    Acl = A - B@Kt; P = Q + Kt.T@R@Kt + gam*Acl.T@P@Acl
rng = np.random.default_rng(1)

def sample_states(T, n, clo, chi):
    c = rng.uniform(clo, chi, n); amp = rng.uniform(0.1, 0.4, n); ph = rng.uniform(0, 2*math.pi, n)
    gc = sum(pw[T-1-j]@vcol for j in range(T)) if T else np.zeros(10)
    mean = c[:, None]*gc[None, :]
    for j in range(T):
        mean = mean + (amp*np.cos(ph+0.27*j))[:, None]*(pw[T-1-j]@w1col)[None, :]
    S = 0.25*pw[T]@pw[T].T + sum(var_th*(pw[T-1-j]@B1)@(pw[T-1-j]@B1).T for j in range(T))
    L = np.linalg.cholesky(S + 1e-12*np.eye(10))
    s = mean + rng.standard_normal((n, 10))@L.T
    x = np.concatenate([s, c[:, None], (amp*np.cos(ph+0.27*T))[:, None], (amp*np.sin(ph+0.27*T))[:, None]], 1)
    return x

for label, lo, hi in (("recorded", 0.1, 0.7), ("strong", 0.7, 1.1)):
    for T in (5, 25, 50, 75, 95):
        x = sample_states(T, 200000, lo, hi)
        g = x[:, 10] + x[:, 11]; H = g >= 0.4
        a0 = -(x@K[T].T)[:, 0]
        aH = a0[H]
        print(f"{label} T={T}: high-load frac {H.mean():.3f}; unconstrained LQ a0* among high-load: "
              f"P(a0*<=-0.85)={np.mean(aH<=-0.85):.3f} P(in band)={np.mean(np.abs(aH+k)<=0.05):.3f} "
              f"P(a0*>-0.75)={np.mean(aH>-0.75):.3f}; low-load P(a0*<=-0.75)={np.mean(a0[~H]<=-0.75):.3f}; median high {np.median(aH):.2f}")
        if label == "recorded":
            mu = x.mean(0)
            print("   approx mean state p1,v1,c:", mu[[0, 1, 10]].round(3))

# neighbour radius quantiles within one time slice (pairs of independent rows, same T)
for T in (25, 75):
    x1 = sample_states(T, 300000, 0.1, 0.7); x2 = sample_states(T, 300000, 0.1, 0.7)
    sameside = (x1[:, 10]+x1[:, 11] >= 0.4) == (x2[:, 10]+x2[:, 11] >= 0.4)
    d = np.sqrt(np.mean(((x1-x2)/std)**2, 1))[sameside]
    for r in (0.5, 0.6):
        print(f"T={T} same-side P(RMS<= {r}) = {np.mean(d<=r):.2e}")
    for frac_label, q in (("32/3840", 32/3840), ("64/3840", 64/3840)):
        print(f"T={T} same-side RMS quantile {frac_label}: {np.quantile(d, q):.3f}")

# Outcome-free settings for the step-4 synthesis: multiset statistics, start-policy values, tilt scales,
# cone anisotropy, FP well-posedness (Hessian definiteness). No actor runs, no J of any arm.
import math, sys
import numpy as np
from scipy import linalg, stats
sys.path.insert(0, ".")
from lq_common import A, B, Qc, Rc, noise, init, g, sb, L, D, Gr, qq, J, Sigma, Ks, kgrad, cos

def multiset(n, s, clip=2.5):
    z = stats.norm.ppf((np.arange(1, n + 1) - 0.5) / n)
    z = np.clip(z, -clip, clip)
    m = np.exp(s * z)
    m = m / m.mean()
    cv = m.std() / m.mean()
    kish = m.sum() ** 2 / (n * (m ** 2).sum())
    p90, p10 = np.percentile(m, 90), np.percentile(m, 10)
    return dict(cv=cv, lo=m.min(), hi=m.max(), kish=kish, r9010=p90 / p10)

for s in (0.47, 0.83, 1.0):
    print("multiset n=10000 s=%.2f" % s, {k: round(float(v), 3) for k, v in multiset(10000, s).items()})
    print("multiset n=256   s=%.2f" % s, {k: round(float(v), 3) for k, v in multiset(256, s).items()})

print("J* %.4f  J(0) %.4f" % (J(Ks), J(0 * Ks)))
K0 = 0.5 * Ks + 0.5 * D
print("S-common start K0 = 0.5K* + 0.5D: J %.4f, gap %.4f" % (J(K0), J(Ks) - J(K0)))
for gain in (0.1338, 0.5):
    print("gain %.4f K*: J %.4f" % (gain, J(gain * Ks)))
M = A - B @ K0
print("sqrt(g) rho(A - B K0) = %.3f" % (math.sqrt(g) * np.max(np.abs(np.linalg.eigvals(M)))))
H = qq(K0, Gr)
print("H_aa eig at K0:", np.round(np.linalg.eigvalsh(H[3:, 3:]), 3))

# tilt scale c for alignment a*: cos = (1 - x)/sqrt((1-x)^2 + x^2), x = c/sqrt(2)
for a in (0.9, 0.5, 0.0, -0.5):
    f = lambda x: (1 - x) / math.sqrt((1 - x) ** 2 + x ** 2) - a
    from scipy.optimize import brentq
    x = brentq(f, 1e-9, 50)
    print("tilt a*=%.1f: c = %.3f" % (a, x * math.sqrt(2)))

# state second moment under K0 rollouts with behaviour noise (time-averaged over the episode)
S0 = Sigma(K0)
print("tr Sigma(K0 rollouts) %.3f; Sigma(K*) %.3f; Sigma(0) %.3f" % (np.trace(S0), np.trace(Sigma(Ks)), np.trace(Sigma(0 * Ks))))

# cone anisotropy: Monte Carlo over Gaussian states N(0, S0) (states only; no outcomes)
rng = np.random.default_rng(0)
X = rng.multivariate_normal(np.zeros(3), S0, size=400000)
v = np.array([1.0, -1.0, 1.0]) / math.sqrt(3)
xh = X / np.linalg.norm(X, axis=1, keepdims=True)
proj = xh @ v
sig = lambda c0: 1 / (1 + np.exp(-10 * (proj - c0)))
lo, hi = -1.0, 1.0
for _ in range(60):
    mid = (lo + hi) / 2
    lo, hi = (mid, hi) if sig(mid).mean() > 0.25 else (lo, mid)
c0 = (lo + hi) / 2
w = sig(c0)
Sw = (X * w[:, None]).T @ X / len(X)
Sx = X.T @ X / len(X)
dK = K0 - Ks  # good data: K_pi - K_b direction; poor: K0 - 0
for name, Kb in (("expert", Ks), ("medium", 0.1338 * Ks), ("poor", 0 * Ks)):
    d = K0 - Kb
    print("cone mean 0.25 (c0=%.3f): cos((K0-Kb) S, (K0-Kb) S_cone) %s = %.4f" % (c0, name, cos(d @ Sx, d @ Sw)))
print("corr(cone weight, ||s||^2) = %.3f" % np.corrcoef(w, (X ** 2).sum(1))[0, 1])

# FP Hessian definiteness for q1_optimistic / cone at K0 needs lambda_0 = alpha / mean|Q1(s, pi0(s))|
_, h0, _ = None, None, None
Pv = linalg.solve_discrete_lyapunov(math.sqrt(g) * M.T, np.vstack([np.eye(3), -K0]).T @ Gr @ np.vstack([np.eye(3), -K0]))
h = g * np.trace(Pv @ noise) / (1 - g)
q_on = np.einsum("ni,ij,nj->n", X, Pv, X) + h
lam0 = 2.5 / np.mean(np.abs(q_on))
print("lambda_0 (clean, K0 states) = %.3f; mean|Q| = %.3f" % (lam0, np.mean(np.abs(q_on))))
for kap in (0.5, 1.0, 2.0):
    Paa = H[3:, 3:] + kap * np.eye(2)
    Hq = -lam0 * np.kron(Sx, Paa) + 0.5 * np.kron(Sx, np.eye(2))
    print("q1_opt kappa=%.1f: H1_aa eig %s, FP Hessian min eig (m=1) %.4f" % (
        kap, np.round(np.linalg.eigvalsh(Paa), 3), np.linalg.eigvalsh(Hq).min()))

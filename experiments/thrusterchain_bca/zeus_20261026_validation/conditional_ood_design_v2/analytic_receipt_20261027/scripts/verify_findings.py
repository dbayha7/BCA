"""Analytic checks of the review findings. Reads only environment.json and results.json (site data)."""
import json, sys
import numpy as np

ENV = r"C:/Users/David/dev/BCA/docs/thrusterchain/site/data/environment.json"
RES = r"C:/Users/David/dev/BCA/docs/thrusterchain/site/data/results.json"
env = json.load(open(ENV))
A = np.array(env["A"], dtype=np.float64)
B = np.array(env["B"], dtype=np.float64)
k = env["physics"]["knee"]; slope = env["physics"]["stall_slope"]; cost = env["physics"]["throttle_cost"]
gamma = 0.98; H = 100

def thrust(a):
    a = np.asarray(a, dtype=np.float64)
    return np.clip(a, -k, k) - slope * np.sign(a) * np.maximum(np.abs(a) - k, 0.0)

print("=== thrust / cost table ===")
for a0 in [-1.0, -0.9, -0.85, -0.8, -0.75, -0.7, -0.6, 0.0, 0.5]:
    print(f"a0={a0:+.2f} thrust={thrust(a0):+.4f} cost={cost*a0*a0:.4f}")

rng = np.random.default_rng(12345)
def draw_start(law, n):
    c = rng.uniform(*law["current"], size=n)
    Aamp = rng.uniform(*law["wave"], size=n)
    ph = rng.uniform(0, 2*np.pi, size=n)
    osc = law["kick"] * rng.standard_normal((n, 10))
    return np.column_stack([osc, c, Aamp*np.cos(ph), Aamp*np.sin(ph)])

rec = env["reset"]["start_laws"]["recorded"]; strong = env["reset"]["start_laws"]["strong_current"]

print("\n=== load probabilities (Monte Carlo, 2e6 draws) ===")
n = 2_000_000
c = rng.uniform(*rec["current"], n); Aamp = rng.uniform(*rec["wave"], n); ph = rng.uniform(0, 2*np.pi, n)
g = c + Aamp*np.cos(ph)
print("P(g>=0.4 | recorded, t=0) =", (g >= 0.4).mean())
print("P(c+A<0.4 | recorded) (no high-load row ever) =", (c + Aamp < 0.4).mean(), " closed form 0.02/0.18 =", 0.02/0.18)
print("P(c-A>=0.4 | recorded) (no low-load row ever) =", (c - Aamp >= 0.4).mean())
cs = rng.uniform(*strong["current"], n); As = rng.uniform(*strong["wave"], n); phs = rng.uniform(0, 2*np.pi, n)
gs = cs + As*np.cos(phs)
print("P(g>=0.4 | strong) =", (gs >= 0.4).mean(), " low-load share =", (gs < 0.4).mean())

# episodes with >=1 hole row: n_high per episode, P = 1-(1-h)^n_high
m = 200_000
c = rng.uniform(*rec["current"], m); Aamp = rng.uniform(*rec["wave"], m); ph = rng.uniform(0, 2*np.pi, m)
t = np.arange(H)
gt = c[:, None] + Aamp[:, None]*np.cos(ph[:, None] + 0.27*t[None, :])
nhigh = (gt >= 0.4).sum(1)
print("mean fraction high-load steps per episode =", nhigh.mean()/H)
for h in [0.025, 0.05, 0.10]:
    print(f"h={h}: expected hole rows train={76800*0.5*h:.0f}, per bin={7680*0.5*h:.0f}, calib={12800*0.5*h:.0f}; "
          f"share of episodes with >=1 hole row = {(1-(1-h)**nhigh).mean():.3f}")
# per-bin emptiness of (episode,bin,high) strata
bins = (gt >= 0.4).reshape(m, 10, 10).sum(2)
print("share of (episode,bin) cells with zero high-load rows =", (bins == 0).mean(),
      "; share of episodes with no high-load rows =", (nhigh == 0).mean())

print("\n=== covariance propagation vs receipt std ===")
Sigma0 = np.zeros((13, 13)); Sigma0[:10, :10] = np.eye(10)*rec["kick"]**2
Sigma0[10, 10] = (rec["current"][1]-rec["current"][0])**2/12
EA2 = (rec["wave"][1]**3 - rec["wave"][0]**3)/(3*(rec["wave"][1]-rec["wave"][0]))
Sigma0[11, 11] = Sigma0[12, 12] = EA2/2
# Var(thrust) for a~U[-1,1]
aa = np.linspace(-1, 1, 2_000_001); vt = np.mean(thrust(aa)**2)
print("Var(thrust) uniform a =", vt)
mu0 = np.zeros(13); mu0[10] = np.mean(rec["current"])
Sig = [Sigma0]; mu = [mu0]
for _ in range(H-1):
    Sig.append(A @ Sig[-1] @ A.T + B @ (vt*np.eye(4)) @ B.T); mu.append(A @ mu[-1])
Sig = np.array(Sig); mu = np.array(mu)
mubar = mu.mean(0)
pooled_var = Sig.diagonal(axis1=1, axis2=2).mean(0) + ((mu - mubar)**2).mean(0)
receipt_std = np.array([0.8719595074653625,0.7674936652183533,0.7971653938293457,0.8453173041343689,0.702329695224762,0.6917358040809631,0.5929064750671387,0.60101318359375,0.38637569546699524,0.3966222405433655,0.17674623429775238,0.1886003315448761,0.18845102190971375])
names = env["state_order"]
for i in range(13):
    print(f"{names[i]:8s} model std={np.sqrt(pooled_var[i]):.4f} receipt={receipt_std[i]:.4f} ratio={np.sqrt(pooled_var[i])/receipt_std[i]:.3f}")

print("\n=== P(two same-time states within standardized RMS rho), Gaussian per-time approx ===")
D = np.diag(1/receipt_std)
def p_within(t, rho, ns=400_000):
    C = 2 * D @ Sig[t] @ D
    L = np.linalg.cholesky(C + 1e-15*np.eye(13))
    z = rng.standard_normal((ns, 13)) @ L.T
    d2 = (z**2).sum(1)/13
    return (d2 <= rho**2).mean(), np.sqrt(d2)
for t in [0, 25, 50, 75]:
    p, d = p_within(t, 0.5)
    # radius giving expected 8 of 767 other episodes
    r8 = np.quantile(d, 8/767); r8half = np.quantile(d, 16/767)
    print(f"t={t:2d}: P(RMS<=0.5)={p:.2e}; expected other-episode same-time points in 767 = {767*p:.3f}; "
          f"radius for E[8 of 767]={r8:.3f}; after halving pool (16/767 quantile)={r8half:.3f}; median pair dist={np.median(d):.3f}")

print("\n=== 4-D action box ===")
side = 0.1
p = side**4/16
print("P(row in box)=", p, " expected in 76,800 =", 76800*p, " in bin-and-load cell (3,840) =", 3840*p,
      " side for 8 rows in cell =", (8*16/3840)**0.25, " expected inside k=64 neighbourhood =", 64*p)
print("a0-marginal: P(|a0-target|<=0.05) =", 0.1*0.5, " expected in k=64 neighbourhood =", 64*0.05)

print("\n=== value gap under neutral continuations (frozen plant, 100 high-load starts) ===")
def reward(x, a):
    return -(np.sum(x[:10]**2) + cost*np.sum(np.asarray(a)**2))
def rollout_value(x0, t0, a_first, cont):
    x = x0.copy(); v = 0.0
    for j in range(H - t0):
        a = a_first if j == 0 else cont(x)
        a32 = np.asarray(a, dtype=np.float32).astype(np.float64)
        v += gamma**j * reward(x, a32)
        x = A @ x + B @ thrust(a32)
    return v
starts = draw_start(rec, 4000); starts = starts[starts[:, 10]+starts[:, 11] >= 0.4][:100]
cands = [-0.9, -0.85, -0.8, -0.75, -0.7, 0.0, 0.5]
zero = lambda x: np.zeros(4)
def prop(x):  # load-opposing proportional rule on thruster 0 only
    return np.array([-np.clip(x[10]+x[11], -1, 1), 0, 0, 0])
for name, cont in [("zero continuation", zero), ("load-opposing proportional", prop)]:
    V = np.array([[rollout_value(x, 0, np.array([a0, 0, 0, 0]), cont) for a0 in cands] for x in starts])
    d1 = V[:, 2]-V[:, 4]; d2 = V[:, 2]-V[:, 0]
    spread6 = V[:, :6].max(1)-V[:, :6].min(1); spread5 = V[:, :5].max(1)-V[:, :5].min(1)
    print(f"{name}: v(-0.8)-v(-0.7) mean {d1.mean():+.3f} [{d1.min():+.2f},{d1.max():+.2f}]; v(-0.8)-v(-0.9) mean {d2.mean():+.3f}; "
          f"spread over 5 band/ring values mean {spread5.mean():.2f} max {spread5.max():.2f}; with a0=0 added mean {spread6.mean():.2f}; "
          f"v(-0.8)-v(0.0) mean {(V[:,2]-V[:,5]).mean():+.2f}; v(-0.8)-v(+0.5) mean {(V[:,2]-V[:,6]).mean():+.2f}")

print("\n=== development banks: within-state spread of v and of error q1-v ===")
res = json.load(open(RES))
states = res["development_banks"]["states"]
vs = []; es = []; q1s = []
for s in states:
    v = np.array(s["v"]); q1 = np.array(s["q1"]); e = q1 - v
    vs.append(v.max()-v.min()); es.append(e.std(ddof=0)); q1s.append(q1.max()-q1.min())
print("n states", len(states), " median v spread", np.median(vs), " median within-state std of (q1-v)", np.median(es),
      " median q1 spread", np.median(q1s), " 25/75 pct within-state error std", np.percentile(es, [25, 75]))
off = res["official"]
for key in ["development_1", "development_2"]:
    q = off[key]["q1_errors"]
    print(key, "mean_absolute_error" , q.get("mean_absolute_error", {}).get("mean") if isinstance(q.get("mean_absolute_error"), dict) else q.get("mean_absolute_error"))

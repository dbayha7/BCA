"""Analytic (not measured) coverage expectations for the conditional OOD draft, plus a receipt hash check.
Read-only: no dataset arrays decoded; no simulator; only closed-form arithmetic and the frozen matrices
from docs/thrusterchain/site/data/environment.json."""
import sys, json, math, hashlib, pathlib
sys.stdout.reconfigure(encoding="utf-8")
import numpy as np

REPO = pathlib.Path(r"C:\Users\David\dev\BCA")
env = json.load(open(REPO / "docs/thrusterchain/site/data/environment.json", encoding="utf-8"))

print("== receipt hash check ==")
rec = json.load(open(REPO / "experiments/thrusterchain_bca/zeus_20261026_validation/conditional_ood_design_v1/SOURCE_RECEIPT.json", encoding="utf-8"))
bad = 0
for f in rec["files"]:
    p = REPO / f["path"]
    h = hashlib.sha256(p.read_bytes()).hexdigest() if p.exists() else None
    ok = h == f["sha256"] and p.stat().st_size == f["bytes"]
    bad += (not ok)
    if not ok: print("  MISMATCH", f["path"], h)
print("  files listed:", len(rec["files"]), " mismatches:", bad)
dp = REPO / rec["dataset"]["path"]
print("  dataset present:", dp.exists(), " bytes:", dp.stat().st_size if dp.exists() else None)
# hash of a 25 MB file is cheap and decodes nothing
print("  dataset sha256 matches:", hashlib.sha256(dp.read_bytes()).hexdigest() == rec["dataset"]["sha256"])
dr = REPO / "experiments/thrusterchain_bca/zeus_20261026_validation/conditional_ood_design_v1/DRAFT_PROTOCOL.md"
print("  draft sha256 matches:", hashlib.sha256(dr.read_bytes()).hexdigest() == rec["draft_document"]["sha256"], dr.stat().st_size)

print("\n== plant facts from environment.json ==")
print("  physics:", env["physics"])
print("  reset start_laws:", json.dumps(env["reset"]["start_laws"], indent=None)[:1500])
print("  observation_bounds:", env["observation_bounds"])
A = np.array(env["A"], dtype=float); B = np.array(env["B"], dtype=float)
print("  A shape", A.shape, " B shape", B.shape)
print("  dynamics.v:", env["dynamics"].get("v"))
print("  dynamics.B1:", env["dynamics"].get("B1"))
print("  equations:", env["dynamics"].get("equations"))

print("\n== action-band share under the uniform recorder (density 1/16 on [-1,1]^4) ==")
k = float(np.float32(0.8))
for h in (0.025, 0.05, 0.10):
    lo, hi = -k - h, -k + h
    share = (hi - lo) / 2.0  # marginal of a0 is uniform on [-1,1]
    print(f"  h={h}: a0 in [{lo:.4f},{hi:.4f}]  share={share:.4f}  rows/76800={share*76800:.0f}  per 100-step episode={share*100:.1f}")

print("\n== P(g >= 0.4) where g = c + w1, c~U[0.1,0.7], w1 = A cos(phi), A~U[0.1,0.4], phi~U[0,2pi) ==")
# numeric integration on a grid (closed form up to the arcsin integral)
cs = np.linspace(0.1, 0.7, 601); As = np.linspace(0.1, 0.4, 301); phis = np.linspace(0, 2*np.pi, 720, endpoint=False)
w1 = (As[:, None] * np.cos(phis)[None, :]).ravel()          # 216,720 values
def p_ge(thr):
    # P(c + w1 >= thr) = E_w1[ P(c >= thr - w1) ], c uniform on [0.1, 0.7]
    need = thr - w1
    pc = np.clip((0.7 - need) / 0.6, 0.0, 1.0)
    return float(pc.mean())
for thr in (0.3, 0.4, 0.5):
    print(f"  threshold {thr}: P = {p_ge(thr):.4f}")
p04 = p_ge(0.4)
# Because the wave rotates with fixed amplitude and uniform phase, the per-step marginal of w1 equals its t=0 marginal.
print("  per-step marginal equals the t=0 law (uniform phase is rotation invariant), so the step share with g>=0.4 is also", round(p04, 4))
print("  BUT within an episode g is deterministic in t: an episode with c + A < 0.4 (c < 0.4 - A) never reaches high load.")
# fraction of episodes that EVER reach g>=0.4: need c + A >= 0.4
ever = np.mean((cs[:, None] + As[None, :]) >= 0.4)
never = 1 - ever
always = np.mean((cs[:, None] - As[None, :]) >= 0.4)
print(f"  episodes that ever reach g>=0.4: {ever:.4f}; never: {never:.4f}; always (c - A >= 0.4): {always:.4f}")
print("  wave period = 2pi/0.27 =", round(2*math.pi/0.27, 2), "steps, so within 100 steps each episode sees ~4.3 wave cycles")

print("\n== joint expectations (action and state independent under the recorder) ==")
for h in (0.025, 0.05, 0.10):
    share = h  # (2h/2)
    rows = share * p04 * 76800
    print(f"  h={h}: expected hole rows in train pool = {rows:.0f} of 76,800 ({100*share*p04:.2f}%); expected per episode = {rows/768:.2f}")
print("  expected hole rows among the 128 calibration episodes (12,800 rows):", round(0.05 * p04 * 12800, 1))
print("  expected calibration episodes with at least one hole row (h=0.05):  cannot be closed-form without g(t) per episode; upper bound = episodes that ever reach g>=0.4 =", round(ever*128,1))
# per-episode: rows with g>=0.4 in an episode = n_hi(c,A); hole hits ~ Binomial(n_hi, 0.05). P(no hit) = 0.95^n_hi.
# approximate n_hi distribution by simulating the deterministic g(t) over the start law grid (analytic, no simulator: g doesn't depend on actions)
rng = np.random.default_rng(0)
N = 50000
c = rng.uniform(0.1, 0.7, N); Aa = rng.uniform(0.1, 0.4, N); ph = rng.uniform(0, 2*np.pi, N)
t = np.arange(100)
g = c[:, None] + Aa[:, None] * np.cos(ph[:, None] + 0.27 * t[None, :])
n_hi = (g >= 0.4).sum(1)
for h in (0.025, 0.05, 0.10):
    pmiss = np.mean((1 - h) ** n_hi)
    print(f"  h={h}: P(an episode has no hole row) = {pmiss:.3f}; expected calibration episodes (of 128) with no eligible hole row = {pmiss*128:.1f}; mean high-load steps per episode = {n_hi.mean():.1f}")
print("  share of steps with g>=0.4 (MC over start law, deterministic wave):", round((g >= 0.4).mean(), 4))

print("\n== one-step physics check: effect of thruster 1 at high load ==")
# s' = A s + 0.3 * B1 thrust(a) + 0.3 * v (c + w1)
def thrust(a, slope=1.5):
    a = np.asarray(a, float)
    return np.clip(a, -k, k) - slope * np.sign(a) * np.maximum(np.abs(a) - k, 0)
v = np.array(env["dynamics"]["v"], float) if isinstance(env["dynamics"].get("v"), list) else None
B1 = np.array(env["dynamics"]["B1"], float) if isinstance(env["dynamics"].get("B1"), list) else None
print("  v:", v); print("  B1:\n", B1)
if v is not None and B1 is not None:
    A10 = A if A.shape == (10, 10) else A[:10, :10]
    s = np.zeros(10); gload = 0.4
    for a0 in (-1.0, -0.85, -0.8, -0.75, -0.6, -0.4, 0.0):
        a = np.array([a0, 0, 0, 0])
        s1 = A10 @ s + 0.3 * B1 @ thrust(a) + 0.3 * v * gload
        print(f"  a0={a0:+.2f}: thrust={thrust(a)[0]:+.3f}  next speeds v1..v5 = {np.round(s1[1::2], 4)}  next positions = {np.round(s1[0::2], 4)}")
    print("  water push per speed row per step at g=0.4:", 0.3 * gload, " thruster-1 counter push at the knee:", 0.3 * (B1 @ thrust([-k, 0, 0, 0]))[1::2])

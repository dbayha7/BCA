# Review of DECISION.md: statistics and power. Recomputes every return number from the recorded values.json files
# of near-80 version 8 (and the gate's 30 stalling-world data cards), then the C1 lead under refusals and the C2 rule's
# null behaviour. Reads only; writes nothing but stdout.
import json, glob, math, re
from pathlib import Path
import numpy as np

RES = Path(r"B:/outputs/shift/beyond-the-data/near80/results-20261020T115246Z/files")
SIT = ["strong_current", "recorded", "big_waves", "big_kicks"]

def load(setting, world, seed):
    p = RES / f"eval/{setting}/{world}/seed{seed}/values.json"
    if not p.exists():
        return None
    d = json.load(open(p))
    ids = d.get("arm_ids") or {a: a for a in d["evaluations"]}
    return {name: {s: d["evaluations"][aid][s]["points"] for s in SIT if s in d["evaluations"][aid]}
            for name, aid in ids.items() if aid in d["evaluations"]}

sets = {"main": [0, 1, 2, 3, 4, 11, 14], "near80": [0, 1, 2, 3, 4], "random_rows": [0, 1, 2, 3, 4]}
P = {}
for setting, seeds in sets.items():
    for s in seeds:
        P[(setting, s)] = {w: load(setting, w, s) for w in ("stall", "no_stall")}

def net(setting, s, arm):
    st, ns = P[(setting, s)]["stall"], P[(setting, s)]["no_stall"]
    gain = st[arm]["strong_current"] - st["none"]["strong_current"]
    loss = ns["none"]["strong_current"] - ns[arm]["strong_current"]
    return gain - loss

print("== NET across worlds, strong current (gain in the stalling world minus loss in the no-stall world)")
for setting, seeds in sets.items():
    arms = [a for a in P[(setting, seeds[0])]["stall"] if a in P[(setting, seeds[0])]["no_stall"]]
    for arm in arms:
        vals = [net(setting, s, arm) for s in seeds]
        print(f"  {setting:12s} {arm:14s} " + ", ".join(f"{v:6.1f}" for v in vals) + f"   mean {np.mean(vals):6.1f}")

print("\n== The composite 'mean-difference switch turning on the hand-set range limit' = range_hinge's stalling gain (switch on in every stalling world, off in every no-stall world, 40 of 40 recorded)")
rows = []
for setting, seeds in sets.items():
    for s in seeds:
        st = P[(setting, s)]["stall"]
        comp = st["range_hinge"]["strong_current"] - st["none"]["strong_current"]
        bca = net(setting, s, "obca_k1")
        rows.append((setting, s, bca, comp, bca - comp))
        print(f"  {setting:12s} seed {s:2d}: BCA net {bca:6.1f}  composite {comp:6.1f}  BCA minus composite {bca-comp:+6.1f}")
d = np.array([r[4] for r in rows])
print(f"  17 seed-settings: composite ahead by {-d.mean():.2f} on average; composite ahead in {(d<0).sum()} of {len(d)}; sd of the difference {d.std(ddof=1):.2f}")
for setting in sets:
    dd = np.array([r[4] for r in rows if r[0] == setting])
    print(f"    {setting}: BCA minus composite mean {dd.mean():+.2f}, sd {dd.std(ddof=1):.2f}, BCA behind in {(dd<0).sum()} of {len(dd)}")
d12 = np.array([r[4] for r in rows if r[0] != "random_rows"])
print(f"  12 seed-settings (main 0-4, 11, 14; near-80 0-4): composite ahead in {(d12<0).sum()} of 12, mean {-d12.mean():.2f}")
from math import comb
k = int((d < 0).sum()); n = len(d)
print(f"  one-sided sign test, composite ahead in {k} of {n}: p = {sum(comb(n,j) for j in range(k,n+1))/2**n:.3f}")

print("\n== Seeds 11 and 14 (description only): leads, keeps, twice-height")
for s in (11, 14):
    st, ns = P[("main", s)]["stall"], P[("main", s)]["no_stall"]
    bca = net("main", s, "obca_k1")
    rivals = {a: net("main", s, a) for a in ("obca_forced", "standin", "range_hinge", "even_pull", "random_1", "random_2")}
    best = max(rivals, key=rivals.get)
    keeps = [st["obca_k1"][x] - st["none"][x] for x in ("recorded", "big_waves", "big_kicks")]
    nostall_loss = [ns["none"][x] - ns["obca_k1"][x] for x in SIT]
    print(f"  seed {s}: BCA net {bca:.1f}; world-blind rivals " + ", ".join(f"{a} {v:.1f}" for a, v in rivals.items())
          + f"; lead over best ({best}) {bca-rivals[best]:.1f}; keeps (recorded, waves, kicks) " + ", ".join(f"{k:.1f}" for k in keeps)
          + f"; no-stall losses {nostall_loss}; k2 minus k1 (stalling) {st['obca_k2']['strong_current']-st['obca_k1']['strong_current']:+.1f}"
          + f"; even pull margin {bca-rivals['even_pull']:.1f}")

print("\n== Is the hand-set range limit below the always-on ramp on every recorded set? (NET across worlds)")
for setting, seeds in sets.items():
    if "obca_forced" not in P[(setting, seeds[0])]["stall"]:
        print(f"  {setting}: obca_forced not trained"); continue
    rh = np.mean([net(setting, s, "range_hinge") for s in seeds]); fo = np.mean([net(setting, s, "obca_forced") for s in seeds])
    print(f"  {setting:12s} seeds {seeds}: range_hinge {rh:.1f}, always-on ramp {fo:.1f}  -> range limit below: {rh < fo}")
for s in (11, 14):
    print(f"  main seed {s}: range_hinge {net('main', s, 'range_hinge'):.1f}, always-on ramp {net('main', s, 'obca_forced'):.1f}")

print("\n== Dose ladder per seed, stalling world, strong current (meangap < fixed 1.0 < conformal < BCA < 2xBCA?)")
for setting, seeds in sets.items():
    if "obca_k2" not in P[(setting, seeds[0])]["stall"]:
        continue
    for s in seeds:
        st = P[(setting, s)]["stall"]
        lad = [st[a]["strong_current"] for a in ("meangap_k1", "fixed_gated", "conformal_k1", "obca_k1", "obca_k2")]
        mono = all(lad[i] < lad[i+1] for i in range(4))
        mono4 = all(lad[i] < lad[i+1] for i in range(3))
        print(f"  {setting:8s} seed {s:2d}: " + ", ".join(f"{v:5.1f}" for v in lad) + f"  monotone through 2xBCA: {mono}; up to BCA: {mono4}; range_hinge {st['range_hinge']['strong_current']:.1f}")

print("\n== The harm in the main setting (full-log actor minus no caution, stalling, strong current, seeds 0-4)")
h = [P[("main", s)]["stall"]["full_none"]["strong_current"] - P[("main", s)]["stall"]["none"]["strong_current"] for s in range(5)]
print(f"  per seed {[round(x,1) for x in h]}, mean {np.mean(h):.1f}")

print("\n== The gate's 30 stalling-world data cards: kept_past80_rows")
gate = {}
for p in glob.glob(r"B:/outputs/shift/beyond-the-data/stage2_gpu/payload/files/runs__shift__pre__data__base__stall__seed*__record.json"):
    d = json.load(open(p)); seed = int(re.search(r"seed(\d+)__", p).group(1))
    wc = d.get("world_card", {})
    gate[seed] = wc.get("kept_past80_rows")
vals = np.array(sorted(gate.values()))
print(f"  n = {len(gate)}; values {vals.tolist()}; mean {vals.mean():.2f}; below 29: {(vals<29).sum()}; <= 14: {(vals<=14).sum()}; <= 18: {(vals<=18).sum()}")
print(f"  seeds with <= 14: {[k for k,v in gate.items() if v<=14]}")

# certification arithmetic
from math import fsum
def pcert(n):
    p = 0.9 ** n
    return fsum(comb(1000, k) * p ** k * (1 - p) ** (1000 - k) for k in range(0, 51))
print("\n== P(certify | n) exact: " + ", ".join(f"{n}: {pcert(n):.3f}" for n in (18, 22, 24, 28, 29, 30, 31, 32, 33, 35)))
per_present = np.mean([pcert(k + 1) for k in vals]); per_present0 = np.mean([pcert(k) for k in vals])
print(f"  present rule, gate spread: per seed {per_present0:.3f} (+1 bank row: {per_present:.3f}); all five {per_present0**5:.3f} ({per_present**5:.3f})")
main10 = [35, 47, 59, 43, 47, 19, 53, 22, 18, 30]
pm = np.mean([pcert(k) for k in main10]); print(f"  present rule, 10 main-setting stalling groups: per seed {pm:.3f}; all five {pm**5:.3f}")

def pois(k, lam): return math.exp(-lam + k * math.log(lam) - math.lgamma(k + 1))
def pseed_pool(factor, kept):
    return np.array([sum(pois(n, factor * k) * pcert(n + 1) for n in range(0, 700)) for k in kept])
for f in (1.8, 2.0, 2.16, 2.3):
    ps = pseed_pool(f, vals); p = ps.mean()
    print(f"  pooled rule, Poisson({f} x kept)+1: per seed {p:.4f}; 5 seeds all certify {p**5:.3f}; 10 seeds: P(0 ref) {p**10:.3f}, P(<=1) {p**10 + 10*p**9*(1-p):.3f}; per-seed refusal 1 in {1/(1-p):.0f}")
# the preview's observed pooled/survivor ratios
prev = {(0,'s'):(70,34),(1,'s'):(106,45),(2,'s'):(121,59),(3,'s'):(100,42),(4,'s'):(110,45),(10,'s'):(35,18),(11,'s'):(109,51),(12,'s'):(46,21),(13,'s'):(38,18),(14,'s'):(58,30)}
r = np.array([a/b for a,b in prev.values()]); print(f"  preview: pooled stalling group / survivors = {np.round(r,2).tolist()}, mean {r.mean():.2f}")

print("\n== C1's net-lead mark under refusals: BCA net on certified seeds, 0 on refused; rival = hand-set range limit")
pairs = [(net("main", s, "obca_k1"), net("main", s, "range_hinge"), net("main", s, "obca_forced")) for s in sets["main"]]
bca7 = np.array([p[0] for p in pairs]); rh7 = np.array([p[1] for p in pairs]); fo7 = np.array([p[2] for p in pairs])
print(f"  7 recorded main seeds: BCA net {np.round(bca7,1).tolist()} (mean {bca7.mean():.1f}, sd {bca7.std(ddof=1):.1f}); range_hinge net {np.round(rh7,1).tolist()} (mean {rh7.mean():.1f}); always-on {np.round(fo7,1).tolist()} (mean {fo7.mean():.1f})")
diff7 = bca7 - rh7
print(f"  BCA minus range_hinge per seed {np.round(diff7,1).tolist()}: mean {diff7.mean():.1f}, sd {diff7.std(ddof=1):.1f}")
rng = np.random.default_rng(20261020)
for n in (5, 10):
    for r in range(0, 4):
        # resample seeds from the 7 recorded pairs (bootstrap); refused seeds: BCA 0, rival unchanged; always-on also refused
        idx = rng.integers(0, 7, size=(200000, n))
        b = bca7[idx]; rh = rh7[idx]; fo = fo7[idx]
        if r:
            b[:, :r] = 0.0; fo[:, :r] = 0.0
        lead_rh = b.mean(1) - rh.mean(1); lead_fo = b.mean(1) - fo.mean(1)
        lead = np.minimum(lead_rh, lead_fo)
        plan = (n - r) / n * 105 - 76.5
        print(f"  n={n}, refusals={r}: lead over the best of (range limit, always-on): mean {lead.mean():.1f}, sd {lead.std():.1f}, P(lead >= 10) {np.mean(lead>=10):.2f}   [planning formula: {plan:.1f}]")

print("\n== C2 rule: false 'ahead' under a true zero edge, and 'any twin ahead' under a true tie; margin 5, 8 of 10")
def run(mu, sd, n, reps=400000):
    x = rng.normal(mu, sd, size=(reps, n)); m = x.mean(1); pos = (x > 0).sum(1)
    return ((m >= 5) & (pos >= 8)).mean(), ((m <= -5) & (pos <= 2)).mean()
for sd in (5, 7.5, 10, 15):
    a, b = run(0, sd, 10)
    print(f"  n=10, true edge 0, sd {sd:4}: 'BCA ahead' {a:.3f}; 'twin ahead' {b:.3f}; any of 3 twins read 'ahead of BCA' (independent) {1-(1-b)**3:.3f}")
for mu, sd in ((5, 5), (5, 10), (10, 10), (-4.5, 5.2), (-3.2, 5.0)):
    a, b = run(mu, sd, 10)
    print(f"  n=10, true edge {mu:+.1f}, sd {sd}: BCA ahead {a:.2f}, twin ahead {b:.2f}, tie {1-a-b:.2f}")
print(f"  coin: >= 8 of 10 positive {sum(comb(10,j) for j in range(8,11))/1024:.4f}; >= 4 of 5 {sum(comb(5,j) for j in range(4,6))/32:.4f}")

print("\n== The dose line: 17.3 points per ln(dose)? pooled heights over conformal's -> predicted gap")
ratios = [2.775/2.480, 1.671/1.200, 1.851/1.639, 1.845/1.408, 2.256/0.987]
pred = [17.33*math.log(x) for x in ratios]
print(f"  ratios {np.round(ratios,2).tolist()}; predicted BCA-conformal points {np.round(pred,1).tolist()}; mean {np.mean(pred):.1f}")
print(f"  2 x plain over BCA: {np.round([2*0.987/2.256, 2*1.2/1.671, 2*1.639/1.851, 2*1.408/1.845, 2*2.48/2.775],2).tolist()} -> plain x2 below BCA on seed 14: {2*0.987 < 2.256}")

print("\n== Version 8 ledger: confirmatory CPU-hours")
led = json.load(open(RES / "looks/confirmatory.json")) if (RES / "looks/confirmatory.json").exists() else None
if led and "resources" in led:
    print("  keys:", list(led["resources"])[:5])

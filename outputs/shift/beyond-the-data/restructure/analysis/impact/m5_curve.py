"""Read-only: recall / false-flag trade-off of saved scores on T3a (no-caution actor's own strong-current commands),
and a simple fit of check-minus-random on deployment recall and false-flag rate. Saved files only."""
import json
import numpy as np
from statistics import NormalDist

KNEE = float(np.float32(0.8))
DET = "/home/user/bca/runs/shift_explore/detect/seed{k}/stall"
S2D = "/home/user/bca/runs/shift_explore/stage2/detect/{v}/stall/fixed/seed{k}"
N = NormalDist()

print("Trade-off on T3a: threshold giving recall 0.90 / 0.95 on past-80% commands -> false-flag rate, T4 (recorded) rate")
res = {}
for name in ("bca_weighted", "bca_uniform", "classifier_ratio", "distance", "two_head"):
    rows = []
    for k in range(5):
        P = np.load(DET.format(k=k) + "/pairs.npz")
        sets, role, src = P["set"], P["role"], P["t3_source"]
        p80 = np.any(np.abs(np.asarray(P["commands"], np.float64)) > KNEE, axis=1)
        t3a_cmd = (sets == "T3") & (src == "actor") & (role == "command")
        t4 = sets == "T4"
        if name.startswith("bca_"):
            v = name[4:]
            sc = np.load(S2D.format(v=v, k=k) + f"/scores/bca_width_{v}_fixed.npz")["scores"]
        else:
            sc = np.load(DET.format(k=k) + f"/scores/{name}.npz")["scores"]
        pos, neg = sc[t3a_cmd & p80], sc[t3a_cmd & ~p80]
        r = {}
        for tr in (0.90, 0.95, 0.98):
            thr = np.quantile(pos, 1 - tr)
            r[tr] = dict(ff=float((neg > thr).mean()), t4=float((sc[t4] > thr).mean()))
        rows.append(r)
    res[name] = rows
    for tr in (0.90, 0.95, 0.98):
        print("%-17s recall %.2f: false-flag %s mean %.2f | recorded-row rate %s mean %.2f" % (
            name, tr, [round(x[tr]["ff"], 2) for x in rows], np.mean([x[tr]["ff"] for x in rows]),
            [round(x[tr]["t4"], 2) for x in rows], np.mean([x[tr]["t4"] for x in rows])))

# binormal equal-variance: AUC needed for TPR at FPR
print()
for tpr in (0.90, 0.95):
    for fpr in (0.15, 0.25, 0.30):
        d = N.inv_cdf(tpr) + N.inv_cdf(1 - fpr)
        print("binormal: recall %.2f at false-flag %.2f needs d' %.2f, AUC %.3f" % (tpr, fpr, d, N.cdf(d / 2 ** 0.5)))
for a in (0.80, 0.82, 0.86):
    d = 2 ** 0.5 * N.inv_cdf(a)
    for fpr in (0.25,):
        print("binormal: AUC %.2f -> recall at false-flag %.2f = %.2f" % (a, fpr, 1 - N.cdf(N.inv_cdf(1 - fpr) - d)))

# deployment: check-minus-random against recall and false-flag (no-caution actor, strong current, clip fallback)
pts = []
S2DEP = "/home/user/bca/runs/shift_explore/stage2/deploy/stall/seed{k}/values.json"
D1 = "/home/user/bca/runs/shift_explore/deploy/seed{k}_stall_gapped.json"
for k in range(5):
    dep = json.load(open(S2DEP.format(k=k)))["results"]["none"]["strong_current"]
    d1 = json.load(open(D1.format(k=k)))["situations"]["strong_current"]["rules"]
    pts.append(("bca", k, dep["bca"]["recall_past80"], dep["bca"]["flag_rate_not_past80"], dep["bca"]["fire_share"],
                dep["check_minus_random"]))
    for t in ("unrec", "ratio", "onestep", "heads", "sa95", "ratio_refuse", "density"):
        a, b = d1[f"{t}:clip99"], d1[f"random:{t}:clip99"]
        pts.append((t, k, a["recall_past80"], a["flag_rate_not_past80"], a["fire_share"], a["points"] - b["points"]))
print()
print("trigger   mean recall  false-flag  fire  minus-random (per seed)")
for t in ("unrec", "onestep", "ratio", "bca", "heads", "sa95", "ratio_refuse", "density"):
    q = [p for p in pts if p[0] == t]
    print("%-12s %.3f %.3f %.3f %6.1f  %s" % (t, np.mean([p[2] for p in q]), np.mean([p[3] for p in q]),
                                              np.mean([p[4] for p in q]), np.mean([p[5] for p in q]),
                                              [round(p[5], 1) for p in q]))
hi = [p for p in pts if p[0] in ("unrec", "ratio", "bca")]
X = np.array([[1, p[2], p[3]] for p in hi])
y = np.array([p[5] for p in hi])
coef, *_ = np.linalg.lstsq(X, y, rcond=None)
pred = X @ coef
print("fit on high-rate triggers (perfect, classifier ratio, BCA; 15 seed-points): minus_random = %.1f + %.1f*recall + %.1f*false_flag; R^2 %.2f"
      % (coef[0], coef[1], coef[2], 1 - ((y - pred) ** 2).sum() / ((y - y.mean()) ** 2).sum()))
for ff in (0.10, 0.24, 0.30):
    print("  recall needed for +10 at false-flag %.2f: %.3f" % (ff, (10 - coef[0] - coef[2] * ff) / coef[1]))
X1 = np.array([[1, p[2]] for p in hi])
c1, *_ = np.linalg.lstsq(X1, y, rcond=None)
p1 = X1 @ c1
print("recall-only fit: minus_random = %.1f + %.1f*recall; R^2 %.2f; recall for +10: %.3f" % (
    c1[0], c1[1], 1 - ((y - p1) ** 2).sum() / ((y - y.mean()) ** 2).sum(), (10 - c1[0]) / c1[1]))
json.dump(dict(tradeoff=res, deploy_points=pts, fit=dict(coef=coef.tolist(), recall_only=c1.tolist())),
          open("/tmp/claude-0/-home-user/9e341792-006d-5a11-a8e9-8fac16ac7f5b/scratchpad/impact/m5_curve.json", "w"),
          indent=1, default=float)

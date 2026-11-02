"""Post hoc, descriptive, fixed networks: the anchor-level trade-off between correcting the stall world's s10 ranking
and adding false caution in the no-stall world, enumerated exactly (every knob value where an anchor's penalized sign
changes), in anchor counts.

BCA: penalty kappa x dU. Even caution: a constant penalty c (Q units). One knob is shared by a seed-arm's stall and
no-stall run (the deployed knob cannot know the world). Two tunings:
- per seed-arm: a knob per seed-arm (an oracle that lets kappa absorb each network's U scale; favors BCA);
- global: one knob for all six seed-arms, which is how kappa_A is used.
Gain = harmful stall anchors ranked correctly by the penalized Q minus by raw Q1 (net). False flips = non-harmful
no-stall anchors (C_true >= 0) that raw Q1 ranks s10 at least as good and the penalized Q ranks worse."""
import json
from pathlib import Path
import numpy as np

VAL = Path(r"C:\Users\David\dev\BCA\experiments\thrusterchain_bca\zeus_20261026_validation")
ATT = VAL / "bca_dose_screen_v1" / "screen_run" / "attempt_1"
KAPPA_A = 63.0
rd = lambda p: json.loads(Path(p).read_text(encoding="utf-8"))  # noqa: E731
NAMES = {"nan": np.nan, "positive_infinity": np.inf, "negative_infinity": -np.inf}
arr = lambda v: np.asarray([[NAMES.get(x, x) for x in r] for r in v], np.float64)  # noqa: E731


def contrasts(entry, slot):
    a = entry["anchors_e2"]
    n = a["gated_slots"]
    q1, qpi, u = arr(a["q1"])[:48], arr(a["qpi_clip"])[:48], arr(a["U"])[:48]
    valid = np.asarray(a["valid"], bool)[:48]
    i, k = n.index(slot), n.index("edge_knee")
    ct, cr, du = qpi[:, i] - qpi[:, k], q1[:, i] - q1[:, k], u[:, i] - u[:, k]
    keep = valid & np.isfinite(ct) & np.isfinite(cr) & np.isfinite(du)
    return ct[keep], cr[keep], du[keep]


def counts(stall, nostall, knob, family):
    """(net gain, false flips) at one knob value."""
    def pen(du):
        return knob * du if family == "bca" else np.full_like(du, knob)
    ct, cr, du = stall
    h = ct < 0
    gain = int(((cr - pen(du))[h] < 0).sum() - (cr[h] < 0).sum())
    ct, cr, du = nostall
    ok = ct >= 0
    ff = int(((cr >= 0) & ((cr - pen(du)) < 0))[ok].sum())
    return gain, ff


def candidates(runs, family):
    """0, every positive sign-change point, the midpoints between them and a point past the last."""
    pts = []
    for ct, cr, du in runs:
        if family == "bca":
            with np.errstate(divide="ignore", invalid="ignore"):
                r = cr / du
            pts.append(r[np.isfinite(r) & (r > 0)])
        else:
            pts.append(cr[cr > 0])
    b = np.unique(np.concatenate(pts))
    mids = (b[:-1] + b[1:]) / 2 if b.size > 1 else np.array([])
    return np.concatenate([[0.0], mids, [b[-1] * 2 + 1 if b.size else 1.0]])


def frontier(pairs, family):
    """pairs: [(stall, nostall)]; one shared knob. Returns {false-flip budget: best summed gain}."""
    runs = [r for p in pairs for r in p]
    pts = [(sum(g for g, _ in c), sum(f for _, f in c))
           for c in ([counts(s, n, k, family) for s, n in pairs] for k in candidates(runs, family))]
    top = max(f for _, f in pts)
    return {b: max(g for g, f in pts if f <= b) for b in range(top + 1)}


out = {"note": __doc__.strip()}
for slot in ("s10", "s05"):
    pairs, per = {}, {}
    for i in (0, 1, 2):
        rep = rd(ATT / f"evaluate_s{i}" / "report.json")
        for arm in ("cutoff", "random"):
            name = f"s{i}_{arm}"
            pairs[name] = (contrasts(rep["runs"][f"{name}_stall3"]["020000"], slot),
                           contrasts(rep["runs"][f"{name}_nostall"]["020000"], slot))
    for name, (s, n) in pairs.items():
        g63, f63 = counts(s, n, KAPPA_A, "bca")
        bca, ramp = frontier([(s, n)], "bca"), frontier([(s, n)], "ramp")
        per[name] = {"harmful_stall": int((s[0] < 0).sum()), "non_harmful_nostall": int((n[0] >= 0).sum()),
                     "bca_at_kappa_A": {"gain": g63, "false_flips": f63},
                     "by_false_flips_0_to_8": {"bca": [bca[min(b, max(bca))] for b in range(9)],
                                               "even": [ramp[min(b, max(ramp))] for b in range(9)]}}
    allp = list(pairs.values())
    g63 = sum(counts(s, n, KAPPA_A, "bca")[0] for s, n in allp)
    f63 = sum(counts(s, n, KAPPA_A, "bca")[1] for s, n in allp)
    gb, gr = frontier(allp, "bca"), frontier(allp, "ramp")
    pick = sorted({0, 5, 11, 22, 33, f63, 66, 88})
    out[slot] = {"per_seed_arm": per,
                 "global": {"harmful_stall_total": int(sum((s[0] < 0).sum() for s, _ in allp)),
                            "non_harmful_nostall_total": int(sum((n[0] >= 0).sum() for _, n in allp)),
                            "bca_at_kappa_A": {"gain": g63, "false_flips": f63},
                            "best_gain_at_false_flips": {str(b): {"bca": gb[min(b, max(gb))], "even": gr[min(b, max(gr))]}
                                                         for b in pick},
                            "budgets_even_ahead": int(sum(gr[min(b, max(gr))] > gb[min(b, max(gb))] for b in range(81))),
                            "budgets_bca_ahead": int(sum(gb[min(b, max(gb))] > gr[min(b, max(gr))] for b in range(81)))}}
print(json.dumps(out, indent=1))

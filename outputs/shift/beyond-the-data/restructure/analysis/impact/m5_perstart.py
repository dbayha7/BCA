"""Read-only: per-start analysis of the Stage 2 deployment check (M5) vs the 10-13 channel triggers.
Reads runs/shift_explore/stage2/deploy/stall/seed<K>/values.json and runs/shift_explore/deploy/seed<K>_stall_gapped.json.
Writes nothing outside this scratchpad."""
import json
import numpy as np

S2 = "/home/user/bca/runs/shift_explore/stage2/deploy/{w}/seed{k}/values.json"
D1 = "/home/user/bca/runs/shift_explore/deploy/seed{k}_{w}_gapped.json"
SIT = "strong_current"


def load(k, w="stall", sit=SIT):
    a = json.load(open(S2.format(w=w, k=k)))["results"]
    b = json.load(open(D1.format(k=k, w=w)))["situations"][sit]["rules"]
    out = {}
    for r in ("none", "bca", "random:bca", "sa95", "random:sa95"):
        out["s2:" + r] = a["none"][sit][r]
    for r in ("none:None", "unrec:clip99", "random:unrec:clip99", "ratio:clip99", "random:ratio:clip99",
              "always:clip99", "sa95:clip99", "heads:clip99", "random:heads:clip99"):
        out["d1:" + r] = b[r]
    return out


def pts(x):
    return np.asarray(x["per_start_points"], float)


rows = []
for k in range(5):
    o = load(k)
    never2, never1 = pts(o["s2:none"]), pts(o["d1:none:None"])
    same_never = float(np.max(np.abs(never2 - never1)))
    same_sa = float(np.max(np.abs(pts(o["s2:sa95"]) - pts(o["d1:sa95:clip99"]))))
    bca, rnd, perf, clipall, ratio, rratio = (pts(o[x]) for x in ("s2:bca", "s2:random:bca", "d1:unrec:clip99",
                                                                     "d1:always:clip99", "d1:ratio:clip99",
                                                                     "d1:random:ratio:clip99"))
    deficit = perf - bca            # per start: what BCA's check leaves vs the perfect detector
    d_r = perf - rnd
    srt = np.sort(deficit)[::-1]
    top10 = float(srt[:20].sum() / max(deficit.sum(), 1e-9))
    rows.append(dict(
        seed=k, never_identical_max_abs=same_never, sa95_identical_max_abs=same_sa,
        mean=dict(never=never2.mean(), bca=bca.mean(), random=rnd.mean(), perfect=perf.mean(), clip_all=clipall.mean(),
                  ratio=ratio.mean(), random_ratio=rratio.mean()),
        bca_minus_random=bca.mean() - rnd.mean(), perfect_minus_bca=deficit.mean(),
        share_of_perfect_minus_bca_from_worst_20_starts=top10,
        starts_bca_below_perfect_by_10=int((deficit > 10).sum()),
        starts_random_below_perfect_by_10=int((d_r > 10).sum()),
        starts_bca_worse_than_random=int((bca < rnd - 1e-6).sum()),
        starts_bca_better_than_random=int((bca > rnd + 1e-6).sum()),
        median_bca_minus_random=float(np.median(bca - rnd)),
        mean_bca_minus_random_when_bca_worse=float((bca - rnd)[bca < rnd].mean()),
        mean_bca_minus_random_when_bca_better=float((bca - rnd)[bca > rnd].mean()),
        quantiles_bca_minus_random={q: float(np.quantile(bca - rnd, q)) for q in (0.05, 0.25, 0.5, 0.75, 0.95)},
        quantiles_perfect_minus_bca={q: float(np.quantile(deficit, q)) for q in (0.5, 0.75, 0.9, 0.95)},
        quantiles_perfect_minus_random={q: float(np.quantile(d_r, q)) for q in (0.5, 0.75, 0.9, 0.95)},
        stats={n: dict(fire=o[x]["fire_share"], recall=o[x]["recall_past80"], precision=o[x]["precision_past80"],
                       false_flag=o[x]["flag_rate_not_past80"], past80_proposed=o[x]["past80_proposed"],
                       past80_applied=o[x]["past80_applied"], points=o[x]["points"])
               for n, x in (("bca", "s2:bca"), ("random_bca", "s2:random:bca"), ("perfect", "d1:unrec:clip99"),
                            ("random_perfect", "d1:random:unrec:clip99"), ("ratio", "d1:ratio:clip99"),
                            ("random_ratio", "d1:random:ratio:clip99"), ("sa95", "s2:sa95"),
                            ("heads", "d1:heads:clip99"), ("random_heads", "d1:random:heads:clip99"),
                            ("clip_all", "d1:always:clip99"))}))

json.dump(rows, open("/tmp/claude-0/-home-user/9e341792-006d-5a11-a8e9-8fac16ac7f5b/scratchpad/impact/m5_perstart.json", "w"),
          indent=1, default=float)
for r in rows:
    print("seed", r["seed"], "never identical:", r["never_identical_max_abs"], "sa95 identical:", r["sa95_identical_max_abs"])
    print("  means", {k: round(v, 1) for k, v in r["mean"].items()})
    print("  bca-random %.1f perfect-bca %.1f; worst20 share %.2f; >10 below perfect: bca %d random %d; bca<random %d bca>random %d"
          % (r["bca_minus_random"], r["perfect_minus_bca"], r["share_of_perfect_minus_bca_from_worst_20_starts"],
             r["starts_bca_below_perfect_by_10"], r["starts_random_below_perfect_by_10"],
             r["starts_bca_worse_than_random"], r["starts_bca_better_than_random"]))
    print("  bca-random quantiles", {k: round(v, 1) for k, v in r["quantiles_bca_minus_random"].items()},
          "mean when worse %.1f / better %.1f" % (r["mean_bca_minus_random_when_bca_worse"],
                                                 r["mean_bca_minus_random_when_bca_better"]))
    print("  perfect-bca q", {k: round(v, 1) for k, v in r["quantiles_perfect_minus_bca"].items()},
          " perfect-random q", {k: round(v, 1) for k, v in r["quantiles_perfect_minus_random"].items()})
    for n, s in r["stats"].items():
        print("   %-15s" % n, " ".join("%s %.3f" % (a, b) for a, b in s.items()))

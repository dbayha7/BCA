"""READ-ONLY (log positions and commands only): which split do the detection script's T4 rows come from, and how
many of the held-out calibration-split full-throttle rows are T4 rows?"""
import sys, json
import numpy as np
sys.path.insert(0, "/home/user/bca")
import experiments.shift.acceptance as A  # noqa: E402
p = A.load_prereg(A.refuse_protected(A.RUNS_DEFAULT)); A.guard(p); variant = A.gate_outcome(p)["variant"]
def gap_rule(t):
    hi = np.sort(t[t > np.quantile(t, 0.99)]); d = np.diff(hi); i = int(np.argmax(d)); return float(hi[i])
out = {}
for rep in range(5):
    for world in ("stall", "no_stall"):
        sd = A.load_seed(p.runs, p.config, variant, world, rep); lg = sd.log; rg = sd.removals["gap"]
        pz = np.load(f"/home/user/bca/runs/shift_explore/detect/seed{rep}/{world}/pairs.npz")
        t4 = np.asarray(pz["log_row"][(pz["set"] == "T4")], np.int64)
        split = np.asarray(lg["infos/split"])[t4]
        kept = rg.kept_train_rows(lg); tk = np.abs(np.asarray(lg["actions"][kept], np.float64)).max(1); t_lo = gap_rule(tk)
        bank = rg.bank_rows(lg)
        cal = lg.split_rows("calibration"); cal = cal[~sd.gap.contains(lg["actions"][cal])]; held = np.setdiff1d(cal, bank)
        th = np.abs(np.asarray(lg["actions"][held], np.float64)).max(1); tail = held[th > t_lo]
        out[f"{rep}_{world}"] = dict(t4_split_counts={int(k): int(v) for k, v in zip(*np.unique(split, return_counts=True))},
                                     held_tail=int(len(tail)), held_tail_in_t4=int(np.isin(tail, t4).sum()),
                                     held_tail_not_t4=int((~np.isin(tail, t4)).sum()))
        print(rep, world, out[f"{rep}_{world}"], flush=True)
json.dump(out, open("/tmp/claude-0/-home-user/9e341792-006d-5a11-a8e9-8fac16ac7f5b/scratchpad/routes/_feasibility/t4_overlap.json", "w"), indent=1)

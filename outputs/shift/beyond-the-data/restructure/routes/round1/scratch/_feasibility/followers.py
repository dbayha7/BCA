"""READ-ONLY feasibility (log positions and commands only; no reward, no next state, no critic): for each full-throttle
row past t_lo (training survivors; calibration-split rows outside the ball and the bank), how many of the next 1..10
steps of the same episode are present in the same usable set (kept training rows / held-out calibration rows), and
what throttle the recorder used on those next steps."""
import json, sys
import numpy as np
sys.path.insert(0, "/home/user/bca")
import experiments.shift.acceptance as A  # noqa: E402

def gap_rule(t):
    hi = np.sort(t[t > np.quantile(t, 0.99)]); d = np.diff(hi); i = int(np.argmax(d)); return float(hi[i])

p = A.load_prereg(A.refuse_protected(A.RUNS_DEFAULT)); A.guard(p)
variant = A.gate_outcome(p)["variant"]
out = {}
for rep in [int(s) for s in sys.argv[1:]]:
    for world in ("stall", "no_stall"):
        sd = A.load_seed(p.runs, p.config, variant, world, rep); lg = sd.log; rg = sd.removals["gap"]
        kept = rg.kept_train_rows(lg)
        tk = np.abs(np.asarray(lg["actions"][kept], np.float64)).max(1); t_lo = gap_rule(tk)
        bank = rg.bank_rows(lg)
        cal = lg.split_rows("calibration"); cal = cal[~sd.gap.contains(lg["actions"][cal])]; held = np.setdiff1d(cal, bank)
        th = np.abs(np.asarray(lg["actions"][held], np.float64)).max(1)
        res = {}
        for name, usable, tail in (("train", kept, kept[tk > t_lo]), ("cal", held, held[th > t_lo])):
            us = set(usable.tolist())
            ep = np.asarray(lg["infos/episode"]); tt = np.asarray(lg["infos/t"])
            avail = {n: 0 for n in (1, 3, 5, 10)}
            nxt_t = []
            for r in tail:
                for n in (1, 3, 5, 10):
                    ok = all((r + i) in us and ep[r + i] == ep[r] and tt[r + i] == tt[r] + i for i in range(1, n + 1))
                    avail[n] += int(ok)
                if r + 1 < len(ep) and ep[r + 1] == ep[r]:
                    nxt_t.append(float(np.abs(np.asarray(lg["actions"][r + 1], np.float64)).max()))
            res[name] = dict(n_tail=int(len(tail)), all_next_n_usable={str(k): v for k, v in avail.items()},
                             next_step_throttle_q=[float(np.quantile(nxt_t, q)) for q in (0.1, 0.5, 0.9)] if nxt_t else [])
        out[f"{rep}_{world}"] = res
        print(rep, world, res, flush=True)
json.dump(out, open(f"/tmp/claude-0/-home-user/9e341792-006d-5a11-a8e9-8fac16ac7f5b/scratchpad/routes/_feasibility/followers_{'_'.join(sys.argv[1:])}.json", "w"), indent=1)

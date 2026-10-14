"""READ-ONLY feasibility count (commands and log positions only; no reward, no next state, no critic).
For each gate seed 0-4 and world: the gap rule's t_lo on the gapped training commands, then how many rows past t_lo
exist in (a) the kept training rows (survivors), (b) the calibration bank, (c) the calibration split outside the
removed ball and outside the bank (Stage 2's tau rows). Also: are training survivors consecutive in time, and what
current they sit at."""
import json, sys
import numpy as np
sys.path.insert(0, "/home/user/bca")
import experiments.shift.acceptance as A  # noqa: E402

def gap_rule(t):
    hi = np.sort(t[t > np.quantile(t, 0.99)])
    d = np.diff(hi)
    i = int(np.argmax(d))
    return float(hi[i]), float(hi[i + 1])

p = A.load_prereg(A.refuse_protected(A.RUNS_DEFAULT))
A.guard(p)
variant = A.gate_outcome(p)["variant"]
out = {}
for rep in [int(s) for s in sys.argv[1:]]:
    for world in ("stall", "no_stall"):
        sd = A.load_seed(p.runs, p.config, variant, world, rep)
        lg = sd.log
        rg = sd.removals["gap"]
        kept = rg.kept_train_rows(lg)
        tk = np.abs(np.asarray(lg["actions"][kept], np.float64)).max(1)
        t_lo, t_up = gap_rule(tk)
        surv = kept[tk > t_lo]
        bank = rg.bank_rows(lg)
        tb = np.abs(np.asarray(lg["actions"][bank], np.float64)).max(1)
        cal = lg.split_rows("calibration")
        cal = cal[~sd.gap.contains(lg["actions"][cal])]
        held = np.setdiff1d(cal, bank)
        th = np.abs(np.asarray(lg["actions"][held], np.float64)).max(1)
        ep = np.asarray(lg["infos/episode"][surv]); tt = np.asarray(lg["infos/t"][surv])
        cur_s = np.asarray(lg["infos/state"][surv])[:, 10]
        held_tail = held[th > t_lo]
        cur_h = np.asarray(lg["infos/state"][held_tail])[:, 10] if len(held_tail) else np.zeros(0)
        # consecutive survivors: same episode, t differs by 1
        order = np.lexsort((tt, ep)); e2, t2 = ep[order], tt[order]
        consec = int(np.sum((np.diff(e2) == 0) & (np.diff(t2) == 1)))
        out[f"{rep}_{world}"] = dict(t_lo=t_lo, t_up=t_up, n_surv=int(len(surv)), n_bank=int(len(bank)),
            n_bank_tail=int((tb > t_lo).sum()), n_heldcal=int(len(held)), n_heldcal_tail=int(len(held_tail)),
            heldcal_tail_episodes=int(len(np.unique(np.asarray(lg["infos/episode"][held_tail])))) if len(held_tail) else 0,
            surv_episodes=int(len(np.unique(ep))), surv_consecutive_pairs=consec,
            surv_current_q=[float(np.quantile(cur_s, q)) for q in (0, .5, 1)],
            heldtail_current_q=[float(np.quantile(cur_h, q)) for q in (0, .5, 1)] if len(cur_h) else [],
            surv_cmd_signs=np.round(np.asarray(lg["actions"][surv], np.float64), 2)[:3].tolist())
        print(rep, world, out[f"{rep}_{world}"], flush=True)
json.dump(out, open(f"/tmp/claude-0/-home-user/9e341792-006d-5a11-a8e9-8fac16ac7f5b/scratchpad/routes/_feasibility/count_rows_{'_'.join(sys.argv[1:])}.json", "w"), indent=1)

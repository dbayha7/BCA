"""READ-ONLY feasibility (commands, states, episode ids only; no reward, no next state, no critic).
(1) Are the two worlds' logs the same episodes (same start state per episode)? (2) Is wave amplitude constant within
an episode? (3) How do survivors, held-out calibration-split tail rows and removed full-throttle rows split by wave
amplitude (cut 0.25 = midpoint of the recorded law U[0.1, 0.4]) and by current (cut 0.4 = midpoint of U[0.1, 0.7])?"""
import json, sys
import numpy as np
sys.path.insert(0, "/home/user/bca")
import experiments.shift.acceptance as A  # noqa: E402
import experiments.shift.thruster_chain as TC  # noqa: E402

def gap_rule(t):
    hi = np.sort(t[t > np.quantile(t, 0.99)]); d = np.diff(hi); i = int(np.argmax(d)); return float(hi[i])

p = A.load_prereg(A.refuse_protected(A.RUNS_DEFAULT)); A.guard(p)
variant = A.gate_outcome(p)["variant"]
out = {}
for rep in [int(s) for s in sys.argv[1:]]:
    st0 = {}
    for world in ("stall", "no_stall"):
        sd = A.load_seed(p.runs, p.config, variant, world, rep); lg = sd.log; rg = sd.removals["gap"]
        ep = np.asarray(lg["infos/episode"]); tt = np.asarray(lg["infos/t"]); S = np.asarray(lg["infos/state"], np.float64)
        st0[world] = (ep[tt == 0], S[tt == 0])
        amp = TC.wave_amplitude(S); cur = S[:, TC.CURRENT]
        # amplitude constant within episode?
        order = np.lexsort((tt, ep)); e2 = ep[order]; a2 = amp[order]; c2 = cur[order]
        same = e2[1:] == e2[:-1]
        amp_jump = float(np.max(np.abs(a2[1:][same] - a2[:-1][same])))
        cur_jump = float(np.max(np.abs(c2[1:][same] - c2[:-1][same])))
        kept = rg.kept_train_rows(lg); tk = np.abs(np.asarray(lg["actions"][kept], np.float64)).max(1); t_lo = gap_rule(tk)
        surv = kept[tk > t_lo]
        bank = rg.bank_rows(lg)
        cal = lg.split_rows("calibration"); cal = cal[~sd.gap.contains(lg["actions"][cal])]; held = np.setdiff1d(cal, bank)
        th = np.abs(np.asarray(lg["actions"][held], np.float64)).max(1); ctail = held[th > t_lo]
        tr = lg.split_rows("train"); removed = tr[sd.gap.contains(lg["actions"][tr])]
        def split(rows):
            return dict(n=int(len(rows)), amp_hi=int((amp[rows] > 0.25).sum()), amp_lo=int((amp[rows] <= 0.25).sum()),
                        cur_hi=int((cur[rows] > 0.4).sum()), cur_lo=int((cur[rows] <= 0.4).sum()))
        out[f"{rep}_{world}"] = dict(t_lo=t_lo, amp_max_jump_within_episode=amp_jump, cur_max_jump_within_episode=cur_jump,
                                     survivors=split(surv), cal_tail=split(ctail), bank=split(bank),
                                     removed_train_full_throttle=split(removed[np.abs(np.asarray(lg["actions"][removed], np.float64)).max(1) > t_lo]))
        print(rep, world, out[f"{rep}_{world}"], flush=True)
    e_s, s_s = st0["stall"]; e_n, s_n = st0["no_stall"]
    o1, o2 = np.argsort(e_s), np.argsort(e_n)
    out[f"{rep}_same_episode_starts"] = bool(np.array_equal(e_s[o1], e_n[o2]) and np.array_equal(s_s[o1], s_n[o2]))
    print(rep, "same starts", out[f"{rep}_same_episode_starts"], flush=True)
json.dump(out, open(f"/tmp/claude-0/-home-user/9e341792-006d-5a11-a8e9-8fac16ac7f5b/scratchpad/routes/_where/feas_amp_{'_'.join(sys.argv[1:])}.json", "w"), indent=1)

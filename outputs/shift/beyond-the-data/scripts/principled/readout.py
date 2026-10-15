"""EXPLORATORY (outside the gate; changes no pre-registered rule). NO BCA: no scale network, no calibrator fit, no
WBCP score threshold, no dose. Stand-in signals only.

Question: what would a conformal band calibrated for the ACTOR's commands (weighted conformal prediction under covariate
shift, Tibshirani et al. 2019) say at the actor's own commands, before any score is looked at?

1. One-step residual at any (state, command): delta*(s, a) = r(s, a) + gamma * min(Q1, Q2)(s', c(s')) - Q1(s, a), with
   (r, s') from one exact simulator step and c the stall-aware controller (the controller the frozen fixed-controller
   critics score). This is the quantity BCA's band covers, at a pair the data never recorded: an oracle band.
2. The score-free part of a weighted conformal band: its abstention set. With target law "the recording controller's
   own noise around the actor's command" (oracle: the recorder's exact normal-driving law, a squashed Gaussian), the
   likelihood ratio at the actor's own command is w = exp(l), l = 0.5 * ||z||^2, z = (u(pi(s)) - u_b(s)) / 0.3 in the
   recorder's pre-squash units (infinite past 80%). Plain weighted split conformal abstains ("cannot certify") iff
   w(x) > alpha / (1 - alpha) * sum_j w_j over the calibration bank (alpha = 0.1). This needs only the weights.
3. Twin-head disagreement |Q1 - Q2| of the frozen critic (an ensemble stand-in).
4. Exact harm labels: Q^c(s, a) vs Q^c(s, clip80(a)) by exact simulation (c = stall-aware controller).

Usage: python readout.py <seed>
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy.stats import rankdata

sys.path.insert(0, "/tmp/claude-0/-home-user/9e341792-006d-5a11-a8e9-8fac16ac7f5b/scratchpad/ood/anatomy")
from common import Seed, actor_cmd, true_q, visited_states, jsonable, log  # noqa: E402
import experiments.shift.learners as L  # noqa: E402
import experiments.shift.recorder as RC  # noqa: E402
import experiments.shift.thruster_chain as TC  # noqa: E402

OUT = Path("/home/user/bca/runs/shift_explore/principled")
LABEL = ("EXPLORATORY: computed outside the gate on saved gate artifacts with stand-in signals only (oracle one-step "
         "residual, exact recorder likelihood ratio, twin-head disagreement); no BCA code was run")
ALPHA = 0.1
N = 3000
HARM_TOL = 0.02
SITUATIONS = ("recorded", "strong_current", "big_waves", "big_kicks")
ACTORS = ("gapped", "full", "clone", "td3bc_gapped")
HARM_ACTORS = ("gapped", "full", "td3bc_gapped")


def auroc(score, label):
    label = np.asarray(label, bool)
    if label.all() or not label.any():
        return None
    s = np.where(np.isfinite(score), score, 1e300)
    r = rankdata(s)
    n1, n0 = label.sum(), (~label).sum()
    return float((r[label].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


class Ratio:
    """Exact likelihood ratio of 'the recorder's normal-driving noise around a centre command' against the recorder's
    normal-driving law, both squashed by the same 0.8 tanh (Jacobians cancel)."""

    def __init__(self, recorder):
        self.K, self.spec = recorder.K, recorder.spec

    def u_b(self, x):
        fl, ex = RC._linear(np.asarray(x, np.float64), self.K)
        return -(self.spec.damping * fl + ex)

    def pre(self, a):
        a = np.asarray(a, np.float64)
        cap = self.spec.normal_driving_cap
        out = cap * np.arctanh(np.clip(a / cap, -1 + 1e-12, 1 - 1e-12))
        return np.where(np.abs(a) >= cap, np.inf, out)

    def ell_own(self, x, a):
        """log w at the actor's own command: 0.5 ||(u(a) - u_b(x)) / wobble||^2 (inf past 80%)."""
        z = (self.pre(a) - self.u_b(x)) / self.spec.wobble
        with np.errstate(invalid="ignore"):
            return 0.5 * np.sum(z * z, axis=-1)

    def log_w(self, x, a, centre):
        """log w at a recorded command a, target centred on the actor's command `centre` at x."""
        ua, uc, ub = self.pre(a), self.pre(centre), self.u_b(x)
        sg = self.spec.wobble
        with np.errstate(invalid="ignore"):
            lw = -0.5 * np.sum(((ua - uc) / sg) ** 2, -1) + 0.5 * np.sum(((ua - ub) / sg) ** 2, -1)
        bad = ~np.isfinite(ua).all(-1) | ~np.isfinite(uc).all(-1)
        return np.where(bad, -np.inf, lw)


def boundary(log_wj, alpha=ALPHA):
    """Plain weighted split conformal: abstain at x iff w(x) > alpha / (1 - alpha) * sum_j w_j. Returns log of it, the
    Kish effective sample size and the share of bank rows with zero weight."""
    m = np.max(log_wj[np.isfinite(log_wj)])
    w = np.exp(np.where(np.isfinite(log_wj), log_wj - m, -np.inf))
    tot = w.sum()
    return dict(log_threshold=float(np.log(alpha / (1 - alpha)) + m + np.log(tot)),
                kish_ess=float(tot ** 2 / np.sum(w ** 2)), zero_weight_share=float((~np.isfinite(log_wj)).mean()),
                bank_rows=int(len(log_wj)))


def residual(critic, prepared, physics, sa, x, a):
    x1, r = TC.dynamics(x, a, physics)
    a1 = sa(x1)
    q1 = L.critic_rows(critic, prepared.normalize(x1), np.asarray(a1, np.float32))[0]
    q0 = L.critic_rows(critic, prepared.normalize(x), np.asarray(a, np.float32))[0]
    y = r + physics.gamma * q1.min(1)
    return y - q0[:, 0], np.abs(q0[:, 0] - q0[:, 1]), q0[:, 0]


def main(rep):
    t0 = time.perf_counter()
    rng = np.random.default_rng([20261013, int(rep), 51])
    S = Seed(rep, worlds=("stall",))
    pg, ph, sd = S.pg, S.physics, S.st
    sa = RC.stall_aware(ph)
    ratio = Ratio(S.recorder)
    critics = {"gapped": S.h2_critic("gapped"), "full": S.h2_critic("full")}
    acts, _ = S.h2_actors()
    pols = {k: L.mlp_policy(acts[k], pg, k) for k in ACTORS}
    log(f"seed {rep}: loaded ({time.perf_counter() - t0:.0f}s)")

    lg = sd.log
    bank = pg.bank_rows
    xb = np.asarray(lg["infos/state"][bank], np.float64)
    ab = np.asarray(lg["actions"][bank], np.float64)
    cal = lg.split_rows("calibration")
    cal = cal[~sd.gap.contains(lg["actions"][cal])]
    cal = np.sort(rng.choice(cal, N, replace=False))
    xh = np.asarray(lg["infos/state"][cal], np.float64)
    ah = np.asarray(lg["actions"][cal], np.float64)
    tr = np.sort(rng.choice(pg.n, N, replace=False))
    xt = pg.states[tr]
    region_t = pg.region[tr]

    res = dict(seed=int(rep), alpha=ALPHA, n=N, harm_tolerance=HARM_TOL, bank_rows=int(len(bank)),
               bank_rows_past_80=int(TC.past_knee(ab).sum()),
               training_rows_past_80=int(TC.past_knee(pg.actions).sum()), training_rows=int(pg.n))
    # boundaries per actor (bank weights depend on the actor's command at each bank state)
    bnd = {}
    for k in ACTORS:
        cb = actor_cmd(acts[k], pg, xb)
        bnd[k] = boundary(ratio.log_w(xb, ab, cb))
        bnd[k]["actor_past_80_share_at_bank_states"] = float(TC.past_knee(cb).mean())
    res["boundaries"] = bnd
    # held-out recorded pairs: how often the actor's boundary flags a recorded pair (its own law's false alarms are
    # judged on target-law draws below)
    res["heldout_recorded"] = {}
    for k in ACTORS:
        lw = ratio.log_w(xh, ah, actor_cmd(acts[k], pg, xh))
        res["heldout_recorded"][k] = dict(abstain_share=float((lw > bnd[k]["log_threshold"]).mean()))
    # target-law draws at held-out states: validity of the abstention rate (should be about alpha or less)
    res["target_law_draws"] = {}
    for k in ACTORS:
        c = actor_cmd(acts[k], pg, xh)
        uc = ratio.pre(c)
        ok = np.isfinite(uc).all(-1)
        z = rng.standard_normal(uc.shape)
        cap = S.recorder.spec.normal_driving_cap
        draw = cap * np.tanh((uc + S.recorder.spec.wobble * z) / cap)
        lw = ratio.log_w(xh[ok], draw[ok], c[ok])
        res["target_law_draws"][k] = dict(defined_share=float(ok.mean()),
                                          abstain_share=float((lw > bnd[k]["log_threshold"]).mean()))
    res["pools"] = {}

    def score_pool(name, x, k, region=None):
        a = actor_cmd(acts[k], pg, x)
        ell = ratio.ell_own(x, a)
        abst = ell > bnd[k]["log_threshold"]
        past = TC.past_knee(a)
        hole = sd.gap.contains(np.asarray(a, np.float32))  # inside the removed ball: no calibration support either
        abst_h = abst | hole
        d_g, dis, q_g = residual(critics["gapped"], pg, ph, sa, x, a)
        d_f, _, _ = residual(critics["full"], pg, ph, sa, x, a)
        rec = dict(n=int(len(x)), past_80_share=float(past.mean()), abstain_share=float(abst.mean()),
                   abstain_share_among_past_80=float(abst[past].mean()) if past.any() else None,
                   abstain_share_among_not_past_80=float(abst[~past].mean()) if (~past).any() else None,
                   residual_gapped_mean_past_80=float(d_g[past].mean()) if past.any() else None,
                   residual_gapped_mean_not_past_80=float(d_g[~past].mean()) if (~past).any() else None,
                   residual_full_mean_past_80=float(d_f[past].mean()) if past.any() else None,
                   abs_residual_gapped_median_not_past_80=float(np.median(np.abs(d_g[~past]))) if (~past).any() else None,
                   twin_gap_mean_past_80=float(dis[past].mean()) if past.any() else None,
                   twin_gap_mean_not_past_80=float(dis[~past].mean()) if (~past).any() else None,
                   auroc_past80_by_ell=auroc(ell, past), auroc_past80_by_abs_residual=auroc(np.abs(d_g), past),
                   auroc_past80_by_twin=auroc(dis, past), in_removed_ball_share=float(hole.mean()),
                   in_removed_ball_not_past_80_share=float((hole & ~past).mean()),
                   abstain_share_with_hole=float(abst_h.mean()),
                   residual_gapped_mean_in_ball_not_past_80=float(d_g[hole & ~past].mean()) if (hole & ~past).any() else None,
                   abs_residual_gapped_median_in_ball_not_past_80=(float(np.median(np.abs(d_g[hole & ~past])))
                                                                  if (hole & ~past).any() else None))
        if region is not None:
            rec["region_share"] = float(region.mean())
            rec["past_80_share_region"] = float(past[region].mean())
            rec["abstain_share_region"] = float(abst[region].mean())
        if k in HARM_ACTORS:
            ac = np.clip(a, -TC.KNEE, TC.KNEE)
            qa = true_q(sa, x, a, ph)
            qc = true_q(sa, x, ac, ph)
            harm = (qc - qa) > HARM_TOL
            qg_c = L.critic_rows(critics["gapped"], pg.normalize(x), np.asarray(ac, np.float32))[0][:, 0]
            driven = harm & ((q_g - qg_c) > 0)
            rec.update(harmful_share=float(harm.mean()), critic_driven_harm_share=float(driven.mean()),
                       harm_mean_given_harmful=float((qc - qa)[harm].mean()) if harm.any() else None,
                       catch_harmful_by_abstention=float(abst[harm].mean()) if harm.any() else None,
                       abstain_on_harmless=float(abst[~harm].mean()) if (~harm).any() else None,
                       auroc_harm_by_ell=auroc(ell, harm), auroc_harm_by_abs_residual=auroc(np.abs(d_g), harm),
                       auroc_harm_by_twin=auroc(dis, harm),
                       abstain_on_harmless_with_hole=float(abst_h[~harm].mean()) if (~harm).any() else None,
                       in_ball_share_of_harmless=float(hole[~harm].mean()) if (~harm).any() else None,
                       residual_gapped_mean_harmful=float(d_g[harm].mean()) if harm.any() else None,
                       true_overrating_mean_harmful=float((q_g - qa)[harm].mean()) if harm.any() else None)
        res["pools"][name] = rec
        log(f"  {name}: past80 {rec['past_80_share']:.3f} abstain {rec['abstain_share']:.3f} "
            f"(past80 {rec['abstain_share_among_past_80']}, other {rec['abstain_share_among_not_past_80']}) "
            f"resid past80 {rec['residual_gapped_mean_past_80']} other {rec['residual_gapped_mean_not_past_80']} "
            f"harm {rec.get('harmful_share')} catch {rec.get('catch_harmful_by_abstention')} "
            f"auroc ell/res/twin {rec.get('auroc_harm_by_ell')}/{rec.get('auroc_harm_by_abs_residual')}/"
            f"{rec.get('auroc_harm_by_twin')}  ({time.perf_counter() - t0:.0f}s)")

    for k in ACTORS:
        score_pool(f"train|{k}", xt, k, region_t)
    for s in SITUATIONS:
        for k in ACTORS:
            x, _ = visited_states(pols[k], s, ph, 200, take=N, rng=rng)
            score_pool(f"{s}|{k}", x, k)
    res["seconds"] = time.perf_counter() - t0
    OUT.mkdir(parents=True, exist_ok=True)
    p = OUT / f"readout_seed{rep}.json"
    p.write_text(json.dumps(jsonable(dict(label=LABEL, **res)), indent=1, sort_keys=True))
    log(f"wrote {p}")


if __name__ == "__main__":
    main(int(sys.argv[1]))

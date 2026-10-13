"""EXPLORATORY (outside the gate, no BCA). Part 3: why extra pull toward the recorded commands on training rows
(check 3's caution) does not reach the strong-current harm.

The harm-placed doses of check 3 are recomputed with the check's own oracle (learners.over_preference_rows on the
frozen plain TD3+BC critic of the gapped log, horizon 600, float32 fast path, learners.harm_doses) for
  - the cloned actor (the doses of the first refresh, step 0; checked against the saved dose summary), and
  - the final harm-placed and no-caution actors (close to the last refresh, step 16,000; the per-row doses of the
    run itself were not saved).
These are exact-harm (oracle) doses, not BCA doses.

Usage: python caution_anatomy.py <seed>
"""
import sys
import time

import numpy as np

from common import *  # noqa: F401,F403

ARMS = ("harm_placed", "strength_matched_even", "amount_matched", "shuffled_0", "shuffled_1", "reversed", "cap2",
        "even_x1.1", "even_x1.25", "even_x1.5", "even_x2", "even_x3")
C_BINS = ((0.1, 0.3), (0.3, 0.5), (0.5, 0.7001))
PUSH_BINS = ((-1, 0.3), (0.3, 0.565), (0.565, 0.8), (0.8, 2.0))


def dose_where(pg, D, doses, pi_none, pi_arm):
    reg = pg.region
    c = pg.states[:, TC.CURRENT]
    p = TC.push(pg.states)
    th_rec = -pg.actions[:, 0].astype(np.float64)
    ex = doses - 1.0
    top = doses >= np.quantile(doses, 0.9)
    past_none = TC.past_knee(pi_none, TC.STALL)
    out = dict(
        rows=int(pg.n), region_share_rows=float(reg.mean()),
        dose_mean=float(doses.mean()), dose_sd=float(doses.std()), dose_max=float(doses.max()),
        share_rows_dose_at_floor=float((doses <= 1.0 + 1e-9).mean()), share_rows_D_zero=float((D <= 0).mean()),
        region_share_of_extra_dose=float(ex[reg].sum() / max(ex.sum(), 1e-12)),
        mean_dose_region=float(doses[reg].mean()), mean_dose_outside=float(doses[~reg].mean()),
        by_current={f"{lo:.1f}-{min(hi, 0.7):.1f}": dict(share_rows=float(((c >= lo) & (c < hi)).mean()),
                                                         mean_dose=float(doses[(c >= lo) & (c < hi)].mean()),
                                                         share_extra=float(ex[(c >= lo) & (c < hi)].sum() / ex.sum()))
                    for lo, hi in C_BINS},
        by_push={f"{lo:g}-{hi:g}": dict(share_rows=float(((p >= lo) & (p < hi)).mean()),
                                        mean_dose=float(doses[(p >= lo) & (p < hi)].mean()) if ((p >= lo) & (p < hi)).any() else None,
                                        share_extra=float(ex[(p >= lo) & (p < hi)].sum() / ex.sum()))
                 for lo, hi in PUSH_BINS},
        top10=dict(region_share=float(reg[top].mean()), current_mean=float(c[top].mean()),
                   push_mean=float(p[top].mean()),
                   recorded_throttle1_mean=float(th_rec[top].mean()),
                   recorded_throttle1_share_ge_0p75=float((np.abs(th_rec[top]) >= 0.75).mean()),
                   none_actor_throttle1_mean=float(-pi_none[top, 0].mean()),
                   arm_actor_throttle1_mean=float(-pi_arm[top, 0].mean()),
                   none_actor_share_past80=float(past_none[top].mean())),
        region=dict(recorded_throttle1_mean=float(th_rec[reg].mean()),
                    recorded_throttle1_share_ge_0p75=float((np.abs(th_rec[reg]) >= 0.75).mean()),
                    recorded_throttle1_share_past80=float((np.abs(th_rec[reg]) > TC.KNEE).mean()),
                    recorded_any_past80=float(TC.past_knee(pg.actions[reg], TC.STALL).mean()),
                    none_actor_throttle1_mean=float(-pi_none[reg, 0].mean()),
                    none_actor_share_past80=float(past_none[reg].mean()),
                    arm_actor_throttle1_mean=float(-pi_arm[reg, 0].mean())),
        all_rows=dict(recorded_throttle1_share_ge_0p75=float((np.abs(th_rec) >= 0.75).mean()),
                      none_actor_share_past80=float(past_none.mean())),
        share_D_mass_on_rows_where_actor_past80=float(D[past_none].sum() / max(D.sum(), 1e-12)),
        corr_dose_push=float(np.corrcoef(doses, p)[0, 1]), corr_dose_current=float(np.corrcoef(doses, c)[0, 1]))
    return out


def change(pa, pb):
    d = pa - pb
    return dict(rms=float(np.sqrt(np.mean(np.sum(d ** 2, 1)))), throttle1=float(-(d[:, 0]).mean()),
                share_past80_arm=float(TC.past_knee(pa, TC.STALL).mean()),
                share_past80_none=float(TC.past_knee(pb, TC.STALL).mean()))


def main(rep):
    t0 = time.perf_counter()
    S = Seed(rep)
    pg = S.pg
    host = S.host("gap")
    critic = host.critic.params
    room, blind, vals = S.room_actors()
    h2, _ = S.h2_actors()
    clone = h2["clone"]
    learn = S.learn
    q_logged = L.critic_rows(critic, pg.train.obs, pg.actions)[0][:, 0]
    res = dict(points={s: v["points"] for s, v in vals["situations"].items()},
               saved_dose_history_first=blind["dose_history"][0]["harm_placed"],
               saved_dose_history_last=blind["dose_history"][-1]["harm_placed"])
    obs = np.asarray(pg.train.obs)
    pi = {k: L.actor_rows(room[k], obs).astype(np.float64) for k in ("none",) + ARMS if k in room}
    pi["clone"] = L.actor_rows(clone, obs).astype(np.float64)
    doses = {}
    for who, params in (("clone_step0", clone), ("harm_placed_final", room["harm_placed"]),
                        ("none_final", room["none"])):
        t = time.perf_counter()
        D = L.over_preference_rows(params, critic, pg, horizon=learn.oracle_horizon, x64=learn.oracle_x64,
                                   q_logged=q_logged, chunk=learn.row_chunk)["D"]
        dz = L.harm_doses(D, "harm")
        doses[who] = dz
        res[f"doses_{who}"] = dose_where(pg, D, dz, pi["none"], pi["harm_placed"])
        log(f"seed {rep}: doses at {who}: mean {dz.mean():.4f} sd {dz.std():.4f} max {dz.max():.4f}; region share of "
            f"extra {res[f'doses_{who}']['region_share_of_extra_dose']:.3f} (rows {pg.region.mean():.3f}) "
            f"({time.perf_counter() - t:.0f}s)")
    log(f"saved first refresh: {res['saved_dose_history_first']}")
    top = doses["harm_placed_final"] >= np.quantile(doses["harm_placed_final"], 0.9)
    # actor change of every arm against no caution, at training and at strong-current states
    ph = S.physics
    none_pol = L.mlp_policy(room["none"], pg)
    sv_none = visited_states(none_pol, "strong_current", ph, 200)[0]
    sv_sa = visited_states(RC.stall_aware(ph), "strong_current", ph, 200)[0]
    rv_none = visited_states(none_pol, "recorded", ph, 200)[0]
    sets = dict(train_all=obs, train_region=obs[pg.region], train_top10_dose=obs[top],
                eval_split=np.asarray(pg.eval_obs), recorded_visit_none=pg.normalize(rv_none),
                strong_visit_none=pg.normalize(sv_none), strong_visit_stall_aware=pg.normalize(sv_sa))
    base = {k: L.actor_rows(room["none"], o).astype(np.float64) for k, o in sets.items()}
    ch = {}
    for arm in ARMS + ("clone",):
        params = clone if arm == "clone" else room.get(arm)
        if params is None:
            continue
        ch[arm] = {k: change(L.actor_rows(params, o).astype(np.float64), base[k]) for k, o in sets.items()}
        r = ch[arm]
        ch[arm]["ratio_strong_to_train_rms"] = r["strong_visit_none"]["rms"] / max(r["train_all"]["rms"], 1e-12)
        log(f"  {arm:22s} rms change train {r['train_all']['rms']:.4f} region {r['train_region']['rms']:.4f} "
            f"top10 {r['train_top10_dose']['rms']:.4f} strong {r['strong_visit_none']['rms']:.4f}; thr1 change train "
            f"{r['train_region']['throttle1']:+.4f} strong {r['strong_visit_none']['throttle1']:+.4f}; past80 strong "
            f"{r['strong_visit_none']['share_past80_arm']:.3f} vs none {r['strong_visit_none']['share_past80_none']:.3f}")
    res["actor_change_vs_none"] = ch
    res["strengths_saved"] = blind["strengths"]
    write(f"caution_anatomy_seed{rep}.json", dict(kind="anatomy-caution", rep=int(rep),
                                                   seconds=time.perf_counter() - t0, **res))


if __name__ == "__main__":
    main(int(sys.argv[1]))

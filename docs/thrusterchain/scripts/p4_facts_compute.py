"""Facts for page pass 4 (read only: no page edit, nothing written under /root/bca-shift).

Run from /mnt/c/Users/David/dev/bca-shift-work/page with
  PYTHONPATH=/root/bca-shift /opt/bca-shift-venv/bin/python scripts/p4_facts_compute.py
Writes tmp/p4/facts/compute.json.

Parts
  A  main test, design seed 900: regenerate the log (dataset.generate, the pipeline's own generator), the densest-15%
     removal (shift.define_gap / make_removal), counts, thruster histograms (50 bins on [-1, 1]) and percentiles.
  B  weighting test, version 5: every quality and seed, training split, thruster-1 commands past 80%, the random flag,
     the 80% hold, the removed region (truth/in_ball); expert seed 0 histograms.
  C  evaluation banks: rollouts of the reference controllers in each situation (stalling world), observation
     percentiles, water-push shares, anchors and points.
  D  rewards worked example; E thrust and cost tables.
"""

import json
import math
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, "/root/bca-shift")

from experiments.shift import dataset as DS  # noqa: E402
from experiments.shift import recorder as RC  # noqa: E402
from experiments.shift import shift as SH  # noqa: E402
from experiments.shift import thruster_chain as TC  # noqa: E402
from experiments.shift import truth as TR  # noqa: E402

OUT = Path("/mnt/c/Users/David/dev/bca-shift-work/page/tmp/p4/facts/compute.json")
IW = Path("/root/bca-shift/runs/shift_iw_v5")
QUALITIES = ("random", "medium-replay", "medium", "medium-expert", "expert")
EDGES = np.linspace(-1.0, 1.0, 51)
PCTS = (1, 5, 10, 25, 50, 75, 90, 95, 99)
KNEE = TC.KNEE
OBS_NAMES = ["position of float 1", "speed of float 1", "position of float 2", "speed of float 2",
             "position of float 3", "speed of float 3", "position of float 4", "speed of float 4",
             "position of float 5", "speed of float 5", "current c", "wave w1 (push the wave adds now)",
             "wave w2 (the wave's push a quarter cycle earlier)"]
T0 = time.time()


def log(msg):
    print(f"[{time.time() - T0:7.1f}s] {msg}", flush=True)


def hist(a):
    counts, _ = np.histogram(np.asarray(a, np.float64), bins=EDGES)
    return [int(c) for c in counts]


def pct(a, ps=PCTS):
    a = np.asarray(a, np.float64)
    return {f"p{p}": float(np.percentile(a, p)) for p in ps}


def r4(x):
    return float(round(float(x), 6))


out = {"about": __doc__, "bin_edges": [float(e) for e in EDGES], "knee": KNEE}

# ------------------------------------------------------------------------------------------------ A: main test
log("A: generating design seed 900 (dataset.generate, stalling world)")
lg = DS.generate(900, "stall")
train = lg.split_rows("train")
A = np.asarray(lg["actions"], np.float32)
X = np.asarray(lg["infos/state"], np.float64)
ft = lg.diagnostic("infos/full_throttle_command")
log("A: defining the gap")
gap, record = SH.define_gap(lg)
rem = SH.make_removal(lg, "gap", gap)
rnd = SH.make_removal(lg, "random", gap)
kept = rem.kept_train_rows(lg)
removed = rem.removed_train_rows(lg)
rkept = rnd.kept_train_rows(lg)
a_tr, a_kept, a_rem = A[train].astype(np.float64), A[kept].astype(np.float64), A[removed].astype(np.float64)
past_any = lambda a: np.any(np.abs(a) > KNEE, axis=1)
region = TC.push(X) > RC.TRIGGER
ep = lg["infos/episode"]
J_train = np.asarray(lg["rewards"], np.float64)[train].reshape(500, 100).sum(1)
ep2 = np.flatnonzero(ep == 2)
in_ball_all = gap.contains(A)
capped_pattern = RC.full_throttle_pattern(RC.textbook_gain()).copy()
capped_pattern_t1 = capped_pattern.copy()
capped_pattern_t1[0] = -0.8
main = {
    "source": "experiments.shift.dataset.generate(900, 'stall'); shift.define_gap; shift.make_removal (regenerated "
              "now, bit-identical generator; checked against the page's seed-900 numbers below)",
    "rows": {"all": int(lg.n_rows), "train": int(len(train)), "kept": int(len(kept)), "removed": int(len(removed)),
             "bank_rows": int(len(lg.bank_rows)), "bank_kept": int((~rem.bank_removed).sum())},
    "gap": {"centre": [float(c) for c in gap.centre], "rho": float(gap.rho), "mass": float(gap.mass),
            "competing_ratio": float(gap.competing_ratio), "rule": gap.rule},
    "full_throttle_pattern": [float(v) for v in capped_pattern],
    "full_throttle_pattern_distance_to_centre": float(np.linalg.norm(capped_pattern - np.asarray(gap.centre))),
    "capped_pattern_thruster1_at_80": [float(v) for v in capped_pattern_t1],
    "capped_pattern_t1_distance_to_centre": float(np.linalg.norm(capped_pattern_t1 - np.asarray(gap.centre))),
    "capped_pattern_scaled_0p8": [float(v) for v in 0.8 * capped_pattern],
    "capped_pattern_scaled_distance_to_centre": float(np.linalg.norm(0.8 * capped_pattern - np.asarray(gap.centre))),
    "ball_projection_per_thruster": [[float(c - gap.rho), float(c + gap.rho)] for c in gap.centre],
    "train": {
        "full_throttle_share": float(ft[train].mean()),
        "past80_any_share": float(past_any(a_tr).mean()),
        "past80_any_count": int(past_any(a_tr).sum()),
        "past80_equals_full_throttle": bool(np.array_equal(past_any(a_tr), ft[train])),
        "region_share": float(region[train].mean()),
        "full_throttle_share_in_region": float(ft[train][region[train]].mean()),
        "mean_return_train_episodes": float(J_train.mean()),
    },
    "kept": {
        "past80_any_count": int(past_any(a_kept).sum()),
        "past80_any_share": float(past_any(a_kept).mean()),
        "t1_past80_count": int((np.abs(a_kept[:, 0]) > KNEE).sum()),
        "t1_below_minus80_count": int((a_kept[:, 0] < -KNEE).sum()),
        "t1_above_plus80_count": int((a_kept[:, 0] > KNEE).sum()),
        "past80_rows_full_throttle_flag": int(ft[kept][past_any(a_kept)].sum()),
        "region_rows_kept_share": float(np.isin(np.flatnonzero(region), kept).sum()
                                        / np.isin(np.flatnonzero(region), train).sum()),
        "identical_rule_share": float(1 - past_any(a_kept).mean()),
    },
    "removed": {
        "full_throttle_removed_share": float(ft[removed].sum() / ft[train].sum()),
        "removed_that_are_full_throttle": float(ft[removed].mean()),
        "removed_normal_driving_count": int((~ft[removed]).sum()),
        "past80_surviving_share": float(past_any(a_kept).sum() / past_any(a_tr).sum()),
    },
    "random_removal": {
        "kept": int(len(rkept)),
        "past80_surviving_share": float(past_any(A[rkept].astype(np.float64)).sum() / past_any(a_tr).sum()),
        "full_throttle_removed_share": float(ft[rnd.removed_train_rows(lg)].sum() / ft[train].sum()),
    },
    "episode2": {
        "rows": [int(ep2[0]), int(ep2[-1])],
        "return": float(np.asarray(lg["rewards"], np.float64)[ep2].sum()),
        "t0_actions": [float(v) for v in A[ep2[0]]],
        "full_throttle_steps": int(ft[ep2].sum()),
        "deleted_rows": int(rem.removed[ep2].sum()),
        "deleted_steps": [int(t) for t in np.flatnonzero(rem.removed[ep2])],
        "deleted_normal_driving_steps": [int(t) for t in np.flatnonzero(rem.removed[ep2] & ~ft[ep2])],
        "region_steps": int(region[ep2].sum()),
    },
    "histograms": {}, "percentiles": {},
}
for k in range(4):
    main["histograms"][f"thruster{k + 1}"] = {
        "full_train": hist(a_tr[:, k]), "kept_after_densest15": hist(a_kept[:, k]), "removed": hist(a_rem[:, k])}
    main["percentiles"][f"thruster{k + 1}"] = {
        "full_train": pct(a_tr[:, k]), "kept_after_densest15": pct(a_kept[:, k]), "removed": pct(a_rem[:, k]),
        "share_below_minus80_full": float((a_tr[:, k] < -KNEE).mean()),
        "share_above_plus80_full": float((a_tr[:, k] > KNEE).mean()),
        "share_below_minus80_kept": float((a_kept[:, k] < -KNEE).mean()),
        "share_above_plus80_kept": float((a_kept[:, k] > KNEE).mean()),
        "min_full": float(a_tr[:, k].min()), "max_full": float(a_tr[:, k].max()),
        "min_kept": float(a_kept[:, k].min()), "max_kept": float(a_kept[:, k].max())}
# thruster 1, finer split: exact counts in the stall zone
t1 = a_tr[:, 0]
main["t1_detail"] = {
    "count_exactly_minus1_full": int((t1 == -1.0).sum()),
    "count_below_minus80_full": int((t1 < -KNEE).sum()),
    "count_below_minus80_kept": int((a_kept[:, 0] < -KNEE).sum()),
    "count_in_minus80_to_minus60_full": int(((t1 >= -KNEE) & (t1 < -0.6)).sum()),
    "count_in_minus80_to_minus60_kept": int(((a_kept[:, 0] >= -KNEE) & (a_kept[:, 0] < -0.6)).sum()),
    "full_throttle_t1_percentiles": pct(t1[ft[train]]),
    "normal_driving_t1_percentiles": pct(t1[~ft[train]]),
    "normal_driving_t1_min_max": [float(t1[~ft[train]].min()), float(t1[~ft[train]].max())],
    "hist_full_throttle_rows": hist(t1[ft[train]]),
    "hist_normal_driving_rows": hist(t1[~ft[train]]),
}
out["main_test_seed900"] = main
log(f"A done: kept {len(kept)}, kept past 80% {main['kept']['past80_any_count']}")

# ------------------------------------------------------------------------------------------------ B: weighting test
import h5py  # noqa: E402

iw = {"source": "runs/shift_iw_v5/<quality>/data/seed<k>/train_law.hdf5 (training split infos/split == 0; "
                "removed region truth/in_ball; random flag infos/random; hold flag infos/hold)", "by_quality": {}}
for q in QUALITIES:
    rows = []
    for s in range(5):
        p = IW / q / "data" / f"seed{s}" / "train_law.hdf5"
        with h5py.File(p, "r") as f:
            a = f["actions"][()].astype(np.float64)
            split = f["infos/split"][()]
            rnd_flag = f["infos/random"][()]
            hold = f["infos/hold"][()]
            inb = f["truth/in_ball"][()]
        tr = split == 0
        a1 = a[tr, 0]
        kp = tr & ~inb
        past_pos = a1 > KNEE
        past_neg = a1 < -KNEE
        past_abs = np.abs(a1) > KNEE
        anyp = np.any(np.abs(a[tr]) > KNEE, axis=1)
        row = dict(seed=s, train_rows=int(tr.sum()), kept_rows=int(kp.sum()), removed_rows=int((tr & inb).sum()),
                   t1_above_plus80_share=float(past_pos.mean()), t1_below_minus80_share=float(past_neg.mean()),
                   t1_past80_share=float(past_abs.mean()), any_past80_share=float(anyp.mean()),
                   t1_past80_random_share=float(rnd_flag[tr][past_abs].mean()) if past_abs.any() else None,
                   any_past80_random_share=float(rnd_flag[tr][anyp].mean()) if anyp.any() else None,
                   t1_past80_kept_share=float((np.abs(a[kp, 0]) > KNEE).mean()),
                   any_past80_kept_share=float(np.any(np.abs(a[kp]) > KNEE, axis=1).mean()),
                   random_share=float(rnd_flag[tr].mean()), hold_share=float(hold[tr].mean()),
                   hold_t1_values=sorted({float(v) for v in a[tr & hold, 0]})[:5],
                   hold_removed_share=float(inb[tr & hold].mean()) if (tr & hold).any() else None)
        rows.append(row)
        if q == "expert" and s == 0:
            iw["expert_seed0"] = {
                "t1_hist_all_train": hist(a1), "t1_hist_hold_removed": hist(a[kp, 0]),
                "t1_hist_removed": hist(a[tr & inb, 0]), "t1_hist_hold_rows": hist(a[tr & hold, 0]),
                "t1_percentiles_all_train": pct(a1), "t1_percentiles_hold_removed": pct(a[kp, 0]),
                "per_thruster_percentiles_all_train": {f"thruster{k + 1}": pct(a[tr, k]) for k in range(4)},
                "per_thruster_percentiles_hold_removed": {f"thruster{k + 1}": pct(a[kp, k]) for k in range(4)},
                "t1_min_max_all_train": [float(a1.min()), float(a1.max())]}
    iw["by_quality"][q] = rows
    sh = [r["t1_past80_share"] for r in rows]
    log(f"B {q}: t1 past 80% {min(sh):.4f}-{max(sh):.4f}")
out["weighting_test_v5"] = iw

# ------------------------------------------------------------------------------------------------ C: banks
ctrls = {"recording": None, "stall_aware": RC.stall_aware(), "rising_thrust": RC.rising_thrust(),
         "normal_driving_only": RC.hand_only()}
an = json.load(open("/root/bca-shift/runs/shift/pre/stage0_base.json"))["record"]["anchors"]
banks = {"source": "experiments.shift.truth.bank_rollout on thruster_chain.bank_starts(situation) (1,000 starts x "
                   "100 steps, stalling world); the recording controller with its fixed bank streams "
                   "(truth.recorder_on_bank); states = the 100 states at which a command is given (t = 0..99)",
         "anchors_file": {s: an[s] for s in TC.SITUATIONS}, "by_situation": {}}
for sit in TC.SITUATIONS:
    res = {}
    for name, pol in ctrls.items():
        pol = TR.recorder_on_bank() if name == "recording" else pol
        run = TR.bank_rollout(pol, sit, record=True)
        S = run.states[:, :-1].reshape(-1, 13)
        C = run.commands.reshape(-1, 4)
        push = TC.push(S)
        d = {"mean_return": float(run.J.mean()), "points": float(TR.points(run.J.mean(), an[sit])),
             "obs_percentiles": {OBS_NAMES[i]: pct(S[:, i], (5, 25, 50, 75, 95)) for i in range(13)},
             "obs_min_max": {OBS_NAMES[i]: [float(S[:, i].min()), float(S[:, i].max())] for i in range(13)},
             "float_radius_percentiles": pct(TC.float_radius(S), (5, 50, 95)),
             "past80_any_share": float(np.any(np.abs(C) > KNEE, axis=1).mean()),
             "t1_at_minus80_share": float((np.abs(C[:, 0] + 0.8) < 1e-12).mean()),
             "any_at_80_limit_share": float(np.any(np.abs(np.abs(C) - 0.8) < 1e-12, axis=1).mean()),
             "command_percentiles": {f"thruster{k + 1}": pct(C[:, k], (5, 25, 50, 75, 95)) for k in range(4)},
             "command_in_ball_share": float(gap.contains(C).mean())}
        if name == "stall_aware":
            d["push"] = {"share_above_trigger": float((push > RC.TRIGGER).mean()),
                         "share_upstream_push_below_0": float((push < 0).mean()),
                         "min": float(push.min()), "max": float(push.max()),
                         "percentiles": pct(push, (5, 50, 95)),
                         "wave_amplitude_min_max": [float(TC.wave_amplitude(S).min()),
                                                    float(TC.wave_amplitude(S).max())]}
        res[name] = d
    banks["by_situation"][sit] = res
    log(f"C {sit}: recording {res['recording']['mean_return']:.4f} stall-aware {res['stall_aware']['mean_return']:.4f}"
        f" rising {res['rising_thrust']['points']:.2f} pts")
# training data observation percentiles (seed 900, training split) for the recorded conditions
banks["seed900_train_obs_percentiles"] = {OBS_NAMES[i]: pct(X[train][:, i], (5, 25, 50, 75, 95)) for i in range(13)}
banks["seed900_kept_obs_percentiles"] = {OBS_NAMES[i]: pct(X[kept][:, i], (5, 25, 50, 75, 95)) for i in range(13)}
out["banks"] = banks

# ------------------------------------------------------------------------------------------------ D: rewards example
K = RC.textbook_gain()
sa = RC.stall_aware()
rt = RC.rising_thrust()


def example(x, label):
    x = np.asarray(x, np.float64)
    base = sa(x[None])[0]
    rows = []
    for mag in (0.8, 0.9, 1.0):
        a = base.copy()
        a[0] = -mag
        for world, phys in (("stall", TC.STALL), ("no_stall", TC.NO_STALL)):
            top = sa if world == "stall" else rt
            x1, r0 = TC.dynamics(x[None], a[None], phys)
            x2, r1 = TC.dynamics(x1, top(x1), phys)
            q_disc = TR.q_value(top, x[None], a[None], phys)[0]
            # 100-step return from this state with the top controller continuing (finite horizon)
            xx, tot = x1.copy(), float(r0[0])
            for _ in range(99):
                xx, rr = TC.dynamics(xx, top(xx), phys)
                tot += float(rr[0])
            rows.append(dict(world=world, thruster1_command=float(-mag), throttle_pct=int(round(mag * 100)),
                             push_t1=float(TC.thrust(np.array([-mag]), phys)[0]),
                             throttle_cost_t1=float(0.3 * mag * mag),
                             throttle_cost_all=float(0.3 * np.sum(a * a)),
                             float_cost_now=float(np.sum(x[:10] ** 2)),
                             reward_now=float(r0[0]),
                             float_cost_next=float(np.sum(x1[0, :10] ** 2)),
                             reward_next_with_top_controller=float(r1[0]),
                             two_step_reward=float(r0[0] + r1[0]),
                             q_discounted_1000=float(q_disc),
                             return_100_steps_top_continues=float(tot),
                             next_state_floats=[float(v) for v in x1[0, :10]]))
    return dict(label=label, state=[float(v) for v in x], push_water=float(TC.push(x)),
                stall_aware_command=[float(v) for v in base], unclipped_textbook=[float(v) for v in -(K @ x)],
                rows=rows)


x_rest = np.zeros(13)
x_rest[TC.CURRENT] = 0.9
rec_row = int(ep2[1])  # seed 900 episode 2, t = 1: the recording controller gave full throttle here
out["rewards_example"] = {
    "note": "thruster 1 set to -0.8, -0.9, -1.0 (upstream); thrusters 2-4 at the stall-aware controller's command; "
            "top controller continues (stall-aware in the stalling world, rising-thrust in the no-stall world)",
    "rest_strong_current": example(x_rest, "floats at rest, current 0.9, no wave"),
    "recorded_row_ep2_t1": example(X[rec_row], f"design seed 900, row {rec_row} (episode 2, t = 1)"),
    "recorded_row_ep2_t1_logged_command": [float(v) for v in A[rec_row]],
}
out["points_per_return_unit"] = {s: {"return_units_per_point": (an[s]["top"] - an[s]["recording"]) / 100,
                                     "recording": an[s]["recording"], "top": an[s]["top"]} for s in TC.SITUATIONS}

# ------------------------------------------------------------------------------------------------ E: thrust and cost
grid = np.round(np.linspace(0, 1, 21), 2)
out["thrust_table"] = [dict(throttle=float(g), push_stall=float(TC.thrust(np.array([g]), TC.STALL)[0]),
                            push_no_stall=float(TC.thrust(np.array([g]), TC.NO_STALL)[0]),
                            cost=float(0.3 * g * g),
                            push_stall_float32_cmd=float(TC.thrust(np.array([np.float32(g)], np.float64),
                                                                   TC.STALL)[0]))
                       for g in grid]
out["K_current_column"] = [float(v) for v in K[:, TC.CURRENT]]
out["K_wave_columns"] = [[float(v) for v in K[:, TC.WAVE1]], [float(v) for v in K[:, TC.WAVE2]]]
OUT.write_text(json.dumps(out, indent=1))
log(f"wrote {OUT}")

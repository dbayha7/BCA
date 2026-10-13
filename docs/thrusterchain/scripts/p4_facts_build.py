"""Assemble notes/p4_facts.json for page pass 4 (read only on the data; writes only the notes file).

Run from /mnt/c/Users/David/dev/bca-shift-work/page with
  PYTHONPATH=/root/bca-shift /opt/bca-shift-venv/bin/python scripts/p4_facts_build.py
after scripts/p4_facts_compute.py and scripts/p4_facts_extra.py (it reads their tmp/p4/facts/*.json).

Histograms here are knee-aware: the edges at +-80% are the float32 knee 0.800000011920929 (TC.KNEE), and a command of
exactly 80% (the float32 of 0.8, which equals the knee) is counted on the at-or-below-80% side, as the environment
treats it (it does not stall).
"""

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, "/root/bca-shift")

from experiments.shift import dataset as DS  # noqa: E402
from experiments.shift import recorder as RC  # noqa: E402
from experiments.shift import shift as SH  # noqa: E402
from experiments.shift import thruster_chain as TC  # noqa: E402

import h5py  # noqa: E402

PAGE = Path("/mnt/c/Users/David/dev/bca-shift-work/page")
C = json.loads((PAGE / "tmp/p4/facts/compute.json").read_text())
X = json.loads((PAGE / "tmp/p4/facts/extra.json").read_text())
OUT = PAGE / "notes/p4_facts.json"
KNEE = TC.KNEE
SIT = ("recorded", "strong_current", "big_waves", "big_kicks")
SIT_LABEL = {"recorded": "recorded conditions", "strong_current": "strong current", "big_waves": "big waves",
             "big_kicks": "big starting pushes"}

SIGNED_EDGES = np.linspace(-1.0, 1.0, 51)
SIGNED_EDGES[5], SIGNED_EDGES[25], SIGNED_EDGES[45] = -KNEE, 0.0, KNEE
MAG_EDGES = np.linspace(0.0, 1.0, 51)
MAG_EDGES[40] = KNEE


def hist_signed(a):
    a = np.asarray(a, np.float64)
    idx = np.empty(a.shape, np.int64)
    neg = a < 0
    idx[neg] = np.searchsorted(SIGNED_EDGES, a[neg], side="right") - 1
    idx[~neg] = np.searchsorted(SIGNED_EDGES, a[~neg], side="left") - 1
    idx[a == 0] = 25
    assert idx.min() >= 0 and idx.max() <= 49
    return [int(v) for v in np.bincount(idx, minlength=50)]


def hist_mag(a):
    m = np.abs(np.asarray(a, np.float64))
    idx = np.searchsorted(MAG_EDGES, m, side="left") - 1
    idx[m == 0] = 0
    assert idx.min() >= 0 and idx.max() <= 49
    return [int(v) for v in np.bincount(idx, minlength=50)]


def r(x, n=4):
    return None if x is None else float(round(float(x), n))


def rp(d, n=3):
    return {k: r(v, n) for k, v in d.items()}


# ------------------------------------------------------------------------------------------------ data again
lg = DS.generate(900, "stall")
gap, _ = SH.define_gap(lg)
rem = SH.make_removal(lg, "gap", gap)
train, kept, removed = lg.split_rows("train"), rem.kept_train_rows(lg), rem.removed_train_rows(lg)
A = np.asarray(lg["actions"], np.float64)
with h5py.File("/root/bca-shift/runs/shift_iw_v5/expert/data/seed0/train_law.hdf5", "r") as f:
    ia = f["actions"][()].astype(np.float64)
    itr = f["infos/split"][()] == 0
    inb = f["truth/in_ball"][()]
    ihold = f["infos/hold"][()]
ikept = itr & ~inb

# settled floats under a steady push (no wave): stall-aware controller and no throttle
settled = {}
for c in (0.4, 0.9):
    for name, pol in (("stall_aware", RC.stall_aware()), ("no_throttle", None)):
        x = np.zeros(13)
        x[TC.CURRENT] = c
        for _ in range(3000):
            a = np.zeros(4) if pol is None else pol(x[None])[0]
            x, _ = TC.dynamics(x[None], a[None], TC.STALL)
            x = x[0]
        a = np.zeros(4) if pol is None else pol(x[None])[0]
        settled[f"c{c}_{name}"] = dict(floats=[r(v) for v in x[:10]], command=[r(v) for v in a],
                                       float_cost=r(np.sum(x[:10] ** 2)))

# ------------------------------------------------------------------------------------------------ assemble
M = C["main_test_seed900"]
B = C["banks"]["by_situation"]
an = C["banks"]["anchors_file"]
H = X["difficulty_harm"]["by_situation"]

facts = {
    "about": "Facts and chart data for ThrusterChain page pass 4. Read with notes/p4_facts.md. Every number here was "
             "computed by scripts/p4_facts_compute.py, p4_facts_extra.py and p4_facts_build.py (read only) or read "
             "from the named source file. Rounded for display; recompute from the scripts for full precision.",
    "made": "2026-10-13",
    "conventions": {
        "thruster_sign": "Commands run from -1 to 1. Negative pushes upstream, against the water. Thruster 1's "
                         "full throttle is -1. 'Past 80%' means |a| > 0.800000011920929 (the float32 knee); a "
                         "command of exactly 80% does not stall.",
        "histogram_bins": "50 bins. Signed: edges linspace(-1, 1, 51) with the edges at -0.8 and +0.8 set to "
                          "-/+0.800000011920929; negative bins are [lo, hi), positive bins (lo, hi], 0 goes to bin 25 "
                          "[0, 0.04]. Magnitude |a|: edges linspace(0, 1, 51) with the 0.8 edge set to the knee; bins "
                          "(lo, hi], 0 goes to bin 0. So bins 0-4 (signed) and 45-49 (signed) and 40-49 (magnitude) "
                          "hold exactly the commands past 80%.",
        "situations": SIT_LABEL,
    },
}

# ---- 1. verified claims
reb = B["recorded"]
facts["verified_claims"] = [
    dict(claim="Only 60 of the 42,500 kept training rows go past 80% (main test, densest-15% removal)",
         verdict="verified", value=dict(kept_rows=M["rows"]["kept"], past80_any_thruster=M["kept"]["past80_any_count"],
                                        share=r(M["kept"]["past80_any_share"], 6)),
         detail="All 60 are thruster 1 pushing upstream past 80% (none of thrusters 2-4 ever goes past 80% in the "
                "data), and all 60 are the recording controller's full-throttle commands that lie outside the "
                "removed ball (0.89% of the 6,712 full-throttle commands survive).",
         source="design seed 900, regenerated with experiments/shift/dataset.generate(900) + shift.define_gap; "
                "page #shift"),
    dict(claim="93.7% of strong-current states are in the full-throttle region",
         verdict="verified", value=r(B["strong_current"]["stall_aware"]["push"]["share_above_trigger"], 5),
         detail="Share of the 100,000 visited states (1,000 bank starts x 100 steps) with c + w1 > 0.565. The push "
                "does not depend on the controller, so it is the same for every controller. Recorded conditions "
                "27.86%, big waves 39.61%, big starting pushes 27.86% (same current and waves as recorded).",
         source="truth.bank_rollout on thruster_chain.bank_starts; page #shift table column 3"),
    dict(claim="The rising-thrust controller scores -54.1 points in strong current",
         verdict="verified", value=dict(mean_return=r(B["strong_current"]["rising_thrust"]["mean_return"]),
                                        points=r(B["strong_current"]["rising_thrust"]["points"], 2)),
         detail="Stalling world. It goes past 80% on 84.0% of the states it visits there.",
         source="recomputed by bank rollout; page #reference-scores"),
    dict(claim="Anchors -79.80 and -26.36", verdict="verified",
         value=dict(recording=r(an["recorded"]["recording"]), stall_aware=r(an["recorded"]["top"])),
         detail="Average 100-step return over the 1,000 recorded-conditions bank starts, stalling world: the "
                "recording controller (0 points) and the stall-aware controller (100 points). Recomputed: "
                f"{reb['recording']['mean_return']:.4f} and {reb['stall_aware']['mean_return']:.4f}.",
         source="/root/bca-shift/runs/shift/pre/stage0_base.json record.anchors; recomputed"),
    dict(claim="Push 0.65 at 90%, 0.5 at full throttle", verdict="verified",
         value=dict(at_90=X["thrust_points"]["0.90"]["stall"], at_100=X["thrust_points"]["1.00"]["stall"]),
         detail="thrust(a) = sign(a)(0.8 - 1.5(|a| - 0.8)) past the knee; because the knee is the float32 of 0.8, "
                "the exact values are 3e-8 above the round numbers (0.65000003, 0.50000003).",
         source="experiments/shift/thruster_chain.py thrust()"),
    dict(claim="The water's push points upstream on 20.3% of big-wave states", verdict="verified",
         value=r(B["big_waves"]["stall_aware"]["push"]["share_upstream_push_below_0"], 5),
         detail="Push c + w1 < 0. Range over visited states -0.595 to 1.389 (page: -0.6 to 1.4). Recorded "
                "conditions: 6.13% upstream, range -0.295 to 1.089; strong current: never upstream, range 0.304 to "
                "1.492.",
         source="bank rollout"),
    dict(claim="Stage 1: a penalty placed by the exact one-step error won back 93.9 of the 98.9 points lost in "
               "strong current", verdict="98.9 verified; 93.9 taken from the page (raw Stage 1 outputs are not on "
                                         "this machine)",
         value=dict(harm=r(H["strong_current"]["harm_mean"], 2), recovered=93.9),
         detail="The 98.9 is the return-cost check's harm in strong current at critic weight 5 (seeds 95.2-102.8). "
                "The 93.9 (seeds 83.4 to 103.9) is in the page's Stage 1 table (#beyond-the-data); its run outputs "
                "(runs/shift_explore/room/) were made in a cloud session and are not in WSL or the work folder. "
                "Precise wording: the penalty, at training states and at the actor's own command, was "
                "[delta*(s, actor's command) - delta*(s, recorded command)]+, how much more the critic's exact "
                "one-step error at the actor's command exceeds that at the recorded command, counted only when "
                "positive (outputs/shift/beyond-the-data/stage1/README.md).",
         source="outputs/shift/stop-report/check2_return/alpha5/pooled.json; page #beyond-the-data"),
    dict(claim="Harm by situation 5.8, 11.8, 16.6, 98.9 points", verdict="verified",
         value={s: r(H[s]["harm_mean"], 2) for s in SIT},
         detail="recorded conditions 5.8, big starting pushes 11.8, big waves 16.6, strong current 98.9: the "
                "actor of the full-log critic minus the actor of the critic trained after the densest-15% removal, "
                "critic weight 5, mean of seeds 0-4.",
         source="outputs/shift/stop-report/check2_return/alpha5/pooled.json situations.*.harm"),
    dict(claim="BCA's own lower-edge penalty is 1/30 to 1/160 of the perfect one and barely tracks it",
         verdict="verified (wording: penalty, not band)",
         value={q: dict(ratio=r(v["band_over_oracle_mean"], 5), one_over=r(v["one_over_ratio"], 1),
                        correlation=r(v["pearson_mean"], 3))
                for q, v in X["iw_lower_edge_penalty_size"]["by_quality"].items()},
         detail="Measured at the actor's commands on training rows: BCA's width there minus its width at the recorded "
                "command, against the same penalty built from the critic's exact one-step error. 1/31-1/33 on the "
                "three lower qualities, 1/58 on medium-expert, 1/163 on expert; row-by-row correlation -0.17 to 0.29 "
                "(quality means).",
         source="runs/shift_iw_v5/<quality>/report/report.json marks.L2.proxy_check"),
    dict(claim="Version-5 data: no thruster-1 commands past 80% in the expert data; 3.6-9.9% in the other qualities, "
               "all random", verdict="partly verified: the 'none' and 'all random' parts hold; the 3.6-9.9% range "
                                     "does not reproduce exactly (see value)",
         value=X["iw_t1_past80_by_direction"],
         detail="Expert: zero training rows past 80% on any thruster, in every seed. Other qualities, training "
                "split, thruster 1 past 80% upstream (a1 < -0.8, the direction of the hold): random 9.98-10.11%, "
                "medium-replay 8.21-8.43%, medium 6.45-6.59%, medium-expert 3.25-3.53% of rows (all rows before the "
                "removal); on the rows kept after the hold removal 3.61-3.90% (medium-expert) to 9.99-10.12% "
                "(random). About as many go past 80% downstream (a1 > +0.8). Every one of these rows is a uniform "
                "random command (infos/random = True on 100% of them). Suggested wording: 'none in the expert data; "
                "in the other qualities about 3% to 10% of training steps push thruster 1 upstream past 80%, and "
                "every one of them is a uniform random command'.",
         source="runs/shift_iw_v5/<quality>/data/seed<k>/train_law.hdf5"),
]

# ---- 2. thrust and cost
grid = np.round(np.arange(0, 1.0001, 0.05), 2)
facts["thrust_and_cost"] = {
    "formula": {"push_stall": "f(a) = a for |a| <= 0.8; sign(a)(0.8 - 1.5(|a| - 0.8)) past 0.8",
                "push_no_stall": "f(a) = a", "throttle_cost": "0.3 a^2 per thruster, on the clipped command, in both "
                                                               "worlds, even while stalling",
                "knee": KNEE, "best_push_stalling_world": "0.8, at 80% throttle"},
    "table": [dict(throttle=float(g), push_stall=r(TC.thrust(np.array([g]), TC.STALL)[0], 8),
                   push_no_stall=float(g), throttle_cost=r(0.3 * g * g, 5),
                   push_minus_cost_stall=r(TC.thrust(np.array([g]), TC.STALL)[0] - 0.3 * g * g, 5),
                   stalls=bool(g > KNEE)) for g in grid],
    "note": "push minus cost is shown only as a convenience for a slider readout; push and cost are in different "
            "units of the problem (push moves the floats; the cost is subtracted from the reward), so the page "
            "should not call it a net value without saying so.",
}

# ---- 3. stall chart data
m_full, m_kept, m_rem = A[train, 0], A[kept, 0], A[removed, 0]
facts["stall_chart"] = {
    "signed_edges": [float(e) for e in SIGNED_EDGES],
    "magnitude_edges": [float(e) for e in MAG_EDGES],
    "main_test_seed900_thruster1": {
        "source": "design seed 900 training split (50,000 rows); kept = after the densest-15% removal (42,500 rows); "
                  "removed = the 7,500 deleted rows",
        "totals": dict(full=int(len(m_full)), kept=int(len(m_kept)), removed=int(len(m_rem))),
        "signed": dict(full=hist_signed(m_full), kept=hist_signed(m_kept), removed=hist_signed(m_rem)),
        "magnitude": dict(full=hist_mag(m_full), kept=hist_mag(m_kept), removed=hist_mag(m_rem)),
        "past80": dict(full=int((np.abs(m_full) > KNEE).sum()), kept=int((np.abs(m_kept) > KNEE).sum())),
        "exactly_full_throttle_minus1": dict(full=int((m_full == -1).sum()), kept=int((m_kept == -1).sum())),
        "upstream_share_full": r((m_full < 0).mean()),
        "positive_max": r(m_full.max()),
        "note": "Every past-80% command is upstream (negative). The removal deletes 6,652 of the 6,712 past-80% "
                "rows; the 60 that stay sit at |a1| 0.96-1.0.",
    },
    "weighting_test_expert_seed0_thruster1": {
        "source": "runs/shift_iw_v5/expert/data/seed0/train_law.hdf5, training split; hold removed = rows outside "
                  "truth/in_ball",
        "totals": dict(all=int(itr.sum()), hold_removed=int(ikept.sum()), removed=int((itr & inb).sum()),
                       hold_rows=int((itr & ihold).sum())),
        "signed": dict(all=hist_signed(ia[itr, 0]), hold_removed=hist_signed(ia[ikept, 0]),
                       removed=hist_signed(ia[itr & inb, 0])),
        "magnitude": dict(all=hist_mag(ia[itr, 0]), hold_removed=hist_mag(ia[ikept, 0]),
                          removed=hist_mag(ia[itr & inb, 0])),
        "hold_value": float(np.unique(ia[itr & ihold, 0])[0]),
        "past80": dict(all=int((np.abs(ia[itr, 0]) > KNEE).sum()), hold_removed=int((np.abs(ia[ikept, 0]) > KNEE).sum())),
        "note": "The 80% hold is thruster 1 at exactly -0.800000011920929 (the knee): at 80%, not past it. The "
                "expert data hold no command past 80% at all.",
    },
}

# ---- 4. four-throttle widget
pt = M["percentiles"]
facts["four_throttle_widget"] = {
    "push_and_cost": "see thrust_and_cost; the same function for every thruster",
    "thrusters": {
        "1": "pushes the speeds of floats 1 and 2 (observation items 1 and 3); the only thruster on float 1",
        "2": "floats 2 and 3 (items 3 and 5)", "3": "floats 3 and 4 (items 5 and 7)",
        "4": "floats 4 and 5 (items 7 and 9); the only thruster on float 5"},
    "one_step_push_into_speeds": {"B_columns_first10": X["B_columns"],
                                  "water_push_column_first10": X["water_push_column"],
                                  "note": "next state = A x + B thrust + water push: each thruster adds 0.3 x its push "
                                          "to the speed numbers of its two floats; the water adds 0.3 (c + w1) to "
                                          "every float's speed number"},
    "main_test_seed900": {
        "percentiles_full": {k: rp(v["full_train"]) for k, v in pt.items()},
        "percentiles_kept": {k: rp(v["kept_after_densest15"]) for k, v in pt.items()},
        "percentiles_removed": {k: rp(v["removed"]) for k, v in pt.items()},
        "min_max_full": {k: [r(v["min_full"]), r(v["max_full"])] for k, v in pt.items()},
        "share_past80_full": {k: r(v["share_below_minus80_full"] + v["share_above_plus80_full"], 5)
                              for k, v in pt.items()},
        "share_past80_kept": {k: r(v["share_below_minus80_kept"] + v["share_above_plus80_kept"], 6)
                              for k, v in pt.items()},
        "signed_hist_full": {f"thruster{k + 1}": hist_signed(A[train, k]) for k in range(4)},
        "signed_hist_kept": {f"thruster{k + 1}": hist_signed(A[kept, k]) for k in range(4)},
        "removed_ball": dict(centre=[r(v) for v in M["gap"]["centre"]], radius=r(M["gap"]["rho"]),
                             projection_per_thruster=[[r(a), r(b)] for a, b in M["ball_projection_per_thruster"]],
                             note="The ball is a 4-D region: a command is removed only if its distance to the centre, "
                                  "over all four throttles together, is at most the radius. The per-thruster "
                                  "intervals are its shadows, not the removal rule."),
        "full_throttle_pattern": [r(v) for v in M["full_throttle_pattern"]],
        "full_throttle_pattern_distance": r(M["full_throttle_pattern_distance_to_centre"]),
        "capped_pattern": [r(v) for v in M["capped_pattern_thruster1_at_80"]],
        "capped_pattern_distance": r(M["capped_pattern_t1_distance_to_centre"]),
        "normal_driving_t1_range": [r(v) for v in M["t1_detail"]["normal_driving_t1_min_max"]],
        "normal_driving_t1_percentiles": rp(M["t1_detail"]["normal_driving_t1_percentiles"]),
        "full_throttle_t1_percentiles": rp(M["t1_detail"]["full_throttle_t1_percentiles"]),
    },
    "commands_on_evaluation_banks": {
        "source": "bank rollouts (stalling world, 1,000 starts x 100 steps): 5th/50th/95th percentile of each "
                  "thruster's command; share of states past 80% on any thruster; share with thruster 1 at its "
                  "-80% limit (stall-aware); share of commands inside the main test's removed ball",
        "by_situation": {s: {c: dict(percentiles={t: rp(v) for t, v in B[s][c]["command_percentiles"].items()},
                                     past80_any=r(B[s][c]["past80_any_share"]),
                                     t1_at_minus80=r(B[s][c]["t1_at_minus80_share"]),
                                     any_at_80_limit=r(B[s][c]["any_at_80_limit_share"]),
                                     in_removed_ball=r(B[s][c]["command_in_ball_share"]))
                             for c in ("recording", "stall_aware", "rising_thrust")} for s in SIT}},
    "weighting_test_expert_seed0": {
        "percentiles_all": {k: rp(v) for k, v in C["weighting_test_v5"]["expert_seed0"][
            "per_thruster_percentiles_all_train"].items()},
        "percentiles_hold_removed": {k: rp(v) for k, v in C["weighting_test_v5"]["expert_seed0"][
            "per_thruster_percentiles_hold_removed"].items()},
        "signed_hist_all": {f"thruster{k + 1}": hist_signed(ia[itr, k]) for k in range(4)},
        "signed_hist_hold_removed": {f"thruster{k + 1}": hist_signed(ia[ikept, k]) for k in range(4)},
    },
}

# ---- 5. observation space
names = [("s[0]", "position number of float 1", "observation 0"), ("s[1]", "speed number of float 1", "observation 1"),
         ("s[2]", "position number of float 2", "observation 2"), ("s[3]", "speed number of float 2", "observation 3"),
         ("s[4]", "position number of float 3", "observation 4"), ("s[5]", "speed number of float 3", "observation 5"),
         ("s[6]", "position number of float 4", "observation 6"), ("s[7]", "speed number of float 4", "observation 7"),
         ("s[8]", "position number of float 5", "observation 8"), ("s[9]", "speed number of float 5", "observation 9"),
         ("c", "strength of the current; never changes during an episode", "observation 10"),
         ("w1", "wave, first number: amplitude x cos(phase), the push the wave adds now", "observation 11"),
         ("w2", "wave, second number: amplitude x sin(phase), the push it added a quarter cycle (5.8 steps) earlier",
          "observation 12")]
okeys = list(B["recorded"]["recording"]["obs_percentiles"].keys())
comp = []
for i, (sym, nm, ob) in enumerate(names):
    k = okeys[i]
    row = dict(index=i, symbol=sym, name=nm, item=ob,
               at_rest=(0.0 if i < 10 else None),
               at_rest_note=("0 at rest" if i < 10 else
                             "no rest value: drawn at the start; c stays fixed" if i == 10 else
                             "no rest value: drawn at the start, then turns 0.27 rad per step (amplitude fixed)"),
               bounds=([-300.0, 300.0] if i < 10 else [0.0, 1.5] if i == 10 else [-1.0, 1.0]),
               ranges={})
    for s in SIT:
        row["ranges"][s] = {
            "recording_controller": rp({p: B[s]["recording"]["obs_percentiles"][k][p] for p in ("p5", "p50", "p95")}),
            "stall_aware_controller": rp({p: B[s]["stall_aware"]["obs_percentiles"][k][p]
                                          for p in ("p5", "p50", "p95")})}
    row["seed900_training_data"] = rp({p: C["banks"]["seed900_train_obs_percentiles"][k][p]
                                       for p in ("p5", "p50", "p95")})
    comp.append(row)
facts["observation_space"] = {
    "source": "5th, 50th and 95th percentiles over the 100,000 states at which a command is given (1,000 evaluation "
              "bank starts x steps 0-99) in each situation, stalling world, for the recording controller (with its "
              "fixed bank noise streams, as for the 0-point anchor) and for the stall-aware controller (the "
              "100-point anchor). The column seed900_training_data is the main test's training split (50,000 rows), "
              "which matches the recording controller in the recorded conditions.",
    "names_from_env_docstring": "s (10): the positions and speeds of the five floats (coordinates 2j and 2j + 1); "
                                "c (1): the strength of the current, never changes during an episode; (w1, w2): "
                                "the wave, a point turning 0.27 rad per step (a 23.3-step cycle), like the hands of "
                                "a clock",
    "components": comp,
    "settled_under_steady_current_no_wave": settled,
    "situation_start_laws": {s: TC.START_LAWS[s] for s in SIT},
    "notes": "c and the wave do not depend on the controller, so their ranges are the same for both controllers. "
             "The wave's range in big waves is wider (amplitude 0.4-0.7). Floats range widest in big starting "
             "pushes (start SD 1.5) and are shifted downstream (negative position numbers) in strong current.",
}

# ---- 6. rewards worked example
ex = X["rewards_example2"]
facts["rewards_example"] = {
    "note": "One state, thruster 1 set to -0.8, -0.9, -1.0 (80%, 90%, 100% upstream); thrusters 2-4 at the "
            "stall-aware controller's command. The reward of this step depends only on the state and the throttle, so "
            "it is the same in both worlds; the stall shows up in the next state's float cost and in the exact value. "
            "Exact value = discounted (0.98) sum of rewards over 1,000 steps with the stall-aware controller "
            "continuing after the first step (the page's 'true value').",
    "states": {},
}
for key, e in ex.items():
    rows = []
    for w in ("stall", "no_stall"):
        for row in e["rows"]:
            if row["world"] == w and row["continues"] == "stall_aware":
                rows.append(dict(world=w, throttle_pct=int(round(-row["thruster1_command"] * 100)),
                                 push=r(row["push_t1"], 5), throttle_cost_t1=r(row["throttle_cost_t1"], 4),
                                 reward_this_step=r(row["reward_now"], 4),
                                 float_cost_next_state=r(row["float_cost_next"], 4),
                                 next_speed_float1=r(row["next_float1_speed"], 4),
                                 exact_value=r(row["q_discounted_1000"], 4)))
    base = {(q["world"], q["throttle_pct"]): q for q in rows}
    for q in rows:
        b = base[(q["world"], 80)]
        q["change_vs_80_reward_this_step"] = r(q["reward_this_step"] - b["reward_this_step"], 4)
        q["change_vs_80_float_cost_next"] = r(q["float_cost_next_state"] - b["float_cost_next_state"], 4)
        q["change_vs_80_exact_value"] = r(q["exact_value"] - b["exact_value"], 4)
    facts["rewards_example"]["states"][key] = dict(label=e["label"], state=[r(v) for v in e["state"]],
                                                   water_push=r(e["push_water"]),
                                                   stall_aware_command=[r(v) for v in e["stall_aware_command"]],
                                                   rows=rows)
facts["rewards_example"]["recommended"] = "rest_c0.6_w0.3 (inside the recorded conditions, in the full-throttle region)"

# ---- 7. points
facts["points"] = {
    "formula": "points = 100 (J - J_rec) / (J_top - J_rec), per situation, on the same 1,000 bank starts",
    "anchors": {s: dict(recording=r(an[s]["recording"], 2), stall_aware=r(an[s]["top"], 2),
                        return_units_per_point=r((an[s]["top"] - an[s]["recording"]) / 100, 4),
                        stall_aware_minus_recording=r(an[s]["top"] - an[s]["recording"], 2)) for s in SIT},
    "source": "/root/bca-shift/runs/shift/pre/stage0_base.json record.anchors (recomputed, identical)",
}

# ---- 8. difficulty
arms = {  # stop report, check 2 table (means over seeds 0-4, points); plain TD3+BC from anatomy.md section 6
    "recorded": dict(cloned=56.3, full_log_critic_actor=91.3, removal_critic_actor=85.5, plain_td3bc=86.1),
    "strong_current": dict(cloned=54.4, full_log_critic_actor=84.6, removal_critic_actor=-14.3, plain_td3bc=-16.8),
    "big_waves": dict(cloned=43.6, full_log_critic_actor=86.6, removal_critic_actor=70.0, plain_td3bc=70.9),
    "big_kicks": dict(cloned=2.0, full_log_critic_actor=64.3, removal_critic_actor=52.5, plain_td3bc=48.7)}
outside = {"recorded": "about 1% (by the float test's 99% mark)", "strong_current": "100%", "big_waves": "100%",
           "big_kicks": "10.9% (stall-aware) to 17.3% (recording controller) of visited states"}
terms = {"recorded": "in-distribution", "big_kicks": "initial-state shift", "big_waves": "disturbance shift",
         "strong_current": "dynamics extrapolation"}
diff = {}
for s in SIT:
    u = (an[s]["top"] - an[s]["recording"]) / 100
    raw = {k: r(an[s]["recording"] + v * u, 1) for k, v in arms[s].items()}
    diff[s] = dict(label=SIT_LABEL[s], standard_term=terms[s],
                   harm_points=r(H[s]["harm_mean"], 2), harm_interval=[r(v, 2) for v in H[s]["harm_interval"]],
                   harm_return_units=r(H[s]["harm_mean"] * u, 1),
                   reference_raw_returns=dict(recording=r(an[s]["recording"], 2), stall_aware=r(an[s]["top"], 2),
                                              rising_thrust=r(B[s]["rising_thrust"]["mean_return"], 2),
                                              normal_driving_only=r(B[s]["normal_driving_only"]["mean_return"], 2)),
                   learned_actor_points=arms[s], learned_actor_raw_returns_derived=raw,
                   past80_share_removal_critic_actor=r(np.mean(H[s]["past80_share_gapped_actor"]), 3),
                   outside_recorded_conditions=outside[s],
                   plain_td3bc_minus_cloning=r(H[s]["plain_td3bc_minus_cloning"], 1))
facts["difficulty"] = {
    "order_by_harm": ["recorded", "big_kicks", "big_waves", "strong_current"],
    "order_by_raw_return_reference_controllers": ["recorded", "big_waves", "strong_current", "big_kicks"],
    "by_situation": diff,
    "harm_source": "outputs/shift/stop-report/check2_return/alpha5/pooled.json (harm = actor of the full-log critic "
                   "minus actor of the critic trained after the densest-15% removal, critic weight 5, seeds 0-4)",
    "arms_source": "outputs/shift/stop-report/README.md check 2 table; plain TD3+BC from "
                   "outputs/shift/beyond-the-data/anatomy.md section 6",
    "note": "Raw returns of the learned actors are derived from their mean points and the anchors (exact up to the "
            "0.1-point rounding of the points). By raw return, big starting pushes is hardest for both reference "
            "controllers and for the cloned and full-log-critic actors; for the actor of the critic trained after "
            "the removal (and plain TD3+BC), strong current is lowest (-275 against -211).",
}

facts["weighting_test_data"] = {
    "past80_by_quality": X["iw_t1_past80_by_direction"],
    "removed_rows_per_quality_seed0": {q: C["weighting_test_v5"]["by_quality"][q][0]["removed_rows"]
                                       for q in C["weighting_test_v5"]["by_quality"]},
    "hold_share_train": {q: [r(x["hold_share"], 4) for x in C["weighting_test_v5"]["by_quality"][q]]
                         for q in C["weighting_test_v5"]["by_quality"]},
    "hold_removed_share": {q: [r(x["hold_removed_share"], 4) for x in C["weighting_test_v5"]["by_quality"][q]]
                           for q in C["weighting_test_v5"]["by_quality"]},
    "source": "runs/shift_iw_v5/<quality>/data/seed<k>/train_law.hdf5",
}
facts["lower_edge_size"] = X["iw_lower_edge_penalty_size"]

# ---- 9. the weighting test's removed ball (the expert's densest-15% ball, shared by every quality of a seed)
balls = {}
for s in range(5):
    w = json.load(open(f"/root/bca-shift/runs/shift_iw_v5/expert/data/seed{s}/world.json"))
    ctr = np.asarray(w["gap"]["centre"], np.float64)
    hold = np.asarray(w["hold_command"], np.float64)
    balls[s] = dict(centre=[r(v) for v in ctr], radius=r(w["gap"]["rho"]), hold_command=[r(v) for v in hold],
                    hold_distance_to_centre=r(np.linalg.norm(hold - ctr)),
                    removed_rows_that_are_holds=r(w["removal"]["removed_that_are_holds"]),
                    holds_removed_training_rows=r(w["removal"]["holds_removed_training"]),
                    holds_removed_design_check_quadrature=r(w["D1"]["checks"]["holds_removed"]["value"]),
                    hold_share_training=r(w["removal"]["hold_share_training"]))
facts["weighting_test_ball"] = {
    "source": "runs/shift_iw_v5/expert/data/seed<k>/world.json (gap, hold_command, removal, D1)",
    "by_seed": balls,
    "note": "The weighting test also uses the densest-15% rule, on the expert's commands; the ball lands on the 80% "
            "hold (thruster 1 at -0.8 with the other three at 0.8 x the full-throttle pattern). About two thirds of "
            "the removed rows are holds; the rest are normal-driving commands near the ball's centre. The page's "
            "'99.57-99.79% of holds removed' is the design check's quadrature value; counting the training rows "
            "gives the holds_removed_training_rows values.",
}

OUT.write_text(json.dumps(facts, indent=1, ensure_ascii=False))
print("wrote", OUT, OUT.stat().st_size, "bytes")
print(json.dumps(settled))
for k, v in facts["rewards_example"]["states"].items():
    print(k)
    for q in v["rows"]:
        print("  ", q)
for s, v in diff.items():
    print(s, v["harm_points"], v["harm_return_units"], v["learned_actor_raw_returns_derived"])
mt = facts["stall_chart"]["main_test_seed900_thruster1"]
print(mt["past80"], sum(mt["signed"]["full"][:5]), sum(mt["signed"]["kept"][:5]), sum(mt["magnitude"]["kept"][40:]))
wt = facts["stall_chart"]["weighting_test_expert_seed0_thruster1"]
print(wt["past80"], wt["signed"]["all"][3:7], wt["magnitude"]["all"][38:42], wt["totals"])

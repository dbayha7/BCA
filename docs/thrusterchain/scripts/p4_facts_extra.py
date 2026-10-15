"""Facts for page pass 4, second part (read only). Writes tmp/p4/facts/extra.json.

Run from /mnt/c/Users/David/dev/bca-shift-work/page with
  PYTHONPATH=/root/bca-shift /opt/bca-shift-venv/bin/python scripts/p4_facts_extra.py
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
from experiments.shift import truth as TR  # noqa: E402

import h5py  # noqa: E402

OUT = Path("/mnt/c/Users/David/dev/bca-shift-work/page/tmp/p4/facts/extra.json")
IW = Path("/root/bca-shift/runs/shift_iw_v5")
STOP = Path("/root/bca-shift/outputs/shift/stop-report")
QUALITIES = ("random", "medium-replay", "medium", "medium-expert", "expert")
KNEE = TC.KNEE
MAG_EDGES = np.linspace(0.0, 1.0, 51)
out = {"magnitude_bin_edges": [float(e) for e in MAG_EDGES]}


def mhist(a):
    c, _ = np.histogram(np.abs(np.asarray(a, np.float64)), bins=MAG_EDGES)
    return [int(v) for v in c]


def summ(v):
    v = np.asarray(v, np.float64)
    return dict(min=float(v.min()), max=float(v.max()), mean=float(v.mean()))


# 1. main test seed 900: |thruster 1| histograms (0..1, 50 bins of 0.02; the 80% edge is bin 40's left edge)
lg = DS.generate(900, "stall")
gap, _ = SH.define_gap(lg)
rem = SH.make_removal(lg, "gap", gap)
train, kept = lg.split_rows("train"), rem.kept_train_rows(lg)
A = np.asarray(lg["actions"], np.float64)
out["main_seed900_t1_magnitude_hist"] = {
    "full_train": mhist(A[train, 0]), "kept_after_densest15": mhist(A[kept, 0]),
    "removed": mhist(A[rem.removed_train_rows(lg), 0]),
    "full_train_upstream_share": float((A[train, 0] < 0).mean()),
    "note": "|a1|; in the main test every thruster-1 command past 80% is upstream (negative); positive commands "
            "never exceed 0.7315"}

# 2. weighting test: directional thruster-1 shares on the kept rows (hold removed), per quality, seeds 0-4
dirs = {}
for q in QUALITIES:
    acc = {k: [] for k in ("all_up", "all_down", "kept_up", "kept_down", "kept_any_dir", "all_any_dir",
                           "random_among_up", "random_among_down", "kept_share_of_train")}
    for s in range(5):
        with h5py.File(IW / q / "data" / f"seed{s}" / "train_law.hdf5", "r") as f:
            a = f["actions"][()].astype(np.float64)
            tr = f["infos/split"][()] == 0
            inb = f["truth/in_ball"][()]
            rnd = f["infos/random"][()]
            if q == "expert" and s == 0:
                out["iw_expert_seed0_t1_magnitude_hist"] = {"all_train": mhist(a[tr, 0]),
                                                            "hold_removed": mhist(a[tr & ~inb, 0])}
        kp = tr & ~inb
        up_all, dn_all = a[tr, 0] < -KNEE, a[tr, 0] > KNEE
        acc["all_up"].append(up_all.mean())
        acc["all_down"].append(dn_all.mean())
        acc["all_any_dir"].append((np.abs(a[tr, 0]) > KNEE).mean())
        acc["kept_up"].append((a[kp, 0] < -KNEE).mean())
        acc["kept_down"].append((a[kp, 0] > KNEE).mean())
        acc["kept_any_dir"].append((np.abs(a[kp, 0]) > KNEE).mean())
        acc["random_among_up"].append(rnd[tr][up_all].mean() if up_all.any() else float("nan"))
        acc["random_among_down"].append(rnd[tr][dn_all].mean() if dn_all.any() else float("nan"))
        acc["kept_share_of_train"].append(kp.sum() / tr.sum())
    dirs[q] = {k: (summ(v) if not np.all(np.isnan(v)) else None) for k, v in acc.items()}
    dirs[q]["seed0_all_up"] = float(acc["all_up"][0])
    dirs[q]["seed0_kept_up"] = float(acc["kept_up"][0])
out["iw_t1_past80_by_direction"] = dirs

# 3. weighting test reports: BCA's own lower-edge penalty against the perfect one (band_over_oracle), correlation,
#    the perfect lower edge's gain, the placed pull's gain
rep = {}
for q in QUALITIES:
    d = json.load(open(IW / q / "report" / "report.json"))
    L1, L2 = d["marks"]["L1"], d["marks"]["L2"]
    pc = L2.get("proxy_check", {})
    bo = pc.get("band_over_oracle", {})
    rep[q] = {"band_over_oracle_mean": bo.get("mean"), "band_over_oracle_per_seed": bo.get("per_seed"),
              "one_over_ratio": (1.0 / bo["mean"]) if bo.get("mean") else None,
              "pearson_mean": pc.get("pearson", {}).get("mean"),
              "pearson_per_seed": pc.get("pearson", {}).get("per_seed"),
              "band_check_e_passed": pc.get("band_check_e_passed"),
              "oracle_minus_none_return": L1.get("oracle_minus_none", {}).get("mean"),
              "oracle_minus_none_lower": L1.get("oracle_minus_none", {}).get("lower_one_sided")}
out["iw_lower_edge_penalty_size"] = {"source": "runs/shift_iw_v5/<quality>/report/report.json, marks.L2.proxy_check "
                                               "(band_over_oracle, pearson) and marks.L1.oracle_minus_none",
                                     "by_quality": rep}

# 4. difficulty: harm per situation (return-cost check, critic weight 5)
d = json.load(open(STOP / "check2_return" / "alpha5" / "pooled.json"))
harm = {}
for s, v in d["situations"].items():
    harm[s] = {"harm_mean": v["harm"]["mean"], "harm_interval": [v["harm"]["lower"], v["harm"]["upper"]],
               "harm_per_seed": v["harm"]["values"],
               "past80_share_full_log_actor": v["mechanism"].get("share_full"),
               "past80_share_gapped_actor": v["mechanism"].get("share_gapped"),
               "plain_td3bc_minus_cloning": v["reported"]["td3bc_minus_cloning"]["mean"],
               "random_removal_harm": v["reported"]["random_harm"]["mean"],
               "no_stall_harm": v["reported"]["no_stall_harm"]["mean"]}
out["difficulty_harm"] = {"source": "outputs/shift/stop-report/check2_return/alpha5/pooled.json (situations.*.harm)",
                          "by_situation": harm}

# 5. rewards example, more states, with the same continuing controller in both worlds
sa, rt = RC.stall_aware(), RC.rising_thrust()


def example(x, label):
    x = np.asarray(x, np.float64)
    base = sa(x[None])[0]
    rows = []
    for mag in (0.8, 0.9, 1.0):
        a = base.copy()
        a[0] = -mag
        for world, phys in (("stall", TC.STALL), ("no_stall", TC.NO_STALL)):
            for cname, cont in (("stall_aware", sa), ("rising_thrust", rt)):
                x1, r0 = TC.dynamics(x[None], a[None], phys)
                q = TR.q_value(cont, x[None], a[None], phys)[0]
                rows.append(dict(world=world, continues=cname, thruster1_command=-mag,
                                 push_t1=float(TC.thrust(np.array([-mag]), phys)[0]),
                                 throttle_cost_t1=0.3 * mag * mag, reward_now=float(r0[0]),
                                 float_cost_next=float(np.sum(x1[0, :10] ** 2)),
                                 next_float1_speed=float(x1[0, 1]), next_float2_speed=float(x1[0, 3]),
                                 q_discounted_1000=float(q)))
    return dict(label=label, state=[float(v) for v in x], push_water=float(TC.push(x)),
                stall_aware_command=[float(v) for v in base], rows=rows)


x1 = np.zeros(13)
x1[TC.CURRENT] = 0.9
x2 = np.zeros(13)
x2[TC.CURRENT], x2[TC.WAVE1] = 0.6, 0.3
out["rewards_example2"] = {
    "rest_c0.9": example(x1, "floats at rest, current 0.9 (strong current), no wave"),
    "rest_c0.6_w0.3": example(x2, "floats at rest, current 0.6, wave w1 = 0.3, w2 = 0 (inside the recorded conditions, "
                                  "in the full-throttle region: push 0.9)")}
# one-step effect of thruster 1's push on the floats: B column 0
Amat, Bmat = TC.matrices(TC.STALL)
out["B_column_thruster1"] = [float(v) for v in Bmat[:10, 0]]
out["B_columns"] = [[float(v) for v in Bmat[:10, k]] for k in range(4)]
out["water_push_column"] = [float(v) for v in Amat[:10, TC.CURRENT]]

# 6. thrust values at the quoted points
out["thrust_points"] = {f"{g:.2f}": dict(stall=float(TC.thrust(np.array([g]), TC.STALL)[0]),
                                         no_stall=float(TC.thrust(np.array([g]), TC.NO_STALL)[0]),
                                         cost=0.3 * g * g)
                        for g in (0.0, 0.2, 0.4, 0.6, 0.8, 0.85, 0.9, 0.95, 1.0)}
OUT.write_text(json.dumps(out, indent=1))
print("wrote", OUT)

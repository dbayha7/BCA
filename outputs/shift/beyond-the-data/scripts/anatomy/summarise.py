"""EXPLORATORY: pool the anatomy outputs over gate seeds (mean and min-max over seeds). Reads
runs/shift_explore/anatomy/*_seed*.json, writes runs/shift_explore/anatomy/summary.json and prints the tables."""
import json
import sys
from pathlib import Path

import numpy as np

OUT = Path("/home/user/bca/runs/shift_explore/anatomy")
LABEL = "EXPLORATORY: computed outside the gate on saved gate artifacts; not a gated output; no BCA code was run"


def load(kind):
    out = {}
    for p in sorted(OUT.glob(f"{kind}_seed*.json")):
        out[int(p.stem.rsplit("seed", 1)[1])] = json.loads(p.read_text())
    return out


def get(d, path):
    for k in path:
        if d is None:
            return None
        d = d.get(k) if isinstance(d, dict) else None
    return d


def pool(runs, path):
    v = [get(r, path) for r in runs.values()]
    v = [x for x in v if isinstance(x, (int, float)) and x is not None and np.isfinite(x)]
    if not v:
        return None
    return dict(mean=float(np.mean(v)), lo=float(np.min(v)), hi=float(np.max(v)), n=len(v))


def fmt(p, d=2):
    if p is None:
        return "  n/a"
    return f"{p['mean']:+.{d}f} [{p['lo']:+.{d}f},{p['hi']:+.{d}f}]"


def main():
    S = dict(label=LABEL)
    cb, sw, ca, me = load("critic_beliefs"), load("actor_sweep"), load("caution_anatomy"), load("mechanism")
    S["seeds"] = dict(critic_beliefs=sorted(cb), actor_sweep=sorted(sw), caution_anatomy=sorted(ca), mechanism=sorted(me))
    print("seeds:", S["seeds"])

    # ---------------------------------------------------------------- part 1
    sets = ["train_region", "train_other", "recorded_visit_stall_aware", "strong_visit_stall_aware",
            "strong_visit_full_actor", "strong_visit_gapped_actor"] + [f"probe_c{c:.1f}" for c in
                                                                     (0.3, 0.5, 0.7, 0.8, 0.9, 1.0, 1.1)]
    critics = ["gapped", "full", "random", "full_placebo", "no_stall_gapped", "no_stall_full"]
    p1 = {}
    for base in ("pattern", "controller"):
        print(f"\nPART 1 ({base} base): critic preference for thruster 1 at full vs 80% minus the truth (mean, "
              "[min,max] over seeds); truth in brackets")
        for s in sets:
            row = {}
            for c in critics + (["host_gapped", "host_full"] if base == "pattern" else []):
                key = f"{base}|{c}"
                row[c] = {m: pool(cb, ["sets", s, key, m]) for m in (
                    "pref_error_mean", "pref_true_mean", "pref_critic_mean", "share_argmax_critic_past80",
                    "share_argmax_true_past80", "rise_0p6_to_0p8_error_mean", "rise_0p6_to_0p8_critic",
                    "rise_0p6_to_0p8_true", "share_critic_prefers_past80", "share_true_prefers_past80")}
            row["current_mean"] = pool(cb, ["sets", s, "current_mean"])
            p1[f"{base}|{s}"] = row
            print(f"  {s:28s} c={row['current_mean']['mean'] if row['current_mean'] else float('nan'):.2f} true "
                  f"{fmt(row['gapped']['pref_true_mean'])} | " + " | ".join(
                      f"{c}: {fmt(row[c]['pref_error_mean'])}" for c in ("gapped", "full", "random", "full_placebo")))
            print(f"  {'':28s} no-stall truth {fmt(row['no_stall_gapped']['pref_true_mean'])} | no_stall_gapped "
                  f"{fmt(row['no_stall_gapped']['pref_error_mean'])} | no_stall_full "
                  f"{fmt(row['no_stall_full']['pref_error_mean'])} | argmax past80 gapped "
                  f"{fmt(row['gapped']['share_argmax_critic_past80'])} full {fmt(row['full']['share_argmax_critic_past80'])}"
                  f" true {fmt(row['gapped']['share_argmax_true_past80'])}")
            if base == "pattern" and row.get("host_gapped", {}).get("pref_error_mean"):
                print(f"  {'':28s} host critics (truth = own actor): gapped {fmt(row['host_gapped']['pref_error_mean'])}"
                      f" full {fmt(row['host_full']['pref_error_mean'])}")
    S["part1_critic_beliefs"] = p1

    # ---------------------------------------------------------------- part 2
    actors = ["ref_stall_aware", "ref_normal_driving", "ref_rising_thrust", "clone", "full", "random", "gapped",
              "td3bc_gapped", "host_swap_full", "host_swap_gapped", "no_stall_full", "no_stall_gapped", "room_none",
              "room_harm_placed", "room_strength_matched_even", "room_amount_matched", "room_cap2", "room_even_x1.5",
              "room_even_x2", "room_even_x3"]
    p2 = {}
    print("\nPART 2: thruster-1 throttle against current (floats at rest, wave 0.25, 8 phases; and 64 training float "
          "configurations)")
    for a in actors:
        r = {}
        for probe in ("rest_amp0.25", "trainfloats_amp0.25"):
            for m in ("crossing_c_mean_throttle", "crossing_c_half_past80", "slope_in_data", "slope_out_of_data",
                      "throttle1_at_c0p5", "throttle1_at_c0p7", "throttle1_at_c0p9", "throttle1_at_c1p1"):
                r[f"{probe}|{m}"] = pool(sw, ["actors", a, probe, m])
            for i, c in ((8, 0.5), (12, 0.7), (16, 0.9), (20, 1.1)):
                v = [get(x, ["actors", a, probe, "share_any_past80"]) for x in sw.values()]
                v = [x[i] for x in v if x is not None]
                r[f"{probe}|share_any_past80_c{c}"] = dict(mean=float(np.mean(v)), lo=float(np.min(v)),
                                                           hi=float(np.max(v)), n=len(v)) if v else None
        n_never = sum(get(x, ["actors", a, "rest_amp0.25", "crossing_c_mean_throttle"]) is None for x in sw.values())
        r["rest_amp0.25|seeds_never_crossing"] = n_never
        r["wave_made_push_0p9_1p0"] = pool(sw, ["actors", a, "high_push_throttle1_wave_made"])
        r["current_made_push_0p9_1p0"] = pool(sw, ["actors", a, "high_push_throttle1_current_made"])
        p2[a] = r
        print(f"  {a:26s} thr1 c=.5 {fmt(r['rest_amp0.25|throttle1_at_c0p5'])} c=.7 {fmt(r['rest_amp0.25|throttle1_at_c0p7'])}"
              f" c=.9 {fmt(r['rest_amp0.25|throttle1_at_c0p9'])} c=1.1 {fmt(r['rest_amp0.25|throttle1_at_c1p1'])}"
              f" cross {fmt(r['rest_amp0.25|crossing_c_mean_throttle'])} (never {n_never})"
              f" slope in {fmt(r['rest_amp0.25|slope_in_data'])} out {fmt(r['rest_amp0.25|slope_out_of_data'])}")
        print(f"  {'':26s} trainfloats: past80 c=.5 {fmt(r['trainfloats_amp0.25|share_any_past80_c0.5'])} c=.7 "
              f"{fmt(r['trainfloats_amp0.25|share_any_past80_c0.7'])} c=.9 {fmt(r['trainfloats_amp0.25|share_any_past80_c0.9'])}"
              f" c=1.1 {fmt(r['trainfloats_amp0.25|share_any_past80_c1.1'])} half-cross "
              f"{fmt(r['trainfloats_amp0.25|crossing_c_half_past80'])}; push .9-1 wave-made "
              f"{fmt(r['wave_made_push_0p9_1p0'])} current-made {fmt(r['current_made_push_0p9_1p0'])}")
    S["part2_actor_sweep"] = p2

    # ---------------------------------------------------------------- part 3
    p3 = {}
    print("\nPART 3: where the exact-harm doses go (check 3's own oracle), and what each caution arm changes")
    for who in ("clone_step0", "harm_placed_final", "none_final"):
        k = f"doses_{who}"
        r = {m: pool(ca, [k, m]) for m in ("dose_mean", "region_share_rows", "region_share_of_extra_dose",
                                           "mean_dose_region", "mean_dose_outside", "share_rows_D_zero",
                                           "share_D_mass_on_rows_where_actor_past80", "corr_dose_push",
                                           "corr_dose_current")}
        for m in ("region_share", "push_mean", "current_mean", "recorded_throttle1_mean",
                  "recorded_throttle1_share_ge_0p75", "none_actor_throttle1_mean", "none_actor_share_past80"):
            r[f"top10|{m}"] = pool(ca, [k, "top10", m])
        for m in ("recorded_throttle1_mean", "recorded_throttle1_share_ge_0p75", "recorded_throttle1_share_past80",
                  "none_actor_throttle1_mean", "none_actor_share_past80"):
            r[f"region|{m}"] = pool(ca, [k, "region", m])
        for b in ("-1-0.3", "0.3-0.565", "0.565-0.8", "0.8-2"):
            r[f"push_{b}|share_rows"] = pool(ca, [k, "by_push", b, "share_rows"])
            r[f"push_{b}|share_extra"] = pool(ca, [k, "by_push", b, "share_extra"])
            r[f"push_{b}|mean_dose"] = pool(ca, [k, "by_push", b, "mean_dose"])
        p3[who] = r
        print(f"  doses at {who}: region rows {fmt(r['region_share_rows'])} region share of extra dose "
              f"{fmt(r['region_share_of_extra_dose'])}; mean dose region {fmt(r['mean_dose_region'], 3)} outside "
              f"{fmt(r['mean_dose_outside'], 3)}; D=0 rows {fmt(r['share_rows_D_zero'])}; D mass where actor past 80% "
              f"{fmt(r['share_D_mass_on_rows_where_actor_past80'])}")
        print(f"     top 10% dose rows: region {fmt(r['top10|region_share'])} push {fmt(r['top10|push_mean'])} recorded "
              f"thr1 {fmt(r['top10|recorded_throttle1_mean'])} (>=0.75: {fmt(r['top10|recorded_throttle1_share_ge_0p75'])})"
              f" no-caution actor thr1 {fmt(r['top10|none_actor_throttle1_mean'])} past80 {fmt(r['top10|none_actor_share_past80'])}")
        print(f"     push bins (share rows / share extra dose / mean dose): " + "; ".join(
            f"{b}: {r[f'push_{b}|share_rows']['mean']:.3f}/{r[f'push_{b}|share_extra']['mean']:.3f}/"
            f"{r[f'push_{b}|mean_dose']['mean']:.3f}" for b in ("-1-0.3", "0.3-0.565", "0.565-0.8", "0.8-2")
            if r[f'push_{b}|mean_dose']))
    r = p3["clone_step0"]
    print(f"  region rows, recorded thruster-1 throttle {fmt(r['region|recorded_throttle1_mean'])}, share >= 0.75 "
          f"{fmt(r['region|recorded_throttle1_share_ge_0p75'], 3)}, past 80% {fmt(r['region|recorded_throttle1_share_past80'], 3)};"
          f" no-caution actor there {fmt(r['region|none_actor_throttle1_mean'])} (past 80% "
          f"{fmt(r['region|none_actor_share_past80'])})")
    arms = ["harm_placed", "strength_matched_even", "amount_matched", "shuffled_0", "reversed", "cap2", "even_x1.5",
            "even_x2", "even_x3", "clone"]
    ch = {}
    print("  change against the no-caution arm: rms command change at training rows / region rows / top-dose rows / "
          "strong-current states; thruster-1 change region / strong; past-80% share in strong current (arm vs none);"
          " points in strong current")
    for a in arms:
        r = {}
        for s in ("train_all", "train_region", "train_top10_dose", "eval_split", "recorded_visit_none",
                  "strong_visit_none", "strong_visit_stall_aware"):
            for m in ("rms", "throttle1", "share_past80_arm", "share_past80_none"):
                r[f"{s}|{m}"] = pool(ca, ["actor_change_vs_none", a, s, m])
        r["ratio_strong_to_train_rms"] = pool(ca, ["actor_change_vs_none", a, "ratio_strong_to_train_rms"])
        r["points_strong_current"] = pool(ca, ["points", "strong_current", a]) if a != "clone" else None
        r["points_recorded"] = pool(ca, ["points", "recorded", a]) if a != "clone" else None
        ch[a] = r
        print(f"   {a:22s} {fmt(r['train_all|rms'], 3)} / {fmt(r['train_region|rms'], 3)} / "
              f"{fmt(r['train_top10_dose|rms'], 3)} / {fmt(r['strong_visit_none|rms'], 3)};  thr1 "
              f"{fmt(r['train_region|throttle1'], 3)} / {fmt(r['strong_visit_none|throttle1'], 3)};  past80 "
              f"{fmt(r['strong_visit_none|share_past80_arm'])} vs {fmt(r['strong_visit_none|share_past80_none'])};"
              f"  pts {fmt(r['points_strong_current'], 1)}")
    p3["actor_change_vs_none"] = ch
    p3["points_none"] = {s: pool(ca, ["points", s, "none"]) for s in ("recorded", "strong_current", "big_waves",
                                                                         "big_kicks")}
    S["part3_caution"] = p3

    # ---------------------------------------------------------------- part 4
    p4 = {}
    print("\nPART 4: what the actor reads during training (training states only)")
    beh = {}
    for a in ("recorded", "clone", "stall_aware", "full", "random", "gapped"):
        beh[a] = {m: pool(me, ["actor_at_training_states", a, m]) for m in (
            "throttle1_mean_region", "throttle1_mean_push_gt_0p8", "share_past80_all", "share_past80_region",
            "share_past80_push_gt_0p8", "share_throttle1_gt_0p75_region")}
        print(f"  {a:12s} thr1 region {fmt(beh[a]['throttle1_mean_region'])} push>0.8 "
              f"{fmt(beh[a]['throttle1_mean_push_gt_0p8'])}; past80 all {fmt(beh[a]['share_past80_all'])} region "
              f"{fmt(beh[a]['share_past80_region'])} push>0.8 {fmt(beh[a]['share_past80_push_gt_0p8'])}")
    p4["behaviour_at_training_states"] = beh
    sl = {}
    bins = ("throttle1_0.00-0.60", "throttle1_0.60-0.75", "throttle1_0.75-0.80", "throttle1_0.80-1.00")
    print("  critic slope along thruster-1 throttle (dQ per unit throttle) at the commands each actor emits at "
          "training states: true / gapped / full / random (share of rows)")
    for a in ("clone", "stall_aware", "full", "gapped"):
        sl[a] = {}
        for b in bins:
            sl[a][b] = {m: pool(me, ["slopes_at_actor_commands", a, b, m]) for m in (
                "share_rows", "dQ_dthrottle1_true", "dQ_dthrottle1_gapped", "dQ_dthrottle1_full",
                "dQ_dthrottle1_random", "grad_gap_gapped_minus_full_norm")}
            x = sl[a][b]
            if x["dQ_dthrottle1_true"]:
                print(f"   {a:12s} {b:20s} n {x['share_rows']['mean']:.3f}: {fmt(x['dQ_dthrottle1_true'])} / "
                      f"{fmt(x['dQ_dthrottle1_gapped'])} / {fmt(x['dQ_dthrottle1_full'])} / {fmt(x['dQ_dthrottle1_random'])}")
        sl[a]["lambda_gapped"] = pool(me, ["slopes_at_actor_commands", a, "lambda_at_alpha5", "gapped"])
        sl[a]["lambda_full"] = pool(me, ["slopes_at_actor_commands", a, "lambda_at_alpha5", "full"])
    print(f"   Q-term weight lambda = 5 / mean|Q| at the clone's commands: gapped {fmt(sl['clone']['lambda_gapped'], 3)}"
          f" full {fmt(sl['clone']['lambda_full'], 3)}")
    p4["slopes"] = sl
    vd = {}
    print("  value differences: critic minus truth (mean) and |gapped - full| (mean)")
    for k in ("recorded_vs_stall_aware", "clone_vs_stall_aware", "gapped_actor_past80_vs_capped",
              "full_actor_past80_vs_capped", "thruster1_full_vs_80_all", "thruster1_full_vs_80_region",
              "thruster1_full_vs_80_push_gt_0p8"):
        vd[k] = {m: pool(me, ["value_differences", k, m]) for m in (
            "n", "true_mean", "gapped_error_mean", "full_error_mean", "random_error_mean", "gapped_error_abs_mean",
            "full_error_abs_mean", "random_error_abs_mean", "gapped_minus_full_abs_mean")}
        x = vd[k]
        print(f"   {k:34s} true {fmt(x['true_mean'])} err gapped {fmt(x['gapped_error_mean'])} full "
              f"{fmt(x['full_error_mean'])} random {fmt(x['random_error_mean'])} |g-f| {fmt(x['gapped_minus_full_abs_mean'])}"
              + (f" abs err g {fmt(x['gapped_error_abs_mean'])} f {fmt(x['full_error_abs_mean'])}"
                 if x['gapped_error_abs_mean'] else ""))
    p4["value_differences"] = vd
    pts = {}
    for s in ("recorded", "strong_current", "big_waves", "big_kicks"):
        pts[s] = {a: pool(me, ["h2_points", s, a]) for a in ("clone", "full", "random", "gapped", "td3bc_gapped",
                                                             "host_swap_full", "host_swap_gapped")}
    p4["h2_points"] = pts
    print("  H2 points strong current: " + " ".join(f"{a} {fmt(v, 1)}" for a, v in pts["strong_current"].items()))
    S["part4_mechanism"] = p4
    (OUT / "summary.json").write_text(json.dumps(S, indent=1, sort_keys=True))
    print(f"\nwrote {OUT / 'summary.json'}")


if __name__ == "__main__":
    main()

"""EXPLORATORY (outside the gate, no BCA). Part 2: thruster-1 throttle of every saved actor as the current rises from
0.1 to 1.2 (recorded maximum 0.7), with the other inputs at typical values.

Inputs: floats at rest (all ten numbers 0) or 64 float configurations drawn from the gapped log's training rows; wave
amplitude 0.25 at 8 phases (and amplitude 0). Throttle of thruster 1 = -command[0] (it pushes against the current).
Second probe: the same push made by waves inside the recorded range (current 0.6, wave amplitude 0.4, 16 phases)
against the same push made by the current alone (no wave).

Usage: python actor_sweep.py <seed>
"""
import sys
import time

import numpy as np

from common import *  # noqa: F401,F403

C_GRID = np.round(np.arange(0.10, 1.2001, 0.05), 3)
PHASES = np.linspace(0, 2 * np.pi, 8, endpoint=False)
N_FLOATS = 64


def make_states(c_grid, floats, amp, phases):
    """(len(c_grid), len(floats) * len(phases), 13)."""
    out = []
    for c in c_grid:
        rows = []
        for f in floats:
            for ph in phases:
                rows.append(np.concatenate([f, [c, amp * np.cos(ph), amp * np.sin(ph)]]))
        out.append(rows)
    return np.asarray(out, np.float64)


def crossing(x, y, level=0.8):
    above = np.where(y > level)[0]
    if len(above) == 0:
        return None
    i = above[0]
    if i == 0:
        return float(x[0])
    return float(x[i - 1] + (level - y[i - 1]) * (x[i] - x[i - 1]) / (y[i] - y[i - 1]))


def slope(x, y, lo, hi):
    m = (x >= lo - 1e-9) & (x <= hi + 1e-9)
    return float(np.polyfit(x[m], y[m], 1)[0])


def main(rep):
    t0 = time.perf_counter()
    S = Seed(rep, worlds=("stall", "no_stall"))
    pg, pt = S.pg, S.prepared("no_stall")
    rng = np.random.default_rng([20261012, int(rep), 4402])
    floats_train = pg.states[rng.choice(pg.n, N_FLOATS, replace=False), :TC.DS]
    h2, _ = S.h2_actors()
    room, _, room_vals = S.room_actors()
    policies = {}
    for k in ("clone", "full", "gapped", "random", "td3bc_gapped", "host_swap_full", "host_swap_gapped"):
        if k in h2:
            policies[k] = L.mlp_policy(h2[k], pg, k)
    for k in ("no_stall_clone", "no_stall_full", "no_stall_gapped"):
        if k in h2:
            policies[k] = L.mlp_policy(h2[k], pt, k)
    for k in ("none", "harm_placed", "strength_matched_even", "amount_matched", "shuffled_0", "reversed", "cap2",
              "even_x1.5", "even_x2", "even_x3"):
        if k in room:
            policies[f"room_{k}"] = L.mlp_policy(room[k], pg, k)
    policies["ref_stall_aware"] = RC.stall_aware(S.physics)
    policies["ref_rising_thrust"] = RC.rising_thrust(S.physics)
    policies["ref_normal_driving"] = RC.hand_only(S.st.spec, S.physics)

    probes = {
        "rest_amp0.25": make_states(C_GRID, [np.zeros(TC.DS)], 0.25, PHASES),
        "rest_amp0": make_states(C_GRID, [np.zeros(TC.DS)], 0.0, [0.0]),
        "trainfloats_amp0.25": make_states(C_GRID, floats_train, 0.25, PHASES),
    }
    wave_ph = np.linspace(0, 2 * np.pi, 16, endpoint=False)
    wave_states = make_states([0.6], [np.zeros(TC.DS)], 0.4, wave_ph)[0]
    cur_states = make_states(np.round(np.arange(0.2, 1.0001, 0.05), 3), [np.zeros(TC.DS)], 0.0, [0.0])[:, 0]
    res = {}
    for name, pol in policies.items():
        r = {}
        for pname, X in probes.items():
            nc, m = X.shape[:2]
            a = np.clip(pol(X.reshape(-1, TC.DX)), -1, 1).reshape(nc, m, TC.DA)
            th1 = -a[..., 0]
            mean = th1.mean(1)
            r[pname] = dict(
                throttle1_mean=mean, throttle1_p90=np.quantile(th1, 0.9, axis=1),
                share_thruster1_past80=(np.abs(a[..., 0]) > TC.KNEE).mean(1),
                share_any_past80=TC.past_knee(a, S.physics).mean(1),
                crossing_c_mean_throttle=crossing(C_GRID, mean),
                crossing_c_half_past80=crossing(C_GRID, (np.abs(a[..., 0]) > TC.KNEE).mean(1), 0.5),
                slope_in_data=slope(C_GRID, mean, 0.1, 0.7), slope_out_of_data=slope(C_GRID, mean, 0.7, 1.2),
                throttle1_at_c0p5=float(mean[np.argmin(np.abs(C_GRID - 0.5))]),
                throttle1_at_c0p7=float(mean[np.argmin(np.abs(C_GRID - 0.7))]),
                throttle1_at_c0p9=float(mean[np.argmin(np.abs(C_GRID - 0.9))]),
                throttle1_at_c1p1=float(mean[np.argmin(np.abs(C_GRID - 1.1))]))
        aw = np.clip(pol(wave_states), -1, 1)
        ac = np.clip(pol(cur_states), -1, 1)
        r["push_by_wave"] = dict(push=TC.push(wave_states), throttle1=-aw[:, 0])
        r["push_by_current"] = dict(push=TC.push(cur_states), throttle1=-ac[:, 0])
        # matched push 0.9-1.0: wave-made (c = 0.6, w1 = 0.3-0.4) against current-made (c = 0.9-1.0, no wave)
        mw = (TC.push(wave_states) >= 0.9)
        mc = (TC.push(cur_states) >= 0.9) & (TC.push(cur_states) <= 1.0 + 1e-9)
        r["high_push_throttle1_wave_made"] = float(-aw[mw, 0].mean())
        r["high_push_throttle1_current_made"] = float(-ac[mc, 0].mean())
        res[name] = r
        log(f"{name:24s} c=0.5 {r['rest_amp0.25']['throttle1_at_c0p5']:.3f}  c=0.7 "
            f"{r['rest_amp0.25']['throttle1_at_c0p7']:.3f}  c=0.9 {r['rest_amp0.25']['throttle1_at_c0p9']:.3f}  "
            f"c=1.1 {r['rest_amp0.25']['throttle1_at_c1p1']:.3f}  cross {r['rest_amp0.25']['crossing_c_mean_throttle']}"
            f"  slope in {r['rest_amp0.25']['slope_in_data']:.2f} out {r['rest_amp0.25']['slope_out_of_data']:.2f}"
            f"  push~0.95 wave {r['high_push_throttle1_wave_made']:.3f} current {r['high_push_throttle1_current_made']:.3f}")
    write(f"actor_sweep_seed{rep}.json", dict(kind="anatomy-actor-sweep", rep=int(rep), c_grid=C_GRID,
                                               seconds=time.perf_counter() - t0, actors=res,
                                               room_points={s: v["points"] for s, v in room_vals["situations"].items()}))


if __name__ == "__main__":
    main(int(sys.argv[1]))

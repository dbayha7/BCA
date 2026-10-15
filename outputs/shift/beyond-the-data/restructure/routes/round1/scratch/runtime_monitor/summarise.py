"""Route 'runtime_monitor', phase 3: nets across the four worlds and the kill-rule verdict (KILL_RULE.md).
EXPLORATORY, designed after seeing seeds 0-4. Reads phase 1/2 files; writes summary.json beside them."""
import sys

sys.dont_write_bytecode = True

import json  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

OUT = Path("/home/user/bca/runs/shift_explore/routes/runtime_monitor")
STEP1 = Path("/home/user/bca/runs/shift_explore/restructure/step1/results.json")
SEEDS = (0, 1, 2, 3, 4)
GAIN = ("S", "M3")
LOSS = ("N", "M4")
ARMS = ("real", "permuted", "mann_whitney", "offline_bank_rest", "fleet", "clip_always", "oracle", "offline_step1")

looks = {s: json.loads((OUT / f"seed{s}" / "looks.json").read_text()) for s in SEEDS}
prep = {s: json.loads((OUT / f"seed{s}" / "prep.json").read_text()) for s in SEEDS}
st1 = json.loads(STEP1.read_text())["per_seed_world"]
off_cert = {"S": "stall", "N": "no_stall", "M3": "no_stall", "M4": "stall"}


def arm_points(s, key, arm):
    r = looks[s]["runs"][f"{key}/strong_current"]
    p = r["points"]
    if arm == "oracle":
        return p["clip_always"] if key in GAIN else p["never"]
    if arm == "offline_step1":
        on = bool(st1[f"{off_cert[key]}/seed{s}"]["certificate"]["on"])
        return p["clip_always"] if on else p["never"]
    return p[arm]


per_world, nets = {}, {}
for arm in ARMS:
    terms = {}
    for key in GAIN + LOSS:
        d = [arm_points(s, key, arm) - looks[s]["runs"][f"{key}/strong_current"]["points"]["never"] for s in SEEDS]
        terms[key] = float(np.mean(d)) if key in GAIN else float(-np.mean(d))   # gains, and losses as positive
        per_world.setdefault(arm, {})[key] = dict(per_seed_change=d, seed_mean=float(np.mean(d)))
    nets[arm] = dict(net=terms["S"] + terms["M3"] - terms["N"] - terms["M4"], gain_S=terms["S"], gain_M3=terms["M3"],
                     loss_N=terms["N"], loss_M4=terms["M4"])

latch = {key: {s: looks[s]["runs"][f"{key}/strong_current"]["latch_after_episode"]["real"] for s in SEEDS}
         for key in GAIN + LOSS}
c1 = any(latch[k][s] is not None for k in LOSS for s in SEEDS)
c2 = any(latch[k][s] is None or latch[k][s] > 5 for k in GAIN for s in SEEDS)
c3 = nets["real"]["net"] < 0.85 * nets["oracle"]["net"]
c4 = nets["permuted"]["net"] >= 0.5 * nets["real"]["net"]
rec_cost = [looks[s]["runs"]["N/recorded"]["points"]["never"] - looks[s]["runs"]["N/recorded"]["points"]["real"]
            for s in SEEDS]
summary = dict(
    label="EXPLORATORY route test (runtime_monitor), designed after seeing seeds 0-4; seeds 0-4 only; no training.",
    plumbing_all_true=all(all(prep[s]["checks"].values()) for s in SEEDS),
    latch_after_episode_real=latch,
    latch_after_episode_other={arm: {k: {s: looks[s]["runs"][f"{k}/strong_current"]["latch_after_episode"][arm]
                                         for s in SEEDS} for k in GAIN + LOSS}
                               for arm in ("permuted", "mann_whitney", "offline_bank_rest", "fleet")},
    max_q05_real_in_no_stall_physics={k: {s: looks[s]["runs"][f"{k}/strong_current"]["max_q05_real"] for s in SEEDS}
                                      for k in LOSS},
    at_switch={k: {s: looks[s]["runs"][f"{k}/strong_current"].get("at_switch") for s in SEEDS} for k in GAIN + LOSS},
    nets=nets, per_world=per_world,
    kill=dict(c1_switch_on_in_no_stall_physics=c1, c2_not_on_by_episode_6_in_stalling_physics=c2,
              c3_net_below_85pct_oracle=c3, c4_permuted_keeps_half=c4,
              shelve=bool(c1 or c2 or c3 or c4)),
    reported=dict(N_recorded_conditions_cost_per_seed=rec_cost, N_recorded_conditions_cost_mean=float(np.mean(rec_cost)),
                  N_recorded_latch={s: looks[s]["runs"]["N/recorded"]["latch_after_episode"]["real"] for s in SEEDS},
                  rank_test_within_5_of_bca=abs(nets["mann_whitney"]["net"] - nets["real"]["net"]) <= 5),
    cpu_seconds=dict(prep=sum(prep[s]["cpu_seconds"] for s in SEEDS), looks=sum(looks[s]["cpu_seconds"] for s in SEEDS)))
(OUT / "summary.json").write_text(json.dumps(summary, indent=1, default=float))
for arm in ARMS:
    n = nets[arm]
    print(f"{arm:18s} net {n['net']:7.1f}  gain S {n['gain_S']:6.1f}  gain M3 {n['gain_M3']:6.1f}  loss N "
          f"{n['loss_N']:6.1f}  loss M4 {n['loss_M4']:6.1f}")
print("latch real", latch)
print("max q05 in no-stall physics", summary["max_q05_real_in_no_stall_physics"])
print("kill", summary["kill"])
print("reported", summary["reported"])
print("cpu", summary["cpu_seconds"])

#!/bin/sh
# M11 (ramp height) real-data recompute, run in the PINNED environment (WSL Ubuntu-24.04, /home/user/bca,
# Python 3.10.20, numpy 2.2.6, jax 0.6.2 CPU) through the bridge wslrun.ps1:
#   pwsh -NoProfile -File <scratchpad>\wslrun.ps1 -ScriptFile <this file> -TimeoutSec 1500 -Name m11ramp
# Reads only: runs/shift_explore/near80/{policy,learners,data}/** (the near-80 version 8 run files) and the pinned
# code (experiments/shift/learners.py, outputs/shift/beyond-the-data/restructure/step2/step2lib.py).
# Writes only under /tmp/bridge/work/M11/. Nothing is trained.
set -e
mkdir -p /tmp/bridge/work/M11
cat > /tmp/bridge/work/M11/m11_recompute.py <<'PYEOF'
"""For every locked seed-world of the near-80 run: (1) recompute the height-0 plumbing check (the largest absolute
parameter difference between every height-0 ramp arm's saved actor and the saved no-caution actor), (2) read the ramp
penalty every saved actor would pay on the recorded training rows (every 4th row), under BCA's switched ramp and under
its own arm's height, with the critic's value at its commands. Output: one JSON per seed-world."""
import json
import sys as _s
_s.dont_write_bytecode = True
import sys
import time

sys.path.insert(0, "/home/user/bca")
sys.path.insert(0, "/home/user/bca/outputs/shift/beyond-the-data/restructure/step2")
import numpy as np  # noqa: E402
import h5py  # noqa: E402
import jax  # noqa: E402
from experiments.shift import learners as L  # noqa: E402
import step2lib as S  # noqa: E402

ROOT = "/home/user/bca/runs/shift_explore/near80"
OUT = "/tmp/bridge/work/M11"
CASES = ([("near80", w, s) for s in range(5) for w in ("stall", "no_stall")]
         + [("random_rows", w, s) for s in range(5) for w in ("stall", "no_stall")]
         + [("main", w, s) for s in (11, 14) for w in ("stall", "no_stall")])
ARMS = ["none", "w0_check", "obca_k1", "obca_k2", "obca_forced", "fixed_gated", "conformal_k1", "meangap_k1",
        "meandiff_k1", "knee_ramp", "range_hinge", "perfect_band", "clone"]
STRIDE = 4


def jl(p):
    with open(p) as f:
        return json.load(f)


def max_param_diff(a, b):
    return max(float(np.max(np.abs(np.asarray(x, np.float64) - np.asarray(y, np.float64))))
               for x, y in zip(jax.tree_util.tree_leaves(a), jax.tree_util.tree_leaves(b)))


def main():
    t0 = time.time()
    print("versions", sys.version.split()[0], np.__version__, jax.__version__, flush=True)
    for setting, world, seed in CASES:
        t1 = time.time()
        d = f"{ROOT}/policy/{setting}/{world}/seed{seed}"
        ld = f"{ROOT}/learners/{setting}/{world}/seed{seed}"
        lock = jl(d + "/lock.json")
        names = lock["names"]
        actors = L.load_tree(d + "/actors.msgpack", {n: L.actor_template() for n in names}, lock["actors_file_sha256"])
        rj = jl(ld + "/reference_actors.json")
        refs = L.load_tree(ld + "/reference_actors.msgpack", {n: L.actor_template() for n in rj["names"]},
                           rj["file_sha256"])
        none = refs["none"]
        cj = jl(ld + "/critic.json")
        critic = L.load_tree(ld + "/critic.msgpack", L.critic_template(), cj["file_sha256"])
        data = cj["result"]["data"]
        mean = np.asarray(data["obs_mean"], np.float32)
        std = np.asarray(data["obs_std"], np.float32)
        with h5py.File(f"{ROOT}/data/{setting}/{world}/seed{seed}/train.hdf5", "r") as f:
            obs = np.asarray(f["observations"][:], np.float32)
            act = np.asarray(f["actions"][:], np.float32)
        rows = np.arange(0, len(obs), STRIDE)
        on = (obs[rows] - mean) / std
        a_rec = act[rows].astype(np.float64)
        at = lock["arm_table"]
        t_lo, t_up = float(at["t_lo"]), float(at["t_up"])
        prim = at["arms"]["obca_k1"]
        # (1) plumbing: every height-0 ramp arm and w0_check against the saved no-caution actor
        plumbing = {}
        for n in names:
            if n == "w0_check" or (n in at["arms"] and float(at["arms"][n]["h"]) == 0.0):
                plumbing[n] = max_param_diff(actors[n], none)
        # the recorded commands and the critic at them
        t_rec = S.throttle(a_rec)
        q_rec = L.critic_rows(critic, on, a_rec.astype(np.float32))[0][:, 0]
        recorded = dict(share_past_t_lo=float((t_rec > t_lo).mean()),
                        share_strictly_between=float(((t_rec > t_lo) & (t_rec < t_up)).mean()),
                        share_at_least_t_up=float((t_rec >= t_up).mean()),
                        share_past_80=float((t_rec > 0.8).mean()),
                        q1_mean=float(q_rec.mean()))
        # (2) penalties at every saved actor's commands
        out = {}
        for n in ARMS:
            if n == "none":
                p = none
            elif n in actors:
                p = actors[n]
            else:
                continue
            cmd = L.actor_rows(p, on).astype(np.float64)
            t = S.throttle(cmd)
            q = L.critic_rows(critic, on, cmd.astype(np.float32))[0][:, 0]
            P = S.ramp_penalty(cmd, a_rec, float(prim["h"]), float(prim["k"]), t_lo, t_up)
            rec = dict(share_past_t_lo=float((t > t_lo).mean()), share_past_80=float((t > 0.8).mean()),
                       share_at_least_t_up=float((t >= t_up).mean()), mean_throttle=float(t.mean()),
                       q1_mean=float(q.mean()), q1_abs_mean=float(np.abs(q).mean()),
                       bca_ramp_h=float(prim["h"]), bca_ramp_k=float(prim["k"]),
                       penalty_under_bca_ramp_mean=float(P.mean()),
                       penalty_under_bca_ramp_share_pos=float((P > 0).mean()),
                       penalty_under_bca_ramp_max=float(P.max()))
            own = at["arms"].get(n)
            if own is not None:
                lo, up = float(own.get("t_lo", t_lo)), float(own.get("t_up", t_up))
                Po = S.ramp_penalty(cmd, a_rec, float(own["h"]), float(own["k"]), lo, up)
                rec.update(own_h=float(own["h"]), own_k=float(own["k"]), own_t_lo=lo, own_t_up=up,
                           own_penalty_mean=float(Po.mean()), own_penalty_share_pos=float((Po > 0).mean()),
                           own_penalty_max=float(Po.max()))
            out[n] = rec
        kw = None
        if "obca_k1" in lock["arms"]:
            e = lock["arms"]["obca_k1"]
            i = e["metric_names"].index("kW_mean")
            kw = [c[i] for c in e["chunk_metrics"]]
        res = dict(setting=setting, world=world, seed=seed, rows_total=int(len(obs)), rows_used=int(len(rows)),
                   stride=STRIDE, t_lo=t_lo, t_up=t_up, arm_table=at, recorded=recorded,
                   plumbing_recomputed=plumbing,
                   plumbing_recorded=dict(b_w0=lock["checks"]["b_w0"], h0=lock["checks"]["h0_reproduces_reference"]),
                   training_kW_mean_per_chunk_obca_k1=kw, arms=out, seconds=time.time() - t1)
        with open(f"{OUT}/{setting}_{world}_seed{seed}.json", "w") as f:
            json.dump(res, f, indent=1, sort_keys=True)
        print(f"{setting}/{world}/seed{seed}: rows {len(rows)} plumbing {plumbing} "
              f"none past t_lo {out['none']['share_past_t_lo']:.3f} obca_k1 past t_lo "
              f"{out.get('obca_k1', {}).get('share_past_t_lo', float('nan')):.3f} "
              f"P(none|bca ramp) {out['none']['penalty_under_bca_ramp_mean']:.4f} "
              f"P(obca_k1|own) {out.get('obca_k1', {}).get('own_penalty_mean', float('nan')):.5f} "
              f"{time.time() - t1:.0f}s", flush=True)
    print("total seconds", round(time.time() - t0))


if __name__ == "__main__":
    main()
PYEOF
cd /home/user/bca
PYTHONPATH=/home/user/bca PYTHONDONTWRITEBYTECODE=1 JAX_PLATFORMS=cpu /tmp/claude-0/bcavenv/bin/python /tmp/bridge/work/M11/m11_recompute.py

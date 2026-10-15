"""READ-ONLY re-evaluation of SAVED Stage 2 policy-test actors (forward passes and exact rollouts of saved actors only;
no training, no band or classifier fit, no run_stage2 phase). Writes only to this scratchpad folder.

Per seed: the actors locked in runs/shift_explore/stage2/policy/seed<K>/actors.msgpack (file SHA-256 and every actor's
tree SHA-256 checked against lock.json). For each scored arm:
  (1) at the gapped log's training states in the full-throttle region (pg.region) and at push > 0.8: share of
      thruster-1 commands past 80%, mean |command 1|, mean |command 1| given past 80%;
  (2) thruster-1 sweep against the current (floats at rest, wave amplitude 0.25, 8 phases; and 64 training floats);
  (3) strong current, exact rollouts from the first N starts of the bank: share past 80%, and depth given past 80%.
Usage: python actor_readout.py <seed> [n_starts]
"""
import json
import sys
import time
from pathlib import Path

import numpy as np

import experiments.shift.acceptance as A
import experiments.shift.learners as L
import experiments.shift.thruster_chain as TC
import experiments.shift.truth as TR

OUT = Path("/tmp/claude-0/-home-user/9e341792-006d-5a11-a8e9-8fac16ac7f5b/scratchpad/impact")
POL = Path("/home/user/bca/runs/shift_explore/stage2/policy")
C_GRID = np.round(np.arange(0.30, 1.2001, 0.1), 3)
PHASES = np.linspace(0, 2 * np.pi, 8, endpoint=False)


def make_states(c_grid, floats, amp, phases):
    out = []
    for c in c_grid:
        rows = []
        for f in floats:
            for ph in phases:
                rows.append(np.concatenate([f, [c, amp * np.cos(ph), amp * np.sin(ph)]]))
        out.append(rows)
    return np.asarray(out, np.float64)


def stats(a0):
    m = np.abs(a0) > TC.KNEE
    return dict(n=int(len(a0)), past80=float(m.mean()), mean_abs=float(np.abs(a0).mean()),
                depth_given_past80=float(np.abs(a0[m]).mean()) if m.any() else None,
                share_ge_0p95=float((np.abs(a0) >= 0.95).mean()))


def main(rep, n_starts):
    t0 = time.perf_counter()
    prereg = A.load_prereg(A.refuse_protected(A.RUNS_DEFAULT))
    A.guard(prereg)
    cfg = prereg.config
    variant = A.gate_outcome(prereg)["variant"]
    sd = A.load_seed(prereg.runs, cfg, variant, "stall", rep)
    pg = sd.prepared("gap", cfg.learn)
    physics = sd.physics
    lock = json.loads((POL / f"seed{rep}" / "lock.json").read_text())
    acts = L.load_tree(POL / f"seed{rep}" / lock["actors_file"], {k: L.actor_template() for k in lock["names"]},
                       lock["actors_file_sha256"])
    names = dict(none="none", full_none="full_none", clone="clone", bca_k1="bca_k1", bca_k2="bca_k2",
                 bca_unpaired="bca_unpaired", bca_shuffled="bca_shuffled", perfect_band="perfect_band",
                 range_hinge="range_hinge", hinge_matched=lock["standin"]["arm"],
                 even_pull=lock["matched_names"]["even_pull"])
    for k, n in names.items():
        if L.tree_sha256(acts[n]) != lock["actor_sha256"][n]:
            raise SystemExit(f"actor {n} differs from the lock")
    obs_all = np.asarray(pg.train.obs)
    reg = np.asarray(pg.region, bool)
    hi = TC.push(pg.states) > 0.8
    rng = np.random.default_rng([20261014, rep, 7])
    floats_train = pg.states[rng.choice(pg.n, 64, replace=False), :TC.DS]
    probes = {"rest_amp0.25": make_states(C_GRID, [np.zeros(TC.DS)], 0.25, PHASES),
              "trainfloats_amp0.25": make_states(C_GRID, floats_train, 0.25, PHASES[::2])}
    starts = TC.bank_starts("strong_current", TC.BANK_SIZE)[:n_starts]
    res = dict(rep=rep, n_starts=n_starts, c_grid=C_GRID.tolist(), region_rows=int(reg.sum()), push_gt_0p8_rows=int(hi.sum()),
               arms={})
    for k, n in names.items():
        a = L.actor_rows(acts[n], obs_all)[:, 0].astype(np.float64)
        r = dict(arm=n, region=stats(a[reg]), push_gt_0p8=stats(a[hi]), outside_region=stats(a[~reg]))
        pol = L.mlp_policy(acts[n], pg, n)
        sw = {}
        for pname, X in probes.items():
            nc, m = X.shape[:2]
            cmd = np.clip(pol(X.reshape(-1, TC.DX)), -1, 1).reshape(nc, m, TC.DA)[..., 0]
            sw[pname] = dict(throttle1_mean=(-cmd).mean(1).tolist(), past80=(np.abs(cmd) > TC.KNEE).mean(1).tolist())
        r["sweep"] = sw
        run = TR.rollout(pol, starts, physics, record=True)
        c0 = np.asarray(run.commands)[..., 0].reshape(-1)
        r["strong_current_rollout"] = stats(c0)
        res["arms"][k] = r
        print(f"seed {rep} {k:14s} region past80 {r['region']['past80']:.3f} depth {r['region']['depth_given_past80']} "
              f"| push>0.8 past80 {r['push_gt_0p8']['past80']:.3f} | SC past80 {r['strong_current_rollout']['past80']:.3f} "
              f"depth {r['strong_current_rollout']['depth_given_past80']:.3f} >=0.95 {r['strong_current_rollout']['share_ge_0p95']:.3f} "
              f"({time.perf_counter() - t0:.0f}s)", flush=True)
    res["seconds"] = time.perf_counter() - t0
    (OUT / f"actor_readout_seed{rep}.json").write_text(json.dumps(res))


if __name__ == "__main__":
    main(int(sys.argv[1]), int(sys.argv[2]) if len(sys.argv) > 2 else 1000)

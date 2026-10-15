"""S0-ONLY VARIANT for gate seeds 3-4 (no trace critics exist there): the gate's gapped and full-log critics only,
row groups bank, surv, t4, kept (no trace candidates); also saves the T4 log rows. Otherwise as below.
READ-ONLY evidence E1 (Angle D, restructure design). Held-out against in-sample one-step residuals at recorded
past-80% rows, on the SAVED trace-test-bed critics (runs/shift_explore/trace, seeds 0-2, shares 0-10%) and the
gate's gapped and full-log fixed-controller critics.

No training, no fit, no BCA module is called (no calibration/*, no td3_bc_bca, no learners.bca_*). Forward passes of
saved critics on recorded log rows only. Writes only to this folder.

The residual is the critic's own training form, exactly as the trace test bed defines it (signals.py docstring):
  delta_i = r_i + 0.98 (1 - done_i) mean_d min_h Q_h(s'_i, clip(c(s'_i) + eps_id, -1, 1)) - min_h Q_h(s_i, a_i)
with the trace bed's own noise draws (signals.target_noise_draw), c the stall-aware controller.

Row groups (log positions; the shared candidates are the SAME positions in both worlds, so worlds are paired):
  cand      the gate's removed training rows past 80% that both worlds have (trace selection pool, ~6,600)
  bankcand  the gate's removed calibration-bank rows past 80% that both worlds have (~130)
  t4        5,000 held-out evaluation-split rows outside the gap (the trace bed's T4)
  bank      the gate's calibration bank (~851 rows, held out from every critic)
  surv      kept training rows past 80% that survived the gate's removal (34-59; in-sample for every critic)
  kept      5,000 random kept training rows (in-sample, ordinary)
Critics: gap (gate), full (gate), trace_<share> for the six trace shares (critic trained on gate kept + added rows).
"""
import json
import sys
import time

import numpy as np

sys.path.insert(0, "/home/user/bca/runs/shift_explore/trace/code")
import tracebed as T  # noqa: E402
import signals as S  # noqa: E402

L, A = T.L, T.A
import jax.numpy as jnp  # noqa: E402

OUT = "/tmp/claude-0/-home-user/9e341792-006d-5a11-a8e9-8fac16ac7f5b/scratchpad/restructure/testbed"
DRAWS = 32


def trace_critic(world, rep, share):
    d = T.job_dir(world, rep, share)
    lock = json.load(open(d / "lock.json"))
    return L.load_tree(d / "critic.msgpack", L.critic_template(), lock["trace_critic_file_sha256"])


def run(rep):
    t0 = time.time()
    g = T.GateSeed(rep)
    sds = g.sd
    out = {}
    for world in T.WORLDS:
        sd = sds[world]
        lg = sd.log
        prep = g.gapped_prepared(world)
        mean, std = prep.obs_mean, prep.obs_std
        n_log = len(lg["actions"])
        rg = sd.removals["gap"]
        kept = rg.kept_train_rows(lg)
        surv = kept[T.past80(lg["actions"][kept], sd.physics)]
        rng = np.random.default_rng([20261014, rep, 501])
        kept_s = np.sort(kept[rng.choice(len(kept), 5000, replace=False)])
        t4rows = S.t4_rows(sd)
        groups = dict(t4=t4rows, bank=rg.bank_rows(lg), surv=surv, kept=kept_s)
        allrows = np.unique(np.concatenate(list(groups.values())))
        pos = {k: np.searchsorted(allrows, v) for k, v in groups.items()}
        x = np.asarray(lg["observations"][allrows], np.float32)
        a = np.asarray(lg["actions"][allrows], np.float32)
        r = np.asarray(lg["rewards"][allrows], np.float64)
        x1 = np.asarray(lg["next_observations"][allrows], np.float32)
        dn = np.asarray(lg["terminals"][allrows], np.float64)
        obs, nxt = (x - mean) / std, (x1 - mean) / std
        fixed = L.FixedController("stall_aware", sd.physics, mean, std)
        c = np.asarray(fixed.apply(fixed.params(), jnp.asarray(nxt)), np.float32)
        noisy = [S.noisy_next_commands(c, S.target_noise_draw(rep, d, allrows, n_log), 0.2, 0.5) for d in range(DRAWS)]
        critics = {"gap": g.critic(world, "gap")[0], "full": g.critic(world, "full")[0]}
        deltas = {}
        for name, cp in critics.items():
            q = L.critic_rows(cp, obs, a)[0].min(1)
            plain = L.critic_rows(cp, nxt, c)[0].min(1)
            m = np.zeros(len(allrows))
            for d in range(DRAWS):
                m += L.critic_rows(cp, nxt, noisy[d])[0].min(1)
            m /= DRAWS
            disc = 0.98 * (1.0 - dn)
            deltas[name] = dict(trained=r + disc * m - q, noiseless=r + disc * plain - q, q=q)
        added, added_bank = {}, {}
        np.savez(f"{OUT}/heldout_seed{rep}_{world}.npz", rows=allrows, actions=a, states=np.asarray(lg["infos/state"][allrows]),
                 **{f"pos_{k}": v for k, v in pos.items()},
                 **{f"delta_{n}": d["trained"] for n, d in deltas.items()},
                 **{f"deltanl_{n}": d["noiseless"] for n, d in deltas.items()},
                 **{f"q_{n}": d["q"] for n, d in deltas.items()},
                 t4_rows=t4rows)
        out[world] = dict(n=dict((k, int(len(v))) for k, v in groups.items()),
                          added=dict((k, int(len(v))) for k, v in added.items()),
                          added_bank=dict((k, int(len(v))) for k, v in added_bank.items()))
        print(rep, world, out[world], f"{time.time() - t0:.0f}s", flush=True)
    return out


if __name__ == "__main__":
    reps = [int(s) for s in sys.argv[1:]] or [0, 1, 2]
    meta = {rep: run(rep) for rep in reps}
    json.dump(meta, open(f"{OUT}/heldout_s0_meta_{'_'.join(map(str, reps))}.json", "w"), indent=1)

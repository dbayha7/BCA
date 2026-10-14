"""READ-ONLY forward passes. delta_j = r_j + gamma min_h Q(s'_j, c(s'_j)) - min_h Q(s_j, a_j) of the SAVED frozen
gapped fixed-controller critic (A.get_critic(train=False)), c = the stall-aware controller, at every kept training
row and every calibration-bank row, both worlds (the residual_evidence stand-in's one_step_errors, restated, and the
same quantity BCA's score uses without TD3's smoothing noise). No training, no fitting. Usage: python e3_delta.py <seed>"""
import sys
import time
import numpy as np
import jax.numpy as jnp

sys.path.insert(0, "/home/user/bca")
import experiments.shift.acceptance as A  # noqa: E402
import experiments.shift.learners as L  # noqa: E402

OUT = "/tmp/claude-0/-home-user/9e341792-006d-5a11-a8e9-8fac16ac7f5b/scratchpad/restructure/calibration"
p = A.load_prereg(A.refuse_protected(A.RUNS_DEFAULT))
A.guard(p)
variant = A.gate_outcome(p)["variant"]
rep = int(sys.argv[1])
for world in ("stall", "no_stall"):
    t = time.time()
    sd = A.load_seed(p.runs, p.config, variant, world, rep)
    pg = sd.prepared("gap", p.config.learn)
    rec, critic = A.get_critic(p, sd, "gap", "stall_aware", train=False)
    fixed = L.FixedController("stall_aware", pg.physics, pg.obs_mean, pg.obs_std)
    out = {}
    for name, tr in (("train", pg.train), ("bank", pg.bank)):
        nxt = np.asarray(tr.next_obs, np.float32)
        a_next = np.asarray(fixed.apply(fixed.params(), jnp.asarray(nxt)), np.float32)
        qn = L.critic_rows(critic, nxt, a_next)[0].min(1)
        q = L.critic_rows(critic, np.asarray(tr.obs), np.asarray(tr.action))[0].min(1)
        out[f"{name}_delta"] = np.asarray(tr.reward, np.float64) + pg.physics.gamma * qn - q
        out[f"{name}_q"] = q
    np.savez(f"{OUT}/delta_seed{rep}_{world}.npz", gamma=pg.physics.gamma, **out)
    print(rep, world, f"{time.time()-t:.1f}s", float(np.std(out["train_delta"])), float(np.std(out["bank_delta"])), flush=True)

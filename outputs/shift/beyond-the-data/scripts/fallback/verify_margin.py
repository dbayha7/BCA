"""EXPLORATORY check for the write-up: how far the gapped actor's past-80% proposals in strong current lie outside the
commands recorded at the 10 nearest training states (seed 0, 200 starts, actor alone)."""
import numpy as np

import experiments.shift.acceptance as A
import experiments.shift.learners as L
import experiments.shift.thruster_chain as TC
import experiments.shift.truth as TR
import sys
sys.path.insert(0, "/tmp/claude-0/-home-user/9e341792-006d-5a11-a8e9-8fac16ac7f5b/scratchpad/ood/fallback")
import fallback as F  # noqa: E402

prereg = A.load_prereg(A.refuse_protected(A.RUNS_DEFAULT))
cfg = prereg.config
sd = A.load_seed(prereg.runs, cfg, "base", "stall", 0)
pg = sd.prepared("gap", cfg.learn)
d = A._variant_root(prereg, "harm_return", "base") / "alpha5" / "seed0"
rec = A.finished(d / "values.json", prereg)
acts = L.load_tree(d / "actors.msgpack", {k: L.actor_template() for k in rec["actor_names"]}, rec["actors_sha256"])
ref = F.Reference(sd, "/tmp/claude-0/-home-user/9e341792-006d-5a11-a8e9-8fac16ac7f5b/scratchpad/ood/fallback/cache")
pol = L.mlp_policy(acts["gapped"], pg)
run = TR.rollout(pol, TC.bank_starts("strong_current")[:200], sd.physics, record=True)
x = run.states[:, :-1].reshape(-1, TC.DX)
a = run.commands.reshape(-1, TC.DA)
past = TC.past_knee(a)
xp, ap = x[past], a[past]
print("past-80% share", past.mean(), "thruster-1 command quantiles", np.quantile(ap[:, 0], [0.1, 0.5, 0.9]).round(3))
print("which thrusters past 80%", (np.abs(ap) > TC.KNEE).mean(0).round(3))
idx = ref.state.tree.query(ref.state.z(xp), k=10)[1]
C = ref.actions[idx]
print("local min of thruster-1 recorded commands, quantiles", np.quantile(C[:, :, 0].min(1), [0.1, 0.5, 0.9]).round(3))
sc = ref.local.score(xp, ap)
print("nearby-commands score quantiles", np.quantile(sc, [0.1, 0.5, 0.9]).round(3), "threshold", ref.local.thr)
dsa = ref.sa.dist(np.concatenate([xp, ap], 1))
dsc = ref.sa.dist(np.concatenate([xp, np.clip(ap, ref.lo99, ref.hi99)], 1))
print("(state, command) distance quantiles", np.quantile(dsa, [0.1, 0.5, 0.9]).round(2), "same states clipped command",
      np.quantile(dsc, [0.1, 0.5, 0.9]).round(2), "threshold", ref.sa.thr)

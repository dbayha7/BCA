"""Read-only probe (no BCA, nothing written under the repository): plain TD3+BC (learners.cotrain) for gate seed 0,
stalling world, gapped log, first N updates; prints the parameter SHA-256 only."""
import json, os, sys, time
sys.path.insert(0, "/home/user/bca")
import experiments.shift.acceptance as A
import experiments.shift.learners as L
n = int(sys.argv[1])
payload = json.load(open(A.RUNS_DEFAULT / A.PREREG_NAME))
cfg = A.config_from_dict(payload["config"])
sd = A.load_seed(A.RUNS_DEFAULT, cfg, "base", "stall", 0)
pg = sd.prepared("gap", cfg.learn)
t = time.perf_counter()
rec, state, _ = L.cotrain(pg, cfg.learn, 0, alpha=5.0, updates=n, log=lambda *a: None)
print(json.dumps(dict(xla_flags=os.environ.get("XLA_FLAGS"), updates=n, failed=rec.get("failed"),
                      actor=L.tree_sha256(state.actor.params)[:16], critic=L.tree_sha256(state.critic.params)[:16],
                      seconds=round(time.perf_counter() - t, 1))), flush=True)

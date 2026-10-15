"""Read-only probe (no BCA, nothing written under the repository): check 2's actor-only call for gate seed 0, stalling
world, gapped critic, from the gate's saved clone; prints the result and saves the parameters to the scratchpad only."""
import json, os, sys, time
sys.path.insert(0, "/home/user/bca")
import numpy as np
import experiments.shift.acceptance as A
import experiments.shift.learners as L
import jax
steps, tag = int(sys.argv[1]), sys.argv[2]
payload = json.load(open(A.RUNS_DEFAULT / A.PREREG_NAME))
digest = __import__("hashlib").sha256((A.RUNS_DEFAULT / A.PREREG_NAME).read_bytes()).hexdigest()
cfg = A.config_from_dict(payload["config"])
learn = cfg.learn
sd = A.load_seed(A.RUNS_DEFAULT, cfg, "base", "stall", 0)
pg = sd.prepared("gap", learn)
root = A.RUNS_DEFAULT / digest[:16]
cr = json.load(open(root / "critics/base/stall/stall_aware_gap/seed0/record.json"))
critic = L.load_tree(root / "critics/base/stall/stall_aware_gap/seed0/critic.msgpack", L.critic_template(), cr["result"]["file_sha256"])
clr = json.load(open(root / "clone/base/stall/seed0/record.json"))
clone = L.load_tree(root / "clone/base/stall/seed0/actor.msgpack", L.actor_template(), clr["result"]["file_sha256"])
t = time.perf_counter()
ao = L.ActorOnly(pg, clone, learn, 0, alpha=5.0, steps=steps, chunk=min(steps, learn.actor_only_chunk))
stacked, info = ao.run([L.Arm("gapped", critic=0)], [critic], weighted=False)
p = ao.arm_params(stacked, 0)
leaves = [np.asarray(x) for x in jax.tree_util.tree_leaves(p)]
out = dict(tag=tag, steps=steps, xla_flags=os.environ.get("XLA_FLAGS"), sha=L.tree_sha256(p)[:16], seconds=round(time.perf_counter() - t, 1))
if steps == learn.actor_only_steps:
    h2 = root / "harm_return/base/alpha5/seed0"
    rec = json.load(open(h2 / "values.json"))
    acts = L.load_tree(h2 / "actors.msgpack", {k: L.actor_template() for k in rec["actor_names"]}, rec["actors_sha256"])
    g = acts["gapped"]
    out["max_abs_diff_vs_gate_gapped"] = max(float(np.max(np.abs(np.asarray(a, np.float64) - np.asarray(b, np.float64))))
                                             for a, b in zip(jax.tree_util.tree_leaves(p), jax.tree_util.tree_leaves(g)))
    obs = np.asarray(pg.eval_obs)
    out["strength_vs_gate_gapped"] = float(L.strength(p, g, pg.eval_obs))
np.savez(f"/tmp/claude-0/-home-user/9e341792-006d-5a11-a8e9-8fac16ac7f5b/scratchpad/isa_probe/{tag}_{steps}.npz", *leaves)
print(json.dumps(out), flush=True)

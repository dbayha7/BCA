"""Feasibility review probe (cloud, no BCA): does the spec's P1a / P2 call reproduce the gate's saved actors bit for
bit on the cloud, and how long does it take? Writes only under the scratchpad."""
import json, sys, time
from pathlib import Path
import numpy as np
import jax
import experiments.shift.acceptance as A
import experiments.shift.learners as L

OUT = Path("/tmp/claude-0/-home-user/9e341792-006d-5a11-a8e9-8fac16ac7f5b/scratchpad/feas_review/out")
mode, s = sys.argv[1], int(sys.argv[2])
t0 = time.perf_counter()
prereg = A.load_prereg(A.refuse_protected(A.RUNS_DEFAULT)); A.guard(prereg)
cfg, learn = prereg.config, prereg.config.learn
t_pre = time.perf_counter() - t0

def clone_read(world):
    d = A._variant_root(prereg, "clone", "base") / world / f"seed{s}"
    rec = A.finished(d / "record.json", prereg)
    return L.load_tree(d / "actor.msgpack", L.actor_template(), rec["result"]["file_sha256"])

def gate_h2():
    d = A._variant_root(prereg, "harm_return", "base") / "alpha5" / f"seed{s}"
    rec = A.finished(d / "values.json", prereg)
    return L.load_tree(d / "actors.msgpack", {k: L.actor_template() for k in rec["actor_names"]}, rec["actors_sha256"])

def mad(a, b):
    return max(float(np.max(np.abs(np.asarray(x, np.float64) - np.asarray(y, np.float64))))
               for x, y in zip(jax.tree_util.tree_leaves(a), jax.tree_util.tree_leaves(b)))

res = dict(mode=mode, seed=s, prereg_seconds=t_pre)
if mode in ("stall", "no_stall"):
    t1 = time.perf_counter()
    sd = A.load_seed(prereg.runs, cfg, "base", mode, s)
    pg = sd.prepared("gap", learn)
    rems = ("gap", "none", "random") if mode == "stall" else ("gap", "none")
    names = ("gapped", "full", "random") if mode == "stall" else ("no_stall_gapped", "no_stall_full")
    crit = [A.get_critic(prereg, sd, r, "stall_aware", train=False)[1] for r in rems]
    clone = clone_read(mode)
    res["load_seconds"] = time.perf_counter() - t1
    t2 = time.perf_counter()
    ao = L.ActorOnly(pg, clone, learn, s, alpha=5.0)
    stacked, info = ao.run([L.Arm(n, critic=i) for i, n in enumerate(names)], crit, weighted=False)
    jax.block_until_ready(stacked)
    res["train_seconds"] = time.perf_counter() - t2
    gate = gate_h2()
    for i, n in enumerate(names):
        p = ao.arm_params(stacked, i)
        res[n] = dict(laptop_sha=L.tree_sha256(p), gate_sha=L.tree_sha256(gate[n]), max_abs_diff=mad(p, gate[n]),
                      failed=bool(info["failed"][i]))
        res[n]["identical"] = res[n]["laptop_sha"] == res[n]["gate_sha"]
    blob = L.save_tree(OUT / f"check2_{mode}_seed{s}.msgpack", {n: ao.arm_params(stacked, i) for i, n in enumerate(names)})
    res["file_sha256"] = blob
elif mode == "host":
    t1 = time.perf_counter()
    sd = A.load_seed(prereg.runs, cfg, "base", "stall", s)
    pg = sd.prepared("gap", learn)
    res["load_seconds"] = time.perf_counter() - t1
    t2 = time.perf_counter()
    r, state, _ = L.cotrain(pg, learn, s, alpha=5.0, table=None, stream=0, out_dir=OUT / f"host_plain_seed{s}",
                            name="TD3+BC plain (stall, gap log)")
    res["train_seconds"] = time.perf_counter() - t2
    grec, gstate = A.get_host(prereg, sd, "gap", 5.0, "plain", train=False)
    res["actor"] = dict(laptop_sha=L.tree_sha256(state.actor.params), gate_sha=L.tree_sha256(gstate.actor.params),
                        max_abs_diff=mad(state.actor.params, gstate.actor.params))
    res["actor"]["identical"] = res["actor"]["laptop_sha"] == res["actor"]["gate_sha"]
    res["critic_identical"] = L.tree_sha256(state.critic.params) == L.tree_sha256(gstate.critic.params)
    res["snap_file_sha"] = [x.get("file_sha256") for x in r["snapshots"]]
    res["gate_snap_file_sha"] = [x.get("file_sha256") for x in grec["result"]["snapshots"]]
    res["failed"] = r["failed"]
res["total_seconds"] = time.perf_counter() - t0
(OUT / f"result_{mode}_seed{s}.json").write_text(json.dumps(res, indent=1, sort_keys=True))
print(json.dumps(res, indent=1, sort_keys=True))

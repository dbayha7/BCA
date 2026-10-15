"""Information-only repeat of the feasibility review's cloud run (V7_SPEC.md C7 "Cloud evidence"). NOT A v7 RESULT.

The review's script (../cloud_review_run/p_probe.py, SHA-256 in REVIEW_SCRIPT below) ran the spec's P1a (the gate's
check-2 call, both worlds) and P2 (the gate's plain TD3+BC host call) for gate seed 2 on the cloud's Xeon and found
every actor bit-identical to the gate's saved one. It did not record how many processors each process saw, and its
file times show that its stalling-world and host runs overlapped. The spec asks for a repeat on one processor. This is
that repeat: the review's computation line for line, on processor 0 only (the driver uses taskset -c 0), with
launch_v7.sh's eight variables and nothing else, the gate's checks unchanged (software check on) and nothing
substituted (as in the review's run).

Changed from the review's script in these ways only:
  * the model files go to --outdir (a scratchpad folder) instead of the review's own folder, and the result JSON is
    written once, with a sidecar, to --record;
  * the environment is checked before anything runs (exactly the eight variables, one processor, /home/user/bca), and
    the record holds the processor affinity, the environment and the processor facts;
  * jax.block_until_ready on the host run's final state before its time is taken (the review timed the stalling-world
    and no-stall runs this way already).

Usage (from /home/user/bca): python p_probe_cpu0.py stall|no_stall|host 2 --outdir DIR --record OUT.json
"""
import sys

sys.dont_write_bytecode = True
import argparse  # noqa: E402
import hashlib  # noqa: E402
import os  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent.parent))
import xla_emul_common as X  # noqa: E402

REVIEW_SCRIPT = dict(path="scratchpad/feas_review/p_probe.py (copy: ../cloud_review_run/p_probe.py)",
                     sha256="e30396adbbdce6fdb0790c0175e6a0c33fa156639386943ff9203bea5b861c4e")
ap = argparse.ArgumentParser()
ap.add_argument("mode", choices=("stall", "no_stall", "host"))
ap.add_argument("seed", type=int)
ap.add_argument("--outdir", required=True)
ap.add_argument("--record", required=True)
a = ap.parse_args()
OUT, REC = Path(a.outdir), Path(a.record)
mode, s = a.mode, a.seed
res = dict(kind="stage2-v7-evidence-feasibility-repeat", information_only=X.INFO, mode=mode, seed=s,
           argv=sys.argv[1:], script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
           repeats=REVIEW_SCRIPT, constants="none substituted (as in the review's run; the cloud's own A, B, K)",
           gate_software_check="unchanged (acceptance.check_software as written)",
           affinity=sorted(os.sched_getaffinity(0)))
problems = X.check_mode("default")
if problems:
    X.refuse(REC, res, "; ".join(problems))
res["environment"] = X.env_facts("default")
OUT.mkdir(parents=True, exist_ok=True)

import numpy as np  # noqa: E402
import jax  # noqa: E402
import experiments.shift.acceptance as A  # noqa: E402
import experiments.shift.learners as L  # noqa: E402

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


res.update(prereg_seconds=t_pre)
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
    jax.block_until_ready(state.actor.params)
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
res["affinity_at_end"] = sorted(os.sched_getaffinity(0))
res["note_on_names"] = "'laptop_sha' is the review script's field name; here it is this cloud run's actor"
X.write(REC, res)
print(f"{mode} seed {s}: done in {res['total_seconds']:.0f} s")
sys.stdout.flush()
os._exit(0)

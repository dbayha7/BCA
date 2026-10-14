"""Information-only probe (V7_SPEC.md C1 "Still to do": rerun xla_repro.py and save its output). NOT A v7 RESULT: it
runs on the cloud's Xeon outside the v7 wrapper, before anything is pinned; the laptop's self-test replaces it.

Question: does the gate's own unchanged actor-only program (learners.ActorOnly, check 2's swap for the stalling world:
the gapped, full and random critics side by side, GATE/acceptance.py:1459-1464, :1475-1483) reproduce check 2's SAVED
actors of one gate seed bit for bit, (a) on an emulated AVX2-only processor and (b) on the cloud's processor as it is
(the control)? This is the arithmetic behind Stage 2's check (b) (1e-5 on every parameter) and behind the spec's P1a.
No BCA code runs. It writes only the JSON given by --out (and its .sha256 sidecar), once.

This is a copy of the earlier session's scratchpad/v7_hw_probe/xla_repro.py (byte copy beside it:
xla_repro_original.py, SHA-256 in ORIGINAL below), changed in these ways only:
  * any gate seed (--seed; the original was fixed at seed 0) and a declared mode (--mode default | emulated), with the
    environment checked against the mode before anything runs (xla_emul_common.check_mode) and the emulation
    confirmed after numpy is imported (no AVX-512 dispatch in numpy, Haswell kernels in both OpenBLAS libraries);
  * the constants A, B and K come from V7/world_constants.json, substituted by V7/code/v7_constants.apply exactly as
    the v7 wrapper does (file SHA-256 and sidecar, local differences, replacement, tripwire), in both modes; the
    original read them from plant_default.npz with the argument tuple ignored (--pin);
  * the gate's software check: in emulated mode XLA_FLAGS alone is excused and every other item is still checked
    (xla_emul_common.gate_software_check); the original's --nosoftcheck skipped the whole check. In default mode it is
    unchanged;
  * v6's stage2_common is loaded from its pinned bytes (V7/code/v7_common.load_v6_common, the same bytes as
    scratchpad/ood/stage2/stage2_common.py) instead of by sys.path;
  * it records per arm both SHA-256 and the failure flag besides equal / max_abs_diff, the seconds split into loading
    and training, the environment and processor, the constants facts and the counts of substitute and tripwire calls,
    and writes them as JSON (the original printed one line).
The computation itself is the original's, line for line.

Usage (from /home/user/bca, one processor): python xla_repro.py --seed S --mode default|emulated --out OUT.json
"""
import sys

sys.dont_write_bytecode = True
import argparse  # noqa: E402
import hashlib  # noqa: E402
import os  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import xla_emul_common as X  # noqa: E402

ORIGINAL = dict(path="scratchpad/v7_hw_probe/xla_repro.py (byte copy: xla_repro_original.py)",
                sha256="25d2d84ab6388f4908bdb7966e919011934ea8589936b757061274c7cb9d6dc8")
T0 = time.perf_counter()
ap = argparse.ArgumentParser()
ap.add_argument("--seed", type=int, required=True)
ap.add_argument("--mode", choices=X.MODES, required=True)
ap.add_argument("--out", required=True)
a = ap.parse_args()
OUT = Path(a.out)
out = dict(kind="stage2-v7-evidence-xla-repro", information_only=X.INFO, seed=a.seed, mode=a.mode, argv=sys.argv[1:],
           script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
           common_sha256=hashlib.sha256((HERE / "xla_emul_common.py").read_bytes()).hexdigest(), derived_from=ORIGINAL,
           question=("check 2's stalling-world swap (gapped, full, random side by side) rerun with the gate's own "
                     "unchanged ActorOnly program: is each arm bit-identical to the gate's saved actor?"))
problems = X.check_mode(a.mode)
if problems:
    X.refuse(OUT, out, "; ".join(problems))
out["environment"] = X.env_facts(a.mode)
if not out["environment"]["mode_confirmed"]:
    X.refuse(OUT, out, f"the {a.mode} mode is not confirmed by numpy's dispatch and OpenBLAS's kernels")
try:
    out["constants"] = X.substitute_constants()
except X.K.V7Refusal as error:
    X.refuse(OUT, out, f"constants: {error}")
out["gate_software_check"] = X.gate_software_check(a.mode)
H = X.harness()
L = H.L
import jax  # noqa: E402

t0 = time.perf_counter()
ctx = H.SeedContext(a.seed, worlds=("stall",))
acts, rec, d = ctx.h2()
pg = ctx.prepared("stall")
named = [("gapped", ctx.critic("stall", "gap")[0]), ("full", ctx.critic("stall", "none")[0]),
         ("random", ctx.critic("stall", "random")[0])]
clone, _ = ctx.clone("stall")
t1 = time.perf_counter()
ao = L.ActorOnly(pg, clone, ctx.learn, ctx.rep, alpha=H.ALPHA)
stacked, info = ao.run([L.Arm(n, critic=i) for i, (n, _) in enumerate(named)], [c for _, c in named],
                       weighted=False, log=lambda *a: None)
jax.block_until_ready(stacked)
t2 = time.perf_counter()
arms = {}
for i, (n, _) in enumerate(named):
    p = ao.arm_params(stacked, i)
    this, gate = L.tree_sha256(p), L.tree_sha256(acts[n])
    arms[n] = dict(equal=this == gate, max_abs_diff=float(H.max_param_diff(p, acts[n])), this_run_sha256=this,
                   gate_sha256=gate, failed=bool(info["failed"][i]))
out.update(arms=arms, all_equal=all(v["equal"] for v in arms.values()),
           gate_check2=dict(folder=str(d), actors_sha256=rec.get("actors_sha256")),
           seconds=dict(load=t1 - t0, train=t2 - t1, total=time.perf_counter() - t0,
                        process=time.perf_counter() - T0))
X.write(OUT, out)
print(f"seed {a.seed} {a.mode}: " + ", ".join(f"{n} equal={v['equal']} max_abs_diff={v['max_abs_diff']:.3g}"
                                              for n, v in arms.items()) + f"; {out['seconds']['total']:.0f} s")
sys.stdout.flush()
os._exit(0)

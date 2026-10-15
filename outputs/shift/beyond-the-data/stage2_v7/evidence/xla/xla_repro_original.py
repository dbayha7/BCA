"""Read-only probe for the v7 design: does the gate's own unchanged actor-only program (learners.ActorOnly, check 2's
swap for the stalling world: gapped, full and random critics side by side) reproduce check 2's SAVED gapped-critic
actor of gate seed 0 when JAX/XLA generates code for an AVX2-only processor (XLA_FLAGS=--xla_cpu_max_isa=AVX2) and
OpenBLAS uses Haswell kernels? This is the arithmetic behind Stage 2's policy check (b) (1e-5 on every parameter).
No BCA code runs; nothing is written except this probe's JSON line on stdout. With --pin, the plant matrix A and the
gain K are replaced by the cloud's bits (from plant_default.npz) so that the gate's loaders accept; with --nosoftcheck
the gate's software-flag comparison is skipped (it would refuse the XLA_FLAGS setting)."""
import sys
sys.dont_write_bytecode = True
import json, os, time
sys.path.insert(0, "/home/user/bca")
import numpy as np
import experiments.shift.thruster_chain as TC
import experiments.shift.recorder as RC
if "--pin" in sys.argv:
    ref = np.load(sys.argv[1])
    A0, B0, K0 = ref["A"].copy(), ref["B"].copy(), ref["K"].copy()
    for m in (A0, B0, K0):
        m.setflags(write=False)
    TC._matrices = lambda gain, wave_turn: (A0, B0)
    TC._columns.cache_clear()
    RC._gain = lambda gain, wave_turn, throttle_cost, gamma: K0
import experiments.shift.acceptance as A
if "--nosoftcheck" in sys.argv:
    A.check_software = lambda payload: None
sys.path.insert(0, "/tmp/claude-0/-home-user/9e341792-006d-5a11-a8e9-8fac16ac7f5b/scratchpad/ood/stage2")
import stage2_common as C
H = C.load_room_harness()
L = H.L
t0 = time.perf_counter()
ctx = H.SeedContext(0, worlds=("stall",))
acts, rec, d = ctx.h2()
pg = ctx.prepared("stall")
named = [("gapped", ctx.critic("stall", "gap")[0]), ("full", ctx.critic("stall", "none")[0]),
         ("random", ctx.critic("stall", "random")[0])]
clone, _ = ctx.clone("stall")
ao = L.ActorOnly(pg, clone, ctx.learn, ctx.rep, alpha=H.ALPHA)
stacked, info = ao.run([L.Arm(n, critic=i) for i, (n, _) in enumerate(named)], [c for _, c in named],
                       weighted=False, log=lambda *a: None)
out = dict(xla_flags=os.environ.get("XLA_FLAGS"), openblas=os.environ.get("OPENBLAS_CORETYPE"), arms={})
for i, (n, _) in enumerate(named):
    p = ao.arm_params(stacked, i)
    out["arms"][n] = dict(equal=L.tree_sha256(p) == L.tree_sha256(acts[n]), max_abs_diff=H.max_param_diff(p, acts[n]))
out["seconds"] = time.perf_counter() - t0
print(json.dumps(out))

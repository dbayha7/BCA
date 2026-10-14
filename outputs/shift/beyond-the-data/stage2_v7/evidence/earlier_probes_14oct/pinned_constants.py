"""Read-only test of one v7 option: on non-SkylakeX kernels, replace the two LAPACK-derived constants (the plant
matrix A from gapped_lq.transition_matrix's SVD norm, and the textbook LQR gain K from scipy's DARE + numpy solve)
by the cloud's exact bits, then run the gate's own unchanged checks: acceptance.load_prereg + guard (anchors,
software) and acceptance.load_seed (log regeneration) on gate seeds 0-4 in both worlds. Writes nothing under runs/."""
import sys
sys.dont_write_bytecode = True
import json, os, time
sys.path.insert(0, "/home/user/bca")
sys.path.insert(0, "/home/user/bca/outputs/shift/beyond-the-data/stage2_gpu/lib")
import numpy as np
import envfacts as E
import experiments.shift.thruster_chain as TC
import experiments.shift.recorder as RC
ref = np.load(sys.argv[1])
pin = "--no-pin" not in sys.argv
if pin:
    A0, B0, K0 = ref["A"].copy(), ref["B"].copy(), ref["K"].copy()
    for m in (A0, B0, K0):
        m.setflags(write=False)
    TC._matrices = lambda gain, wave_turn: (A0, B0)
    TC._columns.cache_clear()
    RC._gain = lambda gain, wave_turn, throttle_cost, gamma: K0
import experiments.shift.acceptance as A
out = dict(openblas=E.openblas_cores(), pinned=pin)
t = time.perf_counter()
try:
    p = A.load_prereg(A.refuse_protected(A.RUNS_DEFAULT))
    A.guard(p)
    out["gate_accepts"] = True
except Exception as e:
    out["gate_accepts"], out["gate_error"] = False, str(e)[:300]
    p = None
seeds = {}
if (p is not None and "--no-seeds" not in sys.argv) or "--seeds-anyway" in sys.argv:
    cfg = A.config_from_dict(json.loads((A.RUNS_DEFAULT / A.PREREG_NAME).read_text())["config"])
    for rep in (0, 1, 2, 3, 4):
        for w in ("stall", "no_stall"):
            try:
                A.load_seed(A.RUNS_DEFAULT, cfg, "base", w, rep)
                seeds[f"{w}/seed{rep}"] = "ok"
            except Exception as e:
                seeds[f"{w}/seed{rep}"] = str(e)[:160]
out["load_seed"] = seeds
out["seconds"] = time.perf_counter() - t
print(json.dumps(out))

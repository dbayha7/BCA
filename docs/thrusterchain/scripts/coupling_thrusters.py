"""Per-thruster reach with and without the coupling C (supplement to coupling_facts.py)."""
import json
import numpy as np
import experiments.shift  # noqa: F401
from experiments.shift import thruster_chain as T
from experiments.signal import gapped_lq as GLQ

A13, B13 = T.matrices(T.STALL)
A10 = np.array(A13[:10, :10]); A0 = GLQ.oscillators(GLQ.STAGE0_SPEC); B = np.array(B13[:10])
pair = lambda i: slice(2 * i, 2 * i + 2)
out = {}
for k in range(4):
    b = B[:, k]
    own = [k, k + 1]
    res = {}
    for name, M in (("with_C", A10), ("without_C", A0)):
        x, sizes = b.copy(), []
        for t in range(100):
            sizes.append([float(np.linalg.norm(x[pair(i)])) for i in range(5)])
            x = M @ x
        sizes = np.array(sizes)                       # sizes after a one-step thrust of 1, steps 1..100
        ss = np.linalg.solve(np.eye(10) - M, b)       # resting offset under a thrust of 1 held forever
        res[name] = dict(
            one_step_max20=[round(float(v), 5) for v in sizes[:20].max(0)],
            one_step_max100=[round(float(v), 5) for v in sizes.max(0)],
            held_offset=[round(float(np.linalg.norm(ss[pair(i)])), 5) for i in range(5)])
    oth = [i for i in range(5) if i not in own]
    res["own_floats"] = [i + 1 for i in own]
    res["largest_other_one_step_20"] = max(res["with_C"]["one_step_max20"][i] for i in oth)
    res["smallest_own_one_step_20"] = min(res["with_C"]["one_step_max20"][i] for i in own)
    res["largest_other_held"] = max(res["with_C"]["held_offset"][i] for i in oth)
    res["smallest_own_held"] = min(res["with_C"]["held_offset"][i] for i in own)
    out[f"thruster_{k + 1}"] = res
    print(k + 1, json.dumps(res))
json.dump(out, open("/mnt/c/Users/David/dev/bca-shift-work/page/notes/coupling_thrusters.json", "w"), indent=1)

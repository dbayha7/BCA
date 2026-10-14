import hashlib, json, os, sys
import numpy as np
from numpy._core._multiarray_umath import __cpu_features__ as F, __cpu_dispatch__ as D
rng = np.random.default_rng(12345)
x = rng.uniform(0.0, 2 * np.pi, size=1_000_000)
u = rng.normal(size=1_000_000) * 3
h = lambda a: hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()[:16]
out = dict(avx512_skx=bool(F.get("AVX512_SKX")), avx2=bool(F.get("AVX2")),
           cos=h(np.cos(x)), sin=h(np.sin(x)), tanh=h(np.tanh(u)), exp=h(np.exp(u)), log=h(np.log(np.abs(u) + 1)))
sys.path.insert(0, "/home/user/bca")
try:
    import experiments.shift.recorder as RC
    import experiments.shift.thruster_chain as TC
    K = RC.textbook_gain(TC.STALL)
    out["K"] = h(K)
except Exception as e:
    out["K"] = "error " + repr(e)[:200]
print(json.dumps(out))

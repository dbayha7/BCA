"""Read-only check: re-run phase 1 for seed 0 into a scratch folder and compare with the stored arrays.npz."""
import sys, importlib.util
sys.dont_write_bytecode = True
from pathlib import Path
import numpy as np
V = Path("/tmp/claude-0/-home-user/9e341792-006d-5a11-a8e9-8fac16ac7f5b/scratchpad/routes/runtime_monitor/verify_adversarial/out")
V.mkdir(parents=True, exist_ok=True)
spec = importlib.util.spec_from_file_location("prep_v", "/tmp/claude-0/-home-user/9e341792-006d-5a11-a8e9-8fac16ac7f5b/scratchpad/routes/runtime_monitor/prep.py")
P = importlib.util.module_from_spec(spec); spec.loader.exec_module(P)
P.OUT = V
P.main(0)
a = np.load(V / "seed0" / "arrays.npz"); b = np.load("/home/user/bca/runs/shift_explore/routes/runtime_monitor/seed0/arrays.npz")
assert set(a.files) == set(b.files)
for k in sorted(a.files):
    print(k, "identical" if np.array_equal(a[k], b[k]) else "DIFF max %g" % np.max(np.abs(a[k].astype(float) - b[k].astype(float))))

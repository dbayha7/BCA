"""Read-only: dump the plant matrix A, B, the LQR gain K and one gate seed's regenerated log arrays (base/stall/seed0)
under the current OpenBLAS kernel, to a scratchpad .npz. Nothing under runs/ is written."""
import sys
sys.dont_write_bytecode = True
import json, os
sys.path.insert(0, "/home/user/bca")
import numpy as np
import experiments.shift.acceptance as A
import experiments.shift.recorder as RC
import experiments.shift.thruster_chain as TC
import experiments.shift.dataset as DS
out = sys.argv[1]
Am, Bm = TC.matrices(TC.STALL)
K = np.asarray(RC.textbook_gain(TC.STALL))
r0 = A.read_json(A.data_dir(A.RUNS_DEFAULT, "base", "stall", 0) / "record.json")
lg = DS.generate(0, "stall", RC.RecorderSpec(**r0["recorder"]), tuple(r0["sizes"]), r0["stall_slope"])
arrs = {k.replace("/", "__"): np.asarray(lg[k]) for k in ("observations", "actions", "rewards", "next_observations",
                                                            "infos/state", "infos/next_state")}
np.savez(out, A=np.asarray(Am), B=np.asarray(Bm), K=K, **arrs)
print("saved", out)

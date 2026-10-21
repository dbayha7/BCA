"""Negative controls for probe_s7.py (made-up ReBRAC actor of smoke_probe.py's fixtures): a float64 adapter with a
small formula defect must fail reading (i)
(v2notes/rr/calibrate.py runs the same defects through the judged path). Usage: smoke_negative.py <worktree> <probe dir> <fixture root>"""
import sys
from pathlib import Path

WT, PROBE_DIR, ROOT = (Path(a).resolve() for a in sys.argv[1:4])
sys.path.insert(0, str(WT))
sys.path.insert(0, str(PROBE_DIR))
import numpy as np  # noqa: E402

from experiments.shift_hosts import actors_f32 as A32  # noqa: E402

A32.configure()
from experiments.shift import thruster_chain as TC  # noqa: E402
from experiments.shift import truth as TR  # noqa: E402
from experiments.shift_hosts import evaluate as E  # noqa: E402
from experiments.shift_hosts import handoff as H  # noqa: E402
from experiments.shift_hosts import policies as P  # noqa: E402

import probe_s7 as PR  # noqa: E402

actor = A32.read_actor_file(ROOT / "runs/shift_hosts/stage1/medium-expert/seed0/rebrac-alone/actors/step15000.npz")
real = P.adapter


class BiasDefect(P.ReBRACPolicy):
    def __call__(self, x):
        h = P._hidden(P._observe(x), self.Ws[:3], self.bs[:3])
        return np.nan_to_num(np.tanh(h @ self.Ws[3] + self.bs[3] + 1e-4))       # an output bias off by 1e-4


class BranchDefect(P.ReBRACPolicy):
    def __call__(self, x):
        out = super().__call__(x)
        bad = np.asarray(x)[:, 0] > 1.5                                           # a wrong branch on rare states
        out[bad] = np.clip(out[bad] + 0.01, -1.0, 1.0)
        return out


for name, cls in (("bias", BiasDefect), ("branch", BranchDefect)):
    P.adapter = lambda a, bank=None, starts=None, readout="scored", cls=cls: cls(a.params)
    pr = PR.Probe(None, (A32, P, E, TR, TC, H))
    r = pr.bank(actor, "scored", "big_kicks", None)
    i = r["i"]
    print(f"negative control {name}: injected gap max {i['inj']:.2e}; float32 reference spread {i['ref']:.2e}; reading "
          f"(i) within max(1e-5, 10 x spread) = {i['bound']:.2e}: {i['passed']}; largest |dpoints| {r['dp'].max():.3e}",
          flush=True)
P.adapter = real

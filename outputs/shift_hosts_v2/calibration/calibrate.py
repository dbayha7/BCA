"""Cloud calibration of probe_s7.py's reading rules (PRESPEC.md 19.6) on MADE-UP ReBRAC actors only, run through the
probe's own judged path (main(), a fixture rebrac-alone run at step 15000 with an s7.json made by
evaluate.closed_loop_check). No trained actor exists here. One case per process:

  calibrate.py <worktree> <probe dir> <fixture root> <out file> <scale> <defect>

scale multiplies every Dense kernel of test_scoring.made_up_actor("rebrac") (defect-free float32 rounding grows with
it); defect is "none", "bias" (the float64 adapter's output bias off by 1e-4) or "branch" (the float64 adapter's
command +0.01 where x[0] > 1.5). With a defect the lane's own s7.json is made with the same defective adapter, as a
real defect in policies.py would be. The weighting banks are the smoke test's made-up stand-ins. The approval check is
replaced by a stand-in with version 1's fingerprint (the cloud has no verifying approval).
"""

import shutil
import sys
from pathlib import Path

WT, PROBE_DIR, ROOT, OUT = (Path(a).resolve() for a in sys.argv[1:5])
SCALE, DEFECT = float(sys.argv[5]), sys.argv[6]
sys.path.insert(0, str(WT))
sys.path.insert(0, str(PROBE_DIR))

import numpy as np  # noqa: E402

from experiments.shift_hosts import actors_f32 as A32  # noqa: E402

A32.configure()
from experiments.shift import thruster_chain as TC  # noqa: E402
from experiments.shift_hosts import evaluate as E  # noqa: E402
from experiments.shift_hosts import gate  # noqa: E402
from experiments.shift_hosts import guard  # noqa: E402
from experiments.shift_hosts import handoff as H  # noqa: E402
from experiments.shift_hosts import policies as P  # noqa: E402
from experiments.shift_hosts.tests import test_scoring as TS  # noqa: E402

import probe_s7 as PR  # noqa: E402

real_starts, real_anchor = E.starts, E.anchor


def fake_starts(bank):
    if E.bank_name(bank) in E.WEIGHTING:
        return np.array(TC.bank_starts("recorded"), np.float64)
    return real_starts(bank)


def fake_anchor(bank):
    if E.bank_name(bank) in E.WEIGHTING:
        return dict(scale="weighting", bottom=-1351.41, top=-80.61, source="made-up stand-in (calibration)")
    return real_anchor(bank)


E.starts, E.anchor = fake_starts, fake_anchor


class BiasDefect(P.ReBRACPolicy):
    def __call__(self, x):
        h = P._hidden(P._observe(x), self.Ws[:3], self.bs[:3])
        return np.nan_to_num(np.tanh(h @ self.Ws[3] + self.bs[3] + 1e-4))        # an output bias off by 1e-4


class BranchDefect(P.ReBRACPolicy):
    def __call__(self, x):
        out = super().__call__(x)
        bad = np.asarray(x)[:, 0] > 1.5                                            # a wrong branch on rare states
        out[bad] = np.clip(out[bad] + 0.01, -1.0, 1.0)
        return out


if DEFECT != "none":
    cls = dict(bias=BiasDefect, branch=BranchDefect)[DEFECT]
    P.adapter = lambda a, bank=None, starts=None, readout="scored", cls=cls: cls(a.params)


def v1_stand_in(purpose):
    return gate.Approval(path="calibration stand-in", fingerprint=PR.V1_FINGERPRINT, prespec_sha256="stand-in",
                         file_sha256="stand-in", commit="0000000stand-in", author_email="stand-in",
                         author_name="stand-in")


gate.require_run_approval = v1_stand_in

if ROOT.exists():
    shutil.rmtree(ROOT)
run = ROOT / "runs" / "shift_hosts" / "stage1" / "medium-expert" / "seed0" / "rebrac-alone"
(run / "actors").mkdir(parents=True)
actor_path = run / "actors" / "step15000.npz"
params, _ = TS.made_up_actor("rebrac", 0)
for k in params["params"]:
    params["params"][k]["kernel"] = (np.asarray(params["params"][k]["kernel"], np.float32)
                                     * np.float32(SCALE)).astype(np.float32)
A32.write_actor_file(actor_path, "rebrac", params,
                     meta=dict(made_up=True, actor_ln=False, norm_obs=False,
                               note=f"made-up parameters, kernels x {SCALE:g} (calibration), not a learned policy"))
actor = A32.read_actor_file(actor_path)
H.write_json(run / "record.json", dict(run_id=f"calibration/rebrac-alone x{SCALE:g} {DEFECT}", outcome="complete",
                                       timings=dict(outer_steps=15000), code=dict(code_manifest_sha256="made-up"),
                                       snapshots={"15000": dict(actor=dict(path=str(actor_path),
                                                                           sha256=actor.npz_sha256))}))
lane = run / "lanes" / "s7-step15000"
lane.mkdir(parents=True)
H.write(lane / "states.npz", dict(obs=np.zeros((1, 13), np.float32)),
        meta=dict(actor_npz_sha256=actor.npz_sha256, made_up=True))
closed = E.closed_loop_check(actor_path, guard.MADE_UP)
H.write_json(lane / "s7.json", dict(closed_loop=closed, passed=True, s7_passed=closed["passed"],
                                    scored=dict(max_abs=0.0, passed=True), learner="rebrac"))
print(f"case scale {SCALE:g} defect {DEFECT}: the lane's closed loop at 0.05 passed {closed['passed']}", flush=True)
code = PR.main(["--repo", str(WT), "--run-dir", str(run), "--out", str(OUT)])
sys.stdout = sys.__stdout__
print(f"probe exit {code}", flush=True)

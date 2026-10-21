"""Cloud smoke test of v2notes/probe_s7.py on MADE-UP actor files only (no trained actor exists here).

Environment: v2notes/../smokevenv (numpy 2.2.6, JAX 0.6.2, flax 0.10.7), BCA_SHIFT_DEPS=v2notes/smoke/deps (gymnasium
1.0.0 via experiments/shift/install_deps.sh), python -B, PYTHONDONTWRITEBYTECODE=1, JAX_PLATFORMS=cpu.
The four situation banks are real (thruster_chain.bank_starts, preregistration.json anchors). The weighting test's
three start banks are not in the cloud, so evaluate.starts and evaluate.anchor are replaced for none/mild/strong by
made-up stand-ins (the recorded bank's starts, version 1's strong-bank anchors). Fixtures go under v2notes/smoke/root:
each alone run has a step-15000 actor with an s7.json made by evaluate.closed_loop_check, and a step-10000 actor
(another made-up seed, no s7.json) for the --step 10000 --compare path.
The cloud has no verifying approval, so gate.require_run_approval is replaced by a stand-in with version 1's
fingerprint for the two probe runs; the refusal checks at the end use the real gate and a stand-in of another
fingerprint.

Usage: smoke_probe.py <worktree> <probe dir> <fixture root> <out file 15000> <out file 10000>
"""

import shutil
import sys
from pathlib import Path

WT, PROBE_DIR, ROOT, OUT, OUT10 = (Path(a).resolve() for a in sys.argv[1:6])
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
from experiments.shift_hosts.tests import test_scoring as TS  # noqa: E402

import probe_s7 as PR  # noqa: E402

real_starts, real_anchor = E.starts, E.anchor
real_gate = gate.require_run_approval


def fake_starts(bank):
    name = E.bank_name(bank)
    if name in E.WEIGHTING:
        return np.array(TC.bank_starts("recorded"), np.float64)
    return real_starts(bank)


def fake_anchor(bank):
    name = E.bank_name(bank)
    if name in E.WEIGHTING:
        return dict(scale="weighting", bottom=-1351.41, top=-80.61, source="made-up stand-in (smoke test)")
    return real_anchor(bank)


def stand_in(fingerprint):
    def approval(purpose):
        return gate.Approval(path="smoke stand-in", fingerprint=fingerprint, prespec_sha256="stand-in",
                             file_sha256="stand-in", commit="0000000stand-in", author_email="stand-in",
                             author_name="stand-in")
    return approval


E.starts, E.anchor = fake_starts, fake_anchor

if ROOT.exists():
    shutil.rmtree(ROOT)
seed0 = ROOT / "runs" / "shift_hosts" / "stage1" / "medium-expert" / "seed0"
for learner in A32.LEARNERS:
    run = seed0 / f"{learner}-alone"
    (run / "actors").mkdir(parents=True)
    snaps = {}
    for step, seed in ((10000, 1), (15000, 0)):
        actor_path = run / "actors" / f"step{step}.npz"
        params, norm = TS.made_up_actor(learner, seed)
        meta = dict(made_up=True, note="made-up parameters (cloud smoke test of probe_s7.py), not a learned policy")
        if learner == "rebrac":
            meta.update(actor_ln=False, norm_obs=False)
        kw = {} if norm is None else dict(obs_mean=norm["obs_mean"], obs_std=norm["obs_std"])
        A32.write_actor_file(actor_path, learner, params, meta=meta, **kw)
        snaps[str(step)] = dict(actor=dict(path=str(actor_path), sha256=A32.read_actor_file(actor_path).npz_sha256))
    H.write_json(run / "record.json", dict(run_id=f"smoke/{learner}-alone", outcome="complete",
                                           timings=dict(outer_steps=15000),
                                           code=dict(code_manifest_sha256="made-up"), snapshots=snaps))
    actor_path = run / "actors" / "step15000.npz"
    lane = run / "lanes" / "s7-step15000"
    lane.mkdir(parents=True)
    H.write(lane / "states.npz", dict(obs=np.zeros((1, 13), np.float32)),
            meta=dict(actor_npz_sha256=snaps["15000"]["actor"]["sha256"], made_up=True))
    closed = E.closed_loop_check(actor_path, guard.MADE_UP)          # the lane's own computation, made-up actor
    H.write_json(lane / "s7.json", dict(closed_loop=closed, passed=True, s7_passed=closed["passed"],
                                        scored=dict(max_abs=0.0, passed=True), learner=learner))
    print(f"fixture {learner}: closed loop passed {closed['passed']}", flush=True)

PR.list_s7(ROOT, H)                      # the fixtures sit outside --repo, so list them directly
gate.require_run_approval = stand_in(PR.V1_FINGERPRINT)
code = PR.main(["--repo", str(WT), "--run-dir", str(seed0 / "rebrac-alone"), "--list-s7", "--compare",
                "--out", str(OUT)])
sys.stdout = sys.__stdout__
print(f"probe exit {code} (step 15000)", flush=True)
code = PR.main(["--repo", str(WT), "--run-dir", str(seed0 / "rebrac-alone"), "--step", "10000", "--compare",
                "--out", str(OUT10)])
sys.stdout = sys.__stdout__
print(f"probe exit {code} (step 10000)", flush=True)


def refused(label, argv):
    try:
        PR.main(argv)
        print(f"REFUSAL CHECK FAILED: {label} was accepted")
    except SystemExit as stop:
        print(f"refusal check: {label} refused (exit {stop.code})")


base = ["--repo", str(WT), "--run-dir", str(seed0 / "rebrac-alone")]
refused("--out inside the repository", base + ["--out", str(WT / "probe_out.txt")])
print(f"  file exists: {(WT / 'probe_out.txt').exists()}")
refused("--out on an existing file", base + ["--out", str(OUT)])
gate.require_run_approval = stand_in("f" * 64)
refused("an approval of another fingerprint", base)
gate.require_run_approval = real_gate
refused("no verifying approval (the real gate on the draft bytes)", base)

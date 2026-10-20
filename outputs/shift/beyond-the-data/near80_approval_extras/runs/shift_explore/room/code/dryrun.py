"""EXPLORATORY dry run of run_seed.py's bookkeeping (manifest, lock, verify, values, refusals, stop, futility, pooled,
no-stall, replication) into a scratch directory, with 40 actor steps, the gate-path reproduction replaced by the saved
actors, the W = 0 tolerance switched off, and the evaluation REPLACED BY A DUMMY: no return, points or score is
computed. Nothing is written under runs/shift_explore/room. Check (e) runs for real (critic surface only); seed 1 is
then forced to fail it, to exercise the stop path.

Usage: python dryrun.py
"""
import shutil
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import harness as H  # noqa: E402
import run_seed as R  # noqa: E402
import write_manifest as WM  # noqa: E402
import write_spec as WS  # noqa: E402
import experiments.shift.acceptance as A  # noqa: E402
import experiments.shift.learners as L  # noqa: E402
import experiments.shift.thruster_chain as TC  # noqa: E402

REAL_OUT = H.OUT
TMP = HERE / "dryrun_out"


def fresh(path):
    """A scratch output folder with the spec as harness.spec_payload states it now and a manifest of the code now."""
    if path.exists():
        shutil.rmtree(path)
    path.mkdir()
    H.OUT = REAL_OUT
    payload = WS.build_payload()
    H.OUT = path
    H.SPEC_PATH = path / "spec.json"
    H.write_json(H.SPEC_PATH, payload)
    WM.write_manifest(None)


def refused(fn, *a, expect=""):
    try:
        fn(*a)
    except SystemExit as e:
        msg = str(e)
        if expect and expect not in msg:
            raise AssertionError(f"refused for another reason: {msg}")
        print(f"REFUSED as expected ({fn.__name__}{a}): {msg[:160]}")
        return
    raise AssertionError(f"{fn.__name__}{a} was not refused")


class Short(H.RoomRunner):
    def __init__(self, *a, **k):
        k["steps"], k["chunk"] = 40, 20
        super().__init__(*a, **k)


def fake_reproduce(ctx, world="stall", log_fn=H.log):
    acts, rec, d = ctx.h2()
    pg = ctx.prepared(world)
    clone, _ = ctx.clone(world)
    rows = L.ActorOnly(pg, clone, ctx.learn, ctx.rep, alpha=H.ALPHA).rows_sha256
    return dict(passed=True, rows_sha256=rows, arms={"gapped": "dry run", "no_stall_gapped": "dry run"},
                note="DRY RUN: not reproduced"), acts


def fake_evaluate(params, prepared, anchors, cfg, failed=False):
    """DUMMY (no rollout, no return): fixed numbers in the evaluation's shape."""
    if params is None or failed:
        return None
    return {s: dict(points=float(len(L.tree_sha256(params)) % 7), capped_points=0.0, unrecorded_throttle_share=0.0)
            for s in TC.SITUATIONS}


H.RoomRunner = Short
H.W0_TOL = float("inf")
R.reproduce_h2 = fake_reproduce
A._evaluate = fake_evaluate
R.thruster_shares = lambda params, prepared, physics: {s: [0.0] * 4 for s in TC.SITUATIONS}
real_shape = H.oracle_shape

fresh(TMP)
# seed 0: the whole path
R.phase_train(0)
refused(R.phase_train, 0, expect="already locked")
refused(R.phase_futility, expect="neither values.json nor stop.json")
R.phase_eval(0)
refused(R.phase_eval, 0, expect="exists")
# seed 1: check (e) forced to fail -> stop.json; every later phase of that seed refuses
H.oracle_shape = lambda *a, **k: dict(real_shape(*a, **k), passed=False)
try:
    R.phase_train(1)
    raise AssertionError("the forced check (e) failure did not stop seed 1")
except R.Stop:
    print("seed 1 stopped as forced")
H.oracle_shape = real_shape
refused(R.phase_train, 1, expect="stopped")
refused(R.phase_eval, 1, expect="stopped")
refused(R.phase_pooled, expect="futility.json")
R.phase_futility()
fut = H.read_json(H.OUT / "futility.json")
assert fut["seeds"]["1"]["status"] == "stopped" and fut["seeds"]["1"]["counts_as_below_5"], fut["seeds"]
refused(R.phase_futility, expect="exists")
refused(R.phase_pooled, expect="saying continue")
# no-stall and replication on seed 0
R.phase_no_stall_train(0)
R.phase_no_stall_eval(0)
refused(R.phase_no_stall_eval, 0, expect="exists")
R.phase_replication_train(0)
R.phase_replication_eval(0)
refused(R.phase_replication_eval, 0, expect="exists")
# pooled: a fabricated 'continue' (dry run only) and stop files for seeds 2-4 written one by one
H.write_json(H.OUT / "futility.json", dict(fut, stop=False, decision="DRY RUN fabricated continue"))
for rep in (2, 3):
    H.write_json(R.seed_dir(rep) / "stop.json", dict(label=H.LABEL, kind="room-stop", rep=rep, stopped=True,
                                                    reason="DRY RUN", spec_sha256=H.file_sha256(H.SPEC_PATH)))
refused(R.phase_pooled, expect="neither values.json nor stop.json")
H.write_json(R.seed_dir(4) / "stop.json", dict(label=H.LABEL, kind="room-stop", rep=4, stopped=True, reason="DRY RUN",
                                               spec_sha256=H.file_sha256(H.SPEC_PATH)))
R.phase_pooled()
pooled = H.read_json(H.OUT / "pooled.json")
assert pooled["complete"] is False and not pooled["room"]["passed"] and not pooled["recovery"]["passed"]
refused(R.phase_pooled, expect="exists")
# a changed script or a changed copy refuses every phase
copy = H.OUT / "code" / "dryrun.py"
keep = copy.read_bytes()
copy.write_bytes(keep + b"\n")
refused(R.phase_eval, 0, expect="differs from the manifest")
copy.write_bytes(keep)

# tampered values: a synthetic values.json (labelled 'synthetic', no spec or lock hash) is refused by the looks
TMP2 = HERE / "dryrun_out2"
fresh(TMP2)
H.write_json(R.seed_dir(0) / "values.json", dict(label="synthetic", kind="room-values", rep=0))
H.write_json(R.seed_dir(1) / "values.json", dict(label="synthetic", kind="room-values", rep=1))
refused(R.phase_futility, expect="labelled output")
H.write_json(R.seed_dir(0) / "values.json", dict(label=H.LABEL, kind="room-values", rep=0,
                                                 spec_sha256=H.file_sha256(H.SPEC_PATH), lock_sha256="0" * 64))
refused(R.phase_futility, expect="no lock")
shutil.rmtree(TMP2)
H.OUT = TMP
print("DRY RUN OK", sorted(str(p.relative_to(TMP)) for p in TMP.rglob("*.json")))

"""EXPLORATORY dry run of run_standins.py's bookkeeping (spec, manifest, room gate, lock, verify, refusals, stop,
no-stall, marks, and the user's approval with a SYNTHETIC chat message: training, evaluation and the no-stall phases run
with NO approval.json, the marks look refuses without one, approve.py refuses after the marks look; spec change C6)
into a scratch directory (scratchpad/ood/standins/dryrun_out), with 40 actor steps, small data-side
fits, the W = 0 tolerance switched off, a SYNTHETIC target strength, and the evaluation REPLACED BY A DUMMY: no
return, points or score is computed. The room test's outputs are not read: the room gate, the room lock and the
room's values are replaced by synthetic stand-ins. Nothing is written under runs/.

Usage: python dryrun.py train | eval | no_stall | marks      (in this order; 'train' starts from a fresh directory)
"""
import shutil
import sys
from pathlib import Path

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import standins as S  # noqa: E402
import run_standins as R  # noqa: E402
import write_manifest as WM  # noqa: E402
import approve as AP  # noqa: E402

L, TC = S.L, S.TC
TMP = HERE / "dryrun_out"
TARGET = 0.004          # synthetic strength target (not the room's)


class Short(S.StandinRunner):
    last_rows_sha = None

    def __init__(self, *a, **k):
        k["steps"], k["chunk"] = 40, 20
        super().__init__(*a, **k)
        Short.last_rows_sha = self.rows_sha256


class LazyLock(dict):
    def __getitem__(self, k):
        return Short.last_rows_sha if k == "rows_sha256" else super().__getitem__(k)


def patch():
    S.OUT = TMP
    S.SPEC_PATH = TMP / "spec.json"
    S.SURF_PER_TYPE, S.SURF_VALID, S.SURF_STEPS, S.CLS_STEPS = 1000, 1000, 200, 50
    S.StandinRunner = Short
    S.W0_TOL = float("inf")
    S.SEEDS = (0,)
    S.room_gate = lambda: dict(dryrun="synthetic room gate (the room's pooled.json is not read)")
    S.room_lock_target = lambda rep, none: (TARGET, LazyLock(dryrun="synthetic target (the room's lock is not read)",
                                                             target_strength=TARGET, rows_sha256="dryrun"))

    def dummy_eval(ctx, lock, acts, prepared, world):
        ev = {n: {s: dict(points=0.0, unrecorded_throttle_share=0.0, capped_points=0.0) for s in TC.SITUATIONS}
              for n in lock["scored_names"]}
        sh = {n: {s: [0.0] * TC.DA for s in TC.SITUATIONS} for n in lock["scored_names"]}
        S.log(f"DUMMY evaluation of {len(ev)} actors (no return computed)")
        return ev, sh

    R.evaluate_locked = dummy_eval

    def room_values(rep):
        pts = {s: dict(perfect_band=60.0, none=0.0, **{"even_pull@1": 10.0}) for s in TC.SITUATIONS}
        return (dict(situations={s: dict(points=pts[s]) for s in TC.SITUATIONS}, matched_names={"even_pull": "even_pull@1"}),
                "synthetic")

    R.room_values = room_values


def refused(fn, *a, expect=""):
    try:
        fn(*a)
    except SystemExit as e:
        msg = str(e)
        if expect and expect not in msg:
            raise AssertionError(f"refused for another reason: {msg}")
        print(f"REFUSED as expected ({getattr(fn, '__name__', fn)}{a}): {msg[:160]}")
        return
    raise AssertionError(f"{fn}{a} was not refused")


def approval_checks(spec_sha, man_sha):
    """The user's choice of mark forms (spec changes C3, C6), with SYNTHETIC chat messages: refusals, one approval
    (written while seed outputs exist), tamper checks. Returns the message file."""
    msg = TMP / "message.txt"
    msg.write_text(f"DRY RUN synthetic message: {spec_sha[:16]} approve F1-F3", encoding="utf8")
    refused(AP.approve_chat, msg, expect="accept chat mode")
    msg.write_text(f"DRY RUN synthetic message: {spec_sha[:16]} approve F1-F3 and decline F1-F3, accept chat mode",
                   encoding="utf8")
    refused(AP.approve_chat, msg, expect="exactly one")
    msg.write_text(f"DRY RUN synthetic message: {'0' * 16} approve F1-F3, accept chat mode", encoding="utf8")
    refused(AP.approve_chat, msg, expect="spec prefix")
    msg.write_text(f"DRY RUN synthetic message: {spec_sha[:16]} approve F1-F3, accept chat mode", encoding="utf8")
    body = AP.approve_chat(msg)      # C6: accepted although every seed-0 output exists
    print("approval written with run outputs present:", body["run_outputs_at_approval"])
    assert body["run_outputs_at_approval"] and body["marks_look_run"] is False
    refused(AP.approve_chat, msg, expect="exists")
    refused(S.require_approval, "0" * 64, man_sha, expect="another spec")
    refused(S.require_approval, spec_sha, "0" * 64, expect="another spec")
    ap = S.approval_path()
    txt = ap.read_text()
    ap.write_text(txt.replace('"re_referenced"', '"literal"', 1))
    refused(S.require_approval, spec_sha, man_sha, expect="sidecar")
    ap.write_text(txt)
    print("approval:", S.require_approval(spec_sha, man_sha)["mark_forms"])
    return msg


def main(step):
    patch()
    if step == "train":
        if TMP.exists():
            shutil.rmtree(TMP)
        TMP.mkdir()
        S.write_json(S.SPEC_PATH, S.spec_payload())
        WM.write_manifest(None)
        # spec change C6: training, matching, the lock and the evaluations run with NO approval.json
        assert not S.approval_path().exists()
        refused(R.phase_marks, expect="no approval")
        refused(R.phase_eval, 0, expect="no lock")
        refused(R.phase_no_stall_train, 0, expect="no lock")
        R.phase_train(0)
        assert not S.approval_path().exists()
        print("TRAIN RAN WITHOUT approval.json")
        refused(R.phase_train, 0, expect="already locked")
        lock = S.read_json(TMP / "seed0" / "lock.json")
        assert lock["approval"] == S.APPROVAL_NOT_READ, lock.get("approval")
        print("matched:", lock["matched_names"], "unmatched strongest:", lock["unmatched_strongest_rung"])
        print("scored:", lock["scored_names"])
        print("check (b):", lock["checks"]["b"])
        print("check (e) passed:", {f: v["passed"] for f, v in lock["checks"]["e"].items() if isinstance(v, dict)})
        print("check (e) on classifier refits:", lock["classifier_refit_check_e_passed"])
        assert lock["classifier_refit_check_e_passed"] and all(
            len(v) == 2 for v in lock["classifier_refit_check_e_passed"].values())
        print("density levels:", {k: v for k, v in lock["shared"]["density_levels"].items() if k.startswith("q95")})
    elif step == "eval":
        assert not S.approval_path().exists()
        R.phase_eval(0, "stall")
        refused(R.phase_eval, 0, "stall", expect="exists")
        assert S.read_json(TMP / "seed0" / "values.json")["approval"] == S.APPROVAL_NOT_READ
        print("EVAL RAN WITHOUT approval.json")
    elif step == "no_stall":
        assert not S.approval_path().exists()
        R.phase_no_stall_train(0)
        refused(R.phase_no_stall_train, 0, expect="already locked")
        R.phase_eval(0, "no_stall")
        assert not S.approval_path().exists()
        print("NO_STALL_TRAIN AND NO_STALL_EVAL RAN WITHOUT approval.json")
        lock = S.read_json(TMP / "seed0" / "no_stall" / "lock.json")
        assert lock["approval"] == S.APPROVAL_NOT_READ
        print("no-stall arms:", lock["locked_from_stall"])
        print("no-stall check (e) on classifier refits:", lock["classifier_refit_check_e_passed"])
        assert lock["classifier_refit_check_e_passed"]
        print("check (b) no-stall:", lock["checks"]["b"])
    elif step == "marks":
        spec_sha, man_sha = S.file_sha256(S.SPEC_PATH), S.file_sha256(TMP / S.MANIFEST_NAME)
        # every seed has its values; the marks look still refuses without the user's choice
        assert not S.approval_path().exists()
        refused(R.phase_marks, expect="no approval")
        assert not S.marks_path().exists()
        refused(R.compute_marks, {}, (0,), expect="mark families")
        AP.show()
        msg = approval_checks(spec_sha, man_sha)
        R.phase_marks()
        refused(R.phase_marks, expect="exists")
        m = S.read_json(S.marks_path())
        print("captured_by (synthetic dummy points, meaningless):", m["captured_by"], "mark forms:", m["mark_forms"],
              m["mark_families"])
        assert m["mark_families"] == list(S.MARK_FORMS["re_referenced"]["mark"]) and set(m["captured_by"]) <= set(
            m["mark_families"])
        assert m["approval"]["sha256"] == S.file_sha256(S.approval_path())
        print({f: (v["c1_gain_at_least_half_of_perfect_band"], v["c2_recorded_at_least_none_minus_3"],
                   v["c3_beats_even_pull_by_15"]) for f, v in m["marks"].items()})
        # approve.py refuses after the marks look: with approval.json present, and with it moved aside (marks.json)
        refused(AP.approve_chat, msg, expect="exists")
        ap, side = S.approval_path(), S.approval_path().with_name(S.APPROVAL_NAME + ".sha256")
        keep = ap.read_bytes(), side.read_bytes()
        ap.unlink()
        side.unlink()
        refused(AP.approve_chat, msg, expect="marks look has run")
        assert not ap.exists()
        ap.write_bytes(keep[0])
        side.write_bytes(keep[1])
        print("APPROVE REFUSED AFTER THE MARKS LOOK")
        # tamper: a changed lock must refuse an evaluation
        p = TMP / "seed0" / "lock.json"
        txt = p.read_text()
        p.write_text(txt.replace('"rep": 0', '"rep": 9', 1))
        refused(R.verify_lock, TMP / "seed0", S.load_spec()[1], expect="sidecar")
        p.write_text(txt)
        print("DRY RUN OK")
    else:
        raise SystemExit(__doc__)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "")

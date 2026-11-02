"""BCA dose screen v1 tests (Windows): the dose rule reproduces DOSE.json; the four in-process changes act as declared
on a fake runtime; the decision rules give each verdict on synthetic numbers; the supervisor runs end to end with a fake
dispatcher against the real exploratory reports, including failure paths; and the real module wiring is checked.
JSON only; no checkpoint opened, nothing trained."""
from pathlib import Path
from types import SimpleNamespace
from unittest import mock
import json
import shutil
import sys
import tempfile
import unittest

HERE = Path(__file__).resolve().parent
for _p in (str(HERE.parent / "knee_edge_ood_v3"), str(HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)
sys.dont_write_bytecode = True

import numpy as np

import dose
import screen
import screen_math as SM
import screen_worker as SW
from core import sha_file


class Dose(unittest.TestCase):
    def test_rule_reproduces_the_recorded_dose(self):
        recorded = json.loads((HERE / "DOSE.json").read_text(encoding="utf-8"))
        again = dose.compute()
        self.assertEqual(again["kappa_A"], recorded["kappa_A"])
        self.assertAlmostEqual(again["kappa_A_unrounded"], recorded["kappa_A_unrounded"], places=12)
        self.assertEqual(again["per_fork"], recorded["per_fork"])
        self.assertEqual(SW.KAPPA_A, float(recorded["kappa_A"]))


class FakeRuntime:
    def __init__(self):
        def make_adapter(context, state, arm, kappa, *rest):
            return SimpleNamespace(arm=arm, kappa=kappa)
        self.make_adapter = make_adapter

        def branch(adapter, state, protocol, seed_index, arm, coefficient):
            kappa = {"native": 0.0, "bca": 3.0, "distance": coefficient}[arm]
            return self.make_adapter(None, state, arm, kappa, None, [10000, 15000], 20000), state
        self.branch = branch


class Changes(unittest.TestCase):
    def test_bca_gets_kappa_a_and_other_arms_are_untouched(self):
        rt = FakeRuntime()
        SW.patch_runtime(rt, kappa=63.0)
        SW.patch_runtime(rt, kappa=63.0)                                   # idempotent
        self.assertEqual(rt.branch(None, None, None, 0, "bca", 0.04)[0].kappa, 63.0)
        self.assertEqual(rt.branch(None, None, None, 0, "native", 0.04)[0].kappa, 0.0)
        self.assertEqual(rt.branch(None, None, None, 0, "distance", 0.04)[0].kappa, 0.04)

    def test_evaluation_restores_with_the_bca_arm_and_checks_the_branch(self):
        rt = FakeRuntime()
        seen = []

        def fake_policies(job, runtime, *rest):
            seen.append(runtime.branch(None, None, None, 0, "native", 0.04)[0])
            return "ok"
        before = rt.branch
        with tempfile.TemporaryDirectory() as root:
            good = {"kind": "bca_branch", "status": "completed", "kappa": SW.KAPPA_A, "dose_sha256": sha_file(HERE / "DOSE.json")}
            (Path(root) / "report.json").write_text(json.dumps(good))
            with mock.patch.object(SW, "_ORIGINAL_POLICIES", fake_policies):
                self.assertEqual(SW.policies_bca({}, rt, None, None, None, "cutoff", {"branch": root}, ()), "ok")
                (Path(root) / "report.json").write_text(json.dumps({**good, "kind": "branch"}))      # a native branch
                with self.assertRaises(ValueError):
                    SW.policies_bca({}, rt, None, None, None, "cutoff", {"branch": root}, ())
        self.assertEqual((seen[0].arm, seen[0].kappa), ("bca", SW.KAPPA_A))
        self.assertIs(rt.branch, before)

    def test_anchor_values_gain_u(self):
        class Policy:
            def width(self, obs, actions):
                return np.abs(actions[:, 0]) + 0.1
        anchors = [{"obs": np.zeros(14, np.float32)} for _ in range(3)]
        with mock.patch.object(SW, "_ORIGINAL_ANCHOR_VALUES", lambda *a, **k: {"q1": 1}):
            out = SW.anchor_values_with_u(Policy(), anchors, "stall3", None, stall_trained=True)
        self.assertEqual(out["U"].shape, (3, len(SW.E.GATED_SLOTS)))
        self.assertEqual(out["q1"], 1)


def eval_report(entry_by_run, j_by_run, seed, penalty_fixes_q=True, with_u=True):
    """Synthetic anchors where the truth says s10 is worse than the knee (true contrast -1) and raw Q1 says better
    (+0.5); with penalty_fixes_q the width rises by 1.0 at s10, so Q1 - 63 U ranks it correctly."""
    names = list(SW.E.GATED_SLOTS)
    k, i = names.index("edge_knee"), names.index("s10")
    qpi = np.zeros((64, len(names)))
    qpi[:, i] = -1.0
    q1 = np.zeros((64, len(names)))
    q1[:, i] = 0.5
    u = np.zeros((64, len(names)))
    if penalty_fixes_q:
        u[:, i] = 1.0
    runs = {}
    for name in (f"s{seed}_{a}_{w}" for a in SM.ARMS for w in SM.WORLDS):
        anchors = {"gated_slots": names, "q1": q1.tolist(), "qpi_clip": qpi.tolist(), "valid": [True] * 64}
        if with_u:
            anchors["U"] = u.tolist()
        runs[name] = {"020000": {"own_world": {"e2_rates": {"r_deep_minus": entry_by_run[name]}, "e2/pi": {"J": j_by_run[name]}},
                                 "anchors_e2": anchors}}
    return {"runs": runs, "transitions": 10, "status": "completed"}


def audit(share_inward):
    n = 100
    a = np.tile([-0.9, 0, 0, 0], (n, 1))
    q = np.tile([0.5, 0, 0, 0], (n, 1))
    pen = np.zeros((n, 4))
    pen[: int(share_inward * n), 0] = -1.0
    return {"per_row": {"actor_actions": a.tolist(), "high_load": [True] * n,
                        "action_loss_gradients": {"q": q.tolist(), "bc": np.zeros((n, 4)).tolist(), "actual_penalty": pen.tolist()}}}


class Decision(unittest.TestCase):
    def build(self, d_entry_stall, d_entry_nostall, d_j_stall, d_j_nostall, seeds=SM.SEEDS, fixes_q=True):
        bca, native, audits = {}, {}, {}
        for i in SM.SEEDS:
            e_n = {f"s{i}_{a}_{w}": 0.7 for a in SM.ARMS for w in SM.WORLDS}
            j_n = {k: -80.0 for k in e_n}
            e_b = {k: v + (d_entry_stall if "stall3" in k else d_entry_nostall) for k, v in e_n.items()}
            j_b = {k: v + (d_j_stall if "stall3" in k else d_j_nostall) for k, v in j_n.items()}
            native[i] = eval_report(e_n, j_n, i, with_u=False)
            if i in seeds:
                bca[i] = eval_report(e_b, j_b, i, penalty_fixes_q=fixes_q)
                audits.update({k: audit(0.6) for k in e_n})
        return SM.decide(bca, native, audits, kappa=63)

    def test_effect_requires_selectivity_and_return(self):
        out = self.build(-0.2, -0.02, 15.0, -1.0)
        self.assertEqual({a: out[a]["verdict"] for a in SM.ARMS}, {"cutoff": "EFFECT", "random": "EFFECT"})
        self.assertAlmostEqual(out["random"]["seeds"][0]["placement"], -0.18)
        self.assertAlmostEqual(out["random"]["seeds"][0]["stall3"]["applied_reversal_descriptive"]["share"], 0.6)
        q = out["random"]["seeds"][0]["stall3"]["anchors_bca"]
        self.assertEqual((q["raw_correct_share_s10"], q["penalized_correct_share_s10"]), (0.0, 1.0))
        self.assertAlmostEqual(q["median_kc_s10"], 1.5)
        self.assertAlmostEqual(q["median_kc_penalized_s10"], 1.5 - 63.0)
        self.assertAlmostEqual(out["random"]["seeds"][0]["q_rank_gain_s10"], 1.0)

    def test_behavior_without_a_q_correction_is_not_effect(self):
        out = self.build(-0.2, -0.02, 15.0, -1.0, fixes_q=False)
        self.assertEqual(out["random"]["verdict"], "BEHAVIORAL EFFECT NOT THROUGH Q")

    def test_equal_reduction_in_both_worlds_is_non_selective(self):
        self.assertEqual(self.build(-0.2, -0.2, 15.0, -1.0)["random"]["verdict"], "NON-SELECTIVE REDUCTION")

    def test_small_return_gain_or_large_nostall_cost_is_not_effect(self):
        self.assertEqual(self.build(-0.2, -0.02, 5.0, -1.0)["random"]["verdict"], "NON-SELECTIVE REDUCTION")
        self.assertEqual(self.build(-0.2, -0.02, 15.0, -8.0)["random"]["verdict"], "NON-SELECTIVE REDUCTION")

    def test_no_effect_is_one_sided_and_harm_is_separate(self):
        self.assertEqual(self.build(0.01, 0.0, 0.0, 0.0)["cutoff"]["verdict"], "NO EFFECT")
        self.assertEqual(self.build(0.10, 0.0, 0.0, 0.0)["cutoff"]["verdict"], "HARM")
        self.assertEqual(self.build(0.0, 0.0, 0.0, -25.0)["cutoff"]["verdict"], "HARM")
        self.assertEqual(self.build(-0.07, 0.0, 0.0, 0.0)["cutoff"]["verdict"], "MIXED")

    def test_missing_seed_is_incomplete_arm(self):
        self.assertEqual(self.build(-0.2, -0.02, 15.0, -1.0, seeds=(0, 1))["random"]["verdict"], "INCOMPLETE ARM")

    def test_non_finite_values_never_meet_a_bar(self):
        self.assertEqual(self.build(-0.2, -0.02, float("nan"), -1.0)["random"]["verdict"], "NON-SELECTIVE REDUCTION")


def native_reports():
    return {i: json.loads((screen.EXP / f"evaluate_s{i}" / "report.json").read_text(encoding="utf-8")) for i in SM.SEEDS}


class Supervisor(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        for f in screen.MANIFEST_FILES:
            shutil.copy2(HERE / f, self.root / f)
        self.manifest = self.root / "SCREEN_MANIFEST.json"
        self.manifest.write_text(json.dumps(screen.manifest_body(self.root)), encoding="utf-8")
        kappa = json.loads((HERE / "DOSE.json").read_text())["kappa_A"]
        self.approval = self.root / "SCREEN_APPROVAL.json"
        self.approval.write_text(json.dumps({"approval": screen.APPROVAL_WORDS.format(kappa=kappa),
                                             "screen_manifest_sha256": sha_file(self.manifest), "kappa_A": kappa}), encoding="utf-8")
        self.native = native_reports()
        self.patches = [mock.patch.object(screen, "MANIFEST", self.manifest), mock.patch.object(screen, "APPROVAL", self.approval),
                        mock.patch.object(screen, "RUN", self.root / "screen_run")]
        for p in self.patches:
            p.start()

    def tearDown(self):
        for p in self.patches:
            p.stop()
        self.tmp.cleanup()

    def dispatcher(self, fail=(), preflight_ok=True):
        seen = []

        def dispatch(job_dir, cap):
            job = json.loads((job_dir / "job.json").read_text())
            seen.append(job["id"])
            assert "integrity" in job
            ok = job["id"] not in fail and (job["kind"] != "preflight" or preflight_ok)
            report = {"status": "completed" if ok else "failed", "kind": job["kind"]}
            if job["kind"] == "bca_branch" and ok:
                (job_dir / "gradient_audit_020000.json").write_text(json.dumps(audit(0.4)))
                report.update(gradient_audits={"20000": {"path": "gradient_audit_020000.json"}})
            if job["kind"] == "evaluate":
                i = job["seed_index"]
                runs = {}
                for a in SM.ARMS:
                    for w in SM.WORLDS:
                        name = f"s{i}_{a}_{w}"
                        own = self.native[i]["runs"][name]["020000"]["own_world"]
                        runs[name] = {"020000": {"own_world": {
                            "e2_rates": {"r_deep_minus": own["e2_rates"]["r_deep_minus"] - (0.2 if w == "stall3" else 0.02)},
                            "e2/pi": {"J": own["e2/pi"]["J"] + (50.0 if w == "stall3" else -1.0)}},
                            "anchors_e2": self.with_q_fix(self.native[i]["runs"][name]["020000"]["anchors_e2"])}}
                report.update(runs=runs, transitions=10)
            (job_dir / "report.json").write_text(json.dumps(report))
            return {"exit_code": 0 if ok else 1, "elapsed_seconds": 1.0}
        return dispatch, seen

    @staticmethod
    def with_q_fix(anchors):
        """The real native anchors plus a width that rises by 1.0 at s10, so Q1 - 63 U ranks s10 below the knee."""
        names = anchors["gated_slots"]
        u = np.zeros((len(anchors["q1"]), len(names)))
        u[:, names.index("s10")] = 1.0
        return {**anchors, "U": u.tolist()}

    def test_end_to_end_effect_against_the_real_native_evaluations(self):
        dispatch, seen = self.dispatcher()
        out = screen.execute(dispatch=dispatch)
        self.assertEqual(out["label"], "COMPLETE", out.get("reasons"))
        self.assertEqual(out["verdicts"], {"cutoff": "EFFECT", "random": "EFFECT"})
        self.assertAlmostEqual(out["decision"]["random"]["seeds"][0]["stall3"]["delta_entry"], -0.2, places=9)
        self.assertEqual(seen[0], "preflight_1")
        self.assertEqual(len([s for s in seen if s.startswith("bca_branch_")]), 12)
        self.assertEqual([s for s in seen if s.startswith("evaluate_")], ["evaluate_s0", "evaluate_s1", "evaluate_s2"])
        self.assertIn("physical", out)
        with self.assertRaises(ValueError):                                      # one attempt only
            screen.execute(dispatch=dispatch)
        self.assertEqual(screen.decide_only()["verdicts"], out["verdicts"])     # JSON-only recompute

    def test_a_failed_branch_is_a_dose_failure_and_the_rest_still_runs(self):
        dispatch, seen = self.dispatcher(fail={"bca_branch_s1_random_stall3"})
        out = screen.execute(dispatch=dispatch)
        self.assertEqual(out["label"], "COMPLETE")
        self.assertEqual(len([s for s in seen if s.startswith("bca_branch_")]), 12)
        self.assertNotIn("evaluate_s1", seen)
        self.assertEqual(out["dose_failures"], ["s1_random_stall3"])
        self.assertEqual(out["verdicts"], {"cutoff": "INCOMPLETE ARM", "random": "INCOMPLETE ARM"})
        self.assertTrue((self.root / "screen_run" / "attempt_1" / "SCREEN_RESULT.json").is_file())

    def test_preflight_failure_uses_no_attempt(self):
        dispatch, _ = self.dispatcher(preflight_ok=False)
        self.assertEqual(screen.execute(dispatch=dispatch)["label"], "PREFLIGHT FAILED")
        self.assertFalse((self.root / "screen_run" / "attempt_1").exists())
        dispatch2, seen2 = self.dispatcher()
        self.assertEqual(screen.execute(dispatch=dispatch2)["label"], "COMPLETE")
        self.assertEqual(seen2[0], "preflight_2")

    def test_a_crash_after_the_attempt_still_writes_the_result(self):
        dispatch, _ = self.dispatcher()
        with mock.patch.object(screen, "decide_from", side_effect=RuntimeError("boom")):
            out = screen.execute(dispatch=dispatch)
        self.assertEqual(out["label"], "INCOMPLETE")
        result = json.loads((self.root / "screen_run" / "attempt_1" / "SCREEN_RESULT.json").read_text())
        self.assertIn("boom", result["reasons"][0])

    def test_approval_binds_files_pins_and_comparators(self):
        bad = self.root / "BAD_APPROVAL.json"
        bad.write_text(json.dumps({"approval": "yes", "screen_manifest_sha256": sha_file(self.manifest)}), encoding="utf-8")
        with self.assertRaises(ValueError):
            screen.check_approval(bad, self.manifest)
        body = json.loads(self.manifest.read_text())
        body["comparator"]["report.json"] = "0" * 64
        self.manifest.write_text(json.dumps(body), encoding="utf-8")
        with self.assertRaises(ValueError):
            screen.check_approval(self.approval, self.manifest)


class Wiring(unittest.TestCase):
    def test_worker_accepts_the_screen_integrity_record(self):
        import integrity
        record = screen.integrity_record()
        self.assertEqual(integrity.verify_record(record, HERE), len(record["files"]))
        self.assertEqual(SW.v3_added_files(record), [])

    def test_real_module_identities_after_patching(self):
        try:
            runtime, evaluate_worker, _, _ = SW.step03.rerun1()
        except Exception as exc:                                                  # pragma: no cover
            self.skipTest(f"rerun1 not importable here: {exc}")
        SW.patch_runtime(runtime)
        self.assertIs(runtime.branch.__globals__["make_adapter"], runtime.make_adapter)
        self.assertEqual(runtime._dose_screen_kappa, SW.KAPPA_A)
        self.assertIs(SW.W1.E, SW.E)
        self.assertIs(SW.W1.evaluate.__globals__, vars(SW.W1))
        import inspect
        self.assertEqual(list(inspect.signature(evaluate_worker.FrozenPolicy.width).parameters), ["self", "observations", "actions"])


if __name__ == "__main__":
    unittest.main()

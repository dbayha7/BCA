"""Stage 2 (A5) tests on Windows: detection AUROC and FPR on known answers, the R1-R3 screens, recovery, repair
guards and label precedence; the per-fork Stage 2 seed views; and the supervisor end to end with a fake dispatcher
(PROMISING, DETECTION-FAIL before any branch, re-attempt sealing). No checkpoint is opened and no model is trained."""
from pathlib import Path
from unittest import mock
import json
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.dont_write_bytecode = True

import numpy as np

import seeds as S
import stage1
import stage2
import stage2_math as SM
import stage2_worker as W2
import step03
import units as U
from core import sha_file


class Math(unittest.TestCase):
    def test_auroc_steps_and_ties(self):
        ood = np.asarray([[3.0], [1.0]])
        ins = np.asarray([[1.0, 2.0, 0.0, 5.0, 1.0], [1.0, 1.0, 1.0, 1.0, 1.0]])
        mean, per = SM.pairwise_auroc(ood, ins)
        self.assertAlmostEqual(per[0], 0.8)                         # 4 of 5 below 3.0
        self.assertAlmostEqual(per[1], 0.5)                         # all ties
        self.assertAlmostEqual(mean, 0.65)

    def test_fpr_at_95(self):
        ood = np.arange(100, dtype=float)[None, :]                 # catch 95%: threshold 5.0
        ins = np.asarray([[4.0, 5.0, 6.0, 1.0]])
        fpr, thr = SM.fpr_at_95(ood, ins)
        self.assertEqual((thr, fpr), (5.0, 0.5))

    def test_detection_rules(self):
        n = 48
        edge = {k: np.full(n, 1.0) for k in SM.NEAR_EDGE}
        cut = {**edge, **{k: np.full(n, 2.0) for k in ("s05", "s10", "s15", "s20", "p10")}}
        ran = {**edge, "s05": np.full(n, 1.0), "s10": np.full(n, 2.0), "s15": np.full(n, 2.0), "s20": np.full(n, 2.0), "p10": np.full(n, 2.0)}
        d = SM.detection(cut, ran)
        self.assertTrue(d["D1"] and d["D2"])
        self.assertAlmostEqual(d["D2_difference"], 0.5)
        d = SM.detection(cut, cut)                                   # U grows with |a0| in both: D1 passes, D2 fails
        self.assertTrue(d["D1"] and not d["D2"])

    def test_t_quantile(self):
        try:
            from scipy.stats import t
        except ImportError:
            self.skipTest("scipy not installed")
        self.assertAlmostEqual(t.ppf(0.975, 2), SM.T_975_DF2, places=9)       # scipy rounds at ~5e-11
        x = SM.T_975_DF2
        self.assertAlmostEqual(0.5 + x / (2 * np.sqrt(2 + x * x)), 0.975, places=15)   # the df = 2 CDF, exactly

    def j(self, g=(4.0, 5.0, 3.5), dist=(1.0, 1.0, 1.0), h_extra=(0.0, 0.0, 0.0)):
        out = {}
        for s in range(3):
            out[s] = {("stall_cutoff", "rep"): 100.0, ("stall_cutoff", "bca"): 100.0 + g[s] + 1.0,
                      ("stall_cutoff", "distance"): 100.0 + g[s] + 1.0 - dist[s],
                      ("stall_random", "rep"): 110.0, ("stall_random", "bca"): 111.0,
                      ("nostall_cutoff", "rep"): 120.0, ("nostall_cutoff", "bca"): 121.0 + h_extra[s]}
        return out

    def test_screens_and_labels(self):
        screen = SM.screens(self.j())
        self.assertTrue(screen["R1"] and screen["R2"] and screen["R3"])
        self.assertAlmostEqual(screen["G_mean"], 4.1666666666, places=6)
        ok_det = {s: {"D1": True, "D2": True} for s in range(3)}
        gains = {"stall_cutoff/bca": [80.0, 90.0, 70.0]}
        self.assertEqual(SM.label(v1_reasons=[], detection_by_seed=ok_det, gains=gains, screen=screen,
                                  damage_by_seed={0: True, 1: False, 2: True})[0], "PROMISING (damage not reproduced in seed 1)")
        self.assertEqual(SM.label(v1_reasons=[], detection_by_seed=ok_det, gains={"x": [80.0, 40.0, 70.0]}, screen=screen,
                                  damage_by_seed={})[0], "INCONCLUSIVE")
        bad = SM.screens(self.j(g=(-1.0, -2.0, -0.5)))
        self.assertEqual(SM.label(v1_reasons=[], detection_by_seed=ok_det, gains=gains, screen=bad, damage_by_seed={})[0],
                         "NOT-PROMISING-consistent")
        mixed = SM.screens(self.j(g=(10.0, -1.0, 2.0)))              # sign fails in one seed only; mean 3.67 passes
        self.assertEqual(SM.label(v1_reasons=[], detection_by_seed=ok_det, gains=gains, screen=mixed, damage_by_seed={})[0],
                         "NOT-PROMISING-mixed")
        r3 = SM.screens(self.j(h_extra=(-10.0, -10.0, -10.0)))
        self.assertFalse(r3["R3"])
        self.assertEqual(SM.label(v1_reasons=[], detection_by_seed=ok_det, gains=gains, screen=r3, damage_by_seed={})[0],
                         "NOT-PROMISING-consistent")
        self.assertEqual(SM.label(v1_reasons=["x"], detection_by_seed=ok_det, gains=gains, screen=screen, damage_by_seed={})[0],
                         "INCOMPLETE")

    def test_recovery_and_repair_guards(self):
        rec = SM.recovery(self.j())
        self.assertTrue(rec["computed"])
        self.assertAlmostEqual(rec["recovery"], (np.mean([5.0, 6.0, 4.5])) / 10.0)
        flat = self.j()
        for s in flat:
            flat[s][("stall_random", "rep")] = 100.0
        self.assertEqual(SM.recovery(flat)["reason"], "not computed (guard)")
        seed = {"bca_entry": 0.05, "rep_entry": 0.30, "bca_delta_clip": 1.0, "rep_delta_clip": 1.5,
                "valid_both": np.ones(4, bool), "rep_kc_s10": np.full(4, 0.2), "bca_kc_s10_score": np.zeros(4)}
        out = SM.repair(seed)
        self.assertTrue(out["entry"])
        self.assertEqual((out["delta_clip"], out["kc"]), ("not computed (guard)", "not computed (guard)"))


class Seeds(unittest.TestCase):
    def test_views_share_init_seeds_and_differ_in_branch_seeds(self):
        record = S.load()
        for i in range(3):
            v = W2.stage2_view(i, "cutoff", "stall3")["seeds"]
            for f in ("native_seed", "aux_seed"):
                self.assertEqual(v[i][f], record["training"][i][f])
            self.assertNotEqual(v[i]["branch_native_seed"], record["training"][i]["branch_native_seed"])
            self.assertNotEqual(v[i]["branch_native_seed"], W2.stage2_view(i, "random", "stall3")["seeds"][i]["branch_native_seed"])

    def test_job_list_and_research_only(self):
        self.assertEqual(len(stage2.JOB_IDS), 1 + 21 + 3)
        self.assertEqual(sum(j.startswith("branch_s0_stall_cutoff_") for j in stage2.JOB_IDS), 3)
        with self.assertRaises(ValueError):
            W2.fork_job({"provenance": "smoke"}, 0, "cutoff", "stall3")


# ------------------------------------------------------------------------------------------------ supervisor

GATED = np.r_[np.ones(48, bool), np.zeros(16, bool)]


def precheck_report(detect_ok=True):
    slots = list(W2.DETECT_SLOTS)
    def arm(kind):
        u = np.ones((64, len(slots)))
        for k, name in enumerate(slots):
            if name.startswith("s") or name == "p10":
                u[:, k] = 2.0 if (kind == "cutoff" or name != "s05") else 1.0
        return {"U": u, "Q1": -30 - u, "own_U": u, "own_a123_past_knee": np.r_[np.zeros(60, bool), np.ones(4, bool)],
                "magnitude_U": {"a1_zero": np.ones(64), "a1_plus_0.7": np.ones(64)}}
    seeds = {i: {"cutoff": arm("cutoff"), "random": arm("random" if detect_ok else "cutoff")} for i in range(3)}
    return {"kind": "precheck", "slots": slots, "gated": GATED, "seeds": seeds}


def evaluation_report(i):
    names = ["edge_knee", "s10"]
    def anchors(bca):
        q = np.tile([-30.0, -32.0], (64, 1))
        return {"gated_slots": names, "qpi_clip": q, "q1": q + (np.asarray([0.0, 1.5]) if not bca else 0.0), "U": np.zeros((64, 2)),
                "valid": np.ones(64, bool), "times": list(np.repeat([0, 25, 50, 75], 16))}
    def free(j, entry, dc):
        return {"e2/pi": {"J": j}, "e1/pi": {"J": 200.0}, "e2/delta_clip": dc, "e2_rates": {"r_deep_minus": entry}}
    cells = {}
    for cell, (data_arm, world, arms) in W2.FORKS.items():
        base = {"stall_cutoff": 100.0, "stall_random": 110.0, "nostall_cutoff": 120.0}[cell]
        branches = {}
        for arm in arms:
            j = base + {"native": 0.0, "bca": 5.0 if cell == "stall_cutoff" else 1.0, "distance": 4.0}[arm]
            branches[arm] = {"free": free(j, 0.3 if arm == "native" else 0.05, 3.0 if arm == "native" else 0.5)}
            if cell == "stall_cutoff" and arm in ("bca", "native"):
                branches[arm]["anchors"] = anchors(arm == "bca")
        cells[cell] = {"J_E1_untrained": 100.0, "branches": branches, "stage1_native_on_stage2_bank": free(base - 1.0, 0.3, 3.0)}
    return {"kind": "evaluate", "seed_index": i, "cells": cells, "transitions": 10}


class FakeDispatch:
    def __init__(self, fail=(), detect_ok=True):
        self.fail, self.detect_ok, self.seen = set(fail), detect_ok, []

    def __call__(self, job_dir, cap):
        job = json.loads((job_dir / "job.json").read_text(encoding="utf-8"))
        self.seen.append(job["id"])
        body = {"preflight": {}, "precheck": precheck_report(self.detect_ok),
                "branch": {"branch_arm": job.get("branch_arm"), "minibatches_differ_from_stage1_native": True}}.get(job["kind"])
        if job["kind"] == "evaluate":
            body = evaluation_report(job["seed_index"])
        if job["kind"] == "branch":
            assert job["warmup_dir"].endswith(f"warmup_s{job['seed_index']}_{job['data_arm']}_{job['world']}")
        ok = job["id"] not in self.fail
        report = {**body, "transitions": 100, "job": job["id"], "kind": job["kind"], "status": "completed" if ok else "failed",
                  "unit_label": "COMPLETE" if ok else "INCOMPLETE", "elapsed_seconds": 1.0,
                  "producer_sha256": U.fingerprint("stage1_worker" if job["kind"] == "preflight" else "stage2_worker", job["kind"])}
        step03._write_new(job_dir / "report.json", W2.W1.sanitize(report))
        return {"exit_code": 0 if ok else 1, "timed_out": False, "interrupted": False, "elapsed_seconds": 1.0}


class Supervisor(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        root = self.root = Path(self.tmp.name)
        s1 = root / "stage1_run" / "attempt_1"
        for unit in stage1.JOB_IDS:
            (s1 / unit).mkdir(parents=True)
            step03._write_new(s1 / unit / "report.json", {"status": "completed"})
            step03._write_new(s1 / unit / "SUPERVISOR_RESULT.json", {"completed": True})
        step03._write_new(s1 / "report.json", {"label": "PASS", "approval": {"manifest_sha256": "a4-fake"}, "reproduced_units": [],
                                               "a5_precondition": {"ok": True}, "a3": {"collection_dir": "c"},
                                               "unit_dirs": {u: str(s1 / u) for u in stage1.JOB_IDS}})
        manifest = {"files": {n: sha_file(stage2.HERE / n) for n in sorted(stage2.A5_REQUIRED_FILES)}}
        (root / "A5_MANIFEST.json").write_text(json.dumps(manifest), encoding="utf-8")
        self.patches = [mock.patch.object(stage2, "RUN", root / "stage2_run"), mock.patch.object(stage2, "MANIFEST", root / "A5_MANIFEST.json"),
                        mock.patch.object(stage2, "APPROVAL", root / "A5_APPROVAL.json"), mock.patch.object(stage1, "RUN", root / "stage1_run"),
                        mock.patch.object(stage2, "A4_MANIFEST_SHA256", "a4-fake"),
                        mock.patch.object(stage2.A, "verify_upstream", lambda here, pins: {"pins": sorted(pins)})]
        for p in self.patches:
            p.start()

    def tearDown(self):
        for p in self.patches:
            p.stop()
        self.tmp.cleanup()

    def approve(self, attempt):
        (self.root / "A5_APPROVAL.json").write_text(json.dumps({
            "approval": stage2.APPROVAL_WORDS + f", attempt {attempt}", "attempt": attempt,
            "package_manifest_sha256": sha_file(self.root / "A5_MANIFEST.json")}), encoding="utf-8")

    def test_promising_end_to_end(self):
        self.approve(1)
        dispatch = FakeDispatch()
        report = stage2.execute(dispatch=dispatch)
        self.assertEqual(report["label"], "PROMISING", report.get("reasons"))
        self.assertEqual(dispatch.seen, ["preflight"] + stage2.JOB_IDS)
        decision = json.loads((stage2.RUN / "attempt_1" / "decision.json").read_text(encoding="utf-8"))
        self.assertTrue(decision["repair"]["0"]["entry"])
        self.assertTrue(decision["recovery"]["computed"])

    def test_detection_fail_trains_no_branch(self):
        self.approve(1)
        dispatch = FakeDispatch(detect_ok=False)
        report = stage2.execute(dispatch=dispatch)
        self.assertEqual(report["label"], "DETECTION-FAIL")
        self.assertEqual(dispatch.seen, ["preflight", "precheck"])

    def test_stage1_unit_that_failed_to_reproduce_is_refused(self):
        unit = self.root / "stage1_run" / "attempt_1" / "branch_s0_cutoff_stall3"
        step03._write_new(unit / "NOT_REPRODUCED.json", {"reference_dir": "x"})
        self.approve(1)
        with self.assertRaises(ValueError):
            stage2.execute(dispatch=FakeDispatch())

    def test_stage1_must_pass(self):
        path = self.root / "stage1_run" / "attempt_1" / "report.json"
        record = json.loads(path.read_text(encoding="utf-8"))
        path.write_text(json.dumps({**record, "label": "FAIL-mixed"}), encoding="utf-8")
        self.approve(1)
        with self.assertRaises(ValueError):
            stage2.execute(dispatch=FakeDispatch())

    def test_reattempt_runs_only_the_failed_branch(self):
        self.approve(1)
        first = stage2.execute(dispatch=FakeDispatch(fail={"branch_s2_stall_random_bca"}))
        self.assertEqual(first["failed_unit"], "branch_s2_stall_random_bca")
        self.approve(2)
        dispatch = FakeDispatch()
        report = stage2.execute(dispatch=dispatch)
        self.assertEqual(dispatch.seen, ["preflight", "branch_s2_stall_random_bca", "branch_s2_stall_random_native", "branch_s2_nostall_cutoff_bca",
                                         "branch_s2_nostall_cutoff_native", "evaluate_s0", "evaluate_s1", "evaluate_s2"])
        self.assertEqual(report["label"], "PROMISING", report.get("reasons"))


if __name__ == "__main__":
    unittest.main()

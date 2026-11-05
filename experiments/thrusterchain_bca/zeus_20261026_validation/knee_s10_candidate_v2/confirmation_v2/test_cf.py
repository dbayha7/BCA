"""Confirmation v2 tests (CPU): the fresh seeds (five rows, disjoint from the screen's and confirmation v1's); the
worker's changes (v2 rows, kappa 189, the cf_evaluate kind, the device kinds); the pre-registered readings on
synthetic reports (C1 every seed, C2 with the 3-of-5 majority, the I1 tolerance 1e-5, the description); the
supervisor end to end with a fake dispatcher (50 units, INCOMPLETE and reuse, the INCONCLUSIVE re-attempt at the v2
tolerance, a third attempt only after an INCOMPLETE second and no attempt 4, approval and manifest checks). No GPU,
no checkpoint opened, no sealed file touched."""
from pathlib import Path
from types import SimpleNamespace
import json
import os
import shutil
import sys
import tempfile
import unittest

HERE = Path(__file__).resolve().parent
CAND = HERE.parent
for _p in (str(CAND.parent / "knee_edge_ood_v3"), str(CAND / "formal_g2"), str(CAND / "native_screen"), str(CAND / "overestimation_v1"), str(HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)
sys.dont_write_bytecode = True
os.environ.setdefault("JAX_PLATFORMS", "cpu")

import numpy as np

import screen_bind as SB                                     # noqa: E402  first: substitutes the recorder
import cf                                                    # noqa: E402
import cf_math as CM                                         # noqa: E402
import cf_seeds as CS                                        # noqa: E402
import cf_worker as CW                                       # noqa: E402
import screen_seeds as SS                                    # noqa: E402
from core import sha_file                                    # noqa: E402

DOSES = CM.DOSES


class Seeds(unittest.TestCase):
    def test_fresh_and_clean(self):
        record = CS.load()
        self.assertTrue(record["audit"]["clean"])
        rows = CS.training_rows()
        self.assertEqual([r["index"] for r in rows], [0, 1, 2, 3, 4])
        mine = {r[f] for r in rows for f in SS.STAGE1_FIELDS}
        screen = {r[f] for r in SS.training_rows() for f in SS.STAGE1_FIELDS}
        self.assertFalse(screen & mine)                                             # disjoint from the screen's
        self.assertFalse(CS.v1_values() & (mine | set(record["numpy"].values())))   # and from confirmation v1's
        self.assertEqual(record["screen_seeds_sha256"], sha_file(SS.SEEDS_PATH))
        self.assertEqual(record["confirmation_v1_seeds_sha256"], sha_file(CS.V1_SEEDS_PATH))
        self.assertEqual(CS.NAMESPACE, "knee_s10_candidate_v2/confirmation_v2/")


class Worker(unittest.TestCase):
    def test_protocol_view_gives_the_confirmation_rows(self):
        self.assertEqual(CW.W1.protocol_view({"provenance": "research"})["seeds"], CS.training_rows())
        with self.assertRaises(ValueError):
            CW.W1.protocol_view({"provenance": "research", "training_seeds": [1, 2, 3]})

    def test_apply_kappa_sets_the_dose_and_patches_the_runtime(self):
        original_rerun1, original_kappa = CW.SW.step03.rerun1, CW.SW.KAPPA
        try:
            runtime = SimpleNamespace(make_adapter=lambda context, state, arm, k, *a, **kw: (arm, k))
            CW.SW.step03.rerun1 = lambda: (runtime, "ew", "ga", "ri")
            self.assertEqual(CW.apply_kappa({"kappa": 189.0}), 189.0)
            self.assertEqual(CW.SW.KAPPA, 189.0)
            rt, *_ = CW.SW.rerun1()
            self.assertEqual(rt.make_adapter(None, None, "bca", 63.0), ("bca", 189.0))
            self.assertEqual(rt.make_adapter(None, None, "native", 0.0), ("native", 0.0))
            for other in (21.0, 63.0, 50.0):                                           # v1's doses are not v2's
                with self.assertRaises(ValueError):
                    CW.apply_kappa({"kappa": other})
            self.assertEqual(CW.apply_kappa({}), 189.0)                                # native jobs: the default dose
        finally:
            CW.SW.step03.rerun1, CW.SW.KAPPA = original_rerun1, original_kappa

    def test_device_kind_binding(self):
        """Change 4 (v1's amendment A2, carried): the recorded kind becomes the machine's one allowed kind."""
        import types
        import jax
        import device_kinds as DK
        K = CW.SW.step03.K
        recorded, devices = K.RECORDED_DEVICE_KIND, jax.devices
        self.assertIs(CW.SW.step03.set_worker_environment, CW.set_worker_environment)
        try:
            for kind in DK.ALLOWED_DEVICE_KINDS:
                jax.devices = lambda *a, kind=kind: [types.SimpleNamespace(device_kind=kind)]
                self.assertEqual(CW.bind_device_kind(), kind)
                self.assertEqual(K.RECORDED_DEVICE_KIND, kind)
            jax.devices = lambda *a: [types.SimpleNamespace(device_kind="NVIDIA A100-SXM4-80GB")]
            with self.assertRaises(ValueError):
                CW.bind_device_kind()
        finally:
            K.RECORDED_DEVICE_KIND, jax.devices = recorded, devices
        self.assertEqual(set(DK.ALLOWED_DEVICE_KINDS), {"NVIDIA GeForce RTX 5070 Ti Laptop GPU", "NVIDIA GeForce RTX 5070 Ti"})

    def test_caps(self):
        self.assertEqual((cf.JOB_SECONDS, cf.TRAIN_SECONDS, cf.EVAL_SECONDS, cf.EVAL_TRANSITIONS), (1_800, 13 * 3_600, 2 * 3_600, 3_000_000))

    def test_kinds(self):
        self.assertIs(CW.SW.KINDS["cf_evaluate"], CW.cf_evaluate)
        self.assertTrue({"warmup", "branch", "bca_branch"} <= set(CW.SW.KINDS))


# ------------------------------------------------------------------------------------------------ synthetic reports

SEEDS = cf.SEEDS
K = 189.0


def report(*, arm, seed, j, e_own, e_data=0.5, e_w=4.0, u=0.1, expo=0.1, expo_deep=None, i1=True, i2=True, i1_rel=0.0,
           returns=None, n_own=48, n_rows=192):
    rng = np.random.default_rng(seed + (0 if arm == "native" else 100))
    qpi_own = -20 + rng.normal(size=n_own)
    qpi_d = -20 + rng.normal(size=(n_rows, 4))
    q1_d = qpi_d + np.concatenate([np.full((n_rows, 1), e_data), np.full((n_rows, 3), e_w)], axis=1)
    rep = {"kind": "cf_evaluate", "arm": arm, "seed_index": seed, "status": "completed", "J": j, "exposure_w": expo,
           "returns": np.full(128, j) if returns is None else np.asarray(returns, float),
           "instrument": {"I1_ok": i1, "I2_ok": i2, "I1_max_rel": i1_rel},
           "own": {"qpi": qpi_own, "q1": qpi_own + e_own, "q2": qpi_own + e_own, "u": np.full(n_own, u)},
           "data": {"slots": ["logged", "s05", "s10", "s20"], "qpi": qpi_d, "q1": q1_d, "q2": q1_d, "u": np.full((n_rows, 4), u)}}
    if expo_deep is not None:
        rep["rates"] = {"by_band": {"deep": {"w_any_depth": expo_deep}, "mid": {"w_any_depth": expo}}}
    return rep


def world(*, dj, expo_native=0.3, expo_bca=None, j0=-80.0, de=-1.0, i1_rel=0.0, i1_rel_native=None, **kw):
    """i1_rel applies to both arms; i1_rel_native, when given, to the native arm only."""
    expo_bca = expo_bca or (0.2,) * 5
    native_i1 = i1_rel if i1_rel_native is None else i1_rel_native
    native = {i: report(arm="native", seed=i, j=j0, e_own=3.0, expo=expo_native, i1_rel=native_i1, **kw) for i in SEEDS}
    bca = {K: {i: report(arm="bca", seed=i, j=j0 + dj[i], e_own=3.0 + de, expo=expo_bca[i], i1_rel=i1_rel, **kw) for i in SEEDS}}
    return {"native": native, "bca": bca}


STALL_OK = (15.0, 45.0, 30.0, 8.0, 12.0)
NOSTALL_OK = (-1.0, -2.0, 0.5, -1.5, 1.0)


def both(stall3=None, nostall=None):
    return {"stall3": stall3 or world(dj=STALL_OK, expo_native=0.7, expo_bca=(0.3,) * 5),
            "nostall": nostall or world(dj=NOSTALL_OK, expo_native=0.8, expo_bca=(0.2,) * 5, j0=-60.0)}


def ns(dj, expo_bca=(0.8,) * 5):
    return world(dj=dj, expo_native=0.8, expo_bca=expo_bca, j0=-60.0)


class Readings(unittest.TestCase):
    def test_confirmed_and_calibrated(self):
        out = CM.decide(both())
        self.assertEqual(out["reading"], "CONFIRMED / CALIBRATED")
        self.assertEqual(out["description"]["table"]["stall3"]["dJ"], list(STALL_OK))
        self.assertEqual(out["paired"]["nostall"]["189"][0]["dJ"], -1.0)
        self.assertEqual(out["bars"]["confirmed_dose"], 189.0)
        self.assertEqual((out["bars"]["majority"], out["bars"]["i1_tol"]), (3, 1e-5))

    def test_c1_needs_every_one_of_five_seeds(self):
        out = CM.decide(both(stall3=world(dj=(15.0, 45.0, 30.0, 1.5, 12.0), expo_native=0.7)))
        self.assertEqual(out["C1"], "NOT CONFIRMED")
        self.assertIn("+1.500", out["reasons"][0])

    def test_c2_labels_and_the_majority(self):
        moved = (0.1,) * 5
        self.assertEqual(CM.decide(both(nostall=ns((-3.0, -4.0, -5.0, -4.0, -2.0))))["C2"], "COST BOUNDED")       # mean -3.6, worst -5
        self.assertEqual(CM.decide(both(nostall=ns((-41.4, -98.3, -20.5, -30.0, -10.0), moved)))["C2"], "OVER-CORRECTION")
        three = (0.1, 0.1, 0.1, 0.8, 0.8)
        self.assertEqual(CM.decide(both(nostall=ns((-9.0, -9.0, -9.0, 0.5, 0.5), three)))["C2"], "OVER-CORRECTION")  # 3 of 5
        two = (0.1, 0.1, 0.8, 0.8, 0.8)
        self.assertEqual(CM.decide(both(nostall=ns((-9.0, -9.0, 0.5, 0.5, 0.5), two)))["C2"], "PLACEBO MIXED")       # 2 of 5
        self.assertEqual(CM.decide(both(nostall=ns((-3.0, -9.0, -9.0, -9.0, 0.5))))["C2"], "PLACEBO MIXED")         # loses without leaving W
        self.assertEqual(CM.decide(both(nostall=ns((0.5, 1.0, -1.0, -2.0, 0.0))))["C2"], "CALIBRATED")
        pw = world(dj=(0.0,) * 5, expo_native=0.005, j0=-60.0)
        self.assertEqual(CM.decide(both(nostall=pw))["C2"], "PLACEBO WITHOUT STAKE")

    def test_validity_and_the_i1_tolerance(self):
        # v1's attempt-3 misses (1.86e-6) are inside v2's tolerance, whatever the evaluator's own 1e-6 flag says
        out = CM.decide(both(nostall=world(dj=NOSTALL_OK, expo_native=0.8, j0=-60.0, i1=False, i1_rel=1.86e-6)))
        self.assertEqual(out["reading"], "CONFIRMED / CALIBRATED", out["reasons"])
        out = CM.decide(both(nostall=world(dj=NOSTALL_OK, expo_native=0.8, j0=-60.0, i1_rel_native=2e-5)))
        self.assertEqual(out["reading"], "INCONCLUSIVE")
        self.assertIn("CONFIRMED", out["all_labels"])
        out = CM.decide(both(nostall=world(dj=NOSTALL_OK, expo_native=0.8, j0=-60.0, i2=False)))
        self.assertEqual(out["reading"], "INCONCLUSIVE")
        out = CM.decide(both(stall3=world(dj=STALL_OK, expo_native=0.0)))
        self.assertEqual(out["reading"], "NOT REACHED")

    def test_description_reads(self):
        lower = np.full(128, -61.0)
        lower[:32] = -59.0                                               # BCA lower on 96 of 128 starts
        nostall = world(dj=NOSTALL_OK, expo_native=0.8, j0=-60.0, expo_deep=0.9)
        nostall["bca"][K][0]["returns"] = lower
        out = CM.decide(both(nostall=nostall))
        d = out["description"]["table"]["nostall"]
        self.assertEqual(d["share_starts_lower"][0], 0.75)
        self.assertEqual(d["native_exposure_deep"], [0.9] * 5)
        self.assertIsNone(out["description"]["table"]["stall3"]["native_exposure_deep"][0])   # absent band: None


# ------------------------------------------------------------------------------------------------ the supervisor

def fake_dispatch(*, fail=(), instrument=None, eval_instrument=None):
    def dispatch(job_dir, cap):
        job = json.loads((Path(job_dir) / "job.json").read_text())
        kind, exit_code = job["kind"], 0
        rep = {"job": job["id"], "kind": kind, "status": "completed", "transitions": 10, "world": job.get("world"), "kappa": job.get("kappa")}
        if kind == "cf_evaluate":
            i, bca = job["seed_index"], job["arm"] == "bca"
            inst = dict(instrument or {})
            if eval_instrument and job["id"] in eval_instrument:
                inst.update(eval_instrument[job["id"]])
            if job["world"] == "stall3":
                rep.update(report(arm=job["arm"], seed=i, j=-80.0 + (STALL_OK[i] if bca else 0.0), e_own=3.0 - (1.0 if bca else 0.0),
                                  expo=(0.3 if bca else 0.7), **inst))
            else:
                rep.update(report(arm=job["arm"], seed=i, j=-60.0 + (NOSTALL_OK[i] if bca else 0.0), e_own=1.0,
                                  expo=(0.2 if bca else 0.8), **inst))
        if job["id"] in fail:
            rep["status"], exit_code = "failed", 1
        rep = {k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in rep.items()}
        for part in ("own", "data"):
            if part in rep:
                rep[part] = {k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in rep[part].items()}
        (Path(job_dir) / "report.json").write_text(json.dumps(rep), encoding="utf-8")
        return {"exit_code": exit_code, "elapsed_seconds": 1.0, "timed_out": False, "interrupted": False}
    return dispatch


class Supervisor(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.manifest = self.tmp / "CF_MANIFEST.json"
        self.manifest.write_text(json.dumps(cf.manifest_body(HERE), indent=2), encoding="utf-8")
        self.approval = self.tmp / "CF_APPROVAL.json"
        self.run = self.tmp / "run"
        self.write_approval(1)

    def write_approval(self, attempt):
        text = " ".join([cf.WORDS[0].format(attempt=attempt)] + list(cf.WORDS[1:]))
        self.approval.write_text(json.dumps({"approval": text, "package_manifest_sha256": sha_file(self.manifest), "attempt": attempt}))

    def execute(self, dispatch):
        return cf.execute(dispatch, run_dir=self.run, approval_path=self.approval, manifest_path=self.manifest, root=HERE,
                          integrity={"package_dir": str(HERE), "files": {}})

    def test_end_to_end(self):
        out = self.execute(fake_dispatch())
        self.assertEqual(out["reading"], "CONFIRMED / CALIBRATED", out.get("reasons"))
        units = {p.name for p in (self.run / "attempt_1").iterdir() if p.is_dir()}
        self.assertEqual(len(units), 50)
        self.assertFalse(any("_k21_" in u or "_k63_" in u for u in units))
        job = json.loads((self.run / "attempt_1" / "bca_branch_s4_k189_nostall" / "job.json").read_text())
        self.assertEqual((job["kappa"], job["world"], job["arm"]), (189.0, "nostall", "cutoff"))
        self.assertTrue(job["native_branch_dir"].endswith("branch_s4_nostall"))
        job = json.loads((self.run / "attempt_1" / "eval_bca_s3_k189_stall3" / "job.json").read_text())
        self.assertTrue(job["runs"]["branch"].endswith("bca_branch_s3_k189_stall3") and job["kappa"] == 189.0)
        self.assertEqual(cf.decide_only(self.run)["reading"], "CONFIRMED / CALIBRATED")
        self.assertEqual(out["description"]["table"]["stall3"]["dJ"], list(STALL_OK))
        self.assertEqual(out["package"], "knee_s10_candidate_v2/confirmation_v2")

    def test_incomplete_then_reuse(self):
        out = self.execute(fake_dispatch(fail=("bca_branch_s3_k189_nostall",)))
        self.assertEqual(out["reading"], "INCOMPLETE")
        self.write_approval(2)
        out = self.execute(fake_dispatch())
        self.assertEqual(out["reading"], "CONFIRMED / CALIBRATED", out.get("reasons"))
        self.assertTrue(out["units"]["bca_branch_s0_k189_nostall"].startswith("reused"))
        self.assertTrue(out["units"]["bca_branch_s4_k189_stall3"].startswith("reused"))
        self.assertEqual(out["units"]["bca_branch_s3_k189_nostall"], "completed")
        self.assertEqual(out["units"]["bca_branch_s4_k189_nostall"], "completed")      # never ran in attempt 1
        self.assertTrue(all(v == "completed" for k, v in out["units"].items() if k.startswith("eval_")))

    def test_inconclusive_reattempt_at_the_v2_tolerance(self):
        """A second attempt after INCONCLUSIVE only when every evaluation's I1 is within 1e-5 (the cause was I2 or a
        non-finite read): v1's 1.86e-6 misses would not block it; a 2e-5 miss does."""
        out = self.execute(fake_dispatch(eval_instrument={"eval_native_s2_stall3": {"i1_rel": 1.86e-6, "i2": False}}))
        self.assertEqual(out["reading"], "INCONCLUSIVE")
        self.write_approval(2)
        self.assertEqual(cf.check_approval(self.approval, self.manifest, self.run, HERE)["attempt"], 2)
        shutil.rmtree(self.run)
        self.write_approval(1)
        out = self.execute(fake_dispatch(eval_instrument={"eval_native_s2_stall3": {"i1_rel": 2e-5}}))
        self.assertEqual(out["reading"], "INCONCLUSIVE")
        self.write_approval(2)
        with self.assertRaisesRegex(ValueError, "within 1e-5 relative"):
            cf.check_approval(self.approval, self.manifest, self.run, HERE)

    def test_attempt_3_only_after_incomplete_attempt_2_and_no_attempt_4(self):
        self.assertEqual(cf.MAX_ATTEMPTS, 3)
        for n, reading in ((1, "INCOMPLETE"), (2, "INCONCLUSIVE")):
            unit = self.run / f"attempt_{n}" / "eval_native_s0_stall3"
            unit.mkdir(parents=True)
            (unit.parent / "CF_RESULT.json").write_text(json.dumps({"reading": reading}))
            (unit / "report.json").write_text(json.dumps({"status": "completed", "instrument": {"I1_max_rel": 0.0}}))
        self.write_approval(3)
        with self.assertRaisesRegex(ValueError, "follows only an INCOMPLETE attempt 2"):
            cf.check_approval(self.approval, self.manifest, self.run, HERE)
        (self.run / "attempt_2" / "CF_RESULT.json").write_text(json.dumps({"reading": "INCOMPLETE"}))
        self.assertEqual(cf.check_approval(self.approval, self.manifest, self.run, HERE)["attempt"], 3)
        (self.run / "attempt_3").mkdir()
        (self.run / "attempt_3" / "CF_RESULT.json").write_text(json.dumps({"reading": "INCOMPLETE"}))
        self.write_approval(4)
        with self.assertRaisesRegex(ValueError, "There is no attempt 4"):
            cf.check_approval(self.approval, self.manifest, self.run, HERE)

    def test_approval_and_manifest_checks(self):
        self.approval.write_text(json.dumps({"approval": "I approve something else", "package_manifest_sha256": sha_file(self.manifest), "attempt": 1}))
        with self.assertRaises(ValueError):
            self.execute(fake_dispatch())
        for missing in range(len(cf.WORDS)):                            # every sentence is required, the I1 one included
            text = " ".join(w.format(attempt=1) for j, w in enumerate(cf.WORDS) if j != missing)
            self.approval.write_text(json.dumps({"approval": text, "package_manifest_sha256": sha_file(self.manifest), "attempt": 1}))
            with self.assertRaises(ValueError):
                cf.check_approval(self.approval, self.manifest, self.run, HERE)
        self.write_approval(1)
        self.manifest.write_text(json.dumps({**json.loads(self.manifest.read_text()), "files": {"cf.py": "0" * 64}}))
        with self.assertRaises(ValueError):
            self.execute(fake_dispatch())
        body = cf.manifest_body()
        self.assertIn("cf_worker.py", body["files"])
        self.assertIn("CONFIRMATION_RULE.md", body["files"])
        self.assertIn("cf_seeds.json", body["files"])
        self.assertIn("oe_eval.py", body["overestimation_v1_files"])
        self.assertIn("screen_worker.py", body["native_screen_files"])
        record = cf.integrity_record()
        self.assertIn(str(HERE / "cf_worker.py"), record["files"])
        self.assertIn(str(CAND / "overestimation_v1" / "oe_eval.py"), record["files"])


if __name__ == "__main__":
    unittest.main()

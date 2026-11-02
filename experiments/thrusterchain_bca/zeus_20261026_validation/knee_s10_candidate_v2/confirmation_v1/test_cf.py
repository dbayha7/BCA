"""Confirmation tests (Windows, CPU): the fresh seeds; the worker's three changes (confirmation rows, the per-job
dose, the cf_evaluate kind); the pre-registered readings on synthetic reports; the supervisor end to end with a
fake dispatcher (54 units, INCOMPLETE and reuse by kappa, approval and manifest checks). No GPU, no checkpoint
opened, no sealed file touched."""
from pathlib import Path
from types import SimpleNamespace
import json
import os
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
        self.assertEqual([r["index"] for r in rows], [0, 1, 2])
        screen = {r[f] for r in SS.training_rows() for f in SS.STAGE1_FIELDS}
        self.assertFalse(screen & {r[f] for r in rows for f in SS.STAGE1_FIELDS})   # disjoint from the screen's
        self.assertEqual(record["screen_seeds_sha256"], sha_file(SS.SEEDS_PATH))


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
            self.assertEqual(CW.apply_kappa({"kappa": 21.0}), 21.0)
            self.assertEqual(CW.SW.KAPPA, 21.0)
            rt, *_ = CW.SW.rerun1()
            self.assertEqual(rt.make_adapter(None, None, "bca", 63.0), ("bca", 21.0))
            self.assertEqual(rt.make_adapter(None, None, "native", 0.0), ("native", 0.0))
            with self.assertRaises(ValueError):
                CW.apply_kappa({"kappa": 50.0})
            self.assertEqual(CW.apply_kappa({}), 63.0)                                 # native jobs: the default dose
        finally:
            CW.SW.step03.rerun1, CW.SW.KAPPA = original_rerun1, original_kappa

    def test_kinds(self):
        self.assertIs(CW.SW.KINDS["cf_evaluate"], CW.cf_evaluate)
        self.assertTrue({"warmup", "branch", "bca_branch"} <= set(CW.SW.KINDS))


# ------------------------------------------------------------------------------------------------ synthetic reports

def report(*, arm, seed, j, e_own, e_data=0.5, e_w=4.0, u=0.1, expo=0.1, i1=True, i2=True, i1_rel=0.0, n_own=48, n_rows=192):
    rng = np.random.default_rng(seed + (0 if arm == "native" else 100))
    qpi_own = -20 + rng.normal(size=n_own)
    qpi_d = -20 + rng.normal(size=(n_rows, 4))
    q1_d = qpi_d + np.concatenate([np.full((n_rows, 1), e_data), np.full((n_rows, 3), e_w)], axis=1)
    return {"kind": "cf_evaluate", "arm": arm, "seed_index": seed, "status": "completed", "J": j, "exposure_w": expo,
            "returns": np.full(128, j), "instrument": {"I1_ok": i1, "I2_ok": i2, "I1_max_rel": i1_rel},
            "own": {"qpi": qpi_own, "q1": qpi_own + e_own, "q2": qpi_own + e_own, "u": np.full(n_own, u)},
            "data": {"slots": ["logged", "s05", "s10", "s20"], "qpi": qpi_d, "q1": q1_d, "q2": q1_d, "u": np.full((n_rows, 4), u)}}


def world(*, dj_by_dose, expo_native=(0.3, 0.3, 0.3), expo_by_dose=None, j0=-80.0, de=(-1.0, -1.0, -1.0), **kw):
    expo_by_dose = expo_by_dose or {k: (0.2, 0.2, 0.2) for k in DOSES}
    native = {i: report(arm="native", seed=i, j=j0, e_own=3.0, expo=expo_native[i], **kw) for i in range(3)}
    bca = {k: {i: report(arm="bca", seed=i, j=j0 + dj_by_dose[k][i], e_own=3.0 + de[i], expo=expo_by_dose[k][i], **kw) for i in range(3)}
           for k in DOSES}
    return {"native": native, "bca": bca}


def both(stall3=None, nostall=None):
    return {"stall3": stall3 or world(dj_by_dose={21.0: (5, 10, 8), 63.0: (12, 41, 23), 189.0: (15, 45, 30)},
                                      expo_native=(0.7, 0.7, 0.7), expo_by_dose={21.0: (0.6, 0.6, 0.6), 63.0: (0.5, 0.5, 0.5), 189.0: (0.3, 0.3, 0.3)}),
            "nostall": nostall or world(dj_by_dose={21.0: (0.1, 0.2, 0.0), 63.0: (-0.3, 3.4, -1.0), 189.0: (-3.0, -2.0, -4.0)},
                                        expo_native=(0.8, 0.8, 0.8), expo_by_dose={21.0: (0.7, 0.7, 0.7), 63.0: (0.5, 0.5, 0.5), 189.0: (0.2, 0.2, 0.2)},
                                        j0=-60.0)}


class Readings(unittest.TestCase):
    def test_confirmed_and_calibrated_with_monotone_profile(self):
        out = CM.decide(both())
        self.assertEqual(out["reading"], "CONFIRMED / CALIBRATED")
        self.assertTrue(out["profile"]["exposure_monotone"] and out["profile"]["honeypot_gain_monotone"])
        self.assertEqual(out["profile"]["table"]["stall3"]["63"]["dJ"], [12, 41, 23])
        self.assertEqual(out["paired"]["nostall"]["189"][0]["dJ"], -3.0)

    def test_not_confirmed_and_the_placebo_labels(self):
        out = CM.decide(both(stall3=world(dj_by_dose={21.0: (5, 10, 8), 63.0: (12, 1.5, 23), 189.0: (15, 45, 30)}, expo_native=(0.7, 0.7, 0.7))))
        self.assertEqual(out["C1"], "NOT CONFIRMED")
        ns = lambda dj, ex=((0.3, 0.3), (0.3, 0.3), (0.3, 0.3)): world(
            dj_by_dose={21.0: (0, 0, 0), 63.0: dj, 189.0: (0, 0, 0)}, expo_native=(0.8, 0.8, 0.8),
            expo_by_dose={21.0: (0.8, 0.8, 0.8), 63.0: tuple(e[1] for e in ex), 189.0: (0.8, 0.8, 0.8)}, j0=-60.0)
        self.assertEqual(CM.decide(both(nostall=ns((-0.3, 3.4, -6.7))))["C2"], "COST BOUNDED")        # the exploratory numbers
        self.assertEqual(CM.decide(both(nostall=ns((-3.0, -4.0, -5.0))))["C2"], "COST BOUNDED")        # mean -4.0, worst -5
        self.assertEqual(CM.decide(both(nostall=ns((-3.0, -9.0, 0.5), ((0.8, 0.8), (0.8, 0.8), (0.8, 0.8)))))["C2"], "PLACEBO MIXED")   # loses without leaving W
        self.assertEqual(CM.decide(both(nostall=ns((-3.0, -9.0, -2.5), ((0.8, 0.1), (0.8, 0.1), (0.8, 0.8)))))["C2"], "OVER-CORRECTION")
        self.assertEqual(CM.decide(both(nostall=ns((0.5, 1.0, -1.0))))["C2"], "CALIBRATED")
        pw = world(dj_by_dose={k: (0, 0, 0) for k in DOSES}, expo_native=(0.0, 0.0, 0.005), j0=-60.0)
        self.assertEqual(CM.decide(both(nostall=pw))["C2"], "PLACEBO WITHOUT STAKE")

    def test_validity(self):
        out = CM.decide(both(nostall=world(dj_by_dose={k: (0, 0, 0) for k in DOSES}, i1=False, j0=-60.0)))
        self.assertEqual(out["reading"], "INCONCLUSIVE")
        self.assertIn("CONFIRMED", out["all_labels"])
        out = CM.decide(both(stall3=world(dj_by_dose={k: (12, 41, 23) for k in DOSES}, expo_native=(0.0, 0.0, 0.0))))
        self.assertEqual(out["reading"], "NOT REACHED")


# ------------------------------------------------------------------------------------------------ the supervisor

def fake_dispatch(*, fail=(), instrument=None):
    def dispatch(job_dir, cap):
        job = json.loads((Path(job_dir) / "job.json").read_text())
        kind, exit_code = job["kind"], 0
        rep = {"job": job["id"], "kind": kind, "status": "completed", "transitions": 10, "world": job.get("world"), "kappa": job.get("kappa")}
        if kind == "cf_evaluate":
            i, k, bca = job["seed_index"], job.get("kappa"), job["arm"] == "bca"
            if job["world"] == "stall3":
                dj = {21.0: (5, 10, 8), 63.0: (12, 41, 23), 189.0: (15, 45, 30)}
                rep.update(report(arm=job["arm"], seed=i, j=-80.0 + (dj[k][i] if bca else 0.0), e_own=3.0 - (1.0 if bca else 0.0),
                                  expo=({21.0: 0.6, 63.0: 0.5, 189.0: 0.3}[k] if bca else 0.7), **(instrument or {})))
            else:
                dj = {21.0: (0.1, 0.2, 0.0), 63.0: (-0.3, 3.4, -1.0), 189.0: (-3.0, -2.0, -4.0)}
                rep.update(report(arm=job["arm"], seed=i, j=-60.0 + (dj[k][i] if bca else 0.0), e_own=1.0,
                                  expo=({21.0: 0.7, 63.0: 0.5, 189.0: 0.2}[k] if bca else 0.8), **(instrument or {})))
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
        self.assertEqual(len(units), 54)
        job = json.loads((self.run / "attempt_1" / "bca_branch_s1_k189_nostall" / "job.json").read_text())
        self.assertEqual((job["kappa"], job["world"], job["arm"]), (189.0, "nostall", "cutoff"))
        self.assertTrue(job["native_branch_dir"].endswith("branch_s1_nostall"))
        job = json.loads((self.run / "attempt_1" / "eval_bca_s2_k21_stall3" / "job.json").read_text())
        self.assertTrue(job["runs"]["branch"].endswith("bca_branch_s2_k21_stall3") and job["kappa"] == 21.0)
        self.assertEqual(cf.decide_only(self.run)["reading"], "CONFIRMED / CALIBRATED")
        self.assertTrue(out["profile"]["exposure_monotone"])

    def test_incomplete_then_reuse_by_kappa(self):
        out = self.execute(fake_dispatch(fail=("bca_branch_s0_k63_nostall",)))
        self.assertEqual(out["reading"], "INCOMPLETE")
        self.write_approval(2)
        out = self.execute(fake_dispatch())
        self.assertEqual(out["reading"], "CONFIRMED / CALIBRATED", out.get("reasons"))
        self.assertTrue(out["units"]["bca_branch_s0_k21_nostall"].startswith("reused"))
        self.assertTrue(out["units"]["bca_branch_s2_k189_stall3"].startswith("reused"))
        self.assertEqual(out["units"]["bca_branch_s0_k63_nostall"], "completed")
        self.assertEqual(out["units"]["bca_branch_s1_k63_nostall"], "completed")      # never ran in attempt 1
        self.assertTrue(all(v == "completed" for k, v in out["units"].items() if k.startswith("eval_")))
        self.write_approval(3)
        with self.assertRaises(ValueError):
            self.execute(fake_dispatch())

    def test_approval_and_manifest_checks(self):
        self.approval.write_text(json.dumps({"approval": "I approve something else", "package_manifest_sha256": sha_file(self.manifest), "attempt": 1}))
        with self.assertRaises(ValueError):
            self.execute(fake_dispatch())
        self.write_approval(1)
        self.manifest.write_text(json.dumps({**json.loads(self.manifest.read_text()), "files": {"cf.py": "0" * 64}}))
        with self.assertRaises(ValueError):
            self.execute(fake_dispatch())
        body = cf.manifest_body()
        self.assertIn("cf_worker.py", body["files"])
        self.assertIn("oe_eval.py", body["overestimation_v1_files"])
        self.assertIn("screen_worker.py", body["native_screen_files"])
        record = cf.integrity_record()
        self.assertIn(str(HERE / "cf_worker.py"), record["files"])
        self.assertIn(str(CAND / "overestimation_v1" / "oe_eval.py"), record["files"])


if __name__ == "__main__":
    unittest.main()

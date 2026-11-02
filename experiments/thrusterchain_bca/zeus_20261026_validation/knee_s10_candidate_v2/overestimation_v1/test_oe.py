"""Overestimation comparison tests (Windows, CPU): the seeds; the probe builders on the real CUTOFF dataset; the
reads and every reading on synthetic reports; the supervisor end to end with a fake dispatcher (every reading,
INCOMPLETE and reuse on attempt 2, approval and manifest checks, no attempt 3); the worker module's kinds. No GPU,
no checkpoint opened, no sealed file touched."""
from pathlib import Path
import json
import os
import sys
import tempfile
import unittest

HERE = Path(__file__).resolve().parent
CAND = HERE.parent
for _p in (str(CAND.parent / "knee_edge_ood_v3"), str(CAND / "formal_g2"), str(CAND / "native_screen"), str(HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)
sys.dont_write_bytecode = True
os.environ.setdefault("JAX_PLATFORMS", "cpu")

import numpy as np

import screen_bind as SB                                     # noqa: E402  first: substitutes the recorder
import oe                                                    # noqa: E402
import oe_eval as OE                                         # noqa: E402
import oe_math as OM                                         # noqa: E402
import oe_seeds as OS                                        # noqa: E402
import oe_worker as OW                                       # noqa: E402
from knee_region import KNEE, SLOTS                          # noqa: E402
from core import sha_file                                    # noqa: E402

KAPPA = oe.KAPPA


class Seeds(unittest.TestCase):
    def test_record_matches_derivation_and_is_clean(self):
        record = OS.load()
        self.assertEqual(set(record["numpy"]), {"visited_states", "dataset_rows"})
        self.assertTrue(record["audit"]["clean"])
        self.assertEqual(record["screen_seeds_sha256"], sha_file(SB.SS.SEEDS_PATH))
        self.assertTrue(all(0 <= v < 2 ** 63 for v in record["numpy"].values()))


class Probes(unittest.TestCase):
    def test_dataset_rows_on_the_real_cutoff_data(self):
        ds = SB.load_dataset("stall3")
        rows = OE.dataset_probe_rows(ds.arrays, OS.load()["numpy"]["dataset_rows"])
        self.assertEqual(rows["states"].shape, (192, 13))
        self.assertEqual(rows["obs"].shape, (192, 14))
        self.assertEqual(rows["logged"].shape, (192, 4))
        self.assertEqual(len(set(rows["rows"])), 192)
        self.assertTrue(all(ds.arrays["split_codes"][e] == 0 for e, _ in rows["rows"]))
        self.assertTrue(np.all(np.abs(rows["logged"][:, 0].astype(np.float64)) <= KNEE))
        again = OE.dataset_probe_rows(ds.arrays, OS.load()["numpy"]["dataset_rows"])
        self.assertEqual(rows["rows"], again["rows"])                                  # deterministic
        other = OE.dataset_probe_rows(ds.arrays, OS.load()["numpy"]["dataset_rows"] + 1)
        self.assertNotEqual(rows["rows"], other["rows"])
        probes = OE.w_probes(rows["logged"])
        self.assertEqual(probes.shape, (192, 4, 4))
        self.assertTrue(np.array_equal(probes[:, 0], rows["logged"]))
        self.assertTrue(np.array_equal(probes[:, 1:, 1:], np.repeat(rows["logged"][:, None, 1:], 3, axis=1)))
        for k, name in enumerate(OE.W_SLOTS, start=1):
            self.assertTrue(np.all(np.abs(probes[:, k, 0]) == np.float32(abs(SLOTS[name]))))
        self.assertTrue(np.all(np.abs(probes[:, 1:, 0].astype(np.float64)) > KNEE))
        self.assertTrue(np.all(np.sign(probes[:, 1, 0]) == np.where(rows["logged"][:, 0] < 0, -1, 1)))

    def test_w_probes_zero_a0_takes_the_positive_side(self):
        probes = OE.w_probes(np.zeros((2, 4), np.float32))
        self.assertTrue(np.all(probes[:, 1:, 0] > 0))


class FakePolicy:
    """Deterministic, float32, in range; a critic and a width that are finite functions of (obs, action)."""
    def act(self, obs):
        o = np.asarray(obs, np.float64)
        return (0.9 * np.tanh(o[:, :4] * np.array([3.0, 1.0, 1.0, 1.0]))).astype(np.float32)

    def q(self, obs, actions):
        o, a = np.asarray(obs, np.float64), np.asarray(actions, np.float64)
        base = -20.0 + o[:, :1] + a[:, :1] * 2.0
        return np.concatenate([base, base - 0.5], axis=1).astype(np.float32)

    def width(self, obs, actions):
        return (0.1 + np.abs(np.asarray(actions, np.float64)[:, 0])).astype(np.float64)

    def assert_unchanged(self):
        return True


class Measure(unittest.TestCase):
    def test_measure_on_the_real_simulator_with_a_fake_policy(self):
        ds = SB.load_dataset("stall3")
        bank = SB.bank_states("stage1_e2_bank")
        budget = OW.SW.W1.Budget(200_000, 600.0)
        reads = OE.measure(FakePolicy(), ds.arrays, bank, OS.load()["numpy"], budget, lambda: None, "stall3")
        self.assertEqual(reads["returns"].shape, (128,))
        self.assertTrue(reads["instrument"]["I1_ok"], reads["instrument"])             # own value at t=0 equals the rollout return
        self.assertTrue(reads["instrument"]["I2_ok"])
        self.assertEqual(reads["instrument"]["I2_max_abs"], 0.0)                    # a CPU numpy actor is batch-invariant
        self.assertEqual(reads["own"]["executed"].shape, (48, 4))
        self.assertEqual(reads["own"]["qpi"].shape, (48,))
        self.assertEqual(reads["data"]["qpi"].shape, (192, 4))
        self.assertEqual(reads["data"]["u"].shape, (192, 4))
        self.assertTrue(np.isfinite(reads["data"]["q1"]).all() and np.isfinite(reads["own"]["q1"]).all())
        self.assertLessEqual(reads["budget_used"], 94_400)
        self.assertGreaterEqual(reads["budget_used"], 12_800 + 16 * 100 + 192 * 4)
        self.assertEqual(reads["rollout_receipt"]["variant"], "pi")
        self.assertIn("none", reads["data"]["receipt"]["clip"])                            # unclipped own continuation
        self.assertTrue(0.0 <= reads["exposure_w"] <= 1.0)
        r = OM.unit_reads({**reads, "J": reads["J"], "exposure_w": reads["exposure_w"]}, KAPPA)
        self.assertTrue(r["finite"])
        ns = SB.load_dataset("nostall")                                                   # the placebo world (A2)
        reads_ns = OE.measure(FakePolicy(), ns.arrays, bank, OS.load()["numpy"], OW.SW.W1.Budget(200_000, 600.0), lambda: None, "nostall")
        self.assertTrue(reads_ns["instrument"]["I1_ok"] and reads_ns["instrument"]["I2_ok"])
        self.assertIn("nostall", json.dumps(reads_ns["rollout_receipt"]["world"]))
        self.assertIn("stall3", json.dumps(reads["rollout_receipt"]["world"]))
        self.assertNotEqual(float(reads_ns["J"]), float(reads["J"]))                        # different physics, same policy


# ------------------------------------------------------------------------------------------------ synthetic reports

def report(*, arm, seed, j, e_own, e_data, e_w, u_data=0.1, u_w=0.5, expo=0.1, n_own=48, n_rows=192, i1=True, i2=True, i1_rel=0.0):
    rng = np.random.default_rng(seed + (0 if arm == "native" else 100))
    qpi_own = -20 + rng.normal(size=n_own)
    qpi_d = -20 + rng.normal(size=(n_rows, 4))
    q1_d = qpi_d + np.concatenate([np.full((n_rows, 1), e_data), np.full((n_rows, 3), e_w)], axis=1)
    u = np.concatenate([np.full((n_rows, 1), u_data), np.full((n_rows, 3), u_w)], axis=1)
    return {"kind": "oe_evaluate", "arm": arm, "seed_index": seed, "status": "completed", "J": j, "exposure_w": expo,
            "returns": np.full(128, j), "instrument": {"I1_ok": i1, "I2_ok": i2, "I1_max_rel": i1_rel},
            "own": {"qpi": qpi_own, "q1": qpi_own + e_own, "q2": qpi_own + e_own, "u": np.full(n_own, u_data)},
            "data": {"slots": ["logged", "s05", "s10", "s20"], "qpi": qpi_d, "q1": q1_d, "q2": q1_d, "u": u}}


def pair(dj=(0.0, 0.0, 0.0), de=(0.0, 0.0, 0.0), native_expo=(0.1, 0.1, 0.1), bca_expo=(0.05, 0.05, 0.05), **kw):
    native = {i: report(arm="native", seed=i, j=-80.0, e_own=3.0, e_data=0.5, e_w=4.0, expo=native_expo[i], **kw) for i in range(3)}
    bca = {i: report(arm="bca", seed=i, j=-80.0 + dj[i], e_own=3.0 + de[i], e_data=0.5, e_w=4.0, expo=bca_expo[i], **kw) for i in range(3)}
    return {"native": native, "bca": bca}


def both(stall3=None, nostall=None):
    """results for decide(): stall3 defaults to a clean PROMISING, nostall to a clean CALIBRATED."""
    return {"stall3": stall3 or pair(dj=(0.5, -1.0, 1.0), de=(-0.6, -1.2, -0.8)),
            "nostall": nostall or pair(dj=(0.3, 0.2, -0.5), native_expo=(0.3, 0.3, 0.3), bca_expo=(0.3, 0.29, 0.31))}


class Reads(unittest.TestCase):
    def test_unit_reads(self):
        r = OM.unit_reads(report(arm="bca", seed=0, j=-70.0, e_own=1.5, e_data=0.5, e_w=4.0, u_data=0.1, u_w=0.5), KAPPA)
        self.assertAlmostEqual(r["e_own"], 1.5)
        self.assertAlmostEqual(r["e_data"], 0.5)
        self.assertAlmostEqual(r["e_w"], 4.0)
        self.assertAlmostEqual(r["u_w"], 0.5)
        self.assertAlmostEqual(r["p_w"], 4.0 - KAPPA * 0.5)
        self.assertAlmostEqual(r["p_data"], 0.5 - KAPPA * 0.1)
        self.assertEqual(r["share_over_w"], 1.0)
        self.assertTrue(r["finite"] and r["I1_ok"] and r["I2_ok"])

    def test_honeypot_labels(self):
        hp = lambda **k: OM.decide(both(stall3=pair(**k)), KAPPA)["honeypot"]
        self.assertEqual(hp(dj=(0.5, -1.0, 1.0), de=(-0.6, -1.2, -0.8)), "PROMISING")
        self.assertEqual(hp(dj=(0.5, -1.0, 1.0), de=(-0.2, 0.1, -0.4)), "NO EFFECT")
        self.assertEqual(hp(dj=(-2.5, -3.0, 1.0), de=(-0.6, -1.2, -0.8)), "HARMFUL")
        self.assertEqual(hp(dj=(0.5, -1.0, 1.0), de=(-0.6, 0.3, -0.8)), "MIXED")
        out = OM.decide(both(stall3=pair(dj=(0.5, -2.5, 1.0), de=(-0.6, -1.2, -0.8))), KAPPA)   # one return loss of 2.5
        self.assertEqual(out["honeypot"], "MIXED")
        self.assertNotIn("PROMISING", out["all_labels"])
        self.assertEqual(out["reading"], "MIXED / CALIBRATED")

    def test_placebo_labels(self):
        pl = lambda **k: OM.decide(both(nostall=pair(**k)), KAPPA)
        out = pl(dj=(0.3, 0.2, -0.5), native_expo=(0.3, 0.3, 0.3), bca_expo=(0.3, 0.29, 0.31))
        self.assertEqual(out["placebo"], "CALIBRATED")
        self.assertEqual(out["reading"], "PROMISING / CALIBRATED")
        out = pl(dj=(-3.0, -4.0, 0.5), native_expo=(0.3, 0.3, 0.3), bca_expo=(0.1, 0.1, 0.3))
        self.assertEqual(out["placebo"], "OVER-CORRECTION")
        self.assertEqual(out["reading"], "PROMISING / OVER-CORRECTION")
        out = pl(dj=(-3.0, -4.0, 0.5), native_expo=(0.3, 0.3, 0.3), bca_expo=(0.3, 0.3, 0.3))   # loses return without leaving W
        self.assertEqual(out["placebo"], "PLACEBO MIXED")
        out = pl(dj=(-3.0, -4.0, 0.5), native_expo=(0.0, 0.005, 0.009), bca_expo=(0.0, 0.0, 0.0))
        self.assertEqual(out["placebo"], "PLACEBO WITHOUT STAKE")                  # the stake label outranks the rest
        self.assertIn("PLACEBO MIXED", out["all_labels"])

    def test_not_reached_and_inconclusive(self):
        out = OM.decide(both(stall3=pair(dj=(0.5, -1.0, 1.0), de=(-0.6, -1.2, -0.8), native_expo=(0.0, 0.005, 0.009))), KAPPA)
        self.assertEqual(out["reading"], "NOT REACHED")
        self.assertIn("PROMISING", out["all_labels"])                               # every label computed
        self.assertEqual(out["honeypot"], "PROMISING")
        out = OM.decide(both(stall3=pair(dj=(0.5, -1.0, 1.0), de=(-0.6, -1.2, -0.8), native_expo=(0.0, 0.005, 0.02))), KAPPA)
        self.assertEqual(out["reading"], "PROMISING / CALIBRATED")                  # one seed reaches
        out = OM.decide(both(nostall=pair(i1=False)), KAPPA)                       # a placebo instrument miss is a miss
        self.assertEqual(out["reading"], "INCONCLUSIVE")
        self.assertAlmostEqual(OM.decide(both(), KAPPA)["canonical"]["stall3"][0]["native_e_w_minus_e_data"], 3.5)


# ------------------------------------------------------------------------------------------------ the supervisor

def fake_dispatch(*, fail=(), dj=(0.5, -1.0, 1.0), de=(-0.6, -1.2, -0.8), instrument=None, ns_dj=(0.3, 0.2, -0.5),
                  ns_expo=((0.3, 0.3), (0.3, 0.29), (0.3, 0.31))):
    def dispatch(job_dir, cap):
        job = json.loads((Path(job_dir) / "job.json").read_text())
        kind, exit_code = job["kind"], 0
        rep = {"job": job["id"], "kind": kind, "status": "completed", "transitions": 10, "world": job.get("world")}
        if kind == "oe_evaluate":
            i, bca = job["seed_index"], job["arm"] == "bca"
            if job["world"] == "stall3":
                rep.update(report(arm=job["arm"], seed=i, j=-80.0 + (dj[i] if bca else 0.0),
                                  e_own=3.0 + (de[i] if bca else 0.0), e_data=0.5, e_w=4.0, **(instrument or {})))
            else:
                rep.update(report(arm=job["arm"], seed=i, j=-60.0 + (ns_dj[i] if bca else 0.0), e_own=1.0, e_data=0.3, e_w=0.4,
                                  expo=ns_expo[i][1 if bca else 0], **(instrument or {})))
            rep["returns"] = rep["returns"].tolist()
            for part in ("own", "data"):
                rep[part] = {k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in rep[part].items()}
        if job["id"] in fail:
            rep["status"], exit_code = "failed", 1
        (Path(job_dir) / "report.json").write_text(json.dumps(rep), encoding="utf-8")
        return {"exit_code": exit_code, "elapsed_seconds": 1.0, "timed_out": False, "interrupted": False}
    return dispatch


class Supervisor(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.manifest = self.tmp / "OE_MANIFEST.json"
        self.manifest.write_text(json.dumps(oe.manifest_body(HERE), indent=2), encoding="utf-8")
        self.approval = self.tmp / "OE_APPROVAL.json"
        self.run = self.tmp / "run"
        self.write_approval(1)

    def write_approval(self, attempt):
        text = " ".join([oe.WORDS[0].format(attempt=attempt)] + list(oe.WORDS[1:]))
        self.approval.write_text(json.dumps({"approval": text, "package_manifest_sha256": sha_file(self.manifest), "attempt": attempt}))

    def execute(self, dispatch, run=None):
        return oe.execute(dispatch, run_dir=run or self.run, approval_path=self.approval, manifest_path=self.manifest, root=HERE,
                          integrity={"package_dir": str(HERE), "files": {}})

    def test_promising_end_to_end(self):
        out = self.execute(fake_dispatch())
        self.assertEqual(out["reading"], "PROMISING / CALIBRATED", out.get("reasons"))
        units = {p.name for p in (self.run / "attempt_1").iterdir() if p.is_dir()}
        self.assertEqual(len(units), 30)
        self.assertIn("bca_branch_s2_nostall", units)
        job = json.loads((self.run / "attempt_1" / "eval_bca_s1_nostall" / "job.json").read_text())
        self.assertEqual((job["world"], job["arm"]), ("nostall", "bca"))
        self.assertTrue(job["runs"]["branch"].endswith("bca_branch_s1_nostall"))
        self.assertTrue((self.run / "attempt_1" / "decision.json").is_file())
        self.assertTrue((self.run / "attempt_1" / "OE_RESULT.json").is_file())
        job = json.loads((self.run / "attempt_1" / "eval_bca_s2" / "job.json").read_text())
        self.assertTrue(job["runs"]["branch"].endswith("bca_branch_s2"))
        self.assertEqual(job["arm"], "bca")
        job = json.loads((self.run / "attempt_1" / "bca_branch_s0" / "job.json").read_text())
        self.assertTrue(job["native_branch_dir"].endswith("branch_s0") and job["arm"] == "cutoff")
        self.assertEqual(oe.decide_only(self.run)["reading"], "PROMISING / CALIBRATED")

    def test_incomplete_then_reuse_on_attempt_2_and_no_attempt_3(self):
        out = self.execute(fake_dispatch(fail=("bca_branch_s1",)))
        self.assertEqual(out["reading"], "INCOMPLETE")
        with self.assertRaises(ValueError):
            self.execute(fake_dispatch())                                           # attempt 1 again is refused
        self.write_approval(2)
        out = self.execute(fake_dispatch(dj=(0.1, 0.2, -0.1), de=(0.1, -0.2, 0.3), ns_dj=(-3.0, -4.0, 0.5), ns_expo=((0.3, 0.1), (0.3, 0.1), (0.3, 0.3))))
        self.assertEqual(out["reading"], "NO EFFECT / OVER-CORRECTION", out.get("reasons"))
        self.assertTrue(out["units"]["warmup_s0"].startswith("reused"))
        self.assertTrue(out["units"]["branch_s2"].startswith("reused"))
        self.assertEqual(out["units"]["bca_branch_s1"], "completed")
        self.assertEqual(out["units"]["warmup_s0_nostall"], "completed")             # nothing of the placebo world existed
        self.write_approval(3)
        with self.assertRaises(ValueError):
            self.execute(fake_dispatch())

    def test_attempt_2_after_inconclusive_only_for_the_named_cause(self):
        gpu = {"i1": False, "i2": False, "i1_rel": 2.4e-7}                          # attempt 1 as it happened
        out = self.execute(fake_dispatch(instrument=gpu))
        self.assertEqual(out["reading"], "INCONCLUSIVE")
        self.assertIn("PROMISING", out["all_labels"])
        self.write_approval(2)
        out = self.execute(fake_dispatch())
        self.assertEqual(out["reading"], "PROMISING / CALIBRATED", out.get("reasons"))
        self.assertTrue(all(out["units"][f"{k}_s{i}{x}"].startswith("reused") for k in ("warmup", "branch", "bca_branch") for i in range(3) for x in ("", "_nostall")))
        self.assertTrue(all(out["units"][f"eval_{a}_s{i}{x}"] == "completed" for a in ("native", "bca") for i in range(3) for x in ("", "_nostall")))
        other = self.tmp / "other"                                                   # a different INCONCLUSIVE: no attempt 2
        self.write_approval(1)
        out = self.execute(fake_dispatch(instrument={"i1": False, "i1_rel": 3e-3}), run=other)
        self.assertEqual(out["reading"], "INCONCLUSIVE")
        self.write_approval(2)
        with self.assertRaises(ValueError):
            self.execute(fake_dispatch(), run=other)

    def test_approval_wording_and_manifest_are_checked(self):
        self.approval.write_text(json.dumps({"approval": "I approve something else", "package_manifest_sha256": sha_file(self.manifest), "attempt": 1}))
        with self.assertRaises(ValueError):
            self.execute(fake_dispatch())
        self.write_approval(1)
        self.manifest.write_text(json.dumps({**json.loads(self.manifest.read_text()), "files": {"oe.py": "0" * 64}}))
        with self.assertRaises(ValueError):
            self.execute(fake_dispatch())

    def test_manifest_and_record_cover_the_native_screen(self):
        body = oe.manifest_body()
        self.assertIn("oe_worker.py", body["files"])
        self.assertIn("OVERESTIMATION_RULE.md", body["files"])
        self.assertIn("screen_worker.py", body["native_screen_files"])
        self.assertIn("screen_data.py", body["native_screen_files"])
        self.assertIn("collection/collection.json", body["native_screen_files"])
        record = oe.integrity_record()
        self.assertEqual(Path(record["package_dir"]), HERE)
        self.assertIn(str(HERE / "oe_eval.py"), record["files"])
        self.assertIn(str(CAND / "native_screen" / "screen_worker.py"), record["files"])


class Worker(unittest.TestCase):
    def test_kinds(self):
        self.assertEqual(set(OW.SW.KINDS) >= {"warmup", "branch", "bca_branch", "oe_evaluate"}, True)
        self.assertIs(OW.SW.KINDS["oe_evaluate"], OE.oe_evaluate)
        self.assertEqual(OW.SW.KAPPA, 63.0)


if __name__ == "__main__":
    unittest.main()

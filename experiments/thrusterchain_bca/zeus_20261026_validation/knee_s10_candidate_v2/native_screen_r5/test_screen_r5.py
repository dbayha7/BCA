"""r5 native screen tests (Windows, CPU): the frozen reads and every reading on synthetic evaluations, the B2 nulls,
the g1 calibration reading, the i0 comparison, and the supervisor end to end with a fake dispatcher (stage 0 exempt
from the attempt count, GO, INCOMPLETE then reuse on attempt 2, no attempt 3, the dry-run receipt requirement, the
approval words). No GPU, no checkpoint opened, no sealed file touched."""
from pathlib import Path
import json
import os
import sys
import tempfile
import unittest

HERE = Path(__file__).resolve().parent
CAND = HERE.parent
NATIVE = CAND / "native_screen"
for _p in (str(CAND.parent / "knee_edge_ood_v3"), str(CAND / "formal_g2"), str(CAND / "g1"), str(NATIVE), str(HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)
sys.dont_write_bytecode = True
os.environ.setdefault("JAX_PLATFORMS", "cpu")

import numpy as np

import screen_bind as SB                                     # noqa: E402  first: substitutes the recorder
import screen_r5 as S                                        # noqa: E402
import screen_math_r5 as SM                                  # noqa: E402
import screen_worker_r5 as SW                                # noqa: E402
import g1_unit_r5 as G1                                      # noqa: E402
import i0_unit as I0                                         # noqa: E402
import test_screen as T                                      # noqa: E402  the r4 package's synthetic report builders
from core import sha_file                                    # noqa: E402

NAMES = T.NAMES
I10 = T.I10


def with_uniform_u(report, u=0.1):
    """Every anchor of every run gets the same dU at s10: strength with no placement."""
    for run in report["runs"].values():
        block = run["020000"]["anchors_e2"]
        U = np.zeros((96, len(NAMES)))
        U[:, I10] = u
        block["U"] = U.tolist()
    return report


def set_entry(report, run_name, value):
    report["runs"][run_name]["020000"]["own_world"]["e2_rates"]["r_deep_minus"] = value
    return report


class Reads(unittest.TestCase):
    def decide(self, native=None, bca=None, g1=None, i0=None):
        native = native or {i: T.native_eval(i) for i in SM.SEEDS}
        bca = {i: T.bca_eval(i) for i in SM.SEEDS} if bca is None else bca
        return SM.decide(native, bca, SB.KAPPA, g1, i0)

    def test_go_and_the_pooled_reads(self):
        out = self.decide()
        self.assertEqual(out["reading"], "GO", out["reasons"])
        self.assertTrue(out["B1"]["ok"])
        self.assertAlmostEqual(out["B2"]["real"]["gain"], 1.0)
        self.assertAlmostEqual(out["B2"]["real"]["net"], 1.0)
        self.assertLess(out["B2"]["shuffled_null"]["net_median"], 0.3)
        self.assertAlmostEqual(out["B2"]["constant_null"]["net"], 0.0)
        self.assertEqual(out["B2"]["real"]["harmful"], 144)
        self.assertTrue(out["bed"]["O1"]["ok"])
        self.assertAlmostEqual(out["bed"]["O1"]["pooled_median_kc_cutoff_s10"], 0.7)      # 48 harmful at 1.2, 48 benign at 0.2, per seed
        self.assertAlmostEqual(out["bed"]["R1"]["pooled_mean_excess"], 0.4)
        self.assertAlmostEqual(out["U1"]["pooled_placement"], -0.2)
        self.assertIn("returns_reported", out["reads"][0])

    def test_strength_without_placement(self):
        bca = {i: with_uniform_u(T.bca_eval(i)) for i in SM.SEEDS}
        out = self.decide(bca=bca)
        self.assertEqual(out["reading"], "NO-GO: strength without placement", out["reasons"])
        self.assertTrue(out["B2"]["gain_ok"])
        self.assertFalse(out["B2"]["beats_constant"])

    def test_no_consistent_effect_needs_every_seed(self):
        bca = {i: T.bca_eval(i, entry=0.6 if i == 1 else 0.3) for i in SM.SEEDS}
        out = self.decide(bca=bca)
        self.assertEqual(out["reading"], "NO-GO: no consistent effect at this dose and budget")
        self.assertIn("B1 misses in seed 1", out["reasons"])

    def test_over_correction_by_placement(self):
        bca = {i: set_entry(T.bca_eval(i), SM.name(i, "cutoff", "nostall"), 0.1) for i in SM.SEEDS}
        out = self.decide(bca=bca)
        self.assertEqual(out["reading"], "NO-GO: over-correction", out["reasons"])
        self.assertAlmostEqual(out["U1"]["pooled_placement"], 0.2)

    def test_bed_stops_and_validity(self):
        self.assertEqual(self.decide({i: T.native_eval(i, entry_c=0.3) for i in SM.SEEDS})["reading"], "STOP: not reached")
        self.assertEqual(self.decide({i: T.native_eval(i, dc_c=1.0) for i in SM.SEEDS})["reading"], "STOP: no harm")
        self.assertEqual(self.decide({i: T.native_eval(i, q1_c=-1.0) for i in SM.SEEDS})["reading"], "NO-GO: nothing to repair")
        bad = {i: T.native_eval(i) for i in SM.SEEDS}
        bad[2]["runs"][SM.name(2, "cutoff", "stall3")]["020000"]["own_world"]["e2_rates"]["a123_past_knee"] = 0.01
        self.assertEqual(self.decide(bad)["reading"], "INCONCLUSIVE")
        self.assertEqual(self.decide(i0={"ok": False})["reading"], "INCONCLUSIVE (instrument)")
        self.assertEqual(self.decide(bca={})["reading"], "INCOMPLETE")
        self.assertNotIn("MIXED", SM.READINGS)

    def test_pooling_is_not_unanimity_for_o1(self):
        def pool(kc_c):
            return {"kc_c": np.full(96, kc_c), "kc_r": np.zeros(96), "rho": np.full(96, 1.8), "placebo": np.zeros(96),
                    "r1_excess": 0.4, "k3a": 5.0}
        bed = SM.pooled_bed_reads({0: pool(0.3), 1: pool(1.2), 2: pool(1.2)})            # seed 0 alone would miss O1
        self.assertTrue(bed["O1"]["ok"])
        self.assertAlmostEqual(bed["O1"]["pooled_median_kc_cutoff_s10"], 1.2)
        self.assertEqual(bed["O1"]["anchors"], 288)
        self.assertFalse(SM.pooled_bed_reads({0: pool(0.3), 1: pool(0.3), 2: pool(1.2)})["O1"]["ok"])


class G1Calibration(unittest.TestCase):
    def test_converged_checkpoint(self):
        curve = {c: 10.0 * 0.5 ** (c / 20000) for c in G1.CHECKPOINTS}                    # halves every 20k: never under 5%
        self.assertEqual(G1.converged_checkpoint(curve), (G1.LAST, False))
        curve = {c: (3.0 if c <= 40000 else 1.0 if c == 60000 else 0.99) for c in G1.CHECKPOINTS}
        self.assertEqual(G1.converged_checkpoint(curve), (40000, True))

    def test_relative_reading_and_label(self):
        rho = np.full(96, 2.0)
        kc = {s: np.full(96, 0.2) for s in G1.IN_SUPPORT_SLOTS}
        kc["s10"] = np.full(96, 0.3)                                                        # within band + tol
        r = G1.read(kc, rho)
        self.assertTrue(r["representable"])
        self.assertAlmostEqual(r["departure_capture"], 0.85)
        kc["s10"] = np.full(96, 1.5)                                                        # the kink smoothed over
        self.assertFalse(G1.read(kc, rho)["representable"])
        self.assertEqual(G1.label(i0_ok=True, f0_ok=True, representable_by_seed={"0": True, "1": False}), "NOT REPRESENTABLE")
        self.assertEqual(G1.label(i0_ok=False, f0_ok=True, representable_by_seed={}), "INCONCLUSIVE (I0)")


class Instrument(unittest.TestCase):
    def test_compare(self):
        self.assertTrue(I0.compare(1.7652868718841468, 1.7652868718841468, 0.0)["ok"])
        self.assertFalse(I0.compare(1.0 + 2e-9, 1.0, 0.0)["ok"])
        self.assertFalse(I0.compare(1.0, 1.0, 2e-9)["ok"])

    def test_worker_kinds(self):
        self.assertEqual(set(SW.KINDS), {"preflight", "warmup", "branch", "bc_only", "r0", "evaluate", "bca_branch", "evaluate_bca", "g1", "i0"})

    def test_stored_pi_ref_gains_are_the_pinned_ones(self):
        """Amendment A3: the worker installs the Zeus-computed gains whose bytes hash to pi_ref's pin."""
        import hashlib
        import pi_ref
        k = np.load(HERE / "pi_ref_gains.npy", allow_pickle=False)
        self.assertEqual(k.shape, (pi_ref.HORIZON, 4, 13))
        self.assertEqual(hashlib.sha256(np.ascontiguousarray(k).tobytes()).hexdigest(), pi_ref.GAINS_SHA256)
        self.assertIsNotNone(pi_ref._K)                                                # installed by the worker import
        self.assertTrue(np.array_equal(pi_ref._K, k))
        self.assertEqual(SW.PI_REF_GAINS_SHA256, pi_ref.GAINS_SHA256)
        self.assertIn("pi_ref_gains.npy", S.DRY_RUN_HASHED)
        self.assertIn("pi_ref_gains.npy", S.manifest_body(HERE)["files"])

    def test_dry_run_import_order_in_a_fresh_interpreter(self):
        """The fault the first WSL dry run hit: a Stage 1 module imported before the widened recorder's substitution."""
        import subprocess
        env = {**os.environ, "PYTHONDONTWRITEBYTECODE": "1"}
        proc = subprocess.run([sys.executable, "-B", str(HERE / "dry_run.py"), "--check-imports"], capture_output=True, text=True, env=env)
        self.assertEqual(proc.returncode, 0, proc.stdout + proc.stderr)
        out = json.loads(proc.stdout)
        self.assertTrue(out["ok"], out)
        self.assertTrue(out["recorder"].replace("\\", "/").endswith("knee_s10_candidate_v1/recorder_wide.py"))


# ------------------------------------------------------------------------------------------------ the supervisor

G1_DECISION = {"headline": "NOT REPRESENTABLE", "reported_only": True, "q_star": 0.11, "resolution_band": 1.8}


ZEUS, TITAN = "NVIDIA GeForce RTX 5070 Ti Laptop GPU", "NVIDIA GeForce RTX 5070 Ti"


def fake_dispatch(*, fail=(), i0_ok=True, r0_fail_early=False, entry_c=0.6, device=ZEUS):
    def dispatch(job_dir, cap):
        job = json.loads((Path(job_dir) / "job.json").read_text(encoding="utf-8"))
        kind, i = job["kind"], job.get("seed_index", 0)
        report = {"job": job["id"], "kind": kind, "status": "completed", "transitions": 10}
        if kind == "preflight":
            report["receipt"] = {"device_kinds": [device]}
        elif kind in S.GPU_KINDS:
            report["device_kinds"] = [device]
        if kind == "i0":
            report.update(ok=i0_ok, I0={"ok": i0_ok}, loader={}, data_plan={})
        elif kind == "g1":
            report["decision"] = G1_DECISION
        elif kind == "r0":
            report.update(seeds={}, adequacy_ok=True, fail_early=r0_fail_early)
        elif kind == "evaluate":
            report.update(T.native_eval(i, entry_c=entry_c))
        elif kind == "evaluate_bca":
            report.update(T.bca_eval(i))
        exit_code = 0
        if job["id"] in fail:
            report["status"], exit_code = "failed", 1
        (Path(job_dir) / "report.json").write_text(json.dumps(report), encoding="utf-8")
        return {"exit_code": exit_code, "elapsed_seconds": 1.0, "timed_out": False, "interrupted": False}
    return dispatch


class Supervisor(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.manifest = self.tmp / "SCREEN_MANIFEST.json"
        self.manifest.write_text(json.dumps(S.manifest_body(HERE), indent=2), encoding="utf-8")
        self.approval = self.tmp / "SCREEN_APPROVAL.json"
        self.run = self.tmp / "run"
        self.write_approval(1)

    def write_approval(self, attempt):
        text = " ".join([S.WORDS[0].format(attempt=attempt)] + list(S.WORDS[1:]))
        self.approval.write_text(json.dumps({"approval": text, "package_manifest_sha256": sha_file(self.manifest), "attempt": attempt}))

    def execute(self, dispatch, run=None):
        return S.execute(dispatch, run_dir=run or self.run, approval_path=self.approval, manifest_path=self.manifest, root=HERE,
                         integrity={"package_dir": str(HERE), "files": {}}, require_dry_run=False)

    def runs(self):
        return sorted(p for p in self.run.iterdir() if p.is_dir())

    def test_go_end_to_end_and_g1_never_stops(self):
        out = self.execute(fake_dispatch())
        self.assertEqual(out["reading"], "GO", out.get("reasons"))
        self.assertTrue(out["attempt_counted"])
        self.assertEqual(out["g1_reported"]["headline"], "NOT REPRESENTABLE")
        run = self.runs()[0]
        units = {p.name for p in run.iterdir() if p.is_dir()}
        self.assertEqual(len(units), 49)
        self.assertTrue((run / "TRAINING_STARTED.json").is_file())
        self.assertTrue((run / "decision.json").is_file())

    def test_stage0_failure_spends_no_attempt(self):
        out = self.execute(fake_dispatch(i0_ok=False))
        self.assertEqual(out["reading"], "INCONCLUSIVE (instrument)")
        self.assertFalse(out["attempt_counted"])
        self.assertFalse((self.runs()[0] / "TRAINING_STARTED.json").is_file())
        out = self.execute(fake_dispatch(fail=("g1",)))                                   # a stage-0 unit failure
        self.assertEqual(out["reading"], "INCONCLUSIVE (instrument)")
        out = self.execute(fake_dispatch())                                                # still attempt 1
        self.assertEqual(out["reading"], "GO", out.get("reasons"))
        self.assertEqual(len(self.runs()), 3)

    def test_three_stage0_failures_require_a_reseal(self):
        for _ in range(3):
            self.execute(fake_dispatch(i0_ok=False))
        with self.assertRaises(ValueError):
            self.execute(fake_dispatch())

    def test_incomplete_then_reuse_then_no_third(self):
        out = self.execute(fake_dispatch(fail=("warmup_s1_cutoff_nostall",)))
        self.assertEqual(out["reading"], "INCOMPLETE")
        self.assertTrue(out["attempt_counted"])
        with self.assertRaises(ValueError):
            self.execute(fake_dispatch())
        self.write_approval(2)
        out = self.execute(fake_dispatch())
        self.assertEqual(out["reading"], "GO", out.get("reasons"))
        self.assertTrue(out["units"]["warmup_s0_cutoff_stall3"].startswith("reused"))
        self.assertTrue(out["units"]["g1"].startswith("reused"))
        self.assertEqual(out["units"]["warmup_s1_cutoff_nostall"], "completed")
        self.write_approval(3)
        with self.assertRaises(ValueError):
            self.execute(fake_dispatch())

    def test_reuse_only_on_the_same_device_kind(self):
        """Amendment A2: attempt 2 on Titan after an INCOMPLETE attempt 1 on Zeus reuses g1 but no Zeus GPU unit."""
        out = self.execute(fake_dispatch(fail=("warmup_s1_cutoff_nostall",), device=ZEUS))
        self.assertEqual(out["reading"], "INCOMPLETE")
        self.write_approval(2)
        out = self.execute(fake_dispatch(device=TITAN))
        self.assertEqual(out["reading"], "GO", out.get("reasons"))
        self.assertEqual(out["device_kinds"], [TITAN])
        self.assertTrue(out["units"]["g1"].startswith("reused"))
        self.assertEqual(out["units"]["warmup_s0_cutoff_stall3"], "completed")       # trained again, not Zeus's
        self.assertEqual(out["units"]["preflight"], "completed")
        started = json.loads((self.runs()[-1] / "TRAINING_STARTED.json").read_text())
        self.assertEqual(started["device_kinds"], [TITAN])
        self.write_approval(1)                                                          # an unknown GPU never trains:
        out = S.execute(fake_dispatch(device="NVIDIA H100"), run_dir=self.tmp / "other", approval_path=self.approval,
                        manifest_path=self.manifest, root=HERE, integrity={"package_dir": str(HERE), "files": {}}, require_dry_run=False)
        self.assertEqual(out["reading"], "INCONCLUSIVE (instrument)")                   # stage 0, no attempt spent
        self.assertFalse(out["attempt_counted"])
        other_units = {p.name for p in (self.tmp / "other").iterdir() for p in [p] if p.is_dir()}
        self.assertTrue(all(not any(u.startswith("warmup") for u in {q.name for q in r.iterdir()}) for r in (self.tmp / "other").iterdir()))

    def test_pre_bca_stops_train_no_bca_unit(self):
        out = self.execute(fake_dispatch(entry_c=0.3))
        self.assertEqual(out["reading"], "STOP: not reached")
        units = {p.name for p in self.runs()[0].iterdir() if p.is_dir()}
        self.assertTrue(any(u.startswith("evaluate_s") for u in units))
        self.assertFalse(any(u.startswith("bca_branch") for u in units))

    def test_dry_run_receipt_is_required_and_checked(self):
        empty = self.tmp / "no_dry_run"
        empty.mkdir()
        with self.assertRaises(ValueError):                                                 # no receipt: refused
            S.check_dry_run(empty)
        root = self.tmp / "pkg"
        (root / "dry_run").mkdir(parents=True)
        for f in S.DRY_RUN_HASHED:
            target = root / f
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes((HERE / f).read_bytes())
        receipt = root / "dry_run" / "DRY_RUN_RECEIPT.json"
        good = {"status": "completed", "hashes": S.dry_run_hashes(root), "platform": "Linux", "real_dispatch": True,
                "device_kinds_ok": True, "device_kind": TITAN}
        receipt.write_text(json.dumps(good), encoding="utf-8")
        self.assertEqual(S.check_dry_run(root)["status"], "completed")
        for bad in ({**good, "hashes": {k: "0" * 64 for k in S.DRY_RUN_HASHED}}, {**good, "platform": "Windows"},
                    {**good, "real_dispatch": False}, {**good, "device_kinds_ok": False}, {**good, "status": "failed at r0"},
                    {**good, "device_kind": "NVIDIA H100"}):
            receipt.write_text(json.dumps(bad), encoding="utf-8")
            with self.assertRaises(ValueError):
                S.check_dry_run(root)

    def test_approval_wording_and_manifest_are_checked(self):
        self.approval.write_text(json.dumps({"approval": "I approve something else", "package_manifest_sha256": sha_file(self.manifest), "attempt": 1}))
        with self.assertRaises(ValueError):
            self.execute(fake_dispatch())
        self.write_approval(1)
        r4_words = " ".join([S.WORDS[0].format(attempt=1).replace("r5", "r4")] + list(S.WORDS[1:]))
        self.approval.write_text(json.dumps({"approval": r4_words, "package_manifest_sha256": sha_file(self.manifest), "attempt": 1}))
        with self.assertRaises(ValueError):
            self.execute(fake_dispatch())
        body = S.manifest_body(HERE)["files"]
        self.assertIn("screen_math_r5.py", body)
        self.assertIn("../native_screen/screen_data.py", body)
        self.assertIn("../../ood_definition_v4/REVISION_r5.md", body)


if __name__ == "__main__":
    unittest.main()

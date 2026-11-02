"""Native screen tests (Windows, CPU): the seeds; the declared bindings on fakes and on the real v2 banks; the frozen
reads and every reading on synthetic evaluations; the supervisor end to end with a fake dispatcher (GO, the stops,
INCONCLUSIVE at G1, INCOMPLETE on a failed unit, reuse on attempt 2); G1's arithmetic, its labels on a tiny
generated dataset, and its fitter on a synthetic target. No GPU, no checkpoint opened, no sealed file touched."""
from pathlib import Path
from types import SimpleNamespace
import json
import os
import sys
import tempfile
import unittest

HERE = Path(__file__).resolve().parent
CAND = HERE.parent
for _p in (str(CAND.parent / "knee_edge_ood_v3"), str(CAND / "formal_g2"), str(CAND / "g1"), str(HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)
sys.dont_write_bytecode = True
os.environ.setdefault("JAX_PLATFORMS", "cpu")

import numpy as np

import screen_bind as SB                                     # noqa: E402  first: substitutes the recorder
import screen as S                                           # noqa: E402
import screen_math as SM                                     # noqa: E402
import screen_seeds as SS                                    # noqa: E402
import screen_worker as SW                                   # noqa: E402
import g1_fit as GF                                          # noqa: E402
import g1_labels as GL                                       # noqa: E402
import g1_math as GM                                         # noqa: E402
import dataset_knee as D                                     # noqa: E402
import fixtures as F                                         # noqa: E402
import pi_ref                                                # noqa: E402
import seeds as V3S                                          # noqa: E402
import stage1_smoke                                          # noqa: E402
import worlds                                                # noqa: E402
from core import sha_file                                    # noqa: E402

NAMES = list(SW.E.GATED_SLOTS)
K, I10, I700 = NAMES.index("edge_knee"), NAMES.index("s10"), NAMES.index("edge_700")


class Seeds(unittest.TestCase):
    def test_record_matches_derivation_and_is_clean(self):
        record = SS.load()
        self.assertTrue(record["audit"]["clean"])
        self.assertEqual(record["numpy"]["random_control"], SB.GS2.load()["numpy"]["random_control"])
        mine = {r[f] for r in record["training"] for f in SS.STAGE1_FIELDS}
        self.assertFalse(mine & stage1_smoke.sealed_values())
        self.assertEqual(len(SS.g1_seeds()), 3)
        self.assertTrue(all(0 <= s < 2 ** 32 for s in SS.g1_seeds()))


class Bind(unittest.TestCase):
    def test_protocol_view(self):
        rows = SB.protocol_view({"provenance": "research"})["seeds"]
        self.assertEqual([r["index"] for r in rows], [0, 1, 2])
        self.assertEqual(rows, SS.training_rows())
        with self.assertRaises(ValueError):
            SB.protocol_view({"provenance": "research", "training_seeds": rows})
        smoke = SB.protocol_view({"provenance": "smoke", "training_seeds": rows})["seeds"]
        self.assertEqual(smoke, rows)

    def test_seeds_for_refuses_foreign_research_data(self):
        fake = SimpleNamespace(metadata={"provenance": "research", "seed": SB.ENTROPY}, digest="not-a-v2-digest")
        with self.assertRaises(ValueError):
            SB.seeds_for(fake, None)

    def test_kappa_patch(self):
        class Runtime:
            def make_adapter(self, context, state, arm, kappa, *rest):
                return SimpleNamespace(arm=arm, kappa=kappa)
        rt = Runtime()
        SB.patch_runtime(rt)
        SB.patch_runtime(rt)
        self.assertEqual(rt.make_adapter(None, None, "bca", 3.0).kappa, SB.KAPPA)
        self.assertEqual(rt.make_adapter(None, None, "native", 0.0).kappa, 0.0)
        self.assertEqual(rt.make_adapter(None, None, "distance", 0.04).kappa, 0.04)

    def test_v2_banks_and_anchors(self):
        starts = SB.bank_states("stage1_e2_bank")
        self.assertEqual(starts.shape, (128, 13))
        self.assertEqual(starts.dtype, np.float64)
        e1 = SB.bank_states("stage1_e1_bank")
        self.assertEqual(e1.shape, (V3S.BANKS["stage1_e1_bank"]["count"], 13))
        e2, e1a = SW.W1.anchors()
        self.assertEqual(len(e2), 96)
        self.assertEqual(sorted({a["t"] for a in e2}), [0, 25, 50])
        self.assertTrue(all(a["both_worlds_identical"] for a in e2))
        self.assertEqual(len(e1a), V3S.BANKS["stage1_e1_anchors"]["count"])

    def test_collection_record_matches_the_formal_datasets(self):
        record = json.loads((S.COLLECTION / "collection.json").read_text(encoding="utf-8"))
        for world, spec in record["datasets"].items():
            self.assertEqual(spec["sha256"], SB.DIGESTS[world])
            self.assertTrue((S.COLLECTION / spec["file"]).is_file())
            self.assertEqual(record["data_plan"][world]["rows"], {"cutoff": 47978, "random": 47978})


# ------------------------------------------------------------------------------------------------ synthetic reports

def anchors_block(q1_s10, qpi_s10, *, stall_trained, q1_s10_benign=None, qpi_s10_benign=None, u_s10_harmful=None, q1_700=0.0):
    """96 anchors: the first 48 'harmful' (s10 truly worse than the knee), the last 48 'benign' when given."""
    n = 96
    qpi, q1 = np.zeros((n, len(NAMES))), np.zeros((n, len(NAMES)))
    qpi[:, I10], q1[:, I10], q1[:, I700] = qpi_s10, q1_s10, q1_700
    if qpi_s10_benign is not None:
        qpi[48:, I10], q1[48:, I10] = qpi_s10_benign, q1_s10_benign
    block = {"gated_slots": NAMES, "q1": q1.tolist(), "qpi_clip": qpi.tolist(), "valid": [True] * n}
    if stall_trained:
        ns = qpi.copy()
        ns[:48, I10] = 0.8                                     # the smooth value: rho_own = 1.8 on the harmful half
        block["qpi_nostall"], block["nostall_slots"] = ns.tolist(), NAMES
    if u_s10_harmful is not None:
        u = np.zeros((n, len(NAMES)))
        u[:48, I10] = u_s10_harmful
        block["U"] = u.tolist()
    return block


def run_entry(entry, j, dc, block, *, stall_trained, j_nostall_world=None, dc_nostall_world=None, entry_nostall_world=0.0):
    own = {"e1/pi": {"J": -120.0}, "e2/pi": {"J": j}, "e2/delta_clip": dc,
           "e2_rates": {"r_deep_minus": entry, "a123_past_knee": 0.0}}
    out = {"own_world": own, "anchors_e2": block}
    if stall_trained:
        out["nostall_world"] = {"e2/pi": {"J": j_nostall_world}, "e2/delta_clip": dc_nostall_world,
                                "e2_rates": {"r_deep_minus": entry_nostall_world, "a123_past_knee": 0.0}}
    return {"J_E1_untrained": -200.0, "020000": out}


def native_eval(i, *, entry_c=0.6, entry_r=0.25, bc=0.2, q1_c=0.2, q1_r=-0.9, dc_c=5.0, dc_r=1.0, dc_c_ns=1.0, j_c=-80.0, j_c_nsw=-60.0):
    runs = {
        SM.name(i, "cutoff", "stall3"): run_entry(entry_c, j_c, dc_c, anchors_block(q1_c, -1.0, stall_trained=True, q1_s10_benign=0.7, qpi_s10_benign=0.5),
                                              stall_trained=True, j_nostall_world=j_c_nsw, dc_nostall_world=dc_c_ns),
        SM.name(i, "random", "stall3"): run_entry(entry_r, -78.0, dc_r, anchors_block(q1_r, -1.0, stall_trained=True, q1_s10_benign=0.5, qpi_s10_benign=0.5),
                                              stall_trained=True, j_nostall_world=-60.0, dc_nostall_world=1.0),
        SM.name(i, "cutoff", "nostall"): run_entry(0.6, -60.0, 1.0, anchors_block(0.8, 0.8, stall_trained=False), stall_trained=False),
        SM.name(i, "random", "nostall"): run_entry(0.6, -60.0, 1.0, anchors_block(0.8, 0.8, stall_trained=False), stall_trained=False),
    }
    return {"runs": runs, "bc_only": {"e2_rates": {"r_deep_minus": bc}}, "status": "completed", "transitions": 10}


def bca_eval(i, *, entry=0.3, j=-70.0, j_ns=-61.0, u=0.1):
    runs = {
        SM.name(i, "cutoff", "stall3"): run_entry(entry, j, 5.0, anchors_block(0.2, -1.0, stall_trained=True, q1_s10_benign=0.7, qpi_s10_benign=0.5, u_s10_harmful=u),
                                              stall_trained=True, j_nostall_world=-60.0, dc_nostall_world=1.0),
        SM.name(i, "random", "stall3"): run_entry(0.2, -75.0, 1.0, anchors_block(-0.9, -1.0, stall_trained=True, u_s10_harmful=u),
                                              stall_trained=True, j_nostall_world=-60.0, dc_nostall_world=1.0),
        SM.name(i, "cutoff", "nostall"): run_entry(0.5, j_ns, 1.0, anchors_block(0.8, 0.8, stall_trained=False, u_s10_harmful=u), stall_trained=False),
        SM.name(i, "random", "nostall"): run_entry(0.5, -60.0, 1.0, anchors_block(0.8, 0.8, stall_trained=False, u_s10_harmful=u), stall_trained=False),
    }
    return {"runs": runs, "status": "completed", "transitions": 10}


G1_PASS = {"headline": "PASS", "q_star": 0.6, "resolution_band": 0.2, "all_labels": ["PASS"]}


class Reads(unittest.TestCase):
    def decide(self, native=None, bca=None, g1=G1_PASS):
        native = native or {i: native_eval(i) for i in SM.SEEDS}
        bca = {i: bca_eval(i) for i in SM.SEEDS} if bca is None else bca
        return SM.decide(native, bca, SB.KAPPA, g1)

    def test_go(self):
        out = self.decide()
        self.assertEqual(out["reading"], "GO", out["reasons"])
        r = out["reads"][0]
        self.assertAlmostEqual(r["O1"]["median_kc_cutoff_s10"], 0.7)
        self.assertAlmostEqual(r["R1"]["excess"], 0.4)
        self.assertAlmostEqual(r["B1"]["delta_entry_stall"], -0.3)
        self.assertAlmostEqual(r["B2"]["q_rank_gain_s10"], 1.0)
        self.assertAlmostEqual(r["B3s"]["selectivity"], 11.0)
        self.assertEqual(r["L1n"]["n_valid_positive_rho"], 48)
        self.assertEqual(r["L1n"]["share"], 1.0)

    def test_stop_not_reached_and_mixed(self):
        self.assertEqual(self.decide({i: native_eval(i, entry_c=0.3) for i in SM.SEEDS})["reading"], "STOP: not reached")
        mixed = {i: native_eval(i, entry_c=0.3 if i == 0 else 0.6) for i in SM.SEEDS}
        self.assertEqual(self.decide(mixed)["reading"], "MIXED")

    def test_nothing_to_repair(self):
        self.assertEqual(self.decide({i: native_eval(i, q1_c=-1.0) for i in SM.SEEDS})["reading"], "NO-GO: nothing to repair")

    def test_no_effect_over_correction_and_inconclusive(self):
        self.assertEqual(self.decide(bca={i: bca_eval(i, entry=0.6) for i in SM.SEEDS})["reading"], "NO-GO: no effect at this dose and budget")
        self.assertEqual(self.decide(bca={i: bca_eval(i, j_ns=-70.0) for i in SM.SEEDS})["reading"], "NO-GO: over-correction")
        out = self.decide(g1={**G1_PASS, "headline": "NOT REPRESENTABLE"})
        self.assertEqual(out["reading"], "INCONCLUSIVE")
        self.assertTrue(any(r.startswith("G1") for r in out["reasons"]))
        self.assertEqual(self.decide(bca={})["reading"], "INCOMPLETE")

    def test_r1_consensus(self):
        self.assertEqual(SM.r1_consensus({i: native_eval(i) for i in SM.SEEDS})[0], "holds")
        self.assertEqual(SM.r1_consensus({i: native_eval(i, entry_c=0.3) for i in SM.SEEDS})[0], "fails")


# ------------------------------------------------------------------------------------------------ the supervisor

def fake_dispatch(*, fail=(), r0_fail_early=False, g1_headline="PASS", entry_c=0.6, fail_error=None):
    def dispatch(job_dir, cap):
        job = json.loads((Path(job_dir) / "job.json").read_text(encoding="utf-8"))
        kind, i = job["kind"], job.get("seed_index", 0)
        report = {"job": job["id"], "kind": kind, "status": "completed", "transitions": 10}
        if kind == "g1":
            report["decision"] = {**G1_PASS, "headline": g1_headline}
        elif kind == "r0":
            report.update(seeds={}, adequacy_ok=True, fail_early=r0_fail_early)
        elif kind == "evaluate":
            report.update(native_eval(i, entry_c=entry_c))
        elif kind == "evaluate_bca":
            report.update(bca_eval(i))
        exit_code = 0
        if job["id"] in fail:
            report["status"], exit_code = "failed", 1
            if fail_error:
                report["error"] = fail_error
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

    def execute(self, dispatch):
        return S.execute(dispatch, run_dir=self.run, approval_path=self.approval, manifest_path=self.manifest, root=HERE,
                         integrity={"package_dir": str(HERE), "files": {}})

    def test_go_end_to_end(self):
        out = self.execute(fake_dispatch())
        self.assertEqual(out["reading"], "GO", out.get("reasons"))
        units = {p.name for p in (self.run / "attempt_1").iterdir() if p.is_dir()}
        self.assertEqual(len(units), 48)
        self.assertTrue((self.run / "attempt_1" / "decision.json").is_file())
        self.assertTrue((self.run / "attempt_1" / "SCREEN_RESULT.json").is_file())
        self.assertIn("bca_branch_s2_random_nostall", units)
        job = json.loads((self.run / "attempt_1" / "bca_branch_s0_cutoff_stall3" / "job.json").read_text())
        self.assertTrue(job["native_branch_dir"].endswith("branch_s0_cutoff_stall3"))

    def test_g1_stops_before_training(self):
        out = self.execute(fake_dispatch(g1_headline="NOT REPRESENTABLE"))
        self.assertEqual(out["reading"], "INCONCLUSIVE")
        units = {p.name for p in (self.run / "attempt_1").iterdir() if p.is_dir()}
        self.assertEqual(units, {"preflight", "g1"})

    def test_r0_and_r1_stops(self):
        out = self.execute(fake_dispatch(r0_fail_early=True))
        self.assertEqual(out["reading"], "STOP: not reached")
        units = {p.name for p in (self.run / "attempt_1").iterdir() if p.is_dir()}
        self.assertFalse(any(u.startswith("branch_") for u in units))
        self.write_approval(1)
        run2 = self.tmp / "run2"
        out = S.execute(fake_dispatch(entry_c=0.3), run_dir=run2, approval_path=self.approval, manifest_path=self.manifest, root=HERE,
                        integrity={"package_dir": str(HERE), "files": {}})
        self.assertEqual(out["reading"], "STOP: not reached")
        units = {p.name for p in (run2 / "attempt_1").iterdir() if p.is_dir()}
        self.assertTrue(any(u.startswith("evaluate_s") for u in units))
        self.assertFalse(any(u.startswith("bca_branch") for u in units))

    def test_incomplete_then_reuse_on_attempt_2(self):
        out = self.execute(fake_dispatch(fail=("warmup_s1_cutoff_nostall",)))
        self.assertEqual(out["reading"], "INCOMPLETE")
        with self.assertRaises(ValueError):                      # attempt 1 again is refused
            self.execute(fake_dispatch())
        self.write_approval(2)
        out = self.execute(fake_dispatch())
        self.assertEqual(out["reading"], "GO", out.get("reasons"))
        self.assertTrue(out["units"]["warmup_s0_cutoff_stall3"].startswith("reused"))
        self.assertEqual(out["units"]["warmup_s1_cutoff_nostall"], "completed")
        self.write_approval(3)
        with self.assertRaises(ValueError):                      # no attempt after a decided attempt 2
            self.execute(fake_dispatch())

    def test_attempt_3_only_after_the_named_loader_fault(self):
        lineage = "KneeDatasetError: Episode 16: initial state lineage differs"
        self.execute(fake_dispatch(fail=("g1",)))                                   # attempt 1 INCOMPLETE at g1
        self.write_approval(2)
        out = self.execute(fake_dispatch(fail=("g1",), fail_error=lineage))         # attempt 2 INCOMPLETE, the named fault
        self.assertEqual(out["reading"], "INCOMPLETE")
        self.write_approval(3)
        out = self.execute(fake_dispatch())                                          # amendment A1: attempt 3 runs
        self.assertEqual(out["reading"], "GO", out.get("reasons"))
        self.assertTrue(out["units"]["preflight"].startswith("reused"))
        self.assertEqual(out["units"]["g1"], "completed")
        self.write_approval(4)
        with self.assertRaises(ValueError):                                          # never a fourth
            self.execute(fake_dispatch())
        other = self.tmp / "other"                                                   # a different attempt-2 fault: no attempt 3
        for attempt, error in ((1, None), (2, "ValueError: something else")):
            self.write_approval(attempt)
            S.execute(fake_dispatch(fail=("g1",), fail_error=error), run_dir=other, approval_path=self.approval,
                      manifest_path=self.manifest, root=HERE, integrity={"package_dir": str(HERE), "files": {}})
        self.write_approval(3)
        with self.assertRaises(ValueError):
            S.execute(fake_dispatch(), run_dir=other, approval_path=self.approval, manifest_path=self.manifest, root=HERE,
                      integrity={"package_dir": str(HERE), "files": {}})

    def test_approval_wording_and_manifest_are_checked(self):
        self.approval.write_text(json.dumps({"approval": "I approve something else", "package_manifest_sha256": sha_file(self.manifest), "attempt": 1}))
        with self.assertRaises(ValueError):
            self.execute(fake_dispatch())
        self.write_approval(1)
        self.manifest.write_text(json.dumps({**json.loads(self.manifest.read_text()), "files": {"screen.py": "0" * 64}}))
        with self.assertRaises(ValueError):
            self.execute(fake_dispatch())


class Data(unittest.TestCase):
    """Change 7 (amendment A1): the authenticated, non-regenerating loader."""

    def test_loads_both_archives_as_the_pipeline_class_without_regeneration(self):
        import dataset_knee as D
        import screen_data as SD
        calls = []
        original = SD.authenticate_producer
        for world in ("stall3", "nostall"):
            ds = SB.load_dataset(world)
            self.assertIs(type(ds), D.KneeDataset)
            self.assertEqual(ds.digest, SB.DIGESTS[world])
            self.assertEqual(ds.metadata["world"], world)
            self.assertEqual(ds.arrays["states"].shape, (1280, 101, 13))
            with self.assertRaises((ValueError, AttributeError)):
                ds.arrays["states"][0, 0, 0] = 1.0                 # immutable bytes
        self.assertIs(D.load, SD.load)                             # change 7 installed
        self.assertEqual(SD.authenticate_producer()["verified_source_files"], 44)
        self.assertEqual(SD.receipt()["resets"], 0)

    def test_refuses_any_other_path(self):
        import screen_data as SD
        with self.assertRaises(ValueError):
            SD.load(HERE / "SCREEN_RULE.md")
        with self.assertRaises(ValueError):
            SD.load(SB.DATASET_FILES["stall3"].parent / "report.json")

    def test_refuses_changed_archive_bytes(self):
        import screen_data as SD
        spec = dict(SD.ARCHIVES["stall3"])
        try:
            SD.ARCHIVES["stall3"] = {**spec, "archive_sha256": "0" * 64}
            with self.assertRaises(ValueError):
                SD.load(SB.DATASET_FILES["stall3"])
        finally:
            SD.ARCHIVES["stall3"] = spec


# ------------------------------------------------------------------------------------------------ the g1 unit

class G1Math(unittest.TestCase):
    def test_band_flags_and_median(self):
        rho = np.array([1.0, 2.0, 0.0, -1.0, 0.5])
        kc = np.array([0.4, 0.8, 0.1, 0.0, -0.2])
        out = GM.l1(kc, rho)
        self.assertEqual(out["n_valid"], 3)
        self.assertEqual(out["n_pass"], 1)                       # 0.4 <= 0.45 passes; 0.8 > 0.75 and -0.2 < -0.15 fail
        self.assertEqual(out["nonpositive_rho"]["count"], 2)

    def test_binomial_and_q_star(self):
        self.assertEqual(GM.binomial_tail(10, 0.5, 0), 1.0)
        self.assertEqual(GM.binomial_tail(10, 0.5, 11), 0.0)
        self.assertAlmostEqual(GM.binomial_tail(10, 0.5, 5), 0.623046875, places=9)
        self.assertEqual(GM.q_star(1.0, 80)["q_star"], 1.0)
        q = GM.q_star(0.7, 80)["q_star"]
        self.assertTrue(0.5 <= q <= 0.7, q)
        self.assertLess(GM.q_star(0.5, 80)["q_star"], q)

    def test_labels(self):
        head, labels = GM.labels(i0_ok=True, f0_ok=True, median_ok_by_seed={0: True, 1: True, 2: True},
                                 share_by_seed={0: 0.8, 1: 0.7, 2: 0.75}, share_pooled=0.75)
        self.assertEqual((head, labels), ("PASS", ["PASS"]))
        head, _ = GM.labels(i0_ok=True, f0_ok=True, median_ok_by_seed={0: True}, share_by_seed={0: 0.5}, share_pooled=0.5)
        self.assertEqual(head, "NOT REPRESENTABLE")
        head, labels = GM.labels(i0_ok=False, f0_ok=True, median_ok_by_seed={0: False}, share_by_seed={0: 0.5}, share_pooled=0.5)
        self.assertEqual(head, "INCONCLUSIVE")
        self.assertEqual(len(labels), 2)


class G1Fit(unittest.TestCase):
    def test_upstream_match(self):
        self.assertTrue(GF.upstream_match())

    def test_fit_learns_a_synthetic_target_and_is_deterministic(self):
        rng = np.random.default_rng(3)
        obs = rng.normal(size=(600, 14)).astype(np.float32)
        act = rng.uniform(-1, 1, size=(600, 4)).astype(np.float32)
        y = (-3.0 * obs[:, 0] + 2.0 * act[:, 0] ** 2 + 1.0).astype(np.float32)
        mean, std = GF.normalizer(obs)
        params, curve = GF.fit(obs, act, y, mean, std, 7, (1000, 3000))
        self.assertLess(curve[3000], curve[1000])
        self.assertLess(curve[3000], 0.5 * float(np.var(y)))
        q = GF.q_values(params[3000], mean, std, obs[:5], act[:5])
        self.assertEqual(q.shape, (5,))
        again, _ = GF.fit(obs, act, y, mean, std, 7, (1000, 3000))
        self.assertTrue(np.array_equal(GF.q_values(again[3000], mean, std, obs[:5], act[:5]), q))
        other, _ = GF.fit(obs, act, y, mean, std, 8, (1000,))
        self.assertFalse(np.array_equal(GF.q_values(other[1000], mean, std, obs[:5], act[:5]), GF.q_values(params[1000], mean, std, obs[:5], act[:5])))


class G1Labels(unittest.TestCase):
    def test_exact_labels_on_a_tiny_dataset(self):
        ds = D.generate(episodes=12, entropy=F.TEST_ENTROPY + 31, world="stall3", provenance="smoke")
        rows = GL.fitting_rows_with_states(ds.arrays)
        self.assertGreater(len(rows["obs"]), 0)
        values, receipt = GL.labels(rows, "stall3")
        self.assertTrue(receipt["known_answer_t99"]["ok"])
        env = worlds.load_environment()
        physics = worlds.physics("stall3")
        r = 7 if len(values) > 7 else 0
        x, t = rows["states"][r].copy(), int(rows["times"][r])
        total, a = 0.0, rows["actions"][r].astype(np.float64)
        for offset in range(100 - t):
            if offset > 0:
                obs = np.concatenate([x, [(100 - (t + offset)) / 100]]).astype(np.float32)[None]
                a = np.clip(pi_ref.act(obs)[0].astype(np.float64), -worlds.KNEE, worlds.KNEE)
            x, reward = env.dynamics(x[None], a[None], physics)
            x, total = x[0], total + 0.98 ** offset * float(reward[0])
        self.assertAlmostEqual(values[r], total, places=9)


if __name__ == "__main__":
    unittest.main()

"""Step 0.4 (A3) tests on Windows: L2 predictors and rules on known polynomials, kNN and L1.5, anchor construction from
the sealed draw rules, and a full execute() dry run on a smoke-provenance 1,280-episode collection (test-only entropy
and seeds) with a stand-in actor. No checkpoint is opened and no benchmark seed is used."""
from pathlib import Path
from types import SimpleNamespace
from unittest import mock
import json
import os
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.dont_write_bytecode = True

import numpy as np

import fixtures as F
import seeds as S
import step03
import step04
import step04_math as SM4
from core import sha_file
from knee_region import GATED_PAST_KNEE, S9, SLOTS


class L2Math(unittest.TestCase):
    def setUp(self):
        self.xs = np.asarray([SLOTS[n] for n in S9])

    def test_predictors_recover_polynomials(self):
        quad = np.tile(-30 + 2 * self.xs - 3 * self.xs ** 2, (4, 1))
        preds = SM4.predictors(quad)
        for name in ("P1", "P3"):
            self.assertTrue(np.allclose(preds[name](-0.9), -30 + 2 * (-0.9) - 3 * 0.81))
        edge = np.asarray([SLOTS[n] for n in SM4.EDGE5])
        line = np.tile(-30 + 5 * self.xs, (2, 1))
        self.assertTrue(np.allclose(SM4.predictors(line)["P2"](-1.0), -35.0))
        self.assertEqual(len(edge), 5)

    def test_rho_hat_takes_the_minimum_and_item5_the_max_error(self):
        q_s9 = np.tile(-30 + 2 * self.xs, (3, 1))                     # linear: every predictor extrapolates the same
        smooth = {n: -30 + 2 * SLOTS[n] for n in GATED_PAST_KNEE}
        q_stall = {n: np.full(3, smooth[n] - 1.0) for n in GATED_PAST_KNEE}          # true drop of 1.0 past the knee
        q_nostall = {n: np.full(3, smooth[n] + 0.05) for n in ("s05", "s10", "s20")}
        res = SM4.neighbor_residuals(q_s9, q_stall, q_nostall)
        self.assertTrue(np.allclose(res["rho_hat"]["s10"], 1.0) and np.allclose(res["item5_error"]["s10"], 0.05))
        items, pv = SM4.l2_rule(res, np.full(3, smooth["edge_knee"] if "edge_knee" in smooth else -31.6), q_stall["s10"], np.ones(3, bool))
        self.assertTrue(items["item5"] and items["item1"] and items["item2"] and items["item3"])
        self.assertFalse(items["item6"])                              # p10 also dropped 1.0 here, so the sign control fails

    def test_predictors_differ_on_curved_data_min_and_max(self):
        xs = self.xs
        q = (-30 + 2 * xs + 4 * xs ** 3 + 3 * np.maximum(xs + 0.775, 0) ** 2)[None, :]     # cubic plus a bend at the edge
        preds = SM4.predictors(q)
        edge = np.asarray([SLOTS[n] for n in SM4.EDGE5])
        line = np.polyfit(edge, q[0, [S9.index(n) for n in SM4.EDGE5]], 1)
        self.assertTrue(np.allclose(preds["P2"](-1.0), np.polyval(line, -1.0)))
        self.assertTrue(np.allclose(preds["P1"](-1.0), np.polyval(np.polyfit(xs, q[0], 2), -1.0)))
        self.assertTrue(np.allclose(preds["P3"](-1.0), np.polyval(np.polyfit(xs, q[0], 3), -1.0)))
        at = {n: np.asarray([p(SLOTS[n]) for p in preds.values()])[:, 0] for n in GATED_PAST_KNEE}
        self.assertGreater(np.ptp(at["s10"]), 1e-3)                     # the three predictors really disagree
        q_w = {n: np.asarray([at[n].mean()]) for n in GATED_PAST_KNEE}
        res = SM4.neighbor_residuals(q, q_w, {n: q_w[n] for n in ("s05", "s10", "s20")})
        self.assertAlmostEqual(float(res["rho_hat"]["s10"][0]), float(at["s10"].min() - at["s10"].mean()))
        self.assertAlmostEqual(float(res["rho_hat_max"]["p10"][0]), float(at["p10"].max() - at["p10"].mean()))
        drop = np.asarray([0.3, 0.2, 0.25])                          # item 4: Q_stall(-KNEE) - Q_stall(s10) >= 0.25 on 70%
        items, _ = SM4.l2_rule({**res, "rho_hat": {k: np.repeat(v, 3) for k, v in res["rho_hat"].items()},
                                "rho_hat_max": {k: np.repeat(v, 3) for k, v in res["rho_hat_max"].items()},
                                "item5_error": {k: np.zeros(3) for k in res["item5_error"]}},
                               drop, np.zeros(3), np.ones(3, bool))
        self.assertAlmostEqual(items["item4_share"], 2 / 3)
        self.assertFalse(items["item4"])

    def test_reach_forecast_half_minus_infinity_is_json_safe(self):
        for bad in (24, 48):
            res = {"rho_hat": {"s05": np.ones(48), "s10": np.ones(48)}}
            pv = np.r_[np.ones(48 - bad, bool), np.zeros(bad, bool)]
            out = SM4.reach_forecast(res, np.ones(48, bool), pv)
            self.assertTrue(out["s05"]["median_is_minus_infinity"])
            self.assertIsNone(out["s05"]["median_rho_hat"])
            self.assertFalse(out["s05"]["at_or_above_bar"])
            with tempfile.TemporaryDirectory() as tmp:
                step03._write_new(Path(tmp) / "f.json", out)              # the strict writer accepts it

    def test_reach_forecast_minus_infinity(self):
        res = {"rho_hat": {"s05": np.asarray([0.9, 0.8, 0.7, 0.2]), "s10": np.asarray([1.0, 0.9, 0.8, 0.1])}}
        out = SM4.reach_forecast(res, np.asarray([True, True, True, True]), np.asarray([True, True, False, True]))
        self.assertEqual(out["minus_infinity"], 1)
        self.assertAlmostEqual(out["s05"]["median_rho_hat"], 0.5)      # (0.2, 0.8, 0.9, -inf) -> (0.2 + 0.8) / 2

    def test_knn_matches_brute_force_and_l15(self):
        rng = np.random.Generator(np.random.PCG64(4))
        rows, eps = rng.normal(size=(500, 14)), np.repeat(np.arange(50), 10)
        q = rng.normal(size=(7, 14))
        brute = np.sort(np.linalg.norm(q[:, None, :] - rows[None], axis=2), axis=1)[:, 63]
        self.assertTrue(np.allclose(SM4.knn_distance(q, rows, eps), brute))
        excl = SM4.knn_distance(rows[:3], rows, eps, exclude_episodes=eps[:3])
        brute_ex = [np.sort(np.linalg.norm(rows[i] - rows[eps != eps[i]], axis=1))[63] for i in range(3)]
        self.assertTrue(np.allclose(excl, brute_ex))
        # Within-group calibration (amendment 1): a sparse group (deep rows at t = 0) is judged against its own
        # leave-one-episode-out distances, so draws from the same group pass and shifted draws fail.
        n_ep, per = 400, 10
        times = np.tile(np.arange(per) * 25 % 100, n_ep)
        deep = np.repeat(np.arange(n_ep) % 5 == 0, per)
        train = rng.normal(size=(n_ep * per, 14))
        train[deep, 10] += 3.0                                                     # the deep group sits apart
        train[:, 13] = times / 100.0
        eps = np.repeat(np.arange(n_ep), per)
        def group_draw(n, t, shift=0.0):
            x = rng.normal(size=(n, 14)); x[:, 10] += 3.0 + shift; x[:, 13] = t / 100.0
            return x
        ok = SM4.l15(train, eps, times, deep, group_draw(40, 0), group_draw(30, 25), np.full(30, 25))
        far = SM4.l15(train, eps, times, deep, group_draw(40, 0, shift=6.0), group_draw(30, 25), np.full(30, 25))
        self.assertTrue(ok["pass"], ok)
        self.assertFalse(far["pass"])

    def test_label_precedence(self):
        good = {s: {"item1": True, "item2": True, "item3": True, "item4": True, "item6": True} for s in range(3)}
        base = dict(plumbing_ok=True, identity_ok=True, equal_ok=True, cross_check_ok=True, item5_by_seed={0: True, 1: True, 2: True},
                    l14_ok=True, l15_ok=True, expected_ok=True, items_by_seed=good)
        self.assertEqual(SM4.label(**base)[0], "PASS")
        self.assertEqual(SM4.label(**dict(base, cross_check_ok=False))[0], "INCOMPLETE")
        self.assertEqual(SM4.label(**dict(base, item5_by_seed={0: True, 1: False, 2: True},
                                          items_by_seed={**good, 1: dict(good[1], item1=False)}))[0], "INCONCLUSIVE")
        self.assertEqual(SM4.label(**dict(base, l15_ok=False))[0], "INCONCLUSIVE")
        self.assertEqual(SM4.label(**dict(base, items_by_seed={**good, 2: dict(good[2], item6=False)}))[0], "FAIL-mixed")
        all_fail = {s: dict(good[s], item2=False) for s in range(3)}
        self.assertEqual(SM4.label(**dict(base, items_by_seed=all_fail))[0], "FAIL-consistent")


class Anchors(unittest.TestCase):
    def test_anchor_rebuild_is_deterministic_and_follows_the_rule(self):
        a, b = step04.build_anchor("step04_e2_anchors_t25", 3), step04.build_anchor("step04_e2_anchors_t25", 3)
        self.assertEqual(a["t"], 25)
        self.assertTrue(np.array_equal(a["state"], b["state"]) and np.array_equal(a["prefix_actions"], b["prefix_actions"]))
        self.assertTrue(np.all(np.abs(a["prefix_actions"][:, 1:].astype(np.float64)) <= 0.7))
        e1 = step04.build_anchor("step04_e1_anchors", 9)
        self.assertEqual(e1["t"], S.bank_item("step04_e1_anchors", 9)["t"])
        import qpi
        anchor = {"reset_seed": a["reset_seed"], "current": a["current"], "prefix_actions": a["prefix_actions"]}
        self.assertTrue(np.array_equal(qpi.raw_anchor_state(anchor), a["state"]))   # raw replay agrees with the env

    def test_layouts(self):
        for world, width in (("stall3", 19), ("nostall", 19), ("stall15_identity", 11)):
            arr, names = step04.layout_probes(2, world)
            self.assertEqual((arr.shape[1], len(names)), (width, width))


class DryRun(unittest.TestCase):
    """execute() end to end: smoke collection of 2 x 1,280 episodes and probes with a stand-in actor."""

    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        root = Path(cls.tmp.name)
        manifest = {"files": {n: sha_file(step04.HERE / n) for n in sorted(step04.A3_REQUIRED_FILES)}}
        (root / "A3_MANIFEST.json").write_text(json.dumps(manifest), encoding="utf-8")
        (root / "A3_APPROVAL.json").write_text(json.dumps({"approval": step04.APPROVAL_WORDS + ", attempt 1",
                                                           "package_manifest_sha256": sha_file(root / "A3_MANIFEST.json"),
                                                           "attempt": 1}), encoding="utf-8")
        forward = F.toy_actor()
        policy = SimpleNamespace(act=forward, assert_unchanged=lambda: None)
        rerun = (SimpleNamespace(load_native=lambda: "n", runtime_receipt=lambda n: {"device_kinds": [step03.K.RECORDED_DEVICE_KIND]}),
                 SimpleNamespace(validate_protocol=lambda p: None, assemble_exports=lambda *a: ({}, {})), None,
                 SimpleNamespace(continuation_root=lambda: root))
        patches = [mock.patch.object(step04, "RUN", root / "step04_run"), mock.patch.object(step04, "MANIFEST", root / "A3_MANIFEST.json"),
                   mock.patch.object(step04, "APPROVAL", root / "A3_APPROVAL.json"),
                   mock.patch.object(step04, "a2_outcome", lambda approval: {"label": "PASS", "flagged": False}),
                   mock.patch.object(step03, "rerun1", lambda: rerun), mock.patch.object(step03, "set_worker_environment", lambda: None),
                   mock.patch.dict(os.environ, step03.WORKER_ENVIRONMENT),
                   mock.patch.object(step03, "build_policy", lambda *a: (policy, None, None))]
        for p in patches:
            p.start()
        try:
            cls.report = step04.execute(entropy=F.TEST_ENTROPY + 1, episodes=1280, provenance="smoke", test_seeds=F.TEST_SEEDS)
        finally:
            for p in patches:
                p.stop()
        cls.dir = root / "step04_run" / "attempt_1"

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_exact_count_is_recorded_and_within_cap(self):
        self.assertGreater(self.report["transitions"], 0)
        self.assertLessEqual(self.report["transitions"], step04.STEP_TRANSITIONS)
        print(f"\n[step 0.4 dry run] exact simulator steps {self.report['transitions']}, "
              f"prospective {self.report['transitions_prospective']}, cap {step04.STEP_TRANSITIONS}")

    def test_report_and_units(self):
        self.assertIn(self.report["label"], ("PASS", "FAIL", "INCONCLUSIVE"), self.report.get("reasons"))
        for name in ("ATTEMPT.json", "collection.json", "probes.json", "report.json", "knee_edge_v1_stall3.npz", "knee_edge_v1_nostall.npz"):
            self.assertTrue((self.dir / name).is_file(), name)

    def test_collection_checks(self):
        c = json.loads((self.dir / "collection.json").read_text())
        self.assertEqual(c["reconcile"]["label"], "PASS")
        self.assertTrue(c["thrust_check"]["ok"] and c["plumbing_ok"] and c["l14_ok"] and c["expected_ok"])
        self.assertTrue(c["l15_ok"], c["l15"])
        self.assertEqual(len(c["anchors"]["e2"]), 64)
        self.assertEqual(c["data_plan"]["stall3"]["receipts"]["cutoff"]["streams"]["train"]["w_rows"], 0)
        import dataset_knee as D
        self.assertEqual(D.load(self.dir / "knee_edge_v1_stall3.npz").digest, c["datasets"]["stall3"]["sha256"])

    def test_exact_properties_hold_for_any_policy(self):
        p = json.loads((self.dir / "probes.json").read_text())
        self.assertEqual(p["checks"], {"identity": True, "equal": True, "cross": True})
        self.assertEqual(set(p["l2"]), {"0", "1", "2"})


class V2Caps(unittest.TestCase):
    def test_counter_counts_every_step_and_fails_closed(self):
        import units
        import worlds
        with units.SimulatorCounter(3) as counter:
            env = worlds.environment("stall3")
            env.reset(seed=1, options={"start": {"current": (0.5, 0.5)}})
            for _ in range(3):
                env.step(np.zeros(4, np.float32))
            self.assertEqual(counter.steps, 3)
            with self.assertRaises(step03.CapExhausted):
                env.step(np.zeros(4, np.float32))
        self.assertEqual(counter.steps, 3)
        module = worlds.load_environment()
        with units.SimulatorCounter(10) as counter2:
            self.assertIsNot(module.dynamics, counter2._originals[2])   # the raw-loop path is wrapped too
            phys = worlds.physics("stall3")
            states = np.tile(np.asarray(env.state, np.float64), (4, 1))
            module.dynamics(states, np.zeros((4, 4), np.float64), phys)
            self.assertEqual(counter2.steps, 4)
        self.assertIs(module.dynamics, counter2._originals[2])
        env2 = worlds.environment("stall3")                 # restored: the original, uncounted environment
        env2.reset(seed=1, options={"start": {"current": (0.5, 0.5)}})
        env2.step(np.zeros(4, np.float32))
        self.assertEqual(counter.steps, 3)

    def test_watchdog_writes_incomplete_and_exits(self):
        calls, written = [], []
        with mock.patch.object(step04, "_finish", lambda d, r: written.append(dict(r))):
            step04._watchdog(Path("x"), {"approval": {}}, exit=calls.append)
        self.assertEqual(calls, [3])
        self.assertEqual(written[0]["label"], "INCOMPLETE")
        self.assertTrue(written[0]["watchdog"])


class V2CarriedStep03(unittest.TestCase):
    """PROTOCOL_V2.md change 1: v1's single step 0.3 report is read and pinned; only a G0.5-share INCONCLUSIVE (or a
    PASS) with known-answer, recount and reach passing is carried, and the approval must carry FLAG_ACK."""
    ACK = {"approval": step04.APPROVAL_WORDS + ", attempt 1. " + step04.FLAG_ACK}

    def write(self, root, **change):
        report = {"approval": {"a2_manifest_sha256": step04.A2_MANIFEST_SHA256}, "label": "INCONCLUSIVE",
                  "reasons": [step04.G05_STOP_REASON + "2"], "known_answer_ok": True, "pooled_past_knee_ok": True,
                  "probe_forecast": {"seed0_full": 0.6}, "visited_forecast": {"seed0_full": 0.6}}
        report.update(change)
        path = Path(root) / "knee_edge_ood_v1" / "report.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(report), encoding="utf-8")
        return path

    def outcome(self, path, approval=None):
        with mock.patch.object(step04, "HERE", path.parent.parent / "knee_edge_ood_v2"):
            return step04.a2_outcome(approval or self.ACK, report_path=path, report_sha256=sha_file(path))

    def test_the_real_v1_report_is_carried(self):
        if not step04.V1_A2_REPORT.is_file():
            self.skipTest("v1 step 0.3 report not present")
        result = step04.a2_outcome(self.ACK)
        self.assertEqual(result["label"], "INCONCLUSIVE")
        self.assertEqual(result["report_sha256"], step04.V1_A2_REPORT_SHA256)

    def test_g05_inconclusive_is_carried_and_others_stop(self):
        with tempfile.TemporaryDirectory() as root:
            self.assertEqual(self.outcome(self.write(root))["label"], "INCONCLUSIVE")
            self.assertEqual(self.outcome(self.write(root, label="PASS (flagged)", reasons=[]))["label"], "PASS (flagged)")
            for change in ({"reasons": ["something else"]},
                           {"reasons": [step04.G05_STOP_REASON + "2", "probe forecast 0.300 < 0.35 in seed0_full"]},
                           {"label": "INCOMPLETE"}, {"label": "FAIL (reach)"},
                           {"known_answer_ok": False}, {"pooled_past_knee_ok": False},
                           {"approval": {"a2_manifest_sha256": "other"}}):
                with self.assertRaises(Exception, msg=str(change)):
                    self.outcome(self.write(root, **change))

    def test_reach_is_recorded_not_gated(self):
        with tempfile.TemporaryDirectory() as root:
            out = self.outcome(self.write(root, probe_forecast={"seed0_full": 0.1}))
            self.assertEqual(out["reach_recorded_not_gated"]["probe_forecast"], {"seed0_full": 0.1})
            self.assertFalse(out["v1_flagged"])
            self.assertTrue(out["v2_prediction_required"])

    def test_v2_never_runs_step03(self):
        self.assertFalse((step04.HERE / "step03_run").exists())
        self.assertFalse((step04.HERE / "A2_APPROVAL.json").exists())

    def test_ack_and_pin_are_required(self):
        with tempfile.TemporaryDirectory() as root:
            path = self.write(root)
            with self.assertRaises(Exception):
                self.outcome(path, approval={"approval": step04.APPROVAL_WORDS + ", attempt 1"})
            with mock.patch.object(step04, "HERE", path.parent.parent / "knee_edge_ood_v2"), self.assertRaises(Exception):
                step04.a2_outcome(self.ACK, report_path=path, report_sha256="0" * 64)


if __name__ == "__main__":
    unittest.main()

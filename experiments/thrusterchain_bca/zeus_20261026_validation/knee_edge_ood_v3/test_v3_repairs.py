"""knee_edge_ood_v3 repair tests (PROTOCOL_V3.md R1-R7), on Windows with synthetic inputs: V0(f) boundaries, the KC
decomposition identity and the arm contrast, exact row counting over every leading dimension with attempted versus
completed counts, the widened integrity closure, the single terminal owner, the cumulative ledger and the exploratory
wrapper's manifest binding. No checkpoint is opened, no model is trained and no benchmark seed is used."""
from pathlib import Path
from unittest import mock
import json
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.dont_write_bytecode = True

import numpy as np

import integrity
import stage1
import stage1_exploratory as X
import stage1_math as SM
import stage2
import step03
import step04
import units as U
import worlds
from core import sha_file
from test_stage1 import seed_inputs


def passing_r0():
    return {"adequacy_ok": True, "fail_early": False}


class R1V0f(unittest.TestCase):
    """V0(f): median rho_hat_R(s10) >= 0.30 and median KC_s10(RANDOM) <= 0.3 x (median rho_hat_R(s10) + 0.15), over
    the R-valid anchors, per seed; INCONCLUSIVE when any seed fails."""

    def rules(self, **change):
        c, r, swap = seed_inputs()
        r.update(change)
        return SM.seed_rules(c, r, swap)

    def test_boundary_equality_passes_and_just_outside_fails(self):
        bound = SM.V0F["kc_ratio"] * (1.5 + SM.V0F["kc_offset"])                  # 0.3 x 1.65 = 0.495
        self.assertAlmostEqual(bound, 0.495)
        self.assertTrue(self.rules(kc_s10=np.full(48, bound))["V0f"]["ok"])
        self.assertFalse(self.rules(kc_s10=np.full(48, np.nextafter(bound, 1.0)))["V0f"]["ok"])
        self.assertTrue(self.rules(rho_s10=np.full(48, 0.30), kc_s10=np.zeros(48))["V0f"]["ok"])
        self.assertFalse(self.rules(rho_s10=np.full(48, np.nextafter(0.30, 0.0)), kc_s10=np.zeros(48))["V0f"]["ok"])

    def test_one_sided_negative_kc_passes(self):
        self.assertTrue(self.rules(kc_s10=np.full(48, -5.0))["V0f"]["ok"])      # the bound is one-sided, as written

    def test_empty_or_nonfinite_r_valid_cannot_pass(self):
        self.assertFalse(self.rules(item5_ok=np.zeros(48, bool))["V0f"]["ok"])  # no R-valid anchor: medians None
        self.assertFalse(self.rules(kc_s10=np.full(48, np.nan))["V0f"]["ok"])
        self.assertFalse(self.rules(rho_s10=np.full(48, np.inf))["V0f"]["ok"])

    def test_report_carries_denominators_and_bound(self):
        out = self.rules()
        self.assertEqual(out["V0f"]["n_r_valid"], 48)
        self.assertAlmostEqual(out["V0f"]["kc_bound"], 0.495)
        self.assertEqual(out["V0f"]["rho_floor"], 0.30)
        self.assertIn("V0f", SM.VALIDITY)

    def test_failure_in_one_seed_is_inconclusive_even_when_all_else_passes(self):
        good = SM.seed_rules(*seed_inputs())
        bad = self.rules(kc_s10=np.full(48, 0.6))
        self.assertTrue(all(bad[k]["ok"] for k in SM.CRITERIA) and all(bad[k]["ok"] for k in ("V0a", "V0b", "V0c", "V0d", "V0e")))
        self.assertFalse(bad["validity_ok"])
        label, reasons, reading = SM.label(v1_reasons=[], r0=passing_r0(), per_seed={0: good, 1: bad, 2: good})
        self.assertEqual((label, reasons, reading), ("INCONCLUSIVE", ["seed 1: V0f fails"], None))


class R2Decomposition(unittest.TestCase):
    names = ["edge_knee", "s05", "s10"]

    def test_d_is_computed_from_the_two_references_and_reconstructs_kc(self):
        rng = np.random.default_rng(5)
        qpi_un = rng.normal(size=(6, 3)) * 50
        qpi_clip = qpi_un + rng.normal(size=(6, 3))
        q1 = qpi_un + rng.normal(size=(6, 3)) * 3                        # a critic with its own, slot-specific errors
        d = SM.kc_decomposition(q1, qpi_clip, qpi_un, self.names, "s10")
        independent = (qpi_un[:, 2] - qpi_un[:, 0]) - (qpi_clip[:, 2] - qpi_clip[:, 0])
        self.assertTrue(np.allclose(d["D"], independent, rtol=0, atol=1e-12))
        self.assertFalse(np.allclose(d["kc_own"], 0.0))
        self.assertTrue(np.all(d["identity_ok"]))
        own = (q1[:, 2] - q1[:, 0]) - (qpi_un[:, 2] - qpi_un[:, 0])
        self.assertTrue(np.allclose(d["kc_own"], own, rtol=0, atol=1e-12))
        swapped = SM.kc_decomposition(q1, qpi_un, qpi_clip, self.names, "s10")       # references swapped: D flips sign
        self.assertTrue(np.allclose(swapped["D"], -independent, rtol=0, atol=1e-12))

    def test_finite_median_keeps_denominators(self):
        out = SM.finite_median(np.asarray([1.0, np.nan, 3.0, 9.0]), np.asarray([True, True, True, False]))
        self.assertEqual(out, {"median": 2.0, "n_finite": 2, "n_mask": 3})
        self.assertIsNone(SM.finite_median(np.asarray([np.nan]), np.asarray([True]))["median"])

    def test_identity_holds_per_anchor_and_perfect_own_critic_has_zero_own_kc(self):
        rng = np.random.default_rng(3)
        qpi_un = rng.normal(size=(6, 3)) * 50
        qpi_clip = qpi_un + rng.normal(size=(6, 3))                      # the clipped reference differs per anchor
        q1 = qpi_un.copy()                                               # a critic exact for the own continuation
        d = SM.kc_decomposition(q1, qpi_clip, qpi_un, self.names, "s05")
        self.assertTrue(np.all(d["identity_ok"]))
        self.assertTrue(np.allclose(d["kc_own"], 0.0))
        self.assertTrue(np.allclose(d["kc_clip"], d["D"]) and not np.allclose(d["D"], 0.0))
        self.assertTrue(np.allclose(d["kc_clip"], d["kc_own"] + d["D"]))

    def test_nonfinite_companion_never_certifies(self):
        qpi_un = np.zeros((3, 3))
        qpi_un[1, 1] = np.nan
        d = SM.kc_decomposition(np.ones((3, 3)), np.zeros((3, 3)), qpi_un, self.names, "s05")
        self.assertEqual(d["identity_ok"].tolist(), [True, False, True])
        missing = SM.kc_decomposition(np.ones((3, 3)), np.zeros((3, 3)), np.full((3, 3), np.nan), self.names, "s05")
        self.assertFalse(np.any(missing["identity_ok"]))                     # a missing companion never certifies

    def test_gap_contrast_cancels_only_when_the_arm_gaps_are_equal(self):
        d_c = np.asarray([0.5, -0.2, 0.9, 0.0])
        mask = np.ones(4, bool)
        same = SM.gap_contrast(d_c, d_c.copy(), mask)
        self.assertTrue(same["cancels_exactly"] and same["median_D_C_minus_D_R"] == 0.0)
        diff = SM.gap_contrast(d_c, d_c - 0.3, mask)
        self.assertFalse(diff["cancels_exactly"])
        self.assertAlmostEqual(diff["median_D_C_minus_D_R"], 0.3)
        self.assertEqual((diff["share_positive"], diff["n_finite"], diff["n_mask"]), (1.0, 4, 4))
        partial = SM.gap_contrast(d_c, np.asarray([0.0, np.nan, 0.0, 0.0]), mask)
        self.assertEqual(partial["n_finite"], 3)
        self.assertIsNone(SM.gap_contrast(d_c, d_c, np.zeros(4, bool))["median_D_C_minus_D_R"])


class R3Counting(unittest.TestCase):
    def setUp(self):
        self.module = worlds.load_environment()
        self.dx, self.da = int(self.module.DX), int(self.module.DA)
        self.phys = worlds.physics("stall3")

    def test_rows_over_every_leading_dimension(self):
        with U.SimulatorCounter() as c:
            self.module.dynamics(np.zeros(self.dx), np.zeros(self.da), self.phys)
            self.assertEqual(c.steps, 1)
            self.module.dynamics(np.zeros((7, self.dx)), np.zeros((7, self.da)), self.phys)
            self.assertEqual(c.steps, 8)
            self.module.dynamics(np.zeros((3, 5, self.dx)), np.zeros((3, 5, self.da)), self.phys)
            self.assertEqual(c.steps, 23)                                              # (E, T, 13): E x T, not E
            self.module.dynamics(np.zeros((1, self.dx)), np.zeros((4, self.da)), self.phys)
            self.assertEqual(c.steps, 27)                                              # broadcast: 4 transitions
            self.module.dynamics(np.zeros((0, self.dx)), np.zeros((0, self.da)), self.phys)
            self.assertEqual(c.steps, 27)                                              # empty batch: nothing
            with self.assertRaises(ValueError):
                self.module.dynamics(np.zeros((2, self.dx)), np.zeros((3, self.da)), self.phys)
            with self.assertRaises(ValueError):
                self.module.dynamics(np.zeros((2, self.dx + 1)), np.zeros((2, self.da)), self.phys)
            self.assertEqual((c.steps, c.attempted), (27, 27))

    def test_cap_is_checked_on_attempted_work_before_the_call(self):
        with U.SimulatorCounter(5) as c:
            self.module.dynamics(np.zeros((3, self.dx)), np.zeros((3, self.da)), self.phys)
            with self.assertRaises(step03.CapExhausted):
                self.module.dynamics(np.zeros((3, self.dx)), np.zeros((3, self.da)), self.phys)
            self.assertEqual((c.steps, c.attempted), (3, 3))                            # a refused call reserves nothing

    def test_attempted_exceeds_completed_when_the_simulator_raises(self):
        original = self.module.dynamics

        def failing(states, actions, physics):
            raise RuntimeError("simulated failure")
        self.module.dynamics = failing
        try:
            with U.SimulatorCounter() as c:
                with self.assertRaises(RuntimeError):
                    self.module.dynamics(np.zeros((4, self.dx)), np.zeros((4, self.da)), self.phys)
                self.assertEqual((c.attempted, c.steps), (4, 0))
            self.assertIs(self.module.dynamics, failing)                                 # restored to what it wrapped
        finally:
            self.module.dynamics = original

    def test_environment_steps_are_not_double_counted(self):
        with U.SimulatorCounter() as c:
            env = worlds.environment("stall3")
            env.reset(seed=1, options={"start": {"current": (0.5, 0.5)}})
            for _ in range(5):
                env.step(np.zeros(4, np.float32))
            self.assertEqual((c.steps, c.attempted), (5, 5))

    def test_prefix_and_continuation_steps_reconcile_with_the_closed_form(self):
        import fixtures as F
        import qpi
        rng = np.random.default_rng(11)
        times = [0, 3, 7]
        anchors = [{"reset_seed": 100 + t, "current": (0.5, 0.5),
                    "prefix_actions": rng.uniform(-0.5, 0.5, size=(t, 4)).astype(np.float32)} for t in times]
        slots = 2
        first = np.zeros((len(times), slots, 4), np.float32)
        with U.SimulatorCounter() as c:
            qpi.raw_values(anchors, first, F.toy_actor(), "stall3")
        prefix = sum(times)                                                   # raw_anchor_state: 1-D, one per step
        continuation = sum(slots * (qpi.HORIZON - t) for t in times)          # every slot row to the horizon
        self.assertEqual((c.steps, c.attempted), (prefix + continuation, prefix + continuation))

    def test_validation_replay_counts_every_transition(self):
        """The path v2 undercounted: KneeDataset validation replays (episodes, 100, 13) states in one dynamics call.
        A 12-episode smoke collection takes 1,200 environment steps to record and 1,200 transitions to replay."""
        import dataset_knee as D
        import fixtures as F
        with U.SimulatorCounter() as c:
            D.generate(episodes=12, entropy=F.TEST_ENTROPY + 7, world="stall3", provenance="smoke")
        self.assertEqual((c.steps, c.attempted), (2400, 2400))


class R4Closure(unittest.TestCase):
    def test_record_covers_the_external_dependencies_and_detects_each_change(self):
        record = U.integrity_record()
        design = str(step03.RERUN1 / "design.py")
        launch = str(stage1.VALIDATION / "matched_policy_pilot_v1" / "launch.py")
        for path in (design, launch, str(Path(stage1.__file__).resolve()), str(step03.RERUN1 / "runtime.py")):
            self.assertIn(path, record["files"], path)
        self.assertEqual(integrity.changed_files(record), [])
        for path in (design, launch):
            tampered = {**record, "files": {**record["files"], path: "0" * 64}}
            self.assertEqual(integrity.changed_files(tampered), [path])
            with self.assertRaises(SystemExit):
                integrity.verify_record(tampered, U.HERE)
        with self.assertRaises(SystemExit):
            integrity.changed_files({"no": "files"})

    def test_payload_modules_and_core_wrapper_are_recorded(self):
        from core import PAYLOAD
        record = U.integrity_record()
        for rel in ("unifloral_td3bc_validation_v1/native_loader.py", "unifloral_checkpoint_v1/training.py",
                    "unifloral_td3bc_validation_v1/candidate.py", "unifloral_bca_pipeline_validation_v1/bca_bridge.py",
                    "transfer_bundle/core_v4/thrusterchain/environment.py",
                    "transfer_bundle/core_v4/thrusterchain/canonical/__init__.py"):
            self.assertIn(str(PAYLOAD / rel), record["files"], rel)
            tampered = {**record, "files": {**record["files"], str(PAYLOAD / rel): "0" * 64}}
            self.assertEqual(integrity.changed_files(tampered), [str(PAYLOAD / rel)])

    def test_an_added_importable_file_is_reported(self):
        with tempfile.TemporaryDirectory() as root:
            root = Path(root)
            (root / "a.py").write_text("x = 1", encoding="utf-8")
            record = {"package_dir": str(root), "files": {str(root / "a.py"): sha_file(root / "a.py")}}
            self.assertEqual(integrity.changed_files(record), [])
            (root / "json.py").write_text("raise SystemExit", encoding="utf-8")       # would shadow an import
            (root / "pkg").mkdir()
            (root / "pkg" / "__init__.py").write_text("", encoding="utf-8")
            (root / "run_dir").mkdir()                                                 # plain folders are fine
            (root / "notes.md").write_text("", encoding="utf-8")                      # non-importable files are fine
            self.assertEqual(integrity.changed_files(record), [f"added: {root / 'json.py'}", f"added: {root / 'pkg'}"])

    def test_every_package_source_must_be_manifest_listed(self):
        with tempfile.TemporaryDirectory() as root:
            root = Path(root)
            (root / "MANIFEST.json").write_text(json.dumps({"files": {"a.py": "0"}}), encoding="utf-8")
            (root / "a.py").write_text("", encoding="utf-8")
            (root / "b.py").write_text("", encoding="utf-8")
            with mock.patch.object(U, "HERE", root):
                with self.assertRaisesRegex(ValueError, "no sealed manifest lists"):
                    U.dependency_closure()

    def test_worker_records_counts_and_integrity_after_on_every_report(self):
        import stage1_worker as W
        with tempfile.TemporaryDirectory() as root:
            root = Path(root)
            watched = root / "watched.py"
            watched.write_text("x = 1", encoding="utf-8")
            record = {"package_dir": str(root / "pkgdir"), "files": {str(watched): sha_file(watched)}}
            (root / "pkgdir").mkdir()

            def edits(job, out, deadline):
                env = worlds.environment("stall3")
                env.reset(seed=1, options={"start": {"current": (0.5, 0.5)}})
                for _ in range(3):
                    env.step(np.zeros(4, np.float32))
                watched.write_text("x = 2", encoding="utf-8")
                return {}

            def fails(job, out, deadline):
                env = worlds.environment("stall3")
                env.reset(seed=1, options={"start": {"current": (0.5, 0.5)}})
                env.step(np.zeros(4, np.float32))
                raise RuntimeError("job failed")
            for kind, fn in (("edits", edits), ("fails", fails)):
                job_dir = root / kind
                job_dir.mkdir()
                (job_dir / "job.json").write_text(json.dumps({"id": kind, "kind": kind, "timeout_seconds": 60,
                                                              "integrity": record}), encoding="utf-8")
                with mock.patch.dict(W.KINDS, {kind: fn}), mock.patch.object(W, "producer", lambda f: None), \
                        mock.patch.object(step03, "set_worker_environment", lambda: None):
                    report = W.run_job(job_dir / "job.json")
                self.assertEqual(report["status"], "failed", kind)
                self.assertIn("integrity_after", report)
                self.assertEqual(report["simulator_steps_exact"], 3 if kind == "edits" else 1)
                self.assertEqual(report["simulator_steps_attempted"], report["simulator_steps_exact"])
            self.assertEqual(json.loads((root / "edits" / "report.json").read_text())["integrity_after"]["changed"], [str(watched)])

    def test_missing_dependency_fails_closed(self):
        with mock.patch.object(U, "EXTERNAL_DEPENDENCIES", ("does_not_exist/nothing.py",)):
            with self.assertRaises(ValueError):
                U.dependency_closure()


class R5Terminal(unittest.TestCase):
    class Counter:
        steps, attempted = 1280, 1290

    def test_exactly_one_terminal_write_with_the_counts(self):
        written = []
        t = U.Terminal(lambda d, r: written.append(dict(r)), counter=self.Counter())
        report = {"label": "PASS"}
        self.assertTrue(t.finish("dir", report))
        self.assertFalse(t.finish("dir", report, label="INCOMPLETE"))
        self.assertEqual(len(written), 1)
        self.assertEqual((written[0]["label"], written[0]["transitions"], written[0]["simulator_steps_attempted"]),
                         ("PASS", 1280, 1290))
        self.assertIn("elapsed_seconds", written[0])
        self.assertEqual(report["label"], "PASS")                                      # the late fields did not land

    def test_watchdog_goes_through_the_owner(self):
        written, exits = [], []
        t = U.Terminal(lambda d, r: written.append(dict(r)), counter=self.Counter())
        step04._watchdog(Path("x"), {"approval": {}}, t, exit=exits.append)
        self.assertEqual(exits, [3])
        self.assertEqual((written[0]["label"], written[0]["watchdog"], written[0]["transitions"]), ("INCOMPLETE", True, 1280))
        t2 = U.Terminal(lambda d, r: written.append(dict(r)))
        t2.finish("x", {"label": "PASS"})                                              # the main thread finished first
        step04._watchdog(Path("x"), {"label": "PASS"}, t2, exit=exits.append)
        self.assertEqual(exits, [3])                                                   # the losing watchdog does not exit
        self.assertEqual([w["label"] for w in written], ["INCOMPLETE", "PASS"])       # no second write

    def test_a_failed_write_leaves_the_owner_open(self):
        written, calls = [], []

        def flaky(d, r):
            calls.append(1)
            if len(calls) == 1:
                raise OSError("disk full")
            written.append(dict(r))
        t = U.Terminal(flaky, counter=self.Counter())
        with self.assertRaises(OSError):
            t.finish("dir", {"label": "PASS"})
        self.assertFalse(t.done)
        self.assertIsNotNone(t.finish("dir", {"label": "INCOMPLETE"}, interrupted="KeyboardInterrupt"))
        self.assertEqual([w["label"] for w in written], ["INCOMPLETE"])
        self.assertIsNone(t.finish("dir", {"label": "PASS"}))

    def test_simultaneous_finish_writes_exactly_once(self):
        import threading
        import time
        written, barrier = [], threading.Barrier(8)

        def slow(d, r):
            time.sleep(0.05)
            written.append(r["who"])
        t = U.Terminal(slow)
        results = []

        def go(i):
            barrier.wait()
            results.append(t.finish("dir", {"who": i}))
        threads = [threading.Thread(target=go, args=(i,)) for i in range(8)]
        for th in threads:
            th.start()
        for th in threads:
            th.join()
        self.assertEqual(len(written), 1)
        self.assertEqual(sum(r is not None for r in results), 1)

    def test_interrupted_step04_records_its_counted_steps(self):
        import fixtures as F
        from types import SimpleNamespace
        import os

        def interrupted_collection(directory, budget, **kw):
            env = worlds.environment("stall3")
            env.reset(seed=1, options={"start": {"current": (0.5, 0.5)}})
            for _ in range(7):
                env.step(np.zeros(4, np.float32))
            raise KeyboardInterrupt
        with tempfile.TemporaryDirectory() as root:
            root = Path(root)
            manifest = {"files": {n: sha_file(step04.HERE / n) for n in sorted(step04.A3_REQUIRED_FILES)}}
            (root / "A3_MANIFEST.json").write_text(json.dumps(manifest), encoding="utf-8")
            (root / "A3_APPROVAL.json").write_text(json.dumps({"approval": step04.APPROVAL_WORDS + ", attempt 1",
                                                               "package_manifest_sha256": sha_file(root / "A3_MANIFEST.json"),
                                                               "attempt": 1}), encoding="utf-8")
            rerun = (SimpleNamespace(load_native=lambda: "n", runtime_receipt=lambda n: {"device_kinds": [step03.K.RECORDED_DEVICE_KIND]}),
                     SimpleNamespace(), None, SimpleNamespace(continuation_root=lambda: root))
            patches = [mock.patch.object(step04, "RUN", root / "step04_run"), mock.patch.object(step04, "MANIFEST", root / "A3_MANIFEST.json"),
                       mock.patch.object(step04, "APPROVAL", root / "A3_APPROVAL.json"),
                       mock.patch.object(step04, "a2_outcome", lambda approval: {"label": "PASS"}),
                       mock.patch.object(step04, "run_collection", interrupted_collection),
                       mock.patch.object(step03, "rerun1", lambda: rerun), mock.patch.object(step03, "set_worker_environment", lambda: None),
                       mock.patch.dict(os.environ, step03.WORKER_ENVIRONMENT)]
            for p in patches:
                p.start()
            try:
                with self.assertRaises(KeyboardInterrupt):
                    step04.execute(entropy=F.TEST_ENTROPY + 1, episodes=1280, provenance="smoke", test_seeds=F.TEST_SEEDS)
            finally:
                for p in patches:
                    p.stop()
            report = json.loads((root / "step04_run" / "attempt_1" / "report.json").read_text(encoding="utf-8"))
            self.assertEqual((report["label"], report["interrupted"]), ("INCOMPLETE", "KeyboardInterrupt"))
            self.assertEqual((report["transitions"], report["simulator_steps_attempted"]), (7, 7))
            self.assertIn("elapsed_seconds", report)


class R6Ledger(unittest.TestCase):
    def test_prior_attempts_known_and_unknown_work(self):
        with tempfile.TemporaryDirectory() as root:
            run = Path(root) / "stage1_run"
            a1, a2 = run / "attempt_1", run / "attempt_2"
            (a1 / "warmup_x").mkdir(parents=True)
            (a1 / "warmup_x" / "SUPERVISOR_RESULT.json").write_text(json.dumps({"elapsed_seconds": 10.5, "completed": True}))
            (a1 / "warmup_x" / "report.json").write_text(json.dumps({"simulator_steps_exact": 1280}))
            (a1 / "branch_x").mkdir()
            (a1 / "branch_x" / "job.json").write_text(json.dumps({"timeout_seconds": 300.0}))     # died without receipts
            (a2 / "warmup_x").mkdir(parents=True)
            (a2 / "warmup_x" / "SUPERVISOR_RESULT.json").write_text(json.dumps({"elapsed_seconds": 99.0}))
            ledger = U.ledger(run, current_attempt="attempt_2")
            self.assertEqual(ledger["prior_known_seconds"], 10.5)
            self.assertEqual(ledger["prior_known_simulator_steps"], 1280)
            self.assertEqual(ledger["prior_unknown_units"], ["attempt_1/branch_x"])
            self.assertEqual(ledger["prior_unknown_seconds_upper_bound"], 300.0)
            self.assertNotIn("attempt_2", ledger["attempts"])
            self.assertEqual(U.ledger(run)["prior_known_seconds"], 109.5)
            self.assertEqual(U.ledger(Path(root) / "missing")["attempts"], {})

    def test_reuse_nonreproduction_and_lost_counts(self):
        def unit(path, seconds=None, steps=None, job=None, completed=True, marker=None, worker_seconds=None):
            path.mkdir(parents=True)
            if seconds is not None:
                (path / "SUPERVISOR_RESULT.json").write_text(json.dumps({"elapsed_seconds": seconds, "completed": completed}))
            rep = {"status": "completed" if completed else "failed"}
            if steps is not None:
                rep["simulator_steps_exact"] = steps
            if worker_seconds is not None:
                rep["elapsed_seconds"] = worker_seconds
            (path / "report.json").write_text(json.dumps(rep))
            (path / "job.json").write_text(json.dumps(job or {}))
            if marker:
                (path / "NOT_REPRODUCED.json").write_text(json.dumps(marker))
        with tempfile.TemporaryDirectory() as root:
            run = Path(root) / "stage1_run"
            a1, a2 = run / "attempt_1", run / "attempt_2"
            unit(a1 / "warmup_a", 100.0, 1280, {"kind": "warmup"})                    # sealed, later reused (not copied)
            unit(a1 / "evaluate_s0", 50.0, None, {"kind": "evaluate", "transitions": 18000}, completed=False)  # lost count
            unit(a1 / "preflight", None, None, {"kind": "preflight"}, worker_seconds=3.0)
            unit(a2 / "warmup_b", 90.0, 1280, {"kind": "warmup"},
                 marker={"reference_dir": str(a1 / "warmup_b"), "reason": "the re-run does not reproduce"})
            unit(a2 / "branch_a", 80.0, None, {"kind": "branch"}, completed=False)       # training kind: unbounded
            prior = U.ledger(run, current_attempt="attempt_2")
            self.assertEqual((prior["prior_known_seconds"], prior["prior_known_simulator_steps"]), (153.0, 1280))
            self.assertEqual(prior["prior_unknown_step_units"], ["attempt_1/evaluate_s0", "attempt_1/preflight"])
            self.assertEqual(prior["prior_unknown_steps_upper_bound"], 18000)
            self.assertEqual(prior["attempts"]["attempt_1"]["preflight"]["seconds_source"], "worker")
            phys = U.physical(a2, prior)
            self.assertEqual(phys["current_attempt"]["known_seconds"], 170.0)
            self.assertEqual(phys["current_attempt"]["not_reproduced_units"], ["warmup_b"])
            self.assertEqual(phys["current_attempt"]["unbounded_step_units"], ["branch_a"])
            self.assertEqual(phys["cumulative_known_seconds"], 323.0)                 # every physical unit once
            self.assertEqual(phys["cumulative_known_simulator_steps"], 2560)
            self.assertEqual(prior["prior_unbounded_step_units"], ["attempt_1/preflight"])   # no allowance: unbounded, listed
            self.assertEqual(phys["cumulative_unbounded_step_units"], 2)


class R7Wrapper(unittest.TestCase):
    def test_wrapper_approval_is_bound_to_its_manifest(self):
        with tempfile.TemporaryDirectory() as root:
            root = Path(root)
            for name in X.MANIFEST_FILES:
                (root / name).write_text(f"content of {name}", encoding="utf-8")
            manifest = root / "EXPLORATORY_MANIFEST.json"
            manifest.write_text(json.dumps({"files": {n: sha_file(root / n) for n in X.MANIFEST_FILES}}), encoding="utf-8")
            approval = root / "EXPLORATORY_APPROVAL.json"
            approval.write_text(json.dumps({"approval": X.APPROVAL_WORDS}), encoding="utf-8")
            with self.assertRaises(ValueError):                                       # the A4-style approval lacks the binding
                X.check_manifest(approval, manifest)
            approval.write_text(json.dumps({"approval": X.APPROVAL_WORDS, "exploratory_manifest_sha256": sha_file(manifest)}),
                                encoding="utf-8")
            self.assertEqual(X.check_manifest(approval, manifest), sha_file(manifest))
            (root / "stage1_exploratory.py").write_text("edited", encoding="utf-8")   # a mutated wrapper invalidates it
            with self.assertRaises(ValueError):
                X.check_manifest(approval, manifest)

    def test_exploratory_output_cannot_unlock_a5(self):
        with tempfile.TemporaryDirectory() as root:
            run = Path(root) / "stage1_run"
            (run / "attempt_1").mkdir(parents=True)
            (run / "attempt_1" / "report.json").write_text(json.dumps(
                {"label": "PASS", "protocol_label": None, "exploratory_label": "PASS", "EXPLORATORY": {"disclosure": "x"},
                 "approval": {"manifest_sha256": stage2.A4_MANIFEST_SHA256}, "a5_precondition": {"ok": True}}), encoding="utf-8")
            with mock.patch.object(stage1, "RUN", run):
                with self.assertRaisesRegex(ValueError, "cannot unlock A5"):
                    stage2.stage1_outcome()
            clean = {"label": "PASS", "approval": {"manifest_sha256": stage2.A4_MANIFEST_SHA256}, "a5_precondition": {"ok": True}}
            (run / "attempt_1" / "report.json").write_text(json.dumps(clean), encoding="utf-8")
            with mock.patch.object(stage1, "RUN", run):                                # passes R7, stops later
                with self.assertRaisesRegex(ValueError, "does not name every unit directory"):
                    stage2.stage1_outcome()

    def test_exploratory_reads_only_the_pinned_step04_record(self):
        with tempfile.TemporaryDirectory() as root:
            run = Path(root) / "step04_run"
            (run / "attempt_1").mkdir(parents=True)
            (run / "attempt_1" / "report.json").write_text(json.dumps({"label": "PASS"}), encoding="utf-8")
            with mock.patch.object(stage1, "STEP04_RUN", run):
                with self.assertRaisesRegex(ValueError, "differs from the file pinned"):
                    X.a3_outcome_exploratory({"approval": stage1.PREDICTION_ACK})

    def test_exploratory_labels_agree_after_the_5_0_relabel(self):
        written = []
        report = {"label": "INCOMPLETE", "failed_unit": "evaluate_s0", "reasons": ["x"],
                  "approval": {"manifest_sha256": "m", "history": [{"failed_unit": "evaluate_s0", "counts_under_5_0": True,
                                                                    "manifest_sha256": "m"}]}}
        with mock.patch.object(X, "stage1_finish", lambda d, r: written.append(dict(r)) or r):
            X._finish_exploratory(Path("x"), report)
        self.assertEqual((written[0]["label"], written[0]["exploratory_label"]), ("INCONCLUSIVE", "INCONCLUSIVE"))
        self.assertIsNone(written[0]["protocol_label"])

    def test_v3_words_name_v3(self):
        self.assertIn("knee_edge_ood_v3", X.APPROVAL_WORDS)
        self.assertIn("stage1_exploratory.py", X.MANIFEST_FILES)


if __name__ == "__main__":
    unittest.main()

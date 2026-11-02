"""Step 0.3 (A2) tests on Windows: the gate math with known answers and boundaries, the saved-JSON readers, the
approval and attempt rules, and full execute() dry runs with stand-ins for rerun1's modules (a positive control that
reproduces, single-entry perturbations that must fail, a preflight failure, a crash mid-run, a refused second attempt
and an onset-only rerun). No model, checkpoint or bank archive is opened."""
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
import frozen_checks as K
import probes as P
import qpi
import step03
import step03_math as SM
from core import sha_file


class GateMath(unittest.TestCase):
    def test_fold_constant_and_r_lin_on_saved_values(self):
        self.assertAlmostEqual(SM.FOLD_CONSTANT, 0.22499999, places=8)
        for s in range(3):
            saved = step03.saved_cell(f"seed{s}_full")["ordinary"]
            r = SM.r_lin(saved["qpi"], saved["actions"])
            self.assertTrue(np.allclose(r, 2 * (saved["qpi"][:, 1] - saved["qpi"][:, 2]) + 0.22499999, atol=1e-8))
            self.assertEqual(int(SM.steep(saved["qpi"], saved["actions"], saved["high_load"]).sum()), 16)

    def test_fold_error_known_numbers(self):
        q = np.zeros((2, 7))
        q[:, SM._column(P.G05_POINTS, -1.0)] = [-10.0, -9.0]
        q[:, SM._column(P.G05_POINTS, -0.5)] = [-10.5, -9.0]
        self.assertTrue(np.allclose(SM.fold_error(q), [0.5 + SM.FOLD_CONSTANT, SM.FOLD_CONSTANT]))

    def test_reach_rows_on_known_quadratics(self):
        a = np.asarray(P.REACH_POINTS)
        for c1, c2, lam in ((3.0, -2.0, 0.04), (-1.0, -0.5, 0.03)):
            q = np.tile(-50.0 + c1 * a + c2 * a * a, (3, 1))
            fit = SM.reach_rows(q, lam=lam)
            s_edge = (c1 * (-0.5) + c2 * 0.25 - c1 * SM.EDGE_POINTS[1] - c2 * SM.EDGE_POINTS[1] ** 2) / SM.EDGE_SPAN
            self.assertTrue(np.allclose(fit["s_edge"], s_edge) and np.allclose(fit["q2"], c2))
            lp = lam * 0.751
            expected = np.clip((-0.7263 + 2 * lp * (s_edge + 1.2 * c2)) / (1 - 4 * lp * c2), -1, 1)
            self.assertTrue(np.allclose(fit["pi0"], expected))
        steep_q = np.tile(-50.0 - 40.0 * a - 2.0 * a * a, (4, 1))                # Q rising toward -1: a strong outward pull
        self.assertEqual(SM.forecast(SM.reach_rows(steep_q, lam=0.04)["pi0"]), 1.0)
        flat_q = np.tile(-50.0 - 0.0 * a - 2.0 * a * a, (4, 1))                  # no pull: pi0 near m, not deep
        self.assertEqual(SM.forecast(SM.reach_rows(flat_q, lam=0.04)["pi0"]), 0.0)
        convex = np.tile(-50.0 + 3.0 * a + 100.0 * a * a, (2, 1))
        self.assertTrue(np.all(SM.reach_rows(convex, lam=0.04)["pi0"] == -1.0))
        self.assertEqual(SM.forecast(np.asarray([SM.DEEP_EDGE, -0.5])), 0.5)

    def test_onset_ratio_asymmetric(self):
        r = SM.onset_ratio([1.0, 1.2, 0.1], [0.0, 0.0, 0.0], [0.5, 0.5, 0.5], [0.0, 0.0, 0.0], [1.5, 2.5, 0.6])
        self.assertAlmostEqual(r["median_abs_KC15_s05"], 0.5)                     # |KC| = 0.5, 0.7, 0.4
        self.assertAlmostEqual(r["median_rho15_s05"], 1.0)                        # rho = 1.0, 2.0, 0.1
        self.assertAlmostEqual(r["ratio"], 0.5)
        self.assertIsNone(SM.onset_ratio([1.0], [0.0], [0.5], [0.0], [0.4])["ratio"])

    def test_label_precedence_and_boundaries(self):
        good = {s: {"median_F": 0.1, "share_F_le_0.5": 0.9} for s in range(3)}
        cells = {c: 0.6 for c in K.CELLS}
        base = dict(known_answer_ok=True, caps_ok=True, g05_by_seed=good, probe_forecast_by_cell=cells,
                    visited_forecast_by_cell=cells, onset_by_seed={})
        L = lambda **kw: SM.label(**dict(base, **kw))
        self.assertEqual(L()[0], "PASS")
        self.assertEqual(L(known_answer_ok=False)[0], "INCOMPLETE")
        self.assertEqual(L(caps_ok=False)[0], "INCOMPLETE")
        self.assertEqual(L(g05_by_seed={**good, 1: {"median_F": 0.9, "share_F_le_0.5": 0.2}},
                           probe_forecast_by_cell={c: 0.1 for c in K.CELLS})[0], "INCONCLUSIVE")
        self.assertEqual(L(g05_by_seed={**good, 1: {"median_F": 0.9, "share_F_le_0.5": 0.25}})[0], "PASS (flagged)")
        self.assertEqual(L(g05_by_seed={**good, 1: {"median_F": 0.45, "share_F_le_0.5": 0.5}})[0], "PASS (flagged)")
        self.assertEqual(L(g05_by_seed={**good, 1: {"median_F": 0.5, "share_F_le_0.5": 0.5625}})[0], "PASS")
        self.assertEqual(L(probe_forecast_by_cell=dict(cells, seed2_hole=0.35))[0], "PASS")
        self.assertEqual(L(probe_forecast_by_cell=dict(cells, seed2_hole=0.3499))[0], "FAIL (reach)")
        self.assertEqual(L(visited_forecast_by_cell=dict(cells, seed0_full=0.25))[0], "PASS")
        self.assertEqual(L(visited_forecast_by_cell=dict(cells, seed0_full=0.2499))[0], "FAIL (reach)")
        self.assertEqual(L(onset_by_seed={0: {"ratio": 0.3}})[0], "PASS")
        self.assertEqual(L(onset_by_seed={0: {"ratio": 0.3001}, 1: None})[0], "PASS (flagged)")


class SavedInputs(unittest.TestCase):
    def test_every_cell_reads_and_is_pinned(self):
        for cell in K.CELLS:
            saved = step03.saved_cell(cell)
            self.assertEqual(saved["ordinary"]["actions"].shape, (32, 6, 4))
            self.assertTrue(np.all(saved["strong_current"]["high_load"]))
            v = saved["strong_current"]["visited"]
            obs = np.asarray(v["observations"], np.float32)
            self.assertEqual(int((np.asarray(v["high_load"], bool) & (obs[:, 10].astype(np.float64) >= 0.9)).sum()), 72)
            self.assertEqual(len(step03.saved_audit(cell)["source_row_ids"]), 768)
        pooled = {c: [0, 0] for c in ("full", "hole", "random")}
        for cell in K.CELLS:
            rows = step03.saved_audit(cell)["per_row"]
            w, n = step03.past_knee_count(rows["actor_actions"], rows["high_load"])
            pooled[cell[6:]][0] += w
            pooled[cell[6:]][1] += n
        self.assertEqual(pooled, step03.POOLED_PAST_KNEE)

    def test_pinned_rerun1_sources_and_a1_manifest(self):
        for name, digest in step03.RERUN1_SOURCES.items():
            self.assertEqual(sha_file(step03.RERUN1 / name), digest, name)
        self.assertEqual(sha_file(step03.A1_MANIFEST), step03.A1_MANIFEST_SHA256)

    def test_worker_environment_refuses_conflicts(self):
        with mock.patch.dict(os.environ, {"XLA_FLAGS": "--something_else"}):
            with mock.patch.object(step03.sys, "modules", {k: v for k, v in sys.modules.items() if k != "jax"}):
                with self.assertRaises(ValueError):
                    step03.set_worker_environment()


# ------------------------------------------------------------------------------------------------ dry runs

class StandIn:
    """A numpy stand-in for rerun1 FrozenPolicy."""

    def __init__(self, seed=3):
        self.forward = F.toy_actor()
        self.w = np.random.Generator(np.random.PCG64(seed)).normal(0, 0.5, (18, 2)).astype(np.float32)

    def act(self, obs):
        return self.forward(np.asarray(obs, np.float32))

    def q(self, obs, actions):
        x = np.concatenate([np.asarray(obs, np.float32), np.asarray(actions, np.float32)], axis=1)
        return (x @ self.w - 40.0 * np.asarray(actions, np.float32)[:, :1] ** 2).astype(np.float32)

    def width(self, obs, actions):
        return np.abs(np.asarray(actions, np.float32)[:, 0]) * np.float32(0.1)

    def assert_unchanged(self):
        pass


def synthetic_world():
    """A synthetic bank and saved records that the stand-in policy reproduces exactly."""
    anchors = [{"reset_seed": 3000 + i, "current": (0.9, 1.1), "prefix_actions": np.zeros((t, 4), np.float32)}
               for i, t in enumerate(np.tile([0, 25, 50, 75], 8))]
    states = np.stack([qpi.raw_anchor_state(a) for a in anchors])
    times = np.tile(np.asarray([0, 25, 50, 75], np.int64), 8)
    obs = np.concatenate([states, ((100 - times) / 100)[:, None]], axis=1).astype(np.float32)
    bank = {c: {"states": states, "times": times, "observations": obs, "bank_id": "synthetic"} for c in step03.CONTEXTS}
    policy = StandIn()
    saved = {}
    for context in step03.CONTEXTS:
        actions = np.concatenate([policy.act(obs)[:, None, :], P.known_answer_slots_1_to_5(32)], axis=1)
        rep = np.repeat(obs, 6, axis=0)
        q = policy.q(rep, actions.reshape(192, 4))
        width = policy.width(rep, actions.reshape(192, 4))
        values, _, _ = qpi.raw_values_from_states(states, times, actions, policy.act, "stall15_identity", clip_all=False)
        high = np.ones(32, bool) if context == "strong_current" else np.arange(32) % 3 != 0
        saved[context] = {"actions": actions, "q1": q[:, 0].astype(np.float64).reshape(32, 6),
                          "twin_min": q.min(axis=1).astype(np.float64).reshape(32, 6),
                          "adjusted": (q[:, 0].astype(np.float64) - 3.0 * width.astype(np.float64)).reshape(32, 6),
                          "qpi": values, "high_load": high, "bank_id": "synthetic", "sha256": "x", "file": "synthetic",
                          "visited": {"observations": np.tile(obs, (4, 1)), "actions": np.tile(policy.act(obs), (4, 1)),
                                      "high_load": np.ones(128, bool)} if context == "strong_current" else None}
    audit_rows = {"actor_actions": policy.act(np.tile(obs, (24, 1))), "high_load": np.arange(768) % 2 == 0}
    lambdas = np.asarray([0.037, 0.036, 0.0365])
    return bank, saved, policy, audit_rows, lambdas


class DryRun(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        root = Path(self.tmp.name)
        self.bank, self.saved, self.policy, self.audit_rows, self.lambdas = synthetic_world()
        manifest = {"files": {n: sha_file(step03.HERE / n) for n in sorted(step03.A2_REQUIRED_FILES)}}
        (root / "A2_MANIFEST.json").write_text(json.dumps(manifest), encoding="utf-8")
        self.manifest_sha = sha_file(root / "A2_MANIFEST.json")
        self.approve(root, 1)
        live = self.audit_rows
        pooled = {c: [3 * x for x in step03.past_knee_count(live["actor_actions"], live["high_load"])] for c in ("full", "hole", "random")}
        self.fail_cell = None
        self.audit_lambdas = self.lambdas
        policy, test = self.policy, self

        class Runtime:
            @staticmethod
            def load_native():
                return "native"

            @staticmethod
            def runtime_receipt(native):
                return {"device_kinds": [K.RECORDED_DEVICE_KIND]}

            @staticmethod
            def create(protocol, seed, condition):
                if test.fail_cell == (seed, condition):
                    raise RuntimeError("simulated crash in create")
                return SimpleNamespace(context=SimpleNamespace(data=SimpleNamespace(receipt={"protected_transition_rows_decoded": 0}))), "template"

            @staticmethod
            def tree_digest(tree):
                return "digest"

            @staticmethod
            def branch(adapter, warm, protocol, seed, arm, coefficient):
                return adapter, "template"

        warm_state = SimpleNamespace(aux=SimpleNamespace(posterior="p"),
                                     microstep=SimpleNamespace(agent_state=SimpleNamespace(actor=SimpleNamespace(params="a"))))
        matching = {"reference_digest": "digest", "anchor_actor_digest": "digest", "distance_coefficient": 1.0}

        class Worker:
            @staticmethod
            def validate_protocol(protocol):
                return None

            @staticmethod
            def assemble_exports(protocol, root_, sha):
                exports = {f"{c}__{s}": {"matching": matching} for c in K.CELLS for s in ("warmup__10000", "native__20000")}
                return exports, {}

            @staticmethod
            def restore_snapshot(runtime, adapter, template, export):
                return warm_state

            @staticmethod
            def FrozenPolicy(runtime, adapter, state):
                return policy

        class Audit:
            @staticmethod
            def audit(adapter, state, ids):
                return {"batches": [{"lambda_from_unpenalized_Q1": v} for v in test.audit_lambdas], "per_row": live}

        rerun = (Runtime, Worker, Audit, SimpleNamespace(continuation_root=lambda: root))
        self.patches = [
            mock.patch.object(step03, "RUN", root / "step03_run"), mock.patch.object(step03, "MANIFEST", root / "A2_MANIFEST.json"),
            mock.patch.object(step03, "APPROVAL", root / "A2_APPROVAL.json"), mock.patch.object(step03, "rerun1", lambda: rerun),
            mock.patch.object(step03, "set_worker_environment", lambda: None),
            mock.patch.dict(os.environ, step03.WORKER_ENVIRONMENT),
            mock.patch.object(step03, "load_banks", lambda: self.bank),
            mock.patch.object(step03, "saved_cell", lambda cell: self.saved),
            mock.patch.object(step03, "saved_audit", lambda cell: {"source_row_ids": list(range(768)), "per_row": live}),
            mock.patch.object(step03, "saved_lambdas", lambda cell: {"values": self.lambdas}),
            mock.patch.object(step03, "POOLED_PAST_KNEE", pooled),
            mock.patch.object(step03.SM, "steep", lambda qpi_, actions, high: np.asarray(high, bool)),
            mock.patch.object(step03.SM, "fold_error", lambda q1: np.zeros(len(q1))),
            mock.patch.object(step03.SM, "forecast", lambda pi0: 0.9),
        ]
        for p in self.patches:
            p.start()
        self.root = root

    def tearDown(self):
        for p in self.patches:
            p.stop()
        self.tmp.cleanup()

    def approve(self, root, attempt):
        (root / "A2_APPROVAL.json").write_text(json.dumps({"approval": step03.APPROVAL_WORDS + ", attempt %d" % attempt,
                                                            "package_manifest_sha256": self.manifest_sha, "attempt": attempt}),
                                               encoding="utf-8")

    def attempt_dir(self, k):
        return self.root / "step03_run" / f"attempt_{k}"

    def test_positive_control_passes_and_writes_everything(self):
        report = step03.execute()
        self.assertTrue(report["label"].startswith("PASS"), report.get("reasons"))
        d = self.attempt_dir(1)
        for name in ["ATTEMPT.json", "gated.json", "report.json"] + [f"cell_{c}.json" for c in K.CELLS]:
            self.assertTrue((d / name).is_file(), name)
        self.assertTrue(all(v is not None and v["rows"] > 0 for v in report["onset"].values()))
        cell = json.loads((d / "cell_seed0_full.json").read_text())
        self.assertTrue(cell["known_answer"]["ordinary"]["ok"] and cell["known_answer"]["identity_T15"]["ok"])
        with self.assertRaises(ValueError):
            step03.execute()                                  # the approval still names attempt 1: refused

    def test_single_perturbations_each_fail(self):
        for field, change in (("qpi", lambda s: dict(s, qpi=s["qpi"] * (1 + 1e-8))),
                              ("q1", lambda s: dict(s, q1=s["q1"] + 1e-3))):
            original = {c: dict(v) for c, v in self.saved.items()}
            self.saved["ordinary"] = change(self.saved["ordinary"])
            report = step03.execute()
            self.assertEqual(report["label"], "INCOMPLETE", field)
            self.saved.update(original)
            for p in sorted((self.root / "step03_run").iterdir()):
                for f in p.iterdir():
                    f.unlink()
                p.rmdir()
        self.audit_lambdas = self.lambdas * (1 + 1e-5)       # lambda only
        self.assertEqual(step03.execute()["label"], "INCOMPLETE")

    def test_preflight_failure_is_exempt_and_computes_nothing(self):
        bad = mock.patch.object(step03, "checked_receipt", lambda runtime, holder: (lambda _: {"device_kinds": ["CPU"]}))
        with bad:
            report = step03.execute()
        self.assertEqual(report["label"], "INCOMPLETE")
        self.assertFalse(report["counts_under_5_0"])
        self.assertEqual(report["ledger"]["reproduction_values"], 0)
        self.assertFalse(any(self.attempt_dir(1).glob("cell_*.json")))

    def test_crash_mid_run_keeps_evidence(self):
        self.fail_cell = (0, "random")                       # the third cell
        report = step03.execute()
        self.assertEqual(report["label"], "INCOMPLETE")
        self.assertTrue(report["counts_under_5_0"])
        d = self.attempt_dir(1)
        self.assertTrue((d / "cell_seed0_full.json").is_file() and (d / "cell_seed0_hole.json").is_file())
        self.assertTrue((d / "report.json").is_file())
        self.approve(self.root, 2)                           # same code again, second failure: INCONCLUSIVE (5.0)
        self.assertEqual(step03.execute()["label"], "INCONCLUSIVE")

    def test_onset_only_rerun_after_an_onset_failure(self):
        with mock.patch.object(step03, "onset", side_effect=RuntimeError("simulated onset failure")):
            first = step03.execute()
        self.assertEqual(first["label"], "INCOMPLETE")
        self.assertTrue(first["gated_label"].startswith("PASS"))
        self.approve(self.root, 2)
        calls = []
        real_build = step03.build_policy
        with mock.patch.object(step03, "build_policy", lambda *a: calls.append(a[0]) or real_build(*a)):
            second = step03.execute()
        self.assertEqual(second["units"], ["onset"])
        self.assertTrue(second["label"].startswith("PASS"))
        self.assertEqual(calls, ["seed0_full", "seed1_full", "seed2_full"])   # only the full cells, for the onset

    def test_mismatch_then_fixed_evaluator_reruns_the_gated_unit(self):
        original = dict(self.saved["ordinary"])
        self.saved["ordinary"] = dict(original, q1=original["q1"] + 1e-3)
        first = step03.execute()
        self.assertEqual((first["label"], first["failed_unit"], first["counts_under_5_0"]), ("INCOMPLETE", "gated", True))
        self.saved["ordinary"] = original                    # the evaluator is "fixed"
        self.approve(self.root, 2)
        second = step03.execute()
        self.assertEqual(second["units"], ["gated", "onset"])
        self.assertTrue(second["label"].startswith("PASS"))

    def test_onset_failing_twice_under_unchanged_code_is_inconclusive(self):
        with mock.patch.object(step03, "onset", side_effect=RuntimeError("simulated onset failure")):
            first = step03.execute()
            self.assertEqual((first["failed_unit"], first["counts_under_5_0"]), ("onset", True))
            self.approve(self.root, 2)
            second = step03.execute()
        self.assertEqual(second["units"], ["onset"])
        self.assertEqual(second["label"], "INCONCLUSIVE")

    def test_a_different_unit_failing_is_not_a_second_failure(self):
        self.fail_cell = (1, "hole")
        self.assertEqual(step03.execute()["failed_unit"], "gated")
        self.fail_cell = None
        self.approve(self.root, 2)
        with mock.patch.object(step03, "onset", side_effect=RuntimeError("simulated onset failure")):
            second = step03.execute()
        self.assertEqual((second["label"], second["failed_unit"]), ("INCOMPLETE", "onset"))

    def test_second_failure_found_across_an_exempt_preflight_attempt(self):
        self.fail_cell = (1, "hole")
        step03.execute()
        self.approve(self.root, 2)
        with mock.patch.object(step03, "checked_receipt", lambda runtime, holder: (lambda _: {"device_kinds": ["CPU"]})):
            self.assertFalse(step03.execute()["counts_under_5_0"])
        self.approve(self.root, 3)
        self.assertEqual(step03.execute()["label"], "INCONCLUSIVE")

    def test_sealed_gated_found_across_an_onset_only_attempt(self):
        with mock.patch.object(step03, "onset", side_effect=RuntimeError("simulated onset failure")):
            step03.execute()
        self.approve(self.root, 2)
        with mock.patch.object(step03, "checked_receipt", lambda runtime, holder: (lambda _: {"device_kinds": ["CPU"]})):
            self.assertEqual(step03.execute()["units"], ["onset"])
        self.approve(self.root, 3)
        third = step03.execute()
        self.assertEqual(third["units"], ["onset"])
        self.assertEqual(third["approval"]["sealed_gated_attempt"], "attempt_1")
        self.assertTrue(third["label"].startswith("PASS"))

    def test_no_reattempt_after_a_final_label(self):
        step03.execute()
        self.approve(self.root, 2)
        with self.assertRaises(ValueError):
            step03.execute()


if __name__ == "__main__":
    unittest.main()

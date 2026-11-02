"""Disposable smoke fixtures only: no physical simulator, JAX or fitted models."""
from collections import namedtuple
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import pickle
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

spec = importlib.util.spec_from_file_location("path_a_smoke_tested", Path(__file__).with_name("smoke.py"))
S = importlib.util.module_from_spec(spec)
spec.loader.exec_module(S)
Actor = namedtuple("Actor", "params")
Agent = namedtuple("Agent", "actor")
Micro = namedtuple("Micro", "rng agent_state reference batch")
Aux = namedtuple("Aux", "posterior")
State = namedtuple("State", "microstep aux aux_rng refresh_at critic_budget critic actor auxiliary calibrations")


def plain(value):
    if isinstance(value, dict):
        return {key: plain(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [plain(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def digest(value):
    return hashlib.sha256(json.dumps(plain(value), sort_keys=True, separators=(",", ":")).encode()).hexdigest()


class FakeRuntime:
    def __init__(self):
        self.calls = []
        self.rejected_calibration = False
        self.fail_native_dispatch = False
        self.corrupt_restored_actor_step = False
        self.saw_step8 = {}
        self.restore_states = []

    host = staticmethod(plain)
    tree_digest = staticmethod(digest)
    config = staticmethod(lambda value: dict(vars(value)))

    def validate(self, adapter, state):
        S.require(int(state.critic_budget) == adapter.budget and list(state.refresh_at) == adapter.schedule,
                  "fake schedule mismatch")
        S.require(state.actor == state.critic // 2 and state.auxiliary == state.critic, "fake counters mismatch")
        S.require(state.calibrations == sum(n <= state.critic for n in adapter.schedule), "fake refresh mismatch")
        return {"counts": {"critic": state.critic, "actor": state.actor, "auxiliary": state.auxiliary,
                           "refresh_accepted": state.calibrations, "phase": state.critic % 2,
                           "reference_ready": state.aux.posterior["ready"]}}

    def make_adapter(self, context, state, arm, kappa, anchor, schedule, budget):
        return SimpleNamespace(context=context, arm=arm, kappa=kappa, schedule=list(schedule), budget=budget)

    def restore_exact(self, adapter, template, raw, diagnostic=None):
        # Bytes are created by this test process only; no external pickle is read.
        value = pickle.loads(raw)
        self.validate(adapter, value)
        self.restore_states.append(copy.deepcopy(value))
        if diagnostic is not None:
            diagnostic["fake_full_state_exact"] = digest(value) == digest(template)
        return value

    def step(self, adapter, state, progress):
        c = state.critic + 1
        self.calls.append((adapter.arm, c, state.critic % 2, tuple(state.microstep.batch)))
        attempt = {"critic": 1, "actor": state.critic % 2, "auxiliary": 0, "refresh": 0}
        completed = dict.fromkeys(attempt, 0)
        def notify(stage):
            progress({"stage": stage, "event": "fake_dispatch", "attempted": dict(attempt), "completed": dict(completed)})
        notify("native_microstep")
        if self.fail_native_dispatch:
            raise RuntimeError("fake uncertain dispatch")
        completed["critic"], completed["actor"] = 1, attempt["actor"]
        notify("native_microstep")
        attempt["auxiliary"] = 1; notify("auxiliary_fit")
        completed["auxiliary"] = 1; notify("auxiliary_fit")
        due = c in adapter.schedule
        if due:
            attempt["refresh"] = 1; notify("calibration_refresh")
            completed["refresh"] = 1; notify("calibration_refresh")
            if self.rejected_calibration:
                raise ValueError("genuine calibration rejected")
        params = dict(state.microstep.agent_state.actor.params)
        batch = tuple([c, c + 1]) if state.critic % 2 == 0 else state.microstep.batch
        if state.critic % 2:
            params["weight"] += sum(batch) * (1 if adapter.arm == "native" else 2)
        if c == 8:
            self.saw_step8[adapter.arm] = self.saw_step8.get(adapter.arm, 0) + 1
            if self.corrupt_restored_actor_step and self.saw_step8[adapter.arm] == 2:
                params["weight"] += 1
        ref = {"ready": True, "threshold": state.calibrations + 1} if due else state.aux.posterior
        micro = state.microstep._replace(rng=state.microstep.rng + 1,
                 agent_state=Agent(Actor(params)), reference=ref, batch=batch)
        out = state._replace(microstep=micro, aux=Aux(ref), aux_rng=state.aux_rng + 1,
                  critic=c, actor=c // 2, auxiliary=c, calibrations=state.calibrations + int(due))
        return out, self.validate(adapter, out)


def services():
    runtime = FakeRuntime()
    native = SimpleNamespace(jnp=np, jax=SimpleNamespace(random=SimpleNamespace(fold_in=lambda key, seed: int(key) * 31 + int(seed))))
    metadata = {"dataset_sha256": "a" * 64, "geometry_sha256": "b" * 64,
                "calibration_row_ids_sha256": "c" * 64, "training_rows": 53669}
    def initialize(rt, nat, protocol, seed, condition, data, training_review):
        S.require(training_review["training_authorized"] and training_review["protocol_sha256"] == digest(protocol), "wrong review")
        S.require(seed == 0 and condition == "bottleneck", "only declared smoke input")
        context = SimpleNamespace(native=native, args=SimpleNamespace(td3_alpha=2.5, batch_size=256, bc_coefficient=1.),
                                  runtime={"fake": True}, normalization={"source": "retained fitting", "rows": 53669})
        ref = {"ready": False, "threshold": 0}
        state = State(Micro(1, Agent(Actor({"weight": 0})), ref, (0, 0)), Aux(ref), 2,
                      np.array([10000], np.int32), np.int32(10000), 0, 0, 0, 0)
        return rt.make_adapter(context, state, "native", 0., {}, [10000], 10000), state
    I = SimpleNamespace(digest=digest, initialize=initialize)
    binding = SimpleNamespace(I=I, prepare=lambda native, condition, protocol: SimpleNamespace(metadata=copy.deepcopy(metadata)))
    value = SimpleNamespace(runtime=runtime, native=native, binding=binding,
                            serialization=SimpleNamespace(to_bytes=lambda x: pickle.dumps(x, protocol=4)),
                            expected_data_receipt={"conditions": {"bottleneck": {"metadata": metadata}}})
    protocol = {"seeds": [{"branch_native_seed": 123, "branch_aux_seed": 456}]}
    review = {"training_authorized": True, "scope": "disposable_path_a_smoke", "protocol_sha256": digest(protocol),
              "dataset_sha256": metadata["dataset_sha256"], "geometry_sha256": metadata["geometry_sha256"]}
    return value, protocol, review


class SmokeFixtures(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.output = Path(self.temp.name)
        self.s, self.p, self.review = services()
        self.events = []

    def run_smoke(self, deadline=lambda: None, persist=None):
        return S.run_learning_smoke(self.p, self.s, self.review, deadline, self.output, persist or self.events.append)

    def test_genuine_short_schedules_paired_actor_restore_and_exact_counts(self):
        policies, receipt = self.run_smoke()
        self.assertEqual(receipt["status"], "passed_learning_smoke")
        self.assertEqual(receipt["counts"]["attempted"], dict(critic=14, actor=8, auxiliary=14, calibrations=3))
        self.assertEqual(receipt["counts"]["attempted"], receipt["counts"]["completed"])
        self.assertEqual(set(policies), {"native", "bca"})
        self.assertEqual(self.s.runtime.saw_step8, {"native": 2, "bca": 2})
        for arm in ("native", "bca"):
            state = policies[arm]["state"]
            self.assertEqual((state.critic, state.actor, state.auxiliary, state.calibrations), (8, 4, 8, 2))
            before = json.loads((self.output / (arm + "_held_batch_7.json")).read_text())
            self.assertEqual(before["cumulative_counts"]["phase"], 1)
            self.assertTrue((self.output / (arm + "_actor_original_8.msgpack")).exists())
            self.assertTrue((self.output / (arm + "_actor_restored_8.msgpack")).exists())
        self.assertNotEqual(policies["native"]["state"].microstep.agent_state.actor.params,
                            policies["bca"]["state"].microstep.agent_state.actor.params)
        self.assertEqual(sum(event.get("dispatch", {}).get("stage") == "calibration_refresh" for event in self.events), 6)
        self.assertTrue(all(call[1] <= 8 for call in self.s.runtime.calls))
        self.assertFalse(receipt["production_checkpoint"])

    def test_pending_dispatch_is_persisted_and_unknown_not_zero(self):
        self.s.runtime.fail_native_dispatch = True
        policies, receipt = self.run_smoke()
        self.assertEqual(policies, {})
        self.assertEqual(receipt["counts"]["attempted"]["critic"], 1)
        self.assertEqual(receipt["counts"]["completed"]["critic"], 0)
        self.assertFalse(receipt["execution_counts_known"])
        self.assertTrue(any(e["counts"]["attempted"]["critic"] == 1 and e["counts"]["completed"]["critic"] == 0
                            for e in self.events))

    def test_real_calibration_failure_stops_without_fabricated_reference(self):
        self.s.runtime.rejected_calibration = True
        policies, receipt = self.run_smoke()
        self.assertEqual(policies, {})
        self.assertEqual(receipt["status"], "failed_learning_smoke")
        self.assertIn("calibration rejected", receipt["error"])
        self.assertEqual(receipt["counts"]["completed"], dict(critic=4, actor=2, auxiliary=4, calibrations=1))
        self.assertEqual(len(self.s.runtime.calls), 4)

    def test_actor_restore_mismatch_is_failure_not_tolerance_relaxation(self):
        self.s.runtime.corrupt_restored_actor_step = True
        policies, receipt = self.run_smoke()
        self.assertEqual(policies, {})
        self.assertIn("restored actor next step differs", receipt["error"])
        self.assertEqual(receipt["counts"]["completed"], dict(critic=9, actor=5, auxiliary=9, calibrations=2))
        self.assertFalse((self.output / "bca_start_4.msgpack").exists())

    def test_wrong_scope_refuses_before_any_output_or_initialization(self):
        self.review["scope"] = "production"
        with self.assertRaisesRegex(ValueError, "smoke-scoped"):
            self.run_smoke()
        self.assertEqual(list(self.output.iterdir()), [])
        self.assertEqual(self.s.runtime.calls, [])

    def test_deadline_fails_before_first_model_update(self):
        def expired():
            raise TimeoutError("smoke wall cap")
        _, receipt = self.run_smoke(deadline=expired)
        self.assertEqual(receipt["counts"]["attempted"], dict.fromkeys(S.KEYS, 0))
        self.assertIn("wall cap", receipt["error"])

    def test_progress_persistence_failure_blocks_dispatch_and_stays_failed(self):
        def fail(event):
            if event.get("dispatch", {}).get("stage") == "native_microstep":
                raise OSError("progress disk full")
            self.events.append(event)
        policies, receipt = self.run_smoke(persist=fail)
        self.assertEqual(policies, {})
        self.assertEqual(receipt["status"], "failed_learning_smoke")
        self.assertEqual(receipt["counts"]["attempted"]["critic"], 1)
        self.assertEqual(receipt["counts"]["completed"]["critic"], 0)

    def test_existing_output_refuses_retry(self):
        (self.output / "old_evidence.json").write_text("{}")
        with self.assertRaisesRegex(ValueError, "new empty"):
            self.run_smoke()
        self.assertEqual(self.s.runtime.calls, [])

    def test_final_receipt_write_failure_propagates(self):
        original = S._save
        def fail(path, value):
            if Path(path).name == "RESULT.json":
                raise OSError("final receipt disk full")
            return original(path, value)
        with patch.object(S, "_save", fail):
            with self.assertRaisesRegex(OSError, "receipt disk full"):
                self.run_smoke()
        self.assertFalse((self.output / "RESULT.json").exists())

    def test_ledger_refuses_extra_update_or_double_count(self):
        ledger = S.Ledger()
        ledger.attempted["critic"] = 14
        ledger.begin()
        with self.assertRaisesRegex(ValueError, "cap exceeded"):
            ledger.dispatch({"attempted": dict(critic=1, actor=0, auxiliary=0, refresh=0),
                             "completed": dict(critic=0, actor=0, auxiliary=0, refresh=0)})


if __name__ == "__main__":
    unittest.main()

"""Synthetic tuple and fake-task fixtures only: no JAX, canonical dynamics or model initialization."""
from collections import namedtuple
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import Mock
import numpy as np

HERE = Path(__file__).resolve().parent
def load(name):
    spec = importlib.util.spec_from_file_location("two_scenario_fixture_" + name, HERE / (name + ".py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

I, E, D = load("integration"), load("evaluation"), load("data")
Native = SimpleNamespace(Transition=namedtuple("Transition", "obs action reward next_obs done"))


def source_fixture():
    lengths = np.ones(1280, np.int64)
    lengths[:4] = 100
    episodes = np.repeat(np.arange(1280), lengths).astype(np.int64)
    times = np.concatenate([np.arange(length) for length in lengths]).astype(np.int64)
    n = len(times)
    states = np.zeros((n, 13), np.float64)
    states[:, 0] = np.arange(n)
    next_states = states.copy()
    next_states[:, 0] += 1
    obs = np.zeros((n, 14), np.float32)
    obs[:, :13], obs[:, -1] = states.astype(np.float32), ((100-times)/100).astype(np.float32)
    nxt = obs.copy()
    nxt[:, :13] = next_states.astype(np.float32)
    nxt[:, -1] = ((99-times)/100).astype(np.float32)
    return dict(states=states, next_states=next_states, observations=obs, next_observations=nxt, actions=np.zeros((n, 4), np.float32),
                rewards=np.arange(n, dtype=np.float64), terminals=times == np.repeat(lengths-1, lengths),
                episode_ids=episodes, row_ids=np.arange(n, dtype=np.int64), times=times,
                roles=np.repeat(np.r_[np.zeros(768, np.uint8), np.ones(256, np.uint8), np.full(256, 2, np.uint8)], lengths),
                fragment_ids=episodes.copy(), families=(episodes % 4).astype(np.int8))


def geometry_at(center=1e6, radius=.1):
    region = {"center": [float(center)] + [0.] * 9, "radius": float(radius)}
    return {"danger": region, "connector": region}


def prepared(records=None, mask=None, geometry=None):
    records = source_fixture() if records is None else records
    geometry = geometry_at() if geometry is None else geometry
    mask = D.retained_mask(records, geometry, "honeypot") if mask is None else mask
    return I.prepare_learner_data(Native, records, mask, geometry=geometry, condition="honeypot", calibration_seed=121312521,
                                 dataset_sha256="a"*64, geometry_sha256="b"*64)


def protocol():
    return dict(gamma=.98, horizon=100, kappa=3., warmup_updates=10000, branch_updates=10000,
                seeds=[dict(native_seed=10+i, aux_seed=20+i, branch_native_seed=30+i, branch_aux_seed=40+i) for i in range(3)])


class FakeTask:
    """This is not ThrusterChain: reward equals the first action; terminal flag is explicit."""
    def __init__(self, end_at=None):
        self.end_at = end_at
        self.calls = []

    def observe(self, state, t):
        return np.r_[state, (100-t)/100].astype(np.float32)

    def step(self, state, t, action):
        self.calls.append((state.copy(), t, action.copy()))
        nxt = state.copy()
        nxt[0] += 1
        return nxt, float(action[0]), bool(t+1 == 100 or self.end_at == t+1), {"fixture": True}


class IntegrationTests(unittest.TestCase):
    def test_exact_counts_and_native_bc_settings(self):
        b = I.validate_protocol(protocol())
        self.assertEqual((b["final_policies"], b["training_jobs"], b["critic_updates"], b["actor_updates"], b["calibrations"]),
                         (12, 18, 180000, 90000, 18))
        self.assertEqual((b["td3_alpha"], b["bc_coefficient"]), (2.5, 1.))
        bad = protocol()
        bad["kappa"] = 63
        with self.assertRaises(ValueError): I.validate_protocol(bad)

    def test_mask_preserves_original_next_tuple_and_true_terminals(self):
        records = source_fixture()
        records["terminals"][0] = False
        data = prepared(records, geometry=geometry_at(2., .5))
        self.assertEqual(data.source_row_ids[:3].tolist(), [0, 3, 4])
        np.testing.assert_array_equal(data.train.next_obs[0], records["next_observations"][0])
        self.assertFalse(np.array_equal(data.train.next_obs[0], data.train.obs[1]))
        self.assertEqual(data.train.done[0], 0.)
        self.assertEqual(data.train.done[data.source_row_ids.tolist().index(99)], 1.)
        self.assertFalse(data.metadata["next_action_used"])

    def test_only_retained_fit_rows_normalize_and_calibrate_separately(self):
        records = source_fixture()
        data = prepared(records, geometry=geometry_at(7.5, .1))
        self.assertNotIn(7, data.normalization_source_row_ids)
        self.assertLess(float(data.normalization_fixture.obs[:, 0].max()), 1164)
        self.assertEqual(len(set(data.heldout_ids["episode_ids"].tolist())), 128)
        self.assertFalse(set(data.source_row_ids.tolist()) & set(data.heldout_ids["row_ids"].tolist()))
        self.assertFalse(data.train.obs.flags.writeable)

    def test_insufficient_calibration_is_not_replaced_with_fit_rows(self):
        records = source_fixture()
        with self.assertRaisesRegex(ValueError, "128 eligible"):
            prepared(records, geometry=geometry_at(1500., 208.8))

    def test_arbitrary_retention_override_rejected(self):
        records = source_fixture()
        mask = np.ones(len(records["actions"]), bool)
        mask[7] = False
        with self.assertRaisesRegex(ValueError, "Retention differs"):
            prepared(records, mask)

    def test_split_leakage_duplicate_identity_and_time_reset_rejected(self):
        for kind in ("split", "duplicate", "time"):
            records = source_fixture()
            if kind == "split": records["episode_ids"][-1] = 0
            if kind == "duplicate": records["row_ids"][1] = 0
            if kind == "time": records["observations"][2, -1] = 1.
            with self.subTest(kind=kind), self.assertRaises(ValueError):
                prepared(records)

    def test_setup_cannot_initialize_models(self):
        runtime = Mock()
        with self.assertRaisesRegex(ValueError, "later reviewed"):
            I.initialize(runtime, Native, protocol(), 0, "honeypot", prepared())
        runtime.runtime_receipt.assert_not_called()
        runtime._initialize_data.assert_not_called()

    def test_later_initializer_uses_original_runtime_and_correct_receipt_description(self):
        data, p = prepared(), protocol()
        context = SimpleNamespace(args=SimpleNamespace(td3_alpha=2.5, gamma=.98, batch_size=256),
                                  normalization={"source": "old description"})
        adapter = SimpleNamespace(context=context, kappa=0.)
        runtime = SimpleNamespace(runtime_receipt=lambda native: {"fixture": True},
                                  _initialize_data=Mock(return_value=(adapter, "state")))
        review = dict(training_authorized=True, protocol_sha256=I.digest(p), dataset_sha256="a"*64, geometry_sha256="b"*64)
        actual, state = I.initialize(runtime, Native, p, 0, "honeypot", data, training_review=review)
        self.assertIs(actual, adapter)
        self.assertEqual(state, "state")
        self.assertEqual(context.normalization["source"], data.metadata["normalization"])
        self.assertIs(runtime._initialize_data.call_args.args[4], data)

    def test_paired_fork_is_the_same_full_state_and_rejects_divergence(self):
        args = SimpleNamespace(td3_alpha=2.5)
        calls = []
        def branch(a, state, p, seed, arm, coefficient):
            calls.append((state, seed, arm, coefficient))
            return SimpleNamespace(kappa=0. if arm == "native" else 3., context=SimpleNamespace(args=args)), {"fork": 1}
        runtime = SimpleNamespace(tree_digest=lambda value: json.dumps(value, sort_keys=True), branch=branch, config=vars)
        result = I.fork_pair(runtime, None, {"warm": 1}, protocol(), 0)
        self.assertEqual(result["native"][1], result["bca"][1])
        self.assertEqual([c[2] for c in calls], ["native", "bca"])
        self.assertIs(calls[0][0], calls[1][0])

    def test_restore_rejects_historical_wrong_task_before_decoder(self):
        runtime = Mock()
        with self.assertRaisesRegex(ValueError, "another task"):
            I.restore_bound(runtime, None, None, b"fixture", dict(dataset_sha256="c"*64), prepared=prepared(),
                            expected_sha256=I.hashlib.sha256(b"fixture").hexdigest(), protocol=protocol(), seed_index=0)
        runtime.restore_exact.assert_not_called()

    def test_restore_binds_seed_protocol_and_normalization_before_decoder(self):
        p, data = protocol(), prepared()
        adapter = SimpleNamespace(context=SimpleNamespace(normalization={"mean": [0.], "std": [1.]}))
        raw = b"fixture"
        runtime = SimpleNamespace(host=lambda value: value, restore_exact=Mock(return_value="restored"),
                                  tree_digest=lambda value: "state_fixture")
        receipt = dict(dataset_sha256="a"*64, geometry_sha256="b"*64, condition="honeypot", seed_index=0,
                       protocol_sha256=I.digest(p), normalization_sha256=I.digest(adapter.context.normalization),
                       state_digest="state_fixture")
        value, _ = I.restore_bound(runtime, adapter, "template", raw, receipt, prepared=data,
            expected_sha256=I.hashlib.sha256(raw).hexdigest(), protocol=p, seed_index=0)
        self.assertEqual(value, "restored")
        for field, bad in (("seed_index", 1), ("protocol_sha256", "wrong"), ("normalization_sha256", "wrong")):
            runtime.restore_exact.reset_mock()
            with self.subTest(field=field), self.assertRaises(ValueError):
                I.restore_bound(runtime, adapter, "template", raw, dict(receipt, **{field: bad}), prepared=data,
                    expected_sha256=I.hashlib.sha256(raw).hexdigest(), protocol=p, seed_index=0)
            runtime.restore_exact.assert_not_called()


class EvaluationTests(unittest.TestCase):
    def test_early_terminal_stops_without_bootstrap(self):
        task = FakeTask(end_at=26)
        forward = lambda obs: np.ones((len(obs), 4), np.float32)
        result = E.rollout(task, np.zeros((1, 13), np.float64), np.array([25]), forward, max_transitions=75)
        self.assertEqual(result["returns"].tolist(), [1.])
        self.assertEqual(result["final_times"].tolist(), [26])
        self.assertEqual(len(task.calls), 1)

    def test_exact_time_forced_action_and_discounted_own_continuation(self):
        task = FakeTask()
        forward = lambda obs: np.full((len(obs), 4), .5, np.float32)
        first = np.zeros((1, 2, 4), np.float32)
        first[0, 0, 0], first[0, 1, 0] = 1., -1.
        q, counts = E.own_policy_q(task, np.zeros((1, 13), np.float64), np.array([98]), first, forward, max_transitions=4)
        np.testing.assert_allclose(q, [[1+.98*.5, -1+.98*.5]], rtol=0, atol=1e-15)
        self.assertEqual([row[1] for row in task.calls], [98, 98, 99, 99])
        self.assertEqual(counts["transitions_completed"], 4)

    def test_cap_and_partial_failure_preserve_attempted_counts(self):
        forward = lambda obs: np.zeros((len(obs), 4), np.float32)
        with self.assertRaisesRegex(ValueError, "cap exhausted") as caught:
            E.rollout(FakeTask(), np.zeros((1, 13), np.float64), np.array([98]), forward, max_transitions=1)
        self.assertEqual(caught.exception.evaluation_counts["transitions_completed"], 1)
        task = FakeTask()
        task.step = Mock(side_effect=RuntimeError("fixture dynamics failure"))
        with self.assertRaises(RuntimeError) as caught:
            E.rollout(task, np.zeros((1, 13), np.float64), np.array([98]), forward, max_transitions=2)
        self.assertEqual(caught.exception.evaluation_counts["transitions_attempted"], 1)
        self.assertEqual(caught.exception.evaluation_counts["transitions_completed"], 0)

    def test_true_horizon_terminal_is_required(self):
        task = FakeTask()
        task.step = lambda x, t, a: (x, 0., False, {})
        with self.assertRaisesRegex(ValueError, "terminate at horizon"):
            E.rollout(task, np.zeros((1, 13), np.float64), np.array([99]),
                      lambda o: np.zeros((len(o), 4), np.float32), max_transitions=1)

    def test_underestimated_fixed_q_cannot_improve_when_lowered(self):
        result = E.score_diagnostics([[-3., -2., -1.]], [[1., 1., 1.]], [[0., 0., 0.]],
                                     error_tolerance=.1, tie_tolerance=1e-10)
        self.assertEqual(result["raw_critic"]["under"], 3)
        self.assertEqual(result["adjusted_actor_score"]["under"], 3)
        self.assertEqual(result["absolute_error_worsened"], 3)
        self.assertEqual(result["absolute_error_improved"], 0)

    def test_harmful_and_beneficial_ranking_changes_both_reported(self):
        q = [[5., 4., 0.], [5., 4., 0.]]
        width = [[1., 0., 0.], [1., 0., 0.]]
        truth = [[0., 4., 0.], [4., 0., 0.]]
        result = E.score_diagnostics(q, width, truth, error_tolerance=.1, tie_tolerance=1e-10)
        ranking = result["common_action_ranking"]
        self.assertEqual((ranking["beneficial_switches"], ranking["harmful_switches"]), (1, 1))
        self.assertEqual(ranking["mean_value_gain"], 0.)

    def test_paired_returns_reject_misaligned_episode_ids(self):
        self.assertEqual(E.paired_returns([1, 2], [2, 4], ["a", "b"], ["a", "b"])["paired_mean_difference"], 1.5)
        with self.assertRaises(ValueError): E.paired_returns([1, 2], [2, 4], ["a", "b"], ["b", "a"])


if __name__ == "__main__":
    unittest.main()

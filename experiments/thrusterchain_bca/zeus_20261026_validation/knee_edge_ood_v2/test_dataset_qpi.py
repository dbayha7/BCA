"""Dataset schema, lineage, twins, cross-rejection, 4.5b world certification and the 4.5 evaluator cross-check.

These tests run the canonical dynamics on tiny smoke datasets (12 episodes) with a test-only entropy; no benchmark
seed is used and nothing is collected for the study.
"""
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.dont_write_bytecode = True

import numpy as np

import bindings as B
import dataset_knee as D
import fixtures as F
import probes as P
import qpi
import reconcile as C
import thrust_check as T
import worlds
from core import GATE, load_datasets, module
from knee_region import KNEE, SLOTS, past_knee_a0

ENTROPY = 77001


class Data(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.stall = D.generate(episodes=12, entropy=ENTROPY, world="stall3", provenance="smoke")
        cls.nostall = D.generate(episodes=12, entropy=ENTROPY, world="nostall", provenance="smoke")

    def test_twins(self):
        s, n = self.stall.arrays, self.nostall.arrays
        self.assertTrue(np.array_equal(s["requested_actions"], n["requested_actions"]))
        self.assertTrue(np.array_equal(s["load_band"], n["load_band"]))
        self.assertTrue(np.array_equal(np.asarray(s["observations"])[..., 10:12].view(np.uint32), np.asarray(n["observations"])[..., 10:12].view(np.uint32)))
        ok, report = C.twin_check(np.asarray(s["states"]), np.asarray(n["states"]), np.asarray(s["actions"]))
        self.assertTrue(ok)
        self.assertTrue(any(r["first_w_step"] is not None for r in report))

    def test_tampering_breaks_lineage_or_replay(self):
        arrays = {k: np.array(v) for k, v in self.stall.arrays.items()}
        arrays["requested_actions"][3, 10, 1] += 1e-12
        arrays["actions"] = arrays["requested_actions"].astype(np.float32)
        with self.assertRaises(D.KneeDatasetError):
            D.validate(arrays, self.stall.metadata)

    def test_cross_rejection(self):
        datasets = load_datasets()
        with self.assertRaises(datasets.DatasetError):
            datasets._validate(dict(self.stall.arrays), self.stall.metadata)
        old = datasets.generate_dataset(episodes=10, seed=55001, stall=True, stall_slope=3.0)
        with self.assertRaises(D.KneeDatasetError):
            D.validate(dict(old.arrays), old.metadata)
        with tempfile.TemporaryDirectory() as tmp:
            new_path, old_path = Path(tmp) / "new.npz", Path(tmp) / "old.npz"
            D.save(self.stall, new_path)
            datasets.save_dataset(old, old_path)
            with self.assertRaises(datasets.DatasetError):
                datasets.load_dataset(new_path)
            with self.assertRaises(D.KneeDatasetError):
                D.load(old_path)
            self.assertEqual(D.load(new_path).digest, self.stall.digest)

    def test_world_certification(self):
        s = self.stall.arrays
        ok, report = T.check(np.asarray(s["states"]), np.asarray(s["actions"]), "stall3")
        self.assertTrue(ok, report)
        self.assertLess(T.identity_rows(np.asarray(s["states"]), np.asarray(s["actions"]), "stall3"), 1e-9)
        # A deliberate slope-1.5 dataset presented as stall3 must fail replay, the slope-3 identity and reconstruction.
        env = worlds.load_environment()
        fake = np.array(s["states"])
        p15 = worlds.physics("stall15_identity")
        for t in range(100):
            fake[:, t + 1], _ = env.dynamics(fake[:, t], np.asarray(s["actions"])[:, t].astype(np.float64), p15)
        ok15, _ = T.check(fake, np.asarray(s["actions"]), "stall3")
        self.assertFalse(ok15)
        self.assertGreater(T.identity_rows(fake, np.asarray(s["actions"]), "stall3"), 1e-9)
        arrays = {k: np.array(v) for k, v in s.items()}
        arrays["states"], arrays["observations"][..., :13] = fake, fake.astype(np.float32)
        with self.assertRaises(D.KneeDatasetError):
            D.validate(arrays, self.stall.metadata)
        with self.assertRaises(ValueError):
            worlds.assert_world(env.Physics(stall=True, stall_slope=1.5), "stall3")

    def test_nostall_dataset_bound_as_stall3_raises(self):
        with self.assertRaises(ValueError):
            B.plan_streams(self.nostall, "stall3", test_seeds=F.TEST_SEEDS)

    def test_research_label_on_smoke_entropy_is_refused(self):
        metadata = dict(self.stall.metadata, provenance="research")
        with self.assertRaises(D.KneeDatasetError):
            D.validate(dict(self.stall.arrays), metadata)

    def test_log_density_tamper_breaks_lineage(self):
        arrays = {k: np.array(v) for k, v in self.stall.arrays.items()}
        arrays["behavior_log_density"][0, 0] += 5e-15
        with self.assertRaises(D.KneeDatasetError):
            D.validate(arrays, self.stall.metadata)

    def test_host_semantics(self):
        s = self.stall.arrays
        self.assertTrue(np.all(np.asarray(s["terminated"])[:, -1]) and not np.any(np.asarray(s["terminated"])[:, :-1]))
        self.assertFalse(np.any(s["truncated"]))


class Evaluators(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        rng = np.random.Generator(np.random.PCG64(5))
        cls.anchors = []
        for i, t in enumerate((0, 25, 50, 75)):
            prefix = np.zeros((t, 4), np.float32)
            prefix[:, 0] = rng.uniform(-0.9, -0.7, t).astype(np.float32)
            prefix[:, 1:] = rng.uniform(-0.7, 0.7, (t, 3)).astype(np.float32)
            cls.anchors.append({"reset_seed": 1000 + i, "current": (0.9, 1.1), "prefix_actions": prefix})
        actor = F.toy_actor()
        cls.seen = []

        def forward(obs):
            out = actor(obs)
            cls.seen.append(out)
            return out
        cls.forward = staticmethod(forward)
        cls.probes = P.build(4, ("edge_knee", "s10", "s20"))

    def test_raw_matches_env_in_every_world_and_the_clip_engages(self):
        for world in ("stall3", "nostall", "stall15_identity"):
            raw, raw_calls, raw_receipt = qpi.raw_values(self.anchors, self.probes, self.forward, world)
            env, env_calls, env_receipt = qpi.env_values(self.anchors, self.probes, self.forward, world)
            ok, worst = qpi.cross_check(raw, env)
            self.assertTrue(ok, (world, worst))
            self.assertEqual(raw_calls, env_calls)
            self.assertEqual(raw_receipt["continuation"]["world"], world)
            self.assertEqual(env_receipt["continuation"]["physics"], worlds.WORLD_DICTS[world])
            self.assertEqual(env_receipt["prefix"]["world"], "stall3")
        outputs = np.concatenate([s.reshape(-1, 4) for s in self.seen]).astype(np.float64)
        self.assertGreater(int(np.sum(np.abs(outputs) > KNEE)), 0)          # the +-KNEE clip really acted

    def test_broken_clip_in_one_evaluator_is_caught(self):
        raw, _, _ = qpi.raw_values(self.anchors, self.probes, self.forward, "stall3", clip_all=True)
        env, _, _ = qpi.env_values(self.anchors, self.probes, self.forward, "stall3", clip_all=False)
        self.assertFalse(qpi.cross_check(raw, env)[0])

    def test_identity_in_both_stalling_worlds_and_world_equality_within_knee(self):
        names = ("s05", "s10", "s20")
        a0 = np.asarray([SLOTS[n] for n in names], np.float32)
        for tag in ("stall3", "stall15_identity"):
            probes = np.concatenate([P.build(4, names), P.build(4, names, kind="identity_image", world=tag)], axis=1)
            values, _, _ = qpi.raw_values(self.anchors, probes, self.forward, tag)
            residual, rec = worlds.identity_residual(values[:, :3], values[:, 3:], a0, tag)
            self.assertEqual(rec["formula"], worlds.LITERAL_FORMULAS[tag])
            self.assertTrue(np.all(np.abs(residual) <= 1e-6 * np.maximum(1, np.abs(values[:, :3]))), tag)
            other = "stall15_identity" if tag == "stall3" else "stall3"
            wrong, _ = worlds.identity_residual(values[:, :3], values[:, 3:], a0, other)
            self.assertTrue(np.any(np.abs(wrong) > 1e-3))                     # a slope mix-up breaks the identity
        inside = P.build(4, ("edge_knee", "edge_700"))
        stall, _, _ = qpi.raw_values(self.anchors, inside, self.forward, "stall3")
        nostall, _, _ = qpi.raw_values(self.anchors, inside, self.forward, "nostall")
        self.assertTrue(qpi.cross_check(stall, nostall)[0])

    def test_wrong_horizon_is_caught_by_cross_check(self):
        raw, _, _ = qpi.raw_values(self.anchors, self.probes, self.forward, "stall3")
        shifted = [dict(a, prefix_actions=np.vstack([a["prefix_actions"], np.zeros((1, 4), np.float32)]))
                   for a in self.anchors]
        wrong, _, _ = qpi.env_values(shifted, self.probes, self.forward, "stall3")
        self.assertFalse(qpi.cross_check(raw, wrong)[0])

    def test_out_of_range_first_action_raises(self):
        bad = self.probes.copy()
        bad[0, 0, 1] = np.float32(1.5)
        with self.assertRaises(ValueError):
            qpi.raw_values(self.anchors, bad, self.forward, "stall3")

    def test_bit_parity_with_the_prior_reference_continuation(self):
        """raw_values_from_states(clip_all=False, stall15_identity) against the gate's own continuation_values."""
        if str(GATE) not in sys.path:
            sys.path.insert(0, str(GATE))
        reference = module("_knee_test_gate_reference", GATE / "reference.py")
        env = worlds.load_environment()
        states = np.stack([qpi.raw_anchor_state(a) for a in self.anchors])
        times = np.asarray([len(a["prefix_actions"]) for a in self.anchors], np.int64)
        banks = np.concatenate([self.probes, P.build(4, [-0.5, -0.25, 0.0], kind="known_answer")], axis=1)
        actor = F.toy_actor()
        batches_ref, batches_new = [], []

        class Policy:
            def act(self, obs):
                batches_ref.append(np.array(obs))
                return actor(obs)

            def assert_unchanged(self):
                pass

        def forward(obs):
            batches_new.append(np.array(obs))
            return actor(obs)
        expected = reference.continuation_values(env, Policy(), states, times, banks, reference.new_ledger())
        got, calls, _ = qpi.raw_values_from_states(states, times, banks, forward, "stall15_identity", clip_all=False)
        self.assertTrue(np.array_equal(got, expected))
        self.assertEqual(calls, len(batches_ref))
        self.assertTrue(all(np.array_equal(a, b) for a, b in zip(batches_ref, batches_new)))


if __name__ == "__main__":
    unittest.main()

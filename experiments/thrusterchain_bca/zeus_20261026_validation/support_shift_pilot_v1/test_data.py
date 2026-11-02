"""Synthetic intact-tuple/role tests; no real data or model execution."""
from collections import namedtuple
from copy import deepcopy
from types import SimpleNamespace
import unittest

import numpy as np
import data
import design

NATIVE = SimpleNamespace(Transition=namedtuple("Transition", data.FIELDS))


def fixture():
    nt, nc = 5, 2
    episode = np.asarray([2, 5, 6, 9, 12, 20, 23], dtype=np.int64)
    obs = np.zeros((nt + nc, 101, 14), dtype=np.float32)
    obs[:, :, 0] = episode[:, None]
    obs[:, :, 1] = np.arange(101)[None, :]
    obs[:, :, -1] = (1 - np.arange(101) / 100).astype(np.float32)
    actions = np.zeros((nt + nc, 100, 4), dtype=np.float32)
    actions[:, :, 0] = np.linspace(-1, 1, 100, dtype=np.float32)
    actions[::2, -1, 0] = .4
    reward = episode[:, None] * 1000 + np.arange(100)[None, :]
    done = np.zeros((nt + nc, 100), dtype=bool); done[:, -1] = True
    source = {"arrays": {"observations": obs, "actions": actions, "rewards": reward,
                         "terminated": done, "truncated": np.zeros_like(done)},
              "train_count": nt, "episode_ids": episode, "trajectory_ids": episode.copy(),
              "role_codes": np.r_[np.zeros(nt, dtype=np.uint8), np.ones(nc, dtype=np.uint8)],
              "dataset_sha256": "synthetic_fixture"}
    plans = design.training_rows(actions[:nt], episode[:nt])
    plan = {"dataset_sha256": source["dataset_sha256"], "row_selection_seeds": dict(design.SEEDS), "conditions": {}}
    def ids(local, episodes):
        return episodes[local // 100] * 100 + local % 100
    for name, rows in plans.items():
        probes = design.one_per_episode(rows, episode[:nt], 100, design.SEEDS["gradient_probes"])
        scale = np.random.Generator(np.random.PCG64(design.SEEDS["scale_bank"])).choice(rows, 256, replace=False)
        eligible = np.flatnonzero(design.support_mask(actions[nt:]).reshape(-1)) if name == "restricted" else np.arange(nc * 100)
        cal = design.one_per_episode(eligible, episode[nt:], 100, design.SEEDS["calibration_rows"])
        a = actions[:nt].reshape(-1, 4)[rows]
        plan["conditions"][name] = {"training_rows": len(rows), "training_episodes": nt,
            "training_source_row_ids_sha256": data._ids_digest(ids(rows, episode[:nt])),
            "gradient_probe_source_row_ids": ids(probes, episode[:nt]).tolist(),
            "scale_bank_source_row_ids": ids(scale, episode[:nt]).tolist(),
            "calibration_source_row_ids": ids(cal, episode[nt:]).tolist(),
            "training_rows_above_gap_cutoff": int(np.sum(a[:, 0] > .5)),
            "calibration_rows_above_gap_cutoff": int(np.sum(actions[nt:].reshape(-1, 4)[cal, 0] > .5)),
            "a0_min": float(a[:, 0].min()), "a0_max": float(a[:, 0].max())}
    return source, plan


class DataTests(unittest.TestCase):
    def setUp(self):
        self.source, self.plan = fixture()

    def test_intact_original_tuples_terminal_and_no_normalization(self):
        result = data._assemble(NATIVE, "restricted", self.source, self.plan)
        self.assertEqual(int(result.train.done.sum()), 3)
        for i, row in enumerate(result.source_row_ids):
            episode, t = divmod(int(row), 100)
            self.assertEqual(result.train.obs[i, 0], episode)
            self.assertEqual(result.train.obs[i, 1], t)
            self.assertEqual(result.train.next_obs[i, 1], t + 1)
            self.assertEqual(result.train.reward[i], episode * 1000 + t)
            self.assertEqual(result.train.done[i], float(t == 99))
        self.assertEqual(result.train.obs.dtype, np.float32)
        self.assertEqual(result.source_row_ids.dtype, np.int64)
        with self.assertRaises(ValueError):
            result.train.action.setflags(write=True)

    def test_matched_counts_support_and_disjoint_roles(self):
        a = data._assemble(NATIVE, "restricted", self.source, self.plan)
        b = data._assemble(NATIVE, "broad_count_matched", self.source, self.plan)
        self.assertEqual(len(a.source_row_ids), len(b.source_row_ids))
        np.testing.assert_array_equal(np.unique(a.source_row_ids // 100, return_counts=True)[1],
                                      np.unique(b.source_row_ids // 100, return_counts=True)[1])
        self.assertTrue(np.all(a.train.action[:, 0] <= .5))
        self.assertTrue(np.any(b.train.action[:, 0] > .5))
        self.assertFalse(np.intersect1d(a.source_row_ids // 100, a.heldout_ids["episode_ids"]).size)
        self.assertEqual(len(np.unique(a.heldout_ids["episode_ids"])), 2)
        for bank in (a.probe, a.scale_bank, a.heldout):
            self.assertTrue(np.all(bank.action[:, 0] <= .5))

    def test_rejects_protected_role_before_assembly(self):
        self.source["role_codes"][-1] = 2
        with self.assertRaisesRegex(ValueError, "Protected"):
            data._assemble(NATIVE, "restricted", self.source, self.plan)

    def test_rejects_episode_overlap(self):
        self.source["episode_ids"][-1] = self.source["episode_ids"][0]
        with self.assertRaisesRegex(ValueError, "identity overlap"):
            data._assemble(NATIVE, "restricted", self.source, self.plan)

    def test_rejects_changed_plan_identity(self):
        for name in ("training_source_row_ids_sha256", "gradient_probe_source_row_ids",
                     "scale_bank_source_row_ids", "calibration_source_row_ids"):
            with self.subTest(name=name):
                plan = deepcopy(self.plan)
                item = plan["conditions"]["restricted"]
                if name.endswith("sha256"):
                    item[name] = "0" * 64
                else:
                    item[name][0] += 1
                with self.assertRaisesRegex(ValueError, "DATA_PLAN"):
                    data._assemble(NATIVE, "restricted", self.source, plan)

    def test_rejects_missing_successor_or_changed_terminal(self):
        self.source["arrays"]["terminated"][0, 0] = True
        with self.assertRaisesRegex(ValueError, "finite-horizon"):
            data._assemble(NATIVE, "restricted", self.source, self.plan)


if __name__ == "__main__":
    unittest.main()

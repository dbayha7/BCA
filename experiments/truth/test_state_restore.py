"""CPU checks for experiments/truth/state_restore.py on synthetic D4RL-format files made in the real simulator.

Checks: the converted-row -> raw-row map of runtime/td3_bc._convert (timeouts dropped, next observation =
raw row + 1); Physics.step equals env.step bit for bit (observation, reward, done) through the real wrapper
chain, including after a restore into a sim with a different history; the one-step gate passes on synthetic
data for both tasks with a zero restore miss; the negative controls fail (states misaligned by one row, and
actions misaligned by one row); the observation-rebuild fallback without infos/* reproduces the infos states
on every coordinate but x, passes the gate and drops exactly the rows at the velocity clip.
Synthetic numbers are test fixtures, not findings.
JAX_PLATFORMS=cpu LD_LIBRARY_PATH=$HOME/.mujoco/mujoco210/bin python -m unittest experiments.truth.test_state_restore
"""

import contextlib
import io
import json
import shutil
import tempfile
import unittest
from pathlib import Path

import numpy as np

from experiments.truth import state_restore as SR
from experiments.truth import synthetic as SY

TASKS = ("hopper", "walker2d")


class StateRestoreTest(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.tmp = Path(tempfile.mkdtemp(prefix="truth-restore-"))
        cls.physics, cls.raw, cls.raw_noinfo, cls.conv = {}, {}, {}, {}
        for task in TASKS:
            physics = SR.Physics(task)
            meta = SY.write_synthetic(cls.tmp, task, n_rows=2000, seed=11, physics=physics)
            cls.physics[task] = physics
            cls.raw[task] = SR.load_raw(physics, meta["with_infos"])
            cls.raw_noinfo[task] = SR.load_raw(physics, meta["without_infos"])
            cls.conv[task] = SR.converted_rows(cls.raw[task])

    @classmethod
    def tearDownClass(cls):
        for physics in cls.physics.values():
            physics.close()
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def gate(self, task, raw=None, offset=0, ids=None, **kwargs):
        raw = self.raw[task] if raw is None else raw
        converted, rows, _ = SR.converted_rows(raw)
        ids = np.arange(len(rows)) if ids is None else ids
        logged, states = SR.gate_inputs(task, raw, rows, converted, ids, offset=offset)
        return SR.one_step_gate(self.physics[task], logged, states, **kwargs), logged, states

    def test_task_aliases_and_synthetic_mark(self):
        self.assertEqual(SR.task_spec("hopper").dataset, "hopper-medium-v2")
        self.assertEqual(SR.task_spec("walker2d-medium-replay-v2").name, "walker2d")
        with self.assertRaises(ValueError):
            SR.task_spec("halfcheetah")
        for path in self.tmp.glob("*.hdf5"):
            self.assertTrue(SY.is_synthetic(path))

    def test_converted_rows_map_to_raw_rows(self):
        for task in TASKS:
            raw, (converted, rows, ids) = self.raw[task], self.conv[task]
            self.assertTrue(raw["timeouts"].any(), "the fixture must contain timeouts")
            self.assertFalse(np.any(raw["timeouts"][rows]), "timeout rows must be dropped")
            self.assertEqual(len(rows), len(raw["rewards"]) - 1 - int(raw["timeouts"][:-1].sum()))
            np.testing.assert_array_equal(converted["observations"], raw["observations"][rows])
            np.testing.assert_array_equal(converted["next_observations"], raw["observations"][rows + 1])
            np.testing.assert_array_equal(converted["actions"], raw["actions"][rows])
            live = ~converted["terminals"]
            np.testing.assert_array_equal(converted["next_observations"][live], raw["next_observations"][rows][live])

    def test_physics_step_is_env_step(self):
        """Bare-MjSim step == env.step (whole wrapper chain), bitwise, also after unrelated history in the sim."""
        for task in TASKS:
            physics, raw = self.physics[task], self.raw[task]
            env, base = physics.env, physics.base
            g = np.random.default_rng(3)
            sim = physics.new_sim()
            for row in g.choice(len(raw["rewards"]) - 1, 25, replace=False):
                qpos, qvel = raw["infos/qpos"][row].astype(np.float64), raw["infos/qvel"][row].astype(np.float64)
                physics.restore(sim, raw["infos/qpos"][row + 1], raw["infos/qvel"][row + 1])
                for _ in range(3):  # unrelated history that a restore must erase
                    physics.step(sim, g.uniform(-1, 1, physics.task.action_dim).astype(np.float32))
                physics.restore(sim, qpos, qvel)
                env.reset()
                base.sim.reset()
                base.set_state(qpos, qvel)
                base.sim.data.qacc_warmstart[:] = 0.0
                np.testing.assert_array_equal(physics.observation(sim), base._get_obs())
                for _ in range(5):
                    a = g.uniform(-1, 1, physics.task.action_dim).astype(np.float32)
                    obs, reward, done, _ = env.step(a)
                    mine, my_done = physics.step(sim, a)
                    np.testing.assert_array_equal(physics.observation(sim), obs)
                    self.assertEqual(mine, reward)
                    self.assertEqual(my_done, bool(done))
                    np.testing.assert_array_equal(sim.data.qpos, base.sim.data.qpos)
                    if done:
                        break

    def test_gate_passes_on_synthetic_data(self):
        for task in TASKS:
            result, _, _ = self.gate(task)
            arrays = result.pop("_arrays")
            self.assertTrue(result["passed"], result)
            self.assertEqual(result["source"], "infos")
            self.assertEqual(result["restore_miss"]["max"], 0.0)
            self.assertEqual(result["terminal_agreement"], 1.0)
            self.assertLess(result["next_obs_miss"]["p99"], 1e-4)  # float32 storage of the state only
            self.assertLess(result["reward_miss"]["p99"], 1e-5)
            self.assertEqual(result["next_obs_compared"], result["rows_checked"])  # next_observations covers terminals
            self.assertGreater(result["terminal_rows"], 0)
            self.assertEqual(len(arrays["restore_miss"]), result["rows_checked"])

    def test_gate_fails_when_states_are_misaligned_by_one(self):
        """Negative control: states from raw row r + 1 paired with row r's observation, action and outcome."""
        for task in TASKS:
            result, _, _ = self.gate(task, offset=1)
            result.pop("_arrays")
            self.assertFalse(result["passed"])
            self.assertFalse(result["passes"]["restore"])
            self.assertFalse(result["passes"]["next_obs"])
            self.assertGreater(result["next_obs_miss"]["median"], 1e-2)

    def test_gate_fails_when_actions_are_misaligned_by_one(self):
        """Negative control: the right state, the next row's action. The restore check passes, the step fails."""
        for task in TASKS:
            raw = self.raw[task]
            converted, rows, _ = self.conv[task]
            ids = np.flatnonzero(~converted["terminals"][:-1])
            logged, states = SR.gate_inputs(task, raw, rows, converted, ids)
            logged = dict(logged, action=np.asarray(converted["actions"])[ids + 1])
            result = SR.one_step_gate(self.physics[task], logged, states)
            result.pop("_arrays")
            self.assertTrue(result["passes"]["restore"])
            self.assertFalse(result["passes"]["next_obs"])
            self.assertFalse(result["passed"])

    def test_rebuild_without_infos(self):
        for task in TASKS:
            spec = SR.task_spec(task)
            converted, rows, _ = self.conv[task]
            raw_n = self.raw_noinfo[task]
            self.assertFalse(SR.has_infos(raw_n))
            rebuilt = SR.logged_states(task, raw_n, rows)
            infos = SR.logged_states(task, self.raw[task], rows)
            self.assertEqual(rebuilt.source, "rebuilt")
            at_clip = np.any(np.abs(raw_n["observations"][rows][:, spec.nq - 1:]) >= 10.0, axis=1)
            np.testing.assert_array_equal(~rebuilt.kept, at_clip)
            self.assertEqual(rebuilt.dropped_at_clip, int(at_clip.sum()))
            k = rebuilt.kept
            np.testing.assert_array_equal(rebuilt.qpos[k, 1:], infos.qpos[k, 1:])
            np.testing.assert_array_equal(rebuilt.qvel[k], infos.qvel[k])
            np.testing.assert_array_equal(rebuilt.qpos[:, 0], 0.0)
            result, _, _ = self.gate(task, raw=raw_n)
            result.pop("_arrays")
            self.assertTrue(result["passed"], result)
            self.assertEqual(result["rows_dropped_at_clip"], int(at_clip.sum()))
            self.assertEqual(result["rows_checked"], int((~at_clip).sum()))
        # Walker2d's synthetic policy falls hard: many rows sit at the clip, so dropping is exercised.
        self.assertGreater(SR.logged_states("walker2d", self.raw_noinfo["walker2d"],
                                            self.conv["walker2d"][1]).dropped_at_clip, 0)

    def test_rebuild_drops_crafted_clip_rows(self):
        obs = self.raw["hopper"]["observations"][:6].copy()
        obs[1, 7], obs[4, 10] = 10.0, -10.0
        _, _, kept = SR.rebuild_from_obs("hopper", obs, x0=2.5)
        np.testing.assert_array_equal(kept, [True, False, True, True, False, True])
        qpos, _, _ = SR.rebuild_from_obs("hopper", obs, x0=2.5)
        np.testing.assert_array_equal(qpos[:, 0], 2.5)

    def test_stage0_command_line(self):
        """keys and gate write strict JSON (and the gate its per-row npz) for an explicit file; the pinned
        lookup refuses a directory without the configured cache file."""
        with contextlib.redirect_stdout(io.StringIO()):
            self._stage0_command_line()

    def _stage0_command_line(self):
        out = self.tmp / "cli"
        noinfo = self.tmp / "walker2d_synthetic_noinfos.hdf5"
        SR.main(["keys", "--dataset", "walker2d", "--file", str(noinfo), "--output", str(out / "keys.json")])
        keys = json.loads((out / "keys.json").read_text())
        self.assertFalse(keys["has_infos_qpos"])
        self.assertTrue(keys["synthetic"])
        self.assertGreater(keys["rows_at_velocity_clip"], 0)
        hopper = self.tmp / "hopper_synthetic.hdf5"
        SR.main(["gate", "--dataset", "hopper", "--file", str(hopper), "--rows", "300", "--output",
                 str(out / "gate.json")])
        gate = json.loads((out / "gate.json").read_text())
        self.assertTrue(gate["passed"])
        self.assertEqual(gate["rows_checked"], 300)
        self.assertFalse(gate["file"]["pinned"])
        with np.load(out / "gate.npz") as npz:
            self.assertEqual(len(npz["converted_ids"]), 300)
        SR.main(["gate", "--dataset", "hopper", "--file", str(hopper), "--rows", "300", "--offset", "1",
                 "--output", str(out / "negative.json")])
        self.assertFalse(json.loads((out / "negative.json").read_text())["passed"])
        with self.assertRaises(ValueError):
            SR.cached_file("hopper", self.tmp)

    def test_state_sd_zeroes_x(self):
        states = SR.logged_states("hopper", self.raw["hopper"], self.conv["hopper"][1])
        sd_q, sd_v = SR.state_sd(states)
        self.assertEqual(sd_q[0], 0.0)
        self.assertTrue(np.all(sd_q[1:] > 0) and np.all(sd_v > 0))


if __name__ == "__main__":
    unittest.main()

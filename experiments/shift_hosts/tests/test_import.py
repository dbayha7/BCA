"""Test (f), import integrity (plan section 4.5): the imported code is the pinned code, imported under a non-__main__
name, and after the repair XLA_FLAGS holds choice 8's flags while JAX's backend had not started. Also checks the
restated training-block builder that every other test and the runner use (unifloral_import.init_like_training_block)
on made-up arrays: no BCA code, no ThrusterChain data. An experiment-local attachment (plan section 3.1) that changes
none of the repository's declared methods.

Run in /opt/bca-venv from the repository root, once per device (the processor, then the card, short):
  env $(launch env) SHIFT_HOSTS_DEVICE=cpu taskset -c 0-3 /opt/bca-venv/bin/python -m experiments.shift_hosts.tests.test_import
  env $(launch env) SHIFT_HOSTS_DEVICE=gpu XLA_PYTHON_CLIENT_PREALLOCATE=false ... (same)
"""

import os
import shutil
import subprocess
import sys
import unittest
from pathlib import Path

import numpy as np

from experiments.shift_hosts import unifloral_import as UI

AREA = "scaffold"
REPO = UI.REPO


def _device():
    return os.environ.get(UI.DEVICE_ENV) or "cpu"


def _leaves_equal(a, b):
    import jax

    la, lb = jax.tree_util.tree_leaves(a), jax.tree_util.tree_leaves(b)
    return len(la) == len(lb) and all(np.asarray(x).tobytes() == np.asarray(y).tobytes() and
                                      np.asarray(x).dtype == np.asarray(y).dtype for x, y in zip(la, lb))


def _child(code, env):
    """Run `code` in a fresh process of this Python from the repository root; returns (exit status, output)."""
    out = subprocess.run([sys.executable, "-c", code], cwd=str(REPO), env=env, capture_output=True, text=True,
                         timeout=300)
    return out.returncode, out.stdout + out.stderr


class ImportIntegrityTest(unittest.TestCase):
    """The four files: fingerprints, module names, XLA_FLAGS, backend, no bytecode."""

    @classmethod
    def setUpClass(cls):
        cls.record = UI.prepare(device=_device())      # first: before anything can start JAX's backend

    def test_fingerprints_equal_the_manifest(self):
        """Every imported file's SHA-256 equals configs/unifloral.json (stop rule S1), checked again now."""
        pinned = UI.manifest_sha256()
        for learner in UI.LEARNERS:
            path = UI.ALGORITHMS / f"{learner}.py"
            self.assertEqual(self.record["sha256"][learner], pinned[f"algorithms/{learner}.py"])
            self.assertEqual(UI.sha256_file(path), pinned[f"algorithms/{learner}.py"])
        self.assertEqual(self.record["unifloral_revision"], UI.REVISION)

    def test_imported_under_their_own_names_not_main(self):
        """Each file is a module named shift_hosts_unifloral_<learner>, loaded from baselines/unifloral/algorithms,
        with the pieces the plan imports unchanged; its training block never ran."""
        for learner in UI.LEARNERS:
            m = UI.load(learner)
            self.assertEqual(m.__name__, UI.MODULE_PREFIX + learner)
            self.assertNotEqual(m.__name__, "__main__")
            self.assertEqual(Path(m.__file__).resolve(), (UI.ALGORITHMS / f"{learner}.py").resolve())
            for piece in ("Args", "Transition", "AgentTrainState", "create_train_state", "make_train_step"):
                self.assertTrue(hasattr(m, piece), f"{learner}.{piece}")
            self.assertEqual(m.Args().num_updates, 1_000_000)          # defaults untouched (td3_bc.py:28 etc.)
        self.assertFalse((REPO / "final_returns").exists())             # the training block writes there

    def test_xla_flags_repaired_before_the_backend_started(self):
        """Every import set Unifloral's own flag (td3_bc.py:19, cql.py:21, iql.py:20, rebrac.py:20); the repair then
        set choice 8's flags before JAX's backend started; full float32 matrix products."""
        flags = self.record["xla_flags"]
        for learner in UI.LEARNERS:
            self.assertEqual(flags["after_each_import"][learner], UI.UNIFLORAL_XLA_FLAGS)
        self.assertEqual(flags["after_import"], UI.UNIFLORAL_XLA_FLAGS)
        self.assertTrue(self.record["unifloral_flags_overwrote"])
        self.assertFalse(self.record["backend_started_before_repair"])
        self.assertEqual(flags["in_effect"], UI.chosen_flags(_device()))
        self.assertIn("--xla_gpu_deterministic_ops=true", flags["in_effect"].split())
        self.assertNotIn(UI.UNIFLORAL_XLA_FLAGS, flags["in_effect"])
        self.assertEqual(os.environ["XLA_FLAGS"], flags["in_effect"])
        self.assertEqual(self.record["matmul_precision"], "highest")
        import jax

        self.assertEqual(jax.config.jax_default_matmul_precision, "highest")

    def test_backend_is_the_requested_device(self):
        import jax

        self.assertTrue(UI.backend_started())
        self.assertEqual(jax.default_backend(), _device())
        self.assertEqual(self.record["backend"], _device())
        if _device() == "gpu":
            self.assertEqual(os.environ.get("XLA_PYTHON_CLIENT_PREALLOCATE"), "false")
        x = jax.numpy.arange(8.0)
        self.assertEqual(float(jax.jit(lambda v: (v * v).sum())(x)), 140.0)

    def test_dockerfile_variables_in_effect(self):
        self.assertEqual(UI.missing_launch_env(), [])
        for name in ("D4RL_SUPPRESS_IMPORT_ERROR", "MUJOCO_GL", "PYOPENGL_PLATFORM"):
            self.assertEqual(self.record["env"][name], UI.DOCKERFILE_ENV[name])

    def test_no_bytecode_written_under_baselines(self):
        caches = [p for p in (REPO / "baselines").rglob("__pycache__")] + list((REPO / "baselines").rglob("*.pyc"))
        self.assertEqual(caches, [])
        self.assertTrue(sys.dont_write_bytecode)

    def test_prepare_twice_returns_the_same_record_and_refuses_another_device(self):
        self.assertIs(UI.prepare(device=_device()), UI._STATE["record"])
        other = "cpu" if _device() == "gpu" else "gpu"
        with self.assertRaises(UI.LaunchEnvironmentError):
            UI.prepare(device=other)


class RefusalTest(unittest.TestCase):
    """What prepare() and the fingerprint check refuse."""

    def setUp(self):
        from experiments.shift_hosts import tests as T

        self.tmp = T.out_dir(AREA) / "tmp_import"
        shutil.rmtree(self.tmp, ignore_errors=True)
        self.tmp.mkdir(parents=True)

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_a_changed_byte_is_refused(self):
        """A copy of td3_bc.py with one byte changed fails the fingerprint check (stop rule S1); the unchanged copy
        passes."""
        pinned = UI.manifest_sha256()["algorithms/td3_bc.py"]
        good = self.tmp / "td3_bc.py"
        shutil.copyfile(UI.ALGORITHMS / "td3_bc.py", good)
        self.assertEqual(UI.verify_file(good, pinned), pinned)
        self.assertEqual(UI.check_pinned("td3_bc", algorithms=self.tmp), pinned)
        blob = bytearray(good.read_bytes())
        blob[100] ^= 1
        good.write_bytes(bytes(blob))
        with self.assertRaises(UI.PinnedCodeError):
            UI.verify_file(good, pinned)
        with self.assertRaises(UI.PinnedCodeError):
            UI.check_pinned("td3_bc", algorithms=self.tmp)
        with self.assertRaises(UI.PinnedCodeError):
            UI.verify_file(self.tmp / "missing.py", pinned)
        with self.assertRaises(ValueError):
            UI.check_pinned("edac")

    def test_import_executes_the_hashed_bytes(self):
        """_import reads the file once: the bytes it compiles are exactly the bytes whose SHA-256 it checked against
        configs/unifloral.json (no second read between the check and the import, no bytecode cache); a changed copy is
        refused before anything is compiled, and sys.modules stays as it was."""
        import hashlib
        from unittest import mock

        name = UI.MODULE_PREFIX + "td3_bc"
        before = sys.modules.get(name)
        pinned = UI.manifest_sha256()["algorithms/td3_bc.py"]
        copy = self.tmp / "td3_bc.py"
        shutil.copyfile(UI.ALGORITHMS / "td3_bc.py", copy)
        blob = bytearray(copy.read_bytes())
        blob[100] ^= 1
        copy.write_bytes(bytes(blob))
        never = mock.Mock(side_effect=AssertionError("nothing may be compiled from a refused file"))
        with mock.patch.object(UI, "compile", never, create=True):
            with self.assertRaises(UI.PinnedCodeError):
                UI._import("td3_bc", algorithms=self.tmp)
        never.assert_not_called()
        self.assertIs(sys.modules.get(name), before)
        shutil.copyfile(UI.ALGORITHMS / "td3_bc.py", copy)
        seen = []

        def spy(source, filename, *args, **kwargs):
            seen.append((hashlib.sha256(source).hexdigest(), str(filename)))
            raise RuntimeError("stopped before executing (the test only looks at what would be compiled)")

        with mock.patch.object(UI, "compile", spy, create=True):
            with self.assertRaises(RuntimeError):
                UI._import("td3_bc", algorithms=self.tmp)
        self.assertEqual(seen, [(pinned, str(copy))])
        self.assertIs(sys.modules.get(name), before)

    def test_refuses_after_the_backend_started(self):
        """In a fresh process: JAX's backend started first, so prepare() refuses (the flags could not take effect)."""
        env = UI.launch_env()
        env[UI.DEVICE_ENV] = _device()
        code = ("import jax; jax.numpy.zeros(1).block_until_ready()\n"
                "from experiments.shift_hosts import unifloral_import as UI\n"
                "try:\n    UI.prepare()\nexcept UI.BackendStartedError:\n    raise SystemExit(17)\n"
                "raise SystemExit(0)\n")
        status, out = _child(code, env)
        self.assertEqual(status, 17, out[-2000:])

    def test_refuses_without_the_dockerfile_variables(self):
        """In a fresh process without LD_LIBRARY_PATH's MuJoCo folder: refused before any import."""
        env = UI.launch_env()
        env["LD_LIBRARY_PATH"] = ""
        env[UI.DEVICE_ENV] = "cpu"
        code = ("from experiments.shift_hosts import unifloral_import as UI\n"
                "try:\n    UI.prepare()\nexcept UI.LaunchEnvironmentError:\n    raise SystemExit(19)\n"
                "raise SystemExit(0)\n")
        status, out = _child(code, env)
        self.assertEqual(status, 19, out[-2000:])

    def test_load_before_prepare_is_refused(self):
        env = UI.launch_env()
        code = ("from experiments.shift_hosts import unifloral_import as UI\n"
                "try:\n    UI.load('td3_bc')\nexcept RuntimeError:\n    raise SystemExit(23)\n"
                "raise SystemExit(0)\n")
        status, out = _child(code, env)
        self.assertEqual(status, 23, out[-2000:])


class TrainingBlockBuilderTest(unittest.TestCase):
    """init_like_training_block and make_transition on made-up arrays (no BCA code): deterministic, Unifloral's own
    objects, the ReBRAC roll, and two finite steps of each imported step."""

    @classmethod
    def setUpClass(cls):
        UI.prepare(device=_device())
        from experiments.shift_hosts.tests import madeup

        cls.m = madeup.build(0)

    def test_rebrac_roll_is_unifloral_and_madeups(self):
        """make_transition's next_action is jnp.roll(actions, -1, axis=0) (rebrac.py:335) and equals the made-up
        builder's np.roll bit for bit; a given array (bank rows) is used as given."""
        import jax.numpy as jnp

        tr = UI.make_transition("rebrac", self.m.rows("train"))
        self.assertEqual(np.asarray(tr.next_action).tobytes(),
                         np.asarray(jnp.roll(self.m.train.action, -1, axis=0)).tobytes())
        self.assertEqual(np.asarray(tr.next_action).tobytes(), self.m.next_action["train"].tobytes())
        bank = UI.make_transition("rebrac", self.m.rows("bank"), next_action=self.m.next_action["bank"])
        self.assertEqual(np.asarray(bank.next_action).tobytes(), self.m.next_action["bank"].tobytes())
        self.assertEqual(type(tr).__name__, "Transition")
        self.assertEqual(tr._fields, ("obs", "action", "reward", "next_obs", "next_action", "done"))

    def test_builds_are_deterministic_and_run(self):
        import jax

        from experiments.shift_hosts import guard

        for learner in UI.LEARNERS:
            with self.subTest(learner=learner):
                args = UI.args_for(learner, seed=7, gamma=0.98, num_updates=100_000)
                a = UI.init_like_training_block(learner, args, self.m.rows("train"))
                b = UI.init_like_training_block(learner, args, self.m.rows("train"))
                self.assertTrue(_leaves_equal((a.rng, a.agent_state), (b.rng, b.agent_state)))
                self.assertEqual(guard.tree_sha256(a.dataset), guard.tree_sha256(self.m.leaves("train", learner)))
                self.assertEqual(len(a.apply_fns), 3 if learner in ("cql", "iql") else 2)
                self.assertEqual(a.obs_mean.shape, (13,))
                step = UI.make_unifloral_step(a, args)
                (rng, state), loss = jax.jit(lambda c: jax.lax.scan(step, c, None, length=2))((a.rng, a.agent_state))
                self.assertTrue(all(bool(np.all(np.isfinite(np.asarray(v)))) for v in loss.values()), learner)
                self.assertFalse(_leaves_equal(state, a.agent_state))


if __name__ == "__main__":
    UI.prepare(device=_device())
    from experiments.shift_hosts import tests as T

    sys.exit(T.run(sys.modules[__name__], AREA, device_name=_device(), extra=UI.record))

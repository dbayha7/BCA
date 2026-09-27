"""Known-answer and adapter tests. Fixtures are never research observations.

Run in the existing Linux/WSL requirements environment with JAX_PLATFORMS=cpu.
All attempts save evidence under runs/ood, including failures. No optimizer runs.
"""

from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
from types import SimpleNamespace
import unittest

import numpy as np

from experiments.ood.oracle import Oracle, enumerate_returns
from experiments.ood.adapters import CheckpointAdapter, parity_gate, continuation_keys, verify_continuation_keys
from experiments.ood.simulator import LocomotionAdapter, tree_hash


OUT = None
REPORT = {"kind": "engineering_fixture_only", "optimizer_updates": 0,
          "scientific_checkpoint_acceptance": False, "checks": []}


def save_json(name, value):
    def encode(x):
        if isinstance(x, np.ndarray):
            return dict(dtype=x.dtype.str, shape=x.shape, values=x.tolist())
        if isinstance(x, np.generic):
            return x.item()
        raise TypeError(type(x).__name__)
    with (OUT / name).open("x") as stream:
        json.dump(value, stream, indent=2, allow_nan=False, default=encode)


def setUpModule():
    global OUT
    from datetime import datetime, timezone
    from experiments.ood.protocol import ROOT, file_hash
    OUT = Path(os.environ.get("BCA_OOD_TEST_OUTPUT", str(ROOT / "runs/ood" /
        ("adapter-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")))))
    OUT.mkdir(parents=True, exist_ok=False)
    import shutil
    (OUT / "source").mkdir()
    for p in Path(__file__).parent.glob("*.py"):
        shutil.copyfile(p, OUT / "source" / p.name)
    save_json("declaration.json", dict(kind="engineering_only", training_updates=0,
        simulator_seeds=[1900927001, 1900927002], max_explicit_transitions_per_environment=64,
        constructor_transitions_per_environment=1, grand_ceiling=130,
        device="cpu", action_gate=1e-6, reward_gate=1e-7,
        source_sha256={p.name: file_hash(p) for p in Path(__file__).parent.glob("*.py")}))
    from runtime.environment import setup
    setup()
    import jax
    if jax.default_backend() != "cpu":
        raise RuntimeError("Fixture requires CPU; never silently select a GPU.")


def tearDownModule():
    save_json("observations.json", REPORT)


class OracleTests(unittest.TestCase):
    def test_exact_returns_and_support_harm_are_different(self):
        # At x=0, actions -1 and 0 are familiar. +1 is unfamiliar and beneficial.
        returns = enumerate_returns(0, [-1.0, 0.0, 1.0], continuation=0.0, horizon=3)
        np.testing.assert_allclose(returns, [-12.1, -3.0, -0.1], rtol=0, atol=1e-14)
        self.assertTrue(Oracle.in_support(0, -1.0))
        self.assertFalse(Oracle.in_support(0, 1.0))
        harm = returns[1] - np.asarray(returns)
        np.testing.assert_allclose(harm, [9.1, 0.0, -2.9], atol=1e-14)
        # Ranking fixtures, not an AUROC implementation (Task C owns metrics).
        self.assertEqual(np.argsort(harm).tolist(), [2, 1, 0])
        self.assertEqual(np.argsort(-harm).tolist(), [0, 1, 2])
        self.assertEqual(len(set(np.ones(3))), 1)
        for first, expected in zip([-1., 0., 1.], returns):
            env, total = Oracle(), 0.
            for action in [first, 0., 0.]:
                _, reward, terminated, truncated = env.step(action)
                total += reward
                if terminated or truncated:
                    break
            self.assertAlmostEqual(total, expected)

    def test_exact_oracle_restore_and_done(self):
        env = Oracle()
        state = env.capture()
        first = env.step(-1.0)
        env.restore(state)
        self.assertEqual(first, env.step(-1.0))
        env.step(-1.0)
        self.assertTrue(env.terminated)
        with self.assertRaises(ValueError):
            env.step(0.0)
        for key in state:
            broken = dict(state)
            del broken[key]
            with self.subTest(field=key), self.assertRaises(ValueError):
                env.restore(broken)


class CheckpointTests(unittest.TestCase):
    def test_roundtrip_original_hosts_and_rejection_cases(self):
        import importlib
        import jax
        import jax.numpy as jnp
        from flax import serialization as S
        from experiments.ood.protocol import file_hash
        archived_initial = {
            "td3_bc/host": "ff5a104bc2323b5c49da297cacaa3f1ca26dcc4e8cc4ae12ec5ad135c9a83230",
            "td3_bc/bca": "956f6107ec6dcc68d5652ffeb917f2d014f3f320076b2d39ba96a052b0929c1e",
            "rebrac/host": "407d77f05207932240195e270d28c1c62bf73c4e27269294631fca28aa35cbf2",
            "rebrac/bca": "87de73ae9000a14aa996cef0f3c451b569955ea209b049fc2a4795b1fb098b2d"}
        for host in ("td3_bc", "rebrac"):
            P = importlib.import_module("algorithms." + host + "_bca")
            for method in ("host", "bca"):
                with self.subTest(host=host, method=method):
                    g = np.random.default_rng(137)
                    arrays = [jnp.asarray(g.normal(size=s).astype(np.float32))
                              for s in ((48, 3), (48, 2), (48,), (48, 3))]
                    arrays[1] = jnp.tanh(arrays[1])
                    arrays.append(jnp.zeros(48))
                    data = (P.BASE.C.TransitionNA(*arrays, arrays[1]) if host == "rebrac"
                            else P.BASE.C.Transition(*arrays))
                    train = jax.tree.map(lambda x: x[:32], data)
                    held = jax.tree.map(lambda x: x[32:], data)
                    kwargs = dict(seed=137, batch_size=8, allow_off_config=True)
                    if host == "rebrac":
                        kwargs.update(hidden_dim=8, actor_n_hiddens=2, critic_n_hiddens=2)
                    args = P.BASE.Args(**kwargs)
                    cfg = (P.Config("host") if method == "host" else P.Config("bca",
                        posterior=P.PosteriorConfig(alpha=.2, credibility=.8, draws=8), blend=.5,
                        iw=P.AffinityIWConfig("affinity", .15 * 2**.5, .25, 0., 32)))
                    rng, state, models = (P.initialize(args, cfg, train) if host == "rebrac"
                                          else P.initialize(args, cfg, 3, 2))
                    initial = hashlib.sha256(S.to_bytes((rng, state, jnp.int32(0)))).hexdigest()
                    self.assertEqual(initial, archived_initial[host + "/" + method])
                    if method == "bca":
                        state, metrics = P.refresh(args, cfg, models, state, train, held,
                            jax.random.PRNGKey(1900927101), training_ids=np.arange(32),
                            heldout_ids=np.arange(32, 48))
                        P.require_valid(metrics)  # Synthetic radius only; no gradient step.
                    prefix = host + "-" + method
                    payload = dict(state=state, training_rng=rng, step=jnp.int32(0))
                    raw = S.to_bytes(payload)
                    path = OUT / (prefix + ".msgpack")
                    path.write_bytes(raw)
                    mean, std = np.array([.2, -.1, .7], np.float32), np.array([1., 2., 3.], np.float32)
                    prepared = SimpleNamespace(training=train, obs_mean=mean, obs_std=std, max_action=1.)
                    raw_obs = np.asarray(held.obs) * std + mean
                    normalized = (jnp.asarray(raw_obs) - mean) / std
                    key = jax.random.PRNGKey(1900927102)
                    reference = dict(actions=np.asarray(models[0].apply(state.native.actor.params, normalized)),
                        target=np.asarray(P.native_target(args, models, state.native, held, key)))
                    if method == "bca":
                        readout = P.bc_readout(models, state, held, cfg.blend)
                        reference.update(width=np.asarray(readout.width), dose=np.asarray(readout.dose))
                    np.savez(OUT / (prefix + "-direct-reference.npz"), **reference)
                    adapter = CheckpointAdapter(path, file_hash(path), host, args, cfg, prepared,
                                                expected_step=0, fixture=True)
                    parity_gate(OUT / (prefix + "-action-gate.npz"), reference["actions"],
                                adapter.actions(raw_obs), device="cpu")
                    np.testing.assert_array_equal(adapter.targets(held, key), reference["target"])
                    # Independent arithmetic and terminal mask, including ReBRAC's recorded-next-action penalty.
                    terminal = held._replace(done=jnp.ones(len(held.obs)))
                    np.testing.assert_array_equal(adapter.targets(terminal, key), held.reward)
                    noisy, target_q = adapter.target_components(held, key)
                    expected = held.reward + (1-held.done) * (args.gamma if host == "rebrac" else args.discount) * target_q
                    np.testing.assert_array_equal(reference["target"], expected)
                    if host == "rebrac":
                        with self.assertRaises(ValueError):
                            adapter.targets(P.BASE.C.Transition(*held[:5]), key)
                    if method == "bca":
                        got = adapter.score(held.obs, held.action, normalized=True)
                        np.testing.assert_array_equal(got["width"], reference["width"])
                        np.testing.assert_array_equal(got["dose"], reference["dose"])
                        independent = got["radius"] * got["unit"] * got["scale"]
                        np.testing.assert_allclose(got["width"], independent, rtol=2e-7, atol=0)
                    else:
                        self.assertIsNone(adapter.score(held.obs, held.action, normalized=True))
                    adapter.assert_unchanged()
                    self.assertEqual(path.read_bytes(), raw)
                    with self.assertRaises(ValueError):
                        CheckpointAdapter(path, "0"*64, host, args, cfg, prepared, expected_step=0, fixture=True)
                    with self.assertRaises(ValueError):
                        CheckpointAdapter(path, file_hash(path), host, args, cfg, prepared)
                    broken = S.msgpack_restore(raw)
                    del broken["state"]["native"]["actor"]["params"]
                    badpath = OUT / (prefix + "-missing-params.msgpack")
                    badpath.write_bytes(S.msgpack_serialize(broken))
                    with self.assertRaises(ValueError):
                        CheckpointAdapter(badpath, file_hash(badpath), host, args, cfg, prepared,
                                          expected_step=0, fixture=True)
                    for kind in ("counter", "rng_shape", "rng_dtype"):
                        broken = S.msgpack_restore(raw)
                        if kind == "counter":
                            broken["state"]["native"]["actor"]["step"] = 1
                        elif kind == "rng_shape":
                            broken["training_rng"] = np.zeros(3, np.uint32)
                        else:
                            broken["training_rng"] = np.asarray(broken["training_rng"], np.float32)
                        badpath = OUT / (prefix + "-" + kind + ".msgpack")
                        badpath.write_bytes(S.msgpack_serialize(broken))
                        with self.subTest(malformed=kind), self.assertRaises(ValueError):
                            CheckpointAdapter(badpath, file_hash(badpath), host, args, cfg, prepared,
                                              expected_step=0, fixture=True)
                    REPORT["checks"].append(dict(fixture=prefix, initial_sha256=initial,
                                                original_initial_hash_matched=True, step=0))

    def test_parity_failure_saves_actual_reference_and_error_before_raising(self):
        path = OUT / "intentional-parity-failure.npz"
        with self.assertRaises(ValueError):
            parity_gate(path, np.zeros(2), np.array([0., 1.01e-6]), device="cpu")
        saved = np.load(path)
        self.assertEqual(float(saved["error"].max()), 1.01e-6)
        with self.assertRaises(FileExistsError):
            parity_gate(path, np.zeros(2), np.zeros(2), device="cpu")
        shape_path = OUT / "intentional-shape-failure.npz"
        with self.assertRaises(ValueError):
            parity_gate(shape_path, np.zeros(2), np.zeros(3), device="cpu")
        self.assertTrue(shape_path.exists())

    def test_shared_stochastic_streams(self):
        keys = continuation_keys(1900927111, 250)
        np.testing.assert_array_equal(keys, continuation_keys(1900927111, 250))
        self.assertEqual(keys.shape, (250, 2))
        self.assertEqual(len({tuple(k) for k in keys}), 250)
        verify_continuation_keys(keys, 1900927111, 250)
        changed = keys.copy()
        changed[0, 0] ^= np.uint32(1)
        with self.assertRaises(ValueError):
            verify_continuation_keys(changed, 1900927111, 250)
        with self.assertRaises(ValueError):
            continuation_keys(True, 250)


class SimulatorTests(unittest.TestCase):
    def test_restore_schema_reward_and_episode_endings(self):
        for index, dataset in enumerate(("hopper-medium-v2", "walker2d-medium-replay-v2")):
            with self.subTest(dataset=dataset):
                sim = LocomotionAdapter(dataset, max_transitions=64,
                                        evidence_dir=OUT / (dataset + "-attempts"))
                try:
                    obs = sim.reset(1900927001 + index)
                    save_json(dataset + "-identity.json", sim.identity)
                    snapshot = sim.capture()
                    save_json(dataset + "-initial-state.json", snapshot)
                    self.assertEqual(sim.snapshot_hash(snapshot), tree_hash(snapshot))
                    for field in snapshot:
                        bad = deepcopy(snapshot)
                        del bad[field]
                        with self.subTest(field=field), self.assertRaises(ValueError):
                            sim.restore(bad)
                    self.assertEqual(sim.transitions, 0)
                    for field, value in (("qpos", snapshot["qpos"].astype(np.float32)),
                                         ("qvel", np.zeros(1)), ("has_reset", False),
                                         ("elapsed_steps", 1001), ("act", np.empty(0))):
                        bad = deepcopy(snapshot)
                        bad[field] = value
                        with self.subTest(malformed=field), self.assertRaises(ValueError):
                            sim.restore(bad)
                    with self.assertRaises(ValueError):
                        sim.restore(snapshot, expected_sha256="0" * 64)
                    a = np.linspace(-.9, .8, sim.action_dim, dtype=np.float32)
                    records = []
                    for action in (a, a.astype(np.float64), -a):
                        sim.restore(snapshot)
                        first = sim.step(action)
                        sim.restore(snapshot)
                        second = sim.step(action)
                        records.extend([first, second])
                        np.testing.assert_array_equal(first["observation"], second["observation"])
                        self.assertEqual(first["reward"], second["reward"])
                        self.assertEqual(first["terminated"], second["terminated"])
                        np.testing.assert_array_equal(first["sim_ctrl"], first["applied_action"])
                        self.assertLessEqual(first["reward_error"], 1e-7)
                    # A non-reset capture tests integration/warm-start state.
                    mid = sim.capture()
                    first = sim.step(a)
                    sim.restore(mid)
                    second = sim.step(a)
                    np.testing.assert_array_equal(first["observation"], second["observation"])
                    records.extend([first, second])
                    # Altered fixture states check native termination versus original time limit.
                    ended = deepcopy(snapshot)
                    ended["qpos"][1] = .3
                    sim.restore(ended)
                    terminal = sim.step(np.zeros(sim.action_dim, np.float32))
                    self.assertTrue(terminal["terminated"])
                    with self.assertRaises(ValueError):
                        sim.step(a)
                    sim.restore(snapshot)
                    sim.env._elapsed_steps = sim.max_episode_steps - 1
                    near_limit = sim.capture()
                    sim.restore(near_limit)
                    truncated = sim.step(np.zeros(sim.action_dim, np.float32))
                    self.assertTrue(truncated["truncated"])
                    self.assertFalse(truncated["terminated"])
                    with self.assertRaises(ValueError):
                        sim.step(a)
                    records.extend([terminal, truncated])
                    sim.restore(snapshot)
                    with self.assertRaises(ValueError):
                        sim.step(np.full(sim.action_dim, np.nan))
                    before = sim.transitions
                    sim.max_transitions = before
                    with self.assertRaises(ValueError):
                        sim.step(a)
                    self.assertEqual(sim.transitions, before)
                    np.savez(OUT / (dataset + "-transitions.npz"),
                             **{f"{i}_{k}": v for i, rec in enumerate(records) for k, v in rec.items()})
                    REPORT["checks"].append(dict(environment=dataset, transitions=sim.transitions,
                        constructor_transitions=1, physics_steps=(sim.transitions+1)*sim.frame_skip,
                        max_reward_error=max(x["reward_error"] for x in records),
                        exact_repeated_observations=True))
                finally:
                    sim.close()


if __name__ == "__main__":
    unittest.main()

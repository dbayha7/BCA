"""CPU checks for experiments/signal/frozen_signals.py: three signals on a frozen TD3+BC critic.

A synthetic HDF5 file drives the real short-train freeze (freeze_scores.py, 20 updates), then the
fit restores it, fits a fresh scale per signal and writes the artifacts. Checks: the min signal's fit
is algorithms.td3_bc_bca.fit_scale bit for bit, alone or fit jointly with the other signals (the
same XLA operations on the same inputs in the same order; fitting other signals adds independent
branches only); all signals see the same batch indices and targets, also across separately compiled
fits; the signal definitions; the artifacts load with d4rl_benchmark.load_pool and the benchmark
runs on them; the scale state restores. The cross-coverage evaluator is checked against a brute-force
row-by-row recomputation on tiny pools, the dose against its closed form.
JAX_PLATFORMS=cpu python -m unittest experiments.signal.test_frozen_signals
"""

import json
import shutil
import tempfile
import unittest
from pathlib import Path

import numpy as np

from calibration.wbcp import calibrate
from experiments.signal import frozen_signals as S
from experiments.wbcp import d4rl_benchmark as B

UPDATES, BLOCK = 40, 15
ALL = ("min", "q1", "maxabs")


def same_tree(a, b):
    import jax

    leaves = [jax.tree_util.tree_leaves(x) for x in (a, b)]
    return len(leaves[0]) == len(leaves[1]) and all(
        np.asarray(x).dtype == np.asarray(y).dtype and np.array_equal(np.asarray(x), np.asarray(y))
        for x, y in zip(*leaves))


def synthetic_heads(episodes=30, length=10, seed=0, gap=0.5):
    """Float32 heads in the heads.npz layout: Q2 near t, Q1 off Q2 by `gap` noise, one random eta per signal."""
    g = np.random.default_rng(seed)
    n = episodes * length
    t = g.normal(size=n).astype(np.float32)
    q2 = (t + g.normal(scale=0.5, size=n)).astype(np.float32)
    q1 = (q2 + gap * g.normal(size=n)).astype(np.float32)
    heads = dict(target=t, q=np.column_stack([q1, q2]).astype(np.float32), unit=np.array([0.7, 0.8, 0.9], np.float32))
    for i, s in enumerate(ALL):
        eta = (0.5 + g.random(n)).astype(np.float32)
        heads["eta_" + s] = eta
        heads["sigma_" + s] = (np.maximum(eta, np.float32(1e-6)) * heads["unit"][i]).astype(np.float32)
    return heads, np.repeat(np.arange(episodes), length), np.tile(np.arange(length), episodes)


DESIGNS = [dict(name="iid", label="iid", n=40, per_episode=None, spacing="random"),
           dict(name="reservation", label="resv", n=40, per_episode=3, spacing="reservation")]
OPTIONS = dict(alpha=0.1, beta=0.95, draws=200)


def brute_bank(heads, episode, timestep, design, b, seed, options):
    """One bank recomputed row by row, without the evaluator's sorted risk tables."""
    t, q = heads["target"], heads["q"]
    r1 = np.abs((t - q[:, 0]).astype(np.float64))
    r2 = np.abs((t - q[:, 1]).astype(np.float64))
    rmin = np.abs((t - np.minimum(q[:, 0], q[:, 1])).astype(np.float64))
    groups = B.episode_groups(episode, timestep) if design["per_episode"] else None
    rows = B.draw_calibration(B.stream(seed, "rows", design["name"], b), design["n"], len(t), groups,
                              design["per_episode"], design["spacing"])
    thresholds, miscoverage = [], []
    for s in ALL:
        sigma = heads["sigma_" + s].astype(np.float64)
        own = {"min": rmin / sigma, "q1": r1 / sigma, "maxabs": np.maximum(r1 / sigma, r2 / sigma)}[s]
        lam = calibrate(own[rows], B.stream(seed, "wbcp", design["name"], b), **options).threshold
        covered = {"min": rmin / sigma <= lam, "q1": r1 / sigma <= lam, "q2": r2 / sigma <= lam,
                   "maxabs": (r1 / sigma <= lam) & (r2 / sigma <= lam)}
        thresholds.append(lam)
        miscoverage.append([np.mean(~covered[r]) for r in S.RESIDUALS])
    return rows.size, np.array(thresholds), np.array(miscoverage)


class SignalDefinitionTests(unittest.TestCase):
    def test_signal_values_and_magnitudes(self):
        t = np.array([0.0, 1.0, -2.0, 3.0], np.float32)
        q = np.array([[1.0, -0.5], [0.0, 3.0], [-1.0, -3.0], [2.0, 4.0]], np.float32)  # last row: a tie
        np.testing.assert_array_equal(S.signal_value("min", t, q), [-0.5, 0.0, -3.0, 2.0])
        np.testing.assert_array_equal(S.signal_value("q1", t, q), [1.0, 0.0, -1.0, 2.0])
        np.testing.assert_array_equal(S.signal_value("maxabs", t, q), [1.0, 3.0, -1.0, 2.0])  # farther head, Q1 on ties
        heads = dict(target=t, q=q)
        mags = S.residual_magnitudes(heads)
        np.testing.assert_array_equal(mags["maxabs"], np.abs(t - S.signal_value("maxabs", t, q)).astype(np.float64))
        np.testing.assert_array_equal(mags["maxabs"], np.maximum(mags["q1"], mags["q2"]))
        np.testing.assert_array_equal(mags["min"], [0.5, 1.0, 1.0, 1.0])
        with self.assertRaises(ValueError):
            S.signal_value("q2", t, q)

    def test_jax_and_numpy_signal_values_agree_bitwise(self):
        import jax.numpy as jnp

        g = np.random.default_rng(1)
        t = g.normal(size=1000).astype(np.float32)
        q = g.normal(size=(1000, 2)).astype(np.float32)
        for s in ALL:
            np.testing.assert_array_equal(np.asarray(jnp.asarray(t) - S.signal_value(s, jnp.asarray(t), jnp.asarray(q), jnp)),
                                          t - S.signal_value(s, t, q))


class CrossCoverageTests(unittest.TestCase):
    seed = 7

    @classmethod
    def setUpClass(cls):
        cls.heads, cls.episode, cls.timestep = synthetic_heads()
        cls.pop = S.population(cls.heads, cls.episode, cls.timestep, ALL)

    def test_banks_match_brute_force(self):
        for design in DESIGNS:
            sizes, thresholds, miscoverage = S.run_banks(self.pop, design, 12, self.seed, OPTIONS)
            for b in range(12):
                size, lam, miss = brute_bank(self.heads, self.episode, self.timestep, design, b, self.seed, OPTIONS)
                self.assertEqual(sizes[b], size)
                np.testing.assert_array_equal(thresholds[b], lam)
                np.testing.assert_array_equal(miscoverage[b], miss)

    def test_bank_streams_do_not_depend_on_the_sweep(self):
        _, thresholds, miscoverage = S.run_banks(self.pop, DESIGNS[1], 8, self.seed, OPTIONS)
        _, lam, miss = S.run_bank(self.pop, DESIGNS[1], 5, self.seed, OPTIONS)
        np.testing.assert_array_equal(thresholds[5], lam)
        np.testing.assert_array_equal(miscoverage[5], miss)

    def test_maxabs_band_covers_each_head_at_least_as_often_as_both(self):
        _, _, miscoverage = S.run_banks(self.pop, DESIGNS[0], 20, self.seed, OPTIONS)
        m = ALL.index("maxabs")
        own = miscoverage[:, m, S.RESIDUALS.index("maxabs")]
        for r in ("min", "q1", "q2"):
            self.assertTrue(np.all(miscoverage[:, m, S.RESIDUALS.index(r)] <= own))

    def test_evaluation_summarizes_the_banks(self):
        result = S.evaluate_arrays(self.heads, self.episode, self.timestep, ALL, DESIGNS, banks=12, seed=self.seed,
                                   dose_banks=5, **OPTIONS)
        for d, design in enumerate(DESIGNS):
            raw = [brute_bank(self.heads, self.episode, self.timestep, design, b, self.seed, OPTIONS) for b in range(12)]
            thresholds = np.stack([r[1] for r in raw])
            miscoverage = np.stack([r[2] for r in raw])
            block = result["designs"][d]
            for i, s in enumerate(ALL):
                for k, r in enumerate(S.RESIDUALS):
                    entry = block["signals"][s]["failure"][r]
                    self.assertEqual(entry["failures"], int(np.sum(miscoverage[:, i, k] > 0.1)))
                    self.assertEqual(entry["trials"], 12)
                    self.assertAlmostEqual(entry["mean_miscoverage"], float(np.mean(miscoverage[:, i, k])), places=15)
                    self.assertLessEqual(entry["ci"][0], entry["fail"])
                    self.assertGreaterEqual(entry["ci"][1], entry["fail"])
                lam_star = result["population"]["lambda_star"][s]
                certified = np.isfinite(thresholds[:, i])  # a small bank may abstain; means use certified banks
                self.assertAlmostEqual(block["signals"][s]["abstain"], 1.0 - float(np.mean(certified)), places=15)
                self.assertAlmostEqual(block["signals"][s]["threshold_over_lambda_star"],
                                       float(np.mean(thresholds[certified, i])) / lam_star, places=12)
            sigma = {s: float(np.mean(self.heads["sigma_" + s].astype(np.float64))) for s in ALL}
            both = np.isfinite(thresholds[:, 2]) & np.isfinite(thresholds[:, 1])
            expected = np.mean(thresholds[both, 2] * sigma["maxabs"] / (thresholds[both, 1] * sigma["q1"]))
            self.assertAlmostEqual(block["width_ratio"]["maxabs/q1"], float(expected), places=12)
        self.assertEqual(set(result["population"]["miscoverage"]["min"]), set(S.RESIDUALS))
        self.assertEqual(result["doses"]["q1"]["banks"]["iid"]["banks"], 5)

    def test_lambda_star_is_the_shortest_valid_threshold(self):
        for s in ALL:
            scores = self.pop.scores[s, s]
            lam = self.pop.risks[s, s].lambda_star(0.1)
            self.assertLessEqual(np.mean(scores > lam), 0.1)
            self.assertGreater(np.mean(scores > np.max(scores[scores < lam])), 0.1)

    def test_equal_heads_and_scales_make_the_signals_identical(self):
        heads = {k: v.copy() for k, v in self.heads.items()}
        heads["q"][:, 0] = heads["q"][:, 1]
        for s in ALL:
            heads["sigma_" + s] = heads["sigma_min"]
        pop = S.population(heads, self.episode, self.timestep, ALL)
        _, thresholds, miscoverage = S.run_banks(pop, DESIGNS[1], 10, self.seed, OPTIONS)
        for i in (1, 2):  # common posterior draws: identical scores give identical thresholds and coverage
            np.testing.assert_array_equal(thresholds[:, i], thresholds[:, 0])
            np.testing.assert_array_equal(miscoverage[:, i], miscoverage[:, 0])

    def test_dose_is_bcas_closed_form(self):
        lam = {s: self.pop.risks[s, s].lambda_star(0.1) for s in ALL}
        table = np.array([[lam["min"] * 1.1, np.inf, lam["maxabs"]], [lam["min"], lam["q1"] * 0.9, np.inf]])
        doses = S.dose_tables(self.heads, ALL, lam, {"x": table}, 0.5)
        for i, s in enumerate(ALL):
            eta = np.maximum(self.heads["eta_" + s].astype(np.float64), 1e-6)

            def closed(threshold):
                le = np.float64(np.float32(threshold)) * eta  # the reference stores the threshold in float32
                return 1.0 + 0.5 * le / (1.0 + le)

            exact = closed(lam[s])
            got = doses[s]["lambda_star"]
            self.assertAlmostEqual(got["mean"], exact.mean(), delta=1e-6)
            self.assertAlmostEqual(got["sd"], exact.std(), delta=1e-6)
            self.assertAlmostEqual(got["p05"], np.quantile(exact, 0.05), delta=1e-6)
            self.assertAlmostEqual(got["p95"], np.quantile(exact, 0.95), delta=1e-6)
            self.assertTrue(1.0 <= got["p05"] <= got["p95"] < 1.5)
            per_bank = [closed(th) if np.isfinite(th) else np.ones_like(eta) for th in table[:, i]]  # abstention: dose 1
            bank = doses[s]["banks"]["x"]
            self.assertAlmostEqual(bank["mean"], np.mean([d.mean() for d in per_bank]), delta=1e-6)
            self.assertAlmostEqual(bank["sd"], np.mean([d.std() for d in per_bank]), delta=1e-6)
            self.assertAlmostEqual(bank["p95"], np.mean([np.quantile(d, 0.95) for d in per_bank]), delta=1e-6)


class FrozenSignalFitTests(unittest.TestCase):
    """The fit on a synthetic short-train pool: identity with fit_scale, shared inputs, artifacts and evaluate."""

    @classmethod
    def setUpClass(cls):
        from experiments.wbcp import freeze_scores as F
        from experiments.wbcp.test_freeze_scores import synthetic_row, write_raw

        cls.tmp = Path(tempfile.mkdtemp(prefix="frozen-signals-"))
        cls.addClassCleanup(shutil.rmtree, cls.tmp, ignore_errors=True)
        write_raw(cls.tmp / "synthetic.hdf5")
        F.freeze_short_train(synthetic_row(cls.tmp), cls.tmp / "pool", updates=20, train_fraction=0.5, block=10,
                             score_batch=10**6, data_dir=cls.tmp)
        cls.result = S.fit(cls.tmp / "pool", cls.tmp / "fit", signals=ALL, updates=UPDATES, block=BLOCK,
                           data_dir=cls.tmp)
        cls.alone = S.fit(cls.tmp / "pool", cls.tmp / "fit_q1", signals=("q1",), updates=UPDATES, block=UPDATES,
                          data_dir=cls.tmp)
        cls.pool = cls.result["pool"]

    def test_min_fit_is_fit_scale(self):
        import jax

        import algorithms.td3_bc_bca as P

        p = self.pool
        args, config, models, training = p["args"], p["config"], p["models"], p["prepared"].training
        fresh = p["state"]._replace(calibrator=p["template"].calibrator, residual_scale=p["template"].residual_scale)
        kb, kn = jax.random.split(jax.random.PRNGKey(3))
        batch = jax.tree_util.tree_map(lambda x: x[jax.random.randint(kb, (args.batch_size,), 0, len(training.obs))],
                                       training)
        expected, diag = P.fit_scale(args, config, models, fresh, batch, kn, p["prepared"].max_action)
        got, mine = S.fit_signal_scale(args, config, models, fresh, batch, kn, "min", p["prepared"].max_action)
        self.assertTrue(same_tree((expected.calibrator, expected.residual_scale), (got.calibrator, got.residual_scale)))
        self.assertEqual(float(diag["scale_loss"]), float(mine["scale_loss"]))

        def reference(carry, _):  # freeze_scores.train_arm's stream with fit_scale alone
            rng, state = carry
            kr, kb, kn = jax.random.split(rng, 3)
            idx = jax.random.randint(kb, (args.batch_size,), 0, len(training.obs))
            state, _ = P.fit_scale(args, config, models, state, jax.tree_util.tree_map(lambda x: x[idx], training), kn,
                                   p["prepared"].max_action)
            return (kr, state), idx

        (_, state), idx = jax.jit(lambda c: jax.lax.scan(reference, c, None, length=UPDATES))((p["rng"], fresh))
        calibrator, unit = self.result["scales"]["min"]  # fit jointly with q1 and maxabs, in blocks of 15
        self.assertTrue(same_tree((state.calibrator, state.residual_scale), (calibrator, unit)))
        import hashlib

        self.assertEqual(hashlib.sha256(np.ascontiguousarray(idx, "<i4").tobytes()).hexdigest(),
                         self.result["record"]["fingerprints"]["batch_indices_sha256"])

    def test_signals_share_batches_and_targets(self):
        joint, alone = self.result["record"]["fingerprints"], self.alone["record"]["fingerprints"]
        for name in ("batch_indices_sha256", "targets_sha256"):
            self.assertEqual(joint[name], alone[name])
        self.assertEqual(joint["accepted_updates"], {s: UPDATES for s in ALL})
        # a signal fit alone (one compiled program) or with the others (another) ends in the same state
        self.assertTrue(same_tree(self.result["scales"]["q1"], self.alone["scales"]["q1"]))
        blocks = self.result["record"]["blocks"]
        self.assertEqual([b["step"] for b in blocks], [15, 30, 40])
        self.assertFalse(same_tree(self.result["scales"]["min"], self.result["scales"]["q1"]))

    def test_artifacts_hold_each_signal(self):
        from experiments.wbcp import freeze_scores as F
        from runtime.provenance import sha

        scored, parent = self.result["scored"], self.result["parent"]
        t, q = scored["target"], scored["q"]
        for s in ALL:
            directory = self.tmp / "fit" / s
            with np.load(directory / "frozen.npz") as archive:
                arrays = {name: archive[name] for name in archive.files}
            F.check_contract(arrays)
            np.testing.assert_array_equal(arrays["residual"], (t - S.signal_value(s, t, q)).astype(np.float64))
            np.testing.assert_array_equal(arrays["sigma"], scored["sigma_" + s].astype(np.float64))
            for name in ("obs", "action", "policy_action", "episode", "timestep", "row", "in_training"):
                np.testing.assert_array_equal(arrays[name], parent[name])
            meta = json.loads((directory / "frozen.json").read_text(encoding="utf8"))
            self.assertEqual((meta["schema"], meta["signal"], meta["source"]), (S.SCHEMA, s, "frozen-signal"))
            self.assertEqual(meta["npz_sha256"], sha(directory / "frozen.npz"))
            self.assertEqual(meta["parent"]["npz_sha256"], sha(self.tmp / "pool" / "frozen.npz"))
            self.assertEqual(meta["parent"]["frozen_json_sha256"], sha(self.tmp / "pool" / "frozen.json"))
            self.assertEqual(meta["parent"]["checkpoint"]["sha256"], sha(self.tmp / "pool" / "checkpoint_20.msgpack"))
            self.assertEqual((meta["sigma_fit"]["updates"], meta["sigma_fit"]["batch_size"]), (UPDATES, 256))
            self.assertEqual(meta["residual_unit"], float(self.result["scales"][s][1]))
            self.assertEqual(max(meta["gate"].values()), 0.0)
            pool = B.load_pool(str(directory), "heldout")
            np.testing.assert_array_equal(pool.scores["normalized"], arrays["score"])
        min_arrays = np.load(self.tmp / "fit" / "min" / "frozen.npz")
        np.testing.assert_array_equal(min_arrays["residual"], parent["residual"])  # the min residual is the parent's
        maxabs = np.load(self.tmp / "fit" / "maxabs" / "frozen.npz")
        np.testing.assert_array_equal(np.abs(maxabs["residual"]),
                                      np.maximum(np.abs((t - q[:, 0]).astype(np.float64)),
                                                 np.abs((t - q[:, 1]).astype(np.float64))))

    def test_scale_state_restores(self):
        from flax import serialization

        p, scored = self.pool, self.result["scored"]
        template = {"calibrator": p["template"].calibrator, "residual_scale": p["template"].residual_scale}
        for s in ALL:
            restored = serialization.from_bytes(template, (self.tmp / "fit" / s / "scale.msgpack").read_bytes())
            self.assertTrue(same_tree((restored["calibrator"], restored["residual_scale"]), self.result["scales"][s]))
            data = p["prepared"].heldout
            eta = np.asarray(p["models"][2].apply(restored["calibrator"].params, data.obs, data.action))
            np.testing.assert_array_equal(eta, scored["eta_" + s])

    def test_benchmark_runs_unchanged_on_an_artifact(self):
        args = B.parse_args(["--frozen", str(self.tmp / "fit" / "q1"), "--n", "30", "--trials", "2", "--draws", "50",
                             "--weight-fit", "40", "--test-mass-size", "40"])
        result = B.run(args)
        self.assertEqual(result["frozen"]["signal"], "q1")
        self.assertEqual(result["population"]["rows"], len(self.result["scored"]["target"]))

    def test_evaluate_writes_every_preregistered_quantity(self):
        designs = [dict(name="iid", label="iid", n=40, per_episode=None, spacing="random"),
                   dict(name="reservation", label="resv", n=12, per_episode=3, spacing="reservation")]
        out = self.tmp / "evaluation.json"
        result, path = S.evaluate(self.tmp / "fit", banks=4, seed=11, designs=designs, output=out)
        self.assertEqual(path, out)
        saved = json.loads(out.read_text(encoding="utf8"))
        self.assertEqual(saved["schema"], S.EVALUATION_SCHEMA)
        for block in saved["designs"]:  # items 1-4: every band against every residual, with CIs
            for s in ALL:
                for r in S.RESIDUALS:
                    entry = block["signals"][s]["failure"][r]
                    self.assertEqual(set(entry) >= {"fail", "ci", "failures", "trials", "mean_miscoverage"}, True)
                self.assertIn("threshold_over_lambda_star", block["signals"][s])
                self.assertIn("half_width_mean", block["signals"][s])
            for pair in ("maxabs/q1", "q1/min", "min/q1"):  # item 5
                self.assertIn(pair, block["width_ratio"])
        for s in ALL:  # item 6 and Addendum A2
            self.assertEqual(set(saved["doses"][s]["lambda_star"]), {"mean", "sd", "p05", "p95"})
            self.assertEqual(set(saved["doses"][s]["banks"]), {"iid", "reservation"})
        self.assertIn("gap_over_residual", saved["head_disagreement"])  # B2
        self.assertIn("status", saved["critic_health"])  # B4
        self.assertEqual(saved["settings"]["fit"]["accepted_updates"], {s: UPDATES for s in ALL})
        with self.assertRaises(FileExistsError):
            S.evaluate(self.tmp / "fit", banks=1, seed=11, designs=designs, output=out)

    def test_fit_refuses_an_existing_directory_and_bad_signals(self):
        with self.assertRaises(FileExistsError):
            S.fit(self.tmp / "pool", self.tmp / "fit", updates=1, data_dir=self.tmp)
        with self.assertRaises(ValueError):
            S.fit(self.tmp / "pool", self.tmp / "bad", signals=("q2",), updates=1, data_dir=self.tmp)


if __name__ == "__main__":
    unittest.main()

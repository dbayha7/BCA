"""Checks for experiments/wbcp/d4rl_benchmark.py on synthetic frozen artifacts; NumPy and SciPy
only, no D4RL files and no RL imports.

python -m unittest experiments.wbcp.test_d4rl_benchmark
"""
import contextlib
import io
import json
import math
import os
import tempfile
import unittest

import numpy as np
from scipy import stats

from calibration import bank
from experiments.wbcp import d4rl_benchmark as bench


def write_synthetic_frozen(directory, episodes=300, seed=0, obs_dim=5, action_dim=2, lengths=(20, 120),
                           heldout_fraction=0.4, response_seed=None):
    """A frozen.npz/frozen.json pair in the wbcp-frozen-scores-v1 layout.

    Observations follow a stationary AR(1) within each episode, a deterministic actor
    gives policy_action, and half the episodes explore with wide action noise. The
    residual scale grows with obs[0] and is largest on the actor's own actions; the
    frozen scale learns only part of the obs[0] effect, so normalized scores stay
    heteroskedastic in both obs and action. response_seed redraws only the residuals.
    """
    rng = np.random.default_rng(seed)
    length = rng.integers(lengths[0], lengths[1] + 1, episodes)
    rows = int(length.sum())
    starts = np.concatenate([[0], np.cumsum(length)[:-1]])
    episode = np.repeat(np.arange(episodes), length)
    timestep = np.arange(rows) - np.repeat(starts, length)
    noise = rng.normal(size=(rows, obs_dim))
    obs = noise.copy()
    for t in range(1, int(length.max())):
        index = starts[length > t] + t
        obs[index] = 0.9 * obs[index - 1] + math.sqrt(1 - 0.81) * noise[index]
    policy_action = np.tanh(obs @ rng.normal(size=(obs_dim, action_dim)) / math.sqrt(obs_dim))
    spread = np.repeat(np.where(rng.random(episodes) < 0.5, 0.6, 0.05), length)
    action = np.clip(policy_action + spread[:, None] * rng.normal(size=(rows, action_dim)), -1, 1)
    in_training = np.repeat(rng.random(episodes) >= heldout_fraction, length)
    deviation = np.linalg.norm(action - policy_action, axis=1)
    response = rng if response_seed is None else np.random.default_rng(response_seed)
    residual = np.exp(0.5 * obs[:, 0]) * (0.5 + 1.5 * np.exp(-deviation / 0.15)) * response.normal(size=rows)
    sigma = 0.9 * np.exp(0.3 * obs[:, 0])
    np.savez(os.path.join(directory, "frozen.npz"), score=np.abs(residual) / sigma, residual=residual,
             sigma=sigma, obs=obs.astype(np.float32), action=action.astype(np.float32),
             policy_action=policy_action.astype(np.float32), episode=episode.astype(np.int64),
             timestep=timestep.astype(np.int64), row=np.arange(rows, dtype=np.int64), in_training=in_training)
    metadata = dict(schema=bench.SCHEMA, algorithm="synthetic", dataset="synthetic-v0", source="synthetic",
                    seed=seed, rows=dict(total=rows, heldout=int((~in_training).sum())))
    with open(os.path.join(directory, "frozen.json"), "w") as handle:
        json.dump(metadata, handle)
    return directory


def benchmark(directory, *extra):
    argv = ["--frozen", directory, "--n", "200", "--draws", "400", "--weight-fit", "300",
            "--test-mass-size", "300", *extra]
    return bench.run(bench.parse_args(argv))


def block(result, **match):
    found = [b for b in result["blocks"] if all(b[key] == value for key, value in match.items())]
    assert len(found) == 1, (match, len(found))
    return found[0]


class ExactRiskTests(unittest.TestCase):
    def test_matches_brute_force_on_random_pools(self):
        generator = np.random.default_rng(0)
        for case in range(40):
            size = int(generator.integers(1, 400))
            scores = np.round(generator.exponential(size=size), int(generator.integers(0, 3)))  # ties
            mass = generator.lognormal(size=size) * (generator.random(size) > 0.2)
            mass[generator.integers(size)] = 1.0  # positive total
            risk = bench.ExactRisk(scores, mass)
            unique = np.unique(scores)
            grid = np.concatenate([scores, (unique[1:] + unique[:-1]) / 2, [unique[0] - 1, unique[-1] + 1, np.inf]])
            brute = np.array([mass[scores > level].sum() / mass.sum() for level in grid])
            np.testing.assert_allclose(risk(grid), brute, rtol=1e-12, atol=1e-15, err_msg=f"case {case}")
            for alpha in (0.02, 0.1, 0.37):
                exact = [s for s in unique if mass[scores > s].sum() / mass.sum() <= alpha]
                with self.subTest(case=case, alpha=alpha):
                    self.assertEqual(risk.lambda_star(alpha), min(exact))
                    # a threshold violates iff it lies below lambda*
                    self.assertTrue(np.array_equal(risk(scores) > alpha, scores < risk.lambda_star(alpha)))

    def test_infinite_threshold_has_zero_risk(self):
        risk = bench.ExactRisk(np.array([1.0, 2.0, 3.0]), np.array([1.0, 2.0, 3.0]))
        self.assertEqual(risk([np.inf])[0], 0.0)
        self.assertEqual(risk([0.0])[0], 1.0)
        self.assertAlmostEqual(risk([1.5])[0], 5 / 6)


class KishTests(unittest.TestCase):
    def test_effective_size_is_scale_free_down_to_subnormal_weights(self):
        weights = np.array([1.0, 2.0, 3.0, 0.5])
        expected = weights.sum() ** 2 / np.dot(weights, weights)
        for scale in (1.0, 1e-300, 5e-320):  # 5e-320: squares underflow to zero (the halfcheetah shift crash)
            with self.subTest(scale=scale):
                self.assertAlmostEqual(bench._kish(weights * scale), expected, delta=1e-3 if scale < 1e-310 else 1e-12)
        self.assertEqual(bench._kish(np.zeros(3)), 0.0)


class TiltTests(unittest.TestCase):
    def test_gamma_zero_is_the_uniform_law(self):
        z = bench.standardize(np.random.default_rng(1).normal(size=997))
        tilt = bench.make_tilt("policy", z, 0.0)
        self.assertTrue(np.all(tilt.mass == 1.0))
        self.assertEqual(tilt.oracle_wbar, 1.0)
        self.assertEqual(tilt.n_eff, 997.0)
        np.testing.assert_array_equal(tilt.oracle_ratio(np.arange(997)), np.ones(997))
        scores = np.random.default_rng(2).exponential(size=997)
        k = math.ceil(0.9 * 997)  # uniform: the k-th order statistic has risk (N - k) / N <= alpha
        self.assertEqual(bench.ExactRisk(scores, tilt.mass).lambda_star(0.1), np.sort(scores)[k - 1])

    def test_oracle_test_mass_is_the_test_expectation_of_the_oracle_ratio(self):
        generator = np.random.default_rng(3)
        z = bench.standardize(generator.gamma(2.0, size=5000))
        for gamma in (-1.5, 0.3, 1.0, 2.5):
            with self.subTest(gamma=gamma):
                tilt = bench.make_tilt("density", z, gamma)
                a = np.exp(gamma * z)
                ratio = a / a.mean()
                np.testing.assert_allclose(tilt.oracle_ratio(np.arange(z.size)), ratio, rtol=1e-12)
                self.assertAlmostEqual(np.mean(ratio), 1.0, places=12)  # E_cal[w*] = 1
                direct = np.sum(a / a.sum() * ratio)  # E_test[w*] over the tilted pool law
                self.assertAlmostEqual(tilt.oracle_wbar, direct, places=10)
                self.assertAlmostEqual(tilt.oracle_wbar, np.mean(a ** 2) / np.mean(a) ** 2, places=10)
                self.assertAlmostEqual(tilt.n_eff, z.size / tilt.oracle_wbar, places=6)
                draws = tilt.sample(np.random.default_rng(4), 200_000)
                values = ratio[draws]
                self.assertLess(abs(values.mean() - direct), 5 * values.std() / math.sqrt(draws.size))

    def test_sampler_follows_the_tilted_masses(self):
        mass = np.array([0.0, 1.0, 3.0, 0.0, 6.0])
        tilt = bench.make_tilt("state", np.log(np.maximum(mass, 1e-300)), 1.0)
        counts = np.bincount(tilt.sample(np.random.default_rng(5), 100_000), minlength=5)
        self.assertEqual(counts[0] + counts[3], 0)
        self.assertGreater(stats.chisquare(counts[[1, 2, 4]], 1e5 * mass[[1, 2, 4]] / 10).pvalue, 1e-4)

    def test_tilts_depend_on_state_and_action_only(self):
        with tempfile.TemporaryDirectory() as first, tempfile.TemporaryDirectory() as second:
            pools = [bench.load_pool(write_synthetic_frozen(first, episodes=80)),
                     bench.load_pool(write_synthetic_frozen(second, episodes=80, response_seed=99))]
            self.assertFalse(np.allclose(pools[0].scores["normalized"], pools[1].scores["normalized"]))
            for kind, options in (("policy", {}), ("density", dict(knn_k=5, knn_reference=500)),
                                  ("state", {}), ("state", dict(state_feature="2"))):
                with self.subTest(kind=kind, options=options):
                    masses = [bench.make_tilt(kind, bench.standardize(bench.tilt_feature(
                        kind, p.obs, p.action, p.policy_action, **options)), 1.3).mass for p in pools]
                    np.testing.assert_array_equal(masses[0], masses[1])

    def test_density_skips_zero_distance_reference_points(self):
        points = np.array([[0.0], [1.0], [3.0], [3.0], [7.0]])
        # positive distances only: row 2 at 3.0 skips both copies of 3.0 in the reference
        np.testing.assert_array_equal(bench.knn_distance(points, points, 1), [1.0, 1.0, 2.0, 2.0, 4.0])
        np.testing.assert_array_equal(bench.knn_distance(points, points, 2), [3.0, 2.0, 3.0, 3.0, 4.0])
        # 3.0 has two zero-distance copies, forcing a wider second query
        np.testing.assert_array_equal(bench.knn_distance(np.array([[2.0], [3.0]]), points, 3), [1.0, 4.0])


class DiscriminatorTests(unittest.TestCase):
    def test_feature_logistic_recovers_a_well_specified_ratio(self):
        generator = np.random.default_rng(6)
        z = bench.standardize(generator.normal(size=200_000))
        tilt = bench.make_tilt("policy", z, 0.8)
        grid = np.linspace(-2, 2, 9)
        truth = np.exp(0.8 * grid) / np.mean(np.exp(0.8 * z))
        for cal_size, test_size in ((20_000, 20_000), (40_000, 10_000)):  # class-prior correction
            with self.subTest(cal_size=cal_size, test_size=test_size):
                cal, test = generator.integers(z.size, size=cal_size), tilt.sample(generator, test_size)
                log_ratio = bench.logistic_log_ratio(z[cal, None], z[test, None], ridge=0.0)
                np.testing.assert_allclose(np.exp(log_ratio(grid[:, None])), truth, rtol=0.08)

    def test_ridge_keeps_separable_data_finite(self):
        cal, test = np.array([[-2.0], [-1.0]]), np.array([[1.0], [2.0]])
        log_ratio = bench.logistic_log_ratio(cal, test, ridge=1e-3)
        values = log_ratio(np.array([[-2.0], [2.0]]))
        self.assertTrue(np.all(np.isfinite(values)) and values[1] > values[0])


class SamplingTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.pool = bench.load_pool(write_synthetic_frozen(self.directory.name, episodes=120))

    def tearDown(self):
        self.directory.cleanup()

    def test_iid_draws_are_uniform_rows(self):
        rows = bench.draw_calibration(np.random.default_rng(7), 250, self.pool.size)
        self.assertEqual(rows.shape, (250,))
        self.assertTrue(0 <= rows.min() and rows.max() < self.pool.size)

    def test_blocks_draw_whole_episodes(self):
        groups = bench.episode_groups(self.pool.episode, self.pool.timestep)
        lengths = np.bincount(self.pool.episode)
        for seed in range(20):
            rows = bench.draw_calibration(np.random.default_rng(seed), 300, self.pool.size, groups)
            ids = self.pool.episode[rows]
            last = ids[-1]
            self.assertTrue(300 <= rows.size < 300 + lengths[last])
            counts = np.bincount(ids, minlength=lengths.size)
            drawn = np.flatnonzero(counts)
            np.testing.assert_array_equal(counts[drawn] % lengths[drawn], 0)  # each draw is a whole episode
            # consecutive rows of one drawn episode run through its timesteps in order
            starts = np.flatnonzero(np.concatenate([[True], self.pool.timestep[rows][1:] == 0]))
            for begin, end in zip(starts, np.append(starts[1:], rows.size)):
                np.testing.assert_array_equal(self.pool.timestep[rows[begin:end]], np.arange(end - begin))
                self.assertEqual(np.unique(ids[begin:end]).size, 1)

    def test_per_episode_bank_draws_k_rows_from_each_episode_draw(self):
        # n = 105 is a multiple of every K here, so no row is trimmed and the slots stay in blocks
        groups = bench.episode_groups(self.pool.episode, self.pool.timestep)
        for k in (1, 3, 7):
            rows = bench.draw_calibration(np.random.default_rng(k), 105, self.pool.size, groups, per_episode=k)
            self.assertEqual(rows.shape, (105,))
            ids = self.pool.episode[rows]
            for start in range(0, 105, k):  # every block of k slots comes from one episode draw
                self.assertEqual(np.unique(ids[start:start + k]).size, 1)

    def test_without_a_remainder_per_episode_draws_are_unchanged_bit_for_bit(self):
        # The sampler before the remainder trim cut the last draw's final slots; with K | n it
        # cut nothing, and the trim must then draw nothing either.
        def truncated(rng, n, groups, k, spacing):
            order, starts, lengths = groups
            draws = -(-n // k)
            episodes = rng.choice(len(starts), size=draws, p=lengths / lengths.sum())
            position = rng.random((draws, k))
            if spacing == "stratified":
                position = (np.arange(k) + position) / k
            offsets = np.minimum((position * lengths[episodes, None]).astype(np.int64), lengths[episodes, None] - 1)
            return order[starts[episodes, None] + offsets].ravel()[:n]

        groups = bench.episode_groups(self.pool.episode, self.pool.timestep)
        for spacing in ("random", "stratified"):
            for n, k in ((1024, 2), (1025, 5), (115, 23), (100, 1)):
                new, old = np.random.default_rng(n + k), np.random.default_rng(n + k)
                np.testing.assert_array_equal(
                    bench.draw_calibration(new, n, self.pool.size, groups, per_episode=k, spacing=spacing),
                    truncated(old, n, groups, k, spacing))
                self.assertEqual(new.bit_generator.state, old.bit_generator.state)

    def test_every_per_episode_spacing_keeps_n_rows_and_every_position_equally(self):
        # K = 23 does not divide n = 1,024 (45 draws, 1,035 rows). With every episode longer than
        # K each pool row has expected count n / N in all three spacings, so each relative-position
        # bin expects n times its share of pool rows. Cutting the last draw's final slots left
        # late positions short in 'stratified'; the untrimmed reservation sampler held 1,035 rows.
        lengths = np.random.default_rng(8).integers(40, 121, size=300)
        starts = np.concatenate([[0], np.cumsum(lengths)[:-1]])
        groups, bins, n, trials = (np.arange(lengths.sum()), starts, lengths), 4, 1024, 1500
        position = (np.arange(lengths.sum()) - np.repeat(starts, lengths) + 0.5) / np.repeat(lengths, lengths)
        which = np.minimum((position * bins).astype(np.int64), bins - 1)
        expected = n * np.bincount(which, minlength=bins) / lengths.sum()
        for spacing in ("random", "stratified", "reservation"):
            counts = np.empty((trials, bins))
            for seed in range(trials):
                rows = bench.draw_calibration(np.random.default_rng(seed), n, lengths.sum(), groups,
                                              per_episode=23, spacing=spacing)
                self.assertEqual(rows.size, n)
                counts[seed] = np.bincount(which[rows], minlength=bins)
            z = (counts.mean(0) - expected) / (counts.std(0, ddof=1) / np.sqrt(trials))
            with self.subTest(spacing=spacing):
                self.assertLess(float(np.max(np.abs(z))), 4.5, (counts.mean(0), expected))

    def test_per_episode_rows_are_marginally_uniform_over_the_pool(self):
        # Each slot is uniform over pool rows iff the episode draw is proportional to length and
        # the position inside it is uniform. Rows of one draw share an episode, so row counts are
        # over-dispersed; test the two stages on independent units instead.
        groups = bench.episode_groups(self.pool.episode, self.pool.timestep)
        lengths = np.bincount(self.pool.episode).astype(float)
        draws, positions = np.zeros(lengths.size), []
        jitter = np.random.default_rng(99)
        for seed in range(400):
            rows = bench.draw_calibration(np.random.default_rng(seed), 500, self.pool.size, groups, per_episode=5)
            ids = self.pool.episode[rows]
            draws += np.bincount(ids[::5], minlength=lengths.size)  # one episode per block of 5 slots
            positions.append((self.pool.timestep[rows] + jitter.random(rows.size)) / lengths[ids])
        present = lengths > 0  # episode ids outside the held-out pool have no rows
        expected = draws.sum() * lengths[present] / lengths[present].sum()
        self.assertGreater(stats.chisquare(draws[present], expected).pvalue, 1e-3)
        # Given its episode, a slot's jittered relative position is Uniform(0, 1).
        self.assertGreater(stats.kstest(np.concatenate(positions), "uniform").pvalue, 1e-3)

    def test_stratified_rows_sit_one_per_segment_and_stay_uniform(self):
        groups = bench.episode_groups(self.pool.episode, self.pool.timestep)
        lengths = np.bincount(self.pool.episode).astype(float)
        k, positions, jitter = 4, [], np.random.default_rng(5)
        for seed in range(300):
            rows = bench.draw_calibration(np.random.default_rng(seed), 400, self.pool.size, groups,
                                          per_episode=k, spacing="stratified")
            ids, steps = self.pool.episode[rows], self.pool.timestep[rows]
            slot = np.tile(np.arange(k), rows.size // k)  # slot j samples the j-th K-th of its episode
            self.assertTrue(np.all(steps >= np.floor(slot * lengths[ids] / k)))
            self.assertTrue(np.all(steps <= np.floor((slot + 1) * lengths[ids] / k)))
            positions.append((steps + jitter.random(rows.size)) / lengths[ids])
        self.assertGreater(stats.kstest(np.concatenate(positions), "uniform").pvalue, 1e-3)
        with self.assertRaises(ValueError):
            bench.draw_calibration(np.random.default_rng(0), 10, self.pool.size, groups, per_episode=2, spacing="even")

    def test_reservation_spacing_runs_bcas_sampler_on_distinct_episodes(self):
        groups = bench.episode_groups(self.pool.episode, self.pool.timestep)
        lengths = np.bincount(self.pool.episode)
        for seed in range(20):
            rows = bench.draw_calibration(np.random.default_rng(seed), 100, self.pool.size, groups,
                                          per_episode=4, spacing="reservation")
            ids = self.pool.episode[rows]
            self.assertEqual(np.unique(rows).size, rows.size)
            drawn = np.unique(ids)
            self.assertEqual(drawn.size, 25)  # ceil(100 / 4) distinct episodes; K | n, so nothing is trimmed
            np.testing.assert_array_equal(np.bincount(ids, minlength=lengths.size)[drawn], np.minimum(4, lengths[drawn]))

    def test_heldout_population_excludes_training_rows(self):
        with np.load(os.path.join(self.directory.name, "frozen.npz")) as archive:
            heldout = int((~archive["in_training"]).sum())
            total = archive["score"].size
        self.assertEqual(self.pool.size, heldout)
        self.assertEqual(bench.load_pool(self.directory.name, "all").size, total)

    def test_rejects_malformed_artifacts(self):
        with tempfile.TemporaryDirectory() as directory:
            write_synthetic_frozen(directory, episodes=10)
            with open(os.path.join(directory, "frozen.json"), "w") as handle:
                json.dump(dict(schema="other"), handle)
            with self.assertRaisesRegex(ValueError, "schema"):
                bench.load_pool(directory)
        with tempfile.TemporaryDirectory() as directory:
            write_synthetic_frozen(directory, episodes=10)
            path = os.path.join(directory, "frozen.npz")
            with np.load(path) as archive:
                data = dict(archive)
            data["score"] = data["score"] * 1.01
            np.savez(path, **data)
            with self.assertRaisesRegex(ValueError, "residual"):
                bench.load_pool(directory)


class BenchmarkTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.directory = tempfile.TemporaryDirectory()
        write_synthetic_frozen(cls.directory.name)
        cls.result = benchmark(cls.directory.name, "--tilt", "policy", "--gamma", "0", "2.5", "--trials", "300")

    @classmethod
    def tearDownClass(cls):
        cls.directory.cleanup()

    def test_exchangeable_control_is_valid(self):
        control = block(self.result, gamma=0.0)
        self.assertEqual(control["oracle_wbar"], 1.0)
        limit = stats.binom.ppf(0.999, 300, 0.05)  # binomial noise around a 5% failure rate
        for arm in ("BQ-CP", "WBCP", "WBCP (oracle w)"):
            with self.subTest(arm=arm):
                self.assertLessEqual(control["arms"][arm]["failures"], limit)

    def test_strong_policy_tilt_breaks_shift_blind_arms(self):
        shifted = block(self.result, gamma=2.5)
        arms = shifted["arms"]
        self.assertGreater(shifted["oracle_wbar"], 1.5)
        self.assertGreater(arms["BQ-CP"]["fail"], 0.6)
        self.assertLessEqual(arms["WBCP (oracle w)"]["failures"], stats.binom.ppf(0.999, 300, 0.05))
        self.assertGreater(arms["BQ-CP"]["fail"] - arms["WBCP (oracle w)"]["fail"], 0.5)
        self.assertGreater(arms["WBCP (oracle w)"]["threshold"], arms["BQ-CP"]["threshold"])

    def test_shift_blind_thresholds_are_shared_across_blocks(self):
        control, shifted = block(self.result, gamma=0.0), block(self.result, gamma=2.5)
        for arm in ("BQ-CP", "RCPS"):
            self.assertEqual(control["arms"][arm]["threshold"], shifted["arms"][arm]["threshold"])

    def test_reproducible_from_seed(self):
        extra = ("--tilt", "policy", "state", "--gamma", "1", "--trials", "12", "--score", "both",
                 "--discriminator", "raw")
        first = benchmark(self.directory.name, *extra, "--seed", "5")
        again = benchmark(self.directory.name, *extra, "--seed", "5")
        forked = benchmark(self.directory.name, *extra, "--seed", "5", "--workers", "3")
        other = benchmark(self.directory.name, *extra, "--seed", "6")
        self.assertEqual(first["blocks"], again["blocks"])
        self.assertEqual(first["blocks"], forked["blocks"])
        self.assertNotEqual(first["blocks"], other["blocks"])
        # a block's numbers do not depend on the rest of the sweep
        alone = benchmark(self.directory.name, "--tilt", "state", "--gamma", "1", "--trials", "12",
                          "--score", "both", "--discriminator", "raw", "--seed", "5")
        self.assertEqual([b for b in first["blocks"] if b["tilt"] == "state"], alone["blocks"])

    def test_arms_report_how_far_banks_exceed_alpha(self):
        arm = block(self.result, gamma=2.5)["arms"]["BQ-CP"]
        self.assertLessEqual(arm["risk_q95"], arm["risk_q99"])
        self.assertGreater(arm["risk_q95"], 0.1)  # shift-blind BQ-CP fails most trials here
        self.assertGreater(arm["excess_given_fail"], 0.0)
        control = block(self.result, gamma=0.0)["arms"]["WBCP (oracle w)"]
        self.assertTrue(control["excess_given_fail"] is None or control["excess_given_fail"] > 0)

    def test_shuffled_tilt_keeps_the_weights_but_drops_their_link_to_the_score(self):
        args = bench.parse_args(["--frozen", self.directory.name, "--tilt", "state", "--gamma", "2"])
        plain = bench.build_setup(args)
        args.shuffle_tilt = 7
        shuffled = bench.build_setup(args)
        np.testing.assert_allclose(np.sort(shuffled.tilts[0].mass), np.sort(plain.tilts[0].mass))
        self.assertFalse(np.array_equal(shuffled.tilts[0].mass, plain.tilts[0].mass))
        permutation = np.random.default_rng(7).permutation(plain.pool.size)
        np.testing.assert_allclose(shuffled.tilts[0].mass, plain.tilts[0].mass[permutation])

    def test_blocks_mode_runs_on_whole_episodes(self):
        result = benchmark(self.directory.name, "--gamma", "1", "--trials", "10", "--blocks", "--tilt", "density",
                           "--knn-k", "5", "--knn-reference", "2000")
        size = result["blocks"][0]["calibration_size"]
        self.assertGreaterEqual(size["min"], 200)
        self.assertGreater(size["max"], 200)

    def test_cli_writes_json_and_refuses_an_existing_output(self):
        with tempfile.TemporaryDirectory() as scratch:
            path = os.path.join(scratch, "out", "bench.json")
            argv = ["--frozen", self.directory.name, "--n", "60", "--trials", "3", "--draws", "100",
                    "--weight-fit", "100", "--test-mass-size", "100", "--output", path]
            with contextlib.redirect_stdout(io.StringIO()):
                bench.main(argv)
            with open(path) as handle:
                saved = json.load(handle)
            self.assertEqual(saved["frozen"]["schema"], bench.SCHEMA)
            self.assertEqual(saved["bank_trim"], bank.REMAINDER_TRIM)  # dependence_evidence.py reads this marker
            self.assertEqual(len(saved["blocks"]), 2)
            self.assertEqual(set(saved["blocks"][0]["arms"]), set(bench.ARMS))
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                bench.parse_args(argv)


if __name__ == "__main__":
    unittest.main()

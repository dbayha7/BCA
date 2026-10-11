"""CPU checks for experiments/truth/acceptance.py and extraction.py, ending in two synthetic end-to-end runs.

Fixtures: synthetic D4RL-format files made in the real simulator (synthetic.py) and train.py-format run
directories around randomly initialised TD3+BC networks of the host's exact architecture
(pools.write_synthetic_run). Every number here is a test fixture on synthetic data, NOT A FINDING. Checks:
  - statistics: partial Spearman, AUC, the cross-validated classifier, block and hierarchical bootstraps, the
    paired z and the strength ladder interpolation on known answers;
  - scorecard known answers: R1, R2 and R3a pass on planted tables built to pass and fail when the planted
    structure is shuffled away;
  - extraction: the actor-only loop through td3_bc_update with bc_multiplier = 1 equals the native
    bc_multiplier=None loop bit for bit (this CPU and XLA build), is deterministic, keeps the critic bit for
    bit, and responds to v;
    the Level-1 operator gives g_Q + b0 = jax.grad of the full-batch actor loss, P0(m) has rho = m rho_host, and
    the closed-form beta reaches its target strength; simulator construction is refused inside forbid_simulator;
  - the pre-registration gate: a wrong hash, an edited body, a mismatched sidecar and a changed source hash are
    refused (the last one unless allow_code_change); freeze refuses a second time and after labels exist, and the
    synthetic_smoke profile refuses real data; training (extract) refuses a wrong hash; R5-signal (BCA's band)
    refuses without --allow-bca-band;
  - Stage 0 on a file without infos/* restores states rebuilt from the observation and passes its gate;
  - end to end, plan 'full' on Hopper with two synthetic hosts: stage0 -> freeze -> run (pools, labels, score,
    screen, extract, evaluate, extraction-score, verdict). Every planned requirement writes its JSON under the same
    pre-registration sha256, marked SYNTHETIC; the matching lock exists and was verified; a second run resumes
    without recomputing; and plan 'graft' on Walker2d (one host) scores exactly R1, R2, R5-world and R6, with no
    pool T and no extraction.
JAX_PLATFORMS=cpu LD_LIBRARY_PATH=$HOME/.mujoco/mujoco210/bin python -m unittest experiments.truth.test_acceptance
"""

import contextlib
import io
import json
import shutil
import tempfile
import unittest
from pathlib import Path

import numpy as np

from experiments.truth import acceptance as A
from experiments.truth import extraction as EX
from experiments.truth import pools as PL
from experiments.truth import state_restore as SR
from experiments.truth import synthetic as SY

SEEDS = (202609171, 202609172)


def quiet():
    return contextlib.redirect_stdout(io.StringIO())


def silent(*_):
    pass


class StatisticsTest(unittest.TestCase):

    def test_known_answers(self):
        g = np.random.default_rng(0)
        z = g.normal(size=2000)
        x, y = z + 0.1 * g.normal(size=2000), z + 0.1 * g.normal(size=2000)
        self.assertGreater(A.spearman(x, y), 0.95)
        self.assertLess(abs(A.partial_spearman(x, y, z)), 0.1)
        w = g.normal(size=2000)
        self.assertAlmostEqual(A.partial_spearman(x, y, w), A.spearman(x, y), delta=0.02)
        self.assertEqual(A.auc([2, 3], [0, 1]), 1.0)
        self.assertEqual(A.auc([0, 1], [2, 3]), 0.0)
        self.assertEqual(A.auc([1, 1], [1, 1]), 0.5)
        far = A.classifier_auc(g.normal(size=(200, 3)) + 3, g.normal(size=(200, 3)), 5, 1e-2, 1)
        same = A.classifier_auc(g.normal(size=(200, 3)), g.normal(size=(200, 3)), 5, 1e-2, 1)
        self.assertGreater(far, 0.95)
        self.assertLess(abs(same - 0.5), 0.15)
        ring = g.normal(size=(300, 2))  # separable only through the squares
        wide = 3 * g.normal(size=(300, 2))
        self.assertGreater(A.classifier_auc(wide, ring, 5, 1e-2, 1), 0.8)
        self.assertEqual(A.hierarchical_ci([[2.0, 2.0], [2.0]], 100, 0), (2.0, 2.0))
        lo, hi = A.hierarchical_ci([[0.0, 1.0], [2.0, 3.0]], 500, 0)
        self.assertTrue(0.0 <= lo <= 1.5 <= hi <= 3.0)
        blocks = np.repeat(np.arange(50), 4)
        vals = A.block_bootstrap(blocks, lambda i: float(len(np.unique(blocks[i]))), 20, 0)
        self.assertTrue(np.all(vals <= 50) and len(vals) == 20)
        self.assertGreater(A.paired_z([1.0, 1.1, 0.9, 1.0]), 3)
        self.assertEqual(A.paired_z([1.0, 1.0]), float("inf"))

    def test_match_ladder(self):
        beta, rule = EX.match_ladder([1, 2, 4], [0.1, 0.2, 0.4], 0.2)
        self.assertAlmostEqual(beta, 2.0)
        beta, _ = EX.match_ladder([1, 2, 4], [0.1, 0.2, 0.4], np.sqrt(0.02))  # log-log midpoint of 1 and 2
        self.assertAlmostEqual(beta, np.sqrt(2.0))
        beta, rule = EX.match_ladder([1, 2], [0.4, 0.8], 0.2)
        self.assertAlmostEqual(beta, 0.5)
        self.assertEqual(rule["rule"], "linear from (0, 0)")
        beta, rule = EX.match_ladder([1, 2], [0.1, 0.2], 0.5)
        self.assertIsNone(beta)
        self.assertFalse(rule["bracketed"])


def planted_tables(n=400, S=9, seed=0, planted=True):
    """Tables shaped like acceptance.truth_tables with errors growing with d_k (planted) or independent of it."""
    g = np.random.default_rng(seed)
    tables = {}
    for name in ("L", "Pi"):
        dk = g.uniform(0.1, 3.0, size=(n, S))
        c = np.where(dk > 2.0, 0, np.where(dk < 1.0, 30, 5)).astype(np.int64)
        scale = dk ** 3 if planted else g.permutation(dk.ravel()).reshape(n, S) ** 3
        q1 = scale * (1 + 0.05 * g.normal(size=(n, S)))
        q1[:, 0] = 0.0
        qpi = np.zeros((n, S))
        length = g.integers(10, 1000, size=(n, S))
        arrays = dict(dk=dk, c=c, episode=np.arange(n) // 4 if name == "L" else -1 - np.arange(n) // 4,
                      in_bank=np.arange(n) < n // 2, residual_b=None, targets=None)
        tables[name] = dict(arrays=arrays, q1=q1, qpi=qpi, length=length, status=np.zeros((n, S), np.int64),
                            gap=np.full((n, S, 2), np.nan), rough=np.full((n, S, 2), np.nan))
    return tables


class ScorecardTest(unittest.TestCase):

    def test_r1_r2_known_answers(self):
        thin = A.PASS_MARKS["thinness"]
        good = planted_tables()
        values, passes = A.score_r2(good, A.PASS_MARKS["R2"], thin, 100, 0)
        self.assertTrue(all(passes.values()), (values, passes))
        bad = planted_tables(planted=False)
        values, passes = A.score_r2(bad, A.PASS_MARKS["R2"], thin, 100, 0)
        self.assertFalse(passes["a"] or passes["d"])
        values, passes = A.score_r1(good, A.PASS_MARKS["R1"], thin)
        self.assertAlmostEqual(values["pi_thin_share"], float(np.mean(good["Pi"]["arrays"]["c"][:, 0] <= 2)))
        self.assertEqual(values["bank_rows"], 200)

    def test_r3a_known_answer(self):
        thin = A.PASS_MARKS["thinness"]
        t = planted_tables(n=200)
        Pi = t["Pi"]
        S = Pi["q1"].shape[1]
        thin_states = Pi["arrays"]["c"][:, 0] <= 2
        Pi["q1"][:, S - 1] = 1.0  # the critic predicts a gain everywhere
        Pi["qpi"][:, S - 1] = np.where(thin_states, -5.0, 0.0)  # the truth loses only in thin states
        for name in ("L", "Pi"):
            t[name]["rough"][:, :2] = np.stack([np.zeros((200, 2)), 0.1 * np.ones((200, 2))], axis=-1)
        values, passes = A.score_r3a(t, A.PASS_MARKS["R3a"], thin)
        self.assertAlmostEqual(values["misled_share"], float(thin_states.mean()))
        self.assertEqual(values["misled_rate_dense"], 0.0)
        self.assertEqual(passes["thin_direction"], True)
        self.assertTrue(passes["thin_vs_dense"])


class HostFixture(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.tmp = Path(tempfile.mkdtemp(prefix="truth-accept-"))
        physics = SR.Physics("hopper")
        (cls.tmp / "data").mkdir()
        meta = SY.write_synthetic(cls.tmp / "data", "hopper", n_rows=1500, seed=5, physics=physics)
        physics.close()
        cls.file = Path(meta["with_infos"])
        with quiet():
            cls.host_run = PL.write_synthetic_run(cls.tmp / "run", "hopper", cls.file, SEEDS[0])
            cls.host = PL.restore_host(cls.host_run["run_dir"], data_dir=cls.file.parent)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)


class ExtractionTest(HostFixture):

    def test_actor_only_loop(self):
        import jax
        import jax.numpy as jnp

        from algorithms.td3_bc import td3_bc_update

        host = self.host
        ids = host.training_ids[:200]
        ex = EX.Extractor(host, ids, batch=32, block=10)
        ones = np.ones(len(ids), np.float32)
        p1, diag = ex.train(ones, 3, 20)
        p2, _ = ex.train(ones, 3, 20)
        self.assertTrue(diag["critic_frozen"])
        for a, b in zip(jax.tree_util.tree_leaves(p1), jax.tree_util.tree_leaves(p2)):
            np.testing.assert_array_equal(np.asarray(a), np.asarray(b))
        p3, _ = ex.train(ones * 4, 3, 20)
        moved = max(float(np.abs(np.asarray(a) - np.asarray(b)).max())
                    for a, b in zip(jax.tree_util.tree_leaves(p1), jax.tree_util.tree_leaves(p3)))
        self.assertGreater(moved, 0)
        args, data, n = ex.args, ex.data, ex.n

        def body(carry, _):  # the same loop with the native bc_multiplier=None path
            state, key, it = carry
            key, kb, kn = jax.random.split(key, 3)
            idx = jax.random.randint(kb, (args.batch_size,), 0, n)
            batch = jax.tree_util.tree_map(lambda x: x[idx], data)
            state, _ = td3_bc_update(args, host.actor.apply, host.critic.apply, state, batch, it, kn, 1.0)
            return (state, key, it + args.policy_freq), None

        key = jax.random.fold_in(jax.random.PRNGKey(3), EX.KEY_EXTRACT)
        (state, _, _), _ = jax.jit(lambda s, k: jax.lax.scan(body, (s, k, jnp.int32(0)), None, length=20))(
            ex._initial(), key)
        for a, b in zip(jax.tree_util.tree_leaves(p1), jax.tree_util.tree_leaves(state.actor.params)):
            np.testing.assert_array_equal(np.asarray(a), np.asarray(b))  # bitwise on this CPU / XLA build

    def test_level1_identities(self):
        import jax
        import jax.numpy as jnp
        from jax.flatten_util import ravel_pytree

        host = self.host
        ids = host.training_ids[:300]
        obs, act = host.converted["observations"][ids], host.converted["actions"][ids]
        g = np.random.default_rng(1)
        screen = EX.Level1Screen(host, obs, act, obs[:50], g.normal(size=(50, 3)), chunk=128)
        s, a = jnp.asarray(host.standardized(obs)), jnp.asarray(act)
        lam = screen.lam

        def loss(p):
            pi = host.actor.apply(p, s)
            q = host.critic.apply(host.native.critic.params, s, pi)[..., 0]
            return -lam * q.mean() + jnp.square(pi - a).mean()

        grad = np.asarray(ravel_pytree(jax.grad(loss)(host.native.actor.params))[0], np.float64)
        total = screen.g_Q + screen.b0
        np.testing.assert_allclose(total, grad, rtol=1e-3, atol=1e-6 * np.abs(grad).max())
        m = np.random.default_rng(2).uniform(0.5, 1.5, size=300)
        b_m = screen.b_of(m[None])[0]
        beta = screen.match_beta(b_m, 2.0)
        self.assertAlmostEqual(np.linalg.norm(screen.b0 + beta * b_m) / np.linalg.norm(screen.g_Q),
                               2.0 * screen.rho_host, places=9)
        self.assertAlmostEqual(float(np.linalg.norm(4 * screen.b0) / np.linalg.norm(screen.g_Q)),
                               4 * screen.rho_host, places=9)
        gains = screen.gains(screen.b0)
        self.assertAlmostEqual(gains["raw"], float(-(screen.grad_J @ screen.b0)))

    def test_forbid_simulator(self):
        with A.forbid_simulator():
            with self.assertRaises(RuntimeError):
                SR.Physics("hopper")
            import gym

            with self.assertRaises(RuntimeError):
                gym.make("hopper-medium-v2")


class PipelineTest(unittest.TestCase):
    """plan 'full' on synthetic Hopper with two random-init hosts."""

    @classmethod
    def setUpClass(cls):
        cls.tmp = Path(tempfile.mkdtemp(prefix="truth-e2e-"))
        physics = SR.Physics("hopper")
        (cls.tmp / "data").mkdir()
        meta = SY.write_synthetic(cls.tmp / "data", "hopper", n_rows=1500, seed=5, physics=physics)
        physics.close()
        cls.file = Path(meta["with_infos"])
        cls.noinfos = Path(meta["without_infos"])
        with quiet():
            cls.runs = {s: PL.write_synthetic_run(cls.tmp / f"run_{s}", "hopper", cls.file, s)["run_dir"]
                        for s in SEEDS}
            cls.world = cls.tmp / "world"
            cls.stage0 = A.stage0(cls.world, "hopper", file=cls.file, run_dir=cls.runs[SEEDS[0]], rows=200,
                                  eps_draws=4)
            cls.sha = A.freeze(cls.world, "hopper", hosts=cls.runs, stage0_dir=cls.world / "stage0",
                               profile="synthetic_smoke", plan="full")
            cls.verdict = A.run_all(cls.world, cls.sha, log=silent)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def test_stage0(self):
        s = self.stage0
        self.assertTrue(s["synthetic"] and s["gate"]["passed"])
        self.assertFalse(s["gate_negative"]["passed"])
        self.assertTrue(s["gate_rebuilt"]["passed"])
        self.assertEqual(s["rebuild"]["rows_identical_except_x"], s["rebuild"]["kept"])
        self.assertTrue(s["split"]["available"] and s["eps_noise"]["available"])
        self.assertEqual(s["split"]["bank_rows"], 48)
        with quiet():  # no infos/* in the file: the gate restores states rebuilt from the observation
            n = A.stage0(self.tmp / "noinfos", "hopper", file=self.noinfos, rows=200)
        self.assertEqual((n["gate"]["source"], n["gate"]["passed"], n["gate_negative"]["passed"]),
                         ("rebuilt", True, False))
        self.assertIsNone(n["gate_rebuilt"])
        self.assertFalse(n["split"]["available"] or n["eps_noise"]["available"])

    def test_every_planned_output(self):
        v = self.verdict
        self.assertEqual(set(v["statuses"]), set(A.PLANS["full"]))
        self.assertTrue(v["complete"] and v["synthetic"])
        self.assertTrue(all(s in ("pass", "fail") for s in v["statuses"].values()))
        for seed in SEEDS:
            for req in A.PER_CRITIC:
                r = A.read_json(A.critic_dir(self.world, seed) / "scores" / f"{req}.json")
                self.assertEqual((r["prereg_sha256"], r["synthetic"]), (self.sha, True))
                self.assertIn("NOT A FINDING", r["note"])
            pools = A.critic_dir(self.world, seed) / "pools"
            L = json.loads((pools / "L.json").read_text())
            self.assertEqual(L["states"], 48 + 24)
            self.assertEqual(len(L["slots"]), 9)
            out = self.world / "extraction" / f"critic_{seed}"
            lock = (out / "matched.json.sha256").read_text().split()[0]
            self.assertEqual(SR.file_sha256(out / "matched.json"), lock)
            evaluation = A.read_json(out / "evaluation.json")
            self.assertTrue(evaluation["matched_lock_verified"])
            matched = A.read_json(out / "matched.json")
            self.assertTrue(matched["simulator_forbidden_during_matching"])
            self.assertEqual(sorted(matched["units"]), ["1", "2"])
            for unit in matched["units"].values():
                self.assertEqual(unit["policies"]["NONE"]["S"], 0.0)
                self.assertTrue(all(p["training"]["critic_frozen"] for p in unit["policies"].values()))
        for req in A.EXTRACTION:
            r = A.read_json(self.world / "extraction" / f"{req}.json")
            self.assertEqual((r["prereg_sha256"], r["synthetic"]), (self.sha, True))
        self.assertTrue(A.read_json(self.world / "extraction" / "R7.json")["passes"]["e"])
        md = (self.world / "verdict.md").read_text()
        self.assertTrue(md.startswith("**SYNTHETIC TEST RUN. NOT A FINDING.**"))

    def test_resume_recomputes_nothing(self):
        before = {p: p.stat().st_mtime_ns for p in self.world.rglob("*.npz")}
        again = A.run_all(self.world, self.sha, log=silent)
        self.assertEqual(again["statuses"], self.verdict["statuses"])
        after = {p: p.stat().st_mtime_ns for p in self.world.rglob("*.npz")}
        self.assertEqual(before, after)

    def copy_world(self, name):
        target = self.tmp / name
        target.mkdir()
        for f in ("prereg.json", "prereg.json.sha256"):
            shutil.copy(self.world / f, target / f)
        return target

    def test_prereg_gate(self):
        with self.assertRaises(A.PreregError):
            A.load_prereg(self.world, "0" * 64)
        with self.assertRaises(A.PreregError):
            A.run_all(self.world, "0" * 64, log=silent)
        with self.assertRaises(A.PreregError):  # training refuses a wrong hash
            A.main(["extract", "--world", str(self.world), "--prereg-sha256", "0" * 64, "--critic", str(SEEDS[0])])
        edited = self.copy_world("edited")
        doc = json.loads((edited / "prereg.json").read_text())
        doc["body"]["pass_marks"]["R4b"]["matched_margin_min"] = 0.0
        text = json.dumps(doc).encode()
        (edited / "prereg.json").write_bytes(text)
        with self.assertRaises(A.PreregError):  # the old hash no longer matches
            A.load_prereg(edited, self.sha)
        digest = A.sha256_hex(text)
        (edited / "prereg.json.sha256").write_text(f"{digest}  prereg.json\n")
        with self.assertRaises(A.PreregError):  # body edited without its embedded hash
            A.load_prereg(edited, digest)
        side = self.copy_world("sidecar")
        (side / "prereg.json.sha256").write_text("f" * 64 + "  prereg.json\n")
        with self.assertRaises(A.PreregError):
            A.load_prereg(side, self.sha)
        code = self.copy_world("code")
        doc = json.loads((code / "prereg.json").read_text())
        doc["body"]["code_sha256"]["experiments/truth/pools.py"] = "0" * 64
        doc["body_sha256"] = A.canonical_sha256(doc["body"])
        text = json.dumps(doc).encode()
        (code / "prereg.json").write_bytes(text)
        digest = A.sha256_hex(text)
        (code / "prereg.json.sha256").write_text(f"{digest}  prereg.json\n")
        with self.assertRaises(A.PreregError):
            A.load_prereg(code, digest)
        body, _, changed = A.load_prereg(code, digest, allow_code_change=True)
        self.assertTrue(changed)
        body, sha, changed = A.load_prereg(self.world, self.sha)
        self.assertEqual((sha, changed, body["synthetic"]), (self.sha, False, True))
        self.assertTrue(body["pass_marks_equal_design"])

    def test_freeze_refusals_and_band_flag(self):
        with self.assertRaises(ValueError):  # labels exist in this world
            A.freeze(self.world, "hopper", hosts=self.runs, stage0_dir=self.world / "stage0", profile="synthetic_smoke")
        fake = self.tmp / "fake_real"
        (fake / "stage0").mkdir(parents=True)
        summary = A.read_json(self.world / "stage0" / "summary.json")
        summary["synthetic"] = False
        (fake / "stage0" / "summary.json").write_text(json.dumps(summary))
        with self.assertRaises(ValueError):
            A.freeze(fake, "hopper", hosts=self.runs, stage0_dir=fake / "stage0", profile="synthetic_smoke")
        body, sha, _ = A.load_prereg(self.world, self.sha)
        with self.assertRaises(A.BandFitDisabled):
            A.stage_r5_signal(self.world, body, sha, SEEDS[0])
        with self.assertRaises(A.BandFitDisabled):
            A.main(["r5-signal", "--world", str(self.world), "--prereg-sha256", self.sha, "--critic", str(SEEDS[0])])
        self.assertFalse((A.critic_dir(self.world, SEEDS[0]) / "scores" / "R5_signal.json").exists())


class GraftTest(unittest.TestCase):
    """plan 'graft' on synthetic Walker2d with one random-init host: R1, R2, R5-world, R6 only."""

    @classmethod
    def setUpClass(cls):
        cls.tmp = Path(tempfile.mkdtemp(prefix="truth-graft-"))
        physics = SR.Physics("walker2d")
        (cls.tmp / "data").mkdir()
        meta = SY.write_synthetic(cls.tmp / "data", "walker2d", n_rows=1500, seed=5, physics=physics)
        physics.close()
        cls.file = Path(meta["with_infos"])
        with quiet():
            run = PL.write_synthetic_run(cls.tmp / "run", "walker2d", cls.file, SEEDS[0])["run_dir"]
            cls.world = cls.tmp / "world"
            A.stage0(cls.world, "walker2d", file=cls.file, run_dir=run, rows=200, eps_draws=4)
            cls.sha = A.freeze(cls.world, "walker2d", hosts={SEEDS[0]: run}, stage0_dir=cls.world / "stage0",
                               profile="synthetic_smoke", plan="graft")
            cls.verdict = A.run_all(cls.world, cls.sha, log=silent)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def test_graft_scope(self):
        v = self.verdict
        self.assertEqual(set(v["statuses"]), {"R1", "R2", "R5_world", "R6"})
        self.assertTrue(v["complete"] and v["synthetic"])
        self.assertIn("External-validity graft", v["meaning"])
        cdir = A.critic_dir(self.world, SEEDS[0])
        self.assertFalse((cdir / "pools" / "T.json").exists())
        self.assertFalse((cdir / "labels" / "T").exists())
        self.assertFalse((self.world / "extraction").exists())
        self.assertEqual(sorted(p.stem for p in (cdir / "scores").glob("*.json")), ["R1", "R2", "R5_world", "R6"])
        L = json.loads((cdir / "pools" / "L.json").read_text())
        self.assertEqual(len(L["slots"]), 15)
        with self.assertRaises(ValueError):  # the graft labels one checkpoint
            A.freeze(self.tmp / "two", "walker2d", hosts={1: "a", 2: "b"}, stage0_dir=self.world / "stage0",
                     profile="synthetic_smoke", plan="graft")


if __name__ == "__main__":
    unittest.main()

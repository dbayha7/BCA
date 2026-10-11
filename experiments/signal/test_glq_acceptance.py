"""Checks of the Gapped LQ acceptance gate (experiments/signal/glq_acceptance.py) on its 'tiny' preset.

What is checked:
- the pre-registration: it round-trips (digest = sidecar = run directory), and every stage's entry point refuses when
  the file is missing, edited without its sidecar, edited with a recomputed sidecar but changed pass marks, or records a
  code hash that differs from the file on disk; a changed dependency is refused unless explicitly allowed (and then
  recorded); it is never overwritten without force; the default path takes only the real preset; the CLI refuses a
  training stage on a tampered file;
- the statistics: AUROC against all pairs (ties count half), Spearman and the partial Spearman against the residual
  definition, the one-sided paired t against scipy, the minimum detectable effect against its noncentral-t power, the
  seed-count rule, the Kish fraction, the 100-step mixture density against scipy, the unittest summary parser;
- the world variants (retunes, ladder rungs, the null world) and the printed plan of each path;
- the population band against a direct per-row computation on the same random numbers; the BCA band refuses without
  its flag and, with it, is only traced (never run);
- end to end, on the tiny preset (2 seeds, 200 updates, 20 Tier A steps): every stage in the gate's order through the
  verdict, each writing strict JSON with a matching sidecar and the pre-registration's digest; per-seed stages resume
  without recomputing; --alpha auto refuses before r3 and while the ladder is unfinished; the blind Tier A record holds
  no return; R5(iii) is recorded as switched off; the verdict table holds every pass mark; the stop rule's outcomes on
  synthetic step statuses. The tiny preset's numbers mean nothing (200 updates); only the machinery is tested.
JAX_PLATFORMS=cpu python -m unittest experiments.signal.test_glq_acceptance
"""
import hashlib
import io
import json
import math
import os
import subprocess
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from dataclasses import replace
from pathlib import Path
from unittest import mock

import numpy as np
from scipy import stats

import jax
import jax.numpy as jnp

from experiments.signal import gapped_lq as G
from experiments.signal import glq_acceptance as A
from experiments.signal import glq_train as T

TINY = A.PRESETS["tiny"]
QUIET = lambda *a, **k: None  # noqa: E731


def strict(path):
    """The file is strict JSON and matches its sidecar."""
    text = Path(path).read_text(encoding="utf8")
    side = Path(str(path) + ".sha256").read_text(encoding="utf8").split()[0]
    json.loads(text, parse_constant=lambda c: (_ for _ in ()).throw(ValueError(c)))
    return side == hashlib.sha256(text.encode("utf8")).hexdigest()


class PreregTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)
        self.path = self.dir / "prereg.json"
        self.digest = A.write_prereg(self.path, TINY)

    def tearDown(self):
        self.tmp.cleanup()

    def rewrite(self, edit):
        payload = json.loads(self.path.read_text(encoding="utf8"))
        edit(payload)
        text = A.canonical(payload)
        self.path.write_text(text, encoding="utf8")
        self.path.with_name(self.path.name + ".sha256").write_text(
            hashlib.sha256(text.encode("utf8")).hexdigest() + "  prereg.json\n", encoding="utf8")

    def test_round_trip(self):
        P = A.load_prereg(self.path, self.dir / "runs")
        self.assertEqual(P.digest, self.digest)
        self.assertEqual(P.digest, hashlib.sha256(self.path.read_bytes()).hexdigest())
        self.assertEqual(P.root, self.dir / "runs" / self.digest[:16])
        self.assertEqual(P.config, TINY)
        self.assertEqual(P.payload["pass_marks"], json.loads(A.canonical(A.PASS_MARKS)))
        self.assertEqual(set(P.payload["code_sha256"]), set(A.CODE_FILES))
        self.assertEqual(P.payload["code_sha256"]["experiments/signal/glq_acceptance.py"],
                         hashlib.sha256(Path(A.__file__).read_bytes()).hexdigest())
        self.assertEqual(P.payload["alpha_ladder"], [2.5, 5.0, 10.0])
        self.assertEqual([r["name"] for r in P.payload["retunes"]], ["thin_share", "wrong_gain"])
        self.assertEqual(P.payload["density_ladder"]["primary"], 0.01)
        self.assertFalse(P.dependency_override)
        real = A.PRESETS["real"]
        self.assertEqual((len(real.lean_seeds), len(real.full_seeds)), (5, 10))

    def test_refusals(self):
        with self.assertRaises(A.PreregError):
            A.load_prereg(self.dir / "missing.json", self.dir)
        with self.assertRaises(FileExistsError):
            A.write_prereg(self.path, TINY)
        text = self.path.read_text(encoding="utf8")
        self.path.write_text(text.replace('"tiny"', '"tinY"'), encoding="utf8")  # edited, sidecar not updated
        with self.assertRaisesRegex(A.PreregError, "sidecar"):
            A.load_prereg(self.path, self.dir)
        A.write_prereg(self.path, TINY, force=True)

        def mark(p):
            p["pass_marks"]["R4_1"]["min"] = 0.01

        self.rewrite(mark)  # sidecar recomputed, marks no longer the code's
        with self.assertRaisesRegex(A.PreregError, "pass marks"):
            A.load_prereg(self.path, self.dir)
        A.write_prereg(self.path, TINY, force=True)

        def code(p):
            p["code_sha256"]["experiments/signal/glq_train.py"] = "0" * 64

        self.rewrite(code)  # as if glq_train.py had changed after the pre-registration
        with self.assertRaisesRegex(A.PreregError, "glq_train.py"):
            A.load_prereg(self.path, self.dir)
        A.write_prereg(self.path, TINY, force=True)

        def dep(p):
            p["dependency_sha256"]["algorithms/td3_bc.py"] = "0" * 64

        self.rewrite(dep)
        with self.assertRaisesRegex(A.PreregError, "td3_bc.py"):
            A.load_prereg(self.path, self.dir)
        self.assertTrue(A.load_prereg(self.path, self.dir, allow_dependency_change=True).dependency_override)

    def test_every_stage_rechecks_on_entry(self):
        """guard: a stage refuses when the file or a hashed code file changed after load_prereg."""
        P = A.load_prereg(self.path, self.dir / "runs")
        self.rewrite(lambda p: p.update(created="later"))
        with self.assertRaisesRegex(A.PreregError, "changed since it was loaded"):
            A.stage_r0r1(P, 0, log=QUIET)
        P = A.load_prereg(self.path, self.dir / "runs")
        P.payload["code_sha256"]["experiments/signal/gapped_lq.py"] = "0" * 64  # as if the file changed meanwhile
        for call in (lambda: A.stage_critics(P, 0, "HOST", log=QUIET), lambda: A.stage_tier_a(P, 0, alpha=2.5),
                     lambda: A.stage_verdict(P, log=QUIET)):
            with self.assertRaisesRegex(A.PreregError, "gapped_lq.py"):
                call()
        self.assertFalse((self.dir / "runs").exists())

    def test_cli_refuses_training_stage_and_default_path(self):
        def code(p):
            p["code_sha256"]["experiments/signal/gapped_lq.py"] = "f" * 64

        self.rewrite(code)
        with self.assertRaises(A.PreregError):
            A.main(["critics", "--seeds", "0", "--mode", "HOST", "--prereg", str(self.path), "--runs",
                    str(self.dir / "runs")])
        with self.assertRaises(SystemExit):
            A.main(["prereg", "--preset", "tiny"])  # the default path takes the real preset only
        self.assertFalse((self.dir / "runs").exists())

    def test_edits_with_a_recomputed_sidecar_are_refused(self):
        """A-review regression: every code-determined field (seeds, sizes, updates, alpha ladder, primary rung,
        retunes, world, training settings) and the dependency list must be what the hashed code writes; before the fix
        an edited config with a recomputed sidecar ran (e.g. lean seeds (7, 8), alpha ladder (10, 20, 40))."""
        edits = {
            "config": lambda p: p["config"].update(lean_seeds=[7, 8]),
            "alpha": lambda p: p["config"].update(alpha_ladder=[10.0, 20.0, 40.0]),
            "rung": lambda p: p["config"].update(primary_rung=0.3),
            "updates": lambda p: p["config"].update(updates=50),
            "alpha_ladder": lambda p: p.update(alpha_ladder=[10.0]),
            "retunes": lambda p: p["retunes"].pop(),
            "world": lambda p: p["world"]["spec"].update(behaviour_fraction=0.4),
            "train_settings": lambda p: p["train_settings"].update(policy_noise=0.0),
            "statement": lambda p: p.update(statement="anything goes"),
        }
        for name, edit in edits.items():
            A.write_prereg(self.path, TINY, force=True)
            self.rewrite(edit)
            with self.assertRaisesRegex(A.PreregError, "differs from what the hashed code writes", msg=name):
                A.load_prereg(self.path, self.dir)
        A.write_prereg(self.path, TINY, force=True)
        self.rewrite(lambda p: p["dependency_sha256"].pop("algorithms/td3_bc.py"))
        with self.assertRaisesRegex(A.PreregError, "dependency list"):
            A.load_prereg(self.path, self.dir, allow_dependency_change=True)
        A.write_prereg(self.path, TINY, force=True)
        self.rewrite(lambda p: p["config"].update(preset="real"))  # a tiny file relabelled as the real gate
        with self.assertRaisesRegex(A.PreregError, "config"):
            A.load_prereg(self.path, self.dir)
        A.write_prereg(self.path, TINY, force=True)
        self.rewrite(lambda p: p.update(created="2026-01-01T00:00:00", numpy="0.0"))  # informational fields only
        self.assertEqual(A.load_prereg(self.path, self.dir).config, TINY)


class StatisticsTests(unittest.TestCase):
    def test_auroc(self):
        rng = np.random.default_rng(0)
        x = np.round(rng.standard_normal(60), 1)  # ties
        y = rng.random(60) < 0.4
        pos, neg = x[y], x[~y]
        brute = np.mean([(p > q) + 0.5 * (p == q) for p in pos for q in neg])
        self.assertAlmostEqual(A.auroc(x, y), brute, places=12)
        self.assertIsNone(A.auroc(x, np.zeros(60, bool)))
        self.assertEqual(A.auroc([1, 2, 3, 4], [False, False, True, True]), 1.0)

    def test_spearman_and_partial(self):
        rng = np.random.default_rng(1)
        z = rng.standard_normal(500)
        x = z + 0.5 * rng.standard_normal(500)
        y = z + 0.5 * rng.standard_normal(500)
        self.assertAlmostEqual(A.spearman(x, y), stats.spearmanr(x, y)[0], places=12)
        self.assertIsNone(A.spearman(np.ones(10), np.arange(10)))
        rx, ry, rz = (stats.rankdata(v) for v in (x, y, z))
        res = lambda a: a - np.polyval(np.polyfit(rz, a, 1), rz)  # noqa: E731
        self.assertAlmostEqual(A.partial_spearman(x, y, z), np.corrcoef(res(rx), res(ry))[0, 1], places=10)
        self.assertLess(abs(A.partial_spearman(x, y, z)), 0.2)  # x and y share only z
        x[3] = np.nan
        self.assertIsNotNone(A.spearman(x, y))

    def test_paired_t_mde_and_seed_rule(self):
        d = [0.03, 0.07, 0.05, 0.06, 0.02]
        t = A.paired_t(d)
        ref = stats.ttest_1samp(d, 0.0, alternative="greater")
        self.assertAlmostEqual(t["p"], ref.pvalue, places=12)
        self.assertAlmostEqual(t["t"], ref.statistic, places=12)
        self.assertEqual((t["n"], t["positive"]), (5, 5))
        self.assertEqual(A.paired_t([0.1, None, 0.2])["n"], 2)
        sd, n = 0.02, 10
        effect = A.mde(sd, n)
        crit = stats.t.ppf(0.95, n - 1)
        power = stats.nct.sf(crit, n - 1, effect / (sd / math.sqrt(n)))
        self.assertAlmostEqual(power, 0.8, delta=0.02)  # the textbook approximation
        need = A.seeds_for_mde(sd, 0.02, 20)
        self.assertLessEqual(A.mde(sd, need), 0.02)
        self.assertGreater(A.mde(sd, need - 1), 0.02)
        self.assertIsNone(A.seeds_for_mde(1.0, 0.02, 20))
        self.assertEqual([A.seed_need(f, n) for f, n in ((0.8, 10), (0.7, 10), (0.8, 5), (0.7, 5), (0.7, 2))],
                         [8, 7, 4, 4, 2])

    def test_kish_and_mixture_density(self):
        self.assertAlmostEqual(A.kish_fraction(np.zeros(50)), 1.0, places=12)
        lw = np.full(50, -1e3)
        lw[0] = 0.0
        self.assertAlmostEqual(A.kish_fraction(lw), 1 / 50, places=12)
        rng = np.random.default_rng(2)
        covs = np.stack([np.eye(3) * (1 + t) + 0.1 * np.ones((3, 3)) for t in range(4)])
        s = rng.standard_normal((7, 3))
        ref = np.log(np.mean([stats.multivariate_normal(np.zeros(3), C).pdf(s) for C in covs], 0))
        np.testing.assert_allclose(A.mixture_logpdf(s, covs, chunk=3), ref, rtol=1e-12)

    def test_unittest_parser(self):
        ok = A.parse_unittest("....\n------\nRan 4 tests in 1.234s\n\nOK\n", 0)
        self.assertEqual((ok["ran"], ok["failures"], ok["errors"], ok["pass_"]), (4, 0, 0, True))
        bad = A.parse_unittest("F.E\n------\nRan 3 tests in 0.5s\n\nFAILED (failures=1, errors=1)\n", 1)
        self.assertEqual((bad["ran"], bad["failures"], bad["errors"], bad["pass_"]), (3, 1, 1, False))
        skip = A.parse_unittest("Ran 2 tests in 0.1s\n\nOK (skipped=1)\n", 0)
        self.assertEqual((skip["skipped"], skip["pass_"]), (1, True))
        self.assertFalse(A.parse_unittest("Traceback ...", 1)["pass_"])


def _synthetic_score(d_wrong, d_thin, ns_even=0.5, extra=0.0):
    """A tierA_score.json-shaped record at level m=2.0 (A-review regression tests of the pooled rules)."""
    rec = lambda fam, ns: dict(family=fam, NS_mix=ns, NS_minus_EVEN=ns - ns_even, reachable=True, converged=True,
                               relative_error=0.0, target_tag="m=2.0")
    matched = {"OR-wrong@m=2.0": rec("OR-wrong", ns_even + d_wrong), "OR-thin@m=2.0": rec("OR-thin", ns_even + d_thin),
               "ANTI@m=2.0": rec("ANTI", ns_even - 0.2), "SHUF0@m=2.0": rec("SHUF0", ns_even - 0.08 + extra),
               "SHUF1@m=2.0": rec("SHUF1", ns_even - 0.08 - extra), "CAP-wrong@m=2.0": rec("CAP-wrong", ns_even)}
    return dict(matched=matched, best_OR_thin_minus_best_EVEN=d_thin, best_OR_wrong_minus_NONE=d_wrong + 0.1,
                best_capped_minus_best_EVEN={"CAP-wrong": 0.0},
                null_shuf_pairs={"m=2.0": [dict(pair=["SHUF0", "SHUF1"], NS_difference=2 * extra)]})


class GateRuleTests(unittest.TestCase):
    """A-review regressions: R2's accuracy mark reads the TD3 fixed-point reference; pooled marks count every seed."""

    def test_r2_accuracy_reads_the_td3_reference(self):
        per = {c: dict(mean_E_pi=1.0, rows=10) for c in G.CONTEXTS}
        pooled = dict(ratio_E_Twrong_rich_main=5.0, thin_share_sum_E2=0.8, rich_main_median_rel_e1_pi=0.17,
                      rich_main_median_rel_e1_logged=0.15, rich_main_median_rel_e1_td3_pi=0.01,
                      rich_main_median_rel_e1_td3_logged=0.02, ratio_E_Twrong_Tright=2.0, mean_E_rich_main=1.0,
                      unstable_contexts=[])
        rec = dict(snapshots=[dict(fraction=1.0, update=100)], updates=100, final_update=100,
                   diagnostics={"100": dict(per_context=per, pooled=pooled)})
        v = A.r2_values(rec, 1.0)
        self.assertTrue(v["marks"]["R2_rel"])  # a perfect TD3 critic reads about 0.17 against Q^pi alone
        self.assertEqual((v["rel"], v["rel_deterministic"]), (0.01, 0.17))
        pooled["rich_main_median_rel_e1_td3_pi"] = 0.06
        self.assertFalse(A.r2_values(rec, 1.0)["marks"]["R2_rel"])
        self.assertEqual(A.PASS_MARKS["R2_rel"]["reference"], "td3_fixed_point")

    def _prereg(self, seeds):
        return A.Prereg(Path("unused"), "0" * 64, {}, replace(TINY, lean_seeds=seeds), Path("unused"))

    def test_r4_counts_every_seed(self):
        """Seed 2's run failed: R4 (1)-(4) all fail. Before the fix R4 (2)-(4) passed on the 2 surviving seeds."""
        P = self._prereg((0, 1, 2))
        good = {0: _synthetic_score(0.10, 0.08), 1: _synthetic_score(0.12, 0.09), 2: _synthetic_score(0.11, 0.085)}
        with mock.patch.object(A, "tier_a_scores", lambda *a: ({k: dict(score=v) for k, v in good.items()}, [])):
            ok = A.r4_evaluate(P, "base", "lean", 2.5)
        self.assertTrue(all(ok["marks"].values()), ok["marks"])
        failed = {0: dict(score=good[0]), 1: dict(score=good[1]), 2: dict(score=None, note="stopped")}
        with mock.patch.object(A, "tier_a_scores", lambda *a: (failed, [])):
            res = A.r4_evaluate(P, "base", "lean", 2.5)
        self.assertTrue(res["complete"])
        self.assertEqual(res["R4_2"]["n"], 2)
        self.assertLess(res["R4_2"]["p"], 0.05)  # the surviving seeds alone would pass
        self.assertEqual(res["marks"], dict(R4_1=False, R4_2=False, R4_3a=False, R4_3b=False, R4_4=False))

    def test_r7_null_and_global_linear_count_every_seed(self):
        P = self._prereg((0, 1, 2))
        main = {k: dict(score=_synthetic_score(0.10 + 0.01 * k, 0.08, extra=0.001 * k)) for k in range(3)}
        null = {k: dict(score=_synthetic_score(0.001 * k, 0.0)) for k in range(3)}

        def scores(prereg, retune, variant, alpha, seeds):
            return (main if variant == "main" else null), []

        with mock.patch.object(A, "tier_a_scores", scores):
            ok = A.r7_evaluate(P, "base", "lean", 2.5)
        self.assertTrue(ok["marks"]["R7c"] and ok["marks"]["R7_gl"], ok["marks"])
        w_main, w_null = (G.World(A.world_spec(TINY, "base", v)) for v in ("main", "null"))
        ratio = float(w_null.mix @ (w_null.j_opt - w_null.j_bc)) / float(w_main.mix @ (w_main.j_opt - w_main.j_bc))
        self.assertAlmostEqual(ratio, 1 / 5.63, places=2)
        self.assertAlmostEqual(ok["R7c"]["mean_in_main_world_units"], ok["R7c"]["diff"]["mean"] * ratio, places=12)
        null[2] = dict(score=None, note="stopped")
        with mock.patch.object(A, "tier_a_scores", scores):
            res = A.r7_evaluate(P, "base", "lean", 2.5)
        self.assertEqual(res["R7c"]["diff"]["n"], 2)
        self.assertFalse(res["marks"]["R7c"])
        self.assertFalse(res["marks"]["R7_gl"])


class WorldAndPlanTests(unittest.TestCase):
    def test_variants_and_retunes(self):
        cfg = A.PRESETS["real"]
        self.assertEqual(A.world_spec(cfg), G.STAGE0_SPEC)
        self.assertEqual(A.world_spec(cfg, variant="null").behaviour_fraction, 1.0)
        self.assertEqual(A.world_spec(cfg, variant="rung0").thin_rung, 0.0)
        self.assertEqual(A.world_spec(cfg, variant="rung0.3").thin_rung, 0.3)
        self.assertEqual(A.rung_variant(cfg, 0.01), "main")
        self.assertEqual([A.rung_variant(cfg, r) for r in (0, 0.03, 0.1)], ["rung0", "rung0.03", "rung0.1"])
        with self.assertRaises(ValueError):
            A.world_spec(cfg, variant="rung0.2")
        thin = A.world_spec(cfg, "thin_share")
        self.assertEqual(thin.mix, (0.20, 0.20, 0.25, 0.25, 0.10))
        self.assertAlmostEqual(sum(thin.mix), 1.0, places=12)
        w = G.World(A.world_spec(cfg, "wrong_gain"))
        np.testing.assert_allclose(w.B[G.T_WRONG][:, 2:], -1.5 * w.B[G.R0][:, 2:], rtol=0, atol=1e-15)
        self.assertEqual(len(A.RETUNE_ORDER), 2)

    def test_plan(self):
        cfg = A.PRESETS["real"]
        lean, full = A.plan(cfg, "lean"), A.plan(cfg, "full")
        stages = {line.split()[2] for line in lean if not line.startswith("#")}
        self.assertEqual(stages, {"prereg", "build_checks", "stage0", "r0r1", "critics", "r3", "r2", "r6", "r5world",
                                  "level1", "tierA", "r4", "r7", "verdict"})
        self.assertFalse(any(" tierB " in line for line in lean))
        self.assertEqual(sum(" tierB " in line for line in full), 10 * len(cfg.tier_b_arms))
        cmds = [line for line in lean if not line.startswith("#")]
        self.assertEqual(sum(" critics " in line and "--variant" not in line for line in cmds), 5 * 3)
        self.assertEqual(sum("--variant rung" in line for line in lean), 4 * 3 * 2)
        first = lambda stage: lean.index(next(x for x in lean if f" {stage} " in x))  # noqa: E731
        self.assertLess(first("r3"), first("r2"))
        buf = io.StringIO()
        with redirect_stdout(buf):
            A.main(["plan", "--path", "full", "--prereg", "/nonexistent.json"])
        self.assertIn("tierB", buf.getvalue())


class BandTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.world = G.World()
        cls.data = G.generate(cls.world, 0, sizes=(10, 10, 10))
        cls.settings = T.TrainSettings(updates=4, chunk=2)
        kind = T.actor_kind("context_linear")
        K = T.bc_gains(cls.data["train"])[0]
        _, cls.state, _ = T.initial_state(T.make_args(cls.settings, 0), kind, kind.params(K), cls.settings.lr)
        cls.kind = kind

    def test_population_band_matches_a_direct_computation(self):
        rows = {k: v[:5] for k, v in self.data["eval"].items()}
        draws, gamma = 6, self.settings.gamma
        band, mean_res, sd_t, q1 = A.population_band(self.world, self.state, self.kind, rows, self.settings, draws,
                                                     np.random.default_rng(9), chunk_rows=5)
        rng = np.random.default_rng(9)
        shocks = rng.standard_normal((5, draws, G.DS))
        noise = np.clip(rng.standard_normal((5, draws, G.DA)) * 0.2, -0.5, 0.5)
        for i in range(5):
            c = int(rows["context"][i])
            s, a = rows["state"][i], rows["action"][i]
            nxt = self.world.A @ s + self.world.B[c] @ a + self.world.noise_sd[c] * shocks[i]
            nobs = np.concatenate([nxt, np.repeat(np.eye(G.NC)[c][None], draws, 0)], 1).astype(np.float32)
            act = np.asarray(self.kind.apply(self.state.actor_target.params, jnp.asarray(nobs))) + noise[i]
            tq = np.asarray(T._CRITIC.apply(self.state.critic_target.params, jnp.asarray(nobs),
                                            jnp.asarray(act, jnp.float32))).min(-1)
            t = rows["reward"][i] + gamma * tq.astype(np.float64)
            q = float(np.asarray(T._CRITIC.apply(self.state.critic.params, jnp.asarray(rows["obs"][i:i + 1],
                                                                                        jnp.float32),
                                                 jnp.asarray(a[None], jnp.float32)))[0, 0])
            self.assertAlmostEqual(q1[i], q, places=5)
            np.testing.assert_allclose(band[i], np.sqrt(np.mean((t - q) ** 2)), rtol=1e-5)
            np.testing.assert_allclose(sd_t[i], t.std(), rtol=1e-4, atol=1e-6)
        # the band reads the logged reward, which is the deterministic r(s, a)
        np.testing.assert_allclose(rows["reward"], -(np.sum(rows["state"] ** 2, 1) + self.world.spec.action_cost *
                                                     np.sum(rows["action"] ** 2, 1)), rtol=1e-12)

    def test_bca_band_is_switched_off(self):
        with self.assertRaisesRegex(RuntimeError, "switched off"):
            A.bca_band(self.world, self.state, self.kind, self.data, self.settings, 0, TINY)
        lowered = A.bca_band(self.world, self.state, self.kind, self.data, self.settings, 0, TINY, allow=True,
                             execute=False)  # traced only: nothing is fitted
        self.assertTrue(hasattr(lowered, "compile"))

    def test_regressor_learns_a_fixed_target(self):
        tr, ev = self.data["train"], self.data["eval"]
        y = np.abs(tr["state"][:, 0]) + 0.5 * (tr["context"] == G.T_WRONG)
        y[:3] = np.nan  # ignored rows
        pred, loss = A.fit_regressor(tr, ev, y=y, settings=self.settings, steps=400, batch=128, lr=3e-3,
                                     rng_rows=np.random.default_rng(0), key=jax.random.PRNGKey(0))
        target = np.abs(ev["state"][:, 0]) + 0.5 * (ev["context"] == G.T_WRONG)
        self.assertGreater(A.spearman(pred, target), 0.8)
        self.assertTrue(np.isfinite(loss))


class EndToEndTests(unittest.TestCase):
    """Every stage of the gate in order on the tiny preset (about a minute on 2 CPUs)."""

    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.dir = Path(cls.tmp.name)
        A.write_prereg(cls.dir / "prereg.json", TINY)
        cls.P = A.load_prereg(cls.dir / "prereg.json", cls.dir / "runs")

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def check(self, path):
        self.assertTrue(strict(path), path)
        self.assertEqual(json.loads(Path(path).read_text())["prereg_sha256"], self.P.digest)

    def test_every_stage_in_order(self):
        P, cfg, root = self.P, TINY, self.P.root
        bc = A.stage_build_checks(P, unittests=False, log=QUIET)
        self.check(root / "build_checks" / "build_checks.json")
        for k in ("B1", "B2", "B3", "B4", "B5", "B6"):
            self.assertTrue(bc["checks"][k]["pass_"], (k, bc["checks"][k]))
        self.assertIsNone(bc["checks"]["B7"]["pass_"])
        s0 = A.stage_stage0(P, log=QUIET)
        self.assertTrue(s0["pass_"], s0["marks"])
        self.check(root / "stage0" / "base" / "stage0.json")
        # one stage through the command line, as its own process
        env = dict(os.environ, JAX_PLATFORMS="cpu", PYTHONDONTWRITEBYTECODE="1")
        proc = subprocess.run([sys.executable, "experiments/signal/glq_acceptance.py", "r0r1", "--seeds", "0",
                               "--prereg", str(P.path), "--runs", str(P.runs)], cwd=A.ROOT, env=env,
                              capture_output=True, text=True)
        self.assertEqual(proc.returncode, 0, proc.stderr[-2000:])
        first = root / "r0r1" / "base" / "seed0.json"
        self.check(first)
        stamp = first.stat().st_mtime_ns
        for k in cfg.lean_seeds:
            rec = A.stage_r0r1(P, k, log=QUIET)
            self.assertTrue(rec["marks"]["R1b"]["pass_"])
        self.assertEqual(first.stat().st_mtime_ns, stamp)  # resumed, not recomputed
        with self.assertRaisesRegex(A.PreregError, "R3 at alpha 2.5 .* incomplete"):
            A.stage_r6(P, 0, "lean", "auto", log=QUIET)
        with self.assertRaises(ValueError):
            A.stage_critics(P, 0, "BCA", log=QUIET)
        for k in cfg.lean_seeds:
            for mode in A.MODES:
                rec = A.stage_critics(P, k, mode, log=QUIET)
                self.assertEqual(rec["final_update"], cfg.updates)
        run = A.critic_dir(P, "base", "main", "HOST", 2.5, 0)
        self.check(run / "stage.json")
        self.assertTrue(strict(run / "run.json"))
        self.assertTrue((run / "snapshot_050.msgpack").exists())
        for rung in cfg.ladder_rungs:
            if A.rung_variant(cfg, rung) != "main":
                for mode in ("HOST", "FQE"):
                    A.stage_critics(P, cfg.ladder_seeds[0], mode, A.rung_variant(cfg, rung), log=QUIET)
        with self.assertRaises(ValueError):
            A.stage_critics(P, 1, "HOST", "rung0.3", log=QUIET)  # not a ladder seed
        r3 = A.stage_r3(P, "lean", 2.5, log=QUIET)
        self.check(A.pooled_path(P, "r3", "base", "lean", 2.5))
        self.assertTrue(r3["complete"])
        self.assertEqual(set(r3["per_seed"]["0"]["marks"]), {"R3_cost", "R3_ns_gap", "R3_thin_shortfall",
                                                             "R3_rich_ns"})
        if not r3["pass_"]:
            with self.assertRaisesRegex(A.PreregError, "alpha 5"):
                A.resolve_alpha(P, "base", "lean", "auto")
        r2 = A.stage_r2(P, "lean", log=QUIET)
        self.assertTrue(r2["complete"])
        self.assertEqual(len(r2["ladder"]["FQE"]["rungs"]), len(cfg.ladder_rungs))
        alpha = 2.5
        for k in cfg.lean_seeds:
            r6 = A.stage_r6(P, k, "lean", alpha, log=QUIET)
            self.assertEqual(set(r6["marks"]), {"R6a", "R6b", "R6c", "R6d", "R1c"})
            r5 = A.stage_r5world(P, k, "lean", alpha, log=QUIET)
            self.assertEqual(set(r5["snapshots"]), {"1", "0.5"})
            self.assertIn("switched off", r5["snapshots"]["1"]["bca_band"]["status"])
            self.assertIsNotNone(r5["snapshots"]["1"]["regressors"]["auroc_ideal"])
            self.assertEqual(set(r5["snapshots"]["1"]["confound_spearman"]["leverage"]),
                             {"pooled", *G.CONTEXTS})
            A.stage_level1(P, k, "lean", alpha, log=QUIET)
            A.stage_tier_a(P, k, "main", "lean", alpha, log=QUIET)
            A.stage_critics(P, k, "HOST", "null", alpha, log=QUIET)
            A.stage_tier_a(P, k, "null", "lean", alpha, log=QUIET)
            A.stage_tier_a(P, k, "global_linear", "lean", alpha, log=QUIET)
        for path in (A.seed_path(P, "r6", "base", alpha, 0), A.seed_path(P, "r5world", "base", alpha, 1),
                     A.seed_path(P, "level1", "base", alpha, 0)):
            self.check(path)
        ta = A.tier_a_dir(P, "base", "main", alpha, 0)
        self.check(ta / "stage.json")
        blind = json.loads((ta / "tierA_blind.json").read_text())
        self.assertFalse({"J", "NS", "NS_mix", "J_mix"} & {k for r in blind["ladder"] for k in r})
        self.assertEqual(blind["glq_match_sha256"], T.M.source_sha256())
        nul = json.loads((A.tier_a_dir(P, "base", "null", alpha, 0) / "tierA_blind.json").read_text())
        self.assertEqual(set(nul["families"]), set(cfg.control_families))
        gl = json.loads((A.tier_a_dir(P, "base", "global_linear", alpha, 0) / "tierA_blind.json").read_text())
        self.assertEqual(gl["actor_class"], "global_linear")
        r4 = A.stage_r4(P, "lean", alpha, log=QUIET)
        self.assertTrue(r4["complete"])
        self.assertIn(r4["level"], ("m=2.0", "m=1.3"))
        r7 = A.stage_r7(P, "lean", alpha, log=QUIET)
        self.assertTrue(r7["complete"])
        self.assertEqual(r7["R7_null_pairs"]["n_pairs"], math.comb(cfg.n_shuf, 2))
        for k in cfg.full_seeds:
            for arm in cfg.tier_b_arms:
                rec = A.stage_tier_b(P, k, arm, "full", alpha, log=QUIET)
                self.assertIn(rec["status"], ("trained", "unreachable in Tier A at this level: not trained"))
        v = A.stage_verdict(P, "full", log=QUIET)
        out = root / "verdict" / "base" / "full"
        self.check(out / "verdict.json")
        md = (out / "verdict.md").read_text()
        self.assertIn("| Step | Requirement | Pass mark | Measured | Result | Seeds |", md)
        ids = {r["id"] for r in v["rows"]}
        self.assertEqual(ids - {"alpha"}, set(A.PASS_MARKS))
        self.assertEqual({r["step"] for r in v["rows"]}, set(range(1, 12)))
        self.assertNotIn(None, [v["step_status"].get(str(k), "missing") for k in (2, 5, 9, 10, 11)])
        lean = A.stage_verdict(P, "lean", log=QUIET)
        self.assertIn(lean["outcome"].split(":")[0].split(" ")[0], ("build", "step(s)", "incomplete", "NOT",
                                                                     "LEAN", "ACCEPTED"))
        self.stopped_run_fails_its_seed(P, alpha)

    def stopped_run_fails_its_seed(self, P, alpha):
        """A HOST run that stopped early (nonfinite) at alpha 5 counts as a failed seed, never a missing one."""
        src = A.critic_dir(P, "base", "main", "HOST", alpha, 0)
        run = json.loads((src / "run.json").read_text())
        first = run["snapshots"][0]
        run.update(snapshots=[first], final_update=first["update"], nonfinite_at=first["update"],
                   diagnostics={str(first["update"]): run["diagnostics"][str(first["update"])]})
        dst = A.critic_dir(P, "base", "main", "HOST", 5.0, 0)
        A.write_json(dst / "run.json", run)
        A.write_json(dst / "stage.json", dict(json.loads((src / "stage.json").read_text()), alpha=5.0))
        self.assertFalse(A.run_complete(run))
        self.assertIsNone(A.final_gains(run))
        self.assertIsNone(A.snapshot_key(run, 1.0))
        self.assertFalse(A.r2_values(run, 1.0)["all"])
        r3 = A.r3_values(G.World(), run, run)
        self.assertFalse(r3["all"])
        self.assertIn("stopped at update", r3["note"])
        with self.assertRaises(A.IncompleteRun):
            A.load_critic(P, "base", "main", "HOST", 5.0, 0, 0.5)
        r6 = A.stage_r6(P, 0, "lean", 5.0, log=QUIET)
        self.assertFalse(any(r6["marks"].values()))
        l1 = A.stage_level1(P, 0, "lean", 5.0, log=QUIET)
        self.assertEqual(l1["marks"], {"L1": False})
        ta = A.stage_tier_a(P, 0, "main", "lean", 5.0, log=QUIET)
        self.assertEqual(ta["status"], "failed")
        scores, missing = A.tier_a_scores(P, "base", "main", 5.0, TINY.lean_seeds)
        self.assertIsNone(scores[0]["score"])  # counted, not missing
        self.assertIn("tierA main seed 1", missing)
        r5 = A.stage_r5world(P, 0, "lean", 5.0, log=QUIET)
        self.assertFalse(any(r5["marks"].values()))

    def test_outcomes(self):
        P = self.P
        ok = {k: True for k in range(1, 12)}
        self.assertTrue(A.outcome(P, "base", ok).startswith("ACCEPTED for error-tracking"))
        self.assertIn("thin-data placement only", A.outcome(P, "base", {**ok, 8: False}))
        self.assertTrue(A.outcome(P, "base", {**ok, 11: None}, "lean").startswith("LEAN PASS"))
        self.assertIn("incomplete", A.outcome(P, "base", {**ok, 11: None}, "full"))
        self.assertIn("next pre-registered retune: thin_share", A.outcome(P, "base", {**ok, 6: False}))
        self.assertIn("next pre-registered retune: wrong_gain", A.outcome(P, "thin_share", {**ok, 9: False}))
        self.assertIn("REJECTED", A.outcome(P, "wrong_gain", {**ok, 5: False}))
        self.assertIn("build failure", A.outcome(P, "base", {**ok, 3: False, 6: False}))
        self.assertIn("NOT ACCEPTED", A.outcome(P, "base", {**ok, 10: False}))


if __name__ == "__main__":
    unittest.main()

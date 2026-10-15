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
- Amendment A1: its record in the pre-registration (refused when edited), the fresh seeds and the 10-round cap; R2 gated
  on the FQE critic's concentration only (1.5 x the thin row share: 0.45 base, 0.75 thin_share; accuracy and HOST
  reported, not gated; a missing HOST run does not hold the gate back); the slope measure against chosen action shifts
  and the argmax of both quadratics; R3's thin-shortfall mark at 1.5 x the mix share in both worlds; R4's level (S2 at
  >= 80% of seeds matched, else S1, else R4 fails), an unmatched seed excluded from the means and counted as not
  positive (positive and negative cases), a failed Tier A seed still failing the pooled marks and R7(b); the record's
  disclosure of the earlier data-only outputs on fresh seeds, and an audit of this machine's runs/glq/ that every
  output on seeds 5-14 outside the current pre-registration's directory is disclosed there;
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
import re
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
from experiments.signal import lq_harness as LQ

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

    def test_amendment_a1_is_recorded(self):
        """Amendment A1: fresh seeds, the 10-round cap, and the amendment block with every old and new rule."""
        real = A.PRESETS["real"]
        self.assertEqual(real.lean_seeds, (5, 6, 7, 8, 9))
        self.assertEqual(real.full_seeds, tuple(range(5, 15)))
        self.assertEqual(real.ladder_seeds, (5, 6, 7))
        self.assertFalse(set(real.full_seeds) & {0, 1, 2, 3, 4})  # seeds 0-4 informed the amendment
        self.assertEqual(real.match_rounds, 10)
        am = A.load_prereg(self.path, self.dir / "runs").payload["amendment"]
        self.assertEqual((am["id"], am["date"], am["post_hoc"]), ("A1", "2026-10-12", True))
        self.assertEqual(am["original_prereg_sha256"],
                         "255369889b400c2b93d5c92ca2e861c71d39554bf490b3c9489a5661be177baf")
        self.assertEqual((am["informed_by_seeds"], am["fresh_seeds"]), ([0, 1, 2, 3, 4], list(range(5, 15))))
        self.assertIn("POST HOC", am["statement"])
        self.assertEqual([c["id"] for c in am["changes"]], ["R2", "R3_thin_shortfall", "matching", "seeds"])
        self.assertTrue(all(c["old"] and c["new"] and c["old"] != c["new"] for c in am["changes"]))
        marks = A.PASS_MARKS
        self.assertEqual((marks["R2_ratio"]["gated_critic"], marks["R2_thin_E2"]["gated_critic"]), ("FQE", "FQE"))
        self.assertEqual((marks["R2_thin_E2"]["share_factor"], marks["R3_thin_shortfall"]["share_factor"]),
                         (1.5, 1.5))
        self.assertNotIn("min", marks["R2_thin_E2"])
        self.assertNotIn("min", marks["R3_thin_shortfall"])
        self.assertFalse(marks["R2_rel"]["gated"] or marks["R2_slope"]["gated"])
        self.assertEqual(marks["R4_level"]["seed_frac"], 0.8)

    def test_amendment_discloses_prior_outputs_on_fresh_seeds(self):
        """Review regression: the record said 'Seeds 5-14 have never been run', but the earlier pre-registrations had
        run the data-only r0r1 stage on seeds 5-9. The record now lists those outputs, each data-only, and states
        what 'fresh' means."""
        am = A.load_prereg(self.path, self.dir / "runs").payload["amendment"]
        self.assertNotIn("have never been run", am["statement"])
        prior = am["prior_outputs_on_fresh_seeds"]
        self.assertEqual({p["preregistration_sha256"] for p in prior},
                         {"218e1af17fdf9c177b1a652a644d8916e4b036bb0d22ee32a6e0956a61c65860",
                          am["original_prereg_sha256"],
                          "12db2dd5f33bbad473944513d909683f4ba641fd42cc0d077ba75f02e58d4471"})
        for p in prior:
            self.assertFalse(p["trains"])
            self.assertNotIn(p["stage"], A.TRAINING_STAGES)
            self.assertLessEqual(set(p["seeds"]), set(am["fresh_seeds"]))
        self.assertIn("prior_outputs_on_fresh_seeds", next(c for c in am["changes"] if c["id"] == "seeds")["new"])

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
            "amendment": lambda p: p["amendment"].update(post_hoc=False),
            "match_rounds": lambda p: p["config"].update(match_rounds=5),
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


class FreshSeedAuditTests(unittest.TestCase):
    """Amendment A1's fresh seeds (review regression): every output under runs/glq/ that names a seed in 5-14 and lies
    outside the current pre-registration's directory is one the amendment discloses (prior_outputs_on_fresh_seeds,
    data-only stages). Reads this machine's runs/glq/ (gitignored); skipped where it does not exist."""

    SEED = re.compile(r"(?:^|[_/.])(?:rep|seed|seeds_)(\d+)(?=$|[_/.])")

    def test_pattern(self):
        names = {"critics/base/main/FQE/rep5/run.json": {5}, "r0r1/base/seed12.json.sha256": {12},
                 "runner/critics_seeds_7_mode_HOST.log": {7}, "critics/base/main/FQE/rep0/errors_100.npz": {0},
                 "critics/base/rung0.03/HOST_alpha10/rep1/snapshot_050.msgpack": {1}, "r0r1/base": set()}
        for rel, want in names.items():
            self.assertEqual({int(k) for k in self.SEED.findall(rel)}, want, rel)

    def test_no_undisclosed_output_on_fresh_seeds(self):
        runs = A.RUNS_DEFAULT
        side = A.PREREG_DEFAULT.with_name(A.PREREG_DEFAULT.name + ".sha256")
        if not runs.is_dir() or not side.exists():
            self.skipTest("no runs/glq/ (or no pre-registration) on this machine")
        current = side.read_text(encoding="utf8").split()[0][:16]
        fresh = set(A.AMENDMENT["fresh_seeds"])
        disclosed = {}
        for p in A.AMENDMENT["prior_outputs_on_fresh_seeds"]:
            disclosed.setdefault((p["preregistration_sha256"][:16], p["stage"]), set()).update(p["seeds"])
        bad = []
        for f in runs.rglob("*"):
            rel = f.relative_to(runs).as_posix()
            parts = rel.split("/")
            if not f.is_file() or parts[0] == current:
                continue
            seeds = {int(k) for k in self.SEED.findall(rel)} & fresh
            if seeds and not seeds <= disclosed.get((parts[0], parts[1] if len(parts) > 1 else ""), set()):
                bad.append(rel)
        self.assertEqual(bad, [], "outputs on fresh seeds that Amendment A1 does not disclose")


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


LEVELS = ("m=2.0", "m=1.3")


def _synthetic_score(d_wrong, d_thin, ns_even=0.5, extra=0.0, unmatched=(), s1_shift=0.0):
    """A tierA_score.json-shaped record at both levels (A-review regression tests of the pooled rules). unmatched holds
    (family, level) pairs that are unreachable (not matched); at S1 OR-wrong's and OR-thin's NS are shifted by s1_shift,
    so a test can tell which level a mean was read at."""
    matched = {}
    for tag in LEVELS:
        for fam, ns in (("OR-wrong", ns_even + d_wrong), ("OR-thin", ns_even + d_thin), ("ANTI", ns_even - 0.2),
                        ("SHUF0", ns_even - 0.08 + extra), ("SHUF1", ns_even - 0.08 - extra), ("CAP-wrong", ns_even)):
            ok = (fam, tag) not in unmatched
            ns += s1_shift if tag == "m=1.3" and fam in ("OR-wrong", "OR-thin") else 0.0
            matched[f"{fam}@{tag}"] = dict(family=fam, NS_mix=ns if ok else None,
                                           NS_minus_EVEN=ns - ns_even if ok else None, reachable=ok, converged=ok,
                                           relative_error=0.0 if ok else None, target_tag=tag)
    return dict(matched=matched, best_OR_thin_minus_best_EVEN=d_thin, best_OR_wrong_minus_NONE=d_wrong + 0.1,
                best_capped_minus_best_EVEN={"CAP-wrong": 0.0},
                null_shuf_pairs={tag: [dict(pair=["SHUF0", "SHUF1"], NS_difference=2 * extra)] for tag in LEVELS})


def _diag(ratio=5.0, thin_E2=0.8, thin_rows=0.30, rel=0.01, thin_E=(3.0, 1.0), rich_E=1.0):
    """A run.json diagnostics entry (summarize_errors' keys that R2 reads)."""
    per = {c: dict(mean_E_pi=rich_E, rows=10) for c in G.CONTEXTS}
    per["T_wrong"]["mean_E_pi"], per["T_right"]["mean_E_pi"] = thin_E
    pooled = dict(ratio_E_Twrong_rich_main=ratio, thin_share_sum_E2=thin_E2, thin_share_rows=thin_rows,
                  rich_main_median_rel_e1_pi=0.17, rich_main_median_rel_e1_logged=0.15,
                  rich_main_median_rel_e1_td3_pi=rel, rich_main_median_rel_e1_td3_logged=0.02,
                  ratio_E_Twrong_Tright=2.0, mean_E_rich_main=rich_E, unstable_contexts=[])
    return dict(per_context=per, pooled=pooled, K=np.zeros((G.NC, G.DA, G.DS)).tolist())


def _run(diag):
    return dict(snapshots=[dict(fraction=1.0, update=100)], updates=100, final_update=100, nonfinite_at=None,
                diagnostics={"100": diag})


def _gains_run(K):
    """A complete run record whose final gains are K (what r3_values reads)."""
    return dict(snapshots=[], updates=1, final_update=1, nonfinite_at=None, diagnostics={"1": dict(K=K.tolist())})


class GateRuleTests(unittest.TestCase):
    """A-review regressions and Amendment A1's rules on synthetic records: R2 gates on FQE's concentration only
    (1.5 x the thin row share; accuracy and HOST reported), the slope measure, R3's 1.5 x mix-share mark, R4's level
    rule and its unmatched seeds."""

    def test_r2_accuracy_is_reported_not_gated(self):
        rec = _run(_diag(rel=0.01))
        v = A.r2_values(rec, 1.0)
        self.assertEqual(set(v["marks"]), {"R2_ratio", "R2_thin_E2"})
        self.assertTrue(v["all"])
        self.assertEqual((v["rel"], v["rel_deterministic"]), (0.01, 0.17))  # the TD3 fixed-point reference
        self.assertEqual(v["reported"]["R2_rel"], 0.01)
        bad = A.r2_values(_run(_diag(rel=1.15)), 1.0)  # 23x the former 5% mark: reported, the seed still passes
        self.assertTrue(bad["all"])
        self.assertEqual(bad["reported"]["R2_rel"], 1.15)
        self.assertIsNone(bad["slope"])  # no rows given: the slope is not measured
        self.assertEqual(A.PASS_MARKS["R2_rel"]["reference"], "td3_fixed_point")

    def test_r2_concentration_marks_and_thresholds_per_world(self):
        """(a) ratio >= 3; (b) thin share of sum E^2 >= 1.5 x the thin row share: 0.45 base, 0.75 thin_share."""
        cfg = A.PRESETS["real"]
        for retune, share in (("base", 0.30), ("thin_share", 0.50)):
            world = G.World(A.world_spec(cfg, retune))
            data = G.generate(world, cfg.lean_seeds[0], cfg.data_seed, sizes=tuple(cfg.sizes))
            rows = float(np.isin(data["eval"]["context"], G.THIN).mean())  # what summarize_errors' thin_share_rows is
            self.assertEqual(rows, share)
            self.assertAlmostEqual(A.thin_mix_share(world), share, places=12)
            thr = 1.5 * share
            self.assertAlmostEqual(A.r2_values(_run(_diag(thin_rows=rows)), 1.0)["thin_E2_min"], thr, places=12)
            self.assertAlmostEqual(thr, {"base": 0.45, "thin_share": 0.75}[retune], places=12)
            self.assertTrue(A.r2_values(_run(_diag(thin_E2=thr + 0.01, thin_rows=rows)), 1.0)["marks"]["R2_thin_E2"])
            self.assertFalse(A.r2_values(_run(_diag(thin_E2=thr - 0.01, thin_rows=rows)), 1.0)["marks"]["R2_thin_E2"])
        # the former 0.60 mark would have passed 0.70 in thin_share and failed 0.50 in the base world
        self.assertFalse(A.r2_values(_run(_diag(thin_E2=0.70, thin_rows=0.5)), 1.0)["all"])
        self.assertTrue(A.r2_values(_run(_diag(thin_E2=0.50, thin_rows=0.3)), 1.0)["all"])
        self.assertTrue(A.r2_values(_run(_diag(ratio=3.0)), 1.0)["marks"]["R2_ratio"])
        self.assertFalse(A.r2_values(_run(_diag(ratio=2.99)), 1.0)["all"])

    def _prereg(self, seeds, **kw):
        return A.Prereg(Path("unused"), "0" * 64, {}, replace(TINY, lean_seeds=seeds, **kw), Path("unused"))

    def _r2(self, fqe_pass, host_pass, ladder_fqe=True, ladder_host=True, host_missing=False):
        """r2_evaluate on synthetic runs: FQE passes R2 in the seeds of fqe_pass, HOST in those of host_pass."""
        seeds, ladder_seeds = (0, 1, 2, 3, 4), (0, 1)
        P = self._prereg(seeds, ladder_seeds=ladder_seeds)
        rungs = [float(r) for r in TINY.ladder_rungs]

        def record(prereg, retune, variant, mode, alpha, rep):
            if mode == "HOST" and host_missing:
                return None
            if variant == "main":
                ok = rep in (fqe_pass if mode == "FQE" else host_pass)
                return _run(_diag(ratio=5.0 if ok else 1.2, thin_E2=0.8 if ok else 0.2))
            rung = 0.01 if variant == "main" else float(variant[4:])
            falls = ladder_fqe if mode == "FQE" else ladder_host
            k = rungs.index(rung)
            thin = (3.0 - 0.5 * k) if falls else (1.0 + 0.5 * k)  # falls with density, or rises
            return _run(_diag(thin_E=(thin, thin), rich_E=1.0 if falls else 0.2))

        with mock.patch.object(A, "_run_record", record):
            return A.r2_evaluate(P, "base", "lean", 2.5)

    def test_r2_gates_on_fqe_only(self):
        res = self._r2(fqe_pass=(0, 1, 2, 3), host_pass=())  # FQE 4 of 5; HOST fails everything
        self.assertTrue(res["complete"])
        self.assertTrue(res["pass_"], res["marks"])
        self.assertEqual((res["passing_seeds"]["FQE"], res["passing_seeds"]["HOST"]), (4, 0))
        self.assertFalse(res["reported"]["HOST"]["main"])
        self.assertFalse(res["reported"]["HOST"]["gated"])
        res = self._r2(fqe_pass=(0, 1, 2), host_pass=(0, 1, 2, 3, 4))  # FQE 3 of 5 (need 4); HOST passes everything
        self.assertFalse(res["pass_"])
        self.assertTrue(res["reported"]["HOST"]["main"])
        # the ladder is judged on FQE: a failing HOST ladder does not fail R2, a failing FQE ladder does
        res = self._r2(fqe_pass=(0, 1, 2, 3, 4), host_pass=(), ladder_host=False)
        self.assertTrue(res["pass_"])
        self.assertFalse(res["reported"]["HOST"]["ladder_falls"] or res["reported"]["HOST"]["ladder_top"])
        res = self._r2(fqe_pass=(0, 1, 2, 3, 4), host_pass=(0, 1, 2, 3, 4), ladder_fqe=False)
        self.assertFalse(res["pass_"])
        self.assertFalse(res["marks"]["ladder_falls"])
        # a missing HOST run is reported, not a gap in the gate
        res = self._r2(fqe_pass=(0, 1, 2, 3, 4), host_pass=(), host_missing=True)
        self.assertTrue(res["complete"] and res["pass_"])
        self.assertTrue(res["missing_reported"])
        self.assertEqual(res["missing"], [])

    def test_slope_measure_known_answer(self):
        """Q1 = Q^pi + delta . a moves the greedy action by -H_aa^-1 delta / 2: built from chosen shifts and checked
        against the argmax of both quadratics; an unstable context gives NaN; r2_slope reads the saved rows."""
        world = G.World()
        rng = np.random.default_rng(3)
        K = world.k_behaviour.copy()
        n = 400
        ctx = rng.integers(G.NC, size=n)
        s = rng.standard_normal((n, G.DS))
        pi = -np.einsum("nij,nj->ni", K[ctx], s)
        want = rng.standard_normal((n, G.DA)) * rng.uniform(0.1, 3.0, (n, 1))  # the shift of each row's action
        g_pi, g_q1, delta = (np.zeros((n, G.DA)) for _ in range(3))
        for c in range(G.NC):
            H, _, _ = LQ.exact_q(world.systems[c], K[c])
            Haa, Has = H[G.DS:, G.DS:], H[G.DS:, :G.DS]
            self.assertTrue(np.all(np.linalg.eigvalsh(Haa) < 0))  # concave in the action
            m = ctx == c
            g_pi[m] = 2 * (s[m] @ Has.T + pi[m] @ Haa)
            delta[m] = -2 * want[m] @ Haa  # H_aa symmetric
            g_q1[m] = g_pi[m] + delta[m]
            # argmax_a of Q^pi and of Q^pi + delta . a, solved directly
            a0 = np.linalg.solve(2 * Haa, -2 * (s[m] @ Has.T).T).T
            a1 = np.linalg.solve(2 * Haa, -(2 * (s[m] @ Has.T) + delta[m]).T).T
            np.testing.assert_allclose(a1 - a0, want[m], rtol=1e-9, atol=1e-12)
        shift = A.slope_shift(world, K, ctx, g_q1, g_pi)
        np.testing.assert_allclose(shift, np.linalg.norm(want, axis=1), rtol=1e-9)
        self.assertTrue(np.all(A.slope_shift(world, K, ctx, g_pi, g_pi) == 0))  # an exact critic does not move it
        bad = K.copy()
        bad[G.R1] = -3 * world.k_star[G.R1]
        self.assertGreaterEqual(world.radius(bad)[G.R1], 1.0)
        out = A.slope_shift(world, bad, ctx, g_q1, g_pi)
        self.assertTrue(np.all(np.isnan(out[ctx == G.R1])))
        self.assertTrue(np.all(np.isfinite(out[ctx != G.R1])))
        with tempfile.TemporaryDirectory() as tmp:
            T.save_error_rows(Path(tmp) / "errors_100.npz", dict(context=ctx, grad_q1_pi=g_q1, grad_qpi_pi=g_pi))
            rec = _run(dict(_diag(), K=K.tolist()))
            got = A.r2_slope(world, rec, "100", tmp)
            rich = np.isin(ctx, G.RICH_MAIN)
            self.assertAlmostEqual(got["median_rich"], float(np.median(np.linalg.norm(want[rich], axis=1))), places=4)
            self.assertAlmostEqual(A.r2_values(rec, 1.0, world, tmp)["slope"], got["median_rich"], places=12)
            self.assertIn("missing", A.r2_slope(world, dict(rec, updates=200), "100", tmp)["note"])  # errors_050

    def test_r3_thin_shortfall_is_1_5_x_the_mix_share(self):
        """Host = K* except the thin contexts (behaviour clone) and R0 (between K* and the clone, t chosen by bisection
        so the thin share of the shortfall hits a target); EXACT = K*."""
        cfg = A.PRESETS["real"]

        def share_at(world, target):
            exact = _gains_run(world.k_star)

            def host(t):
                K = world.k_star.copy()
                for c in G.THIN:
                    K[c] = world.k_behaviour[c]
                K[G.R0] = world.k_star[G.R0] + t * (world.k_behaviour[G.R0] - world.k_star[G.R0])
                return A.r3_values(world, _gains_run(K), exact)
            lo, hi = 0.0, 1.0  # share 1 at t = 0 (thin only), falling as R0's shortfall grows
            for _ in range(40):
                mid = (lo + hi) / 2
                lo, hi = (mid, hi) if host(mid)["thin_shortfall_share"] > target else (lo, mid)
            return host((lo + hi) / 2)

        for retune, thr, below, above in (("base", 0.45, 0.42, 0.50), ("thin_share", 0.75, 0.70, 0.80)):
            world = G.World(A.world_spec(cfg, retune))
            for target, passes in ((below, False), (above, True)):
                v = share_at(world, target)
                self.assertAlmostEqual(v["thin_shortfall_share"], target, places=6)
                self.assertAlmostEqual(v["thin_shortfall_min"], thr, places=12)
                self.assertEqual(v["marks"]["R3_thin_shortfall"], passes, (retune, target))
        # base world: 0.50 passes the amended mark although the former >= 0.60 would have failed it
        self.assertTrue(share_at(G.World(A.world_spec(cfg, "base")), 0.50)["marks"]["R3_thin_shortfall"])

    def _r4(self, scores, seeds):
        P = self._prereg(seeds)
        wrapped = {k: dict(score=v) if "matched" in v else v for k, v in scores.items()}
        with mock.patch.object(A, "tier_a_scores", lambda *a: (wrapped, [])):
            return A.r4_evaluate(P, "base", "lean", 2.5), P

    def test_r4_counts_every_seed(self):
        """Seed 4's Tier A run failed: R4 (1)-(4) fail (rule unchanged by A1), though S2 qualifies (4 of 5)."""
        seeds = (0, 1, 2, 3, 4)
        good = {k: _synthetic_score(0.10 + 0.01 * k, 0.08 + 0.005 * k) for k in seeds}
        ok, _ = self._r4(good, seeds)
        self.assertTrue(all(ok["marks"].values()), ok["marks"])
        self.assertEqual(ok["level"], "m=2.0")
        failed = dict(good)
        failed[4] = dict(score=None, note="stopped")
        res, _ = self._r4(failed, seeds)
        self.assertTrue(res["complete"])
        self.assertEqual(res["failed_seeds"], [4])
        self.assertEqual(res["level"], "m=2.0")
        self.assertEqual(res["R4_2"]["n"], 4)
        self.assertLess(res["R4_2"]["p"], 0.05)  # the surviving seeds alone would pass
        self.assertEqual(res["marks"], dict(R4_level=True, R4_1=False, R4_2=False, R4_3a=False, R4_3b=False,
                                            R4_4=False))
        wrapped = {k: dict(score=v) if "matched" in v else v for k, v in failed.items()}
        with mock.patch.object(A, "tier_a_scores", lambda *a: (wrapped, [])):
            r7 = A.r7_evaluate(self._prereg(seeds), "base", "lean", 2.5)
        self.assertFalse(r7["marks"]["R7b"])  # a failed main seed still fails R7(b)
        self.assertEqual(r7["R7b"]["failed_seeds"], [4])

    def test_r4_unmatched_seed_at_s2_is_not_positive(self):
        """10 seeds: S2 qualifies with 8 matched (need 8). The 2 unmatched seeds leave the means and t-tests (recorded)
        and count as not positive: 7 of 8 included seeds positive is 7 of 10 < 8, so R4(1) fails, although dropping
        them silently (need ceil(0.8 x 8) = 7) would pass it."""
        seeds = tuple(range(10))
        scores = {k: _synthetic_score(0.10 + 0.005 * k, 0.08) for k in seeds}
        for k in (8, 9):
            scores[k] = _synthetic_score(0.10, 0.08, unmatched={("OR-wrong", "m=2.0")})
        res, _ = self._r4(scores, seeds)
        self.assertEqual(res["level"], "m=2.0")
        self.assertEqual(res["level_rule"]["m=2.0"]["matched"], 8)
        self.assertEqual(res["excluded_seeds"]["R4_1 (OR-wrong)"], [8, 9])
        self.assertEqual(res["excluded_seeds"]["R4_2 (OR-thin)"], [])
        self.assertEqual((res["R4_1"]["n"], res["R4_1"]["positive"]), (8, 8))
        self.assertTrue(res["marks"]["R4_1"], res["R4_1"])
        scores[7] = _synthetic_score(-0.001, 0.08)
        res, _ = self._r4(scores, seeds)
        self.assertEqual((res["R4_1"]["n"], res["R4_1"]["positive"]), (8, 7))
        self.assertGreater(res["R4_1"]["mean"], 0.05)
        self.assertLess(res["R4_1"]["p"], 0.01)
        self.assertGreaterEqual(res["R4_1"]["positive"], A.seed_need(0.8, 8))  # a silent drop would pass
        self.assertFalse(res["marks"]["R4_1"])
        self.assertTrue(res["marks"]["R4_2"] and res["marks"]["R4_level"])
        # lean path: one unmatched seed at S2 is 4 of 5 matched; it is excluded and not positive
        lean = {k: _synthetic_score(0.10 + 0.01 * k, 0.08) for k in range(5)}
        lean[2] = _synthetic_score(0.10, 0.08, unmatched={("OR-thin", "m=2.0")})
        res, _ = self._r4(lean, tuple(range(5)))
        self.assertEqual(res["level"], "m=2.0")
        self.assertEqual(res["excluded_seeds"]["R4_2 (OR-thin)"], [2])
        self.assertEqual(res["excluded_seeds"]["thin_minus_shuf"], [2])
        self.assertEqual((res["R4_2"]["n"], res["R4_1"]["n"]), (4, 5))
        self.assertTrue(all(res["marks"].values()), res["marks"])

    def test_r4_level_falls_back_to_s1_and_fails_without_one(self):
        seeds = tuple(range(5))
        scores = {k: _synthetic_score(0.10 + 0.01 * k, 0.08, s1_shift=-0.03) for k in seeds}
        for k in (0, 1):  # OR-thin matched at S2 in 3 of 5 seeds (< 4): S1, where all 5 match
            scores[k] = _synthetic_score(0.10, 0.08, unmatched={("OR-thin", "m=2.0")}, s1_shift=-0.03)
        res, P = self._r4(scores, seeds)
        self.assertEqual(res["level"], "m=1.3")
        self.assertEqual((res["level_rule"]["m=2.0"]["matched"], res["level_rule"]["m=1.3"]["matched"]), (3, 5))
        self.assertFalse(res["level_rule"]["m=2.0"]["qualifies"])
        self.assertTrue(res["marks"]["R4_level"])
        self.assertEqual(res["excluded_seeds"]["R4_2 (OR-thin)"], [])  # matched at the chosen level
        # every mean is read at S1, where OR-wrong and OR-thin sit 0.03 lower than at S2
        self.assertAlmostEqual(res["R4_1"]["mean"], np.mean([0.10, 0.10, 0.12, 0.13, 0.14]) - 0.03, places=12)
        self.assertAlmostEqual(res["R4_2"]["mean"], 0.05, places=12)
        self.assertAlmostEqual(res["anchor_means"]["thin_minus_shuf"], 0.13, places=12)
        self.assertAlmostEqual(res["anchor_means"]["wrong_minus_shuf"], res["R4_1"]["mean"] + 0.08, places=12)
        self.assertTrue(all(res["marks"].values()), res["marks"])
        # 4 of 5 at S2 qualifies S2
        scores[1] = _synthetic_score(0.11, 0.08, s1_shift=-0.03)
        self.assertEqual(self._r4(scores, seeds)[0]["level"], "m=2.0")
        # neither level: OR-wrong unmatched at both levels in 2 seeds -> R4 fails, values at S1 for information
        for k in (3, 4):
            scores[k] = _synthetic_score(0.10, 0.08, unmatched={("OR-wrong", "m=2.0"), ("OR-wrong", "m=1.3")})
        res, P = self._r4(scores, seeds)
        self.assertIsNone(res["level"])
        self.assertEqual(res["evaluated_at"], "m=1.3")
        self.assertFalse(res["marks"]["R4_level"])
        self.assertFalse(res["pass_"])
        self.assertTrue(res["complete"])
        with mock.patch.object(A, "tier_a_scores", lambda *a: ({k: dict(score=v) for k, v in scores.items()}, [])):
            r7 = A.r7_evaluate(P, "base", "lean", 2.5)
        self.assertFalse(r7["complete"])
        self.assertIn("no R4 level", r7["note"])

    def test_r7_null_and_global_linear_count_every_seed(self):
        P = self._prereg((0, 1, 2))
        main = {k: dict(score=_synthetic_score(0.10 + 0.01 * k, 0.08, extra=0.001 * k)) for k in range(3)}
        null = {k: dict(score=_synthetic_score(0.001 * k, 0.0)) for k in range(3)}

        def scores(prereg, retune, variant, alpha, seeds):
            return (main if variant == "main" else null), []

        with mock.patch.object(A, "tier_a_scores", scores):
            ok = A.r7_evaluate(P, "base", "lean", 2.5)
        self.assertTrue(ok["marks"]["R7c"] and ok["marks"]["R7_gl"] and ok["marks"]["R7b"], ok["marks"])
        self.assertEqual(ok["R7b"]["level"], "m=2.0")
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
        self.assertAlmostEqual(r3["per_seed"]["0"]["thin_shortfall_min"], 0.45, places=12)
        if not r3["pass_"]:
            with self.assertRaisesRegex(A.PreregError, "alpha 5"):
                A.resolve_alpha(P, "base", "lean", "auto")
        r2 = A.stage_r2(P, "lean", log=QUIET)
        self.assertTrue(r2["complete"])
        self.assertEqual(len(r2["ladder"]["FQE"]["rungs"]), len(cfg.ladder_rungs))
        self.assertEqual(r2["gated_critic"], "FQE")
        for mode in ("FQE", "HOST"):  # the slope is read from the saved rows of both critics (reported)
            v = r2["per_seed"][mode]["0"]
            self.assertIsNotNone(v["slope"], v["slope_detail"])
            self.assertEqual(v["slope_detail"]["file"], "errors_100.npz")
            self.assertAlmostEqual(v["thin_E2_min"], 0.45, places=12)
        self.assertEqual(set(r2["marks"]), {"main", "ladder_falls", "ladder_top"})
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
        self.assertEqual(set(r4["level_rule"]), {"m=2.0", "m=1.3"})
        self.assertIn("R4_1 (OR-wrong)", r4["excluded_seeds"])
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
        self.assertIn("Amendment A1 (2026-10-12), POST HOC", md)
        rows = {(r["id"], r["requirement"]): r for r in v["rows"]}
        for k in ("R2_rel", "R2_slope", "R2_ratio", "R2_thin_E2"):  # accuracy rows, and HOST's R2
            key = (k, A.PASS_MARKS[k]["name"] + (" - HOST critic" if k in ("R2_ratio", "R2_thin_E2") else ""))
            self.assertEqual(rows[key]["result"], "reported, not gated", key)
        self.assertIn("1.5 x their share of rows", rows[("R2_thin_E2", "R2 thin share of E^2")]["pass_mark"])
        self.assertIn("1.5 x their mix share",
                      rows[("R3_thin_shortfall", "R3 shortfall in thin contexts")]["pass_mark"])
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

"""Checks of the step-4 runner (run_step4.py): blindness, lock-step matching, Level-1 terms, the process outputs.

Blindness (design section 5): the source of run_step4, step4_stage0 and placement_harness never imports or reads
lq_harness.value / policy_gradient (AST scan), never imports score_values / step4_settings (which do), and every process
this module runs has both J functions replaced by functions that raise, so any call on the matching path would fail the
test. No J is computed anywhere in this module. Settings are tiny with seed 7, not the real configuration.

Covered: batched_match = PH.match_knob (knob, S, evaluations) on monotone, non-monotone, unreachable and A500-type
problems; level1_terms = PH.terms (P0/P1/P2/P3/EXACT recipes, plain and per-context actors); the tiny processes of
blocks X-CS (every regime, subsets on), C and R (replicate 0, q2_optimistic): hashes, every [derived] check, ladders,
matched strengths within tolerance (recomputed from the stored K's), decomposition rows, Level-1 additivity,
q2_optimistic = clean bitwise, step 3's anchors in block R, refusal to overwrite, the CLI's per-block settings; P0 at
S(HOOK) / S(OR-raw) on the m >= 1 branch (a synthetic V-shaped ladder shows the whole ladder's first crossing at m < 1,
and the tiny process's reachable reference matches are all above 1 with a dose mean above 1); the pi family's bitwise
batch multiset under the s = 0.47 robustness settings (passes; a program with the default multiset fails; the trace is
placement_loop bit for bit); OR-raw's negative BC weights recorded per row and per ladder t.
JAX_PLATFORMS=cpu python -m unittest experiments.signal.test_run_step4
"""
import ast
import functools
import json
import math
import shutil
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from experiments.signal import lq_harness as LQ
from experiments.signal import placement_harness as PH
from experiments.signal import run_step4 as RUN
from experiments.signal import score_values as SV

TINY = dict(seed=7, train_episodes=12, cal_episodes=40, cal_rows_per_episode=2, eval_episodes=6, episode_length=20,
            fit_steps=40, actor_steps=200, batch_size=64)
SIGNAL = Path(RUN.__file__).resolve().parent
FORBIDDEN_MODULES = ("score_values", "step4_settings", "scorecard_step4", "strength_reanalysis")


def j_references(path):
    """AST nodes of a module that import or read a J function or a module that computes J."""
    tree = ast.parse(Path(path).read_text(encoding="utf8"))
    bad = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            names = [a.name for a in node.names]
            if any(n in RUN.J_FUNCTIONS for n in names) or any(m in (node.module or "") or m in names
                                                                for m in FORBIDDEN_MODULES):
                bad.append((node.lineno, ast.dump(node)))
        elif isinstance(node, ast.Import):
            if any(m in a.name for a in node.names for m in FORBIDDEN_MODULES):
                bad.append((node.lineno, ast.dump(node)))
        elif isinstance(node, ast.Attribute) and node.attr in RUN.J_FUNCTIONS:
            bad.append((node.lineno, ast.dump(node)))
        elif isinstance(node, ast.Name) and node.id in RUN.J_FUNCTIONS:
            bad.append((node.lineno, ast.dump(node)))
    return bad


class NoJ:
    """Context manager: lq_harness.value and policy_gradient raise while it is active."""

    def __enter__(self):
        self.saved = {name: getattr(LQ, name) for name in RUN.J_FUNCTIONS}

        def forbidden(*args, **kwargs):
            raise AssertionError("a J function was called on the blind path")

        for name in RUN.J_FUNCTIONS:
            setattr(LQ, name, forbidden)
        return self

    def __exit__(self, *exc):
        for name, fn in self.saved.items():
            setattr(LQ, name, fn)
        return False


def settings(block, **kw):
    return RUN.settings_for(block, dict(TINY, **kw))


@functools.lru_cache(maxsize=None)
def process(block, level, rep, cases, subsets="off", actor_steps=200):
    """One tiny process (cached per test run), run with the J functions disabled."""
    root = Path(tempfile.mkdtemp(prefix="step4_run_"))
    saved = RUN.REGIMES["A5000"]["steps"]
    RUN.REGIMES["A5000"]["steps"] = 400  # the horizon regime's code path at a test-sized horizon
    try:
        with NoJ():
            out = RUN.run_process(block, level, rep, root, settings(block, actor_steps=actor_steps), "full", cases,
                                  subsets, log=lambda *a: None)
    finally:
        RUN.REGIMES["A5000"]["steps"] = saved
    payload = json.loads((out / "matched_knobs.json").read_text(encoding="utf8"))
    return out, payload, dict(np.load(out / "arrays.npz"))


class SourceTests(unittest.TestCase):
    def test_matching_path_never_references_a_J_function(self):
        for name in ("run_step4.py", "step4_stage0.py", "placement_harness.py"):
            self.assertEqual(j_references(SIGNAL / name), [], name)

    def test_the_scan_catches_references(self):
        tmp = Path(tempfile.mkdtemp())
        try:
            for src in ("from experiments.signal.lq_harness import value\n", "x = LQ.value(s, K)\n",
                        "g = LQ.policy_gradient(s, K)\n", "from experiments.signal import score_values\n",
                        "import experiments.signal.step4_settings\n"):
                (tmp / "m.py").write_text(src)
                self.assertTrue(j_references(tmp / "m.py"), src)
        finally:
            shutil.rmtree(tmp)

    def test_cli_settings_per_block(self):
        parser = RUN.build_parser()
        ns = parser.parse_args(["--block", "R", "--level", "good", "--replicate", "99", "--output", "x"])
        overrides = {k: getattr(ns, k) for k in PH.Settings.__dataclass_fields__}
        self.assertEqual(RUN.settings_for("R", overrides).seed, 20261001)
        self.assertEqual(RUN.settings_for("X-CS", overrides).seed, 20261002)
        ns = parser.parse_args(["--block", "C", "--level", "poor", "--replicate", "999", "--output", "x",
                                "--actor-steps", "15", "--plan", "reduced", "--cases", "clean"])
        cfg = RUN.settings_for("C", {k: getattr(ns, k) for k in PH.Settings.__dataclass_fields__})
        self.assertEqual((cfg.actor_steps, cfg.seed, ns.plan, ns.cases), (15, 20261002, "reduced", ["clean"]))

    def test_grid(self):
        self.assertEqual(len(RUN.grid("R", 0)), 3 * 12)  # 9 cases + q2_optimistic x 3 kappas per offset
        self.assertEqual(len(RUN.grid("R", 3)), 27)
        self.assertEqual(len(RUN.grid("X-CS", 0)), 11)
        self.assertEqual(len(RUN.grid("C", 5)), 3)
        self.assertEqual(RUN.grid("X-EP", 1, cone_kappas=(2.0, 3.0))[-1].id, "cone_k3")
        for block in RUN.BLOCKS:
            self.assertEqual(RUN.grid(block, 0)[0].case, "clean")  # NUIS comes from clean


class MatchingTests(unittest.TestCase):
    def sequential(self, problems, fns):
        return [PH.match_knob(p["ladder"], f, p["target"], p["tol"], p["max_bisect"], p["interpolate"],
                              S_ladder=[f(k) for k in p["ladder"]]) for p, f in zip(problems, fns)]

    def test_batched_match_is_match_knob(self):
        """Lock-step rounds reproduce match_knob's knob sequence and result exactly, FP- and A500-type."""
        rng = np.random.default_rng(1)
        fns, problems = [], []
        for i in range(40):
            a, b, bump = rng.uniform(0.01, 0.1), rng.uniform(0.3, 2), rng.uniform(0, 0.03) * (i % 3 == 0)
            f = (lambda k, a=a, b=b, bump=bump: a * k ** b / (1 + 0.1 * k) + bump * math.sin(5 * math.log(k)))
            fns.append(f)
            ladder = PH.BETA_LADDER if i % 2 else PH.C_LADDER
            cfg = RUN.MATCH["FP" if i % 4 < 2 else "ADAM"]
            target = f(ladder[0]) * (0.5 if i == 7 else 1) + rng.uniform(0, 1) * (f(ladder[-1]) - f(ladder[0]))
            if i == 9:
                target = 10 * f(ladder[-1])  # unreachable
            problems.append(dict(ladder=ladder, S_ladder=[f(k) for k in ladder], target=target, **cfg))
        calls = []

        def evaluate(need):
            calls.append(len(need))
            return [fns[i](k) for i, k in need]

        got = RUN.batched_match(problems, evaluate)
        want = self.sequential(problems, fns)
        for g, w in zip(got, want):
            self.assertEqual(g, w)
        self.assertLessEqual(len(calls), 41)
        self.assertFalse(got[9]["reachable"])

    def test_p0_reference_branch(self):
        """P0(1) = NONE makes S(P0(m)) V-shaped; a reference with dose mean 1.233 must match P0 near m = 1.233, not on
        the m < 1 branch where the whole ladder's first crossing lies (the review's C / poor clean example)."""
        S = lambda m: 0.02 * abs(m - 1.0) * (1.5 if m < 1 else 1.0)  # S(0.5) = 0.015 > S(1.3) = 0.006
        target = S(1.233)
        evaluate = lambda need: [S(k) for _, k in need]
        problem = lambda ladder: dict(ladder=ladder, S_ladder=[S(m) for m in ladder], target=target, **RUN.MATCH["FP"])
        whole = RUN.batched_match([problem(PH.P0_LADDER)], evaluate)[0]
        self.assertLess(whole["knob"], 1.0)  # the defect
        ladder, label = RUN.p0_branch(1.233)
        self.assertEqual((ladder, label), ((1.0, 1.15, 1.3, 1.6, 2.0, 2.5, 3.0, 4.0, 6.0, 10.0), "m>=1"))
        up = RUN.batched_match([problem(ladder)], evaluate)[0]
        self.assertTrue(up["converged"] and up["monotone"])
        self.assertGreater(up["knob"], 1.0)
        self.assertAlmostEqual(up["knob"], 1.233, delta=2e-3)
        self.assertEqual(RUN.p0_branch(0.8), ((0.5, 0.75, 1.0), "m<=1"))
        self.assertEqual(RUN.p0_branch(1.0)[1], "m>=1")

    def test_negative_weight_count(self):
        o = np.array([0.1, 0.5, 0.9, 1.5, 3.0])
        recipe = lambda t: dict(v=1.0 + t * (o - 1.0))
        self.assertEqual([RUN.negative_weights((recipe, t), 5) for t in (0.5, 1.0, 1.2, 2.0, 10.0)], [0, 0, 1, 1, 2])
        self.assertEqual(RUN.negative_weights((lambda m: dict(v=m), 2.0), 5), 0)
        self.assertEqual(RUN.negative_weights((lambda b: dict(u=0.5), 1.0), 5), 0)
        self.assertIsNone(RUN.negative_weights(None, 5))

    def test_helpers(self):
        self.assertEqual([RUN.perm_index(n) for n in ("SHUF3", "STRAT12", "SIG", "SHUFLIN2", "NUIS")], [3, 12, -1, -1, -1])
        self.assertEqual([RUN.bucket(n) for n in (1, 4, 5, 157, 256)], [4, 4, 8, 160, 256])
        tab = RUN.Table((2, 3))
        for i in range(6):
            tab.add(np.full((2, 3), i), ("x", i, "eval"))
        tab.index.update(NONE=0, EXACT=2)
        tab.index["ladder"]["P0"] = [2, 4]
        tab.index["matched"]["P1L:SIG"] = {"S2": dict(row=5, knob=1.0), "S1": dict(row=None)}
        tab.index["decomp"]["P1L:SIG"] = {"S2": dict(reg=3, perp=5)}
        keep = tab.kept_rows()
        self.assertEqual(keep, [0, 2, 3, 4, 5])
        new = RUN.remap(tab.index, {old: i for i, old in enumerate(keep)})
        self.assertEqual((new["NONE"], new["EXACT"], new["ladder"]["P0"]), (0, 1, [1, 3]))
        self.assertEqual(new["matched"]["P1L:SIG"]["S2"]["row"], 4)
        self.assertIsNone(new["matched"]["P1L:SIG"]["S1"]["row"])
        self.assertEqual(new["decomp"]["P1L:SIG"]["S2"], dict(reg=2, perp=4))

    def test_jsonable_is_strict(self):
        x = RUN.jsonable(dict(a=np.float32(1.5), b=float("nan"), c=np.array([1.0, np.inf]), d=(np.int64(2), True)))
        self.assertEqual(json.loads(json.dumps(x, allow_nan=False)), dict(a=1.5, b=None, c=[1.0, None], d=[2, True]))


class Level1Tests(unittest.TestCase):
    """level1_terms (per-row K-gradients, float64) against PH.terms (autodiff of the placement loss, float32)."""

    def cellrun(self, block, case, **kw):
        h = PH.PlacementHarness(settings(block))
        level = RUN.LEVELS[block][0]
        data = RUN.block_data(h, block, level, 0)
        stream = RUN.block_stream(h, block, level, 0, len(data["train"]["state"]))
        run = RUN.CellRun(h, block, level, 0, RUN.CellSpec(case, **kw), data, stream, {})
        with NoJ():
            run.signal()
        return h, run

    def compare(self, h, run):
        n, rng = run.n, np.random.default_rng(2)
        w = run.assign["weights"]["L"]["P1"]["SIG"]
        cp = run.cell.native.critic.params
        pen = PH.penalty_fn("sig", run.cell.models, cp, PH.loop_context(run.cell))
        exact = lambda o, a: run.cell.models[1].exact(cp, o, a)
        cases = [(dict(), dict()), (dict(v=1 + 2.0 * w), dict(bc_w=1 + 2.0 * w)),
                 (dict(u=1 / (1 + 3.0 * w)), dict(q_w=1 / (1 + 3.0 * w))),
                 (dict(pen="sig", c=0.7), dict(pen_fn=pen, pen_c=0.7)), (dict(qread="exact"), dict(exact_fn=exact)),
                 (dict(v=rng.uniform(0.5, 2, n), u=rng.uniform(0.5, 1, n)), None)]
        for recipe, hooks in cases:
            mine = run.level1_terms(recipe)
            if hooks is None:
                hooks = dict(bc_w=recipe["v"], q_w=recipe["u"])
            ref = PH.terms(h.args, run.cell.models, run.cell.params, run.K0, run.obs, run.a, **hooks)
            for key in ("g_Q", "g_pen", "g_BC", "g"):
                scale = max(np.linalg.norm(ref["g"]), 1e-12)
                self.assertLess(np.linalg.norm(mine[key] - ref[key]) / scale, 2e-4, (recipe.keys(), key))
            self.assertAlmostEqual(mine["lam"], ref["lam"], delta=1e-5 * ref["lam"])

    def test_plain_actor(self):
        self.compare(*self.cellrun("X-CS", "tilt", a_star=-0.5))

    def test_context_actor(self):
        self.compare(*self.cellrun("C", "q1opt-local", kappa=1.0))


class ProcessTests(unittest.TestCase):
    """A tiny X-CS process: clean, q1_optimistic and the four tilts, every regime (subsets on), J disabled."""

    @classmethod
    def setUpClass(cls):
        cls.out, cls.payload, cls.arrays = process("X-CS", "expert", 0, ("clean", "q1_optimistic", "tilt"), "on")
        h = PH.PlacementHarness(settings("X-CS"))
        cls.sigma_ev = PH.second_moment(RUN.block_data(h, "X-CS", "expert", 0)["eval"]["state"])

    def test_hashes_and_no_values(self):
        recorded = SV.verify(self.out)
        self.assertEqual(set(recorded), {"matched_knobs.json", "arrays.npz"})
        self.assertFalse(any(k.startswith("J/") for k in self.arrays))
        self.assertFalse((self.out / "values.npz").exists())
        with self.assertRaises(FileExistsError):
            RUN.run_process("X-CS", "expert", 0, self.out.parents[2], settings("X-CS"), cases=("clean",),
                            log=lambda *a: None)

    def test_every_derived_check_passes(self):
        for cid, rec in self.payload["cells"].items():
            names = {c["name"] for c in rec["checks"]}
            self.assertTrue({"same_multiset_global", "zero_knob_fp", "zero_knob_a500", "scale_equivalence_fp",
                             "hooks_off_is_td3_bc_update", "hook_dose_is_step3_bca_path", "pi_batch_multiset"} <= names,
                            cid)
            pi = next(c for c in rec["checks"] if c["name"] == "pi_batch_multiset")
            self.assertEqual(pi["kind"], "derived")  # bitwise, not the old Kish proxy
            for c in rec["checks"]:
                if c["kind"] != "procedural":
                    self.assertTrue(c["ok"], (cid, c))
        q1 = {c["name"] for c in self.payload["cells"]["q1_optimistic_k1"]["checks"]}
        self.assertTrue({"cancellation_q1_optimistic", "p3_oracle_fp_is_exact", "p3_oracle_action_gradient"} <= q1)
        self.assertIn("nuis_is_sig_in_clean", {c["name"] for c in self.payload["cells"]["clean"]["checks"]})

    def test_tables_ladders_and_matches(self):
        for cid, rec in self.payload["cells"].items():
            self.assertEqual(set(rec["regimes"]), {"FP", "A500"} | ({"A5000", "FB"} if cid.startswith("tilt") else set()))
            for reg, r in rec["regimes"].items():
                K = self.arrays[f"K/{cid}/{reg}"]
                S = self.arrays[f"D/{cid}/{reg}/S"]
                idx = r["index"]
                self.assertEqual(len(r["labels"]), len(K))
                self.assertEqual(S[idx["NONE"]], 0.0)
                np.testing.assert_allclose(S, PH.strength(K, K[idx["NONE"]], self.sigma_ev), rtol=1e-12, atol=1e-15)
                for arm, rows in idx["ladder"].items():
                    self.assertTrue(all(r["labels"][i][0] == arm and r["labels"][i][2] == "ladder" for i in rows))
                tol = RUN.MATCH["FP" if reg == "FP" else "ADAM"]["tol"]
                targets = idx["targets"]["usable"]
                for arm, levels in idx["matched"].items():
                    for level, m in levels.items():
                        if m.get("row") is None:
                            continue
                        self.assertEqual(r["labels"][m["row"]][0], arm)
                        self.assertEqual(r["labels"][m["row"]][1], m["knob"])
                        if arm == "P0" and level in RUN.LEVEL_NAMES:
                            self.assertEqual(m["row"], idx["ladder"]["P0"][PH.P0_LADDER.index(m["knob"])])
                        elif level in RUN.LEVEL_NAMES and m["converged"]:
                            self.assertLessEqual(abs(S[m["row"]] / targets[level] - 1), tol + 1e-12, (cid, reg, arm))

    def test_decomposition_rows_and_level1(self):
        for cid, rec in self.payload["cells"].items():
            for reg, r in rec["regimes"].items():
                K, idx = self.arrays[f"K/{cid}/{reg}"], r["index"]
                for arm, levels in idx["decomp"].items():
                    for level, parts in levels.items():
                        arm_K = K[idx["matched"][arm][level]["row"]]
                        np.testing.assert_allclose(K[parts["reg"]] + K[parts["perp"]] - K[idx["NONE"]], arm_K,
                                                   rtol=0, atol=1e-12)
                L1 = self.arrays[f"D/{cid}/{reg}/L1"]
                ok = np.all(np.isfinite(L1), axis=tuple(range(1, L1.ndim)))
                np.testing.assert_allclose(L1[ok, 0] + L1[ok, 1] + L1[ok, 2], L1[ok, 3], rtol=1e-12, atol=1e-14)
                self.assertTrue(ok[idx["NONE"]])

    def test_p0_reference_matches_above_one(self):
        """Every reachable P0 at S(HOOK) / S(OR-raw) lies on m >= 1 when the dose mean is above 1, matches the
        reference's S within tolerance, and records its branch."""
        reached = 0
        for cid, rec in self.payload["cells"].items():
            dose_mean = rec["signal"]["dose_mean"]
            for reg in ("FP", "A500"):
                r = rec["regimes"][reg]
                idx, S = r["index"], self.arrays[f"D/{cid}/{reg}/S"]
                tol = RUN.MATCH["FP" if reg == "FP" else "ADAM"]["tol"]
                for ref in ("HOOK", "OR-raw"):
                    m = idx["matched"]["P0"][f"S({ref})"]
                    self.assertEqual(m["branch"], "m>=1" if m["ref_mean_weight"] >= 1 else "m<=1")
                    self.assertAlmostEqual(m["ref_mean_weight"], dose_mean, delta=1e-5 * dose_mean)
                    if m.get("row") is None:
                        continue
                    reached += 1
                    if dose_mean > 1:
                        self.assertGreaterEqual(m["knob"], 1.0, (cid, reg, ref, m))
                    if m["converged"]:
                        self.assertLessEqual(abs(S[m["row"]] / S[idx["refs"][ref]] - 1), tol + 1e-12, (cid, reg, ref))
        self.assertGreater(reached, 0)

    def test_or_raw_weights_are_clipped_and_the_clipped_rows_recorded(self):
        """Freeze decision G: every BC weight is clipped at 0, so no recipe has a negative weight; the or_raw record
        counts the rows the clip touched (1 + t (o_i - 1) < 0) at each ladder t."""
        clipped_somewhere = False
        for cid, rec in self.payload["cells"].items():
            info = rec["or_raw"]
            neg = self.arrays[f"D/{cid}/FP/neg_bc"]
            idx, labels = rec["regimes"]["FP"]["index"], rec["regimes"]["FP"]["labels"]
            for row, t in zip(idx["ladder"]["ORraw"], RUN.OR_T_LADDER):
                self.assertEqual(labels[row][1], t)
                self.assertEqual(neg[row], 0)  # clipped
                clipped = info["neg_rows"][f"{t:g}"]
                if info["t_nonneg"] is None or t <= info["t_nonneg"]:
                    self.assertEqual(clipped, 0)
                else:
                    self.assertGreater(clipped, 0)
                    clipped_somewhere = True
            finite = [i for i in range(len(labels)) if np.isfinite(neg[i])]
            self.assertTrue(finite)
            self.assertTrue(np.all(neg[finite] == 0), cid)
        self.assertTrue(clipped_somewhere)  # the tiny process does exercise the clip

    def test_or_raw_helper_clips_at_zero(self):
        dose, error = np.full(6, 1.3, np.float32), np.array([0.0, 0.1, 0.5, 1.0, 3.0, 6.0])
        perm = np.arange(6)
        v, _ = PH.or_raw(dose, error, perm, t=3.0)
        self.assertTrue(np.all(v >= 0))
        raw, _ = PH.or_raw(dose, error, perm, t=1.0)
        self.assertTrue(np.any(1.0 + 3.0 * (raw.astype(np.float64) - 1.0) < 0))  # the clip is not vacuous here

    def test_regimes_see_reachable_targets(self):
        """At 200 steps the A500 targets clear the floor somewhere, so Adam matching ran (not only FP)."""
        reached = [m for rec in self.payload["cells"].values() for reg in ("A500", "A5000")
                   for levels in rec["regimes"].get(reg, {"index": {"matched": {}}})["index"]["matched"].values()
                   for m in levels.values() if m.get("row") is not None and m.get("evaluations")]
        self.assertTrue(reached)
        meta = self.payload["meta"]
        self.assertEqual(meta["settings"]["seed"], 7)
        self.assertTrue(meta["subsets"])


class PiMultisetTests(unittest.TestCase):
    """Part 2 item 3, family pi at A500, under the s = 0.47 robustness settings (--rank-s 0.47): every step's sorted
    batch weights of every pi arm equal multiset(B, 0.47, 2.5) bit for bit; a program built with the default multiset
    fails the check; pi_weight_trace is placement_loop bit for bit. J disabled."""

    @classmethod
    def setUpClass(cls):
        cls.h = PH.PlacementHarness(settings("X-CS", rank_s=0.47, actor_steps=30))
        data = RUN.block_data(cls.h, "X-CS", "expert", 0)
        stream = RUN.block_stream(cls.h, "X-CS", "expert", 0, len(data["train"]["state"]))
        cls.cell_run = RUN.CellRun(cls.h, "X-CS", "expert", 0, RUN.CellSpec("tilt", a_star=0.0), data, stream, {})
        with NoJ():
            cls.cell_run.signal()
            cls.cell_run.adam_setup("A500")
        cls.env = cls.cell_run.adam_env["A500"]

    def test_robust_multiset_passes_bitwise(self):
        with NoJ():
            gaps = self.cell_run.pi_multiset_gaps("A500")
            self.cell_run.check_pi_multiset("A500")
        arms = {f"{p}:{a}" for p, a in self.env["pi_maps"]}
        self.assertEqual(set(gaps), arms)
        self.assertTrue({"P1:SIG", "P1:ANTI", "P1:SHUF3", "P2:STRAT2", "P1:OR1", "P2:OR2"} <= arms)
        self.assertTrue(all(same and gap == 0.0 for same, gap in gaps.values()), gaps)
        check = self.cell_run.checks[-1]
        self.assertEqual((check["name"], check["kind"], check["ok"]), ("pi_batch_multiset", "derived", True))
        # the old target (default s = 0.83) is far from this run's multiset: Kish 0.811 against 0.545 at n = 10,000
        kish = lambda M: M.sum() ** 2 / (len(M) * np.sum(M ** 2))
        self.assertAlmostEqual(kish(PH.multiset(10000, 0.47, 2.5)), 0.811, places=3)
        self.assertAlmostEqual(kish(PH.multiset(10000)), 0.545, places=3)

    def test_default_multiset_fails(self):
        saved = dict(self.env["progs"])
        try:
            for name, (sp, ctx, ties) in saved.items():
                if name.startswith("pi:"):
                    self.env["progs"][name] = (replace(sp, rank_s=PH.RANK_S), ctx, ties)
            with NoJ():
                gaps = self.cell_run.pi_multiset_gaps("A500")
            self.assertTrue(gaps and not any(same for same, _ in gaps.values()))
            self.assertTrue(all(gap > 0.1 for _, gap in gaps.values()))
        finally:
            self.env["progs"].clear()
            self.env["progs"].update(saved)

    def test_trace_is_placement_loop(self):
        h, run, env = self.h, self.cell_run, self.env
        for p, arm, knob in (("P2", "STRAT1", dict(beta=2.0)), ("P1", "OR2", dict(b=3.0))):
            prog, j = env["pi_maps"][(p, arm)]
            sp, ctx, ties = env["progs"][prog]
            sp = replace(sp, diagnostics=False)
            inputs = {k: jnp.asarray(knob.get(k, d), jnp.float32) for k, d in (("a", 1.0), ("b", 0.0), ("beta", 0.0),
                                                                               ("c", 0.0))}
            inputs["assign"] = jnp.asarray(j, jnp.int32)
            args = (run.cell.native, run.train32, env["rows"], env["keys"], ties, ctx, inputs)
            with NoJ():
                loop = jax.jit(functools.partial(PH.placement_loop, h.args, run.cell.models, sp))(*args)[0]
                trace, w = jax.jit(functools.partial(RUN.pi_weight_trace, h.args, run.cell.models, sp))(*args)
            np.testing.assert_array_equal(np.asarray(trace.actor.params["K"]), np.asarray(loop.actor.params["K"]))
            M = np.asarray(PH.multiset(h.cfg.batch_size, 0.47, 2.5), np.float32)
            self.assertEqual(np.asarray(w).shape, (h.cfg.actor_steps, h.cfg.batch_size))
            np.testing.assert_array_equal(np.sort(np.asarray(w), axis=-1), np.broadcast_to(M, np.shape(w)))


class BlockTests(unittest.TestCase):
    def test_block_c_process(self):
        out, payload, arrays = process("C", "expert", 0, ("clean", "q1opt-local"), "off", 30)
        rec = payload["cells"]["q1opt-local_k1"]
        names = {c["name"]: c for c in rec["checks"]}
        for name in ("cancellation_q1opt-local", "context_locality_fp", "p3_oracle_fp_is_exact"):
            self.assertTrue(names[name]["ok"], names[name])
        self.assertEqual(arrays["K/q1opt-local_k1/FP"].shape[1:], (4, 2, 3))
        self.assertIn("C2", rec["regimes"]["FP"]["index"]["ladder"])
        for c in rec["checks"] + payload["cells"]["clean"]["checks"]:
            if c["kind"] != "procedural":
                self.assertTrue(c["ok"], c)

    def test_block_r_process(self):
        out, payload, arrays = process("R", "poor", 0, ("clean", "q2_optimistic"), "off", 30)
        cells = payload["cells"]
        self.assertEqual(len(cells), 12)
        for cid, rec in cells.items():
            for c in rec["checks"]:
                if c["kind"] != "procedural":
                    self.assertTrue(c["ok"], (cid, c))
            self.assertEqual(set(rec["regimes"]["A500"]["index"]["anchors"]), set(SV.STEP3_ARMS))
            if rec["case"] == "q2_optimistic":
                self.assertIn("q2_optimistic_is_clean", {c["name"] for c in rec["checks"]})
            if rec["case"] == "clean":
                self.assertIsNotNone(rec["diagnostics"]["alignment_K_step3"])
                np.testing.assert_allclose(arrays[f"K/{cid}/K0"], PH.QUALITY_GAIN["poor"] * 0
                                           + rec["offset"] * LQ.DIRECTION, rtol=0, atol=1e-15)


if __name__ == "__main__":
    unittest.main()

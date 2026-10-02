"""Checks of score_values.py on synthetic process directories: it refuses unhashed or changed K's, computes J of every
stored K, and derives the J-dependent Level-1 numbers, frontier quantities, decomposition and checks; it refuses the
pilot outside --pilot-power and, in that mode, evaluates J only for the power check's rows; it refuses block R's
continuity cells without step 3's file and fails a cell without a step-3 record.

Every K here is a toy gain (K_0 plus seeded perturbations, a nonfinite row, an unstable row), written in the runner's
file layout; no step-4 arm and no real configuration is scored.
JAX_PLATFORMS=cpu python -m unittest experiments.signal.test_score_values
"""
import json
import math
import shutil
import tempfile
import unittest
from dataclasses import asdict
from pathlib import Path
from unittest import mock

import numpy as np

from experiments.signal import lq_harness as LQ
from experiments.signal import placement_harness as PH
from experiments.signal import run_step4 as RUN
from experiments.signal import score_values as SV

SYSTEM = LQ.default_system()
K_STAR = LQ.lqr_gain(SYSTEM)
K0 = 0.5 * K_STAR + 0.5 * LQ.DIRECTION


def toy_gains(rng, n):
    return K0 + 0.05 * rng.standard_normal((n, 2, 3))


def write_process(root, block="X-CS", level="expert", rep=0, settings=None, contexts=False, seed=0, cells=None,
                  arrays=None, pilot=False):
    """A process directory in run_step4's layout with toy gains; returns its path and the arrays written."""
    rng = np.random.default_rng(seed)
    d = Path(root) / block / level / f"rep{rep}"
    d.mkdir(parents=True)
    settings = settings or asdict(PH.Settings())
    if cells is None:
        fp = toy_gains(rng, 8)
        fp[4] = np.nan  # an unbounded fixed point
        fp[5] = -5.0 * K_STAR  # an unstable closed loop
        fp[6], fp[7] = fp[0] + 0.3 * (fp[2] - fp[0]), fp[0] + 0.7 * (fp[2] - fp[0])  # decomposition rows
        a500 = toy_gains(rng, 2)
        a500[1] = a500[0] + 1e-7
        frontier = toy_gains(rng, 5)
        L1 = rng.standard_normal((8, 4, 2, 3))
        L1[:, 3] = L1[:, 0] + L1[:, 1] + L1[:, 2]
        L1[[4, 5, 6, 7]] = np.nan
        arrays = {"K/c1/K0": K0, "K/c1/FP": fp, "K/c1/A500": a500, "K/c1/FP_frontier": frontier,
                  "D/c1/FP/frontier": np.array([[2, .1], [0, .2], [4, .3], [1, .1], [-1, np.nan], [3, .1], [-1, 0],
                                                [-1, 0]], float),
                  "D/c1/FP/L1": L1, "D/c1/vbar_none": rng.uniform(0.5, 2, (2, 3))}
        index = lambda **kw: dict(dict(ladder={}, matched={}, refs={}, anchors={}, decomp={}, checks={}), **kw)
        cells = {"c1": dict(case="tilt", kappa=None, offset=None, checks=[], signal={}, diagnostics={}, regimes=dict(
            FP=dict(labels=[["x", None, "eval"]] * 8,
                    index=index(NONE=0, EXACT=1, matched={"P1L:SIG": {"S2": dict(row=2)}},
                                decomp={"P1L:SIG": {"S2": dict(reg=6, perp=7)}})),
            A500=dict(labels=[["NONE", None, "base"], ["NONE_path", None, "base"]],
                      index=index(NONE=0, NONE_path=1))))}
    np.savez_compressed(d / "arrays.npz", **arrays)
    meta = dict(block=block, level=level, replicate=rep, settings=settings, pilot=pilot)
    (d / "matched_knobs.json").write_text(json.dumps(RUN.jsonable(dict(meta=meta, cells=cells))), encoding="utf8")
    (d / "HASHES.sha256").write_text("".join(f"{RUN.sha256_file(d / n)}  {n}\n" for n in SV.REQUIRED),
                                     encoding="utf8")
    return d, arrays


class ScoreTests(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="score_"))

    def tearDown(self):
        shutil.rmtree(self.tmp)

    def test_J_of_every_stored_K(self):
        d, arrays = write_process(self.tmp)
        values = SV.run([self.tmp], step3_path=None, log=lambda *a: None)[str(d)]
        J = np.load(d / "values.npz")
        fp = arrays["K/c1/FP"]
        expect = [LQ.value(SYSTEM, k) if np.all(np.isfinite(k)) else math.nan for k in fp]
        np.testing.assert_array_equal(J["J/c1/FP"], expect)
        self.assertTrue(math.isnan(J["J/c1/FP"][4]))
        self.assertEqual(J["J/c1/FP"][5], -math.inf)
        self.assertEqual(float(J["J/c1/K0"]), LQ.value(SYSTEM, K0))
        np.testing.assert_array_equal(J["J/c1/FP_frontier"], [LQ.value(SYSTEM, k) for k in arrays["K/c1/FP_frontier"]])
        cell = values["cells"]["c1"]
        self.assertEqual(cell["gap"], LQ.value(SYSTEM, K_STAR) - LQ.value(SYSTEM, K0))
        self.assertEqual(SV.verify(d), SV.read_hashes(d / "HASHES.sha256"))
        recorded = SV.read_hashes(d / SV.VALUES_HASH_FILE)
        self.assertEqual(recorded, {n: RUN.sha256_file(d / n) for n in ("values.json", "values.npz")})
        self.assertEqual(json.loads((d / "values.json").read_text())["meta"]["inputs"], SV.read_hashes(d / SV.HASH_FILE))

    def test_level1_frontier_decomposition_and_checks(self):
        d, arrays = write_process(self.tmp)
        values = SV.score_dir(d, log=lambda *a: None)
        J = np.load(d / "values.npz")
        gJ = LQ.policy_gradient(SYSTEM, K0)
        np.testing.assert_array_equal(J["G/c1/gradJ_K0"], gJ)
        L1, L1J = arrays["D/c1/FP/L1"], J["L1J/c1/FP"]
        none = dict(g_Q=L1[0, 0], g_pen=L1[0, 1], g_BC=L1[0, 2], g=L1[0, 3])
        for i in (1, 2, 3):
            m = PH.level1(dict(g_Q=L1[i, 0], g_pen=L1[i, 1], g_BC=L1[i, 2], g=L1[i, 3]), none, grad_J=gJ,
                          v_bar=arrays["D/c1/vbar_none"])
            np.testing.assert_allclose(L1J[i], [m[k] for k in SV.L1J_FIELDS], rtol=1e-12)
        self.assertTrue(np.all(np.isnan(L1J[4:])))
        gap = values["cells"]["c1"]["gap"]
        Jfp, Jf = J["J/c1/FP"], J["J/c1/FP_frontier"]
        G, Gf = (Jfp - Jfp[0]) / gap, (Jf - Jfp[0]) / gap
        F = J["F/c1/FP"]
        self.assertAlmostEqual(F[2, 0], G[2] - Gf[4], places=12)
        self.assertAlmostEqual(F[3, 1], Gf.max() - G[3], places=12)
        self.assertTrue(math.isnan(F[4, 0]))
        dec = values["cells"]["c1"]["decomposition"]["FP"]["P1L:SIG"]["S2"]
        self.assertAlmostEqual(dec["G"], G[2], places=12)
        self.assertAlmostEqual(dec["G_reg"], G[6], places=12)
        self.assertAlmostEqual(dec["G_perp"] + dec["G_reg"], dec["G"], places=12)
        self.assertAlmostEqual(dec["G_perp_first"], G[7], places=12)
        check = next(c for c in values["checks"] if c["name"].startswith("none_weighted_vs_none_path"))
        self.assertEqual(check["kind"], "eps")
        self.assertAlmostEqual(check["value"], abs(J["J/c1/A500"][1] - J["J/c1/A500"][0]), places=15)
        self.assertTrue(check["ok"])

    def test_context_J_is_the_mean_over_contexts(self):
        rng = np.random.default_rng(4)
        K = K0 + 0.05 * rng.standard_normal((3, 4, 2, 3))
        J = SV.J_of(SYSTEM, K, contexts=True)
        np.testing.assert_allclose(J, [np.mean([LQ.value(SYSTEM, k) for k in Kc]) for Kc in K], rtol=0, atol=0)
        g = SV.grad_J(SYSTEM, np.tile(K0, (4, 1, 1)), contexts=True)
        np.testing.assert_allclose(g[2], LQ.policy_gradient(SYSTEM, K0) / 4, rtol=1e-12)
        with self.assertRaises(ValueError):
            SV.J_rows(SYSTEM, np.zeros((3, 3, 2)))


class RefusalTests(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="refuse_"))

    def tearDown(self):
        shutil.rmtree(self.tmp)

    def test_missing_hash_file(self):
        d, _ = write_process(self.tmp)
        (d / "HASHES.sha256").unlink()
        with self.assertRaises(SV.Refused):
            SV.run([d], step3_path=None, log=lambda *a: None)
        self.assertFalse((d / "values.npz").exists())
        with self.assertRaises(SystemExit) as ctx:
            SV.main([str(d)])
        self.assertEqual(ctx.exception.code, 2)

    def test_changed_after_hashing(self):
        d, arrays = write_process(self.tmp)
        arrays["K/c1/FP"][3] += 1e-12
        np.savez_compressed(d / "arrays.npz", **arrays)
        with self.assertRaises(SV.Refused):
            SV.verify(d)
        d2, _ = write_process(self.tmp, level="poor", seed=1)
        lines = (d2 / "HASHES.sha256").read_text().splitlines()
        (d2 / "HASHES.sha256").write_text(lines[1] + "\n")  # matched_knobs.json no longer listed
        with self.assertRaises(SV.Refused):
            SV.verify(d2)

    def test_one_refused_directory_stops_all(self):
        good, _ = write_process(self.tmp, level="expert")
        bad, _ = write_process(self.tmp, level="poor", seed=1)
        (bad / "HASHES.sha256").unlink()
        with self.assertRaises(SV.Refused):
            SV.run([self.tmp], step3_path=None, log=lambda *a: None)
        self.assertFalse((good / "values.npz").exists())

    def test_no_overwrite(self):
        d, _ = write_process(self.tmp)
        SV.score_dir(d, log=lambda *a: None)
        with self.assertRaises(FileExistsError):
            SV.score_dir(d, log=lambda *a: None)
        SV.score_dir(d, overwrite=True, log=lambda *a: None)


class ContinuityTests(unittest.TestCase):
    """Block R's comparison with step 3's results.json, on a synthetic step-3 file built from toy anchors."""

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="cont_"))
        rng = np.random.default_rng(5)
        anchors = toy_gains(rng, 7)
        settings = asdict(PH.Settings(seed=20261001))
        index = dict(ladder={}, matched={}, refs={}, decomp={}, checks={}, NONE=0, NONE_path=1,
                     anchors={arm: i + 2 for i, arm in enumerate(SV.STEP3_ARMS)})
        diag = dict(q_term_grad_norm=1.25, alignment_K_step3=0.97,
                    **{f"bc_term_grad_norm_{a}": 0.1 * i for i, a in enumerate(SV.STEP3_ARMS)})
        signal = dict(threshold=1.7, dose_mean=1.3, dose_sd=0.02)
        cells = {"clean_o0.5": dict(case="clean", kappa=None, offset=0.5, checks=[], signal=signal, diagnostics=diag,
                                    regimes=dict(A500=dict(labels=[["x", None, "e"]] * 7, index=index)))}
        arrays = {"K/clean_o0.5/K0": K0, "K/clean_o0.5/A500": anchors}
        self.dir, _ = write_process(self.tmp, block="R", level="good", rep=3, settings=settings, cells=cells,
                                    arrays=arrays)
        J0 = LQ.value(SYSTEM, K0)
        metrics = {f"J_change_{a}": dict(values=[None] * 3 + [LQ.value(SYSTEM, anchors[i + 2]) - J0, None])
                   for i, a in enumerate(SV.STEP3_ARMS)}
        metrics.update({name: dict(values=[None] * 3 + [v, None]) for name, v in
                        (("threshold", 1.7), ("dose_mean", 1.3), ("dose_sd", 0.02), ("q_term_grad_norm", 1.25),
                         ("alignment_K", 0.97))})
        metrics.update({f"bc_term_grad_norm_{a}": dict(values=[None] * 3 + [0.1 * i, None])
                        for i, a in enumerate(SV.STEP3_ARMS)})
        self.step3 = dict(meta=dict(settings={k: settings[k] for k in SV.STEP3_SETTINGS}),
                          results=[dict(behavior="good", case="clean", kappa=None, pi_offset=0.5, signal="q1",
                                        metrics=metrics)])
        self.path = self.tmp / "results.json"

    def tearDown(self):
        shutil.rmtree(self.tmp)

    def score(self):
        self.path.write_text(json.dumps(self.step3))
        return SV.score_dir(self.dir, SV.step3_records(self.path), overwrite=True, log=lambda *a: None)

    def test_bitwise_match_passes_and_any_difference_fails(self):
        check = next(c for c in self.score()["checks"] if c["name"].startswith("step3_continuity"))
        self.assertTrue(check["ok"], check)
        self.step3["results"][0]["metrics"]["J_change_oracle"]["values"][3] += 1e-15
        self.step3["results"][0]["metrics"]["threshold"]["values"][3] = 1.7000001
        check = next(c for c in self.score()["checks"] if c["name"].startswith("step3_continuity"))
        self.assertFalse(check["ok"])
        self.assertEqual(len(check["value"]), 2)

    def test_not_applicable_under_other_settings(self):
        self.step3["meta"]["settings"]["fit_steps"] = 4999
        check = next(c for c in self.score()["checks"] if c["name"] == "step3_continuity")
        self.assertTrue(check["ok"])
        self.assertIn("not applicable", check["value"])

    def test_cell_without_a_step3_record_fails(self):
        payload = json.loads((self.dir / "matched_knobs.json").read_text())
        arrays = dict(np.load(self.dir / "arrays.npz"))
        cells = dict(payload["cells"], **{"clean_o1": dict(payload["cells"]["clean_o0.5"], offset=1.0)})
        arrays.update({"K/clean_o1/K0": arrays["K/clean_o0.5/K0"], "K/clean_o1/A500": arrays["K/clean_o0.5/A500"]})
        d, _ = write_process(self.tmp / "second", block="R", level="good", rep=3, settings=payload["meta"]["settings"],
                             cells=cells, arrays=arrays)
        self.path.write_text(json.dumps(self.step3))
        checks = {c["name"]: c for c in SV.score_dir(d, SV.step3_records(self.path), log=lambda *a: None)["checks"]
                  if c["name"].startswith("step3_continuity")}
        self.assertEqual(set(checks), {"step3_continuity:clean_o0.5", "step3_continuity:clean_o1"})
        self.assertTrue(checks["step3_continuity:clean_o0.5"]["ok"])
        missing = checks["step3_continuity:clean_o1"]
        self.assertEqual((missing["kind"], missing["ok"]), ("derived", False))
        self.assertIn("no step-3 record", missing["value"][0])


class ContinuityRequirementTests(unittest.TestCase):
    """Block R replicates 0-4 under step 3's settings are never scored without step 3's results file."""

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="req_"))
        self.step3_settings = asdict(PH.Settings(seed=20261001))

    def tearDown(self):
        shutil.rmtree(self.tmp)

    def test_refused_without_the_file(self):
        d, _ = write_process(self.tmp, block="R", level="good", rep=2, settings=self.step3_settings)
        self.assertTrue(SV.needs_step3(SV.read_meta(d)))
        with self.assertRaises(SV.Refused):
            SV.run([d], step3_path=self.tmp / "missing.json", log=lambda *a: None)
        with self.assertRaises(SV.Refused):
            SV.run([d], step3_path=None, log=lambda *a: None)
        with self.assertRaises(SV.Refused):
            SV.score_dir(d, step3=None, log=lambda *a: None)
        self.assertFalse((d / "values.npz").exists())

    def test_other_directories_need_no_file(self):
        other = dict(self.step3_settings, fit_steps=40)
        dirs = [write_process(self.tmp, block="R", level="good", rep=7, settings=self.step3_settings)[0],
                write_process(self.tmp, block="R", level="poor", rep=2, settings=other)[0],
                write_process(self.tmp, block="X-CS", level="poor", rep=2)[0]]
        self.assertFalse(any(SV.needs_step3(SV.read_meta(d)) for d in dirs))
        SV.run(dirs, step3_path=self.tmp / "missing.json", log=lambda *a: None)
        self.assertTrue(all((d / "values.npz").exists() for d in dirs))


def pilot_process(root, block="C", rep=99, seed=3):
    """A pilot-replicate directory (meta.pilot) with a localized and a clean cell; row 10 is HOOK, rows 7-9 arms whose J
    the pilot must not see (SIG, SHUF0, P3-OR)."""
    rng = np.random.default_rng(seed)
    contexts = block == "C"
    base = np.tile(K0, (4, 1, 1)) if contexts else K0
    shape = base.shape
    index = lambda **kw: dict(dict(ladder={}, matched={}, refs={}, anchors={}, decomp={}, checks={}), **kw)
    arrays, cells = {}, {}
    local = ("q1opt-local_k1", "q1opt-local", 1.0) if contexts else ("cone_k2", "cone", 2.0)
    for cid, case, kappa in (local, ("clean", "clean", None)):
        fp = base + 0.05 * rng.standard_normal((11,) + shape)
        a500 = base + 0.05 * rng.standard_normal((3,) + shape)
        labels = ([["NONE", None, "base"], ["EXACT", None, "base"]] + [["P0", m, "ladder"] for m in (0.5, 1.0, 2.0)]
                  + [["P1L:OR2", 1.5, "eval"], ["P2pi:OR1", 0.7, "eval"], ["P1L:SIG", 0.3, "eval"],
                     ["P1L:SHUF0", 0.4, "eval"], ["P3:OR", 0.1, "eval"], ["HOOK", None, "ref"]])
        matched = {"P0": {"S1": dict(row=None), "S2": dict(row=4)}, "P1L:OR2": {"S2": dict(row=5), "S3": dict(row=None)},
                   "P2pi:OR1": {"S1": dict(row=6)}, "P1L:SIG": {"S2": dict(row=7)}, "P1L:SHUF0": {"S2": dict(row=8)},
                   "P3:OR": {"S2": dict(row=9)}}
        regimes = dict(FP=dict(labels=labels, index=index(NONE=0, EXACT=1, ladder={"P0": [2, 3, 4]}, matched=matched,
                                                          refs={"HOOK": 10})),
                       A500=dict(labels=[["NONE", None, "base"], ["NONE_path", None, "base"], ["P1L:OR2", 2.0, "eval"]],
                                 index=index(NONE=0, NONE_path=1, matched={"P1L:OR2": {"S2": dict(row=2)}})))
        cells[cid] = dict(case=case, kappa=kappa, offset=None, checks=[], signal={}, diagnostics={}, regimes=regimes)
        arrays.update({f"K/{cid}/K0": base, f"K/{cid}/FP": fp, f"K/{cid}/A500": a500})
    return write_process(root, block=block, level="expert", rep=rep, cells=cells, arrays=arrays,
                         pilot=rep == SV.PILOT_REPLICATE)


class PilotTests(unittest.TestCase):
    """Design section 9: the pilot's J only in --pilot-power mode, and only K_0, NONE, EXACT, the P0 ladder and the
    matched OR1 / OR2 rows of block C's localized (and X-CS's cone) cells."""

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="pilot_"))

    def tearDown(self):
        shutil.rmtree(self.tmp)

    def test_refused_without_pilot_mode(self):
        d, _ = pilot_process(self.tmp)
        with self.assertRaises(SV.Refused):
            SV.score_dir(d, log=lambda *a: None)
        with self.assertRaises(SV.Refused):
            SV.run([d], step3_path=None, log=lambda *a: None)
        with self.assertRaises(SystemExit):
            SV.main([str(d)])
        self.assertEqual(sorted(p.name for p in d.iterdir()), ["HASHES.sha256", "arrays.npz", "matched_knobs.json"])

    def test_pilot_power_evaluates_only_the_whitelist(self):
        for block, cid in (("C", "q1opt-local_k1"), ("X-CS", "cone_k2")):
            d, arrays = pilot_process(self.tmp, block=block)
            contexts = block == "C"
            with mock.patch.object(LQ, "value", wraps=LQ.value) as value:
                result = SV.run([d], step3_path=None, pilot_power=True, log=lambda *a: None)[str(d)]
            per_gain = 4 if contexts else 1
            # J_opt, K_0, FP rows 0-6 and 8 (SHUF0, a pilot null since the freeze), A500 rows 0, 2
            self.assertEqual(value.call_count, 1 + per_gain * (1 + 8 + 2))
            seen = [np.asarray(c.args[1]) for c in value.call_args_list[1:]]
            allowed = [arrays[f"K/{cid}/K0"]] + [arrays[f"K/{cid}/FP"][i] for i in (0, 1, 2, 3, 4, 5, 6, 8)] + \
                      [arrays[f"K/{cid}/A500"][i] for i in (0, 2)]
            allowed = [k for K in allowed for k in (K if contexts else [K])]
            self.assertTrue(all(any(np.array_equal(s, a) for a in allowed) for s in seen))
            for forbidden in [arrays[f"K/{cid}/FP"][i] for i in (7, 9, 10)] + list(arrays["K/clean/FP"]):
                f = forbidden[0] if contexts else forbidden
                self.assertFalse(any(np.array_equal(s, f) for s in seen))
            out = np.load(d / "pilot_power.npz")
            self.assertEqual(sorted(out.files), sorted([f"{x}/{cid}/{r}" for x in ("J", "R") for r in ("FP", "A500")]
                                                       + [f"J/{cid}/K0"]))
            rows = np.array([0, 1, 2, 3, 4, 5, 6, 8])
            np.testing.assert_array_equal(out[f"R/{cid}/FP"], rows)
            np.testing.assert_array_equal(out[f"J/{cid}/FP"], SV.J_of(SYSTEM, arrays[f"K/{cid}/FP"][rows], contexts))
            fp = result["cells"][cid]["regimes"]["FP"]
            self.assertEqual(set(fp), {"NONE", "EXACT", "P0:0.5", "P0:1", "P0:2", "P1L:OR2@S2", "P2pi:OR1@S1",
                                       "P1L:SHUF0@S2"})
            self.assertEqual(fp["NONE"]["G"], 0.0)
            self.assertEqual(set(result["cells"]), {cid})
            self.assertFalse((d / "values.npz").exists())
            self.assertEqual(set(SV.read_hashes(d / SV.PILOT_HASH_FILE)), set(SV.PILOT_OUTPUTS))
            shutil.rmtree(self.tmp / block)

    def test_modes_under_one_root(self):
        pilot, _ = pilot_process(self.tmp)
        main, _ = write_process(self.tmp, block="X-CS", level="expert", rep=0)
        logs = []
        SV.run([self.tmp], step3_path=None, log=logs.append)
        self.assertTrue((main / "values.npz").exists())
        self.assertFalse(any((pilot / n).exists() for n in ("values.npz",) + SV.PILOT_OUTPUTS))
        self.assertTrue(any("skipped" in m and "rep99" in m for m in logs))
        SV.run([self.tmp], step3_path=None, pilot_power=True, log=lambda *a: None)
        self.assertTrue((pilot / "pilot_power.npz").exists())
        self.assertFalse((pilot / "values.npz").exists())
        with self.assertRaises(SV.Refused):
            SV.run([main], step3_path=None, pilot_power=True, log=lambda *a: None)
        with self.assertRaises(SV.Refused):
            SV.score_dir(main, overwrite=True, pilot_power=True, log=lambda *a: None)


if __name__ == "__main__":
    unittest.main()

"""Checks of scorecard_step4.py on synthetic cells with fake J values (no LQ system, no step-4 run): pooling and paired
SE, Holm, materiality as a parameter, every DR1 verdict, the gates G0-G2/G2b, DR2-DR7, the file-level loading with
its hash checks (unscored pilot and smoke directories under the root are skipped), the tally and flags of rows lost to
J = -inf / NaN, and 'not testable' (missing) against 'not confirmed' (false) in DR3, DR6 and DR6b.

Each scenario fixes per-replicate contrasts (a constant plus seeded noise) and builds every arm's G from them, so the
expected verdict is known by construction.
python -m unittest experiments.signal.test_scorecard_step4
"""
import json
import math
import shutil
import tempfile
import unittest
from pathlib import Path

import numpy as np
from scipy import stats

from experiments.signal import run_step4 as RUN
from experiments.signal import score_values as SV
from experiments.signal import scorecard_step4 as SC

GAP, J_NONE, NOISE = 0.5, -3.0, 0.002
QUALITIES = SC.QUALITIES
ZERO = dict(T_SHUF=0.0, T_STRAT=0.0, T_NUIS=0.0, T_ANTI=0.0, R=0.0, Str=0.0, I=0.0, C=0.0)


def build_cell(block, level, rep, cid, case, G, sets, refs=None, diag=None, kappa=None):
    """A scorecard Cell whose J rows give G(arm) = G[reg][arm][level] exactly (None: unreachable)."""
    regimes, J = {}, {}
    for reg, arms in G.items():
        rows, matched = [J_NONE], {}
        for arm, levels in arms.items():
            for lv, v in levels.items():
                if v is None:
                    matched.setdefault(arm, {})[lv] = dict(row=None)
                else:
                    rows.append(J_NONE + v * GAP)
                    matched.setdefault(arm, {})[lv] = dict(row=len(rows) - 1)
        ref_rows = {}
        for name, v in (refs or {}).get(reg, {}).items():
            rows.append(J_NONE + v * GAP)
            ref_rows[name] = len(rows) - 1
        regimes[reg] = dict(index=dict(NONE=0, matched=matched, refs=ref_rows))
        J[reg] = np.array(rows)
    record = dict(case=case, kappa=kappa, regimes=regimes, diagnostics=diag or {}, checks=[])
    return SC.Cell(block, level, rep, cid, record, J, GAP, sets)


def arm_values(effects, rng, levels=SC.LEVELS, placements=SC.PLACEMENTS, missing=()):
    """Every arm's G at every level from per-replicate contrasts (effects[placement][contrast] + noise).

    P0 is one arm shared by every placement of a cell, so its G (Str) is the sum of the placements' Str entries (the
    scenarios set it on one placement at most)."""
    out = {}
    p0 = sum(effects.get(p, {}).get("Str", 0.0) for p in placements) + NOISE * rng.standard_normal()
    for lv in levels:
        out.setdefault("P0", {})[lv] = p0
    for p in placements:
        e = dict(ZERO, **effects.get(p, {}))
        x = {k: v + NOISE * rng.standard_normal() for k, v in e.items()}
        for lv in levels:
            gone = (p, lv) in missing
            put = lambda arm, v: out.setdefault(arm, {}).__setitem__(lv, None if gone else v)
            sig = p0 + x["R"]
            if p == "P3":
                put("P3:SIG", sig)
                put("P3:NUIS", sig - x["T_NUIS"])
                put("P3:OR", p0 + x["C"])
                continue
            shuf = sig - x["T_SHUF"]
            put(f"{p}:SIG", sig)
            for k in range(4):
                put(f"{p}:SHUF{k}", shuf)
                put(f"{p}:STRAT{k}", sig - x["T_STRAT"])
            put(f"{p}:ANTI", sig - x["T_ANTI"])
            put(f"{p}:NUIS", sig - x["T_NUIS"])
            put(f"{p}:OR1", shuf + x["I"])
            put(f"{p}:OR2", shuf + x["C"])
    return out


def scenario(fp=None, a500=None, a5000=None, n_cells=2, reps=range(10), seed=0, missing=(), extra=None, hook_gap=0.0,
             diag=None):
    """X-CS M cells (n_cells per level), N cells (clean, noisy_reward), block C localized cells and cone cells."""
    rng = np.random.default_rng(seed)
    sets = {"X-CS": {}, "C": {}}
    cells = []
    effects = dict(FP=fp or {}, A500=a500 if a500 is not None else (fp or {}), A5000=a5000 or {})
    groups = [("X-CS", f"tilt_a{i}", "tilt", "M", QUALITIES, None) for i in range(n_cells)]
    groups += [("X-CS", "clean", "clean", "N", QUALITIES, None), ("X-CS", "noisy_reward", "noisy_reward", "N",
                                                                    QUALITIES, None)]
    groups += [("C", "q1opt-local_k1", "q1opt-local", "P", ("expert", "poor"), 1.0),
               ("C", "tilt-local_a-0.5", "tilt-local", "P", ("expert", "poor"), None)]
    groups += [("X-CS", "cone_k2", "cone", "P", QUALITIES, 2.0)]
    groups += extra or []
    for block, cid, case, label, levels, kappa in groups:
        for q in levels:
            sets.setdefault(block, {}).setdefault(cid, {})[q] = dict(set=label, M_nat=False)
            for r in reps:
                eff = {reg: (e(block, cid, q) if callable(e) else e) for reg, e in effects.items()}
                G = {reg: arm_values(eff[reg], rng, missing=missing if reg == "FP" else ()) for reg in eff}
                refs = {reg: {"HOOK": G[reg]["P0"]["S2"] + hook_gap} for reg in G}
                for reg in G:
                    G[reg]["P0"]["S(HOOK)"] = G[reg]["P0"]["S2"]
                cells.append(build_cell(block, q, r, cid, case, G, sets, refs, diag, kappa))
    return cells


def card(cells, checks=(), **kw):
    return SC.Scorecard(cells, list(checks), **kw)


class PrimitiveTests(unittest.TestCase):
    def test_holm(self):
        adj = SC.holm(dict(a=0.01, b=0.04, c=0.03, d=0.5, e=None))
        self.assertAlmostEqual(adj["a"], 0.05)
        self.assertAlmostEqual(adj["c"], 0.12)
        self.assertAlmostEqual(adj["b"], 0.12)  # monotone: max(0.12, 2 x 0.04)
        self.assertAlmostEqual(adj["d"], 1.0)
        self.assertEqual(adj["e"], 1.0)

    def test_summary_is_the_paired_t_test(self):
        x = np.random.default_rng(1).normal(0.01, 0.01, 10)
        s = SC.summary(list(x) + [math.nan, None])
        t = stats.ttest_1samp(x, 0.0)
        self.assertEqual(s["n"], 10)
        self.assertAlmostEqual(s["se"], x.std(ddof=1) / math.sqrt(10), places=15)
        self.assertAlmostEqual(s["t"], t.statistic, places=10)
        self.assertAlmostEqual(s["p"], t.pvalue, places=10)
        self.assertAlmostEqual(s["p_pos"], t.pvalue / 2 if t.statistic > 0 else 1 - t.pvalue / 2, places=10)
        self.assertIsNone(SC.summary([0.3])["se"])

    def test_pooling_averages_cells_within_a_replicate(self):
        cells = [build_cell("X-CS", "expert", r, cid, "tilt", {"FP": {"P0": {"S2": 0.0}, "P1L:SIG": {"S2": v}}},
                            {"X-CS": {cid: {"expert": dict(set="M")} for cid in ("a", "b")}})
                 for r, cid, v in ((0, "a", 0.02), (0, "b", 0.04), (1, "a", 0.01))]
        pool = SC.Pool(cells)
        s = pool.pooled(cells, "FP", "P1L", "R", "S2")
        np.testing.assert_allclose(s["values"], [0.03, 0.01], rtol=0, atol=1e-15)
        self.assertEqual((s["replicates"], s["cells"]), ([0, 1], 2))
        self.assertTrue(math.isnan(SC.contrasts(cells[0], "FP", "P1L", "S2")["T_SHUF"]))  # no SHUF arm: missing


class VerdictTests(unittest.TestCase):
    TARGETING = dict(T_SHUF=0.05, T_STRAT=0.05, T_NUIS=0.05, T_ANTI=0.1, R=0.05)

    def verdicts(self, cells, **kw):
        return card(cells, **kw).verdicts()

    def assert_all(self, verdicts, placement, expected):
        for q in QUALITIES:
            self.assertEqual(verdicts[placement][q]["verdict"], expected, (placement, q, verdicts[placement][q]))

    def test_v1_targeting_real_and_materiality(self):
        cells = scenario(fp={"P1L": self.TARGETING, "P3": dict(T_NUIS=0.05, R=0.05)})
        v = self.verdicts(cells)
        self.assert_all(v, "P1L", "V1")
        self.assert_all(v, "P3", "V1")
        self.assert_all(v, "P2L", "V8")
        self.assert_all(self.verdicts(cells, materiality=0.1), "P1L", "V8")  # 0.05 is immaterial at 10% of the gap

    def test_v2_v4_v5_v6_v7(self):
        cases = [(dict(self.TARGETING, R=0.0), "V2"), (dict(T_SHUF=0.05), "V4"), (dict(T_SHUF=0.05, T_STRAT=0.05), "V5"),
                 (dict(T_SHUF=-0.05, T_STRAT=-0.05), "V6"), (dict(Str=0.05), "V7")]
        for effects, expected in cases:
            self.assert_all(self.verdicts(scenario(fp={"P2pi": effects}, seed=3)), "P2pi", expected)

    def test_v3_and_short_horizon_reversal(self):
        opposite = dict(self.TARGETING, R=-0.05, T_SHUF=-0.05)
        v = self.verdicts(scenario(fp={"P1L": self.TARGETING}, a500={"P1L": opposite}))
        self.assert_all(v, "P1L", "V3")
        v = self.verdicts(scenario(fp={"P1L": self.TARGETING}, a500={"P1L": opposite}, a5000={"P1L": self.TARGETING}))
        self.assert_all(v, "P1L", "V1 (short-horizon reversal)")
        v = self.verdicts(scenario(fp={"P1L": dict(T_SHUF=-0.05)}, a500={"P1L": self.TARGETING}))
        self.assert_all(v, "P1L", "V3")  # the reverse: A500 meets the T-conditions, FP materially opposite

    def test_fallback_level_and_v0(self):
        cells = scenario(fp={"P1L": self.TARGETING}, missing=[("P1L", "S2")])
        sc = card(cells)
        self.assertEqual(sc.primary[("P1L", "FP")], "S1")
        self.assertEqual(sc.primary[("P2L", "FP")], "S2")
        self.assert_all(sc.verdicts(), "P1L", "V1")
        cells = scenario(fp={"P1L": self.TARGETING}, missing=[("P1L", "S2"), ("P1L", "S1")])
        self.assert_all(self.verdicts(cells), "P1L", "V0")


class GateTests(unittest.TestCase):
    def test_g0(self):
        ok = dict(kind="derived", ok=True, value=None, tol=None, name="a")
        self.assertTrue(card([], [ok]).g0()["passed"])
        self.assertFalse(card([], [ok, dict(ok, ok=False)]).g0()["passed"])
        small = dict(kind="eps", ok=False, value=5e-6, tol=1e-6, name="b")
        g = card([], [small, dict(kind="procedural", ok=False, name="c")]).g0()
        self.assertTrue(g["passed"])
        self.assertEqual((len(g["eps_misses"]), g["procedural_misses"]), (1, 1))
        self.assertFalse(card([], [dict(small, value=2e-4)]).g0()["passed"])  # above 100x the tolerance

    def test_g1(self):
        def effects(block, cid, q):
            if cid == "clean" and q in ("expert", "mixed"):
                return {"P1L": dict(T_STRAT=0.05), "P3": dict(R=0.05)}
            return {}
        g = card(scenario(fp=effects)).g1()
        self.assertFalse(g["placements"]["P1L"]["passed"])
        self.assertEqual(g["placements"]["P1L"]["count"], 2)
        self.assertTrue(g["placements"]["P2L"]["passed"])
        self.assertTrue(g["p3_nuisance_confounded"])

    def test_g2_and_g2b(self):
        local = lambda c: (lambda block, cid, q: {"P2L": dict(C=c)} if block == "C" or cid.startswith("cone") else {})
        g = card(scenario(fp=local(0.05))).g2()
        self.assertTrue(g["passed"])
        self.assertEqual(g["data"], "poor")  # amendment P
        self.assertTrue(g["placements"]["P2L"]["passed"] and not g["placements"]["P1L"]["passed"])
        self.assertFalse(card(scenario(fp=local(0.004))).g2()["passed"])
        # the gate reads poor data only: an effect at expert alone passes G2_expert but not G2
        expert_only = lambda block, cid, q: {"P2L": dict(C=0.05)} if block == "C" and q == "expert" else {}
        sc = card(scenario(fp=expert_only))
        self.assertFalse(sc.g2()["passed"])
        self.assertTrue(sc.g2("expert")["passed"])
        g2b = card(scenario(fp=local(0.05))).g2b()
        self.assertEqual(g2b["flag_kappa"], "2")
        self.assertTrue(g2b["passed"])


class RuleTests(unittest.TestCase):
    def test_dr2_dr5(self):
        info = card(scenario(fp={"P1L": dict(C=0.05), "P2L": dict(C=0.05, I=0.05), "P1pi": dict(T_SHUF=0.05)})
                    ).information()
        self.assertEqual(info["P1L"]["expert"]["reading"], "information class wrong")
        self.assertEqual(info["P2L"]["expert"]["reading"], "signal too weak")
        self.assertEqual(info["P2pi"]["expert"]["reading"], "cannot use perfect information")
        self.assertEqual(info["P3"]["expert"]["reading"], "cannot use perfect information")
        order = card([]).step_order(info)
        self.assertEqual((order["information_class_wrong"], order["signal_too_weak"], order["total"]), (4, 4, 20))
        self.assertFalse(order["step5_first"])

    def test_dr3_dr4_carry_and_poor_data(self):
        nat = [("R", "independent_errors_o1", "independent_errors", "M", ("good",), None)]
        target = dict(T_SHUF=0.05, T_STRAT=0.05, T_NUIS=0.05, T_ANTI=0.1, R=0.05)
        poor_bad = lambda block, cid, q: ({"P1L": dict(target, R=-0.05) if q == "poor" else target, "P2pi": target}
                                          if cid.startswith(("tilt_a", "independent")) else {})  # M and M_nat cells
        cells = scenario(fp=poor_bad, extra=nat)
        for c in cells:
            if c.block == "R":
                c.M_nat = True
        sc = card(cells)
        verdicts, g1 = sc.verdicts(), sc.g1()
        carry = sc.carry(verdicts, g1)
        self.assertIs(carry["P1L"]["qualifies"], True)
        self.assertTrue(carry["P2pi"]["qualifies"] and carry["P2pi"]["conditional_on_step5"])
        self.assertIs(carry["P2L"]["qualifies"], False)
        self.assertEqual(carry["P2L"]["status"], "does not qualify")
        self.assertEqual(carry["P1L"]["M_nat"]["regime"], "FP")
        poor = sc.poor_data(verdicts, carry)
        self.assertEqual(poor["preferred"], ["P2pi"])
        self.assertFalse(poor["none_for_poor"])

    def test_dr6_hypothesis(self):
        harm = lambda block, cid, q: (({"P1L": dict(T_SHUF=0.05, Str=-0.05, R=0.01)} if q == "poor"
                                       else {"P1L": dict(T_SHUF=0.05)}) if cid.startswith("tilt_a") else {})
        dr6 = card(scenario(fp=harm, diag=dict(spearman_sig_e1_L=0.5, spearman_sig_e1_pi=0.1))).hypothesis()
        self.assertTrue(dr6["P1L"]["identification"] and dr6["P1L"]["harm"] and dr6["P1L"]["confirmed"])
        self.assertEqual(dr6["P1L"]["status"], "confirmed")
        self.assertIs(dr6["P1pi"]["identification"], False)  # Spearman at pi below 0.3
        self.assertEqual(dr6["P1pi"]["status"], "not confirmed")
        inter = lambda block, cid, q: {"P2L": dict(R=0.05)} if q == "poor" else {}
        self.assertTrue(card(scenario(fp=inter)).interaction("P1L", "P2L")["confirmed"])

    def test_dr7_hook(self):
        self.assertTrue(card(scenario(hook_gap=0.0)).hook()["A500"]["alpha_retune"])
        self.assertFalse(card(scenario(hook_gap=0.05)).hook()["A500"]["alpha_retune"])

    def test_run_and_markdown(self):
        result = card(scenario(fp={"P1L": VerdictTests.TARGETING})).run()
        text = SC.markdown(result)
        self.assertIn("| P1L | V1 | V1 | V1 | V1 |", text)
        json.dumps(SC.jsonable(result), allow_nan=False)
        self.assertTrue(any(t["placement"] == "P1L" and t["contrast"] == "T_SHUF" and t["material"]
                            for t in result["tables"]))


def find(cells, block="X-CS", cid="tilt_a0", level="expert", rep=0):
    return next(c for c in cells if (c.block, c.cid, c.level, c.rep) == (block, cid, level, rep))


def set_J(cell, reg, value, arm=None, level="S2", row=None):
    """Overwrite one row's J (an arm's matched row, or a given row) with value (-inf: unstable, NaN: unbounded)."""
    if row is None:
        row = cell.rec["regimes"][reg]["index"]["matched"][arm][level]["row"]
    cell.J[reg][row] = value


class LostRowTests(unittest.TestCase):
    """Rows lost to J = -inf (unstable) or NaN (unbounded) are tallied and flag the pooled contrasts that read them."""

    def test_unstable_permutation_flags_the_contrast(self):
        cells = scenario(fp={"P1L": VerdictTests.TARGETING})
        set_J(find(cells), "FP", -math.inf, "P1L:SHUF1")
        sc = card(cells)
        s = sc.contrast("P1L", "T_SHUF", "expert")
        self.assertEqual((s["n"], s["lost_unstable"], s["lost_unbounded"], s["flag"]), (10, 1, 0, True))
        self.assertFalse(sc.contrast("P1L", "T_STRAT", "expert")["flag"])
        self.assertFalse(sc.contrast("P1L", "T_SHUF", "mixed")["flag"])
        self.assertIn("FP:T_SHUF", sc.verdicts()["P1L"]["expert"]["flagged"])
        result = sc.run()
        lost = result["lost"]
        self.assertEqual((lost["unstable"], lost["unbounded"]), (1, 0))
        self.assertEqual([(g["arm"], g["set"], g["level"], g["strength"], g["n"], g["unstable"]) for g in lost["groups"]],
                         [("P1L:SHUF1", "M", "expert", "S2", 20, 1)])  # 2 M cells x 10 replicates
        self.assertEqual(lost["totals"]["X-CS/FP"]["unstable"], 1)
        self.assertGreaterEqual(lost["flagged_contrasts"], 1)
        text = SC.markdown(result)
        self.assertIn("V1 (!)", text)
        self.assertIn("Unbounded and unstable rows", text)
        json.dumps(SC.jsonable(result), allow_nan=False)

    def test_unstable_none_and_unbounded_arm(self):
        cells = scenario(fp={"P1L": VerdictTests.TARGETING})
        c = find(cells, cid="tilt_a1", level="mixed", rep=3)
        set_J(c, "FP", -math.inf, row=c.rec["regimes"]["FP"]["index"]["NONE"])
        set_J(find(cells, level="poor", rep=4), "FP", math.nan, "P1L:SIG")
        sc = card(cells)
        self.assertTrue(all(math.isnan(v) for v in SC.contrasts(c, "FP", "P1L", "S2").values()))
        self.assertEqual(sc.pool.per_cell_lost(c, "FP", "P1L", "S2")["R"], ["NONE:unstable"])
        s = sc.contrast("P1L", "R", "mixed")
        self.assertEqual((s["n"], s["lost_unstable"], s["flag"]), (10, 1, True))
        s = sc.contrast("P1L", "T_NUIS", "poor")
        self.assertEqual((s["lost_unstable"], s["lost_unbounded"], s["flag"]), (0, 1, False))
        groups = {(g["arm"], g["strength"]): g for g in sc.lost_rows()["groups"]}
        self.assertEqual(groups[("NONE", "base")]["unstable"], 1)
        self.assertEqual(groups[("P1L:SIG", "S2")]["unbounded"], 1)

    def test_unreached_with_an_unbounded_ladder_and_dr7(self):
        cells = scenario(hook_gap=0.0)
        c = find(cells, level="medium", rep=2)
        idx = c.rec["regimes"]["FP"]["index"]
        idx["matched"]["P2L:STRAT0"]["S2"]["row"] = None
        c.J["FP"] = np.append(c.J["FP"], math.nan)
        idx["ladder"] = {"P2L:STRAT0": [0, len(c.J["FP"]) - 1]}
        set_J(find(cells, cid="clean", level="poor", rep=1), "A500", -math.inf,
              row=find(cells, cid="clean", level="poor", rep=1).rec["regimes"]["A500"]["index"]["refs"]["HOOK"])
        sc = card(cells)
        groups = {(g["block"], g["regime"], g["arm"], g["strength"]): g for g in sc.lost_rows()["groups"]}
        self.assertEqual(groups[("X-CS", "FP", "P2L:STRAT0", "S2")]["unreached_unbounded_ladder"], 1)
        self.assertEqual(groups[("X-CS", "A500", "HOOK", "ref")]["unstable"], 1)
        hook = sc.hook()["A500"]
        self.assertEqual((hook["lost_unstable"], hook["flag"]), ({"X-CS": 1}, True))


class MissingTests(unittest.TestCase):
    """'not testable' when a reading depends on a missing input, never 'not confirmed'."""

    @staticmethod
    def xep(effects):
        return scenario(fp=lambda block, cid, q: {"P1L": effects} if block == "X-EP" else {},
                        extra=[("X-EP", "tilt_a0.5", "tilt", "M", ("mixed",), None)],
                        diag=dict(corr_sig_poor_mode=0.1))

    def test_dr6b_without_or1(self):
        pathway = dict(T_SHUF=-0.05, I=-0.05, T_NUIS=-0.05)
        cells = self.xep(pathway)
        self.assertEqual(card(cells).mixed_pathway()["status"], "confirmed")
        for effects, failed in ((pathway, []), (dict(pathway, T_SHUF=0.05), ["T_SHUF"])):
            cells = self.xep(effects)
            for c in cells:
                if c.block == "X-EP":  # plan 'reduced': X-EP runs no OR1
                    c.rec["regimes"]["FP"]["index"]["matched"].pop("P1L:OR1")
            dr6b = card(cells).mixed_pathway()
            self.assertEqual((dr6b["status"], dr6b["confirmed"], dr6b["missing"], dr6b["failed"]),
                             ("not testable", None, ["I"], failed))
            self.assertIn("DR6b mixed pathway: not testable (missing: I)", SC.markdown(card(cells).run()))

    def test_dr6_missing_spearman(self):
        harm = lambda block, cid, q: (({"P1L": dict(T_SHUF=0.05, Str=-0.05, R=0.01)} if q == "poor"
                                       else {"P1L": dict(T_SHUF=0.05)}) if cid.startswith("tilt_a") else {})
        cells = scenario(fp=harm, diag=dict(spearman_sig_e1_L=0.5, spearman_sig_e1_pi=0.1))
        for c in cells:
            if c.level == "mixed":
                c.rec = dict(c.rec, diagnostics={})
        dr6 = card(cells).hypothesis()
        self.assertEqual((dr6["P1L"]["status"], dr6["P1L"]["confirmed"]), ("not testable", None))
        self.assertIn("spearman mixed", dr6["P1L"]["missing"])
        self.assertEqual(dr6["P1pi"]["status"], "not confirmed")  # Spearman 0.1 elsewhere: false whatever mixed is

    def test_dr3_and_g1_without_inputs(self):
        m_only = lambda block, cid, q: {"P1L": VerdictTests.TARGETING} if cid.startswith("tilt_a") else {}
        cells = scenario(fp=m_only)  # G1 passes, and there is no M_nat cell
        sc = card(cells)
        carry = sc.carry(sc.verdicts(), sc.g1())
        self.assertEqual((carry["P1L"]["status"], carry["P1L"]["qualifies"]), ("not testable", None))
        self.assertTrue({"M_nat T_SHUF", "M_nat R"} <= set(carry["P1L"]["missing"]))
        self.assertEqual(carry["P2L"]["status"], "does not qualify")  # no V1: false whatever M_nat would show
        self.assertFalse(carry["any"])
        g1 = card([c for c in cells if c.set != "N"]).g1()
        self.assertFalse(g1["placements"]["P1L"]["testable"])
        self.assertIsNone(g1["p3_nuisance_confounded"])


class LoadTests(unittest.TestCase):
    """The file layer: fake process directories (runner + score_values layout) and stage0.json with hashes."""

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="card_"))
        cells = scenario(fp={"P1L": VerdictTests.TARGETING}, reps=(0, 1, 2, 99))
        sets = {}
        by_proc = {}
        for c in cells:
            by_proc.setdefault((c.block, c.level, c.rep), []).append(c)
            sets.setdefault(c.block, {}).setdefault(c.cid, {})[c.level] = dict(set=c.set, M_nat=c.M_nat)
        for (block, level, rep), group in by_proc.items():
            d = self.tmp / block / level / f"rep{rep}"
            d.mkdir(parents=True)
            records = {c.cid: dict(c.rec, checks=[dict(name="x", kind="derived", ok=True, value=None, tol=None)])
                       for c in group}
            meta = dict(block=block, level=level, replicate=rep)
            (d / "matched_knobs.json").write_text(json.dumps(dict(meta=meta, cells=records)))
            np.savez_compressed(d / "arrays.npz", **{f"K/{c.cid}/K0": np.zeros((2, 3)) for c in group})
            (d / SV.HASH_FILE).write_text("".join(f"{SV.sha256_file(d / n)}  {n}\n" for n in SV.REQUIRED))
            np.savez_compressed(d / "values.npz", **{f"J/{c.cid}/{reg}": J for c in group for reg, J in c.J.items()})
            values = dict(cells={c.cid: dict(gap=GAP) for c in group}, checks=[])
            (d / "values.json").write_text(json.dumps(values))
            (d / SV.VALUES_HASH_FILE).write_text(
                "".join(f"{SV.sha256_file(d / n)}  {n}\n" for n in ("values.json", "values.npz")))
        self.stage0 = self.tmp / "stage0.json"
        self.stage0.write_text(json.dumps(dict(checks=[dict(name="s", kind="derived", ok=True)], sets=sets)))
        Path(str(self.stage0) + ".sha256").write_text(f"{SV.sha256_file(self.stage0)}  stage0.json\n")

    def tearDown(self):
        shutil.rmtree(self.tmp)

    def test_load_excludes_the_pilot_and_scores(self):
        cells, checks, stage0, procs = SC.load(self.tmp, self.stage0, log=lambda *a: None)
        self.assertEqual(sorted({c.rep for c in cells}), [0, 1, 2])
        self.assertTrue(all(c.set is not None for c in cells))
        result = SC.Scorecard(cells, checks).run()
        self.assertTrue(result["G0"]["passed"])
        self.assertEqual(result["DR1"]["P1L"]["expert"]["verdict"], "V1")
        out = self.tmp / "card"
        SC.main([str(self.tmp), "--stage0", str(self.stage0), "--output", str(out), "--materiality", "0.02"])
        self.assertEqual(json.loads(Path(str(out) + ".json").read_text())["materiality"], 0.02)

    def test_unscored_pilot_and_smoke_directories_are_skipped(self):
        for d in self.tmp.glob("*/*/rep99"):
            for name in ("values.json", "values.npz", SV.VALUES_HASH_FILE):
                (d / name).unlink()
        smoke = self.tmp / "X-CS" / "expert" / "rep999"
        smoke.mkdir(parents=True)
        (smoke / "matched_knobs.json").write_text(json.dumps(dict(meta=dict(block="X-CS", level="expert",
                                                                            replicate=999), cells={})))
        cells, checks, stage0, procs = SC.load(self.tmp, self.stage0, log=lambda *a: None)
        self.assertEqual(sorted({c.rep for c in cells}), [0, 1, 2])
        self.assertEqual(sorted({p["replicate"] for p in procs}), [0, 1, 2])
        d = self.tmp / "X-CS" / "expert" / "rep2"
        (d / SV.VALUES_HASH_FILE).unlink()  # an analysis replicate that is not scored still stops it
        with self.assertRaises(SV.Refused):
            SC.load(self.tmp, self.stage0, log=lambda *a: None)

    def test_refuses_changed_files(self):
        d = self.tmp / "X-CS" / "expert" / "rep1"
        (d / "values.json").write_text((d / "values.json").read_text().replace("0.5", "0.25"))
        with self.assertRaises(SV.Refused):
            SC.load(self.tmp, self.stage0, log=lambda *a: None)
        self.stage0.write_text(self.stage0.read_text() + " ")
        with self.assertRaises(SV.Refused):
            SC.load(self.tmp, self.stage0, log=lambda *a: None)


if __name__ == "__main__":
    unittest.main()

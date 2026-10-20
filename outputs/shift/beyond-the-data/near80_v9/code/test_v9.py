"""Near-80 (version 9, DRAFT): the tests of what version 9 adds (the build's items, decide/refusal.md section 8).

  TestSeedReservation     the fresh pool 35-49, refused everywhere but its first N_FRESH; every other block refused
  TestPooledGroups        BCA's two groups: exact row sets, no training row but the survivors, no row a scored test or
                          a learner reads, the same two groups for every switch arm
  TestOutcomeFreeCount    counted from commands alone in the data phase, before any training phase can start
  TestRefusalRule         a refusal marks BCA's arms refused (never raised), pads their places, fails only them
  TestNewArms             the plain twins (the mean-difference switch with the ramp and with the range limit)
  TestClaimsBlock         C1 = version 8's claim 4 marks; C2's open values are TODO markers, listed by todo_paths
  TestFreezeRefusesOpen   freeze.py refuses while any TODO marker is left, and writes nothing

On synthetic data and records only: no seed's data is generated or read, nothing is trained, nothing is written into
the repository.

Run: /tmp/claude-0/bcavenv/bin/python3 code/test_v9.py   (from /tmp, PYTHONPATH=/home/user/bca)
"""
import sys

sys.dont_write_bytecode = True

import copy  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import re  # noqa: E402
import shutil  # noqa: E402
import tempfile  # noqa: E402
import types  # noqa: E402
import unittest  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import near80_constants as K  # noqa: E402
import near80lib as N  # noqa: E402
import phases as PH  # noqa: E402
import test_near80 as TN  # noqa: E402
from experiments.shift import dataset as DS  # noqa: E402
from experiments.shift import shift as SH  # noqa: E402

V8_CLAIM4_ARMS = ("none", "clone", "full_none", "w0_check", "obca_k1", "obca_k2", "obca_forced", "fixed_gated",
                  "conformal_k1", "meangap_k1", "perfect_band", "range_hinge", "even_pull", "push_less", "random_1",
                  "random_2", "standin")


def blocks(registry):
    """{seed: block} of every seed range named in a registry's keys ('a-b')."""
    out = {}
    for key in registry:
        a, b = (int(x) for x in key.split("-"))
        for s in range(a, b + 1):
            out[s] = key
    return out


# --------------------------------------------------------------------------------------------- seeds

class TestSeedReservation(unittest.TestCase):
    def test_pool_is_35_49_and_named_nowhere_else(self):
        self.assertEqual(K.FRESH_POOL, tuple(range(35, 50)))
        taken = set(K.EXPLORATORY) | set(K.SPENT) | set(K.RESERVED) | set(K.NAMED_ELSEWHERE)
        self.assertFalse(set(K.FRESH_POOL) & taken)
        reg = blocks(K.SEED_REGISTRY)
        self.assertEqual({s for s, key in reg.items() if key == "35-49"}, set(K.FRESH_POOL))
        for s in taken:
            self.assertIn(s, reg, s)
            self.assertNotEqual(reg[s], "35-49")
        for s in (10, 11, 12, 13, 14):
            self.assertIn(s, K.SPENT)

    def test_confirmatory_is_the_first_n_of_the_pool(self):
        self.assertEqual(K.N_FRESH_CHOICES, (5, 10, 15))
        if isinstance(K.N_FRESH, int):
            self.assertIn(K.N_FRESH, K.N_FRESH_CHOICES)
            self.assertEqual(K.CONFIRMATORY, K.FRESH_POOL[:K.N_FRESH])
        else:
            self.assertEqual(K.N_FRESH, K.TODO)
            self.assertEqual(K.CONFIRMATORY, K.FRESH_POOL)          # stands in while open; the freeze refuses

    def test_refusal_of_every_block(self):
        for s in K.EXPLORATORY + K.CONFIRMATORY:
            self.assertIsNone(K.refusal_of(s), s)
        for s in list(range(5, 35)) + list(range(100, 120)) + list(range(900, 910)) + [50, 99, 120, 1000]:
            why = K.refusal_of(s)
            self.assertIsNotNone(why, s)
            self.assertTrue(why.startswith("refused: "), why)
            with self.assertRaises(ValueError):
                K.group_of(s)
        self.assertIn("spent", K.refusal_of(12))
        self.assertIn("version 6/7", K.refusal_of(7))

    def test_the_official_plan_reads_only_confirmatory_seeds(self):
        keys = PH.table_keys()
        seeds = {k.seed for k in keys if k.seed is not None}
        self.assertEqual(seeds, set(K.CONFIRMATORY))
        smoke = {k.seed for k in PH.smoke_keys() if k.seed is not None}
        self.assertEqual(smoke, {0})
        for k in keys:
            self.assertIn(k.setting, ("main", "x"), k)

    def test_the_facts_script_never_reads_the_pool(self):
        for s in K.FRESH_POOL:
            with self.assertRaises(N.Refused):
                N.check_seeds([s])
            with self.assertRaises(N.Refused):
                N.check_seeds([s], allow_spent=True)


# --------------------------------------------------------------------------------------------- the pooled groups

class TestPooledGroups(TN.Base):
    def test_exact_row_sets_by_brute_force(self):
        h = self.brute_heldout()
        t = np.abs(self.log["actions"][h].astype(np.float64)).max(axis=1)
        self.assertTrue(np.array_equal(self.tail, np.concatenate([self.surv, h[t > self.t_lo]])))
        self.assertTrue(np.array_equal(self.rest, h[t <= self.t_lo]))
        self.assertEqual(len(self.tail) + len(self.rest), len(h) + len(self.surv))

    def test_no_training_row_but_the_survivors(self):
        train = set(self.log.split_rows("train").tolist())
        both = np.concatenate([self.tail, self.rest])
        self.assertEqual({int(r) for r in both if int(r) in train}, set(self.surv.tolist()))
        held = set(self.log.split_rows("calibration").tolist()) | set(self.log.split_rows("evaluation").tolist())
        self.assertTrue({int(r) for r in both if int(r) not in set(self.surv.tolist())} <= held)

    def test_no_group_row_is_in_a_training_file(self):
        """The critics', the clone's and the actors' training rows (the kept training rows) hold no held-out row, and
        the second critic's (kept minus the survivors) holds no group row at all."""
        both = np.concatenate([self.tail, self.rest])
        kept = self.removal.kept_train_rows(self.log)
        self.assertEqual(len(np.intersect1d(kept, np.setdiff1d(both, self.surv))), 0)
        second = np.setdiff1d(kept, self.surv)
        self.assertEqual(len(np.intersect1d(second, both)), 0)
        arrays = DS.d4rl_dict(self.log, kept)
        self.assertTrue(N.training_file_check(arrays, self.log, self.removal)["passed"])

    def test_no_scored_test_reads_a_group_row(self):
        """Every score is a rollout from the fixed test start sets: the evaluation and actor code never builds,
        imports or reads the pooled groups; only the data phase (the count) and the switch phase (the scores) do."""
        import inspect
        import evaluate_phase as EP
        readers = ("pooled_groups", "heldout_rows", "pooled_rows", "switch_scores", "outcome_free_count")
        for name in ("policy_phase.py", "pooled.py", "near80_runner.py", "phase_main.py"):
            src = (HERE / name).read_text(encoding="utf8")
            for r in readers:
                self.assertNotIn(r + "(", src, f"{name} calls {r}")
        for fn in (EP.actor_values, EP.situations_table, EP.compute_values, EP.evaluate):    # the scored numbers
            src = inspect.getsource(fn)
            for r in readers + ("split_rows", "bank_rows"):
                self.assertNotIn(r, src, f"evaluate_phase.{fn.__name__} reads {r}")
            if fn is EP.actor_values:
                self.assertIn("TC.bank_starts(s, n)", src)
        lp = (HERE / "learner_phases.py").read_text(encoding="utf8")
        self.assertIn("switch_scores(critic2, sd, pg, gr", lp)
        sdata = (HERE / "settingdata.py").read_text(encoding="utf8")
        self.assertIn("N.outcome_free_count(", sdata)

    def test_the_same_two_groups_for_every_switch_arm(self):
        """One certificate entry per seed-world, computed from one (tail, rest) pair: BCA's certificate, the
        split-conformal gap, the rank test and the mean difference all come from the same two arrays, and the arm
        table reads that entry for every switch arm."""
        import learner_phases as LP
        S1 = LP.step1lib()
        rng = np.random.default_rng(5)
        tail, rest = rng.normal(0.5, 0.2, 60), rng.normal(0.0, 0.2, 800)
        e = LP.certificate_entry(tail, rest, 35, "stall", dict(t_lo=0.7, t_up=0.95))
        g = e["groups"]
        self.assertEqual((g["n_full_throttle"], g["n_rest"]), (60, 800))
        self.assertAlmostEqual(g["mean_full_throttle"], float(tail.mean()))
        self.assertAlmostEqual(g["mean_rest"], float(rest.mean()))
        self.assertAlmostEqual(g["q90_full_throttle"], float(np.quantile(tail, 0.9, method="inverted_cdf")))
        want = S1.jsonable(S1.certificate(tail, rest, [*K.CERT_TAG_HEAD, 35, K.WORLD_TAG["stall"]], K.CERT))
        self.assertEqual(e["certificate"], want)
        self.assertEqual(e["rank_test"], S1.jsonable(S1.rank_test(tail, rest)))
        import policy_phase as PP
        S = PP.load_step2()[0]
        per = {f"{w}/seed35": e for w in K.WORLDS}
        row = PP.arm_row_from("main", 35, "stall", per, S)
        src = row["sources"]
        self.assertEqual(src["bca_on"], bool(e["certificate"]["on"]))
        self.assertAlmostEqual(src["conformal_gap"], g["q90_full_throttle"] - g["q90_rest"])
        self.assertAlmostEqual(src["mean_gap"], g["mean_full_throttle"] - g["mean_rest"])
        self.assertAlmostEqual(row["arms"]["meandiff_k1"]["h"], src["mean_gap"])
        self.assertEqual(row["arms"]["conformal_k1"]["switch"], src["conformal_gap"] > K.ARM_RULE["conformal_margin"])


# --------------------------------------------------------------------------------------------- the count

class TestOutcomeFreeCount(TN.Base):
    def test_every_training_phase_waits_for_the_data_phase(self):
        """The data phase writes the count; no phase that trains can start before it (needs, transitively)."""
        keys = PH.table_keys()
        by_id = {PH.phase_id(k): k for k in keys}

        def ancestors(k, seen=None):
            seen = set() if seen is None else seen
            for n in PH.needs(k, keys):
                pid = PH.phase_id(n)
                if pid not in seen:
                    seen.add(pid)
                    ancestors(by_id.get(pid, n), seen)
            return seen
        for k in keys:
            if k.kind in PH.TRAINING_ROW_KINDS or k.kind == "critic_full":
                worlds = K.WORLDS if k.kind == "policy" else (k.world,)
                for w in worlds:
                    self.assertIn(PH.phase_id(PH.make_key("data_main", "main", w, k.seed)), ancestors(k), k)

    def test_the_count_reads_commands_only(self):
        c0 = N.outcome_free_count(self.log, self.gap, self.kept, self.step1, K.GAP_RULE)
        arrays = dict(self.log._arrays)
        rng = np.random.default_rng(11)
        for k in ("rewards", "next_observations", "observations", "infos/state", "infos/next_state", "terminals"):
            arrays[k] = rng.permutation(np.asarray(arrays[k]))
        c1 = N.outcome_free_count(DS.Log(arrays, dict(self.log.meta)), self.gap, self.kept, self.step1, K.GAP_RULE)
        self.assertEqual(json.dumps(c0, sort_keys=True, default=str), json.dumps(c1, sort_keys=True, default=str))

    def test_the_switch_must_find_the_counted_sizes(self):
        src = (HERE / "learner_phases.py").read_text(encoding="utf8")
        self.assertIn('checks["counts_equal_the_outcome_free_count"]', src)
        self.assertIn("the held-out rows differ from the data record's", src)
        sd = (HERE / "settingdata.py").read_text(encoding="utf8")
        self.assertIn("group_count", sd)
        self.assertIn("heldout_rows_sha256", sd)

    def test_the_count_is_reported_never_a_stop(self):
        nt = N.no_training_checks([dict(rep=0, world="stall", gate_checks=dict(passed=True),
                                        count=dict(defined=True, n_tail=5, below_bar=True,
                                                   group_checks=dict(passed=True)),
                                        bank_only_reproduces_step1=True)])
        self.assertTrue(nt["passed"])                                  # a small count is not a stop


# --------------------------------------------------------------------------------------------- the refusal rule

def entry(on=True, height=2.5, q90=(0.8, 0.1), mean=(0.5, 0.1), p=0.01):
    return dict(t_lo=0.7, t_up=0.95, certificate=dict(on=on, height=height, q05=0.4),
                groups=dict(n_full_throttle=19, n_rest=900, mean_full_throttle=mean[0], mean_rest=mean[1],
                            q90_full_throttle=q90[0], q90_rest=q90[1]),
                rank_test=dict(U=1.0, p=p))


class TestRefusalRule(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import policy_phase as PP
        cls.PP = PP
        cls.S = PP.load_step2()[0]

    def per(self, stall, no_stall, seed=35):
        return {f"stall/seed{seed}": stall, f"no_stall/seed{seed}": no_stall}

    def test_step2_refuses_where_the_restated_table_marks(self):
        per = self.per(entry(height="inf"), entry(on=False, height="inf"))
        with self.assertRaises((SystemExit, ValueError)):
            self.S.arm_table({"per_seed_world": per}, seeds=[35], rule=K.ARM_RULE)
        t = self.PP.restated_table(self.S, per, 35)
        st, ns = t["stall"]["arms"], t["no_stall"]["arms"]
        self.assertEqual(sorted(n for n, a in st.items() if a["refused"]), ["obca_forced", "obca_k1", "obca_k2"])
        self.assertEqual(sorted(n for n, a in ns.items() if a["refused"]), ["obca_forced"])
        for name in ("fixed_gated", "conformal_k1", "meangap_k1"):
            self.assertFalse(st[name]["refused"], name)
        self.assertIn("not finite", st["obca_k1"]["refused_reason"])

    def test_arm_row_marks_bca_refused_and_keeps_the_twins(self):
        per = self.per(entry(height="inf"), entry(on=False))
        row = self.PP.arm_row_from("main", 35, "stall", per, self.S)
        self.assertTrue(row["bca_refused"])
        self.assertEqual(row["refused"], ["obca_forced", "obca_k1", "obca_k2"])
        self.assertIsNotNone(row["arm_table_refusal"])
        for twin in K.PLAIN_TWINS:
            self.assertFalse(row["arms"][twin]["refused"])
        row_ns = self.PP.arm_row_from("main", 35, "no_stall", per, self.S)
        self.assertFalse(row_ns["bca_refused"])
        self.assertEqual(row_ns["refused"], ["obca_forced"])

    def test_without_a_refusal_the_table_is_step2s(self):
        per = self.per(entry(), entry(on=False))
        for w in K.WORLDS:
            row = self.PP.arm_row_from("main", 35, w, per, self.S)
            self.assertIsNone(row["arm_table_refusal"])
            s2 = self.S.arm_table({"per_seed_world": per}, seeds=[35], rule=K.ARM_RULE)[w][35]
            self.assertTrue(self.PP.same_as_step2(row, s2))
            self.assertEqual(row["refused"], [])

    def test_a_refused_place_keeps_its_pad_in_step2s_program(self):
        per = self.per(entry(height="inf"), entry(on=False))
        row = self.PP.arm_row_from("main", 35, "stall", per, self.S)
        pad = self.PP.refused_pad
        self.assertEqual(self.PP.ramp_layout("main", "stall", row), [
            ("w0_check",), (pad("obca_k1"), pad("obca_k2"), pad("obca_forced"), "fixed_gated", "conformal_k1",
                            "meangap_k1"), ("meandiff_k1",)])
        row_ns = self.PP.arm_row_from("main", 35, "no_stall", per, self.S)
        self.assertEqual(self.PP.ramp_layout("main", "no_stall", row_ns), [
            ("w0_check", "obca_k1", "obca_k2", pad("obca_forced"), "fixed_gated", "conformal_k1", "meangap_k1"),
            ("meandiff_k1",)])

        class MC:
            @staticmethod
            def ramp_arm(name, h, k):
                return (name, h, k)
        self.assertEqual(self.PP.layout_arm(MC, row, pad("obca_k1")), (pad("obca_k1"), 0.0, 0.0))

    def test_refusal_fails_the_bca_arms_only(self):
        self.assertEqual(K.BCA_HEIGHT_ARMS, ("obca_k1", "obca_k2", "obca_forced"))
        self.assertEqual(set(K.BCA_MATCHED_ARMS), {"standin", "even_pull", "push_less", "random_1", "random_2"})
        others = set(K.ARMS["main"]) - set(K.BCA_HEIGHT_ARMS) - set(K.BCA_MATCHED_ARMS)
        self.assertTrue({"none", "clone", "full_none", "w0_check", "fixed_gated", "conformal_k1", "meangap_k1",
                         "meandiff_k1", "meandiff_limit", "perfect_band", "range_hinge"} <= others)

    def test_the_stub_has_a_refusing_seed_in_the_pool(self):
        import stubs as ST
        self.assertTrue(set(ST.STUB_REFUSING_SEEDS) <= set(K.FRESH_POOL[:min(K.N_FRESH_CHOICES)]))
        self.assertLess(ST.STUB_FEW_SURVIVORS, 29)


# --------------------------------------------------------------------------------------------- the new arms

class TestNewArms(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import policy_phase as PP
        cls.PP = PP
        cls.S = PP.load_step2()[0]

    def test_arm_list(self):
        main = K.ARMS["main"]
        for a in V8_CLAIM4_ARMS + K.PLAIN_TWINS:
            self.assertIn(a, main)
        self.assertEqual(len(set(main)), len(main))
        self.assertEqual(K.PLAIN_TWINS, ("meandiff_k1", "meandiff_limit"))
        for a in main:
            self.assertIn(a, K.ARM_PLAIN)
        self.assertEqual(K.SETTINGS, ("main",))

    def test_the_mean_difference_switch(self):
        for gap, on in ((0.11, True), (0.10, False), (0.05, False), (float("nan"), False)):
            per = {f"{w}/seed35": entry(mean=(0.1 + gap, 0.1)) for w in K.WORLDS}
            row = self.PP.arm_row_from("main", 35, "stall", per, self.S)
            md, ml = row["arms"]["meandiff_k1"], row["arms"]["meandiff_limit"]
            self.assertEqual(md["switch"], on, gap)
            self.assertEqual(ml["switch"], on, gap)
            if on:
                self.assertAlmostEqual(md["h"], gap)
            else:
                self.assertEqual(md["h"], 0.0)
            self.assertEqual((md["k"], ml["caution"], ml["coef"]), (1.0, "range_hinge", 1.0 if on else 0.0))
        self.assertEqual(K.MEANDIFF_MARGIN, 0.10)

    def test_programs(self):
        for w in K.WORLDS:
            calls = self.PP.ramp_layout("main", w)
            self.assertEqual(calls[-1], ("meandiff_k1",))
            flat = [n for c in calls for n in c]
            self.assertEqual(sorted(flat), sorted((K.W0,) + K.STEP2_RAMP_ARMS + ("meandiff_k1",)))
        self.assertEqual(self.PP.ramp_layout("main", "stall")[:2], [(K.W0,), K.STEP2_RAMP_ARMS])
        self.assertEqual(self.PP.ramp_layout("main", "no_stall")[0], (K.W0,) + K.STEP2_RAMP_ARMS)
        src = (HERE / "policy_phase.py").read_text(encoding="utf8")
        self.assertIn('H.RoomArm("meandiff_limit", "hinge", (1.0,))', src)


# --------------------------------------------------------------------------------------------- the claims block

class TestClaimsBlock(unittest.TestCase):
    def test_c1_is_version8s_claim4(self):
        v8 = json.loads((K.V8_DIR / "prereg.json").read_text(encoding="utf8"))["constants"]["MARKS"]
        c1 = K.CLAIMS["C1"]["marks"]
        for k in ("net_lead_margin", "keeps_floor", "even_pull_margin"):
            self.assertEqual(c1[k], v8[k], k)
        self.assertEqual(K.MARKS, c1)

    def test_c2_block(self):
        c2 = K.CLAIMS["C2"]
        self.assertEqual(c2["primary"], "obca_k1")
        self.assertEqual(c2["comparators"], ("conformal_k1", "meandiff_k1", "meandiff_limit"))
        self.assertEqual(set(c2["margins"]), set(c2["comparators"]))
        self.assertIn(c2["measure"], K.C2_MEASURES)
        self.assertEqual(K.C2_STATISTICS, ("mean", "paired_t", "interval"))
        self.assertEqual(K.C2_MULTIPLICITY, ("all", "holm", "bonferroni"))
        self.assertEqual(K.DECISION_RULES, ("C2", "C1_and_C2"))

    def test_todo_paths_list_every_open_value(self):
        want = {f"CLAIMS/C2/margins/{t}" for t in K.CLAIMS["C2"]["comparators"]}
        want |= {f"CLAIMS/C2/{k}" for k in ("statistic", "alpha", "min_positive_seeds", "multiplicity",
                                            "noise_floor_max")}
        want |= {"CLAIMS/decision", "OWNER_ANSWER/text", "OWNER_ANSWER/utc", "OWNER_ANSWER/where", "N_FRESH"}
        open_ = set(K.todo_paths())
        if open_:                                                      # a draft: exactly these are open
            self.assertEqual(open_, want)
        self.assertEqual(K.todo_paths(dict(a=None, b=[1, K.TODO], c=dict(d=K.TODO + " x")), "X"), ["X/b/1", "X/c/d"])

    def test_the_draft_json_holds_the_same_open_values(self):
        d = json.loads((K.N80 / "prereg_draft.json").read_text(encoding="utf8")) \
            if (K.N80 / "prereg_draft.json").is_file() else None
        if d is None:
            self.skipTest("the draft was moved by the freeze")
        self.assertEqual(d["claims"]["C2_margins"], K.CLAIMS["C2"]["margins"])
        self.assertEqual(d["claims"]["decision"], K.CLAIMS["decision"])
        self.assertEqual(d["seeds"]["pool"], list(K.FRESH_POOL))


# --------------------------------------------------------------------------------------------- the freeze

class TestFreezeRefusesOpen(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import freeze as FZ
        cls.FZ = FZ

    def test_todo_problems_name_every_open_value(self):
        if not K.todo_paths():
            self.skipTest("the claims block is filled")
        found = self.FZ.todo_problems(K.N80)
        for p in K.todo_paths():
            self.assertTrue(any(x.startswith(p + " is still") for x in found), p)
        self.assertTrue(any(x.startswith("PREREGISTRATION.md:") for x in found))
        self.assertTrue(any(x.startswith("prereg_draft.json:") for x in found))
        self.assertFalse(any(x.startswith("code/near80_constants.py:20 ") for x in found))   # the marker's own line
        self.assertTrue(set(found) <= set(self.FZ.problems(K.N80, Path(tempfile.mkdtemp()) / "out")))

    def test_write_refuses_and_writes_nothing(self):
        if not K.todo_paths():
            self.skipTest("the claims block is filled")
        before = sorted(str(p.relative_to(K.N80)) for p in K.N80.rglob("*"))
        out = Path(tempfile.mkdtemp(prefix="near80v9_freeze_")) / "out"
        try:
            with self.assertRaises(self.FZ.FreezeRefused) as cm:
                self.FZ.write("a test", n80_root=K.N80, out_root=out, tests_fn=lambda *a: {}, facts_fn=lambda *a: None,
                              log=lambda *a: None)
            self.assertIn("is still " + K.TODO, str(cm.exception))
            self.assertEqual(sorted(str(p.relative_to(K.N80)) for p in K.N80.rglob("*")), before)
            self.assertFalse(out.exists())
        finally:
            shutil.rmtree(out.parent, ignore_errors=True)

    def test_a_filled_folder_has_no_todo_problem(self):
        filled = types.SimpleNamespace(TODO=K.TODO, todo_paths=lambda: [])
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            (root / "code").mkdir()
            (root / "code" / "near80_constants.py").write_text(f'TODO = "{K.TODO}"\nN_FRESH = 10\n')
            (root / "PREREGISTRATION.md").write_text("every value filled\n")
            self.assertEqual(self.FZ.todo_problems(root, filled), [])
            (root / "README.md").write_text(f"one left: {K.TODO}\n")
            self.assertEqual(self.FZ.todo_problems(root, filled), [f"README.md:1 still holds {K.TODO}"])

    def test_check_lists_the_open_values(self):
        lines = []
        found = self.FZ.check(K.N80, Path(tempfile.mkdtemp()) / "out", log=lines.append)
        self.assertTrue(found)
        self.assertIn("Problems (write would refuse)", "\n".join(lines))


if __name__ == "__main__":
    unittest.main(verbosity=2)

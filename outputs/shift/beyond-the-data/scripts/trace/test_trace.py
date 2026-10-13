"""EXPLORATORY trace test bed: unit tests on small synthetic arrays only. They never read a ThrusterChain log, critic
or actor, and no BCA function is called anywhere (the test bed has none): the trace selection rule and its pairing
across worlds (on a synthetic two-world stand-in), the noise draws and the one-step error forms, the flag level, the
neighbour scores (residual evidence in both forms, distance, the neighbour offset), the range hinge, the behaviour
density, the set summary with re-referencing, the selectivity rule, the pooled questions, the one-CPU refusal, the
output guard, and launch.sh's handling of stopped and failed jobs (with a stub interpreter: nothing real runs).
Spec v3 (post hoc): the nesting of all six shares, the refusal of the spec-v2 shares (job check, output guard,
launch), the spec SHA-256 each share's outputs must name, the pooled look over a subset of shares, the check that the
spec-v2 entries are reproduced, the map over every share, and launch.sh's explicit CPUs and one-CPU 'all'.
Code manifest version 4: launch.sh allows only CPU 0 or CPU 1 (CPUs 2 and 3 run the stand-ins); with a recording
taskset on PATH, 'all 0 1' pins only to CPUs 0 and 1, and every launch naming CPU 2 or 3 is refused before anything
is pinned or run.

Usage (from /home/user/bca): python -m unittest discover -s <this directory> -p test_trace.py
"""
import math
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np
from scipy.spatial import cKDTree

sys.path.insert(0, str(Path(__file__).resolve().parent))
import tracebed as T  # noqa: E402
import signals as S  # noqa: E402
import run_job as R  # noqa: E402

HERE = Path(__file__).resolve().parent


class TraceSelection(unittest.TestCase):
    def test_count_rule(self):
        self.assertEqual(T.trace_count(0.005, 6603), 33)
        self.assertEqual(T.trace_count(0.02, 6603), 132)
        self.assertEqual(T.trace_count(0.005, 130), 1)
        self.assertEqual(T.trace_count(0.02, 130), 3)
        self.assertEqual(T.trace_count(0.0, 6603), 0)
        self.assertEqual(T.trace_count(0.005, 100), 1)      # 0.5 rounds up (floor(x + 0.5))

    def test_nested_subset_sorted_and_seeded(self):
        cand = np.sort(np.random.default_rng(0).choice(100_000, 6609, replace=False))
        small, k1 = T.select_trace(cand, 0.005, [1, 2, 3])
        big, k2 = T.select_trace(cand, 0.02, [1, 2, 3])
        again, _ = T.select_trace(cand, 0.02, [1, 2, 3])
        other, _ = T.select_trace(cand, 0.02, [1, 2, 4])
        self.assertEqual((k1, k2, len(small), len(big)), (33, 132, 33, 132))
        self.assertTrue(np.isin(small, big).all())
        self.assertTrue(np.isin(big, cand).all())
        self.assertTrue(np.all(np.diff(big) > 0))
        self.assertTrue(np.array_equal(big, again))
        self.assertFalse(np.array_equal(big, other))

    def test_all_six_shares_nest(self):
        cand = np.sort(np.random.default_rng(3).choice(100_000, 6603, replace=False))
        sels = [T.select_trace(cand, s, [20261015, 0, 91]) for s in T.SHARES]
        self.assertEqual(list(T.SHARES), [0.005, 0.01, 0.02, 0.03, 0.05, 0.1])
        self.assertEqual([k for _, k in sels], [33, 66, 132, 198, 330, 660])
        for (small, _), (big, _) in zip(sels[:-1], sels[1:]):
            self.assertTrue(np.isin(small, big).all())
        self.assertEqual(T.trace_count(0.1, 6747), 675)
        self.assertEqual(T.trace_count(0.01, 6459), 65)

    def test_share_dirs_and_labels(self):
        self.assertEqual([T.SHARE_DIR[s] for s in T.POST_HOC_SHARES],
                         ["share_1pct", "share_3pct", "share_5pct", "share_10pct"])
        self.assertEqual(T.SHARE_DIR[float("0.10")], "share_10pct")
        for s in T.POST_HOC_SHARES:
            self.assertEqual(T.share_label(s), "post hoc: chosen after seeing the 0.5% and 2% results")
        self.assertEqual(T.share_origin(0.02), "planned (spec v2)")
        self.assertEqual(T.share_origin(0.0), "reference (the gate's gapped log)")
        self.assertFalse(set(T.POST_HOC_SHARES) & set(T.FROZEN_SHARES))
        with self.assertRaises(ValueError):
            T.share_origin(0.04)

    def test_empty_candidates(self):
        sel, k = T.select_trace(np.zeros(0, np.int64), 0.02, [1])
        self.assertEqual((k, len(sel)), (0, 0))

    def test_tags_have_no_world_and_never_end_in_zero(self):
        self.assertEqual(T.selection_tags(1), [20261015, 1, 91])
        self.assertEqual(T.selection_tags(1, bank=True), [20261015, 1, 91, 1])
        for tags in (T.selection_tags(0), T.selection_tags(0, True)):
            self.assertNotEqual(tags[-1], 0)
        # why: numpy treats a trailing 0 as absent
        self.assertEqual(np.random.default_rng([5, 6, 0]).integers(1000), np.random.default_rng([5, 6]).integers(1000))


class FakeLog(dict):
    def __init__(self, actions, bank_rows):
        super().__init__(actions=actions)
        self._bank = np.asarray(bank_rows, np.int64)

    @property
    def bank_rows(self):
        return self._bank


class FakeRemoval:
    def __init__(self, removed, bank_removed):
        self.removed, self.bank_removed = removed, bank_removed

    def removed_train_rows(self, log):
        return np.flatnonzero(self.removed)


def fake_world(rep, past_rows, removed_rows, bank_rows, bank_past_pos, n=4000):
    """A synthetic stand-in for one world's loaded seed: rows in past_rows have a past-80% command."""
    acts = np.zeros((n, 4), np.float32)
    acts[past_rows, 0] = 0.95
    acts[bank_rows[bank_past_pos], 0] = -1.0
    removed = np.zeros(n, bool)
    removed[removed_rows] = True
    bank_removed = np.zeros(len(bank_rows), bool)
    bank_removed[bank_past_pos] = True
    return SimpleNamespace(rep=rep, log=FakeLog(acts, bank_rows), removals={"gap": FakeRemoval(removed, bank_removed)},
                           physics=T.TC.STALL, gap=None)


class PairedTrace(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(7)
        shared = np.sort(rng.choice(3000, 1000, replace=False))
        spare = np.setdiff1d(np.arange(3000), shared)
        own_s, own_n = spare[:5], spare[5:8]                  # world-specific candidates
        bank_rows = np.arange(3000, 3400, 2)
        self.shared = shared
        self.sds = {"stall": fake_world(0, np.r_[shared, own_s], np.r_[shared, own_s, spare[100:150]], bank_rows,
                                        np.arange(0, 60)),
                    "no_stall": fake_world(0, np.r_[shared, own_n], np.r_[shared, own_n, spare[100:150]], bank_rows,
                                           np.arange(0, 61))}

    def test_same_rows_in_both_worlds_from_shared_candidates(self):
        for share in T.SHARES:
            ts, tn = (T.build_trace(self.sds, w, share) for w in T.WORLDS)
            self.assertTrue(np.array_equal(ts.added_rows, tn.added_rows))
            self.assertTrue(np.array_equal(ts.added_bank, tn.added_bank))
            self.assertTrue(np.array_equal(ts.shared, self.shared))
            self.assertEqual(len(ts.added_rows), T.trace_count(share, len(self.shared)))
            self.assertEqual((len(ts.world_specific), len(tn.world_specific)), (5, 3))
            self.assertFalse(np.isin(ts.added_rows, ts.world_specific).any())
            self.assertEqual(len(ts.shared_bank), 60)            # the no-stall world's 61st bank candidate is its own
            self.assertTrue(np.isin(ts.added_rows, self.shared).all())
            self.assertFalse(ts.removal.removed[ts.added_rows].any())
            self.assertEqual(int(ts.removal.removed.sum()), int(self.sds["stall"].removals["gap"].removed.sum())
                             - len(ts.added_rows))
            self.assertEqual(int((~ts.removal.bank_removed).sum()) - int((~self.sds["stall"].removals["gap"]
                                                                          .bank_removed).sum()), len(ts.added_bank))
        small, big = T.build_trace(self.sds, "stall", 0.005), T.build_trace(self.sds, "stall", 0.02)
        self.assertTrue(np.isin(small.added_rows, big.added_rows).all())
        traces = [T.build_trace(self.sds, "no_stall", s) for s in T.SHARES]
        for a, b in zip(traces[:-1], traces[1:]):
            self.assertTrue(np.isin(a.added_rows, b.added_rows).all())
            self.assertTrue(np.isin(a.added_bank, b.added_bank).all())
        self.assertEqual(len(traces[-1].added_rows), 100)


class OneStepError(unittest.TestCase):
    def test_noise_draws_depend_on_seed_draw_and_row_only(self):
        a = S.target_noise_draw(0, 0, [5, 7], 10)
        b = S.target_noise_draw(0, 0, [5, 7], 100)
        self.assertTrue(np.array_equal(a, b))
        self.assertFalse(np.array_equal(a, S.target_noise_draw(0, 1, [5, 7], 10)))
        self.assertFalse(np.array_equal(a, S.target_noise_draw(1, 0, [5, 7], 10)))
        self.assertEqual(a.shape, (2, 4))

    def test_noisy_next_commands_follow_the_training_rule(self):
        c = np.array([[0.8, -0.8, 0.0, 0.5]], np.float32)
        z = np.array([[10.0, -1.0, 0.5, -10.0]])
        out = S.noisy_next_commands(c, z, 0.2, 0.5)
        self.assertEqual(out.dtype, np.float32)
        self.assertTrue(np.allclose(out, [[1.0, -1.0, 0.1, 0.0]], atol=1e-6))   # eps clipped at 0.5, command at 1

    def test_one_step_forms_and_offset(self):
        rng = np.random.default_rng(5)
        r, q, qp, qm = (rng.normal(size=50) for _ in range(4))
        done = (np.arange(50) % 7 == 0).astype(float)
        o = S.one_step_from_q(r, done, 0.98, q, qp, qm)
        self.assertTrue(np.allclose(o["trained"], r + 0.98 * (1 - done) * qm - q))
        self.assertTrue(np.allclose(o["noiseless"] - o["trained"], o["offset"]))
        self.assertTrue(np.allclose(o["J"], qp - qm))
        self.assertTrue(np.all(o["offset"][done == 1] == 0))

    def test_a_concave_critic_has_a_positive_offset(self):
        """A synthetic critic that drops past 80% gives J > 0 at a controller sitting at the cap, about 0 where the
        critic is flat: the reason the noiseless form is biased in the region."""
        def q(a):
            return -3.0 * np.maximum(np.abs(a[:, 0]) - 0.8, 0.0)
        c = np.array([[0.8, 0, 0, 0], [0.3, 0, 0, 0]], np.float32)
        acc = np.zeros(2)
        for d in range(64):
            acc += q(S.noisy_next_commands(c, S.target_noise_draw(0, d, [0, 1], 2), 0.2, 0.5).astype(np.float64))
        J = q(c.astype(np.float64)) - acc / 64
        self.assertGreater(J[0], 0.1)
        self.assertAlmostEqual(J[1], 0.0, places=6)


class Levels(unittest.TestCase):
    def test_conformal_level(self):
        s = np.arange(1, 101, dtype=float)          # n = 100: ceil(101 * 0.95) = 96th smallest
        self.assertEqual(S.conformal_level(s), 96.0)
        self.assertEqual(S.conformal_level(np.arange(1, 20, dtype=float)), 19.0)   # ceil(20 * 0.95) = 19
        self.assertTrue(math.isinf(S.conformal_level(np.arange(1, 10, dtype=float))))   # ceil(9.5) = 10 > 9
        rng = np.random.default_rng(1)
        x = rng.normal(size=5000)
        self.assertAlmostEqual(float((x > S.conformal_level(x)).mean()), 0.05, delta=0.002)


class NeighbourScores(unittest.TestCase):
    def test_residual_evidence_and_distance(self):
        rng = np.random.default_rng(2)
        Z = rng.normal(size=(500, 3))
        delta = np.full(500, 0.1)
        far = Z[:, 0] > 1.5                       # a cluster the critic overrates: negative one-step errors
        delta[far] = -0.5
        past = far.copy()
        offset = np.where(far, 0.3, 0.1)
        q = np.array([[2.5, 0.0, 0.0], [-2.0, 0.0, 0.0]])
        out = S.neighbour_scores(Z, delta, past, q, delta_noiseless=delta + offset, offset=offset)
        self.assertGreater(out["residual_evidence"][0], 0.0)
        self.assertAlmostEqual(out["residual_evidence"][1], -0.1)
        self.assertTrue(np.allclose(out["residual_evidence_noiseless"],
                                    out["residual_evidence"] - out["neighbour_offset"]))
        self.assertAlmostEqual(out["neighbour_offset"][1], 0.1)
        self.assertTrue(out["neighbour_past80"][0])
        self.assertFalse(out["neighbour_past80"][1])
        d = np.sqrt(((Z - q[1]) ** 2).sum(1))
        self.assertAlmostEqual(out["distance"][1], np.sort(d)[9])
        out2 = S.neighbour_scores(None, delta, past, q, tree=cKDTree(Z))
        self.assertTrue(np.allclose(out2["residual_evidence"], out["residual_evidence"]))
        self.assertNotIn("neighbour_offset", out2)

    def test_hinge_and_density(self):
        lo, hi = np.array([-0.5, -0.5]), np.array([0.5, 0.5])
        a = np.array([[0.0, 0.0], [0.7, 0.0], [0.7, -0.8]])
        self.assertTrue(np.allclose(S.range_hinge(a, lo, hi), [0.0, 0.04, 0.04 + 0.09]))
        c = np.zeros((3, 2))
        self.assertTrue(np.allclose(S.density(a, c, np.array([1.0, 2.0])), [0.0, 0.245, 0.245 + 0.08]))

    def test_cap(self):
        a = np.array([[0.95, -1.0, 0.3, 0.8]], np.float32)
        cap = S.cap80(a)
        self.assertEqual(cap.dtype, np.float32)
        self.assertFalse(T.past80(cap, T.TC.STALL).any())
        self.assertTrue(T.past80(a, T.TC.STALL).all())


def fake_scores(n, n_past, flag, value=2.0):
    sa = {k: np.where(np.arange(n) < int(round(flag * n_past)), value, 0.0) for k in S.SIGNALS}
    sa["neighbour_past80"] = np.zeros(n, bool)
    sa["neighbour_offset"] = np.full(n, 0.1)
    return sa


def fake_set(n_past, flag, rereferenced=None):
    """A set summary as signals.set_summary writes it, with the given flag rate at past-80% pairs for every signal."""
    n = 100
    past = np.zeros(n, bool)
    past[:n_past] = True
    sa, sc = fake_scores(n, n_past, flag), fake_scores(n, n_past, 0.0)
    rec = None
    if rereferenced is not None:
        rec = {k: v.copy() for k, v in sa.items()}
        for k in S.SIGNALS:
            rec[k] = sa[k] - rereferenced
    return S.set_summary(sa, sc, past, {k: 1.0 for k in S.SIGNALS}, rec)


class Summary(unittest.TestCase):
    def test_set_summary_and_selectivity(self):
        st, ns = fake_set(40, 0.75), fake_set(50, 0.2)
        self.assertAlmostEqual(st["residual_evidence"]["flag_past80"], 0.75)
        self.assertAlmostEqual(st["residual_evidence"]["paired_widening"], 0.75)
        self.assertAlmostEqual(st["residual_evidence"]["flag_cap_of_past80"], 0.0)
        self.assertAlmostEqual(st["neighbour_offset_mean_past80"], 0.1)
        self.assertNotIn("rereferenced", st["residual_evidence"])
        self.assertAlmostEqual(S.selectivity(st, ns, "residual_evidence"), 0.55)
        few = fake_set(T.MIN_PAST80_PAIRS - 1, 1.0)
        self.assertIsNone(S.selectivity(few, ns, "residual_evidence"))
        self.assertIsNone(S.selectivity(st, few, "residual_evidence"))

    def test_rereferenced(self):
        st, ns = fake_set(40, 0.5, rereferenced=0.3), fake_set(40, 0.5, rereferenced=0.1)
        self.assertAlmostEqual(st["residual_evidence"]["rereferenced"]["mean_past80"], 0.3)
        self.assertAlmostEqual(st["residual_evidence"]["rereferenced"]["share_above_recorded_past80"], 1.0)
        self.assertAlmostEqual(S.rereferenced_difference(st, ns, "residual_evidence"), 0.2)
        self.assertIsNone(S.rereferenced_difference(fake_set(40, 0.5), ns, "residual_evidence"))


def fake_jobs(harm, sel_re, sel_hinge, stop=None):
    """{(share, seed, world): job} with the given strong-current harm and T1 flag rates (stall world flags at
    0.5 + selectivity, no-stall at 0.5)."""
    jobs = {}
    od = dict(J=dict(region_rows=dict(mean=0.2), outside_region_rows=dict(mean=0.1), past80_rows=None))
    for share in [T.REFERENCE_SHARE] + list(T.SHARES):
        for rep in T.SEEDS:
            for world in T.WORLDS:
                if stop == (share, rep, world):
                    jobs[(share, rep, world)] = dict(stopped=True, reason="test")
                    continue
                sets = {}
                for sname in S.SETS:
                    st = fake_set(60, 0.5, rereferenced=0.2 if world == "stall" else 0.0)
                    if world == "stall" and sname == S.PRIMARY_SET:
                        st["residual_evidence"]["flag_past80"] = 0.5 + sel_re
                        st["range_hinge"]["flag_past80"] = 0.5 + sel_hinge
                    sets[sname] = st
                sit = {s: dict(harm=harm if world == "stall" else -1.0, points={}) for s in T.TC.SITUATIONS}
                readout = dict(sets=sets, stand_ins=dict(offset_diagnostic=od), T4=dict(neighbour_offset_mean=0.1))
                jobs[(share, rep, world)] = dict(stopped=False, values=dict(situations=sit, pull_past_80={}),
                                                 signals=dict(readout=readout))
    return jobs


class Pooled(unittest.TestCase):
    def test_both_yes(self):
        out = R.pool_jobs(fake_jobs(harm=60.0, sel_re=0.3, sel_hinge=0.0))
        for share in T.SHARES:
            o = out[T.SHARE_DIR[share]]
            self.assertTrue(o["Q1_harm_kept"] and o["Q2_residual_evidence_separates"] and o["usable_for_a_harm_claim"])
            self.assertAlmostEqual(o["selectivity_mean"]["T1"]["residual_evidence"], 0.3)
            self.assertAlmostEqual(o["rereferenced_difference_mean"]["T1"]["residual_evidence"], 0.2)
            self.assertAlmostEqual(o["rereferenced_difference_mean"]["T1c"]["residual_evidence"], 0.2)
            self.assertIn("T1c", o["selectivity_mean"])
            row = o["per_seed"][0]["offset_diagnostic"]
            self.assertEqual(row["stall"]["J_region_rows"], 0.2)
            self.assertIsNone(row["stall"]["J_past80_rows"])
        self.assertIsNone(out[T.SHARE_DIR[T.REFERENCE_SHARE]]["usable_for_a_harm_claim"])

    def test_noiseless_form_is_reported_not_judged(self):
        jobs = fake_jobs(harm=60.0, sel_re=0.0, sel_hinge=0.0)
        for (share, rep, world), job in jobs.items():
            if world == "stall":
                job["signals"]["readout"]["sets"]["T1"]["residual_evidence_noiseless"]["flag_past80"] = 0.9
        o = R.pool_jobs(jobs)[T.SHARE_DIR[0.02]]
        self.assertAlmostEqual(o["selectivity_mean"]["T1"]["residual_evidence_noiseless"], 0.4)
        self.assertFalse(o["Q2_residual_evidence_separates"])

    def test_hinge_also_separates_or_harm_gone(self):
        out = R.pool_jobs(fake_jobs(harm=60.0, sel_re=0.3, sel_hinge=0.25))
        self.assertFalse(out[T.SHARE_DIR[0.02]]["Q2_residual_evidence_separates"])
        out = R.pool_jobs(fake_jobs(harm=40.0, sel_re=0.3, sel_hinge=0.0))
        self.assertFalse(out[T.SHARE_DIR[0.02]]["Q1_harm_kept"])
        self.assertFalse(out[T.SHARE_DIR[0.02]]["usable_for_a_harm_claim"])

    def test_a_stopped_seed_gives_no_answer(self):
        out = R.pool_jobs(fake_jobs(harm=60.0, sel_re=0.3, sel_hinge=0.0, stop=(0.02, 1, "no_stall")))
        o = out[T.SHARE_DIR[0.02]]
        self.assertIsNone(o["selectivity_mean"]["T1"]["residual_evidence"])
        self.assertIsNone(o["rereferenced_difference_mean"]["T1"]["residual_evidence"])
        self.assertIsNone(o["per_seed"][1]["offset_diagnostic"]["no_stall"])
        self.assertFalse(o["Q2_residual_evidence_separates"])
        self.assertTrue(o["Q1_harm_kept"])          # the stalling world's harm is still complete
        self.assertTrue(out[T.SHARE_DIR[0.005]]["usable_for_a_harm_claim"])


V3_SPEC = {"worlds": list(T.WORLDS), "seeds": list(T.SEEDS), "shares": list(T.V2_SHARES), "reference_share": 0.0,
           "version": 3,
           "post_hoc_extension": {"shares": list(T.POST_HOC_SHARES),
                                  "frozen_v2": {"spec_sha256": "2" * 64, "shares": list(T.FROZEN_SHARES)}}}


class SpecV3(unittest.TestCase):
    def test_spec_shares_and_which_spec_each_share_names(self):
        self.assertEqual(sorted(T.spec_shares(V3_SPEC)), list(T.SHARES))
        for s in T.FROZEN_SHARES:
            self.assertEqual(T.share_spec_sha256(V3_SPEC, "3" * 64, s), "2" * 64)
        for s in T.POST_HOC_SHARES:
            self.assertEqual(T.share_spec_sha256(V3_SPEC, "3" * 64, s), "3" * 64)
        v2 = {k: v for k, v in V3_SPEC.items() if k != "post_hoc_extension"}
        self.assertEqual(T.share_spec_sha256(v2, "2" * 64, 0.02), "2" * 64)

    def test_check_job_refuses_spec_v2_shares_under_v3(self):
        for s in T.FROZEN_SHARES:
            with self.assertRaises(SystemExit) as e:
                R.check_job(V3_SPEC, "stall", 0, s)
            self.assertIn("spec-v2 job", str(e.exception))
        for s in T.POST_HOC_SHARES:
            R.check_job(V3_SPEC, "no_stall", 2, s)
        with self.assertRaises(SystemExit):
            R.check_job(V3_SPEC, "stall", 0, 0.04)
        with self.assertRaises(SystemExit):
            R.check_job(V3_SPEC, "stall", 3, 0.1)

    def test_main_refuses_a_spec_v2_share_before_loading(self):
        with mock.patch.object(T, "begin", return_value=(V3_SPEC, "3" * 64, "m" * 64)), \
                mock.patch.object(R, "load", side_effect=AssertionError("must refuse before loading")):
            with self.assertRaises(SystemExit) as e:
                R.main(["--world", "stall", "--seed", "0", "--share", "0.02", "--phase", "build"])
            self.assertIn("spec-v2 job", str(e.exception))
            with self.assertRaises(SystemExit) as e:
                R.main(["--world", "stall", "--seed", "0", "--share", "0", "--phase", "reference"])
            self.assertIn("spec-v2 job", str(e.exception))

    def test_pool_a_subset_and_the_v2_entries_check(self):
        jobs = fake_jobs(harm=60.0, sel_re=0.3, sel_hinge=0.0)
        full = R.pool_jobs(jobs)
        part = R.pool_jobs(jobs, list(T.FROZEN_SHARES))
        self.assertEqual(sorted(part), sorted(T.SHARE_DIR[s] for s in T.FROZEN_SHARES))
        for k in part:   # each share is pooled on its own
            self.assertEqual(T.A.canonical(T.jsonable(part[k])), T.A.canonical(T.jsonable(full[k])))
        v2 = {"shares": T.jsonable(part)}
        equal, same = R.v2_entries_check(full, v2)
        self.assertTrue(same)
        v2["shares"]["share_2pct"]["harm_strong_current_mean"] = 1.0
        equal, same = R.v2_entries_check(full, v2)
        self.assertFalse(same)
        self.assertFalse(equal["share_2pct"])
        self.assertTrue(equal["share_0.5pct"])

    def test_map(self):
        jobs = fake_jobs(harm=60.0, sel_re=0.3, sel_hinge=0.0, stop=(0.05, 1, "no_stall"))
        for rep in T.SEEDS:      # the 10% log: the harm is gone
            jobs[(0.1, rep, "stall")]["values"]["situations"]["strong_current"]["harm"] = 10.0
        m = R.share_map(R.pool_jobs(jobs))
        self.assertEqual([r["share"] for r in m["rows"]], [0.0] + list(T.SHARES))
        by = {r["share_dir"]: r for r in m["rows"]}
        self.assertEqual(by["share_0pct"]["usable"], "not judged (reference)")
        self.assertEqual(by["share_0.5pct"]["origin"], "planned (spec v2)")
        self.assertEqual(by["share_3pct"]["origin"], "post hoc (spec v3)")
        self.assertEqual(by["share_1pct"]["usable"], "yes")
        self.assertEqual(by["share_10pct"]["Q1_harm_at_least_50"], "no")
        self.assertEqual(by["share_10pct"]["usable"], "no")
        self.assertEqual(by["share_5pct"]["Q2_trace_detected"], "no answer (a seed is missing)")
        self.assertEqual(by["share_5pct"]["usable"], "no answer (a seed is missing)")
        self.assertAlmostEqual(by["share_1pct"]["selectivity_T1_mean"]["residual_evidence"], 0.3)
        self.assertEqual(sorted(by["share_1pct"]["selectivity_T1_mean"]), sorted(S.SIGNALS))
        self.assertEqual(by["share_10pct"]["harm_strong_current_per_seed"], [10.0, 10.0, 10.0])
        self.assertEqual(m["usable_shares"], [0.5, 1, 2, 3])
        self.assertEqual(m["shares_with_harm_kept"], [0.5, 1, 2, 3, 5])
        self.assertIn("post hoc", m["note"])
        self.assertTrue(any("10%" in line for line in R.map_lines(m)))


class Pinning(unittest.TestCase):
    def test_refuses_unless_one_cpu(self):
        with mock.patch.object(T.os, "sched_getaffinity", return_value={0, 1}):
            with self.assertRaises(SystemExit):
                T.require_one_cpu()
        with mock.patch.object(T.os, "sched_getaffinity", return_value={3}):
            self.assertEqual(T.require_one_cpu(), [3])

    def test_job_phases_refuse_unpinned(self):
        with mock.patch.object(T.os, "sched_getaffinity", return_value={0, 1, 2, 3}), \
                mock.patch.object(T, "begin", return_value=({"worlds": list(T.WORLDS), "seeds": list(T.SEEDS),
                                                            "shares": list(T.SHARES), "reference_share": 0.0},
                                                           "s" * 64, "m" * 64)), \
                mock.patch.object(R, "load", side_effect=AssertionError("must refuse before loading")):
            with self.assertRaises(SystemExit) as e:
                R.main(["--world", "stall", "--seed", "0", "--share", "0.005", "--phase", "build"])
            self.assertIn("one CPU", str(e.exception))


class Guard(unittest.TestCase):
    def test_guard_refuses_protected_and_outside(self):
        for p in (T.ROOT / "runs" / "shift" / "x.json", T.ROOT / "experiments" / "shift" / "x.json",
                  T.ROOT / "runs" / "shift_explore" / "room" / "x.json", Path("/tmp/x.json")):
            with self.assertRaises(SystemExit):
                T.guard_out(p)
        self.assertEqual(T.guard_out(T.OUT / "stall" / "seed0" / "x.json"),
                         (T.OUT / "stall" / "seed0" / "x.json").resolve())

    def test_guard_refuses_spec_v2_outputs(self):
        for p in (T.OUT / "pooled.json", T.OUT / "pooled.json.sha256", T.OUT / "timing.json",
                  T.OUT / "checks_seed0.json", T.OUT / "stall" / "seed1" / "share_2pct" / "values.json",
                  T.OUT / "no_stall" / "seed2" / "share_0.5pct" / "lock.json",
                  T.OUT / "stall" / "seed0" / "share_0pct" / "signals.npz"):
            with self.assertRaises(SystemExit) as e:
                T.guard_out(p)
            self.assertIn("spec-v2 output", str(e.exception))
        for s in T.POST_HOC_SHARES:
            T.guard_out(T.job_dir("no_stall", 2, s) / "values.json")
        T.guard_out(T.OUT / "pooled_v3.json")


STUB = """#!/bin/sh
# stub interpreter for the launch test: logs its arguments; fails when FAIL_ON is set and matches
echo "$*" >> "{calls}"
if [ -n "$FAIL_ON" ]; then case "$*" in *"$FAIL_ON"*) exit 1 ;; esac; fi
exit 0
"""


TASKSET = """#!/bin/sh
# recording taskset for the launch test: logs the CPU it is asked for, then runs the command without pinning it (it
# keeps the test process's own CPU)
[ "$1" = -c ] || exit 9
echo "$2" >> "{pins}"
shift 2
exec "$@"
"""

LAUNCH_CPUS = ("0", "1")   # launch.sh's allowed CPUs (CPUs 2 and 3 run the stand-ins)


class Launch(unittest.TestCase):
    def setUp(self):
        allowed = [str(c) for c in sorted(os.sched_getaffinity(0)) if str(c) in LAUNCH_CPUS]
        if not allowed:
            self.skipTest("this process may not use CPU 0 or CPU 1, the only CPUs launch.sh allows")
        self.cpu = "1" if "1" in allowed else allowed[0]
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)
        self.calls = self.tmp / "calls.txt"
        stub = self.tmp / "py"
        stub.write_text(STUB.format(calls=self.calls))
        stub.chmod(0o755)
        self.env = dict(os.environ, PY=str(stub), OUTDIR=str(self.tmp), NICE="0")
        self.env.pop("FAIL_ON", None)

    def tearDown(self):
        self._tmp.cleanup()

    def references(self, seeds):
        for seed in seeds:
            for w in T.WORLDS:
                d = self.tmp / w / f"seed{seed}" / "share_0pct"
                d.mkdir(parents=True)
                (d / "signals.json").write_text("{}")

    def launch(self, *args, **env):
        return subprocess.run(["sh", str(HERE / "launch.sh")] + list(args), env=dict(self.env, **env),
                              capture_output=True, text=True, timeout=120)

    def lines(self):
        """The stub's calls, without the script path (the first argument)."""
        if not self.calls.exists():
            return []
        out = []
        for line in self.calls.read_text().splitlines():
            script, _, rest = line.partition(" ")
            self.assertTrue(script.endswith("/run_job.py"), line)
            out.append(rest)
        return out

    def test_post_hoc_shares_stopped_job_skipped_and_a_failure_does_not_end_the_launch(self):
        self.references([9])
        stopped = self.tmp / "stall" / "seed9" / "share_1pct"
        stopped.mkdir(parents=True)
        (stopped / "stop.json").write_text("{}")
        p = self.launch("9", self.cpu, FAIL_ON="--world no_stall --seed 9 --share 0.05")
        self.assertEqual(p.returncode, 1, p.stdout + p.stderr)
        lines = self.lines()
        self.assertEqual(len(lines), 7)       # 4 shares x 2 worlds, less the stopped one; no reference is rerun
        self.assertFalse(any("--share 0 " in x or "--share 0.005" in x or "--share 0.02" in x for x in lines))
        self.assertFalse(any("--world stall" in x and "--share 0.01 " in x for x in lines))
        self.assertTrue(any("--world no_stall" in x and "--share 0.01 " in x for x in lines))
        self.assertTrue(lines[-1].endswith("--world no_stall --seed 9 --share 0.1 --phase build train eval signals"))
        log = (self.tmp / "logs" / "v3_seed9.log").read_text()
        self.assertIn("skip: stall seed 9 share_1pct has stopped", log)
        self.assertIn("reference is a spec-v2 job, already written", log)
        self.assertIn("FAILED", log)
        self.assertFalse((self.tmp / "logs" / "seed9.log").exists())

    def test_explicit_shares_and_cpus(self):
        self.references([9])
        p = self.launch("9", self.cpu, "stall", "0.10")
        self.assertEqual(p.returncode, 0, p.stdout + p.stderr)
        self.assertEqual(self.lines(), ["--world stall --seed 9 --share 0.1 --phase build train eval signals"])
        self.assertTrue((self.tmp / "logs" / "v3_seed9_stall_0.10.log").exists())
        for args in (["9", self.cpu, "stall", "0.02"], ["9", self.cpu, "all", "0.005"], ["9", self.cpu, "stall", "0"],
                     ["9", self.cpu, "stall", "0.04"], ["9", self.cpu, "sideways"], ["9", "0-3"], ["9", "1,2"],
                     ["9", "999"], ["all", self.cpu, self.cpu], ["all", self.cpu, "x"], ["x9", self.cpu]):
            p = self.launch(*args)
            self.assertEqual(p.returncode, 2, (args, p.stdout + p.stderr))
        self.assertEqual(len(self.lines()), 1)      # nothing ran after a refusal

    def test_a_missing_reference_is_reported(self):
        p = self.launch("9", self.cpu, "stall", "0.01")
        self.assertEqual(p.returncode, 1)
        self.assertIn("MISSING: stall seed 9 reference", (self.tmp / "logs" / "v3_seed9_stall_0.01.log").read_text())
        self.assertEqual(len(self.lines()), 1)      # the post-hoc job still ran

    def test_all_on_one_cpu(self):
        self.references(T.SEEDS)
        p = self.launch("all", self.cpu)
        self.assertEqual(p.returncode, 0, p.stdout + p.stderr)
        lines = self.lines()
        self.assertEqual(len(lines), len(T.SEEDS) * len(T.WORLDS) * len(T.POST_HOC_SHARES) + 1)
        self.assertEqual(lines[-1], "--phase pooled")
        self.assertTrue(lines[0].startswith("--world stall --seed 0 --share 0.01 "))
        self.assertIn("all: seeds 0, 1 and 2 in turn", (self.tmp / "logs" / "v3_all.log").read_text())

    def recording_taskset(self):
        """A taskset on PATH that records the CPU each call asks for (and pins nothing)."""
        bindir = self.tmp / "bin"
        bindir.mkdir(exist_ok=True)
        self.pins = self.tmp / "pins.txt"
        ts = bindir / "taskset"
        ts.write_text(TASKSET.format(pins=self.pins))
        ts.chmod(0o755)
        return dict(PATH=f"{bindir}:{os.environ.get('PATH', '')}")

    def pinned(self):
        return self.pins.read_text().split() if self.pins.exists() else []

    def test_all_on_cpus_0_and_1_pins_only_there(self):
        self.references(T.SEEDS)
        env = self.recording_taskset()
        p = self.launch("all", "0", "1", **env)
        self.assertEqual(p.returncode, 0, p.stdout + p.stderr)
        lines = self.lines()
        self.assertEqual(len(lines), len(T.SEEDS) * len(T.WORLDS) * len(T.POST_HOC_SHARES) + 1)
        self.assertEqual(lines[-1], "--phase pooled")
        pins = self.pinned()
        self.assertTrue(pins)
        self.assertEqual(set(pins), {"0", "1"})

    def test_cpus_2_and_3_are_refused_before_anything_runs(self):
        self.references(T.SEEDS)
        env = self.recording_taskset()
        for args in (["0", "2"], ["0", "3"], ["2", "3", "stall", "0.01"], ["1", "3", "no_stall", "0.1"],
                     ["pooled", "2"], ["all", "2"], ["all", "3"], ["all", "0", "3"], ["all", "1", "2"],
                     ["all", "2", "1"], ["all", "3", "0"], ["0", "4"]):
            p = self.launch(*args, **env)
            self.assertEqual(p.returncode, 2, (args, p.stdout + p.stderr))
            self.assertIn("is not allowed", p.stdout, args)
        self.assertEqual(self.lines(), [])                                    # no job ran
        self.assertFalse(set(self.pinned()) & {"2", "3", "4"}, self.pinned())  # nothing was pinned to them
        self.assertFalse((self.tmp / "logs").exists())                        # refused before the logs folder


if __name__ == "__main__":
    unittest.main()

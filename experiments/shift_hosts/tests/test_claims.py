"""Tests of claims.py and export.py on SYNTHETIC numbers only (plan sections 5.5, 6.1, 6.2 and 9.1): no ThrusterChain
outcome and no learned policy; the stage driver's dry run read here trained on made-up arrays only. An
experiment-local attachment (plan section 3.1) that changes none of the repository's declared methods.

What is checked:
  statistics   T95 and Student's t with 4 degrees of freedom (closed form) against scipy.stats.t; the mean, sd, lower
               and upper bounds and one-sided p-value against numpy and scipy (ttest_1samp); L > 0 exactly when p < 0.05;
               the power figure of plan section 5.5 (a gain of 6.93 with sd 4.463 gives p about 0.0128, t about 3.47,
               and Holm's first step needs p <= 0.0125)
  Holm         the weighting test's own cases (0.010, 0.020, 0.030 all pass; 0.010, 0.030, 0.040 only the first); all
               pass; ties; a learner with p = 1 stays in and never loosens the bar; exact boundaries; the adjusted
               p-value agrees with the step-down; "pending" until all 5 seeds of all four learners are complete, and
               Claim 2 until every Claim 4a is decided
  Claim 0      the bars (-341.23 and -501.90 from the anchors -800.29 and -35.19; the real start banks' recording
               controllers give 60 and 39 points and the same bars to 0.1), below/at the bar, a non-finite seed, a stop
               of the learner alone or of its room reference; a learner that did not learn reads "not judged" and stays
               in both families with p = 1
  badges       every claim's badges (passed, shelved, flagged, pending, not judged, did not learn), the size bar (1.79
               shelved, 1.81 passed), L <= 0, Claim 4a/4b's coverage bars, Claim 2's p = 1 after a failed 4a, the width
               label, the TD3+BC selection label (medium-expert only), the flags 1c, 2c, 3 and 3b, Claim 5's 2
               percentage points compared exactly on counts, Claim 6 with and without stage 3's twins
  views        the reading order, the 12-row table, the shelving write-up format, the study rule, stage 3's order
  export       an end-to-end build on made-up folders in the runner's, the stage driver's and band_readout's layout
               (runner.execute's record.json, lanes/scores-step<T>.npz, lanes/coverage-step<R>/coverage.{json,npz},
               lanes/s7-step<T>/, driver/status.json, state.json, driver.log and stop reports): every number equals
               its scores file and carries its record's and scores file's SHA-256, every field is read or "not
               recorded" with a reason; records made by other code, tampered or mismatched files are excluded; stops as
               outcomes (S5 on Unifloral's own step, an S5 the operator accepted in the driver's log) or halts; S4 from
               the records; the driver's checks, choice 19 and stop reports; read_run's rules for real records (a
               stand-in identity, no approval); Claim 2's pointer from readouts.dose_change; the reference row from a
               made-up copy of the weighting test's records, cross-checked against its RESULTS.json computed with
               scipy; refusals (a made-up build without a made-up identity, the real copy folder under MADE_UP, writes
               outside their folders, main() today)
  originals    the rules export.py restates against runner.py, stages.py and band_readout.py themselves: schemas,
               canonical_sha256, run_id and the run folder, the plan's and the dry run's schedules (make_schedule),
               env_signature, the driver's card_job and lanes_for (the scores, coverage and S7 file names) and its
               log line for an accepted S5
  dry run      export.build over the stage driver's own dry-run output (made-up arrays, processor, tiny step counts),
               copied, with made-up scores files placed where the scoring lanes write them: the fresh dry run
               runs/shift_hosts/stage0/claims/dryrun (medium-expert, seeds 0-4, every learner; required): all 82
               stage-2 runs read, every field read or explained, the 12-row table filled with 5 seeds, Claims 0, 4a,
               4b, 1 and 2 judged, the driver's checks read; and when present test_stages' dry run (both qualities)
               and its accepted-S5 run (stop reports moved aside by --resume-after-stop)
  supplement   version 4's S7 supplement (supplement.py; PRESPEC.md section 21, post hoc): every package file but
               supplement.CHANGEABLE is version 3's approved bytes (claims.py, export.py, runner.py, evaluate.py,
               band_readout.py, gate.py, approve_hosts.py and every training and scoring file, against the SHA-256
               values of version 3's code manifest), and a one-byte change is found; prespec.json's frozen block, the
               plan's run counts (learner scope 21 or 20, IW scope 15; medium-expert 56, medium 42) and order (the
               driver's), and validate's refusals; check_sources (made-up files, and version 3's own files read only
               when present); the export on made-up folders carries a made-up "version 3" results.json with every key
               canonical-JSON equal and adds only qualities.<quality>.s7_shown, s7_supplement and one caveat (tampering,
               a wrong carried file or SHA-256 refused), the shown block holds no t, p-value, bar, mark, badge, Holm
               entry or verdict, integrity S7 reads "outcome" with seed-1 failures covered by version-4 entries, S1
               fails on a version-3 record in the supplement's root; the same over test_stages' supplement dry run
               (version 4's driver files) when present; R1 (reproduction) compares, ignores what differs by design and
               never raises; claims.judge and the 12-row table ignore a stopped scope's runs (version 3's S7 scenarios)
  integration  (scoring environment only) a made-up actor (made-up parameters, never trained) scored by
               evaluate.score_actor is read by export.read_run, and these tests' scores writer matches evaluate's
               format; the package constants match evaluate's and actors_f32's; the real start banks' anchors;
               ReBRAC's substituted shares from the real prepared manifests (data, not outcomes)

The fresh dry run is made once by the stage driver (training environment, processor, cores 0-3; made-up arrays):
  env D4RL_SUPPRESS_IMPORT_ERROR=1 MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa DISPLAY= \\
    MUJOCO_PY_MUJOCO_PATH=/root/.mujoco/mujoco210 LD_LIBRARY_PATH=/root/.mujoco/mujoco210/bin \\
    PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 JAX_PLATFORMS=cpu SHIFT_HOSTS_DEVICE=cpu taskset -c 0-3 \\
    /opt/bca-venv/bin/python -m experiments.shift_hosts.stages run --dry-run --root runs/shift_hosts/stage0/claims/dryrun \\
    --qualities medium-expert --seeds 0 1 2 3 4 --card-cores 0 1 --lane-cores 2 3
Run from /root/bca-hosts in either environment (the scoring environment runs every case; /opt/bca-venv skips the
integration cases):
  env PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 JAX_PLATFORMS=cpu SHIFT_HOSTS_DEVICE=none taskset -c 0-3 \\
      /opt/bca-shift-venv/bin/python -m experiments.shift_hosts.tests.test_claims
  env PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 JAX_PLATFORMS=cpu SHIFT_HOSTS_DEVICE=none taskset -c 0-3 \\
      /opt/bca-venv/bin/python -m experiments.shift_hosts.tests.test_claims
Results: runs/shift_hosts/stage0/claims/test_claims_none.json and test_claims_none-training-env.json (+ sidecars);
work files under runs/shift_hosts/stage0/claims/work/ (made-up files only).
"""

import copy
import hashlib
import json
import math
import re
import shutil
import sys
import types
import unittest
from pathlib import Path

import numpy as np
from scipy import stats

from experiments.shift_hosts import claims as C
from experiments.shift_hosts import data_copy as DC
from experiments.shift_hosts import export as E
from experiments.shift_hosts import gate
from experiments.shift_hosts import guard
from experiments.shift_hosts import handoff as H
from experiments.shift_hosts import supplement as S
from experiments.shift_hosts import tests as T

AREA = "claims"
REPO = Path(__file__).resolve().parents[3]
WORK = REPO / "runs" / "shift_hosts" / "stage0" / AREA / "work"
SCORING_ENV = np.__version__.startswith("2.")
PLAN_ANCHORS = {"none": dict(random=-800.29, top=-35.19), "mild": dict(random=-1061.42, top=-54.66),
                "strong": dict(random=-1351.41, top=-80.61)}
ANCHORS = dict(anchors=PLAN_ANCHORS, recording_points=None, path=None, sha256=None)
SITUATION_ANCHORS = {b: dict(random=-300.0, top=-30.0) for b in C.SITUATIONS}
BASE = [-200.0, -205.5, -198.25, -202.75, -201.0]           # made-up per-seed returns of a learner alone
NULL1 = [0.31, -0.42, 0.12, 0.44, -0.17]                    # made-up null differences (mean about 0.06)
NULL2 = [0.21, -0.33, 0.14, 0.02, 0.09]
SCORING_CODE = ("experiments/shift_hosts/evaluate.py", "experiments/shift_hosts/policies.py",
                "experiments/shift_hosts/actors_f32.py", "experiments/shift_hosts/handoff.py",
                "experiments/shift_hosts/scoring_cache.py", "experiments/shift/truth.py",
                "experiments/shift/thruster_chain.py", "experiments/shift/recorder.py", "experiments/shift_iw/data.py",
                "experiments/shift_iw/world.py", "outputs/shift/oracle/oracle.py")   # evaluate.code_sha256()'s files


def sha(text):
    return hashlib.sha256(text.encode("utf8")).hexdigest()


def values_with(mean, sd, n=5):
    """n values with exactly this sample mean and (n - 1) standard deviation, up to rounding."""
    z = np.arange(n, dtype=np.float64) - (n - 1) / 2.0
    z = z / z.std(ddof=1)
    return list(mean + sd * z)


# ------------------------------------------------------------------------------------------------ claims inputs

def mk(J=None, *, state="complete", finite=True, events=None, n=1000, coverage=None, pointer=None, tag="x"):
    """A claims-input run (claims.py's module docstring): J {bank: value} (others -100), events {(event, bank): k}."""
    J = J or {}
    ev = events or {}
    done = state == "complete"
    return dict(state=state, finite=bool(finite and done),
                J={b: float(J.get(b, -100.0)) for b in C.BANKS} if done else {},
                points={b: 50.0 for b in C.BANKS} if done else {},
                events={e: {b: int(ev.get((e, b), 10)) for b in C.BANKS} for e in C.EVENTS} if done else {},
                starts={b: n for b in C.BANKS} if done else {}, coverage=coverage, pointer=pointer,
                stop=None if done else dict(rule="S5", reason="made-up stop", kind="outcome"),
                source=dict(record=sha(tag + " record"), scores=sha(tag + " scores") if done else None))


def cov(own=0.92, std=0.85):
    return dict(own=dict(outside=own, inside=0.5, all=0.8), standard=dict(outside=std, inside=0.4, all=0.75))


def scenario(d1=None, d2=None, n1=None, n2=None, alone_none=None, coverage=None, events=None):
    """Made-up runs of one quality: alone strong = BASE; standard = alone + d1[learner]; IW = standard + d2[learner];
    no-shift likewise with n1, n2; IW coverage per learner (default inside the bars); events
    {(learner, setup): {...}}."""
    d1, d2, n1, n2 = d1 or {}, d2 or {}, n1 or {}, n2 or {}
    alone_none, coverage, events = alone_none or {}, coverage or {}, events or {}
    runs = {}
    for L in C.LEARNERS:
        g1, g2 = d1.get(L, NULL1), d2.get(L, NULL2)
        h1, h2 = n1.get(L, NULL2), n2.get(L, NULL1)
        an = alone_none.get(L, [-100.0, -101.0, -99.5, -100.5, -100.25])
        for i, s in enumerate(C.SEEDS):
            a = BASE[i]
            runs.setdefault((L, "alone"), {})[s] = mk({"strong": a, "none": an[i]},
                                                      events=events.get((L, "alone")), tag=f"{L} alone {s}")
            runs.setdefault((L, "standard"), {})[s] = mk({"strong": a + g1[i], "none": an[i] + h1[i]},
                                                         events=events.get((L, "standard")), coverage=cov(),
                                                         tag=f"{L} standard {s}")
            c = coverage.get(L, cov())
            runs.setdefault((L, "iw"), {})[s] = mk({"strong": a + g1[i] + g2[i], "none": an[i] + h1[i] + h2[i]},
                                                   events=events.get((L, "iw")),
                                                   coverage=c[i] if isinstance(c, list) else c,
                                                   pointer=dict(change=0.04, extra=0.3), tag=f"{L} iw {s}")
    return runs


def judge(runs, quality="medium-expert"):
    return C.judge(quality, runs, PLAN_ANCHORS["none"])


def badges(j, cid):
    return {L: j["learners"][L][cid]["badge"] for L in C.LEARNERS}


GAIN = [8.1, 7.9, 8.4, 7.6, 8.0]          # a clear gain (mean 8.0)


class Statistics(unittest.TestCase):
    def test_t95_matches_scipy_and_the_plan(self):
        """T95 equals scipy.stats.t.ppf(0.95, 4) within 1e-12 and rounds to the plan's 2.132."""
        self.assertLess(abs(C.T95 - stats.t.ppf(0.95, 4)), 1e-12)
        self.assertEqual(round(C.T95, 3), 2.132)
        self.assertLess(abs(C.t4_sf(C.T95) - 0.05), 1e-15)

    def test_t4_sf_matches_scipy(self):
        """The closed-form tail of Student's t with 4 degrees of freedom equals scipy's, relative 1e-12."""
        grid = np.concatenate([np.linspace(-60, 60, 2401), [0.0, 1e-9, -1e-9, 3.4954, 3.471, 1e3, -1e3, 1e6]])
        for t in grid:
            ours, ref = C.t4_sf(t), float(stats.t.sf(t, 4))
            self.assertLessEqual(abs(ours - ref), 1e-12 * max(ref, 1e-300) + 1e-300, msg=f"t={t}")
        self.assertEqual(C.t4_sf(math.inf), 0.0)
        self.assertEqual(C.t4_sf(-math.inf), 1.0)

    def test_bounds_and_p_match_scipy(self):
        """mean, sd (n - 1), lower and upper bounds = mean -+ t.ppf(0.95, 4) sd / sqrt(5), and the one-sided p-value
        equal numpy/scipy (ttest_1samp, alternative 'greater'), on 300 made-up vectors; L > 0 exactly when p < 0.05."""
        rng = np.random.default_rng([20261016, 7779, 4101])
        t1 = stats.t.ppf(0.95, 4)
        agree = 0
        for k in range(300):
            v = rng.normal(rng.normal(0, 3), rng.uniform(0.1, 8), 5)
            s = C.summary(v)
            m, sd = float(np.mean(v)), float(np.std(v, ddof=1))
            se = sd / math.sqrt(5)
            self.assertEqual(s["mean"], m)
            self.assertEqual(s["sd"], sd)
            self.assertLess(abs(s["lower"] - (m - t1 * se)), 1e-12 * max(1.0, abs(m)))
            self.assertLess(abs(s["upper"] - (m + t1 * se)), 1e-12 * max(1.0, abs(m)))
            p = float(stats.ttest_1samp(v, 0.0, alternative="greater").pvalue)
            self.assertLess(abs(s["p_one_sided"] - p), 1e-12)
            if abs(s["p_one_sided"] - 0.05) > 1e-9:
                self.assertEqual(s["lower"] > 0, s["p_one_sided"] < 0.05)
                agree += 1
        self.assertGreater(agree, 290)

    def test_summary_edge_cases(self):
        """sd = 0 gives p 0 (mean above 0) or 1, bounds equal to the mean; fewer than 5 seeds or a non-finite seed
        gives no mean, bounds or p (a claim is never judged on fewer seeds); the weighting test's p_one_sided rule."""
        s = C.summary([2.0] * 5)
        self.assertEqual((s["p_one_sided"], s["lower"], s["upper"]), (0.0, 2.0, 2.0))
        self.assertEqual(C.summary([0.0] * 5)["p_one_sided"], 1.0)
        self.assertEqual(C.summary([-1.0] * 5)["p_one_sided"], 1.0)
        for bad in ([1, 2, 3, 4], [1, 2, 3, 4, None], [1, 2, 3, 4, math.nan], [1, 2, 3, 4, math.inf]):
            s = C.summary(bad)
            self.assertFalse(s["complete"])
            self.assertIsNone(s["mean"])
            self.assertIsNone(s["p_one_sided"])
            self.assertEqual(C.family_p(s), 1.0)
        self.assertFalse(C.summary([1, 2, 3, 4, 5], missing=[2])["complete"])

    def test_power_figure_sanity(self):
        """Plan section 5.5: a gain of 6.93 with sd 4.463 gives p about 0.0128 (t about 3.47, lower bound about +2.67);
        Holm's first step needs p <= 0.0125, i.e. t >= 3.495, so that result alone fails it but passes at rank 2 when
        another learner also passes."""
        for mean in (6.93, 6.928):
            s = C.summary(values_with(mean, 4.463))
            self.assertAlmostEqual(s["sd"], 4.463, places=12)
            self.assertAlmostEqual(s["p_one_sided"], 0.0128, delta=5e-5)
            self.assertAlmostEqual(s["t"], 3.47, delta=0.005)
            self.assertAlmostEqual(s["lower"], 2.67, delta=0.01)
        self.assertAlmostEqual(stats.t.ppf(1 - 0.0125, 4), 3.4954, places=4)
        p = C.summary(values_with(6.928, 4.463))["p_one_sided"]
        self.assertGreater(p, 0.0125)
        self.assertEqual(C.holm(dict(td3_bc=p, cql=1.0, iql=1.0, rebrac=1.0))["td3_bc"], False)
        both = C.holm(dict(td3_bc=p, cql=0.001, iql=1.0, rebrac=1.0))
        self.assertTrue(both["td3_bc"] and both["cql"])


class Holm(unittest.TestCase):
    def test_weighting_test_cases(self):
        """The weighting test's own Holm cases (iw/SPEC.md:600): 0.010, 0.020, 0.030 all pass; 0.010, 0.030, 0.040
        only the first; k = 1 is the uncorrected test."""
        self.assertEqual(C.holm(dict(a=0.010, b=0.020, c=0.030)), dict(a=True, b=True, c=True))
        self.assertEqual(C.holm(dict(a=0.010, b=0.030, c=0.040)), dict(a=True, b=False, c=False))
        self.assertEqual(C.holm(dict(a=0.049)), dict(a=True))
        self.assertEqual(C.holm(dict(a=0.051)), dict(a=False))

    def test_all_pass_and_bars(self):
        """Four learners: bars 0.0125, 0.05/3, 0.025, 0.05; all pass when each sorted p meets its bar."""
        rows = C.holm_table(dict(td3_bc=0.010, cql=0.012, iql=0.015, rebrac=0.02))
        self.assertEqual([r["bar"] for r in rows], [0.05 / 4, 0.05 / 3, 0.05 / 2, 0.05])
        self.assertTrue(all(r["passes"] for r in rows))

    def test_ties(self):
        """Tied p-values: the order among ties (by name) never changes who passes."""
        self.assertEqual(C.holm(dict(a=0.0125, b=0.0125, c=0.9, d=0.9)), dict(a=True, b=True, c=False, d=False))
        self.assertEqual(C.holm(dict(a=0.02, b=0.02, c=0.02, d=0.02)), dict(a=False, b=False, c=False, d=False))
        self.assertEqual(C.holm(dict(a=0.01, b=0.01, c=0.01, d=0.01)), dict(a=True, b=True, c=True, d=True))
        rng = np.random.default_rng([20261016, 7779, 4102])
        for _ in range(300):
            p = [float(x) for x in rng.choice([0.001, 0.0125, 0.014, 0.02, 0.03, 0.2, 1.0], 4)]
            base = C.holm(dict(zip("abcd", p)))
            names = "".join(rng.permutation(list("abcd")))      # the same p-values under other names (tie order)
            again = C.holm(dict(zip(names, p)))
            self.assertEqual([again[names[i]] for i in range(4)], [base["abcd"[i]] for i in range(4)])

    def test_p_one_member_stays_in(self):
        """A learner with p = 1 (did not learn, not finite, or Claim 4a failed) stays in the family of four: it fails
        and the others keep the bars of k = 4; dropping it would loosen them (k = 3 lets 0.03 pass)."""
        four = C.holm(dict(td3_bc=0.001, cql=1.0, iql=0.012, rebrac=0.03))
        self.assertEqual(four, dict(td3_bc=True, cql=False, iql=True, rebrac=False))
        three = C.holm(dict(td3_bc=0.001, iql=0.012, rebrac=0.03))
        self.assertTrue(three["rebrac"])

    def test_exact_boundaries(self):
        """p exactly at the bar passes (<=); the next float above fails."""
        self.assertTrue(C.holm(dict(a=0.0125, b=1.0, c=1.0, d=1.0))["a"])
        self.assertFalse(C.holm(dict(a=float(np.nextafter(0.0125, 1.0)), b=1.0, c=1.0, d=1.0))["a"])

    def test_adjusted_agrees_with_step_down(self):
        """The Holm-adjusted p-value is at most 0.05 exactly when the step-down passes (away from exact ties with the
        bar), and is monotone in Holm's order."""
        rng = np.random.default_rng([20261016, 7779, 4103])
        for _ in range(2000):
            p = dict(zip(C.LEARNERS, rng.uniform(0, 0.08, 4) ** rng.uniform(0.5, 2)))
            rows = C.holm_table(p)
            for r in rows:
                if abs(r["adjusted"] - 0.05) > 1e-12:
                    self.assertEqual(r["passes"], r["adjusted"] <= 0.05)
            adj = [r["adjusted"] for r in rows]
            self.assertEqual(adj, sorted(adj))

    def test_families_pending_until_complete(self):
        """Claims 1 and 2 read pending, for all four learners, while any of the 60 claim runs is missing; flags of
        learners with complete inputs are judged; the missing learner's flags wait."""
        runs = scenario(d1=dict(td3_bc=GAIN))
        del runs[("cql", "iw")][3]
        j = judge(runs)
        self.assertFalse(j["families"]["1"]["judged"])
        self.assertFalse(j["families"]["2"]["judged"])
        self.assertEqual(set(badges(j, "1").values()), {C.PENDING})
        self.assertEqual(set(badges(j, "2").values()), {C.PENDING})
        self.assertEqual(badges(j, "1c")["td3_bc"], C.PASSED)
        self.assertEqual(badges(j, "2c")["cql"], C.PENDING)
        self.assertEqual(badges(j, "4a")["cql"], C.PENDING)
        self.assertFalse(j["complete"]["claim_setups"])
        self.assertEqual(j["complete"]["missing"], [dict(learner="cql", setup="iw", seed=3)])

    def test_claim2_waits_for_every_4a(self):
        """With every run present, Claim 2's family waits for each learner's Claim 4a (a missing coverage readout);
        Claim 1's family is judged."""
        runs = scenario(d1=dict(td3_bc=GAIN))
        runs[("iql", "iw")][2] = dict(runs[("iql", "iw")][2], coverage=None)
        j = judge(runs)
        self.assertTrue(j["families"]["1"]["judged"])
        self.assertFalse(j["families"]["2"]["judged"])
        self.assertIn("IQL's Claim 4a", j["families"]["2"]["waiting_for"])
        self.assertEqual(badges(j, "4a")["iql"], C.PENDING)
        self.assertEqual(set(badges(j, "2").values()), {C.PENDING})
        self.assertEqual(badges(j, "1")["td3_bc"], C.PASSED)


class Claim0(unittest.TestCase):
    def test_bars_from_the_plan_anchors(self):
        """60 points = -341.23 and 39 points = -501.90 return units from the no-shift anchors -800.29 and -35.19
        (plan section 6.2: -341.2 and -501.9)."""
        me, md = (C.claim0_bar(q, C.PLAN_NO_SHIFT_ANCHORS) for q in ("medium-expert", "medium"))
        self.assertEqual((me["points"], md["points"]), (60.0, 39.0))
        self.assertAlmostEqual(me["return_units"], -341.23, places=9)
        self.assertAlmostEqual(md["return_units"], -501.901, places=9)
        self.assertEqual((round(me["return_units"], 1), round(md["return_units"], 1)), (-341.2, -501.9))
        self.assertEqual({q: round(p - 10) for q, p in C.CLAIM0_RECORDING_POINTS.items()}, C.CLAIM0_BAR_POINTS)

    @unittest.skipUnless((DC.start_banks_dir() / "start_banks.json").exists(), "the start banks' copy is missing")
    def test_bars_from_the_real_start_banks(self):
        """From the verified copy of start_banks.json (fixed controllers only): the recording controllers' no-shift
        points minus 10 round to 60 and 39, and the bars round to -341.2 and -501.9 return units."""
        a = E.read_start_bank_anchors()
        for q in C.QUALITIES:
            self.assertEqual(round(a["recording_points"][q] - 10), C.CLAIM0_BAR_POINTS[q])
            self.assertEqual(round(a["recording_points"][q], 2), C.CLAIM0_RECORDING_POINTS[q])
        bars = {q: C.claim0_bar(q, a["anchors"]["none"])["return_units"] for q in C.QUALITIES}
        self.assertEqual((round(bars["medium-expert"], 1), round(bars["medium"], 1)), (-341.2, -501.9))
        # the exact bars the verdicts use (claims.py's docstring and PRESPEC section 10 name them)
        self.assertEqual((round(bars["medium-expert"], 4), round(bars["medium"], 4)), (-341.2314, -501.9014))
        self.assertEqual((a["anchors"]["none"]["random"], a["anchors"]["none"]["top"]),
                         (-800.2886042574038, -35.193218771607356))
        self.assertIn("about 5-9 return units", C.POWER_NOTE)
        self.assertIn("5.25-7.25 if all four", C.POWER_NOTE)

    def test_below_at_and_above_the_bar(self):
        """Mean over 5 seeds below the bar: did not learn; above: passed (medium-expert, bar -341.23). Exactly at the
        bar passes ("below" shelves): checked with anchors that make the bar exactly -320.0."""
        bar = C.claim0_bar("medium-expert", PLAN_ANCHORS["none"])["return_units"]
        for shift, want in ((-0.01, C.DID_NOT_LEARN), (0.01, C.PASSED)):
            vals = [bar + shift - 1.0, bar + shift + 1.0, bar + shift, bar + shift - 0.5, bar + shift + 0.5]
            j = judge(scenario(alone_none=dict(cql=vals)))
            self.assertEqual(j["learners"]["cql"]["0"]["badge"], want, msg=shift)
        exact = dict(random=-800.0, top=0.0)
        self.assertEqual(C.claim0_bar("medium-expert", exact)["return_units"], -320.0)
        for v, want in ((-320.0, C.PASSED), (float(np.nextafter(-320.0, -1e9)), C.DID_NOT_LEARN)):
            j = C.judge("medium-expert", scenario(alone_none=dict(cql=[v] * 5)), exact)
            self.assertEqual(j["learners"]["cql"]["0"]["badge"], want, msg=v)

    def test_did_not_learn_rows_and_family(self):
        """A learner that did not learn: Claims 1-6 read not judged, and it stays in both families with p = 1, so the
        others keep the bars of four learners."""
        runs = scenario(d1=dict(td3_bc=GAIN), alone_none=dict(rebrac=[-420.0] * 5))
        j = judge(runs)
        res = j["learners"]["rebrac"]
        self.assertEqual(res["0"]["badge"], C.DID_NOT_LEARN)
        for cid in C.READING_ORDER[1:]:
            self.assertEqual(res[cid]["badge"], C.NOT_JUDGED, msg=cid)
        for cid in ("1", "2"):
            fam = j["families"][cid]
            self.assertTrue(fam["judged"])
            self.assertEqual(fam["p_values"]["rebrac"], 1.0)
            self.assertEqual(len(fam["holm"]), 4)
        self.assertEqual(j["learners"]["td3_bc"]["1"]["family"]["bar"], 0.0125)
        table = C.phone_table(j, runs)
        self.assertEqual({r["status"] for r in table if r["learner"] == "rebrac"}, {C.DID_NOT_LEARN})

    def test_non_finite_and_stops(self):
        """Any seed of the learner alone not finite, a stopped alone run, or a stopped room reference (a failure in
        Unifloral's own code, plan section 6.1): did not learn; a missing alone seed: pending."""
        runs = scenario()
        runs[("td3_bc", "alone")][2] = mk({"strong": -200.0}, finite=False, tag="nf")
        runs[("cql", "alone")][4] = mk(state="stopped", tag="st")
        runs[("iql", "room")] = {0: mk(state="stopped", tag="room")}
        del runs[("rebrac", "alone")][1]
        j = judge(runs)
        self.assertEqual(badges(j, "0"), dict(td3_bc=C.DID_NOT_LEARN, cql=C.DID_NOT_LEARN, iql=C.DID_NOT_LEARN,
                                              rebrac=C.PENDING))
        self.assertIn("room reference", j["learners"]["iql"]["0"]["verdict"])


class Badges(unittest.TestCase):
    def test_badge_names(self):
        self.assertEqual(set(C.BADGES), {"passed", "shelved", "flagged", "pending", "not judged", "did not learn"})

    def test_claim1_size_bar_and_bound(self):
        """1.79 return units is shelved and 1.81 passes (L > 0, Holm passes); a gain with L <= 0 is shelved; the
        TD3+BC pass on medium-expert carries the selection label, on medium it does not."""
        lo = [1.78, 1.80, 1.79, 1.795, 1.785]
        hi = [1.80, 1.82, 1.81, 1.815, 1.805]
        j = judge(scenario(d1=dict(td3_bc=lo)))
        self.assertEqual(j["learners"]["td3_bc"]["1"]["badge"], C.SHELVED)
        self.assertTrue(j["learners"]["td3_bc"]["1"]["family"]["passes"])
        self.assertIn("below +1.8", " ".join(j["learners"]["td3_bc"]["1"]["reasons"]))
        j = judge(scenario(d1=dict(td3_bc=hi)))
        r = j["learners"]["td3_bc"]["1"]
        self.assertEqual(r["badge"], C.PASSED)
        self.assertIn(C.SELECTION_LABEL, r["labels"])
        self.assertIn("Holm-adjusted p < 0.0001", r["verdict"])         # a tiny p never reads as exactly 0
        j = judge(scenario(d1=dict(td3_bc=hi)), quality="medium")
        self.assertEqual(j["learners"]["td3_bc"]["1"]["badge"], C.PASSED)
        self.assertNotIn(C.SELECTION_LABEL, j["learners"]["td3_bc"]["1"]["labels"])
        j = judge(scenario(d1=dict(cql=GAIN, td3_bc=[9.0, -6.0, 12.0, -4.0, 8.0])))
        self.assertEqual(j["learners"]["td3_bc"]["1"]["badge"], C.SHELVED)
        self.assertIn("not above 0", " ".join(j["learners"]["td3_bc"]["1"]["reasons"]))
        self.assertEqual(j["learners"]["cql"]["1"]["badge"], C.PASSED)
        self.assertNotIn(C.SELECTION_LABEL, j["learners"]["cql"]["1"]["labels"])

    def test_claim1_fails_holm_only(self):
        """A gain of +2.5 with a spread that leaves p between 0.0125 and 0.05: size and bound pass, Holm does not (all
        other learners null), so it is shelved with the Holm reason."""
        v = values_with(2.5, 2.0)
        s = C.summary(v)
        self.assertTrue(0.0125 < s["p_one_sided"] < 0.05 and s["lower"] > 0)
        j = judge(scenario(d1=dict(iql=v)))
        r = j["learners"]["iql"]["1"]
        self.assertEqual(r["badge"], C.SHELVED)
        self.assertEqual(len(r["reasons"]), 1)
        self.assertIn("Holm", r["reasons"][0])

    def test_claim4a_4b_bars_and_claim2(self):
        """4a passes inside [89%, 96%] and is shelved outside; a failed 4a gives Claim 2 p = 1 (shelved even with a
        large gain); 4b passes below 88% and is shelved at 88% or more, and a passing Claim 2 is then labeled."""
        for own, want in ((0.8901, C.PASSED), (0.8899, C.SHELVED), (0.9599, C.PASSED), (0.9601, C.SHELVED)):
            j = judge(scenario(coverage=dict(cql=cov(own=own))))
            self.assertEqual(j["learners"]["cql"]["4a"]["badge"], want, msg=own)
        for std, want in ((0.8799, C.PASSED), (0.8801, C.SHELVED)):
            j = judge(scenario(coverage=dict(cql=cov(std=std))))
            self.assertEqual(j["learners"]["cql"]["4b"]["badge"], want, msg=std)
        j = judge(scenario(d2=dict(iql=GAIN), coverage=dict(iql=cov(own=0.97))))
        r = j["learners"]["iql"]["2"]
        self.assertEqual((r["badge"], j["families"]["2"]["p_values"]["iql"]), (C.SHELVED, 1.0))
        self.assertIn("Claim 4a did not pass", r["reasons"])
        j = judge(scenario(d2=dict(iql=GAIN), coverage=dict(iql=cov(std=0.90))))
        r = j["learners"]["iql"]["2"]
        self.assertEqual(r["badge"], C.PASSED)
        self.assertIn(C.WIDTH_LABEL, r["labels"])
        self.assertIn(C.OVERLAP_LABEL, r["labels"])
        j = judge(scenario(d2=dict(iql=GAIN)))
        self.assertNotIn(C.WIDTH_LABEL, j["learners"]["iql"]["2"]["labels"])

    def test_flags(self):
        """1c and 2c (strong mix), 3 and 3b (no shift): flagged when Delta <= -1.8 with an upper bound below 0;
        -1.79, or an upper bound at or above 0, is not flagged (badge passed, 'not flagged')."""
        cost = [-3.1, -2.9, -3.0, -3.2, -2.8]
        j = judge(scenario(d1=dict(td3_bc=cost), d2=dict(cql=cost), n1=dict(iql=cost), n2=dict(rebrac=cost)))
        self.assertEqual(badges(j, "1c"), dict(td3_bc=C.FLAGGED, cql=C.PASSED, iql=C.PASSED, rebrac=C.PASSED))
        self.assertEqual(badges(j, "2c")["cql"], C.FLAGGED)
        self.assertEqual(badges(j, "3b")["iql"], C.FLAGGED)
        self.assertEqual(badges(j, "3")["rebrac"], C.FLAGGED)
        self.assertIn("not flagged", j["learners"]["cql"]["1c"]["verdict"])
        j = judge(scenario(d1=dict(td3_bc=[-1.78, -1.80, -1.79, -1.795, -1.785]),
                           d2=dict(cql=[-9.0, 6.0, -12.0, 4.0, -8.0])))
        self.assertEqual(badges(j, "1c")["td3_bc"], C.PASSED)
        self.assertEqual(badges(j, "2c")["cql"], C.PASSED)
        for cid in C.FLAGS:
            self.assertIn(C.POINTER_LABEL, j["learners"]["td3_bc"][cid]["labels"])

    def test_claim5_counts(self):
        """Claim 5: +2 percentage points of 1,000 starts is 20 starts a seed; mean exactly 2.0 with L > 0 is flagged,
        a total of 99 extra starts over 5 seeds is not, 101 is; the setup's own badge shows on its row."""
        for extra, want in (([20] * 5, C.FLAGGED), ([20, 21, 19, 20, 19], C.PASSED),
                            ([20, 22, 20, 19, 20], C.FLAGGED)):
            runs = scenario()
            for i, s in enumerate(C.SEEDS):
                runs[("rebrac", "iw")][s] = dict(runs[("rebrac", "iw")][s], events=dict(
                    f1={b: (10 + extra[i] if b == "big_waves" else 10) for b in C.BANKS},
                    f2={b: 10 for b in C.BANKS}))
            j = judge(runs)
            r = j["learners"]["rebrac"]["5"]
            self.assertEqual(r["badge"], want, msg=extra)
            self.assertEqual(r["by_setup"], dict(standard=C.PASSED, iw=want))
            hit = [c for c in r["comparisons"]
                   if (c["setup"], c["event"], c["bank"]) == ("iw", "f1", "big_waves")]
            self.assertEqual(hit[0]["meets_bar"], sum(extra) >= 100)
            row = [x for x in C.phone_table(j, runs) if x["learner"] == "rebrac" and x["setup"] == "iw"][0]
            self.assertEqual(row["badges"]["5"], want)

    def test_non_finite_seed(self):
        """A standard seed whose return is not finite: Claim 1 enters Holm with p = 1 and is shelved; Claim 1c reads
        not judged; Claim 2 (it uses the standard setup too) also enters with p = 1."""
        runs = scenario(d1=dict(td3_bc=GAIN))
        runs[("cql", "standard")][1] = mk({"strong": -150.0}, finite=False, coverage=cov(), tag="cql nf")
        j = judge(runs)
        self.assertEqual(j["families"]["1"]["p_values"]["cql"], 1.0)
        self.assertEqual(j["families"]["2"]["p_values"]["cql"], 1.0)
        self.assertEqual(badges(j, "1")["cql"], C.SHELVED)
        self.assertIn("a seed's return is not finite", j["learners"]["cql"]["1"]["reasons"])
        self.assertEqual(badges(j, "1c")["cql"], C.NOT_JUDGED)
        self.assertEqual(badges(j, "1")["td3_bc"], C.PASSED)

    def test_claim6(self):
        """Claim 6: not judged where Claims 1 and 2 did not pass; pending until stage 3's twins exist; passed when
        the gain over each twin is at least +1.8 with L > 0; shelved otherwise (credited to the amount of caution)."""
        runs = scenario(d1=dict(td3_bc=GAIN))
        j = judge(runs)
        self.assertEqual(badges(j, "6"), dict(td3_bc=C.PENDING, cql=C.NOT_JUDGED, iql=C.NOT_JUDGED,
                                              rebrac=C.NOT_JUDGED))
        for twin_gain, want in ((GAIN, C.PASSED), (NULL1, C.SHELVED)):
            r2 = dict(runs)
            r2[("td3_bc", "standard_flattened")] = {
                s: mk({"strong": runs[("td3_bc", "standard")][s]["J"]["strong"] - twin_gain[i]}, tag=f"flat {s}")
                for i, s in enumerate(C.SEEDS)}
            j = judge(r2)
            r = j["learners"]["td3_bc"]["6"]
            self.assertEqual(r["badge"], want)
            self.assertEqual(r["parts"]["2"]["badge"], C.NOT_JUDGED)
            if want == C.SHELVED:
                self.assertIn("amount of caution", r["verdict"])
        self.assertEqual(C.stage3_order(judge(runs)), [dict(learner="td3_bc", setups=["standard_flattened"],
                                                            for_claim="1")])

    def test_claim2_pointer(self):
        """Claim 2's pointer: mean dose change x Delta(standard - alone) / mean standard extra caution."""
        j = judge(scenario(d1=dict(td3_bc=GAIN)))
        p = j["pointers"]["td3_bc"]["claim2"]
        self.assertAlmostEqual(p["value"], 0.04 * float(np.mean(GAIN)) / 0.3, places=9)
        self.assertEqual(p["label"], C.POINTER_LABEL)


class Views(unittest.TestCase):
    def test_reading_order(self):
        """Plan section 6.2: the stop rules S0-S11 (with S6b) first, then 0, 4a, 4b, 1, 1c, 2, 2c, 3, 3b, 5, 6."""
        self.assertEqual(C.READING_ORDER, ("0", "4a", "4b", "1", "1c", "2", "2c", "3", "3b", "5", "6"))
        self.assertEqual(C.STOP_RULES, ("S0", "S1", "S2", "S3", "S4", "S5", "S6", "S6b", "S7", "S8", "S9", "S10",
                                        "S11"))
        j = judge(scenario())
        steps = C.reading_order(j, {"S5": dict(status="outcome")})
        self.assertEqual(steps[0]["step"], "stop rules")
        self.assertEqual(steps[0]["status"]["S5"], "outcome")
        self.assertEqual([s["step"] for s in steps[1:]], list(C.READING_ORDER))
        self.assertEqual(list(j["learners"]["td3_bc"]), list(C.READING_ORDER))

    def test_phone_table(self):
        """12 rows; standard rows show Delta against the learner alone (Claim 1), IW rows against standard BCA (Claim
        2), each with its lower bound and, where judged, the Holm-adjusted p-value; every per-seed value carries its
        records' SHA-256."""
        runs = scenario(d1=dict(td3_bc=GAIN))
        j = judge(runs)
        rows = C.phone_table(j, runs)
        self.assertEqual(len(rows), 12)
        self.assertEqual([(r["learner"], r["setup"]) for r in rows],
                         [(L, s) for L in C.LEARNERS for s in C.CLAIM_SETUPS])
        std = rows[1]
        self.assertEqual((std["delta"]["claim"], std["delta"]["comparison"]), ("1", "standard - alone"))
        self.assertAlmostEqual(std["delta"]["mean"], float(np.mean(GAIN)), places=9)
        self.assertEqual(std["delta"]["holm_adjusted_p"], j["learners"]["td3_bc"]["1"]["family"]["adjusted"])
        self.assertEqual(rows[2]["delta"]["comparison"], "iw - standard")
        self.assertEqual(set(rows[2]["badges"]), {"2", "2c", "3", "4a", "4b", "5", "6"})
        self.assertEqual(set(rows[0]["badges"]), {"0"})
        p = std["delta"]["per_seed"][0]
        self.assertEqual(p["a_source"]["record"], sha("td3_bc standard 0 record"))
        self.assertEqual(p["b_source"]["scores"], sha("td3_bc alone 0 scores"))
        self.assertEqual(rows[0]["strong"]["per_seed"][0]["source"]["record"], sha("td3_bc alone 0 record"))

    def test_writeup_format(self):
        """Every shelved claim and every 'did not learn' gets the user's format, in order: The idea / What we tested
        (with the kill rule) / What happened / Why, citing its place in results.json."""
        runs = scenario(alone_none=dict(rebrac=[-420.0] * 5))
        j = judge(runs)
        ws = C.writeups(j)
        shelved = {(L, c) for L in C.LEARNERS for c in C.READING_ORDER
                   if j["learners"][L][c]["badge"] in (C.SHELVED, C.DID_NOT_LEARN)}
        self.assertEqual({(w["learner"], w["claim"]) for w in ws}, shelved)
        self.assertIn(("rebrac", "0"), shelved)
        for w in ws:
            md = w["markdown"]
            pos = [md.index(h) for h in ("**The idea:**", "**What we tested:**", "**What happened:**", "**Why:**")]
            self.assertEqual(pos, sorted(pos))
            self.assertIn("kill rule", md)
            self.assertIn(f"results.json:qualities.medium-expert.claims.{w['learner']}.{w['claim']}", md)
            self.assertNotIn("labelled", md)

    def test_study_rule(self):
        """The study ends after seeds 0-4 when Claims 1 and 2 are shelved for every learner on every quality run;
        pending while any is pending."""
        none = judge(scenario())
        some = judge(scenario(d1=dict(td3_bc=GAIN)))
        runs = scenario()
        del runs[("td3_bc", "alone")][0]
        pend = judge(runs)
        self.assertTrue(C.study_rule({"medium-expert": none})["ends_after_seeds_0_4"])
        self.assertFalse(C.study_rule({"medium-expert": none, "medium": some})["ends_after_seeds_0_4"])
        self.assertEqual(C.study_rule({"medium-expert": none, "medium": pend})["status"], C.PENDING)

    def test_stage3_order(self):
        """Claim 2's twins in Holm order first, then Claim 1's flattened twins, then the optional estimated arm."""
        gain2 = [9.1, 8.9, 9.4, 8.6, 9.0]
        j = judge(scenario(d1=dict(td3_bc=GAIN, iql=gain2), d2=dict(cql=GAIN, rebrac=gain2)))
        order = [(x["learner"], x["for_claim"]) for x in C.stage3_order(j)]
        p2 = j["families"]["2"]["p_values"]
        first2 = sorted(["cql", "rebrac"], key=lambda L: (p2[L], L))
        p1 = j["families"]["1"]["p_values"]
        first1 = sorted(["td3_bc", "iql"], key=lambda L: (p1[L], L))
        self.assertEqual(order, [(L, "2") for L in first2] + [(L, "1") for L in first1] + [(L, "2") for L in first2])


def s7_entry(learner, scope, setup, quality="medium-expert", stage="stage1", seed=0):
    """A version-3 state.json "s7_stopped" entry as the driver writes it (stages.Driver.s7_failed)."""
    label = quality if stage == "stage1" else f"made-up-{quality}"
    run = f"{stage}/{label}/seed{seed}/{learner}-{setup}"
    return dict(quality=quality, learner=learner, scope=scope, setup=setup, run=run, stage=stage,
                file=f"runs/shift_hosts/{run}/lanes/s7-step15000/s7.json", at_utc="2026-10-20T00:01:10+00:00")


class S7Scopes(unittest.TestCase):
    """Version 3 (PRESPEC.md section 20, post hoc): a failed S7 check stops a learner or its IW setup in that quality;
    the stopped scope's claims read not judged with p = 1 in Holm's families, its runs are left out, and the claims
    wait only for the setups that still run."""

    def test_scopes_from_entries(self):
        """s7_scopes: a stopped learner wins over a stopped IW setup; another quality's entries and malformed ones are
        ignored; a missing scope comes from the setup (S7_SCOPE), and with neither the learner stops; S7_SCOPE equals
        the driver's."""
        e = [s7_entry("rebrac", "iw", "iw"), s7_entry("rebrac", "learner", "standard"), s7_entry("cql", "iw", "iw"),
             s7_entry("iql", "learner", "alone", quality="medium"), "not an entry",
             dict(s7_entry("td3_bc", None, "iw_repeat"), scope=None), dict(s7_entry("td3_bc", None, "x"), scope=None,
                                                                           learner="iql")]
        sc = C.s7_scopes(e, "medium-expert")
        self.assertEqual({L: v["scope"] for L, v in sc.items()},
                         dict(rebrac="learner", cql="iw", td3_bc="iw", iql="learner"))
        self.assertEqual(sc["rebrac"]["runs"], ["stage1/medium-expert/seed0/rebrac-iw",
                                                "stage1/medium-expert/seed0/rebrac-standard"])
        self.assertEqual(set(C.s7_scopes(e, "medium")), {"iql"})
        self.assertEqual(C.s7_scopes(None, "medium"), {})
        self.assertTrue(C.s7_excluded(sc, "rebrac", "room") and C.s7_excluded(sc, "cql", "iw")
                        and C.s7_excluded(sc, "cql", "iw_shuffled"))
        self.assertFalse(C.s7_excluded(sc, "cql", "standard") or C.s7_excluded(sc, "cql", "room"))
        from experiments.shift_hosts import stages as ST
        self.assertEqual(C.S7_SCOPE, ST.S7_SCOPE)

    def test_no_entries_change_nothing(self):
        """No entries (an older state), an empty list or another quality's entries give exactly the version-2
        judgment: every claim, family, pointer and table row equal."""
        runs = scenario(d1=dict(td3_bc=GAIN), d2=dict(cql=GAIN))
        base = judge(runs)
        for entries in (None, [], [s7_entry("rebrac", "learner", "standard", quality="medium")]):
            got = C.judge("medium-expert", runs, PLAN_ANCHORS["none"], s7_stopped=entries)
            self.assertEqual(json.dumps(H.plain(got), sort_keys=True), json.dumps(H.plain(base), sort_keys=True))
            self.assertEqual(C.phone_table(got, runs), C.phone_table(base, runs))
        self.assertEqual(base["s7"], dict(scopes={}, excluded=[]))

    def test_learner_scope(self):
        """ReBRAC stopped by S7 (its standard actor): all its claims read not judged with the failing run named, it
        stays in both families with p = 1, the others keep the bars of four learners (TD3+BC's Claim 1 passes at
        0.0125), its rows read 'stopped by S7' with no numbers, and its runs are listed as excluded."""
        runs = scenario(d1=dict(td3_bc=GAIN, rebrac=GAIN))
        runs[("rebrac", "room")] = {s: mk({"strong": -201.0}, tag=f"rebrac room {s}") for s in C.SEEDS}
        runs[("rebrac", "alone")][2] = mk(state="stopped", tag="stopped after the scope")   # left out, never read
        e = [s7_entry("rebrac", "learner", "standard")]
        j = C.judge("medium-expert", runs, PLAN_ANCHORS["none"], s7_stopped=e)
        res = j["learners"]["rebrac"]
        for cid in C.READING_ORDER:
            self.assertEqual(res[cid]["badge"], C.NOT_JUDGED, msg=cid)
            self.assertIs(res[cid]["s7_stopped"], True)
            self.assertIn("stopped by S7", res[cid]["verdict"])
            self.assertIn("stage1/medium-expert/seed0/rebrac-standard", res[cid]["verdict"])
        self.assertIn("p = 1", res["1"]["verdict"])
        for cid in ("1", "2"):
            fam = j["families"][cid]
            self.assertTrue(fam["judged"])
            self.assertEqual(fam["p_values"]["rebrac"], 1.0)
            self.assertEqual(fam["s7_stopped"], ["rebrac"])
            self.assertEqual(len(fam["holm"]), 4)
            self.assertEqual([r["bar"] for r in fam["holm"]], [0.05 / 4, 0.05 / 3, 0.05 / 2, 0.05])
            self.assertEqual(fam["holm"][-1]["learner"], "rebrac")
            self.assertEqual(res[cid]["family"]["p"], 1.0)
            self.assertEqual(res[cid]["delta"]["per_seed"], [])            # its runs never enter a statistic
        self.assertEqual(j["learners"]["td3_bc"]["1"]["badge"], C.PASSED)
        self.assertEqual(j["learners"]["td3_bc"]["1"]["family"]["bar"], 0.0125)
        base = judge(scenario(d1=dict(td3_bc=GAIN), alone_none=dict(rebrac=[-420.0] * 5)))   # rebrac p = 1 there too
        for cid in ("1", "2"):
            self.assertEqual(j["families"][cid]["holm"], base["families"][cid]["holm"])
        self.assertEqual(len(j["s7"]["excluded"]), 20)
        self.assertEqual({x["why"] for x in j["s7"]["excluded"]}, {"excluded: stopped by S7"})
        self.assertEqual(j["s7"]["scopes"]["rebrac"]["scope"], "learner")
        self.assertEqual(j["complete"]["room_references_expected"], 15)
        self.assertIn("until all 15 are complete", j["complete"]["note"])        # ReBRAC's 5 never run
        rows = [r for r in C.phone_table(j, runs) if r["learner"] == "rebrac"]
        self.assertEqual({r["status"] for r in rows}, {C.S7_STOPPED})
        self.assertTrue(all(r["strong"]["per_seed"] == [] and r["strong"]["mean"] is None for r in rows))
        self.assertIsNone(j["pointers"]["rebrac"]["claim2"]["value"])
        self.assertEqual(C.stage3_order(j), [dict(learner="td3_bc", setups=["standard_flattened"], for_claim="1")])

    def test_iw_scope(self):
        """CQL's IW setup stopped by S7: Claims 2, 2c, 3, 4a and 4b read not judged and CQL enters Claim 2's family
        with p = 1; its Claims 0, 1, 1c, 3b and Claim 5's standard rows are judged; Claim 6's Claim-2 part names S7;
        only the IW row reads 'stopped by S7'."""
        runs = scenario(d1=dict(cql=GAIN), d2=dict(cql=GAIN, td3_bc=GAIN),
                        events={("cql", "standard"): {("f1", "big_waves"): 30}})
        j = C.judge("medium-expert", runs, PLAN_ANCHORS["none"], s7_stopped=[s7_entry("cql", "iw", "iw")])
        res = j["learners"]["cql"]
        for cid in ("2", "2c", "3", "4a", "4b"):
            self.assertEqual(res[cid]["badge"], C.NOT_JUDGED, msg=cid)
            self.assertIn("its IW setup stops", res[cid]["verdict"])
        self.assertEqual((res["0"]["badge"], res["1"]["badge"], res["1c"]["badge"], res["3b"]["badge"]),
                         (C.PASSED, C.PASSED, C.PASSED, C.PASSED))
        self.assertEqual(j["families"]["2"]["p_values"]["cql"], 1.0)
        self.assertEqual(j["families"]["2"]["s7_stopped"], ["cql"])
        self.assertEqual(j["families"]["1"]["s7_stopped"], [])
        self.assertLess(j["families"]["1"]["p_values"]["cql"], 0.0125)
        self.assertEqual(j["learners"]["td3_bc"]["2"]["badge"], C.PASSED)
        self.assertEqual(res["5"]["by_setup"], dict(standard=C.FLAGGED, iw=C.NOT_JUDGED))
        self.assertEqual(res["5"]["badge"], C.FLAGGED)
        self.assertIn("IW BCA is not judged: stopped by S7", res["5"]["verdict"])
        self.assertTrue(all(c["setup"] == "standard" for c in res["5"]["comparisons"]))
        self.assertEqual(res["6"]["parts"]["2"]["badge"], C.NOT_JUDGED)
        self.assertIn("stopped by S7", res["6"]["parts"]["2"]["why"])
        self.assertEqual(res["6"]["parts"]["1"]["badge"], C.PENDING)                # Claim 1 passed: stage 3 waits
        self.assertIn("stopped by S7", j["pointers"]["cql"]["claim2"]["why"])
        status = {r["setup"]: r["status"] for r in C.phone_table(j, runs) if r["learner"] == "cql"}
        self.assertEqual(status, dict(alone=None, standard=None, iw=C.S7_STOPPED))
        self.assertEqual({(x["learner"], x["setup"]) for x in j["s7"]["excluded"]}, {("cql", "iw")})

    def test_complete_with_a_stopped_scope(self):
        """Claims 1 and 2 count a stopped scope as complete: with ReBRAC stopped after its seed 0 (its later seeds
        never trained) and CQL's IW setup stopped after seed 1, both families are judged when the rest is complete;
        without the entries they wait (pending)."""
        runs = scenario(d1=dict(td3_bc=GAIN))
        for s in (1, 2, 3, 4):
            for setup in C.CLAIM_SETUPS:
                del runs[("rebrac", setup)][s]
        for s in (2, 3, 4):
            del runs[("cql", "iw")][s]
        self.assertEqual(set(badges(judge(runs), "1").values()), {C.PENDING})
        e = [s7_entry("rebrac", "learner", "alone", stage="stage2"), s7_entry("cql", "iw", "iw", stage="stage2")]
        j = C.judge("medium-expert", runs, PLAN_ANCHORS["none"], s7_stopped=e)
        self.assertTrue(j["complete"]["claim_setups"])
        self.assertEqual(j["complete"]["s7_not_expected"],
                         [dict(learner="cql", setup="iw")] + [dict(learner="rebrac", setup=s) for s in C.CLAIM_SETUPS])
        self.assertTrue(j["families"]["1"]["judged"] and j["families"]["2"]["judged"])
        self.assertEqual(badges(j, "1"), dict(td3_bc=C.PASSED, cql=C.SHELVED, iql=C.SHELVED, rebrac=C.NOT_JUDGED))
        self.assertEqual(badges(j, "2"), dict(td3_bc=C.SHELVED, cql=C.NOT_JUDGED, iql=C.SHELVED,
                                              rebrac=C.NOT_JUDGED))
        del runs[("iql", "standard")][4]                       # a scope S7 did not stop still has to be complete
        j = C.judge("medium-expert", runs, PLAN_ANCHORS["none"], s7_stopped=e)
        self.assertEqual(badges(j, "1"), dict(td3_bc=C.PENDING, cql=C.PENDING, iql=C.PENDING, rebrac=C.NOT_JUDGED))
        self.assertEqual(j["complete"]["missing"], [dict(learner="iql", setup="standard", seed=4)])

    def test_writeup_and_study_rule(self):
        """A stopped scope gets an S7 write-up in the user's format citing results.json's s7_stopped; the study rule
        counts it as not passing (all others shelved: the study ends; TD3+BC passing: it goes on)."""
        e = [s7_entry("rebrac", "learner", "standard"), s7_entry("iql", "iw", "iw_repeat")]
        j = C.judge("medium-expert", scenario(), PLAN_ANCHORS["none"], s7_stopped=e)
        ws = [w for w in C.writeups(j) if w["claim"] == "S7"]
        self.assertEqual([(w["learner"], w["badge"]) for w in ws], [("iql", C.S7_STOPPED), ("rebrac", C.S7_STOPPED)])
        for w in ws:
            md = w["markdown"]
            pos = [md.index(h) for h in ("**The idea:**", "**What we tested:**", "**What happened:**", "**Why:**")]
            self.assertEqual(pos, sorted(pos))
            self.assertIn(f"results.json:qualities.medium-expert.s7_stopped.scopes.{w['learner']}", md)
            self.assertIn("p = 1", md)
            self.assertIn("S7, with version 2's post-hoc limit of 0.05 points", md)      # PRESPEC section 19.10
        self.assertIn("IW setup stopped by S7", ws[0]["title"])
        self.assertTrue(C.study_rule({"medium-expert": j})["ends_after_seeds_0_4"])
        some = C.judge("medium-expert", scenario(d1=dict(td3_bc=GAIN)), PLAN_ANCHORS["none"], s7_stopped=e)
        self.assertFalse(C.study_rule({"medium-expert": some})["ends_after_seeds_0_4"])


# ------------------------------------------------------------------------------------------------ export

PLAN_SCHEDULE = E.PLAN_SCHEDULE
FINAL, HALF, LAST_REFRESH = E.FINAL, E.HALF, E.PLAN_SCHEDULE["last_refresh"]
DRY_SCHEDULE = dict(num_updates=40, eval_interval=10, steps=40, refresh_steps=[20, 30], switch_step=20,
                    key_hash_steps=[10, 20, 40], s10_step=20, snapshot_steps=[20, 30, 40], last_refresh=30)
PARTS = dict(td3_bc=("actor", "actor_target", "dual_q", "dual_q_target"), rebrac=("actor", "actor_target", "dual_q",
                                                                                   "dual_q_target"),
             cql=("actor", "alpha", "vec_q", "vec_q_target"), iql=("actor", "dual_q", "dual_q_target", "value"))
FAKE_IMPORT_RECORD = dict(python="3.10.12", executable="/opt/bca-venv/bin/python",
                          packages=dict(jax="0.6.2", jaxlib="0.6.2", numpy="1.26.4"), card=None,
                          env=dict(JAX_PLATFORMS="cpu", OMP_NUM_THREADS="1"),
                          xla_flags=dict(in_effect="--xla_cpu_multi_thread_eigen=false"), matmul_precision="highest",
                          jax_platforms="cpu", backend="cpu", devices=["TFRT_CPU_0"], device="cpu",
                          unifloral_revision="f2dc127", manifest_sha256=sha("made-up unifloral manifest"),
                          import_seconds=1.5, platform="made-up")
UNIFLORAL_WEIGHT = 0.9375           # made-up mean of IQL's own actor weights (exact in binary)
FROZEN = {}


def frozen(made_up=True):
    """A made-up build's frozen identity (no approval exists, so fingerprint and pre-specification are None, as the
    runner writes them for made-up runs), or a stand-in real identity for read_run's real-record rules."""
    if made_up not in FROZEN:
        FROZEN[made_up] = (E.Frozen(None, None, gate.code_manifest(), made_up=True) if made_up else
                           E.Frozen(sha("made-up fingerprint: test_claims"), sha("made-up prespec: test_claims"),
                                    gate.code_manifest(), made_up=False))
    return FROZEN[made_up]


def bank_means(strong, none, mild):
    return {b: dict(strong=strong, none=none, mild=mild).get(b, -100.0 - 10.0 * i) for i, b in enumerate(C.BANKS)}


def write_scores(path, learner, actor_sha, actor_meta, means, *, n, permission, code, f1=None, finite=True,
                 seed_tag=0):
    """A scores file in evaluate.write_scores's format (handoff.write: per-start arrays "<readout>/<bank>/<metric>",
    summaries, permission, anchors, cache and code hashes in the meta), with made-up numbers: no simulator, no policy.
    means {bank: mean return}; f1 {bank: starts with the event} (default 4)."""
    rng = np.random.default_rng([20261016, 7779, 4200, C.LEARNERS.index(learner), int(seed_tag) % 2 ** 31])
    arrays, summaries = {}, {}
    anchors = dict({b: dict(scale="weighting", bottom=PLAN_ANCHORS[b]["random"], top=PLAN_ANCHORS[b]["top"])
                    for b in C.WEIGHTING_BANKS},
                   **{b: dict(scale="situation", bottom=SITUATION_ANCHORS[b]["random"],
                              top=SITUATION_ANCHORS[b]["top"]) for b in C.SITUATIONS})
    f1 = f1 or {}
    for readout in E.READOUTS[learner]:
        summaries[readout] = {}
        for b in C.BANKS:
            J = means[b] + (0.0 if readout == "scored" else 0.5) + 0.5 * rng.standard_normal(n)
            if not finite and b == "mild":
                J[3] = np.nan
            a = anchors[b]
            pts = 100.0 * ((J - a["bottom"]) / (a["top"] - a["bottom"]))
            f1a, f2a = np.arange(n) < f1.get(b, 4), np.arange(n) < 4
            regret = np.abs(rng.standard_normal(n))
            key = f"{readout}/{b}"
            arrays.update({f"{key}/J": J, f"{key}/Jd": 0.6 * J, f"{key}/points": pts, f"{key}/f1": f1a,
                           f"{key}/f2": f2a, f"{key}/regret": regret, f"{key}/start": np.arange(n, dtype=np.int64)})
            k = max(1, int(round(0.1 * n)))
            summaries[readout][b] = dict(
                bank=b, scale=a["scale"], n=n, finite=bool(np.all(np.isfinite(J))), J_mean=float(J.mean()),
                J_se=float(J.std() / math.sqrt(n)), J_discounted_mean=float(0.6 * J.mean()),
                points_mean=float(pts.mean()), points_se=float(pts.std() / math.sqrt(n)),
                J_capped_mean=float(J.mean() - 1.0), points_capped_mean=float(pts.mean() - 0.1),
                loss_from_unrecorded_throttle_points=-0.1, loss_from_unrecorded_throttle_return=-1.0,
                stall_share=0.05, regret_mean=float(regret.mean()), regret_points_mean=float(regret.mean() / 10),
                regret_min=float(regret.min()), optimum_J_mean=float(J.mean() + regret.mean()),
                f1_rate=float(f1a.mean()), f2_rate=float(f2a.mean()), f2_threshold=2.0,
                worst10_mean=float(np.sort(J)[:k].mean()), p10=float(np.quantile(J, 0.1)), tracking_mean=50.0,
                energy_mean=5.0, late_peak_mean=1.0, max_cost_identity_error=0.0)
    meta = dict(schema=E.SCORES_SCHEMA, learner=learner, actor_npz_sha256=actor_sha, actor_meta=actor_meta,
                permission=permission, summaries=summaries, cache={b: sha("cache " + b) for b in C.BANKS},
                anchors=anchors, code_sha256=code, environment=dict(note="made-up scores for test_claims"))
    return H.write(path, arrays, meta=meta)


def scoring_code(fz):
    return {p: fz.code_manifest[p] for p in SCORING_CODE if p in fz.code_manifest}


class Fake:
    """Writes made-up run folders, lanes and driver files in the layout runner.py, stages.py and band_readout.py write
    (export.py's module docstring; INTERFACES.md sections 13 and 16): every number is made up, no learner runs."""

    def __init__(self, root, fz, n=200):
        self.root, self.fz, self.n = Path(root), fz, n

    def label(self, q):
        return E.quality_label(q, self.fz.made_up)

    def folder(self, q, L, setup, seed, stage="stage2"):
        return self.root / stage / self.label(q) / f"seed{seed}" / f"{L}-{setup}"

    def run_id(self, q, L, setup, seed, stage="stage2"):
        return E.run_id(stage, self.label(q), seed, L, setup)

    def record(self, q, L, setup, seed, *, stage="stage2", stop=None, kind=None, made_up=None, fingerprint="frozen",
               manifest_sha=None, schedule=None, initial=None, key_shift=None, train=None, qv=None,
               pointer=(0.04, 0.3), label=None):
        """runner.execute's record (INTERFACES.md section 13) with made-up hashes and numbers."""
        sched = dict(schedule or PLAN_SCHEDULE)
        made_up = self.fz.made_up if made_up is None else made_up
        label = label or self.label(q)
        rid = E.run_id(stage, label, seed, L, setup)
        word = f"{q} {L} {seed}"
        bca = setup not in E.UNIFLORAL_SETUPS
        code = dict(code_manifest_sha256=manifest_sha or self.fz.manifest_sha256, files=len(self.fz.code_manifest),
                    fingerprint=self.fz.fingerprint if fingerprint == "frozen" else fingerprint,
                    prespec_sha256=self.fz.prespec_sha256)
        data_id = dict(npz_sha256=sha(f"data {q} {seed}"), manifest_sha256=sha(f"data manifest {q} {seed}"))
        identity = dict(schema=E.RUN_SCHEMA, run_id=rid, stage=stage, quality=label, seed=seed, learner=L, setup=setup,
                        made_up=made_up, device="cpu", schedule=sched, data=data_id, code=code)
        end = sched["steps"] if stop is None else int(math.ceil(stop["step"] / sched["eval_interval"])
                                                      * sched["eval_interval"])

        def parts(tag):
            out = {p: sha(f"{tag} {p} {setup} {word}") for p in PARTS[L]}
            if L == "iql":
                out.update({p: (qv or sha(f"{tag} {p} {word}")) for p in E.IQL_SHARED_PARTS})
            out["all"] = sha(f"{tag} all {setup} {word}")
            return out

        hashes = dict(initial=dict(agent_state=dict({p: sha(f"init {p} {word}") for p in PARTS[L]},
                                                    all=initial or sha("init " + word)),
                                   key=sha("key0 " + word), **({"scale": sha("scale " + word),
                                                                "bca_state": sha("bca0 " + word)} if bca else {})),
                      key_carry={str(t): sha(f"key {t} {word}" + (key_shift or "")) for t in sched["key_hash_steps"]
                                 if t <= end},
                      params_10000=None, final=dict(step=end, losses_sha256=sha(f"losses {setup} {word}"),
                                                    parts=parts("final"), key=sha(f"key end {setup} {word}"),
                                                    agent_state=sha(f"final {setup} {word}")))
        if sched["s10_step"] <= end:
            room = "room " if setup == "room" else ""
            hashes["params_10000"] = dict(step=sched["s10_step"], agent_state=sha(f"p10k {room}{word}"),
                                          key=sha("k10k " + word), parts=parts("p10k"))
        if bca:
            hashes["final"].update(bca_state=sha(f"bca end {setup} {word}"),
                                   bca_state_without_refusal_rule=sha(f"bca end {word}"))
        snapshots = {str(t): dict(path=str(self.folder(q, L, setup, seed, stage) / "snapshots" / f"step{t}.npz"),
                                  sha256=sha(f"snapshot {rid} {t}"),
                                  actor=dict(path=str(self.folder(q, L, setup, seed, stage) / "actors" / f"step{t}.npz"),
                                             sha256=sha(f"actor {rid} {t}")))
                     for t in sched["snapshot_steps"] if t <= end}
        refreshes, readouts = [], None
        if bca and setup != "sham_alone":
            iw = setup == "iw"
            for t in [t for t in sched["refresh_steps"] if t <= end]:
                r = dict(step=t, seconds=0.5, valid=True, stored_inputs_valid=True, certified=True, threshold=1.25,
                         lambda_hat=0.9, lambda_hpd=0.95, n_eff=300.0, sigma_post=0.1, clamp_binds=False, unit=0.3,
                         bank_rows=300, scores_sha256=sha(f"scores {t} {word}"), stream=[1, 2, 3], s3_reproduced=None,
                         iw=None, refusal="most_caution" if iw else "standard")
                if iw:
                    r.update(s3_recomputed_equals_refresh_scores=True, s3_reproduced=True, iw_refused_rows=0,
                             table_seconds=0.2, iw=dict(table=dict(grid=[1.0, 1.01], thresholds=[1.25, 1.3]),
                                                        grid_points=2, monotone=True))
                if L == "iql":
                    tr = dict(mean_weight=0.8125, mean_unifloral_weight=UNIFLORAL_WEIGHT, mean_weight_shrinkage=0.5,
                              positive_advantage_share=0.4, refused_rows=0, supported_share=1.0, inputs_valid=True)
                else:
                    tr = dict(mean_dose=1.25, min_dose=1.0, max_dose=1.5, refused_rows=0, supported_share=1.0,
                              inputs_valid=True)
                r.update(train_rows=tr, refused_rows=0, mean_dose=tr.get("mean_dose", tr.get("mean_weight")))
                refreshes.append(r)
            if iw and sched["last_refresh"] is not None and sched["last_refresh"] <= end and pointer is not None:
                change, extra = pointer
                base = (UNIFLORAL_WEIGHT if L == "iql" else 1.0) + extra
                dc = dict(mean=base + change, change_vs_unweighted=change, mean_abs_change=abs(change),
                          max_abs_change=2 * abs(change), high_current_rows=20, high_current_change=1.5 * change)
                readouts = dict(info=dict(note="made-up"), value="actor weight" if L == "iql" else "dose",
                                dose_change=dc, exact=dict(test_steps={}),
                                estimated_weights=dict(dose_change=dict(dc, change_vs_unweighted=0.9 * change)),
                                shuffled_weights=dict(dose_change=dict(dc, change_vs_unweighted=0.1 * change)))
        sub = None
        if L == "rebrac":
            sub = dict(train=dict(share=0.0778 + 0.001 * seed, highest_current_tenth=dict(share=0.1)),
                       bank=dict(share=0.05))
        return dict(
            schema=E.RUN_SCHEMA, kind=kind or ("stage0-test" if made_up else "run"), identity=identity,
            identity_sha256=E.canonical_sha256(identity), run_id=rid, stage=stage, quality=label, seed=seed, learner=L,
            setup=setup, outcome="stopped" if stop else "complete", stop=stop,
            data=dict(path=f"runs/shift_hosts/data/{q}/seed{seed}/prepared.npz", made_up=made_up,
                      rows=dict(train=1980, full=1980, bank=300, test_strong=1000),
                      train_set="full" if setup == "room" else "train",
                      train_sha256=train or sha(("full " if setup == "room" else "train ") + word),
                      bank_sha256=sha("bank " + word) if bca else None, next_action_substituted=sub, **data_id),
            seeds=dict(rep=seed, unifloral=1, s91=2, s39=3), settings=dict(steps=sched["steps"]),
            code=dict(code, approval=None, unifloral_sha256={}),
            environment=dict(import_record=FAKE_IMPORT_RECORD, host="made-up-host", taskset_cores=[0]),
            timings=dict(started_utc="2026-10-16T00:00:00+00:00", finished_utc="2026-10-16T00:10:00+00:00",
                         wall_seconds=100.0 + seed, processor_seconds=95.0, card_seconds=90.0 + seed,
                         compile_seconds=5.0, first_chunk_seconds=6.0, steady_ms_per_outer_step=0.6, outer_steps=end,
                         chunk_seconds=[1.0], chunk_compile_seconds=[0.0], refresh_seconds=[0.5], table_seconds=[],
                         readout_seconds=[]),
            hashes=hashes, snapshots=snapshots,
            validity=dict(all_steps_valid=stop is None or stop["rule"] != "S5",
                          first_invalid_step=stop["step"] if stop and stop["rule"] == "S5" else None,
                          flags=["inputs_valid"] if bca else [], losses_checked="every floating output"),
            losses=[], refreshes=refreshes, readouts=readouts)

    def actor_meta(self, rec, step):
        """The actor file's meta as runner.Run.snapshot writes it (plus actors_f32's own fields)."""
        return dict(learner=rec["learner"], setup=rec["setup"], step=int(step), quality=rec["quality"],
                    seed=rec["seed"], made_up=rec["data"]["made_up"], run_id=rec["run_id"],
                    identity_sha256=rec["identity_sha256"], actor_tree_sha256=sha("tree " + rec["run_id"]),
                    run_dir="made-up", schema="shift-hosts-actor-v1")

    def run(self, q, L, setup, seed, *, strong=-200.0, none=-100.0, mild=-150.0, finite=True, f1=None,
            coverage=(0.92, 0.86), scored=True, half=True, actor_mismatch=False, code=None, coverage_record=None,
            meta_run_id=None, s7=True, **rec_kw):
        """One run folder: record.json, the scoring lane's lanes/scores-step<T>.npz (final and 50%), the coverage
        lane's lanes/coverage-step<R>/coverage.{json,npz} (standard and IW) and S7's lane (seed 0, alone/standard/IW).
        Returns the record."""
        stage = rec_kw.get("stage", "stage2")
        folder = self.folder(q, L, setup, seed, stage)
        if folder.exists():
            shutil.rmtree(folder)
        folder.mkdir(parents=True)
        rec = self.record(q, L, setup, seed, **rec_kw)
        H.write_json(folder / "record.json", rec)
        rsha = E.sha256_file(folder / "record.json")
        if rec["outcome"] != "complete" or not scored:
            return rec
        sched = rec["identity"]["schedule"]
        perm = (dict(kind="made-up") if rec["data"]["made_up"] else
                dict(kind="approval", fingerprint=self.fz.fingerprint, prespec_sha256=self.fz.prespec_sha256))
        for step in ((sched["steps"], E.half_step(sched)) if half else (sched["steps"],)):
            shift = 0.0 if step == sched["steps"] else -0.5
            am = self.actor_meta(rec, step)
            if meta_run_id:
                am["run_id"] = meta_run_id
            asha = "0" * 64 if actor_mismatch else rec["snapshots"][str(step)]["actor"]["sha256"]
            write_scores(folder / "lanes" / f"scores-step{step}.npz", L, asha, am,
                         bank_means(strong + shift, none + shift, mild + shift), n=self.n, permission=perm,
                         code=code or scoring_code(self.fz), f1=f1, finite=finite,
                         seed_tag=int(sha(f"{L} {setup} {seed} {step}")[:8], 16))
        if setup in E.COVERAGE_SETUPS and coverage is not None:
            self.coverage(folder, rec, coverage, coverage_record or rsha)
        if s7 and seed == 0 and setup in E.S7_SETUPS:
            self.s7(folder, rec, sched["steps"])
        return rec

    def coverage(self, folder, rec, shares, record_sha):
        """band_readout's coverage output: coverage.npz (handoff.write) and then coverage.json over its manifest."""
        last = rec["refreshes"][-1]
        d = folder / "lanes" / f"coverage-step{last['step']}"
        iw = last.get("iw") is not None
        arrays = {"recorded/y": np.zeros(8, np.float32), "recorded/q": np.zeros(8, np.float32)}
        man = H.write(d / "coverage.npz", arrays, meta=dict(schema=E.COVERAGE_SCHEMA, kind="coverage",
                                                            run_id=rec["run_id"], step=last["step"]))

        def summ(c):
            return dict(steps=100, episodes=10, coverage=c, interval=[c - 0.02, c + 0.02], refused_share=0.0,
                        certified_steps=100, certified_coverage=c)

        def arm(c):
            return dict(outside=summ(c), inside=summ(0.5), all=summ(0.8))

        recorded = dict(rows=100, unit=0.3, standard_threshold=1.25, masses_beyond_grid=0, standard=arm(shares[1]),
                        seconds=0.1)
        if iw:
            recorded["iw"] = arm(shares[0])
        H.write_json(d / "coverage.json", dict(
            schema=E.COVERAGE_SCHEMA, kind="coverage", run_id=rec["run_id"], learner=rec["learner"],
            setup=rec["setup"], step=last["step"], record_sha256=record_sha, arrays_npz_sha256=man["npz_sha256"],
            groups=dict(recorded=recorded), standard_threshold=1.25, iw_table=iw, device="cpu"))

    def s7(self, folder, rec, step):
        d = folder / "lanes" / f"s7-step{step}"
        H.write(d / "states.npz", dict(state=np.zeros((4, 13)), obs=np.zeros((4, 13), np.float32)),
                meta=dict(kind="s7-open-loop-states", learner=rec["learner"],
                          actor_npz_sha256=rec["snapshots"][str(step)]["actor"]["sha256"],
                          made_up=rec["data"]["made_up"]))
        H.write_json(d / "s7.json", dict(learner=rec["learner"], rows=4, passed=True, s7_passed=True,
                                         closed_loop=None if rec["data"]["made_up"] else dict(passed=True)))

    def driver(self, *, mode="dry-run", checks=None, claim0=(), accepted=(), s9=True, stop=None, resumed=(),
               s7_stopped=None):
        """The stage driver's files (stages.Driver): status.json, state.json, driver.log (accepted S5 lines), an active
        STOP.json and stop reports moved aside by --resume-after-stop (STOP.json.<time> with STOP.json.sha256.<time>);
        resumed: [(stamp, body, sidecar ok)]; s7_stopped: version 3's state.json entries (left out when None, as an
        older driver's state)."""
        d = self.root / "driver"
        d.mkdir(parents=True, exist_ok=True)
        H.write_json(d / "status.json", dict(schema=E.STATUS_SCHEMA, mode=mode, state="finished", phase="finished",
                                             updated_utc="2026-10-16T01:00:00+00:00"))
        if checks is None:
            checks = {}
            for L in C.LEARNERS:
                checks.update({f"S2/{L}": dict(passed=True), f"S8/{L}": dict(passed=True),
                               f"S4/{L}": dict(passed=True), f"S4/medium-expert/seed0/{L}": dict(passed=True),
                               f"S10/{L}/standard": dict(passed=True, equal=True, readout=C.LEARNERS.index(L) in (0, 2)),
                               f"S10/{L}/iw": dict(passed=True, equal=True, readout=C.LEARNERS.index(L) in (0, 2))})
            checks.update({"S6": dict(passed=True), "S6b/medium-expert/seed0": dict(passed=True)})
        sig = E.env_signature(dict(environment=dict(import_record=FAKE_IMPORT_RECORD)))
        state = dict(schema=E.STATE_SCHEMA, decisions=dict(concurrency=1), checks=checks,
                     claim0_failures=list(claim0), card_walls={}, lanes_done=[],
                     s9_reference=dict(signature=sig, source="made-up") if s9 else None)
        if s7_stopped is not None:
            state["s7_stopped"] = list(s7_stopped)
        H.write_json(d / "state.json", state)
        lines = ["[2026-10-16T00:00:00+00:00] driver starts (dry run): made-up"]
        lines += [f"[2026-10-16T00:30:00+00:00] {rid}: S5 accepted by the operator as the setup's outcome (--accept-s5)"
                  for rid in accepted]
        (d / "driver.log").write_text("\n".join(lines) + "\n", encoding="utf8")
        if stop:
            H.write_json(d / "STOP.json", dict(stop, schema=E.STOP_SCHEMA, mode=mode))
        for stamp, body, good in resumed:
            p = d / f"STOP.json.{stamp}"
            p.write_text(json.dumps(dict(body, schema=E.STOP_SCHEMA, mode=mode)), encoding="utf8")
            digest = E.sha256_file(p) if good else sha("another stop report")
            (d / f"STOP.json.sha256.{stamp}").write_text(f"{digest}  STOP.json\n", encoding="utf8")

    def study(self, q="medium-expert", gains=None, driver=True):
        """Every stage-2 run of one quality: the 60 claim runs, 20 room references and the sham-alone runs of choice
        19 (seed 0, TD3+BC and IQL), and the driver's files."""
        gains = gains or {}
        for L in C.LEARNERS:
            g = gains.get(L, NULL1)
            for i, s in enumerate(C.SEEDS):
                base = BASE[i]
                self.run(q, L, "alone", s, strong=base)
                self.run(q, L, "standard", s, strong=base + g[i],
                         f1=({"big_waves": 9} if L == "td3_bc" else None))
                self.run(q, L, "iw", s, strong=base + g[i] + NULL2[i])
                self.run(q, L, "room", s, strong=base - 1.0)
            if E.CHOICE19[L]:
                self.run(q, L, "sham_alone", 0, strong=BASE[0] + 0.25)
        if driver:
            self.driver()


def fresh(name):
    d = WORK / name
    if d.exists():
        shutil.rmtree(d)
    d.mkdir(parents=True)
    return d


def made_up_build(root, **kw):
    kw.setdefault("reference", False)
    return E.build(guard.MADE_UP, frozen=frozen(), runs_root=root, anchors=ANCHORS, data_root=None, **kw)


def fields_are_read_or_explained(test, listed):
    """Every field of every run read is "read" or "not recorded: <why>" with a reason."""
    for entry in listed:
        test.assertTrue(entry["fields"], msg=entry["run_id"])
        for k, v in entry["fields"].items():
            ok = v == "read" or (isinstance(v, str) and v.startswith("not recorded: ") and len(v) > 24)
            test.assertTrue(ok, msg=f"{entry['run_id']} {k}: {v!r}")


class Export(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.dir = fresh("export_case")
        cls.fake = Fake(cls.dir / "runs", frozen())
        cls.fake.study("medium-expert", gains=dict(td3_bc=GAIN))
        cls.ref = fresh("reference_case") / "source"
        RefFake(cls.ref).write()
        cls.results = E.build(guard.MADE_UP, frozen=frozen(), runs_root=cls.dir / "runs", anchors=ANCHORS,
                              reference_dest=cls.ref, data_root=None)

    def test_end_to_end(self):
        """A made-up build over runner-layout folders: medium-expert judged (TD3+BC's Claim 1 passed with the selection
        label, the others shelved), medium pending; nothing excluded; every table number equals its scores file's
        J_mean and carries the record's and scores file's SHA-256; every field read or explained; the driver's checks
        read; results.json round-trips with its sidecar."""
        r = self.results
        self.assertIs(r["made_up"], True)
        self.assertEqual((r["excluded"], r["unscored"], r["halted_runs"]), ([], [], []))
        self.assertEqual(len(r["runs"]), 60 + 20 + 2)
        fields_are_read_or_explained(self, r["runs"])
        me = r["qualities"]["medium-expert"]
        self.assertEqual(me["claims"]["td3_bc"]["1"]["badge"], C.PASSED)
        self.assertIn(C.SELECTION_LABEL, me["claims"]["td3_bc"]["1"]["labels"])
        self.assertEqual({L: me["claims"][L]["1"]["badge"] for L in C.LEARNERS if L != "td3_bc"},
                         {L: C.SHELVED for L in C.LEARNERS if L != "td3_bc"})
        self.assertEqual(me["claims"]["td3_bc"]["5"]["badge"], C.FLAGGED)       # 9 of 200 starts against 4 = +2.5
        self.assertEqual(me["claims"]["td3_bc"]["5"]["by_setup"], dict(standard=C.FLAGGED, iw=C.PASSED))
        self.assertEqual({me["claims"][L]["0"]["badge"] for L in C.LEARNERS}, {C.PASSED})
        self.assertEqual({me["claims"][L]["4a"]["badge"] for L in C.LEARNERS}, {C.PASSED})   # 92% own, 86% standard
        self.assertEqual({me["claims"][L]["4b"]["badge"] for L in C.LEARNERS}, {C.PASSED})
        md = r["qualities"]["medium"]
        self.assertEqual({md["claims"][L][c]["badge"] for L in C.LEARNERS for c in ("0", "1", "2")}, {C.PENDING})
        self.assertEqual(len(me["table"]), 12)
        for row in me["table"]:
            for p in row["strong"]["per_seed"]:
                folder = self.fake.folder("medium-expert", row["learner"], row["setup"], p["seed"])
                arrays, man = H.read(folder / "lanes" / f"scores-step{FINAL}.npz")
                self.assertEqual(p["v"], man["meta"]["summaries"]["scored"]["strong"]["J_mean"])
                self.assertEqual(p["v"], float(arrays["scored/strong/J"].mean()))
                self.assertEqual(p["source"]["record"], E.sha256_file(folder / "record.json"))
                self.assertEqual(p["source"]["scores"], man["npz_sha256"])
            self.assertIsNotNone(row["strong"]["mean"])
        integ = r["integrity"]
        self.assertEqual({k: integ[k]["status"] for k in ("S1", "S2", "S3", "S4", "S6", "S6b", "S7", "S8", "S9",
                                                           "S11")},
                         {k: "passed" for k in ("S1", "S2", "S3", "S4", "S6", "S6b", "S7", "S8", "S9", "S11")})
        self.assertEqual(integ["S10"]["status"], "passed")
        self.assertEqual(integ["S0"]["status"], "not recorded")
        self.assertIn("SUMMARY.json", integ["S0"]["why"])
        self.assertEqual(integ["driver"]["choice19"], dict(td3_bc=True, cql=False, iql=True, rebrac=False))
        self.assertEqual(set(integ["driver"]["choice19_source"].values()), {"driver"})
        out = WORK / "export_case" / "results.json"
        digest = E.write(r, out)
        self.assertEqual(digest, E.sha256_file(out))
        self.assertEqual(H.read_json(out), json.loads(json.dumps(H.plain(r))))

    def test_expanded_view(self):
        """The expanded view: per-seed metrics on all 7 banks with the records list, the 50% snapshot, compute time,
        band readouts and coverage (with its pairs), the pointers (IW against alone, mild mix, room, sham-alone drift,
        Claim 2's pointer, paired standard errors) and CQL's tanh-of-mean readout; IQL's Q and V identity from
        hashes.final.parts; ReBRAC's substituted next commands from its records."""
        ex = self.results["qualities"]["medium-expert"]["expanded"]
        td = ex["td3_bc"]
        std = td["setups"]["standard"]
        self.assertEqual([x["seed"] for x in std["records"]], list(C.SEEDS))
        self.assertEqual(std["records"][0]["machine"], dict(host="made-up-host", device="cpu", taskset_cores=[0]))
        self.assertEqual(set(std["metrics"]), set(C.BANKS))
        self.assertEqual(len(std["metrics"]["big_kicks"]["regret_mean"]["per_seed"]), 5)
        self.assertEqual(std["half"]["step"], HALF)
        self.assertIsNotNone(std["half"]["strong"]["J"]["mean"])
        self.assertEqual(std["compute"]["card_seconds"]["per_seed"], [90.0, 91.0, 92.0, 93.0, 94.0])
        self.assertAlmostEqual(std["compute"]["steady_ms_per_outer_step"]["mean"], 0.6, places=12)
        self.assertEqual(std["band"]["certified_refreshes"]["per_seed"], [18.0] * 5)
        self.assertEqual(std["coverage"][0]["standard"]["outside"], 0.86)
        self.assertEqual(std["coverage"][0]["pairs"]["recorded"]["standard"]["outside"]["coverage"], 0.86)
        self.assertEqual(set(std["coverage"][0]["pairs_not_recorded"]), {"actor", "visited"})
        iw = td["setups"]["iw"]
        self.assertEqual(iw["coverage"][1]["own_arm"], "iw")
        self.assertEqual(iw["coverage"][1]["own"]["outside"], 0.92)
        self.assertEqual(td["pointers"]["iw_minus_alone"]["strong"]["label"], C.POINTER_LABEL)
        self.assertTrue(td["pointers"]["room_minus_alone"]["strong"]["summary"]["complete"])
        self.assertAlmostEqual(td["pointers"]["room_minus_alone"]["strong"]["summary"]["mean"], -1.0, delta=0.2)
        self.assertEqual(len(td["pointers"]["paired_se"]["standard_minus_alone"]["strong"]), 5)
        self.assertEqual(td["pointers"]["paired_se"]["standard_minus_alone"]["strong"][0]["n"], 200)
        self.assertIsNotNone(td["pointers"]["sham_minus_alone_seed0"]["values"]["strong"])
        self.assertIsNone(ex["cql"]["pointers"]["sham_minus_alone_seed0"]["values"])        # choice 19: no CQL sham
        self.assertTrue(td["pointers"]["half"]["standard_minus_alone"]["strong"]["summary"]["complete"])
        cp = td["pointers"]["claim2_pointer"]
        self.assertAlmostEqual(cp["mean_change"], 0.04, places=12)
        self.assertAlmostEqual(cp["mean_extra"], 0.3, places=12)
        delta = self.results["qualities"]["medium-expert"]["claims"]["td3_bc"]["1"]["delta"]["summary"]["mean"]
        self.assertAlmostEqual(delta, float(np.mean(GAIN)), delta=0.1)
        self.assertEqual(cp["delta_standard_minus_alone"], delta)
        self.assertAlmostEqual(cp["value"], cp["mean_change"] * delta / cp["mean_extra"], places=12)
        self.assertEqual(cp["definition"], E.POINTER_DEFINITION)
        iql_ptr = ex["iql"]["setups"]["iw"]["band"]["pointer"][0]
        self.assertAlmostEqual(iql_ptr["unifloral_mean"], UNIFLORAL_WEIGHT, places=12)
        self.assertAlmostEqual(iql_ptr["extra"], 0.3, places=12)
        self.assertIn("tanh_mean", ex["cql"]["pointers"])
        self.assertIn("tanh_mean", ex["cql"]["setups"]["alone"])
        self.assertEqual([x["identical"] for x in ex["iql"]["readouts"]["q_and_v_identical_across_runs"]],
                         [True] * 5)                      # the made-up records give the three runs one Q and V hash
        aux = {("iql", s, sd): dict(hashes=dict(final=dict(parts={p: sha(f"{p} {sd}") for p in E.IQL_SHARED_PARTS})),
                                    record_sha256=sha(f"{s} {sd}"))
               for s in C.CLAIM_SETUPS for sd in C.SEEDS}
        aux[("iql", "iw", 1)]["hashes"]["final"]["parts"]["value"] = sha("another value network")
        aux[("iql", "alone", 2)]["hashes"]["final"]["parts"]["dual_q"] = sha("drifted at rounding level")
        del aux[("iql", "alone", 3)]
        aux[("iql", "standard", 4)]["stop"] = dict(rule="S5", step=5)
        got = E._iql_identity(aux)
        self.assertEqual([x["identical"] for x in got], [True, False, False, None, None])
        self.assertEqual(got[2]["standard_and_iw_identical"], True)
        self.assertEqual(got[1]["standard_and_iw_identical"], False)
        sub = ex["rebrac"]["readouts"]["substituted_next_commands"]
        self.assertEqual([x["source"] for x in sub], ["record.data.next_action_substituted"] * 5)
        self.assertAlmostEqual(sub[2]["train"], 0.0798, places=12)

    def test_reference_row_made_up(self):
        """The reference row from a made-up copy of the weighting test's records: its deltas equal RESULTS.json's
        (computed here with scipy as make_results does), and every value carries its file's SHA-256."""
        ref = self.results["reference_row"]
        self.assertEqual(ref["name"], E.REFERENCE_NAME)
        self.assertIs(ref["made_up"], True)
        me = ref["qualities"]["medium-expert"]
        self.assertTrue(all(me["results_json_cross_check"].values()))
        self.assertEqual(len(me["results_json_cross_check"]), 3 * 3)
        d = me["deltas"]["standard_minus_alone"]["strong"]
        self.assertAlmostEqual(d["mean"], 6.928, places=9)
        self.assertAlmostEqual(d["p_one_sided"], 0.0128, delta=5e-5)
        v = me["setups"]["alone"]["strong"]["J"]["per_seed"][0]
        self.assertEqual(v["sha256"], E.sha256_file(self.ref / DC.dest_rel(DC.hosts_eval_rel("medium-expert", 0))))

    def test_writeups_and_reading_order_in_results(self):
        me = self.results["qualities"]["medium-expert"]
        self.assertEqual([s["step"] for s in me["reading_order"][1:]], list(C.READING_ORDER))
        self.assertEqual(me["reading_order"][0]["status"]["S2"], "passed")
        self.assertTrue(me["writeups"])
        self.assertTrue(all(w["badge"] in (C.SHELVED, C.DID_NOT_LEARN) for w in me["writeups"]))
        self.assertEqual(me["stage3_order"], [dict(learner="td3_bc", setups=["standard_flattened"], for_claim="1")])


class ExportRefusals(unittest.TestCase):
    def setUp(self):
        self.dir = fresh("refusals_case")
        self.fake = Fake(self.dir / "runs", frozen())

    def build(self, **kw):
        return made_up_build(self.dir / "runs", **kw)

    def reasons(self, r):
        return {e["path"].split("/runs/stage2/made-up-medium-expert/")[-1]: e["reason"] for e in r["excluded"]}

    def test_excluded_records(self):
        """Left out, with the reason: another fingerprint (S1), another code manifest (S1), scores by other code (S1),
        a real record in a made-up build, a scores file of another actor, a scores file whose actor names another run,
        a tampered scores file (even with a fresh sidecar), a summary that differs from its per-start arrays, a
        coverage file of another record, a coverage.npz that is not the one coverage.json names, a record in the wrong
        folder, an identity hash that does not match; a complete run without final scores is 'unscored'."""
        q = "medium-expert"
        f = self.fake
        f.run(q, "td3_bc", "alone", 0, fingerprint=sha("other code"))
        bad_code = scoring_code(frozen())
        bad_code["experiments/shift_hosts/evaluate.py"] = sha("edited evaluate.py")
        f.run(q, "td3_bc", "alone", 1, code=bad_code)
        f.run(q, "td3_bc", "alone", 2, kind="run", made_up=False, label=q)
        f.run(q, "td3_bc", "alone", 3, actor_mismatch=True)
        f.run(q, "td3_bc", "alone", 4)
        p = f.folder(q, "td3_bc", "alone", 4) / "lanes" / f"scores-step{FINAL}.npz"
        blob = bytearray(p.read_bytes())
        blob[len(blob) // 2] ^= 1
        p.write_bytes(bytes(blob))
        H.write_sidecar(p)
        f.run(q, "cql", "alone", 0)
        p = f.folder(q, "cql", "alone", 0) / "lanes" / f"scores-step{FINAL}.npz"
        arrays, man = H.read(p)
        meta = man["meta"]
        meta["summaries"]["scored"]["strong"]["J_mean"] += 1.0
        H.write(p, arrays, meta=meta)
        f.run(q, "cql", "standard", 0, coverage_record=sha("another record"))
        f.run(q, "cql", "alone", 1, scored=False)
        f.run(q, "cql", "alone", 2)
        rec = H.read_json(f.folder(q, "cql", "alone", 2) / "record.json")
        rec["seed"] = 3
        H.write_json(f.folder(q, "cql", "alone", 2) / "record.json", rec)
        f.run(q, "cql", "alone", 3, manifest_sha=sha("another code manifest"))
        f.run(q, "cql", "alone", 4, meta_run_id=f.run_id(q, "cql", "alone", 3))
        f.run(q, "iql", "iw", 0)
        rec = H.read_json(f.folder(q, "iql", "iw", 0) / "record.json")
        rec["identity"]["device"] = "gpu"
        H.write_json(f.folder(q, "iql", "iw", 0) / "record.json", rec)
        f.run(q, "iql", "standard", 1)
        d = f.folder(q, "iql", "standard", 1) / "lanes" / f"coverage-step{LAST_REFRESH}"
        body = H.read_json(d / "coverage.json")                    # it names the first coverage.npz's SHA-256
        H.write(d / "coverage.npz", {"recorded/y": np.ones(8, np.float32)}, meta=dict(note="another coverage.npz"))
        H.write_json(d / "coverage.json", body)
        r = self.build()
        why = self.reasons(r)
        self.assertIn("stop rule S1", why["seed0/td3_bc-alone"])
        self.assertIn("other code than the frozen code", why["seed1/td3_bc-alone"])
        self.assertIn("kind is 'run'", why["seed2/td3_bc-alone"])
        self.assertIn("another actor file", why["seed3/td3_bc-alone"])
        self.assertIn(f"scores-step{FINAL}.npz", why["seed4/td3_bc-alone"])
        self.assertIn("differs from its per-start arrays", why["seed0/cql-alone"])
        self.assertIn("another run record", why["seed0/cql-standard"])
        self.assertIn("another stage, quality, learner, setup or seed", why["seed2/cql-alone"])
        self.assertIn("code manifest", why["seed3/cql-alone"])
        self.assertIn("names another run, identity or step", why["seed4/cql-alone"])
        self.assertIn("identity_sha256", why["seed0/iql-iw"])
        self.assertIn("coverage.npz differs", why["seed1/iql-standard"])
        self.assertEqual([(u["learner"], u["setup"], u["seed"]) for u in r["unscored"]], [("cql", "alone", 1)])
        self.assertEqual(r["integrity"]["S1"]["status"], "failed")
        self.assertEqual(len(why), 12)
        self.assertEqual(r["qualities"][q]["claims"]["td3_bc"]["0"]["badge"], C.PENDING)

    def test_folders_that_are_not_runs(self):
        """A folder of another stage-2 setup (stage 1's iw_ones), a sham alone on seed 1 (choice 19 runs seed 0 only)
        and a seed folder without a number are excluded with the reason."""
        q = "medium-expert"
        self.fake.run(q, "td3_bc", "alone", 0)
        for name in ("seed0/td3_bc-iw_ones", "seed1/iql-sham_alone", "seedX/td3_bc-alone"):
            (self.dir / "runs" / "stage2" / "made-up-medium-expert" / name).mkdir(parents=True)
        r = self.build()
        why = self.reasons(r)
        self.assertEqual(set(why), {"seed0/td3_bc-iw_ones", "seed1/iql-sham_alone", "seedX/td3_bc-alone"})
        self.assertTrue(all("not a learner-setup folder" in v for v in why.values()))
        self.assertEqual(len(r["runs"]), 1)

    def test_stops(self):
        """S5 on the learner alone or its room reference is an outcome (Claim 0: did not learn); S5 on a BCA setup the
        operator accepted (the driver's log line) counts as not finite (p = 1); any other stop halts: integrity says
        stopped and the run is missing, so the claims stay pending; the driver's stop reports are read."""
        q = "medium-expert"
        f = self.fake
        f.study(q, gains=dict(td3_bc=GAIN), driver=False)
        f.run(q, "cql", "alone", 2, stop=dict(rule="S5", reason="made-up non-finite loss", step=1234))
        f.run(q, "iql", "room", 0, stop=dict(rule="S5", reason="made-up", step=9))
        f.run(q, "rebrac", "standard", 1, stop=dict(rule="S5", reason="made-up", step=99))
        accepted = f.run_id(q, "rebrac", "standard", 1)
        f.driver(accepted=[accepted], claim0=[dict(run=f.run_id(q, "cql", "alone", 2), step=1234, reason="made-up"),
                                              dict(run=f.run_id(q, "iql", "room", 0), step=9, reason="made-up")],
                 resumed=[("20261016T003000", dict(rule="S5", reason=f"{accepted} stopped", run=accepted), True)])
        r = self.build()
        c = r["qualities"][q]["claims"]
        self.assertEqual((c["cql"]["0"]["badge"], c["iql"]["0"]["badge"]), (C.DID_NOT_LEARN, C.DID_NOT_LEARN))
        fam = r["qualities"][q]["families"]["1"]
        self.assertEqual((fam["p_values"]["rebrac"], c["rebrac"]["1"]["badge"]), (1.0, C.SHELVED))
        self.assertEqual(c["td3_bc"]["1"]["badge"], C.PASSED)
        integ = r["integrity"]
        self.assertEqual(integ["S5"]["status"], "outcome")
        self.assertEqual(integ["S5"]["driver"]["accepted_by_operator"], [accepted])
        self.assertEqual(integ["S5"]["driver"]["resumed_stops"][0]["verified"], True)
        self.assertEqual(r["halted_runs"], [])
        listed = {x["run_id"]: x for x in r["runs"]}
        self.assertEqual(listed[accepted]["kind"], "outcome")
        stops = [x["stop"] for x in integ["S5"]["export"]["details"]["recorded_as_outcomes"]
                 if (x["learner"], x["setup"], x["seed"]) == ("rebrac", "standard", 1)]
        self.assertEqual(stops[0]["accepted_by_operator"], True)
        fields_are_read_or_explained(self, r["runs"])
        self.assertIn("the run stopped", listed[accepted]["fields"]["scores_final"])
        f.run(q, "td3_bc", "iw", 4, stop=dict(rule="S5", reason="made-up package defect", step=5))
        f.run(q, "rebrac", "alone", 3, stop=dict(rule="S9", reason="made-up environment change", step=0))
        f.driver(accepted=[accepted], stop=dict(rule="S5", reason="made-up package defect", run=f.run_id(
            q, "td3_bc", "iw", 4)))
        r = self.build()
        self.assertEqual(r["integrity"]["S5"]["status"], "stopped")
        self.assertEqual(r["integrity"]["S5"]["driver"]["stop"]["rule"], "S5")
        self.assertEqual(r["integrity"]["S9"]["status"], "stopped")
        self.assertEqual(len(r["halted_runs"]), 2)
        c = r["qualities"][q]["claims"]
        self.assertEqual({L: c[L]["1"]["badge"] for L in C.LEARNERS},
                         dict(td3_bc=C.PENDING, cql=C.NOT_JUDGED, iql=C.NOT_JUDGED, rebrac=C.PENDING))
        self.assertEqual(c["rebrac"]["0"]["badge"], C.PENDING)

    def test_s4_shared_start(self):
        """S4 from the records: a setup whose initial parameters differ from its learner's other setups, a key carry
        that differs before a stop, or other training rows fail the check; a stopped run's carries after its stop are
        not compared."""
        q = "medium-expert"
        for s in ("alone", "standard", "iw"):
            self.fake.run(q, "iql", s, 0, initial=sha("other start") if s == "iw" else None)
            self.fake.run(q, "cql", s, 1, key_shift=" other batches" if s == "standard" else None)
            self.fake.run(q, "rebrac", s, 2, train=sha("other rows") if s == "alone" else None)
            self.fake.run(q, "td3_bc", s, 3, key_shift=" after a stop" if s == "iw" else None,
                          stop=dict(rule="S5", reason="made-up", step=2_000) if s == "iw" else None)
        self.fake.driver()
        r = self.build()
        self.assertEqual(r["integrity"]["S4"]["status"], "failed")
        differ = {(d["learner"], d["what"]) for d in r["integrity"]["S4"]["export"]["details"]["differ"]}
        self.assertEqual(differ, {("iql", "initial parameters"), ("cql", "key carry 2500"),
                                  ("cql", "key carry 50000"), ("cql", "key carry 100000"),
                                  ("rebrac", "training rows")})

    def test_driver_files(self):
        """The driver's files: a failed stage-1 check fails its rule; S10 under choice 19 with unequal states is a
        readout; choice 19 is read from the driver's S10 checks; an S9 failure, a moved stop report whose sidecar does
        not match, and a real driver's files in a made-up build (not read at all)."""
        q = "medium-expert"
        self.fake.run(q, "td3_bc", "alone", 0)
        checks = {"S2/td3_bc": dict(passed=False), "S10/td3_bc/standard": dict(passed=True, readout=True, equal=False),
                  "S10/cql/standard": dict(passed=True, readout=False, equal=True), "S6b/medium-expert/seed1":
                  dict(passed=False), "S9/stage2/made-up-medium-expert/seed0/td3_bc-alone": dict(passed=False)}
        self.fake.driver(checks=checks, resumed=[("20261016T010000", dict(rule="S6", reason="made-up"), False)])
        r = self.build()
        integ = r["integrity"]
        self.assertEqual({k: integ[k]["status"] for k in ("S2", "S6b", "S9", "S10")},
                         dict(S2="failed", S6b="failed", S9="failed", S10="readout"))
        self.assertEqual(integ["driver"]["choice19"], dict(td3_bc=True, cql=False, iql=True, rebrac=False))
        self.assertEqual(integ["driver"]["choice19_source"],
                         dict(td3_bc="driver", cql="driver", iql="export.CHOICE19", rebrac="export.CHOICE19"))
        self.assertEqual(integ["driver"]["resumed_stops"][0]["verified"], False)      # its rule is not trusted
        self.assertEqual(integ["S6"]["status"], "not recorded")
        self.assertEqual([x["resumed_stop"]["file"] for x in integ["unknown_stops"]], ["STOP.json.20261016T010000"])
        self.fake.driver(mode="real")
        r = self.build()
        self.assertFalse(r["integrity"]["driver"]["read"])
        self.assertIn("not a dry-run driver's status", r["integrity"]["driver"]["why"])
        self.assertEqual(r["integrity"]["S2"]["status"], "not recorded")
        self.assertEqual(len(r["runs"]), 1)

    def test_s7_stopped_scopes(self):
        """Version 3: the driver's state.json s7_stopped reaches the claims. ReBRAC stopped after its seed-0 standard
        actor failed S7: its claims read not judged with p = 1, its runs enter no table or expanded-view number and are
        listed as excluded (still listed under runs), S7 reads as an outcome, the caveat and the S7 write-up are in
        results.json, and medium is untouched. A failed S7 check of a scope the driver did not record fails S7."""
        q = "medium-expert"
        f = self.fake
        f.study(q, gains=dict(td3_bc=GAIN), driver=False)
        d = f.folder(q, "rebrac", "standard", 0) / "lanes" / f"s7-step{FINAL}"
        H.write_json(d / "s7.json", dict(learner="rebrac", rows=4, passed=True, s7_passed=False,
                                         closed_loop=dict(passed=False)))
        f.driver()                                              # an older driver: no s7_stopped key
        r = self.build()
        self.assertEqual(r["integrity"]["S7"]["status"], "failed")
        self.assertEqual(r["qualities"][q]["claims"]["rebrac"]["0"]["badge"], C.PASSED)
        self.assertEqual(r["built"]["driver"]["s7_stopped"], [])
        entry = s7_entry("rebrac", "learner", "standard", stage="stage2")
        self.assertEqual(entry["run"], f.run_id(q, "rebrac", "standard", 0))
        checks = {"S7/medium-expert/seed0/rebrac/standard": dict(passed=False, file=entry["file"])}
        f.driver(s7_stopped=[entry], checks=checks)
        r = self.build()
        me = r["qualities"][q]
        self.assertEqual({c: me["claims"]["rebrac"][c]["badge"] for c in C.READING_ORDER},
                         {c: C.NOT_JUDGED for c in C.READING_ORDER})
        self.assertEqual((me["families"]["1"]["p_values"]["rebrac"], me["families"]["2"]["p_values"]["rebrac"]),
                         (1.0, 1.0))
        self.assertEqual(me["claims"]["td3_bc"]["1"]["badge"], C.PASSED)
        self.assertEqual(me["claims"]["td3_bc"]["1"]["family"]["bar"], 0.0125)
        rows = [x for x in me["table"] if x["learner"] == "rebrac"]
        self.assertEqual({x["status"] for x in rows}, {C.S7_STOPPED})
        self.assertTrue(all(x["strong"]["mean"] is None for x in rows))
        self.assertEqual(me["expanded"]["rebrac"]["setups"], {})
        self.assertIsNone(me["expanded"]["rebrac"]["pointers"]["room_minus_alone"]["strong"]["summary"]["mean"])
        self.assertIsNotNone(me["expanded"]["td3_bc"]["setups"]["standard"]["metrics"]["strong"]["J_mean"]["mean"])
        self.assertEqual(len(me["s7_stopped"]["excluded"]), 20)                 # alone, standard, IW and room x 5
        self.assertEqual(r["unscored"], [])
        self.assertEqual(me["s7_stopped"]["scopes"]["rebrac"]["runs"], [entry["run"]])
        self.assertIn("version 3", me["s7_stopped"]["rule"])
        self.assertEqual(len(r["runs"]), 60 + 20 + 2)                          # still read and listed
        self.assertEqual(r["integrity"]["S7"]["status"], "outcome")
        self.assertEqual(r["integrity"]["S7"]["driver"]["status"], "outcome")
        stopped = [x for x in r["integrity"]["S7"]["export"]["details"]["results"] if x["stopped_scope"]]
        self.assertEqual([x["run_id"] for x in stopped], [entry["run"]])
        self.assertEqual(r["built"]["driver"]["s7_stopped"], [entry])
        self.assertEqual(r["integrity"]["driver"]["s7_stopped"], [entry])
        self.assertTrue(any("version 3" in c for c in r["caveats"]))
        self.assertIn(("rebrac", "S7"), {(w["learner"], w["claim"]) for w in me["writeups"]})
        self.assertEqual({r["qualities"]["medium"]["claims"][L]["0"]["badge"] for L in C.LEARNERS}, {C.PENDING})
        # a run of the stopped scope that trained but was never scored (the driver adds it no lanes) is listed as
        # excluded, tagged under unscored and runs, and its compute counts in the totals; an unscored run of a live
        # scope is still plain unscored
        card = me["totals"]["card_seconds"]
        f.run(q, "rebrac", "iw", 1, scored=False)
        f.run(q, "cql", "alone", 1, scored=False)
        r = self.build()
        me = r["qualities"][q]
        self.assertEqual(len(me["s7_stopped"]["excluded"]), 20)
        never = [x for x in me["s7_stopped"]["excluded"] if x["why"] == E.S7_NEVER_SCORED]
        self.assertEqual([(x["learner"], x["setup"], x["seed"]) for x in never], [("rebrac", "iw", 1)])
        self.assertEqual({(u["learner"], u["setup"], u["seed"]): u.get("s7_stopped") for u in r["unscored"]},
                         {("rebrac", "iw", 1): E.S7_NEVER_SCORED, ("cql", "alone", 1): None})
        listed = {(x["learner"], x["setup"], x["seed"]): x for x in r["runs"] if x["quality"] == q}
        self.assertEqual(listed[("rebrac", "iw", 1)]["s7_stopped"], E.S7_NEVER_SCORED)
        self.assertNotIn("s7_stopped", listed[("cql", "alone", 1)])
        self.assertAlmostEqual(me["totals"]["card_seconds"], card, places=6)    # still timed, scored or not
        self.assertEqual(me["claims"]["cql"]["0"]["badge"], C.PENDING)
        f.run(q, "cql", "alone", 1)
        f.run(q, "rebrac", "iw", 1)
        # a recorded scope that does not cover the failure (TD3+BC's IW setup) leaves S7 failed
        f.driver(s7_stopped=[s7_entry("td3_bc", "iw", "iw", stage="stage2")], checks=checks)
        r = self.build()
        self.assertEqual(r["integrity"]["S7"]["status"], "failed")
        self.assertEqual(r["integrity"]["S7"]["driver"]["failed_without_a_stopped_scope"], list(checks))
        me = r["qualities"][q]
        self.assertEqual((me["claims"]["td3_bc"]["1"]["badge"], me["claims"]["td3_bc"]["2"]["badge"]),
                         (C.PASSED, C.NOT_JUDGED))
        self.assertEqual(me["families"]["2"]["p_values"]["td3_bc"], 1.0)
        self.assertEqual(me["expanded"]["td3_bc"]["setups"].keys(), {"alone", "standard", "room", "sham_alone"})

    def test_real_record_rules(self):
        """read_run under a (stand-in) real identity: a real record of the plan's schedule scored under the approval
        is read; a shortened schedule, a made-up record, or scores under the made-up token are refused. No approval is
        needed: these are made-up files read directly by read_run, not a build."""
        fz = frozen(made_up=False)
        f = Fake(self.dir / "real", fz)
        q = "medium"
        f.run(q, "rebrac", "iw", 1)
        got = E.read_run(f.folder(q, "rebrac", "iw", 1), q, "rebrac", "iw", 1, fz)
        self.assertEqual(got["kind"], "outcome")
        self.assertEqual(got["aux"]["run_id"], f"stage2/{q}/seed1/rebrac-iw")
        self.assertEqual(got["run"]["pointer"], dict(change=0.04, extra=got["run"]["pointer"]["extra"]))
        self.assertAlmostEqual(got["run"]["pointer"]["extra"], 0.3, places=12)
        f.run(q, "rebrac", "iw", 2, schedule=DRY_SCHEDULE)
        with self.assertRaises(E.Excluded) as ctx:
            E.read_run(f.folder(q, "rebrac", "iw", 2), q, "rebrac", "iw", 2, fz)
        self.assertIn("not the plan's schedule", str(ctx.exception))
        f.run(q, "rebrac", "iw", 3, made_up=True, kind="stage0-test")
        with self.assertRaises(E.Excluded) as ctx:
            E.read_run(f.folder(q, "rebrac", "iw", 3), q, "rebrac", "iw", 3, fz)
        self.assertIn("kind is 'stage0-test'", str(ctx.exception))
        f.run(q, "rebrac", "iw", 4)
        p = f.folder(q, "rebrac", "iw", 4) / "lanes" / f"scores-step{FINAL}.npz"
        arrays, man = H.read(p)
        H.write(p, arrays, meta=dict(man["meta"], permission=dict(kind="made-up")))
        with self.assertRaises(E.Excluded) as ctx:
            E.read_run(f.folder(q, "rebrac", "iw", 4), q, "rebrac", "iw", 4, fz)
        self.assertIn("permission or made-up marks", str(ctx.exception))

    def test_pointer_definition(self):
        """Claim 2's pointer from readouts.dose_change: change = change_vs_unweighted, extra = the mean standard dose
        minus 1 (IQL: minus refreshes[-1].train_rows.mean_unifloral_weight); a dose change of another refresh or of
        another kind of value is refused."""
        rec = Fake(self.dir, frozen()).record("medium", "td3_bc", "iw", 0, pointer=(0.07, 0.175))
        got, why = E.dose_change_pointer(rec, "td3_bc", rec["identity"]["schedule"])
        self.assertIsNone(why)
        self.assertEqual(got["change"], 0.07)
        self.assertAlmostEqual(got["extra"], 0.175, places=12)
        self.assertAlmostEqual(got["standard_mean"], 1.175, places=12)
        rec = Fake(self.dir, frozen()).record("medium", "iql", "iw", 0, pointer=(-0.0025, -0.12))
        got, _ = E.dose_change_pointer(rec, "iql", rec["identity"]["schedule"])
        self.assertAlmostEqual(got["extra"], -0.12, places=12)
        self.assertEqual(got["unifloral_mean"], UNIFLORAL_WEIGHT)
        with self.assertRaises(E.Excluded):
            E.dose_change_pointer(rec, "td3_bc", rec["identity"]["schedule"])          # "actor weight" is not a dose
        moved = dict(rec, refreshes=rec["refreshes"][:-1])
        with self.assertRaises(E.Excluded):
            E.dose_change_pointer(moved, "iql", rec["identity"]["schedule"])
        none = dict(rec, readouts=None)
        self.assertEqual(E.dose_change_pointer(none, "iql", rec["identity"]["schedule"]),
                         (None, "readouts.dose_change is not in the record"))

    def test_permission_and_writes(self):
        """A made-up build needs a made-up identity; other tokens are refused; a gate.Approval that does not verify
        is refused (no frozen pre-specification exists); made-up results are written only under the stage-0 folder
        and real ones never there."""
        with self.assertRaises(gate.ApprovalError):
            E.build(guard.MADE_UP, runs_root=self.dir / "runs", anchors=ANCHORS, reference=False)
        with self.assertRaises(gate.ApprovalError):
            E.build(guard.MADE_UP, frozen=frozen(made_up=False), runs_root=self.dir, anchors=ANCHORS, reference=False)
        with self.assertRaises(gate.ApprovalError):
            E.build("yes", runs_root=self.dir, anchors=ANCHORS, reference=False)
        fake_approval = gate.Approval("x", "a" * 64, "b" * 64, "c" * 64, "d" * 40, gate.APPROVAL_EMAIL, "someone")
        with self.assertRaises(gate.ApprovalError):
            E.build(fake_approval, runs_root=self.dir, anchors=ANCHORS, reference=False)
        with self.assertRaises(gate.ApprovalError):
            E.reference_row(fake_approval)
        r = self.build()
        outside = REPO / "runs" / "shift_hosts" / "results_made_up_test.json"
        for path in (E.OUT, outside):
            with self.assertRaises(ValueError):
                E.write(r, path)
            self.assertFalse(Path(path).exists())
        real = dict(r, made_up=False)
        with self.assertRaises(ValueError):
            E.write(real, self.dir / "results.json")
        self.assertFalse((self.dir / "results.json").exists())

    def test_main_refuses_today(self):
        """main() asks the gate first and refuses today (no frozen pre-specification); nothing is written."""
        out = self.dir / "should_not_exist.json"
        with self.assertRaises(gate.ApprovalError) as ctx:
            E.main(["--out", str(out)])
        self.assertIn("pre-specification", str(ctx.exception))
        self.assertFalse(out.exists())
        self.assertFalse(E.OUT.exists())

    def test_made_up_never_reads_the_real_copies(self):
        """A made-up reference row pointed at the real copy folder refuses at the copy manifest's made-up mark, before
        any record is read; one without a folder is refused outright."""
        with self.assertRaises(gate.ApprovalError):
            E.reference_row(guard.MADE_UP)
        if (DC.DEST / "copy_manifest.json").exists():
            with self.assertRaises(gate.ApprovalError):
                E.reference_row(guard.MADE_UP, dest=DC.DEST)


# ------------------------------------------------------------------------------------------------ the originals

class Stub:
    """Just enough of a stages.Driver for its own methods card_job, lanes_for, _s7_lane and log."""

    def __init__(self, cfg, folder):
        import collections
        import types

        from experiments.shift_hosts import stages as ST

        self.cfg, self.dir, self.events = cfg, Path(folder), collections.deque(maxlen=30)
        self.made_up_file = lambda quality, rep: Path(folder) / "data" / quality / f"seed{rep}.npz"
        self._s7_lane = types.MethodType(ST.Driver._s7_lane, self)


class Originals(unittest.TestCase):
    """The rules export.py restates (it imports neither runner.py nor stages.py) against the originals."""

    @classmethod
    def setUpClass(cls):
        from experiments.shift_hosts import band_readout as BR
        from experiments.shift_hosts import runner as RN
        from experiments.shift_hosts import stages as ST

        cls.BR, cls.RN, cls.ST = BR, RN, ST
        cls.dir = fresh("originals")

    def test_constants(self):
        RN, ST, BR = self.RN, self.ST, self.BR
        self.assertEqual((E.RUN_SCHEMA, E.STATE_SCHEMA, E.STATUS_SCHEMA, E.STOP_SCHEMA, E.COVERAGE_SCHEMA),
                         (RN.SCHEMA, ST.STATE_SCHEMA, ST.STATUS_SCHEMA, ST.STOP_SCHEMA, BR.SCHEMA))
        self.assertEqual(E.COVERAGE_PAIRS, BR.GROUPS)
        self.assertEqual(E.UNIFLORAL_SETUPS, RN.UNIFLORAL_SETUPS)
        self.assertEqual(E.CHOICE19, ST.CHOICE19)
        self.assertEqual(E.S7_SETUPS, ST.STAGE2_SETUPS)
        self.assertTrue(set(E.STAGE_SETUPS["stage2"]) <= set(RN.SETUPS))
        self.assertEqual((C.LEARNERS, C.QUALITIES, C.SEEDS), (RN.LEARNERS, RN.QUALITIES, ST.SEEDS))
        self.assertEqual(E.RUNS_ROOT, RN.RUNS)
        self.assertEqual(E.DATA, RN.DATA)
        text = (E.PACKAGE / "unifloral_iql_bca.py").read_text(encoding="utf8")
        self.assertIn("return agent_state.dual_q, agent_state.dual_q_target, agent_state.value", text)
        self.assertEqual(E.IQL_SHARED_PARTS, ("dual_q", "dual_q_target", "value"))

    def test_hashes_ids_and_schedules(self):
        """canonical_sha256, run_id and the run folder, the plan's schedule (runner.make_schedule) and the 50% step,
        the dry run's schedule, and env_signature equal the originals'."""
        RN, ST = self.RN, self.ST
        for obj in (dict(b=[1, 2.5, (3, 4)], a=dict(z=np.float32(0.5), y=None)), PLAN_SCHEDULE,
                    Fake(self.dir, frozen()).record("medium", "cql", "iw", 3)["identity"]):
            self.assertEqual(E.canonical_sha256(obj), RN.canonical_sha256(obj))
        self.assertEqual(E.run_id("stage2", "made-up-medium", 3, "cql", "iw"),
                         RN.run_id("stage2", "made-up-medium", 3, "cql", "iw"))
        self.assertEqual(RN.run_dir(self.dir, "stage2", "medium", 1, "iql", "room"),
                         self.dir / E.run_id("stage2", "medium", 1, "iql", "room"))
        plan = RN.make_schedule()
        self.assertEqual(H.plain(plan._asdict()), E.PLAN_SCHEDULE)
        self.assertTrue(RN.is_plan_schedule(plan))
        self.assertEqual((E.FINAL, E.HALF, E.half_step(E.PLAN_SCHEDULE)), (100_000, 50_000, 50_000))
        d = ST.DRY_SCHEDULE
        dry = RN.make_schedule(num_updates=d["num_updates"], eval_interval=d["eval_interval"], warmup=d["warmup"],
                               every=d["every"])
        self.assertEqual(H.plain(dry._asdict()), DRY_SCHEDULE)
        self.assertTrue(E._schedule_ok(DRY_SCHEDULE) and E._schedule_ok(E.PLAN_SCHEDULE))
        rec = dict(environment=dict(import_record=dict(FAKE_IMPORT_RECORD, card=dict(name="made-up card",
                                                                                     driver_version="1.0",
                                                                                     memory_total="1 MiB"))))
        self.assertEqual(E.env_signature(rec), ST.env_signature(rec))
        self.assertEqual(E.env_signature(dict(environment=dict(import_record=FAKE_IMPORT_RECORD))),
                         ST.env_signature(dict(environment=dict(import_record=FAKE_IMPORT_RECORD))))

    def test_driver_layout(self):
        """The stage driver's own card_job and lanes_for name the folders and files export.py reads: the run id and
        folder (made-up and real labels), lanes/scores-step<50%>.npz and scores-step<final>.npz, the coverage lane's
        folder at the last refresh (band_readout writes coverage.json there), S7's folder at the final step; and its
        log line for an accepted S5 is what export.ACCEPTED_S5_LINE reads."""
        ST = self.ST
        for dry, q in ((True, "medium"), (False, "medium-expert")):
            argv = ["run", "--root", str(self.dir / "root")] + (["--dry-run"] if dry else [])
            cfg = ST.config_from(ST.parser().parse_args(argv))
            stub = Stub(cfg, self.dir)
            job = ST.Driver.card_job(stub, q, 2, "cql", "iw", "stage2", cfg.steps, "seed2")
            want = E.run_id("stage2", E.quality_label(q, dry), 2, "cql", "iw")
            self.assertEqual(job.label, want)
            self.assertEqual(job.run_dir, self.dir / "root" / want)
        fz = frozen(made_up=False)
        rec = Fake(self.dir, fz).record("medium-expert", "iql", "iw", 0)
        job = ST.Driver.card_job(stub, "medium-expert", 0, "iql", "iw", "stage2", cfg.steps, "seed0")
        lanes = ST.Driver.lanes_for(stub, job, rec)
        outs = set()
        for lane in lanes:
            for argv, _ in lane.commands:
                if "--out" in argv:
                    outs.add(Path(argv[argv.index("--out") + 1]).relative_to(job.run_dir).as_posix())
        self.assertTrue({f"lanes/scores-step{HALF}.npz", f"lanes/scores-step{FINAL}.npz",
                         f"lanes/coverage-step{LAST_REFRESH}", f"lanes/s7-step{FINAL}/s7.json"} <= outs, outs)
        self.assertIn("S5 accepted by the operator as the setup's outcome (--accept-s5)",
                      (E.PACKAGE / "stages.py").read_text(encoding="utf8"))
        rid = E.run_id("stage2", "made-up-medium-expert", 1, "td3_bc", "standard")
        log_dir = fresh("originals_log")
        stub = Stub(cfg, log_dir)
        ST.Driver.log(stub, f"{rid}: S5 accepted by the operator as the setup's outcome (--accept-s5)")
        line = (log_dir / "driver.log").read_text(encoding="utf8").splitlines()[-1]
        m = E.ACCEPTED_S5_LINE.match(line)
        self.assertIsNotNone(m, line)
        self.assertEqual(m.group(1), rid)


# ------------------------------------------------------------------------------------------------ the driver's dry run

DRY_RUNS = dict(fresh=T.STAGE0 / AREA / "dryrun", stages=T.STAGE0 / "runner" / "test_stages" / "dryrun",
                stop_s5=T.STAGE0 / "runner" / "test_stages" / "stop_s5")


def copy_dry_run(src, dst):
    """The files export.py reads from a dry-run root, copied unchanged: driver/ (without its per-job logs and ledger)
    and each run folder's record.json and lanes/ (S7 and coverage); snapshots and actor files stay where they are."""
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src / "driver", dst / "driver", ignore=shutil.ignore_patterns("logs", "ledger.json*", "nohup.out"))
    for rec in sorted(src.glob("stage[123]/*/seed*/*/record.json")):
        rel_dir = rec.parent.relative_to(src)
        (dst / rel_dir).mkdir(parents=True)
        for n in ("record.json", "record.json.sha256"):
            shutil.copy2(rec.parent / n, dst / rel_dir / n)
        if (rec.parent / "lanes").is_dir():
            shutil.copytree(rec.parent / "lanes", dst / rel_dir / "lanes")


def dry_means(L, setup, seed):
    """Made-up returns for the dry run's scores files: TD3+BC's standard BCA gains GAIN, every other difference is
    small; no-shift returns sit far above Claim 0's bars."""
    i = int(seed) % 5
    base = BASE[i]
    std = base + (GAIN[i] if L == "td3_bc" else NULL1[i])
    strong = dict(alone=base, standard=std, iw=std + NULL2[i], room=base - 1.0, sham_alone=base + 0.25)[setup]
    return strong, -100.0 + 0.1 * i, -150.0


def add_dry_scores(root, n=1000):
    """Made-up scores files where the driver's scoring lane writes them (lanes/scores-step<T>.npz at the final and
    50% steps of every complete stage-2 run), bound to the actor files the runner wrote (their manifests are read,
    never their parameters). Returns {path: J_mean on the strong bank}."""
    written = {}
    code = scoring_code(frozen())
    for rp in sorted(root.glob("stage2/*/seed*/*/record.json")):
        rec = H.read_json(rp)
        if rec["outcome"] != "complete":
            continue
        sched = rec["identity"]["schedule"]
        strong, none, mild = dry_means(rec["learner"], rec["setup"], rec["seed"])
        for step in (sched["steps"], E.half_step(sched)):
            actor = rec["snapshots"][str(step)]["actor"]
            apath = Path(actor["path"])
            apath = apath if apath.is_absolute() else REPO / apath
            am = H.read_manifest(apath)
            assert am["npz_sha256"] == actor["sha256"], apath
            shift = 0.0 if step == sched["steps"] else -0.5
            path = rp.parent / "lanes" / f"scores-step{step}.npz"
            write_scores(path, rec["learner"], actor["sha256"], am["meta"], bank_means(strong + shift, none, mild),
                         n=n, permission=dict(kind="made-up"), code=code,
                         seed_tag=int(hashlib.sha256(rec["run_id"].encode()).hexdigest()[:8], 16) + step)
            written[path] = H.read_manifest(path)["meta"]["summaries"]["scored"]["strong"]["J_mean"]
    return written


class DryRun(unittest.TestCase):
    """export.build over the stage driver's own dry-run output (made-up arrays, tiny step counts; runner.py,
    stages.py and band_readout.py wrote every file), with made-up scores files placed where the scoring lanes would
    put them (the dry run never scores a learned policy). The fresh dry run (runs/shift_hosts/stage0/claims/dryrun:
    medium-expert, seeds 0-4, every learner) is required; test_stages' two dry runs are read too when present."""

    @classmethod
    def setUpClass(cls):
        cls.results, cls.roots, cls.scores = {}, {}, {}
        for name, src in DRY_RUNS.items():
            if not (src / "driver" / "status.json").exists():
                continue
            dst = fresh(f"dryrun_{name}") / "root"
            copy_dry_run(src, dst)
            cls.scores[name] = add_dry_scores(dst)
            first = H.read_json(next(iter(sorted(dst.glob("stage2/*/seed*/*/record.json")))))
            fz = E.Frozen(None, None, gate.code_manifest(), made_up=True,
                          code_manifest_sha256=first["code"]["code_manifest_sha256"])
            cls.roots[name] = dst
            cls.results[name] = E.build(guard.MADE_UP, frozen=fz, runs_root=dst, anchors=ANCHORS, data_root=None,
                                        reference=False)

    def need(self, name):
        if name not in self.results:
            if name == "fresh":
                self.fail(f"the fresh dry run {DRY_RUNS['fresh']} is missing (stages.py run --dry-run --root "
                          f"runs/shift_hosts/stage0/claims/dryrun --qualities medium-expert --seeds 0 1 2 3 4)")
            self.skipTest(f"{DRY_RUNS[name]} is missing")
        return self.results[name], self.roots[name]

    def check_runs(self, r, root):
        """Every run folder read, nothing excluded or halted, every field read or explained, and the fields each kind
        of run must have read."""
        self.assertEqual(r["excluded"], [])
        folders = sorted(root.glob("stage2/*/seed*/*/record.json"))
        self.assertEqual(len(r["runs"]), len(folders))
        fields_are_read_or_explained(self, r["runs"])
        for x in r["runs"]:
            fl, setup, seed = x["fields"], x["setup"], x["seed"]
            stopped = x["kind"] != "outcome" or "the run stopped" in fl["scores_final"]
            must = ["record", "timings", "environment", "hashes.initial", "hashes.key_carry", "hashes.final"]
            if not stopped:
                must += ["hashes.params_10000", "scores_final", "scores_half"]
                if setup in ("standard", "iw"):
                    must += ["refreshes", "coverage"]
                if setup == "iw":
                    must.append("dose_change")
                if seed == 0 and setup in E.S7_SETUPS:
                    must.append("s7")
            self.assertEqual({k: fl[k] for k in must}, {k: "read" for k in must}, msg=x["run_id"])

    def check_table(self, r, quality, scores, seeds):
        """The 12-row table: each row's per-seed returns are its scores files' J_mean bit for bit, with the record's
        and the scores file's SHA-256."""
        q = r["qualities"][quality]
        self.assertEqual([(row["learner"], row["setup"]) for row in q["table"]],
                         [(L, s) for L in C.LEARNERS for s in C.CLAIM_SETUPS])
        by_sha = {H.read_manifest(p)["npz_sha256"]: v for p, v in scores.items()}
        for row in q["table"]:
            per = row["strong"]["per_seed"]
            self.assertEqual([p["seed"] for p in per], list(seeds), msg=(row["learner"], row["setup"]))
            for p in per:
                if p["v"] is None:
                    continue
                self.assertEqual(p["v"], by_sha[p["source"]["scores"]])
                self.assertEqual(len(p["source"]["record"]), 64)

    def test_fresh_dry_run_all_seeds(self):
        """The fresh dry run (medium-expert, seeds 0-4): all 82 stage-2 runs read with every field; the driver's
        files read (dry-run mode, choice 19 from its S10 checks, S2, S4, S6, S6b, S8 and S10 from its state); S1, S3,
        S4, S7, S9 and S11 checked here and passed; the 12-row table filled with 5 seeds per row and a mean; Claims 0,
        4a, 4b, 1 and 2 judged; Claim 2's pointer from each IW record's dose change; IQL's Q and V identical across its
        three runs on every seed (the processor); coverage from band_readout's own files."""
        r, root = self.need("fresh")
        self.check_runs(r, root)
        self.assertEqual(len(r["runs"]), 60 + 20 + 2)
        self.assertEqual((r["halted_runs"], r["unscored"]), ([], []))
        self.check_table(r, "medium-expert", self.scores["fresh"], C.SEEDS)
        integ = r["integrity"]
        self.assertTrue(integ["driver"]["read"])
        self.assertEqual(integ["driver"]["mode"], "dry-run")
        self.assertEqual(set(integ["driver"]["choice19_source"].values()), {"driver"})
        status = {k: integ[k]["status"] for k in C.STOP_RULES}
        for rule in ("S1", "S2", "S3", "S4", "S6", "S6b", "S7", "S8", "S9", "S11"):
            self.assertEqual(status[rule], "passed", msg=(rule, integ[rule]))
        self.assertIn(status["S10"], ("passed", "readout"))
        self.assertEqual((status["S0"], status["S5"]), ("not recorded", "not recorded"))
        self.assertIn("why", integ["S5"])
        me = r["qualities"]["medium-expert"]
        for row in me["table"]:
            self.assertIsNotNone(row["strong"]["mean"], msg=(row["learner"], row["setup"]))
            self.assertIsNotNone(row["none"]["mean"])
            if row["setup"] != "alone":
                self.assertIsNotNone(row["delta"]["mean"])
        self.assertEqual({me["claims"][L]["0"]["badge"] for L in C.LEARNERS}, {C.PASSED})
        for cid in ("4a", "4b"):
            self.assertTrue({me["claims"][L][cid]["badge"] for L in C.LEARNERS} <= {C.PASSED, C.SHELVED}, msg=cid)
        self.assertTrue(me["families"]["1"]["judged"])
        self.assertTrue(me["families"]["2"]["judged"])
        self.assertEqual(me["claims"]["td3_bc"]["1"]["badge"], C.PASSED)
        ex = me["expanded"]
        for L in C.LEARNERS:
            iw = ex[L]["setups"]["iw"]
            for sd, ptr in zip(C.SEEDS, iw["band"]["pointer"]):
                rec = H.read_json(root / "stage2" / "made-up-medium-expert" / f"seed{sd}" / f"{L}-iw" / "record.json")
                dc = rec["readouts"]["dose_change"]
                base = (rec["refreshes"][-1]["train_rows"]["mean_unifloral_weight"] if L == "iql" else 1.0)
                self.assertEqual(ptr["change"], dc["change_vs_unweighted"])
                self.assertEqual(ptr["extra"], (dc["mean"] - dc["change_vs_unweighted"]) - base)
            self.assertIsNotNone(ex[L]["pointers"]["claim2_pointer"]["value"], msg=L)
            cov = [c for c in iw["coverage"]]
            self.assertTrue(all(c is not None and c["own_arm"] == "iw" and c["own"]["outside"] is not None
                                for c in cov), msg=L)
            self.assertTrue(all(set(c["pairs"]) == {"recorded"} for c in cov))
            for s in ("alone", "standard", "iw", "room"):
                self.assertIn(s, ex[L]["setups"])
            self.assertEqual("sham_alone" in ex[L]["setups"], E.CHOICE19[L])
        self.assertEqual([x["identical"] for x in ex["iql"]["readouts"]["q_and_v_identical_across_runs"]],
                         [True] * 5)
        self.assertTrue(all("not recorded" in x["why"] for x in ex["rebrac"]["readouts"]["substituted_next_commands"]))
        self.assertEqual(ex["cql"]["setups"]["alone"]["tanh_mean"]["strong"]["per_seed"][0] is not None, True)
        out = WORK / "dryrun_fresh" / "results.json"
        self.assertEqual(E.write(r, out), E.sha256_file(out))

    def test_stages_dry_run_two_qualities(self):
        """test_stages' dry run (both qualities, seeds 0-1): every run read under the made-up-medium and
        made-up-medium-expert labels; the table holds seeds 0-1 and the claims wait for the other seeds."""
        r, root = self.need("stages")
        self.check_runs(r, root)
        for q in C.QUALITIES:
            self.check_table(r, q, self.scores["stages"], (0, 1))
            self.assertEqual(r["qualities"][q]["claims"]["td3_bc"]["1"]["badge"], C.PENDING)
        self.assertEqual({x["quality"] for x in r["runs"]}, set(C.QUALITIES))

    def test_stop_s5_accepted(self):
        """test_stages' simulated S5 on TD3+BC's standard setup (seed 1), accepted by the operator after two stop
        reports were moved aside: the run is an outcome (not finite), both stop reports are listed and verify against
        their moved sidecars, and nothing halts."""
        r, root = self.need("stop_s5")
        self.check_runs(r, root)
        rid = "stage2/made-up-medium-expert/seed1/td3_bc-standard"
        listed = {x["run_id"]: x for x in r["runs"]}
        self.assertEqual(listed[rid]["kind"], "outcome")
        self.assertEqual(r["halted_runs"], [])
        integ = r["integrity"]
        self.assertEqual(integ["S5"]["status"], "outcome")
        self.assertEqual(integ["driver"]["accepted_s5"], [rid])
        resumed = integ["S5"]["driver"]["resumed_stops"]
        self.assertEqual(len(resumed), 2)
        self.assertTrue(all(x["verified"] and x["rule"] == "S5" for x in resumed))
        row = [x for x in r["qualities"]["medium-expert"]["table"] if (x["learner"], x["setup"]) ==
               ("td3_bc", "standard")][0]
        self.assertEqual([p["v"] is None for p in row["strong"]["per_seed"]], [False, True])
        stop = r["qualities"]["medium-expert"]["expanded"]["td3_bc"]["setups"]["standard"]["records"][1]["stop"]
        self.assertEqual((stop["rule"], stop["kind"], stop["accepted_by_operator"]), ("S5", "outcome", True))


# ---------------------------------------------------------------------------------------------- version 4's supplement

# Version 3's approved bytes of every package file version 4 may not change: the judging and export code, the gate, the
# approval tool and every training and scoring file (their SHA-256 in approval_fe9fb77b0851c75b.json's code manifest,
# equal to `git show 6f9ce2b:<file> | sha256sum`); supplement.code_unchanged checks the rest (tests, external code)
V3_CODE = {f"experiments/shift_hosts/{name}": digest for name, digest in (
    ("__init__.py", "7bca13c60e2625721cbfc582231216e2ac882d3f76abc8e53334b409ff79add1"),
    ("actors_f32.py", "0b8a0d1006d3bbec13aa41157a258a76bc870f39056b155a5947e0a9c7010ba4"),
    ("approve_hosts.py", "1129c254b9487e4efda2a365ac82a784ca6616618dedd7bb51abeac225da9f75"),
    ("band_readout.py", "b59a80f2f90bcb518ba98a8b560c2219ea26fb7316cbfecdd8b940470e7b5f58"),
    ("bca_core.py", "22ad8de8fce2028810a7867f8e969297183dbe15da18632f31beb3d2bc8284a8"),
    ("budget.py", "9990be9c566aa20c82bb45ca4984393f338d463231ee3af8aef093af1a48ffc7"),
    ("claims.py", "b1f7ba7a5206492b9bdcd923d9c5fa8bad739403da0ea152086f4a5cadc096e4"),
    ("data_copy.py", "d4cef6063a9cf6ceea2992eae14ecc3dbe8f513ffa9da9eabe95785bce3213ed"),
    ("evaluate.py", "5e7e42932203dfceebf99b5b17618fab14d1df75f34a1756b00144461e3da5ea"),
    ("export.py", "9ecc3fe09cc76dabcf62a5cf159f82baa9872fac564f076438336c11eba049e2"),
    ("gate.py", "922ac9201a98d115a2333a722d1473331f930a1ac034fc6d2c22dbfc3ef64fef"),
    ("guard.py", "dbc67f11cf8130f36076a519c402a6a9f5a9d348518581ae743c2cbc8408eb00"),
    ("handoff.py", "43a1c59f456a84b5e59c0923ff7eb377b5e8914886dc8eed3794502dcdb24914"),
    ("iw.py", "e076537f8d8b217c1c751269901d57e7e9f82b88d566624930c8bf35c4c26784"),
    ("policies.py", "085d663da118bd85040135571e1882e12fb9a55f76a030e44c12c4979ae27e7f"),
    ("prepare.py", "ed124df60ad76282a6ebbad52877b8bd8f1eb9e851fcae94c3460eaabee81499"),
    ("runner.py", "f4100b1d85001df753e6e9ed3af5b71b3b7b73b81f19052c47c847dc093a12c1"),
    ("scoring_cache.py", "0dbba921ca4d90f26a476a8275cb7d751d8a3ffb8531598efc44dad96bce7bca"),
    ("unifloral_cql_bca.py", "cc0022327f247a308407c87c2bdb42dc87abe68af7082580ea117bd887488819"),
    ("unifloral_import.py", "2cad560d24a1ccfe0fc36fd0e7563d43d9c9cbca620a650615866a6c1f8950a4"),
    ("unifloral_iql_bca.py", "084562a211d32dd5218edd38a0c2991a846d37852ee896207eda50bd33ed2120"),
    ("unifloral_rebrac_bca.py", "f82fab09273debfaa4881a02c3b45103f5440722391ae694d7ecb828d3f46afc"),
    ("unifloral_td3_bc_bca.py", "7fa5e0711c38d15952eedef8a781eea8132a230509206365db83ccba8a1b59cb"))}
# prespec.json's "supplement" block exactly as version 4's build froze it (no later agent changes it)
FROZEN_BLOCK = {
    "version": 4, "root": "runs/shift_hosts/supplement_v4",
    "source": {"fingerprint": "fe9fb77b0851c75b522c337fcb65c2a3d027de71ddfa50dcd6f3affc3bbd1c1c",
               "approval_commit": "6f9ce2b51bf2907328477f4ffa96afff25522559",
               "state_sha256": "e577472ffc68a18079c1cd4ae2f778315d506b6db47a23dad0e5c295fc21df11",
               "ledger_sha256": "25ec5762ab14b1e9e47cf38346cb231908003a904618af385632d4c0a5ca7fbf",
               "results_sha256": "73c3f12e07fac5d026381a7e2c50b01fc681442e3109c5a131e4d69d1d6ba48b"},
    "scopes": [{"quality": "medium-expert", "learner": "rebrac", "scope": "learner"},
               {"quality": "medium-expert", "learner": "iql", "scope": "learner"},
               {"quality": "medium-expert", "learner": "td3_bc", "scope": "iw"},
               {"quality": "medium", "learner": "td3_bc", "scope": "learner"},
               {"quality": "medium", "learner": "iql", "scope": "learner"}],
    "plan": {"medium-expert": {"rebrac": ["alone", "standard", "iw", "room"],
                               "iql": ["sham_alone", "alone", "standard", "iw", "room"],
                               "td3_bc": ["alone", "standard", "iw"]},
             "medium": {"td3_bc": ["sham_alone", "alone", "standard", "iw", "room"],
                        "iql": ["sham_alone", "alone", "standard", "iw", "room"]}},
    "s7": {"seeds": [0, 1, 2, 3, 4], "steps": [50000, 100000], "setups": ["alone", "standard", "iw"]}}
SUPPLEMENT_DRY = T.STAGE0 / "runner" / "test_stages" / "supplement"      # StopTest.test_supplement_dry_run's root
# keys the shown block never holds: a test statistic, a p-value, a bar, a mark, a badge, a Holm entry or a verdict (a
# paired difference's se and one-sided bounds are shown: the owner's decision D5)
SHOWN_FORBIDDEN = {"t", "p", "p_one_sided", "p_values", "bar", "bars", "claim0_bar", "mark", "marks", "badge", "badges",
                   "holm", "holm_adjusted_p", "adjusted", "families", "claims", "table", "verdict", "kill_rule",
                   "window", "interval", "t_quantile_4df", "level"}
AT = "2026-10-21T00:00:00+00:00"
DROP = object()                 # SupplementPlanTest's marker for a key taken out of a block
READOUTS_V4 = {}                # what the version-4 tests report in the result file (extra)


def all_keys(x):
    """Every dictionary key anywhere in x (nested dictionaries and lists)."""
    if isinstance(x, dict):
        for k, v in x.items():
            yield k
            yield from all_keys(v)
    elif isinstance(x, list):
        for v in x:
            yield from all_keys(v)


def made_up_block(scopes, results_sha256):
    """A supplement block in prespec.json's shape for these scopes, carrying a made-up "version 3" results.json of this
    SHA-256 (validate requires version 3's real fingerprint and approval commit; no real file is read)."""
    return dict(version=S.VERSION, root=S.ROOT_REL,
                source=dict(fingerprint=S.V3["fingerprint"], approval_commit=S.V3["approval_commit"],
                            state_sha256=sha("made-up version 3 state.json"),
                            ledger_sha256=sha("made-up version 3 ledger.json"), results_sha256=results_sha256),
                scopes=[dict(e) for e in scopes], plan=S.plan_of(S.scope_map(scopes), E.CHOICE19),
                s7=dict(seeds=list(C.SEEDS), steps=[S.HALF_STEP, S.FINAL_STEP], setups=list(S.S7_SETUPS)))


def v4_entry(learner, scope, setup, run, step, quality="medium-expert"):
    """A version-4 state.json "s7_stopped" entry as stages.Driver.s7_failed writes it (version and actor step added)."""
    return dict(quality=quality, learner=learner, scope=scope, setup=setup, run=run, stage=run.split("/")[0],
                file=f"runs/shift_hosts/supplement_v4/{run}/lanes/s7-step{step}/s7.json", step=step, version=4,
                at_utc=AT)


def strong_of(L, setup, seed):
    """Made-up strong-mix returns as Fake.study gives them (TD3+BC's standard BCA gains GAIN)."""
    base = BASE[seed]
    std = base + (GAIN if L == "td3_bc" else NULL1)[seed]
    return dict(alone=base, standard=std, iw=std + NULL2[seed], room=base - 1.0, sham_alone=base + 0.25)[setup]


def supplement_root(root, plan, failing=(), q="medium-expert"):
    """A made-up supplement root in the stage driver's layout (Fake): every run of the plan ({learner: [setups]}) at
    seeds 0-4 (sham-alone at seed 0) with its scores and coverage files and, as version 4 writes them, S7 lanes on every
    alone, standard and IW run's final and 50% actors (failing: the (learner, setup, seed) whose checks fail). Returns
    (the Fake, the s7_results entries the driver would record)."""
    f = Fake(root, frozen())
    results = []
    for L, setups in plan.items():
        for s in setups:
            for sd in E.SETUP_SEEDS[s]:
                rec = f.run(q, L, s, sd, strong=strong_of(L, s, sd))
                if s not in S.S7_SETUPS:
                    continue
                for step in (HALF, FINAL):
                    folder = f.folder(q, L, s, sd)
                    f.s7(folder, rec, step)
                    path = folder / "lanes" / f"s7-step{step}" / "s7.json"
                    ok = (L, s, sd) not in failing
                    H.write_json(path, dict(learner=L, rows=4, passed=True, s7_passed=ok, closed_loop=None))
                    results.append(dict(label=rec["run_id"] + ("/s7" if step == FINAL else f"/s7-{step}"),
                                        run=rec["run_id"], quality=q, learner=L, setup=s, seed=sd, step=step,
                                        s7_passed=ok, open_loop_passed=True, open_loop_max_abs={}, closed_loop=None,
                                        file=str(path), sha256=E.sha256_file(path), at_utc=AT))
    return f, results


def supplement_driver(f, entries, results):
    """The supplement driver's files: Fake.driver's, with version 4's s7_stopped entries and s7_results."""
    f.driver(s7_stopped=entries)
    path = f.root / "driver" / "state.json"
    H.write_json(path, dict(H.read_json(path), s7_results=list(results)))


def dry_frozen(root):
    """A made-up build's identity for a dry-run root: the code manifest its records were trained under."""
    first = H.read_json(next(iter(sorted(root.glob("stage2/*/seed*/*/record.json")))))
    return E.Frozen(None, None, gate.code_manifest(), made_up=True,
                    code_manifest_sha256=first["code"]["code_manifest_sha256"])


def bind(cfg, folder, *names):
    """Just enough of a stages.Driver for its own methods `names` (no driver files are written)."""
    from experiments.shift_hosts import stages as ST

    stub = types.SimpleNamespace(cfg=cfg, made_up_file=lambda quality, rep: Path(folder) / quality / f"seed{rep}.npz")
    for name in names:
        setattr(stub, name, types.MethodType(getattr(ST.Driver, name), stub))
    return stub


class CodeUnchangedTest(unittest.TestCase):
    """Version 4 may change only supplement.CHANGEABLE (stages.py, test_stages.py, test_claims.py, PRESPEC.md and
    INTERFACES.md) and add supplement.py: every other file of the code manifest, external code included, keeps version
    3's approved bytes. So claims.py, export.py, runner.py, evaluate.py, band_readout.py, gate.py, approve_hosts.py and
    every training and scoring file are version 3's, and no judged number can move through new code."""

    def test_byte_identical_to_version_3(self):
        """V3_CODE equals version 3's approved code manifest (approval_fe9fb77b0851c75b.json) and the bytes on disk;
        supplement.code_unchanged finds no other difference; CHANGEABLE names none of V3_CODE's files; supplement.py
        is the only file added."""
        approval = gate.PACKAGE / S.V3["approval_file"]
        self.assertTrue(approval.is_file(), f"{approval} (version 3's approval, committed at 6f9ce2b) is missing")
        then = json.loads(approval.read_text(encoding="utf8"))
        self.assertEqual(then["fingerprint"], S.V3["fingerprint"])
        now = gate.code_manifest()
        for rel, digest in V3_CODE.items():
            self.assertEqual(then["code_manifest"][rel], digest, rel)       # version 3's approved bytes
            self.assertEqual(gate.sha256_file(gate.REPO / rel), digest, rel)  # are the bytes on disk
            self.assertEqual(now[rel], digest, rel)
        self.assertEqual(S.code_unchanged(now), [])
        self.assertFalse(set(V3_CODE) & set(S.CHANGEABLE))
        self.assertEqual(sorted(set(now) - set(then["code_manifest"])), list(S.ADDED))
        self.assertEqual(sorted(set(then["code_manifest"]) - set(now)), [])
        changed = sorted(p for p, d in then["code_manifest"].items() if now[p] != d)
        self.assertTrue(set(changed) <= set(S.CHANGEABLE), changed)
        READOUTS_V4["changed_against_version_3"] = changed

    def test_a_change_is_found(self):
        """A one-byte change to export.py, a changed external file, an added or a removed file are found; changes
        inside CHANGEABLE and supplement.py's presence or absence are not; version 3's approval must match its sidecar
        and approve version 3's fingerprint."""
        now = gate.code_manifest()
        rel = "experiments/shift_hosts/export.py"
        blob = bytearray((gate.REPO / rel).read_bytes())
        blob[len(blob) // 2] ^= 1
        self.assertEqual(S.code_unchanged(dict(now, **{rel: hashlib.sha256(bytes(blob)).hexdigest()})),
                         [dict(path=rel, change="changed")])
        ext = next(p for p in now if not p.startswith("experiments/shift_hosts/"))
        self.assertEqual(S.code_unchanged(dict(now, **{ext: sha("an edited external file")})),
                         [dict(path=ext, change="changed")])
        new = "experiments/shift_hosts/extra.py"
        self.assertEqual(S.code_unchanged(dict(now, **{new: sha("a new file")})), [dict(path=new, change="added")])
        gone = "experiments/shift_hosts/claims.py"
        self.assertEqual(S.code_unchanged({p: d for p, d in now.items() if p != gone}),
                         [dict(path=gone, change="removed")])
        self.assertEqual(S.code_unchanged(dict(now, **{p: sha("edited " + p) for p in S.CHANGEABLE})), [])
        self.assertEqual(S.code_unchanged({p: d for p, d in now.items() if p not in S.ADDED}), [])
        pkg = fresh("code_unchanged") / "package"
        pkg.mkdir()
        src = gate.PACKAGE / S.V3["approval_file"]
        dup = pkg / src.name
        shutil.copy2(src, dup)
        shutil.copy2(gate.sidecar(src), gate.sidecar(dup))
        self.assertEqual(S.code_unchanged(now, package=pkg), [])
        dup.write_bytes(src.read_bytes() + b" ")                           # it no longer matches its sidecar
        with self.assertRaises(gate.ApprovalError):
            S.code_unchanged(now, package=pkg)
        body = json.loads(src.read_text(encoding="utf8"))
        dup.write_text(json.dumps(dict(body, fingerprint=sha("another version"))), encoding="utf8")
        gate.sidecar(dup).write_text(f"{gate.sha256_file(dup)}  {dup.name}\n", encoding="utf8")
        with self.assertRaises(gate.ApprovalError):                         # an approval of another version
            S.code_unchanged(now, package=pkg)
        dup.unlink()
        with self.assertRaises(gate.ApprovalError):
            S.code_unchanged(now, package=pkg)


class SupplementPlanTest(unittest.TestCase):
    """The frozen supplement block and its plan (supplement.py's module docstring)."""

    def test_frozen_block(self):
        """prespec.json holds exactly the frozen block (json.dumps(indent=1) and a final newline, LF only); it
        validates to its own plan with S7 on every data seed and on the 50,000- and 100,000-step actors; its plan has
        56 stage-2 runs on medium-expert and 42 on medium."""
        self.assertEqual(S.prespec_block(), FROZEN_BLOCK)
        text = (gate.PACKAGE / gate.PRESPEC_NAME).read_bytes().decode("utf8")
        self.assertNotIn("\r", text)
        self.assertEqual(text, json.dumps(json.loads(text), indent=1) + "\n")
        v = S.validate(FROZEN_BLOCK, C.LEARNERS, C.QUALITIES, C.SEEDS, E.CHOICE19)
        self.assertEqual((v["plan"], v["s7_all_seeds"], v["s7_half"]), (FROZEN_BLOCK["plan"], True, True))
        self.assertEqual(v["scopes"], {"medium-expert": dict(rebrac="learner", iql="learner", td3_bc="iw"),
                                       "medium": dict(td3_bc="learner", iql="learner")})
        self.assertEqual({q: len(ids) for q, ids in S.run_ids(v["plan"], C.SEEDS).items()},
                         {"medium-expert": 56, "medium": 42})
        self.assertEqual((S.FINAL_STEP, S.HALF_STEP), (FINAL, HALF))
        self.assertEqual(S.ROOT, gate.REPO / FROZEN_BLOCK["root"])

    def test_scopes_and_counts(self):
        """A learner scope trains 21 runs for a choice-19 learner (TD3+BC, IQL: sham-alone, alone, standard, IW and
        room at seeds 0-4) and 20 for CQL and ReBRAC, an IW scope 15 (IW with alone and standard as comparators);
        scopes_of reads s7_stopped entries as claims.s7_scopes does (a learner scope outranks an IW scope; each
        quality its own)."""
        for L in C.LEARNERS:
            for scope, want in (("learner", 21 if E.CHOICE19[L] else 20), ("iw", 15)):
                plan = S.plan_of({"medium": {L: scope}}, E.CHOICE19)
                self.assertEqual(len(S.run_ids(plan, C.SEEDS)["medium"]), want, (L, scope))
                self.assertEqual(plan["medium"][L], (["sham_alone"] if scope == "learner" and E.CHOICE19[L] else [])
                                 + list(S.SCOPE_SETUPS[scope]))
        e = [s7_entry("rebrac", "iw", "iw"), s7_entry("rebrac", "learner", "standard"), s7_entry("cql", "iw", "iw"),
             s7_entry("iql", "learner", "alone", quality="medium", stage="stage2")]
        got = S.scopes_of(e)
        self.assertEqual(got, {"medium-expert": dict(rebrac="learner", cql="iw"), "medium": dict(iql="learner")})
        for q in C.QUALITIES:
            self.assertEqual(got[q], {L: v["scope"] for L, v in C.s7_scopes(e, q).items()})
        self.assertEqual(S.scopes_of([]), {q: {} for q in C.QUALITIES})
        plan = S.plan_of(got, E.CHOICE19)
        self.assertEqual({q: len(ids) for q, ids in S.run_ids(plan, C.SEEDS).items()},
                         {"medium-expert": 20 + 15, "medium": 21})

    def test_run_ids_follow_the_driver(self):
        """run_ids lists a plan's runs in the stage driver's own order (stages.Driver.plan_units and unit_jobs): the
        sham-alone runs, each seed's runs (learner order, then alone, standard, IW), then the room references."""
        from experiments.shift_hosts import stages as ST

        plan = {"medium": {"iql": ["sham_alone", "alone", "standard", "iw", "room"],
                           "td3_bc": ["alone", "standard", "iw"]}}
        self.assertEqual(S.run_ids(plan, (0, 1), made_up=True)["medium"],
                         ["stage2/made-up-medium/seed0/iql-sham_alone"]
                         + [f"stage2/made-up-medium/seed{k}/{L}-{s}" for k in (0, 1) for L in ("td3_bc", "iql")
                            for s in ("alone", "standard", "iw")]
                         + [f"stage2/made-up-medium/seed{k}/iql-room" for k in (0, 1)])
        self.assertEqual(S.run_ids(plan, (0,))["medium"][0], "stage2/medium/seed0/iql-sham_alone")
        d = fresh("supplement_order")
        pf = d / "plan.json"
        pf.write_text(json.dumps(FROZEN_BLOCK["plan"]), encoding="utf8")
        cfg = ST.config_from(ST.parser().parse_args(["run", "--dry-run", "--root", str(d / "root"),
                                                     "--supplement-plan", str(pf)]))
        stub = bind(cfg, d, "in_plan", "card_job", "plan_units", "unit_jobs")
        made = S.run_ids(FROZEN_BLOCK["plan"], cfg.seeds, cfg.learners, made_up=True)
        for q in C.QUALITIES:
            self.assertEqual([j.label for name, _ in stub.plan_units(q) for j in stub.unit_jobs(q, name)], made[q], q)

    def test_validate_refuses(self):
        """validate refuses a block that is not exactly the frozen shape: other keys, version or root; another
        fingerprint or approval commit; a malformed SHA-256; malformed, unknown or repeated scopes; a plan that is
        not what its scopes give (a run left out or added, a sham-alone run without choice 19, room without alone);
        S7 settings other than seed 0 or every seed, the final or the 50% and final steps, and alone, standard and
        IW; a learner the code does not run."""
        def bad(*changes):
            """FROZEN_BLOCK with each (path, value) set, or that key taken out when value is DROP."""
            b = copy.deepcopy(FROZEN_BLOCK)
            for path, value in changes:
                node = b
                for k in path[:-1]:
                    node = node[k]
                if value is DROP:
                    del node[path[-1]]
                else:
                    node[path[-1]] = value
            return b

        scopes, learner = FROZEN_BLOCK["scopes"], ["sham_alone", "alone", "standard", "iw", "room"]
        cases = dict(
            extra_key=dict(FROZEN_BLOCK, kind="s7-supplement"), missing_key=bad((("s7",), DROP)),
            version=bad((("version",), 3)), root=bad((("root",), "runs/shift_hosts")),
            source_key=bad((("source", "extra"), "x")), fingerprint=bad((("source", "fingerprint"), "0" * 64)),
            commit=bad((("source", "approval_commit"), "bafd86f")), state_sha=bad((("source", "state_sha256"), "x")),
            upper_sha=bad((("source", "results_sha256"), "A" * 64)), no_scopes=bad((("scopes",), [])),
            scope_shape=bad((("scopes",), [dict(quality="medium", learner="iql")])),
            unknown_learner=bad((("scopes",), scopes + [dict(quality="medium", learner="bc", scope="iw")])),
            unknown_scope=bad((("scopes",), [dict(e, scope="setup") for e in scopes])),
            unknown_quality=bad((("scopes",), [dict(e, quality="expert") for e in scopes])),
            repeated=bad((("scopes",), scopes + scopes[:1])),
            run_left_out=bad((("plan", "medium", "iql"), learner[:-1])),
            run_added=bad((("plan", "medium", "cql"), ["alone", "standard", "iw"])),
            sham_without_choice19=bad((("plan", "medium-expert", "rebrac"), learner)),
            room_without_alone=bad((("plan", "medium", "iql"), ["sham_alone", "standard", "iw", "room"])),
            unknown_setup=bad((("plan", "medium", "iql"), learner[:-1] + ["iw_ones"])),
            s7_seeds=bad((("s7", "seeds"), [0, 1])), s7_seed1=bad((("s7", "seeds"), [1])),
            s7_steps=bad((("s7", "steps"), [25000, 100000])), s7_order=bad((("s7", "steps"), [100000, 50000])),
            s7_setups=bad((("s7", "setups"), ["alone", "standard", "iw", "room"])), s7_key=bad((("s7", "room"), True)))
        for name, block in cases.items():
            with self.assertRaises(gate.ApprovalError, msg=name):
                S.validate(block, C.LEARNERS, C.QUALITIES, C.SEEDS, E.CHOICE19)
        with self.assertRaises(gate.ApprovalError):                       # ReBRAC is not a learner of this code
            S.validate(FROZEN_BLOCK, ("td3_bc", "cql", "iql"), C.QUALITIES, C.SEEDS, E.CHOICE19)
        seed0 = bad((("s7", "seeds"), [0]), (("s7", "steps"), [100000]))
        v = S.validate(seed0, C.LEARNERS, C.QUALITIES, C.SEEDS, E.CHOICE19)
        self.assertEqual((v["s7_all_seeds"], v["s7_half"]), (False, False))

    def test_check_sources(self):
        """check_sources (before a real start) refuses version 3's files when one is missing, when its SHA-256 is not
        the block's, when runs/shift_hosts/results_v3.json differs from the committed results.json, and when the
        committed files are not in git HEAD as on disk (made-up files here; nothing real is read)."""
        d = fresh("supplement_sources")
        pkg = d / "package"
        pkg.mkdir()
        with self.assertRaisesRegex(gate.ApprovalError, "missing"):
            S.check_sources(FROZEN_BLOCK, repo=d, package=pkg)
        H.write_json(d / S.SOURCE_FILES["state_sha256"], dict(s7_stopped=[s7_entry("rebrac", "learner", "standard")]))
        H.write_json(d / S.SOURCE_FILES["ledger_sha256"], dict(note="a made-up ledger"))
        H.write_json(d / S.SOURCE_FILES["results_sha256"], dict(made_up=True, qualities={}, caveats=[]))
        with self.assertRaisesRegex(gate.ApprovalError, "not the file the block names"):
            S.check_sources(FROZEN_BLOCK, repo=d, package=pkg)
        block = copy.deepcopy(FROZEN_BLOCK)
        block["source"].update({k: gate.sha256_file(d / rel) for k, rel in S.SOURCE_FILES.items()})
        H.write_json(d / S.RESULTS_V3_PREVIEW, dict(made_up=True, note="another preview"))
        with self.assertRaisesRegex(gate.ApprovalError, "differs from version 3's committed results.json"):
            S.check_sources(block, repo=d, package=pkg)
        (d / S.RESULTS_V3_PREVIEW).unlink()
        with self.assertRaisesRegex(gate.ApprovalError, "committed in git HEAD"):
            S.check_sources(block, repo=d, package=pkg)

    @unittest.skipUnless((gate.REPO / S.SOURCE_FILES["state_sha256"]).is_file(),
                         "version 3's runs/shift_hosts/driver/state.json is not on this machine")
    def test_check_sources_on_version_3s_files(self):
        """Read only, as preflight does before a real start: version 3's driver/state.json, driver/ledger.json and
        committed results.json have the frozen block's SHA-256 values (results_v3.json too, when present), the
        committed results.json and version 3's approval are in git HEAD (approved at 6f9ce2b), and version 3's
        s7_stopped gives exactly the block's scopes; a block naming other scopes is refused. Only SHA-256 values,
        git and state.json's S7 entries are read (no return)."""
        got = S.check_sources(FROZEN_BLOCK)
        self.assertEqual(got["scopes"], S.scope_map(FROZEN_BLOCK["scopes"]))
        self.assertEqual(got["approval"], dict(file=S.V3["approval_file"], commit=S.V3["approval_commit"],
                                               fingerprint=S.V3["fingerprint"]))
        other = copy.deepcopy(FROZEN_BLOCK)
        other["scopes"] = other["scopes"][:-1]
        with self.assertRaisesRegex(gate.ApprovalError, "scopes"):
            S.check_sources(other)



class SupplementBuildTest(unittest.TestCase):
    """supplement.build on made-up folders (Fake): a made-up "version 3" results.json (export.build over a full study
    whose driver recorded ReBRAC's learner scope and TD3+BC's IW setup as stopped by S7) is carried, and the
    supplement's own root holds exactly those scopes' runs at seeds 0-4, with S7 on every alone, standard and IW run's
    final and 50% actors (TD3+BC's IW actors at seed 1 fail, recorded by version-4 entries)."""

    SCOPES = [dict(quality="medium-expert", learner="rebrac", scope="learner"),
              dict(quality="medium-expert", learner="td3_bc", scope="iw")]

    @classmethod
    def setUpClass(cls):
        cls.dir = fresh("supplement_build")
        q = "medium-expert"
        v3 = Fake(cls.dir / "v3", frozen())
        v3.study(q, gains=dict(td3_bc=GAIN), driver=False)
        v3.driver(s7_stopped=[s7_entry("rebrac", "learner", "standard"),
                              s7_entry("td3_bc", "iw", "iw", stage="stage2")])
        cls.carried = cls.dir / "v3_results.json"
        E.write(made_up_build(cls.dir / "v3"), cls.carried)
        cls.v3 = H.read_json(cls.carried)
        cls.block = made_up_block(cls.SCOPES, gate.sha256_file(cls.carried))
        cls.fake, results = supplement_root(cls.dir / "v4", cls.block["plan"][q], failing={("td3_bc", "iw", 1)})
        cls.s7_results = results
        cls.iw1 = cls.fake.run_id(q, "td3_bc", "iw", 1)
        cls.entries = [v4_entry("rebrac", "learner", "standard", "stage1/medium-expert/seed0/rebrac-standard", 15_000),
                       v4_entry("td3_bc", "iw", "iw", cls.iw1, HALF)]
        supplement_driver(cls.fake, cls.entries, results)
        cls.results = cls.build()

    @classmethod
    def build(cls, root=None, block=None, carried=None):
        return S.build(guard.MADE_UP, frozen=frozen(), root=root or cls.dir / "v4", block=block or cls.block,
                       carried_path=carried or cls.carried, v3_root=cls.dir / "v3", anchors=ANCHORS)

    def test_only_the_allowed_keys_are_added(self):
        """Every key of version 3's results.json is canonical-JSON equal in the result; the only additions are
        qualities.<quality>.s7_shown, s7_supplement and one caveat (supplement.CAVEAT, last); the written file reads
        back the same."""
        r, v3 = self.results, self.v3
        self.assertEqual(set(r) - set(v3), {"s7_supplement"})
        for k in v3:
            if k not in ("qualities", "caveats"):
                self.assertEqual(S.canonical(r[k]), S.canonical(v3[k]), k)
        self.assertEqual(r["caveats"], v3["caveats"] + [S.CAVEAT])
        self.assertEqual(set(r["qualities"]), set(v3["qualities"]))
        for q, body in v3["qualities"].items():
            self.assertEqual(set(r["qualities"][q]) - set(body), {"s7_shown"}, q)
            for k in body:
                self.assertEqual(S.canonical(r["qualities"][q][k]), S.canonical(body[k]), (q, k))
        S.assert_judged_unchanged(r, v3)
        out = WORK / "supplement_build" / "results.json"
        self.assertEqual(E.write(r, out), E.sha256_file(out))
        S.assert_judged_unchanged(H.read_json(out), v3)
        self.assertEqual(H.read_json(out)["s7_supplement"]["carried_from"],
                         dict(version=3, path=E.rel(self.carried), sha256=gate.sha256_file(self.carried),
                              fingerprint=S.V3["fingerprint"]))

    def test_judged_keys_cannot_change(self):
        """assert_judged_unchanged refuses any change to a version-3 key (a badge, a p-value, a table number, a key
        taken out), any other added key, a caveat moved or missing, and a carried file that already holds a supplement
        key; the build refuses such a carried file too."""
        r, v3 = self.results, self.v3
        me = "medium-expert"
        drop = next(k for k in sorted(v3) if k not in ("qualities", "caveats"))
        changes = dict(
            badge=lambda x: x["qualities"][me]["claims"]["td3_bc"]["1"].update(badge=C.SHELVED),
            p_value=lambda x: x["qualities"][me]["families"]["1"]["p_values"].update(td3_bc=0.5),
            table=lambda x: x["qualities"][me]["table"][0]["strong"].update(mean=0.0),
            taken_out=lambda x: x.pop(drop),
            top_level=lambda x: x.update(shown_elsewhere=True),
            in_a_quality=lambda x: x["qualities"]["medium"].update(s7_more={}),
            caveat_moved=lambda x: x["caveats"].insert(0, x["caveats"].pop()),
            caveat_missing=lambda x: x["caveats"].pop())
        for name, change in changes.items():
            x = copy.deepcopy(r)
            change(x)
            with self.assertRaises(gate.ApprovalError, msg=name):
                S.assert_judged_unchanged(x, v3)
        twice = dict(copy.deepcopy(v3), s7_supplement=dict(note="an earlier supplement"))
        with self.assertRaises(gate.ApprovalError):
            S.assert_judged_unchanged(dict(r), twice)
        path = WORK / "supplement_build" / "carried_twice.json"
        E.write(twice, path)
        with self.assertRaisesRegex(gate.ApprovalError, "already holds a supplement key"):
            self.build(block=made_up_block(self.SCOPES, gate.sha256_file(path)), carried=path)

    def test_carried_file_must_be_version_3s(self):
        """The build refuses a carried results.json whose SHA-256 is not the block's, one whose sidecar does not
        match, and a block that does not validate; nothing is built."""
        with self.assertRaisesRegex(gate.ApprovalError, "is not version 3's results.json"):
            self.build(block=made_up_block(self.SCOPES, sha("another results.json")))
        bad = WORK / "supplement_build" / "carried_bad.json"
        shutil.copy2(self.carried, bad)
        shutil.copy2(gate.sidecar(self.carried), gate.sidecar(bad))
        bad.write_bytes(self.carried.read_bytes() + b"\n")                 # its sidecar no longer matches
        with self.assertRaisesRegex(gate.ApprovalError, "does not verify"):
            self.build(block=made_up_block(self.SCOPES, gate.sha256_file(bad)), carried=bad)
        with self.assertRaises(gate.ApprovalError):
            self.build(block=dict(self.block, root="runs/shift_hosts"))

    def test_shown_block(self):
        """qualities.medium-expert.s7_shown: label, caveat and post-hoc mark; per scope its setups (a learner scope
        all of them, an IW scope IW only with alone and standard as comparators), per-seed values equal to the scores
        files with their mean and sd, each run's provenance (trained under version 4, R1 equal to version 3's record,
        version 3's exclusion), the paired differences (an IW scope's leave out standard - alone, version 3's judged
        Claim 1), every S7 lane (seeds 0-4, 50% and final actors, verified and bound to its record) and a write-up in
        the user's format; never a t, p-value, bar, mark, badge, Holm entry or verdict (nor in s7_supplement). medium,
        with no scope, says so."""
        q = "medium-expert"
        sh = self.results["qualities"][q]["s7_shown"]
        self.assertEqual((sh["label"], sh["caveat"], sh["post_hoc"], sh["version"]), (S.LABEL, S.CAVEAT, True, 4))
        self.assertEqual(self.results["qualities"]["medium"]["s7_shown"],
                         dict(label=S.LABEL, note="no scope stopped by S7 in this quality"))
        for qq in C.QUALITIES:
            self.assertEqual(set(all_keys(self.results["qualities"][qq]["s7_shown"])) & SHOWN_FORBIDDEN, set(), qq)
        self.assertEqual(set(all_keys(self.results["s7_supplement"])) & {"badge", "badges", "bar", "bars", "holm",
                                                                        "p_one_sided", "verdict", "kill_rule"}, set())
        self.assertEqual(sorted(sh["scopes"]), ["rebrac", "td3_bc"])
        rb, td = sh["scopes"]["rebrac"], sh["scopes"]["td3_bc"]
        self.assertEqual((rb["scope"], td["scope"]), ("learner", "iw"))
        self.assertEqual((rb["setups_trained"], td["setups_trained"]),
                         (["alone", "standard", "iw", "room"], ["alone", "standard", "iw"]))
        self.assertEqual((sorted(rb["setups"]), list(td["setups"])), (["alone", "iw", "room", "standard"], ["iw"]))
        self.assertEqual((rb["comparators"], sorted(td["comparators"])), ({}, ["alone", "standard"]))
        self.assertEqual(sorted({c["comparison"] for c in rb["comparisons"]}),
                         sorted(["standard - alone", "iw - standard", "iw - alone", "room - alone"]))
        self.assertEqual(sorted({c["comparison"] for c in td["comparisons"]}), ["iw - alone", "iw - standard"])
        self.assertEqual(len(rb["comparisons"]), 4 * len(S.PAIRED_BANKS) * 2)
        for c in rb["comparisons"] + td["comparisons"]:
            self.assertEqual(set(c["summary"]), set(S.PAIRED_FIELDS))
            sm = c["summary"]
            if sm["complete"]:          # D5: claims.summary's one-sided 95% bounds, unchanged
                self.assertAlmostEqual(sm["lower"], sm["mean"] - C.T95 * sm["se"], places=9)
                self.assertAlmostEqual(sm["upper"], sm["mean"] + C.T95 * sm["se"], places=9)
            self.assertIn("shown, not judged", c["bounds"])
        for b in rb["setups"].values():     # a setup's own numbers carry no se or bound
            self.assertEqual(set(b["final"]["strong"]["J"]) & {"se", "lower", "upper"}, set())
        std = rb["setups"]["standard"]
        want = [H.read_manifest(self.fake.folder(q, "rebrac", "standard", sd) / "lanes" / f"scores-step{FINAL}.npz")
                ["meta"]["summaries"]["scored"]["strong"]["J_mean"] for sd in C.SEEDS]
        J = std["final"]["strong"]["J"]
        self.assertEqual((J["per_seed"], J["seeds"], J["complete"]), (want, list(C.SEEDS), True))
        self.assertAlmostEqual(J["mean"], float(np.mean(want)), places=9)
        self.assertEqual(std["half"]["strong"]["J"]["seeds"], list(C.SEEDS))
        recs = std["records"]
        self.assertEqual([(x["seed"], x["trained_under"], x["r1"]) for x in recs],
                         [(sd, "version 4", "equal") for sd in C.SEEDS])
        for x in recs:
            folder = self.fake.folder(q, "rebrac", "standard", x["seed"])
            self.assertEqual(x["record_sha256"], E.sha256_file(folder / "record.json"))
            self.assertEqual(x["v3_record_sha256"], E.sha256_file(self.dir / "v3" / folder.relative_to(self.dir / "v4")
                                                                  / "record.json"))
            self.assertEqual(x["v3_excluded"]["why"], "excluded: stopped by S7")
        lanes = sorted((x["setup"], x["seed"], x["step"], x["s7_passed"]) for x in td["s7"])
        self.assertEqual(lanes, sorted(("iw", sd, t, sd != 1) for sd in C.SEEDS for t in (HALF, FINAL)))
        comp = sorted((x["setup"], x["seed"], x["step"], x["s7_passed"]) for x in td["comparator_s7"])
        self.assertEqual(comp, sorted((s, sd, t, True) for s in ("alone", "standard") for sd in C.SEEDS
                                      for t in (HALF, FINAL)))         # version 3's judged runs, counted apart
        self.assertEqual((len(rb["s7"]), rb["comparator_s7"]), (3 * 5 * 2, []))
        self.assertTrue(all(x["verified"] and x["bound_to_record"] for x in td["s7"] + td["comparator_s7"] + rb["s7"]))
        self.assertIn("S7 ran on 10 actors of this scope in the supplement and failed on 2", td["writeup"]["happened"])
        self.assertIn("S7 ran on 20 actors and failed on 0.", td["writeup"]["happened"])
        self.assertIn("S7 ran on 30 actors of this scope in the supplement and failed on 0.", rb["writeup"]["happened"])
        self.assertNotIn("comparators", rb["writeup"]["happened"])
        self.assertEqual(self.results["s7_supplement"]["s7_failures_in_judged_scopes"], [])
        self.assertEqual((td["v4_entries"], rb["v4_entries"]), ([self.entries[1]], [self.entries[0]]))
        self.assertEqual((td["v3"]["scope"], rb["v3"]["scope"]), ("iw", "learner"))
        for L, scope in (("rebrac", rb), ("td3_bc", td)):
            self.assertEqual((scope["label"], scope["writeup"]["status"], scope["writeup"]["claim"]),
                             (S.LABEL, S.LABEL, "S7"))
            md = scope["writeup"]["markdown"]
            pos = [md.index(h) for h in ("**The idea:**", "**What we tested:**", "**What happened:**", "**Why:**")]
            self.assertEqual(pos, sorted(pos))
            self.assertIn(f"results.json:qualities.{q}.s7_shown.scopes.{L}", md)
            self.assertIn("never judged", md)

    def test_supplement_block_and_integrity(self):
        """s7_supplement: the block, plan and S7 settings, the plan check (every planned run recorded, none outside),
        the driver's version-4 entries and s7_results, R1 equal for every run, and integrity: S1 passes and S7 reads
        "outcome", the seed-1 IW failures at both steps covered by a version-4 entry. A record of version 3's
        fingerprint dropped into the supplement's root fails S1 (excluded); without the covering entry S7 fails;
        neither moves a judged key."""
        sup = self.results["s7_supplement"]
        self.assertEqual((sup["version"], sup["label"], sup["post_hoc"], sup["block"]), (4, S.LABEL, True, self.block))
        self.assertEqual(sup["s7_settings"], dict(all_seeds=True, half_step=True))
        self.assertEqual(sup["plan_check"], dict(planned=35, recorded=35, not_recorded=[], outside_the_plan=[]))
        self.assertEqual((sup["s7_stopped"], sup["s7_results"]), (self.entries, self.s7_results))
        self.assertEqual(len(sup["reproduction"]), 35)
        self.assertEqual({e["result"] for e in sup["reproduction"]}, {"equal"})
        integ = sup["integrity"]
        self.assertEqual((integ["S1"]["status"], integ["S7"]["status"]), ("passed", "outcome"))
        bad = [x for x in integ["S7"]["export"]["details"]["results"] if not x["passed"]]
        self.assertEqual(sorted((x["run_id"], x["step"]) for x in bad), [(self.iw1, HALF), (self.iw1, FINAL)])
        self.assertTrue(all(x["stopped_scope"] for x in bad))
        root = fresh("supplement_build_changed") / "v4"
        shutil.copytree(self.dir / "v4", root)
        Fake(root, frozen()).run("medium-expert", "cql", "alone", 0, fingerprint=S.V3["fingerprint"])
        state = H.read_json(root / "driver" / "state.json")
        H.write_json(root / "driver" / "state.json", dict(state, s7_stopped=self.entries[:1]))
        r = self.build(root=root)                       # it builds: no judged key moved
        integ = r["s7_supplement"]["integrity"]
        self.assertEqual((integ["S1"]["status"], integ["S7"]["status"]), ("failed", "failed"))
        why = {e["path"].split("/v4/")[-1]: e["reason"] for e in r["s7_supplement"]["excluded"]}
        self.assertIn("stop rule S1", why["stage2/made-up-medium-expert/seed0/cql-alone"])
        S.assert_judged_unchanged(r, self.v3)

    def test_s7_failures_in_judged_scopes(self):
        """S7 failures on actors of scopes version 3 judged (an IW scope's alone and standard comparators, stage 1's
        actors of a learner version 3 judged, a learner of a quality where version 3 stopped nothing of it) are listed
        (judged_scope_failures) and named in the one appended caveat (caveat_of, which starts with CAVEAT); failures in
        the scopes version 3 stopped, and passing checks, are not. assert_judged_unchanged takes that caveat and
        refuses another, or one that does not start with CAVEAT."""
        me = "medium-expert"
        scopes = {me: {"rebrac": "learner", "td3_bc": "iw"}, "medium": {}}

        def lane(run, learner, setup, ok, q=me):
            return dict(label=f"{run}/s7-{HALF}", run=run, quality=q, learner=learner, setup=setup, seed=None,
                        step=HALF, s7_passed=ok, file=f"{run}/lanes/s7-step{HALF}/s7.json", sha256=sha(run))

        rows = [lane(f"stage2/{me}/seed2/td3_bc-alone", "td3_bc", "alone", False),          # a comparator: judged
                lane(f"stage2/{me}/seed1/td3_bc-iw", "td3_bc", "iw", False),                # the IW scope itself
                lane(f"stage2/{me}/seed3/rebrac-standard", "rebrac", "standard", False),    # a learner scope
                lane(f"stage1/{me}/seed0/td3_bc-iw_repeat", "td3_bc", "iw_repeat", False),  # IW's repeat: the scope
                lane(f"stage1/{me}/seed0/cql-alone", "cql", "alone", False),                # a judged learner
                lane(f"stage2/{me}/seed4/td3_bc-standard", "td3_bc", "standard", True),     # passed
                lane("stage2/medium/seed1/td3_bc-alone", "td3_bc", "alone", False, q="medium")]
        got = S.judged_scope_failures(rows, scopes)
        self.assertEqual([e["run"] for e in got], [rows[0]["run"], rows[4]["run"], rows[6]["run"]])
        self.assertEqual(set(got[0]), {"label", "run", "quality", "learner", "setup", "seed", "step", "file", "sha256"})
        self.assertEqual(S.caveat_of([]), S.CAVEAT)
        cav = S.caveat_of(got)
        self.assertTrue(cav.startswith(S.CAVEAT + " S7 also failed in the supplement on 3 actor(s)"), cav)
        for e in got:
            self.assertIn(f"{e['run']} at {e['step']:,} steps", cav)
        r = copy.deepcopy(self.results)
        r["caveats"][-1] = cav
        S.assert_judged_unchanged(r, self.v3, cav)
        with self.assertRaisesRegex(gate.ApprovalError, "not the last caveat"):
            S.assert_judged_unchanged(r, self.v3)
        other = dict(r, caveats=self.v3["caveats"] + ["another caveat"])
        with self.assertRaisesRegex(gate.ApprovalError, "does not start with"):
            S.assert_judged_unchanged(other, self.v3, "another caveat")

    def test_study_wide_budget(self):
        """s7_supplement.budget: the study-wide totals are version 3's ledger (which already holds versions 1 and 2's
        sessions) plus the supplement's own, each quality's card and processor hours counted once; versions 1 and 2's
        spend is shown under "earlier" as an "of which", never added again."""
        from experiments.shift_hosts import budget as B

        d = fresh("supplement_budget")
        made = {}
        for name, card, proc in (("v3", (0.25, 0.5), (0.4, 0.7)), ("v4", (0.1, 0.0), (0.2, 0.3))):
            led = B.Ledger(d / name / "driver" / "ledger.json", mode="dry-run")
            for q, c, p in zip(C.QUALITIES, card, proc):
                if c:
                    led.add_card(q, f"{name}-{q}", f"session-{name}", 0.0, c * 3600.0)
                led.add_processor(q, f"{name}-{q}", p * 3600.0, "card")
            led.save()
            made[name] = {q: led.totals(q) for q in C.QUALITIES}
        out = S._ledgers(d / "v4", d / "v3", self.block, True)
        self.assertEqual(out["ledgers"], {"version 4": made["v4"], "version 3": made["v3"]})
        for q in C.QUALITIES:
            want = {k: made["v3"][q][k] + made["v4"][q][k] for k in ("card_hours", "processor_hours")}
            self.assertEqual(out["study_wide"]["totals"][q], dict(want, complete=True), q)
        self.assertEqual(out["study_wide"]["earlier"], S.EARLIER_SPEND)
        self.assertIn("not added again", S.EARLIER_SPEND["medium-expert"]["source"])

    def test_main_refuses_today(self):
        """python -m experiments.shift_hosts.supplement export asks the gate first and refuses today (version 4 has
        no terminal approval); nothing is written."""
        out = WORK / "supplement_build" / "should_not_exist.json"
        self.assertEqual(S.main(["export", "--out", str(out)]), 2)
        self.assertFalse(out.exists())
        self.assertFalse(E.OUT.exists())


class SupplementDryRunTest(unittest.TestCase):
    """supplement.build over the stage driver's own supplement dry run (test_stages' StopTest.test_supplement_dry_run:
    version 4's driver wrote every file; S7 failures injected), carrying a made-up "version 3" results.json that
    export.build made from test_stages' plain dry run; both copied, with made-up scores files where the scoring lanes
    write them. Skipped when test_stages' dry runs are not there."""

    @classmethod
    def setUpClass(cls):
        cls.results = None
        if not all((p / "driver" / "status.json").exists() for p in (SUPPLEMENT_DRY, DRY_RUNS["stages"])):
            return
        d = fresh("supplement_dryrun")
        cls.roots = dict(v3=d / "v3", v4=d / "v4")
        for name, src in (("v3", DRY_RUNS["stages"]), ("v4", SUPPLEMENT_DRY)):
            copy_dry_run(src, cls.roots[name])
            add_dry_scores(cls.roots[name])
        cls.carried = d / "v3_results.json"
        E.write(E.build(guard.MADE_UP, frozen=dry_frozen(cls.roots["v3"]), runs_root=cls.roots["v3"], anchors=ANCHORS,
                        data_root=None, reference=False), cls.carried)
        cls.v3 = H.read_json(cls.carried)
        plan = H.read_json(cls.roots["v4"] / "driver" / "state.json")["config"]["plan"]
        scopes = [dict(quality=q, learner=L, scope="learner" if "room" in setups else "iw")
                  for q, by in plan.items() for L, setups in by.items()]
        cls.block = made_up_block(scopes, gate.sha256_file(cls.carried))
        cls.results = S.build(guard.MADE_UP, frozen=dry_frozen(cls.roots["v4"]), root=cls.roots["v4"],
                              block=cls.block, carried_path=cls.carried, v3_root=cls.roots["v3"], anchors=ANCHORS)

    def setUp(self):
        if self.results is None:
            self.skipTest(f"{SUPPLEMENT_DRY} or {DRY_RUNS['stages']} is missing (test_stages has not run)")

    def test_supplement_over_the_drivers_files(self):
        """Version 3's keys unchanged and only the allowed keys added; integrity S1 passes and S7 reads "outcome"
        (every injected failure, stage 1's, seed 0's IQL alone actor at both steps and seed 1's TD3+BC IW actor at
        both steps, covered by a version-4 entry); the driver's 46 S7 results (seeds 0 and 1, steps 20 and 40) are
        carried; the plan check finds every run of seeds 0-1 and nothing outside the plan; R1 is equal for every run
        both dry runs trained (bit for bit); the shown scopes are the plan's, with no forbidden key. Stage 1's TD3+BC
        standard actor belongs to a scope version 3 judged (TD3+BC's IW setup is the stopped scope): its failure is
        listed in s7_failures_in_judged_scopes, named in the appended caveat, and counted among the IW scope's
        comparators, not its own S7 results."""
        r, v3 = self.results, self.v3
        sup = r["s7_supplement"]
        s1 = "stage1/made-up-medium-expert/seed0/td3_bc-standard"
        self.assertEqual([(e["run"], e["step"]) for e in sup["s7_failures_in_judged_scopes"]], [(s1, 30)])
        self.assertEqual((r["caveats"][-1], sup["caveat"]), (S.caveat_of(sup["s7_failures_in_judged_scopes"]),) * 2)
        self.assertIn(f"{s1} at 30 steps", r["caveats"][-1])
        S.assert_judged_unchanged(r, v3, r["caveats"][-1])
        self.assertEqual(set(r) - set(v3), {"s7_supplement"})
        integ = sup["integrity"]
        self.assertEqual((integ["S1"]["status"], integ["S7"]["status"]), ("passed", "outcome"))
        s2 = "stage2/made-up-medium-expert/seed0/iql-alone"
        s3 = "stage2/made-up-medium-expert/seed1/td3_bc-iw"
        bad = [x for x in integ["S7"]["export"]["details"]["results"] if not x["passed"]]
        self.assertEqual(sorted((x["run_id"], x["step"]) for x in bad),
                         sorted([(s1, 30), (s2, 20), (s2, 40), (s3, 20), (s3, 40)]))
        self.assertTrue(all(x["stopped_scope"] for x in bad))
        self.assertTrue(sup["s7_stopped"] and all(e["version"] == 4 for e in sup["s7_stopped"]))
        self.assertEqual(len(sup["s7_results"]), 46)
        self.assertEqual({(x["seed"], x["step"]) for x in sup["s7_results"] if x["run"].startswith("stage2/")},
                         {(0, 20), (0, 40), (1, 20), (1, 40)})
        pc = sup["plan_check"]
        self.assertEqual((pc["recorded"], pc["outside_the_plan"]), (24, []))
        self.assertTrue(all(re.search(r"/seed[234]/", x) for x in pc["not_recorded"]), pc["not_recorded"])
        tally = {}
        for e in sup["reproduction"]:
            tally[e["result"]] = tally.get(e["result"], 0) + 1
        self.assertEqual(tally, dict(equal=10 + 24))
        me = r["qualities"]["medium-expert"]["s7_shown"]["scopes"]
        self.assertEqual({L: s["scope"] for L, s in me.items()}, dict(iql="learner", td3_bc="iw"))
        self.assertEqual({L: s["scope"] for L, s in r["qualities"]["medium"]["s7_shown"]["scopes"].items()},
                         dict(td3_bc="learner"))
        steps = {(x["seed"], x["step"]) for x in me["iql"]["s7"] if x["run"].startswith("stage2/")}
        self.assertEqual(steps, {(0, 20), (0, 40), (1, 20), (1, 40)})
        self.assertTrue(all(x["verified"] and x["bound_to_record"] for x in me["iql"]["s7"]))
        td = me["td3_bc"]
        self.assertEqual({x["setup"] for x in td["s7"]}, {"iw", "iw_repeat"})
        self.assertEqual({x["setup"] for x in td["comparator_s7"]}, {"alone", "standard", "iw_ones"})
        self.assertEqual([x["run"] for x in td["comparator_s7"] if x["s7_passed"] is False], [s1])
        self.assertIn("(s7_supplement.s7_failures_in_judged_scopes)", td["writeup"]["happened"])
        self.assertEqual([x["label"] for x in me["iql"]["s7"] if x["run"].startswith("stage1/")],
                         [x["label"] for x in sup["s7_results"] if x["run"].startswith("stage1/")
                          and x["learner"] == "iql"])
        self.assertEqual(len([x for x in me["iql"]["s7"] if x["run"].startswith("stage1/")]), 5)
        medium = r["qualities"]["medium"]["s7_shown"]["scopes"]["td3_bc"]
        self.assertEqual([x for x in medium["s7"] if x["run"].startswith("stage1/")], [])   # stage 1: medium-expert
        for q in C.QUALITIES:
            self.assertEqual(set(all_keys(r["qualities"][q]["s7_shown"])) & SHOWN_FORBIDDEN, set(), q)
        out = WORK / "supplement_dryrun" / "results.json"
        self.assertEqual(E.write(r, out), E.sha256_file(out))


class ReproductionTest(unittest.TestCase):
    def test_reproduction(self):
        """R1 (supplement.reproduction), a readout that never raises: a supplement record equals version 3's record of
        the same run id when its hashes (initial, key carries, params at 10,000, final), outcome, refreshes without
        timings, snapshot actors' tree SHA-256 and S7 lanes' closed-loop points are equal; identity, code, timings and
        file paths differ by design and are not compared; each difference is named; a run version 3 never trained,
        or whose record names another fingerprint, is "missing" with a note; an unreadable record is "unreadable"."""
        d = fresh("reproduction")
        q = "medium-expert"
        a, b = Fake(d / "v4", frozen()), Fake(d / "v3", frozen())
        for f in (a, b):
            for s in ("alone", "standard", "iw"):
                f.run(q, "cql", s, 0)
            for s in ("alone", "standard"):
                f.run(q, "cql", s, 1)
        a.run(q, "cql", "room", 0)                                          # version 3 never trained it
        rid = lambda s, sd=0: a.run_id(q, "cql", s, sd)

        def edit(f, s, sd, change):
            path = f.folder(q, "cql", s, sd) / "record.json"
            rec = H.read_json(path)
            change(rec)
            H.write_json(path, rec)

        def tree(f, s, digest):
            H.write(f.folder(q, "cql", s, 0) / "actors" / f"step{FINAL}.npz", dict(x=np.zeros(1, np.float32)),
                    meta=dict(actor_tree_sha256=digest))

        edit(a, "alone", 0, lambda r: (r["identity"].update(device="gpu"), r["code"].update(fingerprint=sha("v4")),
                                       r["timings"].update(wall_seconds=1.0), r.update(identity_sha256=sha("x"))))
        edit(a, "standard", 0, lambda r: r["refreshes"][0].update(seconds=9.0, table_seconds=3.0))
        tree(a, "standard", sha("one tree"))
        tree(b, "standard", sha("one tree"))
        edit(a, "iw", 0, lambda r: r["hashes"]["final"].update(key=sha("another key")))
        tree(a, "iw", sha("a tree"))
        tree(b, "iw", sha("another tree"))
        H.write_json(a.folder(q, "cql", "iw", 0) / "lanes" / f"s7-step{FINAL}" / "s7.json",
                     dict(learner="cql", rows=4, passed=True, s7_passed=True, closed_loop={
                         "scored/strong": dict(max_points=0.02, mean_points=0.001, passed=True), "passed": True}))
        edit(a, "alone", 1, lambda r: r.update(outcome="stopped"))
        edit(a, "standard", 1, lambda r: r["refreshes"][0].update(threshold=2.5))
        got = {e["run_id"]: e for e in S.reproduction(d / "v4", d / "v3", None)}
        self.assertEqual(sorted(got), sorted([rid("alone"), rid("standard"), rid("iw"), rid("room"), rid("alone", 1),
                                              rid("standard", 1)]))
        self.assertEqual((got[rid("alone")]["result"], got[rid("standard")]["result"]), ("equal", "equal"))
        self.assertNotIn(f"actor_tree_sha256 at step {FINAL} (no readable actor file on either side)",
                         got[rid("standard")]["not_compared"])
        self.assertEqual((got[rid("iw")]["result"], got[rid("iw")]["differs"]),
                         ("differs", ["hashes.final", f"actor_tree_sha256 at step {FINAL}",
                                      f"S7 closed loop (s7-step{FINAL})"]))
        self.assertEqual(got[rid("alone", 1)]["differs"], ["outcome"])
        self.assertEqual(got[rid("standard", 1)]["differs"], ["refreshes"])
        self.assertEqual((got[rid("room")]["result"], got[rid("room")]["note"]),
                         ("missing", "version 3 has no record of this run"))
        for e in got.values():
            self.assertEqual(e["v4_record_sha256"], E.sha256_file(d / "v4" / e["run_id"] / "record.json"))
        fp = S.reproduction(d / "v4", d / "v3", S.V3["fingerprint"])      # version 3's records here are made-up
        self.assertEqual({e["result"] for e in fp if e["run_id"] != rid("room")}, {"missing"})
        self.assertTrue(all("another fingerprint" in e["note"] for e in fp if e["run_id"] != rid("room")))
        path = a.folder(q, "cql", "iw", 0) / "record.json"
        path.write_text(path.read_text(encoding="utf8") + " ", encoding="utf8")     # it no longer verifies
        broken = {e["run_id"]: e for e in S.reproduction(d / "v4", d / "v3", None)}[rid("iw")]
        self.assertEqual(broken["result"], "unreadable")
        self.assertEqual(S.reproduction(d / "nothing", d / "v3"), [])


class InvarianceTest(unittest.TestCase):
    """Version 4 never passes the supplement's runs to the judging code, and the judging code ignores a stopped
    scope's runs anyway: for version 3's S7 scenarios (S7Scopes.test_learner_scope and test_iw_scope) claims.judge and
    the 12-row table are byte-identical when every number of the stopped scope's runs changes, and when the scope's
    extra scored runs (a supplement's) are added, only s7.excluded lists the added runs."""

    @staticmethod
    def text(judged, excluded=True):
        j = H.plain(judged)
        if not excluded:
            j = dict(j, s7=dict(j["s7"], excluded=None))
        return json.dumps(j, sort_keys=True)

    def check(self, runs, entries, stopped, partial):
        """stopped(learner, setup): whether a run belongs to the stopped scope; partial: version 3's runs of it."""
        base = C.judge("medium-expert", runs, PLAN_ANCHORS["none"], s7_stopped=entries)
        moved = copy.deepcopy(runs)
        n = 0
        for (L, s), seeds in moved.items():
            if stopped(L, s):
                for r in seeds.values():
                    r["J"] = {b: v + 25.0 for b, v in r["J"].items()}
                    r["points"] = {b: v - 7.0 for b, v in r["points"].items()}
                    r["coverage"] = cov(0.10, 0.20)
                    r["pointer"] = dict(change=0.9, extra=0.1) if r.get("pointer") else None
                    n += 1
        self.assertGreater(n, 0)
        got = C.judge("medium-expert", moved, PLAN_ANCHORS["none"], s7_stopped=entries)
        self.assertEqual(self.text(got), self.text(base))
        self.assertEqual(C.phone_table(got, moved), C.phone_table(base, runs))
        few = C.judge("medium-expert", partial, PLAN_ANCHORS["none"], s7_stopped=entries)
        self.assertEqual(self.text(few, excluded=False), self.text(base, excluded=False))
        self.assertEqual(C.phone_table(few, partial), C.phone_table(base, runs))
        self.assertLess(len(few["s7"]["excluded"]), len(base["s7"]["excluded"]))
        return base

    def test_learner_scope(self):
        """ReBRAC stopped by S7 (its standard actor): version 3 trained its seed 0 only, the supplement all five seeds
        and the room references."""
        runs = scenario(d1=dict(td3_bc=GAIN, rebrac=GAIN))
        runs[("rebrac", "room")] = {s: mk({"strong": -201.0}, tag=f"rebrac room {s}") for s in C.SEEDS}
        partial = copy.deepcopy(runs)
        for s in C.CLAIM_SETUPS:
            partial[("rebrac", s)] = {0: partial[("rebrac", s)][0]}
        del partial[("rebrac", "room")]
        base = self.check(runs, [s7_entry("rebrac", "learner", "standard")], lambda L, s: L == "rebrac", partial)
        self.assertEqual(base["learners"]["td3_bc"]["1"]["badge"], C.PASSED)

    def test_iw_scope(self):
        """CQL's IW setup stopped by S7: version 3 trained its IW seed 0 only, the supplement all five seeds."""
        runs = scenario(d1=dict(cql=GAIN), d2=dict(cql=GAIN, td3_bc=GAIN))
        partial = copy.deepcopy(runs)
        partial[("cql", "iw")] = {0: partial[("cql", "iw")][0]}
        base = self.check(runs, [s7_entry("cql", "iw", "iw")], lambda L, s: (L, s) == ("cql", "iw"), partial)
        self.assertEqual(base["learners"]["cql"]["1"]["badge"], C.PASSED)




# ------------------------------------------------------------------------------------------------ the reference copy

def wt_summary(values):
    """make_results.summary with scipy (experiments/shift_iw/coverage.py:318-329 and run.py:1200-1212), as the
    weighting test wrote RESULTS.json."""
    v = np.asarray(values, np.float64)
    m, sd = float(v.mean()), float(v.std(ddof=1))
    se = sd / math.sqrt(len(v))
    t2, t1 = stats.t.ppf(0.975, len(v) - 1), stats.t.ppf(0.95, len(v) - 1)
    p = (0.0 if m > 0 else 1.0) if sd == 0.0 else float(stats.t.sf(m / se, len(v) - 1))
    return dict(per_seed=[float(x) for x in values], n=len(v), mean=m, sd=sd, interval=[m - t2 * se, m + t2 * se],
                lower_one_sided=m - t1 * se, p_one_sided_uncorrected=p)


class RefFake:
    """A made-up copy of the weighting test's reference records in data_copy's layout, with a made-up copy manifest
    and RESULTS.json; every file says made_up true."""

    def __init__(self, dest):
        self.dest = Path(dest)

    def values(self, q, arm, law, seed):
        alone = {"none": -100.0, "mild": -150.0, "strong": -200.0}[law] + 3.0 * seed
        gains = values_with(6.928, 4.463) if law == "strong" else NULL1
        if q == "medium":
            gains = NULL2
        std = alone + gains[seed]
        iw = std + NULL2[seed]
        return dict(alone=alone, room=alone - 1.0, standard=std, iw=iw, iw_single_threshold=iw + 0.1)[arm]

    def write(self):
        entries, inputs, spec = [], {}, sha("made-up spec")

        def put(r, body):
            path = self.dest / DC.dest_rel(r)
            path.parent.mkdir(parents=True, exist_ok=True)
            text = json.dumps(body, sort_keys=True, indent=1)
            path.write_text(text, encoding="utf8")
            d = E.sha256_file(path)
            Path(str(path) + ".sha256").write_text(f"{d}  {path.name}\n", encoding="utf8")
            for x in (r, r + ".sha256"):
                entries.append(dict(rel=x, copy=DC.copy_rel(x), sha256=E.sha256_file(self.dest / DC.dest_rel(x)),
                                    ok=True))
            inputs[str(DC.SOURCE / r)] = d

        qualities = []
        for q in C.QUALITIES:
            for s in C.SEEDS:
                for group, arms in (("hosts", ("alone", "room")),
                                    ("bca", ("standard", "iw", "iw_single_threshold"))):
                    ev = {E.REFERENCE_ARMS[a][1]: {law: dict(J=self.values(q, a, law, s),
                                                             J_se=1.0, points=50.0 + self.values(q, a, law, s) / 10,
                                                             points_se=0.1, finite=True, starts=1000)
                                                   for law in E.REFERENCE_LAWS} for a in arms}
                    body = dict(stage="evaluate", status="finished", spec_sha256=spec, quality=q, seed=s,
                                group=group, evaluations=ev, made_up=True)
                    if group == "hosts":
                        body.update(kind="shift-iw-evaluate")
                        put(DC.hosts_eval_rel(q, s), body)
                    else:
                        body.update(kind="shift-iw-explore-evaluate", label=E.EXPLORE_LABEL, exploratory=True,
                                    smoke=False, missing_arms=[])
                        put(DC.explore_eval_rel(q, s), body)
            comps = {}
            for name, (a, b, key) in E.REFERENCE_DELTAS.items():
                if key is None:
                    continue
                comps[key] = dict(laws={law: dict(J=wt_summary([self.values(q, a, law, s) - self.values(q, b, law, s)
                                                                for s in C.SEEDS]))
                                        for law in E.REFERENCE_LAWS})
            qualities.append(dict(quality=q, comparisons=comps))
        results = dict(kind="shift-iw-explore-results", exploratory=True, made_up=True, inputs_sha256=inputs,
                       qualities=qualities)
        path = self.dest / DC.dest_rel(DC.RESULTS)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(results, indent=1), encoding="utf8")
        entries.append(dict(rel=DC.RESULTS, copy=DC.copy_rel(DC.RESULTS), sha256=E.sha256_file(path), ok=True))
        H.write_json(self.dest / "copy_manifest.json", dict(schema=DC.SCHEMA, all_ok=True, made_up=True,
                                                            entries=entries))


class ReferenceRefusals(unittest.TestCase):
    def setUp(self):
        self.dest = fresh("reference_refusals") / "source"
        RefFake(self.dest).write()

    def test_reads(self):
        ref = E.reference_row(guard.MADE_UP, dest=self.dest)
        self.assertEqual(set(ref["qualities"]), set(C.QUALITIES))
        self.assertEqual(len(ref["sources"]), 1 + 2 * 5 * 2)       # RESULTS.json, then 2 qualities x 5 seeds x 2

    def test_tampered_record_is_refused(self):
        """A changed evaluation record (with a fresh sidecar) no longer matches the copy manifest (stop rule S11)."""
        r = DC.explore_eval_rel("medium", 2)
        p = self.dest / DC.dest_rel(r)
        body = json.loads(p.read_text())
        body["evaluations"]["bca_unweighted"]["strong"]["J"] += 5.0
        p.write_text(json.dumps(body, sort_keys=True, indent=1))
        H.write_sidecar(p)
        with self.assertRaises(DC.CopyError):
            E.reference_row(guard.MADE_UP, dest=self.dest)

    def test_results_json_must_list_the_inputs_and_agree(self):
        """RESULTS.json must name each record with its SHA-256, and its comparison must equal the recomputed one."""
        man = H.read_json(self.dest / "copy_manifest.json")
        rp = self.dest / DC.dest_rel(DC.RESULTS)
        res = json.loads(rp.read_text())

        def rewrite(body):
            rp.write_text(json.dumps(body, indent=1))
            for e in man["entries"]:
                if e["rel"] == DC.RESULTS:
                    e["sha256"] = E.sha256_file(rp)
            H.write_json(self.dest / "copy_manifest.json", man)

        bad = copy.deepcopy(res)
        bad["inputs_sha256"].pop(str(DC.SOURCE / DC.hosts_eval_rel("medium-expert", 0)))
        rewrite(bad)
        with self.assertRaises(ValueError):
            E.reference_row(guard.MADE_UP, dest=self.dest)
        bad = copy.deepcopy(res)
        bad["qualities"][0]["comparisons"]["capped:per_step_minus_unweighted"]["laws"]["strong"]["J"]["mean"] += 1e-6
        rewrite(bad)
        with self.assertRaises(ValueError):
            E.reference_row(guard.MADE_UP, dest=self.dest)
        rewrite(res)
        E.reference_row(guard.MADE_UP, dest=self.dest)


# ------------------------------------------------------------------------------------------------ integration

def made_up_td3_bc_actor():
    """Made-up TD3+BC actor parameters in Unifloral's layout (13-256-256-4), float32, and its normalization."""
    rng = np.random.default_rng([20261016, 7779, 4300])

    def dense(n_in, n_out, scale):
        return dict(kernel=(scale / math.sqrt(n_in) * rng.standard_normal((n_in, n_out))).astype(np.float32),
                    bias=(0.1 * rng.standard_normal(n_out)).astype(np.float32))

    params = {"params": dict(Dense_0=dense(13, 256, math.sqrt(2)), Dense_1=dense(256, 256, math.sqrt(2)),
                             Dense_2=dense(256, 4, 0.5))}
    norm = dict(obs_mean=(0.3 * rng.standard_normal(13)).astype(np.float32),
                obs_std=(0.2 + rng.uniform(size=13)).astype(np.float32))
    return params, norm


@unittest.skipUnless(SCORING_ENV, "needs the scoring environment (numpy 2.2.6, experiments.shift)")
class Integration(unittest.TestCase):
    def test_constants_match_the_package(self):
        """The constants export.py and claims.py restate equal evaluate's, policies' and actors_f32's; evaluate's code
        files are all in the code manifest (so real scores pass the S1 check); data_copy's qualities and seeds."""
        from experiments.shift_hosts import actors_f32 as A32
        from experiments.shift_hosts import evaluate as EV
        self.assertEqual(E.READOUTS, A32.READOUTS)
        self.assertEqual(C.LEARNERS, A32.LEARNERS)
        self.assertEqual(C.BANKS, EV.BANKS)
        self.assertEqual(C.SITUATIONS, tuple(EV.SITUATIONS))
        self.assertEqual(C.WEIGHTING_BANKS, tuple(EV.WEIGHTING))
        self.assertEqual(E.SCORES_SCHEMA, EV.SCORES_SCHEMA)
        self.assertEqual(E.START_BANKS_JSON_SHA256, EV.START_BANKS_JSON_SHA256)
        self.assertEqual((DC.QUALITIES, DC.SEEDS), (C.QUALITIES, C.SEEDS))
        manifest = gate.code_manifest()
        self.assertEqual({p: manifest.get(p) for p in EV.code_sha256()}, EV.code_sha256())
        self.assertEqual(set(EV.code_sha256()), set(SCORING_CODE))

    def test_export_reads_evaluate_scores(self):
        """A made-up TD3+BC actor file (made-up parameters, never trained) in a runner-layout run folder, with the
        meta the runner gives its actor files, scored by evaluate.score_actor on all 7 banks into
        lanes/scores-step<final>.npz (the scoring lane's own call): export.read_run reads it (the claims input equals
        the scores' summaries, the F1/F2 counts equal the rates, every summary field exists), its anchors equal
        start_banks.json's, and these tests' own scores writer has evaluate's meta keys, summary fields and array
        names."""
        from experiments.shift_hosts import actors_f32 as A32
        from experiments.shift_hosts import evaluate as EV
        fz = frozen()
        f = Fake(fresh("integration"), fz)
        rec = f.run("medium-expert", "td3_bc", "alone", 0, scored=False)
        folder = f.folder("medium-expert", "td3_bc", "alone", 0)
        params, norm = made_up_td3_bc_actor()
        actor = folder / "actors" / f"step{FINAL}.npz"
        meta = dict(f.actor_meta(rec, FINAL), note="made-up parameters for test_claims, not a learned policy")
        man = A32.write_actor_file(actor, "td3_bc", params, meta=meta, **norm)
        rec["snapshots"][str(FINAL)]["actor"] = dict(path=str(actor), sha256=man["npz_sha256"])
        H.write_json(folder / "record.json", rec)
        scores = folder / "lanes" / f"scores-step{FINAL}.npz"
        EV.score_actor(actor, guard.MADE_UP, out=scores)
        got = E.read_run(folder, "medium-expert", "td3_bc", "alone", 0, fz)
        self.assertEqual(got["kind"], "outcome")
        self.assertEqual(got["aux"]["fields"]["scores_final"], "read")
        arrays, sman = H.read(scores)
        summ = sman["meta"]["summaries"]["scored"]
        for b in C.BANKS:
            self.assertEqual(got["run"]["J"][b], summ[b]["J_mean"])
            self.assertEqual(got["run"]["events"]["f1"][b], int(round(summ[b]["f1_rate"] * summ[b]["n"])))
            self.assertEqual(got["run"]["starts"][b], 1000)
            self.assertTrue(set(E.SUMMARY_FIELDS) <= set(summ[b]))
        self.assertTrue(got["run"]["finite"])
        self.assertEqual(got["run"]["source"]["scores"], sman["npz_sha256"])
        anchors = E.read_start_bank_anchors()
        bad = E.check_score_anchors(dict(aux={"medium-expert": {("td3_bc", "alone", 0): got["aux"]}}),
                                    anchors["anchors"])
        self.assertEqual(bad, ([], 1))
        mine = folder / "lanes" / "restated-scores.npz"
        write_scores(mine, "td3_bc", man["npz_sha256"], sman["meta"]["actor_meta"],
                     bank_means(-200.0, -100.0, -150.0), n=1000, permission=dict(kind="made-up"),
                     code=scoring_code(fz))
        marr, mman = H.read(mine)
        self.assertEqual(set(mman["meta"]), set(sman["meta"]))
        self.assertTrue(set(marr) <= set(arrays), sorted(set(marr) - set(arrays)))
        for b in C.BANKS:
            self.assertEqual(set(mman["meta"]["summaries"]["scored"][b]), set(summ[b]))
            self.assertTrue(set(mman["meta"]["anchors"][b]) <= set(sman["meta"]["anchors"][b]))

    @unittest.skipUnless((E.DATA / "medium-expert" / "seed0" / "prepared.json").exists(), "no prepared files")
    def test_rebrac_substituted_shares_from_prepared_manifests(self):
        """ReBRAC's substituted next commands, read from the prepared manifests (data preparation records, not
        outcomes) when no record holds them: medium-expert seed 0 has 3,526 of 45,301 training rows (7.78%)."""
        got = E._rebrac_substituted({}, "medium-expert", E.DATA)
        self.assertEqual(len(got), 5)
        self.assertAlmostEqual(got[0]["train"], 3526 / 45301, places=12)
        self.assertTrue(all(0.07 < g["train"] < 0.09 for g in got))


def _extra():
    import scipy

    return dict(numpy=np.__version__, scipy=scipy.__version__, scoring_environment=SCORING_ENV,
                synthetic_only=True, dry_runs={k: str(v) for k, v in DRY_RUNS.items() if (v / "driver").exists()},
                supplement=dict(READOUTS_V4, dry_run=str(SUPPLEMENT_DRY) if SUPPLEMENT_DRY.exists() else None),
                note="claims and export on made-up numbers; no ThrusterChain outcome is read")


if __name__ == "__main__":
    sys.exit(T.run(sys.modules[__name__], AREA, device_name="none" if SCORING_ENV else "none-training-env",
                   extra=_extra))

"""CPU tests of learners_iw.py and run.py (part B).

BCA (scale fit, refresh, thresholds on band scores, widths, doses, co-trained BCA updates) runs here only on small
random arrays (learners_iw.synthetic_prepared and registered fixtures, opened by the SYNTHETIC token), never on
ThrusterChain data. The only work on real ThrusterChain data: a tiny data build (as test_data.py) and two 300-update
host runs (gapped and full log) through the stage runner; no return of any learned policy is computed. The refusals
are checked on that tiny run: the BCA stages without an approval file, every stage with a wrong specification hash, an
approval naming another specification or a tampered sidecar, missing inputs, and real data relabelled as synthetic.
No valid approval file is ever written next to real data: tests that need a valid approval use a root that holds no
data (EmptyRootTest) or only hand-written fixture records (ReportArithmeticTest).
JAX_PLATFORMS=cpu python -m unittest experiments.shift_iw.test_learners_iw
"""

import hashlib
import json
import math
import shutil
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

import numpy as np

import experiments.shift  # noqa: F401

import jax
import jax.numpy as jnp

import algorithms.td3_bc_bca as P
from calibration import reference as REF
from experiments.shift import learners as LR
from experiments.shift import truth as TR
from experiments.shift_iw import coverage as CV
from experiments.shift_iw import data as D
from experiments.shift_iw import learners_iw as LI
from experiments.shift_iw import lower_edge as LE
from experiments.shift_iw import run as RN
from experiments.shift_iw import world as W

QUIET = lambda m: None
D1_OK = dict(passes=True, checks=dict(holds_removed=dict(value=0.995), Z=dict(value=0.84),
                                      in_ball_mild=dict(value=0.25), in_ball_strong=dict(value=0.36)))
TINY = D.Sizes(train=24, calibration=8, held_out=8, test=8, sample=8, grid_points=5, per_point=20, truth_rows=100,
               noise_map_horizon=50, start_bank=20)
SEED = 3


def leaves_equal(a, b):
    la, lb = jax.tree_util.tree_leaves(a), jax.tree_util.tree_leaves(b)
    return len(la) == len(lb) and all(np.asarray(x).tobytes() == np.asarray(y).tobytes() for x, y in zip(la, lb))


def with_threshold_only(band, threshold):
    """The band's frozen reference with only its threshold changed (for comparing doses row by row)."""
    p = band.state.posterior
    return LI.with_threshold(p, threshold, float(p.lambda_hat), float(p.lambda_hpd), float(p.n_eff))


def write_spec_copy(directory):
    """A copy of iw/spec.json with its sidecar, in a temporary directory."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    spec = directory / "spec.json"
    shutil.copy(LI.SPEC_JSON, spec)
    digest = LI.sha256_file(spec)
    LI.sidecar(spec).write_text(f"{digest}  spec.json\n")
    return spec, digest


def write_approval(path, spec_sha, scope=("band", "coverage"), text="Yes, go ahead.", kind=LI.APPROVAL_KIND,
                   sidecar=True, code=None, qualities=None):
    """A test approval file in a temporary directory (the real one is written only by the orchestrator). Every
    BCA-arm scope in it covers every quality unless `qualities` ({scope: [quality, ...]}) says otherwise."""
    path = Path(path)
    if qualities is None:
        qualities = {s: list(W.QUALITIES) for s in scope if s in LI.ARM_SCOPES}
    body = dict(kind=kind, spec_sha256=spec_sha, user_approval_verbatim=text, scope=list(scope), qualities=qualities,
                code_sha256=LI.code_hashes() if code is None else code, written_by="test fixture")
    path.write_text(json.dumps(body, indent=1))
    if sidecar:
        LI.sidecar(path).write_text(f"{LI.sha256_file(path)}  {path.name}\n")
    return path


# ------------------------------------------------------------------------------------------------ the gate

class CodeHashTest(unittest.TestCase):
    def test_every_repository_module_the_runner_loads_is_hashed(self):
        """In a fresh process: import the stage runner (and acceptance.py, learners.py's one lazy import); every
        module then loaded from the repository must be in code_hashes(), so the approval binds all the code that
        runs. Paths are taken as imported (the package is reached through its symlink in the repository)."""
        import os
        import subprocess
        import sys

        probe = ("import json, os, sys\n"
                 "import experiments.shift_iw.run\n"
                 "import experiments.shift.acceptance\n"
                 "print(json.dumps(sorted({os.path.abspath(m.__file__) for m in list(sys.modules.values())\n"
                 "                         if getattr(m, '__file__', None)})))\n")
        out = subprocess.run([sys.executable, "-c", probe], cwd=str(LI.REPO), env=dict(os.environ), check=True,
                             capture_output=True, text=True).stdout
        repo = os.path.abspath(str(LI.REPO)) + os.sep
        loaded = {p[len(repo):].replace(os.sep, "/") for p in json.loads(out.strip().splitlines()[-1])
                  if p.startswith(repo)}
        hashed = set(LI.code_hashes())
        self.assertIn("experiments/shift_iw/run.py", loaded)
        self.assertEqual(sorted(loaded - hashed), [])
        self.assertIn("experiments/shift_iw/launch.sh", hashed)
        for name in ("runtime/networks.py", "runtime/published.py", "calibration/bank.py",
                     "experiments/signal/lq_harness.py", "experiments/signal/gapped_lq.py", "experiments/truth/gap.py",
                     "experiments/shift/dataset.py", "experiments/shift/acceptance.py"):
            self.assertIn(name, hashed)


class GateTest(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="shift_iw_gate_"))
        self.spec, self.digest = write_spec_copy(self.tmp / "iw")

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_spec_needs_a_matching_sidecar(self):
        self.assertEqual(LI.spec_sha256(self.spec), self.digest)
        LI.sidecar(self.spec).unlink()
        with self.assertRaises(LI.SpecError):
            LI.spec_sha256(self.spec)
        LI.sidecar(self.spec).write_text("0" * 64 + "  spec.json\n")
        with self.assertRaises(LI.SpecError):
            LI.spec_sha256(self.spec)

    def test_approval_rules(self):
        path = self.tmp / "approval.json"
        with self.assertRaisesRegex(LI.ApprovalError, "no approval file"):
            LI.verify_approval(path, self.spec)
        write_approval(path, self.digest, sidecar=False)
        with self.assertRaisesRegex(LI.ApprovalError, "sidecar"):
            LI.verify_approval(path, self.spec)
        write_approval(path, "f" * 64)
        with self.assertRaisesRegex(LI.ApprovalError, "another specification"):
            LI.verify_approval(path, self.spec)
        write_approval(path, self.digest, text="  ")
        with self.assertRaisesRegex(LI.ApprovalError, "verbatim"):
            LI.verify_approval(path, self.spec)
        write_approval(path, self.digest, kind="bca-shift-approval")
        with self.assertRaisesRegex(LI.ApprovalError, "not a shift_iw approval"):
            LI.verify_approval(path, self.spec)
        write_approval(path, self.digest, scope=("band", "train everything"))
        with self.assertRaisesRegex(LI.ApprovalError, "scope"):
            LI.verify_approval(path, self.spec)
        write_approval(path, self.digest, scope=("band", "coverage"))
        with self.assertRaisesRegex(LI.ApprovalError, "does not cover the 'bca' stage"):
            LI.verify_approval(path, self.spec, "bca")
        with self.assertRaisesRegex(LI.ApprovalError, "does not cover the 'bca_estimated' stage"):
            LI.verify_approval(path, self.spec, "bca_estimated")
        # the approval binds the code: a changed code file, or no code hashes at all, is refused
        changed = dict(LI.code_hashes(), **{"experiments/shift_iw/run.py": "0" * 64})
        write_approval(path, self.digest, code=changed)
        with self.assertRaisesRegex(LI.ApprovalError, r"code differs.*run\.py"):
            LI.verify_approval(path, self.spec, "band")
        write_approval(path, self.digest, code="none")
        with self.assertRaisesRegex(LI.ApprovalError, "no code_sha256"):
            LI.verify_approval(path, self.spec, "band")
        write_approval(path, self.digest, scope=("band", "coverage"))
        ok = LI.verify_approval(path, self.spec, "band")
        self.assertEqual(ok.spec_sha256, self.digest)
        self.assertEqual(ok.scope, ("band", "coverage"))
        self.assertEqual(ok.text_sha256, hashlib.sha256(b"Yes, go ahead.").hexdigest())
        # version 5, after review: each kind of BCA arm has its own scope, and each arm scope names the qualities
        # the owner's go covers
        self.assertEqual(LI.ARM_SCOPES, ("bca", "bca_estimated", "bca_lower_edge", "bca_lower_edge_estimated"))
        for qualities in ({}, {"bca": []}, {"bca": ["novice"]}, {"bca": ["medium", "medium"]},
                          {"bca": ["medium"], "bca_lower_edge": ["medium"]}, {"coverage": ["medium"]}):
            write_approval(path, self.digest, scope=("band", "coverage", "bca"), qualities=qualities)
            with self.assertRaisesRegex(LI.ApprovalError, "qualities"):
                LI.verify_approval(path, self.spec)
        write_approval(path, self.digest, scope=("band", "coverage", "bca_lower_edge"),
                       qualities={"bca_lower_edge": ["medium", "expert"]})
        ok = LI.verify_approval(path, self.spec, "bca_lower_edge")
        self.assertEqual(ok.qualities, (("bca_lower_edge", ("expert", "medium")),))
        self.assertTrue(ok.covers("bca_lower_edge", "medium") and ok.covers("band", "random"))
        self.assertFalse(ok.covers("bca_lower_edge", "random") or ok.covers("bca", "medium"))
        with self.assertRaisesRegex(LI.ApprovalError, "does not cover the 'bca' stage"):
            LI.verify_approval(path, self.spec, "bca")      # a go for the lower edge is not a go for the capped pull
        write_approval(path, self.digest, scope=("band", "coverage"))
        # the approval names spec.json's SHA-256, so a changed spec (with a matching new sidecar) is refused too
        self.spec.write_text(self.spec.read_text() + "\n")
        LI.sidecar(self.spec).write_text(f"{LI.sha256_file(self.spec)}  spec.json\n")
        with self.assertRaisesRegex(LI.ApprovalError, "another specification"):
            LI.verify_approval(path, self.spec, "band")

    def test_require_gate_re_reads_the_files(self):
        path = write_approval(self.tmp / "approval.json", self.digest, scope=("band",))
        ok = LI.verify_approval(path, self.spec, "band")
        self.assertIs(LI.require_gate(ok, None, "band"), ok)
        with self.assertRaises(LI.ApprovalError):
            LI.require_gate(ok, None, "coverage")      # outside the approved scope
        write_approval(path, self.digest, scope=("band",), text="Yes, go ahead!")   # a different (valid) file
        with self.assertRaisesRegex(LI.ApprovalError, "changed since"):
            LI.require_gate(ok, None, "band")
        path.unlink()
        with self.assertRaisesRegex(LI.ApprovalError, "no approval file"):
            LI.require_gate(ok, None, "band")

    def test_synthetic_token_opens_registered_synthetic_arrays_only(self):
        prep = LI.synthetic_prepared(n=128, bank=32, eval_states=8, seed=11)
        self.assertIs(LI.require_gate(LI.SYNTHETIC, prep, "band"), LI.SYNTHETIC)
        for world in ("stall", "no_stall", "synthetic"):                 # the label must say synthetic
            with self.assertRaises(LI.ApprovalError):
                LI.require_gate(LI.SYNTHETIC, replace(prep, world=world), "band")
        other = replace(prep, train=prep.train._replace(reward=prep.train.reward + 1.0))   # unregistered content
        with self.assertRaisesRegex(LI.ApprovalError, "not registered"):
            LI.require_gate(LI.SYNTHETIC, other, "band")
        for label in (LI.SYNTHETIC_WORLD, None, "stall"):                 # a bare label is not data
            with self.assertRaises(LI.ApprovalError):
                LI.require_gate(LI.SYNTHETIC, label, "band")
        for bad in (None, True, "approved", dict(spec_sha256=self.digest)):
            with self.assertRaises(LI.ApprovalError):
                LI.require_gate(bad, prep, "band")
        with self.assertRaises(LI.ApprovalError):
            LI.register_synthetic(replace(prep, world="stall"))

    def test_bca_functions_refuse_relabelled_data_with_the_synthetic_token(self):
        cfg = LI.settings()
        prep = LI.synthetic_prepared(n=256, bank=64, eval_states=16, seed=12)
        _, host, _ = LR.BASE.initialize(LR.make_args(cfg, 0), LR.DX, LR.DA, 1.0)
        relabelled = replace(prep, world="stall")
        unregistered = replace(prep, bank=prep.bank._replace(reward=prep.bank.reward * 2.0))
        for data in (relabelled, unregistered):
            with self.assertRaises(LI.ApprovalError):
                LI.fit_band(host, data, cfg, 0, LI.SYNTHETIC, steps=2)
            with self.assertRaises(LI.ApprovalError):
                LI.bca_cotrain(data, cfg, 0, LI.SYNTHETIC, updates=2)
        with self.assertRaises(LI.ApprovalError):
            LI.fit_band(host, prep, cfg, 0, None, steps=2)


# ------------------------------------------------------------------------------------------------ the band (synthetic)

class SyntheticBandTest(unittest.TestCase):
    """BCA's band on a small synthetic host (random arrays): the restated fit, the recomputed scores, the weighted
    thresholds, the widths and doses, and refusal through an infinite threshold."""

    @classmethod
    def setUpClass(cls):
        cls.cfg = LI.settings(band_fit_steps=60)
        cls.prep = LI.synthetic_prepared(n=2048, bank=300, eval_states=128)
        _, cls.host, _ = LI.train_host(cls.prep, cls.cfg, 0, updates=200)
        cls.band = LI.fit_band(cls.host, cls.prep, cls.cfg, 0, LI.SYNTHETIC, chunk=25)
        cls.scores, cls.stream = LI.bank_scores(cls.band, cls.prep.bank, cls.cfg)
        cls.c_bank = LI.synthetic_bank_current(cls.prep)
        w = W.ratio(cls.c_bank, "strong")
        cls.arms = {"unweighted": None, "ones": (np.ones(len(w)), 1.0), "exact": (w, W.full_test_mass("strong")),
                    "hopeless": (np.ones(len(w)), 1e6)}
        cls.cals = LI.thresholds(cls.band, cls.scores, cls.stream, cls.arms)

    def test_band_is_valid_and_leaves_the_host_unchanged(self):
        self.assertTrue(self.band.record["refresh"]["posterior_inputs_valid"])
        self.assertTrue(self.band.record["refresh"]["posterior_certified"])
        self.assertEqual(LI.state_sha256(self.band.state.native), LI.state_sha256(self.host))
        _, fresh, _ = LR.BASE.initialize(LR.make_args(self.cfg, 0), LR.DX, LR.DA, 1.0)   # Python-integer step
        LI.fit_band(fresh, self.prep, self.cfg, 0, LI.SYNTHETIC, steps=2)                  # the host check holds
        self.assertTrue(bool(self.band.state.posterior.ready))
        self.assertEqual(len(self.scores), 300)

    def test_fit_equals_learners_bca_band_line_for_line(self):
        """learners.bca_band's own body (without v1's approval check and its dose readout), run here on the same
        synthetic host, gives the same band bit for bit; so does a different chunking."""
        cfg, prep, rep, steps = self.cfg, self.prep, 0, self.cfg.band_fit_steps
        args = LR.make_args(cfg, rep)
        config = LR.bca_config()
        rng, state, models = LR.bca_initial_state(args, config, native=self.host)
        rows = LR.rng_np(cfg, rep, LR.TAG_BAND).integers(prep.n, size=(steps, cfg.batch_size))
        keys = jax.random.split(jax.random.fold_in(rng, LR.TAG_BAND), steps)

        def body(st, inp):
            r, k = inp
            batch = jax.tree_util.tree_map(lambda x: x[r], prep.train)
            st, diag = P.fit_scale(args, config, models, st, batch, k, 1.0)
            return st, diag["scale_fit_accepted"]

        state, _ = jax.jit(lambda st, r, k: jax.lax.scan(body, st, (r, k)))(state, jnp.asarray(rows), keys)
        key = jax.random.fold_in(jax.random.PRNGKey(int(LR.rng_np(cfg, rep, LR.TAG_BAND, 1).integers(2 ** 31 - 1))),
                                 steps)
        state, _, _ = P.refresh(args, config, models, state, prep.bank, key, 1.0,
                                heldout_ids=np.arange(len(prep.bank.obs)))
        self.assertTrue(leaves_equal(state, self.band.state))
        other = LI.fit_band(self.host, prep, cfg, rep, LI.SYNTHETIC, chunk=steps)
        self.assertTrue(leaves_equal(other.state, self.band.state))

    def test_recomputed_scores_reproduce_bcas_own_threshold(self):
        u = self.cals["unweighted"]
        self.assertTrue(LI.reproduces(self.band.state.posterior, u))
        self.assertEqual(np.float32(u.threshold), np.asarray(self.band.state.posterior.threshold))
        ok, _, _ = CV.unit_check(self.scores, self.stream)   # mark W0
        self.assertTrue(ok)
        self.assertTrue(LI.reproduces(self.band.state.posterior, self.cals["ones"]))

    def test_weighting_changes_only_the_threshold(self):
        ref_u = LI.reference_of(self.band, self.cals["unweighted"])
        ref_w = LI.reference_of(self.band, self.cals["exact"])
        self.assertTrue(leaves_equal(ref_u.cal_params, ref_w.cal_params))
        self.assertTrue(leaves_equal(ref_u.residual_scale, ref_w.residual_scale))
        self.assertTrue(bool(REF.reference_valid(ref_w)))
        self.assertNotEqual(float(ref_u.threshold), float(ref_w.threshold))
        self.assertTrue(leaves_equal(ref_u, LI.reference_of(self.band, self.cals["ones"])))
        ru = LI.readout(self.band, ref_u, self.prep.train)
        rw = LI.readout(self.band, ref_w, self.prep.train)
        ratio = rw["width"].astype(np.float64) / ru["width"].astype(np.float64)
        np.testing.assert_allclose(ratio, float(ref_w.threshold) / float(ref_u.threshold), rtol=1e-6)
        order = np.argsort(ru["width"], kind="stable")
        for r, ref in ((ru, ref_u), (rw, ref_w)):                                   # mark M3's statistics
            m = CV.dose_order(r["width"], r["dose"], float(ref.residual_scale))
            self.assertLessEqual(m["max_drop"], 1e-6)
            self.assertLessEqual(m["max_closed_form_gap"], 1e-6)
        self.assertTrue(np.all(np.diff(rw["dose"][order]) >= -1e-7))                # the same order of rows
        self.assertTrue(np.all((ru["dose"] >= 1) & (ru["dose"] < 1.5)) and ru["support"].all() and ru["valid"])
        # the dose moves by at most 0.125 ln(R_w / R_u) per row (spec, expected size of the return effect)
        bound = 0.125 * abs(math.log(float(ref_w.threshold) / float(ref_u.threshold)))
        self.assertLessEqual(float(np.abs(rw["dose"] - ru["dose"]).max()), bound + 1e-6)

    def test_width_is_threshold_times_unit_times_scale(self):
        r = LI.readout(self.band, self.band.state.posterior, self.prep.bank)
        eta = LI.scale_at(self.band, self.prep.bank)
        U = CV.width(float(self.band.state.posterior.threshold), self.band.unit, eta)
        self.assertEqual(r["width"].tobytes(), U.astype(np.float32).tobytes())

    def test_uncertifiable_threshold_refuses_every_step(self):
        cal = self.cals["hopeless"]
        self.assertFalse(cal.certified)
        ref = LI.reference_of(self.band, cal)
        self.assertTrue(bool(REF.reference_valid(ref)))
        r = LI.readout(self.band, ref, self.prep.bank)
        self.assertTrue(np.all(np.isinf(r["width"])))
        self.assertTrue(np.all(r["dose"] == 1.0) and not r["support"].any() and r["valid"])
        y, q = LI.critic_parts(self.host, self.prep.bank, self.cfg, 0, LI.coverage_key(self.cfg, 0, 0))
        self.assertTrue(np.all(CV.outcomes(y, q, r["width"]) == CV.REFUSED))
        rec = LI.threshold_record(cal, *self.arms["hopeless"])
        self.assertEqual(rec["threshold"], "inf")
        self.assertTrue(np.isinf(float(LI.reference_of(self.band, rec).threshold)))

    def test_per_step_table_on_band_scores_is_monotone(self):
        w = self.arms["exact"][0]
        tab = LI.threshold_tables(self.band, self.scores, self.stream,
                                  {"exact": (w, CV.mass_grid([0.13, 7.4], 1.05))})["exact"]
        self.assertTrue(tab.monotone and not tab.running_max_applied)
        self.assertTrue(np.all(tab.thresholds[1:] >= tab.thresholds[:-1]))
        self.assertEqual(tab.thresholds[np.searchsorted(tab.grid, 1.0)],
                         CV.calibrate(self.scores, self.stream, w, 1.0).threshold)
        with self.assertRaises(LI.ApprovalError):
            LI.threshold_tables(replace(self.band, approval=None), self.scores, self.stream, {})

    def test_per_step_equals_unweighted_under_weights_of_one(self):
        """No tilt: the table on weights 1 is [R_u] at the grid [1.0], and the per-step readout at R_u everywhere is
        BCA's own readout bit for bit (widths, doses, support)."""
        ones = np.ones(len(self.scores))
        tab = LI.threshold_tables(self.band, self.scores, self.stream, {"none": (ones, CV.mass_grid(ones, 1.01))})["none"]
        R_u = self.cals["unweighted"].threshold
        self.assertEqual((tab.grid.tolist(), tab.thresholds.tolist()), ([1.0], [R_u]))
        R, _ = CV.lookup(tab, np.ones(len(self.prep.train.obs)))
        ps = LI.per_step_readout(self.band, R, self.prep.train)
        bca = LI.readout(self.band, self.band.state.posterior, self.prep.train)
        for k in ("width", "dose", "support"):
            self.assertEqual(ps[k].tobytes(), bca[k].tobytes(), k)
        self.assertFalse(ps["refused"].any())
        self.assertTrue(ps["valid"])

    def test_per_row_dose_uses_each_rows_own_threshold(self):
        """Each row's width is its own R x unit x scale, and its dose is what BCA's own readout gives that row when
        the band's threshold is set to that row's R."""
        trans = self.prep.bank
        n = len(trans.obs)
        levels = np.array([0.5, 1.0, 2.0, 4.0]) * float(self.band.state.posterior.threshold)
        R = levels[np.arange(n) % 4]
        ps = LI.per_step_readout(self.band, R, trans)
        eta = LI.scale_at(self.band, trans)
        self.assertEqual(ps["width"].tobytes(), CV.width(R, self.band.unit, eta).tobytes())
        for r in levels:
            rows = R == r
            ref = with_threshold_only(self.band, r)
            own = LI.readout(self.band, ref, trans)
            self.assertEqual(ps["dose"][rows].tobytes(), own["dose"][rows].tobytes())
        np.testing.assert_allclose(ps["dose"], CV.closed_form_dose(ps["width"], self.band.unit), atol=1e-6)
        self.assertGreater(len(np.unique(np.round(ps["dose"], 6))), 4)

    def test_refusal_gives_the_largest_dose(self):
        """A row whose own threshold is +inf gets the dose 1 + blend = 1.5 (BCA itself gives it 1); the others are
        untouched."""
        trans = self.prep.bank
        n = len(trans.obs)
        R_u = float(self.band.state.posterior.threshold)
        R = np.where(np.arange(n) % 3 == 0, np.inf, R_u)
        ps = LI.per_step_readout(self.band, R, trans)
        bca = LI.readout(self.band, self.band.state.posterior, trans)
        hopeless = LI.readout(self.band, LI.reference_of(self.band, self.cals["hopeless"]), trans)
        inf = np.isinf(R)
        self.assertTrue(np.all(ps["dose"][inf] == 1.5) and np.all(ps["refused"] == inf))
        self.assertTrue(np.all(np.isinf(ps["width"][inf])) and not ps["support"][inf].any())
        self.assertTrue(np.all(hopeless["dose"] == 1.0))                     # BCA's own rule for R = +inf
        self.assertEqual(ps["dose"][~inf].tobytes(), bca["dose"][~inf].tobytes())
        self.assertTrue(ps["valid"])
        # before the first refresh (reference not ready) every dose is 1, refused or not
        args, config = LR.make_args(self.cfg, 0), LR.bca_config()
        _, fresh, mods = LR.bca_initial_state(args, config)
        out, refused = LI.per_row_dose(mods, fresh, trans, jnp.full(n, jnp.inf), config.blend)
        self.assertTrue(np.all(np.asarray(out.dose) == 1.0) and not np.asarray(refused).any())

    def test_threshold_record_round_trips(self):
        for name in ("unweighted", "exact"):
            rec = json.loads(json.dumps(LI.threshold_record(self.cals[name], *(self.arms[name] or (None, None)))))
            self.assertTrue(leaves_equal(LI.reference_of(self.band, rec), LI.reference_of(self.band, self.cals[name])))

    def test_critic_parts_are_bcas_own_target_and_q(self):
        key = LI.coverage_key(self.cfg, 0, 7)
        n = len(self.prep.bank.obs)
        y, q = LI.critic_parts(self.host, self.prep.bank, self.cfg, 0, key, chunk=n)   # one chunk, no padding
        args = LR.make_args(self.cfg, 0)
        y0 = P.native_target(args, LI.models(), self.host, self.prep.bank, jax.random.fold_in(key, 0), 1.0)
        q0 = LR.CRITIC.apply(self.host.critic.params, self.prep.bank.obs, self.prep.bank.action).min(axis=-1)
        np.testing.assert_allclose(y, np.asarray(y0, np.float64), rtol=2e-6, atol=1e-6)
        np.testing.assert_allclose(q, np.asarray(q0, np.float64), rtol=2e-6, atol=1e-6)
        y2, q2 = LI.critic_parts(self.host, self.prep.bank, self.cfg, 0, key, chunk=128)   # chunked and padded
        np.testing.assert_allclose(q2, q, rtol=2e-6, atol=1e-6)
        self.assertFalse(np.array_equal(y2, y))     # another noise draw per chunk, same law
        r = LI.readout(self.band, self.band.state.posterior, self.prep.bank)
        o = CV.outcomes(y, q, r["width"])
        self.assertLessEqual(RN.hard_coverage_mismatches(o, y, q, r["width"]), 2)

    def test_band_saves_and_loads(self):
        tmp = Path(tempfile.mkdtemp(prefix="shift_iw_band_"))
        try:
            sha = LR.save_tree(tmp / "band.msgpack", self.band.state)
            again = LI.load_band(tmp / "band.msgpack", sha, self.host, self.prep, self.cfg, 0, LI.SYNTHETIC,
                                 json.loads(json.dumps(RN.plain(self.band.record))))
            self.assertTrue(leaves_equal(again.state, self.band.state))
            self.assertEqual(again.refresh_key.tobytes(), self.band.refresh_key.tobytes())
            s2, st2 = LI.bank_scores(again, self.prep.bank, self.cfg)
            self.assertEqual(s2.tobytes(), self.scores.tobytes())
            self.assertEqual(st2.tobytes(), self.stream.tobytes())
        finally:
            shutil.rmtree(tmp, ignore_errors=True)


# ------------------------------------------------------------------------------------------------ BCA arms (synthetic)

class SyntheticBCAArmTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cfg = LI.settings(bca_warmup=20, bca_refresh=10, chunk=10)
        cls.prep = LI.synthetic_prepared(n=1024, bank=200, eval_states=64, seed=1)
        cls.w = W.ratio(LI.synthetic_bank_current(cls.prep), "strong")
        cls.unweighted = LI.bca_cotrain(cls.prep, cls.cfg, 0, LI.SYNTHETIC, updates=40)
        cls.ones = LI.bca_cotrain(cls.prep, cls.cfg, 0, LI.SYNTHETIC, updates=40,
                                  weighting=dict(name="ones", weights=np.ones(len(cls.w)), test_mass=1.0))
        cls.exact = LI.bca_cotrain(cls.prep, cls.cfg, 0, LI.SYNTHETIC, updates=40,
                                   weighting=dict(name="exact strong", weights=cls.w,
                                                  test_mass=W.full_test_mass("strong")))
        cls.c_rows = LI.synthetic_train_current(cls.prep)
        masses = W.ratio(cls.c_rows, "strong")
        cls.per_row = LI.bca_cotrain(cls.prep, cls.cfg, 0, LI.SYNTHETIC, updates=40,
                                     weighting=dict(name="exact strong", mode="per_row", weights=cls.w,
                                                    row_masses=masses, grid=CV.mass_grid(masses, 1.05)))
        ones = np.ones(cls.prep.n)
        cls.per_row_ones = LI.bca_cotrain(cls.prep, cls.cfg, 0, LI.SYNTHETIC, updates=40,
                                          weighting=dict(name="ones", mode="per_row", weights=np.ones(len(cls.w)),
                                                         row_masses=ones, grid=CV.mass_grid(ones, 1.05)))
        cls.refusing = np.where(cls.c_rows > 0.6, 1e6, 1.0)        # rows whose test atom is far too heavy to certify
        cls.per_row_refused = LI.bca_cotrain(cls.prep, cls.cfg, 0, LI.SYNTHETIC, updates=40,
                                             weighting=dict(name="refusing", mode="per_row",
                                                            weights=np.ones(len(cls.w)), row_masses=cls.refusing,
                                                            grid=CV.mass_grid(cls.refusing, 10.0)))

    def test_unweighted_arm_is_td3_bc_bca_itself(self):
        """Before any refresh our loop is td3_bc_bca.make_train_step bit for bit."""
        rec, state, _ = LI.bca_cotrain(self.prep, self.cfg, 0, LI.SYNTHETIC, updates=20)
        self.assertEqual(rec["refreshes"], [])
        args = LR.make_args(self.cfg, 0)
        config = LR.bca_config()
        rng, st0, mods = LR.bca_initial_state(args, config)
        step = P.make_train_step(args, config, mods, self.prep.train, 1.0)
        (_, theirs, _), _ = jax.jit(lambda c: jax.lax.scan(step, c, None, length=20))((rng, st0, jnp.int32(0)))
        self.assertTrue(leaves_equal(state, theirs))

    def test_refreshes_and_records(self):
        rec = self.unweighted[0]
        self.assertFalse(rec["failed"])
        self.assertEqual([e["step"] for e in rec["refreshes"]], [20, 30])
        self.assertTrue(all(e["posterior_certified"] for e in rec["refreshes"]))
        self.assertEqual(rec["uncertified_refreshes"], 0)
        self.assertEqual(rec["final_update"], 40)

    def test_weights_of_one_equal_the_unweighted_arm(self):
        self.assertTrue(leaves_equal(self.ones[1], self.unweighted[1]))
        for e, f in zip(self.ones[0]["refreshes"], self.unweighted[0]["refreshes"]):
            self.assertEqual(e["unweighted_threshold"], f["unweighted_threshold"])
            self.assertEqual(np.float32(e["weighted"]["threshold"]), np.float32(f["unweighted_threshold"]))

    def test_weighting_replaces_only_the_threshold(self):
        rw, ru = self.exact[0]["refreshes"], self.unweighted[0]["refreshes"]
        self.assertEqual(rw[0]["unweighted_threshold"], ru[0]["unweighted_threshold"])   # identical until refresh 1
        self.assertNotEqual(rw[0]["weighted"]["threshold"], rw[0]["unweighted_threshold"])
        self.assertEqual(self.exact[0]["initial_sha256"], self.unweighted[0]["initial_sha256"])
        self.assertEqual(self.exact[0]["batch_stream_sha256"], self.unweighted[0]["batch_stream_sha256"])
        post = self.exact[1].posterior
        self.assertEqual(np.float32(rw[-1]["weighted"]["threshold"]), np.asarray(post.threshold))
        self.assertTrue(bool(REF.reference_valid(post)))
        self.assertFalse(leaves_equal(self.exact[1].native.actor.params, self.unweighted[1].native.actor.params))

    def test_weighting_needs_one_weight_per_bank_row(self):
        with self.assertRaises(ValueError):
            LI.bca_cotrain(self.prep, self.cfg, 0, LI.SYNTHETIC, updates=10,
                           weighting=dict(name="bad", weights=np.ones(3), test_mass=1.0))
        with self.assertRaises(ValueError):                         # one test mass per kept training row
            LI.bca_cotrain(self.prep, self.cfg, 0, LI.SYNTHETIC, updates=10,
                           weighting=dict(name="bad", mode="per_row", weights=self.w, row_masses=np.ones(3),
                                          grid=[1.0]))
        with self.assertRaises(ValueError):                         # a mass beyond the grid
            LI.bca_cotrain(self.prep, self.cfg, 0, LI.SYNTHETIC, updates=10,
                           weighting=dict(name="bad", mode="per_row", weights=self.w,
                                          row_masses=np.full(self.prep.n, 2.0), grid=[1.0]))

    def test_per_row_arm_with_weights_of_one_is_the_unweighted_arm(self):
        """Every weight and every row's mass 1: the table is [R_u], so the per-row arm is BCA itself bit for bit."""
        self.assertTrue(leaves_equal(self.per_row_ones[1], self.unweighted[1]))
        for e, f in zip(self.per_row_ones[0]["refreshes"], self.unweighted[0]["refreshes"]):
            self.assertEqual(len(e["per_row"]["table"]["thresholds"]), 1)
            self.assertEqual(np.float32(e["per_row"]["table"]["thresholds"][0]), np.float32(f["unweighted_threshold"]))
            self.assertEqual(e["per_row"]["refused_rows"], 0)
        self.assertEqual(self.per_row_ones[0]["weighting"]["grid_points"], 1)

    def test_per_row_arm_uses_each_rows_own_threshold(self):
        rec, state, _ = self.per_row
        self.assertFalse(rec["failed"])
        self.assertEqual(rec["weighting"]["mode"], "per_row")
        self.assertLessEqual(rec["weighting"]["max_mass_gap"], 0.05 + 1e-12)
        self.assertEqual(rec["initial_sha256"], self.unweighted[0]["initial_sha256"])
        self.assertEqual(rec["batch_stream_sha256"], self.unweighted[0]["batch_stream_sha256"])
        last = rec["refreshes"][-1]
        self.assertEqual(last["unweighted_threshold"], self.unweighted[0]["refreshes"][-1]["unweighted_threshold"])
        self.assertTrue(rec["tables_monotone"] and not rec["running_max_applied"])
        self.assertEqual(rec["refused_rows_max"], 0)
        rows = last["per_row"]["rows"]
        self.assertLess(rows["min"], rows["max"])                    # rows at different weights, different R
        self.assertEqual(np.asarray(state.posterior.threshold, np.float32),
                         np.float32(last["unweighted_threshold"]))   # BCA's own reference is kept
        self.assertFalse(leaves_equal(state.native.actor.params, self.unweighted[1].native.actor.params))
        self.assertFalse(leaves_equal(state.native.actor.params, self.exact[1].native.actor.params))

    def test_refused_rows_get_the_largest_dose_and_are_counted(self):
        rec, state, _ = self.per_row_refused
        heavy = int(np.sum(self.refusing > 1.0))
        self.assertFalse(rec["failed"])
        self.assertEqual(rec["refused_rows_per_refresh"], [heavy, heavy])
        self.assertEqual(rec["refused_rows_max"], heavy)
        self.assertGreater(rec["batches_with_refused_rows"], 0)
        self.assertEqual(rec["weighting"]["refused_dose"], 1.5)
        self.assertEqual(rec["refreshes"][-1]["per_row"]["table"]["thresholds"][-1], "inf")
        after = [c for c in rec["chunks"] if c["update"] > 20]
        before = [c for c in self.per_row_ones[0]["chunks"] if c["update"] > 20]
        self.assertTrue(all(a["bc_multiplier_mean"] > b["bc_multiplier_mean"] for a, b in zip(after, before)))
        self.assertTrue(all(a["bc_refused_fraction"] > 0 for a in after))


# ------------------------------------------------------------------------------------------------ room check (synthetic)

class SyntheticRoomTest(unittest.TestCase):
    def test_room_check_runs_its_arms_and_matches_blind(self):
        cfg = LI.settings(clone_steps=20, actor_only_steps=8, actor_only_chunk=4, oracle_refresh=4, oracle_horizon=5,
                          match_rounds=2, row_chunk=1024)
        prep = LI.synthetic_prepared(n=512, bank=64, eval_states=64, seed=2)
        _, host, _ = LI.train_host(prep, cfg, 0, updates=50)
        params, rec = LI.room_check(prep, host.critic.params, cfg, 0)
        self.assertEqual(set(LI.ROOM_EVEN_DOSES) | {"placed"}, {a["name"] for a in rec["arms"]})
        self.assertTrue({"none", "placed", "even_x1.1", "even_x1.25", "even_x1.5"} <= set(params))
        self.assertEqual(sorted(a["value"] for a in rec["arms"] if a["kind"] == "even"), [1.0, 1.1, 1.25, 1.5])
        self.assertEqual(rec["strengths"]["none"], 0.0)
        self.assertTrue(all(math.isfinite(v) for v in rec["strengths"].values()))
        self.assertEqual([h["step"] for h in rec["dose_history"]], [0, 4])
        for h in rec["dose_history"]:
            self.assertTrue(1.0 <= h["placed"]["min"] and h["placed"]["max"] < 1.5)
            self.assertEqual(h["even_x1.25"]["min"], h["even_x1.25"]["max"])
        self.assertIn("multiplier", rec["match"])
        self.assertEqual(rec["oracle_calls"], 2)
        tmp = Path(tempfile.mkdtemp(prefix="shift_iw_room_"))
        try:
            sha = LR.save_tree(tmp / "actors.msgpack", params)
            again = LR.load_tree(tmp / "actors.msgpack", LI.room_template(sorted(params)), sha)
            self.assertTrue(leaves_equal(again, params))
        finally:
            shutil.rmtree(tmp, ignore_errors=True)


# ------------------------------------------------------------------------------------------------ evaluation mechanics

class EvaluateTest(unittest.TestCase):
    def test_points_follow_the_rollout_and_the_anchors(self):
        """evaluate_actor's arithmetic on an untrained random network and four synthetic starts per law (not a
        learned policy, not the start banks)."""
        prep = LI.synthetic_prepared(n=64, bank=40, eval_states=8, seed=3)
        params = LR.initial_actor(LI.settings(), 0)
        rng = np.random.default_rng([W.MASTER_SEED, 4242])
        banks = {t: W.draw_starts(4, rng, t) for t in W.TILT_NAMES}
        anchors = {t: dict(anchor=dict(random=-100.0, top=-50.0)) for t in W.TILT_NAMES}
        ev = LI.evaluate_actor(params, prep, banks, anchors)
        J = TR.rollout(LR.mlp_policy(params, prep), banks["strong"], W.PHYSICS).J
        self.assertAlmostEqual(ev["strong"]["J"], float(J.mean()), places=9)
        self.assertAlmostEqual(ev["strong"]["points"], float((100 * (J + 100.0) / 50.0).mean()), places=9)
        self.assertEqual(ev["mild"]["starts"], 4)


# ------------------------------------------------------------------------------------------------ the stage runner

class RunnerTest(unittest.TestCase):
    """A tiny real-data build and two 300-update host runs through run.py; then the refusals. No return is
    computed, and no valid approval file exists in this root."""

    @classmethod
    def setUpClass(cls):
        cls.tmp = Path(tempfile.mkdtemp(prefix="shift_iw_runner_"))
        cls.spec, cls.digest = write_spec_copy(cls.tmp / "iw")
        cls.ctx = RN.Context(root=cls.tmp / "runs", spec_path=cls.spec, log=QUIET)
        RN.stage_data(cls.ctx, [SEED], start_banks=True, sizes=TINY)
        RN.stage_host(cls.ctx, [SEED], ("gap", "full"), updates=300)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def tearDown(self):
        # every test leaves the specification frozen and no approval behind
        LI.sidecar(self.spec).write_text(f"{LI.sha256_file(self.spec)}  spec.json\n")
        for p in (self.ctx.approval_path, LI.sidecar(self.ctx.approval_path)):
            if p.exists():
                p.unlink()
        for base in (self.ctx.root, self.ctx.quality_root):
            self.assertFalse((base / "band").exists(), "no band may exist in this root")
            for d in ("coverage", "bca", "bca_lower_edge"):
                self.assertFalse((base / d).exists(), d)

    def test_tiny_host_runs(self):
        recs = {k: RN.finished(RN.host_dir(self.ctx, k, SEED) / "record.json", self.digest) for k in RN.HOST_KINDS}
        for kind, rec in recs.items():
            self.assertIsNotNone(rec, kind)
            r = rec["result"]
            self.assertFalse(r["failed"])
            self.assertEqual(r["final_update"], 300)
            self.assertEqual(r["args"]["alpha"], 2.5)
            self.assertEqual(r["args"]["discount"], 0.98)
            self.assertEqual([s["update"] for s in r["snapshots"]], [150, 300])
            state = RN.load_host_state(self.ctx, kind, SEED, self.digest)
            self.assertEqual(LR.tree_sha256(state.actor.params), r["snapshots"][-1]["actor_sha256"])
            self.assertNotIn("points", json.dumps(rec))
        g, f = (recs[k]["result"]["data"] for k in ("gap", "full"))
        self.assertEqual(g["obs_mean"], f["obs_mean"])        # both hosts use the gapped normalisation rule
        self.assertEqual(g["obs_std"], f["obs_std"])
        self.assertGreater(f["rows"], g["rows"])
        self.assertEqual(g["removal"], "gap")
        self.assertEqual(f["removal"], "none")
        self.assertFalse((self.ctx.root / "evaluate").exists())
        RN.stage_host(self.ctx, [SEED], ("gap",), updates=300)   # finished: not redone
        self.assertEqual(RN.finished(RN.host_dir(self.ctx, "gap", SEED) / "record.json", self.digest), recs["gap"])

    def test_bca_stages_refuse_without_approval(self):
        for stage in (RN.stage_band, RN.stage_coverage):
            with self.assertRaisesRegex(RN.Refused, "no approval file"):
                stage(self.ctx, [SEED])
        with self.assertRaisesRegex(RN.Refused, "no approval file"):
            RN.stage_bca(self.ctx, [SEED])
        with self.assertRaisesRegex(RN.Refused, "no approval file"):
            RN.stage_bca_lower_edge(self.ctx, [SEED])
        for stage in ("band", "bca-lower-edge"):
            self.assertEqual(RN.main([stage, "--quality", "expert", "--seeds", str(SEED), "--root", str(self.ctx.root),
                                      "--spec", str(self.spec)]), 2)

    def test_bca_stages_refuse_an_approval_of_another_specification(self):
        write_approval(self.ctx.approval_path, "e" * 64, scope=LI.BCA_STAGES)
        for stage in (RN.stage_band, RN.stage_coverage, RN.stage_bca, RN.stage_bca_lower_edge):
            with self.assertRaisesRegex(RN.Refused, "another specification"):
                stage(self.ctx, [SEED])

    def test_bca_stages_refuse_an_approval_without_its_sidecar(self):
        write_approval(self.ctx.approval_path, self.digest, scope=LI.BCA_STAGES, sidecar=False)
        with self.assertRaisesRegex(RN.Refused, "sidecar"):
            RN.stage_band(self.ctx, [SEED])

    def test_bca_stages_refuse_an_approval_of_other_code(self):
        changed = dict(LI.code_hashes(), **{"experiments/shift_iw/coverage.py": "0" * 64})
        write_approval(self.ctx.approval_path, self.digest, scope=LI.BCA_STAGES, code=changed)
        for stage in (RN.stage_band, RN.stage_coverage, RN.stage_bca, RN.stage_bca_lower_edge):
            with self.assertRaisesRegex(RN.Refused, "code differs"):
                stage(self.ctx, [SEED])

    def test_synthetic_token_refuses_relabelled_real_data(self):
        """Real (tiny) ThrusterChain data relabelled as synthetic: the token checks content, so BCA never runs."""
        data = D.load_seed(SEED, self.ctx.data_root)
        host = RN.load_host_state(self.ctx, "gap", SEED, self.digest)
        for kind in ("gap", "none"):
            prep = replace(D.prepare(data, kind, self.ctx.cfg), world=LI.SYNTHETIC_WORLD)
            with self.assertRaisesRegex(LI.ApprovalError, "not registered"):
                LI.fit_band(host, prep, self.ctx.cfg, SEED, LI.SYNTHETIC, steps=2)
            with self.assertRaisesRegex(LI.ApprovalError, "not registered"):
                LI.bca_cotrain(prep, self.ctx.cfg, SEED, LI.SYNTHETIC, updates=2)

    def test_every_stage_refuses_a_wrong_specification_hash(self):
        LI.sidecar(self.spec).write_text("1" * 64 + "  spec.json\n")
        calls = [lambda: RN.stage_data(self.ctx, [SEED], sizes=TINY), lambda: RN.stage_host(self.ctx, [SEED]),
                 lambda: RN.stage_room(self.ctx, [SEED]), lambda: RN.stage_room_lower_edge(self.ctx, [SEED]),
                 lambda: RN.stage_band(self.ctx, [SEED]), lambda: RN.stage_bca_lower_edge(self.ctx, [SEED]),
                 lambda: RN.stage_coverage(self.ctx, [SEED]), lambda: RN.stage_bca(self.ctx, [SEED]),
                 lambda: RN.stage_evaluate(self.ctx, [SEED]), lambda: RN.stage_report(self.ctx, [SEED])]
        for call in calls:
            with self.assertRaisesRegex(RN.Refused, "differs from its sidecar"):
                call()
        LI.sidecar(self.spec).unlink()
        with self.assertRaisesRegex(RN.Refused, "no sidecar"):
            RN.stage_host(self.ctx, [SEED])

    def test_a_changed_specification_unfinishes_everything(self):
        """Records name the specification's SHA-256: after a change (even with a new matching sidecar) the finished
        data and hosts no longer count, so later stages refuse."""
        original = self.spec.read_text()
        try:
            self.spec.write_text(original + "\n")
            LI.sidecar(self.spec).write_text(f"{LI.sha256_file(self.spec)}  spec.json\n")
            with self.assertRaisesRegex(RN.Refused, "has not finished under this specification"):
                RN.stage_room(self.ctx, [SEED])
        finally:
            self.spec.write_text(original)

    def test_stages_refuse_missing_inputs(self):
        with self.assertRaisesRegex(RN.Refused, "data for seed 4"):
            RN.stage_host(self.ctx, [4])
        with self.assertRaisesRegex(RN.Refused, "data for seed 4"):
            RN.stage_room(self.ctx, [4])
        with self.assertRaisesRegex(RN.Refused, "room check of seed 3"):
            RN.stage_evaluate(self.ctx, [SEED], groups=("room",))
        with self.assertRaisesRegex(RN.Refused, "the room check of seed 3"):     # the lower edge's needs the room's
            RN.stage_room_lower_edge(self.ctx, [SEED])
        with self.assertRaisesRegex(RN.Refused, "the lower edge's room check of seed 3"):
            RN.stage_evaluate(self.ctx, [SEED], groups=("room_lower_edge",))
        with self.assertRaisesRegex(RN.Refused, "room check of seed 3"):
            RN.stage_evaluate(self.ctx, [SEED])                                   # the default groups
        with self.assertRaisesRegex(RN.Refused, "data for seed 4"):
            RN.stage_report(self.ctx, [SEED, 4])

    def test_report_marks_pending_without_inputs(self):
        body = RN.build_report(self.ctx, [SEED], self.digest)
        self.assertEqual(body["seeds"], [SEED])
        self.assertFalse(body["complete"])
        for k in ("M1", "M2", "M3", "W0", "W1", "W2", "W3", "W4", "R1", "R2", "L1", "L2"):
            self.assertEqual(body["marks"][k]["status"], "pending", k)
        self.assertEqual(body["marks"]["I1"]["status"], "not run")
        self.assertEqual(body["marks"]["L3"]["status"], "not run")
        self.assertFalse(body["bca_allowed"] or body["bca_lower_edge_allowed"])
        self.assertIn(body["marks"]["D1"]["status"], ("pass", "fail"))   # a tiny build: a fixture, not a finding

    def test_records_need_their_sidecars(self):
        path = RN.host_dir(self.ctx, "gap", SEED) / "record.json"
        side = LI.sidecar(path)
        saved = side.read_text()
        try:
            side.write_text("0" * 64 + "  record.json\n")
            self.assertIsNone(RN.read_record(path))
            with self.assertRaisesRegex(RN.Refused, "gap host of seed 3"):
                RN.stage_room(self.ctx, [SEED])
        finally:
            side.write_text(saved)

    def test_status_and_freeze_spec(self):
        rows = RN.status(self.ctx, [SEED])
        self.assertTrue(rows["host gap 3"] and rows["host full 3"] and rows["data 3"] and rows["start banks"])
        self.assertFalse(rows["approval"] or rows["band gap 3"] or rows["report"])
        fresh = self.tmp / "fresh"
        fresh.mkdir(exist_ok=True)
        spec = fresh / "spec.json"
        shutil.copy(LI.SPEC_JSON, spec)
        ctx = RN.Context(root=self.tmp / "other", spec_path=spec, log=QUIET)
        self.assertEqual(RN.freeze_spec(ctx), LI.sha256_file(spec))
        self.assertEqual(RN.freeze_spec(ctx), LI.sha256_file(spec))      # already frozen: no change
        spec.write_text(spec.read_text() + "\n")
        with self.assertRaisesRegex(RN.Refused, "differs"):
            RN.freeze_spec(ctx)

    def test_plain_json(self):
        rec = RN.plain(dict(a=np.float32(1.5), b=np.inf, c=(np.int64(2), np.bool_(True)), d=np.arange(2),
                            e=jnp.asarray(3.0), f=-np.inf, g=float("nan")))
        self.assertEqual(rec, dict(a=1.5, b="inf", c=[2, True], d=[0, 1], e=3.0, f="-inf", g="nan"))
        json.dumps(rec, allow_nan=False)


class EmptyRootTest(unittest.TestCase):
    """Refusals that need a valid approval file, in a root that holds no data at all: if a refusal failed, there
    would be nothing for BCA to run on."""

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="shift_iw_empty_"))
        self.spec, self.digest = write_spec_copy(self.tmp / "iw")
        self.ctx = RN.Context(root=self.tmp / "runs", spec_path=self.spec, log=QUIET)
        self.ctx.root.mkdir(parents=True)

    def tearDown(self):
        self.assertFalse(any((base / d).exists() for base in (self.ctx.root, self.ctx.quality_root)
                             for d in ("data", "band", "coverage", "bca", "bca_lower_edge")))
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_tampered_sidecar_and_scope(self):
        write_approval(self.ctx.approval_path, self.digest, scope=LI.BCA_STAGES)
        LI.sidecar(self.ctx.approval_path).write_text("0" * 64 + "  approval.json\n")   # tampered
        with self.assertRaisesRegex(RN.Refused, "sidecar"):
            RN.stage_coverage(self.ctx, [SEED])
        write_approval(self.ctx.approval_path, self.digest, scope=("coverage",))
        with self.assertRaisesRegex(RN.Refused, "does not cover the 'band' stage"):
            RN.stage_band(self.ctx, [SEED])
        with self.assertRaisesRegex(RN.Refused, "does not cover the 'bca' stage"):
            RN.stage_bca(self.ctx, [SEED])
        write_approval(self.ctx.approval_path, self.digest, scope=("band", "coverage"))   # no BCA-arm scope
        with self.assertRaisesRegex(RN.Refused, "does not cover the 'bca_lower_edge' stage"):
            RN.stage_bca_lower_edge(self.ctx, [SEED])
        with self.assertRaisesRegex(RN.Refused, "unknown lower-edge BCA arm"):
            RN.stage_bca_lower_edge(self.ctx, [SEED], arms=("exact_strong_single",))

    def test_each_kind_and_quality_needs_its_own_go(self):
        """A go for the capped pull's arms ('bca') does not open the lower edge's ('bca_lower_edge'), nor the other
        way round; and a go names its qualities, so it opens no other quality."""
        write_approval(self.ctx.approval_path, self.digest, scope=("band", "coverage", "bca"))
        with self.assertRaisesRegex(RN.Refused, "does not cover the 'bca_lower_edge' stage"):
            RN.stage_bca_lower_edge(self.ctx, [SEED])
        with self.assertRaisesRegex(RN.Refused, "the report"):            # the capped pull gets past its scope
            RN.stage_bca(self.ctx, [SEED])
        write_approval(self.ctx.approval_path, self.digest, scope=("band", "coverage", "bca_lower_edge"))
        with self.assertRaisesRegex(RN.Refused, "does not cover the 'bca' stage"):
            RN.stage_bca(self.ctx, [SEED])
        with self.assertRaisesRegex(RN.Refused, "the report"):
            RN.stage_bca_lower_edge(self.ctx, [SEED])
        write_approval(self.ctx.approval_path, self.digest, scope=("band", "coverage", "bca", "bca_lower_edge"),
                       qualities={"bca": ["medium"], "bca_lower_edge": ["medium", "medium-expert"]})
        self.assertEqual(self.ctx.quality, "expert")
        with self.assertRaisesRegex(RN.Refused, "'bca' scope does not cover the expert quality"):
            RN.stage_bca(self.ctx, [SEED])
        with self.assertRaisesRegex(RN.Refused, "'bca_lower_edge' scope does not cover the expert quality"):
            RN.stage_bca_lower_edge(self.ctx, [SEED])
        medium = self.ctx.at("medium")
        with self.assertRaisesRegex(RN.Refused, "the report"):           # medium's go gets it past the gate
            RN.stage_bca(medium, [SEED])
        with self.assertRaisesRegex(RN.Refused, "the report"):
            RN.stage_bca_lower_edge(medium, [SEED])
        self.assertIsInstance(RN.approval_for(self.ctx, "band"), LI.Approval)   # band and coverage: every quality

    def test_a_valid_approval_still_needs_every_input(self):
        write_approval(self.ctx.approval_path, self.digest, scope=LI.BCA_STAGES)
        self.assertIsInstance(RN.approval_for(self.ctx, "band"), LI.Approval)
        with self.assertRaisesRegex(RN.Refused, "data for seed 4"):
            RN.stage_band(self.ctx, [4])
        with self.assertRaisesRegex(RN.Refused, "data for seed 4"):
            RN.stage_coverage(self.ctx, [4])
        with self.assertRaisesRegex(RN.Refused, "the report"):
            RN.stage_bca(self.ctx, [4])
        with self.assertRaisesRegex(RN.Refused, "the report"):
            RN.stage_bca_lower_edge(self.ctx, [4])

    def test_the_optional_arm_needs_its_own_scope(self):
        write_approval(self.ctx.approval_path, self.digest, scope=("band", "coverage", "bca", "bca_lower_edge"))
        with self.assertRaisesRegex(RN.Refused, "does not cover the 'bca_estimated' stage"):
            RN.stage_bca(self.ctx, [0], arms=("estimated_strong",))
        with self.assertRaisesRegex(RN.Refused, "the report"):      # the two planned arms get past the scope
            RN.stage_bca(self.ctx, [0])
        with self.assertRaisesRegex(RN.Refused, "does not cover the 'bca_lower_edge_estimated' stage"):
            RN.stage_bca_lower_edge(self.ctx, [0], arms=("estimated_strong",))
        with self.assertRaisesRegex(RN.Refused, "the report"):      # the lower edge's planned arms likewise
            RN.stage_bca_lower_edge(self.ctx, [0])
        write_approval(self.ctx.approval_path, self.digest,
                       scope=("band", "coverage", "bca", "bca_lower_edge", "bca_estimated"))
        with self.assertRaisesRegex(RN.Refused, "does not cover the 'bca_lower_edge_estimated' stage"):
            RN.stage_bca_lower_edge(self.ctx, [0], arms=("estimated_strong",))   # the capped pull's is not enough

    def test_approval_template_prints_a_body_that_verifies(self):
        import contextlib
        import io

        with contextlib.redirect_stdout(io.StringIO()) as out:
            body = RN.approval_template(self.ctx)
        self.assertEqual(json.loads(out.getvalue()), body)
        self.assertFalse(self.ctx.approval_path.exists())               # printed, never written
        self.assertEqual((body["spec_sha256"], body["code_sha256"]), (self.digest, LI.code_hashes()))
        body["user_approval_verbatim"] = "Yes, go."
        self.ctx.approval_path.write_text(json.dumps(body))
        LI.sidecar(self.ctx.approval_path).write_text(f"{LI.sha256_file(self.ctx.approval_path)}  approval.json\n")
        self.assertEqual(RN.approval_for(self.ctx, "coverage").scope, ("band", "coverage"))
        with self.assertRaises(RN.Refused):
            RN.approval_for(self.ctx, "bca")


def synthetic_seed_data(seed, episodes=(30, 16, 10), length=10, test_episodes=10, sample_episodes=10):
    """A data.SeedData of random arrays in the real log format (not ThrusterChain data; world SYNTHETIC_WORLD):
    random states (current at index 10), random commands with 30% of rows at the 80% hold, random rewards; a ball
    around the hold; 5 bank rows per calibration episode; closed-form test masses."""
    from experiments.shift import shift as SH

    rng = np.random.default_rng([W.MASTER_SEED, 8888, int(seed)])
    hold = W.HoldRecorder().hold.astype(np.float32)

    def sim(n):
        S = np.concatenate([0.5 * rng.standard_normal((n, length + 1, 10)),
                            np.repeat(rng.uniform(W.C_LO, W.C_HI, (n, 1, 1)), length + 1, 1),
                            rng.uniform(-0.3, 0.3, (n, length + 1, 2))], 2)
        A = rng.uniform(-0.7, 0.7, (n, length, 4)).astype(np.float32)
        press = rng.uniform(size=(n, length)) < 0.3
        A[press] = hold
        R = -(S[:, :-1, :10] ** 2).sum(-1) - S[:, :-1, 10] + 0.1 * rng.standard_normal((n, length))
        return dict(states=S, commands=A, rewards=R, hard_water=press.copy(), hold=press)

    def make(s, law, split_sizes, bank_rows=()):
        log = D.make_log(s, seed, law, split_sizes, bank_rows)
        log.meta["world"] = LI.SYNTHETIC_WORLD
        return log

    tr, ca, ho = episodes
    bank = np.sort(np.concatenate([e * length + rng.choice(length, 5, replace=False) for e in range(tr, tr + ca)]))
    gap = SH.Gap(centre=tuple(float(v) for v in hold.astype(np.float64)), rho=0.05, mass=0.3, rule="base", q=0.15,
                 competing_ratio=2.0, record_sha256="synthetic")
    train_log = make(sim(tr + ca + ho), "train", episodes, bank)
    train_log = train_log.with_arrays(D.truth_columns(train_log, gap))
    tests = {}
    for t in ("mild", "strong"):
        log = make(sim(test_episodes), t, (0, 0, test_episodes))
        tests[t] = log.with_arrays(D.truth_columns(log, gap))
    samples = {}
    for name in D.SAMPLE_LAWS:
        s = sim(sample_episodes)
        samples[name] = dict(observations=s["states"][:, :-1].astype(np.float32).reshape(-1, 13),
                             actions=s["commands"].reshape(-1, 4),
                             episode=np.repeat(np.arange(sample_episodes, dtype=np.int32), length))
    held = train_log.split_rows("evaluation")
    rows = held[~train_log["truth/in_ball"][held]][:50]
    truth = dict(rows=rows, noise_map=rng.uniform(0.1, 1.0, len(rows)))
    world = dict(quadrature=dict(test_mass_gapped={t: dict(value=W.full_test_mass(t)) for t in W.TILT_NAMES},
                                 test_mass_full={t: W.full_test_mass(t) for t in W.TILT_NAMES}),
                 D1=D1_OK)
    return D.SeedData(int(seed), train_log, tests, samples, truth, world, gap)


class SyntheticPipelineTest(unittest.TestCase):
    """Every stage of run.py end to end on synthetic logs (random arrays in the real log format, world
    SYNTHETIC_WORLD): host, room, band, coverage, bca, evaluate and report, with tiny settings. The stage runner's
    approval check is replaced by the SYNTHETIC token, which learners_iw accepts only on synthetic data, and loading
    returns the synthetic logs, so nothing here touches ThrusterChain data."""

    @classmethod
    def setUpClass(cls):
        from unittest import mock

        cls.tmp = Path(tempfile.mkdtemp(prefix="shift_iw_pipeline_"))
        cls.spec, cls.digest = write_spec_copy(cls.tmp / "iw")
        cfg = LI.settings(updates=40, chunk=10, band_fit_steps=40, clone_steps=20, actor_only_steps=8,
                          actor_only_chunk=4, oracle_refresh=4, oracle_horizon=5, match_rounds=2, bca_warmup=20,
                          bca_refresh=10, strength_states=20, row_chunk=1024)
        cls.ctx = RN.Context(root=cls.tmp / "runs", spec_path=cls.spec, cfg=cfg, log=QUIET)
        cls.seeds = [0, 1]
        cls.data = {s: synthetic_seed_data(s) for s in cls.seeds}
        for s in cls.seeds:                      # the token opens BCA on exactly these arrays (content hashes)
            for removal in ("gap", "none"):
                LI.register_synthetic(D.prepare(cls.data[s], removal, cfg))
        put = lambda path, **f: RN.write_record(path, dict(status="finished", spec_sha256=cls.digest,
                                                           code_sha256=LI.code_hashes(), **f))
        for s in cls.seeds:
            put(RN.data_record_path(cls.ctx, s), D1=D1_OK, world_json_sha256="synthetic", quality="expert", seed=s)
        put(RN.banks_record_path(cls.ctx))
        (cls.ctx.banks_root / "start_banks.json").write_text("{}")   # the synthetic banks below stand in for it
        rng = np.random.default_rng([W.MASTER_SEED, 4343])
        banks = {t: W.draw_starts(4, rng, t) for t in W.TILT_NAMES}
        anchors = {t: dict(anchor=dict(random=-100.0, top=-50.0)) for t in W.TILT_NAMES}
        real_shape = LE.oracle_shape

        def forced_shape(*a, **k):    # check (e) computed, its verdict forced: seed 0 passes, seed 1 stops
            rec = real_shape(*a, **k)
            return dict(rec, passed=int(a[4]) == 0, verdict_forced_by_test=True)

        patches = [mock.patch.object(RN, "load_data", lambda ctx, seed, spec: cls.data[int(seed)]),
                   mock.patch.object(RN, "approval_for", lambda ctx, stage: LI.SYNTHETIC),
                   mock.patch.object(D, "load_start_banks", lambda root: (banks, dict(anchors=anchors))),
                   mock.patch.object(LE, "oracle_shape", forced_shape)]
        for p in patches:
            p.start()
        try:
            RN.stage_host(cls.ctx, cls.seeds)
            RN.stage_room(cls.ctx, cls.seeds)
            RN.stage_room_lower_edge(cls.ctx, cls.seeds)
            # the design's stop rule: the band waits for a report on every seed, by this code, saying D1 passed
            cls.band_refusals = {}
            for case, rep in dict(no_report=None,
                                  d1_fail=dict(seeds=list(RN.SEEDS), marks=dict(D1=dict(status="fail"))),
                                  two_seeds=dict(seeds=cls.seeds, marks=dict(D1=dict(status="pass"))),
                                  other_code=dict(seeds=list(RN.SEEDS), marks=dict(D1=dict(status="pass")),
                                                  code_sha256={"experiments/shift_iw/run.py": "0" * 64})).items():
                if rep is not None:
                    RN.write_record(RN.report_path(cls.ctx),
                                    dict(dict(status="finished", spec_sha256=cls.digest,
                                              code_sha256=LI.code_hashes()), **rep))
                try:
                    RN.stage_band(cls.ctx, cls.seeds)
                    cls.band_refusals[case] = None
                except RN.Refused as error:
                    cls.band_refusals[case] = str(error)
            cls.band_ran_before_design = (cls.ctx.root / "band").exists()
            put(RN.report_path(cls.ctx), seeds=list(RN.SEEDS), marks=dict(D1=dict(status="pass")))   # a fixture
            cls.design_report_sha256 = LI.sha256_file(RN.report_path(cls.ctx))
            RN.stage_band(cls.ctx, cls.seeds)
            RN.stage_coverage(cls.ctx, cls.seeds)
            put(RN.report_path(cls.ctx), seeds=list(RN.SEEDS),                                       # a fixture
                marks=dict(M1=dict(status="pass"), R1=dict(status="pass"), R2=dict(status="pass"),
                           L1=dict(status="fail"), L2=dict(status="pass")))
            RN.stage_bca(cls.ctx, [0])
            try:
                RN.stage_bca_lower_edge(cls.ctx, [0])
                cls.le_refusal = None
            except RN.Refused as error:
                cls.le_refusal = str(error)
            put(RN.report_path(cls.ctx), seeds=list(RN.SEEDS),                                       # a fixture
                marks=dict(M1=dict(status="pass"), R1=dict(status="pass"), R2=dict(status="pass"),
                           L1=dict(status="pass"), L2=dict(status="pass")))
            RN.stage_bca_lower_edge(cls.ctx, [0])
            RN.stage_evaluate(cls.ctx, cls.seeds)                    # hosts, room and room_lower_edge
            RN.stage_evaluate(cls.ctx, [0], ("bca", "bca_lower_edge"))
            cls.report = RN.stage_report(cls.ctx, cls.seeds)
        finally:
            for p in patches:
                p.stop()

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def rec(self, path):
        r = RN.finished(path, self.digest)
        self.assertIsNotNone(r, str(path))
        return r

    def test_band_waits_for_the_design_check(self):
        """The band stage refuses without a report, with D1 failed, with a report on fewer seeds than the
        specification's, and with a report made by other code; nothing is fitted before D1 passes."""
        self.assertFalse(self.band_ran_before_design)
        self.assertRegex(self.band_refusals["no_report"], "the first report")
        self.assertRegex(self.band_refusals["d1_fail"], "D1 is fail")
        self.assertRegex(self.band_refusals["two_seeds"], r"needs a report on seeds \[0, 1, 2, 3, 4\]")
        self.assertRegex(self.band_refusals["other_code"], "other code")
        for s in self.seeds:
            for kind in RN.HOST_KINDS:
                r = self.rec(RN.band_dir(self.ctx, kind, s) / "record.json")
                self.assertEqual(r["design_check"], dict(report_sha256=self.design_report_sha256, D1="pass"))

    def test_band_records(self):
        for s in self.seeds:
            for kind in RN.HOST_KINDS:
                r = self.rec(RN.band_dir(self.ctx, kind, s) / "record.json")
                self.assertTrue(r["band_valid"] and r["W0_unit_check"])
                self.assertEqual((r["quality"], r["seed"]), ("expert", s))
                self.assertLessEqual(r["M3"]["max_drop"], 1e-6)
                self.assertLessEqual(r["M3"]["max_closed_form_gap"], 1e-6)
                self.assertEqual(list(r["inputs_sha256"]), [f"expert/host/{kind}/seed{s}/record.json"] +
                                 ([f"expert/room/seed{s}/record.json"] if kind == "gap" else []))
                # L2's proxy check (gapped host only): BCA's unweighted lower edge against the oracle's at the room
                # check's no-caution actor, and check (e) on BCA's penalty; reported, before any lower-edge training
                p = r["lower_edge_proxy"]
                self.assertEqual(p is None, kind == "full")
                if p is not None:
                    self.assertFalse(p["skipped"])
                    self.assertEqual(p["rows"], r["M3"]["rows"])
                    self.assertTrue(p["band_mean"] >= 0 and p["oracle_mean"] >= 0)
                    if p["oracle_mean"] > 0:
                        self.assertAlmostEqual(p["band_over_oracle"], p["band_mean"] / p["oracle_mean"])
                    self.assertIn("passed", p["band_check_e"])
                    self.assertTrue(p["band_check_e"]["reported_only"])
                arms = ("exact", "estimated") if kind == "gap" else ("exact",)
                self.assertEqual(set(r["thresholds"]),
                                 {"unweighted"} | {f"{a}_{t}_single" for a in arms for t in W.TILT_NAMES})
                self.assertEqual(set(r["tables"]), {f"{a}_{t}" for a in arms for t in W.TILT_NAMES})
                self.assertEqual(r["thresholds"]["exact_none_single"]["threshold"],
                                 r["thresholds"]["unweighted"]["threshold"])
                self.assertAlmostEqual(r["doses"]["unweighted"]["change_vs_unweighted"], 0.0)
                self.assertEqual(r["classifiers"] is None, kind == "full")
                # per step: no tilt is the unweighted threshold; every table monotone; R2's per-row doses
                self.assertTrue(r["W0_per_step_none"])
                self.assertEqual(r["tables"]["exact_none"]["grid"], [1.0])
                self.assertEqual(r["doses_per_step"]["exact_none"]["max_abs_change"], 0.0)
                for name, tab in r["tables"].items():
                    self.assertTrue(tab["monotone"] and not tab["running_max_applied"], name)
                    self.assertLessEqual(tab["test_steps"]["lookup"]["max_mass_gap"] or 0.0, 0.01 + 1e-12)
                    self.assertLessEqual(tab["train_rows"]["lookup"]["max_mass_gap"], 0.01 + 1e-12)
                    self.assertEqual(r["doses_per_step"][name]["refused_rows"],
                                     round(tab["train_rows"]["refused_share"] * r["M3"]["rows"]))
                self.assertIn("high_current_change", r["doses_per_step"]["exact_strong"])
                # the lower edge's scale change per table (L2): R_i / R_u - 1 on kept training rows
                self.assertEqual(set(r["lower_edge_scale"]), set(r["tables"]))
                self.assertEqual(r["lower_edge_scale"]["exact_none"]["scale_change"], 0.0)
                le = r["lower_edge_scale"]["exact_strong"]
                self.assertTrue(math.isfinite(le["scale_change"]) and le["ratio_max"] >= le["ratio_min"] > 0)
                self.assertEqual(le["refused_rows"], r["doses_per_step"]["exact_strong"]["refused_rows"])

    def test_coverage_records(self):
        for s in self.seeds:
            for kind in RN.HOST_KINDS:
                r = self.rec(RN.coverage_dir(self.ctx, kind, s) / "record.json")
                self.assertEqual(set(r["tilts"]), set(W.TILT_NAMES))
                for t, row in r["tilts"].items():
                    self.assertEqual(set(row["arms"]), set(RN.TEST_ARMS[kind]))
                    for arm, a in row["arms"].items():
                        self.assertLessEqual(a["hard_coverage_mismatches"], 2)
                        g = a["groups"]
                        if arm in RN.PER_STEP_ARMS:     # a step whose own threshold is +inf is refused
                            self.assertEqual(a["threshold"], "per step")
                            outside = a["refused_by_threshold_share_certifiable"]
                        else:
                            outside = 0.0 if a["threshold"] != "inf" else 1.0
                        if kind == "gap" and arm != "unweighted":
                            self.assertEqual(g["inside"]["refused_share"], 1.0)
                            self.assertEqual(g["outside"]["refused_share"], outside)
                        else:
                            self.assertEqual(g["all"]["refused_share"], outside)
                self.assertTrue(r["tilts"]["none"]["arms"]["exact"]["W0_per_step_none_identical"])
                self.assertEqual(r["M2"] is None, kind == "full")
                with np.load(RN.coverage_dir(self.ctx, kind, s) / "steps.npz") as z:
                    self.assertIn("strong__unweighted__outcome", z.files)
                    self.assertIn("strong__exact__R", z.files)
                    self.assertEqual(len(z["strong__y"]), r["tilts"]["strong"]["steps"])
                    R = z["strong__exact__R"]
                    certified = np.ones(len(R), bool) if kind == "full" else ~z["strong__in_ball"]
                    self.assertTrue(np.all(np.isinf(R[~certified])))
                    Rc = R[certified][np.argsort(z["strong__current"][certified], kind="stable")]
                    self.assertTrue(np.all(Rc[1:] >= Rc[:-1]))      # a stronger current never gets a lower R
            self.assertEqual(self.rec(RN.coverage_dir(self.ctx, "gap", s) / "record.json")["M2"]["rows"], 50)

    def test_room_and_bca_records(self):
        for s in self.seeds:
            r = self.rec(RN.room_dir(self.ctx, s) / "record.json")
            self.assertTrue({"none", "placed", "even_x1.1", "even_x1.25", "even_x1.5"} <= set(r["room"]["names"]))
        u = self.rec(RN.bca_dir(self.ctx, "unweighted", 0) / "record.json")["result"]
        e = self.rec(RN.bca_dir(self.ctx, "exact_strong", 0) / "record.json")["result"]
        x1 = self.rec(RN.bca_dir(self.ctx, "exact_strong_single", 0) / "record.json")["result"]
        self.assertEqual([x["step"] for x in u["refreshes"]], [20, 30])
        self.assertIsNone(u["weighting"])
        self.assertEqual((e["weighting"]["name"], e["weighting"]["mode"]), ("exact strong (per row)", "per_row"))
        self.assertEqual(e["weighting"]["refused_dose"], 1.5)
        self.assertTrue(all("per_row" in x for x in e["refreshes"] if x["posterior_inputs_valid"]))
        self.assertEqual(len(e["refused_rows_per_refresh"]), 2)
        self.assertEqual(x1["weighting"]["mode"], "single")
        self.assertTrue(all("weighted" in x for x in x1["refreshes"] if x["posterior_inputs_valid"]))
        self.assertEqual(u["refreshes"][0]["unweighted_threshold"], e["refreshes"][0]["unweighted_threshold"])
        self.assertEqual(u["refreshes"][0]["unweighted_threshold"], x1["refreshes"][0]["unweighted_threshold"])

    def test_lower_edge_records(self):
        """The lower edge: check (e) stops seed 1 (no arm trained, no actor saved, evaluated as empty); seed 0 trains
        its oracle arms side by side from the room check's clone; the BCA arms wait for a report saying L1 passed,
        then run and record their thresholds and check (e) on their penalty."""
        r0 = self.rec(RN.room_le_dir(self.ctx, 0) / "record.json")
        r1 = self.rec(RN.room_le_dir(self.ctx, 1) / "record.json")
        self.assertTrue(r1["room_lower_edge"]["stopped"] and not r1["room_lower_edge"]["check_e"]["passed"])
        self.assertIsNone(r1["actors_sha256"])
        self.assertFalse((RN.room_le_dir(self.ctx, 1) / "actors.msgpack").exists())
        le = r0["room_lower_edge"]
        self.assertFalse(le["stopped"])
        self.assertTrue(le["clone_reproduces_room_check"])
        self.assertTrue({"le_none", "le_oracle", "le_oracle_x1.5", "le_even_x2", "le_even_x3"} <= set(le["names"]))
        self.assertEqual(le["k"], {"le_none": 0.0, "le_oracle": 1.0, "le_oracle_x1.5": 1.5})
        self.assertEqual(set(r0["inputs_sha256"]), {"expert/host/gap/seed0/record.json",
                                                    "expert/room/seed0/record.json"})
        self.assertRegex(self.le_refusal, "M1, L1 and L2 pass; the report says L1 is fail")
        band = self.rec(RN.band_dir(self.ctx, "gap", 0) / "record.json")
        for arm in RN.LE_ARMS:
            r = self.rec(RN.bca_le_dir(self.ctx, arm, 0) / "record.json")
            self.assertEqual(r["result"]["arms_in_this_job"], list(RN.LE_ARMS))
            refused = 0 if arm == "unweighted" else band["lower_edge_scale"][arm]["refused_rows"]
            self.assertEqual(r["result"]["arm"]["refused_rows"], refused, arm)
            self.assertIn("passed", r["result"]["arm"]["check_e"])
            self.assertEqual(r["result"]["k"], 1.0)
        u = self.rec(RN.bca_le_dir(self.ctx, "unweighted", 0) / "record.json")["result"]["arm"]
        self.assertAlmostEqual(u["R_mean"], RN.num(band["thresholds"]["unweighted"]["threshold"]), places=6)
        self.assertFalse(any((RN.bca_le_dir(self.ctx, a, s) / "record.json").exists()
                             for a in RN.LE_OPTIONAL_ARMS for s in self.seeds))
        ev0 = self.rec(RN.evaluate_path(self.ctx, 0, "room_lower_edge"))["evaluations"]
        self.assertTrue({"le_none", "le_oracle", "le_oracle_x1.5"} <= set(ev0))
        self.assertEqual(self.rec(RN.evaluate_path(self.ctx, 1, "room_lower_edge"))["evaluations"], {})
        self.assertEqual(set(self.rec(RN.evaluate_path(self.ctx, 0, "bca_lower_edge"))["evaluations"]),
                         {f"le_bca_{a}" for a in RN.LE_ARMS})
        marks = self.report["marks"]
        for k in ("L1", "L2"):            # seed 1 stopped: they fail, shown as pending on two seeds of five
            self.assertEqual((marks[k]["status"], marks[k]["preview_status"]), ("pending", "fail"), k)
            self.assertEqual(marks[k]["stopped_seeds"], [1], k)
        self.assertIn("seed 1:", marks["L1"]["preview_why"])          # the stop names the failed sub-checks
        self.assertIn("check (e) on the oracle penalty failed (", r1["room_lower_edge"]["why"])
        proxy = marks["L2"]["proxy_check"]                             # shown beside L2 even when it fails
        self.assertEqual(len(proxy["band_check_e_passed"]), len(self.seeds))
        self.assertEqual(proxy["band_mean"], [self.rec(RN.band_dir(self.ctx, "gap", s) / "record.json")
                                              ["lower_edge_proxy"]["band_mean"] for s in self.seeds])
        self.assertEqual(marks["L3"]["status"], "not run")
        self.assertFalse(self.report["bca_lower_edge_allowed"])
        self.assertEqual(set(self.report["shown"]["lower_edge"]["bca_records"]), set(RN.LE_ARMS))

    def test_evaluations_and_report(self):
        ev = self.rec(RN.evaluate_path(self.ctx, 0, "hosts"))["evaluations"]
        self.assertEqual(set(ev), {"plain_gap", "plain_full"})
        self.assertEqual(set(ev["plain_gap"]), set(W.TILT_NAMES))
        self.assertIn("placed", self.rec(RN.evaluate_path(self.ctx, 1, "room"))["evaluations"])
        self.assertEqual(set(self.rec(RN.evaluate_path(self.ctx, 0, "bca"))["evaluations"]),
                         {"bca_unweighted", "bca_exact_strong", "bca_exact_strong_single"})
        marks = self.report["marks"]
        self.assertFalse(self.report["complete"])                       # two seeds of five: nothing is judged
        self.assertIn(marks["D1"]["status"], ("pass", "fail"))
        for k in ("M1", "M2", "M3", "W0", "W1", "W2", "W3", "W4", "R1", "R2"):
            self.assertEqual(marks[k]["status"], "pending", k)
            self.assertIn(marks[k]["preview_status"], ("pass", "fail"), k)   # fixtures: computed, not judged
        self.assertIn(marks["I1"]["status"], ("not run", "pending"))
        self.assertFalse(self.report["bca_allowed"])
        self.assertEqual(set(self.rec(RN.evaluate_path(self.ctx, 0, "bca"))["inputs_sha256"]),
                         {"expert/bca/unweighted/seed0/record.json", "expert/bca/exact_strong/seed0/record.json",
                          "expert/bca/exact_strong_single/seed0/record.json", "start_banks/stage.json"})
        self.assertEqual(set(self.report["shown"]["bca_records"]),
                         {"unweighted", "exact_strong", "exact_strong_single"})
        rounded = self.report["shown"]["possibly_rounded_into_refusal_outside_ball"]
        self.assertEqual(set(rounded), {f"{a}_{t}" for a in RN.PER_STEP_ARMS for t in W.TILT_NAMES})
        self.assertTrue(all(v is not None and len(v) == len(self.seeds) and min(v) >= 0 for v in rounded.values()))
        self.assertIsNotNone(self.rec(RN.report_path(self.ctx)))


class ReportArithmeticTest(unittest.TestCase):
    """The pass-mark rules of build_report on hand-written fixture records (every number is a fixture: nothing is
    trained, fitted or evaluated here, and this root holds no data, so a valid approval may be written in it)."""

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="shift_iw_report_"))
        self.spec, self.digest = write_spec_copy(self.tmp / "iw")
        self.ctx = RN.Context(root=self.tmp / "runs", spec_path=self.spec, log=QUIET)
        self.seeds = [0, 1, 2, 3, 4]

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def put(self, path, **fields):
        """A finished fixture record; like the stages' records it names its quality and seed, read from its folder
        (<quality>/.../seed<k>/...) unless given."""
        parts = Path(path).relative_to(self.ctx.root).parts
        if parts[0] != "start_banks":
            fields.setdefault("quality", parts[0])
            seed = [int(p[4:]) for p in parts if p.startswith("seed") and p[4:].isdigit()]
            if seed:
                fields.setdefault("seed", seed[0])
        RN.write_record(path, dict(status="finished", spec_sha256=self.digest, code_sha256=LI.code_hashes(), **fields))
        return path

    @staticmethod
    def entry(points):
        """An evaluation entry: points and a return J = 0.6 points - 80 (1 point = 0.6 return units, about version 4's
        scale under strong; a fixture)."""
        return dict(points=points, J=0.6 * points - 80.0)

    def fixtures(self, unweighted_strong=0.85, dose_change=0.01, placed_gain=5.0, held_out=0.915, in_ball_steps=4000,
                 le_gain=5.0, le_slope=4.0, le_scale=0.1, le_stopped=()):
        """Fixture records of every stage. The lower edge (part B): oracle minus no penalty le_gain return units
        (jittered), return slope le_slope per unit of k above k = 1, scale change le_scale (so L2's prediction is
        le_slope x le_scale), seeds le_stopped stopped by check (e)."""
        jitter = [-0.004, -0.002, 0.0, 0.002, 0.004]
        banks = self.put(RN.banks_record_path(self.ctx))
        for i, s in enumerate(self.seeds):
            j = jitter[i]
            self.put(RN.data_record_path(self.ctx, s), D1=D1_OK)
            hosts = {}
            for kind in RN.HOST_KINDS:
                host = hosts[kind] = self.put(RN.host_dir(self.ctx, kind, s) / "record.json", result=dict(failed=False))
                band = self.put(RN.band_dir(self.ctx, kind, s) / "record.json", band_valid=True, W0_unit_check=True,
                                W0_per_step_none=True,
                                M3=dict(rows=42_500, inversions=3, max_drop=1.2e-7, max_closed_form_gap=1e-7,
                                        supported_all=True, threshold_finite=True),
                                thresholds={n: dict(ratio_to_unweighted=1.1) for n in
                                            [f"{a}_{t}_single" for a in ("exact", "estimated") for t in W.TILT_NAMES]},
                                tables={f"{a}_{t}": dict(test_steps=dict(ratio_min=1.05, ratio_mean=1.1, ratio_max=1.15))
                                        for a in ("exact", "estimated") for t in W.TILT_NAMES},
                                doses=dict(exact_strong_single=dict(change_vs_unweighted=dose_change + 0.001 + j / 10)),
                                doses_per_step=dict(exact_strong=dict(change_vs_unweighted=dose_change + j / 10,
                                                                      mean_abs_change=dose_change + 0.002,
                                                                      high_current_change=dose_change + 0.003,
                                                                      refused_rows=0)),
                                lower_edge_scale=dict(
                                    exact_strong=dict(scale_change=le_scale + j / 10, high_current_scale_change=0.2,
                                                      ratio_max=1.3, refused_rows=0, largest_finite_threshold=2.5),
                                    exact_mild=dict(scale_change=le_scale / 2, high_current_scale_change=0.1,
                                                    ratio_max=1.1, refused_rows=0, largest_finite_threshold=2.0)),
                                classifiers={t: dict(compare=dict(slope_on_c=W.ratio_slope(t) + j))
                                             for t in W.TILT_NAMES},
                                lower_edge_proxy=dict(skipped=False, band_over_oracle=0.4 + j, pearson=0.3,
                                                      spearman=0.35, band_mean=0.01, oracle_mean=0.025,
                                                      band_check_e=dict(passed=s != 4, failed=[] if s != 4 else
                                                                        ["e2: its mean does not rise"]))
                                if kind == "gap" else None,
                                inputs_sha256=RN.links(self.ctx, [host]))
                tilts = {}
                for t in W.TILT_NAMES:
                    unw = {"none": held_out, "mild": 0.90, "strong": unweighted_strong}[t] + j
                    g = lambda cov, inside_refused, outside_refused, inside_cov=0.5: dict(
                        outside=dict(coverage=cov, refused_share=outside_refused),
                        inside=dict(coverage=inside_cov if inside_refused == 0 else 0.0, refused_share=inside_refused),
                        all=dict(coverage=cov))
                    tilts[t] = dict(in_ball_share=0.2, in_ball_steps=in_ball_steps, arms=dict(
                        unweighted=dict(groups=g(unw, 0.0, 0.0)),
                        exact=dict(groups=g(0.925 + j, 1.0, 0.0), refused_by_threshold_share_certifiable=0.0,
                                   **({"W0_per_step_none_identical": True} if t == "none" else {})),
                        estimated=dict(groups=g(0.92 + j, 1.0, 0.001), refused_by_threshold_share_certifiable=0.001),
                        exact_single=dict(groups=g(0.924 + j, 1.0, 0.0)),
                        estimated_single=dict(groups=g(0.919 + j, 1.0, 0.0))))
                self.put(RN.coverage_dir(self.ctx, kind, s) / "record.json", tilts=tilts,
                         forecast=dict(values=dict(mild=0.9, strong=0.86)),
                         M2=dict(error=dict(within_deciles=0.2 + j, pooled=0.5),
                                 noise_map=dict(within_deciles=0.1, pooled=0.6)),
                         inputs_sha256=RN.links(self.ctx, [band, host]))
            room = self.put(RN.room_dir(self.ctx, s) / "record.json", room=dict(match=dict(multiplier=1.4)),
                            inputs_sha256=RN.links(self.ctx, [hosts["gap"]]))
            pts = {"none": 10.0 + j, "placed": 10.0 + placed_gain + 10 * j, "even_x1.1": 11.0 + j,
                   "even_x1.25": 12.5 + j, "even_x1.5": 15.0 + j, LI.MATCHED: 14.0}
            self.put(RN.evaluate_path(self.ctx, s, "room"),
                     evaluations={k: {law: self.entry(v) for law in W.TILT_NAMES} for k, v in pts.items()},
                     inputs_sha256=RN.links(self.ctx, [room, banks]))
            self.put(RN.evaluate_path(self.ctx, s, "hosts"),
                     evaluations={"plain_gap": {law: self.entry(20.0) for law in W.TILT_NAMES},
                                  "plain_full": {law: self.entry(25.0 + j) for law in W.TILT_NAMES}},
                     inputs_sha256=RN.links(self.ctx, [hosts["gap"], hosts["full"], banks]))
            stopped = s in le_stopped
            rle = self.put(RN.room_le_dir(self.ctx, s) / "record.json", room_lower_edge=dict(
                stopped=stopped, names=[] if stopped else ["le_none", "le_oracle", "le_oracle_x1.5", LE.MATCHED],
                check_e=dict(passed=not stopped, rows=2000), check_b=dict(max_param_diff=1e-4),
                strengths={"le_none": 0.001}, clone_reproduces_room_check=True),
                inputs_sha256=RN.links(self.ctx, [hosts["gap"], room]))
            J = {} if stopped else {"le_none": -70.0 + j, "le_oracle": -70.0 + le_gain + 10 * j,
                                    "le_oracle_x1.5": -70.0 + le_gain + 10 * j + 0.5 * le_slope, LE.MATCHED: -69.0}
            self.put(RN.evaluate_path(self.ctx, s, "room_lower_edge"),
                     evaluations={k: {law: dict(J=v, points=(v + 80.0) / 0.6) for law in W.TILT_NAMES}
                                  for k, v in J.items()},
                     inputs_sha256=RN.links(self.ctx, [rle, banks]))

    def statuses(self, body):
        return {k: m["status"] for k, m in body["marks"].items()}

    def test_marks_when_weighting_moves_the_dose_too_little(self):
        self.fixtures()
        body = RN.build_report(self.ctx, self.seeds, self.digest)
        self.assertEqual(self.statuses(body), dict(D1="pass", M1="pass", M2="pass", M3="pass", W0="pass", W1="pass",
                                                   W2="pass", W3="pass", W4="pass", R1="pass", R2="fail",
                                                   I1="not run", L1="pass", L2="fail", L3="not run"))
        self.assertTrue(body["complete"])
        self.assertEqual(body["quality"], "expert")
        self.assertAlmostEqual(body["marks"]["R2"]["points_per_unit_dose"]["mean"], 10.0, places=6)
        self.assertAlmostEqual(body["marks"]["R2"]["predicted_points"], 0.1, places=6)          # per step
        self.assertAlmostEqual(body["marks"]["R2"]["return_per_unit_dose"]["mean"], 6.0, places=6)
        self.assertAlmostEqual(body["marks"]["R2"]["predicted_return"], 0.06, places=6)         # what R2 judges
        self.assertAlmostEqual(body["marks"]["R2"]["single_threshold"]["predicted_return"], 0.066, places=6)
        self.assertAlmostEqual(body["marks"]["R1"]["placed_minus_none"]["mean"], 3.0, places=6)   # return units
        self.assertAlmostEqual(body["marks"]["R1"]["placed_minus_none_points"]["mean"], 5.0, places=6)
        self.assertEqual(body["design"]["0"]["D1"], D1_OK)
        self.assertAlmostEqual(body["marks"]["R2"]["high_current_dose_change"]["mean"], 0.013, places=9)
        self.assertEqual(body["marks"]["R2"]["refused_rows"], [0] * 5)
        self.assertAlmostEqual(body["marks"]["W1"]["single_threshold_coverage"]["strong"]["mean"], 0.924, places=9)
        self.assertIn("confirmed by the owner", body["marks"]["R1"]["gate_choice"])
        self.assertFalse(body["bca_allowed"])
        self.assertAlmostEqual(body["shown"]["full_minus_gapped_plain_td3bc_strong"]["mean"], 5.0)
        self.assertAlmostEqual(body["shown"]["full_minus_gapped_plain_td3bc_strong_return"]["mean"], 3.0)
        self.assertAlmostEqual(body["marks"]["R1"]["placed_minus_matched_even"]["mean"], 0.6)
        self.assertEqual(body["marks"]["R1"]["matched_even_found_per_seed"], [True] * 5)
        self.assertIsNone(body["marks"]["R1"]["matched_even_note"])

    def test_marks_when_the_shift_does_not_break_the_band(self):
        self.fixtures(unweighted_strong=0.905, dose_change=0.4, placed_gain=1.0)
        body = RN.build_report(self.ctx, self.seeds, self.digest)
        self.assertEqual(body["marks"]["W2"]["status"], "fail")
        self.assertEqual(body["marks"]["R1"]["status"], "fail")
        self.assertEqual(body["marks"]["R2"]["status"], "pass")
        self.assertEqual(body["marks"]["I1"]["status"], "not run")
        self.assertFalse(body["bca_allowed"])

    def test_bca_allowed_only_when_m1_r1_and_r2_pass(self):
        self.fixtures(dose_change=0.4)
        body = RN.build_report(self.ctx, self.seeds, self.digest)
        self.assertEqual([body["marks"][k]["status"] for k in ("M1", "R1", "R2")], ["pass"] * 3)
        self.assertTrue(body["bca_allowed"])
        self.assertEqual(body["marks"]["I1"]["status"], "pending")
        self.fixtures(dose_change=0.4, held_out=0.85)
        body = RN.build_report(self.ctx, self.seeds, self.digest)
        self.assertEqual(body["marks"]["M1"]["status"], "fail")
        self.assertFalse(body["bca_allowed"])

    def rewrite(self, path, change):
        rec = RN.read_record(path)
        change(rec)
        self.put(path, **{k: v for k, v in rec.items() if k not in ("status", "spec_sha256", "code_sha256")})

    def test_per_step_refusals_and_the_no_tilt_identity(self):
        """W3: outside the ball the exact arm refuses no step, and the estimated arm refuses exactly the steps whose
        own threshold is +inf. W0: under no tilt the per-step arm must equal the unweighted arm."""
        self.fixtures()
        path = RN.coverage_dir(self.ctx, "gap", 3) / "record.json"
        self.rewrite(path, lambda r: r["tilts"]["strong"]["arms"]["exact"].update(
            refused_by_threshold_share_certifiable=0.002,
            groups=dict(r["tilts"]["strong"]["arms"]["exact"]["groups"],
                        outside=dict(coverage=0.92, refused_share=0.002))))
        self.assertEqual(RN.build_report(self.ctx, self.seeds, self.digest)["marks"]["W3"]["status"], "fail")
        self.fixtures()
        self.rewrite(path, lambda r: r["tilts"]["mild"]["arms"]["estimated"].update(
            refused_by_threshold_share_certifiable=0.0))           # refused 0.001 outside, but none by threshold
        self.assertEqual(RN.build_report(self.ctx, self.seeds, self.digest)["marks"]["W3"]["status"], "fail")
        self.fixtures()
        self.assertEqual(RN.build_report(self.ctx, self.seeds, self.digest)["marks"]["W3"]["status"], "pass")
        self.rewrite(RN.coverage_dir(self.ctx, "full", 1) / "record.json", lambda r: r["tilts"]["none"]["arms"][
            "exact"].update(W0_per_step_none_identical=False))
        w0 = RN.build_report(self.ctx, self.seeds, self.digest)["marks"]["W0"]
        self.assertEqual(w0["status"], "fail")
        self.assertEqual(w0["per_seed"]["full"]["per_step_coverage_identical"], [True, False, True, True, True])

    def test_a_missing_matched_arm_is_flagged(self):
        self.fixtures()
        for s in (1, 3):                                        # the placed arm was stronger than every rung
            rec = RN.read_record(RN.evaluate_path(self.ctx, s, "room"))
            del rec["evaluations"][LI.MATCHED]
            self.put(RN.evaluate_path(self.ctx, s, "room"), **{k: v for k, v in rec.items()
                                                               if k not in ("status", "spec_sha256", "code_sha256")})
        r1 = RN.build_report(self.ctx, self.seeds, self.digest)["marks"]["R1"]
        self.assertEqual(r1["status"], "pass")                  # R1 gates on placed - none (spec)
        self.assertIsNone(r1["placed_minus_matched_even"])
        self.assertEqual(r1["matched_even_found_per_seed"], [True, False, True, False, True])
        self.assertIn("not found", r1["matched_even_note"])

    def test_the_numbers_come_from_the_frozen_spec(self):
        self.assertFalse(any(hasattr(RN, n) for n in ("M1_POOLED", "W1_POOLED", "W2_BELOW", "W4_GAP", "R1_POINTS",
                                                      "R2_POINTS")))
        spec = json.loads(self.spec.read_text())
        th = {m["id"]: m["thresholds"] for m in spec["pass_marks"]}
        self.fixtures()
        self.assertEqual(RN.build_report(self.ctx, self.seeds, self.digest)["thresholds"], th)
        for m in spec["pass_marks"]:              # another specification: stricter W2 and M3
            if m["id"] == "W2":
                m["thresholds"]["unweighted_pooled_below"] = 0.84
            if m["id"] == "M3":
                m["thresholds"]["max_dose_drop"] = 0.0
        self.spec.write_text(json.dumps(spec, indent=2))
        self.digest = LI.sha256_file(self.spec)
        LI.sidecar(self.spec).write_text(f"{self.digest}  spec.json\n")
        with self.assertRaisesRegex(RN.Refused, "not finished under this specification"):
            RN.stage_report(self.ctx, self.seeds)                 # the old records do not count
        self.fixtures()
        st = self.statuses(RN.build_report(self.ctx, self.seeds, self.digest))
        self.assertEqual((st["W2"], st["M3"], st["W1"]), ("fail", "fail", "pass"))

    def test_records_made_by_other_code_are_refused(self):
        self.fixtures()
        path = RN.coverage_dir(self.ctx, "gap", 2) / "record.json"
        rec = RN.read_record(path)
        rec["code_sha256"] = dict(rec["code_sha256"], **{"experiments/shift_iw/run.py": "0" * 64})
        RN.write_record(path, rec)
        with self.assertRaisesRegex(RN.Refused, r"other code.*coverage gap 2"):
            RN.build_report(self.ctx, self.seeds, self.digest)

    def test_a_broken_record_chain_is_refused(self):
        self.fixtures()
        self.put(RN.band_dir(self.ctx, "gap", 1) / "record.json", band_valid=True, note="rewritten after coverage")
        with self.assertRaisesRegex(RN.Refused, r"no longer match.*coverage gap 1: expert/band/gap/seed1/record.json"):
            RN.build_report(self.ctx, self.seeds, self.digest)
        self.fixtures()
        self.put(RN.host_dir(self.ctx, "gap", 4) / "record.json", result=dict(failed=False), note="another host")
        with self.assertRaisesRegex(RN.Refused, "room 4: expert/host/gap/seed4/record.json"):
            RN.build_report(self.ctx, self.seeds, self.digest)

    def test_fewer_seeds_leave_the_cross_seed_marks_pending(self):
        self.fixtures(dose_change=0.4)
        body = RN.build_report(self.ctx, [0, 1], self.digest)
        self.assertFalse(body["complete"])
        for k in ("M1", "R1", "R2", "W1", "W2"):
            self.assertEqual(body["marks"][k]["status"], "pending", k)
            self.assertEqual(body["marks"][k]["preview_status"], "pass", k)
        self.assertEqual(body["marks"]["I1"]["status"], "not run")
        self.assertFalse(body["bca_allowed"])

    def test_an_invalid_band_fails_its_marks(self):
        self.fixtures()
        host = RN.host_dir(self.ctx, "gap", 2) / "record.json"
        self.put(RN.band_dir(self.ctx, "gap", 2) / "record.json", band_valid=False, seed=2, host="gap",
                 inputs_sha256=RN.links(self.ctx, [host]))
        for p in (RN.coverage_dir(self.ctx, "gap", 2) / "record.json",):   # coverage refuses on an invalid band
            p.unlink()
            LI.sidecar(p).unlink()
        body = RN.build_report(self.ctx, self.seeds, self.digest)
        for k in RN.BAND_DEPENDENT:
            self.assertEqual(body["marks"][k]["status"], "fail", k)
            self.assertIn("no valid refresh on seeds [2]", body["marks"][k]["why"])
        self.assertEqual(body["marks"]["R1"]["status"], "pass")
        self.assertFalse(body["bca_allowed"])

    def test_stage_bca_needs_a_complete_report_by_this_code(self):
        self.fixtures(dose_change=0.4)
        write_approval(self.ctx.approval_path, self.digest, scope=("band", "coverage", "bca"))
        RN.stage_report(self.ctx, [0, 1])
        with self.assertRaisesRegex(RN.Refused, r"need a report on seeds \[0, 1, 2, 3, 4\]"):
            RN.stage_bca(self.ctx, self.seeds)
        self.fixtures(dose_change=0.4, held_out=0.85)
        RN.stage_report(self.ctx, self.seeds)
        with self.assertRaisesRegex(RN.Refused, "M1 is fail"):
            RN.stage_bca(self.ctx, self.seeds)
        rec = RN.read_record(RN.report_path(self.ctx))
        rec["marks"]["M1"]["status"] = "pass"
        rec["code_sha256"] = {"old": "code"}
        RN.write_record(RN.report_path(self.ctx), rec)
        with self.assertRaisesRegex(RN.Refused, "report was made by other code"):
            RN.stage_bca(self.ctx, self.seeds)

    # ---------------------------------------------------------------------------------------- spec version 5

    def test_records_live_under_their_quality(self):
        """Per-quality layout: every record of a quality is under <root>/<quality> (the start banks under
        <root>/start_banks), the approval is shared, and no stage reads another quality's records: the same records
        give another quality's report nothing but 'pending'."""
        self.fixtures()
        files = [p.relative_to(self.ctx.root).parts[0] for p in self.ctx.root.rglob("*") if p.is_file()]
        self.assertEqual(set(files), {"expert", "start_banks"})
        other = self.ctx.at("medium")
        self.assertEqual(other.approval_path, self.ctx.approval_path)
        self.assertEqual(self.ctx.approval_path, self.ctx.root / "approval.json")
        self.assertEqual(other.banks_root, self.ctx.banks_root)
        self.assertEqual(RN.host_dir(other, "gap", 0), self.ctx.root / "medium" / "host" / "gap" / "seed0")
        body = RN.build_report(other, self.seeds, self.digest)
        self.assertEqual(body["quality"], "medium")
        self.assertTrue(all(m["status"] in ("pending", "not run") for m in body["marks"].values()))
        with self.assertRaisesRegex(RN.Refused, "data for seed 0"):
            RN.stage_host(other, [0])
        with self.assertRaises(RN.Refused):
            RN.Context(root=self.ctx.root, spec_path=self.spec, quality="novice")
        self.assertEqual(RN.main(["host", "--seeds", "0", "--root", str(self.ctx.root), "--spec", str(self.spec)]),
                         2)   # a per-quality stage without --quality is refused

    def set_room_returns(self, placed_minus_none):
        """Rewrite every seed's room evaluation so that placed minus none is the given return (jittered by +-0.01
        across seeds, mean exact)."""
        for s, j in zip(self.seeds, [-0.01, -0.005, 0.0, 0.005, 0.01]):
            path = RN.evaluate_path(self.ctx, s, "room")
            rec = RN.read_record(path)
            for law in W.TILT_NAMES:
                rec["evaluations"]["placed"][law]["J"] = rec["evaluations"]["none"][law]["J"] + placed_minus_none + j
            self.put(path, **{k: v for k, v in rec.items() if k not in ("status", "spec_sha256", "code_sha256")})

    def test_return_marks_read_return_units_from_the_spec(self):
        """R1 and R2 judge return units against spec.json's 1.8: 1.79 fails and 1.81 passes."""
        spec = json.loads(self.spec.read_text())
        th = {m["id"]: m["thresholds"] for m in spec["pass_marks"]}
        self.assertEqual((th["R1"]["min_return"], th["R2"]["min_abs_predicted_return"]), (1.8, 1.8))
        for value, want in ((1.79, "fail"), (1.81, "pass")):
            self.fixtures()
            self.set_room_returns(value)
            r1 = RN.build_report(self.ctx, self.seeds, self.digest)["marks"]["R1"]
            self.assertEqual(r1["status"], want, value)
            self.assertAlmostEqual(r1["placed_minus_none"]["mean"], value, places=9)
            self.fixtures(dose_change=value / 6.0)            # the even arms' return slope is 6 per unit of dose
            r2 = RN.build_report(self.ctx, self.seeds, self.digest)["marks"]["R2"]
            self.assertAlmostEqual(r2["predicted_return"], value, places=9)
            self.assertEqual(r2["status"], want, value)

    def test_gap_comparison_needs_enough_steps_inside_the_ball(self):
        """W3: with fewer than 500 in-ball steps on a seed the inside-versus-outside clause is 'not judged', while the
        refusal clauses are still judged; with enough steps it is judged."""
        def inside_above_outside(r):          # unweighted coverage inside the ball above outside, every tilt
            for t in W.TILT_NAMES:
                r["tilts"][t]["arms"]["unweighted"]["groups"]["inside"]["coverage"] = 0.99

        for steps, want in ((499, "pass"), (500, "fail")):
            self.fixtures(in_ball_steps=steps)
            for s in self.seeds:
                self.rewrite(RN.coverage_dir(self.ctx, "gap", s) / "record.json", inside_above_outside)
            w3 = RN.build_report(self.ctx, self.seeds, self.digest)["marks"]["W3"]
            self.assertEqual(w3["status"], want, steps)
            judged = steps >= 500
            for t in W.TILT_NAMES:
                u = w3["tilts"][t]["unweighted"]
                self.assertEqual(u["judged"], judged)
                self.assertEqual(u["note"] is None, judged)
                if not judged:
                    self.assertIn("too few steps inside the ball", u["note"])
        self.fixtures(in_ball_steps=100)          # not judged, but a refusal clause still fails the mark
        path = RN.coverage_dir(self.ctx, "gap", 2) / "record.json"
        self.rewrite(path, lambda r: r["tilts"]["strong"]["arms"]["exact"]["groups"]["inside"].update(
            refused_share=0.99))
        self.assertEqual(RN.build_report(self.ctx, self.seeds, self.digest)["marks"]["W3"]["status"], "fail")

    def test_improvement_records_its_p_value_for_the_summary(self):
        """I1 (after R1 and R2 pass): the report records the paired difference in return and its one-sided p-value;
        the verdict is the summary's (Holm), so the report says 'pending'."""
        self.fixtures(dose_change=0.4)
        banks = RN.banks_record_path(self.ctx)
        for s, d in zip(self.seeds, [0.5, 1.0, 1.5, 2.0, 2.5]):
            arms = {"bca_unweighted": -90.0, "bca_exact_strong": -90.0 + d, "bca_exact_strong_single": -90.0 + d / 2}
            recs = []
            for a in RN.BCA_ARMS:
                recs.append(self.put(RN.bca_dir(self.ctx, a, s) / "record.json", result=dict()))
            self.put(RN.evaluate_path(self.ctx, s, "bca"),
                     evaluations={k: {law: dict(J=v, points=(v + 80.0) / 0.6) for law in W.TILT_NAMES}
                                  for k, v in arms.items()},
                     inputs_sha256=RN.links(self.ctx, recs + [banks]))
        i1 = RN.build_report(self.ctx, self.seeds, self.digest)["marks"]["I1"]
        self.assertEqual((i1["status"], i1["ran"], i1["uncorrected_status"]), ("pending", True, "pass"))
        self.assertAlmostEqual(i1["exact_strong_minus_unweighted"]["mean"], 1.5)
        from scipy import stats
        t = 1.5 / (np.std([0.5, 1.0, 1.5, 2.0, 2.5], ddof=1) / math.sqrt(5))
        self.assertAlmostEqual(i1["p_value"], float(stats.t.sf(t, 4)), places=12)
        self.assertEqual(RN.p_one_sided([1.0, 1.0, 1.0]), 0.0)
        self.assertIsNone(RN.p_one_sided([1.0, float("nan"), 2.0]))

    # ---------------------------------------------------------------------------------------- the lower edge (part B)

    def test_lower_edge_marks_read_return_units_from_the_spec(self):
        """L1 (oracle minus no penalty) and L2 (slope above k = 1 times the scale change) against spec.json's 1.8:
        1.79 fails and 1.81 passes; L2's arithmetic and what it reports."""
        spec = json.loads(self.spec.read_text())
        th = {m["id"]: m["thresholds"] for m in spec["pass_marks"]}
        self.assertEqual((th["L1"]["min_return"], th["L2"]["min_abs_predicted_return"]), (1.8, 1.8))
        for value, want in ((1.79, "fail"), (1.81, "pass")):
            self.fixtures(le_gain=value)
            l1 = RN.build_report(self.ctx, self.seeds, self.digest)["marks"]["L1"]
            self.assertEqual(l1["status"], want, value)
            self.assertAlmostEqual(l1["oracle_minus_none"]["mean"], value, places=9)
            self.fixtures(le_slope=value / 0.1, le_scale=0.1)
            l2 = RN.build_report(self.ctx, self.seeds, self.digest)["marks"]["L2"]
            self.assertAlmostEqual(l2["predicted_return"], value, places=9)
            self.assertEqual(l2["status"], want, value)
        self.assertAlmostEqual(l2["return_per_unit_k"]["mean"], 18.1, places=9)
        self.assertEqual((l2["slope_arms"], l2["slope_k"]), (["le_oracle", "le_oracle_x1.5"], [1.0, 1.5]))
        self.assertAlmostEqual(l2["scale_change"]["mean"], 0.1, places=12)
        self.assertAlmostEqual(l2["mild_scale_change"]["mean"], 0.05, places=12)
        self.assertEqual(l2["refused_rows"], [0] * 5)
        self.assertAlmostEqual(l2["predicted_points"], 1.81 / 0.6, places=9)
        proxy = l2["proxy_check"]               # reported beside L2 (no rule): BCA's effective k and check (e)
        self.assertAlmostEqual(proxy["band_over_oracle"]["mean"], 0.4, places=12)
        self.assertEqual(proxy["band_check_e_passed"], [True, True, True, True, False])
        self.assertIn("failed on seeds [4]", proxy["note"])
        self.fixtures(le_gain=-5.0)
        self.assertEqual(RN.build_report(self.ctx, self.seeds, self.digest)["marks"]["L1"]["status"], "fail")

    def test_check_e_stops_fail_the_lower_edge_marks(self):
        self.fixtures(le_slope=40.0, le_stopped=(2,))
        body = RN.build_report(self.ctx, self.seeds, self.digest)
        for k in ("L1", "L2"):
            self.assertEqual(body["marks"][k]["status"], "fail", k)
            self.assertIn("check (e)", body["marks"][k]["why"])
            self.assertEqual(body["marks"][k]["stopped_seeds"], [2])
        self.assertEqual(body["marks"]["L3"]["status"], "not run")
        self.assertFalse(body["bca_lower_edge_allowed"])

    def test_lower_edge_scale_needs_a_valid_band(self):
        self.fixtures(le_slope=40.0)
        host = RN.host_dir(self.ctx, "gap", 2) / "record.json"
        self.put(RN.band_dir(self.ctx, "gap", 2) / "record.json", band_valid=False, seed=2, host="gap",
                 inputs_sha256=RN.links(self.ctx, [host]))
        for p in (RN.coverage_dir(self.ctx, "gap", 2) / "record.json",):
            p.unlink()
            LI.sidecar(p).unlink()
        marks = RN.build_report(self.ctx, self.seeds, self.digest)["marks"]
        self.assertEqual(marks["L2"]["status"], "fail")
        self.assertIn("no valid refresh on seeds [2]", marks["L2"]["why"])
        self.assertEqual(marks["L1"]["status"], "pass")          # the oracle needs no band

    def test_lower_edge_bca_allowed_only_when_m1_l1_and_l2_pass(self):
        self.fixtures(le_slope=40.0)
        body = RN.build_report(self.ctx, self.seeds, self.digest)
        self.assertEqual([body["marks"][k]["status"] for k in ("M1", "L1", "L2")], ["pass"] * 3)
        self.assertTrue(body["bca_lower_edge_allowed"])
        self.assertFalse(body["bca_allowed"])                    # R2 fails: version 4's BCA arms stay off
        self.assertEqual(body["marks"]["L3"]["status"], "pending")
        write_approval(self.ctx.approval_path, self.digest, scope=("band", "coverage", "bca_lower_edge"))
        self.fixtures(le_slope=40.0, held_out=0.85)
        RN.stage_report(self.ctx, self.seeds)
        with self.assertRaisesRegex(RN.Refused, "M1, L1 and L2 pass; the report says M1 is fail"):
            RN.stage_bca_lower_edge(self.ctx, self.seeds)
        self.fixtures(le_slope=1.0)
        RN.stage_report(self.ctx, self.seeds)
        with self.assertRaisesRegex(RN.Refused, "the report says L2 is fail"):
            RN.stage_bca_lower_edge(self.ctx, self.seeds)
        self.fixtures(le_slope=40.0)
        RN.stage_report(self.ctx, [0, 1])
        with self.assertRaisesRegex(RN.Refused, r"need a report on seeds \[0, 1, 2, 3, 4\]"):
            RN.stage_bca_lower_edge(self.ctx, self.seeds)

    def test_lower_edge_improvement_records_its_p_value_for_the_summary(self):
        self.fixtures(le_slope=40.0)
        banks = RN.banks_record_path(self.ctx)
        for s, d in zip(self.seeds, [0.5, 1.0, 1.5, 2.0, 2.5]):
            arms = {"le_bca_unweighted": -90.0, "le_bca_exact_strong": -90.0 + d, "le_bca_exact_mild": -90.0 + d / 2}
            recs = [self.put(RN.bca_le_dir(self.ctx, a, s) / "record.json", result=dict(arm=dict(refused_rows=0)),
                             inputs_sha256=RN.links(self.ctx, [RN.band_dir(self.ctx, "gap", s) / "record.json"]))
                    for a in RN.LE_ARMS]
            self.put(RN.evaluate_path(self.ctx, s, "bca_lower_edge"),
                     evaluations={k: {law: dict(J=v, points=(v + 80.0) / 0.6) for law in W.TILT_NAMES}
                                  for k, v in arms.items()},
                     inputs_sha256=RN.links(self.ctx, recs + [banks]))
        body = RN.build_report(self.ctx, self.seeds, self.digest)
        l3 = body["marks"]["L3"]
        self.assertEqual((l3["status"], l3["ran"], l3["uncorrected_status"]), ("pending", True, "pass"))
        self.assertAlmostEqual(l3["exact_strong_minus_unweighted"]["mean"], 1.5)
        self.assertAlmostEqual(l3["exact_mild_minus_unweighted_mild_bank"]["mean"], 0.75)
        self.assertIsNone(l3["estimated_strong_minus_unweighted"])
        self.assertAlmostEqual(l3["p_value"], RN.p_one_sided([0.5, 1.0, 1.5, 2.0, 2.5]), places=12)
        self.assertEqual(body["marks"]["I1"]["status"], "not run")      # the capped pull's own mark is separate
        self.assertEqual(set(body["shown"]["lower_edge"]["bca_records"]), set(RN.LE_ARMS))

    # ---------------------------------------------------------------------------------------- after review of v5

    def put_bca_evaluations(self, diffs, group="bca"):
        """Evaluations of the capped pull's (group 'bca') or the lower edge's (group 'bca_lower_edge') BCA arms with
        exact strong minus unweighted equal to diffs per seed (fixtures)."""
        banks = RN.banks_record_path(self.ctx)
        for s, d in zip(self.seeds, diffs):
            if group == "bca":
                arms = {"bca_unweighted": -90.0, "bca_exact_strong": -90.0 + d, "bca_exact_strong_single": -90.0}
                recs = [self.put(RN.bca_dir(self.ctx, a, s) / "record.json", result=dict()) for a in RN.BCA_ARMS]
            else:
                arms = {"le_bca_unweighted": -90.0, "le_bca_exact_strong": -90.0 + d, "le_bca_exact_mild": -90.0}
                recs = [self.put(RN.bca_le_dir(self.ctx, a, s) / "record.json", result=dict(arm=dict()),
                                 inputs_sha256=RN.links(self.ctx, [RN.band_dir(self.ctx, "gap", s) / "record.json"]))
                        for a in RN.LE_ARMS]
            self.put(RN.evaluate_path(self.ctx, s, group),
                     evaluations={k: {law: dict(J=v, points=(v + 80.0) / 0.6) for law in W.TILT_NAMES}
                                  for k, v in arms.items()},
                     inputs_sha256=RN.links(self.ctx, recs + [banks]))

    def test_a_seed_that_is_not_a_number_fails_the_improvement_marks(self):
        """A diverged BCA arm (a return that is not finite on one seed) never drops its quality from Holm's family:
        the report gives p = 1, and the uncorrected verdict fails."""
        self.fixtures(dose_change=0.4, le_slope=40.0)
        self.put_bca_evaluations([1.0, 2.0, float("nan"), 1.5, 1.2])
        self.put_bca_evaluations([1.0, 2.0, 3.0, float("inf"), 1.2], group="bca_lower_edge")
        marks = RN.build_report(self.ctx, self.seeds, self.digest)["marks"]
        for k in ("I1", "L3"):
            self.assertEqual((marks[k]["ran"], marks[k]["p_value"], marks[k]["uncorrected_status"]),
                             (True, 1.0, "fail"), k)
            self.assertEqual(marks[k]["nonfinite_seeds"], 1, k)

    def test_the_improvement_marks_need_m1(self):
        """I1 and L3 read 'not run' (not 'pending') when M1 fails, as their stages refuse without it."""
        self.fixtures(dose_change=0.4, le_slope=40.0, held_out=0.85)
        marks = RN.build_report(self.ctx, self.seeds, self.digest)["marks"]
        self.assertEqual([marks[k]["status"] for k in ("M1", "R1", "R2", "L1", "L2")],
                         ["fail", "pass", "pass", "pass", "pass"])
        for k in ("I1", "L3"):
            self.assertEqual(marks[k]["status"], "not run", k)
            self.assertIn("M1", marks[k]["why"], k)

    def test_records_from_another_quality_or_seed_are_refused(self):
        """A record copied from another quality's or seed's folder is refused, even when every record it links
        exists and matches (the reviewer's probe: the expert's lower-edge records copied into medium/)."""
        self.fixtures()
        medium = self.ctx.at("medium")
        for s in self.seeds:
            RN.write_record(RN.data_record_path(medium, s),
                            dict(RN.read_record(RN.data_record_path(self.ctx, s)), quality="medium"))
        for s in self.seeds:
            for src, dst in ((RN.room_le_dir(self.ctx, s) / "record.json", RN.room_le_dir(medium, s) / "record.json"),
                             (RN.evaluate_path(self.ctx, s, "room_lower_edge"),
                              RN.evaluate_path(medium, s, "room_lower_edge"))):
                RN.write_record(dst, RN.read_record(src))
        with self.assertRaisesRegex(RN.Refused, r"another quality or seed.*room_lower_edge 0: the record names "
                                                r"quality 'expert'"):
            RN.build_report(medium, self.seeds, self.digest)
        for s in self.seeds:      # relabelled, but still linking the expert's records
            for path in (RN.room_le_dir(medium, s) / "record.json", RN.evaluate_path(medium, s, "room_lower_edge")):
                RN.write_record(path, dict(RN.read_record(path), quality="medium"))
        with self.assertRaisesRegex(RN.Refused, r"it links expert/host/gap/seed0/record.json, outside medium/"):
            RN.build_report(medium, self.seeds, self.digest)
        self.fixtures()           # another seed's record in seed 2's folder
        RN.write_record(RN.room_dir(self.ctx, 2) / "record.json", RN.read_record(RN.room_dir(self.ctx, 1) /
                                                                                  "record.json"))
        with self.assertRaisesRegex(RN.Refused, r"room 2: the record names quality 'expert' and seed 1"):
            RN.build_report(self.ctx, self.seeds, self.digest)

    def test_the_report_says_whether_the_code_is_approved(self):
        self.fixtures()
        self.assertFalse(RN.build_report(self.ctx, self.seeds, self.digest)["code_is_approved"])
        write_approval(self.ctx.approval_path, self.digest)
        self.assertTrue(RN.build_report(self.ctx, self.seeds, self.digest)["code_is_approved"])
        write_approval(self.ctx.approval_path, self.digest,
                       code=dict(LI.code_hashes(), **{"experiments/shift_iw/run.py": "0" * 64}))
        self.assertFalse(RN.build_report(self.ctx, self.seeds, self.digest)["code_is_approved"])


class SummaryTest(unittest.TestCase):
    """The summary across qualities (spec version 5) on hand-written report records (fixtures; nothing is trained,
    fitted or evaluated, and this root holds no data)."""

    MARKS = ("D1", "M1", "M2", "M3", "W0", "W1", "W2", "W3", "W4", "R1", "R2")

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="shift_iw_summary_"))
        self.spec, self.digest = write_spec_copy(self.tmp / "iw")
        self.ctx = RN.Context(root=self.tmp / "runs", spec_path=self.spec, log=QUIET)
        self.design = {str(s): dict(gap=dict(rho=0.26 + s / 1000, centre=[-0.58, -0.12, -0.43, -0.29]), D1=D1_OK)
                       for s in range(5)}

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def report(self, quality, p=None, fail=(), complete=True, design=None, p_le=None, waiting=(), **extra):
        """A report fixture: I1 ran with p-value p (L3 with p_le), or 'not run'; marks in `waiting` are allowed by
        the report (bca_allowed, bca_lower_edge_allowed) but their arms have not run yet ('pending', no p-value)."""
        marks = {m: dict(status="fail" if m in fail else "pass") for m in self.MARKS}
        ran = lambda v: (dict(status="not run") if v is None else
                         dict(status="pending", ran=True, p_value=v,
                              uncorrected_status="pass" if v is not None and v < 0.05 else "fail"))
        marks["I1"], marks["L3"] = ran(p), ran(p_le)
        for m in waiting:
            marks[m] = dict(status="pending", why="evaluate (bca)")
        extra.setdefault("bca_allowed", p is not None or "I1" in waiting)
        extra.setdefault("bca_lower_edge_allowed", p_le is not None or "L3" in waiting)
        body = dict(status="finished", spec_sha256=self.digest, code_sha256=LI.code_hashes(), quality=quality,
                    seeds=[0, 1, 2, 3, 4], complete=complete, marks=marks, design=design or self.design,
                    shown=dict(plain_td3bc_points={"gap": {"strong": dict(mean=40.0)}}))
        body.update(extra)
        RN.write_record(RN.report_path(self.ctx.at(quality)), body)

    def test_holm(self):
        self.assertEqual(RN.holm({"a": 0.010, "b": 0.020, "c": 0.030}, 0.05), {"a": True, "b": True, "c": True})
        self.assertEqual(RN.holm({"a": 0.010, "b": 0.030, "c": 0.040}, 0.05), {"a": True, "b": False, "c": False})
        self.assertEqual(RN.holm({"a": 0.049}, 0.05), {"a": True})                 # k = 1: the uncorrected test
        self.assertEqual(RN.holm({"a": 0.011, "b": 0.5, "c": 0.5, "d": 0.5, "e": 0.5}, 0.05)["a"], False)  # 0.01

    def test_summary_applies_holm_and_the_claim_rules(self):
        ps = {"medium-replay": 0.010, "medium": 0.030, "medium-expert": 0.040}
        for q in W.QUALITIES:
            self.report(q, p=ps.get(q), fail=("M2",) if q == "random" else ("W2",) if q == "expert" else ())
        body = RN.build_summary(self.ctx, self.digest)
        h = body["holm"]["I1"]
        self.assertTrue(h["judged"])
        self.assertEqual((h["k"], h["ran_in"]), (3, sorted(ps)))
        self.assertEqual(h["passes"], {"medium-replay": True, "medium": False, "medium-expert": False})
        rows = body["rows"]
        self.assertEqual(rows["medium-replay"]["marks"]["I1"], "pass")
        self.assertEqual(rows["medium"]["marks"]["I1"], "fail")
        self.assertEqual(rows["expert"]["marks"]["I1"], "not run")
        self.assertEqual(rows["medium-replay"]["claims"], dict(mechanics=True, coverage=True, return_lower_edge=False,
                                                               **{"return": True}))
        self.assertEqual(rows["medium"]["claims"]["return"], False)
        self.assertEqual(rows["random"]["claims"]["mechanics"], False)      # M2 failed there
        self.assertEqual(rows["expert"]["claims"]["coverage"], False)       # W2 failed there
        self.assertIn("weighting only", rows["random"]["label"])
        self.assertEqual(rows["medium"]["plain_td3bc_points"]["gap"]["strong"]["mean"], 40.0)
        RN.stage_summary(self.ctx)
        self.assertIsNotNone(RN.finished(RN.summary_path(self.ctx), self.digest))

    def test_the_lower_edge_is_its_own_holm_family(self):
        """L3 is judged with Holm's correction over the qualities where it ran, separately from I1, and supports its
        own claim."""
        ps = {"medium": 0.030, "expert": 0.020}
        for q in W.QUALITIES:
            self.report(q, p=0.010 if q == "medium" else None, p_le=ps.get(q))
        body = RN.build_summary(self.ctx, self.digest)
        self.assertEqual((body["holm"]["I1"]["k"], body["holm"]["L3"]["k"]), (1, 2))
        self.assertEqual(body["holm"]["L3"]["passes"], {"medium": True, "expert": True})   # 0.020 <= 0.025, 0.030
        self.assertEqual(body["rows"]["medium"]["marks"]["L3"], "pass")
        self.assertEqual(body["rows"]["random"]["marks"]["L3"], "not run")
        self.assertEqual(body["rows"]["expert"]["claims"], dict(mechanics=True, coverage=True, return_lower_edge=True,
                                                                **{"return": False}))
        self.assertIn("lower edge", body["claim_rules"]["return_lower_edge"]["claim"])
        self.report("expert", p_le=0.040)
        self.assertEqual(RN.build_summary(self.ctx, self.digest)["holm"]["L3"]["passes"],
                         {"medium": False, "expert": False})                        # 0.030 > 0.025: both fail

    def test_holm_waits_for_every_quality(self):
        for q in W.QUALITIES[:-1]:
            self.report(q, p=0.001)
        body = RN.build_summary(self.ctx, self.digest)
        self.assertFalse(body["complete"])
        self.assertFalse(body["holm"]["I1"]["judged"])
        self.assertEqual(body["rows"]["medium"]["marks"]["I1"], "pending")
        self.assertFalse(body["rows"]["expert"]["report"])
        self.report("expert", p=0.001, complete=False)
        self.assertFalse(RN.build_summary(self.ctx, self.digest)["holm"]["I1"]["judged"])

    def test_holms_family_is_every_quality_whose_report_allows_the_arms(self):
        """The reviewer's case: the expert's I1 ran (p = 0.04) while medium's report allows the arms but they have
        not run. The family is {expert, medium}, so nothing is judged and no claim is made; a claim made at k = 1
        would be withdrawn once medium runs. The owner's recorded decline (declined.json) closes the family with
        medium at p = 1, so declining never loosens the bar."""
        for q in W.QUALITIES:
            self.report(q, p=0.04 if q == "expert" else None, waiting=("I1",) if q == "medium" else ())
        body = RN.build_summary(self.ctx, self.digest)
        h = body["holm"]["I1"]
        self.assertTrue(body["complete"])
        self.assertEqual((h["family"], h["ran_in"], h["waiting_for"], h["k"]),
                         (["expert", "medium"], ["expert"], ["medium"], 2))
        self.assertFalse(h["judged"])
        self.assertIn("waiting for ['medium']", h["note"])
        self.assertEqual((body["rows"]["expert"]["marks"]["I1"], body["rows"]["medium"]["marks"]["I1"]),
                         ("pending", "pending"))
        self.assertFalse(body["rows"]["expert"]["claims"]["return"])
        self.assertEqual(body["rows"]["random"]["marks"]["I1"], "not run")
        self.report("medium", p=0.04)                                   # medium ran: k = 2, 0.04 > 0.025
        h = RN.build_summary(self.ctx, self.digest)["holm"]["I1"]
        self.assertEqual((h["judged"], h["k"], h["passes"]), (True, 2, {"expert": False, "medium": False}))
        self.report("medium", waiting=("I1",))                          # medium declined instead
        RN.write_record(self.ctx.declined_path, dict(kind=RN.DECLINED_KIND, spec_sha256=self.digest,
                                                     user_words_verbatim="Skip medium's capped pull.",
                                                     declined=dict(I1=["medium"])))
        body = RN.build_summary(self.ctx, self.digest)
        h = body["holm"]["I1"]
        self.assertEqual((h["judged"], h["declined"], h["p_values"]), (True, ["medium"], {"expert": 0.04,
                                                                                         "medium": 1.0}))
        self.assertEqual(h["passes"], {"expert": False, "medium": False})
        self.assertEqual(body["rows"]["medium"]["marks"]["I1"], "declined")
        self.assertEqual(body["declined"], {"I1": ["medium"]})
        self.report("expert", p=0.004)                                  # 0.004 <= 0.025: the expert passes
        body = RN.build_summary(self.ctx, self.digest)
        self.assertEqual(body["holm"]["I1"]["passes"], {"expert": True, "medium": False})
        self.assertTrue(body["rows"]["expert"]["claims"]["return"])
        self.assertEqual(body["holm"]["L3"]["family"], [])              # its own family: empty here
        for bad in (dict(kind="other"), dict(spec_sha256="e" * 64), dict(user_words_verbatim=" "),
                    dict(declined=dict(I2=["medium"])), dict(declined=dict(I1=["novice"]))):
            RN.write_record(self.ctx.declined_path, dict(dict(kind=RN.DECLINED_KIND, spec_sha256=self.digest,
                                                              user_words_verbatim="Skip it.",
                                                              declined=dict(I1=["medium"])), **bad))
            with self.assertRaisesRegex(RN.Refused, "declined.json must be"):
                RN.build_summary(self.ctx, self.digest)
        LI.sidecar(self.ctx.declined_path).write_text("0" * 64 + "  declined.json\n")
        with self.assertRaisesRegex(RN.Refused, "declined.json does not match its sidecar"):
            RN.build_summary(self.ctx, self.digest)

    def test_a_member_whose_p_value_is_missing_fails_and_stays_in_the_family(self):
        """L3 ran on medium but a seed's return was not finite (p_value missing): medium stays in the family with
        p = 1 and fails; the family keeps k = 5, so the others face the full bar."""
        ps = {"random": 0.004, "medium-replay": 0.011, "medium": None, "medium-expert": 0.02, "expert": 0.03}
        for q in W.QUALITIES:
            self.report(q, p_le=0.5 if q == "medium" else ps[q])
        path = RN.report_path(self.ctx.at("medium"))
        rec = RN.read_record(path)
        rec["marks"]["L3"]["p_value"] = None
        RN.write_record(path, rec)
        h = RN.build_summary(self.ctx, self.digest)["holm"]["L3"]
        self.assertEqual((h["k"], h["p_values"]["medium"]), (5, 1.0))
        self.assertEqual(h["passes"], {"random": True, "medium-replay": True, "medium": False, "medium-expert": False,
                                       "expert": False})
        self.assertEqual(RN._family_p("nan"), 1.0)
        self.assertEqual(RN._family_p(0.02), 0.02)

    def test_summary_refuses_reports_from_another_folder_or_with_a_broken_sidecar(self):
        for q in W.QUALITIES:
            self.report(q)
        RN.write_record(RN.report_path(self.ctx.at("medium")),
                        RN.read_record(RN.report_path(self.ctx.at("expert"))))      # the expert's report, copied
        with self.assertRaisesRegex(RN.Refused, "medium: the report in this quality's folder is of quality 'expert'"):
            RN.build_summary(self.ctx, self.digest)
        self.report("medium")
        LI.sidecar(RN.report_path(self.ctx.at("random"))).write_text("0" * 64 + "  report.json\n")
        with self.assertRaisesRegex(RN.Refused, "random: the report does not match its sidecar"):
            RN.build_summary(self.ctx, self.digest)

    def test_summary_refuses_reports_that_do_not_belong_together(self):
        for q in W.QUALITIES:
            self.report(q)
        RN.build_summary(self.ctx, self.digest)
        other_ball = json.loads(json.dumps(self.design))
        other_ball["3"]["gap"]["rho"] = 0.3
        self.report("medium", design=other_ball)
        with self.assertRaisesRegex(RN.Refused, "seed 3: the qualities' reports give different balls"):
            RN.build_summary(self.ctx, self.digest)
        other_d1 = json.loads(json.dumps(self.design))
        other_d1["1"]["D1"] = dict(D1_OK, passes=False)
        self.report("medium", design=other_d1)
        with self.assertRaisesRegex(RN.Refused, "seed 1: the qualities' reports give different design-check"):
            RN.build_summary(self.ctx, self.digest)
        self.report("medium")
        self.report("random", code_sha256={"experiments/shift_iw/run.py": "0" * 64})
        with self.assertRaisesRegex(RN.Refused, "random: the report was made by other code"):
            RN.build_summary(self.ctx, self.digest)
        self.report("random", spec_sha256="e" * 64)
        with self.assertRaisesRegex(RN.Refused, "random: the report was made under another specification"):
            RN.build_summary(self.ctx, self.digest)


if __name__ == "__main__":
    unittest.main()

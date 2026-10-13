"""CPU tests of lower_edge.py (spec version 5, part B: the lower-edge use of BCA's band and its room check).

BCA (the band, its widths and the lower edge's BCA arms) runs here only on small random arrays opened by the SYNTHETIC
token (learners_iw.synthetic_prepared), never on ThrusterChain data. The room check's oracle needs no BCA. The cloud's
Stage 1 harness (outputs/shift/beyond-the-data/stage1/code/harness.py, read-only) is loaded only here, to compare the
restated parts with its own functions; the run never imports it.
JAX_PLATFORMS=cpu python -m unittest experiments.shift_iw.test_lower_edge
"""

import importlib.util
import json
import math
import shutil
import sys
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

import numpy as np

import experiments.shift  # noqa: F401

import jax
import jax.numpy as jnp
from flax.training.train_state import TrainState
from jax.experimental import enable_x64

from experiments.shift import learners as LR
from experiments.shift_iw import learners_iw as LI
from experiments.shift_iw import lower_edge as LE
from experiments.shift_iw import world as W
from experiments.shift_iw.test_learners_iw import write_approval, write_spec_copy

SPEC = json.loads(LI.SPEC_JSON.read_text(encoding="utf8"))
RULE_E = SPEC["lower_edge"]["check_e"]
HARNESS_PATH = LI.REPO / LE.HARNESS


def setting(**kw):
    return LI.settings(clone_steps=40, actor_only_steps=8, actor_only_chunk=4, match_rounds=2, row_chunk=1024,
                       strength_states=64, **kw)


def synthetic(n=1024, seed=21):
    """A synthetic Prepared (registered for the token) whose region is the hard-water rows, as data.prepare sets it."""
    prep = LI.synthetic_prepared(n=n, bank=200, eval_states=64, seed=seed)
    prep = replace(prep, region=W.HoldRecorder().hard_water(prep.states))
    return LI.register_synthetic(prep)


def same(a, b):
    la, lb = jax.tree_util.tree_leaves(a), jax.tree_util.tree_leaves(b)
    return len(la) == len(lb) and all(np.array_equal(np.asarray(x), np.asarray(y)) for x, y in zip(la, lb))


def load_harness():
    """The cloud's Stage 1 harness, read-only (it puts its cloud root on sys.path at import; that is undone)."""
    saved = list(sys.path)
    try:
        spec = importlib.util.spec_from_file_location("stage1_harness_readonly", HARNESS_PATH)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
    finally:
        sys.path[:] = saved
    return mod


class Fixture:
    """One small synthetic host and its clone, shared by the classes below."""

    _built = None

    @classmethod
    def get(cls):
        if cls._built is None:
            cfg = setting(band_fit_steps=60)
            prep = synthetic()
            _, host, _ = LI.train_host(prep, cfg, 0, updates=200)
            clone, _ = LR.clone(prep, cfg, 0)
            cls._built = dict(cfg=cfg, prep=prep, host=host, critic=host.critic.params, clone=clone)
        return cls._built


# ------------------------------------------------------------------------------------------------ the oracle (no BCA)

class OracleTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        f = Fixture.get()
        cls.cfg, cls.prep, cls.critic, cls.clone = f["cfg"], f["prep"], f["critic"], f["clone"]
        cls.shared, cls.info = LE.oracle_shared(cls.prep, cls.critic)

    def test_the_step_is_exact_float64(self):
        """The in-program delta* (float64 step under enable_x64) equals the float64 reference computed outside the
        program, and is zero-referenced: the paired penalty is 0 at every recorded command."""
        rows = np.arange(300)
        ref = LE.oracle64(self.critic, self.prep, self.prep.states[rows], self.prep.actions[rows])
        d = np.asarray(self.shared["d_logged"])[rows]
        self.assertEqual(np.asarray(self.shared["d_logged"]).dtype, np.float64)
        np.testing.assert_allclose(d, ref, rtol=0, atol=1e-9)
        pi = np.asarray(LR.actor_rows(self.clone, np.asarray(self.prep.train.obs)[rows]))
        moved = LE.oracle_delta_rows(self.critic, self.shared, rows, pi)
        np.testing.assert_allclose(moved, LE.oracle64(self.critic, self.prep, self.prep.states[rows], pi), atol=1e-9)
        with enable_x64():
            W0 = LE.oracle_penalty(W.PHYSICS, self.critic, self.shared, jnp.asarray(rows, jnp.int32), None,
                                   jnp.asarray(self.prep.actions[rows]), None)
        # 0 at the recorded command up to float32 rounding of the critic (another batch size than d_logged's)
        self.assertLessEqual(float(jnp.max(jnp.abs(W0))), 1e-5)
        self.assertGreater(float(np.mean(np.asarray(W0) == 0.0)), 0.5)
        self.assertEqual(self.info["next_command"], LE.controller(W.PHYSICS).name)

    def test_k_zero_is_the_room_checks_step_up_to_adams_rounding(self):
        """One step of the float64-step program with k = 0 equals learners.actor_step except for Adam's bias
        correction, which enable_x64 computes in float64 and rounds (about one float32 step)."""
        args = LR.make_args(self.cfg, 0, self.cfg.alpha)
        actor = TrainState.create(apply_fn=LR.ACTOR.apply, params=self.clone, tx=LR.adam(self.cfg.lr))
        idx = jnp.arange(256, dtype=jnp.int32)
        batch = jax.tree_util.tree_map(lambda x: x[idx], self.prep.train)
        arm = dict(k=jnp.float32(0.0), pull=jnp.ones(self.prep.n, jnp.float32))
        pen = LE._penalty("oracle", W.PHYSICS)
        with enable_x64():
            mine, aux = jax.jit(lambda a: LE.actor_step(args, pen, a, self.critic, batch, idx, self.shared, arm))(actor)
        theirs, _ = jax.jit(lambda a, t: LR.actor_step(args, a, self.critic, batch, t[idx]))(
            actor, jnp.ones(self.prep.n, jnp.float32))
        self.assertLessEqual(LE.max_param_diff(mine.params, theirs.params), 1e-7)
        self.assertTrue(float(aux[4]) >= 0.0)                 # the W mean is reported even at k = 0

    def test_judge_shape(self):
        """Check (e)'s verdict on hand-built surfaces: a penalty that rises past 80% passes; one that falls just past
        the knee fails (e1-e3); one that rises but leaves the best throttle where the critic alone puts it fails e4."""
        grid = [float(m) for m in RULE_E["grid"]]
        n = 40
        Q1 = {m: np.full(n, 1.0 + 2.0 * m) for m in grid}               # the critic alone prefers full throttle
        rising = {m: np.full(n, 10.0 * max(m - 0.8, 0.0)) for m in grid}
        slopes_up = {float(m): np.full(n, 10.0 if m >= 0.8 else 0.0) for m in RULE_E["slope_points"]}
        rec = LE.judge_shape(grid, rising, Q1, slopes_up, RULE_E)
        self.assertTrue(rec["passed"], rec)
        self.assertEqual(rec["q1_alone_best_past80_share"], 1.0)
        self.assertEqual(rec["best_past80_share"], 0.0)
        falling = {m: np.full(n, 0.2 - 2.0 * max(m - 0.8, 0.0) + 3.0 * max(m - 0.86, 0.0)) for m in grid}
        slopes_down = {float(m): np.full(n, -2.0 if 0.8 <= m < 0.86 else 1.0) for m in RULE_E["slope_points"]}
        rec = LE.judge_shape(grid, falling, Q1, slopes_down, RULE_E)
        self.assertFalse(rec["passed"])
        self.assertFalse(rec["e1_fall_share_ok"] or rec["e2_mean_rises"] or rec["e3_slope_past_knee_positive"])
        self.assertEqual([f[:3] for f in rec["failed"]], ["e1:", "e2:", "e3:", "e4:"])     # named, for the stop
        self.assertEqual(LE.judge_shape(grid, rising, Q1, slopes_up, RULE_E)["failed"], [])
        weak = {m: np.full(n, 0.1 * max(m - 0.8, 0.0)) for m in grid}    # rises, but too gently to move the optimum
        rec = LE.judge_shape(grid, weak, Q1, slopes_up, RULE_E)
        self.assertTrue(rec["e1_fall_share_ok"] and rec["e2_mean_rises"] and rec["e3_slope_past_knee_positive"])
        self.assertFalse(rec["e4_fewer_best_past80_than_q1"] or rec["e4_passed"] or rec["passed"])
        self.assertEqual(len(rec["failed"]), 1)
        self.assertTrue(rec["failed"][0].startswith("e4:"))
        bad = {**slopes_up, 0.9: np.full(n, np.nan)}                      # a slope that is not finite fails e3
        rec = LE.judge_shape(grid, rising, Q1, bad, RULE_E)
        self.assertFalse(rec["e3_slope_past_knee_positive"])
        self.assertTrue(rec["failed"][0].startswith("e3:"))

    def test_e4_is_not_applicable_when_the_critic_never_prefers_past_80(self):
        """The reviewer's case: a penalty that rises cleanly past 80% (e1-e3 pass) on a critic that alone already
        peaks at 75% on every row. The harness's strict e4 cannot pass (0 is not fewer than 0); here e4 is recorded
        as not applicable and passes, because Q1 - W never prefers past 80% either. A penalty that makes Q1 - W prefer
        past 80% where the critic alone never did still fails e4."""
        grid = [float(m) for m in RULE_E["grid"]]
        n = 40
        Q1 = {m: np.full(n, 1.0 - 4.0 * (m - 0.75) ** 2) for m in grid}  # peaks at 75%
        rising = {m: np.full(n, 10.0 * max(m - 0.8, 0.0)) for m in grid}
        slopes_up = {float(m): np.full(n, 10.0 if m >= 0.8 else 0.0) for m in RULE_E["slope_points"]}
        rec = LE.judge_shape(grid, rising, Q1, slopes_up, RULE_E)
        self.assertEqual((rec["q1_alone_best_past80_share"], rec["best_past80_share"]), (0.0, 0.0))
        self.assertFalse(rec["e4_fewer_best_past80_than_q1"])          # the harness's rule
        self.assertTrue(rec["e4_not_applicable"] and rec["e4_passed"] and rec["passed"])
        self.assertIn("no over-preference past 80%", rec["note"])
        # a penalty that rewards full throttle while still rising from 80% to 81%: Q1 - W now prefers past 80%
        pull = {m: np.full(n, 10.0 * max(m - 0.8, 0.0) - (40.0 if m == 1.0 else 0.0)) for m in grid}
        rec = LE.judge_shape(grid, pull, Q1, slopes_up, RULE_E)
        self.assertTrue(rec["e4_not_applicable"])
        self.assertFalse(rec["e4_passed"] or rec["passed"])

    def test_check_e_runs_on_the_oracle_and_its_rows_are_seeded(self):
        rec = LE.oracle_shape(self.critic, self.shared, self.prep, self.clone, 3, RULE_E)
        rows = LE.shape_rows(self.prep, 3, RULE_E["rows"])
        self.assertTrue(np.all(self.prep.region[rows]))
        self.assertEqual(rec["rows"], len(rows))
        np.testing.assert_array_equal(rows, LE.shape_rows(self.prep, 3, RULE_E["rows"]))
        few = lambda rep: LE.shape_rows(self.prep, rep, 50)                # fewer rows than the region: a seeded draw
        np.testing.assert_array_equal(few(3), few(3))
        self.assertFalse(np.array_equal(few(3), few(4)))
        self.assertEqual(set(rec["mean_W"]), {f"{m:g}" for m in RULE_E["grid"]})
        self.assertTrue(all(math.isfinite(v) for v in rec["mean_W"].values()))
        for key in ("e1_fall_share_ok", "e2_mean_rises", "e3_slope_past_knee_positive", "e4_fewer_best_past80_than_q1",
                    "passed"):
            self.assertIn(key, rec)
        empty = replace(self.prep, region=np.zeros(self.prep.n, bool))
        self.assertFalse(LE.oracle_shape(self.critic, self.shared, empty, self.clone, 3, RULE_E)["passed"])

    def test_room_check_stops_on_check_e_and_otherwise_trains_side_by_side(self):
        rule = LE.RoomRule(k=dict(SPEC["lower_edge"]["room_check"]["arms"]),
                           even_extra=tuple(SPEC["lower_edge"]["room_check"]["even_pull"]["extra_rungs"]),
                           check_e=RULE_E)
        ao = LR.ActorOnly(self.prep, self.clone, self.cfg, 0, alpha=self.cfg.alpha)
        arms = [LR.Arm("none", "even", 1.0), LR.Arm("even_x1.1", "even", 1.1), LR.Arm("even_x1.25", "even", 1.25),
                LR.Arm("even_x1.5", "even", 1.5)]
        st, _ = ao.run(arms, [self.critic])
        room = {a.name: ao.arm_params(st, i) for i, a in enumerate(arms)}
        _, clone_rec = LR.clone(self.prep, self.cfg, 0)
        record = dict(clone=clone_rec, strengths={a.name: LR.strength(room[a.name], room["none"], self.prep.eval_obs)
                                                  for a in arms})
        stop = replace(rule, check_e=dict(RULE_E, max_fall_share=-1.0))      # e1 cannot pass
        params, rec = LE.room_check(self.prep, self.critic, room, record, self.cfg, 0, stop)
        self.assertEqual(params, {})
        self.assertTrue(rec["stopped"] and not rec["check_e"]["passed"])
        self.assertTrue(rec["clone_reproduces_room_check"])
        real = LE.oracle_shape
        try:                                                                  # the verdict forced to pass
            LE.oracle_shape = lambda *a, **k: dict(real(*a, **k), passed=True)
            params, rec = LE.room_check(self.prep, self.critic, room, record, self.cfg, 0, rule)
        finally:
            LE.oracle_shape = real
        self.assertFalse(rec["stopped"])
        self.assertTrue({"le_none", "le_oracle", "le_oracle_x1.5", "le_even_x2", "le_even_x3"} <= set(params))
        self.assertEqual(sorted(rec["names"]), sorted(params))
        self.assertEqual([a["k"] for a in rec["training"]["arms"]], [0.0, 1.0, 1.5])
        self.assertEqual(rec["training"]["rows_sha256"], ao.rows_sha256)           # the room check's stream
        self.assertEqual(rec["training"]["start_sha256"], LR.tree_sha256(self.clone))
        self.assertEqual([x[0] for x in rec["ladder"]], [1.1, 1.25, 1.5, 2.0, 3.0])
        self.assertIn("multiplier", rec["match"])
        self.assertTrue(math.isfinite(rec["check_b"]["max_param_diff"]))
        self.assertTrue(all(math.isfinite(v) for v in rec["strengths"].values()))


# ------------------------------------------------------------------------------------------------ the band (synthetic)

class BandPenaltyTest(unittest.TestCase):
    """The lower edge's BCA arms on a synthetic band (the SYNTHETIC token on registered synthetic arrays)."""

    @classmethod
    def setUpClass(cls):
        f = Fixture.get()
        cls.cfg, cls.prep, cls.host, cls.critic, cls.clone = f["cfg"], f["prep"], f["host"], f["critic"], f["clone"]
        cls.band = LI.fit_band(cls.host, cls.prep, cls.cfg, 0, LI.SYNTHETIC, chunk=30)
        cls.R_u = float(cls.band.state.posterior.threshold)
        cls.shared = LE.band_shared(cls.band)
        cls.args = LR.make_args(cls.cfg, 0, cls.cfg.alpha)

    def width(self, shared, R, obs, a):
        return np.asarray(LE.band_width(shared["cal_params"], shared["unit"], jnp.asarray(obs), jnp.asarray(a),
                                        jnp.asarray(R, jnp.float32)))

    def test_width_is_bcas_own(self):
        """U = R x unit x max(eta, 1e-6) in float32, equal to BCA's own readout at the recorded commands."""
        rd = LI.readout(self.band, self.band.state.posterior, self.prep.train)
        u = LE.logged_widths(self.band, self.prep, np.full(self.prep.n, self.R_u, np.float32))
        np.testing.assert_array_equal(u, rd["width"])

    def test_every_band_reader_passes_the_gate(self):
        """The lower edge's band readers (band_shared, widths_at, logged_widths, band_shape, proxy_check and the
        runner of kind 'band') refuse a band whose approval does not cover their scope, no band at all, and a band
        fitted on other data, so the lower edge cannot train on real data outside run.py's gated stage."""
        R = np.full(self.prep.n, self.R_u, np.float32)
        arm = dict(R=R, u_logged=LE.logged_widths(self.band, self.prep, R))
        tmp = Path(tempfile.mkdtemp(prefix="shift_iw_le_readers_"))     # holds no data
        try:
            spec, digest = write_spec_copy(tmp / "iw")
            path = write_approval(tmp / "approval.json", digest, scope=("band", "coverage"))
            gated = replace(self.band, approval=LI.verify_approval(path, spec, "band"))
            calls = dict(
                shared=lambda b: LE.band_shared(b),
                widths=lambda b: LE.widths_at(b, self.prep, self.prep.train.action, R),
                logged=lambda b: LE.logged_widths(b, self.prep, R),
                shape=lambda b: LE.band_shape(b, arm, self.critic, self.prep, self.clone, 0, RULE_E),
                runner=lambda b: LE.Runner(self.prep, self.clone, self.critic, self.cfg, 0, "band", None, band=b))
            for name, call in calls.items():
                with self.assertRaisesRegex(LI.ApprovalError, "does not cover the 'bca_lower_edge' stage", msg=name):
                    call(gated)
                with self.assertRaises(LI.ApprovalError, msg=name):
                    call(None)
            # the band stage's scope opens the band stage's reader (the proxy check), not the BCA arms'
            self.assertEqual(len(LE.logged_widths(gated, self.prep, R, stage="band")), self.prep.n)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)
        other = synthetic(n=self.prep.n, seed=22)                      # registered, but not this band's data
        with self.assertRaisesRegex(ValueError, "fitted on other data"):
            LE.logged_widths(self.band, other, R)
        with self.assertRaisesRegex(ValueError, "fitted on other data"):
            LE.Runner(other, self.clone, self.critic, self.cfg, 0, "band", None, band=self.band)

    def test_proxy_check_compares_bcas_lower_edge_with_the_oracles(self):
        """L2's proxy check (reported): on every kept training row at the no-caution actor's command, BCA's paired
        penalty at R_u against the oracle's paired penalty, their ratio of means and correlations, and check (e) on
        BCA's penalty; it reads the band at scope 'band' and trains nothing."""
        before = LR.tree_sha256(self.clone)
        rec = LE.proxy_check(self.band, self.prep, self.critic, self.clone, self.R_u, 0, RULE_E)
        self.assertFalse(rec["skipped"])
        self.assertEqual(rec["rows"], self.prep.n)
        a = np.asarray(LR.actor_rows(self.clone, np.asarray(self.prep.train.obs)), np.float32)
        R = np.full(self.prep.n, self.R_u, np.float32)
        w_band = np.maximum(LE.widths_at(self.band, self.prep, a, R) - LE.logged_widths(self.band, self.prep, R), 0)
        self.assertAlmostEqual(rec["band_mean"], float(w_band.astype(np.float64).mean()), places=12)
        shared, _ = LE.oracle_shared(self.prep, self.critic)
        d = LE.oracle_delta_rows(self.critic, shared, np.arange(self.prep.n), a)
        w_oracle = np.maximum(np.asarray(shared["d_logged"]) - d, 0.0)
        self.assertAlmostEqual(rec["oracle_mean"], float(w_oracle.mean()), places=12)
        self.assertAlmostEqual(rec["band_over_oracle"], rec["band_mean"] / rec["oracle_mean"], places=12)
        self.assertTrue(-1.0 <= rec["pearson"] <= 1.0 and -1.0 <= rec["spearman"] <= 1.0)
        self.assertIn("passed", rec["band_check_e"])
        self.assertEqual(LR.tree_sha256(self.clone), before)
        self.assertTrue(LE.proxy_check(self.band, self.prep, self.critic, self.clone, np.inf, 0, RULE_E)["skipped"])

    def test_per_step_R_scales_the_penalty_per_state(self):
        """In the paired form the penalty at each state scales with that state's own R: W(R_i) = (R_i / R_u) W(R_u)."""
        rng = np.random.default_rng(5)
        n = 512
        obs = np.asarray(self.prep.train.obs)[:n]
        a_rec = np.asarray(self.prep.train.action)[:n]
        a_new = np.clip(a_rec + 0.3 * rng.standard_normal(a_rec.shape), -1, 1).astype(np.float32)
        ratio = (1.0 + 0.4 * rng.uniform(size=n)).astype(np.float32)
        R = (self.R_u * ratio).astype(np.float32)
        pen = lambda RR: np.maximum(self.width(self.shared, RR, obs, a_new) - self.width(self.shared, RR, obs, a_rec),
                                    0)
        base, scaled = pen(np.full(n, self.R_u, np.float32)), pen(R)
        self.assertGreater(np.mean(base > 0), 0.1)
        tol = 1e-6 * float(np.max(self.width(self.shared, R, obs, a_new)))       # float32 rounding of two widths
        np.testing.assert_allclose(scaled, ratio * base, rtol=1e-5, atol=tol)
        clear = np.abs(base) > tol
        self.assertTrue(np.all((scaled[clear] > 0) == (base[clear] > 0)))

    def test_cannot_certify_gives_the_largest_finite_width(self):
        R, cap, refused = LE.capped_thresholds([1.0, np.inf, 2.0, np.inf], [0.5, 1.0, 1.7, 2.5, np.inf])
        np.testing.assert_array_equal(R, [1.0, 2.5, 2.0, 2.5])
        self.assertEqual((cap, refused.tolist()), (2.5, [False, True, False, True]))
        with self.assertRaisesRegex(ValueError, "certifies nothing"):
            LE.capped_thresholds([np.inf, 1.0], [np.inf])
        with self.assertRaises(ValueError):
            LE.capped_thresholds([np.nan], [1.0])
        sc = LE.scale_change([self.R_u, np.inf], [self.R_u, 1.5 * self.R_u, np.inf], self.R_u, high=[False, True])
        self.assertAlmostEqual(sc["scale_change"], 0.25)
        self.assertAlmostEqual(sc["high_current_scale_change"], 0.5)
        self.assertEqual((sc["refused_rows"], sc["largest_finite_threshold"]), (1, 1.5 * self.R_u))
        self.assertIsNone(LE.scale_change([1.0], [1.0], np.inf))
        # in an arm: a refused row reads the table's largest finite threshold, and is counted
        n = self.prep.n
        R_rows = np.full(n, 1.2 * self.R_u)
        R_rows[::7] = np.inf
        pool = [self.R_u, 1.2 * self.R_u, 1.6 * self.R_u, np.inf]
        params, rec = LE.band_arms(self.prep, self.band, self.critic,
                                   [dict(name="exact_strong", R=R_rows, pool=pool)], 1.0, self.cfg, 0, LI.SYNTHETIC)
        arm = rec["arms"]["exact_strong"]
        self.assertEqual(arm["refused_rows"], len(range(0, n, 7)))
        self.assertAlmostEqual(arm["largest_finite_threshold"], 1.6 * self.R_u)
        self.assertAlmostEqual(arm["R_max"], 1.6 * self.R_u)
        self.assertTrue(LR._finite_tree(params["exact_strong"]))

    def run_arms(self, shared, arms):
        """The arms one after another, each in its own program (as band_arms runs them)."""
        runner = LE.Runner(self.prep, self.clone, self.critic, self.cfg, 0, "band", shared, band=self.band)
        return runner.run(arms, separate=True)[0]

    def arm(self, name, k, R, shared=None):
        R32 = np.asarray(R, np.float32)
        return dict(name=name, k=k, R=R32, u_logged=LE.logged_widths(self.band, self.prep, R32, shared=shared))

    def test_gradient_reaches_the_actor_only_through_the_scale_networks_command_input(self):
        """With the scale network's command-input weights set to 0 the penalty no longer depends on the command, and
        the arm trains exactly as with k = 0; with the real network it does not. The scale network itself (and the
        whole band) is unchanged by training."""
        n = self.prep.n
        R = np.full(n, self.R_u)
        cal = jax.tree_util.tree_map(np.asarray, self.shared["cal_params"])
        first = sorted(cal["params"], key=lambda k: int(k.rsplit("_", 1)[1]))[0]
        kernel = np.array(cal["params"][first]["kernel"])
        self.assertEqual(kernel.shape[0], LR.DX + LR.DA)          # inputs: the observation, then the command
        blind_kernel = kernel.copy()
        blind_kernel[LR.DX:] = 0.0
        blind = dict(self.shared, cal_params=jax.tree_util.tree_map(jnp.asarray, {"params": dict(
            cal["params"], **{first: dict(cal["params"][first], kernel=blind_kernel)})}))
        before = LI.state_sha256(self.band.state)
        p = self.run_arms(blind, [self.arm("k0", 0.0, R, blind), self.arm("k1", 1.0, R, blind)])
        self.assertTrue(same(p["k0"], p["k1"]))
        p = self.run_arms(self.shared, [self.arm("k0", 0.0, R), self.arm("k1", 1.0, R)])
        self.assertFalse(same(p["k0"], p["k1"]))
        self.assertEqual(LI.state_sha256(self.band.state), before)
        # the loss is differentiated in the actor's parameters only
        idx = jnp.arange(64, dtype=jnp.int32)
        batch = jax.tree_util.tree_map(lambda x: x[idx], self.prep.train)
        actor = TrainState.create(apply_fn=LR.ACTOR.apply, params=self.clone, tx=LR.adam(self.cfg.lr))
        tree = dict(k=jnp.float32(1.0), pull=jnp.ones(n, jnp.float32), R=jnp.full(n, self.R_u, jnp.float32),
                    u_logged=jnp.asarray(self.arm("u", 1.0, R)["u_logged"]))
        new, _ = LE.actor_step(self.args, LE.band_penalty, actor, self.critic, batch, idx, self.shared, tree)
        self.assertEqual(jax.tree_util.tree_structure(new.params), jax.tree_util.tree_structure(self.clone))

    def test_k_zero_is_learners_actor_step_and_a_one_arm_actor_only_run_bit_for_bit(self):
        """With k = 0 one step is learners.actor_step's, and a one-arm program equals a one-arm learners.ActorOnly
        no-caution run bit for bit. (The room check's own 'none' arm, trained five arms side by side, differs from
        both by lane rounding; no mark depends on it.)"""
        n = self.prep.n
        R = np.full(n, self.R_u)
        arm = self.arm("k0", 0.0, R)
        idx = jnp.arange(256, dtype=jnp.int32)
        batch = jax.tree_util.tree_map(lambda x: x[idx], self.prep.train)
        actor = TrainState.create(apply_fn=LR.ACTOR.apply, params=self.clone, tx=LR.adam(self.cfg.lr))
        tree = dict(k=jnp.float32(0.0), pull=jnp.ones(n, jnp.float32), R=jnp.asarray(arm["R"]),
                    u_logged=jnp.asarray(arm["u_logged"]))
        mine, _ = jax.jit(lambda a: LE.actor_step(self.args, LE.band_penalty, a, self.critic, batch, idx,
                                                  self.shared, tree))(actor)
        theirs, _ = jax.jit(lambda a, t: LR.actor_step(self.args, a, self.critic, batch, t[idx]))(
            actor, jnp.ones(n, jnp.float32))
        self.assertTrue(same(mine.params, theirs.params))
        ao = LR.ActorOnly(self.prep, self.clone, self.cfg, 0, alpha=self.cfg.alpha)
        st, _ = ao.run([LR.Arm("none", "even", 1.0)], [self.critic])
        self.assertTrue(same(self.run_arms(self.shared, [arm])["k0"], ao.arm_params(st, 0)))

    def test_weights_of_one_give_the_unweighted_arm(self):
        """Per-step thresholds all equal to R_u: the per-step arm is the unweighted arm bit for bit."""
        R = np.full(self.prep.n, self.R_u)
        p = self.run_arms(self.shared, [self.arm("unweighted", 1.0, R), self.arm("exact_strong", 1.0, R)])
        self.assertTrue(same(p["unweighted"], p["exact_strong"]))

    def test_band_arms_record_and_check_e_on_the_band_penalty(self):
        n = self.prep.n
        ao = LR.ActorOnly(self.prep, self.clone, self.cfg, 0, alpha=self.cfg.alpha)
        st, _ = ao.run([LR.Arm("none", "even", 1.0)], [self.critic])
        R_s = self.R_u * (1.0 + 0.3 * np.random.default_rng(1).uniform(size=n))
        arms = [dict(name="unweighted", R=np.full(n, self.R_u), pool=[self.R_u]),
                dict(name="exact_strong", R=R_s, pool=R_s)]
        params, rec = LE.band_arms(self.prep, self.band, self.critic, arms, 1.0, self.cfg, 0, LI.SYNTHETIC,
                                   none_params=ao.arm_params(st, 0), check_e_rule=RULE_E)
        self.assertEqual(set(params), {"unweighted", "exact_strong"})
        self.assertEqual(rec["k"], 1.0)
        self.assertEqual(rec["training"]["rows_sha256"], ao.rows_sha256)
        for name in params:
            a = rec["arms"][name]
            self.assertEqual(a["refused_rows"], 0)
            self.assertTrue(a["check_e"]["reported_only"])
            self.assertIn("passed", a["check_e"])
        self.assertAlmostEqual(rec["arms"]["unweighted"]["R_mean"], self.R_u, places=5)

    def test_bca_arms_refuse_without_scope_bca_lower_edge_and_on_unregistered_data(self):
        arms = [dict(name="unweighted", R=np.full(self.prep.n, self.R_u), pool=[self.R_u])]
        tmp = Path(tempfile.mkdtemp(prefix="shift_iw_le_gate_"))     # holds no data
        try:
            spec, digest = write_spec_copy(tmp / "iw")
            path = write_approval(tmp / "approval.json", digest, scope=("band", "coverage"))
            approval = LI.verify_approval(path, spec, "band")
            with self.assertRaisesRegex(LI.ApprovalError, "does not cover the 'bca_lower_edge' stage"):
                LE.band_arms(self.prep, self.band, self.critic, arms, 1.0, self.cfg, 0, approval)
            path = write_approval(tmp / "approval.json", digest, scope=("band", "coverage", "bca", "bca_estimated"))
            approval = LI.verify_approval(path, spec, "bca")       # the capped pull's go opens nothing here
            with self.assertRaisesRegex(LI.ApprovalError, "does not cover the 'bca_lower_edge' stage"):
                LE.band_arms(self.prep, self.band, self.critic, arms, 1.0, self.cfg, 0, approval)
            path = write_approval(tmp / "approval.json", digest, scope=("band", "coverage", "bca_lower_edge"))
            approval = LI.verify_approval(path, spec, "bca_lower_edge")
            with self.assertRaisesRegex(LI.ApprovalError, "does not cover the 'bca_lower_edge_estimated' stage"):
                LE.band_arms(self.prep, self.band, self.critic, arms, 1.0, self.cfg, 0, approval,
                             stage="bca_lower_edge_estimated")
        finally:
            shutil.rmtree(tmp, ignore_errors=True)
        for bad in (None, True, "approved"):
            with self.assertRaises(LI.ApprovalError):
                LE.band_arms(self.prep, self.band, self.critic, arms, 1.0, self.cfg, 0, bad)
        relabelled = replace(self.prep, world="stall")
        unregistered = replace(self.prep, train=self.prep.train._replace(reward=self.prep.train.reward + 1.0))
        for data in (relabelled, unregistered):
            with self.assertRaises(LI.ApprovalError):
                LE.band_arms(data, self.band, self.critic, arms, 1.0, self.cfg, 0, LI.SYNTHETIC)


# ------------------------------------------------------------------------------------------------ the cloud's harness

@unittest.skipUnless(HARNESS_PATH.is_file(), f"{LE.HARNESS} is not in this repository")
class HarnessTest(unittest.TestCase):
    """The restated parts against the cloud's Stage 1 harness (read-only) on synthetic arrays."""

    @classmethod
    def setUpClass(cls):
        f = Fixture.get()
        cls.cfg, cls.prep, cls.critic, cls.clone = f["cfg"], f["prep"], f["critic"], f["clone"]
        cls.H = load_harness()
        cls.hcfg = cls.H.ProgramConfig(physics=W.PHYSICS)
        hshared = dict(raw32=jnp.asarray(np.asarray(cls.prep.states, np.float32)), obs=cls.prep.train.obs,
                       mean=jnp.asarray(cls.prep.obs_mean), std=jnp.asarray(cls.prep.obs_std))
        cls.hshared, _ = cls.H.with_logged_delta(hshared, cls.critic, cls.hcfg, cls.prep)
        cls.shared, _ = LE.oracle_shared(cls.prep, cls.critic)

    def test_one_step_oracle(self):
        """delta*: the harness's float64 readout equals ours to 1e-9; its float32 in-program step is within its own
        precision bound (1e-3) of our float64 in-program step; the paired main form equals band_terms' main form."""
        rows = np.arange(400)
        pi = np.asarray(LR.actor_rows(self.clone, np.asarray(self.prep.train.obs)[rows]))
        ours = LE.oracle_delta_rows(self.critic, self.shared, rows, pi)
        h64 = self.H.oracle64(self.critic, self.prep, self.prep.states[rows], pi.astype(np.float64), W.PHYSICS)
        np.testing.assert_allclose(ours, h64, atol=1e-9, rtol=0)
        h32 = self.H.oracle32(self.critic, self.hshared, self.hcfg, rows, pi)
        self.assertLessEqual(float(np.max(np.abs(ours - h32))), self.H.ORACLE_PRECISION_TOL)
        d_rec = np.asarray(self.shared["d_logged"])[rows]
        main = self.H.band_terms(np.asarray(self.H.MAIN_BAND), ours, d_rec)
        with enable_x64():
            mine = np.asarray(LE.oracle_penalty(W.PHYSICS, self.critic, self.shared, jnp.asarray(rows, jnp.int32), None,
                                                jnp.asarray(pi), None))
        np.testing.assert_allclose(mine, main, rtol=0, atol=1e-6)    # float32 critic rounding at another batch size
        np.testing.assert_array_equal(np.maximum(d_rec - ours, 0.0), main)

    def test_actor_step(self):
        """With no penalty the harness's actor_step_w (check 3's per-row cloning form) and ours are the same step;
        with the main oracle at k = 1 the two losses and penalties agree up to the float32-against-float64 step."""
        args = LR.make_args(self.cfg, 0, self.cfg.alpha)
        actor = TrainState.create(apply_fn=LR.ACTOR.apply, params=self.clone, tx=LR.adam(self.cfg.lr))
        idx = jnp.arange(256, dtype=jnp.int32)
        batch = jax.tree_util.tree_map(lambda x: x[idx], self.prep.train)
        weighted = replace(self.hcfg, bc_form="weighted")
        h_arm = dict(coef=jnp.zeros(1, jnp.float32), pull=jnp.float32(1.0))
        theirs, _ = jax.jit(lambda a: self.H.actor_step_w(args, "plain", weighted, a, self.critic, batch, idx,
                                                          self.hshared, h_arm))(actor)
        band = dict(cal_params=LI.models()[2].init(jax.random.PRNGKey(0), jnp.zeros(LR.DX), jnp.zeros(LR.DA)),
                    unit=jnp.float32(1.0))
        tree = dict(k=jnp.float32(0.0), pull=jnp.ones(self.prep.n, jnp.float32), R=jnp.ones(self.prep.n, jnp.float32),
                    u_logged=jnp.zeros(self.prep.n, jnp.float32))
        mine, _ = jax.jit(lambda a: LE.actor_step(args, LE.band_penalty, a, self.critic, batch, idx, band, tree))(actor)
        self.assertTrue(same(mine.params, theirs.params))
        h_band = dict(coef=jnp.asarray(self.H.MAIN_BAND, jnp.float32), pull=jnp.float32(1.0))
        _, h_aux = jax.jit(lambda a: self.H.actor_step_w(args, "band", weighted, a, self.critic, batch, idx,
                                                         self.hshared, h_band))(actor)
        pen = LE._penalty("oracle", W.PHYSICS)
        arm = dict(k=jnp.float32(1.0), pull=jnp.ones(self.prep.n, jnp.float32))
        with enable_x64():
            _, aux = jax.jit(lambda a: LE.actor_step(args, pen, a, self.critic, batch, idx, self.shared, arm))(actor)
        self.assertAlmostEqual(float(aux[0]), float(h_aux[0]), delta=1e-3)      # loss
        self.assertAlmostEqual(float(aux[4]), float(h_aux[4]), delta=1e-3)      # mean of k W

    def test_check_e(self):
        """Check (e) on the same rows: ours (float64 step) and the harness's (float32 step) read the same surface."""
        rows = self.H.shape_rows(self.prep, 0)
        self.assertGreater(len(rows), 50)
        theirs = self.H.oracle_shape(self.critic, self.hshared, self.hcfg, self.prep, self.clone, 0,
                                     forms={"main (perfect_band)": self.H.MAIN_BAND})
        h = theirs["forms"]["main (perfect_band)"]
        ours = LE.oracle_shape(self.critic, self.shared, self.prep, self.clone, 0, RULE_E, rows=rows)
        self.assertEqual(list(ours["grid"]), list(self.H.SHAPE_GRID))
        self.assertEqual(RULE_E["max_fall_share"], self.H.SHAPE_MAX_FALL_SHARE)
        self.assertEqual(tuple(RULE_E["slope_points"]), tuple(self.H.SHAPE_GRAD_POINTS))
        self.assertEqual(ours["q1_alone_best_past80_share"], theirs["q1_alone_best_past80_share"])
        for m, v in h["mean_W"].items():
            self.assertAlmostEqual(ours["mean_W"][m], v, delta=1e-3, msg=m)
        self.assertAlmostEqual(ours["fall_share_80_to_81"], h["fall_share_80_to_81"], delta=0.02)
        self.assertAlmostEqual(ours["best_past80_share"], h["best_past80_share"], delta=0.02)
        for key in ("e2_mean_rises", "e4_fewer_best_past80_than_q1"):
            self.assertEqual(ours[key], h[key], key)


if __name__ == "__main__":
    unittest.main()

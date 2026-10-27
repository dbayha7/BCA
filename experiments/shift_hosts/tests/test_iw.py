"""Tests of iw.py on made-up arrays (plan sections 4.2, 4.5 and 4.6; stop rules S2 and S3; choices 16 and 20): the grid,
rounding each mass up, weights of 1 (grid [1.0], R(1) = the stored standard threshold, the standard state bit for bit),
refused rows under both refusal rules (the dose of TD3+BC, ReBRAC and CQL, and IQL's weights), each row's own threshold,
the S3 reproduction check, the IW refresh step and its guard, and the last refresh's readouts (estimated weights and
the shuffled-weights placebo). An experiment-local attachment (plan section 3.1) that changes none of the repository's
declared methods.

Training environment (/opt/bca-venv); the processor by default. On the card (SHIFT_HOSTS_DEVICE=gpu) only the classes
whose arithmetic runs in JAX are run (CARD_CLASSES: the per-row dose and weights, as the restated steps compile them;
the tables are numpy on the processor either way), to keep the shared card's time short. It runs BCA's code (bca_core's
refresh tail, iw.py, and the ports' calibration.dose.frozen_level_dose and calibration.iql_reference.weights_at_reference
for comparison) on made-up arrays only, opened by the guard (approval item 8). The JAX comparisons run op by op
(jax.disable_jit) and compiled (jax.jit), each side as its own program, op by op against op by op and compiled against
compiled.
  env $(launch env) SHIFT_HOSTS_DEVICE=cpu JAX_PLATFORMS=cpu taskset -c 22-23 /opt/bca-venv/bin/python -m \
      experiments.shift_hosts.tests.test_iw
"""

import json
import math
import sys
import time
import unittest
from types import SimpleNamespace
from unittest import mock

import numpy as np

from experiments.shift_hosts import unifloral_import as UI

AREA = "iw"
BLEND, BETA, CAP = 0.5, 3.0, 100.0          # BCA's blend; Unifloral IQL's args.beta and args.exp_adv_clip (iql.py:44, 46)
_FIX = {}


def _bits(x):
    import jax

    return [(np.asarray(v).dtype.str, np.asarray(v).shape, np.asarray(v).tobytes())
            for v in jax.tree_util.tree_leaves(x)]


def _same(a, b):
    return _bits(a) == _bits(b)


def fixture():
    """Built once per process: the made-up arrays (tests/madeup.py, registered with the guard); a made-up standard
    refresh on the made-up bank by BCA's own refresh tail (bca_core.apply_refresh) with made-up targets, critic values,
    a fresh scale network (TD3+BC's constructor) and unit 0.3; the scale network's eta at the training rows; made-up
    advantages; the exact weights' masses and the table on them."""
    if _FIX:
        return _FIX
    UI.prepare()
    import jax
    import jax.numpy as jnp

    from experiments.shift_hosts import bca_core as BC
    from experiments.shift_hosts import iw as IW
    from experiments.shift_hosts.tests import madeup

    m = madeup.build(0)
    rng = np.random.default_rng([20261016, 7779, 60])
    base = BC.fit_key_base(BC.seeds(0).s91)
    scale = BC.make_scale_state("td3_bc", jnp.asarray(m.train.obs.mean(0)), jnp.asarray(m.train.obs.std(0)),
                                BC.scale_init_key(base, "td3_bc"))
    y = m.bank.reward
    q = (y - (0.2 * (1.0 + m.current["bank"]) * rng.standard_normal(len(y))).astype(np.float32)).astype(np.float32)
    eta_bank = scale.apply_fn(scale.params, m.bank.obs, m.bank.action)
    key = BC.posterior_key(BC.refresh_key(BC.seeds(0).s39, 10_000), "td3_bc")
    unit = jnp.asarray(0.3, jnp.float32)
    iw0 = BC.initial_state(scale, m.n, most_caution=True)._replace(unit=unit)
    st0 = BC.initial_state(scale, m.n, most_caution=False)._replace(unit=unit)
    refreshed = BC.apply_refresh(iw0, y, q, eta_bank, key)
    standard = BC.apply_refresh(st0, y, q, eta_bank, key)
    reference = refreshed.state.reference
    eta = scale.apply_fn(reference.cal_params, m.train.obs, m.train.action)
    adv = (0.8 * rng.standard_normal(m.n)).astype(np.float32)
    adv[::11], adv[::13], adv[::17] = 0.0, 2.5, -4.0          # zeros, capped rows (3 x 2.5 > ln 100), large negatives
    masses = m.weight["train"]
    test_masses = m.weight["strong"][~m.in_ball["strong"]]
    grid = IW.mass_grid(np.concatenate([masses, test_masses]))
    started = time.perf_counter()
    table = IW.threshold_table(refreshed.scores, refreshed.stream, m.weight["bank"], grid)
    _FIX.update(m=m, scale=scale, y=y, q=q, eta_bank=eta_bank, key=key, refreshed=refreshed, standard=standard,
                reference=reference, eta=jnp.asarray(eta), adv=jnp.asarray(adv), masses=masses,
                test_masses=test_masses, grid=grid, table=table, table_seconds=time.perf_counter() - started,
                R_u=float(IW.calibrate(refreshed.scores, refreshed.stream).threshold), not_ready=iw0.reference)
    return _FIX


def _both(fn, *args):
    """fn(*args) op by op and compiled (each its own program): (eager result, compiled result)."""
    import jax

    with jax.disable_jit():
        eager = fn(*args)
    return eager, jax.jit(fn)(*args)


def _with_threshold(reference, t):
    """The reference at another threshold t, for comparing doses and weights row by row (after the weighting test's
    with_threshold_only, test_learners_iw.py:54-57). lambda_hat and lambda_hpd are set to t too, so the reference stays
    valid (reference_valid needs threshold = max(lambda_hat, lambda_hpd)), which the IQL port's width checks."""
    import jax.numpy as jnp

    t = jnp.asarray(t, jnp.float32)
    return reference._replace(threshold=t, lambda_hat=t, lambda_hpd=t)


def _port_weights(reference, eta, adv):
    from calibration.iql_reference import weights_at_reference

    return weights_at_reference(SimpleNamespace(decision_gain=1.0), reference, eta, adv, BETA, CAP)


class GridTest(unittest.TestCase):
    """mass_grid: t = 1.01^k from the largest grid mass at most the smallest mass to the smallest at least the largest."""

    @classmethod
    def setUpClass(cls):
        cls.f = fixture()
        from experiments.shift_hosts import iw as IW
        from experiments.shift_hosts.tests import madeup

        cls.IW, cls.madeup = IW, madeup

    def test_grid_brackets_every_mass_and_holds_one(self):
        g = self.IW.mass_grid([0.135, 1.7, 7.39])
        self.assertIn(1.0, g.tolist())                                   # 1.0 exactly when the masses reach it
        self.assertLessEqual(g[0], 0.135)
        self.assertGreater(g[1], 0.135)                                  # the largest grid mass at most the smallest
        self.assertGreaterEqual(g[-1], 7.39)
        self.assertLess(g[-2], 7.39)
        np.testing.assert_allclose(g[1:] / g[:-1], 1.01, rtol=1e-12)
        self.assertEqual(self.IW.GRID_RATIO, 1.01)

    def test_about_400_points_for_the_exact_weights(self):
        """The exact strong-law weight runs from w(0.1) = 0.135 to w(1.1) = 7.389 (plan section 4.2): about 400 grid
        points."""
        lo, hi = self.madeup.ratio_strong([0.1, 1.1])
        self.assertAlmostEqual(lo, 0.1353, places=4)
        self.assertAlmostEqual(hi, 7.389, places=3)
        g = self.IW.mass_grid([lo, hi])
        self.assertTrue(395 <= len(g) <= 410, len(g))
        self.assertTrue(len(self.f["grid"]) <= len(g) + 1)

    def test_weights_of_one_give_the_grid_one(self):
        self.assertEqual(self.IW.mass_grid(np.ones(50)).tolist(), [1.0])
        self.assertEqual(self.IW.mass_grid([1.0]).tolist(), [1.0])

    def test_bad_inputs_are_refused(self):
        for bad in ([0.0, 1.0], [np.inf], [np.nan], [], [-1.0]):
            with self.assertRaises(ValueError):
                self.IW.mass_grid(bad)
        for ratio in (1.0, 0.5, np.inf, np.nan):
            with self.assertRaises(ValueError):
                self.IW.mass_grid([1.0], ratio)

    def test_grid_points_depend_only_on_k(self):
        """A grid built on more masses holds the same points where they overlap, so each training row's grid point
        (and threshold) does not depend on how far the grid extends."""
        f = self.f
        small = self.IW.mass_grid(f["masses"])
        big = f["grid"]
        self.assertTrue(set(small.tolist()) <= set(big.tolist()))
        a = small[np.searchsorted(small, f["masses"], side="left")]
        b = big[np.searchsorted(big, f["masses"], side="left")]
        self.assertEqual(a.tobytes(), b.tobytes())


class TableTest(unittest.TestCase):
    """threshold_table: calibrate at every grid mass with the same stream; monotone; the running maximum."""

    @classmethod
    def setUpClass(cls):
        cls.f = fixture()
        from experiments.shift_hosts import iw as IW

        cls.IW = IW

    def test_each_grid_point_is_calibrate_itself(self):
        f, IW = self.f, self.IW
        tab = f["table"]
        for k in range(0, len(tab.grid), 47):
            c = IW.calibrate(f["refreshed"].scores, f["refreshed"].stream, f["m"].weight["bank"], float(tab.grid[k]))
            self.assertEqual(np.float64(c.threshold).tobytes(), tab.raw[k].tobytes(), k)
        first = IW.calibrate(f["refreshed"].scores, f["refreshed"].stream, f["m"].weight["bank"], float(tab.grid[0]))
        self.assertEqual((tab.lambda_hat, tab.n_eff), (first.lambda_hat, first.n_eff))

    def test_monotone_and_reaching_refusal(self):
        """With the same stream at every test mass, R(t) never falls; it reaches +inf when the test atom is too heavy
        to certify (masses up to 5,000 on a 1.05 grid)."""
        f, IW = self.f, self.IW
        tab = f["table"]
        self.assertTrue(tab.monotone and not tab.running_max_applied)
        self.assertEqual(tab.thresholds.tobytes(), tab.raw.tobytes())
        self.assertTrue(np.all(np.isfinite(tab.thresholds)))             # 300 bank rows certify every exact mass
        heavy = IW.threshold_table(f["refreshed"].scores, f["refreshed"].stream, f["m"].weight["bank"],
                                   IW.mass_grid([0.05, 5000.0], 1.05))
        self.assertTrue(heavy.monotone and not heavy.running_max_applied)
        self.assertTrue(np.isfinite(heavy.raw[0]) and np.isinf(heavy.raw[-1]))
        self.assertTrue(np.all(heavy.raw[1:] >= heavy.raw[:-1]))

    def test_running_maximum_enforces_monotone_thresholds(self):
        IW = self.IW
        fake = iter([1.0, 3.0, 2.0, 4.0])
        with mock.patch.object(IW, "calibrate", lambda *a, **k: SimpleNamespace(threshold=next(fake), lambda_hat=0.5,
                                                                                n_eff=10.0)):
            tab = IW.threshold_table(np.ones(5), [1], np.ones(5), [1.0, 2.0, 3.0, 4.0])
        self.assertFalse(tab.monotone)
        self.assertTrue(tab.running_max_applied)
        self.assertEqual(tab.raw.tolist(), [1.0, 3.0, 2.0, 4.0])
        self.assertEqual(tab.thresholds.tolist(), [1.0, 3.0, 3.0, 4.0])
        with self.assertRaises(ValueError):
            IW.threshold_table(np.ones(5), [1], np.ones(5), [2.0, 1.0])

    def test_summary_round_trips(self):
        f, IW = self.f, self.IW
        s = f["table"].summary(f["R_u"])
        back = IW.table_from_summary(json.loads(json.dumps(s)))
        for name in ("grid", "raw", "thresholds"):
            self.assertEqual(getattr(back, name).tobytes(), getattr(f["table"], name).tobytes())
        self.assertEqual(s["points"], len(f["grid"]))
        self.assertIn("ratio_to_unweighted_at_ends", s)


class RoundingUpTest(unittest.TestCase):
    """lookup: each mass is rounded up to the next grid point, which never lowers a row's threshold."""

    @classmethod
    def setUpClass(cls):
        cls.f = fixture()
        from experiments.shift_hosts import iw as IW

        cls.IW = IW

    def test_each_mass_is_rounded_up_to_the_next_grid_point(self):
        f, IW = self.f, self.IW
        tab = f["table"]
        for masses in (f["masses"], f["test_masses"]):
            R, idx = IW.lookup(tab, masses)
            self.assertTrue(np.all(tab.grid[idx] >= masses))
            self.assertTrue(np.all((idx == 0) | (tab.grid[np.maximum(idx - 1, 0)] < masses)))
            self.assertEqual(R.tobytes(), tab.thresholds[idx].tobytes())
            gaps = IW.lookup_gaps(tab, masses, idx)
            self.assertLessEqual(gaps["max_mass_gap"], 0.01 + 1e-12)
            self.assertEqual(gaps["possibly_rounded_into_refusal"], 0)
        on = tab.grid[[3, 40, 200]]
        self.assertEqual(IW.lookup(tab, on)[1].tolist(), [3, 40, 200])     # a mass on a grid point stays there

    def test_rounding_up_is_conservative(self):
        """No row's looked-up threshold is below the threshold at its own exact mass (R does not fall in t)."""
        f, IW = self.f, self.IW
        pick = np.random.default_rng([20261016, 7779, 61]).choice(len(f["masses"]), 30, replace=False)
        masses = f["masses"][pick]
        R, _ = IW.lookup(f["table"], masses)
        exact = np.array([IW.calibrate(f["refreshed"].scores, f["refreshed"].stream, f["m"].weight["bank"],
                                       float(t)).threshold for t in masses])
        self.assertTrue(np.all(R >= exact))

    def test_masses_beyond_the_grid_and_bad_masses_are_refused(self):
        tab = self.f["table"]
        for bad in ([tab.grid[-1] * 1.5], [0.0], [np.inf], [-1.0], [np.nan]):
            with self.assertRaises(ValueError):
                self.IW.lookup(tab, bad)
        self.assertEqual(self.IW.lookup(tab, [tab.grid[0] / 2])[1].tolist(), [0])   # below the grid: its first point

    def test_row_thresholds_are_the_lookup_in_float32(self):
        f, IW = self.f, self.IW
        R32 = IW.row_thresholds(f["table"], f["masses"])
        R, idx = IW.lookup(f["table"], f["masses"])
        self.assertEqual(R32.dtype, np.float32)
        self.assertEqual(R32.tobytes(), R.astype(np.float32).tobytes())
        self.assertEqual(R32.tobytes(), IW.table_float32(f["table"])[idx].tobytes())
        big = IW.Table(np.array([1.0]), np.array([1e39]), np.array([1e39]), 0.0, 1.0, True, False)
        with self.assertRaises(ValueError):                              # a finite R that overflows float32
            IW.row_thresholds(big, [1.0])
        inf = IW.Table(np.array([1.0]), np.array([np.inf]), np.array([np.inf]), 0.0, 1.0, True, False)
        self.assertTrue(np.isposinf(IW.row_thresholds(inf, [1.0])[0]))


class WeightsOfOneTest(unittest.TestCase):
    """Weights of 1: the grid is [1.0], R(1) is the stored standard threshold, and IW is standard BCA bit for bit (stop
    rule S2's arithmetic; test (b)'s model is test_learners_iw.py:524)."""

    @classmethod
    def setUpClass(cls):
        cls.f = fixture()
        from experiments.shift_hosts import guard
        from experiments.shift_hosts import iw as IW

        cls.IW, cls.guard = IW, guard
        f = cls.f
        n_bank = len(f["m"].bank.reward)
        cls.ones = IW.iw_refresh(f["refreshed"], np.ones(n_bank), np.ones(f["m"].n), IW.mass_grid(np.ones(f["m"].n)),
                                 permission=guard.MADE_UP, bank=f["m"])

    def test_grid_is_one_and_R1_is_the_standard_threshold(self):
        f, IW = self.f, self.IW
        tab = self.ones.table
        self.assertEqual(tab.grid.tolist(), [1.0])
        self.assertEqual(tab.thresholds.tolist(), [f["R_u"]])
        self.assertEqual(np.float32(tab.thresholds[0]).tobytes(),
                         np.asarray(f["reference"].threshold, np.float32).tobytes())
        self.assertTrue(np.isfinite(f["R_u"]))
        self.assertTrue(IW.reproduces(f["reference"], f["refreshed"].scores, f["refreshed"].stream))

    def test_iw_refresh_with_weights_of_one_is_the_standard_state(self):
        """Every leaf of the IW state after the IW step equals the standard state's (same start, same refresh), except
        the carried refusal rule itself."""
        new, std = self.ones.state, self.f["standard"].state
        self.assertTrue(_same(new.thresholds, std.thresholds))
        self.assertTrue(_same(new._replace(most_caution=std.most_caution), std))
        self.assertTrue(bool(new.most_caution) and not bool(std.most_caution))
        rec = self.ones.record
        self.assertEqual((rec["applied"], rec["s3_reproduced"], rec["refused_rows"]), (True, True, 0))
        self.assertEqual(rec["iw"]["grid_points"], 1)

    def test_doses_at_the_standard_threshold_are_frozen_level_dose(self):
        """per_row_dose with every R_i the stored threshold equals calibration/dose.frozen_level_dose (and the ports'
        bc_readout) bit for bit, under both refusal rules, op by op and compiled."""
        import jax.numpy as jnp

        from calibration.dose import frozen_level_dose
        from calibration.reference import reference_valid

        f, IW = self.f, self.IW
        R = self.ones.state.thresholds
        theirs = _both(lambda r, e: frozen_level_dose(r, e, BLEND), f["reference"], f["eta"])
        for flag in (False, True):
            ours = _both(lambda r, e, x, c: IW.per_row_dose(r, e, x, BLEND, c), f["reference"], f["eta"], R,
                         jnp.asarray(flag))
            for (d, refused), t in zip(ours, theirs):
                for name in ("width", "dose", "support_mask", "has_support"):
                    self.assertTrue(_same(getattr(d, name), getattr(t, name)), (flag, name))
                self.assertTrue(_same(d.inputs_valid, t.inputs_valid & reference_valid(f["reference"])))
                self.assertFalse(bool(np.any(np.asarray(refused))))
        dose = np.asarray(ours[0][0].dose)
        self.assertTrue(np.all((dose > 1.0) & (dose < 1.5)) and bool(ours[0][0].inputs_valid))

    def test_iql_weights_at_the_standard_threshold_are_the_ports(self):
        """per_row_iql_weights with every R_i the stored threshold equals the IQL port's weights_at_reference
        (calibration/iql_reference.py:174-181) bit for bit, every field, under both refusal rules."""
        import jax.numpy as jnp

        f, IW = self.f, self.IW
        R = self.ones.state.thresholds
        theirs = _both(_port_weights, f["reference"], f["eta"], f["adv"])
        for flag in (False, True):
            ours = _both(lambda r, e, x, a, c: IW.per_row_iql_weights(r, e, x, a, BETA, CAP, c), f["reference"],
                         f["eta"], R, f["adv"], jnp.asarray(flag))
            for (w, refused), t in zip(ours, theirs):
                self.assertTrue(_same(w, t), flag)
                self.assertFalse(bool(np.any(np.asarray(refused))))
        w, adv = np.asarray(ours[0][0].weights), np.asarray(f["adv"])
        self.assertTrue(np.all(w[adv > 0] >= 1.0) and np.all(w[adv > 0] <= CAP) and np.all(w > 0))
        plain = np.minimum(np.exp(adv.astype(np.float64) * BETA), CAP)
        self.assertTrue(np.all(w[adv > 0] <= plain[adv > 0] * (1 + 1e-6)))   # BCA only shrinks positive rows


class RefusedRowsTest(unittest.TestCase):
    """Refused rows (R = +inf on every third row, test (b3)'s pattern): standard BCA's rule (dose 1; IQL weight 0 and no
    actor update when no row is supported) and IW's most caution (dose 1.5; IQL weight 1 where A > 0, Unifloral's weight
    elsewhere). Every other row unchanged."""

    @classmethod
    def setUpClass(cls):
        import jax.numpy as jnp

        cls.f = fixture()
        from experiments.shift_hosts import iw as IW

        cls.IW = IW
        f = cls.f
        n = f["m"].n
        cls.inf = np.arange(n) % 3 == 0
        cls.R = jnp.asarray(np.where(cls.inf, np.inf, np.float32(f["reference"].threshold)).astype(np.float32))
        cls.R_all = jnp.full((n,), jnp.inf, jnp.float32)

    def _dose(self, R, flag, reference=None):
        import jax.numpy as jnp

        ref = self.f["reference"] if reference is None else reference
        return _both(lambda r, e, x, c: self.IW.per_row_dose(r, e, x, BLEND, c), ref, self.f["eta"], R,
                     jnp.asarray(flag))

    def _weights(self, R, flag, reference=None):
        import jax.numpy as jnp

        ref = self.f["reference"] if reference is None else reference
        return _both(lambda r, e, x, a, c: self.IW.per_row_iql_weights(r, e, x, a, BETA, CAP, c), ref,
                     self.f["eta"], R, self.f["adv"], jnp.asarray(flag))

    def test_doses_under_both_rules(self):
        from calibration.dose import frozen_level_dose

        f, inf = self.f, self.inf
        port = _both(lambda r, e: frozen_level_dose(r, e, BLEND), f["reference"], f["eta"])
        standard, most = self._dose(self.R, False), self._dose(self.R, True)
        for mode in range(2):
            (s, s_ref), (w, w_ref), t = standard[mode], most[mode], port[mode]
            for out, refused in ((s, s_ref), (w, w_ref)):
                self.assertEqual(np.asarray(refused).tolist(), inf.tolist())
                self.assertTrue(np.all(np.isposinf(np.asarray(out.width)[inf])))
                self.assertFalse(np.asarray(out.support_mask)[inf].any())
                self.assertTrue(bool(out.inputs_valid) and bool(out.has_support))
                for name in ("dose", "width", "support_mask"):                     # every other row unchanged
                    self.assertEqual(np.asarray(getattr(out, name))[~inf].tobytes(),
                                     np.asarray(getattr(t, name))[~inf].tobytes(), name)
            self.assertTrue(np.all(np.asarray(s.dose)[inf] == 1.0))                # standard BCA's rule
            self.assertTrue(np.all(np.asarray(w.dose)[inf] == 1.5))                # the most caution, 1 + blend
            self.assertEqual(np.asarray(w.dose).dtype, np.float32)

    def test_every_row_refused(self):
        for flag, value in ((False, 1.0), (True, 1.5)):
            for d, refused in self._dose(self.R_all, flag):
                self.assertTrue(np.all(np.asarray(d.dose) == value) and np.all(np.asarray(refused)))
                self.assertFalse(bool(d.has_support))
                self.assertTrue(bool(d.inputs_valid))

    def test_iql_standard_rule_gives_weight_zero_and_skips(self):
        f, inf = self.f, self.inf
        adv = np.asarray(f["adv"])
        port = _both(_port_weights, f["reference"], f["eta"], f["adv"])
        for (w, refused), t in zip(self._weights(self.R, False), port):
            self.assertEqual(np.asarray(refused).tolist(), inf.tolist())
            self.assertTrue(np.all(np.asarray(w.weights)[inf] == 0.0))             # both signs of the advantage
            self.assertTrue(np.any(adv[inf] > 0) and np.any(adv[inf] <= 0))
            self.assertFalse(np.asarray(w.support_mask)[inf].any())
            for name in ("weights", "support_mask", "used_advantage"):
                self.assertEqual(np.asarray(getattr(w, name))[~inf].tobytes(),
                                 np.asarray(getattr(t, name))[~inf].tobytes(), name)
            self.assertTrue(bool(w.has_support) and bool(w.inputs_valid))
        for w, _ in self._weights(self.R_all, False):                              # no supported row: skip
            self.assertFalse(bool(w.has_support))
            self.assertTrue(np.all(np.asarray(w.weights) == 0.0) and bool(w.inputs_valid))

    def test_iql_most_caution_gives_plain_cloning_on_positive_rows(self):
        import jax
        import jax.numpy as jnp

        f, inf = self.f, self.inf
        adv = np.asarray(f["adv"])
        pos = adv > 0
        plain = _both(lambda a: jnp.exp(a * BETA).clip(max=CAP), f["adv"])         # Unifloral's weight, iql.py:245
        standard = self._weights(self.R, False)
        for mode, (w, refused) in enumerate(self._weights(self.R, True)):
            ww = np.asarray(w.weights)
            self.assertTrue(np.all(ww[inf & pos] == 1.0))
            self.assertEqual(ww[inf & ~pos].tobytes(), np.asarray(plain[mode])[inf & ~pos].tobytes())
            self.assertTrue(np.asarray(w.support_mask)[inf].all())
            s = standard[mode][0]
            for name in ("weights", "support_mask", "used_advantage"):             # every other row unchanged
                self.assertEqual(np.asarray(getattr(w, name))[~inf].tobytes(),
                                 np.asarray(getattr(s, name))[~inf].tobytes(), name)
            used = np.asarray(w.used_advantage)
            self.assertTrue(np.all(used[inf & pos] == 0.0) and used[inf & ~pos].tobytes() == adv[inf & ~pos].tobytes())
        for mode, (w, _) in enumerate(self._weights(self.R_all, True)):            # every row refused: no skip
            self.assertTrue(bool(w.has_support) and bool(w.inputs_valid))
            expect = np.where(pos, np.float32(1.0), np.asarray(plain[mode]))
            self.assertEqual(np.asarray(w.weights).tobytes(), expect.tobytes())
        del jax

    def test_before_the_first_refresh_nothing_changes(self):
        """Reference not ready (every threshold +inf, as BCAState starts): every dose exactly 1, nothing refused, and
        IQL's weights exactly Unifloral's jnp.exp(adv * beta).clip(max=cap) (iql.py:245), under both rules."""
        import jax.numpy as jnp

        f = self.f
        ready = f["not_ready"]
        self.assertFalse(bool(ready.ready))
        plain = _both(lambda a: jnp.exp(a * BETA).clip(max=CAP), f["adv"])
        self.assertTrue(np.any(np.asarray(plain[0]) == CAP))                       # some rows are capped
        for flag in (False, True):
            for d, refused in self._dose(self.R_all, flag, reference=ready):
                self.assertTrue(np.all(np.asarray(d.dose) == 1.0) and np.all(np.asarray(d.width) == 0.0))
                self.assertFalse(bool(np.any(np.asarray(refused))))
                self.assertTrue(bool(d.inputs_valid) and bool(d.has_support))
            for mode, (w, refused) in enumerate(self._weights(self.R_all, flag, reference=ready)):
                self.assertEqual(np.asarray(w.weights).tobytes(), np.asarray(plain[mode]).tobytes(), (flag, mode))
                self.assertTrue(np.asarray(w.support_mask).all() and bool(w.has_support) and bool(w.inputs_valid))
                self.assertFalse(bool(np.any(np.asarray(refused))))

    def test_the_refusal_rule_is_a_traced_flag(self):
        """Standard and IW BCA compile one program: the rule is a traced boolean, not a constant."""
        import jax
        import jax.numpy as jnp

        f, traces = self.f, []

        def dose(r, e, x, c):
            traces.append(1)
            return self.IW.per_row_dose(r, e, x, BLEND, c)[0].dose

        def weights(r, e, x, a, c):
            traces.append(2)
            return self.IW.per_row_iql_weights(r, e, x, a, BETA, CAP, c)[0].weights

        jd, jw = jax.jit(dose), jax.jit(weights)
        a, b = jd(f["reference"], f["eta"], self.R, jnp.asarray(False)), jd(f["reference"], f["eta"], self.R,
                                                                              jnp.asarray(True))
        c, d = (jw(f["reference"], f["eta"], self.R, f["adv"], jnp.asarray(v)) for v in (False, True))
        self.assertEqual(traces, [1, 2])
        self.assertFalse(_same(a, b) or _same(c, d))
        with self.assertRaises(TypeError):
            self.IW.per_row_dose(f["reference"], f["eta"], self.R, BLEND, jnp.zeros(3, bool))
        with self.assertRaises(TypeError):
            self.IW.per_row_dose(f["reference"], f["eta"], self.R, BLEND, 1)
        with self.assertRaises(ValueError):
            self.IW.per_row_dose(f["reference"], f["eta"], self.R[:-1], BLEND, True)
        with self.assertRaises(ValueError):
            self.IW.per_row_iql_weights(f["reference"], f["eta"], self.R, f["adv"][:-1], BETA, CAP, True)


class EachRowsOwnThresholdTest(unittest.TestCase):
    """Each row is read at its own threshold (test (b2)'s pattern, thresholds at 4 levels, test_learners_iw.py:370-386):
    row i's dose (IQL: weight) is what BCA's own single-threshold rule gives that row at R_i."""

    @classmethod
    def setUpClass(cls):
        cls.f = fixture()
        from experiments.shift_hosts import iw as IW

        cls.IW = IW

    def test_doses_and_weights_at_four_levels(self):
        import jax.numpy as jnp

        from calibration.dose import frozen_level_dose

        f, IW = self.f, self.IW
        R_u = np.float32(f["reference"].threshold)
        levels = (np.float32(f["m"].R_levels) * R_u).astype(np.float32)          # [0.5, 1, 2, 4] x R_u by row % 4
        R = jnp.asarray(levels)
        dose = _both(lambda r, e, x: IW.per_row_dose(r, e, x, BLEND, True)[0], f["reference"], f["eta"], R)
        wts = _both(lambda r, e, x, a: IW.per_row_iql_weights(r, e, x, a, BETA, CAP, False)[0], f["reference"],
                    f["eta"], R, f["adv"])
        for level in np.unique(levels):
            rows = levels == level
            ref = _with_threshold(f["reference"], level)
            own = _both(lambda r, e: frozen_level_dose(r, e, BLEND), ref, f["eta"])
            port = _both(_port_weights, ref, f["eta"], f["adv"])
            for mode in range(2):
                self.assertEqual(np.asarray(dose[mode].dose)[rows].tobytes(),
                                 np.asarray(own[mode].dose)[rows].tobytes())
                self.assertEqual(np.asarray(dose[mode].width)[rows].tobytes(),
                                 np.asarray(own[mode].width)[rows].tobytes())
                self.assertEqual(np.asarray(wts[mode].weights)[rows].tobytes(),
                                 np.asarray(port[mode].weights)[rows].tobytes())
        d = np.asarray(dose[0].dose)
        means = [d[levels == lv].mean() for lv in np.unique(levels)]
        self.assertTrue(np.all(np.diff(means) > 0))                                # a larger R, a larger dose
        np.testing.assert_allclose(d, 1 + BLEND * np.asarray(dose[0].width, np.float64) /
                                   (np.asarray(dose[0].width, np.float64) + float(f["reference"].residual_scale)),
                                   atol=1e-6)

    def test_table_thresholds_row_by_row(self):
        """The IW table's own per-row thresholds (rounded-up masses): every row's dose equals frozen_level_dose at that
        row's R (compiled; every distinct R checked)."""
        import jax
        import jax.numpy as jnp

        from calibration.dose import frozen_level_dose

        f, IW = self.f, self.IW
        R = IW.row_thresholds(f["table"], f["masses"])
        mine = jax.jit(lambda r, e, x: IW.per_row_dose(r, e, x, BLEND, True)[0].dose)(f["reference"], f["eta"],
                                                                                     jnp.asarray(R))
        one = jax.jit(lambda r, e, t: frozen_level_dose(r._replace(threshold=t), e, BLEND).dose)
        mine = np.asarray(mine)
        distinct = np.unique(R)
        self.assertGreater(len(distinct), 1)
        for t in distinct:
            rows = R == t
            self.assertEqual(mine[rows].tobytes(),
                             np.asarray(one(f["reference"], f["eta"], jnp.asarray(t, jnp.float32)))[rows].tobytes())


class S3Test(unittest.TestCase):
    """Stop rule S3: an unweighted calibrate on the bank scores reproduces the stored standard threshold bit for bit,
    and anything else is caught."""

    @classmethod
    def setUpClass(cls):
        cls.f = fixture()
        from experiments.shift_hosts import guard
        from experiments.shift_hosts import iw as IW

        cls.IW, cls.guard = IW, guard

    def test_the_refresh_scores_and_stream_reproduce_the_threshold(self):
        f = self.f
        self.assertTrue(self.IW.reproduces(f["reference"], f["refreshed"].scores, f["refreshed"].stream))
        self.assertTrue(self.IW.reproduces(f["reference"], f["refreshed"].scores.copy(), list(f["refreshed"].stream)))

    def test_another_stream_or_score_is_caught(self):
        """reproduces is True exactly when all four stored fields match. Another stream can select the same score
        (the threshold is one of the bank scores, and lambda_hat does not depend on the stream), so ten other streams
        are tried: each verdict must equal the field-by-field comparison, and at least one stream must be caught. A
        changed score at the threshold is always caught."""
        f, IW = self.f, self.IW
        ref, caught = f["reference"], 0
        for k in range(1, 11):
            stream = np.asarray(f["refreshed"].stream, np.uint32).copy()
            stream[-1] += np.uint32(k)
            c = IW.calibrate(f["refreshed"].scores, stream)
            same = all(np.float32(a).tobytes() == np.asarray(b, np.float32).tobytes()
                       for a, b in ((c.threshold, ref.threshold), (c.lambda_hat, ref.lambda_hat),
                                    (c.lambda_hpd, ref.lambda_hpd), (c.n_eff, ref.n_eff)))
            self.assertEqual(IW.reproduces(ref, f["refreshed"].scores, stream), same, k)
            caught += not same
        self.assertGreater(caught, 0)
        scores = f["refreshed"].scores.copy()
        k = int(np.argmin(np.abs(scores - f["R_u"])))                              # the score the threshold sits on
        self.assertEqual(scores[k], f["R_u"])
        scores[k] = np.nextafter(scores[k], np.inf) * (1 + 1e-6)
        self.assertFalse(IW.reproduces(f["reference"], scores, f["refreshed"].stream))
        other = IW.calibrate(f["refreshed"].scores, f["refreshed"].stream, np.ones(len(scores)) * 2.0, 3.0)
        self.assertFalse(IW.reproduces_calibration(f["reference"], other))

    def test_iw_refresh_stops_on_S3(self):
        f, IW = self.f, self.IW
        scores = f["refreshed"].scores.copy()
        k = int(np.argmin(np.abs(scores - f["R_u"])))
        scores[k] = scores[k] * (1 + 1e-6)
        with self.assertRaises(IW.ReproductionError):
            IW.iw_refresh(f["refreshed"], f["m"].weight["bank"], f["masses"], f["grid"], permission=self.guard.MADE_UP,
                          bank=f["m"], scores=scores)

    def test_an_invalid_refresh_keeps_the_previous_thresholds(self):
        f, IW = self.f, self.IW
        bad = f["refreshed"]._replace(state=f["refreshed"].state._replace(thresholds=f["refreshed"].state.thresholds * 2),
                                      record=dict(f["refreshed"].record, valid=False))
        out = IW.iw_refresh(bad, f["m"].weight["bank"], f["masses"], f["grid"], permission=self.guard.MADE_UP,
                            bank=f["m"])
        self.assertIs(out.state, bad.state)
        self.assertEqual((out.record["applied"], out.record["s3_reproduced"], out.table), (False, None, None))


class IWRefreshTest(unittest.TestCase):
    """iw_refresh: each training row's own threshold replaces the standard one; the reference stays BCA's own."""

    @classmethod
    def setUpClass(cls):
        cls.f = fixture()
        from experiments.shift_hosts import guard
        from experiments.shift_hosts import iw as IW

        cls.IW, cls.guard = IW, guard
        f = cls.f
        cls.out = IW.iw_refresh(f["refreshed"], f["m"].weight["bank"], f["masses"], f["grid"],
                                permission=guard.MADE_UP, bank=f["m"].leaves("bank", "td3_bc"))

    def test_rows_get_their_own_thresholds(self):
        f, out, IW = self.f, self.out, self.IW
        R32 = IW.row_thresholds(f["table"], f["masses"])
        self.assertEqual(np.asarray(out.state.thresholds).tobytes(), R32.tobytes())
        self.assertEqual(out.R_rows.tobytes(), IW.lookup(f["table"], f["masses"])[0].tobytes())
        for name in ("grid", "raw", "thresholds"):
            self.assertEqual(getattr(out.table, name).tobytes(), getattr(f["table"], name).tobytes())
        self.assertTrue(_same(out.state._replace(thresholds=f["refreshed"].state.thresholds), f["refreshed"].state))
        R = np.asarray(out.state.thresholds, np.float64)
        self.assertLess(R.min(), R.max())                                          # rows at different weights
        order = np.argsort(f["masses"], kind="stable")
        self.assertTrue(np.all(np.diff(R[order]) >= 0))                            # a heavier row, an R at least as large
        rec = out.record
        self.assertEqual((rec["applied"], rec["s3_reproduced"], rec["refused_rows"]), (True, True, 0))
        self.assertEqual(rec["iw"]["grid_points"], len(f["grid"]))
        self.assertLessEqual(rec["iw"]["max_mass_gap"], 0.01 + 1e-12)
        self.assertTrue(rec["iw"]["monotone"] and not rec["iw"]["running_max_applied"])
        from experiments.shift_hosts import handoff as H

        json.dumps(H.plain(rec))

    def test_the_guard_and_the_setup_are_checked(self):
        from experiments.shift_hosts import gate

        f, IW = self.f, self.IW
        args = (f["refreshed"], f["m"].weight["bank"], f["masses"], f["grid"])
        for permission in (None, "MADE_UP", object()):
            with self.assertRaises(gate.ApprovalError):
                IW.iw_refresh(*args, permission=permission, bank=f["m"])
        bank = f["m"].leaves("bank", "td3_bc")
        with self.assertRaises(gate.ApprovalError):                                # not registered content
            IW.iw_refresh(*args, permission=self.guard.MADE_UP, bank=(bank[0] + np.float32(1e-3),) + bank[1:])
        with self.assertRaises(gate.ApprovalError):
            IW.threshold_tables(f["refreshed"].scores, f["refreshed"].stream, {}, permission=None, bank=f["m"])
        with self.assertRaises(ValueError):                                        # a standard setup's state
            IW.iw_refresh(f["standard"], *args[1:], permission=self.guard.MADE_UP, bank=f["m"])
        with self.assertRaises(ValueError):
            IW.iw_refresh(f["refreshed"], f["m"].weight["bank"][:-1], f["masses"], f["grid"],
                          permission=self.guard.MADE_UP, bank=f["m"])
        with self.assertRaises(ValueError):
            IW.iw_refresh(f["refreshed"], f["m"].weight["bank"], f["masses"][:-1], f["grid"],
                          permission=self.guard.MADE_UP, bank=f["m"])

    def test_refused_rows_are_counted(self):
        """A 36-row bank (the least that certifies at weight 1) cannot certify the heaviest masses: those rows are
        refused (R = +inf), counted, and stored as +inf."""
        import jax.numpy as jnp

        from experiments.shift_hosts import bca_core as BC

        f, IW = self.f, self.IW
        st = BC.initial_state(f["scale"], f["m"].n, most_caution=True)._replace(unit=jnp.asarray(0.3, jnp.float32))
        small = BC.apply_refresh(st, f["y"][:36], f["q"][:36], f["eta_bank"][:36], f["key"])
        self.assertTrue(small.record["certified"])
        out = IW.iw_refresh(small, f["m"].weight["bank"][:36], f["masses"], f["grid"], permission=self.guard.MADE_UP,
                            bank=f["m"])
        R = np.asarray(out.state.thresholds)
        self.assertGreater(out.record["refused_rows"], 0)
        self.assertEqual(out.record["refused_rows"], int(np.sum(np.isinf(R))))
        heavy = f["masses"] >= np.min(f["masses"][np.isinf(R)])
        self.assertTrue(np.all(np.isinf(R[heavy])))                                # refusal sets in at a mass
        self.assertTrue(np.isfinite(R).any())


class ReadoutTest(unittest.TestCase):
    """The last refresh's readouts: the estimated-weight table (choice 20) and the shuffled-weights placebo (plan
    section 4.6)."""

    @classmethod
    def setUpClass(cls):
        cls.f = fixture()
        from experiments.shift_hosts import guard
        from experiments.shift_hosts import iw as IW

        cls.IW, cls.guard = IW, guard

    def test_placebo_permutation(self):
        IW = self.IW
        p = IW.placebo_permutation(0, 300)
        self.assertEqual(p.tolist(), np.random.default_rng([20261016, 0, 95]).permutation(300).tolist())
        self.assertEqual(sorted(p.tolist()), list(range(300)))
        self.assertGreater(int(np.sum(p != np.arange(300))), 250)
        self.assertEqual(p.tolist(), IW.placebo_permutation(0, 300).tolist())
        self.assertNotEqual(p.tolist(), IW.placebo_permutation(1, 300).tolist())
        self.assertEqual(IW.TAG_PLACEBO, 95)

    def test_last_refresh_arms_and_tables(self):
        f, IW = self.f, self.IW
        w = f["m"].weight["bank"]
        masses = np.concatenate([f["masses"], f["test_masses"]])
        noise = np.exp(0.1 * np.random.default_rng([20261016, 7779, 62]).standard_normal(len(w) + len(masses)))
        est = (w * noise[:len(w)], masses * noise[len(w):])
        arms, info = IW.last_refresh_arms(0, (w, masses), est)
        perm = IW.placebo_permutation(0, len(w))
        self.assertEqual(arms["shuffled"][0].tobytes(), w[perm].tobytes())
        self.assertEqual(arms["shuffled"][1].tobytes(), arms["exact"][1].tobytes())
        self.assertEqual(arms["exact"][1].tobytes(), f["grid"].tobytes())
        self.assertEqual(arms["estimated"][1].tobytes(), IW.mass_grid(est[1]).tobytes())
        self.assertEqual(info["placebo"]["stream"], [20261016, 0, 95])
        self.assertAlmostEqual(info["kish"]["shuffled"], info["kish"]["exact"], places=9)
        started = time.perf_counter()
        tabs = IW.threshold_tables(f["refreshed"].scores, f["refreshed"].stream, arms, permission=self.guard.MADE_UP,
                                   bank=f["m"])
        seconds = time.perf_counter() - started
        self.assertEqual(sorted(tabs), ["estimated", "exact", "shuffled"])
        self.assertEqual(tabs["exact"].thresholds.tobytes(), f["table"].thresholds.tobytes())
        self.assertNotEqual(tabs["shuffled"].thresholds.tobytes(), tabs["exact"].thresholds.tobytes())
        for t in tabs.values():
            self.assertTrue(t.monotone)
        # the placebo keeps each row's own (exact) mass; the estimate looks rows up at their estimated masses
        R_sh = IW.lookup(tabs["shuffled"], f["masses"])[0]
        R_est = IW.lookup(tabs["estimated"], est[1][:len(f["masses"])])[0]
        self.assertEqual(R_sh.shape, R_est.shape)
        print(f"\n  three readout tables ({len(f['grid'])} and {len(arms['estimated'][1])} grid points, "
              f"{len(w)} bank rows): {seconds:.1f} s; the fixture's exact table: {f['table_seconds']:.1f} s")

    def test_weights_of_one_make_the_placebo_and_the_estimate_the_exact_table(self):
        f, IW = self.f, self.IW
        ones = np.ones(len(f["m"].weight["bank"]))
        arms, _ = IW.last_refresh_arms(0, (ones, np.ones(5)), (ones, np.ones(5)))
        tabs = IW.threshold_tables(f["refreshed"].scores, f["refreshed"].stream, arms, permission=self.guard.MADE_UP,
                                   bank=f["m"])
        for name in ("estimated", "shuffled"):
            self.assertEqual(tabs[name].thresholds.tobytes(), tabs["exact"].thresholds.tobytes())
        self.assertEqual(tabs["exact"].thresholds.tolist(), [f["R_u"]])


class RefusalRuleTest(unittest.TestCase):
    """Choice 16: each learner's refusal rule, and the carried flag of each setup."""

    def test_rules_for_every_learner(self):
        from experiments.shift_hosts import bca_core as BC
        from experiments.shift_hosts import handoff as H
        from experiments.shift_hosts import iw as IW

        self.assertEqual(sorted(IW.REFUSAL), sorted(BC.LEARNERS))
        self.assertEqual({k: v["new_rule"] for k, v in IW.REFUSAL.items()},
                         dict(td3_bc=False, cql=True, rebrac=True, iql=True))
        for setup, flag in (("standard", False), ("iw", True), ("iw_ones", True), ("iw_repeat", True),
                            ("sham_alone", False)):
            self.assertIs(IW.most_caution(setup), flag)
        with self.assertRaises(ValueError):
            IW.most_caution("alone")
        for learner in BC.LEARNERS:
            for flag in (False, True):
                rec = IW.refusal_record(learner, flag)
                json.dumps(H.plain(rec))
                self.assertEqual(rec["rule"], IW.refusal_name(flag))
                if learner != "iql":
                    self.assertEqual(rec["refused_dose"], 1.5 if flag else 1.0)
        self.assertFalse(IW.refusal_record("td3_bc", True)["new_rule"])
        self.assertTrue(IW.refusal_record("cql", True)["new_rule"])
        self.assertFalse(IW.refusal_record("cql", False)["new_rule"])


class DoseChangeTest(unittest.TestCase):

    def test_dose_change(self):
        from experiments.shift_hosts import iw as IW

        base = np.array([1.1, 1.2, 1.3, 1.4])
        dose = np.array([1.15, 1.2, 1.25, 1.5])
        out = IW.dose_change(dose, base, high=[False, False, True, True])
        d = dose - base
        self.assertEqual(out["change_vs_unweighted"], float(d.mean()))
        self.assertEqual(out["mean_abs_change"], float(np.abs(d).mean()))
        self.assertEqual(out["max_abs_change"], float(np.abs(d).max()))
        self.assertEqual(out["high_current_rows"], 2)
        self.assertEqual(out["high_current_change"], float(d[2:].mean()))
        self.assertTrue(math.isclose(out["mean"], float(dose.mean())))


CARD_CLASSES = ("WeightsOfOneTest", "RefusedRowsTest", "EachRowsOwnThresholdTest")


def _card_module():
    """A module holding only CARD_CLASSES, under this module's name (for tests.run on the card)."""
    import types

    mod = types.ModuleType("experiments.shift_hosts.tests.test_iw")
    for name in CARD_CLASSES:
        setattr(mod, name, globals()[name])
    return mod


if __name__ == "__main__":
    UI.prepare()
    from experiments.shift_hosts import tests as T

    module = _card_module() if T.device() == "gpu" else sys.modules[__name__]
    sys.exit(T.run(module, AREA, extra=UI.record))

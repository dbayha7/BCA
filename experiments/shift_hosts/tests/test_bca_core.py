"""Test (c1), shared BCA arithmetic = the ports' (plan section 4.5): each port's own BCA state is built on made-up
arrays; the port's own target y, critic value q and keys are recorded at every fit and fed to bca_core's fit cores; the
new scale-network parameters, optimizer state and unit must equal the ports' (td3_bc_bca.fit_scale, cql_bca.fit_scale,
rebrac_bca.fit_scale, iql_scale.BootstrapScaleFitter.update) bit for bit. The restated constructors' optimizer settings
(learning rate 1e-3; epsilon 1e-8, and 1e-5 for IQL) are checked against the ports' own train states, with controls
showing the check would see another epsilon. Also: the width and dose against calibration/dose.frozen_level_dose, the
refresh's calibrate wrapper against reference.freeze_reference, and the seed and key rules. Processor only. An
experiment-local attachment (plan section 3.1) that changes none of the repository's declared methods.

This runs BCA's code (the ports' fits and refreshes and the restated cores) on made-up arrays only, opened by the
guard (approval item 8). Each fit runs both op by op (jax.disable_jit) and compiled (jax.jit, port and core each as
its own program).
  env $(launch env) SHIFT_HOSTS_DEVICE=cpu taskset -c 0-3 /opt/bca-venv/bin/python -m \
      experiments.shift_hosts.tests.test_bca_core
"""

import sys
import unittest

import numpy as np

from experiments.shift_hosts import unifloral_import as UI

AREA = "scaffold"
BATCH, STEPS = 256, 4
MODES = ("eager", "jit")


def _bits(x):
    import jax

    return [(np.asarray(v).dtype.str, np.asarray(v).shape, np.asarray(v).tobytes())
            for v in jax.tree_util.tree_leaves(x)]


def _same(a, b):
    return _bits(a) == _bits(b)


class _Ports:
    """The four ports' BCA states on made-up arrays, built by the ports' own constructors."""

    def __init__(self, m):
        import jax
        import jax.numpy as jnp

        import algorithms.cql as CQ
        import algorithms.cql_bca as CB
        import algorithms.iql as IQ
        import algorithms.rebrac as RB
        import algorithms.rebrac_bca as RBB
        import algorithms.td3_bc as TB
        import algorithms.td3_bc_bca as TBB
        import calibration.iql_reference as IR
        import calibration.iql_scale as IS
        import runtime.networks as C
        from calibration.reference import WBCPConfig

        self.m = m
        self.mods = dict(TB=TB, TBB=TBB, CQ=CQ, CB=CB, RB=RB, RBB=RBB, IQ=IQ, IS=IS, IR=IR, C=C)
        tr = m.train
        self.train = C.Transition(*(jnp.asarray(x) for x in (tr.obs, tr.action, tr.reward, tr.next_obs, tr.done)))
        self.train_na = C.TransitionNA(*(jnp.asarray(x) for x in (tr.obs, tr.action, tr.reward, tr.next_obs, tr.done,
                                                                  m.next_action["train"])))
        self.td3 = dict(args=TB.Args(allow_off_config=True),
                        config=TBB.Config(arm="bca", posterior=WBCPConfig(), blend=0.5))
        self.td3["rng"], self.td3["state"], self.td3["models"] = TBB.initialize(self.td3["args"], self.td3["config"],
                                                                              13, 4)
        self.cql = dict(args=CQ.Args(allow_off_config=True), config=CB.Config(arm="bca"))
        self.cql["rng"], self.cql["state"], self.cql["models"] = CB.initialize(self.cql["args"], self.cql["config"],
                                                                             13, 4)
        self.rebrac = dict(args=RB.Args(allow_off_config=True),
                           config=RBB.Config(arm="bca", posterior=WBCPConfig(), blend=0.5))
        self.rebrac["rng"], self.rebrac["state"], self.rebrac["models"] = RBB.initialize(
            self.rebrac["args"], self.rebrac["config"], self.train_na)
        iargs = IQ.Args(allow_off_config=True)
        agent, _, _ = IQ.initialize_agent(iargs, 13, 4, 1.0)
        posterior = IR.PosteriorArgs(mode="full", reserve_size=300)
        fitter = IS.BootstrapScaleFitter(posterior, agent.qf.apply_fn, 13, 4, iargs.seed, weight_width=True)
        self.iql = dict(args=iargs, agent=agent, fitter=fitter, state=fitter.initial, posterior=posterior)
        self.keys = [jax.random.fold_in(jax.random.PRNGKey(20261016), 7779 + k) for k in range(STEPS)]
        rng = np.random.default_rng([20261016, 7779, 2])
        self.rows = [rng.integers(0, m.n, BATCH) for _ in range(STEPS)]

    def batch(self, k, na=False):
        import jax

        src = self.train_na if na else self.train
        return jax.tree_util.tree_map(lambda x: x[self.rows[k]], src)


class FitCoreAgainstPortsTest(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        UI.prepare(device="cpu")
        from experiments.shift_hosts import guard
        from experiments.shift_hosts.tests import madeup

        cls.m = madeup.build(0)
        guard.require_made_up(cls.m)          # BCA's code runs on these arrays because they are registered as made-up
        cls.p = _Ports(cls.m)

    def _generic(self, learner, mode):
        """STEPS fits of one port and of fit_scale_core from the same start, each fed the port's own y, q and key."""
        import jax

        from experiments.shift_hosts import bca_core as BC

        P, M = self.p, self.p.mods
        spec = dict(td3_bc=(P.td3, M["TBB"]), cql=(P.cql, M["CB"]), rebrac=(P.rebrac, M["RBB"]))[learner]
        d, port = spec
        args, config, models = d["args"], d["config"], d["models"]
        state = d["state"]
        scale, unit = state.calibrator, state.residual_scale
        fold = BC.FOLDS[learner]

        def port_fit(st, b, k):
            if learner == "rebrac":
                return port.fit_scale(args, config, models, st, b, k)
            return port.fit_scale(args, config, models, st, b, k, 1.0)

        def port_parts(st, b, k):
            if learner == "td3_bc":
                y = port.native_target(args, models, st.native, b, jax.random.fold_in(k, fold.fit_noise), 1.0)
                q = jax.lax.stop_gradient(models[1].apply(st.native.critic.params, b.obs, b.action).min(axis=-1))
            elif learner == "cql":
                y = port.native_target(args, models, st.native, b, jax.random.fold_in(k, fold.fit_noise), 1.0)
                q = jax.lax.stop_gradient(jax.numpy.minimum(models[1].apply(st.native.critic1.params, b.obs, b.action),
                                                            models[2].apply(st.native.critic2.params, b.obs, b.action)))
            else:
                y = port.native_target(args, models, st.native, b, jax.random.fold_in(k, fold.fit_noise))
                q = jax.lax.stop_gradient(models[1].apply(st.native.critic.params, b.obs, b.action).min(0))
            return y, q

        def core(sc, u, o, a, y, q, k):
            return BC.fit_scale_core(sc, u, o, a, y, q, k, bootstrap_fold=fold.bootstrap)

        if mode == "jit":
            port_fit, port_parts, core = jax.jit(port_fit), jax.jit(port_parts), jax.jit(core)
        for k in range(STEPS):
            b, key = P.batch(k, na=(learner == "rebrac")), P.keys[k]
            y, q = port_parts(state, b, key)
            new_state, diag = port_fit(state, b, key)
            ours = core(scale, unit, b.obs, b.action, y, q, key)
            self.assertTrue(bool(diag["scale_inputs_valid"]) and bool(ours.valid), (learner, mode, k))
            self.assertTrue(_same(new_state.calibrator.params, ours.scale.params), (learner, mode, k, "params"))
            self.assertTrue(_same(new_state.calibrator.opt_state, ours.scale.opt_state), (learner, mode, k, "adam"))
            self.assertEqual(int(new_state.calibrator.step), int(ours.scale.step))
            self.assertTrue(_same(new_state.residual_scale, ours.unit), (learner, mode, k, "unit"))
            self.assertTrue(_same(diag["scale_loss"], ours.loss), (learner, mode, k, "loss"))
            self.assertFalse(_same(new_state.calibrator.params, state.calibrator.params))   # the fit moved
            state, scale, unit = new_state, ours.scale, ours.unit

    def _iql(self, mode):
        import jax

        from experiments.shift_hosts import bca_core as BC

        P, IQ = self.p, self.p.mods["IQ"]
        fitter, agent, args = P.iql["fitter"], P.iql["agent"], P.iql["args"]
        cal = P.iql["state"]
        scale, unit = cal.calibrator, cal.resid_scale
        q_apply, v_apply = agent.qf.apply_fn, agent.vf.apply_fn

        def nuisance(st, b):
            return IQ.nuisance_update(args, q_apply, v_apply, st, b)

        def port_fit(c, st, b, y, k, step):
            return fitter.update(c, st, b, y, k, step)

        def heads_of(st, b):
            return q_apply(st.qf_target.params, b.obs, b.action)

        def core(sc, u, o, a, y, h, k):
            return BC.iql_fit_core(sc, u, o, a, y, h, k)

        if mode == "jit":
            nuisance, port_fit, heads_of, core = (jax.jit(nuisance), jax.jit(port_fit), jax.jit(heads_of),
                                                  jax.jit(core))
        for k in range(STEPS):
            b, key = P.batch(k), P.keys[k]
            agent, _, y, _ = nuisance(agent, b)
            h = heads_of(agent, b)
            out, readout, accepted = port_fit(cal, agent, b, y, key, k + 1)
            ours = core(scale, unit, b.obs, b.action, y, h, key)
            self.assertTrue(bool(accepted) and bool(ours.valid), (mode, k))
            self.assertTrue(_same(out.calibrator.params, ours.scale.params), (mode, k, "params"))
            self.assertTrue(_same(out.calibrator.opt_state, ours.scale.opt_state), (mode, k, "adam"))
            self.assertTrue(_same(out.resid_scale, ours.unit), (mode, k, "unit"))
            self.assertTrue(_same(readout.diagnostics["cal_loss"], ours.loss), (mode, k, "loss"))
            self.assertFalse(_same(out.calibrator.params, cal.calibrator.params))
            cal, scale, unit = out, ours.scale, ours.unit

    def test_td3_bc_fit_core_equals_port(self):
        import jax

        with jax.disable_jit():
            self._generic("td3_bc", "eager")
        self._generic("td3_bc", "jit")

    def test_cql_fit_core_equals_port(self):
        import jax

        with jax.disable_jit():
            self._generic("cql", "eager")
        self._generic("cql", "jit")

    def test_rebrac_fit_core_equals_port(self):
        import jax

        with jax.disable_jit():
            self._generic("rebrac", "eager")
        self._generic("rebrac", "jit")

    def test_iql_fit_core_equals_port(self):
        import jax

        with jax.disable_jit():
            self._iql("eager")
        self._iql("jit")


class ConstructorTest(unittest.TestCase):
    """The restated scale-network constructors against each port's own, including the optimizer settings."""

    @classmethod
    def setUpClass(cls):
        UI.prepare(device="cpu")
        from experiments.shift_hosts.tests import madeup

        cls.m = madeup.build(0)
        cls.p = _Ports(cls.m)

    def _adam_matches(self, port_state, ours, other_eps):
        """Same params, then the same three gradient steps (one tiny, where epsilon matters, two ordinary) give the same
        params and Adam state bit for bit; the same steps at another epsilon (or rate) give different params."""
        import jax
        import jax.numpy as jnp
        import optax
        from flax.training.train_state import TrainState

        self.assertTrue(_same(port_state.params, ours.params))
        leaves, tree = jax.tree_util.tree_flatten(port_state.params)
        rng = np.random.default_rng(5)
        grads = [jax.tree_util.tree_unflatten(tree, [jnp.asarray(rng.standard_normal(x.shape) * s, x.dtype)
                                                     for x in leaves]) for s in (3e-9, 1.0, 0.1)]
        a, b = port_state, ours
        wrong_eps = TrainState.create(apply_fn=ours.apply_fn, params=ours.params, tx=optax.adam(1e-3, eps=other_eps))
        wrong_lr = TrainState.create(apply_fn=ours.apply_fn, params=ours.params,
                                     tx=optax.adam(2e-3, eps=1e-5 if other_eps == 1e-8 else 1e-8))
        for g in grads:
            a, b = a.apply_gradients(grads=g), b.apply_gradients(grads=g)
            wrong_eps, wrong_lr = wrong_eps.apply_gradients(grads=g), wrong_lr.apply_gradients(grads=g)
            self.assertTrue(_same(a.params, b.params) and _same(a.opt_state, b.opt_state))
        self.assertFalse(_same(a.params, wrong_eps.params))
        self.assertFalse(_same(a.params, wrong_lr.params))

    def test_calibrator_constructors_equal_the_ports(self):
        """TD3+BC and CQL (Calibrator(0, 1), key fold_in(rng, 1128352841)) and ReBRAC (the training rows' mean and
        std, key fold_in(rng, 1380141390)): same parameters, same apply, optax.adam(1e-3) at epsilon 1e-8."""
        import jax

        from experiments.shift_hosts import bca_core as BC

        P = self.p
        for learner, d in (("td3_bc", P.td3), ("cql", P.cql), ("rebrac", P.rebrac)):
            with self.subTest(learner=learner):
                port = d["state"].calibrator
                key = jax.random.fold_in(d["rng"], BC.FOLDS[learner].scale_init)
                if learner == "rebrac":
                    mean, std = d["state"].cal_obs_mean, d["state"].cal_obs_std
                else:
                    mean, std = jax.numpy.zeros(13), jax.numpy.ones(13)
                ours = BC.make_scale_state(learner, mean, std, key)
                b = P.batch(0)
                self.assertTrue(_same(port.apply_fn(port.params, b.obs, b.action),
                                      ours.apply_fn(ours.params, b.obs, b.action)))
                self._adam_matches(port, ours, other_eps=1e-5)

    def test_iql_constructor_equals_the_port(self):
        """IQL: StateDependentCalibrator(0, 0.999, 64) with key fold_in(PRNGKey(seed), 1111703881) through
        create_cal_train_state: the same parameters and apply, optax.adam(1e-3) at epsilon 1e-5."""
        import jax

        from experiments.shift_hosts import bca_core as BC

        port = self.p.iql["state"].calibrator
        key = jax.random.fold_in(jax.random.PRNGKey(self.p.iql["args"].seed), BC.FOLDS["iql"].scale_init)
        ours = BC.make_scale_state("iql", jax.numpy.zeros(13), jax.numpy.full(13, 1.0 - 0.001), key)
        b = self.p.batch(0)
        self.assertTrue(_same(port.apply_fn(port.params, b.obs, b.action), ours.apply_fn(ours.params, b.obs, b.action)))
        self._adam_matches(port, ours, other_eps=1e-8)


class DoseAndRefreshTest(unittest.TestCase):
    """bca_core's width, dose and refresh wrapper against the ports' own functions."""

    @classmethod
    def setUpClass(cls):
        UI.prepare(device="cpu")
        import jax
        import jax.numpy as jnp

        from experiments.shift_hosts import bca_core as BC
        from experiments.shift_hosts.tests import madeup

        cls.m = madeup.build(0)
        cls.p = _Ports(cls.m)
        d = cls.p.td3
        args, models, state = d["args"], d["models"], d["state"]
        bk = cls.m.bank
        cls.bank = cls.p.mods["C"].Transition(*(jnp.asarray(x) for x in bk))
        key = jax.random.PRNGKey(39)
        cls.y = cls.p.mods["TBB"].native_target(args, models, state.native, cls.bank,
                                                jax.random.fold_in(key, 1213156420), 1.0)
        cls.q = models[1].apply(state.native.critic.params, cls.bank.obs, cls.bank.action).min(axis=-1)
        cls.eta = models[2].apply(state.calibrator.params, cls.bank.obs, cls.bank.action)
        cls.state, cls.key = state, jax.random.fold_in(key, 1347375956)
        cls.cal = BC.calibrate_bank(state.calibrator.params, state.residual_scale, cls.eta, cls.y - cls.q, cls.key)

    def test_calibrate_bank_is_freeze_reference(self):
        from calibration.reference import freeze_reference, WBCPConfig

        ref, valid, diag = freeze_reference(self.state.calibrator.params, self.state.residual_scale, self.eta,
                                            self.y - self.q, self.key, WBCPConfig())
        self.assertTrue(valid and self.cal.valid)
        self.assertTrue(_same(ref, self.cal.reference))
        self.assertTrue(np.isfinite(float(ref.threshold)) and bool(ref.ready))

    def test_scores_and_stream_reproduce_the_threshold(self):
        """An unweighted wbcp.calibrate on the returned scores and stream gives the stored threshold, lambda_hat,
        lambda_hpd and n_eff bit for bit (the reproduction S3 needs)."""
        from calibration import wbcp

        c = wbcp.calibrate(self.cal.scores, np.random.default_rng(self.cal.stream), alpha=0.1, beta=0.95, draws=1000)
        r = self.cal.reference
        for a, b in ((c.threshold, r.threshold), (c.lambda_hat, r.lambda_hat), (c.lambda_hpd, r.lambda_hpd),
                     (c.n_eff, r.n_eff)):
            self.assertEqual(np.float32(a).tobytes(), np.asarray(b, np.float32).tobytes())
        self.assertEqual(self.cal.scores.dtype, np.float64)
        self.assertEqual(self.cal.stream.dtype, np.uint32)

    def test_bank_minimum(self):
        from experiments.shift_hosts import bca_core as BC

        p = self.state.calibrator.params
        u = self.state.residual_scale
        with self.assertRaises(ValueError):
            BC.calibrate_bank(p, u, self.eta[:35], (self.y - self.q)[:35], self.key)
        self.assertTrue(BC.calibrate_bank(p, u, self.eta[:36], (self.y - self.q)[:36], self.key).valid)

    def test_width_and_dose_equal_frozen_level_dose(self):
        """dose(width(R, u, eta), u, 0.5) equals calibration/dose.frozen_level_dose bit for bit; not ready: dose exactly
        1; R = +inf: dose 1, unsupported (standard BCA's rule)."""
        import jax.numpy as jnp

        from calibration.dose import frozen_level_dose
        from experiments.shift_hosts import bca_core as BC

        ref = self.cal.reference
        mine = BC.dose(BC.width(ref.threshold, ref.residual_scale, self.eta), ref.residual_scale, 0.5)
        theirs = frozen_level_dose(ref, self.eta, 0.5)
        self.assertTrue(_same(mine.dose, theirs.dose) and _same(mine.width, theirs.width))
        d = np.asarray(mine.dose)
        self.assertTrue(np.all((d > 1.0) & (d < 1.5)))
        np.testing.assert_allclose(d, 1 + 0.5 * np.asarray(mine.width, np.float64) /
                                   (np.asarray(mine.width, np.float64) + float(ref.residual_scale)), rtol=1e-6)
        fresh = BC.initial_state(BC.make_scale_state("td3_bc", jnp.zeros(13), jnp.ones(13), self.key), 5)
        self.assertTrue(np.all(np.asarray(BC.frozen_dose(fresh.reference, self.eta, 0.5).dose) == 1.0))
        inf = BC.dose(BC.width(jnp.inf, ref.residual_scale, self.eta), ref.residual_scale, 0.5)
        self.assertTrue(np.all(np.asarray(inf.dose) == 1.0) and not bool(np.any(np.asarray(inf.support_mask))))

    def test_apply_refresh(self):
        """The refresh tail: freezes the live scale network and unit, sets the reference and fills every row's
        threshold; its record is JSON-ready."""
        import jax.numpy as jnp

        from experiments.shift_hosts import bca_core as BC
        from experiments.shift_hosts import handoff as H

        st = BC.initial_state(self.state.calibrator, self.m.n)
        st = st._replace(unit=jnp.asarray(self.state.residual_scale, jnp.float32))
        out = BC.apply_refresh(st, self.y, self.q, self.eta, self.key)
        self.assertTrue(out.record["valid"] and out.record["certified"])
        self.assertTrue(_same(out.state.reference, self.cal.reference))
        self.assertTrue(np.all(np.asarray(out.state.thresholds) == np.float32(self.cal.reference.threshold)))
        self.assertEqual(out.state.thresholds.shape, (self.m.n,))
        self.assertEqual(out.scores.tobytes(), self.cal.scores.tobytes())
        self.assertTrue(out.record["new_reference_valid"])
        again = BC.apply_refresh(st, self.y, self.q, self.eta, self.key, check_new_reference=True)
        self.assertTrue(again.record["valid"] and _same(again.state.reference, out.state.reference))
        H.plain(out.record)

    def test_new_reference_check(self):
        """ReBRAC's refresh check of the NEW frozen reference (rebrac_bca.py:370; check_new_reference=True): a refresh
        whose frozen reference fails reference_valid (here a made-up one whose threshold is not max(lambda_hat,
        lambda_hpd)) is not valid and keeps the previous state; without the check (the TD3+BC, CQL and IQL ports) the
        same refresh is accepted, as those ports accept it. The record says new_reference_valid false either way."""
        from unittest import mock

        import jax.numpy as jnp

        from experiments.shift_hosts import bca_core as BC

        st = BC.initial_state(self.state.calibrator, self.m.n)
        st = st._replace(unit=jnp.asarray(self.state.residual_scale, jnp.float32))
        broken = self.cal._replace(reference=self.cal.reference._replace(
            threshold=jnp.asarray(float(self.cal.reference.threshold) * 2.0, jnp.float32)))
        self.assertFalse(bool(BC.reference_valid(broken.reference)))
        with mock.patch.object(BC, "calibrate_bank", lambda *a, **k: broken):
            strict = BC.apply_refresh(st, self.y, self.q, self.eta, self.key, check_new_reference=True)
            loose = BC.apply_refresh(st, self.y, self.q, self.eta, self.key)
        self.assertFalse(strict.record["valid"])
        self.assertFalse(strict.record["new_reference_valid"])
        self.assertIs(strict.state, st)
        self.assertTrue(loose.record["valid"])
        self.assertFalse(loose.record["new_reference_valid"])
        self.assertTrue(_same(loose.state.reference, broken.reference))


class SeedTest(unittest.TestCase):

    def test_seed_rules(self):
        import jax

        from experiments.shift_hosts import bca_core as BC

        UI.prepare(device="cpu")
        s = BC.seeds(0)
        for tag, value in ((31, s.unifloral), (91, s.s91), (39, s.s39)):
            self.assertEqual(value, int(np.random.default_rng([20261016, 0, tag]).integers(2 ** 31 - 1)))
        base = BC.fit_key_base(s.s91)
        self.assertTrue(_same(BC.fit_key(base, 7), jax.random.fold_in(jax.random.PRNGKey(s.s91), 7)))
        self.assertTrue(_same(BC.scale_init_key(base, "rebrac"),
                              jax.random.fold_in(jax.random.fold_in(base, 0), 1380141390)))
        self.assertTrue(_same(BC.refresh_key(s.s39, 10_000),
                              jax.random.fold_in(jax.random.PRNGKey(s.s39), 10_000)))
        steps = BC.refresh_steps()
        self.assertEqual((steps[0], steps[-1], len(steps)), (10_000, 95_000, 18))
        with self.assertRaises(ValueError):
            BC.fit_noise_key(base, "iql")


if __name__ == "__main__":
    UI.prepare(device="cpu")
    from experiments.shift_hosts import tests as T

    sys.exit(T.run(sys.modules[__name__], AREA, device_name="cpu", extra=UI.record))

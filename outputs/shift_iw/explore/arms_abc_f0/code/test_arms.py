"""EXPLORATORY: unit checks of arms A, B, C, F0 and the scorer on small synthetic arrays (frozen pass marks, section 8
step 1). Nothing here reads ThrusterChain data or writes outside a temporary folder.

  cd <sandbox>/repo; PYTHONPATH=<sandbox>/repo:<sandbox>/explore:<this folder> JAX_PLATFORMS=cpu python -B -m unittest
  test_arms   (run_arms.sh test does this with the jobs' own environment)
"""
import sys

sys.dont_write_bytecode = True

import inspect  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import shutil  # noqa: E402
import tempfile  # noqa: E402
import unittest  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

import common as C  # noqa: E402

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
from flax.training.train_state import TrainState  # noqa: E402

from experiments.shift import learners as LR  # noqa: E402
from experiments.shift_iw import learners_iw as LI  # noqa: E402
from experiments.shift_iw import world as W  # noqa: E402

import arm_a  # noqa: E402
import arm_b  # noqa: E402
import arm_c  # noqa: E402

_SYN = {}


def synthetic(n=2048):
    """A registered synthetic Prepared, a short TD3+BC host on it and BCA's band on the host (SYNTHETIC token)."""
    if n not in _SYN:
        cfg = LI.settings(updates=300, clone_steps=200, band_fit_steps=200, actor_only_steps=500,
                          actor_only_chunk=100, chunk=300)
        prep = LI.synthetic_prepared(n=n, bank=300, eval_states=64)
        _, host, _ = LI.train_host(prep, cfg, 0, updates=300)
        band = LI.fit_band(host, prep, cfg, 0, LI.SYNTHETIC, steps=200)
        _SYN[n] = (cfg, prep, host, band)
    return _SYN[n]


class ArmA(unittest.TestCase):
    def test_margin_inf_is_none_bit_for_bit_over_a_full_chunk(self):
        cfg = LI.settings(clone_steps=200, actor_only_steps=4000, actor_only_chunk=4000)
        prep = LI.synthetic_prepared(n=2048, bank=300, eval_states=64)
        clone, _ = LR.clone(prep, cfg, 0)
        runner = LR.ActorOnly(prep, clone, cfg, 0, alpha=cfg.alpha)
        critic = LR.CRITIC.init(jax.random.PRNGKey(5), jnp.zeros(LR.DX), jnp.zeros(LR.DA))
        D = np.random.default_rng(1).normal(size=prep.n) * 10
        guard = np.ones(prep.n, bool)
        w_a1 = arm_a.renormalized(arm_a.margin_release(D, guard, np.full(prep.n, np.inf)))
        tables = np.stack([np.ones(prep.n), w_a1]).astype(np.float32)
        self.assertTrue(np.array_equal(tables[0], tables[1]))
        prog = runner.program(2, True, True)
        states = jax.tree_util.tree_map(lambda x: jnp.broadcast_to(x, (2,) + jnp.shape(x)), runner.start)
        st, _ = prog(critic, states, prep.train, runner.rows[0], jnp.asarray(tables))
        a, b = runner.arm_params(st.params, 0), runner.arm_params(st.params, 1)
        self.assertEqual(LR.tree_sha256(a), LR.tree_sha256(b))

    def test_mean_weight_is_one_after_renormalization(self):
        rng = np.random.default_rng(2)
        for n in (7, 10, 1000, 45301, 49919):
            for k in (0, 1, 3, n // 3, n - 1):
                rel = np.zeros(n, bool)
                rel[rng.choice(n, k, replace=False)] = True
                w = arm_a.renormalized(rel)
                self.assertTrue(np.all(w[rel] == 0))
                self.assertLessEqual(abs(math.fsum(w) / n - 1.0), 4 * np.finfo(float).eps, (n, k))
        self.assertTrue(np.all(arm_a.renormalized(np.ones(5, bool)) == 0))

    def test_guard_blocks_commands_outside_the_bounds(self):
        lo, hi = np.full(4, -0.9), np.full(4, 0.9)
        pi = np.array([[0.0, 0.0, 0.0, 0.0], [0.95, 0, 0, 0], [0, -0.91, 0, 0], [0.9, -0.9, 0.9, -0.9]], np.float32)
        g = arm_a.guard_of(pi, lo, hi)
        self.assertEqual(g.tolist(), [True, False, False, bool(np.float32(0.9) <= 0.9)])
        rel = arm_a.margin_release(np.full(4, 1e9), g, 0.0)
        self.assertEqual(rel.tolist(), g.tolist())

    def test_matched_and_untargeted_release_k_guard_passing_rows(self):
        D = np.array([5.0, 9.0, 9.0, 1.0, 7.0, 3.0])
        guard = np.array([True, True, True, True, False, True])
        tie = np.array([5, 4, 3, 2, 1, 0])
        m = arm_a.matched_release(D, guard, 2, tie)
        self.assertEqual(np.flatnonzero(m).tolist(), [1, 2])              # tie broken by tie order (3 < 4)
        m3 = arm_a.matched_release(D, guard, 3, tie)
        self.assertEqual(np.flatnonzero(m3).tolist(), [0, 1, 2])
        self.assertEqual(int(arm_a.matched_release(D, guard, 99, tie).sum()), 5)
        u = arm_a.untargeted_release(guard, 3, np.random.default_rng(0))
        self.assertEqual(int(u.sum()), 3)
        self.assertFalse(u[4])
        dose = np.linspace(1.0, 1.5, 6)
        self.assertEqual(arm_a.a1c_weights(m, dose).tolist(), np.where(m, 0.0, dose).tolist())

    def test_band_reads_refused_without_the_gate(self):
        cfg, prep, host, band = synthetic()
        ungated = LI.Band(band.world, 0, band.removal, None, band.state, band.refresh_key, band.record, band.data_sha256)
        with self.assertRaises(LI.ApprovalError):
            LI.readout(ungated, band.state.posterior, prep.train)
        with self.assertRaises(LI.ApprovalError):
            LI.per_step_readout(ungated, np.ones(prep.n), prep.train)
        other = LI.Band(band.world, 0, band.removal, LI.SYNTHETIC, band.state, band.refresh_key, band.record, "0" * 64)
        with self.assertRaises(LI.ApprovalError):
            LI.readout(other, band.state.posterior, prep.train)

    def test_end_to_end_on_synthetic_data(self):
        cfg, prep, host, band = synthetic()
        rd = LI.readout(band, band.state.posterior, prep.train, stage="band")
        self.assertTrue(rd["valid"])
        U = np.asarray(rd["width"], np.float64)
        R_iw = np.where(np.arange(prep.n) % 7 == 0, np.inf, float(band.state.posterior.threshold) * 1.3)
        U_iw = np.asarray(LI.per_step_readout(band, R_iw, prep.train, stage="band")["width"], np.float64)
        self.assertTrue(np.all(np.isinf(U_iw[::7])))
        s, info = C.plain_spread(host, prep, cfg, 0)
        self.assertTrue(info["finite"] and s > 0)
        obs, a = np.asarray(prep.train.obs, np.float32), np.asarray(prep.actions, np.float32)
        q_rec = C.q_min(host.critic.params, obs, a)
        clone, _ = LR.clone(prep, cfg, 0)
        runner = LR.ActorOnly(prep, clone, cfg, 0, alpha=cfg.alpha)
        a64 = a.astype(np.float64)
        X = arm_a.Inputs(obs, q_rec, U, U_iw, rd["dose"], s, np.quantile(a64, 0.005, 0), np.quantile(a64, 0.995, 0),
                         runner.tie, runner.perms[0], 0, np.arange(prep.n) % 3 == 0)
        n_arms = len(arm_a.ARMS)
        prog = runner.program(n_arms, True, True)
        states = jax.tree_util.tree_map(lambda x: jnp.broadcast_to(x, (n_arms,) + jnp.shape(x)), runner.start)
        for c in range(runner.steps // runner.chunk):
            params = {name: runner.arm_params(states.params, i) for i, name in enumerate(arm_a.ARMS)}
            pis = {name: C.actor_out(params[name], obs) for name in arm_a.ARMS}
            D = {name: C.q_min(host.critic.params, obs, pis[name]) - q_rec for name in arm_a.ARMS}
            tables, info = arm_a.refresh(X, pis, D, c)
            self.assertEqual(tables.shape, (n_arms, prep.n))
            self.assertTrue(np.all(np.isfinite(tables)))
            k = info["A1"]["k_A1"]
            self.assertEqual(info["matched_share"]["released"], min(k, int(arm_a.guard_of(
                pis["matched_share"], X.lo, X.hi).sum())))
            self.assertEqual(info["untargeted"]["released"], min(k, int(arm_a.guard_of(
                pis["untargeted"], X.lo, X.hi).sum())))
            for name in ("A1", "margin0", "plain_1s", "matched_share", "untargeted", "A1_IW"):
                if info[name]["released"] < prep.n:
                    self.assertAlmostEqual(info[name]["weight_mean_float64"], 1.0, places=12)
            self.assertIn("released_share_random_rows", info["A1"])
            states, _ = prog(host.critic.params, states, prep.train, runner.rows[c], jnp.asarray(tables))
        self.assertTrue(all(C.finite_tree(runner.arm_params(states.params, i)) for i in range(n_arms)))
        k0 = arm_a.k0_check(info["A1"], "medium", 0)
        self.assertFalse(k0["binding"])
        self.assertTrue(arm_a.k0_check(info["A1"], "medium", 1)["binding"])
        self.assertFalse(arm_a.k0_check(info["A1"], "expert", 1)["applies"])


class ArmBC(unittest.TestCase):
    def test_all_ones_map_reproduces_evaluate_actor_on_all_three_banks(self):
        prep = LI.synthetic_prepared(n=256, bank=64, eval_states=16)
        params = LR.ACTOR.init(jax.random.PRNGKey(11), jnp.zeros(LR.DX))
        banks = {law: W.draw_starts(40, np.random.default_rng([3, i]), law) for i, law in enumerate(W.TILT_NAMES)}
        anchors = {law: {"anchor": {"random": -900.0 - 10 * i, "top": -60.0}} for i, law in enumerate(W.TILT_NAMES)}
        ref = LI.evaluate_actor(params, prep, banks, anchors)
        pol = C.ceiling_policy(C.actor_policy(params, prep), [1.0, 1.0, 1.0, 1.0])
        self.assertEqual(json.dumps(C.evaluate_policy(pol, banks, anchors), sort_keys=True),
                         json.dumps(ref, sort_keys=True))
        clipped = C.evaluate_policy(C.ceiling_policy(C.actor_policy(params, prep), [0.05] * 4), banks, anchors)
        self.assertNotEqual(clipped["strong"]["J"], ref["strong"]["J"])

    def test_empty_rung_refused_under_every_rule(self):
        e = np.zeros(0)
        for rule in arm_b.WALK_RULES:
            ok, stat = arm_b.decide_rule(rule, e, e, e, 1.0, 1.0, 0, 1, 1, 0, 1.0)
            self.assertFalse(ok, rule)
            self.assertIsNone(stat, rule)

    def test_negative_refused_and_positive_certified_under_every_rule(self):
        rng = np.random.default_rng(4)
        neg = -rng.uniform(0.1, 1.0, 1000)
        pos = rng.uniform(0.1, 1.0, 1000)
        w = np.ones(1000)
        for rule in arm_b.WALK_RULES:
            ok, _ = arm_b.decide_rule(rule, neg, w, w, 1.0, 1.0, 0, 1, 1, 0, float(neg.max() - neg.min()))
            self.assertFalse(ok, rule)
            ok, _ = arm_b.decide_rule(rule, pos, w, w, 1.0, 1.0, 0, 1, 1, 0, float(pos.max() - pos.min()))
            self.assertTrue(ok, rule)

    def test_identity_permutation_reproduces_bcas_map(self):
        rng = np.random.default_rng(5)
        j = rng.integers(1, 11, 3000)
        g = rng.normal(0.4, 1.0, 3000) - 0.08 * j
        w = rng.uniform(0.2, 3.0, 3000)
        base = arm_b.thruster_map("bca", j, g, w, w, 1.3, 1.3, 0, 1, 0)
        ident = arm_b.thruster_map("bca", j, g, w, w, 1.3, 1.3, 0, 1, 0, perm=np.arange(3000))
        self.assertEqual(json.dumps(base), json.dumps(ident))
        # the shuffled rule is BCA's rule after the permutation: with the identity and BCA's own stream it is BCA's map
        shuf = arm_b.thruster_map("shuffled", j, g, w, w, 1.3, 1.3, 0, 1, -arm_b.BCA_RULES["shuffled"],
                                  perm=np.arange(3000))
        self.assertEqual(json.dumps(base), json.dumps(shuf))

    def test_rungs_walk_and_tries(self):
        E = arm_b.EDGES
        v = np.array([0.5, 0.50001, np.float32(0.8), 0.80001, 1.0, 0.3], np.float32)
        self.assertEqual(arm_b.rung_index(v).tolist(), [0, 1, 6, 7, 10, 0])
        self.assertEqual(arm_b.walk([True, True, False, True], E), float(E[2]))
        self.assertEqual(arm_b.walk([False] * 10, E), 0.5)
        self.assertEqual(arm_b.walk([True] * 10, E), 1.0)
        j = np.array([7, 7, 7, 7])
        a = np.array([0.82, 0.82, -0.82, 0.82], np.float32)
        pi = np.array([0.9, -0.9, -0.81, 0.79], np.float32)
        self.assertEqual(arm_b.try_mask(a, pi, j).tolist(), [True, False, True, False])
        self.assertTrue(arm_b.k1_numbers({"rules": {"bca": [{"rungs": [{"tries": 0}] * 6 + [{"tries": 30}]}]}},
                                         {"bca": [float(E[7]), 1, 1, 1]})["fires"])
        self.assertTrue(arm_b.k1_numbers({"rules": {"bca": [{"rungs": [{"tries": 0}] * 7}]}},
                                         {"bca": [float(E[2]), 1, 1, 1]})["over_refusal"])

    def test_matched_global(self):
        cmds = np.zeros((10, 10, 4))
        cmds[..., 0] = np.linspace(0, 1, 100).reshape(10, 10)
        t, p = arm_b.matched_global(cmds, [0.5, 1.0, 1.0, 1.0])
        self.assertAlmostEqual(p, 50 / 400)
        self.assertGreater(np.mean(np.abs(cmds) > t), 0.0)
        self.assertEqual(arm_b.matched_global(np.zeros((2, 2, 4)), [0.5] * 4), (1.0, 0.0))

    def test_c_loop_equals_actor_only_run(self):
        cfg = LI.settings(clone_steps=100, actor_only_steps=300, actor_only_chunk=100)
        prep = LI.synthetic_prepared(n=512, bank=64, eval_states=16)
        clone, _ = LR.clone(prep, cfg, 0)
        critic = LR.CRITIC.init(jax.random.PRNGKey(9), jnp.zeros(LR.DX), jnp.zeros(LR.DA))
        r1 = LR.ActorOnly(prep, clone, cfg, 0, alpha=cfg.alpha)
        ref, _ = r1.run(arm_c.arms(), [critic])
        r2 = LR.ActorOnly(prep, clone, cfg, 0, alpha=cfg.alpha)
        saved = {}
        mine, _, stopped = arm_c.train(r2, critic, prep.train, save_8000=lambda p: saved.setdefault("p", p))
        self.assertIsNone(stopped)
        self.assertIn("p", saved)
        self.assertEqual(LR.tree_sha256(ref), LR.tree_sha256(mine))

    def test_kb1(self):
        self.assertTrue(arm_c.kb1(1.0, 3.5, 0.2, True)["fires"])
        self.assertFalse(arm_c.kb1(1.0, 2.9, 0.2, True)["fires"])
        self.assertTrue(arm_c.kb1(-1.0, 0.7, 0.2, True)["fires"])
        self.assertFalse(arm_c.kb1(-1.0, 0.5, 0.2, True)["fires"])
        self.assertTrue(arm_c.kb1(1.0, 0.0, 0.2, False)["fires"])


class F0(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import arm_f0
        cls.F = arm_f0

    def tree(self, n, inside_every=5):
        rng = np.random.default_rng(6)
        T = self.F.T
        grid = np.full(T.MAXG, np.inf, np.float32)
        grid[:3] = [0.5, 1.0, 2.0]
        thr = np.full(T.MAXG, np.inf, np.float32)
        thr[:3] = [1.0, 1.5, 2.0]
        return dict(k=jnp.float32(1.0), pull=jnp.ones(n, jnp.float32),
                    mu=jnp.asarray(rng.normal(0, 0.3, (n, 4)), jnp.float32), eps=jnp.full(n, 0.2, jnp.float32),
                    hard=jnp.zeros(n, jnp.float32), log_w=jnp.zeros(n, jnp.float32), log_Z=jnp.zeros(n, jnp.float32),
                    centre=jnp.asarray([0.0, 0.0, 0.0, 0.0], jnp.float32), rho2=jnp.float32(0.04),
                    log_floor=jnp.float32(-9.0), grid=jnp.asarray(grid), thresholds=jnp.asarray(thr),
                    n_grid=jnp.int32(3), cap=jnp.float32(2.0), u_logged=jnp.asarray(rng.uniform(0, .1, n), jnp.float32),
                    q1_rec=jnp.asarray(rng.normal(-5, 1, n), jnp.float32))

    def shared(self):
        cal = LI.models()[2].init(jax.random.PRNGKey(2), jnp.zeros((1, LR.DX)), jnp.zeros((1, LR.DA)))
        return dict(cal_params=cal, unit=jnp.float32(0.7))

    def test_penalty_source_differs_from_a2_only_in_the_last_line(self):
        def body(fn):
            lines = inspect.getsource(fn).splitlines()[1:]
            return [ln.split("#")[0].rstrip() for ln in lines if ln.strip() and not ln.strip().startswith('"""')]
        a2, f0 = body(self.F.F1.penalty_band_f1), body(self.F.penalty_f0)
        self.assertEqual(len(a2), len(f0))
        self.assertEqual(a2[:-1], f0[:-1])
        self.assertEqual(a2[-1].strip(), "return jnp.where(inf_ref, w_ref, w_cert)")
        self.assertEqual(f0[-1].strip(), "return jnp.where(inf_ref, w_ref, 0.0)")

    def test_penalty_values(self):
        n = 400
        tree, shared = self.tree(n), self.shared()
        rng = np.random.default_rng(7)
        obs = jnp.asarray(rng.normal(0, 1, (n, LR.DX)), jnp.float32)
        a = rng.uniform(-0.6, 0.6, (n, 4)).astype(np.float32)
        a[::5] = 0.01                                                        # inside the ball: no credit
        a = jnp.asarray(a)
        critic = LR.CRITIC.init(jax.random.PRNGKey(3), jnp.zeros(LR.DX), jnp.zeros(LR.DA))
        idx = jnp.arange(n)
        _, inf_ref, _ = self.F.F1.band_read_f1(tree, idx, a)
        inf_ref = np.asarray(inf_ref)
        self.assertTrue(inf_ref[::5].all())
        f0 = np.asarray(self.F.penalty_f0(critic, shared, idx, obs, a, tree))
        a2 = np.asarray(self.F.F1.penalty_band_f1(critic, shared, idx, obs, a, tree))
        self.assertTrue(np.array_equal(f0[inf_ref], a2[inf_ref]))
        self.assertTrue(np.all(f0[~inf_ref] == 0.0))
        tree2, shared2 = self.F.band_parts_replaced(tree, shared)
        self.assertTrue(np.array_equal(f0, np.asarray(self.F.penalty_f0(critic, shared2, idx, obs, a, tree2))))
        off = self.F.flags_off(tree)
        self.assertFalse(np.asarray(self.F.F1.band_read_f1(off, idx, a)[1]).any())

    def test_kf2_on_synthetic_data(self):
        cfg = LI.settings(clone_steps=100)
        prep = LI.synthetic_prepared(n=600, bank=64, eval_states=16)
        clone, _ = LR.clone(prep, cfg, 0)
        critic = LR.CRITIC.init(jax.random.PRNGKey(4), jnp.zeros(LR.DX), jnp.zeros(LR.DA))
        args = LR.make_args(cfg, 0, cfg.alpha)
        st = TrainState.create(apply_fn=LR.ACTOR.apply, params=clone, tx=LR.adam(cfg.lr))
        rows = jnp.asarray(np.random.default_rng(8).integers(prep.n, size=(300, 256)), jnp.int32)
        tree, shared = self.tree(prep.n), self.shared()
        off = self.F.flags_off(tree)
        mine, _ = self.F.program(args, "f0")(critic, st, prep.train, shared, rows, off)
        ref, _ = self.F.program(args, "zero")(critic, st, prep.train, shared, rows, off)
        self.assertEqual(LR.tree_sha256(mine.params), LR.tree_sha256(ref.params))
        on, _ = self.F.program(args, "f0")(critic, st, prep.train, shared, rows, tree)
        tree2, shared2 = self.F.band_parts_replaced(tree, shared)
        alt, _ = self.F.program(args, "f0")(critic, st, prep.train, shared2, rows, tree2)
        self.assertEqual(LR.tree_sha256(on.params), LR.tree_sha256(alt.params))
        a2, _ = self.F.program(args, "a2")(critic, st, prep.train, shared, rows, tree)
        self.assertNotEqual(LR.tree_sha256(a2.params), LR.tree_sha256(on.params))   # the band's part does act in A2


class Scorer(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="arms_test_"))

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_refuses_a_changed_frozen_file(self):
        f = self.tmp / C.PASSMARKS.name
        shutil.copy(C.PASSMARKS, f)
        shutil.copy(str(C.PASSMARKS) + ".sha256", str(f) + ".sha256")
        f.chmod(0o644)
        self.assertEqual(C.require_passmarks(f), C.PASSMARKS_SHA256)
        f.write_text(f.read_text() + "\nmoved\n")
        with self.assertRaises(C.Refused):
            C.require_passmarks(f)
        Path(str(f) + ".sha256").chmod(0o644)
        Path(str(f) + ".sha256").write_text(f"{C.sha256_file(f)}  {f.name}\n")
        with self.assertRaises(C.Refused):      # sidecar rewritten to match: still not the frozen digest
            C.require_passmarks(f)

    def test_refuses_records_without_the_label_or_with_another_digest(self):
        good = dict(label=C.label(), passmarks_sha256=C.PASSMARKS_SHA256, exploratory=True, status="finished")
        C.write_record(self.tmp / "good.json", good)
        self.assertEqual(C.arm_record(self.tmp / "good.json")["status"], "finished")
        for name, rec in (("nolabel", dict(good, label="something else")),
                          ("other", dict(good, passmarks_sha256="0" * 64)),
                          ("notexpl", dict(good, exploratory=False))):
            C.write_record(self.tmp / f"{name}.json", rec)
            with self.assertRaises(C.Refused):
                C.arm_record(self.tmp / f"{name}.json")
        (self.tmp / "good.json").write_text(json.dumps(dict(good, status="edited")))
        with self.assertRaises(C.Refused):      # no longer matches its sidecar
            C.arm_record(self.tmp / "good.json")

    def test_mark_arithmetic(self):
        import score
        s = score.summary([2.5, 3.0, -0.1, 2.2, 2.6])
        self.assertAlmostEqual(s["mean"], 2.04)
        self.assertEqual(s["positives"], 4)
        self.assertTrue(score.at_least(s, 2.0, 4))
        self.assertFalse(score.at_least(score.summary([2.5, 3.0, 0.0, 2.2, 2.6]), 2.0, 5))   # a tie is not positive
        self.assertFalse(score.at_least(score.summary([2.5, None, 3, 3, 3]), 2.0, 4))         # missing: not met
        self.assertTrue(score.at_least(score.summary([2.0] * 5), 2.0, 5))                     # inclusive


class Aggregation(unittest.TestCase):
    """screen.py, kills.py and score.py marks end to end on fabricated records in a temporary ARMS_SMOKE folder:
    the survivors, the kill rules across seeds, every mark and the fixed wording."""

    def setUp(self):
        import os
        self.tmp = Path(tempfile.mkdtemp(prefix="arms_agg_"))
        self.old = os.environ.get("ARMS_SMOKE")
        os.environ["ARMS_SMOKE"] = str(self.tmp)

    def tearDown(self):
        import os
        if self.old is None:
            os.environ.pop("ARMS_SMOKE", None)
        else:
            os.environ["ARMS_SMOKE"] = self.old
        shutil.rmtree(self.tmp, ignore_errors=True)

    def rec(self, arm, q, k, **fields):
        body = dict(label=C.label(), exploratory=True, passmarks_sha256=C.PASSMARKS_SHA256, arm=arm, quality=q,
                    seed=k, status="finished", stopped_by=None)
        body.update(fields)
        C.write_record(C.arm_dir(arm, q, k) / "record.json", body)

    @staticmethod
    def ev(points):
        return {law: dict(J=-100.0 + p, points=p, finite=True) for law, p in points.items()}

    def fabricate_arms(self, k, f0_points=None):
        rungs = [dict(rung=j, tries=40) for j in range(1, 11)]
        for q in C.QUALITIES:
            self.rec("A", q, k, kills=dict(K0=dict(applies=q in C.K0_QUALITIES, passes=True, recorder_share=0.05,
                                                   random_share=0.3),
                                           K1=dict(fires=False, released_share=0.2, D_mean=0.1, U_mean=0.2),
                                           K2_value=dict(d=-3.5 if (q == "random" and k in (1, 2, 3)) else 0.4)),
                     inputs=dict(s=0.5, unit=0.3, U_mean_over_s=0.7), history=[dict(arms=dict(A1={}))])
            self.rec("B", q, k, maps=dict(bca=[0.8, 1.0, 1.0, 1.0]), details=dict(rules=dict(bca=[dict(rungs=rungs)])),
                     tries=dict(rows=1000),
                     kills=dict(K1=dict(fires=False), K2_over_refusal=False, K3=dict(fires=False),
                                K4_clip_share_thruster1=0.1,
                                **({"K0_room": dict(room=2.0)} if q == "medium-expert" else {})))
            self.rec("C", q, k, maps=dict(bca=[0.85, 1.0, 1.0, 1.0]),
                     kills=dict(KB1=dict(fires=False, G_rho1=0.1, G_rho05=0.2, bar=0.3),
                                KB0_seed=dict(share_rho05=0.05, share_rho1=0.01, passes=True),
                                guard_void_own_map=dict(fires=False), K3=dict(fires=False)))
        for q in C.F0_QUALITIES:
            f0 = (f0_points or {}).get(q, 50.0 if q == "medium-expert" else 51.3)
            self.rec("F0", q, k, checks={}, chunks=[], evaluation=self.ev(dict(none=f0, mild=f0, strong=f0)))
            self.a2(q, k)

    def a2(self, q, k, evaluation=True):
        """A2's record (result F's script) and, unless evaluation is False, its evaluation written after it."""
        import os
        d = C.f1_out() / q / "A2_cmd_strong" / f"seed{k}"
        d.mkdir(parents=True, exist_ok=True)
        (d / "record.json").write_text(json.dumps(dict(arm="A2_cmd_strong", finite=True, seed=k, quality=q)))
        p = C.f1_out() / q / f"evaluate_seed{k}.json"
        if evaluation:
            p.write_text(json.dumps(dict(quality=q, seed=k, evaluations=dict(
                A2_cmd_strong=self.ev(dict(none=51.5, mild=51.5, strong=51.5))))))
            t = (d / "record.json").stat().st_mtime + 1
            os.utime(p, (t, t))
        elif p.exists():
            p.unlink()

    def d1_ok(self):
        for q in ("random", "medium-replay", "medium"):
            C.write_atomic(C.state() / f"d1_{q}.json", json.dumps(dict(all_identical=True, allows_band=True,
                                                                        sandbox_report_D1_status="pass")))

    def all_evals(self, f_seeds=C.SEEDS):
        """Evaluation records of every arm, quality and seed (the scorer reads only the allowed ones)."""
        import score
        for q in C.QUALITIES:
            for k in C.SEEDS:
                a = dict(none=50.0, A1=53.0 if q in C.P1_QUALITIES else 50.5, untargeted=50.0, margin0=51.0,
                         plain_1s=51.5, plain_2s=51.0, plain_4s=50.5, matched_share=52.5, shuffled_band=50.0,
                         A1_IW=53.0, A1c=50.0, **{"even_x1.33": 49.0}, host=50.0)
                self.eval_rec("A", q, k, {n: self.ev(dict(none=v, mild=v, strong=v)) for n, v in a.items()})
                me = q == "medium-expert"
                b = {"host": 60.0, "host+bca": 63.0 if me else 60.0, "host+range_limit": 60.5,
                     "host+shuffled": 61.0, "host+matched_global": 61.5, "host+knee_clip": 62.0,
                     "host+bca_unweighted": 60.0, "host+bca_exact": 60.0, "host+bca_shifted": 60.0,
                     **{f"host+{t}": (62.5 if (me and t == "wmedian") else 60.0) for t in score.PLAIN_B}}
                self.eval_rec("B", q, k, {n: self.ev(dict(none=v, mild=v, strong=v)) for n, v in b.items()})
                weak = q in C.WEAK
                c = {"rho1": 50.0, "rho0.5": 52.0 if weak else 50.0, "rho0.5+bca": 54.0 if weak else 50.0,
                     "rho0.5+range_limit": 52.0, "rho0.5+shuffled": 52.0, "rho0.5+matched_global": 52.5,
                     "rho0.5+knee_clip": 52.0, "rho1+bca_built_for_rho1": 50.0, "host": 50.0,
                     **{f"rho0.5+{t}": 53.0 for t in score.PLAIN_B}}
                self.eval_rec("C", q, k, {n: self.ev(dict(none=v, mild=v, strong=v)) for n, v in c.items()})
        for q in C.F0_QUALITIES:
            for k in f_seeds:
                self.eval_rec("F", q, k, {}, situations=dict(
                    A2={s: dict(points=51.5, finite=True) for s in C.SITUATIONS},
                    F0={s: dict(points=50.0, finite=True) for s in C.SITUATIONS}))

    def eval_rec(self, arm, q, k, evaluations, situations=None):
        body = dict(label=C.label(), exploratory=True, passmarks_sha256=C.PASSMARKS_SHA256, arm=f"eval_{arm}",
                    quality=q, seed=k, status="finished", evaluations=evaluations, situations=situations or {})
        import score
        C.write_record(score.eval_path(arm, q, k), body)

    def test_screen_kills_and_marks(self):
        import kills
        import score
        import screen
        self.d1_ok()
        self.fabricate_arms(0)
        self.assertEqual(screen.main(), 0)
        S = C.arm_record(C.docs() / "SCREEN_seed0.json")
        self.assertTrue(S["complete"])
        self.assertEqual(S["survivors"]["A"], list(C.QUALITIES))
        self.assertEqual(S["survivors"]["F0"], list(C.F0_QUALITIES))
        self.assertFalse(S["B_rules"]["K0"]["fires"])
        self.assertTrue(S["C_rules"]["KB0"]["holds"])
        self.assertAlmostEqual(S["F0"]["medium-expert"]["A2_minus_F0_strong"], 1.5)
        self.assertIn("No final return of arm A, B or C", (C.docs() / "SCREEN_seed0.md").read_text())
        for k in (1, 2, 3, 4):
            self.fabricate_arms(k)
        self.assertEqual(kills.main(), 0)
        K = C.arm_record(C.docs() / "KILLS.json")
        self.assertTrue(K["complete"])
        self.assertEqual(K["A"]["random"]["stopped_by"], "K2")          # -3.5 on three of five seeds
        self.assertTrue(K["A"]["medium"]["go_on"])
        self.assertTrue(K["C_rules"]["B2_B4_scored"])
        with self.assertRaises(C.Refused):                              # no evaluation yet: scoring refuses
            score.marks_job()
        self.all_evals()
        self.assertEqual(score.marks_job(), 0)
        R = C.arm_record(C.docs() / "results.json")
        A, B, Cm, F = R["A"], R["B"], R["C"], R["F0"]
        self.assertEqual(A["passing"], list(C.P1_QUALITIES))
        self.assertEqual(A["met"], dict(P1=True, P2=False, P3=True, P4=False))   # random stopped by K2: P2 missing
        self.assertFalse(A["GO"])
        self.assertTrue(any(x.startswith("Stopped by K2 on random, seeds 1, 2, 3: A1 - none") for x in A["wording"]))
        self.assertTrue(any("missing on random" in x for x in A["wording"]))
        self.assertTrue(any("Filtering helps" in x and "matched_share" in x for x in A["wording"]))
        self.assertEqual(B["met"], dict(G1=True, G2=True, G3=True, G4=False))
        self.assertTrue(B["GO"])
        self.assertEqual(B["G4"]["best_plain_test"], "wmedian")
        self.assertTrue(any("did no better than the wmedian" in x for x in B["wording"]))
        self.assertEqual(Cm["passing"], list(C.WEAK))
        self.assertEqual(Cm["met"], dict(B1=True, B2=True, B3=True, B4=True, B5=True))
        self.assertTrue(Cm["GO"])
        self.assertTrue(any(x.startswith("Exploratory: with BCA's certificate") for x in Cm["wording"]))
        self.assertIn("rho0.5+knee_clip-host", Cm["info"]["random"])              # information against the host
        self.assertIn("rho1+bca_built_for_rho1-host", Cm["info"]["random"])
        self.assertEqual(F["credit"], ["medium-expert"])
        self.assertTrue(any("partly from BCA's band (on medium-expert); on expert" in x for x in F["wording"]))
        sit = R["info"]["situations"]
        self.assertAlmostEqual(sit["F/expert"]["recorded"]["A2-F0"]["mean"], 1.5)
        self.assertIn("A/medium", sit)
        self.assertIn("EXPLORATORY", (C.docs() / "RESULTS.md").read_text())
        self.assertIn("four situations", (C.docs() / "RESULTS.md").read_text())

    def test_screen_stop_and_not_run_wording(self):
        """Finding 2: a rule that fires at the screen reaches RESULTS as "Stopped by <rule> on <quality, seed>:
        <reason>."; seeds not run after a quality-wide stop are not reported as stops."""
        import kills
        import score
        import screen
        self.d1_ok()
        self.fabricate_arms(0)
        self.rec("A", "random", 0, status="stopped", stopped_by="K1", reason="at the step-4,000 refresh A1 releases "
                 "more than 60% of the kept rows", kills=dict(K1=dict(fires=True)))
        self.assertEqual(screen.main(), 0)
        for k in (1, 2, 3, 4):
            self.fabricate_arms(k)
        self.rec("A", "medium", 3, status="stopped", stopped_by="K1", reason="K1 at seed 3")
        for k in (1, 2, 4):
            self.rec("A", "medium", k, status="stopped", stopped_by="K1", not_run=True,
                     reason="not run: K1 fired on medium seed 3 (A stops on the whole quality)")
        self.assertEqual(kills.main(), 0)
        K = C.arm_record(C.docs() / "KILLS.json")
        self.assertEqual(K["A"]["random"]["stopped_by"], "K1")
        self.assertEqual(K["A"]["random"]["seed"], 0)
        self.all_evals()
        self.assertEqual(score.marks_job(), 0)
        W = C.arm_record(C.docs() / "results.json")["A"]["wording"]
        self.assertIn("Stopped by K1 on random, seed 0: at the step-4,000 refresh A1 releases more than 60% of the kept "
                      "rows.", W)
        self.assertIn("Stopped by K1 on medium, seed 3: K1 at seed 3.", W)
        self.assertFalse(any("medium, seed 1" in x or "medium, seed 2" in x for x in W))

    def test_f0_not_decided(self):
        """Finding 1: an undecided quality gets only "not decided: <reason>"; no conclusion about it, and none at
        all when neither quality is decided."""
        import kills
        import score
        import screen
        self.d1_ok()
        self.fabricate_arms(0)
        self.rec("F0", "expert", 0, status="stopped", stopped_by="KF1", reason="flags differ", checks={})
        self.assertEqual(screen.main(), 0)
        self.assertEqual(C.arm_record(C.docs() / "SCREEN_seed0.json")["survivors"]["F0"], ["medium-expert"])
        for k in (1, 2, 3, 4):
            self.fabricate_arms(k)
        self.rec("F0", "medium-expert", 2, status="stopped", stopped_by="KF2", reason="a band input reaches F0")
        self.assertEqual(kills.main(), 0)
        self.all_evals(f_seeds=C.SEEDS)
        self.assertEqual(score.marks_job(), 0)
        F = C.arm_record(C.docs() / "results.json")["F0"]
        self.assertEqual(F["decided"], [])
        self.assertEqual(F["wording"], ["medium-expert: not decided: stopped by KF2 on seed 2: a band input reaches F0.",
                                        "expert: not decided: stopped by KF1 on seed 0: flags differ."])

    def test_f0_one_decided_and_missing_a2_evaluation(self):
        """Finding 1 and 8: with expert undecided, the medium-expert line has no clause about expert; a missing A2
        evaluation keeps the kill checks incomplete, so nothing is scored on it."""
        import kills
        import score
        import screen
        self.d1_ok()
        self.fabricate_arms(0)
        self.assertEqual(screen.main(), 0)
        for k in (1, 2, 3, 4):
            self.fabricate_arms(k)
        self.a2("expert", 3, evaluation=False)
        self.assertEqual(kills.main(), 0)
        K = C.arm_record(C.docs() / "KILLS.json")
        self.assertFalse(K["complete"])
        self.assertTrue(any("A2's evaluation of expert seed 3" in m for m in K["missing"]))
        with self.assertRaises(C.Refused):
            score.marks_job()
        self.rec("F0", "expert", 3, status="stopped", stopped_by="input", reason="A2's actor of expert seed 3 is not "
                 "finite")
        self.assertEqual(kills.main(), 0)
        self.assertTrue(C.arm_record(C.docs() / "KILLS.json")["complete"])
        self.all_evals()
        self.assertEqual(score.marks_job(), 0)
        F = C.arm_record(C.docs() / "results.json")["F0"]
        self.assertEqual(F["decided"], ["medium-expert"])
        self.assertIn("Result F's gain comes partly from BCA's band (on medium-expert).", F["wording"])
        self.assertFalse(any("on expert it comes" in x for x in F["wording"]))
        self.assertIn("expert: not decided: seed 3 not run: A2's actor of expert seed 3 is not finite.", F["wording"])

    def test_k3_counts_screen_records_and_returns(self):
        """Finding 3: a seed-0 K3 on one quality and a non-finite return on another stop B (two such seeds); C's two
        K3 seeds stop C."""
        import kills
        import score
        import screen
        self.d1_ok()
        self.fabricate_arms(0)
        self.rec("B", "random", 0, status="stopped", stopped_by="K3", reason="non-finite values: non-finite gain",
                 kills=dict(K3=dict(fires=True, reasons=["non-finite gain"])))
        self.rec("C", "medium-expert", 0, kills=dict(KB1=dict(fires=False), KB0_seed=dict(share_rho05=0.05,
                 share_rho1=0.01, passes=True), guard_void_own_map=dict(fires=False),
                 K3=dict(fires=True, reasons=["non-finite gain"])), maps=dict(bca=[0.85, 1, 1, 1]))
        self.assertEqual(screen.main(), 0)
        S = C.arm_record(C.docs() / "SCREEN_seed0.json")
        self.assertNotIn("random", S["survivors"]["B"])
        self.assertNotIn("medium-expert", S["survivors"]["C"])
        for k in (1, 2, 3, 4):
            self.fabricate_arms(k)
        self.rec("C", "medium", 4, status="stopped", stopped_by="K3", reason="non-finite values: x",
                 kills=dict(K3=dict(fires=True, reasons=["x"])))
        self.assertEqual(kills.main(), 0)
        K = C.arm_record(C.docs() / "KILLS.json")
        self.assertIn("random seed 0", K["B"]["k3_seeds"])
        self.assertEqual(K["C_rules"]["stopped_by"], ["K3"])
        self.all_evals()
        bad = {law: dict(J=float("nan"), points=float("nan"), finite=False) for law in ("none", "mild", "strong")}
        r = C.arm_record(score.eval_path("B", "medium", 2))
        r["evaluations"]["host+bca"] = bad
        self.eval_rec("B", "medium", 2, r["evaluations"])
        self.assertEqual(score.marks_job(), 0)
        R = C.arm_record(C.docs() / "results.json")
        self.assertTrue(R["B"]["K3"]["stops_B"])
        self.assertEqual(R["B"]["K3"]["seeds"], ["medium seed 2", "random seed 0"])
        self.assertFalse(R["B"]["GO"])
        self.assertTrue(any(x.startswith("Stopped by K3 on medium, seed 2; random, seed 0") for x in R["B"]["wording"]))
        self.assertIn(score.whole_arm_stopped("B"), R["B"]["wording"])
        self.assertFalse(R["C"]["GO"])
        self.assertIn(score.whole_arm_stopped("C"), R["C"]["wording"])

    def test_guard_void_reads_b_maps_of_seeds_1_4_after_a_screen_stop(self):
        """Finding 4: B stopped at the screen by K0 (no room) while C survives: B's maps of seeds 1-4 are still built
        for every quality (rt.jobs_continue) and their K1 voids C's guard."""
        import kills
        import rt
        import score
        import screen
        self.d1_ok()
        self.fabricate_arms(0)
        rungs = [dict(rung=j, tries=40) for j in range(1, 11)]
        self.rec("B", "medium-expert", 0, maps=dict(bca=[0.8, 1.0, 1.0, 1.0]),
                 details=dict(rules=dict(bca=[dict(rungs=rungs)])), tries=dict(rows=1000),
                 kills=dict(K1=dict(fires=False), K2_over_refusal=False, K3=dict(fires=False),
                            K4_clip_share_thruster1=0.1, K0_room=dict(room=0.2)))
        self.assertEqual(screen.main(), 0)
        S = C.arm_record(C.docs() / "SCREEN_seed0.json")
        self.assertEqual(S["B_stopped_by"], ["K0"])
        old = rt.ARMS
        try:
            rt.ARMS = C.docs()
            jobs = [j for j, _ in rt.jobs_continue()]
        finally:
            rt.ARMS = old
        self.assertTrue(all(f"B_{q}_s{k}" in jobs for q in C.QUALITIES for k in (1, 2, 3, 4)))
        self.assertFalse(any(j.startswith("eval_B_") for j in jobs))
        self.assertEqual(jobs[-1], "marks")
        for k in (1, 2, 3, 4):
            self.fabricate_arms(k)
        self.rec("B", "medium", 3, maps=dict(bca=[0.85, 1.0, 1.0, 1.0]), tries=dict(rows=1000),
                 kills=dict(K1=dict(fires=True), K2_over_refusal=False, K3=dict(fires=False),
                            K4_clip_share_thruster1=0.1))
        self.assertEqual(kills.main(), 0)
        K = C.arm_record(C.docs() / "KILLS.json")
        self.assertEqual(K["B"]["stopped_by"], ["K0", "K1"])
        self.assertFalse(K["B"]["go_on"])
        self.assertTrue(K["C_rules"]["guard_void"]["by_B"])
        self.assertFalse(K["C_rules"]["B2_B4_scored"])
        self.all_evals()
        self.assertEqual(score.marks_job(), 0)
        R = C.arm_record(C.docs() / "results.json")
        self.assertTrue(any(x.startswith("Stopped by K0 on medium-expert, seed 0") for x in R["B"]["wording"]))
        self.assertTrue(any(x.startswith("Halving the pull, guarded by BCA's ceiling, met B1") for x in
                            R["C"]["wording"]))

    def test_summaries_done_only_when_complete_and_current(self):
        """Finding 7 and 8: the screen and the kill checks count as done only when complete and built on the current
        records; scoring only for the current kill checks."""
        import kills
        import rt
        import score
        import screen
        self.d1_ok()
        self.fabricate_arms(0)
        (C.arm_dir("C", "medium", 0) / "record.json").unlink()
        self.assertEqual(screen.main(), 0)
        old = rt.ARMS
        try:
            rt.ARMS = C.docs()
            self.assertFalse(rt.done("screen"))                          # written, but not complete
            self.fabricate_arms(0)
            self.assertEqual(screen.main(), 0)
            self.assertTrue(rt.done("screen"))
            self.rec("C", "medium", 0, maps=dict(bca=[0.9, 1, 1, 1]), kills=dict(KB1=dict(fires=False),
                     KB0_seed=dict(share_rho05=0.05, share_rho1=0.01, passes=True),
                     guard_void_own_map=dict(fires=False), K3=dict(fires=False)))
            self.assertFalse(rt.done("screen"))                          # an input changed after it was built
            self.assertEqual(screen.main(), 0)
            for k in (1, 2, 3, 4):
                self.fabricate_arms(k)
            (C.arm_dir("B", "expert", 2) / "record.json").unlink()
            self.assertEqual(kills.main(), 0)
            self.assertFalse(rt.done("kills"))
            self.fabricate_arms(2)
            self.assertEqual(kills.main(), 0)
            self.assertTrue(rt.done("kills"))
            self.all_evals()
            self.assertEqual(score.marks_job(), 0)
            self.assertTrue(rt.done("marks"))
            self.rec("C", "medium", 1, maps=dict(bca=[0.95, 1, 1, 1]), kills=dict(KB1=dict(fires=False),
                     KB0_seed=dict(share_rho05=0.05, share_rho1=0.01, passes=True),
                     guard_void_own_map=dict(fires=False), K3=dict(fires=False)))
            self.assertFalse(rt.done("kills"))                           # a record it read changed
            self.assertFalse(rt.done("marks"))                           # and so did one the scoring read
            self.assertEqual(kills.main(), 0)                            # new kill checks: old scoring is stale
            self.assertTrue(rt.done("kills"))
            self.assertFalse(rt.done("marks"))
        finally:
            rt.ARMS = old


class Runner(unittest.TestCase):
    """The scheduler of run_arms.sh (its functions, unchanged, with rt and run_job stubbed) and rt.py's label."""

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="arms_sched_"))

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_resume_redoes_an_incomplete_screen(self):
        """Finding 7: input job b fails, the settling screen runs and is incomplete; on resume b is redone and so is
        the screen (it is never 'finished' while incomplete). Scoring is not settling: blocked by a failed input."""
        import subprocess
        src = (C.ARMS_DIR / "run_arms.sh").read_text()
        funcs = src[src.index("# ---- scheduler (begin)"):src.index("# ---- scheduler (end)")]
        (self.tmp / "sched_funcs.sh").write_text(funcs)
        (self.tmp / "done").mkdir()
        harness = r"""set -euo pipefail
H=$1; STATE=$H; LOGS=$H; CPUS="1 2"; DONE=$H/done
say() { echo "$*"; }
rt() {
  case $1 in
    jobs) printf 'a	
b	
c	b
screen	a,b,c
marks	screen,b
' ;;
    done) case $2 in screen) [ -f "$DONE/screen" ] && grep -qx complete "$DONE/screen" ;; *) [ -f "$DONE/$2" ] ;; esac ;;
  esac
}
refresh_label() { :; }
run_job() {
  local j=$1
  case $j in
    b) [ -f "$H/b_ok" ] || return 1 ;;
    screen) if [ -f "$DONE/a" ] && [ -f "$DONE/b" ] && [ -f "$DONE/c" ]; then echo complete > "$DONE/screen";
            else echo incomplete > "$DONE/screen"; fi; rt done screen; return ;;
  esac
  touch "$DONE/$j"
}
source "$H/sched_funcs.sh"
load_jobs start
schedule
echo "SCREEN $(cat "$DONE/screen")"
for j in a b c screen marks; do echo "STATE $j ${ST[$j]}"; done
"""
        (self.tmp / "h.sh").write_text(harness)
        r1 = subprocess.run(["bash", str(self.tmp / "h.sh"), str(self.tmp)], capture_output=True, text=True)
        self.assertIn("SCREEN incomplete", r1.stdout, r1.stdout + r1.stderr)
        self.assertIn("STATE marks blocked", r1.stdout)
        (self.tmp / "b_ok").touch()
        r2 = subprocess.run(["bash", str(self.tmp / "h.sh"), str(self.tmp)], capture_output=True, text=True)
        self.assertNotIn("skip (finished): screen", r2.stdout)
        self.assertIn("SCREEN complete", r2.stdout, r2.stdout + r2.stderr)
        self.assertIn("STATE marks done", r2.stdout)

    def test_label_lists_new_files_under_runs_only(self):
        import rt
        saved = (rt.REPO, rt.STATE, rt.V5, rt.EX)
        try:
            rt.REPO, rt.STATE = self.tmp / "repo", self.tmp / "state"
            rt.V5, rt.EX = rt.REPO / "runs" / "shift_iw_v5", rt.REPO / "runs" / "shift_iw_v5_explore"
            for d in (rt.V5, rt.EX, rt.STATE):
                d.mkdir(parents=True)
            old = rt.V5 / "old.json"
            old.write_text("{}")
            rt.write_atomic(rt.STATE / "baseline_sandbox.json", json.dumps(dict(files={str(old): rt.sha(old)})))
            new = rt.EX / "arms_abc_f0" / "A" / "record.json"
            new.parent.mkdir(parents=True)
            new.write_text('{"a": 1}')
            self.assertEqual(rt.label(), 0)
            text = (rt.V5 / rt.LABEL_FILE).read_text()
            self.assertIn(str(new), text)
            self.assertNotIn(str(old), text)
            self.assertNotIn(rt.LABEL_FILE + "  sha256", text)
            self.assertIn(str(new), (rt.EX / rt.LABEL_FILE).read_text())
            new.write_text('{"a": 22}')                                      # changed: rehashed
            self.assertEqual(rt.label(), 0)
            self.assertIn(rt.sha(new), (rt.V5 / rt.LABEL_FILE).read_text())
        finally:
            rt.REPO, rt.STATE, rt.V5, rt.EX = saved


class NonFiniteB(unittest.TestCase):
    """Finding 3: non-finite gains or weights stop the seed (K3) instead of crashing wbcp.calibrate; a rung whose
    tries all weigh 0 is refused."""

    class FakeTries:
        def __init__(self, g_bad=False, w_bad=False):
            rng = np.random.default_rng(3)
            self.n = 400
            self.a32 = rng.uniform(-1, 1, (self.n, 4)).astype(np.float32)
            self.w = rng.uniform(0.5, 2.0, self.n)
            if w_bad:
                self.w[5] = np.inf
            self.w_exact = np.full(self.n, np.nan)          # information rule only: skipped, never a crash
            self.test_mass, self.test_mass_exact, self.seed = 1.2, 1.0, 0
            self.g_bad = g_bad

        def gains(self, grid):
            edges = arm_b.EDGES if grid == "standard" else arm_b.SHIFTED
            out = []
            for kk in range(4):
                j = arm_b.rung_index(np.abs(self.a32[:, kk]), edges)
                cand = np.flatnonzero(j > 0)
                g = np.linspace(-1, 2, len(cand))
                if self.g_bad and kk == 2:
                    g[3] = np.nan
                out.append((cand, j[cand], g))
            return out

    def test_nonfinite_gain_or_weight_raises_k3(self):
        pi = np.sign(self.FakeTries().a32).astype(np.float32)
        with self.assertRaises(arm_b.NonFinite) as e:
            arm_b.maps_for(self.FakeTries(g_bad=True), pi, 0)
        self.assertIn("thruster 3", str(e.exception))
        with self.assertRaises(arm_b.NonFinite):
            arm_b.maps_for(self.FakeTries(w_bad=True), pi, 0)
        maps, det = arm_b.maps_for(self.FakeTries(), pi, 0)
        self.assertNotIn("bca_exact", maps)
        self.assertIn("bca_exact", det["skipped"])

    def test_zero_weight_rung_refused(self):
        ok, stat = arm_b.bca_decide(np.ones(20), np.zeros(20), 1.0, np.random.default_rng(0))
        self.assertEqual((ok, stat), (False, None))


if __name__ == "__main__":
    unittest.main()

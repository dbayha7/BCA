"""CPU tests of the learners (tiny update counts on a small log; every number is a test fixture, not a finding).

Checks: the plain TD3+BC run equals td3_bc.make_train_step bit for bit, even caution x1 equals plain, and a second
random stream keeps the initial parameters; the actor-only step equals td3_bc_update's actor step bit for bit; a
fixed-controller critic keeps its controller fixed, critics of one seed share their initial parameters and stream, the
placebo changes only the initial parameters; behaviour cloning is deterministic and lowers its loss; actor-only arms
share one batch stream (identical arms give identical actors), leave the frozen critic unchanged and refresh the exact
over-preference on schedule; the caution tables (harm-placed and cap-2 dose forms, amount-matched, shuffled, reversed)
give known answers and placements only permute the dose multiset; strength is 0 for the same actor; blind matching
computes no return; the exact Q of a learned actor equals truth.py's (float64) and its float32 path is close; the
over-preference parts add up; evaluation reports every situation; and BCA: its update traces identically to
td3_bc_bca.update, every placement traces, and every BCA path refuses without a valid approval file before doing
anything (nothing BCA-related is ever executed here).
JAX_PLATFORMS=cpu python -m unittest experiments.shift.test_learners
"""

import json
import sys
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest import mock

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402

import algorithms.td3_bc as BASE  # noqa: E402
import algorithms.td3_bc_bca as P  # noqa: E402
from experiments.shift import dataset as DS  # noqa: E402
from experiments.shift import learners as L  # noqa: E402
from experiments.shift import recorder as RC  # noqa: E402
from experiments.shift import shift as SH  # noqa: E402
from experiments.shift import thruster_chain as TC  # noqa: E402
from experiments.shift import truth as TR  # noqa: E402

SET = replace(L.LearnSettings(), updates=200, chunk=100, clone_steps=100, actor_only_steps=100, actor_only_chunk=50,
              oracle_refresh=50, oracle_horizon=50, strength_states=200, row_chunk=1024, band_fit_steps=20,
              bca_warmup=100, bca_refresh=50)
CACHE = {}


def data():
    if "data" not in CACHE:
        lg = DS.generate(900, "stall", sizes=(40, 10, 10))
        gap, _ = SH.define_gap(lg)
        rg, full = SH.make_removal(lg, "gap", gap), SH.make_removal(lg, "none")
        norm = L.normalization(lg, rg)
        CACHE["data"] = dict(log=lg, gap=gap, pg=L.prepare(lg, rg, norm, SET), pf=L.prepare(lg, full, norm, SET))
    return CACHE["data"]


def host():
    if "host" not in CACHE:
        CACHE["host"] = L.cotrain(data()["pg"], SET, 0)
    return CACHE["host"]


def cloned():
    if "clone" not in CACHE:
        CACHE["clone"] = L.clone(data()["pg"], SET, 0, chunk=100)
    return CACHE["clone"]


def leaves_equal(a, b):
    la, lb = jax.tree_util.tree_leaves(a), jax.tree_util.tree_leaves(b)
    return len(la) == len(lb) and all(np.array_equal(np.asarray(x), np.asarray(y)) for x, y in zip(la, lb))


class PlainTD3BCTest(unittest.TestCase):
    def test_plain_equals_make_train_step_and_even_x1(self):
        pg = data()["pg"]
        rec, state, snaps = host()
        args = L.make_args(SET, 0)
        rng, s0, (actor, critic) = BASE.initialize(args, TC.DX, TC.DA, 1.0)
        step = BASE.make_train_step(args, actor.apply, critic.apply, pg.train, 1.0)
        direct = jax.jit(lambda r: jax.lax.scan(step, r, None, length=SET.updates))((rng, s0, jnp.int32(0)))[0][1]
        self.assertTrue(leaves_equal(state, direct))
        self.assertEqual(sorted(snaps), [100, 200])
        self.assertFalse(rec["failed"])
        _, even1, _ = L.cotrain(pg, SET, 0, table=np.ones(pg.n))
        d = max(float(np.abs(np.asarray(x) - np.asarray(y)).max()) for x, y in
                zip(jax.tree_util.tree_leaves(even1), jax.tree_util.tree_leaves(state)))
        self.assertLess(d, 1e-6)

    def test_second_stream_keeps_initial_parameters(self):
        pg = data()["pg"]
        rec0, s0, _ = host()
        rec1, s1, _ = L.cotrain(pg, SET, 0, stream=1, updates=100)
        self.assertEqual(rec0["initial_sha256"], rec1["initial_sha256"])
        _, s0b, _ = L.cotrain(pg, SET, 0, updates=100)
        self.assertFalse(leaves_equal(s0b.critic.params, s1.critic.params))
        with self.assertRaises(ValueError):
            L.cotrain(pg, SET, 0, table=np.ones(3), updates=10)

    def test_actor_step_equals_td3_bc_update(self):
        pg = data()["pg"]
        args = L.make_args(SET, 0)
        _, s0, (actor, critic) = BASE.initialize(args, TC.DX, TC.DA, 1.0)
        batch = jax.tree_util.tree_map(lambda x: x[:256], pg.train)
        theirs = jax.jit(lambda s, b, w: BASE.td3_bc_update(args, actor.apply, critic.apply, s, b, 2,
                                                            jax.random.PRNGKey(4), 1.0, bc_multiplier=w)[0])
        mine = jax.jit(lambda a, c, b, w: L.actor_step(args, a, c, b, w)[0])
        new = jax.jit(lambda s, b: BASE.td3_bc_update(args, actor.apply, critic.apply, s, b, 2, jax.random.PRNGKey(4),
                                                      1.0)[0])(s0, batch)
        ours = jax.jit(lambda a, c, b: L.actor_step(args, a, c, b)[0])(s0.actor, new.critic.params, batch)
        self.assertTrue(leaves_equal(ours.params, new.actor.params))
        w = jnp.full(256, 1.7, jnp.float32)
        new_w = theirs(s0, batch, w)
        self.assertTrue(leaves_equal(mine(s0.actor, new_w.critic.params, batch, w).params, new_w.actor.params))


class BatchStreamTest(unittest.TestCase):
    """The batch-row stream hash a co-trained run records is the stream td3_bc.make_train_step (plain arms) and
    bca_train_step (BCA arms) really draw, so the scorer can check that the arms of a seed share it."""

    def _rows_drawn(self, step, carry, steps):
        return np.asarray(jax.jit(lambda c: jax.lax.scan(step, c, None, length=steps))(carry)[1])

    def test_recorded_stream_is_the_drawn_stream(self):
        pg = data()["pg"]
        args = L.make_args(SET, 0)
        rng, s0, (actor, critic) = BASE.initialize(args, TC.DX, TC.DA, 1.0)
        labelled = pg.train._replace(obs=jnp.tile(jnp.arange(pg.n, dtype=jnp.float32)[:, None], (1, TC.DX)))
        k = 5
        with mock.patch.object(BASE, "td3_bc_update", lambda a, af, cf, st, b, it, r, m, **kw: (st, b.obs[:, 0])):
            plain = self._rows_drawn(BASE.make_train_step(args, actor.apply, critic.apply, labelled, 1.0),
                                     (rng, s0, jnp.int32(0)), k)
        np.testing.assert_array_equal(plain.astype(np.int64), L.batch_stream(rng, pg.n, args.batch_size, k))
        # the BCA step's batch rows (its update replaced by a stand-in that returns the rows: nothing BCA runs)
        config = L.bca_config()
        rng_b, bst, models = L.bca_initial_state(args, config)
        fake = lambda a, c, m, st, b, it, r, ma, pl, sg: (st, dict(inputs_valid=jnp.asarray(True), rows=b.obs[:, 0]))
        with mock.patch.object(L, "bca_update", fake):
            out = jax.jit(lambda c: jax.lax.scan(L.bca_train_step(args, config, models, labelled, 1.0), c, None,
                                                 length=k))((rng_b, bst, jnp.int32(0)))[1]["rows"]
        np.testing.assert_array_equal(np.asarray(out).astype(np.int64), plain.astype(np.int64))
        rec = host()[0]
        self.assertEqual(rec["batch_stream_sha256"], L.batch_stream_sha256(rng, pg.n, args.batch_size))
        rec1, _, _ = L.cotrain(pg, SET, 0, stream=1, updates=10)
        self.assertNotEqual(rec1["batch_stream_sha256"], rec["batch_stream_sha256"])
        self.assertEqual(rec1["data"]["sha256"], rec["data"]["sha256"])


class FixedCriticTest(unittest.TestCase):
    def test_fixed_controller_critics(self):
        d = data()
        rec_g, st_g = L.fixed_critic(d["pg"], "stall_aware", SET, 0, updates=100)
        rec_f, st_f = L.fixed_critic(d["pf"], "stall_aware", SET, 0, updates=100)
        rec_p, st_p = L.fixed_critic(d["pf"], "stall_aware", SET, 0, placebo=True, updates=100)
        rec_r, st_r = L.fixed_critic(d["pg"], "rising_thrust", SET, 0, updates=100)
        fixed = L.FixedController("stall_aware", d["pg"].physics, d["pg"].obs_mean, d["pg"].obs_std)
        self.assertTrue(leaves_equal(st_g.actor.params, fixed.params()))
        self.assertTrue(leaves_equal(st_g.actor_target.params, fixed.params()))
        self.assertEqual(rec_g["initial_sha256"], rec_f["initial_sha256"])
        self.assertEqual(rec_g["initial_sha256"], rec_r["initial_sha256"])
        self.assertNotEqual(rec_p["initial_sha256"], rec_f["initial_sha256"])
        self.assertNotEqual(rec_g["critic_sha256"], rec_f["critic_sha256"])
        x = d["pg"].states[:20]
        cmd = np.asarray(fixed.apply(fixed.params(), jnp.asarray(d["pg"].normalize(x))))
        np.testing.assert_allclose(cmd, RC.stall_aware(TC.STALL)(x), atol=1e-5)
        self.assertLessEqual(np.abs(cmd).max(), 0.8 + 1e-7)
        with self.assertRaises(ValueError):
            L.FixedController("recording", TC.STALL, d["pg"].obs_mean, d["pg"].obs_std)


class CloneAndActorOnlyTest(unittest.TestCase):
    def test_clone_is_deterministic_and_learns(self):
        params, rec = cloned()
        again, rec2 = L.clone(data()["pg"], SET, 0, chunk=100)
        self.assertTrue(leaves_equal(params, again))
        self.assertEqual(rec["rows_sha256"], rec2["rows_sha256"])
        params2, rec3 = L.clone(data()["pg"], SET, 0, steps=200, chunk=100)
        self.assertLess(rec3["losses"][-1], rec3["losses"][0])
        args = L.make_args(SET, 0)
        _, s0, _ = BASE.initialize(args, TC.DX, TC.DA, 1.0)
        self.assertTrue(leaves_equal(L.initial_actor(SET, 0), s0.actor.params))

    def test_arms_share_one_stream_and_critic_stays_frozen(self):
        pg = data()["pg"]
        critic = host()[1].critic.params
        before = jax.tree_util.tree_map(lambda x: np.array(x), critic)
        ao = L.ActorOnly(pg, cloned()[0], SET, 0)
        stacked, info = ao.run([L.Arm("a"), L.Arm("b"), L.Arm("c", "even", 2.0)], [critic])
        self.assertTrue(leaves_equal(ao.arm_params(stacked, 0), ao.arm_params(stacked, 1)))
        self.assertFalse(leaves_equal(ao.arm_params(stacked, 0), ao.arm_params(stacked, 2)))
        self.assertTrue(leaves_equal(before, critic))
        self.assertEqual(info["oracle_calls"], 0)
        ao2 = L.ActorOnly(pg, cloned()[0], SET, 0)
        self.assertEqual(ao.rows_sha256, ao2.rows_sha256)
        with self.assertRaises(ValueError):
            ao.run([L.Arm("a", "even", 2.0)], [critic], weighted=False)
        with self.assertRaises(ValueError):
            ao.run([L.Arm("x", "amount", source="none")], [critic])

    def test_room_arms_tables_and_refresh(self):
        pg = data()["pg"]
        critic = host()[1].critic.params
        ao = L.ActorOnly(pg, cloned()[0], SET, 0)
        arms = L.room_arms()
        with mock.patch.object(TR, "rollout", side_effect=AssertionError("no return during training")):
            stacked, info = ao.run(arms, [critic])
        self.assertEqual(info["oracle_calls"], 2 * (SET.actor_only_steps // SET.oracle_refresh))
        self.assertEqual([h["step"] for h in info["history"]], [0, 50])
        last = info["history"][-1]
        self.assertGreaterEqual(last["harm_placed"]["min"], 1.0)
        self.assertLess(last["harm_placed"]["max"], 1.5)
        self.assertLess(last["cap2"]["max"], 2.0)
        self.assertAlmostEqual(last["amount_matched"]["mean"], last["harm_placed"]["mean"], places=5)
        self.assertAlmostEqual(last["amount_matched"]["sd"], 0.0, places=6)
        for k in ("shuffled_0", "shuffled_1", "reversed"):
            self.assertAlmostEqual(last[k]["mean"], last["harm_placed"]["mean"], places=5)
        none = ao.arm_params(stacked, 0)
        self.assertEqual(L.strength(none, none, pg.eval_obs), 0.0)
        self.assertGreater(L.strength(ao.arm_params(stacked, 5), none, pg.eval_obs), 0.0)

    def test_tables_known_answers(self):
        np.testing.assert_allclose(L.harm_doses(np.zeros(4)), np.ones(4))
        D = np.array([0.0, 1.0])
        np.testing.assert_allclose(L.harm_doses(D), [1.0, 1.0 + 0.5 * 6 / 7])
        np.testing.assert_allclose(L.harm_doses(D, "cap2"), [1.0, 1.0 + 6 / 7])
        doses = np.array([1.0, 1.1, 1.2, 1.3])
        np.testing.assert_allclose(L.reverse_ranks(doses, np.array([4.0, 3.0, 2.0, 1.0]), np.arange(4)),
                                   [1.0, 1.1, 1.2, 1.3])
        np.testing.assert_allclose(L.reverse_ranks(doses, np.array([1.0, 2.0, 3.0, 4.0]), np.arange(4)),
                                   [1.3, 1.2, 1.1, 1.0])
        pg = data()["pg"]
        ao = L.ActorOnly(pg, cloned()[0], SET, 0)
        Dr = np.random.default_rng(0).exponential(size=pg.n)
        arms = [L.Arm("h", "harm"), L.Arm("am", "amount", source="h"), L.Arm("s", "shuffled", source="h", index=1),
                L.Arm("r", "reversed", source="h"), L.Arm("e", "even", 1.25)]
        t = ao.tables(arms, {"h": Dr})
        np.testing.assert_allclose(np.sort(t[2]), np.sort(t[0]), rtol=1e-6)
        np.testing.assert_allclose(np.sort(t[3]), np.sort(t[0]), rtol=1e-6)
        self.assertLess(np.corrcoef(t[3], Dr)[0, 1], 0)
        self.assertGreater(np.corrcoef(t[0], Dr)[0, 1], 0)
        np.testing.assert_allclose(t[1], np.full(pg.n, t[0].mean()), rtol=1e-6)
        np.testing.assert_allclose(t[4], 1.25)

    def test_placements_permute_only(self):
        v = jnp.asarray(np.random.default_rng(1).uniform(1, 1.5, 64), jnp.float32)
        sig = jnp.asarray(np.random.default_rng(2).standard_normal(64), jnp.float32)
        np.testing.assert_array_equal(np.asarray(L.place(v, "own", None)), np.asarray(v))
        np.testing.assert_allclose(np.sort(np.asarray(L.place(v, "shuffled", jax.random.PRNGKey(0)))),
                                   np.sort(np.asarray(v)))
        nm = np.asarray(L.place(v, "noise_map", None, sig))
        np.testing.assert_allclose(np.sort(nm), np.sort(np.asarray(v)))
        self.assertEqual(int(np.argmax(nm)), int(np.argmax(np.asarray(sig))))
        self.assertAlmostEqual(float(np.mean(L.place(v, "flattened", None))), float(np.mean(v)), places=5)
        with self.assertRaises(ValueError):
            L.place(v, "noise_map", None)

    def test_blind_matching_computes_no_return(self):
        pg = data()["pg"]
        critic = host()[1].critic.params
        ao = L.ActorOnly(pg, cloned()[0], SET, 0)
        arms = [L.Arm("none")] + [L.Arm(f"e{m}", "even", m) for m in (1.25, 2.0)]
        stacked, _ = ao.run(arms, [critic])
        none = ao.arm_params(stacked, 0)
        ladder = [(m, L.strength(ao.arm_params(stacked, i + 1), none, pg.eval_obs)) for i, m in enumerate((1.25, 2.0))]
        target = 0.5 * (ladder[0][1] + ladder[1][1])
        with mock.patch.object(TR, "rollout", side_effect=AssertionError("no return before the lock")), \
                mock.patch.object(L, "evaluate_policy", side_effect=AssertionError("no return before the lock")):
            rec, evaluated = L.match_even(ao, [critic], none, pg.eval_obs, ladder, target, SET)
        self.assertTrue(rec["reachable"])
        self.assertTrue(1.25 <= rec["multiplier"] <= 2.0)
        self.assertEqual(rec["glq_match_sha256"], L.M.source_sha256())


class TruthOfLearnedActorsTest(unittest.TestCase):
    def test_exact_q_matches_truth(self):
        pg = data()["pg"]
        params = cloned()[0]
        pol = L.mlp_policy(params, pg)
        x, a = pg.states[:30], pg.actions[:30].astype(np.float64)
        q = L.exact_q(params, pg, x, a, horizon=300, chunk=16)
        np.testing.assert_allclose(q, TR.q_value_jax(pol, x, a, horizon=300), rtol=0, atol=1e-12)
        np.testing.assert_allclose(q, TR.q_value(pol, x, a, horizon=300), rtol=0, atol=1e-10)
        q32 = L.exact_q(params, pg, x, a, horizon=300, x64=False, chunk=16)
        np.testing.assert_allclose(q32, q, rtol=1e-4)

    def test_over_preference_parts(self):
        pg = data()["pg"]
        params, critic = cloned()[0], host()[1].critic.params
        rows = np.arange(0, pg.n, 97)
        op = L.over_preference_rows(params, critic, pg, rows=rows, horizon=100, x64=True, chunk=64)
        d = (op["q1_policy"] - op["q1_logged"]) - (op["true_policy"] - op["true_logged"])
        np.testing.assert_allclose(op["D"], np.maximum(d, 0.0))
        self.assertTrue(np.all(op["D"] >= 0))
        np.testing.assert_allclose(op["policy"], L.actor_rows(params, np.asarray(pg.train.obs)[rows]))

    def test_evaluation_and_loss_rate(self):
        pg = data()["pg"]
        an = TR.anchors(TC.STALL, n=20)
        ev = L.evaluate_actor(cloned()[0], pg, an, n=20)
        self.assertEqual(set(ev), set(TC.SITUATIONS))
        for s, v in ev.items():
            self.assertTrue(v["finite"])
            self.assertAlmostEqual(v["points"], float(TR.points(v["J"], an[s])), places=6)
            self.assertGreaterEqual(v["unrecorded_throttle_share"], 0.0)
        attr = dict(terms=np.array([[-2.0, -1.0], [0.0, -1.0]]), commands=np.array([[[0.9, 0, 0, 0], [0.1, 0, 0, 0]],
                                                                                    [[0.1, 0, 0, 0], [0.1, 0, 0, 0]]]),
                    max_sum_error=0.0)
        lr = L.loss_rate_ratio(attr, TC.STALL)
        self.assertEqual(lr["loss_rate_past_80"], 2.0)
        self.assertAlmostEqual(lr["loss_rate_other"], 2.0 / 3.0)


class BCASwitchedOffTest(unittest.TestCase):
    """Nothing BCA-related runs here: the program is traced, and every path refuses without approval."""

    def test_bca_update_traces_identically(self):
        pg = data()["pg"]
        args = L.make_args(SET, 0)
        config = L.bca_config()
        rng, state, models = L.bca_initial_state(args, config)
        batch = jax.tree_util.tree_map(lambda x: x[:256], pg.train)
        j1 = jax.make_jaxpr(lambda s, b, r, it: P.update(args, config, models, s, b, it, r, 1.0))(
            state, batch, rng, jnp.int32(2))
        j2 = jax.make_jaxpr(lambda s, b, r, it: L.bca_update(args, config, models, s, b, it, r, 1.0))(
            state, batch, rng, jnp.int32(2))
        self.assertEqual(str(j1), str(j2))
        s1 = P.make_train_step(args, config, models, pg.train, 1.0)
        s2 = L.bca_train_step(args, config, models, pg.train, 1.0)
        c0 = (rng, state, jnp.int32(0))
        k1 = jax.make_jaxpr(lambda c: jax.lax.scan(s1, c, None, length=2))(c0)
        k2 = jax.make_jaxpr(lambda c: jax.lax.scan(s2, c, None, length=2))(c0)
        self.assertEqual(str(k1), str(k2))
        self.assertEqual(config.blend, 0.5)
        self.assertEqual(config.posterior.alpha, 0.1)

    def test_every_placement_traces(self):
        pg = data()["pg"]
        for placement in L.PLACEMENTS:
            lowered = L.trace_bca(pg, SET, 0, placement, signal=np.ones(pg.n))
            self.assertIn("func", lowered.as_text()[:2000])

    def test_every_bca_path_refuses_without_approval(self):
        pg = data()["pg"]
        host_state = host()[1]
        forbidden = AssertionError("a BCA component ran")
        tmp = Path(tempfile.mkdtemp())
        report = tmp / "report.json"
        report.write_text(json.dumps(dict(prereg_sha256="x", bca_allowed=True)))
        fake = L.Approval(str(tmp / "approval.json"), str(report), "0" * 64, "x", "0" * 64)
        bad_file = tmp / "approval_bad.json"
        bad_file.write_text(json.dumps(dict(kind="bca-shift-approval", report_path=str(report), report_sha256="0" * 64,
                                            prereg_sha256="x")))
        with mock.patch.object(P, "fit_scale", side_effect=forbidden), \
                mock.patch.object(P, "refresh", side_effect=forbidden), \
                mock.patch.object(P, "bc_readout", side_effect=forbidden), \
                mock.patch.object(P, "update", side_effect=forbidden), \
                mock.patch.object(P, "make_train_step", side_effect=forbidden):
            for approval in (None, "approved", fake):
                with self.assertRaises(L.ApprovalError):
                    L.bca_cotrain(pg, SET, 0, approval)
                for placement in L.PLACEMENTS:
                    with self.assertRaises(L.ApprovalError):
                        L.bca_cotrain(pg, SET, 0, approval, placement=placement, noise_map=np.ones(pg.n))
                with self.assertRaises(L.ApprovalError):
                    L.bca_band(host_state, pg, SET, 0, approval)
                with self.assertRaises(L.ApprovalError):
                    L.bca_doses(host_state, pg, approval)
                with self.assertRaises(L.ApprovalError):
                    L.identity_placement_check(pg, SET, 0, approval)
            for path in (tmp / "missing.json", bad_file):
                with self.assertRaises(L.ApprovalError):
                    L.verify_approval(path)
            bad_file.with_name(bad_file.name + ".sha256").write_text("0" * 64 + "  approval_bad.json\n")
            with self.assertRaises(L.ApprovalError):
                L.verify_approval(bad_file)

    def test_hand_written_approval_pair_is_refused(self):
        """A report saying bca_allowed plus an approval holding its SHA-256, written by hand, is refused: the
        approval must belong to the one official pre-registration in runs/shift that acceptance.load_prereg accepts
        (the review's probe). Nothing BCA-related runs."""
        import hashlib

        def write(path, body):
            text = json.dumps(body, sort_keys=True)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(text)
            digest = hashlib.sha256(text.encode()).hexdigest()
            path.with_name(path.name + ".sha256").write_text(f"{digest}  {path.name}\n")
            return digest

        tmp = Path(tempfile.mkdtemp())
        report = tmp / "report.json"
        rsha = write(report, dict(prereg_sha256="anything", bca_allowed=True))
        approval = tmp / "approval.json"
        write(approval, dict(kind="bca-shift-approval", report_path=str(report), report_sha256=rsha,
                             prereg_sha256="anything"))
        with self.assertRaises(L.ApprovalError):
            L.verify_approval(approval)
        # the official layout, written by hand in a temporary 'runs/shift': refused by the preset, then by
        # acceptance.load_prereg (the real preset runs only from the repository's runs/shift)
        runs = tmp / "runs" / "shift"
        for preset in ("smoke", "real"):
            psha = write(runs / "preregistration.json", dict(kind="bca-shift-preregistration", config=dict(
                preset=preset)))
            root = runs / psha[:16]
            rsha = write(root / "report" / "report.json", dict(kind="shift-report", prereg_sha256=psha,
                                                              bca_allowed=True))
            write(root / "approval.json", dict(kind="bca-shift-approval", report_sha256=rsha, prereg_sha256=psha,
                                               report_path=str(root / "report" / "report.json")))
            with mock.patch.object(L, "OFFICIAL_RUNS", runs):
                with self.assertRaises(L.ApprovalError) as ctx:
                    L.verify_approval(root / "approval.json")
            self.assertIn("real-preset" if preset == "smoke" else "does not verify", str(ctx.exception))
        self.assertFalse(L._PREREG_VERIFIED)


if __name__ == "__main__":
    unittest.main()

"""Stage 2 unit tests. SYNTHETIC ONLY: every array here is random or hand-made, every network is randomly
initialised; no ThrusterChain log, critic, actor or band is loaded, and no Stage 2 phase is run on real data. BCA's
own functions (td3_bc_bca.fit_scale and refresh, wbcp.calibrate) run only on small random arrays.

What is tested:
  - the width readout U(s, a): its value against BCA's formula, its gradient in the command (against finite
    differences) flowing through the frozen scale network's command input and never into its parameters, "cannot
    certify" giving U_max with finite gradients, 0 before the first refresh, the weighted form;
  - the paired one-sided penalty (the hinge): zero at the recorded command, never negative, its gradient;
  - the BCA actor step with k = 0 against the room harness's own step (same loss term for term);
  - the co-trained host's update with no penalty against td3_bc.td3_bc_update;
  - BCA's band fit on random arrays (scale fit, refresh, readout) and the tabulated weighted threshold;
  - the approval: terminal mode, chat mode, and every refusal (no approval, not a terminal, wrong characters, draft,
    changed pre-registration, changed approval file, changed code, existing approval, chat not exactly the approval
    sentence (negations, questions, quotations, added conditions), chat not allowed, a pinned component that changed
    between writing and approving);
  - the modes a pre-registration does not accept: version 6 accepts only local-terminal; approve.py terminal and chat
    refuse, and so does every BCA entry point for an approval.json in either mode, even one written by hand;
  - the local-terminal route (version 6): approve_local.py's folder checks and approval body on a synthetic committed
    folder; 'approve.py import', which reads the commit facts itself from a stand-in for git (FakeGit: no test runs
    git), and its refusals (no file, a file nobody committed, an unpushed commit, two adding commits, the file changed
    with or without its sidecar, wrong characters, another mode, another version's approval, another manifest or
    approve_local.py, an author or committer that is the cloud session's identity, a commit or committed folder that
    does not hold the current files, changed pins, the mode not accepted, an approval already in place); the import
    record, without which every BCA entry point refuses until it is on origin's branch; a new version approved with
    a new file while the old one stays; an imported approval voided by a later edit of prereg.json, of the committed
    file, of the import record, of the stored copy or of the git record, or by a deletion on origin's branch; a
    hand-written runs-side approval for a commit git does not hold; git() refusing any query that is not read-only;
    'approve.py show' saying whether the fingerprint can be approved now;
  - every BCA entry point refuses without a valid approval before touching any data;
  - the output guard, the stand-in selection rule and the detection marks' logic on made-up numbers (all five
    detection stand-ins needed on every seed; seeds 0-4 carry no claim);
  - the write-once phases' input rules on made-up files in a temporary folder: the detection marks refuse while a
    seed's inputs are missing and accept a recorded failure; the deployment check waits for the version decision; a
    fresh seed's recorded data or learner failure counts as a stopped seed; gate seeds' pair sets are never built in
    the rehearsal's folder (a temporary stand-in for it);
  - the declared gradient paths of the weighted width and the inert uniform "cannot certify".

Run (1 process, low priority):
  cd /home/user/bca && PYTHONPATH=/home/user/bca PYTHONDONTWRITEBYTECODE=1 JAX_PLATFORMS=cpu OMP_NUM_THREADS=1 \
    nice -n 15 /tmp/claude-0/bcavenv/bin/python <stage2>/test_stage2.py
"""

import hashlib
import io
import json
import math
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, "/home/user/bca")

import numpy as np  # noqa: E402
import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402

import stage2_common as C  # noqa: E402
import band as B  # noqa: E402
import weights as Wt  # noqa: E402
import approve as AP  # noqa: E402
import approve_local as AL  # noqa: E402
import detect_contract as DC  # noqa: E402

TMP_ROOT = HERE.parent.parent / "stage2_tests"


def synthetic_band(dx=13, da=4, seed=0, R=1.7, unit=0.3, certified=True, ready=True):
    cal = B.calibrator(dx)
    p = cal.init(jax.random.PRNGKey(seed), jnp.zeros((1, dx)), jnp.zeros((1, da)))
    band = B.uniform_band(p, unit, R if certified else math.inf, R, 5.0, ready=ready)
    return cal, band


class Tty(io.StringIO):
    def isatty(self):
        return True

    def fileno(self):
        raise OSError("no file descriptor (test)")


class NotTty(io.StringIO):
    def isatty(self):
        return False


# --------------------------------------------------------------------------------------------- width and hinge

class WidthTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(1)
        self.obs = jnp.asarray(rng.normal(size=(8, 13)), jnp.float32)
        self.a = jnp.asarray(rng.uniform(-1, 1, size=(8, 4)), jnp.float32)

    def test_value_is_bca_formula(self):
        cal, band = synthetic_band()
        u = B.width_uniform(cal, band, self.obs, self.a)
        eta = cal.apply(band.cal_params, self.obs, self.a)
        np.testing.assert_allclose(np.asarray(u), 1.7 * 0.3 * np.maximum(np.asarray(eta), 1e-6), rtol=1e-6)

    def test_gradient_in_command_matches_finite_differences(self):
        cal, band = synthetic_band()
        g = B.width_grad_rows(cal, band, np.asarray(self.obs), np.asarray(self.a), False)
        self.assertTrue(np.all(np.isfinite(g)))
        self.assertGreater(np.abs(g).max(), 0.0)          # the width can steer the command
        h = 1e-3
        for i in range(4):
            for j in range(4):
                ap, am = np.asarray(self.a).copy(), np.asarray(self.a).copy()
                ap[i, j] += h
                am[i, j] -= h
                up = B.width_rows(cal, band, np.asarray(self.obs)[i:i + 1], ap[i:i + 1], False)[0]
                um = B.width_rows(cal, band, np.asarray(self.obs)[i:i + 1], am[i:i + 1], False)[0]
                fd = (up - um) / (2 * h)
                self.assertLess(abs(fd - g[i, j]), 2e-3 + 2e-2 * abs(g[i, j]), (i, j, fd, g[i, j]))

    def test_no_gradient_into_frozen_parameters(self):
        cal, band = synthetic_band()
        grads = jax.grad(lambda p: B.width_uniform(cal, band._replace(cal_params=p), self.obs, self.a).sum())(
            band.cal_params)
        self.assertEqual(max(float(jnp.abs(x).max()) for x in jax.tree_util.tree_leaves(grads)), 0.0)

    def test_cannot_certify_is_maximum_caution(self):
        cal, band = synthetic_band(certified=False)
        u = np.asarray(B.width_uniform(cal, band, self.obs, self.a))
        np.testing.assert_allclose(u, 5.0)                 # U_max everywhere, never 0
        g = B.width_grad_rows(cal, band, np.asarray(self.obs), np.asarray(self.a), False)
        self.assertTrue(np.all(np.isfinite(g)))            # no inf * 0 in the gradient
        self.assertEqual(float(np.abs(g).max()), 0.0)

    def test_not_ready_is_zero(self):
        cal, band = synthetic_band(ready=False)
        self.assertEqual(float(jnp.abs(B.width_uniform(cal, band, self.obs, self.a)).max()), 0.0)

    def test_hinge(self):
        u = jnp.asarray([0.1, 0.4, 0.9])
        r = jnp.asarray([0.5, 0.5, 0.5])
        np.testing.assert_allclose(np.asarray(B.paired(u, r)), [0.0, 0.0, 0.4], atol=1e-7)
        # gradient: zero where the command's width is below the recorded one, one above; none into the reference
        gu = jax.grad(lambda x: B.paired(x, r).sum())(u)
        gr = jax.grad(lambda y: B.paired(u, y).sum())(r)
        np.testing.assert_allclose(np.asarray(gu), [0.0, 0.0, 1.0])
        np.testing.assert_allclose(np.asarray(gr), [0.0, 0.0, 0.0])

    def test_hinge_zero_at_recorded_command(self):
        cal, band = synthetic_band()
        u_rec = B.width_uniform(cal, band, self.obs, self.a)
        np.testing.assert_array_equal(np.asarray(B.paired(B.width_uniform(cal, band, self.obs, self.a), u_rec)), 0.0)
        q1 = jnp.ones(8)
        obj = B.lower_edge_objective(q1, u_rec + 0.25, u_rec, 1.0)
        np.testing.assert_allclose(np.asarray(obj), 0.75, rtol=1e-6)
        obj = B.lower_edge_objective(q1, u_rec - 0.25, u_rec, 1.0, k_unpaired=0.0)   # never a reward
        np.testing.assert_allclose(np.asarray(obj), 1.0, rtol=1e-6)

    def test_weighted_width(self):
        cal, band = synthetic_band()
        cls_p, _ = Wt.cls_init(3, 13, 4)
        log_t = jnp.linspace(-3.0, 3.0, B.GRID)
        R_tab = jnp.linspace(1.0, 2.0, B.GRID)
        wb = band._replace(log_t=log_t.astype(jnp.float32), R_tab=R_tab.astype(jnp.float32),
                           l_star=B.f32(2.0), cls_params=cls_p, U_max=B.f32(4.0))
        u = np.asarray(B.width_weighted(cal, wb, self.obs, self.a))
        lw = np.asarray(Wt.logit(cls_p, self.obs, self.a))
        eta = np.maximum(np.asarray(cal.apply(band.cal_params, self.obs, self.a)), 1e-6)
        want = np.interp(lw, np.asarray(log_t), np.asarray(R_tab)) * 0.3 * eta + 4.0 * np.log1p(np.exp(lw - 2.0))
        np.testing.assert_allclose(u, want, rtol=1e-5)
        g = B.width_grad_rows(cal, wb, np.asarray(self.obs), np.asarray(self.a), True)
        self.assertTrue(np.all(np.isfinite(g)))


# --------------------------------------------------------------------------------------------- programs

class ProgramTests(unittest.TestCase):
    def test_bca_actor_step_with_k0_equals_room_step(self):
        import policy as PO
        H, L = PO.H, PO.L
        rng = np.random.default_rng(2)
        n = 64
        obs = jnp.asarray(rng.normal(size=(n, 13)), jnp.float32)
        act = jnp.asarray(rng.uniform(-1, 1, size=(n, 4)), jnp.float32)
        from runtime.networks import Transition
        train = Transition(obs, act, jnp.zeros(n), obs, jnp.zeros(n))
        actor_p = L.ACTOR.init(jax.random.PRNGKey(0), jnp.zeros(13))
        critic_p = L.CRITIC.init(jax.random.PRNGKey(1), jnp.zeros(13), jnp.zeros(4))
        from flax.training.train_state import TrainState
        st = TrainState.create(apply_fn=L.ACTOR.apply, params=actor_p, tx=L.adam(3e-4))
        args = L.make_args(L.LearnSettings(), 0, 5.0)
        cal, band = synthetic_band()
        shared = dict(obs=obs, perm=jnp.asarray(rng.permutation(n), jnp.int32), band=band,
                      u_rec=B.width_uniform(cal, band, obs, act))
        rows = jnp.arange(16)
        batch = jax.tree_util.tree_map(lambda x: x[rows], train)
        a0, aux0 = H.actor_step_w(args, "plain", H.ProgramConfig(), st, critic_p, batch, rows, {},
                                  dict(coef=jnp.zeros(1), pull=jnp.float32(1.0)))
        a1, aux1 = PO.actor_step_bca(args, "bca_u", st, critic_p, batch, rows, shared,
                                     dict(coef=jnp.zeros(3), pull=jnp.float32(1.0)))
        diff = max(float(jnp.abs(x - y).max()) for x, y in zip(jax.tree_util.tree_leaves(a0.params),
                                                                jax.tree_util.tree_leaves(a1.params)))
        self.assertLessEqual(diff, 1e-7)
        a2, aux2 = PO.actor_step_bca(args, "bca_u", st, critic_p, batch, rows, shared,
                                     dict(coef=jnp.asarray([1.0, 0.0, 0.0]), pull=jnp.float32(1.0)))
        self.assertTrue(np.isfinite(float(aux2[0])))

    def test_runner_on_random_arrays(self):
        """BCARunner (the room's RoomRunner with the BCA kinds), 8 steps on 64 random rows, uniform and weighted kinds
        vmapped side by side with the room's plain kind (synthetic data and networks)."""
        import policy as PO
        from types import SimpleNamespace
        from runtime.networks import Transition
        H, L = PO.H, PO.L
        r = np.random.default_rng(8)
        n = 64
        obs = jnp.asarray(r.normal(size=(n, 13)), jnp.float32)
        act = jnp.asarray(r.uniform(-1, 1, size=(n, 4)), jnp.float32)
        prep = SimpleNamespace(train=Transition(obs, act, jnp.zeros(n), obs, jnp.zeros(n)), n=n)
        actor_p = L.ACTOR.init(jax.random.PRNGKey(9), jnp.zeros(13))
        critic_p = L.CRITIC.init(jax.random.PRNGKey(10), jnp.zeros(13), jnp.zeros(4))
        cal, band = synthetic_band()
        shared = dict(obs=obs, perm=jnp.asarray(r.permutation(n), jnp.int32), band=band,
                      u_rec=B.width_uniform(cal, band, obs, act))
        cls_p, _ = Wt.cls_init(0, 13, 4)
        wband = band._replace(log_t=jnp.linspace(-3, 3, B.GRID).astype(jnp.float32),
                              R_tab=jnp.linspace(1, 2, B.GRID).astype(jnp.float32), l_star=B.f32(2.0),
                              cls_params=cls_p)
        calls = []

        def refit(p):
            calls.append(1)
            return wband, np.asarray(B.width_weighted(cal, wband, obs, act)), dict(finite=True)

        runner = PO.BCARunner(prep, actor_p, critic_p, L.LearnSettings(), 0, cfg=H.ProgramConfig(), shared=shared,
                              random_nets={}, steps=8, chunk=4, refit=refit)
        out = runner.run([PO.bca_arm("w0", "bca_u", (0, 0, 0)), H.RoomArm("plain", "plain", (0.0,), 1.0),
                          PO.bca_arm("k1", "bca_u", (1, 0, 0)), PO.bca_arm("k1w", "bca_w", (1, 0, 0)),
                          PO.bca_arm("unp", "bca_u", (0, 1, 0)), PO.bca_arm("shuf", "bca_u", (0, 0, 1))])
        self.assertTrue(all(v["finite"] for v in out.values()))
        self.assertLessEqual(H.max_param_diff(out["w0"]["params"], out["plain"]["params"]), 1e-6)
        self.assertGreater(H.max_param_diff(out["k1"]["params"], out["plain"]["params"]), 0.0)
        self.assertEqual(len(calls), 2)                     # one weighted arm, refitted before each of 2 chunks

    def test_bca_penalty_moves_the_actor_through_the_command(self):
        import policy as PO
        L = PO.L
        rng = np.random.default_rng(3)
        n = 32
        obs = jnp.asarray(rng.normal(size=(n, 13)), jnp.float32)
        actor_p = L.ACTOR.init(jax.random.PRNGKey(4), jnp.zeros(13))
        cal, band = synthetic_band()
        shared = dict(obs=obs, perm=jnp.arange(n, dtype=jnp.int32), band=band,
                      u_rec=jnp.zeros(n))          # recorded width 0: the paired penalty equals U at the actor's command
        rows = jnp.arange(n)
        arm = dict(coef=jnp.asarray([1.0, 0.0, 0.0]), pull=jnp.float32(1.0))
        g = jax.grad(lambda p: PO.bca_penalty("bca_u", shared, arm, rows, L.ACTOR.apply(p, obs)).sum())(actor_p)
        self.assertGreater(max(float(jnp.abs(x).max()) for x in jax.tree_util.tree_leaves(g)), 0.0)

    def test_host_update_without_penalty_equals_td3_bc(self):
        import host as HO
        import algorithms.td3_bc as BASE
        from runtime.networks import Transition
        args = BASE.Args(seed=0, batch_size=16, alpha=5.0, discount=0.98)
        rng, state, (actor, critic) = BASE.initialize(args, 6, 2, 1.0)
        r = np.random.default_rng(5)
        batch = Transition(jnp.asarray(r.normal(size=(16, 6)), jnp.float32),
                           jnp.asarray(r.uniform(-1, 1, size=(16, 2)), jnp.float32),
                           jnp.asarray(r.normal(size=16), jnp.float32), jnp.asarray(r.normal(size=(16, 6)), jnp.float32),
                           jnp.zeros(16, jnp.float32))
        for it in (1, 2):
            s0, m0 = BASE.td3_bc_update(args, actor.apply, critic.apply, state, batch, it, rng, 1.0)
            s1, m1 = HO.td3_bc_lower_edge_update(args, actor.apply, critic.apply, state, batch, it, rng, 1.0, None)
            for x, y in zip(jax.tree_util.tree_leaves(s0), jax.tree_util.tree_leaves(s1)):
                np.testing.assert_array_equal(np.asarray(x), np.asarray(y))
            s2, _ = HO.td3_bc_lower_edge_update(args, actor.apply, critic.apply, state, batch, it, rng, 1.0,
                                                lambda pi: 0.0 * pi.sum(-1))
            d = max(float(np.max(np.abs(np.asarray(x) - np.asarray(y))))
                    for x, y in zip(jax.tree_util.tree_leaves(s0), jax.tree_util.tree_leaves(s2)))
            self.assertLessEqual(d, 1e-6)

    def test_band_fit_on_random_arrays(self):
        """BCA's own scale fit and refresh on 300 random rows and a 60-row random bank (synthetic), then the readout."""
        import algorithms.td3_bc as BASE
        from runtime.networks import Transition
        dx, da, n, nb = 5, 2, 300, 60
        r = np.random.default_rng(6)

        def tr(m):
            return Transition(jnp.asarray(r.normal(size=(m, dx)), jnp.float32),
                              jnp.asarray(r.uniform(-1, 1, size=(m, da)), jnp.float32),
                              jnp.asarray(r.normal(size=m), jnp.float32),
                              jnp.asarray(r.normal(size=(m, dx)), jnp.float32), jnp.zeros(m, jnp.float32))

        train, bank = tr(n), tr(nb)
        args = BASE.Args(seed=0, batch_size=32, alpha=5.0, discount=0.98)
        config = B.bca_config()
        _, native, (actor, critic) = BASE.initialize(args, dx, da, 1.0)
        rng, state, cal = B.initial_state(args, config, native, dx, da)
        models = (actor, critic, cal)
        steps = 40
        rows, keys, rkey = B.band_rows_and_keys(rng, n, steps, 32, 0)
        state, acc, diag = B.fit_band_arrays(args, config, models, state, train, bank, rows, keys, rkey)
        self.assertGreater(acc, 0.9)
        band = B.band_from_reference(state.posterior, cal, np.asarray(bank.obs), np.asarray(bank.action))
        self.assertTrue(bool(band.certified) and np.isfinite(float(band.U_max)))
        u = B.width_rows(cal, band, np.asarray(train.obs), np.asarray(train.action), False)
        self.assertTrue(np.all(np.isfinite(u)) and np.all(u > 0))
        self.assertLessEqual(u.max(), float(band.R) * float(band.unit) * 1e6)
        scores, _ = B.bank_scores(args, models, state, bank, rkey, state.posterior.cal_params,
                                  state.posterior.residual_scale)
        chk = B.uniform_threshold_check(scores, rkey, config, state.posterior.threshold)
        self.assertTrue(chk["equal"], chk)                 # bank_scores reproduces BCA's own refresh exactly

    def test_host_schedule_on_random_arrays(self):
        """The co-trained host's BCA step (scale fit + lower-edge actor) before and after a refresh, uniform and
        weighted extras, on 300 random rows of a 5-dimensional toy problem (synthetic)."""
        import host as HO
        import algorithms.td3_bc as BASE
        import algorithms.td3_bc_bca as P
        from runtime.networks import Transition
        dx, da, n, nb = 5, 2, 300, 60
        r = np.random.default_rng(11)

        def tr(m):
            return Transition(jnp.asarray(r.normal(size=(m, dx)), jnp.float32),
                              jnp.asarray(r.uniform(-1, 1, size=(m, da)), jnp.float32),
                              jnp.asarray(r.normal(size=m), jnp.float32),
                              jnp.asarray(r.normal(size=(m, dx)), jnp.float32), jnp.zeros(m, jnp.float32))

        train, bank = tr(n), tr(nb)
        args = BASE.Args(seed=0, batch_size=32, alpha=5.0, discount=0.98)
        config = B.bca_config()
        for weighted in (False, True):
            _, native, (actor, critic) = BASE.initialize(args, dx, da, 1.0)
            rng, state, cal = B.initial_state(args, config, native, dx, da)
            models = (actor, critic, cal)
            p0 = Wt.cls_init(0, dx, da)[0] if weighted else None
            extras = HO.initial_extras(weighted, p0)
            step = HO.make_bca_step(args, config, models, train, jnp.asarray([1.0, 0.0], jnp.float32), weighted)
            prog = jax.jit(lambda c: jax.lax.scan(step, c, None, length=10))
            carry, m = prog((rng, state, extras, jnp.int32(0)))
            self.assertTrue(bool(np.all(np.asarray(m["inputs_valid"]))))
            self.assertEqual(float(np.abs(np.asarray(m["kW_mean"])).max()), 0.0)     # width 0 before the refresh
            key = jax.random.PRNGKey(3)
            new, diag, _ = P.refresh(args, config, models, carry[1], bank, key, 1.0, heldout_ids=np.arange(nb))
            cloned = np.asarray(train.action)
            ex, ediag = HO.refresh_extras(args, config, models, new, bank, key, weighted, 0, actor_apply=actor.apply,
                                          train=train, cloned=cloned, cls_p0=p0,
                                          cls_rows=Wt.cls_rows(0, n, steps=50, batch=32))
            self.assertIsNotNone(ex, ediag)
            self.assertTrue(np.isfinite(float(ex.U_max)) and float(ex.U_max) > 0)
            carry, m = prog((carry[0], new, ex, carry[3]))
            self.assertTrue(bool(np.all(np.asarray(m["inputs_valid"]))))
            self.assertTrue(np.all(np.isfinite(np.asarray(m["kW_mean"]))))
            self.assertGreater(float(np.asarray(m["width_recorded_mean"]).mean()), 0.0)

    def test_tabulated_weighted_threshold(self):
        r = np.random.default_rng(7)
        scores = np.abs(r.normal(size=200))
        lw = r.normal(scale=0.5, size=200)
        config = B.bca_config()
        log_t, R, R_c, l_star = B.tabulate_threshold(scores, lw, 0, config)
        fin = np.isfinite(R)
        self.assertTrue(fin.any() and (~fin).any())        # it reaches "cannot certify" inside the grid
        self.assertTrue(np.all(np.diff(R[fin]) >= 0))      # nondecreasing in the test mass
        self.assertTrue(np.all(np.isfinite(R_c)))
        self.assertTrue(np.isfinite(l_star) and l_star == log_t[np.flatnonzero(~fin)[0]])
        first_inf = np.flatnonzero(~fin)[0]
        self.assertTrue(np.all(~fin[first_inf:]))          # once refused, refused for every larger test mass


# --------------------------------------------------------------------------------------------- approval

FAKE_PINS = dict(room=dict(manifest_sha256="a" * 64), detect_manifest_sha256="b" * 64)


def fake_pins():
    return json.loads(json.dumps(FAKE_PINS)), []


class ApprovalFixture:
    """Synthetic pre-registrations in a temporary folder. The other jobs' pins are a fixed stand-in (fake_pins), so
    the tests never depend on the live state of the room test, the stand-ins or the detection script."""

    def term(self, stdin=None, input_fn=None, pins_fn=fake_pins):
        return AP.approve_terminal(self.paths, Tty() if stdin is None else stdin,
                                   input_fn or (lambda _: self.prefix()), lambda *a: None, pins_fn=pins_fn)

    def chat(self, msg, pins_fn=fake_pins):
        return AP.approve_chat(msg, self.paths, lambda *a: None, pins_fn=pins_fn)

    def sentence(self):
        return C.chat_sentence(self.prefix())

    def setUp(self):
        TMP_ROOT.mkdir(parents=True, exist_ok=True)
        self.tmp = Path(tempfile.mkdtemp(dir=TMP_ROOT))
        self.code = self.tmp / "code_src"
        self.code.mkdir()
        (self.code / "a.py").write_text("x = 1\n")
        (self.code / "b.py").write_text("y = 2\n")
        (self.code / "PREREGISTRATION.md").write_text("# synthetic\n")
        (self.code / "approve_local.py").write_text("# synthetic stand-in for approve_local.py\n")
        self.paths = C.Paths(self.tmp / "out", code_src=self.code, repo=self.tmp / "repo" / "stage2")
        self.paths.out.mkdir()
        self.write_manifest()
        self.write_prereg()

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def write_manifest(self):
        files = {n: C.file_sha256(self.code / n) for n in ("a.py", "b.py", "PREREGISTRATION.md", "approve_local.py")}
        self.paths.code_copies.mkdir(exist_ok=True)
        for n in files:
            shutil.copy2(self.code / n, self.paths.code_copies / n)
        C.write_json(self.paths.manifest, dict(kind="stage2-code-manifest", files=files, repository_files={}),
                     self.paths, overwrite=True)

    def write_prereg(self, approvable=True, chat=True, extra=None, local=True, cloud=True):
        """A synthetic prereg.json. It accepts every mode by default, so that the code of each mode is tested; version
        6 itself accepts only local-terminal (test_version_6_accepts_only_local_terminal)."""
        man = C.read_json(self.paths.manifest)
        body = dict(kind="stage2-preregistration", label=C.LABEL, approvable=approvable,
                    missing_pins=[] if approvable else ["detect_module"],
                    approval=dict(chat_mode_allowed=chat, local_terminal_allowed=local, cloud_terminal_allowed=cloud,
                                  local_terminal=dict(branch="synthetic-branch")),
                    code=dict(code_manifest_sha256=C.file_sha256(self.paths.manifest), files=man["files"]),
                    pins=FAKE_PINS, extra=extra)
        return C.write_json(self.paths.prereg, body, self.paths, overwrite=True)

    def prefix(self):
        return C.load_prereg(self.paths).prefix


class ApprovalTests(ApprovalFixture, unittest.TestCase):
    def test_no_prereg_or_no_approval_refuses(self):
        self.assertRaises(C.ApprovalError, C.verify_approval, C.Paths(self.tmp / "nothing", code_src=self.code))
        self.assertRaises(C.ApprovalError, C.verify_approval, self.paths)

    def test_terminal_mode(self):
        self.assertRaises(AP.ApproveError, self.term, NotTty())
        self.assertFalse(self.paths.approval.exists())
        self.assertRaises(AP.ApproveError, self.term, Tty(), lambda _: "0" * 16)
        self.assertFalse(self.paths.approval.exists())
        self.term()
        ap = C.require_approval(self.paths)
        self.assertEqual(ap.mode, "terminal")
        body = json.loads(self.paths.approval.read_text())
        self.assertTrue(body["process_chain"])                 # evidence only: this test process and its parents
        self.assertIn("pseudo-terminal", body["terminal_mode_caveat"])
        self.assertIn("accepts every default", body["defaults_accepted"])
        # a second approval is refused
        self.assertRaises(AP.ApproveError, self.term)

    def test_changed_prereg_voids_approval(self):
        self.term()
        self.write_prereg(extra="edited after approval")          # a new, consistent prereg.json and sidecar
        with self.assertRaises(C.ApprovalError) as cm:
            C.require_approval(self.paths)
        self.assertIn("another version", str(cm.exception))

    def test_prereg_changed_without_sidecar_refuses(self):
        self.term()
        self.paths.prereg.write_text(self.paths.prereg.read_text().replace('"approvable": true', '"approvable": true '))
        self.assertRaises(C.ApprovalError, C.require_approval, self.paths)

    def test_changed_approval_file_refuses(self):
        self.term()
        body = json.loads(self.paths.approval.read_text())
        body["mode"] = "chat"
        self.paths.approval.write_text(json.dumps(body))
        self.assertRaises(C.ApprovalError, C.require_approval, self.paths)

    def test_forged_terminal_body_refuses(self):
        self.term()
        body = json.loads(self.paths.approval.read_text())
        body["typed"] = "f" * 16
        C.write_json(self.paths.approval, body, self.paths, overwrite=True)    # consistent sidecar, wrong typing
        self.assertRaises(C.ApprovalError, C.require_approval, self.paths)

    def test_changed_code_refuses(self):
        self.term()
        C.require_approval(self.paths)
        (self.code / "a.py").write_text("x = 3\n")
        self.assertRaises(C.ApprovalError, C.require_approval, self.paths)
        self.assertEqual(C.verify_approval(self.paths).mode, "terminal")     # the approval itself still verifies

    def test_changed_code_copy_refuses(self):
        self.term()
        (self.paths.code_copies / "b.py").write_text("y = 3\n")
        self.assertRaises(C.ApprovalError, C.require_approval, self.paths)

    def test_phases_refuse_a_draft_or_no_approval(self):
        import run_stage2 as RS
        self.assertRaises(SystemExit, RS.final_prereg, C.Paths(self.tmp / "nothing", code_src=self.code))
        self.write_prereg(approvable=False)
        self.assertRaises(SystemExit, RS.final_prereg, self.paths)
        self.assertRaises((SystemExit, C.ApprovalError), RS.bca_gate, self.paths)
        self.write_prereg(approvable=True)
        RS.final_prereg(self.paths)                                  # final, but not approved:
        self.assertRaises(C.ApprovalError, RS.bca_gate, self.paths)
        self.term()
        with self.assertRaises(SystemExit) as cm:                    # approved, but no best_standin.json yet
            RS.bca_gate(self.paths, pins_fn=fake_pins)
        self.assertIn("best_standin.json", str(cm.exception))
        C.write_json(self.paths.out / "best_standin.json", dict(best=None), self.paths)
        RS.bca_gate(self.paths, pins_fn=fake_pins)                   # everything in place
        with self.assertRaises(SystemExit) as cm:                    # another job's component changed since
            RS.bca_gate(self.paths, pins_fn=lambda: (dict(FAKE_PINS, detect_manifest_sha256="d" * 64), []))
        self.assertIn("detect_manifest_sha256", str(cm.exception))

    def test_draft_refuses(self):
        self.write_prereg(approvable=False)
        self.assertRaises(AP.ApproveError, self.term)
        msg = self.tmp / "m.txt"
        msg.write_text(self.sentence())
        self.assertRaises(AP.ApproveError, self.chat, msg)

    def test_chat_mode(self):
        msg = self.tmp / "m.txt"
        pre = self.prefix()
        refused = [
            f"approve {pre}",                                                   # no acceptance of chat mode
            "approve, accept chat mode",                                        # no prefix
            f"approve {pre[:15]}, accept chat mode",                            # 15 characters only
            f"approve {pre}0, accept chat mode",                                # 17 characters
            f"I do NOT accept chat mode. What is {pre}?",                       # a negation (the reviewer's cases)
            f"Should I accept chat mode for {pre}? Not approving yet.",         # a question
            f"Here is the builder's text: the user would type or send `{pre}`. Why would anyone accept chat mode?",
            f"{pre} - explain what 'accept chat mode' means before I decide",
            f"{pre}, accept chat mode, but read M5 on the BCA k=1 actor",       # an added condition
            f"approve {pre}, accept chat mode, but read M5 on the BCA k=1 actor",
            f"approve {pre}, accept chat mode. Also run the host only after the policy marks",
            f"I approve {pre}, I accept chat mode",                             # not the sentence
            f"do not approve {pre}, accept chat mode",
            f"`approve {pre}, accept chat mode`",                               # quoted
        ]
        for text in refused:
            with self.subTest(text=text):
                msg.write_text(text)
                self.assertRaises(AP.ApproveError, self.chat, msg)
                self.assertFalse(C.message_approves(text, pre))
        self.assertFalse(self.paths.approval.exists())
        text = f"  Approve {pre.upper()},  Accept Chat Mode.\n"              # case, spaces, final full stop ignored
        msg.write_text(text)
        self.chat(msg)
        ap = C.require_approval(self.paths)
        self.assertEqual(ap.mode, "chat")
        body = json.loads(self.paths.approval.read_text())
        self.assertEqual(body["message_verbatim"], text)
        self.assertEqual(body["message_sha256"], hashlib.sha256(text.encode()).hexdigest())
        self.assertIn("trusting the orchestrating agent", body["chat_mode_caveat"])
        self.assertEqual(body["required_sentence"], self.sentence())

    def test_chat_sentence_for_another_prereg_refuses(self):
        msg = self.tmp / "m.txt"
        msg.write_text(C.chat_sentence("0123456789abcdef"))
        self.assertRaises(AP.ApproveError, self.chat, msg)

    def test_chat_not_allowed_refuses(self):
        self.write_prereg(chat=False)
        msg = self.tmp / "m.txt"
        msg.write_text(self.sentence())
        self.assertRaises(AP.ApproveError, self.chat, msg)

    def test_pin_changed_since_writing_refuses_approval(self):
        """A pinned component of another job that changed between writing prereg.json and approving it."""
        moved = lambda: (dict(FAKE_PINS, detect_manifest_sha256="c" * 64), [])       # noqa: E731
        missing = lambda: (FAKE_PINS, ["standins_py (changed)"])                      # noqa: E731
        msg = self.tmp / "m.txt"
        msg.write_text(self.sentence())
        for pins_fn in (moved, missing):
            self.assertRaises(AP.ApproveError, self.term, pins_fn=pins_fn)
            self.assertRaises(AP.ApproveError, self.chat, msg, pins_fn=pins_fn)
        self.assertFalse(self.paths.approval.exists())
        self.chat(msg)                                               # unchanged pins: approved
        self.assertEqual(C.require_approval(self.paths).mode, "chat")

    def test_version_6_accepts_only_local_terminal(self):
        import prereg_payload as PP
        section = PP.approval()
        self.assertEqual(C.allowed_modes(dict(approval=section)), [C.LOCAL_MODE])
        self.assertIs(section["cloud_terminal_allowed"], False)
        self.assertIs(section["chat_mode_allowed"], False)
        self.assertEqual(C.allowed_modes(dict(approval={})), [])                  # a missing flag is false

    def test_modes_not_accepted_refuse(self):
        """The user's choice is enforced: with cloud_terminal_allowed and chat_mode_allowed false, approve.py terminal
        and chat refuse, and every BCA entry point refuses an approval.json in either mode, even one written by hand
        with a consistent sidecar."""
        self.write_prereg(chat=False, cloud=False)
        self.assertRaises(AP.ApproveError, self.term)                             # a pseudo-terminal would not help
        msg = self.tmp / "m.txt"
        msg.write_text(self.sentence())
        self.assertRaises(AP.ApproveError, self.chat, msg)
        self.assertFalse(self.paths.approval.exists())
        prereg = C.load_prereg(self.paths)
        by_hand = {
            "terminal": dict(mode="terminal", typed=prereg.prefix, stdin_isatty=True),
            "chat": dict(mode="chat", message_verbatim=self.sentence(),
                         message_sha256=hashlib.sha256(self.sentence().encode()).hexdigest()),
        }
        for mode, fields in by_hand.items():
            with self.subTest(mode=mode):
                C.write_json(self.paths.approval, dict(kind="stage2-approval", label=C.LABEL,
                                                       prereg_path=str(self.paths.prereg),
                                                       prereg_sha256=prereg.sha256, **fields), self.paths,
                             overwrite=True)
                with self.assertRaises(C.ApprovalError) as cm:
                    C.require_approval(self.paths)
                self.assertIn(f"does not accept approval mode '{mode}'", str(cm.exception))

    def test_chat_message_edited_in_approval_refuses(self):
        msg = self.tmp / "m.txt"
        msg.write_text(self.sentence())
        self.chat(msg)
        body = json.loads(self.paths.approval.read_text())
        body["message_verbatim"] = "something else"
        C.write_json(self.paths.approval, body, self.paths, overwrite=True)
        self.assertRaises(C.ApprovalError, C.require_approval, self.paths)


class FakeGit:
    """A stand-in for stage2_common.git: no test runs git. It holds commits (their files, author, committer and parent)
    and origin's branch, and answers exactly the read-only queries stage2_common makes, as git would."""

    def __init__(self, branch="synthetic-branch", email="noreply@anthropic.com"):
        self.commits, self.refs, self.branch, self.email, self.calls, self.n = {}, {}, branch, email, [], 0

    @property
    def ref(self):
        return "refs/remotes/origin/" + self.branch

    def commit(self, changes=None, deleted=(), author="Synthetic User <user@example.org>", committer=None,
               parent=None, push=True):
        parent = self.refs.get(self.ref) if parent is None else parent
        files = dict(self.commits[parent]["files"]) if parent else {}
        files.update(changes or {})
        for rel in deleted:
            files.pop(rel)
        self.n += 1
        h = hashlib.sha1(f"synthetic commit {self.n}".encode()).hexdigest()
        self.commits[h] = dict(parent=parent, files=files, author=author, committer=committer or author)
        if push:
            self.refs[self.ref] = h
        return h

    def history(self, h):
        out = []
        while h:
            out.append(h)
            h = self.commits[h]["parent"]
        return out

    def _rev(self, rev):
        return self.refs.get(rev, rev if rev in self.commits else None)

    def __call__(self, *args):
        self.calls.append(args)
        a = list(args)
        if a[0] == "rev-parse" and a[1:3] == ["--verify", "--quiet"] and a[3].endswith("^{commit}"):
            h = self._rev(a[3][:-len("^{commit}")])
            return (0, (h + "\n").encode()) if h else (1, b"")
        if a[0] == "cat-file" and a[1] == "-e" and a[2].endswith("^{commit}"):
            return (0, b"") if a[2][:-len("^{commit}")] in self.commits else (128, b"")
        if a[0] == "cat-file" and a[1] == "blob":
            rev, rel = a[2].split(":", 1)
            h = self._rev(rev)
            if h is None or rel not in self.commits[h]["files"]:
                return 128, b""
            return 0, self.commits[h]["files"][rel]
        if a[0] == "log" and a[1:3] == ["--no-renames", "--diff-filter=A"] and a[3] == "--format=%H":
            ref, sep, rel = a[4:7]
            assert sep == "--"
            h = self._rev(ref)
            if h is None:
                return 128, b""
            added = [c for c in self.history(h) if rel in self.commits[c]["files"] and not (
                self.commits[c]["parent"] and rel in self.commits[self.commits[c]["parent"]]["files"])]
            return 0, "".join(c + "\n" for c in added).encode()
        if a[0] == "log" and a[1] == "-1":
            c = self.commits.get(a[3])
            if c is None:
                return 128, b""
            if a[2] == "--format=%G?":
                return 0, b"N\n"
            assert a[2] == "--format=%an%x00%ae%x00%cn%x00%ce%x00%P"
            (an, ae), (cn, ce) = (C._AUTHOR_RE.fullmatch(c[k]).groups() for k in ("author", "committer"))
            return 0, f"{an}\0{ae}\0{cn}\0{ce}\0{c['parent'] or ''}\n".encode()
        if a[0] == "merge-base" and a[1] == "--is-ancestor":
            h = self._rev(a[3])
            return (0, b"") if h and a[2] in self.history(h) else (1, b"")
        if a[0] == "config" and a[1:] == ["--get", "user.email"]:
            return (0, (self.email + "\n").encode()) if self.email else (1, b"")
        raise AssertionError(f"unexpected git query {args}")


class LocalTerminalTests(ApprovalFixture, unittest.TestCase):
    """The only route version 6 accepts: a synthetic committed folder, a local approval made by approve_local.py's own
    functions (its git queries replaced), a stand-in for git holding the user's commit and origin's branch, and
    'approve.py import', which reads the commit facts itself."""

    USER = "Synthetic User <user@example.org>"

    def setUp(self):
        super().setUp()
        self.git = FakeGit()
        self.saved_git, C.git = C.git, self.git
        self.write_prereg(chat=False, cloud=False)          # as version 6: only local-terminal is accepted
        self.push_folder()

    def tearDown(self):
        C.git = self.saved_git
        super().tearDown()

    def rel(self, name):
        return C.REPO_STAGE2_REL + "/" + name

    def commit_folder(self):
        """Copy what is committed: prereg.json, the manifest, code/, PREREGISTRATION.md, approve_local.py. Earlier
        approval files and import records stay."""
        repo = self.paths.repo
        if (repo / "code").exists():
            shutil.rmtree(repo / "code")
        (repo / "code").mkdir(parents=True)
        for name in ("prereg.json", "prereg.json.sha256", "code_manifest.json", "code_manifest.json.sha256"):
            shutil.copy2(self.paths.out / name, repo / name)
        for name in C.read_json(self.paths.manifest)["files"]:
            shutil.copy2(self.paths.code_copies / name, repo / "code" / name)
        for name in ("PREREGISTRATION.md", "approve_local.py"):
            shutil.copy2(self.code / name, repo / name)
        return repo

    def push_folder(self, **kw):
        """The orchestrating agent's commit of the committed folder, on origin's branch."""
        repo = self.commit_folder()
        files = {self.rel(str(p.relative_to(repo))): p.read_bytes() for p in repo.rglob("*") if p.is_file()}
        return self.git.commit(files, author="Claude <noreply@anthropic.com>", **kw)

    def local_approve(self, typed=None, **changes):
        """approve_local.py's checks and approval body (its git queries replaced), written as the script writes it."""
        repo = self.paths.repo
        info = AL.check_folder(str(repo), script_path=str(repo / "approve_local.py"))
        saved, AL.git = AL.git, (lambda *a: None)
        try:
            body = AL.approval_body(info, typed or info["sha256"][:16])
        finally:
            AL.git = saved
        body.update(changes)
        return self.write_local(body)

    def write_local(self, body, sidecar=True):
        text = json.dumps(body, indent=1, sort_keys=True).encode("utf8")
        f = self.paths.local_approval(self.prefix())
        f.write_bytes(text)
        if sidecar:
            f.with_name(f.name + ".sha256").write_text(f"{hashlib.sha256(text).hexdigest()}  {f.name}\n")
        return hashlib.sha256(text).hexdigest()

    def user_push(self, author=USER, committer=None, push=True, **changes):
        """The user's run of approve_local.py, then 'git add' of the two files, commit and push."""
        sha = self.local_approve(**changes)
        f = self.paths.local_approval(self.prefix())
        side = f.with_name(f.name + ".sha256")
        c = self.git.commit({self.rel(f.name): f.read_bytes(), self.rel(side.name): side.read_bytes()},
                            author=author, committer=committer, push=push)
        return sha, c

    def imp(self, pins_fn=fake_pins):
        return AP.import_local(self.paths, lambda *a: None, pins_fn=pins_fn)

    def push_record(self):
        rec = self.paths.import_record(self.prefix())
        side = rec.with_name(rec.name + ".sha256")
        return self.git.commit({self.rel(rec.name): rec.read_bytes(), self.rel(side.name): side.read_bytes()},
                               author="Claude <noreply@anthropic.com>")

    def refuses(self, pattern=None, fn=None):
        with self.assertRaises((AP.ApproveError, C.ApprovalError)) as cm:
            (fn or self.imp)()
        if pattern:
            self.assertIn(pattern, str(cm.exception))
        return str(cm.exception)

    def test_git_queries_are_read_only(self):
        C.git = self.saved_git                                                   # the real function; nothing is run
        for args in (("commit", "-m", "x"), ("push",), ("config", "user.email", "x@example.org"), ("checkout", "x"),
                     ("fetch",), ()):
            with self.subTest(args=args):
                self.assertRaises(ValueError, C.git, *args)

    def test_local_folder_checks(self):
        repo = self.commit_folder()
        info = AL.check_folder(str(repo), script_path=str(repo / "approve_local.py"))
        self.assertEqual(info["sha256"], C.load_prereg(self.paths).sha256)
        (repo / "code" / "a.py").write_text("x = 9\n")                          # a committed copy changed
        self.assertRaises(AL.Refused, AL.check_folder, str(repo), str(repo / "approve_local.py"))
        repo = self.commit_folder()
        (repo / "code" / "extra.py").write_text("z = 0\n")                       # a file the manifest does not list
        self.assertRaises(AL.Refused, AL.check_folder, str(repo), str(repo / "approve_local.py"))
        repo = self.commit_folder()
        self.assertRaises(AL.Refused, AL.check_folder, str(repo), str(HERE / "approve_local.py"))  # not the pinned one
        (repo / "prereg.json").write_bytes((repo / "prereg.json").read_bytes().replace(b"\n", b"\r\n"))
        with self.assertRaises(AL.Refused) as cm:                               # line endings converted
            AL.check_folder(str(repo), str(repo / "approve_local.py"))
        self.assertIn("CR LF", str(cm.exception))
        self.write_prereg(local=False)
        repo = self.commit_folder()
        self.assertRaises(AL.Refused, AL.check_folder, str(repo), str(repo / "approve_local.py"))

    def test_import_accepted_and_verified(self):
        local_sha, commit = self.user_push(committer="Synthetic User <user@example.org>")
        body = self.imp()
        self.assertTrue(self.paths.import_record(self.prefix()).exists())
        # the import record is not on origin's branch yet: every BCA entry point refuses
        self.refuses("not on origin's branch", lambda: C.require_approval(self.paths))
        self.push_record()
        ap = C.require_approval(self.paths)
        self.assertEqual(ap.mode, C.LOCAL_MODE)
        stored = json.loads(self.paths.approval.read_text())
        self.assertEqual(stored["git"]["commit"], commit)                         # read from git, not passed in
        self.assertEqual(stored["git"]["author"], self.USER)
        self.assertEqual(stored["git"]["committer"], self.USER)
        self.assertEqual(stored["git"]["signature_status"], "N")
        self.assertEqual(stored["local_approval"]["sha256"], local_sha)
        self.assertEqual(stored["local_approval"]["path"], C.local_approval_rel(self.prefix()))
        self.assertEqual(stored["local_approval"]["text"].encode("utf8"),
                         self.paths.local_approval(self.prefix()).read_bytes())
        self.assertIn(C.load_prereg(self.paths).sha256, body["statement"])
        self.assertIn("not authenticated", stored["local_mode_caveat"])
        rec = json.loads(self.paths.import_record(self.prefix()).read_text())
        self.assertEqual((rec["commit"], rec["author"], rec["approval_file_sha256"]), (commit, self.USER, local_sha))
        self.assertEqual(rec["runs_side_approval_sha256"], C.file_sha256(self.paths.approval))
        self.assertTrue(all(q[0] in ("rev-parse", "cat-file", "log", "merge-base", "config") for q in self.git.calls))
        self.refuses("exists already")                                            # a second import is refused

    def test_import_refusals(self):
        def fresh():
            self.tearDown()
            self.setUp()
        cases = {
            "wrong characters": dict(typed="0" * 16),
            "another mode": dict(mode="terminal"),
            "another version": dict(prereg_sha256="d94ac6b835a6d7ee5fd8a18024af278d909ebb21687e2e884029886c931bf98f",
                                    typed="d94ac6b835a6d7ee",
                                    statement="covers d94ac6b835a6d7ee5fd8a18024af278d909ebb21687e2e884029886c931bf98f"),
            "not a terminal": dict(stdin_isatty=False),
            "another kind": dict(kind="stage2-approval"),
            "another file": dict(prereg_path="runs/shift_explore/stage2/prereg.json"),
            "no statement": dict(statement="I approve."),
            "another manifest": dict(code_manifest_sha256="0" * 64),
            "another approve_local.py": dict(approve_local_sha256="0" * 64),
        }
        for why, change in cases.items():
            with self.subTest(case=why):
                fresh()
                self.user_push(**change)
                self.refuses()
                self.assertFalse(self.paths.approval.exists())
        git_cases = {
            "not pushed": dict(push=False),
            "the author is the session": dict(author="Claude <noreply@anthropic.com>"),
            "the author is the session, other case": dict(author="Someone <NoReply@Anthropic.com>"),
            "the committer is the session": dict(committer="Claude <noreply@anthropic.com>"),
            "the session's configured email": dict(author="Agent <agent@session.example>"),
        }
        for why, kw in git_cases.items():
            with self.subTest(case=why):
                fresh()
                self.git.email = "agent@session.example"
                self.user_push(**kw)
                self.refuses()
                self.assertFalse(self.paths.approval.exists())
        f = lambda: self.paths.local_approval(self.prefix())                      # noqa: E731
        with self.subTest(case="no file in the folder"):
            fresh()
            self.refuses("pull the user's commit first")
        with self.subTest(case="a file nobody committed"):
            fresh()
            self.local_approve()
            self.refuses("exactly one commit")
        with self.subTest(case="the file changed after the commit, sidecar redone"):
            fresh()
            self.user_push()
            text = f().read_bytes().replace(b'"python"', b'"python_"')
            f().write_bytes(text)
            f().with_name(f().name + ".sha256").write_text(hashlib.sha256(text).hexdigest() + "  x\n")
            self.refuses("SHA-256 differ")
        with self.subTest(case="the file changed without its sidecar"):
            fresh()
            self.user_push()
            f().write_bytes(f().read_bytes() + b" ")
            self.refuses("sidecar")
        with self.subTest(case="deleted and added again: two adding commits"):
            fresh()
            self.user_push()
            self.git.commit(deleted=[self.rel(f().name)])
            self.git.commit({self.rel(f().name): f().read_bytes()})
            self.refuses("exactly one commit")
        with self.subTest(case="origin's branch no longer holds the file"):
            fresh()
            self.user_push()
            self.git.commit(deleted=[self.rel(f().name)])
            self.refuses("no longer holds")
        with self.subTest(case="the user's commit holds other files than the current ones"):
            fresh()
            stale = self.git.commit({self.rel("code/a.py"): b"x = 0\n"})          # e.g. approved from an older checkout
            self.user_push()
            self.refuses("does not hold the current pre-registration's files")
            self.assertTrue(stale)
        with self.subTest(case="the committed folder here changed"):
            fresh()
            self.user_push()
            (self.paths.repo / "code" / "b.py").write_text("y = 5\n")
            self.refuses("committed folder")
        with self.subTest(case="the runs side moved on"):
            fresh()
            self.user_push()
            self.write_prereg(chat=False, cloud=False, extra="a later version")
            self.refuses()
        with self.subTest(case="another job's pin moved"):
            fresh()
            self.user_push()
            self.refuses("differs now", lambda: self.imp(
                pins_fn=lambda: (dict(FAKE_PINS, detect_manifest_sha256="c" * 64), [])))
        with self.subTest(case="everything in place"):
            fresh()
            self.user_push()
            self.imp()
            self.push_record()
            self.assertEqual(C.require_approval(self.paths).mode, C.LOCAL_MODE)

    def test_import_refused_when_mode_not_allowed(self):
        self.write_prereg(local=False, chat=False, cloud=False)
        self.push_folder()
        self.assertRaises(AL.Refused, self.local_approve)                         # refused on the user's machine
        repo = self.paths.repo
        prereg = C.load_prereg(self.paths)
        info = dict(prereg=prereg.payload, sha256=prereg.sha256, manifest_sha256=C.file_sha256(self.paths.manifest),
                    script_sha256=C.file_sha256(repo / "approve_local.py"))
        saved, AL.git = AL.git, (lambda *a: None)
        try:
            self.write_local(AL.approval_body(info, prereg.prefix))               # a file made by hand anyway
        finally:
            AL.git = saved
        self.refuses("does not accept approval mode 'local-terminal'")            # and refused at import

    def test_new_version_is_approved_with_a_new_file(self):
        """A pin moved after the user approved: the new version gets its own file; the old one stays and blocks
        nothing; each file has exactly one adding commit."""
        self.user_push()
        old_prefix = self.prefix()
        old_file = self.paths.local_approval(old_prefix)
        self.write_prereg(chat=False, cloud=False, extra="version written after a pin moved")
        self.push_folder()
        self.assertNotEqual(self.prefix(), old_prefix)
        self.refuses("no local approval of this fingerprint")                     # the old approval does not count
        AL.check_folder(str(self.paths.repo), str(self.paths.repo / "approve_local.py"))   # not blocked by the old file
        self.user_push()
        self.assertTrue(old_file.exists())
        self.imp()
        self.push_record()
        self.assertEqual(C.require_approval(self.paths).mode, C.LOCAL_MODE)

    def test_imported_approval_voided(self):
        self.user_push()
        self.imp()
        self.push_record()
        C.require_approval(self.paths)
        f = self.paths.local_approval(self.prefix())
        rec = self.paths.import_record(self.prefix())
        saved = f.read_bytes()
        f.unlink()                                                                # the committed file removed
        self.refuses(fn=lambda: C.require_approval(self.paths))
        f.write_bytes(saved + b" ")                                               # or changed
        self.refuses(fn=lambda: C.require_approval(self.paths))
        f.write_bytes(saved)
        C.require_approval(self.paths)
        saved_rec = rec.read_bytes()
        rec.write_bytes(saved_rec.replace(b"synthetic", b"Synthetic"))           # the import record edited here
        self.refuses(fn=lambda: C.require_approval(self.paths))
        rec.write_bytes(saved_rec)
        tip = self.git.refs[self.git.ref]
        self.git.commit(deleted=[self.rel(rec.name)])                             # the record removed on GitHub
        self.refuses("not on origin's branch", lambda: C.require_approval(self.paths))
        self.git.refs[self.git.ref] = tip
        self.git.commit(deleted=[self.rel(f.name)])                               # the approval removed on GitHub
        self.refuses("no longer holds", lambda: C.require_approval(self.paths))
        self.git.refs[self.git.ref] = tip
        C.require_approval(self.paths)
        original = self.paths.approval.read_bytes()
        body = json.loads(original)
        body["local_approval"]["text"] = body["local_approval"]["text"].replace('"mode": "local-terminal"',
                                                                                '"mode": "chat"')
        C.write_json(self.paths.approval, body, self.paths, overwrite=True)       # stored copy edited, sidecar redone
        self.refuses(fn=lambda: C.require_approval(self.paths))
        body["local_approval"]["sha256"] = hashlib.sha256(body["local_approval"]["text"].encode()).hexdigest()
        C.write_json(self.paths.approval, body, self.paths, overwrite=True)       # ... and its SHA-256 too
        self.refuses(fn=lambda: C.require_approval(self.paths))
        for key, value in (("author", "Claude <noreply@anthropic.com>"), ("author", "Other User <other@example.org>"),
                           ("committer", "Claude <noreply@anthropic.com>"), ("commit", "ab" * 20)):
            with self.subTest(edited=key):
                body = json.loads(original)
                body["git"][key] = value
                C.write_json(self.paths.approval, body, self.paths, overwrite=True)   # the git record edited
                self.refuses(fn=lambda: C.require_approval(self.paths))
        body = json.loads(original)
        body["note"] = "another runs-side approval, consistent in itself"
        C.write_json(self.paths.approval, body, self.paths, overwrite=True)       # not the file the record names
        self.refuses("does not match this approval", lambda: C.require_approval(self.paths))
        C.write_json(self.paths.approval, json.loads(original), self.paths, overwrite=True)
        self.assertEqual(self.paths.approval.read_bytes(), original)
        C.require_approval(self.paths)
        self.write_prereg(chat=False, cloud=False, extra="edited after approval")  # prereg.json edited
        self.refuses(fn=lambda: C.require_approval(self.paths))

    def test_hand_written_runs_side_approval(self):
        """What the code can and cannot stop: a runs-side file written by hand in local-terminal mode, for a commit
        git does not hold, is refused; one that copies a genuine import passes only with the user's real commit and the
        pushed record that names it, so it is visible on GitHub (pre-registration section 9)."""
        self.user_push()
        prereg = C.load_prereg(self.paths)
        f = self.paths.local_approval(prereg.prefix)
        text = f.read_text()
        fake = dict(kind="stage2-approval", label=C.LABEL, mode=C.LOCAL_MODE, prereg_path=str(self.paths.prereg),
                    prereg_sha256=prereg.sha256, typed=prereg.prefix, stdin_isatty=True,
                    local_approval=dict(path=C.local_approval_rel(prereg.prefix), sha256=C.file_sha256(f), text=text),
                    git=dict(commit="de" * 20, author=self.USER, committer=self.USER,
                             committed_sha256=C.file_sha256(f)))
        C.write_json(self.paths.approval, fake, self.paths)
        self.refuses("is not a commit in this repository", lambda: C.require_approval(self.paths))

    def test_show_says_whether_approvable_now(self):
        out = []
        prereg, ok = AP.show(self.paths, out.append, pins_fn=fake_pins)
        self.assertTrue(ok)
        self.assertIn("Approvable now: yes", out[-1])
        self.assertIn(prereg.prefix, out[-1])
        self.assertIn("Accepted approval modes: local-terminal\n", out[-1])
        moved = lambda: (dict(FAKE_PINS, detect_manifest_sha256="c" * 64), [])   # noqa: E731
        _, ok = AP.show(self.paths, out.append, pins_fn=moved)
        self.assertFalse(ok)
        self.assertIn("Approvable now: NO", out[-1])
        self.assertIn("detect_manifest_sha256", out[-1])
        (self.code / "a.py").write_text("x = 3\n")
        _, ok = AP.show(self.paths, out.append, pins_fn=fake_pins)
        self.assertFalse(ok)
        self.assertIn("a.py", out[-1])


class EntryPointRefusalTests(unittest.TestCase):
    """Every BCA entry point refuses without a valid approval of the REAL pre-registration, before any data access
    (the arguments other than the approval are None, so any data access would raise a different error)."""

    def test_entry_points_refuse(self):
        import policy as PO
        import deploy_check as DCK
        import host as HO
        import run_stage2 as RS
        forged = C.Approval(path=str(C.PATHS.approval), file_sha256="0" * 64, prereg_path=str(C.PATHS.prereg),
                            prereg_sha256="0" * 64, mode="terminal")
        calls = {
            "band.fit_on_context": lambda: B.fit_on_context(None, "stall", "fixed", forged, None),
            "band.fit_weighted_on_context": lambda: B.fit_weighted_on_context(None, "stall", "fixed", forged, None,
                                                                              None),
            "band.load_band": lambda: B.load_band(None, forged, 13, 4),
            "band.WidthReader": lambda: B.WidthReader(None, None, None, None, forged, 13),
            "band.weighted_refitter": lambda: B.weighted_refitter(None, "stall", None, None, None, None, forged),
            "policy.train": lambda: PO.train(0, forged, "uniform", None, None),
            "policy.evaluate": lambda: PO.evaluate(0, forged),
            "policy.train_no_stall": lambda: PO.train_no_stall(0, forged, "uniform"),
            "deploy_check.fit_tau": lambda: DCK.fit_tau(None, "stall", None, forged, None),
            "deploy_check.run": lambda: DCK.run(None, "stall", None, None, forged, None),
            "host.bca_host_run": lambda: HO.bca_host_run(None, None, 0, forged),
            "host.train": lambda: HO.train(0, "stall", forged, "uniform"),
            "host.evaluate": lambda: HO.evaluate(0, "stall", forged),
            "run_stage2.bca_gate": lambda: RS.bca_gate(),
        }
        for name, call in calls.items():
            with self.subTest(entry=name):
                with self.assertRaises((C.ApprovalError, SystemExit)) as cm:
                    call()
                self.assertIn("refused", str(cm.exception))


# --------------------------------------------------------------------------------------------- small rules

class RuleTests(unittest.TestCase):
    def test_guard_out(self):
        TMP_ROOT.mkdir(parents=True, exist_ok=True)
        tmp = Path(tempfile.mkdtemp(dir=TMP_ROOT))
        try:
            p = C.Paths(tmp)
            self.assertRaises(SystemExit, C.guard_out, tmp.parent / "elsewhere.json", p)
            for bad in (C.ROOT / "runs" / "shift" / "x.json", C.ROOT / "experiments" / "x.json",
                        C.ROOM_OUT / "x.json", C.STANDINS_SRC / "x.json", C.DETECT_SRC / "x.json"):
                self.assertRaises(SystemExit, C.guard_out, bad, C.Paths(bad.parent))
            C.write_json(tmp / "a.json", dict(x=1), p)
            self.assertRaises(SystemExit, C.write_json, tmp / "a.json", dict(x=2), p)   # written once
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    def test_best_standin_rule(self):
        import run_stage2 as RS

        def m(cap, gain):
            return dict(captures_the_room=cap, gain_strong_current=dict(mean=gain))

        marks = dict(range_hinge=m(True, 80.0), density=m(True, 85.0), classifier_ratio=m(False, 99.0),
                     distance=m(False, None), two_head=m(False, 10.0))
        self.assertEqual(RS.select_best_standin(marks)[0], "density")      # best among those that capture
        marks["density"] = m(False, 85.0)
        marks["range_hinge"] = m(False, 80.0)
        self.assertEqual(RS.select_best_standin(marks)[0], "classifier_ratio")   # none captures: best gain
        marks = {k: m(False, None) for k in RS.MAIN_STANDINS}
        self.assertIsNone(RS.select_best_standin(marks)[0])
        literal = ("range_hinge", "density_plan", "classifier_ratio", "distance_plan", "two_head_plan")
        marks = dict(range_hinge=m(False, 70.0), density=m(True, 99.0), density_plan=m(True, 60.0),
                     classifier_ratio=m(False, 90.0), distance=m(True, 98.0), distance_plan=m(False, 95.0),
                     two_head=m(True, 97.0), two_head_plan=m(False, 10.0))
        self.assertEqual(RS.select_best_standin(marks, literal)[0], "density_plan")   # the user's forms only
        self.assertEqual(RS.select_best_standin(marks)[0], "density")

    GOOD = dict(widening_T1_harmful=0.9, widening_T3_harmful=0.8, size_T1_paired=1.4, size_T1_raw=1.6,
                ranking_T12=0.85, ranking_T3=0.8, rate_above_level_T4=0.05, catch_rate_T3_past80=0.7,
                flag_rate_past80=0.6, n_harmful_T1=100, n_harmful_T3=200)

    def standins(self, **best):
        weak = dict(self.GOOD, ranking_T12=0.6, ranking_T3=0.6)
        st = {n: [weak] * 5 for n in DC.STANDIN_NAMES}
        st["distance"] = [dict(self.GOOD, **best)] * 5
        return st

    def test_detection_marks(self):
        good = self.GOOD
        st = self.standins(ranking_T12=0.86, ranking_T3=0.78)
        out = DC.detection_marks([good] * 5, st, group="confirmatory")
        self.assertTrue(out["claim"] and out["claim_rule_met"] and not out["direction_or_size_failed"])
        self.assertEqual(out["best_standin"]["ranking_T12"]["name"], "distance")
        st = self.standins(ranking_T12=0.9, ranking_T3=0.78)                      # BCA 0.05 below the best
        self.assertFalse(DC.detection_marks([good] * 5, st, group="confirmatory")["claim"])
        weak = [good] * 3 + [dict(good, size_T1_paired=0.5)] * 2                   # size in only 3 of 5 seeds
        out = DC.detection_marks(weak, self.standins(), group="confirmatory")
        self.assertTrue(out["direction_or_size_failed"] and not out["claim"])
        out = DC.detection_marks([good] * 4 + [None], self.standins())              # a missing seed fails
        self.assertFalse(out["passed"])
        self.assertTrue(out["direction_or_size_failed"])

    def test_detection_marks_need_every_standin_on_every_seed(self):
        """The reviewer's case: BCA 0.80 on every seed, every stand-in 0.97 on 4 seeds and missing on 1. The rivals
        are held to BCA's standard: an incomplete comparison fails the 'not below the best stand-in' part."""
        good = dict(self.GOOD, ranking_T12=0.80, ranking_T3=0.80)
        strong = dict(self.GOOD, ranking_T12=0.97, ranking_T3=0.97)
        cases = {
            "one seed missing": {n: [strong] * 4 + [None] for n in DC.STANDIN_NAMES},
            "no stand-in at all": {},
            "four seeds only": {n: [strong] * 4 for n in DC.STANDIN_NAMES},
            "one stand-in absent": {n: [strong] * 5 for n in DC.STANDIN_NAMES[1:]},
            "a non-finite value": {n: [strong] * 4 + [dict(strong, ranking_T3=float("nan"))]
                                   for n in DC.STANDIN_NAMES},
            "a None value": {n: [strong] * 4 + [dict(strong, ranking_T12=None)] for n in DC.STANDIN_NAMES},
        }
        for name, st in cases.items():
            with self.subTest(case=name):
                out = DC.detection_marks([good] * 5, st, group="confirmatory")
                self.assertFalse(out["not_below_best_standin"]["passed"])
                self.assertFalse(out["not_below_best_standin"]["complete"])
                self.assertFalse(out["claim"])
                self.assertTrue(out["direction"]["passed"] and out["size"]["passed"] and out["ranking"]["passed"])
        out = DC.detection_marks([good] * 5, {n: [strong] * 5 for n in DC.STANDIN_NAMES}, group="confirmatory")
        self.assertTrue(out["not_below_best_standin"]["complete"] and not out["claim"])     # complete: 0.80 < 0.95
        st = {n: [dict(self.GOOD, ranking_T12=0.79, ranking_T3=0.79)] * 5 for n in DC.STANDIN_NAMES}
        self.assertTrue(DC.detection_marks([good] * 5, st, group="confirmatory")["claim"])  # within 0.02

    def test_exploratory_detection_marks_carry_no_claim(self):
        st = self.standins(ranking_T12=0.5, ranking_T3=0.5)
        out = DC.detection_marks([self.GOOD] * 5, st, group="exploratory")
        self.assertTrue(out["claim_rule_met"])
        self.assertIsNone(out["claim"])
        self.assertTrue(out["claim_text"].startswith("exploratory, not a claim"))
        self.assertNotIn("BCA flags", out["claim_text"])
        out = DC.detection_marks([self.GOOD] * 4 + [None], st, group="exploratory")
        self.assertIsNone(out["claim"])

    def test_claim_fields(self):
        import run_stage2 as RS
        f = RS.claim_fields("exploratory", True, True)
        self.assertIsNone(f["flags_claim"])
        self.assertIsNone(f["detection_claim"])
        self.assertTrue(f["marks_met"])
        self.assertTrue(f["flags_claim_text"].startswith("exploratory, not a claim"))
        self.assertNotIn("repairs the harm", f["flags_claim_text"])
        f = RS.claim_fields("confirmatory", True, True)
        self.assertTrue(f["flags_claim"])
        self.assertIn("repairs the harm", f["flags_claim_text"])
        self.assertFalse(RS.claim_fields("confirmatory", True, False)["flags_claim"])

    def test_measures_nested_and_component_pins(self):
        flat = {k: 0.5 for k in DC.MEASURE_KEYS}
        self.assertEqual(DC.check_measures(dict(measures=flat, joined={}))["ranking_T12"], 0.5)
        self.assertRaises(SystemExit, DC.check_measures, dict(measures={"ranking_T12": 0.5}))
        TMP_ROOT.mkdir(parents=True, exist_ok=True)
        tmp = Path(tempfile.mkdtemp(dir=TMP_ROOT))
        try:
            (tmp / "code").mkdir()
            (tmp / "code" / "m.py").write_text("z = 1\n")
            p = C.Paths(tmp)
            C.write_json(tmp / "code_manifest.json", dict(files={"m.py": C.file_sha256(tmp / "code" / "m.py")}), p)
            sha = C.file_sha256(tmp / "code_manifest.json")
            C.verify_component(tmp, sha)
            self.assertRaises(SystemExit, C.verify_component, tmp, "0" * 64)       # another manifest
            self.assertRaises(SystemExit, C.verify_component, tmp, None)           # nothing pinned
            (tmp / "code" / "m.py").write_text("z = 2\n")
            self.assertRaises(SystemExit, C.verify_component, tmp, sha)            # a changed file
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    def test_claim_sentence(self):
        import policy as PO
        s = PO.claim_sentence(dict(harm=98.9, X=80.0, Y=0.9, Z=95.0, W=20.0, V=60.0))
        self.assertIn("recovers 80.0 of 98.9 points", s)
        self.assertIn("keeps 90%", s)


class WriteOnceInputTests(unittest.TestCase):
    """The write-once phases' input rules, on made-up record files in a temporary folder (no data, no BCA)."""

    def setUp(self):
        import run_stage2 as RS
        self.RS = RS
        TMP_ROOT.mkdir(parents=True, exist_ok=True)
        self.tmp = Path(tempfile.mkdtemp(dir=TMP_ROOT))
        self.paths = C.Paths(self.tmp / "out", code_src=HERE)
        self.paths.out.mkdir()

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def put(self, path, body=None):
        C.write_json(path, dict(body or {}), self.paths, overwrite=True)

    def complete(self, rep, world, critic, version="uniform", tau=True):
        RS = self.RS
        d = RS.band_dir(rep, world, critic, version, self.paths)
        self.put(d / "band.json", dict(failed=False))
        self.put(d / "scores_record.json")
        if tau:
            self.put(d / "tau.json", dict(tau=1.0))
        sw = RS.seed_world_dir(rep, world, RS.pairs_root(rep, self.paths))
        self.put(sw / "labels.json")
        if world == "stall":
            for n in DC.STANDIN_NAMES:
                self.put(sw / "scores" / f"{n}.json")

    def all_reads(self, rep, tau=True):
        for world, critic, v in self.RS.DETECTION_READS:
            self.complete(rep, world, critic, v or "uniform", tau=tau)

    def test_detection_marks_refuse_incomplete_inputs_and_accept_recorded_failures(self):
        import fresh as F
        RS = self.RS
        # seed 5: its data checks failed (on record); seed 6: every band fit failed (on record); 7 and 9 complete;
        # seed 8: tau not written yet
        self.put(F.fresh_dir(5, "no_stall", "data.json", paths=self.paths), dict(passed=False))
        for world, critic, v in RS.DETECTION_READS:
            self.put(RS.band_dir(6, world, critic, v or "uniform", self.paths) / "band.json",
                     dict(failed=True, reason="not ready"))
        self.all_reads(7)
        self.all_reads(9)
        self.all_reads(8, tau=False)
        with self.assertRaises(SystemExit) as cm:
            RS.detection_completeness("confirmatory", "uniform", self.paths)
        self.assertIn("seed 8", str(cm.exception))
        self.assertNotIn("seed 5", str(cm.exception))
        self.assertNotIn("seed 6", str(cm.exception))
        self.all_reads(8)
        (RS.seed_world_dir(9, "stall", RS.pairs_root(9, self.paths)) / "scores" / "two_head.json").unlink()
        with self.assertRaises(SystemExit) as cm:                     # a stand-in's scores not written yet
            RS.detection_completeness("confirmatory", "uniform", self.paths)
        self.assertIn("two_head", str(cm.exception))
        self.complete(9, "stall", "fixed")
        st = RS.detection_completeness("confirmatory", "uniform", self.paths)
        self.assertEqual(st[(5, "stall", "fixed")][0], "failed")
        self.assertEqual(st[(6, "no_stall", "fixed")][0], "failed")
        self.assertEqual(st[(7, "stall", "host")][0], "complete")
        self.assertEqual(st[(8, "stall", "fixed")][0], "complete")
        # the weighted version: the host critic is read as written (uniform); a failed uniform fit fails it too
        for rep in (7, 8, 9):
            for world in ("stall", "no_stall"):
                self.complete(rep, world, "fixed", "weighted")
        st = RS.detection_completeness("confirmatory", "weighted", self.paths)
        self.assertEqual(st[(6, "stall", "fixed")][0], "failed")
        self.assertEqual(st[(7, "stall", "host")][1][0].parent.parent.parent.parent.name, "uniform")

    def test_deploy_waits_for_the_version_decision(self):
        RS = self.RS
        self.assertRaises(SystemExit, RS.deploy_version, self.paths)            # no decision yet
        d = RS.detect_root(self.paths)
        self.put(d / "marks_exploratory_uniform.json", dict(marks=dict(direction_or_size_failed=False)))
        self.assertEqual(RS.deploy_version(self.paths), "uniform")
        self.put(d / "marks_exploratory_uniform.json", dict(marks=dict(direction_or_size_failed=True)))
        self.assertRaises(SystemExit, RS.deploy_version, self.paths)            # weighted declared, not yet run
        self.put(d / "marks_exploratory_weighted.json", dict(marks=dict(direction_or_size_failed=False)))
        self.assertEqual(RS.deploy_version(self.paths), "weighted")
        self.put(d / "marks_exploratory_weighted.json", dict(marks=dict(direction_or_size_failed=True)))
        self.assertEqual(RS.deploy_version(self.paths), "uniform")              # neither passed: uniform

    def test_recorded_fresh_failure_is_a_stopped_seed(self):
        import fresh as F
        import policy as PO
        self.assertIsNone(F.recorded_failure(5, self.paths))
        self.assertIsNone(F.recorded_failure(0, self.paths))                     # gate seeds never
        self.put(F.fresh_dir(5, "stall", "data.json", paths=self.paths), dict(passed=True))
        self.put(F.fresh_dir(5, "no_stall", "data.json", paths=self.paths), dict(passed=False))
        self.assertIn("data checks", F.recorded_failure(5, self.paths))
        self.put(F.fresh_dir(6, "stall", "learners.json", paths=self.paths),
                 dict(failed=True, reason="fresh learner clone of seed 6 failed"))
        self.assertIn("clone", F.recorded_failure(6, self.paths))
        self.put(F.fresh_dir(7, "stall", "learners.json", paths=self.paths), dict(failed=False))
        self.assertIsNone(F.recorded_failure(7, self.paths))
        ap = C.Approval(path="x", file_sha256="0" * 64, prereg_path="y", prereg_sha256="1" * 64, mode="chat")
        self.assertEqual(PO.load_values(5, "stall", ap, self.paths)[0], "stopped")
        self.assertEqual(PO.load_values(6, "no_stall", ap, self.paths)[0], "stopped")
        self.assertEqual(PO.load_values(7, "stall", ap, self.paths)[0], "missing")
        self.assertTrue(self.RS.recorded_seed_failure(5, "stall", "fixed", "uniform", self.paths)
                        .startswith("fresh seed 5"))

    def test_gate_seed_pair_sets_are_never_built_in_the_rehearsal_folder(self):
        RS = self.RS
        calls = []

        class FakeDetect:
            @staticmethod
            def pair_sets(ctx, world, out):
                calls.append(Path(out))
                return dict(set=np.array(["T4"]), log_row=np.array([3]))

        rehearsal = self.tmp / "rehearsal"                            # a stand-in for runs/shift_explore/detect
        with self.assertRaises(SystemExit) as cm:
            RS.locked_pair_sets(FakeDetect, object(), "stall", 0, self.paths, rehearsal)
        self.assertIn("never builds", str(cm.exception))
        self.assertEqual(calls, [])                                   # pair_sets (which would build) never called
        self.assertFalse(rehearsal.exists())
        (rehearsal / "seed0" / "stall").mkdir(parents=True)
        (rehearsal / "seed0" / "stall" / RS.PAIRS_RECORD).write_text("{}")
        RS.locked_pair_sets(FakeDetect, object(), "stall", 0, self.paths, rehearsal)    # exists: load only
        self.assertEqual(calls, [rehearsal])
        RS.locked_pair_sets(FakeDetect, object(), "stall", 5, self.paths, rehearsal)    # fresh: Stage 2's folder
        self.assertEqual(calls[-1], self.paths.out / "detect_fresh")


class DeclaredBehaviourTests(unittest.TestCase):
    """Synthetic: the declared gradient paths of the weighted width and the inert uniform 'cannot certify'."""

    def setUp(self):
        rng = np.random.default_rng(12)
        self.obs = jnp.asarray(rng.normal(size=(40, 13)), jnp.float32)
        self.a = jnp.asarray(rng.uniform(-1, 1, size=(40, 4)), jnp.float32)

    def test_weighted_gradient_flows_through_the_classifier_and_threshold(self):
        cal, band = synthetic_band()
        cls_p, _ = Wt.cls_init(1, 13, 4)
        wb = band._replace(log_t=jnp.linspace(-3, 3, B.GRID).astype(jnp.float32),
                           R_tab=jnp.linspace(1, 3, B.GRID).astype(jnp.float32), l_star=B.f32(1.0),
                           cls_params=cls_p, U_max=B.f32(2.0))
        e = lambda a: jnp.maximum(cal.apply(band.cal_params, self.obs, a), 1e-6)            # noqa: E731
        lw = lambda a: Wt.logit(cls_p, self.obs, a)                                         # noqa: E731
        threshold_only = jax.grad(lambda a: (jnp.interp(lw(a), wb.log_t, wb.R_tab) * wb.unit
                                             * jax.lax.stop_gradient(e(a))).sum())(self.a)
        refusal_only = jax.grad(lambda a: (wb.U_max * jax.nn.softplus(lw(a) - wb.l_star)).sum())(self.a)
        full = jax.grad(lambda a: B.width_weighted(cal, wb, self.obs, a).sum())(self.a)
        self.assertGreater(float(jnp.abs(threshold_only).mean()), 0.0)       # through R_c(w(s, a)): declared
        self.assertGreater(float(jnp.abs(refusal_only).mean()), 0.0)         # through the refusal term: declared
        def eta_path_only(a):
            lw0 = jax.lax.stop_gradient(lw(a))
            return (jnp.interp(lw0, wb.log_t, wb.R_tab) * wb.unit * e(a)
                    + wb.U_max * jax.nn.softplus(lw0 - wb.l_star)).sum()

        no_cls = jax.grad(eta_path_only)(self.a)
        self.assertGreater(float(jnp.abs(full - no_cls).max()), 1e-4)        # the classifier path is live
        np.testing.assert_allclose(np.asarray(full), np.asarray(no_cls + threshold_only + refusal_only),
                                   rtol=1e-4, atol=1e-6)                     # exactly these three paths
        grads = jax.grad(lambda p: B.width_weighted(cal, wb._replace(cls_params=p), self.obs, self.a).sum())(cls_p)
        self.assertEqual(max(float(jnp.abs(x).max()) for x in jax.tree_util.tree_leaves(grads)), 0.0)

    def test_uncertified_uniform_is_inert(self):
        import deploy_check as DCK
        cal, band = synthetic_band(certified=False)
        u_rec = B.width_uniform(cal, band, self.obs, self.a)
        pen = lambda a: B.paired(B.width_uniform(cal, band, self.obs, a), u_rec).sum()      # noqa: E731
        self.assertEqual(float(pen(self.a + 0.3)), 0.0)
        self.assertEqual(float(jnp.abs(jax.grad(pen)(self.a + 0.3)).max()), 0.0)
        unp = jax.grad(lambda a: B.width_uniform(cal, band, self.obs, a).sum())(self.a)
        self.assertEqual(float(jnp.abs(unp).max()), 0.0)
        tau = DCK.conformal_level(np.asarray(u_rec, np.float64))
        self.assertEqual(tau, 5.0)                                            # U_max
        u_pi = np.asarray(B.width_uniform(cal, band, self.obs, self.a + 0.5), np.float64)
        self.assertEqual(int(np.sum(~(u_pi <= tau))), 0)                      # the check never fires (declared)


if __name__ == "__main__":
    unittest.main(verbosity=2)

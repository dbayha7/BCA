"""Stage-0 tests of the restated Unifloral TD3+BC step, unifloral_td3_bc_bca.py (plan section 4.5, TD3+BC's rows), on
made-up arrays only (tests/madeup.py, registered with the guard; approval item 8): never ThrusterChain data, and no
learned policy's return is computed. An experiment-local attachment (plan section 3.1) that changes none of the
repository's declared methods.

  source   make_train_step is td3_bc.py:174-279 line for line: without its "# BCA:" lines it is Unifloral's text minus
           the two changed lines (174, 245); _own_target's _compute_target is td3_bc.py:191-208 verbatim.
  (a1)     BCA absent = the imported make_train_step: 60 outer steps from Unifloral's own initializer, bit for bit
           (every leaf of the learner state and its optimizer states, the key carry and the per-step losses).
  (a2)     BCA present before any refresh (dose exactly 1), standard and IW setups: the same 60 steps, bit for bit on
           the processor; on the card bit for bit or a recorded rounding-level difference (every leaf within 1e-6
           relative); anything larger fails. BCA really ran: 60 accepted fits moved the scale network and the unit.
  (a3)     a fixed made-up dose in [1, 1.5) (madeup.dose): the step's actor loss and its gradient equal plan 4.5's
           formula written outside the step, -lambda mean Q1 + mean(m bc) (td3_bc.py:239, 243-245), within relative
           1e-5, while a dose of 1 or the dose on the Q term misses by far more; the dose changes nothing else (key,
           critics, target critics, the other losses). Extra cross-check with the imported file (the plan names none
           for TD3+BC): every dose 1.5 gives 1.5 x Unifloral's own actor loss and lambda at td3_alpha = 2.5 / 1.5.
  (a4)     with Unifloral's own key the band's y equals the first critic update's targets, and q the minimum over the
           two critics of its q_pred, bit for bit (processor); with BCA's own keys q is unchanged and y is Unifloral's
           target at BCA's draw split(fold_in(fold_in(PRNGKey(s91), t), 1179210836), 256).
  (b2)     thresholds at 4 levels: batch row b's dose equals the dose computed outside the step at R[batch_indices[b]],
           and BCA's own frozen_level_dose at that level; a lookup shifted by one row would not match.
  (b3)     R = +inf on every third row: standard BCA gives those rows dose 1 and IW BCA 1.5; every other row is the
           same under both; before the first refresh nothing is refused.
  refresh  refresh()'s bank scores: recomputed from the stored states (recompute_bank_scores, the same compiled
           program) bit for bit; equal to a direct computation written independently to float32 rounding (relative
           1e-5; another refresh key's are not); an unweighted calibrate on them and the posterior stream reproduces
           the stored threshold bit for bit (stop rule S3).
  guards   the factory and the refresh refuse unregistered arrays, a missing permission, another learner's setup and
           test mode without BCA; a step whose BCA inputs are not valid is discarded whole, random key included
           (algorithms/td3_bc_bca.py:340-342), and require_valid stops on it.

On the processor every test runs (SHIFT_HOSTS_DEVICE=cpu JAX_PLATFORMS=cpu). On the card (SHIFT_HOSTS_DEVICE=gpu
XLA_PYTHON_CLIENT_PREALLOCATE=false, no JAX_PLATFORMS) the source test, (a1), (a2) and (a3) run; (a4), (b2), (b3), the
refresh and the guards are processor tests and are skipped there. From /root/bca-hosts in /opt/bca-venv with the
Dockerfile's variables (INTERFACES.md section 2):
  env D4RL_SUPPRESS_IMPORT_ERROR=1 MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa DISPLAY= \
      MUJOCO_PY_MUJOCO_PATH=/root/.mujoco/mujoco210 LD_LIBRARY_PATH=/root/.mujoco/mujoco210/bin \
      PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 SHIFT_HOSTS_DEVICE=cpu JAX_PLATFORMS=cpu \
      taskset -c 4-7 /opt/bca-venv/bin/python -m experiments.shift_hosts.tests.test_td3_bc
The result goes to runs/shift_hosts/stage0/td3bc/test_td3_bc_<device>.json with a sidecar; its "details" hold each
test's readouts (leaf counts, the largest differences, the card's rounding record for (a2)).
"""

import ast
import dataclasses
import inspect
import os
import sys
import textwrap
import unittest

import numpy as np

from experiments.shift_hosts import unifloral_import as UI

AREA = "td3bc"
LEARNER = "td3_bc"
STEPS = 60                  # outer steps of (a1) and (a2)
CARD_REL = 1e-6             # (a2) on the card: the largest rounding-level difference allowed, per leaf
F32_REL = 1e-5              # "to float32 rounding" (plan 4.5)
PROGRAM_REL = 1e-6          # the same float32 arithmetic compiled as two different programs (a few rounding units)
DETAILS = {}                # each test's readouts, written into the result file


def _device():
    return os.environ.get(UI.DEVICE_ENV) or "cpu"


ON_PROCESSOR = _device() == "cpu"
processor_only = unittest.skipUnless(ON_PROCESSOR, "a processor test (plan 4.5); the card runs source, a1, a2, a3")


# ------------------------------------------------------------------------------------------------ comparisons

def leaf_report(a, b):
    """Every leaf of two pytrees in jax.tree_util order: its path, dtype, shape, whether dtype, shape and bytes are
    equal, and the relative difference max|a - b| / max|b| (0 when the bytes are equal; inf for a different dtype or
    shape, a non-float leaf that differs, or a NaN); for a float leaf that differs, how many elements differ and the
    largest difference in units of the last place of b's elements (max_ulp)."""
    import jax

    la, ta = jax.tree_util.tree_flatten_with_path(a)
    lb, tb = jax.tree_util.tree_flatten_with_path(b)
    if ta != tb:
        raise AssertionError(f"the two trees differ in structure:\n{ta}\n{tb}")
    rows = []
    for (path, x), (_, y) in zip(la, lb):
        x, y = np.asarray(x), np.asarray(y)
        same = x.dtype == y.dtype and x.shape == y.shape and x.tobytes() == y.tobytes()
        rel, n_diff, max_ulp = 0.0, 0, 0.0
        if not same:
            rel = float("inf")
            if x.dtype == y.dtype and x.shape == y.shape and np.issubdtype(x.dtype, np.floating) and x.size:
                d = np.abs(x.astype(np.float64) - y.astype(np.float64))
                s = float(np.max(np.abs(y.astype(np.float64))))
                if np.all(np.isfinite(d)):
                    dm = float(np.max(d))
                    rel = dm / s if s > 0 else (0.0 if dm == 0 else float("inf"))
                    max_ulp = float(np.max(d / np.spacing(np.abs(y)).astype(np.float64)))
                n_diff = int(np.count_nonzero(x != y))
        rows.append(dict(path=jax.tree_util.keystr(path), dtype=str(x.dtype), shape=list(x.shape), equal=bool(same),
                         rel=rel, n_diff=n_diff, max_ulp=max_ulp, size=int(x.size),
                         integer=not np.issubdtype(x.dtype, np.floating)))
    return rows


def summary(rows):
    """A JSON-ready summary of leaf_report: counts, the largest relative difference and where it is, and each unequal
    leaf with its number of differing elements and largest difference in units of the last place."""
    worst = max(rows, key=lambda r: r["rel"]) if rows else None
    return dict(leaves=len(rows), equal=sum(r["equal"] for r in rows), bit_for_bit=all(r["equal"] for r in rows),
                max_rel=worst["rel"] if worst else 0.0, worst_leaf=worst["path"] if worst and worst["rel"] else None,
                unequal=[dict(path=r["path"], elements=r["size"], differing=r["n_diff"], max_ulp=r["max_ulp"],
                              rel=r["rel"]) for r in rows if not r["equal"]][:20])


def bits_equal(a, b):
    return all(r["equal"] for r in leaf_report(a, b))


def relative(a, b):
    """max|a - b| / max|b| in float64."""
    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    s = float(np.max(np.abs(b)))
    d = float(np.max(np.abs(a - b)))
    return d / s if s > 0 else (0.0 if d == 0 else float("inf"))


def tree_relative(a, b):
    """The largest per-leaf relative difference of two float pytrees."""
    import jax

    return max(relative(x, y) for x, y in zip(jax.tree_util.tree_leaves(a), jax.tree_util.tree_leaves(b)))


def first(tree):
    """The first outer step's slice of a scan's stacked outputs."""
    import jax

    return jax.tree_util.tree_map(lambda x: x[0], tree)


# ------------------------------------------------------------------------------------------------ fixture

class _Fixture:
    """Made-up arrays (madeup.build(0)), Unifloral's own initial state for TD3+BC (unifloral_import's restatement of the
    training block, td3_bc.py:285, 308-324) at the plan's settings (seed of data seed 0, gamma 0.98, 100,000 updates),
    and the step functions, each built once so its compiled scans are reused by every test class."""

    def __init__(self):
        UI.prepare(device=_device())
        from experiments.shift_hosts import bca_core as BC
        from experiments.shift_hosts import guard
        from experiments.shift_hosts import unifloral_td3_bc_bca as TB
        from experiments.shift_hosts.tests import madeup

        self.BC, self.guard, self.TB, self.U = BC, guard, TB, UI.load(LEARNER)
        self.m = madeup.build(0)
        self.seeds = BC.seeds(0)
        self.args = UI.args_for(LEARNER, seed=self.seeds.unifloral, gamma=0.98, num_updates=100_000)
        self.built = UI.init_like_training_block(LEARNER, self.args, self.m.rows("train"))
        self.base = BC.fit_key_base(self.seeds.s91)
        self.setup = BC.BCASetup(LEARNER, self.base, guard.MADE_UP)
        self.scale0 = BC.make_scale_state(LEARNER, self.built.obs_mean, self.built.obs_std,
                                          BC.scale_init_key(self.base, LEARNER))
        self.bank = UI.make_transition(LEARNER, self.m.rows("bank"))
        self.steps, self.cache = {}, {}

    def step(self, name):
        if name not in self.steps:
            BC, TB, b, a = self.BC, self.TB, self.built, self.args
            bca = dict(bca=self.setup)
            make = {
                "unifloral": lambda: UI.make_unifloral_step(b, a),
                "unifloral_alpha": lambda: UI.make_unifloral_step(
                    b, dataclasses.replace(a, td3_alpha=a.td3_alpha / 1.5)),
                "alone": lambda: TB.make_train_step(a, *b.apply_fns, b.dataset),
                "bca": lambda: TB.make_train_step(a, *b.apply_fns, b.dataset, **bca),
                "plain": lambda: TB.make_train_step(a, *b.apply_fns, b.dataset, **bca, test=BC.TestOptions()),
                "unifloral_keys": lambda: TB.make_train_step(a, *b.apply_fns, b.dataset, **bca,
                                                             test=BC.TestOptions(unifloral_keys=True)),
                "override": lambda: TB.make_train_step(a, *b.apply_fns, b.dataset, **bca,
                                                       test=BC.TestOptions(dose_override=self.m.dose)),
                "override_15": lambda: TB.make_train_step(
                    a, *b.apply_fns, b.dataset, **bca,
                    test=BC.TestOptions(dose_override=np.full(self.m.n, 1.5, np.float32))),
            }[name]
            self.steps[name] = make()
        return self.steps[name]

    def scan(self, name, carry, n):
        """n outer steps of a step function, scanned as Unifloral's training block scans it (td3_bc.py:332-337)."""
        import jax

        return jax.lax.scan(self.step(name), carry, None, length=n)

    def initial(self, most_caution=None):
        """Unifloral's initial carry (rng, agent_state); with most_caution (False standard, True IW) BCA's initial
        state appended (bca_core.initial_state: unit 1, reference not ready, every threshold +inf, step 0)."""
        c = (self.built.rng, self.built.agent_state)
        if most_caution is None:
            return c
        return c + (self.BC.initial_state(self.scale0, self.m.n, most_caution=most_caution),)

    def cached(self, key, make):
        if key not in self.cache:
            self.cache[key] = make()
        return self.cache[key]

    def unifloral60(self):
        return self.cached("unifloral60", lambda: self.scan("unifloral", self.initial(), STEPS))

    def bca60(self, most_caution):
        return self.cached(("bca60", most_caution), lambda: self.scan("bca", self.initial(most_caution), STEPS))

    def refreshed(self):
        """The standard carry after bca60 and a refresh on the made-up bank after outer step 60 (a ready reference
        with a finite threshold), and the refresh's output (bca_core.Refreshed)."""
        def make():
            (rng, st, bs), _ = self.bca60(False)
            out = self.TB.refresh(self.args, self.built.apply_fns, st, bs, self.bank,
                                  self.BC.refresh_key(self.seeds.s39, STEPS), permission=self.guard.MADE_UP)
            return (rng, st, out.state), out
        return self.cached("refreshed", make)

    def with_thresholds(self, carry, R, most_caution):
        import jax.numpy as jnp

        rng, st, bs = carry
        return rng, st, bs._replace(thresholds=jnp.asarray(R, jnp.float32),
                                    most_caution=jnp.asarray(bool(most_caution)))

    def rows(self, idx):
        import jax

        return jax.tree_util.tree_map(lambda x: x[np.asarray(idx)], self.built.dataset)


_FIXTURE = []


def fixture():
    if not _FIXTURE:
        _FIXTURE.append(_Fixture())
    return _FIXTURE[0]


def external_actor_loss(F, actor_params, critic_params, rows, m, alpha, place="bc"):
    """Plan 4.5's formula for TD3+BC, written outside the step and batched (no per-row vmap, no distrax sampling):
    pi = tanh of the actor's output (td3_bc.py:109-113, 236-237), Q1 = the first critic at (s, pi) (238, 240),
    bc_i = mean_j (pi_ij - a_ij)^2 (239), lambda = alpha / (mean |Q1| + 1e-7) with no gradient (243-244),
    loss = -lambda mean Q1 + mean_i(m_i bc_i) (245 with the dose). place="q" puts the dose on the Q term instead, a
    wrong placement for the control. Returns (loss, gradient with respect to the actor's parameters)."""
    import jax
    import jax.numpy as jnp

    actor_apply, q_apply = F.built.apply_fns
    m = jnp.asarray(m, jnp.float32)

    def loss(params):
        pi = jnp.tanh(actor_apply(params, rows.obs).distribution.loc)
        q1 = q_apply(critic_params, rows.obs, pi)[:, 0]
        bc = jnp.mean(jnp.square(pi - rows.action), axis=1)
        lam = jax.lax.stop_gradient(alpha / (jnp.mean(jnp.abs(q1)) + 1e-7))
        if place == "q":
            return -lam * jnp.mean(m * q1) + jnp.mean(bc)
        return -lam * jnp.mean(q1) + jnp.mean(m * bc)

    return jax.value_and_grad(loss)(actor_params)


def outside_dose(F, R, eta, unit):
    """The dose outside the step, by BCA's own width and dose (bca_core.width, bca_core.dose = calibration/dose.py's
    level_critic_dose; blend 0.5), compiled on its own."""
    import jax

    BC = F.BC
    return np.asarray(jax.jit(lambda r, e, u: BC.dose(BC.width(r, u, e), u, BC.SETTINGS.blend).dose)(R, eta, unit))


# ------------------------------------------------------------------------------------------------ source

class SourceTest(unittest.TestCase):
    """The restated step is Unifloral's text line for line, plus BCA's marked lines."""

    @classmethod
    def setUpClass(cls):
        UI.prepare(device=_device())
        from experiments.shift_hosts import unifloral_td3_bc_bca as TB

        cls.TB = TB
        cls.theirs = (UI.ALGORITHMS / "td3_bc.py").read_text(encoding="utf8").splitlines()
        text = open(inspect.getsourcefile(TB), encoding="utf8").read()
        cls.ours, cls.tree = text.splitlines(), ast.parse(text)

    def _function(self, name):
        fn = next(n for n in self.tree.body if isinstance(n, ast.FunctionDef) and n.name == name)
        return self.ours[fn.lineno - 1:fn.end_lineno]

    def test_make_train_step_is_unifloral_line_for_line(self):
        """Removing every line marked "# BCA:" from make_train_step leaves td3_bc.py:174-279 exactly, minus the two
        changed lines (174, the signature; 245, the actor loss), which carry the mark; the vendored file is the pinned
        one (stop rule S1)."""
        TB = self.TB
        UI.check_pinned(LEARNER)
        lo, hi = TB.STEP_LINES
        ours = self._function("make_train_step")
        theirs = self.theirs[lo - 1:hi]
        kept = [line for line in ours if "# BCA:" not in line]
        expected = [line for i, line in enumerate(theirs, start=lo) if i not in TB.CHANGED_LINES]
        self.assertEqual(kept, expected)
        self.assertEqual(sorted(TB.CHANGED_LINES), [174, 245])
        self.assertEqual(theirs[0], "def make_train_step(args, actor_apply_fn, q_apply_fn, dataset):")
        self.assertTrue(ours[0].startswith(
            "def make_train_step(args, actor_apply_fn, q_apply_fn, dataset, *, bca=None, test=None):"))
        self.assertEqual(theirs[245 - lo], "            actor_loss = -lambda_ * q.mean() + bc_loss.mean()")
        dosed = [line for line in ours if "actor_loss = -lambda_ * q.mean() + " in line]
        self.assertEqual(len(dosed), 1)
        self.assertTrue(dosed[0].startswith("            actor_loss = -lambda_ * q.mean() + (bc_loss if bca is None "
                                            "else band.m * bc_loss).mean()  # BCA:"))
        DETAILS["source"] = dict(unifloral_lines=f"td3_bc.py:{lo}-{hi}", unifloral_line_count=len(theirs),
                                 kept_verbatim=len(kept), bca_lines=len(ours) - len(kept),
                                 changed_lines=sorted(TB.CHANGED_LINES), sha256=UI.check_pinned(LEARNER))

    def test_own_target_is_unifloral_compute_target(self):
        """_own_target's _compute_target is td3_bc.py:191-208 verbatim, indentation aside, and _own_target only
        returns it."""
        lo, hi = self.TB.TARGET_LINES
        body = self._function("_own_target")
        start = next(i for i, line in enumerate(body) if line.strip().startswith("def _compute_target("))
        theirs = textwrap.dedent("\n".join(self.theirs[lo - 1:hi])).splitlines()
        ours = textwrap.dedent("\n".join(body[start:start + len(theirs)])).splitlines()
        self.assertEqual(ours, theirs)
        self.assertEqual([line.strip() for line in body[start + len(theirs):] if line.strip()],
                         ["return _compute_target"])

    def test_uses_the_imported_modules(self):
        """The restated step's globals (jax, jnp, optax) are the very modules Unifloral's file uses."""
        U = UI.load(LEARNER)
        for name in ("jax", "jnp", "optax"):
            self.assertIs(getattr(self.TB, name), getattr(U, name))


# ------------------------------------------------------------------------------------------------ (a1), (a2)

class A1CopyIsUnifloralTest(unittest.TestCase):
    """(a1) With BCA absent the restated step is Unifloral's own step (plan 4.5)."""

    @classmethod
    def setUpClass(cls):
        cls.F = fixture()

    def test_a1_sixty_steps_bit_for_bit(self):
        """60 outer steps from Unifloral's own initializer: every leaf of the learner state and its optimizer states,
        the key carry and the per-step losses equal the imported make_train_step's, bit for bit."""
        F = self.F
        ref = F.unifloral60()
        ours = F.scan("alone", F.initial(), STEPS)
        rows = leaf_report(ours, ref)
        DETAILS["a1"] = dict(device=_device(), steps=STEPS, **summary(rows))
        self.assertEqual([r["path"] for r in rows if not r["equal"]], [])
        (rng, st), loss = ours
        # Not vacuous: the learner trained, at Unifloral's schedule (2 critic updates and 1 actor update per step).
        self.assertFalse(bits_equal(st.actor.params, F.built.agent_state.actor.params))
        self.assertFalse(bits_equal(rng, F.built.rng))
        counts = (int(st.actor.step), int(st.dual_q.step), int(st.actor_target.step), int(st.dual_q_target.step))
        self.assertEqual(counts, (STEPS, 2 * STEPS, STEPS, STEPS))
        self.assertEqual(sorted(loss), ["actor_loss", "bc_loss", "lambda", "q_loss", "q_mean"])
        self.assertTrue(all(np.all(np.isfinite(np.asarray(v))) and np.shape(v) == (STEPS,) for v in loss.values()))
        # The comparison sees the smallest change: one float32 rounding unit in one parameter.
        import jax

        original = ref[0][1].actor.params["params"]["Dense_0"]["kernel"]
        kernel = np.array(original)
        kernel[0, 0] = np.nextafter(kernel[0, 0], np.float32(np.inf))
        self.assertFalse(bits_equal(jax.numpy.asarray(kernel), original))


class A2BCAPresentDoseOneTest(unittest.TestCase):
    """(a2) BCA present before its first refresh changes nothing (plan 4.5)."""

    @classmethod
    def setUpClass(cls):
        cls.F = fixture()

    def test_a2_standard_and_iw_before_any_refresh(self):
        """Standard and IW setups, 60 outer steps from Unifloral's own initializer: Unifloral's part of the carry
        (key, learner state, optimizer states) and Unifloral's per-step losses equal the imported step's, bit for bit
        on the processor; on the card bit for bit or every leaf within 1e-6 relative (recorded). Every dose is exactly
        1, every step valid, and BCA really ran (60 accepted fits)."""
        F = self.F
        (rng_u, st_u), loss_u = F.unifloral60()
        record = {}
        for most_caution, name in ((False, "standard"), (True, "iw")):
            (rng, st, bs), out = F.bca60(most_caution)
            rows = leaf_report(((rng, st), {k: out[k] for k in loss_u}), ((rng_u, st_u), loss_u))
            s = summary(rows)
            integers_equal = all(r["equal"] for r in rows if r["integer"])
            s.update(state_and_key_bit_for_bit=bits_equal((rng, st), (rng_u, st_u)),
                     losses_bit_for_bit=bits_equal({k: out[k] for k in loss_u}, loss_u),
                     integer_leaves_equal=integers_equal,
                     verdict="bit for bit" if s["bit_for_bit"] else
                     ("rounding-level difference (within 1e-6 relative)" if integers_equal and s["max_rel"] <= CARD_REL
                      else "larger difference"))
            record[name] = s
            for flag in ("inputs_valid", "scale_inputs_valid", "dose_inputs_valid"):
                self.assertTrue(bool(np.all(out[flag])), (name, flag))
            self.assertTrue(bool(np.all(np.asarray(out["dose_min"]) == 1.0)))
            self.assertTrue(bool(np.all(np.asarray(out["dose_max"]) == 1.0)))
            self.assertFalse(bool(np.any(out["posterior_ready"])))
            self.assertTrue(bool(np.all(np.asarray(out["refused_fraction"]) == 0.0)))
            self.assertEqual((int(bs.step), int(bs.scale.step)), (STEPS, STEPS))      # 60 fits, all accepted
            self.assertFalse(bits_equal(bs.scale.params, F.scale0.params))
            self.assertNotEqual(float(bs.unit), 1.0)
            self.assertFalse(bool(bs.reference.ready))
        DETAILS["a2"] = dict(device=_device(), steps=STEPS, card_rel_allowed=CARD_REL, **record)
        for name, s in record.items():
            if ON_PROCESSOR:
                self.assertTrue(s["bit_for_bit"], (name, s))
            else:
                self.assertTrue(s["integer_leaves_equal"] and s["max_rel"] <= CARD_REL, (name, s))
        # The two setups compile one program and, before a refresh, carry the same values but the refusal rule.
        (_, st_s, bs_s), out_s = F.bca60(False)
        (_, st_i, bs_i), out_i = F.bca60(True)
        self.assertTrue(bits_equal((st_s, bs_s._replace(most_caution=None), out_s),
                                   (st_i, bs_i._replace(most_caution=None), out_i)))


# ------------------------------------------------------------------------------------------------ (a3)

class A3RealDoseTest(unittest.TestCase):
    """(a3) A real dose acts at the right place and strength, and nowhere else (plan 4.5)."""

    @classmethod
    def setUpClass(cls):
        cls.F = fixture()

    def test_a3_loss_and_gradient_equal_the_formula(self):
        """Three outer steps with the fixed made-up dose (madeup.dose, in [1, 1.5)): each step's actor loss and its
        gradient equal -lambda mean Q1 + mean(m bc) computed outside the step, within relative 1e-5; the formula
        without the dose, or with the dose on the Q term, misses by more than 100 times that."""
        F = self.F
        carry = F.bca60(False)[0]
        record = []
        for _ in range(3):
            new, out = F.scan("override", carry, 1)
            self.assertTrue(bool(out["inputs_valid"][0]))
            idx = np.asarray(out["batch_indices"][0])
            m = F.m.dose[idx]
            self.assertEqual(np.asarray(out["dose"][0]).tobytes(), m.tobytes())      # the dose used, row by row
            rows = F.rows(idx)
            actor, critics = carry[1].actor.params, new[1].dual_q.params             # critics after both updates
            loss, grad = external_actor_loss(F, actor, critics, rows, m, F.args.td3_alpha)
            loss_1, grad_1 = external_actor_loss(F, actor, critics, rows, np.ones_like(m), F.args.td3_alpha)
            loss_q, grad_q = external_actor_loss(F, actor, critics, rows, m, F.args.td3_alpha, place="q")
            step_loss, step_grad = out["dosed_loss"][0], first(out["dosed_grad"])
            r = dict(loss_rel=relative(step_loss, loss), grad_rel=tree_relative(step_grad, grad),
                     undosed_loss_rel=relative(step_loss, loss_1), undosed_grad_rel=tree_relative(step_grad, grad_1),
                     q_place_loss_rel=relative(step_loss, loss_q), q_place_grad_rel=tree_relative(step_grad, grad_q),
                     dose_mean=float(np.mean(m)), loss=float(step_loss))
            record.append(r)
            self.assertLessEqual(r["loss_rel"], F32_REL, r)
            self.assertLessEqual(r["grad_rel"], F32_REL, r)
            for k in ("undosed_loss_rel", "undosed_grad_rel", "q_place_loss_rel", "q_place_grad_rel"):
                self.assertGreater(r[k], 100 * F32_REL, (k, r))
            carry = new
        DETAILS["a3_formula"] = dict(device=_device(), steps=record)

    def test_a3_the_dose_changes_nothing_else(self):
        """One outer step with the made-up dose against the imported step from the same carry: the key, both critics,
        both target critics, q_loss, q_mean, lambda and bc_loss (Unifloral's undosed mean) are unchanged (bit for bit
        on the processor; within 1e-6 relative on the card); the actor, its target copy and actor_loss differ."""
        F = self.F
        start = F.bca60(False)[0]
        (rng, st, _), out = F.scan("override", start, 1)
        (rng_u, st_u), loss_u = F.scan("unifloral", start[:2], 1)
        same = dict(rng=(rng, rng_u), dual_q=(st.dual_q, st_u.dual_q),
                    dual_q_target=(st.dual_q_target, st_u.dual_q_target),
                    **{k: (out[k], loss_u[k]) for k in ("q_loss", "q_mean", "lambda", "bc_loss")})
        differ = dict(actor=(st.actor.params, st_u.actor.params), actor_target=(st.actor_target.params,
                                                                               st_u.actor_target.params),
                      actor_loss=(out["actor_loss"], loss_u["actor_loss"]))
        rec = {}
        for name, (a, b) in same.items():
            s = summary(leaf_report(a, b))
            rec[name] = dict(bit_for_bit=s["bit_for_bit"], max_rel=s["max_rel"])
            if ON_PROCESSOR or name == "rng":
                self.assertTrue(s["bit_for_bit"], (name, s))
            else:
                self.assertLessEqual(s["max_rel"], CARD_REL, (name, s))
        for name, (a, b) in differ.items():
            rec[name] = dict(bit_for_bit=bits_equal(a, b), max_rel=tree_relative(a, b))
            self.assertFalse(rec[name]["bit_for_bit"], name)
        self.assertGreater(rec["actor_loss"]["max_rel"], 1e-3)
        DETAILS["a3_nothing_else"] = dict(device=_device(), **rec)

    def test_a3_cross_check_with_the_imported_file(self):
        """Extra (the plan names no imported-file cross-check for TD3+BC): every dose 1.5 makes the actor loss
        -lambda mean Q1 + 1.5 mean bc = 1.5 x (-(lambda / 1.5) mean Q1 + mean bc), so one step from the same carry gives
        1.5 x the imported step's actor_loss and lambda at td3_alpha = 2.5 / 1.5, within relative 1e-5, and the same
        critic losses, q_mean and bc_loss. A dose on the Q term would not."""
        F = self.F
        start = F.bca60(False)[0]
        _, out = F.scan("override_15", start, 1)
        _, loss_x = F.scan("unifloral_alpha", start[:2], 1)
        r = dict(actor_loss_rel=relative(out["actor_loss"][0], 1.5 * np.float64(loss_x["actor_loss"][0])),
                 lambda_rel=relative(out["lambda"][0], 1.5 * np.float64(loss_x["lambda"][0])),
                 actor_loss=float(out["actor_loss"][0]), unifloral_actor_loss_at_alpha=float(loss_x["actor_loss"][0]),
                 td3_alpha=F.args.td3_alpha / 1.5)
        for k in ("q_loss", "q_mean", "bc_loss"):
            r[k + "_rel"] = relative(out[k][0], loss_x[k][0])
            if ON_PROCESSOR:
                self.assertEqual(np.asarray(out[k][0]).tobytes(), np.asarray(loss_x[k][0]).tobytes(), k)
            else:
                self.assertLessEqual(r[k + "_rel"], CARD_REL, k)
        self.assertLessEqual(r["actor_loss_rel"], F32_REL, r)
        self.assertLessEqual(r["lambda_rel"], F32_REL, r)
        DETAILS["a3_cross_check"] = dict(device=_device(), **r)


# ------------------------------------------------------------------------------------------------ (a4)

@processor_only
class A4OwnTargetAndCriticTest(unittest.TestCase):
    """(a4) The band reads the learner's own target and critic (plan 4.5)."""

    @classmethod
    def setUpClass(cls):
        cls.F = fixture()

    def test_a4_with_unifloral_keys_y_and_q_are_unifloral_s_own(self):
        """In test mode with Unifloral's own per-row keys (those of its first critic update, td3_bc.py:210-211), the
        band's y equals that update's targets (td3_bc.py:212) and q equals the minimum over the two critics of its
        q_pred (td3_bc.py:217, at the parameters _q_loss_fn receives), bit for bit, at every row of 5 outer steps from
        Unifloral's own initializer and 5 more after 60 steps of training."""
        F = self.F
        record = {}
        for label, carry in (("from_init", F.initial(False)), ("after_60", F.bca60(False)[0])):
            _, out = F.scan("unifloral_keys", carry, 5)
            y, uy = np.asarray(out["bca_y"]), np.asarray(out["unifloral_y"])
            q, uq = np.asarray(out["bca_q"]), np.asarray(out["unifloral_q"])
            self.assertEqual((y.shape, q.shape), ((5, F.args.batch_size),) * 2)
            self.assertEqual(y.tobytes(), uy.tobytes(), label)
            self.assertEqual(q.tobytes(), uq.tobytes(), label)
            self.assertTrue(bool(np.all(out["inputs_valid"])))
            record[label] = dict(steps=5, rows=int(y.size), y_bit_for_bit=True, q_bit_for_bit=True,
                                 y_mean=float(y.mean()), q_mean=float(q.mean()))
        DETAILS["a4_unifloral_keys"] = record

    def test_a4_with_bca_keys_y_is_the_same_function_at_bca_s_draw(self):
        """In the normal fit (BCA's own keys): q still equals Unifloral's, bit for bit (q has no noise); y differs from
        the first critic update's targets (another noise draw); and y and q equal Unifloral's own target and critic
        value at BCA's keys split(fold_in(fold_in(PRNGKey(s91), t), 1179210836), 256) from the state at the start of
        the step, computed outside the step (band_parts compiled as its own program), within 1e-6 relative (two
        compiled programs may differ by a few float32 rounding units; recorded)."""
        import jax

        F, BC, TB = self.F, self.F.BC, self.F.TB
        carry = F.bca60(False)[0]
        outside = jax.jit(lambda st, r, k: TB.band_parts(F.args, F.built.apply_fns, st, r, k))
        record = []
        for _ in range(3):
            new, out = F.scan("plain", carry, 1)
            q, uq = np.asarray(out["bca_q"][0]), np.asarray(out["unifloral_q"][0])
            y, uy = np.asarray(out["bca_y"][0]), np.asarray(out["unifloral_y"][0])
            self.assertEqual(q.tobytes(), uq.tobytes())
            self.assertGreater(float(np.mean(y != uy)), 0.9)
            t = int(carry[2].step) + 1
            noise_key = BC.fit_noise_key(BC.fit_key(F.base, t), LEARNER)
            literal = jax.random.fold_in(jax.random.fold_in(jax.random.PRNGKey(F.seeds.s91), t), 1179210836)
            self.assertTrue(bits_equal(noise_key, literal))
            rows = F.rows(out["batch_indices"][0])
            y_out, q_out = outside(carry[1], rows, noise_key)
            r = dict(t=t, rows_where_y_differs_from_unifloral_draw=float(np.mean(y != uy)),
                     unifloral_draw_rel=relative(y, uy), outside_y_rel=relative(y, y_out),
                     outside_q_rel=relative(q, q_out), outside_y_bit_for_bit=np.asarray(y_out).tobytes() == y.tobytes(),
                     outside_q_bit_for_bit=np.asarray(q_out).tobytes() == q.tobytes())
            self.assertLessEqual(r["outside_y_rel"], PROGRAM_REL, r)
            self.assertLessEqual(r["outside_q_rel"], PROGRAM_REL, r)
            self.assertGreater(r["unifloral_draw_rel"], 10 * PROGRAM_REL, r)      # another draw is far from rounding
            record.append(r)
            carry = new
        DETAILS["a4_bca_keys"] = record


# ------------------------------------------------------------------------------------------------ (b2), (b3)

@processor_only
class B2RowThresholdTest(unittest.TestCase):
    """(b2) Each batch row reads its own threshold through Unifloral's batch_indices (plan 4.5)."""

    @classmethod
    def setUpClass(cls):
        cls.F = fixture()

    def test_b2_each_row_reads_its_own_threshold(self):
        """After a refresh (a ready reference), thresholds at 4 levels [0.5, 1, 2, 4][row % 4]: batch row b's
        threshold is R[batch_indices[b]], and its dose equals, bit for bit, the dose computed outside the step at that
        threshold and the step's eta (bca_core.width and dose) and BCA's own frozen_level_dose with the reference's
        threshold set to that level. The step's eta is the frozen scale network's at the batch rows (recomputed outside
        as its own program: within 1e-6 relative, and so is the dose from it). Standard and IW agree (no row refused);
        a lookup shifted by one row would give other doses; the actor loss used these doses."""
        import jax
        import jax.numpy as jnp

        from calibration.dose import frozen_level_dose

        F = self.F
        carry, refreshed = F.refreshed()
        self.assertTrue(refreshed.record["valid"] and refreshed.record["certified"])
        ref = carry[2].reference
        outs = {mc: F.scan("plain", F.with_thresholds(carry, F.m.R_levels, mc), 1) for mc in (False, True)}
        new, out = outs[False]
        self.assertTrue(bool(out["inputs_valid"][0]) and bool(outs[True][1]["inputs_valid"][0]))
        idx = np.asarray(out["batch_indices"][0])
        R_b = F.m.R_levels[idx]
        eta, dose = np.asarray(out["eta"][0]), np.asarray(out["dose"][0])
        self.assertEqual(np.asarray(out["R_rows"][0]).tobytes(), R_b.tobytes())
        rows = F.rows(idx)
        scale_apply = carry[2].scale.apply_fn
        eta_out = np.asarray(jax.jit(scale_apply)(ref.cal_params, rows.obs, rows.action))
        self.assertLessEqual(relative(eta, eta_out), PROGRAM_REL)
        expected = outside_dose(F, R_b, eta, ref.residual_scale)
        self.assertEqual(expected.tobytes(), dose.tobytes())
        from_outside_eta = outside_dose(F, R_b, eta_out, ref.residual_scale)
        self.assertLessEqual(relative(from_outside_eta, dose), PROGRAM_REL)
        own = jax.jit(lambda r, e: frozen_level_dose(r, e, 0.5).dose)
        per_level = {}
        for level in (0.5, 1.0, 2.0, 4.0):
            hit = R_b == np.float32(level)
            d = np.asarray(own(ref._replace(threshold=jnp.asarray(level, jnp.float32)), eta))
            self.assertEqual(d[hit].tobytes(), dose[hit].tobytes(), level)
            per_level[str(level)] = dict(rows=int(hit.sum()), mean_dose=float(dose[hit].mean()))
        self.assertEqual(np.asarray(outs[True][1]["dose"][0]).tobytes(), dose.tobytes())
        self.assertTrue(bool(np.all((dose > 1.0) & (dose < 1.5))))
        shifted = outside_dose(F, F.m.R_levels[(idx + 1) % F.m.n], eta, ref.residual_scale)
        mismatch = float(np.mean(shifted != dose))
        self.assertGreater(mismatch, 0.9)
        actor, critics = carry[1].actor.params, new[1].dual_q.params
        loss, grad = external_actor_loss(F, actor, critics, rows, dose, F.args.td3_alpha)
        self.assertLessEqual(relative(out["dosed_loss"][0], loss), F32_REL)
        self.assertLessEqual(tree_relative(first(out["dosed_grad"]), grad), F32_REL)
        DETAILS["b2"] = dict(threshold=float(ref.threshold), unit=float(ref.residual_scale), levels=per_level,
                             distinct_doses=int(len(np.unique(dose))), shifted_lookup_mismatch=mismatch,
                             dose_min=float(dose.min()), dose_max=float(dose.max()),
                             dose_from_step_eta_bit_for_bit=True,
                             eta_outside_bit_for_bit=eta_out.tobytes() == eta.tobytes(),
                             eta_outside_rel=relative(eta, eta_out),
                             dose_from_outside_eta_rel=relative(from_outside_eta, dose))


@processor_only
class B3RefusedRowsTest(unittest.TestCase):
    """(b3) Both refusal rules work as stated (plan 4.2, 4.5; choice 16)."""

    @classmethod
    def setUpClass(cls):
        cls.F = fixture()

    def test_b3_refused_rows(self):
        """After a refresh, R = +inf on every third row and 1 elsewhere (madeup.R_refused): the rows read as refused are
        exactly those with batch_indices % 3 == 0; standard BCA gives them dose 1 and IW BCA 1.5 (the weighting test's
        own rule, learners_iw.py:681-682); their width is infinite; every other row has the same dose under both rules,
        equal to the dose outside the step at R = 1; both steps are valid and their actor losses used these doses."""
        F = self.F
        carry, _ = F.refreshed()
        ref = carry[2].reference
        res = {mc: F.scan("plain", F.with_thresholds(carry, F.m.R_refused, mc), 1) for mc in (False, True)}
        (new_s, std), (new_i, iw) = res[False], res[True]
        idx = np.asarray(std["batch_indices"][0])
        self.assertEqual(idx.tobytes(), np.asarray(iw["batch_indices"][0]).tobytes())       # the same batch
        refused = idx % 3 == 0
        self.assertTrue(0 < refused.sum() < len(idx))
        for out in (std, iw):
            self.assertTrue(bool(out["inputs_valid"][0]))
            self.assertEqual(np.asarray(out["refused"][0]).tolist(), refused.tolist())
            self.assertTrue(bool(np.all(np.isposinf(np.asarray(out["width"][0])[refused]))))
            self.assertAlmostEqual(float(out["refused_fraction"][0]), float(refused.mean()), places=6)
        d_std, d_iw = np.asarray(std["dose"][0]), np.asarray(iw["dose"][0])
        self.assertTrue(bool(np.all(d_std[refused] == 1.0)))
        self.assertTrue(bool(np.all(d_iw[refused] == 1.5)))
        self.assertEqual(d_std[~refused].tobytes(), d_iw[~refused].tobytes())
        eta = np.asarray(std["eta"][0])
        expected = outside_dose(F, np.ones(int((~refused).sum()), np.float32), eta[~refused], ref.residual_scale)
        self.assertEqual(expected.tobytes(), d_std[~refused].tobytes())
        self.assertTrue(bool(np.all((d_std[~refused] > 1.0) & (d_std[~refused] < 1.5))))
        rows = F.rows(idx)
        for (new, out), d in ((res[False], d_std), (res[True], d_iw)):
            loss, _ = external_actor_loss(F, carry[1].actor.params, new[1].dual_q.params, rows, d, F.args.td3_alpha)
            self.assertLessEqual(relative(out["dosed_loss"][0], loss), F32_REL)
        self.assertNotEqual(float(std["dosed_loss"][0]), float(iw["dosed_loss"][0]))
        # Before the first refresh nothing is refused, whatever the thresholds: every dose is exactly 1.
        early = F.with_thresholds(F.bca60(True)[0], F.m.R_refused, True)
        _, out0 = F.scan("plain", early, 1)
        self.assertTrue(bool(np.all(np.asarray(out0["dose"][0]) == 1.0)))
        self.assertFalse(bool(np.any(out0["refused"][0])))
        DETAILS["b3"] = dict(rows=int(len(idx)), refused_rows=int(refused.sum()), standard_refused_dose=1.0,
                             iw_refused_dose=1.5, other_rows_equal=True,
                             dosed_loss_standard=float(std["dosed_loss"][0]), dosed_loss_iw=float(iw["dosed_loss"][0]))


# ------------------------------------------------------------------------------------------------ refresh

def direct_scores(F, agent_state, bca_state, refresh_key):
    """The bank scores computed directly, batched (no per-row vmap), with the port's literal constant: per-row noise
    keys split(fold_in(refresh key, 1213156420), n) (algorithms/td3_bc_bca.py:249; td3_bc.py:210-211), each split 3
    ways with the third drawing N(0, 1) for the 4 commands (td3_bc.py:196-198); the noisy target command
    clip(tanh(actor_target(s')) + clip(0.2 noise, +-0.5), +-1) (lines 195-201); y = r + 0.98 ((1 - d) min of the 2
    target critics there) (204-208); q = min(Q1, Q2)(s, a); eta = the live scale network; score = |y - q| /
    (max(eta, 1e-6) u) in float64, as calibration/reference.freeze_reference scores (reference.py:164-175)."""
    import jax
    import jax.numpy as jnp

    actor_apply, q_apply = F.built.apply_fns
    bank = F.bank
    n = bank.obs.shape[0]

    def compute(st, scale, unit, key):
        keys = jax.random.split(jax.random.fold_in(key, 1213156420), n)
        noise = jax.vmap(lambda k: jax.random.normal(jax.random.split(k, 3)[2], (4,)))(keys)
        noise = jnp.clip(noise * 0.2, -0.5, 0.5)
        command = jnp.clip(jnp.tanh(actor_apply(st.actor_target.params, bank.next_obs).distribution.loc) + noise,
                           -1, 1)
        y = bank.reward + 0.98 * ((1.0 - bank.done) * q_apply(st.dual_q_target.params, bank.next_obs, command).min(1))
        q = q_apply(st.dual_q.params, bank.obs, bank.action).min(1)
        eta = scale.apply_fn(scale.params, bank.obs, bank.action)
        return y, q, jnp.maximum(eta, 1e-6) * unit

    y, q, scale = jax.jit(compute)(agent_state, bca_state.scale, bca_state.unit, refresh_key)
    return np.abs(np.asarray(y - q, np.float64)) / np.asarray(scale, np.float64), (y, q)


@processor_only
class RefreshTest(unittest.TestCase):
    """TD3+BC's refresh: bank scores from a stored state (plan 4.1, 4.2; stop rule S3)."""

    @classmethod
    def setUpClass(cls):
        cls.F = fixture()

    def test_refresh_scores_equal_a_direct_computation(self):
        """After 60 outer steps, refresh() on the 300-row made-up bank. Its scores, recomputed from the stored states
        (the agent state, and the BCA state the refresh produced) by recompute_bank_scores, are the same bit for bit
        (the same compiled program), with the same stream, the words of fold_in(refresh key, 1347375956). They equal a
        direct computation written independently (batched, the port's literal constants) to float32 rounding (relative
        1e-5; measured and recorded, with whether it is bit for bit); another refresh key's scores do not. An
        unweighted calibrate on the scores and stream reproduces the stored threshold, lambda_hat, lambda_hpd and n_eff
        bit for bit (iw.reproduces, stop rule S3). The reference freezes the live scale network and unit; every row's
        threshold is set to R; the record is JSON-ready."""
        import jax

        from calibration import wbcp
        from experiments.shift_hosts import handoff as H
        from experiments.shift_hosts import iw as IW

        F = self.F
        (rng, st, bs), _ = F.bca60(False)
        key = F.BC.refresh_key(F.seeds.s39, STEPS)
        self.assertTrue(bits_equal(key, jax.random.fold_in(jax.random.PRNGKey(F.seeds.s39), STEPS)))
        carry, out = F.refreshed()
        self.assertEqual(out.scores.dtype, np.float64)
        self.assertEqual(out.scores.shape, (F.m.bank.obs.shape[0],))
        again, stream_again = F.TB.recompute_bank_scores(F.args, F.built.apply_fns, st, out.state, F.bank, key,
                                                         permission=F.guard.MADE_UP)
        self.assertEqual(again.tobytes(), out.scores.tobytes())
        self.assertEqual(stream_again.tobytes(), out.stream.tobytes())
        self.assertTrue(IW.reproduces(out.state.reference, again, stream_again))
        scores, (y, q) = direct_scores(F, st, bs, key)
        direct_rel = relative(out.scores, scores)
        self.assertLessEqual(direct_rel, F32_REL)
        other, _ = direct_scores(F, st, bs, F.BC.refresh_key(F.seeds.s39, STEPS + 1))
        other_rel = relative(out.scores, other)
        self.assertGreater(other_rel, 100 * F32_REL)
        y_tb, q_tb = F.TB.band_parts(F.args, F.built.apply_fns, st, F.bank, F.BC.refresh_noise_key(key, LEARNER))
        self.assertLessEqual(relative(y_tb, y), 1e-6)            # band_parts run op by op (not compiled), to rounding
        self.assertLessEqual(relative(q_tb, q), 1e-6)
        stream = np.asarray(jax.random.fold_in(key, 1347375956), np.uint32).ravel()
        self.assertEqual(out.stream.tobytes(), stream.tobytes())
        ref = out.state.reference
        self.assertTrue(out.record["valid"] and out.record["certified"] and bool(ref.ready))
        self.assertTrue(np.isfinite(float(ref.threshold)))
        self.assertTrue(IW.reproduces(ref, out.scores, out.stream))
        c = wbcp.calibrate(out.scores, np.random.default_rng(stream), alpha=0.1, beta=0.95, draws=1000)
        for a, b in ((c.threshold, ref.threshold), (c.lambda_hat, ref.lambda_hat), (c.lambda_hpd, ref.lambda_hpd),
                     (c.n_eff, ref.n_eff)):
            self.assertEqual(np.float32(a).tobytes(), np.asarray(b, np.float32).tobytes())
        # The independently compiled scores give the threshold only to rounding: S3 must use the refresh's own scores
        # (or recompute_bank_scores, the same program), as iw.iw_refresh does by default.
        c_direct = wbcp.calibrate(scores, np.random.default_rng(stream), alpha=0.1, beta=0.95, draws=1000)
        direct_threshold_rel = relative(np.float32(c_direct.threshold), np.asarray(ref.threshold, np.float32))
        self.assertLessEqual(direct_threshold_rel, F32_REL)
        self.assertTrue(bits_equal(ref.cal_params, bs.scale.params))
        self.assertEqual(np.asarray(ref.residual_scale).tobytes(), np.asarray(bs.unit, np.float32).tobytes())
        thresholds = np.asarray(out.state.thresholds)
        self.assertEqual(thresholds.shape, (F.m.n,))
        self.assertTrue(bool(np.all(thresholds == np.float32(ref.threshold))))
        self.assertTrue(bits_equal((out.state.scale, out.state.unit, out.state.step), (bs.scale, bs.unit, bs.step)))
        H.plain(out.record)
        DETAILS["refresh"] = dict(bank_rows=int(scores.size), recomputed_from_stored_state_bit_for_bit=True,
                                  direct_bit_for_bit=out.scores.tobytes() == scores.tobytes(), direct_rel=direct_rel,
                                  other_refresh_key_rel=other_rel, direct_scores_threshold_rel=direct_threshold_rel,
                                  direct_scores_reproduce_threshold=bool(IW.reproduces(ref, scores, stream)),
                                  threshold=float(ref.threshold), lambda_hat=float(ref.lambda_hat),
                                  lambda_hpd=float(ref.lambda_hpd), n_eff=float(ref.n_eff),
                                  unit=float(ref.residual_scale), s3_reproduced=True,
                                  scores_sha256=out.record["scores_sha256"])


# ------------------------------------------------------------------------------------------------ guards and validity

@processor_only
class GuardAndValidityTest(unittest.TestCase):
    """The ports' safety checks carried over (plan 3.1, 4.1) and the made-up-data guard at every BCA entry point."""

    @classmethod
    def setUpClass(cls):
        cls.F = fixture()

    def test_factory_and_refresh_refuse(self):
        """With BCA present the factory refuses arrays that are not registered as made-up (one value changed), a missing
        permission, another learner's setup, IQL's width_override and a dose_override of the wrong length; test mode
        needs BCA; the refresh refuses a bank subset and a missing permission. BCA absent needs no permission."""
        import jax

        F, BC, TB, guard = self.F, self.F.BC, self.F.TB, self.F.guard
        a, apply_fns, data = F.args, F.built.apply_fns, F.built.dataset
        changed = data._replace(reward=data.reward.at[0].add(1.0))
        with self.assertRaises(guard.ApprovalError):
            TB.make_train_step(a, *apply_fns, changed, bca=F.setup)
        with self.assertRaises(guard.ApprovalError):
            TB.make_train_step(a, *apply_fns, data, bca=BC.BCASetup(LEARNER, F.base, None))
        with self.assertRaises(ValueError):
            TB.make_train_step(a, *apply_fns, data, bca=BC.BCASetup("cql", F.base, guard.MADE_UP))
        with self.assertRaises(ValueError):
            TB.make_train_step(a, *apply_fns, data, test=BC.TestOptions())
        with self.assertRaises(ValueError):
            TB.make_train_step(a, *apply_fns, data, bca=F.setup, test=BC.TestOptions(width_override=F.m.width))
        with self.assertRaises(ValueError):
            TB.make_train_step(a, *apply_fns, data, bca=F.setup, test=BC.TestOptions(dose_override=F.m.dose[:-1]))
        TB.make_train_step(a, *apply_fns, changed)                     # the learner alone: Unifloral's own step
        (_, st, bs), _ = F.bca60(False)
        key = BC.refresh_key(F.seeds.s39, STEPS)
        part = jax.tree_util.tree_map(lambda x: x[:-1], F.bank)
        with self.assertRaises(guard.ApprovalError):
            TB.refresh(a, apply_fns, st, bs, part, key, permission=guard.MADE_UP)
        with self.assertRaises(guard.ApprovalError):
            TB.refresh(a, apply_fns, st, bs, F.bank, key, permission=None)
        with self.assertRaises(guard.ApprovalError):
            TB.recompute_bank_scores(a, apply_fns, st, bs, part, key, permission=guard.MADE_UP)
        with self.assertRaises(guard.ApprovalError):
            TB.recompute_bank_scores(a, apply_fns, st, bs, F.bank, key, permission=None)
        # the step's own trace-time check of the per-row thresholds (one float32 per training row), as ReBRAC's
        import jax.numpy as jnp

        rng0 = F.bca60(False)[0][0]
        for wrong in (bs.thresholds[:-1], jnp.concatenate([bs.thresholds, bs.thresholds[:1]]),
                      bs.thresholds.astype(jnp.float16)):
            with self.assertRaisesRegex(ValueError, "one threshold per training row"):
                F.scan("bca", (rng0, st, bs._replace(thresholds=wrong)), 1)

    def test_an_invalid_step_is_discarded_whole(self):
        """A step whose BCA inputs are not valid (here the unit is 0, so the scale fit is not finite) returns
        inputs_valid False and keeps the old carry whole, random key, learner state and BCA state included
        (algorithms/td3_bc_bca.py:340-342); require_valid then stops (S5), and passes 60 valid steps."""
        import jax.numpy as jnp

        F = self.F
        rng, st, bs = F.bca60(False)[0]
        bad = (rng, st, bs._replace(unit=jnp.asarray(0.0, jnp.float32)))
        new, out = F.scan("bca", bad, 1)
        self.assertFalse(bool(out["inputs_valid"][0]))
        self.assertFalse(bool(out["scale_inputs_valid"][0]))
        self.assertTrue(bits_equal(new, bad))
        with self.assertRaises(FloatingPointError):
            F.TB.require_valid(out)
        F.TB.require_valid(F.bca60(False)[1])
        good, out_good = F.scan("bca", (rng, st, bs), 1)
        self.assertTrue(bool(out_good["inputs_valid"][0]))
        self.assertFalse(bits_equal(good[0], rng))
        DETAILS["validity"] = dict(invalid_step_kept_old_carry=True, valid_step_advanced_key=True)


if __name__ == "__main__":
    UI.prepare(device=_device())
    from experiments.shift_hosts import tests as T

    sys.exit(T.run(sys.modules[__name__], AREA, device_name=_device(),
                   extra=lambda: dict(unifloral_import=UI.record(), details=DETAILS)))

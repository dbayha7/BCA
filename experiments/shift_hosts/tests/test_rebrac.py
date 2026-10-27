"""ReBRAC's stage-0 tests (plan section 4.5) of the restated Unifloral step unifloral_rebrac_bca.py, on made-up arrays
only (tests/madeup.py, registered with the guard); BCA's code runs on them under approval item 8. No ThrusterChain data,
no learned policy's return. An experiment-local attachment (plan section 3.1) that changes none of the repository's
declared methods.

  source  The restated make_train_step's lines not marked "# BCA:" are rebrac.py:199-309 in order (the pinned file),
          and own_target's body is rebrac.py:216-235.
  (a1)    BCA absent = the imported make_train_step: 60 outer steps from Unifloral's own initializer, bit for bit
          (every leaf of the learner state and optimizer states, the key carry, every per-step loss); the traced
          programs are identical. Processor and card.
  (a2)    BCA present, no refresh (dose exactly 1), standard and IW refusal rules: Unifloral's part of the carry and
          Unifloral's per-step losses equal the imported step's after 60 steps, bit for bit on the processor; on the
          card bit for bit or every leaf within 1e-6 relative (recorded), larger fails.
  (a3)    A fixed made-up dose in [1, 1.5): the actor loss and its gradient equal -lambda mean q + 0.001 mean(m bc)
          written outside the step (rebrac.py:268, 272-274) to float32 rounding (relative 1e-5); controls show a
          missing or misplaced dose would fail; the critics and key carry are untouched. Cross-check: every dose 1.5
          against the imported file with actor_bc_coef = 0.0015 after one step. Added: the same at actor_bc_coef 100,
          where the cloning term dominates, so the per-row placement itself is checked at 1e-5.
  (a4)    With Unifloral's own key the band's y equals the first critic update's targets (rebrac.py:237-240) and q the
          minimum over the two critics of its q_pred (rebrac.py:245), bit for bit on the processor.
  (b2)    Thresholds at 4 levels: the threshold the step reads for batch row b is R[batch_indices[b]] bit for bit, and
          its dose equals BCA's own single-threshold dose computed outside the step at that R to float32 rounding.
  (b3)    R = +inf on every third row: standard BCA gives those rows dose 1, IW BCA 1.5 (choice 16), exactly; other
          rows get the same dose under both rules, bit for bit.
  validity An invalid step (a non-positive unit, or a dose outside [1, 2]) keeps the whole carry, key included.
  refresh The bank scores of refresh() from a stored state equal a direct computation; the threshold reproduces from
          the returned scores and stream; ReBRAC's refresh constants 1380470604 and 1380994899.
  (e)     The made-up arrays' next commands equal jnp.roll(actions, -1, axis=0) bit for bit (rebrac.py:335).
  entry   The BCA entry point's refusals (guard, learner, test options).

"Leaf-scale relative difference" of two arrays: max |a - b| / max |b| over the leaf; "to float32 rounding" means at most
1e-5 (plan 4.5). Readouts go to runs/shift_hosts/stage0/rebrac/test_rebrac_details_<device>.json (with a sidecar).

Run from the repository root in /opt/bca-venv, on the processor, then on the card (short):
  env D4RL_SUPPRESS_IMPORT_ERROR=1 MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa DISPLAY= \
      MUJOCO_PY_MUJOCO_PATH=/root/.mujoco/mujoco210 LD_LIBRARY_PATH=/root/.mujoco/mujoco210/bin \
      PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 JAX_PLATFORMS=cpu SHIFT_HOSTS_DEVICE=cpu \
      taskset -c 8-11 /opt/bca-venv/bin/python -m experiments.shift_hosts.tests.test_rebrac
  (card: SHIFT_HOSTS_DEVICE=gpu XLA_PYTHON_CLIENT_PREALLOCATE=false and no JAX_PLATFORMS)
"""

import dataclasses
import inspect
import math
import os
import sys
import unittest

import numpy as np

from experiments.shift_hosts import unifloral_import as UI

AREA = "rebrac"
STEPS = 60              # (a1), (a2): outer steps
SHORT = 3               # test-mode runs (a4, b2, b3)
REL_CARD = 1e-6         # (a2) on the card: every leaf within 1e-6 relative after 60 steps
REL_F32 = 1e-5          # "to float32 rounding" (plan 4.5)
UNIFLORAL_KEYS = ("actor_loss", "q_loss", "q_mean", "lambda", "bc_loss", "critic_bc_loss")    # rebrac.py:299-306
DETAILS = {}


def _device():
    return os.environ.get(UI.DEVICE_ENV) or "cpu"


def _cpu():
    return _device() == "cpu"


# ------------------------------------------------------------------------------------------------ comparisons

def _rel(a, b):
    """Leaf-scale relative difference max |a - b| / max |b|; inf when NaNs or infinities disagree."""
    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    if a.shape != b.shape:
        return math.inf
    if a.size == 0:
        return 0.0
    if not np.array_equal(np.isnan(a), np.isnan(b)):
        return math.inf
    both = ~np.isnan(a)
    if np.any((np.isinf(a) | np.isinf(b)) & both & (a != b)):
        return math.inf
    fin = both & np.isfinite(a) & np.isfinite(b)
    if not fin.any():
        return 0.0
    d = float(np.max(np.abs(a[fin] - b[fin])))
    s = float(np.max(np.abs(b[fin])))
    if d == 0.0:
        return 0.0
    return d / s if s > 0 else math.inf


def _compare(a, b):
    """Leaf by leaf in jax.tree_util order: the same tree structure, dtypes and shapes are required (AssertionError
    otherwise). Returns {leaves, equal (every leaf's bytes), int_equal (every non-floating leaf's bytes), max_rel (the
    largest leaf-scale relative difference of floating leaves), differing: [(path, rel)]}. A bare Python number (such
    as TrainState.create's step 0) goes through jnp.asarray first, as guard.state_sha256 does."""
    import jax
    import jax.numpy as jnp

    a, b = (jax.tree_util.tree_map(lambda x: x if hasattr(x, "dtype") else jnp.asarray(x), t) for t in (a, b))
    la, ta = jax.tree_util.tree_flatten_with_path(a)
    lb, tb = jax.tree_util.tree_flatten_with_path(b)
    if ta != tb:
        raise AssertionError(f"tree structures differ:\n{ta}\n{tb}")
    out = dict(leaves=len(la), equal=True, int_equal=True, max_rel=0.0, differing=[])
    for (path, x), (_, y) in zip(la, lb):
        x, y = np.asarray(x), np.asarray(y)
        name = jax.tree_util.keystr(path)
        if x.dtype != y.dtype or x.shape != y.shape:
            raise AssertionError(f"{name}: {x.dtype}{x.shape} against {y.dtype}{y.shape}")
        if x.tobytes() == y.tobytes():
            continue
        out["equal"] = False
        if np.issubdtype(x.dtype, np.floating):
            r = _rel(x, y)
            out["max_rel"] = max(out["max_rel"], r)
            out["differing"].append((name, r))
        else:
            out["int_equal"] = False
            out["differing"].append((name, "not floating"))
    return out


def _summary(rep, limit=12):
    return dict(leaves=rep["leaves"], bit_for_bit=rep["equal"], non_floating_equal=rep["int_equal"],
                max_leaf_relative_difference=rep["max_rel"], differing_leaves=len(rep["differing"]),
                first_differing=[[p, r] for p, r in rep["differing"][:limit]])


def _bits_equal(a, b):
    return _compare(a, b)["equal"]


def _step(tree, s):
    """Entry s of every leaf of a stacked scan output."""
    import jax

    return jax.tree_util.tree_map(lambda x: x[s], tree)


def _host(tree):
    import jax

    return jax.tree_util.tree_map(np.asarray, tree)


# ------------------------------------------------------------------------------------------------ shared fixture

class _Fixture:
    """Everything the tests share, built once and lazily (test classes run in alphabetical order): the made-up arrays,
    Unifloral's own start (unifloral_import.init_like_training_block, the plan's settings for data seed 0), BCA's start,
    the step functions, their compiled 60-step / 1-step / 3-step scans, and the runs."""

    def __init__(self):
        UI.prepare(device=_device())
        import jax
        import jax.numpy as jnp

        from experiments.shift_hosts import bca_core as BC
        from experiments.shift_hosts import guard
        from experiments.shift_hosts import unifloral_rebrac_bca as RB
        from experiments.shift_hosts.tests import madeup

        self.jax, self.jnp, self.BC, self.guard, self.RB = jax, jnp, BC, guard, RB
        self.U = UI.load("rebrac")
        self.m = madeup.build(0)
        self.seeds = BC.seeds(0)
        self.args = UI.args_for("rebrac", seed=self.seeds.unifloral, gamma=0.98, num_updates=100_000)
        self.built = UI.init_like_training_block("rebrac", self.args, self.m.rows("train"))
        self.bank = UI.make_transition("rebrac", self.m.rows("bank"), next_action=self.m.next_action["bank"])
        self.base = BC.fit_key_base(self.seeds.s91)
        self.setup = BC.BCASetup(RB.LEARNER, self.base, guard.MADE_UP)
        self.refresh_key = BC.refresh_key(self.seeds.s39, STEPS)
        self._steps, self._programs, self._runs = {}, {}, {}

    def start(self, most_caution=False):
        """BCA's state before the first refresh (INTERFACES.md section 10, "Setups from these pieces")."""
        BC = self.BC
        scale = BC.make_scale_state("rebrac", self.built.obs_mean, self.built.obs_std,
                                    BC.scale_init_key(self.base, "rebrac"))
        return BC.initial_state(scale, self.m.n, most_caution=most_caution)

    def carry0(self, most_caution=False):
        return (self.built.rng, self.built.agent_state, self.start(most_caution))

    def step(self, name):
        if name not in self._steps:
            self._steps[name] = self._make(name)
        return self._steps[name]

    def _make(self, name):
        RB, BC, b, a = self.RB, self.BC, self.built, self.args
        T = BC.TestOptions
        if name == "imported":
            return UI.make_unifloral_step(b, a)
        if name == "imported_0015":
            return UI.make_unifloral_step(b, dataclasses.replace(a, actor_bc_coef=0.0015))
        if name == "absent":
            return RB.make_train_step(a, *b.apply_fns, b.dataset)
        if name == "present":
            return RB.make_train_step(a, *b.apply_fns, b.dataset, bca=self.setup)
        if name == "test":
            return RB.make_train_step(a, *b.apply_fns, b.dataset, bca=self.setup, test=T())
        if name == "test_keys":
            return RB.make_train_step(a, *b.apply_fns, b.dataset, bca=self.setup, test=T(unifloral_keys=True))
        if name == "test_dose":
            return RB.make_train_step(a, *b.apply_fns, b.dataset, bca=self.setup, test=T(dose_override=self.m.dose))
        if name == "test_15":
            return RB.make_train_step(a, *b.apply_fns, b.dataset, bca=self.setup,
                                      test=T(dose_override=np.full(self.m.n, 1.5, np.float32)))
        if name == "test_dose_coef100":
            return RB.make_train_step(dataclasses.replace(a, actor_bc_coef=100.0), *b.apply_fns, b.dataset,
                                      bca=self.setup, test=T(dose_override=self.m.dose))
        if name == "test_25":
            return RB.make_train_step(a, *b.apply_fns, b.dataset, bca=self.setup,
                                      test=T(dose_override=np.full(self.m.n, 2.5, np.float32)))
        raise KeyError(name)

    def run(self, name, carry, length):
        """`length` outer steps of step `name` from `carry` in one jax.lax.scan (jitted, compiled once per name and
        length). Returns (final carry, stacked per-step outputs)."""
        key = (name, length)
        if key not in self._programs:
            step, jax = self.step(name), self.jax
            self._programs[key] = jax.jit(lambda c: jax.lax.scan(step, c, None, length=length))
        out = self._programs[key](carry)
        self.jax.block_until_ready(out)
        return out

    def cached(self, key, fn):
        if key not in self._runs:
            self._runs[key] = fn()
        return self._runs[key]

    def imported60(self):
        return self.cached("imported60", lambda: self.run("imported", (self.built.rng, self.built.agent_state), STEPS))

    def absent60(self):
        return self.cached("absent60", lambda: self.run("absent", (self.built.rng, self.built.agent_state), STEPS))

    def present60(self, most_caution=False):
        return self.cached(("present60", most_caution),
                           lambda: self.run("present", self.carry0(most_caution), STEPS))

    def refreshed(self):
        """refresh() after the 60 standard BCA-present steps, on the made-up bank (300 rows)."""
        def go():
            carry, _ = self.present60(False)
            return self.RB.refresh(self.args, self.built.apply_fns, carry[1], carry[2], self.bank, self.refresh_key,
                                   permission=self.guard.MADE_UP)
        return self.cached("refreshed", go)


_FIX = []


def fixture():
    if not _FIX:
        _FIX.append(_Fixture())
    return _FIX[0]


def _check_close(test, rep, rel, what):
    """Bit for bit on the processor; on the card bit for bit or every floating leaf within `rel` (leaf scale) with
    every other leaf equal."""
    if _cpu():
        test.assertTrue(rep["equal"], f"{what}: {_summary(rep)}")
    else:
        test.assertTrue(rep["equal"] or (rep["int_equal"] and rep["max_rel"] <= rel), f"{what}: {_summary(rep)}")


def external_actor_loss(actor_apply, q_apply, actor_params, critic_params, obs, action, m, coef):
    """The actor loss written outside the step (plan 4.5, test (a3)): -lambda mean q + coef mean(m bc), with
    pi = tanh of the actor's output (the deterministic tanh actor, rebrac.py:133-139), q = min(Q1, Q2)(s, pi(s)),
    bc = sum_j (pi_j(s) - a_j)^2, lambda = 1 / (mean |q| + 1e-7) with no gradient (rebrac.py:265-274)."""
    import jax
    import jax.numpy as jnp

    pi_action = jnp.tanh(actor_apply(actor_params, obs).distribution.loc)
    q = q_apply(critic_params, obs, pi_action).min(-1)
    bc = jnp.square(pi_action - action).sum(-1)
    lam = jax.lax.stop_gradient(1.0 / (jnp.abs(q).mean() + 1e-7))
    return -lam * q.mean() + coef * jnp.mean(m * bc)


def _external(F, actor_params, critic_params, idx, m, coef):
    import jax

    actor_apply, q_apply = F.built.apply_fns
    obs, action = F.built.dataset.obs[idx], F.built.dataset.action[idx]
    f = lambda p: external_actor_loss(actor_apply, q_apply, p, critic_params, obs, action, m, coef)
    return jax.value_and_grad(f)(actor_params)


def _outside_doses(F, reference, idx, R, blend=0.5):
    """Each batch row's dose and width computed outside the step: BCA's own single-threshold dose
    (calibration.dose.frozen_level_dose) with the reference's threshold set to the row's own R[idx[b]], one level at a
    time (as shift_iw/test_learners_iw.py:371-386 does), eta from the frozen scale network at the batch rows."""
    import jax
    import jax.numpy as jnp

    from calibration.dose import frozen_level_dose

    scale_apply = F.start().scale.apply_fn
    obs, action = F.built.dataset.obs[idx], F.built.dataset.action[idx]
    eta = jax.jit(scale_apply)(reference.cal_params, obs, action)
    level = jax.jit(lambda ref, e: frozen_level_dose(ref, e, blend))
    Rb = np.asarray(R)[np.asarray(idx)]
    dose = np.full(len(Rb), np.nan, np.float32)
    width = np.full(len(Rb), np.nan, np.float32)
    for r in np.unique(Rb):
        own = level(reference._replace(threshold=jnp.asarray(r, jnp.float32)), eta)
        rows = Rb == r
        dose[rows] = np.asarray(own.dose)[rows]
        width[rows] = np.asarray(own.width)[rows]
    return dose, width, np.asarray(eta)


# ------------------------------------------------------------------------------------------------ tests

class SourceTest(unittest.TestCase):
    """The restated lines are Unifloral's own, verbatim; everything else is marked "# BCA:"."""

    @classmethod
    def setUpClass(cls):
        UI.prepare(device=_device())
        from experiments.shift_hosts import unifloral_rebrac_bca as RB

        cls.RB = RB
        cls.pinned = UI.check_pinned("rebrac")          # the source compared below is the pinned file (stop rule S1)
        cls.lines = (UI.REPO / RB.SOURCE).read_text(encoding="utf8").splitlines()

    @staticmethod
    def _code(lines):
        return [ln.strip() for ln in lines if ln.strip()]

    def test_make_train_step_lines_are_unifloral_lines(self):
        """Dropping the lines marked "# BCA:" from the restated make_train_step leaves rebrac.py:200-309 exactly, text
        and indentation and blank lines included, except Unifloral's def line (199, which gains bca and test) and the
        declared REINDENTED_LINES (the critic scan, 255-260, and the actor loss, 274), each exactly 4 spaces deeper
        inside BCA's else: branches. The indentation-free comparison is kept as well."""
        a, b = self.RB.RESTATED_LINES
        unifloral = self.lines[a - 1:b]
        self.assertTrue(unifloral[0].startswith("def make_train_step(args, actor_apply_fn, q_apply_fn, dataset):"))
        self.assertEqual(unifloral[-1].strip(), "return _train_step")
        mine = inspect.getsource(self.RB.make_train_step).splitlines()
        self.assertTrue(mine[0].startswith("def make_train_step(args, actor_apply_fn, q_apply_fn, dataset, *, "
                                           "bca=None, test=None):"))
        self.assertIn("# BCA:", mine[0])
        kept = [ln for ln in mine[1:] if "# BCA:" not in ln]
        expected = [("    " + ln if i in self.RB.REINDENTED_LINES else ln)
                    for i, ln in enumerate(self.lines[a:b], start=a + 1)]
        self.assertEqual(len(kept), len(expected))
        for i, (got, want) in enumerate(zip(kept, expected), start=a + 1):
            self.assertEqual(got, want, f"rebrac.py:{i}")
        self.assertEqual(sorted(self.RB.REINDENTED_LINES), [255, 256, 257, 258, 259, 260, 274])
        self.assertEqual(self.lines[274 - 1].strip(),
                         "actor_loss = -lambda_ * q.mean() + args.actor_bc_coef * bc_loss.mean()")
        self.assertEqual(self.lines[255 - 1].strip(), "(rng, agent_state), (q_loss, critic_bc_loss) = jax.lax.scan(")
        self.assertEqual(self._code(kept), self._code(unifloral[1:]))
        marked = [ln for ln in mine if "# BCA:" in ln]
        DETAILS["source"] = dict(unifloral_lines=f"rebrac.py:{a}-{b}", unifloral_sha256=self.pinned,
                                 restated_code_lines=len(self._code(mine)), unifloral_code_lines=len(kept) + 1,
                                 bca_marked_lines=len(marked), exact_with_indentation=True,
                                 reindented_lines=sorted(self.RB.REINDENTED_LINES))

    def test_own_target_is_unifloral_compute_target(self):
        """own_target's inner function is rebrac.py:216-235 verbatim."""
        a, b = self.RB.TARGET_LINES
        unifloral = self.lines[a - 1:b]
        self.assertEqual(unifloral[0].strip(), "def _compute_target(rng, transition):")
        mine = inspect.getsource(self.RB.own_target).splitlines()
        first = next(i for i, ln in enumerate(mine) if ln.strip() == "def _compute_target(rng, transition):")
        last = next(i for i, ln in enumerate(mine) if ln.strip().startswith("return transition.reward"))
        self.assertEqual(self._code(mine[first:last + 1]), self._code(unifloral))

    def test_unifloral_pieces_are_the_imported_ones(self):
        U = UI.load("rebrac")
        for name in ("Args", "Transition", "AgentTrainState", "create_train_state", "DualQNetwork",
                     "DeterministicTanhActor"):
            self.assertIs(getattr(self.RB, name), getattr(U, name))
        self.assertIs(self.RB.make_train_step_unifloral, U.make_train_step)
        self.assertEqual(self.RB.BC.FOLDS["rebrac"].refresh_noise, 1380470604)
        self.assertEqual(self.RB.BC.FOLDS["rebrac"].posterior, 1380994899)


class A1CopyEqualsUnifloralTest(unittest.TestCase):
    """(a1) BCA absent: the restated step is Unifloral's imported step, bit for bit (processor and card)."""

    def test_a1_sixty_steps_bit_for_bit(self):
        """60 outer steps from Unifloral's own initializer: every leaf of (key carry, learner state with optimizer
        states) and every per-step loss is bit for bit equal."""
        F = fixture()
        carry_u, loss_u = F.imported60()
        carry_r, loss_r = F.absent60()
        rep_c, rep_l = _compare(carry_r, carry_u), _compare(loss_r, loss_u)
        DETAILS["a1"] = dict(device=_device(), steps=STEPS, carry=_summary(rep_c), losses=_summary(rep_l),
                             loss_keys=sorted(loss_u))
        self.assertTrue(rep_c["equal"], _summary(rep_c))
        self.assertTrue(rep_l["equal"], _summary(rep_l))
        self.assertEqual(sorted(loss_r), sorted(UNIFLORAL_KEYS))
        self.assertFalse(_bits_equal(carry_u[1], F.built.agent_state))          # the learner moved
        self.assertEqual(int(carry_u[1].actor.step), STEPS)
        self.assertTrue(all(np.all(np.isfinite(np.asarray(v))) for v in loss_u.values()))

    def test_a1_same_traced_program(self):
        """With bca=None the restated step traces to the same program as the imported one."""
        F = fixture()
        jax = F.jax
        c = (F.built.rng, F.built.agent_state)
        mine = str(jax.make_jaxpr(F.step("absent"))(c, None))
        theirs = str(jax.make_jaxpr(F.step("imported"))(c, None))
        DETAILS.setdefault("a1", {})["same_jaxpr"] = mine == theirs
        self.assertEqual(mine, theirs)


class A2DoseOneEqualsUnifloralTest(unittest.TestCase):
    """(a2) BCA present before the first refresh (dose exactly 1) changes nothing of Unifloral's learner."""

    def _check(self, most_caution):
        F = fixture()
        carry_u, loss_u = F.imported60()
        carry_b, loss_b = F.present60(most_caution)
        rep_c = _compare(carry_b[:2], carry_u)
        rep_l = _compare({k: loss_b[k] for k in loss_u}, loss_u)
        flags = {k: np.asarray(loss_b[k]) for k in ("inputs_valid", "scale_inputs_valid", "dose_inputs_valid",
                                                     "batch_valid", "dose_in_range", "posterior_ready")}
        bca = carry_b[2]
        start = F.start(most_caution)
        record = dict(device=_device(), steps=STEPS, rule="iw" if most_caution else "standard",
                      carry=_summary(rep_c), losses=_summary(rep_l),
                      every_step_valid=bool(flags["inputs_valid"].all()),
                      dose_min=float(np.min(loss_b["dose_min"])), dose_max=float(np.max(loss_b["dose_max"])),
                      bca_step=int(bca.step), unit_after=float(bca.unit),
                      scale_network_moved=not _bits_equal(bca.scale.params, start.scale.params))
        if not (rep_c["equal"] and rep_l["equal"]):
            record["card_rounding_within_1e-6"] = bool(rep_c["int_equal"] and rep_l["int_equal"]
                                                       and max(rep_c["max_rel"], rep_l["max_rel"]) <= REL_CARD)
        DETAILS[f"a2_{record['rule']}"] = record
        _check_close(self, rep_c, REL_CARD, "a2 carry")
        _check_close(self, rep_l, REL_CARD, "a2 losses")
        for name in ("inputs_valid", "scale_inputs_valid", "dose_inputs_valid", "batch_valid", "dose_in_range"):
            self.assertTrue(flags[name].all(), name)
        self.assertFalse(flags["posterior_ready"].any())
        self.assertTrue(np.all(np.asarray(loss_b["dose_min"]) == 1.0) and np.all(np.asarray(loss_b["dose_max"]) == 1.0))
        self.assertTrue(np.all(np.asarray(loss_b["refused_fraction"]) == 0.0))
        self.assertTrue(np.all(np.asarray(loss_b["support_fraction"]) == 1.0))
        self.assertEqual(int(bca.step), STEPS)
        self.assertTrue(record["scale_network_moved"])                           # the band fit ran at every step
        self.assertNotEqual(float(bca.unit), 1.0)
        self.assertTrue(np.all(np.isfinite(np.asarray(loss_b["scale_loss"]))))
        self.assertEqual(bool(bca.most_caution), most_caution)

    def test_a2_standard_rule(self):
        self._check(False)

    def test_a2_iw_rule(self):
        self._check(True)


class A3RealDoseTest(unittest.TestCase):
    """(a3) A fixed made-up dose in [1, 1.5): at the right place and strength, and nothing else changes."""

    def _one_step(self, name):
        """One outer step of step `name` from Unifloral's start (the imported steps carry (rng, agent_state), the
        restated steps with BCA present also BCA's start)."""
        F = fixture()
        carry = (F.built.rng, F.built.agent_state) if name.startswith("imported") else F.carry0()
        return F.cached(("a3", name), lambda: F.run(name, carry, 1))

    def test_a3_loss_and_gradient_equal_the_external_formula(self):
        """dosed_loss and dosed_grad (the step's actor loss and gradient) equal -lambda mean q + 0.001 mean(m bc)
        computed outside the step at the step's batch, the step-start actor and the critics after both critic
        updates (rebrac.py:263-274), to float32 rounding; the formula without the dose, or with the dose on q, is
        farther away than that."""
        F = fixture()
        carry, out = self._one_step("test_dose")
        o = _step(out, 0)
        idx = np.asarray(o["batch_indices"])
        m = F.m.dose[idx]
        self.assertEqual(np.asarray(o["dose"]).tobytes(), m.tobytes())             # the override, read by indices
        actor0, critic1 = F.built.agent_state.actor.params, carry[1].dual_q.params
        L, G = _external(F, actor0, critic1, idx, m, F.args.actor_bc_coef)
        rel_loss = _rel(o["dosed_loss"], L)
        rep_g = _compare(_host(o["dosed_grad"]), _host(G))
        L1, G1 = _external(F, actor0, critic1, idx, np.ones_like(m), F.args.actor_bc_coef)   # control: no dose
        actor_apply, q_apply = F.built.apply_fns
        obs, action = F.built.dataset.obs[idx], F.built.dataset.action[idx]

        def on_q(p):                                                                   # control: dose on q
            import jax.numpy as jnp

            pi = jnp.tanh(actor_apply(p, obs).distribution.loc)
            q = q_apply(critic1, obs, pi).min(-1)
            lam = F.jax.lax.stop_gradient(1.0 / (jnp.abs(q).mean() + 1e-7))
            return -lam * jnp.mean(m * q) + F.args.actor_bc_coef * jnp.mean(jnp.square(pi - action).sum(-1))

        Lq = on_q(actor0)
        DETAILS["a3_formula"] = dict(dosed_loss=float(o["dosed_loss"]), external=float(L), loss_rel=rel_loss,
                                     grad=_summary(rep_g), control_no_dose_loss_rel=_rel(o["dosed_loss"], L1),
                                     control_no_dose_grad_rel=_compare(_host(o["dosed_grad"]), _host(G1))["max_rel"],
                                     control_dose_on_q_loss_rel=_rel(o["dosed_loss"], Lq),
                                     dose_min=float(m.min()), dose_max=float(m.max()))
        self.assertLessEqual(rel_loss, REL_F32)
        self.assertTrue(rep_g["int_equal"] and rep_g["max_rel"] <= REL_F32, _summary(rep_g))
        self.assertGreater(DETAILS["a3_formula"]["control_no_dose_loss_rel"], REL_F32)
        self.assertGreater(DETAILS["a3_formula"]["control_no_dose_grad_rel"], REL_F32)
        self.assertGreater(DETAILS["a3_formula"]["control_dose_on_q_loss_rel"], REL_F32)
        self.assertTrue(bool(np.asarray(o["inputs_valid"])) and bool(np.asarray(o["dose_in_range"])))
        self.assertTrue(np.all((m >= 1.0) & (m < 1.5)))

    def test_a3_nothing_else_changes(self):
        """The dose touches only the actor: after one step the critics, target critics and key carry equal Unifloral's
        own step's (the critic update does not read actor_bc_coef), while the actor differs."""
        F = fixture()
        carry_d, _ = self._one_step("test_dose")
        carry_u, _ = self._one_step("imported_0015")
        rep = _compare((carry_d[0], carry_d[1].dual_q, carry_d[1].dual_q_target),
                       (carry_u[0], carry_u[1].dual_q, carry_u[1].dual_q_target))
        DETAILS["a3_critics_untouched"] = _summary(rep)
        _check_close(self, rep, REL_CARD, "a3 critics")
        self.assertFalse(_bits_equal(carry_d[1].actor.params, carry_u[1].actor.params))

    def test_a3_cross_check_dose_15_against_coef_0015(self):
        """Every dose 1.5 in the restated step against the imported file with actor_bc_coef = 0.0015, one step: every
        leaf (parameters, optimizer states, key carry) and Unifloral's losses to float32 rounding. Control: dropping
        the dose would move the actor loss by 0.0005 x the mean cloning error, far more than that."""
        F = fixture()
        carry_b, out_b = self._one_step("test_15")
        carry_u, loss_u = self._one_step("imported_0015")
        rep_c = _compare(_host(carry_b[:2]), _host(carry_u))
        rels = {k: _rel(out_b[k], loss_u[k]) for k in UNIFLORAL_KEYS}
        o = _step(out_b, 0)
        no_dose_shift = 0.5 * F.args.actor_bc_coef * float(o["bc_loss"]) / abs(float(o["actor_loss"]))
        DETAILS["a3_cross_check"] = dict(carry=_summary(rep_c), losses_rel=rels, control_no_dose_rel=no_dose_shift)
        self.assertTrue(rep_c["int_equal"] and rep_c["max_rel"] <= REL_F32, _summary(rep_c))
        self.assertTrue(all(r <= REL_F32 for r in rels.values()), rels)
        self.assertGreater(no_dose_shift, REL_F32)
        self.assertTrue(np.all(np.asarray(o["dose"]) == 1.5))
        self.assertTrue(bool(np.asarray(o["inputs_valid"])) and bool(np.asarray(o["dose_in_range"])))

    def test_a3_per_row_placement_at_a_large_coefficient(self):
        """Added check: at actor_bc_coef = 100 the cloning term dominates the loss, so equality with the external
        formula at relative 1e-5 pins each row's dose to its own row; a dose shifted by one row, or applied after the
        mean, is farther away than that."""
        F = fixture()
        carry, out = self._one_step("test_dose_coef100")
        o = _step(out, 0)
        idx = np.asarray(o["batch_indices"])
        m = F.m.dose[idx]
        actor0, critic1 = F.built.agent_state.actor.params, carry[1].dual_q.params
        L, G = _external(F, actor0, critic1, idx, m, 100.0)
        Ls, Gs = _external(F, actor0, critic1, idx, np.roll(m, 1), 100.0)
        Lm, Gm = _external(F, actor0, critic1, idx, np.full_like(m, m.mean()), 100.0)
        rep_g = _compare(_host(o["dosed_grad"]), _host(G))
        rec = dict(loss_rel=_rel(o["dosed_loss"], L), grad=_summary(rep_g),
                   control_shifted_loss_rel=_rel(o["dosed_loss"], Ls),
                   control_shifted_grad_rel=_compare(_host(o["dosed_grad"]), _host(Gs))["max_rel"],
                   control_mean_dose_loss_rel=_rel(o["dosed_loss"], Lm),
                   control_mean_dose_grad_rel=_compare(_host(o["dosed_grad"]), _host(Gm))["max_rel"])
        DETAILS["a3_placement_coef100"] = rec
        self.assertLessEqual(rec["loss_rel"], REL_F32)
        self.assertTrue(rep_g["int_equal"] and rep_g["max_rel"] <= REL_F32, _summary(rep_g))
        self.assertGreater(max(rec["control_shifted_loss_rel"], rec["control_shifted_grad_rel"]), REL_F32)
        self.assertGreater(max(rec["control_mean_dose_loss_rel"], rec["control_mean_dose_grad_rel"]), REL_F32)


class A4OwnTargetAndCriticTest(unittest.TestCase):
    """(a4) The band reads the learner's own target and critic."""

    def test_a4_with_unifloral_keys_y_and_q_are_unifloral_internal_values(self):
        """In test mode with Unifloral's own key, at each of 3 steps: y equals the first critic update's targets and q
        the minimum over the two critics of its q_pred, bit for bit on the processor (the card: recorded, within
        float32 rounding)."""
        F = fixture()
        carry, out = F.cached("a4_keys", lambda: F.run("test_keys", F.carry0(), SHORT))
        rec = []
        for s in range(SHORT):
            o = _step(out, s)
            ry, rq = _compare(o["bca_y"], o["unifloral_y"]), _compare(o["bca_q"], o["unifloral_q"])
            rec.append(dict(step=s + 1, y=_summary(ry), q=_summary(rq)))
            if _cpu():
                self.assertTrue(ry["equal"], (s, _summary(ry)))
                self.assertTrue(rq["equal"], (s, _summary(rq)))
            else:
                self.assertLessEqual(max(ry["max_rel"], rq["max_rel"]), REL_F32)
            y, q = np.asarray(o["bca_y"]), np.asarray(o["bca_q"])
            self.assertTrue(np.all(np.isfinite(y)) and np.all(np.isfinite(q)))
            self.assertGreater(np.std(y), 0.0)
            self.assertFalse(np.array_equal(y, q))
            self.assertTrue(bool(np.asarray(o["inputs_valid"])))
        DETAILS["a4"] = dict(device=_device(), steps=rec)

    def test_a4_control_bca_keys_change_y_only(self):
        """Control: with BCA's own keys (normal runs) y is another noise draw of the same target, so it differs from
        Unifloral's targets, while q still equals Unifloral's q_pred; the key choice never touches Unifloral's own
        learner or key carry."""
        F = fixture()
        carry_k, out_k = F.cached("a4_keys", lambda: F.run("test_keys", F.carry0(), SHORT))
        carry_b, out_b = F.cached("a4_bca", lambda: F.run("test", F.carry0(), SHORT))
        o = _step(out_b, 0)
        self.assertFalse(np.array_equal(np.asarray(o["bca_y"]), np.asarray(o["unifloral_y"])))
        rq = _compare(o["bca_q"], o["unifloral_q"])
        rep = _compare(carry_b[:2], carry_k[:2])
        DETAILS["a4_control"] = dict(q=_summary(rq), unifloral_carry=_summary(rep),
                                     y_mean_abs_difference=float(np.mean(np.abs(np.asarray(o["bca_y"]) -
                                                                                np.asarray(o["unifloral_y"])))))
        _check_close(self, rq, REL_F32, "a4 control q")
        _check_close(self, rep, REL_CARD, "a4 control carry")
        self.assertFalse(_bits_equal(carry_b[2].scale.params, carry_k[2].scale.params))   # the fits did differ


class B2PerRowThresholdTest(unittest.TestCase):
    """(b2) Each batch row reads its own threshold through Unifloral's batch_indices."""

    def test_b2_each_row_reads_its_own_threshold(self):
        """Thresholds at 4 levels ([0.5, 1, 2, 4][row % 4], as shift_iw/test_learners_iw.py:375-377) after a refresh,
        3 steps. Exact on every device: the threshold the step read for batch row b is R[batch_indices[b]]. Computed
        outside the step at R[batch_indices[b]]: eta (the frozen scale network at the row), the width
        R x max(eta, 1e-6) x u and the dose by BCA's own single-threshold rule (calibration.dose.frozen_level_dose, one
        level at a time) equal the step's to float32 rounding, and the rows equal bit for bit are counted (compiled in
        another program, the dose's arithmetic may round one unit in the last place differently). Control: reading R
        at the batch position b instead moves most rows' doses far beyond rounding."""
        F = fixture()
        jnp = F.jnp
        c60, _ = F.present60(False)
        ref = F.refreshed().state
        self.assertTrue(bool(ref.reference.ready))
        R = F.m.R_levels
        state = ref._replace(thresholds=jnp.asarray(R))
        carry, out = F.cached("b2", lambda: F.run("test", (c60[0], c60[1], state), SHORT))
        rec = []
        for s in range(SHORT):
            o = _step(out, s)
            idx = np.asarray(o["batch_indices"])
            got_R, got_eta = np.asarray(o["threshold"]), np.asarray(o["eta"])
            got_w, got_d = np.asarray(o["width"]), np.asarray(o["dose"])
            self.assertEqual(got_R.tobytes(), R[idx].tobytes())                        # the lookup itself: exact
            dose, width, eta = _outside_doses(F, ref.reference, idx, R)
            mis = _misaligned_dose(F, ref.reference, got_eta, R, len(idx))            # control: row b read at R[b]
            levels = {float(r): float(np.mean(got_d[R[idx] == r])) for r in np.unique(R[idx])}
            rec.append(dict(step=s + 1, batch=len(idx), threshold_read_bit_for_bit=True,
                            eta_rel=_rel(got_eta, eta), eta_bit_for_bit=got_eta.tobytes() == eta.tobytes(),
                            width_rel=_rel(got_w, width), width_rows_bit_equal=int(np.sum(got_w == width)),
                            dose_rel=_rel(got_d, dose), dose_rows_bit_equal=int(np.sum(got_d == dose)),
                            dose_max_ulps=_ulps(got_d, dose), mean_dose_per_level=levels,
                            rows_moved_if_misaligned=int(np.sum(np.abs(mis - got_d) > 1e-4))))
            for key in ("eta_rel", "width_rel", "dose_rel"):
                self.assertLessEqual(rec[-1][key], REL_F32, (s, key))
            self.assertEqual(len(levels), 4)
            means = [levels[r] for r in sorted(levels)]
            self.assertTrue(all(a < b for a, b in zip(means, means[1:])), means)      # wider band, larger dose
            self.assertGreater(rec[-1]["rows_moved_if_misaligned"], len(idx) // 2)
            self.assertTrue(bool(np.asarray(o["inputs_valid"])) and bool(np.asarray(o["posterior_ready"])))
            self.assertEqual(float(np.asarray(o["refused_fraction"])), 0.0)
            self.assertTrue(np.all((got_d > 1.0) & (got_d < 1.5)))
        DETAILS["b2"] = dict(device=_device(), threshold_levels=[0.5, 1.0, 2.0, 4.0], steps=rec)


def _ulps(a, b):
    """The largest distance in units in the last place between two float32 arrays of positive finite values."""
    a, b = np.asarray(a, np.float32), np.asarray(b, np.float32)
    fin = np.isfinite(a) & np.isfinite(b)
    if not fin.any():
        return 0
    return int(np.max(np.abs(a[fin].view(np.int32).astype(np.int64) - b[fin].view(np.int32).astype(np.int64))))


def _misaligned_dose(F, reference, eta, R, n):
    """The doses a misaligned lookup would give: batch row b at R[b] (its position) instead of R[batch_indices[b]],
    with the step's own eta."""
    import jax.numpy as jnp

    from calibration.dose import frozen_level_dose

    Rpos = np.asarray(R)[:n]
    out = np.full(n, np.nan, np.float32)
    for r in np.unique(Rpos):
        own = frozen_level_dose(reference._replace(threshold=jnp.asarray(r, jnp.float32)), jnp.asarray(eta), 0.5)
        out[Rpos == r] = np.asarray(own.dose)[Rpos == r]
    return out


class B3RefusedRowsTest(unittest.TestCase):
    """(b3) Refused rows (R = +inf on every third row): standard BCA dose 1, IW BCA dose 1.5; other rows unchanged."""

    def test_b3_standard_and_iw_refusal_rules(self):
        """After a refresh, R = +inf where row % 3 == 0 and 1 elsewhere (as shift_iw/test_learners_iw.py:393-394), 3
        steps under each rule from the same carry. Exact: the thresholds read are R[batch_indices]; refused rows
        (ready and R = +inf) get dose 1 under standard BCA's rule and 1.5 under IW's (choice 16), an infinite width and
        the refused flag; every other row gets the same dose under both rules, bit for bit; refused_fraction is the
        batch's share of refused rows; every step stays valid. Other rows' doses equal BCA's own dose at R = 1
        computed outside the step, to float32 rounding (bit-equal rows counted)."""
        F = fixture()
        jnp = F.jnp
        c60, _ = F.present60(False)
        ref = F.refreshed().state
        R = F.m.R_refused
        runs = {}
        for rule, caution in (("standard", False), ("iw", True)):
            state = ref._replace(thresholds=jnp.asarray(R), most_caution=jnp.asarray(caution))
            runs[rule] = F.cached(("b3", rule), lambda st=state: F.run("test", (c60[0], c60[1], st), SHORT))
        rec = []
        for s in range(SHORT):
            o_s, o_i = _step(runs["standard"][1], s), _step(runs["iw"][1], s)
            idx = np.asarray(o_s["batch_indices"])
            self.assertEqual(idx.tobytes(), np.asarray(o_i["batch_indices"]).tobytes())   # the same batches
            inf = np.isinf(R[idx])
            self.assertTrue(inf.any() and (~inf).any())
            d_s, d_i = np.asarray(o_s["dose"]), np.asarray(o_i["dose"])
            one, _, _ = _outside_doses(F, ref.reference, idx, np.ones_like(R))          # BCA's dose at R = 1
            self.assertTrue(np.all(d_s[inf] == 1.0))                                     # standard BCA's rule
            self.assertTrue(np.all(d_i[inf] == np.float32(1.5)))                         # IW: most caution (choice 16)
            self.assertEqual(d_s[~inf].tobytes(), d_i[~inf].tobytes())                  # other rows: the same dose
            self.assertLessEqual(_rel(d_s[~inf], one[~inf]), REL_F32)
            for o in (o_s, o_i):
                self.assertEqual(np.asarray(o["threshold"]).tobytes(), R[idx].tobytes())
                self.assertEqual(np.asarray(o["refused"]).tolist(), inf.tolist())
                self.assertTrue(np.all(np.isinf(np.asarray(o["width"])[inf])))
                self.assertEqual(float(np.asarray(o["refused_fraction"])), float(np.mean(inf.astype(np.float32))))
                self.assertTrue(bool(np.asarray(o["inputs_valid"])) and bool(np.asarray(o["dose_in_range"])))
            rec.append(dict(step=s + 1, refused_rows=int(inf.sum()), batch=len(idx),
                            standard_refused_dose=sorted(set(d_s[inf].tolist())),
                            iw_refused_dose=sorted(set(d_i[inf].tolist())),
                            other_rows_equal_between_rules=d_s[~inf].tobytes() == d_i[~inf].tobytes(),
                            other_rows_rel_to_dose_at_R1=_rel(d_s[~inf], one[~inf]),
                            other_rows_bit_equal_dose_at_R1=int(np.sum(d_s[~inf] == one[~inf])),
                            other_rows=int((~inf).sum()), max_ulps=_ulps(d_s[~inf], one[~inf])))
        self.assertFalse(_bits_equal(runs["standard"][0][1].actor.params, runs["iw"][0][1].actor.params))
        DETAILS["b3"] = dict(device=_device(), steps=rec)


class RefreshTest(unittest.TestCase):
    """ReBRAC's refresh: bank scores from a stored state equal a direct computation."""

    def test_refresh_scores_equal_a_direct_computation(self):
        """After 60 BCA steps: refresh()'s bank scores |y - q| / (max(eta, 1e-6) u) equal the same scores written
        directly from rebrac.py's arithmetic (target noise from fold_in(refresh key, 1380470604), split per row as
        Unifloral splits, Unifloral's three-way split per row, tanh actor, clipped noise, min of the target critics
        minus 0.01 x the distance to the bank's next command, discount 0.98; q = min(Q1, Q2)(s, a); eta from the live
        scale network), bit for bit on the processor. The threshold reproduces from the returned scores and stream
        (stream = the words of fold_in(refresh key, 1380994899)), and every row's threshold is set to it."""
        F = fixture()
        jax, jnp = F.jax, F.jnp
        from calibration import wbcp

        c60, _ = F.present60(False)
        agent_state, bca = c60[1], c60[2]
        out = F.refreshed()
        self.assertTrue(out.record["valid"] and out.record["certified"], out.record)
        for name, value in (("policy_noise", 0.2), ("noise_clip", 0.5), ("critic_bc_coef", 0.01), ("gamma", 0.98)):
            self.assertEqual(getattr(F.args, name), value)
        rows = F.bank
        n = rows.obs.shape[0]
        actor_apply, q_apply = F.built.apply_fns
        keys = jax.random.split(jax.random.fold_in(F.refresh_key, 1380470604), n)
        rng_noise = jax.vmap(lambda k: jax.random.split(k, 3)[2])(keys)
        eps = jax.vmap(lambda k: jax.random.normal(k, (4,)))(rng_noise)
        noise = jnp.clip(eps * 0.2, -0.5, 0.5)
        a_next = jnp.clip(jnp.tanh(actor_apply(agent_state.actor_target.params, rows.next_obs).distribution.loc)
                          + noise, -1, 1)
        q_next = q_apply(agent_state.dual_q_target.params, rows.next_obs, a_next).min(-1)
        penalty = jnp.square(a_next - rows.next_action).sum(-1)
        y = rows.reward + 0.98 * ((1.0 - rows.done) * (q_next - 0.01 * penalty))
        q = q_apply(agent_state.dual_q.params, rows.obs, rows.action).min(-1)
        eta = bca.scale.apply_fn(bca.scale.params, rows.obs, rows.action)
        scores = np.abs(np.asarray(y - q, np.float64)) / np.asarray(jnp.maximum(eta, 1e-6) * bca.unit, np.float64)
        stream = np.asarray(jax.random.fold_in(F.refresh_key, 1380994899), np.uint32).ravel()
        c = wbcp.calibrate(out.scores, np.random.default_rng(out.stream), alpha=0.1, beta=0.95, draws=1000)
        r = out.state.reference
        rec = dict(device=_device(), bank_rows=n, scores_bits_equal=out.scores.tobytes() == scores.tobytes(),
                   scores_rel=_rel(out.scores, scores), threshold=float(r.threshold), n_eff=float(r.n_eff),
                   lambda_hat=float(r.lambda_hat), lambda_hpd=float(r.lambda_hpd), unit=float(bca.unit),
                   record=out.record)
        DETAILS["refresh"] = rec
        if _cpu():
            self.assertEqual(out.scores.tobytes(), scores.tobytes())
        else:
            self.assertLessEqual(rec["scores_rel"], REL_F32)
        self.assertEqual(out.stream.tobytes(), stream.tobytes())
        for a, b in ((c.threshold, r.threshold), (c.lambda_hat, r.lambda_hat), (c.lambda_hpd, r.lambda_hpd),
                     (c.n_eff, r.n_eff)):
            self.assertEqual(np.float32(a).tobytes(), np.asarray(b, np.float32).tobytes())
        self.assertTrue(np.isfinite(float(r.threshold)) and bool(r.ready))
        self.assertTrue(np.all(np.asarray(out.state.thresholds) == np.asarray(r.threshold)))
        self.assertEqual(out.state.thresholds.shape, (F.m.n,))
        self.assertTrue(_bits_equal(r.cal_params, bca.scale.params))               # the live network, frozen
        self.assertEqual(np.asarray(r.residual_scale).tobytes(), np.asarray(bca.unit, np.float32).tobytes())
        self.assertTrue(_bits_equal((out.state.scale, out.state.unit, out.state.step), (bca.scale, bca.unit, bca.step)))

    def test_band_parts_are_the_refresh_inputs(self):
        """band_parts with the refresh's noise key gives the y and q the direct computation gives (processor: bit for
        bit), and another key gives another y with the same q."""
        F = fixture()
        c60, _ = F.present60(False)
        BC = F.BC
        y, q = F.RB.band_parts(F.args, F.built.apply_fns, c60[1], F.bank,
                               BC.refresh_noise_key(F.refresh_key, "rebrac"))
        y2, q2 = F.RB.band_parts(F.args, F.built.apply_fns, c60[1], F.bank, F.jax.random.PRNGKey(5))
        self.assertEqual(np.asarray(q).tobytes(), np.asarray(q2).tobytes())
        self.assertFalse(np.array_equal(np.asarray(y), np.asarray(y2)))
        residual = np.abs(np.asarray(y - q, np.float64))
        scale = np.asarray(F.jnp.maximum(c60[2].scale.apply_fn(c60[2].scale.params, F.bank.obs, F.bank.action),
                                         1e-6) * c60[2].unit, np.float64)
        if _cpu():
            self.assertEqual((residual / scale).tobytes(), F.refreshed().scores.tobytes())
        else:
            self.assertLessEqual(_rel(residual / scale, F.refreshed().scores), REL_F32)

    def test_refresh_refuses_unregistered_rows_and_wrong_permissions(self):
        F = fixture()
        from experiments.shift_hosts.gate import ApprovalError

        c60, _ = F.present60(False)
        changed = F.bank._replace(reward=F.bank.reward.at[0].add(1.0))
        with self.assertRaises(ApprovalError):
            F.RB.refresh(F.args, F.built.apply_fns, c60[1], c60[2], changed, F.refresh_key,
                         permission=F.guard.MADE_UP)
        for permission in (None, "MADE_UP", True):
            with self.assertRaises(ApprovalError):
                F.RB.refresh(F.args, F.built.apply_fns, c60[1], c60[2], F.bank, F.refresh_key, permission=permission)


class ENextCommandTest(unittest.TestCase):
    """(e) ReBRAC's next command on the made-up arrays is Unifloral's roll (rebrac.py:335)."""

    def test_e_next_commands_are_unifloral_roll(self):
        """For the training rows and the strong-law test log: the Transition's next_action (unifloral_import's
        make_transition), the made-up builder's np.roll and jnp.roll(actions, -1, axis=0) are bit for bit equal; the
        last row wraps to the first; some rows get a substituted next command (removed row or episode end)."""
        F = fixture()
        jnp = F.jnp
        rec = {}
        for which in ("train", "strong"):
            rows = F.m.rows(which)
            rolled = np.asarray(jnp.roll(rows["action"], -1, axis=0))
            tr = UI.make_transition("rebrac", rows)
            self.assertEqual(np.asarray(tr.next_action).tobytes(), rolled.tobytes())
            self.assertEqual(F.m.next_action[which].tobytes(), rolled.tobytes())
            self.assertEqual(rolled.dtype, np.float32)
            self.assertEqual(rolled[-1].tobytes(), np.asarray(rows["action"][0]).tobytes())
            rec[which] = dict(rows=len(rolled), substituted=int(np.sum(~F.m.next_true[which])))
        self.assertEqual(np.asarray(F.built.dataset.next_action).tobytes(), F.m.next_action["train"].tobytes())
        self.assertGreater(int(np.sum(~F.m.next_true["train"][:-1])), 0)
        F.guard.require_made_up(F.built.dataset)          # the Transition the restated step receives is registered
        F.guard.require_made_up(F.bank)
        DETAILS["e"] = rec


class EntryPointTest(unittest.TestCase):
    """The BCA entry point refuses what it must (outside jit)."""

    def test_entry_point_refusals(self):
        F = fixture()
        RB, BC, b, a = F.RB, F.BC, F.built, F.args
        from experiments.shift_hosts.gate import ApprovalError

        changed = b.dataset._replace(reward=b.dataset.reward.at[3].multiply(2.0))
        with self.assertRaises(ApprovalError):                                     # not registered as made-up
            RB.make_train_step(a, *b.apply_fns, changed, bca=F.setup)
        with self.assertRaises(ApprovalError):                                     # no permission
            RB.make_train_step(a, *b.apply_fns, b.dataset, bca=BC.BCASetup("rebrac", F.base, None))
        with self.assertRaises(ValueError):                                        # another learner's setup
            RB.make_train_step(a, *b.apply_fns, b.dataset, bca=BC.BCASetup("td3_bc", F.base, F.guard.MADE_UP))
        with self.assertRaises(ValueError):                                        # test options need BCA
            RB.make_train_step(a, *b.apply_fns, b.dataset, test=BC.TestOptions())
        with self.assertRaises(ValueError):                                        # IQL's option
            RB.make_train_step(a, *b.apply_fns, b.dataset, bca=F.setup, test=BC.TestOptions(width_override=F.m.width))
        for bad in (F.m.dose.astype(np.float64), F.m.dose[:-1], np.where(F.m.dose > 1.2, np.nan, F.m.dose)):
            with self.assertRaises(ValueError):
                RB.make_train_step(a, *b.apply_fns, b.dataset, bca=F.setup, test=BC.TestOptions(dose_override=bad))
        td3 = UI.make_transition("td3_bc", F.m.rows("train"))                    # registered, but not ReBRAC's rows
        with self.assertRaises(ValueError):
            RB.make_train_step(a, *b.apply_fns, td3, bca=F.setup)
        self.assertTrue(callable(RB.make_train_step(a, *b.apply_fns, b.dataset)))   # BCA absent: no checks added


class ValidityTest(unittest.TestCase):
    """The port's per-step validity flag (rebrac_bca.py:394-453): a step that is not valid keeps the whole carry, key
    included, and require_valid reports it (stop rule S5)."""

    def test_invalid_steps_keep_the_whole_carry(self):
        """A non-positive unit makes the band fit invalid: every step is refused and the carry after 3 steps is the
        carry before them, bit for bit (Unifloral's key, learner and optimizer states, BCA's state)."""
        F = fixture()
        start = F.carry0()
        bad = (start[0], start[1], start[2]._replace(unit=F.jnp.asarray(-1.0, F.jnp.float32)))
        carry, out = F.run("test", bad, SHORT)
        self.assertTrue(_bits_equal(carry, bad))
        self.assertFalse(np.any(np.asarray(out["inputs_valid"])))
        self.assertFalse(np.any(np.asarray(out["scale_inputs_valid"])))
        with self.assertRaises(FloatingPointError):
            F.RB.require_valid(out)
        DETAILS["validity_unit"] = dict(steps=SHORT, carry_kept_bit_for_bit=True)

    def test_a_dose_outside_one_to_two_is_refused(self):
        """Every dose 2.5 (outside the port's [1, 2], rebrac_bca.py:416): dose_in_range is false, the step is not
        valid and the carry is kept; doses of 1.5 pass the same check (test (a3))."""
        F = fixture()
        start = F.carry0()
        carry, out = F.run("test_25", start, 1)
        self.assertFalse(bool(np.asarray(out["dose_in_range"])[0]))
        self.assertFalse(bool(np.asarray(out["inputs_valid"])[0]))
        self.assertTrue(bool(np.asarray(out["scale_inputs_valid"])[0]))
        self.assertTrue(_bits_equal(carry, start))
        with self.assertRaisesRegex(FloatingPointError, "inputs_valid is false at step 1"):
            F.RB.require_valid(out)
        ok = dict(inputs_valid=np.ones(4, bool), dose_in_range=np.ones(4, bool))
        F.RB.require_valid(ok)
        DETAILS["validity_dose_range"] = dict(dose=2.5, refused=True)


def tearDownModule():
    if DETAILS:
        from experiments.shift_hosts import handoff as H
        from experiments.shift_hosts import tests as T

        path = T.out_dir(AREA) / f"test_rebrac_details_{_device()}.json"
        H.write_json(path, dict(schema="shift-hosts-stage0-rebrac-details-v1", device=_device(), details=DETAILS))
        print(f"details -> {path}")


if __name__ == "__main__":
    UI.prepare(device=_device())
    from experiments.shift_hosts import tests as T

    sys.exit(T.run(sys.modules[__name__], AREA, device_name=_device(), extra=UI.record))

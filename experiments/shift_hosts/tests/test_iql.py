"""IQL's stage-0 tests (plan section 4.5; area "iql") for experiments/shift_hosts/unifloral_iql_bca.py, on made-up
arrays only (tests/madeup.py, registered with the guard): no ThrusterChain data, no band fit on real arrays, no learned
policy's return. BCA's code runs here on made-up arrays only (approval item 8). An experiment-local attachment (plan
section 3.1) that changes none of the repository's declared methods.

Every case starts from Unifloral's own initializer (unifloral_import.init_like_training_block) with Unifloral's Args
defaults and the plan's seed rule, gamma 0.98 and num_updates 100,000 (plan 3.2), on made-up seed 0.
  (a1)  BCA absent = Unifloral's imported make_train_step: 60 outer steps, bit for bit (every leaf of the learner
        state and optimizer states, the key carry, every per-step loss). Processor and card.
  (a2)  BCA present, never refreshed (IQL's weights exactly Unifloral's): the same 60 steps, bit for bit on the
        processor; on the card bit for bit, or every leaf within 1e-6 relative (recorded). Run without test options.
        Also (processor, test mode): every row's weight is Unifloral's exp_adv bit for bit at every step.
  (a3)  A fixed made-up width per training row (madeup.width): the step's actor loss and gradient equal, to float32
        rounding, mean_i(w_i nll_i) written outside the step with w_i = 1 + A_i / (A_i + U_i) (min(e^(3 A_i), 100) - 1)
        for A_i > 0 and min(e^(3 A_i), 100) otherwise (calibration/advantage.py:32-70; iql.py:245-255), the Gaussian
        negative log density written out by hand and its gradient by jax.grad. No cross-check is named for IQL.
  (a4)  y and q fed to the band fit are Unifloral's q_targets (iql.py:221) and v_target (iql.py:218-219), bit for bit
        inside the step; recomputed outside from the step-start state they agree to rounding; and the vintage controls
        (V after its update, the online critics, the target before averaging) differ by far more than rounding.
  (b2)  Thresholds at 4 levels (madeup.R_levels): each batch row's width is R[batch_indices[b]] x max(eta, 1e-6) x u
        and its weight is the value computed outside at R[batch_indices[b]]; a misaligned lookup would be seen.
  (b3)  Refused rows (madeup.R_refused, +inf on every third row): standard rule weight 0 (unsupported), IW rule weight
        1 where A > 0 and Unifloral's weight elsewhere, every other row unchanged; with every row refused, standard
        skips the actor step (actor, Adam state and step count unchanged) and IW does not.
  refresh   Bank scores computed by refresh() from a stored state equal a direct computation bit for bit; the
        threshold reproduces from the scores and stream; the reference freezes the live scale network and unit; the
        refresh's y and q are what the next step's band reads (choice 15).
  shared critic  The restated step's Q and V never read the actor: on the processor Q, V and the target critic
        (parameters, Adam states, step counts), the key carry and the Q and V losses are bit for bit the imported alone
        run's over 60 steps with a refresh at step 20, for standard and IW, while the actors differ (plan 3.7, 5.4); on
        the card standard and IW share them bit for bit and the alone run agrees to (a2)'s card rule (recorded).
  restated text  The restated make_train_step without its "# BCA:" lines is Unifloral's iql.py:194-268 character for
        character.
  validity  An invalid weight input discards the step whole, random key included; a failed band fit is rejected and
        flagged while the learner's update of that step stands (unifloral_iql_bca.py docstring, "validity").
  guard     BCA's entry points refuse unregistered arrays, a wrong token, another learner's setup; the gate refuses
        real-data runs (no terminal approval exists).

Words (plan 4.5; INTERFACES.md section 3): bit for bit = the same dtype, shape and bytes of every leaf; to float32
rounding = relative difference at most 1e-5, the relative difference of two arrays being max|a - b| / max|b| (per leaf
for trees). On the processor XLA contracts a * b + c into one fused multiply-add inside compiled programs (measured
while building this test: x * y + z compiled and eager differ in the last bit), so values recomputed outside the step
match the step's bits only when they come from an identically compiled program; such outside comparisons are judged
to 1e-6 and their bit equality is recorded as a readout (READOUTS, written into the result file).

Run from /root/bca-hosts in /opt/bca-venv with the Dockerfile's variables (unifloral_import.launch_env()), on cores
16-19 (the processor, then the card; on the card only (a1), (a2) and the shared-critic case run, the rest are
skipped):
  env D4RL_SUPPRESS_IMPORT_ERROR=1 MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa DISPLAY= \
    MUJOCO_PY_MUJOCO_PATH=/root/.mujoco/mujoco210 LD_LIBRARY_PATH=/root/.mujoco/mujoco210/bin \
    PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 JAX_PLATFORMS=cpu SHIFT_HOSTS_DEVICE=cpu \
    taskset -c 16-19 /opt/bca-venv/bin/python -m experiments.shift_hosts.tests.test_iql
  (card: SHIFT_HOSTS_DEVICE=gpu XLA_PYTHON_CLIENT_PREALLOCATE=false and no JAX_PLATFORMS)
Result: runs/shift_hosts/stage0/iql/test_iql_<device>.json (+ .sha256).
"""

import math
import sys
import unittest

import numpy as np

from experiments.shift_hosts import tests as T
from experiments.shift_hosts import unifloral_import as UI

AREA = "iql"
STEPS = 60                  # (a1), (a2): 60 outer steps
REFRESH_AT = 20             # the made-up refresh for (b2), (b3), refresh, shared critic
ROUNDING = 1e-5             # "to float32 rounding" (plan 4.5)
OUTSIDE = 1e-6              # outside recomputations of in-step values
CARD_RELATIVE = 1e-6        # (a2) on the card: every leaf within 1e-6 relative
BETA, CAP = 3.0, 100.0      # Unifloral's beta and exp_adv_clip (iql.py:44-46), checked against Args below
READOUTS = {}
ON_CARD = T.device() == "gpu"
processor_only = unittest.skipIf(ON_CARD, "processor-only case: on the card only (a1), (a2) and the shared critic run")


# ------------------------------------------------------------------------------------------------ comparison helpers

def _leaves(tree):
    import jax

    return [np.asarray(x) for x in jax.tree_util.tree_leaves(tree)]


def _bits(tree):
    return [(a.dtype.str, a.shape, a.tobytes()) for a in _leaves(tree)]


def _same(a, b):
    """Bit for bit: every leaf the same dtype, shape and bytes (INTERFACES.md section 3)."""
    return _bits(a) == _bits(b)


def _same_state(a, b):
    """_same after jnp.asarray on every leaf, so a Python step counter (TrainState.create's 0) and its int32 array
    compare alike (guard.state_sha256's rule)."""
    import jax
    import jax.numpy as jnp

    return _same(jax.tree_util.tree_map(jnp.asarray, a), jax.tree_util.tree_map(jnp.asarray, b))


def _rel(a, b):
    """max|a - b| / max|b| in float64 (0 when equal; inf for a shape, dtype-kind or integer mismatch)."""
    a, b = np.asarray(a), np.asarray(b)
    if a.shape != b.shape:
        return math.inf
    if a.dtype.kind in "biu" or b.dtype.kind in "biu":
        return 0.0 if np.array_equal(a, b) else math.inf
    a, b = a.astype(np.float64), b.astype(np.float64)
    if not (np.all(np.isfinite(a)) and np.all(np.isfinite(b))):
        return 0.0 if np.array_equal(a, b) else math.inf
    d, s = float(np.max(np.abs(a - b), initial=0.0)), float(np.max(np.abs(b), initial=0.0))
    return 0.0 if d == 0.0 else (d / s if s > 0.0 else math.inf)


def _max_rel(a, b):
    """The largest per-leaf _rel over two trees of the same structure."""
    la, lb = _leaves(a), _leaves(b)
    if len(la) != len(lb):
        return math.inf
    return max([_rel(x, y) for x, y in zip(la, lb)], default=0.0)


def _plain(x):
    if isinstance(x, (bool, np.bool_)):
        return bool(x)
    if isinstance(x, (int, np.integer)):
        return int(x)
    if isinstance(x, (float, np.floating)):
        x = float(x)
        return x if math.isfinite(x) else repr(x)
    if isinstance(x, dict):
        return {k: _plain(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_plain(v) for v in x]
    return x


def _record(name, **values):
    READOUTS[name] = _plain(dict(device=T.device(), **values))


def external_weights(adv, width, beta=BETA, cap=CAP):
    """The plan's formula, outside the step, in float64: w = 1 + A / (A + U) (min(e^(beta A), cap) - 1) for A > 0,
    min(e^(beta A), cap) otherwise (plan 4.3; calibration/advantage.py:32-70; iql.py:245). U = +inf gives 1 for A > 0
    (the limit as the width grows without bound)."""
    a, u = np.asarray(adv, np.float64), np.asarray(width, np.float64)
    native = np.minimum(np.exp(beta * a), cap)
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        shrunk = 1.0 + a / (a + u) * (native - 1.0)
    return np.where(a > 0.0, shrunk, native)


def gaussian_nll(actor_apply, params, obs, action):
    """-sum_j log N(a_j; mu_j(s), sigma_j), written out by hand from the actor's mean and spread (iql.py:117-132, 251)."""
    import jax.numpy as jnp

    pi = actor_apply(params, obs)
    z = (action - pi.loc) / pi.scale
    return jnp.sum(0.5 * z * z + jnp.log(pi.scale) + 0.5 * jnp.log(2.0 * jnp.pi), axis=-1)


# ------------------------------------------------------------------------------------------------ fixture

class Fixture:
    """Made-up seed 0, Unifloral's initializer, BCA's starting state and the steps under test (built once)."""

    def __init__(self):
        import jax

        from experiments.shift_hosts import bca_core as BC
        from experiments.shift_hosts import guard
        from experiments.shift_hosts import unifloral_iql_bca as IQ
        from experiments.shift_hosts.tests import madeup

        self.jax, self.BC, self.IQ, self.guard, self.U = jax, BC, IQ, guard, UI.load("iql")
        self.m = madeup.build(0)
        self.seeds = BC.seeds(0)
        self.args = UI.args_for("iql", seed=self.seeds.unifloral, gamma=0.98, num_updates=100_000)
        if (self.args.beta, self.args.exp_adv_clip, self.args.batch_size) != (BETA, CAP, 256):
            raise AssertionError("Unifloral's IQL defaults changed")
        self.built = UI.init_like_training_block("iql", self.args, self.m.rows("train"))
        self.bank = UI.make_transition("iql", self.m.rows("bank"))
        self.base = BC.fit_key_base(self.seeds.s91)
        self.scale = BC.make_scale_state("iql", self.built.obs_mean, self.built.obs_std,
                                         BC.scale_init_key(self.base, "iql"))
        self.setup = BC.BCASetup("iql", self.base, guard.MADE_UP)
        self._steps, self._jit, self._cache = {}, {}, {}

    # steps -------------------------------------------------------------------------------------------------------
    def step(self, name):
        """'alone' Unifloral's imported step; 'absent' the restated step with BCA absent; 'std' BCA present (no test
        options; the IW setup uses the same program with other carried values); 'test' test mode; 'width' test mode
        with madeup.width as the width override."""
        if name not in self._steps:
            b, a, BC = self.built, self.args, self.BC
            if name == "alone":
                fn = UI.make_unifloral_step(b, a)
            elif name == "absent":
                fn = self.IQ.make_train_step(a, *b.apply_fns, b.dataset)
            elif name == "std":
                fn = self.IQ.make_train_step(a, *b.apply_fns, b.dataset, bca=self.setup)
            elif name == "test":
                fn = self.IQ.make_train_step(a, *b.apply_fns, b.dataset, bca=self.setup,
                                             test=BC.TestOptions(unifloral_keys=True))
            elif name == "width":
                fn = self.IQ.make_train_step(a, *b.apply_fns, b.dataset, bca=self.setup,
                                             test=BC.TestOptions(width_override=self.m.width))
            else:
                raise KeyError(name)
            self._steps[name] = fn
        return self._steps[name]

    def jstep(self, name):
        """One outer step, compiled on its own (jax.jit of the step)."""
        if name not in self._jit:
            self._jit[name] = self.jax.jit(self.step(name))
        return self._jit[name]

    def scan(self, name, carry, n):
        """n outer steps in one lax.scan, as the training block's chunk loop runs them (iql.py:324-329)."""
        return self.jax.lax.scan(self.step(name), carry, None, n)

    def start(self, most_caution=False):
        """(rng, agent_state, bca_state) at Unifloral's initializer; BCA's state not ready (bca_core.initial_state)."""
        return (self.built.rng, self.built.agent_state,
                self.BC.initial_state(self.scale, self.m.n, most_caution=most_caution))

    # cached runs ---------------------------------------------------------------------------------------------------
    def alone_run(self):
        """60 steps of Unifloral's imported step from its initializer."""
        if "alone60" not in self._cache:
            self._cache["alone60"] = self.scan("alone", (self.built.rng, self.built.agent_state), 60)
        return self._cache["alone60"]

    def sham_run(self):
        """60 steps of the restated step with BCA present and never refreshed from the start: (a2)'s run, and choice
        19's "sham alone" (the BCA setups' compiled program with weights exactly Unifloral's)."""
        if "sham60" not in self._cache:
            self._cache["sham60"] = self.scan("std", self.start(), 60)
        return self._cache["sham60"]

    def refreshed(self):
        """20 outer steps of the standard step from the start, then a refresh on the made-up bank at the key
        refresh_key(s39, 20): (carry after 20 steps, Refreshed, refresh key, losses of the 20 steps)."""
        if "refreshed" not in self._cache:
            carry, out = self.scan("std", self.start(), REFRESH_AT)
            rk = self.BC.refresh_key(self.seeds.s39, REFRESH_AT)
            r = self.IQ.refresh(self.args, self.built.apply_fns, carry[1], carry[2], self.bank, rk,
                                permission=self.guard.MADE_UP)
            self._cache["refreshed"] = (carry, r, rk, out)
        return self._cache["refreshed"]

    # outside computations ------------------------------------------------------------------------------------------
    def outside(self, carry):
        """Unifloral's own lines restated in this test and compiled on their own, from a step-start state: the batch
        draw (iql.py:200-204), the target averaging (207-211), the targets (218-221) and the advantage (232); plus
        the vintage controls (the online critics, the target before averaging)."""
        if "outside" not in self._jit:
            jax, args, ds = self.jax, self.args, self.built.dataset
            _, q_apply, v_apply = self.built.apply_fns

            def f(rng, agent_state):
                import optax

                rng, rng_batch = jax.random.split(rng)
                idx = jax.random.randint(rng_batch, (args.batch_size,), 0, len(ds.obs))
                batch = jax.tree_util.tree_map(lambda x: x[idx], ds)
                target = optax.incremental_update(agent_state.dual_q.params, agent_state.dual_q_target.params,
                                                  args.polyak_step_size)
                heads = q_apply(target, batch.obs, batch.action)
                v_target = heads.min(-1)
                next_v = v_apply(agent_state.value.params, batch.next_obs)
                y = batch.reward + args.gamma * (1 - batch.done) * next_v
                adv = v_target - v_apply(agent_state.value.params, batch.obs)
                online = q_apply(agent_state.dual_q.params, batch.obs, batch.action).min(-1)
                stored = q_apply(agent_state.dual_q_target.params, batch.obs, batch.action).min(-1)
                return dict(idx=idx, batch=batch, heads=heads, v_target=v_target, y=y, adv=adv, online=online,
                            stored_target=stored)

            self._jit["outside"] = jax.jit(f)
        return self._jit["outside"](carry[0], carry[1])


_FIXTURE = []


def fixture():
    UI.prepare()
    if not _FIXTURE:
        _FIXTURE.append(Fixture())
    return _FIXTURE[0]


def _unifloral_losses(out):
    from experiments.shift_hosts import unifloral_iql_bca as IQ

    return {k: out[k] for k in IQ.UNIFLORAL_LOSSES}


# ------------------------------------------------------------------------------------------------ (a1), (a2)

class A1A2Test(unittest.TestCase):
    """(a1) and (a2): on the processor and on the card."""

    @classmethod
    def setUpClass(cls):
        cls.F = fixture()

    def test_a1_bca_absent_equals_imported_step(self):
        """(a1) BCA absent: the restated step equals Unifloral's imported make_train_step bit for bit over 60 outer
        steps (learner state, optimizer states, key carry, per-step losses)."""
        F = self.F
        ref_carry, ref_loss = F.alone_run()
        carry, loss = F.scan("absent", (F.built.rng, F.built.agent_state), STEPS)
        same_key, same_state, same_loss = _same(carry[0], ref_carry[0]), _same(carry[1], ref_carry[1]), \
            _same(loss, ref_loss)
        moved = not _same(ref_carry[1].actor.params, F.built.agent_state.actor.params)
        finite = all(bool(np.all(np.isfinite(np.asarray(v)))) for v in ref_loss.values())
        _record("a1", steps=STEPS, leaves=len(_leaves(carry)), bit_for_bit_key=same_key, bit_for_bit_state=same_state,
                bit_for_bit_losses=same_loss, max_relative=_max_rel((carry, loss), (ref_carry, ref_loss)),
                state_moved=moved, losses_finite=finite, loss_keys=sorted(loss))
        self.assertTrue(moved and finite)
        self.assertEqual(sorted(loss), ["actor_loss", "q_loss", "value_loss"])
        self.assertTrue(same_key, "key carry differs")
        self.assertTrue(same_state, "learner state differs")
        self.assertTrue(same_loss, "per-step losses differ")

    def test_a2_bca_present_before_refresh_equals_imported_step(self):
        """(a2) BCA present, never refreshed (no test options): the learner state, optimizer states, key carry and
        Unifloral's per-step losses equal the imported step's over 60 steps: bit for bit on the processor; on the card
        bit for bit or within 1e-6 relative on every leaf (recorded). BCA itself ran: every step valid, the scale
        network fitted, the actor updated every step, no refusal."""
        F = self.F
        ref_carry, ref_loss = F.alone_run()
        carry, out = F.sham_run()
        losses = _unifloral_losses(out)
        same_key, same_state, same_loss = _same(carry[0], ref_carry[0]), _same(carry[1], ref_carry[1]), \
            _same(losses, ref_loss)
        rel_state, rel_loss = _max_rel(carry[1], ref_carry[1]), _max_rel(losses, ref_loss)
        bca_after = carry[2]
        flags = {k: bool(np.all(np.asarray(out[k]))) for k in ("inputs_valid", "scale_inputs_valid",
                                                               "dose_inputs_valid", "actor_updated")}
        scale_moved = not _same(bca_after.scale.params, F.scale.params)
        _record("a2", steps=STEPS, bit_for_bit_key=same_key, bit_for_bit_state=same_state,
                bit_for_bit_losses=same_loss, max_relative_state=rel_state, max_relative_losses=rel_loss,
                per_group_relative={name: _max_rel(getattr(carry[1], name), getattr(ref_carry[1], name))
                                    for name in ("actor", "dual_q", "dual_q_target", "value")},
                flags=flags, posterior_ready_any=bool(np.any(np.asarray(out["posterior_ready"]))),
                supported_fraction_min=float(np.min(np.asarray(out["supported_fraction"]))),
                refused_fraction_max=float(np.max(np.asarray(out["refused_fraction"]))),
                bca_step=int(np.asarray(bca_after.step)), unit_after=float(np.asarray(bca_after.unit)),
                scale_network_moved=scale_moved, scale_steps=int(np.asarray(bca_after.scale.step)),
                weight_mean_range=[float(np.min(np.asarray(out["weight_mean"]))),
                                   float(np.max(np.asarray(out["weight_mean"])))],
                card_rule=None if not ON_CARD else f"bit for bit, or every leaf within {CARD_RELATIVE} relative")
        self.assertTrue(all(flags.values()), flags)
        self.assertFalse(bool(np.any(np.asarray(out["posterior_ready"]))))
        self.assertEqual(float(np.min(np.asarray(out["supported_fraction"]))), 1.0)
        self.assertEqual(float(np.max(np.asarray(out["refused_fraction"]))), 0.0)
        self.assertEqual(int(np.asarray(bca_after.step)), STEPS)
        self.assertTrue(scale_moved and float(np.asarray(bca_after.unit)) != 1.0)
        self.assertTrue(same_key, "key carry differs")
        if ON_CARD:
            self.assertTrue((same_state and same_loss) or (rel_state <= CARD_RELATIVE),
                            f"card: largest relative leaf difference {rel_state} > {CARD_RELATIVE}")
        else:
            self.assertTrue(same_state, f"learner state differs (largest relative {rel_state})")
            self.assertTrue(same_loss, f"Unifloral's losses differ (largest relative {rel_loss})")

    @processor_only
    def test_a2_weights_are_unifloral_exp_adv_in_test_mode(self):
        """(a2), supplement (test mode, processor): before any refresh every row's weight is Unifloral's own
        jnp.exp(adv * beta).clip(max=100), bit for bit, at every one of the 60 steps; widths 0, every row supported,
        none refused."""
        F = self.F
        carry, out = F.scan("test", F.start(), STEPS)
        w, wu = np.asarray(out["weight"]), np.asarray(out["unifloral_weight"])
        self.assertEqual(w.shape, (STEPS, F.args.batch_size))
        ref_carry, _ = F.alone_run()
        _record("a2_test_mode", weights_bit_for_bit=_same(w, wu), widths_all_zero=bool(np.all(np.asarray(out["width"]) == 0)),
                carry_equals_alone=_same(carry[:2], ref_carry), adv_positive_share=float(np.mean(np.asarray(out["adv"]) > 0)),
                weight_range=[float(w.min()), float(w.max())])
        self.assertTrue(_same(w, wu))
        self.assertTrue(np.all(np.asarray(out["width"]) == 0.0))
        self.assertTrue(np.all(np.asarray(out["support_mask"])) and np.all(np.asarray(out["has_support"])))
        self.assertFalse(np.any(np.asarray(out["refused"])))
        self.assertTrue(0.05 < float(np.mean(np.asarray(out["adv"]) > 0)) < 0.95)   # both signs occur


# ------------------------------------------------------------------------------------------------ (a3), (a4)

@processor_only
class A3A4Test(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.F = fixture()

    def test_a3_fixed_width_loss_and_gradient_equal_the_formula(self):
        """(a3) A fixed made-up width per training row (read through batch_indices): the step's actor loss and its
        gradient equal mean_i(w_i nll_i) written outside the step, w by the plan's formula, to float32 rounding, at 3
        consecutive steps; rows with A <= 0 keep Unifloral's weight bit for bit; rows with A > 0 are shrunk."""
        import jax
        import jax.numpy as jnp

        F = self.F
        actor_apply = F.built.apply_fns[0]
        js, carry, rows = F.jstep("width"), F.start(), []
        for i in range(3):
            new, out = js(carry, None)
            o = F.outside(carry)
            idx = np.asarray(out["batch_indices"])
            self.assertTrue(np.array_equal(idx, np.asarray(o["idx"])), "batch draw differs")
            adv_in, adv_out = np.asarray(out["adv"]), np.asarray(o["adv"])
            width = np.asarray(F.m.width)[idx]
            self.assertTrue(np.array_equal(np.asarray(out["width"]), width))
            w_ext = external_weights(adv_out, width)
            w_in = np.asarray(out["weight"])
            pos, neg = adv_in > 0.0, adv_in <= 0.0
            wu = np.asarray(out["unifloral_weight"])
            params, batch = carry[1].actor.params, o["batch"]
            w32 = jnp.asarray(w_ext, jnp.float32)

            def loss_ext(p):
                return jnp.mean(w32 * gaussian_nll(actor_apply, p, batch.obs, batch.action))

            L, G = jax.value_and_grad(loss_ext)(params)
            L_unif = float(jnp.mean(jnp.asarray(wu) * gaussian_nll(actor_apply, params, batch.obs, batch.action)))
            rec = dict(step=i + 1, adv_outside_rel=_rel(adv_in, adv_out), adv_bits=_same(adv_in, adv_out),
                       weight_rel=_rel(w_in, w_ext), loss=float(np.asarray(out["dosed_loss"])), loss_formula=float(L),
                       loss_rel=_rel(out["dosed_loss"], L), grad_rel=_max_rel(out["dosed_grad"], G),
                       loss_with_unifloral_weights=L_unif, positive_rows=int(pos.sum()),
                       shrunk_rows=int(np.sum(pos & (wu > 1.0) & (w_in < wu))),
                       mean_weight=float(w_in.mean()), mean_unifloral_weight=float(wu.mean()))
            rows.append(rec)
            self.assertLessEqual(rec["adv_outside_rel"], OUTSIDE)
            self.assertLessEqual(rec["weight_rel"], ROUNDING)
            self.assertLessEqual(rec["loss_rel"], ROUNDING)
            self.assertLessEqual(rec["grad_rel"], ROUNDING)
            self.assertTrue(_same(w_in[neg], wu[neg]), "rows with A <= 0 must keep Unifloral's weight")
            self.assertTrue(np.all(w_in[pos] >= 1.0) and np.all(w_in[pos] <= wu[pos]))
            self.assertGreater(rec["shrunk_rows"], 10)
            self.assertGreater(abs(L_unif - float(L)) / abs(float(L)), 1e-3)   # the weights changed the loss
            self.assertTrue(bool(np.asarray(out["inputs_valid"])) and bool(np.asarray(out["actor_updated"])))
            carry = new
        _record("a3", steps=rows, rule="relative difference max|a-b|/max|b| <= 1e-5 (per leaf for the gradient)")

    def test_a4_band_reads_the_learners_own_target_and_critic(self):
        """(a4) In test mode (with Unifloral's keys; IQL's target has no randomness): the fit's y is Unifloral's
        q_targets (iql.py:221) and its q is Unifloral's v_target (iql.py:218-219), bit for bit; recomputed outside
        from the step-start state they agree to rounding; at a state 20 steps in, V after its update, the online
        critics and the target before averaging all differ by far more than rounding (the vintage is the one the
        plan names)."""
        F = self.F
        _, q_apply, v_apply = F.built.apply_fns
        carry20, _, _, _ = F.refreshed()
        starts = [("start", F.start()), ("step20", (carry20[0], carry20[1], F.start()[2]))]
        js, rows = F.jstep("test"), []
        for label, carry in starts:
            for i in range(2):
                new, out = js(carry, None)
                o = F.outside(carry)
                b = o["batch"]
                y_in, q_in = np.asarray(out["bca_y"]), np.asarray(out["bca_q"])
                self.assertTrue(_same(y_in, out["unifloral_y"]), "y is not Unifloral's q_targets")
                self.assertTrue(_same(q_in, out["unifloral_q"]), "q is not Unifloral's v_target")
                self.assertTrue(_same(out["bca_heads"], o["heads"]) or _rel(out["bca_heads"], o["heads"]) <= OUTSIDE)
                y_new = np.asarray(b.reward + F.args.gamma * (1 - b.done) * v_apply(new[1].value.params, b.next_obs))
                rec = dict(state=label, step=i + 1, y_bits=_same(y_in, o["y"]), q_bits=_same(q_in, o["v_target"]),
                           y_rel=_rel(y_in, o["y"]), q_rel=_rel(q_in, o["v_target"]),
                           control_y_after_v_update=_rel(y_new, y_in),
                           control_q_online=_rel(o["online"], q_in),
                           control_q_target_before_averaging=_rel(o["stored_target"], q_in))
                rows.append(rec)
                self.assertLessEqual(rec["y_rel"], OUTSIDE)
                self.assertLessEqual(rec["q_rel"], OUTSIDE)
                if label == "step20":
                    for c in ("control_y_after_v_update", "control_q_online", "control_q_target_before_averaging"):
                        self.assertGreater(rec[c], 100 * OUTSIDE, c)
                carry = new
        _record("a4", rows=rows, note="IQL's band target has no randomness: unifloral_keys changes nothing")


# ------------------------------------------------------------------------------------------------ (b2), (b3)

@processor_only
class B2B3Test(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.F = fixture()
        from experiments.shift_hosts import iw

        cls.IW = iw
        carry20, r, rk, _ = cls.F.refreshed()
        cls.carry20, cls.r = carry20, r
        if not (r.record["valid"] and r.record["certified"]):
            raise AssertionError(f"the made-up refresh did not certify: {r.record}")

    def _outside_weights(self, ref, eta, R_rows, adv, most_caution):
        import jax
        import jax.numpy as jnp

        f = jax.jit(lambda e, R, a, mc: self.IW.per_row_iql_weights(ref, e, R, a, BETA, CAP, mc))
        w, refused = f(jnp.asarray(eta), jnp.asarray(R_rows), jnp.asarray(adv), jnp.asarray(bool(most_caution)))
        return np.asarray(w.weights), np.asarray(refused), np.asarray(w.support_mask)

    def test_b2_each_rows_own_threshold(self):
        """(b2) Thresholds at 4 levels across training rows: each batch row's width is R[batch_indices[b]] x
        max(eta, 1e-6) x u, and its weight equals the value computed outside the step at R[batch_indices[b]] (by
        iw.per_row_iql_weights on the step's own eta and advantage, and by the plan's formula in float64); the same
        computation at a misaligned row is far off."""
        import jax.numpy as jnp

        F = self.F
        R = np.asarray(F.m.R_levels)
        state = self.r.state._replace(thresholds=jnp.asarray(R))
        ref = state.reference
        u = float(np.asarray(ref.residual_scale))
        js, carry, rows = F.jstep("test"), (self.carry20[0], self.carry20[1], state), []
        for i in range(3):
            new, out = js(carry, None)
            idx = np.asarray(out["batch_indices"])
            eta, adv, w_in = np.asarray(out["eta"]), np.asarray(out["adv"]), np.asarray(out["weight"])
            scale = np.maximum(eta.astype(np.float64), 1e-6) * u
            levels = np.asarray(out["width"], np.float64) / scale
            w_iw, refused, support = self._outside_weights(ref, eta, R[idx], adv, False)
            w_ext = external_weights(adv, R[idx].astype(np.float64) * scale)
            shifted = (idx + 1) % F.m.n
            w_mis = external_weights(adv, R[shifted].astype(np.float64) * scale)
            pos = adv > 0
            moved = pos & (R[idx] != R[shifted])
            rec = dict(step=i + 1, level_rel=_rel(levels, R[idx]), weight_bits_vs_iw=_same(w_in, w_iw),
                       weight_rel_vs_iw=_rel(w_in, w_iw), weight_rel_vs_formula=_rel(w_in, w_ext),
                       misaligned_rel=_rel(w_in[moved], w_mis[moved]) if moved.any() else None,
                       levels_in_batch=sorted(set(float(x) for x in R[idx])), positive_rows=int(pos.sum()))
            rows.append(rec)
            self.assertLessEqual(rec["level_rel"], 1e-6)
            self.assertLessEqual(rec["weight_rel_vs_iw"], OUTSIDE)
            self.assertLessEqual(rec["weight_rel_vs_formula"], ROUNDING)
            self.assertEqual(rec["levels_in_batch"], [0.5, 1.0, 2.0, 4.0])
            self.assertGreater(rec["misaligned_rel"], 1e-3)
            self.assertFalse(np.any(np.asarray(out["refused"])) or np.any(refused))
            self.assertTrue(np.all(np.asarray(out["support_mask"])) and bool(np.asarray(out["actor_updated"])))
            carry = new
        _record("b2", rows=rows, threshold_levels=[0.5, 1.0, 2.0, 4.0], reference_threshold=self.r.record["threshold"])

    def test_b3_refused_rows_standard_and_iw_rules(self):
        """(b3) R = +inf on every third training row (a ready band): standard BCA gives those rows weight 0
        (unsupported); IW gives weight 1 where A > 0 and Unifloral's weight elsewhere; every other row has the same
        weight under both rules and equals the formula at its own threshold; Q, V and the target critic do not depend
        on the rule."""
        import jax.numpy as jnp

        F = self.F
        R = np.asarray(F.m.R_refused)
        base = self.r.state._replace(thresholds=jnp.asarray(R))
        u = float(np.asarray(base.reference.residual_scale))
        js = F.jstep("test")
        new_s, out_s = js((self.carry20[0], self.carry20[1], base._replace(most_caution=jnp.asarray(False))), None)
        new_i, out_i = js((self.carry20[0], self.carry20[1], base._replace(most_caution=jnp.asarray(True))), None)
        idx = np.asarray(out_s["batch_indices"])
        self.assertTrue(np.array_equal(idx, np.asarray(out_i["batch_indices"])))
        adv = np.asarray(out_s["adv"])
        self.assertTrue(_same(adv, out_i["adv"]))
        refused_rows = np.isinf(R[idx])
        pos = adv > 0
        ws, wi, wu = np.asarray(out_s["weight"]), np.asarray(out_i["weight"]), np.asarray(out_s["unifloral_weight"])
        scale = np.maximum(np.asarray(out_s["eta"], np.float64), 1e-6) * u
        w_ext = external_weights(adv, np.where(refused_rows, np.inf, R[idx].astype(np.float64)) * scale)
        rec = dict(refused_rows=int(refused_rows.sum()), refused_positive=int((refused_rows & pos).sum()),
                   refused_nonpositive=int((refused_rows & ~pos).sum()),
                   standard_refused_all_zero=bool(np.all(ws[refused_rows] == 0.0)),
                   standard_refused_unsupported=bool(not np.any(np.asarray(out_s["support_mask"])[refused_rows])),
                   iw_refused_positive_all_one=bool(np.all(wi[refused_rows & pos] == 1.0)),
                   iw_refused_nonpositive_unifloral=_same(wi[refused_rows & ~pos], wu[refused_rows & ~pos]),
                   other_rows_same_bits=_same(ws[~refused_rows], wi[~refused_rows]),
                   other_rows_rel_formula=_rel(ws[~refused_rows], w_ext[~refused_rows]),
                   iw_refused_rel_formula_positive=_rel(wi[refused_rows & pos], w_ext[refused_rows & pos]),
                   refused_flag_matches=bool(np.array_equal(np.asarray(out_s["refused"]), refused_rows)
                                             and np.array_equal(np.asarray(out_i["refused"]), refused_rows)),
                   shared_critic_bits=_same(F.IQ.critic_value_parts(new_s[1]), F.IQ.critic_value_parts(new_i[1])),
                   actors_differ=not _same(new_s[1].actor.params, new_i[1].actor.params),
                   refused_fraction=[float(np.asarray(out_s["refused_fraction"])),
                                     float(np.asarray(out_i["refused_fraction"]))])
        _record("b3", **rec)
        self.assertGreater(rec["refused_positive"], 5)
        self.assertGreater(rec["refused_nonpositive"], 5)
        for k in ("standard_refused_all_zero", "standard_refused_unsupported", "iw_refused_positive_all_one",
                  "iw_refused_nonpositive_unifloral", "other_rows_same_bits", "refused_flag_matches",
                  "shared_critic_bits", "actors_differ"):
            self.assertTrue(rec[k], k)
        self.assertLessEqual(rec["other_rows_rel_formula"], ROUNDING)
        self.assertEqual(rec["iw_refused_rel_formula_positive"], 0.0)   # the formula's limit is exactly 1
        for out in (out_s, out_i):
            self.assertTrue(bool(np.asarray(out["inputs_valid"])) and bool(np.asarray(out["actor_updated"])))
            self.assertTrue(bool(np.asarray(out["has_support"])))

    def test_b3_no_supported_row_skips_the_actor_step(self):
        """(b3) Every training row refused: standard BCA has no supported row, so the actor step is skipped (its
        parameters, Adam state and step count stay as they were, so its cosine schedule does not advance), the step
        stays valid, and Q, V and the target critic move exactly as Unifloral's step moves them; under IW the actor
        step runs with weight 1 where A > 0 and Unifloral's weight elsewhere."""
        import jax.numpy as jnp

        F = self.F
        all_inf = jnp.full((F.m.n,), jnp.inf, jnp.float32)
        base = self.r.state._replace(thresholds=all_inf)
        rng20, agent20 = self.carry20[0], self.carry20[1]
        js = F.jstep("test")
        new_s, out_s = js((rng20, agent20, base._replace(most_caution=jnp.asarray(False))), None)
        new_i, out_i = js((rng20, agent20, base._replace(most_caution=jnp.asarray(True))), None)
        alone_new, alone_loss = F.jstep("alone")((rng20, agent20), None)
        adv, wi, wu = np.asarray(out_i["adv"]), np.asarray(out_i["weight"]), np.asarray(out_i["unifloral_weight"])
        pos = adv > 0
        crit = F.IQ.critic_value_parts
        rec = dict(standard_has_support=bool(np.asarray(out_s["has_support"])),
                   standard_actor_updated=bool(np.asarray(out_s["actor_updated"])),
                   standard_inputs_valid=bool(np.asarray(out_s["inputs_valid"])),
                   standard_weights_all_zero=bool(np.all(np.asarray(out_s["weight"]) == 0.0)),
                   standard_actor_unchanged=_same(new_s[1].actor, agent20.actor),
                   standard_actor_step=int(np.asarray(new_s[1].actor.step)),
                   start_actor_step=int(np.asarray(agent20.actor.step)),
                   standard_bca_step=int(np.asarray(new_s[2].step)),
                   standard_critic_value_rel_vs_alone=_max_rel(crit(new_s[1]), crit(alone_new[1])),
                   standard_critic_value_bits_vs_alone=_same(crit(new_s[1]), crit(alone_new[1])),
                   standard_key_bits_vs_alone=_same(new_s[0], alone_new[0]),
                   iw_has_support=bool(np.asarray(out_i["has_support"])),
                   iw_actor_updated=bool(np.asarray(out_i["actor_updated"])),
                   iw_actor_moved=not _same(new_i[1].actor.params, agent20.actor.params),
                   iw_positive_all_one=bool(np.all(wi[pos] == 1.0)), iw_nonpositive_unifloral=_same(wi[~pos], wu[~pos]),
                   iw_actor_step=int(np.asarray(new_i[1].actor.step)))
        _record("b3_skip", **rec)
        self.assertFalse(rec["standard_has_support"] or rec["standard_actor_updated"])
        self.assertTrue(rec["standard_inputs_valid"] and rec["standard_weights_all_zero"])
        self.assertTrue(rec["standard_actor_unchanged"])
        self.assertEqual(rec["standard_actor_step"], rec["start_actor_step"])
        self.assertEqual(rec["standard_bca_step"], REFRESH_AT + 1)
        self.assertLessEqual(rec["standard_critic_value_rel_vs_alone"], OUTSIDE)
        self.assertTrue(rec["standard_key_bits_vs_alone"])
        for k in ("iw_has_support", "iw_actor_updated", "iw_actor_moved", "iw_positive_all_one",
                  "iw_nonpositive_unifloral"):
            self.assertTrue(rec[k], k)
        self.assertEqual(rec["iw_actor_step"], rec["start_actor_step"] + 1)


# ------------------------------------------------------------------------------------------------ refresh

@processor_only
class RefreshTest(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.F = fixture()

    def test_refresh_bank_scores_equal_a_direct_computation(self):
        """refresh(): from the stored state after 20 steps, the bank scores equal |y - q| / (u max(eta, 1e-6)) computed
        directly here (y = r + 0.98 (1 - d) V(s') with the stored V; q = min over the heads of Polyak(stored target,
        stored critic, 0.005); eta from the live scale network), bit for bit; an unweighted wbcp.calibrate on the
        scores and stream reproduces the threshold; the posterior stream is fold_in(refresh key, 1347375956); the
        reference holds the live scale network and unit; every row's threshold is the standard one."""
        import jax
        import jax.numpy as jnp

        from calibration import wbcp

        F = self.F
        carry20, r, rk, _ = F.refreshed()
        agent, bca = carry20[1], carry20[2]
        bank, (_, q_apply, v_apply) = F.bank, F.built.apply_fns
        s = F.args.polyak_step_size
        target = jax.tree_util.tree_map(lambda new, old: s * new + (1.0 - s) * old,
                                        agent.dual_q.params, agent.dual_q_target.params)
        q = jnp.min(q_apply(target, bank.obs, bank.action), axis=-1)
        y = bank.reward + F.args.gamma * (1.0 - bank.done) * v_apply(agent.value.params, bank.next_obs)
        eta = bca.scale.apply_fn(bca.scale.params, bank.obs, bank.action)
        scale = np.asarray(jnp.maximum(eta, 1e-6) * bca.unit, np.float64)
        direct = np.abs(np.asarray(y - q, np.float64)) / scale
        c = wbcp.calibrate(r.scores, np.random.default_rng(r.stream), alpha=0.1, beta=0.95, draws=1000)
        stream = np.asarray(jax.random.fold_in(rk, 1347375956), np.uint32).ravel()
        ref = r.state.reference
        rec = dict(valid=r.record["valid"], certified=r.record["certified"], threshold=r.record["threshold"],
                   bank_rows=r.record["bank_rows"], scores_bits=_same(r.scores, direct),
                   scores_rel=_rel(r.scores, direct),
                   threshold_reproduced=np.float32(c.threshold).tobytes() == np.asarray(ref.threshold).tobytes(),
                   stream_is_posterior_key=bool(np.array_equal(r.stream, stream)),
                   frozen_scale_is_live=_same(ref.cal_params, bca.scale.params),
                   frozen_unit_is_live=_same(ref.residual_scale, bca.unit),
                   thresholds_filled=bool(np.all(np.asarray(r.state.thresholds) == np.asarray(ref.threshold))),
                   scale_and_unit_kept=_same((r.state.scale, r.state.unit), (bca.scale, bca.unit)),
                   unit=float(np.asarray(bca.unit)))
        _record("refresh_scores", **rec)
        for k in ("valid", "certified", "scores_bits", "threshold_reproduced", "stream_is_posterior_key",
                  "frozen_scale_is_live", "frozen_unit_is_live", "thresholds_filled", "scale_and_unit_kept"):
            self.assertTrue(rec[k], k)
        self.assertTrue(math.isfinite(float(np.asarray(ref.threshold))) and bool(np.asarray(ref.ready)))
        self.assertEqual(r.scores.dtype, np.float64)

    def test_refresh_reads_what_the_next_step_reads(self):
        """Choice 15: band_parts at the next step's batch rows, from the stored state, gives that step's band y and q
        (the target copy its advantage reads), to rounding; the stored target without the averaging step is far off.
        The refresh refuses an unregistered bank."""
        import jax

        F = self.F
        carry20, r, rk, _ = F.refreshed()
        new, out = F.jstep("test")((carry20[0], carry20[1], r.state), None)
        idx = np.asarray(out["batch_indices"])
        rows = jax.tree_util.tree_map(lambda x: x[idx], F.built.dataset)
        y, q = F.IQ.band_parts(F.args, F.built.apply_fns, carry20[1], rows)
        stored = np.asarray(F.built.apply_fns[1](carry20[1].dual_q_target.params, rows.obs, rows.action).min(-1))
        rec = dict(y_rel=_rel(out["bca_y"], y), q_rel=_rel(out["bca_q"], q), y_bits=_same(out["bca_y"], y),
                   q_bits=_same(out["bca_q"], q), control_stored_target_rel=_rel(stored, out["bca_q"]))
        _record("refresh_vintage", **rec)
        self.assertLessEqual(rec["y_rel"], OUTSIDE)
        self.assertLessEqual(rec["q_rel"], OUTSIDE)
        self.assertGreater(rec["control_stored_target_rel"], 100 * OUTSIDE)
        bad = F.bank._replace(reward=F.bank.reward.at[0].add(1.0))
        with self.assertRaises(F.guard.ApprovalError):
            F.IQ.refresh(F.args, F.built.apply_fns, carry20[1], carry20[2], bad, rk, permission=F.guard.MADE_UP)


# ------------------------------------------------------------------------------------------------ shared critic

class SharedCriticTest(unittest.TestCase):
    """On the processor and on the card (plan 3.7 and 5.4: whether IQL's Q and V are bit-identical across its three
    runs)."""

    @classmethod
    def setUpClass(cls):
        cls.F = fixture()

    def test_q_and_v_never_read_the_actor(self):
        """Plan 3.7 / 5.4: Unifloral's Q and V updates never read the actor. Alone (imported step) against the
        restated step with BCA working (a refresh after 20 steps, then 40 steps with the band's weights; standard,
        and IW with per-row thresholds and refused rows). Processor: Q, V and the target critic (parameters, Adam
        states, step counts), the key carry and the per-step Q and V losses are bit for bit the alone run's over all 60
        steps, while the actors differ. Card: the standard and IW runs and the never-refreshed "sham alone" of choice 19
        (one compiled program) share Q, V, the target critic and the key bit for bit; against the alone run (another
        program) the key is bit for bit and Q, V and the target critic agree to (a2)'s card rule (bit for bit or within
        1e-6 relative), recorded."""
        import jax.numpy as jnp

        F = self.F
        a20, la20 = F.scan("alone", (F.built.rng, F.built.agent_state), REFRESH_AT)
        a60, la40 = F.scan("alone", a20, STEPS - REFRESH_AT)
        c20, r, _, ls20 = F.refreshed()
        Rl, Rr = jnp.asarray(F.m.R_levels), jnp.asarray(F.m.R_refused)
        iw_state = r.state._replace(thresholds=jnp.where(jnp.isinf(Rr), jnp.inf, r.state.thresholds * Rl),
                                    most_caution=jnp.asarray(True))
        runs = {"standard": F.scan("std", (c20[0], c20[1], r.state), STEPS - REFRESH_AT),
                "iw": F.scan("std", (c20[0], c20[1], iw_state), STEPS - REFRESH_AT)}
        crit = F.IQ.critic_value_parts
        qv = ("q_loss", "value_loss")
        sham = F.sham_run()
        rec = dict(alone_chunked_equals_alone60=_same(a60, F.alone_run()[0]),
                   sham_vs_standard_critic_value_bits=_same(crit(sham[0][1]), crit(runs["standard"][0][1])),
                   sham_vs_standard_key_bits=_same(sham[0][0], runs["standard"][0][0]),
                   sham_vs_alone_critic_value_rel=_max_rel(crit(sham[0][1]), crit(a60[1])),
                   first20_actor_bits=_same(c20[1].actor, a20[1].actor),
                   first20_actor_rel=_max_rel(c20[1].actor, a20[1].actor),
                   standard_vs_iw_critic_value_bits=_same(crit(runs["standard"][0][1]), crit(runs["iw"][0][1])),
                   standard_vs_iw_key_bits=_same(runs["standard"][0][0], runs["iw"][0][0]),
                   standard_vs_iw_qv_losses_bits=_same({k: runs["standard"][1][k] for k in qv},
                                                       {k: runs["iw"][1][k] for k in qv}))
        for name, (c, l40) in runs.items():
            rec[name] = dict(
                critic_value_bits=_same(crit(c[1]), crit(a60[1])),
                critic_value_rel=_max_rel(crit(c[1]), crit(a60[1])),
                key_bits=_same(c[0], a60[0]),
                q_value_losses_bits=_same((ls20["q_loss"], ls20["value_loss"], l40["q_loss"], l40["value_loss"]),
                                          (la20["q_loss"], la20["value_loss"], la40["q_loss"], la40["value_loss"])),
                actor_differs=not _same(c[1].actor.params, a60[1].actor.params),
                actor_rel=_max_rel(c[1].actor.params, a60[1].actor.params),
                all_valid=bool(np.all(np.asarray(l40["inputs_valid"]))),
                ready=bool(np.all(np.asarray(l40["posterior_ready"]))),
                actor_updated=bool(np.all(np.asarray(l40["actor_updated"]))),
                refused_fraction_mean=float(np.mean(np.asarray(l40["refused_fraction"]))),
                weight_mean_mean=float(np.mean(np.asarray(l40["weight_mean"]))))
        rec["iw_actor_differs_from_standard"] = not _same(runs["iw"][0][1].actor.params,
                                                          runs["standard"][0][1].actor.params)
        _record("shared_critic", **rec)
        for k in ("standard_vs_iw_critic_value_bits", "standard_vs_iw_key_bits", "standard_vs_iw_qv_losses_bits",
                  "iw_actor_differs_from_standard", "sham_vs_standard_critic_value_bits", "sham_vs_standard_key_bits"):
            self.assertTrue(rec[k], k)
        for name in runs:
            for k in ("key_bits", "actor_differs", "all_valid", "ready", "actor_updated"):
                self.assertTrue(rec[name][k], f"{name}: {k}")
            if ON_CARD:
                self.assertLessEqual(rec[name]["critic_value_rel"], CARD_RELATIVE, name)
            else:
                self.assertTrue(rec[name]["critic_value_bits"] and rec[name]["q_value_losses_bits"], name)
        if not ON_CARD:
            self.assertTrue(rec["first20_actor_bits"])
        self.assertGreater(rec["iw"]["refused_fraction_mean"], 0.2)
        self.assertGreater(rec["standard"]["actor_rel"], 100 * CARD_RELATIVE)   # far beyond rounding: BCA acted


# ------------------------------------------------------------------------------------------------ the restated text

@processor_only
class RestatedTextTest(unittest.TestCase):

    def test_restated_lines_are_unifloral_verbatim(self):
        """The restated make_train_step with every line marked "# BCA:" removed is Unifloral's iql.py:194-268, line for
        line, character for character (after the pinned file's SHA-256 check); its def line is Unifloral's line 193
        with the keyword arguments bca and test added."""
        import inspect

        from experiments.shift_hosts import unifloral_iql_bca as IQ

        UI.prepare()
        UI.check_pinned("iql")
        source = (UI.ALGORITHMS / "iql.py").read_text(encoding="utf8").splitlines()
        first, last = IQ.RESTATED_LINES
        unifloral = source[first - 1:last]
        mine, _ = inspect.getsourcelines(IQ.make_train_step)
        mine = [line.rstrip("\n") for line in mine]
        kept = [line for line in mine if "# BCA:" not in line]
        marked = [line for line in mine if "# BCA:" in line]
        _record("restated_text", unifloral_lines=f"{first}-{last}", unifloral_line_count=len(unifloral),
                kept_lines=len(kept), bca_lines=len(marked),
                first_difference=next((i for i, (a, b) in enumerate(zip(kept, unifloral[1:])) if a != b), None))
        self.assertTrue(unifloral[0].startswith("def make_train_step(") and unifloral[-1] == "    return _train_step")
        self.assertTrue(mine[0].startswith(unifloral[0][:-2] + ", *, bca=None, test=None):"))
        self.assertEqual(kept, unifloral[1:])


# ------------------------------------------------------------------------------------------------ validity

@processor_only
class ValidityTest(unittest.TestCase):
    """The per-step validity flag (plan 4.1; INTERFACES.md section 10, step 7) and what an invalid step keeps."""

    @classmethod
    def setUpClass(cls):
        cls.F = fixture()

    def test_invalid_weights_discard_the_step_whole(self):
        """A ready band whose frozen unit is NaN makes the weights' inputs invalid: the flag is false and the whole
        carry is kept, Unifloral's random key included (td3_bc_bca.py:340-342)."""
        import jax.numpy as jnp

        F = self.F
        carry20, r, _, _ = F.refreshed()
        bad_ref = r.state.reference._replace(residual_scale=jnp.asarray(jnp.nan, jnp.float32))
        start = (carry20[0], carry20[1], r.state._replace(reference=bad_ref))
        new, out = F.jstep("std")(start, None)
        rec = dict(inputs_valid=bool(np.asarray(out["inputs_valid"])),
                   dose_inputs_valid=bool(np.asarray(out["dose_inputs_valid"])),
                   scale_inputs_valid=bool(np.asarray(out["scale_inputs_valid"])),
                   carry_kept_whole=_same_state(new, start), actor_updated=bool(np.asarray(out["actor_updated"])))
        _record("validity_weights", **rec)
        self.assertFalse(rec["inputs_valid"] or rec["dose_inputs_valid"] or rec["actor_updated"])
        self.assertTrue(rec["scale_inputs_valid"])
        self.assertTrue(rec["carry_kept_whole"])

    def test_a_failed_fit_is_rejected_and_flagged(self):
        """A NaN running unit makes the band fit invalid: the flag is false and BCA's state (scale network, unit,
        step count) is kept; the learner's own update of that step stands, bit for bit Unifloral's step (module
        docstring, "validity")."""
        import jax.numpy as jnp

        F = self.F
        bca0 = F.start()[2]
        start = (F.built.rng, F.built.agent_state, bca0._replace(unit=jnp.asarray(jnp.nan, jnp.float32)))
        new, out = F.jstep("std")(start, None)
        alone_new, _ = F.jstep("alone")((F.built.rng, F.built.agent_state), None)
        rec = dict(inputs_valid=bool(np.asarray(out["inputs_valid"])),
                   scale_inputs_valid=bool(np.asarray(out["scale_inputs_valid"])),
                   bca_state_kept=_same_state(new[2], start[2]),
                   learner_is_unifloral_step=_same_state(new[:2], alone_new))
        _record("validity_fit", **rec)
        self.assertFalse(rec["inputs_valid"] or rec["scale_inputs_valid"])
        self.assertTrue(rec["bca_state_kept"])
        self.assertTrue(rec["learner_is_unifloral_step"])


# ------------------------------------------------------------------------------------------------ guard

@processor_only
class GuardTest(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.F = fixture()

    def test_bca_entry_points_refuse_unregistered_arrays(self):
        """The factory with BCA present refuses unregistered training rows (one changed value), a token other than
        MADE_UP and another learner's setup; test options need BCA and IQL takes no dose override; refresh() refuses
        an unregistered bank (test_refresh_reads_what_the_next_step_reads); the gate refuses any real-data run (no
        terminal approval exists)."""
        from experiments.shift_hosts import gate

        F = self.F
        a, fns, ds = F.args, F.built.apply_fns, F.built.dataset
        bad = ds._replace(reward=ds.reward.at[0].add(1.0))
        with self.assertRaises(F.guard.ApprovalError):
            F.IQ.make_train_step(a, *fns, bad, bca=F.setup)
        with self.assertRaises(F.guard.ApprovalError):
            F.IQ.make_train_step(a, *fns, ds, bca=F.BC.BCASetup("iql", F.base, "MADE_UP"))
        with self.assertRaises(ValueError):
            F.IQ.make_train_step(a, *fns, ds, bca=F.BC.BCASetup("td3_bc", F.base, F.guard.MADE_UP))
        with self.assertRaises(ValueError):
            F.IQ.make_train_step(a, *fns, ds, test=F.BC.TestOptions())
        with self.assertRaises(ValueError):
            F.IQ.make_train_step(a, *fns, ds, bca=F.setup, test=F.BC.TestOptions(dose_override=F.m.dose))
        with self.assertRaises(gate.ApprovalError):
            gate.require_run_approval("iql stage-0 test: a real-data run must be refused")
        # test options under any permission but the made-up token are refused (as the other three steps refuse them):
        # here the guard is made to pass a stand-in permission, so only the test-options check can refuse
        from unittest import mock

        stand_in = object()
        with mock.patch.object(F.guard, "require_bca", lambda permission, *data: permission):
            with self.assertRaisesRegex(F.guard.ApprovalError, "made-up arrays only"):
                F.IQ.make_train_step(a, *fns, ds, bca=F.BC.BCASetup("iql", F.base, stand_in),
                                     test=F.BC.TestOptions(width_override=F.m.width))
            with self.assertRaisesRegex(F.guard.ApprovalError, "made-up arrays only"):
                F.IQ.make_train_step(a, *fns, ds, bca=F.BC.BCASetup("iql", F.base, stand_in), test=F.BC.TestOptions())
            F.IQ.make_train_step(a, *fns, ds, bca=F.BC.BCASetup("iql", F.base, stand_in))     # no test options: passes
        F.IQ.make_train_step(a, *fns, ds, bca=F.setup)          # the registered made-up rows are accepted
        F.IQ.make_train_step(a, *fns, ds, bca=F.setup, test=F.BC.TestOptions())   # test options with the token


if __name__ == "__main__":
    UI.prepare()
    sys.exit(T.run(sys.modules[__name__], AREA, extra=lambda: dict(unifloral_import=UI.record(), readouts=READOUTS)))

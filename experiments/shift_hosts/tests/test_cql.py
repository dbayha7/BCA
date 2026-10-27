"""Stage-0 tests of the restated CQL step, unifloral_cql_bca.py (plan section 4.5), on made-up arrays only
(tests/madeup.py, opened by the guard): no ThrusterChain data is read, nothing trains on it, and no learned policy's
return is computed. An experiment-local attachment (plan section 3.1) that changes none of the repository's declared
methods.

  (a0) every unmarked line of the restated _train_step is Unifloral's cql.py:195-325, verbatim and in order
  (a1) BCA absent = Unifloral's imported make_train_step: 60 outer steps from Unifloral's own initializer, bit for bit
       (every leaf of the learner state, the optimizer states, the key carry and the per-step losses); processor and card
  (a2) BCA present before any refresh (dose exactly 1), standard and IW: the same comparison. The pre-refresh step
       (dose_active=False, the one a run uses until its first refresh): bit for bit on the processor; on the card bit
       for bit, or every leaf within 1e-6 relative (recorded); larger fails. The dose-active step at dose 1: bit for
       bit on the processor; on the card its difference is recorded (a sham alone's program, choice 19)
  (a3) a fixed made-up dose in [1, 1.5): the critic loss and its gradient equal plan 4.5's external formula to float32
       rounding (relative 1e-5); every dose 1.5 equals the imported step with cql_min_q_weight = 15 after one step
  (a4) with Unifloral's own key at that point, the band's y and q equal Unifloral's target (cql.py:272-275) and the
       minimum over the ten critics of _q_loss_fn's q_pred (cql.py:293), bit for bit; with BCA's own key, y equals a
       replay of CQL's target outside the step, Unifloral's key carry is untouched, and the same step without test
       options is Unifloral's bit for bit (test mode's extra outputs may move the critic by one rounding step)
  (b2) thresholds at 4 levels: batch row b's dose equals the value computed outside the step at R[batch_indices[b]]
       (frozen_level_dose at that row's level: compiled, bit for bit; op by op, within 1e-6, since a compiled program
       may fuse a multiply and an add into one rounding; a shifted lookup is off on most rows)
  (b3) R = +inf on a third of the rows: the standard rule gives dose 1, the IW rule 1.5 (exactly); every other row
       as at its own threshold (compiled: bit for bit; op by op: within 1e-6) and identical under both rules
  refresh: the bank scores from a stored state equal a direct computation; the stream reproduces the threshold
  guards: the port's dose guard makes the step not valid and keeps the old carry whole; the pre-refresh step refuses a
  ready band; the made-up guard refuses other arrays and other permissions; the start-up recompiles are counted
"Bit for bit" and "to float32 rounding" are INTERFACES.md section 3's; a relative difference is, per leaf,
max|a - b| / max|b|. The processor runs every test; the card runs (a0)-(a2) (the others are processor tests in plan
4.5). Measurements go to the result file's extra.details.
  env $(launch env) SHIFT_HOSTS_DEVICE=cpu JAX_PLATFORMS=cpu taskset -c 12-15 /opt/bca-venv/bin/python -m \
      experiments.shift_hosts.tests.test_cql
"""

import dataclasses
import sys
import time
import unittest
from pathlib import Path

import numpy as np

from experiments.shift_hosts import unifloral_import as UI

AREA = "cql"
LEARNER = "cql"
STEPS = 60                  # plan 4.5, (a1) and (a2)
CARD_TOLERANCE = 1e-6       # (a2) on the card
F32_TOLERANCE = 1e-5        # "to float32 rounding" (plan 4.5)
DETAILS = {}
_FIX = {}


def _device():
    from experiments.shift_hosts import tests as T

    return T.device()


def processor_only(fn):
    return unittest.skipUnless(_device() == "cpu", "a processor test (plan section 4.5)")(fn)


# ------------------------------------------------------------------------------------------------ comparisons

def _flat(tree):
    import jax

    leaves, _ = jax.tree_util.tree_flatten_with_path(tree)
    return [(jax.tree_util.keystr(p), np.asarray(v)) for p, v in leaves]


def _same(a, b):
    return compare(a, b)["bit_for_bit"]


def compare(a, b):
    """bit_for_bit: every leaf has the same dtype, shape and bytes, in jax.tree_util order (INTERFACES.md section 3).
    max_rel: the largest per-leaf max|a - b| / max|b| over the differing leaves (inf for a differing integer, key, bool
    or shape). Also the differing leaves' paths."""
    fa, fb = _flat(a), _flat(b)
    if [p for p, _ in fa] != [p for p, _ in fb]:
        return dict(bit_for_bit=False, n_leaves=len(fa), n_differing=-1, differing=["<structure>"],
                    max_rel=float("inf"), worst="<structure>")
    differing, worst, where = [], 0.0, None
    for (p, x), (_, y) in zip(fa, fb):
        if x.dtype == y.dtype and x.shape == y.shape and x.tobytes() == y.tobytes():
            continue
        differing.append(p)
        if x.dtype != y.dtype or x.shape != y.shape or not np.issubdtype(x.dtype, np.floating):
            rel = float("inf")
        else:
            x64, y64 = x.astype(np.float64), y.astype(np.float64)
            if not (np.all(np.isfinite(x64)) and np.all(np.isfinite(y64))):
                rel = 0.0 if np.array_equal(x64, y64, equal_nan=True) else float("inf")
            else:
                diff = float(np.max(np.abs(x64 - y64))) if x64.size else 0.0
                scale = float(np.max(np.abs(y64))) if y64.size else 0.0
                rel = diff / scale if scale > 0 else (0.0 if diff == 0 else float("inf"))
        if where is None or rel >= worst:
            worst, where = rel, p
    return dict(bit_for_bit=not differing, n_leaves=len(fa), n_differing=len(differing), differing=differing[:12],
                max_rel=float(worst), worst=where)


def _rel(a, b):
    a, b = float(a), float(b)
    return abs(a - b) / abs(b) if b != 0 else (0.0 if a == b else float("inf"))


# ------------------------------------------------------------------------------------------------ fixture

def fixture():
    """Made-up arrays (seed 0, registered with the guard), Unifloral's CQL built as its training block builds it (seed
    from bca_core.seeds(0), gamma 0.98, 100,000 updates), the bank as Unifloral's Transition, BCA's fit-key base."""
    if _FIX:
        return _FIX
    UI.prepare()
    from experiments.shift_hosts import bca_core as BC
    from experiments.shift_hosts import guard
    from experiments.shift_hosts import unifloral_cql_bca as CB
    from experiments.shift_hosts.tests import madeup

    m = madeup.build(0)
    s = BC.seeds(0)
    args = UI.args_for(LEARNER, seed=s.unifloral, gamma=0.98, num_updates=100_000)
    built = UI.init_like_training_block(LEARNER, args, m.rows("train"))
    bank = UI.make_transition(LEARNER, m.rows("bank"))
    guard.require_made_up(built.dataset, bank)          # BCA's code may run on exactly these arrays
    _FIX.update(m=m, s=s, args=args, built=built, bank=bank, base=BC.fit_key_base(s.s91), U=UI.load(LEARNER), CB=CB,
                BC=BC, guard=guard)
    DETAILS.setdefault("fixture", dict(train_rows=m.n, bank_rows=int(len(m.bank.obs)), unifloral_seed=s.unifloral,
                                       s91=s.s91, s39=s.s39, batch_size=args.batch_size, gamma=args.gamma,
                                       num_critics=args.num_critics, cql_min_q_weight=args.cql_min_q_weight,
                                       device=_device()))
    return _FIX


def bca_setup(F):
    return F["BC"].BCASetup(LEARNER, F["base"], F["guard"].MADE_UP)


def bca_start(F, most_caution=False):
    """BCA's state before the first refresh (INTERFACES.md section 10, "Setups from these pieces")."""
    BC, built = F["BC"], F["built"]
    scale = BC.make_scale_state(LEARNER, built.obs_mean, built.obs_std, BC.scale_init_key(F["base"], LEARNER))
    return BC.initial_state(scale, F["m"].n, most_caution=most_caution)


def scan(step, carry, n):
    """jit(lax.scan(step, carry, None, n)), as Unifloral scans its step; returns (carry, outputs), the function and
    the seconds (compile included)."""
    import jax

    f = jax.jit(lambda c: jax.lax.scan(step, c, None, n))
    t0 = time.perf_counter()
    out = jax.block_until_ready(f(carry))
    return out, f, time.perf_counter() - t0


def imported60(F):
    """Unifloral's own imported step, 60 outer steps from its own initializer."""
    if "imported60" not in F:
        built, args = F["built"], F["args"]
        step = F["U"].make_train_step(args, *built.apply_fns, built.dataset)
        out, _, sec = scan(step, (built.rng, built.agent_state), STEPS)
        F["imported60"] = out
        DETAILS.setdefault("seconds", {})["imported_60_steps_compile_included"] = round(sec, 2)
    return F["imported60"]


def test_step(F, name, **options):
    """A jitted restated step with BCA present and test options (one compile per kind, reused)."""
    import jax

    key = ("test_step", name)
    if key not in F:
        built = F["built"]
        step = F["CB"].make_train_step(F["args"], *built.apply_fns, built.dataset, bca=bca_setup(F),
                                       test=F["BC"].TestOptions(**options))
        F[key] = jax.jit(step)
    return F[key]


def run1(f, carry):
    import jax

    return jax.block_until_ready(f(carry, None))


def stored(F):
    """A stored state for the refresh: 60 imported steps, then 5 steps of the standard BCA step (before any refresh)."""
    if "stored" not in F:
        c60, _ = imported60(F)
        built = F["built"]
        step = F["CB"].make_train_step(F["args"], *built.apply_fns, built.dataset, bca=bca_setup(F))
        (c, metrics), _, _ = scan(step, (c60[0], c60[1], bca_start(F)), 5)
        if not bool(np.all(np.asarray(metrics["inputs_valid"]))):
            raise AssertionError("the 5 BCA steps before the refresh were not valid")
        F["stored"] = c
    return F["stored"]


def refreshed(F):
    """The stored state refreshed on the made-up bank with the refresh key of outer step 10,000."""
    if "refreshed" not in F:
        rng, agent, bca = stored(F)
        rk = F["BC"].refresh_key(F["s"].s39, 10_000)
        out = F["CB"].refresh(F["args"], F["built"].apply_fns, agent, bca, F["bank"], rk, permission=F["guard"].MADE_UP)
        F["refreshed"] = (rng, agent, bca, out, rk)
        DETAILS["refresh_record"] = out.record
    return F["refreshed"]


def replay(F):
    """Unifloral's draws of one CQL step, written out again outside the step from its key (cql.py:199-203, 219, 242,
    272-273, 282-288), CQL's target with given keys (cql.py:223, 261-275), the minimum over the ten critics, and plan
    4.5's external formula for the dosed critic loss. Each is its own compiled program."""
    if "replay" in F:
        return F["replay"]
    import jax
    import jax.numpy as jnp

    args, built = F["args"], F["built"]
    actor_apply, q_apply, alpha_apply = built.apply_fns
    n, B, gamma = len(built.dataset.obs), args.batch_size, float(args.gamma)

    def keys_of(rng):
        rng, rng_batch = jax.random.split(rng)
        rng, _rng_alpha = jax.random.split(rng)
        rng, _rng_actor = jax.random.split(rng)
        rng, rng_next_v = jax.random.split(rng)
        rng, rng_pi, rng_next = jax.random.split(rng, 3)
        rng, rng_random = jax.random.split(rng)
        return dict(batch=rng_batch, next_v=jax.random.split(rng_next_v, B), pi=rng_pi, next=rng_next,
                    random=rng_random, carry=rng)

    def batch_of(rng):
        bi = jax.random.randint(keys_of(rng)["batch"], (B,), 0, n)
        return bi, jax.tree_util.tree_map(lambda x: x[bi], built.dataset)

    def y_of(keys, rows, actor_params, alpha_params, target_params):
        """r + gamma (1 - d) [min over the ten target critics at a' - alpha sum_j log pi(a'_j | s')], a' ~ pi(. | s')
        with one key per row."""
        alpha = jnp.exp(alpha_apply(alpha_params))

        def one(key, s_next):
            a_next, logp = actor_apply(actor_params, s_next).sample_and_log_prob(seed=key)
            return jnp.min(q_apply(target_params, s_next, a_next)) - alpha * jnp.sum(logp)

        return rows.reward + gamma * (1.0 - rows.done) * jax.vmap(one)(keys, rows.next_obs)

    def q_min(params, obs, action):
        return jnp.min(q_apply(params, obs, action), axis=-1)

    def draws(rng, actor_params):
        k = keys_of(rng)
        _, batch = batch_of(rng)
        pi = actor_apply(actor_params, batch.obs).sample(seed=k["pi"])
        pi_next = actor_apply(actor_params, batch.next_obs).sample(seed=k["next"])
        rand = jax.random.uniform(k["random"], batch.action.shape, minval=-1.0, maxval=1.0)
        return pi, pi_next, rand

    def external(params, batch, y, rand, pi, pi_next, m):
        """Plan 4.5: the squared-error term (cql.py:294-295) + 10 x the mean over rows i and critics k of
        m_i (q_ood_i - Q_k(s_i, a_i)), with q_ood_i = T log sum exp(value / T) over the 40 values of row i: ten critics
        at a random command, at the actor's command at s, at the actor's command at s' scored at s', and at the
        recorded command (cql.py:297-305)."""
        T, W = args.cql_temperature, args.cql_min_q_weight
        q_sa = q_apply(params, batch.obs, batch.action)                                          # [B, 10]
        squared = jnp.mean(jnp.sum(jnp.square(q_sa - y[:, None]), axis=1))
        pool = jnp.concatenate([q_apply(params, batch.obs, rand), q_apply(params, batch.obs, pi),
                                q_apply(params, batch.next_obs, pi_next), q_sa], axis=1)          # [B, 40]
        q_ood = T * jax.nn.logsumexp(pool / T, axis=1)
        return squared + W * jnp.mean(m[:, None] * (q_ood[:, None] - q_sa))

    F["replay"] = dict(keys_of=jax.jit(keys_of), batch_of=jax.jit(batch_of), y_of=jax.jit(y_of),
                       q_min=jax.jit(q_min), draws=jax.jit(draws), external=jax.jit(jax.value_and_grad(external)))
    return F["replay"]


def as_arrays(tree):
    """Every leaf through jnp.asarray (Python step counters become int32 arrays), for comparing a carry with its start."""
    import jax
    import jax.numpy as jnp

    return jax.tree_util.tree_map(jnp.asarray, tree)


# ------------------------------------------------------------------------------------------------ (a0)-(a2)

class CopyTest(unittest.TestCase):
    """(a0)-(a2): the restated step is Unifloral's step, with BCA absent and with BCA present before any refresh."""

    @classmethod
    def setUpClass(cls):
        UI.prepare()
        cls.F = fixture()

    def test_a0_restated_lines_are_unifloral(self):
        """(a0) Every unmarked line of the restated _train_step is Unifloral's cql.py:195-325, verbatim, in order."""
        CB = self.F["CB"]
        ours = Path(CB.__file__).read_text(encoding="utf8").splitlines()
        i0 = next(i for i, line in enumerate(ours) if line.strip() == "def _train_step(runner_state, _):")
        i1 = next(i for i, line in enumerate(ours) if line.strip() == "return _train_step")
        kept = [line for line in ours[i0:i1 + 1] if "# BCA:" not in line and line.strip()]
        UI.check_pinned(LEARNER)
        theirs = (UI.ALGORITHMS / "cql.py").read_text(encoding="utf8").splitlines()
        lo, hi = CB.UNIFLORAL_STEP_LINES
        self.assertEqual(theirs[lo - 1].strip(), "def _train_step(runner_state, _):")
        self.assertEqual(theirs[hi - 1].strip(), "return _train_step")
        ref = [line for line in theirs[lo - 1:hi] if line.strip()]
        self.assertEqual([line.strip() for line in kept], [line.strip() for line in ref])
        indent = lambda line: len(line) - len(line.lstrip())
        shifted = {r.strip(): indent(k) - indent(r) for k, r in zip(kept, ref) if indent(k) != indent(r)}
        self.assertEqual(shifted, {theirs[303].strip(): 4, theirs[309].strip(): 4})
        marked = [line for line in ours[i0:i1 + 1] if "# BCA:" in line]
        DETAILS["a0"] = dict(unifloral_lines=f"cql.py:{lo}-{hi}", unifloral_nonblank=len(ref), kept=len(kept),
                             bca_lines=len(marked), one_level_deeper=sorted(shifted))

    def test_a1_bca_absent_equals_imported(self):
        """(a1) BCA absent: 60 outer steps equal Unifloral's imported step bit for bit (state, Adam, key, losses)."""
        F = self.F
        built = F["built"]
        ci, li = imported60(F)
        step = F["CB"].make_train_step(F["args"], *built.apply_fns, built.dataset)
        (cr, lr), _, sec = scan(step, (built.rng, built.agent_state), STEPS)
        state, losses = compare(cr, ci), compare(lr, li)
        DETAILS["a1"] = dict(device=_device(), steps=STEPS, state=state, losses=losses, seconds=round(sec, 2),
                             loss_keys=sorted(lr))
        self.assertEqual(sorted(lr), sorted(li))
        self.assertTrue(state["bit_for_bit"], state)
        self.assertTrue(losses["bit_for_bit"], losses)
        self.assertFalse(_same(cr[1].vec_q.params, built.agent_state.vec_q.params))     # the critics moved
        self.assertFalse(_same(cr[1].actor.params, built.agent_state.actor.params))     # the actor moved
        for k, v in lr.items():
            self.assertTrue(np.all(np.isfinite(np.asarray(v))), k)

    def test_a2_bca_present_before_refresh_equals_imported(self):
        """(a2) BCA present, no refresh (dose exactly 1), standard and IW: 60 steps equal Unifloral's imported step."""
        F = self.F
        built = F["built"]
        ci, li = imported60(F)
        dev = _device()
        results = DETAILS.setdefault("a2", dict(device=dev, steps=STEPS))
        runs = (("pre_refresh_standard", False, False), ("pre_refresh_iw", True, False),
                ("dose_active_standard", False, True), ("dose_active_iw", True, True))
        for name, most_caution, dose_active in runs:
            step = F["CB"].make_train_step(F["args"], *built.apply_fns, built.dataset, bca=bca_setup(F),
                                           dose_active=dose_active)
            start = bca_start(F, most_caution)
            (cr, lr), _, sec = scan(step, (built.rng, built.agent_state, start), STEPS)
            state = compare(cr[:2], ci)
            losses = compare({k: lr[k] for k in li}, li)
            dose_min, dose_max = np.asarray(lr["dose_min"]), np.asarray(lr["dose_max"])
            bca = cr[2]
            r = results[name] = dict(
                dose_active=dose_active, most_caution=most_caution, state=state, losses=losses, seconds=round(sec, 2),
                bit_for_bit=state["bit_for_bit"] and losses["bit_for_bit"],
                max_rel=max(state["max_rel"], losses["max_rel"]),
                all_valid=bool(np.all(np.asarray(lr["inputs_valid"]))),
                dose_exactly_one=bool(np.all(dose_min == 1.0) and np.all(dose_max == 1.0)),
                posterior_ready_any=bool(np.any(np.asarray(lr["posterior_ready"]))),
                bca_step=int(bca.step), unit_after=float(bca.unit),
                scale_loss_first_last=[float(np.asarray(lr["scale_loss"])[0]),
                                       float(np.asarray(lr["scale_loss"])[-1])])
            r["verdict"] = ("bit for bit" if r["bit_for_bit"] else
                            "rounding-level, within 1e-6 (card exception, choice 19)" if dev == "gpu"
                            and r["max_rel"] <= CARD_TOLERANCE else "beyond 1e-6")
            self.assertTrue(r["all_valid"], name)
            self.assertTrue(r["dose_exactly_one"], name)
            self.assertFalse(r["posterior_ready_any"], name)
            self.assertTrue(np.all(np.asarray(lr["refused_fraction"]) == 0.0), name)
            self.assertTrue(np.all(np.asarray(lr["dose_guard_ok"])), name)
            self.assertEqual(int(bca.step), STEPS)
            self.assertFalse(_same(bca.scale.params, start.scale.params))                  # the band was fitted
            self.assertNotEqual(float(bca.unit), 1.0)
        for name, most_caution, dose_active in runs:
            r = results[name]
            if dev == "cpu" or not dose_active:
                # the processor: both forms bit for bit. The card: the pre-refresh program (the one that runs before
                # any refresh) bit for bit, or every leaf within 1e-6 after 60 steps (plan 4.5); larger fails
                self.assertTrue(r["bit_for_bit"] or (dev == "gpu" and r["max_rel"] <= CARD_TOLERANCE),
                                (name, r["state"], r["losses"]))
        # the card's dose-active form at dose 1 is recorded, not asserted: it is what a never-refreshed sham alone runs
        # (choice 19), whose drift from the alone run is a readout


# ------------------------------------------------------------------------------------------------ (a3), (a4)

class DoseAndBandTest(unittest.TestCase):
    """(a3) the dose's place and strength; (a4) the band's y and q are the learner's own."""

    @classmethod
    def setUpClass(cls):
        UI.prepare()
        cls.F = fixture()

    @processor_only
    def test_a3_dose_scales_each_rows_gap_as_the_external_formula(self):
        """(a3) A fixed made-up dose: the critic loss and gradient equal plan 4.5's formula (relative 1e-5)."""
        import jax.numpy as jnp

        F = self.F
        (rng, agent), _ = imported60(F)
        dose = F["m"].dose
        carry, out = run1(test_step(F, "a3", dose_override=dose), (rng, agent, bca_start(F)))
        self.assertTrue(bool(out["inputs_valid"]) and bool(out["dose_guard_ok"]))
        R = replay(F)
        bi, batch = R["batch_of"](rng)
        self.assertTrue(_same(bi, out["batch_indices"]))
        m = jnp.asarray(dose)[bi]
        self.assertTrue(_same(out["dose"], m))                          # the override, read through batch_indices
        k = R["keys_of"](rng)
        new = carry[1]      # the updated actor, temperature and target critics (the critic update changes none)
        y = R["y_of"](k["next_v"], batch, new.actor.params, new.alpha.params, new.vec_q_target.params)
        pi, pi_next, rand = R["draws"](rng, new.actor.params)
        replayed = dict(target=compare(y, out["unifloral_y"]), pi_actions=compare(pi, out["pi_actions"]),
                        pi_next_actions=compare(pi_next, out["pi_next_actions"]),
                        cql_random_actions=compare(rand, out["cql_random_actions"]))
        for name, r in replayed.items():
            self.assertLessEqual(r["max_rel"], F32_TOLERANCE, (name, r))
        loss, grad = R["external"](agent.vec_q.params, batch, y, rand, pi, pi_next, m)
        loss_rel = _rel(out["dosed_loss"], loss)
        grad_cmp = compare(out["dosed_grad"], grad)
        loss1, grad1 = R["external"](agent.vec_q.params, batch, y, rand, pi, pi_next, jnp.ones_like(m))
        control = dict(loss_rel=_rel(out["dosed_loss"], loss1), grad_max_rel=compare(out["dosed_grad"], grad1)["max_rel"])
        DETAILS["a3"] = dict(loss_rel=loss_rel, grad=grad_cmp, loss=float(out["dosed_loss"]), external=float(loss),
                             replayed_inputs=replayed, dose_range=[float(m.min()), float(m.max())],
                             control_with_dose_one=control)
        self.assertLessEqual(loss_rel, F32_TOLERANCE)
        self.assertLessEqual(grad_cmp["max_rel"], F32_TOLERANCE, grad_cmp)
        self.assertGreater(control["loss_rel"], 1e-3)                    # the check can tell a dose of 1 apart
        self.assertGreater(control["grad_max_rel"], 1e-3)

    @processor_only
    def test_a3_every_dose_one_and_a_half_equals_weight_fifteen(self):
        """(a3) Every dose 1.5 = the imported step with cql_min_q_weight = 15, after one step, to float32 rounding."""
        import jax

        F = self.F
        built = F["built"]
        (rng, agent), _ = imported60(F)
        carry_b, out_b = run1(test_step(F, "a3_15", dose_override=np.full(F["m"].n, 1.5, np.float32)),
                              (rng, agent, bca_start(F)))
        args15 = dataclasses.replace(F["args"], cql_min_q_weight=15.0)
        imported = jax.jit(F["U"].make_train_step(args15, *built.apply_fns, built.dataset))
        carry_u, loss_u = run1(imported, (rng, agent))
        parts = dict(key=compare(carry_b[0], carry_u[0]), actor=compare(carry_b[1].actor, carry_u[1].actor),
                     alpha=compare(carry_b[1].alpha, carry_u[1].alpha),
                     vec_q_target=compare(carry_b[1].vec_q_target, carry_u[1].vec_q_target),
                     vec_q=compare(carry_b[1].vec_q, carry_u[1].vec_q),
                     critic_loss_rel=_rel(out_b["critic_loss"], loss_u["critic_loss"]),
                     other_losses=compare({k: out_b[k] for k in loss_u if k != "critic_loss"},
                                          {k: v for k, v in loss_u.items() if k != "critic_loss"}))
        DETAILS["a3_weight_15"] = parts
        self.assertTrue(bool(out_b["inputs_valid"]))
        self.assertTrue(parts["key"]["bit_for_bit"])
        for name in ("actor", "alpha", "vec_q_target", "vec_q", "other_losses"):
            self.assertLessEqual(parts[name]["max_rel"], F32_TOLERANCE, (name, parts[name]))
        self.assertLessEqual(parts["critic_loss_rel"], F32_TOLERANCE)
        self.assertFalse(_same(carry_b[1].vec_q.params, agent.vec_q.params))

    @processor_only
    def test_a4_with_unifloral_key_y_and_q_are_unifloral_own(self):
        """(a4) With Unifloral's key: y = Unifloral's target, q = min over ten of _q_loss_fn's q_pred, bit for bit."""
        F = self.F
        (rng, agent), _ = imported60(F)
        _, out = run1(test_step(F, "a4", unifloral_keys=True), (rng, agent, bca_start(F)))
        y, q = compare(out["bca_y"], out["unifloral_y"]), compare(out["bca_q"], out["unifloral_q"])
        DETAILS["a4_unifloral_key"] = dict(y=y, q=q, y_dtype=str(np.asarray(out["bca_y"]).dtype),
                                           shape=list(np.shape(out["bca_y"])))
        self.assertEqual(np.shape(out["bca_y"]), (F["args"].batch_size,))
        self.assertTrue(y["bit_for_bit"], y)
        self.assertTrue(q["bit_for_bit"], q)
        self.assertTrue(bool(out["inputs_valid"]))

    @processor_only
    def test_a4_with_bca_key_y_is_the_learners_target_and_unifloral_key_untouched(self):
        """(a4) With BCA's own key: y = a replay of CQL's target, q = min of the ten critics; Unifloral's key unmoved."""
        import jax

        F = self.F
        BC = F["BC"]
        built = F["built"]
        (rng, agent), _ = imported60(F)
        carry, out = run1(test_step(F, "plain"), (rng, agent, bca_start(F)))
        R = replay(F)
        _, batch = R["batch_of"](rng)
        keys = jax.random.split(jax.random.fold_in(jax.random.fold_in(F["base"], 1), BC.FOLDS[LEARNER].fit_noise),
                                F["args"].batch_size)              # fit_key(PRNGKey(s91), t = 1), then 1179210836
        new = carry[1]
        y = R["y_of"](keys, batch, new.actor.params, new.alpha.params, new.vec_q_target.params)
        q = R["q_min"](agent.vec_q.params, batch.obs, batch.action)
        imported = jax.jit(F["U"].make_train_step(F["args"], *built.apply_fns, built.dataset))
        carry_u, loss_u = run1(imported, (rng, agent))
        normal = jax.jit(F["CB"].make_train_step(F["args"], *built.apply_fns, built.dataset, bca=bca_setup(F)))
        carry_n, loss_n = run1(normal, (rng, agent, bca_start(F)))
        res = dict(y=compare(out["bca_y"], y), q=compare(out["bca_q"], q),
                   own_draw_differs_from_unifloral=not _same(out["bca_y"], out["unifloral_y"]),
                   key_carry=compare(carry[0], carry_u[0]),
                   test_mode_learner_state=compare(carry[1], carry_u[1]),
                   test_mode_losses=compare({k: out[k] for k in loss_u}, loss_u),
                   normal_mode_key_and_learner_state=compare(carry_n[:2], carry_u),
                   normal_mode_losses=compare({k: loss_n[k] for k in loss_u}, loss_u),
                   note="test mode's extra outputs (q_pred through has_aux, the gradient, the draws) change how XLA "
                        "fuses the critic update, so its critic may differ from Unifloral's by rounding; the same step "
                        "without test options is Unifloral's bit for bit (here for one step, and (a2) for 60)")
        DETAILS["a4_bca_key"] = res
        self.assertLessEqual(res["y"]["max_rel"], F32_TOLERANCE, res["y"])
        self.assertLessEqual(res["q"]["max_rel"], F32_TOLERANCE, res["q"])
        self.assertTrue(res["own_draw_differs_from_unifloral"])
        self.assertTrue(res["key_carry"]["bit_for_bit"])
        self.assertLessEqual(res["test_mode_learner_state"]["max_rel"], F32_TOLERANCE, res["test_mode_learner_state"])
        self.assertTrue(res["test_mode_losses"]["bit_for_bit"], res["test_mode_losses"])
        self.assertTrue(res["normal_mode_key_and_learner_state"]["bit_for_bit"],
                        res["normal_mode_key_and_learner_state"])
        self.assertTrue(res["normal_mode_losses"]["bit_for_bit"], res["normal_mode_losses"])


# ------------------------------------------------------------------------------------------------ (b2), (b3), refresh

class ThresholdAndRefreshTest(unittest.TestCase):
    """(b2) each row's own threshold; (b3) refused rows; the refresh's bank scores."""

    @classmethod
    def setUpClass(cls):
        UI.prepare()
        cls.F = fixture()

    def _expected(self, reference, eta, levels, compiled):
        """Per batch row, calibration.dose.frozen_level_dose (the ports' dose, one threshold) at that row's own
        threshold, op by op or compiled. Inside a compiled program XLA may fuse a multiply and an add into one rounding
        (1 + blend x ratio), so op by op and compiled can differ by one float32 step; both are recorded."""
        import jax
        import jax.numpy as jnp

        from calibration.dose import frozen_level_dose

        if "fld" not in self.F:
            self.F["fld"] = jax.jit(lambda ref, e, level: frozen_level_dose(ref._replace(threshold=level), e, 0.5))
        dose, width = np.full(eta.shape, np.nan, np.float32), np.full(eta.shape, np.nan, np.float32)
        for level in np.unique(levels):
            L = jnp.asarray(level, jnp.float32)
            d = (self.F["fld"](reference, eta, L) if compiled
                 else frozen_level_dose(reference._replace(threshold=L), eta, 0.5))
            rows = levels == level
            dose[rows], width[rows] = np.asarray(d.dose)[rows], np.asarray(d.width)[rows]
        return dose, width

    @processor_only
    def test_b2_each_batch_rows_dose_reads_its_own_threshold(self):
        """(b2) Thresholds at 4 levels: batch row b's dose equals the outside value at R[batch_indices[b]]."""
        import jax
        import jax.numpy as jnp

        F = self.F
        m = F["m"]
        rng, agent, _, out_r, _ = refreshed(F)
        st = out_r.state
        self.assertTrue(out_r.record["valid"] and out_r.record["certified"])
        self.assertTrue(np.isfinite(float(st.reference.threshold)))
        R = replay(F)
        _, batch = R["batch_of"](rng)
        eta_outside = jax.jit(st.scale.apply_fn)(st.reference.cal_params, batch.obs, batch.action)
        res = DETAILS.setdefault("b2", {})
        for name, most_caution in (("standard_rule", False), ("iw_rule", True)):
            bca = st._replace(thresholds=jnp.asarray(m.R_levels), most_caution=jnp.asarray(most_caution))
            _, out = run1(test_step(F, "plain"), (rng, agent, bca))
            bi = np.asarray(out["batch_indices"])
            levels = m.R_levels[bi]
            got = np.asarray(out["dose"])
            r = res[name] = dict(levels_in_batch=sorted(float(v) for v in np.unique(levels)),
                                 dose_range=[float(got.min()), float(got.max())],
                                 eta_vs_outside=compare(out["eta"], eta_outside))
            for mode, compiled in (("compiled", True), ("op_by_op", False)):
                dose, width = self._expected(st.reference, out["eta"], levels, compiled)
                shifted, _ = self._expected(st.reference, out["eta"], m.R_levels[(bi + 1) % m.n], compiled)
                r[mode] = dict(dose=compare(got, dose), width=compare(out["width"], width),
                               misaligned_rows_off_by_more_than_1e_4=float(np.mean(np.abs(shifted - got) > 1e-4)))
            self.assertTrue(bool(out["inputs_valid"]) and bool(out["posterior_ready"]))
            self.assertEqual(float(out["refused_fraction"]), 0.0)
            self.assertEqual(len(r["levels_in_batch"]), 4)
            self.assertTrue(r["compiled"]["dose"]["bit_for_bit"], r["compiled"]["dose"])
            for mode in ("compiled", "op_by_op"):
                self.assertLessEqual(r[mode]["dose"]["max_rel"], CARD_TOLERANCE, (mode, r[mode]["dose"]))
                self.assertTrue(r[mode]["width"]["bit_for_bit"], (mode, r[mode]["width"]))
                self.assertGreater(r[mode]["misaligned_rows_off_by_more_than_1e_4"], 0.5)   # misalignment would show
            self.assertLessEqual(r["eta_vs_outside"]["max_rel"], F32_TOLERANCE)
            self.assertTrue(np.all((got > 1.0) & (got < 1.5)))

    @processor_only
    def test_b3_refused_rows_standard_and_iw_rules(self):
        """(b3) R = +inf on every third row: standard dose 1, IW dose 1.5; every other row as at its own threshold."""
        import jax.numpy as jnp

        F = self.F
        m = F["m"]
        rng, agent, _, out_r, _ = refreshed(F)
        st = out_r.state
        doses, res = {}, DETAILS.setdefault("b3", {})
        for name, most_caution, refused_dose in (("standard_rule", False, 1.0), ("iw_rule", True, 1.5)):
            bca = st._replace(thresholds=jnp.asarray(m.R_refused), most_caution=jnp.asarray(most_caution))
            _, out = run1(test_step(F, "plain"), (rng, agent, bca))
            bi = np.asarray(out["batch_indices"])
            refused = bi % 3 == 0
            got = np.asarray(out["dose"])
            r = res[name] = dict(
                refused_rows=int(refused.sum()),
                refused_flag_matches=bool(np.array_equal(np.asarray(out["refused"]), refused)),
                refused_fraction=float(out["refused_fraction"]), support_fraction=float(out["support_fraction"]),
                refused_doses=sorted(set(float(v) for v in got[refused])), valid=bool(out["inputs_valid"]))
            for mode, compiled in (("compiled", True), ("op_by_op", False)):
                expected, _ = self._expected(st.reference, out["eta"], np.ones(bi.shape, np.float32), compiled)
                r["others_" + mode] = compare(got[~refused], expected[~refused])
            self.assertGreater(r["refused_rows"], 0)
            self.assertTrue(r["refused_flag_matches"])
            self.assertEqual(r["refused_fraction"], float(np.float32(refused.mean())))
            self.assertTrue(np.all(got[refused] == np.float32(refused_dose)), r["refused_doses"])
            self.assertTrue(r["others_compiled"]["bit_for_bit"], r["others_compiled"])
            for mode in ("compiled", "op_by_op"):
                self.assertLessEqual(r["others_" + mode]["max_rel"], CARD_TOLERANCE, (mode, r["others_" + mode]))
            self.assertEqual(r["support_fraction"], float(np.float32((~refused).mean())))
            self.assertTrue(r["valid"])
            doses[name] = got
        same_others = bool(np.array_equal(doses["standard_rule"][~refused], doses["iw_rule"][~refused]))
        res["other_rows_identical_under_both_rules"] = same_others
        self.assertTrue(same_others)

    @processor_only
    def test_refresh_bank_scores_equal_direct_computation(self):
        """Refresh: the bank scores from a stored state equal a direct computation; stream and threshold reproduce."""
        import jax
        import jax.numpy as jnp

        from calibration import wbcp

        F = self.F
        bank = F["bank"]
        _, agent, bca, out_r, rk = refreshed(F)
        n = int(bank.obs.shape[0])
        keys = jax.random.split(jax.random.fold_in(rk, 1213156420), n)          # algorithms/cql_bca.py:244
        R = replay(F)
        y = R["y_of"](keys, bank, agent.actor.params, agent.alpha.params, agent.vec_q_target.params)
        q = R["q_min"](agent.vec_q.params, bank.obs, bank.action)
        eta = bca.scale.apply_fn(bca.scale.params, bank.obs, bank.action)        # the live scale network, op by op
        scale = np.asarray(jnp.maximum(eta, 1e-06) * bca.unit, np.float64)
        scores = np.abs(np.asarray(y - q, np.float64)) / scale
        stream = np.asarray(jax.random.fold_in(rk, 1347375956), np.uint32).ravel()   # cql_bca.py:259
        c = wbcp.calibrate(out_r.scores, np.random.default_rng(out_r.stream), alpha=0.1, beta=0.95, draws=1000)
        ref = out_r.state.reference
        reproduced = all(np.float32(a).tobytes() == np.asarray(b, np.float32).tobytes() for a, b in (
            (c.threshold, ref.threshold), (c.lambda_hat, ref.lambda_hat), (c.lambda_hpd, ref.lambda_hpd),
            (c.n_eff, ref.n_eff)))
        res = dict(scores=compare(out_r.scores, scores), stream_equal=bool(np.array_equal(out_r.stream, stream)),
                   threshold_reproduced=reproduced, record=out_r.record,
                   frozen_scale_is_live=_same(ref.cal_params, bca.scale.params),
                   frozen_unit_is_live=bool(np.float32(ref.residual_scale) == np.float32(bca.unit)),
                   thresholds_filled=bool(np.all(np.asarray(out_r.state.thresholds) == np.float32(ref.threshold))))
        DETAILS["refresh"] = res
        self.assertEqual(out_r.scores.dtype, np.float64)
        self.assertTrue(res["scores"]["bit_for_bit"], res["scores"])
        self.assertTrue(res["stream_equal"])
        self.assertTrue(res["threshold_reproduced"])
        self.assertTrue(out_r.record["valid"] and out_r.record["certified"])
        self.assertEqual(out_r.record["bank_rows"], n)
        self.assertTrue(res["frozen_scale_is_live"] and res["frozen_unit_is_live"] and res["thresholds_filled"])
        self.assertTrue(bool(ref.ready))


# ------------------------------------------------------------------------------------------------ guards, compiles

class GuardTest(unittest.TestCase):
    """The port's dose guard and the made-up guard at every BCA entry point."""

    @classmethod
    def setUpClass(cls):
        UI.prepare()
        cls.F = fixture()

    @processor_only
    def test_negative_or_nonfinite_dose_makes_the_step_not_valid(self):
        """The port's guard (algorithms/cql.py:304-306): a negative or non-finite dose keeps the old carry whole."""
        F = self.F
        (rng, agent), _ = imported60(F)
        bi0 = int(np.asarray(replay(F)["batch_of"](rng)[0])[0])         # a row this step draws
        one_bad = F["m"].dose.copy()
        one_bad[bi0] = -0.25
        res = DETAILS.setdefault("dose_guard", {})
        for name, bad in (("all_negative", np.full(F["m"].n, -1.0, np.float32)),
                          ("all_nan", np.full(F["m"].n, np.nan, np.float32)),
                          ("one_negative_row_in_batch", one_bad)):
            start = as_arrays((rng, agent, bca_start(F)))
            carry, out = run1(test_step(F, "guard_" + name, dose_override=bad), start)
            res[name] = dict(valid=bool(out["inputs_valid"]), guard_ok=bool(out["dose_guard_ok"]),
                             carry_kept_whole=_same(carry, start),
                             critic_loss_finite=bool(np.isfinite(float(out["critic_loss"]))))
            self.assertFalse(res[name]["valid"], name)
            self.assertFalse(res[name]["guard_ok"], name)
            self.assertTrue(res[name]["carry_kept_whole"], name)

    @processor_only
    def test_made_up_guard_at_every_bca_entry(self):
        """The factory and the refresh refuse unregistered arrays and anything but the made-up token or an approval."""
        import jax.numpy as jnp

        F = self.F
        BC, CB, guard, built = F["BC"], F["CB"], F["guard"], F["built"]
        args, fns = F["args"], built.apply_fns
        changed = built.dataset._replace(reward=built.dataset.reward.at[0].add(1.0))
        with self.assertRaises(guard.ApprovalError):
            CB.make_train_step(args, *fns, changed, bca=bca_setup(F))
        with self.assertRaises(guard.ApprovalError):
            CB.make_train_step(args, *fns, built.dataset, bca=BC.BCASetup(LEARNER, F["base"], "made up"))
        with self.assertRaises(ValueError):
            CB.make_train_step(args, *fns, built.dataset, bca=BC.BCASetup("td3_bc", F["base"], guard.MADE_UP))
        with self.assertRaises(ValueError):
            CB.make_train_step(args, *fns, built.dataset, test=BC.TestOptions())
        with self.assertRaises(ValueError):
            CB.make_train_step(args, *fns, built.dataset, dose_active=False)
        with self.assertRaises(ValueError):
            CB.make_train_step(args, *fns, built.dataset, bca=bca_setup(F), dose_active=False,
                               test=BC.TestOptions(dose_override=F["m"].dose))
        _, agent, bca = stored(F)
        rk = BC.refresh_key(F["s"].s39, 10_000)
        bank_changed = F["bank"]._replace(obs=F["bank"].obs.at[0, 0].add(1.0))
        with self.assertRaises(guard.ApprovalError):
            CB.refresh(args, fns, agent, bca, bank_changed, rk, permission=guard.MADE_UP)
        with self.assertRaises(guard.ApprovalError):
            CB.refresh(args, fns, agent, bca, F["bank"], rk, permission=None)
        subset = F["bank"]._replace(**{k: v[:100] for k, v in F["bank"]._asdict().items()})
        with self.assertRaises(guard.ApprovalError):
            CB.refresh(args, fns, agent, bca, subset, rk, permission=guard.MADE_UP)
        self.assertTrue(np.all(np.asarray(jnp.isfinite(F["bank"].obs))))
        # the step's trace-time check of the per-row thresholds (one float32 per training row), as ReBRAC's
        import jax

        rng0 = built.rng
        for program in (dict(dose_active=False), dict()):
            step = CB.make_train_step(args, *fns, built.dataset, bca=bca_setup(F), **program)
            for wrong in (bca.thresholds[:-1], jnp.concatenate([bca.thresholds, bca.thresholds[:1]]),
                          bca.thresholds.astype(jnp.float16)):
                with self.assertRaisesRegex(ValueError, "one threshold per"):
                    jax.lax.scan(step, (rng0, agent, bca._replace(thresholds=wrong)), None, length=1)

    @processor_only
    def test_pre_refresh_step_refuses_a_ready_band(self):
        """dose_active=False is the step before the first refresh: given a ready band it is not valid (old carry kept)."""
        import jax

        F = self.F
        built = F["built"]
        rng, agent, _, out_r, _ = refreshed(F)
        start = as_arrays((rng, agent, out_r.state))
        pre = jax.jit(F["CB"].make_train_step(F["args"], *built.apply_fns, built.dataset, bca=bca_setup(F),
                                              dose_active=False))
        carry, out = run1(pre, start)
        post = jax.jit(F["CB"].make_train_step(F["args"], *built.apply_fns, built.dataset, bca=bca_setup(F)))
        _, out_post = run1(post, start)
        res = DETAILS.setdefault("pre_refresh_step_with_ready_band", {})
        res.update(posterior_ready=bool(out["posterior_ready"]), valid=bool(out["inputs_valid"]),
                   carry_kept_whole=_same(carry, start), dose_active_step_valid=bool(out_post["inputs_valid"]))
        self.assertTrue(res["posterior_ready"])
        self.assertFalse(res["valid"])
        self.assertTrue(res["carry_kept_whole"])
        self.assertTrue(res["dose_active_step_valid"])

    @processor_only
    def test_startup_recompiles_counted(self):
        """The first chunk calls compile again as the carry's types settle (environment report 3.3, item 4): counted."""
        import jax

        F = self.F
        built = F["built"]
        counts, carries = {}, {}
        for name, step, carry in (
                ("imported", F["U"].make_train_step(F["args"], *built.apply_fns, built.dataset),
                 (built.rng, built.agent_state)),
                ("bca_standard", F["CB"].make_train_step(F["args"], *built.apply_fns, built.dataset, bca=bca_setup(F),
                                                         dose_active=False),
                 (built.rng, built.agent_state, bca_start(F)))):
            f = jax.jit(lambda c, step=step: jax.lax.scan(step, c, None, 5))
            sizes, seconds = [], []
            for _ in range(4):
                t0 = time.perf_counter()
                carry, out = jax.block_until_ready(f(carry))
                seconds.append(round(time.perf_counter() - t0, 2))
                sizes.append(int(f._cache_size()))
                if "inputs_valid" in out:
                    self.assertTrue(bool(np.all(np.asarray(out["inputs_valid"]))))
            counts[name] = dict(cache_size_after_each_call=sizes, seconds_per_call=seconds)
            carries[name] = carry
        same = compare(carries["bca_standard"][:2], carries["imported"])
        counts["bca_learner_state_equals_imported_after_4_chunks_of_5"] = same
        DETAILS["startup_recompiles"] = counts
        self.assertTrue(same["bit_for_bit"], same)          # the recompiles change no value
        self.assertLessEqual(counts["bca_standard"]["cache_size_after_each_call"][-1], 4)


def _extra():
    return dict(import_record=UI.record(), details=DETAILS)


if __name__ == "__main__":
    UI.prepare()
    from experiments.shift_hosts import tests as T

    sys.exit(T.run(sys.modules[__name__], AREA, extra=_extra))
